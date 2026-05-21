import torch
import torch.nn.functional as F


def build_position_validity_mask(vocab_size, token_pos, tokenizer, device, enabled):
    """Return (vocab_size,) bool mask: True = invalid token at this decoding position.

    Empty mask if `enabled` is False. Otherwise: at each box position only that
    position's valid codes (+ BOS/EOS) are allowed; at the cls position only
    cls_codes (+ BOS/EOS) are allowed.
    """
    token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if not enabled:
        return token_mask

    if not hasattr(tokenizer, "box_cols"):
        raise NotImplementedError(
            "Box token limiting is not supported without box_cols"
        )

    cls_tok_id = tokenizer._cls_pos
    box_cols = tokenizer.box_cols

    pos_to_codes = {
        box_cols[0]: tokenizer.x_codes,
        box_cols[1]: tokenizer.y_codes,
        box_cols[2]: tokenizer.z_codes,
        box_cols[3]: tokenizer.dx_codes,
        box_cols[4]: tokenizer.dy_codes,
        box_cols[5]: tokenizer.dz_codes,
        box_cols[6]: tokenizer.heading_codes,
        cls_tok_id: tokenizer.cls_codes,
    }
    if tokenizer.predict_velocity:
        pos_to_codes[box_cols[7]] = tokenizer.vx_codes
        pos_to_codes[box_cols[8]] = tokenizer.vy_codes

    token_mask[:] = True
    token_mask[tokenizer.BOS_code] = False
    token_mask[tokenizer.EOS_code] = False
    if token_pos in pos_to_codes:
        codes = torch.as_tensor(
            pos_to_codes[token_pos], dtype=torch.long, device=device
        )
        token_mask[codes] = False
    return token_mask


# From: trl/trainer/utils.py
def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
    where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
    along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
    leading dimensions.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
            are preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`):
            Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
            the cost of more iterations.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)
