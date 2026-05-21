import torch
import torch.nn as nn

from transformers import GPT2Config, GPT2Model
from transformers.generation import BeamSearchScorer
from transformers.utils import logging

from ..model_utils.sequence_dense_head_utils import (
    selective_log_softmax,
    entropy_from_logits,
    build_position_validity_mask,
)

logger = logging.get_logger(__name__)


class AutoReg3DHead(nn.Module):
    def __init__(
        self,
        model_cfg,
        input_channels,
        num_class,
        class_names,
        grid_size,
        point_cloud_range,
        **kwargs,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.dim = input_channels

        feature_grid_resolution = model_cfg.get("FEATURE_GRID_RESOLUTION", [2, 2])

        self.eos_skip_num = model_cfg.get("EOS_SKIP_NUM", 0)
        self.inner_dim_factor = model_cfg.get("INNER_DIM_FACTOR", 2)
        self.activation = model_cfg.get("ACTIVATION", "relu")
        self.decode_mode = model_cfg.get("DECODE_MODE", "greedy")
        self.limit_box_token_to_valid = model_cfg.get("LIMIT_BOX_TOKEN_TO_VALID", False)

        # Autocast settings
        self.autocast_eval = model_cfg.get("AUTOCAST_EVAL", False)
        self.autocast_dtype = model_cfg.get("AUTOCAST_DTYPE", "bf16")
        self._use_amp = self.autocast_eval and torch.cuda.is_available()
        _use_bf16 = (
            str(self.autocast_dtype).lower() == "bf16"
            and torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        self._amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16


        # Decoder
        self.pad_token_id = model_cfg.PAD_TOKEN_ID
        decoder_config = GPT2Config(
            vocab_size=model_cfg.VOCAB_SIZE,
            n_positions=self.model_cfg.MAX_SEQ_LENGTH,
            n_embd=self.dim,
            n_layer=self.model_cfg.NUM_LAYERS,
            n_head=self.model_cfg.NUM_HEADS,
            n_inner=self.dim * self.inner_dim_factor,
            activation_function=self.activation,
            embd_pdrop=model_cfg.get("EMBD_PDROP", 0.05),
            add_cross_attention=True,
            pad_token_id=self.pad_token_id,
        )
        self.decoder = GPT2Model(config=decoder_config)
        # Replace wte to honor padding_idx; HF's GPT2Model ignores pad_token_id when building the embedding.
        self.decoder.wte = nn.Embedding(
            decoder_config.vocab_size,
            decoder_config.hidden_size,
            padding_idx=decoder_config.pad_token_id,
        )

        self.output = nn.Linear(self.dim, model_cfg.VOCAB_SIZE)

        self.feat_H = grid_size[0] // feature_grid_resolution[0]
        self.feat_W = grid_size[1] // feature_grid_resolution[1]

        self.encoder_xpos_embed = nn.Parameter(
            torch.randn(1, self.feat_H, self.dim) * 0.02
        )
        self.encoder_ypos_embed = nn.Parameter(
            torch.randn(1, self.feat_W, self.dim) * 0.02
        )
        self.encoder_pos_drop = nn.Dropout(p=0.05)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.trunc_normal_(self.encoder_xpos_embed, std=0.02)
        nn.init.trunc_normal_(self.encoder_ypos_embed, std=0.02)
        self.decoder._init_weights(self.decoder.wte)
        nn.init.trunc_normal_(self.decoder.wpe.weight, std=0.02)

    def forward(
        self, encoder_out, tgt, use_cache=False, return_attentions: bool = False
    ):
        tgt_padding_mask = self._pad_mask(tgt)  # (B, L)

        encoder_out = self._prepare_encoder_features(encoder_out)

        decoder_return = self.decoder(
            input_ids=tgt,
            encoder_hidden_states=encoder_out,
            attention_mask=tgt_padding_mask,
            use_cache=use_cache,
            output_attentions=return_attentions,
        )
        hidden_state = decoder_return.last_hidden_state  # (B, L, D)

        out = self.output(hidden_state)

        if return_attentions and not use_cache:
            return out, decoder_return.attentions, decoder_return.cross_attentions
        return out

    def forward_train(self, batch_dict):

        encoder_out = batch_dict["spatial_features_2d"]  # (B, C, H, W)
        x = encoder_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        y_input = batch_dict["gt_box_seqs_inputs"].long()  # gt_box_seqs[:, :-1]
        y_expected = batch_dict["gt_box_seqs_expected"].long()  # gt_box_seqs[:, 1:]

        preds = self.forward(x, y_input)

        loss = self.criterion(
            preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1)
        )

        tb_dict = {"ar_loss_nll": loss.item()}

        return loss, tb_dict

    def predict(
        self,
        encoder_out,
        tgt,
        past_key_values=None,
        use_cache=True,
        full_attention_mask=None,
        encoder_features=None,
    ):
        # 1.0 for position we want to attend and 0.0 for masked positions
        if full_attention_mask is not None:
            tgt_padding_mask = full_attention_mask  # (B, L)
        else:
            tgt_padding_mask = self._pad_mask(tgt)  # (B, L_cur)

        features = (
            encoder_features
            if encoder_features is not None
            else self._prepare_encoder_features(encoder_out)
        )

        decoder_return = self.decoder(
            input_ids=tgt,
            encoder_hidden_states=features,
            attention_mask=tgt_padding_mask,
            use_cache=use_cache,
            past_key_values=past_key_values if use_cache else None,
        )

        hidden_state = decoder_return.last_hidden_state  # (B, L, D)

        out = self.output(hidden_state)
        if use_cache:
            return out[:, -1, :], decoder_return.past_key_values

        return out[:, -1, :]

    def generate(self, batch_dict, tokenizer, init_preds=None, temperature=1.0):
        encoder_out = batch_dict["spatial_features_2d"]  # (B, C, H, W)
        encoder_out = encoder_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        encoder_features = self._prepare_encoder_features(encoder_out)

        if self.decode_mode == "beam":
            return self._generate_beam(
                batch_dict,
                tokenizer,
                encoder_features,
                init_preds=init_preds,
                num_beams=int(self.model_cfg.get("NUM_BEAMS", 1)),
                length_penalty=self.model_cfg.get("BEAM_LENGTH_PENALTY", 1.0),
                early_stopping=self.model_cfg.get("BEAM_EARLY_STOP", True),
                temperature=temperature,
            )

        return self._generate_sample(
            tokenizer,
            encoder_out,
            encoder_features,
            init_preds=init_preds,
            temperature=temperature,
        )

    def _prepare_encoder_features(self, encoder_out):
        encoder_pos_bev = (
            self.encoder_xpos_embed[:, :, None, :]
            + self.encoder_ypos_embed[:, None, :, :]
        )
        encoder_pos_bev = encoder_pos_bev.view(1, -1, self.dim)
        encoder_pos_bev = encoder_pos_bev.to(
            device=encoder_out.device, dtype=encoder_out.dtype
        )
        encoder_out = self.encoder_pos_drop(encoder_out + encoder_pos_bev)
        return encoder_out.contiguous()

    def _pad_mask(self, tgt):
        pad_mask = (tgt != self.pad_token_id).float()  # (B, L)
        return pad_mask

    def _generate_sample(
        self, tokenizer, encoder_out, encoder_features, init_preds=None, temperature=1.0
    ):
        """Per-step decoding for decode_mode in {'greedy', 'sample'}.

        The first loop iteration feeds the full prefix (init_preds or BOS) to
        predict() with past_key_values=None to prime the KV cache; subsequent
        iterations feed one token at a time against the growing cache.
        """
        if self.decode_mode == "greedy":
            sample_fn = lambda logits: logits.argmax(dim=-1, keepdim=True)
        elif self.decode_mode == "sample":
            sample_fn = (
                lambda logits: torch.distributions.Categorical(logits=logits)
                .sample()
                .unsqueeze(-1)
            )
        else:
            raise ValueError(
                f"_generate_sample reached with decode_mode={self.decode_mode!r}; "
                f"expected 'greedy' or 'sample'."
            )

        BOS_code = tokenizer.BOS_code
        tok_per_seq = tokenizer.tok_per_seq
        device = encoder_out.device
        B = encoder_out.size(0)

        # Initialize prefix and prefill confs for any input prefix boxes.
        if init_preds is not None:
            batch_preds = init_preds.long()
            num_prefix_boxes = (batch_preds.size(1) - 1) // tok_per_seq
            confs = list(torch.ones((num_prefix_boxes, B), device=device).unbind(dim=0))
        else:
            batch_preds = torch.full((B, 1), BOS_code, device=device, dtype=torch.long)
            confs = []

        preds = batch_preds  # full prefix on the first predict()
        step_mask = self._pad_mask(batch_preds)
        past_key_values = None
        eos_skip_count = torch.full(
            (B, 1), int(self.eos_skip_num), device=device, dtype=torch.long
        )
        done_mask = torch.zeros(B, dtype=torch.bool, device=device)

        with torch.no_grad(), torch.autocast(
            "cuda", dtype=self._amp_dtype, enabled=self._use_amp
        ):
            for i in range(batch_preds.size(1), self.model_cfg.MAX_SEQ_LENGTH):
                logits, past_key_values = self.predict(
                    encoder_out,
                    preds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    full_attention_mask=step_mask,
                    encoder_features=encoder_features,
                )

                eos_skip_before = eos_skip_count.clone()
                preds, confs, eos_mask = self.sample_from_logits(
                    logits,
                    confs,
                    tokenizer,
                    eos_skip_count,
                    token_idx=i,
                    sample_fn=sample_fn,
                    temperature=temperature,
                )
                done_mask = done_mask | (
                    eos_mask.squeeze(1) & (eos_skip_before == 0).squeeze(1)
                )
                eos_skip_count = torch.where(
                    eos_mask, eos_skip_count - 1, eos_skip_count
                ).clamp(min=0)

                batch_preds = torch.cat([batch_preds, preds], dim=1)
                if done_mask.all():
                    break
                step_mask = torch.cat(
                    [step_mask, torch.ones_like(preds, dtype=torch.float)], dim=1
                )

        return batch_preds, confs

    def _generate_beam(
        self,
        batch_dict,
        tokenizer,
        encoder_features,
        init_preds,
        num_beams,
        length_penalty,
        early_stopping,
        temperature,
    ):
        """Beam search decoding with KV-cache via predict(). Returns (batch_preds, confs).

        Uses HF BeamSearchScorer for active/completed-pool bookkeeping with
        length-penalty applied at finalization. The whole batch runs through one
        scorer; per step the model sees a single (B*num_beams, ...) call.
        """
        device = encoder_features.device
        B = encoder_features.size(0)
        K = int(num_beams)
        BOS_code = tokenizer.BOS_code
        EOS_code = tokenizer.EOS_code
        PAD_code = tokenizer.PAD_code
        tok_per_seq = tokenizer.tok_per_seq
        max_len = self.model_cfg.MAX_SEQ_LENGTH

        if init_preds is not None:
            prefix_ids = init_preds.long()
        else:
            prefix_ids = torch.full((B, 1), BOS_code, device=device, dtype=torch.long)
        L0 = prefix_ids.size(1)
        prefix_mask = self._pad_mask(prefix_ids)  # (B, L0)

        with torch.no_grad(), torch.autocast(
            "cuda", dtype=self._amp_dtype, enabled=self._use_amp
        ):
            prime_logits, base_pkv = self.predict(
                encoder_features,
                prefix_ids,
                past_key_values=None,
                use_cache=True,
                full_attention_mask=prefix_mask,
                encoder_features=encoder_features,
            )
        V = prime_logits.size(-1)

        # Expand to B*K rows. Row order: [b0_k0, b0_k1, ..., b0_k(K-1), b1_k0, ...].
        enc_expanded = encoder_features.repeat_interleave(K, dim=0)  # (B*K, Lf, D)
        input_ids = prefix_ids.repeat_interleave(K, dim=0)  # (B*K, L0)
        prefix_mask_expanded = prefix_mask.repeat_interleave(K, dim=0)  # (B*K, L0)
        past = tuple(
            (
                layer[0].repeat_interleave(K, dim=0).contiguous(),
                layer[1].repeat_interleave(K, dim=0).contiguous(),
            )
            for layer in base_pkv
        )
        eos_skip_count = torch.full(
            (B * K,), int(self.eos_skip_num), device=device, dtype=torch.long
        )

        # HF init trick: zero out the first beam, -inf for the rest, so the first
        # scorer step picks K distinct tokens from a single (shared) prefix.
        beam_scores = torch.full(
            (B, K), float("-inf"), device=device, dtype=torch.float
        )
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)  # (B*K,)

        scorer = BeamSearchScorer(
            batch_size=B,
            num_beams=K,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=1,
        )

        def _topk_view(logprobs_flat):
            """logprobs_flat: (B*K, V) -> (next_scores, next_tokens, next_indices) each (B, 2K)."""
            scores = beam_scores.unsqueeze(-1) + logprobs_flat  # (B*K, V)
            scores = scores.view(B, K * V)
            topk_scores, topk_idx = torch.topk(scores, 2 * K, dim=-1)
            token_ids = topk_idx % V
            beam_indices = torch.div(topk_idx, V, rounding_mode="floor")
            return topk_scores, token_ids, beam_indices

        def _reorder_past(past_, beam_idx_):
            return tuple(
                (l[0].index_select(0, beam_idx_), l[1].index_select(0, beam_idx_))
                for l in past_
            )

        # next_tokens / next_indices outlive the loop for scorer.finalize;
        # initialized so they exist even if the loop body never runs.
        next_tokens = next_indices = None
        step_idx = L0
        logits = (
            prime_logits  # iteration 0 reuses prime; later iterations refresh below.
        )

        with torch.no_grad(), torch.autocast(
            "cuda", dtype=self._amp_dtype, enabled=self._use_amp
        ):
            while step_idx < max_len and not scorer.is_done:
                # Mask invalid tokens, scale, log-softmax.
                token_pos = (step_idx - 1) % tok_per_seq
                token_mask = build_position_validity_mask(
                    V,
                    token_pos,
                    tokenizer,
                    device,
                    enabled=self.limit_box_token_to_valid,
                )
                logits = logits.masked_fill(token_mask.unsqueeze(0), float("-inf"))
                logprobs = torch.log_softmax(logits / temperature, dim=-1)

                # Prime call produces (B, V); expand once so subsequent ops are uniform.
                if logprobs.size(0) == B:
                    logprobs = logprobs.repeat_interleave(K, dim=0)

                # Block EOS while exploration budget remains.
                block = eos_skip_count > 0
                if block.any():
                    logprobs[block, EOS_code] = float("-inf")

                # Score, route via HF scorer.
                next_scores, next_tokens, next_indices = _topk_view(logprobs)
                beam_outputs = scorer.process(
                    input_ids,
                    next_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=PAD_code,
                    eos_token_id=EOS_code,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                next_beam_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                # Append picked tokens, reorder caches, update EOS-skip budget.
                input_ids = torch.cat(
                    [input_ids[beam_idx], next_beam_tokens.unsqueeze(-1)], dim=1
                )
                past = _reorder_past(past, beam_idx)
                eos_skip_count = eos_skip_count[beam_idx]
                eos_skip_count = torch.where(
                    next_beam_tokens == EOS_code,
                    (eos_skip_count - 1).clamp_min(0),
                    eos_skip_count,
                )
                step_idx += 1

                # Refresh logits for the next iteration via one batched predict() call.
                if step_idx < max_len and not scorer.is_done:
                    extra_len = step_idx - L0
                    ones = torch.ones(
                        B * K,
                        extra_len,
                        device=device,
                        dtype=prefix_mask_expanded.dtype,
                    )
                    step_mask = torch.cat([prefix_mask_expanded, ones], dim=1)
                    logits, past = self.predict(
                        enc_expanded,
                        input_ids[:, -1:],
                        past_key_values=past,
                        use_cache=True,
                        full_attention_mask=step_mask,
                        encoder_features=enc_expanded,
                    )  # (B*K, V)

        sequence_outputs = scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=PAD_code,
            eos_token_id=EOS_code,
            max_length=max_len,
        )
        batch_preds = sequence_outputs["sequences"]  # (B, out_len)

        # Post-hoc per-class-token confidences.
        confs = []
        num_prefix_boxes = 0
        if init_preds is not None:
            num_prefix_boxes = max(0, (init_preds.size(1) - 1) // tok_per_seq)
            for _ in range(num_prefix_boxes):
                confs.append(torch.ones((B,), device=device, dtype=torch.float32))

        with torch.no_grad():
            logps, _, valid_mask = self._get_token_logps_and_entropies(
                batch_dict,
                batch_preds,
                tokenizer,
                temperature=temperature,
                compute_entropy=False,
            )

        L = batch_preds.size(1)
        max_boxes_total = max(0, (L - 1 + (tok_per_seq - 1)) // tok_per_seq)
        for k in range(num_prefix_boxes, max_boxes_total):
            t = 1 + k * tok_per_seq + tokenizer._cls_pos
            if t >= L:
                break
            vm = valid_mask[:, t - 1]
            c = torch.zeros((B,), device=device, dtype=torch.float32)
            c[vm] = torch.exp(logps[vm, t - 1])
            confs.append(c.detach())

        return batch_preds, confs

    def sample_from_logits(
        self,
        logits,
        confs,
        tokenizer,
        eos_skip_count,
        token_idx,
        sample_fn,
        temperature=1.0,
    ):
        EOS_code = tokenizer.EOS_code
        tok_per_seq = tokenizer.tok_per_seq
        token_pos = (
            token_idx - 1
        ) % tok_per_seq  # position of the current token in one object sequence

        is_cls_token = token_pos == tokenizer._cls_pos

        probs_original = torch.softmax(logits, dim=-1)  # (B, V)

        # true for masking invalid tokens
        token_mask = build_position_validity_mask(
            tokenizer.vocab_size,
            token_pos,
            tokenizer,
            logits.device,
            enabled=self.limit_box_token_to_valid,
        )

        masked_logits = logits.masked_fill(token_mask.unsqueeze(0), float("-inf"))
        masked_logits = masked_logits / temperature  # apply temperature scaling
        preds = sample_fn(masked_logits)  # (B, 1)

        # Get probability from original distribution for confidence tracking
        preds_prob = probs_original.gather(
            1, preds
        )  # (B, 1), probability of sampled token

        if is_cls_token:
            cls_conf = preds_prob.reshape(-1).detach()
            confs.append(
                cls_conf
            )  # append the confidence of the class token: list of (B)

        # Skip EOS while skip budget remains: replace with second-most-probable token.
        eos_mask = preds == EOS_code  # (B, 1)
        second_best = torch.topk(masked_logits, k=2, dim=-1).indices[:, 1:2]  # (B, 1)
        should_skip = eos_mask & (eos_skip_count > 0)
        preds = torch.where(should_skip, second_best, preds)
        return preds, confs, eos_mask

    def generate_with_logps_and_entropies(
        self, batch_dict, tokenizer, temperature=1.0, compute_entropy=False
    ):
        batch_preds, _ = self.generate(
            batch_dict, tokenizer, temperature=temperature
        )  # (B, L)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            old_per_tok_logps, old_per_tok_entropies, valid_mask = (
                self._get_token_logps_and_entropies(
                    batch_dict,
                    batch_preds,
                    tokenizer,
                    temperature=temperature,
                    compute_entropy=compute_entropy,
                )
            )  # logps: (B, L-1), entropies: (B, L-1) # no BOS

        if was_training:
            self.train()

        output = {
            "pred_seqs": batch_preds,  # (B, L)
            "old_per_tok_logps": old_per_tok_logps,  # (B, L-1)
            "old_per_tok_entropies": old_per_tok_entropies,  # (B, L-1) or None
            "valid_mask": valid_mask,  # (B, L-1)
        }
        return output

    def _get_token_logps_and_entropies(
        self, batch_dict, input_ids, tokenizer, temperature=1.0, compute_entropy=False
    ):

        encoder_out = batch_dict["spatial_features_2d"]  # (B, C, H, W)
        x = encoder_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # PAD tokens after first EOS
        targets = input_ids[:, 1:]
        prefix = input_ids[:, :-1]
        eos_seen = torch.cumsum(prefix.eq(tokenizer.EOS_code), dim=1) > 0
        valid_mask = (~eos_seen) & (
            ~targets.eq(self.pad_token_id)
        )  # (B, L-1), mask out EOS+PAD tokens

        scored_ids = input_ids.clone()
        scored_ids[:, 1:][
            ~valid_mask
        ] = tokenizer.PAD_code  # set invalid tokens to PAD for scoring

        logits = self.forward(x, scored_ids, use_cache=False)  # (B, L, V)

        # Exclude the last value: it corresponds to the next token pred
        logits = logits[:, :-1, :]  # (B, L-1, V)

        logits = logits / temperature

        completion_ids = input_ids[:, 1:]  # (B, L-1)

        logps = selective_log_softmax(
            logits, completion_ids
        )  # compute logprobs # (B, L-1)

        entropies = None
        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)  # (B, L-1)

        return logps, entropies, valid_mask
