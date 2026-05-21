import numpy as np
import torch

PAD_TOKEN = 0


class SequenceTokenizer:
    """
    Tokenizes 3D bounding boxes into discrete sequences using one
    independent vocabulary per numerical dimension. Each box becomes
    `tok_per_seq` tokens — one class token plus 7 numerical (or 9 with
    velocity) — interleaved per `class_placement` ('first' / 'last' /
    'middle').

    Vocab layout (token id 0 reserved for PAD):
      [PAD] [x codes] [y codes] [z codes]
            [dx codes] [dy codes] [dz codes] [heading codes]
            [vx codes] [vy codes]                # only if predict_velocity
            [class codes]
            [BOS] [EOS]
    """

    def __init__(
        self,
        min_box_range,
        max_box_range,
        point_cloud_range,
        numerical_vocab_sizes,
        class_vocab_size,
        class_placement=None,
        predict_velocity=False,
    ):
        self.min_box_range = min_box_range
        self.max_box_range = max_box_range
        self.point_cloud_range = point_cloud_range
        self.class_vocab_size = class_vocab_size
        self.class_placement = class_placement
        self.predict_velocity = predict_velocity

        # Box numerical dims: x, y, z, dx, dy, dz, heading, [vx, vy]
        if predict_velocity:
            self.tok_per_seq = 10  # 9 numerical + 1 class
            n_box_dims = 9
        else:
            self.tok_per_seq = 8  # 7 numerical + 1 class
            n_box_dims = 7

        assert (
            len(self.min_box_range) == n_box_dims - 3
        ), "min_box_range length mismatch"
        assert (
            len(self.max_box_range) == n_box_dims - 3
        ), "max_box_range length mismatch"

        self.numerical_ranges = np.zeros((2, n_box_dims))
        self.numerical_ranges[0, :3] = self.point_cloud_range[:3]
        self.numerical_ranges[1, :3] = self.point_cloud_range[3:]
        self.numerical_ranges[0, 3:] = self.min_box_range
        self.numerical_ranges[1, 3:] = self.max_box_range

        self.numerical_vocab_sizes = np.array(numerical_vocab_sizes)[np.newaxis, :]
        self.vocab_size = (
            self.class_vocab_size + self.numerical_vocab_sizes.sum() + 3
        )  # PAD + BOS + EOS

        self._cls_pos, self.box_cols = self._get_token_layout()
        self._init_codes()

    def _get_token_layout(self):
        """Return (cls_pos, box_cols): the column index of the class token
        and the columns occupied by numerical tokens, within one box's
        `tok_per_seq`-long row."""
        T = self.tok_per_seq
        if self.class_placement == "first":
            # [cls, x, y, z, dx, dy, dz, heading, (vx, vy)]
            return 0, list(range(1, T))
        if self.class_placement == "last":
            # [x, y, z, dx, dy, dz, heading, (vx, vy), cls]
            return T - 1, list(range(0, T - 1))
        if self.class_placement == "middle":
            # [x, y, z, cls, dx, dy, dz, heading, (vx, vy)]
            return 3, list(range(0, 3)) + list(range(4, T))
        raise ValueError(f"Invalid class_placement: {self.class_placement}")

    def _init_codes(self):
        """Pre-compute fixed token-id ranges for each dim and for class."""

        self.box_tok_offs = (
            np.concatenate(
                [np.array([0]), self.numerical_vocab_sizes.cumsum()[:-1]], axis=0
            )
            + 1
        )

        self.cls_tok_offs = int(self.numerical_vocab_sizes.sum() + 1)

        self.PAD_code = PAD_TOKEN
        self.BOS_code = self.cls_tok_offs + self.class_vocab_size
        self.EOS_code = self.BOS_code + 1

        sizes = self.numerical_vocab_sizes.flatten()

        self.cls_codes = np.arange(
            self.cls_tok_offs, self.cls_tok_offs + self.class_vocab_size
        )
        self.x_codes = np.arange(self.box_tok_offs[0], self.box_tok_offs[0] + sizes[0])
        self.y_codes = np.arange(self.box_tok_offs[1], self.box_tok_offs[1] + sizes[1])
        self.z_codes = np.arange(self.box_tok_offs[2], self.box_tok_offs[2] + sizes[2])
        self.dx_codes = np.arange(self.box_tok_offs[3], self.box_tok_offs[3] + sizes[3])
        self.dy_codes = np.arange(self.box_tok_offs[4], self.box_tok_offs[4] + sizes[4])
        self.dz_codes = np.arange(self.box_tok_offs[5], self.box_tok_offs[5] + sizes[5])
        self.heading_codes = np.arange(
            self.box_tok_offs[6], self.box_tok_offs[6] + sizes[6]
        )

        # Noise set to be first class
        self.noise_cls_code = self.cls_codes[0]

        if self.predict_velocity:
            self.vx_codes = np.arange(
                self.box_tok_offs[7], self.box_tok_offs[7] + sizes[7]
            )
            self.vy_codes = np.arange(
                self.box_tok_offs[8], self.box_tok_offs[8] + sizes[8]
            )

    def __len__(self):
        return self.vocab_size

    def tokenize_numerical_vals(self, num_arr, value_range, vocab_size, tok_offs=0):
        """Quantize per-dim numerical values into token ids.

        Args:
            num_arr     : (N, D) numerical values.
            value_range : (2, D) [min, max] bounds per dim.
            vocab_size  : per-dim vocab sizes; broadcasts against num_arr.
            tok_offs    : per-dim base token id; broadcasts against num_arr.

        Returns:
            (N, D) int array of token ids in [tok_offs, tok_offs + vocab_size).
        """

        num_arr = (num_arr - value_range[0]) / (value_range[1] - value_range[0])

        # Velocity dims can be NaN, map to 0
        num_arr[np.isnan(num_arr)] = 0.0

        token_idx = np.round(num_arr * (vocab_size - 1)).astype(int)
        token_idx = np.clip(token_idx, 0, vocab_size - 1)
        return token_idx + tok_offs

    def tokenize_class_discrete(self, cls_arr, tok_offs=0):
        """Convert class ids to class-token ids by adding `tok_offs`."""
        return cls_arr.astype(int) + tok_offs

    def _assemble(self, box_toks, cls_toks):
        """Interleave per-box numerical and class tokens per `class_placement`,
        flatten across boxes, and bracket with BOS / EOS."""
        # Lay out one box per row, columns chosen by class_placement.
        seq = np.empty((box_toks.shape[0], self.tok_per_seq), dtype=int)
        seq[:, self._cls_pos] = cls_toks
        seq[:, self.box_cols] = box_toks
        # Flatten boxes into a single stream and frame with BOS / EOS.
        return np.concatenate([[self.BOS_code], seq.reshape(-1), [self.EOS_code]])

    def __call__(self, gt_boxes, return_out_seq=False):
        """Tokenize ground-truth boxes into a discrete input sequence.

        Args:
            gt_boxes        : (N, 8) or (N, 10). Last column is class id;
                              preceding columns are numerical box dims in
                              order: x, y, z, dx, dy, dz, heading, [vx, vy].
            return_out_seq  : If True, also return a target sequence where
                              noise-class boxes' numerical tokens are
                              replaced by PAD (so the box-dim loss skips them).

        Returns:
            seq                 : 1-D input sequence            (return_out_seq=False)
            (seq, seq_masked)   : (input, target) pair          (return_out_seq=True)
        """

        box_toks = self.tokenize_numerical_vals(
            gt_boxes[:, : self.tok_per_seq - 1],
            self.numerical_ranges,
            self.numerical_vocab_sizes,
            tok_offs=self.box_tok_offs,
        )

        cls_toks = self.tokenize_class_discrete(
            gt_boxes[:, self.tok_per_seq - 1],
            tok_offs=self.cls_tok_offs,
        )

        seq = self._assemble(box_toks, cls_toks)
        if not return_out_seq:
            return seq

        box_toks_masked = box_toks.copy()
        box_toks_masked[cls_toks == self.noise_cls_code] = self.PAD_code
        seq_masked = self._assemble(box_toks_masked, cls_toks)
        return seq, seq_masked

    def decode(self, sequence):
        """Inverse of `__call__`: turn a token sequence back into class ids
        and real-valued box parameters.

        Args:
            sequence : 1-D tensor of token ids, expected to start with BOS.

        Returns:
            cls : (N,)   integer class ids (post-offset removal).
            box : (N, D) real-valued box dims, where D = tok_per_seq - 1.
        """

        eos_idx = torch.where(sequence == self.EOS_code)[0]
        eos_idx = eos_idx[0] if len(eos_idx) > 0 else None
        sequence = sequence[1:eos_idx]

        # If the sequence was truncated mid-box, drop the partial trailing group.
        rem = len(sequence) % self.tok_per_seq
        if rem:
            sequence = sequence[:-rem]
        sequence = sequence.reshape(-1, self.tok_per_seq)

        cls_toks = sequence[:, self._cls_pos]
        box_toks = sequence[:, self.box_cols]

        # De-quantize box tokens back to real values.
        device = sequence.device
        box_tok_offs_t = torch.from_numpy(self.box_tok_offs).to(device)
        ranges_t = torch.from_numpy(self.numerical_ranges).to(device)
        sizes_t = torch.from_numpy(self.numerical_vocab_sizes).to(device)

        cls = cls_toks - self.cls_tok_offs
        box = (box_toks - box_tok_offs_t).float()
        box = box / (sizes_t - 1) * (ranges_t[1] - ranges_t[0]) + ranges_t[0]
        return cls, box.float()
