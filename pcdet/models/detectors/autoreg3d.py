import torch
from .detector3d_template import Detector3DTemplate
from pcdet.utils.sequence_utils import SequenceTokenizer


class AutoReg3D(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.training_mode = 'sft'

        # Tokenizer
        tokenizer_config = model_cfg.TOKENIZER_CONFIG

        self.tokenizer = SequenceTokenizer(
            min_box_range=tokenizer_config.MIN_BOX_RANGE,
            max_box_range=tokenizer_config.MAX_BOX_RANGE,
            point_cloud_range=self.dataset.point_cloud_range,
            numerical_vocab_sizes=tokenizer_config.NUMERICAL_VOCAB_SIZES,
            class_vocab_size=tokenizer_config.CLASS_VOCAB_SIZE,
            class_placement=tokenizer_config.get("CLASS_PLACEMENT", None),
            predict_velocity=tokenizer_config.get("PREDICT_VELOCITY", False),
        )

        # overload config to pass to dense_head model initialization
        model_cfg.DENSE_HEAD.PAD_TOKEN_ID = self.tokenizer.PAD_code

        # Inform dataset of PAD token for target sequence padding
        self.dataset.set_pad_token_id(self.tokenizer.PAD_code)

        self.module_list = self.build_networks()

        self.backbone_modules = self.module_list[:-2]
        self.bev_compressor = self.module_list[-2]
        self.dense_head = self.module_list[-1]

        self._current_epoch = 0
        self._unfreeze_epoch = 0

    def set_freeze_epoch(self, cur_epoch, unfreeze_epoch):
        """Trainer hook: record the freeze schedule for the current epoch. Returns
        whether the backbone is frozen this epoch; the freeze itself is (re)applied
        by train()."""
        self._current_epoch = cur_epoch
        self._unfreeze_epoch = unfreeze_epoch
        return self._backbone_frozen()

    def _backbone_frozen(self):
        # UNFREEZE_BACKBONE_EPOCH == -1 keeps the backbone frozen for the whole run.
        return self._unfreeze_epoch == -1 or self._current_epoch < self._unfreeze_epoch

    def train(self, mode=True):
        super().train(mode)
        if mode:
            frozen = self._backbone_frozen()
            for m in self.backbone_modules:
                m.eval() if frozen else m.train()
                for p in m.parameters():
                    p.requires_grad = not frozen
        return self

    def forward_features(self, batch_dict):
        for cur_module in self.backbone_modules:
            batch_dict = cur_module(batch_dict)
        batch_dict = self.bev_compressor(batch_dict)
        return batch_dict

    def forward(self, batch_dict):
        batch_dict = self.forward_features(batch_dict)

        if self.training:
            # forward the head on all targets
            loss, tb_dict = self.dense_head.forward_train(batch_dict)
            ret_dict = {"loss": loss}
            disp_dict = {}

            return ret_dict, tb_dict, disp_dict
        else:
            # forward the head on all proposals
            batch_preds, confs = self.dense_head.generate(
                batch_dict, tokenizer=self.tokenizer
            )
            pred_dicts, recall_dicts = self.post_processing(
                batch_dict, batch_preds, confs
            )

            return pred_dicts, recall_dicts

    def post_processing(self, batch_dict, batch_preds, confs):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        recall_dict = {}
        pred_dicts = []

        # confs: B x (max_len // tok_per_seq)
        if isinstance(confs, list):
            # If confs is a list from autoregressive generation, it can be empty when no boxes are predicted.
            # Guard that case by creating a zero-width tensor; otherwise stack along object dimension.
            if len(confs) == 0:
                confs = torch.zeros((batch_preds.size(0), 0), device=batch_preds.device)
            else:
                confs = torch.stack(confs, dim=1)

        # batch_preds: shape (B, max_len)
        for index, pred_seq in enumerate(batch_preds):
            pred_cls, pred_boxes = self.tokenizer.decode(pred_seq)  # Nboxes, Nboxes x 7
            scores = confs[index, : pred_boxes.shape[0]]  # Nboxes

            pred_cls[pred_cls < 0] = 0
            pred_cls[pred_cls > self.num_class] = 0

            # filter noise boxes
            noise_cls_mask = pred_cls == 0  # 0 is the noise class id
            final_boxes = pred_boxes[~noise_cls_mask]
            final_scores = scores[~noise_cls_mask]
            final_labels = pred_cls[~noise_cls_mask]

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
                "pred_seq": pred_seq,  # save for debugging
            }
            pred_dicts.append(record_dict)

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict,
                batch_index=index,
                data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
            )

        return pred_dicts, recall_dict
