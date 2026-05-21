import torch
from torch import nn

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class ConvBEVCompressor(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size):
        super().__init__()
        self.model_cfg = model_cfg

        self.num_bev_features = self.model_cfg.COMPRESSED_BEV_FEATURES
        self.grid_size = self.model_cfg.COMPRESSED_GRID_SIZE
        self.block = BasicBlock2D(in_channels=input_channels,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)
        self.init_weights()


    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch_dict):
        spatial_features_2d = batch_dict['spatial_features_2d']  # (B, C, H, W)
        bev_features = self.block(spatial_features_2d)  # (B, C, H, W) -> (B, C', H', W')
        batch_dict["spatial_features_2d"] = bev_features
        return batch_dict
