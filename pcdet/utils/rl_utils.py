import torch

def reward_fn(pred_boxes, pred_labels, gt_boxes, gt_labels):
    """
    Compute reward based on IoU between predicted boxes and ground truth boxes.
    Args:
        pred_boxes (torch.Tensor): Predicted boxes of shape (N, 7).
        pred_labels (torch.Tensor): Predicted class labels of shape (N,).
        gt_boxes (torch.Tensor): Ground truth boxes of shape (M, 7).
        gt_labels (torch.Tensor): Ground truth class labels of shape (M,).
        use_giou (bool): Whether to use GIoU instead of IoU.
    Returns:
        rewards (torch.Tensor): Computed reward (scalar).
    """
    from pcdet.ops.iou3d_nms import iou3d_nms_utils


    ious = iou3d_nms_utils.boxes_iou3d_gpu(gt_boxes[:, :7], pred_boxes[:, :7])  # (M, N)

    precision_reward_list = []
    recall_reward_list = []
    reward_cls_list = []
    labels_to_consider = torch.unique(torch.cat([gt_labels, pred_labels]))
    for cls in labels_to_consider:
        cls_mask_gt = (gt_labels == cls)
        cls_mask_pred = (pred_labels == cls)
        if torch.sum(cls_mask_gt) == 0 or torch.sum(cls_mask_pred) == 0:
            precision_reward_list.append(torch.tensor(0.0, device=pred_boxes.device))
            recall_reward_list.append(torch.tensor(0.0, device=pred_boxes.device))
            reward_cls_list.append(torch.tensor(0.0, device=pred_boxes.device))
            continue
        ious_cls = ious[cls_mask_gt][:, cls_mask_pred]  # (M_cls, N_cls)
        max_ious_cls, _ = torch.max(ious_cls, dim=1)  # (M_cls,)
        cls_match_iou = torch.clamp(max_ious_cls, min=0.0)  # set rows with no match to 0
        recall_reward_cls = torch.sum(cls_match_iou) / torch.sum(cls_mask_gt).float()   # scalar
        precision_reward_cls = torch.sum(cls_match_iou) / torch.sum(cls_mask_pred).float()  # scalar
        reward_cls = 2 * (recall_reward_cls * precision_reward_cls) / (recall_reward_cls + precision_reward_cls + 1e-6)
        
        reward_cls_list.append(reward_cls)
        precision_reward_list.append(precision_reward_cls)
        recall_reward_list.append(recall_reward_cls)
    
    if len(reward_cls_list) == 0:
        return torch.tensor(0.0, device=pred_boxes.device), torch.tensor(0.0, device=pred_boxes.device), torch.tensor(0.0, device=pred_boxes.device)
    reward = torch.mean(torch.stack(reward_cls_list))
    precision_reward = torch.mean(torch.stack(precision_reward_list))
    recall_reward = torch.mean(torch.stack(recall_reward_list))

    return reward, recall_reward, precision_reward

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])