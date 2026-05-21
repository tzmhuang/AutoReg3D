"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import iou3d_nms_cuda
from typing import List, Tuple, Optional


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d

def boxes_aligned_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N,)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], 1))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_aligned_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def paired_boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], 1))).zero_()  # (N, ``)
    iou3d_nms_cuda.paired_boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d.view(-1)


def _circular_mean(angles: torch.Tensor) -> torch.Tensor:
    """Compute circular mean of angles (radians), returns scalar tensor."""
    s = torch.sin(angles).mean()
    c = torch.cos(angles).mean()
    return torch.atan2(s, c)

def _circular_median(angles: torch.Tensor) -> torch.Tensor:
    """
    Approximate circular median by unwrapping around the first angle
    and taking median; good enough for tight clusters.
    """
    base = angles[0]
    # unwrap around base
    unwrapped = (angles - base + torch.pi) % (2 * torch.pi) - torch.pi
    return (torch.median(unwrapped).values + base + torch.pi) % (2 * torch.pi) - torch.pi

@torch.no_grad()
def iou_cluster_boxes3d(
    boxes: torch.Tensor,
    iou_thresh: float = 0.3,
    min_cluster_size: int = 1,
    return_labels: bool = True,
    classes: Optional[torch.Tensor] = None,  # <--- NEW
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Cluster 3D boxes by building a graph where edges connect pairs with IoU >= iou_thresh,
    then extracting connected components. Optionally class-aware.

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading]. Should be on CUDA if your IoU kernel requires it.
        iou_thresh: IoU threshold to connect boxes in the graph.
        min_cluster_size: discard clusters smaller than this size (set 1 to keep all).
        return_labels: if True, also return an (N,) tensor of cluster labels (-1 for discarded/noise).
        classes: (N,) int tensor with class ids, or None.
                 If given, boxes of different classes will NEVER be in the same cluster.

    Returns:
        clusters: list of 1D LongTensors with indices per cluster (length = num_clusters_kept).
        labels (optional): (N,) LongTensor of cluster ids in [0..K-1], or -1 if discarded/noise.
    """
    assert boxes.ndim == 2 and boxes.size(1) == 7, "boxes must be (N, 7)"

    N = boxes.size(0)
    device = boxes.device
    if N == 0:
        empty_labels = torch.empty(0, dtype=torch.long, device=device) if return_labels else None
        return [], empty_labels

    if classes is not None:
        assert classes.shape[0] == N, "classes must be shape (N,)"
        # make sure same device for comparisons
        classes = classes.to(device)

    # --- pairwise IoU on GPU ---
    # iou_mat: (N, N)
    iou_mat = boxes_iou3d_gpu(boxes, boxes)

    # Build adjacency (include self to simplify BFS seed)
    adj = iou_mat >= iou_thresh
    adj.fill_diagonal_(True)

    # ---- CLASS AWARENESS: zero out connections between different classes ----
    if classes is not None:
        # same_class[i,j] = True iff classes[i] == classes[j]
        same_class = classes.view(-1, 1).eq(classes.view(1, -1))
        adj &= same_class

    # We'll do a simple BFS on CPU for clarity; convert once to keep it fast.
    adj_cpu = adj.cpu()
    visited = torch.zeros(N, dtype=torch.bool)
    clusters: List[torch.Tensor] = []
    labels = torch.full((N,), -1, dtype=torch.long) if return_labels else None

    from collections import deque
    cid = 0
    for i in range(N):
        if visited[i]:
            continue
        # BFS component
        q = deque([int(i)])
        visited[i] = True
        comp = [int(i)]
        while q:
            u = q.popleft()
            neigh = torch.nonzero(adj_cpu[u], as_tuple=False).flatten().tolist()
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        comp_tensor = torch.tensor(comp, dtype=torch.long)
        if comp_tensor.numel() >= min_cluster_size:
            if labels is not None:
                labels[comp_tensor] = cid
            clusters.append(comp_tensor)
            cid += 1
        else:
            # smaller than min_cluster_size -> keep as noise (-1)
            pass

    if return_labels:
        return clusters, labels.to(device)
    else:
        return clusters, None


# ------------------------ OPTIONAL: merge (box voting) ------------------------

def _circular_mean(angles: torch.Tensor) -> torch.Tensor:
    """Compute circular mean of angles (radians), returns scalar tensor."""
    s = torch.sin(angles).mean()
    c = torch.cos(angles).mean()
    return torch.atan2(s, c)

def _circular_median(angles: torch.Tensor) -> torch.Tensor:
    """
    Approximate circular median by unwrapping around the first angle
    and taking median; good enough for tight clusters.
    """
    base = angles[0]
    # unwrap around base
    unwrapped = (angles - base + torch.pi) % (2 * torch.pi) - torch.pi
    return (torch.median(unwrapped).values + base + torch.pi) % (2 * torch.pi) - torch.pi

@torch.no_grad()
def merge_clusters_box_voting_3d(
    boxes: torch.Tensor,
    clusters: List[torch.Tensor],
    method: str = "mean",  # "mean" or "median"
) -> torch.Tensor:
    """
    Merge each cluster of 3D boxes into one consensus box (voting).
    Uses equal weights (no scores). Heading is handled as circular stat.

    Args:
        boxes: (N, 7) or (N, 9)
               [x, y, z, dx, dy, dz, heading(, vx, vy)]
               original boxes (same tensor used for clustering)
        clusters: list of index tensors (from iou_cluster_boxes3d)
        method: "mean" or "median" (median is more robust to outliers)

    Returns:
        merged: (K, 7) or (K, 9) merged boxes (same device/dtype as input)
    """
    assert method in ("mean", "median")
    if len(clusters) == 0:
        return boxes.new_zeros((0, boxes.size(1)))

    D = boxes.size(1)
    assert D in (7, 9), f"Expected boxes to have 7 or 9 dims, got {D}"

    merged_list = []
    for idxs in clusters:
        B = boxes[idxs]  # (k, D)

        pos = B[:, :3]        # x, y, z
        size = B[:, 3:6]      # dx, dy, dz
        ang = B[:, 6]         # heading

        has_vel = (D == 9)
        if has_vel:
            vel = B[:, 7:9]   # vx, vy

        if method == "mean":
            xyz = pos.mean(dim=0)
            dxdydz = size.mean(dim=0)
            heading = _circular_mean(ang)
            if has_vel:
                vxy = vel.mean(dim=0)
        else:
            xyz = pos.median(dim=0).values
            dxdydz = size.median(dim=0).values
            heading = _circular_median(ang)
            if has_vel:
                vxy = vel.median(dim=0).values

        parts = [xyz, dxdydz, heading.view(1)]
        if has_vel:
            parts.append(vxy)

        merged_list.append(torch.cat(parts, dim=0))

    return torch.stack(merged_list, dim=0)
