import json
import os
from collections import defaultdict
from typing import Callable, Tuple

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import (
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.evaluate import NuScenesEval


def _exact_global_pr_f1(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
) -> Tuple[float, float, float]:
    # Mirrors the matching block of nuscenes.eval.detection.algo.accumulate
    npos = sum(1 for b in gt_boxes.all if b.detection_name == class_name)
    if npos == 0:
        return 0.0, 0.0, 0.0

    pred_list = [b for b in pred_boxes.all if b.detection_name == class_name]
    if not pred_list:
        return 0.0, 0.0, 0.0

    pred_list.sort(key=lambda b: b.detection_score, reverse=True)

    taken = set()
    tp = 0
    for pred in pred_list:
        min_dist = np.inf
        match_idx = None
        for gt_idx, gt in enumerate(gt_boxes[pred.sample_token]):
            if gt.detection_name == class_name and (pred.sample_token, gt_idx) not in taken:
                d = dist_fcn(gt, pred)
                if d < min_dist:
                    min_dist = d
                    match_idx = gt_idx
        if min_dist < dist_th:
            taken.add((pred.sample_token, match_idx))
            tp += 1

    fp = len(pred_list) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / npos
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(prec), float(rec), float(f1)


class NuScenesEvalWithF1(NuScenesEval):
    """
    Standard nuScenes detection eval plus per-class precision / recall / F1.
    """

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        metrics, mdl = super().evaluate()

        extra = {
            'global_precs': defaultdict(dict),
            'global_recs': defaultdict(dict),
            'f1_scores': defaultdict(dict),
        }
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                p, r, f1 = _exact_global_pr_f1(
                    self.gt_boxes,
                    self.pred_boxes,
                    class_name,
                    self.cfg.dist_fcn_callable,
                    dist_th,
                )
                extra['global_precs'][class_name][dist_th] = p
                extra['global_recs'][class_name][dist_th] = r
                extra['f1_scores'][class_name][dist_th] = f1

        self._extra = {k: {c: dict(v) for c, v in d.items()} for k, d in extra.items()}
        return metrics, mdl

    def main(self, plot_examples: int = 0, render_curves: bool = True):
        summary = super().main(plot_examples=plot_examples, render_curves=render_curves)
        summary.update(self._extra)

        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        f1_vals = [v for d in self._extra['f1_scores'].values() for v in d.values()]
        mean_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0
        print('Mean F1: %.4f' % mean_f1)

        return summary
