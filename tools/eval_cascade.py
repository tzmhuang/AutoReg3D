import _init_path
import argparse
import datetime
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from easydict import EasyDict

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

# Prefer TF32 for float32 matmuls and convolutions
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch >= 2.0
except AttributeError:
    pass


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--prior_cfg_file', type=str, required=True, help='cfg yaml for the prior model')
    parser.add_argument('--completion_cfg_file', type=str, required=True, help='cfg yaml for the completion model')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--prior_ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--prior_pkl_file', type=str, default=None, help='prior prediction pkl file')
    parser.add_argument('--completion_ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--agg_iou_thresh', type=float, default=0.1, help='aggregation iou threshold')
    parser.add_argument('--min_cluster_num', type=int, default=1, help='minimum number of boxes to form a cluster')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    # completion cfg becomes the "main" cfg: drives dataloader, output dir, post-processing thresholds
    cfg_from_yaml_file(args.completion_cfg_file, cfg)
    cfg.TAG = Path(args.completion_cfg_file).stem + '_cascade'
    cfg.EXP_GROUP_PATH = '/'.join(args.completion_cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # prior cfg is isolated: we only use its MODEL/CLASS_NAMES to build the prior network
    prior_cfg = EasyDict()
    prior_cfg.ROOT_DIR = cfg.ROOT_DIR
    cfg_from_yaml_file(args.prior_cfg_file, prior_cfg)

    assert prior_cfg.CLASS_NAMES == cfg.CLASS_NAMES, \
        'prior and completion configs must share CLASS_NAMES (got prior=%s, completion=%s)' \
        % (prior_cfg.CLASS_NAMES, cfg.CLASS_NAMES)

    np.random.seed(1024)

    return args, cfg, prior_cfg


def run_completion_one_sample(completion_model, data_dict, batch_dict, batch_index, prior_condition_tokens):
    token_id = batch_dict['metadata'][batch_index]['token']
    condition_seq_in = prior_condition_tokens[token_id].unsqueeze(0).cuda(non_blocking=True)

    sample_dict = dict(data_dict)
    sample_dict['spatial_features_2d'] = data_dict['spatial_features_2d'][batch_index:batch_index + 1]
    if 'gt_boxes' in data_dict:
        sample_dict['gt_boxes'] = data_dict['gt_boxes'][batch_index:batch_index + 1]
    if 'rois' in data_dict:
        sample_dict['rois'] = data_dict['rois'][batch_index:batch_index + 1]
    sample_dict['batch_size'] = 1

    completion_dict, confs = completion_model.dense_head.generate(
        sample_dict, completion_model.tokenizer, init_preds=condition_seq_in
    )
    return completion_model.post_processing(sample_dict, completion_dict, confs)


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def merge_overlapping_predictions(det_annos, iou_thresh, min_cluster_size):
    for idx in tqdm.tqdm(range(len(det_annos))):
        annos = det_annos[idx]
        all_boxes = annos['boxes_lidar']
        class_labels = annos['pred_labels']

        class_labels_tensor = torch.from_numpy(class_labels).cuda()
        all_boxes_tensor = torch.from_numpy(all_boxes).cuda().float()
        clusters, _ = iou3d_nms_utils.iou_cluster_boxes3d(
            all_boxes_tensor[:, :7],
            iou_thresh=iou_thresh,
            min_cluster_size=min_cluster_size,
            classes=class_labels_tensor,
        )
        merged_boxes = iou3d_nms_utils.merge_clusters_box_voting_3d(all_boxes_tensor, clusters, method='mean')
        merged_boxes_labels = [class_labels[c[0]].item() for c in clusters]

        det_annos[idx]['boxes_lidar'] = merged_boxes.cpu().numpy()
        det_annos[idx]['pred_labels'] = np.array(merged_boxes_labels)
        det_annos[idx]['score'] = np.array([np.mean(annos['score'][c.numpy()]).item() for c in clusters])
        det_annos[idx]['name'] = [annos['name'][c[0]] for c in clusters]


def eval_one_epoch_condition(cfg, args, completion_model, prior_condition_tokens, dataloader, logger,
                             dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names

    det_annos = []

    if getattr(args, 'infer_time', False):
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** GENERATING PREDICTIONS *****************')
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        completion_model = torch.nn.parallel.DistributedDataParallel(
                completion_model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    completion_model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        # Backbone runs on the full batch. AR generation is per-sample because
        # each frame has a different-length prior prefix to condition on.
        with torch.no_grad():
            data_dict = completion_model.forward_features(batch_dict)

            pred_dicts = []
            recall_dicts = {}
            for batch_index in range(batch_dict['batch_size']):
                pred_dicts_b, recall_dict_b = run_completion_one_sample(
                    completion_model, data_dict, batch_dict, batch_index, prior_condition_tokens
                )
                pred_dicts.extend(pred_dicts_b)
                for k, v in recall_dict_b.items():
                    recall_dicts[k] = recall_dicts.get(k, 0) + v

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, recall_dicts, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of GENERATION *****************')
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))


    merge_overlapping_predictions(
        det_annos,
        iou_thresh=float(args.agg_iou_thresh),
        min_cluster_size=args.min_cluster_num,
    )

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def load_prior_prediction_pkl(prior_pred_file):
    with open(prior_pred_file, 'rb') as f:
        prior_pred_annos = pickle.load(f)
    return prior_pred_annos


def build_prior_condition_tokens(prior_pred_annos, tokenizer):
    condition_tokens = {}
    eos_code = tokenizer.EOS_code

    for item in prior_pred_annos:
        boxes_lidar = item['boxes_lidar']
        pred_labels = item['pred_labels']
        prior_pred_np = np.empty((boxes_lidar.shape[0], boxes_lidar.shape[1] + 1), dtype=boxes_lidar.dtype)
        prior_pred_np[:, :-1] = boxes_lidar
        prior_pred_np[:, -1] = pred_labels

        np.random.shuffle(prior_pred_np)
        condition_seq = tokenizer(prior_pred_np)

        condition_seq = condition_seq[:-1] if condition_seq[-1] == eos_code else condition_seq[condition_seq != eos_code]
        condition_tokens[item['metadata']['token']] = torch.as_tensor(condition_seq, dtype=torch.long)

    return condition_tokens


def run_condition_sampling(prior_model, completion_model, test_loader, args, eval_output_dir, logger, dist_test=False):
    prior_dir = eval_output_dir / 'prior_eval'
    prior_dir.mkdir(parents=True, exist_ok=True)
    cached_prior_pkl = prior_dir / 'result.pkl'

    if args.prior_pkl_file is not None:
        prior_pkl_path = Path(args.prior_pkl_file)
        logger.info('Using user-provided prior pkl: %s' % prior_pkl_path)
    elif cached_prior_pkl.exists():
        prior_pkl_path = cached_prior_pkl
        logger.info('Found cached prior eval result at %s, skipping prior eval' % prior_pkl_path)
    else:
        assert args.prior_ckpt is not None, '--prior_ckpt is required (or pass --prior_pkl_file)'
        logger.info('Running prior eval with ckpt: %s' % args.prior_ckpt)
        prior_model.load_params_from_file(filename=args.prior_ckpt, logger=logger, to_cpu=True,
                                          pre_trained_path=args.pretrained_model)
        prior_model.cuda()
        eval_utils.eval_one_epoch(
            cfg, args, prior_model, test_loader, epoch_id='prior',
            logger=logger, dist_test=dist_test, result_dir=prior_dir
        )
        prior_pkl_path = cached_prior_pkl
        if dist_test:
            torch.distributed.barrier()

    # release prior model memory before loading the completion model
    del prior_model
    torch.cuda.empty_cache()

    prior_pred_annos = load_prior_prediction_pkl(prior_pkl_path)

    assert args.completion_ckpt is not None, '--completion_ckpt is required'
    logger.info('Loading completion ckpt: %s' % args.completion_ckpt)
    completion_model.load_params_from_file(filename=args.completion_ckpt, logger=logger, to_cpu=True,
                                           pre_trained_path=args.pretrained_model)
    completion_model.cuda()

    prior_condition_tokens = build_prior_condition_tokens(prior_pred_annos, completion_model.tokenizer)

    return eval_one_epoch_condition(
        cfg, args, completion_model, prior_condition_tokens, test_loader, logger,
        dist_test=dist_test, result_dir=eval_output_dir
    )


def main():
    args, cfg, prior_cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval' / cfg.DATA_CONFIG.DATA_SPLIT['test']
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model_prior = build_network(model_cfg=prior_cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model_completion = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    with torch.no_grad():
        run_condition_sampling(model_prior, model_completion, test_loader, args, eval_output_dir, logger, dist_test=dist_test)


if __name__ == '__main__':
    main()
