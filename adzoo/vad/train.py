# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.utils import get_dist_info, init_dist
from os import path as osp


from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import collect_env, get_root_logger
from mmcv.utils import set_random_seed

from mmcv.utils import TORCH_VERSION, digit_version
from adzoo.vad.apis.train import custom_train_model

import cv2
cv2.setNumThreads(1)

import sys
sys.path.append('')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--load-from', 
        help='the checkpoint file to load weights from (without optimizer state)')
    
    parser.add_argument(
        '--load-from-perception',
        help='checkpoint for the perception part (backbone / BEV encoder)')
    parser.add_argument(
        '--load-from-m2m',
        help='checkpoint for mid-to-mid / planning (M2M) part')
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def load_checkpoint_flexible(model, checkpoint_path, logger, map_location='cpu', strict=False):
    """
    Flexibly load checkpoint with better error handling and logging.
    
    Args:
        model: The model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        logger: Logger for output
        map_location: Device to load checkpoint
        strict: Whether to strictly enforce that the keys in checkpoint match the model
        
    Returns:
        checkpoint: Loaded checkpoint dict
    """
    from mmcv.utils import load_checkpoint
    
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    
    # Load checkpoint
    checkpoint = load_checkpoint(
        model, 
        checkpoint_path, 
        map_location=map_location,
        strict=strict,
        logger=logger
    )
    
    # Log loaded info
    if 'meta' in checkpoint and 'epoch' in checkpoint['meta']:
        logger.info(f'Checkpoint epoch: {checkpoint["meta"]["epoch"]}')
    
    if hasattr(checkpoint, 'keys'):
        logger.info(f'Checkpoint keys: {list(checkpoint.keys())}')
    
    return checkpoint

def load_checkpoint_exclude_keys(model, checkpoint_path, logger, 
                                 exclude_keys=None, map_location='cpu'):
    """
    Load a checkpoint while skipping selected keys.

    Args:
        model: the model to load into
        checkpoint_path: path to the checkpoint file
        logger: logger
        exclude_keys: keys to skip (exact match or prefix)
        map_location: device
    """
    if exclude_keys is None:
        exclude_keys = []
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Filter out excluded keys
    filtered_state_dict = {}
    excluded_count = 0
    
    for k, v in state_dict.items():
        # Check if key should be excluded
        should_exclude = False
        for exclude_pattern in exclude_keys:
            if k == exclude_pattern or k.startswith(exclude_pattern):
                should_exclude = True
                break
        
        if should_exclude:
            excluded_count += 1
            logger.info(f"  [EXCLUDED] {k}")
        else:
            filtered_state_dict[k] = v
    
    # Load filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    logger.info(f"Loaded {len(filtered_state_dict)} keys from checkpoint")
    logger.info(f"Excluded {excluded_count} keys")
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    return model

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    if args.load_from is not None:
        # Load pretrained weights (without optimizer state)
        logger.info('='*60)
        logger.info('Loading pretrained weights...')
        logger.info('='*60)
        
        load_checkpoint_flexible(
            model, 
            args.load_from, 
            logger,
            map_location='cpu',
            strict=False  # Allow partial loading
        )
        logger.info('Successfully loaded pretrained weights!')
        logger.info('='*60)
    
    # M2M & E2E !!
    perc_ckpt = None
    if args.load_from_perception is not None:
        perc_ckpt = args.load_from_perception
    elif args.load_from is not None:
        perc_ckpt = args.load_from

    if perc_ckpt is not None and osp.isfile(perc_ckpt):
        logger.info('=' * 60)
        logger.info(f'[Pretrain #1] Loading perception checkpoint from: {perc_ckpt}')
        logger.info('=' * 60)

        load_checkpoint_flexible(
            model,
            perc_ckpt,
            logger,
            map_location='cpu',
            strict=False  # load only the keys that match
        )

        logger.info('[Pretrain #1] Perception checkpoint loaded (strict=False).')
        logger.info('=' * 60)

    # =========================================================
    # (2) mid-to-mid / planning pre-training
    # =========================================================
    if args.load_from_m2m is not None and osp.isfile(args.load_from_m2m):
        logger.info('=' * 60)
        logger.info(f'[Pretrain #2] Loading M2M checkpoint from: {args.load_from_m2m}')
        logger.info('=' * 60)
        
        exclude_keys = [
            'pts_bbox_head.transformer',
            'pts_bbox_head.motion_decoder',
            'pts_bbox_head.motion_mode_query',
            'pts_bbox_head.pos_mlp_sa',
            'pts_bbox_head.bev_embedding',
            'pts_bbox_head.query_embedding',
            'pts_bbox_head.map_query_embedding',
            'pts_bbox_head.map_instance_embedding',
            'pts_bbox_head.map_pts_embedding',
            
            'pts_bbox_head.cls_branches',
            'pts_bbox_head.reg_branches',
            'pts_bbox_head.map_cls_branches',
            'pts_bbox_head.map_reg_branches',
        ]
        # exclude_keys = [
        #     # 'pts_bbox_head.transformer',
        #     'pts_bbox_head.motion_decoder',
        #     'pts_bbox_head.motion_mode_query',
        #     # 'pts_bbox_head.pos_mlp_sa',
        #     # 'pts_bbox_head.bev_embedding',
        #     # 'pts_bbox_head.query_embedding',
        #     # 'pts_bbox_head.map_query_embedding',
        #     # 'pts_bbox_head.map_instance_embedding',
        #     # 'pts_bbox_head.map_pts_embedding',
            
        #     # 'pts_bbox_head.cls_branches',
        #     # 'pts_bbox_head.reg_branches',
        #     # 'pts_bbox_head.map_cls_branches',
        #     # 'pts_bbox_head.map_reg_branches',
        # ]

        load_checkpoint_exclude_keys(
            model,
            args.load_from_m2m,
            logger,
            exclude_keys=exclude_keys,
            map_location='cpu'
        )

        logger.info('[Pretrain #2] M2M checkpoint loaded (strict=False).')
        logger.info('=' * 60)

    # logger.info(f'Model:\n{model}')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # Optional Weights & Biases logging. Disabled unless WANDB_PROJECT is set;
    # every other setting (entity, run name, API key, offline mode) is taken
    # from the usual WANDB_* environment variables.
    if os.environ.get('WANDB_PROJECT'):
        import wandb
        wandb.init(name=os.environ.get('WANDB_NAME',
                                       cfg.work_dir.rstrip('/').split('/')[-1]),
                   config=cfg)
    
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
