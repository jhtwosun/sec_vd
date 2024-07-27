import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger

import sys
sys.path.append('/home/gyojin.han/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/')
from estimator.utils import RunnerInfo, setup_env, log_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.trainer import Trainer
from estimator.datasets.transformers import to_tensor

import warnings
warnings.filterwarnings(action='ignore')

import cv2
from torchvision import transforms
from estimator.datasets.transformers import Resize as ResizeDA

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode')
    parser.add_argument(
        '--log-name',
        type=str, default='',
        help='log_name for wandb')
    parser.add_argument(
        '--tags',
        type=str, default='',
        help='tags for wandb')
    parser.add_argument(
        '--seed',
        type=int, default=208,
        help='for debug')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    
    args = parse_args()
    
    # if args.debug:
    #     torch.autograd.set_detect_anomaly(True) # for debug

    # load config
    cfg = Config.fromfile(args.config)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    cfg.work_dir = args.work_dir
    cfg.work_dir = osp.join(cfg.work_dir, args.log_name)
    
    mkdir_or_exist(cfg.work_dir)
    cfg.debug = args.debug
    cfg.log_name = args.log_name
    tags = args.tags
    if ',' in tags:
        tag_list = tags.split(',')
    else:
        tag_list = [tags]
    cfg.tags = tag_list
    
    # fix seed
    seed = args.seed
    fix_random_seed(seed)
    
    # start dist training
    distributed = False
    # if cfg.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
    rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # prepare basic text logger
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', timestamp)
    log_cfg.setdefault('logger_name', 'patchstitcher')
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)
    
    # save some information useful during the training
    runner_info = RunnerInfo()
    runner_info.config = cfg # ideally, cfg should not be changed during process. information should be temp saved in runner_info
    runner_info.logger = logger # easier way: use print_log("infos", logger='current')
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    
    # start wandb
    if runner_info.rank == 0 and cfg.debug == False:
        wandb.init(
            project=cfg.project, 
            name=cfg.log_name+"_"+runner_info.timestamp, 
            tags=cfg.tags, 
            dir=runner_info.work_dir,
            config=cfg, # have a test
            settings=wandb.Settings(start_method="fork"))
        
        wandb.define_metric("Val/step")
        wandb.define_metric("Val/*", step_metric="Val/step")
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/*", step_metric="Train/step")
    
    log_env(cfg, env_cfg, runner_info, logger)
    
    # resume training (future)
    cfg.resume = args.resume
    
    # build model
    model = build_model(cfg.model)
    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        if cfg.get('convert_syncbn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runner_info.rank], output_device=runner_info.rank,
                                                          find_unused_parameters=cfg.get('find_unused_parameters', False))
        logger.info(model)
    else:
        model = model.cuda(runner_info.rank)
        logger.info(model)
    
    # model.depth_model.load_state_dict(torch.load('/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/work_dir/depthanything_vitl_mvs/mvs_vits_slice22_prompt4/checkpoint_50.pth')['model_state_dict'])
    # model.depth_model.load_state_dict(torch.load('/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/work_dir/depthanything_vitl_mvs/mvs_vits_slice44_prompt4/checkpoint_50.pth')['model_state_dict'])
    model.depth_model.load_state_dict(torch.load('/home/gyojin.han/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/pretrained_ckpt/depth_anything_v2_vits.pth'))
    # model.depth_model.load_state_dict(torch.load('/home/gyojin.han/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/pretrained_ckpt/depth_anything_v2_vitl.pth'))


    model.eval()
    
    
    amp = cfg.get('amp', None)
    if amp is not None: 
        if amp == 'tf32':
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("Using TF32 training")
        else:
            assert amp in ['bf16', 'fp16'], f"amp should be 'bf16' or 'fp16', got {amp}"
            logger.info(f"Using {amp} mixed precision training")
    else:
        logger.info("Using FP32 training")
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize = ResizeDA(1904, 1064, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
    margin_width = 50
    video_path = '/home/gyojin.han/ftp_shared_data/dataset/Inter4K/Inter4K/60fps/UHD'
    if os.path.isfile(video_path):
        if video_path.endswith('txt'):
            with open(video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [video_path]
    else:
        filenames = os.listdir(video_path)
        filenames = [os.path.join(video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    os.makedirs(args.work_dir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(frame_width, frame_height)
        # frame_width, frame_height = 1904, 1064
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        filename = os.path.basename(filename)
        output_path = os.path.join(args.work_dir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        frame_idx = 0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame = resize(dict(image=frame))['image']
            frame = normalize(to_tensor(frame))
            frame = frame.unsqueeze(0).cuda(runner_info.rank)
            with torch.no_grad():
                depth, log_dict = model(mode='infer', cai_mode='m1', process_num=4, image_hr=frame, depth_gt=None, disp_gt=None)

            depth = F.interpolate(depth.squeeze(0)[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
            depth = depth.cpu().numpy().astype(np.uint8)
            video_idx = filename.split('.')[0].zfill(4)
            npy_name = str(frame_idx).zfill(4) + '.npy'
            os.makedirs('/home/gyojin.han/ftp_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/' + video_idx +'/depths/', exist_ok=True)
            np.save('/home/gyojin.han/ftp_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/' + video_idx + '/depths/' + npy_name, depth)
            frame_idx += 1
            # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            
            # split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            # combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
            
            # out.write(combined_frame)
        
        raw_video.release()
        # out.release()
        
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings(action='ignore')

    main()