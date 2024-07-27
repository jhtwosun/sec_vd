_base_ = [
    '../_base_/datasets/inter4k.py', 
]

min_depth=1e-3
max_depth=80
patch_process_shape=(1064, 1904)

amp = None #tf32 bf16 fp16 None

core_config=dict(  
    encoder_type='vits_prompt',
    pretrained_resource='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/pretrained_ckpt/depth_anything_v2_vits.pth',
    use_pretrained_midas=True,
    train_midas=False,    
    img_size=patch_process_shape,
    num_slice=(2, 2),
    num_prompt_tokens=4
)

model=dict(
    type='RelativeDepthModel',
    patch_process_shape=patch_process_shape,
    min_depth=min_depth,
    max_depth=max_depth,
    core_cfg=core_config,
    sigloss=dict(type='AffineInvariantLoss'))

#collect_input_args=['image_lr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs', 'image_hr']
collect_input_args=[ 'depth_gt', 'image_hr', 'disp_gt']
project='patchfusion'

train_cfg=dict(max_epochs=50, val_interval=2, save_checkpoint_interval=10, log_interval=100, train_log_img_interval=500, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

optim_wrapper=dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    # optimizer=dict(type='AdamW', lr=0.0002/50, weight_decay=0.01),
    clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
        }))

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=1,
    final_div_factor=10000,
    pct_start=0.5,
    three_phase=False,)

env_cfg=dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='forkserver'),
    dist_cfg=dict(backend='nccl'))

convert_syncbn=True
find_unused_parameters=True


train_dataloader=dict(
    dataset=dict(
        resize_mode='depth-anything',
        transform_cfg=dict(
            network_process_size=patch_process_shape)))

val_dataloader=dict(
    dataset=dict(
        resize_mode='depth-anything',
        transform_cfg=dict(
            network_process_size=patch_process_shape)))