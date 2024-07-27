

train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='train',
        data_root='/mnt/nfs_shared_data/dataset/unreal4k',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/u4k/train.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            random_crop=True, # random_crop_size will be set as patch_raw_shape
            network_process_size=[384, 512])))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='infer',
        data_root='/mnt/nfs_shared_data/dataset/unreal4k',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/u4k/val.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))

test_in_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='infer',
        data_root='/mnt/nfs_shared_data/dataset/unreal4k',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/u4k/test_in.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))


test_out_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='infer',
        data_root='/mnt/nfs_shared_data/dataset/unreal4k',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/u4k/test_out.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))