

train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='Inter4KDataset',
        mode='train',
        data_root='/mnt/nfs_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/inter4k/train.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            random_crop=False, # random_crop_size will be set as patch_raw_shape
            network_process_size=[1064, 1904])))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='Inter4KDataset',
        mode='infer',
        data_root='/mnt/nfs_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/inter4k/val.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[1064, 1904])))

test_in_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='Inter4KDataset',
        mode='infer',
        data_root='/mnt/nfs_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/inter4k/train.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[1064, 1904])))


test_out_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='Inter4KDataset',
        mode='infer',
        data_root='/mnt/nfs_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/',
        split='/home/dongjae.lee/ftp_shared_internal/Projects/SEC_VD/1.Code/PatchFusion/splits/mvs/test.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[1064, 1904])))