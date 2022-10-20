num_classes=18
model = dict(
    type='EncoderDecoder',
    pretrained='/data/pretrained/convnext_small_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],#
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],#
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),            
            dict(type='LovaszLoss', loss_name='loss_lova',
            reduction='none', loss_weight=0.5)]
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.75,1.0,1.5,2.0,2.5,3.0],#[0.75,1.0,1.25,1.5,1.75,2.0]
        flip=True,
        # flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,norm_per_pic=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type='Seg18Dataset',
        img_dir='/data/fusai_release/train/images',#test/images2
        img_suffix='.tif',
        load_mean_std=True,
        split="valid.txt",#选择图像id
        k_fold_use=False,
        pipeline=test_pipeline))