_base_ = [
    '/data/mmseg/configs/_base_/default_runtime.py',
    '/data/mmseg/configs/_base_/schedules/schedule_160k.py'
]

runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)
ignore_index = 0
# model settings
workflow = [('train', 100), ('val', 1)]
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
        #dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
checkpoint_file=''#"/data/checkpoints_convnext/swin_base_patch4_window7_224_22k.pth"
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],#[2, 2, 6, 2] base:[2, 2, 18, 2]
        num_heads=[3, 6, 12, 24],#[3, 6, 12, 24] [4, 8, 16, 32]
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],#[96, 192, 384, 768] [128, 256, 512, 1024]
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,#384 512
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,#2e-4
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
            #'head':dict(lr_mult=10.0)#convnext
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# dataset settings
dataset_type = 'Seg8Dataset'
data_root = '/data/chusai_release'
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# [0.2943120709757581, 0.25569606427073105, 0.24184218997413898], [0.167524461849462, 0.13983301451377886, 0.1284466821323334]

albumentations = [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.5
            ),
            dict(type='RandomBrightnessContrast', p=0.5),
            dict(type='GaussNoise', p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                    ],
                p=0.1
            ),
]
size = 512
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MyLoadAnnotations'),
    dict(type='Resize', img_scale=(size, size), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(size, size), cat_max_ratio=0.75),
    dict(type='Albu', transforms=albumentations),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=ignore_index),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5,0.75,1.0,1.25,1.5,1.75,2.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        val_mode=False,
        k_fold_value=5,
        k_fold_start=1,
        ignore_index=ignore_index,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        val_mode=True,
        k_fold_value=5,
        k_fold_start=1,
        ignore_index=ignore_index,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='testA/images',
        val_mode=True,
        k_fold_value=1,
        pipeline=test_pipeline))
