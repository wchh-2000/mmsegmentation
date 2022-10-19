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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,#
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albumentations = [#比较去掉ShiftScaleRotate的data_time 精度
    dict(type='RandomBrightnessContrast', p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=30,
        p=0.3),
    dict(
        type='RGBShift',
        r_shift_limit=20,
        g_shift_limit=20,
        b_shift_limit=20,
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', blur_limit=7, p=1.0)
        ],
        p=0.1)
]
size = 512
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(size, size), ratio_range=(0.5, 3)),
    dict(type='RandomCrop', crop_size=(size, size), cat_max_ratio=0.75),
    dict(type='Albu', transforms=albumentations),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=45),#替代ShiftScaleRotate
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
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
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_type = 'Seg18Dataset'
# CLASSES = ['background', 'water','transport', 'building','agricultural','grass', 'forest','barren','others']
CLASSES = ['Background','Waters', 'Road', 'Construction', 'Airport', 'Railway Station', 'Photovoltaic panels', 'Parking Lot', 'Playground',
           'Farmland', 'Greenhouse', 'Grass', 'Artificial grass', 'Forest', 'Artificial forest', 'Bare soil', 'Artificial bare soil', 'Other']
data = dict(
    samples_per_gpu=2,#12
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        img_dir='/data/fusai_release/train/images',
        ann_dir='/data/fusai_release/train/labels_18',
        img_suffix='.tif',
        seg_map_suffix='.png',
        split='/data/mmseg/C_run/train.txt',
        k_fold_use=False,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        img_dir='/data/fusai_release/test/images2',
        img_suffix='.tif',
        seg_map_suffix='.png',
        k_fold_use=False,
        pipeline=test_pipeline))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=1e-4,#2e-4
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)),
            head=dict(lr_mult=10.0)
            ))
#optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    power=1.0,
    min_lr=1e-6,
    by_epoch=False)
# runner = dict(type='IterBasedRunner', max_iters=40000)
runner = dict(type='EpochBasedRunner', max_epochs=30)
# optimizer_config = dict()
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
# checkpoint_config = dict(by_epoch=False, interval=40000)
checkpoint_config = dict(by_epoch=True, interval=10,max_keep_ckpts=2)
# evaluation = dict(interval=60, metric='mIoU', pre_eval=True)