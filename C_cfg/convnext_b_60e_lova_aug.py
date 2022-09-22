num_classes=9
model = dict(
    type='EncoderDecoder',
    pretrained='/data/checkpoints/convnext_base_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128,256, 512, 1024],#
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128,256, 512, 1024],#
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
        in_channels=512,#
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
albumentations = [#改过
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.2,
        scale_limit=0.2,
        rotate_limit=20,
        # shift_limit=0,#lower and upper bounds should lie in range [0, 1] 与random crop功能重复
        # scale_limit=0,#(1-scale_limit, 1+scale_limit) 与resize ratio range功能重复
        # rotate_limit=45,
        # border_mode=0,value=0,
        p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5),
    dict(type="OpticalDistortion",distort_limit=0.2,
        border_mode=0,value=0,p=0.1),#0:cv2.BORDER_CONSTANT
    dict(type='GaussNoise', var_limit=(10.0, 40.0), p=0.3),
    dict(
        type='RGBShift',
        r_shift_limit=10,
        g_shift_limit=20,
        b_shift_limit=30,
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', blur_limit=7, p=1.0)#(3, 7)
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
    #dict(type='RandomRotate', prob=0.5, degree=45),#替代ShiftScaleRotate
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
dataset_type = 'CustomDataset'
CLASSES = ['background', 'water','transport', 'building','agricultural','grass', 'forest','barren','others'
               ]
data = dict(
    samples_per_gpu=24,#16
    workers_per_gpu=12,#8
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        img_dir='/data/chusai_release/train/images',
        ann_dir='/data/chusai_release/train/labels_9',
        img_suffix='.tif',
        seg_map_suffix='.png',
        k_fold_use=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        img_dir='/data/chusai_release/train/images',
        ann_dir='/data/chusai_release/train/labels_9',
        img_suffix='.tif',
        seg_map_suffix='.png',
        k_fold_use=False,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        img_dir='/data/chusai_release/testA/images',
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
    lr=2e-4,#2e-4
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
    min_lr=0.0,
    by_epoch=False)
# runner = dict(type='IterBasedRunner', max_iters=40000)
runner = dict(type='EpochBasedRunner', max_epochs=60)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
# checkpoint_config = dict(by_epoch=False, interval=40000)
checkpoint_config = dict(by_epoch=True, interval=60)
evaluation = dict(interval=60, metric='mIoU', pre_eval=True)