num_classes=9
large=dict(
    type='EncoderDecoder',
    pretrained='/data/checkpoints/convnext_large_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],#
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],#
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False),
    test_cfg=dict(mode='whole'))
# base=dict(
#     type='EncoderDecoder',
#     pretrained='/data/checkpoints/convnext_base_22k_224.pth',
#     backbone=dict(
#         type='ConvNeXt',
#         in_chans=3,
#         depths=[3, 3, 27, 3],
#         dims=[128,256, 512, 1024],#
#         drop_path_rate=0.4,
#         layer_scale_init_value=1.0,
#         out_indices=[0, 1, 2, 3]),
#     decode_head=dict(
#         type='UPerHead',
#         in_channels=[128,256, 512, 1024],#
#         in_index=[0, 1, 2, 3],
#         pool_scales=(1, 2, 3, 6),
#         channels=512,
#         dropout_ratio=0.1,
#         num_classes=num_classes,
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         align_corners=False,
#         loss_decode=[
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),            
#             dict(type='LovaszLoss', loss_name='loss_lova',
#             reduction='none', loss_weight=0.5)]
#         ),
#     test_cfg=dict(mode='whole'))
model = [large,large]

size = 512
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
    samples_per_gpu=12,#16
    workers_per_gpu=8,#8
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        img_dir='/data/chusai_release/testA/images',
        img_suffix='.tif',
        seg_map_suffix='.png',
        k_fold_use=False,
        pipeline=test_pipeline))

