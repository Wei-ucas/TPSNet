find_unused_parameters=True
num_fiducial=8
tps_size=(0.25,1)
model = dict(
    type='TPSNet',
    from_p2=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='TPSHead',
        in_channels=256,
        num_sample=20,
        scales=(8, 16, 32),
        sample_size=(8,32),
        loss=dict(type='TPSLoss',gauss_center=True,
                  point_loss=True, with_BA=False,border_relax_thr=0.8),
        num_fiducial=num_fiducial,
        fiducial_dist="cross",
        num_convs=4),
    recog_head=dict(
        type='TPSRecogHead',
        recognizer=dict(
            type='ATTPredictor',
            in_channels=256,
            num_classes=98,
            max_seq_len=25,
            start_idx=96,
            padding_idx=97,
            weight=1.0
        ),
        with_coord=True,
        num_convs=2,
        num_sample_per_ins=2,
        image_size=(960,960),
        num_fiducial=num_fiducial,
        fiducial_dist='cross',
        sample_size=(8,32),
        add_gt =True,
        convertor=dict(type='AttnConvertor', lower=True, max_seq_len=25,
                       dict_list=[' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']),
        loss=dict(type='TFLoss',reduction='mean')
    )
)

train_cfg = None
test_cfg = None

dataset_type = 'IcdarE2EDataset'
data_root = 'data/my_synthtext/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile',
         ),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=960, scale=(3. / 4, 5. / 2)),
    dict(
        type='RandomCropPolyInstancesWithText',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.1),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=30,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=960, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='TPSTargets',
        level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
        tps_size=tps_size,
        with_area=True,
        gauss_center = True,
    ),
    dict(
        type='CustomFormatBundle',
        keys=['polygons_area','gt_texts','lv_tps_coeffs'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps', 'polygons_area', 'gt_texts','lv_tps_coeffs'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1800, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=[
            'data/totaltext/totaltext_train.json'],
        img_prefix=[
            'data/totaltext/imgs/training'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/totaltext/totaltext_test.json',
        img_prefix='data/totaltext/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/totaltext/totaltext_test.json',
        img_prefix='data/totaltext/imgs',
        pipeline=test_pipeline,))
evaluation = dict(interval=1000, metric='hmean-e2e',by_epoch=False)

# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.90, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step', step=[4000,8000],by_epoch=False,
                 warmup='linear',warmup_iters=500,warmup_ratio=0.001,
                 )
runner = {
            'type': 'IterBasedRunner',
            'max_iters': 10000
        }

checkpoint_config = dict(interval=1000, by_epoch=False)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
