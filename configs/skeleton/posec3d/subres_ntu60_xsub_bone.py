_base_ = '../../_base_/default_runtime.py'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='sp_ResNet3dSlowOnly',
        depth=50,
        pretrained='/work/gyz_Projects/mmaction222/mmaction2/checkpoints/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
        conv_cfg = dict({'type': "subm"}),
        ),
    cls_head=dict(
        type='I3DHead',
        #loss_cls= dict(type='BCELossWithLogits'),
        in_channels=128,
        num_classes=60,
        dropout_ratio=0.5,
        average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file = '/work/gyz_Projects/mmaction2_effi/mmaction21/data/ntu60_2d.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
sig = 0.6
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    # dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=sig,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56)),
    dict(type='CenterCrop', crop_size=48),
    dict(
        type='GeneratePoseTarget',
        sigma=sig,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=sig,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='xsub_train',
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=35,
        # begin=10,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]

# load_from = "work_dirs/spfocal_ntu60_xsub_1/epoch_9.pth"
# resume = 'work_dirs/spfocal_ntu60_xsub/epoch_21.pth'
find_unused_parameters = True
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9,
                    weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2))
visualizer = dict(
    type='ActionVisualizer',
    vis_backends = [dict(type='LocalVisBackend'),
                    dict(type='TensorboardVisBackend')]
)
