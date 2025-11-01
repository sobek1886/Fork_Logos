work_dir = 'data/experiments/logos_autsl_wlasl_init'

batch_size = 16
num_workers = 8
num_classes_AUTSL = 226
num_classes_LOGOS = 2005
num_classes_WLASL = 2000
no_event_label_LOGOS = 0

dataset_len = 160000 + 32500 + 18205
epoch_scale = 160000/dataset_len

model_wrapper_cfg = {
    'find_unused_parameters': True,
}

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MViT',
        arch='small',
        drop_path_rate=0.1,
        dim_mul_in_attention=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/model/logos_stage1_model.pth',
            prefix='backbone',
        ),
    ),
    data_preprocessor=dict(
        type='GateActionDataPreprocessor',
        preprocessors={
            "WLASL": dict(
                type='ActionDataPreprocessor',
                mean=[140.99762122, 129.92701646, 125.25081198],
                std=[62.07248248, 62.94645644, 61.42221137],
                blending=dict(
                    type='RandomBatchAugment',
                    augments=[
                        dict(type='MixupBlending', alpha=0.8, num_classes=num_classes_WLASL),
                        dict(type='CutmixBlending', alpha=1, num_classes=num_classes_WLASL),
                    ],
                ),
                format_shape='NCTHW',
            ),
            "LOGOS": dict(
                type='ActionDataPreprocessor',
                mean=[140.99762122, 129.92701646, 125.25081198],
                std=[62.07248248, 62.94645644, 61.42221137],
                blending=dict(
                    type='RandomBatchAugment',
                    augments=[
                        dict(type='MixupBlending', alpha=0.8, num_classes=num_classes_LOGOS),
                        dict(type='CutmixBlending', alpha=1, num_classes=num_classes_LOGOS),
                    ],
                ),
                format_shape='NCTHW',
            ),
            "AUTSL": dict(
                type='ActionDataPreprocessor',
                mean=[140.99762122, 129.92701646, 125.25081198],
                std=[62.07248248, 62.94645644, 61.42221137],
                blending=dict(
                    type='RandomBatchAugment',
                    augments=[
                        dict(type='MixupBlending', alpha=0.8, num_classes=num_classes_AUTSL),
                        dict(type='CutmixBlending', alpha=1, num_classes=num_classes_AUTSL),
                    ],
                ),
                format_shape='NCTHW',
            ),
        }
    ),
    cls_head=dict(
        type='GateHead',
        heads={
            "WLASL": dict(
                type='MViTRegHead',
                loss_cls=dict(type='CrossEntropyLoss'),
                loss_bounds=dict(type='MSELoss', loss_weight=2.5),
                in_channels=768,
                num_classes=num_classes_WLASL,
                label_smooth_eps=0.1,
                average_clips='prob',
                dropout_ratio=0.,
                init_scale=0.001,
            ),
            "LOGOS": dict(
                type='MViTRegHead',
                loss_cls=dict(type='CrossEntropyLoss'),
                loss_bounds=dict(type='MSELoss', loss_weight=2.5),
                in_channels=768,
                num_classes=num_classes_LOGOS,
                label_smooth_eps=0.1,
                average_clips='prob',
                dropout_ratio=0.,
                init_scale=0.001,
            ),
            "AUTSL": dict(
                type='MViTRegHead',
                loss_cls=dict(type='CrossEntropyLoss'),
                loss_bounds=dict(type='MSELoss', loss_weight=2.5),
                in_channels=768,
                num_classes=num_classes_AUTSL,
                label_smooth_eps=0.1,
                average_clips='prob',
                dropout_ratio=0.,
                init_scale=0.001,
            ),
        }
    )
)

default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # early_stopping=dict(type='EarlyStoppingHook', monitor='acc/top1', patience=7, min_delta=0.003),
    checkpoint=dict(type='CheckpointHook', interval=max(1, int(1*epoch_scale)), save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
)
env_cfg = dict(
    cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl')
)
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
)
log_level = 'INFO'
load_from = None

# dataset settings
data_root_LOGOS = 'data/Logos'
ann_file_train_LOGOS = 'data/Logos/logos_train_v5_jpg95_300.hdf5video'
ann_file_test_LOGOS = 'data/Logos/logos_test_v5_jpg95_300.hdf5video'

data_root_WLASL = 'data/WLASL'
ann_file_train_WLASL = 'data/WLASL/wlasl_train_jpg95_300.hdf5video'
ann_file_test_WLASL  = 'data/WLASL/wlasl_test_jpg95_300.hdf5video'

data_root_AUTSL = 'data/AUTSL'
ann_file_train_AUTSL = 'data/AUTSL/autsl_train_jpg95_300.hdf5video'
ann_file_test_AUTSL  = 'data/AUTSL/autsl_test_jpg95_300.hdf5video'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='Hdf5VideoInit'),
    dict(
        type='SampleFrames',
        delta_left=[-5, 5],
        delta_right=[-5, 5],
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        out_of_bound_opt='repeat_last',
    ),
    dict(
        type='UniformAug',
        num_ops=1,
        augs=[
            dict(type='RandomDropConnector', drop_ratio=0.1, p=0.5),
            dict(type='RandomAddConnector', add_ratio=0.3, p=0.25),
            dict(type='SpeedChangeConnector', speed_change=2, p=0.25),
            dict(type='SpeedChangeConnector', speed_change=0.5, p=0.25)
        ],
    ),
    dict(type='Hdf5VideoDecode'),
    dict(type='Resize', scale=(300, 300)),
    dict(type='ColorJitter', brightness=0.1, contrast=0.005, saturation=0, hue=0.05, p=0.5),
    dict(type='RandomNoise', mode='s&p', amount=[0.001, 0.005], p=0.5),
    dict(type='Sharpness', sharpness_factor=[0.5, 2], p=0.35),
    dict(type='Flip', blacklisted_labels=[233, 234, 371, 372, 452, 969, 1473, 1537, 1606, 1657,], flip_ratio=0.5),
    dict(type='RandomErasing', erase_prob=0.25),
    dict(type='AlbumentationAugs', augs=dict(type='ImageCompression', quality_lower=80, quality_upper=100, p=0.15)),
    dict(type='AlbumentationAugs', augs=dict(type='Downscale', scale_min=0.4, scale_max=0.8, interpolation=1, p=0.15)),
    dict(type='SquarePadding', out_shape=(300, 300)),
    dict(type='RandomCrop', size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]

val_pipeline = [
    dict(type='Hdf5VideoInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        out_of_bound_opt='repeat_last',
    ),
    dict(type='Hdf5VideoDecode'),
    dict(type='Resize', scale=(300, 300)),
    dict(type='SquarePadding', out_shape=(300, 300)),
    dict(type='CenterCrop', crop_size=(224, 224)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]

test_pipeline = val_pipeline

limit_samples_number_to = 0

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True and num_workers > 0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MergeDataset',
        datasets = {
            "WLASL": dict(
                type='Hdf5VideoDataset',
                no_event_label=None,
                start_and_end_from_labels=True,
                ann_file=ann_file_train_WLASL,
                data_prefix=dict(video=data_root_WLASL),
                limit_samples_number_to=limit_samples_number_to,
                pipeline=train_pipeline,
            ),
            "LOGOS": dict(
                type='Hdf5VideoDataset',
                no_event_label=no_event_label_LOGOS,
                start_and_end_from_labels=True,
                ann_file=ann_file_train_LOGOS,
                data_prefix=dict(video=data_root_LOGOS),
                limit_samples_number_to=limit_samples_number_to,
                pipeline=train_pipeline,
            ),
            "AUTSL": dict(
                type='Hdf5VideoDataset',
                no_event_label=None,
                start_and_end_from_labels=True,
                ann_file=ann_file_train_AUTSL,
                data_prefix=dict(video=data_root_AUTSL),
                limit_samples_number_to=limit_samples_number_to,
                pipeline=train_pipeline,
            ),
        }
    )
)
val_dataloader_LOGOS = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True and num_workers > 0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MergeDataset',
        datasets = {
            "LOGOS": dict(
                type='Hdf5VideoDataset',
                no_event_label=no_event_label_LOGOS,
                start_and_end_from_labels=True,
                ann_file=ann_file_test_LOGOS,
                data_prefix=dict(video=data_root_LOGOS),
                pipeline=val_pipeline,
                limit_samples_number_to=limit_samples_number_to,
                test_mode=True,
            ),
        }
    ),
)

val_dataloader_WLASL = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True and num_workers > 0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MergeDataset',
        datasets = {
            "WLASL": dict(
                type='Hdf5VideoDataset',
                no_event_label=None,
                start_and_end_from_labels=True,
                ann_file=ann_file_test_WLASL,
                data_prefix=dict(video=data_root_WLASL),
                pipeline=val_pipeline,
                limit_samples_number_to=limit_samples_number_to,
                test_mode=True,
            ),
        }
    ),
)

val_dataloader_AUTSL = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True and num_workers > 0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MergeDataset',
        datasets = {
            "AUTSL": dict(
                type='Hdf5VideoDataset',
                no_event_label=None,
                start_and_end_from_labels=True,
                ann_file=ann_file_test_AUTSL,
                data_prefix=dict(video=data_root_AUTSL),
                pipeline=val_pipeline,
                limit_samples_number_to=limit_samples_number_to,
                test_mode=True,
            ),
        }
    ),
)

val_dataloader = val_dataloader_WLASL
test_dataloader = val_dataloader

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=int(50*epoch_scale),
    val_begin=1,
    val_interval=max(1, int(1*epoch_scale)),
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 4.8e-3
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_rate=0.75,
        decay_type='layer_wise',
        num_layers=16
    ),
    clip_grad=dict(max_norm=5, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1 / 600,
        by_epoch=True,
        begin=0,
        end=int(5*epoch_scale),
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=int(35*epoch_scale),
        eta_min_ratio=1 / 60,
        by_epoch=True,
        begin=int(5*epoch_scale),
        end=int(40*epoch_scale),
        convert_to_iter_based=True,
    )
]
auto_scale_lr = dict(enable=False, base_batch_size=64)
launcher = 'pytorch'
randomness = dict(seed=42, diff_rank_seed=False, deterministic=False)
