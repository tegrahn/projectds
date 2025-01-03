# Contributed by Michail Bakalianos and Georgios Tsouderos
_base_ = ['../../_base_/default_runtime.py']

# DATASETS
clip_len = 50 #Number of frames in clip
data_set = 'hqfds' 

dataset_type = 'VideoDataset'
data_root = f'data/{data_set}/{clip_len}_train/'
data_root_val = f'data/{data_set}/{clip_len}_validation/'
ann_file_train = f'data/{data_set}/{clip_len}_train.txt'
ann_file_test = f'data/{data_set}/{clip_len}_validation.txt'
ann_file_val = f'data/{data_set}/{clip_len}_validation.txt'

# TRAIN

train_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=clip_len, frame_interval=1, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="RandomCrop", size=224),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=3,  # From VideoMAEv2 repo
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(video=data_root),
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        num_classes=2,
    ),
)

# VALIDATION

val_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames", clip_len=clip_len, frame_interval=1, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="CenterCrop", crop_size=224),  # From VideoMAEv2 repo
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

val_dataloader = dict(
    batch_size=3,  # From VideoMAEv2 repo
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(video=data_root_val),
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        num_classes=2,
    ),
)

# TEST

test_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames", clip_len=clip_len, frame_interval=1, num_clips=1, test_mode=True
    ),  # From VideoMAEv2 repo
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),  # From VideoMAEv2 repo
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

test_dataloader = dict(
    batch_size=16,  # From VideoMAEv2 repo
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(video=data_root_val),
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        num_classes=2,
    ),
)

# MODEL
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=clip_len,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=2,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# SCHEDULER

param_scheduler = [
    dict(
        type="LinearLR",
        by_epoch=True,
        convert_to_iter_based=True,
        start_factor=1e-3,
        end_factor=1,
        begin=0,
        end=5,
    ),  # From VideoMAEv2 repo - Warmup
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=1e-6,
        begin=5,
        end=35,
    ),
]

# OPTIMIZER
optim_wrapper = dict(
    type="AmpOptimWrapper",  # Automatic Mixed Precision may speed up trainig
    optimizer=dict(
        type="AdamW",  # From VideoMAEv2 repo
        lr=1e-3,  # From VideoMAEv2 repo
        weight_decay=0.1,  # From VideoMAEv2 repo
        betas=(0.9, 0.999),  # From VideoMAEv2 repo
    ),
    clip_grad=dict(max_norm=5, norm_type=2),  # From VideoMAEv2 repo
)

# TRAIN CONFIG
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=150, val_interval=1)

# VALIDATION CONFIG
val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='AccMetric'),  # Accuracy metric
        dict(type='ConfusionMatrix')         # Confusion matrix metric
    ]
)
val_cfg = dict(type="ValLoop")

# TEST CONFIG
test_evaluator = dict(
    type="AccMetric",
)
test_cfg = dict(type="TestLoop")

# PRETRAINED
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"
