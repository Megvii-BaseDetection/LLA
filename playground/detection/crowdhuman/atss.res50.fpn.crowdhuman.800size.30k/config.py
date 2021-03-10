import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            NUM_CLASSES=1,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            REG_WEIGHT=2.0,
            NORM_SYNC=True,
        ),
        ATSS=dict(
            ANCHOR_SCALE=8,
            TOPK=9,
        ),
    ),
    DATASETS=dict(
        TRAIN=("crowdhuman_train",),
        TEST=("crowdhuman_val",),
    ),
    SOLVER=dict(
        IMS_PER_BATCH=16,
        BASE_LR=0.01,
        STEPS=(18750, 24735),
        MAX_ITER=28125,
    ),
    TEST=dict(
        DETECTIONS_PER_IMAGE=300,
        ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800,), max_size=1400, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=14000, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class ATSSConfig(FCOSConfig):
    def __init__(self):
        super(ATSSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ATSSConfig()
