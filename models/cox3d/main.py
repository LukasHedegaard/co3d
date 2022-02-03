""" CoX3D main """
from ride import Main  # isort:skip
from functools import partial

from ride import Configs, RideModule
from ride.metrics import MetricSelector, TopKAccuracyMetric
from ride.optimizers import SgdOneCycleOptimizer
from ride.utils.logging import getLogger

from datasets import ActionRecognitionDatasets
from metrics import CalibratedMeanAveragePrecisionMetric
from models.common import Co3dBase
from models.cox3d.modules.x3d import CoX3D

logger = getLogger("CoX3D")


class CoX3DRide(
    RideModule,
    Co3dBase,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    MetricSelector(
        kinetics400=TopKAccuracyMetric(1),
        tvseries=CalibratedMeanAveragePrecisionMetric,
        default_config="kinetics400",
    ),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="x3d_num_groups",
            type=int,
            default=1,
            strategy="constant",
            description="Number of groups.",
        )
        c.add(
            name="x3d_width_per_group",
            type=int,
            default=64,
            strategy="constant",
            description="Width of each group.",
        )
        c.add(
            name="x3d_width_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Width expansion factor.",
        )
        c.add(
            name="x3d_depth_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Depth expansion factor.",
        )
        c.add(
            name="x3d_bottleneck_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Bottleneck expansion factor for the 3x3x3 conv.",
        )
        c.add(
            name="x3d_conv1_dim",
            type=int,
            default=12,
            strategy="constant",
            description="Dimensions of the first 3x3 conv layer.",
        )
        c.add(
            name="x3d_conv5_dim",
            type=int,
            default=2048,
            strategy="constant",
            description="Dimensions of the last linear layer before classificaiton.",
        )
        c.add(
            name="x3d_use_channelwise_3x3x3",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="Whether to use channelwise (=depthwise) convolution in the center (3x3x3) convolution operation of the residual blocks.",
        )
        c.add(
            name="x3d_dropout_rate",
            type=float,
            default=0.5,
            choices=[0.0, 1.0],
            strategy="uniform",
            description="Dropout rate before final projection in the backbone.",
        )
        c.add(
            name="x3d_head_activation",
            type=str,
            default="softmax",
            choices=["softmax", "sigmoid"],
            strategy="choice",
            description="Activation layer for the output head.",
        )
        c.add(
            name="x3d_head_batchnorm",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Whether to use a BatchNorm layer before the classifier.",
        )
        c.add(
            name="x3d_fc_std_init",
            type=float,
            default=0.01,
            strategy="choice",
            description="The std to initialize the fc layer(s).",
        )
        c.add(
            name="x3d_final_batchnorm_zero_init",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="If true, initialize the gamma of the final BN of each block to zero.",
        )
        return c

    def __init__(self, hparams):
        self.module = CoX3D(
            self.dim_in,
            self.hparams.image_size,
            self.hparams.temporal_window_size,
            self.dataloader.num_classes,  # from ActionRecognitionDatasets
            self.hparams.x3d_conv1_dim,
            self.hparams.x3d_conv5_dim,
            self.hparams.x3d_num_groups,
            self.hparams.x3d_width_per_group,
            self.hparams.x3d_width_factor,
            self.hparams.x3d_depth_factor,
            self.hparams.x3d_bottleneck_factor,
            self.hparams.x3d_use_channelwise_3x3x3,
            self.hparams.x3d_dropout_rate,
            self.hparams.x3d_head_activation,
            self.hparams.x3d_head_batchnorm,
            self.hparams.x3d_fc_std_init,
            self.hparams.x3d_final_batchnorm_zero_init,
            self.hparams.co3d_temporal_fill,
            se_scope="frame",
        )

        # Ensure that state-dict is flattened
        self.load_state_dict = partial(self.module.load_state_dict, flatten=True)
        self.state_dict = partial(self.module.state_dict, flatten=True)


if __name__ == "__main__":
    Main(CoX3DRide).argparse()
