""" CoX3D main """
from ride import Main  # isort:skip

from ride.metrics import TopKAccuracyMetric
from ride.optimizers import SgdOneCycleOptimizer
from ride.utils.logging import getLogger

from datasets import ActionRecognitionDatasets
from models.common import Co3dBase, CoResNet
from models.slowfast.model_loading import map_loaded_weights_from_caffe2
from ride import Configs, RideModule

logger = getLogger("CoI3D")


class CoI3DRide(
    RideModule,
    Co3dBase,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    TopKAccuracyMetric(1),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="resnet_depth",
            type=int,
            default=50,
            strategy="constant",
            choices=[50, 101],
            description="Number of layers in ResNet.",
        )
        c.add(
            name="resnet_num_groups",
            type=int,
            default=1,
            strategy="constant",
            description="Number of groups.",
        )
        c.add(
            name="resnet_width_per_group",
            type=int,
            default=64,
            strategy="constant",
            description="Width of each group.",
        )
        c.add(
            name="resnet_dropout_rate",
            type=float,
            default=0.5,
            choices=[0.0, 1.0],
            strategy="uniform",
            description="Dropout rate before final projection in the backbone.",
        )
        c.add(
            name="resnet_head_act",
            type=str,
            default="softmax",
            choices=["softmax", "sigmoid"],
            strategy="choice",
            description="Activation layer for the output head.",
        )
        c.add(
            name="resnet_fc_std_init",
            type=float,
            default=0.01,
            strategy="choice",
            description="The std to initialize the fc layer(s).",
        )
        c.add(
            name="resnet_final_batchnorm_zero_init",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="If true, initialize the gamma of the final BN of each block to zero.",
        )
        return c

    def __init__(self, hparams):
        self.module = CoResNet(
            arch="i3d",
            dim_in=self.dim_in,
            image_size=self.hparams.image_size,
            temporal_window_size=self.hparams.temporal_window_size,
            num_classes=self.dataloader.num_classes,  # from ActionRecognitionDatasets,
            resnet_depth=self.hparams.resnet_depth,
            resnet_num_groups=self.hparams.resnet_num_groups,
            resnet_width_per_group=self.hparams.resnet_width_per_group,
            resnet_dropout_rate=self.hparams.resnet_dropout_rate,
            resnet_fc_std_init=self.hparams.resnet_fc_std_init,
            resnet_final_batchnorm_zero_init=self.hparams.resnet_final_batchnorm_zero_init,
            resnet_head_act=self.hparams.resnet_head_act,
            enable_detection=self.hparams.enable_detection,
            align_detection=self.hparams.align_detection,
            temporal_fill=self.hparams.co3d_temporal_fill,
        )

    def map_loaded_weights(self, file, loaded_state_dict):
        # Called from FinetuneMixin

        def convert_key(k):
            if not k:
                return k
            k = "module." + k
            k = k.replace(".s1.", ".0.")
            k = k.replace(".s2.", ".1.")
            k = k.replace(".s3.", ".2.")
            k = k.replace(".s4.", ".3.")
            k = k.replace(".s5.", ".4.")
            k = k.replace(".head.", ".5.")
            k = k.replace("pathway0_stem.", "")
            k = k.replace("a_bn", "norm_a")
            k = k.replace("b_bn", "norm_b")
            k = k.replace("c_bn", "norm_c")
            k = k.replace(".bn.", ".norm.")
            k = k.replace(".a.", ".conv_a.")
            k = k.replace(".b.", ".conv_b.")
            k = k.replace(".c.", ".conv_c.")
            k = k.replace("branch1_conv", "0.0.branch1")
            k = k.replace("branch1_norm", "0.0.branch1_bn")
            k = k.replace("branch2", "0.1.branch2")
            k = k.replace("pathway0_res0.branch1", "pathway0_res0.0.0.0.branch1")
            k = k.replace(
                "4.pathway0_res0.0.0.0.branch1", "4.pathway0_res0.0.0.branch1"
            )
            return k

        if file[-4:] == ".pkl":
            return map_loaded_weights_from_caffe2(loaded_state_dict, self, convert_key)
        return loaded_state_dict


if __name__ == "__main__":
    Main(CoI3DRide).argparse()
