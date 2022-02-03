from ride import Main  # isort:skip
from ride.optimizers import SgdOneCycleOptimizer

from datasets import ActionRecognitionDatasets
from models.slowfast.video_model_builder import ResNet
from ride import Configs, RideModule, TopKAccuracyMetric


class Slow(
    RideModule,
    ResNet,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    TopKAccuracyMetric(1, 3, 5),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="resnet_depth",
            type=int,
            default=50,
            strategy="choice",
            choices=[50, 101],
            description="Network depth (num layers) for ResNet.",
        )
        c.add(
            name="head_activation",
            type=str,
            default="softmax",
            strategy="choice",
            choices=["softmax", "sigmoid"],
            description="Activation function for prediction head.",
        )
        c.add(
            name="dropout_rate",
            type=float,
            default=0.5,
            choices=[0.0, 1.0],
            strategy="uniform",
            description="Dropout rate before final projection in the backbone.",
        )
        c.add(
            name="image_size",
            type=int,
            default=224,
            strategy="constant",
            description="Target image size.",
        )
        c.add(
            name="temporal_window_size",
            type=int,
            default=8,
            strategy="choice",
            description="Temporal window size for global average pool.",
        )
        # Detecion hparams not exposed
        return c

    def __init__(self, hparams):
        dim_in = 3
        self.hparams.frames_per_clip = self.hparams.temporal_window_size
        image_size = self.hparams.image_size
        self.input_shape = (
            dim_in,
            self.hparams.temporal_window_size,
            image_size,
            image_size,
        )

        num_block_temp_kernel = {
            50: [[3], [4], [6], [3]],
            101: [[3], [4], [23], [3]],
        }[hparams.resnet_depth]

        ResNet.__init__(
            self,
            model_arch="slow",
            resnet_depth=self.hparams.resnet_depth,
            image_size=image_size,
            temporal_window_size=self.hparams.temporal_window_size,
            num_classes=self.dataloader.num_classes,  # from ActionRecognitionDatasets
            dropout_rate=self.hparams.dropout_rate,
            head_activation=self.hparams.head_activation,
            dim_in=[dim_in],
            num_block_temp_kernel=num_block_temp_kernel,
        )

    def forward(self, x):
        # The original slowfast code was set up to use multiple paths, wrap the input
        x = [x]
        return ResNet.forward(self, x)

    def map_loaded_weights(self, finetune_from_weights, state_dict):
        # Map state_dict for "Slow" weights
        state_dict = {
            (
                k.replace(".res_blocks.", ".res")
                .replace("blocks.0", "s1.pathway0_stem")
                .replace("blocks.1.", "s2.pathway0_")
                .replace("blocks.2.", "s3.pathway0_")
                .replace("blocks.3.", "s4.pathway0_")
                .replace("blocks.4.", "s5.pathway0_")
                .replace("blocks.5.", "head.")
                .replace("norm", "bn")
                .replace("bn_a", "a_bn")
                .replace("bn_b", "b_bn")
                .replace("bn_c", "c_bn")
                .replace("_conv", "")
                .replace("conv_", "")
                .replace("proj", "projection")
            ): v
            for k, v in state_dict.items()
        }
        return state_dict


if __name__ == "__main__":
    Main(Slow).argparse()
