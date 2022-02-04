import torch
from ride.optimizers import SgdOneCycleOptimizer

from datasets import ActionRecognitionDatasets
from models.slowfast.video_model_builder import SlowFast
from ride import Configs, RideModule, TopKAccuracyMetric

from .model_loading import map_loaded_weights_from_caffe2


class SlowFastRide(
    RideModule,
    SlowFast,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    TopKAccuracyMetric(1, 3, 5),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()  # type: ignore
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
            name="slowfast_alpha",
            type=int,
            default=8,
            choices=[2, 4, 8, 16],
            strategy="uniform",
            description="Multiplier for frame rate in Fast pathway relative to Slow pathway.",
        )
        c.add(
            name="slowfast_beta_inv",
            type=int,
            default=8,
            choices=[2, 4, 8, 16],
            strategy="loguniform",
            description="Inverse multiplier for channel reduction ratio of Fast pathway relative to Slow pathway.",
        )
        c.add(
            name="slowfast_fusion_conv_channel_ratio",
            type=float,
            default=2,
            choices=[1.0, 3.0],
            strategy="uniform",
            description="Ratio of channel dimensions between the Slow and Fast pathways.",
        )
        c.add(
            name="slowfast_fusion_kernel_size",
            type=int,
            default=5,
            choices=[3, 7],
            strategy="uniform",
            description="Kernel dimension used for fusing information from Fast pathway to Slow pathway.",
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
        self.input_shape = (
            dim_in,
            self.hparams.temporal_window_size,
            self.hparams.image_size,
            self.hparams.image_size,
        )

        SlowFast.__init__(
            self,
            slowfast_alpha=self.hparams.slowfast_alpha,
            slowfast_beta_inv=self.hparams.slowfast_beta_inv,
            slowfast_fusion_conv_channel_ratio=self.hparams.slowfast_fusion_conv_channel_ratio,
            slowfast_fusion_kernel_size=self.hparams.slowfast_fusion_kernel_size,
            resnet_depth=self.hparams.resnet_depth,
            image_size=self.hparams.image_size,
            temporal_window_size=self.hparams.temporal_window_size,
            num_classes=self.dataloader.num_classes,  # from ActionRecognitionDatasets
            dropout_rate=self.hparams.dropout_rate,
            head_activation=self.hparams.head_activation,
            dim_in=[dim_in, dim_in],
        )

    def forward(self, x):
        # The original slowfast code was set up to use multiple paths
        # Package the input video appropriately
        x = pack_slowfast_pathways(x, alpha=self.hparams.slowfast_alpha)
        return SlowFast.forward(self, x)

    def map_loaded_weights(self, file, loaded_state_dict):
        # Called from FinetuneMixin
        if file[-4:] == ".pkl":
            return map_loaded_weights_from_caffe2(loaded_state_dict, self)
        return loaded_state_dict


def pack_slowfast_pathways(frames: torch.Tensor, alpha: float):
    """Packs slow and fast pathways for a video batch of shape (B,C,T,H,W)"""

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    t_idx = 2
    slow_pathway = torch.index_select(
        frames,
        t_idx,
        torch.linspace(
            start=0,
            end=frames.shape[t_idx] - 1,
            steps=frames.shape[t_idx] // alpha,
            dtype=torch.long,
            device=frames.device,
        ),
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list
