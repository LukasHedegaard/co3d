import torch
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm3d


def convert_batchnorm3d(instance: BatchNorm3d, *args, **kwargs):
    m = BatchNorm2d(
        instance.num_features,
        instance.eps,
        instance.momentum,
        instance.affine,
        instance.track_running_stats,
    )
    with torch.no_grad():
        m.running_mean.copy_(instance.running_mean)
        m.running_var.copy_(instance.running_var)
    return m


# # Previous effort at implementation, which turned out to be obsolete because BatchNorm3d and BatchNorm2d behave similarly in eval mode
# import itertools
# from functools import partial
# from typing import List, Tuple
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Module
# from torch.nn.parameter import Parameter

# # (running_mean, running_sum_of_squares, circular_buffer_index)
# _State = Tuple[Tensor, Tensor, int]


# class RBatchNorm3d(Module):
#     def __init__(
#         self,
#         window_size: int,
#         num_features: int,
#         eps: float = 1e-5,
#         momentum: float = 0.1,
#         affine: bool = True,
#         track_running_stats: bool = True,
#         temporal_fill="replicate",
#     ):
#         super(RBatchNorm3d, self).__init__()
#         assert window_size > 0
#         self.window_size = window_size
#         self.num_features = num_features
#         self.eps = eps
#         # Approximately compensate for window size
#         self.momentum = momentum ** (1.0 / window_size)
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         assert temporal_fill in {"replicate", "unity"}
#         self.temporal_fill = temporal_fill

#         if self.affine:
#             self.weight = Parameter(torch.ones(num_features, dtype=torch.float))
#             self.bias = Parameter(torch.zeros(num_features, dtype=torch.float))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)

#         self.register_buffer("running_mean", torch.zeros(window_size, num_features))
#         self.register_buffer("running_var", torch.ones(window_size, num_features))
#         self.register_buffer(
#             "running_count", torch.zeros(num_features, dtype=torch.long)
#         )
#         self.register_buffer("circular_buffer_index", torch.tensor(0, dtype=torch.long))

#         self.init_state = partial(
#             _init_state, state_len=self.window_size, temporal_fill=temporal_fill,
#         )

#     def forward(self, input: Tensor) -> Tensor:
#         state = (
#             self.running_mean,
#             self.running_var,
#             self.running_count,
#             self.circular_buffer_index,
#         )
#         output, new_state = self._forward(input, state)
#         (
#             self.running_mean,
#             self.running_var,
#             self.running_count,
#             self.circular_buffer_index,
#         ) = new_state
#         return output

#     def _forward(self, input: Tensor, prev_state: _State) -> Tuple[Tensor, _State]:
#         assert len(input.shape) == 4, "Only a single frame should be passed at a time."
#         B, C, H, W = input.shape

#         # Compute channel-wise mean and var for input
#         input_mean = input.mean(dim=[0, 2, 3])
#         input_var = input.var(unbiased=False, dim=[0, 2, 3])
#         input_count = torch.tensor(B * H * W)

#         # Prepare previous state
#         if prev_state is None:
#             prev_state = self.init_state(input_mean, input_var, input_count)
#         running_mean, running_var, running_count, index = prev_state
#         if running_count.sum() == 0:
#             running_count = input_count * torch.ones(self.window_size, dtype=torch.long)

#         running_mean[index] = input_mean
#         running_var[index] = input_var
#         running_count[index] = input_count

#         total_mean = running_mean.mean(dim=0)
#         total_count = running_count.sum(dim=0)
#         total_var_biased = (
#             torch.sum((running_var + running_mean ** 2) * running_count[:, None], dim=0)
#             / total_count  # (total_count + 1) for unbiased
#             - total_mean ** 2
#         )

#         # Compute batchnorm output
#         output = F.batch_norm(
#             input,
#             total_mean if not self.training or self.track_running_stats else None,
#             total_var_biased if not self.training or self.track_running_stats else None,
#             self.weight,
#             self.bias,
#             self.training,
#             self.momentum,
#             self.eps,
#         )
#         output = (input - total_mean) / (torch.sqrt(total_var_biased + self.eps))
#         output = (
#             output * self.weight[None, :, None, None, None]
#             + self.bias[None, :, None, None, None]
#         )

#         # Update next state
#         new_index = (index + 1) % self.window_size  # Circular buffer
#         next_state = (running_mean, running_var, running_count, new_index)

#         return output, next_state

#     def forward_regular(self, input: Tensor):
#         assert (
#             len(input.shape) == 5
#         ), "A tensor of size B,C,T,H,W should be passed as input."
#         T = input.shape[2]
#         outs = []
#         for t in range(T):
#             o = self.forward(input[:, :, t])
#             outs.append(o)
#         if len(outs) > 0:
#             outs = torch.stack(outs, dim=2)
#         else:
#             outs = torch.tensor([])
#         return outs

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#         copy_and_duplicate_keys={"running_mean", "running_var"},
#     ):
#         r"""Copies parameters and buffers from :attr:`state_dict` into only
#         this module, but not its descendants. This is called on every submodule
#         in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
#         module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
#         For state dicts without metadata, :attr:`local_metadata` is empty.
#         Subclasses can achieve class-specific backward compatible loading using
#         the version number at `local_metadata.get("version", None)`.

#         .. note::
#             :attr:`state_dict` is not the same object as the input
#             :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
#             it can be modified.

#         Arguments:
#             state_dict (dict): a dict containing parameters and
#                 persistent buffers.
#             prefix (str): the prefix for parameters and buffers used in this
#                 module
#             local_metadata (dict): a dict containing the metadata for this module.
#                 See
#             strict (bool): whether to strictly enforce that the keys in
#                 :attr:`state_dict` with :attr:`prefix` match the names of
#                 parameters and buffers in this module
#             missing_keys (list of str): if ``strict=True``, add missing keys to
#                 this list
#             unexpected_keys (list of str): if ``strict=True``, add unexpected
#                 keys to this list
#             error_msgs (list of str): error messages should be added to this
#                 list, and will be reported together in
#                 :meth:`~torch.nn.Module.load_state_dict`
#         """
#         for hook in self._load_state_dict_pre_hooks.values():
#             hook(
#                 state_dict,
#                 prefix,
#                 local_metadata,
#                 strict,
#                 missing_keys,
#                 unexpected_keys,
#                 error_msgs,
#             )

#         persistent_buffers = {
#             k: v
#             for k, v in self._buffers.items()
#             if k not in self._non_persistent_buffers_set
#         }
#         local_name_params = itertools.chain(
#             self._parameters.items(), persistent_buffers.items()
#         )
#         local_state = {k: v for k, v in local_name_params if v is not None}

#         for name, param in local_state.items():
#             key = prefix + name
#             if key in state_dict:
#                 input_param = state_dict[key]

#                 # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
#                 if len(param.shape) == 0 and len(input_param.shape) == 1:
#                     input_param = input_param[0]

#                 # Modification from the torch lib implementation: Check for broadcastability instead of strict shape equality
#                 if not is_broadcastable(input_param, param):
#                     # local shape should match the one in checkpoint
#                     error_msgs.append(
#                         "size mismatch for {}: copying a param with shape {} from checkpoint, "
#                         "the shape in current model is {}.".format(
#                             key, input_param.shape, param.shape
#                         )
#                     )
#                     continue

#                 try:
#                     with torch.no_grad():
#                         param.copy_(input_param)
#                 except Exception as ex:
#                     error_msgs.append(
#                         'While copying the parameter named "{}", '
#                         "whose dimensions in the model are {} and "
#                         "whose dimensions in the checkpoint are {}, "
#                         "an exception occurred : {}.".format(
#                             key, param.size(), input_param.size(), ex.args
#                         )
#                     )
#             elif strict:
#                 missing_keys.append(key)

#         if strict:
#             for key in state_dict.keys():
#                 if key.startswith(prefix):
#                     input_name = key[len(prefix) :]
#                     input_name = input_name.split(".", 1)[
#                         0
#                     ]  # get the name of param/buffer/child
#                     if (
#                         input_name not in self._modules
#                         and input_name not in local_state
#                     ):
#                         unexpected_keys.append(key)


# def _init_state(
#     mean: Tensor, var: Tensor, numel: int, state_len: int, temporal_fill="replicate",
# ) -> List[Tensor]:
#     assert state_len > 0
#     assert temporal_fill in {"unit", "replicate"}
#     running_mean = {
#         "zeros": torch.zeros(state_len, *list(mean.shape), dtype="float"),
#         "replicate": mean.repeat(state_len, 1),
#     }[temporal_fill]
#     running_var = {
#         "zeros": numel * torch.ones(state_len, *list(var.shape), dtype="float"),
#         "replicate": var.repeat(state_len, 1),
#     }[temporal_fill]
#     running_count = numel * torch.ones(state_len, dtype=torch.long)
#     index = 0
#     return (running_mean, running_var, running_count, index)


# def is_broadcastable(tensor1: torch.Tensor, tensor2: torch.Tensor):
#     shape1, shape2 = list(tensor1.shape), list(tensor2.shape)
#     max_len = max(len(shape1), len(shape2))

#     # Forward direction
#     forward_ok = True
#     for i in range(max_len):
#         s1 = shape1[i] if len(shape1) > i else None
#         s2 = shape2[i] if len(shape2) > i else None
#         if not ((s1 == s2) or (s1 in {None, 1}) or (s2 in {None, 1})):
#             forward_ok = False
#             break

#     # Backwards direction
#     backward_ok = True
#     for i in range(1, max_len + 1):
#         s1 = shape1[-i] if len(shape1) >= i else None
#         s2 = shape2[-i] if len(shape2) >= i else None
#         if not ((s1 == s2) or (s1 in {None, 1}) or (s2 in {None, 1})):
#             backward_ok = False
#             break

#     return forward_ok or backward_ok
