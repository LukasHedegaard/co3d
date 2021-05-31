#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified from https://github.com/facebookresearch/SlowFast

"""Caffe2 to PyTorch checkpoint name converting utility."""

import copy
import re
from collections import OrderedDict

import numpy as np
import torch

from ride.utils.logging import getLogger

logger = getLogger(__name__)


def get_name_convert_func():
    """
    Get the function to convert Caffe2 layer names to PyTorch layer names.
    Returns:
        (func): function to convert parameter name from Caffe2 format to PyTorch
        format.
    """
    pairs = [
        # ------------------------------------------------------------
        # 'nonlocal_conv3_1_theta_w' -> 's3.pathway0_nonlocal3.conv_g.weight'
        [r"^nonlocal_conv([0-9]+)_([0-9]+)_(.*)", r"s\1.pathway0_nonlocal\2_\3",],
        # 'theta' -> 'conv_theta'
        [r"^(.*)_nonlocal([0-9]+)_(theta)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'g' -> 'conv_g'
        [r"^(.*)_nonlocal([0-9]+)_(g)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'phi' -> 'conv_phi'
        [r"^(.*)_nonlocal([0-9]+)_(phi)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'out' -> 'conv_out'
        [r"^(.*)_nonlocal([0-9]+)_(out)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'nonlocal_conv4_5_bn_s' -> 's4.pathway0_nonlocal3.bn.weight'
        [r"^(.*)_nonlocal([0-9]+)_(bn)_(.*)", r"\1_nonlocal\2.\3.\4"],
        # ------------------------------------------------------------
        # 't_pool1_subsample_bn' -> 's1_fuse.conv_f2s.bn.running_mean'
        [r"^t_pool1_subsample_bn_(.*)", r"s1_fuse.bn.\1"],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [r"^t_pool1_subsample_(.*)", r"s1_fuse.conv_f2s.\1"],
        # 't_res4_5_branch2c_bn_subsample_bn_rm' -> 's4_fuse.conv_f2s.bias'
        [r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_bn_(.*)", r"s\1_fuse.bn.\3",],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_(.*)",
            r"s\1_fuse.conv_f2s.\3",
        ],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway0_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^res_conv1_bn_(.*)", r"s1.pathway0_stem.bn.\1"],
        # 'conv1_xy_w_momentum' -> 's1.pathway0_stem.conv_xy.'
        [r"^conv1_xy(.*)", r"s1.pathway0_stem.conv_xy\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway0_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^res_conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway1_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^t_res_conv1_bn_(.*)", r"s1.pathway1_stem.bn.\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^t_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway1_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^t_res_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # ------------------------------------------------------------
        # pred_ -> head.projection.
        [r"pred_(.*)", r"head.projection.\1"],
        # '.b_bn_fc' -> '.se.fc'
        [r"(.*)b_bn_fc(.*)", r"\1se.fc\2"],
        # conv_5 -> head.conv_5.
        [r"conv_5(.*)", r"head.conv_5\1"],
        # conv_5 -> head.conv_5.
        [r"lin_5(.*)", r"head.lin_5\1"],
        # '.bn_b' -> '.weight'
        [r"(.*)bn.b\Z", r"\1bn.bias"],
        # '.bn_s' -> '.weight'
        [r"(.*)bn.s\Z", r"\1bn.weight"],
        # '_bn_rm' -> '.running_mean'
        [r"(.*)bn.rm\Z", r"\1bn.running_mean"],
        # '_bn_riv' -> '.running_var'
        [r"(.*)bn.riv\Z", r"\1bn.running_var"],
        # '_b' -> '.bias'
        [r"(.*)[\._]b\Z", r"\1.bias"],
        # '_w' -> '.weight'
        [r"(.*)[\._]w\Z", r"\1.weight"],
    ]

    def convert_caffe2_name_to_pytorch(caffe2_layer_name):
        """
        Convert the caffe2_layer_name to pytorch format by apply the list of
        regular expressions.
        Args:
            caffe2_layer_name (str): caffe2 layer name.
        Returns:
            (str): pytorch layer name.
        """
        for source, dest in pairs:
            caffe2_layer_name = re.sub(source, dest, caffe2_layer_name)
        return caffe2_layer_name

    return convert_caffe2_name_to_pytorch


def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info("Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape))
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            logger.info(
                "Unexpected {}: {} -|> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def map_loaded_weights_from_caffe2(caffe2_checkpoint, model):
    state_dict = OrderedDict()
    name_convert_func = get_name_convert_func()
    for key in caffe2_checkpoint["blobs"].keys():
        converted_key = name_convert_func(key)
        converted_key = c2_normal_to_sub_bn(converted_key, model.state_dict())
        if converted_key in model.state_dict():
            c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
            model_blob_shape = model.state_dict()[converted_key].shape

            # expand shape dims if they differ (eg for converting linear to conv params)
            if len(c2_blob_shape) < len(model_blob_shape):
                c2_blob_shape += (1,) * (len(model_blob_shape) - len(c2_blob_shape))
                caffe2_checkpoint["blobs"][key] = np.reshape(
                    caffe2_checkpoint["blobs"][key], c2_blob_shape
                )
            # Load BN stats to Sub-BN.
            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                caffe2_checkpoint["blobs"][key] = np.concatenate(
                    [caffe2_checkpoint["blobs"][key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

            if c2_blob_shape == tuple(model_blob_shape):
                state_dict[converted_key] = torch.tensor(
                    caffe2_checkpoint["blobs"][key]
                ).clone()
                # logger.info(
                #     "{}: {} => {}: {}".format(
                #         key, c2_blob_shape, converted_key, tuple(model_blob_shape),
                #     )
                # )
            else:
                logger.warn(
                    "!! {}: {} does not match {}: {}".format(
                        key, c2_blob_shape, converted_key, tuple(model_blob_shape),
                    )
                )
        else:
            if not any(prefix in key for prefix in ["momentum", "lr", "model_iter"]):
                logger.warn(
                    "!! {}: can not be converted, got {}".format(key, converted_key)
                )
    return state_dict
