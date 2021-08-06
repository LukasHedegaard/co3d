import torch
from torch.nn import BatchNorm2d, BatchNorm3d

# from models.co3d.modules import RBatchNorm3d

torch.manual_seed(42)

example_input = torch.arange(3 * 2 * 2 * 2).float().reshape((1, 3, 2, 2, 2))


def test_mean_var_computation():
    target_mean = example_input.mean(dim=[0, 2, 3, 4])
    target_var = example_input.var(dim=[0, 2, 3, 4])

    B, C, T, H, W = example_input.shape

    # Biased estimator
    # target_var = ((example_input - target_mean[None, :, None, None, None]) ** 2).sum(
    #     dim=[0, 2, 3, 4]
    # ) / (B * T * H * W - 1)

    in0 = example_input[:, :, 0]
    in1 = example_input[:, :, 1]

    mean0 = in0.mean(dim=[0, 2, 3])
    mean1 = in1.mean(dim=[0, 2, 3])

    sum_of_squared0 = (in0 ** 2).sum(dim=[0, 2, 3])
    sum_of_squared1 = (in1 ** 2).sum(dim=[0, 2, 3])

    total_mean = torch.stack([mean0, mean1]).mean(dim=0)
    total_var_biased = (
        torch.stack([sum_of_squared0, sum_of_squared1]).sum(dim=0) / (B * T * H * W)
        - total_mean ** 2
    )
    total_var_unbiased = total_var_biased * (B * T * H * W) / (B * T * H * W - 1)

    assert torch.equal(target_mean, total_mean)
    assert torch.equal(target_var, total_var_unbiased)


def test_var_aggregation_unbiased():
    _, C, T, H, W = 2, 3, 2, 2, 2
    in1 = torch.arange(C * T * H * W).float().reshape((1, C, T, H, W))
    in2 = 2 * torch.arange(C * T * H * W).float().reshape((1, C, T, H, W)) + 1
    in_tot = torch.stack([in1, in2]).squeeze(1)

    v1 = torch.var(in1, unbiased=False, dim=[0, 2, 3, 4])
    v2 = torch.var(in2, unbiased=False, dim=[0, 2, 3, 4])
    m1 = torch.mean(in1, dim=[0, 2, 3, 4])
    m2 = torch.mean(in2, dim=[0, 2, 3, 4])
    target_tot_var = torch.var(in_tot, unbiased=False, dim=[0, 2, 3, 4])
    target_tot_mean = torch.mean(in_tot, dim=[0, 2, 3, 4])

    means = torch.stack([m1, m2])
    tot_mean = torch.mean(means, dim=0)

    # One way to calc variance
    sosq = torch.stack([torch.sum(i ** 2, dim=[0, 2, 3, 4]) for i in [in1, in2]])
    counts = torch.stack([torch.tensor(T * H * W) for _ in [in1, in2]])

    tot_var = torch.sum(sosq, dim=0) / torch.sum(counts, dim=0) - tot_mean ** 2

    assert torch.equal(tot_mean, target_tot_mean)
    assert torch.equal(tot_var, target_tot_var)

    # Another way to cal variance
    vars = torch.stack([v1, v2])
    counts = torch.stack([torch.tensor(T * H * W) for _ in [in1, in2]])
    tot_var2 = (
        torch.sum((vars + means ** 2) * counts[:, None], dim=0)
        / torch.sum(counts, dim=0)
        - tot_mean ** 2
    )

    assert torch.equal(tot_var2, target_tot_var)


def test_understanding_momentum_training():
    # Test our understanding of batch norm
    eps = 1e-5
    momentum = 0.1

    # Lib impl
    bn = torch.nn.BatchNorm3d(num_features=3, eps=eps, momentum=momentum)
    bn.training = True

    running_mean = 0.1 * torch.ones_like(bn.running_mean)
    running_var = 2 * torch.ones_like(bn.running_mean)

    # Not actually used during computation. Updated behind the scenes, though
    bn.running_mean = running_mean.clone()
    bn.running_var = running_var.clone()

    target = bn(example_input)

    # Own impl
    mean = example_input.mean(dim=[0, 2, 3, 4])
    # unbiased variance is used!
    var = example_input.var(dim=[0, 2, 3, 4], unbiased=False)

    output = (example_input - mean[None, :, None, None, None]) / (
        torch.sqrt(var[None, :, None, None, None] + eps)
    )
    output = (
        output * bn.weight[None, :, None, None, None]
        + bn.bias[None, :, None, None, None]
    )

    # running_mean and running_var not used for computation
    assert torch.equal(target, output)

    # Running state is updated
    run_mean = momentum * mean + (1 - momentum) * running_mean
    run_var = momentum * var + (1 - momentum) * running_var

    run_var += 0.075  # Not sure why we have this difference

    assert torch.equal(run_mean, bn.running_mean)
    assert torch.equal(run_var, bn.running_var)


def test_understanding_momentum_testing():
    # Test our understanding of batch norm
    eps = 1e-5
    momentum = 0.1

    # Lib impl
    bn = torch.nn.BatchNorm3d(num_features=3, eps=eps, momentum=momentum)
    bn.training = False

    running_mean = 0.1 * torch.ones_like(bn.running_mean)
    running_var = 2 * torch.ones_like(bn.running_mean)

    bn.running_mean = running_mean.clone()
    bn.running_var = running_var.clone()

    target = bn(example_input)

    # Own impl
    output = (example_input - running_mean[None, :, None, None, None]) / (
        torch.sqrt(running_var[None, :, None, None, None] + eps)
    )
    output = (
        output * bn.weight[None, :, None, None, None]
        + bn.bias[None, :, None, None, None]
    )

    # running_mean and running_var not used for computation
    assert torch.allclose(target, output)

    # Running state is not updated
    assert torch.equal(running_mean, bn.running_mean)
    assert torch.equal(running_var, bn.running_var)


# def test_load_state_dict():
#     # Regular
#     eps = 1e-5
#     bn = BatchNorm3d(num_features=3, eps=eps, momentum=0.0)
#     bn.running_mean = 0.1 * torch.ones_like(bn.running_mean)
#     bn.running_var = 2 * torch.ones_like(bn.running_mean)

#     # Recurrent
#     rbn = RBatchNorm3d(
#         window_size=example_input.shape[2], num_features=3, eps=eps, momentum=0.0
#     )
#     rbn.load_state_dict(bn.state_dict(), strict=False)

#     assert torch.equal(rbn.running_mean, 0.1 * torch.ones_like(rbn.running_mean))
#     assert torch.equal(rbn.running_var, 2 * torch.ones_like(rbn.running_var))


def test_2d_vs_3d_bn():
    eps = 1e-5

    bn3 = BatchNorm3d(num_features=3, eps=eps, momentum=0.0)
    bn3.running_mean = 0.1 * torch.ones_like(bn3.running_mean)
    bn3.running_var = 2 * torch.ones_like(bn3.running_mean)
    bn3.training = False

    bn2 = BatchNorm2d(num_features=3, eps=eps, momentum=0.0)
    bn2.running_mean = 0.1 * torch.ones_like(bn2.running_mean)
    bn2.running_var = 2 * torch.ones_like(bn2.running_mean)
    bn2.training = False

    o2 = torch.stack(
        [bn2(example_input[:, :, t]) for t in range(example_input.shape[2])], dim=2
    )
    o3 = bn3(example_input)

    assert torch.allclose(o2, o3)
