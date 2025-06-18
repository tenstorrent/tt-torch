# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import tt_torch
import copy


class SimplifiedMnistModel(torch.nn.Module):
    def __init__(self):
        super(SimplifiedMnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # No dropout as random generation is not supported yet.
        # self.dropout1 = nn.Dropout(0)
        # self.dropout2 = nn.Dropout(0)
        self.fc1 = nn.Linear(36864, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # No max pool as max pool with indices is not supported yet
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


@pytest.mark.parametrize("num_steps", range(1, 5))
def test_minst_ttxla(num_steps):
    class TrainTestBase:
        from typing import Type

        def __init__(self, device, model, optimizer=None, loss_fn=None):
            self.device = device
            # dtype should be set from user code, not here
            self.model = copy.deepcopy(model).to(self.device).train()

            if optimizer is not None:
                self.optimizer = optimizer(self.model.parameters())
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters())

            self.loss_fn = loss_fn if loss_fn is not None else None

        def update_target(self, target_output):
            if target_output is not None:
                if isinstance(target_output, torch.Tensor):
                    self.target = target_output.to(self.device)
                elif isinstance(target_output, (list, tuple)):
                    self.target = type(target_output)(
                        t.to(self.device) for t in target_output
                    )
                elif isinstance(target_output, dict):
                    self.target = {
                        k: v.to(self.device) for k, v in target_output.items()
                    }
                else:
                    raise TypeError(f"Unsupported target type: {type(target_output)}")
            else:
                self.target = None

        def infer_targets_from_result(self, result):
            """
            infers dummy targets matching typical deep learning model output shapes.
            """

            def infer_single(t: torch.Tensor):
                shape = t.shape
                ndim = t.ndim
                device = t.device
                dtype = t.dtype

                # --- Tensor cases ---

                # (B, C, ...)
                if ndim >= 2:
                    B = shape[0]
                    ch = shape[1]
                    # multi-class segmentation (B, C>1, ...)
                    if ch > 1 and dtype.is_floating_point:
                        # e.g. [B, C, H, W] --> target: [B, H, W] long
                        return torch.randint(
                            0, ch, (B, *shape[2:]), device=device, dtype=torch.long
                        )
                    # binary segmentation (B, 1, ...)
                    elif ch == 1 and dtype.is_floating_point:
                        return torch.randint(
                            0, 2, (B, *shape[2:]), device=device, dtype=torch.long
                        )
                    # (B, 1) (float): binary clf
                    elif ndim == 2 and ch == 1 and dtype.is_floating_point:
                        return torch.randint(
                            0, 2, (B,), device=device, dtype=torch.long
                        )
                    # classification (B, C) float
                    elif ndim == 2 and ch > 1 and dtype.is_floating_point:
                        return torch.randint(
                            0, ch, (B,), device=device, dtype=torch.long
                        )
                    # (B, S, V): token classification
                    elif ndim == 3 and dtype.is_floating_point:
                        V = shape[2]
                        if V > 1:
                            return torch.randint(
                                0, V, (B, shape[1]), device=device, dtype=torch.long
                            )
                        else:  # V == 1, regression per step or binary token targets (rare)
                            return torch.randint(
                                0, 2, (B, shape[1]), device=device, dtype=torch.long
                            )
                    else:
                        return torch.rand_like(t)
                # (B,) (float): binary clf or regression
                elif ndim == 1 and dtype.is_floating_point:
                    return torch.randint(0, 2, shape, device=device, dtype=torch.long)
                # scalar
                elif ndim == 0 and dtype.is_floating_point:
                    return torch.randint(0, 2, shape, device=device, dtype=torch.long)
                # fallback: int type
                elif dtype in (
                    torch.int64,
                    torch.int32,
                    torch.int16,
                    torch.int8,
                    torch.uint8,
                ):
                    info = torch.iinfo(dtype)
                    return torch.randint(
                        info.min, info.max, shape, device=device, dtype=dtype
                    )
                else:
                    # regression
                    return torch.rand_like(t)

            # container
            if isinstance(result, torch.Tensor):
                return infer_single(result)
            elif isinstance(result, (list, tuple)):
                return type(result)(infer_single(t) for t in result)
            elif isinstance(result, dict):
                return {k: infer_single(v) for k, v in result.items()}
            else:
                raise TypeError(f"Unsupported result type: {type(result)}")

        def infer_loss_fn(self, output, target):
            # Extract dtype, shape
            out_dtype = output.dtype
            target_dtype = target.dtype
            out_shape = output.shape

            # === Multi-class classification: shape (N, C), output float, target int/long, target shape (N)
            # Classic CrossEntropyLoss case where output are logits and target are class indices
            if (
                out_dtype.is_floating_point
                and target_dtype == torch.long
                and output.ndim >= 2
                and out_shape[1] > 1
                and target.shape == output.shape[:1]
            ):
                return nn.NLLLoss()

            # === Binary classification (BCE or BCEWithLogits):
            # - output float, target usually 0/1, target is float or int, shapes can be (N,), (N,1), (N,H,W), etc.
            bce_like = False
            if out_dtype.is_floating_point:
                # BCE shape match (output/target same shape or can be broadcast)
                if output.shape == target.shape:
                    bce_like = (
                        target_dtype == torch.long or target_dtype.is_floating_point
                    )
                # (N,1) output (common for BCE), target may be (N,) or (N,1)
                if (
                    output.ndim == 2
                    and output.shape[1] == 1
                    and (
                        target.shape == (output.shape[0],)
                        or target.shape == output.shape
                    )
                ):
                    bce_like = True
                # (N,) output (squeezed logits)
                if output.ndim == 1 and target.shape == output.shape:
                    bce_like = True
            if bce_like:
                # BCEWithLogitsLoss expects target as float, so wrap with cast
                class _BCEWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.crit = nn.BCEWithLogitsLoss()

                    def forward(self, out, tgt):
                        return self.crit(out, target.float())

                return _BCEWrapper()

            # === Regression (MSE): output & target float, same shape
            if (
                out_dtype.is_floating_point
                and target_dtype.is_floating_point
                and output.shape == target.shape
            ):
                return nn.MSELoss()

            # === Fallback to CrossEntropy: output float (N,C), target long (N,)
            # If above conditions fail but this is still likely classification, return CrossEntropy
            if (
                out_dtype.is_floating_point
                and target_dtype == torch.long
                and output.ndim == 2
                and output.shape[1] > 1
                and target.shape == (output.shape[0],)
            ):
                return nn.CrossEntropyLoss()

            # === No matching loss found
            raise ValueError(
                f"Cannot infer a PyTorch loss function for:\n"
                f"  output: shape={output.shape}, dtype={output.dtype}\n"
                f"  target: shape={target.shape}, dtype={target.dtype}"
            )

        def run_forward(self):
            result = self.model(self.test_input)
            if self.target is None:
                self.target = self.infer_targets_from_result(result)
            if self.loss_fn is None:
                self.loss_fn = self.infer_loss_fn(result, self.target)

            return result

        def run_backward(self, result, target):
            loss = self.loss_fn(result, target)
            loss.backward()
            return loss

        def step_optimizer(self):
            xm.optimizer_step(self.optimizer)
            xm.mark_step()

        def run(self, test_input, target_output):
            # Run forward pass
            self.test_input = test_input.to(self.device)
            self.update_target(target_output)
            self.optimizer.zero_grad()
            result = self.run_forward()

            # Run backward pass
            loss = self.run_backward(result, self.target)

            # Step optimizer
            self.step_optimizer()
            return result, loss

        def get_parameters(self):
            return {
                name: param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

    class TrainTestCpu(TrainTestBase):
        def __init__(self, model, optimizer=None, loss_fn=None):
            super().__init__("cpu", model, optimizer, loss_fn)

    class TrainTestXLA(TrainTestBase):
        def __init__(self, device, model, optimizer=None, loss_fn=None):
            super().__init__(device, model, optimizer, loss_fn)

    model = SimplifiedMnistModel()
    model = model.to(torch.bfloat16)

    inputs_and_targets = []
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)
    data_iterator = iter(dataloader)
    for _ in range(num_steps):
        test_input, target = next(data_iterator)
        test_input = test_input.to(torch.bfloat16)
        inputs_and_targets.append((test_input, target))

    def train_xla(
        rank, model, test_inputs_and_targets, num_devices, optimizer=None, loss_fn=None
    ):
        import numpy as np

        device = xm.xla_device(rank)
        model = model.to(device)

        local_inputs_and_targets = test_inputs_and_targets[rank::num_devices]
        xla_instance = TrainTestXLA(device, model, optimizer=optimizer, loss_fn=loss_fn)
        for input, target in local_inputs_and_targets:
            xla_instance.run(input, target)

        save_path = f"xla_trained_model_params_{rank}.pth"
        torch.save(xla_instance.get_parameters(), save_path)

    # Run XLA test
    import torch_xla.distributed.xla_multiprocessing as xmp

    num_devices = len(xm.get_xla_supported_devices())

    if num_devices > 1:
        xmp.spawn(
            train_xla,
            args=(model, inputs_and_targets, num_devices, None, nn.NLLLoss()),
            nprocs=num_devices,
            start_method="spawn",
        )
    else:
        # If only one device, run directly without spawning
        train_xla(0, model, inputs_and_targets, num_devices, None, nn.NLLLoss())

    # Run CPU test
    cpu_test = TrainTestCpu(model, loss_fn=nn.NLLLoss())
    for i in range(num_steps):
        cpu_test.run(*inputs_and_targets[i])
    torch.save(cpu_test.get_parameters(), "cpu_trained_model_params.pth")

    # Verify training result
    xla_params_loaded = torch.load(
        f"xla_trained_model_params_{0}.pth", map_location="cpu"
    )
    cpu_params_loaded = torch.load("cpu_trained_model_params.pth", map_location="cpu")

    for param_cpu, param_xla in zip(
        cpu_params_loaded.values(), xla_params_loaded.values()
    ):
        # Convert XLA parameter to CPU for comparison
        param_xla_cpu = param_xla.to("cpu")
        torch.testing.assert_close(param_cpu, param_xla_cpu, rtol=1e-2, atol=1e-2)
