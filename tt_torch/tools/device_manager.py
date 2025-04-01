# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
import warnings


class DeviceManager:
    _devices = set()
    _parent = None
    _num_available = tt_mlir.get_num_available_devices()

    @staticmethod
    def _get_parent_mesh_options(
        device_ids=None,
        num_hw_cqs=None,
        enable_async_ttnn=None,
        enable_program_cache=None,
        l1_small_size=None,
        dispatch_core_type=None,
    ):
        options = tt_mlir.MeshDeviceOptions()
        if device_ids is not None:
            options.device_ids = device_ids
        if num_hw_cqs is not None:
            options.num_hw_cqs = num_hw_cqs
        if enable_async_ttnn is not None:
            options.enable_async_ttnn = enable_async_ttnn
        if enable_program_cache is not None:
            options.enable_program_cache = enable_program_cache
        if l1_small_size is not None:
            options.l1_small_size = l1_small_size
        if dispatch_core_type is not None:
            options.dispatch_core_type = dispatch_core_type
        return options

    @classmethod
    def get_num_available_devices(self):
        """Get the number of available devices."""
        return self._num_available

    @classmethod
    def get_available_devices(
        self,
        mesh_shape=None,
        device_ids=None,
        num_hw_cqs=None,
        enable_async_ttnn=None,
        enable_program_cache=None,
        l1_small_size=None,
        dispatch_core_type=None,
    ):
        if mesh_shape is None:
            mesh_shape = [1, self._num_available]
        if len(self._devices) > 0:
            assert self._parent is not None, "Parent device is not set."
            return list(self._devices)

        assert len(mesh_shape) == 2, "Mesh shape must be a list of two integers."
        assert (
            mesh_shape[0] * mesh_shape[1] <= self._num_available
        ), "Mesh shape exceeds available devices."

        options = self._get_parent_mesh_options(
            device_ids,
            num_hw_cqs,
            enable_async_ttnn,
            enable_program_cache,
            l1_small_size,
            dispatch_core_type,
        )

        # Secure a parent mesh device
        self._parent = tt_mlir.open_mesh_device(
            mesh_shape=mesh_shape,
            options=options,
        )
        num_devices = mesh_shape[0] * mesh_shape[1]
        for i in range(num_devices):
            sub_device = tt_mlir.create_sub_mesh_device(self._parent, (1, 1), (0, i))
            self._devices.add(sub_device)
        return list(self._devices)

    @classmethod
    def release_devices(self, device=None):
        devices_copy = self._devices.copy()
        if device is None:
            # If no device is specified, release all devices
            for device in devices_copy:
                tt_mlir.release_sub_mesh_device(device)
                self._devices.remove(device)
        else:
            if device in devices_copy:
                tt_mlir.release_sub_mesh_device(device)
                self._devices.remove(device)
            else:
                warnings.warn(
                    f"Device {device} not found in the list of managed devices. Ignoring it."
                )

        if len(self._devices) == 0:
            assert (
                self._parent is not None
            ), "Trying to release a non-existent parent device"
            tt_mlir.close_mesh_device(self._parent)
            self._parent = None
