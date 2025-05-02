# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
from typing import KeysView


class DeviceManager:
    # Dictionary to keep track of each parent mesh and
    # associated sub_meshes
    _devices: dict[tt_mlir.Device, set[tt_mlir.Device]] = {}

    # Dictionaries to keep track of the mesh shape for each device
    _parent_shapes: dict[tt_mlir.Device, tuple[int, int]] = {}
    _submesh_shapes: dict[tt_mlir.Device, tuple[int, int]] = {}

    @staticmethod
    def _get_parent_mesh_options(
        device_ids=None,
        num_hw_cqs=None,
        enable_program_cache=None,
        l1_small_size=None,
        dispatch_core_type=None,
    ) -> tt_mlir.MeshDeviceOptions:
        options = tt_mlir.MeshDeviceOptions()
        if device_ids is not None:
            options.device_ids = device_ids
        if num_hw_cqs is not None:
            options.num_hw_cqs = num_hw_cqs
        if enable_program_cache is not None:
            options.enable_program_cache = enable_program_cache
        if l1_small_size is not None:
            options.l1_small_size = l1_small_size
        if dispatch_core_type is not None:
            options.dispatch_core_type = dispatch_core_type
        return options

    @classmethod
    def get_num_available_devices(cls) -> int:
        """
        Returns the number of available devices.
        """
        return tt_mlir.get_num_available_devices()

    @classmethod
    def create_parent_mesh_device(
        cls,
        mesh_shape,
        device_ids=None,
        num_hw_cqs=None,
        enable_program_cache=None,
        l1_small_size=None,
        dispatch_core_type=None,
    ) -> tt_mlir.Device:
        """
        Acquires a new tt_mlir.Device object representing a parent mesh device and returns it.
        The returned parent device can either be used directly for model execution, or can be used
        to create sub mesh devices.
        """
        num_available = cls.get_num_available_devices()
        assert len(mesh_shape) == 2, "Mesh shape must be a list of two integers."
        assert (
            mesh_shape[0] * mesh_shape[1] <= num_available
        ), "Mesh shape exceeds available devices."
        options = cls._get_parent_mesh_options(
            device_ids,
            num_hw_cqs,
            enable_program_cache,
            l1_small_size,
            dispatch_core_type,
        )
        parent_mesh = tt_mlir.open_mesh_device(mesh_shape=mesh_shape, options=options)
        cls._devices[parent_mesh] = set()
        cls._parent_shapes[parent_mesh] = mesh_shape
        return parent_mesh

    @classmethod
    def get_parent_devices(cls) -> KeysView[tt_mlir.Device]:
        """
        Returns a list of all currently acquired parent mesh devices.
        """
        return cls._devices.keys()

    @classmethod
    def release_parent_device(
        cls, parent_device: tt_mlir.Device, cleanup_sub_devices: bool = False
    ):
        """
        Releases the specified parent mesh device, if there are no open sub-devices
        associated with the given parent.

        If cleanup_sub_devices is True, it will also close all acquired sub mesh devices
        associated with the given parent device.
        """
        assert parent_device in cls._devices, "Parent Device not found."
        sub_devices = cls._devices[parent_device]
        assert (
            len(sub_devices) == 0 or cleanup_sub_devices
        ), "Sub devices still exist under this parent mesh."
        if cleanup_sub_devices:
            for sub_device in sub_devices.copy():
                cls.release_sub_mesh_device(sub_device, parent=parent_device)
        tt_mlir.close_mesh_device(parent_device)
        del cls._devices[parent_device]
        del cls._parent_shapes[parent_device]

    @classmethod
    def create_sub_mesh_device(
        cls,
        parent_mesh: tt_mlir.Device,
        mesh_offset: tuple[int, int],
        mesh_shape: tuple[int, int] = (1, 1),
    ) -> tt_mlir.Device:
        """
        Creates a sub mesh device under the given parent mesh device and returns it.
        """
        assert parent_mesh in cls._devices, "Parent mesh not found."

        parent_shape = cls._parent_shapes[parent_mesh]
        capacity = parent_shape[0] * parent_shape[1]
        curr_usage = mesh_shape[0] * mesh_shape[1]

        for sub_device in cls._devices[parent_mesh]:
            shape = cls._submesh_shapes[sub_device]
            curr_usage += shape[0] * shape[1]
        assert curr_usage <= capacity, "Sub mesh shape is too big."

        # TODO: Add an additional check to ensure that BOTH the provided mesh_offset
        # and mesh_shape are valid and fit within the parent mesh.

        sub_device = tt_mlir.create_sub_mesh_device(
            parent_mesh, mesh_shape, mesh_offset
        )
        cls._devices[parent_mesh].add(sub_device)
        cls._submesh_shapes[sub_device] = mesh_shape
        return sub_device

    @classmethod
    def get_sub_mesh_devices(cls, parent_mesh: tt_mlir.Device) -> set[tt_mlir.Device]:
        """
        Returns all acquired sub mesh devices under a given parent mesh device.
        """
        assert parent_mesh in cls._devices, "Parent mesh not found."
        return cls._devices[parent_mesh]

    @classmethod
    def release_sub_mesh_device(
        cls,
        sub_device: tt_mlir.Device,
        cleanup_parent: bool = False,
        parent: tt_mlir.Device = None,
    ):
        """
        Closes the specified sub_mesh device.
        If cleanup_parent is True, it will also close the parent mesh device
        if the sub device is the last one under that parent.
        """
        if parent is not None:
            # Parent mesh is specified by caller, validate it
            assert (
                parent in cls._devices
            ), "Parent mesh not found in the list of managed devices."
            assert (
                sub_device in cls._devices[parent]
            ), "Sub device not found in the specified parent mesh."
        else:
            # Parent mesh is not specified, find it
            for parent_mesh, sub_devices in cls._devices.items():
                if sub_device in sub_devices:
                    parent = parent_mesh
                    break
            assert parent is not None, "Sub device not found in any parent mesh."
        tt_mlir.release_sub_mesh_device(sub_device)
        cls._devices[parent].remove(sub_device)
        del cls._submesh_shapes[sub_device]
        if cleanup_parent:
            if len(cls._devices[parent]) == 0:
                tt_mlir.close_mesh_device(parent)
                del cls._devices[parent]
                del cls._parent_shapes[parent]

    @classmethod
    def acquire_available_devices(
        cls,
        num_devices=None,
        device_ids=None,
        num_hw_cqs=None,
        enable_program_cache=None,
        l1_small_size=None,
        dispatch_core_type=None,
    ) -> tuple[tt_mlir.Device, list[tt_mlir.Device]]:
        """
        Opens a parent mesh and makes `num_device` 1x1 sub mesh devices available for use in a 1D mesh.
        If `num_device` is None, all available devices will be acquired.
        Returns a tuple of the parent mesh device and a list of sub mesh devices.

        The sub mesh device list should be used for all compute tasks.
        """
        num_available = tt_mlir.get_num_available_devices()
        if num_devices is None:
            num_devices = num_available
        mesh_shape = [1, num_devices]
        parent = cls.create_parent_mesh_device(
            mesh_shape,
            device_ids,
            num_hw_cqs,
            enable_program_cache,
            l1_small_size,
            dispatch_core_type,
        )

        for i in range(num_devices):
            cls.create_sub_mesh_device(parent, (0, i))
        return (parent, list(cls._devices[parent]))
