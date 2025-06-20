# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch, MagicMock
from tt_torch.tools.device_manager import DeviceManager


class MockDevice:
    def __init__(self, id):
        self.device_id = id

    def __eq__(self, other):
        return self.device_id == other.device_id

    def __hash__(self):
        return hash(self.device_id)


@pytest.fixture
def mock_tt_mlir():
    # Reset the DeviceManager state before each test
    DeviceManager._devices = {}
    DeviceManager._parent_shapes = {}
    DeviceManager._submesh_shapes = {}

    # Mock the tt_mlir module and its methods
    with patch("tt_torch.tools.device_manager.tt_mlir") as mock_mlir:
        mock_mlir.get_num_available_devices.return_value = 4
        mock_mlir.MeshDeviceOptions.return_value = MagicMock()
        mock_mlir.open_mesh_device.side_effect = [MockDevice(i) for i in range(4, 8)]
        mock_mlir.create_sub_mesh_device = MagicMock()
        mock_mlir.create_sub_mesh_device.side_effect = [MockDevice(i) for i in range(4)]
        mock_mlir.release_sub_mesh_device.return_value = MagicMock()
        mock_mlir.close_mesh_device.return_value = MagicMock()
        yield mock_mlir


def test_create_parent_mesh_device(mock_tt_mlir):
    # Create a 1x1 parent mesh
    device = DeviceManager.create_parent_mesh_device((1, 1))

    # Attempt to create a 2x3 parent mesh
    with pytest.raises(AssertionError, match="Mesh shape exceeds available devices."):
        DeviceManager.create_parent_mesh_device((2, 3))

    # Attempt to create a 3D mesh with invalid shape
    with pytest.raises(
        AssertionError, match="Mesh shape must be a list of two integers."
    ):
        DeviceManager.create_parent_mesh_device((1, 1, 1))

    assert mock_tt_mlir.open_mesh_device.call_count == 1
    assert device in DeviceManager._devices
    assert len(DeviceManager._devices[device]) == 0
    assert len(DeviceManager._submesh_shapes) == 0
    assert DeviceManager._parent_shapes[device] == (1, 1)

    # Release the parent mesh
    DeviceManager.release_parent_device(device)


def test_create_sub_mesh_device(mock_tt_mlir):
    with pytest.raises(AssertionError, match="Parent mesh not found."):
        DeviceManager.create_sub_mesh_device(MockDevice(99), (1, 1))

    parent = DeviceManager.create_parent_mesh_device((2, 2))
    valid_subdevices = []
    for i in range(4):
        subdevice = DeviceManager.create_sub_mesh_device(
            parent, mesh_offset=(0, i), mesh_shape=(1, 1)
        )
        valid_subdevices.append(subdevice)

    with pytest.raises(AssertionError, match="Sub mesh shape is too big."):
        DeviceManager.create_sub_mesh_device(
            parent, mesh_offset=(0, 4), mesh_shape=(1, 1)
        )

    assert mock_tt_mlir.create_sub_mesh_device.call_count == 4
    for subdevice in valid_subdevices:
        assert subdevice in DeviceManager._devices[parent]
        assert DeviceManager._submesh_shapes[subdevice] == (1, 1)

    # Release the parent mesh
    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)


def test_release_sub_mesh_device(mock_tt_mlir):
    with pytest.raises(
        AssertionError, match="Parent mesh not found in the list of managed devices."
    ):
        DeviceManager.release_sub_mesh_device(
            sub_device=MockDevice(99), parent=MockDevice(98)
        )
    parent = DeviceManager.create_parent_mesh_device((1, 2))
    with pytest.raises(
        AssertionError, match="Sub device not found in the specified parent mesh."
    ):
        DeviceManager.release_sub_mesh_device(sub_device=MockDevice(99), parent=parent)

    with pytest.raises(
        AssertionError, match="Sub device not found in any parent mesh."
    ):
        DeviceManager.release_sub_mesh_device(sub_device=MockDevice(99))

    subdevice_1 = DeviceManager.create_sub_mesh_device(
        parent, mesh_offset=(0, 0), mesh_shape=(1, 1)
    )
    subdevice_2 = DeviceManager.create_sub_mesh_device(
        parent, mesh_offset=(0, 1), mesh_shape=(1, 1)
    )

    DeviceManager.release_sub_mesh_device(subdevice_1)

    assert subdevice_1 not in DeviceManager._devices[parent]
    assert subdevice_1 not in DeviceManager._submesh_shapes
    assert subdevice_2 in DeviceManager._devices[parent]

    DeviceManager.release_sub_mesh_device(subdevice_2, parent=parent)
    assert subdevice_2 not in DeviceManager._devices[parent]
    assert subdevice_2 not in DeviceManager._submesh_shapes
    assert len(DeviceManager._devices[parent]) == 0

    subdevice_3 = DeviceManager.create_sub_mesh_device(
        parent, mesh_offset=(0, 0), mesh_shape=(1, 1)
    )
    DeviceManager.release_sub_mesh_device(subdevice_3, cleanup_parent=True)
    assert mock_tt_mlir.release_sub_mesh_device.call_count == 3
    assert parent not in DeviceManager._devices
    assert subdevice_3 not in DeviceManager._devices
    assert len(DeviceManager._devices) == 0


def test_release_parent_device(mock_tt_mlir):
    with pytest.raises(AssertionError, match="Parent Device not found."):
        DeviceManager.release_parent_device(MockDevice(99))

    parent = DeviceManager.create_parent_mesh_device((1, 2))
    subdevice = DeviceManager.create_sub_mesh_device(parent, (0, 0))

    with pytest.raises(
        AssertionError, match="Sub devices still exist under this parent mesh."
    ):
        DeviceManager.release_parent_device(parent, cleanup_sub_devices=False)

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
    assert parent not in DeviceManager._devices
    assert parent not in DeviceManager._parent_shapes
    assert subdevice not in DeviceManager._submesh_shapes
