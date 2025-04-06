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
    DeviceManager._devices = set()
    DeviceManager._parent = None

    # Mock the tt_mlir module and its methods
    with patch("tt_torch.tools.device_manager.tt_mlir") as mock_mlir:
        mock_mlir.get_num_available_devices.return_value = 4
        mock_mlir.MeshDeviceOptions.return_value = MagicMock()
        mock_mlir.open_mesh_device.return_value = MockDevice(-1)
        mock_mlir.create_sub_mesh_device = MagicMock()
        mock_mlir.create_sub_mesh_device.side_effect = [MockDevice(i) for i in range(4)]
        mock_mlir.release_sub_mesh_device.return_value = MagicMock()
        mock_mlir.close_mesh_device.return_value = MagicMock()
        yield mock_mlir


def test_get_available_devices_all(mock_tt_mlir):
    """Test acquiring all available devices."""
    devices = DeviceManager.get_available_devices()

    assert mock_tt_mlir.open_mesh_device.call_count == 1
    assert mock_tt_mlir.create_sub_mesh_device.call_count == 4
    assert len(devices) == 4
    assert DeviceManager._parent is not None


def test_get_available_devices_reuse(mock_tt_mlir):
    """Test get_available_devices returns existing devices if already previously created."""
    # Call once to initialize
    initial_devices = DeviceManager.get_available_devices()
    # Call again to see if the same devices are returned
    reused_devices = DeviceManager.get_available_devices()
    # The parent and mesh subdevice calls should not be made again
    # since we are reusing the devices
    assert mock_tt_mlir.open_mesh_device.call_count == 1
    assert mock_tt_mlir.create_sub_mesh_device.call_count == 4
    assert len(initial_devices) == len(reused_devices)
    assert DeviceManager._parent is not None
    for i in range(4):
        assert initial_devices[i] == reused_devices[i]


def test_release_devices_all(mock_tt_mlir):
    """Releasing all devices should close the parent as well."""
    DeviceManager.get_available_devices()
    DeviceManager.release_devices()
    assert DeviceManager._parent is None
    assert len(DeviceManager._devices) == 0


def test_release_devices_single(mock_tt_mlir):
    """Releasing a single device should release only that device."""
    devices = DeviceManager.get_available_devices()
    one_device = devices[0]
    DeviceManager.release_devices(device=one_device)
    assert one_device not in DeviceManager._devices
    assert len(DeviceManager._devices) == 3
    assert DeviceManager._parent is not None


def test_release_devices_non_existent(mock_tt_mlir):
    """Releasing non-existent device should warn and not affect existing devices."""
    DeviceManager.get_available_devices()
    with pytest.warns(UserWarning, match="not found in the list of managed devices"):
        DeviceManager.release_devices(device=MockDevice(99))
    assert len(DeviceManager._devices) == 4
    assert DeviceManager._parent is not None
