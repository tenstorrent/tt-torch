from tt_torch.tools.device_manager import DeviceManager

import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)

def worker():
    print("Worker process ID: ", mp.current_process().pid)
    worker_p = list(DeviceManager.get_parent_devices())
    print("Parent devices in worker process: ", worker_p)
    worker_devices = DeviceManager.get_sub_mesh_devices(worker_p[0])
    print("Sub mesh devices in worker process: ", worker_devices)

def main():
    parent, devices = DeviceManager.acquire_available_devices()
    print("Devices acquired in main process: ")
    print("Parent: ", parent)
    print("Devices: ", devices)

    with mp.Pool(processes=2) as pool:
        pool.apply(worker)
        pool.apply(worker)

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)



if __name__ == '__main__':
    print("Current get_start_method: ", mp.get_start_method())
    main()