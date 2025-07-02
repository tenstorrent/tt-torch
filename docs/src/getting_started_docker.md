# Getting Started with Docker
This document walks you through how to set up TT-Torch using a Docker image. There are two other available options for getting started:
* [Installing a Wheel](getting_started.md) - if you do not want to use Docker, and prefer to use a virtual environment by itself instead, use this method.
* [Building from Source](getting_started_build_from_source.md) - if you plan to develop TT-Forge-FE further, you must build from source, and should use this method.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

2. Reboot your machine.

3. Please ensure that after you run this script, after you complete reboot, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

4. When your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

![TT-SMI](./imgs/tt_smi.png)