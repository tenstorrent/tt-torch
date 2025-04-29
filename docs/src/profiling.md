# Profiling

## Introduction

tt-torch uses the tt-metal Tracy fork to manage a Tracy server. Tracy is a single process profiler, and uses a client-server model to perform performance tracing on both host calls and on-device operations. tt-torch implements a wrapper called `profile.py` with custom orchestration logic to handle the spawning of the Tracy capture server and the client workload to be profiled, as well as report generation and data postprocessing functionality.

## Prerequisites

- In the tt-torch building step ([Building](./build.md)), it is required to configure your cmake build with the additional cmake directive `TT_RUNTIME_ENABLE_PERF_TRACE=ON`.
    - i.e. run: `cmake -G Ninja -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON`
- Paths in this document are given relative to the repo root.

## Usage

The `profile.py` tool is the recommended entrypoint for profiling workloads in tt-torch.

```
profile.py [-h] [-o OUTPUT_PATH] [-p PORT] "test_command"
```
**Note: The `test_command` must be quoted!**


As a minimal example, this will run and profile the MNIST test:
```
python tt_torch/tools/profile.py "pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]"
```

It generates the following report: `results/perf/device_ops_perf_trace.csv`. This report displays a table of operations executed on device and rich timing and configuration data associated with them.

## Limitations

- Tracy is a single process profiler and will not work with multiprocessed workflows. This includes tests parameterized by `op_by_op_shlo` and `op_by_op_torch`, which run individual ops in isolated processes.
- To view traces, you can use `install/tt-metal/generated/profiler/.logs/tracy_profile_log_host.tracy`.
    - This is a `.tracy` file that can be consumed by the tt-metal Tracy GUI and produce visual profiling traces of host and device activity.
    - You must use the tt-metal Tracy GUI to view this file. Refer to the [GUI section](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html#gui) in the tt-metal profiling documentation. Other sections are not applicable to tt-torch profiling.

## Troubleshooting

- `tt-torch/install/tt-metal/tools/profiler/bin/capture-release -o tracy_profile_log_host.tracy -f -p 8086' timed out after X seconds`
    - Tracy uses a client-server model to communicate profiling data between the Tracy capture server and the client being profiled.
    - Communication between client and server is done on a given port (default: 8086) as specified with the `-p` option.
    - If there are multiple tracy clients/server processes active at once or previous processes are left dangling, or other processes on host occupying port 8086, there may be contention and unexpected behaviour including capture server timeouts.
    - This may be addressed by manually specifying an unused port with the -p option to `profile.py`.
