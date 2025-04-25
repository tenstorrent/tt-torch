# Profiling

## Introduction

tt-torch uses the tt-metal Tracy fork to manage a Tracy server. Tracy is a single process profiler, and uses a client-server model to perform performance tracing on both host calls and device operations. 

## Prerequisites

## Limitations



## Troubleshooting

- `tt-torch/install/tt-metal/tools/profiler/bin/capture-release -o tracy_profile_log_host.tracy -f -p 8086' timed out after X seconds`
    - Tracy uses a client-server model to communicate profiling data between the Tracy capture server and the client being profiled. 
    - Communication between client and server is done on a given port (default: 8086) as specified with the `-p` option.
    - If there are multiple tracy clients/server processes active at once or previous processes are left dangling, or other processes on host occupy port 8086, there may be contention and unexpected behaviour including capture server timeouts
    - This may be addressed by manually specifying an unused port with the -p option to `profile.py`
