# ttnn.arange
This table is a trace for ttnn.arange op. Traces are generated from nightly tt-torch runs. To see nightly runs: <a href="https://github.com/tenstorrent/tt-torch/actions/workflows/nightly-tests.yml">Nightly Runs</a>

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 16 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[16,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 250 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x8>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[250,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 250, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 768 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x24>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[768,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 768, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 4 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[4,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 4, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 80 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x3>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[80,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 80, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 6 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[6,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 6, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 16 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[16,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 6 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[6,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 6, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 16 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[16,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 3 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[3,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 3, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 25281 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x791>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[25281,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 25281, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 7 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[7,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 7, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 160 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x5>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[160,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 160, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 8 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[8,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 8, 'si32', 'dram') | nan | nan |
| ttnn.arange | !ttnn.device |  | dtype: #tt.supportedDataTypes<si32> <br> end: 1 : i64 <br> memory_config: #ttnn.memory_config<#dram, <<1x1>>, <interleaved>> <br> start: 0 : i64 <br> step: 1 : i64 | tensor<[1,si32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'si32', 'dram') | nan | nan |
