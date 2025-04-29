# ttnn.full
This table is a trace for ttnn.full op. Traces are generated from nightly tt-torch runs. To see nightly runs: <a href="https://github.com/tenstorrent/tt-torch/actions/workflows/nightly-tests.yml">Nightly Runs</a>

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,256,7,25281,2,f32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 45359104 + d1 * 177184 + d2 * 25312 + d3, d4), memory_config: (1417472, 1, 'tile<32x32, f32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[16,250,250,1,ui32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 64000 + d1 * 256 + d2, d3), memory_config: (32000, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,2640,768,1,ui32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 2027520 + d1 * 768 + d2, d3), memory_config: (63360, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,300,4,1,ui32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 9600 + d1 * 32 + d2, d3), memory_config: (300, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,300,80,1,ui32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 28800 + d1 * 96 + d2, d3), memory_config: (900, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[16,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[16,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[6,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 6, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[6,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 6, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,16,6,192,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 18432 + d1 * 1152 + d2 * 192 + d3, d4), memory_config: (576, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[16,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[16,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 16, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[6,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 6, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[6,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 6, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,16,6,192,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 18432 + d1 * 1152 + d2 * 192 + d3, d4), memory_config: (576, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[3,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 3, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[3,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 3, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,1,1,1,3,1,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4, d5, d6), mapping_to: (d0 * 96 + d1 * 96 + d2 * 96 + d3 * 96 + d4 * 32 + d5, d6), memory_config: (3, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,256,7,25281,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 45359104 + d1 * 177184 + d2 * 25312 + d3, d4), memory_config: (1417472, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,256,7,25281,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 45359104 + d1 * 177184 + d2 * 25312 + d3, d4), memory_config: (1417472, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,256,7,25281,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 45359104 + d1 * 177184 + d2 * 25312 + d3, d4), memory_config: (1417472, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[7,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 7, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[7,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 7, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[25281,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 25281, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[25281,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 25281, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,7,25281,2,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4), mapping_to: (d0 * 5662944 + d1 * 808992 + d2 * 32 + d3, d4), memory_config: (176967, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[1,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[8,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 8, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[8,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 8, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[160,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 160, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 1.000000e+00 : f32 | tensor<[160,ui32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 160, 'ui32', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,8,160,160,1,1,ui32]> | mapping_from: (d0, d1, d2, d3, d4, d5), mapping_to: (d0 * 6553600 + d1 * 819200 + d2 * 5120 + d3 * 32 + d4, d5), memory_config: (204800, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,16,14,14,bf16]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 512 + d1 * 32 + d2, d3), memory_config: (16, 1, 'tile<32x32, bf16>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,16,28,28,bf16]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 512 + d1 * 32 + d2, d3), memory_config: (16, 1, 'tile<32x32, bf16>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,4,14,14,bf16]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 128 + d1 * 32 + d2, d3), memory_config: (4, 1, 'tile<32x32, bf16>', 'dram') | nan | nan |
| ttnn.full | !ttnn.device |  | fillValue: 0.000000e+00 : f32 | tensor<[1,256,7,25281,ui32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 8192 + d1 * 32 + d2, d3), memory_config: (256, 791, 'tile<32x32, u32>', 'dram') | nan | nan |
