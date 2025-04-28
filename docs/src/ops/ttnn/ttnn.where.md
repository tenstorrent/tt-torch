# ttnn.where
This table is a trace for ttnn.where op. Traces are generated from nightly tt-torch runs. To see nightly runs: <a href="https://github.com/tenstorrent/tt-torch/actions/workflows/nightly-tests.yml">Nightly Runs</a>

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.where | tensor<[1,256,7,25281,si32]> <br> tensor<[1,256,7,25281,si32]> <br> tensor<[1,256,7,25281,si32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 8192 + d1 * 32 + d2, d3), memory_config: (256, 791, 'tile<32x32, si32>', 'dram') <br> mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 8192 + d1 * 32 + d2, d3), memory_config: (256, 791, 'tile<32x32, si32>', 'dram') <br> mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 8192 + d1 * 32 + d2, d3), memory_config: (256, 791, 'tile<32x32, si32>', 'dram') |  | tensor<[1,256,7,25281,si32]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 8192 + d1 * 32 + d2, d3), memory_config: (256, 791, 'tile<32x32, si32>', 'dram') | nan | nan |
