# ttnn.scatter

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.scatter | tensor<[1,3,720,1280,bf16]> <br> tensor<[1,3,720,1280,bf16]> <br> tensor<[1,3,720,1280,bf16]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 2160 + d1 * 720 + d2, d3), memory_config: (68, 40, 'tile<32x32, bf16>', 'dram') <br> mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 2160 + d1 * 720 + d2, d3), memory_config: (68, 40, 'tile<32x32, bf16>', 'dram') <br> mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 2160 + d1 * 720 + d2, d3), memory_config: (68, 40, 'tile<32x32, bf16>', 'dram') | operandSegmentSizes: array<i32: 2, 1> | tensor<[1,3,720,1280,bf16]> | mapping_from: (d0, d1, d2, d3), mapping_to: (d0 * 2160 + d1 * 720 + d2, d3), memory_config: (68, 40, 'tile<32x32, bf16>', 'dram') | nan | nan |
