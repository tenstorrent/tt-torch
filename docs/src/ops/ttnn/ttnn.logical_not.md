# ttnn.logical_not

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.logical_not | tensor<[1,bf16]> <br> tensor<[1,bf16]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') <br> mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') | operandSegmentSizes: array<i32: 1, 1> | tensor<[1,bf16]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') | nan | nan |
