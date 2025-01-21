# ttnn.remainder

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | Runs on TTNN | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|--------------|-----|------|
| ttnn.remainder | tensor<[1,i32]> <br> tensor<[1,i32]> <br> tensor<[1,i32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') <br> mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') <br> mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') | operandSegmentSizes: array<i32: 2, 1> | tensor<[1,i32]> | mapping_from: (d0), mapping_to: (0, d0), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') | no | nan | nan |
