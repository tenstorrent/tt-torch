# ttnn.abs

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.abs | tensor<[15,15,i32]> <br> tensor<[15,15,i32]> | mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') <br> mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') | operandSegmentSizes: array<i32: 1, 1> | tensor<[15,15,i32]> | mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') | -0.27 | nan |
| ttnn.abs | tensor<[15,15,i32]> <br> tensor<[15,15,i32]> | mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') <br> mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') | operandSegmentSizes: array<i32: 1, 1> | tensor<[15,15,i32]> | mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') | nan | nan |
