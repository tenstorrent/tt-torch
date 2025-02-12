# ttnn.ge

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |
|------|--------------|---------------|------------|---------------|----------------|-----|------|
| ttnn.ge | tensor<[5,5,ui32]> <br> tensor<[5,5,ui32]> <br> tensor<[5,5,bf16]> | mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') <br> mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, u32>', 'dram') <br> mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') | operandSegmentSizes: array<i32: 2, 1> | tensor<[5,5,bf16]> | mapping_from: (d0, d1), mapping_to: (d0, d1), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') | nan | nan |
