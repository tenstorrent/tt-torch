# ttnn.neg

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts |
|------|--------------|---------------|------------|---------------|----------------|
| ttnn.neg | tensor<[1,32,32,64,bf16]> <br> tensor<[1,32,32,64,bf16]> | mapping_from: ('d0', 'd1', 'd2', 'd3'), mapping_to: ('d0 * 1024 + d1 * 32 + d2', 'd3'), memory_config: (32, 2, 'tile<32x32, bf16>', 'dram') <br> mapping_from: ('d0', 'd1', 'd2', 'd3'), mapping_to: ('d0 * 1024 + d1 * 32 + d2', 'd3'), memory_config: (32, 2, 'tile<32x32, bf16>', 'dram') | operandSegmentSizes: array<i32: 1, 1> | tensor<[1,32,32,64,bf16]> | mapping_from: ('d0', 'd1', 'd2', 'd3'), mapping_to: ('d0 * 1024 + d1 * 32 + d2', 'd3'), memory_config: (32, 2, 'tile<32x32, bf16>', 'dram') |
