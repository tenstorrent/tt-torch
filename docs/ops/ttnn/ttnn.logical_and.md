# ttnn.logical_and

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts |
|------|--------------|---------------|------------|---------------|----------------|
| ttnn.logical_and | tensor<[19,bf16]> <br> tensor<[19,bf16]> <br> tensor<[19,bf16]> | mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') <br> mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') <br> mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') | operandSegmentSizes: array<i32: 2, 1> | tensor<[19,bf16]> | mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 1, 'tile<32x32, bf16>', 'dram') |