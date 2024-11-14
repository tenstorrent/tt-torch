# ttnn.exp

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts |
|------|--------------|---------------|------------|---------------|----------------|
| ttnn.exp | tensor<[160,f32]> <br> tensor<[160,f32]> | mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 5, 'tile<32x32, f32>', 'dram') <br> mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 5, 'tile<32x32, f32>', 'dram') | operandSegmentSizes: array<i32: 1, 1> | tensor<[160,f32]> | mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 5, 'tile<32x32, f32>', 'dram') |
| ttnn.exp | tensor<[19,256008,f32]> <br> tensor<[19,256008,f32]> | mapping_from: ('d0', 'd1'), mapping_to: ('d0', 'd1'), memory_config: (1, 8001, 'tile<32x32, f32>', 'dram') <br> mapping_from: ('d0', 'd1'), mapping_to: ('d0', 'd1'), memory_config: (1, 8001, 'tile<32x32, f32>', 'dram') | operandSegmentSizes: array<i32: 1, 1> | tensor<[19,256008,f32]> | mapping_from: ('d0', 'd1'), mapping_to: ('d0', 'd1'), memory_config: (1, 8001, 'tile<32x32, f32>', 'dram') |
