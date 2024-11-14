# ttnn.sum

| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts |
|------|--------------|---------------|------------|---------------|----------------|
| ttnn.sum | tensor<[19,256008,f32]> | mapping_from: ('d0', 'd1'), mapping_to: ('d0', 'd1'), memory_config: (1, 8001, 'tile<32x32, f32>', 'dram') | dim_arg: [1 : i32] <br> keep_dim: False | tensor<[19,f32]> | mapping_from: ('d0',), mapping_to: ('0', 'd0'), memory_config: (1, 1, 'tile<32x32, f32>', 'dram') |
