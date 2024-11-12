# ttnn.exp

| Name | Args | Attributes | Input Shapes | Output Shapes | Layouts |
|------|------|------------|--------------|---------------|--------|
| ttnn.exp | %2, %3 | operandSegmentSizes: array<i32: 1, 1> |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 5, 'tile<32x32, f32>', 'dram')}, {'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.exp | %2, %3 | operandSegmentSizes: array<i32: 1, 1> |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8001, 'tile<32x32, f32>', 'dram')}, {'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8001, 'tile<32x32, f32>', 'dram')}] |
