# ttnn.neg

| Name | Args | Attributes | Input Shapes | Output Shapes | Layouts |
|------|------|------------|--------------|---------------|--------|
| ttnn.neg | %2, %3 | operandSegmentSizes: array<i32: 1, 1> |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 2, 'tile<32x32, bf16>', 'dram')}, {'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 2, 'tile<32x32, bf16>', 'dram')}] |
