# ttnn.from_device

| Name | Args | Attributes | Input Shapes | Output Shapes | Layouts |
|------|------|------------|--------------|---------------|--------|
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 1000, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (19, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (6, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4096 + d1 * 32 + d2', 'd3'), 'memory_config': (128, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 16 + d2', 'd3'), 'memory_config': (256, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (256, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1344 + d1 * 14 + d2', 'd3'), 'memory_config': (42, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4096 + d1 * 16 + d2', 'd3'), 'memory_config': (128, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 32 + d2', 'd3'), 'memory_config': (256, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (150, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 896 + d1 * 28 + d2', 'd3'), 'memory_config': (28, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 8 + d2', 'd3'), 'memory_config': (256, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 14 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1280 + d1 * 40 + d2', 'd3'), 'memory_config': (40, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 50 + d1', 'd2'), 'memory_config': (2, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 6720 + d1 * 7 + d2', 'd3'), 'memory_config': (210, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 + d1 + d2', 'd3'), 'memory_config': (1, 29, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8064 + d1 * 56 + d2', 'd3'), 'memory_config': (252, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (512, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 18, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (38, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 320 + d1', 'd2'), 'memory_config': (10, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (7, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (29, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (38, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 256 + d2', 'd3'), 'memory_config': (512, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 72 + d1 * 8 + d2', 'd3'), 'memory_config': (3, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 324 + d1 * 27 + d2', 'd3'), 'memory_config': (11, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 40960 + d1 * 64 + d2', 'd3'), 'memory_config': (1280, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (96, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 432 + d1 * 27 + d2', 'd3'), 'memory_config': (14, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (150, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (10, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 80, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (23, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (5, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3840 + d1 * 120 + d2', 'd3'), 'memory_config': (120, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 8001, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (10, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8064 + d1 * 56 + d2', 'd3'), 'memory_config': (252, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 128 + d2', 'd3'), 'memory_config': (256, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 896 + d1 * 14 + d2', 'd3'), 'memory_config': (28, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (600, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (256, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (12, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 16384 + d1', 'd2'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 100 + d1', 'd2'), 'memory_config': (19, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 320, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 80 + d1', 'd2'), 'memory_config': (20, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 16 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 6000 + d1 * 1200 + d2', 'd3'), 'memory_config': (188, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 512 + d1', 'd2'), 'memory_config': (16, 150, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2412 + d1 * 201 + d2', 'd3'), 'memory_config': (76, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (4, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 256 + d2', 'd3'), 'memory_config': (256, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 256 + d2', 'd3'), 'memory_config': (512, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (256, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8001, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 256 + d1 + d2', 'd3'), 'memory_config': (8, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 15360 + d1 * 240 + d2', 'd3'), 'memory_config': (480, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (136, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (23, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (29, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (46, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 16, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1344 + d1 * 56 + d2', 'd3'), 'memory_config': (42, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (25, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (4, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 1024 + d2', 'd3'), 'memory_config': (256, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1920 + d1 * 30 + d2', 'd3'), 'memory_config': (60, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (20, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (38, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 100 + d1 * 100 + d2', 'd3'), 'memory_config': (19, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 108 + d1 * 9 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 256 + d2', 'd3'), 'memory_config': (256, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 32 + d2', 'd3'), 'memory_config': (640, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 9 + d2', 'd3'), 'memory_config': (5, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (188, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 100 + d1', 'd2'), 'memory_config': (4, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 64 + d1 * 8 + d2', 'd3'), 'memory_config': (2, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 16 + d2', 'd3'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 320 + d1', 'd2'), 'memory_config': (10, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 64 + d2', 'd3'), 'memory_config': (640, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 96 + d1 * 8 + d2', 'd3'), 'memory_config': (3, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 64 + d2', 'd3'), 'memory_config': (512, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 8 + d2', 'd3'), 'memory_config': (512, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (256, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 14 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 40960 + d1 * 64 + d2', 'd3'), 'memory_config': (1280, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3840 + d1 * 60 + d2', 'd3'), 'memory_config': (120, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 2048 + d2', 'd3'), 'memory_config': (512, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (150, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (24, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (40, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 12, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 160, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 32 + d2', 'd3'), 'memory_config': (256, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1280 + d1 * 40 + d2', 'd3'), 'memory_config': (40, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (32, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 8 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 120 + d2', 'd3'), 'memory_config': (960, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 938, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 128 + d2', 'd3'), 'memory_config': (512, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 100 + d1', 'd2'), 'memory_config': (4, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 256, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 28 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (5, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32768 + d1 * 4096 + d2', 'd3'), 'memory_config': (1024, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (46, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 256 + d2', 'd3'), 'memory_config': (64, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 25, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 256, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 960 + d1 * 30 + d2', 'd3'), 'memory_config': (30, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 768 + d1 * 64 + d2', 'd3'), 'memory_config': (24, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (47, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 576 + d1 * 64 + d2', 'd3'), 'memory_config': (18, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 25, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 11520 + d1 * 90 + d2', 'd3'), 'memory_config': (360, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 600, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (75, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 16 + d2', 'd3'), 'memory_config': (5, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2412 + d1 * 12 + d2', 'd3'), 'memory_config': (76, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1120 + d1 * 7 + d2', 'd3'), 'memory_config': (35, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 108 + d1 * 9 + d2', 'd3'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3840 + d1 * 60 + d2', 'd3'), 'memory_config': (120, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 480 + d2', 'd3'), 'memory_config': (960, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (3, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 61440 + d1 * 64 + d2', 'd3'), 'memory_config': (1920, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 16384 + d1', 'd2'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 512 + d1 + d2', 'd3'), 'memory_config': (16, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (38, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7680 + d1 * 120 + d2', 'd3'), 'memory_config': (240, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 25, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1792 + d1 * 112 + d2', 'd3'), 'memory_config': (56, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 32 + d2', 'd3'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 600 + d1 * 2 + d2', 'd3'), 'memory_config': (19, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (29, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 512 + d2', 'd3'), 'memory_config': (512, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 7 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4032 + d1 * 28 + d2', 'd3'), 'memory_config': (126, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 120 + d1 * 10 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5376 + d1 * 56 + d2', 'd3'), 'memory_config': (168, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1920 + d1 * 30 + d2', 'd3'), 'memory_config': (60, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (3, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1920 + d1 * 30 + d2', 'd3'), 'memory_config': (60, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 256, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 960 + d1', 'd2'), 'memory_config': (30, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 64 + d2', 'd3'), 'memory_config': (256, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 640 + d1 * 20 + d2', 'd3'), 'memory_config': (20, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (150, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (64, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (38, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 30, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (38, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 48, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, u32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2688 + d1 * 14 + d2', 'd3'), 'memory_config': (84, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (5, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 56 + d2', 'd3'), 'memory_config': (112, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 201 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 64 + d2', 'd3'), 'memory_config': (256, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1344 + d1 * 56 + d2', 'd3'), 'memory_config': (42, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32768 + d1 * 4096 + d2', 'd3'), 'memory_config': (1024, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4150 + d1', 'd2'), 'memory_config': (130, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 19200 + d1 * 19200 + d2', 'd3'), 'memory_config': (600, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 256, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8001, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8001, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (150, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 256 + d1 + d2', 'd3'), 'memory_config': (8, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 256 + d2', 'd3'), 'memory_config': (64, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (5, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 324 + d1 * 27 + d2', 'd3'), 'memory_config': (11, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (6, 46, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 144 + d1', 'd2'), 'memory_config': (5, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 600 + d1 * 50 + d2', 'd3'), 'memory_config': (19, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (7, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 12 + d1 * 12 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (7, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 48, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 128 + d1 + d2', 'd3'), 'memory_config': (4, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 16 + d2', 'd3'), 'memory_config': (640, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7168 + d1 * 112 + d2', 'd3'), 'memory_config': (224, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 201 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1280 + d1 + d2', 'd3'), 'memory_config': (40, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9600 + d1 * 4800 + d2', 'd3'), 'memory_config': (300, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4800 + d1 * 80 + d2', 'd3'), 'memory_config': (150, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 256, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 108 + d1 * 9 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (29, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (16, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3152 + d1 * 16 + d2', 'd3'), 'memory_config': (99, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 11776 + d1 * 23 + d2', 'd3'), 'memory_config': (368, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2560 + d1 * 80 + d2', 'd3'), 'memory_config': (80, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8064 + d1 * 14 + d2', 'd3'), 'memory_config': (252, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4335 + d1 * 3 + d2', 'd3'), 'memory_config': (136, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 128 + d1', 'd2'), 'memory_config': (4, 150, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 128 + d1 + d2', 'd3'), 'memory_config': (4, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (5, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1120 + d1 * 7 + d2', 'd3'), 'memory_config': (35, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 30, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 12, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (46, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9600 + d1 * 50 + d2', 'd3'), 'memory_config': (300, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 960 + d1 * 30 + d2', 'd3'), 'memory_config': (30, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 16 + d2', 'd3'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (1213, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 32 + d2', 'd3'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (16, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 80, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 32 + d2', 'd3'), 'memory_config': (640, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2364 + d1 * 197 + d2', 'd3'), 'memory_config': (74, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 32 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 160, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (1024, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (3, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 160, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 16 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 46080 + d1 * 45 + d2', 'd3'), 'memory_config': (1440, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (256, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 15360 + d1 * 240 + d2', 'd3'), 'memory_config': (480, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 29, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (20, 160, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 38400 + d1 * 30 + d2', 'd3'), 'memory_config': (1200, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 168 + d1 * 14 + d2', 'd3'), 'memory_config': (6, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 300 + d1 * 20 + d2', 'd3'), 'memory_config': (10, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 16, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (32, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (16, 16, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 32 + d2', 'd3'), 'memory_config': (960, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 16 + d2', 'd3'), 'memory_config': (640, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 8 + d2', 'd3'), 'memory_config': (64, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 960 + d1 * 30 + d2', 'd3'), 'memory_config': (30, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9600 + d1 * 2 + d2', 'd3'), 'memory_config': (300, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 9, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (3, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4335 + d1 * 1445 + d2', 'd3'), 'memory_config': (136, 46, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 19 + d1 * 19 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2240 + d1 * 7 + d2', 'd3'), 'memory_config': (70, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 9 + d2', 'd3'), 'memory_config': (5, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 80, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 201 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 960 + d1 * 30 + d2', 'd3'), 'memory_config': (30, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (23, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1280 + d1 * 40 + d2', 'd3'), 'memory_config': (40, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 25, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 480 + d2', 'd3'), 'memory_config': (960, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 50 + d1', 'd2'), 'memory_config': (2, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (7360, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 768 + d1', 'd2'), 'memory_config': (24, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (64, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (75, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 256 + d2', 'd3'), 'memory_config': (64, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 304 + d1 * 16 + d2', 'd3'), 'memory_config': (10, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 720 + d1', 'd2'), 'memory_config': (68, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 80, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 640 + d1 * 20 + d2', 'd3'), 'memory_config': (20, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (512, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 8 + d2', 'd3'), 'memory_config': (64, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (5, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1280 + d1 * 40 + d2', 'd3'), 'memory_config': (40, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4096 + d1 * 64 + d2', 'd3'), 'memory_config': (128, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 98, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (29, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 80, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 938, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 201 + d1', 'd2'), 'memory_config': (76, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 344, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 320, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 512 + d1 + d2', 'd3'), 'memory_config': (16, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (18, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 48, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 80, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 64 + d2', 'd3'), 'memory_config': (512, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 7813, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 120 + d1 * 10 + d2', 'd3'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5376 + d1 * 14 + d2', 'd3'), 'memory_config': (168, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (38, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 512, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (1, 7813, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 72 + d1 * 9 + d2', 'd3'), 'memory_config': (3, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 12 + d2', 'd3'), 'memory_config': (5, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 60 + d1', 'd2'), 'memory_config': (2, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 9 + d2', 'd3'), 'memory_config': (5, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 64 + d2', 'd3'), 'memory_config': (640, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 14336 + d1 * 7 + d2', 'd3'), 'memory_config': (448, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 19200 + d1 + d2', 'd3'), 'memory_config': (600, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 7 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (16, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (24, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 304 + d1 * 19 + d2', 'd3'), 'memory_config': (10, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (5, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (19, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 15360 + d1 * 240 + d2', 'd3'), 'memory_config': (480, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7680 + d1 * 60 + d2', 'd3'), 'memory_config': (240, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 50 + d1', 'd2'), 'memory_config': (2, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 61440 + d1 * 64 + d2', 'd3'), 'memory_config': (1920, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2412 + d1 * 201 + d2', 'd3'), 'memory_config': (76, 7, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32768 + d1 * 8 + d2', 'd3'), 'memory_config': (1024, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (3, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (19, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9600 + d1 * 30 + d2', 'd3'), 'memory_config': (300, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (150, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 72 + d1 * 9 + d2', 'd3'), 'memory_config': (3, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5888 + d1 * 23 + d2', 'd3'), 'memory_config': (184, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 72 + d1 * 8 + d2', 'd3'), 'memory_config': (3, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 48, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 2048 + d2', 'd3'), 'memory_config': (512, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 7813, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 14 + d1 * 14 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (20, 160, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2560 + d1 * 80 + d2', 'd3'), 'memory_config': (80, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 640 + d1 * 20 + d2', 'd3'), 'memory_config': (20, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 512 + d1', 'd2'), 'memory_config': (16, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 938, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 168 + d1 * 14 + d2', 'd3'), 'memory_config': (6, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 64 + d1 + d2', 'd3'), 'memory_config': (2, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (29, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 640 + d1 + d2', 'd3'), 'memory_config': (20, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (7, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (64, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 512 + d1', 'd2'), 'memory_config': (16, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (600, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32 + d1 * 32 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (25, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 32 + d2', 'd3'), 'memory_config': (640, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 168 + d1 * 14 + d2', 'd3'), 'memory_config': (6, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 300 + d1 + d2', 'd3'), 'memory_config': (10, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 30 + d1', 'd2'), 'memory_config': (1, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 108 + d1 * 12 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 80, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1280 + d1', 'd2'), 'memory_config': (40, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 50 + d1', 'd2'), 'memory_config': (2, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 320, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (38, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (38, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5120 + d1 * 1024 + d2', 'd3'), 'memory_config': (160, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 320 + d1', 'd2'), 'memory_config': (10, 38, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 120 + d1', 'd2'), 'memory_config': (4, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 8 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 320, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 6000 + d1 * 5 + d2', 'd3'), 'memory_config': (188, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (136, 46, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7680 + d1 * 120 + d2', 'd3'), 'memory_config': (240, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (7, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 30, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7 + d1 * 7 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 15360 + d1 * 120 + d2', 'd3'), 'memory_config': (480, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 9 + d2', 'd3'), 'memory_config': (5, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 9, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4032 + d1 * 7 + d2', 'd3'), 'memory_config': (126, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 16384 + d1', 'd2'), 'memory_config': (512, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 256, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (32, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 100 + d1', 'd2'), 'memory_config': (19, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10 + d1 * 10 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 120 + d1 * 120 + d2', 'd3'), 'memory_config': (4, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (24, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 896 + d1 * 14 + d2', 'd3'), 'memory_config': (28, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (7, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 100 + d1', 'd2'), 'memory_config': (4, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 64 + d2', 'd3'), 'memory_config': (512, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 23040 + d1 * 45 + d2', 'd3'), 'memory_config': (720, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (80, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 344, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 160 + d1', 'd2'), 'memory_config': (40, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (300, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (16, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 48, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 24576 + d1 * 64 + d2', 'd3'), 'memory_config': (768, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1344 + d1 * 14 + d2', 'd3'), 'memory_config': (42, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 80 + d1', 'd2'), 'memory_config': (20, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 96, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 160, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 128 + d1', 'd2'), 'memory_config': (4, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 1024 + d2', 'd3'), 'memory_config': (256, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 25, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 61440 + d1 * 64 + d2', 'd3'), 'memory_config': (1920, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 12 + d2', 'd3'), 'memory_config': (5, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 23040 + d1 * 180 + d2', 'd3'), 'memory_config': (720, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 256 + d1 * 16 + d2', 'd3'), 'memory_config': (8, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 80, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2400 + d1 * 8 + d2', 'd3'), 'memory_config': (75, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 128 + d1', 'd2'), 'memory_config': (64, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 32 + d2', 'd3'), 'memory_config': (960, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4335 + d1 * 1445 + d2', 'd3'), 'memory_config': (136, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 320 + d1 + d2', 'd3'), 'memory_config': (10, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 320, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 32 + d2', 'd3'), 'memory_config': (960, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (64, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 7813, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 768 + d1', 'd2'), 'memory_config': (24, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (512, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 192 + d1', 'd2'), 'memory_config': (6, 42, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (12, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (4, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (600, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 256, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (150, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2400 + d1 * 300 + d2', 'd3'), 'memory_config': (75, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 168 + d1 * 12 + d2', 'd3'), 'memory_config': (6, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (150, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 8001, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 60 + d2', 'd3'), 'memory_config': (960, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1200 + d1', 'd2'), 'memory_config': (188, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1500 + d1 * 5 + d2', 'd3'), 'memory_config': (47, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 640 + d1 * 20 + d2', 'd3'), 'memory_config': (20, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 344, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (7, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (512, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (18, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2364 + d1 * 197 + d2', 'd3'), 'memory_config': (74, 7, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5120 + d1 * 32 + d2', 'd3'), 'memory_config': (160, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 128 + d1', 'd2'), 'memory_config': (128, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, u32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (512, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 320, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 256, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 56 + d1 * 7 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (1, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2364 + d1 * 197 + d2', 'd3'), 'memory_config': (74, 7, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 7813, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 160, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 1024 + d2', 'd3'), 'memory_config': (256, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 100 + d1 + d2', 'd3'), 'memory_config': (19, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4096 + d1 * 16 + d2', 'd3'), 'memory_config': (128, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 100 + d1 * 100 + d2', 'd3'), 'memory_config': (19, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 160, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (19, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 19200 + d1 * 19200 + d2', 'd3'), 'memory_config': (600, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (48, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 256 + d2', 'd3'), 'memory_config': (64, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (2, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 320 + d1 * 10 + d2', 'd3'), 'memory_config': (10, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 193 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1280 + d1 * 40 + d2', 'd3'), 'memory_config': (40, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout7', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (46, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 56 + d2', 'd3'), 'memory_config': (112, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 46080 + d1 * 90 + d2', 'd3'), 'memory_config': (1440, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 640 + d1', 'd2'), 'memory_config': (20, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 120 + d1 * 12 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 + d2', 'd3'), 'memory_config': (64, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 160, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (1213, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8960 + d1 * 7 + d2', 'd3'), 'memory_config': (280, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5120 + d1 * 5 + d2', 'd3'), 'memory_config': (160, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 304 + d1 * 19 + d2', 'd3'), 'memory_config': (10, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (5, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 80, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (3, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (150, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (1024, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32 + d1 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 128 + d2', 'd3'), 'memory_config': (512, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (7, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (7, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 196 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 96, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32768 + d1 * 4096 + d2', 'd3'), 'memory_config': (1024, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5376 + d1 * 28 + d2', 'd3'), 'memory_config': (168, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5376 + d1 * 28 + d2', 'd3'), 'memory_config': (168, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (4, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 729 + d1 * 27 + d2', 'd3'), 'memory_config': (23, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 23040 + d1 * 90 + d2', 'd3'), 'memory_config': (720, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 960 + d1 * 15 + d2', 'd3'), 'memory_config': (30, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 14 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 * 32 + d2', 'd3'), 'memory_config': (32, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 6720 + d1 * 7 + d2', 'd3'), 'memory_config': (210, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (10, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (160, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (30, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 128 + d2', 'd3'), 'memory_config': (256, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 48, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (29, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 40960 + d1 * 32 + d2', 'd3'), 'memory_config': (1280, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (4, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 7 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5376 + d1 * 14 + d2', 'd3'), 'memory_config': (168, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 60 + d1 * 60 + d2', 'd3'), 'memory_config': (2, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (20, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1000, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (344, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30 + d1 * 30 + d2', 'd3'), 'memory_config': (1, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 72 + d1 * 8 + d2', 'd3'), 'memory_config': (3, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 18, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1280 + d1', 'd2'), 'memory_config': (40, 38, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 600, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (16, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (40, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 344, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (6, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9216 + d1 * 12 + d2', 'd3'), 'memory_config': (288, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 160 + d1', 'd2'), 'memory_config': (40, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 100 + d1 * 100 + d2', 'd3'), 'memory_config': (19, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (600, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 640 + d1 * 20 + d2', 'd3'), 'memory_config': (20, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 256 + d2', 'd3'), 'memory_config': (64, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (46, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 30720 + d1 * 15 + d2', 'd3'), 'memory_config': (960, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9 + d1 * 9 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 8 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 729 + d1 * 27 + d2', 'd3'), 'memory_config': (23, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (8, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (16, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8064 + d1 * 14 + d2', 'd3'), 'memory_config': (252, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (64, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (160, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19200 + d1', 'd2'), 'memory_config': (600, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 160 + d1', 'd2'), 'memory_config': (5, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 20480 + d1 * 16 + d2', 'd3'), 'memory_config': (640, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 32 + d1 * 32 + d2', 'd3'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (99, 7, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (38, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 6000 + d1 * 1200 + d2', 'd3'), 'memory_config': (188, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (32, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 9 + d2', 'd3'), 'memory_config': (5, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 938, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 16384 + d1', 'd2'), 'memory_config': (512, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (29, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 512, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 768 + d1', 'd2'), 'memory_config': (24, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 64, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 7 + d1', 'd2'), 'memory_config': (1, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 16, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 9, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (128, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 16, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 7813, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 10 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1920 + d1 * 60 + d2', 'd3'), 'memory_config': (60, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3584 + d1 * 28 + d2', 'd3'), 'memory_config': (112, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 19 + d1 * 19 + d2', 'd3'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2400 + d1 * 300 + d2', 'd3'), 'memory_config': (75, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (2, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1280 + d1', 'd2'), 'memory_config': (40, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 320 + d1 * 10 + d2', 'd3'), 'memory_config': (10, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (150, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 120 + d1 * 10 + d2', 'd3'), 'memory_config': (4, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (8, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 20, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 72 + d1 * 9 + d2', 'd3'), 'memory_config': (3, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (18, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3840 + d1 * 30 + d2', 'd3'), 'memory_config': (120, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (80, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 201 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (150, 16, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 40, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (6, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 920 + d1 * 40 + d2', 'd3'), 'memory_config': (29, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (24, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (16, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (3, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 2048 + d1', 'd2'), 'memory_config': (64, 16, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 16384 + d1', 'd2'), 'memory_config': (512, 8, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1445 + d1', 'd2'), 'memory_config': (46, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (4, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (64, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1024 + d1 + d2', 'd3'), 'memory_config': (32, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 100 + d1', 'd2'), 'memory_config': (4, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (600, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 10, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 8001, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 960 + d1 * 30 + d2', 'd3'), 'memory_config': (30, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 25, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 8 + d2', 'd3'), 'memory_config': (64, 3, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 3, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 40 + d1', 'd2'), 'memory_config': (10, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 300 + d1', 'd2'), 'memory_config': (10, 64, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 14 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 256 + d1', 'd2'), 'memory_config': (8, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (30, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 4096 + d1 * 64 + d2', 'd3'), 'memory_config': (128, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 144 + d1 * 12 + d2', 'd3'), 'memory_config': (5, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (4, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4800 + d1', 'd2'), 'memory_config': (300, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 16 + d2', 'd3'), 'memory_config': (256, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 160, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 938, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10752 + d1 * 112 + d2', 'd3'), 'memory_config': (336, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 938, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 1024 + d1', 'd2'), 'memory_config': (32, 80, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (46, 6, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 40 + d1', 'd2'), 'memory_config': (10, 128, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 5, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (5, 2, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 46080 + d1 * 180 + d2', 'd3'), 'memory_config': (1440, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 96 + d1 * 8 + d2', 'd3'), 'memory_config': (3, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 324 + d1 * 27 + d2', 'd3'), 'memory_config': (11, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 19200 + d1 * 160 + d2', 'd3'), 'memory_config': (600, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 3152 + d1 * 197 + d2', 'd3'), 'memory_config': (99, 7, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 9 + d1', 'd2'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 + d1', 'd2'), 'memory_config': (6, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 320 + d1 * 10 + d2', 'd3'), 'memory_config': (10, 32, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 896 + d1 * 28 + d2', 'd3'), 'memory_config': (28, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 1024 + d2', 'd3'), 'memory_config': (256, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (2, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 9600 + d1 * 4800 + d2', 'd3'), 'memory_config': (300, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (40, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 197 + d1', 'd2'), 'memory_config': (7, 24, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 12 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 19 + d1', 'd2'), 'memory_config': (1, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 344, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (32, 20, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout1', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (128, 128, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0',), 'mapping_to': ('0', 'd0'), 'memory_config': (1, 160, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 64 + d1', 'd2'), 'memory_config': (24, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout5', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (38, 40, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7680 + d1 * 120 + d2', 'd3'), 'memory_config': (240, 5, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (8, 320, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 5120 + d1 * 1024 + d2', 'd3'), 'memory_config': (160, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (15, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (64, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 2048 + d1 * 256 + d2', 'd3'), 'memory_config': (64, 8, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 8 + d1', 'd2'), 'memory_config': (1, 24, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (10, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 16384 + d1 * 8 + d2', 'd3'), 'memory_config': (512, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout2', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 11776 + d1 * 23 + d2', 'd3'), 'memory_config': (368, 2, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 4, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 10240 + d1 * 32 + d2', 'd3'), 'memory_config': (320, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 8192 + d1 * 32 + d2', 'd3'), 'memory_config': (256, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 6, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 32 + d1', 'd2'), 'memory_config': (1, 1000, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 1200 + d1 * 40 + d2', 'd3'), 'memory_config': (38, 10, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout4', 'mapping_from': ('d0', 'd1'), 'mapping_to': ('d0', 'd1'), 'memory_config': (1, 1, 'tile<32x32, f32>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 7168 + d1 * 14 + d2', 'd3'), 'memory_config': (224, 1, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2', 'd3'), 'mapping_to': ('d0 * 320 + d1 * 10 + d2', 'd3'), 'memory_config': (10, 32, 'tile<32x32, bf16>', 'dram')}] |
| ttnn.from_device |  |  |  |  | [{'id': '#layout3', 'mapping_from': ('d0', 'd1', 'd2'), 'mapping_to': ('d0 * 4096 + d1', 'd2'), 'memory_config': (1024, 1, 'tile<32x32, f32>', 'dram')}] |
