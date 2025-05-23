# TTNN OP Traces

The following pages have traces of operations that are currently not being compiled correctly. They can be updated by running:
```
python tt_torch/tools/generate_md.py --excel_path <path to xlsx file> --md_dir docs/src/ops/ttnn --json_dir docs/src/ops/ttnn --failures_only
```

# How to read these files?

The *.md/ *.json files store information related to ops from ttnn graphs. A TTNN Graph could look like the following

```
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = ""x86_64-pc-linux-gnu""}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 14336 + d1 * 14 + d2, d3), <1x1>, memref<14336x14xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3072 + d1 * 3 + d2, d3), <1x1>, memref<3145728x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 14336 + d1 * 14 + d2, d3), <1x1>, memref<448x1x!tt.tile<32x32, bf16>, #dram>, interleaved>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 14336 + d1 * 1024 + d2, d3), <1x1>, memref<448x1x!tt.tile<32x32, bf16>, #dram>, interleaved>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 196 + d1 * 14 + d2, d3), <1x1>, memref<7x32x!tt.tile<32x32, bf16>, #dram>, interleaved>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func @main(%arg0: tensor<1x1024x14x14xbf16, #ttnn_layout>, %arg1: tensor<1024x1024x3x3xbf16, #ttnn_layout1>) -> tensor<1x1024x14x14xbf16, #ttnn_layout> {
    %0 = ""ttnn.get_device""() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = ""ttnn.to_device""(%arg0, %0) <{memory_config = #ttnn.memory_config<<interleaved>, #dram, <<448x1>>>}> : (tensor<1x1024x14x14xbf16, #ttnn_layout>, !tt.device<#device>) -> tensor<1x1024x14x14xbf16, #ttnn_layout2>
    %2 = ""ttnn.to_layout""(%1) <{layout = #ttnn.layout<tile>}> : (tensor<1x1024x14x14xbf16, #ttnn_layout2>) -> tensor<1x1024x14x14xbf16, #ttnn_layout2>
    ""ttnn.deallocate""(%1) <{force = false}> : (tensor<1x1024x14x14xbf16, #ttnn_layout2>) -> ()
    %3 = ""ttnn.transpose""(%2) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x1024x14x14xbf16, #ttnn_layout2>) -> tensor<1x14x1024x14xbf16, #ttnn_layout3>
    ""ttnn.deallocate""(%2) <{force = false}> : (tensor<1x1024x14x14xbf16, #ttnn_layout2>) -> ()
    %4 = ""ttnn.transpose""(%3) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x14x1024x14xbf16, #ttnn_layout3>) -> tensor<1x14x14x1024xbf16, #ttnn_layout4>
    ""ttnn.deallocate""(%3) <{force = false}> : (tensor<1x14x1024x14xbf16, #ttnn_layout3>) -> ()
    %5 = ""ttnn.reshape""(%4) <{shape = [1 : i32, 1 : i32, 196 : i32, 1024 : i32]}> : (tensor<1x14x14x1024xbf16, #ttnn_layout4>) -> tensor<1x1x196x1024xbf16, #ttnn_layout4>
    ""ttnn.deallocate""(%4) <{force = false}> : (tensor<1x14x14x1024xbf16, #ttnn_layout4>) -> ()
    %6 = ""ttnn.empty""(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, #dram, <<7x32>>>, shape = #ttnn.shape<1x1x196x1024>}> : (!tt.device<#device>) -> tensor<1x1x196x1024xbf16, #ttnn_layout4>
    %7 = ""ttnn.conv2d""(%5, %arg1, %6, %0) <{batch_size = 1 : i32, dilation_height = 1 : i32, dilation_width = 1 : i32, groups = 1 : i32, in_channels = 1024 : i32, input_height = 14 : i32, input_width = 14 : i32, kernel_height = 3 : i32, kernel_width = 3 : i32, out_channels = 1024 : i32, padding_height = 1 : i32, padding_width = 1 : i32, stride_height = 1 : i32, stride_width = 1 : i32}> : (tensor<1x1x196x1024xbf16, #ttnn_layout4>, tensor<1024x1024x3x3xbf16, #ttnn_layout1>, tensor<1x1x196x1024xbf16, #ttnn_layout4>, !tt.device<#device>) -> tensor<1x1x196x1024xbf16, #ttnn_layout4>
    ""ttnn.deallocate""(%5) <{force = false}> : (tensor<1x1x196x1024xbf16, #ttnn_layout4>) -> ()
    %8 = ""ttnn.reshape""(%7) <{shape = [1 : i32, 14 : i32, 14 : i32, 1024 : i32]}> : (tensor<1x1x196x1024xbf16, #ttnn_layout4>) -> tensor<1x14x14x1024xbf16, #ttnn_layout4>
    ""ttnn.deallocate""(%6) <{force = false}> : (tensor<1x1x196x1024xbf16, #ttnn_layout4>) -> ()
    %9 = ""ttnn.transpose""(%8) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x14x14x1024xbf16, #ttnn_layout4>) -> tensor<1x14x1024x14xbf16, #ttnn_layout3>
    ""ttnn.deallocate""(%8) <{force = false}> : (tensor<1x14x14x1024xbf16, #ttnn_layout4>) -> ()
    %10 = ""ttnn.transpose""(%9) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x14x1024x14xbf16, #ttnn_layout3>) -> tensor<1x1024x14x14xbf16, #ttnn_layout2>
    ""ttnn.deallocate""(%9) <{force = false}> : (tensor<1x14x1024x14xbf16, #ttnn_layout3>) -> ()
    %11 = ""ttnn.from_device""(%10) : (tensor<1x1024x14x14xbf16, #ttnn_layout2>) -> tensor<1x1024x14x14xbf16, #ttnn_layout>
    ""ttnn.deallocate""(%10) <{force = false}> : (tensor<1x1024x14x14xbf16, #ttnn_layout2>) -> ()
    %12 = ""ttnn.to_layout""(%11) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1024x14x14xbf16, #ttnn_layout>) -> tensor<1x1024x14x14xbf16, #ttnn_layout>
    ""ttnn.deallocate""(%11) <{force = false}> : (tensor<1x1024x14x14xbf16, #ttnn_layout>) -> ()
    return %12 : tensor<1x1024x14x14xbf16, #ttnn_layout>
  }
}
```
Each line that starts with #number refers to an operation. The parser parses through all TTNN graphs generated by models under tt-torch and compiles all ops by the same name together.

## Name

The name of the operation, i.e. _ttnn.add, ttnn.matmul_

## Input/ Output Shapes

The shapes of the input/ output arguments to the operation, last element is the data type (i.e. bf16, i32)

Note: Some operations take the output as the last input.

## Input/ Output Layouts

Please refer to [tt-mlir tensor layout documentation](https://docs.tenstorrent.com/tt-mlir/specs/tensor-layout.html)

### Mapping From/ To

i.e. (d0, d1, d2, d3) -> (d0 * 3072 + d1 * 3 + d2, d3)

### Memory Config

i.e. <448x1x!tt.tile<32x32, bf16>, #dram>

- "tile" refers to tilized memory
- "dram" refers to dram memory
- "system_memory" refers to unformatted weight tensor on host
- "interleaved" refers to interleaved memory

## Attributes

Parameters passed into the operation.

## Runs on TTNN

Yes / No/ N/A

## PCC

Pearson's correlation coefficient

## ATOL

The tolerance on absolute differences
