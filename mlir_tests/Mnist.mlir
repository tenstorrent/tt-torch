module {
  func.func @main(%arg0: tensor<1x1x28x28xbf16>, %arg1: tensor<32x1x3x3xbf16>, %arg2: tensor<32xbf16>, %arg3: tensor<64x32x3x3xbf16>, %arg4: tensor<64xbf16>, %arg5: tensor<9216x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128x10xf32>, %arg8: tensor<10xf32>) -> tensor<1x10xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x32x26x26xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x24x24xbf16>
    %cst_1 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<1x128xbf16>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_5 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x28x28xbf16>, tensor<32x1x3x3xbf16>) -> tensor<1x32x26x26xbf16>
    %1 = stablehlo.reshape %arg2 : (tensor<32xbf16>) -> tensor<32x1x1xbf16>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x32x26x26xbf16>) -> tensor<1x32x26x26xbf16>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<32x1x1xbf16>) -> tensor<1x32x26x26xbf16>
    %4 = stablehlo.add %2, %3 : tensor<1x32x26x26xbf16>
    %5 = stablehlo.maximum %4, %cst : tensor<1x32x26x26xbf16>
    %6 = stablehlo.convolution(%5, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x26x26xbf16>, tensor<64x32x3x3xbf16>) -> tensor<1x64x24x24xbf16>
    %7 = stablehlo.reshape %arg4 : (tensor<64xbf16>) -> tensor<64x1x1xbf16>
    %8 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<1x64x24x24xbf16>) -> tensor<1x64x24x24xbf16>
    %9 = stablehlo.broadcast_in_dim %7, dims = [1, 2, 3] : (tensor<64x1x1xbf16>) -> tensor<1x64x24x24xbf16>
    %10 = stablehlo.add %8, %9 : tensor<1x64x24x24xbf16>
    %11 = stablehlo.maximum %10, %cst_0 : tensor<1x64x24x24xbf16>
    %12 = "stablehlo.reduce_window"(%11, %cst_1) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg9: tensor<bf16>, %arg10: tensor<bf16>):
      %49 = stablehlo.maximum %arg9, %arg10 : tensor<bf16>
      stablehlo.return %49 : tensor<bf16>
    }) : (tensor<1x64x24x24xbf16>, tensor<bf16>) -> tensor<1x64x12x12xbf16>
    %13 = stablehlo.reshape %12 : (tensor<1x64x12x12xbf16>) -> tensor<1x9216xbf16>
    %14 = stablehlo.convert %13 : (tensor<1x9216xbf16>) -> tensor<1x9216xf32>
    %15 = stablehlo.dot_general %14, %arg5, contracting_dims = [1] x [0] : (tensor<1x9216xf32>, tensor<9216x128xf32>) -> tensor<1x128xf32>
    %16 = stablehlo.convert %cst_5 : (tensor<1xi64>) -> tensor<1xf32>
    %17 = stablehlo.reshape %16 : (tensor<1xf32>) -> tensor<f32>
    %18 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %19 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<1x128xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<1x128xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %22 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %23 = stablehlo.add %21, %22 : tensor<1x128xf32>
    %24 = stablehlo.convert %23 : (tensor<1x128xf32>) -> tensor<1x128xbf16>
    %25 = stablehlo.maximum %24, %cst_2 : tensor<1x128xbf16>
    %26 = stablehlo.convert %25 : (tensor<1x128xbf16>) -> tensor<1x128xf32>
    %27 = stablehlo.dot_general %26, %arg7, contracting_dims = [1] x [0] : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %29 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<1x10xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<1x10xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %32 = stablehlo.broadcast_in_dim %arg8, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
    %33 = stablehlo.add %31, %32 : tensor<1x10xf32>
    %34 = stablehlo.convert %33 : (tensor<1x10xf32>) -> tensor<1x10xbf16>
    %35 = stablehlo.convert %34 : (tensor<1x10xbf16>) -> tensor<1x10xf32>
    %36 = stablehlo.reduce(%35 init: %cst_3) applies stablehlo.maximum across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %37 = stablehlo.reshape %36 : (tensor<1xf32>) -> tensor<1x1xf32>
    %38 = stablehlo.broadcast_in_dim %35, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x10xf32>
    %40 = stablehlo.subtract %38, %39 : tensor<1x10xf32>
    %41 = stablehlo.exponential %40 : tensor<1x10xf32>
    %42 = stablehlo.reduce(%41 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %43 = stablehlo.reshape %42 : (tensor<1xf32>) -> tensor<1x1xf32>
    %44 = stablehlo.log %43 : tensor<1x1xf32>
    %45 = stablehlo.broadcast_in_dim %40, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x10xf32>
    %47 = stablehlo.subtract %45, %46 : tensor<1x10xf32>
    %48 = stablehlo.convert %47 : (tensor<1x10xf32>) -> tensor<1x10xbf16>
    return %48 : tensor<1x10xbf16>
  }
}
