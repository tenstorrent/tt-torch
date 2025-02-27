module {
  func.func @main(%arg0: tensor<1x8xi64>, %arg1: tensor<1x8xi64>, %arg2: tensor<30522x768xbf16>, %arg3: tensor<2x768xbf16>, %arg4: tensor<768xbf16>, %arg5: tensor<768xbf16>, %arg6: tensor<1x8x768xbf16>) -> tensor<1x8x768xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<768> : tensor<1xi64>
    %cst_2 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg2, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<30522x768xbf16>, tensor<1x8xi64>) -> tensor<1x8x768xbf16>
    %1 = stablehlo.convert %0 : tensor<1x8x768xbf16>
    %2 = "stablehlo.gather"(%arg3, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<2x768xbf16>, tensor<1x8xi64>) -> tensor<1x8x768xbf16>
    %3 = stablehlo.convert %2 : tensor<1x8x768xbf16>
    %4 = stablehlo.add %1, %3 : tensor<1x8x768xbf16>
    %5 = stablehlo.add %4, %arg6 : tensor<1x8x768xbf16>
    %6 = stablehlo.convert %5 : (tensor<1x8x768xbf16>) -> tensor<1x8x768xf32>
    %7 = stablehlo.convert %6 : (tensor<1x8x768xf32>) -> tensor<1x8x768xf64>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x8x768xf64>, tensor<f64>) -> tensor<1x8xf64>
    %9 = stablehlo.reshape %8 : (tensor<1x8xf64>) -> tensor<1x8x1xf64>
    %10 = stablehlo.convert %cst_1 : (tensor<1xi64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x1xf64>
    %13 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f64>) -> tensor<1x8x1xf64>
    %14 = stablehlo.divide %12, %13 : tensor<1x8x1xf64>
    %15 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x8x768xf64>) -> tensor<1x8x768xf64>
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x768xf64>
    %17 = stablehlo.subtract %15, %16 : tensor<1x8x768xf64>
    %18 = stablehlo.multiply %17, %17 : tensor<1x8x768xf64>
    %19 = stablehlo.reduce(%18 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x8x768xf64>, tensor<f64>) -> tensor<1x8xf64>
    %20 = stablehlo.reshape %19 : (tensor<1x8xf64>) -> tensor<1x8x1xf64>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x1xf64>
    %22 = stablehlo.divide %21, %13 : tensor<1x8x1xf64>
    %23 = stablehlo.convert %22 : (tensor<1x8x1xf64>) -> tensor<1x8x1xf32>
    %24 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x8x768xf32>, tensor<f32>) -> tensor<1x8xf32>
    %25 = stablehlo.reshape %24 : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %26 = stablehlo.convert %cst_1 : (tensor<1xi64>) -> tensor<1xf32>
    %27 = stablehlo.reshape %26 : (tensor<1xf32>) -> tensor<f32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    %29 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %30 = stablehlo.divide %28, %29 : tensor<1x8x1xf32>
    %31 = stablehlo.convert %cst_2 : (tensor<1xf64>) -> tensor<1xf32>
    %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
    %33 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %35 = stablehlo.add %33, %34 : tensor<1x8x1xf32>
    %36 = stablehlo.rsqrt %35 : tensor<1x8x1xf32>
    %37 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %38 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x768xf32>
    %39 = stablehlo.subtract %37, %38 : tensor<1x8x768xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %41 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x8x768xf32>
    %43 = stablehlo.convert %arg4 : (tensor<768xbf16>) -> tensor<768xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [2] : (tensor<768xf32>) -> tensor<1x8x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x8x768xf32>
    %47 = stablehlo.convert %arg5 : (tensor<768xbf16>) -> tensor<768xf32>
    %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [2] : (tensor<768xf32>) -> tensor<1x8x768xf32>
    %50 = stablehlo.add %48, %49 : tensor<1x8x768xf32>
    %51 = stablehlo.convert %50 : (tensor<1x8x768xf32>) -> tensor<1x8x768xbf16>
    return %51 : tensor<1x8x768xbf16>
  }
}
