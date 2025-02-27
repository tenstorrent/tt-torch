module {
  func.func @main(%arg0: tensor<1x8xi64>, %arg1: tensor<1x8xi64>, %arg2: tensor<1x8xi64>, %arg3: tensor<30528x768xbf16>, %arg4: tensor<2x768xbf16>, %arg5: tensor<768xbf16>, %arg6: tensor<768xbf16>, %arg7: tensor<1x8x768xbf16>) -> (tensor<1x8x768xbf16>, tensor<1x1x1x8xbf16>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<1xf64>
    %cst_2 = arith.constant dense<-3.3895313892515355E+38> : tensor<1xf64>
    %cst_3 = arith.constant dense<768> : tensor<1xi64>
    %cst_4 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
    %0 = stablehlo.reshape %arg1 : (tensor<1x8xi64>) -> tensor<1x1x8xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x1x8xi64>) -> tensor<1x1x1x8xi64>
    %2 = stablehlo.convert %1 : (tensor<1x1x1x8xi64>) -> tensor<1x1x1x8xbf16>
    %3 = stablehlo.convert %cst_1 : (tensor<1xf64>) -> tensor<1xbf16>
    %4 = stablehlo.reshape %3 : (tensor<1xbf16>) -> tensor<bf16>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<bf16>) -> tensor<1x1x1x8xbf16>
    %6 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
    %7 = stablehlo.subtract %5, %6 : tensor<1x1x1x8xbf16>
    %8 = stablehlo.convert %cst_2 : (tensor<1xf64>) -> tensor<1xbf16>
    %9 = stablehlo.reshape %8 : (tensor<1xbf16>) -> tensor<bf16>
    %10 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
    %11 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<bf16>) -> tensor<1x1x1x8xbf16>
    %12 = stablehlo.multiply %10, %11 : tensor<1x1x1x8xbf16>
    %13 = "stablehlo.gather"(%arg3, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<30528x768xbf16>, tensor<1x8xi64>) -> tensor<1x8x768xbf16>
    %14 = stablehlo.convert %13 : tensor<1x8x768xbf16>
    %15 = "stablehlo.gather"(%arg4, %arg2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<2x768xbf16>, tensor<1x8xi64>) -> tensor<1x8x768xbf16>
    %16 = stablehlo.convert %15 : tensor<1x8x768xbf16>
    %17 = stablehlo.add %14, %arg7 : tensor<1x8x768xbf16>
    %18 = stablehlo.add %17, %16 : tensor<1x8x768xbf16>
    %19 = stablehlo.convert %18 : (tensor<1x8x768xbf16>) -> tensor<1x8x768xf32>
    %20 = stablehlo.convert %19 : (tensor<1x8x768xf32>) -> tensor<1x8x768xf64>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x8x768xf64>, tensor<f64>) -> tensor<1x8xf64>
    %22 = stablehlo.reshape %21 : (tensor<1x8xf64>) -> tensor<1x8x1xf64>
    %23 = stablehlo.convert %cst_3 : (tensor<1xi64>) -> tensor<1xf64>
    %24 = stablehlo.reshape %23 : (tensor<1xf64>) -> tensor<f64>
    %25 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x1xf64>
    %26 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f64>) -> tensor<1x8x1xf64>
    %27 = stablehlo.divide %25, %26 : tensor<1x8x1xf64>
    %28 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x8x768xf64>) -> tensor<1x8x768xf64>
    %29 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x768xf64>
    %30 = stablehlo.subtract %28, %29 : tensor<1x8x768xf64>
    %31 = stablehlo.multiply %30, %30 : tensor<1x8x768xf64>
    %32 = stablehlo.reduce(%31 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x8x768xf64>, tensor<f64>) -> tensor<1x8xf64>
    %33 = stablehlo.reshape %32 : (tensor<1x8xf64>) -> tensor<1x8x1xf64>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x1xf64>
    %35 = stablehlo.divide %34, %26 : tensor<1x8x1xf64>
    %36 = stablehlo.convert %35 : (tensor<1x8x1xf64>) -> tensor<1x8x1xf32>
    %37 = stablehlo.reduce(%19 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x8x768xf32>, tensor<f32>) -> tensor<1x8xf32>
    %38 = stablehlo.reshape %37 : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %39 = stablehlo.convert %cst_3 : (tensor<1xi64>) -> tensor<1xf32>
    %40 = stablehlo.reshape %39 : (tensor<1xf32>) -> tensor<f32>
    %41 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    %42 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %43 = stablehlo.divide %41, %42 : tensor<1x8x1xf32>
    %44 = stablehlo.convert %cst_4 : (tensor<1xf64>) -> tensor<1xf32>
    %45 = stablehlo.reshape %44 : (tensor<1xf32>) -> tensor<f32>
    %46 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x8x1xf32>
    %49 = stablehlo.rsqrt %48 : tensor<1x8x1xf32>
    %50 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %51 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x768xf32>
    %52 = stablehlo.subtract %50, %51 : tensor<1x8x768xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %54 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x768xf32>
    %55 = stablehlo.multiply %53, %54 : tensor<1x8x768xf32>
    %56 = stablehlo.convert %arg5 : (tensor<768xbf16>) -> tensor<768xf32>
    %57 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %58 = stablehlo.broadcast_in_dim %56, dims = [2] : (tensor<768xf32>) -> tensor<1x8x768xf32>
    %59 = stablehlo.multiply %57, %58 : tensor<1x8x768xf32>
    %60 = stablehlo.convert %arg6 : (tensor<768xbf16>) -> tensor<768xf32>
    %61 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2] : (tensor<1x8x768xf32>) -> tensor<1x8x768xf32>
    %62 = stablehlo.broadcast_in_dim %60, dims = [2] : (tensor<768xf32>) -> tensor<1x8x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x8x768xf32>
    %64 = stablehlo.convert %63 : (tensor<1x8x768xf32>) -> tensor<1x8x768xbf16>
    return %64, %12 : tensor<1x8x768xbf16>, tensor<1x1x1x8xbf16>
  }
}
