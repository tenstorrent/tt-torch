module {
  func.func @main(%arg0: tensor<1x784xbf16>, %arg1: tensor<784x128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64x12xf32>, %arg6: tensor<12xf32>, %arg7: tensor<12x3xf32>, %arg8: tensor<3xf32>, %arg9: tensor<3x12xf32>, %arg10: tensor<12xf32>, %arg11: tensor<12x64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64x128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x784xf32>, %arg16: tensor<784xf32>) -> tensor<1x784xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x128xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x64xbf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<1x12xbf16>
    %cst_2 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convert %arg0 : (tensor<1x784xbf16>) -> tensor<1x784xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [0] : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.convert %cst_2 : (tensor<1xi64>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %5 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1x128xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x128xf32>
    %10 = stablehlo.convert %9 : (tensor<1x128xf32>) -> tensor<1x128xbf16>
    %11 = stablehlo.maximum %10, %cst : tensor<1x128xbf16>
    %12 = stablehlo.convert %11 : (tensor<1x128xbf16>) -> tensor<1x128xf32>
    %13 = stablehlo.dot_general %12, %arg3, contracting_dims = [1] x [0] : (tensor<1x128xf32>, tensor<128x64xf32>) -> tensor<1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<1x64xf32>
    %15 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<1x64xf32>
    %18 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x64xf32>
    %20 = stablehlo.convert %19 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %21 = stablehlo.maximum %20, %cst_0 : tensor<1x64xbf16>
    %22 = stablehlo.convert %21 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %23 = stablehlo.dot_general %22, %arg5, contracting_dims = [1] x [0] : (tensor<1x64xf32>, tensor<64x12xf32>) -> tensor<1x12xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x12xf32>) -> tensor<1x12xf32>
    %25 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x12xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<1x12xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x12xf32>) -> tensor<1x12xf32>
    %28 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<12xf32>) -> tensor<1x12xf32>
    %29 = stablehlo.add %27, %28 : tensor<1x12xf32>
    %30 = stablehlo.convert %29 : (tensor<1x12xf32>) -> tensor<1x12xbf16>
    %31 = stablehlo.maximum %30, %cst_1 : tensor<1x12xbf16>
    %32 = stablehlo.convert %31 : (tensor<1x12xbf16>) -> tensor<1x12xf32>
    %33 = stablehlo.dot_general %32, %arg7, contracting_dims = [1] x [0] : (tensor<1x12xf32>, tensor<12x3xf32>) -> tensor<1x3xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %35 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x3xf32>
    %36 = stablehlo.multiply %34, %35 : tensor<1x3xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %38 = stablehlo.broadcast_in_dim %arg8, dims = [1] : (tensor<3xf32>) -> tensor<1x3xf32>
    %39 = stablehlo.add %37, %38 : tensor<1x3xf32>
    %40 = stablehlo.convert %39 : (tensor<1x3xf32>) -> tensor<1x3xbf16>
    %41 = stablehlo.convert %40 : (tensor<1x3xbf16>) -> tensor<1x3xf32>
    %42 = stablehlo.dot_general %41, %arg9, contracting_dims = [1] x [0] : (tensor<1x3xf32>, tensor<3x12xf32>) -> tensor<1x12xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1] : (tensor<1x12xf32>) -> tensor<1x12xf32>
    %44 = stablehlo.multiply %43, %25 : tensor<1x12xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<1x12xf32>) -> tensor<1x12xf32>
    %46 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<12xf32>) -> tensor<1x12xf32>
    %47 = stablehlo.add %45, %46 : tensor<1x12xf32>
    %48 = stablehlo.convert %47 : (tensor<1x12xf32>) -> tensor<1x12xbf16>
    %49 = stablehlo.maximum %48, %cst_1 : tensor<1x12xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x12xbf16>) -> tensor<1x12xf32>
    %51 = stablehlo.dot_general %50, %arg11, contracting_dims = [1] x [0] : (tensor<1x12xf32>, tensor<12x64xf32>) -> tensor<1x64xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<1x64xf32>
    %53 = stablehlo.multiply %52, %15 : tensor<1x64xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<1x64xf32>
    %55 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %56 = stablehlo.add %54, %55 : tensor<1x64xf32>
    %57 = stablehlo.convert %56 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %58 = stablehlo.maximum %57, %cst_0 : tensor<1x64xbf16>
    %59 = stablehlo.convert %58 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %60 = stablehlo.dot_general %59, %arg13, contracting_dims = [1] x [0] : (tensor<1x64xf32>, tensor<64x128xf32>) -> tensor<1x128xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %62 = stablehlo.multiply %61, %5 : tensor<1x128xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %64 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %65 = stablehlo.add %63, %64 : tensor<1x128xf32>
    %66 = stablehlo.convert %65 : (tensor<1x128xf32>) -> tensor<1x128xbf16>
    %67 = stablehlo.maximum %66, %cst : tensor<1x128xbf16>
    %68 = stablehlo.convert %67 : (tensor<1x128xbf16>) -> tensor<1x128xf32>
    %69 = stablehlo.dot_general %68, %arg15, contracting_dims = [1] x [0] : (tensor<1x128xf32>, tensor<128x784xf32>) -> tensor<1x784xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784xf32>
    %71 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x784xf32>
    %72 = stablehlo.multiply %70, %71 : tensor<1x784xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784xf32>
    %74 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %75 = stablehlo.add %73, %74 : tensor<1x784xf32>
    %76 = stablehlo.convert %75 : (tensor<1x784xf32>) -> tensor<1x784xbf16>
    return %76 : tensor<1x784xbf16>
  }
}
