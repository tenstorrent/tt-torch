module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16>, %arg2: tensor<64x64x3x3xbf16>, %arg3: tensor<64x64x3x3xbf16>, %arg4: tensor<64x64x3x3xbf16>, %arg5: tensor<64x64x3x3xbf16>, %arg6: tensor<128x64x3x3xbf16>, %arg7: tensor<128x128x3x3xbf16>, %arg8: tensor<128x64x1x1xbf16>, %arg9: tensor<128x128x3x3xbf16>, %arg10: tensor<128x128x3x3xbf16>, %arg11: tensor<256x128x3x3xbf16>, %arg12: tensor<256x256x3x3xbf16>, %arg13: tensor<256x128x1x1xbf16>, %arg14: tensor<256x256x3x3xbf16>, %arg15: tensor<256x256x3x3xbf16>, %arg16: tensor<512x256x3x3xbf16>, %arg17: tensor<512x512x3x3xbf16>, %arg18: tensor<512x256x1x1xbf16>, %arg19: tensor<512x512x3x3xbf16>, %arg20: tensor<512x512x3x3xbf16>, %arg21: tensor<64x1x1xf32>, %arg22: tensor<64x1x1xf32>, %arg23: tensor<64x1x1xbf16>, %arg24: tensor<64x1x1xbf16>, %arg25: tensor<64x1x1xf32>, %arg26: tensor<64x1x1xf32>, %arg27: tensor<64x1x1xbf16>, %arg28: tensor<64x1x1xbf16>, %arg29: tensor<64x1x1xf32>, %arg30: tensor<64x1x1xf32>, %arg31: tensor<64x1x1xbf16>, %arg32: tensor<64x1x1xbf16>, %arg33: tensor<64x1x1xf32>, %arg34: tensor<64x1x1xf32>, %arg35: tensor<64x1x1xbf16>, %arg36: tensor<64x1x1xbf16>, %arg37: tensor<64x1x1xf32>, %arg38: tensor<64x1x1xf32>, %arg39: tensor<64x1x1xbf16>, %arg40: tensor<64x1x1xbf16>, %arg41: tensor<128x1x1xf32>, %arg42: tensor<128x1x1xf32>, %arg43: tensor<128x1x1xbf16>, %arg44: tensor<128x1x1xbf16>, %arg45: tensor<128x1x1xf32>, %arg46: tensor<128x1x1xf32>, %arg47: tensor<128x1x1xbf16>, %arg48: tensor<128x1x1xbf16>, %arg49: tensor<128x1x1xf32>, %arg50: tensor<128x1x1xf32>, %arg51: tensor<128x1x1xbf16>, %arg52: tensor<128x1x1xbf16>, %arg53: tensor<128x1x1xf32>, %arg54: tensor<128x1x1xf32>, %arg55: tensor<128x1x1xbf16>, %arg56: tensor<128x1x1xbf16>, %arg57: tensor<128x1x1xf32>, %arg58: tensor<128x1x1xf32>, %arg59: tensor<128x1x1xbf16>, %arg60: tensor<128x1x1xbf16>, %arg61: tensor<256x1x1xf32>, %arg62: tensor<256x1x1xf32>, %arg63: tensor<256x1x1xbf16>, %arg64: tensor<256x1x1xbf16>, %arg65: tensor<256x1x1xf32>, %arg66: tensor<256x1x1xf32>, %arg67: tensor<256x1x1xbf16>, %arg68: tensor<256x1x1xbf16>, %arg69: tensor<256x1x1xf32>, %arg70: tensor<256x1x1xf32>, %arg71: tensor<256x1x1xbf16>, %arg72: tensor<256x1x1xbf16>, %arg73: tensor<256x1x1xf32>, %arg74: tensor<256x1x1xf32>, %arg75: tensor<256x1x1xbf16>, %arg76: tensor<256x1x1xbf16>, %arg77: tensor<256x1x1xf32>, %arg78: tensor<256x1x1xf32>, %arg79: tensor<256x1x1xbf16>, %arg80: tensor<256x1x1xbf16>, %arg81: tensor<512x1x1xf32>, %arg82: tensor<512x1x1xf32>, %arg83: tensor<512x1x1xbf16>, %arg84: tensor<512x1x1xbf16>, %arg85: tensor<512x1x1xf32>, %arg86: tensor<512x1x1xf32>, %arg87: tensor<512x1x1xbf16>, %arg88: tensor<512x1x1xbf16>, %arg89: tensor<512x1x1xf32>, %arg90: tensor<512x1x1xf32>, %arg91: tensor<512x1x1xbf16>, %arg92: tensor<512x1x1xbf16>, %arg93: tensor<512x1x1xf32>, %arg94: tensor<512x1x1xf32>, %arg95: tensor<512x1x1xbf16>, %arg96: tensor<512x1x1xbf16>, %arg97: tensor<512x1x1xf32>, %arg98: tensor<512x1x1xf32>, %arg99: tensor<512x1x1xbf16>, %arg100: tensor<512x1x1xbf16>, %arg101: tensor<512x1000xf32>, %arg102: tensor<1000xf32>) -> tensor<1x1000xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x64x112x112xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x56x56xbf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x28x28xbf16>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x14x14xbf16>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x7x7xbf16>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_6 = arith.constant dense<49> : tensor<1xi64>
    %cst_7 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x112x112xbf16>
    %1 = stablehlo.convert %0 : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = stablehlo.broadcast_in_dim %arg21, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %4 = stablehlo.subtract %2, %3 : tensor<1x64x112x112xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg22, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x64x112x112xf32>
    %8 = stablehlo.convert %arg23 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x64x112x112xf32>
    %12 = stablehlo.convert %arg24 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %15 = stablehlo.add %13, %14 : tensor<1x64x112x112xf32>
    %16 = stablehlo.convert %15 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xbf16>
    %17 = stablehlo.maximum %16, %cst : tensor<1x64x112x112xbf16>
    %18 = "stablehlo.reduce_window"(%17, %cst_0) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg103: tensor<bf16>, %arg104: tensor<bf16>):
      %385 = stablehlo.maximum %arg103, %arg104 : tensor<bf16>
      stablehlo.return %385 : tensor<bf16>
    }) : (tensor<1x64x112x112xbf16>, tensor<bf16>) -> tensor<1x64x56x56xbf16>
    %19 = stablehlo.convolution(%18, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %20 = stablehlo.convert %19 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %22 = stablehlo.broadcast_in_dim %arg25, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %23 = stablehlo.subtract %21, %22 : tensor<1x64x56x56xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %25 = stablehlo.broadcast_in_dim %arg26, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<1x64x56x56xf32>
    %27 = stablehlo.convert %arg27 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %28 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %29 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<1x64x56x56xf32>
    %31 = stablehlo.convert %arg28 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %33 = stablehlo.broadcast_in_dim %31, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %34 = stablehlo.add %32, %33 : tensor<1x64x56x56xf32>
    %35 = stablehlo.convert %34 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %36 = stablehlo.maximum %35, %cst_1 : tensor<1x64x56x56xbf16>
    %37 = stablehlo.convolution(%36, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %38 = stablehlo.convert %37 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %40 = stablehlo.broadcast_in_dim %arg29, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %41 = stablehlo.subtract %39, %40 : tensor<1x64x56x56xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %43 = stablehlo.broadcast_in_dim %arg30, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<1x64x56x56xf32>
    %45 = stablehlo.convert %arg31 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %48 = stablehlo.multiply %46, %47 : tensor<1x64x56x56xf32>
    %49 = stablehlo.convert %arg32 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %51 = stablehlo.broadcast_in_dim %49, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x64x56x56xf32>
    %53 = stablehlo.convert %52 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %54 = stablehlo.add %53, %18 : tensor<1x64x56x56xbf16>
    %55 = stablehlo.maximum %54, %cst_1 : tensor<1x64x56x56xbf16>
    %56 = stablehlo.convolution(%55, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %57 = stablehlo.convert %56 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %59 = stablehlo.broadcast_in_dim %arg33, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %60 = stablehlo.subtract %58, %59 : tensor<1x64x56x56xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %62 = stablehlo.broadcast_in_dim %arg34, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %63 = stablehlo.multiply %61, %62 : tensor<1x64x56x56xf32>
    %64 = stablehlo.convert %arg35 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %65 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %66 = stablehlo.broadcast_in_dim %64, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %67 = stablehlo.multiply %65, %66 : tensor<1x64x56x56xf32>
    %68 = stablehlo.convert %arg36 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %69 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %70 = stablehlo.broadcast_in_dim %68, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %71 = stablehlo.add %69, %70 : tensor<1x64x56x56xf32>
    %72 = stablehlo.convert %71 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %73 = stablehlo.maximum %72, %cst_1 : tensor<1x64x56x56xbf16>
    %74 = stablehlo.convolution(%73, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %75 = stablehlo.convert %74 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %77 = stablehlo.broadcast_in_dim %arg37, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %78 = stablehlo.subtract %76, %77 : tensor<1x64x56x56xf32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %80 = stablehlo.broadcast_in_dim %arg38, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %81 = stablehlo.multiply %79, %80 : tensor<1x64x56x56xf32>
    %82 = stablehlo.convert %arg39 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %83 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %84 = stablehlo.broadcast_in_dim %82, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %85 = stablehlo.multiply %83, %84 : tensor<1x64x56x56xf32>
    %86 = stablehlo.convert %arg40 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %87 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %88 = stablehlo.broadcast_in_dim %86, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %89 = stablehlo.add %87, %88 : tensor<1x64x56x56xf32>
    %90 = stablehlo.convert %89 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %91 = stablehlo.add %90, %55 : tensor<1x64x56x56xbf16>
    %92 = stablehlo.maximum %91, %cst_1 : tensor<1x64x56x56xbf16>
    %93 = stablehlo.convolution(%92, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<128x64x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %94 = stablehlo.convert %93 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %96 = stablehlo.broadcast_in_dim %arg41, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %97 = stablehlo.subtract %95, %96 : tensor<1x128x28x28xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %99 = stablehlo.broadcast_in_dim %arg42, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<1x128x28x28xf32>
    %101 = stablehlo.convert %arg43 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %102 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %103 = stablehlo.broadcast_in_dim %101, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %104 = stablehlo.multiply %102, %103 : tensor<1x128x28x28xf32>
    %105 = stablehlo.convert %arg44 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %106 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %107 = stablehlo.broadcast_in_dim %105, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %108 = stablehlo.add %106, %107 : tensor<1x128x28x28xf32>
    %109 = stablehlo.convert %108 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %110 = stablehlo.maximum %109, %cst_2 : tensor<1x128x28x28xbf16>
    %111 = stablehlo.convolution(%110, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %112 = stablehlo.convert %111 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %114 = stablehlo.broadcast_in_dim %arg45, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %115 = stablehlo.subtract %113, %114 : tensor<1x128x28x28xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %117 = stablehlo.broadcast_in_dim %arg46, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %118 = stablehlo.multiply %116, %117 : tensor<1x128x28x28xf32>
    %119 = stablehlo.convert %arg47 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %120 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %121 = stablehlo.broadcast_in_dim %119, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %122 = stablehlo.multiply %120, %121 : tensor<1x128x28x28xf32>
    %123 = stablehlo.convert %arg48 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %124 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %125 = stablehlo.broadcast_in_dim %123, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %126 = stablehlo.add %124, %125 : tensor<1x128x28x28xf32>
    %127 = stablehlo.convert %126 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %128 = stablehlo.convolution(%92, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<128x64x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %129 = stablehlo.convert %128 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %131 = stablehlo.broadcast_in_dim %arg49, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %132 = stablehlo.subtract %130, %131 : tensor<1x128x28x28xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %134 = stablehlo.broadcast_in_dim %arg50, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %135 = stablehlo.multiply %133, %134 : tensor<1x128x28x28xf32>
    %136 = stablehlo.convert %arg51 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %137 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %138 = stablehlo.broadcast_in_dim %136, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %139 = stablehlo.multiply %137, %138 : tensor<1x128x28x28xf32>
    %140 = stablehlo.convert %arg52 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %141 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %142 = stablehlo.broadcast_in_dim %140, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %143 = stablehlo.add %141, %142 : tensor<1x128x28x28xf32>
    %144 = stablehlo.convert %143 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %145 = stablehlo.add %127, %144 : tensor<1x128x28x28xbf16>
    %146 = stablehlo.maximum %145, %cst_2 : tensor<1x128x28x28xbf16>
    %147 = stablehlo.convolution(%146, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %148 = stablehlo.convert %147 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %150 = stablehlo.broadcast_in_dim %arg53, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %151 = stablehlo.subtract %149, %150 : tensor<1x128x28x28xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %153 = stablehlo.broadcast_in_dim %arg54, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %154 = stablehlo.multiply %152, %153 : tensor<1x128x28x28xf32>
    %155 = stablehlo.convert %arg55 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %156 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %157 = stablehlo.broadcast_in_dim %155, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %158 = stablehlo.multiply %156, %157 : tensor<1x128x28x28xf32>
    %159 = stablehlo.convert %arg56 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %160 = stablehlo.broadcast_in_dim %158, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %161 = stablehlo.broadcast_in_dim %159, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %162 = stablehlo.add %160, %161 : tensor<1x128x28x28xf32>
    %163 = stablehlo.convert %162 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %164 = stablehlo.maximum %163, %cst_2 : tensor<1x128x28x28xbf16>
    %165 = stablehlo.convolution(%164, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %166 = stablehlo.convert %165 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %168 = stablehlo.broadcast_in_dim %arg57, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %169 = stablehlo.subtract %167, %168 : tensor<1x128x28x28xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %171 = stablehlo.broadcast_in_dim %arg58, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %172 = stablehlo.multiply %170, %171 : tensor<1x128x28x28xf32>
    %173 = stablehlo.convert %arg59 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %174 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %175 = stablehlo.broadcast_in_dim %173, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %176 = stablehlo.multiply %174, %175 : tensor<1x128x28x28xf32>
    %177 = stablehlo.convert %arg60 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %178 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %179 = stablehlo.broadcast_in_dim %177, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %180 = stablehlo.add %178, %179 : tensor<1x128x28x28xf32>
    %181 = stablehlo.convert %180 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %182 = stablehlo.add %181, %146 : tensor<1x128x28x28xbf16>
    %183 = stablehlo.maximum %182, %cst_2 : tensor<1x128x28x28xbf16>
    %184 = stablehlo.convolution(%183, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %185 = stablehlo.convert %184 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %186 = stablehlo.broadcast_in_dim %185, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %187 = stablehlo.broadcast_in_dim %arg61, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %188 = stablehlo.subtract %186, %187 : tensor<1x256x14x14xf32>
    %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %190 = stablehlo.broadcast_in_dim %arg62, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %191 = stablehlo.multiply %189, %190 : tensor<1x256x14x14xf32>
    %192 = stablehlo.convert %arg63 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %193 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %194 = stablehlo.broadcast_in_dim %192, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %195 = stablehlo.multiply %193, %194 : tensor<1x256x14x14xf32>
    %196 = stablehlo.convert %arg64 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %197 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %198 = stablehlo.broadcast_in_dim %196, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %199 = stablehlo.add %197, %198 : tensor<1x256x14x14xf32>
    %200 = stablehlo.convert %199 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %201 = stablehlo.maximum %200, %cst_3 : tensor<1x256x14x14xbf16>
    %202 = stablehlo.convolution(%201, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %203 = stablehlo.convert %202 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %205 = stablehlo.broadcast_in_dim %arg65, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %206 = stablehlo.subtract %204, %205 : tensor<1x256x14x14xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %208 = stablehlo.broadcast_in_dim %arg66, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %209 = stablehlo.multiply %207, %208 : tensor<1x256x14x14xf32>
    %210 = stablehlo.convert %arg67 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %211 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %212 = stablehlo.broadcast_in_dim %210, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %213 = stablehlo.multiply %211, %212 : tensor<1x256x14x14xf32>
    %214 = stablehlo.convert %arg68 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %215 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %216 = stablehlo.broadcast_in_dim %214, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %217 = stablehlo.add %215, %216 : tensor<1x256x14x14xf32>
    %218 = stablehlo.convert %217 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %219 = stablehlo.convolution(%183, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<256x128x1x1xbf16>) -> tensor<1x256x14x14xbf16>
    %220 = stablehlo.convert %219 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %222 = stablehlo.broadcast_in_dim %arg69, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %223 = stablehlo.subtract %221, %222 : tensor<1x256x14x14xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %225 = stablehlo.broadcast_in_dim %arg70, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<1x256x14x14xf32>
    %227 = stablehlo.convert %arg71 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %228 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %229 = stablehlo.broadcast_in_dim %227, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %230 = stablehlo.multiply %228, %229 : tensor<1x256x14x14xf32>
    %231 = stablehlo.convert %arg72 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %232 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %233 = stablehlo.broadcast_in_dim %231, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %234 = stablehlo.add %232, %233 : tensor<1x256x14x14xf32>
    %235 = stablehlo.convert %234 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %236 = stablehlo.add %218, %235 : tensor<1x256x14x14xbf16>
    %237 = stablehlo.maximum %236, %cst_3 : tensor<1x256x14x14xbf16>
    %238 = stablehlo.convolution(%237, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %239 = stablehlo.convert %238 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %241 = stablehlo.broadcast_in_dim %arg73, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %242 = stablehlo.subtract %240, %241 : tensor<1x256x14x14xf32>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %244 = stablehlo.broadcast_in_dim %arg74, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x256x14x14xf32>
    %246 = stablehlo.convert %arg75 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %247 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %248 = stablehlo.broadcast_in_dim %246, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %249 = stablehlo.multiply %247, %248 : tensor<1x256x14x14xf32>
    %250 = stablehlo.convert %arg76 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %251 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %252 = stablehlo.broadcast_in_dim %250, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %253 = stablehlo.add %251, %252 : tensor<1x256x14x14xf32>
    %254 = stablehlo.convert %253 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %255 = stablehlo.maximum %254, %cst_3 : tensor<1x256x14x14xbf16>
    %256 = stablehlo.convolution(%255, %arg15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %257 = stablehlo.convert %256 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %258 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %259 = stablehlo.broadcast_in_dim %arg77, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %260 = stablehlo.subtract %258, %259 : tensor<1x256x14x14xf32>
    %261 = stablehlo.broadcast_in_dim %260, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %262 = stablehlo.broadcast_in_dim %arg78, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %263 = stablehlo.multiply %261, %262 : tensor<1x256x14x14xf32>
    %264 = stablehlo.convert %arg79 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %265 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %266 = stablehlo.broadcast_in_dim %264, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %267 = stablehlo.multiply %265, %266 : tensor<1x256x14x14xf32>
    %268 = stablehlo.convert %arg80 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %269 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %270 = stablehlo.broadcast_in_dim %268, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %271 = stablehlo.add %269, %270 : tensor<1x256x14x14xf32>
    %272 = stablehlo.convert %271 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %273 = stablehlo.add %272, %237 : tensor<1x256x14x14xbf16>
    %274 = stablehlo.maximum %273, %cst_3 : tensor<1x256x14x14xbf16>
    %275 = stablehlo.convolution(%274, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %276 = stablehlo.convert %275 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %277 = stablehlo.broadcast_in_dim %276, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %278 = stablehlo.broadcast_in_dim %arg81, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %279 = stablehlo.subtract %277, %278 : tensor<1x512x7x7xf32>
    %280 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %281 = stablehlo.broadcast_in_dim %arg82, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %282 = stablehlo.multiply %280, %281 : tensor<1x512x7x7xf32>
    %283 = stablehlo.convert %arg83 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %284 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %285 = stablehlo.broadcast_in_dim %283, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %286 = stablehlo.multiply %284, %285 : tensor<1x512x7x7xf32>
    %287 = stablehlo.convert %arg84 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %288 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %289 = stablehlo.broadcast_in_dim %287, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %290 = stablehlo.add %288, %289 : tensor<1x512x7x7xf32>
    %291 = stablehlo.convert %290 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %292 = stablehlo.maximum %291, %cst_4 : tensor<1x512x7x7xbf16>
    %293 = stablehlo.convolution(%292, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %294 = stablehlo.convert %293 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %296 = stablehlo.broadcast_in_dim %arg85, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %297 = stablehlo.subtract %295, %296 : tensor<1x512x7x7xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %299 = stablehlo.broadcast_in_dim %arg86, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %300 = stablehlo.multiply %298, %299 : tensor<1x512x7x7xf32>
    %301 = stablehlo.convert %arg87 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %302 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %303 = stablehlo.broadcast_in_dim %301, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x512x7x7xf32>
    %305 = stablehlo.convert %arg88 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %306 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %307 = stablehlo.broadcast_in_dim %305, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %308 = stablehlo.add %306, %307 : tensor<1x512x7x7xf32>
    %309 = stablehlo.convert %308 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %310 = stablehlo.convolution(%274, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<512x256x1x1xbf16>) -> tensor<1x512x7x7xbf16>
    %311 = stablehlo.convert %310 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %313 = stablehlo.broadcast_in_dim %arg89, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %314 = stablehlo.subtract %312, %313 : tensor<1x512x7x7xf32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %316 = stablehlo.broadcast_in_dim %arg90, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %317 = stablehlo.multiply %315, %316 : tensor<1x512x7x7xf32>
    %318 = stablehlo.convert %arg91 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %319 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %320 = stablehlo.broadcast_in_dim %318, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %321 = stablehlo.multiply %319, %320 : tensor<1x512x7x7xf32>
    %322 = stablehlo.convert %arg92 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %323 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %324 = stablehlo.broadcast_in_dim %322, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %325 = stablehlo.add %323, %324 : tensor<1x512x7x7xf32>
    %326 = stablehlo.convert %325 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %327 = stablehlo.add %309, %326 : tensor<1x512x7x7xbf16>
    %328 = stablehlo.maximum %327, %cst_4 : tensor<1x512x7x7xbf16>
    %329 = stablehlo.convolution(%328, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %330 = stablehlo.convert %329 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %332 = stablehlo.broadcast_in_dim %arg93, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %333 = stablehlo.subtract %331, %332 : tensor<1x512x7x7xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %335 = stablehlo.broadcast_in_dim %arg94, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %336 = stablehlo.multiply %334, %335 : tensor<1x512x7x7xf32>
    %337 = stablehlo.convert %arg95 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %338 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %339 = stablehlo.broadcast_in_dim %337, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %340 = stablehlo.multiply %338, %339 : tensor<1x512x7x7xf32>
    %341 = stablehlo.convert %arg96 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %342 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %343 = stablehlo.broadcast_in_dim %341, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %344 = stablehlo.add %342, %343 : tensor<1x512x7x7xf32>
    %345 = stablehlo.convert %344 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %346 = stablehlo.maximum %345, %cst_4 : tensor<1x512x7x7xbf16>
    %347 = stablehlo.convolution(%346, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %348 = stablehlo.convert %347 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %350 = stablehlo.broadcast_in_dim %arg97, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %351 = stablehlo.subtract %349, %350 : tensor<1x512x7x7xf32>
    %352 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %353 = stablehlo.broadcast_in_dim %arg98, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %354 = stablehlo.multiply %352, %353 : tensor<1x512x7x7xf32>
    %355 = stablehlo.convert %arg99 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %356 = stablehlo.broadcast_in_dim %354, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %357 = stablehlo.broadcast_in_dim %355, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %358 = stablehlo.multiply %356, %357 : tensor<1x512x7x7xf32>
    %359 = stablehlo.convert %arg100 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %360 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %361 = stablehlo.broadcast_in_dim %359, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %362 = stablehlo.add %360, %361 : tensor<1x512x7x7xf32>
    %363 = stablehlo.convert %362 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %364 = stablehlo.add %363, %328 : tensor<1x512x7x7xbf16>
    %365 = stablehlo.maximum %364, %cst_4 : tensor<1x512x7x7xbf16>
    %366 = stablehlo.reduce(%365 init: %cst_5) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x512x7x7xbf16>, tensor<bf16>) -> tensor<1x512xbf16>
    %367 = stablehlo.reshape %366 : (tensor<1x512xbf16>) -> tensor<1x512x1x1xbf16>
    %368 = stablehlo.convert %cst_6 : (tensor<1xi64>) -> tensor<1xbf16>
    %369 = stablehlo.reshape %368 : (tensor<1xbf16>) -> tensor<bf16>
    %370 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %371 = stablehlo.broadcast_in_dim %369, dims = [] : (tensor<bf16>) -> tensor<1x512x1x1xbf16>
    %372 = stablehlo.divide %370, %371 : tensor<1x512x1x1xbf16>
    %373 = stablehlo.reshape %372 : (tensor<1x512x1x1xbf16>) -> tensor<1x512xbf16>
    %374 = stablehlo.convert %373 : (tensor<1x512xbf16>) -> tensor<1x512xf32>
    %375 = stablehlo.dot_general %374, %arg101, contracting_dims = [1] x [0] : (tensor<1x512xf32>, tensor<512x1000xf32>) -> tensor<1x1000xf32>
    %376 = stablehlo.convert %cst_7 : (tensor<1xi64>) -> tensor<1xf32>
    %377 = stablehlo.reshape %376 : (tensor<1xf32>) -> tensor<f32>
    %378 = stablehlo.broadcast_in_dim %375, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %379 = stablehlo.broadcast_in_dim %377, dims = [] : (tensor<f32>) -> tensor<1x1000xf32>
    %380 = stablehlo.multiply %378, %379 : tensor<1x1000xf32>
    %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %382 = stablehlo.broadcast_in_dim %arg102, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %383 = stablehlo.add %381, %382 : tensor<1x1000xf32>
    %384 = stablehlo.convert %383 : (tensor<1x1000xf32>) -> tensor<1x1000xbf16>
    return %384 : tensor<1x1000xbf16>
  }
}
