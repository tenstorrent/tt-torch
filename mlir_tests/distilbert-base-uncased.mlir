module {
  func.func @main(%arg0: tensor<1x16xi64>, %arg1: tensor<1x16xi64>, %arg2: tensor<30522x768xbf16>, %arg3: tensor<768xbf16>, %arg4: tensor<768xbf16>, %arg5: tensor<768xbf16>, %arg6: tensor<768xbf16>, %arg7: tensor<768xbf16>, %arg8: tensor<768xbf16>, %arg9: tensor<768xbf16>, %arg10: tensor<768xbf16>, %arg11: tensor<768xbf16>, %arg12: tensor<768xbf16>, %arg13: tensor<768xbf16>, %arg14: tensor<768xbf16>, %arg15: tensor<768xbf16>, %arg16: tensor<768xbf16>, %arg17: tensor<768xbf16>, %arg18: tensor<768xbf16>, %arg19: tensor<768xbf16>, %arg20: tensor<768xbf16>, %arg21: tensor<768xbf16>, %arg22: tensor<768xbf16>, %arg23: tensor<768xbf16>, %arg24: tensor<768xbf16>, %arg25: tensor<768xbf16>, %arg26: tensor<768xbf16>, %arg27: tensor<768xbf16>, %arg28: tensor<768xbf16>, %arg29: tensor<1x16x768xbf16>, %arg30: tensor<bf16>, %arg31: tensor<768x768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768x768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768x768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x3072xf32>, %arg40: tensor<3072xf32>, %arg41: tensor<3072x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<768x768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768x768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768x3072xf32>, %arg52: tensor<3072xf32>, %arg53: tensor<3072x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768x3072xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<3072x768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768x768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x3072xf32>, %arg76: tensor<3072xf32>, %arg77: tensor<3072x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768x768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768x768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768x768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x3072xf32>, %arg88: tensor<3072xf32>, %arg89: tensor<3072x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768x768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<768x768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768x768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768x3072xf32>, %arg100: tensor<3072xf32>, %arg101: tensor<3072x768xf32>, %arg102: tensor<768xf32>) -> tensor<1x16x768xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<1x16x3072xbf16>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<1x16x3072xbf16>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<1x16x3072xbf16>
    %cst_5 = stablehlo.constant dense<-4.000000e+00> : tensor<1x16x3072xf32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<1x16x3072xf32>
    %cst_7 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x16x3072xf32>
    %cst_8 = stablehlo.constant dense<2.77068146E-8> : tensor<1x16x3072xf32>
    %cst_9 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x16x3072xf32>
    %cst_10 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x16x3072xf32>
    %cst_11 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x16x3072xf32>
    %cst_12 = stablehlo.constant dense<-2.954600e-03> : tensor<1x16x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.0160960332> : tensor<1x16x3072xf32>
    %cst_14 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x16x3072xf32>
    %cst_15 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x16x3072xf32>
    %cst_16 = stablehlo.constant dense<-0.00168282702> : tensor<1x16x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.00737332925> : tensor<1x16x3072xf32>
    %cst_18 = stablehlo.constant dense<-0.0142647391> : tensor<1x16x3072xf32>
    %cst_19 = stablehlo.constant dense<-1.000000e+00> : tensor<1x16x3072xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x16x3072xf32>
    %cst_21 = arith.constant dense<768> : tensor<1xi64>
    %cst_22 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
    %cst_23 = arith.constant dense<1.000000e+00> : tensor<1xf64>
    %cst_24 = arith.constant dense<1> : tensor<1xi64>
    %cst_25 = arith.constant dense<0.35355339059327379> : tensor<1xf64>
    %cst_26 = arith.constant dense<0xFFF0000000000000> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg2, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<30522x768xbf16>, tensor<1x16xi64>) -> tensor<1x16x768xbf16>
    %1 = stablehlo.convert %0 : tensor<1x16x768xbf16>
    %2 = stablehlo.add %1, %arg29 : tensor<1x16x768xbf16>
    %3 = stablehlo.convert %2 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %4 = stablehlo.convert %3 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %6 = stablehlo.reshape %5 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %7 = stablehlo.convert %cst_21 : (tensor<1xi64>) -> tensor<1xf64>
    %8 = stablehlo.reshape %7 : (tensor<1xf64>) -> tensor<f64>
    %9 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %10 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f64>) -> tensor<1x16x1xf64>
    %11 = stablehlo.divide %9, %10 : tensor<1x16x1xf64>
    %12 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %14 = stablehlo.subtract %12, %13 : tensor<1x16x768xf64>
    %15 = stablehlo.multiply %14, %14 : tensor<1x16x768xf64>
    %16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %17 = stablehlo.reshape %16 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %19 = stablehlo.divide %18, %10 : tensor<1x16x1xf64>
    %20 = stablehlo.convert %19 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %21 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %22 = stablehlo.reshape %21 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %23 = stablehlo.convert %cst_21 : (tensor<1xi64>) -> tensor<1xf32>
    %24 = stablehlo.reshape %23 : (tensor<1xf32>) -> tensor<f32>
    %25 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %26 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %27 = stablehlo.divide %25, %26 : tensor<1x16x1xf32>
    %28 = stablehlo.convert %cst_22 : (tensor<1xf64>) -> tensor<1xf32>
    %29 = stablehlo.reshape %28 : (tensor<1xf32>) -> tensor<f32>
    %30 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %31 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %32 = stablehlo.add %30, %31 : tensor<1x16x1xf32>
    %33 = stablehlo.rsqrt %32 : tensor<1x16x1xf32>
    %34 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %35 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %36 = stablehlo.subtract %34, %35 : tensor<1x16x768xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %38 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %39 = stablehlo.multiply %37, %38 : tensor<1x16x768xf32>
    %40 = stablehlo.convert %arg3 : (tensor<768xbf16>) -> tensor<768xf32>
    %41 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %42 = stablehlo.broadcast_in_dim %40, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x16x768xf32>
    %44 = stablehlo.convert %arg4 : (tensor<768xbf16>) -> tensor<768xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %47 = stablehlo.add %45, %46 : tensor<1x16x768xf32>
    %48 = stablehlo.convert %47 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %49 = stablehlo.reshape %arg1 : (tensor<1x16xi64>) -> tensor<1x1x16xi64>
    %50 = stablehlo.reshape %49 : (tensor<1x1x16xi64>) -> tensor<1x1x1x16xi64>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xi64>) -> tensor<1x1x16x16xi64>
    %52 = stablehlo.convert %51 : (tensor<1x1x16x16xi64>) -> tensor<1x1x16x16xbf16>
    %53 = stablehlo.convert %cst_23 : (tensor<1xf64>) -> tensor<1xbf16>
    %54 = stablehlo.reshape %53 : (tensor<1xbf16>) -> tensor<bf16>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<bf16>) -> tensor<1x1x16x16xbf16>
    %56 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xbf16>) -> tensor<1x1x16x16xbf16>
    %57 = stablehlo.subtract %55, %56 : tensor<1x1x16x16xbf16>
    %58 = stablehlo.convert %57 : (tensor<1x1x16x16xbf16>) -> tensor<1x1x16x16xi1>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xi1>) -> tensor<1x1x16x16xi1>
    %60 = stablehlo.broadcast_in_dim %arg30, dims = [] : (tensor<bf16>) -> tensor<1x1x16x16xbf16>
    %61 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xbf16>) -> tensor<1x1x16x16xbf16>
    %62 = stablehlo.select %59, %60, %61 : tensor<1x1x16x16xi1>, tensor<1x1x16x16xbf16>
    %63 = stablehlo.reshape %48 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %64 = stablehlo.convert %63 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %65 = stablehlo.dot_general %64, %arg31, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %66 = stablehlo.convert %cst_24 : (tensor<1xi64>) -> tensor<1xf32>
    %67 = stablehlo.reshape %66 : (tensor<1xf32>) -> tensor<f32>
    %68 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %69 = stablehlo.broadcast_in_dim %67, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %70 = stablehlo.multiply %68, %69 : tensor<16x768xf32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %72 = stablehlo.broadcast_in_dim %arg32, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x768xf32>
    %74 = stablehlo.convert %73 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %75 = stablehlo.reshape %74 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %76 = stablehlo.reshape %75 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %77 = stablehlo.transpose %76, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %78 = stablehlo.dot_general %64, %arg33, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %80 = stablehlo.multiply %79, %69 : tensor<16x768xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %82 = stablehlo.broadcast_in_dim %arg34, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %83 = stablehlo.add %81, %82 : tensor<16x768xf32>
    %84 = stablehlo.convert %83 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %85 = stablehlo.reshape %84 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %86 = stablehlo.reshape %85 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %87 = stablehlo.transpose %86, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %88 = stablehlo.dot_general %64, %arg35, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %90 = stablehlo.multiply %89, %69 : tensor<16x768xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %92 = stablehlo.broadcast_in_dim %arg36, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %93 = stablehlo.add %91, %92 : tensor<16x768xf32>
    %94 = stablehlo.convert %93 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %95 = stablehlo.reshape %94 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %96 = stablehlo.reshape %95 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %97 = stablehlo.transpose %96, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %98 = stablehlo.convert %77 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %99 = stablehlo.convert %87 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %100 = stablehlo.convert %97 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %101 = stablehlo.convert %cst_25 : (tensor<1xf64>) -> tensor<1xf32>
    %102 = stablehlo.reshape %101 : (tensor<1xf32>) -> tensor<f32>
    %103 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %104 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<f32>) -> tensor<1x12x16x64xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<1x12x16x64xf32>
    %106 = stablehlo.transpose %99, dims = [0, 1, 3, 2] : (tensor<1x12x16x64xf32>) -> tensor<1x12x64x16xf32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2, 3] : (tensor<1x12x64x16xf32>) -> tensor<1x12x64x16xf32>
    %108 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<f32>) -> tensor<1x12x64x16xf32>
    %109 = stablehlo.multiply %107, %108 : tensor<1x12x64x16xf32>
    %110 = stablehlo.reshape %105 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %111 = stablehlo.reshape %109 : (tensor<1x12x64x16xf32>) -> tensor<12x64x16xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2] : (tensor<12x64x16xf32>) -> tensor<12x64x16xf32>
    %113 = stablehlo.dot_general %110, %112, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x64xf32>, tensor<12x64x16xf32>) -> tensor<12x16x16xf32>
    %114 = stablehlo.reshape %113 : (tensor<12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %115 = stablehlo.convert %62 : (tensor<1x1x16x16xbf16>) -> tensor<1x1x16x16xf32>
    %116 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %117 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xf32>) -> tensor<1x12x16x16xf32>
    %118 = stablehlo.add %116, %117 : tensor<1x12x16x16xf32>
    %119 = stablehlo.reduce(%118 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %120 = stablehlo.reshape %119 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %121 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %122 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %123 = stablehlo.subtract %121, %122 : tensor<1x12x16x16xf32>
    %124 = stablehlo.exponential %123 : tensor<1x12x16x16xf32>
    %125 = stablehlo.reduce(%124 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %126 = stablehlo.reshape %125 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %127 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %128 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %129 = stablehlo.divide %127, %128 : tensor<1x12x16x16xf32>
    %130 = stablehlo.convert %cst_26 : tensor<1xf64>
    %131 = stablehlo.reshape %130 : (tensor<1xf64>) -> tensor<f64>
    %132 = stablehlo.convert %131 : (tensor<f64>) -> tensor<f32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [] : (tensor<f32>) -> tensor<1x12x16x16xf32>
    %134 = stablehlo.compare  EQ, %121, %133,  FLOAT : (tensor<1x12x16x16xf32>, tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xi1>
    %135 = stablehlo.reduce(%134 init: %c) applies stablehlo.and across dimensions = [3] : (tensor<1x12x16x16xi1>, tensor<i1>) -> tensor<1x12x16xi1>
    %136 = stablehlo.reshape %135 : (tensor<1x12x16xi1>) -> tensor<1x12x16x1xi1>
    %137 = stablehlo.convert %cst : (tensor<f64>) -> tensor<f32>
    %138 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xi1>) -> tensor<1x12x16x16xi1>
    %139 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<f32>) -> tensor<1x12x16x16xf32>
    %140 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %141 = stablehlo.select %138, %139, %140 : tensor<1x12x16x16xi1>, tensor<1x12x16x16xf32>
    %142 = stablehlo.reshape %141 : (tensor<1x12x16x16xf32>) -> tensor<12x16x16xf32>
    %143 = stablehlo.reshape %100 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %145 = stablehlo.dot_general %142, %144, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x16xf32>, tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %146 = stablehlo.reshape %145 : (tensor<12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %147 = stablehlo.convert %146 : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xbf16>
    %148 = stablehlo.transpose %147, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %149 = stablehlo.transpose %148, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %150 = stablehlo.transpose %149, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %151 = stablehlo.reshape %150 : (tensor<1x16x12x64xbf16>) -> tensor<1x16x768xbf16>
    %152 = stablehlo.reshape %151 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %153 = stablehlo.convert %152 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %154 = stablehlo.dot_general %153, %arg37, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %156 = stablehlo.multiply %155, %69 : tensor<16x768xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %158 = stablehlo.broadcast_in_dim %arg38, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %159 = stablehlo.add %157, %158 : tensor<16x768xf32>
    %160 = stablehlo.convert %159 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %161 = stablehlo.reshape %160 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %162 = stablehlo.add %161, %48 : tensor<1x16x768xbf16>
    %163 = stablehlo.convert %162 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %164 = stablehlo.convert %163 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %165 = stablehlo.reduce(%164 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %166 = stablehlo.reshape %165 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %168 = stablehlo.divide %167, %10 : tensor<1x16x1xf64>
    %169 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %170 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %171 = stablehlo.subtract %169, %170 : tensor<1x16x768xf64>
    %172 = stablehlo.multiply %171, %171 : tensor<1x16x768xf64>
    %173 = stablehlo.reduce(%172 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %176 = stablehlo.divide %175, %10 : tensor<1x16x1xf64>
    %177 = stablehlo.convert %176 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %178 = stablehlo.reduce(%163 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %179 = stablehlo.reshape %178 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %181 = stablehlo.divide %180, %26 : tensor<1x16x1xf32>
    %182 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %183 = stablehlo.add %182, %31 : tensor<1x16x1xf32>
    %184 = stablehlo.rsqrt %183 : tensor<1x16x1xf32>
    %185 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %186 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %187 = stablehlo.subtract %185, %186 : tensor<1x16x768xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %189 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %190 = stablehlo.multiply %188, %189 : tensor<1x16x768xf32>
    %191 = stablehlo.convert %arg5 : (tensor<768xbf16>) -> tensor<768xf32>
    %192 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %193 = stablehlo.broadcast_in_dim %191, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %194 = stablehlo.multiply %192, %193 : tensor<1x16x768xf32>
    %195 = stablehlo.convert %arg6 : (tensor<768xbf16>) -> tensor<768xf32>
    %196 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %197 = stablehlo.broadcast_in_dim %195, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %198 = stablehlo.add %196, %197 : tensor<1x16x768xf32>
    %199 = stablehlo.convert %198 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %200 = stablehlo.reshape %199 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %201 = stablehlo.convert %200 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %202 = stablehlo.dot_general %201, %arg39, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x3072xf32>) -> tensor<16x3072xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %204 = stablehlo.broadcast_in_dim %67, dims = [] : (tensor<f32>) -> tensor<16x3072xf32>
    %205 = stablehlo.multiply %203, %204 : tensor<16x3072xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %207 = stablehlo.broadcast_in_dim %arg40, dims = [1] : (tensor<3072xf32>) -> tensor<16x3072xf32>
    %208 = stablehlo.add %206, %207 : tensor<16x3072xf32>
    %209 = stablehlo.convert %208 : (tensor<16x3072xf32>) -> tensor<16x3072xbf16>
    %210 = stablehlo.reshape %209 : (tensor<16x3072xbf16>) -> tensor<1x16x3072xbf16>
    %211 = stablehlo.multiply %210, %cst_4 : tensor<1x16x3072xbf16>
    %212 = stablehlo.rsqrt %cst_3 : tensor<1x16x3072xbf16>
    %213 = stablehlo.multiply %210, %212 : tensor<1x16x3072xbf16>
    %214 = stablehlo.convert %213 : (tensor<1x16x3072xbf16>) -> tensor<1x16x3072xf32>
    %215 = stablehlo.clamp %cst_5, %214, %cst_6 : tensor<1x16x3072xf32>
    %216 = stablehlo.multiply %215, %215 : tensor<1x16x3072xf32>
    %217 = stablehlo.multiply %cst_7, %216 : tensor<1x16x3072xf32>
    %218 = stablehlo.add %217, %cst_8 : tensor<1x16x3072xf32>
    %219 = stablehlo.multiply %218, %216 : tensor<1x16x3072xf32>
    %220 = stablehlo.add %219, %cst_9 : tensor<1x16x3072xf32>
    %221 = stablehlo.multiply %220, %216 : tensor<1x16x3072xf32>
    %222 = stablehlo.add %221, %cst_10 : tensor<1x16x3072xf32>
    %223 = stablehlo.multiply %222, %216 : tensor<1x16x3072xf32>
    %224 = stablehlo.add %223, %cst_11 : tensor<1x16x3072xf32>
    %225 = stablehlo.multiply %224, %216 : tensor<1x16x3072xf32>
    %226 = stablehlo.add %225, %cst_12 : tensor<1x16x3072xf32>
    %227 = stablehlo.multiply %226, %216 : tensor<1x16x3072xf32>
    %228 = stablehlo.add %227, %cst_13 : tensor<1x16x3072xf32>
    %229 = stablehlo.multiply %cst_14, %216 : tensor<1x16x3072xf32>
    %230 = stablehlo.add %229, %cst_15 : tensor<1x16x3072xf32>
    %231 = stablehlo.multiply %230, %216 : tensor<1x16x3072xf32>
    %232 = stablehlo.add %231, %cst_16 : tensor<1x16x3072xf32>
    %233 = stablehlo.multiply %232, %216 : tensor<1x16x3072xf32>
    %234 = stablehlo.add %233, %cst_17 : tensor<1x16x3072xf32>
    %235 = stablehlo.multiply %234, %216 : tensor<1x16x3072xf32>
    %236 = stablehlo.add %235, %cst_18 : tensor<1x16x3072xf32>
    %237 = stablehlo.multiply %215, %228 : tensor<1x16x3072xf32>
    %238 = stablehlo.divide %237, %236 : tensor<1x16x3072xf32>
    %239 = stablehlo.clamp %cst_19, %238, %cst_20 : tensor<1x16x3072xf32>
    %240 = stablehlo.convert %239 : (tensor<1x16x3072xf32>) -> tensor<1x16x3072xbf16>
    %241 = stablehlo.add %240, %cst_2 : tensor<1x16x3072xbf16>
    %242 = stablehlo.multiply %241, %211 : tensor<1x16x3072xbf16>
    %243 = stablehlo.reshape %242 : (tensor<1x16x3072xbf16>) -> tensor<16x3072xbf16>
    %244 = stablehlo.convert %243 : (tensor<16x3072xbf16>) -> tensor<16x3072xf32>
    %245 = stablehlo.dot_general %244, %arg41, contracting_dims = [1] x [0] : (tensor<16x3072xf32>, tensor<3072x768xf32>) -> tensor<16x768xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %247 = stablehlo.multiply %246, %69 : tensor<16x768xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %249 = stablehlo.broadcast_in_dim %arg42, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %250 = stablehlo.add %248, %249 : tensor<16x768xf32>
    %251 = stablehlo.convert %250 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %252 = stablehlo.reshape %251 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %253 = stablehlo.add %252, %199 : tensor<1x16x768xbf16>
    %254 = stablehlo.convert %253 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %255 = stablehlo.convert %254 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %256 = stablehlo.reduce(%255 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %257 = stablehlo.reshape %256 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %258 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %259 = stablehlo.divide %258, %10 : tensor<1x16x1xf64>
    %260 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %261 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %262 = stablehlo.subtract %260, %261 : tensor<1x16x768xf64>
    %263 = stablehlo.multiply %262, %262 : tensor<1x16x768xf64>
    %264 = stablehlo.reduce(%263 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %265 = stablehlo.reshape %264 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %267 = stablehlo.divide %266, %10 : tensor<1x16x1xf64>
    %268 = stablehlo.convert %267 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %269 = stablehlo.reduce(%254 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %270 = stablehlo.reshape %269 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %271 = stablehlo.broadcast_in_dim %270, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %272 = stablehlo.divide %271, %26 : tensor<1x16x1xf32>
    %273 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %274 = stablehlo.add %273, %31 : tensor<1x16x1xf32>
    %275 = stablehlo.rsqrt %274 : tensor<1x16x1xf32>
    %276 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %277 = stablehlo.broadcast_in_dim %272, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %278 = stablehlo.subtract %276, %277 : tensor<1x16x768xf32>
    %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %280 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %281 = stablehlo.multiply %279, %280 : tensor<1x16x768xf32>
    %282 = stablehlo.convert %arg7 : (tensor<768xbf16>) -> tensor<768xf32>
    %283 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %284 = stablehlo.broadcast_in_dim %282, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %285 = stablehlo.multiply %283, %284 : tensor<1x16x768xf32>
    %286 = stablehlo.convert %arg8 : (tensor<768xbf16>) -> tensor<768xf32>
    %287 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %288 = stablehlo.broadcast_in_dim %286, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %289 = stablehlo.add %287, %288 : tensor<1x16x768xf32>
    %290 = stablehlo.convert %289 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %291 = stablehlo.reshape %290 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %292 = stablehlo.convert %291 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %293 = stablehlo.dot_general %292, %arg43, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %295 = stablehlo.multiply %294, %69 : tensor<16x768xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %297 = stablehlo.broadcast_in_dim %arg44, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %298 = stablehlo.add %296, %297 : tensor<16x768xf32>
    %299 = stablehlo.convert %298 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %300 = stablehlo.reshape %299 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %301 = stablehlo.reshape %300 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %303 = stablehlo.dot_general %292, %arg45, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %305 = stablehlo.multiply %304, %69 : tensor<16x768xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %307 = stablehlo.broadcast_in_dim %arg46, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %308 = stablehlo.add %306, %307 : tensor<16x768xf32>
    %309 = stablehlo.convert %308 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %310 = stablehlo.reshape %309 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %311 = stablehlo.reshape %310 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %313 = stablehlo.dot_general %292, %arg47, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %315 = stablehlo.multiply %314, %69 : tensor<16x768xf32>
    %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %317 = stablehlo.broadcast_in_dim %arg48, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %318 = stablehlo.add %316, %317 : tensor<16x768xf32>
    %319 = stablehlo.convert %318 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %320 = stablehlo.reshape %319 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %321 = stablehlo.reshape %320 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %322 = stablehlo.transpose %321, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %323 = stablehlo.convert %302 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %324 = stablehlo.convert %312 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %325 = stablehlo.convert %322 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %326 = stablehlo.broadcast_in_dim %323, dims = [0, 1, 2, 3] : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %327 = stablehlo.multiply %326, %104 : tensor<1x12x16x64xf32>
    %328 = stablehlo.transpose %324, dims = [0, 1, 3, 2] : (tensor<1x12x16x64xf32>) -> tensor<1x12x64x16xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2, 3] : (tensor<1x12x64x16xf32>) -> tensor<1x12x64x16xf32>
    %330 = stablehlo.multiply %329, %108 : tensor<1x12x64x16xf32>
    %331 = stablehlo.reshape %327 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %332 = stablehlo.reshape %330 : (tensor<1x12x64x16xf32>) -> tensor<12x64x16xf32>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2] : (tensor<12x64x16xf32>) -> tensor<12x64x16xf32>
    %334 = stablehlo.dot_general %331, %333, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x64xf32>, tensor<12x64x16xf32>) -> tensor<12x16x16xf32>
    %335 = stablehlo.reshape %334 : (tensor<12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %336 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %337 = stablehlo.add %336, %117 : tensor<1x12x16x16xf32>
    %338 = stablehlo.reduce(%337 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %339 = stablehlo.reshape %338 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %340 = stablehlo.broadcast_in_dim %337, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %341 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %342 = stablehlo.subtract %340, %341 : tensor<1x12x16x16xf32>
    %343 = stablehlo.exponential %342 : tensor<1x12x16x16xf32>
    %344 = stablehlo.reduce(%343 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %345 = stablehlo.reshape %344 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %346 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %347 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %348 = stablehlo.divide %346, %347 : tensor<1x12x16x16xf32>
    %349 = stablehlo.compare  EQ, %340, %133,  FLOAT : (tensor<1x12x16x16xf32>, tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xi1>
    %350 = stablehlo.reduce(%349 init: %c) applies stablehlo.and across dimensions = [3] : (tensor<1x12x16x16xi1>, tensor<i1>) -> tensor<1x12x16xi1>
    %351 = stablehlo.reshape %350 : (tensor<1x12x16xi1>) -> tensor<1x12x16x1xi1>
    %352 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xi1>) -> tensor<1x12x16x16xi1>
    %353 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %354 = stablehlo.select %352, %139, %353 : tensor<1x12x16x16xi1>, tensor<1x12x16x16xf32>
    %355 = stablehlo.reshape %354 : (tensor<1x12x16x16xf32>) -> tensor<12x16x16xf32>
    %356 = stablehlo.reshape %325 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2] : (tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %358 = stablehlo.dot_general %355, %357, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x16xf32>, tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %359 = stablehlo.reshape %358 : (tensor<12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %360 = stablehlo.convert %359 : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xbf16>
    %361 = stablehlo.transpose %360, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %362 = stablehlo.transpose %361, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %363 = stablehlo.transpose %362, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %364 = stablehlo.reshape %363 : (tensor<1x16x12x64xbf16>) -> tensor<1x16x768xbf16>
    %365 = stablehlo.reshape %364 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %366 = stablehlo.convert %365 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %367 = stablehlo.dot_general %366, %arg49, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %369 = stablehlo.multiply %368, %69 : tensor<16x768xf32>
    %370 = stablehlo.broadcast_in_dim %369, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %371 = stablehlo.broadcast_in_dim %arg50, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %372 = stablehlo.add %370, %371 : tensor<16x768xf32>
    %373 = stablehlo.convert %372 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %374 = stablehlo.reshape %373 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %375 = stablehlo.add %374, %290 : tensor<1x16x768xbf16>
    %376 = stablehlo.convert %375 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %377 = stablehlo.convert %376 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %379 = stablehlo.reshape %378 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %381 = stablehlo.divide %380, %10 : tensor<1x16x1xf64>
    %382 = stablehlo.broadcast_in_dim %377, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %383 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %384 = stablehlo.subtract %382, %383 : tensor<1x16x768xf64>
    %385 = stablehlo.multiply %384, %384 : tensor<1x16x768xf64>
    %386 = stablehlo.reduce(%385 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %387 = stablehlo.reshape %386 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %389 = stablehlo.divide %388, %10 : tensor<1x16x1xf64>
    %390 = stablehlo.convert %389 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %391 = stablehlo.reduce(%376 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %392 = stablehlo.reshape %391 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %394 = stablehlo.divide %393, %26 : tensor<1x16x1xf32>
    %395 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %396 = stablehlo.add %395, %31 : tensor<1x16x1xf32>
    %397 = stablehlo.rsqrt %396 : tensor<1x16x1xf32>
    %398 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %399 = stablehlo.broadcast_in_dim %394, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %400 = stablehlo.subtract %398, %399 : tensor<1x16x768xf32>
    %401 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %402 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %403 = stablehlo.multiply %401, %402 : tensor<1x16x768xf32>
    %404 = stablehlo.convert %arg9 : (tensor<768xbf16>) -> tensor<768xf32>
    %405 = stablehlo.broadcast_in_dim %403, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %406 = stablehlo.broadcast_in_dim %404, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %407 = stablehlo.multiply %405, %406 : tensor<1x16x768xf32>
    %408 = stablehlo.convert %arg10 : (tensor<768xbf16>) -> tensor<768xf32>
    %409 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %410 = stablehlo.broadcast_in_dim %408, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %411 = stablehlo.add %409, %410 : tensor<1x16x768xf32>
    %412 = stablehlo.convert %411 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %413 = stablehlo.reshape %412 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %414 = stablehlo.convert %413 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %415 = stablehlo.dot_general %414, %arg51, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x3072xf32>) -> tensor<16x3072xf32>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %417 = stablehlo.multiply %416, %204 : tensor<16x3072xf32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %419 = stablehlo.broadcast_in_dim %arg52, dims = [1] : (tensor<3072xf32>) -> tensor<16x3072xf32>
    %420 = stablehlo.add %418, %419 : tensor<16x3072xf32>
    %421 = stablehlo.convert %420 : (tensor<16x3072xf32>) -> tensor<16x3072xbf16>
    %422 = stablehlo.reshape %421 : (tensor<16x3072xbf16>) -> tensor<1x16x3072xbf16>
    %423 = stablehlo.multiply %422, %cst_4 : tensor<1x16x3072xbf16>
    %424 = stablehlo.multiply %422, %212 : tensor<1x16x3072xbf16>
    %425 = stablehlo.convert %424 : (tensor<1x16x3072xbf16>) -> tensor<1x16x3072xf32>
    %426 = stablehlo.clamp %cst_5, %425, %cst_6 : tensor<1x16x3072xf32>
    %427 = stablehlo.multiply %426, %426 : tensor<1x16x3072xf32>
    %428 = stablehlo.multiply %cst_7, %427 : tensor<1x16x3072xf32>
    %429 = stablehlo.add %428, %cst_8 : tensor<1x16x3072xf32>
    %430 = stablehlo.multiply %429, %427 : tensor<1x16x3072xf32>
    %431 = stablehlo.add %430, %cst_9 : tensor<1x16x3072xf32>
    %432 = stablehlo.multiply %431, %427 : tensor<1x16x3072xf32>
    %433 = stablehlo.add %432, %cst_10 : tensor<1x16x3072xf32>
    %434 = stablehlo.multiply %433, %427 : tensor<1x16x3072xf32>
    %435 = stablehlo.add %434, %cst_11 : tensor<1x16x3072xf32>
    %436 = stablehlo.multiply %435, %427 : tensor<1x16x3072xf32>
    %437 = stablehlo.add %436, %cst_12 : tensor<1x16x3072xf32>
    %438 = stablehlo.multiply %437, %427 : tensor<1x16x3072xf32>
    %439 = stablehlo.add %438, %cst_13 : tensor<1x16x3072xf32>
    %440 = stablehlo.multiply %cst_14, %427 : tensor<1x16x3072xf32>
    %441 = stablehlo.add %440, %cst_15 : tensor<1x16x3072xf32>
    %442 = stablehlo.multiply %441, %427 : tensor<1x16x3072xf32>
    %443 = stablehlo.add %442, %cst_16 : tensor<1x16x3072xf32>
    %444 = stablehlo.multiply %443, %427 : tensor<1x16x3072xf32>
    %445 = stablehlo.add %444, %cst_17 : tensor<1x16x3072xf32>
    %446 = stablehlo.multiply %445, %427 : tensor<1x16x3072xf32>
    %447 = stablehlo.add %446, %cst_18 : tensor<1x16x3072xf32>
    %448 = stablehlo.multiply %426, %439 : tensor<1x16x3072xf32>
    %449 = stablehlo.divide %448, %447 : tensor<1x16x3072xf32>
    %450 = stablehlo.clamp %cst_19, %449, %cst_20 : tensor<1x16x3072xf32>
    %451 = stablehlo.convert %450 : (tensor<1x16x3072xf32>) -> tensor<1x16x3072xbf16>
    %452 = stablehlo.add %451, %cst_2 : tensor<1x16x3072xbf16>
    %453 = stablehlo.multiply %452, %423 : tensor<1x16x3072xbf16>
    %454 = stablehlo.reshape %453 : (tensor<1x16x3072xbf16>) -> tensor<16x3072xbf16>
    %455 = stablehlo.convert %454 : (tensor<16x3072xbf16>) -> tensor<16x3072xf32>
    %456 = stablehlo.dot_general %455, %arg53, contracting_dims = [1] x [0] : (tensor<16x3072xf32>, tensor<3072x768xf32>) -> tensor<16x768xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %458 = stablehlo.multiply %457, %69 : tensor<16x768xf32>
    %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %460 = stablehlo.broadcast_in_dim %arg54, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %461 = stablehlo.add %459, %460 : tensor<16x768xf32>
    %462 = stablehlo.convert %461 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %463 = stablehlo.reshape %462 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %464 = stablehlo.add %463, %412 : tensor<1x16x768xbf16>
    %465 = stablehlo.convert %464 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %466 = stablehlo.convert %465 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %467 = stablehlo.reduce(%466 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %468 = stablehlo.reshape %467 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %469 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %470 = stablehlo.divide %469, %10 : tensor<1x16x1xf64>
    %471 = stablehlo.broadcast_in_dim %466, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %472 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %473 = stablehlo.subtract %471, %472 : tensor<1x16x768xf64>
    %474 = stablehlo.multiply %473, %473 : tensor<1x16x768xf64>
    %475 = stablehlo.reduce(%474 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %476 = stablehlo.reshape %475 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %478 = stablehlo.divide %477, %10 : tensor<1x16x1xf64>
    %479 = stablehlo.convert %478 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %480 = stablehlo.reduce(%465 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %481 = stablehlo.reshape %480 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %483 = stablehlo.divide %482, %26 : tensor<1x16x1xf32>
    %484 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %485 = stablehlo.add %484, %31 : tensor<1x16x1xf32>
    %486 = stablehlo.rsqrt %485 : tensor<1x16x1xf32>
    %487 = stablehlo.broadcast_in_dim %465, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %488 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %489 = stablehlo.subtract %487, %488 : tensor<1x16x768xf32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %491 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %492 = stablehlo.multiply %490, %491 : tensor<1x16x768xf32>
    %493 = stablehlo.convert %arg11 : (tensor<768xbf16>) -> tensor<768xf32>
    %494 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %495 = stablehlo.broadcast_in_dim %493, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %496 = stablehlo.multiply %494, %495 : tensor<1x16x768xf32>
    %497 = stablehlo.convert %arg12 : (tensor<768xbf16>) -> tensor<768xf32>
    %498 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %499 = stablehlo.broadcast_in_dim %497, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %500 = stablehlo.add %498, %499 : tensor<1x16x768xf32>
    %501 = stablehlo.convert %500 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %502 = stablehlo.reshape %501 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %503 = stablehlo.convert %502 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %504 = stablehlo.dot_general %503, %arg55, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %505 = stablehlo.broadcast_in_dim %504, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %506 = stablehlo.multiply %505, %69 : tensor<16x768xf32>
    %507 = stablehlo.broadcast_in_dim %506, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %508 = stablehlo.broadcast_in_dim %arg56, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %509 = stablehlo.add %507, %508 : tensor<16x768xf32>
    %510 = stablehlo.convert %509 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %511 = stablehlo.reshape %510 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %512 = stablehlo.reshape %511 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %513 = stablehlo.transpose %512, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %514 = stablehlo.dot_general %503, %arg57, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %516 = stablehlo.multiply %515, %69 : tensor<16x768xf32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %518 = stablehlo.broadcast_in_dim %arg58, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %519 = stablehlo.add %517, %518 : tensor<16x768xf32>
    %520 = stablehlo.convert %519 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %521 = stablehlo.reshape %520 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %522 = stablehlo.reshape %521 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %523 = stablehlo.transpose %522, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %524 = stablehlo.dot_general %503, %arg59, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %525 = stablehlo.broadcast_in_dim %524, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %526 = stablehlo.multiply %525, %69 : tensor<16x768xf32>
    %527 = stablehlo.broadcast_in_dim %526, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %528 = stablehlo.broadcast_in_dim %arg60, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %529 = stablehlo.add %527, %528 : tensor<16x768xf32>
    %530 = stablehlo.convert %529 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %531 = stablehlo.reshape %530 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %532 = stablehlo.reshape %531 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %533 = stablehlo.transpose %532, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %534 = stablehlo.convert %513 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %535 = stablehlo.convert %523 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %536 = stablehlo.convert %533 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %537 = stablehlo.broadcast_in_dim %534, dims = [0, 1, 2, 3] : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %538 = stablehlo.multiply %537, %104 : tensor<1x12x16x64xf32>
    %539 = stablehlo.transpose %535, dims = [0, 1, 3, 2] : (tensor<1x12x16x64xf32>) -> tensor<1x12x64x16xf32>
    %540 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2, 3] : (tensor<1x12x64x16xf32>) -> tensor<1x12x64x16xf32>
    %541 = stablehlo.multiply %540, %108 : tensor<1x12x64x16xf32>
    %542 = stablehlo.reshape %538 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %543 = stablehlo.reshape %541 : (tensor<1x12x64x16xf32>) -> tensor<12x64x16xf32>
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2] : (tensor<12x64x16xf32>) -> tensor<12x64x16xf32>
    %545 = stablehlo.dot_general %542, %544, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x64xf32>, tensor<12x64x16xf32>) -> tensor<12x16x16xf32>
    %546 = stablehlo.reshape %545 : (tensor<12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %547 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %548 = stablehlo.add %547, %117 : tensor<1x12x16x16xf32>
    %549 = stablehlo.reduce(%548 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %550 = stablehlo.reshape %549 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %551 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %552 = stablehlo.broadcast_in_dim %550, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %553 = stablehlo.subtract %551, %552 : tensor<1x12x16x16xf32>
    %554 = stablehlo.exponential %553 : tensor<1x12x16x16xf32>
    %555 = stablehlo.reduce(%554 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %556 = stablehlo.reshape %555 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %557 = stablehlo.broadcast_in_dim %554, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %558 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %559 = stablehlo.divide %557, %558 : tensor<1x12x16x16xf32>
    %560 = stablehlo.compare  EQ, %551, %133,  FLOAT : (tensor<1x12x16x16xf32>, tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xi1>
    %561 = stablehlo.reduce(%560 init: %c) applies stablehlo.and across dimensions = [3] : (tensor<1x12x16x16xi1>, tensor<i1>) -> tensor<1x12x16xi1>
    %562 = stablehlo.reshape %561 : (tensor<1x12x16xi1>) -> tensor<1x12x16x1xi1>
    %563 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xi1>) -> tensor<1x12x16x16xi1>
    %564 = stablehlo.broadcast_in_dim %559, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %565 = stablehlo.select %563, %139, %564 : tensor<1x12x16x16xi1>, tensor<1x12x16x16xf32>
    %566 = stablehlo.reshape %565 : (tensor<1x12x16x16xf32>) -> tensor<12x16x16xf32>
    %567 = stablehlo.reshape %536 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %568 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2] : (tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %569 = stablehlo.dot_general %566, %568, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x16xf32>, tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %570 = stablehlo.reshape %569 : (tensor<12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %571 = stablehlo.convert %570 : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xbf16>
    %572 = stablehlo.transpose %571, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %573 = stablehlo.transpose %572, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %574 = stablehlo.transpose %573, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %575 = stablehlo.reshape %574 : (tensor<1x16x12x64xbf16>) -> tensor<1x16x768xbf16>
    %576 = stablehlo.reshape %575 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %577 = stablehlo.convert %576 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %578 = stablehlo.dot_general %577, %arg61, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %579 = stablehlo.broadcast_in_dim %578, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %580 = stablehlo.multiply %579, %69 : tensor<16x768xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %582 = stablehlo.broadcast_in_dim %arg62, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %583 = stablehlo.add %581, %582 : tensor<16x768xf32>
    %584 = stablehlo.convert %583 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %585 = stablehlo.reshape %584 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %586 = stablehlo.add %585, %501 : tensor<1x16x768xbf16>
    %587 = stablehlo.convert %586 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %588 = stablehlo.convert %587 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %589 = stablehlo.reduce(%588 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %590 = stablehlo.reshape %589 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %591 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %592 = stablehlo.divide %591, %10 : tensor<1x16x1xf64>
    %593 = stablehlo.broadcast_in_dim %588, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %594 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %595 = stablehlo.subtract %593, %594 : tensor<1x16x768xf64>
    %596 = stablehlo.multiply %595, %595 : tensor<1x16x768xf64>
    %597 = stablehlo.reduce(%596 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %598 = stablehlo.reshape %597 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %599 = stablehlo.broadcast_in_dim %598, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %600 = stablehlo.divide %599, %10 : tensor<1x16x1xf64>
    %601 = stablehlo.convert %600 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %602 = stablehlo.reduce(%587 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %603 = stablehlo.reshape %602 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %604 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %605 = stablehlo.divide %604, %26 : tensor<1x16x1xf32>
    %606 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %607 = stablehlo.add %606, %31 : tensor<1x16x1xf32>
    %608 = stablehlo.rsqrt %607 : tensor<1x16x1xf32>
    %609 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %610 = stablehlo.broadcast_in_dim %605, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %611 = stablehlo.subtract %609, %610 : tensor<1x16x768xf32>
    %612 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %613 = stablehlo.broadcast_in_dim %608, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %614 = stablehlo.multiply %612, %613 : tensor<1x16x768xf32>
    %615 = stablehlo.convert %arg13 : (tensor<768xbf16>) -> tensor<768xf32>
    %616 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %617 = stablehlo.broadcast_in_dim %615, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %618 = stablehlo.multiply %616, %617 : tensor<1x16x768xf32>
    %619 = stablehlo.convert %arg14 : (tensor<768xbf16>) -> tensor<768xf32>
    %620 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %621 = stablehlo.broadcast_in_dim %619, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %622 = stablehlo.add %620, %621 : tensor<1x16x768xf32>
    %623 = stablehlo.convert %622 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %624 = stablehlo.reshape %623 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %625 = stablehlo.convert %624 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %626 = stablehlo.dot_general %625, %arg63, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x3072xf32>) -> tensor<16x3072xf32>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %628 = stablehlo.multiply %627, %204 : tensor<16x3072xf32>
    %629 = stablehlo.broadcast_in_dim %628, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %630 = stablehlo.broadcast_in_dim %arg64, dims = [1] : (tensor<3072xf32>) -> tensor<16x3072xf32>
    %631 = stablehlo.add %629, %630 : tensor<16x3072xf32>
    %632 = stablehlo.convert %631 : (tensor<16x3072xf32>) -> tensor<16x3072xbf16>
    %633 = stablehlo.reshape %632 : (tensor<16x3072xbf16>) -> tensor<1x16x3072xbf16>
    %634 = stablehlo.multiply %633, %cst_4 : tensor<1x16x3072xbf16>
    %635 = stablehlo.multiply %633, %212 : tensor<1x16x3072xbf16>
    %636 = stablehlo.convert %635 : (tensor<1x16x3072xbf16>) -> tensor<1x16x3072xf32>
    %637 = stablehlo.clamp %cst_5, %636, %cst_6 : tensor<1x16x3072xf32>
    %638 = stablehlo.multiply %637, %637 : tensor<1x16x3072xf32>
    %639 = stablehlo.multiply %cst_7, %638 : tensor<1x16x3072xf32>
    %640 = stablehlo.add %639, %cst_8 : tensor<1x16x3072xf32>
    %641 = stablehlo.multiply %640, %638 : tensor<1x16x3072xf32>
    %642 = stablehlo.add %641, %cst_9 : tensor<1x16x3072xf32>
    %643 = stablehlo.multiply %642, %638 : tensor<1x16x3072xf32>
    %644 = stablehlo.add %643, %cst_10 : tensor<1x16x3072xf32>
    %645 = stablehlo.multiply %644, %638 : tensor<1x16x3072xf32>
    %646 = stablehlo.add %645, %cst_11 : tensor<1x16x3072xf32>
    %647 = stablehlo.multiply %646, %638 : tensor<1x16x3072xf32>
    %648 = stablehlo.add %647, %cst_12 : tensor<1x16x3072xf32>
    %649 = stablehlo.multiply %648, %638 : tensor<1x16x3072xf32>
    %650 = stablehlo.add %649, %cst_13 : tensor<1x16x3072xf32>
    %651 = stablehlo.multiply %cst_14, %638 : tensor<1x16x3072xf32>
    %652 = stablehlo.add %651, %cst_15 : tensor<1x16x3072xf32>
    %653 = stablehlo.multiply %652, %638 : tensor<1x16x3072xf32>
    %654 = stablehlo.add %653, %cst_16 : tensor<1x16x3072xf32>
    %655 = stablehlo.multiply %654, %638 : tensor<1x16x3072xf32>
    %656 = stablehlo.add %655, %cst_17 : tensor<1x16x3072xf32>
    %657 = stablehlo.multiply %656, %638 : tensor<1x16x3072xf32>
    %658 = stablehlo.add %657, %cst_18 : tensor<1x16x3072xf32>
    %659 = stablehlo.multiply %637, %650 : tensor<1x16x3072xf32>
    %660 = stablehlo.divide %659, %658 : tensor<1x16x3072xf32>
    %661 = stablehlo.clamp %cst_19, %660, %cst_20 : tensor<1x16x3072xf32>
    %662 = stablehlo.convert %661 : (tensor<1x16x3072xf32>) -> tensor<1x16x3072xbf16>
    %663 = stablehlo.add %662, %cst_2 : tensor<1x16x3072xbf16>
    %664 = stablehlo.multiply %663, %634 : tensor<1x16x3072xbf16>
    %665 = stablehlo.reshape %664 : (tensor<1x16x3072xbf16>) -> tensor<16x3072xbf16>
    %666 = stablehlo.convert %665 : (tensor<16x3072xbf16>) -> tensor<16x3072xf32>
    %667 = stablehlo.dot_general %666, %arg65, contracting_dims = [1] x [0] : (tensor<16x3072xf32>, tensor<3072x768xf32>) -> tensor<16x768xf32>
    %668 = stablehlo.broadcast_in_dim %667, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %669 = stablehlo.multiply %668, %69 : tensor<16x768xf32>
    %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %671 = stablehlo.broadcast_in_dim %arg66, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %672 = stablehlo.add %670, %671 : tensor<16x768xf32>
    %673 = stablehlo.convert %672 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %674 = stablehlo.reshape %673 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %675 = stablehlo.add %674, %623 : tensor<1x16x768xbf16>
    %676 = stablehlo.convert %675 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %677 = stablehlo.convert %676 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %678 = stablehlo.reduce(%677 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %679 = stablehlo.reshape %678 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %681 = stablehlo.divide %680, %10 : tensor<1x16x1xf64>
    %682 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %683 = stablehlo.broadcast_in_dim %681, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %684 = stablehlo.subtract %682, %683 : tensor<1x16x768xf64>
    %685 = stablehlo.multiply %684, %684 : tensor<1x16x768xf64>
    %686 = stablehlo.reduce(%685 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %687 = stablehlo.reshape %686 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %689 = stablehlo.divide %688, %10 : tensor<1x16x1xf64>
    %690 = stablehlo.convert %689 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %691 = stablehlo.reduce(%676 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %692 = stablehlo.reshape %691 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %694 = stablehlo.divide %693, %26 : tensor<1x16x1xf32>
    %695 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %696 = stablehlo.add %695, %31 : tensor<1x16x1xf32>
    %697 = stablehlo.rsqrt %696 : tensor<1x16x1xf32>
    %698 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %699 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %700 = stablehlo.subtract %698, %699 : tensor<1x16x768xf32>
    %701 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %702 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %703 = stablehlo.multiply %701, %702 : tensor<1x16x768xf32>
    %704 = stablehlo.convert %arg15 : (tensor<768xbf16>) -> tensor<768xf32>
    %705 = stablehlo.broadcast_in_dim %703, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %706 = stablehlo.broadcast_in_dim %704, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %707 = stablehlo.multiply %705, %706 : tensor<1x16x768xf32>
    %708 = stablehlo.convert %arg16 : (tensor<768xbf16>) -> tensor<768xf32>
    %709 = stablehlo.broadcast_in_dim %707, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %710 = stablehlo.broadcast_in_dim %708, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %711 = stablehlo.add %709, %710 : tensor<1x16x768xf32>
    %712 = stablehlo.convert %711 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %713 = stablehlo.reshape %712 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %714 = stablehlo.convert %713 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %715 = stablehlo.dot_general %714, %arg67, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %717 = stablehlo.multiply %716, %69 : tensor<16x768xf32>
    %718 = stablehlo.broadcast_in_dim %717, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %719 = stablehlo.broadcast_in_dim %arg68, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %720 = stablehlo.add %718, %719 : tensor<16x768xf32>
    %721 = stablehlo.convert %720 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %722 = stablehlo.reshape %721 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %723 = stablehlo.reshape %722 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %724 = stablehlo.transpose %723, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %725 = stablehlo.dot_general %714, %arg69, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %726 = stablehlo.broadcast_in_dim %725, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %727 = stablehlo.multiply %726, %69 : tensor<16x768xf32>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %729 = stablehlo.broadcast_in_dim %arg70, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %730 = stablehlo.add %728, %729 : tensor<16x768xf32>
    %731 = stablehlo.convert %730 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %732 = stablehlo.reshape %731 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %733 = stablehlo.reshape %732 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %734 = stablehlo.transpose %733, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %735 = stablehlo.dot_general %714, %arg71, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %736 = stablehlo.broadcast_in_dim %735, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %737 = stablehlo.multiply %736, %69 : tensor<16x768xf32>
    %738 = stablehlo.broadcast_in_dim %737, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %739 = stablehlo.broadcast_in_dim %arg72, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %740 = stablehlo.add %738, %739 : tensor<16x768xf32>
    %741 = stablehlo.convert %740 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %742 = stablehlo.reshape %741 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %743 = stablehlo.reshape %742 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %744 = stablehlo.transpose %743, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %745 = stablehlo.convert %724 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %746 = stablehlo.convert %734 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %747 = stablehlo.convert %744 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %748 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2, 3] : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %749 = stablehlo.multiply %748, %104 : tensor<1x12x16x64xf32>
    %750 = stablehlo.transpose %746, dims = [0, 1, 3, 2] : (tensor<1x12x16x64xf32>) -> tensor<1x12x64x16xf32>
    %751 = stablehlo.broadcast_in_dim %750, dims = [0, 1, 2, 3] : (tensor<1x12x64x16xf32>) -> tensor<1x12x64x16xf32>
    %752 = stablehlo.multiply %751, %108 : tensor<1x12x64x16xf32>
    %753 = stablehlo.reshape %749 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %754 = stablehlo.reshape %752 : (tensor<1x12x64x16xf32>) -> tensor<12x64x16xf32>
    %755 = stablehlo.broadcast_in_dim %754, dims = [0, 1, 2] : (tensor<12x64x16xf32>) -> tensor<12x64x16xf32>
    %756 = stablehlo.dot_general %753, %755, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x64xf32>, tensor<12x64x16xf32>) -> tensor<12x16x16xf32>
    %757 = stablehlo.reshape %756 : (tensor<12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %758 = stablehlo.broadcast_in_dim %757, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %759 = stablehlo.add %758, %117 : tensor<1x12x16x16xf32>
    %760 = stablehlo.reduce(%759 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %761 = stablehlo.reshape %760 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %762 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %763 = stablehlo.broadcast_in_dim %761, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %764 = stablehlo.subtract %762, %763 : tensor<1x12x16x16xf32>
    %765 = stablehlo.exponential %764 : tensor<1x12x16x16xf32>
    %766 = stablehlo.reduce(%765 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %767 = stablehlo.reshape %766 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %768 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %769 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %770 = stablehlo.divide %768, %769 : tensor<1x12x16x16xf32>
    %771 = stablehlo.compare  EQ, %762, %133,  FLOAT : (tensor<1x12x16x16xf32>, tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xi1>
    %772 = stablehlo.reduce(%771 init: %c) applies stablehlo.and across dimensions = [3] : (tensor<1x12x16x16xi1>, tensor<i1>) -> tensor<1x12x16xi1>
    %773 = stablehlo.reshape %772 : (tensor<1x12x16xi1>) -> tensor<1x12x16x1xi1>
    %774 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xi1>) -> tensor<1x12x16x16xi1>
    %775 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %776 = stablehlo.select %774, %139, %775 : tensor<1x12x16x16xi1>, tensor<1x12x16x16xf32>
    %777 = stablehlo.reshape %776 : (tensor<1x12x16x16xf32>) -> tensor<12x16x16xf32>
    %778 = stablehlo.reshape %747 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %779 = stablehlo.broadcast_in_dim %778, dims = [0, 1, 2] : (tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %780 = stablehlo.dot_general %777, %779, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x16xf32>, tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %781 = stablehlo.reshape %780 : (tensor<12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %782 = stablehlo.convert %781 : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xbf16>
    %783 = stablehlo.transpose %782, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %784 = stablehlo.transpose %783, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %785 = stablehlo.transpose %784, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %786 = stablehlo.reshape %785 : (tensor<1x16x12x64xbf16>) -> tensor<1x16x768xbf16>
    %787 = stablehlo.reshape %786 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %788 = stablehlo.convert %787 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %789 = stablehlo.dot_general %788, %arg73, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %791 = stablehlo.multiply %790, %69 : tensor<16x768xf32>
    %792 = stablehlo.broadcast_in_dim %791, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %793 = stablehlo.broadcast_in_dim %arg74, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %794 = stablehlo.add %792, %793 : tensor<16x768xf32>
    %795 = stablehlo.convert %794 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %796 = stablehlo.reshape %795 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %797 = stablehlo.add %796, %712 : tensor<1x16x768xbf16>
    %798 = stablehlo.convert %797 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %799 = stablehlo.convert %798 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %800 = stablehlo.reduce(%799 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %801 = stablehlo.reshape %800 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %802 = stablehlo.broadcast_in_dim %801, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %803 = stablehlo.divide %802, %10 : tensor<1x16x1xf64>
    %804 = stablehlo.broadcast_in_dim %799, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %805 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %806 = stablehlo.subtract %804, %805 : tensor<1x16x768xf64>
    %807 = stablehlo.multiply %806, %806 : tensor<1x16x768xf64>
    %808 = stablehlo.reduce(%807 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %809 = stablehlo.reshape %808 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %810 = stablehlo.broadcast_in_dim %809, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %811 = stablehlo.divide %810, %10 : tensor<1x16x1xf64>
    %812 = stablehlo.convert %811 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %813 = stablehlo.reduce(%798 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %814 = stablehlo.reshape %813 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %815 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %816 = stablehlo.divide %815, %26 : tensor<1x16x1xf32>
    %817 = stablehlo.broadcast_in_dim %812, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %818 = stablehlo.add %817, %31 : tensor<1x16x1xf32>
    %819 = stablehlo.rsqrt %818 : tensor<1x16x1xf32>
    %820 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %821 = stablehlo.broadcast_in_dim %816, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %822 = stablehlo.subtract %820, %821 : tensor<1x16x768xf32>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %824 = stablehlo.broadcast_in_dim %819, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %825 = stablehlo.multiply %823, %824 : tensor<1x16x768xf32>
    %826 = stablehlo.convert %arg17 : (tensor<768xbf16>) -> tensor<768xf32>
    %827 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %828 = stablehlo.broadcast_in_dim %826, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %829 = stablehlo.multiply %827, %828 : tensor<1x16x768xf32>
    %830 = stablehlo.convert %arg18 : (tensor<768xbf16>) -> tensor<768xf32>
    %831 = stablehlo.broadcast_in_dim %829, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %832 = stablehlo.broadcast_in_dim %830, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %833 = stablehlo.add %831, %832 : tensor<1x16x768xf32>
    %834 = stablehlo.convert %833 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %835 = stablehlo.reshape %834 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %836 = stablehlo.convert %835 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %837 = stablehlo.dot_general %836, %arg75, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x3072xf32>) -> tensor<16x3072xf32>
    %838 = stablehlo.broadcast_in_dim %837, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %839 = stablehlo.multiply %838, %204 : tensor<16x3072xf32>
    %840 = stablehlo.broadcast_in_dim %839, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %841 = stablehlo.broadcast_in_dim %arg76, dims = [1] : (tensor<3072xf32>) -> tensor<16x3072xf32>
    %842 = stablehlo.add %840, %841 : tensor<16x3072xf32>
    %843 = stablehlo.convert %842 : (tensor<16x3072xf32>) -> tensor<16x3072xbf16>
    %844 = stablehlo.reshape %843 : (tensor<16x3072xbf16>) -> tensor<1x16x3072xbf16>
    %845 = stablehlo.multiply %844, %cst_4 : tensor<1x16x3072xbf16>
    %846 = stablehlo.multiply %844, %212 : tensor<1x16x3072xbf16>
    %847 = stablehlo.convert %846 : (tensor<1x16x3072xbf16>) -> tensor<1x16x3072xf32>
    %848 = stablehlo.clamp %cst_5, %847, %cst_6 : tensor<1x16x3072xf32>
    %849 = stablehlo.multiply %848, %848 : tensor<1x16x3072xf32>
    %850 = stablehlo.multiply %cst_7, %849 : tensor<1x16x3072xf32>
    %851 = stablehlo.add %850, %cst_8 : tensor<1x16x3072xf32>
    %852 = stablehlo.multiply %851, %849 : tensor<1x16x3072xf32>
    %853 = stablehlo.add %852, %cst_9 : tensor<1x16x3072xf32>
    %854 = stablehlo.multiply %853, %849 : tensor<1x16x3072xf32>
    %855 = stablehlo.add %854, %cst_10 : tensor<1x16x3072xf32>
    %856 = stablehlo.multiply %855, %849 : tensor<1x16x3072xf32>
    %857 = stablehlo.add %856, %cst_11 : tensor<1x16x3072xf32>
    %858 = stablehlo.multiply %857, %849 : tensor<1x16x3072xf32>
    %859 = stablehlo.add %858, %cst_12 : tensor<1x16x3072xf32>
    %860 = stablehlo.multiply %859, %849 : tensor<1x16x3072xf32>
    %861 = stablehlo.add %860, %cst_13 : tensor<1x16x3072xf32>
    %862 = stablehlo.multiply %cst_14, %849 : tensor<1x16x3072xf32>
    %863 = stablehlo.add %862, %cst_15 : tensor<1x16x3072xf32>
    %864 = stablehlo.multiply %863, %849 : tensor<1x16x3072xf32>
    %865 = stablehlo.add %864, %cst_16 : tensor<1x16x3072xf32>
    %866 = stablehlo.multiply %865, %849 : tensor<1x16x3072xf32>
    %867 = stablehlo.add %866, %cst_17 : tensor<1x16x3072xf32>
    %868 = stablehlo.multiply %867, %849 : tensor<1x16x3072xf32>
    %869 = stablehlo.add %868, %cst_18 : tensor<1x16x3072xf32>
    %870 = stablehlo.multiply %848, %861 : tensor<1x16x3072xf32>
    %871 = stablehlo.divide %870, %869 : tensor<1x16x3072xf32>
    %872 = stablehlo.clamp %cst_19, %871, %cst_20 : tensor<1x16x3072xf32>
    %873 = stablehlo.convert %872 : (tensor<1x16x3072xf32>) -> tensor<1x16x3072xbf16>
    %874 = stablehlo.add %873, %cst_2 : tensor<1x16x3072xbf16>
    %875 = stablehlo.multiply %874, %845 : tensor<1x16x3072xbf16>
    %876 = stablehlo.reshape %875 : (tensor<1x16x3072xbf16>) -> tensor<16x3072xbf16>
    %877 = stablehlo.convert %876 : (tensor<16x3072xbf16>) -> tensor<16x3072xf32>
    %878 = stablehlo.dot_general %877, %arg77, contracting_dims = [1] x [0] : (tensor<16x3072xf32>, tensor<3072x768xf32>) -> tensor<16x768xf32>
    %879 = stablehlo.broadcast_in_dim %878, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %880 = stablehlo.multiply %879, %69 : tensor<16x768xf32>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %882 = stablehlo.broadcast_in_dim %arg78, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %883 = stablehlo.add %881, %882 : tensor<16x768xf32>
    %884 = stablehlo.convert %883 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %885 = stablehlo.reshape %884 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %886 = stablehlo.add %885, %834 : tensor<1x16x768xbf16>
    %887 = stablehlo.convert %886 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %888 = stablehlo.convert %887 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %889 = stablehlo.reduce(%888 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %890 = stablehlo.reshape %889 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %891 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %892 = stablehlo.divide %891, %10 : tensor<1x16x1xf64>
    %893 = stablehlo.broadcast_in_dim %888, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %894 = stablehlo.broadcast_in_dim %892, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %895 = stablehlo.subtract %893, %894 : tensor<1x16x768xf64>
    %896 = stablehlo.multiply %895, %895 : tensor<1x16x768xf64>
    %897 = stablehlo.reduce(%896 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %898 = stablehlo.reshape %897 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %899 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %900 = stablehlo.divide %899, %10 : tensor<1x16x1xf64>
    %901 = stablehlo.convert %900 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %902 = stablehlo.reduce(%887 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %903 = stablehlo.reshape %902 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %905 = stablehlo.divide %904, %26 : tensor<1x16x1xf32>
    %906 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %907 = stablehlo.add %906, %31 : tensor<1x16x1xf32>
    %908 = stablehlo.rsqrt %907 : tensor<1x16x1xf32>
    %909 = stablehlo.broadcast_in_dim %887, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %910 = stablehlo.broadcast_in_dim %905, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %911 = stablehlo.subtract %909, %910 : tensor<1x16x768xf32>
    %912 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %913 = stablehlo.broadcast_in_dim %908, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %914 = stablehlo.multiply %912, %913 : tensor<1x16x768xf32>
    %915 = stablehlo.convert %arg19 : (tensor<768xbf16>) -> tensor<768xf32>
    %916 = stablehlo.broadcast_in_dim %914, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %917 = stablehlo.broadcast_in_dim %915, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %918 = stablehlo.multiply %916, %917 : tensor<1x16x768xf32>
    %919 = stablehlo.convert %arg20 : (tensor<768xbf16>) -> tensor<768xf32>
    %920 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %921 = stablehlo.broadcast_in_dim %919, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %922 = stablehlo.add %920, %921 : tensor<1x16x768xf32>
    %923 = stablehlo.convert %922 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %924 = stablehlo.reshape %923 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %925 = stablehlo.convert %924 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %926 = stablehlo.dot_general %925, %arg79, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %927 = stablehlo.broadcast_in_dim %926, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %928 = stablehlo.multiply %927, %69 : tensor<16x768xf32>
    %929 = stablehlo.broadcast_in_dim %928, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %930 = stablehlo.broadcast_in_dim %arg80, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %931 = stablehlo.add %929, %930 : tensor<16x768xf32>
    %932 = stablehlo.convert %931 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %933 = stablehlo.reshape %932 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %934 = stablehlo.reshape %933 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %935 = stablehlo.transpose %934, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %936 = stablehlo.dot_general %925, %arg81, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %937 = stablehlo.broadcast_in_dim %936, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %938 = stablehlo.multiply %937, %69 : tensor<16x768xf32>
    %939 = stablehlo.broadcast_in_dim %938, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %940 = stablehlo.broadcast_in_dim %arg82, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %941 = stablehlo.add %939, %940 : tensor<16x768xf32>
    %942 = stablehlo.convert %941 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %943 = stablehlo.reshape %942 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %944 = stablehlo.reshape %943 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %945 = stablehlo.transpose %944, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %946 = stablehlo.dot_general %925, %arg83, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %947 = stablehlo.broadcast_in_dim %946, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %948 = stablehlo.multiply %947, %69 : tensor<16x768xf32>
    %949 = stablehlo.broadcast_in_dim %948, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %950 = stablehlo.broadcast_in_dim %arg84, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %951 = stablehlo.add %949, %950 : tensor<16x768xf32>
    %952 = stablehlo.convert %951 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %953 = stablehlo.reshape %952 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %954 = stablehlo.reshape %953 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %955 = stablehlo.transpose %954, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %956 = stablehlo.convert %935 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %957 = stablehlo.convert %945 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %958 = stablehlo.convert %955 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %959 = stablehlo.broadcast_in_dim %956, dims = [0, 1, 2, 3] : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %960 = stablehlo.multiply %959, %104 : tensor<1x12x16x64xf32>
    %961 = stablehlo.transpose %957, dims = [0, 1, 3, 2] : (tensor<1x12x16x64xf32>) -> tensor<1x12x64x16xf32>
    %962 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2, 3] : (tensor<1x12x64x16xf32>) -> tensor<1x12x64x16xf32>
    %963 = stablehlo.multiply %962, %108 : tensor<1x12x64x16xf32>
    %964 = stablehlo.reshape %960 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %965 = stablehlo.reshape %963 : (tensor<1x12x64x16xf32>) -> tensor<12x64x16xf32>
    %966 = stablehlo.broadcast_in_dim %965, dims = [0, 1, 2] : (tensor<12x64x16xf32>) -> tensor<12x64x16xf32>
    %967 = stablehlo.dot_general %964, %966, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x64xf32>, tensor<12x64x16xf32>) -> tensor<12x16x16xf32>
    %968 = stablehlo.reshape %967 : (tensor<12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %969 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %970 = stablehlo.add %969, %117 : tensor<1x12x16x16xf32>
    %971 = stablehlo.reduce(%970 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %972 = stablehlo.reshape %971 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %973 = stablehlo.broadcast_in_dim %970, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %974 = stablehlo.broadcast_in_dim %972, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %975 = stablehlo.subtract %973, %974 : tensor<1x12x16x16xf32>
    %976 = stablehlo.exponential %975 : tensor<1x12x16x16xf32>
    %977 = stablehlo.reduce(%976 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %978 = stablehlo.reshape %977 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %979 = stablehlo.broadcast_in_dim %976, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %980 = stablehlo.broadcast_in_dim %978, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %981 = stablehlo.divide %979, %980 : tensor<1x12x16x16xf32>
    %982 = stablehlo.compare  EQ, %973, %133,  FLOAT : (tensor<1x12x16x16xf32>, tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xi1>
    %983 = stablehlo.reduce(%982 init: %c) applies stablehlo.and across dimensions = [3] : (tensor<1x12x16x16xi1>, tensor<i1>) -> tensor<1x12x16xi1>
    %984 = stablehlo.reshape %983 : (tensor<1x12x16xi1>) -> tensor<1x12x16x1xi1>
    %985 = stablehlo.broadcast_in_dim %984, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xi1>) -> tensor<1x12x16x16xi1>
    %986 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %987 = stablehlo.select %985, %139, %986 : tensor<1x12x16x16xi1>, tensor<1x12x16x16xf32>
    %988 = stablehlo.reshape %987 : (tensor<1x12x16x16xf32>) -> tensor<12x16x16xf32>
    %989 = stablehlo.reshape %958 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %990 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2] : (tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %991 = stablehlo.dot_general %988, %990, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x16xf32>, tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %992 = stablehlo.reshape %991 : (tensor<12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %993 = stablehlo.convert %992 : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xbf16>
    %994 = stablehlo.transpose %993, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %995 = stablehlo.transpose %994, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %996 = stablehlo.transpose %995, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %997 = stablehlo.reshape %996 : (tensor<1x16x12x64xbf16>) -> tensor<1x16x768xbf16>
    %998 = stablehlo.reshape %997 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %999 = stablehlo.convert %998 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %1000 = stablehlo.dot_general %999, %arg85, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %1001 = stablehlo.broadcast_in_dim %1000, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1002 = stablehlo.multiply %1001, %69 : tensor<16x768xf32>
    %1003 = stablehlo.broadcast_in_dim %1002, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1004 = stablehlo.broadcast_in_dim %arg86, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1005 = stablehlo.add %1003, %1004 : tensor<16x768xf32>
    %1006 = stablehlo.convert %1005 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1007 = stablehlo.reshape %1006 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1008 = stablehlo.add %1007, %923 : tensor<1x16x768xbf16>
    %1009 = stablehlo.convert %1008 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %1010 = stablehlo.convert %1009 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %1011 = stablehlo.reduce(%1010 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1012 = stablehlo.reshape %1011 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1014 = stablehlo.divide %1013, %10 : tensor<1x16x1xf64>
    %1015 = stablehlo.broadcast_in_dim %1010, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %1016 = stablehlo.broadcast_in_dim %1014, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %1017 = stablehlo.subtract %1015, %1016 : tensor<1x16x768xf64>
    %1018 = stablehlo.multiply %1017, %1017 : tensor<1x16x768xf64>
    %1019 = stablehlo.reduce(%1018 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1020 = stablehlo.reshape %1019 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1021 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1022 = stablehlo.divide %1021, %10 : tensor<1x16x1xf64>
    %1023 = stablehlo.convert %1022 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %1024 = stablehlo.reduce(%1009 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %1025 = stablehlo.reshape %1024 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %1026 = stablehlo.broadcast_in_dim %1025, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1027 = stablehlo.divide %1026, %26 : tensor<1x16x1xf32>
    %1028 = stablehlo.broadcast_in_dim %1023, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1029 = stablehlo.add %1028, %31 : tensor<1x16x1xf32>
    %1030 = stablehlo.rsqrt %1029 : tensor<1x16x1xf32>
    %1031 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1032 = stablehlo.broadcast_in_dim %1027, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1033 = stablehlo.subtract %1031, %1032 : tensor<1x16x768xf32>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1035 = stablehlo.broadcast_in_dim %1030, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1036 = stablehlo.multiply %1034, %1035 : tensor<1x16x768xf32>
    %1037 = stablehlo.convert %arg21 : (tensor<768xbf16>) -> tensor<768xf32>
    %1038 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1039 = stablehlo.broadcast_in_dim %1037, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1040 = stablehlo.multiply %1038, %1039 : tensor<1x16x768xf32>
    %1041 = stablehlo.convert %arg22 : (tensor<768xbf16>) -> tensor<768xf32>
    %1042 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1043 = stablehlo.broadcast_in_dim %1041, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1044 = stablehlo.add %1042, %1043 : tensor<1x16x768xf32>
    %1045 = stablehlo.convert %1044 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %1046 = stablehlo.reshape %1045 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %1047 = stablehlo.convert %1046 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %1048 = stablehlo.dot_general %1047, %arg87, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x3072xf32>) -> tensor<16x3072xf32>
    %1049 = stablehlo.broadcast_in_dim %1048, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %1050 = stablehlo.multiply %1049, %204 : tensor<16x3072xf32>
    %1051 = stablehlo.broadcast_in_dim %1050, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %1052 = stablehlo.broadcast_in_dim %arg88, dims = [1] : (tensor<3072xf32>) -> tensor<16x3072xf32>
    %1053 = stablehlo.add %1051, %1052 : tensor<16x3072xf32>
    %1054 = stablehlo.convert %1053 : (tensor<16x3072xf32>) -> tensor<16x3072xbf16>
    %1055 = stablehlo.reshape %1054 : (tensor<16x3072xbf16>) -> tensor<1x16x3072xbf16>
    %1056 = stablehlo.multiply %1055, %cst_4 : tensor<1x16x3072xbf16>
    %1057 = stablehlo.multiply %1055, %212 : tensor<1x16x3072xbf16>
    %1058 = stablehlo.convert %1057 : (tensor<1x16x3072xbf16>) -> tensor<1x16x3072xf32>
    %1059 = stablehlo.clamp %cst_5, %1058, %cst_6 : tensor<1x16x3072xf32>
    %1060 = stablehlo.multiply %1059, %1059 : tensor<1x16x3072xf32>
    %1061 = stablehlo.multiply %cst_7, %1060 : tensor<1x16x3072xf32>
    %1062 = stablehlo.add %1061, %cst_8 : tensor<1x16x3072xf32>
    %1063 = stablehlo.multiply %1062, %1060 : tensor<1x16x3072xf32>
    %1064 = stablehlo.add %1063, %cst_9 : tensor<1x16x3072xf32>
    %1065 = stablehlo.multiply %1064, %1060 : tensor<1x16x3072xf32>
    %1066 = stablehlo.add %1065, %cst_10 : tensor<1x16x3072xf32>
    %1067 = stablehlo.multiply %1066, %1060 : tensor<1x16x3072xf32>
    %1068 = stablehlo.add %1067, %cst_11 : tensor<1x16x3072xf32>
    %1069 = stablehlo.multiply %1068, %1060 : tensor<1x16x3072xf32>
    %1070 = stablehlo.add %1069, %cst_12 : tensor<1x16x3072xf32>
    %1071 = stablehlo.multiply %1070, %1060 : tensor<1x16x3072xf32>
    %1072 = stablehlo.add %1071, %cst_13 : tensor<1x16x3072xf32>
    %1073 = stablehlo.multiply %cst_14, %1060 : tensor<1x16x3072xf32>
    %1074 = stablehlo.add %1073, %cst_15 : tensor<1x16x3072xf32>
    %1075 = stablehlo.multiply %1074, %1060 : tensor<1x16x3072xf32>
    %1076 = stablehlo.add %1075, %cst_16 : tensor<1x16x3072xf32>
    %1077 = stablehlo.multiply %1076, %1060 : tensor<1x16x3072xf32>
    %1078 = stablehlo.add %1077, %cst_17 : tensor<1x16x3072xf32>
    %1079 = stablehlo.multiply %1078, %1060 : tensor<1x16x3072xf32>
    %1080 = stablehlo.add %1079, %cst_18 : tensor<1x16x3072xf32>
    %1081 = stablehlo.multiply %1059, %1072 : tensor<1x16x3072xf32>
    %1082 = stablehlo.divide %1081, %1080 : tensor<1x16x3072xf32>
    %1083 = stablehlo.clamp %cst_19, %1082, %cst_20 : tensor<1x16x3072xf32>
    %1084 = stablehlo.convert %1083 : (tensor<1x16x3072xf32>) -> tensor<1x16x3072xbf16>
    %1085 = stablehlo.add %1084, %cst_2 : tensor<1x16x3072xbf16>
    %1086 = stablehlo.multiply %1085, %1056 : tensor<1x16x3072xbf16>
    %1087 = stablehlo.reshape %1086 : (tensor<1x16x3072xbf16>) -> tensor<16x3072xbf16>
    %1088 = stablehlo.convert %1087 : (tensor<16x3072xbf16>) -> tensor<16x3072xf32>
    %1089 = stablehlo.dot_general %1088, %arg89, contracting_dims = [1] x [0] : (tensor<16x3072xf32>, tensor<3072x768xf32>) -> tensor<16x768xf32>
    %1090 = stablehlo.broadcast_in_dim %1089, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1091 = stablehlo.multiply %1090, %69 : tensor<16x768xf32>
    %1092 = stablehlo.broadcast_in_dim %1091, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1093 = stablehlo.broadcast_in_dim %arg90, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1094 = stablehlo.add %1092, %1093 : tensor<16x768xf32>
    %1095 = stablehlo.convert %1094 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1096 = stablehlo.reshape %1095 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1097 = stablehlo.add %1096, %1045 : tensor<1x16x768xbf16>
    %1098 = stablehlo.convert %1097 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %1099 = stablehlo.convert %1098 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %1100 = stablehlo.reduce(%1099 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1101 = stablehlo.reshape %1100 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1102 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1103 = stablehlo.divide %1102, %10 : tensor<1x16x1xf64>
    %1104 = stablehlo.broadcast_in_dim %1099, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %1105 = stablehlo.broadcast_in_dim %1103, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %1106 = stablehlo.subtract %1104, %1105 : tensor<1x16x768xf64>
    %1107 = stablehlo.multiply %1106, %1106 : tensor<1x16x768xf64>
    %1108 = stablehlo.reduce(%1107 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1109 = stablehlo.reshape %1108 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1110 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1111 = stablehlo.divide %1110, %10 : tensor<1x16x1xf64>
    %1112 = stablehlo.convert %1111 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %1113 = stablehlo.reduce(%1098 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %1114 = stablehlo.reshape %1113 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1116 = stablehlo.divide %1115, %26 : tensor<1x16x1xf32>
    %1117 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1118 = stablehlo.add %1117, %31 : tensor<1x16x1xf32>
    %1119 = stablehlo.rsqrt %1118 : tensor<1x16x1xf32>
    %1120 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1121 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1122 = stablehlo.subtract %1120, %1121 : tensor<1x16x768xf32>
    %1123 = stablehlo.broadcast_in_dim %1122, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1124 = stablehlo.broadcast_in_dim %1119, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1125 = stablehlo.multiply %1123, %1124 : tensor<1x16x768xf32>
    %1126 = stablehlo.convert %arg23 : (tensor<768xbf16>) -> tensor<768xf32>
    %1127 = stablehlo.broadcast_in_dim %1125, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1128 = stablehlo.broadcast_in_dim %1126, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1129 = stablehlo.multiply %1127, %1128 : tensor<1x16x768xf32>
    %1130 = stablehlo.convert %arg24 : (tensor<768xbf16>) -> tensor<768xf32>
    %1131 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1132 = stablehlo.broadcast_in_dim %1130, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1133 = stablehlo.add %1131, %1132 : tensor<1x16x768xf32>
    %1134 = stablehlo.convert %1133 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %1135 = stablehlo.reshape %1134 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %1136 = stablehlo.convert %1135 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %1137 = stablehlo.dot_general %1136, %arg91, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1139 = stablehlo.multiply %1138, %69 : tensor<16x768xf32>
    %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1141 = stablehlo.broadcast_in_dim %arg92, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1142 = stablehlo.add %1140, %1141 : tensor<16x768xf32>
    %1143 = stablehlo.convert %1142 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1144 = stablehlo.reshape %1143 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1145 = stablehlo.reshape %1144 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %1146 = stablehlo.transpose %1145, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %1147 = stablehlo.dot_general %1136, %arg93, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %1148 = stablehlo.broadcast_in_dim %1147, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1149 = stablehlo.multiply %1148, %69 : tensor<16x768xf32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1151 = stablehlo.broadcast_in_dim %arg94, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1152 = stablehlo.add %1150, %1151 : tensor<16x768xf32>
    %1153 = stablehlo.convert %1152 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1154 = stablehlo.reshape %1153 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1155 = stablehlo.reshape %1154 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %1156 = stablehlo.transpose %1155, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %1157 = stablehlo.dot_general %1136, %arg95, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %1158 = stablehlo.broadcast_in_dim %1157, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1159 = stablehlo.multiply %1158, %69 : tensor<16x768xf32>
    %1160 = stablehlo.broadcast_in_dim %1159, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1161 = stablehlo.broadcast_in_dim %arg96, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1162 = stablehlo.add %1160, %1161 : tensor<16x768xf32>
    %1163 = stablehlo.convert %1162 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1164 = stablehlo.reshape %1163 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1165 = stablehlo.reshape %1164 : (tensor<1x16x768xbf16>) -> tensor<1x16x12x64xbf16>
    %1166 = stablehlo.transpose %1165, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %1167 = stablehlo.convert %1146 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %1168 = stablehlo.convert %1156 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %1169 = stablehlo.convert %1166 : (tensor<1x12x16x64xbf16>) -> tensor<1x12x16x64xf32>
    %1170 = stablehlo.broadcast_in_dim %1167, dims = [0, 1, 2, 3] : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1171 = stablehlo.multiply %1170, %104 : tensor<1x12x16x64xf32>
    %1172 = stablehlo.transpose %1168, dims = [0, 1, 3, 2] : (tensor<1x12x16x64xf32>) -> tensor<1x12x64x16xf32>
    %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1, 2, 3] : (tensor<1x12x64x16xf32>) -> tensor<1x12x64x16xf32>
    %1174 = stablehlo.multiply %1173, %108 : tensor<1x12x64x16xf32>
    %1175 = stablehlo.reshape %1171 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %1176 = stablehlo.reshape %1174 : (tensor<1x12x64x16xf32>) -> tensor<12x64x16xf32>
    %1177 = stablehlo.broadcast_in_dim %1176, dims = [0, 1, 2] : (tensor<12x64x16xf32>) -> tensor<12x64x16xf32>
    %1178 = stablehlo.dot_general %1175, %1177, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x64xf32>, tensor<12x64x16xf32>) -> tensor<12x16x16xf32>
    %1179 = stablehlo.reshape %1178 : (tensor<12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %1180 = stablehlo.broadcast_in_dim %1179, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %1181 = stablehlo.add %1180, %117 : tensor<1x12x16x16xf32>
    %1182 = stablehlo.reduce(%1181 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %1183 = stablehlo.reshape %1182 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %1184 = stablehlo.broadcast_in_dim %1181, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %1185 = stablehlo.broadcast_in_dim %1183, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %1186 = stablehlo.subtract %1184, %1185 : tensor<1x12x16x16xf32>
    %1187 = stablehlo.exponential %1186 : tensor<1x12x16x16xf32>
    %1188 = stablehlo.reduce(%1187 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x16x16xf32>, tensor<f32>) -> tensor<1x12x16xf32>
    %1189 = stablehlo.reshape %1188 : (tensor<1x12x16xf32>) -> tensor<1x12x16x1xf32>
    %1190 = stablehlo.broadcast_in_dim %1187, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %1191 = stablehlo.broadcast_in_dim %1189, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xf32>) -> tensor<1x12x16x16xf32>
    %1192 = stablehlo.divide %1190, %1191 : tensor<1x12x16x16xf32>
    %1193 = stablehlo.compare  EQ, %1184, %133,  FLOAT : (tensor<1x12x16x16xf32>, tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xi1>
    %1194 = stablehlo.reduce(%1193 init: %c) applies stablehlo.and across dimensions = [3] : (tensor<1x12x16x16xi1>, tensor<i1>) -> tensor<1x12x16xi1>
    %1195 = stablehlo.reshape %1194 : (tensor<1x12x16xi1>) -> tensor<1x12x16x1xi1>
    %1196 = stablehlo.broadcast_in_dim %1195, dims = [0, 1, 2, 3] : (tensor<1x12x16x1xi1>) -> tensor<1x12x16x16xi1>
    %1197 = stablehlo.broadcast_in_dim %1192, dims = [0, 1, 2, 3] : (tensor<1x12x16x16xf32>) -> tensor<1x12x16x16xf32>
    %1198 = stablehlo.select %1196, %139, %1197 : tensor<1x12x16x16xi1>, tensor<1x12x16x16xf32>
    %1199 = stablehlo.reshape %1198 : (tensor<1x12x16x16xf32>) -> tensor<12x16x16xf32>
    %1200 = stablehlo.reshape %1169 : (tensor<1x12x16x64xf32>) -> tensor<12x16x64xf32>
    %1201 = stablehlo.broadcast_in_dim %1200, dims = [0, 1, 2] : (tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %1202 = stablehlo.dot_general %1199, %1201, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x16x16xf32>, tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %1203 = stablehlo.reshape %1202 : (tensor<12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1204 = stablehlo.convert %1203 : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xbf16>
    %1205 = stablehlo.transpose %1204, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %1206 = stablehlo.transpose %1205, dims = [0, 2, 1, 3] : (tensor<1x16x12x64xbf16>) -> tensor<1x12x16x64xbf16>
    %1207 = stablehlo.transpose %1206, dims = [0, 2, 1, 3] : (tensor<1x12x16x64xbf16>) -> tensor<1x16x12x64xbf16>
    %1208 = stablehlo.reshape %1207 : (tensor<1x16x12x64xbf16>) -> tensor<1x16x768xbf16>
    %1209 = stablehlo.reshape %1208 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %1210 = stablehlo.convert %1209 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %1211 = stablehlo.dot_general %1210, %arg97, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x768xf32>) -> tensor<16x768xf32>
    %1212 = stablehlo.broadcast_in_dim %1211, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1213 = stablehlo.multiply %1212, %69 : tensor<16x768xf32>
    %1214 = stablehlo.broadcast_in_dim %1213, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1215 = stablehlo.broadcast_in_dim %arg98, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1216 = stablehlo.add %1214, %1215 : tensor<16x768xf32>
    %1217 = stablehlo.convert %1216 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1218 = stablehlo.reshape %1217 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1219 = stablehlo.add %1218, %1134 : tensor<1x16x768xbf16>
    %1220 = stablehlo.convert %1219 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %1221 = stablehlo.convert %1220 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %1222 = stablehlo.reduce(%1221 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1223 = stablehlo.reshape %1222 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1224 = stablehlo.broadcast_in_dim %1223, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1225 = stablehlo.divide %1224, %10 : tensor<1x16x1xf64>
    %1226 = stablehlo.broadcast_in_dim %1221, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %1227 = stablehlo.broadcast_in_dim %1225, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %1228 = stablehlo.subtract %1226, %1227 : tensor<1x16x768xf64>
    %1229 = stablehlo.multiply %1228, %1228 : tensor<1x16x768xf64>
    %1230 = stablehlo.reduce(%1229 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1231 = stablehlo.reshape %1230 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1232 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1233 = stablehlo.divide %1232, %10 : tensor<1x16x1xf64>
    %1234 = stablehlo.convert %1233 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %1235 = stablehlo.reduce(%1220 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %1236 = stablehlo.reshape %1235 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %1237 = stablehlo.broadcast_in_dim %1236, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1238 = stablehlo.divide %1237, %26 : tensor<1x16x1xf32>
    %1239 = stablehlo.broadcast_in_dim %1234, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1240 = stablehlo.add %1239, %31 : tensor<1x16x1xf32>
    %1241 = stablehlo.rsqrt %1240 : tensor<1x16x1xf32>
    %1242 = stablehlo.broadcast_in_dim %1220, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1243 = stablehlo.broadcast_in_dim %1238, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1244 = stablehlo.subtract %1242, %1243 : tensor<1x16x768xf32>
    %1245 = stablehlo.broadcast_in_dim %1244, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1246 = stablehlo.broadcast_in_dim %1241, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1247 = stablehlo.multiply %1245, %1246 : tensor<1x16x768xf32>
    %1248 = stablehlo.convert %arg25 : (tensor<768xbf16>) -> tensor<768xf32>
    %1249 = stablehlo.broadcast_in_dim %1247, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1250 = stablehlo.broadcast_in_dim %1248, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1251 = stablehlo.multiply %1249, %1250 : tensor<1x16x768xf32>
    %1252 = stablehlo.convert %arg26 : (tensor<768xbf16>) -> tensor<768xf32>
    %1253 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1254 = stablehlo.broadcast_in_dim %1252, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1255 = stablehlo.add %1253, %1254 : tensor<1x16x768xf32>
    %1256 = stablehlo.convert %1255 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    %1257 = stablehlo.reshape %1256 : (tensor<1x16x768xbf16>) -> tensor<16x768xbf16>
    %1258 = stablehlo.convert %1257 : (tensor<16x768xbf16>) -> tensor<16x768xf32>
    %1259 = stablehlo.dot_general %1258, %arg99, contracting_dims = [1] x [0] : (tensor<16x768xf32>, tensor<768x3072xf32>) -> tensor<16x3072xf32>
    %1260 = stablehlo.broadcast_in_dim %1259, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %1261 = stablehlo.multiply %1260, %204 : tensor<16x3072xf32>
    %1262 = stablehlo.broadcast_in_dim %1261, dims = [0, 1] : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %1263 = stablehlo.broadcast_in_dim %arg100, dims = [1] : (tensor<3072xf32>) -> tensor<16x3072xf32>
    %1264 = stablehlo.add %1262, %1263 : tensor<16x3072xf32>
    %1265 = stablehlo.convert %1264 : (tensor<16x3072xf32>) -> tensor<16x3072xbf16>
    %1266 = stablehlo.reshape %1265 : (tensor<16x3072xbf16>) -> tensor<1x16x3072xbf16>
    %1267 = stablehlo.multiply %1266, %cst_4 : tensor<1x16x3072xbf16>
    %1268 = stablehlo.multiply %1266, %212 : tensor<1x16x3072xbf16>
    %1269 = stablehlo.convert %1268 : (tensor<1x16x3072xbf16>) -> tensor<1x16x3072xf32>
    %1270 = stablehlo.clamp %cst_5, %1269, %cst_6 : tensor<1x16x3072xf32>
    %1271 = stablehlo.multiply %1270, %1270 : tensor<1x16x3072xf32>
    %1272 = stablehlo.multiply %cst_7, %1271 : tensor<1x16x3072xf32>
    %1273 = stablehlo.add %1272, %cst_8 : tensor<1x16x3072xf32>
    %1274 = stablehlo.multiply %1273, %1271 : tensor<1x16x3072xf32>
    %1275 = stablehlo.add %1274, %cst_9 : tensor<1x16x3072xf32>
    %1276 = stablehlo.multiply %1275, %1271 : tensor<1x16x3072xf32>
    %1277 = stablehlo.add %1276, %cst_10 : tensor<1x16x3072xf32>
    %1278 = stablehlo.multiply %1277, %1271 : tensor<1x16x3072xf32>
    %1279 = stablehlo.add %1278, %cst_11 : tensor<1x16x3072xf32>
    %1280 = stablehlo.multiply %1279, %1271 : tensor<1x16x3072xf32>
    %1281 = stablehlo.add %1280, %cst_12 : tensor<1x16x3072xf32>
    %1282 = stablehlo.multiply %1281, %1271 : tensor<1x16x3072xf32>
    %1283 = stablehlo.add %1282, %cst_13 : tensor<1x16x3072xf32>
    %1284 = stablehlo.multiply %cst_14, %1271 : tensor<1x16x3072xf32>
    %1285 = stablehlo.add %1284, %cst_15 : tensor<1x16x3072xf32>
    %1286 = stablehlo.multiply %1285, %1271 : tensor<1x16x3072xf32>
    %1287 = stablehlo.add %1286, %cst_16 : tensor<1x16x3072xf32>
    %1288 = stablehlo.multiply %1287, %1271 : tensor<1x16x3072xf32>
    %1289 = stablehlo.add %1288, %cst_17 : tensor<1x16x3072xf32>
    %1290 = stablehlo.multiply %1289, %1271 : tensor<1x16x3072xf32>
    %1291 = stablehlo.add %1290, %cst_18 : tensor<1x16x3072xf32>
    %1292 = stablehlo.multiply %1270, %1283 : tensor<1x16x3072xf32>
    %1293 = stablehlo.divide %1292, %1291 : tensor<1x16x3072xf32>
    %1294 = stablehlo.clamp %cst_19, %1293, %cst_20 : tensor<1x16x3072xf32>
    %1295 = stablehlo.convert %1294 : (tensor<1x16x3072xf32>) -> tensor<1x16x3072xbf16>
    %1296 = stablehlo.add %1295, %cst_2 : tensor<1x16x3072xbf16>
    %1297 = stablehlo.multiply %1296, %1267 : tensor<1x16x3072xbf16>
    %1298 = stablehlo.reshape %1297 : (tensor<1x16x3072xbf16>) -> tensor<16x3072xbf16>
    %1299 = stablehlo.convert %1298 : (tensor<16x3072xbf16>) -> tensor<16x3072xf32>
    %1300 = stablehlo.dot_general %1299, %arg101, contracting_dims = [1] x [0] : (tensor<16x3072xf32>, tensor<3072x768xf32>) -> tensor<16x768xf32>
    %1301 = stablehlo.broadcast_in_dim %1300, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1302 = stablehlo.multiply %1301, %69 : tensor<16x768xf32>
    %1303 = stablehlo.broadcast_in_dim %1302, dims = [0, 1] : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1304 = stablehlo.broadcast_in_dim %arg102, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1305 = stablehlo.add %1303, %1304 : tensor<16x768xf32>
    %1306 = stablehlo.convert %1305 : (tensor<16x768xf32>) -> tensor<16x768xbf16>
    %1307 = stablehlo.reshape %1306 : (tensor<16x768xbf16>) -> tensor<1x16x768xbf16>
    %1308 = stablehlo.add %1307, %1256 : tensor<1x16x768xbf16>
    %1309 = stablehlo.convert %1308 : (tensor<1x16x768xbf16>) -> tensor<1x16x768xf32>
    %1310 = stablehlo.convert %1309 : (tensor<1x16x768xf32>) -> tensor<1x16x768xf64>
    %1311 = stablehlo.reduce(%1310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1312 = stablehlo.reshape %1311 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1313 = stablehlo.broadcast_in_dim %1312, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1314 = stablehlo.divide %1313, %10 : tensor<1x16x1xf64>
    %1315 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2] : (tensor<1x16x768xf64>) -> tensor<1x16x768xf64>
    %1316 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x768xf64>
    %1317 = stablehlo.subtract %1315, %1316 : tensor<1x16x768xf64>
    %1318 = stablehlo.multiply %1317, %1317 : tensor<1x16x768xf64>
    %1319 = stablehlo.reduce(%1318 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf64>, tensor<f64>) -> tensor<1x16xf64>
    %1320 = stablehlo.reshape %1319 : (tensor<1x16xf64>) -> tensor<1x16x1xf64>
    %1321 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2] : (tensor<1x16x1xf64>) -> tensor<1x16x1xf64>
    %1322 = stablehlo.divide %1321, %10 : tensor<1x16x1xf64>
    %1323 = stablehlo.convert %1322 : (tensor<1x16x1xf64>) -> tensor<1x16x1xf32>
    %1324 = stablehlo.reduce(%1309 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16xf32>
    %1325 = stablehlo.reshape %1324 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %1326 = stablehlo.broadcast_in_dim %1325, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1327 = stablehlo.divide %1326, %26 : tensor<1x16x1xf32>
    %1328 = stablehlo.broadcast_in_dim %1323, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1329 = stablehlo.add %1328, %31 : tensor<1x16x1xf32>
    %1330 = stablehlo.rsqrt %1329 : tensor<1x16x1xf32>
    %1331 = stablehlo.broadcast_in_dim %1309, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1332 = stablehlo.broadcast_in_dim %1327, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1333 = stablehlo.subtract %1331, %1332 : tensor<1x16x768xf32>
    %1334 = stablehlo.broadcast_in_dim %1333, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1335 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x768xf32>
    %1336 = stablehlo.multiply %1334, %1335 : tensor<1x16x768xf32>
    %1337 = stablehlo.convert %arg27 : (tensor<768xbf16>) -> tensor<768xf32>
    %1338 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1339 = stablehlo.broadcast_in_dim %1337, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1340 = stablehlo.multiply %1338, %1339 : tensor<1x16x768xf32>
    %1341 = stablehlo.convert %arg28 : (tensor<768xbf16>) -> tensor<768xf32>
    %1342 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
    %1343 = stablehlo.broadcast_in_dim %1341, dims = [2] : (tensor<768xf32>) -> tensor<1x16x768xf32>
    %1344 = stablehlo.add %1342, %1343 : tensor<1x16x768xf32>
    %1345 = stablehlo.convert %1344 : (tensor<1x16x768xf32>) -> tensor<1x16x768xbf16>
    return %1345 : tensor<1x16x768xbf16>
  }
}
