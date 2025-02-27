module {
  func.func @main(%arg0: tensor<1x3x32x128xbf16>, %arg1: tensor<768x3x4x4xbf16>, %arg2: tensor<768xbf16>, %arg3: tensor<1x257x768xbf16>, %arg4: tensor<768xbf16>, %arg5: tensor<768xbf16>, %arg6: tensor<768xbf16>, %arg7: tensor<768xbf16>, %arg8: tensor<768xbf16>, %arg9: tensor<768xbf16>, %arg10: tensor<768xbf16>, %arg11: tensor<768xbf16>, %arg12: tensor<768xbf16>, %arg13: tensor<768xbf16>, %arg14: tensor<768xbf16>, %arg15: tensor<768xbf16>, %arg16: tensor<768xbf16>, %arg17: tensor<768xbf16>, %arg18: tensor<768xbf16>, %arg19: tensor<768xbf16>, %arg20: tensor<768xbf16>, %arg21: tensor<768xbf16>, %arg22: tensor<768xbf16>, %arg23: tensor<768xbf16>, %arg24: tensor<768xbf16>, %arg25: tensor<768xbf16>, %arg26: tensor<768xbf16>, %arg27: tensor<768xbf16>, %arg28: tensor<768xbf16>, %arg29: tensor<768xbf16>, %arg30: tensor<768xbf16>, %arg31: tensor<768xbf16>, %arg32: tensor<768xbf16>, %arg33: tensor<768xbf16>, %arg34: tensor<768xbf16>, %arg35: tensor<768xbf16>, %arg36: tensor<768xbf16>, %arg37: tensor<768xbf16>, %arg38: tensor<768xbf16>, %arg39: tensor<768xbf16>, %arg40: tensor<768xbf16>, %arg41: tensor<768xbf16>, %arg42: tensor<768xbf16>, %arg43: tensor<768xbf16>, %arg44: tensor<768xbf16>, %arg45: tensor<768xbf16>, %arg46: tensor<768xbf16>, %arg47: tensor<768xbf16>, %arg48: tensor<768xbf16>, %arg49: tensor<768xbf16>, %arg50: tensor<768xbf16>, %arg51: tensor<768xbf16>, %arg52: tensor<768xbf16>, %arg53: tensor<768xbf16>, %arg54: tensor<768x96x1x1xbf16>, %arg55: tensor<27x768x1x1xbf16>, %arg56: tensor<768x96x1x1xbf16>, %arg57: tensor<768xbf16>, %arg58: tensor<768xbf16>, %arg59: tensor<768xbf16>, %arg60: tensor<768xbf16>, %arg61: tensor<768x96x1x1xbf16>, %arg62: tensor<27x768x1x1xbf16>, %arg63: tensor<768x96x1x1xbf16>, %arg64: tensor<768xbf16>, %arg65: tensor<768xbf16>, %arg66: tensor<768xbf16>, %arg67: tensor<768xbf16>, %arg68: tensor<768x96x1x1xbf16>, %arg69: tensor<27x768x1x1xbf16>, %arg70: tensor<768x96x1x1xbf16>, %arg71: tensor<768xbf16>, %arg72: tensor<768xbf16>, %arg73: tensor<1x1x768xbf16>, %arg74: tensor<768x2304xf32>, %arg75: tensor<2304xf32>, %arg76: tensor<768x768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768x3072xf32>, %arg79: tensor<3072xf32>, %arg80: tensor<3072x768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768x2304xf32>, %arg83: tensor<2304xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768x3072xf32>, %arg87: tensor<3072xf32>, %arg88: tensor<3072x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x2304xf32>, %arg91: tensor<2304xf32>, %arg92: tensor<768x768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768x3072xf32>, %arg95: tensor<3072xf32>, %arg96: tensor<3072x768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768x2304xf32>, %arg99: tensor<2304xf32>, %arg100: tensor<768x768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768x3072xf32>, %arg103: tensor<3072xf32>, %arg104: tensor<3072x768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<768x2304xf32>, %arg107: tensor<2304xf32>, %arg108: tensor<768x768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<3072xf32>, %arg112: tensor<3072x768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768x2304xf32>, %arg115: tensor<2304xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768x3072xf32>, %arg119: tensor<3072xf32>, %arg120: tensor<3072x768xf32>, %arg121: tensor<768xf32>, %arg122: tensor<768x2304xf32>, %arg123: tensor<2304xf32>, %arg124: tensor<768x768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768x3072xf32>, %arg127: tensor<3072xf32>, %arg128: tensor<3072x768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768x2304xf32>, %arg131: tensor<2304xf32>, %arg132: tensor<768x768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768x3072xf32>, %arg135: tensor<3072xf32>, %arg136: tensor<3072x768xf32>, %arg137: tensor<768xf32>, %arg138: tensor<768x2304xf32>, %arg139: tensor<2304xf32>, %arg140: tensor<768x768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768x3072xf32>, %arg143: tensor<3072xf32>, %arg144: tensor<3072x768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768x2304xf32>, %arg147: tensor<2304xf32>, %arg148: tensor<768x768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768x3072xf32>, %arg151: tensor<3072xf32>, %arg152: tensor<3072x768xf32>, %arg153: tensor<768xf32>, %arg154: tensor<768x2304xf32>, %arg155: tensor<2304xf32>, %arg156: tensor<768x768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768x3072xf32>, %arg159: tensor<3072xf32>, %arg160: tensor<3072x768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768x2304xf32>, %arg163: tensor<2304xf32>, %arg164: tensor<768x768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x3072xf32>, %arg167: tensor<3072xf32>, %arg168: tensor<3072x768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768x38xf32>, %arg171: tensor<38xf32>, %arg172: tensor<768x50257xf32>, %arg173: tensor<50257xf32>, %arg174: tensor<768x30522xf32>, %arg175: tensor<30522xf32>) -> (tensor<1x27x38xbf16>, tensor<1x27x50257xbf16>, tensor<1x27x30522xbf16>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<1x257x3072xbf16>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<1x257x3072xbf16>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<1x257x3072xbf16>
    %cst_5 = stablehlo.constant dense<-4.000000e+00> : tensor<1x257x3072xf32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<1x257x3072xf32>
    %cst_7 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x257x3072xf32>
    %cst_8 = stablehlo.constant dense<2.77068146E-8> : tensor<1x257x3072xf32>
    %cst_9 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x257x3072xf32>
    %cst_10 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x257x3072xf32>
    %cst_11 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x257x3072xf32>
    %cst_12 = stablehlo.constant dense<-2.954600e-03> : tensor<1x257x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.0160960332> : tensor<1x257x3072xf32>
    %cst_14 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x257x3072xf32>
    %cst_15 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x257x3072xf32>
    %cst_16 = stablehlo.constant dense<-0.00168282702> : tensor<1x257x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.00737332925> : tensor<1x257x3072xf32>
    %cst_18 = stablehlo.constant dense<-0.0142647391> : tensor<1x257x3072xf32>
    %cst_19 = stablehlo.constant dense<-1.000000e+00> : tensor<1x257x3072xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x257x3072xf32>
    %cst_21 = arith.constant dense<768> : tensor<1xi64>
    %cst_22 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_23 = arith.constant dense<1> : tensor<1xi64>
    %cst_24 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x32x128xbf16>, tensor<768x3x4x4xbf16>) -> tensor<1x768x8x32xbf16>
    %1 = stablehlo.reshape %arg2 : (tensor<768xbf16>) -> tensor<768x1x1xbf16>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x768x8x32xbf16>) -> tensor<1x768x8x32xbf16>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<768x1x1xbf16>) -> tensor<1x768x8x32xbf16>
    %4 = stablehlo.add %2, %3 : tensor<1x768x8x32xbf16>
    %5 = stablehlo.reshape %4 : (tensor<1x768x8x32xbf16>) -> tensor<1x768x256xbf16>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1] : (tensor<1x768x256xbf16>) -> tensor<1x256x768xbf16>
    %7 = stablehlo.concatenate %arg73, %6, dim = 1 : (tensor<1x1x768xbf16>, tensor<1x256x768xbf16>) -> tensor<1x257x768xbf16>
    %8 = stablehlo.add %7, %arg3 : tensor<1x257x768xbf16>
    %9 = stablehlo.convert %8 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %10 = stablehlo.convert %9 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %12 = stablehlo.reshape %11 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %13 = stablehlo.convert %cst_21 : (tensor<1xi64>) -> tensor<1xf64>
    %14 = stablehlo.reshape %13 : (tensor<1xf64>) -> tensor<f64>
    %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %16 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f64>) -> tensor<1x257x1xf64>
    %17 = stablehlo.divide %15, %16 : tensor<1x257x1xf64>
    %18 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %20 = stablehlo.subtract %18, %19 : tensor<1x257x768xf64>
    %21 = stablehlo.multiply %20, %20 : tensor<1x257x768xf64>
    %22 = stablehlo.reduce(%21 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %23 = stablehlo.reshape %22 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %25 = stablehlo.divide %24, %16 : tensor<1x257x1xf64>
    %26 = stablehlo.convert %25 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %27 = stablehlo.reduce(%9 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %28 = stablehlo.reshape %27 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %29 = stablehlo.convert %cst_21 : (tensor<1xi64>) -> tensor<1xf32>
    %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
    %31 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<1x257x1xf32>
    %33 = stablehlo.divide %31, %32 : tensor<1x257x1xf32>
    %34 = stablehlo.convert %cst_22 : (tensor<1xf64>) -> tensor<1xf32>
    %35 = stablehlo.reshape %34 : (tensor<1xf32>) -> tensor<f32>
    %36 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %37 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<f32>) -> tensor<1x257x1xf32>
    %38 = stablehlo.add %36, %37 : tensor<1x257x1xf32>
    %39 = stablehlo.rsqrt %38 : tensor<1x257x1xf32>
    %40 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %41 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %42 = stablehlo.subtract %40, %41 : tensor<1x257x768xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %44 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<1x257x768xf32>
    %46 = stablehlo.convert %arg4 : (tensor<768xbf16>) -> tensor<768xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %48 = stablehlo.broadcast_in_dim %46, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %49 = stablehlo.multiply %47, %48 : tensor<1x257x768xf32>
    %50 = stablehlo.convert %arg5 : (tensor<768xbf16>) -> tensor<768xf32>
    %51 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %52 = stablehlo.broadcast_in_dim %50, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %53 = stablehlo.add %51, %52 : tensor<1x257x768xf32>
    %54 = stablehlo.convert %53 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %55 = stablehlo.reshape %54 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %56 = stablehlo.convert %55 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %57 = stablehlo.dot_general %56, %arg74, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %58 = stablehlo.convert %cst_23 : (tensor<1xi64>) -> tensor<1xf32>
    %59 = stablehlo.reshape %58 : (tensor<1xf32>) -> tensor<f32>
    %60 = stablehlo.broadcast_in_dim %57, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %61 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<257x2304xf32>
    %62 = stablehlo.multiply %60, %61 : tensor<257x2304xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %64 = stablehlo.broadcast_in_dim %arg75, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %65 = stablehlo.add %63, %64 : tensor<257x2304xf32>
    %66 = stablehlo.convert %65 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %67 = stablehlo.reshape %66 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %68 = stablehlo.reshape %67 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %69 = stablehlo.transpose %68, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %70 = stablehlo.slice %69 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %71 = stablehlo.reshape %70 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %72 = stablehlo.slice %69 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %73 = stablehlo.reshape %72 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %74 = stablehlo.slice %69 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %75 = stablehlo.reshape %74 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %76 = stablehlo.transpose %73, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %77 = stablehlo.reshape %71 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %78 = stablehlo.reshape %76 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %80 = stablehlo.dot_general %77, %79, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %81 = stablehlo.reshape %80 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %82 = stablehlo.convert %cst_24 : (tensor<1xf64>) -> tensor<1xbf16>
    %83 = stablehlo.reshape %82 : (tensor<1xbf16>) -> tensor<bf16>
    %84 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %85 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<bf16>) -> tensor<1x12x257x257xbf16>
    %86 = stablehlo.multiply %84, %85 : tensor<1x12x257x257xbf16>
    %87 = stablehlo.convert %86 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %88 = stablehlo.reduce(%87 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %89 = stablehlo.reshape %88 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %90 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %91 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %92 = stablehlo.subtract %90, %91 : tensor<1x12x257x257xf32>
    %93 = stablehlo.exponential %92 : tensor<1x12x257x257xf32>
    %94 = stablehlo.reduce(%93 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %95 = stablehlo.reshape %94 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %96 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %97 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %98 = stablehlo.divide %96, %97 : tensor<1x12x257x257xf32>
    %99 = stablehlo.convert %98 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %100 = stablehlo.reshape %99 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %101 = stablehlo.reshape %75 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %103 = stablehlo.dot_general %100, %102, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %104 = stablehlo.reshape %103 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %105 = stablehlo.transpose %104, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %106 = stablehlo.reshape %105 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %107 = stablehlo.reshape %106 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %108 = stablehlo.convert %107 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %109 = stablehlo.dot_general %108, %arg76, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %111 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<257x768xf32>
    %112 = stablehlo.multiply %110, %111 : tensor<257x768xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %114 = stablehlo.broadcast_in_dim %arg77, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %115 = stablehlo.add %113, %114 : tensor<257x768xf32>
    %116 = stablehlo.convert %115 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %117 = stablehlo.reshape %116 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %118 = stablehlo.add %117, %8 : tensor<1x257x768xbf16>
    %119 = stablehlo.convert %118 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %120 = stablehlo.convert %119 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %121 = stablehlo.reduce(%120 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %122 = stablehlo.reshape %121 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %124 = stablehlo.divide %123, %16 : tensor<1x257x1xf64>
    %125 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %126 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %127 = stablehlo.subtract %125, %126 : tensor<1x257x768xf64>
    %128 = stablehlo.multiply %127, %127 : tensor<1x257x768xf64>
    %129 = stablehlo.reduce(%128 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %130 = stablehlo.reshape %129 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %132 = stablehlo.divide %131, %16 : tensor<1x257x1xf64>
    %133 = stablehlo.convert %132 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %134 = stablehlo.reduce(%119 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %135 = stablehlo.reshape %134 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %137 = stablehlo.divide %136, %32 : tensor<1x257x1xf32>
    %138 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %139 = stablehlo.add %138, %37 : tensor<1x257x1xf32>
    %140 = stablehlo.rsqrt %139 : tensor<1x257x1xf32>
    %141 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %142 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %143 = stablehlo.subtract %141, %142 : tensor<1x257x768xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %145 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %146 = stablehlo.multiply %144, %145 : tensor<1x257x768xf32>
    %147 = stablehlo.convert %arg6 : (tensor<768xbf16>) -> tensor<768xf32>
    %148 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %149 = stablehlo.broadcast_in_dim %147, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %150 = stablehlo.multiply %148, %149 : tensor<1x257x768xf32>
    %151 = stablehlo.convert %arg7 : (tensor<768xbf16>) -> tensor<768xf32>
    %152 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %153 = stablehlo.broadcast_in_dim %151, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %154 = stablehlo.add %152, %153 : tensor<1x257x768xf32>
    %155 = stablehlo.convert %154 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %156 = stablehlo.reshape %155 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %157 = stablehlo.convert %156 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %158 = stablehlo.dot_general %157, %arg78, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %160 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<257x3072xf32>
    %161 = stablehlo.multiply %159, %160 : tensor<257x3072xf32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %163 = stablehlo.broadcast_in_dim %arg79, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %164 = stablehlo.add %162, %163 : tensor<257x3072xf32>
    %165 = stablehlo.convert %164 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %166 = stablehlo.reshape %165 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %167 = stablehlo.multiply %166, %cst_4 : tensor<1x257x3072xbf16>
    %168 = stablehlo.rsqrt %cst_3 : tensor<1x257x3072xbf16>
    %169 = stablehlo.multiply %166, %168 : tensor<1x257x3072xbf16>
    %170 = stablehlo.convert %169 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %171 = stablehlo.clamp %cst_5, %170, %cst_6 : tensor<1x257x3072xf32>
    %172 = stablehlo.multiply %171, %171 : tensor<1x257x3072xf32>
    %173 = stablehlo.multiply %cst_7, %172 : tensor<1x257x3072xf32>
    %174 = stablehlo.add %173, %cst_8 : tensor<1x257x3072xf32>
    %175 = stablehlo.multiply %174, %172 : tensor<1x257x3072xf32>
    %176 = stablehlo.add %175, %cst_9 : tensor<1x257x3072xf32>
    %177 = stablehlo.multiply %176, %172 : tensor<1x257x3072xf32>
    %178 = stablehlo.add %177, %cst_10 : tensor<1x257x3072xf32>
    %179 = stablehlo.multiply %178, %172 : tensor<1x257x3072xf32>
    %180 = stablehlo.add %179, %cst_11 : tensor<1x257x3072xf32>
    %181 = stablehlo.multiply %180, %172 : tensor<1x257x3072xf32>
    %182 = stablehlo.add %181, %cst_12 : tensor<1x257x3072xf32>
    %183 = stablehlo.multiply %182, %172 : tensor<1x257x3072xf32>
    %184 = stablehlo.add %183, %cst_13 : tensor<1x257x3072xf32>
    %185 = stablehlo.multiply %cst_14, %172 : tensor<1x257x3072xf32>
    %186 = stablehlo.add %185, %cst_15 : tensor<1x257x3072xf32>
    %187 = stablehlo.multiply %186, %172 : tensor<1x257x3072xf32>
    %188 = stablehlo.add %187, %cst_16 : tensor<1x257x3072xf32>
    %189 = stablehlo.multiply %188, %172 : tensor<1x257x3072xf32>
    %190 = stablehlo.add %189, %cst_17 : tensor<1x257x3072xf32>
    %191 = stablehlo.multiply %190, %172 : tensor<1x257x3072xf32>
    %192 = stablehlo.add %191, %cst_18 : tensor<1x257x3072xf32>
    %193 = stablehlo.multiply %171, %184 : tensor<1x257x3072xf32>
    %194 = stablehlo.divide %193, %192 : tensor<1x257x3072xf32>
    %195 = stablehlo.clamp %cst_19, %194, %cst_20 : tensor<1x257x3072xf32>
    %196 = stablehlo.convert %195 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %197 = stablehlo.add %196, %cst_2 : tensor<1x257x3072xbf16>
    %198 = stablehlo.multiply %197, %167 : tensor<1x257x3072xbf16>
    %199 = stablehlo.reshape %198 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %200 = stablehlo.convert %199 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %201 = stablehlo.dot_general %200, %arg80, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %203 = stablehlo.multiply %202, %111 : tensor<257x768xf32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %205 = stablehlo.broadcast_in_dim %arg81, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %206 = stablehlo.add %204, %205 : tensor<257x768xf32>
    %207 = stablehlo.convert %206 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %208 = stablehlo.reshape %207 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %209 = stablehlo.add %118, %208 : tensor<1x257x768xbf16>
    %210 = stablehlo.convert %209 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %211 = stablehlo.convert %210 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %212 = stablehlo.reduce(%211 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %213 = stablehlo.reshape %212 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %215 = stablehlo.divide %214, %16 : tensor<1x257x1xf64>
    %216 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %217 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %218 = stablehlo.subtract %216, %217 : tensor<1x257x768xf64>
    %219 = stablehlo.multiply %218, %218 : tensor<1x257x768xf64>
    %220 = stablehlo.reduce(%219 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %221 = stablehlo.reshape %220 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %222 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %223 = stablehlo.divide %222, %16 : tensor<1x257x1xf64>
    %224 = stablehlo.convert %223 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %225 = stablehlo.reduce(%210 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %226 = stablehlo.reshape %225 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %228 = stablehlo.divide %227, %32 : tensor<1x257x1xf32>
    %229 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %230 = stablehlo.add %229, %37 : tensor<1x257x1xf32>
    %231 = stablehlo.rsqrt %230 : tensor<1x257x1xf32>
    %232 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %233 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %234 = stablehlo.subtract %232, %233 : tensor<1x257x768xf32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %236 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %237 = stablehlo.multiply %235, %236 : tensor<1x257x768xf32>
    %238 = stablehlo.convert %arg8 : (tensor<768xbf16>) -> tensor<768xf32>
    %239 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %240 = stablehlo.broadcast_in_dim %238, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %241 = stablehlo.multiply %239, %240 : tensor<1x257x768xf32>
    %242 = stablehlo.convert %arg9 : (tensor<768xbf16>) -> tensor<768xf32>
    %243 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %244 = stablehlo.broadcast_in_dim %242, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %245 = stablehlo.add %243, %244 : tensor<1x257x768xf32>
    %246 = stablehlo.convert %245 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %247 = stablehlo.reshape %246 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %248 = stablehlo.convert %247 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %249 = stablehlo.dot_general %248, %arg82, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %251 = stablehlo.multiply %250, %61 : tensor<257x2304xf32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %253 = stablehlo.broadcast_in_dim %arg83, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %254 = stablehlo.add %252, %253 : tensor<257x2304xf32>
    %255 = stablehlo.convert %254 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %256 = stablehlo.reshape %255 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %257 = stablehlo.reshape %256 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %258 = stablehlo.transpose %257, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %259 = stablehlo.slice %258 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %260 = stablehlo.reshape %259 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %261 = stablehlo.slice %258 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %262 = stablehlo.reshape %261 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %263 = stablehlo.slice %258 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %264 = stablehlo.reshape %263 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %265 = stablehlo.transpose %262, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %266 = stablehlo.reshape %260 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %267 = stablehlo.reshape %265 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %269 = stablehlo.dot_general %266, %268, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %270 = stablehlo.reshape %269 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %271 = stablehlo.broadcast_in_dim %270, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %272 = stablehlo.multiply %271, %85 : tensor<1x12x257x257xbf16>
    %273 = stablehlo.convert %272 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %274 = stablehlo.reduce(%273 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %275 = stablehlo.reshape %274 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %276 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %277 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %278 = stablehlo.subtract %276, %277 : tensor<1x12x257x257xf32>
    %279 = stablehlo.exponential %278 : tensor<1x12x257x257xf32>
    %280 = stablehlo.reduce(%279 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %281 = stablehlo.reshape %280 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %282 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %283 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %284 = stablehlo.divide %282, %283 : tensor<1x12x257x257xf32>
    %285 = stablehlo.convert %284 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %286 = stablehlo.reshape %285 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %287 = stablehlo.reshape %264 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %288 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %289 = stablehlo.dot_general %286, %288, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %290 = stablehlo.reshape %289 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %291 = stablehlo.transpose %290, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %292 = stablehlo.reshape %291 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %293 = stablehlo.reshape %292 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %294 = stablehlo.convert %293 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %295 = stablehlo.dot_general %294, %arg84, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %297 = stablehlo.multiply %296, %111 : tensor<257x768xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %299 = stablehlo.broadcast_in_dim %arg85, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %300 = stablehlo.add %298, %299 : tensor<257x768xf32>
    %301 = stablehlo.convert %300 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %302 = stablehlo.reshape %301 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %303 = stablehlo.add %302, %209 : tensor<1x257x768xbf16>
    %304 = stablehlo.convert %303 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %305 = stablehlo.convert %304 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %307 = stablehlo.reshape %306 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %309 = stablehlo.divide %308, %16 : tensor<1x257x1xf64>
    %310 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %311 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %312 = stablehlo.subtract %310, %311 : tensor<1x257x768xf64>
    %313 = stablehlo.multiply %312, %312 : tensor<1x257x768xf64>
    %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %315 = stablehlo.reshape %314 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %317 = stablehlo.divide %316, %16 : tensor<1x257x1xf64>
    %318 = stablehlo.convert %317 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %319 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %320 = stablehlo.reshape %319 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %322 = stablehlo.divide %321, %32 : tensor<1x257x1xf32>
    %323 = stablehlo.broadcast_in_dim %318, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %324 = stablehlo.add %323, %37 : tensor<1x257x1xf32>
    %325 = stablehlo.rsqrt %324 : tensor<1x257x1xf32>
    %326 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %327 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %328 = stablehlo.subtract %326, %327 : tensor<1x257x768xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %330 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %331 = stablehlo.multiply %329, %330 : tensor<1x257x768xf32>
    %332 = stablehlo.convert %arg10 : (tensor<768xbf16>) -> tensor<768xf32>
    %333 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %334 = stablehlo.broadcast_in_dim %332, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %335 = stablehlo.multiply %333, %334 : tensor<1x257x768xf32>
    %336 = stablehlo.convert %arg11 : (tensor<768xbf16>) -> tensor<768xf32>
    %337 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %338 = stablehlo.broadcast_in_dim %336, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %339 = stablehlo.add %337, %338 : tensor<1x257x768xf32>
    %340 = stablehlo.convert %339 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %341 = stablehlo.reshape %340 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %342 = stablehlo.convert %341 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %343 = stablehlo.dot_general %342, %arg86, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %345 = stablehlo.multiply %344, %160 : tensor<257x3072xf32>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %347 = stablehlo.broadcast_in_dim %arg87, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %348 = stablehlo.add %346, %347 : tensor<257x3072xf32>
    %349 = stablehlo.convert %348 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %350 = stablehlo.reshape %349 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %351 = stablehlo.multiply %350, %cst_4 : tensor<1x257x3072xbf16>
    %352 = stablehlo.multiply %350, %168 : tensor<1x257x3072xbf16>
    %353 = stablehlo.convert %352 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %354 = stablehlo.clamp %cst_5, %353, %cst_6 : tensor<1x257x3072xf32>
    %355 = stablehlo.multiply %354, %354 : tensor<1x257x3072xf32>
    %356 = stablehlo.multiply %cst_7, %355 : tensor<1x257x3072xf32>
    %357 = stablehlo.add %356, %cst_8 : tensor<1x257x3072xf32>
    %358 = stablehlo.multiply %357, %355 : tensor<1x257x3072xf32>
    %359 = stablehlo.add %358, %cst_9 : tensor<1x257x3072xf32>
    %360 = stablehlo.multiply %359, %355 : tensor<1x257x3072xf32>
    %361 = stablehlo.add %360, %cst_10 : tensor<1x257x3072xf32>
    %362 = stablehlo.multiply %361, %355 : tensor<1x257x3072xf32>
    %363 = stablehlo.add %362, %cst_11 : tensor<1x257x3072xf32>
    %364 = stablehlo.multiply %363, %355 : tensor<1x257x3072xf32>
    %365 = stablehlo.add %364, %cst_12 : tensor<1x257x3072xf32>
    %366 = stablehlo.multiply %365, %355 : tensor<1x257x3072xf32>
    %367 = stablehlo.add %366, %cst_13 : tensor<1x257x3072xf32>
    %368 = stablehlo.multiply %cst_14, %355 : tensor<1x257x3072xf32>
    %369 = stablehlo.add %368, %cst_15 : tensor<1x257x3072xf32>
    %370 = stablehlo.multiply %369, %355 : tensor<1x257x3072xf32>
    %371 = stablehlo.add %370, %cst_16 : tensor<1x257x3072xf32>
    %372 = stablehlo.multiply %371, %355 : tensor<1x257x3072xf32>
    %373 = stablehlo.add %372, %cst_17 : tensor<1x257x3072xf32>
    %374 = stablehlo.multiply %373, %355 : tensor<1x257x3072xf32>
    %375 = stablehlo.add %374, %cst_18 : tensor<1x257x3072xf32>
    %376 = stablehlo.multiply %354, %367 : tensor<1x257x3072xf32>
    %377 = stablehlo.divide %376, %375 : tensor<1x257x3072xf32>
    %378 = stablehlo.clamp %cst_19, %377, %cst_20 : tensor<1x257x3072xf32>
    %379 = stablehlo.convert %378 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %380 = stablehlo.add %379, %cst_2 : tensor<1x257x3072xbf16>
    %381 = stablehlo.multiply %380, %351 : tensor<1x257x3072xbf16>
    %382 = stablehlo.reshape %381 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %383 = stablehlo.convert %382 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %384 = stablehlo.dot_general %383, %arg88, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %386 = stablehlo.multiply %385, %111 : tensor<257x768xf32>
    %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %388 = stablehlo.broadcast_in_dim %arg89, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %389 = stablehlo.add %387, %388 : tensor<257x768xf32>
    %390 = stablehlo.convert %389 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %391 = stablehlo.reshape %390 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %392 = stablehlo.add %303, %391 : tensor<1x257x768xbf16>
    %393 = stablehlo.convert %392 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %394 = stablehlo.convert %393 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %395 = stablehlo.reduce(%394 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %396 = stablehlo.reshape %395 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %397 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %398 = stablehlo.divide %397, %16 : tensor<1x257x1xf64>
    %399 = stablehlo.broadcast_in_dim %394, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %400 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %401 = stablehlo.subtract %399, %400 : tensor<1x257x768xf64>
    %402 = stablehlo.multiply %401, %401 : tensor<1x257x768xf64>
    %403 = stablehlo.reduce(%402 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %404 = stablehlo.reshape %403 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %405 = stablehlo.broadcast_in_dim %404, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %406 = stablehlo.divide %405, %16 : tensor<1x257x1xf64>
    %407 = stablehlo.convert %406 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %408 = stablehlo.reduce(%393 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %409 = stablehlo.reshape %408 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %410 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %411 = stablehlo.divide %410, %32 : tensor<1x257x1xf32>
    %412 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %413 = stablehlo.add %412, %37 : tensor<1x257x1xf32>
    %414 = stablehlo.rsqrt %413 : tensor<1x257x1xf32>
    %415 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %416 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %417 = stablehlo.subtract %415, %416 : tensor<1x257x768xf32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %419 = stablehlo.broadcast_in_dim %414, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %420 = stablehlo.multiply %418, %419 : tensor<1x257x768xf32>
    %421 = stablehlo.convert %arg12 : (tensor<768xbf16>) -> tensor<768xf32>
    %422 = stablehlo.broadcast_in_dim %420, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %423 = stablehlo.broadcast_in_dim %421, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %424 = stablehlo.multiply %422, %423 : tensor<1x257x768xf32>
    %425 = stablehlo.convert %arg13 : (tensor<768xbf16>) -> tensor<768xf32>
    %426 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %427 = stablehlo.broadcast_in_dim %425, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %428 = stablehlo.add %426, %427 : tensor<1x257x768xf32>
    %429 = stablehlo.convert %428 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %430 = stablehlo.reshape %429 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %431 = stablehlo.convert %430 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %432 = stablehlo.dot_general %431, %arg90, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %433 = stablehlo.broadcast_in_dim %432, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %434 = stablehlo.multiply %433, %61 : tensor<257x2304xf32>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %436 = stablehlo.broadcast_in_dim %arg91, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %437 = stablehlo.add %435, %436 : tensor<257x2304xf32>
    %438 = stablehlo.convert %437 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %439 = stablehlo.reshape %438 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %440 = stablehlo.reshape %439 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %441 = stablehlo.transpose %440, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %442 = stablehlo.slice %441 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %443 = stablehlo.reshape %442 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %444 = stablehlo.slice %441 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %445 = stablehlo.reshape %444 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %446 = stablehlo.slice %441 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %447 = stablehlo.reshape %446 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %448 = stablehlo.transpose %445, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %449 = stablehlo.reshape %443 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %450 = stablehlo.reshape %448 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %451 = stablehlo.broadcast_in_dim %450, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %452 = stablehlo.dot_general %449, %451, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %453 = stablehlo.reshape %452 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %455 = stablehlo.multiply %454, %85 : tensor<1x12x257x257xbf16>
    %456 = stablehlo.convert %455 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %457 = stablehlo.reduce(%456 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %458 = stablehlo.reshape %457 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %459 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %460 = stablehlo.broadcast_in_dim %458, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %461 = stablehlo.subtract %459, %460 : tensor<1x12x257x257xf32>
    %462 = stablehlo.exponential %461 : tensor<1x12x257x257xf32>
    %463 = stablehlo.reduce(%462 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %464 = stablehlo.reshape %463 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %465 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %466 = stablehlo.broadcast_in_dim %464, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %467 = stablehlo.divide %465, %466 : tensor<1x12x257x257xf32>
    %468 = stablehlo.convert %467 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %469 = stablehlo.reshape %468 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %470 = stablehlo.reshape %447 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %471 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %472 = stablehlo.dot_general %469, %471, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %473 = stablehlo.reshape %472 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %474 = stablehlo.transpose %473, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %475 = stablehlo.reshape %474 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %476 = stablehlo.reshape %475 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %477 = stablehlo.convert %476 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %478 = stablehlo.dot_general %477, %arg92, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %480 = stablehlo.multiply %479, %111 : tensor<257x768xf32>
    %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %482 = stablehlo.broadcast_in_dim %arg93, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %483 = stablehlo.add %481, %482 : tensor<257x768xf32>
    %484 = stablehlo.convert %483 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %485 = stablehlo.reshape %484 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %486 = stablehlo.add %485, %392 : tensor<1x257x768xbf16>
    %487 = stablehlo.convert %486 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %488 = stablehlo.convert %487 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %489 = stablehlo.reduce(%488 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %490 = stablehlo.reshape %489 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %491 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %492 = stablehlo.divide %491, %16 : tensor<1x257x1xf64>
    %493 = stablehlo.broadcast_in_dim %488, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %494 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %495 = stablehlo.subtract %493, %494 : tensor<1x257x768xf64>
    %496 = stablehlo.multiply %495, %495 : tensor<1x257x768xf64>
    %497 = stablehlo.reduce(%496 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %498 = stablehlo.reshape %497 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %499 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %500 = stablehlo.divide %499, %16 : tensor<1x257x1xf64>
    %501 = stablehlo.convert %500 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %502 = stablehlo.reduce(%487 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %503 = stablehlo.reshape %502 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %504 = stablehlo.broadcast_in_dim %503, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %505 = stablehlo.divide %504, %32 : tensor<1x257x1xf32>
    %506 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %507 = stablehlo.add %506, %37 : tensor<1x257x1xf32>
    %508 = stablehlo.rsqrt %507 : tensor<1x257x1xf32>
    %509 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %510 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %511 = stablehlo.subtract %509, %510 : tensor<1x257x768xf32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %513 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %514 = stablehlo.multiply %512, %513 : tensor<1x257x768xf32>
    %515 = stablehlo.convert %arg14 : (tensor<768xbf16>) -> tensor<768xf32>
    %516 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %517 = stablehlo.broadcast_in_dim %515, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %518 = stablehlo.multiply %516, %517 : tensor<1x257x768xf32>
    %519 = stablehlo.convert %arg15 : (tensor<768xbf16>) -> tensor<768xf32>
    %520 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %521 = stablehlo.broadcast_in_dim %519, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %522 = stablehlo.add %520, %521 : tensor<1x257x768xf32>
    %523 = stablehlo.convert %522 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %524 = stablehlo.reshape %523 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %525 = stablehlo.convert %524 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %526 = stablehlo.dot_general %525, %arg94, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %527 = stablehlo.broadcast_in_dim %526, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %528 = stablehlo.multiply %527, %160 : tensor<257x3072xf32>
    %529 = stablehlo.broadcast_in_dim %528, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %530 = stablehlo.broadcast_in_dim %arg95, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %531 = stablehlo.add %529, %530 : tensor<257x3072xf32>
    %532 = stablehlo.convert %531 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %533 = stablehlo.reshape %532 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %534 = stablehlo.multiply %533, %cst_4 : tensor<1x257x3072xbf16>
    %535 = stablehlo.multiply %533, %168 : tensor<1x257x3072xbf16>
    %536 = stablehlo.convert %535 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %537 = stablehlo.clamp %cst_5, %536, %cst_6 : tensor<1x257x3072xf32>
    %538 = stablehlo.multiply %537, %537 : tensor<1x257x3072xf32>
    %539 = stablehlo.multiply %cst_7, %538 : tensor<1x257x3072xf32>
    %540 = stablehlo.add %539, %cst_8 : tensor<1x257x3072xf32>
    %541 = stablehlo.multiply %540, %538 : tensor<1x257x3072xf32>
    %542 = stablehlo.add %541, %cst_9 : tensor<1x257x3072xf32>
    %543 = stablehlo.multiply %542, %538 : tensor<1x257x3072xf32>
    %544 = stablehlo.add %543, %cst_10 : tensor<1x257x3072xf32>
    %545 = stablehlo.multiply %544, %538 : tensor<1x257x3072xf32>
    %546 = stablehlo.add %545, %cst_11 : tensor<1x257x3072xf32>
    %547 = stablehlo.multiply %546, %538 : tensor<1x257x3072xf32>
    %548 = stablehlo.add %547, %cst_12 : tensor<1x257x3072xf32>
    %549 = stablehlo.multiply %548, %538 : tensor<1x257x3072xf32>
    %550 = stablehlo.add %549, %cst_13 : tensor<1x257x3072xf32>
    %551 = stablehlo.multiply %cst_14, %538 : tensor<1x257x3072xf32>
    %552 = stablehlo.add %551, %cst_15 : tensor<1x257x3072xf32>
    %553 = stablehlo.multiply %552, %538 : tensor<1x257x3072xf32>
    %554 = stablehlo.add %553, %cst_16 : tensor<1x257x3072xf32>
    %555 = stablehlo.multiply %554, %538 : tensor<1x257x3072xf32>
    %556 = stablehlo.add %555, %cst_17 : tensor<1x257x3072xf32>
    %557 = stablehlo.multiply %556, %538 : tensor<1x257x3072xf32>
    %558 = stablehlo.add %557, %cst_18 : tensor<1x257x3072xf32>
    %559 = stablehlo.multiply %537, %550 : tensor<1x257x3072xf32>
    %560 = stablehlo.divide %559, %558 : tensor<1x257x3072xf32>
    %561 = stablehlo.clamp %cst_19, %560, %cst_20 : tensor<1x257x3072xf32>
    %562 = stablehlo.convert %561 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %563 = stablehlo.add %562, %cst_2 : tensor<1x257x3072xbf16>
    %564 = stablehlo.multiply %563, %534 : tensor<1x257x3072xbf16>
    %565 = stablehlo.reshape %564 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %566 = stablehlo.convert %565 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %567 = stablehlo.dot_general %566, %arg96, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %568 = stablehlo.broadcast_in_dim %567, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %569 = stablehlo.multiply %568, %111 : tensor<257x768xf32>
    %570 = stablehlo.broadcast_in_dim %569, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %571 = stablehlo.broadcast_in_dim %arg97, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %572 = stablehlo.add %570, %571 : tensor<257x768xf32>
    %573 = stablehlo.convert %572 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %574 = stablehlo.reshape %573 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %575 = stablehlo.add %486, %574 : tensor<1x257x768xbf16>
    %576 = stablehlo.convert %575 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %577 = stablehlo.convert %576 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %578 = stablehlo.reduce(%577 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %579 = stablehlo.reshape %578 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %581 = stablehlo.divide %580, %16 : tensor<1x257x1xf64>
    %582 = stablehlo.broadcast_in_dim %577, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %583 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %584 = stablehlo.subtract %582, %583 : tensor<1x257x768xf64>
    %585 = stablehlo.multiply %584, %584 : tensor<1x257x768xf64>
    %586 = stablehlo.reduce(%585 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %587 = stablehlo.reshape %586 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %588 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %589 = stablehlo.divide %588, %16 : tensor<1x257x1xf64>
    %590 = stablehlo.convert %589 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %591 = stablehlo.reduce(%576 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %592 = stablehlo.reshape %591 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %594 = stablehlo.divide %593, %32 : tensor<1x257x1xf32>
    %595 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %596 = stablehlo.add %595, %37 : tensor<1x257x1xf32>
    %597 = stablehlo.rsqrt %596 : tensor<1x257x1xf32>
    %598 = stablehlo.broadcast_in_dim %576, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %599 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %600 = stablehlo.subtract %598, %599 : tensor<1x257x768xf32>
    %601 = stablehlo.broadcast_in_dim %600, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %602 = stablehlo.broadcast_in_dim %597, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %603 = stablehlo.multiply %601, %602 : tensor<1x257x768xf32>
    %604 = stablehlo.convert %arg16 : (tensor<768xbf16>) -> tensor<768xf32>
    %605 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %606 = stablehlo.broadcast_in_dim %604, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %607 = stablehlo.multiply %605, %606 : tensor<1x257x768xf32>
    %608 = stablehlo.convert %arg17 : (tensor<768xbf16>) -> tensor<768xf32>
    %609 = stablehlo.broadcast_in_dim %607, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %610 = stablehlo.broadcast_in_dim %608, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %611 = stablehlo.add %609, %610 : tensor<1x257x768xf32>
    %612 = stablehlo.convert %611 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %613 = stablehlo.reshape %612 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %614 = stablehlo.convert %613 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %615 = stablehlo.dot_general %614, %arg98, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %616 = stablehlo.broadcast_in_dim %615, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %617 = stablehlo.multiply %616, %61 : tensor<257x2304xf32>
    %618 = stablehlo.broadcast_in_dim %617, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %619 = stablehlo.broadcast_in_dim %arg99, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %620 = stablehlo.add %618, %619 : tensor<257x2304xf32>
    %621 = stablehlo.convert %620 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %622 = stablehlo.reshape %621 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %623 = stablehlo.reshape %622 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %624 = stablehlo.transpose %623, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %625 = stablehlo.slice %624 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %626 = stablehlo.reshape %625 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %627 = stablehlo.slice %624 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %628 = stablehlo.reshape %627 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %629 = stablehlo.slice %624 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %630 = stablehlo.reshape %629 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %631 = stablehlo.transpose %628, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %632 = stablehlo.reshape %626 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %633 = stablehlo.reshape %631 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %634 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %635 = stablehlo.dot_general %632, %634, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %636 = stablehlo.reshape %635 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %637 = stablehlo.broadcast_in_dim %636, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %638 = stablehlo.multiply %637, %85 : tensor<1x12x257x257xbf16>
    %639 = stablehlo.convert %638 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %640 = stablehlo.reduce(%639 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %641 = stablehlo.reshape %640 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %642 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %643 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %644 = stablehlo.subtract %642, %643 : tensor<1x12x257x257xf32>
    %645 = stablehlo.exponential %644 : tensor<1x12x257x257xf32>
    %646 = stablehlo.reduce(%645 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %647 = stablehlo.reshape %646 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %648 = stablehlo.broadcast_in_dim %645, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %649 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %650 = stablehlo.divide %648, %649 : tensor<1x12x257x257xf32>
    %651 = stablehlo.convert %650 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %652 = stablehlo.reshape %651 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %653 = stablehlo.reshape %630 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %655 = stablehlo.dot_general %652, %654, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %656 = stablehlo.reshape %655 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %657 = stablehlo.transpose %656, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %658 = stablehlo.reshape %657 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %659 = stablehlo.reshape %658 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %660 = stablehlo.convert %659 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %661 = stablehlo.dot_general %660, %arg100, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %662 = stablehlo.broadcast_in_dim %661, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %663 = stablehlo.multiply %662, %111 : tensor<257x768xf32>
    %664 = stablehlo.broadcast_in_dim %663, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %665 = stablehlo.broadcast_in_dim %arg101, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %666 = stablehlo.add %664, %665 : tensor<257x768xf32>
    %667 = stablehlo.convert %666 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %668 = stablehlo.reshape %667 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %669 = stablehlo.add %668, %575 : tensor<1x257x768xbf16>
    %670 = stablehlo.convert %669 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %671 = stablehlo.convert %670 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %672 = stablehlo.reduce(%671 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %673 = stablehlo.reshape %672 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %674 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %675 = stablehlo.divide %674, %16 : tensor<1x257x1xf64>
    %676 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %677 = stablehlo.broadcast_in_dim %675, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %678 = stablehlo.subtract %676, %677 : tensor<1x257x768xf64>
    %679 = stablehlo.multiply %678, %678 : tensor<1x257x768xf64>
    %680 = stablehlo.reduce(%679 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %681 = stablehlo.reshape %680 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %682 = stablehlo.broadcast_in_dim %681, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %683 = stablehlo.divide %682, %16 : tensor<1x257x1xf64>
    %684 = stablehlo.convert %683 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %685 = stablehlo.reduce(%670 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %686 = stablehlo.reshape %685 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %687 = stablehlo.broadcast_in_dim %686, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %688 = stablehlo.divide %687, %32 : tensor<1x257x1xf32>
    %689 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %690 = stablehlo.add %689, %37 : tensor<1x257x1xf32>
    %691 = stablehlo.rsqrt %690 : tensor<1x257x1xf32>
    %692 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %693 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %694 = stablehlo.subtract %692, %693 : tensor<1x257x768xf32>
    %695 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %696 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %697 = stablehlo.multiply %695, %696 : tensor<1x257x768xf32>
    %698 = stablehlo.convert %arg18 : (tensor<768xbf16>) -> tensor<768xf32>
    %699 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %700 = stablehlo.broadcast_in_dim %698, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %701 = stablehlo.multiply %699, %700 : tensor<1x257x768xf32>
    %702 = stablehlo.convert %arg19 : (tensor<768xbf16>) -> tensor<768xf32>
    %703 = stablehlo.broadcast_in_dim %701, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %704 = stablehlo.broadcast_in_dim %702, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %705 = stablehlo.add %703, %704 : tensor<1x257x768xf32>
    %706 = stablehlo.convert %705 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %707 = stablehlo.reshape %706 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %708 = stablehlo.convert %707 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %709 = stablehlo.dot_general %708, %arg102, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %711 = stablehlo.multiply %710, %160 : tensor<257x3072xf32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %713 = stablehlo.broadcast_in_dim %arg103, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %714 = stablehlo.add %712, %713 : tensor<257x3072xf32>
    %715 = stablehlo.convert %714 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %716 = stablehlo.reshape %715 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %717 = stablehlo.multiply %716, %cst_4 : tensor<1x257x3072xbf16>
    %718 = stablehlo.multiply %716, %168 : tensor<1x257x3072xbf16>
    %719 = stablehlo.convert %718 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %720 = stablehlo.clamp %cst_5, %719, %cst_6 : tensor<1x257x3072xf32>
    %721 = stablehlo.multiply %720, %720 : tensor<1x257x3072xf32>
    %722 = stablehlo.multiply %cst_7, %721 : tensor<1x257x3072xf32>
    %723 = stablehlo.add %722, %cst_8 : tensor<1x257x3072xf32>
    %724 = stablehlo.multiply %723, %721 : tensor<1x257x3072xf32>
    %725 = stablehlo.add %724, %cst_9 : tensor<1x257x3072xf32>
    %726 = stablehlo.multiply %725, %721 : tensor<1x257x3072xf32>
    %727 = stablehlo.add %726, %cst_10 : tensor<1x257x3072xf32>
    %728 = stablehlo.multiply %727, %721 : tensor<1x257x3072xf32>
    %729 = stablehlo.add %728, %cst_11 : tensor<1x257x3072xf32>
    %730 = stablehlo.multiply %729, %721 : tensor<1x257x3072xf32>
    %731 = stablehlo.add %730, %cst_12 : tensor<1x257x3072xf32>
    %732 = stablehlo.multiply %731, %721 : tensor<1x257x3072xf32>
    %733 = stablehlo.add %732, %cst_13 : tensor<1x257x3072xf32>
    %734 = stablehlo.multiply %cst_14, %721 : tensor<1x257x3072xf32>
    %735 = stablehlo.add %734, %cst_15 : tensor<1x257x3072xf32>
    %736 = stablehlo.multiply %735, %721 : tensor<1x257x3072xf32>
    %737 = stablehlo.add %736, %cst_16 : tensor<1x257x3072xf32>
    %738 = stablehlo.multiply %737, %721 : tensor<1x257x3072xf32>
    %739 = stablehlo.add %738, %cst_17 : tensor<1x257x3072xf32>
    %740 = stablehlo.multiply %739, %721 : tensor<1x257x3072xf32>
    %741 = stablehlo.add %740, %cst_18 : tensor<1x257x3072xf32>
    %742 = stablehlo.multiply %720, %733 : tensor<1x257x3072xf32>
    %743 = stablehlo.divide %742, %741 : tensor<1x257x3072xf32>
    %744 = stablehlo.clamp %cst_19, %743, %cst_20 : tensor<1x257x3072xf32>
    %745 = stablehlo.convert %744 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %746 = stablehlo.add %745, %cst_2 : tensor<1x257x3072xbf16>
    %747 = stablehlo.multiply %746, %717 : tensor<1x257x3072xbf16>
    %748 = stablehlo.reshape %747 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %749 = stablehlo.convert %748 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %750 = stablehlo.dot_general %749, %arg104, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %751 = stablehlo.broadcast_in_dim %750, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %752 = stablehlo.multiply %751, %111 : tensor<257x768xf32>
    %753 = stablehlo.broadcast_in_dim %752, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %754 = stablehlo.broadcast_in_dim %arg105, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %755 = stablehlo.add %753, %754 : tensor<257x768xf32>
    %756 = stablehlo.convert %755 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %757 = stablehlo.reshape %756 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %758 = stablehlo.add %669, %757 : tensor<1x257x768xbf16>
    %759 = stablehlo.convert %758 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %760 = stablehlo.convert %759 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %761 = stablehlo.reduce(%760 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %762 = stablehlo.reshape %761 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %763 = stablehlo.broadcast_in_dim %762, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %764 = stablehlo.divide %763, %16 : tensor<1x257x1xf64>
    %765 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %766 = stablehlo.broadcast_in_dim %764, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %767 = stablehlo.subtract %765, %766 : tensor<1x257x768xf64>
    %768 = stablehlo.multiply %767, %767 : tensor<1x257x768xf64>
    %769 = stablehlo.reduce(%768 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %770 = stablehlo.reshape %769 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %772 = stablehlo.divide %771, %16 : tensor<1x257x1xf64>
    %773 = stablehlo.convert %772 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %774 = stablehlo.reduce(%759 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %775 = stablehlo.reshape %774 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %776 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %777 = stablehlo.divide %776, %32 : tensor<1x257x1xf32>
    %778 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %779 = stablehlo.add %778, %37 : tensor<1x257x1xf32>
    %780 = stablehlo.rsqrt %779 : tensor<1x257x1xf32>
    %781 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %782 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %783 = stablehlo.subtract %781, %782 : tensor<1x257x768xf32>
    %784 = stablehlo.broadcast_in_dim %783, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %785 = stablehlo.broadcast_in_dim %780, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %786 = stablehlo.multiply %784, %785 : tensor<1x257x768xf32>
    %787 = stablehlo.convert %arg20 : (tensor<768xbf16>) -> tensor<768xf32>
    %788 = stablehlo.broadcast_in_dim %786, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %789 = stablehlo.broadcast_in_dim %787, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %790 = stablehlo.multiply %788, %789 : tensor<1x257x768xf32>
    %791 = stablehlo.convert %arg21 : (tensor<768xbf16>) -> tensor<768xf32>
    %792 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %793 = stablehlo.broadcast_in_dim %791, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %794 = stablehlo.add %792, %793 : tensor<1x257x768xf32>
    %795 = stablehlo.convert %794 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %796 = stablehlo.reshape %795 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %797 = stablehlo.convert %796 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %798 = stablehlo.dot_general %797, %arg106, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %800 = stablehlo.multiply %799, %61 : tensor<257x2304xf32>
    %801 = stablehlo.broadcast_in_dim %800, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %802 = stablehlo.broadcast_in_dim %arg107, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %803 = stablehlo.add %801, %802 : tensor<257x2304xf32>
    %804 = stablehlo.convert %803 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %805 = stablehlo.reshape %804 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %806 = stablehlo.reshape %805 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %807 = stablehlo.transpose %806, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %808 = stablehlo.slice %807 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %809 = stablehlo.reshape %808 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %810 = stablehlo.slice %807 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %811 = stablehlo.reshape %810 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %812 = stablehlo.slice %807 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %813 = stablehlo.reshape %812 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %814 = stablehlo.transpose %811, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %815 = stablehlo.reshape %809 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %816 = stablehlo.reshape %814 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %817 = stablehlo.broadcast_in_dim %816, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %818 = stablehlo.dot_general %815, %817, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %819 = stablehlo.reshape %818 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %820 = stablehlo.broadcast_in_dim %819, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %821 = stablehlo.multiply %820, %85 : tensor<1x12x257x257xbf16>
    %822 = stablehlo.convert %821 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %823 = stablehlo.reduce(%822 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %824 = stablehlo.reshape %823 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %825 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %826 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %827 = stablehlo.subtract %825, %826 : tensor<1x12x257x257xf32>
    %828 = stablehlo.exponential %827 : tensor<1x12x257x257xf32>
    %829 = stablehlo.reduce(%828 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %830 = stablehlo.reshape %829 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %831 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %832 = stablehlo.broadcast_in_dim %830, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %833 = stablehlo.divide %831, %832 : tensor<1x12x257x257xf32>
    %834 = stablehlo.convert %833 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %835 = stablehlo.reshape %834 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %836 = stablehlo.reshape %813 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %838 = stablehlo.dot_general %835, %837, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %839 = stablehlo.reshape %838 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %840 = stablehlo.transpose %839, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %841 = stablehlo.reshape %840 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %842 = stablehlo.reshape %841 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %843 = stablehlo.convert %842 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %844 = stablehlo.dot_general %843, %arg108, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %845 = stablehlo.broadcast_in_dim %844, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %846 = stablehlo.multiply %845, %111 : tensor<257x768xf32>
    %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %848 = stablehlo.broadcast_in_dim %arg109, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %849 = stablehlo.add %847, %848 : tensor<257x768xf32>
    %850 = stablehlo.convert %849 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %851 = stablehlo.reshape %850 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %852 = stablehlo.add %851, %758 : tensor<1x257x768xbf16>
    %853 = stablehlo.convert %852 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %854 = stablehlo.convert %853 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %855 = stablehlo.reduce(%854 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %856 = stablehlo.reshape %855 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %857 = stablehlo.broadcast_in_dim %856, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %858 = stablehlo.divide %857, %16 : tensor<1x257x1xf64>
    %859 = stablehlo.broadcast_in_dim %854, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %860 = stablehlo.broadcast_in_dim %858, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %861 = stablehlo.subtract %859, %860 : tensor<1x257x768xf64>
    %862 = stablehlo.multiply %861, %861 : tensor<1x257x768xf64>
    %863 = stablehlo.reduce(%862 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %864 = stablehlo.reshape %863 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %865 = stablehlo.broadcast_in_dim %864, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %866 = stablehlo.divide %865, %16 : tensor<1x257x1xf64>
    %867 = stablehlo.convert %866 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %868 = stablehlo.reduce(%853 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %869 = stablehlo.reshape %868 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %871 = stablehlo.divide %870, %32 : tensor<1x257x1xf32>
    %872 = stablehlo.broadcast_in_dim %867, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %873 = stablehlo.add %872, %37 : tensor<1x257x1xf32>
    %874 = stablehlo.rsqrt %873 : tensor<1x257x1xf32>
    %875 = stablehlo.broadcast_in_dim %853, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %876 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %877 = stablehlo.subtract %875, %876 : tensor<1x257x768xf32>
    %878 = stablehlo.broadcast_in_dim %877, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %879 = stablehlo.broadcast_in_dim %874, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %880 = stablehlo.multiply %878, %879 : tensor<1x257x768xf32>
    %881 = stablehlo.convert %arg22 : (tensor<768xbf16>) -> tensor<768xf32>
    %882 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %883 = stablehlo.broadcast_in_dim %881, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %884 = stablehlo.multiply %882, %883 : tensor<1x257x768xf32>
    %885 = stablehlo.convert %arg23 : (tensor<768xbf16>) -> tensor<768xf32>
    %886 = stablehlo.broadcast_in_dim %884, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %887 = stablehlo.broadcast_in_dim %885, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %888 = stablehlo.add %886, %887 : tensor<1x257x768xf32>
    %889 = stablehlo.convert %888 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %890 = stablehlo.reshape %889 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %891 = stablehlo.convert %890 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %892 = stablehlo.dot_general %891, %arg110, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %893 = stablehlo.broadcast_in_dim %892, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %894 = stablehlo.multiply %893, %160 : tensor<257x3072xf32>
    %895 = stablehlo.broadcast_in_dim %894, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %896 = stablehlo.broadcast_in_dim %arg111, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %897 = stablehlo.add %895, %896 : tensor<257x3072xf32>
    %898 = stablehlo.convert %897 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %899 = stablehlo.reshape %898 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %900 = stablehlo.multiply %899, %cst_4 : tensor<1x257x3072xbf16>
    %901 = stablehlo.multiply %899, %168 : tensor<1x257x3072xbf16>
    %902 = stablehlo.convert %901 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %903 = stablehlo.clamp %cst_5, %902, %cst_6 : tensor<1x257x3072xf32>
    %904 = stablehlo.multiply %903, %903 : tensor<1x257x3072xf32>
    %905 = stablehlo.multiply %cst_7, %904 : tensor<1x257x3072xf32>
    %906 = stablehlo.add %905, %cst_8 : tensor<1x257x3072xf32>
    %907 = stablehlo.multiply %906, %904 : tensor<1x257x3072xf32>
    %908 = stablehlo.add %907, %cst_9 : tensor<1x257x3072xf32>
    %909 = stablehlo.multiply %908, %904 : tensor<1x257x3072xf32>
    %910 = stablehlo.add %909, %cst_10 : tensor<1x257x3072xf32>
    %911 = stablehlo.multiply %910, %904 : tensor<1x257x3072xf32>
    %912 = stablehlo.add %911, %cst_11 : tensor<1x257x3072xf32>
    %913 = stablehlo.multiply %912, %904 : tensor<1x257x3072xf32>
    %914 = stablehlo.add %913, %cst_12 : tensor<1x257x3072xf32>
    %915 = stablehlo.multiply %914, %904 : tensor<1x257x3072xf32>
    %916 = stablehlo.add %915, %cst_13 : tensor<1x257x3072xf32>
    %917 = stablehlo.multiply %cst_14, %904 : tensor<1x257x3072xf32>
    %918 = stablehlo.add %917, %cst_15 : tensor<1x257x3072xf32>
    %919 = stablehlo.multiply %918, %904 : tensor<1x257x3072xf32>
    %920 = stablehlo.add %919, %cst_16 : tensor<1x257x3072xf32>
    %921 = stablehlo.multiply %920, %904 : tensor<1x257x3072xf32>
    %922 = stablehlo.add %921, %cst_17 : tensor<1x257x3072xf32>
    %923 = stablehlo.multiply %922, %904 : tensor<1x257x3072xf32>
    %924 = stablehlo.add %923, %cst_18 : tensor<1x257x3072xf32>
    %925 = stablehlo.multiply %903, %916 : tensor<1x257x3072xf32>
    %926 = stablehlo.divide %925, %924 : tensor<1x257x3072xf32>
    %927 = stablehlo.clamp %cst_19, %926, %cst_20 : tensor<1x257x3072xf32>
    %928 = stablehlo.convert %927 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %929 = stablehlo.add %928, %cst_2 : tensor<1x257x3072xbf16>
    %930 = stablehlo.multiply %929, %900 : tensor<1x257x3072xbf16>
    %931 = stablehlo.reshape %930 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %932 = stablehlo.convert %931 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %933 = stablehlo.dot_general %932, %arg112, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %934 = stablehlo.broadcast_in_dim %933, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %935 = stablehlo.multiply %934, %111 : tensor<257x768xf32>
    %936 = stablehlo.broadcast_in_dim %935, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %937 = stablehlo.broadcast_in_dim %arg113, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %938 = stablehlo.add %936, %937 : tensor<257x768xf32>
    %939 = stablehlo.convert %938 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %940 = stablehlo.reshape %939 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %941 = stablehlo.add %852, %940 : tensor<1x257x768xbf16>
    %942 = stablehlo.convert %941 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %943 = stablehlo.convert %942 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %944 = stablehlo.reduce(%943 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %945 = stablehlo.reshape %944 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %946 = stablehlo.broadcast_in_dim %945, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %947 = stablehlo.divide %946, %16 : tensor<1x257x1xf64>
    %948 = stablehlo.broadcast_in_dim %943, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %949 = stablehlo.broadcast_in_dim %947, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %950 = stablehlo.subtract %948, %949 : tensor<1x257x768xf64>
    %951 = stablehlo.multiply %950, %950 : tensor<1x257x768xf64>
    %952 = stablehlo.reduce(%951 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %953 = stablehlo.reshape %952 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %954 = stablehlo.broadcast_in_dim %953, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %955 = stablehlo.divide %954, %16 : tensor<1x257x1xf64>
    %956 = stablehlo.convert %955 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %957 = stablehlo.reduce(%942 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %958 = stablehlo.reshape %957 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %959 = stablehlo.broadcast_in_dim %958, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %960 = stablehlo.divide %959, %32 : tensor<1x257x1xf32>
    %961 = stablehlo.broadcast_in_dim %956, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %962 = stablehlo.add %961, %37 : tensor<1x257x1xf32>
    %963 = stablehlo.rsqrt %962 : tensor<1x257x1xf32>
    %964 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %965 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %966 = stablehlo.subtract %964, %965 : tensor<1x257x768xf32>
    %967 = stablehlo.broadcast_in_dim %966, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %968 = stablehlo.broadcast_in_dim %963, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %969 = stablehlo.multiply %967, %968 : tensor<1x257x768xf32>
    %970 = stablehlo.convert %arg24 : (tensor<768xbf16>) -> tensor<768xf32>
    %971 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %972 = stablehlo.broadcast_in_dim %970, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %973 = stablehlo.multiply %971, %972 : tensor<1x257x768xf32>
    %974 = stablehlo.convert %arg25 : (tensor<768xbf16>) -> tensor<768xf32>
    %975 = stablehlo.broadcast_in_dim %973, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %976 = stablehlo.broadcast_in_dim %974, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %977 = stablehlo.add %975, %976 : tensor<1x257x768xf32>
    %978 = stablehlo.convert %977 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %979 = stablehlo.reshape %978 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %980 = stablehlo.convert %979 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %981 = stablehlo.dot_general %980, %arg114, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %983 = stablehlo.multiply %982, %61 : tensor<257x2304xf32>
    %984 = stablehlo.broadcast_in_dim %983, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %985 = stablehlo.broadcast_in_dim %arg115, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %986 = stablehlo.add %984, %985 : tensor<257x2304xf32>
    %987 = stablehlo.convert %986 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %988 = stablehlo.reshape %987 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %989 = stablehlo.reshape %988 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %990 = stablehlo.transpose %989, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %991 = stablehlo.slice %990 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %992 = stablehlo.reshape %991 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %993 = stablehlo.slice %990 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %994 = stablehlo.reshape %993 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %995 = stablehlo.slice %990 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %996 = stablehlo.reshape %995 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %997 = stablehlo.transpose %994, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %998 = stablehlo.reshape %992 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %999 = stablehlo.reshape %997 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1000 = stablehlo.broadcast_in_dim %999, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1001 = stablehlo.dot_general %998, %1000, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %1002 = stablehlo.reshape %1001 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1003 = stablehlo.broadcast_in_dim %1002, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1004 = stablehlo.multiply %1003, %85 : tensor<1x12x257x257xbf16>
    %1005 = stablehlo.convert %1004 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %1006 = stablehlo.reduce(%1005 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1007 = stablehlo.reshape %1006 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1008 = stablehlo.broadcast_in_dim %1005, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1009 = stablehlo.broadcast_in_dim %1007, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1010 = stablehlo.subtract %1008, %1009 : tensor<1x12x257x257xf32>
    %1011 = stablehlo.exponential %1010 : tensor<1x12x257x257xf32>
    %1012 = stablehlo.reduce(%1011 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1013 = stablehlo.reshape %1012 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1014 = stablehlo.broadcast_in_dim %1011, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1015 = stablehlo.broadcast_in_dim %1013, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1016 = stablehlo.divide %1014, %1015 : tensor<1x12x257x257xf32>
    %1017 = stablehlo.convert %1016 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %1018 = stablehlo.reshape %1017 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %1019 = stablehlo.reshape %996 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1020 = stablehlo.broadcast_in_dim %1019, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1021 = stablehlo.dot_general %1018, %1020, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1022 = stablehlo.reshape %1021 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1023 = stablehlo.transpose %1022, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %1024 = stablehlo.reshape %1023 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %1025 = stablehlo.reshape %1024 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1026 = stablehlo.convert %1025 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1027 = stablehlo.dot_general %1026, %arg116, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %1028 = stablehlo.broadcast_in_dim %1027, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1029 = stablehlo.multiply %1028, %111 : tensor<257x768xf32>
    %1030 = stablehlo.broadcast_in_dim %1029, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1031 = stablehlo.broadcast_in_dim %arg117, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1032 = stablehlo.add %1030, %1031 : tensor<257x768xf32>
    %1033 = stablehlo.convert %1032 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1034 = stablehlo.reshape %1033 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1035 = stablehlo.add %1034, %941 : tensor<1x257x768xbf16>
    %1036 = stablehlo.convert %1035 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1037 = stablehlo.convert %1036 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1038 = stablehlo.reduce(%1037 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1039 = stablehlo.reshape %1038 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1041 = stablehlo.divide %1040, %16 : tensor<1x257x1xf64>
    %1042 = stablehlo.broadcast_in_dim %1037, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1043 = stablehlo.broadcast_in_dim %1041, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1044 = stablehlo.subtract %1042, %1043 : tensor<1x257x768xf64>
    %1045 = stablehlo.multiply %1044, %1044 : tensor<1x257x768xf64>
    %1046 = stablehlo.reduce(%1045 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1047 = stablehlo.reshape %1046 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1048 = stablehlo.broadcast_in_dim %1047, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1049 = stablehlo.divide %1048, %16 : tensor<1x257x1xf64>
    %1050 = stablehlo.convert %1049 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1051 = stablehlo.reduce(%1036 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1052 = stablehlo.reshape %1051 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1053 = stablehlo.broadcast_in_dim %1052, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1054 = stablehlo.divide %1053, %32 : tensor<1x257x1xf32>
    %1055 = stablehlo.broadcast_in_dim %1050, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1056 = stablehlo.add %1055, %37 : tensor<1x257x1xf32>
    %1057 = stablehlo.rsqrt %1056 : tensor<1x257x1xf32>
    %1058 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1059 = stablehlo.broadcast_in_dim %1054, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1060 = stablehlo.subtract %1058, %1059 : tensor<1x257x768xf32>
    %1061 = stablehlo.broadcast_in_dim %1060, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1062 = stablehlo.broadcast_in_dim %1057, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1063 = stablehlo.multiply %1061, %1062 : tensor<1x257x768xf32>
    %1064 = stablehlo.convert %arg26 : (tensor<768xbf16>) -> tensor<768xf32>
    %1065 = stablehlo.broadcast_in_dim %1063, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1066 = stablehlo.broadcast_in_dim %1064, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1067 = stablehlo.multiply %1065, %1066 : tensor<1x257x768xf32>
    %1068 = stablehlo.convert %arg27 : (tensor<768xbf16>) -> tensor<768xf32>
    %1069 = stablehlo.broadcast_in_dim %1067, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1070 = stablehlo.broadcast_in_dim %1068, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1071 = stablehlo.add %1069, %1070 : tensor<1x257x768xf32>
    %1072 = stablehlo.convert %1071 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1073 = stablehlo.reshape %1072 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1074 = stablehlo.convert %1073 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1075 = stablehlo.dot_general %1074, %arg118, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1077 = stablehlo.multiply %1076, %160 : tensor<257x3072xf32>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1079 = stablehlo.broadcast_in_dim %arg119, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %1080 = stablehlo.add %1078, %1079 : tensor<257x3072xf32>
    %1081 = stablehlo.convert %1080 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %1082 = stablehlo.reshape %1081 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %1083 = stablehlo.multiply %1082, %cst_4 : tensor<1x257x3072xbf16>
    %1084 = stablehlo.multiply %1082, %168 : tensor<1x257x3072xbf16>
    %1085 = stablehlo.convert %1084 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %1086 = stablehlo.clamp %cst_5, %1085, %cst_6 : tensor<1x257x3072xf32>
    %1087 = stablehlo.multiply %1086, %1086 : tensor<1x257x3072xf32>
    %1088 = stablehlo.multiply %cst_7, %1087 : tensor<1x257x3072xf32>
    %1089 = stablehlo.add %1088, %cst_8 : tensor<1x257x3072xf32>
    %1090 = stablehlo.multiply %1089, %1087 : tensor<1x257x3072xf32>
    %1091 = stablehlo.add %1090, %cst_9 : tensor<1x257x3072xf32>
    %1092 = stablehlo.multiply %1091, %1087 : tensor<1x257x3072xf32>
    %1093 = stablehlo.add %1092, %cst_10 : tensor<1x257x3072xf32>
    %1094 = stablehlo.multiply %1093, %1087 : tensor<1x257x3072xf32>
    %1095 = stablehlo.add %1094, %cst_11 : tensor<1x257x3072xf32>
    %1096 = stablehlo.multiply %1095, %1087 : tensor<1x257x3072xf32>
    %1097 = stablehlo.add %1096, %cst_12 : tensor<1x257x3072xf32>
    %1098 = stablehlo.multiply %1097, %1087 : tensor<1x257x3072xf32>
    %1099 = stablehlo.add %1098, %cst_13 : tensor<1x257x3072xf32>
    %1100 = stablehlo.multiply %cst_14, %1087 : tensor<1x257x3072xf32>
    %1101 = stablehlo.add %1100, %cst_15 : tensor<1x257x3072xf32>
    %1102 = stablehlo.multiply %1101, %1087 : tensor<1x257x3072xf32>
    %1103 = stablehlo.add %1102, %cst_16 : tensor<1x257x3072xf32>
    %1104 = stablehlo.multiply %1103, %1087 : tensor<1x257x3072xf32>
    %1105 = stablehlo.add %1104, %cst_17 : tensor<1x257x3072xf32>
    %1106 = stablehlo.multiply %1105, %1087 : tensor<1x257x3072xf32>
    %1107 = stablehlo.add %1106, %cst_18 : tensor<1x257x3072xf32>
    %1108 = stablehlo.multiply %1086, %1099 : tensor<1x257x3072xf32>
    %1109 = stablehlo.divide %1108, %1107 : tensor<1x257x3072xf32>
    %1110 = stablehlo.clamp %cst_19, %1109, %cst_20 : tensor<1x257x3072xf32>
    %1111 = stablehlo.convert %1110 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %1112 = stablehlo.add %1111, %cst_2 : tensor<1x257x3072xbf16>
    %1113 = stablehlo.multiply %1112, %1083 : tensor<1x257x3072xbf16>
    %1114 = stablehlo.reshape %1113 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %1115 = stablehlo.convert %1114 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %1116 = stablehlo.dot_general %1115, %arg120, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %1117 = stablehlo.broadcast_in_dim %1116, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1118 = stablehlo.multiply %1117, %111 : tensor<257x768xf32>
    %1119 = stablehlo.broadcast_in_dim %1118, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1120 = stablehlo.broadcast_in_dim %arg121, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1121 = stablehlo.add %1119, %1120 : tensor<257x768xf32>
    %1122 = stablehlo.convert %1121 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1123 = stablehlo.reshape %1122 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1124 = stablehlo.add %1035, %1123 : tensor<1x257x768xbf16>
    %1125 = stablehlo.convert %1124 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1126 = stablehlo.convert %1125 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1127 = stablehlo.reduce(%1126 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1128 = stablehlo.reshape %1127 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1129 = stablehlo.broadcast_in_dim %1128, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1130 = stablehlo.divide %1129, %16 : tensor<1x257x1xf64>
    %1131 = stablehlo.broadcast_in_dim %1126, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1132 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1133 = stablehlo.subtract %1131, %1132 : tensor<1x257x768xf64>
    %1134 = stablehlo.multiply %1133, %1133 : tensor<1x257x768xf64>
    %1135 = stablehlo.reduce(%1134 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1136 = stablehlo.reshape %1135 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1138 = stablehlo.divide %1137, %16 : tensor<1x257x1xf64>
    %1139 = stablehlo.convert %1138 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1140 = stablehlo.reduce(%1125 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1141 = stablehlo.reshape %1140 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1142 = stablehlo.broadcast_in_dim %1141, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1143 = stablehlo.divide %1142, %32 : tensor<1x257x1xf32>
    %1144 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1145 = stablehlo.add %1144, %37 : tensor<1x257x1xf32>
    %1146 = stablehlo.rsqrt %1145 : tensor<1x257x1xf32>
    %1147 = stablehlo.broadcast_in_dim %1125, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1148 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1149 = stablehlo.subtract %1147, %1148 : tensor<1x257x768xf32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1151 = stablehlo.broadcast_in_dim %1146, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1152 = stablehlo.multiply %1150, %1151 : tensor<1x257x768xf32>
    %1153 = stablehlo.convert %arg28 : (tensor<768xbf16>) -> tensor<768xf32>
    %1154 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1155 = stablehlo.broadcast_in_dim %1153, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1156 = stablehlo.multiply %1154, %1155 : tensor<1x257x768xf32>
    %1157 = stablehlo.convert %arg29 : (tensor<768xbf16>) -> tensor<768xf32>
    %1158 = stablehlo.broadcast_in_dim %1156, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1159 = stablehlo.broadcast_in_dim %1157, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1160 = stablehlo.add %1158, %1159 : tensor<1x257x768xf32>
    %1161 = stablehlo.convert %1160 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1162 = stablehlo.reshape %1161 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1163 = stablehlo.convert %1162 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1164 = stablehlo.dot_general %1163, %arg122, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %1165 = stablehlo.broadcast_in_dim %1164, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1166 = stablehlo.multiply %1165, %61 : tensor<257x2304xf32>
    %1167 = stablehlo.broadcast_in_dim %1166, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1168 = stablehlo.broadcast_in_dim %arg123, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %1169 = stablehlo.add %1167, %1168 : tensor<257x2304xf32>
    %1170 = stablehlo.convert %1169 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %1171 = stablehlo.reshape %1170 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %1172 = stablehlo.reshape %1171 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %1173 = stablehlo.transpose %1172, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %1174 = stablehlo.slice %1173 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1175 = stablehlo.reshape %1174 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1176 = stablehlo.slice %1173 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1177 = stablehlo.reshape %1176 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1178 = stablehlo.slice %1173 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1179 = stablehlo.reshape %1178 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1180 = stablehlo.transpose %1177, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %1181 = stablehlo.reshape %1175 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1182 = stablehlo.reshape %1180 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1183 = stablehlo.broadcast_in_dim %1182, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1184 = stablehlo.dot_general %1181, %1183, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %1185 = stablehlo.reshape %1184 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1186 = stablehlo.broadcast_in_dim %1185, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1187 = stablehlo.multiply %1186, %85 : tensor<1x12x257x257xbf16>
    %1188 = stablehlo.convert %1187 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %1189 = stablehlo.reduce(%1188 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1190 = stablehlo.reshape %1189 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1191 = stablehlo.broadcast_in_dim %1188, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1192 = stablehlo.broadcast_in_dim %1190, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1193 = stablehlo.subtract %1191, %1192 : tensor<1x12x257x257xf32>
    %1194 = stablehlo.exponential %1193 : tensor<1x12x257x257xf32>
    %1195 = stablehlo.reduce(%1194 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1196 = stablehlo.reshape %1195 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1197 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1198 = stablehlo.broadcast_in_dim %1196, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1199 = stablehlo.divide %1197, %1198 : tensor<1x12x257x257xf32>
    %1200 = stablehlo.convert %1199 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %1201 = stablehlo.reshape %1200 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %1202 = stablehlo.reshape %1179 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1203 = stablehlo.broadcast_in_dim %1202, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1204 = stablehlo.dot_general %1201, %1203, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1205 = stablehlo.reshape %1204 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1206 = stablehlo.transpose %1205, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %1207 = stablehlo.reshape %1206 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %1208 = stablehlo.reshape %1207 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1209 = stablehlo.convert %1208 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1210 = stablehlo.dot_general %1209, %arg124, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1212 = stablehlo.multiply %1211, %111 : tensor<257x768xf32>
    %1213 = stablehlo.broadcast_in_dim %1212, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1214 = stablehlo.broadcast_in_dim %arg125, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1215 = stablehlo.add %1213, %1214 : tensor<257x768xf32>
    %1216 = stablehlo.convert %1215 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1217 = stablehlo.reshape %1216 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1218 = stablehlo.add %1217, %1124 : tensor<1x257x768xbf16>
    %1219 = stablehlo.convert %1218 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1220 = stablehlo.convert %1219 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1221 = stablehlo.reduce(%1220 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1222 = stablehlo.reshape %1221 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1224 = stablehlo.divide %1223, %16 : tensor<1x257x1xf64>
    %1225 = stablehlo.broadcast_in_dim %1220, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1226 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1227 = stablehlo.subtract %1225, %1226 : tensor<1x257x768xf64>
    %1228 = stablehlo.multiply %1227, %1227 : tensor<1x257x768xf64>
    %1229 = stablehlo.reduce(%1228 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1230 = stablehlo.reshape %1229 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1231 = stablehlo.broadcast_in_dim %1230, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1232 = stablehlo.divide %1231, %16 : tensor<1x257x1xf64>
    %1233 = stablehlo.convert %1232 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1234 = stablehlo.reduce(%1219 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1235 = stablehlo.reshape %1234 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1237 = stablehlo.divide %1236, %32 : tensor<1x257x1xf32>
    %1238 = stablehlo.broadcast_in_dim %1233, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1239 = stablehlo.add %1238, %37 : tensor<1x257x1xf32>
    %1240 = stablehlo.rsqrt %1239 : tensor<1x257x1xf32>
    %1241 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1242 = stablehlo.broadcast_in_dim %1237, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1243 = stablehlo.subtract %1241, %1242 : tensor<1x257x768xf32>
    %1244 = stablehlo.broadcast_in_dim %1243, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1245 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1246 = stablehlo.multiply %1244, %1245 : tensor<1x257x768xf32>
    %1247 = stablehlo.convert %arg30 : (tensor<768xbf16>) -> tensor<768xf32>
    %1248 = stablehlo.broadcast_in_dim %1246, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1249 = stablehlo.broadcast_in_dim %1247, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1250 = stablehlo.multiply %1248, %1249 : tensor<1x257x768xf32>
    %1251 = stablehlo.convert %arg31 : (tensor<768xbf16>) -> tensor<768xf32>
    %1252 = stablehlo.broadcast_in_dim %1250, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1253 = stablehlo.broadcast_in_dim %1251, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1254 = stablehlo.add %1252, %1253 : tensor<1x257x768xf32>
    %1255 = stablehlo.convert %1254 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1256 = stablehlo.reshape %1255 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1257 = stablehlo.convert %1256 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1258 = stablehlo.dot_general %1257, %arg126, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %1259 = stablehlo.broadcast_in_dim %1258, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1260 = stablehlo.multiply %1259, %160 : tensor<257x3072xf32>
    %1261 = stablehlo.broadcast_in_dim %1260, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1262 = stablehlo.broadcast_in_dim %arg127, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %1263 = stablehlo.add %1261, %1262 : tensor<257x3072xf32>
    %1264 = stablehlo.convert %1263 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %1265 = stablehlo.reshape %1264 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %1266 = stablehlo.multiply %1265, %cst_4 : tensor<1x257x3072xbf16>
    %1267 = stablehlo.multiply %1265, %168 : tensor<1x257x3072xbf16>
    %1268 = stablehlo.convert %1267 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %1269 = stablehlo.clamp %cst_5, %1268, %cst_6 : tensor<1x257x3072xf32>
    %1270 = stablehlo.multiply %1269, %1269 : tensor<1x257x3072xf32>
    %1271 = stablehlo.multiply %cst_7, %1270 : tensor<1x257x3072xf32>
    %1272 = stablehlo.add %1271, %cst_8 : tensor<1x257x3072xf32>
    %1273 = stablehlo.multiply %1272, %1270 : tensor<1x257x3072xf32>
    %1274 = stablehlo.add %1273, %cst_9 : tensor<1x257x3072xf32>
    %1275 = stablehlo.multiply %1274, %1270 : tensor<1x257x3072xf32>
    %1276 = stablehlo.add %1275, %cst_10 : tensor<1x257x3072xf32>
    %1277 = stablehlo.multiply %1276, %1270 : tensor<1x257x3072xf32>
    %1278 = stablehlo.add %1277, %cst_11 : tensor<1x257x3072xf32>
    %1279 = stablehlo.multiply %1278, %1270 : tensor<1x257x3072xf32>
    %1280 = stablehlo.add %1279, %cst_12 : tensor<1x257x3072xf32>
    %1281 = stablehlo.multiply %1280, %1270 : tensor<1x257x3072xf32>
    %1282 = stablehlo.add %1281, %cst_13 : tensor<1x257x3072xf32>
    %1283 = stablehlo.multiply %cst_14, %1270 : tensor<1x257x3072xf32>
    %1284 = stablehlo.add %1283, %cst_15 : tensor<1x257x3072xf32>
    %1285 = stablehlo.multiply %1284, %1270 : tensor<1x257x3072xf32>
    %1286 = stablehlo.add %1285, %cst_16 : tensor<1x257x3072xf32>
    %1287 = stablehlo.multiply %1286, %1270 : tensor<1x257x3072xf32>
    %1288 = stablehlo.add %1287, %cst_17 : tensor<1x257x3072xf32>
    %1289 = stablehlo.multiply %1288, %1270 : tensor<1x257x3072xf32>
    %1290 = stablehlo.add %1289, %cst_18 : tensor<1x257x3072xf32>
    %1291 = stablehlo.multiply %1269, %1282 : tensor<1x257x3072xf32>
    %1292 = stablehlo.divide %1291, %1290 : tensor<1x257x3072xf32>
    %1293 = stablehlo.clamp %cst_19, %1292, %cst_20 : tensor<1x257x3072xf32>
    %1294 = stablehlo.convert %1293 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %1295 = stablehlo.add %1294, %cst_2 : tensor<1x257x3072xbf16>
    %1296 = stablehlo.multiply %1295, %1266 : tensor<1x257x3072xbf16>
    %1297 = stablehlo.reshape %1296 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %1298 = stablehlo.convert %1297 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %1299 = stablehlo.dot_general %1298, %arg128, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1301 = stablehlo.multiply %1300, %111 : tensor<257x768xf32>
    %1302 = stablehlo.broadcast_in_dim %1301, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1303 = stablehlo.broadcast_in_dim %arg129, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1304 = stablehlo.add %1302, %1303 : tensor<257x768xf32>
    %1305 = stablehlo.convert %1304 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1306 = stablehlo.reshape %1305 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1307 = stablehlo.add %1218, %1306 : tensor<1x257x768xbf16>
    %1308 = stablehlo.convert %1307 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1309 = stablehlo.convert %1308 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1310 = stablehlo.reduce(%1309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1311 = stablehlo.reshape %1310 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1312 = stablehlo.broadcast_in_dim %1311, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1313 = stablehlo.divide %1312, %16 : tensor<1x257x1xf64>
    %1314 = stablehlo.broadcast_in_dim %1309, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1315 = stablehlo.broadcast_in_dim %1313, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1316 = stablehlo.subtract %1314, %1315 : tensor<1x257x768xf64>
    %1317 = stablehlo.multiply %1316, %1316 : tensor<1x257x768xf64>
    %1318 = stablehlo.reduce(%1317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1319 = stablehlo.reshape %1318 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1320 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1321 = stablehlo.divide %1320, %16 : tensor<1x257x1xf64>
    %1322 = stablehlo.convert %1321 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1323 = stablehlo.reduce(%1308 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1324 = stablehlo.reshape %1323 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1325 = stablehlo.broadcast_in_dim %1324, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1326 = stablehlo.divide %1325, %32 : tensor<1x257x1xf32>
    %1327 = stablehlo.broadcast_in_dim %1322, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1328 = stablehlo.add %1327, %37 : tensor<1x257x1xf32>
    %1329 = stablehlo.rsqrt %1328 : tensor<1x257x1xf32>
    %1330 = stablehlo.broadcast_in_dim %1308, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1331 = stablehlo.broadcast_in_dim %1326, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1332 = stablehlo.subtract %1330, %1331 : tensor<1x257x768xf32>
    %1333 = stablehlo.broadcast_in_dim %1332, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1334 = stablehlo.broadcast_in_dim %1329, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1335 = stablehlo.multiply %1333, %1334 : tensor<1x257x768xf32>
    %1336 = stablehlo.convert %arg32 : (tensor<768xbf16>) -> tensor<768xf32>
    %1337 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1338 = stablehlo.broadcast_in_dim %1336, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1339 = stablehlo.multiply %1337, %1338 : tensor<1x257x768xf32>
    %1340 = stablehlo.convert %arg33 : (tensor<768xbf16>) -> tensor<768xf32>
    %1341 = stablehlo.broadcast_in_dim %1339, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1342 = stablehlo.broadcast_in_dim %1340, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1343 = stablehlo.add %1341, %1342 : tensor<1x257x768xf32>
    %1344 = stablehlo.convert %1343 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1345 = stablehlo.reshape %1344 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1346 = stablehlo.convert %1345 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1347 = stablehlo.dot_general %1346, %arg130, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1349 = stablehlo.multiply %1348, %61 : tensor<257x2304xf32>
    %1350 = stablehlo.broadcast_in_dim %1349, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1351 = stablehlo.broadcast_in_dim %arg131, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %1352 = stablehlo.add %1350, %1351 : tensor<257x2304xf32>
    %1353 = stablehlo.convert %1352 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %1354 = stablehlo.reshape %1353 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %1355 = stablehlo.reshape %1354 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %1356 = stablehlo.transpose %1355, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %1357 = stablehlo.slice %1356 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1358 = stablehlo.reshape %1357 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1359 = stablehlo.slice %1356 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1360 = stablehlo.reshape %1359 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1361 = stablehlo.slice %1356 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1362 = stablehlo.reshape %1361 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1363 = stablehlo.transpose %1360, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %1364 = stablehlo.reshape %1358 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1365 = stablehlo.reshape %1363 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1367 = stablehlo.dot_general %1364, %1366, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %1368 = stablehlo.reshape %1367 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1369 = stablehlo.broadcast_in_dim %1368, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1370 = stablehlo.multiply %1369, %85 : tensor<1x12x257x257xbf16>
    %1371 = stablehlo.convert %1370 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %1372 = stablehlo.reduce(%1371 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1373 = stablehlo.reshape %1372 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1374 = stablehlo.broadcast_in_dim %1371, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1375 = stablehlo.broadcast_in_dim %1373, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1376 = stablehlo.subtract %1374, %1375 : tensor<1x12x257x257xf32>
    %1377 = stablehlo.exponential %1376 : tensor<1x12x257x257xf32>
    %1378 = stablehlo.reduce(%1377 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1379 = stablehlo.reshape %1378 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1380 = stablehlo.broadcast_in_dim %1377, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1381 = stablehlo.broadcast_in_dim %1379, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1382 = stablehlo.divide %1380, %1381 : tensor<1x12x257x257xf32>
    %1383 = stablehlo.convert %1382 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %1384 = stablehlo.reshape %1383 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %1385 = stablehlo.reshape %1362 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1386 = stablehlo.broadcast_in_dim %1385, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1387 = stablehlo.dot_general %1384, %1386, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1388 = stablehlo.reshape %1387 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1389 = stablehlo.transpose %1388, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %1390 = stablehlo.reshape %1389 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %1391 = stablehlo.reshape %1390 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1392 = stablehlo.convert %1391 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1393 = stablehlo.dot_general %1392, %arg132, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %1394 = stablehlo.broadcast_in_dim %1393, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1395 = stablehlo.multiply %1394, %111 : tensor<257x768xf32>
    %1396 = stablehlo.broadcast_in_dim %1395, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1397 = stablehlo.broadcast_in_dim %arg133, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1398 = stablehlo.add %1396, %1397 : tensor<257x768xf32>
    %1399 = stablehlo.convert %1398 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1400 = stablehlo.reshape %1399 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1401 = stablehlo.add %1400, %1307 : tensor<1x257x768xbf16>
    %1402 = stablehlo.convert %1401 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1403 = stablehlo.convert %1402 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1404 = stablehlo.reduce(%1403 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1405 = stablehlo.reshape %1404 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1406 = stablehlo.broadcast_in_dim %1405, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1407 = stablehlo.divide %1406, %16 : tensor<1x257x1xf64>
    %1408 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1409 = stablehlo.broadcast_in_dim %1407, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1410 = stablehlo.subtract %1408, %1409 : tensor<1x257x768xf64>
    %1411 = stablehlo.multiply %1410, %1410 : tensor<1x257x768xf64>
    %1412 = stablehlo.reduce(%1411 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1413 = stablehlo.reshape %1412 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1414 = stablehlo.broadcast_in_dim %1413, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1415 = stablehlo.divide %1414, %16 : tensor<1x257x1xf64>
    %1416 = stablehlo.convert %1415 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1417 = stablehlo.reduce(%1402 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1418 = stablehlo.reshape %1417 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1419 = stablehlo.broadcast_in_dim %1418, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1420 = stablehlo.divide %1419, %32 : tensor<1x257x1xf32>
    %1421 = stablehlo.broadcast_in_dim %1416, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1422 = stablehlo.add %1421, %37 : tensor<1x257x1xf32>
    %1423 = stablehlo.rsqrt %1422 : tensor<1x257x1xf32>
    %1424 = stablehlo.broadcast_in_dim %1402, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1425 = stablehlo.broadcast_in_dim %1420, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1426 = stablehlo.subtract %1424, %1425 : tensor<1x257x768xf32>
    %1427 = stablehlo.broadcast_in_dim %1426, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1428 = stablehlo.broadcast_in_dim %1423, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1429 = stablehlo.multiply %1427, %1428 : tensor<1x257x768xf32>
    %1430 = stablehlo.convert %arg34 : (tensor<768xbf16>) -> tensor<768xf32>
    %1431 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1432 = stablehlo.broadcast_in_dim %1430, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1433 = stablehlo.multiply %1431, %1432 : tensor<1x257x768xf32>
    %1434 = stablehlo.convert %arg35 : (tensor<768xbf16>) -> tensor<768xf32>
    %1435 = stablehlo.broadcast_in_dim %1433, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1436 = stablehlo.broadcast_in_dim %1434, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1437 = stablehlo.add %1435, %1436 : tensor<1x257x768xf32>
    %1438 = stablehlo.convert %1437 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1439 = stablehlo.reshape %1438 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1440 = stablehlo.convert %1439 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1441 = stablehlo.dot_general %1440, %arg134, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %1442 = stablehlo.broadcast_in_dim %1441, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1443 = stablehlo.multiply %1442, %160 : tensor<257x3072xf32>
    %1444 = stablehlo.broadcast_in_dim %1443, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1445 = stablehlo.broadcast_in_dim %arg135, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %1446 = stablehlo.add %1444, %1445 : tensor<257x3072xf32>
    %1447 = stablehlo.convert %1446 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %1448 = stablehlo.reshape %1447 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %1449 = stablehlo.multiply %1448, %cst_4 : tensor<1x257x3072xbf16>
    %1450 = stablehlo.multiply %1448, %168 : tensor<1x257x3072xbf16>
    %1451 = stablehlo.convert %1450 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %1452 = stablehlo.clamp %cst_5, %1451, %cst_6 : tensor<1x257x3072xf32>
    %1453 = stablehlo.multiply %1452, %1452 : tensor<1x257x3072xf32>
    %1454 = stablehlo.multiply %cst_7, %1453 : tensor<1x257x3072xf32>
    %1455 = stablehlo.add %1454, %cst_8 : tensor<1x257x3072xf32>
    %1456 = stablehlo.multiply %1455, %1453 : tensor<1x257x3072xf32>
    %1457 = stablehlo.add %1456, %cst_9 : tensor<1x257x3072xf32>
    %1458 = stablehlo.multiply %1457, %1453 : tensor<1x257x3072xf32>
    %1459 = stablehlo.add %1458, %cst_10 : tensor<1x257x3072xf32>
    %1460 = stablehlo.multiply %1459, %1453 : tensor<1x257x3072xf32>
    %1461 = stablehlo.add %1460, %cst_11 : tensor<1x257x3072xf32>
    %1462 = stablehlo.multiply %1461, %1453 : tensor<1x257x3072xf32>
    %1463 = stablehlo.add %1462, %cst_12 : tensor<1x257x3072xf32>
    %1464 = stablehlo.multiply %1463, %1453 : tensor<1x257x3072xf32>
    %1465 = stablehlo.add %1464, %cst_13 : tensor<1x257x3072xf32>
    %1466 = stablehlo.multiply %cst_14, %1453 : tensor<1x257x3072xf32>
    %1467 = stablehlo.add %1466, %cst_15 : tensor<1x257x3072xf32>
    %1468 = stablehlo.multiply %1467, %1453 : tensor<1x257x3072xf32>
    %1469 = stablehlo.add %1468, %cst_16 : tensor<1x257x3072xf32>
    %1470 = stablehlo.multiply %1469, %1453 : tensor<1x257x3072xf32>
    %1471 = stablehlo.add %1470, %cst_17 : tensor<1x257x3072xf32>
    %1472 = stablehlo.multiply %1471, %1453 : tensor<1x257x3072xf32>
    %1473 = stablehlo.add %1472, %cst_18 : tensor<1x257x3072xf32>
    %1474 = stablehlo.multiply %1452, %1465 : tensor<1x257x3072xf32>
    %1475 = stablehlo.divide %1474, %1473 : tensor<1x257x3072xf32>
    %1476 = stablehlo.clamp %cst_19, %1475, %cst_20 : tensor<1x257x3072xf32>
    %1477 = stablehlo.convert %1476 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %1478 = stablehlo.add %1477, %cst_2 : tensor<1x257x3072xbf16>
    %1479 = stablehlo.multiply %1478, %1449 : tensor<1x257x3072xbf16>
    %1480 = stablehlo.reshape %1479 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %1481 = stablehlo.convert %1480 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %1482 = stablehlo.dot_general %1481, %arg136, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %1483 = stablehlo.broadcast_in_dim %1482, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1484 = stablehlo.multiply %1483, %111 : tensor<257x768xf32>
    %1485 = stablehlo.broadcast_in_dim %1484, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1486 = stablehlo.broadcast_in_dim %arg137, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1487 = stablehlo.add %1485, %1486 : tensor<257x768xf32>
    %1488 = stablehlo.convert %1487 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1489 = stablehlo.reshape %1488 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1490 = stablehlo.add %1401, %1489 : tensor<1x257x768xbf16>
    %1491 = stablehlo.convert %1490 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1492 = stablehlo.convert %1491 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1493 = stablehlo.reduce(%1492 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1494 = stablehlo.reshape %1493 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1495 = stablehlo.broadcast_in_dim %1494, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1496 = stablehlo.divide %1495, %16 : tensor<1x257x1xf64>
    %1497 = stablehlo.broadcast_in_dim %1492, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1498 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1499 = stablehlo.subtract %1497, %1498 : tensor<1x257x768xf64>
    %1500 = stablehlo.multiply %1499, %1499 : tensor<1x257x768xf64>
    %1501 = stablehlo.reduce(%1500 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1502 = stablehlo.reshape %1501 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1503 = stablehlo.broadcast_in_dim %1502, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1504 = stablehlo.divide %1503, %16 : tensor<1x257x1xf64>
    %1505 = stablehlo.convert %1504 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1506 = stablehlo.reduce(%1491 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1507 = stablehlo.reshape %1506 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1508 = stablehlo.broadcast_in_dim %1507, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1509 = stablehlo.divide %1508, %32 : tensor<1x257x1xf32>
    %1510 = stablehlo.broadcast_in_dim %1505, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1511 = stablehlo.add %1510, %37 : tensor<1x257x1xf32>
    %1512 = stablehlo.rsqrt %1511 : tensor<1x257x1xf32>
    %1513 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1514 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1515 = stablehlo.subtract %1513, %1514 : tensor<1x257x768xf32>
    %1516 = stablehlo.broadcast_in_dim %1515, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1517 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1518 = stablehlo.multiply %1516, %1517 : tensor<1x257x768xf32>
    %1519 = stablehlo.convert %arg36 : (tensor<768xbf16>) -> tensor<768xf32>
    %1520 = stablehlo.broadcast_in_dim %1518, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1521 = stablehlo.broadcast_in_dim %1519, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1522 = stablehlo.multiply %1520, %1521 : tensor<1x257x768xf32>
    %1523 = stablehlo.convert %arg37 : (tensor<768xbf16>) -> tensor<768xf32>
    %1524 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1525 = stablehlo.broadcast_in_dim %1523, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1526 = stablehlo.add %1524, %1525 : tensor<1x257x768xf32>
    %1527 = stablehlo.convert %1526 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1528 = stablehlo.reshape %1527 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1529 = stablehlo.convert %1528 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1530 = stablehlo.dot_general %1529, %arg138, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %1531 = stablehlo.broadcast_in_dim %1530, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1532 = stablehlo.multiply %1531, %61 : tensor<257x2304xf32>
    %1533 = stablehlo.broadcast_in_dim %1532, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1534 = stablehlo.broadcast_in_dim %arg139, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %1535 = stablehlo.add %1533, %1534 : tensor<257x2304xf32>
    %1536 = stablehlo.convert %1535 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %1537 = stablehlo.reshape %1536 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %1538 = stablehlo.reshape %1537 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %1539 = stablehlo.transpose %1538, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %1540 = stablehlo.slice %1539 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1541 = stablehlo.reshape %1540 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1542 = stablehlo.slice %1539 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1543 = stablehlo.reshape %1542 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1544 = stablehlo.slice %1539 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1545 = stablehlo.reshape %1544 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1546 = stablehlo.transpose %1543, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %1547 = stablehlo.reshape %1541 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1548 = stablehlo.reshape %1546 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1549 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1550 = stablehlo.dot_general %1547, %1549, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %1551 = stablehlo.reshape %1550 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1552 = stablehlo.broadcast_in_dim %1551, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1553 = stablehlo.multiply %1552, %85 : tensor<1x12x257x257xbf16>
    %1554 = stablehlo.convert %1553 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %1555 = stablehlo.reduce(%1554 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1556 = stablehlo.reshape %1555 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1557 = stablehlo.broadcast_in_dim %1554, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1558 = stablehlo.broadcast_in_dim %1556, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1559 = stablehlo.subtract %1557, %1558 : tensor<1x12x257x257xf32>
    %1560 = stablehlo.exponential %1559 : tensor<1x12x257x257xf32>
    %1561 = stablehlo.reduce(%1560 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1562 = stablehlo.reshape %1561 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1563 = stablehlo.broadcast_in_dim %1560, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1564 = stablehlo.broadcast_in_dim %1562, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1565 = stablehlo.divide %1563, %1564 : tensor<1x12x257x257xf32>
    %1566 = stablehlo.convert %1565 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %1567 = stablehlo.reshape %1566 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %1568 = stablehlo.reshape %1545 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1569 = stablehlo.broadcast_in_dim %1568, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1570 = stablehlo.dot_general %1567, %1569, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1571 = stablehlo.reshape %1570 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1572 = stablehlo.transpose %1571, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %1573 = stablehlo.reshape %1572 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %1574 = stablehlo.reshape %1573 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1575 = stablehlo.convert %1574 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1576 = stablehlo.dot_general %1575, %arg140, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %1577 = stablehlo.broadcast_in_dim %1576, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1578 = stablehlo.multiply %1577, %111 : tensor<257x768xf32>
    %1579 = stablehlo.broadcast_in_dim %1578, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1580 = stablehlo.broadcast_in_dim %arg141, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1581 = stablehlo.add %1579, %1580 : tensor<257x768xf32>
    %1582 = stablehlo.convert %1581 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1583 = stablehlo.reshape %1582 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1584 = stablehlo.add %1583, %1490 : tensor<1x257x768xbf16>
    %1585 = stablehlo.convert %1584 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1586 = stablehlo.convert %1585 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1587 = stablehlo.reduce(%1586 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1588 = stablehlo.reshape %1587 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1589 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1590 = stablehlo.divide %1589, %16 : tensor<1x257x1xf64>
    %1591 = stablehlo.broadcast_in_dim %1586, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1592 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1593 = stablehlo.subtract %1591, %1592 : tensor<1x257x768xf64>
    %1594 = stablehlo.multiply %1593, %1593 : tensor<1x257x768xf64>
    %1595 = stablehlo.reduce(%1594 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1596 = stablehlo.reshape %1595 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1597 = stablehlo.broadcast_in_dim %1596, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1598 = stablehlo.divide %1597, %16 : tensor<1x257x1xf64>
    %1599 = stablehlo.convert %1598 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1600 = stablehlo.reduce(%1585 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1601 = stablehlo.reshape %1600 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1602 = stablehlo.broadcast_in_dim %1601, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1603 = stablehlo.divide %1602, %32 : tensor<1x257x1xf32>
    %1604 = stablehlo.broadcast_in_dim %1599, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1605 = stablehlo.add %1604, %37 : tensor<1x257x1xf32>
    %1606 = stablehlo.rsqrt %1605 : tensor<1x257x1xf32>
    %1607 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1608 = stablehlo.broadcast_in_dim %1603, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1609 = stablehlo.subtract %1607, %1608 : tensor<1x257x768xf32>
    %1610 = stablehlo.broadcast_in_dim %1609, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1611 = stablehlo.broadcast_in_dim %1606, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1612 = stablehlo.multiply %1610, %1611 : tensor<1x257x768xf32>
    %1613 = stablehlo.convert %arg38 : (tensor<768xbf16>) -> tensor<768xf32>
    %1614 = stablehlo.broadcast_in_dim %1612, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1615 = stablehlo.broadcast_in_dim %1613, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1616 = stablehlo.multiply %1614, %1615 : tensor<1x257x768xf32>
    %1617 = stablehlo.convert %arg39 : (tensor<768xbf16>) -> tensor<768xf32>
    %1618 = stablehlo.broadcast_in_dim %1616, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1619 = stablehlo.broadcast_in_dim %1617, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1620 = stablehlo.add %1618, %1619 : tensor<1x257x768xf32>
    %1621 = stablehlo.convert %1620 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1622 = stablehlo.reshape %1621 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1623 = stablehlo.convert %1622 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1624 = stablehlo.dot_general %1623, %arg142, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1626 = stablehlo.multiply %1625, %160 : tensor<257x3072xf32>
    %1627 = stablehlo.broadcast_in_dim %1626, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1628 = stablehlo.broadcast_in_dim %arg143, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %1629 = stablehlo.add %1627, %1628 : tensor<257x3072xf32>
    %1630 = stablehlo.convert %1629 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %1631 = stablehlo.reshape %1630 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %1632 = stablehlo.multiply %1631, %cst_4 : tensor<1x257x3072xbf16>
    %1633 = stablehlo.multiply %1631, %168 : tensor<1x257x3072xbf16>
    %1634 = stablehlo.convert %1633 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %1635 = stablehlo.clamp %cst_5, %1634, %cst_6 : tensor<1x257x3072xf32>
    %1636 = stablehlo.multiply %1635, %1635 : tensor<1x257x3072xf32>
    %1637 = stablehlo.multiply %cst_7, %1636 : tensor<1x257x3072xf32>
    %1638 = stablehlo.add %1637, %cst_8 : tensor<1x257x3072xf32>
    %1639 = stablehlo.multiply %1638, %1636 : tensor<1x257x3072xf32>
    %1640 = stablehlo.add %1639, %cst_9 : tensor<1x257x3072xf32>
    %1641 = stablehlo.multiply %1640, %1636 : tensor<1x257x3072xf32>
    %1642 = stablehlo.add %1641, %cst_10 : tensor<1x257x3072xf32>
    %1643 = stablehlo.multiply %1642, %1636 : tensor<1x257x3072xf32>
    %1644 = stablehlo.add %1643, %cst_11 : tensor<1x257x3072xf32>
    %1645 = stablehlo.multiply %1644, %1636 : tensor<1x257x3072xf32>
    %1646 = stablehlo.add %1645, %cst_12 : tensor<1x257x3072xf32>
    %1647 = stablehlo.multiply %1646, %1636 : tensor<1x257x3072xf32>
    %1648 = stablehlo.add %1647, %cst_13 : tensor<1x257x3072xf32>
    %1649 = stablehlo.multiply %cst_14, %1636 : tensor<1x257x3072xf32>
    %1650 = stablehlo.add %1649, %cst_15 : tensor<1x257x3072xf32>
    %1651 = stablehlo.multiply %1650, %1636 : tensor<1x257x3072xf32>
    %1652 = stablehlo.add %1651, %cst_16 : tensor<1x257x3072xf32>
    %1653 = stablehlo.multiply %1652, %1636 : tensor<1x257x3072xf32>
    %1654 = stablehlo.add %1653, %cst_17 : tensor<1x257x3072xf32>
    %1655 = stablehlo.multiply %1654, %1636 : tensor<1x257x3072xf32>
    %1656 = stablehlo.add %1655, %cst_18 : tensor<1x257x3072xf32>
    %1657 = stablehlo.multiply %1635, %1648 : tensor<1x257x3072xf32>
    %1658 = stablehlo.divide %1657, %1656 : tensor<1x257x3072xf32>
    %1659 = stablehlo.clamp %cst_19, %1658, %cst_20 : tensor<1x257x3072xf32>
    %1660 = stablehlo.convert %1659 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %1661 = stablehlo.add %1660, %cst_2 : tensor<1x257x3072xbf16>
    %1662 = stablehlo.multiply %1661, %1632 : tensor<1x257x3072xbf16>
    %1663 = stablehlo.reshape %1662 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %1664 = stablehlo.convert %1663 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %1665 = stablehlo.dot_general %1664, %arg144, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %1666 = stablehlo.broadcast_in_dim %1665, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1667 = stablehlo.multiply %1666, %111 : tensor<257x768xf32>
    %1668 = stablehlo.broadcast_in_dim %1667, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1669 = stablehlo.broadcast_in_dim %arg145, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1670 = stablehlo.add %1668, %1669 : tensor<257x768xf32>
    %1671 = stablehlo.convert %1670 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1672 = stablehlo.reshape %1671 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1673 = stablehlo.add %1584, %1672 : tensor<1x257x768xbf16>
    %1674 = stablehlo.convert %1673 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1675 = stablehlo.convert %1674 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1676 = stablehlo.reduce(%1675 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1677 = stablehlo.reshape %1676 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1678 = stablehlo.broadcast_in_dim %1677, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1679 = stablehlo.divide %1678, %16 : tensor<1x257x1xf64>
    %1680 = stablehlo.broadcast_in_dim %1675, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1681 = stablehlo.broadcast_in_dim %1679, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1682 = stablehlo.subtract %1680, %1681 : tensor<1x257x768xf64>
    %1683 = stablehlo.multiply %1682, %1682 : tensor<1x257x768xf64>
    %1684 = stablehlo.reduce(%1683 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1685 = stablehlo.reshape %1684 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1686 = stablehlo.broadcast_in_dim %1685, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1687 = stablehlo.divide %1686, %16 : tensor<1x257x1xf64>
    %1688 = stablehlo.convert %1687 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1689 = stablehlo.reduce(%1674 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1690 = stablehlo.reshape %1689 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1691 = stablehlo.broadcast_in_dim %1690, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1692 = stablehlo.divide %1691, %32 : tensor<1x257x1xf32>
    %1693 = stablehlo.broadcast_in_dim %1688, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1694 = stablehlo.add %1693, %37 : tensor<1x257x1xf32>
    %1695 = stablehlo.rsqrt %1694 : tensor<1x257x1xf32>
    %1696 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1697 = stablehlo.broadcast_in_dim %1692, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1698 = stablehlo.subtract %1696, %1697 : tensor<1x257x768xf32>
    %1699 = stablehlo.broadcast_in_dim %1698, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1700 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1701 = stablehlo.multiply %1699, %1700 : tensor<1x257x768xf32>
    %1702 = stablehlo.convert %arg40 : (tensor<768xbf16>) -> tensor<768xf32>
    %1703 = stablehlo.broadcast_in_dim %1701, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1704 = stablehlo.broadcast_in_dim %1702, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1705 = stablehlo.multiply %1703, %1704 : tensor<1x257x768xf32>
    %1706 = stablehlo.convert %arg41 : (tensor<768xbf16>) -> tensor<768xf32>
    %1707 = stablehlo.broadcast_in_dim %1705, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1708 = stablehlo.broadcast_in_dim %1706, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1709 = stablehlo.add %1707, %1708 : tensor<1x257x768xf32>
    %1710 = stablehlo.convert %1709 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1711 = stablehlo.reshape %1710 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1712 = stablehlo.convert %1711 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1713 = stablehlo.dot_general %1712, %arg146, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %1714 = stablehlo.broadcast_in_dim %1713, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1715 = stablehlo.multiply %1714, %61 : tensor<257x2304xf32>
    %1716 = stablehlo.broadcast_in_dim %1715, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1717 = stablehlo.broadcast_in_dim %arg147, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %1718 = stablehlo.add %1716, %1717 : tensor<257x2304xf32>
    %1719 = stablehlo.convert %1718 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %1720 = stablehlo.reshape %1719 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %1721 = stablehlo.reshape %1720 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %1722 = stablehlo.transpose %1721, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %1723 = stablehlo.slice %1722 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1724 = stablehlo.reshape %1723 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1725 = stablehlo.slice %1722 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1726 = stablehlo.reshape %1725 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1727 = stablehlo.slice %1722 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1728 = stablehlo.reshape %1727 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1729 = stablehlo.transpose %1726, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %1730 = stablehlo.reshape %1724 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1731 = stablehlo.reshape %1729 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1733 = stablehlo.dot_general %1730, %1732, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %1734 = stablehlo.reshape %1733 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1735 = stablehlo.broadcast_in_dim %1734, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1736 = stablehlo.multiply %1735, %85 : tensor<1x12x257x257xbf16>
    %1737 = stablehlo.convert %1736 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %1738 = stablehlo.reduce(%1737 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1739 = stablehlo.reshape %1738 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1740 = stablehlo.broadcast_in_dim %1737, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1741 = stablehlo.broadcast_in_dim %1739, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1742 = stablehlo.subtract %1740, %1741 : tensor<1x12x257x257xf32>
    %1743 = stablehlo.exponential %1742 : tensor<1x12x257x257xf32>
    %1744 = stablehlo.reduce(%1743 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1745 = stablehlo.reshape %1744 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1746 = stablehlo.broadcast_in_dim %1743, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1747 = stablehlo.broadcast_in_dim %1745, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1748 = stablehlo.divide %1746, %1747 : tensor<1x12x257x257xf32>
    %1749 = stablehlo.convert %1748 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %1750 = stablehlo.reshape %1749 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %1751 = stablehlo.reshape %1728 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1752 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1753 = stablehlo.dot_general %1750, %1752, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1754 = stablehlo.reshape %1753 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1755 = stablehlo.transpose %1754, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %1756 = stablehlo.reshape %1755 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %1757 = stablehlo.reshape %1756 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1758 = stablehlo.convert %1757 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1759 = stablehlo.dot_general %1758, %arg148, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %1760 = stablehlo.broadcast_in_dim %1759, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1761 = stablehlo.multiply %1760, %111 : tensor<257x768xf32>
    %1762 = stablehlo.broadcast_in_dim %1761, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1763 = stablehlo.broadcast_in_dim %arg149, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1764 = stablehlo.add %1762, %1763 : tensor<257x768xf32>
    %1765 = stablehlo.convert %1764 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1766 = stablehlo.reshape %1765 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1767 = stablehlo.add %1766, %1673 : tensor<1x257x768xbf16>
    %1768 = stablehlo.convert %1767 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1769 = stablehlo.convert %1768 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1770 = stablehlo.reduce(%1769 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1771 = stablehlo.reshape %1770 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1772 = stablehlo.broadcast_in_dim %1771, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1773 = stablehlo.divide %1772, %16 : tensor<1x257x1xf64>
    %1774 = stablehlo.broadcast_in_dim %1769, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1775 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1776 = stablehlo.subtract %1774, %1775 : tensor<1x257x768xf64>
    %1777 = stablehlo.multiply %1776, %1776 : tensor<1x257x768xf64>
    %1778 = stablehlo.reduce(%1777 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1779 = stablehlo.reshape %1778 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1780 = stablehlo.broadcast_in_dim %1779, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1781 = stablehlo.divide %1780, %16 : tensor<1x257x1xf64>
    %1782 = stablehlo.convert %1781 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1783 = stablehlo.reduce(%1768 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1784 = stablehlo.reshape %1783 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1785 = stablehlo.broadcast_in_dim %1784, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1786 = stablehlo.divide %1785, %32 : tensor<1x257x1xf32>
    %1787 = stablehlo.broadcast_in_dim %1782, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1788 = stablehlo.add %1787, %37 : tensor<1x257x1xf32>
    %1789 = stablehlo.rsqrt %1788 : tensor<1x257x1xf32>
    %1790 = stablehlo.broadcast_in_dim %1768, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1791 = stablehlo.broadcast_in_dim %1786, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1792 = stablehlo.subtract %1790, %1791 : tensor<1x257x768xf32>
    %1793 = stablehlo.broadcast_in_dim %1792, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1794 = stablehlo.broadcast_in_dim %1789, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1795 = stablehlo.multiply %1793, %1794 : tensor<1x257x768xf32>
    %1796 = stablehlo.convert %arg42 : (tensor<768xbf16>) -> tensor<768xf32>
    %1797 = stablehlo.broadcast_in_dim %1795, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1798 = stablehlo.broadcast_in_dim %1796, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1799 = stablehlo.multiply %1797, %1798 : tensor<1x257x768xf32>
    %1800 = stablehlo.convert %arg43 : (tensor<768xbf16>) -> tensor<768xf32>
    %1801 = stablehlo.broadcast_in_dim %1799, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1802 = stablehlo.broadcast_in_dim %1800, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1803 = stablehlo.add %1801, %1802 : tensor<1x257x768xf32>
    %1804 = stablehlo.convert %1803 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1805 = stablehlo.reshape %1804 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1806 = stablehlo.convert %1805 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1807 = stablehlo.dot_general %1806, %arg150, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %1808 = stablehlo.broadcast_in_dim %1807, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1809 = stablehlo.multiply %1808, %160 : tensor<257x3072xf32>
    %1810 = stablehlo.broadcast_in_dim %1809, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1811 = stablehlo.broadcast_in_dim %arg151, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %1812 = stablehlo.add %1810, %1811 : tensor<257x3072xf32>
    %1813 = stablehlo.convert %1812 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %1814 = stablehlo.reshape %1813 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %1815 = stablehlo.multiply %1814, %cst_4 : tensor<1x257x3072xbf16>
    %1816 = stablehlo.multiply %1814, %168 : tensor<1x257x3072xbf16>
    %1817 = stablehlo.convert %1816 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %1818 = stablehlo.clamp %cst_5, %1817, %cst_6 : tensor<1x257x3072xf32>
    %1819 = stablehlo.multiply %1818, %1818 : tensor<1x257x3072xf32>
    %1820 = stablehlo.multiply %cst_7, %1819 : tensor<1x257x3072xf32>
    %1821 = stablehlo.add %1820, %cst_8 : tensor<1x257x3072xf32>
    %1822 = stablehlo.multiply %1821, %1819 : tensor<1x257x3072xf32>
    %1823 = stablehlo.add %1822, %cst_9 : tensor<1x257x3072xf32>
    %1824 = stablehlo.multiply %1823, %1819 : tensor<1x257x3072xf32>
    %1825 = stablehlo.add %1824, %cst_10 : tensor<1x257x3072xf32>
    %1826 = stablehlo.multiply %1825, %1819 : tensor<1x257x3072xf32>
    %1827 = stablehlo.add %1826, %cst_11 : tensor<1x257x3072xf32>
    %1828 = stablehlo.multiply %1827, %1819 : tensor<1x257x3072xf32>
    %1829 = stablehlo.add %1828, %cst_12 : tensor<1x257x3072xf32>
    %1830 = stablehlo.multiply %1829, %1819 : tensor<1x257x3072xf32>
    %1831 = stablehlo.add %1830, %cst_13 : tensor<1x257x3072xf32>
    %1832 = stablehlo.multiply %cst_14, %1819 : tensor<1x257x3072xf32>
    %1833 = stablehlo.add %1832, %cst_15 : tensor<1x257x3072xf32>
    %1834 = stablehlo.multiply %1833, %1819 : tensor<1x257x3072xf32>
    %1835 = stablehlo.add %1834, %cst_16 : tensor<1x257x3072xf32>
    %1836 = stablehlo.multiply %1835, %1819 : tensor<1x257x3072xf32>
    %1837 = stablehlo.add %1836, %cst_17 : tensor<1x257x3072xf32>
    %1838 = stablehlo.multiply %1837, %1819 : tensor<1x257x3072xf32>
    %1839 = stablehlo.add %1838, %cst_18 : tensor<1x257x3072xf32>
    %1840 = stablehlo.multiply %1818, %1831 : tensor<1x257x3072xf32>
    %1841 = stablehlo.divide %1840, %1839 : tensor<1x257x3072xf32>
    %1842 = stablehlo.clamp %cst_19, %1841, %cst_20 : tensor<1x257x3072xf32>
    %1843 = stablehlo.convert %1842 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %1844 = stablehlo.add %1843, %cst_2 : tensor<1x257x3072xbf16>
    %1845 = stablehlo.multiply %1844, %1815 : tensor<1x257x3072xbf16>
    %1846 = stablehlo.reshape %1845 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %1847 = stablehlo.convert %1846 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %1848 = stablehlo.dot_general %1847, %arg152, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %1849 = stablehlo.broadcast_in_dim %1848, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1850 = stablehlo.multiply %1849, %111 : tensor<257x768xf32>
    %1851 = stablehlo.broadcast_in_dim %1850, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1852 = stablehlo.broadcast_in_dim %arg153, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1853 = stablehlo.add %1851, %1852 : tensor<257x768xf32>
    %1854 = stablehlo.convert %1853 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1855 = stablehlo.reshape %1854 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1856 = stablehlo.add %1767, %1855 : tensor<1x257x768xbf16>
    %1857 = stablehlo.convert %1856 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1858 = stablehlo.convert %1857 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1859 = stablehlo.reduce(%1858 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1860 = stablehlo.reshape %1859 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1861 = stablehlo.broadcast_in_dim %1860, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1862 = stablehlo.divide %1861, %16 : tensor<1x257x1xf64>
    %1863 = stablehlo.broadcast_in_dim %1858, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1864 = stablehlo.broadcast_in_dim %1862, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1865 = stablehlo.subtract %1863, %1864 : tensor<1x257x768xf64>
    %1866 = stablehlo.multiply %1865, %1865 : tensor<1x257x768xf64>
    %1867 = stablehlo.reduce(%1866 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1868 = stablehlo.reshape %1867 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1869 = stablehlo.broadcast_in_dim %1868, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1870 = stablehlo.divide %1869, %16 : tensor<1x257x1xf64>
    %1871 = stablehlo.convert %1870 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1872 = stablehlo.reduce(%1857 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1873 = stablehlo.reshape %1872 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1874 = stablehlo.broadcast_in_dim %1873, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1875 = stablehlo.divide %1874, %32 : tensor<1x257x1xf32>
    %1876 = stablehlo.broadcast_in_dim %1871, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1877 = stablehlo.add %1876, %37 : tensor<1x257x1xf32>
    %1878 = stablehlo.rsqrt %1877 : tensor<1x257x1xf32>
    %1879 = stablehlo.broadcast_in_dim %1857, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1880 = stablehlo.broadcast_in_dim %1875, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1881 = stablehlo.subtract %1879, %1880 : tensor<1x257x768xf32>
    %1882 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1883 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1884 = stablehlo.multiply %1882, %1883 : tensor<1x257x768xf32>
    %1885 = stablehlo.convert %arg44 : (tensor<768xbf16>) -> tensor<768xf32>
    %1886 = stablehlo.broadcast_in_dim %1884, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1887 = stablehlo.broadcast_in_dim %1885, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1888 = stablehlo.multiply %1886, %1887 : tensor<1x257x768xf32>
    %1889 = stablehlo.convert %arg45 : (tensor<768xbf16>) -> tensor<768xf32>
    %1890 = stablehlo.broadcast_in_dim %1888, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1891 = stablehlo.broadcast_in_dim %1889, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1892 = stablehlo.add %1890, %1891 : tensor<1x257x768xf32>
    %1893 = stablehlo.convert %1892 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1894 = stablehlo.reshape %1893 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1895 = stablehlo.convert %1894 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1896 = stablehlo.dot_general %1895, %arg154, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %1897 = stablehlo.broadcast_in_dim %1896, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1898 = stablehlo.multiply %1897, %61 : tensor<257x2304xf32>
    %1899 = stablehlo.broadcast_in_dim %1898, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %1900 = stablehlo.broadcast_in_dim %arg155, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %1901 = stablehlo.add %1899, %1900 : tensor<257x2304xf32>
    %1902 = stablehlo.convert %1901 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %1903 = stablehlo.reshape %1902 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %1904 = stablehlo.reshape %1903 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %1905 = stablehlo.transpose %1904, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %1906 = stablehlo.slice %1905 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1907 = stablehlo.reshape %1906 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1908 = stablehlo.slice %1905 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1909 = stablehlo.reshape %1908 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1910 = stablehlo.slice %1905 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %1911 = stablehlo.reshape %1910 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1912 = stablehlo.transpose %1909, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %1913 = stablehlo.reshape %1907 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1914 = stablehlo.reshape %1912 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1915 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %1916 = stablehlo.dot_general %1913, %1915, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %1917 = stablehlo.reshape %1916 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1918 = stablehlo.broadcast_in_dim %1917, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %1919 = stablehlo.multiply %1918, %85 : tensor<1x12x257x257xbf16>
    %1920 = stablehlo.convert %1919 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %1921 = stablehlo.reduce(%1920 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1922 = stablehlo.reshape %1921 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1923 = stablehlo.broadcast_in_dim %1920, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1924 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1925 = stablehlo.subtract %1923, %1924 : tensor<1x12x257x257xf32>
    %1926 = stablehlo.exponential %1925 : tensor<1x12x257x257xf32>
    %1927 = stablehlo.reduce(%1926 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %1928 = stablehlo.reshape %1927 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %1929 = stablehlo.broadcast_in_dim %1926, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %1930 = stablehlo.broadcast_in_dim %1928, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %1931 = stablehlo.divide %1929, %1930 : tensor<1x12x257x257xf32>
    %1932 = stablehlo.convert %1931 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %1933 = stablehlo.reshape %1932 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %1934 = stablehlo.reshape %1911 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1935 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1936 = stablehlo.dot_general %1933, %1935, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %1937 = stablehlo.reshape %1936 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %1938 = stablehlo.transpose %1937, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %1939 = stablehlo.reshape %1938 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %1940 = stablehlo.reshape %1939 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1941 = stablehlo.convert %1940 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1942 = stablehlo.dot_general %1941, %arg156, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %1943 = stablehlo.broadcast_in_dim %1942, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1944 = stablehlo.multiply %1943, %111 : tensor<257x768xf32>
    %1945 = stablehlo.broadcast_in_dim %1944, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %1946 = stablehlo.broadcast_in_dim %arg157, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %1947 = stablehlo.add %1945, %1946 : tensor<257x768xf32>
    %1948 = stablehlo.convert %1947 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %1949 = stablehlo.reshape %1948 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %1950 = stablehlo.add %1949, %1856 : tensor<1x257x768xbf16>
    %1951 = stablehlo.convert %1950 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %1952 = stablehlo.convert %1951 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %1953 = stablehlo.reduce(%1952 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1954 = stablehlo.reshape %1953 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1956 = stablehlo.divide %1955, %16 : tensor<1x257x1xf64>
    %1957 = stablehlo.broadcast_in_dim %1952, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %1958 = stablehlo.broadcast_in_dim %1956, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %1959 = stablehlo.subtract %1957, %1958 : tensor<1x257x768xf64>
    %1960 = stablehlo.multiply %1959, %1959 : tensor<1x257x768xf64>
    %1961 = stablehlo.reduce(%1960 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %1962 = stablehlo.reshape %1961 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %1963 = stablehlo.broadcast_in_dim %1962, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %1964 = stablehlo.divide %1963, %16 : tensor<1x257x1xf64>
    %1965 = stablehlo.convert %1964 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %1966 = stablehlo.reduce(%1951 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %1967 = stablehlo.reshape %1966 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %1968 = stablehlo.broadcast_in_dim %1967, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1969 = stablehlo.divide %1968, %32 : tensor<1x257x1xf32>
    %1970 = stablehlo.broadcast_in_dim %1965, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %1971 = stablehlo.add %1970, %37 : tensor<1x257x1xf32>
    %1972 = stablehlo.rsqrt %1971 : tensor<1x257x1xf32>
    %1973 = stablehlo.broadcast_in_dim %1951, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1974 = stablehlo.broadcast_in_dim %1969, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1975 = stablehlo.subtract %1973, %1974 : tensor<1x257x768xf32>
    %1976 = stablehlo.broadcast_in_dim %1975, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1977 = stablehlo.broadcast_in_dim %1972, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %1978 = stablehlo.multiply %1976, %1977 : tensor<1x257x768xf32>
    %1979 = stablehlo.convert %arg46 : (tensor<768xbf16>) -> tensor<768xf32>
    %1980 = stablehlo.broadcast_in_dim %1978, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1981 = stablehlo.broadcast_in_dim %1979, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1982 = stablehlo.multiply %1980, %1981 : tensor<1x257x768xf32>
    %1983 = stablehlo.convert %arg47 : (tensor<768xbf16>) -> tensor<768xf32>
    %1984 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %1985 = stablehlo.broadcast_in_dim %1983, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %1986 = stablehlo.add %1984, %1985 : tensor<1x257x768xf32>
    %1987 = stablehlo.convert %1986 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %1988 = stablehlo.reshape %1987 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %1989 = stablehlo.convert %1988 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %1990 = stablehlo.dot_general %1989, %arg158, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %1991 = stablehlo.broadcast_in_dim %1990, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1992 = stablehlo.multiply %1991, %160 : tensor<257x3072xf32>
    %1993 = stablehlo.broadcast_in_dim %1992, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %1994 = stablehlo.broadcast_in_dim %arg159, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %1995 = stablehlo.add %1993, %1994 : tensor<257x3072xf32>
    %1996 = stablehlo.convert %1995 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %1997 = stablehlo.reshape %1996 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %1998 = stablehlo.multiply %1997, %cst_4 : tensor<1x257x3072xbf16>
    %1999 = stablehlo.multiply %1997, %168 : tensor<1x257x3072xbf16>
    %2000 = stablehlo.convert %1999 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %2001 = stablehlo.clamp %cst_5, %2000, %cst_6 : tensor<1x257x3072xf32>
    %2002 = stablehlo.multiply %2001, %2001 : tensor<1x257x3072xf32>
    %2003 = stablehlo.multiply %cst_7, %2002 : tensor<1x257x3072xf32>
    %2004 = stablehlo.add %2003, %cst_8 : tensor<1x257x3072xf32>
    %2005 = stablehlo.multiply %2004, %2002 : tensor<1x257x3072xf32>
    %2006 = stablehlo.add %2005, %cst_9 : tensor<1x257x3072xf32>
    %2007 = stablehlo.multiply %2006, %2002 : tensor<1x257x3072xf32>
    %2008 = stablehlo.add %2007, %cst_10 : tensor<1x257x3072xf32>
    %2009 = stablehlo.multiply %2008, %2002 : tensor<1x257x3072xf32>
    %2010 = stablehlo.add %2009, %cst_11 : tensor<1x257x3072xf32>
    %2011 = stablehlo.multiply %2010, %2002 : tensor<1x257x3072xf32>
    %2012 = stablehlo.add %2011, %cst_12 : tensor<1x257x3072xf32>
    %2013 = stablehlo.multiply %2012, %2002 : tensor<1x257x3072xf32>
    %2014 = stablehlo.add %2013, %cst_13 : tensor<1x257x3072xf32>
    %2015 = stablehlo.multiply %cst_14, %2002 : tensor<1x257x3072xf32>
    %2016 = stablehlo.add %2015, %cst_15 : tensor<1x257x3072xf32>
    %2017 = stablehlo.multiply %2016, %2002 : tensor<1x257x3072xf32>
    %2018 = stablehlo.add %2017, %cst_16 : tensor<1x257x3072xf32>
    %2019 = stablehlo.multiply %2018, %2002 : tensor<1x257x3072xf32>
    %2020 = stablehlo.add %2019, %cst_17 : tensor<1x257x3072xf32>
    %2021 = stablehlo.multiply %2020, %2002 : tensor<1x257x3072xf32>
    %2022 = stablehlo.add %2021, %cst_18 : tensor<1x257x3072xf32>
    %2023 = stablehlo.multiply %2001, %2014 : tensor<1x257x3072xf32>
    %2024 = stablehlo.divide %2023, %2022 : tensor<1x257x3072xf32>
    %2025 = stablehlo.clamp %cst_19, %2024, %cst_20 : tensor<1x257x3072xf32>
    %2026 = stablehlo.convert %2025 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %2027 = stablehlo.add %2026, %cst_2 : tensor<1x257x3072xbf16>
    %2028 = stablehlo.multiply %2027, %1998 : tensor<1x257x3072xbf16>
    %2029 = stablehlo.reshape %2028 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %2030 = stablehlo.convert %2029 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %2031 = stablehlo.dot_general %2030, %arg160, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %2032 = stablehlo.broadcast_in_dim %2031, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %2033 = stablehlo.multiply %2032, %111 : tensor<257x768xf32>
    %2034 = stablehlo.broadcast_in_dim %2033, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %2035 = stablehlo.broadcast_in_dim %arg161, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %2036 = stablehlo.add %2034, %2035 : tensor<257x768xf32>
    %2037 = stablehlo.convert %2036 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %2038 = stablehlo.reshape %2037 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %2039 = stablehlo.add %1950, %2038 : tensor<1x257x768xbf16>
    %2040 = stablehlo.convert %2039 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %2041 = stablehlo.convert %2040 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %2042 = stablehlo.reduce(%2041 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %2043 = stablehlo.reshape %2042 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %2044 = stablehlo.broadcast_in_dim %2043, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %2045 = stablehlo.divide %2044, %16 : tensor<1x257x1xf64>
    %2046 = stablehlo.broadcast_in_dim %2041, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %2047 = stablehlo.broadcast_in_dim %2045, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %2048 = stablehlo.subtract %2046, %2047 : tensor<1x257x768xf64>
    %2049 = stablehlo.multiply %2048, %2048 : tensor<1x257x768xf64>
    %2050 = stablehlo.reduce(%2049 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %2051 = stablehlo.reshape %2050 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %2052 = stablehlo.broadcast_in_dim %2051, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %2053 = stablehlo.divide %2052, %16 : tensor<1x257x1xf64>
    %2054 = stablehlo.convert %2053 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %2055 = stablehlo.reduce(%2040 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %2056 = stablehlo.reshape %2055 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %2057 = stablehlo.broadcast_in_dim %2056, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %2058 = stablehlo.divide %2057, %32 : tensor<1x257x1xf32>
    %2059 = stablehlo.broadcast_in_dim %2054, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %2060 = stablehlo.add %2059, %37 : tensor<1x257x1xf32>
    %2061 = stablehlo.rsqrt %2060 : tensor<1x257x1xf32>
    %2062 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2063 = stablehlo.broadcast_in_dim %2058, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %2064 = stablehlo.subtract %2062, %2063 : tensor<1x257x768xf32>
    %2065 = stablehlo.broadcast_in_dim %2064, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2066 = stablehlo.broadcast_in_dim %2061, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %2067 = stablehlo.multiply %2065, %2066 : tensor<1x257x768xf32>
    %2068 = stablehlo.convert %arg48 : (tensor<768xbf16>) -> tensor<768xf32>
    %2069 = stablehlo.broadcast_in_dim %2067, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2070 = stablehlo.broadcast_in_dim %2068, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2071 = stablehlo.multiply %2069, %2070 : tensor<1x257x768xf32>
    %2072 = stablehlo.convert %arg49 : (tensor<768xbf16>) -> tensor<768xf32>
    %2073 = stablehlo.broadcast_in_dim %2071, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2074 = stablehlo.broadcast_in_dim %2072, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2075 = stablehlo.add %2073, %2074 : tensor<1x257x768xf32>
    %2076 = stablehlo.convert %2075 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %2077 = stablehlo.reshape %2076 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %2078 = stablehlo.convert %2077 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %2079 = stablehlo.dot_general %2078, %arg162, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x2304xf32>) -> tensor<257x2304xf32>
    %2080 = stablehlo.broadcast_in_dim %2079, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %2081 = stablehlo.multiply %2080, %61 : tensor<257x2304xf32>
    %2082 = stablehlo.broadcast_in_dim %2081, dims = [0, 1] : (tensor<257x2304xf32>) -> tensor<257x2304xf32>
    %2083 = stablehlo.broadcast_in_dim %arg163, dims = [1] : (tensor<2304xf32>) -> tensor<257x2304xf32>
    %2084 = stablehlo.add %2082, %2083 : tensor<257x2304xf32>
    %2085 = stablehlo.convert %2084 : (tensor<257x2304xf32>) -> tensor<257x2304xbf16>
    %2086 = stablehlo.reshape %2085 : (tensor<257x2304xbf16>) -> tensor<1x257x2304xbf16>
    %2087 = stablehlo.reshape %2086 : (tensor<1x257x2304xbf16>) -> tensor<1x257x3x12x64xbf16>
    %2088 = stablehlo.transpose %2087, dims = [2, 0, 3, 1, 4] : (tensor<1x257x3x12x64xbf16>) -> tensor<3x1x12x257x64xbf16>
    %2089 = stablehlo.slice %2088 [0:1, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %2090 = stablehlo.reshape %2089 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %2091 = stablehlo.slice %2088 [1:2, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %2092 = stablehlo.reshape %2091 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %2093 = stablehlo.slice %2088 [2:3, 0:1, 0:12, 0:257, 0:64] : (tensor<3x1x12x257x64xbf16>) -> tensor<1x1x12x257x64xbf16>
    %2094 = stablehlo.reshape %2093 : (tensor<1x1x12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %2095 = stablehlo.transpose %2092, dims = [0, 1, 3, 2] : (tensor<1x12x257x64xbf16>) -> tensor<1x12x64x257xbf16>
    %2096 = stablehlo.reshape %2090 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %2097 = stablehlo.reshape %2095 : (tensor<1x12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %2098 = stablehlo.broadcast_in_dim %2097, dims = [0, 1, 2] : (tensor<12x64x257xbf16>) -> tensor<12x64x257xbf16>
    %2099 = stablehlo.dot_general %2096, %2098, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x64xbf16>, tensor<12x64x257xbf16>) -> tensor<12x257x257xbf16>
    %2100 = stablehlo.reshape %2099 : (tensor<12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %2101 = stablehlo.broadcast_in_dim %2100, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xbf16>
    %2102 = stablehlo.multiply %2101, %85 : tensor<1x12x257x257xbf16>
    %2103 = stablehlo.convert %2102 : (tensor<1x12x257x257xbf16>) -> tensor<1x12x257x257xf32>
    %2104 = stablehlo.reduce(%2103 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %2105 = stablehlo.reshape %2104 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %2106 = stablehlo.broadcast_in_dim %2103, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %2107 = stablehlo.broadcast_in_dim %2105, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %2108 = stablehlo.subtract %2106, %2107 : tensor<1x12x257x257xf32>
    %2109 = stablehlo.exponential %2108 : tensor<1x12x257x257xf32>
    %2110 = stablehlo.reduce(%2109 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x257x257xf32>, tensor<f32>) -> tensor<1x12x257xf32>
    %2111 = stablehlo.reshape %2110 : (tensor<1x12x257xf32>) -> tensor<1x12x257x1xf32>
    %2112 = stablehlo.broadcast_in_dim %2109, dims = [0, 1, 2, 3] : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xf32>
    %2113 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 2, 3] : (tensor<1x12x257x1xf32>) -> tensor<1x12x257x257xf32>
    %2114 = stablehlo.divide %2112, %2113 : tensor<1x12x257x257xf32>
    %2115 = stablehlo.convert %2114 : (tensor<1x12x257x257xf32>) -> tensor<1x12x257x257xbf16>
    %2116 = stablehlo.reshape %2115 : (tensor<1x12x257x257xbf16>) -> tensor<12x257x257xbf16>
    %2117 = stablehlo.reshape %2094 : (tensor<1x12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %2118 = stablehlo.broadcast_in_dim %2117, dims = [0, 1, 2] : (tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %2119 = stablehlo.dot_general %2116, %2118, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x257x257xbf16>, tensor<12x257x64xbf16>) -> tensor<12x257x64xbf16>
    %2120 = stablehlo.reshape %2119 : (tensor<12x257x64xbf16>) -> tensor<1x12x257x64xbf16>
    %2121 = stablehlo.transpose %2120, dims = [0, 2, 1, 3] : (tensor<1x12x257x64xbf16>) -> tensor<1x257x12x64xbf16>
    %2122 = stablehlo.reshape %2121 : (tensor<1x257x12x64xbf16>) -> tensor<1x257x768xbf16>
    %2123 = stablehlo.reshape %2122 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %2124 = stablehlo.convert %2123 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %2125 = stablehlo.dot_general %2124, %arg164, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x768xf32>) -> tensor<257x768xf32>
    %2126 = stablehlo.broadcast_in_dim %2125, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %2127 = stablehlo.multiply %2126, %111 : tensor<257x768xf32>
    %2128 = stablehlo.broadcast_in_dim %2127, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %2129 = stablehlo.broadcast_in_dim %arg165, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %2130 = stablehlo.add %2128, %2129 : tensor<257x768xf32>
    %2131 = stablehlo.convert %2130 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %2132 = stablehlo.reshape %2131 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %2133 = stablehlo.add %2132, %2039 : tensor<1x257x768xbf16>
    %2134 = stablehlo.convert %2133 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %2135 = stablehlo.convert %2134 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %2136 = stablehlo.reduce(%2135 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %2137 = stablehlo.reshape %2136 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %2138 = stablehlo.broadcast_in_dim %2137, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %2139 = stablehlo.divide %2138, %16 : tensor<1x257x1xf64>
    %2140 = stablehlo.broadcast_in_dim %2135, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %2141 = stablehlo.broadcast_in_dim %2139, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %2142 = stablehlo.subtract %2140, %2141 : tensor<1x257x768xf64>
    %2143 = stablehlo.multiply %2142, %2142 : tensor<1x257x768xf64>
    %2144 = stablehlo.reduce(%2143 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %2145 = stablehlo.reshape %2144 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %2146 = stablehlo.broadcast_in_dim %2145, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %2147 = stablehlo.divide %2146, %16 : tensor<1x257x1xf64>
    %2148 = stablehlo.convert %2147 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %2149 = stablehlo.reduce(%2134 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %2150 = stablehlo.reshape %2149 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %2151 = stablehlo.broadcast_in_dim %2150, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %2152 = stablehlo.divide %2151, %32 : tensor<1x257x1xf32>
    %2153 = stablehlo.broadcast_in_dim %2148, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %2154 = stablehlo.add %2153, %37 : tensor<1x257x1xf32>
    %2155 = stablehlo.rsqrt %2154 : tensor<1x257x1xf32>
    %2156 = stablehlo.broadcast_in_dim %2134, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2157 = stablehlo.broadcast_in_dim %2152, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %2158 = stablehlo.subtract %2156, %2157 : tensor<1x257x768xf32>
    %2159 = stablehlo.broadcast_in_dim %2158, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2160 = stablehlo.broadcast_in_dim %2155, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %2161 = stablehlo.multiply %2159, %2160 : tensor<1x257x768xf32>
    %2162 = stablehlo.convert %arg50 : (tensor<768xbf16>) -> tensor<768xf32>
    %2163 = stablehlo.broadcast_in_dim %2161, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2164 = stablehlo.broadcast_in_dim %2162, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2165 = stablehlo.multiply %2163, %2164 : tensor<1x257x768xf32>
    %2166 = stablehlo.convert %arg51 : (tensor<768xbf16>) -> tensor<768xf32>
    %2167 = stablehlo.broadcast_in_dim %2165, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2168 = stablehlo.broadcast_in_dim %2166, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2169 = stablehlo.add %2167, %2168 : tensor<1x257x768xf32>
    %2170 = stablehlo.convert %2169 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %2171 = stablehlo.reshape %2170 : (tensor<1x257x768xbf16>) -> tensor<257x768xbf16>
    %2172 = stablehlo.convert %2171 : (tensor<257x768xbf16>) -> tensor<257x768xf32>
    %2173 = stablehlo.dot_general %2172, %arg166, contracting_dims = [1] x [0] : (tensor<257x768xf32>, tensor<768x3072xf32>) -> tensor<257x3072xf32>
    %2174 = stablehlo.broadcast_in_dim %2173, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %2175 = stablehlo.multiply %2174, %160 : tensor<257x3072xf32>
    %2176 = stablehlo.broadcast_in_dim %2175, dims = [0, 1] : (tensor<257x3072xf32>) -> tensor<257x3072xf32>
    %2177 = stablehlo.broadcast_in_dim %arg167, dims = [1] : (tensor<3072xf32>) -> tensor<257x3072xf32>
    %2178 = stablehlo.add %2176, %2177 : tensor<257x3072xf32>
    %2179 = stablehlo.convert %2178 : (tensor<257x3072xf32>) -> tensor<257x3072xbf16>
    %2180 = stablehlo.reshape %2179 : (tensor<257x3072xbf16>) -> tensor<1x257x3072xbf16>
    %2181 = stablehlo.multiply %2180, %cst_4 : tensor<1x257x3072xbf16>
    %2182 = stablehlo.multiply %2180, %168 : tensor<1x257x3072xbf16>
    %2183 = stablehlo.convert %2182 : (tensor<1x257x3072xbf16>) -> tensor<1x257x3072xf32>
    %2184 = stablehlo.clamp %cst_5, %2183, %cst_6 : tensor<1x257x3072xf32>
    %2185 = stablehlo.multiply %2184, %2184 : tensor<1x257x3072xf32>
    %2186 = stablehlo.multiply %cst_7, %2185 : tensor<1x257x3072xf32>
    %2187 = stablehlo.add %2186, %cst_8 : tensor<1x257x3072xf32>
    %2188 = stablehlo.multiply %2187, %2185 : tensor<1x257x3072xf32>
    %2189 = stablehlo.add %2188, %cst_9 : tensor<1x257x3072xf32>
    %2190 = stablehlo.multiply %2189, %2185 : tensor<1x257x3072xf32>
    %2191 = stablehlo.add %2190, %cst_10 : tensor<1x257x3072xf32>
    %2192 = stablehlo.multiply %2191, %2185 : tensor<1x257x3072xf32>
    %2193 = stablehlo.add %2192, %cst_11 : tensor<1x257x3072xf32>
    %2194 = stablehlo.multiply %2193, %2185 : tensor<1x257x3072xf32>
    %2195 = stablehlo.add %2194, %cst_12 : tensor<1x257x3072xf32>
    %2196 = stablehlo.multiply %2195, %2185 : tensor<1x257x3072xf32>
    %2197 = stablehlo.add %2196, %cst_13 : tensor<1x257x3072xf32>
    %2198 = stablehlo.multiply %cst_14, %2185 : tensor<1x257x3072xf32>
    %2199 = stablehlo.add %2198, %cst_15 : tensor<1x257x3072xf32>
    %2200 = stablehlo.multiply %2199, %2185 : tensor<1x257x3072xf32>
    %2201 = stablehlo.add %2200, %cst_16 : tensor<1x257x3072xf32>
    %2202 = stablehlo.multiply %2201, %2185 : tensor<1x257x3072xf32>
    %2203 = stablehlo.add %2202, %cst_17 : tensor<1x257x3072xf32>
    %2204 = stablehlo.multiply %2203, %2185 : tensor<1x257x3072xf32>
    %2205 = stablehlo.add %2204, %cst_18 : tensor<1x257x3072xf32>
    %2206 = stablehlo.multiply %2184, %2197 : tensor<1x257x3072xf32>
    %2207 = stablehlo.divide %2206, %2205 : tensor<1x257x3072xf32>
    %2208 = stablehlo.clamp %cst_19, %2207, %cst_20 : tensor<1x257x3072xf32>
    %2209 = stablehlo.convert %2208 : (tensor<1x257x3072xf32>) -> tensor<1x257x3072xbf16>
    %2210 = stablehlo.add %2209, %cst_2 : tensor<1x257x3072xbf16>
    %2211 = stablehlo.multiply %2210, %2181 : tensor<1x257x3072xbf16>
    %2212 = stablehlo.reshape %2211 : (tensor<1x257x3072xbf16>) -> tensor<257x3072xbf16>
    %2213 = stablehlo.convert %2212 : (tensor<257x3072xbf16>) -> tensor<257x3072xf32>
    %2214 = stablehlo.dot_general %2213, %arg168, contracting_dims = [1] x [0] : (tensor<257x3072xf32>, tensor<3072x768xf32>) -> tensor<257x768xf32>
    %2215 = stablehlo.broadcast_in_dim %2214, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %2216 = stablehlo.multiply %2215, %111 : tensor<257x768xf32>
    %2217 = stablehlo.broadcast_in_dim %2216, dims = [0, 1] : (tensor<257x768xf32>) -> tensor<257x768xf32>
    %2218 = stablehlo.broadcast_in_dim %arg169, dims = [1] : (tensor<768xf32>) -> tensor<257x768xf32>
    %2219 = stablehlo.add %2217, %2218 : tensor<257x768xf32>
    %2220 = stablehlo.convert %2219 : (tensor<257x768xf32>) -> tensor<257x768xbf16>
    %2221 = stablehlo.reshape %2220 : (tensor<257x768xbf16>) -> tensor<1x257x768xbf16>
    %2222 = stablehlo.add %2133, %2221 : tensor<1x257x768xbf16>
    %2223 = stablehlo.convert %2222 : (tensor<1x257x768xbf16>) -> tensor<1x257x768xf32>
    %2224 = stablehlo.convert %2223 : (tensor<1x257x768xf32>) -> tensor<1x257x768xf64>
    %2225 = stablehlo.reduce(%2224 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %2226 = stablehlo.reshape %2225 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %2227 = stablehlo.broadcast_in_dim %2226, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %2228 = stablehlo.divide %2227, %16 : tensor<1x257x1xf64>
    %2229 = stablehlo.broadcast_in_dim %2224, dims = [0, 1, 2] : (tensor<1x257x768xf64>) -> tensor<1x257x768xf64>
    %2230 = stablehlo.broadcast_in_dim %2228, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x768xf64>
    %2231 = stablehlo.subtract %2229, %2230 : tensor<1x257x768xf64>
    %2232 = stablehlo.multiply %2231, %2231 : tensor<1x257x768xf64>
    %2233 = stablehlo.reduce(%2232 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf64>, tensor<f64>) -> tensor<1x257xf64>
    %2234 = stablehlo.reshape %2233 : (tensor<1x257xf64>) -> tensor<1x257x1xf64>
    %2235 = stablehlo.broadcast_in_dim %2234, dims = [0, 1, 2] : (tensor<1x257x1xf64>) -> tensor<1x257x1xf64>
    %2236 = stablehlo.divide %2235, %16 : tensor<1x257x1xf64>
    %2237 = stablehlo.convert %2236 : (tensor<1x257x1xf64>) -> tensor<1x257x1xf32>
    %2238 = stablehlo.reduce(%2223 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x257x768xf32>, tensor<f32>) -> tensor<1x257xf32>
    %2239 = stablehlo.reshape %2238 : (tensor<1x257xf32>) -> tensor<1x257x1xf32>
    %2240 = stablehlo.broadcast_in_dim %2239, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %2241 = stablehlo.divide %2240, %32 : tensor<1x257x1xf32>
    %2242 = stablehlo.broadcast_in_dim %2237, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x1xf32>
    %2243 = stablehlo.add %2242, %37 : tensor<1x257x1xf32>
    %2244 = stablehlo.rsqrt %2243 : tensor<1x257x1xf32>
    %2245 = stablehlo.broadcast_in_dim %2223, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2246 = stablehlo.broadcast_in_dim %2241, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %2247 = stablehlo.subtract %2245, %2246 : tensor<1x257x768xf32>
    %2248 = stablehlo.broadcast_in_dim %2247, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2249 = stablehlo.broadcast_in_dim %2244, dims = [0, 1, 2] : (tensor<1x257x1xf32>) -> tensor<1x257x768xf32>
    %2250 = stablehlo.multiply %2248, %2249 : tensor<1x257x768xf32>
    %2251 = stablehlo.convert %arg52 : (tensor<768xbf16>) -> tensor<768xf32>
    %2252 = stablehlo.broadcast_in_dim %2250, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2253 = stablehlo.broadcast_in_dim %2251, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2254 = stablehlo.multiply %2252, %2253 : tensor<1x257x768xf32>
    %2255 = stablehlo.convert %arg53 : (tensor<768xbf16>) -> tensor<768xf32>
    %2256 = stablehlo.broadcast_in_dim %2254, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2257 = stablehlo.broadcast_in_dim %2255, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2258 = stablehlo.add %2256, %2257 : tensor<1x257x768xf32>
    %2259 = stablehlo.convert %2258 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %2260 = stablehlo.transpose %2259, dims = [0, 2, 1] : (tensor<1x257x768xbf16>) -> tensor<1x768x257xbf16>
    %2261 = stablehlo.reshape %2260 : (tensor<1x768x257xbf16>) -> tensor<1x768x257x1xbf16>
    %2262 = stablehlo.convolution(%2261, %arg54) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 8 : i64} : (tensor<1x768x257x1xbf16>, tensor<768x96x1x1xbf16>) -> tensor<1x768x257x1xbf16>
    %2263 = stablehlo.convolution(%2262, %arg55) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x768x257x1xbf16>, tensor<27x768x1x1xbf16>) -> tensor<1x27x257x1xbf16>
    %2264 = stablehlo.reshape %2263 : (tensor<1x27x257x1xbf16>) -> tensor<1x27x257xbf16>
    %2265 = stablehlo.convert %2264 : (tensor<1x27x257xbf16>) -> tensor<1x27x257xf32>
    %2266 = stablehlo.reduce(%2265 init: %cst_1) applies stablehlo.maximum across dimensions = [2] : (tensor<1x27x257xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2267 = stablehlo.reshape %2266 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2268 = stablehlo.broadcast_in_dim %2265, dims = [0, 1, 2] : (tensor<1x27x257xf32>) -> tensor<1x27x257xf32>
    %2269 = stablehlo.broadcast_in_dim %2267, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x257xf32>
    %2270 = stablehlo.subtract %2268, %2269 : tensor<1x27x257xf32>
    %2271 = stablehlo.exponential %2270 : tensor<1x27x257xf32>
    %2272 = stablehlo.reduce(%2271 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x27x257xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2273 = stablehlo.reshape %2272 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2274 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2] : (tensor<1x27x257xf32>) -> tensor<1x27x257xf32>
    %2275 = stablehlo.broadcast_in_dim %2273, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x257xf32>
    %2276 = stablehlo.divide %2274, %2275 : tensor<1x27x257xf32>
    %2277 = stablehlo.convert %2276 : (tensor<1x27x257xf32>) -> tensor<1x27x257xbf16>
    %2278 = stablehlo.convolution(%2261, %arg56) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 8 : i64} : (tensor<1x768x257x1xbf16>, tensor<768x96x1x1xbf16>) -> tensor<1x768x257x1xbf16>
    %2279 = stablehlo.reshape %2278 : (tensor<1x768x257x1xbf16>) -> tensor<1x768x257xbf16>
    %2280 = stablehlo.transpose %2279, dims = [0, 2, 1] : (tensor<1x768x257xbf16>) -> tensor<1x257x768xbf16>
    %2281 = stablehlo.reshape %2277 : (tensor<1x27x257xbf16>) -> tensor<1x27x257x1xbf16>
    %2282 = stablehlo.transpose %2281, dims = [0, 1, 3, 2] : (tensor<1x27x257x1xbf16>) -> tensor<1x27x1x257xbf16>
    %2283 = stablehlo.reshape %2280 : (tensor<1x257x768xbf16>) -> tensor<1x257x768x1xbf16>
    %2284 = stablehlo.transpose %2283, dims = [0, 3, 2, 1] : (tensor<1x257x768x1xbf16>) -> tensor<1x1x768x257xbf16>
    %2285 = stablehlo.transpose %2282, dims = [1, 3, 0, 2] : (tensor<1x27x1x257xbf16>) -> tensor<27x257x1x1xbf16>
    %2286 = stablehlo.reshape %2285 : (tensor<27x257x1x1xbf16>) -> tensor<1x27x257xbf16>
    %2287 = stablehlo.transpose %2284, dims = [3, 0, 2, 1] : (tensor<1x1x768x257xbf16>) -> tensor<257x1x768x1xbf16>
    %2288 = stablehlo.reshape %2287 : (tensor<257x1x768x1xbf16>) -> tensor<1x257x768xbf16>
    %2289 = stablehlo.broadcast_in_dim %2288, dims = [0, 1, 2] : (tensor<1x257x768xbf16>) -> tensor<1x257x768xbf16>
    %2290 = stablehlo.dot_general %2286, %2289, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x27x257xbf16>, tensor<1x257x768xbf16>) -> tensor<1x27x768xbf16>
    %2291 = stablehlo.reshape %2290 : (tensor<1x27x768xbf16>) -> tensor<27x1x1x768xbf16>
    %2292 = stablehlo.transpose %2291, dims = [2, 0, 3, 1] : (tensor<27x1x1x768xbf16>) -> tensor<1x27x768x1xbf16>
    %2293 = stablehlo.reshape %2292 : (tensor<1x27x768x1xbf16>) -> tensor<1x27x768xbf16>
    %2294 = stablehlo.convert %2293 : (tensor<1x27x768xbf16>) -> tensor<1x27x768xf32>
    %2295 = stablehlo.convert %2294 : (tensor<1x27x768xf32>) -> tensor<1x27x768xf64>
    %2296 = stablehlo.reduce(%2295 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf64>, tensor<f64>) -> tensor<1x27xf64>
    %2297 = stablehlo.reshape %2296 : (tensor<1x27xf64>) -> tensor<1x27x1xf64>
    %2298 = stablehlo.broadcast_in_dim %2297, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x1xf64>
    %2299 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f64>) -> tensor<1x27x1xf64>
    %2300 = stablehlo.divide %2298, %2299 : tensor<1x27x1xf64>
    %2301 = stablehlo.broadcast_in_dim %2295, dims = [0, 1, 2] : (tensor<1x27x768xf64>) -> tensor<1x27x768xf64>
    %2302 = stablehlo.broadcast_in_dim %2300, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x768xf64>
    %2303 = stablehlo.subtract %2301, %2302 : tensor<1x27x768xf64>
    %2304 = stablehlo.multiply %2303, %2303 : tensor<1x27x768xf64>
    %2305 = stablehlo.reduce(%2304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf64>, tensor<f64>) -> tensor<1x27xf64>
    %2306 = stablehlo.reshape %2305 : (tensor<1x27xf64>) -> tensor<1x27x1xf64>
    %2307 = stablehlo.broadcast_in_dim %2306, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x1xf64>
    %2308 = stablehlo.divide %2307, %2299 : tensor<1x27x1xf64>
    %2309 = stablehlo.convert %2308 : (tensor<1x27x1xf64>) -> tensor<1x27x1xf32>
    %2310 = stablehlo.reduce(%2294 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2311 = stablehlo.reshape %2310 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2312 = stablehlo.broadcast_in_dim %2311, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x1xf32>
    %2313 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2314 = stablehlo.divide %2312, %2313 : tensor<1x27x1xf32>
    %2315 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x1xf32>
    %2316 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2317 = stablehlo.add %2315, %2316 : tensor<1x27x1xf32>
    %2318 = stablehlo.rsqrt %2317 : tensor<1x27x1xf32>
    %2319 = stablehlo.broadcast_in_dim %2294, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2320 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x768xf32>
    %2321 = stablehlo.subtract %2319, %2320 : tensor<1x27x768xf32>
    %2322 = stablehlo.broadcast_in_dim %2321, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2323 = stablehlo.broadcast_in_dim %2318, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x768xf32>
    %2324 = stablehlo.multiply %2322, %2323 : tensor<1x27x768xf32>
    %2325 = stablehlo.convert %arg57 : (tensor<768xbf16>) -> tensor<768xf32>
    %2326 = stablehlo.broadcast_in_dim %2324, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2327 = stablehlo.broadcast_in_dim %2325, dims = [2] : (tensor<768xf32>) -> tensor<1x27x768xf32>
    %2328 = stablehlo.multiply %2326, %2327 : tensor<1x27x768xf32>
    %2329 = stablehlo.convert %arg58 : (tensor<768xbf16>) -> tensor<768xf32>
    %2330 = stablehlo.broadcast_in_dim %2328, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2331 = stablehlo.broadcast_in_dim %2329, dims = [2] : (tensor<768xf32>) -> tensor<1x27x768xf32>
    %2332 = stablehlo.add %2330, %2331 : tensor<1x27x768xf32>
    %2333 = stablehlo.convert %2332 : (tensor<1x27x768xf32>) -> tensor<1x27x768xbf16>
    %2334 = stablehlo.convert %arg59 : (tensor<768xbf16>) -> tensor<768xf32>
    %2335 = stablehlo.broadcast_in_dim %2334, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2336 = stablehlo.multiply %2252, %2335 : tensor<1x257x768xf32>
    %2337 = stablehlo.convert %arg60 : (tensor<768xbf16>) -> tensor<768xf32>
    %2338 = stablehlo.broadcast_in_dim %2336, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2339 = stablehlo.broadcast_in_dim %2337, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2340 = stablehlo.add %2338, %2339 : tensor<1x257x768xf32>
    %2341 = stablehlo.convert %2340 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %2342 = stablehlo.transpose %2341, dims = [0, 2, 1] : (tensor<1x257x768xbf16>) -> tensor<1x768x257xbf16>
    %2343 = stablehlo.reshape %2342 : (tensor<1x768x257xbf16>) -> tensor<1x768x257x1xbf16>
    %2344 = stablehlo.convolution(%2343, %arg61) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 8 : i64} : (tensor<1x768x257x1xbf16>, tensor<768x96x1x1xbf16>) -> tensor<1x768x257x1xbf16>
    %2345 = stablehlo.convolution(%2344, %arg62) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x768x257x1xbf16>, tensor<27x768x1x1xbf16>) -> tensor<1x27x257x1xbf16>
    %2346 = stablehlo.reshape %2345 : (tensor<1x27x257x1xbf16>) -> tensor<1x27x257xbf16>
    %2347 = stablehlo.convert %2346 : (tensor<1x27x257xbf16>) -> tensor<1x27x257xf32>
    %2348 = stablehlo.reduce(%2347 init: %cst_1) applies stablehlo.maximum across dimensions = [2] : (tensor<1x27x257xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2349 = stablehlo.reshape %2348 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2350 = stablehlo.broadcast_in_dim %2347, dims = [0, 1, 2] : (tensor<1x27x257xf32>) -> tensor<1x27x257xf32>
    %2351 = stablehlo.broadcast_in_dim %2349, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x257xf32>
    %2352 = stablehlo.subtract %2350, %2351 : tensor<1x27x257xf32>
    %2353 = stablehlo.exponential %2352 : tensor<1x27x257xf32>
    %2354 = stablehlo.reduce(%2353 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x27x257xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2355 = stablehlo.reshape %2354 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2356 = stablehlo.broadcast_in_dim %2353, dims = [0, 1, 2] : (tensor<1x27x257xf32>) -> tensor<1x27x257xf32>
    %2357 = stablehlo.broadcast_in_dim %2355, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x257xf32>
    %2358 = stablehlo.divide %2356, %2357 : tensor<1x27x257xf32>
    %2359 = stablehlo.convert %2358 : (tensor<1x27x257xf32>) -> tensor<1x27x257xbf16>
    %2360 = stablehlo.convolution(%2343, %arg63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 8 : i64} : (tensor<1x768x257x1xbf16>, tensor<768x96x1x1xbf16>) -> tensor<1x768x257x1xbf16>
    %2361 = stablehlo.reshape %2360 : (tensor<1x768x257x1xbf16>) -> tensor<1x768x257xbf16>
    %2362 = stablehlo.transpose %2361, dims = [0, 2, 1] : (tensor<1x768x257xbf16>) -> tensor<1x257x768xbf16>
    %2363 = stablehlo.reshape %2359 : (tensor<1x27x257xbf16>) -> tensor<1x27x257x1xbf16>
    %2364 = stablehlo.transpose %2363, dims = [0, 1, 3, 2] : (tensor<1x27x257x1xbf16>) -> tensor<1x27x1x257xbf16>
    %2365 = stablehlo.reshape %2362 : (tensor<1x257x768xbf16>) -> tensor<1x257x768x1xbf16>
    %2366 = stablehlo.transpose %2365, dims = [0, 3, 2, 1] : (tensor<1x257x768x1xbf16>) -> tensor<1x1x768x257xbf16>
    %2367 = stablehlo.transpose %2364, dims = [1, 3, 0, 2] : (tensor<1x27x1x257xbf16>) -> tensor<27x257x1x1xbf16>
    %2368 = stablehlo.reshape %2367 : (tensor<27x257x1x1xbf16>) -> tensor<1x27x257xbf16>
    %2369 = stablehlo.transpose %2366, dims = [3, 0, 2, 1] : (tensor<1x1x768x257xbf16>) -> tensor<257x1x768x1xbf16>
    %2370 = stablehlo.reshape %2369 : (tensor<257x1x768x1xbf16>) -> tensor<1x257x768xbf16>
    %2371 = stablehlo.broadcast_in_dim %2370, dims = [0, 1, 2] : (tensor<1x257x768xbf16>) -> tensor<1x257x768xbf16>
    %2372 = stablehlo.dot_general %2368, %2371, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x27x257xbf16>, tensor<1x257x768xbf16>) -> tensor<1x27x768xbf16>
    %2373 = stablehlo.reshape %2372 : (tensor<1x27x768xbf16>) -> tensor<27x1x1x768xbf16>
    %2374 = stablehlo.transpose %2373, dims = [2, 0, 3, 1] : (tensor<27x1x1x768xbf16>) -> tensor<1x27x768x1xbf16>
    %2375 = stablehlo.reshape %2374 : (tensor<1x27x768x1xbf16>) -> tensor<1x27x768xbf16>
    %2376 = stablehlo.convert %2375 : (tensor<1x27x768xbf16>) -> tensor<1x27x768xf32>
    %2377 = stablehlo.convert %2376 : (tensor<1x27x768xf32>) -> tensor<1x27x768xf64>
    %2378 = stablehlo.reduce(%2377 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf64>, tensor<f64>) -> tensor<1x27xf64>
    %2379 = stablehlo.reshape %2378 : (tensor<1x27xf64>) -> tensor<1x27x1xf64>
    %2380 = stablehlo.broadcast_in_dim %2379, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x1xf64>
    %2381 = stablehlo.divide %2380, %2299 : tensor<1x27x1xf64>
    %2382 = stablehlo.broadcast_in_dim %2377, dims = [0, 1, 2] : (tensor<1x27x768xf64>) -> tensor<1x27x768xf64>
    %2383 = stablehlo.broadcast_in_dim %2381, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x768xf64>
    %2384 = stablehlo.subtract %2382, %2383 : tensor<1x27x768xf64>
    %2385 = stablehlo.multiply %2384, %2384 : tensor<1x27x768xf64>
    %2386 = stablehlo.reduce(%2385 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf64>, tensor<f64>) -> tensor<1x27xf64>
    %2387 = stablehlo.reshape %2386 : (tensor<1x27xf64>) -> tensor<1x27x1xf64>
    %2388 = stablehlo.broadcast_in_dim %2387, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x1xf64>
    %2389 = stablehlo.divide %2388, %2299 : tensor<1x27x1xf64>
    %2390 = stablehlo.convert %2389 : (tensor<1x27x1xf64>) -> tensor<1x27x1xf32>
    %2391 = stablehlo.reduce(%2376 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2392 = stablehlo.reshape %2391 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2393 = stablehlo.broadcast_in_dim %2392, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x1xf32>
    %2394 = stablehlo.divide %2393, %2313 : tensor<1x27x1xf32>
    %2395 = stablehlo.broadcast_in_dim %2390, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x1xf32>
    %2396 = stablehlo.add %2395, %2316 : tensor<1x27x1xf32>
    %2397 = stablehlo.rsqrt %2396 : tensor<1x27x1xf32>
    %2398 = stablehlo.broadcast_in_dim %2376, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2399 = stablehlo.broadcast_in_dim %2394, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x768xf32>
    %2400 = stablehlo.subtract %2398, %2399 : tensor<1x27x768xf32>
    %2401 = stablehlo.broadcast_in_dim %2400, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2402 = stablehlo.broadcast_in_dim %2397, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x768xf32>
    %2403 = stablehlo.multiply %2401, %2402 : tensor<1x27x768xf32>
    %2404 = stablehlo.convert %arg64 : (tensor<768xbf16>) -> tensor<768xf32>
    %2405 = stablehlo.broadcast_in_dim %2403, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2406 = stablehlo.broadcast_in_dim %2404, dims = [2] : (tensor<768xf32>) -> tensor<1x27x768xf32>
    %2407 = stablehlo.multiply %2405, %2406 : tensor<1x27x768xf32>
    %2408 = stablehlo.convert %arg65 : (tensor<768xbf16>) -> tensor<768xf32>
    %2409 = stablehlo.broadcast_in_dim %2407, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2410 = stablehlo.broadcast_in_dim %2408, dims = [2] : (tensor<768xf32>) -> tensor<1x27x768xf32>
    %2411 = stablehlo.add %2409, %2410 : tensor<1x27x768xf32>
    %2412 = stablehlo.convert %2411 : (tensor<1x27x768xf32>) -> tensor<1x27x768xbf16>
    %2413 = stablehlo.convert %arg66 : (tensor<768xbf16>) -> tensor<768xf32>
    %2414 = stablehlo.broadcast_in_dim %2413, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2415 = stablehlo.multiply %2252, %2414 : tensor<1x257x768xf32>
    %2416 = stablehlo.convert %arg67 : (tensor<768xbf16>) -> tensor<768xf32>
    %2417 = stablehlo.broadcast_in_dim %2415, dims = [0, 1, 2] : (tensor<1x257x768xf32>) -> tensor<1x257x768xf32>
    %2418 = stablehlo.broadcast_in_dim %2416, dims = [2] : (tensor<768xf32>) -> tensor<1x257x768xf32>
    %2419 = stablehlo.add %2417, %2418 : tensor<1x257x768xf32>
    %2420 = stablehlo.convert %2419 : (tensor<1x257x768xf32>) -> tensor<1x257x768xbf16>
    %2421 = stablehlo.transpose %2420, dims = [0, 2, 1] : (tensor<1x257x768xbf16>) -> tensor<1x768x257xbf16>
    %2422 = stablehlo.reshape %2421 : (tensor<1x768x257xbf16>) -> tensor<1x768x257x1xbf16>
    %2423 = stablehlo.convolution(%2422, %arg68) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 8 : i64} : (tensor<1x768x257x1xbf16>, tensor<768x96x1x1xbf16>) -> tensor<1x768x257x1xbf16>
    %2424 = stablehlo.convolution(%2423, %arg69) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x768x257x1xbf16>, tensor<27x768x1x1xbf16>) -> tensor<1x27x257x1xbf16>
    %2425 = stablehlo.reshape %2424 : (tensor<1x27x257x1xbf16>) -> tensor<1x27x257xbf16>
    %2426 = stablehlo.convert %2425 : (tensor<1x27x257xbf16>) -> tensor<1x27x257xf32>
    %2427 = stablehlo.reduce(%2426 init: %cst_1) applies stablehlo.maximum across dimensions = [2] : (tensor<1x27x257xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2428 = stablehlo.reshape %2427 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2429 = stablehlo.broadcast_in_dim %2426, dims = [0, 1, 2] : (tensor<1x27x257xf32>) -> tensor<1x27x257xf32>
    %2430 = stablehlo.broadcast_in_dim %2428, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x257xf32>
    %2431 = stablehlo.subtract %2429, %2430 : tensor<1x27x257xf32>
    %2432 = stablehlo.exponential %2431 : tensor<1x27x257xf32>
    %2433 = stablehlo.reduce(%2432 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x27x257xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2434 = stablehlo.reshape %2433 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2435 = stablehlo.broadcast_in_dim %2432, dims = [0, 1, 2] : (tensor<1x27x257xf32>) -> tensor<1x27x257xf32>
    %2436 = stablehlo.broadcast_in_dim %2434, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x257xf32>
    %2437 = stablehlo.divide %2435, %2436 : tensor<1x27x257xf32>
    %2438 = stablehlo.convert %2437 : (tensor<1x27x257xf32>) -> tensor<1x27x257xbf16>
    %2439 = stablehlo.convolution(%2422, %arg70) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 8 : i64} : (tensor<1x768x257x1xbf16>, tensor<768x96x1x1xbf16>) -> tensor<1x768x257x1xbf16>
    %2440 = stablehlo.reshape %2439 : (tensor<1x768x257x1xbf16>) -> tensor<1x768x257xbf16>
    %2441 = stablehlo.transpose %2440, dims = [0, 2, 1] : (tensor<1x768x257xbf16>) -> tensor<1x257x768xbf16>
    %2442 = stablehlo.reshape %2438 : (tensor<1x27x257xbf16>) -> tensor<1x27x257x1xbf16>
    %2443 = stablehlo.transpose %2442, dims = [0, 1, 3, 2] : (tensor<1x27x257x1xbf16>) -> tensor<1x27x1x257xbf16>
    %2444 = stablehlo.reshape %2441 : (tensor<1x257x768xbf16>) -> tensor<1x257x768x1xbf16>
    %2445 = stablehlo.transpose %2444, dims = [0, 3, 2, 1] : (tensor<1x257x768x1xbf16>) -> tensor<1x1x768x257xbf16>
    %2446 = stablehlo.transpose %2443, dims = [1, 3, 0, 2] : (tensor<1x27x1x257xbf16>) -> tensor<27x257x1x1xbf16>
    %2447 = stablehlo.reshape %2446 : (tensor<27x257x1x1xbf16>) -> tensor<1x27x257xbf16>
    %2448 = stablehlo.transpose %2445, dims = [3, 0, 2, 1] : (tensor<1x1x768x257xbf16>) -> tensor<257x1x768x1xbf16>
    %2449 = stablehlo.reshape %2448 : (tensor<257x1x768x1xbf16>) -> tensor<1x257x768xbf16>
    %2450 = stablehlo.broadcast_in_dim %2449, dims = [0, 1, 2] : (tensor<1x257x768xbf16>) -> tensor<1x257x768xbf16>
    %2451 = stablehlo.dot_general %2447, %2450, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x27x257xbf16>, tensor<1x257x768xbf16>) -> tensor<1x27x768xbf16>
    %2452 = stablehlo.reshape %2451 : (tensor<1x27x768xbf16>) -> tensor<27x1x1x768xbf16>
    %2453 = stablehlo.transpose %2452, dims = [2, 0, 3, 1] : (tensor<27x1x1x768xbf16>) -> tensor<1x27x768x1xbf16>
    %2454 = stablehlo.reshape %2453 : (tensor<1x27x768x1xbf16>) -> tensor<1x27x768xbf16>
    %2455 = stablehlo.convert %2454 : (tensor<1x27x768xbf16>) -> tensor<1x27x768xf32>
    %2456 = stablehlo.convert %2455 : (tensor<1x27x768xf32>) -> tensor<1x27x768xf64>
    %2457 = stablehlo.reduce(%2456 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf64>, tensor<f64>) -> tensor<1x27xf64>
    %2458 = stablehlo.reshape %2457 : (tensor<1x27xf64>) -> tensor<1x27x1xf64>
    %2459 = stablehlo.broadcast_in_dim %2458, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x1xf64>
    %2460 = stablehlo.divide %2459, %2299 : tensor<1x27x1xf64>
    %2461 = stablehlo.broadcast_in_dim %2456, dims = [0, 1, 2] : (tensor<1x27x768xf64>) -> tensor<1x27x768xf64>
    %2462 = stablehlo.broadcast_in_dim %2460, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x768xf64>
    %2463 = stablehlo.subtract %2461, %2462 : tensor<1x27x768xf64>
    %2464 = stablehlo.multiply %2463, %2463 : tensor<1x27x768xf64>
    %2465 = stablehlo.reduce(%2464 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf64>, tensor<f64>) -> tensor<1x27xf64>
    %2466 = stablehlo.reshape %2465 : (tensor<1x27xf64>) -> tensor<1x27x1xf64>
    %2467 = stablehlo.broadcast_in_dim %2466, dims = [0, 1, 2] : (tensor<1x27x1xf64>) -> tensor<1x27x1xf64>
    %2468 = stablehlo.divide %2467, %2299 : tensor<1x27x1xf64>
    %2469 = stablehlo.convert %2468 : (tensor<1x27x1xf64>) -> tensor<1x27x1xf32>
    %2470 = stablehlo.reduce(%2455 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x27x768xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2471 = stablehlo.reshape %2470 : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2472 = stablehlo.broadcast_in_dim %2471, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x1xf32>
    %2473 = stablehlo.divide %2472, %2313 : tensor<1x27x1xf32>
    %2474 = stablehlo.broadcast_in_dim %2469, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x1xf32>
    %2475 = stablehlo.add %2474, %2316 : tensor<1x27x1xf32>
    %2476 = stablehlo.rsqrt %2475 : tensor<1x27x1xf32>
    %2477 = stablehlo.broadcast_in_dim %2455, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2478 = stablehlo.broadcast_in_dim %2473, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x768xf32>
    %2479 = stablehlo.subtract %2477, %2478 : tensor<1x27x768xf32>
    %2480 = stablehlo.broadcast_in_dim %2479, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2481 = stablehlo.broadcast_in_dim %2476, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x768xf32>
    %2482 = stablehlo.multiply %2480, %2481 : tensor<1x27x768xf32>
    %2483 = stablehlo.convert %arg71 : (tensor<768xbf16>) -> tensor<768xf32>
    %2484 = stablehlo.broadcast_in_dim %2482, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2485 = stablehlo.broadcast_in_dim %2483, dims = [2] : (tensor<768xf32>) -> tensor<1x27x768xf32>
    %2486 = stablehlo.multiply %2484, %2485 : tensor<1x27x768xf32>
    %2487 = stablehlo.convert %arg72 : (tensor<768xbf16>) -> tensor<768xf32>
    %2488 = stablehlo.broadcast_in_dim %2486, dims = [0, 1, 2] : (tensor<1x27x768xf32>) -> tensor<1x27x768xf32>
    %2489 = stablehlo.broadcast_in_dim %2487, dims = [2] : (tensor<768xf32>) -> tensor<1x27x768xf32>
    %2490 = stablehlo.add %2488, %2489 : tensor<1x27x768xf32>
    %2491 = stablehlo.convert %2490 : (tensor<1x27x768xf32>) -> tensor<1x27x768xbf16>
    %2492 = stablehlo.reshape %2333 : (tensor<1x27x768xbf16>) -> tensor<27x768xbf16>
    %2493 = stablehlo.convert %2492 : (tensor<27x768xbf16>) -> tensor<27x768xf32>
    %2494 = stablehlo.dot_general %2493, %arg170, contracting_dims = [1] x [0] : (tensor<27x768xf32>, tensor<768x38xf32>) -> tensor<27x38xf32>
    %2495 = stablehlo.broadcast_in_dim %2494, dims = [0, 1] : (tensor<27x38xf32>) -> tensor<27x38xf32>
    %2496 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<27x38xf32>
    %2497 = stablehlo.multiply %2495, %2496 : tensor<27x38xf32>
    %2498 = stablehlo.broadcast_in_dim %2497, dims = [0, 1] : (tensor<27x38xf32>) -> tensor<27x38xf32>
    %2499 = stablehlo.broadcast_in_dim %arg171, dims = [1] : (tensor<38xf32>) -> tensor<27x38xf32>
    %2500 = stablehlo.add %2498, %2499 : tensor<27x38xf32>
    %2501 = stablehlo.convert %2500 : (tensor<27x38xf32>) -> tensor<27x38xbf16>
    %2502 = stablehlo.reshape %2501 : (tensor<27x38xbf16>) -> tensor<1x27x38xbf16>
    %2503 = stablehlo.reshape %2412 : (tensor<1x27x768xbf16>) -> tensor<27x768xbf16>
    %2504 = stablehlo.convert %2503 : (tensor<27x768xbf16>) -> tensor<27x768xf32>
    %2505 = stablehlo.dot_general %2504, %arg172, contracting_dims = [1] x [0] : (tensor<27x768xf32>, tensor<768x50257xf32>) -> tensor<27x50257xf32>
    %2506 = stablehlo.broadcast_in_dim %2505, dims = [0, 1] : (tensor<27x50257xf32>) -> tensor<27x50257xf32>
    %2507 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<27x50257xf32>
    %2508 = stablehlo.multiply %2506, %2507 : tensor<27x50257xf32>
    %2509 = stablehlo.broadcast_in_dim %2508, dims = [0, 1] : (tensor<27x50257xf32>) -> tensor<27x50257xf32>
    %2510 = stablehlo.broadcast_in_dim %arg173, dims = [1] : (tensor<50257xf32>) -> tensor<27x50257xf32>
    %2511 = stablehlo.add %2509, %2510 : tensor<27x50257xf32>
    %2512 = stablehlo.convert %2511 : (tensor<27x50257xf32>) -> tensor<27x50257xbf16>
    %2513 = stablehlo.reshape %2512 : (tensor<27x50257xbf16>) -> tensor<1x27x50257xbf16>
    %2514 = stablehlo.reshape %2491 : (tensor<1x27x768xbf16>) -> tensor<27x768xbf16>
    %2515 = stablehlo.convert %2514 : (tensor<27x768xbf16>) -> tensor<27x768xf32>
    %2516 = stablehlo.dot_general %2515, %arg174, contracting_dims = [1] x [0] : (tensor<27x768xf32>, tensor<768x30522xf32>) -> tensor<27x30522xf32>
    %2517 = stablehlo.broadcast_in_dim %2516, dims = [0, 1] : (tensor<27x30522xf32>) -> tensor<27x30522xf32>
    %2518 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<27x30522xf32>
    %2519 = stablehlo.multiply %2517, %2518 : tensor<27x30522xf32>
    %2520 = stablehlo.broadcast_in_dim %2519, dims = [0, 1] : (tensor<27x30522xf32>) -> tensor<27x30522xf32>
    %2521 = stablehlo.broadcast_in_dim %arg175, dims = [1] : (tensor<30522xf32>) -> tensor<27x30522xf32>
    %2522 = stablehlo.add %2520, %2521 : tensor<27x30522xf32>
    %2523 = stablehlo.convert %2522 : (tensor<27x30522xf32>) -> tensor<27x30522xbf16>
    %2524 = stablehlo.reshape %2523 : (tensor<27x30522xbf16>) -> tensor<1x27x30522xbf16>
    return %2502, %2513, %2524 : tensor<1x27x38xbf16>, tensor<1x27x50257xbf16>, tensor<1x27x30522xbf16>
  }
}
