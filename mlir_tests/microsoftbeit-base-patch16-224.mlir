module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<768x3x16x16xbf16>, %arg2: tensor<768xbf16>, %arg3: tensor<768xbf16>, %arg4: tensor<768xbf16>, %arg5: tensor<768xbf16>, %arg6: tensor<768xbf16>, %arg7: tensor<768xbf16>, %arg8: tensor<768xbf16>, %arg9: tensor<768xbf16>, %arg10: tensor<768xbf16>, %arg11: tensor<768xbf16>, %arg12: tensor<768xbf16>, %arg13: tensor<768xbf16>, %arg14: tensor<768xbf16>, %arg15: tensor<768xbf16>, %arg16: tensor<768xbf16>, %arg17: tensor<768xbf16>, %arg18: tensor<768xbf16>, %arg19: tensor<768xbf16>, %arg20: tensor<768xbf16>, %arg21: tensor<768xbf16>, %arg22: tensor<768xbf16>, %arg23: tensor<768xbf16>, %arg24: tensor<768xbf16>, %arg25: tensor<768xbf16>, %arg26: tensor<768xbf16>, %arg27: tensor<768xbf16>, %arg28: tensor<768xbf16>, %arg29: tensor<768xbf16>, %arg30: tensor<768xbf16>, %arg31: tensor<768xbf16>, %arg32: tensor<768xbf16>, %arg33: tensor<768xbf16>, %arg34: tensor<768xbf16>, %arg35: tensor<768xbf16>, %arg36: tensor<768xbf16>, %arg37: tensor<768xbf16>, %arg38: tensor<768xbf16>, %arg39: tensor<768xbf16>, %arg40: tensor<768xbf16>, %arg41: tensor<768xbf16>, %arg42: tensor<768xbf16>, %arg43: tensor<768xbf16>, %arg44: tensor<768xbf16>, %arg45: tensor<768xbf16>, %arg46: tensor<768xbf16>, %arg47: tensor<768xbf16>, %arg48: tensor<768xbf16>, %arg49: tensor<768xbf16>, %arg50: tensor<768xbf16>, %arg51: tensor<768xbf16>, %arg52: tensor<768xbf16>, %arg53: tensor<768xbf16>, %arg54: tensor<768xbf16>, %arg55: tensor<768xbf16>, %arg56: tensor<768xbf16>, %arg57: tensor<768xbf16>, %arg58: tensor<768xbf16>, %arg59: tensor<768xbf16>, %arg60: tensor<768xbf16>, %arg61: tensor<768xbf16>, %arg62: tensor<768xbf16>, %arg63: tensor<768xbf16>, %arg64: tensor<768xbf16>, %arg65: tensor<768xbf16>, %arg66: tensor<768xbf16>, %arg67: tensor<768xbf16>, %arg68: tensor<768xbf16>, %arg69: tensor<768xbf16>, %arg70: tensor<768xbf16>, %arg71: tensor<768xbf16>, %arg72: tensor<768xbf16>, %arg73: tensor<768xbf16>, %arg74: tensor<768xbf16>, %arg75: tensor<768xbf16>, %arg76: tensor<768xbf16>, %arg77: tensor<1x1x768xbf16>, %arg78: tensor<768x768xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768x768xbf16>, %arg81: tensor<768x768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<1x12x197x197xbf16>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768x3072xf32>, %arg87: tensor<3072xf32>, %arg88: tensor<3072x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x768xf32>, %arg91: tensor<768xf32>, %arg92: tensor<768x768xbf16>, %arg93: tensor<768x768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<1x12x197x197xbf16>, %arg96: tensor<768x768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768x3072xf32>, %arg99: tensor<3072xf32>, %arg100: tensor<3072x768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768x768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xbf16>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<1x12x197x197xbf16>, %arg108: tensor<768x768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<3072xf32>, %arg112: tensor<3072x768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768x768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768x768xbf16>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<1x12x197x197xbf16>, %arg120: tensor<768x768xf32>, %arg121: tensor<768xf32>, %arg122: tensor<768x3072xf32>, %arg123: tensor<3072xf32>, %arg124: tensor<3072x768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768x768xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768x768xbf16>, %arg129: tensor<768x768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<1x12x197x197xbf16>, %arg132: tensor<768x768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768x3072xf32>, %arg135: tensor<3072xf32>, %arg136: tensor<3072x768xf32>, %arg137: tensor<768xf32>, %arg138: tensor<768x768xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768x768xbf16>, %arg141: tensor<768x768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<1x12x197x197xbf16>, %arg144: tensor<768x768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768x3072xf32>, %arg147: tensor<3072xf32>, %arg148: tensor<3072x768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768x768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768x768xbf16>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<1x12x197x197xbf16>, %arg156: tensor<768x768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768x3072xf32>, %arg159: tensor<3072xf32>, %arg160: tensor<3072x768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768x768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768x768xbf16>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<1x12x197x197xbf16>, %arg168: tensor<768x768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768x3072xf32>, %arg171: tensor<3072xf32>, %arg172: tensor<3072x768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768x768xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768x768xbf16>, %arg177: tensor<768x768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<1x12x197x197xbf16>, %arg180: tensor<768x768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768x3072xf32>, %arg183: tensor<3072xf32>, %arg184: tensor<3072x768xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x768xf32>, %arg187: tensor<768xf32>, %arg188: tensor<768x768xbf16>, %arg189: tensor<768x768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<1x12x197x197xbf16>, %arg192: tensor<768x768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768x3072xf32>, %arg195: tensor<3072xf32>, %arg196: tensor<3072x768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<768x768xf32>, %arg199: tensor<768xf32>, %arg200: tensor<768x768xbf16>, %arg201: tensor<768x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x12x197x197xbf16>, %arg204: tensor<768x768xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768x3072xf32>, %arg207: tensor<3072xf32>, %arg208: tensor<3072x768xf32>, %arg209: tensor<768xf32>, %arg210: tensor<768x768xf32>, %arg211: tensor<768xf32>, %arg212: tensor<768x768xbf16>, %arg213: tensor<768x768xf32>, %arg214: tensor<768xf32>, %arg215: tensor<1x12x197x197xbf16>, %arg216: tensor<768x768xf32>, %arg217: tensor<768xf32>, %arg218: tensor<768x3072xf32>, %arg219: tensor<3072xf32>, %arg220: tensor<3072x768xf32>, %arg221: tensor<768xf32>, %arg222: tensor<768x1000xf32>, %arg223: tensor<1000xf32>) -> tensor<1x1000xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<1x197x3072xbf16>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<1x197x3072xbf16>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<1x197x3072xbf16>
    %cst_5 = stablehlo.constant dense<-4.000000e+00> : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<2.77068146E-8> : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x197x3072xf32>
    %cst_10 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x197x3072xf32>
    %cst_12 = stablehlo.constant dense<-2.954600e-03> : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.0160960332> : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<-0.00168282702> : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.00737332925> : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<-0.0142647391> : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-1.000000e+00> : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_22 = arith.constant dense<768> : tensor<1xi64>
    %cst_23 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
    %cst_24 = arith.constant dense<1> : tensor<1xi64>
    %cst_25 = arith.constant dense<8.000000e+00> : tensor<1xf64>
    %cst_26 = arith.constant dense<196> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [16, 16], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xbf16>, tensor<768x3x16x16xbf16>) -> tensor<1x768x14x14xbf16>
    %1 = stablehlo.reshape %arg2 : (tensor<768xbf16>) -> tensor<768x1x1xbf16>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x768x14x14xbf16>) -> tensor<1x768x14x14xbf16>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<768x1x1xbf16>) -> tensor<1x768x14x14xbf16>
    %4 = stablehlo.add %2, %3 : tensor<1x768x14x14xbf16>
    %5 = stablehlo.reshape %4 : (tensor<1x768x14x14xbf16>) -> tensor<1x768x196xbf16>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1] : (tensor<1x768x196xbf16>) -> tensor<1x196x768xbf16>
    %7 = stablehlo.concatenate %arg77, %6, dim = 1 : (tensor<1x1x768xbf16>, tensor<1x196x768xbf16>) -> tensor<1x197x768xbf16>
    %8 = stablehlo.convert %7 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %9 = stablehlo.convert %8 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %11 = stablehlo.reshape %10 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %12 = stablehlo.convert %cst_22 : (tensor<1xi64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %15 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<1x197x1xf64>
    %16 = stablehlo.divide %14, %15 : tensor<1x197x1xf64>
    %17 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %19 = stablehlo.subtract %17, %18 : tensor<1x197x768xf64>
    %20 = stablehlo.multiply %19, %19 : tensor<1x197x768xf64>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %22 = stablehlo.reshape %21 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %24 = stablehlo.divide %23, %15 : tensor<1x197x1xf64>
    %25 = stablehlo.convert %24 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %26 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %27 = stablehlo.reshape %26 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %28 = stablehlo.convert %cst_22 : (tensor<1xi64>) -> tensor<1xf32>
    %29 = stablehlo.reshape %28 : (tensor<1xf32>) -> tensor<f32>
    %30 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %31 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<1x197x1xf32>
    %32 = stablehlo.divide %30, %31 : tensor<1x197x1xf32>
    %33 = stablehlo.convert %cst_23 : (tensor<1xf64>) -> tensor<1xf32>
    %34 = stablehlo.reshape %33 : (tensor<1xf32>) -> tensor<f32>
    %35 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %36 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<1x197x1xf32>
    %37 = stablehlo.add %35, %36 : tensor<1x197x1xf32>
    %38 = stablehlo.rsqrt %37 : tensor<1x197x1xf32>
    %39 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %40 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %41 = stablehlo.subtract %39, %40 : tensor<1x197x768xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %43 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<1x197x768xf32>
    %45 = stablehlo.convert %arg3 : (tensor<768xbf16>) -> tensor<768xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %48 = stablehlo.multiply %46, %47 : tensor<1x197x768xf32>
    %49 = stablehlo.convert %arg4 : (tensor<768xbf16>) -> tensor<768xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %51 = stablehlo.broadcast_in_dim %49, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x197x768xf32>
    %53 = stablehlo.convert %52 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %54 = stablehlo.reshape %53 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %55 = stablehlo.convert %54 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %56 = stablehlo.dot_general %55, %arg78, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %57 = stablehlo.convert %cst_24 : (tensor<1xi64>) -> tensor<1xf32>
    %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
    %59 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %60 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %61 = stablehlo.multiply %59, %60 : tensor<197x768xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %63 = stablehlo.broadcast_in_dim %arg79, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %64 = stablehlo.add %62, %63 : tensor<197x768xf32>
    %65 = stablehlo.convert %64 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %66 = stablehlo.reshape %65 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %67 = stablehlo.dot_general %54, %arg80, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %68 = stablehlo.reshape %67 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %69 = stablehlo.reshape %68 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %70 = stablehlo.transpose %69, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %71 = stablehlo.dot_general %55, %arg81, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %73 = stablehlo.multiply %72, %60 : tensor<197x768xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %75 = stablehlo.broadcast_in_dim %arg82, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<197x768xf32>
    %77 = stablehlo.convert %76 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %78 = stablehlo.reshape %77 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %79 = stablehlo.reshape %78 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %80 = stablehlo.transpose %79, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %81 = stablehlo.reshape %66 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %82 = stablehlo.transpose %81, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %83 = stablehlo.transpose %70, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %84 = stablehlo.reshape %82 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %85 = stablehlo.reshape %83 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %87 = stablehlo.dot_general %84, %86, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %88 = stablehlo.reshape %87 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %89 = stablehlo.convert %cst_25 : (tensor<1xf64>) -> tensor<1xbf16>
    %90 = stablehlo.reshape %89 : (tensor<1xbf16>) -> tensor<bf16>
    %91 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %92 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<bf16>) -> tensor<1x12x197x197xbf16>
    %93 = stablehlo.divide %91, %92 : tensor<1x12x197x197xbf16>
    %94 = stablehlo.add %93, %arg83 : tensor<1x12x197x197xbf16>
    %95 = stablehlo.convert %94 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %96 = stablehlo.reduce(%95 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %97 = stablehlo.reshape %96 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %98 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %99 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %100 = stablehlo.subtract %98, %99 : tensor<1x12x197x197xf32>
    %101 = stablehlo.exponential %100 : tensor<1x12x197x197xf32>
    %102 = stablehlo.reduce(%101 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %103 = stablehlo.reshape %102 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %104 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %105 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %106 = stablehlo.divide %104, %105 : tensor<1x12x197x197xf32>
    %107 = stablehlo.convert %106 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %108 = stablehlo.reshape %107 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %109 = stablehlo.reshape %80 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %111 = stablehlo.dot_general %108, %110, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %112 = stablehlo.reshape %111 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %113 = stablehlo.transpose %112, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %114 = stablehlo.reshape %113 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %115 = stablehlo.reshape %114 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %116 = stablehlo.convert %115 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %117 = stablehlo.dot_general %116, %arg84, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %119 = stablehlo.multiply %118, %60 : tensor<197x768xf32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %121 = stablehlo.broadcast_in_dim %arg85, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %122 = stablehlo.add %120, %121 : tensor<197x768xf32>
    %123 = stablehlo.convert %122 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %124 = stablehlo.reshape %123 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %125 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %126 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %127 = stablehlo.multiply %125, %126 : tensor<1x197x768xbf16>
    %128 = stablehlo.add %127, %7 : tensor<1x197x768xbf16>
    %129 = stablehlo.convert %128 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %130 = stablehlo.convert %129 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %131 = stablehlo.reduce(%130 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %132 = stablehlo.reshape %131 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %134 = stablehlo.divide %133, %15 : tensor<1x197x1xf64>
    %135 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %136 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %137 = stablehlo.subtract %135, %136 : tensor<1x197x768xf64>
    %138 = stablehlo.multiply %137, %137 : tensor<1x197x768xf64>
    %139 = stablehlo.reduce(%138 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %140 = stablehlo.reshape %139 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %141 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %142 = stablehlo.divide %141, %15 : tensor<1x197x1xf64>
    %143 = stablehlo.convert %142 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %144 = stablehlo.reduce(%129 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %145 = stablehlo.reshape %144 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %147 = stablehlo.divide %146, %31 : tensor<1x197x1xf32>
    %148 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %149 = stablehlo.add %148, %36 : tensor<1x197x1xf32>
    %150 = stablehlo.rsqrt %149 : tensor<1x197x1xf32>
    %151 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %152 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %153 = stablehlo.subtract %151, %152 : tensor<1x197x768xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %155 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %156 = stablehlo.multiply %154, %155 : tensor<1x197x768xf32>
    %157 = stablehlo.convert %arg6 : (tensor<768xbf16>) -> tensor<768xf32>
    %158 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %159 = stablehlo.broadcast_in_dim %157, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %160 = stablehlo.multiply %158, %159 : tensor<1x197x768xf32>
    %161 = stablehlo.convert %arg7 : (tensor<768xbf16>) -> tensor<768xf32>
    %162 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %163 = stablehlo.broadcast_in_dim %161, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %164 = stablehlo.add %162, %163 : tensor<1x197x768xf32>
    %165 = stablehlo.convert %164 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %166 = stablehlo.reshape %165 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %167 = stablehlo.convert %166 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %168 = stablehlo.dot_general %167, %arg86, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %170 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<197x3072xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %173 = stablehlo.broadcast_in_dim %arg87, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %174 = stablehlo.add %172, %173 : tensor<197x3072xf32>
    %175 = stablehlo.convert %174 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %176 = stablehlo.reshape %175 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %177 = stablehlo.multiply %176, %cst_4 : tensor<1x197x3072xbf16>
    %178 = stablehlo.rsqrt %cst_3 : tensor<1x197x3072xbf16>
    %179 = stablehlo.multiply %176, %178 : tensor<1x197x3072xbf16>
    %180 = stablehlo.convert %179 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %181 = stablehlo.clamp %cst_5, %180, %cst_6 : tensor<1x197x3072xf32>
    %182 = stablehlo.multiply %181, %181 : tensor<1x197x3072xf32>
    %183 = stablehlo.multiply %cst_7, %182 : tensor<1x197x3072xf32>
    %184 = stablehlo.add %183, %cst_8 : tensor<1x197x3072xf32>
    %185 = stablehlo.multiply %184, %182 : tensor<1x197x3072xf32>
    %186 = stablehlo.add %185, %cst_9 : tensor<1x197x3072xf32>
    %187 = stablehlo.multiply %186, %182 : tensor<1x197x3072xf32>
    %188 = stablehlo.add %187, %cst_10 : tensor<1x197x3072xf32>
    %189 = stablehlo.multiply %188, %182 : tensor<1x197x3072xf32>
    %190 = stablehlo.add %189, %cst_11 : tensor<1x197x3072xf32>
    %191 = stablehlo.multiply %190, %182 : tensor<1x197x3072xf32>
    %192 = stablehlo.add %191, %cst_12 : tensor<1x197x3072xf32>
    %193 = stablehlo.multiply %192, %182 : tensor<1x197x3072xf32>
    %194 = stablehlo.add %193, %cst_13 : tensor<1x197x3072xf32>
    %195 = stablehlo.multiply %cst_14, %182 : tensor<1x197x3072xf32>
    %196 = stablehlo.add %195, %cst_15 : tensor<1x197x3072xf32>
    %197 = stablehlo.multiply %196, %182 : tensor<1x197x3072xf32>
    %198 = stablehlo.add %197, %cst_16 : tensor<1x197x3072xf32>
    %199 = stablehlo.multiply %198, %182 : tensor<1x197x3072xf32>
    %200 = stablehlo.add %199, %cst_17 : tensor<1x197x3072xf32>
    %201 = stablehlo.multiply %200, %182 : tensor<1x197x3072xf32>
    %202 = stablehlo.add %201, %cst_18 : tensor<1x197x3072xf32>
    %203 = stablehlo.multiply %181, %194 : tensor<1x197x3072xf32>
    %204 = stablehlo.divide %203, %202 : tensor<1x197x3072xf32>
    %205 = stablehlo.clamp %cst_19, %204, %cst_20 : tensor<1x197x3072xf32>
    %206 = stablehlo.convert %205 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %207 = stablehlo.add %206, %cst_2 : tensor<1x197x3072xbf16>
    %208 = stablehlo.multiply %207, %177 : tensor<1x197x3072xbf16>
    %209 = stablehlo.reshape %208 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %210 = stablehlo.convert %209 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %211 = stablehlo.dot_general %210, %arg88, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %213 = stablehlo.multiply %212, %60 : tensor<197x768xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %215 = stablehlo.broadcast_in_dim %arg89, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %216 = stablehlo.add %214, %215 : tensor<197x768xf32>
    %217 = stablehlo.convert %216 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %218 = stablehlo.reshape %217 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %219 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %220 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %221 = stablehlo.multiply %219, %220 : tensor<1x197x768xbf16>
    %222 = stablehlo.add %221, %128 : tensor<1x197x768xbf16>
    %223 = stablehlo.convert %222 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %224 = stablehlo.convert %223 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %225 = stablehlo.reduce(%224 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %226 = stablehlo.reshape %225 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %228 = stablehlo.divide %227, %15 : tensor<1x197x1xf64>
    %229 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %230 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %231 = stablehlo.subtract %229, %230 : tensor<1x197x768xf64>
    %232 = stablehlo.multiply %231, %231 : tensor<1x197x768xf64>
    %233 = stablehlo.reduce(%232 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %234 = stablehlo.reshape %233 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %236 = stablehlo.divide %235, %15 : tensor<1x197x1xf64>
    %237 = stablehlo.convert %236 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %238 = stablehlo.reduce(%223 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %239 = stablehlo.reshape %238 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %241 = stablehlo.divide %240, %31 : tensor<1x197x1xf32>
    %242 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %243 = stablehlo.add %242, %36 : tensor<1x197x1xf32>
    %244 = stablehlo.rsqrt %243 : tensor<1x197x1xf32>
    %245 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %246 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %247 = stablehlo.subtract %245, %246 : tensor<1x197x768xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %249 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %250 = stablehlo.multiply %248, %249 : tensor<1x197x768xf32>
    %251 = stablehlo.convert %arg9 : (tensor<768xbf16>) -> tensor<768xf32>
    %252 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %253 = stablehlo.broadcast_in_dim %251, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %254 = stablehlo.multiply %252, %253 : tensor<1x197x768xf32>
    %255 = stablehlo.convert %arg10 : (tensor<768xbf16>) -> tensor<768xf32>
    %256 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %257 = stablehlo.broadcast_in_dim %255, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %258 = stablehlo.add %256, %257 : tensor<1x197x768xf32>
    %259 = stablehlo.convert %258 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %260 = stablehlo.reshape %259 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %261 = stablehlo.convert %260 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %262 = stablehlo.dot_general %261, %arg90, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %263 = stablehlo.broadcast_in_dim %262, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %264 = stablehlo.multiply %263, %60 : tensor<197x768xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %266 = stablehlo.broadcast_in_dim %arg91, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %267 = stablehlo.add %265, %266 : tensor<197x768xf32>
    %268 = stablehlo.convert %267 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %269 = stablehlo.reshape %268 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %270 = stablehlo.dot_general %260, %arg92, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %271 = stablehlo.reshape %270 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %272 = stablehlo.reshape %271 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %273 = stablehlo.transpose %272, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %274 = stablehlo.dot_general %261, %arg93, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %276 = stablehlo.multiply %275, %60 : tensor<197x768xf32>
    %277 = stablehlo.broadcast_in_dim %276, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %278 = stablehlo.broadcast_in_dim %arg94, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %279 = stablehlo.add %277, %278 : tensor<197x768xf32>
    %280 = stablehlo.convert %279 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %281 = stablehlo.reshape %280 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %282 = stablehlo.reshape %281 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %283 = stablehlo.transpose %282, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %284 = stablehlo.reshape %269 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %285 = stablehlo.transpose %284, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %286 = stablehlo.transpose %273, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %287 = stablehlo.reshape %285 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %288 = stablehlo.reshape %286 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %290 = stablehlo.dot_general %287, %289, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %291 = stablehlo.reshape %290 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %292 = stablehlo.broadcast_in_dim %291, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %293 = stablehlo.divide %292, %92 : tensor<1x12x197x197xbf16>
    %294 = stablehlo.add %293, %arg95 : tensor<1x12x197x197xbf16>
    %295 = stablehlo.convert %294 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %296 = stablehlo.reduce(%295 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %297 = stablehlo.reshape %296 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %298 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %299 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %300 = stablehlo.subtract %298, %299 : tensor<1x12x197x197xf32>
    %301 = stablehlo.exponential %300 : tensor<1x12x197x197xf32>
    %302 = stablehlo.reduce(%301 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %303 = stablehlo.reshape %302 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %304 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %305 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %306 = stablehlo.divide %304, %305 : tensor<1x12x197x197xf32>
    %307 = stablehlo.convert %306 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %308 = stablehlo.reshape %307 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %309 = stablehlo.reshape %283 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %311 = stablehlo.dot_general %308, %310, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %312 = stablehlo.reshape %311 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %313 = stablehlo.transpose %312, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %314 = stablehlo.reshape %313 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %315 = stablehlo.reshape %314 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %316 = stablehlo.convert %315 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %317 = stablehlo.dot_general %316, %arg96, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %319 = stablehlo.multiply %318, %60 : tensor<197x768xf32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %321 = stablehlo.broadcast_in_dim %arg97, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %322 = stablehlo.add %320, %321 : tensor<197x768xf32>
    %323 = stablehlo.convert %322 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %324 = stablehlo.reshape %323 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %325 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %326 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %327 = stablehlo.multiply %325, %326 : tensor<1x197x768xbf16>
    %328 = stablehlo.add %327, %222 : tensor<1x197x768xbf16>
    %329 = stablehlo.convert %328 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %330 = stablehlo.convert %329 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %331 = stablehlo.reduce(%330 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %332 = stablehlo.reshape %331 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %334 = stablehlo.divide %333, %15 : tensor<1x197x1xf64>
    %335 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %336 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %337 = stablehlo.subtract %335, %336 : tensor<1x197x768xf64>
    %338 = stablehlo.multiply %337, %337 : tensor<1x197x768xf64>
    %339 = stablehlo.reduce(%338 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %340 = stablehlo.reshape %339 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %342 = stablehlo.divide %341, %15 : tensor<1x197x1xf64>
    %343 = stablehlo.convert %342 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %344 = stablehlo.reduce(%329 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %345 = stablehlo.reshape %344 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %347 = stablehlo.divide %346, %31 : tensor<1x197x1xf32>
    %348 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %349 = stablehlo.add %348, %36 : tensor<1x197x1xf32>
    %350 = stablehlo.rsqrt %349 : tensor<1x197x1xf32>
    %351 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %352 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %353 = stablehlo.subtract %351, %352 : tensor<1x197x768xf32>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %355 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %356 = stablehlo.multiply %354, %355 : tensor<1x197x768xf32>
    %357 = stablehlo.convert %arg12 : (tensor<768xbf16>) -> tensor<768xf32>
    %358 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %359 = stablehlo.broadcast_in_dim %357, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<1x197x768xf32>
    %361 = stablehlo.convert %arg13 : (tensor<768xbf16>) -> tensor<768xf32>
    %362 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %363 = stablehlo.broadcast_in_dim %361, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %364 = stablehlo.add %362, %363 : tensor<1x197x768xf32>
    %365 = stablehlo.convert %364 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %366 = stablehlo.reshape %365 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %367 = stablehlo.convert %366 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %368 = stablehlo.dot_general %367, %arg98, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %370 = stablehlo.multiply %369, %170 : tensor<197x3072xf32>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %372 = stablehlo.broadcast_in_dim %arg99, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %373 = stablehlo.add %371, %372 : tensor<197x3072xf32>
    %374 = stablehlo.convert %373 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %375 = stablehlo.reshape %374 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %376 = stablehlo.multiply %375, %cst_4 : tensor<1x197x3072xbf16>
    %377 = stablehlo.multiply %375, %178 : tensor<1x197x3072xbf16>
    %378 = stablehlo.convert %377 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %379 = stablehlo.clamp %cst_5, %378, %cst_6 : tensor<1x197x3072xf32>
    %380 = stablehlo.multiply %379, %379 : tensor<1x197x3072xf32>
    %381 = stablehlo.multiply %cst_7, %380 : tensor<1x197x3072xf32>
    %382 = stablehlo.add %381, %cst_8 : tensor<1x197x3072xf32>
    %383 = stablehlo.multiply %382, %380 : tensor<1x197x3072xf32>
    %384 = stablehlo.add %383, %cst_9 : tensor<1x197x3072xf32>
    %385 = stablehlo.multiply %384, %380 : tensor<1x197x3072xf32>
    %386 = stablehlo.add %385, %cst_10 : tensor<1x197x3072xf32>
    %387 = stablehlo.multiply %386, %380 : tensor<1x197x3072xf32>
    %388 = stablehlo.add %387, %cst_11 : tensor<1x197x3072xf32>
    %389 = stablehlo.multiply %388, %380 : tensor<1x197x3072xf32>
    %390 = stablehlo.add %389, %cst_12 : tensor<1x197x3072xf32>
    %391 = stablehlo.multiply %390, %380 : tensor<1x197x3072xf32>
    %392 = stablehlo.add %391, %cst_13 : tensor<1x197x3072xf32>
    %393 = stablehlo.multiply %cst_14, %380 : tensor<1x197x3072xf32>
    %394 = stablehlo.add %393, %cst_15 : tensor<1x197x3072xf32>
    %395 = stablehlo.multiply %394, %380 : tensor<1x197x3072xf32>
    %396 = stablehlo.add %395, %cst_16 : tensor<1x197x3072xf32>
    %397 = stablehlo.multiply %396, %380 : tensor<1x197x3072xf32>
    %398 = stablehlo.add %397, %cst_17 : tensor<1x197x3072xf32>
    %399 = stablehlo.multiply %398, %380 : tensor<1x197x3072xf32>
    %400 = stablehlo.add %399, %cst_18 : tensor<1x197x3072xf32>
    %401 = stablehlo.multiply %379, %392 : tensor<1x197x3072xf32>
    %402 = stablehlo.divide %401, %400 : tensor<1x197x3072xf32>
    %403 = stablehlo.clamp %cst_19, %402, %cst_20 : tensor<1x197x3072xf32>
    %404 = stablehlo.convert %403 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %405 = stablehlo.add %404, %cst_2 : tensor<1x197x3072xbf16>
    %406 = stablehlo.multiply %405, %376 : tensor<1x197x3072xbf16>
    %407 = stablehlo.reshape %406 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %408 = stablehlo.convert %407 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %409 = stablehlo.dot_general %408, %arg100, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %410 = stablehlo.broadcast_in_dim %409, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %411 = stablehlo.multiply %410, %60 : tensor<197x768xf32>
    %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %413 = stablehlo.broadcast_in_dim %arg101, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %414 = stablehlo.add %412, %413 : tensor<197x768xf32>
    %415 = stablehlo.convert %414 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %416 = stablehlo.reshape %415 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %417 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %418 = stablehlo.broadcast_in_dim %416, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %419 = stablehlo.multiply %417, %418 : tensor<1x197x768xbf16>
    %420 = stablehlo.add %419, %328 : tensor<1x197x768xbf16>
    %421 = stablehlo.convert %420 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %422 = stablehlo.convert %421 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %423 = stablehlo.reduce(%422 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %424 = stablehlo.reshape %423 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %426 = stablehlo.divide %425, %15 : tensor<1x197x1xf64>
    %427 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %428 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %429 = stablehlo.subtract %427, %428 : tensor<1x197x768xf64>
    %430 = stablehlo.multiply %429, %429 : tensor<1x197x768xf64>
    %431 = stablehlo.reduce(%430 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %432 = stablehlo.reshape %431 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %433 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %434 = stablehlo.divide %433, %15 : tensor<1x197x1xf64>
    %435 = stablehlo.convert %434 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %436 = stablehlo.reduce(%421 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %437 = stablehlo.reshape %436 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %439 = stablehlo.divide %438, %31 : tensor<1x197x1xf32>
    %440 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %441 = stablehlo.add %440, %36 : tensor<1x197x1xf32>
    %442 = stablehlo.rsqrt %441 : tensor<1x197x1xf32>
    %443 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %444 = stablehlo.broadcast_in_dim %439, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %445 = stablehlo.subtract %443, %444 : tensor<1x197x768xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %447 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %448 = stablehlo.multiply %446, %447 : tensor<1x197x768xf32>
    %449 = stablehlo.convert %arg15 : (tensor<768xbf16>) -> tensor<768xf32>
    %450 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %451 = stablehlo.broadcast_in_dim %449, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %452 = stablehlo.multiply %450, %451 : tensor<1x197x768xf32>
    %453 = stablehlo.convert %arg16 : (tensor<768xbf16>) -> tensor<768xf32>
    %454 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %455 = stablehlo.broadcast_in_dim %453, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %456 = stablehlo.add %454, %455 : tensor<1x197x768xf32>
    %457 = stablehlo.convert %456 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %458 = stablehlo.reshape %457 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %459 = stablehlo.convert %458 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %460 = stablehlo.dot_general %459, %arg102, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %462 = stablehlo.multiply %461, %60 : tensor<197x768xf32>
    %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %464 = stablehlo.broadcast_in_dim %arg103, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %465 = stablehlo.add %463, %464 : tensor<197x768xf32>
    %466 = stablehlo.convert %465 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %467 = stablehlo.reshape %466 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %468 = stablehlo.dot_general %458, %arg104, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %469 = stablehlo.reshape %468 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %470 = stablehlo.reshape %469 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %471 = stablehlo.transpose %470, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %472 = stablehlo.dot_general %459, %arg105, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %473 = stablehlo.broadcast_in_dim %472, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %474 = stablehlo.multiply %473, %60 : tensor<197x768xf32>
    %475 = stablehlo.broadcast_in_dim %474, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %476 = stablehlo.broadcast_in_dim %arg106, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %477 = stablehlo.add %475, %476 : tensor<197x768xf32>
    %478 = stablehlo.convert %477 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %479 = stablehlo.reshape %478 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %480 = stablehlo.reshape %479 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %481 = stablehlo.transpose %480, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %482 = stablehlo.reshape %467 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %483 = stablehlo.transpose %482, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %484 = stablehlo.transpose %471, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %485 = stablehlo.reshape %483 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %486 = stablehlo.reshape %484 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %488 = stablehlo.dot_general %485, %487, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %489 = stablehlo.reshape %488 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %491 = stablehlo.divide %490, %92 : tensor<1x12x197x197xbf16>
    %492 = stablehlo.add %491, %arg107 : tensor<1x12x197x197xbf16>
    %493 = stablehlo.convert %492 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %494 = stablehlo.reduce(%493 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %495 = stablehlo.reshape %494 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %496 = stablehlo.broadcast_in_dim %493, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %497 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %498 = stablehlo.subtract %496, %497 : tensor<1x12x197x197xf32>
    %499 = stablehlo.exponential %498 : tensor<1x12x197x197xf32>
    %500 = stablehlo.reduce(%499 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %501 = stablehlo.reshape %500 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %502 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %503 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %504 = stablehlo.divide %502, %503 : tensor<1x12x197x197xf32>
    %505 = stablehlo.convert %504 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %506 = stablehlo.reshape %505 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %507 = stablehlo.reshape %481 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %508 = stablehlo.broadcast_in_dim %507, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %509 = stablehlo.dot_general %506, %508, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %510 = stablehlo.reshape %509 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %511 = stablehlo.transpose %510, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %512 = stablehlo.reshape %511 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %513 = stablehlo.reshape %512 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %514 = stablehlo.convert %513 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %515 = stablehlo.dot_general %514, %arg108, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %517 = stablehlo.multiply %516, %60 : tensor<197x768xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %519 = stablehlo.broadcast_in_dim %arg109, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %520 = stablehlo.add %518, %519 : tensor<197x768xf32>
    %521 = stablehlo.convert %520 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %522 = stablehlo.reshape %521 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %523 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %524 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %525 = stablehlo.multiply %523, %524 : tensor<1x197x768xbf16>
    %526 = stablehlo.add %525, %420 : tensor<1x197x768xbf16>
    %527 = stablehlo.convert %526 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %528 = stablehlo.convert %527 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %529 = stablehlo.reduce(%528 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %530 = stablehlo.reshape %529 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %532 = stablehlo.divide %531, %15 : tensor<1x197x1xf64>
    %533 = stablehlo.broadcast_in_dim %528, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %534 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %535 = stablehlo.subtract %533, %534 : tensor<1x197x768xf64>
    %536 = stablehlo.multiply %535, %535 : tensor<1x197x768xf64>
    %537 = stablehlo.reduce(%536 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %538 = stablehlo.reshape %537 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %540 = stablehlo.divide %539, %15 : tensor<1x197x1xf64>
    %541 = stablehlo.convert %540 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %542 = stablehlo.reduce(%527 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %543 = stablehlo.reshape %542 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %545 = stablehlo.divide %544, %31 : tensor<1x197x1xf32>
    %546 = stablehlo.broadcast_in_dim %541, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %547 = stablehlo.add %546, %36 : tensor<1x197x1xf32>
    %548 = stablehlo.rsqrt %547 : tensor<1x197x1xf32>
    %549 = stablehlo.broadcast_in_dim %527, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %550 = stablehlo.broadcast_in_dim %545, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %551 = stablehlo.subtract %549, %550 : tensor<1x197x768xf32>
    %552 = stablehlo.broadcast_in_dim %551, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %553 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %554 = stablehlo.multiply %552, %553 : tensor<1x197x768xf32>
    %555 = stablehlo.convert %arg18 : (tensor<768xbf16>) -> tensor<768xf32>
    %556 = stablehlo.broadcast_in_dim %554, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %557 = stablehlo.broadcast_in_dim %555, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %558 = stablehlo.multiply %556, %557 : tensor<1x197x768xf32>
    %559 = stablehlo.convert %arg19 : (tensor<768xbf16>) -> tensor<768xf32>
    %560 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %561 = stablehlo.broadcast_in_dim %559, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %562 = stablehlo.add %560, %561 : tensor<1x197x768xf32>
    %563 = stablehlo.convert %562 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %564 = stablehlo.reshape %563 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %565 = stablehlo.convert %564 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %566 = stablehlo.dot_general %565, %arg110, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %568 = stablehlo.multiply %567, %170 : tensor<197x3072xf32>
    %569 = stablehlo.broadcast_in_dim %568, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %570 = stablehlo.broadcast_in_dim %arg111, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %571 = stablehlo.add %569, %570 : tensor<197x3072xf32>
    %572 = stablehlo.convert %571 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %573 = stablehlo.reshape %572 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %574 = stablehlo.multiply %573, %cst_4 : tensor<1x197x3072xbf16>
    %575 = stablehlo.multiply %573, %178 : tensor<1x197x3072xbf16>
    %576 = stablehlo.convert %575 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %577 = stablehlo.clamp %cst_5, %576, %cst_6 : tensor<1x197x3072xf32>
    %578 = stablehlo.multiply %577, %577 : tensor<1x197x3072xf32>
    %579 = stablehlo.multiply %cst_7, %578 : tensor<1x197x3072xf32>
    %580 = stablehlo.add %579, %cst_8 : tensor<1x197x3072xf32>
    %581 = stablehlo.multiply %580, %578 : tensor<1x197x3072xf32>
    %582 = stablehlo.add %581, %cst_9 : tensor<1x197x3072xf32>
    %583 = stablehlo.multiply %582, %578 : tensor<1x197x3072xf32>
    %584 = stablehlo.add %583, %cst_10 : tensor<1x197x3072xf32>
    %585 = stablehlo.multiply %584, %578 : tensor<1x197x3072xf32>
    %586 = stablehlo.add %585, %cst_11 : tensor<1x197x3072xf32>
    %587 = stablehlo.multiply %586, %578 : tensor<1x197x3072xf32>
    %588 = stablehlo.add %587, %cst_12 : tensor<1x197x3072xf32>
    %589 = stablehlo.multiply %588, %578 : tensor<1x197x3072xf32>
    %590 = stablehlo.add %589, %cst_13 : tensor<1x197x3072xf32>
    %591 = stablehlo.multiply %cst_14, %578 : tensor<1x197x3072xf32>
    %592 = stablehlo.add %591, %cst_15 : tensor<1x197x3072xf32>
    %593 = stablehlo.multiply %592, %578 : tensor<1x197x3072xf32>
    %594 = stablehlo.add %593, %cst_16 : tensor<1x197x3072xf32>
    %595 = stablehlo.multiply %594, %578 : tensor<1x197x3072xf32>
    %596 = stablehlo.add %595, %cst_17 : tensor<1x197x3072xf32>
    %597 = stablehlo.multiply %596, %578 : tensor<1x197x3072xf32>
    %598 = stablehlo.add %597, %cst_18 : tensor<1x197x3072xf32>
    %599 = stablehlo.multiply %577, %590 : tensor<1x197x3072xf32>
    %600 = stablehlo.divide %599, %598 : tensor<1x197x3072xf32>
    %601 = stablehlo.clamp %cst_19, %600, %cst_20 : tensor<1x197x3072xf32>
    %602 = stablehlo.convert %601 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %603 = stablehlo.add %602, %cst_2 : tensor<1x197x3072xbf16>
    %604 = stablehlo.multiply %603, %574 : tensor<1x197x3072xbf16>
    %605 = stablehlo.reshape %604 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %606 = stablehlo.convert %605 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %607 = stablehlo.dot_general %606, %arg112, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %609 = stablehlo.multiply %608, %60 : tensor<197x768xf32>
    %610 = stablehlo.broadcast_in_dim %609, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %611 = stablehlo.broadcast_in_dim %arg113, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %612 = stablehlo.add %610, %611 : tensor<197x768xf32>
    %613 = stablehlo.convert %612 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %614 = stablehlo.reshape %613 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %615 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %616 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %617 = stablehlo.multiply %615, %616 : tensor<1x197x768xbf16>
    %618 = stablehlo.add %617, %526 : tensor<1x197x768xbf16>
    %619 = stablehlo.convert %618 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %620 = stablehlo.convert %619 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %621 = stablehlo.reduce(%620 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %622 = stablehlo.reshape %621 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %623 = stablehlo.broadcast_in_dim %622, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %624 = stablehlo.divide %623, %15 : tensor<1x197x1xf64>
    %625 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %626 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %627 = stablehlo.subtract %625, %626 : tensor<1x197x768xf64>
    %628 = stablehlo.multiply %627, %627 : tensor<1x197x768xf64>
    %629 = stablehlo.reduce(%628 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %630 = stablehlo.reshape %629 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %632 = stablehlo.divide %631, %15 : tensor<1x197x1xf64>
    %633 = stablehlo.convert %632 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %634 = stablehlo.reduce(%619 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %635 = stablehlo.reshape %634 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %636 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %637 = stablehlo.divide %636, %31 : tensor<1x197x1xf32>
    %638 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %639 = stablehlo.add %638, %36 : tensor<1x197x1xf32>
    %640 = stablehlo.rsqrt %639 : tensor<1x197x1xf32>
    %641 = stablehlo.broadcast_in_dim %619, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %642 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %643 = stablehlo.subtract %641, %642 : tensor<1x197x768xf32>
    %644 = stablehlo.broadcast_in_dim %643, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %645 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %646 = stablehlo.multiply %644, %645 : tensor<1x197x768xf32>
    %647 = stablehlo.convert %arg21 : (tensor<768xbf16>) -> tensor<768xf32>
    %648 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %649 = stablehlo.broadcast_in_dim %647, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %650 = stablehlo.multiply %648, %649 : tensor<1x197x768xf32>
    %651 = stablehlo.convert %arg22 : (tensor<768xbf16>) -> tensor<768xf32>
    %652 = stablehlo.broadcast_in_dim %650, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %653 = stablehlo.broadcast_in_dim %651, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %654 = stablehlo.add %652, %653 : tensor<1x197x768xf32>
    %655 = stablehlo.convert %654 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %656 = stablehlo.reshape %655 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %657 = stablehlo.convert %656 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %658 = stablehlo.dot_general %657, %arg114, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %659 = stablehlo.broadcast_in_dim %658, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %660 = stablehlo.multiply %659, %60 : tensor<197x768xf32>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %662 = stablehlo.broadcast_in_dim %arg115, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %663 = stablehlo.add %661, %662 : tensor<197x768xf32>
    %664 = stablehlo.convert %663 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %665 = stablehlo.reshape %664 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %666 = stablehlo.dot_general %656, %arg116, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %667 = stablehlo.reshape %666 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %668 = stablehlo.reshape %667 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %669 = stablehlo.transpose %668, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %670 = stablehlo.dot_general %657, %arg117, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %672 = stablehlo.multiply %671, %60 : tensor<197x768xf32>
    %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %674 = stablehlo.broadcast_in_dim %arg118, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %675 = stablehlo.add %673, %674 : tensor<197x768xf32>
    %676 = stablehlo.convert %675 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %677 = stablehlo.reshape %676 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %678 = stablehlo.reshape %677 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %679 = stablehlo.transpose %678, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %680 = stablehlo.reshape %665 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %681 = stablehlo.transpose %680, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %682 = stablehlo.transpose %669, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %683 = stablehlo.reshape %681 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %684 = stablehlo.reshape %682 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %685 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %686 = stablehlo.dot_general %683, %685, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %687 = stablehlo.reshape %686 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %689 = stablehlo.divide %688, %92 : tensor<1x12x197x197xbf16>
    %690 = stablehlo.add %689, %arg119 : tensor<1x12x197x197xbf16>
    %691 = stablehlo.convert %690 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %692 = stablehlo.reduce(%691 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %693 = stablehlo.reshape %692 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %694 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %695 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %696 = stablehlo.subtract %694, %695 : tensor<1x12x197x197xf32>
    %697 = stablehlo.exponential %696 : tensor<1x12x197x197xf32>
    %698 = stablehlo.reduce(%697 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %699 = stablehlo.reshape %698 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %700 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %701 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %702 = stablehlo.divide %700, %701 : tensor<1x12x197x197xf32>
    %703 = stablehlo.convert %702 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %704 = stablehlo.reshape %703 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %705 = stablehlo.reshape %679 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %706 = stablehlo.broadcast_in_dim %705, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %707 = stablehlo.dot_general %704, %706, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %708 = stablehlo.reshape %707 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %709 = stablehlo.transpose %708, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %710 = stablehlo.reshape %709 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %711 = stablehlo.reshape %710 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %712 = stablehlo.convert %711 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %713 = stablehlo.dot_general %712, %arg120, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %714 = stablehlo.broadcast_in_dim %713, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %715 = stablehlo.multiply %714, %60 : tensor<197x768xf32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %717 = stablehlo.broadcast_in_dim %arg121, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %718 = stablehlo.add %716, %717 : tensor<197x768xf32>
    %719 = stablehlo.convert %718 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %720 = stablehlo.reshape %719 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %721 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %722 = stablehlo.broadcast_in_dim %720, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %723 = stablehlo.multiply %721, %722 : tensor<1x197x768xbf16>
    %724 = stablehlo.add %723, %618 : tensor<1x197x768xbf16>
    %725 = stablehlo.convert %724 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %726 = stablehlo.convert %725 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %727 = stablehlo.reduce(%726 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %728 = stablehlo.reshape %727 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %729 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %730 = stablehlo.divide %729, %15 : tensor<1x197x1xf64>
    %731 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %732 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %733 = stablehlo.subtract %731, %732 : tensor<1x197x768xf64>
    %734 = stablehlo.multiply %733, %733 : tensor<1x197x768xf64>
    %735 = stablehlo.reduce(%734 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %736 = stablehlo.reshape %735 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %737 = stablehlo.broadcast_in_dim %736, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %738 = stablehlo.divide %737, %15 : tensor<1x197x1xf64>
    %739 = stablehlo.convert %738 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %740 = stablehlo.reduce(%725 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %741 = stablehlo.reshape %740 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %742 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %743 = stablehlo.divide %742, %31 : tensor<1x197x1xf32>
    %744 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %745 = stablehlo.add %744, %36 : tensor<1x197x1xf32>
    %746 = stablehlo.rsqrt %745 : tensor<1x197x1xf32>
    %747 = stablehlo.broadcast_in_dim %725, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %748 = stablehlo.broadcast_in_dim %743, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %749 = stablehlo.subtract %747, %748 : tensor<1x197x768xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %751 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %752 = stablehlo.multiply %750, %751 : tensor<1x197x768xf32>
    %753 = stablehlo.convert %arg24 : (tensor<768xbf16>) -> tensor<768xf32>
    %754 = stablehlo.broadcast_in_dim %752, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %755 = stablehlo.broadcast_in_dim %753, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %756 = stablehlo.multiply %754, %755 : tensor<1x197x768xf32>
    %757 = stablehlo.convert %arg25 : (tensor<768xbf16>) -> tensor<768xf32>
    %758 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %759 = stablehlo.broadcast_in_dim %757, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %760 = stablehlo.add %758, %759 : tensor<1x197x768xf32>
    %761 = stablehlo.convert %760 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %762 = stablehlo.reshape %761 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %763 = stablehlo.convert %762 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %764 = stablehlo.dot_general %763, %arg122, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %766 = stablehlo.multiply %765, %170 : tensor<197x3072xf32>
    %767 = stablehlo.broadcast_in_dim %766, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %768 = stablehlo.broadcast_in_dim %arg123, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %769 = stablehlo.add %767, %768 : tensor<197x3072xf32>
    %770 = stablehlo.convert %769 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %771 = stablehlo.reshape %770 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %772 = stablehlo.multiply %771, %cst_4 : tensor<1x197x3072xbf16>
    %773 = stablehlo.multiply %771, %178 : tensor<1x197x3072xbf16>
    %774 = stablehlo.convert %773 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %775 = stablehlo.clamp %cst_5, %774, %cst_6 : tensor<1x197x3072xf32>
    %776 = stablehlo.multiply %775, %775 : tensor<1x197x3072xf32>
    %777 = stablehlo.multiply %cst_7, %776 : tensor<1x197x3072xf32>
    %778 = stablehlo.add %777, %cst_8 : tensor<1x197x3072xf32>
    %779 = stablehlo.multiply %778, %776 : tensor<1x197x3072xf32>
    %780 = stablehlo.add %779, %cst_9 : tensor<1x197x3072xf32>
    %781 = stablehlo.multiply %780, %776 : tensor<1x197x3072xf32>
    %782 = stablehlo.add %781, %cst_10 : tensor<1x197x3072xf32>
    %783 = stablehlo.multiply %782, %776 : tensor<1x197x3072xf32>
    %784 = stablehlo.add %783, %cst_11 : tensor<1x197x3072xf32>
    %785 = stablehlo.multiply %784, %776 : tensor<1x197x3072xf32>
    %786 = stablehlo.add %785, %cst_12 : tensor<1x197x3072xf32>
    %787 = stablehlo.multiply %786, %776 : tensor<1x197x3072xf32>
    %788 = stablehlo.add %787, %cst_13 : tensor<1x197x3072xf32>
    %789 = stablehlo.multiply %cst_14, %776 : tensor<1x197x3072xf32>
    %790 = stablehlo.add %789, %cst_15 : tensor<1x197x3072xf32>
    %791 = stablehlo.multiply %790, %776 : tensor<1x197x3072xf32>
    %792 = stablehlo.add %791, %cst_16 : tensor<1x197x3072xf32>
    %793 = stablehlo.multiply %792, %776 : tensor<1x197x3072xf32>
    %794 = stablehlo.add %793, %cst_17 : tensor<1x197x3072xf32>
    %795 = stablehlo.multiply %794, %776 : tensor<1x197x3072xf32>
    %796 = stablehlo.add %795, %cst_18 : tensor<1x197x3072xf32>
    %797 = stablehlo.multiply %775, %788 : tensor<1x197x3072xf32>
    %798 = stablehlo.divide %797, %796 : tensor<1x197x3072xf32>
    %799 = stablehlo.clamp %cst_19, %798, %cst_20 : tensor<1x197x3072xf32>
    %800 = stablehlo.convert %799 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %801 = stablehlo.add %800, %cst_2 : tensor<1x197x3072xbf16>
    %802 = stablehlo.multiply %801, %772 : tensor<1x197x3072xbf16>
    %803 = stablehlo.reshape %802 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %804 = stablehlo.convert %803 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %805 = stablehlo.dot_general %804, %arg124, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %806 = stablehlo.broadcast_in_dim %805, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %807 = stablehlo.multiply %806, %60 : tensor<197x768xf32>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %809 = stablehlo.broadcast_in_dim %arg125, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %810 = stablehlo.add %808, %809 : tensor<197x768xf32>
    %811 = stablehlo.convert %810 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %812 = stablehlo.reshape %811 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %813 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %814 = stablehlo.broadcast_in_dim %812, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %815 = stablehlo.multiply %813, %814 : tensor<1x197x768xbf16>
    %816 = stablehlo.add %815, %724 : tensor<1x197x768xbf16>
    %817 = stablehlo.convert %816 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %818 = stablehlo.convert %817 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %819 = stablehlo.reduce(%818 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %820 = stablehlo.reshape %819 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %822 = stablehlo.divide %821, %15 : tensor<1x197x1xf64>
    %823 = stablehlo.broadcast_in_dim %818, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %824 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %825 = stablehlo.subtract %823, %824 : tensor<1x197x768xf64>
    %826 = stablehlo.multiply %825, %825 : tensor<1x197x768xf64>
    %827 = stablehlo.reduce(%826 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %828 = stablehlo.reshape %827 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %830 = stablehlo.divide %829, %15 : tensor<1x197x1xf64>
    %831 = stablehlo.convert %830 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %832 = stablehlo.reduce(%817 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %833 = stablehlo.reshape %832 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %834 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %835 = stablehlo.divide %834, %31 : tensor<1x197x1xf32>
    %836 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %837 = stablehlo.add %836, %36 : tensor<1x197x1xf32>
    %838 = stablehlo.rsqrt %837 : tensor<1x197x1xf32>
    %839 = stablehlo.broadcast_in_dim %817, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %840 = stablehlo.broadcast_in_dim %835, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %841 = stablehlo.subtract %839, %840 : tensor<1x197x768xf32>
    %842 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %843 = stablehlo.broadcast_in_dim %838, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %844 = stablehlo.multiply %842, %843 : tensor<1x197x768xf32>
    %845 = stablehlo.convert %arg27 : (tensor<768xbf16>) -> tensor<768xf32>
    %846 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %847 = stablehlo.broadcast_in_dim %845, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %848 = stablehlo.multiply %846, %847 : tensor<1x197x768xf32>
    %849 = stablehlo.convert %arg28 : (tensor<768xbf16>) -> tensor<768xf32>
    %850 = stablehlo.broadcast_in_dim %848, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %851 = stablehlo.broadcast_in_dim %849, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %852 = stablehlo.add %850, %851 : tensor<1x197x768xf32>
    %853 = stablehlo.convert %852 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %854 = stablehlo.reshape %853 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %855 = stablehlo.convert %854 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %856 = stablehlo.dot_general %855, %arg126, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %857 = stablehlo.broadcast_in_dim %856, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %858 = stablehlo.multiply %857, %60 : tensor<197x768xf32>
    %859 = stablehlo.broadcast_in_dim %858, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %860 = stablehlo.broadcast_in_dim %arg127, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %861 = stablehlo.add %859, %860 : tensor<197x768xf32>
    %862 = stablehlo.convert %861 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %863 = stablehlo.reshape %862 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %864 = stablehlo.dot_general %854, %arg128, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %865 = stablehlo.reshape %864 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %866 = stablehlo.reshape %865 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %867 = stablehlo.transpose %866, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %868 = stablehlo.dot_general %855, %arg129, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %869 = stablehlo.broadcast_in_dim %868, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %870 = stablehlo.multiply %869, %60 : tensor<197x768xf32>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %872 = stablehlo.broadcast_in_dim %arg130, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %873 = stablehlo.add %871, %872 : tensor<197x768xf32>
    %874 = stablehlo.convert %873 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %875 = stablehlo.reshape %874 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %876 = stablehlo.reshape %875 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %877 = stablehlo.transpose %876, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %878 = stablehlo.reshape %863 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %879 = stablehlo.transpose %878, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %880 = stablehlo.transpose %867, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %881 = stablehlo.reshape %879 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %882 = stablehlo.reshape %880 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %884 = stablehlo.dot_general %881, %883, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %885 = stablehlo.reshape %884 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %886 = stablehlo.broadcast_in_dim %885, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %887 = stablehlo.divide %886, %92 : tensor<1x12x197x197xbf16>
    %888 = stablehlo.add %887, %arg131 : tensor<1x12x197x197xbf16>
    %889 = stablehlo.convert %888 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %890 = stablehlo.reduce(%889 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %891 = stablehlo.reshape %890 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %892 = stablehlo.broadcast_in_dim %889, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %893 = stablehlo.broadcast_in_dim %891, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %894 = stablehlo.subtract %892, %893 : tensor<1x12x197x197xf32>
    %895 = stablehlo.exponential %894 : tensor<1x12x197x197xf32>
    %896 = stablehlo.reduce(%895 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %897 = stablehlo.reshape %896 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %898 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %899 = stablehlo.broadcast_in_dim %897, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %900 = stablehlo.divide %898, %899 : tensor<1x12x197x197xf32>
    %901 = stablehlo.convert %900 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %902 = stablehlo.reshape %901 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %903 = stablehlo.reshape %877 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %905 = stablehlo.dot_general %902, %904, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %906 = stablehlo.reshape %905 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %907 = stablehlo.transpose %906, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %908 = stablehlo.reshape %907 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %909 = stablehlo.reshape %908 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %910 = stablehlo.convert %909 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %911 = stablehlo.dot_general %910, %arg132, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %912 = stablehlo.broadcast_in_dim %911, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %913 = stablehlo.multiply %912, %60 : tensor<197x768xf32>
    %914 = stablehlo.broadcast_in_dim %913, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %915 = stablehlo.broadcast_in_dim %arg133, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %916 = stablehlo.add %914, %915 : tensor<197x768xf32>
    %917 = stablehlo.convert %916 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %918 = stablehlo.reshape %917 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %919 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %920 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %921 = stablehlo.multiply %919, %920 : tensor<1x197x768xbf16>
    %922 = stablehlo.add %921, %816 : tensor<1x197x768xbf16>
    %923 = stablehlo.convert %922 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %924 = stablehlo.convert %923 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %925 = stablehlo.reduce(%924 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %926 = stablehlo.reshape %925 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %927 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %928 = stablehlo.divide %927, %15 : tensor<1x197x1xf64>
    %929 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %930 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %931 = stablehlo.subtract %929, %930 : tensor<1x197x768xf64>
    %932 = stablehlo.multiply %931, %931 : tensor<1x197x768xf64>
    %933 = stablehlo.reduce(%932 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %934 = stablehlo.reshape %933 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %936 = stablehlo.divide %935, %15 : tensor<1x197x1xf64>
    %937 = stablehlo.convert %936 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %938 = stablehlo.reduce(%923 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %939 = stablehlo.reshape %938 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %941 = stablehlo.divide %940, %31 : tensor<1x197x1xf32>
    %942 = stablehlo.broadcast_in_dim %937, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %943 = stablehlo.add %942, %36 : tensor<1x197x1xf32>
    %944 = stablehlo.rsqrt %943 : tensor<1x197x1xf32>
    %945 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %946 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %947 = stablehlo.subtract %945, %946 : tensor<1x197x768xf32>
    %948 = stablehlo.broadcast_in_dim %947, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %949 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %950 = stablehlo.multiply %948, %949 : tensor<1x197x768xf32>
    %951 = stablehlo.convert %arg30 : (tensor<768xbf16>) -> tensor<768xf32>
    %952 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %953 = stablehlo.broadcast_in_dim %951, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %954 = stablehlo.multiply %952, %953 : tensor<1x197x768xf32>
    %955 = stablehlo.convert %arg31 : (tensor<768xbf16>) -> tensor<768xf32>
    %956 = stablehlo.broadcast_in_dim %954, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %957 = stablehlo.broadcast_in_dim %955, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %958 = stablehlo.add %956, %957 : tensor<1x197x768xf32>
    %959 = stablehlo.convert %958 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %960 = stablehlo.reshape %959 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %961 = stablehlo.convert %960 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %962 = stablehlo.dot_general %961, %arg134, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %964 = stablehlo.multiply %963, %170 : tensor<197x3072xf32>
    %965 = stablehlo.broadcast_in_dim %964, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %966 = stablehlo.broadcast_in_dim %arg135, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %967 = stablehlo.add %965, %966 : tensor<197x3072xf32>
    %968 = stablehlo.convert %967 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %969 = stablehlo.reshape %968 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %970 = stablehlo.multiply %969, %cst_4 : tensor<1x197x3072xbf16>
    %971 = stablehlo.multiply %969, %178 : tensor<1x197x3072xbf16>
    %972 = stablehlo.convert %971 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %973 = stablehlo.clamp %cst_5, %972, %cst_6 : tensor<1x197x3072xf32>
    %974 = stablehlo.multiply %973, %973 : tensor<1x197x3072xf32>
    %975 = stablehlo.multiply %cst_7, %974 : tensor<1x197x3072xf32>
    %976 = stablehlo.add %975, %cst_8 : tensor<1x197x3072xf32>
    %977 = stablehlo.multiply %976, %974 : tensor<1x197x3072xf32>
    %978 = stablehlo.add %977, %cst_9 : tensor<1x197x3072xf32>
    %979 = stablehlo.multiply %978, %974 : tensor<1x197x3072xf32>
    %980 = stablehlo.add %979, %cst_10 : tensor<1x197x3072xf32>
    %981 = stablehlo.multiply %980, %974 : tensor<1x197x3072xf32>
    %982 = stablehlo.add %981, %cst_11 : tensor<1x197x3072xf32>
    %983 = stablehlo.multiply %982, %974 : tensor<1x197x3072xf32>
    %984 = stablehlo.add %983, %cst_12 : tensor<1x197x3072xf32>
    %985 = stablehlo.multiply %984, %974 : tensor<1x197x3072xf32>
    %986 = stablehlo.add %985, %cst_13 : tensor<1x197x3072xf32>
    %987 = stablehlo.multiply %cst_14, %974 : tensor<1x197x3072xf32>
    %988 = stablehlo.add %987, %cst_15 : tensor<1x197x3072xf32>
    %989 = stablehlo.multiply %988, %974 : tensor<1x197x3072xf32>
    %990 = stablehlo.add %989, %cst_16 : tensor<1x197x3072xf32>
    %991 = stablehlo.multiply %990, %974 : tensor<1x197x3072xf32>
    %992 = stablehlo.add %991, %cst_17 : tensor<1x197x3072xf32>
    %993 = stablehlo.multiply %992, %974 : tensor<1x197x3072xf32>
    %994 = stablehlo.add %993, %cst_18 : tensor<1x197x3072xf32>
    %995 = stablehlo.multiply %973, %986 : tensor<1x197x3072xf32>
    %996 = stablehlo.divide %995, %994 : tensor<1x197x3072xf32>
    %997 = stablehlo.clamp %cst_19, %996, %cst_20 : tensor<1x197x3072xf32>
    %998 = stablehlo.convert %997 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %999 = stablehlo.add %998, %cst_2 : tensor<1x197x3072xbf16>
    %1000 = stablehlo.multiply %999, %970 : tensor<1x197x3072xbf16>
    %1001 = stablehlo.reshape %1000 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %1002 = stablehlo.convert %1001 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %1003 = stablehlo.dot_general %1002, %arg136, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1005 = stablehlo.multiply %1004, %60 : tensor<197x768xf32>
    %1006 = stablehlo.broadcast_in_dim %1005, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1007 = stablehlo.broadcast_in_dim %arg137, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1008 = stablehlo.add %1006, %1007 : tensor<197x768xf32>
    %1009 = stablehlo.convert %1008 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1010 = stablehlo.reshape %1009 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1011 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1012 = stablehlo.broadcast_in_dim %1010, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1013 = stablehlo.multiply %1011, %1012 : tensor<1x197x768xbf16>
    %1014 = stablehlo.add %1013, %922 : tensor<1x197x768xbf16>
    %1015 = stablehlo.convert %1014 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1016 = stablehlo.convert %1015 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1017 = stablehlo.reduce(%1016 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1018 = stablehlo.reshape %1017 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1019 = stablehlo.broadcast_in_dim %1018, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1020 = stablehlo.divide %1019, %15 : tensor<1x197x1xf64>
    %1021 = stablehlo.broadcast_in_dim %1016, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1022 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1023 = stablehlo.subtract %1021, %1022 : tensor<1x197x768xf64>
    %1024 = stablehlo.multiply %1023, %1023 : tensor<1x197x768xf64>
    %1025 = stablehlo.reduce(%1024 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1026 = stablehlo.reshape %1025 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1027 = stablehlo.broadcast_in_dim %1026, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1028 = stablehlo.divide %1027, %15 : tensor<1x197x1xf64>
    %1029 = stablehlo.convert %1028 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1030 = stablehlo.reduce(%1015 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1031 = stablehlo.reshape %1030 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1032 = stablehlo.broadcast_in_dim %1031, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1033 = stablehlo.divide %1032, %31 : tensor<1x197x1xf32>
    %1034 = stablehlo.broadcast_in_dim %1029, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1035 = stablehlo.add %1034, %36 : tensor<1x197x1xf32>
    %1036 = stablehlo.rsqrt %1035 : tensor<1x197x1xf32>
    %1037 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1038 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1039 = stablehlo.subtract %1037, %1038 : tensor<1x197x768xf32>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1041 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1042 = stablehlo.multiply %1040, %1041 : tensor<1x197x768xf32>
    %1043 = stablehlo.convert %arg33 : (tensor<768xbf16>) -> tensor<768xf32>
    %1044 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1045 = stablehlo.broadcast_in_dim %1043, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1046 = stablehlo.multiply %1044, %1045 : tensor<1x197x768xf32>
    %1047 = stablehlo.convert %arg34 : (tensor<768xbf16>) -> tensor<768xf32>
    %1048 = stablehlo.broadcast_in_dim %1046, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1049 = stablehlo.broadcast_in_dim %1047, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1050 = stablehlo.add %1048, %1049 : tensor<1x197x768xf32>
    %1051 = stablehlo.convert %1050 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1052 = stablehlo.reshape %1051 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1053 = stablehlo.convert %1052 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1054 = stablehlo.dot_general %1053, %arg138, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1055 = stablehlo.broadcast_in_dim %1054, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1056 = stablehlo.multiply %1055, %60 : tensor<197x768xf32>
    %1057 = stablehlo.broadcast_in_dim %1056, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1058 = stablehlo.broadcast_in_dim %arg139, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1059 = stablehlo.add %1057, %1058 : tensor<197x768xf32>
    %1060 = stablehlo.convert %1059 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1061 = stablehlo.reshape %1060 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1062 = stablehlo.dot_general %1052, %arg140, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %1063 = stablehlo.reshape %1062 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1064 = stablehlo.reshape %1063 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1065 = stablehlo.transpose %1064, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1066 = stablehlo.dot_general %1053, %arg141, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1067 = stablehlo.broadcast_in_dim %1066, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1068 = stablehlo.multiply %1067, %60 : tensor<197x768xf32>
    %1069 = stablehlo.broadcast_in_dim %1068, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1070 = stablehlo.broadcast_in_dim %arg142, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1071 = stablehlo.add %1069, %1070 : tensor<197x768xf32>
    %1072 = stablehlo.convert %1071 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1073 = stablehlo.reshape %1072 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1074 = stablehlo.reshape %1073 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1075 = stablehlo.transpose %1074, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1076 = stablehlo.reshape %1061 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1077 = stablehlo.transpose %1076, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1078 = stablehlo.transpose %1065, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %1079 = stablehlo.reshape %1077 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1080 = stablehlo.reshape %1078 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1081 = stablehlo.broadcast_in_dim %1080, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1082 = stablehlo.dot_general %1079, %1081, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %1083 = stablehlo.reshape %1082 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1084 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1085 = stablehlo.divide %1084, %92 : tensor<1x12x197x197xbf16>
    %1086 = stablehlo.add %1085, %arg143 : tensor<1x12x197x197xbf16>
    %1087 = stablehlo.convert %1086 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %1088 = stablehlo.reduce(%1087 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1089 = stablehlo.reshape %1088 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1090 = stablehlo.broadcast_in_dim %1087, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1091 = stablehlo.broadcast_in_dim %1089, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1092 = stablehlo.subtract %1090, %1091 : tensor<1x12x197x197xf32>
    %1093 = stablehlo.exponential %1092 : tensor<1x12x197x197xf32>
    %1094 = stablehlo.reduce(%1093 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1095 = stablehlo.reshape %1094 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1096 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1097 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1098 = stablehlo.divide %1096, %1097 : tensor<1x12x197x197xf32>
    %1099 = stablehlo.convert %1098 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %1100 = stablehlo.reshape %1099 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %1101 = stablehlo.reshape %1075 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1102 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1103 = stablehlo.dot_general %1100, %1102, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1104 = stablehlo.reshape %1103 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1105 = stablehlo.transpose %1104, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %1106 = stablehlo.reshape %1105 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %1107 = stablehlo.reshape %1106 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1108 = stablehlo.convert %1107 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1109 = stablehlo.dot_general %1108, %arg144, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1110 = stablehlo.broadcast_in_dim %1109, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1111 = stablehlo.multiply %1110, %60 : tensor<197x768xf32>
    %1112 = stablehlo.broadcast_in_dim %1111, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1113 = stablehlo.broadcast_in_dim %arg145, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1114 = stablehlo.add %1112, %1113 : tensor<197x768xf32>
    %1115 = stablehlo.convert %1114 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1116 = stablehlo.reshape %1115 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1117 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1118 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1119 = stablehlo.multiply %1117, %1118 : tensor<1x197x768xbf16>
    %1120 = stablehlo.add %1119, %1014 : tensor<1x197x768xbf16>
    %1121 = stablehlo.convert %1120 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1122 = stablehlo.convert %1121 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1123 = stablehlo.reduce(%1122 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1124 = stablehlo.reshape %1123 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1125 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1126 = stablehlo.divide %1125, %15 : tensor<1x197x1xf64>
    %1127 = stablehlo.broadcast_in_dim %1122, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1128 = stablehlo.broadcast_in_dim %1126, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1129 = stablehlo.subtract %1127, %1128 : tensor<1x197x768xf64>
    %1130 = stablehlo.multiply %1129, %1129 : tensor<1x197x768xf64>
    %1131 = stablehlo.reduce(%1130 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1133 = stablehlo.broadcast_in_dim %1132, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1134 = stablehlo.divide %1133, %15 : tensor<1x197x1xf64>
    %1135 = stablehlo.convert %1134 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1136 = stablehlo.reduce(%1121 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1137 = stablehlo.reshape %1136 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1139 = stablehlo.divide %1138, %31 : tensor<1x197x1xf32>
    %1140 = stablehlo.broadcast_in_dim %1135, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1141 = stablehlo.add %1140, %36 : tensor<1x197x1xf32>
    %1142 = stablehlo.rsqrt %1141 : tensor<1x197x1xf32>
    %1143 = stablehlo.broadcast_in_dim %1121, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1144 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1145 = stablehlo.subtract %1143, %1144 : tensor<1x197x768xf32>
    %1146 = stablehlo.broadcast_in_dim %1145, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1147 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1148 = stablehlo.multiply %1146, %1147 : tensor<1x197x768xf32>
    %1149 = stablehlo.convert %arg36 : (tensor<768xbf16>) -> tensor<768xf32>
    %1150 = stablehlo.broadcast_in_dim %1148, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1151 = stablehlo.broadcast_in_dim %1149, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1152 = stablehlo.multiply %1150, %1151 : tensor<1x197x768xf32>
    %1153 = stablehlo.convert %arg37 : (tensor<768xbf16>) -> tensor<768xf32>
    %1154 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1155 = stablehlo.broadcast_in_dim %1153, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1156 = stablehlo.add %1154, %1155 : tensor<1x197x768xf32>
    %1157 = stablehlo.convert %1156 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1158 = stablehlo.reshape %1157 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1159 = stablehlo.convert %1158 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1160 = stablehlo.dot_general %1159, %arg146, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %1161 = stablehlo.broadcast_in_dim %1160, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1162 = stablehlo.multiply %1161, %170 : tensor<197x3072xf32>
    %1163 = stablehlo.broadcast_in_dim %1162, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1164 = stablehlo.broadcast_in_dim %arg147, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %1165 = stablehlo.add %1163, %1164 : tensor<197x3072xf32>
    %1166 = stablehlo.convert %1165 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %1167 = stablehlo.reshape %1166 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %1168 = stablehlo.multiply %1167, %cst_4 : tensor<1x197x3072xbf16>
    %1169 = stablehlo.multiply %1167, %178 : tensor<1x197x3072xbf16>
    %1170 = stablehlo.convert %1169 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %1171 = stablehlo.clamp %cst_5, %1170, %cst_6 : tensor<1x197x3072xf32>
    %1172 = stablehlo.multiply %1171, %1171 : tensor<1x197x3072xf32>
    %1173 = stablehlo.multiply %cst_7, %1172 : tensor<1x197x3072xf32>
    %1174 = stablehlo.add %1173, %cst_8 : tensor<1x197x3072xf32>
    %1175 = stablehlo.multiply %1174, %1172 : tensor<1x197x3072xf32>
    %1176 = stablehlo.add %1175, %cst_9 : tensor<1x197x3072xf32>
    %1177 = stablehlo.multiply %1176, %1172 : tensor<1x197x3072xf32>
    %1178 = stablehlo.add %1177, %cst_10 : tensor<1x197x3072xf32>
    %1179 = stablehlo.multiply %1178, %1172 : tensor<1x197x3072xf32>
    %1180 = stablehlo.add %1179, %cst_11 : tensor<1x197x3072xf32>
    %1181 = stablehlo.multiply %1180, %1172 : tensor<1x197x3072xf32>
    %1182 = stablehlo.add %1181, %cst_12 : tensor<1x197x3072xf32>
    %1183 = stablehlo.multiply %1182, %1172 : tensor<1x197x3072xf32>
    %1184 = stablehlo.add %1183, %cst_13 : tensor<1x197x3072xf32>
    %1185 = stablehlo.multiply %cst_14, %1172 : tensor<1x197x3072xf32>
    %1186 = stablehlo.add %1185, %cst_15 : tensor<1x197x3072xf32>
    %1187 = stablehlo.multiply %1186, %1172 : tensor<1x197x3072xf32>
    %1188 = stablehlo.add %1187, %cst_16 : tensor<1x197x3072xf32>
    %1189 = stablehlo.multiply %1188, %1172 : tensor<1x197x3072xf32>
    %1190 = stablehlo.add %1189, %cst_17 : tensor<1x197x3072xf32>
    %1191 = stablehlo.multiply %1190, %1172 : tensor<1x197x3072xf32>
    %1192 = stablehlo.add %1191, %cst_18 : tensor<1x197x3072xf32>
    %1193 = stablehlo.multiply %1171, %1184 : tensor<1x197x3072xf32>
    %1194 = stablehlo.divide %1193, %1192 : tensor<1x197x3072xf32>
    %1195 = stablehlo.clamp %cst_19, %1194, %cst_20 : tensor<1x197x3072xf32>
    %1196 = stablehlo.convert %1195 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %1197 = stablehlo.add %1196, %cst_2 : tensor<1x197x3072xbf16>
    %1198 = stablehlo.multiply %1197, %1168 : tensor<1x197x3072xbf16>
    %1199 = stablehlo.reshape %1198 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %1200 = stablehlo.convert %1199 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %1201 = stablehlo.dot_general %1200, %arg148, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1203 = stablehlo.multiply %1202, %60 : tensor<197x768xf32>
    %1204 = stablehlo.broadcast_in_dim %1203, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1205 = stablehlo.broadcast_in_dim %arg149, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1206 = stablehlo.add %1204, %1205 : tensor<197x768xf32>
    %1207 = stablehlo.convert %1206 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1208 = stablehlo.reshape %1207 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1209 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1210 = stablehlo.broadcast_in_dim %1208, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1211 = stablehlo.multiply %1209, %1210 : tensor<1x197x768xbf16>
    %1212 = stablehlo.add %1211, %1120 : tensor<1x197x768xbf16>
    %1213 = stablehlo.convert %1212 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1214 = stablehlo.convert %1213 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1215 = stablehlo.reduce(%1214 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1216 = stablehlo.reshape %1215 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1218 = stablehlo.divide %1217, %15 : tensor<1x197x1xf64>
    %1219 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1220 = stablehlo.broadcast_in_dim %1218, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1221 = stablehlo.subtract %1219, %1220 : tensor<1x197x768xf64>
    %1222 = stablehlo.multiply %1221, %1221 : tensor<1x197x768xf64>
    %1223 = stablehlo.reduce(%1222 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1224 = stablehlo.reshape %1223 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1225 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1226 = stablehlo.divide %1225, %15 : tensor<1x197x1xf64>
    %1227 = stablehlo.convert %1226 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1228 = stablehlo.reduce(%1213 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1229 = stablehlo.reshape %1228 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1231 = stablehlo.divide %1230, %31 : tensor<1x197x1xf32>
    %1232 = stablehlo.broadcast_in_dim %1227, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1233 = stablehlo.add %1232, %36 : tensor<1x197x1xf32>
    %1234 = stablehlo.rsqrt %1233 : tensor<1x197x1xf32>
    %1235 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1236 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1237 = stablehlo.subtract %1235, %1236 : tensor<1x197x768xf32>
    %1238 = stablehlo.broadcast_in_dim %1237, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1239 = stablehlo.broadcast_in_dim %1234, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1240 = stablehlo.multiply %1238, %1239 : tensor<1x197x768xf32>
    %1241 = stablehlo.convert %arg39 : (tensor<768xbf16>) -> tensor<768xf32>
    %1242 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1243 = stablehlo.broadcast_in_dim %1241, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1244 = stablehlo.multiply %1242, %1243 : tensor<1x197x768xf32>
    %1245 = stablehlo.convert %arg40 : (tensor<768xbf16>) -> tensor<768xf32>
    %1246 = stablehlo.broadcast_in_dim %1244, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1247 = stablehlo.broadcast_in_dim %1245, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1248 = stablehlo.add %1246, %1247 : tensor<1x197x768xf32>
    %1249 = stablehlo.convert %1248 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1250 = stablehlo.reshape %1249 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1251 = stablehlo.convert %1250 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1252 = stablehlo.dot_general %1251, %arg150, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1253 = stablehlo.broadcast_in_dim %1252, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1254 = stablehlo.multiply %1253, %60 : tensor<197x768xf32>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1256 = stablehlo.broadcast_in_dim %arg151, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1257 = stablehlo.add %1255, %1256 : tensor<197x768xf32>
    %1258 = stablehlo.convert %1257 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1259 = stablehlo.reshape %1258 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1260 = stablehlo.dot_general %1250, %arg152, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %1261 = stablehlo.reshape %1260 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1262 = stablehlo.reshape %1261 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1263 = stablehlo.transpose %1262, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1264 = stablehlo.dot_general %1251, %arg153, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1265 = stablehlo.broadcast_in_dim %1264, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1266 = stablehlo.multiply %1265, %60 : tensor<197x768xf32>
    %1267 = stablehlo.broadcast_in_dim %1266, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1268 = stablehlo.broadcast_in_dim %arg154, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1269 = stablehlo.add %1267, %1268 : tensor<197x768xf32>
    %1270 = stablehlo.convert %1269 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1271 = stablehlo.reshape %1270 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1272 = stablehlo.reshape %1271 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1273 = stablehlo.transpose %1272, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1274 = stablehlo.reshape %1259 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1275 = stablehlo.transpose %1274, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1276 = stablehlo.transpose %1263, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %1277 = stablehlo.reshape %1275 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1278 = stablehlo.reshape %1276 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1279 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1280 = stablehlo.dot_general %1277, %1279, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %1281 = stablehlo.reshape %1280 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1282 = stablehlo.broadcast_in_dim %1281, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1283 = stablehlo.divide %1282, %92 : tensor<1x12x197x197xbf16>
    %1284 = stablehlo.add %1283, %arg155 : tensor<1x12x197x197xbf16>
    %1285 = stablehlo.convert %1284 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %1286 = stablehlo.reduce(%1285 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1287 = stablehlo.reshape %1286 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1288 = stablehlo.broadcast_in_dim %1285, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1289 = stablehlo.broadcast_in_dim %1287, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1290 = stablehlo.subtract %1288, %1289 : tensor<1x12x197x197xf32>
    %1291 = stablehlo.exponential %1290 : tensor<1x12x197x197xf32>
    %1292 = stablehlo.reduce(%1291 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1293 = stablehlo.reshape %1292 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1294 = stablehlo.broadcast_in_dim %1291, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1295 = stablehlo.broadcast_in_dim %1293, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1296 = stablehlo.divide %1294, %1295 : tensor<1x12x197x197xf32>
    %1297 = stablehlo.convert %1296 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %1298 = stablehlo.reshape %1297 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %1299 = stablehlo.reshape %1273 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1301 = stablehlo.dot_general %1298, %1300, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1302 = stablehlo.reshape %1301 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1303 = stablehlo.transpose %1302, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %1304 = stablehlo.reshape %1303 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %1305 = stablehlo.reshape %1304 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1306 = stablehlo.convert %1305 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1307 = stablehlo.dot_general %1306, %arg156, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1308 = stablehlo.broadcast_in_dim %1307, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1309 = stablehlo.multiply %1308, %60 : tensor<197x768xf32>
    %1310 = stablehlo.broadcast_in_dim %1309, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1311 = stablehlo.broadcast_in_dim %arg157, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1312 = stablehlo.add %1310, %1311 : tensor<197x768xf32>
    %1313 = stablehlo.convert %1312 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1314 = stablehlo.reshape %1313 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1315 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1316 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1317 = stablehlo.multiply %1315, %1316 : tensor<1x197x768xbf16>
    %1318 = stablehlo.add %1317, %1212 : tensor<1x197x768xbf16>
    %1319 = stablehlo.convert %1318 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1320 = stablehlo.convert %1319 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1321 = stablehlo.reduce(%1320 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1322 = stablehlo.reshape %1321 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1324 = stablehlo.divide %1323, %15 : tensor<1x197x1xf64>
    %1325 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1326 = stablehlo.broadcast_in_dim %1324, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1327 = stablehlo.subtract %1325, %1326 : tensor<1x197x768xf64>
    %1328 = stablehlo.multiply %1327, %1327 : tensor<1x197x768xf64>
    %1329 = stablehlo.reduce(%1328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1330 = stablehlo.reshape %1329 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1332 = stablehlo.divide %1331, %15 : tensor<1x197x1xf64>
    %1333 = stablehlo.convert %1332 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1334 = stablehlo.reduce(%1319 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1335 = stablehlo.reshape %1334 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1336 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1337 = stablehlo.divide %1336, %31 : tensor<1x197x1xf32>
    %1338 = stablehlo.broadcast_in_dim %1333, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1339 = stablehlo.add %1338, %36 : tensor<1x197x1xf32>
    %1340 = stablehlo.rsqrt %1339 : tensor<1x197x1xf32>
    %1341 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1342 = stablehlo.broadcast_in_dim %1337, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1343 = stablehlo.subtract %1341, %1342 : tensor<1x197x768xf32>
    %1344 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1345 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1346 = stablehlo.multiply %1344, %1345 : tensor<1x197x768xf32>
    %1347 = stablehlo.convert %arg42 : (tensor<768xbf16>) -> tensor<768xf32>
    %1348 = stablehlo.broadcast_in_dim %1346, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1349 = stablehlo.broadcast_in_dim %1347, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1350 = stablehlo.multiply %1348, %1349 : tensor<1x197x768xf32>
    %1351 = stablehlo.convert %arg43 : (tensor<768xbf16>) -> tensor<768xf32>
    %1352 = stablehlo.broadcast_in_dim %1350, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1353 = stablehlo.broadcast_in_dim %1351, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1354 = stablehlo.add %1352, %1353 : tensor<1x197x768xf32>
    %1355 = stablehlo.convert %1354 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1356 = stablehlo.reshape %1355 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1357 = stablehlo.convert %1356 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1358 = stablehlo.dot_general %1357, %arg158, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %1359 = stablehlo.broadcast_in_dim %1358, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1360 = stablehlo.multiply %1359, %170 : tensor<197x3072xf32>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1362 = stablehlo.broadcast_in_dim %arg159, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %1363 = stablehlo.add %1361, %1362 : tensor<197x3072xf32>
    %1364 = stablehlo.convert %1363 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %1365 = stablehlo.reshape %1364 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %1366 = stablehlo.multiply %1365, %cst_4 : tensor<1x197x3072xbf16>
    %1367 = stablehlo.multiply %1365, %178 : tensor<1x197x3072xbf16>
    %1368 = stablehlo.convert %1367 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %1369 = stablehlo.clamp %cst_5, %1368, %cst_6 : tensor<1x197x3072xf32>
    %1370 = stablehlo.multiply %1369, %1369 : tensor<1x197x3072xf32>
    %1371 = stablehlo.multiply %cst_7, %1370 : tensor<1x197x3072xf32>
    %1372 = stablehlo.add %1371, %cst_8 : tensor<1x197x3072xf32>
    %1373 = stablehlo.multiply %1372, %1370 : tensor<1x197x3072xf32>
    %1374 = stablehlo.add %1373, %cst_9 : tensor<1x197x3072xf32>
    %1375 = stablehlo.multiply %1374, %1370 : tensor<1x197x3072xf32>
    %1376 = stablehlo.add %1375, %cst_10 : tensor<1x197x3072xf32>
    %1377 = stablehlo.multiply %1376, %1370 : tensor<1x197x3072xf32>
    %1378 = stablehlo.add %1377, %cst_11 : tensor<1x197x3072xf32>
    %1379 = stablehlo.multiply %1378, %1370 : tensor<1x197x3072xf32>
    %1380 = stablehlo.add %1379, %cst_12 : tensor<1x197x3072xf32>
    %1381 = stablehlo.multiply %1380, %1370 : tensor<1x197x3072xf32>
    %1382 = stablehlo.add %1381, %cst_13 : tensor<1x197x3072xf32>
    %1383 = stablehlo.multiply %cst_14, %1370 : tensor<1x197x3072xf32>
    %1384 = stablehlo.add %1383, %cst_15 : tensor<1x197x3072xf32>
    %1385 = stablehlo.multiply %1384, %1370 : tensor<1x197x3072xf32>
    %1386 = stablehlo.add %1385, %cst_16 : tensor<1x197x3072xf32>
    %1387 = stablehlo.multiply %1386, %1370 : tensor<1x197x3072xf32>
    %1388 = stablehlo.add %1387, %cst_17 : tensor<1x197x3072xf32>
    %1389 = stablehlo.multiply %1388, %1370 : tensor<1x197x3072xf32>
    %1390 = stablehlo.add %1389, %cst_18 : tensor<1x197x3072xf32>
    %1391 = stablehlo.multiply %1369, %1382 : tensor<1x197x3072xf32>
    %1392 = stablehlo.divide %1391, %1390 : tensor<1x197x3072xf32>
    %1393 = stablehlo.clamp %cst_19, %1392, %cst_20 : tensor<1x197x3072xf32>
    %1394 = stablehlo.convert %1393 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %1395 = stablehlo.add %1394, %cst_2 : tensor<1x197x3072xbf16>
    %1396 = stablehlo.multiply %1395, %1366 : tensor<1x197x3072xbf16>
    %1397 = stablehlo.reshape %1396 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %1398 = stablehlo.convert %1397 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %1399 = stablehlo.dot_general %1398, %arg160, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %1400 = stablehlo.broadcast_in_dim %1399, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1401 = stablehlo.multiply %1400, %60 : tensor<197x768xf32>
    %1402 = stablehlo.broadcast_in_dim %1401, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1403 = stablehlo.broadcast_in_dim %arg161, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1404 = stablehlo.add %1402, %1403 : tensor<197x768xf32>
    %1405 = stablehlo.convert %1404 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1406 = stablehlo.reshape %1405 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1407 = stablehlo.broadcast_in_dim %arg44, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1408 = stablehlo.broadcast_in_dim %1406, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1409 = stablehlo.multiply %1407, %1408 : tensor<1x197x768xbf16>
    %1410 = stablehlo.add %1409, %1318 : tensor<1x197x768xbf16>
    %1411 = stablehlo.convert %1410 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1412 = stablehlo.convert %1411 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1413 = stablehlo.reduce(%1412 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1414 = stablehlo.reshape %1413 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1416 = stablehlo.divide %1415, %15 : tensor<1x197x1xf64>
    %1417 = stablehlo.broadcast_in_dim %1412, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1418 = stablehlo.broadcast_in_dim %1416, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1419 = stablehlo.subtract %1417, %1418 : tensor<1x197x768xf64>
    %1420 = stablehlo.multiply %1419, %1419 : tensor<1x197x768xf64>
    %1421 = stablehlo.reduce(%1420 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1422 = stablehlo.reshape %1421 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1424 = stablehlo.divide %1423, %15 : tensor<1x197x1xf64>
    %1425 = stablehlo.convert %1424 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1426 = stablehlo.reduce(%1411 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1427 = stablehlo.reshape %1426 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1429 = stablehlo.divide %1428, %31 : tensor<1x197x1xf32>
    %1430 = stablehlo.broadcast_in_dim %1425, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1431 = stablehlo.add %1430, %36 : tensor<1x197x1xf32>
    %1432 = stablehlo.rsqrt %1431 : tensor<1x197x1xf32>
    %1433 = stablehlo.broadcast_in_dim %1411, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1434 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1435 = stablehlo.subtract %1433, %1434 : tensor<1x197x768xf32>
    %1436 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1437 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1438 = stablehlo.multiply %1436, %1437 : tensor<1x197x768xf32>
    %1439 = stablehlo.convert %arg45 : (tensor<768xbf16>) -> tensor<768xf32>
    %1440 = stablehlo.broadcast_in_dim %1438, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1441 = stablehlo.broadcast_in_dim %1439, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1442 = stablehlo.multiply %1440, %1441 : tensor<1x197x768xf32>
    %1443 = stablehlo.convert %arg46 : (tensor<768xbf16>) -> tensor<768xf32>
    %1444 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1445 = stablehlo.broadcast_in_dim %1443, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1446 = stablehlo.add %1444, %1445 : tensor<1x197x768xf32>
    %1447 = stablehlo.convert %1446 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1448 = stablehlo.reshape %1447 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1449 = stablehlo.convert %1448 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1450 = stablehlo.dot_general %1449, %arg162, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1451 = stablehlo.broadcast_in_dim %1450, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1452 = stablehlo.multiply %1451, %60 : tensor<197x768xf32>
    %1453 = stablehlo.broadcast_in_dim %1452, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1454 = stablehlo.broadcast_in_dim %arg163, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1455 = stablehlo.add %1453, %1454 : tensor<197x768xf32>
    %1456 = stablehlo.convert %1455 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1457 = stablehlo.reshape %1456 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1458 = stablehlo.dot_general %1448, %arg164, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %1459 = stablehlo.reshape %1458 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1460 = stablehlo.reshape %1459 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1461 = stablehlo.transpose %1460, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1462 = stablehlo.dot_general %1449, %arg165, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1464 = stablehlo.multiply %1463, %60 : tensor<197x768xf32>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1466 = stablehlo.broadcast_in_dim %arg166, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1467 = stablehlo.add %1465, %1466 : tensor<197x768xf32>
    %1468 = stablehlo.convert %1467 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1469 = stablehlo.reshape %1468 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1470 = stablehlo.reshape %1469 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1471 = stablehlo.transpose %1470, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1472 = stablehlo.reshape %1457 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1473 = stablehlo.transpose %1472, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1474 = stablehlo.transpose %1461, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %1475 = stablehlo.reshape %1473 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1476 = stablehlo.reshape %1474 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1477 = stablehlo.broadcast_in_dim %1476, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1478 = stablehlo.dot_general %1475, %1477, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %1479 = stablehlo.reshape %1478 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1480 = stablehlo.broadcast_in_dim %1479, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1481 = stablehlo.divide %1480, %92 : tensor<1x12x197x197xbf16>
    %1482 = stablehlo.add %1481, %arg167 : tensor<1x12x197x197xbf16>
    %1483 = stablehlo.convert %1482 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %1484 = stablehlo.reduce(%1483 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1485 = stablehlo.reshape %1484 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1486 = stablehlo.broadcast_in_dim %1483, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1487 = stablehlo.broadcast_in_dim %1485, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1488 = stablehlo.subtract %1486, %1487 : tensor<1x12x197x197xf32>
    %1489 = stablehlo.exponential %1488 : tensor<1x12x197x197xf32>
    %1490 = stablehlo.reduce(%1489 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1491 = stablehlo.reshape %1490 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1492 = stablehlo.broadcast_in_dim %1489, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1493 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1494 = stablehlo.divide %1492, %1493 : tensor<1x12x197x197xf32>
    %1495 = stablehlo.convert %1494 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %1496 = stablehlo.reshape %1495 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %1497 = stablehlo.reshape %1471 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1498 = stablehlo.broadcast_in_dim %1497, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1499 = stablehlo.dot_general %1496, %1498, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1500 = stablehlo.reshape %1499 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1501 = stablehlo.transpose %1500, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %1502 = stablehlo.reshape %1501 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %1503 = stablehlo.reshape %1502 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1504 = stablehlo.convert %1503 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1505 = stablehlo.dot_general %1504, %arg168, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1506 = stablehlo.broadcast_in_dim %1505, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1507 = stablehlo.multiply %1506, %60 : tensor<197x768xf32>
    %1508 = stablehlo.broadcast_in_dim %1507, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1509 = stablehlo.broadcast_in_dim %arg169, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1510 = stablehlo.add %1508, %1509 : tensor<197x768xf32>
    %1511 = stablehlo.convert %1510 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1512 = stablehlo.reshape %1511 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1513 = stablehlo.broadcast_in_dim %arg47, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1514 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1515 = stablehlo.multiply %1513, %1514 : tensor<1x197x768xbf16>
    %1516 = stablehlo.add %1515, %1410 : tensor<1x197x768xbf16>
    %1517 = stablehlo.convert %1516 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1518 = stablehlo.convert %1517 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1519 = stablehlo.reduce(%1518 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1520 = stablehlo.reshape %1519 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1521 = stablehlo.broadcast_in_dim %1520, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1522 = stablehlo.divide %1521, %15 : tensor<1x197x1xf64>
    %1523 = stablehlo.broadcast_in_dim %1518, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1524 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1525 = stablehlo.subtract %1523, %1524 : tensor<1x197x768xf64>
    %1526 = stablehlo.multiply %1525, %1525 : tensor<1x197x768xf64>
    %1527 = stablehlo.reduce(%1526 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1528 = stablehlo.reshape %1527 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1529 = stablehlo.broadcast_in_dim %1528, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1530 = stablehlo.divide %1529, %15 : tensor<1x197x1xf64>
    %1531 = stablehlo.convert %1530 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1532 = stablehlo.reduce(%1517 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1533 = stablehlo.reshape %1532 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1534 = stablehlo.broadcast_in_dim %1533, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1535 = stablehlo.divide %1534, %31 : tensor<1x197x1xf32>
    %1536 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1537 = stablehlo.add %1536, %36 : tensor<1x197x1xf32>
    %1538 = stablehlo.rsqrt %1537 : tensor<1x197x1xf32>
    %1539 = stablehlo.broadcast_in_dim %1517, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1540 = stablehlo.broadcast_in_dim %1535, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1541 = stablehlo.subtract %1539, %1540 : tensor<1x197x768xf32>
    %1542 = stablehlo.broadcast_in_dim %1541, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1543 = stablehlo.broadcast_in_dim %1538, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1544 = stablehlo.multiply %1542, %1543 : tensor<1x197x768xf32>
    %1545 = stablehlo.convert %arg48 : (tensor<768xbf16>) -> tensor<768xf32>
    %1546 = stablehlo.broadcast_in_dim %1544, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1547 = stablehlo.broadcast_in_dim %1545, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1548 = stablehlo.multiply %1546, %1547 : tensor<1x197x768xf32>
    %1549 = stablehlo.convert %arg49 : (tensor<768xbf16>) -> tensor<768xf32>
    %1550 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1551 = stablehlo.broadcast_in_dim %1549, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1552 = stablehlo.add %1550, %1551 : tensor<1x197x768xf32>
    %1553 = stablehlo.convert %1552 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1554 = stablehlo.reshape %1553 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1555 = stablehlo.convert %1554 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1556 = stablehlo.dot_general %1555, %arg170, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %1557 = stablehlo.broadcast_in_dim %1556, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1558 = stablehlo.multiply %1557, %170 : tensor<197x3072xf32>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1560 = stablehlo.broadcast_in_dim %arg171, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %1561 = stablehlo.add %1559, %1560 : tensor<197x3072xf32>
    %1562 = stablehlo.convert %1561 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %1563 = stablehlo.reshape %1562 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %1564 = stablehlo.multiply %1563, %cst_4 : tensor<1x197x3072xbf16>
    %1565 = stablehlo.multiply %1563, %178 : tensor<1x197x3072xbf16>
    %1566 = stablehlo.convert %1565 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %1567 = stablehlo.clamp %cst_5, %1566, %cst_6 : tensor<1x197x3072xf32>
    %1568 = stablehlo.multiply %1567, %1567 : tensor<1x197x3072xf32>
    %1569 = stablehlo.multiply %cst_7, %1568 : tensor<1x197x3072xf32>
    %1570 = stablehlo.add %1569, %cst_8 : tensor<1x197x3072xf32>
    %1571 = stablehlo.multiply %1570, %1568 : tensor<1x197x3072xf32>
    %1572 = stablehlo.add %1571, %cst_9 : tensor<1x197x3072xf32>
    %1573 = stablehlo.multiply %1572, %1568 : tensor<1x197x3072xf32>
    %1574 = stablehlo.add %1573, %cst_10 : tensor<1x197x3072xf32>
    %1575 = stablehlo.multiply %1574, %1568 : tensor<1x197x3072xf32>
    %1576 = stablehlo.add %1575, %cst_11 : tensor<1x197x3072xf32>
    %1577 = stablehlo.multiply %1576, %1568 : tensor<1x197x3072xf32>
    %1578 = stablehlo.add %1577, %cst_12 : tensor<1x197x3072xf32>
    %1579 = stablehlo.multiply %1578, %1568 : tensor<1x197x3072xf32>
    %1580 = stablehlo.add %1579, %cst_13 : tensor<1x197x3072xf32>
    %1581 = stablehlo.multiply %cst_14, %1568 : tensor<1x197x3072xf32>
    %1582 = stablehlo.add %1581, %cst_15 : tensor<1x197x3072xf32>
    %1583 = stablehlo.multiply %1582, %1568 : tensor<1x197x3072xf32>
    %1584 = stablehlo.add %1583, %cst_16 : tensor<1x197x3072xf32>
    %1585 = stablehlo.multiply %1584, %1568 : tensor<1x197x3072xf32>
    %1586 = stablehlo.add %1585, %cst_17 : tensor<1x197x3072xf32>
    %1587 = stablehlo.multiply %1586, %1568 : tensor<1x197x3072xf32>
    %1588 = stablehlo.add %1587, %cst_18 : tensor<1x197x3072xf32>
    %1589 = stablehlo.multiply %1567, %1580 : tensor<1x197x3072xf32>
    %1590 = stablehlo.divide %1589, %1588 : tensor<1x197x3072xf32>
    %1591 = stablehlo.clamp %cst_19, %1590, %cst_20 : tensor<1x197x3072xf32>
    %1592 = stablehlo.convert %1591 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %1593 = stablehlo.add %1592, %cst_2 : tensor<1x197x3072xbf16>
    %1594 = stablehlo.multiply %1593, %1564 : tensor<1x197x3072xbf16>
    %1595 = stablehlo.reshape %1594 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %1596 = stablehlo.convert %1595 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %1597 = stablehlo.dot_general %1596, %arg172, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %1598 = stablehlo.broadcast_in_dim %1597, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1599 = stablehlo.multiply %1598, %60 : tensor<197x768xf32>
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1601 = stablehlo.broadcast_in_dim %arg173, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1602 = stablehlo.add %1600, %1601 : tensor<197x768xf32>
    %1603 = stablehlo.convert %1602 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1604 = stablehlo.reshape %1603 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1605 = stablehlo.broadcast_in_dim %arg50, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1606 = stablehlo.broadcast_in_dim %1604, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1607 = stablehlo.multiply %1605, %1606 : tensor<1x197x768xbf16>
    %1608 = stablehlo.add %1607, %1516 : tensor<1x197x768xbf16>
    %1609 = stablehlo.convert %1608 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1610 = stablehlo.convert %1609 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1611 = stablehlo.reduce(%1610 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1612 = stablehlo.reshape %1611 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1613 = stablehlo.broadcast_in_dim %1612, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1614 = stablehlo.divide %1613, %15 : tensor<1x197x1xf64>
    %1615 = stablehlo.broadcast_in_dim %1610, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1616 = stablehlo.broadcast_in_dim %1614, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1617 = stablehlo.subtract %1615, %1616 : tensor<1x197x768xf64>
    %1618 = stablehlo.multiply %1617, %1617 : tensor<1x197x768xf64>
    %1619 = stablehlo.reduce(%1618 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1620 = stablehlo.reshape %1619 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1621 = stablehlo.broadcast_in_dim %1620, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1622 = stablehlo.divide %1621, %15 : tensor<1x197x1xf64>
    %1623 = stablehlo.convert %1622 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1624 = stablehlo.reduce(%1609 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1625 = stablehlo.reshape %1624 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1626 = stablehlo.broadcast_in_dim %1625, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1627 = stablehlo.divide %1626, %31 : tensor<1x197x1xf32>
    %1628 = stablehlo.broadcast_in_dim %1623, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1629 = stablehlo.add %1628, %36 : tensor<1x197x1xf32>
    %1630 = stablehlo.rsqrt %1629 : tensor<1x197x1xf32>
    %1631 = stablehlo.broadcast_in_dim %1609, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1632 = stablehlo.broadcast_in_dim %1627, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1633 = stablehlo.subtract %1631, %1632 : tensor<1x197x768xf32>
    %1634 = stablehlo.broadcast_in_dim %1633, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1635 = stablehlo.broadcast_in_dim %1630, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1636 = stablehlo.multiply %1634, %1635 : tensor<1x197x768xf32>
    %1637 = stablehlo.convert %arg51 : (tensor<768xbf16>) -> tensor<768xf32>
    %1638 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1639 = stablehlo.broadcast_in_dim %1637, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1640 = stablehlo.multiply %1638, %1639 : tensor<1x197x768xf32>
    %1641 = stablehlo.convert %arg52 : (tensor<768xbf16>) -> tensor<768xf32>
    %1642 = stablehlo.broadcast_in_dim %1640, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1643 = stablehlo.broadcast_in_dim %1641, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1644 = stablehlo.add %1642, %1643 : tensor<1x197x768xf32>
    %1645 = stablehlo.convert %1644 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1646 = stablehlo.reshape %1645 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1647 = stablehlo.convert %1646 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1648 = stablehlo.dot_general %1647, %arg174, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1649 = stablehlo.broadcast_in_dim %1648, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1650 = stablehlo.multiply %1649, %60 : tensor<197x768xf32>
    %1651 = stablehlo.broadcast_in_dim %1650, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1652 = stablehlo.broadcast_in_dim %arg175, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1653 = stablehlo.add %1651, %1652 : tensor<197x768xf32>
    %1654 = stablehlo.convert %1653 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1655 = stablehlo.reshape %1654 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1656 = stablehlo.dot_general %1646, %arg176, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %1657 = stablehlo.reshape %1656 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1658 = stablehlo.reshape %1657 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1659 = stablehlo.transpose %1658, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1660 = stablehlo.dot_general %1647, %arg177, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1661 = stablehlo.broadcast_in_dim %1660, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1662 = stablehlo.multiply %1661, %60 : tensor<197x768xf32>
    %1663 = stablehlo.broadcast_in_dim %1662, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1664 = stablehlo.broadcast_in_dim %arg178, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1665 = stablehlo.add %1663, %1664 : tensor<197x768xf32>
    %1666 = stablehlo.convert %1665 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1667 = stablehlo.reshape %1666 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1668 = stablehlo.reshape %1667 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1669 = stablehlo.transpose %1668, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1670 = stablehlo.reshape %1655 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1671 = stablehlo.transpose %1670, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1672 = stablehlo.transpose %1659, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %1673 = stablehlo.reshape %1671 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1674 = stablehlo.reshape %1672 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1676 = stablehlo.dot_general %1673, %1675, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %1677 = stablehlo.reshape %1676 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1678 = stablehlo.broadcast_in_dim %1677, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1679 = stablehlo.divide %1678, %92 : tensor<1x12x197x197xbf16>
    %1680 = stablehlo.add %1679, %arg179 : tensor<1x12x197x197xbf16>
    %1681 = stablehlo.convert %1680 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %1682 = stablehlo.reduce(%1681 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1683 = stablehlo.reshape %1682 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1684 = stablehlo.broadcast_in_dim %1681, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1685 = stablehlo.broadcast_in_dim %1683, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1686 = stablehlo.subtract %1684, %1685 : tensor<1x12x197x197xf32>
    %1687 = stablehlo.exponential %1686 : tensor<1x12x197x197xf32>
    %1688 = stablehlo.reduce(%1687 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1689 = stablehlo.reshape %1688 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1690 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1691 = stablehlo.broadcast_in_dim %1689, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1692 = stablehlo.divide %1690, %1691 : tensor<1x12x197x197xf32>
    %1693 = stablehlo.convert %1692 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %1694 = stablehlo.reshape %1693 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %1695 = stablehlo.reshape %1669 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1697 = stablehlo.dot_general %1694, %1696, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1698 = stablehlo.reshape %1697 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1699 = stablehlo.transpose %1698, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %1700 = stablehlo.reshape %1699 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %1701 = stablehlo.reshape %1700 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1702 = stablehlo.convert %1701 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1703 = stablehlo.dot_general %1702, %arg180, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1704 = stablehlo.broadcast_in_dim %1703, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1705 = stablehlo.multiply %1704, %60 : tensor<197x768xf32>
    %1706 = stablehlo.broadcast_in_dim %1705, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1707 = stablehlo.broadcast_in_dim %arg181, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1708 = stablehlo.add %1706, %1707 : tensor<197x768xf32>
    %1709 = stablehlo.convert %1708 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1710 = stablehlo.reshape %1709 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1711 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1712 = stablehlo.broadcast_in_dim %1710, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1713 = stablehlo.multiply %1711, %1712 : tensor<1x197x768xbf16>
    %1714 = stablehlo.add %1713, %1608 : tensor<1x197x768xbf16>
    %1715 = stablehlo.convert %1714 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1716 = stablehlo.convert %1715 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1717 = stablehlo.reduce(%1716 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1718 = stablehlo.reshape %1717 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1719 = stablehlo.broadcast_in_dim %1718, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1720 = stablehlo.divide %1719, %15 : tensor<1x197x1xf64>
    %1721 = stablehlo.broadcast_in_dim %1716, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1722 = stablehlo.broadcast_in_dim %1720, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1723 = stablehlo.subtract %1721, %1722 : tensor<1x197x768xf64>
    %1724 = stablehlo.multiply %1723, %1723 : tensor<1x197x768xf64>
    %1725 = stablehlo.reduce(%1724 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1726 = stablehlo.reshape %1725 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1727 = stablehlo.broadcast_in_dim %1726, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1728 = stablehlo.divide %1727, %15 : tensor<1x197x1xf64>
    %1729 = stablehlo.convert %1728 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1730 = stablehlo.reduce(%1715 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1731 = stablehlo.reshape %1730 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1733 = stablehlo.divide %1732, %31 : tensor<1x197x1xf32>
    %1734 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1735 = stablehlo.add %1734, %36 : tensor<1x197x1xf32>
    %1736 = stablehlo.rsqrt %1735 : tensor<1x197x1xf32>
    %1737 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1738 = stablehlo.broadcast_in_dim %1733, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1739 = stablehlo.subtract %1737, %1738 : tensor<1x197x768xf32>
    %1740 = stablehlo.broadcast_in_dim %1739, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1741 = stablehlo.broadcast_in_dim %1736, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1742 = stablehlo.multiply %1740, %1741 : tensor<1x197x768xf32>
    %1743 = stablehlo.convert %arg54 : (tensor<768xbf16>) -> tensor<768xf32>
    %1744 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1745 = stablehlo.broadcast_in_dim %1743, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1746 = stablehlo.multiply %1744, %1745 : tensor<1x197x768xf32>
    %1747 = stablehlo.convert %arg55 : (tensor<768xbf16>) -> tensor<768xf32>
    %1748 = stablehlo.broadcast_in_dim %1746, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1749 = stablehlo.broadcast_in_dim %1747, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1750 = stablehlo.add %1748, %1749 : tensor<1x197x768xf32>
    %1751 = stablehlo.convert %1750 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1752 = stablehlo.reshape %1751 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1753 = stablehlo.convert %1752 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1754 = stablehlo.dot_general %1753, %arg182, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %1755 = stablehlo.broadcast_in_dim %1754, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1756 = stablehlo.multiply %1755, %170 : tensor<197x3072xf32>
    %1757 = stablehlo.broadcast_in_dim %1756, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1758 = stablehlo.broadcast_in_dim %arg183, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %1759 = stablehlo.add %1757, %1758 : tensor<197x3072xf32>
    %1760 = stablehlo.convert %1759 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %1761 = stablehlo.reshape %1760 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %1762 = stablehlo.multiply %1761, %cst_4 : tensor<1x197x3072xbf16>
    %1763 = stablehlo.multiply %1761, %178 : tensor<1x197x3072xbf16>
    %1764 = stablehlo.convert %1763 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %1765 = stablehlo.clamp %cst_5, %1764, %cst_6 : tensor<1x197x3072xf32>
    %1766 = stablehlo.multiply %1765, %1765 : tensor<1x197x3072xf32>
    %1767 = stablehlo.multiply %cst_7, %1766 : tensor<1x197x3072xf32>
    %1768 = stablehlo.add %1767, %cst_8 : tensor<1x197x3072xf32>
    %1769 = stablehlo.multiply %1768, %1766 : tensor<1x197x3072xf32>
    %1770 = stablehlo.add %1769, %cst_9 : tensor<1x197x3072xf32>
    %1771 = stablehlo.multiply %1770, %1766 : tensor<1x197x3072xf32>
    %1772 = stablehlo.add %1771, %cst_10 : tensor<1x197x3072xf32>
    %1773 = stablehlo.multiply %1772, %1766 : tensor<1x197x3072xf32>
    %1774 = stablehlo.add %1773, %cst_11 : tensor<1x197x3072xf32>
    %1775 = stablehlo.multiply %1774, %1766 : tensor<1x197x3072xf32>
    %1776 = stablehlo.add %1775, %cst_12 : tensor<1x197x3072xf32>
    %1777 = stablehlo.multiply %1776, %1766 : tensor<1x197x3072xf32>
    %1778 = stablehlo.add %1777, %cst_13 : tensor<1x197x3072xf32>
    %1779 = stablehlo.multiply %cst_14, %1766 : tensor<1x197x3072xf32>
    %1780 = stablehlo.add %1779, %cst_15 : tensor<1x197x3072xf32>
    %1781 = stablehlo.multiply %1780, %1766 : tensor<1x197x3072xf32>
    %1782 = stablehlo.add %1781, %cst_16 : tensor<1x197x3072xf32>
    %1783 = stablehlo.multiply %1782, %1766 : tensor<1x197x3072xf32>
    %1784 = stablehlo.add %1783, %cst_17 : tensor<1x197x3072xf32>
    %1785 = stablehlo.multiply %1784, %1766 : tensor<1x197x3072xf32>
    %1786 = stablehlo.add %1785, %cst_18 : tensor<1x197x3072xf32>
    %1787 = stablehlo.multiply %1765, %1778 : tensor<1x197x3072xf32>
    %1788 = stablehlo.divide %1787, %1786 : tensor<1x197x3072xf32>
    %1789 = stablehlo.clamp %cst_19, %1788, %cst_20 : tensor<1x197x3072xf32>
    %1790 = stablehlo.convert %1789 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %1791 = stablehlo.add %1790, %cst_2 : tensor<1x197x3072xbf16>
    %1792 = stablehlo.multiply %1791, %1762 : tensor<1x197x3072xbf16>
    %1793 = stablehlo.reshape %1792 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %1794 = stablehlo.convert %1793 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %1795 = stablehlo.dot_general %1794, %arg184, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1797 = stablehlo.multiply %1796, %60 : tensor<197x768xf32>
    %1798 = stablehlo.broadcast_in_dim %1797, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1799 = stablehlo.broadcast_in_dim %arg185, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1800 = stablehlo.add %1798, %1799 : tensor<197x768xf32>
    %1801 = stablehlo.convert %1800 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1802 = stablehlo.reshape %1801 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1803 = stablehlo.broadcast_in_dim %arg56, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1804 = stablehlo.broadcast_in_dim %1802, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1805 = stablehlo.multiply %1803, %1804 : tensor<1x197x768xbf16>
    %1806 = stablehlo.add %1805, %1714 : tensor<1x197x768xbf16>
    %1807 = stablehlo.convert %1806 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1808 = stablehlo.convert %1807 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1809 = stablehlo.reduce(%1808 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1810 = stablehlo.reshape %1809 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1811 = stablehlo.broadcast_in_dim %1810, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1812 = stablehlo.divide %1811, %15 : tensor<1x197x1xf64>
    %1813 = stablehlo.broadcast_in_dim %1808, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1814 = stablehlo.broadcast_in_dim %1812, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1815 = stablehlo.subtract %1813, %1814 : tensor<1x197x768xf64>
    %1816 = stablehlo.multiply %1815, %1815 : tensor<1x197x768xf64>
    %1817 = stablehlo.reduce(%1816 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1818 = stablehlo.reshape %1817 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1820 = stablehlo.divide %1819, %15 : tensor<1x197x1xf64>
    %1821 = stablehlo.convert %1820 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1822 = stablehlo.reduce(%1807 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1823 = stablehlo.reshape %1822 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1824 = stablehlo.broadcast_in_dim %1823, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1825 = stablehlo.divide %1824, %31 : tensor<1x197x1xf32>
    %1826 = stablehlo.broadcast_in_dim %1821, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1827 = stablehlo.add %1826, %36 : tensor<1x197x1xf32>
    %1828 = stablehlo.rsqrt %1827 : tensor<1x197x1xf32>
    %1829 = stablehlo.broadcast_in_dim %1807, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1830 = stablehlo.broadcast_in_dim %1825, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1831 = stablehlo.subtract %1829, %1830 : tensor<1x197x768xf32>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1833 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1834 = stablehlo.multiply %1832, %1833 : tensor<1x197x768xf32>
    %1835 = stablehlo.convert %arg57 : (tensor<768xbf16>) -> tensor<768xf32>
    %1836 = stablehlo.broadcast_in_dim %1834, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1837 = stablehlo.broadcast_in_dim %1835, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1838 = stablehlo.multiply %1836, %1837 : tensor<1x197x768xf32>
    %1839 = stablehlo.convert %arg58 : (tensor<768xbf16>) -> tensor<768xf32>
    %1840 = stablehlo.broadcast_in_dim %1838, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1841 = stablehlo.broadcast_in_dim %1839, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1842 = stablehlo.add %1840, %1841 : tensor<1x197x768xf32>
    %1843 = stablehlo.convert %1842 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1844 = stablehlo.reshape %1843 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1845 = stablehlo.convert %1844 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1846 = stablehlo.dot_general %1845, %arg186, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1847 = stablehlo.broadcast_in_dim %1846, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1848 = stablehlo.multiply %1847, %60 : tensor<197x768xf32>
    %1849 = stablehlo.broadcast_in_dim %1848, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1850 = stablehlo.broadcast_in_dim %arg187, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1851 = stablehlo.add %1849, %1850 : tensor<197x768xf32>
    %1852 = stablehlo.convert %1851 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1853 = stablehlo.reshape %1852 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1854 = stablehlo.dot_general %1844, %arg188, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %1855 = stablehlo.reshape %1854 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1856 = stablehlo.reshape %1855 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1857 = stablehlo.transpose %1856, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1858 = stablehlo.dot_general %1845, %arg189, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1859 = stablehlo.broadcast_in_dim %1858, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1860 = stablehlo.multiply %1859, %60 : tensor<197x768xf32>
    %1861 = stablehlo.broadcast_in_dim %1860, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1862 = stablehlo.broadcast_in_dim %arg190, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1863 = stablehlo.add %1861, %1862 : tensor<197x768xf32>
    %1864 = stablehlo.convert %1863 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1865 = stablehlo.reshape %1864 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1866 = stablehlo.reshape %1865 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1867 = stablehlo.transpose %1866, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1868 = stablehlo.reshape %1853 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %1869 = stablehlo.transpose %1868, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1870 = stablehlo.transpose %1857, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %1871 = stablehlo.reshape %1869 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1872 = stablehlo.reshape %1870 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1873 = stablehlo.broadcast_in_dim %1872, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %1874 = stablehlo.dot_general %1871, %1873, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %1875 = stablehlo.reshape %1874 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1876 = stablehlo.broadcast_in_dim %1875, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %1877 = stablehlo.divide %1876, %92 : tensor<1x12x197x197xbf16>
    %1878 = stablehlo.add %1877, %arg191 : tensor<1x12x197x197xbf16>
    %1879 = stablehlo.convert %1878 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %1880 = stablehlo.reduce(%1879 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1881 = stablehlo.reshape %1880 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1882 = stablehlo.broadcast_in_dim %1879, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1883 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1884 = stablehlo.subtract %1882, %1883 : tensor<1x12x197x197xf32>
    %1885 = stablehlo.exponential %1884 : tensor<1x12x197x197xf32>
    %1886 = stablehlo.reduce(%1885 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %1887 = stablehlo.reshape %1886 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %1888 = stablehlo.broadcast_in_dim %1885, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1889 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %1890 = stablehlo.divide %1888, %1889 : tensor<1x12x197x197xf32>
    %1891 = stablehlo.convert %1890 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %1892 = stablehlo.reshape %1891 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %1893 = stablehlo.reshape %1867 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1894 = stablehlo.broadcast_in_dim %1893, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1895 = stablehlo.dot_general %1892, %1894, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %1896 = stablehlo.reshape %1895 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %1897 = stablehlo.transpose %1896, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %1898 = stablehlo.reshape %1897 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %1899 = stablehlo.reshape %1898 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1900 = stablehlo.convert %1899 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1901 = stablehlo.dot_general %1900, %arg192, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %1902 = stablehlo.broadcast_in_dim %1901, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1903 = stablehlo.multiply %1902, %60 : tensor<197x768xf32>
    %1904 = stablehlo.broadcast_in_dim %1903, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1905 = stablehlo.broadcast_in_dim %arg193, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1906 = stablehlo.add %1904, %1905 : tensor<197x768xf32>
    %1907 = stablehlo.convert %1906 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %1908 = stablehlo.reshape %1907 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %1909 = stablehlo.broadcast_in_dim %arg59, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %1910 = stablehlo.broadcast_in_dim %1908, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %1911 = stablehlo.multiply %1909, %1910 : tensor<1x197x768xbf16>
    %1912 = stablehlo.add %1911, %1806 : tensor<1x197x768xbf16>
    %1913 = stablehlo.convert %1912 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %1914 = stablehlo.convert %1913 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %1915 = stablehlo.reduce(%1914 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1916 = stablehlo.reshape %1915 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1917 = stablehlo.broadcast_in_dim %1916, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1918 = stablehlo.divide %1917, %15 : tensor<1x197x1xf64>
    %1919 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %1920 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %1921 = stablehlo.subtract %1919, %1920 : tensor<1x197x768xf64>
    %1922 = stablehlo.multiply %1921, %1921 : tensor<1x197x768xf64>
    %1923 = stablehlo.reduce(%1922 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1924 = stablehlo.reshape %1923 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1925 = stablehlo.broadcast_in_dim %1924, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1926 = stablehlo.divide %1925, %15 : tensor<1x197x1xf64>
    %1927 = stablehlo.convert %1926 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1928 = stablehlo.reduce(%1913 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1929 = stablehlo.reshape %1928 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1930 = stablehlo.broadcast_in_dim %1929, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1931 = stablehlo.divide %1930, %31 : tensor<1x197x1xf32>
    %1932 = stablehlo.broadcast_in_dim %1927, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1933 = stablehlo.add %1932, %36 : tensor<1x197x1xf32>
    %1934 = stablehlo.rsqrt %1933 : tensor<1x197x1xf32>
    %1935 = stablehlo.broadcast_in_dim %1913, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1936 = stablehlo.broadcast_in_dim %1931, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1937 = stablehlo.subtract %1935, %1936 : tensor<1x197x768xf32>
    %1938 = stablehlo.broadcast_in_dim %1937, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1939 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %1940 = stablehlo.multiply %1938, %1939 : tensor<1x197x768xf32>
    %1941 = stablehlo.convert %arg60 : (tensor<768xbf16>) -> tensor<768xf32>
    %1942 = stablehlo.broadcast_in_dim %1940, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1943 = stablehlo.broadcast_in_dim %1941, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1944 = stablehlo.multiply %1942, %1943 : tensor<1x197x768xf32>
    %1945 = stablehlo.convert %arg61 : (tensor<768xbf16>) -> tensor<768xf32>
    %1946 = stablehlo.broadcast_in_dim %1944, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1947 = stablehlo.broadcast_in_dim %1945, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %1948 = stablehlo.add %1946, %1947 : tensor<1x197x768xf32>
    %1949 = stablehlo.convert %1948 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %1950 = stablehlo.reshape %1949 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %1951 = stablehlo.convert %1950 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %1952 = stablehlo.dot_general %1951, %arg194, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %1953 = stablehlo.broadcast_in_dim %1952, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1954 = stablehlo.multiply %1953, %170 : tensor<197x3072xf32>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %1956 = stablehlo.broadcast_in_dim %arg195, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %1957 = stablehlo.add %1955, %1956 : tensor<197x3072xf32>
    %1958 = stablehlo.convert %1957 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %1959 = stablehlo.reshape %1958 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %1960 = stablehlo.multiply %1959, %cst_4 : tensor<1x197x3072xbf16>
    %1961 = stablehlo.multiply %1959, %178 : tensor<1x197x3072xbf16>
    %1962 = stablehlo.convert %1961 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %1963 = stablehlo.clamp %cst_5, %1962, %cst_6 : tensor<1x197x3072xf32>
    %1964 = stablehlo.multiply %1963, %1963 : tensor<1x197x3072xf32>
    %1965 = stablehlo.multiply %cst_7, %1964 : tensor<1x197x3072xf32>
    %1966 = stablehlo.add %1965, %cst_8 : tensor<1x197x3072xf32>
    %1967 = stablehlo.multiply %1966, %1964 : tensor<1x197x3072xf32>
    %1968 = stablehlo.add %1967, %cst_9 : tensor<1x197x3072xf32>
    %1969 = stablehlo.multiply %1968, %1964 : tensor<1x197x3072xf32>
    %1970 = stablehlo.add %1969, %cst_10 : tensor<1x197x3072xf32>
    %1971 = stablehlo.multiply %1970, %1964 : tensor<1x197x3072xf32>
    %1972 = stablehlo.add %1971, %cst_11 : tensor<1x197x3072xf32>
    %1973 = stablehlo.multiply %1972, %1964 : tensor<1x197x3072xf32>
    %1974 = stablehlo.add %1973, %cst_12 : tensor<1x197x3072xf32>
    %1975 = stablehlo.multiply %1974, %1964 : tensor<1x197x3072xf32>
    %1976 = stablehlo.add %1975, %cst_13 : tensor<1x197x3072xf32>
    %1977 = stablehlo.multiply %cst_14, %1964 : tensor<1x197x3072xf32>
    %1978 = stablehlo.add %1977, %cst_15 : tensor<1x197x3072xf32>
    %1979 = stablehlo.multiply %1978, %1964 : tensor<1x197x3072xf32>
    %1980 = stablehlo.add %1979, %cst_16 : tensor<1x197x3072xf32>
    %1981 = stablehlo.multiply %1980, %1964 : tensor<1x197x3072xf32>
    %1982 = stablehlo.add %1981, %cst_17 : tensor<1x197x3072xf32>
    %1983 = stablehlo.multiply %1982, %1964 : tensor<1x197x3072xf32>
    %1984 = stablehlo.add %1983, %cst_18 : tensor<1x197x3072xf32>
    %1985 = stablehlo.multiply %1963, %1976 : tensor<1x197x3072xf32>
    %1986 = stablehlo.divide %1985, %1984 : tensor<1x197x3072xf32>
    %1987 = stablehlo.clamp %cst_19, %1986, %cst_20 : tensor<1x197x3072xf32>
    %1988 = stablehlo.convert %1987 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %1989 = stablehlo.add %1988, %cst_2 : tensor<1x197x3072xbf16>
    %1990 = stablehlo.multiply %1989, %1960 : tensor<1x197x3072xbf16>
    %1991 = stablehlo.reshape %1990 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %1992 = stablehlo.convert %1991 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %1993 = stablehlo.dot_general %1992, %arg196, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1995 = stablehlo.multiply %1994, %60 : tensor<197x768xf32>
    %1996 = stablehlo.broadcast_in_dim %1995, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %1997 = stablehlo.broadcast_in_dim %arg197, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1998 = stablehlo.add %1996, %1997 : tensor<197x768xf32>
    %1999 = stablehlo.convert %1998 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2000 = stablehlo.reshape %1999 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2001 = stablehlo.broadcast_in_dim %arg62, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %2002 = stablehlo.broadcast_in_dim %2000, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %2003 = stablehlo.multiply %2001, %2002 : tensor<1x197x768xbf16>
    %2004 = stablehlo.add %2003, %1912 : tensor<1x197x768xbf16>
    %2005 = stablehlo.convert %2004 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %2006 = stablehlo.convert %2005 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %2007 = stablehlo.reduce(%2006 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2008 = stablehlo.reshape %2007 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2009 = stablehlo.broadcast_in_dim %2008, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2010 = stablehlo.divide %2009, %15 : tensor<1x197x1xf64>
    %2011 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %2012 = stablehlo.broadcast_in_dim %2010, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %2013 = stablehlo.subtract %2011, %2012 : tensor<1x197x768xf64>
    %2014 = stablehlo.multiply %2013, %2013 : tensor<1x197x768xf64>
    %2015 = stablehlo.reduce(%2014 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2016 = stablehlo.reshape %2015 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2017 = stablehlo.broadcast_in_dim %2016, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2018 = stablehlo.divide %2017, %15 : tensor<1x197x1xf64>
    %2019 = stablehlo.convert %2018 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2020 = stablehlo.reduce(%2005 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2021 = stablehlo.reshape %2020 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2023 = stablehlo.divide %2022, %31 : tensor<1x197x1xf32>
    %2024 = stablehlo.broadcast_in_dim %2019, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2025 = stablehlo.add %2024, %36 : tensor<1x197x1xf32>
    %2026 = stablehlo.rsqrt %2025 : tensor<1x197x1xf32>
    %2027 = stablehlo.broadcast_in_dim %2005, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2028 = stablehlo.broadcast_in_dim %2023, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2029 = stablehlo.subtract %2027, %2028 : tensor<1x197x768xf32>
    %2030 = stablehlo.broadcast_in_dim %2029, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2031 = stablehlo.broadcast_in_dim %2026, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2032 = stablehlo.multiply %2030, %2031 : tensor<1x197x768xf32>
    %2033 = stablehlo.convert %arg63 : (tensor<768xbf16>) -> tensor<768xf32>
    %2034 = stablehlo.broadcast_in_dim %2032, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2035 = stablehlo.broadcast_in_dim %2033, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2036 = stablehlo.multiply %2034, %2035 : tensor<1x197x768xf32>
    %2037 = stablehlo.convert %arg64 : (tensor<768xbf16>) -> tensor<768xf32>
    %2038 = stablehlo.broadcast_in_dim %2036, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2039 = stablehlo.broadcast_in_dim %2037, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2040 = stablehlo.add %2038, %2039 : tensor<1x197x768xf32>
    %2041 = stablehlo.convert %2040 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %2042 = stablehlo.reshape %2041 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %2043 = stablehlo.convert %2042 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %2044 = stablehlo.dot_general %2043, %arg198, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %2045 = stablehlo.broadcast_in_dim %2044, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2046 = stablehlo.multiply %2045, %60 : tensor<197x768xf32>
    %2047 = stablehlo.broadcast_in_dim %2046, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2048 = stablehlo.broadcast_in_dim %arg199, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2049 = stablehlo.add %2047, %2048 : tensor<197x768xf32>
    %2050 = stablehlo.convert %2049 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2051 = stablehlo.reshape %2050 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2052 = stablehlo.dot_general %2042, %arg200, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %2053 = stablehlo.reshape %2052 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2054 = stablehlo.reshape %2053 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %2055 = stablehlo.transpose %2054, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2056 = stablehlo.dot_general %2043, %arg201, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %2057 = stablehlo.broadcast_in_dim %2056, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2058 = stablehlo.multiply %2057, %60 : tensor<197x768xf32>
    %2059 = stablehlo.broadcast_in_dim %2058, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2060 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2061 = stablehlo.add %2059, %2060 : tensor<197x768xf32>
    %2062 = stablehlo.convert %2061 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2063 = stablehlo.reshape %2062 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2064 = stablehlo.reshape %2063 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %2065 = stablehlo.transpose %2064, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2066 = stablehlo.reshape %2051 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %2067 = stablehlo.transpose %2066, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2068 = stablehlo.transpose %2055, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %2069 = stablehlo.reshape %2067 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2070 = stablehlo.reshape %2068 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %2071 = stablehlo.broadcast_in_dim %2070, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %2072 = stablehlo.dot_general %2069, %2071, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %2073 = stablehlo.reshape %2072 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %2074 = stablehlo.broadcast_in_dim %2073, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %2075 = stablehlo.divide %2074, %92 : tensor<1x12x197x197xbf16>
    %2076 = stablehlo.add %2075, %arg203 : tensor<1x12x197x197xbf16>
    %2077 = stablehlo.convert %2076 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %2078 = stablehlo.reduce(%2077 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %2079 = stablehlo.reshape %2078 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %2080 = stablehlo.broadcast_in_dim %2077, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %2081 = stablehlo.broadcast_in_dim %2079, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %2082 = stablehlo.subtract %2080, %2081 : tensor<1x12x197x197xf32>
    %2083 = stablehlo.exponential %2082 : tensor<1x12x197x197xf32>
    %2084 = stablehlo.reduce(%2083 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %2085 = stablehlo.reshape %2084 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %2086 = stablehlo.broadcast_in_dim %2083, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %2087 = stablehlo.broadcast_in_dim %2085, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %2088 = stablehlo.divide %2086, %2087 : tensor<1x12x197x197xf32>
    %2089 = stablehlo.convert %2088 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %2090 = stablehlo.reshape %2089 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %2091 = stablehlo.reshape %2065 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2092 = stablehlo.broadcast_in_dim %2091, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2093 = stablehlo.dot_general %2090, %2092, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2094 = stablehlo.reshape %2093 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2095 = stablehlo.transpose %2094, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %2096 = stablehlo.reshape %2095 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %2097 = stablehlo.reshape %2096 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %2098 = stablehlo.convert %2097 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %2099 = stablehlo.dot_general %2098, %arg204, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %2100 = stablehlo.broadcast_in_dim %2099, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2101 = stablehlo.multiply %2100, %60 : tensor<197x768xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2103 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2104 = stablehlo.add %2102, %2103 : tensor<197x768xf32>
    %2105 = stablehlo.convert %2104 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2106 = stablehlo.reshape %2105 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2107 = stablehlo.broadcast_in_dim %arg65, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %2108 = stablehlo.broadcast_in_dim %2106, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %2109 = stablehlo.multiply %2107, %2108 : tensor<1x197x768xbf16>
    %2110 = stablehlo.add %2109, %2004 : tensor<1x197x768xbf16>
    %2111 = stablehlo.convert %2110 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %2112 = stablehlo.convert %2111 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %2113 = stablehlo.reduce(%2112 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2114 = stablehlo.reshape %2113 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2115 = stablehlo.broadcast_in_dim %2114, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2116 = stablehlo.divide %2115, %15 : tensor<1x197x1xf64>
    %2117 = stablehlo.broadcast_in_dim %2112, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %2118 = stablehlo.broadcast_in_dim %2116, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %2119 = stablehlo.subtract %2117, %2118 : tensor<1x197x768xf64>
    %2120 = stablehlo.multiply %2119, %2119 : tensor<1x197x768xf64>
    %2121 = stablehlo.reduce(%2120 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2122 = stablehlo.reshape %2121 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2123 = stablehlo.broadcast_in_dim %2122, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2124 = stablehlo.divide %2123, %15 : tensor<1x197x1xf64>
    %2125 = stablehlo.convert %2124 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2126 = stablehlo.reduce(%2111 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2127 = stablehlo.reshape %2126 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2128 = stablehlo.broadcast_in_dim %2127, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2129 = stablehlo.divide %2128, %31 : tensor<1x197x1xf32>
    %2130 = stablehlo.broadcast_in_dim %2125, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2131 = stablehlo.add %2130, %36 : tensor<1x197x1xf32>
    %2132 = stablehlo.rsqrt %2131 : tensor<1x197x1xf32>
    %2133 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2134 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2135 = stablehlo.subtract %2133, %2134 : tensor<1x197x768xf32>
    %2136 = stablehlo.broadcast_in_dim %2135, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2137 = stablehlo.broadcast_in_dim %2132, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2138 = stablehlo.multiply %2136, %2137 : tensor<1x197x768xf32>
    %2139 = stablehlo.convert %arg66 : (tensor<768xbf16>) -> tensor<768xf32>
    %2140 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2141 = stablehlo.broadcast_in_dim %2139, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2142 = stablehlo.multiply %2140, %2141 : tensor<1x197x768xf32>
    %2143 = stablehlo.convert %arg67 : (tensor<768xbf16>) -> tensor<768xf32>
    %2144 = stablehlo.broadcast_in_dim %2142, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2145 = stablehlo.broadcast_in_dim %2143, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2146 = stablehlo.add %2144, %2145 : tensor<1x197x768xf32>
    %2147 = stablehlo.convert %2146 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %2148 = stablehlo.reshape %2147 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %2149 = stablehlo.convert %2148 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %2150 = stablehlo.dot_general %2149, %arg206, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %2151 = stablehlo.broadcast_in_dim %2150, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %2152 = stablehlo.multiply %2151, %170 : tensor<197x3072xf32>
    %2153 = stablehlo.broadcast_in_dim %2152, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %2154 = stablehlo.broadcast_in_dim %arg207, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2155 = stablehlo.add %2153, %2154 : tensor<197x3072xf32>
    %2156 = stablehlo.convert %2155 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %2157 = stablehlo.reshape %2156 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %2158 = stablehlo.multiply %2157, %cst_4 : tensor<1x197x3072xbf16>
    %2159 = stablehlo.multiply %2157, %178 : tensor<1x197x3072xbf16>
    %2160 = stablehlo.convert %2159 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %2161 = stablehlo.clamp %cst_5, %2160, %cst_6 : tensor<1x197x3072xf32>
    %2162 = stablehlo.multiply %2161, %2161 : tensor<1x197x3072xf32>
    %2163 = stablehlo.multiply %cst_7, %2162 : tensor<1x197x3072xf32>
    %2164 = stablehlo.add %2163, %cst_8 : tensor<1x197x3072xf32>
    %2165 = stablehlo.multiply %2164, %2162 : tensor<1x197x3072xf32>
    %2166 = stablehlo.add %2165, %cst_9 : tensor<1x197x3072xf32>
    %2167 = stablehlo.multiply %2166, %2162 : tensor<1x197x3072xf32>
    %2168 = stablehlo.add %2167, %cst_10 : tensor<1x197x3072xf32>
    %2169 = stablehlo.multiply %2168, %2162 : tensor<1x197x3072xf32>
    %2170 = stablehlo.add %2169, %cst_11 : tensor<1x197x3072xf32>
    %2171 = stablehlo.multiply %2170, %2162 : tensor<1x197x3072xf32>
    %2172 = stablehlo.add %2171, %cst_12 : tensor<1x197x3072xf32>
    %2173 = stablehlo.multiply %2172, %2162 : tensor<1x197x3072xf32>
    %2174 = stablehlo.add %2173, %cst_13 : tensor<1x197x3072xf32>
    %2175 = stablehlo.multiply %cst_14, %2162 : tensor<1x197x3072xf32>
    %2176 = stablehlo.add %2175, %cst_15 : tensor<1x197x3072xf32>
    %2177 = stablehlo.multiply %2176, %2162 : tensor<1x197x3072xf32>
    %2178 = stablehlo.add %2177, %cst_16 : tensor<1x197x3072xf32>
    %2179 = stablehlo.multiply %2178, %2162 : tensor<1x197x3072xf32>
    %2180 = stablehlo.add %2179, %cst_17 : tensor<1x197x3072xf32>
    %2181 = stablehlo.multiply %2180, %2162 : tensor<1x197x3072xf32>
    %2182 = stablehlo.add %2181, %cst_18 : tensor<1x197x3072xf32>
    %2183 = stablehlo.multiply %2161, %2174 : tensor<1x197x3072xf32>
    %2184 = stablehlo.divide %2183, %2182 : tensor<1x197x3072xf32>
    %2185 = stablehlo.clamp %cst_19, %2184, %cst_20 : tensor<1x197x3072xf32>
    %2186 = stablehlo.convert %2185 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %2187 = stablehlo.add %2186, %cst_2 : tensor<1x197x3072xbf16>
    %2188 = stablehlo.multiply %2187, %2158 : tensor<1x197x3072xbf16>
    %2189 = stablehlo.reshape %2188 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %2190 = stablehlo.convert %2189 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %2191 = stablehlo.dot_general %2190, %arg208, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2193 = stablehlo.multiply %2192, %60 : tensor<197x768xf32>
    %2194 = stablehlo.broadcast_in_dim %2193, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2195 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2196 = stablehlo.add %2194, %2195 : tensor<197x768xf32>
    %2197 = stablehlo.convert %2196 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2198 = stablehlo.reshape %2197 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2199 = stablehlo.broadcast_in_dim %arg68, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %2200 = stablehlo.broadcast_in_dim %2198, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %2201 = stablehlo.multiply %2199, %2200 : tensor<1x197x768xbf16>
    %2202 = stablehlo.add %2201, %2110 : tensor<1x197x768xbf16>
    %2203 = stablehlo.convert %2202 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %2204 = stablehlo.convert %2203 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %2205 = stablehlo.reduce(%2204 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2206 = stablehlo.reshape %2205 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2207 = stablehlo.broadcast_in_dim %2206, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2208 = stablehlo.divide %2207, %15 : tensor<1x197x1xf64>
    %2209 = stablehlo.broadcast_in_dim %2204, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %2210 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %2211 = stablehlo.subtract %2209, %2210 : tensor<1x197x768xf64>
    %2212 = stablehlo.multiply %2211, %2211 : tensor<1x197x768xf64>
    %2213 = stablehlo.reduce(%2212 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2214 = stablehlo.reshape %2213 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2215 = stablehlo.broadcast_in_dim %2214, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2216 = stablehlo.divide %2215, %15 : tensor<1x197x1xf64>
    %2217 = stablehlo.convert %2216 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2218 = stablehlo.reduce(%2203 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2219 = stablehlo.reshape %2218 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2220 = stablehlo.broadcast_in_dim %2219, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2221 = stablehlo.divide %2220, %31 : tensor<1x197x1xf32>
    %2222 = stablehlo.broadcast_in_dim %2217, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2223 = stablehlo.add %2222, %36 : tensor<1x197x1xf32>
    %2224 = stablehlo.rsqrt %2223 : tensor<1x197x1xf32>
    %2225 = stablehlo.broadcast_in_dim %2203, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2226 = stablehlo.broadcast_in_dim %2221, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2227 = stablehlo.subtract %2225, %2226 : tensor<1x197x768xf32>
    %2228 = stablehlo.broadcast_in_dim %2227, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2229 = stablehlo.broadcast_in_dim %2224, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2230 = stablehlo.multiply %2228, %2229 : tensor<1x197x768xf32>
    %2231 = stablehlo.convert %arg69 : (tensor<768xbf16>) -> tensor<768xf32>
    %2232 = stablehlo.broadcast_in_dim %2230, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2233 = stablehlo.broadcast_in_dim %2231, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2234 = stablehlo.multiply %2232, %2233 : tensor<1x197x768xf32>
    %2235 = stablehlo.convert %arg70 : (tensor<768xbf16>) -> tensor<768xf32>
    %2236 = stablehlo.broadcast_in_dim %2234, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2237 = stablehlo.broadcast_in_dim %2235, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2238 = stablehlo.add %2236, %2237 : tensor<1x197x768xf32>
    %2239 = stablehlo.convert %2238 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %2240 = stablehlo.reshape %2239 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %2241 = stablehlo.convert %2240 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %2242 = stablehlo.dot_general %2241, %arg210, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %2243 = stablehlo.broadcast_in_dim %2242, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2244 = stablehlo.multiply %2243, %60 : tensor<197x768xf32>
    %2245 = stablehlo.broadcast_in_dim %2244, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2246 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2247 = stablehlo.add %2245, %2246 : tensor<197x768xf32>
    %2248 = stablehlo.convert %2247 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2249 = stablehlo.reshape %2248 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2250 = stablehlo.dot_general %2240, %arg212, contracting_dims = [1] x [0] : (tensor<197x768xbf16>, tensor<768x768xbf16>) -> tensor<197x768xbf16>
    %2251 = stablehlo.reshape %2250 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2252 = stablehlo.reshape %2251 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %2253 = stablehlo.transpose %2252, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2254 = stablehlo.dot_general %2241, %arg213, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %2255 = stablehlo.broadcast_in_dim %2254, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2256 = stablehlo.multiply %2255, %60 : tensor<197x768xf32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2258 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2259 = stablehlo.add %2257, %2258 : tensor<197x768xf32>
    %2260 = stablehlo.convert %2259 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2261 = stablehlo.reshape %2260 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2262 = stablehlo.reshape %2261 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %2263 = stablehlo.transpose %2262, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2264 = stablehlo.reshape %2249 : (tensor<1x197x768xbf16>) -> tensor<1x197x12x64xbf16>
    %2265 = stablehlo.transpose %2264, dims = [0, 2, 1, 3] : (tensor<1x197x12x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2266 = stablehlo.transpose %2253, dims = [0, 1, 3, 2] : (tensor<1x12x197x64xbf16>) -> tensor<1x12x64x197xbf16>
    %2267 = stablehlo.reshape %2265 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2268 = stablehlo.reshape %2266 : (tensor<1x12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %2269 = stablehlo.broadcast_in_dim %2268, dims = [0, 1, 2] : (tensor<12x64x197xbf16>) -> tensor<12x64x197xbf16>
    %2270 = stablehlo.dot_general %2267, %2269, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %2271 = stablehlo.reshape %2270 : (tensor<12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %2272 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xbf16>
    %2273 = stablehlo.divide %2272, %92 : tensor<1x12x197x197xbf16>
    %2274 = stablehlo.add %2273, %arg215 : tensor<1x12x197x197xbf16>
    %2275 = stablehlo.convert %2274 : (tensor<1x12x197x197xbf16>) -> tensor<1x12x197x197xf32>
    %2276 = stablehlo.reduce(%2275 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %2277 = stablehlo.reshape %2276 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %2278 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %2279 = stablehlo.broadcast_in_dim %2277, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %2280 = stablehlo.subtract %2278, %2279 : tensor<1x12x197x197xf32>
    %2281 = stablehlo.exponential %2280 : tensor<1x12x197x197xf32>
    %2282 = stablehlo.reduce(%2281 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %2283 = stablehlo.reshape %2282 : (tensor<1x12x197xf32>) -> tensor<1x12x197x1xf32>
    %2284 = stablehlo.broadcast_in_dim %2281, dims = [0, 1, 2, 3] : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %2285 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2, 3] : (tensor<1x12x197x1xf32>) -> tensor<1x12x197x197xf32>
    %2286 = stablehlo.divide %2284, %2285 : tensor<1x12x197x197xf32>
    %2287 = stablehlo.convert %2286 : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xbf16>
    %2288 = stablehlo.reshape %2287 : (tensor<1x12x197x197xbf16>) -> tensor<12x197x197xbf16>
    %2289 = stablehlo.reshape %2263 : (tensor<1x12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2290 = stablehlo.broadcast_in_dim %2289, dims = [0, 1, 2] : (tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2291 = stablehlo.dot_general %2288, %2290, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    %2292 = stablehlo.reshape %2291 : (tensor<12x197x64xbf16>) -> tensor<1x12x197x64xbf16>
    %2293 = stablehlo.transpose %2292, dims = [0, 2, 1, 3] : (tensor<1x12x197x64xbf16>) -> tensor<1x197x12x64xbf16>
    %2294 = stablehlo.reshape %2293 : (tensor<1x197x12x64xbf16>) -> tensor<1x197x768xbf16>
    %2295 = stablehlo.reshape %2294 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %2296 = stablehlo.convert %2295 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %2297 = stablehlo.dot_general %2296, %arg216, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %2298 = stablehlo.broadcast_in_dim %2297, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2299 = stablehlo.multiply %2298, %60 : tensor<197x768xf32>
    %2300 = stablehlo.broadcast_in_dim %2299, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2301 = stablehlo.broadcast_in_dim %arg217, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2302 = stablehlo.add %2300, %2301 : tensor<197x768xf32>
    %2303 = stablehlo.convert %2302 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2304 = stablehlo.reshape %2303 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2305 = stablehlo.broadcast_in_dim %arg71, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %2306 = stablehlo.broadcast_in_dim %2304, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %2307 = stablehlo.multiply %2305, %2306 : tensor<1x197x768xbf16>
    %2308 = stablehlo.add %2307, %2202 : tensor<1x197x768xbf16>
    %2309 = stablehlo.convert %2308 : (tensor<1x197x768xbf16>) -> tensor<1x197x768xf32>
    %2310 = stablehlo.convert %2309 : (tensor<1x197x768xf32>) -> tensor<1x197x768xf64>
    %2311 = stablehlo.reduce(%2310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2312 = stablehlo.reshape %2311 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2313 = stablehlo.broadcast_in_dim %2312, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2314 = stablehlo.divide %2313, %15 : tensor<1x197x1xf64>
    %2315 = stablehlo.broadcast_in_dim %2310, dims = [0, 1, 2] : (tensor<1x197x768xf64>) -> tensor<1x197x768xf64>
    %2316 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x768xf64>
    %2317 = stablehlo.subtract %2315, %2316 : tensor<1x197x768xf64>
    %2318 = stablehlo.multiply %2317, %2317 : tensor<1x197x768xf64>
    %2319 = stablehlo.reduce(%2318 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2320 = stablehlo.reshape %2319 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2321 = stablehlo.broadcast_in_dim %2320, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2322 = stablehlo.divide %2321, %15 : tensor<1x197x1xf64>
    %2323 = stablehlo.convert %2322 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2324 = stablehlo.reduce(%2309 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2325 = stablehlo.reshape %2324 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2326 = stablehlo.broadcast_in_dim %2325, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2327 = stablehlo.divide %2326, %31 : tensor<1x197x1xf32>
    %2328 = stablehlo.broadcast_in_dim %2323, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2329 = stablehlo.add %2328, %36 : tensor<1x197x1xf32>
    %2330 = stablehlo.rsqrt %2329 : tensor<1x197x1xf32>
    %2331 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2332 = stablehlo.broadcast_in_dim %2327, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2333 = stablehlo.subtract %2331, %2332 : tensor<1x197x768xf32>
    %2334 = stablehlo.broadcast_in_dim %2333, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2335 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %2336 = stablehlo.multiply %2334, %2335 : tensor<1x197x768xf32>
    %2337 = stablehlo.convert %arg72 : (tensor<768xbf16>) -> tensor<768xf32>
    %2338 = stablehlo.broadcast_in_dim %2336, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2339 = stablehlo.broadcast_in_dim %2337, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2340 = stablehlo.multiply %2338, %2339 : tensor<1x197x768xf32>
    %2341 = stablehlo.convert %arg73 : (tensor<768xbf16>) -> tensor<768xf32>
    %2342 = stablehlo.broadcast_in_dim %2340, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %2343 = stablehlo.broadcast_in_dim %2341, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %2344 = stablehlo.add %2342, %2343 : tensor<1x197x768xf32>
    %2345 = stablehlo.convert %2344 : (tensor<1x197x768xf32>) -> tensor<1x197x768xbf16>
    %2346 = stablehlo.reshape %2345 : (tensor<1x197x768xbf16>) -> tensor<197x768xbf16>
    %2347 = stablehlo.convert %2346 : (tensor<197x768xbf16>) -> tensor<197x768xf32>
    %2348 = stablehlo.dot_general %2347, %arg218, contracting_dims = [1] x [0] : (tensor<197x768xf32>, tensor<768x3072xf32>) -> tensor<197x3072xf32>
    %2349 = stablehlo.broadcast_in_dim %2348, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %2350 = stablehlo.multiply %2349, %170 : tensor<197x3072xf32>
    %2351 = stablehlo.broadcast_in_dim %2350, dims = [0, 1] : (tensor<197x3072xf32>) -> tensor<197x3072xf32>
    %2352 = stablehlo.broadcast_in_dim %arg219, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2353 = stablehlo.add %2351, %2352 : tensor<197x3072xf32>
    %2354 = stablehlo.convert %2353 : (tensor<197x3072xf32>) -> tensor<197x3072xbf16>
    %2355 = stablehlo.reshape %2354 : (tensor<197x3072xbf16>) -> tensor<1x197x3072xbf16>
    %2356 = stablehlo.multiply %2355, %cst_4 : tensor<1x197x3072xbf16>
    %2357 = stablehlo.multiply %2355, %178 : tensor<1x197x3072xbf16>
    %2358 = stablehlo.convert %2357 : (tensor<1x197x3072xbf16>) -> tensor<1x197x3072xf32>
    %2359 = stablehlo.clamp %cst_5, %2358, %cst_6 : tensor<1x197x3072xf32>
    %2360 = stablehlo.multiply %2359, %2359 : tensor<1x197x3072xf32>
    %2361 = stablehlo.multiply %cst_7, %2360 : tensor<1x197x3072xf32>
    %2362 = stablehlo.add %2361, %cst_8 : tensor<1x197x3072xf32>
    %2363 = stablehlo.multiply %2362, %2360 : tensor<1x197x3072xf32>
    %2364 = stablehlo.add %2363, %cst_9 : tensor<1x197x3072xf32>
    %2365 = stablehlo.multiply %2364, %2360 : tensor<1x197x3072xf32>
    %2366 = stablehlo.add %2365, %cst_10 : tensor<1x197x3072xf32>
    %2367 = stablehlo.multiply %2366, %2360 : tensor<1x197x3072xf32>
    %2368 = stablehlo.add %2367, %cst_11 : tensor<1x197x3072xf32>
    %2369 = stablehlo.multiply %2368, %2360 : tensor<1x197x3072xf32>
    %2370 = stablehlo.add %2369, %cst_12 : tensor<1x197x3072xf32>
    %2371 = stablehlo.multiply %2370, %2360 : tensor<1x197x3072xf32>
    %2372 = stablehlo.add %2371, %cst_13 : tensor<1x197x3072xf32>
    %2373 = stablehlo.multiply %cst_14, %2360 : tensor<1x197x3072xf32>
    %2374 = stablehlo.add %2373, %cst_15 : tensor<1x197x3072xf32>
    %2375 = stablehlo.multiply %2374, %2360 : tensor<1x197x3072xf32>
    %2376 = stablehlo.add %2375, %cst_16 : tensor<1x197x3072xf32>
    %2377 = stablehlo.multiply %2376, %2360 : tensor<1x197x3072xf32>
    %2378 = stablehlo.add %2377, %cst_17 : tensor<1x197x3072xf32>
    %2379 = stablehlo.multiply %2378, %2360 : tensor<1x197x3072xf32>
    %2380 = stablehlo.add %2379, %cst_18 : tensor<1x197x3072xf32>
    %2381 = stablehlo.multiply %2359, %2372 : tensor<1x197x3072xf32>
    %2382 = stablehlo.divide %2381, %2380 : tensor<1x197x3072xf32>
    %2383 = stablehlo.clamp %cst_19, %2382, %cst_20 : tensor<1x197x3072xf32>
    %2384 = stablehlo.convert %2383 : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xbf16>
    %2385 = stablehlo.add %2384, %cst_2 : tensor<1x197x3072xbf16>
    %2386 = stablehlo.multiply %2385, %2356 : tensor<1x197x3072xbf16>
    %2387 = stablehlo.reshape %2386 : (tensor<1x197x3072xbf16>) -> tensor<197x3072xbf16>
    %2388 = stablehlo.convert %2387 : (tensor<197x3072xbf16>) -> tensor<197x3072xf32>
    %2389 = stablehlo.dot_general %2388, %arg220, contracting_dims = [1] x [0] : (tensor<197x3072xf32>, tensor<3072x768xf32>) -> tensor<197x768xf32>
    %2390 = stablehlo.broadcast_in_dim %2389, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2391 = stablehlo.multiply %2390, %60 : tensor<197x768xf32>
    %2392 = stablehlo.broadcast_in_dim %2391, dims = [0, 1] : (tensor<197x768xf32>) -> tensor<197x768xf32>
    %2393 = stablehlo.broadcast_in_dim %arg221, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %2394 = stablehlo.add %2392, %2393 : tensor<197x768xf32>
    %2395 = stablehlo.convert %2394 : (tensor<197x768xf32>) -> tensor<197x768xbf16>
    %2396 = stablehlo.reshape %2395 : (tensor<197x768xbf16>) -> tensor<1x197x768xbf16>
    %2397 = stablehlo.broadcast_in_dim %arg74, dims = [2] : (tensor<768xbf16>) -> tensor<1x197x768xbf16>
    %2398 = stablehlo.broadcast_in_dim %2396, dims = [0, 1, 2] : (tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    %2399 = stablehlo.multiply %2397, %2398 : tensor<1x197x768xbf16>
    %2400 = stablehlo.add %2399, %2308 : tensor<1x197x768xbf16>
    %2401 = stablehlo.slice %2400 [0:1, 1:197, 0:768] : (tensor<1x197x768xbf16>) -> tensor<1x196x768xbf16>
    %2402 = stablehlo.reduce(%2401 init: %cst_21) applies stablehlo.add across dimensions = [1] : (tensor<1x196x768xbf16>, tensor<bf16>) -> tensor<1x768xbf16>
    %2403 = stablehlo.convert %cst_26 : (tensor<1xi64>) -> tensor<1xbf16>
    %2404 = stablehlo.reshape %2403 : (tensor<1xbf16>) -> tensor<bf16>
    %2405 = stablehlo.broadcast_in_dim %2402, dims = [0, 1] : (tensor<1x768xbf16>) -> tensor<1x768xbf16>
    %2406 = stablehlo.broadcast_in_dim %2404, dims = [] : (tensor<bf16>) -> tensor<1x768xbf16>
    %2407 = stablehlo.divide %2405, %2406 : tensor<1x768xbf16>
    %2408 = stablehlo.convert %2407 : (tensor<1x768xbf16>) -> tensor<1x768xf32>
    %2409 = stablehlo.convert %2408 : (tensor<1x768xf32>) -> tensor<1x768xf64>
    %2410 = stablehlo.reduce(%2409 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x768xf64>, tensor<f64>) -> tensor<1xf64>
    %2411 = stablehlo.reshape %2410 : (tensor<1xf64>) -> tensor<1x1xf64>
    %2412 = stablehlo.broadcast_in_dim %2411, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %2413 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %2414 = stablehlo.divide %2412, %2413 : tensor<1x1xf64>
    %2415 = stablehlo.broadcast_in_dim %2409, dims = [0, 1] : (tensor<1x768xf64>) -> tensor<1x768xf64>
    %2416 = stablehlo.broadcast_in_dim %2414, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x768xf64>
    %2417 = stablehlo.subtract %2415, %2416 : tensor<1x768xf64>
    %2418 = stablehlo.multiply %2417, %2417 : tensor<1x768xf64>
    %2419 = stablehlo.reduce(%2418 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x768xf64>, tensor<f64>) -> tensor<1xf64>
    %2420 = stablehlo.reshape %2419 : (tensor<1xf64>) -> tensor<1x1xf64>
    %2421 = stablehlo.broadcast_in_dim %2420, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %2422 = stablehlo.divide %2421, %2413 : tensor<1x1xf64>
    %2423 = stablehlo.convert %2422 : (tensor<1x1xf64>) -> tensor<1x1xf32>
    %2424 = stablehlo.reduce(%2408 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<1x768xf32>, tensor<f32>) -> tensor<1xf32>
    %2425 = stablehlo.reshape %2424 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2426 = stablehlo.broadcast_in_dim %2425, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2427 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2428 = stablehlo.divide %2426, %2427 : tensor<1x1xf32>
    %2429 = stablehlo.broadcast_in_dim %2423, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2430 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2431 = stablehlo.add %2429, %2430 : tensor<1x1xf32>
    %2432 = stablehlo.rsqrt %2431 : tensor<1x1xf32>
    %2433 = stablehlo.broadcast_in_dim %2408, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %2434 = stablehlo.broadcast_in_dim %2428, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x768xf32>
    %2435 = stablehlo.subtract %2433, %2434 : tensor<1x768xf32>
    %2436 = stablehlo.broadcast_in_dim %2435, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %2437 = stablehlo.broadcast_in_dim %2432, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x768xf32>
    %2438 = stablehlo.multiply %2436, %2437 : tensor<1x768xf32>
    %2439 = stablehlo.convert %arg75 : (tensor<768xbf16>) -> tensor<768xf32>
    %2440 = stablehlo.broadcast_in_dim %2438, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %2441 = stablehlo.broadcast_in_dim %2439, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2442 = stablehlo.multiply %2440, %2441 : tensor<1x768xf32>
    %2443 = stablehlo.convert %arg76 : (tensor<768xbf16>) -> tensor<768xf32>
    %2444 = stablehlo.broadcast_in_dim %2442, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %2445 = stablehlo.broadcast_in_dim %2443, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2446 = stablehlo.add %2444, %2445 : tensor<1x768xf32>
    %2447 = stablehlo.convert %2446 : (tensor<1x768xf32>) -> tensor<1x768xbf16>
    %2448 = stablehlo.convert %2447 : (tensor<1x768xbf16>) -> tensor<1x768xf32>
    %2449 = stablehlo.dot_general %2448, %arg222, contracting_dims = [1] x [0] : (tensor<1x768xf32>, tensor<768x1000xf32>) -> tensor<1x1000xf32>
    %2450 = stablehlo.broadcast_in_dim %2449, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %2451 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<1x1000xf32>
    %2452 = stablehlo.multiply %2450, %2451 : tensor<1x1000xf32>
    %2453 = stablehlo.broadcast_in_dim %2452, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %2454 = stablehlo.broadcast_in_dim %arg223, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %2455 = stablehlo.add %2453, %2454 : tensor<1x1000xf32>
    %2456 = stablehlo.convert %2455 : (tensor<1x1000xf32>) -> tensor<1x1000xbf16>
    return %2456 : tensor<1x1000xbf16>
  }
}
