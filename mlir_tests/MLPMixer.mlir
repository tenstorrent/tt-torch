module {
  func.func @main(%arg0: tensor<1x3x256x256xbf16>, %arg1: tensor<512xbf16>, %arg2: tensor<512xbf16>, %arg3: tensor<1024x256x1xbf16>, %arg4: tensor<1024xbf16>, %arg5: tensor<256x1024x1xbf16>, %arg6: tensor<256xbf16>, %arg7: tensor<512xbf16>, %arg8: tensor<512xbf16>, %arg9: tensor<512xbf16>, %arg10: tensor<512xbf16>, %arg11: tensor<1024x256x1xbf16>, %arg12: tensor<1024xbf16>, %arg13: tensor<256x1024x1xbf16>, %arg14: tensor<256xbf16>, %arg15: tensor<512xbf16>, %arg16: tensor<512xbf16>, %arg17: tensor<512xbf16>, %arg18: tensor<512xbf16>, %arg19: tensor<1024x256x1xbf16>, %arg20: tensor<1024xbf16>, %arg21: tensor<256x1024x1xbf16>, %arg22: tensor<256xbf16>, %arg23: tensor<512xbf16>, %arg24: tensor<512xbf16>, %arg25: tensor<512xbf16>, %arg26: tensor<512xbf16>, %arg27: tensor<1024x256x1xbf16>, %arg28: tensor<1024xbf16>, %arg29: tensor<256x1024x1xbf16>, %arg30: tensor<256xbf16>, %arg31: tensor<512xbf16>, %arg32: tensor<512xbf16>, %arg33: tensor<512xbf16>, %arg34: tensor<512xbf16>, %arg35: tensor<1024x256x1xbf16>, %arg36: tensor<1024xbf16>, %arg37: tensor<256x1024x1xbf16>, %arg38: tensor<256xbf16>, %arg39: tensor<512xbf16>, %arg40: tensor<512xbf16>, %arg41: tensor<512xbf16>, %arg42: tensor<512xbf16>, %arg43: tensor<1024x256x1xbf16>, %arg44: tensor<1024xbf16>, %arg45: tensor<256x1024x1xbf16>, %arg46: tensor<256xbf16>, %arg47: tensor<512xbf16>, %arg48: tensor<512xbf16>, %arg49: tensor<512xbf16>, %arg50: tensor<512xbf16>, %arg51: tensor<1024x256x1xbf16>, %arg52: tensor<1024xbf16>, %arg53: tensor<256x1024x1xbf16>, %arg54: tensor<256xbf16>, %arg55: tensor<512xbf16>, %arg56: tensor<512xbf16>, %arg57: tensor<512xbf16>, %arg58: tensor<512xbf16>, %arg59: tensor<1024x256x1xbf16>, %arg60: tensor<1024xbf16>, %arg61: tensor<256x1024x1xbf16>, %arg62: tensor<256xbf16>, %arg63: tensor<512xbf16>, %arg64: tensor<512xbf16>, %arg65: tensor<512xbf16>, %arg66: tensor<512xbf16>, %arg67: tensor<1024x256x1xbf16>, %arg68: tensor<1024xbf16>, %arg69: tensor<256x1024x1xbf16>, %arg70: tensor<256xbf16>, %arg71: tensor<512xbf16>, %arg72: tensor<512xbf16>, %arg73: tensor<512xbf16>, %arg74: tensor<512xbf16>, %arg75: tensor<1024x256x1xbf16>, %arg76: tensor<1024xbf16>, %arg77: tensor<256x1024x1xbf16>, %arg78: tensor<256xbf16>, %arg79: tensor<512xbf16>, %arg80: tensor<512xbf16>, %arg81: tensor<512xbf16>, %arg82: tensor<512xbf16>, %arg83: tensor<1024x256x1xbf16>, %arg84: tensor<1024xbf16>, %arg85: tensor<256x1024x1xbf16>, %arg86: tensor<256xbf16>, %arg87: tensor<512xbf16>, %arg88: tensor<512xbf16>, %arg89: tensor<512xbf16>, %arg90: tensor<512xbf16>, %arg91: tensor<1024x256x1xbf16>, %arg92: tensor<1024xbf16>, %arg93: tensor<256x1024x1xbf16>, %arg94: tensor<256xbf16>, %arg95: tensor<512xbf16>, %arg96: tensor<512xbf16>, %arg97: tensor<512xbf16>, %arg98: tensor<512xbf16>, %arg99: tensor<768x512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<512x256xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256x512xf32>, %arg104: tensor<512xf32>, %arg105: tensor<512x256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256x512xf32>, %arg108: tensor<512xf32>, %arg109: tensor<512x256xf32>, %arg110: tensor<256xf32>, %arg111: tensor<256x512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<512x256xf32>, %arg114: tensor<256xf32>, %arg115: tensor<256x512xf32>, %arg116: tensor<512xf32>, %arg117: tensor<512x256xf32>, %arg118: tensor<256xf32>, %arg119: tensor<256x512xf32>, %arg120: tensor<512xf32>, %arg121: tensor<512x256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256x512xf32>, %arg124: tensor<512xf32>, %arg125: tensor<512x256xf32>, %arg126: tensor<256xf32>, %arg127: tensor<256x512xf32>, %arg128: tensor<512xf32>, %arg129: tensor<512x256xf32>, %arg130: tensor<256xf32>, %arg131: tensor<256x512xf32>, %arg132: tensor<512xf32>, %arg133: tensor<512x256xf32>, %arg134: tensor<256xf32>, %arg135: tensor<256x512xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512x256xf32>, %arg138: tensor<256xf32>, %arg139: tensor<256x512xf32>, %arg140: tensor<512xf32>, %arg141: tensor<512x256xf32>, %arg142: tensor<256xf32>, %arg143: tensor<256x512xf32>, %arg144: tensor<512xf32>, %arg145: tensor<512x256xf32>, %arg146: tensor<256xf32>, %arg147: tensor<256x512xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512x1000xf32>, %arg150: tensor<1000xf32>) -> tensor<1x1000xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<1x1024x512xbf16>
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<1x1024x512xbf16>
    %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<1x1024x512xbf16>
    %cst_4 = stablehlo.constant dense<-4.000000e+00> : tensor<1x1024x512xf32>
    %cst_5 = stablehlo.constant dense<4.000000e+00> : tensor<1x1024x512xf32>
    %cst_6 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x1024x512xf32>
    %cst_7 = stablehlo.constant dense<2.77068146E-8> : tensor<1x1024x512xf32>
    %cst_8 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x1024x512xf32>
    %cst_9 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x1024x512xf32>
    %cst_10 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x1024x512xf32>
    %cst_11 = stablehlo.constant dense<-2.954600e-03> : tensor<1x1024x512xf32>
    %cst_12 = stablehlo.constant dense<-0.0160960332> : tensor<1x1024x512xf32>
    %cst_13 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x1024x512xf32>
    %cst_14 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x1024x512xf32>
    %cst_15 = stablehlo.constant dense<-0.00168282702> : tensor<1x1024x512xf32>
    %cst_16 = stablehlo.constant dense<-0.00737332925> : tensor<1x1024x512xf32>
    %cst_17 = stablehlo.constant dense<-0.0142647391> : tensor<1x1024x512xf32>
    %cst_18 = stablehlo.constant dense<-1.000000e+00> : tensor<1x1024x512xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<1x1024x512xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x256x256xbf16>
    %cst_21 = stablehlo.constant dense<2.000000e+00> : tensor<1x256x256xbf16>
    %cst_22 = stablehlo.constant dense<5.000000e-01> : tensor<1x256x256xbf16>
    %cst_23 = stablehlo.constant dense<-4.000000e+00> : tensor<1x256x256xf32>
    %cst_24 = stablehlo.constant dense<4.000000e+00> : tensor<1x256x256xf32>
    %cst_25 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x256x256xf32>
    %cst_26 = stablehlo.constant dense<2.77068146E-8> : tensor<1x256x256xf32>
    %cst_27 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x256x256xf32>
    %cst_28 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x256x256xf32>
    %cst_29 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x256x256xf32>
    %cst_30 = stablehlo.constant dense<-2.954600e-03> : tensor<1x256x256xf32>
    %cst_31 = stablehlo.constant dense<-0.0160960332> : tensor<1x256x256xf32>
    %cst_32 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x256x256xf32>
    %cst_33 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x256x256xf32>
    %cst_34 = stablehlo.constant dense<-0.00168282702> : tensor<1x256x256xf32>
    %cst_35 = stablehlo.constant dense<-0.00737332925> : tensor<1x256x256xf32>
    %cst_36 = stablehlo.constant dense<-0.0142647391> : tensor<1x256x256xf32>
    %cst_37 = stablehlo.constant dense<-1.000000e+00> : tensor<1x256x256xf32>
    %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<1x256x256xf32>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_40 = arith.constant dense<1> : tensor<1xi64>
    %cst_41 = arith.constant dense<512> : tensor<1xi64>
    %cst_42 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_43 = arith.constant dense<256> : tensor<1xi64>
    %0 = stablehlo.reshape %arg0 : (tensor<1x3x256x256xbf16>) -> tensor<1x3x16x16x16x16xbf16>
    %1 = stablehlo.transpose %0, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x3x16x16x16x16xbf16>) -> tensor<1x16x16x16x16x3xbf16>
    %2 = stablehlo.reshape %1 : (tensor<1x16x16x16x16x3xbf16>) -> tensor<1x256x768xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x256x768xbf16>) -> tensor<256x768xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x768xbf16>) -> tensor<256x768xf32>
    %5 = stablehlo.dot_general %4, %arg99, contracting_dims = [1] x [0] : (tensor<256x768xf32>, tensor<768x512xf32>) -> tensor<256x512xf32>
    %6 = stablehlo.convert %cst_40 : (tensor<1xi64>) -> tensor<1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
    %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x512xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg100, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x512xf32>
    %14 = stablehlo.convert %13 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %15 = stablehlo.reshape %14 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %16 = stablehlo.convert %15 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %17 = stablehlo.convert %16 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %19 = stablehlo.reshape %18 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %20 = stablehlo.convert %cst_41 : (tensor<1xi64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %23 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f64>) -> tensor<1x256x1xf64>
    %24 = stablehlo.divide %22, %23 : tensor<1x256x1xf64>
    %25 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %26 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %27 = stablehlo.subtract %25, %26 : tensor<1x256x512xf64>
    %28 = stablehlo.multiply %27, %27 : tensor<1x256x512xf64>
    %29 = stablehlo.reduce(%28 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %30 = stablehlo.reshape %29 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %32 = stablehlo.divide %31, %23 : tensor<1x256x1xf64>
    %33 = stablehlo.convert %32 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %34 = stablehlo.reduce(%16 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %36 = stablehlo.convert %cst_41 : (tensor<1xi64>) -> tensor<1xf32>
    %37 = stablehlo.reshape %36 : (tensor<1xf32>) -> tensor<f32>
    %38 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %40 = stablehlo.divide %38, %39 : tensor<1x256x1xf32>
    %41 = stablehlo.convert %cst_42 : (tensor<1xf64>) -> tensor<1xf32>
    %42 = stablehlo.reshape %41 : (tensor<1xf32>) -> tensor<f32>
    %43 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %45 = stablehlo.add %43, %44 : tensor<1x256x1xf32>
    %46 = stablehlo.rsqrt %45 : tensor<1x256x1xf32>
    %47 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %48 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %49 = stablehlo.subtract %47, %48 : tensor<1x256x512xf32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %51 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %52 = stablehlo.multiply %50, %51 : tensor<1x256x512xf32>
    %53 = stablehlo.convert %arg1 : (tensor<512xbf16>) -> tensor<512xf32>
    %54 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %55 = stablehlo.broadcast_in_dim %53, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %56 = stablehlo.multiply %54, %55 : tensor<1x256x512xf32>
    %57 = stablehlo.convert %arg2 : (tensor<512xbf16>) -> tensor<512xf32>
    %58 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %59 = stablehlo.broadcast_in_dim %57, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x256x512xf32>
    %61 = stablehlo.convert %60 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %62 = stablehlo.convolution(%61, %arg3) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %63 = stablehlo.reshape %arg4 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %64 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %65 = stablehlo.broadcast_in_dim %63, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %66 = stablehlo.add %64, %65 : tensor<1x1024x512xbf16>
    %67 = stablehlo.multiply %66, %cst_3 : tensor<1x1024x512xbf16>
    %68 = stablehlo.rsqrt %cst_2 : tensor<1x1024x512xbf16>
    %69 = stablehlo.multiply %66, %68 : tensor<1x1024x512xbf16>
    %70 = stablehlo.convert %69 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %71 = stablehlo.clamp %cst_4, %70, %cst_5 : tensor<1x1024x512xf32>
    %72 = stablehlo.multiply %71, %71 : tensor<1x1024x512xf32>
    %73 = stablehlo.multiply %cst_6, %72 : tensor<1x1024x512xf32>
    %74 = stablehlo.add %73, %cst_7 : tensor<1x1024x512xf32>
    %75 = stablehlo.multiply %74, %72 : tensor<1x1024x512xf32>
    %76 = stablehlo.add %75, %cst_8 : tensor<1x1024x512xf32>
    %77 = stablehlo.multiply %76, %72 : tensor<1x1024x512xf32>
    %78 = stablehlo.add %77, %cst_9 : tensor<1x1024x512xf32>
    %79 = stablehlo.multiply %78, %72 : tensor<1x1024x512xf32>
    %80 = stablehlo.add %79, %cst_10 : tensor<1x1024x512xf32>
    %81 = stablehlo.multiply %80, %72 : tensor<1x1024x512xf32>
    %82 = stablehlo.add %81, %cst_11 : tensor<1x1024x512xf32>
    %83 = stablehlo.multiply %82, %72 : tensor<1x1024x512xf32>
    %84 = stablehlo.add %83, %cst_12 : tensor<1x1024x512xf32>
    %85 = stablehlo.multiply %cst_13, %72 : tensor<1x1024x512xf32>
    %86 = stablehlo.add %85, %cst_14 : tensor<1x1024x512xf32>
    %87 = stablehlo.multiply %86, %72 : tensor<1x1024x512xf32>
    %88 = stablehlo.add %87, %cst_15 : tensor<1x1024x512xf32>
    %89 = stablehlo.multiply %88, %72 : tensor<1x1024x512xf32>
    %90 = stablehlo.add %89, %cst_16 : tensor<1x1024x512xf32>
    %91 = stablehlo.multiply %90, %72 : tensor<1x1024x512xf32>
    %92 = stablehlo.add %91, %cst_17 : tensor<1x1024x512xf32>
    %93 = stablehlo.multiply %71, %84 : tensor<1x1024x512xf32>
    %94 = stablehlo.divide %93, %92 : tensor<1x1024x512xf32>
    %95 = stablehlo.clamp %cst_18, %94, %cst_19 : tensor<1x1024x512xf32>
    %96 = stablehlo.convert %95 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %97 = stablehlo.add %96, %cst_1 : tensor<1x1024x512xbf16>
    %98 = stablehlo.multiply %97, %67 : tensor<1x1024x512xbf16>
    %99 = stablehlo.convolution(%98, %arg5) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %100 = stablehlo.reshape %arg6 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %101 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %102 = stablehlo.broadcast_in_dim %100, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %103 = stablehlo.add %101, %102 : tensor<1x256x512xbf16>
    %104 = stablehlo.add %103, %15 : tensor<1x256x512xbf16>
    %105 = stablehlo.convert %104 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %106 = stablehlo.convert %105 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %107 = stablehlo.reduce(%106 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %108 = stablehlo.reshape %107 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %110 = stablehlo.divide %109, %23 : tensor<1x256x1xf64>
    %111 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %112 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %113 = stablehlo.subtract %111, %112 : tensor<1x256x512xf64>
    %114 = stablehlo.multiply %113, %113 : tensor<1x256x512xf64>
    %115 = stablehlo.reduce(%114 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %116 = stablehlo.reshape %115 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %118 = stablehlo.divide %117, %23 : tensor<1x256x1xf64>
    %119 = stablehlo.convert %118 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %120 = stablehlo.reduce(%105 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %121 = stablehlo.reshape %120 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %123 = stablehlo.divide %122, %39 : tensor<1x256x1xf32>
    %124 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %125 = stablehlo.add %124, %44 : tensor<1x256x1xf32>
    %126 = stablehlo.rsqrt %125 : tensor<1x256x1xf32>
    %127 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %128 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %129 = stablehlo.subtract %127, %128 : tensor<1x256x512xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %131 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %132 = stablehlo.multiply %130, %131 : tensor<1x256x512xf32>
    %133 = stablehlo.convert %arg7 : (tensor<512xbf16>) -> tensor<512xf32>
    %134 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %135 = stablehlo.broadcast_in_dim %133, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %136 = stablehlo.multiply %134, %135 : tensor<1x256x512xf32>
    %137 = stablehlo.convert %arg8 : (tensor<512xbf16>) -> tensor<512xf32>
    %138 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %139 = stablehlo.broadcast_in_dim %137, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %140 = stablehlo.add %138, %139 : tensor<1x256x512xf32>
    %141 = stablehlo.convert %140 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %142 = stablehlo.reshape %141 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %143 = stablehlo.convert %142 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %144 = stablehlo.dot_general %143, %arg101, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %146 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %147 = stablehlo.multiply %145, %146 : tensor<256x256xf32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %149 = stablehlo.broadcast_in_dim %arg102, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %150 = stablehlo.add %148, %149 : tensor<256x256xf32>
    %151 = stablehlo.convert %150 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %152 = stablehlo.reshape %151 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %153 = stablehlo.multiply %152, %cst_22 : tensor<1x256x256xbf16>
    %154 = stablehlo.rsqrt %cst_21 : tensor<1x256x256xbf16>
    %155 = stablehlo.multiply %152, %154 : tensor<1x256x256xbf16>
    %156 = stablehlo.convert %155 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %157 = stablehlo.clamp %cst_23, %156, %cst_24 : tensor<1x256x256xf32>
    %158 = stablehlo.multiply %157, %157 : tensor<1x256x256xf32>
    %159 = stablehlo.multiply %cst_25, %158 : tensor<1x256x256xf32>
    %160 = stablehlo.add %159, %cst_26 : tensor<1x256x256xf32>
    %161 = stablehlo.multiply %160, %158 : tensor<1x256x256xf32>
    %162 = stablehlo.add %161, %cst_27 : tensor<1x256x256xf32>
    %163 = stablehlo.multiply %162, %158 : tensor<1x256x256xf32>
    %164 = stablehlo.add %163, %cst_28 : tensor<1x256x256xf32>
    %165 = stablehlo.multiply %164, %158 : tensor<1x256x256xf32>
    %166 = stablehlo.add %165, %cst_29 : tensor<1x256x256xf32>
    %167 = stablehlo.multiply %166, %158 : tensor<1x256x256xf32>
    %168 = stablehlo.add %167, %cst_30 : tensor<1x256x256xf32>
    %169 = stablehlo.multiply %168, %158 : tensor<1x256x256xf32>
    %170 = stablehlo.add %169, %cst_31 : tensor<1x256x256xf32>
    %171 = stablehlo.multiply %cst_32, %158 : tensor<1x256x256xf32>
    %172 = stablehlo.add %171, %cst_33 : tensor<1x256x256xf32>
    %173 = stablehlo.multiply %172, %158 : tensor<1x256x256xf32>
    %174 = stablehlo.add %173, %cst_34 : tensor<1x256x256xf32>
    %175 = stablehlo.multiply %174, %158 : tensor<1x256x256xf32>
    %176 = stablehlo.add %175, %cst_35 : tensor<1x256x256xf32>
    %177 = stablehlo.multiply %176, %158 : tensor<1x256x256xf32>
    %178 = stablehlo.add %177, %cst_36 : tensor<1x256x256xf32>
    %179 = stablehlo.multiply %157, %170 : tensor<1x256x256xf32>
    %180 = stablehlo.divide %179, %178 : tensor<1x256x256xf32>
    %181 = stablehlo.clamp %cst_37, %180, %cst_38 : tensor<1x256x256xf32>
    %182 = stablehlo.convert %181 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %183 = stablehlo.add %182, %cst_20 : tensor<1x256x256xbf16>
    %184 = stablehlo.multiply %183, %153 : tensor<1x256x256xbf16>
    %185 = stablehlo.reshape %184 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %186 = stablehlo.convert %185 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %187 = stablehlo.dot_general %186, %arg103, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %189 = stablehlo.multiply %188, %9 : tensor<256x512xf32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %191 = stablehlo.broadcast_in_dim %arg104, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %192 = stablehlo.add %190, %191 : tensor<256x512xf32>
    %193 = stablehlo.convert %192 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %194 = stablehlo.reshape %193 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %195 = stablehlo.add %194, %104 : tensor<1x256x512xbf16>
    %196 = stablehlo.convert %195 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %197 = stablehlo.convert %196 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %198 = stablehlo.reduce(%197 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %199 = stablehlo.reshape %198 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %201 = stablehlo.divide %200, %23 : tensor<1x256x1xf64>
    %202 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %203 = stablehlo.broadcast_in_dim %201, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %204 = stablehlo.subtract %202, %203 : tensor<1x256x512xf64>
    %205 = stablehlo.multiply %204, %204 : tensor<1x256x512xf64>
    %206 = stablehlo.reduce(%205 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %207 = stablehlo.reshape %206 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %209 = stablehlo.divide %208, %23 : tensor<1x256x1xf64>
    %210 = stablehlo.convert %209 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %211 = stablehlo.reduce(%196 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %212 = stablehlo.reshape %211 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %214 = stablehlo.divide %213, %39 : tensor<1x256x1xf32>
    %215 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %216 = stablehlo.add %215, %44 : tensor<1x256x1xf32>
    %217 = stablehlo.rsqrt %216 : tensor<1x256x1xf32>
    %218 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %219 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %220 = stablehlo.subtract %218, %219 : tensor<1x256x512xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %222 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %223 = stablehlo.multiply %221, %222 : tensor<1x256x512xf32>
    %224 = stablehlo.convert %arg9 : (tensor<512xbf16>) -> tensor<512xf32>
    %225 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %226 = stablehlo.broadcast_in_dim %224, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %227 = stablehlo.multiply %225, %226 : tensor<1x256x512xf32>
    %228 = stablehlo.convert %arg10 : (tensor<512xbf16>) -> tensor<512xf32>
    %229 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %230 = stablehlo.broadcast_in_dim %228, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %231 = stablehlo.add %229, %230 : tensor<1x256x512xf32>
    %232 = stablehlo.convert %231 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %233 = stablehlo.convolution(%232, %arg11) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %234 = stablehlo.reshape %arg12 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %235 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %236 = stablehlo.broadcast_in_dim %234, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %237 = stablehlo.add %235, %236 : tensor<1x1024x512xbf16>
    %238 = stablehlo.multiply %237, %cst_3 : tensor<1x1024x512xbf16>
    %239 = stablehlo.multiply %237, %68 : tensor<1x1024x512xbf16>
    %240 = stablehlo.convert %239 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %241 = stablehlo.clamp %cst_4, %240, %cst_5 : tensor<1x1024x512xf32>
    %242 = stablehlo.multiply %241, %241 : tensor<1x1024x512xf32>
    %243 = stablehlo.multiply %cst_6, %242 : tensor<1x1024x512xf32>
    %244 = stablehlo.add %243, %cst_7 : tensor<1x1024x512xf32>
    %245 = stablehlo.multiply %244, %242 : tensor<1x1024x512xf32>
    %246 = stablehlo.add %245, %cst_8 : tensor<1x1024x512xf32>
    %247 = stablehlo.multiply %246, %242 : tensor<1x1024x512xf32>
    %248 = stablehlo.add %247, %cst_9 : tensor<1x1024x512xf32>
    %249 = stablehlo.multiply %248, %242 : tensor<1x1024x512xf32>
    %250 = stablehlo.add %249, %cst_10 : tensor<1x1024x512xf32>
    %251 = stablehlo.multiply %250, %242 : tensor<1x1024x512xf32>
    %252 = stablehlo.add %251, %cst_11 : tensor<1x1024x512xf32>
    %253 = stablehlo.multiply %252, %242 : tensor<1x1024x512xf32>
    %254 = stablehlo.add %253, %cst_12 : tensor<1x1024x512xf32>
    %255 = stablehlo.multiply %cst_13, %242 : tensor<1x1024x512xf32>
    %256 = stablehlo.add %255, %cst_14 : tensor<1x1024x512xf32>
    %257 = stablehlo.multiply %256, %242 : tensor<1x1024x512xf32>
    %258 = stablehlo.add %257, %cst_15 : tensor<1x1024x512xf32>
    %259 = stablehlo.multiply %258, %242 : tensor<1x1024x512xf32>
    %260 = stablehlo.add %259, %cst_16 : tensor<1x1024x512xf32>
    %261 = stablehlo.multiply %260, %242 : tensor<1x1024x512xf32>
    %262 = stablehlo.add %261, %cst_17 : tensor<1x1024x512xf32>
    %263 = stablehlo.multiply %241, %254 : tensor<1x1024x512xf32>
    %264 = stablehlo.divide %263, %262 : tensor<1x1024x512xf32>
    %265 = stablehlo.clamp %cst_18, %264, %cst_19 : tensor<1x1024x512xf32>
    %266 = stablehlo.convert %265 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %267 = stablehlo.add %266, %cst_1 : tensor<1x1024x512xbf16>
    %268 = stablehlo.multiply %267, %238 : tensor<1x1024x512xbf16>
    %269 = stablehlo.convolution(%268, %arg13) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %270 = stablehlo.reshape %arg14 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %271 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %272 = stablehlo.broadcast_in_dim %270, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %273 = stablehlo.add %271, %272 : tensor<1x256x512xbf16>
    %274 = stablehlo.add %273, %195 : tensor<1x256x512xbf16>
    %275 = stablehlo.convert %274 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %276 = stablehlo.convert %275 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %277 = stablehlo.reduce(%276 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %278 = stablehlo.reshape %277 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %280 = stablehlo.divide %279, %23 : tensor<1x256x1xf64>
    %281 = stablehlo.broadcast_in_dim %276, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %282 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %283 = stablehlo.subtract %281, %282 : tensor<1x256x512xf64>
    %284 = stablehlo.multiply %283, %283 : tensor<1x256x512xf64>
    %285 = stablehlo.reduce(%284 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %286 = stablehlo.reshape %285 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %288 = stablehlo.divide %287, %23 : tensor<1x256x1xf64>
    %289 = stablehlo.convert %288 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %290 = stablehlo.reduce(%275 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %291 = stablehlo.reshape %290 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %292 = stablehlo.broadcast_in_dim %291, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %293 = stablehlo.divide %292, %39 : tensor<1x256x1xf32>
    %294 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %295 = stablehlo.add %294, %44 : tensor<1x256x1xf32>
    %296 = stablehlo.rsqrt %295 : tensor<1x256x1xf32>
    %297 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %298 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %299 = stablehlo.subtract %297, %298 : tensor<1x256x512xf32>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %301 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %302 = stablehlo.multiply %300, %301 : tensor<1x256x512xf32>
    %303 = stablehlo.convert %arg15 : (tensor<512xbf16>) -> tensor<512xf32>
    %304 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %305 = stablehlo.broadcast_in_dim %303, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %306 = stablehlo.multiply %304, %305 : tensor<1x256x512xf32>
    %307 = stablehlo.convert %arg16 : (tensor<512xbf16>) -> tensor<512xf32>
    %308 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %309 = stablehlo.broadcast_in_dim %307, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %310 = stablehlo.add %308, %309 : tensor<1x256x512xf32>
    %311 = stablehlo.convert %310 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %312 = stablehlo.reshape %311 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %313 = stablehlo.convert %312 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %314 = stablehlo.dot_general %313, %arg105, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %316 = stablehlo.multiply %315, %146 : tensor<256x256xf32>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %318 = stablehlo.broadcast_in_dim %arg106, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %319 = stablehlo.add %317, %318 : tensor<256x256xf32>
    %320 = stablehlo.convert %319 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %321 = stablehlo.reshape %320 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %322 = stablehlo.multiply %321, %cst_22 : tensor<1x256x256xbf16>
    %323 = stablehlo.multiply %321, %154 : tensor<1x256x256xbf16>
    %324 = stablehlo.convert %323 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %325 = stablehlo.clamp %cst_23, %324, %cst_24 : tensor<1x256x256xf32>
    %326 = stablehlo.multiply %325, %325 : tensor<1x256x256xf32>
    %327 = stablehlo.multiply %cst_25, %326 : tensor<1x256x256xf32>
    %328 = stablehlo.add %327, %cst_26 : tensor<1x256x256xf32>
    %329 = stablehlo.multiply %328, %326 : tensor<1x256x256xf32>
    %330 = stablehlo.add %329, %cst_27 : tensor<1x256x256xf32>
    %331 = stablehlo.multiply %330, %326 : tensor<1x256x256xf32>
    %332 = stablehlo.add %331, %cst_28 : tensor<1x256x256xf32>
    %333 = stablehlo.multiply %332, %326 : tensor<1x256x256xf32>
    %334 = stablehlo.add %333, %cst_29 : tensor<1x256x256xf32>
    %335 = stablehlo.multiply %334, %326 : tensor<1x256x256xf32>
    %336 = stablehlo.add %335, %cst_30 : tensor<1x256x256xf32>
    %337 = stablehlo.multiply %336, %326 : tensor<1x256x256xf32>
    %338 = stablehlo.add %337, %cst_31 : tensor<1x256x256xf32>
    %339 = stablehlo.multiply %cst_32, %326 : tensor<1x256x256xf32>
    %340 = stablehlo.add %339, %cst_33 : tensor<1x256x256xf32>
    %341 = stablehlo.multiply %340, %326 : tensor<1x256x256xf32>
    %342 = stablehlo.add %341, %cst_34 : tensor<1x256x256xf32>
    %343 = stablehlo.multiply %342, %326 : tensor<1x256x256xf32>
    %344 = stablehlo.add %343, %cst_35 : tensor<1x256x256xf32>
    %345 = stablehlo.multiply %344, %326 : tensor<1x256x256xf32>
    %346 = stablehlo.add %345, %cst_36 : tensor<1x256x256xf32>
    %347 = stablehlo.multiply %325, %338 : tensor<1x256x256xf32>
    %348 = stablehlo.divide %347, %346 : tensor<1x256x256xf32>
    %349 = stablehlo.clamp %cst_37, %348, %cst_38 : tensor<1x256x256xf32>
    %350 = stablehlo.convert %349 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %351 = stablehlo.add %350, %cst_20 : tensor<1x256x256xbf16>
    %352 = stablehlo.multiply %351, %322 : tensor<1x256x256xbf16>
    %353 = stablehlo.reshape %352 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %354 = stablehlo.convert %353 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %355 = stablehlo.dot_general %354, %arg107, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %357 = stablehlo.multiply %356, %9 : tensor<256x512xf32>
    %358 = stablehlo.broadcast_in_dim %357, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %359 = stablehlo.broadcast_in_dim %arg108, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %360 = stablehlo.add %358, %359 : tensor<256x512xf32>
    %361 = stablehlo.convert %360 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %362 = stablehlo.reshape %361 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %363 = stablehlo.add %362, %274 : tensor<1x256x512xbf16>
    %364 = stablehlo.convert %363 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %365 = stablehlo.convert %364 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %366 = stablehlo.reduce(%365 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %367 = stablehlo.reshape %366 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %369 = stablehlo.divide %368, %23 : tensor<1x256x1xf64>
    %370 = stablehlo.broadcast_in_dim %365, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %371 = stablehlo.broadcast_in_dim %369, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %372 = stablehlo.subtract %370, %371 : tensor<1x256x512xf64>
    %373 = stablehlo.multiply %372, %372 : tensor<1x256x512xf64>
    %374 = stablehlo.reduce(%373 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %375 = stablehlo.reshape %374 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %377 = stablehlo.divide %376, %23 : tensor<1x256x1xf64>
    %378 = stablehlo.convert %377 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %379 = stablehlo.reduce(%364 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %380 = stablehlo.reshape %379 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %382 = stablehlo.divide %381, %39 : tensor<1x256x1xf32>
    %383 = stablehlo.broadcast_in_dim %378, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %384 = stablehlo.add %383, %44 : tensor<1x256x1xf32>
    %385 = stablehlo.rsqrt %384 : tensor<1x256x1xf32>
    %386 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %387 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %388 = stablehlo.subtract %386, %387 : tensor<1x256x512xf32>
    %389 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %390 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %391 = stablehlo.multiply %389, %390 : tensor<1x256x512xf32>
    %392 = stablehlo.convert %arg17 : (tensor<512xbf16>) -> tensor<512xf32>
    %393 = stablehlo.broadcast_in_dim %391, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %394 = stablehlo.broadcast_in_dim %392, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %395 = stablehlo.multiply %393, %394 : tensor<1x256x512xf32>
    %396 = stablehlo.convert %arg18 : (tensor<512xbf16>) -> tensor<512xf32>
    %397 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %398 = stablehlo.broadcast_in_dim %396, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %399 = stablehlo.add %397, %398 : tensor<1x256x512xf32>
    %400 = stablehlo.convert %399 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %401 = stablehlo.convolution(%400, %arg19) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %402 = stablehlo.reshape %arg20 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %403 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %404 = stablehlo.broadcast_in_dim %402, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %405 = stablehlo.add %403, %404 : tensor<1x1024x512xbf16>
    %406 = stablehlo.multiply %405, %cst_3 : tensor<1x1024x512xbf16>
    %407 = stablehlo.multiply %405, %68 : tensor<1x1024x512xbf16>
    %408 = stablehlo.convert %407 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %409 = stablehlo.clamp %cst_4, %408, %cst_5 : tensor<1x1024x512xf32>
    %410 = stablehlo.multiply %409, %409 : tensor<1x1024x512xf32>
    %411 = stablehlo.multiply %cst_6, %410 : tensor<1x1024x512xf32>
    %412 = stablehlo.add %411, %cst_7 : tensor<1x1024x512xf32>
    %413 = stablehlo.multiply %412, %410 : tensor<1x1024x512xf32>
    %414 = stablehlo.add %413, %cst_8 : tensor<1x1024x512xf32>
    %415 = stablehlo.multiply %414, %410 : tensor<1x1024x512xf32>
    %416 = stablehlo.add %415, %cst_9 : tensor<1x1024x512xf32>
    %417 = stablehlo.multiply %416, %410 : tensor<1x1024x512xf32>
    %418 = stablehlo.add %417, %cst_10 : tensor<1x1024x512xf32>
    %419 = stablehlo.multiply %418, %410 : tensor<1x1024x512xf32>
    %420 = stablehlo.add %419, %cst_11 : tensor<1x1024x512xf32>
    %421 = stablehlo.multiply %420, %410 : tensor<1x1024x512xf32>
    %422 = stablehlo.add %421, %cst_12 : tensor<1x1024x512xf32>
    %423 = stablehlo.multiply %cst_13, %410 : tensor<1x1024x512xf32>
    %424 = stablehlo.add %423, %cst_14 : tensor<1x1024x512xf32>
    %425 = stablehlo.multiply %424, %410 : tensor<1x1024x512xf32>
    %426 = stablehlo.add %425, %cst_15 : tensor<1x1024x512xf32>
    %427 = stablehlo.multiply %426, %410 : tensor<1x1024x512xf32>
    %428 = stablehlo.add %427, %cst_16 : tensor<1x1024x512xf32>
    %429 = stablehlo.multiply %428, %410 : tensor<1x1024x512xf32>
    %430 = stablehlo.add %429, %cst_17 : tensor<1x1024x512xf32>
    %431 = stablehlo.multiply %409, %422 : tensor<1x1024x512xf32>
    %432 = stablehlo.divide %431, %430 : tensor<1x1024x512xf32>
    %433 = stablehlo.clamp %cst_18, %432, %cst_19 : tensor<1x1024x512xf32>
    %434 = stablehlo.convert %433 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %435 = stablehlo.add %434, %cst_1 : tensor<1x1024x512xbf16>
    %436 = stablehlo.multiply %435, %406 : tensor<1x1024x512xbf16>
    %437 = stablehlo.convolution(%436, %arg21) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %438 = stablehlo.reshape %arg22 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %439 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %440 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %441 = stablehlo.add %439, %440 : tensor<1x256x512xbf16>
    %442 = stablehlo.add %441, %363 : tensor<1x256x512xbf16>
    %443 = stablehlo.convert %442 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %444 = stablehlo.convert %443 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %445 = stablehlo.reduce(%444 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %446 = stablehlo.reshape %445 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %448 = stablehlo.divide %447, %23 : tensor<1x256x1xf64>
    %449 = stablehlo.broadcast_in_dim %444, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %450 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %451 = stablehlo.subtract %449, %450 : tensor<1x256x512xf64>
    %452 = stablehlo.multiply %451, %451 : tensor<1x256x512xf64>
    %453 = stablehlo.reduce(%452 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %454 = stablehlo.reshape %453 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %455 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %456 = stablehlo.divide %455, %23 : tensor<1x256x1xf64>
    %457 = stablehlo.convert %456 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %458 = stablehlo.reduce(%443 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %459 = stablehlo.reshape %458 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %461 = stablehlo.divide %460, %39 : tensor<1x256x1xf32>
    %462 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %463 = stablehlo.add %462, %44 : tensor<1x256x1xf32>
    %464 = stablehlo.rsqrt %463 : tensor<1x256x1xf32>
    %465 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %466 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %467 = stablehlo.subtract %465, %466 : tensor<1x256x512xf32>
    %468 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %469 = stablehlo.broadcast_in_dim %464, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %470 = stablehlo.multiply %468, %469 : tensor<1x256x512xf32>
    %471 = stablehlo.convert %arg23 : (tensor<512xbf16>) -> tensor<512xf32>
    %472 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %473 = stablehlo.broadcast_in_dim %471, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %474 = stablehlo.multiply %472, %473 : tensor<1x256x512xf32>
    %475 = stablehlo.convert %arg24 : (tensor<512xbf16>) -> tensor<512xf32>
    %476 = stablehlo.broadcast_in_dim %474, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %477 = stablehlo.broadcast_in_dim %475, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %478 = stablehlo.add %476, %477 : tensor<1x256x512xf32>
    %479 = stablehlo.convert %478 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %480 = stablehlo.reshape %479 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %481 = stablehlo.convert %480 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %482 = stablehlo.dot_general %481, %arg109, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %484 = stablehlo.multiply %483, %146 : tensor<256x256xf32>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %486 = stablehlo.broadcast_in_dim %arg110, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %487 = stablehlo.add %485, %486 : tensor<256x256xf32>
    %488 = stablehlo.convert %487 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %489 = stablehlo.reshape %488 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %490 = stablehlo.multiply %489, %cst_22 : tensor<1x256x256xbf16>
    %491 = stablehlo.multiply %489, %154 : tensor<1x256x256xbf16>
    %492 = stablehlo.convert %491 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %493 = stablehlo.clamp %cst_23, %492, %cst_24 : tensor<1x256x256xf32>
    %494 = stablehlo.multiply %493, %493 : tensor<1x256x256xf32>
    %495 = stablehlo.multiply %cst_25, %494 : tensor<1x256x256xf32>
    %496 = stablehlo.add %495, %cst_26 : tensor<1x256x256xf32>
    %497 = stablehlo.multiply %496, %494 : tensor<1x256x256xf32>
    %498 = stablehlo.add %497, %cst_27 : tensor<1x256x256xf32>
    %499 = stablehlo.multiply %498, %494 : tensor<1x256x256xf32>
    %500 = stablehlo.add %499, %cst_28 : tensor<1x256x256xf32>
    %501 = stablehlo.multiply %500, %494 : tensor<1x256x256xf32>
    %502 = stablehlo.add %501, %cst_29 : tensor<1x256x256xf32>
    %503 = stablehlo.multiply %502, %494 : tensor<1x256x256xf32>
    %504 = stablehlo.add %503, %cst_30 : tensor<1x256x256xf32>
    %505 = stablehlo.multiply %504, %494 : tensor<1x256x256xf32>
    %506 = stablehlo.add %505, %cst_31 : tensor<1x256x256xf32>
    %507 = stablehlo.multiply %cst_32, %494 : tensor<1x256x256xf32>
    %508 = stablehlo.add %507, %cst_33 : tensor<1x256x256xf32>
    %509 = stablehlo.multiply %508, %494 : tensor<1x256x256xf32>
    %510 = stablehlo.add %509, %cst_34 : tensor<1x256x256xf32>
    %511 = stablehlo.multiply %510, %494 : tensor<1x256x256xf32>
    %512 = stablehlo.add %511, %cst_35 : tensor<1x256x256xf32>
    %513 = stablehlo.multiply %512, %494 : tensor<1x256x256xf32>
    %514 = stablehlo.add %513, %cst_36 : tensor<1x256x256xf32>
    %515 = stablehlo.multiply %493, %506 : tensor<1x256x256xf32>
    %516 = stablehlo.divide %515, %514 : tensor<1x256x256xf32>
    %517 = stablehlo.clamp %cst_37, %516, %cst_38 : tensor<1x256x256xf32>
    %518 = stablehlo.convert %517 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %519 = stablehlo.add %518, %cst_20 : tensor<1x256x256xbf16>
    %520 = stablehlo.multiply %519, %490 : tensor<1x256x256xbf16>
    %521 = stablehlo.reshape %520 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %522 = stablehlo.convert %521 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %523 = stablehlo.dot_general %522, %arg111, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %524 = stablehlo.broadcast_in_dim %523, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %525 = stablehlo.multiply %524, %9 : tensor<256x512xf32>
    %526 = stablehlo.broadcast_in_dim %525, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %527 = stablehlo.broadcast_in_dim %arg112, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %528 = stablehlo.add %526, %527 : tensor<256x512xf32>
    %529 = stablehlo.convert %528 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %530 = stablehlo.reshape %529 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %531 = stablehlo.add %530, %442 : tensor<1x256x512xbf16>
    %532 = stablehlo.convert %531 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %533 = stablehlo.convert %532 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %534 = stablehlo.reduce(%533 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %535 = stablehlo.reshape %534 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %536 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %537 = stablehlo.divide %536, %23 : tensor<1x256x1xf64>
    %538 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %539 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %540 = stablehlo.subtract %538, %539 : tensor<1x256x512xf64>
    %541 = stablehlo.multiply %540, %540 : tensor<1x256x512xf64>
    %542 = stablehlo.reduce(%541 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %543 = stablehlo.reshape %542 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %545 = stablehlo.divide %544, %23 : tensor<1x256x1xf64>
    %546 = stablehlo.convert %545 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %547 = stablehlo.reduce(%532 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %548 = stablehlo.reshape %547 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %549 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %550 = stablehlo.divide %549, %39 : tensor<1x256x1xf32>
    %551 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %552 = stablehlo.add %551, %44 : tensor<1x256x1xf32>
    %553 = stablehlo.rsqrt %552 : tensor<1x256x1xf32>
    %554 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %555 = stablehlo.broadcast_in_dim %550, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %556 = stablehlo.subtract %554, %555 : tensor<1x256x512xf32>
    %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %558 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %559 = stablehlo.multiply %557, %558 : tensor<1x256x512xf32>
    %560 = stablehlo.convert %arg25 : (tensor<512xbf16>) -> tensor<512xf32>
    %561 = stablehlo.broadcast_in_dim %559, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %562 = stablehlo.broadcast_in_dim %560, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %563 = stablehlo.multiply %561, %562 : tensor<1x256x512xf32>
    %564 = stablehlo.convert %arg26 : (tensor<512xbf16>) -> tensor<512xf32>
    %565 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %566 = stablehlo.broadcast_in_dim %564, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %567 = stablehlo.add %565, %566 : tensor<1x256x512xf32>
    %568 = stablehlo.convert %567 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %569 = stablehlo.convolution(%568, %arg27) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %570 = stablehlo.reshape %arg28 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %571 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %572 = stablehlo.broadcast_in_dim %570, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %573 = stablehlo.add %571, %572 : tensor<1x1024x512xbf16>
    %574 = stablehlo.multiply %573, %cst_3 : tensor<1x1024x512xbf16>
    %575 = stablehlo.multiply %573, %68 : tensor<1x1024x512xbf16>
    %576 = stablehlo.convert %575 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %577 = stablehlo.clamp %cst_4, %576, %cst_5 : tensor<1x1024x512xf32>
    %578 = stablehlo.multiply %577, %577 : tensor<1x1024x512xf32>
    %579 = stablehlo.multiply %cst_6, %578 : tensor<1x1024x512xf32>
    %580 = stablehlo.add %579, %cst_7 : tensor<1x1024x512xf32>
    %581 = stablehlo.multiply %580, %578 : tensor<1x1024x512xf32>
    %582 = stablehlo.add %581, %cst_8 : tensor<1x1024x512xf32>
    %583 = stablehlo.multiply %582, %578 : tensor<1x1024x512xf32>
    %584 = stablehlo.add %583, %cst_9 : tensor<1x1024x512xf32>
    %585 = stablehlo.multiply %584, %578 : tensor<1x1024x512xf32>
    %586 = stablehlo.add %585, %cst_10 : tensor<1x1024x512xf32>
    %587 = stablehlo.multiply %586, %578 : tensor<1x1024x512xf32>
    %588 = stablehlo.add %587, %cst_11 : tensor<1x1024x512xf32>
    %589 = stablehlo.multiply %588, %578 : tensor<1x1024x512xf32>
    %590 = stablehlo.add %589, %cst_12 : tensor<1x1024x512xf32>
    %591 = stablehlo.multiply %cst_13, %578 : tensor<1x1024x512xf32>
    %592 = stablehlo.add %591, %cst_14 : tensor<1x1024x512xf32>
    %593 = stablehlo.multiply %592, %578 : tensor<1x1024x512xf32>
    %594 = stablehlo.add %593, %cst_15 : tensor<1x1024x512xf32>
    %595 = stablehlo.multiply %594, %578 : tensor<1x1024x512xf32>
    %596 = stablehlo.add %595, %cst_16 : tensor<1x1024x512xf32>
    %597 = stablehlo.multiply %596, %578 : tensor<1x1024x512xf32>
    %598 = stablehlo.add %597, %cst_17 : tensor<1x1024x512xf32>
    %599 = stablehlo.multiply %577, %590 : tensor<1x1024x512xf32>
    %600 = stablehlo.divide %599, %598 : tensor<1x1024x512xf32>
    %601 = stablehlo.clamp %cst_18, %600, %cst_19 : tensor<1x1024x512xf32>
    %602 = stablehlo.convert %601 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %603 = stablehlo.add %602, %cst_1 : tensor<1x1024x512xbf16>
    %604 = stablehlo.multiply %603, %574 : tensor<1x1024x512xbf16>
    %605 = stablehlo.convolution(%604, %arg29) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %606 = stablehlo.reshape %arg30 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %607 = stablehlo.broadcast_in_dim %605, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %608 = stablehlo.broadcast_in_dim %606, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %609 = stablehlo.add %607, %608 : tensor<1x256x512xbf16>
    %610 = stablehlo.add %609, %531 : tensor<1x256x512xbf16>
    %611 = stablehlo.convert %610 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %612 = stablehlo.convert %611 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %613 = stablehlo.reduce(%612 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %614 = stablehlo.reshape %613 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %616 = stablehlo.divide %615, %23 : tensor<1x256x1xf64>
    %617 = stablehlo.broadcast_in_dim %612, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %618 = stablehlo.broadcast_in_dim %616, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %619 = stablehlo.subtract %617, %618 : tensor<1x256x512xf64>
    %620 = stablehlo.multiply %619, %619 : tensor<1x256x512xf64>
    %621 = stablehlo.reduce(%620 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %622 = stablehlo.reshape %621 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %623 = stablehlo.broadcast_in_dim %622, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %624 = stablehlo.divide %623, %23 : tensor<1x256x1xf64>
    %625 = stablehlo.convert %624 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %626 = stablehlo.reduce(%611 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %627 = stablehlo.reshape %626 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %628 = stablehlo.broadcast_in_dim %627, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %629 = stablehlo.divide %628, %39 : tensor<1x256x1xf32>
    %630 = stablehlo.broadcast_in_dim %625, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %631 = stablehlo.add %630, %44 : tensor<1x256x1xf32>
    %632 = stablehlo.rsqrt %631 : tensor<1x256x1xf32>
    %633 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %634 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %635 = stablehlo.subtract %633, %634 : tensor<1x256x512xf32>
    %636 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %637 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %638 = stablehlo.multiply %636, %637 : tensor<1x256x512xf32>
    %639 = stablehlo.convert %arg31 : (tensor<512xbf16>) -> tensor<512xf32>
    %640 = stablehlo.broadcast_in_dim %638, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %641 = stablehlo.broadcast_in_dim %639, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %642 = stablehlo.multiply %640, %641 : tensor<1x256x512xf32>
    %643 = stablehlo.convert %arg32 : (tensor<512xbf16>) -> tensor<512xf32>
    %644 = stablehlo.broadcast_in_dim %642, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %645 = stablehlo.broadcast_in_dim %643, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %646 = stablehlo.add %644, %645 : tensor<1x256x512xf32>
    %647 = stablehlo.convert %646 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %648 = stablehlo.reshape %647 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %649 = stablehlo.convert %648 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %650 = stablehlo.dot_general %649, %arg113, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %651 = stablehlo.broadcast_in_dim %650, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %652 = stablehlo.multiply %651, %146 : tensor<256x256xf32>
    %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %654 = stablehlo.broadcast_in_dim %arg114, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %655 = stablehlo.add %653, %654 : tensor<256x256xf32>
    %656 = stablehlo.convert %655 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %657 = stablehlo.reshape %656 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %658 = stablehlo.multiply %657, %cst_22 : tensor<1x256x256xbf16>
    %659 = stablehlo.multiply %657, %154 : tensor<1x256x256xbf16>
    %660 = stablehlo.convert %659 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %661 = stablehlo.clamp %cst_23, %660, %cst_24 : tensor<1x256x256xf32>
    %662 = stablehlo.multiply %661, %661 : tensor<1x256x256xf32>
    %663 = stablehlo.multiply %cst_25, %662 : tensor<1x256x256xf32>
    %664 = stablehlo.add %663, %cst_26 : tensor<1x256x256xf32>
    %665 = stablehlo.multiply %664, %662 : tensor<1x256x256xf32>
    %666 = stablehlo.add %665, %cst_27 : tensor<1x256x256xf32>
    %667 = stablehlo.multiply %666, %662 : tensor<1x256x256xf32>
    %668 = stablehlo.add %667, %cst_28 : tensor<1x256x256xf32>
    %669 = stablehlo.multiply %668, %662 : tensor<1x256x256xf32>
    %670 = stablehlo.add %669, %cst_29 : tensor<1x256x256xf32>
    %671 = stablehlo.multiply %670, %662 : tensor<1x256x256xf32>
    %672 = stablehlo.add %671, %cst_30 : tensor<1x256x256xf32>
    %673 = stablehlo.multiply %672, %662 : tensor<1x256x256xf32>
    %674 = stablehlo.add %673, %cst_31 : tensor<1x256x256xf32>
    %675 = stablehlo.multiply %cst_32, %662 : tensor<1x256x256xf32>
    %676 = stablehlo.add %675, %cst_33 : tensor<1x256x256xf32>
    %677 = stablehlo.multiply %676, %662 : tensor<1x256x256xf32>
    %678 = stablehlo.add %677, %cst_34 : tensor<1x256x256xf32>
    %679 = stablehlo.multiply %678, %662 : tensor<1x256x256xf32>
    %680 = stablehlo.add %679, %cst_35 : tensor<1x256x256xf32>
    %681 = stablehlo.multiply %680, %662 : tensor<1x256x256xf32>
    %682 = stablehlo.add %681, %cst_36 : tensor<1x256x256xf32>
    %683 = stablehlo.multiply %661, %674 : tensor<1x256x256xf32>
    %684 = stablehlo.divide %683, %682 : tensor<1x256x256xf32>
    %685 = stablehlo.clamp %cst_37, %684, %cst_38 : tensor<1x256x256xf32>
    %686 = stablehlo.convert %685 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %687 = stablehlo.add %686, %cst_20 : tensor<1x256x256xbf16>
    %688 = stablehlo.multiply %687, %658 : tensor<1x256x256xbf16>
    %689 = stablehlo.reshape %688 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %690 = stablehlo.convert %689 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %691 = stablehlo.dot_general %690, %arg115, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %692 = stablehlo.broadcast_in_dim %691, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %693 = stablehlo.multiply %692, %9 : tensor<256x512xf32>
    %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %695 = stablehlo.broadcast_in_dim %arg116, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %696 = stablehlo.add %694, %695 : tensor<256x512xf32>
    %697 = stablehlo.convert %696 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %698 = stablehlo.reshape %697 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %699 = stablehlo.add %698, %610 : tensor<1x256x512xbf16>
    %700 = stablehlo.convert %699 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %701 = stablehlo.convert %700 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %702 = stablehlo.reduce(%701 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %703 = stablehlo.reshape %702 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %704 = stablehlo.broadcast_in_dim %703, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %705 = stablehlo.divide %704, %23 : tensor<1x256x1xf64>
    %706 = stablehlo.broadcast_in_dim %701, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %707 = stablehlo.broadcast_in_dim %705, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %708 = stablehlo.subtract %706, %707 : tensor<1x256x512xf64>
    %709 = stablehlo.multiply %708, %708 : tensor<1x256x512xf64>
    %710 = stablehlo.reduce(%709 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %711 = stablehlo.reshape %710 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %713 = stablehlo.divide %712, %23 : tensor<1x256x1xf64>
    %714 = stablehlo.convert %713 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %715 = stablehlo.reduce(%700 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %716 = stablehlo.reshape %715 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %718 = stablehlo.divide %717, %39 : tensor<1x256x1xf32>
    %719 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %720 = stablehlo.add %719, %44 : tensor<1x256x1xf32>
    %721 = stablehlo.rsqrt %720 : tensor<1x256x1xf32>
    %722 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %723 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %724 = stablehlo.subtract %722, %723 : tensor<1x256x512xf32>
    %725 = stablehlo.broadcast_in_dim %724, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %726 = stablehlo.broadcast_in_dim %721, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %727 = stablehlo.multiply %725, %726 : tensor<1x256x512xf32>
    %728 = stablehlo.convert %arg33 : (tensor<512xbf16>) -> tensor<512xf32>
    %729 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %730 = stablehlo.broadcast_in_dim %728, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %731 = stablehlo.multiply %729, %730 : tensor<1x256x512xf32>
    %732 = stablehlo.convert %arg34 : (tensor<512xbf16>) -> tensor<512xf32>
    %733 = stablehlo.broadcast_in_dim %731, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %734 = stablehlo.broadcast_in_dim %732, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %735 = stablehlo.add %733, %734 : tensor<1x256x512xf32>
    %736 = stablehlo.convert %735 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %737 = stablehlo.convolution(%736, %arg35) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %738 = stablehlo.reshape %arg36 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %739 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %740 = stablehlo.broadcast_in_dim %738, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %741 = stablehlo.add %739, %740 : tensor<1x1024x512xbf16>
    %742 = stablehlo.multiply %741, %cst_3 : tensor<1x1024x512xbf16>
    %743 = stablehlo.multiply %741, %68 : tensor<1x1024x512xbf16>
    %744 = stablehlo.convert %743 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %745 = stablehlo.clamp %cst_4, %744, %cst_5 : tensor<1x1024x512xf32>
    %746 = stablehlo.multiply %745, %745 : tensor<1x1024x512xf32>
    %747 = stablehlo.multiply %cst_6, %746 : tensor<1x1024x512xf32>
    %748 = stablehlo.add %747, %cst_7 : tensor<1x1024x512xf32>
    %749 = stablehlo.multiply %748, %746 : tensor<1x1024x512xf32>
    %750 = stablehlo.add %749, %cst_8 : tensor<1x1024x512xf32>
    %751 = stablehlo.multiply %750, %746 : tensor<1x1024x512xf32>
    %752 = stablehlo.add %751, %cst_9 : tensor<1x1024x512xf32>
    %753 = stablehlo.multiply %752, %746 : tensor<1x1024x512xf32>
    %754 = stablehlo.add %753, %cst_10 : tensor<1x1024x512xf32>
    %755 = stablehlo.multiply %754, %746 : tensor<1x1024x512xf32>
    %756 = stablehlo.add %755, %cst_11 : tensor<1x1024x512xf32>
    %757 = stablehlo.multiply %756, %746 : tensor<1x1024x512xf32>
    %758 = stablehlo.add %757, %cst_12 : tensor<1x1024x512xf32>
    %759 = stablehlo.multiply %cst_13, %746 : tensor<1x1024x512xf32>
    %760 = stablehlo.add %759, %cst_14 : tensor<1x1024x512xf32>
    %761 = stablehlo.multiply %760, %746 : tensor<1x1024x512xf32>
    %762 = stablehlo.add %761, %cst_15 : tensor<1x1024x512xf32>
    %763 = stablehlo.multiply %762, %746 : tensor<1x1024x512xf32>
    %764 = stablehlo.add %763, %cst_16 : tensor<1x1024x512xf32>
    %765 = stablehlo.multiply %764, %746 : tensor<1x1024x512xf32>
    %766 = stablehlo.add %765, %cst_17 : tensor<1x1024x512xf32>
    %767 = stablehlo.multiply %745, %758 : tensor<1x1024x512xf32>
    %768 = stablehlo.divide %767, %766 : tensor<1x1024x512xf32>
    %769 = stablehlo.clamp %cst_18, %768, %cst_19 : tensor<1x1024x512xf32>
    %770 = stablehlo.convert %769 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %771 = stablehlo.add %770, %cst_1 : tensor<1x1024x512xbf16>
    %772 = stablehlo.multiply %771, %742 : tensor<1x1024x512xbf16>
    %773 = stablehlo.convolution(%772, %arg37) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %774 = stablehlo.reshape %arg38 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %775 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %776 = stablehlo.broadcast_in_dim %774, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %777 = stablehlo.add %775, %776 : tensor<1x256x512xbf16>
    %778 = stablehlo.add %777, %699 : tensor<1x256x512xbf16>
    %779 = stablehlo.convert %778 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %780 = stablehlo.convert %779 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %781 = stablehlo.reduce(%780 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %782 = stablehlo.reshape %781 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %783 = stablehlo.broadcast_in_dim %782, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %784 = stablehlo.divide %783, %23 : tensor<1x256x1xf64>
    %785 = stablehlo.broadcast_in_dim %780, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %786 = stablehlo.broadcast_in_dim %784, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %787 = stablehlo.subtract %785, %786 : tensor<1x256x512xf64>
    %788 = stablehlo.multiply %787, %787 : tensor<1x256x512xf64>
    %789 = stablehlo.reduce(%788 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %790 = stablehlo.reshape %789 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %792 = stablehlo.divide %791, %23 : tensor<1x256x1xf64>
    %793 = stablehlo.convert %792 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %794 = stablehlo.reduce(%779 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %795 = stablehlo.reshape %794 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %797 = stablehlo.divide %796, %39 : tensor<1x256x1xf32>
    %798 = stablehlo.broadcast_in_dim %793, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %799 = stablehlo.add %798, %44 : tensor<1x256x1xf32>
    %800 = stablehlo.rsqrt %799 : tensor<1x256x1xf32>
    %801 = stablehlo.broadcast_in_dim %779, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %802 = stablehlo.broadcast_in_dim %797, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %803 = stablehlo.subtract %801, %802 : tensor<1x256x512xf32>
    %804 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %805 = stablehlo.broadcast_in_dim %800, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %806 = stablehlo.multiply %804, %805 : tensor<1x256x512xf32>
    %807 = stablehlo.convert %arg39 : (tensor<512xbf16>) -> tensor<512xf32>
    %808 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %809 = stablehlo.broadcast_in_dim %807, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %810 = stablehlo.multiply %808, %809 : tensor<1x256x512xf32>
    %811 = stablehlo.convert %arg40 : (tensor<512xbf16>) -> tensor<512xf32>
    %812 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %813 = stablehlo.broadcast_in_dim %811, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %814 = stablehlo.add %812, %813 : tensor<1x256x512xf32>
    %815 = stablehlo.convert %814 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %816 = stablehlo.reshape %815 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %817 = stablehlo.convert %816 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %818 = stablehlo.dot_general %817, %arg117, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %819 = stablehlo.broadcast_in_dim %818, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %820 = stablehlo.multiply %819, %146 : tensor<256x256xf32>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %822 = stablehlo.broadcast_in_dim %arg118, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %823 = stablehlo.add %821, %822 : tensor<256x256xf32>
    %824 = stablehlo.convert %823 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %825 = stablehlo.reshape %824 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %826 = stablehlo.multiply %825, %cst_22 : tensor<1x256x256xbf16>
    %827 = stablehlo.multiply %825, %154 : tensor<1x256x256xbf16>
    %828 = stablehlo.convert %827 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %829 = stablehlo.clamp %cst_23, %828, %cst_24 : tensor<1x256x256xf32>
    %830 = stablehlo.multiply %829, %829 : tensor<1x256x256xf32>
    %831 = stablehlo.multiply %cst_25, %830 : tensor<1x256x256xf32>
    %832 = stablehlo.add %831, %cst_26 : tensor<1x256x256xf32>
    %833 = stablehlo.multiply %832, %830 : tensor<1x256x256xf32>
    %834 = stablehlo.add %833, %cst_27 : tensor<1x256x256xf32>
    %835 = stablehlo.multiply %834, %830 : tensor<1x256x256xf32>
    %836 = stablehlo.add %835, %cst_28 : tensor<1x256x256xf32>
    %837 = stablehlo.multiply %836, %830 : tensor<1x256x256xf32>
    %838 = stablehlo.add %837, %cst_29 : tensor<1x256x256xf32>
    %839 = stablehlo.multiply %838, %830 : tensor<1x256x256xf32>
    %840 = stablehlo.add %839, %cst_30 : tensor<1x256x256xf32>
    %841 = stablehlo.multiply %840, %830 : tensor<1x256x256xf32>
    %842 = stablehlo.add %841, %cst_31 : tensor<1x256x256xf32>
    %843 = stablehlo.multiply %cst_32, %830 : tensor<1x256x256xf32>
    %844 = stablehlo.add %843, %cst_33 : tensor<1x256x256xf32>
    %845 = stablehlo.multiply %844, %830 : tensor<1x256x256xf32>
    %846 = stablehlo.add %845, %cst_34 : tensor<1x256x256xf32>
    %847 = stablehlo.multiply %846, %830 : tensor<1x256x256xf32>
    %848 = stablehlo.add %847, %cst_35 : tensor<1x256x256xf32>
    %849 = stablehlo.multiply %848, %830 : tensor<1x256x256xf32>
    %850 = stablehlo.add %849, %cst_36 : tensor<1x256x256xf32>
    %851 = stablehlo.multiply %829, %842 : tensor<1x256x256xf32>
    %852 = stablehlo.divide %851, %850 : tensor<1x256x256xf32>
    %853 = stablehlo.clamp %cst_37, %852, %cst_38 : tensor<1x256x256xf32>
    %854 = stablehlo.convert %853 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %855 = stablehlo.add %854, %cst_20 : tensor<1x256x256xbf16>
    %856 = stablehlo.multiply %855, %826 : tensor<1x256x256xbf16>
    %857 = stablehlo.reshape %856 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %858 = stablehlo.convert %857 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %859 = stablehlo.dot_general %858, %arg119, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %860 = stablehlo.broadcast_in_dim %859, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %861 = stablehlo.multiply %860, %9 : tensor<256x512xf32>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %863 = stablehlo.broadcast_in_dim %arg120, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %864 = stablehlo.add %862, %863 : tensor<256x512xf32>
    %865 = stablehlo.convert %864 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %866 = stablehlo.reshape %865 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %867 = stablehlo.add %866, %778 : tensor<1x256x512xbf16>
    %868 = stablehlo.convert %867 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %869 = stablehlo.convert %868 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %870 = stablehlo.reduce(%869 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %871 = stablehlo.reshape %870 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %872 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %873 = stablehlo.divide %872, %23 : tensor<1x256x1xf64>
    %874 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %875 = stablehlo.broadcast_in_dim %873, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %876 = stablehlo.subtract %874, %875 : tensor<1x256x512xf64>
    %877 = stablehlo.multiply %876, %876 : tensor<1x256x512xf64>
    %878 = stablehlo.reduce(%877 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %879 = stablehlo.reshape %878 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %880 = stablehlo.broadcast_in_dim %879, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %881 = stablehlo.divide %880, %23 : tensor<1x256x1xf64>
    %882 = stablehlo.convert %881 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %883 = stablehlo.reduce(%868 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %884 = stablehlo.reshape %883 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %885 = stablehlo.broadcast_in_dim %884, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %886 = stablehlo.divide %885, %39 : tensor<1x256x1xf32>
    %887 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %888 = stablehlo.add %887, %44 : tensor<1x256x1xf32>
    %889 = stablehlo.rsqrt %888 : tensor<1x256x1xf32>
    %890 = stablehlo.broadcast_in_dim %868, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %891 = stablehlo.broadcast_in_dim %886, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %892 = stablehlo.subtract %890, %891 : tensor<1x256x512xf32>
    %893 = stablehlo.broadcast_in_dim %892, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %894 = stablehlo.broadcast_in_dim %889, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %895 = stablehlo.multiply %893, %894 : tensor<1x256x512xf32>
    %896 = stablehlo.convert %arg41 : (tensor<512xbf16>) -> tensor<512xf32>
    %897 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %898 = stablehlo.broadcast_in_dim %896, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %899 = stablehlo.multiply %897, %898 : tensor<1x256x512xf32>
    %900 = stablehlo.convert %arg42 : (tensor<512xbf16>) -> tensor<512xf32>
    %901 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %902 = stablehlo.broadcast_in_dim %900, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %903 = stablehlo.add %901, %902 : tensor<1x256x512xf32>
    %904 = stablehlo.convert %903 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %905 = stablehlo.convolution(%904, %arg43) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %906 = stablehlo.reshape %arg44 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %907 = stablehlo.broadcast_in_dim %905, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %908 = stablehlo.broadcast_in_dim %906, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %909 = stablehlo.add %907, %908 : tensor<1x1024x512xbf16>
    %910 = stablehlo.multiply %909, %cst_3 : tensor<1x1024x512xbf16>
    %911 = stablehlo.multiply %909, %68 : tensor<1x1024x512xbf16>
    %912 = stablehlo.convert %911 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %913 = stablehlo.clamp %cst_4, %912, %cst_5 : tensor<1x1024x512xf32>
    %914 = stablehlo.multiply %913, %913 : tensor<1x1024x512xf32>
    %915 = stablehlo.multiply %cst_6, %914 : tensor<1x1024x512xf32>
    %916 = stablehlo.add %915, %cst_7 : tensor<1x1024x512xf32>
    %917 = stablehlo.multiply %916, %914 : tensor<1x1024x512xf32>
    %918 = stablehlo.add %917, %cst_8 : tensor<1x1024x512xf32>
    %919 = stablehlo.multiply %918, %914 : tensor<1x1024x512xf32>
    %920 = stablehlo.add %919, %cst_9 : tensor<1x1024x512xf32>
    %921 = stablehlo.multiply %920, %914 : tensor<1x1024x512xf32>
    %922 = stablehlo.add %921, %cst_10 : tensor<1x1024x512xf32>
    %923 = stablehlo.multiply %922, %914 : tensor<1x1024x512xf32>
    %924 = stablehlo.add %923, %cst_11 : tensor<1x1024x512xf32>
    %925 = stablehlo.multiply %924, %914 : tensor<1x1024x512xf32>
    %926 = stablehlo.add %925, %cst_12 : tensor<1x1024x512xf32>
    %927 = stablehlo.multiply %cst_13, %914 : tensor<1x1024x512xf32>
    %928 = stablehlo.add %927, %cst_14 : tensor<1x1024x512xf32>
    %929 = stablehlo.multiply %928, %914 : tensor<1x1024x512xf32>
    %930 = stablehlo.add %929, %cst_15 : tensor<1x1024x512xf32>
    %931 = stablehlo.multiply %930, %914 : tensor<1x1024x512xf32>
    %932 = stablehlo.add %931, %cst_16 : tensor<1x1024x512xf32>
    %933 = stablehlo.multiply %932, %914 : tensor<1x1024x512xf32>
    %934 = stablehlo.add %933, %cst_17 : tensor<1x1024x512xf32>
    %935 = stablehlo.multiply %913, %926 : tensor<1x1024x512xf32>
    %936 = stablehlo.divide %935, %934 : tensor<1x1024x512xf32>
    %937 = stablehlo.clamp %cst_18, %936, %cst_19 : tensor<1x1024x512xf32>
    %938 = stablehlo.convert %937 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %939 = stablehlo.add %938, %cst_1 : tensor<1x1024x512xbf16>
    %940 = stablehlo.multiply %939, %910 : tensor<1x1024x512xbf16>
    %941 = stablehlo.convolution(%940, %arg45) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %942 = stablehlo.reshape %arg46 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %943 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %944 = stablehlo.broadcast_in_dim %942, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %945 = stablehlo.add %943, %944 : tensor<1x256x512xbf16>
    %946 = stablehlo.add %945, %867 : tensor<1x256x512xbf16>
    %947 = stablehlo.convert %946 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %948 = stablehlo.convert %947 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %949 = stablehlo.reduce(%948 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %950 = stablehlo.reshape %949 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %951 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %952 = stablehlo.divide %951, %23 : tensor<1x256x1xf64>
    %953 = stablehlo.broadcast_in_dim %948, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %954 = stablehlo.broadcast_in_dim %952, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %955 = stablehlo.subtract %953, %954 : tensor<1x256x512xf64>
    %956 = stablehlo.multiply %955, %955 : tensor<1x256x512xf64>
    %957 = stablehlo.reduce(%956 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %958 = stablehlo.reshape %957 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %959 = stablehlo.broadcast_in_dim %958, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %960 = stablehlo.divide %959, %23 : tensor<1x256x1xf64>
    %961 = stablehlo.convert %960 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %962 = stablehlo.reduce(%947 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %963 = stablehlo.reshape %962 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %964 = stablehlo.broadcast_in_dim %963, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %965 = stablehlo.divide %964, %39 : tensor<1x256x1xf32>
    %966 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %967 = stablehlo.add %966, %44 : tensor<1x256x1xf32>
    %968 = stablehlo.rsqrt %967 : tensor<1x256x1xf32>
    %969 = stablehlo.broadcast_in_dim %947, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %970 = stablehlo.broadcast_in_dim %965, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %971 = stablehlo.subtract %969, %970 : tensor<1x256x512xf32>
    %972 = stablehlo.broadcast_in_dim %971, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %973 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %974 = stablehlo.multiply %972, %973 : tensor<1x256x512xf32>
    %975 = stablehlo.convert %arg47 : (tensor<512xbf16>) -> tensor<512xf32>
    %976 = stablehlo.broadcast_in_dim %974, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %977 = stablehlo.broadcast_in_dim %975, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %978 = stablehlo.multiply %976, %977 : tensor<1x256x512xf32>
    %979 = stablehlo.convert %arg48 : (tensor<512xbf16>) -> tensor<512xf32>
    %980 = stablehlo.broadcast_in_dim %978, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %981 = stablehlo.broadcast_in_dim %979, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %982 = stablehlo.add %980, %981 : tensor<1x256x512xf32>
    %983 = stablehlo.convert %982 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %984 = stablehlo.reshape %983 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %985 = stablehlo.convert %984 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %986 = stablehlo.dot_general %985, %arg121, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %987 = stablehlo.broadcast_in_dim %986, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %988 = stablehlo.multiply %987, %146 : tensor<256x256xf32>
    %989 = stablehlo.broadcast_in_dim %988, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %990 = stablehlo.broadcast_in_dim %arg122, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %991 = stablehlo.add %989, %990 : tensor<256x256xf32>
    %992 = stablehlo.convert %991 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %993 = stablehlo.reshape %992 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %994 = stablehlo.multiply %993, %cst_22 : tensor<1x256x256xbf16>
    %995 = stablehlo.multiply %993, %154 : tensor<1x256x256xbf16>
    %996 = stablehlo.convert %995 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %997 = stablehlo.clamp %cst_23, %996, %cst_24 : tensor<1x256x256xf32>
    %998 = stablehlo.multiply %997, %997 : tensor<1x256x256xf32>
    %999 = stablehlo.multiply %cst_25, %998 : tensor<1x256x256xf32>
    %1000 = stablehlo.add %999, %cst_26 : tensor<1x256x256xf32>
    %1001 = stablehlo.multiply %1000, %998 : tensor<1x256x256xf32>
    %1002 = stablehlo.add %1001, %cst_27 : tensor<1x256x256xf32>
    %1003 = stablehlo.multiply %1002, %998 : tensor<1x256x256xf32>
    %1004 = stablehlo.add %1003, %cst_28 : tensor<1x256x256xf32>
    %1005 = stablehlo.multiply %1004, %998 : tensor<1x256x256xf32>
    %1006 = stablehlo.add %1005, %cst_29 : tensor<1x256x256xf32>
    %1007 = stablehlo.multiply %1006, %998 : tensor<1x256x256xf32>
    %1008 = stablehlo.add %1007, %cst_30 : tensor<1x256x256xf32>
    %1009 = stablehlo.multiply %1008, %998 : tensor<1x256x256xf32>
    %1010 = stablehlo.add %1009, %cst_31 : tensor<1x256x256xf32>
    %1011 = stablehlo.multiply %cst_32, %998 : tensor<1x256x256xf32>
    %1012 = stablehlo.add %1011, %cst_33 : tensor<1x256x256xf32>
    %1013 = stablehlo.multiply %1012, %998 : tensor<1x256x256xf32>
    %1014 = stablehlo.add %1013, %cst_34 : tensor<1x256x256xf32>
    %1015 = stablehlo.multiply %1014, %998 : tensor<1x256x256xf32>
    %1016 = stablehlo.add %1015, %cst_35 : tensor<1x256x256xf32>
    %1017 = stablehlo.multiply %1016, %998 : tensor<1x256x256xf32>
    %1018 = stablehlo.add %1017, %cst_36 : tensor<1x256x256xf32>
    %1019 = stablehlo.multiply %997, %1010 : tensor<1x256x256xf32>
    %1020 = stablehlo.divide %1019, %1018 : tensor<1x256x256xf32>
    %1021 = stablehlo.clamp %cst_37, %1020, %cst_38 : tensor<1x256x256xf32>
    %1022 = stablehlo.convert %1021 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1023 = stablehlo.add %1022, %cst_20 : tensor<1x256x256xbf16>
    %1024 = stablehlo.multiply %1023, %994 : tensor<1x256x256xbf16>
    %1025 = stablehlo.reshape %1024 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1026 = stablehlo.convert %1025 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1027 = stablehlo.dot_general %1026, %arg123, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %1028 = stablehlo.broadcast_in_dim %1027, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1029 = stablehlo.multiply %1028, %9 : tensor<256x512xf32>
    %1030 = stablehlo.broadcast_in_dim %1029, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1031 = stablehlo.broadcast_in_dim %arg124, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %1032 = stablehlo.add %1030, %1031 : tensor<256x512xf32>
    %1033 = stablehlo.convert %1032 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1034 = stablehlo.reshape %1033 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %1035 = stablehlo.add %1034, %946 : tensor<1x256x512xbf16>
    %1036 = stablehlo.convert %1035 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1037 = stablehlo.convert %1036 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1038 = stablehlo.reduce(%1037 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1039 = stablehlo.reshape %1038 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1041 = stablehlo.divide %1040, %23 : tensor<1x256x1xf64>
    %1042 = stablehlo.broadcast_in_dim %1037, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1043 = stablehlo.broadcast_in_dim %1041, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1044 = stablehlo.subtract %1042, %1043 : tensor<1x256x512xf64>
    %1045 = stablehlo.multiply %1044, %1044 : tensor<1x256x512xf64>
    %1046 = stablehlo.reduce(%1045 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1047 = stablehlo.reshape %1046 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1048 = stablehlo.broadcast_in_dim %1047, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1049 = stablehlo.divide %1048, %23 : tensor<1x256x1xf64>
    %1050 = stablehlo.convert %1049 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1051 = stablehlo.reduce(%1036 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1052 = stablehlo.reshape %1051 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1053 = stablehlo.broadcast_in_dim %1052, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1054 = stablehlo.divide %1053, %39 : tensor<1x256x1xf32>
    %1055 = stablehlo.broadcast_in_dim %1050, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1056 = stablehlo.add %1055, %44 : tensor<1x256x1xf32>
    %1057 = stablehlo.rsqrt %1056 : tensor<1x256x1xf32>
    %1058 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1059 = stablehlo.broadcast_in_dim %1054, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1060 = stablehlo.subtract %1058, %1059 : tensor<1x256x512xf32>
    %1061 = stablehlo.broadcast_in_dim %1060, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1062 = stablehlo.broadcast_in_dim %1057, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1063 = stablehlo.multiply %1061, %1062 : tensor<1x256x512xf32>
    %1064 = stablehlo.convert %arg49 : (tensor<512xbf16>) -> tensor<512xf32>
    %1065 = stablehlo.broadcast_in_dim %1063, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1066 = stablehlo.broadcast_in_dim %1064, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1067 = stablehlo.multiply %1065, %1066 : tensor<1x256x512xf32>
    %1068 = stablehlo.convert %arg50 : (tensor<512xbf16>) -> tensor<512xf32>
    %1069 = stablehlo.broadcast_in_dim %1067, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1070 = stablehlo.broadcast_in_dim %1068, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1071 = stablehlo.add %1069, %1070 : tensor<1x256x512xf32>
    %1072 = stablehlo.convert %1071 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1073 = stablehlo.convolution(%1072, %arg51) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %1074 = stablehlo.reshape %arg52 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %1075 = stablehlo.broadcast_in_dim %1073, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %1076 = stablehlo.broadcast_in_dim %1074, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %1077 = stablehlo.add %1075, %1076 : tensor<1x1024x512xbf16>
    %1078 = stablehlo.multiply %1077, %cst_3 : tensor<1x1024x512xbf16>
    %1079 = stablehlo.multiply %1077, %68 : tensor<1x1024x512xbf16>
    %1080 = stablehlo.convert %1079 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %1081 = stablehlo.clamp %cst_4, %1080, %cst_5 : tensor<1x1024x512xf32>
    %1082 = stablehlo.multiply %1081, %1081 : tensor<1x1024x512xf32>
    %1083 = stablehlo.multiply %cst_6, %1082 : tensor<1x1024x512xf32>
    %1084 = stablehlo.add %1083, %cst_7 : tensor<1x1024x512xf32>
    %1085 = stablehlo.multiply %1084, %1082 : tensor<1x1024x512xf32>
    %1086 = stablehlo.add %1085, %cst_8 : tensor<1x1024x512xf32>
    %1087 = stablehlo.multiply %1086, %1082 : tensor<1x1024x512xf32>
    %1088 = stablehlo.add %1087, %cst_9 : tensor<1x1024x512xf32>
    %1089 = stablehlo.multiply %1088, %1082 : tensor<1x1024x512xf32>
    %1090 = stablehlo.add %1089, %cst_10 : tensor<1x1024x512xf32>
    %1091 = stablehlo.multiply %1090, %1082 : tensor<1x1024x512xf32>
    %1092 = stablehlo.add %1091, %cst_11 : tensor<1x1024x512xf32>
    %1093 = stablehlo.multiply %1092, %1082 : tensor<1x1024x512xf32>
    %1094 = stablehlo.add %1093, %cst_12 : tensor<1x1024x512xf32>
    %1095 = stablehlo.multiply %cst_13, %1082 : tensor<1x1024x512xf32>
    %1096 = stablehlo.add %1095, %cst_14 : tensor<1x1024x512xf32>
    %1097 = stablehlo.multiply %1096, %1082 : tensor<1x1024x512xf32>
    %1098 = stablehlo.add %1097, %cst_15 : tensor<1x1024x512xf32>
    %1099 = stablehlo.multiply %1098, %1082 : tensor<1x1024x512xf32>
    %1100 = stablehlo.add %1099, %cst_16 : tensor<1x1024x512xf32>
    %1101 = stablehlo.multiply %1100, %1082 : tensor<1x1024x512xf32>
    %1102 = stablehlo.add %1101, %cst_17 : tensor<1x1024x512xf32>
    %1103 = stablehlo.multiply %1081, %1094 : tensor<1x1024x512xf32>
    %1104 = stablehlo.divide %1103, %1102 : tensor<1x1024x512xf32>
    %1105 = stablehlo.clamp %cst_18, %1104, %cst_19 : tensor<1x1024x512xf32>
    %1106 = stablehlo.convert %1105 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %1107 = stablehlo.add %1106, %cst_1 : tensor<1x1024x512xbf16>
    %1108 = stablehlo.multiply %1107, %1078 : tensor<1x1024x512xbf16>
    %1109 = stablehlo.convolution(%1108, %arg53) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %1110 = stablehlo.reshape %arg54 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %1111 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %1112 = stablehlo.broadcast_in_dim %1110, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %1113 = stablehlo.add %1111, %1112 : tensor<1x256x512xbf16>
    %1114 = stablehlo.add %1113, %1035 : tensor<1x256x512xbf16>
    %1115 = stablehlo.convert %1114 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1116 = stablehlo.convert %1115 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1117 = stablehlo.reduce(%1116 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1118 = stablehlo.reshape %1117 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1119 = stablehlo.broadcast_in_dim %1118, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1120 = stablehlo.divide %1119, %23 : tensor<1x256x1xf64>
    %1121 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1122 = stablehlo.broadcast_in_dim %1120, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1123 = stablehlo.subtract %1121, %1122 : tensor<1x256x512xf64>
    %1124 = stablehlo.multiply %1123, %1123 : tensor<1x256x512xf64>
    %1125 = stablehlo.reduce(%1124 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1126 = stablehlo.reshape %1125 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1127 = stablehlo.broadcast_in_dim %1126, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1128 = stablehlo.divide %1127, %23 : tensor<1x256x1xf64>
    %1129 = stablehlo.convert %1128 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1130 = stablehlo.reduce(%1115 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1131 = stablehlo.reshape %1130 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1132 = stablehlo.broadcast_in_dim %1131, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1133 = stablehlo.divide %1132, %39 : tensor<1x256x1xf32>
    %1134 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1135 = stablehlo.add %1134, %44 : tensor<1x256x1xf32>
    %1136 = stablehlo.rsqrt %1135 : tensor<1x256x1xf32>
    %1137 = stablehlo.broadcast_in_dim %1115, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1138 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1139 = stablehlo.subtract %1137, %1138 : tensor<1x256x512xf32>
    %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1141 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1142 = stablehlo.multiply %1140, %1141 : tensor<1x256x512xf32>
    %1143 = stablehlo.convert %arg55 : (tensor<512xbf16>) -> tensor<512xf32>
    %1144 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1145 = stablehlo.broadcast_in_dim %1143, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1146 = stablehlo.multiply %1144, %1145 : tensor<1x256x512xf32>
    %1147 = stablehlo.convert %arg56 : (tensor<512xbf16>) -> tensor<512xf32>
    %1148 = stablehlo.broadcast_in_dim %1146, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1149 = stablehlo.broadcast_in_dim %1147, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1150 = stablehlo.add %1148, %1149 : tensor<1x256x512xf32>
    %1151 = stablehlo.convert %1150 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1152 = stablehlo.reshape %1151 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %1153 = stablehlo.convert %1152 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1154 = stablehlo.dot_general %1153, %arg125, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1156 = stablehlo.multiply %1155, %146 : tensor<256x256xf32>
    %1157 = stablehlo.broadcast_in_dim %1156, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1158 = stablehlo.broadcast_in_dim %arg126, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1159 = stablehlo.add %1157, %1158 : tensor<256x256xf32>
    %1160 = stablehlo.convert %1159 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1161 = stablehlo.reshape %1160 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1162 = stablehlo.multiply %1161, %cst_22 : tensor<1x256x256xbf16>
    %1163 = stablehlo.multiply %1161, %154 : tensor<1x256x256xbf16>
    %1164 = stablehlo.convert %1163 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1165 = stablehlo.clamp %cst_23, %1164, %cst_24 : tensor<1x256x256xf32>
    %1166 = stablehlo.multiply %1165, %1165 : tensor<1x256x256xf32>
    %1167 = stablehlo.multiply %cst_25, %1166 : tensor<1x256x256xf32>
    %1168 = stablehlo.add %1167, %cst_26 : tensor<1x256x256xf32>
    %1169 = stablehlo.multiply %1168, %1166 : tensor<1x256x256xf32>
    %1170 = stablehlo.add %1169, %cst_27 : tensor<1x256x256xf32>
    %1171 = stablehlo.multiply %1170, %1166 : tensor<1x256x256xf32>
    %1172 = stablehlo.add %1171, %cst_28 : tensor<1x256x256xf32>
    %1173 = stablehlo.multiply %1172, %1166 : tensor<1x256x256xf32>
    %1174 = stablehlo.add %1173, %cst_29 : tensor<1x256x256xf32>
    %1175 = stablehlo.multiply %1174, %1166 : tensor<1x256x256xf32>
    %1176 = stablehlo.add %1175, %cst_30 : tensor<1x256x256xf32>
    %1177 = stablehlo.multiply %1176, %1166 : tensor<1x256x256xf32>
    %1178 = stablehlo.add %1177, %cst_31 : tensor<1x256x256xf32>
    %1179 = stablehlo.multiply %cst_32, %1166 : tensor<1x256x256xf32>
    %1180 = stablehlo.add %1179, %cst_33 : tensor<1x256x256xf32>
    %1181 = stablehlo.multiply %1180, %1166 : tensor<1x256x256xf32>
    %1182 = stablehlo.add %1181, %cst_34 : tensor<1x256x256xf32>
    %1183 = stablehlo.multiply %1182, %1166 : tensor<1x256x256xf32>
    %1184 = stablehlo.add %1183, %cst_35 : tensor<1x256x256xf32>
    %1185 = stablehlo.multiply %1184, %1166 : tensor<1x256x256xf32>
    %1186 = stablehlo.add %1185, %cst_36 : tensor<1x256x256xf32>
    %1187 = stablehlo.multiply %1165, %1178 : tensor<1x256x256xf32>
    %1188 = stablehlo.divide %1187, %1186 : tensor<1x256x256xf32>
    %1189 = stablehlo.clamp %cst_37, %1188, %cst_38 : tensor<1x256x256xf32>
    %1190 = stablehlo.convert %1189 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1191 = stablehlo.add %1190, %cst_20 : tensor<1x256x256xbf16>
    %1192 = stablehlo.multiply %1191, %1162 : tensor<1x256x256xbf16>
    %1193 = stablehlo.reshape %1192 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1194 = stablehlo.convert %1193 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1195 = stablehlo.dot_general %1194, %arg127, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %1196 = stablehlo.broadcast_in_dim %1195, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1197 = stablehlo.multiply %1196, %9 : tensor<256x512xf32>
    %1198 = stablehlo.broadcast_in_dim %1197, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1199 = stablehlo.broadcast_in_dim %arg128, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %1200 = stablehlo.add %1198, %1199 : tensor<256x512xf32>
    %1201 = stablehlo.convert %1200 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1202 = stablehlo.reshape %1201 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %1203 = stablehlo.add %1202, %1114 : tensor<1x256x512xbf16>
    %1204 = stablehlo.convert %1203 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1205 = stablehlo.convert %1204 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1206 = stablehlo.reduce(%1205 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1207 = stablehlo.reshape %1206 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1208 = stablehlo.broadcast_in_dim %1207, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1209 = stablehlo.divide %1208, %23 : tensor<1x256x1xf64>
    %1210 = stablehlo.broadcast_in_dim %1205, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1211 = stablehlo.broadcast_in_dim %1209, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1212 = stablehlo.subtract %1210, %1211 : tensor<1x256x512xf64>
    %1213 = stablehlo.multiply %1212, %1212 : tensor<1x256x512xf64>
    %1214 = stablehlo.reduce(%1213 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1215 = stablehlo.reshape %1214 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1216 = stablehlo.broadcast_in_dim %1215, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1217 = stablehlo.divide %1216, %23 : tensor<1x256x1xf64>
    %1218 = stablehlo.convert %1217 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1219 = stablehlo.reduce(%1204 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1220 = stablehlo.reshape %1219 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1221 = stablehlo.broadcast_in_dim %1220, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1222 = stablehlo.divide %1221, %39 : tensor<1x256x1xf32>
    %1223 = stablehlo.broadcast_in_dim %1218, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1224 = stablehlo.add %1223, %44 : tensor<1x256x1xf32>
    %1225 = stablehlo.rsqrt %1224 : tensor<1x256x1xf32>
    %1226 = stablehlo.broadcast_in_dim %1204, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1227 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1228 = stablehlo.subtract %1226, %1227 : tensor<1x256x512xf32>
    %1229 = stablehlo.broadcast_in_dim %1228, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1230 = stablehlo.broadcast_in_dim %1225, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1231 = stablehlo.multiply %1229, %1230 : tensor<1x256x512xf32>
    %1232 = stablehlo.convert %arg57 : (tensor<512xbf16>) -> tensor<512xf32>
    %1233 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1234 = stablehlo.broadcast_in_dim %1232, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1235 = stablehlo.multiply %1233, %1234 : tensor<1x256x512xf32>
    %1236 = stablehlo.convert %arg58 : (tensor<512xbf16>) -> tensor<512xf32>
    %1237 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1238 = stablehlo.broadcast_in_dim %1236, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1239 = stablehlo.add %1237, %1238 : tensor<1x256x512xf32>
    %1240 = stablehlo.convert %1239 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1241 = stablehlo.convolution(%1240, %arg59) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %1242 = stablehlo.reshape %arg60 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %1243 = stablehlo.broadcast_in_dim %1241, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %1244 = stablehlo.broadcast_in_dim %1242, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %1245 = stablehlo.add %1243, %1244 : tensor<1x1024x512xbf16>
    %1246 = stablehlo.multiply %1245, %cst_3 : tensor<1x1024x512xbf16>
    %1247 = stablehlo.multiply %1245, %68 : tensor<1x1024x512xbf16>
    %1248 = stablehlo.convert %1247 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %1249 = stablehlo.clamp %cst_4, %1248, %cst_5 : tensor<1x1024x512xf32>
    %1250 = stablehlo.multiply %1249, %1249 : tensor<1x1024x512xf32>
    %1251 = stablehlo.multiply %cst_6, %1250 : tensor<1x1024x512xf32>
    %1252 = stablehlo.add %1251, %cst_7 : tensor<1x1024x512xf32>
    %1253 = stablehlo.multiply %1252, %1250 : tensor<1x1024x512xf32>
    %1254 = stablehlo.add %1253, %cst_8 : tensor<1x1024x512xf32>
    %1255 = stablehlo.multiply %1254, %1250 : tensor<1x1024x512xf32>
    %1256 = stablehlo.add %1255, %cst_9 : tensor<1x1024x512xf32>
    %1257 = stablehlo.multiply %1256, %1250 : tensor<1x1024x512xf32>
    %1258 = stablehlo.add %1257, %cst_10 : tensor<1x1024x512xf32>
    %1259 = stablehlo.multiply %1258, %1250 : tensor<1x1024x512xf32>
    %1260 = stablehlo.add %1259, %cst_11 : tensor<1x1024x512xf32>
    %1261 = stablehlo.multiply %1260, %1250 : tensor<1x1024x512xf32>
    %1262 = stablehlo.add %1261, %cst_12 : tensor<1x1024x512xf32>
    %1263 = stablehlo.multiply %cst_13, %1250 : tensor<1x1024x512xf32>
    %1264 = stablehlo.add %1263, %cst_14 : tensor<1x1024x512xf32>
    %1265 = stablehlo.multiply %1264, %1250 : tensor<1x1024x512xf32>
    %1266 = stablehlo.add %1265, %cst_15 : tensor<1x1024x512xf32>
    %1267 = stablehlo.multiply %1266, %1250 : tensor<1x1024x512xf32>
    %1268 = stablehlo.add %1267, %cst_16 : tensor<1x1024x512xf32>
    %1269 = stablehlo.multiply %1268, %1250 : tensor<1x1024x512xf32>
    %1270 = stablehlo.add %1269, %cst_17 : tensor<1x1024x512xf32>
    %1271 = stablehlo.multiply %1249, %1262 : tensor<1x1024x512xf32>
    %1272 = stablehlo.divide %1271, %1270 : tensor<1x1024x512xf32>
    %1273 = stablehlo.clamp %cst_18, %1272, %cst_19 : tensor<1x1024x512xf32>
    %1274 = stablehlo.convert %1273 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %1275 = stablehlo.add %1274, %cst_1 : tensor<1x1024x512xbf16>
    %1276 = stablehlo.multiply %1275, %1246 : tensor<1x1024x512xbf16>
    %1277 = stablehlo.convolution(%1276, %arg61) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %1278 = stablehlo.reshape %arg62 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %1279 = stablehlo.broadcast_in_dim %1277, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %1280 = stablehlo.broadcast_in_dim %1278, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %1281 = stablehlo.add %1279, %1280 : tensor<1x256x512xbf16>
    %1282 = stablehlo.add %1281, %1203 : tensor<1x256x512xbf16>
    %1283 = stablehlo.convert %1282 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1284 = stablehlo.convert %1283 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1285 = stablehlo.reduce(%1284 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1286 = stablehlo.reshape %1285 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1288 = stablehlo.divide %1287, %23 : tensor<1x256x1xf64>
    %1289 = stablehlo.broadcast_in_dim %1284, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1290 = stablehlo.broadcast_in_dim %1288, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1291 = stablehlo.subtract %1289, %1290 : tensor<1x256x512xf64>
    %1292 = stablehlo.multiply %1291, %1291 : tensor<1x256x512xf64>
    %1293 = stablehlo.reduce(%1292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1294 = stablehlo.reshape %1293 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1295 = stablehlo.broadcast_in_dim %1294, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1296 = stablehlo.divide %1295, %23 : tensor<1x256x1xf64>
    %1297 = stablehlo.convert %1296 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1298 = stablehlo.reduce(%1283 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1299 = stablehlo.reshape %1298 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1301 = stablehlo.divide %1300, %39 : tensor<1x256x1xf32>
    %1302 = stablehlo.broadcast_in_dim %1297, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1303 = stablehlo.add %1302, %44 : tensor<1x256x1xf32>
    %1304 = stablehlo.rsqrt %1303 : tensor<1x256x1xf32>
    %1305 = stablehlo.broadcast_in_dim %1283, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1306 = stablehlo.broadcast_in_dim %1301, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1307 = stablehlo.subtract %1305, %1306 : tensor<1x256x512xf32>
    %1308 = stablehlo.broadcast_in_dim %1307, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1309 = stablehlo.broadcast_in_dim %1304, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1310 = stablehlo.multiply %1308, %1309 : tensor<1x256x512xf32>
    %1311 = stablehlo.convert %arg63 : (tensor<512xbf16>) -> tensor<512xf32>
    %1312 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1313 = stablehlo.broadcast_in_dim %1311, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1314 = stablehlo.multiply %1312, %1313 : tensor<1x256x512xf32>
    %1315 = stablehlo.convert %arg64 : (tensor<512xbf16>) -> tensor<512xf32>
    %1316 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1317 = stablehlo.broadcast_in_dim %1315, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1318 = stablehlo.add %1316, %1317 : tensor<1x256x512xf32>
    %1319 = stablehlo.convert %1318 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1320 = stablehlo.reshape %1319 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %1321 = stablehlo.convert %1320 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1322 = stablehlo.dot_general %1321, %arg129, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1324 = stablehlo.multiply %1323, %146 : tensor<256x256xf32>
    %1325 = stablehlo.broadcast_in_dim %1324, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1326 = stablehlo.broadcast_in_dim %arg130, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1327 = stablehlo.add %1325, %1326 : tensor<256x256xf32>
    %1328 = stablehlo.convert %1327 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1329 = stablehlo.reshape %1328 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1330 = stablehlo.multiply %1329, %cst_22 : tensor<1x256x256xbf16>
    %1331 = stablehlo.multiply %1329, %154 : tensor<1x256x256xbf16>
    %1332 = stablehlo.convert %1331 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1333 = stablehlo.clamp %cst_23, %1332, %cst_24 : tensor<1x256x256xf32>
    %1334 = stablehlo.multiply %1333, %1333 : tensor<1x256x256xf32>
    %1335 = stablehlo.multiply %cst_25, %1334 : tensor<1x256x256xf32>
    %1336 = stablehlo.add %1335, %cst_26 : tensor<1x256x256xf32>
    %1337 = stablehlo.multiply %1336, %1334 : tensor<1x256x256xf32>
    %1338 = stablehlo.add %1337, %cst_27 : tensor<1x256x256xf32>
    %1339 = stablehlo.multiply %1338, %1334 : tensor<1x256x256xf32>
    %1340 = stablehlo.add %1339, %cst_28 : tensor<1x256x256xf32>
    %1341 = stablehlo.multiply %1340, %1334 : tensor<1x256x256xf32>
    %1342 = stablehlo.add %1341, %cst_29 : tensor<1x256x256xf32>
    %1343 = stablehlo.multiply %1342, %1334 : tensor<1x256x256xf32>
    %1344 = stablehlo.add %1343, %cst_30 : tensor<1x256x256xf32>
    %1345 = stablehlo.multiply %1344, %1334 : tensor<1x256x256xf32>
    %1346 = stablehlo.add %1345, %cst_31 : tensor<1x256x256xf32>
    %1347 = stablehlo.multiply %cst_32, %1334 : tensor<1x256x256xf32>
    %1348 = stablehlo.add %1347, %cst_33 : tensor<1x256x256xf32>
    %1349 = stablehlo.multiply %1348, %1334 : tensor<1x256x256xf32>
    %1350 = stablehlo.add %1349, %cst_34 : tensor<1x256x256xf32>
    %1351 = stablehlo.multiply %1350, %1334 : tensor<1x256x256xf32>
    %1352 = stablehlo.add %1351, %cst_35 : tensor<1x256x256xf32>
    %1353 = stablehlo.multiply %1352, %1334 : tensor<1x256x256xf32>
    %1354 = stablehlo.add %1353, %cst_36 : tensor<1x256x256xf32>
    %1355 = stablehlo.multiply %1333, %1346 : tensor<1x256x256xf32>
    %1356 = stablehlo.divide %1355, %1354 : tensor<1x256x256xf32>
    %1357 = stablehlo.clamp %cst_37, %1356, %cst_38 : tensor<1x256x256xf32>
    %1358 = stablehlo.convert %1357 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1359 = stablehlo.add %1358, %cst_20 : tensor<1x256x256xbf16>
    %1360 = stablehlo.multiply %1359, %1330 : tensor<1x256x256xbf16>
    %1361 = stablehlo.reshape %1360 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1362 = stablehlo.convert %1361 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1363 = stablehlo.dot_general %1362, %arg131, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %1364 = stablehlo.broadcast_in_dim %1363, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1365 = stablehlo.multiply %1364, %9 : tensor<256x512xf32>
    %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1367 = stablehlo.broadcast_in_dim %arg132, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %1368 = stablehlo.add %1366, %1367 : tensor<256x512xf32>
    %1369 = stablehlo.convert %1368 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1370 = stablehlo.reshape %1369 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %1371 = stablehlo.add %1370, %1282 : tensor<1x256x512xbf16>
    %1372 = stablehlo.convert %1371 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1373 = stablehlo.convert %1372 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1374 = stablehlo.reduce(%1373 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1375 = stablehlo.reshape %1374 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1376 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1377 = stablehlo.divide %1376, %23 : tensor<1x256x1xf64>
    %1378 = stablehlo.broadcast_in_dim %1373, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1379 = stablehlo.broadcast_in_dim %1377, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1380 = stablehlo.subtract %1378, %1379 : tensor<1x256x512xf64>
    %1381 = stablehlo.multiply %1380, %1380 : tensor<1x256x512xf64>
    %1382 = stablehlo.reduce(%1381 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1383 = stablehlo.reshape %1382 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1384 = stablehlo.broadcast_in_dim %1383, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1385 = stablehlo.divide %1384, %23 : tensor<1x256x1xf64>
    %1386 = stablehlo.convert %1385 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1387 = stablehlo.reduce(%1372 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1388 = stablehlo.reshape %1387 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1389 = stablehlo.broadcast_in_dim %1388, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1390 = stablehlo.divide %1389, %39 : tensor<1x256x1xf32>
    %1391 = stablehlo.broadcast_in_dim %1386, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1392 = stablehlo.add %1391, %44 : tensor<1x256x1xf32>
    %1393 = stablehlo.rsqrt %1392 : tensor<1x256x1xf32>
    %1394 = stablehlo.broadcast_in_dim %1372, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1395 = stablehlo.broadcast_in_dim %1390, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1396 = stablehlo.subtract %1394, %1395 : tensor<1x256x512xf32>
    %1397 = stablehlo.broadcast_in_dim %1396, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1398 = stablehlo.broadcast_in_dim %1393, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1399 = stablehlo.multiply %1397, %1398 : tensor<1x256x512xf32>
    %1400 = stablehlo.convert %arg65 : (tensor<512xbf16>) -> tensor<512xf32>
    %1401 = stablehlo.broadcast_in_dim %1399, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1402 = stablehlo.broadcast_in_dim %1400, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1403 = stablehlo.multiply %1401, %1402 : tensor<1x256x512xf32>
    %1404 = stablehlo.convert %arg66 : (tensor<512xbf16>) -> tensor<512xf32>
    %1405 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1406 = stablehlo.broadcast_in_dim %1404, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1407 = stablehlo.add %1405, %1406 : tensor<1x256x512xf32>
    %1408 = stablehlo.convert %1407 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1409 = stablehlo.convolution(%1408, %arg67) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %1410 = stablehlo.reshape %arg68 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %1411 = stablehlo.broadcast_in_dim %1409, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %1412 = stablehlo.broadcast_in_dim %1410, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %1413 = stablehlo.add %1411, %1412 : tensor<1x1024x512xbf16>
    %1414 = stablehlo.multiply %1413, %cst_3 : tensor<1x1024x512xbf16>
    %1415 = stablehlo.multiply %1413, %68 : tensor<1x1024x512xbf16>
    %1416 = stablehlo.convert %1415 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %1417 = stablehlo.clamp %cst_4, %1416, %cst_5 : tensor<1x1024x512xf32>
    %1418 = stablehlo.multiply %1417, %1417 : tensor<1x1024x512xf32>
    %1419 = stablehlo.multiply %cst_6, %1418 : tensor<1x1024x512xf32>
    %1420 = stablehlo.add %1419, %cst_7 : tensor<1x1024x512xf32>
    %1421 = stablehlo.multiply %1420, %1418 : tensor<1x1024x512xf32>
    %1422 = stablehlo.add %1421, %cst_8 : tensor<1x1024x512xf32>
    %1423 = stablehlo.multiply %1422, %1418 : tensor<1x1024x512xf32>
    %1424 = stablehlo.add %1423, %cst_9 : tensor<1x1024x512xf32>
    %1425 = stablehlo.multiply %1424, %1418 : tensor<1x1024x512xf32>
    %1426 = stablehlo.add %1425, %cst_10 : tensor<1x1024x512xf32>
    %1427 = stablehlo.multiply %1426, %1418 : tensor<1x1024x512xf32>
    %1428 = stablehlo.add %1427, %cst_11 : tensor<1x1024x512xf32>
    %1429 = stablehlo.multiply %1428, %1418 : tensor<1x1024x512xf32>
    %1430 = stablehlo.add %1429, %cst_12 : tensor<1x1024x512xf32>
    %1431 = stablehlo.multiply %cst_13, %1418 : tensor<1x1024x512xf32>
    %1432 = stablehlo.add %1431, %cst_14 : tensor<1x1024x512xf32>
    %1433 = stablehlo.multiply %1432, %1418 : tensor<1x1024x512xf32>
    %1434 = stablehlo.add %1433, %cst_15 : tensor<1x1024x512xf32>
    %1435 = stablehlo.multiply %1434, %1418 : tensor<1x1024x512xf32>
    %1436 = stablehlo.add %1435, %cst_16 : tensor<1x1024x512xf32>
    %1437 = stablehlo.multiply %1436, %1418 : tensor<1x1024x512xf32>
    %1438 = stablehlo.add %1437, %cst_17 : tensor<1x1024x512xf32>
    %1439 = stablehlo.multiply %1417, %1430 : tensor<1x1024x512xf32>
    %1440 = stablehlo.divide %1439, %1438 : tensor<1x1024x512xf32>
    %1441 = stablehlo.clamp %cst_18, %1440, %cst_19 : tensor<1x1024x512xf32>
    %1442 = stablehlo.convert %1441 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %1443 = stablehlo.add %1442, %cst_1 : tensor<1x1024x512xbf16>
    %1444 = stablehlo.multiply %1443, %1414 : tensor<1x1024x512xbf16>
    %1445 = stablehlo.convolution(%1444, %arg69) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %1446 = stablehlo.reshape %arg70 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %1447 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %1448 = stablehlo.broadcast_in_dim %1446, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %1449 = stablehlo.add %1447, %1448 : tensor<1x256x512xbf16>
    %1450 = stablehlo.add %1449, %1371 : tensor<1x256x512xbf16>
    %1451 = stablehlo.convert %1450 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1452 = stablehlo.convert %1451 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1453 = stablehlo.reduce(%1452 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1454 = stablehlo.reshape %1453 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1455 = stablehlo.broadcast_in_dim %1454, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1456 = stablehlo.divide %1455, %23 : tensor<1x256x1xf64>
    %1457 = stablehlo.broadcast_in_dim %1452, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1458 = stablehlo.broadcast_in_dim %1456, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1459 = stablehlo.subtract %1457, %1458 : tensor<1x256x512xf64>
    %1460 = stablehlo.multiply %1459, %1459 : tensor<1x256x512xf64>
    %1461 = stablehlo.reduce(%1460 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1462 = stablehlo.reshape %1461 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1464 = stablehlo.divide %1463, %23 : tensor<1x256x1xf64>
    %1465 = stablehlo.convert %1464 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1466 = stablehlo.reduce(%1451 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1467 = stablehlo.reshape %1466 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1468 = stablehlo.broadcast_in_dim %1467, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1469 = stablehlo.divide %1468, %39 : tensor<1x256x1xf32>
    %1470 = stablehlo.broadcast_in_dim %1465, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1471 = stablehlo.add %1470, %44 : tensor<1x256x1xf32>
    %1472 = stablehlo.rsqrt %1471 : tensor<1x256x1xf32>
    %1473 = stablehlo.broadcast_in_dim %1451, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1474 = stablehlo.broadcast_in_dim %1469, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1475 = stablehlo.subtract %1473, %1474 : tensor<1x256x512xf32>
    %1476 = stablehlo.broadcast_in_dim %1475, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1477 = stablehlo.broadcast_in_dim %1472, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1478 = stablehlo.multiply %1476, %1477 : tensor<1x256x512xf32>
    %1479 = stablehlo.convert %arg71 : (tensor<512xbf16>) -> tensor<512xf32>
    %1480 = stablehlo.broadcast_in_dim %1478, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1481 = stablehlo.broadcast_in_dim %1479, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1482 = stablehlo.multiply %1480, %1481 : tensor<1x256x512xf32>
    %1483 = stablehlo.convert %arg72 : (tensor<512xbf16>) -> tensor<512xf32>
    %1484 = stablehlo.broadcast_in_dim %1482, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1485 = stablehlo.broadcast_in_dim %1483, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1486 = stablehlo.add %1484, %1485 : tensor<1x256x512xf32>
    %1487 = stablehlo.convert %1486 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1488 = stablehlo.reshape %1487 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %1489 = stablehlo.convert %1488 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1490 = stablehlo.dot_general %1489, %arg133, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %1491 = stablehlo.broadcast_in_dim %1490, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1492 = stablehlo.multiply %1491, %146 : tensor<256x256xf32>
    %1493 = stablehlo.broadcast_in_dim %1492, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1494 = stablehlo.broadcast_in_dim %arg134, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1495 = stablehlo.add %1493, %1494 : tensor<256x256xf32>
    %1496 = stablehlo.convert %1495 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1497 = stablehlo.reshape %1496 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1498 = stablehlo.multiply %1497, %cst_22 : tensor<1x256x256xbf16>
    %1499 = stablehlo.multiply %1497, %154 : tensor<1x256x256xbf16>
    %1500 = stablehlo.convert %1499 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1501 = stablehlo.clamp %cst_23, %1500, %cst_24 : tensor<1x256x256xf32>
    %1502 = stablehlo.multiply %1501, %1501 : tensor<1x256x256xf32>
    %1503 = stablehlo.multiply %cst_25, %1502 : tensor<1x256x256xf32>
    %1504 = stablehlo.add %1503, %cst_26 : tensor<1x256x256xf32>
    %1505 = stablehlo.multiply %1504, %1502 : tensor<1x256x256xf32>
    %1506 = stablehlo.add %1505, %cst_27 : tensor<1x256x256xf32>
    %1507 = stablehlo.multiply %1506, %1502 : tensor<1x256x256xf32>
    %1508 = stablehlo.add %1507, %cst_28 : tensor<1x256x256xf32>
    %1509 = stablehlo.multiply %1508, %1502 : tensor<1x256x256xf32>
    %1510 = stablehlo.add %1509, %cst_29 : tensor<1x256x256xf32>
    %1511 = stablehlo.multiply %1510, %1502 : tensor<1x256x256xf32>
    %1512 = stablehlo.add %1511, %cst_30 : tensor<1x256x256xf32>
    %1513 = stablehlo.multiply %1512, %1502 : tensor<1x256x256xf32>
    %1514 = stablehlo.add %1513, %cst_31 : tensor<1x256x256xf32>
    %1515 = stablehlo.multiply %cst_32, %1502 : tensor<1x256x256xf32>
    %1516 = stablehlo.add %1515, %cst_33 : tensor<1x256x256xf32>
    %1517 = stablehlo.multiply %1516, %1502 : tensor<1x256x256xf32>
    %1518 = stablehlo.add %1517, %cst_34 : tensor<1x256x256xf32>
    %1519 = stablehlo.multiply %1518, %1502 : tensor<1x256x256xf32>
    %1520 = stablehlo.add %1519, %cst_35 : tensor<1x256x256xf32>
    %1521 = stablehlo.multiply %1520, %1502 : tensor<1x256x256xf32>
    %1522 = stablehlo.add %1521, %cst_36 : tensor<1x256x256xf32>
    %1523 = stablehlo.multiply %1501, %1514 : tensor<1x256x256xf32>
    %1524 = stablehlo.divide %1523, %1522 : tensor<1x256x256xf32>
    %1525 = stablehlo.clamp %cst_37, %1524, %cst_38 : tensor<1x256x256xf32>
    %1526 = stablehlo.convert %1525 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1527 = stablehlo.add %1526, %cst_20 : tensor<1x256x256xbf16>
    %1528 = stablehlo.multiply %1527, %1498 : tensor<1x256x256xbf16>
    %1529 = stablehlo.reshape %1528 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1530 = stablehlo.convert %1529 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1531 = stablehlo.dot_general %1530, %arg135, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %1532 = stablehlo.broadcast_in_dim %1531, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1533 = stablehlo.multiply %1532, %9 : tensor<256x512xf32>
    %1534 = stablehlo.broadcast_in_dim %1533, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1535 = stablehlo.broadcast_in_dim %arg136, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %1536 = stablehlo.add %1534, %1535 : tensor<256x512xf32>
    %1537 = stablehlo.convert %1536 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1538 = stablehlo.reshape %1537 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %1539 = stablehlo.add %1538, %1450 : tensor<1x256x512xbf16>
    %1540 = stablehlo.convert %1539 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1541 = stablehlo.convert %1540 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1542 = stablehlo.reduce(%1541 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1543 = stablehlo.reshape %1542 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1544 = stablehlo.broadcast_in_dim %1543, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1545 = stablehlo.divide %1544, %23 : tensor<1x256x1xf64>
    %1546 = stablehlo.broadcast_in_dim %1541, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1547 = stablehlo.broadcast_in_dim %1545, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1548 = stablehlo.subtract %1546, %1547 : tensor<1x256x512xf64>
    %1549 = stablehlo.multiply %1548, %1548 : tensor<1x256x512xf64>
    %1550 = stablehlo.reduce(%1549 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1551 = stablehlo.reshape %1550 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1552 = stablehlo.broadcast_in_dim %1551, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1553 = stablehlo.divide %1552, %23 : tensor<1x256x1xf64>
    %1554 = stablehlo.convert %1553 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1555 = stablehlo.reduce(%1540 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1556 = stablehlo.reshape %1555 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1557 = stablehlo.broadcast_in_dim %1556, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1558 = stablehlo.divide %1557, %39 : tensor<1x256x1xf32>
    %1559 = stablehlo.broadcast_in_dim %1554, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1560 = stablehlo.add %1559, %44 : tensor<1x256x1xf32>
    %1561 = stablehlo.rsqrt %1560 : tensor<1x256x1xf32>
    %1562 = stablehlo.broadcast_in_dim %1540, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1563 = stablehlo.broadcast_in_dim %1558, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1564 = stablehlo.subtract %1562, %1563 : tensor<1x256x512xf32>
    %1565 = stablehlo.broadcast_in_dim %1564, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1566 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1567 = stablehlo.multiply %1565, %1566 : tensor<1x256x512xf32>
    %1568 = stablehlo.convert %arg73 : (tensor<512xbf16>) -> tensor<512xf32>
    %1569 = stablehlo.broadcast_in_dim %1567, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1570 = stablehlo.broadcast_in_dim %1568, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1571 = stablehlo.multiply %1569, %1570 : tensor<1x256x512xf32>
    %1572 = stablehlo.convert %arg74 : (tensor<512xbf16>) -> tensor<512xf32>
    %1573 = stablehlo.broadcast_in_dim %1571, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1574 = stablehlo.broadcast_in_dim %1572, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1575 = stablehlo.add %1573, %1574 : tensor<1x256x512xf32>
    %1576 = stablehlo.convert %1575 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1577 = stablehlo.convolution(%1576, %arg75) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %1578 = stablehlo.reshape %arg76 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %1579 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %1580 = stablehlo.broadcast_in_dim %1578, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %1581 = stablehlo.add %1579, %1580 : tensor<1x1024x512xbf16>
    %1582 = stablehlo.multiply %1581, %cst_3 : tensor<1x1024x512xbf16>
    %1583 = stablehlo.multiply %1581, %68 : tensor<1x1024x512xbf16>
    %1584 = stablehlo.convert %1583 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %1585 = stablehlo.clamp %cst_4, %1584, %cst_5 : tensor<1x1024x512xf32>
    %1586 = stablehlo.multiply %1585, %1585 : tensor<1x1024x512xf32>
    %1587 = stablehlo.multiply %cst_6, %1586 : tensor<1x1024x512xf32>
    %1588 = stablehlo.add %1587, %cst_7 : tensor<1x1024x512xf32>
    %1589 = stablehlo.multiply %1588, %1586 : tensor<1x1024x512xf32>
    %1590 = stablehlo.add %1589, %cst_8 : tensor<1x1024x512xf32>
    %1591 = stablehlo.multiply %1590, %1586 : tensor<1x1024x512xf32>
    %1592 = stablehlo.add %1591, %cst_9 : tensor<1x1024x512xf32>
    %1593 = stablehlo.multiply %1592, %1586 : tensor<1x1024x512xf32>
    %1594 = stablehlo.add %1593, %cst_10 : tensor<1x1024x512xf32>
    %1595 = stablehlo.multiply %1594, %1586 : tensor<1x1024x512xf32>
    %1596 = stablehlo.add %1595, %cst_11 : tensor<1x1024x512xf32>
    %1597 = stablehlo.multiply %1596, %1586 : tensor<1x1024x512xf32>
    %1598 = stablehlo.add %1597, %cst_12 : tensor<1x1024x512xf32>
    %1599 = stablehlo.multiply %cst_13, %1586 : tensor<1x1024x512xf32>
    %1600 = stablehlo.add %1599, %cst_14 : tensor<1x1024x512xf32>
    %1601 = stablehlo.multiply %1600, %1586 : tensor<1x1024x512xf32>
    %1602 = stablehlo.add %1601, %cst_15 : tensor<1x1024x512xf32>
    %1603 = stablehlo.multiply %1602, %1586 : tensor<1x1024x512xf32>
    %1604 = stablehlo.add %1603, %cst_16 : tensor<1x1024x512xf32>
    %1605 = stablehlo.multiply %1604, %1586 : tensor<1x1024x512xf32>
    %1606 = stablehlo.add %1605, %cst_17 : tensor<1x1024x512xf32>
    %1607 = stablehlo.multiply %1585, %1598 : tensor<1x1024x512xf32>
    %1608 = stablehlo.divide %1607, %1606 : tensor<1x1024x512xf32>
    %1609 = stablehlo.clamp %cst_18, %1608, %cst_19 : tensor<1x1024x512xf32>
    %1610 = stablehlo.convert %1609 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %1611 = stablehlo.add %1610, %cst_1 : tensor<1x1024x512xbf16>
    %1612 = stablehlo.multiply %1611, %1582 : tensor<1x1024x512xbf16>
    %1613 = stablehlo.convolution(%1612, %arg77) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %1614 = stablehlo.reshape %arg78 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %1615 = stablehlo.broadcast_in_dim %1613, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %1616 = stablehlo.broadcast_in_dim %1614, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %1617 = stablehlo.add %1615, %1616 : tensor<1x256x512xbf16>
    %1618 = stablehlo.add %1617, %1539 : tensor<1x256x512xbf16>
    %1619 = stablehlo.convert %1618 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1620 = stablehlo.convert %1619 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1621 = stablehlo.reduce(%1620 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1622 = stablehlo.reshape %1621 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1623 = stablehlo.broadcast_in_dim %1622, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1624 = stablehlo.divide %1623, %23 : tensor<1x256x1xf64>
    %1625 = stablehlo.broadcast_in_dim %1620, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1626 = stablehlo.broadcast_in_dim %1624, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1627 = stablehlo.subtract %1625, %1626 : tensor<1x256x512xf64>
    %1628 = stablehlo.multiply %1627, %1627 : tensor<1x256x512xf64>
    %1629 = stablehlo.reduce(%1628 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1630 = stablehlo.reshape %1629 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1631 = stablehlo.broadcast_in_dim %1630, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1632 = stablehlo.divide %1631, %23 : tensor<1x256x1xf64>
    %1633 = stablehlo.convert %1632 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1634 = stablehlo.reduce(%1619 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1635 = stablehlo.reshape %1634 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1636 = stablehlo.broadcast_in_dim %1635, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1637 = stablehlo.divide %1636, %39 : tensor<1x256x1xf32>
    %1638 = stablehlo.broadcast_in_dim %1633, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1639 = stablehlo.add %1638, %44 : tensor<1x256x1xf32>
    %1640 = stablehlo.rsqrt %1639 : tensor<1x256x1xf32>
    %1641 = stablehlo.broadcast_in_dim %1619, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1642 = stablehlo.broadcast_in_dim %1637, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1643 = stablehlo.subtract %1641, %1642 : tensor<1x256x512xf32>
    %1644 = stablehlo.broadcast_in_dim %1643, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1645 = stablehlo.broadcast_in_dim %1640, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1646 = stablehlo.multiply %1644, %1645 : tensor<1x256x512xf32>
    %1647 = stablehlo.convert %arg79 : (tensor<512xbf16>) -> tensor<512xf32>
    %1648 = stablehlo.broadcast_in_dim %1646, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1649 = stablehlo.broadcast_in_dim %1647, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1650 = stablehlo.multiply %1648, %1649 : tensor<1x256x512xf32>
    %1651 = stablehlo.convert %arg80 : (tensor<512xbf16>) -> tensor<512xf32>
    %1652 = stablehlo.broadcast_in_dim %1650, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1653 = stablehlo.broadcast_in_dim %1651, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1654 = stablehlo.add %1652, %1653 : tensor<1x256x512xf32>
    %1655 = stablehlo.convert %1654 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1656 = stablehlo.reshape %1655 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %1657 = stablehlo.convert %1656 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1658 = stablehlo.dot_general %1657, %arg137, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %1659 = stablehlo.broadcast_in_dim %1658, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1660 = stablehlo.multiply %1659, %146 : tensor<256x256xf32>
    %1661 = stablehlo.broadcast_in_dim %1660, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1662 = stablehlo.broadcast_in_dim %arg138, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1663 = stablehlo.add %1661, %1662 : tensor<256x256xf32>
    %1664 = stablehlo.convert %1663 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1665 = stablehlo.reshape %1664 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1666 = stablehlo.multiply %1665, %cst_22 : tensor<1x256x256xbf16>
    %1667 = stablehlo.multiply %1665, %154 : tensor<1x256x256xbf16>
    %1668 = stablehlo.convert %1667 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1669 = stablehlo.clamp %cst_23, %1668, %cst_24 : tensor<1x256x256xf32>
    %1670 = stablehlo.multiply %1669, %1669 : tensor<1x256x256xf32>
    %1671 = stablehlo.multiply %cst_25, %1670 : tensor<1x256x256xf32>
    %1672 = stablehlo.add %1671, %cst_26 : tensor<1x256x256xf32>
    %1673 = stablehlo.multiply %1672, %1670 : tensor<1x256x256xf32>
    %1674 = stablehlo.add %1673, %cst_27 : tensor<1x256x256xf32>
    %1675 = stablehlo.multiply %1674, %1670 : tensor<1x256x256xf32>
    %1676 = stablehlo.add %1675, %cst_28 : tensor<1x256x256xf32>
    %1677 = stablehlo.multiply %1676, %1670 : tensor<1x256x256xf32>
    %1678 = stablehlo.add %1677, %cst_29 : tensor<1x256x256xf32>
    %1679 = stablehlo.multiply %1678, %1670 : tensor<1x256x256xf32>
    %1680 = stablehlo.add %1679, %cst_30 : tensor<1x256x256xf32>
    %1681 = stablehlo.multiply %1680, %1670 : tensor<1x256x256xf32>
    %1682 = stablehlo.add %1681, %cst_31 : tensor<1x256x256xf32>
    %1683 = stablehlo.multiply %cst_32, %1670 : tensor<1x256x256xf32>
    %1684 = stablehlo.add %1683, %cst_33 : tensor<1x256x256xf32>
    %1685 = stablehlo.multiply %1684, %1670 : tensor<1x256x256xf32>
    %1686 = stablehlo.add %1685, %cst_34 : tensor<1x256x256xf32>
    %1687 = stablehlo.multiply %1686, %1670 : tensor<1x256x256xf32>
    %1688 = stablehlo.add %1687, %cst_35 : tensor<1x256x256xf32>
    %1689 = stablehlo.multiply %1688, %1670 : tensor<1x256x256xf32>
    %1690 = stablehlo.add %1689, %cst_36 : tensor<1x256x256xf32>
    %1691 = stablehlo.multiply %1669, %1682 : tensor<1x256x256xf32>
    %1692 = stablehlo.divide %1691, %1690 : tensor<1x256x256xf32>
    %1693 = stablehlo.clamp %cst_37, %1692, %cst_38 : tensor<1x256x256xf32>
    %1694 = stablehlo.convert %1693 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1695 = stablehlo.add %1694, %cst_20 : tensor<1x256x256xbf16>
    %1696 = stablehlo.multiply %1695, %1666 : tensor<1x256x256xbf16>
    %1697 = stablehlo.reshape %1696 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1698 = stablehlo.convert %1697 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1699 = stablehlo.dot_general %1698, %arg139, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1701 = stablehlo.multiply %1700, %9 : tensor<256x512xf32>
    %1702 = stablehlo.broadcast_in_dim %1701, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1703 = stablehlo.broadcast_in_dim %arg140, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %1704 = stablehlo.add %1702, %1703 : tensor<256x512xf32>
    %1705 = stablehlo.convert %1704 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1706 = stablehlo.reshape %1705 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %1707 = stablehlo.add %1706, %1618 : tensor<1x256x512xbf16>
    %1708 = stablehlo.convert %1707 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1709 = stablehlo.convert %1708 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1710 = stablehlo.reduce(%1709 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1711 = stablehlo.reshape %1710 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1712 = stablehlo.broadcast_in_dim %1711, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1713 = stablehlo.divide %1712, %23 : tensor<1x256x1xf64>
    %1714 = stablehlo.broadcast_in_dim %1709, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1715 = stablehlo.broadcast_in_dim %1713, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1716 = stablehlo.subtract %1714, %1715 : tensor<1x256x512xf64>
    %1717 = stablehlo.multiply %1716, %1716 : tensor<1x256x512xf64>
    %1718 = stablehlo.reduce(%1717 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1719 = stablehlo.reshape %1718 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1720 = stablehlo.broadcast_in_dim %1719, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1721 = stablehlo.divide %1720, %23 : tensor<1x256x1xf64>
    %1722 = stablehlo.convert %1721 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1723 = stablehlo.reduce(%1708 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1724 = stablehlo.reshape %1723 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1725 = stablehlo.broadcast_in_dim %1724, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1726 = stablehlo.divide %1725, %39 : tensor<1x256x1xf32>
    %1727 = stablehlo.broadcast_in_dim %1722, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1728 = stablehlo.add %1727, %44 : tensor<1x256x1xf32>
    %1729 = stablehlo.rsqrt %1728 : tensor<1x256x1xf32>
    %1730 = stablehlo.broadcast_in_dim %1708, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1731 = stablehlo.broadcast_in_dim %1726, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1732 = stablehlo.subtract %1730, %1731 : tensor<1x256x512xf32>
    %1733 = stablehlo.broadcast_in_dim %1732, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1734 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1735 = stablehlo.multiply %1733, %1734 : tensor<1x256x512xf32>
    %1736 = stablehlo.convert %arg81 : (tensor<512xbf16>) -> tensor<512xf32>
    %1737 = stablehlo.broadcast_in_dim %1735, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1738 = stablehlo.broadcast_in_dim %1736, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1739 = stablehlo.multiply %1737, %1738 : tensor<1x256x512xf32>
    %1740 = stablehlo.convert %arg82 : (tensor<512xbf16>) -> tensor<512xf32>
    %1741 = stablehlo.broadcast_in_dim %1739, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1742 = stablehlo.broadcast_in_dim %1740, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1743 = stablehlo.add %1741, %1742 : tensor<1x256x512xf32>
    %1744 = stablehlo.convert %1743 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1745 = stablehlo.convolution(%1744, %arg83) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %1746 = stablehlo.reshape %arg84 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %1747 = stablehlo.broadcast_in_dim %1745, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %1748 = stablehlo.broadcast_in_dim %1746, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %1749 = stablehlo.add %1747, %1748 : tensor<1x1024x512xbf16>
    %1750 = stablehlo.multiply %1749, %cst_3 : tensor<1x1024x512xbf16>
    %1751 = stablehlo.multiply %1749, %68 : tensor<1x1024x512xbf16>
    %1752 = stablehlo.convert %1751 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %1753 = stablehlo.clamp %cst_4, %1752, %cst_5 : tensor<1x1024x512xf32>
    %1754 = stablehlo.multiply %1753, %1753 : tensor<1x1024x512xf32>
    %1755 = stablehlo.multiply %cst_6, %1754 : tensor<1x1024x512xf32>
    %1756 = stablehlo.add %1755, %cst_7 : tensor<1x1024x512xf32>
    %1757 = stablehlo.multiply %1756, %1754 : tensor<1x1024x512xf32>
    %1758 = stablehlo.add %1757, %cst_8 : tensor<1x1024x512xf32>
    %1759 = stablehlo.multiply %1758, %1754 : tensor<1x1024x512xf32>
    %1760 = stablehlo.add %1759, %cst_9 : tensor<1x1024x512xf32>
    %1761 = stablehlo.multiply %1760, %1754 : tensor<1x1024x512xf32>
    %1762 = stablehlo.add %1761, %cst_10 : tensor<1x1024x512xf32>
    %1763 = stablehlo.multiply %1762, %1754 : tensor<1x1024x512xf32>
    %1764 = stablehlo.add %1763, %cst_11 : tensor<1x1024x512xf32>
    %1765 = stablehlo.multiply %1764, %1754 : tensor<1x1024x512xf32>
    %1766 = stablehlo.add %1765, %cst_12 : tensor<1x1024x512xf32>
    %1767 = stablehlo.multiply %cst_13, %1754 : tensor<1x1024x512xf32>
    %1768 = stablehlo.add %1767, %cst_14 : tensor<1x1024x512xf32>
    %1769 = stablehlo.multiply %1768, %1754 : tensor<1x1024x512xf32>
    %1770 = stablehlo.add %1769, %cst_15 : tensor<1x1024x512xf32>
    %1771 = stablehlo.multiply %1770, %1754 : tensor<1x1024x512xf32>
    %1772 = stablehlo.add %1771, %cst_16 : tensor<1x1024x512xf32>
    %1773 = stablehlo.multiply %1772, %1754 : tensor<1x1024x512xf32>
    %1774 = stablehlo.add %1773, %cst_17 : tensor<1x1024x512xf32>
    %1775 = stablehlo.multiply %1753, %1766 : tensor<1x1024x512xf32>
    %1776 = stablehlo.divide %1775, %1774 : tensor<1x1024x512xf32>
    %1777 = stablehlo.clamp %cst_18, %1776, %cst_19 : tensor<1x1024x512xf32>
    %1778 = stablehlo.convert %1777 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %1779 = stablehlo.add %1778, %cst_1 : tensor<1x1024x512xbf16>
    %1780 = stablehlo.multiply %1779, %1750 : tensor<1x1024x512xbf16>
    %1781 = stablehlo.convolution(%1780, %arg85) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %1782 = stablehlo.reshape %arg86 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %1783 = stablehlo.broadcast_in_dim %1781, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %1784 = stablehlo.broadcast_in_dim %1782, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %1785 = stablehlo.add %1783, %1784 : tensor<1x256x512xbf16>
    %1786 = stablehlo.add %1785, %1707 : tensor<1x256x512xbf16>
    %1787 = stablehlo.convert %1786 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1788 = stablehlo.convert %1787 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1789 = stablehlo.reduce(%1788 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1790 = stablehlo.reshape %1789 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1791 = stablehlo.broadcast_in_dim %1790, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1792 = stablehlo.divide %1791, %23 : tensor<1x256x1xf64>
    %1793 = stablehlo.broadcast_in_dim %1788, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1794 = stablehlo.broadcast_in_dim %1792, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1795 = stablehlo.subtract %1793, %1794 : tensor<1x256x512xf64>
    %1796 = stablehlo.multiply %1795, %1795 : tensor<1x256x512xf64>
    %1797 = stablehlo.reduce(%1796 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1798 = stablehlo.reshape %1797 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1799 = stablehlo.broadcast_in_dim %1798, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1800 = stablehlo.divide %1799, %23 : tensor<1x256x1xf64>
    %1801 = stablehlo.convert %1800 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1802 = stablehlo.reduce(%1787 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1803 = stablehlo.reshape %1802 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1804 = stablehlo.broadcast_in_dim %1803, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1805 = stablehlo.divide %1804, %39 : tensor<1x256x1xf32>
    %1806 = stablehlo.broadcast_in_dim %1801, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1807 = stablehlo.add %1806, %44 : tensor<1x256x1xf32>
    %1808 = stablehlo.rsqrt %1807 : tensor<1x256x1xf32>
    %1809 = stablehlo.broadcast_in_dim %1787, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1810 = stablehlo.broadcast_in_dim %1805, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1811 = stablehlo.subtract %1809, %1810 : tensor<1x256x512xf32>
    %1812 = stablehlo.broadcast_in_dim %1811, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1813 = stablehlo.broadcast_in_dim %1808, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1814 = stablehlo.multiply %1812, %1813 : tensor<1x256x512xf32>
    %1815 = stablehlo.convert %arg87 : (tensor<512xbf16>) -> tensor<512xf32>
    %1816 = stablehlo.broadcast_in_dim %1814, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1817 = stablehlo.broadcast_in_dim %1815, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1818 = stablehlo.multiply %1816, %1817 : tensor<1x256x512xf32>
    %1819 = stablehlo.convert %arg88 : (tensor<512xbf16>) -> tensor<512xf32>
    %1820 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1821 = stablehlo.broadcast_in_dim %1819, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1822 = stablehlo.add %1820, %1821 : tensor<1x256x512xf32>
    %1823 = stablehlo.convert %1822 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1824 = stablehlo.reshape %1823 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %1825 = stablehlo.convert %1824 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1826 = stablehlo.dot_general %1825, %arg141, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %1827 = stablehlo.broadcast_in_dim %1826, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1828 = stablehlo.multiply %1827, %146 : tensor<256x256xf32>
    %1829 = stablehlo.broadcast_in_dim %1828, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1830 = stablehlo.broadcast_in_dim %arg142, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1831 = stablehlo.add %1829, %1830 : tensor<256x256xf32>
    %1832 = stablehlo.convert %1831 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1833 = stablehlo.reshape %1832 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1834 = stablehlo.multiply %1833, %cst_22 : tensor<1x256x256xbf16>
    %1835 = stablehlo.multiply %1833, %154 : tensor<1x256x256xbf16>
    %1836 = stablehlo.convert %1835 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1837 = stablehlo.clamp %cst_23, %1836, %cst_24 : tensor<1x256x256xf32>
    %1838 = stablehlo.multiply %1837, %1837 : tensor<1x256x256xf32>
    %1839 = stablehlo.multiply %cst_25, %1838 : tensor<1x256x256xf32>
    %1840 = stablehlo.add %1839, %cst_26 : tensor<1x256x256xf32>
    %1841 = stablehlo.multiply %1840, %1838 : tensor<1x256x256xf32>
    %1842 = stablehlo.add %1841, %cst_27 : tensor<1x256x256xf32>
    %1843 = stablehlo.multiply %1842, %1838 : tensor<1x256x256xf32>
    %1844 = stablehlo.add %1843, %cst_28 : tensor<1x256x256xf32>
    %1845 = stablehlo.multiply %1844, %1838 : tensor<1x256x256xf32>
    %1846 = stablehlo.add %1845, %cst_29 : tensor<1x256x256xf32>
    %1847 = stablehlo.multiply %1846, %1838 : tensor<1x256x256xf32>
    %1848 = stablehlo.add %1847, %cst_30 : tensor<1x256x256xf32>
    %1849 = stablehlo.multiply %1848, %1838 : tensor<1x256x256xf32>
    %1850 = stablehlo.add %1849, %cst_31 : tensor<1x256x256xf32>
    %1851 = stablehlo.multiply %cst_32, %1838 : tensor<1x256x256xf32>
    %1852 = stablehlo.add %1851, %cst_33 : tensor<1x256x256xf32>
    %1853 = stablehlo.multiply %1852, %1838 : tensor<1x256x256xf32>
    %1854 = stablehlo.add %1853, %cst_34 : tensor<1x256x256xf32>
    %1855 = stablehlo.multiply %1854, %1838 : tensor<1x256x256xf32>
    %1856 = stablehlo.add %1855, %cst_35 : tensor<1x256x256xf32>
    %1857 = stablehlo.multiply %1856, %1838 : tensor<1x256x256xf32>
    %1858 = stablehlo.add %1857, %cst_36 : tensor<1x256x256xf32>
    %1859 = stablehlo.multiply %1837, %1850 : tensor<1x256x256xf32>
    %1860 = stablehlo.divide %1859, %1858 : tensor<1x256x256xf32>
    %1861 = stablehlo.clamp %cst_37, %1860, %cst_38 : tensor<1x256x256xf32>
    %1862 = stablehlo.convert %1861 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1863 = stablehlo.add %1862, %cst_20 : tensor<1x256x256xbf16>
    %1864 = stablehlo.multiply %1863, %1834 : tensor<1x256x256xbf16>
    %1865 = stablehlo.reshape %1864 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1866 = stablehlo.convert %1865 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1867 = stablehlo.dot_general %1866, %arg143, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %1868 = stablehlo.broadcast_in_dim %1867, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1869 = stablehlo.multiply %1868, %9 : tensor<256x512xf32>
    %1870 = stablehlo.broadcast_in_dim %1869, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %1871 = stablehlo.broadcast_in_dim %arg144, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %1872 = stablehlo.add %1870, %1871 : tensor<256x512xf32>
    %1873 = stablehlo.convert %1872 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1874 = stablehlo.reshape %1873 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %1875 = stablehlo.add %1874, %1786 : tensor<1x256x512xbf16>
    %1876 = stablehlo.convert %1875 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1877 = stablehlo.convert %1876 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1878 = stablehlo.reduce(%1877 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1879 = stablehlo.reshape %1878 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1880 = stablehlo.broadcast_in_dim %1879, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1881 = stablehlo.divide %1880, %23 : tensor<1x256x1xf64>
    %1882 = stablehlo.broadcast_in_dim %1877, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1883 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1884 = stablehlo.subtract %1882, %1883 : tensor<1x256x512xf64>
    %1885 = stablehlo.multiply %1884, %1884 : tensor<1x256x512xf64>
    %1886 = stablehlo.reduce(%1885 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1887 = stablehlo.reshape %1886 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1888 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1889 = stablehlo.divide %1888, %23 : tensor<1x256x1xf64>
    %1890 = stablehlo.convert %1889 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1891 = stablehlo.reduce(%1876 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1892 = stablehlo.reshape %1891 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1893 = stablehlo.broadcast_in_dim %1892, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1894 = stablehlo.divide %1893, %39 : tensor<1x256x1xf32>
    %1895 = stablehlo.broadcast_in_dim %1890, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1896 = stablehlo.add %1895, %44 : tensor<1x256x1xf32>
    %1897 = stablehlo.rsqrt %1896 : tensor<1x256x1xf32>
    %1898 = stablehlo.broadcast_in_dim %1876, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1899 = stablehlo.broadcast_in_dim %1894, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1900 = stablehlo.subtract %1898, %1899 : tensor<1x256x512xf32>
    %1901 = stablehlo.broadcast_in_dim %1900, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1902 = stablehlo.broadcast_in_dim %1897, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1903 = stablehlo.multiply %1901, %1902 : tensor<1x256x512xf32>
    %1904 = stablehlo.convert %arg89 : (tensor<512xbf16>) -> tensor<512xf32>
    %1905 = stablehlo.broadcast_in_dim %1903, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1906 = stablehlo.broadcast_in_dim %1904, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1907 = stablehlo.multiply %1905, %1906 : tensor<1x256x512xf32>
    %1908 = stablehlo.convert %arg90 : (tensor<512xbf16>) -> tensor<512xf32>
    %1909 = stablehlo.broadcast_in_dim %1907, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1910 = stablehlo.broadcast_in_dim %1908, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1911 = stablehlo.add %1909, %1910 : tensor<1x256x512xf32>
    %1912 = stablehlo.convert %1911 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1913 = stablehlo.convolution(%1912, %arg91) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x512xbf16>, tensor<1024x256x1xbf16>) -> tensor<1x1024x512xbf16>
    %1914 = stablehlo.reshape %arg92 : (tensor<1024xbf16>) -> tensor<1024x1xbf16>
    %1915 = stablehlo.broadcast_in_dim %1913, dims = [0, 1, 2] : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xbf16>
    %1916 = stablehlo.broadcast_in_dim %1914, dims = [1, 2] : (tensor<1024x1xbf16>) -> tensor<1x1024x512xbf16>
    %1917 = stablehlo.add %1915, %1916 : tensor<1x1024x512xbf16>
    %1918 = stablehlo.multiply %1917, %cst_3 : tensor<1x1024x512xbf16>
    %1919 = stablehlo.multiply %1917, %68 : tensor<1x1024x512xbf16>
    %1920 = stablehlo.convert %1919 : (tensor<1x1024x512xbf16>) -> tensor<1x1024x512xf32>
    %1921 = stablehlo.clamp %cst_4, %1920, %cst_5 : tensor<1x1024x512xf32>
    %1922 = stablehlo.multiply %1921, %1921 : tensor<1x1024x512xf32>
    %1923 = stablehlo.multiply %cst_6, %1922 : tensor<1x1024x512xf32>
    %1924 = stablehlo.add %1923, %cst_7 : tensor<1x1024x512xf32>
    %1925 = stablehlo.multiply %1924, %1922 : tensor<1x1024x512xf32>
    %1926 = stablehlo.add %1925, %cst_8 : tensor<1x1024x512xf32>
    %1927 = stablehlo.multiply %1926, %1922 : tensor<1x1024x512xf32>
    %1928 = stablehlo.add %1927, %cst_9 : tensor<1x1024x512xf32>
    %1929 = stablehlo.multiply %1928, %1922 : tensor<1x1024x512xf32>
    %1930 = stablehlo.add %1929, %cst_10 : tensor<1x1024x512xf32>
    %1931 = stablehlo.multiply %1930, %1922 : tensor<1x1024x512xf32>
    %1932 = stablehlo.add %1931, %cst_11 : tensor<1x1024x512xf32>
    %1933 = stablehlo.multiply %1932, %1922 : tensor<1x1024x512xf32>
    %1934 = stablehlo.add %1933, %cst_12 : tensor<1x1024x512xf32>
    %1935 = stablehlo.multiply %cst_13, %1922 : tensor<1x1024x512xf32>
    %1936 = stablehlo.add %1935, %cst_14 : tensor<1x1024x512xf32>
    %1937 = stablehlo.multiply %1936, %1922 : tensor<1x1024x512xf32>
    %1938 = stablehlo.add %1937, %cst_15 : tensor<1x1024x512xf32>
    %1939 = stablehlo.multiply %1938, %1922 : tensor<1x1024x512xf32>
    %1940 = stablehlo.add %1939, %cst_16 : tensor<1x1024x512xf32>
    %1941 = stablehlo.multiply %1940, %1922 : tensor<1x1024x512xf32>
    %1942 = stablehlo.add %1941, %cst_17 : tensor<1x1024x512xf32>
    %1943 = stablehlo.multiply %1921, %1934 : tensor<1x1024x512xf32>
    %1944 = stablehlo.divide %1943, %1942 : tensor<1x1024x512xf32>
    %1945 = stablehlo.clamp %cst_18, %1944, %cst_19 : tensor<1x1024x512xf32>
    %1946 = stablehlo.convert %1945 : (tensor<1x1024x512xf32>) -> tensor<1x1024x512xbf16>
    %1947 = stablehlo.add %1946, %cst_1 : tensor<1x1024x512xbf16>
    %1948 = stablehlo.multiply %1947, %1918 : tensor<1x1024x512xbf16>
    %1949 = stablehlo.convolution(%1948, %arg93) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x512xbf16>, tensor<256x1024x1xbf16>) -> tensor<1x256x512xbf16>
    %1950 = stablehlo.reshape %arg94 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %1951 = stablehlo.broadcast_in_dim %1949, dims = [0, 1, 2] : (tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %1952 = stablehlo.broadcast_in_dim %1950, dims = [1, 2] : (tensor<256x1xbf16>) -> tensor<1x256x512xbf16>
    %1953 = stablehlo.add %1951, %1952 : tensor<1x256x512xbf16>
    %1954 = stablehlo.add %1953, %1875 : tensor<1x256x512xbf16>
    %1955 = stablehlo.convert %1954 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %1956 = stablehlo.convert %1955 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %1957 = stablehlo.reduce(%1956 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1958 = stablehlo.reshape %1957 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1959 = stablehlo.broadcast_in_dim %1958, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1960 = stablehlo.divide %1959, %23 : tensor<1x256x1xf64>
    %1961 = stablehlo.broadcast_in_dim %1956, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %1962 = stablehlo.broadcast_in_dim %1960, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %1963 = stablehlo.subtract %1961, %1962 : tensor<1x256x512xf64>
    %1964 = stablehlo.multiply %1963, %1963 : tensor<1x256x512xf64>
    %1965 = stablehlo.reduce(%1964 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1966 = stablehlo.reshape %1965 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1967 = stablehlo.broadcast_in_dim %1966, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1968 = stablehlo.divide %1967, %23 : tensor<1x256x1xf64>
    %1969 = stablehlo.convert %1968 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1970 = stablehlo.reduce(%1955 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1971 = stablehlo.reshape %1970 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1972 = stablehlo.broadcast_in_dim %1971, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1973 = stablehlo.divide %1972, %39 : tensor<1x256x1xf32>
    %1974 = stablehlo.broadcast_in_dim %1969, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1975 = stablehlo.add %1974, %44 : tensor<1x256x1xf32>
    %1976 = stablehlo.rsqrt %1975 : tensor<1x256x1xf32>
    %1977 = stablehlo.broadcast_in_dim %1955, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1978 = stablehlo.broadcast_in_dim %1973, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1979 = stablehlo.subtract %1977, %1978 : tensor<1x256x512xf32>
    %1980 = stablehlo.broadcast_in_dim %1979, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1981 = stablehlo.broadcast_in_dim %1976, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %1982 = stablehlo.multiply %1980, %1981 : tensor<1x256x512xf32>
    %1983 = stablehlo.convert %arg95 : (tensor<512xbf16>) -> tensor<512xf32>
    %1984 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1985 = stablehlo.broadcast_in_dim %1983, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1986 = stablehlo.multiply %1984, %1985 : tensor<1x256x512xf32>
    %1987 = stablehlo.convert %arg96 : (tensor<512xbf16>) -> tensor<512xf32>
    %1988 = stablehlo.broadcast_in_dim %1986, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %1989 = stablehlo.broadcast_in_dim %1987, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %1990 = stablehlo.add %1988, %1989 : tensor<1x256x512xf32>
    %1991 = stablehlo.convert %1990 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %1992 = stablehlo.reshape %1991 : (tensor<1x256x512xbf16>) -> tensor<256x512xbf16>
    %1993 = stablehlo.convert %1992 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1994 = stablehlo.dot_general %1993, %arg145, contracting_dims = [1] x [0] : (tensor<256x512xf32>, tensor<512x256xf32>) -> tensor<256x256xf32>
    %1995 = stablehlo.broadcast_in_dim %1994, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1996 = stablehlo.multiply %1995, %146 : tensor<256x256xf32>
    %1997 = stablehlo.broadcast_in_dim %1996, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1998 = stablehlo.broadcast_in_dim %arg146, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1999 = stablehlo.add %1997, %1998 : tensor<256x256xf32>
    %2000 = stablehlo.convert %1999 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2001 = stablehlo.reshape %2000 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2002 = stablehlo.multiply %2001, %cst_22 : tensor<1x256x256xbf16>
    %2003 = stablehlo.multiply %2001, %154 : tensor<1x256x256xbf16>
    %2004 = stablehlo.convert %2003 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %2005 = stablehlo.clamp %cst_23, %2004, %cst_24 : tensor<1x256x256xf32>
    %2006 = stablehlo.multiply %2005, %2005 : tensor<1x256x256xf32>
    %2007 = stablehlo.multiply %cst_25, %2006 : tensor<1x256x256xf32>
    %2008 = stablehlo.add %2007, %cst_26 : tensor<1x256x256xf32>
    %2009 = stablehlo.multiply %2008, %2006 : tensor<1x256x256xf32>
    %2010 = stablehlo.add %2009, %cst_27 : tensor<1x256x256xf32>
    %2011 = stablehlo.multiply %2010, %2006 : tensor<1x256x256xf32>
    %2012 = stablehlo.add %2011, %cst_28 : tensor<1x256x256xf32>
    %2013 = stablehlo.multiply %2012, %2006 : tensor<1x256x256xf32>
    %2014 = stablehlo.add %2013, %cst_29 : tensor<1x256x256xf32>
    %2015 = stablehlo.multiply %2014, %2006 : tensor<1x256x256xf32>
    %2016 = stablehlo.add %2015, %cst_30 : tensor<1x256x256xf32>
    %2017 = stablehlo.multiply %2016, %2006 : tensor<1x256x256xf32>
    %2018 = stablehlo.add %2017, %cst_31 : tensor<1x256x256xf32>
    %2019 = stablehlo.multiply %cst_32, %2006 : tensor<1x256x256xf32>
    %2020 = stablehlo.add %2019, %cst_33 : tensor<1x256x256xf32>
    %2021 = stablehlo.multiply %2020, %2006 : tensor<1x256x256xf32>
    %2022 = stablehlo.add %2021, %cst_34 : tensor<1x256x256xf32>
    %2023 = stablehlo.multiply %2022, %2006 : tensor<1x256x256xf32>
    %2024 = stablehlo.add %2023, %cst_35 : tensor<1x256x256xf32>
    %2025 = stablehlo.multiply %2024, %2006 : tensor<1x256x256xf32>
    %2026 = stablehlo.add %2025, %cst_36 : tensor<1x256x256xf32>
    %2027 = stablehlo.multiply %2005, %2018 : tensor<1x256x256xf32>
    %2028 = stablehlo.divide %2027, %2026 : tensor<1x256x256xf32>
    %2029 = stablehlo.clamp %cst_37, %2028, %cst_38 : tensor<1x256x256xf32>
    %2030 = stablehlo.convert %2029 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %2031 = stablehlo.add %2030, %cst_20 : tensor<1x256x256xbf16>
    %2032 = stablehlo.multiply %2031, %2002 : tensor<1x256x256xbf16>
    %2033 = stablehlo.reshape %2032 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2034 = stablehlo.convert %2033 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %2035 = stablehlo.dot_general %2034, %arg147, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    %2036 = stablehlo.broadcast_in_dim %2035, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %2037 = stablehlo.multiply %2036, %9 : tensor<256x512xf32>
    %2038 = stablehlo.broadcast_in_dim %2037, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512xf32>
    %2039 = stablehlo.broadcast_in_dim %arg148, dims = [1] : (tensor<512xf32>) -> tensor<256x512xf32>
    %2040 = stablehlo.add %2038, %2039 : tensor<256x512xf32>
    %2041 = stablehlo.convert %2040 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %2042 = stablehlo.reshape %2041 : (tensor<256x512xbf16>) -> tensor<1x256x512xbf16>
    %2043 = stablehlo.add %2042, %1954 : tensor<1x256x512xbf16>
    %2044 = stablehlo.convert %2043 : (tensor<1x256x512xbf16>) -> tensor<1x256x512xf32>
    %2045 = stablehlo.convert %2044 : (tensor<1x256x512xf32>) -> tensor<1x256x512xf64>
    %2046 = stablehlo.reduce(%2045 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2047 = stablehlo.reshape %2046 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2048 = stablehlo.broadcast_in_dim %2047, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2049 = stablehlo.divide %2048, %23 : tensor<1x256x1xf64>
    %2050 = stablehlo.broadcast_in_dim %2045, dims = [0, 1, 2] : (tensor<1x256x512xf64>) -> tensor<1x256x512xf64>
    %2051 = stablehlo.broadcast_in_dim %2049, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x512xf64>
    %2052 = stablehlo.subtract %2050, %2051 : tensor<1x256x512xf64>
    %2053 = stablehlo.multiply %2052, %2052 : tensor<1x256x512xf64>
    %2054 = stablehlo.reduce(%2053 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2055 = stablehlo.reshape %2054 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2056 = stablehlo.broadcast_in_dim %2055, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2057 = stablehlo.divide %2056, %23 : tensor<1x256x1xf64>
    %2058 = stablehlo.convert %2057 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2059 = stablehlo.reduce(%2044 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x512xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2060 = stablehlo.reshape %2059 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2061 = stablehlo.broadcast_in_dim %2060, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2062 = stablehlo.divide %2061, %39 : tensor<1x256x1xf32>
    %2063 = stablehlo.broadcast_in_dim %2058, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2064 = stablehlo.add %2063, %44 : tensor<1x256x1xf32>
    %2065 = stablehlo.rsqrt %2064 : tensor<1x256x1xf32>
    %2066 = stablehlo.broadcast_in_dim %2044, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %2067 = stablehlo.broadcast_in_dim %2062, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %2068 = stablehlo.subtract %2066, %2067 : tensor<1x256x512xf32>
    %2069 = stablehlo.broadcast_in_dim %2068, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %2070 = stablehlo.broadcast_in_dim %2065, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x512xf32>
    %2071 = stablehlo.multiply %2069, %2070 : tensor<1x256x512xf32>
    %2072 = stablehlo.convert %arg97 : (tensor<512xbf16>) -> tensor<512xf32>
    %2073 = stablehlo.broadcast_in_dim %2071, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %2074 = stablehlo.broadcast_in_dim %2072, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %2075 = stablehlo.multiply %2073, %2074 : tensor<1x256x512xf32>
    %2076 = stablehlo.convert %arg98 : (tensor<512xbf16>) -> tensor<512xf32>
    %2077 = stablehlo.broadcast_in_dim %2075, dims = [0, 1, 2] : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
    %2078 = stablehlo.broadcast_in_dim %2076, dims = [2] : (tensor<512xf32>) -> tensor<1x256x512xf32>
    %2079 = stablehlo.add %2077, %2078 : tensor<1x256x512xf32>
    %2080 = stablehlo.convert %2079 : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
    %2081 = stablehlo.transpose %2080, dims = [0, 2, 1] : (tensor<1x256x512xbf16>) -> tensor<1x512x256xbf16>
    %2082 = stablehlo.reduce(%2081 init: %cst_39) applies stablehlo.add across dimensions = [2] : (tensor<1x512x256xbf16>, tensor<bf16>) -> tensor<1x512xbf16>
    %2083 = stablehlo.convert %cst_43 : (tensor<1xi64>) -> tensor<1xbf16>
    %2084 = stablehlo.reshape %2083 : (tensor<1xbf16>) -> tensor<bf16>
    %2085 = stablehlo.broadcast_in_dim %2082, dims = [0, 1] : (tensor<1x512xbf16>) -> tensor<1x512xbf16>
    %2086 = stablehlo.broadcast_in_dim %2084, dims = [] : (tensor<bf16>) -> tensor<1x512xbf16>
    %2087 = stablehlo.divide %2085, %2086 : tensor<1x512xbf16>
    %2088 = stablehlo.convert %2087 : (tensor<1x512xbf16>) -> tensor<1x512xf32>
    %2089 = stablehlo.dot_general %2088, %arg149, contracting_dims = [1] x [0] : (tensor<1x512xf32>, tensor<512x1000xf32>) -> tensor<1x1000xf32>
    %2090 = stablehlo.broadcast_in_dim %2089, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %2091 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<1x1000xf32>
    %2092 = stablehlo.multiply %2090, %2091 : tensor<1x1000xf32>
    %2093 = stablehlo.broadcast_in_dim %2092, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %2094 = stablehlo.broadcast_in_dim %arg150, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %2095 = stablehlo.add %2093, %2094 : tensor<1x1000xf32>
    %2096 = stablehlo.convert %2095 : (tensor<1x1000xf32>) -> tensor<1x1000xbf16>
    return %2096 : tensor<1x1000xbf16>
  }
}
