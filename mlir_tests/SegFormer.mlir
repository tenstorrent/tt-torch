module {
  func.func @main(%arg0: tensor<1x3x512x512xbf16>, %arg1: tensor<32x3x7x7xbf16>, %arg2: tensor<32xbf16>, %arg3: tensor<32xbf16>, %arg4: tensor<32xbf16>, %arg5: tensor<32xbf16>, %arg6: tensor<32xbf16>, %arg7: tensor<32x32x8x8xbf16>, %arg8: tensor<32xbf16>, %arg9: tensor<32xbf16>, %arg10: tensor<32xbf16>, %arg11: tensor<32xbf16>, %arg12: tensor<32xbf16>, %arg13: tensor<128x1x3x3xbf16>, %arg14: tensor<128xbf16>, %arg15: tensor<32xbf16>, %arg16: tensor<32xbf16>, %arg17: tensor<32xbf16>, %arg18: tensor<32x32x8x8xbf16>, %arg19: tensor<32xbf16>, %arg20: tensor<32xbf16>, %arg21: tensor<32xbf16>, %arg22: tensor<32xbf16>, %arg23: tensor<32xbf16>, %arg24: tensor<128x1x3x3xbf16>, %arg25: tensor<128xbf16>, %arg26: tensor<32xbf16>, %arg27: tensor<32xbf16>, %arg28: tensor<32xbf16>, %arg29: tensor<64x32x3x3xbf16>, %arg30: tensor<64xbf16>, %arg31: tensor<64xbf16>, %arg32: tensor<64xbf16>, %arg33: tensor<64xbf16>, %arg34: tensor<64xbf16>, %arg35: tensor<64x64x4x4xbf16>, %arg36: tensor<64xbf16>, %arg37: tensor<64xbf16>, %arg38: tensor<64xbf16>, %arg39: tensor<64xbf16>, %arg40: tensor<64xbf16>, %arg41: tensor<256x1x3x3xbf16>, %arg42: tensor<256xbf16>, %arg43: tensor<64xbf16>, %arg44: tensor<64xbf16>, %arg45: tensor<64xbf16>, %arg46: tensor<64x64x4x4xbf16>, %arg47: tensor<64xbf16>, %arg48: tensor<64xbf16>, %arg49: tensor<64xbf16>, %arg50: tensor<64xbf16>, %arg51: tensor<64xbf16>, %arg52: tensor<256x1x3x3xbf16>, %arg53: tensor<256xbf16>, %arg54: tensor<64xbf16>, %arg55: tensor<64xbf16>, %arg56: tensor<64xbf16>, %arg57: tensor<160x64x3x3xbf16>, %arg58: tensor<160xbf16>, %arg59: tensor<160xbf16>, %arg60: tensor<160xbf16>, %arg61: tensor<160xbf16>, %arg62: tensor<160xbf16>, %arg63: tensor<160x160x2x2xbf16>, %arg64: tensor<160xbf16>, %arg65: tensor<160xbf16>, %arg66: tensor<160xbf16>, %arg67: tensor<160xbf16>, %arg68: tensor<160xbf16>, %arg69: tensor<640x1x3x3xbf16>, %arg70: tensor<640xbf16>, %arg71: tensor<160xbf16>, %arg72: tensor<160xbf16>, %arg73: tensor<160xbf16>, %arg74: tensor<160x160x2x2xbf16>, %arg75: tensor<160xbf16>, %arg76: tensor<160xbf16>, %arg77: tensor<160xbf16>, %arg78: tensor<160xbf16>, %arg79: tensor<160xbf16>, %arg80: tensor<640x1x3x3xbf16>, %arg81: tensor<640xbf16>, %arg82: tensor<160xbf16>, %arg83: tensor<160xbf16>, %arg84: tensor<160xbf16>, %arg85: tensor<256x160x3x3xbf16>, %arg86: tensor<256xbf16>, %arg87: tensor<256xbf16>, %arg88: tensor<256xbf16>, %arg89: tensor<256xbf16>, %arg90: tensor<256xbf16>, %arg91: tensor<256xbf16>, %arg92: tensor<256xbf16>, %arg93: tensor<1024x1x3x3xbf16>, %arg94: tensor<1024xbf16>, %arg95: tensor<256xbf16>, %arg96: tensor<256xbf16>, %arg97: tensor<256xbf16>, %arg98: tensor<256xbf16>, %arg99: tensor<256xbf16>, %arg100: tensor<1024x1x3x3xbf16>, %arg101: tensor<1024xbf16>, %arg102: tensor<256xbf16>, %arg103: tensor<256xbf16>, %arg104: tensor<256xbf16>, %arg105: tensor<256xbf16>, %arg106: tensor<256xbf16>, %arg107: tensor<256xbf16>, %arg108: tensor<256xbf16>, %arg109: tensor<256x1024x1x1xbf16>, %arg110: tensor<150x256x1x1xbf16>, %arg111: tensor<150xbf16>, %arg112: tensor<32x32xf32>, %arg113: tensor<32xf32>, %arg114: tensor<32x32xf32>, %arg115: tensor<32xf32>, %arg116: tensor<32x32xf32>, %arg117: tensor<32xf32>, %arg118: tensor<32x32xf32>, %arg119: tensor<32xf32>, %arg120: tensor<32x128xf32>, %arg121: tensor<128xf32>, %arg122: tensor<128x32xbf16>, %arg123: tensor<32x32xf32>, %arg124: tensor<32xf32>, %arg125: tensor<32x32xf32>, %arg126: tensor<32xf32>, %arg127: tensor<32x32xf32>, %arg128: tensor<32xf32>, %arg129: tensor<32x32xf32>, %arg130: tensor<32xf32>, %arg131: tensor<32x128xf32>, %arg132: tensor<128xf32>, %arg133: tensor<128x32xbf16>, %arg134: tensor<64x64xf32>, %arg135: tensor<64xf32>, %arg136: tensor<64x64xf32>, %arg137: tensor<64xf32>, %arg138: tensor<64x64xf32>, %arg139: tensor<64xf32>, %arg140: tensor<64x64xf32>, %arg141: tensor<64xf32>, %arg142: tensor<64x256xf32>, %arg143: tensor<256xf32>, %arg144: tensor<256x64xbf16>, %arg145: tensor<64x64xf32>, %arg146: tensor<64xf32>, %arg147: tensor<64x64xf32>, %arg148: tensor<64xf32>, %arg149: tensor<64x64xf32>, %arg150: tensor<64xf32>, %arg151: tensor<64x64xf32>, %arg152: tensor<64xf32>, %arg153: tensor<64x256xf32>, %arg154: tensor<256xf32>, %arg155: tensor<256x64xbf16>, %arg156: tensor<160x160xf32>, %arg157: tensor<160xf32>, %arg158: tensor<160x160xf32>, %arg159: tensor<160xf32>, %arg160: tensor<160x160xf32>, %arg161: tensor<160xf32>, %arg162: tensor<160x160xf32>, %arg163: tensor<160xf32>, %arg164: tensor<160x640xf32>, %arg165: tensor<640xf32>, %arg166: tensor<640x160xbf16>, %arg167: tensor<160x160xf32>, %arg168: tensor<160xf32>, %arg169: tensor<160x160xf32>, %arg170: tensor<160xf32>, %arg171: tensor<160x160xf32>, %arg172: tensor<160xf32>, %arg173: tensor<160x160xf32>, %arg174: tensor<160xf32>, %arg175: tensor<160x640xf32>, %arg176: tensor<640xf32>, %arg177: tensor<640x160xbf16>, %arg178: tensor<256x256xf32>, %arg179: tensor<256xf32>, %arg180: tensor<256x256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256x256xf32>, %arg183: tensor<256xf32>, %arg184: tensor<256x256xf32>, %arg185: tensor<256xf32>, %arg186: tensor<256x1024xf32>, %arg187: tensor<1024xf32>, %arg188: tensor<1024x256xbf16>, %arg189: tensor<256x256xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256x256xf32>, %arg192: tensor<256xf32>, %arg193: tensor<256x256xf32>, %arg194: tensor<256xf32>, %arg195: tensor<256x256xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256x1024xf32>, %arg198: tensor<1024xf32>, %arg199: tensor<1024x256xbf16>, %arg200: tensor<32x256xbf16>, %arg201: tensor<256x128x128xbf16>, %arg202: tensor<256x128x128xbf16>, %arg203: tensor<64x256xbf16>, %arg204: tensor<256x64x128xbf16>, %arg205: tensor<256x64x128xbf16>, %arg206: tensor<160x256xbf16>, %arg207: tensor<256x32x128xbf16>, %arg208: tensor<256x32x128xbf16>, %arg209: tensor<256x256xbf16>, %arg210: tensor<256x16x128xbf16>, %arg211: tensor<256x16x128xbf16>, %arg212: tensor<256x1x1xf32>, %arg213: tensor<256x1x1xf32>, %arg214: tensor<256x1x1xbf16>, %arg215: tensor<256x1x1xbf16>) -> tensor<1x150x128x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<1x16384x128xbf16>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<1x16384x128xbf16>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<1x16384x128xbf16>
    %cst_5 = stablehlo.constant dense<-4.000000e+00> : tensor<1x16384x128xf32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<1x16384x128xf32>
    %cst_7 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x16384x128xf32>
    %cst_8 = stablehlo.constant dense<2.77068146E-8> : tensor<1x16384x128xf32>
    %cst_9 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x16384x128xf32>
    %cst_10 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x16384x128xf32>
    %cst_11 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x16384x128xf32>
    %cst_12 = stablehlo.constant dense<-2.954600e-03> : tensor<1x16384x128xf32>
    %cst_13 = stablehlo.constant dense<-0.0160960332> : tensor<1x16384x128xf32>
    %cst_14 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x16384x128xf32>
    %cst_15 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x16384x128xf32>
    %cst_16 = stablehlo.constant dense<-0.00168282702> : tensor<1x16384x128xf32>
    %cst_17 = stablehlo.constant dense<-0.00737332925> : tensor<1x16384x128xf32>
    %cst_18 = stablehlo.constant dense<-0.0142647391> : tensor<1x16384x128xf32>
    %cst_19 = stablehlo.constant dense<-1.000000e+00> : tensor<1x16384x128xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x16384x128xf32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<1x4096x256xbf16>
    %cst_22 = stablehlo.constant dense<2.000000e+00> : tensor<1x4096x256xbf16>
    %cst_23 = stablehlo.constant dense<5.000000e-01> : tensor<1x4096x256xbf16>
    %cst_24 = stablehlo.constant dense<-4.000000e+00> : tensor<1x4096x256xf32>
    %cst_25 = stablehlo.constant dense<4.000000e+00> : tensor<1x4096x256xf32>
    %cst_26 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x4096x256xf32>
    %cst_27 = stablehlo.constant dense<2.77068146E-8> : tensor<1x4096x256xf32>
    %cst_28 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x4096x256xf32>
    %cst_29 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x4096x256xf32>
    %cst_30 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x4096x256xf32>
    %cst_31 = stablehlo.constant dense<-2.954600e-03> : tensor<1x4096x256xf32>
    %cst_32 = stablehlo.constant dense<-0.0160960332> : tensor<1x4096x256xf32>
    %cst_33 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x4096x256xf32>
    %cst_34 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x4096x256xf32>
    %cst_35 = stablehlo.constant dense<-0.00168282702> : tensor<1x4096x256xf32>
    %cst_36 = stablehlo.constant dense<-0.00737332925> : tensor<1x4096x256xf32>
    %cst_37 = stablehlo.constant dense<-0.0142647391> : tensor<1x4096x256xf32>
    %cst_38 = stablehlo.constant dense<-1.000000e+00> : tensor<1x4096x256xf32>
    %cst_39 = stablehlo.constant dense<1.000000e+00> : tensor<1x4096x256xf32>
    %cst_40 = stablehlo.constant dense<1.000000e+00> : tensor<1x1024x640xbf16>
    %cst_41 = stablehlo.constant dense<2.000000e+00> : tensor<1x1024x640xbf16>
    %cst_42 = stablehlo.constant dense<5.000000e-01> : tensor<1x1024x640xbf16>
    %cst_43 = stablehlo.constant dense<-4.000000e+00> : tensor<1x1024x640xf32>
    %cst_44 = stablehlo.constant dense<4.000000e+00> : tensor<1x1024x640xf32>
    %cst_45 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x1024x640xf32>
    %cst_46 = stablehlo.constant dense<2.77068146E-8> : tensor<1x1024x640xf32>
    %cst_47 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x1024x640xf32>
    %cst_48 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x1024x640xf32>
    %cst_49 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x1024x640xf32>
    %cst_50 = stablehlo.constant dense<-2.954600e-03> : tensor<1x1024x640xf32>
    %cst_51 = stablehlo.constant dense<-0.0160960332> : tensor<1x1024x640xf32>
    %cst_52 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x1024x640xf32>
    %cst_53 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x1024x640xf32>
    %cst_54 = stablehlo.constant dense<-0.00168282702> : tensor<1x1024x640xf32>
    %cst_55 = stablehlo.constant dense<-0.00737332925> : tensor<1x1024x640xf32>
    %cst_56 = stablehlo.constant dense<-0.0142647391> : tensor<1x1024x640xf32>
    %cst_57 = stablehlo.constant dense<-1.000000e+00> : tensor<1x1024x640xf32>
    %cst_58 = stablehlo.constant dense<1.000000e+00> : tensor<1x1024x640xf32>
    %cst_59 = stablehlo.constant dense<1.000000e+00> : tensor<1x256x1024xbf16>
    %cst_60 = stablehlo.constant dense<2.000000e+00> : tensor<1x256x1024xbf16>
    %cst_61 = stablehlo.constant dense<5.000000e-01> : tensor<1x256x1024xbf16>
    %cst_62 = stablehlo.constant dense<-4.000000e+00> : tensor<1x256x1024xf32>
    %cst_63 = stablehlo.constant dense<4.000000e+00> : tensor<1x256x1024xf32>
    %cst_64 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x256x1024xf32>
    %cst_65 = stablehlo.constant dense<2.77068146E-8> : tensor<1x256x1024xf32>
    %cst_66 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x256x1024xf32>
    %cst_67 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x256x1024xf32>
    %cst_68 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x256x1024xf32>
    %cst_69 = stablehlo.constant dense<-2.954600e-03> : tensor<1x256x1024xf32>
    %cst_70 = stablehlo.constant dense<-0.0160960332> : tensor<1x256x1024xf32>
    %cst_71 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x256x1024xf32>
    %cst_72 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x256x1024xf32>
    %cst_73 = stablehlo.constant dense<-0.00168282702> : tensor<1x256x1024xf32>
    %cst_74 = stablehlo.constant dense<-0.00737332925> : tensor<1x256x1024xf32>
    %cst_75 = stablehlo.constant dense<-0.0142647391> : tensor<1x256x1024xf32>
    %cst_76 = stablehlo.constant dense<-1.000000e+00> : tensor<1x256x1024xf32>
    %cst_77 = stablehlo.constant dense<1.000000e+00> : tensor<1x256x1024xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x128x128xbf16>
    %cst_79 = arith.constant dense<32> : tensor<1xi64>
    %cst_80 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_81 = arith.constant dense<1> : tensor<1xi64>
    %cst_82 = arith.constant dense<5.6568542494923806> : tensor<1xf64>
    %cst_83 = arith.constant dense<64> : tensor<1xi64>
    %cst_84 = arith.constant dense<160> : tensor<1xi64>
    %cst_85 = arith.constant dense<256> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [4, 4], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x512x512xbf16>, tensor<32x3x7x7xbf16>) -> tensor<1x32x128x128xbf16>
    %1 = stablehlo.reshape %arg2 : (tensor<32xbf16>) -> tensor<32x1x1xbf16>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<32x1x1xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = stablehlo.add %2, %3 : tensor<1x32x128x128xbf16>
    %5 = stablehlo.reshape %4 : (tensor<1x32x128x128xbf16>) -> tensor<1x32x16384xbf16>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1] : (tensor<1x32x16384xbf16>) -> tensor<1x16384x32xbf16>
    %7 = stablehlo.convert %6 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xf32>
    %8 = stablehlo.convert %7 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf64>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %10 = stablehlo.reshape %9 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %11 = stablehlo.convert %cst_79 : (tensor<1xi64>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f64>) -> tensor<1x16384x1xf64>
    %15 = stablehlo.divide %13, %14 : tensor<1x16384x1xf64>
    %16 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x16384x32xf64>) -> tensor<1x16384x32xf64>
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x32xf64>
    %18 = stablehlo.subtract %16, %17 : tensor<1x16384x32xf64>
    %19 = stablehlo.multiply %18, %18 : tensor<1x16384x32xf64>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %21 = stablehlo.reshape %20 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %23 = stablehlo.divide %22, %14 : tensor<1x16384x1xf64>
    %24 = stablehlo.convert %23 : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf32>
    %25 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %26 = stablehlo.reshape %25 : (tensor<1x16384xf32>) -> tensor<1x16384x1xf32>
    %27 = stablehlo.convert %cst_79 : (tensor<1xi64>) -> tensor<1xf32>
    %28 = stablehlo.reshape %27 : (tensor<1xf32>) -> tensor<f32>
    %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %30 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<f32>) -> tensor<1x16384x1xf32>
    %31 = stablehlo.divide %29, %30 : tensor<1x16384x1xf32>
    %32 = stablehlo.convert %cst_80 : (tensor<1xf64>) -> tensor<1xf32>
    %33 = stablehlo.reshape %32 : (tensor<1xf32>) -> tensor<f32>
    %34 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %35 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<1x16384x1xf32>
    %36 = stablehlo.add %34, %35 : tensor<1x16384x1xf32>
    %37 = stablehlo.rsqrt %36 : tensor<1x16384x1xf32>
    %38 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %39 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %40 = stablehlo.subtract %38, %39 : tensor<1x16384x32xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %42 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x16384x32xf32>
    %44 = stablehlo.convert %arg3 : (tensor<32xbf16>) -> tensor<32xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1x16384x32xf32>
    %48 = stablehlo.convert %arg4 : (tensor<32xbf16>) -> tensor<32xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x16384x32xf32>
    %52 = stablehlo.convert %51 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xbf16>
    %53 = stablehlo.convert %52 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xf32>
    %54 = stablehlo.convert %53 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf64>
    %55 = stablehlo.reduce(%54 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %56 = stablehlo.reshape %55 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %58 = stablehlo.divide %57, %14 : tensor<1x16384x1xf64>
    %59 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<1x16384x32xf64>) -> tensor<1x16384x32xf64>
    %60 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x32xf64>
    %61 = stablehlo.subtract %59, %60 : tensor<1x16384x32xf64>
    %62 = stablehlo.multiply %61, %61 : tensor<1x16384x32xf64>
    %63 = stablehlo.reduce(%62 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %64 = stablehlo.reshape %63 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %66 = stablehlo.divide %65, %14 : tensor<1x16384x1xf64>
    %67 = stablehlo.convert %66 : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf32>
    %68 = stablehlo.reduce(%53 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %69 = stablehlo.reshape %68 : (tensor<1x16384xf32>) -> tensor<1x16384x1xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %71 = stablehlo.divide %70, %30 : tensor<1x16384x1xf32>
    %72 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %73 = stablehlo.add %72, %35 : tensor<1x16384x1xf32>
    %74 = stablehlo.rsqrt %73 : tensor<1x16384x1xf32>
    %75 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %76 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %77 = stablehlo.subtract %75, %76 : tensor<1x16384x32xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %79 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %80 = stablehlo.multiply %78, %79 : tensor<1x16384x32xf32>
    %81 = stablehlo.convert %arg5 : (tensor<32xbf16>) -> tensor<32xf32>
    %82 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %83 = stablehlo.broadcast_in_dim %81, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %84 = stablehlo.multiply %82, %83 : tensor<1x16384x32xf32>
    %85 = stablehlo.convert %arg6 : (tensor<32xbf16>) -> tensor<32xf32>
    %86 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %87 = stablehlo.broadcast_in_dim %85, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x16384x32xf32>
    %89 = stablehlo.convert %88 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xbf16>
    %90 = stablehlo.reshape %89 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %91 = stablehlo.convert %90 : (tensor<16384x32xbf16>) -> tensor<16384x32xf32>
    %92 = stablehlo.dot_general %91, %arg112, contracting_dims = [1] x [0] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %93 = stablehlo.convert %cst_81 : (tensor<1xi64>) -> tensor<1xf32>
    %94 = stablehlo.reshape %93 : (tensor<1xf32>) -> tensor<f32>
    %95 = stablehlo.broadcast_in_dim %92, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %96 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<16384x32xf32>
    %97 = stablehlo.multiply %95, %96 : tensor<16384x32xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %99 = stablehlo.broadcast_in_dim %arg113, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %100 = stablehlo.add %98, %99 : tensor<16384x32xf32>
    %101 = stablehlo.convert %100 : (tensor<16384x32xf32>) -> tensor<16384x32xbf16>
    %102 = stablehlo.reshape %101 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %103 = stablehlo.reshape %102 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1x32xbf16>
    %104 = stablehlo.transpose %103, dims = [0, 2, 1, 3] : (tensor<1x16384x1x32xbf16>) -> tensor<1x1x16384x32xbf16>
    %105 = stablehlo.transpose %89, dims = [0, 2, 1] : (tensor<1x16384x32xbf16>) -> tensor<1x32x16384xbf16>
    %106 = stablehlo.reshape %105 : (tensor<1x32x16384xbf16>) -> tensor<1x32x128x128xbf16>
    %107 = stablehlo.convolution(%106, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x128x128xbf16>, tensor<32x32x8x8xbf16>) -> tensor<1x32x16x16xbf16>
    %108 = stablehlo.reshape %arg8 : (tensor<32xbf16>) -> tensor<32x1x1xbf16>
    %109 = stablehlo.broadcast_in_dim %107, dims = [0, 1, 2, 3] : (tensor<1x32x16x16xbf16>) -> tensor<1x32x16x16xbf16>
    %110 = stablehlo.broadcast_in_dim %108, dims = [1, 2, 3] : (tensor<32x1x1xbf16>) -> tensor<1x32x16x16xbf16>
    %111 = stablehlo.add %109, %110 : tensor<1x32x16x16xbf16>
    %112 = stablehlo.reshape %111 : (tensor<1x32x16x16xbf16>) -> tensor<1x32x256xbf16>
    %113 = stablehlo.transpose %112, dims = [0, 2, 1] : (tensor<1x32x256xbf16>) -> tensor<1x256x32xbf16>
    %114 = stablehlo.convert %113 : (tensor<1x256x32xbf16>) -> tensor<1x256x32xf32>
    %115 = stablehlo.convert %114 : (tensor<1x256x32xf32>) -> tensor<1x256x32xf64>
    %116 = stablehlo.reduce(%115 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf64>, tensor<f64>) -> tensor<1x256xf64>
    %117 = stablehlo.reshape %116 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %119 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f64>) -> tensor<1x256x1xf64>
    %120 = stablehlo.divide %118, %119 : tensor<1x256x1xf64>
    %121 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2] : (tensor<1x256x32xf64>) -> tensor<1x256x32xf64>
    %122 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x32xf64>
    %123 = stablehlo.subtract %121, %122 : tensor<1x256x32xf64>
    %124 = stablehlo.multiply %123, %123 : tensor<1x256x32xf64>
    %125 = stablehlo.reduce(%124 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf64>, tensor<f64>) -> tensor<1x256xf64>
    %126 = stablehlo.reshape %125 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %128 = stablehlo.divide %127, %119 : tensor<1x256x1xf64>
    %129 = stablehlo.convert %128 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %130 = stablehlo.reduce(%114 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf32>, tensor<f32>) -> tensor<1x256xf32>
    %131 = stablehlo.reshape %130 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %133 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %134 = stablehlo.divide %132, %133 : tensor<1x256x1xf32>
    %135 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %136 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %137 = stablehlo.add %135, %136 : tensor<1x256x1xf32>
    %138 = stablehlo.rsqrt %137 : tensor<1x256x1xf32>
    %139 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %140 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x32xf32>
    %141 = stablehlo.subtract %139, %140 : tensor<1x256x32xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %143 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x32xf32>
    %144 = stablehlo.multiply %142, %143 : tensor<1x256x32xf32>
    %145 = stablehlo.convert %arg9 : (tensor<32xbf16>) -> tensor<32xf32>
    %146 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %147 = stablehlo.broadcast_in_dim %145, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
    %148 = stablehlo.multiply %146, %147 : tensor<1x256x32xf32>
    %149 = stablehlo.convert %arg10 : (tensor<32xbf16>) -> tensor<32xf32>
    %150 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %151 = stablehlo.broadcast_in_dim %149, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
    %152 = stablehlo.add %150, %151 : tensor<1x256x32xf32>
    %153 = stablehlo.convert %152 : (tensor<1x256x32xf32>) -> tensor<1x256x32xbf16>
    %154 = stablehlo.reshape %153 : (tensor<1x256x32xbf16>) -> tensor<256x32xbf16>
    %155 = stablehlo.convert %154 : (tensor<256x32xbf16>) -> tensor<256x32xf32>
    %156 = stablehlo.dot_general %155, %arg114, contracting_dims = [1] x [0] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %158 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<256x32xf32>
    %159 = stablehlo.multiply %157, %158 : tensor<256x32xf32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %161 = stablehlo.broadcast_in_dim %arg115, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %162 = stablehlo.add %160, %161 : tensor<256x32xf32>
    %163 = stablehlo.convert %162 : (tensor<256x32xf32>) -> tensor<256x32xbf16>
    %164 = stablehlo.reshape %163 : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %165 = stablehlo.reshape %164 : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %166 = stablehlo.transpose %165, dims = [0, 2, 1, 3] : (tensor<1x256x1x32xbf16>) -> tensor<1x1x256x32xbf16>
    %167 = stablehlo.dot_general %155, %arg116, contracting_dims = [1] x [0] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %169 = stablehlo.multiply %168, %158 : tensor<256x32xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %171 = stablehlo.broadcast_in_dim %arg117, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %172 = stablehlo.add %170, %171 : tensor<256x32xf32>
    %173 = stablehlo.convert %172 : (tensor<256x32xf32>) -> tensor<256x32xbf16>
    %174 = stablehlo.reshape %173 : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %175 = stablehlo.reshape %174 : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %176 = stablehlo.transpose %175, dims = [0, 2, 1, 3] : (tensor<1x256x1x32xbf16>) -> tensor<1x1x256x32xbf16>
    %177 = stablehlo.transpose %166, dims = [0, 1, 3, 2] : (tensor<1x1x256x32xbf16>) -> tensor<1x1x32x256xbf16>
    %178 = stablehlo.reshape %104 : (tensor<1x1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %179 = stablehlo.reshape %177 : (tensor<1x1x32x256xbf16>) -> tensor<1x32x256xbf16>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<1x32x256xbf16>) -> tensor<1x32x256xbf16>
    %181 = stablehlo.dot_general %178, %180, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x16384x32xbf16>, tensor<1x32x256xbf16>) -> tensor<1x16384x256xbf16>
    %182 = stablehlo.reshape %181 : (tensor<1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %183 = stablehlo.convert %cst_82 : (tensor<1xf64>) -> tensor<1xbf16>
    %184 = stablehlo.reshape %183 : (tensor<1xbf16>) -> tensor<bf16>
    %185 = stablehlo.broadcast_in_dim %182, dims = [0, 1, 2, 3] : (tensor<1x1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %186 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<bf16>) -> tensor<1x1x16384x256xbf16>
    %187 = stablehlo.divide %185, %186 : tensor<1x1x16384x256xbf16>
    %188 = stablehlo.convert %187 : (tensor<1x1x16384x256xbf16>) -> tensor<1x1x16384x256xf32>
    %189 = stablehlo.reduce(%188 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %190 = stablehlo.reshape %189 : (tensor<1x1x16384xf32>) -> tensor<1x1x16384x1xf32>
    %191 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x1x16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %192 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2, 3] : (tensor<1x1x16384x1xf32>) -> tensor<1x1x16384x256xf32>
    %193 = stablehlo.subtract %191, %192 : tensor<1x1x16384x256xf32>
    %194 = stablehlo.exponential %193 : tensor<1x1x16384x256xf32>
    %195 = stablehlo.reduce(%194 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %196 = stablehlo.reshape %195 : (tensor<1x1x16384xf32>) -> tensor<1x1x16384x1xf32>
    %197 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2, 3] : (tensor<1x1x16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %198 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2, 3] : (tensor<1x1x16384x1xf32>) -> tensor<1x1x16384x256xf32>
    %199 = stablehlo.divide %197, %198 : tensor<1x1x16384x256xf32>
    %200 = stablehlo.convert %199 : (tensor<1x1x16384x256xf32>) -> tensor<1x1x16384x256xbf16>
    %201 = stablehlo.reshape %200 : (tensor<1x1x16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %202 = stablehlo.reshape %176 : (tensor<1x1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2] : (tensor<1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %204 = stablehlo.dot_general %201, %203, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x16384x256xbf16>, tensor<1x256x32xbf16>) -> tensor<1x16384x32xbf16>
    %205 = stablehlo.reshape %204 : (tensor<1x16384x32xbf16>) -> tensor<1x1x16384x32xbf16>
    %206 = stablehlo.transpose %205, dims = [0, 2, 1, 3] : (tensor<1x1x16384x32xbf16>) -> tensor<1x16384x1x32xbf16>
    %207 = stablehlo.reshape %206 : (tensor<1x16384x1x32xbf16>) -> tensor<1x16384x32xbf16>
    %208 = stablehlo.reshape %207 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %209 = stablehlo.convert %208 : (tensor<16384x32xbf16>) -> tensor<16384x32xf32>
    %210 = stablehlo.dot_general %209, %arg118, contracting_dims = [1] x [0] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %212 = stablehlo.multiply %211, %96 : tensor<16384x32xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %214 = stablehlo.broadcast_in_dim %arg119, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %215 = stablehlo.add %213, %214 : tensor<16384x32xf32>
    %216 = stablehlo.convert %215 : (tensor<16384x32xf32>) -> tensor<16384x32xbf16>
    %217 = stablehlo.reshape %216 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %218 = stablehlo.add %217, %52 : tensor<1x16384x32xbf16>
    %219 = stablehlo.convert %218 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xf32>
    %220 = stablehlo.convert %219 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf64>
    %221 = stablehlo.reduce(%220 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %222 = stablehlo.reshape %221 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %223 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %224 = stablehlo.divide %223, %14 : tensor<1x16384x1xf64>
    %225 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1x16384x32xf64>) -> tensor<1x16384x32xf64>
    %226 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x32xf64>
    %227 = stablehlo.subtract %225, %226 : tensor<1x16384x32xf64>
    %228 = stablehlo.multiply %227, %227 : tensor<1x16384x32xf64>
    %229 = stablehlo.reduce(%228 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %230 = stablehlo.reshape %229 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %232 = stablehlo.divide %231, %14 : tensor<1x16384x1xf64>
    %233 = stablehlo.convert %232 : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf32>
    %234 = stablehlo.reduce(%219 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %235 = stablehlo.reshape %234 : (tensor<1x16384xf32>) -> tensor<1x16384x1xf32>
    %236 = stablehlo.broadcast_in_dim %235, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %237 = stablehlo.divide %236, %30 : tensor<1x16384x1xf32>
    %238 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %239 = stablehlo.add %238, %35 : tensor<1x16384x1xf32>
    %240 = stablehlo.rsqrt %239 : tensor<1x16384x1xf32>
    %241 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %242 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %243 = stablehlo.subtract %241, %242 : tensor<1x16384x32xf32>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %245 = stablehlo.broadcast_in_dim %240, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %246 = stablehlo.multiply %244, %245 : tensor<1x16384x32xf32>
    %247 = stablehlo.convert %arg11 : (tensor<32xbf16>) -> tensor<32xf32>
    %248 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %249 = stablehlo.broadcast_in_dim %247, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %250 = stablehlo.multiply %248, %249 : tensor<1x16384x32xf32>
    %251 = stablehlo.convert %arg12 : (tensor<32xbf16>) -> tensor<32xf32>
    %252 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %253 = stablehlo.broadcast_in_dim %251, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %254 = stablehlo.add %252, %253 : tensor<1x16384x32xf32>
    %255 = stablehlo.convert %254 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xbf16>
    %256 = stablehlo.reshape %255 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %257 = stablehlo.convert %256 : (tensor<16384x32xbf16>) -> tensor<16384x32xf32>
    %258 = stablehlo.dot_general %257, %arg120, contracting_dims = [1] x [0] : (tensor<16384x32xf32>, tensor<32x128xf32>) -> tensor<16384x128xf32>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1] : (tensor<16384x128xf32>) -> tensor<16384x128xf32>
    %260 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<16384x128xf32>
    %261 = stablehlo.multiply %259, %260 : tensor<16384x128xf32>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1] : (tensor<16384x128xf32>) -> tensor<16384x128xf32>
    %263 = stablehlo.broadcast_in_dim %arg121, dims = [1] : (tensor<128xf32>) -> tensor<16384x128xf32>
    %264 = stablehlo.add %262, %263 : tensor<16384x128xf32>
    %265 = stablehlo.convert %264 : (tensor<16384x128xf32>) -> tensor<16384x128xbf16>
    %266 = stablehlo.reshape %265 : (tensor<16384x128xbf16>) -> tensor<1x16384x128xbf16>
    %267 = stablehlo.transpose %266, dims = [0, 2, 1] : (tensor<1x16384x128xbf16>) -> tensor<1x128x16384xbf16>
    %268 = stablehlo.reshape %267 : (tensor<1x128x16384xbf16>) -> tensor<1x128x128x128xbf16>
    %269 = stablehlo.convolution(%268, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x128x128xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x128x128xbf16>
    %270 = stablehlo.reshape %arg14 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %271 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %272 = stablehlo.broadcast_in_dim %270, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x128x128xbf16>
    %273 = stablehlo.add %271, %272 : tensor<1x128x128x128xbf16>
    %274 = stablehlo.reshape %273 : (tensor<1x128x128x128xbf16>) -> tensor<1x128x16384xbf16>
    %275 = stablehlo.transpose %274, dims = [0, 2, 1] : (tensor<1x128x16384xbf16>) -> tensor<1x16384x128xbf16>
    %276 = stablehlo.multiply %275, %cst_4 : tensor<1x16384x128xbf16>
    %277 = stablehlo.rsqrt %cst_3 : tensor<1x16384x128xbf16>
    %278 = stablehlo.multiply %275, %277 : tensor<1x16384x128xbf16>
    %279 = stablehlo.convert %278 : (tensor<1x16384x128xbf16>) -> tensor<1x16384x128xf32>
    %280 = stablehlo.clamp %cst_5, %279, %cst_6 : tensor<1x16384x128xf32>
    %281 = stablehlo.multiply %280, %280 : tensor<1x16384x128xf32>
    %282 = stablehlo.multiply %cst_7, %281 : tensor<1x16384x128xf32>
    %283 = stablehlo.add %282, %cst_8 : tensor<1x16384x128xf32>
    %284 = stablehlo.multiply %283, %281 : tensor<1x16384x128xf32>
    %285 = stablehlo.add %284, %cst_9 : tensor<1x16384x128xf32>
    %286 = stablehlo.multiply %285, %281 : tensor<1x16384x128xf32>
    %287 = stablehlo.add %286, %cst_10 : tensor<1x16384x128xf32>
    %288 = stablehlo.multiply %287, %281 : tensor<1x16384x128xf32>
    %289 = stablehlo.add %288, %cst_11 : tensor<1x16384x128xf32>
    %290 = stablehlo.multiply %289, %281 : tensor<1x16384x128xf32>
    %291 = stablehlo.add %290, %cst_12 : tensor<1x16384x128xf32>
    %292 = stablehlo.multiply %291, %281 : tensor<1x16384x128xf32>
    %293 = stablehlo.add %292, %cst_13 : tensor<1x16384x128xf32>
    %294 = stablehlo.multiply %cst_14, %281 : tensor<1x16384x128xf32>
    %295 = stablehlo.add %294, %cst_15 : tensor<1x16384x128xf32>
    %296 = stablehlo.multiply %295, %281 : tensor<1x16384x128xf32>
    %297 = stablehlo.add %296, %cst_16 : tensor<1x16384x128xf32>
    %298 = stablehlo.multiply %297, %281 : tensor<1x16384x128xf32>
    %299 = stablehlo.add %298, %cst_17 : tensor<1x16384x128xf32>
    %300 = stablehlo.multiply %299, %281 : tensor<1x16384x128xf32>
    %301 = stablehlo.add %300, %cst_18 : tensor<1x16384x128xf32>
    %302 = stablehlo.multiply %280, %293 : tensor<1x16384x128xf32>
    %303 = stablehlo.divide %302, %301 : tensor<1x16384x128xf32>
    %304 = stablehlo.clamp %cst_19, %303, %cst_20 : tensor<1x16384x128xf32>
    %305 = stablehlo.convert %304 : (tensor<1x16384x128xf32>) -> tensor<1x16384x128xbf16>
    %306 = stablehlo.add %305, %cst_2 : tensor<1x16384x128xbf16>
    %307 = stablehlo.multiply %306, %276 : tensor<1x16384x128xbf16>
    %308 = stablehlo.reshape %307 : (tensor<1x16384x128xbf16>) -> tensor<16384x128xbf16>
    %309 = stablehlo.dot_general %308, %arg122, contracting_dims = [1] x [0] : (tensor<16384x128xbf16>, tensor<128x32xbf16>) -> tensor<16384x32xbf16>
    %310 = stablehlo.reshape %309 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2] : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %312 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %313 = stablehlo.add %311, %312 : tensor<1x16384x32xbf16>
    %314 = stablehlo.reshape %313 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %315 = stablehlo.reshape %314 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %316 = stablehlo.add %315, %218 : tensor<1x16384x32xbf16>
    %317 = stablehlo.convert %316 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xf32>
    %318 = stablehlo.convert %317 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf64>
    %319 = stablehlo.reduce(%318 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %320 = stablehlo.reshape %319 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %322 = stablehlo.divide %321, %14 : tensor<1x16384x1xf64>
    %323 = stablehlo.broadcast_in_dim %318, dims = [0, 1, 2] : (tensor<1x16384x32xf64>) -> tensor<1x16384x32xf64>
    %324 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x32xf64>
    %325 = stablehlo.subtract %323, %324 : tensor<1x16384x32xf64>
    %326 = stablehlo.multiply %325, %325 : tensor<1x16384x32xf64>
    %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %328 = stablehlo.reshape %327 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %330 = stablehlo.divide %329, %14 : tensor<1x16384x1xf64>
    %331 = stablehlo.convert %330 : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf32>
    %332 = stablehlo.reduce(%317 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %333 = stablehlo.reshape %332 : (tensor<1x16384xf32>) -> tensor<1x16384x1xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %335 = stablehlo.divide %334, %30 : tensor<1x16384x1xf32>
    %336 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %337 = stablehlo.add %336, %35 : tensor<1x16384x1xf32>
    %338 = stablehlo.rsqrt %337 : tensor<1x16384x1xf32>
    %339 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %340 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %341 = stablehlo.subtract %339, %340 : tensor<1x16384x32xf32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %343 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %344 = stablehlo.multiply %342, %343 : tensor<1x16384x32xf32>
    %345 = stablehlo.convert %arg16 : (tensor<32xbf16>) -> tensor<32xf32>
    %346 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %347 = stablehlo.broadcast_in_dim %345, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %348 = stablehlo.multiply %346, %347 : tensor<1x16384x32xf32>
    %349 = stablehlo.convert %arg17 : (tensor<32xbf16>) -> tensor<32xf32>
    %350 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %351 = stablehlo.broadcast_in_dim %349, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %352 = stablehlo.add %350, %351 : tensor<1x16384x32xf32>
    %353 = stablehlo.convert %352 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xbf16>
    %354 = stablehlo.reshape %353 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %355 = stablehlo.convert %354 : (tensor<16384x32xbf16>) -> tensor<16384x32xf32>
    %356 = stablehlo.dot_general %355, %arg123, contracting_dims = [1] x [0] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %358 = stablehlo.multiply %357, %96 : tensor<16384x32xf32>
    %359 = stablehlo.broadcast_in_dim %358, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %360 = stablehlo.broadcast_in_dim %arg124, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %361 = stablehlo.add %359, %360 : tensor<16384x32xf32>
    %362 = stablehlo.convert %361 : (tensor<16384x32xf32>) -> tensor<16384x32xbf16>
    %363 = stablehlo.reshape %362 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %364 = stablehlo.reshape %363 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1x32xbf16>
    %365 = stablehlo.transpose %364, dims = [0, 2, 1, 3] : (tensor<1x16384x1x32xbf16>) -> tensor<1x1x16384x32xbf16>
    %366 = stablehlo.transpose %353, dims = [0, 2, 1] : (tensor<1x16384x32xbf16>) -> tensor<1x32x16384xbf16>
    %367 = stablehlo.reshape %366 : (tensor<1x32x16384xbf16>) -> tensor<1x32x128x128xbf16>
    %368 = stablehlo.convolution(%367, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x128x128xbf16>, tensor<32x32x8x8xbf16>) -> tensor<1x32x16x16xbf16>
    %369 = stablehlo.reshape %arg19 : (tensor<32xbf16>) -> tensor<32x1x1xbf16>
    %370 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2, 3] : (tensor<1x32x16x16xbf16>) -> tensor<1x32x16x16xbf16>
    %371 = stablehlo.broadcast_in_dim %369, dims = [1, 2, 3] : (tensor<32x1x1xbf16>) -> tensor<1x32x16x16xbf16>
    %372 = stablehlo.add %370, %371 : tensor<1x32x16x16xbf16>
    %373 = stablehlo.reshape %372 : (tensor<1x32x16x16xbf16>) -> tensor<1x32x256xbf16>
    %374 = stablehlo.transpose %373, dims = [0, 2, 1] : (tensor<1x32x256xbf16>) -> tensor<1x256x32xbf16>
    %375 = stablehlo.convert %374 : (tensor<1x256x32xbf16>) -> tensor<1x256x32xf32>
    %376 = stablehlo.convert %375 : (tensor<1x256x32xf32>) -> tensor<1x256x32xf64>
    %377 = stablehlo.reduce(%376 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf64>, tensor<f64>) -> tensor<1x256xf64>
    %378 = stablehlo.reshape %377 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %379 = stablehlo.broadcast_in_dim %378, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %380 = stablehlo.divide %379, %119 : tensor<1x256x1xf64>
    %381 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<1x256x32xf64>) -> tensor<1x256x32xf64>
    %382 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x32xf64>
    %383 = stablehlo.subtract %381, %382 : tensor<1x256x32xf64>
    %384 = stablehlo.multiply %383, %383 : tensor<1x256x32xf64>
    %385 = stablehlo.reduce(%384 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf64>, tensor<f64>) -> tensor<1x256xf64>
    %386 = stablehlo.reshape %385 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %388 = stablehlo.divide %387, %119 : tensor<1x256x1xf64>
    %389 = stablehlo.convert %388 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %390 = stablehlo.reduce(%375 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf32>, tensor<f32>) -> tensor<1x256xf32>
    %391 = stablehlo.reshape %390 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %393 = stablehlo.divide %392, %133 : tensor<1x256x1xf32>
    %394 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %395 = stablehlo.add %394, %136 : tensor<1x256x1xf32>
    %396 = stablehlo.rsqrt %395 : tensor<1x256x1xf32>
    %397 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %398 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x32xf32>
    %399 = stablehlo.subtract %397, %398 : tensor<1x256x32xf32>
    %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %401 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x32xf32>
    %402 = stablehlo.multiply %400, %401 : tensor<1x256x32xf32>
    %403 = stablehlo.convert %arg20 : (tensor<32xbf16>) -> tensor<32xf32>
    %404 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %405 = stablehlo.broadcast_in_dim %403, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
    %406 = stablehlo.multiply %404, %405 : tensor<1x256x32xf32>
    %407 = stablehlo.convert %arg21 : (tensor<32xbf16>) -> tensor<32xf32>
    %408 = stablehlo.broadcast_in_dim %406, dims = [0, 1, 2] : (tensor<1x256x32xf32>) -> tensor<1x256x32xf32>
    %409 = stablehlo.broadcast_in_dim %407, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
    %410 = stablehlo.add %408, %409 : tensor<1x256x32xf32>
    %411 = stablehlo.convert %410 : (tensor<1x256x32xf32>) -> tensor<1x256x32xbf16>
    %412 = stablehlo.reshape %411 : (tensor<1x256x32xbf16>) -> tensor<256x32xbf16>
    %413 = stablehlo.convert %412 : (tensor<256x32xbf16>) -> tensor<256x32xf32>
    %414 = stablehlo.dot_general %413, %arg125, contracting_dims = [1] x [0] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %416 = stablehlo.multiply %415, %158 : tensor<256x32xf32>
    %417 = stablehlo.broadcast_in_dim %416, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %418 = stablehlo.broadcast_in_dim %arg126, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %419 = stablehlo.add %417, %418 : tensor<256x32xf32>
    %420 = stablehlo.convert %419 : (tensor<256x32xf32>) -> tensor<256x32xbf16>
    %421 = stablehlo.reshape %420 : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %422 = stablehlo.reshape %421 : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] : (tensor<1x256x1x32xbf16>) -> tensor<1x1x256x32xbf16>
    %424 = stablehlo.dot_general %413, %arg127, contracting_dims = [1] x [0] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %426 = stablehlo.multiply %425, %158 : tensor<256x32xf32>
    %427 = stablehlo.broadcast_in_dim %426, dims = [0, 1] : (tensor<256x32xf32>) -> tensor<256x32xf32>
    %428 = stablehlo.broadcast_in_dim %arg128, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %429 = stablehlo.add %427, %428 : tensor<256x32xf32>
    %430 = stablehlo.convert %429 : (tensor<256x32xf32>) -> tensor<256x32xbf16>
    %431 = stablehlo.reshape %430 : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %432 = stablehlo.reshape %431 : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %433 = stablehlo.transpose %432, dims = [0, 2, 1, 3] : (tensor<1x256x1x32xbf16>) -> tensor<1x1x256x32xbf16>
    %434 = stablehlo.transpose %423, dims = [0, 1, 3, 2] : (tensor<1x1x256x32xbf16>) -> tensor<1x1x32x256xbf16>
    %435 = stablehlo.reshape %365 : (tensor<1x1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %436 = stablehlo.reshape %434 : (tensor<1x1x32x256xbf16>) -> tensor<1x32x256xbf16>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2] : (tensor<1x32x256xbf16>) -> tensor<1x32x256xbf16>
    %438 = stablehlo.dot_general %435, %437, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x16384x32xbf16>, tensor<1x32x256xbf16>) -> tensor<1x16384x256xbf16>
    %439 = stablehlo.reshape %438 : (tensor<1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %440 = stablehlo.broadcast_in_dim %439, dims = [0, 1, 2, 3] : (tensor<1x1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %441 = stablehlo.divide %440, %186 : tensor<1x1x16384x256xbf16>
    %442 = stablehlo.convert %441 : (tensor<1x1x16384x256xbf16>) -> tensor<1x1x16384x256xf32>
    %443 = stablehlo.reduce(%442 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %444 = stablehlo.reshape %443 : (tensor<1x1x16384xf32>) -> tensor<1x1x16384x1xf32>
    %445 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x1x16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %446 = stablehlo.broadcast_in_dim %444, dims = [0, 1, 2, 3] : (tensor<1x1x16384x1xf32>) -> tensor<1x1x16384x256xf32>
    %447 = stablehlo.subtract %445, %446 : tensor<1x1x16384x256xf32>
    %448 = stablehlo.exponential %447 : tensor<1x1x16384x256xf32>
    %449 = stablehlo.reduce(%448 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %450 = stablehlo.reshape %449 : (tensor<1x1x16384xf32>) -> tensor<1x1x16384x1xf32>
    %451 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x1x16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %452 = stablehlo.broadcast_in_dim %450, dims = [0, 1, 2, 3] : (tensor<1x1x16384x1xf32>) -> tensor<1x1x16384x256xf32>
    %453 = stablehlo.divide %451, %452 : tensor<1x1x16384x256xf32>
    %454 = stablehlo.convert %453 : (tensor<1x1x16384x256xf32>) -> tensor<1x1x16384x256xbf16>
    %455 = stablehlo.reshape %454 : (tensor<1x1x16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %456 = stablehlo.reshape %433 : (tensor<1x1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2] : (tensor<1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %458 = stablehlo.dot_general %455, %457, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x16384x256xbf16>, tensor<1x256x32xbf16>) -> tensor<1x16384x32xbf16>
    %459 = stablehlo.reshape %458 : (tensor<1x16384x32xbf16>) -> tensor<1x1x16384x32xbf16>
    %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] : (tensor<1x1x16384x32xbf16>) -> tensor<1x16384x1x32xbf16>
    %461 = stablehlo.reshape %460 : (tensor<1x16384x1x32xbf16>) -> tensor<1x16384x32xbf16>
    %462 = stablehlo.reshape %461 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %463 = stablehlo.convert %462 : (tensor<16384x32xbf16>) -> tensor<16384x32xf32>
    %464 = stablehlo.dot_general %463, %arg129, contracting_dims = [1] x [0] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %465 = stablehlo.broadcast_in_dim %464, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %466 = stablehlo.multiply %465, %96 : tensor<16384x32xf32>
    %467 = stablehlo.broadcast_in_dim %466, dims = [0, 1] : (tensor<16384x32xf32>) -> tensor<16384x32xf32>
    %468 = stablehlo.broadcast_in_dim %arg130, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %469 = stablehlo.add %467, %468 : tensor<16384x32xf32>
    %470 = stablehlo.convert %469 : (tensor<16384x32xf32>) -> tensor<16384x32xbf16>
    %471 = stablehlo.reshape %470 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %472 = stablehlo.add %471, %316 : tensor<1x16384x32xbf16>
    %473 = stablehlo.convert %472 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xf32>
    %474 = stablehlo.convert %473 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf64>
    %475 = stablehlo.reduce(%474 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %476 = stablehlo.reshape %475 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %478 = stablehlo.divide %477, %14 : tensor<1x16384x1xf64>
    %479 = stablehlo.broadcast_in_dim %474, dims = [0, 1, 2] : (tensor<1x16384x32xf64>) -> tensor<1x16384x32xf64>
    %480 = stablehlo.broadcast_in_dim %478, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x32xf64>
    %481 = stablehlo.subtract %479, %480 : tensor<1x16384x32xf64>
    %482 = stablehlo.multiply %481, %481 : tensor<1x16384x32xf64>
    %483 = stablehlo.reduce(%482 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %484 = stablehlo.reshape %483 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %486 = stablehlo.divide %485, %14 : tensor<1x16384x1xf64>
    %487 = stablehlo.convert %486 : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf32>
    %488 = stablehlo.reduce(%473 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %489 = stablehlo.reshape %488 : (tensor<1x16384xf32>) -> tensor<1x16384x1xf32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %491 = stablehlo.divide %490, %30 : tensor<1x16384x1xf32>
    %492 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %493 = stablehlo.add %492, %35 : tensor<1x16384x1xf32>
    %494 = stablehlo.rsqrt %493 : tensor<1x16384x1xf32>
    %495 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %496 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %497 = stablehlo.subtract %495, %496 : tensor<1x16384x32xf32>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %499 = stablehlo.broadcast_in_dim %494, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %500 = stablehlo.multiply %498, %499 : tensor<1x16384x32xf32>
    %501 = stablehlo.convert %arg22 : (tensor<32xbf16>) -> tensor<32xf32>
    %502 = stablehlo.broadcast_in_dim %500, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %503 = stablehlo.broadcast_in_dim %501, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %504 = stablehlo.multiply %502, %503 : tensor<1x16384x32xf32>
    %505 = stablehlo.convert %arg23 : (tensor<32xbf16>) -> tensor<32xf32>
    %506 = stablehlo.broadcast_in_dim %504, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %507 = stablehlo.broadcast_in_dim %505, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %508 = stablehlo.add %506, %507 : tensor<1x16384x32xf32>
    %509 = stablehlo.convert %508 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xbf16>
    %510 = stablehlo.reshape %509 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %511 = stablehlo.convert %510 : (tensor<16384x32xbf16>) -> tensor<16384x32xf32>
    %512 = stablehlo.dot_general %511, %arg131, contracting_dims = [1] x [0] : (tensor<16384x32xf32>, tensor<32x128xf32>) -> tensor<16384x128xf32>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1] : (tensor<16384x128xf32>) -> tensor<16384x128xf32>
    %514 = stablehlo.multiply %513, %260 : tensor<16384x128xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1] : (tensor<16384x128xf32>) -> tensor<16384x128xf32>
    %516 = stablehlo.broadcast_in_dim %arg132, dims = [1] : (tensor<128xf32>) -> tensor<16384x128xf32>
    %517 = stablehlo.add %515, %516 : tensor<16384x128xf32>
    %518 = stablehlo.convert %517 : (tensor<16384x128xf32>) -> tensor<16384x128xbf16>
    %519 = stablehlo.reshape %518 : (tensor<16384x128xbf16>) -> tensor<1x16384x128xbf16>
    %520 = stablehlo.transpose %519, dims = [0, 2, 1] : (tensor<1x16384x128xbf16>) -> tensor<1x128x16384xbf16>
    %521 = stablehlo.reshape %520 : (tensor<1x128x16384xbf16>) -> tensor<1x128x128x128xbf16>
    %522 = stablehlo.convolution(%521, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x128x128xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x128x128xbf16>
    %523 = stablehlo.reshape %arg25 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %524 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %525 = stablehlo.broadcast_in_dim %523, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x128x128xbf16>
    %526 = stablehlo.add %524, %525 : tensor<1x128x128x128xbf16>
    %527 = stablehlo.reshape %526 : (tensor<1x128x128x128xbf16>) -> tensor<1x128x16384xbf16>
    %528 = stablehlo.transpose %527, dims = [0, 2, 1] : (tensor<1x128x16384xbf16>) -> tensor<1x16384x128xbf16>
    %529 = stablehlo.multiply %528, %cst_4 : tensor<1x16384x128xbf16>
    %530 = stablehlo.multiply %528, %277 : tensor<1x16384x128xbf16>
    %531 = stablehlo.convert %530 : (tensor<1x16384x128xbf16>) -> tensor<1x16384x128xf32>
    %532 = stablehlo.clamp %cst_5, %531, %cst_6 : tensor<1x16384x128xf32>
    %533 = stablehlo.multiply %532, %532 : tensor<1x16384x128xf32>
    %534 = stablehlo.multiply %cst_7, %533 : tensor<1x16384x128xf32>
    %535 = stablehlo.add %534, %cst_8 : tensor<1x16384x128xf32>
    %536 = stablehlo.multiply %535, %533 : tensor<1x16384x128xf32>
    %537 = stablehlo.add %536, %cst_9 : tensor<1x16384x128xf32>
    %538 = stablehlo.multiply %537, %533 : tensor<1x16384x128xf32>
    %539 = stablehlo.add %538, %cst_10 : tensor<1x16384x128xf32>
    %540 = stablehlo.multiply %539, %533 : tensor<1x16384x128xf32>
    %541 = stablehlo.add %540, %cst_11 : tensor<1x16384x128xf32>
    %542 = stablehlo.multiply %541, %533 : tensor<1x16384x128xf32>
    %543 = stablehlo.add %542, %cst_12 : tensor<1x16384x128xf32>
    %544 = stablehlo.multiply %543, %533 : tensor<1x16384x128xf32>
    %545 = stablehlo.add %544, %cst_13 : tensor<1x16384x128xf32>
    %546 = stablehlo.multiply %cst_14, %533 : tensor<1x16384x128xf32>
    %547 = stablehlo.add %546, %cst_15 : tensor<1x16384x128xf32>
    %548 = stablehlo.multiply %547, %533 : tensor<1x16384x128xf32>
    %549 = stablehlo.add %548, %cst_16 : tensor<1x16384x128xf32>
    %550 = stablehlo.multiply %549, %533 : tensor<1x16384x128xf32>
    %551 = stablehlo.add %550, %cst_17 : tensor<1x16384x128xf32>
    %552 = stablehlo.multiply %551, %533 : tensor<1x16384x128xf32>
    %553 = stablehlo.add %552, %cst_18 : tensor<1x16384x128xf32>
    %554 = stablehlo.multiply %532, %545 : tensor<1x16384x128xf32>
    %555 = stablehlo.divide %554, %553 : tensor<1x16384x128xf32>
    %556 = stablehlo.clamp %cst_19, %555, %cst_20 : tensor<1x16384x128xf32>
    %557 = stablehlo.convert %556 : (tensor<1x16384x128xf32>) -> tensor<1x16384x128xbf16>
    %558 = stablehlo.add %557, %cst_2 : tensor<1x16384x128xbf16>
    %559 = stablehlo.multiply %558, %529 : tensor<1x16384x128xbf16>
    %560 = stablehlo.reshape %559 : (tensor<1x16384x128xbf16>) -> tensor<16384x128xbf16>
    %561 = stablehlo.dot_general %560, %arg133, contracting_dims = [1] x [0] : (tensor<16384x128xbf16>, tensor<128x32xbf16>) -> tensor<16384x32xbf16>
    %562 = stablehlo.reshape %561 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %563 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2] : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %564 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %565 = stablehlo.add %563, %564 : tensor<1x16384x32xbf16>
    %566 = stablehlo.reshape %565 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %567 = stablehlo.reshape %566 : (tensor<16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %568 = stablehlo.add %567, %472 : tensor<1x16384x32xbf16>
    %569 = stablehlo.convert %568 : (tensor<1x16384x32xbf16>) -> tensor<1x16384x32xf32>
    %570 = stablehlo.convert %569 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf64>
    %571 = stablehlo.reduce(%570 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %572 = stablehlo.reshape %571 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %574 = stablehlo.divide %573, %14 : tensor<1x16384x1xf64>
    %575 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2] : (tensor<1x16384x32xf64>) -> tensor<1x16384x32xf64>
    %576 = stablehlo.broadcast_in_dim %574, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x32xf64>
    %577 = stablehlo.subtract %575, %576 : tensor<1x16384x32xf64>
    %578 = stablehlo.multiply %577, %577 : tensor<1x16384x32xf64>
    %579 = stablehlo.reduce(%578 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf64>, tensor<f64>) -> tensor<1x16384xf64>
    %580 = stablehlo.reshape %579 : (tensor<1x16384xf64>) -> tensor<1x16384x1xf64>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2] : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf64>
    %582 = stablehlo.divide %581, %14 : tensor<1x16384x1xf64>
    %583 = stablehlo.convert %582 : (tensor<1x16384x1xf64>) -> tensor<1x16384x1xf32>
    %584 = stablehlo.reduce(%569 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %585 = stablehlo.reshape %584 : (tensor<1x16384xf32>) -> tensor<1x16384x1xf32>
    %586 = stablehlo.broadcast_in_dim %585, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %587 = stablehlo.divide %586, %30 : tensor<1x16384x1xf32>
    %588 = stablehlo.broadcast_in_dim %583, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x1xf32>
    %589 = stablehlo.add %588, %35 : tensor<1x16384x1xf32>
    %590 = stablehlo.rsqrt %589 : tensor<1x16384x1xf32>
    %591 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %592 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %593 = stablehlo.subtract %591, %592 : tensor<1x16384x32xf32>
    %594 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %595 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2] : (tensor<1x16384x1xf32>) -> tensor<1x16384x32xf32>
    %596 = stablehlo.multiply %594, %595 : tensor<1x16384x32xf32>
    %597 = stablehlo.convert %arg27 : (tensor<32xbf16>) -> tensor<32xf32>
    %598 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %599 = stablehlo.broadcast_in_dim %597, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %600 = stablehlo.multiply %598, %599 : tensor<1x16384x32xf32>
    %601 = stablehlo.convert %arg28 : (tensor<32xbf16>) -> tensor<32xf32>
    %602 = stablehlo.broadcast_in_dim %600, dims = [0, 1, 2] : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xf32>
    %603 = stablehlo.broadcast_in_dim %601, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
    %604 = stablehlo.add %602, %603 : tensor<1x16384x32xf32>
    %605 = stablehlo.convert %604 : (tensor<1x16384x32xf32>) -> tensor<1x16384x32xbf16>
    %606 = stablehlo.reshape %605 : (tensor<1x16384x32xbf16>) -> tensor<1x128x128x32xbf16>
    %607 = stablehlo.transpose %606, dims = [0, 3, 1, 2] : (tensor<1x128x128x32xbf16>) -> tensor<1x32x128x128xbf16>
    %608 = stablehlo.convolution(%607, %arg29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x128x128xbf16>, tensor<64x32x3x3xbf16>) -> tensor<1x64x64x64xbf16>
    %609 = stablehlo.reshape %arg30 : (tensor<64xbf16>) -> tensor<64x1x1xbf16>
    %610 = stablehlo.broadcast_in_dim %608, dims = [0, 1, 2, 3] : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %611 = stablehlo.broadcast_in_dim %609, dims = [1, 2, 3] : (tensor<64x1x1xbf16>) -> tensor<1x64x64x64xbf16>
    %612 = stablehlo.add %610, %611 : tensor<1x64x64x64xbf16>
    %613 = stablehlo.reshape %612 : (tensor<1x64x64x64xbf16>) -> tensor<1x64x4096xbf16>
    %614 = stablehlo.transpose %613, dims = [0, 2, 1] : (tensor<1x64x4096xbf16>) -> tensor<1x4096x64xbf16>
    %615 = stablehlo.convert %614 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xf32>
    %616 = stablehlo.convert %615 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf64>
    %617 = stablehlo.reduce(%616 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %618 = stablehlo.reshape %617 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %619 = stablehlo.convert %cst_83 : (tensor<1xi64>) -> tensor<1xf64>
    %620 = stablehlo.reshape %619 : (tensor<1xf64>) -> tensor<f64>
    %621 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %622 = stablehlo.broadcast_in_dim %620, dims = [] : (tensor<f64>) -> tensor<1x4096x1xf64>
    %623 = stablehlo.divide %621, %622 : tensor<1x4096x1xf64>
    %624 = stablehlo.broadcast_in_dim %616, dims = [0, 1, 2] : (tensor<1x4096x64xf64>) -> tensor<1x4096x64xf64>
    %625 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x64xf64>
    %626 = stablehlo.subtract %624, %625 : tensor<1x4096x64xf64>
    %627 = stablehlo.multiply %626, %626 : tensor<1x4096x64xf64>
    %628 = stablehlo.reduce(%627 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %629 = stablehlo.reshape %628 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %630 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %631 = stablehlo.divide %630, %622 : tensor<1x4096x1xf64>
    %632 = stablehlo.convert %631 : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf32>
    %633 = stablehlo.reduce(%615 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %634 = stablehlo.reshape %633 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %635 = stablehlo.convert %cst_83 : (tensor<1xi64>) -> tensor<1xf32>
    %636 = stablehlo.reshape %635 : (tensor<1xf32>) -> tensor<f32>
    %637 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %638 = stablehlo.broadcast_in_dim %636, dims = [] : (tensor<f32>) -> tensor<1x4096x1xf32>
    %639 = stablehlo.divide %637, %638 : tensor<1x4096x1xf32>
    %640 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %641 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<1x4096x1xf32>
    %642 = stablehlo.add %640, %641 : tensor<1x4096x1xf32>
    %643 = stablehlo.rsqrt %642 : tensor<1x4096x1xf32>
    %644 = stablehlo.broadcast_in_dim %615, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %645 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %646 = stablehlo.subtract %644, %645 : tensor<1x4096x64xf32>
    %647 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %648 = stablehlo.broadcast_in_dim %643, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %649 = stablehlo.multiply %647, %648 : tensor<1x4096x64xf32>
    %650 = stablehlo.convert %arg31 : (tensor<64xbf16>) -> tensor<64xf32>
    %651 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %652 = stablehlo.broadcast_in_dim %650, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %653 = stablehlo.multiply %651, %652 : tensor<1x4096x64xf32>
    %654 = stablehlo.convert %arg32 : (tensor<64xbf16>) -> tensor<64xf32>
    %655 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %656 = stablehlo.broadcast_in_dim %654, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %657 = stablehlo.add %655, %656 : tensor<1x4096x64xf32>
    %658 = stablehlo.convert %657 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xbf16>
    %659 = stablehlo.convert %658 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xf32>
    %660 = stablehlo.convert %659 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf64>
    %661 = stablehlo.reduce(%660 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %662 = stablehlo.reshape %661 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %663 = stablehlo.broadcast_in_dim %662, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %664 = stablehlo.divide %663, %622 : tensor<1x4096x1xf64>
    %665 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2] : (tensor<1x4096x64xf64>) -> tensor<1x4096x64xf64>
    %666 = stablehlo.broadcast_in_dim %664, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x64xf64>
    %667 = stablehlo.subtract %665, %666 : tensor<1x4096x64xf64>
    %668 = stablehlo.multiply %667, %667 : tensor<1x4096x64xf64>
    %669 = stablehlo.reduce(%668 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %670 = stablehlo.reshape %669 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %672 = stablehlo.divide %671, %622 : tensor<1x4096x1xf64>
    %673 = stablehlo.convert %672 : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf32>
    %674 = stablehlo.reduce(%659 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %675 = stablehlo.reshape %674 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %677 = stablehlo.divide %676, %638 : tensor<1x4096x1xf32>
    %678 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %679 = stablehlo.add %678, %641 : tensor<1x4096x1xf32>
    %680 = stablehlo.rsqrt %679 : tensor<1x4096x1xf32>
    %681 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %682 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %683 = stablehlo.subtract %681, %682 : tensor<1x4096x64xf32>
    %684 = stablehlo.broadcast_in_dim %683, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %685 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %686 = stablehlo.multiply %684, %685 : tensor<1x4096x64xf32>
    %687 = stablehlo.convert %arg33 : (tensor<64xbf16>) -> tensor<64xf32>
    %688 = stablehlo.broadcast_in_dim %686, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %689 = stablehlo.broadcast_in_dim %687, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %690 = stablehlo.multiply %688, %689 : tensor<1x4096x64xf32>
    %691 = stablehlo.convert %arg34 : (tensor<64xbf16>) -> tensor<64xf32>
    %692 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %693 = stablehlo.broadcast_in_dim %691, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %694 = stablehlo.add %692, %693 : tensor<1x4096x64xf32>
    %695 = stablehlo.convert %694 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xbf16>
    %696 = stablehlo.reshape %695 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %697 = stablehlo.convert %696 : (tensor<4096x64xbf16>) -> tensor<4096x64xf32>
    %698 = stablehlo.dot_general %697, %arg134, contracting_dims = [1] x [0] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %699 = stablehlo.broadcast_in_dim %698, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %700 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<4096x64xf32>
    %701 = stablehlo.multiply %699, %700 : tensor<4096x64xf32>
    %702 = stablehlo.broadcast_in_dim %701, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %703 = stablehlo.broadcast_in_dim %arg135, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
    %704 = stablehlo.add %702, %703 : tensor<4096x64xf32>
    %705 = stablehlo.convert %704 : (tensor<4096x64xf32>) -> tensor<4096x64xbf16>
    %706 = stablehlo.reshape %705 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %707 = stablehlo.reshape %706 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x2x32xbf16>
    %708 = stablehlo.transpose %707, dims = [0, 2, 1, 3] : (tensor<1x4096x2x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %709 = stablehlo.transpose %695, dims = [0, 2, 1] : (tensor<1x4096x64xbf16>) -> tensor<1x64x4096xbf16>
    %710 = stablehlo.reshape %709 : (tensor<1x64x4096xbf16>) -> tensor<1x64x64x64xbf16>
    %711 = stablehlo.convolution(%710, %arg35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x64x64xbf16>, tensor<64x64x4x4xbf16>) -> tensor<1x64x16x16xbf16>
    %712 = stablehlo.reshape %arg36 : (tensor<64xbf16>) -> tensor<64x1x1xbf16>
    %713 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x64x16x16xbf16>) -> tensor<1x64x16x16xbf16>
    %714 = stablehlo.broadcast_in_dim %712, dims = [1, 2, 3] : (tensor<64x1x1xbf16>) -> tensor<1x64x16x16xbf16>
    %715 = stablehlo.add %713, %714 : tensor<1x64x16x16xbf16>
    %716 = stablehlo.reshape %715 : (tensor<1x64x16x16xbf16>) -> tensor<1x64x256xbf16>
    %717 = stablehlo.transpose %716, dims = [0, 2, 1] : (tensor<1x64x256xbf16>) -> tensor<1x256x64xbf16>
    %718 = stablehlo.convert %717 : (tensor<1x256x64xbf16>) -> tensor<1x256x64xf32>
    %719 = stablehlo.convert %718 : (tensor<1x256x64xf32>) -> tensor<1x256x64xf64>
    %720 = stablehlo.reduce(%719 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x64xf64>, tensor<f64>) -> tensor<1x256xf64>
    %721 = stablehlo.reshape %720 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %722 = stablehlo.broadcast_in_dim %721, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %723 = stablehlo.broadcast_in_dim %620, dims = [] : (tensor<f64>) -> tensor<1x256x1xf64>
    %724 = stablehlo.divide %722, %723 : tensor<1x256x1xf64>
    %725 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2] : (tensor<1x256x64xf64>) -> tensor<1x256x64xf64>
    %726 = stablehlo.broadcast_in_dim %724, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x64xf64>
    %727 = stablehlo.subtract %725, %726 : tensor<1x256x64xf64>
    %728 = stablehlo.multiply %727, %727 : tensor<1x256x64xf64>
    %729 = stablehlo.reduce(%728 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x64xf64>, tensor<f64>) -> tensor<1x256xf64>
    %730 = stablehlo.reshape %729 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %731 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %732 = stablehlo.divide %731, %723 : tensor<1x256x1xf64>
    %733 = stablehlo.convert %732 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %734 = stablehlo.reduce(%718 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256xf32>
    %735 = stablehlo.reshape %734 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %736 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %737 = stablehlo.broadcast_in_dim %636, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %738 = stablehlo.divide %736, %737 : tensor<1x256x1xf32>
    %739 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %740 = stablehlo.add %739, %136 : tensor<1x256x1xf32>
    %741 = stablehlo.rsqrt %740 : tensor<1x256x1xf32>
    %742 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %743 = stablehlo.broadcast_in_dim %738, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x64xf32>
    %744 = stablehlo.subtract %742, %743 : tensor<1x256x64xf32>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %746 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x64xf32>
    %747 = stablehlo.multiply %745, %746 : tensor<1x256x64xf32>
    %748 = stablehlo.convert %arg37 : (tensor<64xbf16>) -> tensor<64xf32>
    %749 = stablehlo.broadcast_in_dim %747, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %750 = stablehlo.broadcast_in_dim %748, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
    %751 = stablehlo.multiply %749, %750 : tensor<1x256x64xf32>
    %752 = stablehlo.convert %arg38 : (tensor<64xbf16>) -> tensor<64xf32>
    %753 = stablehlo.broadcast_in_dim %751, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %754 = stablehlo.broadcast_in_dim %752, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
    %755 = stablehlo.add %753, %754 : tensor<1x256x64xf32>
    %756 = stablehlo.convert %755 : (tensor<1x256x64xf32>) -> tensor<1x256x64xbf16>
    %757 = stablehlo.reshape %756 : (tensor<1x256x64xbf16>) -> tensor<256x64xbf16>
    %758 = stablehlo.convert %757 : (tensor<256x64xbf16>) -> tensor<256x64xf32>
    %759 = stablehlo.dot_general %758, %arg136, contracting_dims = [1] x [0] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %760 = stablehlo.broadcast_in_dim %759, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %761 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<256x64xf32>
    %762 = stablehlo.multiply %760, %761 : tensor<256x64xf32>
    %763 = stablehlo.broadcast_in_dim %762, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %764 = stablehlo.broadcast_in_dim %arg137, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
    %765 = stablehlo.add %763, %764 : tensor<256x64xf32>
    %766 = stablehlo.convert %765 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    %767 = stablehlo.reshape %766 : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %768 = stablehlo.reshape %767 : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %769 = stablehlo.transpose %768, dims = [0, 2, 1, 3] : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %770 = stablehlo.dot_general %758, %arg138, contracting_dims = [1] x [0] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %772 = stablehlo.multiply %771, %761 : tensor<256x64xf32>
    %773 = stablehlo.broadcast_in_dim %772, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %774 = stablehlo.broadcast_in_dim %arg139, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
    %775 = stablehlo.add %773, %774 : tensor<256x64xf32>
    %776 = stablehlo.convert %775 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    %777 = stablehlo.reshape %776 : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %778 = stablehlo.reshape %777 : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %779 = stablehlo.transpose %778, dims = [0, 2, 1, 3] : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %780 = stablehlo.transpose %769, dims = [0, 1, 3, 2] : (tensor<1x2x256x32xbf16>) -> tensor<1x2x32x256xbf16>
    %781 = stablehlo.reshape %708 : (tensor<1x2x4096x32xbf16>) -> tensor<2x4096x32xbf16>
    %782 = stablehlo.reshape %780 : (tensor<1x2x32x256xbf16>) -> tensor<2x32x256xbf16>
    %783 = stablehlo.broadcast_in_dim %782, dims = [0, 1, 2] : (tensor<2x32x256xbf16>) -> tensor<2x32x256xbf16>
    %784 = stablehlo.dot_general %781, %783, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4096x32xbf16>, tensor<2x32x256xbf16>) -> tensor<2x4096x256xbf16>
    %785 = stablehlo.reshape %784 : (tensor<2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %786 = stablehlo.broadcast_in_dim %785, dims = [0, 1, 2, 3] : (tensor<1x2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %787 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<bf16>) -> tensor<1x2x4096x256xbf16>
    %788 = stablehlo.divide %786, %787 : tensor<1x2x4096x256xbf16>
    %789 = stablehlo.convert %788 : (tensor<1x2x4096x256xbf16>) -> tensor<1x2x4096x256xf32>
    %790 = stablehlo.reduce(%789 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %791 = stablehlo.reshape %790 : (tensor<1x2x4096xf32>) -> tensor<1x2x4096x1xf32>
    %792 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2, 3] : (tensor<1x2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %793 = stablehlo.broadcast_in_dim %791, dims = [0, 1, 2, 3] : (tensor<1x2x4096x1xf32>) -> tensor<1x2x4096x256xf32>
    %794 = stablehlo.subtract %792, %793 : tensor<1x2x4096x256xf32>
    %795 = stablehlo.exponential %794 : tensor<1x2x4096x256xf32>
    %796 = stablehlo.reduce(%795 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %797 = stablehlo.reshape %796 : (tensor<1x2x4096xf32>) -> tensor<1x2x4096x1xf32>
    %798 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %799 = stablehlo.broadcast_in_dim %797, dims = [0, 1, 2, 3] : (tensor<1x2x4096x1xf32>) -> tensor<1x2x4096x256xf32>
    %800 = stablehlo.divide %798, %799 : tensor<1x2x4096x256xf32>
    %801 = stablehlo.convert %800 : (tensor<1x2x4096x256xf32>) -> tensor<1x2x4096x256xbf16>
    %802 = stablehlo.reshape %801 : (tensor<1x2x4096x256xbf16>) -> tensor<2x4096x256xbf16>
    %803 = stablehlo.reshape %779 : (tensor<1x2x256x32xbf16>) -> tensor<2x256x32xbf16>
    %804 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2] : (tensor<2x256x32xbf16>) -> tensor<2x256x32xbf16>
    %805 = stablehlo.dot_general %802, %804, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4096x256xbf16>, tensor<2x256x32xbf16>) -> tensor<2x4096x32xbf16>
    %806 = stablehlo.reshape %805 : (tensor<2x4096x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %807 = stablehlo.transpose %806, dims = [0, 2, 1, 3] : (tensor<1x2x4096x32xbf16>) -> tensor<1x4096x2x32xbf16>
    %808 = stablehlo.reshape %807 : (tensor<1x4096x2x32xbf16>) -> tensor<1x4096x64xbf16>
    %809 = stablehlo.reshape %808 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %810 = stablehlo.convert %809 : (tensor<4096x64xbf16>) -> tensor<4096x64xf32>
    %811 = stablehlo.dot_general %810, %arg140, contracting_dims = [1] x [0] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %812 = stablehlo.broadcast_in_dim %811, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %813 = stablehlo.multiply %812, %700 : tensor<4096x64xf32>
    %814 = stablehlo.broadcast_in_dim %813, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %815 = stablehlo.broadcast_in_dim %arg141, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
    %816 = stablehlo.add %814, %815 : tensor<4096x64xf32>
    %817 = stablehlo.convert %816 : (tensor<4096x64xf32>) -> tensor<4096x64xbf16>
    %818 = stablehlo.reshape %817 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %819 = stablehlo.add %818, %658 : tensor<1x4096x64xbf16>
    %820 = stablehlo.convert %819 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xf32>
    %821 = stablehlo.convert %820 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf64>
    %822 = stablehlo.reduce(%821 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %823 = stablehlo.reshape %822 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %824 = stablehlo.broadcast_in_dim %823, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %825 = stablehlo.divide %824, %622 : tensor<1x4096x1xf64>
    %826 = stablehlo.broadcast_in_dim %821, dims = [0, 1, 2] : (tensor<1x4096x64xf64>) -> tensor<1x4096x64xf64>
    %827 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x64xf64>
    %828 = stablehlo.subtract %826, %827 : tensor<1x4096x64xf64>
    %829 = stablehlo.multiply %828, %828 : tensor<1x4096x64xf64>
    %830 = stablehlo.reduce(%829 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %831 = stablehlo.reshape %830 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %832 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %833 = stablehlo.divide %832, %622 : tensor<1x4096x1xf64>
    %834 = stablehlo.convert %833 : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf32>
    %835 = stablehlo.reduce(%820 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %836 = stablehlo.reshape %835 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %838 = stablehlo.divide %837, %638 : tensor<1x4096x1xf32>
    %839 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %840 = stablehlo.add %839, %641 : tensor<1x4096x1xf32>
    %841 = stablehlo.rsqrt %840 : tensor<1x4096x1xf32>
    %842 = stablehlo.broadcast_in_dim %820, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %843 = stablehlo.broadcast_in_dim %838, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %844 = stablehlo.subtract %842, %843 : tensor<1x4096x64xf32>
    %845 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %846 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %847 = stablehlo.multiply %845, %846 : tensor<1x4096x64xf32>
    %848 = stablehlo.convert %arg39 : (tensor<64xbf16>) -> tensor<64xf32>
    %849 = stablehlo.broadcast_in_dim %847, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %850 = stablehlo.broadcast_in_dim %848, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %851 = stablehlo.multiply %849, %850 : tensor<1x4096x64xf32>
    %852 = stablehlo.convert %arg40 : (tensor<64xbf16>) -> tensor<64xf32>
    %853 = stablehlo.broadcast_in_dim %851, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %854 = stablehlo.broadcast_in_dim %852, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %855 = stablehlo.add %853, %854 : tensor<1x4096x64xf32>
    %856 = stablehlo.convert %855 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xbf16>
    %857 = stablehlo.reshape %856 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %858 = stablehlo.convert %857 : (tensor<4096x64xbf16>) -> tensor<4096x64xf32>
    %859 = stablehlo.dot_general %858, %arg142, contracting_dims = [1] x [0] : (tensor<4096x64xf32>, tensor<64x256xf32>) -> tensor<4096x256xf32>
    %860 = stablehlo.broadcast_in_dim %859, dims = [0, 1] : (tensor<4096x256xf32>) -> tensor<4096x256xf32>
    %861 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %862 = stablehlo.multiply %860, %861 : tensor<4096x256xf32>
    %863 = stablehlo.broadcast_in_dim %862, dims = [0, 1] : (tensor<4096x256xf32>) -> tensor<4096x256xf32>
    %864 = stablehlo.broadcast_in_dim %arg143, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
    %865 = stablehlo.add %863, %864 : tensor<4096x256xf32>
    %866 = stablehlo.convert %865 : (tensor<4096x256xf32>) -> tensor<4096x256xbf16>
    %867 = stablehlo.reshape %866 : (tensor<4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %868 = stablehlo.transpose %867, dims = [0, 2, 1] : (tensor<1x4096x256xbf16>) -> tensor<1x256x4096xbf16>
    %869 = stablehlo.reshape %868 : (tensor<1x256x4096xbf16>) -> tensor<1x256x64x64xbf16>
    %870 = stablehlo.convolution(%869, %arg41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x64x64xbf16>, tensor<256x1x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %871 = stablehlo.reshape %arg42 : (tensor<256xbf16>) -> tensor<256x1x1xbf16>
    %872 = stablehlo.broadcast_in_dim %870, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %873 = stablehlo.broadcast_in_dim %871, dims = [1, 2, 3] : (tensor<256x1x1xbf16>) -> tensor<1x256x64x64xbf16>
    %874 = stablehlo.add %872, %873 : tensor<1x256x64x64xbf16>
    %875 = stablehlo.reshape %874 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x4096xbf16>
    %876 = stablehlo.transpose %875, dims = [0, 2, 1] : (tensor<1x256x4096xbf16>) -> tensor<1x4096x256xbf16>
    %877 = stablehlo.multiply %876, %cst_23 : tensor<1x4096x256xbf16>
    %878 = stablehlo.rsqrt %cst_22 : tensor<1x4096x256xbf16>
    %879 = stablehlo.multiply %876, %878 : tensor<1x4096x256xbf16>
    %880 = stablehlo.convert %879 : (tensor<1x4096x256xbf16>) -> tensor<1x4096x256xf32>
    %881 = stablehlo.clamp %cst_24, %880, %cst_25 : tensor<1x4096x256xf32>
    %882 = stablehlo.multiply %881, %881 : tensor<1x4096x256xf32>
    %883 = stablehlo.multiply %cst_26, %882 : tensor<1x4096x256xf32>
    %884 = stablehlo.add %883, %cst_27 : tensor<1x4096x256xf32>
    %885 = stablehlo.multiply %884, %882 : tensor<1x4096x256xf32>
    %886 = stablehlo.add %885, %cst_28 : tensor<1x4096x256xf32>
    %887 = stablehlo.multiply %886, %882 : tensor<1x4096x256xf32>
    %888 = stablehlo.add %887, %cst_29 : tensor<1x4096x256xf32>
    %889 = stablehlo.multiply %888, %882 : tensor<1x4096x256xf32>
    %890 = stablehlo.add %889, %cst_30 : tensor<1x4096x256xf32>
    %891 = stablehlo.multiply %890, %882 : tensor<1x4096x256xf32>
    %892 = stablehlo.add %891, %cst_31 : tensor<1x4096x256xf32>
    %893 = stablehlo.multiply %892, %882 : tensor<1x4096x256xf32>
    %894 = stablehlo.add %893, %cst_32 : tensor<1x4096x256xf32>
    %895 = stablehlo.multiply %cst_33, %882 : tensor<1x4096x256xf32>
    %896 = stablehlo.add %895, %cst_34 : tensor<1x4096x256xf32>
    %897 = stablehlo.multiply %896, %882 : tensor<1x4096x256xf32>
    %898 = stablehlo.add %897, %cst_35 : tensor<1x4096x256xf32>
    %899 = stablehlo.multiply %898, %882 : tensor<1x4096x256xf32>
    %900 = stablehlo.add %899, %cst_36 : tensor<1x4096x256xf32>
    %901 = stablehlo.multiply %900, %882 : tensor<1x4096x256xf32>
    %902 = stablehlo.add %901, %cst_37 : tensor<1x4096x256xf32>
    %903 = stablehlo.multiply %881, %894 : tensor<1x4096x256xf32>
    %904 = stablehlo.divide %903, %902 : tensor<1x4096x256xf32>
    %905 = stablehlo.clamp %cst_38, %904, %cst_39 : tensor<1x4096x256xf32>
    %906 = stablehlo.convert %905 : (tensor<1x4096x256xf32>) -> tensor<1x4096x256xbf16>
    %907 = stablehlo.add %906, %cst_21 : tensor<1x4096x256xbf16>
    %908 = stablehlo.multiply %907, %877 : tensor<1x4096x256xbf16>
    %909 = stablehlo.reshape %908 : (tensor<1x4096x256xbf16>) -> tensor<4096x256xbf16>
    %910 = stablehlo.dot_general %909, %arg144, contracting_dims = [1] x [0] : (tensor<4096x256xbf16>, tensor<256x64xbf16>) -> tensor<4096x64xbf16>
    %911 = stablehlo.reshape %910 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %912 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %913 = stablehlo.broadcast_in_dim %arg43, dims = [2] : (tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %914 = stablehlo.add %912, %913 : tensor<1x4096x64xbf16>
    %915 = stablehlo.reshape %914 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %916 = stablehlo.reshape %915 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %917 = stablehlo.add %916, %819 : tensor<1x4096x64xbf16>
    %918 = stablehlo.convert %917 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xf32>
    %919 = stablehlo.convert %918 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf64>
    %920 = stablehlo.reduce(%919 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %921 = stablehlo.reshape %920 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %922 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %923 = stablehlo.divide %922, %622 : tensor<1x4096x1xf64>
    %924 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2] : (tensor<1x4096x64xf64>) -> tensor<1x4096x64xf64>
    %925 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x64xf64>
    %926 = stablehlo.subtract %924, %925 : tensor<1x4096x64xf64>
    %927 = stablehlo.multiply %926, %926 : tensor<1x4096x64xf64>
    %928 = stablehlo.reduce(%927 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %929 = stablehlo.reshape %928 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %930 = stablehlo.broadcast_in_dim %929, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %931 = stablehlo.divide %930, %622 : tensor<1x4096x1xf64>
    %932 = stablehlo.convert %931 : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf32>
    %933 = stablehlo.reduce(%918 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %934 = stablehlo.reshape %933 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %936 = stablehlo.divide %935, %638 : tensor<1x4096x1xf32>
    %937 = stablehlo.broadcast_in_dim %932, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %938 = stablehlo.add %937, %641 : tensor<1x4096x1xf32>
    %939 = stablehlo.rsqrt %938 : tensor<1x4096x1xf32>
    %940 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %941 = stablehlo.broadcast_in_dim %936, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %942 = stablehlo.subtract %940, %941 : tensor<1x4096x64xf32>
    %943 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %944 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %945 = stablehlo.multiply %943, %944 : tensor<1x4096x64xf32>
    %946 = stablehlo.convert %arg44 : (tensor<64xbf16>) -> tensor<64xf32>
    %947 = stablehlo.broadcast_in_dim %945, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %948 = stablehlo.broadcast_in_dim %946, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %949 = stablehlo.multiply %947, %948 : tensor<1x4096x64xf32>
    %950 = stablehlo.convert %arg45 : (tensor<64xbf16>) -> tensor<64xf32>
    %951 = stablehlo.broadcast_in_dim %949, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %952 = stablehlo.broadcast_in_dim %950, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %953 = stablehlo.add %951, %952 : tensor<1x4096x64xf32>
    %954 = stablehlo.convert %953 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xbf16>
    %955 = stablehlo.reshape %954 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %956 = stablehlo.convert %955 : (tensor<4096x64xbf16>) -> tensor<4096x64xf32>
    %957 = stablehlo.dot_general %956, %arg145, contracting_dims = [1] x [0] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %959 = stablehlo.multiply %958, %700 : tensor<4096x64xf32>
    %960 = stablehlo.broadcast_in_dim %959, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %961 = stablehlo.broadcast_in_dim %arg146, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
    %962 = stablehlo.add %960, %961 : tensor<4096x64xf32>
    %963 = stablehlo.convert %962 : (tensor<4096x64xf32>) -> tensor<4096x64xbf16>
    %964 = stablehlo.reshape %963 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %965 = stablehlo.reshape %964 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x2x32xbf16>
    %966 = stablehlo.transpose %965, dims = [0, 2, 1, 3] : (tensor<1x4096x2x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %967 = stablehlo.transpose %954, dims = [0, 2, 1] : (tensor<1x4096x64xbf16>) -> tensor<1x64x4096xbf16>
    %968 = stablehlo.reshape %967 : (tensor<1x64x4096xbf16>) -> tensor<1x64x64x64xbf16>
    %969 = stablehlo.convolution(%968, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x64x64xbf16>, tensor<64x64x4x4xbf16>) -> tensor<1x64x16x16xbf16>
    %970 = stablehlo.reshape %arg47 : (tensor<64xbf16>) -> tensor<64x1x1xbf16>
    %971 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2, 3] : (tensor<1x64x16x16xbf16>) -> tensor<1x64x16x16xbf16>
    %972 = stablehlo.broadcast_in_dim %970, dims = [1, 2, 3] : (tensor<64x1x1xbf16>) -> tensor<1x64x16x16xbf16>
    %973 = stablehlo.add %971, %972 : tensor<1x64x16x16xbf16>
    %974 = stablehlo.reshape %973 : (tensor<1x64x16x16xbf16>) -> tensor<1x64x256xbf16>
    %975 = stablehlo.transpose %974, dims = [0, 2, 1] : (tensor<1x64x256xbf16>) -> tensor<1x256x64xbf16>
    %976 = stablehlo.convert %975 : (tensor<1x256x64xbf16>) -> tensor<1x256x64xf32>
    %977 = stablehlo.convert %976 : (tensor<1x256x64xf32>) -> tensor<1x256x64xf64>
    %978 = stablehlo.reduce(%977 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x64xf64>, tensor<f64>) -> tensor<1x256xf64>
    %979 = stablehlo.reshape %978 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %980 = stablehlo.broadcast_in_dim %979, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %981 = stablehlo.divide %980, %723 : tensor<1x256x1xf64>
    %982 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2] : (tensor<1x256x64xf64>) -> tensor<1x256x64xf64>
    %983 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x64xf64>
    %984 = stablehlo.subtract %982, %983 : tensor<1x256x64xf64>
    %985 = stablehlo.multiply %984, %984 : tensor<1x256x64xf64>
    %986 = stablehlo.reduce(%985 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x64xf64>, tensor<f64>) -> tensor<1x256xf64>
    %987 = stablehlo.reshape %986 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %988 = stablehlo.broadcast_in_dim %987, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %989 = stablehlo.divide %988, %723 : tensor<1x256x1xf64>
    %990 = stablehlo.convert %989 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %991 = stablehlo.reduce(%976 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256xf32>
    %992 = stablehlo.reshape %991 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %993 = stablehlo.broadcast_in_dim %992, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %994 = stablehlo.divide %993, %737 : tensor<1x256x1xf32>
    %995 = stablehlo.broadcast_in_dim %990, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %996 = stablehlo.add %995, %136 : tensor<1x256x1xf32>
    %997 = stablehlo.rsqrt %996 : tensor<1x256x1xf32>
    %998 = stablehlo.broadcast_in_dim %976, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %999 = stablehlo.broadcast_in_dim %994, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x64xf32>
    %1000 = stablehlo.subtract %998, %999 : tensor<1x256x64xf32>
    %1001 = stablehlo.broadcast_in_dim %1000, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %1002 = stablehlo.broadcast_in_dim %997, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x64xf32>
    %1003 = stablehlo.multiply %1001, %1002 : tensor<1x256x64xf32>
    %1004 = stablehlo.convert %arg48 : (tensor<64xbf16>) -> tensor<64xf32>
    %1005 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %1006 = stablehlo.broadcast_in_dim %1004, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
    %1007 = stablehlo.multiply %1005, %1006 : tensor<1x256x64xf32>
    %1008 = stablehlo.convert %arg49 : (tensor<64xbf16>) -> tensor<64xf32>
    %1009 = stablehlo.broadcast_in_dim %1007, dims = [0, 1, 2] : (tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
    %1010 = stablehlo.broadcast_in_dim %1008, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
    %1011 = stablehlo.add %1009, %1010 : tensor<1x256x64xf32>
    %1012 = stablehlo.convert %1011 : (tensor<1x256x64xf32>) -> tensor<1x256x64xbf16>
    %1013 = stablehlo.reshape %1012 : (tensor<1x256x64xbf16>) -> tensor<256x64xbf16>
    %1014 = stablehlo.convert %1013 : (tensor<256x64xbf16>) -> tensor<256x64xf32>
    %1015 = stablehlo.dot_general %1014, %arg147, contracting_dims = [1] x [0] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %1017 = stablehlo.multiply %1016, %761 : tensor<256x64xf32>
    %1018 = stablehlo.broadcast_in_dim %1017, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %1019 = stablehlo.broadcast_in_dim %arg148, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
    %1020 = stablehlo.add %1018, %1019 : tensor<256x64xf32>
    %1021 = stablehlo.convert %1020 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    %1022 = stablehlo.reshape %1021 : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %1023 = stablehlo.reshape %1022 : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %1024 = stablehlo.transpose %1023, dims = [0, 2, 1, 3] : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %1025 = stablehlo.dot_general %1014, %arg149, contracting_dims = [1] x [0] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %1026 = stablehlo.broadcast_in_dim %1025, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %1027 = stablehlo.multiply %1026, %761 : tensor<256x64xf32>
    %1028 = stablehlo.broadcast_in_dim %1027, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64xf32>
    %1029 = stablehlo.broadcast_in_dim %arg150, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
    %1030 = stablehlo.add %1028, %1029 : tensor<256x64xf32>
    %1031 = stablehlo.convert %1030 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    %1032 = stablehlo.reshape %1031 : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %1033 = stablehlo.reshape %1032 : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %1034 = stablehlo.transpose %1033, dims = [0, 2, 1, 3] : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %1035 = stablehlo.transpose %1024, dims = [0, 1, 3, 2] : (tensor<1x2x256x32xbf16>) -> tensor<1x2x32x256xbf16>
    %1036 = stablehlo.reshape %966 : (tensor<1x2x4096x32xbf16>) -> tensor<2x4096x32xbf16>
    %1037 = stablehlo.reshape %1035 : (tensor<1x2x32x256xbf16>) -> tensor<2x32x256xbf16>
    %1038 = stablehlo.broadcast_in_dim %1037, dims = [0, 1, 2] : (tensor<2x32x256xbf16>) -> tensor<2x32x256xbf16>
    %1039 = stablehlo.dot_general %1036, %1038, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4096x32xbf16>, tensor<2x32x256xbf16>) -> tensor<2x4096x256xbf16>
    %1040 = stablehlo.reshape %1039 : (tensor<2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2, 3] : (tensor<1x2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %1042 = stablehlo.divide %1041, %787 : tensor<1x2x4096x256xbf16>
    %1043 = stablehlo.convert %1042 : (tensor<1x2x4096x256xbf16>) -> tensor<1x2x4096x256xf32>
    %1044 = stablehlo.reduce(%1043 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %1045 = stablehlo.reshape %1044 : (tensor<1x2x4096xf32>) -> tensor<1x2x4096x1xf32>
    %1046 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2, 3] : (tensor<1x2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %1047 = stablehlo.broadcast_in_dim %1045, dims = [0, 1, 2, 3] : (tensor<1x2x4096x1xf32>) -> tensor<1x2x4096x256xf32>
    %1048 = stablehlo.subtract %1046, %1047 : tensor<1x2x4096x256xf32>
    %1049 = stablehlo.exponential %1048 : tensor<1x2x4096x256xf32>
    %1050 = stablehlo.reduce(%1049 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %1051 = stablehlo.reshape %1050 : (tensor<1x2x4096xf32>) -> tensor<1x2x4096x1xf32>
    %1052 = stablehlo.broadcast_in_dim %1049, dims = [0, 1, 2, 3] : (tensor<1x2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %1053 = stablehlo.broadcast_in_dim %1051, dims = [0, 1, 2, 3] : (tensor<1x2x4096x1xf32>) -> tensor<1x2x4096x256xf32>
    %1054 = stablehlo.divide %1052, %1053 : tensor<1x2x4096x256xf32>
    %1055 = stablehlo.convert %1054 : (tensor<1x2x4096x256xf32>) -> tensor<1x2x4096x256xbf16>
    %1056 = stablehlo.reshape %1055 : (tensor<1x2x4096x256xbf16>) -> tensor<2x4096x256xbf16>
    %1057 = stablehlo.reshape %1034 : (tensor<1x2x256x32xbf16>) -> tensor<2x256x32xbf16>
    %1058 = stablehlo.broadcast_in_dim %1057, dims = [0, 1, 2] : (tensor<2x256x32xbf16>) -> tensor<2x256x32xbf16>
    %1059 = stablehlo.dot_general %1056, %1058, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4096x256xbf16>, tensor<2x256x32xbf16>) -> tensor<2x4096x32xbf16>
    %1060 = stablehlo.reshape %1059 : (tensor<2x4096x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %1061 = stablehlo.transpose %1060, dims = [0, 2, 1, 3] : (tensor<1x2x4096x32xbf16>) -> tensor<1x4096x2x32xbf16>
    %1062 = stablehlo.reshape %1061 : (tensor<1x4096x2x32xbf16>) -> tensor<1x4096x64xbf16>
    %1063 = stablehlo.reshape %1062 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %1064 = stablehlo.convert %1063 : (tensor<4096x64xbf16>) -> tensor<4096x64xf32>
    %1065 = stablehlo.dot_general %1064, %arg151, contracting_dims = [1] x [0] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %1066 = stablehlo.broadcast_in_dim %1065, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %1067 = stablehlo.multiply %1066, %700 : tensor<4096x64xf32>
    %1068 = stablehlo.broadcast_in_dim %1067, dims = [0, 1] : (tensor<4096x64xf32>) -> tensor<4096x64xf32>
    %1069 = stablehlo.broadcast_in_dim %arg152, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
    %1070 = stablehlo.add %1068, %1069 : tensor<4096x64xf32>
    %1071 = stablehlo.convert %1070 : (tensor<4096x64xf32>) -> tensor<4096x64xbf16>
    %1072 = stablehlo.reshape %1071 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %1073 = stablehlo.add %1072, %917 : tensor<1x4096x64xbf16>
    %1074 = stablehlo.convert %1073 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xf32>
    %1075 = stablehlo.convert %1074 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf64>
    %1076 = stablehlo.reduce(%1075 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %1077 = stablehlo.reshape %1076 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %1079 = stablehlo.divide %1078, %622 : tensor<1x4096x1xf64>
    %1080 = stablehlo.broadcast_in_dim %1075, dims = [0, 1, 2] : (tensor<1x4096x64xf64>) -> tensor<1x4096x64xf64>
    %1081 = stablehlo.broadcast_in_dim %1079, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x64xf64>
    %1082 = stablehlo.subtract %1080, %1081 : tensor<1x4096x64xf64>
    %1083 = stablehlo.multiply %1082, %1082 : tensor<1x4096x64xf64>
    %1084 = stablehlo.reduce(%1083 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %1085 = stablehlo.reshape %1084 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %1086 = stablehlo.broadcast_in_dim %1085, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %1087 = stablehlo.divide %1086, %622 : tensor<1x4096x1xf64>
    %1088 = stablehlo.convert %1087 : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf32>
    %1089 = stablehlo.reduce(%1074 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1090 = stablehlo.reshape %1089 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1091 = stablehlo.broadcast_in_dim %1090, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %1092 = stablehlo.divide %1091, %638 : tensor<1x4096x1xf32>
    %1093 = stablehlo.broadcast_in_dim %1088, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %1094 = stablehlo.add %1093, %641 : tensor<1x4096x1xf32>
    %1095 = stablehlo.rsqrt %1094 : tensor<1x4096x1xf32>
    %1096 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1097 = stablehlo.broadcast_in_dim %1092, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %1098 = stablehlo.subtract %1096, %1097 : tensor<1x4096x64xf32>
    %1099 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1100 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %1101 = stablehlo.multiply %1099, %1100 : tensor<1x4096x64xf32>
    %1102 = stablehlo.convert %arg50 : (tensor<64xbf16>) -> tensor<64xf32>
    %1103 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1104 = stablehlo.broadcast_in_dim %1102, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %1105 = stablehlo.multiply %1103, %1104 : tensor<1x4096x64xf32>
    %1106 = stablehlo.convert %arg51 : (tensor<64xbf16>) -> tensor<64xf32>
    %1107 = stablehlo.broadcast_in_dim %1105, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1108 = stablehlo.broadcast_in_dim %1106, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %1109 = stablehlo.add %1107, %1108 : tensor<1x4096x64xf32>
    %1110 = stablehlo.convert %1109 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xbf16>
    %1111 = stablehlo.reshape %1110 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %1112 = stablehlo.convert %1111 : (tensor<4096x64xbf16>) -> tensor<4096x64xf32>
    %1113 = stablehlo.dot_general %1112, %arg153, contracting_dims = [1] x [0] : (tensor<4096x64xf32>, tensor<64x256xf32>) -> tensor<4096x256xf32>
    %1114 = stablehlo.broadcast_in_dim %1113, dims = [0, 1] : (tensor<4096x256xf32>) -> tensor<4096x256xf32>
    %1115 = stablehlo.multiply %1114, %861 : tensor<4096x256xf32>
    %1116 = stablehlo.broadcast_in_dim %1115, dims = [0, 1] : (tensor<4096x256xf32>) -> tensor<4096x256xf32>
    %1117 = stablehlo.broadcast_in_dim %arg154, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
    %1118 = stablehlo.add %1116, %1117 : tensor<4096x256xf32>
    %1119 = stablehlo.convert %1118 : (tensor<4096x256xf32>) -> tensor<4096x256xbf16>
    %1120 = stablehlo.reshape %1119 : (tensor<4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %1121 = stablehlo.transpose %1120, dims = [0, 2, 1] : (tensor<1x4096x256xbf16>) -> tensor<1x256x4096xbf16>
    %1122 = stablehlo.reshape %1121 : (tensor<1x256x4096xbf16>) -> tensor<1x256x64x64xbf16>
    %1123 = stablehlo.convolution(%1122, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x64x64xbf16>, tensor<256x1x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %1124 = stablehlo.reshape %arg53 : (tensor<256xbf16>) -> tensor<256x1x1xbf16>
    %1125 = stablehlo.broadcast_in_dim %1123, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1126 = stablehlo.broadcast_in_dim %1124, dims = [1, 2, 3] : (tensor<256x1x1xbf16>) -> tensor<1x256x64x64xbf16>
    %1127 = stablehlo.add %1125, %1126 : tensor<1x256x64x64xbf16>
    %1128 = stablehlo.reshape %1127 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x4096xbf16>
    %1129 = stablehlo.transpose %1128, dims = [0, 2, 1] : (tensor<1x256x4096xbf16>) -> tensor<1x4096x256xbf16>
    %1130 = stablehlo.multiply %1129, %cst_23 : tensor<1x4096x256xbf16>
    %1131 = stablehlo.multiply %1129, %878 : tensor<1x4096x256xbf16>
    %1132 = stablehlo.convert %1131 : (tensor<1x4096x256xbf16>) -> tensor<1x4096x256xf32>
    %1133 = stablehlo.clamp %cst_24, %1132, %cst_25 : tensor<1x4096x256xf32>
    %1134 = stablehlo.multiply %1133, %1133 : tensor<1x4096x256xf32>
    %1135 = stablehlo.multiply %cst_26, %1134 : tensor<1x4096x256xf32>
    %1136 = stablehlo.add %1135, %cst_27 : tensor<1x4096x256xf32>
    %1137 = stablehlo.multiply %1136, %1134 : tensor<1x4096x256xf32>
    %1138 = stablehlo.add %1137, %cst_28 : tensor<1x4096x256xf32>
    %1139 = stablehlo.multiply %1138, %1134 : tensor<1x4096x256xf32>
    %1140 = stablehlo.add %1139, %cst_29 : tensor<1x4096x256xf32>
    %1141 = stablehlo.multiply %1140, %1134 : tensor<1x4096x256xf32>
    %1142 = stablehlo.add %1141, %cst_30 : tensor<1x4096x256xf32>
    %1143 = stablehlo.multiply %1142, %1134 : tensor<1x4096x256xf32>
    %1144 = stablehlo.add %1143, %cst_31 : tensor<1x4096x256xf32>
    %1145 = stablehlo.multiply %1144, %1134 : tensor<1x4096x256xf32>
    %1146 = stablehlo.add %1145, %cst_32 : tensor<1x4096x256xf32>
    %1147 = stablehlo.multiply %cst_33, %1134 : tensor<1x4096x256xf32>
    %1148 = stablehlo.add %1147, %cst_34 : tensor<1x4096x256xf32>
    %1149 = stablehlo.multiply %1148, %1134 : tensor<1x4096x256xf32>
    %1150 = stablehlo.add %1149, %cst_35 : tensor<1x4096x256xf32>
    %1151 = stablehlo.multiply %1150, %1134 : tensor<1x4096x256xf32>
    %1152 = stablehlo.add %1151, %cst_36 : tensor<1x4096x256xf32>
    %1153 = stablehlo.multiply %1152, %1134 : tensor<1x4096x256xf32>
    %1154 = stablehlo.add %1153, %cst_37 : tensor<1x4096x256xf32>
    %1155 = stablehlo.multiply %1133, %1146 : tensor<1x4096x256xf32>
    %1156 = stablehlo.divide %1155, %1154 : tensor<1x4096x256xf32>
    %1157 = stablehlo.clamp %cst_38, %1156, %cst_39 : tensor<1x4096x256xf32>
    %1158 = stablehlo.convert %1157 : (tensor<1x4096x256xf32>) -> tensor<1x4096x256xbf16>
    %1159 = stablehlo.add %1158, %cst_21 : tensor<1x4096x256xbf16>
    %1160 = stablehlo.multiply %1159, %1130 : tensor<1x4096x256xbf16>
    %1161 = stablehlo.reshape %1160 : (tensor<1x4096x256xbf16>) -> tensor<4096x256xbf16>
    %1162 = stablehlo.dot_general %1161, %arg155, contracting_dims = [1] x [0] : (tensor<4096x256xbf16>, tensor<256x64xbf16>) -> tensor<4096x64xbf16>
    %1163 = stablehlo.reshape %1162 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %1164 = stablehlo.broadcast_in_dim %1163, dims = [0, 1, 2] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %1165 = stablehlo.broadcast_in_dim %arg54, dims = [2] : (tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %1166 = stablehlo.add %1164, %1165 : tensor<1x4096x64xbf16>
    %1167 = stablehlo.reshape %1166 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %1168 = stablehlo.reshape %1167 : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %1169 = stablehlo.add %1168, %1073 : tensor<1x4096x64xbf16>
    %1170 = stablehlo.convert %1169 : (tensor<1x4096x64xbf16>) -> tensor<1x4096x64xf32>
    %1171 = stablehlo.convert %1170 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf64>
    %1172 = stablehlo.reduce(%1171 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %1173 = stablehlo.reshape %1172 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %1174 = stablehlo.broadcast_in_dim %1173, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %1175 = stablehlo.divide %1174, %622 : tensor<1x4096x1xf64>
    %1176 = stablehlo.broadcast_in_dim %1171, dims = [0, 1, 2] : (tensor<1x4096x64xf64>) -> tensor<1x4096x64xf64>
    %1177 = stablehlo.broadcast_in_dim %1175, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x64xf64>
    %1178 = stablehlo.subtract %1176, %1177 : tensor<1x4096x64xf64>
    %1179 = stablehlo.multiply %1178, %1178 : tensor<1x4096x64xf64>
    %1180 = stablehlo.reduce(%1179 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf64>, tensor<f64>) -> tensor<1x4096xf64>
    %1181 = stablehlo.reshape %1180 : (tensor<1x4096xf64>) -> tensor<1x4096x1xf64>
    %1182 = stablehlo.broadcast_in_dim %1181, dims = [0, 1, 2] : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf64>
    %1183 = stablehlo.divide %1182, %622 : tensor<1x4096x1xf64>
    %1184 = stablehlo.convert %1183 : (tensor<1x4096x1xf64>) -> tensor<1x4096x1xf32>
    %1185 = stablehlo.reduce(%1170 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1186 = stablehlo.reshape %1185 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1187 = stablehlo.broadcast_in_dim %1186, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %1188 = stablehlo.divide %1187, %638 : tensor<1x4096x1xf32>
    %1189 = stablehlo.broadcast_in_dim %1184, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %1190 = stablehlo.add %1189, %641 : tensor<1x4096x1xf32>
    %1191 = stablehlo.rsqrt %1190 : tensor<1x4096x1xf32>
    %1192 = stablehlo.broadcast_in_dim %1170, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1193 = stablehlo.broadcast_in_dim %1188, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %1194 = stablehlo.subtract %1192, %1193 : tensor<1x4096x64xf32>
    %1195 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1196 = stablehlo.broadcast_in_dim %1191, dims = [0, 1, 2] : (tensor<1x4096x1xf32>) -> tensor<1x4096x64xf32>
    %1197 = stablehlo.multiply %1195, %1196 : tensor<1x4096x64xf32>
    %1198 = stablehlo.convert %arg55 : (tensor<64xbf16>) -> tensor<64xf32>
    %1199 = stablehlo.broadcast_in_dim %1197, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1200 = stablehlo.broadcast_in_dim %1198, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %1201 = stablehlo.multiply %1199, %1200 : tensor<1x4096x64xf32>
    %1202 = stablehlo.convert %arg56 : (tensor<64xbf16>) -> tensor<64xf32>
    %1203 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2] : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    %1204 = stablehlo.broadcast_in_dim %1202, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
    %1205 = stablehlo.add %1203, %1204 : tensor<1x4096x64xf32>
    %1206 = stablehlo.convert %1205 : (tensor<1x4096x64xf32>) -> tensor<1x4096x64xbf16>
    %1207 = stablehlo.reshape %1206 : (tensor<1x4096x64xbf16>) -> tensor<1x64x64x64xbf16>
    %1208 = stablehlo.transpose %1207, dims = [0, 3, 1, 2] : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %1209 = stablehlo.convolution(%1208, %arg57) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x64x64xbf16>, tensor<160x64x3x3xbf16>) -> tensor<1x160x32x32xbf16>
    %1210 = stablehlo.reshape %arg58 : (tensor<160xbf16>) -> tensor<160x1x1xbf16>
    %1211 = stablehlo.broadcast_in_dim %1209, dims = [0, 1, 2, 3] : (tensor<1x160x32x32xbf16>) -> tensor<1x160x32x32xbf16>
    %1212 = stablehlo.broadcast_in_dim %1210, dims = [1, 2, 3] : (tensor<160x1x1xbf16>) -> tensor<1x160x32x32xbf16>
    %1213 = stablehlo.add %1211, %1212 : tensor<1x160x32x32xbf16>
    %1214 = stablehlo.reshape %1213 : (tensor<1x160x32x32xbf16>) -> tensor<1x160x1024xbf16>
    %1215 = stablehlo.transpose %1214, dims = [0, 2, 1] : (tensor<1x160x1024xbf16>) -> tensor<1x1024x160xbf16>
    %1216 = stablehlo.convert %1215 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xf32>
    %1217 = stablehlo.convert %1216 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf64>
    %1218 = stablehlo.reduce(%1217 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1219 = stablehlo.reshape %1218 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1220 = stablehlo.convert %cst_84 : (tensor<1xi64>) -> tensor<1xf64>
    %1221 = stablehlo.reshape %1220 : (tensor<1xf64>) -> tensor<f64>
    %1222 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1223 = stablehlo.broadcast_in_dim %1221, dims = [] : (tensor<f64>) -> tensor<1x1024x1xf64>
    %1224 = stablehlo.divide %1222, %1223 : tensor<1x1024x1xf64>
    %1225 = stablehlo.broadcast_in_dim %1217, dims = [0, 1, 2] : (tensor<1x1024x160xf64>) -> tensor<1x1024x160xf64>
    %1226 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x160xf64>
    %1227 = stablehlo.subtract %1225, %1226 : tensor<1x1024x160xf64>
    %1228 = stablehlo.multiply %1227, %1227 : tensor<1x1024x160xf64>
    %1229 = stablehlo.reduce(%1228 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1230 = stablehlo.reshape %1229 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1231 = stablehlo.broadcast_in_dim %1230, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1232 = stablehlo.divide %1231, %1223 : tensor<1x1024x1xf64>
    %1233 = stablehlo.convert %1232 : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf32>
    %1234 = stablehlo.reduce(%1216 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1235 = stablehlo.reshape %1234 : (tensor<1x1024xf32>) -> tensor<1x1024x1xf32>
    %1236 = stablehlo.convert %cst_84 : (tensor<1xi64>) -> tensor<1xf32>
    %1237 = stablehlo.reshape %1236 : (tensor<1xf32>) -> tensor<f32>
    %1238 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1239 = stablehlo.broadcast_in_dim %1237, dims = [] : (tensor<f32>) -> tensor<1x1024x1xf32>
    %1240 = stablehlo.divide %1238, %1239 : tensor<1x1024x1xf32>
    %1241 = stablehlo.broadcast_in_dim %1233, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1242 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<1x1024x1xf32>
    %1243 = stablehlo.add %1241, %1242 : tensor<1x1024x1xf32>
    %1244 = stablehlo.rsqrt %1243 : tensor<1x1024x1xf32>
    %1245 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1246 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1247 = stablehlo.subtract %1245, %1246 : tensor<1x1024x160xf32>
    %1248 = stablehlo.broadcast_in_dim %1247, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1249 = stablehlo.broadcast_in_dim %1244, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1250 = stablehlo.multiply %1248, %1249 : tensor<1x1024x160xf32>
    %1251 = stablehlo.convert %arg59 : (tensor<160xbf16>) -> tensor<160xf32>
    %1252 = stablehlo.broadcast_in_dim %1250, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1253 = stablehlo.broadcast_in_dim %1251, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1254 = stablehlo.multiply %1252, %1253 : tensor<1x1024x160xf32>
    %1255 = stablehlo.convert %arg60 : (tensor<160xbf16>) -> tensor<160xf32>
    %1256 = stablehlo.broadcast_in_dim %1254, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1257 = stablehlo.broadcast_in_dim %1255, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1258 = stablehlo.add %1256, %1257 : tensor<1x1024x160xf32>
    %1259 = stablehlo.convert %1258 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xbf16>
    %1260 = stablehlo.convert %1259 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xf32>
    %1261 = stablehlo.convert %1260 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf64>
    %1262 = stablehlo.reduce(%1261 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1263 = stablehlo.reshape %1262 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1264 = stablehlo.broadcast_in_dim %1263, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1265 = stablehlo.divide %1264, %1223 : tensor<1x1024x1xf64>
    %1266 = stablehlo.broadcast_in_dim %1261, dims = [0, 1, 2] : (tensor<1x1024x160xf64>) -> tensor<1x1024x160xf64>
    %1267 = stablehlo.broadcast_in_dim %1265, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x160xf64>
    %1268 = stablehlo.subtract %1266, %1267 : tensor<1x1024x160xf64>
    %1269 = stablehlo.multiply %1268, %1268 : tensor<1x1024x160xf64>
    %1270 = stablehlo.reduce(%1269 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1271 = stablehlo.reshape %1270 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1272 = stablehlo.broadcast_in_dim %1271, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1273 = stablehlo.divide %1272, %1223 : tensor<1x1024x1xf64>
    %1274 = stablehlo.convert %1273 : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf32>
    %1275 = stablehlo.reduce(%1260 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1276 = stablehlo.reshape %1275 : (tensor<1x1024xf32>) -> tensor<1x1024x1xf32>
    %1277 = stablehlo.broadcast_in_dim %1276, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1278 = stablehlo.divide %1277, %1239 : tensor<1x1024x1xf32>
    %1279 = stablehlo.broadcast_in_dim %1274, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1280 = stablehlo.add %1279, %1242 : tensor<1x1024x1xf32>
    %1281 = stablehlo.rsqrt %1280 : tensor<1x1024x1xf32>
    %1282 = stablehlo.broadcast_in_dim %1260, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1283 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1284 = stablehlo.subtract %1282, %1283 : tensor<1x1024x160xf32>
    %1285 = stablehlo.broadcast_in_dim %1284, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1286 = stablehlo.broadcast_in_dim %1281, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1287 = stablehlo.multiply %1285, %1286 : tensor<1x1024x160xf32>
    %1288 = stablehlo.convert %arg61 : (tensor<160xbf16>) -> tensor<160xf32>
    %1289 = stablehlo.broadcast_in_dim %1287, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1290 = stablehlo.broadcast_in_dim %1288, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1291 = stablehlo.multiply %1289, %1290 : tensor<1x1024x160xf32>
    %1292 = stablehlo.convert %arg62 : (tensor<160xbf16>) -> tensor<160xf32>
    %1293 = stablehlo.broadcast_in_dim %1291, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1294 = stablehlo.broadcast_in_dim %1292, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1295 = stablehlo.add %1293, %1294 : tensor<1x1024x160xf32>
    %1296 = stablehlo.convert %1295 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xbf16>
    %1297 = stablehlo.reshape %1296 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1298 = stablehlo.convert %1297 : (tensor<1024x160xbf16>) -> tensor<1024x160xf32>
    %1299 = stablehlo.dot_general %1298, %arg156, contracting_dims = [1] x [0] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1301 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<1024x160xf32>
    %1302 = stablehlo.multiply %1300, %1301 : tensor<1024x160xf32>
    %1303 = stablehlo.broadcast_in_dim %1302, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1304 = stablehlo.broadcast_in_dim %arg157, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1305 = stablehlo.add %1303, %1304 : tensor<1024x160xf32>
    %1306 = stablehlo.convert %1305 : (tensor<1024x160xf32>) -> tensor<1024x160xbf16>
    %1307 = stablehlo.reshape %1306 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1308 = stablehlo.reshape %1307 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x5x32xbf16>
    %1309 = stablehlo.transpose %1308, dims = [0, 2, 1, 3] : (tensor<1x1024x5x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %1310 = stablehlo.transpose %1296, dims = [0, 2, 1] : (tensor<1x1024x160xbf16>) -> tensor<1x160x1024xbf16>
    %1311 = stablehlo.reshape %1310 : (tensor<1x160x1024xbf16>) -> tensor<1x160x32x32xbf16>
    %1312 = stablehlo.convolution(%1311, %arg63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x160x32x32xbf16>, tensor<160x160x2x2xbf16>) -> tensor<1x160x16x16xbf16>
    %1313 = stablehlo.reshape %arg64 : (tensor<160xbf16>) -> tensor<160x1x1xbf16>
    %1314 = stablehlo.broadcast_in_dim %1312, dims = [0, 1, 2, 3] : (tensor<1x160x16x16xbf16>) -> tensor<1x160x16x16xbf16>
    %1315 = stablehlo.broadcast_in_dim %1313, dims = [1, 2, 3] : (tensor<160x1x1xbf16>) -> tensor<1x160x16x16xbf16>
    %1316 = stablehlo.add %1314, %1315 : tensor<1x160x16x16xbf16>
    %1317 = stablehlo.reshape %1316 : (tensor<1x160x16x16xbf16>) -> tensor<1x160x256xbf16>
    %1318 = stablehlo.transpose %1317, dims = [0, 2, 1] : (tensor<1x160x256xbf16>) -> tensor<1x256x160xbf16>
    %1319 = stablehlo.convert %1318 : (tensor<1x256x160xbf16>) -> tensor<1x256x160xf32>
    %1320 = stablehlo.convert %1319 : (tensor<1x256x160xf32>) -> tensor<1x256x160xf64>
    %1321 = stablehlo.reduce(%1320 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1322 = stablehlo.reshape %1321 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1324 = stablehlo.broadcast_in_dim %1221, dims = [] : (tensor<f64>) -> tensor<1x256x1xf64>
    %1325 = stablehlo.divide %1323, %1324 : tensor<1x256x1xf64>
    %1326 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2] : (tensor<1x256x160xf64>) -> tensor<1x256x160xf64>
    %1327 = stablehlo.broadcast_in_dim %1325, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x160xf64>
    %1328 = stablehlo.subtract %1326, %1327 : tensor<1x256x160xf64>
    %1329 = stablehlo.multiply %1328, %1328 : tensor<1x256x160xf64>
    %1330 = stablehlo.reduce(%1329 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1331 = stablehlo.reshape %1330 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1332 = stablehlo.broadcast_in_dim %1331, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1333 = stablehlo.divide %1332, %1324 : tensor<1x256x1xf64>
    %1334 = stablehlo.convert %1333 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1335 = stablehlo.reduce(%1319 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1336 = stablehlo.reshape %1335 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1337 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1338 = stablehlo.broadcast_in_dim %1237, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %1339 = stablehlo.divide %1337, %1338 : tensor<1x256x1xf32>
    %1340 = stablehlo.broadcast_in_dim %1334, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1341 = stablehlo.add %1340, %136 : tensor<1x256x1xf32>
    %1342 = stablehlo.rsqrt %1341 : tensor<1x256x1xf32>
    %1343 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1344 = stablehlo.broadcast_in_dim %1339, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x160xf32>
    %1345 = stablehlo.subtract %1343, %1344 : tensor<1x256x160xf32>
    %1346 = stablehlo.broadcast_in_dim %1345, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1347 = stablehlo.broadcast_in_dim %1342, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x160xf32>
    %1348 = stablehlo.multiply %1346, %1347 : tensor<1x256x160xf32>
    %1349 = stablehlo.convert %arg65 : (tensor<160xbf16>) -> tensor<160xf32>
    %1350 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1351 = stablehlo.broadcast_in_dim %1349, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
    %1352 = stablehlo.multiply %1350, %1351 : tensor<1x256x160xf32>
    %1353 = stablehlo.convert %arg66 : (tensor<160xbf16>) -> tensor<160xf32>
    %1354 = stablehlo.broadcast_in_dim %1352, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1355 = stablehlo.broadcast_in_dim %1353, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
    %1356 = stablehlo.add %1354, %1355 : tensor<1x256x160xf32>
    %1357 = stablehlo.convert %1356 : (tensor<1x256x160xf32>) -> tensor<1x256x160xbf16>
    %1358 = stablehlo.reshape %1357 : (tensor<1x256x160xbf16>) -> tensor<256x160xbf16>
    %1359 = stablehlo.convert %1358 : (tensor<256x160xbf16>) -> tensor<256x160xf32>
    %1360 = stablehlo.dot_general %1359, %arg158, contracting_dims = [1] x [0] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1362 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<256x160xf32>
    %1363 = stablehlo.multiply %1361, %1362 : tensor<256x160xf32>
    %1364 = stablehlo.broadcast_in_dim %1363, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1365 = stablehlo.broadcast_in_dim %arg159, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
    %1366 = stablehlo.add %1364, %1365 : tensor<256x160xf32>
    %1367 = stablehlo.convert %1366 : (tensor<256x160xf32>) -> tensor<256x160xbf16>
    %1368 = stablehlo.reshape %1367 : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %1369 = stablehlo.reshape %1368 : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %1370 = stablehlo.transpose %1369, dims = [0, 2, 1, 3] : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %1371 = stablehlo.dot_general %1359, %arg160, contracting_dims = [1] x [0] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %1372 = stablehlo.broadcast_in_dim %1371, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1373 = stablehlo.multiply %1372, %1362 : tensor<256x160xf32>
    %1374 = stablehlo.broadcast_in_dim %1373, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1375 = stablehlo.broadcast_in_dim %arg161, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
    %1376 = stablehlo.add %1374, %1375 : tensor<256x160xf32>
    %1377 = stablehlo.convert %1376 : (tensor<256x160xf32>) -> tensor<256x160xbf16>
    %1378 = stablehlo.reshape %1377 : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %1379 = stablehlo.reshape %1378 : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %1380 = stablehlo.transpose %1379, dims = [0, 2, 1, 3] : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %1381 = stablehlo.transpose %1370, dims = [0, 1, 3, 2] : (tensor<1x5x256x32xbf16>) -> tensor<1x5x32x256xbf16>
    %1382 = stablehlo.reshape %1309 : (tensor<1x5x1024x32xbf16>) -> tensor<5x1024x32xbf16>
    %1383 = stablehlo.reshape %1381 : (tensor<1x5x32x256xbf16>) -> tensor<5x32x256xbf16>
    %1384 = stablehlo.broadcast_in_dim %1383, dims = [0, 1, 2] : (tensor<5x32x256xbf16>) -> tensor<5x32x256xbf16>
    %1385 = stablehlo.dot_general %1382, %1384, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x1024x32xbf16>, tensor<5x32x256xbf16>) -> tensor<5x1024x256xbf16>
    %1386 = stablehlo.reshape %1385 : (tensor<5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %1387 = stablehlo.broadcast_in_dim %1386, dims = [0, 1, 2, 3] : (tensor<1x5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %1388 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<bf16>) -> tensor<1x5x1024x256xbf16>
    %1389 = stablehlo.divide %1387, %1388 : tensor<1x5x1024x256xbf16>
    %1390 = stablehlo.convert %1389 : (tensor<1x5x1024x256xbf16>) -> tensor<1x5x1024x256xf32>
    %1391 = stablehlo.reduce(%1390 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %1392 = stablehlo.reshape %1391 : (tensor<1x5x1024xf32>) -> tensor<1x5x1024x1xf32>
    %1393 = stablehlo.broadcast_in_dim %1390, dims = [0, 1, 2, 3] : (tensor<1x5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %1394 = stablehlo.broadcast_in_dim %1392, dims = [0, 1, 2, 3] : (tensor<1x5x1024x1xf32>) -> tensor<1x5x1024x256xf32>
    %1395 = stablehlo.subtract %1393, %1394 : tensor<1x5x1024x256xf32>
    %1396 = stablehlo.exponential %1395 : tensor<1x5x1024x256xf32>
    %1397 = stablehlo.reduce(%1396 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %1398 = stablehlo.reshape %1397 : (tensor<1x5x1024xf32>) -> tensor<1x5x1024x1xf32>
    %1399 = stablehlo.broadcast_in_dim %1396, dims = [0, 1, 2, 3] : (tensor<1x5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %1400 = stablehlo.broadcast_in_dim %1398, dims = [0, 1, 2, 3] : (tensor<1x5x1024x1xf32>) -> tensor<1x5x1024x256xf32>
    %1401 = stablehlo.divide %1399, %1400 : tensor<1x5x1024x256xf32>
    %1402 = stablehlo.convert %1401 : (tensor<1x5x1024x256xf32>) -> tensor<1x5x1024x256xbf16>
    %1403 = stablehlo.reshape %1402 : (tensor<1x5x1024x256xbf16>) -> tensor<5x1024x256xbf16>
    %1404 = stablehlo.reshape %1380 : (tensor<1x5x256x32xbf16>) -> tensor<5x256x32xbf16>
    %1405 = stablehlo.broadcast_in_dim %1404, dims = [0, 1, 2] : (tensor<5x256x32xbf16>) -> tensor<5x256x32xbf16>
    %1406 = stablehlo.dot_general %1403, %1405, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x1024x256xbf16>, tensor<5x256x32xbf16>) -> tensor<5x1024x32xbf16>
    %1407 = stablehlo.reshape %1406 : (tensor<5x1024x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %1408 = stablehlo.transpose %1407, dims = [0, 2, 1, 3] : (tensor<1x5x1024x32xbf16>) -> tensor<1x1024x5x32xbf16>
    %1409 = stablehlo.reshape %1408 : (tensor<1x1024x5x32xbf16>) -> tensor<1x1024x160xbf16>
    %1410 = stablehlo.reshape %1409 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1411 = stablehlo.convert %1410 : (tensor<1024x160xbf16>) -> tensor<1024x160xf32>
    %1412 = stablehlo.dot_general %1411, %arg162, contracting_dims = [1] x [0] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %1413 = stablehlo.broadcast_in_dim %1412, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1414 = stablehlo.multiply %1413, %1301 : tensor<1024x160xf32>
    %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1416 = stablehlo.broadcast_in_dim %arg163, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1417 = stablehlo.add %1415, %1416 : tensor<1024x160xf32>
    %1418 = stablehlo.convert %1417 : (tensor<1024x160xf32>) -> tensor<1024x160xbf16>
    %1419 = stablehlo.reshape %1418 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1420 = stablehlo.add %1419, %1259 : tensor<1x1024x160xbf16>
    %1421 = stablehlo.convert %1420 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xf32>
    %1422 = stablehlo.convert %1421 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf64>
    %1423 = stablehlo.reduce(%1422 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1424 = stablehlo.reshape %1423 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1425 = stablehlo.broadcast_in_dim %1424, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1426 = stablehlo.divide %1425, %1223 : tensor<1x1024x1xf64>
    %1427 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2] : (tensor<1x1024x160xf64>) -> tensor<1x1024x160xf64>
    %1428 = stablehlo.broadcast_in_dim %1426, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x160xf64>
    %1429 = stablehlo.subtract %1427, %1428 : tensor<1x1024x160xf64>
    %1430 = stablehlo.multiply %1429, %1429 : tensor<1x1024x160xf64>
    %1431 = stablehlo.reduce(%1430 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1432 = stablehlo.reshape %1431 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1434 = stablehlo.divide %1433, %1223 : tensor<1x1024x1xf64>
    %1435 = stablehlo.convert %1434 : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf32>
    %1436 = stablehlo.reduce(%1421 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1437 = stablehlo.reshape %1436 : (tensor<1x1024xf32>) -> tensor<1x1024x1xf32>
    %1438 = stablehlo.broadcast_in_dim %1437, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1439 = stablehlo.divide %1438, %1239 : tensor<1x1024x1xf32>
    %1440 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1441 = stablehlo.add %1440, %1242 : tensor<1x1024x1xf32>
    %1442 = stablehlo.rsqrt %1441 : tensor<1x1024x1xf32>
    %1443 = stablehlo.broadcast_in_dim %1421, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1444 = stablehlo.broadcast_in_dim %1439, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1445 = stablehlo.subtract %1443, %1444 : tensor<1x1024x160xf32>
    %1446 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1447 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1448 = stablehlo.multiply %1446, %1447 : tensor<1x1024x160xf32>
    %1449 = stablehlo.convert %arg67 : (tensor<160xbf16>) -> tensor<160xf32>
    %1450 = stablehlo.broadcast_in_dim %1448, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1451 = stablehlo.broadcast_in_dim %1449, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1452 = stablehlo.multiply %1450, %1451 : tensor<1x1024x160xf32>
    %1453 = stablehlo.convert %arg68 : (tensor<160xbf16>) -> tensor<160xf32>
    %1454 = stablehlo.broadcast_in_dim %1452, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1455 = stablehlo.broadcast_in_dim %1453, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1456 = stablehlo.add %1454, %1455 : tensor<1x1024x160xf32>
    %1457 = stablehlo.convert %1456 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xbf16>
    %1458 = stablehlo.reshape %1457 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1459 = stablehlo.convert %1458 : (tensor<1024x160xbf16>) -> tensor<1024x160xf32>
    %1460 = stablehlo.dot_general %1459, %arg164, contracting_dims = [1] x [0] : (tensor<1024x160xf32>, tensor<160x640xf32>) -> tensor<1024x640xf32>
    %1461 = stablehlo.broadcast_in_dim %1460, dims = [0, 1] : (tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1462 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<1024x640xf32>
    %1463 = stablehlo.multiply %1461, %1462 : tensor<1024x640xf32>
    %1464 = stablehlo.broadcast_in_dim %1463, dims = [0, 1] : (tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1465 = stablehlo.broadcast_in_dim %arg165, dims = [1] : (tensor<640xf32>) -> tensor<1024x640xf32>
    %1466 = stablehlo.add %1464, %1465 : tensor<1024x640xf32>
    %1467 = stablehlo.convert %1466 : (tensor<1024x640xf32>) -> tensor<1024x640xbf16>
    %1468 = stablehlo.reshape %1467 : (tensor<1024x640xbf16>) -> tensor<1x1024x640xbf16>
    %1469 = stablehlo.transpose %1468, dims = [0, 2, 1] : (tensor<1x1024x640xbf16>) -> tensor<1x640x1024xbf16>
    %1470 = stablehlo.reshape %1469 : (tensor<1x640x1024xbf16>) -> tensor<1x640x32x32xbf16>
    %1471 = stablehlo.convolution(%1470, %arg69) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<1x640x32x32xbf16>, tensor<640x1x3x3xbf16>) -> tensor<1x640x32x32xbf16>
    %1472 = stablehlo.reshape %arg70 : (tensor<640xbf16>) -> tensor<640x1x1xbf16>
    %1473 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2, 3] : (tensor<1x640x32x32xbf16>) -> tensor<1x640x32x32xbf16>
    %1474 = stablehlo.broadcast_in_dim %1472, dims = [1, 2, 3] : (tensor<640x1x1xbf16>) -> tensor<1x640x32x32xbf16>
    %1475 = stablehlo.add %1473, %1474 : tensor<1x640x32x32xbf16>
    %1476 = stablehlo.reshape %1475 : (tensor<1x640x32x32xbf16>) -> tensor<1x640x1024xbf16>
    %1477 = stablehlo.transpose %1476, dims = [0, 2, 1] : (tensor<1x640x1024xbf16>) -> tensor<1x1024x640xbf16>
    %1478 = stablehlo.multiply %1477, %cst_42 : tensor<1x1024x640xbf16>
    %1479 = stablehlo.rsqrt %cst_41 : tensor<1x1024x640xbf16>
    %1480 = stablehlo.multiply %1477, %1479 : tensor<1x1024x640xbf16>
    %1481 = stablehlo.convert %1480 : (tensor<1x1024x640xbf16>) -> tensor<1x1024x640xf32>
    %1482 = stablehlo.clamp %cst_43, %1481, %cst_44 : tensor<1x1024x640xf32>
    %1483 = stablehlo.multiply %1482, %1482 : tensor<1x1024x640xf32>
    %1484 = stablehlo.multiply %cst_45, %1483 : tensor<1x1024x640xf32>
    %1485 = stablehlo.add %1484, %cst_46 : tensor<1x1024x640xf32>
    %1486 = stablehlo.multiply %1485, %1483 : tensor<1x1024x640xf32>
    %1487 = stablehlo.add %1486, %cst_47 : tensor<1x1024x640xf32>
    %1488 = stablehlo.multiply %1487, %1483 : tensor<1x1024x640xf32>
    %1489 = stablehlo.add %1488, %cst_48 : tensor<1x1024x640xf32>
    %1490 = stablehlo.multiply %1489, %1483 : tensor<1x1024x640xf32>
    %1491 = stablehlo.add %1490, %cst_49 : tensor<1x1024x640xf32>
    %1492 = stablehlo.multiply %1491, %1483 : tensor<1x1024x640xf32>
    %1493 = stablehlo.add %1492, %cst_50 : tensor<1x1024x640xf32>
    %1494 = stablehlo.multiply %1493, %1483 : tensor<1x1024x640xf32>
    %1495 = stablehlo.add %1494, %cst_51 : tensor<1x1024x640xf32>
    %1496 = stablehlo.multiply %cst_52, %1483 : tensor<1x1024x640xf32>
    %1497 = stablehlo.add %1496, %cst_53 : tensor<1x1024x640xf32>
    %1498 = stablehlo.multiply %1497, %1483 : tensor<1x1024x640xf32>
    %1499 = stablehlo.add %1498, %cst_54 : tensor<1x1024x640xf32>
    %1500 = stablehlo.multiply %1499, %1483 : tensor<1x1024x640xf32>
    %1501 = stablehlo.add %1500, %cst_55 : tensor<1x1024x640xf32>
    %1502 = stablehlo.multiply %1501, %1483 : tensor<1x1024x640xf32>
    %1503 = stablehlo.add %1502, %cst_56 : tensor<1x1024x640xf32>
    %1504 = stablehlo.multiply %1482, %1495 : tensor<1x1024x640xf32>
    %1505 = stablehlo.divide %1504, %1503 : tensor<1x1024x640xf32>
    %1506 = stablehlo.clamp %cst_57, %1505, %cst_58 : tensor<1x1024x640xf32>
    %1507 = stablehlo.convert %1506 : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xbf16>
    %1508 = stablehlo.add %1507, %cst_40 : tensor<1x1024x640xbf16>
    %1509 = stablehlo.multiply %1508, %1478 : tensor<1x1024x640xbf16>
    %1510 = stablehlo.reshape %1509 : (tensor<1x1024x640xbf16>) -> tensor<1024x640xbf16>
    %1511 = stablehlo.dot_general %1510, %arg166, contracting_dims = [1] x [0] : (tensor<1024x640xbf16>, tensor<640x160xbf16>) -> tensor<1024x160xbf16>
    %1512 = stablehlo.reshape %1511 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1513 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2] : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1514 = stablehlo.broadcast_in_dim %arg71, dims = [2] : (tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %1515 = stablehlo.add %1513, %1514 : tensor<1x1024x160xbf16>
    %1516 = stablehlo.reshape %1515 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1517 = stablehlo.reshape %1516 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1518 = stablehlo.add %1517, %1420 : tensor<1x1024x160xbf16>
    %1519 = stablehlo.convert %1518 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xf32>
    %1520 = stablehlo.convert %1519 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf64>
    %1521 = stablehlo.reduce(%1520 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1522 = stablehlo.reshape %1521 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1523 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1524 = stablehlo.divide %1523, %1223 : tensor<1x1024x1xf64>
    %1525 = stablehlo.broadcast_in_dim %1520, dims = [0, 1, 2] : (tensor<1x1024x160xf64>) -> tensor<1x1024x160xf64>
    %1526 = stablehlo.broadcast_in_dim %1524, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x160xf64>
    %1527 = stablehlo.subtract %1525, %1526 : tensor<1x1024x160xf64>
    %1528 = stablehlo.multiply %1527, %1527 : tensor<1x1024x160xf64>
    %1529 = stablehlo.reduce(%1528 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1530 = stablehlo.reshape %1529 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1531 = stablehlo.broadcast_in_dim %1530, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1532 = stablehlo.divide %1531, %1223 : tensor<1x1024x1xf64>
    %1533 = stablehlo.convert %1532 : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf32>
    %1534 = stablehlo.reduce(%1519 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1535 = stablehlo.reshape %1534 : (tensor<1x1024xf32>) -> tensor<1x1024x1xf32>
    %1536 = stablehlo.broadcast_in_dim %1535, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1537 = stablehlo.divide %1536, %1239 : tensor<1x1024x1xf32>
    %1538 = stablehlo.broadcast_in_dim %1533, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1539 = stablehlo.add %1538, %1242 : tensor<1x1024x1xf32>
    %1540 = stablehlo.rsqrt %1539 : tensor<1x1024x1xf32>
    %1541 = stablehlo.broadcast_in_dim %1519, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1542 = stablehlo.broadcast_in_dim %1537, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1543 = stablehlo.subtract %1541, %1542 : tensor<1x1024x160xf32>
    %1544 = stablehlo.broadcast_in_dim %1543, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1545 = stablehlo.broadcast_in_dim %1540, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1546 = stablehlo.multiply %1544, %1545 : tensor<1x1024x160xf32>
    %1547 = stablehlo.convert %arg72 : (tensor<160xbf16>) -> tensor<160xf32>
    %1548 = stablehlo.broadcast_in_dim %1546, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1549 = stablehlo.broadcast_in_dim %1547, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1550 = stablehlo.multiply %1548, %1549 : tensor<1x1024x160xf32>
    %1551 = stablehlo.convert %arg73 : (tensor<160xbf16>) -> tensor<160xf32>
    %1552 = stablehlo.broadcast_in_dim %1550, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1553 = stablehlo.broadcast_in_dim %1551, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1554 = stablehlo.add %1552, %1553 : tensor<1x1024x160xf32>
    %1555 = stablehlo.convert %1554 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xbf16>
    %1556 = stablehlo.reshape %1555 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1557 = stablehlo.convert %1556 : (tensor<1024x160xbf16>) -> tensor<1024x160xf32>
    %1558 = stablehlo.dot_general %1557, %arg167, contracting_dims = [1] x [0] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1560 = stablehlo.multiply %1559, %1301 : tensor<1024x160xf32>
    %1561 = stablehlo.broadcast_in_dim %1560, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1562 = stablehlo.broadcast_in_dim %arg168, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1563 = stablehlo.add %1561, %1562 : tensor<1024x160xf32>
    %1564 = stablehlo.convert %1563 : (tensor<1024x160xf32>) -> tensor<1024x160xbf16>
    %1565 = stablehlo.reshape %1564 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1566 = stablehlo.reshape %1565 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x5x32xbf16>
    %1567 = stablehlo.transpose %1566, dims = [0, 2, 1, 3] : (tensor<1x1024x5x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %1568 = stablehlo.transpose %1555, dims = [0, 2, 1] : (tensor<1x1024x160xbf16>) -> tensor<1x160x1024xbf16>
    %1569 = stablehlo.reshape %1568 : (tensor<1x160x1024xbf16>) -> tensor<1x160x32x32xbf16>
    %1570 = stablehlo.convolution(%1569, %arg74) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x160x32x32xbf16>, tensor<160x160x2x2xbf16>) -> tensor<1x160x16x16xbf16>
    %1571 = stablehlo.reshape %arg75 : (tensor<160xbf16>) -> tensor<160x1x1xbf16>
    %1572 = stablehlo.broadcast_in_dim %1570, dims = [0, 1, 2, 3] : (tensor<1x160x16x16xbf16>) -> tensor<1x160x16x16xbf16>
    %1573 = stablehlo.broadcast_in_dim %1571, dims = [1, 2, 3] : (tensor<160x1x1xbf16>) -> tensor<1x160x16x16xbf16>
    %1574 = stablehlo.add %1572, %1573 : tensor<1x160x16x16xbf16>
    %1575 = stablehlo.reshape %1574 : (tensor<1x160x16x16xbf16>) -> tensor<1x160x256xbf16>
    %1576 = stablehlo.transpose %1575, dims = [0, 2, 1] : (tensor<1x160x256xbf16>) -> tensor<1x256x160xbf16>
    %1577 = stablehlo.convert %1576 : (tensor<1x256x160xbf16>) -> tensor<1x256x160xf32>
    %1578 = stablehlo.convert %1577 : (tensor<1x256x160xf32>) -> tensor<1x256x160xf64>
    %1579 = stablehlo.reduce(%1578 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1580 = stablehlo.reshape %1579 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1581 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1582 = stablehlo.divide %1581, %1324 : tensor<1x256x1xf64>
    %1583 = stablehlo.broadcast_in_dim %1578, dims = [0, 1, 2] : (tensor<1x256x160xf64>) -> tensor<1x256x160xf64>
    %1584 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x160xf64>
    %1585 = stablehlo.subtract %1583, %1584 : tensor<1x256x160xf64>
    %1586 = stablehlo.multiply %1585, %1585 : tensor<1x256x160xf64>
    %1587 = stablehlo.reduce(%1586 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1588 = stablehlo.reshape %1587 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1589 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1590 = stablehlo.divide %1589, %1324 : tensor<1x256x1xf64>
    %1591 = stablehlo.convert %1590 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1592 = stablehlo.reduce(%1577 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1593 = stablehlo.reshape %1592 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1594 = stablehlo.broadcast_in_dim %1593, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1595 = stablehlo.divide %1594, %1338 : tensor<1x256x1xf32>
    %1596 = stablehlo.broadcast_in_dim %1591, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1597 = stablehlo.add %1596, %136 : tensor<1x256x1xf32>
    %1598 = stablehlo.rsqrt %1597 : tensor<1x256x1xf32>
    %1599 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1600 = stablehlo.broadcast_in_dim %1595, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x160xf32>
    %1601 = stablehlo.subtract %1599, %1600 : tensor<1x256x160xf32>
    %1602 = stablehlo.broadcast_in_dim %1601, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1603 = stablehlo.broadcast_in_dim %1598, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x160xf32>
    %1604 = stablehlo.multiply %1602, %1603 : tensor<1x256x160xf32>
    %1605 = stablehlo.convert %arg76 : (tensor<160xbf16>) -> tensor<160xf32>
    %1606 = stablehlo.broadcast_in_dim %1604, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1607 = stablehlo.broadcast_in_dim %1605, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
    %1608 = stablehlo.multiply %1606, %1607 : tensor<1x256x160xf32>
    %1609 = stablehlo.convert %arg77 : (tensor<160xbf16>) -> tensor<160xf32>
    %1610 = stablehlo.broadcast_in_dim %1608, dims = [0, 1, 2] : (tensor<1x256x160xf32>) -> tensor<1x256x160xf32>
    %1611 = stablehlo.broadcast_in_dim %1609, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
    %1612 = stablehlo.add %1610, %1611 : tensor<1x256x160xf32>
    %1613 = stablehlo.convert %1612 : (tensor<1x256x160xf32>) -> tensor<1x256x160xbf16>
    %1614 = stablehlo.reshape %1613 : (tensor<1x256x160xbf16>) -> tensor<256x160xbf16>
    %1615 = stablehlo.convert %1614 : (tensor<256x160xbf16>) -> tensor<256x160xf32>
    %1616 = stablehlo.dot_general %1615, %arg169, contracting_dims = [1] x [0] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %1617 = stablehlo.broadcast_in_dim %1616, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1618 = stablehlo.multiply %1617, %1362 : tensor<256x160xf32>
    %1619 = stablehlo.broadcast_in_dim %1618, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1620 = stablehlo.broadcast_in_dim %arg170, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
    %1621 = stablehlo.add %1619, %1620 : tensor<256x160xf32>
    %1622 = stablehlo.convert %1621 : (tensor<256x160xf32>) -> tensor<256x160xbf16>
    %1623 = stablehlo.reshape %1622 : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %1624 = stablehlo.reshape %1623 : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %1625 = stablehlo.transpose %1624, dims = [0, 2, 1, 3] : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %1626 = stablehlo.dot_general %1615, %arg171, contracting_dims = [1] x [0] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %1627 = stablehlo.broadcast_in_dim %1626, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1628 = stablehlo.multiply %1627, %1362 : tensor<256x160xf32>
    %1629 = stablehlo.broadcast_in_dim %1628, dims = [0, 1] : (tensor<256x160xf32>) -> tensor<256x160xf32>
    %1630 = stablehlo.broadcast_in_dim %arg172, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
    %1631 = stablehlo.add %1629, %1630 : tensor<256x160xf32>
    %1632 = stablehlo.convert %1631 : (tensor<256x160xf32>) -> tensor<256x160xbf16>
    %1633 = stablehlo.reshape %1632 : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %1634 = stablehlo.reshape %1633 : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %1635 = stablehlo.transpose %1634, dims = [0, 2, 1, 3] : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %1636 = stablehlo.transpose %1625, dims = [0, 1, 3, 2] : (tensor<1x5x256x32xbf16>) -> tensor<1x5x32x256xbf16>
    %1637 = stablehlo.reshape %1567 : (tensor<1x5x1024x32xbf16>) -> tensor<5x1024x32xbf16>
    %1638 = stablehlo.reshape %1636 : (tensor<1x5x32x256xbf16>) -> tensor<5x32x256xbf16>
    %1639 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2] : (tensor<5x32x256xbf16>) -> tensor<5x32x256xbf16>
    %1640 = stablehlo.dot_general %1637, %1639, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x1024x32xbf16>, tensor<5x32x256xbf16>) -> tensor<5x1024x256xbf16>
    %1641 = stablehlo.reshape %1640 : (tensor<5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %1642 = stablehlo.broadcast_in_dim %1641, dims = [0, 1, 2, 3] : (tensor<1x5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %1643 = stablehlo.divide %1642, %1388 : tensor<1x5x1024x256xbf16>
    %1644 = stablehlo.convert %1643 : (tensor<1x5x1024x256xbf16>) -> tensor<1x5x1024x256xf32>
    %1645 = stablehlo.reduce(%1644 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %1646 = stablehlo.reshape %1645 : (tensor<1x5x1024xf32>) -> tensor<1x5x1024x1xf32>
    %1647 = stablehlo.broadcast_in_dim %1644, dims = [0, 1, 2, 3] : (tensor<1x5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %1648 = stablehlo.broadcast_in_dim %1646, dims = [0, 1, 2, 3] : (tensor<1x5x1024x1xf32>) -> tensor<1x5x1024x256xf32>
    %1649 = stablehlo.subtract %1647, %1648 : tensor<1x5x1024x256xf32>
    %1650 = stablehlo.exponential %1649 : tensor<1x5x1024x256xf32>
    %1651 = stablehlo.reduce(%1650 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %1652 = stablehlo.reshape %1651 : (tensor<1x5x1024xf32>) -> tensor<1x5x1024x1xf32>
    %1653 = stablehlo.broadcast_in_dim %1650, dims = [0, 1, 2, 3] : (tensor<1x5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %1654 = stablehlo.broadcast_in_dim %1652, dims = [0, 1, 2, 3] : (tensor<1x5x1024x1xf32>) -> tensor<1x5x1024x256xf32>
    %1655 = stablehlo.divide %1653, %1654 : tensor<1x5x1024x256xf32>
    %1656 = stablehlo.convert %1655 : (tensor<1x5x1024x256xf32>) -> tensor<1x5x1024x256xbf16>
    %1657 = stablehlo.reshape %1656 : (tensor<1x5x1024x256xbf16>) -> tensor<5x1024x256xbf16>
    %1658 = stablehlo.reshape %1635 : (tensor<1x5x256x32xbf16>) -> tensor<5x256x32xbf16>
    %1659 = stablehlo.broadcast_in_dim %1658, dims = [0, 1, 2] : (tensor<5x256x32xbf16>) -> tensor<5x256x32xbf16>
    %1660 = stablehlo.dot_general %1657, %1659, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x1024x256xbf16>, tensor<5x256x32xbf16>) -> tensor<5x1024x32xbf16>
    %1661 = stablehlo.reshape %1660 : (tensor<5x1024x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %1662 = stablehlo.transpose %1661, dims = [0, 2, 1, 3] : (tensor<1x5x1024x32xbf16>) -> tensor<1x1024x5x32xbf16>
    %1663 = stablehlo.reshape %1662 : (tensor<1x1024x5x32xbf16>) -> tensor<1x1024x160xbf16>
    %1664 = stablehlo.reshape %1663 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1665 = stablehlo.convert %1664 : (tensor<1024x160xbf16>) -> tensor<1024x160xf32>
    %1666 = stablehlo.dot_general %1665, %arg173, contracting_dims = [1] x [0] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %1667 = stablehlo.broadcast_in_dim %1666, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1668 = stablehlo.multiply %1667, %1301 : tensor<1024x160xf32>
    %1669 = stablehlo.broadcast_in_dim %1668, dims = [0, 1] : (tensor<1024x160xf32>) -> tensor<1024x160xf32>
    %1670 = stablehlo.broadcast_in_dim %arg174, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1671 = stablehlo.add %1669, %1670 : tensor<1024x160xf32>
    %1672 = stablehlo.convert %1671 : (tensor<1024x160xf32>) -> tensor<1024x160xbf16>
    %1673 = stablehlo.reshape %1672 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1674 = stablehlo.add %1673, %1518 : tensor<1x1024x160xbf16>
    %1675 = stablehlo.convert %1674 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xf32>
    %1676 = stablehlo.convert %1675 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf64>
    %1677 = stablehlo.reduce(%1676 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1678 = stablehlo.reshape %1677 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1679 = stablehlo.broadcast_in_dim %1678, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1680 = stablehlo.divide %1679, %1223 : tensor<1x1024x1xf64>
    %1681 = stablehlo.broadcast_in_dim %1676, dims = [0, 1, 2] : (tensor<1x1024x160xf64>) -> tensor<1x1024x160xf64>
    %1682 = stablehlo.broadcast_in_dim %1680, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x160xf64>
    %1683 = stablehlo.subtract %1681, %1682 : tensor<1x1024x160xf64>
    %1684 = stablehlo.multiply %1683, %1683 : tensor<1x1024x160xf64>
    %1685 = stablehlo.reduce(%1684 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1686 = stablehlo.reshape %1685 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1687 = stablehlo.broadcast_in_dim %1686, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1688 = stablehlo.divide %1687, %1223 : tensor<1x1024x1xf64>
    %1689 = stablehlo.convert %1688 : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf32>
    %1690 = stablehlo.reduce(%1675 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1691 = stablehlo.reshape %1690 : (tensor<1x1024xf32>) -> tensor<1x1024x1xf32>
    %1692 = stablehlo.broadcast_in_dim %1691, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1693 = stablehlo.divide %1692, %1239 : tensor<1x1024x1xf32>
    %1694 = stablehlo.broadcast_in_dim %1689, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1695 = stablehlo.add %1694, %1242 : tensor<1x1024x1xf32>
    %1696 = stablehlo.rsqrt %1695 : tensor<1x1024x1xf32>
    %1697 = stablehlo.broadcast_in_dim %1675, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1698 = stablehlo.broadcast_in_dim %1693, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1699 = stablehlo.subtract %1697, %1698 : tensor<1x1024x160xf32>
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1701 = stablehlo.broadcast_in_dim %1696, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1702 = stablehlo.multiply %1700, %1701 : tensor<1x1024x160xf32>
    %1703 = stablehlo.convert %arg78 : (tensor<160xbf16>) -> tensor<160xf32>
    %1704 = stablehlo.broadcast_in_dim %1702, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1705 = stablehlo.broadcast_in_dim %1703, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1706 = stablehlo.multiply %1704, %1705 : tensor<1x1024x160xf32>
    %1707 = stablehlo.convert %arg79 : (tensor<160xbf16>) -> tensor<160xf32>
    %1708 = stablehlo.broadcast_in_dim %1706, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1709 = stablehlo.broadcast_in_dim %1707, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1710 = stablehlo.add %1708, %1709 : tensor<1x1024x160xf32>
    %1711 = stablehlo.convert %1710 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xbf16>
    %1712 = stablehlo.reshape %1711 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1713 = stablehlo.convert %1712 : (tensor<1024x160xbf16>) -> tensor<1024x160xf32>
    %1714 = stablehlo.dot_general %1713, %arg175, contracting_dims = [1] x [0] : (tensor<1024x160xf32>, tensor<160x640xf32>) -> tensor<1024x640xf32>
    %1715 = stablehlo.broadcast_in_dim %1714, dims = [0, 1] : (tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1716 = stablehlo.multiply %1715, %1462 : tensor<1024x640xf32>
    %1717 = stablehlo.broadcast_in_dim %1716, dims = [0, 1] : (tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1718 = stablehlo.broadcast_in_dim %arg176, dims = [1] : (tensor<640xf32>) -> tensor<1024x640xf32>
    %1719 = stablehlo.add %1717, %1718 : tensor<1024x640xf32>
    %1720 = stablehlo.convert %1719 : (tensor<1024x640xf32>) -> tensor<1024x640xbf16>
    %1721 = stablehlo.reshape %1720 : (tensor<1024x640xbf16>) -> tensor<1x1024x640xbf16>
    %1722 = stablehlo.transpose %1721, dims = [0, 2, 1] : (tensor<1x1024x640xbf16>) -> tensor<1x640x1024xbf16>
    %1723 = stablehlo.reshape %1722 : (tensor<1x640x1024xbf16>) -> tensor<1x640x32x32xbf16>
    %1724 = stablehlo.convolution(%1723, %arg80) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<1x640x32x32xbf16>, tensor<640x1x3x3xbf16>) -> tensor<1x640x32x32xbf16>
    %1725 = stablehlo.reshape %arg81 : (tensor<640xbf16>) -> tensor<640x1x1xbf16>
    %1726 = stablehlo.broadcast_in_dim %1724, dims = [0, 1, 2, 3] : (tensor<1x640x32x32xbf16>) -> tensor<1x640x32x32xbf16>
    %1727 = stablehlo.broadcast_in_dim %1725, dims = [1, 2, 3] : (tensor<640x1x1xbf16>) -> tensor<1x640x32x32xbf16>
    %1728 = stablehlo.add %1726, %1727 : tensor<1x640x32x32xbf16>
    %1729 = stablehlo.reshape %1728 : (tensor<1x640x32x32xbf16>) -> tensor<1x640x1024xbf16>
    %1730 = stablehlo.transpose %1729, dims = [0, 2, 1] : (tensor<1x640x1024xbf16>) -> tensor<1x1024x640xbf16>
    %1731 = stablehlo.multiply %1730, %cst_42 : tensor<1x1024x640xbf16>
    %1732 = stablehlo.multiply %1730, %1479 : tensor<1x1024x640xbf16>
    %1733 = stablehlo.convert %1732 : (tensor<1x1024x640xbf16>) -> tensor<1x1024x640xf32>
    %1734 = stablehlo.clamp %cst_43, %1733, %cst_44 : tensor<1x1024x640xf32>
    %1735 = stablehlo.multiply %1734, %1734 : tensor<1x1024x640xf32>
    %1736 = stablehlo.multiply %cst_45, %1735 : tensor<1x1024x640xf32>
    %1737 = stablehlo.add %1736, %cst_46 : tensor<1x1024x640xf32>
    %1738 = stablehlo.multiply %1737, %1735 : tensor<1x1024x640xf32>
    %1739 = stablehlo.add %1738, %cst_47 : tensor<1x1024x640xf32>
    %1740 = stablehlo.multiply %1739, %1735 : tensor<1x1024x640xf32>
    %1741 = stablehlo.add %1740, %cst_48 : tensor<1x1024x640xf32>
    %1742 = stablehlo.multiply %1741, %1735 : tensor<1x1024x640xf32>
    %1743 = stablehlo.add %1742, %cst_49 : tensor<1x1024x640xf32>
    %1744 = stablehlo.multiply %1743, %1735 : tensor<1x1024x640xf32>
    %1745 = stablehlo.add %1744, %cst_50 : tensor<1x1024x640xf32>
    %1746 = stablehlo.multiply %1745, %1735 : tensor<1x1024x640xf32>
    %1747 = stablehlo.add %1746, %cst_51 : tensor<1x1024x640xf32>
    %1748 = stablehlo.multiply %cst_52, %1735 : tensor<1x1024x640xf32>
    %1749 = stablehlo.add %1748, %cst_53 : tensor<1x1024x640xf32>
    %1750 = stablehlo.multiply %1749, %1735 : tensor<1x1024x640xf32>
    %1751 = stablehlo.add %1750, %cst_54 : tensor<1x1024x640xf32>
    %1752 = stablehlo.multiply %1751, %1735 : tensor<1x1024x640xf32>
    %1753 = stablehlo.add %1752, %cst_55 : tensor<1x1024x640xf32>
    %1754 = stablehlo.multiply %1753, %1735 : tensor<1x1024x640xf32>
    %1755 = stablehlo.add %1754, %cst_56 : tensor<1x1024x640xf32>
    %1756 = stablehlo.multiply %1734, %1747 : tensor<1x1024x640xf32>
    %1757 = stablehlo.divide %1756, %1755 : tensor<1x1024x640xf32>
    %1758 = stablehlo.clamp %cst_57, %1757, %cst_58 : tensor<1x1024x640xf32>
    %1759 = stablehlo.convert %1758 : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xbf16>
    %1760 = stablehlo.add %1759, %cst_40 : tensor<1x1024x640xbf16>
    %1761 = stablehlo.multiply %1760, %1731 : tensor<1x1024x640xbf16>
    %1762 = stablehlo.reshape %1761 : (tensor<1x1024x640xbf16>) -> tensor<1024x640xbf16>
    %1763 = stablehlo.dot_general %1762, %arg177, contracting_dims = [1] x [0] : (tensor<1024x640xbf16>, tensor<640x160xbf16>) -> tensor<1024x160xbf16>
    %1764 = stablehlo.reshape %1763 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1765 = stablehlo.broadcast_in_dim %1764, dims = [0, 1, 2] : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1766 = stablehlo.broadcast_in_dim %arg82, dims = [2] : (tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %1767 = stablehlo.add %1765, %1766 : tensor<1x1024x160xbf16>
    %1768 = stablehlo.reshape %1767 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %1769 = stablehlo.reshape %1768 : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %1770 = stablehlo.add %1769, %1674 : tensor<1x1024x160xbf16>
    %1771 = stablehlo.convert %1770 : (tensor<1x1024x160xbf16>) -> tensor<1x1024x160xf32>
    %1772 = stablehlo.convert %1771 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf64>
    %1773 = stablehlo.reduce(%1772 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1774 = stablehlo.reshape %1773 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1775 = stablehlo.broadcast_in_dim %1774, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1776 = stablehlo.divide %1775, %1223 : tensor<1x1024x1xf64>
    %1777 = stablehlo.broadcast_in_dim %1772, dims = [0, 1, 2] : (tensor<1x1024x160xf64>) -> tensor<1x1024x160xf64>
    %1778 = stablehlo.broadcast_in_dim %1776, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x160xf64>
    %1779 = stablehlo.subtract %1777, %1778 : tensor<1x1024x160xf64>
    %1780 = stablehlo.multiply %1779, %1779 : tensor<1x1024x160xf64>
    %1781 = stablehlo.reduce(%1780 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf64>, tensor<f64>) -> tensor<1x1024xf64>
    %1782 = stablehlo.reshape %1781 : (tensor<1x1024xf64>) -> tensor<1x1024x1xf64>
    %1783 = stablehlo.broadcast_in_dim %1782, dims = [0, 1, 2] : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %1784 = stablehlo.divide %1783, %1223 : tensor<1x1024x1xf64>
    %1785 = stablehlo.convert %1784 : (tensor<1x1024x1xf64>) -> tensor<1x1024x1xf32>
    %1786 = stablehlo.reduce(%1771 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1787 = stablehlo.reshape %1786 : (tensor<1x1024xf32>) -> tensor<1x1024x1xf32>
    %1788 = stablehlo.broadcast_in_dim %1787, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1789 = stablehlo.divide %1788, %1239 : tensor<1x1024x1xf32>
    %1790 = stablehlo.broadcast_in_dim %1785, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1791 = stablehlo.add %1790, %1242 : tensor<1x1024x1xf32>
    %1792 = stablehlo.rsqrt %1791 : tensor<1x1024x1xf32>
    %1793 = stablehlo.broadcast_in_dim %1771, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1794 = stablehlo.broadcast_in_dim %1789, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1795 = stablehlo.subtract %1793, %1794 : tensor<1x1024x160xf32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1797 = stablehlo.broadcast_in_dim %1792, dims = [0, 1, 2] : (tensor<1x1024x1xf32>) -> tensor<1x1024x160xf32>
    %1798 = stablehlo.multiply %1796, %1797 : tensor<1x1024x160xf32>
    %1799 = stablehlo.convert %arg83 : (tensor<160xbf16>) -> tensor<160xf32>
    %1800 = stablehlo.broadcast_in_dim %1798, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1801 = stablehlo.broadcast_in_dim %1799, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1802 = stablehlo.multiply %1800, %1801 : tensor<1x1024x160xf32>
    %1803 = stablehlo.convert %arg84 : (tensor<160xbf16>) -> tensor<160xf32>
    %1804 = stablehlo.broadcast_in_dim %1802, dims = [0, 1, 2] : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xf32>
    %1805 = stablehlo.broadcast_in_dim %1803, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
    %1806 = stablehlo.add %1804, %1805 : tensor<1x1024x160xf32>
    %1807 = stablehlo.convert %1806 : (tensor<1x1024x160xf32>) -> tensor<1x1024x160xbf16>
    %1808 = stablehlo.reshape %1807 : (tensor<1x1024x160xbf16>) -> tensor<1x32x32x160xbf16>
    %1809 = stablehlo.transpose %1808, dims = [0, 3, 1, 2] : (tensor<1x32x32x160xbf16>) -> tensor<1x160x32x32xbf16>
    %1810 = stablehlo.convolution(%1809, %arg85) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x160x32x32xbf16>, tensor<256x160x3x3xbf16>) -> tensor<1x256x16x16xbf16>
    %1811 = stablehlo.reshape %arg86 : (tensor<256xbf16>) -> tensor<256x1x1xbf16>
    %1812 = stablehlo.broadcast_in_dim %1810, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xbf16>) -> tensor<1x256x16x16xbf16>
    %1813 = stablehlo.broadcast_in_dim %1811, dims = [1, 2, 3] : (tensor<256x1x1xbf16>) -> tensor<1x256x16x16xbf16>
    %1814 = stablehlo.add %1812, %1813 : tensor<1x256x16x16xbf16>
    %1815 = stablehlo.reshape %1814 : (tensor<1x256x16x16xbf16>) -> tensor<1x256x256xbf16>
    %1816 = stablehlo.transpose %1815, dims = [0, 2, 1] : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %1817 = stablehlo.convert %1816 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1818 = stablehlo.convert %1817 : (tensor<1x256x256xf32>) -> tensor<1x256x256xf64>
    %1819 = stablehlo.reduce(%1818 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1820 = stablehlo.reshape %1819 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1821 = stablehlo.convert %cst_85 : (tensor<1xi64>) -> tensor<1xf64>
    %1822 = stablehlo.reshape %1821 : (tensor<1xf64>) -> tensor<f64>
    %1823 = stablehlo.broadcast_in_dim %1820, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1824 = stablehlo.broadcast_in_dim %1822, dims = [] : (tensor<f64>) -> tensor<1x256x1xf64>
    %1825 = stablehlo.divide %1823, %1824 : tensor<1x256x1xf64>
    %1826 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<1x256x256xf64>) -> tensor<1x256x256xf64>
    %1827 = stablehlo.broadcast_in_dim %1825, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x256xf64>
    %1828 = stablehlo.subtract %1826, %1827 : tensor<1x256x256xf64>
    %1829 = stablehlo.multiply %1828, %1828 : tensor<1x256x256xf64>
    %1830 = stablehlo.reduce(%1829 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1831 = stablehlo.reshape %1830 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1833 = stablehlo.divide %1832, %1824 : tensor<1x256x1xf64>
    %1834 = stablehlo.convert %1833 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1835 = stablehlo.reduce(%1817 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1836 = stablehlo.reshape %1835 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1837 = stablehlo.convert %cst_85 : (tensor<1xi64>) -> tensor<1xf32>
    %1838 = stablehlo.reshape %1837 : (tensor<1xf32>) -> tensor<f32>
    %1839 = stablehlo.broadcast_in_dim %1836, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1840 = stablehlo.broadcast_in_dim %1838, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %1841 = stablehlo.divide %1839, %1840 : tensor<1x256x1xf32>
    %1842 = stablehlo.broadcast_in_dim %1834, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1843 = stablehlo.add %1842, %136 : tensor<1x256x1xf32>
    %1844 = stablehlo.rsqrt %1843 : tensor<1x256x1xf32>
    %1845 = stablehlo.broadcast_in_dim %1817, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1846 = stablehlo.broadcast_in_dim %1841, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %1847 = stablehlo.subtract %1845, %1846 : tensor<1x256x256xf32>
    %1848 = stablehlo.broadcast_in_dim %1847, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1849 = stablehlo.broadcast_in_dim %1844, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %1850 = stablehlo.multiply %1848, %1849 : tensor<1x256x256xf32>
    %1851 = stablehlo.convert %arg87 : (tensor<256xbf16>) -> tensor<256xf32>
    %1852 = stablehlo.broadcast_in_dim %1850, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1853 = stablehlo.broadcast_in_dim %1851, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %1854 = stablehlo.multiply %1852, %1853 : tensor<1x256x256xf32>
    %1855 = stablehlo.convert %arg88 : (tensor<256xbf16>) -> tensor<256xf32>
    %1856 = stablehlo.broadcast_in_dim %1854, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1857 = stablehlo.broadcast_in_dim %1855, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %1858 = stablehlo.add %1856, %1857 : tensor<1x256x256xf32>
    %1859 = stablehlo.convert %1858 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1860 = stablehlo.convert %1859 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1861 = stablehlo.convert %1860 : (tensor<1x256x256xf32>) -> tensor<1x256x256xf64>
    %1862 = stablehlo.reduce(%1861 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1863 = stablehlo.reshape %1862 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1864 = stablehlo.broadcast_in_dim %1863, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1865 = stablehlo.divide %1864, %1824 : tensor<1x256x1xf64>
    %1866 = stablehlo.broadcast_in_dim %1861, dims = [0, 1, 2] : (tensor<1x256x256xf64>) -> tensor<1x256x256xf64>
    %1867 = stablehlo.broadcast_in_dim %1865, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x256xf64>
    %1868 = stablehlo.subtract %1866, %1867 : tensor<1x256x256xf64>
    %1869 = stablehlo.multiply %1868, %1868 : tensor<1x256x256xf64>
    %1870 = stablehlo.reduce(%1869 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1871 = stablehlo.reshape %1870 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1872 = stablehlo.broadcast_in_dim %1871, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1873 = stablehlo.divide %1872, %1824 : tensor<1x256x1xf64>
    %1874 = stablehlo.convert %1873 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1875 = stablehlo.reduce(%1860 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1876 = stablehlo.reshape %1875 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1877 = stablehlo.broadcast_in_dim %1876, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1878 = stablehlo.divide %1877, %1840 : tensor<1x256x1xf32>
    %1879 = stablehlo.broadcast_in_dim %1874, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1880 = stablehlo.add %1879, %136 : tensor<1x256x1xf32>
    %1881 = stablehlo.rsqrt %1880 : tensor<1x256x1xf32>
    %1882 = stablehlo.broadcast_in_dim %1860, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1883 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %1884 = stablehlo.subtract %1882, %1883 : tensor<1x256x256xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1886 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %1887 = stablehlo.multiply %1885, %1886 : tensor<1x256x256xf32>
    %1888 = stablehlo.convert %arg89 : (tensor<256xbf16>) -> tensor<256xf32>
    %1889 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1890 = stablehlo.broadcast_in_dim %1888, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %1891 = stablehlo.multiply %1889, %1890 : tensor<1x256x256xf32>
    %1892 = stablehlo.convert %arg90 : (tensor<256xbf16>) -> tensor<256xf32>
    %1893 = stablehlo.broadcast_in_dim %1891, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1894 = stablehlo.broadcast_in_dim %1892, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %1895 = stablehlo.add %1893, %1894 : tensor<1x256x256xf32>
    %1896 = stablehlo.convert %1895 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %1897 = stablehlo.reshape %1896 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1898 = stablehlo.convert %1897 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1899 = stablehlo.dot_general %1898, %arg178, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %1900 = stablehlo.broadcast_in_dim %1899, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1901 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %1902 = stablehlo.multiply %1900, %1901 : tensor<256x256xf32>
    %1903 = stablehlo.broadcast_in_dim %1902, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1904 = stablehlo.broadcast_in_dim %arg179, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1905 = stablehlo.add %1903, %1904 : tensor<256x256xf32>
    %1906 = stablehlo.convert %1905 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1907 = stablehlo.reshape %1906 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1908 = stablehlo.reshape %1907 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1909 = stablehlo.transpose %1908, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1910 = stablehlo.dot_general %1898, %arg180, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %1911 = stablehlo.broadcast_in_dim %1910, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1912 = stablehlo.multiply %1911, %1901 : tensor<256x256xf32>
    %1913 = stablehlo.broadcast_in_dim %1912, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1914 = stablehlo.broadcast_in_dim %arg181, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1915 = stablehlo.add %1913, %1914 : tensor<256x256xf32>
    %1916 = stablehlo.convert %1915 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1917 = stablehlo.reshape %1916 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1918 = stablehlo.reshape %1917 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1919 = stablehlo.transpose %1918, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1920 = stablehlo.dot_general %1898, %arg182, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %1921 = stablehlo.broadcast_in_dim %1920, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1922 = stablehlo.multiply %1921, %1901 : tensor<256x256xf32>
    %1923 = stablehlo.broadcast_in_dim %1922, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1924 = stablehlo.broadcast_in_dim %arg183, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1925 = stablehlo.add %1923, %1924 : tensor<256x256xf32>
    %1926 = stablehlo.convert %1925 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1927 = stablehlo.reshape %1926 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1928 = stablehlo.reshape %1927 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1929 = stablehlo.transpose %1928, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1930 = stablehlo.transpose %1919, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %1931 = stablehlo.reshape %1909 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1932 = stablehlo.reshape %1930 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1933 = stablehlo.broadcast_in_dim %1932, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1934 = stablehlo.dot_general %1931, %1933, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %1935 = stablehlo.reshape %1934 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1936 = stablehlo.broadcast_in_dim %1935, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1937 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<bf16>) -> tensor<1x8x256x256xbf16>
    %1938 = stablehlo.divide %1936, %1937 : tensor<1x8x256x256xbf16>
    %1939 = stablehlo.convert %1938 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %1940 = stablehlo.reduce(%1939 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1941 = stablehlo.reshape %1940 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1942 = stablehlo.broadcast_in_dim %1939, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1943 = stablehlo.broadcast_in_dim %1941, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1944 = stablehlo.subtract %1942, %1943 : tensor<1x8x256x256xf32>
    %1945 = stablehlo.exponential %1944 : tensor<1x8x256x256xf32>
    %1946 = stablehlo.reduce(%1945 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1947 = stablehlo.reshape %1946 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1948 = stablehlo.broadcast_in_dim %1945, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1949 = stablehlo.broadcast_in_dim %1947, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1950 = stablehlo.divide %1948, %1949 : tensor<1x8x256x256xf32>
    %1951 = stablehlo.convert %1950 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %1952 = stablehlo.reshape %1951 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %1953 = stablehlo.reshape %1929 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1954 = stablehlo.broadcast_in_dim %1953, dims = [0, 1, 2] : (tensor<8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1955 = stablehlo.dot_general %1952, %1954, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1956 = stablehlo.reshape %1955 : (tensor<8x256x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1957 = stablehlo.transpose %1956, dims = [0, 2, 1, 3] : (tensor<1x8x256x32xbf16>) -> tensor<1x256x8x32xbf16>
    %1958 = stablehlo.reshape %1957 : (tensor<1x256x8x32xbf16>) -> tensor<1x256x256xbf16>
    %1959 = stablehlo.reshape %1958 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %1960 = stablehlo.convert %1959 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %1961 = stablehlo.dot_general %1960, %arg184, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %1962 = stablehlo.broadcast_in_dim %1961, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1963 = stablehlo.multiply %1962, %1901 : tensor<256x256xf32>
    %1964 = stablehlo.broadcast_in_dim %1963, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1965 = stablehlo.broadcast_in_dim %arg185, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1966 = stablehlo.add %1964, %1965 : tensor<256x256xf32>
    %1967 = stablehlo.convert %1966 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1968 = stablehlo.reshape %1967 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1969 = stablehlo.add %1968, %1859 : tensor<1x256x256xbf16>
    %1970 = stablehlo.convert %1969 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %1971 = stablehlo.convert %1970 : (tensor<1x256x256xf32>) -> tensor<1x256x256xf64>
    %1972 = stablehlo.reduce(%1971 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1973 = stablehlo.reshape %1972 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1974 = stablehlo.broadcast_in_dim %1973, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1975 = stablehlo.divide %1974, %1824 : tensor<1x256x1xf64>
    %1976 = stablehlo.broadcast_in_dim %1971, dims = [0, 1, 2] : (tensor<1x256x256xf64>) -> tensor<1x256x256xf64>
    %1977 = stablehlo.broadcast_in_dim %1975, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x256xf64>
    %1978 = stablehlo.subtract %1976, %1977 : tensor<1x256x256xf64>
    %1979 = stablehlo.multiply %1978, %1978 : tensor<1x256x256xf64>
    %1980 = stablehlo.reduce(%1979 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1981 = stablehlo.reshape %1980 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1982 = stablehlo.broadcast_in_dim %1981, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1983 = stablehlo.divide %1982, %1824 : tensor<1x256x1xf64>
    %1984 = stablehlo.convert %1983 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1985 = stablehlo.reduce(%1970 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1986 = stablehlo.reshape %1985 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1987 = stablehlo.broadcast_in_dim %1986, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1988 = stablehlo.divide %1987, %1840 : tensor<1x256x1xf32>
    %1989 = stablehlo.broadcast_in_dim %1984, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1990 = stablehlo.add %1989, %136 : tensor<1x256x1xf32>
    %1991 = stablehlo.rsqrt %1990 : tensor<1x256x1xf32>
    %1992 = stablehlo.broadcast_in_dim %1970, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1993 = stablehlo.broadcast_in_dim %1988, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %1994 = stablehlo.subtract %1992, %1993 : tensor<1x256x256xf32>
    %1995 = stablehlo.broadcast_in_dim %1994, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %1996 = stablehlo.broadcast_in_dim %1991, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %1997 = stablehlo.multiply %1995, %1996 : tensor<1x256x256xf32>
    %1998 = stablehlo.convert %arg91 : (tensor<256xbf16>) -> tensor<256xf32>
    %1999 = stablehlo.broadcast_in_dim %1997, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2000 = stablehlo.broadcast_in_dim %1998, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2001 = stablehlo.multiply %1999, %2000 : tensor<1x256x256xf32>
    %2002 = stablehlo.convert %arg92 : (tensor<256xbf16>) -> tensor<256xf32>
    %2003 = stablehlo.broadcast_in_dim %2001, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2004 = stablehlo.broadcast_in_dim %2002, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2005 = stablehlo.add %2003, %2004 : tensor<1x256x256xf32>
    %2006 = stablehlo.convert %2005 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %2007 = stablehlo.reshape %2006 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2008 = stablehlo.convert %2007 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %2009 = stablehlo.dot_general %2008, %arg186, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %2010 = stablehlo.broadcast_in_dim %2009, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %2011 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %2012 = stablehlo.multiply %2010, %2011 : tensor<256x1024xf32>
    %2013 = stablehlo.broadcast_in_dim %2012, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %2014 = stablehlo.broadcast_in_dim %arg187, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024xf32>
    %2015 = stablehlo.add %2013, %2014 : tensor<256x1024xf32>
    %2016 = stablehlo.convert %2015 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2017 = stablehlo.reshape %2016 : (tensor<256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %2018 = stablehlo.transpose %2017, dims = [0, 2, 1] : (tensor<1x256x1024xbf16>) -> tensor<1x1024x256xbf16>
    %2019 = stablehlo.reshape %2018 : (tensor<1x1024x256xbf16>) -> tensor<1x1024x16x16xbf16>
    %2020 = stablehlo.convolution(%2019, %arg93) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<1x1024x16x16xbf16>, tensor<1024x1x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %2021 = stablehlo.reshape %arg94 : (tensor<1024xbf16>) -> tensor<1024x1x1xbf16>
    %2022 = stablehlo.broadcast_in_dim %2020, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %2023 = stablehlo.broadcast_in_dim %2021, dims = [1, 2, 3] : (tensor<1024x1x1xbf16>) -> tensor<1x1024x16x16xbf16>
    %2024 = stablehlo.add %2022, %2023 : tensor<1x1024x16x16xbf16>
    %2025 = stablehlo.reshape %2024 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x256xbf16>
    %2026 = stablehlo.transpose %2025, dims = [0, 2, 1] : (tensor<1x1024x256xbf16>) -> tensor<1x256x1024xbf16>
    %2027 = stablehlo.multiply %2026, %cst_61 : tensor<1x256x1024xbf16>
    %2028 = stablehlo.rsqrt %cst_60 : tensor<1x256x1024xbf16>
    %2029 = stablehlo.multiply %2026, %2028 : tensor<1x256x1024xbf16>
    %2030 = stablehlo.convert %2029 : (tensor<1x256x1024xbf16>) -> tensor<1x256x1024xf32>
    %2031 = stablehlo.clamp %cst_62, %2030, %cst_63 : tensor<1x256x1024xf32>
    %2032 = stablehlo.multiply %2031, %2031 : tensor<1x256x1024xf32>
    %2033 = stablehlo.multiply %cst_64, %2032 : tensor<1x256x1024xf32>
    %2034 = stablehlo.add %2033, %cst_65 : tensor<1x256x1024xf32>
    %2035 = stablehlo.multiply %2034, %2032 : tensor<1x256x1024xf32>
    %2036 = stablehlo.add %2035, %cst_66 : tensor<1x256x1024xf32>
    %2037 = stablehlo.multiply %2036, %2032 : tensor<1x256x1024xf32>
    %2038 = stablehlo.add %2037, %cst_67 : tensor<1x256x1024xf32>
    %2039 = stablehlo.multiply %2038, %2032 : tensor<1x256x1024xf32>
    %2040 = stablehlo.add %2039, %cst_68 : tensor<1x256x1024xf32>
    %2041 = stablehlo.multiply %2040, %2032 : tensor<1x256x1024xf32>
    %2042 = stablehlo.add %2041, %cst_69 : tensor<1x256x1024xf32>
    %2043 = stablehlo.multiply %2042, %2032 : tensor<1x256x1024xf32>
    %2044 = stablehlo.add %2043, %cst_70 : tensor<1x256x1024xf32>
    %2045 = stablehlo.multiply %cst_71, %2032 : tensor<1x256x1024xf32>
    %2046 = stablehlo.add %2045, %cst_72 : tensor<1x256x1024xf32>
    %2047 = stablehlo.multiply %2046, %2032 : tensor<1x256x1024xf32>
    %2048 = stablehlo.add %2047, %cst_73 : tensor<1x256x1024xf32>
    %2049 = stablehlo.multiply %2048, %2032 : tensor<1x256x1024xf32>
    %2050 = stablehlo.add %2049, %cst_74 : tensor<1x256x1024xf32>
    %2051 = stablehlo.multiply %2050, %2032 : tensor<1x256x1024xf32>
    %2052 = stablehlo.add %2051, %cst_75 : tensor<1x256x1024xf32>
    %2053 = stablehlo.multiply %2031, %2044 : tensor<1x256x1024xf32>
    %2054 = stablehlo.divide %2053, %2052 : tensor<1x256x1024xf32>
    %2055 = stablehlo.clamp %cst_76, %2054, %cst_77 : tensor<1x256x1024xf32>
    %2056 = stablehlo.convert %2055 : (tensor<1x256x1024xf32>) -> tensor<1x256x1024xbf16>
    %2057 = stablehlo.add %2056, %cst_59 : tensor<1x256x1024xbf16>
    %2058 = stablehlo.multiply %2057, %2027 : tensor<1x256x1024xbf16>
    %2059 = stablehlo.reshape %2058 : (tensor<1x256x1024xbf16>) -> tensor<256x1024xbf16>
    %2060 = stablehlo.dot_general %2059, %arg188, contracting_dims = [1] x [0] : (tensor<256x1024xbf16>, tensor<1024x256xbf16>) -> tensor<256x256xbf16>
    %2061 = stablehlo.reshape %2060 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2062 = stablehlo.broadcast_in_dim %2061, dims = [0, 1, 2] : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %2063 = stablehlo.broadcast_in_dim %arg95, dims = [2] : (tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %2064 = stablehlo.add %2062, %2063 : tensor<1x256x256xbf16>
    %2065 = stablehlo.reshape %2064 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2066 = stablehlo.reshape %2065 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2067 = stablehlo.add %2066, %1969 : tensor<1x256x256xbf16>
    %2068 = stablehlo.convert %2067 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %2069 = stablehlo.convert %2068 : (tensor<1x256x256xf32>) -> tensor<1x256x256xf64>
    %2070 = stablehlo.reduce(%2069 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2071 = stablehlo.reshape %2070 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2072 = stablehlo.broadcast_in_dim %2071, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2073 = stablehlo.divide %2072, %1824 : tensor<1x256x1xf64>
    %2074 = stablehlo.broadcast_in_dim %2069, dims = [0, 1, 2] : (tensor<1x256x256xf64>) -> tensor<1x256x256xf64>
    %2075 = stablehlo.broadcast_in_dim %2073, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x256xf64>
    %2076 = stablehlo.subtract %2074, %2075 : tensor<1x256x256xf64>
    %2077 = stablehlo.multiply %2076, %2076 : tensor<1x256x256xf64>
    %2078 = stablehlo.reduce(%2077 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2079 = stablehlo.reshape %2078 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2080 = stablehlo.broadcast_in_dim %2079, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2081 = stablehlo.divide %2080, %1824 : tensor<1x256x1xf64>
    %2082 = stablehlo.convert %2081 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2083 = stablehlo.reduce(%2068 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2084 = stablehlo.reshape %2083 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2085 = stablehlo.broadcast_in_dim %2084, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2086 = stablehlo.divide %2085, %1840 : tensor<1x256x1xf32>
    %2087 = stablehlo.broadcast_in_dim %2082, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2088 = stablehlo.add %2087, %136 : tensor<1x256x1xf32>
    %2089 = stablehlo.rsqrt %2088 : tensor<1x256x1xf32>
    %2090 = stablehlo.broadcast_in_dim %2068, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2091 = stablehlo.broadcast_in_dim %2086, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %2092 = stablehlo.subtract %2090, %2091 : tensor<1x256x256xf32>
    %2093 = stablehlo.broadcast_in_dim %2092, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2094 = stablehlo.broadcast_in_dim %2089, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %2095 = stablehlo.multiply %2093, %2094 : tensor<1x256x256xf32>
    %2096 = stablehlo.convert %arg96 : (tensor<256xbf16>) -> tensor<256xf32>
    %2097 = stablehlo.broadcast_in_dim %2095, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2098 = stablehlo.broadcast_in_dim %2096, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2099 = stablehlo.multiply %2097, %2098 : tensor<1x256x256xf32>
    %2100 = stablehlo.convert %arg97 : (tensor<256xbf16>) -> tensor<256xf32>
    %2101 = stablehlo.broadcast_in_dim %2099, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2102 = stablehlo.broadcast_in_dim %2100, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2103 = stablehlo.add %2101, %2102 : tensor<1x256x256xf32>
    %2104 = stablehlo.convert %2103 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %2105 = stablehlo.reshape %2104 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2106 = stablehlo.convert %2105 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %2107 = stablehlo.dot_general %2106, %arg189, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %2108 = stablehlo.broadcast_in_dim %2107, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2109 = stablehlo.multiply %2108, %1901 : tensor<256x256xf32>
    %2110 = stablehlo.broadcast_in_dim %2109, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2111 = stablehlo.broadcast_in_dim %arg190, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2112 = stablehlo.add %2110, %2111 : tensor<256x256xf32>
    %2113 = stablehlo.convert %2112 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2114 = stablehlo.reshape %2113 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2115 = stablehlo.reshape %2114 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2116 = stablehlo.transpose %2115, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2117 = stablehlo.dot_general %2106, %arg191, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %2118 = stablehlo.broadcast_in_dim %2117, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2119 = stablehlo.multiply %2118, %1901 : tensor<256x256xf32>
    %2120 = stablehlo.broadcast_in_dim %2119, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2121 = stablehlo.broadcast_in_dim %arg192, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2122 = stablehlo.add %2120, %2121 : tensor<256x256xf32>
    %2123 = stablehlo.convert %2122 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2124 = stablehlo.reshape %2123 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2125 = stablehlo.reshape %2124 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2126 = stablehlo.transpose %2125, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2127 = stablehlo.dot_general %2106, %arg193, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %2128 = stablehlo.broadcast_in_dim %2127, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2129 = stablehlo.multiply %2128, %1901 : tensor<256x256xf32>
    %2130 = stablehlo.broadcast_in_dim %2129, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2131 = stablehlo.broadcast_in_dim %arg194, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2132 = stablehlo.add %2130, %2131 : tensor<256x256xf32>
    %2133 = stablehlo.convert %2132 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2134 = stablehlo.reshape %2133 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2135 = stablehlo.reshape %2134 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2136 = stablehlo.transpose %2135, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2137 = stablehlo.transpose %2126, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %2138 = stablehlo.reshape %2116 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2139 = stablehlo.reshape %2137 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2140 = stablehlo.broadcast_in_dim %2139, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2141 = stablehlo.dot_general %2138, %2140, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %2142 = stablehlo.reshape %2141 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2143 = stablehlo.broadcast_in_dim %2142, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2144 = stablehlo.divide %2143, %1937 : tensor<1x8x256x256xbf16>
    %2145 = stablehlo.convert %2144 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %2146 = stablehlo.reduce(%2145 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2147 = stablehlo.reshape %2146 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2148 = stablehlo.broadcast_in_dim %2145, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2149 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2150 = stablehlo.subtract %2148, %2149 : tensor<1x8x256x256xf32>
    %2151 = stablehlo.exponential %2150 : tensor<1x8x256x256xf32>
    %2152 = stablehlo.reduce(%2151 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2153 = stablehlo.reshape %2152 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2154 = stablehlo.broadcast_in_dim %2151, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2155 = stablehlo.broadcast_in_dim %2153, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2156 = stablehlo.divide %2154, %2155 : tensor<1x8x256x256xf32>
    %2157 = stablehlo.convert %2156 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %2158 = stablehlo.reshape %2157 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %2159 = stablehlo.reshape %2136 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2160 = stablehlo.broadcast_in_dim %2159, dims = [0, 1, 2] : (tensor<8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2161 = stablehlo.dot_general %2158, %2160, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2162 = stablehlo.reshape %2161 : (tensor<8x256x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2163 = stablehlo.transpose %2162, dims = [0, 2, 1, 3] : (tensor<1x8x256x32xbf16>) -> tensor<1x256x8x32xbf16>
    %2164 = stablehlo.reshape %2163 : (tensor<1x256x8x32xbf16>) -> tensor<1x256x256xbf16>
    %2165 = stablehlo.reshape %2164 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2166 = stablehlo.convert %2165 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %2167 = stablehlo.dot_general %2166, %arg195, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %2168 = stablehlo.broadcast_in_dim %2167, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2169 = stablehlo.multiply %2168, %1901 : tensor<256x256xf32>
    %2170 = stablehlo.broadcast_in_dim %2169, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2171 = stablehlo.broadcast_in_dim %arg196, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2172 = stablehlo.add %2170, %2171 : tensor<256x256xf32>
    %2173 = stablehlo.convert %2172 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2174 = stablehlo.reshape %2173 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2175 = stablehlo.add %2174, %2067 : tensor<1x256x256xbf16>
    %2176 = stablehlo.convert %2175 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %2177 = stablehlo.convert %2176 : (tensor<1x256x256xf32>) -> tensor<1x256x256xf64>
    %2178 = stablehlo.reduce(%2177 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2179 = stablehlo.reshape %2178 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2180 = stablehlo.broadcast_in_dim %2179, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2181 = stablehlo.divide %2180, %1824 : tensor<1x256x1xf64>
    %2182 = stablehlo.broadcast_in_dim %2177, dims = [0, 1, 2] : (tensor<1x256x256xf64>) -> tensor<1x256x256xf64>
    %2183 = stablehlo.broadcast_in_dim %2181, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x256xf64>
    %2184 = stablehlo.subtract %2182, %2183 : tensor<1x256x256xf64>
    %2185 = stablehlo.multiply %2184, %2184 : tensor<1x256x256xf64>
    %2186 = stablehlo.reduce(%2185 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2187 = stablehlo.reshape %2186 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2188 = stablehlo.broadcast_in_dim %2187, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2189 = stablehlo.divide %2188, %1824 : tensor<1x256x1xf64>
    %2190 = stablehlo.convert %2189 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2191 = stablehlo.reduce(%2176 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2192 = stablehlo.reshape %2191 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2193 = stablehlo.broadcast_in_dim %2192, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2194 = stablehlo.divide %2193, %1840 : tensor<1x256x1xf32>
    %2195 = stablehlo.broadcast_in_dim %2190, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2196 = stablehlo.add %2195, %136 : tensor<1x256x1xf32>
    %2197 = stablehlo.rsqrt %2196 : tensor<1x256x1xf32>
    %2198 = stablehlo.broadcast_in_dim %2176, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2199 = stablehlo.broadcast_in_dim %2194, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %2200 = stablehlo.subtract %2198, %2199 : tensor<1x256x256xf32>
    %2201 = stablehlo.broadcast_in_dim %2200, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2202 = stablehlo.broadcast_in_dim %2197, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %2203 = stablehlo.multiply %2201, %2202 : tensor<1x256x256xf32>
    %2204 = stablehlo.convert %arg98 : (tensor<256xbf16>) -> tensor<256xf32>
    %2205 = stablehlo.broadcast_in_dim %2203, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2206 = stablehlo.broadcast_in_dim %2204, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2207 = stablehlo.multiply %2205, %2206 : tensor<1x256x256xf32>
    %2208 = stablehlo.convert %arg99 : (tensor<256xbf16>) -> tensor<256xf32>
    %2209 = stablehlo.broadcast_in_dim %2207, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2210 = stablehlo.broadcast_in_dim %2208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2211 = stablehlo.add %2209, %2210 : tensor<1x256x256xf32>
    %2212 = stablehlo.convert %2211 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %2213 = stablehlo.reshape %2212 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2214 = stablehlo.convert %2213 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %2215 = stablehlo.dot_general %2214, %arg197, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %2216 = stablehlo.broadcast_in_dim %2215, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %2217 = stablehlo.multiply %2216, %2011 : tensor<256x1024xf32>
    %2218 = stablehlo.broadcast_in_dim %2217, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %2219 = stablehlo.broadcast_in_dim %arg198, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024xf32>
    %2220 = stablehlo.add %2218, %2219 : tensor<256x1024xf32>
    %2221 = stablehlo.convert %2220 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2222 = stablehlo.reshape %2221 : (tensor<256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %2223 = stablehlo.transpose %2222, dims = [0, 2, 1] : (tensor<1x256x1024xbf16>) -> tensor<1x1024x256xbf16>
    %2224 = stablehlo.reshape %2223 : (tensor<1x1024x256xbf16>) -> tensor<1x1024x16x16xbf16>
    %2225 = stablehlo.convolution(%2224, %arg100) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<1x1024x16x16xbf16>, tensor<1024x1x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %2226 = stablehlo.reshape %arg101 : (tensor<1024xbf16>) -> tensor<1024x1x1xbf16>
    %2227 = stablehlo.broadcast_in_dim %2225, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %2228 = stablehlo.broadcast_in_dim %2226, dims = [1, 2, 3] : (tensor<1024x1x1xbf16>) -> tensor<1x1024x16x16xbf16>
    %2229 = stablehlo.add %2227, %2228 : tensor<1x1024x16x16xbf16>
    %2230 = stablehlo.reshape %2229 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x256xbf16>
    %2231 = stablehlo.transpose %2230, dims = [0, 2, 1] : (tensor<1x1024x256xbf16>) -> tensor<1x256x1024xbf16>
    %2232 = stablehlo.multiply %2231, %cst_61 : tensor<1x256x1024xbf16>
    %2233 = stablehlo.multiply %2231, %2028 : tensor<1x256x1024xbf16>
    %2234 = stablehlo.convert %2233 : (tensor<1x256x1024xbf16>) -> tensor<1x256x1024xf32>
    %2235 = stablehlo.clamp %cst_62, %2234, %cst_63 : tensor<1x256x1024xf32>
    %2236 = stablehlo.multiply %2235, %2235 : tensor<1x256x1024xf32>
    %2237 = stablehlo.multiply %cst_64, %2236 : tensor<1x256x1024xf32>
    %2238 = stablehlo.add %2237, %cst_65 : tensor<1x256x1024xf32>
    %2239 = stablehlo.multiply %2238, %2236 : tensor<1x256x1024xf32>
    %2240 = stablehlo.add %2239, %cst_66 : tensor<1x256x1024xf32>
    %2241 = stablehlo.multiply %2240, %2236 : tensor<1x256x1024xf32>
    %2242 = stablehlo.add %2241, %cst_67 : tensor<1x256x1024xf32>
    %2243 = stablehlo.multiply %2242, %2236 : tensor<1x256x1024xf32>
    %2244 = stablehlo.add %2243, %cst_68 : tensor<1x256x1024xf32>
    %2245 = stablehlo.multiply %2244, %2236 : tensor<1x256x1024xf32>
    %2246 = stablehlo.add %2245, %cst_69 : tensor<1x256x1024xf32>
    %2247 = stablehlo.multiply %2246, %2236 : tensor<1x256x1024xf32>
    %2248 = stablehlo.add %2247, %cst_70 : tensor<1x256x1024xf32>
    %2249 = stablehlo.multiply %cst_71, %2236 : tensor<1x256x1024xf32>
    %2250 = stablehlo.add %2249, %cst_72 : tensor<1x256x1024xf32>
    %2251 = stablehlo.multiply %2250, %2236 : tensor<1x256x1024xf32>
    %2252 = stablehlo.add %2251, %cst_73 : tensor<1x256x1024xf32>
    %2253 = stablehlo.multiply %2252, %2236 : tensor<1x256x1024xf32>
    %2254 = stablehlo.add %2253, %cst_74 : tensor<1x256x1024xf32>
    %2255 = stablehlo.multiply %2254, %2236 : tensor<1x256x1024xf32>
    %2256 = stablehlo.add %2255, %cst_75 : tensor<1x256x1024xf32>
    %2257 = stablehlo.multiply %2235, %2248 : tensor<1x256x1024xf32>
    %2258 = stablehlo.divide %2257, %2256 : tensor<1x256x1024xf32>
    %2259 = stablehlo.clamp %cst_76, %2258, %cst_77 : tensor<1x256x1024xf32>
    %2260 = stablehlo.convert %2259 : (tensor<1x256x1024xf32>) -> tensor<1x256x1024xbf16>
    %2261 = stablehlo.add %2260, %cst_59 : tensor<1x256x1024xbf16>
    %2262 = stablehlo.multiply %2261, %2232 : tensor<1x256x1024xbf16>
    %2263 = stablehlo.reshape %2262 : (tensor<1x256x1024xbf16>) -> tensor<256x1024xbf16>
    %2264 = stablehlo.dot_general %2263, %arg199, contracting_dims = [1] x [0] : (tensor<256x1024xbf16>, tensor<1024x256xbf16>) -> tensor<256x256xbf16>
    %2265 = stablehlo.reshape %2264 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2266 = stablehlo.broadcast_in_dim %2265, dims = [0, 1, 2] : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %2267 = stablehlo.broadcast_in_dim %arg102, dims = [2] : (tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %2268 = stablehlo.add %2266, %2267 : tensor<1x256x256xbf16>
    %2269 = stablehlo.reshape %2268 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2270 = stablehlo.reshape %2269 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2271 = stablehlo.add %2270, %2175 : tensor<1x256x256xbf16>
    %2272 = stablehlo.convert %2271 : (tensor<1x256x256xbf16>) -> tensor<1x256x256xf32>
    %2273 = stablehlo.convert %2272 : (tensor<1x256x256xf32>) -> tensor<1x256x256xf64>
    %2274 = stablehlo.reduce(%2273 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2275 = stablehlo.reshape %2274 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2276 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2277 = stablehlo.divide %2276, %1824 : tensor<1x256x1xf64>
    %2278 = stablehlo.broadcast_in_dim %2273, dims = [0, 1, 2] : (tensor<1x256x256xf64>) -> tensor<1x256x256xf64>
    %2279 = stablehlo.broadcast_in_dim %2277, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x256xf64>
    %2280 = stablehlo.subtract %2278, %2279 : tensor<1x256x256xf64>
    %2281 = stablehlo.multiply %2280, %2280 : tensor<1x256x256xf64>
    %2282 = stablehlo.reduce(%2281 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2283 = stablehlo.reshape %2282 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2284 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2285 = stablehlo.divide %2284, %1824 : tensor<1x256x1xf64>
    %2286 = stablehlo.convert %2285 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2287 = stablehlo.reduce(%2272 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2288 = stablehlo.reshape %2287 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2289 = stablehlo.broadcast_in_dim %2288, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2290 = stablehlo.divide %2289, %1840 : tensor<1x256x1xf32>
    %2291 = stablehlo.broadcast_in_dim %2286, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2292 = stablehlo.add %2291, %136 : tensor<1x256x1xf32>
    %2293 = stablehlo.rsqrt %2292 : tensor<1x256x1xf32>
    %2294 = stablehlo.broadcast_in_dim %2272, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2295 = stablehlo.broadcast_in_dim %2290, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %2296 = stablehlo.subtract %2294, %2295 : tensor<1x256x256xf32>
    %2297 = stablehlo.broadcast_in_dim %2296, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2298 = stablehlo.broadcast_in_dim %2293, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x256xf32>
    %2299 = stablehlo.multiply %2297, %2298 : tensor<1x256x256xf32>
    %2300 = stablehlo.convert %arg103 : (tensor<256xbf16>) -> tensor<256xf32>
    %2301 = stablehlo.broadcast_in_dim %2299, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2302 = stablehlo.broadcast_in_dim %2300, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2303 = stablehlo.multiply %2301, %2302 : tensor<1x256x256xf32>
    %2304 = stablehlo.convert %arg104 : (tensor<256xbf16>) -> tensor<256xf32>
    %2305 = stablehlo.broadcast_in_dim %2303, dims = [0, 1, 2] : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
    %2306 = stablehlo.broadcast_in_dim %2304, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %2307 = stablehlo.add %2305, %2306 : tensor<1x256x256xf32>
    %2308 = stablehlo.convert %2307 : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
    %2309 = stablehlo.reshape %2308 : (tensor<1x256x256xbf16>) -> tensor<1x16x16x256xbf16>
    %2310 = stablehlo.transpose %2309, dims = [0, 3, 1, 2] : (tensor<1x16x16x256xbf16>) -> tensor<1x256x16x16xbf16>
    %2311 = stablehlo.reshape %607 : (tensor<1x32x128x128xbf16>) -> tensor<1x32x16384xbf16>
    %2312 = stablehlo.transpose %2311, dims = [0, 2, 1] : (tensor<1x32x16384xbf16>) -> tensor<1x16384x32xbf16>
    %2313 = stablehlo.reshape %2312 : (tensor<1x16384x32xbf16>) -> tensor<16384x32xbf16>
    %2314 = stablehlo.dot_general %2313, %arg200, contracting_dims = [1] x [0] : (tensor<16384x32xbf16>, tensor<32x256xbf16>) -> tensor<16384x256xbf16>
    %2315 = stablehlo.reshape %2314 : (tensor<16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %2316 = stablehlo.broadcast_in_dim %2315, dims = [0, 1, 2] : (tensor<1x16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %2317 = stablehlo.broadcast_in_dim %arg105, dims = [2] : (tensor<256xbf16>) -> tensor<1x16384x256xbf16>
    %2318 = stablehlo.add %2316, %2317 : tensor<1x16384x256xbf16>
    %2319 = stablehlo.reshape %2318 : (tensor<1x16384x256xbf16>) -> tensor<16384x256xbf16>
    %2320 = stablehlo.reshape %2319 : (tensor<16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %2321 = stablehlo.transpose %2320, dims = [0, 2, 1] : (tensor<1x16384x256xbf16>) -> tensor<1x256x16384xbf16>
    %2322 = stablehlo.reshape %2321 : (tensor<1x256x16384xbf16>) -> tensor<1x256x128x128xbf16>
    %2323 = stablehlo.transpose %2322, dims = [0, 1, 3, 2] : (tensor<1x256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2324 = stablehlo.reshape %2323 : (tensor<1x256x128x128xbf16>) -> tensor<256x128x128xbf16>
    %2325 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1, 2] : (tensor<256x128x128xbf16>) -> tensor<256x128x128xbf16>
    %2326 = stablehlo.dot_general %2324, %2325, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x128xbf16>, tensor<256x128x128xbf16>) -> tensor<256x128x128xbf16>
    %2327 = stablehlo.reshape %2326 : (tensor<256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2328 = stablehlo.transpose %2327, dims = [0, 1, 3, 2] : (tensor<1x256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2329 = stablehlo.reshape %2328 : (tensor<1x256x128x128xbf16>) -> tensor<256x128x128xbf16>
    %2330 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<256x128x128xbf16>) -> tensor<256x128x128xbf16>
    %2331 = stablehlo.dot_general %2329, %2330, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x128xbf16>, tensor<256x128x128xbf16>) -> tensor<256x128x128xbf16>
    %2332 = stablehlo.reshape %2331 : (tensor<256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2333 = stablehlo.reshape %1208 : (tensor<1x64x64x64xbf16>) -> tensor<1x64x4096xbf16>
    %2334 = stablehlo.transpose %2333, dims = [0, 2, 1] : (tensor<1x64x4096xbf16>) -> tensor<1x4096x64xbf16>
    %2335 = stablehlo.reshape %2334 : (tensor<1x4096x64xbf16>) -> tensor<4096x64xbf16>
    %2336 = stablehlo.dot_general %2335, %arg203, contracting_dims = [1] x [0] : (tensor<4096x64xbf16>, tensor<64x256xbf16>) -> tensor<4096x256xbf16>
    %2337 = stablehlo.reshape %2336 : (tensor<4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %2338 = stablehlo.broadcast_in_dim %2337, dims = [0, 1, 2] : (tensor<1x4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %2339 = stablehlo.broadcast_in_dim %arg106, dims = [2] : (tensor<256xbf16>) -> tensor<1x4096x256xbf16>
    %2340 = stablehlo.add %2338, %2339 : tensor<1x4096x256xbf16>
    %2341 = stablehlo.reshape %2340 : (tensor<1x4096x256xbf16>) -> tensor<4096x256xbf16>
    %2342 = stablehlo.reshape %2341 : (tensor<4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %2343 = stablehlo.transpose %2342, dims = [0, 2, 1] : (tensor<1x4096x256xbf16>) -> tensor<1x256x4096xbf16>
    %2344 = stablehlo.reshape %2343 : (tensor<1x256x4096xbf16>) -> tensor<1x256x64x64xbf16>
    %2345 = stablehlo.transpose %2344, dims = [0, 1, 3, 2] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %2346 = stablehlo.reshape %2345 : (tensor<1x256x64x64xbf16>) -> tensor<256x64x64xbf16>
    %2347 = stablehlo.broadcast_in_dim %arg204, dims = [0, 1, 2] : (tensor<256x64x128xbf16>) -> tensor<256x64x128xbf16>
    %2348 = stablehlo.dot_general %2346, %2347, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x64x64xbf16>, tensor<256x64x128xbf16>) -> tensor<256x64x128xbf16>
    %2349 = stablehlo.reshape %2348 : (tensor<256x64x128xbf16>) -> tensor<1x256x64x128xbf16>
    %2350 = stablehlo.transpose %2349, dims = [0, 1, 3, 2] : (tensor<1x256x64x128xbf16>) -> tensor<1x256x128x64xbf16>
    %2351 = stablehlo.reshape %2350 : (tensor<1x256x128x64xbf16>) -> tensor<256x128x64xbf16>
    %2352 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1, 2] : (tensor<256x64x128xbf16>) -> tensor<256x64x128xbf16>
    %2353 = stablehlo.dot_general %2351, %2352, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x64xbf16>, tensor<256x64x128xbf16>) -> tensor<256x128x128xbf16>
    %2354 = stablehlo.reshape %2353 : (tensor<256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2355 = stablehlo.reshape %1809 : (tensor<1x160x32x32xbf16>) -> tensor<1x160x1024xbf16>
    %2356 = stablehlo.transpose %2355, dims = [0, 2, 1] : (tensor<1x160x1024xbf16>) -> tensor<1x1024x160xbf16>
    %2357 = stablehlo.reshape %2356 : (tensor<1x1024x160xbf16>) -> tensor<1024x160xbf16>
    %2358 = stablehlo.dot_general %2357, %arg206, contracting_dims = [1] x [0] : (tensor<1024x160xbf16>, tensor<160x256xbf16>) -> tensor<1024x256xbf16>
    %2359 = stablehlo.reshape %2358 : (tensor<1024x256xbf16>) -> tensor<1x1024x256xbf16>
    %2360 = stablehlo.broadcast_in_dim %2359, dims = [0, 1, 2] : (tensor<1x1024x256xbf16>) -> tensor<1x1024x256xbf16>
    %2361 = stablehlo.broadcast_in_dim %arg107, dims = [2] : (tensor<256xbf16>) -> tensor<1x1024x256xbf16>
    %2362 = stablehlo.add %2360, %2361 : tensor<1x1024x256xbf16>
    %2363 = stablehlo.reshape %2362 : (tensor<1x1024x256xbf16>) -> tensor<1024x256xbf16>
    %2364 = stablehlo.reshape %2363 : (tensor<1024x256xbf16>) -> tensor<1x1024x256xbf16>
    %2365 = stablehlo.transpose %2364, dims = [0, 2, 1] : (tensor<1x1024x256xbf16>) -> tensor<1x256x1024xbf16>
    %2366 = stablehlo.reshape %2365 : (tensor<1x256x1024xbf16>) -> tensor<1x256x32x32xbf16>
    %2367 = stablehlo.transpose %2366, dims = [0, 1, 3, 2] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %2368 = stablehlo.reshape %2367 : (tensor<1x256x32x32xbf16>) -> tensor<256x32x32xbf16>
    %2369 = stablehlo.broadcast_in_dim %arg207, dims = [0, 1, 2] : (tensor<256x32x128xbf16>) -> tensor<256x32x128xbf16>
    %2370 = stablehlo.dot_general %2368, %2369, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x32xbf16>, tensor<256x32x128xbf16>) -> tensor<256x32x128xbf16>
    %2371 = stablehlo.reshape %2370 : (tensor<256x32x128xbf16>) -> tensor<1x256x32x128xbf16>
    %2372 = stablehlo.transpose %2371, dims = [0, 1, 3, 2] : (tensor<1x256x32x128xbf16>) -> tensor<1x256x128x32xbf16>
    %2373 = stablehlo.reshape %2372 : (tensor<1x256x128x32xbf16>) -> tensor<256x128x32xbf16>
    %2374 = stablehlo.broadcast_in_dim %arg208, dims = [0, 1, 2] : (tensor<256x32x128xbf16>) -> tensor<256x32x128xbf16>
    %2375 = stablehlo.dot_general %2373, %2374, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x32xbf16>, tensor<256x32x128xbf16>) -> tensor<256x128x128xbf16>
    %2376 = stablehlo.reshape %2375 : (tensor<256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2377 = stablehlo.reshape %2310 : (tensor<1x256x16x16xbf16>) -> tensor<1x256x256xbf16>
    %2378 = stablehlo.transpose %2377, dims = [0, 2, 1] : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %2379 = stablehlo.reshape %2378 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2380 = stablehlo.dot_general %2379, %arg209, contracting_dims = [1] x [0] : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %2381 = stablehlo.reshape %2380 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2382 = stablehlo.broadcast_in_dim %2381, dims = [0, 1, 2] : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %2383 = stablehlo.broadcast_in_dim %arg108, dims = [2] : (tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %2384 = stablehlo.add %2382, %2383 : tensor<1x256x256xbf16>
    %2385 = stablehlo.reshape %2384 : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %2386 = stablehlo.reshape %2385 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2387 = stablehlo.transpose %2386, dims = [0, 2, 1] : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %2388 = stablehlo.reshape %2387 : (tensor<1x256x256xbf16>) -> tensor<1x256x16x16xbf16>
    %2389 = stablehlo.transpose %2388, dims = [0, 1, 3, 2] : (tensor<1x256x16x16xbf16>) -> tensor<1x256x16x16xbf16>
    %2390 = stablehlo.reshape %2389 : (tensor<1x256x16x16xbf16>) -> tensor<256x16x16xbf16>
    %2391 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1, 2] : (tensor<256x16x128xbf16>) -> tensor<256x16x128xbf16>
    %2392 = stablehlo.dot_general %2390, %2391, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x16x16xbf16>, tensor<256x16x128xbf16>) -> tensor<256x16x128xbf16>
    %2393 = stablehlo.reshape %2392 : (tensor<256x16x128xbf16>) -> tensor<1x256x16x128xbf16>
    %2394 = stablehlo.transpose %2393, dims = [0, 1, 3, 2] : (tensor<1x256x16x128xbf16>) -> tensor<1x256x128x16xbf16>
    %2395 = stablehlo.reshape %2394 : (tensor<1x256x128x16xbf16>) -> tensor<256x128x16xbf16>
    %2396 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1, 2] : (tensor<256x16x128xbf16>) -> tensor<256x16x128xbf16>
    %2397 = stablehlo.dot_general %2395, %2396, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x16xbf16>, tensor<256x16x128xbf16>) -> tensor<256x128x128xbf16>
    %2398 = stablehlo.reshape %2397 : (tensor<256x128x128xbf16>) -> tensor<1x256x128x128xbf16>
    %2399 = stablehlo.concatenate %2398, %2376, %2354, %2332, dim = 1 : (tensor<1x256x128x128xbf16>, tensor<1x256x128x128xbf16>, tensor<1x256x128x128xbf16>, tensor<1x256x128x128xbf16>) -> tensor<1x1024x128x128xbf16>
    %2400 = stablehlo.convolution(%2399, %arg109) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x128x128xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x128x128xbf16>
    %2401 = stablehlo.convert %2400 : (tensor<1x256x128x128xbf16>) -> tensor<1x256x128x128xf32>
    %2402 = stablehlo.broadcast_in_dim %2401, dims = [0, 1, 2, 3] : (tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    %2403 = stablehlo.broadcast_in_dim %arg212, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %2404 = stablehlo.subtract %2402, %2403 : tensor<1x256x128x128xf32>
    %2405 = stablehlo.broadcast_in_dim %2404, dims = [0, 1, 2, 3] : (tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    %2406 = stablehlo.broadcast_in_dim %arg213, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %2407 = stablehlo.multiply %2405, %2406 : tensor<1x256x128x128xf32>
    %2408 = stablehlo.convert %arg214 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %2409 = stablehlo.broadcast_in_dim %2407, dims = [0, 1, 2, 3] : (tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    %2410 = stablehlo.broadcast_in_dim %2408, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %2411 = stablehlo.multiply %2409, %2410 : tensor<1x256x128x128xf32>
    %2412 = stablehlo.convert %arg215 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %2413 = stablehlo.broadcast_in_dim %2411, dims = [0, 1, 2, 3] : (tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xf32>
    %2414 = stablehlo.broadcast_in_dim %2412, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %2415 = stablehlo.add %2413, %2414 : tensor<1x256x128x128xf32>
    %2416 = stablehlo.convert %2415 : (tensor<1x256x128x128xf32>) -> tensor<1x256x128x128xbf16>
    %2417 = stablehlo.maximum %2416, %cst_78 : tensor<1x256x128x128xbf16>
    %2418 = stablehlo.convolution(%2417, %arg110) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x128x128xbf16>, tensor<150x256x1x1xbf16>) -> tensor<1x150x128x128xbf16>
    %2419 = stablehlo.reshape %arg111 : (tensor<150xbf16>) -> tensor<150x1x1xbf16>
    %2420 = stablehlo.broadcast_in_dim %2418, dims = [0, 1, 2, 3] : (tensor<1x150x128x128xbf16>) -> tensor<1x150x128x128xbf16>
    %2421 = stablehlo.broadcast_in_dim %2419, dims = [1, 2, 3] : (tensor<150x1x1xbf16>) -> tensor<1x150x128x128xbf16>
    %2422 = stablehlo.add %2420, %2421 : tensor<1x150x128x128xbf16>
    return %2422 : tensor<1x150x128x128xbf16>
  }
}
