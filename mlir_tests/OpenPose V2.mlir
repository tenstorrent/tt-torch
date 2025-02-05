module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<32x3x3x3xbf16>, %arg2: tensor<32x1x3x3xbf16>, %arg3: tensor<64x32x1x1xbf16>, %arg4: tensor<64x1x3x3xbf16>, %arg5: tensor<128x64x1x1xbf16>, %arg6: tensor<128x1x3x3xbf16>, %arg7: tensor<128x128x1x1xbf16>, %arg8: tensor<128x1x3x3xbf16>, %arg9: tensor<256x128x1x1xbf16>, %arg10: tensor<256x1x3x3xbf16>, %arg11: tensor<256x256x1x1xbf16>, %arg12: tensor<256x1x3x3xbf16>, %arg13: tensor<512x256x1x1xbf16>, %arg14: tensor<512x1x3x3xbf16>, %arg15: tensor<512x512x1x1xbf16>, %arg16: tensor<512x1x3x3xbf16>, %arg17: tensor<512x512x1x1xbf16>, %arg18: tensor<512x1x3x3xbf16>, %arg19: tensor<512x512x1x1xbf16>, %arg20: tensor<512x1x3x3xbf16>, %arg21: tensor<512x512x1x1xbf16>, %arg22: tensor<512x1x3x3xbf16>, %arg23: tensor<512x512x1x1xbf16>, %arg24: tensor<128x512x1x1xbf16>, %arg25: tensor<128xbf16>, %arg26: tensor<128x1x3x3xbf16>, %arg27: tensor<128x128x1x1xbf16>, %arg28: tensor<128x1x3x3xbf16>, %arg29: tensor<128x128x1x1xbf16>, %arg30: tensor<128x1x3x3xbf16>, %arg31: tensor<128x128x1x1xbf16>, %arg32: tensor<128x128x3x3xbf16>, %arg33: tensor<128xbf16>, %arg34: tensor<128x128x3x3xbf16>, %arg35: tensor<128xbf16>, %arg36: tensor<128x128x3x3xbf16>, %arg37: tensor<128xbf16>, %arg38: tensor<128x128x3x3xbf16>, %arg39: tensor<128xbf16>, %arg40: tensor<512x128x1x1xbf16>, %arg41: tensor<512xbf16>, %arg42: tensor<19x512x1x1xbf16>, %arg43: tensor<19xbf16>, %arg44: tensor<512x128x1x1xbf16>, %arg45: tensor<512xbf16>, %arg46: tensor<38x512x1x1xbf16>, %arg47: tensor<38xbf16>, %arg48: tensor<128x185x1x1xbf16>, %arg49: tensor<128xbf16>, %arg50: tensor<128x128x3x3xbf16>, %arg51: tensor<128xbf16>, %arg52: tensor<128x128x3x3xbf16>, %arg53: tensor<128xbf16>, %arg54: tensor<128x128x1x1xbf16>, %arg55: tensor<128xbf16>, %arg56: tensor<128x128x3x3xbf16>, %arg57: tensor<128xbf16>, %arg58: tensor<128x128x3x3xbf16>, %arg59: tensor<128xbf16>, %arg60: tensor<128x128x1x1xbf16>, %arg61: tensor<128xbf16>, %arg62: tensor<128x128x3x3xbf16>, %arg63: tensor<128xbf16>, %arg64: tensor<128x128x3x3xbf16>, %arg65: tensor<128xbf16>, %arg66: tensor<128x128x1x1xbf16>, %arg67: tensor<128xbf16>, %arg68: tensor<128x128x3x3xbf16>, %arg69: tensor<128xbf16>, %arg70: tensor<128x128x3x3xbf16>, %arg71: tensor<128xbf16>, %arg72: tensor<128x128x1x1xbf16>, %arg73: tensor<128xbf16>, %arg74: tensor<128x128x3x3xbf16>, %arg75: tensor<128xbf16>, %arg76: tensor<128x128x3x3xbf16>, %arg77: tensor<128xbf16>, %arg78: tensor<128x128x1x1xbf16>, %arg79: tensor<128xbf16>, %arg80: tensor<19x128x1x1xbf16>, %arg81: tensor<19xbf16>, %arg82: tensor<128x128x1x1xbf16>, %arg83: tensor<128xbf16>, %arg84: tensor<38x128x1x1xbf16>, %arg85: tensor<38xbf16>, %arg86: tensor<32x1x1xf32>, %arg87: tensor<32x1x1xf32>, %arg88: tensor<32x1x1xbf16>, %arg89: tensor<32x1x1xbf16>, %arg90: tensor<32x1x1xf32>, %arg91: tensor<32x1x1xf32>, %arg92: tensor<32x1x1xbf16>, %arg93: tensor<32x1x1xbf16>, %arg94: tensor<64x1x1xf32>, %arg95: tensor<64x1x1xf32>, %arg96: tensor<64x1x1xbf16>, %arg97: tensor<64x1x1xbf16>, %arg98: tensor<64x1x1xf32>, %arg99: tensor<64x1x1xf32>, %arg100: tensor<64x1x1xbf16>, %arg101: tensor<64x1x1xbf16>, %arg102: tensor<128x1x1xf32>, %arg103: tensor<128x1x1xf32>, %arg104: tensor<128x1x1xbf16>, %arg105: tensor<128x1x1xbf16>, %arg106: tensor<128x1x1xf32>, %arg107: tensor<128x1x1xf32>, %arg108: tensor<128x1x1xbf16>, %arg109: tensor<128x1x1xbf16>, %arg110: tensor<128x1x1xf32>, %arg111: tensor<128x1x1xf32>, %arg112: tensor<128x1x1xbf16>, %arg113: tensor<128x1x1xbf16>, %arg114: tensor<128x1x1xf32>, %arg115: tensor<128x1x1xf32>, %arg116: tensor<128x1x1xbf16>, %arg117: tensor<128x1x1xbf16>, %arg118: tensor<256x1x1xf32>, %arg119: tensor<256x1x1xf32>, %arg120: tensor<256x1x1xbf16>, %arg121: tensor<256x1x1xbf16>, %arg122: tensor<256x1x1xf32>, %arg123: tensor<256x1x1xf32>, %arg124: tensor<256x1x1xbf16>, %arg125: tensor<256x1x1xbf16>, %arg126: tensor<256x1x1xf32>, %arg127: tensor<256x1x1xf32>, %arg128: tensor<256x1x1xbf16>, %arg129: tensor<256x1x1xbf16>, %arg130: tensor<256x1x1xf32>, %arg131: tensor<256x1x1xf32>, %arg132: tensor<256x1x1xbf16>, %arg133: tensor<256x1x1xbf16>, %arg134: tensor<512x1x1xf32>, %arg135: tensor<512x1x1xf32>, %arg136: tensor<512x1x1xbf16>, %arg137: tensor<512x1x1xbf16>, %arg138: tensor<512x1x1xf32>, %arg139: tensor<512x1x1xf32>, %arg140: tensor<512x1x1xbf16>, %arg141: tensor<512x1x1xbf16>, %arg142: tensor<512x1x1xf32>, %arg143: tensor<512x1x1xf32>, %arg144: tensor<512x1x1xbf16>, %arg145: tensor<512x1x1xbf16>, %arg146: tensor<512x1x1xf32>, %arg147: tensor<512x1x1xf32>, %arg148: tensor<512x1x1xbf16>, %arg149: tensor<512x1x1xbf16>, %arg150: tensor<512x1x1xf32>, %arg151: tensor<512x1x1xf32>, %arg152: tensor<512x1x1xbf16>, %arg153: tensor<512x1x1xbf16>, %arg154: tensor<512x1x1xf32>, %arg155: tensor<512x1x1xf32>, %arg156: tensor<512x1x1xbf16>, %arg157: tensor<512x1x1xbf16>, %arg158: tensor<512x1x1xf32>, %arg159: tensor<512x1x1xf32>, %arg160: tensor<512x1x1xbf16>, %arg161: tensor<512x1x1xbf16>, %arg162: tensor<512x1x1xf32>, %arg163: tensor<512x1x1xf32>, %arg164: tensor<512x1x1xbf16>, %arg165: tensor<512x1x1xbf16>, %arg166: tensor<512x1x1xf32>, %arg167: tensor<512x1x1xf32>, %arg168: tensor<512x1x1xbf16>, %arg169: tensor<512x1x1xbf16>, %arg170: tensor<512x1x1xf32>, %arg171: tensor<512x1x1xf32>, %arg172: tensor<512x1x1xbf16>, %arg173: tensor<512x1x1xbf16>, %arg174: tensor<512x1x1xf32>, %arg175: tensor<512x1x1xf32>, %arg176: tensor<512x1x1xbf16>, %arg177: tensor<512x1x1xbf16>, %arg178: tensor<128x1x1xf32>, %arg179: tensor<128x1x1xf32>, %arg180: tensor<128x1x1xbf16>, %arg181: tensor<128x1x1xbf16>, %arg182: tensor<128x1x1xf32>, %arg183: tensor<128x1x1xf32>, %arg184: tensor<128x1x1xbf16>, %arg185: tensor<128x1x1xbf16>, %arg186: tensor<128x1x1xf32>, %arg187: tensor<128x1x1xf32>, %arg188: tensor<128x1x1xbf16>, %arg189: tensor<128x1x1xbf16>, %arg190: tensor<128x1x1xf32>, %arg191: tensor<128x1x1xf32>, %arg192: tensor<128x1x1xbf16>, %arg193: tensor<128x1x1xbf16>, %arg194: tensor<128x1x1xf32>, %arg195: tensor<128x1x1xf32>, %arg196: tensor<128x1x1xbf16>, %arg197: tensor<128x1x1xbf16>, %arg198: tensor<128x1x1xf32>, %arg199: tensor<128x1x1xf32>, %arg200: tensor<128x1x1xbf16>, %arg201: tensor<128x1x1xbf16>, %arg202: tensor<128x1x1xf32>, %arg203: tensor<128x1x1xf32>, %arg204: tensor<128x1x1xbf16>, %arg205: tensor<128x1x1xbf16>, %arg206: tensor<128x1x1xf32>, %arg207: tensor<128x1x1xf32>, %arg208: tensor<128x1x1xbf16>, %arg209: tensor<128x1x1xbf16>, %arg210: tensor<128x1x1xf32>, %arg211: tensor<128x1x1xf32>, %arg212: tensor<128x1x1xbf16>, %arg213: tensor<128x1x1xbf16>, %arg214: tensor<128x1x1xf32>, %arg215: tensor<128x1x1xf32>, %arg216: tensor<128x1x1xbf16>, %arg217: tensor<128x1x1xbf16>) -> tensor<1x57x28x28xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x32x112x112xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x112x112xbf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x56x56xbf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x56x56xbf16>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x28x28xbf16>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x28x28xbf16>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x28x28xbf16>
    %cst_6 = arith.constant dense<1> : tensor<1xi64>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<1xf64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x32x112x112xbf16>
    %1 = stablehlo.convert %0 : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %3 = stablehlo.broadcast_in_dim %arg86, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %4 = stablehlo.subtract %2, %3 : tensor<1x32x112x112xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg87, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x32x112x112xf32>
    %8 = stablehlo.convert %arg88 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x32x112x112xf32>
    %12 = stablehlo.convert %arg89 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %15 = stablehlo.add %13, %14 : tensor<1x32x112x112xf32>
    %16 = stablehlo.convert %15 : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
    %17 = stablehlo.maximum %16, %cst : tensor<1x32x112x112xbf16>
    %18 = stablehlo.convolution(%17, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<1x32x112x112xbf16>, tensor<32x1x3x3xbf16>) -> tensor<1x32x112x112xbf16>
    %19 = stablehlo.convert %18 : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %21 = stablehlo.broadcast_in_dim %arg90, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %22 = stablehlo.subtract %20, %21 : tensor<1x32x112x112xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %24 = stablehlo.broadcast_in_dim %arg91, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<1x32x112x112xf32>
    %26 = stablehlo.convert %arg92 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %27 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %28 = stablehlo.broadcast_in_dim %26, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<1x32x112x112xf32>
    %30 = stablehlo.convert %arg93 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %31 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %33 = stablehlo.add %31, %32 : tensor<1x32x112x112xf32>
    %34 = stablehlo.convert %33 : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
    %35 = stablehlo.maximum %34, %cst : tensor<1x32x112x112xbf16>
    %36 = stablehlo.convolution(%35, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x112x112xbf16>, tensor<64x32x1x1xbf16>) -> tensor<1x64x112x112xbf16>
    %37 = stablehlo.convert %36 : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %39 = stablehlo.broadcast_in_dim %arg94, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %40 = stablehlo.subtract %38, %39 : tensor<1x64x112x112xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %42 = stablehlo.broadcast_in_dim %arg95, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x64x112x112xf32>
    %44 = stablehlo.convert %arg96 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1x64x112x112xf32>
    %48 = stablehlo.convert %arg97 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x64x112x112xf32>
    %52 = stablehlo.convert %51 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xbf16>
    %53 = stablehlo.maximum %52, %cst_0 : tensor<1x64x112x112xbf16>
    %54 = stablehlo.convolution(%53, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<1x64x112x112xbf16>, tensor<64x1x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %55 = stablehlo.convert %54 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %57 = stablehlo.broadcast_in_dim %arg98, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %58 = stablehlo.subtract %56, %57 : tensor<1x64x56x56xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %60 = stablehlo.broadcast_in_dim %arg99, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %61 = stablehlo.multiply %59, %60 : tensor<1x64x56x56xf32>
    %62 = stablehlo.convert %arg100 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %63 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %64 = stablehlo.broadcast_in_dim %62, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %65 = stablehlo.multiply %63, %64 : tensor<1x64x56x56xf32>
    %66 = stablehlo.convert %arg101 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %67 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %68 = stablehlo.broadcast_in_dim %66, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x64x56x56xf32>
    %70 = stablehlo.convert %69 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %71 = stablehlo.maximum %70, %cst_1 : tensor<1x64x56x56xbf16>
    %72 = stablehlo.convolution(%71, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<128x64x1x1xbf16>) -> tensor<1x128x56x56xbf16>
    %73 = stablehlo.convert %72 : (tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %75 = stablehlo.broadcast_in_dim %arg102, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %76 = stablehlo.subtract %74, %75 : tensor<1x128x56x56xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %78 = stablehlo.broadcast_in_dim %arg103, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %79 = stablehlo.multiply %77, %78 : tensor<1x128x56x56xf32>
    %80 = stablehlo.convert %arg104 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %81 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %82 = stablehlo.broadcast_in_dim %80, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<1x128x56x56xf32>
    %84 = stablehlo.convert %arg105 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %85 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %86 = stablehlo.broadcast_in_dim %84, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %87 = stablehlo.add %85, %86 : tensor<1x128x56x56xf32>
    %88 = stablehlo.convert %87 : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xbf16>
    %89 = stablehlo.maximum %88, %cst_2 : tensor<1x128x56x56xbf16>
    %90 = stablehlo.convolution(%89, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x56x56xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x56x56xbf16>
    %91 = stablehlo.convert %90 : (tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %93 = stablehlo.broadcast_in_dim %arg106, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %94 = stablehlo.subtract %92, %93 : tensor<1x128x56x56xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %96 = stablehlo.broadcast_in_dim %arg107, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %97 = stablehlo.multiply %95, %96 : tensor<1x128x56x56xf32>
    %98 = stablehlo.convert %arg108 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %99 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %100 = stablehlo.broadcast_in_dim %98, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %101 = stablehlo.multiply %99, %100 : tensor<1x128x56x56xf32>
    %102 = stablehlo.convert %arg109 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %103 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %104 = stablehlo.broadcast_in_dim %102, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %105 = stablehlo.add %103, %104 : tensor<1x128x56x56xf32>
    %106 = stablehlo.convert %105 : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xbf16>
    %107 = stablehlo.maximum %106, %cst_2 : tensor<1x128x56x56xbf16>
    %108 = stablehlo.convolution(%107, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x56x56xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x56x56xbf16>
    %109 = stablehlo.convert %108 : (tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xf32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %111 = stablehlo.broadcast_in_dim %arg110, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %112 = stablehlo.subtract %110, %111 : tensor<1x128x56x56xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %114 = stablehlo.broadcast_in_dim %arg111, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %115 = stablehlo.multiply %113, %114 : tensor<1x128x56x56xf32>
    %116 = stablehlo.convert %arg112 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %117 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %118 = stablehlo.broadcast_in_dim %116, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %119 = stablehlo.multiply %117, %118 : tensor<1x128x56x56xf32>
    %120 = stablehlo.convert %arg113 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %121 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %122 = stablehlo.broadcast_in_dim %120, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %123 = stablehlo.add %121, %122 : tensor<1x128x56x56xf32>
    %124 = stablehlo.convert %123 : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xbf16>
    %125 = stablehlo.maximum %124, %cst_2 : tensor<1x128x56x56xbf16>
    %126 = stablehlo.convolution(%125, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x56x56xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %127 = stablehlo.convert %126 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %129 = stablehlo.broadcast_in_dim %arg114, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %130 = stablehlo.subtract %128, %129 : tensor<1x128x28x28xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %132 = stablehlo.broadcast_in_dim %arg115, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %133 = stablehlo.multiply %131, %132 : tensor<1x128x28x28xf32>
    %134 = stablehlo.convert %arg116 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %135 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %136 = stablehlo.broadcast_in_dim %134, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %137 = stablehlo.multiply %135, %136 : tensor<1x128x28x28xf32>
    %138 = stablehlo.convert %arg117 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %139 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %140 = stablehlo.broadcast_in_dim %138, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %141 = stablehlo.add %139, %140 : tensor<1x128x28x28xf32>
    %142 = stablehlo.convert %141 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %143 = stablehlo.maximum %142, %cst_3 : tensor<1x128x28x28xbf16>
    %144 = stablehlo.convolution(%143, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<256x128x1x1xbf16>) -> tensor<1x256x28x28xbf16>
    %145 = stablehlo.convert %144 : (tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xf32>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %147 = stablehlo.broadcast_in_dim %arg118, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %148 = stablehlo.subtract %146, %147 : tensor<1x256x28x28xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %150 = stablehlo.broadcast_in_dim %arg119, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %151 = stablehlo.multiply %149, %150 : tensor<1x256x28x28xf32>
    %152 = stablehlo.convert %arg120 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %153 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %154 = stablehlo.broadcast_in_dim %152, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %155 = stablehlo.multiply %153, %154 : tensor<1x256x28x28xf32>
    %156 = stablehlo.convert %arg121 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %157 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %158 = stablehlo.broadcast_in_dim %156, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %159 = stablehlo.add %157, %158 : tensor<1x256x28x28xf32>
    %160 = stablehlo.convert %159 : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xbf16>
    %161 = stablehlo.maximum %160, %cst_4 : tensor<1x256x28x28xbf16>
    %162 = stablehlo.convolution(%161, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x28x28xbf16>, tensor<256x1x3x3xbf16>) -> tensor<1x256x28x28xbf16>
    %163 = stablehlo.convert %162 : (tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %165 = stablehlo.broadcast_in_dim %arg122, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %166 = stablehlo.subtract %164, %165 : tensor<1x256x28x28xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %168 = stablehlo.broadcast_in_dim %arg123, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %169 = stablehlo.multiply %167, %168 : tensor<1x256x28x28xf32>
    %170 = stablehlo.convert %arg124 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %171 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %172 = stablehlo.broadcast_in_dim %170, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %173 = stablehlo.multiply %171, %172 : tensor<1x256x28x28xf32>
    %174 = stablehlo.convert %arg125 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %175 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %176 = stablehlo.broadcast_in_dim %174, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %177 = stablehlo.add %175, %176 : tensor<1x256x28x28xf32>
    %178 = stablehlo.convert %177 : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xbf16>
    %179 = stablehlo.maximum %178, %cst_4 : tensor<1x256x28x28xbf16>
    %180 = stablehlo.convolution(%179, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x28x28xbf16>, tensor<256x256x1x1xbf16>) -> tensor<1x256x28x28xbf16>
    %181 = stablehlo.convert %180 : (tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %183 = stablehlo.broadcast_in_dim %arg126, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %184 = stablehlo.subtract %182, %183 : tensor<1x256x28x28xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %186 = stablehlo.broadcast_in_dim %arg127, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %187 = stablehlo.multiply %185, %186 : tensor<1x256x28x28xf32>
    %188 = stablehlo.convert %arg128 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %189 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %190 = stablehlo.broadcast_in_dim %188, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %191 = stablehlo.multiply %189, %190 : tensor<1x256x28x28xf32>
    %192 = stablehlo.convert %arg129 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %193 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %194 = stablehlo.broadcast_in_dim %192, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %195 = stablehlo.add %193, %194 : tensor<1x256x28x28xf32>
    %196 = stablehlo.convert %195 : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xbf16>
    %197 = stablehlo.maximum %196, %cst_4 : tensor<1x256x28x28xbf16>
    %198 = stablehlo.convolution(%197, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x28x28xbf16>, tensor<256x1x3x3xbf16>) -> tensor<1x256x28x28xbf16>
    %199 = stablehlo.convert %198 : (tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xf32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %201 = stablehlo.broadcast_in_dim %arg130, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %202 = stablehlo.subtract %200, %201 : tensor<1x256x28x28xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %204 = stablehlo.broadcast_in_dim %arg131, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %205 = stablehlo.multiply %203, %204 : tensor<1x256x28x28xf32>
    %206 = stablehlo.convert %arg132 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %207 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %208 = stablehlo.broadcast_in_dim %206, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %209 = stablehlo.multiply %207, %208 : tensor<1x256x28x28xf32>
    %210 = stablehlo.convert %arg133 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %211 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %212 = stablehlo.broadcast_in_dim %210, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %213 = stablehlo.add %211, %212 : tensor<1x256x28x28xf32>
    %214 = stablehlo.convert %213 : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xbf16>
    %215 = stablehlo.maximum %214, %cst_4 : tensor<1x256x28x28xbf16>
    %216 = stablehlo.convolution(%215, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x28x28xbf16>, tensor<512x256x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %217 = stablehlo.convert %216 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %218 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %219 = stablehlo.broadcast_in_dim %arg134, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %220 = stablehlo.subtract %218, %219 : tensor<1x512x28x28xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %222 = stablehlo.broadcast_in_dim %arg135, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %223 = stablehlo.multiply %221, %222 : tensor<1x512x28x28xf32>
    %224 = stablehlo.convert %arg136 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %225 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %226 = stablehlo.broadcast_in_dim %224, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %227 = stablehlo.multiply %225, %226 : tensor<1x512x28x28xf32>
    %228 = stablehlo.convert %arg137 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %229 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %230 = stablehlo.broadcast_in_dim %228, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %231 = stablehlo.add %229, %230 : tensor<1x512x28x28xf32>
    %232 = stablehlo.convert %231 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %233 = stablehlo.maximum %232, %cst_5 : tensor<1x512x28x28xbf16>
    %234 = stablehlo.convolution(%233, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x1x3x3xbf16>) -> tensor<1x512x28x28xbf16>
    %235 = stablehlo.convert %234 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %236 = stablehlo.broadcast_in_dim %235, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %237 = stablehlo.broadcast_in_dim %arg138, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %238 = stablehlo.subtract %236, %237 : tensor<1x512x28x28xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %240 = stablehlo.broadcast_in_dim %arg139, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %241 = stablehlo.multiply %239, %240 : tensor<1x512x28x28xf32>
    %242 = stablehlo.convert %arg140 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %243 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %244 = stablehlo.broadcast_in_dim %242, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x512x28x28xf32>
    %246 = stablehlo.convert %arg141 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %247 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %248 = stablehlo.broadcast_in_dim %246, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %249 = stablehlo.add %247, %248 : tensor<1x512x28x28xf32>
    %250 = stablehlo.convert %249 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %251 = stablehlo.maximum %250, %cst_5 : tensor<1x512x28x28xbf16>
    %252 = stablehlo.convolution(%251, %arg15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %253 = stablehlo.convert %252 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %254 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %255 = stablehlo.broadcast_in_dim %arg142, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %256 = stablehlo.subtract %254, %255 : tensor<1x512x28x28xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %258 = stablehlo.broadcast_in_dim %arg143, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %259 = stablehlo.multiply %257, %258 : tensor<1x512x28x28xf32>
    %260 = stablehlo.convert %arg144 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %261 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %262 = stablehlo.broadcast_in_dim %260, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %263 = stablehlo.multiply %261, %262 : tensor<1x512x28x28xf32>
    %264 = stablehlo.convert %arg145 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %265 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %266 = stablehlo.broadcast_in_dim %264, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %267 = stablehlo.add %265, %266 : tensor<1x512x28x28xf32>
    %268 = stablehlo.convert %267 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %269 = stablehlo.maximum %268, %cst_5 : tensor<1x512x28x28xbf16>
    %270 = stablehlo.convolution(%269, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x1x3x3xbf16>) -> tensor<1x512x28x28xbf16>
    %271 = stablehlo.convert %270 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %273 = stablehlo.broadcast_in_dim %arg146, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %274 = stablehlo.subtract %272, %273 : tensor<1x512x28x28xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %276 = stablehlo.broadcast_in_dim %arg147, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %277 = stablehlo.multiply %275, %276 : tensor<1x512x28x28xf32>
    %278 = stablehlo.convert %arg148 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %279 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %280 = stablehlo.broadcast_in_dim %278, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %281 = stablehlo.multiply %279, %280 : tensor<1x512x28x28xf32>
    %282 = stablehlo.convert %arg149 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %283 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %284 = stablehlo.broadcast_in_dim %282, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %285 = stablehlo.add %283, %284 : tensor<1x512x28x28xf32>
    %286 = stablehlo.convert %285 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %287 = stablehlo.maximum %286, %cst_5 : tensor<1x512x28x28xbf16>
    %288 = stablehlo.convolution(%287, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %289 = stablehlo.convert %288 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %291 = stablehlo.broadcast_in_dim %arg150, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %292 = stablehlo.subtract %290, %291 : tensor<1x512x28x28xf32>
    %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %294 = stablehlo.broadcast_in_dim %arg151, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %295 = stablehlo.multiply %293, %294 : tensor<1x512x28x28xf32>
    %296 = stablehlo.convert %arg152 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %297 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %298 = stablehlo.broadcast_in_dim %296, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %299 = stablehlo.multiply %297, %298 : tensor<1x512x28x28xf32>
    %300 = stablehlo.convert %arg153 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %301 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %302 = stablehlo.broadcast_in_dim %300, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %303 = stablehlo.add %301, %302 : tensor<1x512x28x28xf32>
    %304 = stablehlo.convert %303 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %305 = stablehlo.maximum %304, %cst_5 : tensor<1x512x28x28xbf16>
    %306 = stablehlo.convolution(%305, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x1x3x3xbf16>) -> tensor<1x512x28x28xbf16>
    %307 = stablehlo.convert %306 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %309 = stablehlo.broadcast_in_dim %arg154, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %310 = stablehlo.subtract %308, %309 : tensor<1x512x28x28xf32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %312 = stablehlo.broadcast_in_dim %arg155, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %313 = stablehlo.multiply %311, %312 : tensor<1x512x28x28xf32>
    %314 = stablehlo.convert %arg156 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %315 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %316 = stablehlo.broadcast_in_dim %314, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %317 = stablehlo.multiply %315, %316 : tensor<1x512x28x28xf32>
    %318 = stablehlo.convert %arg157 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %319 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %320 = stablehlo.broadcast_in_dim %318, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %321 = stablehlo.add %319, %320 : tensor<1x512x28x28xf32>
    %322 = stablehlo.convert %321 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %323 = stablehlo.maximum %322, %cst_5 : tensor<1x512x28x28xbf16>
    %324 = stablehlo.convolution(%323, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %325 = stablehlo.convert %324 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %327 = stablehlo.broadcast_in_dim %arg158, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %328 = stablehlo.subtract %326, %327 : tensor<1x512x28x28xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %330 = stablehlo.broadcast_in_dim %arg159, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %331 = stablehlo.multiply %329, %330 : tensor<1x512x28x28xf32>
    %332 = stablehlo.convert %arg160 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %333 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %334 = stablehlo.broadcast_in_dim %332, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %335 = stablehlo.multiply %333, %334 : tensor<1x512x28x28xf32>
    %336 = stablehlo.convert %arg161 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %337 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %338 = stablehlo.broadcast_in_dim %336, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %339 = stablehlo.add %337, %338 : tensor<1x512x28x28xf32>
    %340 = stablehlo.convert %339 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %341 = stablehlo.maximum %340, %cst_5 : tensor<1x512x28x28xbf16>
    %342 = stablehlo.convolution(%341, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x1x3x3xbf16>) -> tensor<1x512x28x28xbf16>
    %343 = stablehlo.convert %342 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %345 = stablehlo.broadcast_in_dim %arg162, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %346 = stablehlo.subtract %344, %345 : tensor<1x512x28x28xf32>
    %347 = stablehlo.broadcast_in_dim %346, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %348 = stablehlo.broadcast_in_dim %arg163, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %349 = stablehlo.multiply %347, %348 : tensor<1x512x28x28xf32>
    %350 = stablehlo.convert %arg164 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %351 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %352 = stablehlo.broadcast_in_dim %350, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %353 = stablehlo.multiply %351, %352 : tensor<1x512x28x28xf32>
    %354 = stablehlo.convert %arg165 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %355 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %356 = stablehlo.broadcast_in_dim %354, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %357 = stablehlo.add %355, %356 : tensor<1x512x28x28xf32>
    %358 = stablehlo.convert %357 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %359 = stablehlo.maximum %358, %cst_5 : tensor<1x512x28x28xbf16>
    %360 = stablehlo.convolution(%359, %arg21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %361 = stablehlo.convert %360 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %363 = stablehlo.broadcast_in_dim %arg166, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %364 = stablehlo.subtract %362, %363 : tensor<1x512x28x28xf32>
    %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %366 = stablehlo.broadcast_in_dim %arg167, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %367 = stablehlo.multiply %365, %366 : tensor<1x512x28x28xf32>
    %368 = stablehlo.convert %arg168 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %369 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %370 = stablehlo.broadcast_in_dim %368, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %371 = stablehlo.multiply %369, %370 : tensor<1x512x28x28xf32>
    %372 = stablehlo.convert %arg169 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %373 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %374 = stablehlo.broadcast_in_dim %372, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %375 = stablehlo.add %373, %374 : tensor<1x512x28x28xf32>
    %376 = stablehlo.convert %375 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %377 = stablehlo.maximum %376, %cst_5 : tensor<1x512x28x28xbf16>
    %378 = stablehlo.convolution(%377, %arg22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x1x3x3xbf16>) -> tensor<1x512x28x28xbf16>
    %379 = stablehlo.convert %378 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %381 = stablehlo.broadcast_in_dim %arg170, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %382 = stablehlo.subtract %380, %381 : tensor<1x512x28x28xf32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %384 = stablehlo.broadcast_in_dim %arg171, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %385 = stablehlo.multiply %383, %384 : tensor<1x512x28x28xf32>
    %386 = stablehlo.convert %arg172 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %387 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %388 = stablehlo.broadcast_in_dim %386, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %389 = stablehlo.multiply %387, %388 : tensor<1x512x28x28xf32>
    %390 = stablehlo.convert %arg173 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %391 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %392 = stablehlo.broadcast_in_dim %390, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %393 = stablehlo.add %391, %392 : tensor<1x512x28x28xf32>
    %394 = stablehlo.convert %393 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %395 = stablehlo.maximum %394, %cst_5 : tensor<1x512x28x28xbf16>
    %396 = stablehlo.convolution(%395, %arg23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<512x512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %397 = stablehlo.convert %396 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %399 = stablehlo.broadcast_in_dim %arg174, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %400 = stablehlo.subtract %398, %399 : tensor<1x512x28x28xf32>
    %401 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %402 = stablehlo.broadcast_in_dim %arg175, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %403 = stablehlo.multiply %401, %402 : tensor<1x512x28x28xf32>
    %404 = stablehlo.convert %arg176 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %405 = stablehlo.broadcast_in_dim %403, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %406 = stablehlo.broadcast_in_dim %404, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %407 = stablehlo.multiply %405, %406 : tensor<1x512x28x28xf32>
    %408 = stablehlo.convert %arg177 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %409 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %410 = stablehlo.broadcast_in_dim %408, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %411 = stablehlo.add %409, %410 : tensor<1x512x28x28xf32>
    %412 = stablehlo.convert %411 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %413 = stablehlo.maximum %412, %cst_5 : tensor<1x512x28x28xbf16>
    %414 = stablehlo.convolution(%413, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %415 = stablehlo.reshape %arg25 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %416 = stablehlo.broadcast_in_dim %414, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %417 = stablehlo.broadcast_in_dim %415, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %418 = stablehlo.add %416, %417 : tensor<1x128x28x28xbf16>
    %419 = stablehlo.maximum %418, %cst_3 : tensor<1x128x28x28xbf16>
    %420 = stablehlo.convolution(%419, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %421 = stablehlo.convert %c : (tensor<i64>) -> tensor<bf16>
    %422 = stablehlo.broadcast_in_dim %421, dims = [] : (tensor<bf16>) -> tensor<1x128x28x28xbf16>
    %423 = stablehlo.broadcast_in_dim %420, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %424 = stablehlo.maximum %422, %423 : tensor<1x128x28x28xbf16>
    %425 = stablehlo.convert %cst_6 : (tensor<1xi64>) -> tensor<1xbf16>
    %426 = stablehlo.reshape %425 : (tensor<1xbf16>) -> tensor<bf16>
    %427 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %428 = stablehlo.broadcast_in_dim %426, dims = [] : (tensor<bf16>) -> tensor<1x128x28x28xbf16>
    %429 = stablehlo.multiply %427, %428 : tensor<1x128x28x28xbf16>
    %430 = stablehlo.minimum %422, %423 : tensor<1x128x28x28xbf16>
    %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %432 = stablehlo.multiply %431, %428 : tensor<1x128x28x28xbf16>
    %433 = stablehlo.exponential %432 : tensor<1x128x28x28xbf16>
    %434 = stablehlo.convert %cst_7 : (tensor<1xf64>) -> tensor<1xbf16>
    %435 = stablehlo.reshape %434 : (tensor<1xbf16>) -> tensor<bf16>
    %436 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %437 = stablehlo.broadcast_in_dim %435, dims = [] : (tensor<bf16>) -> tensor<1x128x28x28xbf16>
    %438 = stablehlo.subtract %436, %437 : tensor<1x128x28x28xbf16>
    %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %440 = stablehlo.multiply %439, %428 : tensor<1x128x28x28xbf16>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %442 = stablehlo.multiply %441, %437 : tensor<1x128x28x28xbf16>
    %443 = stablehlo.add %429, %442 : tensor<1x128x28x28xbf16>
    %444 = stablehlo.convolution(%443, %arg27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %445 = stablehlo.broadcast_in_dim %444, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %446 = stablehlo.maximum %422, %445 : tensor<1x128x28x28xbf16>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %448 = stablehlo.multiply %447, %428 : tensor<1x128x28x28xbf16>
    %449 = stablehlo.minimum %422, %445 : tensor<1x128x28x28xbf16>
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %451 = stablehlo.multiply %450, %428 : tensor<1x128x28x28xbf16>
    %452 = stablehlo.exponential %451 : tensor<1x128x28x28xbf16>
    %453 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %454 = stablehlo.subtract %453, %437 : tensor<1x128x28x28xbf16>
    %455 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %456 = stablehlo.multiply %455, %428 : tensor<1x128x28x28xbf16>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %458 = stablehlo.multiply %457, %437 : tensor<1x128x28x28xbf16>
    %459 = stablehlo.add %448, %458 : tensor<1x128x28x28xbf16>
    %460 = stablehlo.convolution(%459, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %462 = stablehlo.maximum %422, %461 : tensor<1x128x28x28xbf16>
    %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %464 = stablehlo.multiply %463, %428 : tensor<1x128x28x28xbf16>
    %465 = stablehlo.minimum %422, %461 : tensor<1x128x28x28xbf16>
    %466 = stablehlo.broadcast_in_dim %465, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %467 = stablehlo.multiply %466, %428 : tensor<1x128x28x28xbf16>
    %468 = stablehlo.exponential %467 : tensor<1x128x28x28xbf16>
    %469 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %470 = stablehlo.subtract %469, %437 : tensor<1x128x28x28xbf16>
    %471 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %472 = stablehlo.multiply %471, %428 : tensor<1x128x28x28xbf16>
    %473 = stablehlo.broadcast_in_dim %472, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %474 = stablehlo.multiply %473, %437 : tensor<1x128x28x28xbf16>
    %475 = stablehlo.add %464, %474 : tensor<1x128x28x28xbf16>
    %476 = stablehlo.convolution(%475, %arg29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %478 = stablehlo.maximum %422, %477 : tensor<1x128x28x28xbf16>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %480 = stablehlo.multiply %479, %428 : tensor<1x128x28x28xbf16>
    %481 = stablehlo.minimum %422, %477 : tensor<1x128x28x28xbf16>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %483 = stablehlo.multiply %482, %428 : tensor<1x128x28x28xbf16>
    %484 = stablehlo.exponential %483 : tensor<1x128x28x28xbf16>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %486 = stablehlo.subtract %485, %437 : tensor<1x128x28x28xbf16>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %488 = stablehlo.multiply %487, %428 : tensor<1x128x28x28xbf16>
    %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %490 = stablehlo.multiply %489, %437 : tensor<1x128x28x28xbf16>
    %491 = stablehlo.add %480, %490 : tensor<1x128x28x28xbf16>
    %492 = stablehlo.convolution(%491, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x1x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %493 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %494 = stablehlo.maximum %422, %493 : tensor<1x128x28x28xbf16>
    %495 = stablehlo.broadcast_in_dim %494, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %496 = stablehlo.multiply %495, %428 : tensor<1x128x28x28xbf16>
    %497 = stablehlo.minimum %422, %493 : tensor<1x128x28x28xbf16>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %499 = stablehlo.multiply %498, %428 : tensor<1x128x28x28xbf16>
    %500 = stablehlo.exponential %499 : tensor<1x128x28x28xbf16>
    %501 = stablehlo.broadcast_in_dim %500, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %502 = stablehlo.subtract %501, %437 : tensor<1x128x28x28xbf16>
    %503 = stablehlo.broadcast_in_dim %502, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %504 = stablehlo.multiply %503, %428 : tensor<1x128x28x28xbf16>
    %505 = stablehlo.broadcast_in_dim %504, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %506 = stablehlo.multiply %505, %437 : tensor<1x128x28x28xbf16>
    %507 = stablehlo.add %496, %506 : tensor<1x128x28x28xbf16>
    %508 = stablehlo.convolution(%507, %arg31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %509 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %510 = stablehlo.maximum %422, %509 : tensor<1x128x28x28xbf16>
    %511 = stablehlo.broadcast_in_dim %510, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %512 = stablehlo.multiply %511, %428 : tensor<1x128x28x28xbf16>
    %513 = stablehlo.minimum %422, %509 : tensor<1x128x28x28xbf16>
    %514 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %515 = stablehlo.multiply %514, %428 : tensor<1x128x28x28xbf16>
    %516 = stablehlo.exponential %515 : tensor<1x128x28x28xbf16>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %518 = stablehlo.subtract %517, %437 : tensor<1x128x28x28xbf16>
    %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %520 = stablehlo.multiply %519, %428 : tensor<1x128x28x28xbf16>
    %521 = stablehlo.broadcast_in_dim %520, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %522 = stablehlo.multiply %521, %437 : tensor<1x128x28x28xbf16>
    %523 = stablehlo.add %512, %522 : tensor<1x128x28x28xbf16>
    %524 = stablehlo.add %419, %523 : tensor<1x128x28x28xbf16>
    %525 = stablehlo.convolution(%524, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %526 = stablehlo.reshape %arg33 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %527 = stablehlo.broadcast_in_dim %525, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %528 = stablehlo.broadcast_in_dim %526, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %529 = stablehlo.add %527, %528 : tensor<1x128x28x28xbf16>
    %530 = stablehlo.maximum %529, %cst_3 : tensor<1x128x28x28xbf16>
    %531 = stablehlo.convolution(%530, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %532 = stablehlo.reshape %arg35 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %533 = stablehlo.broadcast_in_dim %531, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %534 = stablehlo.broadcast_in_dim %532, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %535 = stablehlo.add %533, %534 : tensor<1x128x28x28xbf16>
    %536 = stablehlo.maximum %535, %cst_3 : tensor<1x128x28x28xbf16>
    %537 = stablehlo.convolution(%536, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %538 = stablehlo.reshape %arg37 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %539 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %540 = stablehlo.broadcast_in_dim %538, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %541 = stablehlo.add %539, %540 : tensor<1x128x28x28xbf16>
    %542 = stablehlo.maximum %541, %cst_3 : tensor<1x128x28x28xbf16>
    %543 = stablehlo.convolution(%542, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %544 = stablehlo.reshape %arg39 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %545 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %546 = stablehlo.broadcast_in_dim %544, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %547 = stablehlo.add %545, %546 : tensor<1x128x28x28xbf16>
    %548 = stablehlo.maximum %547, %cst_3 : tensor<1x128x28x28xbf16>
    %549 = stablehlo.convolution(%548, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %550 = stablehlo.reshape %arg41 : (tensor<512xbf16>) -> tensor<512x1x1xbf16>
    %551 = stablehlo.broadcast_in_dim %549, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %552 = stablehlo.broadcast_in_dim %550, dims = [1, 2, 3] : (tensor<512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %553 = stablehlo.add %551, %552 : tensor<1x512x28x28xbf16>
    %554 = stablehlo.maximum %553, %cst_5 : tensor<1x512x28x28xbf16>
    %555 = stablehlo.convolution(%554, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<19x512x1x1xbf16>) -> tensor<1x19x28x28xbf16>
    %556 = stablehlo.reshape %arg43 : (tensor<19xbf16>) -> tensor<19x1x1xbf16>
    %557 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2, 3] : (tensor<1x19x28x28xbf16>) -> tensor<1x19x28x28xbf16>
    %558 = stablehlo.broadcast_in_dim %556, dims = [1, 2, 3] : (tensor<19x1x1xbf16>) -> tensor<1x19x28x28xbf16>
    %559 = stablehlo.add %557, %558 : tensor<1x19x28x28xbf16>
    %560 = stablehlo.convolution(%548, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %561 = stablehlo.reshape %arg45 : (tensor<512xbf16>) -> tensor<512x1x1xbf16>
    %562 = stablehlo.broadcast_in_dim %560, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %563 = stablehlo.broadcast_in_dim %561, dims = [1, 2, 3] : (tensor<512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %564 = stablehlo.add %562, %563 : tensor<1x512x28x28xbf16>
    %565 = stablehlo.maximum %564, %cst_5 : tensor<1x512x28x28xbf16>
    %566 = stablehlo.convolution(%565, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<38x512x1x1xbf16>) -> tensor<1x38x28x28xbf16>
    %567 = stablehlo.reshape %arg47 : (tensor<38xbf16>) -> tensor<38x1x1xbf16>
    %568 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2, 3] : (tensor<1x38x28x28xbf16>) -> tensor<1x38x28x28xbf16>
    %569 = stablehlo.broadcast_in_dim %567, dims = [1, 2, 3] : (tensor<38x1x1xbf16>) -> tensor<1x38x28x28xbf16>
    %570 = stablehlo.add %568, %569 : tensor<1x38x28x28xbf16>
    %571 = stablehlo.concatenate %530, %559, %570, dim = 1 : (tensor<1x128x28x28xbf16>, tensor<1x19x28x28xbf16>, tensor<1x38x28x28xbf16>) -> tensor<1x185x28x28xbf16>
    %572 = stablehlo.slice %571 [0:1, 0:128, 0:28, 0:28] : (tensor<1x185x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %573 = stablehlo.convolution(%571, %arg48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x185x28x28xbf16>, tensor<128x185x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %574 = stablehlo.reshape %arg49 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %575 = stablehlo.broadcast_in_dim %573, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %576 = stablehlo.broadcast_in_dim %574, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %577 = stablehlo.add %575, %576 : tensor<1x128x28x28xbf16>
    %578 = stablehlo.maximum %577, %cst_3 : tensor<1x128x28x28xbf16>
    %579 = stablehlo.convolution(%578, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %580 = stablehlo.reshape %arg51 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %581 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %582 = stablehlo.broadcast_in_dim %580, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %583 = stablehlo.add %581, %582 : tensor<1x128x28x28xbf16>
    %584 = stablehlo.convert %583 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %585 = stablehlo.broadcast_in_dim %584, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %586 = stablehlo.broadcast_in_dim %arg178, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %587 = stablehlo.subtract %585, %586 : tensor<1x128x28x28xf32>
    %588 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %589 = stablehlo.broadcast_in_dim %arg179, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %590 = stablehlo.multiply %588, %589 : tensor<1x128x28x28xf32>
    %591 = stablehlo.convert %arg180 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %592 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %593 = stablehlo.broadcast_in_dim %591, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %594 = stablehlo.multiply %592, %593 : tensor<1x128x28x28xf32>
    %595 = stablehlo.convert %arg181 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %596 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %597 = stablehlo.broadcast_in_dim %595, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %598 = stablehlo.add %596, %597 : tensor<1x128x28x28xf32>
    %599 = stablehlo.convert %598 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %600 = stablehlo.maximum %599, %cst_3 : tensor<1x128x28x28xbf16>
    %601 = stablehlo.convolution(%600, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %602 = stablehlo.reshape %arg53 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %603 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %604 = stablehlo.broadcast_in_dim %602, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %605 = stablehlo.add %603, %604 : tensor<1x128x28x28xbf16>
    %606 = stablehlo.convert %605 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %607 = stablehlo.broadcast_in_dim %606, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %608 = stablehlo.broadcast_in_dim %arg182, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %609 = stablehlo.subtract %607, %608 : tensor<1x128x28x28xf32>
    %610 = stablehlo.broadcast_in_dim %609, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %611 = stablehlo.broadcast_in_dim %arg183, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %612 = stablehlo.multiply %610, %611 : tensor<1x128x28x28xf32>
    %613 = stablehlo.convert %arg184 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %614 = stablehlo.broadcast_in_dim %612, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %615 = stablehlo.broadcast_in_dim %613, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %616 = stablehlo.multiply %614, %615 : tensor<1x128x28x28xf32>
    %617 = stablehlo.convert %arg185 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %618 = stablehlo.broadcast_in_dim %616, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %619 = stablehlo.broadcast_in_dim %617, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %620 = stablehlo.add %618, %619 : tensor<1x128x28x28xf32>
    %621 = stablehlo.convert %620 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %622 = stablehlo.maximum %621, %cst_3 : tensor<1x128x28x28xbf16>
    %623 = stablehlo.add %578, %622 : tensor<1x128x28x28xbf16>
    %624 = stablehlo.convolution(%623, %arg54) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %625 = stablehlo.reshape %arg55 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %626 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %627 = stablehlo.broadcast_in_dim %625, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %628 = stablehlo.add %626, %627 : tensor<1x128x28x28xbf16>
    %629 = stablehlo.maximum %628, %cst_3 : tensor<1x128x28x28xbf16>
    %630 = stablehlo.convolution(%629, %arg56) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %631 = stablehlo.reshape %arg57 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %632 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %633 = stablehlo.broadcast_in_dim %631, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %634 = stablehlo.add %632, %633 : tensor<1x128x28x28xbf16>
    %635 = stablehlo.convert %634 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %636 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %637 = stablehlo.broadcast_in_dim %arg186, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %638 = stablehlo.subtract %636, %637 : tensor<1x128x28x28xf32>
    %639 = stablehlo.broadcast_in_dim %638, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %640 = stablehlo.broadcast_in_dim %arg187, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %641 = stablehlo.multiply %639, %640 : tensor<1x128x28x28xf32>
    %642 = stablehlo.convert %arg188 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %643 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %644 = stablehlo.broadcast_in_dim %642, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %645 = stablehlo.multiply %643, %644 : tensor<1x128x28x28xf32>
    %646 = stablehlo.convert %arg189 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %647 = stablehlo.broadcast_in_dim %645, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %648 = stablehlo.broadcast_in_dim %646, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %649 = stablehlo.add %647, %648 : tensor<1x128x28x28xf32>
    %650 = stablehlo.convert %649 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %651 = stablehlo.maximum %650, %cst_3 : tensor<1x128x28x28xbf16>
    %652 = stablehlo.convolution(%651, %arg58) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %653 = stablehlo.reshape %arg59 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %654 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %655 = stablehlo.broadcast_in_dim %653, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %656 = stablehlo.add %654, %655 : tensor<1x128x28x28xbf16>
    %657 = stablehlo.convert %656 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %658 = stablehlo.broadcast_in_dim %657, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %659 = stablehlo.broadcast_in_dim %arg190, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %660 = stablehlo.subtract %658, %659 : tensor<1x128x28x28xf32>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %662 = stablehlo.broadcast_in_dim %arg191, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %663 = stablehlo.multiply %661, %662 : tensor<1x128x28x28xf32>
    %664 = stablehlo.convert %arg192 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %665 = stablehlo.broadcast_in_dim %663, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %666 = stablehlo.broadcast_in_dim %664, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %667 = stablehlo.multiply %665, %666 : tensor<1x128x28x28xf32>
    %668 = stablehlo.convert %arg193 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %669 = stablehlo.broadcast_in_dim %667, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %670 = stablehlo.broadcast_in_dim %668, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %671 = stablehlo.add %669, %670 : tensor<1x128x28x28xf32>
    %672 = stablehlo.convert %671 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %673 = stablehlo.maximum %672, %cst_3 : tensor<1x128x28x28xbf16>
    %674 = stablehlo.add %629, %673 : tensor<1x128x28x28xbf16>
    %675 = stablehlo.convolution(%674, %arg60) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %676 = stablehlo.reshape %arg61 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %677 = stablehlo.broadcast_in_dim %675, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %678 = stablehlo.broadcast_in_dim %676, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %679 = stablehlo.add %677, %678 : tensor<1x128x28x28xbf16>
    %680 = stablehlo.maximum %679, %cst_3 : tensor<1x128x28x28xbf16>
    %681 = stablehlo.convolution(%680, %arg62) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %682 = stablehlo.reshape %arg63 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %683 = stablehlo.broadcast_in_dim %681, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %684 = stablehlo.broadcast_in_dim %682, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %685 = stablehlo.add %683, %684 : tensor<1x128x28x28xbf16>
    %686 = stablehlo.convert %685 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %687 = stablehlo.broadcast_in_dim %686, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %688 = stablehlo.broadcast_in_dim %arg194, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %689 = stablehlo.subtract %687, %688 : tensor<1x128x28x28xf32>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %691 = stablehlo.broadcast_in_dim %arg195, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %692 = stablehlo.multiply %690, %691 : tensor<1x128x28x28xf32>
    %693 = stablehlo.convert %arg196 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %694 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %695 = stablehlo.broadcast_in_dim %693, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %696 = stablehlo.multiply %694, %695 : tensor<1x128x28x28xf32>
    %697 = stablehlo.convert %arg197 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %698 = stablehlo.broadcast_in_dim %696, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %699 = stablehlo.broadcast_in_dim %697, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %700 = stablehlo.add %698, %699 : tensor<1x128x28x28xf32>
    %701 = stablehlo.convert %700 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %702 = stablehlo.maximum %701, %cst_3 : tensor<1x128x28x28xbf16>
    %703 = stablehlo.convolution(%702, %arg64) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %704 = stablehlo.reshape %arg65 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %705 = stablehlo.broadcast_in_dim %703, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %706 = stablehlo.broadcast_in_dim %704, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %707 = stablehlo.add %705, %706 : tensor<1x128x28x28xbf16>
    %708 = stablehlo.convert %707 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %710 = stablehlo.broadcast_in_dim %arg198, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %711 = stablehlo.subtract %709, %710 : tensor<1x128x28x28xf32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %713 = stablehlo.broadcast_in_dim %arg199, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %714 = stablehlo.multiply %712, %713 : tensor<1x128x28x28xf32>
    %715 = stablehlo.convert %arg200 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %716 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %717 = stablehlo.broadcast_in_dim %715, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %718 = stablehlo.multiply %716, %717 : tensor<1x128x28x28xf32>
    %719 = stablehlo.convert %arg201 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %720 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %721 = stablehlo.broadcast_in_dim %719, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %722 = stablehlo.add %720, %721 : tensor<1x128x28x28xf32>
    %723 = stablehlo.convert %722 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %724 = stablehlo.maximum %723, %cst_3 : tensor<1x128x28x28xbf16>
    %725 = stablehlo.add %680, %724 : tensor<1x128x28x28xbf16>
    %726 = stablehlo.convolution(%725, %arg66) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %727 = stablehlo.reshape %arg67 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %728 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %729 = stablehlo.broadcast_in_dim %727, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %730 = stablehlo.add %728, %729 : tensor<1x128x28x28xbf16>
    %731 = stablehlo.maximum %730, %cst_3 : tensor<1x128x28x28xbf16>
    %732 = stablehlo.convolution(%731, %arg68) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %733 = stablehlo.reshape %arg69 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %734 = stablehlo.broadcast_in_dim %732, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %735 = stablehlo.broadcast_in_dim %733, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %736 = stablehlo.add %734, %735 : tensor<1x128x28x28xbf16>
    %737 = stablehlo.convert %736 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %738 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %739 = stablehlo.broadcast_in_dim %arg202, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %740 = stablehlo.subtract %738, %739 : tensor<1x128x28x28xf32>
    %741 = stablehlo.broadcast_in_dim %740, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %742 = stablehlo.broadcast_in_dim %arg203, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %743 = stablehlo.multiply %741, %742 : tensor<1x128x28x28xf32>
    %744 = stablehlo.convert %arg204 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %745 = stablehlo.broadcast_in_dim %743, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %746 = stablehlo.broadcast_in_dim %744, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %747 = stablehlo.multiply %745, %746 : tensor<1x128x28x28xf32>
    %748 = stablehlo.convert %arg205 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %749 = stablehlo.broadcast_in_dim %747, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %750 = stablehlo.broadcast_in_dim %748, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %751 = stablehlo.add %749, %750 : tensor<1x128x28x28xf32>
    %752 = stablehlo.convert %751 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %753 = stablehlo.maximum %752, %cst_3 : tensor<1x128x28x28xbf16>
    %754 = stablehlo.convolution(%753, %arg70) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %755 = stablehlo.reshape %arg71 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %756 = stablehlo.broadcast_in_dim %754, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %757 = stablehlo.broadcast_in_dim %755, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %758 = stablehlo.add %756, %757 : tensor<1x128x28x28xbf16>
    %759 = stablehlo.convert %758 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %760 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %761 = stablehlo.broadcast_in_dim %arg206, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %762 = stablehlo.subtract %760, %761 : tensor<1x128x28x28xf32>
    %763 = stablehlo.broadcast_in_dim %762, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %764 = stablehlo.broadcast_in_dim %arg207, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %765 = stablehlo.multiply %763, %764 : tensor<1x128x28x28xf32>
    %766 = stablehlo.convert %arg208 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %767 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %768 = stablehlo.broadcast_in_dim %766, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %769 = stablehlo.multiply %767, %768 : tensor<1x128x28x28xf32>
    %770 = stablehlo.convert %arg209 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %771 = stablehlo.broadcast_in_dim %769, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %772 = stablehlo.broadcast_in_dim %770, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %773 = stablehlo.add %771, %772 : tensor<1x128x28x28xf32>
    %774 = stablehlo.convert %773 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %775 = stablehlo.maximum %774, %cst_3 : tensor<1x128x28x28xbf16>
    %776 = stablehlo.add %731, %775 : tensor<1x128x28x28xbf16>
    %777 = stablehlo.convolution(%776, %arg72) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %778 = stablehlo.reshape %arg73 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %779 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %780 = stablehlo.broadcast_in_dim %778, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %781 = stablehlo.add %779, %780 : tensor<1x128x28x28xbf16>
    %782 = stablehlo.maximum %781, %cst_3 : tensor<1x128x28x28xbf16>
    %783 = stablehlo.convolution(%782, %arg74) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %784 = stablehlo.reshape %arg75 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %785 = stablehlo.broadcast_in_dim %783, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %786 = stablehlo.broadcast_in_dim %784, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %787 = stablehlo.add %785, %786 : tensor<1x128x28x28xbf16>
    %788 = stablehlo.convert %787 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %790 = stablehlo.broadcast_in_dim %arg210, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %791 = stablehlo.subtract %789, %790 : tensor<1x128x28x28xf32>
    %792 = stablehlo.broadcast_in_dim %791, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %793 = stablehlo.broadcast_in_dim %arg211, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %794 = stablehlo.multiply %792, %793 : tensor<1x128x28x28xf32>
    %795 = stablehlo.convert %arg212 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %796 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %797 = stablehlo.broadcast_in_dim %795, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %798 = stablehlo.multiply %796, %797 : tensor<1x128x28x28xf32>
    %799 = stablehlo.convert %arg213 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %800 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %801 = stablehlo.broadcast_in_dim %799, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %802 = stablehlo.add %800, %801 : tensor<1x128x28x28xf32>
    %803 = stablehlo.convert %802 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %804 = stablehlo.maximum %803, %cst_3 : tensor<1x128x28x28xbf16>
    %805 = stablehlo.convolution(%804, %arg76) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %806 = stablehlo.reshape %arg77 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %807 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %808 = stablehlo.broadcast_in_dim %806, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %809 = stablehlo.add %807, %808 : tensor<1x128x28x28xbf16>
    %810 = stablehlo.convert %809 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %812 = stablehlo.broadcast_in_dim %arg214, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %813 = stablehlo.subtract %811, %812 : tensor<1x128x28x28xf32>
    %814 = stablehlo.broadcast_in_dim %813, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %815 = stablehlo.broadcast_in_dim %arg215, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %816 = stablehlo.multiply %814, %815 : tensor<1x128x28x28xf32>
    %817 = stablehlo.convert %arg216 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %818 = stablehlo.broadcast_in_dim %816, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %819 = stablehlo.broadcast_in_dim %817, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %820 = stablehlo.multiply %818, %819 : tensor<1x128x28x28xf32>
    %821 = stablehlo.convert %arg217 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %822 = stablehlo.broadcast_in_dim %820, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %823 = stablehlo.broadcast_in_dim %821, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %824 = stablehlo.add %822, %823 : tensor<1x128x28x28xf32>
    %825 = stablehlo.convert %824 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %826 = stablehlo.maximum %825, %cst_3 : tensor<1x128x28x28xbf16>
    %827 = stablehlo.add %782, %826 : tensor<1x128x28x28xbf16>
    %828 = stablehlo.convolution(%827, %arg78) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %829 = stablehlo.reshape %arg79 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %830 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %831 = stablehlo.broadcast_in_dim %829, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %832 = stablehlo.add %830, %831 : tensor<1x128x28x28xbf16>
    %833 = stablehlo.maximum %832, %cst_3 : tensor<1x128x28x28xbf16>
    %834 = stablehlo.convolution(%833, %arg80) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<19x128x1x1xbf16>) -> tensor<1x19x28x28xbf16>
    %835 = stablehlo.reshape %arg81 : (tensor<19xbf16>) -> tensor<19x1x1xbf16>
    %836 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2, 3] : (tensor<1x19x28x28xbf16>) -> tensor<1x19x28x28xbf16>
    %837 = stablehlo.broadcast_in_dim %835, dims = [1, 2, 3] : (tensor<19x1x1xbf16>) -> tensor<1x19x28x28xbf16>
    %838 = stablehlo.add %836, %837 : tensor<1x19x28x28xbf16>
    %839 = stablehlo.convolution(%827, %arg82) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %840 = stablehlo.reshape %arg83 : (tensor<128xbf16>) -> tensor<128x1x1xbf16>
    %841 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %842 = stablehlo.broadcast_in_dim %840, dims = [1, 2, 3] : (tensor<128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %843 = stablehlo.add %841, %842 : tensor<1x128x28x28xbf16>
    %844 = stablehlo.maximum %843, %cst_3 : tensor<1x128x28x28xbf16>
    %845 = stablehlo.convolution(%844, %arg84) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<38x128x1x1xbf16>) -> tensor<1x38x28x28xbf16>
    %846 = stablehlo.reshape %arg85 : (tensor<38xbf16>) -> tensor<38x1x1xbf16>
    %847 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x38x28x28xbf16>) -> tensor<1x38x28x28xbf16>
    %848 = stablehlo.broadcast_in_dim %846, dims = [1, 2, 3] : (tensor<38x1x1xbf16>) -> tensor<1x38x28x28xbf16>
    %849 = stablehlo.add %847, %848 : tensor<1x38x28x28xbf16>
    %850 = stablehlo.concatenate %572, %838, %849, dim = 1 : (tensor<1x128x28x28xbf16>, tensor<1x19x28x28xbf16>, tensor<1x38x28x28xbf16>) -> tensor<1x185x28x28xbf16>
    %851 = stablehlo.slice %850 [0:1, 128:185, 0:28, 0:28] : (tensor<1x185x28x28xbf16>) -> tensor<1x57x28x28xbf16>
    return %851 : tensor<1x57x28x28xbf16>
  }
}
