module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16>, %arg2: tensor<64x64x1x1xbf16>, %arg3: tensor<64x64x3x3xbf16>, %arg4: tensor<256x64x1x1xbf16>, %arg5: tensor<256x64x1x1xbf16>, %arg6: tensor<64x256x1x1xbf16>, %arg7: tensor<64x64x3x3xbf16>, %arg8: tensor<256x64x1x1xbf16>, %arg9: tensor<64x256x1x1xbf16>, %arg10: tensor<64x64x3x3xbf16>, %arg11: tensor<256x64x1x1xbf16>, %arg12: tensor<128x256x1x1xbf16>, %arg13: tensor<128x128x3x3xbf16>, %arg14: tensor<512x128x1x1xbf16>, %arg15: tensor<512x256x1x1xbf16>, %arg16: tensor<128x512x1x1xbf16>, %arg17: tensor<128x128x3x3xbf16>, %arg18: tensor<512x128x1x1xbf16>, %arg19: tensor<128x512x1x1xbf16>, %arg20: tensor<128x128x3x3xbf16>, %arg21: tensor<512x128x1x1xbf16>, %arg22: tensor<128x512x1x1xbf16>, %arg23: tensor<128x128x3x3xbf16>, %arg24: tensor<512x128x1x1xbf16>, %arg25: tensor<256x512x1x1xbf16>, %arg26: tensor<256x256x3x3xbf16>, %arg27: tensor<1024x256x1x1xbf16>, %arg28: tensor<1024x512x1x1xbf16>, %arg29: tensor<256x1024x1x1xbf16>, %arg30: tensor<256x256x3x3xbf16>, %arg31: tensor<1024x256x1x1xbf16>, %arg32: tensor<256x1024x1x1xbf16>, %arg33: tensor<256x256x3x3xbf16>, %arg34: tensor<1024x256x1x1xbf16>, %arg35: tensor<256x1024x1x1xbf16>, %arg36: tensor<256x256x3x3xbf16>, %arg37: tensor<1024x256x1x1xbf16>, %arg38: tensor<256x1024x1x1xbf16>, %arg39: tensor<256x256x3x3xbf16>, %arg40: tensor<1024x256x1x1xbf16>, %arg41: tensor<256x1024x1x1xbf16>, %arg42: tensor<256x256x3x3xbf16>, %arg43: tensor<1024x256x1x1xbf16>, %arg44: tensor<512x1024x1x1xbf16>, %arg45: tensor<512x512x3x3xbf16>, %arg46: tensor<2048x512x1x1xbf16>, %arg47: tensor<2048x1024x1x1xbf16>, %arg48: tensor<512x2048x1x1xbf16>, %arg49: tensor<512x512x3x3xbf16>, %arg50: tensor<2048x512x1x1xbf16>, %arg51: tensor<512x2048x1x1xbf16>, %arg52: tensor<512x512x3x3xbf16>, %arg53: tensor<2048x512x1x1xbf16>, %arg54: tensor<64x1x1xf32>, %arg55: tensor<64x1x1xf32>, %arg56: tensor<64x1x1xbf16>, %arg57: tensor<64x1x1xbf16>, %arg58: tensor<64x1x1xf32>, %arg59: tensor<64x1x1xf32>, %arg60: tensor<64x1x1xbf16>, %arg61: tensor<64x1x1xbf16>, %arg62: tensor<64x1x1xf32>, %arg63: tensor<64x1x1xf32>, %arg64: tensor<64x1x1xbf16>, %arg65: tensor<64x1x1xbf16>, %arg66: tensor<256x1x1xf32>, %arg67: tensor<256x1x1xf32>, %arg68: tensor<256x1x1xbf16>, %arg69: tensor<256x1x1xbf16>, %arg70: tensor<256x1x1xf32>, %arg71: tensor<256x1x1xf32>, %arg72: tensor<256x1x1xbf16>, %arg73: tensor<256x1x1xbf16>, %arg74: tensor<64x1x1xf32>, %arg75: tensor<64x1x1xf32>, %arg76: tensor<64x1x1xbf16>, %arg77: tensor<64x1x1xbf16>, %arg78: tensor<64x1x1xf32>, %arg79: tensor<64x1x1xf32>, %arg80: tensor<64x1x1xbf16>, %arg81: tensor<64x1x1xbf16>, %arg82: tensor<256x1x1xf32>, %arg83: tensor<256x1x1xf32>, %arg84: tensor<256x1x1xbf16>, %arg85: tensor<256x1x1xbf16>, %arg86: tensor<64x1x1xf32>, %arg87: tensor<64x1x1xf32>, %arg88: tensor<64x1x1xbf16>, %arg89: tensor<64x1x1xbf16>, %arg90: tensor<64x1x1xf32>, %arg91: tensor<64x1x1xf32>, %arg92: tensor<64x1x1xbf16>, %arg93: tensor<64x1x1xbf16>, %arg94: tensor<256x1x1xf32>, %arg95: tensor<256x1x1xf32>, %arg96: tensor<256x1x1xbf16>, %arg97: tensor<256x1x1xbf16>, %arg98: tensor<128x1x1xf32>, %arg99: tensor<128x1x1xf32>, %arg100: tensor<128x1x1xbf16>, %arg101: tensor<128x1x1xbf16>, %arg102: tensor<128x1x1xf32>, %arg103: tensor<128x1x1xf32>, %arg104: tensor<128x1x1xbf16>, %arg105: tensor<128x1x1xbf16>, %arg106: tensor<512x1x1xf32>, %arg107: tensor<512x1x1xf32>, %arg108: tensor<512x1x1xbf16>, %arg109: tensor<512x1x1xbf16>, %arg110: tensor<512x1x1xf32>, %arg111: tensor<512x1x1xf32>, %arg112: tensor<512x1x1xbf16>, %arg113: tensor<512x1x1xbf16>, %arg114: tensor<128x1x1xf32>, %arg115: tensor<128x1x1xf32>, %arg116: tensor<128x1x1xbf16>, %arg117: tensor<128x1x1xbf16>, %arg118: tensor<128x1x1xf32>, %arg119: tensor<128x1x1xf32>, %arg120: tensor<128x1x1xbf16>, %arg121: tensor<128x1x1xbf16>, %arg122: tensor<512x1x1xf32>, %arg123: tensor<512x1x1xf32>, %arg124: tensor<512x1x1xbf16>, %arg125: tensor<512x1x1xbf16>, %arg126: tensor<128x1x1xf32>, %arg127: tensor<128x1x1xf32>, %arg128: tensor<128x1x1xbf16>, %arg129: tensor<128x1x1xbf16>, %arg130: tensor<128x1x1xf32>, %arg131: tensor<128x1x1xf32>, %arg132: tensor<128x1x1xbf16>, %arg133: tensor<128x1x1xbf16>, %arg134: tensor<512x1x1xf32>, %arg135: tensor<512x1x1xf32>, %arg136: tensor<512x1x1xbf16>, %arg137: tensor<512x1x1xbf16>, %arg138: tensor<128x1x1xf32>, %arg139: tensor<128x1x1xf32>, %arg140: tensor<128x1x1xbf16>, %arg141: tensor<128x1x1xbf16>, %arg142: tensor<128x1x1xf32>, %arg143: tensor<128x1x1xf32>, %arg144: tensor<128x1x1xbf16>, %arg145: tensor<128x1x1xbf16>, %arg146: tensor<512x1x1xf32>, %arg147: tensor<512x1x1xf32>, %arg148: tensor<512x1x1xbf16>, %arg149: tensor<512x1x1xbf16>, %arg150: tensor<256x1x1xf32>, %arg151: tensor<256x1x1xf32>, %arg152: tensor<256x1x1xbf16>, %arg153: tensor<256x1x1xbf16>, %arg154: tensor<256x1x1xf32>, %arg155: tensor<256x1x1xf32>, %arg156: tensor<256x1x1xbf16>, %arg157: tensor<256x1x1xbf16>, %arg158: tensor<1024x1x1xf32>, %arg159: tensor<1024x1x1xf32>, %arg160: tensor<1024x1x1xbf16>, %arg161: tensor<1024x1x1xbf16>, %arg162: tensor<1024x1x1xf32>, %arg163: tensor<1024x1x1xf32>, %arg164: tensor<1024x1x1xbf16>, %arg165: tensor<1024x1x1xbf16>, %arg166: tensor<256x1x1xf32>, %arg167: tensor<256x1x1xf32>, %arg168: tensor<256x1x1xbf16>, %arg169: tensor<256x1x1xbf16>, %arg170: tensor<256x1x1xf32>, %arg171: tensor<256x1x1xf32>, %arg172: tensor<256x1x1xbf16>, %arg173: tensor<256x1x1xbf16>, %arg174: tensor<1024x1x1xf32>, %arg175: tensor<1024x1x1xf32>, %arg176: tensor<1024x1x1xbf16>, %arg177: tensor<1024x1x1xbf16>, %arg178: tensor<256x1x1xf32>, %arg179: tensor<256x1x1xf32>, %arg180: tensor<256x1x1xbf16>, %arg181: tensor<256x1x1xbf16>, %arg182: tensor<256x1x1xf32>, %arg183: tensor<256x1x1xf32>, %arg184: tensor<256x1x1xbf16>, %arg185: tensor<256x1x1xbf16>, %arg186: tensor<1024x1x1xf32>, %arg187: tensor<1024x1x1xf32>, %arg188: tensor<1024x1x1xbf16>, %arg189: tensor<1024x1x1xbf16>, %arg190: tensor<256x1x1xf32>, %arg191: tensor<256x1x1xf32>, %arg192: tensor<256x1x1xbf16>, %arg193: tensor<256x1x1xbf16>, %arg194: tensor<256x1x1xf32>, %arg195: tensor<256x1x1xf32>, %arg196: tensor<256x1x1xbf16>, %arg197: tensor<256x1x1xbf16>, %arg198: tensor<1024x1x1xf32>, %arg199: tensor<1024x1x1xf32>, %arg200: tensor<1024x1x1xbf16>, %arg201: tensor<1024x1x1xbf16>, %arg202: tensor<256x1x1xf32>, %arg203: tensor<256x1x1xf32>, %arg204: tensor<256x1x1xbf16>, %arg205: tensor<256x1x1xbf16>, %arg206: tensor<256x1x1xf32>, %arg207: tensor<256x1x1xf32>, %arg208: tensor<256x1x1xbf16>, %arg209: tensor<256x1x1xbf16>, %arg210: tensor<1024x1x1xf32>, %arg211: tensor<1024x1x1xf32>, %arg212: tensor<1024x1x1xbf16>, %arg213: tensor<1024x1x1xbf16>, %arg214: tensor<256x1x1xf32>, %arg215: tensor<256x1x1xf32>, %arg216: tensor<256x1x1xbf16>, %arg217: tensor<256x1x1xbf16>, %arg218: tensor<256x1x1xf32>, %arg219: tensor<256x1x1xf32>, %arg220: tensor<256x1x1xbf16>, %arg221: tensor<256x1x1xbf16>, %arg222: tensor<1024x1x1xf32>, %arg223: tensor<1024x1x1xf32>, %arg224: tensor<1024x1x1xbf16>, %arg225: tensor<1024x1x1xbf16>, %arg226: tensor<512x1x1xf32>, %arg227: tensor<512x1x1xf32>, %arg228: tensor<512x1x1xbf16>, %arg229: tensor<512x1x1xbf16>, %arg230: tensor<512x1x1xf32>, %arg231: tensor<512x1x1xf32>, %arg232: tensor<512x1x1xbf16>, %arg233: tensor<512x1x1xbf16>, %arg234: tensor<2048x1x1xf32>, %arg235: tensor<2048x1x1xf32>, %arg236: tensor<2048x1x1xbf16>, %arg237: tensor<2048x1x1xbf16>, %arg238: tensor<2048x1x1xf32>, %arg239: tensor<2048x1x1xf32>, %arg240: tensor<2048x1x1xbf16>, %arg241: tensor<2048x1x1xbf16>, %arg242: tensor<512x1x1xf32>, %arg243: tensor<512x1x1xf32>, %arg244: tensor<512x1x1xbf16>, %arg245: tensor<512x1x1xbf16>, %arg246: tensor<512x1x1xf32>, %arg247: tensor<512x1x1xf32>, %arg248: tensor<512x1x1xbf16>, %arg249: tensor<512x1x1xbf16>, %arg250: tensor<2048x1x1xf32>, %arg251: tensor<2048x1x1xf32>, %arg252: tensor<2048x1x1xbf16>, %arg253: tensor<2048x1x1xbf16>, %arg254: tensor<512x1x1xf32>, %arg255: tensor<512x1x1xf32>, %arg256: tensor<512x1x1xbf16>, %arg257: tensor<512x1x1xbf16>, %arg258: tensor<512x1x1xf32>, %arg259: tensor<512x1x1xf32>, %arg260: tensor<512x1x1xbf16>, %arg261: tensor<512x1x1xbf16>, %arg262: tensor<2048x1x1xf32>, %arg263: tensor<2048x1x1xf32>, %arg264: tensor<2048x1x1xbf16>, %arg265: tensor<2048x1x1xbf16>, %arg266: tensor<2048x1000xf32>, %arg267: tensor<1000xf32>) -> tensor<1x1000xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x64x112x112xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x56x56xbf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x56x56xbf16>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x56x56xbf16>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x28x28xbf16>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x28x28xbf16>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x28x28xbf16>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x14x14xbf16>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<1x1024x14x14xbf16>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x14x14xbf16>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x7x7xbf16>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<1x2048x7x7xbf16>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_13 = arith.constant dense<49> : tensor<1xi64>
    %cst_14 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x112x112xbf16>
    %1 = stablehlo.convert %0 : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = stablehlo.broadcast_in_dim %arg54, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %4 = stablehlo.subtract %2, %3 : tensor<1x64x112x112xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg55, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x64x112x112xf32>
    %8 = stablehlo.convert %arg56 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x64x112x112xf32>
    %12 = stablehlo.convert %arg57 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %15 = stablehlo.add %13, %14 : tensor<1x64x112x112xf32>
    %16 = stablehlo.convert %15 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xbf16>
    %17 = stablehlo.maximum %16, %cst : tensor<1x64x112x112xbf16>
    %18 = "stablehlo.reduce_window"(%17, %cst_0) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg268: tensor<bf16>, %arg269: tensor<bf16>):
      %986 = stablehlo.maximum %arg268, %arg269 : tensor<bf16>
      stablehlo.return %986 : tensor<bf16>
    }) : (tensor<1x64x112x112xbf16>, tensor<bf16>) -> tensor<1x64x56x56xbf16>
    %19 = stablehlo.convolution(%18, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<1x64x56x56xbf16>
    %20 = stablehlo.convert %19 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %22 = stablehlo.broadcast_in_dim %arg58, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %23 = stablehlo.subtract %21, %22 : tensor<1x64x56x56xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %25 = stablehlo.broadcast_in_dim %arg59, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<1x64x56x56xf32>
    %27 = stablehlo.convert %arg60 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %28 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %29 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<1x64x56x56xf32>
    %31 = stablehlo.convert %arg61 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %33 = stablehlo.broadcast_in_dim %31, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %34 = stablehlo.add %32, %33 : tensor<1x64x56x56xf32>
    %35 = stablehlo.convert %34 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %36 = stablehlo.maximum %35, %cst_1 : tensor<1x64x56x56xbf16>
    %37 = stablehlo.convolution(%36, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %38 = stablehlo.convert %37 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %40 = stablehlo.broadcast_in_dim %arg62, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %41 = stablehlo.subtract %39, %40 : tensor<1x64x56x56xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %43 = stablehlo.broadcast_in_dim %arg63, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<1x64x56x56xf32>
    %45 = stablehlo.convert %arg64 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %48 = stablehlo.multiply %46, %47 : tensor<1x64x56x56xf32>
    %49 = stablehlo.convert %arg65 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %51 = stablehlo.broadcast_in_dim %49, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x64x56x56xf32>
    %53 = stablehlo.convert %52 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %54 = stablehlo.maximum %53, %cst_1 : tensor<1x64x56x56xbf16>
    %55 = stablehlo.convolution(%54, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x56x56xbf16>
    %56 = stablehlo.convert %55 : (tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %58 = stablehlo.broadcast_in_dim %arg66, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %59 = stablehlo.subtract %57, %58 : tensor<1x256x56x56xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %61 = stablehlo.broadcast_in_dim %arg67, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %62 = stablehlo.multiply %60, %61 : tensor<1x256x56x56xf32>
    %63 = stablehlo.convert %arg68 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %64 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %65 = stablehlo.broadcast_in_dim %63, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %66 = stablehlo.multiply %64, %65 : tensor<1x256x56x56xf32>
    %67 = stablehlo.convert %arg69 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %68 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %69 = stablehlo.broadcast_in_dim %67, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %70 = stablehlo.add %68, %69 : tensor<1x256x56x56xf32>
    %71 = stablehlo.convert %70 : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xbf16>
    %72 = stablehlo.convolution(%18, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x56x56xbf16>
    %73 = stablehlo.convert %72 : (tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %75 = stablehlo.broadcast_in_dim %arg70, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %76 = stablehlo.subtract %74, %75 : tensor<1x256x56x56xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %78 = stablehlo.broadcast_in_dim %arg71, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %79 = stablehlo.multiply %77, %78 : tensor<1x256x56x56xf32>
    %80 = stablehlo.convert %arg72 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %81 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %82 = stablehlo.broadcast_in_dim %80, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<1x256x56x56xf32>
    %84 = stablehlo.convert %arg73 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %85 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %86 = stablehlo.broadcast_in_dim %84, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %87 = stablehlo.add %85, %86 : tensor<1x256x56x56xf32>
    %88 = stablehlo.convert %87 : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xbf16>
    %89 = stablehlo.add %71, %88 : tensor<1x256x56x56xbf16>
    %90 = stablehlo.maximum %89, %cst_2 : tensor<1x256x56x56xbf16>
    %91 = stablehlo.convolution(%90, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<1x64x56x56xbf16>
    %92 = stablehlo.convert %91 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %94 = stablehlo.broadcast_in_dim %arg74, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %95 = stablehlo.subtract %93, %94 : tensor<1x64x56x56xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %97 = stablehlo.broadcast_in_dim %arg75, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %98 = stablehlo.multiply %96, %97 : tensor<1x64x56x56xf32>
    %99 = stablehlo.convert %arg76 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %100 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %101 = stablehlo.broadcast_in_dim %99, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %102 = stablehlo.multiply %100, %101 : tensor<1x64x56x56xf32>
    %103 = stablehlo.convert %arg77 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %104 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %105 = stablehlo.broadcast_in_dim %103, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %106 = stablehlo.add %104, %105 : tensor<1x64x56x56xf32>
    %107 = stablehlo.convert %106 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %108 = stablehlo.maximum %107, %cst_1 : tensor<1x64x56x56xbf16>
    %109 = stablehlo.convolution(%108, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %110 = stablehlo.convert %109 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %112 = stablehlo.broadcast_in_dim %arg78, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %113 = stablehlo.subtract %111, %112 : tensor<1x64x56x56xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %115 = stablehlo.broadcast_in_dim %arg79, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %116 = stablehlo.multiply %114, %115 : tensor<1x64x56x56xf32>
    %117 = stablehlo.convert %arg80 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %118 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %119 = stablehlo.broadcast_in_dim %117, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %120 = stablehlo.multiply %118, %119 : tensor<1x64x56x56xf32>
    %121 = stablehlo.convert %arg81 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %122 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %123 = stablehlo.broadcast_in_dim %121, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %124 = stablehlo.add %122, %123 : tensor<1x64x56x56xf32>
    %125 = stablehlo.convert %124 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %126 = stablehlo.maximum %125, %cst_1 : tensor<1x64x56x56xbf16>
    %127 = stablehlo.convolution(%126, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x56x56xbf16>
    %128 = stablehlo.convert %127 : (tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %130 = stablehlo.broadcast_in_dim %arg82, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %131 = stablehlo.subtract %129, %130 : tensor<1x256x56x56xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %133 = stablehlo.broadcast_in_dim %arg83, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %134 = stablehlo.multiply %132, %133 : tensor<1x256x56x56xf32>
    %135 = stablehlo.convert %arg84 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %136 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %137 = stablehlo.broadcast_in_dim %135, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %138 = stablehlo.multiply %136, %137 : tensor<1x256x56x56xf32>
    %139 = stablehlo.convert %arg85 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %140 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %141 = stablehlo.broadcast_in_dim %139, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %142 = stablehlo.add %140, %141 : tensor<1x256x56x56xf32>
    %143 = stablehlo.convert %142 : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xbf16>
    %144 = stablehlo.add %143, %90 : tensor<1x256x56x56xbf16>
    %145 = stablehlo.maximum %144, %cst_2 : tensor<1x256x56x56xbf16>
    %146 = stablehlo.convolution(%145, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<1x64x56x56xbf16>
    %147 = stablehlo.convert %146 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %149 = stablehlo.broadcast_in_dim %arg86, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %150 = stablehlo.subtract %148, %149 : tensor<1x64x56x56xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %152 = stablehlo.broadcast_in_dim %arg87, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %153 = stablehlo.multiply %151, %152 : tensor<1x64x56x56xf32>
    %154 = stablehlo.convert %arg88 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %155 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %156 = stablehlo.broadcast_in_dim %154, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %157 = stablehlo.multiply %155, %156 : tensor<1x64x56x56xf32>
    %158 = stablehlo.convert %arg89 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %159 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %160 = stablehlo.broadcast_in_dim %158, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %161 = stablehlo.add %159, %160 : tensor<1x64x56x56xf32>
    %162 = stablehlo.convert %161 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %163 = stablehlo.maximum %162, %cst_1 : tensor<1x64x56x56xbf16>
    %164 = stablehlo.convolution(%163, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x56x56xbf16>
    %165 = stablehlo.convert %164 : (tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xf32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %167 = stablehlo.broadcast_in_dim %arg90, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %168 = stablehlo.subtract %166, %167 : tensor<1x64x56x56xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %170 = stablehlo.broadcast_in_dim %arg91, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<1x64x56x56xf32>
    %172 = stablehlo.convert %arg92 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %173 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %174 = stablehlo.broadcast_in_dim %172, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %175 = stablehlo.multiply %173, %174 : tensor<1x64x56x56xf32>
    %176 = stablehlo.convert %arg93 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %177 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2, 3] : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %178 = stablehlo.broadcast_in_dim %176, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %179 = stablehlo.add %177, %178 : tensor<1x64x56x56xf32>
    %180 = stablehlo.convert %179 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xbf16>
    %181 = stablehlo.maximum %180, %cst_1 : tensor<1x64x56x56xbf16>
    %182 = stablehlo.convolution(%181, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x56x56xbf16>
    %183 = stablehlo.convert %182 : (tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %185 = stablehlo.broadcast_in_dim %arg94, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %186 = stablehlo.subtract %184, %185 : tensor<1x256x56x56xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %188 = stablehlo.broadcast_in_dim %arg95, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %189 = stablehlo.multiply %187, %188 : tensor<1x256x56x56xf32>
    %190 = stablehlo.convert %arg96 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %191 = stablehlo.broadcast_in_dim %189, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %192 = stablehlo.broadcast_in_dim %190, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %193 = stablehlo.multiply %191, %192 : tensor<1x256x56x56xf32>
    %194 = stablehlo.convert %arg97 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %195 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2, 3] : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %196 = stablehlo.broadcast_in_dim %194, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %197 = stablehlo.add %195, %196 : tensor<1x256x56x56xf32>
    %198 = stablehlo.convert %197 : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xbf16>
    %199 = stablehlo.add %198, %145 : tensor<1x256x56x56xbf16>
    %200 = stablehlo.maximum %199, %cst_2 : tensor<1x256x56x56xbf16>
    %201 = stablehlo.convolution(%200, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x56x56xbf16>
    %202 = stablehlo.convert %201 : (tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %204 = stablehlo.broadcast_in_dim %arg98, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %205 = stablehlo.subtract %203, %204 : tensor<1x128x56x56xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %207 = stablehlo.broadcast_in_dim %arg99, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %208 = stablehlo.multiply %206, %207 : tensor<1x128x56x56xf32>
    %209 = stablehlo.convert %arg100 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %210 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %211 = stablehlo.broadcast_in_dim %209, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %212 = stablehlo.multiply %210, %211 : tensor<1x128x56x56xf32>
    %213 = stablehlo.convert %arg101 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %214 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2, 3] : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %215 = stablehlo.broadcast_in_dim %213, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %216 = stablehlo.add %214, %215 : tensor<1x128x56x56xf32>
    %217 = stablehlo.convert %216 : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xbf16>
    %218 = stablehlo.maximum %217, %cst_3 : tensor<1x128x56x56xbf16>
    %219 = stablehlo.convolution(%218, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %220 = stablehlo.convert %219 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %222 = stablehlo.broadcast_in_dim %arg102, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %223 = stablehlo.subtract %221, %222 : tensor<1x128x28x28xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %225 = stablehlo.broadcast_in_dim %arg103, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<1x128x28x28xf32>
    %227 = stablehlo.convert %arg104 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %228 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %229 = stablehlo.broadcast_in_dim %227, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %230 = stablehlo.multiply %228, %229 : tensor<1x128x28x28xf32>
    %231 = stablehlo.convert %arg105 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %232 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %233 = stablehlo.broadcast_in_dim %231, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %234 = stablehlo.add %232, %233 : tensor<1x128x28x28xf32>
    %235 = stablehlo.convert %234 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %236 = stablehlo.maximum %235, %cst_4 : tensor<1x128x28x28xbf16>
    %237 = stablehlo.convolution(%236, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %238 = stablehlo.convert %237 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %240 = stablehlo.broadcast_in_dim %arg106, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %241 = stablehlo.subtract %239, %240 : tensor<1x512x28x28xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %243 = stablehlo.broadcast_in_dim %arg107, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %244 = stablehlo.multiply %242, %243 : tensor<1x512x28x28xf32>
    %245 = stablehlo.convert %arg108 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %246 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %247 = stablehlo.broadcast_in_dim %245, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %248 = stablehlo.multiply %246, %247 : tensor<1x512x28x28xf32>
    %249 = stablehlo.convert %arg109 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %250 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %251 = stablehlo.broadcast_in_dim %249, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %252 = stablehlo.add %250, %251 : tensor<1x512x28x28xf32>
    %253 = stablehlo.convert %252 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %254 = stablehlo.convolution(%200, %arg15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xbf16>, tensor<512x256x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %255 = stablehlo.convert %254 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %257 = stablehlo.broadcast_in_dim %arg110, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %258 = stablehlo.subtract %256, %257 : tensor<1x512x28x28xf32>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %260 = stablehlo.broadcast_in_dim %arg111, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %261 = stablehlo.multiply %259, %260 : tensor<1x512x28x28xf32>
    %262 = stablehlo.convert %arg112 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %263 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %264 = stablehlo.broadcast_in_dim %262, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %265 = stablehlo.multiply %263, %264 : tensor<1x512x28x28xf32>
    %266 = stablehlo.convert %arg113 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %267 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %268 = stablehlo.broadcast_in_dim %266, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %269 = stablehlo.add %267, %268 : tensor<1x512x28x28xf32>
    %270 = stablehlo.convert %269 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %271 = stablehlo.add %253, %270 : tensor<1x512x28x28xbf16>
    %272 = stablehlo.maximum %271, %cst_5 : tensor<1x512x28x28xbf16>
    %273 = stablehlo.convolution(%272, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %274 = stablehlo.convert %273 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %276 = stablehlo.broadcast_in_dim %arg114, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %277 = stablehlo.subtract %275, %276 : tensor<1x128x28x28xf32>
    %278 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %279 = stablehlo.broadcast_in_dim %arg115, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %280 = stablehlo.multiply %278, %279 : tensor<1x128x28x28xf32>
    %281 = stablehlo.convert %arg116 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %282 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %283 = stablehlo.broadcast_in_dim %281, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %284 = stablehlo.multiply %282, %283 : tensor<1x128x28x28xf32>
    %285 = stablehlo.convert %arg117 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %286 = stablehlo.broadcast_in_dim %284, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %287 = stablehlo.broadcast_in_dim %285, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %288 = stablehlo.add %286, %287 : tensor<1x128x28x28xf32>
    %289 = stablehlo.convert %288 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %290 = stablehlo.maximum %289, %cst_4 : tensor<1x128x28x28xbf16>
    %291 = stablehlo.convolution(%290, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %292 = stablehlo.convert %291 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %294 = stablehlo.broadcast_in_dim %arg118, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %295 = stablehlo.subtract %293, %294 : tensor<1x128x28x28xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %297 = stablehlo.broadcast_in_dim %arg119, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %298 = stablehlo.multiply %296, %297 : tensor<1x128x28x28xf32>
    %299 = stablehlo.convert %arg120 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %300 = stablehlo.broadcast_in_dim %298, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %301 = stablehlo.broadcast_in_dim %299, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %302 = stablehlo.multiply %300, %301 : tensor<1x128x28x28xf32>
    %303 = stablehlo.convert %arg121 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %304 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %305 = stablehlo.broadcast_in_dim %303, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %306 = stablehlo.add %304, %305 : tensor<1x128x28x28xf32>
    %307 = stablehlo.convert %306 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %308 = stablehlo.maximum %307, %cst_4 : tensor<1x128x28x28xbf16>
    %309 = stablehlo.convolution(%308, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %310 = stablehlo.convert %309 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %312 = stablehlo.broadcast_in_dim %arg122, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %313 = stablehlo.subtract %311, %312 : tensor<1x512x28x28xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %315 = stablehlo.broadcast_in_dim %arg123, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %316 = stablehlo.multiply %314, %315 : tensor<1x512x28x28xf32>
    %317 = stablehlo.convert %arg124 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %318 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %319 = stablehlo.broadcast_in_dim %317, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %320 = stablehlo.multiply %318, %319 : tensor<1x512x28x28xf32>
    %321 = stablehlo.convert %arg125 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %322 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %323 = stablehlo.broadcast_in_dim %321, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %324 = stablehlo.add %322, %323 : tensor<1x512x28x28xf32>
    %325 = stablehlo.convert %324 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %326 = stablehlo.add %325, %272 : tensor<1x512x28x28xbf16>
    %327 = stablehlo.maximum %326, %cst_5 : tensor<1x512x28x28xbf16>
    %328 = stablehlo.convolution(%327, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %329 = stablehlo.convert %328 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %331 = stablehlo.broadcast_in_dim %arg126, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %332 = stablehlo.subtract %330, %331 : tensor<1x128x28x28xf32>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %334 = stablehlo.broadcast_in_dim %arg127, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %335 = stablehlo.multiply %333, %334 : tensor<1x128x28x28xf32>
    %336 = stablehlo.convert %arg128 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %337 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %338 = stablehlo.broadcast_in_dim %336, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %339 = stablehlo.multiply %337, %338 : tensor<1x128x28x28xf32>
    %340 = stablehlo.convert %arg129 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %341 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %342 = stablehlo.broadcast_in_dim %340, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %343 = stablehlo.add %341, %342 : tensor<1x128x28x28xf32>
    %344 = stablehlo.convert %343 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %345 = stablehlo.maximum %344, %cst_4 : tensor<1x128x28x28xbf16>
    %346 = stablehlo.convolution(%345, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %347 = stablehlo.convert %346 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %349 = stablehlo.broadcast_in_dim %arg130, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %350 = stablehlo.subtract %348, %349 : tensor<1x128x28x28xf32>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %352 = stablehlo.broadcast_in_dim %arg131, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %353 = stablehlo.multiply %351, %352 : tensor<1x128x28x28xf32>
    %354 = stablehlo.convert %arg132 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %355 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %356 = stablehlo.broadcast_in_dim %354, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %357 = stablehlo.multiply %355, %356 : tensor<1x128x28x28xf32>
    %358 = stablehlo.convert %arg133 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %359 = stablehlo.broadcast_in_dim %357, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %360 = stablehlo.broadcast_in_dim %358, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %361 = stablehlo.add %359, %360 : tensor<1x128x28x28xf32>
    %362 = stablehlo.convert %361 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %363 = stablehlo.maximum %362, %cst_4 : tensor<1x128x28x28xbf16>
    %364 = stablehlo.convolution(%363, %arg21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %365 = stablehlo.convert %364 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %367 = stablehlo.broadcast_in_dim %arg134, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %368 = stablehlo.subtract %366, %367 : tensor<1x512x28x28xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %370 = stablehlo.broadcast_in_dim %arg135, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %371 = stablehlo.multiply %369, %370 : tensor<1x512x28x28xf32>
    %372 = stablehlo.convert %arg136 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %373 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %374 = stablehlo.broadcast_in_dim %372, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %375 = stablehlo.multiply %373, %374 : tensor<1x512x28x28xf32>
    %376 = stablehlo.convert %arg137 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %377 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %378 = stablehlo.broadcast_in_dim %376, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %379 = stablehlo.add %377, %378 : tensor<1x512x28x28xf32>
    %380 = stablehlo.convert %379 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %381 = stablehlo.add %380, %327 : tensor<1x512x28x28xbf16>
    %382 = stablehlo.maximum %381, %cst_5 : tensor<1x512x28x28xbf16>
    %383 = stablehlo.convolution(%382, %arg22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %384 = stablehlo.convert %383 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %386 = stablehlo.broadcast_in_dim %arg138, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %387 = stablehlo.subtract %385, %386 : tensor<1x128x28x28xf32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %389 = stablehlo.broadcast_in_dim %arg139, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %390 = stablehlo.multiply %388, %389 : tensor<1x128x28x28xf32>
    %391 = stablehlo.convert %arg140 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %392 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %393 = stablehlo.broadcast_in_dim %391, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %394 = stablehlo.multiply %392, %393 : tensor<1x128x28x28xf32>
    %395 = stablehlo.convert %arg141 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %396 = stablehlo.broadcast_in_dim %394, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %397 = stablehlo.broadcast_in_dim %395, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %398 = stablehlo.add %396, %397 : tensor<1x128x28x28xf32>
    %399 = stablehlo.convert %398 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %400 = stablehlo.maximum %399, %cst_4 : tensor<1x128x28x28xbf16>
    %401 = stablehlo.convolution(%400, %arg23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x28x28xbf16>
    %402 = stablehlo.convert %401 : (tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xf32>
    %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %404 = stablehlo.broadcast_in_dim %arg142, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %405 = stablehlo.subtract %403, %404 : tensor<1x128x28x28xf32>
    %406 = stablehlo.broadcast_in_dim %405, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %407 = stablehlo.broadcast_in_dim %arg143, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %408 = stablehlo.multiply %406, %407 : tensor<1x128x28x28xf32>
    %409 = stablehlo.convert %arg144 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %410 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %411 = stablehlo.broadcast_in_dim %409, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %412 = stablehlo.multiply %410, %411 : tensor<1x128x28x28xf32>
    %413 = stablehlo.convert %arg145 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %414 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2, 3] : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %415 = stablehlo.broadcast_in_dim %413, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %416 = stablehlo.add %414, %415 : tensor<1x128x28x28xf32>
    %417 = stablehlo.convert %416 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xbf16>
    %418 = stablehlo.maximum %417, %cst_4 : tensor<1x128x28x28xbf16>
    %419 = stablehlo.convolution(%418, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %420 = stablehlo.convert %419 : (tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xf32>
    %421 = stablehlo.broadcast_in_dim %420, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %422 = stablehlo.broadcast_in_dim %arg146, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %423 = stablehlo.subtract %421, %422 : tensor<1x512x28x28xf32>
    %424 = stablehlo.broadcast_in_dim %423, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %425 = stablehlo.broadcast_in_dim %arg147, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %426 = stablehlo.multiply %424, %425 : tensor<1x512x28x28xf32>
    %427 = stablehlo.convert %arg148 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %428 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %429 = stablehlo.broadcast_in_dim %427, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %430 = stablehlo.multiply %428, %429 : tensor<1x512x28x28xf32>
    %431 = stablehlo.convert %arg149 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %432 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %433 = stablehlo.broadcast_in_dim %431, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %434 = stablehlo.add %432, %433 : tensor<1x512x28x28xf32>
    %435 = stablehlo.convert %434 : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xbf16>
    %436 = stablehlo.add %435, %382 : tensor<1x512x28x28xbf16>
    %437 = stablehlo.maximum %436, %cst_5 : tensor<1x512x28x28xbf16>
    %438 = stablehlo.convolution(%437, %arg25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x28x28xbf16>
    %439 = stablehlo.convert %438 : (tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xf32>
    %440 = stablehlo.broadcast_in_dim %439, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %441 = stablehlo.broadcast_in_dim %arg150, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %442 = stablehlo.subtract %440, %441 : tensor<1x256x28x28xf32>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %444 = stablehlo.broadcast_in_dim %arg151, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %445 = stablehlo.multiply %443, %444 : tensor<1x256x28x28xf32>
    %446 = stablehlo.convert %arg152 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %447 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %448 = stablehlo.broadcast_in_dim %446, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %449 = stablehlo.multiply %447, %448 : tensor<1x256x28x28xf32>
    %450 = stablehlo.convert %arg153 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %451 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2, 3] : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %452 = stablehlo.broadcast_in_dim %450, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %453 = stablehlo.add %451, %452 : tensor<1x256x28x28xf32>
    %454 = stablehlo.convert %453 : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xbf16>
    %455 = stablehlo.maximum %454, %cst_6 : tensor<1x256x28x28xbf16>
    %456 = stablehlo.convolution(%455, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %457 = stablehlo.convert %456 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %459 = stablehlo.broadcast_in_dim %arg154, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %460 = stablehlo.subtract %458, %459 : tensor<1x256x14x14xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %462 = stablehlo.broadcast_in_dim %arg155, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %463 = stablehlo.multiply %461, %462 : tensor<1x256x14x14xf32>
    %464 = stablehlo.convert %arg156 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %465 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %466 = stablehlo.broadcast_in_dim %464, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %467 = stablehlo.multiply %465, %466 : tensor<1x256x14x14xf32>
    %468 = stablehlo.convert %arg157 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %469 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %470 = stablehlo.broadcast_in_dim %468, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %471 = stablehlo.add %469, %470 : tensor<1x256x14x14xf32>
    %472 = stablehlo.convert %471 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %473 = stablehlo.maximum %472, %cst_7 : tensor<1x256x14x14xbf16>
    %474 = stablehlo.convolution(%473, %arg27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %475 = stablehlo.convert %474 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %476 = stablehlo.broadcast_in_dim %475, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %477 = stablehlo.broadcast_in_dim %arg158, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %478 = stablehlo.subtract %476, %477 : tensor<1x1024x14x14xf32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %480 = stablehlo.broadcast_in_dim %arg159, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %481 = stablehlo.multiply %479, %480 : tensor<1x1024x14x14xf32>
    %482 = stablehlo.convert %arg160 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %483 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %484 = stablehlo.broadcast_in_dim %482, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %485 = stablehlo.multiply %483, %484 : tensor<1x1024x14x14xf32>
    %486 = stablehlo.convert %arg161 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %487 = stablehlo.broadcast_in_dim %485, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %488 = stablehlo.broadcast_in_dim %486, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %489 = stablehlo.add %487, %488 : tensor<1x1024x14x14xf32>
    %490 = stablehlo.convert %489 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %491 = stablehlo.convolution(%437, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %492 = stablehlo.convert %491 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %493 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %494 = stablehlo.broadcast_in_dim %arg162, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %495 = stablehlo.subtract %493, %494 : tensor<1x1024x14x14xf32>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %497 = stablehlo.broadcast_in_dim %arg163, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %498 = stablehlo.multiply %496, %497 : tensor<1x1024x14x14xf32>
    %499 = stablehlo.convert %arg164 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %500 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %501 = stablehlo.broadcast_in_dim %499, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %502 = stablehlo.multiply %500, %501 : tensor<1x1024x14x14xf32>
    %503 = stablehlo.convert %arg165 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %504 = stablehlo.broadcast_in_dim %502, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %505 = stablehlo.broadcast_in_dim %503, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %506 = stablehlo.add %504, %505 : tensor<1x1024x14x14xf32>
    %507 = stablehlo.convert %506 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %508 = stablehlo.add %490, %507 : tensor<1x1024x14x14xbf16>
    %509 = stablehlo.maximum %508, %cst_8 : tensor<1x1024x14x14xbf16>
    %510 = stablehlo.convolution(%509, %arg29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x14x14xbf16>
    %511 = stablehlo.convert %510 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %513 = stablehlo.broadcast_in_dim %arg166, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %514 = stablehlo.subtract %512, %513 : tensor<1x256x14x14xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %516 = stablehlo.broadcast_in_dim %arg167, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %517 = stablehlo.multiply %515, %516 : tensor<1x256x14x14xf32>
    %518 = stablehlo.convert %arg168 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %519 = stablehlo.broadcast_in_dim %517, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %520 = stablehlo.broadcast_in_dim %518, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %521 = stablehlo.multiply %519, %520 : tensor<1x256x14x14xf32>
    %522 = stablehlo.convert %arg169 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %523 = stablehlo.broadcast_in_dim %521, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %524 = stablehlo.broadcast_in_dim %522, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %525 = stablehlo.add %523, %524 : tensor<1x256x14x14xf32>
    %526 = stablehlo.convert %525 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %527 = stablehlo.maximum %526, %cst_7 : tensor<1x256x14x14xbf16>
    %528 = stablehlo.convolution(%527, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %529 = stablehlo.convert %528 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %530 = stablehlo.broadcast_in_dim %529, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %531 = stablehlo.broadcast_in_dim %arg170, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %532 = stablehlo.subtract %530, %531 : tensor<1x256x14x14xf32>
    %533 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %534 = stablehlo.broadcast_in_dim %arg171, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %535 = stablehlo.multiply %533, %534 : tensor<1x256x14x14xf32>
    %536 = stablehlo.convert %arg172 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %537 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %538 = stablehlo.broadcast_in_dim %536, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %539 = stablehlo.multiply %537, %538 : tensor<1x256x14x14xf32>
    %540 = stablehlo.convert %arg173 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %541 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %542 = stablehlo.broadcast_in_dim %540, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %543 = stablehlo.add %541, %542 : tensor<1x256x14x14xf32>
    %544 = stablehlo.convert %543 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %545 = stablehlo.maximum %544, %cst_7 : tensor<1x256x14x14xbf16>
    %546 = stablehlo.convolution(%545, %arg31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %547 = stablehlo.convert %546 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %548 = stablehlo.broadcast_in_dim %547, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %549 = stablehlo.broadcast_in_dim %arg174, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %550 = stablehlo.subtract %548, %549 : tensor<1x1024x14x14xf32>
    %551 = stablehlo.broadcast_in_dim %550, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %552 = stablehlo.broadcast_in_dim %arg175, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %553 = stablehlo.multiply %551, %552 : tensor<1x1024x14x14xf32>
    %554 = stablehlo.convert %arg176 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %555 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %556 = stablehlo.broadcast_in_dim %554, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %557 = stablehlo.multiply %555, %556 : tensor<1x1024x14x14xf32>
    %558 = stablehlo.convert %arg177 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %559 = stablehlo.broadcast_in_dim %557, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %560 = stablehlo.broadcast_in_dim %558, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %561 = stablehlo.add %559, %560 : tensor<1x1024x14x14xf32>
    %562 = stablehlo.convert %561 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %563 = stablehlo.add %562, %509 : tensor<1x1024x14x14xbf16>
    %564 = stablehlo.maximum %563, %cst_8 : tensor<1x1024x14x14xbf16>
    %565 = stablehlo.convolution(%564, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x14x14xbf16>
    %566 = stablehlo.convert %565 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %568 = stablehlo.broadcast_in_dim %arg178, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %569 = stablehlo.subtract %567, %568 : tensor<1x256x14x14xf32>
    %570 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %571 = stablehlo.broadcast_in_dim %arg179, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %572 = stablehlo.multiply %570, %571 : tensor<1x256x14x14xf32>
    %573 = stablehlo.convert %arg180 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %574 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %575 = stablehlo.broadcast_in_dim %573, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %576 = stablehlo.multiply %574, %575 : tensor<1x256x14x14xf32>
    %577 = stablehlo.convert %arg181 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %578 = stablehlo.broadcast_in_dim %576, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %579 = stablehlo.broadcast_in_dim %577, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %580 = stablehlo.add %578, %579 : tensor<1x256x14x14xf32>
    %581 = stablehlo.convert %580 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %582 = stablehlo.maximum %581, %cst_7 : tensor<1x256x14x14xbf16>
    %583 = stablehlo.convolution(%582, %arg33) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %584 = stablehlo.convert %583 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %585 = stablehlo.broadcast_in_dim %584, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %586 = stablehlo.broadcast_in_dim %arg182, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %587 = stablehlo.subtract %585, %586 : tensor<1x256x14x14xf32>
    %588 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %589 = stablehlo.broadcast_in_dim %arg183, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %590 = stablehlo.multiply %588, %589 : tensor<1x256x14x14xf32>
    %591 = stablehlo.convert %arg184 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %592 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %593 = stablehlo.broadcast_in_dim %591, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %594 = stablehlo.multiply %592, %593 : tensor<1x256x14x14xf32>
    %595 = stablehlo.convert %arg185 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %596 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %597 = stablehlo.broadcast_in_dim %595, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %598 = stablehlo.add %596, %597 : tensor<1x256x14x14xf32>
    %599 = stablehlo.convert %598 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %600 = stablehlo.maximum %599, %cst_7 : tensor<1x256x14x14xbf16>
    %601 = stablehlo.convolution(%600, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %602 = stablehlo.convert %601 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %603 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %604 = stablehlo.broadcast_in_dim %arg186, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %605 = stablehlo.subtract %603, %604 : tensor<1x1024x14x14xf32>
    %606 = stablehlo.broadcast_in_dim %605, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %607 = stablehlo.broadcast_in_dim %arg187, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %608 = stablehlo.multiply %606, %607 : tensor<1x1024x14x14xf32>
    %609 = stablehlo.convert %arg188 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %610 = stablehlo.broadcast_in_dim %608, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %611 = stablehlo.broadcast_in_dim %609, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %612 = stablehlo.multiply %610, %611 : tensor<1x1024x14x14xf32>
    %613 = stablehlo.convert %arg189 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %614 = stablehlo.broadcast_in_dim %612, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %615 = stablehlo.broadcast_in_dim %613, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %616 = stablehlo.add %614, %615 : tensor<1x1024x14x14xf32>
    %617 = stablehlo.convert %616 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %618 = stablehlo.add %617, %564 : tensor<1x1024x14x14xbf16>
    %619 = stablehlo.maximum %618, %cst_8 : tensor<1x1024x14x14xbf16>
    %620 = stablehlo.convolution(%619, %arg35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x14x14xbf16>
    %621 = stablehlo.convert %620 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %622 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %623 = stablehlo.broadcast_in_dim %arg190, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %624 = stablehlo.subtract %622, %623 : tensor<1x256x14x14xf32>
    %625 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %626 = stablehlo.broadcast_in_dim %arg191, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %627 = stablehlo.multiply %625, %626 : tensor<1x256x14x14xf32>
    %628 = stablehlo.convert %arg192 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %629 = stablehlo.broadcast_in_dim %627, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %630 = stablehlo.broadcast_in_dim %628, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %631 = stablehlo.multiply %629, %630 : tensor<1x256x14x14xf32>
    %632 = stablehlo.convert %arg193 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %633 = stablehlo.broadcast_in_dim %631, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %634 = stablehlo.broadcast_in_dim %632, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %635 = stablehlo.add %633, %634 : tensor<1x256x14x14xf32>
    %636 = stablehlo.convert %635 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %637 = stablehlo.maximum %636, %cst_7 : tensor<1x256x14x14xbf16>
    %638 = stablehlo.convolution(%637, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %639 = stablehlo.convert %638 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %640 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %641 = stablehlo.broadcast_in_dim %arg194, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %642 = stablehlo.subtract %640, %641 : tensor<1x256x14x14xf32>
    %643 = stablehlo.broadcast_in_dim %642, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %644 = stablehlo.broadcast_in_dim %arg195, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %645 = stablehlo.multiply %643, %644 : tensor<1x256x14x14xf32>
    %646 = stablehlo.convert %arg196 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %647 = stablehlo.broadcast_in_dim %645, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %648 = stablehlo.broadcast_in_dim %646, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %649 = stablehlo.multiply %647, %648 : tensor<1x256x14x14xf32>
    %650 = stablehlo.convert %arg197 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %651 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %652 = stablehlo.broadcast_in_dim %650, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %653 = stablehlo.add %651, %652 : tensor<1x256x14x14xf32>
    %654 = stablehlo.convert %653 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %655 = stablehlo.maximum %654, %cst_7 : tensor<1x256x14x14xbf16>
    %656 = stablehlo.convolution(%655, %arg37) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %657 = stablehlo.convert %656 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %658 = stablehlo.broadcast_in_dim %657, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %659 = stablehlo.broadcast_in_dim %arg198, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %660 = stablehlo.subtract %658, %659 : tensor<1x1024x14x14xf32>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %662 = stablehlo.broadcast_in_dim %arg199, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %663 = stablehlo.multiply %661, %662 : tensor<1x1024x14x14xf32>
    %664 = stablehlo.convert %arg200 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %665 = stablehlo.broadcast_in_dim %663, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %666 = stablehlo.broadcast_in_dim %664, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %667 = stablehlo.multiply %665, %666 : tensor<1x1024x14x14xf32>
    %668 = stablehlo.convert %arg201 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %669 = stablehlo.broadcast_in_dim %667, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %670 = stablehlo.broadcast_in_dim %668, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %671 = stablehlo.add %669, %670 : tensor<1x1024x14x14xf32>
    %672 = stablehlo.convert %671 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %673 = stablehlo.add %672, %619 : tensor<1x1024x14x14xbf16>
    %674 = stablehlo.maximum %673, %cst_8 : tensor<1x1024x14x14xbf16>
    %675 = stablehlo.convolution(%674, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x14x14xbf16>
    %676 = stablehlo.convert %675 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %677 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %678 = stablehlo.broadcast_in_dim %arg202, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %679 = stablehlo.subtract %677, %678 : tensor<1x256x14x14xf32>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %681 = stablehlo.broadcast_in_dim %arg203, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %682 = stablehlo.multiply %680, %681 : tensor<1x256x14x14xf32>
    %683 = stablehlo.convert %arg204 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %684 = stablehlo.broadcast_in_dim %682, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %685 = stablehlo.broadcast_in_dim %683, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %686 = stablehlo.multiply %684, %685 : tensor<1x256x14x14xf32>
    %687 = stablehlo.convert %arg205 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %688 = stablehlo.broadcast_in_dim %686, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %689 = stablehlo.broadcast_in_dim %687, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %690 = stablehlo.add %688, %689 : tensor<1x256x14x14xf32>
    %691 = stablehlo.convert %690 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %692 = stablehlo.maximum %691, %cst_7 : tensor<1x256x14x14xbf16>
    %693 = stablehlo.convolution(%692, %arg39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %694 = stablehlo.convert %693 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %695 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %696 = stablehlo.broadcast_in_dim %arg206, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %697 = stablehlo.subtract %695, %696 : tensor<1x256x14x14xf32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %699 = stablehlo.broadcast_in_dim %arg207, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %700 = stablehlo.multiply %698, %699 : tensor<1x256x14x14xf32>
    %701 = stablehlo.convert %arg208 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %702 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %703 = stablehlo.broadcast_in_dim %701, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %704 = stablehlo.multiply %702, %703 : tensor<1x256x14x14xf32>
    %705 = stablehlo.convert %arg209 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %706 = stablehlo.broadcast_in_dim %704, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %707 = stablehlo.broadcast_in_dim %705, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %708 = stablehlo.add %706, %707 : tensor<1x256x14x14xf32>
    %709 = stablehlo.convert %708 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %710 = stablehlo.maximum %709, %cst_7 : tensor<1x256x14x14xbf16>
    %711 = stablehlo.convolution(%710, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %712 = stablehlo.convert %711 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %713 = stablehlo.broadcast_in_dim %712, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %714 = stablehlo.broadcast_in_dim %arg210, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %715 = stablehlo.subtract %713, %714 : tensor<1x1024x14x14xf32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %717 = stablehlo.broadcast_in_dim %arg211, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %718 = stablehlo.multiply %716, %717 : tensor<1x1024x14x14xf32>
    %719 = stablehlo.convert %arg212 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %720 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %721 = stablehlo.broadcast_in_dim %719, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %722 = stablehlo.multiply %720, %721 : tensor<1x1024x14x14xf32>
    %723 = stablehlo.convert %arg213 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %724 = stablehlo.broadcast_in_dim %722, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %725 = stablehlo.broadcast_in_dim %723, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %726 = stablehlo.add %724, %725 : tensor<1x1024x14x14xf32>
    %727 = stablehlo.convert %726 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %728 = stablehlo.add %727, %674 : tensor<1x1024x14x14xbf16>
    %729 = stablehlo.maximum %728, %cst_8 : tensor<1x1024x14x14xbf16>
    %730 = stablehlo.convolution(%729, %arg41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x14x14xbf16>
    %731 = stablehlo.convert %730 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %732 = stablehlo.broadcast_in_dim %731, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %733 = stablehlo.broadcast_in_dim %arg214, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %734 = stablehlo.subtract %732, %733 : tensor<1x256x14x14xf32>
    %735 = stablehlo.broadcast_in_dim %734, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %736 = stablehlo.broadcast_in_dim %arg215, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %737 = stablehlo.multiply %735, %736 : tensor<1x256x14x14xf32>
    %738 = stablehlo.convert %arg216 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %739 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %740 = stablehlo.broadcast_in_dim %738, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %741 = stablehlo.multiply %739, %740 : tensor<1x256x14x14xf32>
    %742 = stablehlo.convert %arg217 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %743 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %744 = stablehlo.broadcast_in_dim %742, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %745 = stablehlo.add %743, %744 : tensor<1x256x14x14xf32>
    %746 = stablehlo.convert %745 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %747 = stablehlo.maximum %746, %cst_7 : tensor<1x256x14x14xbf16>
    %748 = stablehlo.convolution(%747, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x14x14xbf16>
    %749 = stablehlo.convert %748 : (tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %751 = stablehlo.broadcast_in_dim %arg218, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %752 = stablehlo.subtract %750, %751 : tensor<1x256x14x14xf32>
    %753 = stablehlo.broadcast_in_dim %752, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %754 = stablehlo.broadcast_in_dim %arg219, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %755 = stablehlo.multiply %753, %754 : tensor<1x256x14x14xf32>
    %756 = stablehlo.convert %arg220 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %757 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %758 = stablehlo.broadcast_in_dim %756, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %759 = stablehlo.multiply %757, %758 : tensor<1x256x14x14xf32>
    %760 = stablehlo.convert %arg221 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %761 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2, 3] : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %762 = stablehlo.broadcast_in_dim %760, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %763 = stablehlo.add %761, %762 : tensor<1x256x14x14xf32>
    %764 = stablehlo.convert %763 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xbf16>
    %765 = stablehlo.maximum %764, %cst_7 : tensor<1x256x14x14xbf16>
    %766 = stablehlo.convolution(%765, %arg43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %767 = stablehlo.convert %766 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xf32>
    %768 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %769 = stablehlo.broadcast_in_dim %arg222, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %770 = stablehlo.subtract %768, %769 : tensor<1x1024x14x14xf32>
    %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %772 = stablehlo.broadcast_in_dim %arg223, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %773 = stablehlo.multiply %771, %772 : tensor<1x1024x14x14xf32>
    %774 = stablehlo.convert %arg224 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %775 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %776 = stablehlo.broadcast_in_dim %774, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %777 = stablehlo.multiply %775, %776 : tensor<1x1024x14x14xf32>
    %778 = stablehlo.convert %arg225 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %779 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %780 = stablehlo.broadcast_in_dim %778, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %781 = stablehlo.add %779, %780 : tensor<1x1024x14x14xf32>
    %782 = stablehlo.convert %781 : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xbf16>
    %783 = stablehlo.add %782, %729 : tensor<1x1024x14x14xbf16>
    %784 = stablehlo.maximum %783, %cst_8 : tensor<1x1024x14x14xbf16>
    %785 = stablehlo.convolution(%784, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x14x14xbf16>
    %786 = stablehlo.convert %785 : (tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xf32>
    %787 = stablehlo.broadcast_in_dim %786, dims = [0, 1, 2, 3] : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %788 = stablehlo.broadcast_in_dim %arg226, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x14x14xf32>
    %789 = stablehlo.subtract %787, %788 : tensor<1x512x14x14xf32>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2, 3] : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %791 = stablehlo.broadcast_in_dim %arg227, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x14x14xf32>
    %792 = stablehlo.multiply %790, %791 : tensor<1x512x14x14xf32>
    %793 = stablehlo.convert %arg228 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %794 = stablehlo.broadcast_in_dim %792, dims = [0, 1, 2, 3] : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %795 = stablehlo.broadcast_in_dim %793, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x14x14xf32>
    %796 = stablehlo.multiply %794, %795 : tensor<1x512x14x14xf32>
    %797 = stablehlo.convert %arg229 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %798 = stablehlo.broadcast_in_dim %796, dims = [0, 1, 2, 3] : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %799 = stablehlo.broadcast_in_dim %797, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x14x14xf32>
    %800 = stablehlo.add %798, %799 : tensor<1x512x14x14xf32>
    %801 = stablehlo.convert %800 : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xbf16>
    %802 = stablehlo.maximum %801, %cst_9 : tensor<1x512x14x14xbf16>
    %803 = stablehlo.convolution(%802, %arg45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %804 = stablehlo.convert %803 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %806 = stablehlo.broadcast_in_dim %arg230, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %807 = stablehlo.subtract %805, %806 : tensor<1x512x7x7xf32>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %809 = stablehlo.broadcast_in_dim %arg231, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %810 = stablehlo.multiply %808, %809 : tensor<1x512x7x7xf32>
    %811 = stablehlo.convert %arg232 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %812 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %813 = stablehlo.broadcast_in_dim %811, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %814 = stablehlo.multiply %812, %813 : tensor<1x512x7x7xf32>
    %815 = stablehlo.convert %arg233 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %816 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %817 = stablehlo.broadcast_in_dim %815, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %818 = stablehlo.add %816, %817 : tensor<1x512x7x7xf32>
    %819 = stablehlo.convert %818 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %820 = stablehlo.maximum %819, %cst_10 : tensor<1x512x7x7xbf16>
    %821 = stablehlo.convolution(%820, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %822 = stablehlo.convert %821 : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xf32>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %824 = stablehlo.broadcast_in_dim %arg234, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %825 = stablehlo.subtract %823, %824 : tensor<1x2048x7x7xf32>
    %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %827 = stablehlo.broadcast_in_dim %arg235, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %828 = stablehlo.multiply %826, %827 : tensor<1x2048x7x7xf32>
    %829 = stablehlo.convert %arg236 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %830 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %831 = stablehlo.broadcast_in_dim %829, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %832 = stablehlo.multiply %830, %831 : tensor<1x2048x7x7xf32>
    %833 = stablehlo.convert %arg237 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %834 = stablehlo.broadcast_in_dim %832, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %835 = stablehlo.broadcast_in_dim %833, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %836 = stablehlo.add %834, %835 : tensor<1x2048x7x7xf32>
    %837 = stablehlo.convert %836 : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xbf16>
    %838 = stablehlo.convolution(%784, %arg47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %839 = stablehlo.convert %838 : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xf32>
    %840 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %841 = stablehlo.broadcast_in_dim %arg238, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %842 = stablehlo.subtract %840, %841 : tensor<1x2048x7x7xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %844 = stablehlo.broadcast_in_dim %arg239, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %845 = stablehlo.multiply %843, %844 : tensor<1x2048x7x7xf32>
    %846 = stablehlo.convert %arg240 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %847 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %848 = stablehlo.broadcast_in_dim %846, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %849 = stablehlo.multiply %847, %848 : tensor<1x2048x7x7xf32>
    %850 = stablehlo.convert %arg241 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %851 = stablehlo.broadcast_in_dim %849, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %852 = stablehlo.broadcast_in_dim %850, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %853 = stablehlo.add %851, %852 : tensor<1x2048x7x7xf32>
    %854 = stablehlo.convert %853 : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xbf16>
    %855 = stablehlo.add %837, %854 : tensor<1x2048x7x7xbf16>
    %856 = stablehlo.maximum %855, %cst_11 : tensor<1x2048x7x7xbf16>
    %857 = stablehlo.convolution(%856, %arg48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x512x7x7xbf16>
    %858 = stablehlo.convert %857 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %859 = stablehlo.broadcast_in_dim %858, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %860 = stablehlo.broadcast_in_dim %arg242, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %861 = stablehlo.subtract %859, %860 : tensor<1x512x7x7xf32>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %863 = stablehlo.broadcast_in_dim %arg243, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %864 = stablehlo.multiply %862, %863 : tensor<1x512x7x7xf32>
    %865 = stablehlo.convert %arg244 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %866 = stablehlo.broadcast_in_dim %864, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %867 = stablehlo.broadcast_in_dim %865, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %868 = stablehlo.multiply %866, %867 : tensor<1x512x7x7xf32>
    %869 = stablehlo.convert %arg245 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %870 = stablehlo.broadcast_in_dim %868, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %871 = stablehlo.broadcast_in_dim %869, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %872 = stablehlo.add %870, %871 : tensor<1x512x7x7xf32>
    %873 = stablehlo.convert %872 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %874 = stablehlo.maximum %873, %cst_10 : tensor<1x512x7x7xbf16>
    %875 = stablehlo.convolution(%874, %arg49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %876 = stablehlo.convert %875 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %877 = stablehlo.broadcast_in_dim %876, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %878 = stablehlo.broadcast_in_dim %arg246, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %879 = stablehlo.subtract %877, %878 : tensor<1x512x7x7xf32>
    %880 = stablehlo.broadcast_in_dim %879, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %881 = stablehlo.broadcast_in_dim %arg247, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %882 = stablehlo.multiply %880, %881 : tensor<1x512x7x7xf32>
    %883 = stablehlo.convert %arg248 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %884 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %885 = stablehlo.broadcast_in_dim %883, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %886 = stablehlo.multiply %884, %885 : tensor<1x512x7x7xf32>
    %887 = stablehlo.convert %arg249 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %888 = stablehlo.broadcast_in_dim %886, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %889 = stablehlo.broadcast_in_dim %887, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %890 = stablehlo.add %888, %889 : tensor<1x512x7x7xf32>
    %891 = stablehlo.convert %890 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %892 = stablehlo.maximum %891, %cst_10 : tensor<1x512x7x7xbf16>
    %893 = stablehlo.convolution(%892, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %894 = stablehlo.convert %893 : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xf32>
    %895 = stablehlo.broadcast_in_dim %894, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %896 = stablehlo.broadcast_in_dim %arg250, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %897 = stablehlo.subtract %895, %896 : tensor<1x2048x7x7xf32>
    %898 = stablehlo.broadcast_in_dim %897, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %899 = stablehlo.broadcast_in_dim %arg251, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %900 = stablehlo.multiply %898, %899 : tensor<1x2048x7x7xf32>
    %901 = stablehlo.convert %arg252 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %902 = stablehlo.broadcast_in_dim %900, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %903 = stablehlo.broadcast_in_dim %901, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %904 = stablehlo.multiply %902, %903 : tensor<1x2048x7x7xf32>
    %905 = stablehlo.convert %arg253 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %906 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %907 = stablehlo.broadcast_in_dim %905, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %908 = stablehlo.add %906, %907 : tensor<1x2048x7x7xf32>
    %909 = stablehlo.convert %908 : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xbf16>
    %910 = stablehlo.add %909, %856 : tensor<1x2048x7x7xbf16>
    %911 = stablehlo.maximum %910, %cst_11 : tensor<1x2048x7x7xbf16>
    %912 = stablehlo.convolution(%911, %arg51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x512x7x7xbf16>
    %913 = stablehlo.convert %912 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %914 = stablehlo.broadcast_in_dim %913, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %915 = stablehlo.broadcast_in_dim %arg254, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %916 = stablehlo.subtract %914, %915 : tensor<1x512x7x7xf32>
    %917 = stablehlo.broadcast_in_dim %916, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %918 = stablehlo.broadcast_in_dim %arg255, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %919 = stablehlo.multiply %917, %918 : tensor<1x512x7x7xf32>
    %920 = stablehlo.convert %arg256 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %921 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %922 = stablehlo.broadcast_in_dim %920, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %923 = stablehlo.multiply %921, %922 : tensor<1x512x7x7xf32>
    %924 = stablehlo.convert %arg257 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %925 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %926 = stablehlo.broadcast_in_dim %924, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %927 = stablehlo.add %925, %926 : tensor<1x512x7x7xf32>
    %928 = stablehlo.convert %927 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %929 = stablehlo.maximum %928, %cst_10 : tensor<1x512x7x7xbf16>
    %930 = stablehlo.convolution(%929, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %931 = stablehlo.convert %930 : (tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xf32>
    %932 = stablehlo.broadcast_in_dim %931, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %933 = stablehlo.broadcast_in_dim %arg258, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %934 = stablehlo.subtract %932, %933 : tensor<1x512x7x7xf32>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %936 = stablehlo.broadcast_in_dim %arg259, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %937 = stablehlo.multiply %935, %936 : tensor<1x512x7x7xf32>
    %938 = stablehlo.convert %arg260 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %939 = stablehlo.broadcast_in_dim %937, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %940 = stablehlo.broadcast_in_dim %938, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %941 = stablehlo.multiply %939, %940 : tensor<1x512x7x7xf32>
    %942 = stablehlo.convert %arg261 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %943 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %944 = stablehlo.broadcast_in_dim %942, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %945 = stablehlo.add %943, %944 : tensor<1x512x7x7xf32>
    %946 = stablehlo.convert %945 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xbf16>
    %947 = stablehlo.maximum %946, %cst_10 : tensor<1x512x7x7xbf16>
    %948 = stablehlo.convolution(%947, %arg53) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %949 = stablehlo.convert %948 : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xf32>
    %950 = stablehlo.broadcast_in_dim %949, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %951 = stablehlo.broadcast_in_dim %arg262, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %952 = stablehlo.subtract %950, %951 : tensor<1x2048x7x7xf32>
    %953 = stablehlo.broadcast_in_dim %952, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %954 = stablehlo.broadcast_in_dim %arg263, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %955 = stablehlo.multiply %953, %954 : tensor<1x2048x7x7xf32>
    %956 = stablehlo.convert %arg264 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %957 = stablehlo.broadcast_in_dim %955, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %958 = stablehlo.broadcast_in_dim %956, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %959 = stablehlo.multiply %957, %958 : tensor<1x2048x7x7xf32>
    %960 = stablehlo.convert %arg265 : (tensor<2048x1x1xbf16>) -> tensor<2048x1x1xf32>
    %961 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2, 3] : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %962 = stablehlo.broadcast_in_dim %960, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %963 = stablehlo.add %961, %962 : tensor<1x2048x7x7xf32>
    %964 = stablehlo.convert %963 : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xbf16>
    %965 = stablehlo.add %964, %911 : tensor<1x2048x7x7xbf16>
    %966 = stablehlo.maximum %965, %cst_11 : tensor<1x2048x7x7xbf16>
    %967 = stablehlo.reduce(%966 init: %cst_12) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x2048x7x7xbf16>, tensor<bf16>) -> tensor<1x2048xbf16>
    %968 = stablehlo.reshape %967 : (tensor<1x2048xbf16>) -> tensor<1x2048x1x1xbf16>
    %969 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xbf16>
    %970 = stablehlo.reshape %969 : (tensor<1xbf16>) -> tensor<bf16>
    %971 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %972 = stablehlo.broadcast_in_dim %970, dims = [] : (tensor<bf16>) -> tensor<1x2048x1x1xbf16>
    %973 = stablehlo.divide %971, %972 : tensor<1x2048x1x1xbf16>
    %974 = stablehlo.reshape %973 : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048xbf16>
    %975 = stablehlo.convert %974 : (tensor<1x2048xbf16>) -> tensor<1x2048xf32>
    %976 = stablehlo.dot_general %975, %arg266, contracting_dims = [1] x [0] : (tensor<1x2048xf32>, tensor<2048x1000xf32>) -> tensor<1x1000xf32>
    %977 = stablehlo.convert %cst_14 : (tensor<1xi64>) -> tensor<1xf32>
    %978 = stablehlo.reshape %977 : (tensor<1xf32>) -> tensor<f32>
    %979 = stablehlo.broadcast_in_dim %976, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %980 = stablehlo.broadcast_in_dim %978, dims = [] : (tensor<f32>) -> tensor<1x1000xf32>
    %981 = stablehlo.multiply %979, %980 : tensor<1x1000xf32>
    %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %983 = stablehlo.broadcast_in_dim %arg267, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %984 = stablehlo.add %982, %983 : tensor<1x1000xf32>
    %985 = stablehlo.convert %984 : (tensor<1x1000xf32>) -> tensor<1x1000xbf16>
    return %985 : tensor<1x1000xbf16>
  }
}
