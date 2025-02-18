module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<32x3x3x3xbf16>, %arg2: tensor<32x1x3x3xbf16>, %arg3: tensor<16x32x1x1xbf16>, %arg4: tensor<96x16x1x1xbf16>, %arg5: tensor<96x1x3x3xbf16>, %arg6: tensor<24x96x1x1xbf16>, %arg7: tensor<144x24x1x1xbf16>, %arg8: tensor<144x1x3x3xbf16>, %arg9: tensor<24x144x1x1xbf16>, %arg10: tensor<144x24x1x1xbf16>, %arg11: tensor<144x1x3x3xbf16>, %arg12: tensor<32x144x1x1xbf16>, %arg13: tensor<192x32x1x1xbf16>, %arg14: tensor<192x1x3x3xbf16>, %arg15: tensor<32x192x1x1xbf16>, %arg16: tensor<192x32x1x1xbf16>, %arg17: tensor<192x1x3x3xbf16>, %arg18: tensor<32x192x1x1xbf16>, %arg19: tensor<192x32x1x1xbf16>, %arg20: tensor<192x1x3x3xbf16>, %arg21: tensor<64x192x1x1xbf16>, %arg22: tensor<384x64x1x1xbf16>, %arg23: tensor<384x1x3x3xbf16>, %arg24: tensor<64x384x1x1xbf16>, %arg25: tensor<384x64x1x1xbf16>, %arg26: tensor<384x1x3x3xbf16>, %arg27: tensor<64x384x1x1xbf16>, %arg28: tensor<384x64x1x1xbf16>, %arg29: tensor<384x1x3x3xbf16>, %arg30: tensor<64x384x1x1xbf16>, %arg31: tensor<384x64x1x1xbf16>, %arg32: tensor<384x1x3x3xbf16>, %arg33: tensor<96x384x1x1xbf16>, %arg34: tensor<576x96x1x1xbf16>, %arg35: tensor<576x1x3x3xbf16>, %arg36: tensor<96x576x1x1xbf16>, %arg37: tensor<576x96x1x1xbf16>, %arg38: tensor<576x1x3x3xbf16>, %arg39: tensor<96x576x1x1xbf16>, %arg40: tensor<576x96x1x1xbf16>, %arg41: tensor<576x1x3x3xbf16>, %arg42: tensor<160x576x1x1xbf16>, %arg43: tensor<960x160x1x1xbf16>, %arg44: tensor<960x1x3x3xbf16>, %arg45: tensor<160x960x1x1xbf16>, %arg46: tensor<960x160x1x1xbf16>, %arg47: tensor<960x1x3x3xbf16>, %arg48: tensor<160x960x1x1xbf16>, %arg49: tensor<960x160x1x1xbf16>, %arg50: tensor<960x1x3x3xbf16>, %arg51: tensor<320x960x1x1xbf16>, %arg52: tensor<1280x320x1x1xbf16>, %arg53: tensor<32x1x1xf32>, %arg54: tensor<32x1x1xf32>, %arg55: tensor<32x1x1xbf16>, %arg56: tensor<32x1x1xbf16>, %arg57: tensor<32x1x1xf32>, %arg58: tensor<32x1x1xf32>, %arg59: tensor<32x1x1xbf16>, %arg60: tensor<32x1x1xbf16>, %arg61: tensor<16x1x1xf32>, %arg62: tensor<16x1x1xf32>, %arg63: tensor<16x1x1xbf16>, %arg64: tensor<16x1x1xbf16>, %arg65: tensor<96x1x1xf32>, %arg66: tensor<96x1x1xf32>, %arg67: tensor<96x1x1xbf16>, %arg68: tensor<96x1x1xbf16>, %arg69: tensor<96x1x1xf32>, %arg70: tensor<96x1x1xf32>, %arg71: tensor<96x1x1xbf16>, %arg72: tensor<96x1x1xbf16>, %arg73: tensor<24x1x1xf32>, %arg74: tensor<24x1x1xf32>, %arg75: tensor<24x1x1xbf16>, %arg76: tensor<24x1x1xbf16>, %arg77: tensor<144x1x1xf32>, %arg78: tensor<144x1x1xf32>, %arg79: tensor<144x1x1xbf16>, %arg80: tensor<144x1x1xbf16>, %arg81: tensor<144x1x1xf32>, %arg82: tensor<144x1x1xf32>, %arg83: tensor<144x1x1xbf16>, %arg84: tensor<144x1x1xbf16>, %arg85: tensor<24x1x1xf32>, %arg86: tensor<24x1x1xf32>, %arg87: tensor<24x1x1xbf16>, %arg88: tensor<24x1x1xbf16>, %arg89: tensor<144x1x1xf32>, %arg90: tensor<144x1x1xf32>, %arg91: tensor<144x1x1xbf16>, %arg92: tensor<144x1x1xbf16>, %arg93: tensor<144x1x1xf32>, %arg94: tensor<144x1x1xf32>, %arg95: tensor<144x1x1xbf16>, %arg96: tensor<144x1x1xbf16>, %arg97: tensor<32x1x1xf32>, %arg98: tensor<32x1x1xf32>, %arg99: tensor<32x1x1xbf16>, %arg100: tensor<32x1x1xbf16>, %arg101: tensor<192x1x1xf32>, %arg102: tensor<192x1x1xf32>, %arg103: tensor<192x1x1xbf16>, %arg104: tensor<192x1x1xbf16>, %arg105: tensor<192x1x1xf32>, %arg106: tensor<192x1x1xf32>, %arg107: tensor<192x1x1xbf16>, %arg108: tensor<192x1x1xbf16>, %arg109: tensor<32x1x1xf32>, %arg110: tensor<32x1x1xf32>, %arg111: tensor<32x1x1xbf16>, %arg112: tensor<32x1x1xbf16>, %arg113: tensor<192x1x1xf32>, %arg114: tensor<192x1x1xf32>, %arg115: tensor<192x1x1xbf16>, %arg116: tensor<192x1x1xbf16>, %arg117: tensor<192x1x1xf32>, %arg118: tensor<192x1x1xf32>, %arg119: tensor<192x1x1xbf16>, %arg120: tensor<192x1x1xbf16>, %arg121: tensor<32x1x1xf32>, %arg122: tensor<32x1x1xf32>, %arg123: tensor<32x1x1xbf16>, %arg124: tensor<32x1x1xbf16>, %arg125: tensor<192x1x1xf32>, %arg126: tensor<192x1x1xf32>, %arg127: tensor<192x1x1xbf16>, %arg128: tensor<192x1x1xbf16>, %arg129: tensor<192x1x1xf32>, %arg130: tensor<192x1x1xf32>, %arg131: tensor<192x1x1xbf16>, %arg132: tensor<192x1x1xbf16>, %arg133: tensor<64x1x1xf32>, %arg134: tensor<64x1x1xf32>, %arg135: tensor<64x1x1xbf16>, %arg136: tensor<64x1x1xbf16>, %arg137: tensor<384x1x1xf32>, %arg138: tensor<384x1x1xf32>, %arg139: tensor<384x1x1xbf16>, %arg140: tensor<384x1x1xbf16>, %arg141: tensor<384x1x1xf32>, %arg142: tensor<384x1x1xf32>, %arg143: tensor<384x1x1xbf16>, %arg144: tensor<384x1x1xbf16>, %arg145: tensor<64x1x1xf32>, %arg146: tensor<64x1x1xf32>, %arg147: tensor<64x1x1xbf16>, %arg148: tensor<64x1x1xbf16>, %arg149: tensor<384x1x1xf32>, %arg150: tensor<384x1x1xf32>, %arg151: tensor<384x1x1xbf16>, %arg152: tensor<384x1x1xbf16>, %arg153: tensor<384x1x1xf32>, %arg154: tensor<384x1x1xf32>, %arg155: tensor<384x1x1xbf16>, %arg156: tensor<384x1x1xbf16>, %arg157: tensor<64x1x1xf32>, %arg158: tensor<64x1x1xf32>, %arg159: tensor<64x1x1xbf16>, %arg160: tensor<64x1x1xbf16>, %arg161: tensor<384x1x1xf32>, %arg162: tensor<384x1x1xf32>, %arg163: tensor<384x1x1xbf16>, %arg164: tensor<384x1x1xbf16>, %arg165: tensor<384x1x1xf32>, %arg166: tensor<384x1x1xf32>, %arg167: tensor<384x1x1xbf16>, %arg168: tensor<384x1x1xbf16>, %arg169: tensor<64x1x1xf32>, %arg170: tensor<64x1x1xf32>, %arg171: tensor<64x1x1xbf16>, %arg172: tensor<64x1x1xbf16>, %arg173: tensor<384x1x1xf32>, %arg174: tensor<384x1x1xf32>, %arg175: tensor<384x1x1xbf16>, %arg176: tensor<384x1x1xbf16>, %arg177: tensor<384x1x1xf32>, %arg178: tensor<384x1x1xf32>, %arg179: tensor<384x1x1xbf16>, %arg180: tensor<384x1x1xbf16>, %arg181: tensor<96x1x1xf32>, %arg182: tensor<96x1x1xf32>, %arg183: tensor<96x1x1xbf16>, %arg184: tensor<96x1x1xbf16>, %arg185: tensor<576x1x1xf32>, %arg186: tensor<576x1x1xf32>, %arg187: tensor<576x1x1xbf16>, %arg188: tensor<576x1x1xbf16>, %arg189: tensor<576x1x1xf32>, %arg190: tensor<576x1x1xf32>, %arg191: tensor<576x1x1xbf16>, %arg192: tensor<576x1x1xbf16>, %arg193: tensor<96x1x1xf32>, %arg194: tensor<96x1x1xf32>, %arg195: tensor<96x1x1xbf16>, %arg196: tensor<96x1x1xbf16>, %arg197: tensor<576x1x1xf32>, %arg198: tensor<576x1x1xf32>, %arg199: tensor<576x1x1xbf16>, %arg200: tensor<576x1x1xbf16>, %arg201: tensor<576x1x1xf32>, %arg202: tensor<576x1x1xf32>, %arg203: tensor<576x1x1xbf16>, %arg204: tensor<576x1x1xbf16>, %arg205: tensor<96x1x1xf32>, %arg206: tensor<96x1x1xf32>, %arg207: tensor<96x1x1xbf16>, %arg208: tensor<96x1x1xbf16>, %arg209: tensor<576x1x1xf32>, %arg210: tensor<576x1x1xf32>, %arg211: tensor<576x1x1xbf16>, %arg212: tensor<576x1x1xbf16>, %arg213: tensor<576x1x1xf32>, %arg214: tensor<576x1x1xf32>, %arg215: tensor<576x1x1xbf16>, %arg216: tensor<576x1x1xbf16>, %arg217: tensor<160x1x1xf32>, %arg218: tensor<160x1x1xf32>, %arg219: tensor<160x1x1xbf16>, %arg220: tensor<160x1x1xbf16>, %arg221: tensor<960x1x1xf32>, %arg222: tensor<960x1x1xf32>, %arg223: tensor<960x1x1xbf16>, %arg224: tensor<960x1x1xbf16>, %arg225: tensor<960x1x1xf32>, %arg226: tensor<960x1x1xf32>, %arg227: tensor<960x1x1xbf16>, %arg228: tensor<960x1x1xbf16>, %arg229: tensor<160x1x1xf32>, %arg230: tensor<160x1x1xf32>, %arg231: tensor<160x1x1xbf16>, %arg232: tensor<160x1x1xbf16>, %arg233: tensor<960x1x1xf32>, %arg234: tensor<960x1x1xf32>, %arg235: tensor<960x1x1xbf16>, %arg236: tensor<960x1x1xbf16>, %arg237: tensor<960x1x1xf32>, %arg238: tensor<960x1x1xf32>, %arg239: tensor<960x1x1xbf16>, %arg240: tensor<960x1x1xbf16>, %arg241: tensor<160x1x1xf32>, %arg242: tensor<160x1x1xf32>, %arg243: tensor<160x1x1xbf16>, %arg244: tensor<160x1x1xbf16>, %arg245: tensor<960x1x1xf32>, %arg246: tensor<960x1x1xf32>, %arg247: tensor<960x1x1xbf16>, %arg248: tensor<960x1x1xbf16>, %arg249: tensor<960x1x1xf32>, %arg250: tensor<960x1x1xf32>, %arg251: tensor<960x1x1xbf16>, %arg252: tensor<960x1x1xbf16>, %arg253: tensor<320x1x1xf32>, %arg254: tensor<320x1x1xf32>, %arg255: tensor<320x1x1xbf16>, %arg256: tensor<320x1x1xbf16>, %arg257: tensor<1280x1x1xf32>, %arg258: tensor<1280x1x1xf32>, %arg259: tensor<1280x1x1xbf16>, %arg260: tensor<1280x1x1xbf16>, %arg261: tensor<1280x1000xf32>, %arg262: tensor<1000xf32>) -> tensor<1x1000xbf16> {
    %cst = stablehlo.constant dense<6.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_2 = arith.constant dense<49> : tensor<1xi64>
    %cst_3 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x32x112x112xbf16>
    %1 = stablehlo.convert %0 : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %3 = stablehlo.broadcast_in_dim %arg53, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %4 = stablehlo.subtract %2, %3 : tensor<1x32x112x112xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg54, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x32x112x112xf32>
    %8 = stablehlo.convert %arg55 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x32x112x112xf32>
    %12 = stablehlo.convert %arg56 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %15 = stablehlo.add %13, %14 : tensor<1x32x112x112xf32>
    %16 = stablehlo.convert %15 : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
    %17 = stablehlo.convert %cst_0 : (tensor<f64>) -> tensor<bf16>
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
    %19 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x32x112x112xbf16>
    %20 = stablehlo.maximum %18, %19 : tensor<1x32x112x112xbf16>
    %21 = stablehlo.convert %cst : (tensor<f64>) -> tensor<bf16>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x32x112x112xbf16>
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
    %24 = stablehlo.minimum %22, %23 : tensor<1x32x112x112xbf16>
    %25 = stablehlo.convolution(%24, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<1x32x112x112xbf16>, tensor<32x1x3x3xbf16>) -> tensor<1x32x112x112xbf16>
    %26 = stablehlo.convert %25 : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %28 = stablehlo.broadcast_in_dim %arg57, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<1x32x112x112xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %31 = stablehlo.broadcast_in_dim %arg58, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<1x32x112x112xf32>
    %33 = stablehlo.convert %arg59 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %35 = stablehlo.broadcast_in_dim %33, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %36 = stablehlo.multiply %34, %35 : tensor<1x32x112x112xf32>
    %37 = stablehlo.convert %arg60 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %38 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x112x112xf32>
    %40 = stablehlo.add %38, %39 : tensor<1x32x112x112xf32>
    %41 = stablehlo.convert %40 : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
    %43 = stablehlo.maximum %42, %19 : tensor<1x32x112x112xbf16>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
    %45 = stablehlo.minimum %22, %44 : tensor<1x32x112x112xbf16>
    %46 = stablehlo.convolution(%45, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x112x112xbf16>, tensor<16x32x1x1xbf16>) -> tensor<1x16x112x112xbf16>
    %47 = stablehlo.convert %46 : (tensor<1x16x112x112xbf16>) -> tensor<1x16x112x112xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %49 = stablehlo.broadcast_in_dim %arg61, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %50 = stablehlo.subtract %48, %49 : tensor<1x16x112x112xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %52 = stablehlo.broadcast_in_dim %arg62, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %53 = stablehlo.multiply %51, %52 : tensor<1x16x112x112xf32>
    %54 = stablehlo.convert %arg63 : (tensor<16x1x1xbf16>) -> tensor<16x1x1xf32>
    %55 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %56 = stablehlo.broadcast_in_dim %54, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %57 = stablehlo.multiply %55, %56 : tensor<1x16x112x112xf32>
    %58 = stablehlo.convert %arg64 : (tensor<16x1x1xbf16>) -> tensor<16x1x1xf32>
    %59 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %60 = stablehlo.broadcast_in_dim %58, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %61 = stablehlo.add %59, %60 : tensor<1x16x112x112xf32>
    %62 = stablehlo.convert %61 : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xbf16>
    %63 = stablehlo.convolution(%62, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x112x112xbf16>, tensor<96x16x1x1xbf16>) -> tensor<1x96x112x112xbf16>
    %64 = stablehlo.convert %63 : (tensor<1x96x112x112xbf16>) -> tensor<1x96x112x112xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %66 = stablehlo.broadcast_in_dim %arg65, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x112x112xf32>
    %67 = stablehlo.subtract %65, %66 : tensor<1x96x112x112xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %69 = stablehlo.broadcast_in_dim %arg66, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x112x112xf32>
    %70 = stablehlo.multiply %68, %69 : tensor<1x96x112x112xf32>
    %71 = stablehlo.convert %arg67 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %72 = stablehlo.broadcast_in_dim %70, dims = [0, 1, 2, 3] : (tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %73 = stablehlo.broadcast_in_dim %71, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x112x112xf32>
    %74 = stablehlo.multiply %72, %73 : tensor<1x96x112x112xf32>
    %75 = stablehlo.convert %arg68 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %76 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %77 = stablehlo.broadcast_in_dim %75, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x112x112xf32>
    %78 = stablehlo.add %76, %77 : tensor<1x96x112x112xf32>
    %79 = stablehlo.convert %78 : (tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xbf16>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<1x96x112x112xbf16>) -> tensor<1x96x112x112xbf16>
    %81 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x96x112x112xbf16>
    %82 = stablehlo.maximum %80, %81 : tensor<1x96x112x112xbf16>
    %83 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x96x112x112xbf16>
    %84 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x96x112x112xbf16>) -> tensor<1x96x112x112xbf16>
    %85 = stablehlo.minimum %83, %84 : tensor<1x96x112x112xbf16>
    %86 = stablehlo.convolution(%85, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<1x96x112x112xbf16>, tensor<96x1x3x3xbf16>) -> tensor<1x96x56x56xbf16>
    %87 = stablehlo.convert %86 : (tensor<1x96x56x56xbf16>) -> tensor<1x96x56x56xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2, 3] : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %89 = stablehlo.broadcast_in_dim %arg69, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x56x56xf32>
    %90 = stablehlo.subtract %88, %89 : tensor<1x96x56x56xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %92 = stablehlo.broadcast_in_dim %arg70, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x56x56xf32>
    %93 = stablehlo.multiply %91, %92 : tensor<1x96x56x56xf32>
    %94 = stablehlo.convert %arg71 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %95 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %96 = stablehlo.broadcast_in_dim %94, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x56x56xf32>
    %97 = stablehlo.multiply %95, %96 : tensor<1x96x56x56xf32>
    %98 = stablehlo.convert %arg72 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %99 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %100 = stablehlo.broadcast_in_dim %98, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x56x56xf32>
    %101 = stablehlo.add %99, %100 : tensor<1x96x56x56xf32>
    %102 = stablehlo.convert %101 : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xbf16>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x96x56x56xbf16>) -> tensor<1x96x56x56xbf16>
    %104 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x96x56x56xbf16>
    %105 = stablehlo.maximum %103, %104 : tensor<1x96x56x56xbf16>
    %106 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x96x56x56xbf16>
    %107 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2, 3] : (tensor<1x96x56x56xbf16>) -> tensor<1x96x56x56xbf16>
    %108 = stablehlo.minimum %106, %107 : tensor<1x96x56x56xbf16>
    %109 = stablehlo.convolution(%108, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x96x56x56xbf16>, tensor<24x96x1x1xbf16>) -> tensor<1x24x56x56xbf16>
    %110 = stablehlo.convert %109 : (tensor<1x24x56x56xbf16>) -> tensor<1x24x56x56xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %112 = stablehlo.broadcast_in_dim %arg73, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %113 = stablehlo.subtract %111, %112 : tensor<1x24x56x56xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %115 = stablehlo.broadcast_in_dim %arg74, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %116 = stablehlo.multiply %114, %115 : tensor<1x24x56x56xf32>
    %117 = stablehlo.convert %arg75 : (tensor<24x1x1xbf16>) -> tensor<24x1x1xf32>
    %118 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %119 = stablehlo.broadcast_in_dim %117, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %120 = stablehlo.multiply %118, %119 : tensor<1x24x56x56xf32>
    %121 = stablehlo.convert %arg76 : (tensor<24x1x1xbf16>) -> tensor<24x1x1xf32>
    %122 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %123 = stablehlo.broadcast_in_dim %121, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %124 = stablehlo.add %122, %123 : tensor<1x24x56x56xf32>
    %125 = stablehlo.convert %124 : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xbf16>
    %126 = stablehlo.convolution(%125, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x24x56x56xbf16>, tensor<144x24x1x1xbf16>) -> tensor<1x144x56x56xbf16>
    %127 = stablehlo.convert %126 : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %129 = stablehlo.broadcast_in_dim %arg77, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %130 = stablehlo.subtract %128, %129 : tensor<1x144x56x56xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %132 = stablehlo.broadcast_in_dim %arg78, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %133 = stablehlo.multiply %131, %132 : tensor<1x144x56x56xf32>
    %134 = stablehlo.convert %arg79 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %135 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %136 = stablehlo.broadcast_in_dim %134, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %137 = stablehlo.multiply %135, %136 : tensor<1x144x56x56xf32>
    %138 = stablehlo.convert %arg80 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %139 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %140 = stablehlo.broadcast_in_dim %138, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %141 = stablehlo.add %139, %140 : tensor<1x144x56x56xf32>
    %142 = stablehlo.convert %141 : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xbf16>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    %144 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x144x56x56xbf16>
    %145 = stablehlo.maximum %143, %144 : tensor<1x144x56x56xbf16>
    %146 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x144x56x56xbf16>
    %147 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    %148 = stablehlo.minimum %146, %147 : tensor<1x144x56x56xbf16>
    %149 = stablehlo.convolution(%148, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<1x144x56x56xbf16>, tensor<144x1x3x3xbf16>) -> tensor<1x144x56x56xbf16>
    %150 = stablehlo.convert %149 : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %152 = stablehlo.broadcast_in_dim %arg81, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %153 = stablehlo.subtract %151, %152 : tensor<1x144x56x56xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %155 = stablehlo.broadcast_in_dim %arg82, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %156 = stablehlo.multiply %154, %155 : tensor<1x144x56x56xf32>
    %157 = stablehlo.convert %arg83 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %158 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %159 = stablehlo.broadcast_in_dim %157, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %160 = stablehlo.multiply %158, %159 : tensor<1x144x56x56xf32>
    %161 = stablehlo.convert %arg84 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %162 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %163 = stablehlo.broadcast_in_dim %161, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %164 = stablehlo.add %162, %163 : tensor<1x144x56x56xf32>
    %165 = stablehlo.convert %164 : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xbf16>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    %167 = stablehlo.maximum %166, %144 : tensor<1x144x56x56xbf16>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    %169 = stablehlo.minimum %146, %168 : tensor<1x144x56x56xbf16>
    %170 = stablehlo.convolution(%169, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x144x56x56xbf16>, tensor<24x144x1x1xbf16>) -> tensor<1x24x56x56xbf16>
    %171 = stablehlo.convert %170 : (tensor<1x24x56x56xbf16>) -> tensor<1x24x56x56xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %173 = stablehlo.broadcast_in_dim %arg85, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %174 = stablehlo.subtract %172, %173 : tensor<1x24x56x56xf32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %176 = stablehlo.broadcast_in_dim %arg86, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %177 = stablehlo.multiply %175, %176 : tensor<1x24x56x56xf32>
    %178 = stablehlo.convert %arg87 : (tensor<24x1x1xbf16>) -> tensor<24x1x1xf32>
    %179 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %180 = stablehlo.broadcast_in_dim %178, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %181 = stablehlo.multiply %179, %180 : tensor<1x24x56x56xf32>
    %182 = stablehlo.convert %arg88 : (tensor<24x1x1xbf16>) -> tensor<24x1x1xf32>
    %183 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2, 3] : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %184 = stablehlo.broadcast_in_dim %182, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x56x56xf32>
    %185 = stablehlo.add %183, %184 : tensor<1x24x56x56xf32>
    %186 = stablehlo.convert %185 : (tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xbf16>
    %187 = stablehlo.add %125, %186 : tensor<1x24x56x56xbf16>
    %188 = stablehlo.convolution(%187, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x24x56x56xbf16>, tensor<144x24x1x1xbf16>) -> tensor<1x144x56x56xbf16>
    %189 = stablehlo.convert %188 : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xf32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %191 = stablehlo.broadcast_in_dim %arg89, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %192 = stablehlo.subtract %190, %191 : tensor<1x144x56x56xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %194 = stablehlo.broadcast_in_dim %arg90, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %195 = stablehlo.multiply %193, %194 : tensor<1x144x56x56xf32>
    %196 = stablehlo.convert %arg91 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %197 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %198 = stablehlo.broadcast_in_dim %196, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %199 = stablehlo.multiply %197, %198 : tensor<1x144x56x56xf32>
    %200 = stablehlo.convert %arg92 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %201 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %202 = stablehlo.broadcast_in_dim %200, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x56x56xf32>
    %203 = stablehlo.add %201, %202 : tensor<1x144x56x56xf32>
    %204 = stablehlo.convert %203 : (tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xbf16>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    %206 = stablehlo.maximum %205, %144 : tensor<1x144x56x56xbf16>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    %208 = stablehlo.minimum %146, %207 : tensor<1x144x56x56xbf16>
    %209 = stablehlo.convolution(%208, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<1x144x56x56xbf16>, tensor<144x1x3x3xbf16>) -> tensor<1x144x28x28xbf16>
    %210 = stablehlo.convert %209 : (tensor<1x144x28x28xbf16>) -> tensor<1x144x28x28xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2, 3] : (tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %212 = stablehlo.broadcast_in_dim %arg93, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x28x28xf32>
    %213 = stablehlo.subtract %211, %212 : tensor<1x144x28x28xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %215 = stablehlo.broadcast_in_dim %arg94, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x28x28xf32>
    %216 = stablehlo.multiply %214, %215 : tensor<1x144x28x28xf32>
    %217 = stablehlo.convert %arg95 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %218 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2, 3] : (tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %219 = stablehlo.broadcast_in_dim %217, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x28x28xf32>
    %220 = stablehlo.multiply %218, %219 : tensor<1x144x28x28xf32>
    %221 = stablehlo.convert %arg96 : (tensor<144x1x1xbf16>) -> tensor<144x1x1xf32>
    %222 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2, 3] : (tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %223 = stablehlo.broadcast_in_dim %221, dims = [1, 2, 3] : (tensor<144x1x1xf32>) -> tensor<1x144x28x28xf32>
    %224 = stablehlo.add %222, %223 : tensor<1x144x28x28xf32>
    %225 = stablehlo.convert %224 : (tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xbf16>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2, 3] : (tensor<1x144x28x28xbf16>) -> tensor<1x144x28x28xbf16>
    %227 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x144x28x28xbf16>
    %228 = stablehlo.maximum %226, %227 : tensor<1x144x28x28xbf16>
    %229 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x144x28x28xbf16>
    %230 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2, 3] : (tensor<1x144x28x28xbf16>) -> tensor<1x144x28x28xbf16>
    %231 = stablehlo.minimum %229, %230 : tensor<1x144x28x28xbf16>
    %232 = stablehlo.convolution(%231, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x144x28x28xbf16>, tensor<32x144x1x1xbf16>) -> tensor<1x32x28x28xbf16>
    %233 = stablehlo.convert %232 : (tensor<1x32x28x28xbf16>) -> tensor<1x32x28x28xf32>
    %234 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %235 = stablehlo.broadcast_in_dim %arg97, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %236 = stablehlo.subtract %234, %235 : tensor<1x32x28x28xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %238 = stablehlo.broadcast_in_dim %arg98, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %239 = stablehlo.multiply %237, %238 : tensor<1x32x28x28xf32>
    %240 = stablehlo.convert %arg99 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %241 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %242 = stablehlo.broadcast_in_dim %240, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %243 = stablehlo.multiply %241, %242 : tensor<1x32x28x28xf32>
    %244 = stablehlo.convert %arg100 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %245 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %246 = stablehlo.broadcast_in_dim %244, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %247 = stablehlo.add %245, %246 : tensor<1x32x28x28xf32>
    %248 = stablehlo.convert %247 : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xbf16>
    %249 = stablehlo.convolution(%248, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x28x28xbf16>, tensor<192x32x1x1xbf16>) -> tensor<1x192x28x28xbf16>
    %250 = stablehlo.convert %249 : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %252 = stablehlo.broadcast_in_dim %arg101, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %253 = stablehlo.subtract %251, %252 : tensor<1x192x28x28xf32>
    %254 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %255 = stablehlo.broadcast_in_dim %arg102, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %256 = stablehlo.multiply %254, %255 : tensor<1x192x28x28xf32>
    %257 = stablehlo.convert %arg103 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %258 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %259 = stablehlo.broadcast_in_dim %257, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %260 = stablehlo.multiply %258, %259 : tensor<1x192x28x28xf32>
    %261 = stablehlo.convert %arg104 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %262 = stablehlo.broadcast_in_dim %260, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %263 = stablehlo.broadcast_in_dim %261, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %264 = stablehlo.add %262, %263 : tensor<1x192x28x28xf32>
    %265 = stablehlo.convert %264 : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xbf16>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %267 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x192x28x28xbf16>
    %268 = stablehlo.maximum %266, %267 : tensor<1x192x28x28xbf16>
    %269 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x192x28x28xbf16>
    %270 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %271 = stablehlo.minimum %269, %270 : tensor<1x192x28x28xbf16>
    %272 = stablehlo.convolution(%271, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x192x28x28xbf16>, tensor<192x1x3x3xbf16>) -> tensor<1x192x28x28xbf16>
    %273 = stablehlo.convert %272 : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %275 = stablehlo.broadcast_in_dim %arg105, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %276 = stablehlo.subtract %274, %275 : tensor<1x192x28x28xf32>
    %277 = stablehlo.broadcast_in_dim %276, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %278 = stablehlo.broadcast_in_dim %arg106, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %279 = stablehlo.multiply %277, %278 : tensor<1x192x28x28xf32>
    %280 = stablehlo.convert %arg107 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %281 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %282 = stablehlo.broadcast_in_dim %280, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %283 = stablehlo.multiply %281, %282 : tensor<1x192x28x28xf32>
    %284 = stablehlo.convert %arg108 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %285 = stablehlo.broadcast_in_dim %283, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %286 = stablehlo.broadcast_in_dim %284, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %287 = stablehlo.add %285, %286 : tensor<1x192x28x28xf32>
    %288 = stablehlo.convert %287 : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xbf16>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %290 = stablehlo.maximum %289, %267 : tensor<1x192x28x28xbf16>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %292 = stablehlo.minimum %269, %291 : tensor<1x192x28x28xbf16>
    %293 = stablehlo.convolution(%292, %arg15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x192x28x28xbf16>, tensor<32x192x1x1xbf16>) -> tensor<1x32x28x28xbf16>
    %294 = stablehlo.convert %293 : (tensor<1x32x28x28xbf16>) -> tensor<1x32x28x28xf32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %296 = stablehlo.broadcast_in_dim %arg109, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %297 = stablehlo.subtract %295, %296 : tensor<1x32x28x28xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %299 = stablehlo.broadcast_in_dim %arg110, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %300 = stablehlo.multiply %298, %299 : tensor<1x32x28x28xf32>
    %301 = stablehlo.convert %arg111 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %302 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %303 = stablehlo.broadcast_in_dim %301, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x32x28x28xf32>
    %305 = stablehlo.convert %arg112 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %306 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %307 = stablehlo.broadcast_in_dim %305, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %308 = stablehlo.add %306, %307 : tensor<1x32x28x28xf32>
    %309 = stablehlo.convert %308 : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xbf16>
    %310 = stablehlo.add %248, %309 : tensor<1x32x28x28xbf16>
    %311 = stablehlo.convolution(%310, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x28x28xbf16>, tensor<192x32x1x1xbf16>) -> tensor<1x192x28x28xbf16>
    %312 = stablehlo.convert %311 : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xf32>
    %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %314 = stablehlo.broadcast_in_dim %arg113, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %315 = stablehlo.subtract %313, %314 : tensor<1x192x28x28xf32>
    %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %317 = stablehlo.broadcast_in_dim %arg114, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %318 = stablehlo.multiply %316, %317 : tensor<1x192x28x28xf32>
    %319 = stablehlo.convert %arg115 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %320 = stablehlo.broadcast_in_dim %318, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %321 = stablehlo.broadcast_in_dim %319, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %322 = stablehlo.multiply %320, %321 : tensor<1x192x28x28xf32>
    %323 = stablehlo.convert %arg116 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %324 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %325 = stablehlo.broadcast_in_dim %323, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %326 = stablehlo.add %324, %325 : tensor<1x192x28x28xf32>
    %327 = stablehlo.convert %326 : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xbf16>
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %329 = stablehlo.maximum %328, %267 : tensor<1x192x28x28xbf16>
    %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %331 = stablehlo.minimum %269, %330 : tensor<1x192x28x28xbf16>
    %332 = stablehlo.convolution(%331, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x192x28x28xbf16>, tensor<192x1x3x3xbf16>) -> tensor<1x192x28x28xbf16>
    %333 = stablehlo.convert %332 : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %335 = stablehlo.broadcast_in_dim %arg117, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %336 = stablehlo.subtract %334, %335 : tensor<1x192x28x28xf32>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %338 = stablehlo.broadcast_in_dim %arg118, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %339 = stablehlo.multiply %337, %338 : tensor<1x192x28x28xf32>
    %340 = stablehlo.convert %arg119 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %341 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %342 = stablehlo.broadcast_in_dim %340, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %343 = stablehlo.multiply %341, %342 : tensor<1x192x28x28xf32>
    %344 = stablehlo.convert %arg120 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %345 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %346 = stablehlo.broadcast_in_dim %344, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %347 = stablehlo.add %345, %346 : tensor<1x192x28x28xf32>
    %348 = stablehlo.convert %347 : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xbf16>
    %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %350 = stablehlo.maximum %349, %267 : tensor<1x192x28x28xbf16>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %352 = stablehlo.minimum %269, %351 : tensor<1x192x28x28xbf16>
    %353 = stablehlo.convolution(%352, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x192x28x28xbf16>, tensor<32x192x1x1xbf16>) -> tensor<1x32x28x28xbf16>
    %354 = stablehlo.convert %353 : (tensor<1x32x28x28xbf16>) -> tensor<1x32x28x28xf32>
    %355 = stablehlo.broadcast_in_dim %354, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %356 = stablehlo.broadcast_in_dim %arg121, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %357 = stablehlo.subtract %355, %356 : tensor<1x32x28x28xf32>
    %358 = stablehlo.broadcast_in_dim %357, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %359 = stablehlo.broadcast_in_dim %arg122, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<1x32x28x28xf32>
    %361 = stablehlo.convert %arg123 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %362 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %363 = stablehlo.broadcast_in_dim %361, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %364 = stablehlo.multiply %362, %363 : tensor<1x32x28x28xf32>
    %365 = stablehlo.convert %arg124 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %366 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %367 = stablehlo.broadcast_in_dim %365, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x28x28xf32>
    %368 = stablehlo.add %366, %367 : tensor<1x32x28x28xf32>
    %369 = stablehlo.convert %368 : (tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xbf16>
    %370 = stablehlo.add %310, %369 : tensor<1x32x28x28xbf16>
    %371 = stablehlo.convolution(%370, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x28x28xbf16>, tensor<192x32x1x1xbf16>) -> tensor<1x192x28x28xbf16>
    %372 = stablehlo.convert %371 : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xf32>
    %373 = stablehlo.broadcast_in_dim %372, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %374 = stablehlo.broadcast_in_dim %arg125, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %375 = stablehlo.subtract %373, %374 : tensor<1x192x28x28xf32>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %377 = stablehlo.broadcast_in_dim %arg126, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %378 = stablehlo.multiply %376, %377 : tensor<1x192x28x28xf32>
    %379 = stablehlo.convert %arg127 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %380 = stablehlo.broadcast_in_dim %378, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %381 = stablehlo.broadcast_in_dim %379, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %382 = stablehlo.multiply %380, %381 : tensor<1x192x28x28xf32>
    %383 = stablehlo.convert %arg128 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %384 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %385 = stablehlo.broadcast_in_dim %383, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x28x28xf32>
    %386 = stablehlo.add %384, %385 : tensor<1x192x28x28xf32>
    %387 = stablehlo.convert %386 : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xbf16>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %389 = stablehlo.maximum %388, %267 : tensor<1x192x28x28xbf16>
    %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    %391 = stablehlo.minimum %269, %390 : tensor<1x192x28x28xbf16>
    %392 = stablehlo.convolution(%391, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x192x28x28xbf16>, tensor<192x1x3x3xbf16>) -> tensor<1x192x14x14xbf16>
    %393 = stablehlo.convert %392 : (tensor<1x192x14x14xbf16>) -> tensor<1x192x14x14xf32>
    %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2, 3] : (tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %395 = stablehlo.broadcast_in_dim %arg129, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x14x14xf32>
    %396 = stablehlo.subtract %394, %395 : tensor<1x192x14x14xf32>
    %397 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2, 3] : (tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %398 = stablehlo.broadcast_in_dim %arg130, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x14x14xf32>
    %399 = stablehlo.multiply %397, %398 : tensor<1x192x14x14xf32>
    %400 = stablehlo.convert %arg131 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %401 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2, 3] : (tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %402 = stablehlo.broadcast_in_dim %400, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x14x14xf32>
    %403 = stablehlo.multiply %401, %402 : tensor<1x192x14x14xf32>
    %404 = stablehlo.convert %arg132 : (tensor<192x1x1xbf16>) -> tensor<192x1x1xf32>
    %405 = stablehlo.broadcast_in_dim %403, dims = [0, 1, 2, 3] : (tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %406 = stablehlo.broadcast_in_dim %404, dims = [1, 2, 3] : (tensor<192x1x1xf32>) -> tensor<1x192x14x14xf32>
    %407 = stablehlo.add %405, %406 : tensor<1x192x14x14xf32>
    %408 = stablehlo.convert %407 : (tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xbf16>
    %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2, 3] : (tensor<1x192x14x14xbf16>) -> tensor<1x192x14x14xbf16>
    %410 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x192x14x14xbf16>
    %411 = stablehlo.maximum %409, %410 : tensor<1x192x14x14xbf16>
    %412 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x192x14x14xbf16>
    %413 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2, 3] : (tensor<1x192x14x14xbf16>) -> tensor<1x192x14x14xbf16>
    %414 = stablehlo.minimum %412, %413 : tensor<1x192x14x14xbf16>
    %415 = stablehlo.convolution(%414, %arg21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x192x14x14xbf16>, tensor<64x192x1x1xbf16>) -> tensor<1x64x14x14xbf16>
    %416 = stablehlo.convert %415 : (tensor<1x64x14x14xbf16>) -> tensor<1x64x14x14xf32>
    %417 = stablehlo.broadcast_in_dim %416, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %418 = stablehlo.broadcast_in_dim %arg133, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %419 = stablehlo.subtract %417, %418 : tensor<1x64x14x14xf32>
    %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %421 = stablehlo.broadcast_in_dim %arg134, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %422 = stablehlo.multiply %420, %421 : tensor<1x64x14x14xf32>
    %423 = stablehlo.convert %arg135 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %424 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %425 = stablehlo.broadcast_in_dim %423, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %426 = stablehlo.multiply %424, %425 : tensor<1x64x14x14xf32>
    %427 = stablehlo.convert %arg136 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %428 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %429 = stablehlo.broadcast_in_dim %427, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %430 = stablehlo.add %428, %429 : tensor<1x64x14x14xf32>
    %431 = stablehlo.convert %430 : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xbf16>
    %432 = stablehlo.convolution(%431, %arg22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x14x14xbf16>, tensor<384x64x1x1xbf16>) -> tensor<1x384x14x14xbf16>
    %433 = stablehlo.convert %432 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %434 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %435 = stablehlo.broadcast_in_dim %arg137, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %436 = stablehlo.subtract %434, %435 : tensor<1x384x14x14xf32>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %438 = stablehlo.broadcast_in_dim %arg138, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %439 = stablehlo.multiply %437, %438 : tensor<1x384x14x14xf32>
    %440 = stablehlo.convert %arg139 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %441 = stablehlo.broadcast_in_dim %439, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %442 = stablehlo.broadcast_in_dim %440, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %443 = stablehlo.multiply %441, %442 : tensor<1x384x14x14xf32>
    %444 = stablehlo.convert %arg140 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %445 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %446 = stablehlo.broadcast_in_dim %444, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %447 = stablehlo.add %445, %446 : tensor<1x384x14x14xf32>
    %448 = stablehlo.convert %447 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %449 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %450 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x384x14x14xbf16>
    %451 = stablehlo.maximum %449, %450 : tensor<1x384x14x14xbf16>
    %452 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x384x14x14xbf16>
    %453 = stablehlo.broadcast_in_dim %451, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %454 = stablehlo.minimum %452, %453 : tensor<1x384x14x14xbf16>
    %455 = stablehlo.convolution(%454, %arg23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x384x14x14xbf16>, tensor<384x1x3x3xbf16>) -> tensor<1x384x14x14xbf16>
    %456 = stablehlo.convert %455 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %458 = stablehlo.broadcast_in_dim %arg141, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %459 = stablehlo.subtract %457, %458 : tensor<1x384x14x14xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %461 = stablehlo.broadcast_in_dim %arg142, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %462 = stablehlo.multiply %460, %461 : tensor<1x384x14x14xf32>
    %463 = stablehlo.convert %arg143 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %464 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %465 = stablehlo.broadcast_in_dim %463, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %466 = stablehlo.multiply %464, %465 : tensor<1x384x14x14xf32>
    %467 = stablehlo.convert %arg144 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %468 = stablehlo.broadcast_in_dim %466, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %469 = stablehlo.broadcast_in_dim %467, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %470 = stablehlo.add %468, %469 : tensor<1x384x14x14xf32>
    %471 = stablehlo.convert %470 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %472 = stablehlo.broadcast_in_dim %471, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %473 = stablehlo.maximum %472, %450 : tensor<1x384x14x14xbf16>
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %475 = stablehlo.minimum %452, %474 : tensor<1x384x14x14xbf16>
    %476 = stablehlo.convolution(%475, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x14x14xbf16>, tensor<64x384x1x1xbf16>) -> tensor<1x64x14x14xbf16>
    %477 = stablehlo.convert %476 : (tensor<1x64x14x14xbf16>) -> tensor<1x64x14x14xf32>
    %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %479 = stablehlo.broadcast_in_dim %arg145, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %480 = stablehlo.subtract %478, %479 : tensor<1x64x14x14xf32>
    %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %482 = stablehlo.broadcast_in_dim %arg146, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %483 = stablehlo.multiply %481, %482 : tensor<1x64x14x14xf32>
    %484 = stablehlo.convert %arg147 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %485 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %486 = stablehlo.broadcast_in_dim %484, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %487 = stablehlo.multiply %485, %486 : tensor<1x64x14x14xf32>
    %488 = stablehlo.convert %arg148 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %489 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %490 = stablehlo.broadcast_in_dim %488, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %491 = stablehlo.add %489, %490 : tensor<1x64x14x14xf32>
    %492 = stablehlo.convert %491 : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xbf16>
    %493 = stablehlo.add %431, %492 : tensor<1x64x14x14xbf16>
    %494 = stablehlo.convolution(%493, %arg25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x14x14xbf16>, tensor<384x64x1x1xbf16>) -> tensor<1x384x14x14xbf16>
    %495 = stablehlo.convert %494 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %497 = stablehlo.broadcast_in_dim %arg149, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %498 = stablehlo.subtract %496, %497 : tensor<1x384x14x14xf32>
    %499 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %500 = stablehlo.broadcast_in_dim %arg150, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %501 = stablehlo.multiply %499, %500 : tensor<1x384x14x14xf32>
    %502 = stablehlo.convert %arg151 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %503 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %504 = stablehlo.broadcast_in_dim %502, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %505 = stablehlo.multiply %503, %504 : tensor<1x384x14x14xf32>
    %506 = stablehlo.convert %arg152 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %507 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %508 = stablehlo.broadcast_in_dim %506, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %509 = stablehlo.add %507, %508 : tensor<1x384x14x14xf32>
    %510 = stablehlo.convert %509 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %511 = stablehlo.broadcast_in_dim %510, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %512 = stablehlo.maximum %511, %450 : tensor<1x384x14x14xbf16>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %514 = stablehlo.minimum %452, %513 : tensor<1x384x14x14xbf16>
    %515 = stablehlo.convolution(%514, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x384x14x14xbf16>, tensor<384x1x3x3xbf16>) -> tensor<1x384x14x14xbf16>
    %516 = stablehlo.convert %515 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %518 = stablehlo.broadcast_in_dim %arg153, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %519 = stablehlo.subtract %517, %518 : tensor<1x384x14x14xf32>
    %520 = stablehlo.broadcast_in_dim %519, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %521 = stablehlo.broadcast_in_dim %arg154, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %522 = stablehlo.multiply %520, %521 : tensor<1x384x14x14xf32>
    %523 = stablehlo.convert %arg155 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %524 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %525 = stablehlo.broadcast_in_dim %523, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %526 = stablehlo.multiply %524, %525 : tensor<1x384x14x14xf32>
    %527 = stablehlo.convert %arg156 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %528 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %529 = stablehlo.broadcast_in_dim %527, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %530 = stablehlo.add %528, %529 : tensor<1x384x14x14xf32>
    %531 = stablehlo.convert %530 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %532 = stablehlo.broadcast_in_dim %531, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %533 = stablehlo.maximum %532, %450 : tensor<1x384x14x14xbf16>
    %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %535 = stablehlo.minimum %452, %534 : tensor<1x384x14x14xbf16>
    %536 = stablehlo.convolution(%535, %arg27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x14x14xbf16>, tensor<64x384x1x1xbf16>) -> tensor<1x64x14x14xbf16>
    %537 = stablehlo.convert %536 : (tensor<1x64x14x14xbf16>) -> tensor<1x64x14x14xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %539 = stablehlo.broadcast_in_dim %arg157, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %540 = stablehlo.subtract %538, %539 : tensor<1x64x14x14xf32>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %542 = stablehlo.broadcast_in_dim %arg158, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %543 = stablehlo.multiply %541, %542 : tensor<1x64x14x14xf32>
    %544 = stablehlo.convert %arg159 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %545 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %546 = stablehlo.broadcast_in_dim %544, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %547 = stablehlo.multiply %545, %546 : tensor<1x64x14x14xf32>
    %548 = stablehlo.convert %arg160 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %549 = stablehlo.broadcast_in_dim %547, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %550 = stablehlo.broadcast_in_dim %548, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %551 = stablehlo.add %549, %550 : tensor<1x64x14x14xf32>
    %552 = stablehlo.convert %551 : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xbf16>
    %553 = stablehlo.add %493, %552 : tensor<1x64x14x14xbf16>
    %554 = stablehlo.convolution(%553, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x14x14xbf16>, tensor<384x64x1x1xbf16>) -> tensor<1x384x14x14xbf16>
    %555 = stablehlo.convert %554 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %556 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %557 = stablehlo.broadcast_in_dim %arg161, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %558 = stablehlo.subtract %556, %557 : tensor<1x384x14x14xf32>
    %559 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %560 = stablehlo.broadcast_in_dim %arg162, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %561 = stablehlo.multiply %559, %560 : tensor<1x384x14x14xf32>
    %562 = stablehlo.convert %arg163 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %563 = stablehlo.broadcast_in_dim %561, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %564 = stablehlo.broadcast_in_dim %562, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %565 = stablehlo.multiply %563, %564 : tensor<1x384x14x14xf32>
    %566 = stablehlo.convert %arg164 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %567 = stablehlo.broadcast_in_dim %565, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %568 = stablehlo.broadcast_in_dim %566, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %569 = stablehlo.add %567, %568 : tensor<1x384x14x14xf32>
    %570 = stablehlo.convert %569 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %572 = stablehlo.maximum %571, %450 : tensor<1x384x14x14xbf16>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %574 = stablehlo.minimum %452, %573 : tensor<1x384x14x14xbf16>
    %575 = stablehlo.convolution(%574, %arg29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x384x14x14xbf16>, tensor<384x1x3x3xbf16>) -> tensor<1x384x14x14xbf16>
    %576 = stablehlo.convert %575 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %577 = stablehlo.broadcast_in_dim %576, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %578 = stablehlo.broadcast_in_dim %arg165, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %579 = stablehlo.subtract %577, %578 : tensor<1x384x14x14xf32>
    %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %581 = stablehlo.broadcast_in_dim %arg166, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %582 = stablehlo.multiply %580, %581 : tensor<1x384x14x14xf32>
    %583 = stablehlo.convert %arg167 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %584 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %585 = stablehlo.broadcast_in_dim %583, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %586 = stablehlo.multiply %584, %585 : tensor<1x384x14x14xf32>
    %587 = stablehlo.convert %arg168 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %588 = stablehlo.broadcast_in_dim %586, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %589 = stablehlo.broadcast_in_dim %587, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %590 = stablehlo.add %588, %589 : tensor<1x384x14x14xf32>
    %591 = stablehlo.convert %590 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %592 = stablehlo.broadcast_in_dim %591, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %593 = stablehlo.maximum %592, %450 : tensor<1x384x14x14xbf16>
    %594 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %595 = stablehlo.minimum %452, %594 : tensor<1x384x14x14xbf16>
    %596 = stablehlo.convolution(%595, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x14x14xbf16>, tensor<64x384x1x1xbf16>) -> tensor<1x64x14x14xbf16>
    %597 = stablehlo.convert %596 : (tensor<1x64x14x14xbf16>) -> tensor<1x64x14x14xf32>
    %598 = stablehlo.broadcast_in_dim %597, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %599 = stablehlo.broadcast_in_dim %arg169, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %600 = stablehlo.subtract %598, %599 : tensor<1x64x14x14xf32>
    %601 = stablehlo.broadcast_in_dim %600, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %602 = stablehlo.broadcast_in_dim %arg170, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %603 = stablehlo.multiply %601, %602 : tensor<1x64x14x14xf32>
    %604 = stablehlo.convert %arg171 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %605 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %606 = stablehlo.broadcast_in_dim %604, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %607 = stablehlo.multiply %605, %606 : tensor<1x64x14x14xf32>
    %608 = stablehlo.convert %arg172 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %609 = stablehlo.broadcast_in_dim %607, dims = [0, 1, 2, 3] : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %610 = stablehlo.broadcast_in_dim %608, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x14x14xf32>
    %611 = stablehlo.add %609, %610 : tensor<1x64x14x14xf32>
    %612 = stablehlo.convert %611 : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xbf16>
    %613 = stablehlo.add %553, %612 : tensor<1x64x14x14xbf16>
    %614 = stablehlo.convolution(%613, %arg31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x14x14xbf16>, tensor<384x64x1x1xbf16>) -> tensor<1x384x14x14xbf16>
    %615 = stablehlo.convert %614 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %616 = stablehlo.broadcast_in_dim %615, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %617 = stablehlo.broadcast_in_dim %arg173, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %618 = stablehlo.subtract %616, %617 : tensor<1x384x14x14xf32>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %620 = stablehlo.broadcast_in_dim %arg174, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %621 = stablehlo.multiply %619, %620 : tensor<1x384x14x14xf32>
    %622 = stablehlo.convert %arg175 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %623 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %624 = stablehlo.broadcast_in_dim %622, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %625 = stablehlo.multiply %623, %624 : tensor<1x384x14x14xf32>
    %626 = stablehlo.convert %arg176 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %627 = stablehlo.broadcast_in_dim %625, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %628 = stablehlo.broadcast_in_dim %626, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %629 = stablehlo.add %627, %628 : tensor<1x384x14x14xf32>
    %630 = stablehlo.convert %629 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %632 = stablehlo.maximum %631, %450 : tensor<1x384x14x14xbf16>
    %633 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %634 = stablehlo.minimum %452, %633 : tensor<1x384x14x14xbf16>
    %635 = stablehlo.convolution(%634, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x384x14x14xbf16>, tensor<384x1x3x3xbf16>) -> tensor<1x384x14x14xbf16>
    %636 = stablehlo.convert %635 : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xf32>
    %637 = stablehlo.broadcast_in_dim %636, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %638 = stablehlo.broadcast_in_dim %arg177, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %639 = stablehlo.subtract %637, %638 : tensor<1x384x14x14xf32>
    %640 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %641 = stablehlo.broadcast_in_dim %arg178, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %642 = stablehlo.multiply %640, %641 : tensor<1x384x14x14xf32>
    %643 = stablehlo.convert %arg179 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %644 = stablehlo.broadcast_in_dim %642, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %645 = stablehlo.broadcast_in_dim %643, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %646 = stablehlo.multiply %644, %645 : tensor<1x384x14x14xf32>
    %647 = stablehlo.convert %arg180 : (tensor<384x1x1xbf16>) -> tensor<384x1x1xf32>
    %648 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %649 = stablehlo.broadcast_in_dim %647, dims = [1, 2, 3] : (tensor<384x1x1xf32>) -> tensor<1x384x14x14xf32>
    %650 = stablehlo.add %648, %649 : tensor<1x384x14x14xf32>
    %651 = stablehlo.convert %650 : (tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xbf16>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %653 = stablehlo.maximum %652, %450 : tensor<1x384x14x14xbf16>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2, 3] : (tensor<1x384x14x14xbf16>) -> tensor<1x384x14x14xbf16>
    %655 = stablehlo.minimum %452, %654 : tensor<1x384x14x14xbf16>
    %656 = stablehlo.convolution(%655, %arg33) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x14x14xbf16>, tensor<96x384x1x1xbf16>) -> tensor<1x96x14x14xbf16>
    %657 = stablehlo.convert %656 : (tensor<1x96x14x14xbf16>) -> tensor<1x96x14x14xf32>
    %658 = stablehlo.broadcast_in_dim %657, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %659 = stablehlo.broadcast_in_dim %arg181, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %660 = stablehlo.subtract %658, %659 : tensor<1x96x14x14xf32>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %662 = stablehlo.broadcast_in_dim %arg182, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %663 = stablehlo.multiply %661, %662 : tensor<1x96x14x14xf32>
    %664 = stablehlo.convert %arg183 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %665 = stablehlo.broadcast_in_dim %663, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %666 = stablehlo.broadcast_in_dim %664, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %667 = stablehlo.multiply %665, %666 : tensor<1x96x14x14xf32>
    %668 = stablehlo.convert %arg184 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %669 = stablehlo.broadcast_in_dim %667, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %670 = stablehlo.broadcast_in_dim %668, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %671 = stablehlo.add %669, %670 : tensor<1x96x14x14xf32>
    %672 = stablehlo.convert %671 : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xbf16>
    %673 = stablehlo.convolution(%672, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x96x14x14xbf16>, tensor<576x96x1x1xbf16>) -> tensor<1x576x14x14xbf16>
    %674 = stablehlo.convert %673 : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xf32>
    %675 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %676 = stablehlo.broadcast_in_dim %arg185, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %677 = stablehlo.subtract %675, %676 : tensor<1x576x14x14xf32>
    %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %679 = stablehlo.broadcast_in_dim %arg186, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %680 = stablehlo.multiply %678, %679 : tensor<1x576x14x14xf32>
    %681 = stablehlo.convert %arg187 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %682 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %683 = stablehlo.broadcast_in_dim %681, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %684 = stablehlo.multiply %682, %683 : tensor<1x576x14x14xf32>
    %685 = stablehlo.convert %arg188 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %686 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %687 = stablehlo.broadcast_in_dim %685, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %688 = stablehlo.add %686, %687 : tensor<1x576x14x14xf32>
    %689 = stablehlo.convert %688 : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xbf16>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %691 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x576x14x14xbf16>
    %692 = stablehlo.maximum %690, %691 : tensor<1x576x14x14xbf16>
    %693 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x576x14x14xbf16>
    %694 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %695 = stablehlo.minimum %693, %694 : tensor<1x576x14x14xbf16>
    %696 = stablehlo.convolution(%695, %arg35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x576x14x14xbf16>, tensor<576x1x3x3xbf16>) -> tensor<1x576x14x14xbf16>
    %697 = stablehlo.convert %696 : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xf32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %699 = stablehlo.broadcast_in_dim %arg189, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %700 = stablehlo.subtract %698, %699 : tensor<1x576x14x14xf32>
    %701 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %702 = stablehlo.broadcast_in_dim %arg190, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %703 = stablehlo.multiply %701, %702 : tensor<1x576x14x14xf32>
    %704 = stablehlo.convert %arg191 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %705 = stablehlo.broadcast_in_dim %703, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %706 = stablehlo.broadcast_in_dim %704, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %707 = stablehlo.multiply %705, %706 : tensor<1x576x14x14xf32>
    %708 = stablehlo.convert %arg192 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %709 = stablehlo.broadcast_in_dim %707, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %710 = stablehlo.broadcast_in_dim %708, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %711 = stablehlo.add %709, %710 : tensor<1x576x14x14xf32>
    %712 = stablehlo.convert %711 : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xbf16>
    %713 = stablehlo.broadcast_in_dim %712, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %714 = stablehlo.maximum %713, %691 : tensor<1x576x14x14xbf16>
    %715 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %716 = stablehlo.minimum %693, %715 : tensor<1x576x14x14xbf16>
    %717 = stablehlo.convolution(%716, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x576x14x14xbf16>, tensor<96x576x1x1xbf16>) -> tensor<1x96x14x14xbf16>
    %718 = stablehlo.convert %717 : (tensor<1x96x14x14xbf16>) -> tensor<1x96x14x14xf32>
    %719 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %720 = stablehlo.broadcast_in_dim %arg193, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %721 = stablehlo.subtract %719, %720 : tensor<1x96x14x14xf32>
    %722 = stablehlo.broadcast_in_dim %721, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %723 = stablehlo.broadcast_in_dim %arg194, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %724 = stablehlo.multiply %722, %723 : tensor<1x96x14x14xf32>
    %725 = stablehlo.convert %arg195 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %726 = stablehlo.broadcast_in_dim %724, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %727 = stablehlo.broadcast_in_dim %725, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %728 = stablehlo.multiply %726, %727 : tensor<1x96x14x14xf32>
    %729 = stablehlo.convert %arg196 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %730 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %731 = stablehlo.broadcast_in_dim %729, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %732 = stablehlo.add %730, %731 : tensor<1x96x14x14xf32>
    %733 = stablehlo.convert %732 : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xbf16>
    %734 = stablehlo.add %672, %733 : tensor<1x96x14x14xbf16>
    %735 = stablehlo.convolution(%734, %arg37) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x96x14x14xbf16>, tensor<576x96x1x1xbf16>) -> tensor<1x576x14x14xbf16>
    %736 = stablehlo.convert %735 : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xf32>
    %737 = stablehlo.broadcast_in_dim %736, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %738 = stablehlo.broadcast_in_dim %arg197, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %739 = stablehlo.subtract %737, %738 : tensor<1x576x14x14xf32>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %741 = stablehlo.broadcast_in_dim %arg198, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %742 = stablehlo.multiply %740, %741 : tensor<1x576x14x14xf32>
    %743 = stablehlo.convert %arg199 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %744 = stablehlo.broadcast_in_dim %742, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %745 = stablehlo.broadcast_in_dim %743, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %746 = stablehlo.multiply %744, %745 : tensor<1x576x14x14xf32>
    %747 = stablehlo.convert %arg200 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %748 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %749 = stablehlo.broadcast_in_dim %747, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %750 = stablehlo.add %748, %749 : tensor<1x576x14x14xf32>
    %751 = stablehlo.convert %750 : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xbf16>
    %752 = stablehlo.broadcast_in_dim %751, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %753 = stablehlo.maximum %752, %691 : tensor<1x576x14x14xbf16>
    %754 = stablehlo.broadcast_in_dim %753, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %755 = stablehlo.minimum %693, %754 : tensor<1x576x14x14xbf16>
    %756 = stablehlo.convolution(%755, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x576x14x14xbf16>, tensor<576x1x3x3xbf16>) -> tensor<1x576x14x14xbf16>
    %757 = stablehlo.convert %756 : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xf32>
    %758 = stablehlo.broadcast_in_dim %757, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %759 = stablehlo.broadcast_in_dim %arg201, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %760 = stablehlo.subtract %758, %759 : tensor<1x576x14x14xf32>
    %761 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %762 = stablehlo.broadcast_in_dim %arg202, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %763 = stablehlo.multiply %761, %762 : tensor<1x576x14x14xf32>
    %764 = stablehlo.convert %arg203 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %765 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %766 = stablehlo.broadcast_in_dim %764, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %767 = stablehlo.multiply %765, %766 : tensor<1x576x14x14xf32>
    %768 = stablehlo.convert %arg204 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %769 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %770 = stablehlo.broadcast_in_dim %768, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %771 = stablehlo.add %769, %770 : tensor<1x576x14x14xf32>
    %772 = stablehlo.convert %771 : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xbf16>
    %773 = stablehlo.broadcast_in_dim %772, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %774 = stablehlo.maximum %773, %691 : tensor<1x576x14x14xbf16>
    %775 = stablehlo.broadcast_in_dim %774, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %776 = stablehlo.minimum %693, %775 : tensor<1x576x14x14xbf16>
    %777 = stablehlo.convolution(%776, %arg39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x576x14x14xbf16>, tensor<96x576x1x1xbf16>) -> tensor<1x96x14x14xbf16>
    %778 = stablehlo.convert %777 : (tensor<1x96x14x14xbf16>) -> tensor<1x96x14x14xf32>
    %779 = stablehlo.broadcast_in_dim %778, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %780 = stablehlo.broadcast_in_dim %arg205, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %781 = stablehlo.subtract %779, %780 : tensor<1x96x14x14xf32>
    %782 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %783 = stablehlo.broadcast_in_dim %arg206, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %784 = stablehlo.multiply %782, %783 : tensor<1x96x14x14xf32>
    %785 = stablehlo.convert %arg207 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %786 = stablehlo.broadcast_in_dim %784, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %787 = stablehlo.broadcast_in_dim %785, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %788 = stablehlo.multiply %786, %787 : tensor<1x96x14x14xf32>
    %789 = stablehlo.convert %arg208 : (tensor<96x1x1xbf16>) -> tensor<96x1x1xf32>
    %790 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %791 = stablehlo.broadcast_in_dim %789, dims = [1, 2, 3] : (tensor<96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %792 = stablehlo.add %790, %791 : tensor<1x96x14x14xf32>
    %793 = stablehlo.convert %792 : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xbf16>
    %794 = stablehlo.add %734, %793 : tensor<1x96x14x14xbf16>
    %795 = stablehlo.convolution(%794, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x96x14x14xbf16>, tensor<576x96x1x1xbf16>) -> tensor<1x576x14x14xbf16>
    %796 = stablehlo.convert %795 : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xf32>
    %797 = stablehlo.broadcast_in_dim %796, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %798 = stablehlo.broadcast_in_dim %arg209, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %799 = stablehlo.subtract %797, %798 : tensor<1x576x14x14xf32>
    %800 = stablehlo.broadcast_in_dim %799, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %801 = stablehlo.broadcast_in_dim %arg210, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %802 = stablehlo.multiply %800, %801 : tensor<1x576x14x14xf32>
    %803 = stablehlo.convert %arg211 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %804 = stablehlo.broadcast_in_dim %802, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %805 = stablehlo.broadcast_in_dim %803, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %806 = stablehlo.multiply %804, %805 : tensor<1x576x14x14xf32>
    %807 = stablehlo.convert %arg212 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %808 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %809 = stablehlo.broadcast_in_dim %807, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x14x14xf32>
    %810 = stablehlo.add %808, %809 : tensor<1x576x14x14xf32>
    %811 = stablehlo.convert %810 : (tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xbf16>
    %812 = stablehlo.broadcast_in_dim %811, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %813 = stablehlo.maximum %812, %691 : tensor<1x576x14x14xbf16>
    %814 = stablehlo.broadcast_in_dim %813, dims = [0, 1, 2, 3] : (tensor<1x576x14x14xbf16>) -> tensor<1x576x14x14xbf16>
    %815 = stablehlo.minimum %693, %814 : tensor<1x576x14x14xbf16>
    %816 = stablehlo.convolution(%815, %arg41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x576x14x14xbf16>, tensor<576x1x3x3xbf16>) -> tensor<1x576x7x7xbf16>
    %817 = stablehlo.convert %816 : (tensor<1x576x7x7xbf16>) -> tensor<1x576x7x7xf32>
    %818 = stablehlo.broadcast_in_dim %817, dims = [0, 1, 2, 3] : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %819 = stablehlo.broadcast_in_dim %arg213, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %820 = stablehlo.subtract %818, %819 : tensor<1x576x7x7xf32>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0, 1, 2, 3] : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %822 = stablehlo.broadcast_in_dim %arg214, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %823 = stablehlo.multiply %821, %822 : tensor<1x576x7x7xf32>
    %824 = stablehlo.convert %arg215 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %825 = stablehlo.broadcast_in_dim %823, dims = [0, 1, 2, 3] : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %826 = stablehlo.broadcast_in_dim %824, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %827 = stablehlo.multiply %825, %826 : tensor<1x576x7x7xf32>
    %828 = stablehlo.convert %arg216 : (tensor<576x1x1xbf16>) -> tensor<576x1x1xf32>
    %829 = stablehlo.broadcast_in_dim %827, dims = [0, 1, 2, 3] : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %830 = stablehlo.broadcast_in_dim %828, dims = [1, 2, 3] : (tensor<576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %831 = stablehlo.add %829, %830 : tensor<1x576x7x7xf32>
    %832 = stablehlo.convert %831 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xbf16>
    %833 = stablehlo.broadcast_in_dim %832, dims = [0, 1, 2, 3] : (tensor<1x576x7x7xbf16>) -> tensor<1x576x7x7xbf16>
    %834 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x576x7x7xbf16>
    %835 = stablehlo.maximum %833, %834 : tensor<1x576x7x7xbf16>
    %836 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x576x7x7xbf16>
    %837 = stablehlo.broadcast_in_dim %835, dims = [0, 1, 2, 3] : (tensor<1x576x7x7xbf16>) -> tensor<1x576x7x7xbf16>
    %838 = stablehlo.minimum %836, %837 : tensor<1x576x7x7xbf16>
    %839 = stablehlo.convolution(%838, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x576x7x7xbf16>, tensor<160x576x1x1xbf16>) -> tensor<1x160x7x7xbf16>
    %840 = stablehlo.convert %839 : (tensor<1x160x7x7xbf16>) -> tensor<1x160x7x7xf32>
    %841 = stablehlo.broadcast_in_dim %840, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %842 = stablehlo.broadcast_in_dim %arg217, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %843 = stablehlo.subtract %841, %842 : tensor<1x160x7x7xf32>
    %844 = stablehlo.broadcast_in_dim %843, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %845 = stablehlo.broadcast_in_dim %arg218, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %846 = stablehlo.multiply %844, %845 : tensor<1x160x7x7xf32>
    %847 = stablehlo.convert %arg219 : (tensor<160x1x1xbf16>) -> tensor<160x1x1xf32>
    %848 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %849 = stablehlo.broadcast_in_dim %847, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %850 = stablehlo.multiply %848, %849 : tensor<1x160x7x7xf32>
    %851 = stablehlo.convert %arg220 : (tensor<160x1x1xbf16>) -> tensor<160x1x1xf32>
    %852 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %853 = stablehlo.broadcast_in_dim %851, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %854 = stablehlo.add %852, %853 : tensor<1x160x7x7xf32>
    %855 = stablehlo.convert %854 : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xbf16>
    %856 = stablehlo.convolution(%855, %arg43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x160x7x7xbf16>, tensor<960x160x1x1xbf16>) -> tensor<1x960x7x7xbf16>
    %857 = stablehlo.convert %856 : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xf32>
    %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %859 = stablehlo.broadcast_in_dim %arg221, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %860 = stablehlo.subtract %858, %859 : tensor<1x960x7x7xf32>
    %861 = stablehlo.broadcast_in_dim %860, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %862 = stablehlo.broadcast_in_dim %arg222, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %863 = stablehlo.multiply %861, %862 : tensor<1x960x7x7xf32>
    %864 = stablehlo.convert %arg223 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %865 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %866 = stablehlo.broadcast_in_dim %864, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %867 = stablehlo.multiply %865, %866 : tensor<1x960x7x7xf32>
    %868 = stablehlo.convert %arg224 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %869 = stablehlo.broadcast_in_dim %867, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %870 = stablehlo.broadcast_in_dim %868, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %871 = stablehlo.add %869, %870 : tensor<1x960x7x7xf32>
    %872 = stablehlo.convert %871 : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xbf16>
    %873 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %874 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x960x7x7xbf16>
    %875 = stablehlo.maximum %873, %874 : tensor<1x960x7x7xbf16>
    %876 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x960x7x7xbf16>
    %877 = stablehlo.broadcast_in_dim %875, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %878 = stablehlo.minimum %876, %877 : tensor<1x960x7x7xbf16>
    %879 = stablehlo.convolution(%878, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x960x7x7xbf16>, tensor<960x1x3x3xbf16>) -> tensor<1x960x7x7xbf16>
    %880 = stablehlo.convert %879 : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xf32>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %882 = stablehlo.broadcast_in_dim %arg225, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %883 = stablehlo.subtract %881, %882 : tensor<1x960x7x7xf32>
    %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %885 = stablehlo.broadcast_in_dim %arg226, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %886 = stablehlo.multiply %884, %885 : tensor<1x960x7x7xf32>
    %887 = stablehlo.convert %arg227 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %888 = stablehlo.broadcast_in_dim %886, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %889 = stablehlo.broadcast_in_dim %887, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %890 = stablehlo.multiply %888, %889 : tensor<1x960x7x7xf32>
    %891 = stablehlo.convert %arg228 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %892 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %893 = stablehlo.broadcast_in_dim %891, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %894 = stablehlo.add %892, %893 : tensor<1x960x7x7xf32>
    %895 = stablehlo.convert %894 : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xbf16>
    %896 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %897 = stablehlo.maximum %896, %874 : tensor<1x960x7x7xbf16>
    %898 = stablehlo.broadcast_in_dim %897, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %899 = stablehlo.minimum %876, %898 : tensor<1x960x7x7xbf16>
    %900 = stablehlo.convolution(%899, %arg45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x960x7x7xbf16>, tensor<160x960x1x1xbf16>) -> tensor<1x160x7x7xbf16>
    %901 = stablehlo.convert %900 : (tensor<1x160x7x7xbf16>) -> tensor<1x160x7x7xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %903 = stablehlo.broadcast_in_dim %arg229, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %904 = stablehlo.subtract %902, %903 : tensor<1x160x7x7xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %906 = stablehlo.broadcast_in_dim %arg230, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %907 = stablehlo.multiply %905, %906 : tensor<1x160x7x7xf32>
    %908 = stablehlo.convert %arg231 : (tensor<160x1x1xbf16>) -> tensor<160x1x1xf32>
    %909 = stablehlo.broadcast_in_dim %907, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %910 = stablehlo.broadcast_in_dim %908, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %911 = stablehlo.multiply %909, %910 : tensor<1x160x7x7xf32>
    %912 = stablehlo.convert %arg232 : (tensor<160x1x1xbf16>) -> tensor<160x1x1xf32>
    %913 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %914 = stablehlo.broadcast_in_dim %912, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %915 = stablehlo.add %913, %914 : tensor<1x160x7x7xf32>
    %916 = stablehlo.convert %915 : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xbf16>
    %917 = stablehlo.add %855, %916 : tensor<1x160x7x7xbf16>
    %918 = stablehlo.convolution(%917, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x160x7x7xbf16>, tensor<960x160x1x1xbf16>) -> tensor<1x960x7x7xbf16>
    %919 = stablehlo.convert %918 : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xf32>
    %920 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %921 = stablehlo.broadcast_in_dim %arg233, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %922 = stablehlo.subtract %920, %921 : tensor<1x960x7x7xf32>
    %923 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %924 = stablehlo.broadcast_in_dim %arg234, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %925 = stablehlo.multiply %923, %924 : tensor<1x960x7x7xf32>
    %926 = stablehlo.convert %arg235 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %927 = stablehlo.broadcast_in_dim %925, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %928 = stablehlo.broadcast_in_dim %926, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %929 = stablehlo.multiply %927, %928 : tensor<1x960x7x7xf32>
    %930 = stablehlo.convert %arg236 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %931 = stablehlo.broadcast_in_dim %929, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %932 = stablehlo.broadcast_in_dim %930, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %933 = stablehlo.add %931, %932 : tensor<1x960x7x7xf32>
    %934 = stablehlo.convert %933 : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xbf16>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %936 = stablehlo.maximum %935, %874 : tensor<1x960x7x7xbf16>
    %937 = stablehlo.broadcast_in_dim %936, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %938 = stablehlo.minimum %876, %937 : tensor<1x960x7x7xbf16>
    %939 = stablehlo.convolution(%938, %arg47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x960x7x7xbf16>, tensor<960x1x3x3xbf16>) -> tensor<1x960x7x7xbf16>
    %940 = stablehlo.convert %939 : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xf32>
    %941 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %942 = stablehlo.broadcast_in_dim %arg237, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %943 = stablehlo.subtract %941, %942 : tensor<1x960x7x7xf32>
    %944 = stablehlo.broadcast_in_dim %943, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %945 = stablehlo.broadcast_in_dim %arg238, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %946 = stablehlo.multiply %944, %945 : tensor<1x960x7x7xf32>
    %947 = stablehlo.convert %arg239 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %948 = stablehlo.broadcast_in_dim %946, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %949 = stablehlo.broadcast_in_dim %947, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %950 = stablehlo.multiply %948, %949 : tensor<1x960x7x7xf32>
    %951 = stablehlo.convert %arg240 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %952 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %953 = stablehlo.broadcast_in_dim %951, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %954 = stablehlo.add %952, %953 : tensor<1x960x7x7xf32>
    %955 = stablehlo.convert %954 : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xbf16>
    %956 = stablehlo.broadcast_in_dim %955, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %957 = stablehlo.maximum %956, %874 : tensor<1x960x7x7xbf16>
    %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %959 = stablehlo.minimum %876, %958 : tensor<1x960x7x7xbf16>
    %960 = stablehlo.convolution(%959, %arg48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x960x7x7xbf16>, tensor<160x960x1x1xbf16>) -> tensor<1x160x7x7xbf16>
    %961 = stablehlo.convert %960 : (tensor<1x160x7x7xbf16>) -> tensor<1x160x7x7xf32>
    %962 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %963 = stablehlo.broadcast_in_dim %arg241, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %964 = stablehlo.subtract %962, %963 : tensor<1x160x7x7xf32>
    %965 = stablehlo.broadcast_in_dim %964, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %966 = stablehlo.broadcast_in_dim %arg242, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %967 = stablehlo.multiply %965, %966 : tensor<1x160x7x7xf32>
    %968 = stablehlo.convert %arg243 : (tensor<160x1x1xbf16>) -> tensor<160x1x1xf32>
    %969 = stablehlo.broadcast_in_dim %967, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %970 = stablehlo.broadcast_in_dim %968, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %971 = stablehlo.multiply %969, %970 : tensor<1x160x7x7xf32>
    %972 = stablehlo.convert %arg244 : (tensor<160x1x1xbf16>) -> tensor<160x1x1xf32>
    %973 = stablehlo.broadcast_in_dim %971, dims = [0, 1, 2, 3] : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %974 = stablehlo.broadcast_in_dim %972, dims = [1, 2, 3] : (tensor<160x1x1xf32>) -> tensor<1x160x7x7xf32>
    %975 = stablehlo.add %973, %974 : tensor<1x160x7x7xf32>
    %976 = stablehlo.convert %975 : (tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xbf16>
    %977 = stablehlo.add %917, %976 : tensor<1x160x7x7xbf16>
    %978 = stablehlo.convolution(%977, %arg49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x160x7x7xbf16>, tensor<960x160x1x1xbf16>) -> tensor<1x960x7x7xbf16>
    %979 = stablehlo.convert %978 : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xf32>
    %980 = stablehlo.broadcast_in_dim %979, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %981 = stablehlo.broadcast_in_dim %arg245, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %982 = stablehlo.subtract %980, %981 : tensor<1x960x7x7xf32>
    %983 = stablehlo.broadcast_in_dim %982, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %984 = stablehlo.broadcast_in_dim %arg246, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %985 = stablehlo.multiply %983, %984 : tensor<1x960x7x7xf32>
    %986 = stablehlo.convert %arg247 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %987 = stablehlo.broadcast_in_dim %985, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %988 = stablehlo.broadcast_in_dim %986, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %989 = stablehlo.multiply %987, %988 : tensor<1x960x7x7xf32>
    %990 = stablehlo.convert %arg248 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %991 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %992 = stablehlo.broadcast_in_dim %990, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %993 = stablehlo.add %991, %992 : tensor<1x960x7x7xf32>
    %994 = stablehlo.convert %993 : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xbf16>
    %995 = stablehlo.broadcast_in_dim %994, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %996 = stablehlo.maximum %995, %874 : tensor<1x960x7x7xbf16>
    %997 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %998 = stablehlo.minimum %876, %997 : tensor<1x960x7x7xbf16>
    %999 = stablehlo.convolution(%998, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x960x7x7xbf16>, tensor<960x1x3x3xbf16>) -> tensor<1x960x7x7xbf16>
    %1000 = stablehlo.convert %999 : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xf32>
    %1001 = stablehlo.broadcast_in_dim %1000, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %1002 = stablehlo.broadcast_in_dim %arg249, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %1003 = stablehlo.subtract %1001, %1002 : tensor<1x960x7x7xf32>
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %1005 = stablehlo.broadcast_in_dim %arg250, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %1006 = stablehlo.multiply %1004, %1005 : tensor<1x960x7x7xf32>
    %1007 = stablehlo.convert %arg251 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %1008 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %1009 = stablehlo.broadcast_in_dim %1007, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %1010 = stablehlo.multiply %1008, %1009 : tensor<1x960x7x7xf32>
    %1011 = stablehlo.convert %arg252 : (tensor<960x1x1xbf16>) -> tensor<960x1x1xf32>
    %1012 = stablehlo.broadcast_in_dim %1010, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %1013 = stablehlo.broadcast_in_dim %1011, dims = [1, 2, 3] : (tensor<960x1x1xf32>) -> tensor<1x960x7x7xf32>
    %1014 = stablehlo.add %1012, %1013 : tensor<1x960x7x7xf32>
    %1015 = stablehlo.convert %1014 : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xbf16>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %1017 = stablehlo.maximum %1016, %874 : tensor<1x960x7x7xbf16>
    %1018 = stablehlo.broadcast_in_dim %1017, dims = [0, 1, 2, 3] : (tensor<1x960x7x7xbf16>) -> tensor<1x960x7x7xbf16>
    %1019 = stablehlo.minimum %876, %1018 : tensor<1x960x7x7xbf16>
    %1020 = stablehlo.convolution(%1019, %arg51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x960x7x7xbf16>, tensor<320x960x1x1xbf16>) -> tensor<1x320x7x7xbf16>
    %1021 = stablehlo.convert %1020 : (tensor<1x320x7x7xbf16>) -> tensor<1x320x7x7xf32>
    %1022 = stablehlo.broadcast_in_dim %1021, dims = [0, 1, 2, 3] : (tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %1023 = stablehlo.broadcast_in_dim %arg253, dims = [1, 2, 3] : (tensor<320x1x1xf32>) -> tensor<1x320x7x7xf32>
    %1024 = stablehlo.subtract %1022, %1023 : tensor<1x320x7x7xf32>
    %1025 = stablehlo.broadcast_in_dim %1024, dims = [0, 1, 2, 3] : (tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %1026 = stablehlo.broadcast_in_dim %arg254, dims = [1, 2, 3] : (tensor<320x1x1xf32>) -> tensor<1x320x7x7xf32>
    %1027 = stablehlo.multiply %1025, %1026 : tensor<1x320x7x7xf32>
    %1028 = stablehlo.convert %arg255 : (tensor<320x1x1xbf16>) -> tensor<320x1x1xf32>
    %1029 = stablehlo.broadcast_in_dim %1027, dims = [0, 1, 2, 3] : (tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %1030 = stablehlo.broadcast_in_dim %1028, dims = [1, 2, 3] : (tensor<320x1x1xf32>) -> tensor<1x320x7x7xf32>
    %1031 = stablehlo.multiply %1029, %1030 : tensor<1x320x7x7xf32>
    %1032 = stablehlo.convert %arg256 : (tensor<320x1x1xbf16>) -> tensor<320x1x1xf32>
    %1033 = stablehlo.broadcast_in_dim %1031, dims = [0, 1, 2, 3] : (tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %1034 = stablehlo.broadcast_in_dim %1032, dims = [1, 2, 3] : (tensor<320x1x1xf32>) -> tensor<1x320x7x7xf32>
    %1035 = stablehlo.add %1033, %1034 : tensor<1x320x7x7xf32>
    %1036 = stablehlo.convert %1035 : (tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xbf16>
    %1037 = stablehlo.convolution(%1036, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x320x7x7xbf16>, tensor<1280x320x1x1xbf16>) -> tensor<1x1280x7x7xbf16>
    %1038 = stablehlo.convert %1037 : (tensor<1x1280x7x7xbf16>) -> tensor<1x1280x7x7xf32>
    %1039 = stablehlo.broadcast_in_dim %1038, dims = [0, 1, 2, 3] : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %1040 = stablehlo.broadcast_in_dim %arg257, dims = [1, 2, 3] : (tensor<1280x1x1xf32>) -> tensor<1x1280x7x7xf32>
    %1041 = stablehlo.subtract %1039, %1040 : tensor<1x1280x7x7xf32>
    %1042 = stablehlo.broadcast_in_dim %1041, dims = [0, 1, 2, 3] : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %1043 = stablehlo.broadcast_in_dim %arg258, dims = [1, 2, 3] : (tensor<1280x1x1xf32>) -> tensor<1x1280x7x7xf32>
    %1044 = stablehlo.multiply %1042, %1043 : tensor<1x1280x7x7xf32>
    %1045 = stablehlo.convert %arg259 : (tensor<1280x1x1xbf16>) -> tensor<1280x1x1xf32>
    %1046 = stablehlo.broadcast_in_dim %1044, dims = [0, 1, 2, 3] : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %1047 = stablehlo.broadcast_in_dim %1045, dims = [1, 2, 3] : (tensor<1280x1x1xf32>) -> tensor<1x1280x7x7xf32>
    %1048 = stablehlo.multiply %1046, %1047 : tensor<1x1280x7x7xf32>
    %1049 = stablehlo.convert %arg260 : (tensor<1280x1x1xbf16>) -> tensor<1280x1x1xf32>
    %1050 = stablehlo.broadcast_in_dim %1048, dims = [0, 1, 2, 3] : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %1051 = stablehlo.broadcast_in_dim %1049, dims = [1, 2, 3] : (tensor<1280x1x1xf32>) -> tensor<1x1280x7x7xf32>
    %1052 = stablehlo.add %1050, %1051 : tensor<1x1280x7x7xf32>
    %1053 = stablehlo.convert %1052 : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xbf16>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2, 3] : (tensor<1x1280x7x7xbf16>) -> tensor<1x1280x7x7xbf16>
    %1055 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x1280x7x7xbf16>
    %1056 = stablehlo.maximum %1054, %1055 : tensor<1x1280x7x7xbf16>
    %1057 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<1x1280x7x7xbf16>
    %1058 = stablehlo.broadcast_in_dim %1056, dims = [0, 1, 2, 3] : (tensor<1x1280x7x7xbf16>) -> tensor<1x1280x7x7xbf16>
    %1059 = stablehlo.minimum %1057, %1058 : tensor<1x1280x7x7xbf16>
    %1060 = stablehlo.reduce(%1059 init: %cst_1) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x1280x7x7xbf16>, tensor<bf16>) -> tensor<1x1280xbf16>
    %1061 = stablehlo.reshape %1060 : (tensor<1x1280xbf16>) -> tensor<1x1280x1x1xbf16>
    %1062 = stablehlo.convert %cst_2 : (tensor<1xi64>) -> tensor<1xbf16>
    %1063 = stablehlo.reshape %1062 : (tensor<1xbf16>) -> tensor<bf16>
    %1064 = stablehlo.broadcast_in_dim %1061, dims = [0, 1, 2, 3] : (tensor<1x1280x1x1xbf16>) -> tensor<1x1280x1x1xbf16>
    %1065 = stablehlo.broadcast_in_dim %1063, dims = [] : (tensor<bf16>) -> tensor<1x1280x1x1xbf16>
    %1066 = stablehlo.divide %1064, %1065 : tensor<1x1280x1x1xbf16>
    %1067 = stablehlo.reshape %1066 : (tensor<1x1280x1x1xbf16>) -> tensor<1x1280xbf16>
    %1068 = stablehlo.convert %1067 : (tensor<1x1280xbf16>) -> tensor<1x1280xf32>
    %1069 = stablehlo.dot_general %1068, %arg261, contracting_dims = [1] x [0] : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
    %1070 = stablehlo.convert %cst_3 : (tensor<1xi64>) -> tensor<1xf32>
    %1071 = stablehlo.reshape %1070 : (tensor<1xf32>) -> tensor<f32>
    %1072 = stablehlo.broadcast_in_dim %1069, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %1073 = stablehlo.broadcast_in_dim %1071, dims = [] : (tensor<f32>) -> tensor<1x1000xf32>
    %1074 = stablehlo.multiply %1072, %1073 : tensor<1x1000xf32>
    %1075 = stablehlo.broadcast_in_dim %1074, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %1076 = stablehlo.broadcast_in_dim %arg262, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %1077 = stablehlo.add %1075, %1076 : tensor<1x1000xf32>
    %1078 = stablehlo.convert %1077 : (tensor<1x1000xf32>) -> tensor<1x1000xbf16>
    return %1078 : tensor<1x1000xbf16>
  }
}
