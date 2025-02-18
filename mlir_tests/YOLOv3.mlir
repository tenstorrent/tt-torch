module {
  func.func @main(%arg0: tensor<1x3x512x512xbf16>, %arg1: tensor<32x3x3x3xbf16>, %arg2: tensor<64x32x3x3xbf16>, %arg3: tensor<32x64x1x1xbf16>, %arg4: tensor<64x32x3x3xbf16>, %arg5: tensor<128x64x3x3xbf16>, %arg6: tensor<64x128x1x1xbf16>, %arg7: tensor<128x64x3x3xbf16>, %arg8: tensor<64x128x1x1xbf16>, %arg9: tensor<128x64x3x3xbf16>, %arg10: tensor<256x128x3x3xbf16>, %arg11: tensor<128x256x1x1xbf16>, %arg12: tensor<256x128x3x3xbf16>, %arg13: tensor<128x256x1x1xbf16>, %arg14: tensor<256x128x3x3xbf16>, %arg15: tensor<128x256x1x1xbf16>, %arg16: tensor<256x128x3x3xbf16>, %arg17: tensor<128x256x1x1xbf16>, %arg18: tensor<256x128x3x3xbf16>, %arg19: tensor<128x256x1x1xbf16>, %arg20: tensor<256x128x3x3xbf16>, %arg21: tensor<128x256x1x1xbf16>, %arg22: tensor<256x128x3x3xbf16>, %arg23: tensor<128x256x1x1xbf16>, %arg24: tensor<256x128x3x3xbf16>, %arg25: tensor<128x256x1x1xbf16>, %arg26: tensor<256x128x3x3xbf16>, %arg27: tensor<512x256x3x3xbf16>, %arg28: tensor<256x512x1x1xbf16>, %arg29: tensor<512x256x3x3xbf16>, %arg30: tensor<256x512x1x1xbf16>, %arg31: tensor<512x256x3x3xbf16>, %arg32: tensor<256x512x1x1xbf16>, %arg33: tensor<512x256x3x3xbf16>, %arg34: tensor<256x512x1x1xbf16>, %arg35: tensor<512x256x3x3xbf16>, %arg36: tensor<256x512x1x1xbf16>, %arg37: tensor<512x256x3x3xbf16>, %arg38: tensor<256x512x1x1xbf16>, %arg39: tensor<512x256x3x3xbf16>, %arg40: tensor<256x512x1x1xbf16>, %arg41: tensor<512x256x3x3xbf16>, %arg42: tensor<256x512x1x1xbf16>, %arg43: tensor<512x256x3x3xbf16>, %arg44: tensor<1024x512x3x3xbf16>, %arg45: tensor<512x1024x1x1xbf16>, %arg46: tensor<1024x512x3x3xbf16>, %arg47: tensor<512x1024x1x1xbf16>, %arg48: tensor<1024x512x3x3xbf16>, %arg49: tensor<512x1024x1x1xbf16>, %arg50: tensor<1024x512x3x3xbf16>, %arg51: tensor<512x1024x1x1xbf16>, %arg52: tensor<1024x512x3x3xbf16>, %arg53: tensor<512x1024x1x1xbf16>, %arg54: tensor<1024x512x3x3xbf16>, %arg55: tensor<512x1024x1x1xbf16>, %arg56: tensor<1024x512x3x3xbf16>, %arg57: tensor<512x1024x1x1xbf16>, %arg58: tensor<1024x512x3x3xbf16>, %arg59: tensor<255x1024x1x1xbf16>, %arg60: tensor<255xbf16>, %arg61: tensor<256x512x1x1xbf16>, %arg62: tensor<256x768x1x1xbf16>, %arg63: tensor<512x256x3x3xbf16>, %arg64: tensor<256x512x1x1xbf16>, %arg65: tensor<512x256x3x3xbf16>, %arg66: tensor<256x512x1x1xbf16>, %arg67: tensor<512x256x3x3xbf16>, %arg68: tensor<255x512x1x1xbf16>, %arg69: tensor<255xbf16>, %arg70: tensor<128x256x1x1xbf16>, %arg71: tensor<128x384x1x1xbf16>, %arg72: tensor<256x128x3x3xbf16>, %arg73: tensor<128x256x1x1xbf16>, %arg74: tensor<256x128x3x3xbf16>, %arg75: tensor<128x256x1x1xbf16>, %arg76: tensor<256x128x3x3xbf16>, %arg77: tensor<255x256x1x1xbf16>, %arg78: tensor<255xbf16>, %arg79: tensor<32x1x1xf32>, %arg80: tensor<32x1x1xf32>, %arg81: tensor<32x1x1xbf16>, %arg82: tensor<32x1x1xbf16>, %arg83: tensor<64x1x1xf32>, %arg84: tensor<64x1x1xf32>, %arg85: tensor<64x1x1xbf16>, %arg86: tensor<64x1x1xbf16>, %arg87: tensor<32x1x1xf32>, %arg88: tensor<32x1x1xf32>, %arg89: tensor<32x1x1xbf16>, %arg90: tensor<32x1x1xbf16>, %arg91: tensor<64x1x1xf32>, %arg92: tensor<64x1x1xf32>, %arg93: tensor<64x1x1xbf16>, %arg94: tensor<64x1x1xbf16>, %arg95: tensor<128x1x1xf32>, %arg96: tensor<128x1x1xf32>, %arg97: tensor<128x1x1xbf16>, %arg98: tensor<128x1x1xbf16>, %arg99: tensor<64x1x1xf32>, %arg100: tensor<64x1x1xf32>, %arg101: tensor<64x1x1xbf16>, %arg102: tensor<64x1x1xbf16>, %arg103: tensor<128x1x1xf32>, %arg104: tensor<128x1x1xf32>, %arg105: tensor<128x1x1xbf16>, %arg106: tensor<128x1x1xbf16>, %arg107: tensor<64x1x1xf32>, %arg108: tensor<64x1x1xf32>, %arg109: tensor<64x1x1xbf16>, %arg110: tensor<64x1x1xbf16>, %arg111: tensor<128x1x1xf32>, %arg112: tensor<128x1x1xf32>, %arg113: tensor<128x1x1xbf16>, %arg114: tensor<128x1x1xbf16>, %arg115: tensor<256x1x1xf32>, %arg116: tensor<256x1x1xf32>, %arg117: tensor<256x1x1xbf16>, %arg118: tensor<256x1x1xbf16>, %arg119: tensor<128x1x1xf32>, %arg120: tensor<128x1x1xf32>, %arg121: tensor<128x1x1xbf16>, %arg122: tensor<128x1x1xbf16>, %arg123: tensor<256x1x1xf32>, %arg124: tensor<256x1x1xf32>, %arg125: tensor<256x1x1xbf16>, %arg126: tensor<256x1x1xbf16>, %arg127: tensor<128x1x1xf32>, %arg128: tensor<128x1x1xf32>, %arg129: tensor<128x1x1xbf16>, %arg130: tensor<128x1x1xbf16>, %arg131: tensor<256x1x1xf32>, %arg132: tensor<256x1x1xf32>, %arg133: tensor<256x1x1xbf16>, %arg134: tensor<256x1x1xbf16>, %arg135: tensor<128x1x1xf32>, %arg136: tensor<128x1x1xf32>, %arg137: tensor<128x1x1xbf16>, %arg138: tensor<128x1x1xbf16>, %arg139: tensor<256x1x1xf32>, %arg140: tensor<256x1x1xf32>, %arg141: tensor<256x1x1xbf16>, %arg142: tensor<256x1x1xbf16>, %arg143: tensor<128x1x1xf32>, %arg144: tensor<128x1x1xf32>, %arg145: tensor<128x1x1xbf16>, %arg146: tensor<128x1x1xbf16>, %arg147: tensor<256x1x1xf32>, %arg148: tensor<256x1x1xf32>, %arg149: tensor<256x1x1xbf16>, %arg150: tensor<256x1x1xbf16>, %arg151: tensor<128x1x1xf32>, %arg152: tensor<128x1x1xf32>, %arg153: tensor<128x1x1xbf16>, %arg154: tensor<128x1x1xbf16>, %arg155: tensor<256x1x1xf32>, %arg156: tensor<256x1x1xf32>, %arg157: tensor<256x1x1xbf16>, %arg158: tensor<256x1x1xbf16>, %arg159: tensor<128x1x1xf32>, %arg160: tensor<128x1x1xf32>, %arg161: tensor<128x1x1xbf16>, %arg162: tensor<128x1x1xbf16>, %arg163: tensor<256x1x1xf32>, %arg164: tensor<256x1x1xf32>, %arg165: tensor<256x1x1xbf16>, %arg166: tensor<256x1x1xbf16>, %arg167: tensor<128x1x1xf32>, %arg168: tensor<128x1x1xf32>, %arg169: tensor<128x1x1xbf16>, %arg170: tensor<128x1x1xbf16>, %arg171: tensor<256x1x1xf32>, %arg172: tensor<256x1x1xf32>, %arg173: tensor<256x1x1xbf16>, %arg174: tensor<256x1x1xbf16>, %arg175: tensor<128x1x1xf32>, %arg176: tensor<128x1x1xf32>, %arg177: tensor<128x1x1xbf16>, %arg178: tensor<128x1x1xbf16>, %arg179: tensor<256x1x1xf32>, %arg180: tensor<256x1x1xf32>, %arg181: tensor<256x1x1xbf16>, %arg182: tensor<256x1x1xbf16>, %arg183: tensor<512x1x1xf32>, %arg184: tensor<512x1x1xf32>, %arg185: tensor<512x1x1xbf16>, %arg186: tensor<512x1x1xbf16>, %arg187: tensor<256x1x1xf32>, %arg188: tensor<256x1x1xf32>, %arg189: tensor<256x1x1xbf16>, %arg190: tensor<256x1x1xbf16>, %arg191: tensor<512x1x1xf32>, %arg192: tensor<512x1x1xf32>, %arg193: tensor<512x1x1xbf16>, %arg194: tensor<512x1x1xbf16>, %arg195: tensor<256x1x1xf32>, %arg196: tensor<256x1x1xf32>, %arg197: tensor<256x1x1xbf16>, %arg198: tensor<256x1x1xbf16>, %arg199: tensor<512x1x1xf32>, %arg200: tensor<512x1x1xf32>, %arg201: tensor<512x1x1xbf16>, %arg202: tensor<512x1x1xbf16>, %arg203: tensor<256x1x1xf32>, %arg204: tensor<256x1x1xf32>, %arg205: tensor<256x1x1xbf16>, %arg206: tensor<256x1x1xbf16>, %arg207: tensor<512x1x1xf32>, %arg208: tensor<512x1x1xf32>, %arg209: tensor<512x1x1xbf16>, %arg210: tensor<512x1x1xbf16>, %arg211: tensor<256x1x1xf32>, %arg212: tensor<256x1x1xf32>, %arg213: tensor<256x1x1xbf16>, %arg214: tensor<256x1x1xbf16>, %arg215: tensor<512x1x1xf32>, %arg216: tensor<512x1x1xf32>, %arg217: tensor<512x1x1xbf16>, %arg218: tensor<512x1x1xbf16>, %arg219: tensor<256x1x1xf32>, %arg220: tensor<256x1x1xf32>, %arg221: tensor<256x1x1xbf16>, %arg222: tensor<256x1x1xbf16>, %arg223: tensor<512x1x1xf32>, %arg224: tensor<512x1x1xf32>, %arg225: tensor<512x1x1xbf16>, %arg226: tensor<512x1x1xbf16>, %arg227: tensor<256x1x1xf32>, %arg228: tensor<256x1x1xf32>, %arg229: tensor<256x1x1xbf16>, %arg230: tensor<256x1x1xbf16>, %arg231: tensor<512x1x1xf32>, %arg232: tensor<512x1x1xf32>, %arg233: tensor<512x1x1xbf16>, %arg234: tensor<512x1x1xbf16>, %arg235: tensor<256x1x1xf32>, %arg236: tensor<256x1x1xf32>, %arg237: tensor<256x1x1xbf16>, %arg238: tensor<256x1x1xbf16>, %arg239: tensor<512x1x1xf32>, %arg240: tensor<512x1x1xf32>, %arg241: tensor<512x1x1xbf16>, %arg242: tensor<512x1x1xbf16>, %arg243: tensor<256x1x1xf32>, %arg244: tensor<256x1x1xf32>, %arg245: tensor<256x1x1xbf16>, %arg246: tensor<256x1x1xbf16>, %arg247: tensor<512x1x1xf32>, %arg248: tensor<512x1x1xf32>, %arg249: tensor<512x1x1xbf16>, %arg250: tensor<512x1x1xbf16>, %arg251: tensor<1024x1x1xf32>, %arg252: tensor<1024x1x1xf32>, %arg253: tensor<1024x1x1xbf16>, %arg254: tensor<1024x1x1xbf16>, %arg255: tensor<512x1x1xf32>, %arg256: tensor<512x1x1xf32>, %arg257: tensor<512x1x1xbf16>, %arg258: tensor<512x1x1xbf16>, %arg259: tensor<1024x1x1xf32>, %arg260: tensor<1024x1x1xf32>, %arg261: tensor<1024x1x1xbf16>, %arg262: tensor<1024x1x1xbf16>, %arg263: tensor<512x1x1xf32>, %arg264: tensor<512x1x1xf32>, %arg265: tensor<512x1x1xbf16>, %arg266: tensor<512x1x1xbf16>, %arg267: tensor<1024x1x1xf32>, %arg268: tensor<1024x1x1xf32>, %arg269: tensor<1024x1x1xbf16>, %arg270: tensor<1024x1x1xbf16>, %arg271: tensor<512x1x1xf32>, %arg272: tensor<512x1x1xf32>, %arg273: tensor<512x1x1xbf16>, %arg274: tensor<512x1x1xbf16>, %arg275: tensor<1024x1x1xf32>, %arg276: tensor<1024x1x1xf32>, %arg277: tensor<1024x1x1xbf16>, %arg278: tensor<1024x1x1xbf16>, %arg279: tensor<512x1x1xf32>, %arg280: tensor<512x1x1xf32>, %arg281: tensor<512x1x1xbf16>, %arg282: tensor<512x1x1xbf16>, %arg283: tensor<1024x1x1xf32>, %arg284: tensor<1024x1x1xf32>, %arg285: tensor<1024x1x1xbf16>, %arg286: tensor<1024x1x1xbf16>, %arg287: tensor<512x1x1xf32>, %arg288: tensor<512x1x1xf32>, %arg289: tensor<512x1x1xbf16>, %arg290: tensor<512x1x1xbf16>, %arg291: tensor<1024x1x1xf32>, %arg292: tensor<1024x1x1xf32>, %arg293: tensor<1024x1x1xbf16>, %arg294: tensor<1024x1x1xbf16>, %arg295: tensor<512x1x1xf32>, %arg296: tensor<512x1x1xf32>, %arg297: tensor<512x1x1xbf16>, %arg298: tensor<512x1x1xbf16>, %arg299: tensor<1024x1x1xf32>, %arg300: tensor<1024x1x1xf32>, %arg301: tensor<1024x1x1xbf16>, %arg302: tensor<1024x1x1xbf16>, %arg303: tensor<512x1x1xf32>, %arg304: tensor<512x1x1xf32>, %arg305: tensor<512x1x1xbf16>, %arg306: tensor<512x1x1xbf16>, %arg307: tensor<1024x1x1xf32>, %arg308: tensor<1024x1x1xf32>, %arg309: tensor<1024x1x1xbf16>, %arg310: tensor<1024x1x1xbf16>, %arg311: tensor<256x1x1xf32>, %arg312: tensor<256x1x1xf32>, %arg313: tensor<256x1x1xbf16>, %arg314: tensor<256x1x1xbf16>, %arg315: tensor<256x16x32xbf16>, %arg316: tensor<256x16x32xbf16>, %arg317: tensor<256x1x1xf32>, %arg318: tensor<256x1x1xf32>, %arg319: tensor<256x1x1xbf16>, %arg320: tensor<256x1x1xbf16>, %arg321: tensor<512x1x1xf32>, %arg322: tensor<512x1x1xf32>, %arg323: tensor<512x1x1xbf16>, %arg324: tensor<512x1x1xbf16>, %arg325: tensor<256x1x1xf32>, %arg326: tensor<256x1x1xf32>, %arg327: tensor<256x1x1xbf16>, %arg328: tensor<256x1x1xbf16>, %arg329: tensor<512x1x1xf32>, %arg330: tensor<512x1x1xf32>, %arg331: tensor<512x1x1xbf16>, %arg332: tensor<512x1x1xbf16>, %arg333: tensor<256x1x1xf32>, %arg334: tensor<256x1x1xf32>, %arg335: tensor<256x1x1xbf16>, %arg336: tensor<256x1x1xbf16>, %arg337: tensor<512x1x1xf32>, %arg338: tensor<512x1x1xf32>, %arg339: tensor<512x1x1xbf16>, %arg340: tensor<512x1x1xbf16>, %arg341: tensor<128x1x1xf32>, %arg342: tensor<128x1x1xf32>, %arg343: tensor<128x1x1xbf16>, %arg344: tensor<128x1x1xbf16>, %arg345: tensor<128x32x64xbf16>, %arg346: tensor<128x32x64xbf16>, %arg347: tensor<128x1x1xf32>, %arg348: tensor<128x1x1xf32>, %arg349: tensor<128x1x1xbf16>, %arg350: tensor<128x1x1xbf16>, %arg351: tensor<256x1x1xf32>, %arg352: tensor<256x1x1xf32>, %arg353: tensor<256x1x1xbf16>, %arg354: tensor<256x1x1xbf16>, %arg355: tensor<128x1x1xf32>, %arg356: tensor<128x1x1xf32>, %arg357: tensor<128x1x1xbf16>, %arg358: tensor<128x1x1xbf16>, %arg359: tensor<256x1x1xf32>, %arg360: tensor<256x1x1xf32>, %arg361: tensor<256x1x1xbf16>, %arg362: tensor<256x1x1xbf16>, %arg363: tensor<128x1x1xf32>, %arg364: tensor<128x1x1xf32>, %arg365: tensor<128x1x1xbf16>, %arg366: tensor<128x1x1xbf16>, %arg367: tensor<256x1x1xf32>, %arg368: tensor<256x1x1xf32>, %arg369: tensor<256x1x1xbf16>, %arg370: tensor<256x1x1xbf16>) -> (tensor<1x255x16x16xbf16>, tensor<1x255x32x32xbf16>, tensor<1x255x64x64xbf16>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %cst = arith.constant dense<1.000000e-01> : tensor<1xf64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x512x512xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x32x512x512xbf16>
    %1 = stablehlo.convert %0 : (tensor<1x32x512x512xbf16>) -> tensor<1x32x512x512xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
    %3 = stablehlo.broadcast_in_dim %arg79, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x512x512xf32>
    %4 = stablehlo.subtract %2, %3 : tensor<1x32x512x512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg80, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x512x512xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x32x512x512xf32>
    %8 = stablehlo.convert %arg81 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x512x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x32x512x512xf32>
    %12 = stablehlo.convert %arg82 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x512x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<1x32x512x512xf32>
    %16 = stablehlo.convert %15 : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xbf16>
    %17 = stablehlo.convert %c : (tensor<i64>) -> tensor<bf16>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x32x512x512xbf16>
    %19 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x32x512x512xbf16>) -> tensor<1x32x512x512xbf16>
    %20 = stablehlo.maximum %18, %19 : tensor<1x32x512x512xbf16>
    %21 = stablehlo.minimum %18, %19 : tensor<1x32x512x512xbf16>
    %22 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    %23 = stablehlo.reshape %22 : (tensor<1xbf16>) -> tensor<bf16>
    %24 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2, 3] : (tensor<1x32x512x512xbf16>) -> tensor<1x32x512x512xbf16>
    %25 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x32x512x512xbf16>
    %26 = stablehlo.multiply %24, %25 : tensor<1x32x512x512xbf16>
    %27 = stablehlo.add %20, %26 : tensor<1x32x512x512xbf16>
    %28 = stablehlo.convolution(%27, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x512x512xbf16>, tensor<64x32x3x3xbf16>) -> tensor<1x64x256x256xbf16>
    %29 = stablehlo.convert %28 : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %31 = stablehlo.broadcast_in_dim %arg83, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %32 = stablehlo.subtract %30, %31 : tensor<1x64x256x256xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %34 = stablehlo.broadcast_in_dim %arg84, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<1x64x256x256xf32>
    %36 = stablehlo.convert %arg85 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %37 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %38 = stablehlo.broadcast_in_dim %36, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %39 = stablehlo.multiply %37, %38 : tensor<1x64x256x256xf32>
    %40 = stablehlo.convert %arg86 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %41 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %42 = stablehlo.broadcast_in_dim %40, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %43 = stablehlo.add %41, %42 : tensor<1x64x256x256xf32>
    %44 = stablehlo.convert %43 : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xbf16>
    %45 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x64x256x256xbf16>
    %46 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>
    %47 = stablehlo.maximum %45, %46 : tensor<1x64x256x256xbf16>
    %48 = stablehlo.minimum %45, %46 : tensor<1x64x256x256xbf16>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>
    %50 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x64x256x256xbf16>
    %51 = stablehlo.multiply %49, %50 : tensor<1x64x256x256xbf16>
    %52 = stablehlo.add %47, %51 : tensor<1x64x256x256xbf16>
    %53 = stablehlo.convolution(%52, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x256x256xbf16>, tensor<32x64x1x1xbf16>) -> tensor<1x32x256x256xbf16>
    %54 = stablehlo.convert %53 : (tensor<1x32x256x256xbf16>) -> tensor<1x32x256x256xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xf32>
    %56 = stablehlo.broadcast_in_dim %arg87, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x256x256xf32>
    %57 = stablehlo.subtract %55, %56 : tensor<1x32x256x256xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xf32>
    %59 = stablehlo.broadcast_in_dim %arg88, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x256x256xf32>
    %60 = stablehlo.multiply %58, %59 : tensor<1x32x256x256xf32>
    %61 = stablehlo.convert %arg89 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2, 3] : (tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xf32>
    %63 = stablehlo.broadcast_in_dim %61, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x256x256xf32>
    %64 = stablehlo.multiply %62, %63 : tensor<1x32x256x256xf32>
    %65 = stablehlo.convert %arg90 : (tensor<32x1x1xbf16>) -> tensor<32x1x1xf32>
    %66 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xf32>
    %67 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x256x256xf32>
    %68 = stablehlo.add %66, %67 : tensor<1x32x256x256xf32>
    %69 = stablehlo.convert %68 : (tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xbf16>
    %70 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x32x256x256xbf16>
    %71 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x32x256x256xbf16>) -> tensor<1x32x256x256xbf16>
    %72 = stablehlo.maximum %70, %71 : tensor<1x32x256x256xbf16>
    %73 = stablehlo.minimum %70, %71 : tensor<1x32x256x256xbf16>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x32x256x256xbf16>) -> tensor<1x32x256x256xbf16>
    %75 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x32x256x256xbf16>
    %76 = stablehlo.multiply %74, %75 : tensor<1x32x256x256xbf16>
    %77 = stablehlo.add %72, %76 : tensor<1x32x256x256xbf16>
    %78 = stablehlo.convolution(%77, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x256x256xbf16>, tensor<64x32x3x3xbf16>) -> tensor<1x64x256x256xbf16>
    %79 = stablehlo.convert %78 : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %81 = stablehlo.broadcast_in_dim %arg91, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %82 = stablehlo.subtract %80, %81 : tensor<1x64x256x256xf32>
    %83 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %84 = stablehlo.broadcast_in_dim %arg92, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %85 = stablehlo.multiply %83, %84 : tensor<1x64x256x256xf32>
    %86 = stablehlo.convert %arg93 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %87 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %88 = stablehlo.broadcast_in_dim %86, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %89 = stablehlo.multiply %87, %88 : tensor<1x64x256x256xf32>
    %90 = stablehlo.convert %arg94 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %91 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xf32>
    %92 = stablehlo.broadcast_in_dim %90, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x256x256xf32>
    %93 = stablehlo.add %91, %92 : tensor<1x64x256x256xf32>
    %94 = stablehlo.convert %93 : (tensor<1x64x256x256xf32>) -> tensor<1x64x256x256xbf16>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>
    %96 = stablehlo.maximum %45, %95 : tensor<1x64x256x256xbf16>
    %97 = stablehlo.minimum %45, %95 : tensor<1x64x256x256xbf16>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>
    %99 = stablehlo.multiply %98, %50 : tensor<1x64x256x256xbf16>
    %100 = stablehlo.add %96, %99 : tensor<1x64x256x256xbf16>
    %101 = stablehlo.add %100, %52 : tensor<1x64x256x256xbf16>
    %102 = stablehlo.convolution(%101, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x256x256xbf16>, tensor<128x64x3x3xbf16>) -> tensor<1x128x128x128xbf16>
    %103 = stablehlo.convert %102 : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xf32>
    %104 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %105 = stablehlo.broadcast_in_dim %arg95, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %106 = stablehlo.subtract %104, %105 : tensor<1x128x128x128xf32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %108 = stablehlo.broadcast_in_dim %arg96, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %109 = stablehlo.multiply %107, %108 : tensor<1x128x128x128xf32>
    %110 = stablehlo.convert %arg97 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %111 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %112 = stablehlo.broadcast_in_dim %110, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %113 = stablehlo.multiply %111, %112 : tensor<1x128x128x128xf32>
    %114 = stablehlo.convert %arg98 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %115 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %116 = stablehlo.broadcast_in_dim %114, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %117 = stablehlo.add %115, %116 : tensor<1x128x128x128xf32>
    %118 = stablehlo.convert %117 : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xbf16>
    %119 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x128x128x128xbf16>
    %120 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %121 = stablehlo.maximum %119, %120 : tensor<1x128x128x128xbf16>
    %122 = stablehlo.minimum %119, %120 : tensor<1x128x128x128xbf16>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %124 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x128x128x128xbf16>
    %125 = stablehlo.multiply %123, %124 : tensor<1x128x128x128xbf16>
    %126 = stablehlo.add %121, %125 : tensor<1x128x128x128xbf16>
    %127 = stablehlo.convolution(%126, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x128x128xbf16>, tensor<64x128x1x1xbf16>) -> tensor<1x64x128x128xbf16>
    %128 = stablehlo.convert %127 : (tensor<1x64x128x128xbf16>) -> tensor<1x64x128x128xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %130 = stablehlo.broadcast_in_dim %arg99, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %131 = stablehlo.subtract %129, %130 : tensor<1x64x128x128xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %133 = stablehlo.broadcast_in_dim %arg100, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %134 = stablehlo.multiply %132, %133 : tensor<1x64x128x128xf32>
    %135 = stablehlo.convert %arg101 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %136 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %137 = stablehlo.broadcast_in_dim %135, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %138 = stablehlo.multiply %136, %137 : tensor<1x64x128x128xf32>
    %139 = stablehlo.convert %arg102 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %140 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %141 = stablehlo.broadcast_in_dim %139, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %142 = stablehlo.add %140, %141 : tensor<1x64x128x128xf32>
    %143 = stablehlo.convert %142 : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xbf16>
    %144 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x64x128x128xbf16>
    %145 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xbf16>) -> tensor<1x64x128x128xbf16>
    %146 = stablehlo.maximum %144, %145 : tensor<1x64x128x128xbf16>
    %147 = stablehlo.minimum %144, %145 : tensor<1x64x128x128xbf16>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xbf16>) -> tensor<1x64x128x128xbf16>
    %149 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x64x128x128xbf16>
    %150 = stablehlo.multiply %148, %149 : tensor<1x64x128x128xbf16>
    %151 = stablehlo.add %146, %150 : tensor<1x64x128x128xbf16>
    %152 = stablehlo.convolution(%151, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x128x128xbf16>, tensor<128x64x3x3xbf16>) -> tensor<1x128x128x128xbf16>
    %153 = stablehlo.convert %152 : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %155 = stablehlo.broadcast_in_dim %arg103, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %156 = stablehlo.subtract %154, %155 : tensor<1x128x128x128xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %158 = stablehlo.broadcast_in_dim %arg104, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %159 = stablehlo.multiply %157, %158 : tensor<1x128x128x128xf32>
    %160 = stablehlo.convert %arg105 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %161 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %162 = stablehlo.broadcast_in_dim %160, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %163 = stablehlo.multiply %161, %162 : tensor<1x128x128x128xf32>
    %164 = stablehlo.convert %arg106 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %165 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %166 = stablehlo.broadcast_in_dim %164, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %167 = stablehlo.add %165, %166 : tensor<1x128x128x128xf32>
    %168 = stablehlo.convert %167 : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xbf16>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %170 = stablehlo.maximum %119, %169 : tensor<1x128x128x128xbf16>
    %171 = stablehlo.minimum %119, %169 : tensor<1x128x128x128xbf16>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %173 = stablehlo.multiply %172, %124 : tensor<1x128x128x128xbf16>
    %174 = stablehlo.add %170, %173 : tensor<1x128x128x128xbf16>
    %175 = stablehlo.add %174, %126 : tensor<1x128x128x128xbf16>
    %176 = stablehlo.convolution(%175, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x128x128xbf16>, tensor<64x128x1x1xbf16>) -> tensor<1x64x128x128xbf16>
    %177 = stablehlo.convert %176 : (tensor<1x64x128x128xbf16>) -> tensor<1x64x128x128xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %179 = stablehlo.broadcast_in_dim %arg107, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %180 = stablehlo.subtract %178, %179 : tensor<1x64x128x128xf32>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %182 = stablehlo.broadcast_in_dim %arg108, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %183 = stablehlo.multiply %181, %182 : tensor<1x64x128x128xf32>
    %184 = stablehlo.convert %arg109 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %185 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %186 = stablehlo.broadcast_in_dim %184, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %187 = stablehlo.multiply %185, %186 : tensor<1x64x128x128xf32>
    %188 = stablehlo.convert %arg110 : (tensor<64x1x1xbf16>) -> tensor<64x1x1xf32>
    %189 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %190 = stablehlo.broadcast_in_dim %188, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x128x128xf32>
    %191 = stablehlo.add %189, %190 : tensor<1x64x128x128xf32>
    %192 = stablehlo.convert %191 : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xbf16>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xbf16>) -> tensor<1x64x128x128xbf16>
    %194 = stablehlo.maximum %144, %193 : tensor<1x64x128x128xbf16>
    %195 = stablehlo.minimum %144, %193 : tensor<1x64x128x128xbf16>
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2, 3] : (tensor<1x64x128x128xbf16>) -> tensor<1x64x128x128xbf16>
    %197 = stablehlo.multiply %196, %149 : tensor<1x64x128x128xbf16>
    %198 = stablehlo.add %194, %197 : tensor<1x64x128x128xbf16>
    %199 = stablehlo.convolution(%198, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x128x128xbf16>, tensor<128x64x3x3xbf16>) -> tensor<1x128x128x128xbf16>
    %200 = stablehlo.convert %199 : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xf32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %202 = stablehlo.broadcast_in_dim %arg111, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %203 = stablehlo.subtract %201, %202 : tensor<1x128x128x128xf32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %205 = stablehlo.broadcast_in_dim %arg112, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %206 = stablehlo.multiply %204, %205 : tensor<1x128x128x128xf32>
    %207 = stablehlo.convert %arg113 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %208 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %209 = stablehlo.broadcast_in_dim %207, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %210 = stablehlo.multiply %208, %209 : tensor<1x128x128x128xf32>
    %211 = stablehlo.convert %arg114 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %212 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %213 = stablehlo.broadcast_in_dim %211, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x128x128xf32>
    %214 = stablehlo.add %212, %213 : tensor<1x128x128x128xf32>
    %215 = stablehlo.convert %214 : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xbf16>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %217 = stablehlo.maximum %119, %216 : tensor<1x128x128x128xbf16>
    %218 = stablehlo.minimum %119, %216 : tensor<1x128x128x128xbf16>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %220 = stablehlo.multiply %219, %124 : tensor<1x128x128x128xbf16>
    %221 = stablehlo.add %217, %220 : tensor<1x128x128x128xbf16>
    %222 = stablehlo.add %221, %175 : tensor<1x128x128x128xbf16>
    %223 = stablehlo.convolution(%222, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x128x128xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %224 = stablehlo.convert %223 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %226 = stablehlo.broadcast_in_dim %arg115, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %227 = stablehlo.subtract %225, %226 : tensor<1x256x64x64xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %229 = stablehlo.broadcast_in_dim %arg116, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %230 = stablehlo.multiply %228, %229 : tensor<1x256x64x64xf32>
    %231 = stablehlo.convert %arg117 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %232 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %233 = stablehlo.broadcast_in_dim %231, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %234 = stablehlo.multiply %232, %233 : tensor<1x256x64x64xf32>
    %235 = stablehlo.convert %arg118 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %236 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %237 = stablehlo.broadcast_in_dim %235, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %238 = stablehlo.add %236, %237 : tensor<1x256x64x64xf32>
    %239 = stablehlo.convert %238 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %240 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x256x64x64xbf16>
    %241 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %242 = stablehlo.maximum %240, %241 : tensor<1x256x64x64xbf16>
    %243 = stablehlo.minimum %240, %241 : tensor<1x256x64x64xbf16>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %245 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x256x64x64xbf16>
    %246 = stablehlo.multiply %244, %245 : tensor<1x256x64x64xbf16>
    %247 = stablehlo.add %242, %246 : tensor<1x256x64x64xbf16>
    %248 = stablehlo.convolution(%247, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %249 = stablehlo.convert %248 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %251 = stablehlo.broadcast_in_dim %arg119, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %252 = stablehlo.subtract %250, %251 : tensor<1x128x64x64xf32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %254 = stablehlo.broadcast_in_dim %arg120, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %255 = stablehlo.multiply %253, %254 : tensor<1x128x64x64xf32>
    %256 = stablehlo.convert %arg121 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %257 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %258 = stablehlo.broadcast_in_dim %256, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %259 = stablehlo.multiply %257, %258 : tensor<1x128x64x64xf32>
    %260 = stablehlo.convert %arg122 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %261 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %262 = stablehlo.broadcast_in_dim %260, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %263 = stablehlo.add %261, %262 : tensor<1x128x64x64xf32>
    %264 = stablehlo.convert %263 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %265 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x128x64x64xbf16>
    %266 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %267 = stablehlo.maximum %265, %266 : tensor<1x128x64x64xbf16>
    %268 = stablehlo.minimum %265, %266 : tensor<1x128x64x64xbf16>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %270 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x128x64x64xbf16>
    %271 = stablehlo.multiply %269, %270 : tensor<1x128x64x64xbf16>
    %272 = stablehlo.add %267, %271 : tensor<1x128x64x64xbf16>
    %273 = stablehlo.convolution(%272, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %274 = stablehlo.convert %273 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %276 = stablehlo.broadcast_in_dim %arg123, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %277 = stablehlo.subtract %275, %276 : tensor<1x256x64x64xf32>
    %278 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %279 = stablehlo.broadcast_in_dim %arg124, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %280 = stablehlo.multiply %278, %279 : tensor<1x256x64x64xf32>
    %281 = stablehlo.convert %arg125 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %282 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %283 = stablehlo.broadcast_in_dim %281, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %284 = stablehlo.multiply %282, %283 : tensor<1x256x64x64xf32>
    %285 = stablehlo.convert %arg126 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %286 = stablehlo.broadcast_in_dim %284, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %287 = stablehlo.broadcast_in_dim %285, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %288 = stablehlo.add %286, %287 : tensor<1x256x64x64xf32>
    %289 = stablehlo.convert %288 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %291 = stablehlo.maximum %240, %290 : tensor<1x256x64x64xbf16>
    %292 = stablehlo.minimum %240, %290 : tensor<1x256x64x64xbf16>
    %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %294 = stablehlo.multiply %293, %245 : tensor<1x256x64x64xbf16>
    %295 = stablehlo.add %291, %294 : tensor<1x256x64x64xbf16>
    %296 = stablehlo.add %295, %247 : tensor<1x256x64x64xbf16>
    %297 = stablehlo.convolution(%296, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %298 = stablehlo.convert %297 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %299 = stablehlo.broadcast_in_dim %298, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %300 = stablehlo.broadcast_in_dim %arg127, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %301 = stablehlo.subtract %299, %300 : tensor<1x128x64x64xf32>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %303 = stablehlo.broadcast_in_dim %arg128, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x128x64x64xf32>
    %305 = stablehlo.convert %arg129 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %306 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %307 = stablehlo.broadcast_in_dim %305, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %308 = stablehlo.multiply %306, %307 : tensor<1x128x64x64xf32>
    %309 = stablehlo.convert %arg130 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %310 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %311 = stablehlo.broadcast_in_dim %309, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %312 = stablehlo.add %310, %311 : tensor<1x128x64x64xf32>
    %313 = stablehlo.convert %312 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %315 = stablehlo.maximum %265, %314 : tensor<1x128x64x64xbf16>
    %316 = stablehlo.minimum %265, %314 : tensor<1x128x64x64xbf16>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %318 = stablehlo.multiply %317, %270 : tensor<1x128x64x64xbf16>
    %319 = stablehlo.add %315, %318 : tensor<1x128x64x64xbf16>
    %320 = stablehlo.convolution(%319, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %321 = stablehlo.convert %320 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %323 = stablehlo.broadcast_in_dim %arg131, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %324 = stablehlo.subtract %322, %323 : tensor<1x256x64x64xf32>
    %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %326 = stablehlo.broadcast_in_dim %arg132, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %327 = stablehlo.multiply %325, %326 : tensor<1x256x64x64xf32>
    %328 = stablehlo.convert %arg133 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %329 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %330 = stablehlo.broadcast_in_dim %328, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %331 = stablehlo.multiply %329, %330 : tensor<1x256x64x64xf32>
    %332 = stablehlo.convert %arg134 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %333 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %334 = stablehlo.broadcast_in_dim %332, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %335 = stablehlo.add %333, %334 : tensor<1x256x64x64xf32>
    %336 = stablehlo.convert %335 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %338 = stablehlo.maximum %240, %337 : tensor<1x256x64x64xbf16>
    %339 = stablehlo.minimum %240, %337 : tensor<1x256x64x64xbf16>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %341 = stablehlo.multiply %340, %245 : tensor<1x256x64x64xbf16>
    %342 = stablehlo.add %338, %341 : tensor<1x256x64x64xbf16>
    %343 = stablehlo.add %342, %296 : tensor<1x256x64x64xbf16>
    %344 = stablehlo.convolution(%343, %arg15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %345 = stablehlo.convert %344 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %347 = stablehlo.broadcast_in_dim %arg135, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %348 = stablehlo.subtract %346, %347 : tensor<1x128x64x64xf32>
    %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %350 = stablehlo.broadcast_in_dim %arg136, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %351 = stablehlo.multiply %349, %350 : tensor<1x128x64x64xf32>
    %352 = stablehlo.convert %arg137 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %353 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %354 = stablehlo.broadcast_in_dim %352, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %355 = stablehlo.multiply %353, %354 : tensor<1x128x64x64xf32>
    %356 = stablehlo.convert %arg138 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %357 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %358 = stablehlo.broadcast_in_dim %356, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %359 = stablehlo.add %357, %358 : tensor<1x128x64x64xf32>
    %360 = stablehlo.convert %359 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %362 = stablehlo.maximum %265, %361 : tensor<1x128x64x64xbf16>
    %363 = stablehlo.minimum %265, %361 : tensor<1x128x64x64xbf16>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %365 = stablehlo.multiply %364, %270 : tensor<1x128x64x64xbf16>
    %366 = stablehlo.add %362, %365 : tensor<1x128x64x64xbf16>
    %367 = stablehlo.convolution(%366, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %368 = stablehlo.convert %367 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %370 = stablehlo.broadcast_in_dim %arg139, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %371 = stablehlo.subtract %369, %370 : tensor<1x256x64x64xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %373 = stablehlo.broadcast_in_dim %arg140, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %374 = stablehlo.multiply %372, %373 : tensor<1x256x64x64xf32>
    %375 = stablehlo.convert %arg141 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %376 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %377 = stablehlo.broadcast_in_dim %375, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %378 = stablehlo.multiply %376, %377 : tensor<1x256x64x64xf32>
    %379 = stablehlo.convert %arg142 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %380 = stablehlo.broadcast_in_dim %378, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %381 = stablehlo.broadcast_in_dim %379, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %382 = stablehlo.add %380, %381 : tensor<1x256x64x64xf32>
    %383 = stablehlo.convert %382 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %385 = stablehlo.maximum %240, %384 : tensor<1x256x64x64xbf16>
    %386 = stablehlo.minimum %240, %384 : tensor<1x256x64x64xbf16>
    %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %388 = stablehlo.multiply %387, %245 : tensor<1x256x64x64xbf16>
    %389 = stablehlo.add %385, %388 : tensor<1x256x64x64xbf16>
    %390 = stablehlo.add %389, %343 : tensor<1x256x64x64xbf16>
    %391 = stablehlo.convolution(%390, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %392 = stablehlo.convert %391 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %394 = stablehlo.broadcast_in_dim %arg143, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %395 = stablehlo.subtract %393, %394 : tensor<1x128x64x64xf32>
    %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %397 = stablehlo.broadcast_in_dim %arg144, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %398 = stablehlo.multiply %396, %397 : tensor<1x128x64x64xf32>
    %399 = stablehlo.convert %arg145 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %400 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %401 = stablehlo.broadcast_in_dim %399, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %402 = stablehlo.multiply %400, %401 : tensor<1x128x64x64xf32>
    %403 = stablehlo.convert %arg146 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %404 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %405 = stablehlo.broadcast_in_dim %403, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %406 = stablehlo.add %404, %405 : tensor<1x128x64x64xf32>
    %407 = stablehlo.convert %406 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %408 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %409 = stablehlo.maximum %265, %408 : tensor<1x128x64x64xbf16>
    %410 = stablehlo.minimum %265, %408 : tensor<1x128x64x64xbf16>
    %411 = stablehlo.broadcast_in_dim %410, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %412 = stablehlo.multiply %411, %270 : tensor<1x128x64x64xbf16>
    %413 = stablehlo.add %409, %412 : tensor<1x128x64x64xbf16>
    %414 = stablehlo.convolution(%413, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %415 = stablehlo.convert %414 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %417 = stablehlo.broadcast_in_dim %arg147, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %418 = stablehlo.subtract %416, %417 : tensor<1x256x64x64xf32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %420 = stablehlo.broadcast_in_dim %arg148, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %421 = stablehlo.multiply %419, %420 : tensor<1x256x64x64xf32>
    %422 = stablehlo.convert %arg149 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %423 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %424 = stablehlo.broadcast_in_dim %422, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %425 = stablehlo.multiply %423, %424 : tensor<1x256x64x64xf32>
    %426 = stablehlo.convert %arg150 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %427 = stablehlo.broadcast_in_dim %425, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %428 = stablehlo.broadcast_in_dim %426, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %429 = stablehlo.add %427, %428 : tensor<1x256x64x64xf32>
    %430 = stablehlo.convert %429 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %432 = stablehlo.maximum %240, %431 : tensor<1x256x64x64xbf16>
    %433 = stablehlo.minimum %240, %431 : tensor<1x256x64x64xbf16>
    %434 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %435 = stablehlo.multiply %434, %245 : tensor<1x256x64x64xbf16>
    %436 = stablehlo.add %432, %435 : tensor<1x256x64x64xbf16>
    %437 = stablehlo.add %436, %390 : tensor<1x256x64x64xbf16>
    %438 = stablehlo.convolution(%437, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %439 = stablehlo.convert %438 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %440 = stablehlo.broadcast_in_dim %439, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %441 = stablehlo.broadcast_in_dim %arg151, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %442 = stablehlo.subtract %440, %441 : tensor<1x128x64x64xf32>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %444 = stablehlo.broadcast_in_dim %arg152, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %445 = stablehlo.multiply %443, %444 : tensor<1x128x64x64xf32>
    %446 = stablehlo.convert %arg153 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %447 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %448 = stablehlo.broadcast_in_dim %446, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %449 = stablehlo.multiply %447, %448 : tensor<1x128x64x64xf32>
    %450 = stablehlo.convert %arg154 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %451 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %452 = stablehlo.broadcast_in_dim %450, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %453 = stablehlo.add %451, %452 : tensor<1x128x64x64xf32>
    %454 = stablehlo.convert %453 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %455 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %456 = stablehlo.maximum %265, %455 : tensor<1x128x64x64xbf16>
    %457 = stablehlo.minimum %265, %455 : tensor<1x128x64x64xbf16>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %459 = stablehlo.multiply %458, %270 : tensor<1x128x64x64xbf16>
    %460 = stablehlo.add %456, %459 : tensor<1x128x64x64xbf16>
    %461 = stablehlo.convolution(%460, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %462 = stablehlo.convert %461 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %464 = stablehlo.broadcast_in_dim %arg155, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %465 = stablehlo.subtract %463, %464 : tensor<1x256x64x64xf32>
    %466 = stablehlo.broadcast_in_dim %465, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %467 = stablehlo.broadcast_in_dim %arg156, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %468 = stablehlo.multiply %466, %467 : tensor<1x256x64x64xf32>
    %469 = stablehlo.convert %arg157 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %470 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %471 = stablehlo.broadcast_in_dim %469, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %472 = stablehlo.multiply %470, %471 : tensor<1x256x64x64xf32>
    %473 = stablehlo.convert %arg158 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %474 = stablehlo.broadcast_in_dim %472, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %475 = stablehlo.broadcast_in_dim %473, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %476 = stablehlo.add %474, %475 : tensor<1x256x64x64xf32>
    %477 = stablehlo.convert %476 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %479 = stablehlo.maximum %240, %478 : tensor<1x256x64x64xbf16>
    %480 = stablehlo.minimum %240, %478 : tensor<1x256x64x64xbf16>
    %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %482 = stablehlo.multiply %481, %245 : tensor<1x256x64x64xbf16>
    %483 = stablehlo.add %479, %482 : tensor<1x256x64x64xbf16>
    %484 = stablehlo.add %483, %437 : tensor<1x256x64x64xbf16>
    %485 = stablehlo.convolution(%484, %arg21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %486 = stablehlo.convert %485 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %488 = stablehlo.broadcast_in_dim %arg159, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %489 = stablehlo.subtract %487, %488 : tensor<1x128x64x64xf32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %491 = stablehlo.broadcast_in_dim %arg160, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %492 = stablehlo.multiply %490, %491 : tensor<1x128x64x64xf32>
    %493 = stablehlo.convert %arg161 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %494 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %495 = stablehlo.broadcast_in_dim %493, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %496 = stablehlo.multiply %494, %495 : tensor<1x128x64x64xf32>
    %497 = stablehlo.convert %arg162 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %498 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %499 = stablehlo.broadcast_in_dim %497, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %500 = stablehlo.add %498, %499 : tensor<1x128x64x64xf32>
    %501 = stablehlo.convert %500 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %502 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %503 = stablehlo.maximum %265, %502 : tensor<1x128x64x64xbf16>
    %504 = stablehlo.minimum %265, %502 : tensor<1x128x64x64xbf16>
    %505 = stablehlo.broadcast_in_dim %504, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %506 = stablehlo.multiply %505, %270 : tensor<1x128x64x64xbf16>
    %507 = stablehlo.add %503, %506 : tensor<1x128x64x64xbf16>
    %508 = stablehlo.convolution(%507, %arg22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %509 = stablehlo.convert %508 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %510 = stablehlo.broadcast_in_dim %509, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %511 = stablehlo.broadcast_in_dim %arg163, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %512 = stablehlo.subtract %510, %511 : tensor<1x256x64x64xf32>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %514 = stablehlo.broadcast_in_dim %arg164, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %515 = stablehlo.multiply %513, %514 : tensor<1x256x64x64xf32>
    %516 = stablehlo.convert %arg165 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %517 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %518 = stablehlo.broadcast_in_dim %516, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %519 = stablehlo.multiply %517, %518 : tensor<1x256x64x64xf32>
    %520 = stablehlo.convert %arg166 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %521 = stablehlo.broadcast_in_dim %519, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %522 = stablehlo.broadcast_in_dim %520, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %523 = stablehlo.add %521, %522 : tensor<1x256x64x64xf32>
    %524 = stablehlo.convert %523 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %525 = stablehlo.broadcast_in_dim %524, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %526 = stablehlo.maximum %240, %525 : tensor<1x256x64x64xbf16>
    %527 = stablehlo.minimum %240, %525 : tensor<1x256x64x64xbf16>
    %528 = stablehlo.broadcast_in_dim %527, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %529 = stablehlo.multiply %528, %245 : tensor<1x256x64x64xbf16>
    %530 = stablehlo.add %526, %529 : tensor<1x256x64x64xbf16>
    %531 = stablehlo.add %530, %484 : tensor<1x256x64x64xbf16>
    %532 = stablehlo.convolution(%531, %arg23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %533 = stablehlo.convert %532 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %535 = stablehlo.broadcast_in_dim %arg167, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %536 = stablehlo.subtract %534, %535 : tensor<1x128x64x64xf32>
    %537 = stablehlo.broadcast_in_dim %536, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %538 = stablehlo.broadcast_in_dim %arg168, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %539 = stablehlo.multiply %537, %538 : tensor<1x128x64x64xf32>
    %540 = stablehlo.convert %arg169 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %541 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %542 = stablehlo.broadcast_in_dim %540, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %543 = stablehlo.multiply %541, %542 : tensor<1x128x64x64xf32>
    %544 = stablehlo.convert %arg170 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %545 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %546 = stablehlo.broadcast_in_dim %544, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %547 = stablehlo.add %545, %546 : tensor<1x128x64x64xf32>
    %548 = stablehlo.convert %547 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %549 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %550 = stablehlo.maximum %265, %549 : tensor<1x128x64x64xbf16>
    %551 = stablehlo.minimum %265, %549 : tensor<1x128x64x64xbf16>
    %552 = stablehlo.broadcast_in_dim %551, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %553 = stablehlo.multiply %552, %270 : tensor<1x128x64x64xbf16>
    %554 = stablehlo.add %550, %553 : tensor<1x128x64x64xbf16>
    %555 = stablehlo.convolution(%554, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %556 = stablehlo.convert %555 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %558 = stablehlo.broadcast_in_dim %arg171, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %559 = stablehlo.subtract %557, %558 : tensor<1x256x64x64xf32>
    %560 = stablehlo.broadcast_in_dim %559, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %561 = stablehlo.broadcast_in_dim %arg172, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %562 = stablehlo.multiply %560, %561 : tensor<1x256x64x64xf32>
    %563 = stablehlo.convert %arg173 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %564 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %565 = stablehlo.broadcast_in_dim %563, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %566 = stablehlo.multiply %564, %565 : tensor<1x256x64x64xf32>
    %567 = stablehlo.convert %arg174 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %568 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %569 = stablehlo.broadcast_in_dim %567, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %570 = stablehlo.add %568, %569 : tensor<1x256x64x64xf32>
    %571 = stablehlo.convert %570 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %572 = stablehlo.broadcast_in_dim %571, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %573 = stablehlo.maximum %240, %572 : tensor<1x256x64x64xbf16>
    %574 = stablehlo.minimum %240, %572 : tensor<1x256x64x64xbf16>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %576 = stablehlo.multiply %575, %245 : tensor<1x256x64x64xbf16>
    %577 = stablehlo.add %573, %576 : tensor<1x256x64x64xbf16>
    %578 = stablehlo.add %577, %531 : tensor<1x256x64x64xbf16>
    %579 = stablehlo.convolution(%578, %arg25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %580 = stablehlo.convert %579 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %582 = stablehlo.broadcast_in_dim %arg175, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %583 = stablehlo.subtract %581, %582 : tensor<1x128x64x64xf32>
    %584 = stablehlo.broadcast_in_dim %583, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %585 = stablehlo.broadcast_in_dim %arg176, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %586 = stablehlo.multiply %584, %585 : tensor<1x128x64x64xf32>
    %587 = stablehlo.convert %arg177 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %588 = stablehlo.broadcast_in_dim %586, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %589 = stablehlo.broadcast_in_dim %587, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %590 = stablehlo.multiply %588, %589 : tensor<1x128x64x64xf32>
    %591 = stablehlo.convert %arg178 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %592 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %593 = stablehlo.broadcast_in_dim %591, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %594 = stablehlo.add %592, %593 : tensor<1x128x64x64xf32>
    %595 = stablehlo.convert %594 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %596 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %597 = stablehlo.maximum %265, %596 : tensor<1x128x64x64xbf16>
    %598 = stablehlo.minimum %265, %596 : tensor<1x128x64x64xbf16>
    %599 = stablehlo.broadcast_in_dim %598, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %600 = stablehlo.multiply %599, %270 : tensor<1x128x64x64xbf16>
    %601 = stablehlo.add %597, %600 : tensor<1x128x64x64xbf16>
    %602 = stablehlo.convolution(%601, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %603 = stablehlo.convert %602 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %604 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %605 = stablehlo.broadcast_in_dim %arg179, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %606 = stablehlo.subtract %604, %605 : tensor<1x256x64x64xf32>
    %607 = stablehlo.broadcast_in_dim %606, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %608 = stablehlo.broadcast_in_dim %arg180, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %609 = stablehlo.multiply %607, %608 : tensor<1x256x64x64xf32>
    %610 = stablehlo.convert %arg181 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %611 = stablehlo.broadcast_in_dim %609, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %612 = stablehlo.broadcast_in_dim %610, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %613 = stablehlo.multiply %611, %612 : tensor<1x256x64x64xf32>
    %614 = stablehlo.convert %arg182 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %615 = stablehlo.broadcast_in_dim %613, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %616 = stablehlo.broadcast_in_dim %614, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %617 = stablehlo.add %615, %616 : tensor<1x256x64x64xf32>
    %618 = stablehlo.convert %617 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %620 = stablehlo.maximum %240, %619 : tensor<1x256x64x64xbf16>
    %621 = stablehlo.minimum %240, %619 : tensor<1x256x64x64xbf16>
    %622 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %623 = stablehlo.multiply %622, %245 : tensor<1x256x64x64xbf16>
    %624 = stablehlo.add %620, %623 : tensor<1x256x64x64xbf16>
    %625 = stablehlo.add %624, %578 : tensor<1x256x64x64xbf16>
    %626 = stablehlo.convolution(%625, %arg27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %627 = stablehlo.convert %626 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %628 = stablehlo.broadcast_in_dim %627, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %629 = stablehlo.broadcast_in_dim %arg183, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %630 = stablehlo.subtract %628, %629 : tensor<1x512x32x32xf32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %632 = stablehlo.broadcast_in_dim %arg184, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %633 = stablehlo.multiply %631, %632 : tensor<1x512x32x32xf32>
    %634 = stablehlo.convert %arg185 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %635 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %636 = stablehlo.broadcast_in_dim %634, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %637 = stablehlo.multiply %635, %636 : tensor<1x512x32x32xf32>
    %638 = stablehlo.convert %arg186 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %639 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %640 = stablehlo.broadcast_in_dim %638, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %641 = stablehlo.add %639, %640 : tensor<1x512x32x32xf32>
    %642 = stablehlo.convert %641 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %643 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x512x32x32xbf16>
    %644 = stablehlo.broadcast_in_dim %642, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %645 = stablehlo.maximum %643, %644 : tensor<1x512x32x32xbf16>
    %646 = stablehlo.minimum %643, %644 : tensor<1x512x32x32xbf16>
    %647 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %648 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x512x32x32xbf16>
    %649 = stablehlo.multiply %647, %648 : tensor<1x512x32x32xbf16>
    %650 = stablehlo.add %645, %649 : tensor<1x512x32x32xbf16>
    %651 = stablehlo.convolution(%650, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %652 = stablehlo.convert %651 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %654 = stablehlo.broadcast_in_dim %arg187, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %655 = stablehlo.subtract %653, %654 : tensor<1x256x32x32xf32>
    %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %657 = stablehlo.broadcast_in_dim %arg188, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %658 = stablehlo.multiply %656, %657 : tensor<1x256x32x32xf32>
    %659 = stablehlo.convert %arg189 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %660 = stablehlo.broadcast_in_dim %658, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %661 = stablehlo.broadcast_in_dim %659, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %662 = stablehlo.multiply %660, %661 : tensor<1x256x32x32xf32>
    %663 = stablehlo.convert %arg190 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %664 = stablehlo.broadcast_in_dim %662, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %665 = stablehlo.broadcast_in_dim %663, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %666 = stablehlo.add %664, %665 : tensor<1x256x32x32xf32>
    %667 = stablehlo.convert %666 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %668 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x256x32x32xbf16>
    %669 = stablehlo.broadcast_in_dim %667, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %670 = stablehlo.maximum %668, %669 : tensor<1x256x32x32xbf16>
    %671 = stablehlo.minimum %668, %669 : tensor<1x256x32x32xbf16>
    %672 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %673 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x256x32x32xbf16>
    %674 = stablehlo.multiply %672, %673 : tensor<1x256x32x32xbf16>
    %675 = stablehlo.add %670, %674 : tensor<1x256x32x32xbf16>
    %676 = stablehlo.convolution(%675, %arg29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %677 = stablehlo.convert %676 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %679 = stablehlo.broadcast_in_dim %arg191, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %680 = stablehlo.subtract %678, %679 : tensor<1x512x32x32xf32>
    %681 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %682 = stablehlo.broadcast_in_dim %arg192, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %683 = stablehlo.multiply %681, %682 : tensor<1x512x32x32xf32>
    %684 = stablehlo.convert %arg193 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %685 = stablehlo.broadcast_in_dim %683, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %686 = stablehlo.broadcast_in_dim %684, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %687 = stablehlo.multiply %685, %686 : tensor<1x512x32x32xf32>
    %688 = stablehlo.convert %arg194 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %689 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %690 = stablehlo.broadcast_in_dim %688, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %691 = stablehlo.add %689, %690 : tensor<1x512x32x32xf32>
    %692 = stablehlo.convert %691 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %694 = stablehlo.maximum %643, %693 : tensor<1x512x32x32xbf16>
    %695 = stablehlo.minimum %643, %693 : tensor<1x512x32x32xbf16>
    %696 = stablehlo.broadcast_in_dim %695, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %697 = stablehlo.multiply %696, %648 : tensor<1x512x32x32xbf16>
    %698 = stablehlo.add %694, %697 : tensor<1x512x32x32xbf16>
    %699 = stablehlo.add %698, %650 : tensor<1x512x32x32xbf16>
    %700 = stablehlo.convolution(%699, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %701 = stablehlo.convert %700 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %702 = stablehlo.broadcast_in_dim %701, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %703 = stablehlo.broadcast_in_dim %arg195, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %704 = stablehlo.subtract %702, %703 : tensor<1x256x32x32xf32>
    %705 = stablehlo.broadcast_in_dim %704, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %706 = stablehlo.broadcast_in_dim %arg196, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %707 = stablehlo.multiply %705, %706 : tensor<1x256x32x32xf32>
    %708 = stablehlo.convert %arg197 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %709 = stablehlo.broadcast_in_dim %707, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %710 = stablehlo.broadcast_in_dim %708, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %711 = stablehlo.multiply %709, %710 : tensor<1x256x32x32xf32>
    %712 = stablehlo.convert %arg198 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %713 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %714 = stablehlo.broadcast_in_dim %712, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %715 = stablehlo.add %713, %714 : tensor<1x256x32x32xf32>
    %716 = stablehlo.convert %715 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %718 = stablehlo.maximum %668, %717 : tensor<1x256x32x32xbf16>
    %719 = stablehlo.minimum %668, %717 : tensor<1x256x32x32xbf16>
    %720 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %721 = stablehlo.multiply %720, %673 : tensor<1x256x32x32xbf16>
    %722 = stablehlo.add %718, %721 : tensor<1x256x32x32xbf16>
    %723 = stablehlo.convolution(%722, %arg31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %724 = stablehlo.convert %723 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %725 = stablehlo.broadcast_in_dim %724, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %726 = stablehlo.broadcast_in_dim %arg199, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %727 = stablehlo.subtract %725, %726 : tensor<1x512x32x32xf32>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %729 = stablehlo.broadcast_in_dim %arg200, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %730 = stablehlo.multiply %728, %729 : tensor<1x512x32x32xf32>
    %731 = stablehlo.convert %arg201 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %732 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %733 = stablehlo.broadcast_in_dim %731, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %734 = stablehlo.multiply %732, %733 : tensor<1x512x32x32xf32>
    %735 = stablehlo.convert %arg202 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %736 = stablehlo.broadcast_in_dim %734, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %737 = stablehlo.broadcast_in_dim %735, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %738 = stablehlo.add %736, %737 : tensor<1x512x32x32xf32>
    %739 = stablehlo.convert %738 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %741 = stablehlo.maximum %643, %740 : tensor<1x512x32x32xbf16>
    %742 = stablehlo.minimum %643, %740 : tensor<1x512x32x32xbf16>
    %743 = stablehlo.broadcast_in_dim %742, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %744 = stablehlo.multiply %743, %648 : tensor<1x512x32x32xbf16>
    %745 = stablehlo.add %741, %744 : tensor<1x512x32x32xbf16>
    %746 = stablehlo.add %745, %699 : tensor<1x512x32x32xbf16>
    %747 = stablehlo.convolution(%746, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %748 = stablehlo.convert %747 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %749 = stablehlo.broadcast_in_dim %748, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %750 = stablehlo.broadcast_in_dim %arg203, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %751 = stablehlo.subtract %749, %750 : tensor<1x256x32x32xf32>
    %752 = stablehlo.broadcast_in_dim %751, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %753 = stablehlo.broadcast_in_dim %arg204, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %754 = stablehlo.multiply %752, %753 : tensor<1x256x32x32xf32>
    %755 = stablehlo.convert %arg205 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %756 = stablehlo.broadcast_in_dim %754, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %757 = stablehlo.broadcast_in_dim %755, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %758 = stablehlo.multiply %756, %757 : tensor<1x256x32x32xf32>
    %759 = stablehlo.convert %arg206 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %760 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %761 = stablehlo.broadcast_in_dim %759, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %762 = stablehlo.add %760, %761 : tensor<1x256x32x32xf32>
    %763 = stablehlo.convert %762 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %764 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %765 = stablehlo.maximum %668, %764 : tensor<1x256x32x32xbf16>
    %766 = stablehlo.minimum %668, %764 : tensor<1x256x32x32xbf16>
    %767 = stablehlo.broadcast_in_dim %766, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %768 = stablehlo.multiply %767, %673 : tensor<1x256x32x32xbf16>
    %769 = stablehlo.add %765, %768 : tensor<1x256x32x32xbf16>
    %770 = stablehlo.convolution(%769, %arg33) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %771 = stablehlo.convert %770 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %772 = stablehlo.broadcast_in_dim %771, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %773 = stablehlo.broadcast_in_dim %arg207, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %774 = stablehlo.subtract %772, %773 : tensor<1x512x32x32xf32>
    %775 = stablehlo.broadcast_in_dim %774, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %776 = stablehlo.broadcast_in_dim %arg208, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %777 = stablehlo.multiply %775, %776 : tensor<1x512x32x32xf32>
    %778 = stablehlo.convert %arg209 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %779 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %780 = stablehlo.broadcast_in_dim %778, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %781 = stablehlo.multiply %779, %780 : tensor<1x512x32x32xf32>
    %782 = stablehlo.convert %arg210 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %783 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %784 = stablehlo.broadcast_in_dim %782, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %785 = stablehlo.add %783, %784 : tensor<1x512x32x32xf32>
    %786 = stablehlo.convert %785 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %787 = stablehlo.broadcast_in_dim %786, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %788 = stablehlo.maximum %643, %787 : tensor<1x512x32x32xbf16>
    %789 = stablehlo.minimum %643, %787 : tensor<1x512x32x32xbf16>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %791 = stablehlo.multiply %790, %648 : tensor<1x512x32x32xbf16>
    %792 = stablehlo.add %788, %791 : tensor<1x512x32x32xbf16>
    %793 = stablehlo.add %792, %746 : tensor<1x512x32x32xbf16>
    %794 = stablehlo.convolution(%793, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %795 = stablehlo.convert %794 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %797 = stablehlo.broadcast_in_dim %arg211, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %798 = stablehlo.subtract %796, %797 : tensor<1x256x32x32xf32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %800 = stablehlo.broadcast_in_dim %arg212, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %801 = stablehlo.multiply %799, %800 : tensor<1x256x32x32xf32>
    %802 = stablehlo.convert %arg213 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %803 = stablehlo.broadcast_in_dim %801, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %804 = stablehlo.broadcast_in_dim %802, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %805 = stablehlo.multiply %803, %804 : tensor<1x256x32x32xf32>
    %806 = stablehlo.convert %arg214 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %807 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %808 = stablehlo.broadcast_in_dim %806, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %809 = stablehlo.add %807, %808 : tensor<1x256x32x32xf32>
    %810 = stablehlo.convert %809 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %812 = stablehlo.maximum %668, %811 : tensor<1x256x32x32xbf16>
    %813 = stablehlo.minimum %668, %811 : tensor<1x256x32x32xbf16>
    %814 = stablehlo.broadcast_in_dim %813, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %815 = stablehlo.multiply %814, %673 : tensor<1x256x32x32xbf16>
    %816 = stablehlo.add %812, %815 : tensor<1x256x32x32xbf16>
    %817 = stablehlo.convolution(%816, %arg35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %818 = stablehlo.convert %817 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %819 = stablehlo.broadcast_in_dim %818, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %820 = stablehlo.broadcast_in_dim %arg215, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %821 = stablehlo.subtract %819, %820 : tensor<1x512x32x32xf32>
    %822 = stablehlo.broadcast_in_dim %821, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %823 = stablehlo.broadcast_in_dim %arg216, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %824 = stablehlo.multiply %822, %823 : tensor<1x512x32x32xf32>
    %825 = stablehlo.convert %arg217 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %826 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %827 = stablehlo.broadcast_in_dim %825, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %828 = stablehlo.multiply %826, %827 : tensor<1x512x32x32xf32>
    %829 = stablehlo.convert %arg218 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %830 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %831 = stablehlo.broadcast_in_dim %829, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %832 = stablehlo.add %830, %831 : tensor<1x512x32x32xf32>
    %833 = stablehlo.convert %832 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %834 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %835 = stablehlo.maximum %643, %834 : tensor<1x512x32x32xbf16>
    %836 = stablehlo.minimum %643, %834 : tensor<1x512x32x32xbf16>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %838 = stablehlo.multiply %837, %648 : tensor<1x512x32x32xbf16>
    %839 = stablehlo.add %835, %838 : tensor<1x512x32x32xbf16>
    %840 = stablehlo.add %839, %793 : tensor<1x512x32x32xbf16>
    %841 = stablehlo.convolution(%840, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %842 = stablehlo.convert %841 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %844 = stablehlo.broadcast_in_dim %arg219, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %845 = stablehlo.subtract %843, %844 : tensor<1x256x32x32xf32>
    %846 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %847 = stablehlo.broadcast_in_dim %arg220, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %848 = stablehlo.multiply %846, %847 : tensor<1x256x32x32xf32>
    %849 = stablehlo.convert %arg221 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %850 = stablehlo.broadcast_in_dim %848, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %851 = stablehlo.broadcast_in_dim %849, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %852 = stablehlo.multiply %850, %851 : tensor<1x256x32x32xf32>
    %853 = stablehlo.convert %arg222 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %854 = stablehlo.broadcast_in_dim %852, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %855 = stablehlo.broadcast_in_dim %853, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %856 = stablehlo.add %854, %855 : tensor<1x256x32x32xf32>
    %857 = stablehlo.convert %856 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %859 = stablehlo.maximum %668, %858 : tensor<1x256x32x32xbf16>
    %860 = stablehlo.minimum %668, %858 : tensor<1x256x32x32xbf16>
    %861 = stablehlo.broadcast_in_dim %860, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %862 = stablehlo.multiply %861, %673 : tensor<1x256x32x32xbf16>
    %863 = stablehlo.add %859, %862 : tensor<1x256x32x32xbf16>
    %864 = stablehlo.convolution(%863, %arg37) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %865 = stablehlo.convert %864 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %867 = stablehlo.broadcast_in_dim %arg223, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %868 = stablehlo.subtract %866, %867 : tensor<1x512x32x32xf32>
    %869 = stablehlo.broadcast_in_dim %868, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %870 = stablehlo.broadcast_in_dim %arg224, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %871 = stablehlo.multiply %869, %870 : tensor<1x512x32x32xf32>
    %872 = stablehlo.convert %arg225 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %873 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %874 = stablehlo.broadcast_in_dim %872, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %875 = stablehlo.multiply %873, %874 : tensor<1x512x32x32xf32>
    %876 = stablehlo.convert %arg226 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %877 = stablehlo.broadcast_in_dim %875, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %878 = stablehlo.broadcast_in_dim %876, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %879 = stablehlo.add %877, %878 : tensor<1x512x32x32xf32>
    %880 = stablehlo.convert %879 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %882 = stablehlo.maximum %643, %881 : tensor<1x512x32x32xbf16>
    %883 = stablehlo.minimum %643, %881 : tensor<1x512x32x32xbf16>
    %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %885 = stablehlo.multiply %884, %648 : tensor<1x512x32x32xbf16>
    %886 = stablehlo.add %882, %885 : tensor<1x512x32x32xbf16>
    %887 = stablehlo.add %886, %840 : tensor<1x512x32x32xbf16>
    %888 = stablehlo.convolution(%887, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %889 = stablehlo.convert %888 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %890 = stablehlo.broadcast_in_dim %889, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %891 = stablehlo.broadcast_in_dim %arg227, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %892 = stablehlo.subtract %890, %891 : tensor<1x256x32x32xf32>
    %893 = stablehlo.broadcast_in_dim %892, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %894 = stablehlo.broadcast_in_dim %arg228, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %895 = stablehlo.multiply %893, %894 : tensor<1x256x32x32xf32>
    %896 = stablehlo.convert %arg229 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %897 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %898 = stablehlo.broadcast_in_dim %896, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %899 = stablehlo.multiply %897, %898 : tensor<1x256x32x32xf32>
    %900 = stablehlo.convert %arg230 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %901 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %902 = stablehlo.broadcast_in_dim %900, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %903 = stablehlo.add %901, %902 : tensor<1x256x32x32xf32>
    %904 = stablehlo.convert %903 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %906 = stablehlo.maximum %668, %905 : tensor<1x256x32x32xbf16>
    %907 = stablehlo.minimum %668, %905 : tensor<1x256x32x32xbf16>
    %908 = stablehlo.broadcast_in_dim %907, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %909 = stablehlo.multiply %908, %673 : tensor<1x256x32x32xbf16>
    %910 = stablehlo.add %906, %909 : tensor<1x256x32x32xbf16>
    %911 = stablehlo.convolution(%910, %arg39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %912 = stablehlo.convert %911 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %913 = stablehlo.broadcast_in_dim %912, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %914 = stablehlo.broadcast_in_dim %arg231, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %915 = stablehlo.subtract %913, %914 : tensor<1x512x32x32xf32>
    %916 = stablehlo.broadcast_in_dim %915, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %917 = stablehlo.broadcast_in_dim %arg232, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %918 = stablehlo.multiply %916, %917 : tensor<1x512x32x32xf32>
    %919 = stablehlo.convert %arg233 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %920 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %921 = stablehlo.broadcast_in_dim %919, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %922 = stablehlo.multiply %920, %921 : tensor<1x512x32x32xf32>
    %923 = stablehlo.convert %arg234 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %924 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %925 = stablehlo.broadcast_in_dim %923, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %926 = stablehlo.add %924, %925 : tensor<1x512x32x32xf32>
    %927 = stablehlo.convert %926 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %928 = stablehlo.broadcast_in_dim %927, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %929 = stablehlo.maximum %643, %928 : tensor<1x512x32x32xbf16>
    %930 = stablehlo.minimum %643, %928 : tensor<1x512x32x32xbf16>
    %931 = stablehlo.broadcast_in_dim %930, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %932 = stablehlo.multiply %931, %648 : tensor<1x512x32x32xbf16>
    %933 = stablehlo.add %929, %932 : tensor<1x512x32x32xbf16>
    %934 = stablehlo.add %933, %887 : tensor<1x512x32x32xbf16>
    %935 = stablehlo.convolution(%934, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %936 = stablehlo.convert %935 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %937 = stablehlo.broadcast_in_dim %936, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %938 = stablehlo.broadcast_in_dim %arg235, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %939 = stablehlo.subtract %937, %938 : tensor<1x256x32x32xf32>
    %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %941 = stablehlo.broadcast_in_dim %arg236, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %942 = stablehlo.multiply %940, %941 : tensor<1x256x32x32xf32>
    %943 = stablehlo.convert %arg237 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %944 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %945 = stablehlo.broadcast_in_dim %943, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %946 = stablehlo.multiply %944, %945 : tensor<1x256x32x32xf32>
    %947 = stablehlo.convert %arg238 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %948 = stablehlo.broadcast_in_dim %946, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %949 = stablehlo.broadcast_in_dim %947, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %950 = stablehlo.add %948, %949 : tensor<1x256x32x32xf32>
    %951 = stablehlo.convert %950 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %952 = stablehlo.broadcast_in_dim %951, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %953 = stablehlo.maximum %668, %952 : tensor<1x256x32x32xbf16>
    %954 = stablehlo.minimum %668, %952 : tensor<1x256x32x32xbf16>
    %955 = stablehlo.broadcast_in_dim %954, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %956 = stablehlo.multiply %955, %673 : tensor<1x256x32x32xbf16>
    %957 = stablehlo.add %953, %956 : tensor<1x256x32x32xbf16>
    %958 = stablehlo.convolution(%957, %arg41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %959 = stablehlo.convert %958 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %960 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %961 = stablehlo.broadcast_in_dim %arg239, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %962 = stablehlo.subtract %960, %961 : tensor<1x512x32x32xf32>
    %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %964 = stablehlo.broadcast_in_dim %arg240, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %965 = stablehlo.multiply %963, %964 : tensor<1x512x32x32xf32>
    %966 = stablehlo.convert %arg241 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %967 = stablehlo.broadcast_in_dim %965, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %968 = stablehlo.broadcast_in_dim %966, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %969 = stablehlo.multiply %967, %968 : tensor<1x512x32x32xf32>
    %970 = stablehlo.convert %arg242 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %971 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %972 = stablehlo.broadcast_in_dim %970, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %973 = stablehlo.add %971, %972 : tensor<1x512x32x32xf32>
    %974 = stablehlo.convert %973 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %975 = stablehlo.broadcast_in_dim %974, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %976 = stablehlo.maximum %643, %975 : tensor<1x512x32x32xbf16>
    %977 = stablehlo.minimum %643, %975 : tensor<1x512x32x32xbf16>
    %978 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %979 = stablehlo.multiply %978, %648 : tensor<1x512x32x32xbf16>
    %980 = stablehlo.add %976, %979 : tensor<1x512x32x32xbf16>
    %981 = stablehlo.add %980, %934 : tensor<1x512x32x32xbf16>
    %982 = stablehlo.convolution(%981, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %983 = stablehlo.convert %982 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %984 = stablehlo.broadcast_in_dim %983, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %985 = stablehlo.broadcast_in_dim %arg243, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %986 = stablehlo.subtract %984, %985 : tensor<1x256x32x32xf32>
    %987 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %988 = stablehlo.broadcast_in_dim %arg244, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %989 = stablehlo.multiply %987, %988 : tensor<1x256x32x32xf32>
    %990 = stablehlo.convert %arg245 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %991 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %992 = stablehlo.broadcast_in_dim %990, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %993 = stablehlo.multiply %991, %992 : tensor<1x256x32x32xf32>
    %994 = stablehlo.convert %arg246 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %995 = stablehlo.broadcast_in_dim %993, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %996 = stablehlo.broadcast_in_dim %994, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %997 = stablehlo.add %995, %996 : tensor<1x256x32x32xf32>
    %998 = stablehlo.convert %997 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1000 = stablehlo.maximum %668, %999 : tensor<1x256x32x32xbf16>
    %1001 = stablehlo.minimum %668, %999 : tensor<1x256x32x32xbf16>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1003 = stablehlo.multiply %1002, %673 : tensor<1x256x32x32xbf16>
    %1004 = stablehlo.add %1000, %1003 : tensor<1x256x32x32xbf16>
    %1005 = stablehlo.convolution(%1004, %arg43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %1006 = stablehlo.convert %1005 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %1007 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1008 = stablehlo.broadcast_in_dim %arg247, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1009 = stablehlo.subtract %1007, %1008 : tensor<1x512x32x32xf32>
    %1010 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1011 = stablehlo.broadcast_in_dim %arg248, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1012 = stablehlo.multiply %1010, %1011 : tensor<1x512x32x32xf32>
    %1013 = stablehlo.convert %arg249 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1014 = stablehlo.broadcast_in_dim %1012, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1015 = stablehlo.broadcast_in_dim %1013, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1016 = stablehlo.multiply %1014, %1015 : tensor<1x512x32x32xf32>
    %1017 = stablehlo.convert %arg250 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1018 = stablehlo.broadcast_in_dim %1016, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1019 = stablehlo.broadcast_in_dim %1017, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1020 = stablehlo.add %1018, %1019 : tensor<1x512x32x32xf32>
    %1021 = stablehlo.convert %1020 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %1022 = stablehlo.broadcast_in_dim %1021, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1023 = stablehlo.maximum %643, %1022 : tensor<1x512x32x32xbf16>
    %1024 = stablehlo.minimum %643, %1022 : tensor<1x512x32x32xbf16>
    %1025 = stablehlo.broadcast_in_dim %1024, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1026 = stablehlo.multiply %1025, %648 : tensor<1x512x32x32xbf16>
    %1027 = stablehlo.add %1023, %1026 : tensor<1x512x32x32xbf16>
    %1028 = stablehlo.add %1027, %981 : tensor<1x512x32x32xbf16>
    %1029 = stablehlo.convolution(%1028, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1030 = stablehlo.convert %1029 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1032 = stablehlo.broadcast_in_dim %arg251, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1033 = stablehlo.subtract %1031, %1032 : tensor<1x1024x16x16xf32>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1035 = stablehlo.broadcast_in_dim %arg252, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1036 = stablehlo.multiply %1034, %1035 : tensor<1x1024x16x16xf32>
    %1037 = stablehlo.convert %arg253 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1038 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1039 = stablehlo.broadcast_in_dim %1037, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1040 = stablehlo.multiply %1038, %1039 : tensor<1x1024x16x16xf32>
    %1041 = stablehlo.convert %arg254 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1042 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1043 = stablehlo.broadcast_in_dim %1041, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1044 = stablehlo.add %1042, %1043 : tensor<1x1024x16x16xf32>
    %1045 = stablehlo.convert %1044 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1046 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x1024x16x16xbf16>
    %1047 = stablehlo.broadcast_in_dim %1045, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1048 = stablehlo.maximum %1046, %1047 : tensor<1x1024x16x16xbf16>
    %1049 = stablehlo.minimum %1046, %1047 : tensor<1x1024x16x16xbf16>
    %1050 = stablehlo.broadcast_in_dim %1049, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1051 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x1024x16x16xbf16>
    %1052 = stablehlo.multiply %1050, %1051 : tensor<1x1024x16x16xbf16>
    %1053 = stablehlo.add %1048, %1052 : tensor<1x1024x16x16xbf16>
    %1054 = stablehlo.convolution(%1053, %arg45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1055 = stablehlo.convert %1054 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1056 = stablehlo.broadcast_in_dim %1055, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1057 = stablehlo.broadcast_in_dim %arg255, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1058 = stablehlo.subtract %1056, %1057 : tensor<1x512x16x16xf32>
    %1059 = stablehlo.broadcast_in_dim %1058, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1060 = stablehlo.broadcast_in_dim %arg256, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1061 = stablehlo.multiply %1059, %1060 : tensor<1x512x16x16xf32>
    %1062 = stablehlo.convert %arg257 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1063 = stablehlo.broadcast_in_dim %1061, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1064 = stablehlo.broadcast_in_dim %1062, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1065 = stablehlo.multiply %1063, %1064 : tensor<1x512x16x16xf32>
    %1066 = stablehlo.convert %arg258 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1067 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1068 = stablehlo.broadcast_in_dim %1066, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1069 = stablehlo.add %1067, %1068 : tensor<1x512x16x16xf32>
    %1070 = stablehlo.convert %1069 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1071 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x512x16x16xbf16>
    %1072 = stablehlo.broadcast_in_dim %1070, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1073 = stablehlo.maximum %1071, %1072 : tensor<1x512x16x16xbf16>
    %1074 = stablehlo.minimum %1071, %1072 : tensor<1x512x16x16xbf16>
    %1075 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1076 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x512x16x16xbf16>
    %1077 = stablehlo.multiply %1075, %1076 : tensor<1x512x16x16xbf16>
    %1078 = stablehlo.add %1073, %1077 : tensor<1x512x16x16xbf16>
    %1079 = stablehlo.convolution(%1078, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1080 = stablehlo.convert %1079 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1081 = stablehlo.broadcast_in_dim %1080, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1082 = stablehlo.broadcast_in_dim %arg259, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1083 = stablehlo.subtract %1081, %1082 : tensor<1x1024x16x16xf32>
    %1084 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1085 = stablehlo.broadcast_in_dim %arg260, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1086 = stablehlo.multiply %1084, %1085 : tensor<1x1024x16x16xf32>
    %1087 = stablehlo.convert %arg261 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1088 = stablehlo.broadcast_in_dim %1086, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1089 = stablehlo.broadcast_in_dim %1087, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1090 = stablehlo.multiply %1088, %1089 : tensor<1x1024x16x16xf32>
    %1091 = stablehlo.convert %arg262 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1092 = stablehlo.broadcast_in_dim %1090, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1093 = stablehlo.broadcast_in_dim %1091, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1094 = stablehlo.add %1092, %1093 : tensor<1x1024x16x16xf32>
    %1095 = stablehlo.convert %1094 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1096 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1097 = stablehlo.maximum %1046, %1096 : tensor<1x1024x16x16xbf16>
    %1098 = stablehlo.minimum %1046, %1096 : tensor<1x1024x16x16xbf16>
    %1099 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1100 = stablehlo.multiply %1099, %1051 : tensor<1x1024x16x16xbf16>
    %1101 = stablehlo.add %1097, %1100 : tensor<1x1024x16x16xbf16>
    %1102 = stablehlo.add %1101, %1053 : tensor<1x1024x16x16xbf16>
    %1103 = stablehlo.convolution(%1102, %arg47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1104 = stablehlo.convert %1103 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1105 = stablehlo.broadcast_in_dim %1104, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1106 = stablehlo.broadcast_in_dim %arg263, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1107 = stablehlo.subtract %1105, %1106 : tensor<1x512x16x16xf32>
    %1108 = stablehlo.broadcast_in_dim %1107, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1109 = stablehlo.broadcast_in_dim %arg264, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1110 = stablehlo.multiply %1108, %1109 : tensor<1x512x16x16xf32>
    %1111 = stablehlo.convert %arg265 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1112 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1113 = stablehlo.broadcast_in_dim %1111, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1114 = stablehlo.multiply %1112, %1113 : tensor<1x512x16x16xf32>
    %1115 = stablehlo.convert %arg266 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1116 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1117 = stablehlo.broadcast_in_dim %1115, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1118 = stablehlo.add %1116, %1117 : tensor<1x512x16x16xf32>
    %1119 = stablehlo.convert %1118 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1120 = stablehlo.broadcast_in_dim %1119, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1121 = stablehlo.maximum %1071, %1120 : tensor<1x512x16x16xbf16>
    %1122 = stablehlo.minimum %1071, %1120 : tensor<1x512x16x16xbf16>
    %1123 = stablehlo.broadcast_in_dim %1122, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1124 = stablehlo.multiply %1123, %1076 : tensor<1x512x16x16xbf16>
    %1125 = stablehlo.add %1121, %1124 : tensor<1x512x16x16xbf16>
    %1126 = stablehlo.convolution(%1125, %arg48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1127 = stablehlo.convert %1126 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1128 = stablehlo.broadcast_in_dim %1127, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1129 = stablehlo.broadcast_in_dim %arg267, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1130 = stablehlo.subtract %1128, %1129 : tensor<1x1024x16x16xf32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1132 = stablehlo.broadcast_in_dim %arg268, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1133 = stablehlo.multiply %1131, %1132 : tensor<1x1024x16x16xf32>
    %1134 = stablehlo.convert %arg269 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1135 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1136 = stablehlo.broadcast_in_dim %1134, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1137 = stablehlo.multiply %1135, %1136 : tensor<1x1024x16x16xf32>
    %1138 = stablehlo.convert %arg270 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1139 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1140 = stablehlo.broadcast_in_dim %1138, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1141 = stablehlo.add %1139, %1140 : tensor<1x1024x16x16xf32>
    %1142 = stablehlo.convert %1141 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1143 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1144 = stablehlo.maximum %1046, %1143 : tensor<1x1024x16x16xbf16>
    %1145 = stablehlo.minimum %1046, %1143 : tensor<1x1024x16x16xbf16>
    %1146 = stablehlo.broadcast_in_dim %1145, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1147 = stablehlo.multiply %1146, %1051 : tensor<1x1024x16x16xbf16>
    %1148 = stablehlo.add %1144, %1147 : tensor<1x1024x16x16xbf16>
    %1149 = stablehlo.add %1148, %1102 : tensor<1x1024x16x16xbf16>
    %1150 = stablehlo.convolution(%1149, %arg49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1151 = stablehlo.convert %1150 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1152 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1153 = stablehlo.broadcast_in_dim %arg271, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1154 = stablehlo.subtract %1152, %1153 : tensor<1x512x16x16xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1156 = stablehlo.broadcast_in_dim %arg272, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1157 = stablehlo.multiply %1155, %1156 : tensor<1x512x16x16xf32>
    %1158 = stablehlo.convert %arg273 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1159 = stablehlo.broadcast_in_dim %1157, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1160 = stablehlo.broadcast_in_dim %1158, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1161 = stablehlo.multiply %1159, %1160 : tensor<1x512x16x16xf32>
    %1162 = stablehlo.convert %arg274 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1163 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1164 = stablehlo.broadcast_in_dim %1162, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1165 = stablehlo.add %1163, %1164 : tensor<1x512x16x16xf32>
    %1166 = stablehlo.convert %1165 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1167 = stablehlo.broadcast_in_dim %1166, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1168 = stablehlo.maximum %1071, %1167 : tensor<1x512x16x16xbf16>
    %1169 = stablehlo.minimum %1071, %1167 : tensor<1x512x16x16xbf16>
    %1170 = stablehlo.broadcast_in_dim %1169, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1171 = stablehlo.multiply %1170, %1076 : tensor<1x512x16x16xbf16>
    %1172 = stablehlo.add %1168, %1171 : tensor<1x512x16x16xbf16>
    %1173 = stablehlo.convolution(%1172, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1174 = stablehlo.convert %1173 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1176 = stablehlo.broadcast_in_dim %arg275, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1177 = stablehlo.subtract %1175, %1176 : tensor<1x1024x16x16xf32>
    %1178 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1179 = stablehlo.broadcast_in_dim %arg276, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1180 = stablehlo.multiply %1178, %1179 : tensor<1x1024x16x16xf32>
    %1181 = stablehlo.convert %arg277 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1182 = stablehlo.broadcast_in_dim %1180, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1183 = stablehlo.broadcast_in_dim %1181, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1184 = stablehlo.multiply %1182, %1183 : tensor<1x1024x16x16xf32>
    %1185 = stablehlo.convert %arg278 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1186 = stablehlo.broadcast_in_dim %1184, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1187 = stablehlo.broadcast_in_dim %1185, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1188 = stablehlo.add %1186, %1187 : tensor<1x1024x16x16xf32>
    %1189 = stablehlo.convert %1188 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1190 = stablehlo.broadcast_in_dim %1189, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1191 = stablehlo.maximum %1046, %1190 : tensor<1x1024x16x16xbf16>
    %1192 = stablehlo.minimum %1046, %1190 : tensor<1x1024x16x16xbf16>
    %1193 = stablehlo.broadcast_in_dim %1192, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1194 = stablehlo.multiply %1193, %1051 : tensor<1x1024x16x16xbf16>
    %1195 = stablehlo.add %1191, %1194 : tensor<1x1024x16x16xbf16>
    %1196 = stablehlo.add %1195, %1149 : tensor<1x1024x16x16xbf16>
    %1197 = stablehlo.convolution(%1196, %arg51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1198 = stablehlo.convert %1197 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1199 = stablehlo.broadcast_in_dim %1198, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1200 = stablehlo.broadcast_in_dim %arg279, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1201 = stablehlo.subtract %1199, %1200 : tensor<1x512x16x16xf32>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1203 = stablehlo.broadcast_in_dim %arg280, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1204 = stablehlo.multiply %1202, %1203 : tensor<1x512x16x16xf32>
    %1205 = stablehlo.convert %arg281 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1206 = stablehlo.broadcast_in_dim %1204, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1207 = stablehlo.broadcast_in_dim %1205, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1208 = stablehlo.multiply %1206, %1207 : tensor<1x512x16x16xf32>
    %1209 = stablehlo.convert %arg282 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1210 = stablehlo.broadcast_in_dim %1208, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1211 = stablehlo.broadcast_in_dim %1209, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1212 = stablehlo.add %1210, %1211 : tensor<1x512x16x16xf32>
    %1213 = stablehlo.convert %1212 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1214 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1215 = stablehlo.maximum %1071, %1214 : tensor<1x512x16x16xbf16>
    %1216 = stablehlo.minimum %1071, %1214 : tensor<1x512x16x16xbf16>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1218 = stablehlo.multiply %1217, %1076 : tensor<1x512x16x16xbf16>
    %1219 = stablehlo.add %1215, %1218 : tensor<1x512x16x16xbf16>
    %1220 = stablehlo.convolution(%1219, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1221 = stablehlo.convert %1220 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1222 = stablehlo.broadcast_in_dim %1221, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1223 = stablehlo.broadcast_in_dim %arg283, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1224 = stablehlo.subtract %1222, %1223 : tensor<1x1024x16x16xf32>
    %1225 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1226 = stablehlo.broadcast_in_dim %arg284, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1227 = stablehlo.multiply %1225, %1226 : tensor<1x1024x16x16xf32>
    %1228 = stablehlo.convert %arg285 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1229 = stablehlo.broadcast_in_dim %1227, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1230 = stablehlo.broadcast_in_dim %1228, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1231 = stablehlo.multiply %1229, %1230 : tensor<1x1024x16x16xf32>
    %1232 = stablehlo.convert %arg286 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1233 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1234 = stablehlo.broadcast_in_dim %1232, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1235 = stablehlo.add %1233, %1234 : tensor<1x1024x16x16xf32>
    %1236 = stablehlo.convert %1235 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1237 = stablehlo.broadcast_in_dim %1236, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1238 = stablehlo.maximum %1046, %1237 : tensor<1x1024x16x16xbf16>
    %1239 = stablehlo.minimum %1046, %1237 : tensor<1x1024x16x16xbf16>
    %1240 = stablehlo.broadcast_in_dim %1239, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1241 = stablehlo.multiply %1240, %1051 : tensor<1x1024x16x16xbf16>
    %1242 = stablehlo.add %1238, %1241 : tensor<1x1024x16x16xbf16>
    %1243 = stablehlo.add %1242, %1196 : tensor<1x1024x16x16xbf16>
    %1244 = stablehlo.convolution(%1243, %arg53) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1245 = stablehlo.convert %1244 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1246 = stablehlo.broadcast_in_dim %1245, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1247 = stablehlo.broadcast_in_dim %arg287, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1248 = stablehlo.subtract %1246, %1247 : tensor<1x512x16x16xf32>
    %1249 = stablehlo.broadcast_in_dim %1248, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1250 = stablehlo.broadcast_in_dim %arg288, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1251 = stablehlo.multiply %1249, %1250 : tensor<1x512x16x16xf32>
    %1252 = stablehlo.convert %arg289 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1253 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1254 = stablehlo.broadcast_in_dim %1252, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1255 = stablehlo.multiply %1253, %1254 : tensor<1x512x16x16xf32>
    %1256 = stablehlo.convert %arg290 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1257 = stablehlo.broadcast_in_dim %1255, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1258 = stablehlo.broadcast_in_dim %1256, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1259 = stablehlo.add %1257, %1258 : tensor<1x512x16x16xf32>
    %1260 = stablehlo.convert %1259 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1261 = stablehlo.broadcast_in_dim %1260, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1262 = stablehlo.maximum %1071, %1261 : tensor<1x512x16x16xbf16>
    %1263 = stablehlo.minimum %1071, %1261 : tensor<1x512x16x16xbf16>
    %1264 = stablehlo.broadcast_in_dim %1263, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1265 = stablehlo.multiply %1264, %1076 : tensor<1x512x16x16xbf16>
    %1266 = stablehlo.add %1262, %1265 : tensor<1x512x16x16xbf16>
    %1267 = stablehlo.convolution(%1266, %arg54) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1268 = stablehlo.convert %1267 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1269 = stablehlo.broadcast_in_dim %1268, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1270 = stablehlo.broadcast_in_dim %arg291, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1271 = stablehlo.subtract %1269, %1270 : tensor<1x1024x16x16xf32>
    %1272 = stablehlo.broadcast_in_dim %1271, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1273 = stablehlo.broadcast_in_dim %arg292, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1274 = stablehlo.multiply %1272, %1273 : tensor<1x1024x16x16xf32>
    %1275 = stablehlo.convert %arg293 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1276 = stablehlo.broadcast_in_dim %1274, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1277 = stablehlo.broadcast_in_dim %1275, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1278 = stablehlo.multiply %1276, %1277 : tensor<1x1024x16x16xf32>
    %1279 = stablehlo.convert %arg294 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1280 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1281 = stablehlo.broadcast_in_dim %1279, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1282 = stablehlo.add %1280, %1281 : tensor<1x1024x16x16xf32>
    %1283 = stablehlo.convert %1282 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1284 = stablehlo.broadcast_in_dim %1283, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1285 = stablehlo.maximum %1046, %1284 : tensor<1x1024x16x16xbf16>
    %1286 = stablehlo.minimum %1046, %1284 : tensor<1x1024x16x16xbf16>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1288 = stablehlo.multiply %1287, %1051 : tensor<1x1024x16x16xbf16>
    %1289 = stablehlo.add %1285, %1288 : tensor<1x1024x16x16xbf16>
    %1290 = stablehlo.convolution(%1289, %arg55) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1291 = stablehlo.convert %1290 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1292 = stablehlo.broadcast_in_dim %1291, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1293 = stablehlo.broadcast_in_dim %arg295, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1294 = stablehlo.subtract %1292, %1293 : tensor<1x512x16x16xf32>
    %1295 = stablehlo.broadcast_in_dim %1294, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1296 = stablehlo.broadcast_in_dim %arg296, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1297 = stablehlo.multiply %1295, %1296 : tensor<1x512x16x16xf32>
    %1298 = stablehlo.convert %arg297 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1299 = stablehlo.broadcast_in_dim %1297, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1300 = stablehlo.broadcast_in_dim %1298, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1301 = stablehlo.multiply %1299, %1300 : tensor<1x512x16x16xf32>
    %1302 = stablehlo.convert %arg298 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1303 = stablehlo.broadcast_in_dim %1301, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1304 = stablehlo.broadcast_in_dim %1302, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1305 = stablehlo.add %1303, %1304 : tensor<1x512x16x16xf32>
    %1306 = stablehlo.convert %1305 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1307 = stablehlo.broadcast_in_dim %1306, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1308 = stablehlo.maximum %1071, %1307 : tensor<1x512x16x16xbf16>
    %1309 = stablehlo.minimum %1071, %1307 : tensor<1x512x16x16xbf16>
    %1310 = stablehlo.broadcast_in_dim %1309, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1311 = stablehlo.multiply %1310, %1076 : tensor<1x512x16x16xbf16>
    %1312 = stablehlo.add %1308, %1311 : tensor<1x512x16x16xbf16>
    %1313 = stablehlo.convolution(%1312, %arg56) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1314 = stablehlo.convert %1313 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1315 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1316 = stablehlo.broadcast_in_dim %arg299, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1317 = stablehlo.subtract %1315, %1316 : tensor<1x1024x16x16xf32>
    %1318 = stablehlo.broadcast_in_dim %1317, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1319 = stablehlo.broadcast_in_dim %arg300, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1320 = stablehlo.multiply %1318, %1319 : tensor<1x1024x16x16xf32>
    %1321 = stablehlo.convert %arg301 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1322 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1323 = stablehlo.broadcast_in_dim %1321, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1324 = stablehlo.multiply %1322, %1323 : tensor<1x1024x16x16xf32>
    %1325 = stablehlo.convert %arg302 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1326 = stablehlo.broadcast_in_dim %1324, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1327 = stablehlo.broadcast_in_dim %1325, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1328 = stablehlo.add %1326, %1327 : tensor<1x1024x16x16xf32>
    %1329 = stablehlo.convert %1328 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1330 = stablehlo.broadcast_in_dim %1329, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1331 = stablehlo.maximum %1046, %1330 : tensor<1x1024x16x16xbf16>
    %1332 = stablehlo.minimum %1046, %1330 : tensor<1x1024x16x16xbf16>
    %1333 = stablehlo.broadcast_in_dim %1332, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1334 = stablehlo.multiply %1333, %1051 : tensor<1x1024x16x16xbf16>
    %1335 = stablehlo.add %1331, %1334 : tensor<1x1024x16x16xbf16>
    %1336 = stablehlo.convolution(%1335, %arg57) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x16x16xbf16>
    %1337 = stablehlo.convert %1336 : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xf32>
    %1338 = stablehlo.broadcast_in_dim %1337, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1339 = stablehlo.broadcast_in_dim %arg303, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1340 = stablehlo.subtract %1338, %1339 : tensor<1x512x16x16xf32>
    %1341 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1342 = stablehlo.broadcast_in_dim %arg304, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1343 = stablehlo.multiply %1341, %1342 : tensor<1x512x16x16xf32>
    %1344 = stablehlo.convert %arg305 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1345 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1346 = stablehlo.broadcast_in_dim %1344, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1347 = stablehlo.multiply %1345, %1346 : tensor<1x512x16x16xf32>
    %1348 = stablehlo.convert %arg306 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1349 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %1350 = stablehlo.broadcast_in_dim %1348, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x16x16xf32>
    %1351 = stablehlo.add %1349, %1350 : tensor<1x512x16x16xf32>
    %1352 = stablehlo.convert %1351 : (tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xbf16>
    %1353 = stablehlo.broadcast_in_dim %1352, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1354 = stablehlo.maximum %1071, %1353 : tensor<1x512x16x16xbf16>
    %1355 = stablehlo.minimum %1071, %1353 : tensor<1x512x16x16xbf16>
    %1356 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2, 3] : (tensor<1x512x16x16xbf16>) -> tensor<1x512x16x16xbf16>
    %1357 = stablehlo.multiply %1356, %1076 : tensor<1x512x16x16xbf16>
    %1358 = stablehlo.add %1354, %1357 : tensor<1x512x16x16xbf16>
    %1359 = stablehlo.convolution(%1358, %arg58) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<1024x512x3x3xbf16>) -> tensor<1x1024x16x16xbf16>
    %1360 = stablehlo.convert %1359 : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xf32>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1362 = stablehlo.broadcast_in_dim %arg307, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1363 = stablehlo.subtract %1361, %1362 : tensor<1x1024x16x16xf32>
    %1364 = stablehlo.broadcast_in_dim %1363, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1365 = stablehlo.broadcast_in_dim %arg308, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1366 = stablehlo.multiply %1364, %1365 : tensor<1x1024x16x16xf32>
    %1367 = stablehlo.convert %arg309 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1368 = stablehlo.broadcast_in_dim %1366, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1369 = stablehlo.broadcast_in_dim %1367, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1370 = stablehlo.multiply %1368, %1369 : tensor<1x1024x16x16xf32>
    %1371 = stablehlo.convert %arg310 : (tensor<1024x1x1xbf16>) -> tensor<1024x1x1xf32>
    %1372 = stablehlo.broadcast_in_dim %1370, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
    %1373 = stablehlo.broadcast_in_dim %1371, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x16x16xf32>
    %1374 = stablehlo.add %1372, %1373 : tensor<1x1024x16x16xf32>
    %1375 = stablehlo.convert %1374 : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xbf16>
    %1376 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1377 = stablehlo.maximum %1046, %1376 : tensor<1x1024x16x16xbf16>
    %1378 = stablehlo.minimum %1046, %1376 : tensor<1x1024x16x16xbf16>
    %1379 = stablehlo.broadcast_in_dim %1378, dims = [0, 1, 2, 3] : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %1380 = stablehlo.multiply %1379, %1051 : tensor<1x1024x16x16xbf16>
    %1381 = stablehlo.add %1377, %1380 : tensor<1x1024x16x16xbf16>
    %1382 = stablehlo.convolution(%1381, %arg59) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x16x16xbf16>, tensor<255x1024x1x1xbf16>) -> tensor<1x255x16x16xbf16>
    %1383 = stablehlo.reshape %arg60 : (tensor<255xbf16>) -> tensor<255x1x1xbf16>
    %1384 = stablehlo.broadcast_in_dim %1382, dims = [0, 1, 2, 3] : (tensor<1x255x16x16xbf16>) -> tensor<1x255x16x16xbf16>
    %1385 = stablehlo.broadcast_in_dim %1383, dims = [1, 2, 3] : (tensor<255x1x1xbf16>) -> tensor<1x255x16x16xbf16>
    %1386 = stablehlo.add %1384, %1385 : tensor<1x255x16x16xbf16>
    %1387 = stablehlo.convolution(%1358, %arg61) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x16x16xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x16x16xbf16>
    %1388 = stablehlo.convert %1387 : (tensor<1x256x16x16xbf16>) -> tensor<1x256x16x16xf32>
    %1389 = stablehlo.broadcast_in_dim %1388, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %1390 = stablehlo.broadcast_in_dim %arg311, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x16x16xf32>
    %1391 = stablehlo.subtract %1389, %1390 : tensor<1x256x16x16xf32>
    %1392 = stablehlo.broadcast_in_dim %1391, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %1393 = stablehlo.broadcast_in_dim %arg312, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x16x16xf32>
    %1394 = stablehlo.multiply %1392, %1393 : tensor<1x256x16x16xf32>
    %1395 = stablehlo.convert %arg313 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1396 = stablehlo.broadcast_in_dim %1394, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %1397 = stablehlo.broadcast_in_dim %1395, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x16x16xf32>
    %1398 = stablehlo.multiply %1396, %1397 : tensor<1x256x16x16xf32>
    %1399 = stablehlo.convert %arg314 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1400 = stablehlo.broadcast_in_dim %1398, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %1401 = stablehlo.broadcast_in_dim %1399, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x16x16xf32>
    %1402 = stablehlo.add %1400, %1401 : tensor<1x256x16x16xf32>
    %1403 = stablehlo.convert %1402 : (tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xbf16>
    %1404 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x256x16x16xbf16>
    %1405 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xbf16>) -> tensor<1x256x16x16xbf16>
    %1406 = stablehlo.maximum %1404, %1405 : tensor<1x256x16x16xbf16>
    %1407 = stablehlo.minimum %1404, %1405 : tensor<1x256x16x16xbf16>
    %1408 = stablehlo.broadcast_in_dim %1407, dims = [0, 1, 2, 3] : (tensor<1x256x16x16xbf16>) -> tensor<1x256x16x16xbf16>
    %1409 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x256x16x16xbf16>
    %1410 = stablehlo.multiply %1408, %1409 : tensor<1x256x16x16xbf16>
    %1411 = stablehlo.add %1406, %1410 : tensor<1x256x16x16xbf16>
    %1412 = stablehlo.transpose %1411, dims = [0, 1, 3, 2] : (tensor<1x256x16x16xbf16>) -> tensor<1x256x16x16xbf16>
    %1413 = stablehlo.reshape %1412 : (tensor<1x256x16x16xbf16>) -> tensor<256x16x16xbf16>
    %1414 = stablehlo.broadcast_in_dim %arg315, dims = [0, 1, 2] : (tensor<256x16x32xbf16>) -> tensor<256x16x32xbf16>
    %1415 = stablehlo.dot_general %1413, %1414, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x16x16xbf16>, tensor<256x16x32xbf16>) -> tensor<256x16x32xbf16>
    %1416 = stablehlo.reshape %1415 : (tensor<256x16x32xbf16>) -> tensor<1x256x16x32xbf16>
    %1417 = stablehlo.transpose %1416, dims = [0, 1, 3, 2] : (tensor<1x256x16x32xbf16>) -> tensor<1x256x32x16xbf16>
    %1418 = stablehlo.reshape %1417 : (tensor<1x256x32x16xbf16>) -> tensor<256x32x16xbf16>
    %1419 = stablehlo.broadcast_in_dim %arg316, dims = [0, 1, 2] : (tensor<256x16x32xbf16>) -> tensor<256x16x32xbf16>
    %1420 = stablehlo.dot_general %1418, %1419, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x16xbf16>, tensor<256x16x32xbf16>) -> tensor<256x32x32xbf16>
    %1421 = stablehlo.reshape %1420 : (tensor<256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1422 = stablehlo.concatenate %1421, %1028, dim = 1 : (tensor<1x256x32x32xbf16>, tensor<1x512x32x32xbf16>) -> tensor<1x768x32x32xbf16>
    %1423 = stablehlo.convolution(%1422, %arg62) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x768x32x32xbf16>, tensor<256x768x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %1424 = stablehlo.convert %1423 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %1425 = stablehlo.broadcast_in_dim %1424, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1426 = stablehlo.broadcast_in_dim %arg317, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1427 = stablehlo.subtract %1425, %1426 : tensor<1x256x32x32xf32>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1429 = stablehlo.broadcast_in_dim %arg318, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1430 = stablehlo.multiply %1428, %1429 : tensor<1x256x32x32xf32>
    %1431 = stablehlo.convert %arg319 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1432 = stablehlo.broadcast_in_dim %1430, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1433 = stablehlo.broadcast_in_dim %1431, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1434 = stablehlo.multiply %1432, %1433 : tensor<1x256x32x32xf32>
    %1435 = stablehlo.convert %arg320 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1436 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1437 = stablehlo.broadcast_in_dim %1435, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1438 = stablehlo.add %1436, %1437 : tensor<1x256x32x32xf32>
    %1439 = stablehlo.convert %1438 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %1440 = stablehlo.broadcast_in_dim %1439, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1441 = stablehlo.maximum %668, %1440 : tensor<1x256x32x32xbf16>
    %1442 = stablehlo.minimum %668, %1440 : tensor<1x256x32x32xbf16>
    %1443 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1444 = stablehlo.multiply %1443, %673 : tensor<1x256x32x32xbf16>
    %1445 = stablehlo.add %1441, %1444 : tensor<1x256x32x32xbf16>
    %1446 = stablehlo.convolution(%1445, %arg63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %1447 = stablehlo.convert %1446 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %1448 = stablehlo.broadcast_in_dim %1447, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1449 = stablehlo.broadcast_in_dim %arg321, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1450 = stablehlo.subtract %1448, %1449 : tensor<1x512x32x32xf32>
    %1451 = stablehlo.broadcast_in_dim %1450, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1452 = stablehlo.broadcast_in_dim %arg322, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1453 = stablehlo.multiply %1451, %1452 : tensor<1x512x32x32xf32>
    %1454 = stablehlo.convert %arg323 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1455 = stablehlo.broadcast_in_dim %1453, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1456 = stablehlo.broadcast_in_dim %1454, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1457 = stablehlo.multiply %1455, %1456 : tensor<1x512x32x32xf32>
    %1458 = stablehlo.convert %arg324 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1459 = stablehlo.broadcast_in_dim %1457, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1460 = stablehlo.broadcast_in_dim %1458, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1461 = stablehlo.add %1459, %1460 : tensor<1x512x32x32xf32>
    %1462 = stablehlo.convert %1461 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1464 = stablehlo.maximum %643, %1463 : tensor<1x512x32x32xbf16>
    %1465 = stablehlo.minimum %643, %1463 : tensor<1x512x32x32xbf16>
    %1466 = stablehlo.broadcast_in_dim %1465, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1467 = stablehlo.multiply %1466, %648 : tensor<1x512x32x32xbf16>
    %1468 = stablehlo.add %1464, %1467 : tensor<1x512x32x32xbf16>
    %1469 = stablehlo.convolution(%1468, %arg64) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %1470 = stablehlo.convert %1469 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %1471 = stablehlo.broadcast_in_dim %1470, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1472 = stablehlo.broadcast_in_dim %arg325, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1473 = stablehlo.subtract %1471, %1472 : tensor<1x256x32x32xf32>
    %1474 = stablehlo.broadcast_in_dim %1473, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1475 = stablehlo.broadcast_in_dim %arg326, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1476 = stablehlo.multiply %1474, %1475 : tensor<1x256x32x32xf32>
    %1477 = stablehlo.convert %arg327 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1478 = stablehlo.broadcast_in_dim %1476, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1479 = stablehlo.broadcast_in_dim %1477, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1480 = stablehlo.multiply %1478, %1479 : tensor<1x256x32x32xf32>
    %1481 = stablehlo.convert %arg328 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1482 = stablehlo.broadcast_in_dim %1480, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1483 = stablehlo.broadcast_in_dim %1481, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1484 = stablehlo.add %1482, %1483 : tensor<1x256x32x32xf32>
    %1485 = stablehlo.convert %1484 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %1486 = stablehlo.broadcast_in_dim %1485, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1487 = stablehlo.maximum %668, %1486 : tensor<1x256x32x32xbf16>
    %1488 = stablehlo.minimum %668, %1486 : tensor<1x256x32x32xbf16>
    %1489 = stablehlo.broadcast_in_dim %1488, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1490 = stablehlo.multiply %1489, %673 : tensor<1x256x32x32xbf16>
    %1491 = stablehlo.add %1487, %1490 : tensor<1x256x32x32xbf16>
    %1492 = stablehlo.convolution(%1491, %arg65) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %1493 = stablehlo.convert %1492 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %1494 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1495 = stablehlo.broadcast_in_dim %arg329, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1496 = stablehlo.subtract %1494, %1495 : tensor<1x512x32x32xf32>
    %1497 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1498 = stablehlo.broadcast_in_dim %arg330, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1499 = stablehlo.multiply %1497, %1498 : tensor<1x512x32x32xf32>
    %1500 = stablehlo.convert %arg331 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1501 = stablehlo.broadcast_in_dim %1499, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1502 = stablehlo.broadcast_in_dim %1500, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1503 = stablehlo.multiply %1501, %1502 : tensor<1x512x32x32xf32>
    %1504 = stablehlo.convert %arg332 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1505 = stablehlo.broadcast_in_dim %1503, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1506 = stablehlo.broadcast_in_dim %1504, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1507 = stablehlo.add %1505, %1506 : tensor<1x512x32x32xf32>
    %1508 = stablehlo.convert %1507 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %1509 = stablehlo.broadcast_in_dim %1508, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1510 = stablehlo.maximum %643, %1509 : tensor<1x512x32x32xbf16>
    %1511 = stablehlo.minimum %643, %1509 : tensor<1x512x32x32xbf16>
    %1512 = stablehlo.broadcast_in_dim %1511, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1513 = stablehlo.multiply %1512, %648 : tensor<1x512x32x32xbf16>
    %1514 = stablehlo.add %1510, %1513 : tensor<1x512x32x32xbf16>
    %1515 = stablehlo.convolution(%1514, %arg66) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x32x32xbf16>
    %1516 = stablehlo.convert %1515 : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xf32>
    %1517 = stablehlo.broadcast_in_dim %1516, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1518 = stablehlo.broadcast_in_dim %arg333, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1519 = stablehlo.subtract %1517, %1518 : tensor<1x256x32x32xf32>
    %1520 = stablehlo.broadcast_in_dim %1519, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1521 = stablehlo.broadcast_in_dim %arg334, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1522 = stablehlo.multiply %1520, %1521 : tensor<1x256x32x32xf32>
    %1523 = stablehlo.convert %arg335 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1524 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1525 = stablehlo.broadcast_in_dim %1523, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1526 = stablehlo.multiply %1524, %1525 : tensor<1x256x32x32xf32>
    %1527 = stablehlo.convert %arg336 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1528 = stablehlo.broadcast_in_dim %1526, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xf32>
    %1529 = stablehlo.broadcast_in_dim %1527, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x32x32xf32>
    %1530 = stablehlo.add %1528, %1529 : tensor<1x256x32x32xf32>
    %1531 = stablehlo.convert %1530 : (tensor<1x256x32x32xf32>) -> tensor<1x256x32x32xbf16>
    %1532 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1533 = stablehlo.maximum %668, %1532 : tensor<1x256x32x32xbf16>
    %1534 = stablehlo.minimum %668, %1532 : tensor<1x256x32x32xbf16>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [0, 1, 2, 3] : (tensor<1x256x32x32xbf16>) -> tensor<1x256x32x32xbf16>
    %1536 = stablehlo.multiply %1535, %673 : tensor<1x256x32x32xbf16>
    %1537 = stablehlo.add %1533, %1536 : tensor<1x256x32x32xbf16>
    %1538 = stablehlo.convolution(%1537, %arg67) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<512x256x3x3xbf16>) -> tensor<1x512x32x32xbf16>
    %1539 = stablehlo.convert %1538 : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xf32>
    %1540 = stablehlo.broadcast_in_dim %1539, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1541 = stablehlo.broadcast_in_dim %arg337, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1542 = stablehlo.subtract %1540, %1541 : tensor<1x512x32x32xf32>
    %1543 = stablehlo.broadcast_in_dim %1542, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1544 = stablehlo.broadcast_in_dim %arg338, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1545 = stablehlo.multiply %1543, %1544 : tensor<1x512x32x32xf32>
    %1546 = stablehlo.convert %arg339 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1547 = stablehlo.broadcast_in_dim %1545, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1548 = stablehlo.broadcast_in_dim %1546, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1549 = stablehlo.multiply %1547, %1548 : tensor<1x512x32x32xf32>
    %1550 = stablehlo.convert %arg340 : (tensor<512x1x1xbf16>) -> tensor<512x1x1xf32>
    %1551 = stablehlo.broadcast_in_dim %1549, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xf32>
    %1552 = stablehlo.broadcast_in_dim %1550, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x32x32xf32>
    %1553 = stablehlo.add %1551, %1552 : tensor<1x512x32x32xf32>
    %1554 = stablehlo.convert %1553 : (tensor<1x512x32x32xf32>) -> tensor<1x512x32x32xbf16>
    %1555 = stablehlo.broadcast_in_dim %1554, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1556 = stablehlo.maximum %643, %1555 : tensor<1x512x32x32xbf16>
    %1557 = stablehlo.minimum %643, %1555 : tensor<1x512x32x32xbf16>
    %1558 = stablehlo.broadcast_in_dim %1557, dims = [0, 1, 2, 3] : (tensor<1x512x32x32xbf16>) -> tensor<1x512x32x32xbf16>
    %1559 = stablehlo.multiply %1558, %648 : tensor<1x512x32x32xbf16>
    %1560 = stablehlo.add %1556, %1559 : tensor<1x512x32x32xbf16>
    %1561 = stablehlo.convolution(%1560, %arg68) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x32x32xbf16>, tensor<255x512x1x1xbf16>) -> tensor<1x255x32x32xbf16>
    %1562 = stablehlo.reshape %arg69 : (tensor<255xbf16>) -> tensor<255x1x1xbf16>
    %1563 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2, 3] : (tensor<1x255x32x32xbf16>) -> tensor<1x255x32x32xbf16>
    %1564 = stablehlo.broadcast_in_dim %1562, dims = [1, 2, 3] : (tensor<255x1x1xbf16>) -> tensor<1x255x32x32xbf16>
    %1565 = stablehlo.add %1563, %1564 : tensor<1x255x32x32xbf16>
    %1566 = stablehlo.convolution(%1537, %arg70) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x32x32xbf16>
    %1567 = stablehlo.convert %1566 : (tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xf32>
    %1568 = stablehlo.broadcast_in_dim %1567, dims = [0, 1, 2, 3] : (tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
    %1569 = stablehlo.broadcast_in_dim %arg341, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x32x32xf32>
    %1570 = stablehlo.subtract %1568, %1569 : tensor<1x128x32x32xf32>
    %1571 = stablehlo.broadcast_in_dim %1570, dims = [0, 1, 2, 3] : (tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
    %1572 = stablehlo.broadcast_in_dim %arg342, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x32x32xf32>
    %1573 = stablehlo.multiply %1571, %1572 : tensor<1x128x32x32xf32>
    %1574 = stablehlo.convert %arg343 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1575 = stablehlo.broadcast_in_dim %1573, dims = [0, 1, 2, 3] : (tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
    %1576 = stablehlo.broadcast_in_dim %1574, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x32x32xf32>
    %1577 = stablehlo.multiply %1575, %1576 : tensor<1x128x32x32xf32>
    %1578 = stablehlo.convert %arg344 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1579 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2, 3] : (tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
    %1580 = stablehlo.broadcast_in_dim %1578, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x32x32xf32>
    %1581 = stablehlo.add %1579, %1580 : tensor<1x128x32x32xf32>
    %1582 = stablehlo.convert %1581 : (tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xbf16>
    %1583 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<bf16>) -> tensor<1x128x32x32xbf16>
    %1584 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2, 3] : (tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>
    %1585 = stablehlo.maximum %1583, %1584 : tensor<1x128x32x32xbf16>
    %1586 = stablehlo.minimum %1583, %1584 : tensor<1x128x32x32xbf16>
    %1587 = stablehlo.broadcast_in_dim %1586, dims = [0, 1, 2, 3] : (tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>
    %1588 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<bf16>) -> tensor<1x128x32x32xbf16>
    %1589 = stablehlo.multiply %1587, %1588 : tensor<1x128x32x32xbf16>
    %1590 = stablehlo.add %1585, %1589 : tensor<1x128x32x32xbf16>
    %1591 = stablehlo.transpose %1590, dims = [0, 1, 3, 2] : (tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>
    %1592 = stablehlo.reshape %1591 : (tensor<1x128x32x32xbf16>) -> tensor<128x32x32xbf16>
    %1593 = stablehlo.broadcast_in_dim %arg345, dims = [0, 1, 2] : (tensor<128x32x64xbf16>) -> tensor<128x32x64xbf16>
    %1594 = stablehlo.dot_general %1592, %1593, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x32x32xbf16>, tensor<128x32x64xbf16>) -> tensor<128x32x64xbf16>
    %1595 = stablehlo.reshape %1594 : (tensor<128x32x64xbf16>) -> tensor<1x128x32x64xbf16>
    %1596 = stablehlo.transpose %1595, dims = [0, 1, 3, 2] : (tensor<1x128x32x64xbf16>) -> tensor<1x128x64x32xbf16>
    %1597 = stablehlo.reshape %1596 : (tensor<1x128x64x32xbf16>) -> tensor<128x64x32xbf16>
    %1598 = stablehlo.broadcast_in_dim %arg346, dims = [0, 1, 2] : (tensor<128x32x64xbf16>) -> tensor<128x32x64xbf16>
    %1599 = stablehlo.dot_general %1597, %1598, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x64x32xbf16>, tensor<128x32x64xbf16>) -> tensor<128x64x64xbf16>
    %1600 = stablehlo.reshape %1599 : (tensor<128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1601 = stablehlo.concatenate %1600, %625, dim = 1 : (tensor<1x128x64x64xbf16>, tensor<1x256x64x64xbf16>) -> tensor<1x384x64x64xbf16>
    %1602 = stablehlo.convolution(%1601, %arg71) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x64x64xbf16>, tensor<128x384x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %1603 = stablehlo.convert %1602 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %1604 = stablehlo.broadcast_in_dim %1603, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1605 = stablehlo.broadcast_in_dim %arg347, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1606 = stablehlo.subtract %1604, %1605 : tensor<1x128x64x64xf32>
    %1607 = stablehlo.broadcast_in_dim %1606, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1608 = stablehlo.broadcast_in_dim %arg348, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1609 = stablehlo.multiply %1607, %1608 : tensor<1x128x64x64xf32>
    %1610 = stablehlo.convert %arg349 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1611 = stablehlo.broadcast_in_dim %1609, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1612 = stablehlo.broadcast_in_dim %1610, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1613 = stablehlo.multiply %1611, %1612 : tensor<1x128x64x64xf32>
    %1614 = stablehlo.convert %arg350 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1615 = stablehlo.broadcast_in_dim %1613, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1616 = stablehlo.broadcast_in_dim %1614, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1617 = stablehlo.add %1615, %1616 : tensor<1x128x64x64xf32>
    %1618 = stablehlo.convert %1617 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %1619 = stablehlo.broadcast_in_dim %1618, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1620 = stablehlo.maximum %265, %1619 : tensor<1x128x64x64xbf16>
    %1621 = stablehlo.minimum %265, %1619 : tensor<1x128x64x64xbf16>
    %1622 = stablehlo.broadcast_in_dim %1621, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1623 = stablehlo.multiply %1622, %270 : tensor<1x128x64x64xbf16>
    %1624 = stablehlo.add %1620, %1623 : tensor<1x128x64x64xbf16>
    %1625 = stablehlo.convolution(%1624, %arg72) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %1626 = stablehlo.convert %1625 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %1627 = stablehlo.broadcast_in_dim %1626, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1628 = stablehlo.broadcast_in_dim %arg351, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1629 = stablehlo.subtract %1627, %1628 : tensor<1x256x64x64xf32>
    %1630 = stablehlo.broadcast_in_dim %1629, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1631 = stablehlo.broadcast_in_dim %arg352, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1632 = stablehlo.multiply %1630, %1631 : tensor<1x256x64x64xf32>
    %1633 = stablehlo.convert %arg353 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1634 = stablehlo.broadcast_in_dim %1632, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1635 = stablehlo.broadcast_in_dim %1633, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1636 = stablehlo.multiply %1634, %1635 : tensor<1x256x64x64xf32>
    %1637 = stablehlo.convert %arg354 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1638 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1639 = stablehlo.broadcast_in_dim %1637, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1640 = stablehlo.add %1638, %1639 : tensor<1x256x64x64xf32>
    %1641 = stablehlo.convert %1640 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %1642 = stablehlo.broadcast_in_dim %1641, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1643 = stablehlo.maximum %240, %1642 : tensor<1x256x64x64xbf16>
    %1644 = stablehlo.minimum %240, %1642 : tensor<1x256x64x64xbf16>
    %1645 = stablehlo.broadcast_in_dim %1644, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1646 = stablehlo.multiply %1645, %245 : tensor<1x256x64x64xbf16>
    %1647 = stablehlo.add %1643, %1646 : tensor<1x256x64x64xbf16>
    %1648 = stablehlo.convolution(%1647, %arg73) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %1649 = stablehlo.convert %1648 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %1650 = stablehlo.broadcast_in_dim %1649, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1651 = stablehlo.broadcast_in_dim %arg355, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1652 = stablehlo.subtract %1650, %1651 : tensor<1x128x64x64xf32>
    %1653 = stablehlo.broadcast_in_dim %1652, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1654 = stablehlo.broadcast_in_dim %arg356, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1655 = stablehlo.multiply %1653, %1654 : tensor<1x128x64x64xf32>
    %1656 = stablehlo.convert %arg357 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1657 = stablehlo.broadcast_in_dim %1655, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1658 = stablehlo.broadcast_in_dim %1656, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1659 = stablehlo.multiply %1657, %1658 : tensor<1x128x64x64xf32>
    %1660 = stablehlo.convert %arg358 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1661 = stablehlo.broadcast_in_dim %1659, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1662 = stablehlo.broadcast_in_dim %1660, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1663 = stablehlo.add %1661, %1662 : tensor<1x128x64x64xf32>
    %1664 = stablehlo.convert %1663 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %1665 = stablehlo.broadcast_in_dim %1664, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1666 = stablehlo.maximum %265, %1665 : tensor<1x128x64x64xbf16>
    %1667 = stablehlo.minimum %265, %1665 : tensor<1x128x64x64xbf16>
    %1668 = stablehlo.broadcast_in_dim %1667, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1669 = stablehlo.multiply %1668, %270 : tensor<1x128x64x64xbf16>
    %1670 = stablehlo.add %1666, %1669 : tensor<1x128x64x64xbf16>
    %1671 = stablehlo.convolution(%1670, %arg74) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %1672 = stablehlo.convert %1671 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %1673 = stablehlo.broadcast_in_dim %1672, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1674 = stablehlo.broadcast_in_dim %arg359, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1675 = stablehlo.subtract %1673, %1674 : tensor<1x256x64x64xf32>
    %1676 = stablehlo.broadcast_in_dim %1675, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1677 = stablehlo.broadcast_in_dim %arg360, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1678 = stablehlo.multiply %1676, %1677 : tensor<1x256x64x64xf32>
    %1679 = stablehlo.convert %arg361 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1680 = stablehlo.broadcast_in_dim %1678, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1681 = stablehlo.broadcast_in_dim %1679, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1682 = stablehlo.multiply %1680, %1681 : tensor<1x256x64x64xf32>
    %1683 = stablehlo.convert %arg362 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1684 = stablehlo.broadcast_in_dim %1682, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1685 = stablehlo.broadcast_in_dim %1683, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1686 = stablehlo.add %1684, %1685 : tensor<1x256x64x64xf32>
    %1687 = stablehlo.convert %1686 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1689 = stablehlo.maximum %240, %1688 : tensor<1x256x64x64xbf16>
    %1690 = stablehlo.minimum %240, %1688 : tensor<1x256x64x64xbf16>
    %1691 = stablehlo.broadcast_in_dim %1690, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1692 = stablehlo.multiply %1691, %245 : tensor<1x256x64x64xbf16>
    %1693 = stablehlo.add %1689, %1692 : tensor<1x256x64x64xbf16>
    %1694 = stablehlo.convolution(%1693, %arg75) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x64x64xbf16>
    %1695 = stablehlo.convert %1694 : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xf32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1697 = stablehlo.broadcast_in_dim %arg363, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1698 = stablehlo.subtract %1696, %1697 : tensor<1x128x64x64xf32>
    %1699 = stablehlo.broadcast_in_dim %1698, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1700 = stablehlo.broadcast_in_dim %arg364, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1701 = stablehlo.multiply %1699, %1700 : tensor<1x128x64x64xf32>
    %1702 = stablehlo.convert %arg365 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1703 = stablehlo.broadcast_in_dim %1701, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1704 = stablehlo.broadcast_in_dim %1702, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1705 = stablehlo.multiply %1703, %1704 : tensor<1x128x64x64xf32>
    %1706 = stablehlo.convert %arg366 : (tensor<128x1x1xbf16>) -> tensor<128x1x1xf32>
    %1707 = stablehlo.broadcast_in_dim %1705, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %1708 = stablehlo.broadcast_in_dim %1706, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %1709 = stablehlo.add %1707, %1708 : tensor<1x128x64x64xf32>
    %1710 = stablehlo.convert %1709 : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xbf16>
    %1711 = stablehlo.broadcast_in_dim %1710, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1712 = stablehlo.maximum %265, %1711 : tensor<1x128x64x64xbf16>
    %1713 = stablehlo.minimum %265, %1711 : tensor<1x128x64x64xbf16>
    %1714 = stablehlo.broadcast_in_dim %1713, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xbf16>) -> tensor<1x128x64x64xbf16>
    %1715 = stablehlo.multiply %1714, %270 : tensor<1x128x64x64xbf16>
    %1716 = stablehlo.add %1712, %1715 : tensor<1x128x64x64xbf16>
    %1717 = stablehlo.convolution(%1716, %arg76) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x64x64xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x64x64xbf16>
    %1718 = stablehlo.convert %1717 : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xf32>
    %1719 = stablehlo.broadcast_in_dim %1718, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1720 = stablehlo.broadcast_in_dim %arg367, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1721 = stablehlo.subtract %1719, %1720 : tensor<1x256x64x64xf32>
    %1722 = stablehlo.broadcast_in_dim %1721, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1723 = stablehlo.broadcast_in_dim %arg368, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1724 = stablehlo.multiply %1722, %1723 : tensor<1x256x64x64xf32>
    %1725 = stablehlo.convert %arg369 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1726 = stablehlo.broadcast_in_dim %1724, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1727 = stablehlo.broadcast_in_dim %1725, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1728 = stablehlo.multiply %1726, %1727 : tensor<1x256x64x64xf32>
    %1729 = stablehlo.convert %arg370 : (tensor<256x1x1xbf16>) -> tensor<256x1x1xf32>
    %1730 = stablehlo.broadcast_in_dim %1728, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
    %1731 = stablehlo.broadcast_in_dim %1729, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x64x64xf32>
    %1732 = stablehlo.add %1730, %1731 : tensor<1x256x64x64xf32>
    %1733 = stablehlo.convert %1732 : (tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xbf16>
    %1734 = stablehlo.broadcast_in_dim %1733, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1735 = stablehlo.maximum %240, %1734 : tensor<1x256x64x64xbf16>
    %1736 = stablehlo.minimum %240, %1734 : tensor<1x256x64x64xbf16>
    %1737 = stablehlo.broadcast_in_dim %1736, dims = [0, 1, 2, 3] : (tensor<1x256x64x64xbf16>) -> tensor<1x256x64x64xbf16>
    %1738 = stablehlo.multiply %1737, %245 : tensor<1x256x64x64xbf16>
    %1739 = stablehlo.add %1735, %1738 : tensor<1x256x64x64xbf16>
    %1740 = stablehlo.convolution(%1739, %arg77) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x64x64xbf16>, tensor<255x256x1x1xbf16>) -> tensor<1x255x64x64xbf16>
    %1741 = stablehlo.reshape %arg78 : (tensor<255xbf16>) -> tensor<255x1x1xbf16>
    %1742 = stablehlo.broadcast_in_dim %1740, dims = [0, 1, 2, 3] : (tensor<1x255x64x64xbf16>) -> tensor<1x255x64x64xbf16>
    %1743 = stablehlo.broadcast_in_dim %1741, dims = [1, 2, 3] : (tensor<255x1x1xbf16>) -> tensor<1x255x64x64xbf16>
    %1744 = stablehlo.add %1742, %1743 : tensor<1x255x64x64xbf16>
    return %1386, %1565, %1744 : tensor<1x255x16x16xbf16>, tensor<1x255x32x32xbf16>, tensor<1x255x64x64xbf16>
  }
}
