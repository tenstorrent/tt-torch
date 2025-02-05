module {
  func.func @main(%arg0: tensor<1x3x720x1280xbf16>, %arg1: tensor<64x3x7x7xbf16>, %arg2: tensor<64x64x1x1xbf16>, %arg3: tensor<64x64x3x3xbf16>, %arg4: tensor<256x64x1x1xbf16>, %arg5: tensor<256x64x1x1xbf16>, %arg6: tensor<64x256x1x1xbf16>, %arg7: tensor<64x64x3x3xbf16>, %arg8: tensor<256x64x1x1xbf16>, %arg9: tensor<64x256x1x1xbf16>, %arg10: tensor<64x64x3x3xbf16>, %arg11: tensor<256x64x1x1xbf16>, %arg12: tensor<128x256x1x1xbf16>, %arg13: tensor<128x128x3x3xbf16>, %arg14: tensor<512x128x1x1xbf16>, %arg15: tensor<512x256x1x1xbf16>, %arg16: tensor<128x512x1x1xbf16>, %arg17: tensor<128x128x3x3xbf16>, %arg18: tensor<512x128x1x1xbf16>, %arg19: tensor<128x512x1x1xbf16>, %arg20: tensor<128x128x3x3xbf16>, %arg21: tensor<512x128x1x1xbf16>, %arg22: tensor<128x512x1x1xbf16>, %arg23: tensor<128x128x3x3xbf16>, %arg24: tensor<512x128x1x1xbf16>, %arg25: tensor<256x512x1x1xbf16>, %arg26: tensor<256x256x3x3xbf16>, %arg27: tensor<1024x256x1x1xbf16>, %arg28: tensor<1024x512x1x1xbf16>, %arg29: tensor<256x1024x1x1xbf16>, %arg30: tensor<256x256x3x3xbf16>, %arg31: tensor<1024x256x1x1xbf16>, %arg32: tensor<256x1024x1x1xbf16>, %arg33: tensor<256x256x3x3xbf16>, %arg34: tensor<1024x256x1x1xbf16>, %arg35: tensor<256x1024x1x1xbf16>, %arg36: tensor<256x256x3x3xbf16>, %arg37: tensor<1024x256x1x1xbf16>, %arg38: tensor<256x1024x1x1xbf16>, %arg39: tensor<256x256x3x3xbf16>, %arg40: tensor<1024x256x1x1xbf16>, %arg41: tensor<256x1024x1x1xbf16>, %arg42: tensor<256x256x3x3xbf16>, %arg43: tensor<1024x256x1x1xbf16>, %arg44: tensor<512x1024x1x1xbf16>, %arg45: tensor<512x512x3x3xbf16>, %arg46: tensor<2048x512x1x1xbf16>, %arg47: tensor<2048x1024x1x1xbf16>, %arg48: tensor<512x2048x1x1xbf16>, %arg49: tensor<512x512x3x3xbf16>, %arg50: tensor<2048x512x1x1xbf16>, %arg51: tensor<512x2048x1x1xbf16>, %arg52: tensor<512x512x3x3xbf16>, %arg53: tensor<2048x512x1x1xbf16>, %arg54: tensor<256x2048x1x1xbf16>, %arg55: tensor<256xbf16>, %arg56: tensor<256xbf16>, %arg57: tensor<256xbf16>, %arg58: tensor<256xbf16>, %arg59: tensor<256xbf16>, %arg60: tensor<256xbf16>, %arg61: tensor<256xbf16>, %arg62: tensor<256xbf16>, %arg63: tensor<256xbf16>, %arg64: tensor<256xbf16>, %arg65: tensor<256xbf16>, %arg66: tensor<256xbf16>, %arg67: tensor<256xbf16>, %arg68: tensor<256xbf16>, %arg69: tensor<256xbf16>, %arg70: tensor<256xbf16>, %arg71: tensor<256xbf16>, %arg72: tensor<256xbf16>, %arg73: tensor<256xbf16>, %arg74: tensor<256xbf16>, %arg75: tensor<256xbf16>, %arg76: tensor<256xbf16>, %arg77: tensor<256xbf16>, %arg78: tensor<256xbf16>, %arg79: tensor<256xbf16>, %arg80: tensor<256xbf16>, %arg81: tensor<256xbf16>, %arg82: tensor<256xbf16>, %arg83: tensor<256xbf16>, %arg84: tensor<256xbf16>, %arg85: tensor<256xbf16>, %arg86: tensor<256xbf16>, %arg87: tensor<256xbf16>, %arg88: tensor<256xbf16>, %arg89: tensor<256xbf16>, %arg90: tensor<256xbf16>, %arg91: tensor<256xbf16>, %arg92: tensor<256xbf16>, %arg93: tensor<256xbf16>, %arg94: tensor<256xbf16>, %arg95: tensor<256xbf16>, %arg96: tensor<256xbf16>, %arg97: tensor<256xbf16>, %arg98: tensor<256xbf16>, %arg99: tensor<256xbf16>, %arg100: tensor<256xbf16>, %arg101: tensor<256xbf16>, %arg102: tensor<256xbf16>, %arg103: tensor<256xbf16>, %arg104: tensor<256xbf16>, %arg105: tensor<256xbf16>, %arg106: tensor<256xbf16>, %arg107: tensor<256xbf16>, %arg108: tensor<256xbf16>, %arg109: tensor<256xbf16>, %arg110: tensor<256xbf16>, %arg111: tensor<256xbf16>, %arg112: tensor<256xbf16>, %arg113: tensor<256xbf16>, %arg114: tensor<256xbf16>, %arg115: tensor<256xbf16>, %arg116: tensor<92xbf16>, %arg117: tensor<256xbf16>, %arg118: tensor<256xbf16>, %arg119: tensor<4xbf16>, %arg120: tensor<3x720x1280xbf16>, %arg121: tensor<1x3x720x1280xbf16>, %arg122: tensor<1x64x1x1xbf16>, %arg123: tensor<1x64x1x1xbf16>, %arg124: tensor<1x64x1x1xbf16>, %arg125: tensor<1x64x1x1xbf16>, %arg126: tensor<1x64x1x1xbf16>, %arg127: tensor<1x64x1x1xbf16>, %arg128: tensor<1x256x1x1xbf16>, %arg129: tensor<1x256x1x1xbf16>, %arg130: tensor<1x256x1x1xbf16>, %arg131: tensor<1x256x1x1xbf16>, %arg132: tensor<1x64x1x1xbf16>, %arg133: tensor<1x64x1x1xbf16>, %arg134: tensor<1x64x1x1xbf16>, %arg135: tensor<1x64x1x1xbf16>, %arg136: tensor<1x256x1x1xbf16>, %arg137: tensor<1x256x1x1xbf16>, %arg138: tensor<1x64x1x1xbf16>, %arg139: tensor<1x64x1x1xbf16>, %arg140: tensor<1x64x1x1xbf16>, %arg141: tensor<1x64x1x1xbf16>, %arg142: tensor<1x256x1x1xbf16>, %arg143: tensor<1x256x1x1xbf16>, %arg144: tensor<1x128x1x1xbf16>, %arg145: tensor<1x128x1x1xbf16>, %arg146: tensor<1x128x1x1xbf16>, %arg147: tensor<1x128x1x1xbf16>, %arg148: tensor<1x512x1x1xbf16>, %arg149: tensor<1x512x1x1xbf16>, %arg150: tensor<1x512x1x1xbf16>, %arg151: tensor<1x512x1x1xbf16>, %arg152: tensor<1x128x1x1xbf16>, %arg153: tensor<1x128x1x1xbf16>, %arg154: tensor<1x128x1x1xbf16>, %arg155: tensor<1x128x1x1xbf16>, %arg156: tensor<1x512x1x1xbf16>, %arg157: tensor<1x512x1x1xbf16>, %arg158: tensor<1x128x1x1xbf16>, %arg159: tensor<1x128x1x1xbf16>, %arg160: tensor<1x128x1x1xbf16>, %arg161: tensor<1x128x1x1xbf16>, %arg162: tensor<1x512x1x1xbf16>, %arg163: tensor<1x512x1x1xbf16>, %arg164: tensor<1x128x1x1xbf16>, %arg165: tensor<1x128x1x1xbf16>, %arg166: tensor<1x128x1x1xbf16>, %arg167: tensor<1x128x1x1xbf16>, %arg168: tensor<1x512x1x1xbf16>, %arg169: tensor<1x512x1x1xbf16>, %arg170: tensor<1x256x1x1xbf16>, %arg171: tensor<1x256x1x1xbf16>, %arg172: tensor<1x256x1x1xbf16>, %arg173: tensor<1x256x1x1xbf16>, %arg174: tensor<1x1024x1x1xbf16>, %arg175: tensor<1x1024x1x1xbf16>, %arg176: tensor<1x1024x1x1xbf16>, %arg177: tensor<1x1024x1x1xbf16>, %arg178: tensor<1x256x1x1xbf16>, %arg179: tensor<1x256x1x1xbf16>, %arg180: tensor<1x256x1x1xbf16>, %arg181: tensor<1x256x1x1xbf16>, %arg182: tensor<1x1024x1x1xbf16>, %arg183: tensor<1x1024x1x1xbf16>, %arg184: tensor<1x256x1x1xbf16>, %arg185: tensor<1x256x1x1xbf16>, %arg186: tensor<1x256x1x1xbf16>, %arg187: tensor<1x256x1x1xbf16>, %arg188: tensor<1x1024x1x1xbf16>, %arg189: tensor<1x1024x1x1xbf16>, %arg190: tensor<1x256x1x1xbf16>, %arg191: tensor<1x256x1x1xbf16>, %arg192: tensor<1x256x1x1xbf16>, %arg193: tensor<1x256x1x1xbf16>, %arg194: tensor<1x1024x1x1xbf16>, %arg195: tensor<1x1024x1x1xbf16>, %arg196: tensor<1x256x1x1xbf16>, %arg197: tensor<1x256x1x1xbf16>, %arg198: tensor<1x256x1x1xbf16>, %arg199: tensor<1x256x1x1xbf16>, %arg200: tensor<1x1024x1x1xbf16>, %arg201: tensor<1x1024x1x1xbf16>, %arg202: tensor<1x256x1x1xbf16>, %arg203: tensor<1x256x1x1xbf16>, %arg204: tensor<1x256x1x1xbf16>, %arg205: tensor<1x256x1x1xbf16>, %arg206: tensor<1x1024x1x1xbf16>, %arg207: tensor<1x1024x1x1xbf16>, %arg208: tensor<1x512x1x1xbf16>, %arg209: tensor<1x512x1x1xbf16>, %arg210: tensor<1x512x1x1xbf16>, %arg211: tensor<1x512x1x1xbf16>, %arg212: tensor<1x2048x1x1xbf16>, %arg213: tensor<1x2048x1x1xbf16>, %arg214: tensor<1x2048x1x1xbf16>, %arg215: tensor<1x2048x1x1xbf16>, %arg216: tensor<1x512x1x1xbf16>, %arg217: tensor<1x512x1x1xbf16>, %arg218: tensor<1x512x1x1xbf16>, %arg219: tensor<1x512x1x1xbf16>, %arg220: tensor<1x2048x1x1xbf16>, %arg221: tensor<1x2048x1x1xbf16>, %arg222: tensor<1x512x1x1xbf16>, %arg223: tensor<1x512x1x1xbf16>, %arg224: tensor<1x512x1x1xbf16>, %arg225: tensor<1x512x1x1xbf16>, %arg226: tensor<1x2048x1x1xbf16>, %arg227: tensor<1x2048x1x1xbf16>, %arg228: tensor<920x1x256xbf16>, %arg229: tensor<256x256xbf16>, %arg230: tensor<256xbf16>, %arg231: tensor<256x256xbf16>, %arg232: tensor<256xbf16>, %arg233: tensor<256x256xbf16>, %arg234: tensor<256xbf16>, %arg235: tensor<8x1x920xbf16>, %arg236: tensor<256x256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256x2048xf32>, %arg239: tensor<2048xf32>, %arg240: tensor<2048x256xf32>, %arg241: tensor<256xf32>, %arg242: tensor<256x256xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256x256xf32>, %arg245: tensor<256xf32>, %arg246: tensor<256x256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8x1x920xbf16>, %arg249: tensor<256x256xf32>, %arg250: tensor<256xf32>, %arg251: tensor<256x2048xf32>, %arg252: tensor<2048xf32>, %arg253: tensor<2048x256xf32>, %arg254: tensor<256xf32>, %arg255: tensor<256x256xf32>, %arg256: tensor<256xf32>, %arg257: tensor<256x256xf32>, %arg258: tensor<256xf32>, %arg259: tensor<256x256xf32>, %arg260: tensor<256xf32>, %arg261: tensor<8x1x920xbf16>, %arg262: tensor<256x256xf32>, %arg263: tensor<256xf32>, %arg264: tensor<256x2048xf32>, %arg265: tensor<2048xf32>, %arg266: tensor<2048x256xf32>, %arg267: tensor<256xf32>, %arg268: tensor<256x256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256x256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256x256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<8x1x920xbf16>, %arg275: tensor<256x256xf32>, %arg276: tensor<256xf32>, %arg277: tensor<256x2048xf32>, %arg278: tensor<2048xf32>, %arg279: tensor<2048x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256x256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256x256xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256x256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<8x1x920xbf16>, %arg288: tensor<256x256xf32>, %arg289: tensor<256xf32>, %arg290: tensor<256x2048xf32>, %arg291: tensor<2048xf32>, %arg292: tensor<2048x256xf32>, %arg293: tensor<256xf32>, %arg294: tensor<256x256xf32>, %arg295: tensor<256xf32>, %arg296: tensor<256x256xf32>, %arg297: tensor<256xf32>, %arg298: tensor<256x256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<8x1x920xbf16>, %arg301: tensor<256x256xf32>, %arg302: tensor<256xf32>, %arg303: tensor<256x2048xf32>, %arg304: tensor<2048xf32>, %arg305: tensor<2048x256xf32>, %arg306: tensor<256xf32>, %arg307: tensor<256x256xf32>, %arg308: tensor<256xf32>, %arg309: tensor<256x256xf32>, %arg310: tensor<256xf32>, %arg311: tensor<8x1x920xbf16>, %arg312: tensor<8x100x32xbf16>, %arg313: tensor<256x256xf32>, %arg314: tensor<256xf32>, %arg315: tensor<100x1x256xbf16>, %arg316: tensor<256x2048xf32>, %arg317: tensor<2048xf32>, %arg318: tensor<2048x256xf32>, %arg319: tensor<256xf32>, %arg320: tensor<100x1x256xbf16>, %arg321: tensor<256x256xf32>, %arg322: tensor<256xf32>, %arg323: tensor<256x256xf32>, %arg324: tensor<256xf32>, %arg325: tensor<256x256xf32>, %arg326: tensor<256xf32>, %arg327: tensor<256x256xf32>, %arg328: tensor<256xf32>, %arg329: tensor<256x256xf32>, %arg330: tensor<256xf32>, %arg331: tensor<256x256xf32>, %arg332: tensor<256xf32>, %arg333: tensor<256x256xf32>, %arg334: tensor<256xf32>, %arg335: tensor<8x1x920xbf16>, %arg336: tensor<256x256xf32>, %arg337: tensor<256xf32>, %arg338: tensor<256x2048xf32>, %arg339: tensor<2048xf32>, %arg340: tensor<2048x256xf32>, %arg341: tensor<256xf32>, %arg342: tensor<256x256xf32>, %arg343: tensor<256xf32>, %arg344: tensor<256x256xf32>, %arg345: tensor<256xf32>, %arg346: tensor<256x256xf32>, %arg347: tensor<256xf32>, %arg348: tensor<256x256xf32>, %arg349: tensor<256xf32>, %arg350: tensor<256x256xf32>, %arg351: tensor<256xf32>, %arg352: tensor<256x256xf32>, %arg353: tensor<256xf32>, %arg354: tensor<256x256xf32>, %arg355: tensor<256xf32>, %arg356: tensor<8x1x920xbf16>, %arg357: tensor<256x256xf32>, %arg358: tensor<256xf32>, %arg359: tensor<256x2048xf32>, %arg360: tensor<2048xf32>, %arg361: tensor<2048x256xf32>, %arg362: tensor<256xf32>, %arg363: tensor<256x256xf32>, %arg364: tensor<256xf32>, %arg365: tensor<256x256xf32>, %arg366: tensor<256xf32>, %arg367: tensor<256x256xf32>, %arg368: tensor<256xf32>, %arg369: tensor<256x256xf32>, %arg370: tensor<256xf32>, %arg371: tensor<256x256xf32>, %arg372: tensor<256xf32>, %arg373: tensor<256x256xf32>, %arg374: tensor<256xf32>, %arg375: tensor<256x256xf32>, %arg376: tensor<256xf32>, %arg377: tensor<8x1x920xbf16>, %arg378: tensor<256x256xf32>, %arg379: tensor<256xf32>, %arg380: tensor<256x2048xf32>, %arg381: tensor<2048xf32>, %arg382: tensor<2048x256xf32>, %arg383: tensor<256xf32>, %arg384: tensor<256x256xf32>, %arg385: tensor<256xf32>, %arg386: tensor<256x256xf32>, %arg387: tensor<256xf32>, %arg388: tensor<256x256xf32>, %arg389: tensor<256xf32>, %arg390: tensor<256x256xf32>, %arg391: tensor<256xf32>, %arg392: tensor<256x256xf32>, %arg393: tensor<256xf32>, %arg394: tensor<256x256xf32>, %arg395: tensor<256xf32>, %arg396: tensor<256x256xf32>, %arg397: tensor<256xf32>, %arg398: tensor<8x1x920xbf16>, %arg399: tensor<256x256xf32>, %arg400: tensor<256xf32>, %arg401: tensor<256x2048xf32>, %arg402: tensor<2048xf32>, %arg403: tensor<2048x256xf32>, %arg404: tensor<256xf32>, %arg405: tensor<256x256xf32>, %arg406: tensor<256xf32>, %arg407: tensor<256x256xf32>, %arg408: tensor<256xf32>, %arg409: tensor<256x256xf32>, %arg410: tensor<256xf32>, %arg411: tensor<256x256xf32>, %arg412: tensor<256xf32>, %arg413: tensor<256x256xf32>, %arg414: tensor<256xf32>, %arg415: tensor<256x256xf32>, %arg416: tensor<256xf32>, %arg417: tensor<256x256xf32>, %arg418: tensor<256xf32>, %arg419: tensor<8x1x920xbf16>, %arg420: tensor<256x256xf32>, %arg421: tensor<256xf32>, %arg422: tensor<256x2048xf32>, %arg423: tensor<2048xf32>, %arg424: tensor<2048x256xf32>, %arg425: tensor<256xf32>, %arg426: tensor<256x92xbf16>, %arg427: tensor<256x256xbf16>, %arg428: tensor<256x256xbf16>, %arg429: tensor<256x4xbf16>) -> (tensor<1x100x92xbf16>, tensor<1x100x4xbf16>) {
    %c = stablehlo.constant dense<0> : tensor<1x1xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x64x360x640xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x180x320xbf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x180x320xbf16>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x180x320xbf16>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x90x160xbf16>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x90x160xbf16>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x90x160xbf16>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x45x80xbf16>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<1x1024x45x80xbf16>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x45x80xbf16>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x23x40xbf16>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<1x2048x23x40xbf16>
    %cst_12 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<920x1x2048xbf16>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<100x1x2048xbf16>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<6x1x100x256xbf16>
    %cst_18 = arith.constant dense<0.17677669529663689> : tensor<1xf64>
    %cst_19 = arith.constant dense<1> : tensor<1xi64>
    %cst_20 = arith.constant dense<256> : tensor<1xi64>
    %cst_21 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %0 = stablehlo.reshape %arg0 : (tensor<1x3x720x1280xbf16>) -> tensor<3x720x1280xbf16>
    %1 = stablehlo.reshape %0 : (tensor<3x720x1280xbf16>) -> tensor<1x3x720x1280xbf16>
    %2 = "stablehlo.scatter"(%arg121, %c, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg430: tensor<bf16>, %arg431: tensor<bf16>):
      stablehlo.return %arg431 : tensor<bf16>
    }) : (tensor<1x3x720x1280xbf16>, tensor<1x1xi64>, tensor<1x3x720x1280xbf16>) -> tensor<1x3x720x1280xbf16>
    %3 = stablehlo.convolution(%2, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x720x1280xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x360x640xbf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3] : (tensor<1x64x360x640xbf16>) -> tensor<1x64x360x640xbf16>
    %5 = stablehlo.broadcast_in_dim %arg122, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x360x640xbf16>
    %6 = stablehlo.multiply %4, %5 : tensor<1x64x360x640xbf16>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<1x64x360x640xbf16>) -> tensor<1x64x360x640xbf16>
    %8 = stablehlo.broadcast_in_dim %arg123, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x360x640xbf16>
    %9 = stablehlo.add %7, %8 : tensor<1x64x360x640xbf16>
    %10 = stablehlo.maximum %9, %cst : tensor<1x64x360x640xbf16>
    %11 = "stablehlo.reduce_window"(%10, %cst_0) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg430: tensor<bf16>, %arg431: tensor<bf16>):
      %3230 = stablehlo.maximum %arg430, %arg431 : tensor<bf16>
      stablehlo.return %3230 : tensor<bf16>
    }) : (tensor<1x64x360x640xbf16>, tensor<bf16>) -> tensor<1x64x180x320xbf16>
    %12 = stablehlo.convolution(%11, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<64x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %14 = stablehlo.broadcast_in_dim %arg124, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %15 = stablehlo.multiply %13, %14 : tensor<1x64x180x320xbf16>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %17 = stablehlo.broadcast_in_dim %arg125, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %18 = stablehlo.add %16, %17 : tensor<1x64x180x320xbf16>
    %19 = stablehlo.maximum %18, %cst_1 : tensor<1x64x180x320xbf16>
    %20 = stablehlo.convolution(%19, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x180x320xbf16>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %22 = stablehlo.broadcast_in_dim %arg126, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %23 = stablehlo.multiply %21, %22 : tensor<1x64x180x320xbf16>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %25 = stablehlo.broadcast_in_dim %arg127, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %26 = stablehlo.add %24, %25 : tensor<1x64x180x320xbf16>
    %27 = stablehlo.maximum %26, %cst_1 : tensor<1x64x180x320xbf16>
    %28 = stablehlo.convolution(%27, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %30 = stablehlo.broadcast_in_dim %arg128, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %31 = stablehlo.multiply %29, %30 : tensor<1x256x180x320xbf16>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %33 = stablehlo.broadcast_in_dim %arg129, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %34 = stablehlo.add %32, %33 : tensor<1x256x180x320xbf16>
    %35 = stablehlo.convolution(%11, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %37 = stablehlo.broadcast_in_dim %arg130, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %38 = stablehlo.multiply %36, %37 : tensor<1x256x180x320xbf16>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %40 = stablehlo.broadcast_in_dim %arg131, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %41 = stablehlo.add %39, %40 : tensor<1x256x180x320xbf16>
    %42 = stablehlo.add %34, %41 : tensor<1x256x180x320xbf16>
    %43 = stablehlo.maximum %42, %cst_2 : tensor<1x256x180x320xbf16>
    %44 = stablehlo.convolution(%43, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x180x320xbf16>, tensor<64x256x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %46 = stablehlo.broadcast_in_dim %arg132, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %47 = stablehlo.multiply %45, %46 : tensor<1x64x180x320xbf16>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %49 = stablehlo.broadcast_in_dim %arg133, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %50 = stablehlo.add %48, %49 : tensor<1x64x180x320xbf16>
    %51 = stablehlo.maximum %50, %cst_1 : tensor<1x64x180x320xbf16>
    %52 = stablehlo.convolution(%51, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x180x320xbf16>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %54 = stablehlo.broadcast_in_dim %arg134, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %55 = stablehlo.multiply %53, %54 : tensor<1x64x180x320xbf16>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %57 = stablehlo.broadcast_in_dim %arg135, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %58 = stablehlo.add %56, %57 : tensor<1x64x180x320xbf16>
    %59 = stablehlo.maximum %58, %cst_1 : tensor<1x64x180x320xbf16>
    %60 = stablehlo.convolution(%59, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %62 = stablehlo.broadcast_in_dim %arg136, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %63 = stablehlo.multiply %61, %62 : tensor<1x256x180x320xbf16>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %65 = stablehlo.broadcast_in_dim %arg137, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %66 = stablehlo.add %64, %65 : tensor<1x256x180x320xbf16>
    %67 = stablehlo.add %66, %43 : tensor<1x256x180x320xbf16>
    %68 = stablehlo.maximum %67, %cst_2 : tensor<1x256x180x320xbf16>
    %69 = stablehlo.convolution(%68, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x180x320xbf16>, tensor<64x256x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %71 = stablehlo.broadcast_in_dim %arg138, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %72 = stablehlo.multiply %70, %71 : tensor<1x64x180x320xbf16>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %74 = stablehlo.broadcast_in_dim %arg139, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %75 = stablehlo.add %73, %74 : tensor<1x64x180x320xbf16>
    %76 = stablehlo.maximum %75, %cst_1 : tensor<1x64x180x320xbf16>
    %77 = stablehlo.convolution(%76, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x64x180x320xbf16>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %79 = stablehlo.broadcast_in_dim %arg140, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %80 = stablehlo.multiply %78, %79 : tensor<1x64x180x320xbf16>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<1x64x180x320xbf16>) -> tensor<1x64x180x320xbf16>
    %82 = stablehlo.broadcast_in_dim %arg141, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xbf16>) -> tensor<1x64x180x320xbf16>
    %83 = stablehlo.add %81, %82 : tensor<1x64x180x320xbf16>
    %84 = stablehlo.maximum %83, %cst_1 : tensor<1x64x180x320xbf16>
    %85 = stablehlo.convolution(%84, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x180x320xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %87 = stablehlo.broadcast_in_dim %arg142, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %88 = stablehlo.multiply %86, %87 : tensor<1x256x180x320xbf16>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x256x180x320xbf16>) -> tensor<1x256x180x320xbf16>
    %90 = stablehlo.broadcast_in_dim %arg143, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x180x320xbf16>
    %91 = stablehlo.add %89, %90 : tensor<1x256x180x320xbf16>
    %92 = stablehlo.add %91, %68 : tensor<1x256x180x320xbf16>
    %93 = stablehlo.maximum %92, %cst_2 : tensor<1x256x180x320xbf16>
    %94 = stablehlo.convolution(%93, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x180x320xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x128x180x320xbf16>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x128x180x320xbf16>) -> tensor<1x128x180x320xbf16>
    %96 = stablehlo.broadcast_in_dim %arg144, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x180x320xbf16>
    %97 = stablehlo.multiply %95, %96 : tensor<1x128x180x320xbf16>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x128x180x320xbf16>) -> tensor<1x128x180x320xbf16>
    %99 = stablehlo.broadcast_in_dim %arg145, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x180x320xbf16>
    %100 = stablehlo.add %98, %99 : tensor<1x128x180x320xbf16>
    %101 = stablehlo.maximum %100, %cst_3 : tensor<1x128x180x320xbf16>
    %102 = stablehlo.convolution(%101, %arg13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x180x320xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %104 = stablehlo.broadcast_in_dim %arg146, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %105 = stablehlo.multiply %103, %104 : tensor<1x128x90x160xbf16>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %107 = stablehlo.broadcast_in_dim %arg147, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %108 = stablehlo.add %106, %107 : tensor<1x128x90x160xbf16>
    %109 = stablehlo.maximum %108, %cst_4 : tensor<1x128x90x160xbf16>
    %110 = stablehlo.convolution(%109, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %112 = stablehlo.broadcast_in_dim %arg148, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %113 = stablehlo.multiply %111, %112 : tensor<1x512x90x160xbf16>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %115 = stablehlo.broadcast_in_dim %arg149, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %116 = stablehlo.add %114, %115 : tensor<1x512x90x160xbf16>
    %117 = stablehlo.convolution(%93, %arg15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x180x320xbf16>, tensor<512x256x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %119 = stablehlo.broadcast_in_dim %arg150, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %120 = stablehlo.multiply %118, %119 : tensor<1x512x90x160xbf16>
    %121 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %122 = stablehlo.broadcast_in_dim %arg151, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %123 = stablehlo.add %121, %122 : tensor<1x512x90x160xbf16>
    %124 = stablehlo.add %116, %123 : tensor<1x512x90x160xbf16>
    %125 = stablehlo.maximum %124, %cst_5 : tensor<1x512x90x160xbf16>
    %126 = stablehlo.convolution(%125, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x90x160xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %128 = stablehlo.broadcast_in_dim %arg152, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %129 = stablehlo.multiply %127, %128 : tensor<1x128x90x160xbf16>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %131 = stablehlo.broadcast_in_dim %arg153, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %132 = stablehlo.add %130, %131 : tensor<1x128x90x160xbf16>
    %133 = stablehlo.maximum %132, %cst_4 : tensor<1x128x90x160xbf16>
    %134 = stablehlo.convolution(%133, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %136 = stablehlo.broadcast_in_dim %arg154, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %137 = stablehlo.multiply %135, %136 : tensor<1x128x90x160xbf16>
    %138 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %139 = stablehlo.broadcast_in_dim %arg155, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %140 = stablehlo.add %138, %139 : tensor<1x128x90x160xbf16>
    %141 = stablehlo.maximum %140, %cst_4 : tensor<1x128x90x160xbf16>
    %142 = stablehlo.convolution(%141, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %144 = stablehlo.broadcast_in_dim %arg156, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %145 = stablehlo.multiply %143, %144 : tensor<1x512x90x160xbf16>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %147 = stablehlo.broadcast_in_dim %arg157, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %148 = stablehlo.add %146, %147 : tensor<1x512x90x160xbf16>
    %149 = stablehlo.add %148, %125 : tensor<1x512x90x160xbf16>
    %150 = stablehlo.maximum %149, %cst_5 : tensor<1x512x90x160xbf16>
    %151 = stablehlo.convolution(%150, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x90x160xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %153 = stablehlo.broadcast_in_dim %arg158, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %154 = stablehlo.multiply %152, %153 : tensor<1x128x90x160xbf16>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %156 = stablehlo.broadcast_in_dim %arg159, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %157 = stablehlo.add %155, %156 : tensor<1x128x90x160xbf16>
    %158 = stablehlo.maximum %157, %cst_4 : tensor<1x128x90x160xbf16>
    %159 = stablehlo.convolution(%158, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %161 = stablehlo.broadcast_in_dim %arg160, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %162 = stablehlo.multiply %160, %161 : tensor<1x128x90x160xbf16>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %164 = stablehlo.broadcast_in_dim %arg161, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %165 = stablehlo.add %163, %164 : tensor<1x128x90x160xbf16>
    %166 = stablehlo.maximum %165, %cst_4 : tensor<1x128x90x160xbf16>
    %167 = stablehlo.convolution(%166, %arg21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %169 = stablehlo.broadcast_in_dim %arg162, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %170 = stablehlo.multiply %168, %169 : tensor<1x512x90x160xbf16>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %172 = stablehlo.broadcast_in_dim %arg163, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %173 = stablehlo.add %171, %172 : tensor<1x512x90x160xbf16>
    %174 = stablehlo.add %173, %150 : tensor<1x512x90x160xbf16>
    %175 = stablehlo.maximum %174, %cst_5 : tensor<1x512x90x160xbf16>
    %176 = stablehlo.convolution(%175, %arg22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x90x160xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %177 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %178 = stablehlo.broadcast_in_dim %arg164, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %179 = stablehlo.multiply %177, %178 : tensor<1x128x90x160xbf16>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %181 = stablehlo.broadcast_in_dim %arg165, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %182 = stablehlo.add %180, %181 : tensor<1x128x90x160xbf16>
    %183 = stablehlo.maximum %182, %cst_4 : tensor<1x128x90x160xbf16>
    %184 = stablehlo.convolution(%183, %arg23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %186 = stablehlo.broadcast_in_dim %arg166, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %187 = stablehlo.multiply %185, %186 : tensor<1x128x90x160xbf16>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    %189 = stablehlo.broadcast_in_dim %arg167, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xbf16>) -> tensor<1x128x90x160xbf16>
    %190 = stablehlo.add %188, %189 : tensor<1x128x90x160xbf16>
    %191 = stablehlo.maximum %190, %cst_4 : tensor<1x128x90x160xbf16>
    %192 = stablehlo.convolution(%191, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x90x160xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %194 = stablehlo.broadcast_in_dim %arg168, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %195 = stablehlo.multiply %193, %194 : tensor<1x512x90x160xbf16>
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2, 3] : (tensor<1x512x90x160xbf16>) -> tensor<1x512x90x160xbf16>
    %197 = stablehlo.broadcast_in_dim %arg169, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x90x160xbf16>
    %198 = stablehlo.add %196, %197 : tensor<1x512x90x160xbf16>
    %199 = stablehlo.add %198, %175 : tensor<1x512x90x160xbf16>
    %200 = stablehlo.maximum %199, %cst_5 : tensor<1x512x90x160xbf16>
    %201 = stablehlo.convolution(%200, %arg25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x90x160xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x256x90x160xbf16>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 1, 2, 3] : (tensor<1x256x90x160xbf16>) -> tensor<1x256x90x160xbf16>
    %203 = stablehlo.broadcast_in_dim %arg170, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x90x160xbf16>
    %204 = stablehlo.multiply %202, %203 : tensor<1x256x90x160xbf16>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2, 3] : (tensor<1x256x90x160xbf16>) -> tensor<1x256x90x160xbf16>
    %206 = stablehlo.broadcast_in_dim %arg171, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x90x160xbf16>
    %207 = stablehlo.add %205, %206 : tensor<1x256x90x160xbf16>
    %208 = stablehlo.maximum %207, %cst_6 : tensor<1x256x90x160xbf16>
    %209 = stablehlo.convolution(%208, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x90x160xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x45x80xbf16>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %211 = stablehlo.broadcast_in_dim %arg172, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %212 = stablehlo.multiply %210, %211 : tensor<1x256x45x80xbf16>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %214 = stablehlo.broadcast_in_dim %arg173, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %215 = stablehlo.add %213, %214 : tensor<1x256x45x80xbf16>
    %216 = stablehlo.maximum %215, %cst_7 : tensor<1x256x45x80xbf16>
    %217 = stablehlo.convolution(%216, %arg27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %218 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %219 = stablehlo.broadcast_in_dim %arg174, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %220 = stablehlo.multiply %218, %219 : tensor<1x1024x45x80xbf16>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %222 = stablehlo.broadcast_in_dim %arg175, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %223 = stablehlo.add %221, %222 : tensor<1x1024x45x80xbf16>
    %224 = stablehlo.convolution(%200, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x90x160xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %225 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %226 = stablehlo.broadcast_in_dim %arg176, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %227 = stablehlo.multiply %225, %226 : tensor<1x1024x45x80xbf16>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %229 = stablehlo.broadcast_in_dim %arg177, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %230 = stablehlo.add %228, %229 : tensor<1x1024x45x80xbf16>
    %231 = stablehlo.add %223, %230 : tensor<1x1024x45x80xbf16>
    %232 = stablehlo.maximum %231, %cst_8 : tensor<1x1024x45x80xbf16>
    %233 = stablehlo.convolution(%232, %arg29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %234 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %235 = stablehlo.broadcast_in_dim %arg178, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %236 = stablehlo.multiply %234, %235 : tensor<1x256x45x80xbf16>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %238 = stablehlo.broadcast_in_dim %arg179, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %239 = stablehlo.add %237, %238 : tensor<1x256x45x80xbf16>
    %240 = stablehlo.maximum %239, %cst_7 : tensor<1x256x45x80xbf16>
    %241 = stablehlo.convolution(%240, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x45x80xbf16>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %243 = stablehlo.broadcast_in_dim %arg180, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %244 = stablehlo.multiply %242, %243 : tensor<1x256x45x80xbf16>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %246 = stablehlo.broadcast_in_dim %arg181, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %247 = stablehlo.add %245, %246 : tensor<1x256x45x80xbf16>
    %248 = stablehlo.maximum %247, %cst_7 : tensor<1x256x45x80xbf16>
    %249 = stablehlo.convolution(%248, %arg31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %251 = stablehlo.broadcast_in_dim %arg182, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %252 = stablehlo.multiply %250, %251 : tensor<1x1024x45x80xbf16>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %254 = stablehlo.broadcast_in_dim %arg183, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %255 = stablehlo.add %253, %254 : tensor<1x1024x45x80xbf16>
    %256 = stablehlo.add %255, %232 : tensor<1x1024x45x80xbf16>
    %257 = stablehlo.maximum %256, %cst_8 : tensor<1x1024x45x80xbf16>
    %258 = stablehlo.convolution(%257, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %260 = stablehlo.broadcast_in_dim %arg184, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %261 = stablehlo.multiply %259, %260 : tensor<1x256x45x80xbf16>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %263 = stablehlo.broadcast_in_dim %arg185, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %264 = stablehlo.add %262, %263 : tensor<1x256x45x80xbf16>
    %265 = stablehlo.maximum %264, %cst_7 : tensor<1x256x45x80xbf16>
    %266 = stablehlo.convolution(%265, %arg33) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x45x80xbf16>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %268 = stablehlo.broadcast_in_dim %arg186, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %269 = stablehlo.multiply %267, %268 : tensor<1x256x45x80xbf16>
    %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %271 = stablehlo.broadcast_in_dim %arg187, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %272 = stablehlo.add %270, %271 : tensor<1x256x45x80xbf16>
    %273 = stablehlo.maximum %272, %cst_7 : tensor<1x256x45x80xbf16>
    %274 = stablehlo.convolution(%273, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %276 = stablehlo.broadcast_in_dim %arg188, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %277 = stablehlo.multiply %275, %276 : tensor<1x1024x45x80xbf16>
    %278 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %279 = stablehlo.broadcast_in_dim %arg189, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %280 = stablehlo.add %278, %279 : tensor<1x1024x45x80xbf16>
    %281 = stablehlo.add %280, %257 : tensor<1x1024x45x80xbf16>
    %282 = stablehlo.maximum %281, %cst_8 : tensor<1x1024x45x80xbf16>
    %283 = stablehlo.convolution(%282, %arg35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %284 = stablehlo.broadcast_in_dim %283, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %285 = stablehlo.broadcast_in_dim %arg190, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %286 = stablehlo.multiply %284, %285 : tensor<1x256x45x80xbf16>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %288 = stablehlo.broadcast_in_dim %arg191, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %289 = stablehlo.add %287, %288 : tensor<1x256x45x80xbf16>
    %290 = stablehlo.maximum %289, %cst_7 : tensor<1x256x45x80xbf16>
    %291 = stablehlo.convolution(%290, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x45x80xbf16>
    %292 = stablehlo.broadcast_in_dim %291, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %293 = stablehlo.broadcast_in_dim %arg192, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %294 = stablehlo.multiply %292, %293 : tensor<1x256x45x80xbf16>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %296 = stablehlo.broadcast_in_dim %arg193, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %297 = stablehlo.add %295, %296 : tensor<1x256x45x80xbf16>
    %298 = stablehlo.maximum %297, %cst_7 : tensor<1x256x45x80xbf16>
    %299 = stablehlo.convolution(%298, %arg37) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %301 = stablehlo.broadcast_in_dim %arg194, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %302 = stablehlo.multiply %300, %301 : tensor<1x1024x45x80xbf16>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %304 = stablehlo.broadcast_in_dim %arg195, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %305 = stablehlo.add %303, %304 : tensor<1x1024x45x80xbf16>
    %306 = stablehlo.add %305, %282 : tensor<1x1024x45x80xbf16>
    %307 = stablehlo.maximum %306, %cst_8 : tensor<1x1024x45x80xbf16>
    %308 = stablehlo.convolution(%307, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %310 = stablehlo.broadcast_in_dim %arg196, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %311 = stablehlo.multiply %309, %310 : tensor<1x256x45x80xbf16>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %313 = stablehlo.broadcast_in_dim %arg197, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %314 = stablehlo.add %312, %313 : tensor<1x256x45x80xbf16>
    %315 = stablehlo.maximum %314, %cst_7 : tensor<1x256x45x80xbf16>
    %316 = stablehlo.convolution(%315, %arg39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x45x80xbf16>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %318 = stablehlo.broadcast_in_dim %arg198, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %319 = stablehlo.multiply %317, %318 : tensor<1x256x45x80xbf16>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %321 = stablehlo.broadcast_in_dim %arg199, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %322 = stablehlo.add %320, %321 : tensor<1x256x45x80xbf16>
    %323 = stablehlo.maximum %322, %cst_7 : tensor<1x256x45x80xbf16>
    %324 = stablehlo.convolution(%323, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %326 = stablehlo.broadcast_in_dim %arg200, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %327 = stablehlo.multiply %325, %326 : tensor<1x1024x45x80xbf16>
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %329 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %330 = stablehlo.add %328, %329 : tensor<1x1024x45x80xbf16>
    %331 = stablehlo.add %330, %307 : tensor<1x1024x45x80xbf16>
    %332 = stablehlo.maximum %331, %cst_8 : tensor<1x1024x45x80xbf16>
    %333 = stablehlo.convolution(%332, %arg41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %335 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %336 = stablehlo.multiply %334, %335 : tensor<1x256x45x80xbf16>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %338 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %339 = stablehlo.add %337, %338 : tensor<1x256x45x80xbf16>
    %340 = stablehlo.maximum %339, %cst_7 : tensor<1x256x45x80xbf16>
    %341 = stablehlo.convolution(%340, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x256x45x80xbf16>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %343 = stablehlo.broadcast_in_dim %arg204, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %344 = stablehlo.multiply %342, %343 : tensor<1x256x45x80xbf16>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x256x45x80xbf16>) -> tensor<1x256x45x80xbf16>
    %346 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xbf16>) -> tensor<1x256x45x80xbf16>
    %347 = stablehlo.add %345, %346 : tensor<1x256x45x80xbf16>
    %348 = stablehlo.maximum %347, %cst_7 : tensor<1x256x45x80xbf16>
    %349 = stablehlo.convolution(%348, %arg43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x45x80xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %351 = stablehlo.broadcast_in_dim %arg206, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %352 = stablehlo.multiply %350, %351 : tensor<1x1024x45x80xbf16>
    %353 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2, 3] : (tensor<1x1024x45x80xbf16>) -> tensor<1x1024x45x80xbf16>
    %354 = stablehlo.broadcast_in_dim %arg207, dims = [0, 1, 2, 3] : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x45x80xbf16>
    %355 = stablehlo.add %353, %354 : tensor<1x1024x45x80xbf16>
    %356 = stablehlo.add %355, %332 : tensor<1x1024x45x80xbf16>
    %357 = stablehlo.maximum %356, %cst_8 : tensor<1x1024x45x80xbf16>
    %358 = stablehlo.convolution(%357, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x45x80xbf16>
    %359 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2, 3] : (tensor<1x512x45x80xbf16>) -> tensor<1x512x45x80xbf16>
    %360 = stablehlo.broadcast_in_dim %arg208, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x45x80xbf16>
    %361 = stablehlo.multiply %359, %360 : tensor<1x512x45x80xbf16>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x512x45x80xbf16>) -> tensor<1x512x45x80xbf16>
    %363 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x45x80xbf16>
    %364 = stablehlo.add %362, %363 : tensor<1x512x45x80xbf16>
    %365 = stablehlo.maximum %364, %cst_9 : tensor<1x512x45x80xbf16>
    %366 = stablehlo.convolution(%365, %arg45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x45x80xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x23x40xbf16>
    %367 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %368 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %369 = stablehlo.multiply %367, %368 : tensor<1x512x23x40xbf16>
    %370 = stablehlo.broadcast_in_dim %369, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %371 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %372 = stablehlo.add %370, %371 : tensor<1x512x23x40xbf16>
    %373 = stablehlo.maximum %372, %cst_10 : tensor<1x512x23x40xbf16>
    %374 = stablehlo.convolution(%373, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x23x40xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %375 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %376 = stablehlo.broadcast_in_dim %arg212, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %377 = stablehlo.multiply %375, %376 : tensor<1x2048x23x40xbf16>
    %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %379 = stablehlo.broadcast_in_dim %arg213, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %380 = stablehlo.add %378, %379 : tensor<1x2048x23x40xbf16>
    %381 = stablehlo.convolution(%357, %arg47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x45x80xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %383 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %384 = stablehlo.multiply %382, %383 : tensor<1x2048x23x40xbf16>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %386 = stablehlo.broadcast_in_dim %arg215, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %387 = stablehlo.add %385, %386 : tensor<1x2048x23x40xbf16>
    %388 = stablehlo.add %380, %387 : tensor<1x2048x23x40xbf16>
    %389 = stablehlo.maximum %388, %cst_11 : tensor<1x2048x23x40xbf16>
    %390 = stablehlo.convolution(%389, %arg48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x23x40xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %391 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %392 = stablehlo.broadcast_in_dim %arg216, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %393 = stablehlo.multiply %391, %392 : tensor<1x512x23x40xbf16>
    %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %395 = stablehlo.broadcast_in_dim %arg217, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %396 = stablehlo.add %394, %395 : tensor<1x512x23x40xbf16>
    %397 = stablehlo.maximum %396, %cst_10 : tensor<1x512x23x40xbf16>
    %398 = stablehlo.convolution(%397, %arg49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x23x40xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x23x40xbf16>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %400 = stablehlo.broadcast_in_dim %arg218, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %401 = stablehlo.multiply %399, %400 : tensor<1x512x23x40xbf16>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %403 = stablehlo.broadcast_in_dim %arg219, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %404 = stablehlo.add %402, %403 : tensor<1x512x23x40xbf16>
    %405 = stablehlo.maximum %404, %cst_10 : tensor<1x512x23x40xbf16>
    %406 = stablehlo.convolution(%405, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x23x40xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %407 = stablehlo.broadcast_in_dim %406, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %408 = stablehlo.broadcast_in_dim %arg220, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %409 = stablehlo.multiply %407, %408 : tensor<1x2048x23x40xbf16>
    %410 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %411 = stablehlo.broadcast_in_dim %arg221, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %412 = stablehlo.add %410, %411 : tensor<1x2048x23x40xbf16>
    %413 = stablehlo.add %412, %389 : tensor<1x2048x23x40xbf16>
    %414 = stablehlo.maximum %413, %cst_11 : tensor<1x2048x23x40xbf16>
    %415 = stablehlo.convolution(%414, %arg51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x23x40xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %417 = stablehlo.broadcast_in_dim %arg222, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %418 = stablehlo.multiply %416, %417 : tensor<1x512x23x40xbf16>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %420 = stablehlo.broadcast_in_dim %arg223, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %421 = stablehlo.add %419, %420 : tensor<1x512x23x40xbf16>
    %422 = stablehlo.maximum %421, %cst_10 : tensor<1x512x23x40xbf16>
    %423 = stablehlo.convolution(%422, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x23x40xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x23x40xbf16>
    %424 = stablehlo.broadcast_in_dim %423, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %425 = stablehlo.broadcast_in_dim %arg224, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %426 = stablehlo.multiply %424, %425 : tensor<1x512x23x40xbf16>
    %427 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x512x23x40xbf16>) -> tensor<1x512x23x40xbf16>
    %428 = stablehlo.broadcast_in_dim %arg225, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xbf16>) -> tensor<1x512x23x40xbf16>
    %429 = stablehlo.add %427, %428 : tensor<1x512x23x40xbf16>
    %430 = stablehlo.maximum %429, %cst_10 : tensor<1x512x23x40xbf16>
    %431 = stablehlo.convolution(%430, %arg53) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x23x40xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %432 = stablehlo.broadcast_in_dim %431, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %433 = stablehlo.broadcast_in_dim %arg226, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %434 = stablehlo.multiply %432, %433 : tensor<1x2048x23x40xbf16>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1, 2, 3] : (tensor<1x2048x23x40xbf16>) -> tensor<1x2048x23x40xbf16>
    %436 = stablehlo.broadcast_in_dim %arg227, dims = [0, 1, 2, 3] : (tensor<1x2048x1x1xbf16>) -> tensor<1x2048x23x40xbf16>
    %437 = stablehlo.add %435, %436 : tensor<1x2048x23x40xbf16>
    %438 = stablehlo.add %437, %414 : tensor<1x2048x23x40xbf16>
    %439 = stablehlo.maximum %438, %cst_11 : tensor<1x2048x23x40xbf16>
    %440 = stablehlo.convolution(%439, %arg54) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x23x40xbf16>, tensor<256x2048x1x1xbf16>) -> tensor<1x256x23x40xbf16>
    %441 = stablehlo.reshape %arg55 : (tensor<256xbf16>) -> tensor<256x1x1xbf16>
    %442 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x256x23x40xbf16>) -> tensor<1x256x23x40xbf16>
    %443 = stablehlo.broadcast_in_dim %441, dims = [1, 2, 3] : (tensor<256x1x1xbf16>) -> tensor<1x256x23x40xbf16>
    %444 = stablehlo.add %442, %443 : tensor<1x256x23x40xbf16>
    %445 = stablehlo.reshape %444 : (tensor<1x256x23x40xbf16>) -> tensor<1x256x920xbf16>
    %446 = stablehlo.transpose %445, dims = [2, 0, 1] : (tensor<1x256x920xbf16>) -> tensor<920x1x256xbf16>
    %447 = stablehlo.add %446, %arg228 : tensor<920x1x256xbf16>
    %448 = stablehlo.reshape %447 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %449 = stablehlo.dot_general %448, %arg229, contracting_dims = [1] x [0] : (tensor<920x256xbf16>, tensor<256x256xbf16>) -> tensor<920x256xbf16>
    %450 = stablehlo.reshape %449 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %451 = stablehlo.broadcast_in_dim %450, dims = [0, 1, 2] : (tensor<920x1x256xbf16>) -> tensor<920x1x256xbf16>
    %452 = stablehlo.broadcast_in_dim %arg230, dims = [2] : (tensor<256xbf16>) -> tensor<920x1x256xbf16>
    %453 = stablehlo.add %451, %452 : tensor<920x1x256xbf16>
    %454 = stablehlo.reshape %453 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %455 = stablehlo.dot_general %448, %arg231, contracting_dims = [1] x [0] : (tensor<920x256xbf16>, tensor<256x256xbf16>) -> tensor<920x256xbf16>
    %456 = stablehlo.reshape %455 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2] : (tensor<920x1x256xbf16>) -> tensor<920x1x256xbf16>
    %458 = stablehlo.broadcast_in_dim %arg232, dims = [2] : (tensor<256xbf16>) -> tensor<920x1x256xbf16>
    %459 = stablehlo.add %457, %458 : tensor<920x1x256xbf16>
    %460 = stablehlo.reshape %459 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %461 = stablehlo.reshape %446 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %462 = stablehlo.dot_general %461, %arg233, contracting_dims = [1] x [0] : (tensor<920x256xbf16>, tensor<256x256xbf16>) -> tensor<920x256xbf16>
    %463 = stablehlo.reshape %462 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<920x1x256xbf16>) -> tensor<920x1x256xbf16>
    %465 = stablehlo.broadcast_in_dim %arg234, dims = [2] : (tensor<256xbf16>) -> tensor<920x1x256xbf16>
    %466 = stablehlo.add %464, %465 : tensor<920x1x256xbf16>
    %467 = stablehlo.reshape %466 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %468 = stablehlo.reshape %454 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %469 = stablehlo.reshape %468 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %470 = stablehlo.transpose %469, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %471 = stablehlo.convert %cst_18 : (tensor<1xf64>) -> tensor<1xbf16>
    %472 = stablehlo.reshape %471 : (tensor<1xbf16>) -> tensor<bf16>
    %473 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %474 = stablehlo.broadcast_in_dim %472, dims = [] : (tensor<bf16>) -> tensor<8x920x32xbf16>
    %475 = stablehlo.multiply %473, %474 : tensor<8x920x32xbf16>
    %476 = stablehlo.reshape %460 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %477 = stablehlo.reshape %476 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %478 = stablehlo.transpose %477, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %479 = stablehlo.transpose %478, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %481 = stablehlo.dot_general %475, %480, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x920x920xbf16>
    %482 = stablehlo.convert %cst_19 : (tensor<1xi64>) -> tensor<1xbf16>
    %483 = stablehlo.reshape %482 : (tensor<1xbf16>) -> tensor<bf16>
    %484 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %485 = stablehlo.broadcast_in_dim %483, dims = [] : (tensor<bf16>) -> tensor<8x920x920xbf16>
    %486 = stablehlo.multiply %484, %485 : tensor<8x920x920xbf16>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %488 = stablehlo.broadcast_in_dim %arg235, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x920x920xbf16>
    %489 = stablehlo.add %487, %488 : tensor<8x920x920xbf16>
    %490 = stablehlo.convert %489 : (tensor<8x920x920xbf16>) -> tensor<8x920x920xf32>
    %491 = stablehlo.reduce(%490 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %492 = stablehlo.reshape %491 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %493 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %494 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %495 = stablehlo.subtract %493, %494 : tensor<8x920x920xf32>
    %496 = stablehlo.exponential %495 : tensor<8x920x920xf32>
    %497 = stablehlo.reduce(%496 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %498 = stablehlo.reshape %497 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %499 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %500 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %501 = stablehlo.divide %499, %500 : tensor<8x920x920xf32>
    %502 = stablehlo.convert %501 : (tensor<8x920x920xf32>) -> tensor<8x920x920xbf16>
    %503 = stablehlo.reshape %467 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %504 = stablehlo.reshape %503 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %505 = stablehlo.transpose %504, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %506 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %507 = stablehlo.dot_general %502, %506, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %508 = stablehlo.transpose %507, dims = [1, 0, 2] : (tensor<8x920x32xbf16>) -> tensor<920x8x32xbf16>
    %509 = stablehlo.reshape %508 : (tensor<920x8x32xbf16>) -> tensor<920x256xbf16>
    %510 = stablehlo.convert %509 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %511 = stablehlo.dot_general %510, %arg236, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %512 = stablehlo.convert %cst_19 : (tensor<1xi64>) -> tensor<1xf32>
    %513 = stablehlo.reshape %512 : (tensor<1xf32>) -> tensor<f32>
    %514 = stablehlo.broadcast_in_dim %511, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %515 = stablehlo.broadcast_in_dim %513, dims = [] : (tensor<f32>) -> tensor<920x256xf32>
    %516 = stablehlo.multiply %514, %515 : tensor<920x256xf32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %518 = stablehlo.broadcast_in_dim %arg237, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %519 = stablehlo.add %517, %518 : tensor<920x256xf32>
    %520 = stablehlo.convert %519 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %521 = stablehlo.reshape %520 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %522 = stablehlo.add %446, %521 : tensor<920x1x256xbf16>
    %523 = stablehlo.convert %522 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %524 = stablehlo.convert %523 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %525 = stablehlo.reduce(%524 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %526 = stablehlo.reshape %525 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %527 = stablehlo.convert %cst_20 : (tensor<1xi64>) -> tensor<1xf64>
    %528 = stablehlo.reshape %527 : (tensor<1xf64>) -> tensor<f64>
    %529 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %530 = stablehlo.broadcast_in_dim %528, dims = [] : (tensor<f64>) -> tensor<920x1x1xf64>
    %531 = stablehlo.divide %529, %530 : tensor<920x1x1xf64>
    %532 = stablehlo.broadcast_in_dim %524, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %533 = stablehlo.broadcast_in_dim %531, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %534 = stablehlo.subtract %532, %533 : tensor<920x1x256xf64>
    %535 = stablehlo.multiply %534, %534 : tensor<920x1x256xf64>
    %536 = stablehlo.reduce(%535 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %537 = stablehlo.reshape %536 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %539 = stablehlo.divide %538, %530 : tensor<920x1x1xf64>
    %540 = stablehlo.convert %539 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %541 = stablehlo.reduce(%523 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %542 = stablehlo.reshape %541 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %543 = stablehlo.convert %cst_20 : (tensor<1xi64>) -> tensor<1xf32>
    %544 = stablehlo.reshape %543 : (tensor<1xf32>) -> tensor<f32>
    %545 = stablehlo.broadcast_in_dim %542, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %546 = stablehlo.broadcast_in_dim %544, dims = [] : (tensor<f32>) -> tensor<920x1x1xf32>
    %547 = stablehlo.divide %545, %546 : tensor<920x1x1xf32>
    %548 = stablehlo.convert %cst_21 : (tensor<1xf64>) -> tensor<1xf32>
    %549 = stablehlo.reshape %548 : (tensor<1xf32>) -> tensor<f32>
    %550 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %551 = stablehlo.broadcast_in_dim %549, dims = [] : (tensor<f32>) -> tensor<920x1x1xf32>
    %552 = stablehlo.add %550, %551 : tensor<920x1x1xf32>
    %553 = stablehlo.rsqrt %552 : tensor<920x1x1xf32>
    %554 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %555 = stablehlo.broadcast_in_dim %547, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %556 = stablehlo.subtract %554, %555 : tensor<920x1x256xf32>
    %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %558 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %559 = stablehlo.multiply %557, %558 : tensor<920x1x256xf32>
    %560 = stablehlo.convert %arg56 : (tensor<256xbf16>) -> tensor<256xf32>
    %561 = stablehlo.broadcast_in_dim %559, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %562 = stablehlo.broadcast_in_dim %560, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %563 = stablehlo.multiply %561, %562 : tensor<920x1x256xf32>
    %564 = stablehlo.convert %arg57 : (tensor<256xbf16>) -> tensor<256xf32>
    %565 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %566 = stablehlo.broadcast_in_dim %564, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %567 = stablehlo.add %565, %566 : tensor<920x1x256xf32>
    %568 = stablehlo.convert %567 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %569 = stablehlo.reshape %568 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %570 = stablehlo.convert %569 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %571 = stablehlo.dot_general %570, %arg238, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x2048xf32>) -> tensor<920x2048xf32>
    %572 = stablehlo.broadcast_in_dim %571, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %573 = stablehlo.broadcast_in_dim %513, dims = [] : (tensor<f32>) -> tensor<920x2048xf32>
    %574 = stablehlo.multiply %572, %573 : tensor<920x2048xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %576 = stablehlo.broadcast_in_dim %arg239, dims = [1] : (tensor<2048xf32>) -> tensor<920x2048xf32>
    %577 = stablehlo.add %575, %576 : tensor<920x2048xf32>
    %578 = stablehlo.convert %577 : (tensor<920x2048xf32>) -> tensor<920x2048xbf16>
    %579 = stablehlo.reshape %578 : (tensor<920x2048xbf16>) -> tensor<920x1x2048xbf16>
    %580 = stablehlo.maximum %579, %cst_15 : tensor<920x1x2048xbf16>
    %581 = stablehlo.reshape %580 : (tensor<920x1x2048xbf16>) -> tensor<920x2048xbf16>
    %582 = stablehlo.convert %581 : (tensor<920x2048xbf16>) -> tensor<920x2048xf32>
    %583 = stablehlo.dot_general %582, %arg240, contracting_dims = [1] x [0] : (tensor<920x2048xf32>, tensor<2048x256xf32>) -> tensor<920x256xf32>
    %584 = stablehlo.broadcast_in_dim %583, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %585 = stablehlo.multiply %584, %515 : tensor<920x256xf32>
    %586 = stablehlo.broadcast_in_dim %585, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %587 = stablehlo.broadcast_in_dim %arg241, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %588 = stablehlo.add %586, %587 : tensor<920x256xf32>
    %589 = stablehlo.convert %588 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %590 = stablehlo.reshape %589 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %591 = stablehlo.add %568, %590 : tensor<920x1x256xbf16>
    %592 = stablehlo.convert %591 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %593 = stablehlo.convert %592 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %594 = stablehlo.reduce(%593 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %595 = stablehlo.reshape %594 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %596 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %597 = stablehlo.divide %596, %530 : tensor<920x1x1xf64>
    %598 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %599 = stablehlo.broadcast_in_dim %597, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %600 = stablehlo.subtract %598, %599 : tensor<920x1x256xf64>
    %601 = stablehlo.multiply %600, %600 : tensor<920x1x256xf64>
    %602 = stablehlo.reduce(%601 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %603 = stablehlo.reshape %602 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %604 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %605 = stablehlo.divide %604, %530 : tensor<920x1x1xf64>
    %606 = stablehlo.convert %605 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %607 = stablehlo.reduce(%592 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %608 = stablehlo.reshape %607 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %609 = stablehlo.broadcast_in_dim %608, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %610 = stablehlo.divide %609, %546 : tensor<920x1x1xf32>
    %611 = stablehlo.broadcast_in_dim %606, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %612 = stablehlo.add %611, %551 : tensor<920x1x1xf32>
    %613 = stablehlo.rsqrt %612 : tensor<920x1x1xf32>
    %614 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %615 = stablehlo.broadcast_in_dim %610, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %616 = stablehlo.subtract %614, %615 : tensor<920x1x256xf32>
    %617 = stablehlo.broadcast_in_dim %616, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %618 = stablehlo.broadcast_in_dim %613, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %619 = stablehlo.multiply %617, %618 : tensor<920x1x256xf32>
    %620 = stablehlo.convert %arg58 : (tensor<256xbf16>) -> tensor<256xf32>
    %621 = stablehlo.broadcast_in_dim %619, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %622 = stablehlo.broadcast_in_dim %620, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %623 = stablehlo.multiply %621, %622 : tensor<920x1x256xf32>
    %624 = stablehlo.convert %arg59 : (tensor<256xbf16>) -> tensor<256xf32>
    %625 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %626 = stablehlo.broadcast_in_dim %624, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %627 = stablehlo.add %625, %626 : tensor<920x1x256xf32>
    %628 = stablehlo.convert %627 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %629 = stablehlo.add %628, %arg228 : tensor<920x1x256xbf16>
    %630 = stablehlo.reshape %629 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %631 = stablehlo.convert %630 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %632 = stablehlo.dot_general %631, %arg242, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %634 = stablehlo.multiply %633, %515 : tensor<920x256xf32>
    %635 = stablehlo.broadcast_in_dim %634, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %636 = stablehlo.broadcast_in_dim %arg243, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %637 = stablehlo.add %635, %636 : tensor<920x256xf32>
    %638 = stablehlo.convert %637 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %639 = stablehlo.reshape %638 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %640 = stablehlo.dot_general %631, %arg244, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %641 = stablehlo.broadcast_in_dim %640, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %642 = stablehlo.multiply %641, %515 : tensor<920x256xf32>
    %643 = stablehlo.broadcast_in_dim %642, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %644 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %645 = stablehlo.add %643, %644 : tensor<920x256xf32>
    %646 = stablehlo.convert %645 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %647 = stablehlo.reshape %646 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %648 = stablehlo.reshape %628 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %649 = stablehlo.convert %648 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %650 = stablehlo.dot_general %649, %arg246, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %651 = stablehlo.broadcast_in_dim %650, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %652 = stablehlo.multiply %651, %515 : tensor<920x256xf32>
    %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %654 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %655 = stablehlo.add %653, %654 : tensor<920x256xf32>
    %656 = stablehlo.convert %655 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %657 = stablehlo.reshape %656 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %658 = stablehlo.reshape %639 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %659 = stablehlo.transpose %658, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %660 = stablehlo.reshape %647 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %661 = stablehlo.transpose %660, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %662 = stablehlo.reshape %657 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %663 = stablehlo.transpose %662, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %664 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %665 = stablehlo.multiply %664, %474 : tensor<8x920x32xbf16>
    %666 = stablehlo.transpose %661, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %667 = stablehlo.broadcast_in_dim %666, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %668 = stablehlo.dot_general %665, %667, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x920x920xbf16>
    %669 = stablehlo.broadcast_in_dim %668, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %670 = stablehlo.multiply %669, %485 : tensor<8x920x920xbf16>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %672 = stablehlo.broadcast_in_dim %arg248, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x920x920xbf16>
    %673 = stablehlo.add %671, %672 : tensor<8x920x920xbf16>
    %674 = stablehlo.convert %673 : (tensor<8x920x920xbf16>) -> tensor<8x920x920xf32>
    %675 = stablehlo.reduce(%674 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %676 = stablehlo.reshape %675 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %677 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %678 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %679 = stablehlo.subtract %677, %678 : tensor<8x920x920xf32>
    %680 = stablehlo.exponential %679 : tensor<8x920x920xf32>
    %681 = stablehlo.reduce(%680 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %682 = stablehlo.reshape %681 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %683 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %684 = stablehlo.broadcast_in_dim %682, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %685 = stablehlo.divide %683, %684 : tensor<8x920x920xf32>
    %686 = stablehlo.convert %685 : (tensor<8x920x920xf32>) -> tensor<8x920x920xbf16>
    %687 = stablehlo.broadcast_in_dim %663, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %688 = stablehlo.dot_general %686, %687, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %689 = stablehlo.transpose %688, dims = [1, 0, 2] : (tensor<8x920x32xbf16>) -> tensor<920x8x32xbf16>
    %690 = stablehlo.reshape %689 : (tensor<920x8x32xbf16>) -> tensor<920x256xbf16>
    %691 = stablehlo.convert %690 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %692 = stablehlo.dot_general %691, %arg249, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %694 = stablehlo.multiply %693, %515 : tensor<920x256xf32>
    %695 = stablehlo.broadcast_in_dim %694, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %696 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %697 = stablehlo.add %695, %696 : tensor<920x256xf32>
    %698 = stablehlo.convert %697 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %699 = stablehlo.reshape %698 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %700 = stablehlo.add %628, %699 : tensor<920x1x256xbf16>
    %701 = stablehlo.convert %700 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %702 = stablehlo.convert %701 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %703 = stablehlo.reduce(%702 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %704 = stablehlo.reshape %703 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %705 = stablehlo.broadcast_in_dim %704, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %706 = stablehlo.divide %705, %530 : tensor<920x1x1xf64>
    %707 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %708 = stablehlo.broadcast_in_dim %706, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %709 = stablehlo.subtract %707, %708 : tensor<920x1x256xf64>
    %710 = stablehlo.multiply %709, %709 : tensor<920x1x256xf64>
    %711 = stablehlo.reduce(%710 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %712 = stablehlo.reshape %711 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %713 = stablehlo.broadcast_in_dim %712, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %714 = stablehlo.divide %713, %530 : tensor<920x1x1xf64>
    %715 = stablehlo.convert %714 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %716 = stablehlo.reduce(%701 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %717 = stablehlo.reshape %716 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %718 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %719 = stablehlo.divide %718, %546 : tensor<920x1x1xf32>
    %720 = stablehlo.broadcast_in_dim %715, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %721 = stablehlo.add %720, %551 : tensor<920x1x1xf32>
    %722 = stablehlo.rsqrt %721 : tensor<920x1x1xf32>
    %723 = stablehlo.broadcast_in_dim %701, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %724 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %725 = stablehlo.subtract %723, %724 : tensor<920x1x256xf32>
    %726 = stablehlo.broadcast_in_dim %725, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %727 = stablehlo.broadcast_in_dim %722, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %728 = stablehlo.multiply %726, %727 : tensor<920x1x256xf32>
    %729 = stablehlo.convert %arg60 : (tensor<256xbf16>) -> tensor<256xf32>
    %730 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %731 = stablehlo.broadcast_in_dim %729, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %732 = stablehlo.multiply %730, %731 : tensor<920x1x256xf32>
    %733 = stablehlo.convert %arg61 : (tensor<256xbf16>) -> tensor<256xf32>
    %734 = stablehlo.broadcast_in_dim %732, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %735 = stablehlo.broadcast_in_dim %733, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %736 = stablehlo.add %734, %735 : tensor<920x1x256xf32>
    %737 = stablehlo.convert %736 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %738 = stablehlo.reshape %737 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %739 = stablehlo.convert %738 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %740 = stablehlo.dot_general %739, %arg251, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x2048xf32>) -> tensor<920x2048xf32>
    %741 = stablehlo.broadcast_in_dim %740, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %742 = stablehlo.multiply %741, %573 : tensor<920x2048xf32>
    %743 = stablehlo.broadcast_in_dim %742, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %744 = stablehlo.broadcast_in_dim %arg252, dims = [1] : (tensor<2048xf32>) -> tensor<920x2048xf32>
    %745 = stablehlo.add %743, %744 : tensor<920x2048xf32>
    %746 = stablehlo.convert %745 : (tensor<920x2048xf32>) -> tensor<920x2048xbf16>
    %747 = stablehlo.reshape %746 : (tensor<920x2048xbf16>) -> tensor<920x1x2048xbf16>
    %748 = stablehlo.maximum %747, %cst_15 : tensor<920x1x2048xbf16>
    %749 = stablehlo.reshape %748 : (tensor<920x1x2048xbf16>) -> tensor<920x2048xbf16>
    %750 = stablehlo.convert %749 : (tensor<920x2048xbf16>) -> tensor<920x2048xf32>
    %751 = stablehlo.dot_general %750, %arg253, contracting_dims = [1] x [0] : (tensor<920x2048xf32>, tensor<2048x256xf32>) -> tensor<920x256xf32>
    %752 = stablehlo.broadcast_in_dim %751, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %753 = stablehlo.multiply %752, %515 : tensor<920x256xf32>
    %754 = stablehlo.broadcast_in_dim %753, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %755 = stablehlo.broadcast_in_dim %arg254, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %756 = stablehlo.add %754, %755 : tensor<920x256xf32>
    %757 = stablehlo.convert %756 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %758 = stablehlo.reshape %757 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %759 = stablehlo.add %737, %758 : tensor<920x1x256xbf16>
    %760 = stablehlo.convert %759 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %761 = stablehlo.convert %760 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %762 = stablehlo.reduce(%761 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %763 = stablehlo.reshape %762 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %764 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %765 = stablehlo.divide %764, %530 : tensor<920x1x1xf64>
    %766 = stablehlo.broadcast_in_dim %761, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %767 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %768 = stablehlo.subtract %766, %767 : tensor<920x1x256xf64>
    %769 = stablehlo.multiply %768, %768 : tensor<920x1x256xf64>
    %770 = stablehlo.reduce(%769 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %771 = stablehlo.reshape %770 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %772 = stablehlo.broadcast_in_dim %771, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %773 = stablehlo.divide %772, %530 : tensor<920x1x1xf64>
    %774 = stablehlo.convert %773 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %775 = stablehlo.reduce(%760 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %776 = stablehlo.reshape %775 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %777 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %778 = stablehlo.divide %777, %546 : tensor<920x1x1xf32>
    %779 = stablehlo.broadcast_in_dim %774, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %780 = stablehlo.add %779, %551 : tensor<920x1x1xf32>
    %781 = stablehlo.rsqrt %780 : tensor<920x1x1xf32>
    %782 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %783 = stablehlo.broadcast_in_dim %778, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %784 = stablehlo.subtract %782, %783 : tensor<920x1x256xf32>
    %785 = stablehlo.broadcast_in_dim %784, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %786 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %787 = stablehlo.multiply %785, %786 : tensor<920x1x256xf32>
    %788 = stablehlo.convert %arg62 : (tensor<256xbf16>) -> tensor<256xf32>
    %789 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %790 = stablehlo.broadcast_in_dim %788, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %791 = stablehlo.multiply %789, %790 : tensor<920x1x256xf32>
    %792 = stablehlo.convert %arg63 : (tensor<256xbf16>) -> tensor<256xf32>
    %793 = stablehlo.broadcast_in_dim %791, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %794 = stablehlo.broadcast_in_dim %792, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %795 = stablehlo.add %793, %794 : tensor<920x1x256xf32>
    %796 = stablehlo.convert %795 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %797 = stablehlo.add %796, %arg228 : tensor<920x1x256xbf16>
    %798 = stablehlo.reshape %797 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %799 = stablehlo.convert %798 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %800 = stablehlo.dot_general %799, %arg255, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %801 = stablehlo.broadcast_in_dim %800, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %802 = stablehlo.multiply %801, %515 : tensor<920x256xf32>
    %803 = stablehlo.broadcast_in_dim %802, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %804 = stablehlo.broadcast_in_dim %arg256, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %805 = stablehlo.add %803, %804 : tensor<920x256xf32>
    %806 = stablehlo.convert %805 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %807 = stablehlo.reshape %806 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %808 = stablehlo.dot_general %799, %arg257, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %809 = stablehlo.broadcast_in_dim %808, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %810 = stablehlo.multiply %809, %515 : tensor<920x256xf32>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %812 = stablehlo.broadcast_in_dim %arg258, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %813 = stablehlo.add %811, %812 : tensor<920x256xf32>
    %814 = stablehlo.convert %813 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %815 = stablehlo.reshape %814 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %816 = stablehlo.reshape %796 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %817 = stablehlo.convert %816 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %818 = stablehlo.dot_general %817, %arg259, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %819 = stablehlo.broadcast_in_dim %818, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %820 = stablehlo.multiply %819, %515 : tensor<920x256xf32>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %822 = stablehlo.broadcast_in_dim %arg260, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %823 = stablehlo.add %821, %822 : tensor<920x256xf32>
    %824 = stablehlo.convert %823 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %825 = stablehlo.reshape %824 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %826 = stablehlo.reshape %807 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %827 = stablehlo.transpose %826, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %828 = stablehlo.reshape %815 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %829 = stablehlo.transpose %828, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %830 = stablehlo.reshape %825 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %831 = stablehlo.transpose %830, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %832 = stablehlo.broadcast_in_dim %827, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %833 = stablehlo.multiply %832, %474 : tensor<8x920x32xbf16>
    %834 = stablehlo.transpose %829, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %835 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %836 = stablehlo.dot_general %833, %835, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x920x920xbf16>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %838 = stablehlo.multiply %837, %485 : tensor<8x920x920xbf16>
    %839 = stablehlo.broadcast_in_dim %838, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %840 = stablehlo.broadcast_in_dim %arg261, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x920x920xbf16>
    %841 = stablehlo.add %839, %840 : tensor<8x920x920xbf16>
    %842 = stablehlo.convert %841 : (tensor<8x920x920xbf16>) -> tensor<8x920x920xf32>
    %843 = stablehlo.reduce(%842 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %844 = stablehlo.reshape %843 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %845 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %846 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %847 = stablehlo.subtract %845, %846 : tensor<8x920x920xf32>
    %848 = stablehlo.exponential %847 : tensor<8x920x920xf32>
    %849 = stablehlo.reduce(%848 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %850 = stablehlo.reshape %849 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %851 = stablehlo.broadcast_in_dim %848, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %852 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %853 = stablehlo.divide %851, %852 : tensor<8x920x920xf32>
    %854 = stablehlo.convert %853 : (tensor<8x920x920xf32>) -> tensor<8x920x920xbf16>
    %855 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %856 = stablehlo.dot_general %854, %855, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %857 = stablehlo.transpose %856, dims = [1, 0, 2] : (tensor<8x920x32xbf16>) -> tensor<920x8x32xbf16>
    %858 = stablehlo.reshape %857 : (tensor<920x8x32xbf16>) -> tensor<920x256xbf16>
    %859 = stablehlo.convert %858 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %860 = stablehlo.dot_general %859, %arg262, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %861 = stablehlo.broadcast_in_dim %860, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %862 = stablehlo.multiply %861, %515 : tensor<920x256xf32>
    %863 = stablehlo.broadcast_in_dim %862, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %864 = stablehlo.broadcast_in_dim %arg263, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %865 = stablehlo.add %863, %864 : tensor<920x256xf32>
    %866 = stablehlo.convert %865 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %867 = stablehlo.reshape %866 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %868 = stablehlo.add %796, %867 : tensor<920x1x256xbf16>
    %869 = stablehlo.convert %868 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %870 = stablehlo.convert %869 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %871 = stablehlo.reduce(%870 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %872 = stablehlo.reshape %871 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %873 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %874 = stablehlo.divide %873, %530 : tensor<920x1x1xf64>
    %875 = stablehlo.broadcast_in_dim %870, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %876 = stablehlo.broadcast_in_dim %874, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %877 = stablehlo.subtract %875, %876 : tensor<920x1x256xf64>
    %878 = stablehlo.multiply %877, %877 : tensor<920x1x256xf64>
    %879 = stablehlo.reduce(%878 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %880 = stablehlo.reshape %879 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %882 = stablehlo.divide %881, %530 : tensor<920x1x1xf64>
    %883 = stablehlo.convert %882 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %884 = stablehlo.reduce(%869 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %885 = stablehlo.reshape %884 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %886 = stablehlo.broadcast_in_dim %885, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %887 = stablehlo.divide %886, %546 : tensor<920x1x1xf32>
    %888 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %889 = stablehlo.add %888, %551 : tensor<920x1x1xf32>
    %890 = stablehlo.rsqrt %889 : tensor<920x1x1xf32>
    %891 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %892 = stablehlo.broadcast_in_dim %887, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %893 = stablehlo.subtract %891, %892 : tensor<920x1x256xf32>
    %894 = stablehlo.broadcast_in_dim %893, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %895 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %896 = stablehlo.multiply %894, %895 : tensor<920x1x256xf32>
    %897 = stablehlo.convert %arg64 : (tensor<256xbf16>) -> tensor<256xf32>
    %898 = stablehlo.broadcast_in_dim %896, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %899 = stablehlo.broadcast_in_dim %897, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %900 = stablehlo.multiply %898, %899 : tensor<920x1x256xf32>
    %901 = stablehlo.convert %arg65 : (tensor<256xbf16>) -> tensor<256xf32>
    %902 = stablehlo.broadcast_in_dim %900, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %903 = stablehlo.broadcast_in_dim %901, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %904 = stablehlo.add %902, %903 : tensor<920x1x256xf32>
    %905 = stablehlo.convert %904 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %906 = stablehlo.reshape %905 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %907 = stablehlo.convert %906 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %908 = stablehlo.dot_general %907, %arg264, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x2048xf32>) -> tensor<920x2048xf32>
    %909 = stablehlo.broadcast_in_dim %908, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %910 = stablehlo.multiply %909, %573 : tensor<920x2048xf32>
    %911 = stablehlo.broadcast_in_dim %910, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %912 = stablehlo.broadcast_in_dim %arg265, dims = [1] : (tensor<2048xf32>) -> tensor<920x2048xf32>
    %913 = stablehlo.add %911, %912 : tensor<920x2048xf32>
    %914 = stablehlo.convert %913 : (tensor<920x2048xf32>) -> tensor<920x2048xbf16>
    %915 = stablehlo.reshape %914 : (tensor<920x2048xbf16>) -> tensor<920x1x2048xbf16>
    %916 = stablehlo.maximum %915, %cst_15 : tensor<920x1x2048xbf16>
    %917 = stablehlo.reshape %916 : (tensor<920x1x2048xbf16>) -> tensor<920x2048xbf16>
    %918 = stablehlo.convert %917 : (tensor<920x2048xbf16>) -> tensor<920x2048xf32>
    %919 = stablehlo.dot_general %918, %arg266, contracting_dims = [1] x [0] : (tensor<920x2048xf32>, tensor<2048x256xf32>) -> tensor<920x256xf32>
    %920 = stablehlo.broadcast_in_dim %919, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %921 = stablehlo.multiply %920, %515 : tensor<920x256xf32>
    %922 = stablehlo.broadcast_in_dim %921, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %923 = stablehlo.broadcast_in_dim %arg267, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %924 = stablehlo.add %922, %923 : tensor<920x256xf32>
    %925 = stablehlo.convert %924 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %926 = stablehlo.reshape %925 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %927 = stablehlo.add %905, %926 : tensor<920x1x256xbf16>
    %928 = stablehlo.convert %927 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %929 = stablehlo.convert %928 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %930 = stablehlo.reduce(%929 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %931 = stablehlo.reshape %930 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %932 = stablehlo.broadcast_in_dim %931, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %933 = stablehlo.divide %932, %530 : tensor<920x1x1xf64>
    %934 = stablehlo.broadcast_in_dim %929, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %935 = stablehlo.broadcast_in_dim %933, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %936 = stablehlo.subtract %934, %935 : tensor<920x1x256xf64>
    %937 = stablehlo.multiply %936, %936 : tensor<920x1x256xf64>
    %938 = stablehlo.reduce(%937 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %939 = stablehlo.reshape %938 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %941 = stablehlo.divide %940, %530 : tensor<920x1x1xf64>
    %942 = stablehlo.convert %941 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %943 = stablehlo.reduce(%928 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %944 = stablehlo.reshape %943 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %945 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %946 = stablehlo.divide %945, %546 : tensor<920x1x1xf32>
    %947 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %948 = stablehlo.add %947, %551 : tensor<920x1x1xf32>
    %949 = stablehlo.rsqrt %948 : tensor<920x1x1xf32>
    %950 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %951 = stablehlo.broadcast_in_dim %946, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %952 = stablehlo.subtract %950, %951 : tensor<920x1x256xf32>
    %953 = stablehlo.broadcast_in_dim %952, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %954 = stablehlo.broadcast_in_dim %949, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %955 = stablehlo.multiply %953, %954 : tensor<920x1x256xf32>
    %956 = stablehlo.convert %arg66 : (tensor<256xbf16>) -> tensor<256xf32>
    %957 = stablehlo.broadcast_in_dim %955, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %958 = stablehlo.broadcast_in_dim %956, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %959 = stablehlo.multiply %957, %958 : tensor<920x1x256xf32>
    %960 = stablehlo.convert %arg67 : (tensor<256xbf16>) -> tensor<256xf32>
    %961 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %962 = stablehlo.broadcast_in_dim %960, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %963 = stablehlo.add %961, %962 : tensor<920x1x256xf32>
    %964 = stablehlo.convert %963 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %965 = stablehlo.add %964, %arg228 : tensor<920x1x256xbf16>
    %966 = stablehlo.reshape %965 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %967 = stablehlo.convert %966 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %968 = stablehlo.dot_general %967, %arg268, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %969 = stablehlo.broadcast_in_dim %968, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %970 = stablehlo.multiply %969, %515 : tensor<920x256xf32>
    %971 = stablehlo.broadcast_in_dim %970, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %972 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %973 = stablehlo.add %971, %972 : tensor<920x256xf32>
    %974 = stablehlo.convert %973 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %975 = stablehlo.reshape %974 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %976 = stablehlo.dot_general %967, %arg270, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %977 = stablehlo.broadcast_in_dim %976, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %978 = stablehlo.multiply %977, %515 : tensor<920x256xf32>
    %979 = stablehlo.broadcast_in_dim %978, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %980 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %981 = stablehlo.add %979, %980 : tensor<920x256xf32>
    %982 = stablehlo.convert %981 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %983 = stablehlo.reshape %982 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %984 = stablehlo.reshape %964 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %985 = stablehlo.convert %984 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %986 = stablehlo.dot_general %985, %arg272, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %987 = stablehlo.broadcast_in_dim %986, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %988 = stablehlo.multiply %987, %515 : tensor<920x256xf32>
    %989 = stablehlo.broadcast_in_dim %988, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %990 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %991 = stablehlo.add %989, %990 : tensor<920x256xf32>
    %992 = stablehlo.convert %991 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %993 = stablehlo.reshape %992 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %994 = stablehlo.reshape %975 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %995 = stablehlo.transpose %994, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %996 = stablehlo.reshape %983 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %997 = stablehlo.transpose %996, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %998 = stablehlo.reshape %993 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %999 = stablehlo.transpose %998, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1000 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1001 = stablehlo.multiply %1000, %474 : tensor<8x920x32xbf16>
    %1002 = stablehlo.transpose %997, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %1003 = stablehlo.broadcast_in_dim %1002, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %1004 = stablehlo.dot_general %1001, %1003, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x920x920xbf16>
    %1005 = stablehlo.broadcast_in_dim %1004, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %1006 = stablehlo.multiply %1005, %485 : tensor<8x920x920xbf16>
    %1007 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %1008 = stablehlo.broadcast_in_dim %arg274, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x920x920xbf16>
    %1009 = stablehlo.add %1007, %1008 : tensor<8x920x920xbf16>
    %1010 = stablehlo.convert %1009 : (tensor<8x920x920xbf16>) -> tensor<8x920x920xf32>
    %1011 = stablehlo.reduce(%1010 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %1012 = stablehlo.reshape %1011 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %1013 = stablehlo.broadcast_in_dim %1010, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %1014 = stablehlo.broadcast_in_dim %1012, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %1015 = stablehlo.subtract %1013, %1014 : tensor<8x920x920xf32>
    %1016 = stablehlo.exponential %1015 : tensor<8x920x920xf32>
    %1017 = stablehlo.reduce(%1016 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %1018 = stablehlo.reshape %1017 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %1019 = stablehlo.broadcast_in_dim %1016, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %1020 = stablehlo.broadcast_in_dim %1018, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %1021 = stablehlo.divide %1019, %1020 : tensor<8x920x920xf32>
    %1022 = stablehlo.convert %1021 : (tensor<8x920x920xf32>) -> tensor<8x920x920xbf16>
    %1023 = stablehlo.broadcast_in_dim %999, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1024 = stablehlo.dot_general %1022, %1023, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1025 = stablehlo.transpose %1024, dims = [1, 0, 2] : (tensor<8x920x32xbf16>) -> tensor<920x8x32xbf16>
    %1026 = stablehlo.reshape %1025 : (tensor<920x8x32xbf16>) -> tensor<920x256xbf16>
    %1027 = stablehlo.convert %1026 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1028 = stablehlo.dot_general %1027, %arg275, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1029 = stablehlo.broadcast_in_dim %1028, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1030 = stablehlo.multiply %1029, %515 : tensor<920x256xf32>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1032 = stablehlo.broadcast_in_dim %arg276, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1033 = stablehlo.add %1031, %1032 : tensor<920x256xf32>
    %1034 = stablehlo.convert %1033 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1035 = stablehlo.reshape %1034 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1036 = stablehlo.add %964, %1035 : tensor<920x1x256xbf16>
    %1037 = stablehlo.convert %1036 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %1038 = stablehlo.convert %1037 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %1039 = stablehlo.reduce(%1038 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1040 = stablehlo.reshape %1039 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1042 = stablehlo.divide %1041, %530 : tensor<920x1x1xf64>
    %1043 = stablehlo.broadcast_in_dim %1038, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %1044 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %1045 = stablehlo.subtract %1043, %1044 : tensor<920x1x256xf64>
    %1046 = stablehlo.multiply %1045, %1045 : tensor<920x1x256xf64>
    %1047 = stablehlo.reduce(%1046 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1048 = stablehlo.reshape %1047 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1049 = stablehlo.broadcast_in_dim %1048, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1050 = stablehlo.divide %1049, %530 : tensor<920x1x1xf64>
    %1051 = stablehlo.convert %1050 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %1052 = stablehlo.reduce(%1037 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %1053 = stablehlo.reshape %1052 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1055 = stablehlo.divide %1054, %546 : tensor<920x1x1xf32>
    %1056 = stablehlo.broadcast_in_dim %1051, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1057 = stablehlo.add %1056, %551 : tensor<920x1x1xf32>
    %1058 = stablehlo.rsqrt %1057 : tensor<920x1x1xf32>
    %1059 = stablehlo.broadcast_in_dim %1037, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1060 = stablehlo.broadcast_in_dim %1055, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1061 = stablehlo.subtract %1059, %1060 : tensor<920x1x256xf32>
    %1062 = stablehlo.broadcast_in_dim %1061, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1063 = stablehlo.broadcast_in_dim %1058, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1064 = stablehlo.multiply %1062, %1063 : tensor<920x1x256xf32>
    %1065 = stablehlo.convert %arg68 : (tensor<256xbf16>) -> tensor<256xf32>
    %1066 = stablehlo.broadcast_in_dim %1064, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1067 = stablehlo.broadcast_in_dim %1065, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1068 = stablehlo.multiply %1066, %1067 : tensor<920x1x256xf32>
    %1069 = stablehlo.convert %arg69 : (tensor<256xbf16>) -> tensor<256xf32>
    %1070 = stablehlo.broadcast_in_dim %1068, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1071 = stablehlo.broadcast_in_dim %1069, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1072 = stablehlo.add %1070, %1071 : tensor<920x1x256xf32>
    %1073 = stablehlo.convert %1072 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %1074 = stablehlo.reshape %1073 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1075 = stablehlo.convert %1074 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1076 = stablehlo.dot_general %1075, %arg277, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x2048xf32>) -> tensor<920x2048xf32>
    %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %1078 = stablehlo.multiply %1077, %573 : tensor<920x2048xf32>
    %1079 = stablehlo.broadcast_in_dim %1078, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %1080 = stablehlo.broadcast_in_dim %arg278, dims = [1] : (tensor<2048xf32>) -> tensor<920x2048xf32>
    %1081 = stablehlo.add %1079, %1080 : tensor<920x2048xf32>
    %1082 = stablehlo.convert %1081 : (tensor<920x2048xf32>) -> tensor<920x2048xbf16>
    %1083 = stablehlo.reshape %1082 : (tensor<920x2048xbf16>) -> tensor<920x1x2048xbf16>
    %1084 = stablehlo.maximum %1083, %cst_15 : tensor<920x1x2048xbf16>
    %1085 = stablehlo.reshape %1084 : (tensor<920x1x2048xbf16>) -> tensor<920x2048xbf16>
    %1086 = stablehlo.convert %1085 : (tensor<920x2048xbf16>) -> tensor<920x2048xf32>
    %1087 = stablehlo.dot_general %1086, %arg279, contracting_dims = [1] x [0] : (tensor<920x2048xf32>, tensor<2048x256xf32>) -> tensor<920x256xf32>
    %1088 = stablehlo.broadcast_in_dim %1087, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1089 = stablehlo.multiply %1088, %515 : tensor<920x256xf32>
    %1090 = stablehlo.broadcast_in_dim %1089, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1091 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1092 = stablehlo.add %1090, %1091 : tensor<920x256xf32>
    %1093 = stablehlo.convert %1092 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1094 = stablehlo.reshape %1093 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1095 = stablehlo.add %1073, %1094 : tensor<920x1x256xbf16>
    %1096 = stablehlo.convert %1095 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %1097 = stablehlo.convert %1096 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %1098 = stablehlo.reduce(%1097 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1099 = stablehlo.reshape %1098 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1100 = stablehlo.broadcast_in_dim %1099, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1101 = stablehlo.divide %1100, %530 : tensor<920x1x1xf64>
    %1102 = stablehlo.broadcast_in_dim %1097, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %1103 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %1104 = stablehlo.subtract %1102, %1103 : tensor<920x1x256xf64>
    %1105 = stablehlo.multiply %1104, %1104 : tensor<920x1x256xf64>
    %1106 = stablehlo.reduce(%1105 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1107 = stablehlo.reshape %1106 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1108 = stablehlo.broadcast_in_dim %1107, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1109 = stablehlo.divide %1108, %530 : tensor<920x1x1xf64>
    %1110 = stablehlo.convert %1109 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %1111 = stablehlo.reduce(%1096 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %1112 = stablehlo.reshape %1111 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %1113 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1114 = stablehlo.divide %1113, %546 : tensor<920x1x1xf32>
    %1115 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1116 = stablehlo.add %1115, %551 : tensor<920x1x1xf32>
    %1117 = stablehlo.rsqrt %1116 : tensor<920x1x1xf32>
    %1118 = stablehlo.broadcast_in_dim %1096, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1119 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1120 = stablehlo.subtract %1118, %1119 : tensor<920x1x256xf32>
    %1121 = stablehlo.broadcast_in_dim %1120, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1122 = stablehlo.broadcast_in_dim %1117, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1123 = stablehlo.multiply %1121, %1122 : tensor<920x1x256xf32>
    %1124 = stablehlo.convert %arg70 : (tensor<256xbf16>) -> tensor<256xf32>
    %1125 = stablehlo.broadcast_in_dim %1123, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1126 = stablehlo.broadcast_in_dim %1124, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1127 = stablehlo.multiply %1125, %1126 : tensor<920x1x256xf32>
    %1128 = stablehlo.convert %arg71 : (tensor<256xbf16>) -> tensor<256xf32>
    %1129 = stablehlo.broadcast_in_dim %1127, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1130 = stablehlo.broadcast_in_dim %1128, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1131 = stablehlo.add %1129, %1130 : tensor<920x1x256xf32>
    %1132 = stablehlo.convert %1131 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %1133 = stablehlo.add %1132, %arg228 : tensor<920x1x256xbf16>
    %1134 = stablehlo.reshape %1133 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1135 = stablehlo.convert %1134 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1136 = stablehlo.dot_general %1135, %arg281, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1138 = stablehlo.multiply %1137, %515 : tensor<920x256xf32>
    %1139 = stablehlo.broadcast_in_dim %1138, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1140 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1141 = stablehlo.add %1139, %1140 : tensor<920x256xf32>
    %1142 = stablehlo.convert %1141 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1143 = stablehlo.reshape %1142 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1144 = stablehlo.dot_general %1135, %arg283, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1145 = stablehlo.broadcast_in_dim %1144, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1146 = stablehlo.multiply %1145, %515 : tensor<920x256xf32>
    %1147 = stablehlo.broadcast_in_dim %1146, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1148 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1149 = stablehlo.add %1147, %1148 : tensor<920x256xf32>
    %1150 = stablehlo.convert %1149 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1151 = stablehlo.reshape %1150 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1152 = stablehlo.reshape %1132 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1153 = stablehlo.convert %1152 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1154 = stablehlo.dot_general %1153, %arg285, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1156 = stablehlo.multiply %1155, %515 : tensor<920x256xf32>
    %1157 = stablehlo.broadcast_in_dim %1156, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1158 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1159 = stablehlo.add %1157, %1158 : tensor<920x256xf32>
    %1160 = stablehlo.convert %1159 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1161 = stablehlo.reshape %1160 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1162 = stablehlo.reshape %1143 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1163 = stablehlo.transpose %1162, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1164 = stablehlo.reshape %1151 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1165 = stablehlo.transpose %1164, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1166 = stablehlo.reshape %1161 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1167 = stablehlo.transpose %1166, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1168 = stablehlo.broadcast_in_dim %1163, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1169 = stablehlo.multiply %1168, %474 : tensor<8x920x32xbf16>
    %1170 = stablehlo.transpose %1165, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %1171 = stablehlo.broadcast_in_dim %1170, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %1172 = stablehlo.dot_general %1169, %1171, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x920x920xbf16>
    %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %1174 = stablehlo.multiply %1173, %485 : tensor<8x920x920xbf16>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %1176 = stablehlo.broadcast_in_dim %arg287, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x920x920xbf16>
    %1177 = stablehlo.add %1175, %1176 : tensor<8x920x920xbf16>
    %1178 = stablehlo.convert %1177 : (tensor<8x920x920xbf16>) -> tensor<8x920x920xf32>
    %1179 = stablehlo.reduce(%1178 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %1180 = stablehlo.reshape %1179 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %1181 = stablehlo.broadcast_in_dim %1178, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %1182 = stablehlo.broadcast_in_dim %1180, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %1183 = stablehlo.subtract %1181, %1182 : tensor<8x920x920xf32>
    %1184 = stablehlo.exponential %1183 : tensor<8x920x920xf32>
    %1185 = stablehlo.reduce(%1184 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %1186 = stablehlo.reshape %1185 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %1187 = stablehlo.broadcast_in_dim %1184, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %1188 = stablehlo.broadcast_in_dim %1186, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %1189 = stablehlo.divide %1187, %1188 : tensor<8x920x920xf32>
    %1190 = stablehlo.convert %1189 : (tensor<8x920x920xf32>) -> tensor<8x920x920xbf16>
    %1191 = stablehlo.broadcast_in_dim %1167, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1192 = stablehlo.dot_general %1190, %1191, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1193 = stablehlo.transpose %1192, dims = [1, 0, 2] : (tensor<8x920x32xbf16>) -> tensor<920x8x32xbf16>
    %1194 = stablehlo.reshape %1193 : (tensor<920x8x32xbf16>) -> tensor<920x256xbf16>
    %1195 = stablehlo.convert %1194 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1196 = stablehlo.dot_general %1195, %arg288, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1197 = stablehlo.broadcast_in_dim %1196, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1198 = stablehlo.multiply %1197, %515 : tensor<920x256xf32>
    %1199 = stablehlo.broadcast_in_dim %1198, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1200 = stablehlo.broadcast_in_dim %arg289, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1201 = stablehlo.add %1199, %1200 : tensor<920x256xf32>
    %1202 = stablehlo.convert %1201 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1203 = stablehlo.reshape %1202 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1204 = stablehlo.add %1132, %1203 : tensor<920x1x256xbf16>
    %1205 = stablehlo.convert %1204 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %1206 = stablehlo.convert %1205 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %1207 = stablehlo.reduce(%1206 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1208 = stablehlo.reshape %1207 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1209 = stablehlo.broadcast_in_dim %1208, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1210 = stablehlo.divide %1209, %530 : tensor<920x1x1xf64>
    %1211 = stablehlo.broadcast_in_dim %1206, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %1212 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %1213 = stablehlo.subtract %1211, %1212 : tensor<920x1x256xf64>
    %1214 = stablehlo.multiply %1213, %1213 : tensor<920x1x256xf64>
    %1215 = stablehlo.reduce(%1214 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1216 = stablehlo.reshape %1215 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1218 = stablehlo.divide %1217, %530 : tensor<920x1x1xf64>
    %1219 = stablehlo.convert %1218 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %1220 = stablehlo.reduce(%1205 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %1221 = stablehlo.reshape %1220 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %1222 = stablehlo.broadcast_in_dim %1221, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1223 = stablehlo.divide %1222, %546 : tensor<920x1x1xf32>
    %1224 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1225 = stablehlo.add %1224, %551 : tensor<920x1x1xf32>
    %1226 = stablehlo.rsqrt %1225 : tensor<920x1x1xf32>
    %1227 = stablehlo.broadcast_in_dim %1205, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1228 = stablehlo.broadcast_in_dim %1223, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1229 = stablehlo.subtract %1227, %1228 : tensor<920x1x256xf32>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1231 = stablehlo.broadcast_in_dim %1226, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1232 = stablehlo.multiply %1230, %1231 : tensor<920x1x256xf32>
    %1233 = stablehlo.convert %arg72 : (tensor<256xbf16>) -> tensor<256xf32>
    %1234 = stablehlo.broadcast_in_dim %1232, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1235 = stablehlo.broadcast_in_dim %1233, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1236 = stablehlo.multiply %1234, %1235 : tensor<920x1x256xf32>
    %1237 = stablehlo.convert %arg73 : (tensor<256xbf16>) -> tensor<256xf32>
    %1238 = stablehlo.broadcast_in_dim %1236, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1239 = stablehlo.broadcast_in_dim %1237, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1240 = stablehlo.add %1238, %1239 : tensor<920x1x256xf32>
    %1241 = stablehlo.convert %1240 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %1242 = stablehlo.reshape %1241 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1243 = stablehlo.convert %1242 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1244 = stablehlo.dot_general %1243, %arg290, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x2048xf32>) -> tensor<920x2048xf32>
    %1245 = stablehlo.broadcast_in_dim %1244, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %1246 = stablehlo.multiply %1245, %573 : tensor<920x2048xf32>
    %1247 = stablehlo.broadcast_in_dim %1246, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %1248 = stablehlo.broadcast_in_dim %arg291, dims = [1] : (tensor<2048xf32>) -> tensor<920x2048xf32>
    %1249 = stablehlo.add %1247, %1248 : tensor<920x2048xf32>
    %1250 = stablehlo.convert %1249 : (tensor<920x2048xf32>) -> tensor<920x2048xbf16>
    %1251 = stablehlo.reshape %1250 : (tensor<920x2048xbf16>) -> tensor<920x1x2048xbf16>
    %1252 = stablehlo.maximum %1251, %cst_15 : tensor<920x1x2048xbf16>
    %1253 = stablehlo.reshape %1252 : (tensor<920x1x2048xbf16>) -> tensor<920x2048xbf16>
    %1254 = stablehlo.convert %1253 : (tensor<920x2048xbf16>) -> tensor<920x2048xf32>
    %1255 = stablehlo.dot_general %1254, %arg292, contracting_dims = [1] x [0] : (tensor<920x2048xf32>, tensor<2048x256xf32>) -> tensor<920x256xf32>
    %1256 = stablehlo.broadcast_in_dim %1255, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1257 = stablehlo.multiply %1256, %515 : tensor<920x256xf32>
    %1258 = stablehlo.broadcast_in_dim %1257, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1259 = stablehlo.broadcast_in_dim %arg293, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1260 = stablehlo.add %1258, %1259 : tensor<920x256xf32>
    %1261 = stablehlo.convert %1260 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1262 = stablehlo.reshape %1261 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1263 = stablehlo.add %1241, %1262 : tensor<920x1x256xbf16>
    %1264 = stablehlo.convert %1263 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %1265 = stablehlo.convert %1264 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %1266 = stablehlo.reduce(%1265 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1267 = stablehlo.reshape %1266 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1268 = stablehlo.broadcast_in_dim %1267, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1269 = stablehlo.divide %1268, %530 : tensor<920x1x1xf64>
    %1270 = stablehlo.broadcast_in_dim %1265, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %1271 = stablehlo.broadcast_in_dim %1269, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %1272 = stablehlo.subtract %1270, %1271 : tensor<920x1x256xf64>
    %1273 = stablehlo.multiply %1272, %1272 : tensor<920x1x256xf64>
    %1274 = stablehlo.reduce(%1273 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1275 = stablehlo.reshape %1274 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1276 = stablehlo.broadcast_in_dim %1275, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1277 = stablehlo.divide %1276, %530 : tensor<920x1x1xf64>
    %1278 = stablehlo.convert %1277 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %1279 = stablehlo.reduce(%1264 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %1280 = stablehlo.reshape %1279 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %1281 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1282 = stablehlo.divide %1281, %546 : tensor<920x1x1xf32>
    %1283 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1284 = stablehlo.add %1283, %551 : tensor<920x1x1xf32>
    %1285 = stablehlo.rsqrt %1284 : tensor<920x1x1xf32>
    %1286 = stablehlo.broadcast_in_dim %1264, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1287 = stablehlo.broadcast_in_dim %1282, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1288 = stablehlo.subtract %1286, %1287 : tensor<920x1x256xf32>
    %1289 = stablehlo.broadcast_in_dim %1288, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1290 = stablehlo.broadcast_in_dim %1285, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1291 = stablehlo.multiply %1289, %1290 : tensor<920x1x256xf32>
    %1292 = stablehlo.convert %arg74 : (tensor<256xbf16>) -> tensor<256xf32>
    %1293 = stablehlo.broadcast_in_dim %1291, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1294 = stablehlo.broadcast_in_dim %1292, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1295 = stablehlo.multiply %1293, %1294 : tensor<920x1x256xf32>
    %1296 = stablehlo.convert %arg75 : (tensor<256xbf16>) -> tensor<256xf32>
    %1297 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1298 = stablehlo.broadcast_in_dim %1296, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1299 = stablehlo.add %1297, %1298 : tensor<920x1x256xf32>
    %1300 = stablehlo.convert %1299 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %1301 = stablehlo.add %1300, %arg228 : tensor<920x1x256xbf16>
    %1302 = stablehlo.reshape %1301 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1303 = stablehlo.convert %1302 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1304 = stablehlo.dot_general %1303, %arg294, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1305 = stablehlo.broadcast_in_dim %1304, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1306 = stablehlo.multiply %1305, %515 : tensor<920x256xf32>
    %1307 = stablehlo.broadcast_in_dim %1306, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1308 = stablehlo.broadcast_in_dim %arg295, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1309 = stablehlo.add %1307, %1308 : tensor<920x256xf32>
    %1310 = stablehlo.convert %1309 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1311 = stablehlo.reshape %1310 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1312 = stablehlo.dot_general %1303, %arg296, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1313 = stablehlo.broadcast_in_dim %1312, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1314 = stablehlo.multiply %1313, %515 : tensor<920x256xf32>
    %1315 = stablehlo.broadcast_in_dim %1314, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1316 = stablehlo.broadcast_in_dim %arg297, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1317 = stablehlo.add %1315, %1316 : tensor<920x256xf32>
    %1318 = stablehlo.convert %1317 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1319 = stablehlo.reshape %1318 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1320 = stablehlo.reshape %1300 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1321 = stablehlo.convert %1320 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1322 = stablehlo.dot_general %1321, %arg298, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1324 = stablehlo.multiply %1323, %515 : tensor<920x256xf32>
    %1325 = stablehlo.broadcast_in_dim %1324, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1326 = stablehlo.broadcast_in_dim %arg299, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1327 = stablehlo.add %1325, %1326 : tensor<920x256xf32>
    %1328 = stablehlo.convert %1327 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1329 = stablehlo.reshape %1328 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1330 = stablehlo.reshape %1311 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1331 = stablehlo.transpose %1330, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1332 = stablehlo.reshape %1319 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1333 = stablehlo.transpose %1332, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1334 = stablehlo.reshape %1329 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1335 = stablehlo.transpose %1334, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1336 = stablehlo.broadcast_in_dim %1331, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1337 = stablehlo.multiply %1336, %474 : tensor<8x920x32xbf16>
    %1338 = stablehlo.transpose %1333, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %1340 = stablehlo.dot_general %1337, %1339, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x920x920xbf16>
    %1341 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %1342 = stablehlo.multiply %1341, %485 : tensor<8x920x920xbf16>
    %1343 = stablehlo.broadcast_in_dim %1342, dims = [0, 1, 2] : (tensor<8x920x920xbf16>) -> tensor<8x920x920xbf16>
    %1344 = stablehlo.broadcast_in_dim %arg300, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x920x920xbf16>
    %1345 = stablehlo.add %1343, %1344 : tensor<8x920x920xbf16>
    %1346 = stablehlo.convert %1345 : (tensor<8x920x920xbf16>) -> tensor<8x920x920xf32>
    %1347 = stablehlo.reduce(%1346 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %1348 = stablehlo.reshape %1347 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %1349 = stablehlo.broadcast_in_dim %1346, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %1350 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %1351 = stablehlo.subtract %1349, %1350 : tensor<8x920x920xf32>
    %1352 = stablehlo.exponential %1351 : tensor<8x920x920xf32>
    %1353 = stablehlo.reduce(%1352 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x920x920xf32>, tensor<f32>) -> tensor<8x920xf32>
    %1354 = stablehlo.reshape %1353 : (tensor<8x920xf32>) -> tensor<8x920x1xf32>
    %1355 = stablehlo.broadcast_in_dim %1352, dims = [0, 1, 2] : (tensor<8x920x920xf32>) -> tensor<8x920x920xf32>
    %1356 = stablehlo.broadcast_in_dim %1354, dims = [0, 1, 2] : (tensor<8x920x1xf32>) -> tensor<8x920x920xf32>
    %1357 = stablehlo.divide %1355, %1356 : tensor<8x920x920xf32>
    %1358 = stablehlo.convert %1357 : (tensor<8x920x920xf32>) -> tensor<8x920x920xbf16>
    %1359 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1360 = stablehlo.dot_general %1358, %1359, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x920x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1361 = stablehlo.transpose %1360, dims = [1, 0, 2] : (tensor<8x920x32xbf16>) -> tensor<920x8x32xbf16>
    %1362 = stablehlo.reshape %1361 : (tensor<920x8x32xbf16>) -> tensor<920x256xbf16>
    %1363 = stablehlo.convert %1362 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1364 = stablehlo.dot_general %1363, %arg301, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1365 = stablehlo.broadcast_in_dim %1364, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1366 = stablehlo.multiply %1365, %515 : tensor<920x256xf32>
    %1367 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1368 = stablehlo.broadcast_in_dim %arg302, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1369 = stablehlo.add %1367, %1368 : tensor<920x256xf32>
    %1370 = stablehlo.convert %1369 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1371 = stablehlo.reshape %1370 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1372 = stablehlo.add %1300, %1371 : tensor<920x1x256xbf16>
    %1373 = stablehlo.convert %1372 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %1374 = stablehlo.convert %1373 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %1375 = stablehlo.reduce(%1374 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1376 = stablehlo.reshape %1375 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1377 = stablehlo.broadcast_in_dim %1376, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1378 = stablehlo.divide %1377, %530 : tensor<920x1x1xf64>
    %1379 = stablehlo.broadcast_in_dim %1374, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %1380 = stablehlo.broadcast_in_dim %1378, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %1381 = stablehlo.subtract %1379, %1380 : tensor<920x1x256xf64>
    %1382 = stablehlo.multiply %1381, %1381 : tensor<920x1x256xf64>
    %1383 = stablehlo.reduce(%1382 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1384 = stablehlo.reshape %1383 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1386 = stablehlo.divide %1385, %530 : tensor<920x1x1xf64>
    %1387 = stablehlo.convert %1386 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %1388 = stablehlo.reduce(%1373 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %1389 = stablehlo.reshape %1388 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %1390 = stablehlo.broadcast_in_dim %1389, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1391 = stablehlo.divide %1390, %546 : tensor<920x1x1xf32>
    %1392 = stablehlo.broadcast_in_dim %1387, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1393 = stablehlo.add %1392, %551 : tensor<920x1x1xf32>
    %1394 = stablehlo.rsqrt %1393 : tensor<920x1x1xf32>
    %1395 = stablehlo.broadcast_in_dim %1373, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1396 = stablehlo.broadcast_in_dim %1391, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1397 = stablehlo.subtract %1395, %1396 : tensor<920x1x256xf32>
    %1398 = stablehlo.broadcast_in_dim %1397, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1399 = stablehlo.broadcast_in_dim %1394, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1400 = stablehlo.multiply %1398, %1399 : tensor<920x1x256xf32>
    %1401 = stablehlo.convert %arg76 : (tensor<256xbf16>) -> tensor<256xf32>
    %1402 = stablehlo.broadcast_in_dim %1400, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1403 = stablehlo.broadcast_in_dim %1401, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1404 = stablehlo.multiply %1402, %1403 : tensor<920x1x256xf32>
    %1405 = stablehlo.convert %arg77 : (tensor<256xbf16>) -> tensor<256xf32>
    %1406 = stablehlo.broadcast_in_dim %1404, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1407 = stablehlo.broadcast_in_dim %1405, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1408 = stablehlo.add %1406, %1407 : tensor<920x1x256xf32>
    %1409 = stablehlo.convert %1408 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %1410 = stablehlo.reshape %1409 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1411 = stablehlo.convert %1410 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1412 = stablehlo.dot_general %1411, %arg303, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x2048xf32>) -> tensor<920x2048xf32>
    %1413 = stablehlo.broadcast_in_dim %1412, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %1414 = stablehlo.multiply %1413, %573 : tensor<920x2048xf32>
    %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1] : (tensor<920x2048xf32>) -> tensor<920x2048xf32>
    %1416 = stablehlo.broadcast_in_dim %arg304, dims = [1] : (tensor<2048xf32>) -> tensor<920x2048xf32>
    %1417 = stablehlo.add %1415, %1416 : tensor<920x2048xf32>
    %1418 = stablehlo.convert %1417 : (tensor<920x2048xf32>) -> tensor<920x2048xbf16>
    %1419 = stablehlo.reshape %1418 : (tensor<920x2048xbf16>) -> tensor<920x1x2048xbf16>
    %1420 = stablehlo.maximum %1419, %cst_15 : tensor<920x1x2048xbf16>
    %1421 = stablehlo.reshape %1420 : (tensor<920x1x2048xbf16>) -> tensor<920x2048xbf16>
    %1422 = stablehlo.convert %1421 : (tensor<920x2048xbf16>) -> tensor<920x2048xf32>
    %1423 = stablehlo.dot_general %1422, %arg305, contracting_dims = [1] x [0] : (tensor<920x2048xf32>, tensor<2048x256xf32>) -> tensor<920x256xf32>
    %1424 = stablehlo.broadcast_in_dim %1423, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1425 = stablehlo.multiply %1424, %515 : tensor<920x256xf32>
    %1426 = stablehlo.broadcast_in_dim %1425, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1427 = stablehlo.broadcast_in_dim %arg306, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1428 = stablehlo.add %1426, %1427 : tensor<920x256xf32>
    %1429 = stablehlo.convert %1428 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1430 = stablehlo.reshape %1429 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1431 = stablehlo.add %1409, %1430 : tensor<920x1x256xbf16>
    %1432 = stablehlo.convert %1431 : (tensor<920x1x256xbf16>) -> tensor<920x1x256xf32>
    %1433 = stablehlo.convert %1432 : (tensor<920x1x256xf32>) -> tensor<920x1x256xf64>
    %1434 = stablehlo.reduce(%1433 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1435 = stablehlo.reshape %1434 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1436 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1437 = stablehlo.divide %1436, %530 : tensor<920x1x1xf64>
    %1438 = stablehlo.broadcast_in_dim %1433, dims = [0, 1, 2] : (tensor<920x1x256xf64>) -> tensor<920x1x256xf64>
    %1439 = stablehlo.broadcast_in_dim %1437, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x256xf64>
    %1440 = stablehlo.subtract %1438, %1439 : tensor<920x1x256xf64>
    %1441 = stablehlo.multiply %1440, %1440 : tensor<920x1x256xf64>
    %1442 = stablehlo.reduce(%1441 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf64>, tensor<f64>) -> tensor<920x1xf64>
    %1443 = stablehlo.reshape %1442 : (tensor<920x1xf64>) -> tensor<920x1x1xf64>
    %1444 = stablehlo.broadcast_in_dim %1443, dims = [0, 1, 2] : (tensor<920x1x1xf64>) -> tensor<920x1x1xf64>
    %1445 = stablehlo.divide %1444, %530 : tensor<920x1x1xf64>
    %1446 = stablehlo.convert %1445 : (tensor<920x1x1xf64>) -> tensor<920x1x1xf32>
    %1447 = stablehlo.reduce(%1432 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<920x1x256xf32>, tensor<f32>) -> tensor<920x1xf32>
    %1448 = stablehlo.reshape %1447 : (tensor<920x1xf32>) -> tensor<920x1x1xf32>
    %1449 = stablehlo.broadcast_in_dim %1448, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1450 = stablehlo.divide %1449, %546 : tensor<920x1x1xf32>
    %1451 = stablehlo.broadcast_in_dim %1446, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x1xf32>
    %1452 = stablehlo.add %1451, %551 : tensor<920x1x1xf32>
    %1453 = stablehlo.rsqrt %1452 : tensor<920x1x1xf32>
    %1454 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1455 = stablehlo.broadcast_in_dim %1450, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1456 = stablehlo.subtract %1454, %1455 : tensor<920x1x256xf32>
    %1457 = stablehlo.broadcast_in_dim %1456, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1458 = stablehlo.broadcast_in_dim %1453, dims = [0, 1, 2] : (tensor<920x1x1xf32>) -> tensor<920x1x256xf32>
    %1459 = stablehlo.multiply %1457, %1458 : tensor<920x1x256xf32>
    %1460 = stablehlo.convert %arg78 : (tensor<256xbf16>) -> tensor<256xf32>
    %1461 = stablehlo.broadcast_in_dim %1459, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1462 = stablehlo.broadcast_in_dim %1460, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1463 = stablehlo.multiply %1461, %1462 : tensor<920x1x256xf32>
    %1464 = stablehlo.convert %arg79 : (tensor<256xbf16>) -> tensor<256xf32>
    %1465 = stablehlo.broadcast_in_dim %1463, dims = [0, 1, 2] : (tensor<920x1x256xf32>) -> tensor<920x1x256xf32>
    %1466 = stablehlo.broadcast_in_dim %1464, dims = [2] : (tensor<256xf32>) -> tensor<920x1x256xf32>
    %1467 = stablehlo.add %1465, %1466 : tensor<920x1x256xf32>
    %1468 = stablehlo.convert %1467 : (tensor<920x1x256xf32>) -> tensor<920x1x256xbf16>
    %1469 = stablehlo.add %1468, %arg228 : tensor<920x1x256xbf16>
    %1470 = stablehlo.reshape %1469 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1471 = stablehlo.convert %1470 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1472 = stablehlo.dot_general %1471, %arg307, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1473 = stablehlo.broadcast_in_dim %1472, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1474 = stablehlo.multiply %1473, %515 : tensor<920x256xf32>
    %1475 = stablehlo.broadcast_in_dim %1474, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1476 = stablehlo.broadcast_in_dim %arg308, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1477 = stablehlo.add %1475, %1476 : tensor<920x256xf32>
    %1478 = stablehlo.convert %1477 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1479 = stablehlo.reshape %1478 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1480 = stablehlo.reshape %1468 : (tensor<920x1x256xbf16>) -> tensor<920x256xbf16>
    %1481 = stablehlo.convert %1480 : (tensor<920x256xbf16>) -> tensor<920x256xf32>
    %1482 = stablehlo.dot_general %1481, %arg309, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1483 = stablehlo.broadcast_in_dim %1482, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1484 = stablehlo.multiply %1483, %515 : tensor<920x256xf32>
    %1485 = stablehlo.broadcast_in_dim %1484, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1486 = stablehlo.broadcast_in_dim %arg310, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1487 = stablehlo.add %1485, %1486 : tensor<920x256xf32>
    %1488 = stablehlo.convert %1487 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1489 = stablehlo.reshape %1488 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1490 = stablehlo.reshape %1479 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1491 = stablehlo.transpose %1490, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1492 = stablehlo.reshape %1489 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1493 = stablehlo.transpose %1492, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1494 = stablehlo.transpose %1491, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %1495 = stablehlo.broadcast_in_dim %1494, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %1496 = stablehlo.dot_general %arg312, %1495, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    %1497 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %1498 = stablehlo.broadcast_in_dim %483, dims = [] : (tensor<bf16>) -> tensor<8x100x920xbf16>
    %1499 = stablehlo.multiply %1497, %1498 : tensor<8x100x920xbf16>
    %1500 = stablehlo.broadcast_in_dim %1499, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %1501 = stablehlo.broadcast_in_dim %arg311, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x100x920xbf16>
    %1502 = stablehlo.add %1500, %1501 : tensor<8x100x920xbf16>
    %1503 = stablehlo.convert %1502 : (tensor<8x100x920xbf16>) -> tensor<8x100x920xf32>
    %1504 = stablehlo.reduce(%1503 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %1505 = stablehlo.reshape %1504 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %1506 = stablehlo.broadcast_in_dim %1503, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %1507 = stablehlo.broadcast_in_dim %1505, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %1508 = stablehlo.subtract %1506, %1507 : tensor<8x100x920xf32>
    %1509 = stablehlo.exponential %1508 : tensor<8x100x920xf32>
    %1510 = stablehlo.reduce(%1509 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %1511 = stablehlo.reshape %1510 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %1512 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %1513 = stablehlo.broadcast_in_dim %1511, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %1514 = stablehlo.divide %1512, %1513 : tensor<8x100x920xf32>
    %1515 = stablehlo.convert %1514 : (tensor<8x100x920xf32>) -> tensor<8x100x920xbf16>
    %1516 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1517 = stablehlo.dot_general %1515, %1516, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x100x32xbf16>
    %1518 = stablehlo.transpose %1517, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %1519 = stablehlo.reshape %1518 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %1520 = stablehlo.convert %1519 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1521 = stablehlo.dot_general %1520, %arg313, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1522 = stablehlo.broadcast_in_dim %1521, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1523 = stablehlo.broadcast_in_dim %513, dims = [] : (tensor<f32>) -> tensor<100x256xf32>
    %1524 = stablehlo.multiply %1522, %1523 : tensor<100x256xf32>
    %1525 = stablehlo.broadcast_in_dim %1524, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1526 = stablehlo.broadcast_in_dim %arg314, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1527 = stablehlo.add %1525, %1526 : tensor<100x256xf32>
    %1528 = stablehlo.convert %1527 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1529 = stablehlo.reshape %1528 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1530 = stablehlo.add %arg315, %1529 : tensor<100x1x256xbf16>
    %1531 = stablehlo.convert %1530 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1532 = stablehlo.convert %1531 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1533 = stablehlo.reduce(%1532 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1534 = stablehlo.reshape %1533 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1536 = stablehlo.broadcast_in_dim %528, dims = [] : (tensor<f64>) -> tensor<100x1x1xf64>
    %1537 = stablehlo.divide %1535, %1536 : tensor<100x1x1xf64>
    %1538 = stablehlo.broadcast_in_dim %1532, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1539 = stablehlo.broadcast_in_dim %1537, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1540 = stablehlo.subtract %1538, %1539 : tensor<100x1x256xf64>
    %1541 = stablehlo.multiply %1540, %1540 : tensor<100x1x256xf64>
    %1542 = stablehlo.reduce(%1541 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1543 = stablehlo.reshape %1542 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1544 = stablehlo.broadcast_in_dim %1543, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1545 = stablehlo.divide %1544, %1536 : tensor<100x1x1xf64>
    %1546 = stablehlo.convert %1545 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1547 = stablehlo.reduce(%1531 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1548 = stablehlo.reshape %1547 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1549 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1550 = stablehlo.broadcast_in_dim %544, dims = [] : (tensor<f32>) -> tensor<100x1x1xf32>
    %1551 = stablehlo.divide %1549, %1550 : tensor<100x1x1xf32>
    %1552 = stablehlo.broadcast_in_dim %1546, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1553 = stablehlo.broadcast_in_dim %549, dims = [] : (tensor<f32>) -> tensor<100x1x1xf32>
    %1554 = stablehlo.add %1552, %1553 : tensor<100x1x1xf32>
    %1555 = stablehlo.rsqrt %1554 : tensor<100x1x1xf32>
    %1556 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1557 = stablehlo.broadcast_in_dim %1551, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1558 = stablehlo.subtract %1556, %1557 : tensor<100x1x256xf32>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1560 = stablehlo.broadcast_in_dim %1555, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1561 = stablehlo.multiply %1559, %1560 : tensor<100x1x256xf32>
    %1562 = stablehlo.convert %arg80 : (tensor<256xbf16>) -> tensor<256xf32>
    %1563 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1564 = stablehlo.broadcast_in_dim %1562, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1565 = stablehlo.multiply %1563, %1564 : tensor<100x1x256xf32>
    %1566 = stablehlo.convert %arg81 : (tensor<256xbf16>) -> tensor<256xf32>
    %1567 = stablehlo.broadcast_in_dim %1565, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1568 = stablehlo.broadcast_in_dim %1566, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1569 = stablehlo.add %1567, %1568 : tensor<100x1x256xf32>
    %1570 = stablehlo.convert %1569 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1571 = stablehlo.reshape %1570 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1572 = stablehlo.convert %1571 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1573 = stablehlo.dot_general %1572, %arg316, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x2048xf32>) -> tensor<100x2048xf32>
    %1574 = stablehlo.broadcast_in_dim %1573, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %1575 = stablehlo.broadcast_in_dim %513, dims = [] : (tensor<f32>) -> tensor<100x2048xf32>
    %1576 = stablehlo.multiply %1574, %1575 : tensor<100x2048xf32>
    %1577 = stablehlo.broadcast_in_dim %1576, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %1578 = stablehlo.broadcast_in_dim %arg317, dims = [1] : (tensor<2048xf32>) -> tensor<100x2048xf32>
    %1579 = stablehlo.add %1577, %1578 : tensor<100x2048xf32>
    %1580 = stablehlo.convert %1579 : (tensor<100x2048xf32>) -> tensor<100x2048xbf16>
    %1581 = stablehlo.reshape %1580 : (tensor<100x2048xbf16>) -> tensor<100x1x2048xbf16>
    %1582 = stablehlo.maximum %1581, %cst_16 : tensor<100x1x2048xbf16>
    %1583 = stablehlo.reshape %1582 : (tensor<100x1x2048xbf16>) -> tensor<100x2048xbf16>
    %1584 = stablehlo.convert %1583 : (tensor<100x2048xbf16>) -> tensor<100x2048xf32>
    %1585 = stablehlo.dot_general %1584, %arg318, contracting_dims = [1] x [0] : (tensor<100x2048xf32>, tensor<2048x256xf32>) -> tensor<100x256xf32>
    %1586 = stablehlo.broadcast_in_dim %1585, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1587 = stablehlo.multiply %1586, %1523 : tensor<100x256xf32>
    %1588 = stablehlo.broadcast_in_dim %1587, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1589 = stablehlo.broadcast_in_dim %arg319, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1590 = stablehlo.add %1588, %1589 : tensor<100x256xf32>
    %1591 = stablehlo.convert %1590 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1592 = stablehlo.reshape %1591 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1593 = stablehlo.add %1570, %1592 : tensor<100x1x256xbf16>
    %1594 = stablehlo.convert %1593 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1595 = stablehlo.convert %1594 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1596 = stablehlo.reduce(%1595 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1597 = stablehlo.reshape %1596 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1598 = stablehlo.broadcast_in_dim %1597, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1599 = stablehlo.divide %1598, %1536 : tensor<100x1x1xf64>
    %1600 = stablehlo.broadcast_in_dim %1595, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1601 = stablehlo.broadcast_in_dim %1599, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1602 = stablehlo.subtract %1600, %1601 : tensor<100x1x256xf64>
    %1603 = stablehlo.multiply %1602, %1602 : tensor<100x1x256xf64>
    %1604 = stablehlo.reduce(%1603 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1605 = stablehlo.reshape %1604 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1606 = stablehlo.broadcast_in_dim %1605, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1607 = stablehlo.divide %1606, %1536 : tensor<100x1x1xf64>
    %1608 = stablehlo.convert %1607 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1609 = stablehlo.reduce(%1594 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1610 = stablehlo.reshape %1609 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1611 = stablehlo.broadcast_in_dim %1610, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1612 = stablehlo.divide %1611, %1550 : tensor<100x1x1xf32>
    %1613 = stablehlo.broadcast_in_dim %1608, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1614 = stablehlo.add %1613, %1553 : tensor<100x1x1xf32>
    %1615 = stablehlo.rsqrt %1614 : tensor<100x1x1xf32>
    %1616 = stablehlo.broadcast_in_dim %1594, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1617 = stablehlo.broadcast_in_dim %1612, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1618 = stablehlo.subtract %1616, %1617 : tensor<100x1x256xf32>
    %1619 = stablehlo.broadcast_in_dim %1618, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1620 = stablehlo.broadcast_in_dim %1615, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1621 = stablehlo.multiply %1619, %1620 : tensor<100x1x256xf32>
    %1622 = stablehlo.convert %arg82 : (tensor<256xbf16>) -> tensor<256xf32>
    %1623 = stablehlo.broadcast_in_dim %1621, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1624 = stablehlo.broadcast_in_dim %1622, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1625 = stablehlo.multiply %1623, %1624 : tensor<100x1x256xf32>
    %1626 = stablehlo.convert %arg83 : (tensor<256xbf16>) -> tensor<256xf32>
    %1627 = stablehlo.broadcast_in_dim %1625, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1628 = stablehlo.broadcast_in_dim %1626, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1629 = stablehlo.add %1627, %1628 : tensor<100x1x256xf32>
    %1630 = stablehlo.convert %1629 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1631 = stablehlo.convert %1630 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1632 = stablehlo.convert %1631 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1633 = stablehlo.reduce(%1632 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1634 = stablehlo.reshape %1633 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1635 = stablehlo.broadcast_in_dim %1634, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1636 = stablehlo.divide %1635, %1536 : tensor<100x1x1xf64>
    %1637 = stablehlo.broadcast_in_dim %1632, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1638 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1639 = stablehlo.subtract %1637, %1638 : tensor<100x1x256xf64>
    %1640 = stablehlo.multiply %1639, %1639 : tensor<100x1x256xf64>
    %1641 = stablehlo.reduce(%1640 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1642 = stablehlo.reshape %1641 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1643 = stablehlo.broadcast_in_dim %1642, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1644 = stablehlo.divide %1643, %1536 : tensor<100x1x1xf64>
    %1645 = stablehlo.convert %1644 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1646 = stablehlo.reduce(%1631 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1647 = stablehlo.reshape %1646 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1648 = stablehlo.broadcast_in_dim %1647, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1649 = stablehlo.divide %1648, %1550 : tensor<100x1x1xf32>
    %1650 = stablehlo.broadcast_in_dim %1645, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1651 = stablehlo.add %1650, %1553 : tensor<100x1x1xf32>
    %1652 = stablehlo.rsqrt %1651 : tensor<100x1x1xf32>
    %1653 = stablehlo.broadcast_in_dim %1631, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1654 = stablehlo.broadcast_in_dim %1649, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1655 = stablehlo.subtract %1653, %1654 : tensor<100x1x256xf32>
    %1656 = stablehlo.broadcast_in_dim %1655, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1657 = stablehlo.broadcast_in_dim %1652, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1658 = stablehlo.multiply %1656, %1657 : tensor<100x1x256xf32>
    %1659 = stablehlo.convert %arg84 : (tensor<256xbf16>) -> tensor<256xf32>
    %1660 = stablehlo.broadcast_in_dim %1658, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1661 = stablehlo.broadcast_in_dim %1659, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1662 = stablehlo.multiply %1660, %1661 : tensor<100x1x256xf32>
    %1663 = stablehlo.convert %arg85 : (tensor<256xbf16>) -> tensor<256xf32>
    %1664 = stablehlo.broadcast_in_dim %1662, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1665 = stablehlo.broadcast_in_dim %1663, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1666 = stablehlo.add %1664, %1665 : tensor<100x1x256xf32>
    %1667 = stablehlo.convert %1666 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1668 = stablehlo.add %1630, %arg320 : tensor<100x1x256xbf16>
    %1669 = stablehlo.reshape %1668 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1670 = stablehlo.convert %1669 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1671 = stablehlo.dot_general %1670, %arg321, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1672 = stablehlo.broadcast_in_dim %1671, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1673 = stablehlo.multiply %1672, %1523 : tensor<100x256xf32>
    %1674 = stablehlo.broadcast_in_dim %1673, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1675 = stablehlo.broadcast_in_dim %arg322, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1676 = stablehlo.add %1674, %1675 : tensor<100x256xf32>
    %1677 = stablehlo.convert %1676 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1678 = stablehlo.reshape %1677 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1679 = stablehlo.dot_general %1670, %arg323, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1680 = stablehlo.broadcast_in_dim %1679, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1681 = stablehlo.multiply %1680, %1523 : tensor<100x256xf32>
    %1682 = stablehlo.broadcast_in_dim %1681, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1683 = stablehlo.broadcast_in_dim %arg324, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1684 = stablehlo.add %1682, %1683 : tensor<100x256xf32>
    %1685 = stablehlo.convert %1684 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1686 = stablehlo.reshape %1685 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1687 = stablehlo.reshape %1630 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1688 = stablehlo.convert %1687 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1689 = stablehlo.dot_general %1688, %arg325, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1690 = stablehlo.broadcast_in_dim %1689, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1691 = stablehlo.multiply %1690, %1523 : tensor<100x256xf32>
    %1692 = stablehlo.broadcast_in_dim %1691, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1693 = stablehlo.broadcast_in_dim %arg326, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1694 = stablehlo.add %1692, %1693 : tensor<100x256xf32>
    %1695 = stablehlo.convert %1694 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1696 = stablehlo.reshape %1695 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1697 = stablehlo.reshape %1678 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %1698 = stablehlo.transpose %1697, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %1699 = stablehlo.reshape %1686 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %1700 = stablehlo.transpose %1699, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %1701 = stablehlo.reshape %1696 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %1702 = stablehlo.transpose %1701, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %1703 = stablehlo.broadcast_in_dim %1698, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %1704 = stablehlo.broadcast_in_dim %472, dims = [] : (tensor<bf16>) -> tensor<8x100x32xbf16>
    %1705 = stablehlo.multiply %1703, %1704 : tensor<8x100x32xbf16>
    %1706 = stablehlo.transpose %1700, dims = [0, 2, 1] : (tensor<8x100x32xbf16>) -> tensor<8x32x100xbf16>
    %1707 = stablehlo.broadcast_in_dim %1706, dims = [0, 1, 2] : (tensor<8x32x100xbf16>) -> tensor<8x32x100xbf16>
    %1708 = stablehlo.dot_general %1705, %1707, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x100xbf16>) -> tensor<8x100x100xbf16>
    %1709 = stablehlo.convert %1708 : (tensor<8x100x100xbf16>) -> tensor<8x100x100xf32>
    %1710 = stablehlo.reduce(%1709 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %1711 = stablehlo.reshape %1710 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %1712 = stablehlo.broadcast_in_dim %1709, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %1713 = stablehlo.broadcast_in_dim %1711, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %1714 = stablehlo.subtract %1712, %1713 : tensor<8x100x100xf32>
    %1715 = stablehlo.exponential %1714 : tensor<8x100x100xf32>
    %1716 = stablehlo.reduce(%1715 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %1717 = stablehlo.reshape %1716 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %1718 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %1719 = stablehlo.broadcast_in_dim %1717, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %1720 = stablehlo.divide %1718, %1719 : tensor<8x100x100xf32>
    %1721 = stablehlo.convert %1720 : (tensor<8x100x100xf32>) -> tensor<8x100x100xbf16>
    %1722 = stablehlo.broadcast_in_dim %1702, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %1723 = stablehlo.dot_general %1721, %1722, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x100xbf16>, tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %1724 = stablehlo.transpose %1723, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %1725 = stablehlo.reshape %1724 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %1726 = stablehlo.convert %1725 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1727 = stablehlo.dot_general %1726, %arg327, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1728 = stablehlo.broadcast_in_dim %1727, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1729 = stablehlo.multiply %1728, %1523 : tensor<100x256xf32>
    %1730 = stablehlo.broadcast_in_dim %1729, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1731 = stablehlo.broadcast_in_dim %arg328, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1732 = stablehlo.add %1730, %1731 : tensor<100x256xf32>
    %1733 = stablehlo.convert %1732 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1734 = stablehlo.reshape %1733 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1735 = stablehlo.add %1630, %1734 : tensor<100x1x256xbf16>
    %1736 = stablehlo.convert %1735 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1737 = stablehlo.convert %1736 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1738 = stablehlo.reduce(%1737 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1739 = stablehlo.reshape %1738 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1740 = stablehlo.broadcast_in_dim %1739, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1741 = stablehlo.divide %1740, %1536 : tensor<100x1x1xf64>
    %1742 = stablehlo.broadcast_in_dim %1737, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1743 = stablehlo.broadcast_in_dim %1741, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1744 = stablehlo.subtract %1742, %1743 : tensor<100x1x256xf64>
    %1745 = stablehlo.multiply %1744, %1744 : tensor<100x1x256xf64>
    %1746 = stablehlo.reduce(%1745 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1747 = stablehlo.reshape %1746 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1748 = stablehlo.broadcast_in_dim %1747, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1749 = stablehlo.divide %1748, %1536 : tensor<100x1x1xf64>
    %1750 = stablehlo.convert %1749 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1751 = stablehlo.reduce(%1736 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1752 = stablehlo.reshape %1751 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1753 = stablehlo.broadcast_in_dim %1752, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1754 = stablehlo.divide %1753, %1550 : tensor<100x1x1xf32>
    %1755 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1756 = stablehlo.add %1755, %1553 : tensor<100x1x1xf32>
    %1757 = stablehlo.rsqrt %1756 : tensor<100x1x1xf32>
    %1758 = stablehlo.broadcast_in_dim %1736, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1759 = stablehlo.broadcast_in_dim %1754, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1760 = stablehlo.subtract %1758, %1759 : tensor<100x1x256xf32>
    %1761 = stablehlo.broadcast_in_dim %1760, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1762 = stablehlo.broadcast_in_dim %1757, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1763 = stablehlo.multiply %1761, %1762 : tensor<100x1x256xf32>
    %1764 = stablehlo.convert %arg86 : (tensor<256xbf16>) -> tensor<256xf32>
    %1765 = stablehlo.broadcast_in_dim %1763, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1766 = stablehlo.broadcast_in_dim %1764, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1767 = stablehlo.multiply %1765, %1766 : tensor<100x1x256xf32>
    %1768 = stablehlo.convert %arg87 : (tensor<256xbf16>) -> tensor<256xf32>
    %1769 = stablehlo.broadcast_in_dim %1767, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1770 = stablehlo.broadcast_in_dim %1768, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1771 = stablehlo.add %1769, %1770 : tensor<100x1x256xf32>
    %1772 = stablehlo.convert %1771 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1773 = stablehlo.add %1772, %arg320 : tensor<100x1x256xbf16>
    %1774 = stablehlo.reshape %1773 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1775 = stablehlo.convert %1774 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1776 = stablehlo.dot_general %1775, %arg329, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1777 = stablehlo.broadcast_in_dim %1776, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1778 = stablehlo.multiply %1777, %1523 : tensor<100x256xf32>
    %1779 = stablehlo.broadcast_in_dim %1778, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1780 = stablehlo.broadcast_in_dim %arg330, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1781 = stablehlo.add %1779, %1780 : tensor<100x256xf32>
    %1782 = stablehlo.convert %1781 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1783 = stablehlo.reshape %1782 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1784 = stablehlo.dot_general %1471, %arg331, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1785 = stablehlo.broadcast_in_dim %1784, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1786 = stablehlo.multiply %1785, %515 : tensor<920x256xf32>
    %1787 = stablehlo.broadcast_in_dim %1786, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1788 = stablehlo.broadcast_in_dim %arg332, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1789 = stablehlo.add %1787, %1788 : tensor<920x256xf32>
    %1790 = stablehlo.convert %1789 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1791 = stablehlo.reshape %1790 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1792 = stablehlo.dot_general %1481, %arg333, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %1793 = stablehlo.broadcast_in_dim %1792, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1794 = stablehlo.multiply %1793, %515 : tensor<920x256xf32>
    %1795 = stablehlo.broadcast_in_dim %1794, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %1796 = stablehlo.broadcast_in_dim %arg334, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %1797 = stablehlo.add %1795, %1796 : tensor<920x256xf32>
    %1798 = stablehlo.convert %1797 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %1799 = stablehlo.reshape %1798 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %1800 = stablehlo.reshape %1783 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %1801 = stablehlo.transpose %1800, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %1802 = stablehlo.reshape %1791 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1803 = stablehlo.transpose %1802, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1804 = stablehlo.reshape %1799 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %1805 = stablehlo.transpose %1804, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %1806 = stablehlo.broadcast_in_dim %1801, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %1807 = stablehlo.multiply %1806, %1704 : tensor<8x100x32xbf16>
    %1808 = stablehlo.transpose %1803, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %1809 = stablehlo.broadcast_in_dim %1808, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %1810 = stablehlo.dot_general %1807, %1809, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    %1811 = stablehlo.broadcast_in_dim %1810, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %1812 = stablehlo.multiply %1811, %1498 : tensor<8x100x920xbf16>
    %1813 = stablehlo.broadcast_in_dim %1812, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %1814 = stablehlo.broadcast_in_dim %arg335, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x100x920xbf16>
    %1815 = stablehlo.add %1813, %1814 : tensor<8x100x920xbf16>
    %1816 = stablehlo.convert %1815 : (tensor<8x100x920xbf16>) -> tensor<8x100x920xf32>
    %1817 = stablehlo.reduce(%1816 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %1818 = stablehlo.reshape %1817 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %1819 = stablehlo.broadcast_in_dim %1816, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %1820 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %1821 = stablehlo.subtract %1819, %1820 : tensor<8x100x920xf32>
    %1822 = stablehlo.exponential %1821 : tensor<8x100x920xf32>
    %1823 = stablehlo.reduce(%1822 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %1824 = stablehlo.reshape %1823 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %1825 = stablehlo.broadcast_in_dim %1822, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %1826 = stablehlo.broadcast_in_dim %1824, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %1827 = stablehlo.divide %1825, %1826 : tensor<8x100x920xf32>
    %1828 = stablehlo.convert %1827 : (tensor<8x100x920xf32>) -> tensor<8x100x920xbf16>
    %1829 = stablehlo.broadcast_in_dim %1805, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %1830 = stablehlo.dot_general %1828, %1829, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x100x32xbf16>
    %1831 = stablehlo.transpose %1830, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %1832 = stablehlo.reshape %1831 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %1833 = stablehlo.convert %1832 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1834 = stablehlo.dot_general %1833, %arg336, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1835 = stablehlo.broadcast_in_dim %1834, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1836 = stablehlo.multiply %1835, %1523 : tensor<100x256xf32>
    %1837 = stablehlo.broadcast_in_dim %1836, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1838 = stablehlo.broadcast_in_dim %arg337, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1839 = stablehlo.add %1837, %1838 : tensor<100x256xf32>
    %1840 = stablehlo.convert %1839 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1841 = stablehlo.reshape %1840 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1842 = stablehlo.add %1772, %1841 : tensor<100x1x256xbf16>
    %1843 = stablehlo.convert %1842 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1844 = stablehlo.convert %1843 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1845 = stablehlo.reduce(%1844 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1846 = stablehlo.reshape %1845 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1847 = stablehlo.broadcast_in_dim %1846, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1848 = stablehlo.divide %1847, %1536 : tensor<100x1x1xf64>
    %1849 = stablehlo.broadcast_in_dim %1844, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1850 = stablehlo.broadcast_in_dim %1848, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1851 = stablehlo.subtract %1849, %1850 : tensor<100x1x256xf64>
    %1852 = stablehlo.multiply %1851, %1851 : tensor<100x1x256xf64>
    %1853 = stablehlo.reduce(%1852 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1854 = stablehlo.reshape %1853 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1855 = stablehlo.broadcast_in_dim %1854, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1856 = stablehlo.divide %1855, %1536 : tensor<100x1x1xf64>
    %1857 = stablehlo.convert %1856 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1858 = stablehlo.reduce(%1843 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1859 = stablehlo.reshape %1858 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1860 = stablehlo.broadcast_in_dim %1859, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1861 = stablehlo.divide %1860, %1550 : tensor<100x1x1xf32>
    %1862 = stablehlo.broadcast_in_dim %1857, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1863 = stablehlo.add %1862, %1553 : tensor<100x1x1xf32>
    %1864 = stablehlo.rsqrt %1863 : tensor<100x1x1xf32>
    %1865 = stablehlo.broadcast_in_dim %1843, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1866 = stablehlo.broadcast_in_dim %1861, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1867 = stablehlo.subtract %1865, %1866 : tensor<100x1x256xf32>
    %1868 = stablehlo.broadcast_in_dim %1867, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1869 = stablehlo.broadcast_in_dim %1864, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1870 = stablehlo.multiply %1868, %1869 : tensor<100x1x256xf32>
    %1871 = stablehlo.convert %arg88 : (tensor<256xbf16>) -> tensor<256xf32>
    %1872 = stablehlo.broadcast_in_dim %1870, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1873 = stablehlo.broadcast_in_dim %1871, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1874 = stablehlo.multiply %1872, %1873 : tensor<100x1x256xf32>
    %1875 = stablehlo.convert %arg89 : (tensor<256xbf16>) -> tensor<256xf32>
    %1876 = stablehlo.broadcast_in_dim %1874, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1877 = stablehlo.broadcast_in_dim %1875, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1878 = stablehlo.add %1876, %1877 : tensor<100x1x256xf32>
    %1879 = stablehlo.convert %1878 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1880 = stablehlo.reshape %1879 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1881 = stablehlo.convert %1880 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1882 = stablehlo.dot_general %1881, %arg338, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x2048xf32>) -> tensor<100x2048xf32>
    %1883 = stablehlo.broadcast_in_dim %1882, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %1884 = stablehlo.multiply %1883, %1575 : tensor<100x2048xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %1886 = stablehlo.broadcast_in_dim %arg339, dims = [1] : (tensor<2048xf32>) -> tensor<100x2048xf32>
    %1887 = stablehlo.add %1885, %1886 : tensor<100x2048xf32>
    %1888 = stablehlo.convert %1887 : (tensor<100x2048xf32>) -> tensor<100x2048xbf16>
    %1889 = stablehlo.reshape %1888 : (tensor<100x2048xbf16>) -> tensor<100x1x2048xbf16>
    %1890 = stablehlo.maximum %1889, %cst_16 : tensor<100x1x2048xbf16>
    %1891 = stablehlo.reshape %1890 : (tensor<100x1x2048xbf16>) -> tensor<100x2048xbf16>
    %1892 = stablehlo.convert %1891 : (tensor<100x2048xbf16>) -> tensor<100x2048xf32>
    %1893 = stablehlo.dot_general %1892, %arg340, contracting_dims = [1] x [0] : (tensor<100x2048xf32>, tensor<2048x256xf32>) -> tensor<100x256xf32>
    %1894 = stablehlo.broadcast_in_dim %1893, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1895 = stablehlo.multiply %1894, %1523 : tensor<100x256xf32>
    %1896 = stablehlo.broadcast_in_dim %1895, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1897 = stablehlo.broadcast_in_dim %arg341, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1898 = stablehlo.add %1896, %1897 : tensor<100x256xf32>
    %1899 = stablehlo.convert %1898 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1900 = stablehlo.reshape %1899 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1901 = stablehlo.add %1879, %1900 : tensor<100x1x256xbf16>
    %1902 = stablehlo.convert %1901 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1903 = stablehlo.convert %1902 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1904 = stablehlo.reduce(%1903 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1905 = stablehlo.reshape %1904 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1906 = stablehlo.broadcast_in_dim %1905, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1907 = stablehlo.divide %1906, %1536 : tensor<100x1x1xf64>
    %1908 = stablehlo.broadcast_in_dim %1903, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1909 = stablehlo.broadcast_in_dim %1907, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1910 = stablehlo.subtract %1908, %1909 : tensor<100x1x256xf64>
    %1911 = stablehlo.multiply %1910, %1910 : tensor<100x1x256xf64>
    %1912 = stablehlo.reduce(%1911 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1913 = stablehlo.reshape %1912 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1914 = stablehlo.broadcast_in_dim %1913, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1915 = stablehlo.divide %1914, %1536 : tensor<100x1x1xf64>
    %1916 = stablehlo.convert %1915 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1917 = stablehlo.reduce(%1902 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1918 = stablehlo.reshape %1917 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1919 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1920 = stablehlo.divide %1919, %1550 : tensor<100x1x1xf32>
    %1921 = stablehlo.broadcast_in_dim %1916, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1922 = stablehlo.add %1921, %1553 : tensor<100x1x1xf32>
    %1923 = stablehlo.rsqrt %1922 : tensor<100x1x1xf32>
    %1924 = stablehlo.broadcast_in_dim %1902, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1925 = stablehlo.broadcast_in_dim %1920, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1926 = stablehlo.subtract %1924, %1925 : tensor<100x1x256xf32>
    %1927 = stablehlo.broadcast_in_dim %1926, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1928 = stablehlo.broadcast_in_dim %1923, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1929 = stablehlo.multiply %1927, %1928 : tensor<100x1x256xf32>
    %1930 = stablehlo.convert %arg90 : (tensor<256xbf16>) -> tensor<256xf32>
    %1931 = stablehlo.broadcast_in_dim %1929, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1932 = stablehlo.broadcast_in_dim %1930, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1933 = stablehlo.multiply %1931, %1932 : tensor<100x1x256xf32>
    %1934 = stablehlo.convert %arg91 : (tensor<256xbf16>) -> tensor<256xf32>
    %1935 = stablehlo.broadcast_in_dim %1933, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1936 = stablehlo.broadcast_in_dim %1934, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %1937 = stablehlo.add %1935, %1936 : tensor<100x1x256xf32>
    %1938 = stablehlo.convert %1937 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1939 = stablehlo.convert %1938 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %1940 = stablehlo.convert %1939 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %1941 = stablehlo.reduce(%1940 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1942 = stablehlo.reshape %1941 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1943 = stablehlo.broadcast_in_dim %1942, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1944 = stablehlo.divide %1943, %1536 : tensor<100x1x1xf64>
    %1945 = stablehlo.broadcast_in_dim %1940, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %1946 = stablehlo.broadcast_in_dim %1944, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %1947 = stablehlo.subtract %1945, %1946 : tensor<100x1x256xf64>
    %1948 = stablehlo.multiply %1947, %1947 : tensor<100x1x256xf64>
    %1949 = stablehlo.reduce(%1948 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %1950 = stablehlo.reshape %1949 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %1951 = stablehlo.broadcast_in_dim %1950, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %1952 = stablehlo.divide %1951, %1536 : tensor<100x1x1xf64>
    %1953 = stablehlo.convert %1952 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %1954 = stablehlo.reduce(%1939 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %1955 = stablehlo.reshape %1954 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %1956 = stablehlo.broadcast_in_dim %1955, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1957 = stablehlo.divide %1956, %1550 : tensor<100x1x1xf32>
    %1958 = stablehlo.broadcast_in_dim %1953, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %1959 = stablehlo.add %1958, %1553 : tensor<100x1x1xf32>
    %1960 = stablehlo.rsqrt %1959 : tensor<100x1x1xf32>
    %1961 = stablehlo.broadcast_in_dim %1939, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1962 = stablehlo.broadcast_in_dim %1957, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1963 = stablehlo.subtract %1961, %1962 : tensor<100x1x256xf32>
    %1964 = stablehlo.broadcast_in_dim %1963, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1965 = stablehlo.broadcast_in_dim %1960, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %1966 = stablehlo.multiply %1964, %1965 : tensor<100x1x256xf32>
    %1967 = stablehlo.broadcast_in_dim %1966, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1968 = stablehlo.multiply %1967, %1661 : tensor<100x1x256xf32>
    %1969 = stablehlo.broadcast_in_dim %1968, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %1970 = stablehlo.add %1969, %1665 : tensor<100x1x256xf32>
    %1971 = stablehlo.convert %1970 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %1972 = stablehlo.add %1938, %arg320 : tensor<100x1x256xbf16>
    %1973 = stablehlo.reshape %1972 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1974 = stablehlo.convert %1973 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1975 = stablehlo.dot_general %1974, %arg342, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1976 = stablehlo.broadcast_in_dim %1975, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1977 = stablehlo.multiply %1976, %1523 : tensor<100x256xf32>
    %1978 = stablehlo.broadcast_in_dim %1977, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1979 = stablehlo.broadcast_in_dim %arg343, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1980 = stablehlo.add %1978, %1979 : tensor<100x256xf32>
    %1981 = stablehlo.convert %1980 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1982 = stablehlo.reshape %1981 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1983 = stablehlo.dot_general %1974, %arg344, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1984 = stablehlo.broadcast_in_dim %1983, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1985 = stablehlo.multiply %1984, %1523 : tensor<100x256xf32>
    %1986 = stablehlo.broadcast_in_dim %1985, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1987 = stablehlo.broadcast_in_dim %arg345, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1988 = stablehlo.add %1986, %1987 : tensor<100x256xf32>
    %1989 = stablehlo.convert %1988 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %1990 = stablehlo.reshape %1989 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %1991 = stablehlo.reshape %1938 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %1992 = stablehlo.convert %1991 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %1993 = stablehlo.dot_general %1992, %arg346, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1995 = stablehlo.multiply %1994, %1523 : tensor<100x256xf32>
    %1996 = stablehlo.broadcast_in_dim %1995, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %1997 = stablehlo.broadcast_in_dim %arg347, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %1998 = stablehlo.add %1996, %1997 : tensor<100x256xf32>
    %1999 = stablehlo.convert %1998 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2000 = stablehlo.reshape %1999 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2001 = stablehlo.reshape %1982 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2002 = stablehlo.transpose %2001, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2003 = stablehlo.reshape %1990 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2004 = stablehlo.transpose %2003, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2005 = stablehlo.reshape %2000 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2006 = stablehlo.transpose %2005, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2007 = stablehlo.broadcast_in_dim %2002, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2008 = stablehlo.multiply %2007, %1704 : tensor<8x100x32xbf16>
    %2009 = stablehlo.transpose %2004, dims = [0, 2, 1] : (tensor<8x100x32xbf16>) -> tensor<8x32x100xbf16>
    %2010 = stablehlo.broadcast_in_dim %2009, dims = [0, 1, 2] : (tensor<8x32x100xbf16>) -> tensor<8x32x100xbf16>
    %2011 = stablehlo.dot_general %2008, %2010, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x100xbf16>) -> tensor<8x100x100xbf16>
    %2012 = stablehlo.convert %2011 : (tensor<8x100x100xbf16>) -> tensor<8x100x100xf32>
    %2013 = stablehlo.reduce(%2012 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2014 = stablehlo.reshape %2013 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2015 = stablehlo.broadcast_in_dim %2012, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2016 = stablehlo.broadcast_in_dim %2014, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2017 = stablehlo.subtract %2015, %2016 : tensor<8x100x100xf32>
    %2018 = stablehlo.exponential %2017 : tensor<8x100x100xf32>
    %2019 = stablehlo.reduce(%2018 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2020 = stablehlo.reshape %2019 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2021 = stablehlo.broadcast_in_dim %2018, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2022 = stablehlo.broadcast_in_dim %2020, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2023 = stablehlo.divide %2021, %2022 : tensor<8x100x100xf32>
    %2024 = stablehlo.convert %2023 : (tensor<8x100x100xf32>) -> tensor<8x100x100xbf16>
    %2025 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2026 = stablehlo.dot_general %2024, %2025, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x100xbf16>, tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2027 = stablehlo.transpose %2026, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2028 = stablehlo.reshape %2027 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2029 = stablehlo.convert %2028 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2030 = stablehlo.dot_general %2029, %arg348, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2031 = stablehlo.broadcast_in_dim %2030, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2032 = stablehlo.multiply %2031, %1523 : tensor<100x256xf32>
    %2033 = stablehlo.broadcast_in_dim %2032, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2034 = stablehlo.broadcast_in_dim %arg349, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2035 = stablehlo.add %2033, %2034 : tensor<100x256xf32>
    %2036 = stablehlo.convert %2035 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2037 = stablehlo.reshape %2036 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2038 = stablehlo.add %1938, %2037 : tensor<100x1x256xbf16>
    %2039 = stablehlo.convert %2038 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2040 = stablehlo.convert %2039 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2041 = stablehlo.reduce(%2040 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2042 = stablehlo.reshape %2041 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2043 = stablehlo.broadcast_in_dim %2042, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2044 = stablehlo.divide %2043, %1536 : tensor<100x1x1xf64>
    %2045 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2046 = stablehlo.broadcast_in_dim %2044, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2047 = stablehlo.subtract %2045, %2046 : tensor<100x1x256xf64>
    %2048 = stablehlo.multiply %2047, %2047 : tensor<100x1x256xf64>
    %2049 = stablehlo.reduce(%2048 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2050 = stablehlo.reshape %2049 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2051 = stablehlo.broadcast_in_dim %2050, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2052 = stablehlo.divide %2051, %1536 : tensor<100x1x1xf64>
    %2053 = stablehlo.convert %2052 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2054 = stablehlo.reduce(%2039 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2055 = stablehlo.reshape %2054 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2056 = stablehlo.broadcast_in_dim %2055, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2057 = stablehlo.divide %2056, %1550 : tensor<100x1x1xf32>
    %2058 = stablehlo.broadcast_in_dim %2053, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2059 = stablehlo.add %2058, %1553 : tensor<100x1x1xf32>
    %2060 = stablehlo.rsqrt %2059 : tensor<100x1x1xf32>
    %2061 = stablehlo.broadcast_in_dim %2039, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2062 = stablehlo.broadcast_in_dim %2057, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2063 = stablehlo.subtract %2061, %2062 : tensor<100x1x256xf32>
    %2064 = stablehlo.broadcast_in_dim %2063, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2065 = stablehlo.broadcast_in_dim %2060, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2066 = stablehlo.multiply %2064, %2065 : tensor<100x1x256xf32>
    %2067 = stablehlo.convert %arg92 : (tensor<256xbf16>) -> tensor<256xf32>
    %2068 = stablehlo.broadcast_in_dim %2066, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2069 = stablehlo.broadcast_in_dim %2067, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2070 = stablehlo.multiply %2068, %2069 : tensor<100x1x256xf32>
    %2071 = stablehlo.convert %arg93 : (tensor<256xbf16>) -> tensor<256xf32>
    %2072 = stablehlo.broadcast_in_dim %2070, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2073 = stablehlo.broadcast_in_dim %2071, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2074 = stablehlo.add %2072, %2073 : tensor<100x1x256xf32>
    %2075 = stablehlo.convert %2074 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2076 = stablehlo.add %2075, %arg320 : tensor<100x1x256xbf16>
    %2077 = stablehlo.reshape %2076 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2078 = stablehlo.convert %2077 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2079 = stablehlo.dot_general %2078, %arg350, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2080 = stablehlo.broadcast_in_dim %2079, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2081 = stablehlo.multiply %2080, %1523 : tensor<100x256xf32>
    %2082 = stablehlo.broadcast_in_dim %2081, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2083 = stablehlo.broadcast_in_dim %arg351, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2084 = stablehlo.add %2082, %2083 : tensor<100x256xf32>
    %2085 = stablehlo.convert %2084 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2086 = stablehlo.reshape %2085 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2087 = stablehlo.dot_general %1471, %arg352, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2088 = stablehlo.broadcast_in_dim %2087, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2089 = stablehlo.multiply %2088, %515 : tensor<920x256xf32>
    %2090 = stablehlo.broadcast_in_dim %2089, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2091 = stablehlo.broadcast_in_dim %arg353, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %2092 = stablehlo.add %2090, %2091 : tensor<920x256xf32>
    %2093 = stablehlo.convert %2092 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %2094 = stablehlo.reshape %2093 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %2095 = stablehlo.dot_general %1481, %arg354, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2096 = stablehlo.broadcast_in_dim %2095, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2097 = stablehlo.multiply %2096, %515 : tensor<920x256xf32>
    %2098 = stablehlo.broadcast_in_dim %2097, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2099 = stablehlo.broadcast_in_dim %arg355, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %2100 = stablehlo.add %2098, %2099 : tensor<920x256xf32>
    %2101 = stablehlo.convert %2100 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %2102 = stablehlo.reshape %2101 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %2103 = stablehlo.reshape %2086 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2104 = stablehlo.transpose %2103, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2105 = stablehlo.reshape %2094 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %2106 = stablehlo.transpose %2105, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %2107 = stablehlo.reshape %2102 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %2108 = stablehlo.transpose %2107, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %2109 = stablehlo.broadcast_in_dim %2104, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2110 = stablehlo.multiply %2109, %1704 : tensor<8x100x32xbf16>
    %2111 = stablehlo.transpose %2106, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %2112 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %2113 = stablehlo.dot_general %2110, %2112, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    %2114 = stablehlo.broadcast_in_dim %2113, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %2115 = stablehlo.multiply %2114, %1498 : tensor<8x100x920xbf16>
    %2116 = stablehlo.broadcast_in_dim %2115, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %2117 = stablehlo.broadcast_in_dim %arg356, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x100x920xbf16>
    %2118 = stablehlo.add %2116, %2117 : tensor<8x100x920xbf16>
    %2119 = stablehlo.convert %2118 : (tensor<8x100x920xbf16>) -> tensor<8x100x920xf32>
    %2120 = stablehlo.reduce(%2119 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2121 = stablehlo.reshape %2120 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2122 = stablehlo.broadcast_in_dim %2119, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %2123 = stablehlo.broadcast_in_dim %2121, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %2124 = stablehlo.subtract %2122, %2123 : tensor<8x100x920xf32>
    %2125 = stablehlo.exponential %2124 : tensor<8x100x920xf32>
    %2126 = stablehlo.reduce(%2125 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2127 = stablehlo.reshape %2126 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2128 = stablehlo.broadcast_in_dim %2125, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %2129 = stablehlo.broadcast_in_dim %2127, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %2130 = stablehlo.divide %2128, %2129 : tensor<8x100x920xf32>
    %2131 = stablehlo.convert %2130 : (tensor<8x100x920xf32>) -> tensor<8x100x920xbf16>
    %2132 = stablehlo.broadcast_in_dim %2108, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %2133 = stablehlo.dot_general %2131, %2132, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x100x32xbf16>
    %2134 = stablehlo.transpose %2133, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2135 = stablehlo.reshape %2134 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2136 = stablehlo.convert %2135 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2137 = stablehlo.dot_general %2136, %arg357, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2138 = stablehlo.broadcast_in_dim %2137, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2139 = stablehlo.multiply %2138, %1523 : tensor<100x256xf32>
    %2140 = stablehlo.broadcast_in_dim %2139, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2141 = stablehlo.broadcast_in_dim %arg358, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2142 = stablehlo.add %2140, %2141 : tensor<100x256xf32>
    %2143 = stablehlo.convert %2142 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2144 = stablehlo.reshape %2143 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2145 = stablehlo.add %2075, %2144 : tensor<100x1x256xbf16>
    %2146 = stablehlo.convert %2145 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2147 = stablehlo.convert %2146 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2148 = stablehlo.reduce(%2147 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2149 = stablehlo.reshape %2148 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2150 = stablehlo.broadcast_in_dim %2149, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2151 = stablehlo.divide %2150, %1536 : tensor<100x1x1xf64>
    %2152 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2153 = stablehlo.broadcast_in_dim %2151, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2154 = stablehlo.subtract %2152, %2153 : tensor<100x1x256xf64>
    %2155 = stablehlo.multiply %2154, %2154 : tensor<100x1x256xf64>
    %2156 = stablehlo.reduce(%2155 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2157 = stablehlo.reshape %2156 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2158 = stablehlo.broadcast_in_dim %2157, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2159 = stablehlo.divide %2158, %1536 : tensor<100x1x1xf64>
    %2160 = stablehlo.convert %2159 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2161 = stablehlo.reduce(%2146 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2162 = stablehlo.reshape %2161 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2163 = stablehlo.broadcast_in_dim %2162, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2164 = stablehlo.divide %2163, %1550 : tensor<100x1x1xf32>
    %2165 = stablehlo.broadcast_in_dim %2160, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2166 = stablehlo.add %2165, %1553 : tensor<100x1x1xf32>
    %2167 = stablehlo.rsqrt %2166 : tensor<100x1x1xf32>
    %2168 = stablehlo.broadcast_in_dim %2146, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2169 = stablehlo.broadcast_in_dim %2164, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2170 = stablehlo.subtract %2168, %2169 : tensor<100x1x256xf32>
    %2171 = stablehlo.broadcast_in_dim %2170, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2172 = stablehlo.broadcast_in_dim %2167, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2173 = stablehlo.multiply %2171, %2172 : tensor<100x1x256xf32>
    %2174 = stablehlo.convert %arg94 : (tensor<256xbf16>) -> tensor<256xf32>
    %2175 = stablehlo.broadcast_in_dim %2173, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2176 = stablehlo.broadcast_in_dim %2174, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2177 = stablehlo.multiply %2175, %2176 : tensor<100x1x256xf32>
    %2178 = stablehlo.convert %arg95 : (tensor<256xbf16>) -> tensor<256xf32>
    %2179 = stablehlo.broadcast_in_dim %2177, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2180 = stablehlo.broadcast_in_dim %2178, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2181 = stablehlo.add %2179, %2180 : tensor<100x1x256xf32>
    %2182 = stablehlo.convert %2181 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2183 = stablehlo.reshape %2182 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2184 = stablehlo.convert %2183 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2185 = stablehlo.dot_general %2184, %arg359, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x2048xf32>) -> tensor<100x2048xf32>
    %2186 = stablehlo.broadcast_in_dim %2185, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %2187 = stablehlo.multiply %2186, %1575 : tensor<100x2048xf32>
    %2188 = stablehlo.broadcast_in_dim %2187, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %2189 = stablehlo.broadcast_in_dim %arg360, dims = [1] : (tensor<2048xf32>) -> tensor<100x2048xf32>
    %2190 = stablehlo.add %2188, %2189 : tensor<100x2048xf32>
    %2191 = stablehlo.convert %2190 : (tensor<100x2048xf32>) -> tensor<100x2048xbf16>
    %2192 = stablehlo.reshape %2191 : (tensor<100x2048xbf16>) -> tensor<100x1x2048xbf16>
    %2193 = stablehlo.maximum %2192, %cst_16 : tensor<100x1x2048xbf16>
    %2194 = stablehlo.reshape %2193 : (tensor<100x1x2048xbf16>) -> tensor<100x2048xbf16>
    %2195 = stablehlo.convert %2194 : (tensor<100x2048xbf16>) -> tensor<100x2048xf32>
    %2196 = stablehlo.dot_general %2195, %arg361, contracting_dims = [1] x [0] : (tensor<100x2048xf32>, tensor<2048x256xf32>) -> tensor<100x256xf32>
    %2197 = stablehlo.broadcast_in_dim %2196, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2198 = stablehlo.multiply %2197, %1523 : tensor<100x256xf32>
    %2199 = stablehlo.broadcast_in_dim %2198, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2200 = stablehlo.broadcast_in_dim %arg362, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2201 = stablehlo.add %2199, %2200 : tensor<100x256xf32>
    %2202 = stablehlo.convert %2201 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2203 = stablehlo.reshape %2202 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2204 = stablehlo.add %2182, %2203 : tensor<100x1x256xbf16>
    %2205 = stablehlo.convert %2204 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2206 = stablehlo.convert %2205 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2207 = stablehlo.reduce(%2206 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2208 = stablehlo.reshape %2207 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2209 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2210 = stablehlo.divide %2209, %1536 : tensor<100x1x1xf64>
    %2211 = stablehlo.broadcast_in_dim %2206, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2212 = stablehlo.broadcast_in_dim %2210, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2213 = stablehlo.subtract %2211, %2212 : tensor<100x1x256xf64>
    %2214 = stablehlo.multiply %2213, %2213 : tensor<100x1x256xf64>
    %2215 = stablehlo.reduce(%2214 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2216 = stablehlo.reshape %2215 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2217 = stablehlo.broadcast_in_dim %2216, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2218 = stablehlo.divide %2217, %1536 : tensor<100x1x1xf64>
    %2219 = stablehlo.convert %2218 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2220 = stablehlo.reduce(%2205 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2221 = stablehlo.reshape %2220 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2222 = stablehlo.broadcast_in_dim %2221, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2223 = stablehlo.divide %2222, %1550 : tensor<100x1x1xf32>
    %2224 = stablehlo.broadcast_in_dim %2219, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2225 = stablehlo.add %2224, %1553 : tensor<100x1x1xf32>
    %2226 = stablehlo.rsqrt %2225 : tensor<100x1x1xf32>
    %2227 = stablehlo.broadcast_in_dim %2205, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2228 = stablehlo.broadcast_in_dim %2223, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2229 = stablehlo.subtract %2227, %2228 : tensor<100x1x256xf32>
    %2230 = stablehlo.broadcast_in_dim %2229, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2231 = stablehlo.broadcast_in_dim %2226, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2232 = stablehlo.multiply %2230, %2231 : tensor<100x1x256xf32>
    %2233 = stablehlo.convert %arg96 : (tensor<256xbf16>) -> tensor<256xf32>
    %2234 = stablehlo.broadcast_in_dim %2232, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2235 = stablehlo.broadcast_in_dim %2233, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2236 = stablehlo.multiply %2234, %2235 : tensor<100x1x256xf32>
    %2237 = stablehlo.convert %arg97 : (tensor<256xbf16>) -> tensor<256xf32>
    %2238 = stablehlo.broadcast_in_dim %2236, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2239 = stablehlo.broadcast_in_dim %2237, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2240 = stablehlo.add %2238, %2239 : tensor<100x1x256xf32>
    %2241 = stablehlo.convert %2240 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2242 = stablehlo.convert %2241 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2243 = stablehlo.convert %2242 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2244 = stablehlo.reduce(%2243 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2245 = stablehlo.reshape %2244 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2246 = stablehlo.broadcast_in_dim %2245, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2247 = stablehlo.divide %2246, %1536 : tensor<100x1x1xf64>
    %2248 = stablehlo.broadcast_in_dim %2243, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2249 = stablehlo.broadcast_in_dim %2247, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2250 = stablehlo.subtract %2248, %2249 : tensor<100x1x256xf64>
    %2251 = stablehlo.multiply %2250, %2250 : tensor<100x1x256xf64>
    %2252 = stablehlo.reduce(%2251 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2253 = stablehlo.reshape %2252 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2254 = stablehlo.broadcast_in_dim %2253, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2255 = stablehlo.divide %2254, %1536 : tensor<100x1x1xf64>
    %2256 = stablehlo.convert %2255 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2257 = stablehlo.reduce(%2242 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2258 = stablehlo.reshape %2257 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2259 = stablehlo.broadcast_in_dim %2258, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2260 = stablehlo.divide %2259, %1550 : tensor<100x1x1xf32>
    %2261 = stablehlo.broadcast_in_dim %2256, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2262 = stablehlo.add %2261, %1553 : tensor<100x1x1xf32>
    %2263 = stablehlo.rsqrt %2262 : tensor<100x1x1xf32>
    %2264 = stablehlo.broadcast_in_dim %2242, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2265 = stablehlo.broadcast_in_dim %2260, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2266 = stablehlo.subtract %2264, %2265 : tensor<100x1x256xf32>
    %2267 = stablehlo.broadcast_in_dim %2266, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2268 = stablehlo.broadcast_in_dim %2263, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2269 = stablehlo.multiply %2267, %2268 : tensor<100x1x256xf32>
    %2270 = stablehlo.broadcast_in_dim %2269, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2271 = stablehlo.multiply %2270, %1661 : tensor<100x1x256xf32>
    %2272 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2273 = stablehlo.add %2272, %1665 : tensor<100x1x256xf32>
    %2274 = stablehlo.convert %2273 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2275 = stablehlo.add %2241, %arg320 : tensor<100x1x256xbf16>
    %2276 = stablehlo.reshape %2275 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2277 = stablehlo.convert %2276 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2278 = stablehlo.dot_general %2277, %arg363, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2279 = stablehlo.broadcast_in_dim %2278, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2280 = stablehlo.multiply %2279, %1523 : tensor<100x256xf32>
    %2281 = stablehlo.broadcast_in_dim %2280, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2282 = stablehlo.broadcast_in_dim %arg364, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2283 = stablehlo.add %2281, %2282 : tensor<100x256xf32>
    %2284 = stablehlo.convert %2283 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2285 = stablehlo.reshape %2284 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2286 = stablehlo.dot_general %2277, %arg365, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2287 = stablehlo.broadcast_in_dim %2286, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2288 = stablehlo.multiply %2287, %1523 : tensor<100x256xf32>
    %2289 = stablehlo.broadcast_in_dim %2288, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2290 = stablehlo.broadcast_in_dim %arg366, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2291 = stablehlo.add %2289, %2290 : tensor<100x256xf32>
    %2292 = stablehlo.convert %2291 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2293 = stablehlo.reshape %2292 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2294 = stablehlo.reshape %2241 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2295 = stablehlo.convert %2294 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2296 = stablehlo.dot_general %2295, %arg367, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2297 = stablehlo.broadcast_in_dim %2296, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2298 = stablehlo.multiply %2297, %1523 : tensor<100x256xf32>
    %2299 = stablehlo.broadcast_in_dim %2298, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2300 = stablehlo.broadcast_in_dim %arg368, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2301 = stablehlo.add %2299, %2300 : tensor<100x256xf32>
    %2302 = stablehlo.convert %2301 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2303 = stablehlo.reshape %2302 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2304 = stablehlo.reshape %2285 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2305 = stablehlo.transpose %2304, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2306 = stablehlo.reshape %2293 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2307 = stablehlo.transpose %2306, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2308 = stablehlo.reshape %2303 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2309 = stablehlo.transpose %2308, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2310 = stablehlo.broadcast_in_dim %2305, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2311 = stablehlo.multiply %2310, %1704 : tensor<8x100x32xbf16>
    %2312 = stablehlo.transpose %2307, dims = [0, 2, 1] : (tensor<8x100x32xbf16>) -> tensor<8x32x100xbf16>
    %2313 = stablehlo.broadcast_in_dim %2312, dims = [0, 1, 2] : (tensor<8x32x100xbf16>) -> tensor<8x32x100xbf16>
    %2314 = stablehlo.dot_general %2311, %2313, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x100xbf16>) -> tensor<8x100x100xbf16>
    %2315 = stablehlo.convert %2314 : (tensor<8x100x100xbf16>) -> tensor<8x100x100xf32>
    %2316 = stablehlo.reduce(%2315 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2317 = stablehlo.reshape %2316 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2318 = stablehlo.broadcast_in_dim %2315, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2319 = stablehlo.broadcast_in_dim %2317, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2320 = stablehlo.subtract %2318, %2319 : tensor<8x100x100xf32>
    %2321 = stablehlo.exponential %2320 : tensor<8x100x100xf32>
    %2322 = stablehlo.reduce(%2321 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2323 = stablehlo.reshape %2322 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2324 = stablehlo.broadcast_in_dim %2321, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2325 = stablehlo.broadcast_in_dim %2323, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2326 = stablehlo.divide %2324, %2325 : tensor<8x100x100xf32>
    %2327 = stablehlo.convert %2326 : (tensor<8x100x100xf32>) -> tensor<8x100x100xbf16>
    %2328 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2329 = stablehlo.dot_general %2327, %2328, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x100xbf16>, tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2330 = stablehlo.transpose %2329, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2331 = stablehlo.reshape %2330 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2332 = stablehlo.convert %2331 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2333 = stablehlo.dot_general %2332, %arg369, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2334 = stablehlo.broadcast_in_dim %2333, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2335 = stablehlo.multiply %2334, %1523 : tensor<100x256xf32>
    %2336 = stablehlo.broadcast_in_dim %2335, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2337 = stablehlo.broadcast_in_dim %arg370, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2338 = stablehlo.add %2336, %2337 : tensor<100x256xf32>
    %2339 = stablehlo.convert %2338 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2340 = stablehlo.reshape %2339 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2341 = stablehlo.add %2241, %2340 : tensor<100x1x256xbf16>
    %2342 = stablehlo.convert %2341 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2343 = stablehlo.convert %2342 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2344 = stablehlo.reduce(%2343 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2345 = stablehlo.reshape %2344 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2346 = stablehlo.broadcast_in_dim %2345, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2347 = stablehlo.divide %2346, %1536 : tensor<100x1x1xf64>
    %2348 = stablehlo.broadcast_in_dim %2343, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2349 = stablehlo.broadcast_in_dim %2347, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2350 = stablehlo.subtract %2348, %2349 : tensor<100x1x256xf64>
    %2351 = stablehlo.multiply %2350, %2350 : tensor<100x1x256xf64>
    %2352 = stablehlo.reduce(%2351 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2353 = stablehlo.reshape %2352 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2354 = stablehlo.broadcast_in_dim %2353, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2355 = stablehlo.divide %2354, %1536 : tensor<100x1x1xf64>
    %2356 = stablehlo.convert %2355 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2357 = stablehlo.reduce(%2342 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2358 = stablehlo.reshape %2357 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2359 = stablehlo.broadcast_in_dim %2358, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2360 = stablehlo.divide %2359, %1550 : tensor<100x1x1xf32>
    %2361 = stablehlo.broadcast_in_dim %2356, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2362 = stablehlo.add %2361, %1553 : tensor<100x1x1xf32>
    %2363 = stablehlo.rsqrt %2362 : tensor<100x1x1xf32>
    %2364 = stablehlo.broadcast_in_dim %2342, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2365 = stablehlo.broadcast_in_dim %2360, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2366 = stablehlo.subtract %2364, %2365 : tensor<100x1x256xf32>
    %2367 = stablehlo.broadcast_in_dim %2366, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2368 = stablehlo.broadcast_in_dim %2363, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2369 = stablehlo.multiply %2367, %2368 : tensor<100x1x256xf32>
    %2370 = stablehlo.convert %arg98 : (tensor<256xbf16>) -> tensor<256xf32>
    %2371 = stablehlo.broadcast_in_dim %2369, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2372 = stablehlo.broadcast_in_dim %2370, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2373 = stablehlo.multiply %2371, %2372 : tensor<100x1x256xf32>
    %2374 = stablehlo.convert %arg99 : (tensor<256xbf16>) -> tensor<256xf32>
    %2375 = stablehlo.broadcast_in_dim %2373, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2376 = stablehlo.broadcast_in_dim %2374, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2377 = stablehlo.add %2375, %2376 : tensor<100x1x256xf32>
    %2378 = stablehlo.convert %2377 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2379 = stablehlo.add %2378, %arg320 : tensor<100x1x256xbf16>
    %2380 = stablehlo.reshape %2379 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2381 = stablehlo.convert %2380 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2382 = stablehlo.dot_general %2381, %arg371, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2383 = stablehlo.broadcast_in_dim %2382, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2384 = stablehlo.multiply %2383, %1523 : tensor<100x256xf32>
    %2385 = stablehlo.broadcast_in_dim %2384, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2386 = stablehlo.broadcast_in_dim %arg372, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2387 = stablehlo.add %2385, %2386 : tensor<100x256xf32>
    %2388 = stablehlo.convert %2387 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2389 = stablehlo.reshape %2388 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2390 = stablehlo.dot_general %1471, %arg373, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2391 = stablehlo.broadcast_in_dim %2390, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2392 = stablehlo.multiply %2391, %515 : tensor<920x256xf32>
    %2393 = stablehlo.broadcast_in_dim %2392, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2394 = stablehlo.broadcast_in_dim %arg374, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %2395 = stablehlo.add %2393, %2394 : tensor<920x256xf32>
    %2396 = stablehlo.convert %2395 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %2397 = stablehlo.reshape %2396 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %2398 = stablehlo.dot_general %1481, %arg375, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2399 = stablehlo.broadcast_in_dim %2398, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2400 = stablehlo.multiply %2399, %515 : tensor<920x256xf32>
    %2401 = stablehlo.broadcast_in_dim %2400, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2402 = stablehlo.broadcast_in_dim %arg376, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %2403 = stablehlo.add %2401, %2402 : tensor<920x256xf32>
    %2404 = stablehlo.convert %2403 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %2405 = stablehlo.reshape %2404 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %2406 = stablehlo.reshape %2389 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2407 = stablehlo.transpose %2406, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2408 = stablehlo.reshape %2397 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %2409 = stablehlo.transpose %2408, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %2410 = stablehlo.reshape %2405 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %2411 = stablehlo.transpose %2410, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %2412 = stablehlo.broadcast_in_dim %2407, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2413 = stablehlo.multiply %2412, %1704 : tensor<8x100x32xbf16>
    %2414 = stablehlo.transpose %2409, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %2415 = stablehlo.broadcast_in_dim %2414, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %2416 = stablehlo.dot_general %2413, %2415, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    %2417 = stablehlo.broadcast_in_dim %2416, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %2418 = stablehlo.multiply %2417, %1498 : tensor<8x100x920xbf16>
    %2419 = stablehlo.broadcast_in_dim %2418, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %2420 = stablehlo.broadcast_in_dim %arg377, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x100x920xbf16>
    %2421 = stablehlo.add %2419, %2420 : tensor<8x100x920xbf16>
    %2422 = stablehlo.convert %2421 : (tensor<8x100x920xbf16>) -> tensor<8x100x920xf32>
    %2423 = stablehlo.reduce(%2422 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2424 = stablehlo.reshape %2423 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2425 = stablehlo.broadcast_in_dim %2422, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %2426 = stablehlo.broadcast_in_dim %2424, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %2427 = stablehlo.subtract %2425, %2426 : tensor<8x100x920xf32>
    %2428 = stablehlo.exponential %2427 : tensor<8x100x920xf32>
    %2429 = stablehlo.reduce(%2428 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2430 = stablehlo.reshape %2429 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2431 = stablehlo.broadcast_in_dim %2428, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %2432 = stablehlo.broadcast_in_dim %2430, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %2433 = stablehlo.divide %2431, %2432 : tensor<8x100x920xf32>
    %2434 = stablehlo.convert %2433 : (tensor<8x100x920xf32>) -> tensor<8x100x920xbf16>
    %2435 = stablehlo.broadcast_in_dim %2411, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %2436 = stablehlo.dot_general %2434, %2435, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x100x32xbf16>
    %2437 = stablehlo.transpose %2436, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2438 = stablehlo.reshape %2437 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2439 = stablehlo.convert %2438 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2440 = stablehlo.dot_general %2439, %arg378, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2441 = stablehlo.broadcast_in_dim %2440, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2442 = stablehlo.multiply %2441, %1523 : tensor<100x256xf32>
    %2443 = stablehlo.broadcast_in_dim %2442, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2444 = stablehlo.broadcast_in_dim %arg379, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2445 = stablehlo.add %2443, %2444 : tensor<100x256xf32>
    %2446 = stablehlo.convert %2445 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2447 = stablehlo.reshape %2446 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2448 = stablehlo.add %2378, %2447 : tensor<100x1x256xbf16>
    %2449 = stablehlo.convert %2448 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2450 = stablehlo.convert %2449 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2451 = stablehlo.reduce(%2450 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2452 = stablehlo.reshape %2451 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2453 = stablehlo.broadcast_in_dim %2452, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2454 = stablehlo.divide %2453, %1536 : tensor<100x1x1xf64>
    %2455 = stablehlo.broadcast_in_dim %2450, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2456 = stablehlo.broadcast_in_dim %2454, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2457 = stablehlo.subtract %2455, %2456 : tensor<100x1x256xf64>
    %2458 = stablehlo.multiply %2457, %2457 : tensor<100x1x256xf64>
    %2459 = stablehlo.reduce(%2458 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2460 = stablehlo.reshape %2459 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2461 = stablehlo.broadcast_in_dim %2460, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2462 = stablehlo.divide %2461, %1536 : tensor<100x1x1xf64>
    %2463 = stablehlo.convert %2462 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2464 = stablehlo.reduce(%2449 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2465 = stablehlo.reshape %2464 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2466 = stablehlo.broadcast_in_dim %2465, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2467 = stablehlo.divide %2466, %1550 : tensor<100x1x1xf32>
    %2468 = stablehlo.broadcast_in_dim %2463, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2469 = stablehlo.add %2468, %1553 : tensor<100x1x1xf32>
    %2470 = stablehlo.rsqrt %2469 : tensor<100x1x1xf32>
    %2471 = stablehlo.broadcast_in_dim %2449, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2472 = stablehlo.broadcast_in_dim %2467, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2473 = stablehlo.subtract %2471, %2472 : tensor<100x1x256xf32>
    %2474 = stablehlo.broadcast_in_dim %2473, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2475 = stablehlo.broadcast_in_dim %2470, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2476 = stablehlo.multiply %2474, %2475 : tensor<100x1x256xf32>
    %2477 = stablehlo.convert %arg100 : (tensor<256xbf16>) -> tensor<256xf32>
    %2478 = stablehlo.broadcast_in_dim %2476, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2479 = stablehlo.broadcast_in_dim %2477, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2480 = stablehlo.multiply %2478, %2479 : tensor<100x1x256xf32>
    %2481 = stablehlo.convert %arg101 : (tensor<256xbf16>) -> tensor<256xf32>
    %2482 = stablehlo.broadcast_in_dim %2480, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2483 = stablehlo.broadcast_in_dim %2481, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2484 = stablehlo.add %2482, %2483 : tensor<100x1x256xf32>
    %2485 = stablehlo.convert %2484 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2486 = stablehlo.reshape %2485 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2487 = stablehlo.convert %2486 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2488 = stablehlo.dot_general %2487, %arg380, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x2048xf32>) -> tensor<100x2048xf32>
    %2489 = stablehlo.broadcast_in_dim %2488, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %2490 = stablehlo.multiply %2489, %1575 : tensor<100x2048xf32>
    %2491 = stablehlo.broadcast_in_dim %2490, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %2492 = stablehlo.broadcast_in_dim %arg381, dims = [1] : (tensor<2048xf32>) -> tensor<100x2048xf32>
    %2493 = stablehlo.add %2491, %2492 : tensor<100x2048xf32>
    %2494 = stablehlo.convert %2493 : (tensor<100x2048xf32>) -> tensor<100x2048xbf16>
    %2495 = stablehlo.reshape %2494 : (tensor<100x2048xbf16>) -> tensor<100x1x2048xbf16>
    %2496 = stablehlo.maximum %2495, %cst_16 : tensor<100x1x2048xbf16>
    %2497 = stablehlo.reshape %2496 : (tensor<100x1x2048xbf16>) -> tensor<100x2048xbf16>
    %2498 = stablehlo.convert %2497 : (tensor<100x2048xbf16>) -> tensor<100x2048xf32>
    %2499 = stablehlo.dot_general %2498, %arg382, contracting_dims = [1] x [0] : (tensor<100x2048xf32>, tensor<2048x256xf32>) -> tensor<100x256xf32>
    %2500 = stablehlo.broadcast_in_dim %2499, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2501 = stablehlo.multiply %2500, %1523 : tensor<100x256xf32>
    %2502 = stablehlo.broadcast_in_dim %2501, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2503 = stablehlo.broadcast_in_dim %arg383, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2504 = stablehlo.add %2502, %2503 : tensor<100x256xf32>
    %2505 = stablehlo.convert %2504 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2506 = stablehlo.reshape %2505 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2507 = stablehlo.add %2485, %2506 : tensor<100x1x256xbf16>
    %2508 = stablehlo.convert %2507 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2509 = stablehlo.convert %2508 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2510 = stablehlo.reduce(%2509 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2511 = stablehlo.reshape %2510 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2512 = stablehlo.broadcast_in_dim %2511, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2513 = stablehlo.divide %2512, %1536 : tensor<100x1x1xf64>
    %2514 = stablehlo.broadcast_in_dim %2509, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2515 = stablehlo.broadcast_in_dim %2513, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2516 = stablehlo.subtract %2514, %2515 : tensor<100x1x256xf64>
    %2517 = stablehlo.multiply %2516, %2516 : tensor<100x1x256xf64>
    %2518 = stablehlo.reduce(%2517 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2519 = stablehlo.reshape %2518 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2520 = stablehlo.broadcast_in_dim %2519, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2521 = stablehlo.divide %2520, %1536 : tensor<100x1x1xf64>
    %2522 = stablehlo.convert %2521 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2523 = stablehlo.reduce(%2508 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2524 = stablehlo.reshape %2523 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2525 = stablehlo.broadcast_in_dim %2524, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2526 = stablehlo.divide %2525, %1550 : tensor<100x1x1xf32>
    %2527 = stablehlo.broadcast_in_dim %2522, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2528 = stablehlo.add %2527, %1553 : tensor<100x1x1xf32>
    %2529 = stablehlo.rsqrt %2528 : tensor<100x1x1xf32>
    %2530 = stablehlo.broadcast_in_dim %2508, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2531 = stablehlo.broadcast_in_dim %2526, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2532 = stablehlo.subtract %2530, %2531 : tensor<100x1x256xf32>
    %2533 = stablehlo.broadcast_in_dim %2532, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2534 = stablehlo.broadcast_in_dim %2529, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2535 = stablehlo.multiply %2533, %2534 : tensor<100x1x256xf32>
    %2536 = stablehlo.convert %arg102 : (tensor<256xbf16>) -> tensor<256xf32>
    %2537 = stablehlo.broadcast_in_dim %2535, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2538 = stablehlo.broadcast_in_dim %2536, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2539 = stablehlo.multiply %2537, %2538 : tensor<100x1x256xf32>
    %2540 = stablehlo.convert %arg103 : (tensor<256xbf16>) -> tensor<256xf32>
    %2541 = stablehlo.broadcast_in_dim %2539, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2542 = stablehlo.broadcast_in_dim %2540, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2543 = stablehlo.add %2541, %2542 : tensor<100x1x256xf32>
    %2544 = stablehlo.convert %2543 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2545 = stablehlo.convert %2544 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2546 = stablehlo.convert %2545 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2547 = stablehlo.reduce(%2546 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2548 = stablehlo.reshape %2547 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2549 = stablehlo.broadcast_in_dim %2548, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2550 = stablehlo.divide %2549, %1536 : tensor<100x1x1xf64>
    %2551 = stablehlo.broadcast_in_dim %2546, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2552 = stablehlo.broadcast_in_dim %2550, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2553 = stablehlo.subtract %2551, %2552 : tensor<100x1x256xf64>
    %2554 = stablehlo.multiply %2553, %2553 : tensor<100x1x256xf64>
    %2555 = stablehlo.reduce(%2554 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2556 = stablehlo.reshape %2555 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2557 = stablehlo.broadcast_in_dim %2556, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2558 = stablehlo.divide %2557, %1536 : tensor<100x1x1xf64>
    %2559 = stablehlo.convert %2558 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2560 = stablehlo.reduce(%2545 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2561 = stablehlo.reshape %2560 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2562 = stablehlo.broadcast_in_dim %2561, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2563 = stablehlo.divide %2562, %1550 : tensor<100x1x1xf32>
    %2564 = stablehlo.broadcast_in_dim %2559, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2565 = stablehlo.add %2564, %1553 : tensor<100x1x1xf32>
    %2566 = stablehlo.rsqrt %2565 : tensor<100x1x1xf32>
    %2567 = stablehlo.broadcast_in_dim %2545, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2568 = stablehlo.broadcast_in_dim %2563, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2569 = stablehlo.subtract %2567, %2568 : tensor<100x1x256xf32>
    %2570 = stablehlo.broadcast_in_dim %2569, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2571 = stablehlo.broadcast_in_dim %2566, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2572 = stablehlo.multiply %2570, %2571 : tensor<100x1x256xf32>
    %2573 = stablehlo.broadcast_in_dim %2572, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2574 = stablehlo.multiply %2573, %1661 : tensor<100x1x256xf32>
    %2575 = stablehlo.broadcast_in_dim %2574, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2576 = stablehlo.add %2575, %1665 : tensor<100x1x256xf32>
    %2577 = stablehlo.convert %2576 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2578 = stablehlo.add %2544, %arg320 : tensor<100x1x256xbf16>
    %2579 = stablehlo.reshape %2578 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2580 = stablehlo.convert %2579 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2581 = stablehlo.dot_general %2580, %arg384, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2582 = stablehlo.broadcast_in_dim %2581, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2583 = stablehlo.multiply %2582, %1523 : tensor<100x256xf32>
    %2584 = stablehlo.broadcast_in_dim %2583, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2585 = stablehlo.broadcast_in_dim %arg385, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2586 = stablehlo.add %2584, %2585 : tensor<100x256xf32>
    %2587 = stablehlo.convert %2586 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2588 = stablehlo.reshape %2587 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2589 = stablehlo.dot_general %2580, %arg386, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2590 = stablehlo.broadcast_in_dim %2589, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2591 = stablehlo.multiply %2590, %1523 : tensor<100x256xf32>
    %2592 = stablehlo.broadcast_in_dim %2591, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2593 = stablehlo.broadcast_in_dim %arg387, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2594 = stablehlo.add %2592, %2593 : tensor<100x256xf32>
    %2595 = stablehlo.convert %2594 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2596 = stablehlo.reshape %2595 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2597 = stablehlo.reshape %2544 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2598 = stablehlo.convert %2597 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2599 = stablehlo.dot_general %2598, %arg388, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2600 = stablehlo.broadcast_in_dim %2599, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2601 = stablehlo.multiply %2600, %1523 : tensor<100x256xf32>
    %2602 = stablehlo.broadcast_in_dim %2601, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2603 = stablehlo.broadcast_in_dim %arg389, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2604 = stablehlo.add %2602, %2603 : tensor<100x256xf32>
    %2605 = stablehlo.convert %2604 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2606 = stablehlo.reshape %2605 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2607 = stablehlo.reshape %2588 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2608 = stablehlo.transpose %2607, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2609 = stablehlo.reshape %2596 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2610 = stablehlo.transpose %2609, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2611 = stablehlo.reshape %2606 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2612 = stablehlo.transpose %2611, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2613 = stablehlo.broadcast_in_dim %2608, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2614 = stablehlo.multiply %2613, %1704 : tensor<8x100x32xbf16>
    %2615 = stablehlo.transpose %2610, dims = [0, 2, 1] : (tensor<8x100x32xbf16>) -> tensor<8x32x100xbf16>
    %2616 = stablehlo.broadcast_in_dim %2615, dims = [0, 1, 2] : (tensor<8x32x100xbf16>) -> tensor<8x32x100xbf16>
    %2617 = stablehlo.dot_general %2614, %2616, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x100xbf16>) -> tensor<8x100x100xbf16>
    %2618 = stablehlo.convert %2617 : (tensor<8x100x100xbf16>) -> tensor<8x100x100xf32>
    %2619 = stablehlo.reduce(%2618 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2620 = stablehlo.reshape %2619 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2621 = stablehlo.broadcast_in_dim %2618, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2622 = stablehlo.broadcast_in_dim %2620, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2623 = stablehlo.subtract %2621, %2622 : tensor<8x100x100xf32>
    %2624 = stablehlo.exponential %2623 : tensor<8x100x100xf32>
    %2625 = stablehlo.reduce(%2624 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2626 = stablehlo.reshape %2625 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2627 = stablehlo.broadcast_in_dim %2624, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2628 = stablehlo.broadcast_in_dim %2626, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2629 = stablehlo.divide %2627, %2628 : tensor<8x100x100xf32>
    %2630 = stablehlo.convert %2629 : (tensor<8x100x100xf32>) -> tensor<8x100x100xbf16>
    %2631 = stablehlo.broadcast_in_dim %2612, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2632 = stablehlo.dot_general %2630, %2631, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x100xbf16>, tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2633 = stablehlo.transpose %2632, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2634 = stablehlo.reshape %2633 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2635 = stablehlo.convert %2634 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2636 = stablehlo.dot_general %2635, %arg390, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2637 = stablehlo.broadcast_in_dim %2636, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2638 = stablehlo.multiply %2637, %1523 : tensor<100x256xf32>
    %2639 = stablehlo.broadcast_in_dim %2638, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2640 = stablehlo.broadcast_in_dim %arg391, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2641 = stablehlo.add %2639, %2640 : tensor<100x256xf32>
    %2642 = stablehlo.convert %2641 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2643 = stablehlo.reshape %2642 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2644 = stablehlo.add %2544, %2643 : tensor<100x1x256xbf16>
    %2645 = stablehlo.convert %2644 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2646 = stablehlo.convert %2645 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2647 = stablehlo.reduce(%2646 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2648 = stablehlo.reshape %2647 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2649 = stablehlo.broadcast_in_dim %2648, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2650 = stablehlo.divide %2649, %1536 : tensor<100x1x1xf64>
    %2651 = stablehlo.broadcast_in_dim %2646, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2652 = stablehlo.broadcast_in_dim %2650, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2653 = stablehlo.subtract %2651, %2652 : tensor<100x1x256xf64>
    %2654 = stablehlo.multiply %2653, %2653 : tensor<100x1x256xf64>
    %2655 = stablehlo.reduce(%2654 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2656 = stablehlo.reshape %2655 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2657 = stablehlo.broadcast_in_dim %2656, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2658 = stablehlo.divide %2657, %1536 : tensor<100x1x1xf64>
    %2659 = stablehlo.convert %2658 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2660 = stablehlo.reduce(%2645 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2661 = stablehlo.reshape %2660 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2662 = stablehlo.broadcast_in_dim %2661, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2663 = stablehlo.divide %2662, %1550 : tensor<100x1x1xf32>
    %2664 = stablehlo.broadcast_in_dim %2659, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2665 = stablehlo.add %2664, %1553 : tensor<100x1x1xf32>
    %2666 = stablehlo.rsqrt %2665 : tensor<100x1x1xf32>
    %2667 = stablehlo.broadcast_in_dim %2645, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2668 = stablehlo.broadcast_in_dim %2663, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2669 = stablehlo.subtract %2667, %2668 : tensor<100x1x256xf32>
    %2670 = stablehlo.broadcast_in_dim %2669, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2671 = stablehlo.broadcast_in_dim %2666, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2672 = stablehlo.multiply %2670, %2671 : tensor<100x1x256xf32>
    %2673 = stablehlo.convert %arg104 : (tensor<256xbf16>) -> tensor<256xf32>
    %2674 = stablehlo.broadcast_in_dim %2672, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2675 = stablehlo.broadcast_in_dim %2673, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2676 = stablehlo.multiply %2674, %2675 : tensor<100x1x256xf32>
    %2677 = stablehlo.convert %arg105 : (tensor<256xbf16>) -> tensor<256xf32>
    %2678 = stablehlo.broadcast_in_dim %2676, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2679 = stablehlo.broadcast_in_dim %2677, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2680 = stablehlo.add %2678, %2679 : tensor<100x1x256xf32>
    %2681 = stablehlo.convert %2680 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2682 = stablehlo.add %2681, %arg320 : tensor<100x1x256xbf16>
    %2683 = stablehlo.reshape %2682 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2684 = stablehlo.convert %2683 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2685 = stablehlo.dot_general %2684, %arg392, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2686 = stablehlo.broadcast_in_dim %2685, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2687 = stablehlo.multiply %2686, %1523 : tensor<100x256xf32>
    %2688 = stablehlo.broadcast_in_dim %2687, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2689 = stablehlo.broadcast_in_dim %arg393, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2690 = stablehlo.add %2688, %2689 : tensor<100x256xf32>
    %2691 = stablehlo.convert %2690 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2692 = stablehlo.reshape %2691 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2693 = stablehlo.dot_general %1471, %arg394, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2694 = stablehlo.broadcast_in_dim %2693, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2695 = stablehlo.multiply %2694, %515 : tensor<920x256xf32>
    %2696 = stablehlo.broadcast_in_dim %2695, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2697 = stablehlo.broadcast_in_dim %arg395, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %2698 = stablehlo.add %2696, %2697 : tensor<920x256xf32>
    %2699 = stablehlo.convert %2698 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %2700 = stablehlo.reshape %2699 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %2701 = stablehlo.dot_general %1481, %arg396, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2702 = stablehlo.broadcast_in_dim %2701, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2703 = stablehlo.multiply %2702, %515 : tensor<920x256xf32>
    %2704 = stablehlo.broadcast_in_dim %2703, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2705 = stablehlo.broadcast_in_dim %arg397, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %2706 = stablehlo.add %2704, %2705 : tensor<920x256xf32>
    %2707 = stablehlo.convert %2706 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %2708 = stablehlo.reshape %2707 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %2709 = stablehlo.reshape %2692 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2710 = stablehlo.transpose %2709, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2711 = stablehlo.reshape %2700 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %2712 = stablehlo.transpose %2711, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %2713 = stablehlo.reshape %2708 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %2714 = stablehlo.transpose %2713, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %2715 = stablehlo.broadcast_in_dim %2710, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2716 = stablehlo.multiply %2715, %1704 : tensor<8x100x32xbf16>
    %2717 = stablehlo.transpose %2712, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %2718 = stablehlo.broadcast_in_dim %2717, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %2719 = stablehlo.dot_general %2716, %2718, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    %2720 = stablehlo.broadcast_in_dim %2719, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %2721 = stablehlo.multiply %2720, %1498 : tensor<8x100x920xbf16>
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %2723 = stablehlo.broadcast_in_dim %arg398, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x100x920xbf16>
    %2724 = stablehlo.add %2722, %2723 : tensor<8x100x920xbf16>
    %2725 = stablehlo.convert %2724 : (tensor<8x100x920xbf16>) -> tensor<8x100x920xf32>
    %2726 = stablehlo.reduce(%2725 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2727 = stablehlo.reshape %2726 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2728 = stablehlo.broadcast_in_dim %2725, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %2729 = stablehlo.broadcast_in_dim %2727, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %2730 = stablehlo.subtract %2728, %2729 : tensor<8x100x920xf32>
    %2731 = stablehlo.exponential %2730 : tensor<8x100x920xf32>
    %2732 = stablehlo.reduce(%2731 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2733 = stablehlo.reshape %2732 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2734 = stablehlo.broadcast_in_dim %2731, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %2735 = stablehlo.broadcast_in_dim %2733, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %2736 = stablehlo.divide %2734, %2735 : tensor<8x100x920xf32>
    %2737 = stablehlo.convert %2736 : (tensor<8x100x920xf32>) -> tensor<8x100x920xbf16>
    %2738 = stablehlo.broadcast_in_dim %2714, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %2739 = stablehlo.dot_general %2737, %2738, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x100x32xbf16>
    %2740 = stablehlo.transpose %2739, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2741 = stablehlo.reshape %2740 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2742 = stablehlo.convert %2741 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2743 = stablehlo.dot_general %2742, %arg399, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2744 = stablehlo.broadcast_in_dim %2743, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2745 = stablehlo.multiply %2744, %1523 : tensor<100x256xf32>
    %2746 = stablehlo.broadcast_in_dim %2745, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2747 = stablehlo.broadcast_in_dim %arg400, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2748 = stablehlo.add %2746, %2747 : tensor<100x256xf32>
    %2749 = stablehlo.convert %2748 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2750 = stablehlo.reshape %2749 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2751 = stablehlo.add %2681, %2750 : tensor<100x1x256xbf16>
    %2752 = stablehlo.convert %2751 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2753 = stablehlo.convert %2752 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2754 = stablehlo.reduce(%2753 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2755 = stablehlo.reshape %2754 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2756 = stablehlo.broadcast_in_dim %2755, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2757 = stablehlo.divide %2756, %1536 : tensor<100x1x1xf64>
    %2758 = stablehlo.broadcast_in_dim %2753, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2759 = stablehlo.broadcast_in_dim %2757, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2760 = stablehlo.subtract %2758, %2759 : tensor<100x1x256xf64>
    %2761 = stablehlo.multiply %2760, %2760 : tensor<100x1x256xf64>
    %2762 = stablehlo.reduce(%2761 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2763 = stablehlo.reshape %2762 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2764 = stablehlo.broadcast_in_dim %2763, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2765 = stablehlo.divide %2764, %1536 : tensor<100x1x1xf64>
    %2766 = stablehlo.convert %2765 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2767 = stablehlo.reduce(%2752 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2768 = stablehlo.reshape %2767 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2769 = stablehlo.broadcast_in_dim %2768, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2770 = stablehlo.divide %2769, %1550 : tensor<100x1x1xf32>
    %2771 = stablehlo.broadcast_in_dim %2766, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2772 = stablehlo.add %2771, %1553 : tensor<100x1x1xf32>
    %2773 = stablehlo.rsqrt %2772 : tensor<100x1x1xf32>
    %2774 = stablehlo.broadcast_in_dim %2752, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2775 = stablehlo.broadcast_in_dim %2770, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2776 = stablehlo.subtract %2774, %2775 : tensor<100x1x256xf32>
    %2777 = stablehlo.broadcast_in_dim %2776, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2778 = stablehlo.broadcast_in_dim %2773, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2779 = stablehlo.multiply %2777, %2778 : tensor<100x1x256xf32>
    %2780 = stablehlo.convert %arg106 : (tensor<256xbf16>) -> tensor<256xf32>
    %2781 = stablehlo.broadcast_in_dim %2779, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2782 = stablehlo.broadcast_in_dim %2780, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2783 = stablehlo.multiply %2781, %2782 : tensor<100x1x256xf32>
    %2784 = stablehlo.convert %arg107 : (tensor<256xbf16>) -> tensor<256xf32>
    %2785 = stablehlo.broadcast_in_dim %2783, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2786 = stablehlo.broadcast_in_dim %2784, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2787 = stablehlo.add %2785, %2786 : tensor<100x1x256xf32>
    %2788 = stablehlo.convert %2787 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2789 = stablehlo.reshape %2788 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2790 = stablehlo.convert %2789 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2791 = stablehlo.dot_general %2790, %arg401, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x2048xf32>) -> tensor<100x2048xf32>
    %2792 = stablehlo.broadcast_in_dim %2791, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %2793 = stablehlo.multiply %2792, %1575 : tensor<100x2048xf32>
    %2794 = stablehlo.broadcast_in_dim %2793, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %2795 = stablehlo.broadcast_in_dim %arg402, dims = [1] : (tensor<2048xf32>) -> tensor<100x2048xf32>
    %2796 = stablehlo.add %2794, %2795 : tensor<100x2048xf32>
    %2797 = stablehlo.convert %2796 : (tensor<100x2048xf32>) -> tensor<100x2048xbf16>
    %2798 = stablehlo.reshape %2797 : (tensor<100x2048xbf16>) -> tensor<100x1x2048xbf16>
    %2799 = stablehlo.maximum %2798, %cst_16 : tensor<100x1x2048xbf16>
    %2800 = stablehlo.reshape %2799 : (tensor<100x1x2048xbf16>) -> tensor<100x2048xbf16>
    %2801 = stablehlo.convert %2800 : (tensor<100x2048xbf16>) -> tensor<100x2048xf32>
    %2802 = stablehlo.dot_general %2801, %arg403, contracting_dims = [1] x [0] : (tensor<100x2048xf32>, tensor<2048x256xf32>) -> tensor<100x256xf32>
    %2803 = stablehlo.broadcast_in_dim %2802, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2804 = stablehlo.multiply %2803, %1523 : tensor<100x256xf32>
    %2805 = stablehlo.broadcast_in_dim %2804, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2806 = stablehlo.broadcast_in_dim %arg404, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2807 = stablehlo.add %2805, %2806 : tensor<100x256xf32>
    %2808 = stablehlo.convert %2807 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2809 = stablehlo.reshape %2808 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2810 = stablehlo.add %2788, %2809 : tensor<100x1x256xbf16>
    %2811 = stablehlo.convert %2810 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2812 = stablehlo.convert %2811 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2813 = stablehlo.reduce(%2812 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2814 = stablehlo.reshape %2813 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2815 = stablehlo.broadcast_in_dim %2814, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2816 = stablehlo.divide %2815, %1536 : tensor<100x1x1xf64>
    %2817 = stablehlo.broadcast_in_dim %2812, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2818 = stablehlo.broadcast_in_dim %2816, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2819 = stablehlo.subtract %2817, %2818 : tensor<100x1x256xf64>
    %2820 = stablehlo.multiply %2819, %2819 : tensor<100x1x256xf64>
    %2821 = stablehlo.reduce(%2820 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2822 = stablehlo.reshape %2821 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2823 = stablehlo.broadcast_in_dim %2822, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2824 = stablehlo.divide %2823, %1536 : tensor<100x1x1xf64>
    %2825 = stablehlo.convert %2824 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2826 = stablehlo.reduce(%2811 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2827 = stablehlo.reshape %2826 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2828 = stablehlo.broadcast_in_dim %2827, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2829 = stablehlo.divide %2828, %1550 : tensor<100x1x1xf32>
    %2830 = stablehlo.broadcast_in_dim %2825, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2831 = stablehlo.add %2830, %1553 : tensor<100x1x1xf32>
    %2832 = stablehlo.rsqrt %2831 : tensor<100x1x1xf32>
    %2833 = stablehlo.broadcast_in_dim %2811, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2834 = stablehlo.broadcast_in_dim %2829, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2835 = stablehlo.subtract %2833, %2834 : tensor<100x1x256xf32>
    %2836 = stablehlo.broadcast_in_dim %2835, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2837 = stablehlo.broadcast_in_dim %2832, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2838 = stablehlo.multiply %2836, %2837 : tensor<100x1x256xf32>
    %2839 = stablehlo.convert %arg108 : (tensor<256xbf16>) -> tensor<256xf32>
    %2840 = stablehlo.broadcast_in_dim %2838, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2841 = stablehlo.broadcast_in_dim %2839, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2842 = stablehlo.multiply %2840, %2841 : tensor<100x1x256xf32>
    %2843 = stablehlo.convert %arg109 : (tensor<256xbf16>) -> tensor<256xf32>
    %2844 = stablehlo.broadcast_in_dim %2842, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2845 = stablehlo.broadcast_in_dim %2843, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2846 = stablehlo.add %2844, %2845 : tensor<100x1x256xf32>
    %2847 = stablehlo.convert %2846 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2848 = stablehlo.convert %2847 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2849 = stablehlo.convert %2848 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2850 = stablehlo.reduce(%2849 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2851 = stablehlo.reshape %2850 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2852 = stablehlo.broadcast_in_dim %2851, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2853 = stablehlo.divide %2852, %1536 : tensor<100x1x1xf64>
    %2854 = stablehlo.broadcast_in_dim %2849, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2855 = stablehlo.broadcast_in_dim %2853, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2856 = stablehlo.subtract %2854, %2855 : tensor<100x1x256xf64>
    %2857 = stablehlo.multiply %2856, %2856 : tensor<100x1x256xf64>
    %2858 = stablehlo.reduce(%2857 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2859 = stablehlo.reshape %2858 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2860 = stablehlo.broadcast_in_dim %2859, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2861 = stablehlo.divide %2860, %1536 : tensor<100x1x1xf64>
    %2862 = stablehlo.convert %2861 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2863 = stablehlo.reduce(%2848 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2864 = stablehlo.reshape %2863 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2865 = stablehlo.broadcast_in_dim %2864, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2866 = stablehlo.divide %2865, %1550 : tensor<100x1x1xf32>
    %2867 = stablehlo.broadcast_in_dim %2862, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2868 = stablehlo.add %2867, %1553 : tensor<100x1x1xf32>
    %2869 = stablehlo.rsqrt %2868 : tensor<100x1x1xf32>
    %2870 = stablehlo.broadcast_in_dim %2848, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2871 = stablehlo.broadcast_in_dim %2866, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2872 = stablehlo.subtract %2870, %2871 : tensor<100x1x256xf32>
    %2873 = stablehlo.broadcast_in_dim %2872, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2874 = stablehlo.broadcast_in_dim %2869, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2875 = stablehlo.multiply %2873, %2874 : tensor<100x1x256xf32>
    %2876 = stablehlo.broadcast_in_dim %2875, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2877 = stablehlo.multiply %2876, %1661 : tensor<100x1x256xf32>
    %2878 = stablehlo.broadcast_in_dim %2877, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2879 = stablehlo.add %2878, %1665 : tensor<100x1x256xf32>
    %2880 = stablehlo.convert %2879 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2881 = stablehlo.add %2847, %arg320 : tensor<100x1x256xbf16>
    %2882 = stablehlo.reshape %2881 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2883 = stablehlo.convert %2882 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2884 = stablehlo.dot_general %2883, %arg405, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2885 = stablehlo.broadcast_in_dim %2884, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2886 = stablehlo.multiply %2885, %1523 : tensor<100x256xf32>
    %2887 = stablehlo.broadcast_in_dim %2886, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2888 = stablehlo.broadcast_in_dim %arg406, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2889 = stablehlo.add %2887, %2888 : tensor<100x256xf32>
    %2890 = stablehlo.convert %2889 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2891 = stablehlo.reshape %2890 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2892 = stablehlo.dot_general %2883, %arg407, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2893 = stablehlo.broadcast_in_dim %2892, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2894 = stablehlo.multiply %2893, %1523 : tensor<100x256xf32>
    %2895 = stablehlo.broadcast_in_dim %2894, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2896 = stablehlo.broadcast_in_dim %arg408, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2897 = stablehlo.add %2895, %2896 : tensor<100x256xf32>
    %2898 = stablehlo.convert %2897 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2899 = stablehlo.reshape %2898 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2900 = stablehlo.reshape %2847 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2901 = stablehlo.convert %2900 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2902 = stablehlo.dot_general %2901, %arg409, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2903 = stablehlo.broadcast_in_dim %2902, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2904 = stablehlo.multiply %2903, %1523 : tensor<100x256xf32>
    %2905 = stablehlo.broadcast_in_dim %2904, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2906 = stablehlo.broadcast_in_dim %arg410, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2907 = stablehlo.add %2905, %2906 : tensor<100x256xf32>
    %2908 = stablehlo.convert %2907 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2909 = stablehlo.reshape %2908 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2910 = stablehlo.reshape %2891 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2911 = stablehlo.transpose %2910, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2912 = stablehlo.reshape %2899 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2913 = stablehlo.transpose %2912, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2914 = stablehlo.reshape %2909 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %2915 = stablehlo.transpose %2914, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %2916 = stablehlo.broadcast_in_dim %2911, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2917 = stablehlo.multiply %2916, %1704 : tensor<8x100x32xbf16>
    %2918 = stablehlo.transpose %2913, dims = [0, 2, 1] : (tensor<8x100x32xbf16>) -> tensor<8x32x100xbf16>
    %2919 = stablehlo.broadcast_in_dim %2918, dims = [0, 1, 2] : (tensor<8x32x100xbf16>) -> tensor<8x32x100xbf16>
    %2920 = stablehlo.dot_general %2917, %2919, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x100xbf16>) -> tensor<8x100x100xbf16>
    %2921 = stablehlo.convert %2920 : (tensor<8x100x100xbf16>) -> tensor<8x100x100xf32>
    %2922 = stablehlo.reduce(%2921 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2923 = stablehlo.reshape %2922 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2924 = stablehlo.broadcast_in_dim %2921, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2925 = stablehlo.broadcast_in_dim %2923, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2926 = stablehlo.subtract %2924, %2925 : tensor<8x100x100xf32>
    %2927 = stablehlo.exponential %2926 : tensor<8x100x100xf32>
    %2928 = stablehlo.reduce(%2927 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x100xf32>, tensor<f32>) -> tensor<8x100xf32>
    %2929 = stablehlo.reshape %2928 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %2930 = stablehlo.broadcast_in_dim %2927, dims = [0, 1, 2] : (tensor<8x100x100xf32>) -> tensor<8x100x100xf32>
    %2931 = stablehlo.broadcast_in_dim %2929, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x100xf32>
    %2932 = stablehlo.divide %2930, %2931 : tensor<8x100x100xf32>
    %2933 = stablehlo.convert %2932 : (tensor<8x100x100xf32>) -> tensor<8x100x100xbf16>
    %2934 = stablehlo.broadcast_in_dim %2915, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2935 = stablehlo.dot_general %2933, %2934, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x100xbf16>, tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %2936 = stablehlo.transpose %2935, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %2937 = stablehlo.reshape %2936 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %2938 = stablehlo.convert %2937 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2939 = stablehlo.dot_general %2938, %arg411, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2940 = stablehlo.broadcast_in_dim %2939, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2941 = stablehlo.multiply %2940, %1523 : tensor<100x256xf32>
    %2942 = stablehlo.broadcast_in_dim %2941, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2943 = stablehlo.broadcast_in_dim %arg412, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2944 = stablehlo.add %2942, %2943 : tensor<100x256xf32>
    %2945 = stablehlo.convert %2944 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2946 = stablehlo.reshape %2945 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2947 = stablehlo.add %2847, %2946 : tensor<100x1x256xbf16>
    %2948 = stablehlo.convert %2947 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %2949 = stablehlo.convert %2948 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %2950 = stablehlo.reduce(%2949 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2951 = stablehlo.reshape %2950 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2952 = stablehlo.broadcast_in_dim %2951, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2953 = stablehlo.divide %2952, %1536 : tensor<100x1x1xf64>
    %2954 = stablehlo.broadcast_in_dim %2949, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %2955 = stablehlo.broadcast_in_dim %2953, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %2956 = stablehlo.subtract %2954, %2955 : tensor<100x1x256xf64>
    %2957 = stablehlo.multiply %2956, %2956 : tensor<100x1x256xf64>
    %2958 = stablehlo.reduce(%2957 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %2959 = stablehlo.reshape %2958 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %2960 = stablehlo.broadcast_in_dim %2959, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %2961 = stablehlo.divide %2960, %1536 : tensor<100x1x1xf64>
    %2962 = stablehlo.convert %2961 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %2963 = stablehlo.reduce(%2948 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %2964 = stablehlo.reshape %2963 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %2965 = stablehlo.broadcast_in_dim %2964, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2966 = stablehlo.divide %2965, %1550 : tensor<100x1x1xf32>
    %2967 = stablehlo.broadcast_in_dim %2962, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %2968 = stablehlo.add %2967, %1553 : tensor<100x1x1xf32>
    %2969 = stablehlo.rsqrt %2968 : tensor<100x1x1xf32>
    %2970 = stablehlo.broadcast_in_dim %2948, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2971 = stablehlo.broadcast_in_dim %2966, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2972 = stablehlo.subtract %2970, %2971 : tensor<100x1x256xf32>
    %2973 = stablehlo.broadcast_in_dim %2972, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2974 = stablehlo.broadcast_in_dim %2969, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %2975 = stablehlo.multiply %2973, %2974 : tensor<100x1x256xf32>
    %2976 = stablehlo.convert %arg110 : (tensor<256xbf16>) -> tensor<256xf32>
    %2977 = stablehlo.broadcast_in_dim %2975, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2978 = stablehlo.broadcast_in_dim %2976, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2979 = stablehlo.multiply %2977, %2978 : tensor<100x1x256xf32>
    %2980 = stablehlo.convert %arg111 : (tensor<256xbf16>) -> tensor<256xf32>
    %2981 = stablehlo.broadcast_in_dim %2979, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %2982 = stablehlo.broadcast_in_dim %2980, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %2983 = stablehlo.add %2981, %2982 : tensor<100x1x256xf32>
    %2984 = stablehlo.convert %2983 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %2985 = stablehlo.add %2984, %arg320 : tensor<100x1x256xbf16>
    %2986 = stablehlo.reshape %2985 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %2987 = stablehlo.convert %2986 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %2988 = stablehlo.dot_general %2987, %arg413, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %2989 = stablehlo.broadcast_in_dim %2988, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2990 = stablehlo.multiply %2989, %1523 : tensor<100x256xf32>
    %2991 = stablehlo.broadcast_in_dim %2990, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %2992 = stablehlo.broadcast_in_dim %arg414, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %2993 = stablehlo.add %2991, %2992 : tensor<100x256xf32>
    %2994 = stablehlo.convert %2993 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %2995 = stablehlo.reshape %2994 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %2996 = stablehlo.dot_general %1471, %arg415, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %2997 = stablehlo.broadcast_in_dim %2996, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %2998 = stablehlo.multiply %2997, %515 : tensor<920x256xf32>
    %2999 = stablehlo.broadcast_in_dim %2998, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %3000 = stablehlo.broadcast_in_dim %arg416, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %3001 = stablehlo.add %2999, %3000 : tensor<920x256xf32>
    %3002 = stablehlo.convert %3001 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %3003 = stablehlo.reshape %3002 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %3004 = stablehlo.dot_general %1481, %arg417, contracting_dims = [1] x [0] : (tensor<920x256xf32>, tensor<256x256xf32>) -> tensor<920x256xf32>
    %3005 = stablehlo.broadcast_in_dim %3004, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %3006 = stablehlo.multiply %3005, %515 : tensor<920x256xf32>
    %3007 = stablehlo.broadcast_in_dim %3006, dims = [0, 1] : (tensor<920x256xf32>) -> tensor<920x256xf32>
    %3008 = stablehlo.broadcast_in_dim %arg418, dims = [1] : (tensor<256xf32>) -> tensor<920x256xf32>
    %3009 = stablehlo.add %3007, %3008 : tensor<920x256xf32>
    %3010 = stablehlo.convert %3009 : (tensor<920x256xf32>) -> tensor<920x256xbf16>
    %3011 = stablehlo.reshape %3010 : (tensor<920x256xbf16>) -> tensor<920x1x256xbf16>
    %3012 = stablehlo.reshape %2995 : (tensor<100x1x256xbf16>) -> tensor<100x8x32xbf16>
    %3013 = stablehlo.transpose %3012, dims = [1, 0, 2] : (tensor<100x8x32xbf16>) -> tensor<8x100x32xbf16>
    %3014 = stablehlo.reshape %3003 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %3015 = stablehlo.transpose %3014, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %3016 = stablehlo.reshape %3011 : (tensor<920x1x256xbf16>) -> tensor<920x8x32xbf16>
    %3017 = stablehlo.transpose %3016, dims = [1, 0, 2] : (tensor<920x8x32xbf16>) -> tensor<8x920x32xbf16>
    %3018 = stablehlo.broadcast_in_dim %3013, dims = [0, 1, 2] : (tensor<8x100x32xbf16>) -> tensor<8x100x32xbf16>
    %3019 = stablehlo.multiply %3018, %1704 : tensor<8x100x32xbf16>
    %3020 = stablehlo.transpose %3015, dims = [0, 2, 1] : (tensor<8x920x32xbf16>) -> tensor<8x32x920xbf16>
    %3021 = stablehlo.broadcast_in_dim %3020, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    %3022 = stablehlo.dot_general %3019, %3021, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    %3023 = stablehlo.broadcast_in_dim %3022, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %3024 = stablehlo.multiply %3023, %1498 : tensor<8x100x920xbf16>
    %3025 = stablehlo.broadcast_in_dim %3024, dims = [0, 1, 2] : (tensor<8x100x920xbf16>) -> tensor<8x100x920xbf16>
    %3026 = stablehlo.broadcast_in_dim %arg419, dims = [0, 1, 2] : (tensor<8x1x920xbf16>) -> tensor<8x100x920xbf16>
    %3027 = stablehlo.add %3025, %3026 : tensor<8x100x920xbf16>
    %3028 = stablehlo.convert %3027 : (tensor<8x100x920xbf16>) -> tensor<8x100x920xf32>
    %3029 = stablehlo.reduce(%3028 init: %cst_12) applies stablehlo.maximum across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %3030 = stablehlo.reshape %3029 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %3031 = stablehlo.broadcast_in_dim %3028, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %3032 = stablehlo.broadcast_in_dim %3030, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %3033 = stablehlo.subtract %3031, %3032 : tensor<8x100x920xf32>
    %3034 = stablehlo.exponential %3033 : tensor<8x100x920xf32>
    %3035 = stablehlo.reduce(%3034 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x100x920xf32>, tensor<f32>) -> tensor<8x100xf32>
    %3036 = stablehlo.reshape %3035 : (tensor<8x100xf32>) -> tensor<8x100x1xf32>
    %3037 = stablehlo.broadcast_in_dim %3034, dims = [0, 1, 2] : (tensor<8x100x920xf32>) -> tensor<8x100x920xf32>
    %3038 = stablehlo.broadcast_in_dim %3036, dims = [0, 1, 2] : (tensor<8x100x1xf32>) -> tensor<8x100x920xf32>
    %3039 = stablehlo.divide %3037, %3038 : tensor<8x100x920xf32>
    %3040 = stablehlo.convert %3039 : (tensor<8x100x920xf32>) -> tensor<8x100x920xbf16>
    %3041 = stablehlo.broadcast_in_dim %3017, dims = [0, 1, 2] : (tensor<8x920x32xbf16>) -> tensor<8x920x32xbf16>
    %3042 = stablehlo.dot_general %3040, %3041, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x920xbf16>, tensor<8x920x32xbf16>) -> tensor<8x100x32xbf16>
    %3043 = stablehlo.transpose %3042, dims = [1, 0, 2] : (tensor<8x100x32xbf16>) -> tensor<100x8x32xbf16>
    %3044 = stablehlo.reshape %3043 : (tensor<100x8x32xbf16>) -> tensor<100x256xbf16>
    %3045 = stablehlo.convert %3044 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %3046 = stablehlo.dot_general %3045, %arg420, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x256xf32>) -> tensor<100x256xf32>
    %3047 = stablehlo.broadcast_in_dim %3046, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %3048 = stablehlo.multiply %3047, %1523 : tensor<100x256xf32>
    %3049 = stablehlo.broadcast_in_dim %3048, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %3050 = stablehlo.broadcast_in_dim %arg421, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %3051 = stablehlo.add %3049, %3050 : tensor<100x256xf32>
    %3052 = stablehlo.convert %3051 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %3053 = stablehlo.reshape %3052 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %3054 = stablehlo.add %2984, %3053 : tensor<100x1x256xbf16>
    %3055 = stablehlo.convert %3054 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %3056 = stablehlo.convert %3055 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %3057 = stablehlo.reduce(%3056 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %3058 = stablehlo.reshape %3057 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %3059 = stablehlo.broadcast_in_dim %3058, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %3060 = stablehlo.divide %3059, %1536 : tensor<100x1x1xf64>
    %3061 = stablehlo.broadcast_in_dim %3056, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %3062 = stablehlo.broadcast_in_dim %3060, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %3063 = stablehlo.subtract %3061, %3062 : tensor<100x1x256xf64>
    %3064 = stablehlo.multiply %3063, %3063 : tensor<100x1x256xf64>
    %3065 = stablehlo.reduce(%3064 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %3066 = stablehlo.reshape %3065 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %3067 = stablehlo.broadcast_in_dim %3066, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %3068 = stablehlo.divide %3067, %1536 : tensor<100x1x1xf64>
    %3069 = stablehlo.convert %3068 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %3070 = stablehlo.reduce(%3055 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %3071 = stablehlo.reshape %3070 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %3072 = stablehlo.broadcast_in_dim %3071, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %3073 = stablehlo.divide %3072, %1550 : tensor<100x1x1xf32>
    %3074 = stablehlo.broadcast_in_dim %3069, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %3075 = stablehlo.add %3074, %1553 : tensor<100x1x1xf32>
    %3076 = stablehlo.rsqrt %3075 : tensor<100x1x1xf32>
    %3077 = stablehlo.broadcast_in_dim %3055, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3078 = stablehlo.broadcast_in_dim %3073, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %3079 = stablehlo.subtract %3077, %3078 : tensor<100x1x256xf32>
    %3080 = stablehlo.broadcast_in_dim %3079, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3081 = stablehlo.broadcast_in_dim %3076, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %3082 = stablehlo.multiply %3080, %3081 : tensor<100x1x256xf32>
    %3083 = stablehlo.convert %arg112 : (tensor<256xbf16>) -> tensor<256xf32>
    %3084 = stablehlo.broadcast_in_dim %3082, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3085 = stablehlo.broadcast_in_dim %3083, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %3086 = stablehlo.multiply %3084, %3085 : tensor<100x1x256xf32>
    %3087 = stablehlo.convert %arg113 : (tensor<256xbf16>) -> tensor<256xf32>
    %3088 = stablehlo.broadcast_in_dim %3086, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3089 = stablehlo.broadcast_in_dim %3087, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %3090 = stablehlo.add %3088, %3089 : tensor<100x1x256xf32>
    %3091 = stablehlo.convert %3090 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %3092 = stablehlo.reshape %3091 : (tensor<100x1x256xbf16>) -> tensor<100x256xbf16>
    %3093 = stablehlo.convert %3092 : (tensor<100x256xbf16>) -> tensor<100x256xf32>
    %3094 = stablehlo.dot_general %3093, %arg422, contracting_dims = [1] x [0] : (tensor<100x256xf32>, tensor<256x2048xf32>) -> tensor<100x2048xf32>
    %3095 = stablehlo.broadcast_in_dim %3094, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %3096 = stablehlo.multiply %3095, %1575 : tensor<100x2048xf32>
    %3097 = stablehlo.broadcast_in_dim %3096, dims = [0, 1] : (tensor<100x2048xf32>) -> tensor<100x2048xf32>
    %3098 = stablehlo.broadcast_in_dim %arg423, dims = [1] : (tensor<2048xf32>) -> tensor<100x2048xf32>
    %3099 = stablehlo.add %3097, %3098 : tensor<100x2048xf32>
    %3100 = stablehlo.convert %3099 : (tensor<100x2048xf32>) -> tensor<100x2048xbf16>
    %3101 = stablehlo.reshape %3100 : (tensor<100x2048xbf16>) -> tensor<100x1x2048xbf16>
    %3102 = stablehlo.maximum %3101, %cst_16 : tensor<100x1x2048xbf16>
    %3103 = stablehlo.reshape %3102 : (tensor<100x1x2048xbf16>) -> tensor<100x2048xbf16>
    %3104 = stablehlo.convert %3103 : (tensor<100x2048xbf16>) -> tensor<100x2048xf32>
    %3105 = stablehlo.dot_general %3104, %arg424, contracting_dims = [1] x [0] : (tensor<100x2048xf32>, tensor<2048x256xf32>) -> tensor<100x256xf32>
    %3106 = stablehlo.broadcast_in_dim %3105, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %3107 = stablehlo.multiply %3106, %1523 : tensor<100x256xf32>
    %3108 = stablehlo.broadcast_in_dim %3107, dims = [0, 1] : (tensor<100x256xf32>) -> tensor<100x256xf32>
    %3109 = stablehlo.broadcast_in_dim %arg425, dims = [1] : (tensor<256xf32>) -> tensor<100x256xf32>
    %3110 = stablehlo.add %3108, %3109 : tensor<100x256xf32>
    %3111 = stablehlo.convert %3110 : (tensor<100x256xf32>) -> tensor<100x256xbf16>
    %3112 = stablehlo.reshape %3111 : (tensor<100x256xbf16>) -> tensor<100x1x256xbf16>
    %3113 = stablehlo.add %3091, %3112 : tensor<100x1x256xbf16>
    %3114 = stablehlo.convert %3113 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %3115 = stablehlo.convert %3114 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %3116 = stablehlo.reduce(%3115 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %3117 = stablehlo.reshape %3116 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %3118 = stablehlo.broadcast_in_dim %3117, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %3119 = stablehlo.divide %3118, %1536 : tensor<100x1x1xf64>
    %3120 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %3121 = stablehlo.broadcast_in_dim %3119, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %3122 = stablehlo.subtract %3120, %3121 : tensor<100x1x256xf64>
    %3123 = stablehlo.multiply %3122, %3122 : tensor<100x1x256xf64>
    %3124 = stablehlo.reduce(%3123 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %3125 = stablehlo.reshape %3124 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %3126 = stablehlo.broadcast_in_dim %3125, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %3127 = stablehlo.divide %3126, %1536 : tensor<100x1x1xf64>
    %3128 = stablehlo.convert %3127 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %3129 = stablehlo.reduce(%3114 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %3130 = stablehlo.reshape %3129 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %3131 = stablehlo.broadcast_in_dim %3130, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %3132 = stablehlo.divide %3131, %1550 : tensor<100x1x1xf32>
    %3133 = stablehlo.broadcast_in_dim %3128, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %3134 = stablehlo.add %3133, %1553 : tensor<100x1x1xf32>
    %3135 = stablehlo.rsqrt %3134 : tensor<100x1x1xf32>
    %3136 = stablehlo.broadcast_in_dim %3114, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3137 = stablehlo.broadcast_in_dim %3132, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %3138 = stablehlo.subtract %3136, %3137 : tensor<100x1x256xf32>
    %3139 = stablehlo.broadcast_in_dim %3138, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3140 = stablehlo.broadcast_in_dim %3135, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %3141 = stablehlo.multiply %3139, %3140 : tensor<100x1x256xf32>
    %3142 = stablehlo.convert %arg114 : (tensor<256xbf16>) -> tensor<256xf32>
    %3143 = stablehlo.broadcast_in_dim %3141, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3144 = stablehlo.broadcast_in_dim %3142, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %3145 = stablehlo.multiply %3143, %3144 : tensor<100x1x256xf32>
    %3146 = stablehlo.convert %arg115 : (tensor<256xbf16>) -> tensor<256xf32>
    %3147 = stablehlo.broadcast_in_dim %3145, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3148 = stablehlo.broadcast_in_dim %3146, dims = [2] : (tensor<256xf32>) -> tensor<100x1x256xf32>
    %3149 = stablehlo.add %3147, %3148 : tensor<100x1x256xf32>
    %3150 = stablehlo.convert %3149 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %3151 = stablehlo.convert %3150 : (tensor<100x1x256xbf16>) -> tensor<100x1x256xf32>
    %3152 = stablehlo.convert %3151 : (tensor<100x1x256xf32>) -> tensor<100x1x256xf64>
    %3153 = stablehlo.reduce(%3152 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %3154 = stablehlo.reshape %3153 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %3155 = stablehlo.broadcast_in_dim %3154, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %3156 = stablehlo.divide %3155, %1536 : tensor<100x1x1xf64>
    %3157 = stablehlo.broadcast_in_dim %3152, dims = [0, 1, 2] : (tensor<100x1x256xf64>) -> tensor<100x1x256xf64>
    %3158 = stablehlo.broadcast_in_dim %3156, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x256xf64>
    %3159 = stablehlo.subtract %3157, %3158 : tensor<100x1x256xf64>
    %3160 = stablehlo.multiply %3159, %3159 : tensor<100x1x256xf64>
    %3161 = stablehlo.reduce(%3160 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf64>, tensor<f64>) -> tensor<100x1xf64>
    %3162 = stablehlo.reshape %3161 : (tensor<100x1xf64>) -> tensor<100x1x1xf64>
    %3163 = stablehlo.broadcast_in_dim %3162, dims = [0, 1, 2] : (tensor<100x1x1xf64>) -> tensor<100x1x1xf64>
    %3164 = stablehlo.divide %3163, %1536 : tensor<100x1x1xf64>
    %3165 = stablehlo.convert %3164 : (tensor<100x1x1xf64>) -> tensor<100x1x1xf32>
    %3166 = stablehlo.reduce(%3151 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<100x1x256xf32>, tensor<f32>) -> tensor<100x1xf32>
    %3167 = stablehlo.reshape %3166 : (tensor<100x1xf32>) -> tensor<100x1x1xf32>
    %3168 = stablehlo.broadcast_in_dim %3167, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %3169 = stablehlo.divide %3168, %1550 : tensor<100x1x1xf32>
    %3170 = stablehlo.broadcast_in_dim %3165, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x1xf32>
    %3171 = stablehlo.add %3170, %1553 : tensor<100x1x1xf32>
    %3172 = stablehlo.rsqrt %3171 : tensor<100x1x1xf32>
    %3173 = stablehlo.broadcast_in_dim %3151, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3174 = stablehlo.broadcast_in_dim %3169, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %3175 = stablehlo.subtract %3173, %3174 : tensor<100x1x256xf32>
    %3176 = stablehlo.broadcast_in_dim %3175, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3177 = stablehlo.broadcast_in_dim %3172, dims = [0, 1, 2] : (tensor<100x1x1xf32>) -> tensor<100x1x256xf32>
    %3178 = stablehlo.multiply %3176, %3177 : tensor<100x1x256xf32>
    %3179 = stablehlo.broadcast_in_dim %3178, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3180 = stablehlo.multiply %3179, %1661 : tensor<100x1x256xf32>
    %3181 = stablehlo.broadcast_in_dim %3180, dims = [0, 1, 2] : (tensor<100x1x256xf32>) -> tensor<100x1x256xf32>
    %3182 = stablehlo.add %3181, %1665 : tensor<100x1x256xf32>
    %3183 = stablehlo.convert %3182 : (tensor<100x1x256xf32>) -> tensor<100x1x256xbf16>
    %3184 = stablehlo.reshape %1667 : (tensor<100x1x256xbf16>) -> tensor<1x100x1x256xbf16>
    %3185 = stablehlo.reshape %1971 : (tensor<100x1x256xbf16>) -> tensor<1x100x1x256xbf16>
    %3186 = stablehlo.reshape %2274 : (tensor<100x1x256xbf16>) -> tensor<1x100x1x256xbf16>
    %3187 = stablehlo.reshape %2577 : (tensor<100x1x256xbf16>) -> tensor<1x100x1x256xbf16>
    %3188 = stablehlo.reshape %2880 : (tensor<100x1x256xbf16>) -> tensor<1x100x1x256xbf16>
    %3189 = stablehlo.reshape %3183 : (tensor<100x1x256xbf16>) -> tensor<1x100x1x256xbf16>
    %3190 = stablehlo.concatenate %3184, %3185, %3186, %3187, %3188, %3189, dim = 0 : (tensor<1x100x1x256xbf16>, tensor<1x100x1x256xbf16>, tensor<1x100x1x256xbf16>, tensor<1x100x1x256xbf16>, tensor<1x100x1x256xbf16>, tensor<1x100x1x256xbf16>) -> tensor<6x100x1x256xbf16>
    %3191 = stablehlo.transpose %3190, dims = [0, 2, 1, 3] : (tensor<6x100x1x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3192 = stablehlo.reshape %3191 : (tensor<6x1x100x256xbf16>) -> tensor<600x256xbf16>
    %3193 = stablehlo.dot_general %3192, %arg426, contracting_dims = [1] x [0] : (tensor<600x256xbf16>, tensor<256x92xbf16>) -> tensor<600x92xbf16>
    %3194 = stablehlo.reshape %3193 : (tensor<600x92xbf16>) -> tensor<6x1x100x92xbf16>
    %3195 = stablehlo.broadcast_in_dim %3194, dims = [0, 1, 2, 3] : (tensor<6x1x100x92xbf16>) -> tensor<6x1x100x92xbf16>
    %3196 = stablehlo.broadcast_in_dim %arg116, dims = [3] : (tensor<92xbf16>) -> tensor<6x1x100x92xbf16>
    %3197 = stablehlo.add %3195, %3196 : tensor<6x1x100x92xbf16>
    %3198 = stablehlo.reshape %3197 : (tensor<6x1x100x92xbf16>) -> tensor<600x92xbf16>
    %3199 = stablehlo.dot_general %3192, %arg427, contracting_dims = [1] x [0] : (tensor<600x256xbf16>, tensor<256x256xbf16>) -> tensor<600x256xbf16>
    %3200 = stablehlo.reshape %3199 : (tensor<600x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3201 = stablehlo.broadcast_in_dim %3200, dims = [0, 1, 2, 3] : (tensor<6x1x100x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3202 = stablehlo.broadcast_in_dim %arg117, dims = [3] : (tensor<256xbf16>) -> tensor<6x1x100x256xbf16>
    %3203 = stablehlo.add %3201, %3202 : tensor<6x1x100x256xbf16>
    %3204 = stablehlo.reshape %3203 : (tensor<6x1x100x256xbf16>) -> tensor<600x256xbf16>
    %3205 = stablehlo.reshape %3204 : (tensor<600x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3206 = stablehlo.maximum %3205, %cst_17 : tensor<6x1x100x256xbf16>
    %3207 = stablehlo.reshape %3206 : (tensor<6x1x100x256xbf16>) -> tensor<600x256xbf16>
    %3208 = stablehlo.dot_general %3207, %arg428, contracting_dims = [1] x [0] : (tensor<600x256xbf16>, tensor<256x256xbf16>) -> tensor<600x256xbf16>
    %3209 = stablehlo.reshape %3208 : (tensor<600x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3210 = stablehlo.broadcast_in_dim %3209, dims = [0, 1, 2, 3] : (tensor<6x1x100x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3211 = stablehlo.broadcast_in_dim %arg118, dims = [3] : (tensor<256xbf16>) -> tensor<6x1x100x256xbf16>
    %3212 = stablehlo.add %3210, %3211 : tensor<6x1x100x256xbf16>
    %3213 = stablehlo.reshape %3212 : (tensor<6x1x100x256xbf16>) -> tensor<600x256xbf16>
    %3214 = stablehlo.reshape %3213 : (tensor<600x256xbf16>) -> tensor<6x1x100x256xbf16>
    %3215 = stablehlo.maximum %3214, %cst_17 : tensor<6x1x100x256xbf16>
    %3216 = stablehlo.reshape %3215 : (tensor<6x1x100x256xbf16>) -> tensor<600x256xbf16>
    %3217 = stablehlo.dot_general %3216, %arg429, contracting_dims = [1] x [0] : (tensor<600x256xbf16>, tensor<256x4xbf16>) -> tensor<600x4xbf16>
    %3218 = stablehlo.reshape %3217 : (tensor<600x4xbf16>) -> tensor<6x1x100x4xbf16>
    %3219 = stablehlo.broadcast_in_dim %3218, dims = [0, 1, 2, 3] : (tensor<6x1x100x4xbf16>) -> tensor<6x1x100x4xbf16>
    %3220 = stablehlo.broadcast_in_dim %arg119, dims = [3] : (tensor<4xbf16>) -> tensor<6x1x100x4xbf16>
    %3221 = stablehlo.add %3219, %3220 : tensor<6x1x100x4xbf16>
    %3222 = stablehlo.reshape %3221 : (tensor<6x1x100x4xbf16>) -> tensor<600x4xbf16>
    %3223 = stablehlo.reshape %3222 : (tensor<600x4xbf16>) -> tensor<6x1x100x4xbf16>
    %3224 = stablehlo.logistic %3223 : tensor<6x1x100x4xbf16>
    %3225 = stablehlo.slice %3224 [5:6, 0:1, 0:100, 0:4] : (tensor<6x1x100x4xbf16>) -> tensor<1x1x100x4xbf16>
    %3226 = stablehlo.reshape %3225 : (tensor<1x1x100x4xbf16>) -> tensor<1x100x4xbf16>
    %3227 = stablehlo.reshape %3198 : (tensor<600x92xbf16>) -> tensor<6x1x100x92xbf16>
    %3228 = stablehlo.slice %3227 [5:6, 0:1, 0:100, 0:92] : (tensor<6x1x100x92xbf16>) -> tensor<1x1x100x92xbf16>
    %3229 = stablehlo.reshape %3228 : (tensor<1x1x100x92xbf16>) -> tensor<1x100x92xbf16>
    return %3229, %3226 : tensor<1x100x92xbf16>, tensor<1x100x4xbf16>
  }
}
