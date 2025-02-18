module {
  func.func @main(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16x1x3x3xf32>, %arg3: tensor<16x16x1x1xf32>, %arg4: tensor<64x16x1x1xf32>, %arg5: tensor<64x1x3x3xf32>, %arg6: tensor<24x64x1x1xf32>, %arg7: tensor<72x24x1x1xf32>, %arg8: tensor<72x1x3x3xf32>, %arg9: tensor<24x72x1x1xf32>, %arg10: tensor<72x24x1x1xf32>, %arg11: tensor<72x1x5x5xf32>, %arg12: tensor<24x72x1x1xf32>, %arg13: tensor<24xf32>, %arg14: tensor<72x24x1x1xf32>, %arg15: tensor<72xf32>, %arg16: tensor<40x72x1x1xf32>, %arg17: tensor<120x40x1x1xf32>, %arg18: tensor<120x1x5x5xf32>, %arg19: tensor<32x120x1x1xf32>, %arg20: tensor<32xf32>, %arg21: tensor<120x32x1x1xf32>, %arg22: tensor<120xf32>, %arg23: tensor<40x120x1x1xf32>, %arg24: tensor<120x40x1x1xf32>, %arg25: tensor<120x1x5x5xf32>, %arg26: tensor<32x120x1x1xf32>, %arg27: tensor<32xf32>, %arg28: tensor<120x32x1x1xf32>, %arg29: tensor<120xf32>, %arg30: tensor<40x120x1x1xf32>, %arg31: tensor<240x40x1x1xf32>, %arg32: tensor<240x1x3x3xf32>, %arg33: tensor<80x240x1x1xf32>, %arg34: tensor<200x80x1x1xf32>, %arg35: tensor<200x1x3x3xf32>, %arg36: tensor<80x200x1x1xf32>, %arg37: tensor<184x80x1x1xf32>, %arg38: tensor<184x1x3x3xf32>, %arg39: tensor<80x184x1x1xf32>, %arg40: tensor<184x80x1x1xf32>, %arg41: tensor<184x1x3x3xf32>, %arg42: tensor<80x184x1x1xf32>, %arg43: tensor<480x80x1x1xf32>, %arg44: tensor<480x1x3x3xf32>, %arg45: tensor<120x480x1x1xf32>, %arg46: tensor<120xf32>, %arg47: tensor<480x120x1x1xf32>, %arg48: tensor<480xf32>, %arg49: tensor<112x480x1x1xf32>, %arg50: tensor<672x112x1x1xf32>, %arg51: tensor<672x1x3x3xf32>, %arg52: tensor<168x672x1x1xf32>, %arg53: tensor<168xf32>, %arg54: tensor<672x168x1x1xf32>, %arg55: tensor<672xf32>, %arg56: tensor<112x672x1x1xf32>, %arg57: tensor<672x112x1x1xf32>, %arg58: tensor<672x1x5x5xf32>, %arg59: tensor<168x672x1x1xf32>, %arg60: tensor<168xf32>, %arg61: tensor<672x168x1x1xf32>, %arg62: tensor<672xf32>, %arg63: tensor<80x672x1x1xf32>, %arg64: tensor<480x80x1x1xf32>, %arg65: tensor<480x1x5x5xf32>, %arg66: tensor<120x480x1x1xf32>, %arg67: tensor<120xf32>, %arg68: tensor<480x120x1x1xf32>, %arg69: tensor<480xf32>, %arg70: tensor<80x480x1x1xf32>, %arg71: tensor<480x80x1x1xf32>, %arg72: tensor<480x1x5x5xf32>, %arg73: tensor<120x480x1x1xf32>, %arg74: tensor<120xf32>, %arg75: tensor<480x120x1x1xf32>, %arg76: tensor<480xf32>, %arg77: tensor<80x480x1x1xf32>, %arg78: tensor<480x80x1x1xf32>, %arg79: tensor<256x480x1x1xf32>, %arg80: tensor<256x1x3x3xf32>, %arg81: tensor<512x256x1x1xf32>, %arg82: tensor<128x512x1x1xf32>, %arg83: tensor<128x1x3x3xf32>, %arg84: tensor<256x128x1x1xf32>, %arg85: tensor<128x256x1x1xf32>, %arg86: tensor<128x1x3x3xf32>, %arg87: tensor<256x128x1x1xf32>, %arg88: tensor<64x256x1x1xf32>, %arg89: tensor<64x1x3x3xf32>, %arg90: tensor<128x64x1x1xf32>, %arg91: tensor<672x1x3x3xf32>, %arg92: tensor<24x672x1x1xf32>, %arg93: tensor<24xf32>, %arg94: tensor<480x1x3x3xf32>, %arg95: tensor<24x480x1x1xf32>, %arg96: tensor<24xf32>, %arg97: tensor<512x1x3x3xf32>, %arg98: tensor<24x512x1x1xf32>, %arg99: tensor<24xf32>, %arg100: tensor<256x1x3x3xf32>, %arg101: tensor<24x256x1x1xf32>, %arg102: tensor<24xf32>, %arg103: tensor<256x1x3x3xf32>, %arg104: tensor<24x256x1x1xf32>, %arg105: tensor<24xf32>, %arg106: tensor<128x1x3x3xf32>, %arg107: tensor<24x128x1x1xf32>, %arg108: tensor<24xf32>, %arg109: tensor<672x1x3x3xf32>, %arg110: tensor<546x672x1x1xf32>, %arg111: tensor<546xf32>, %arg112: tensor<480x1x3x3xf32>, %arg113: tensor<546x480x1x1xf32>, %arg114: tensor<546xf32>, %arg115: tensor<512x1x3x3xf32>, %arg116: tensor<546x512x1x1xf32>, %arg117: tensor<546xf32>, %arg118: tensor<256x1x3x3xf32>, %arg119: tensor<546x256x1x1xf32>, %arg120: tensor<546xf32>, %arg121: tensor<256x1x3x3xf32>, %arg122: tensor<546x256x1x1xf32>, %arg123: tensor<546xf32>, %arg124: tensor<128x1x3x3xf32>, %arg125: tensor<546x128x1x1xf32>, %arg126: tensor<546xf32>, %arg127: tensor<3x1x1xf32>, %arg128: tensor<3x1x1xf32>, %arg129: tensor<3x320x320xf32>, %arg130: tensor<3x320x320xf32>, %arg131: tensor<16x1x1xf32>, %arg132: tensor<16x1x1xf32>, %arg133: tensor<16x1x1xf32>, %arg134: tensor<16x1x1xf32>, %arg135: tensor<16x1x1xf32>, %arg136: tensor<16x1x1xf32>, %arg137: tensor<16x1x1xf32>, %arg138: tensor<16x1x1xf32>, %arg139: tensor<16x1x1xf32>, %arg140: tensor<16x1x1xf32>, %arg141: tensor<16x1x1xf32>, %arg142: tensor<16x1x1xf32>, %arg143: tensor<64x1x1xf32>, %arg144: tensor<64x1x1xf32>, %arg145: tensor<64x1x1xf32>, %arg146: tensor<64x1x1xf32>, %arg147: tensor<64x1x1xf32>, %arg148: tensor<64x1x1xf32>, %arg149: tensor<64x1x1xf32>, %arg150: tensor<64x1x1xf32>, %arg151: tensor<24x1x1xf32>, %arg152: tensor<24x1x1xf32>, %arg153: tensor<24x1x1xf32>, %arg154: tensor<24x1x1xf32>, %arg155: tensor<72x1x1xf32>, %arg156: tensor<72x1x1xf32>, %arg157: tensor<72x1x1xf32>, %arg158: tensor<72x1x1xf32>, %arg159: tensor<72x1x1xf32>, %arg160: tensor<72x1x1xf32>, %arg161: tensor<72x1x1xf32>, %arg162: tensor<72x1x1xf32>, %arg163: tensor<24x1x1xf32>, %arg164: tensor<24x1x1xf32>, %arg165: tensor<24x1x1xf32>, %arg166: tensor<24x1x1xf32>, %arg167: tensor<72x1x1xf32>, %arg168: tensor<72x1x1xf32>, %arg169: tensor<72x1x1xf32>, %arg170: tensor<72x1x1xf32>, %arg171: tensor<72x1x1xf32>, %arg172: tensor<72x1x1xf32>, %arg173: tensor<72x1x1xf32>, %arg174: tensor<72x1x1xf32>, %arg175: tensor<40x1x1xf32>, %arg176: tensor<40x1x1xf32>, %arg177: tensor<40x1x1xf32>, %arg178: tensor<40x1x1xf32>, %arg179: tensor<120x1x1xf32>, %arg180: tensor<120x1x1xf32>, %arg181: tensor<120x1x1xf32>, %arg182: tensor<120x1x1xf32>, %arg183: tensor<120x1x1xf32>, %arg184: tensor<120x1x1xf32>, %arg185: tensor<120x1x1xf32>, %arg186: tensor<120x1x1xf32>, %arg187: tensor<40x1x1xf32>, %arg188: tensor<40x1x1xf32>, %arg189: tensor<40x1x1xf32>, %arg190: tensor<40x1x1xf32>, %arg191: tensor<120x1x1xf32>, %arg192: tensor<120x1x1xf32>, %arg193: tensor<120x1x1xf32>, %arg194: tensor<120x1x1xf32>, %arg195: tensor<120x1x1xf32>, %arg196: tensor<120x1x1xf32>, %arg197: tensor<120x1x1xf32>, %arg198: tensor<120x1x1xf32>, %arg199: tensor<40x1x1xf32>, %arg200: tensor<40x1x1xf32>, %arg201: tensor<40x1x1xf32>, %arg202: tensor<40x1x1xf32>, %arg203: tensor<240x1x1xf32>, %arg204: tensor<240x1x1xf32>, %arg205: tensor<240x1x1xf32>, %arg206: tensor<240x1x1xf32>, %arg207: tensor<240x1x1xf32>, %arg208: tensor<240x1x1xf32>, %arg209: tensor<240x1x1xf32>, %arg210: tensor<240x1x1xf32>, %arg211: tensor<80x1x1xf32>, %arg212: tensor<80x1x1xf32>, %arg213: tensor<80x1x1xf32>, %arg214: tensor<80x1x1xf32>, %arg215: tensor<200x1x1xf32>, %arg216: tensor<200x1x1xf32>, %arg217: tensor<200x1x1xf32>, %arg218: tensor<200x1x1xf32>, %arg219: tensor<200x1x1xf32>, %arg220: tensor<200x1x1xf32>, %arg221: tensor<200x1x1xf32>, %arg222: tensor<200x1x1xf32>, %arg223: tensor<80x1x1xf32>, %arg224: tensor<80x1x1xf32>, %arg225: tensor<80x1x1xf32>, %arg226: tensor<80x1x1xf32>, %arg227: tensor<184x1x1xf32>, %arg228: tensor<184x1x1xf32>, %arg229: tensor<184x1x1xf32>, %arg230: tensor<184x1x1xf32>, %arg231: tensor<184x1x1xf32>, %arg232: tensor<184x1x1xf32>, %arg233: tensor<184x1x1xf32>, %arg234: tensor<184x1x1xf32>, %arg235: tensor<80x1x1xf32>, %arg236: tensor<80x1x1xf32>, %arg237: tensor<80x1x1xf32>, %arg238: tensor<80x1x1xf32>, %arg239: tensor<184x1x1xf32>, %arg240: tensor<184x1x1xf32>, %arg241: tensor<184x1x1xf32>, %arg242: tensor<184x1x1xf32>, %arg243: tensor<184x1x1xf32>, %arg244: tensor<184x1x1xf32>, %arg245: tensor<184x1x1xf32>, %arg246: tensor<184x1x1xf32>, %arg247: tensor<80x1x1xf32>, %arg248: tensor<80x1x1xf32>, %arg249: tensor<80x1x1xf32>, %arg250: tensor<80x1x1xf32>, %arg251: tensor<480x1x1xf32>, %arg252: tensor<480x1x1xf32>, %arg253: tensor<480x1x1xf32>, %arg254: tensor<480x1x1xf32>, %arg255: tensor<480x1x1xf32>, %arg256: tensor<480x1x1xf32>, %arg257: tensor<480x1x1xf32>, %arg258: tensor<480x1x1xf32>, %arg259: tensor<112x1x1xf32>, %arg260: tensor<112x1x1xf32>, %arg261: tensor<112x1x1xf32>, %arg262: tensor<112x1x1xf32>, %arg263: tensor<672x1x1xf32>, %arg264: tensor<672x1x1xf32>, %arg265: tensor<672x1x1xf32>, %arg266: tensor<672x1x1xf32>, %arg267: tensor<672x1x1xf32>, %arg268: tensor<672x1x1xf32>, %arg269: tensor<672x1x1xf32>, %arg270: tensor<672x1x1xf32>, %arg271: tensor<112x1x1xf32>, %arg272: tensor<112x1x1xf32>, %arg273: tensor<112x1x1xf32>, %arg274: tensor<112x1x1xf32>, %arg275: tensor<672x1x1xf32>, %arg276: tensor<672x1x1xf32>, %arg277: tensor<672x1x1xf32>, %arg278: tensor<672x1x1xf32>, %arg279: tensor<672x1x1xf32>, %arg280: tensor<672x1x1xf32>, %arg281: tensor<672x1x1xf32>, %arg282: tensor<672x1x1xf32>, %arg283: tensor<80x1x1xf32>, %arg284: tensor<80x1x1xf32>, %arg285: tensor<80x1x1xf32>, %arg286: tensor<80x1x1xf32>, %arg287: tensor<480x1x1xf32>, %arg288: tensor<480x1x1xf32>, %arg289: tensor<480x1x1xf32>, %arg290: tensor<480x1x1xf32>, %arg291: tensor<480x1x1xf32>, %arg292: tensor<480x1x1xf32>, %arg293: tensor<480x1x1xf32>, %arg294: tensor<480x1x1xf32>, %arg295: tensor<80x1x1xf32>, %arg296: tensor<80x1x1xf32>, %arg297: tensor<80x1x1xf32>, %arg298: tensor<80x1x1xf32>, %arg299: tensor<480x1x1xf32>, %arg300: tensor<480x1x1xf32>, %arg301: tensor<480x1x1xf32>, %arg302: tensor<480x1x1xf32>, %arg303: tensor<480x1x1xf32>, %arg304: tensor<480x1x1xf32>, %arg305: tensor<480x1x1xf32>, %arg306: tensor<480x1x1xf32>, %arg307: tensor<80x1x1xf32>, %arg308: tensor<80x1x1xf32>, %arg309: tensor<80x1x1xf32>, %arg310: tensor<80x1x1xf32>, %arg311: tensor<480x1x1xf32>, %arg312: tensor<480x1x1xf32>, %arg313: tensor<480x1x1xf32>, %arg314: tensor<480x1x1xf32>, %arg315: tensor<256x1x1xf32>, %arg316: tensor<256x1x1xf32>, %arg317: tensor<256x1x1xf32>, %arg318: tensor<256x1x1xf32>, %arg319: tensor<256x1x1xf32>, %arg320: tensor<256x1x1xf32>, %arg321: tensor<256x1x1xf32>, %arg322: tensor<256x1x1xf32>, %arg323: tensor<512x1x1xf32>, %arg324: tensor<512x1x1xf32>, %arg325: tensor<512x1x1xf32>, %arg326: tensor<512x1x1xf32>, %arg327: tensor<128x1x1xf32>, %arg328: tensor<128x1x1xf32>, %arg329: tensor<128x1x1xf32>, %arg330: tensor<128x1x1xf32>, %arg331: tensor<128x1x1xf32>, %arg332: tensor<128x1x1xf32>, %arg333: tensor<128x1x1xf32>, %arg334: tensor<128x1x1xf32>, %arg335: tensor<256x1x1xf32>, %arg336: tensor<256x1x1xf32>, %arg337: tensor<256x1x1xf32>, %arg338: tensor<256x1x1xf32>, %arg339: tensor<128x1x1xf32>, %arg340: tensor<128x1x1xf32>, %arg341: tensor<128x1x1xf32>, %arg342: tensor<128x1x1xf32>, %arg343: tensor<128x1x1xf32>, %arg344: tensor<128x1x1xf32>, %arg345: tensor<128x1x1xf32>, %arg346: tensor<128x1x1xf32>, %arg347: tensor<256x1x1xf32>, %arg348: tensor<256x1x1xf32>, %arg349: tensor<256x1x1xf32>, %arg350: tensor<256x1x1xf32>, %arg351: tensor<64x1x1xf32>, %arg352: tensor<64x1x1xf32>, %arg353: tensor<64x1x1xf32>, %arg354: tensor<64x1x1xf32>, %arg355: tensor<64x1x1xf32>, %arg356: tensor<64x1x1xf32>, %arg357: tensor<64x1x1xf32>, %arg358: tensor<64x1x1xf32>, %arg359: tensor<128x1x1xf32>, %arg360: tensor<128x1x1xf32>, %arg361: tensor<128x1x1xf32>, %arg362: tensor<128x1x1xf32>, %arg363: tensor<672x1x1xf32>, %arg364: tensor<672x1x1xf32>, %arg365: tensor<672x1x1xf32>, %arg366: tensor<672x1x1xf32>, %arg367: tensor<480x1x1xf32>, %arg368: tensor<480x1x1xf32>, %arg369: tensor<480x1x1xf32>, %arg370: tensor<480x1x1xf32>, %arg371: tensor<512x1x1xf32>, %arg372: tensor<512x1x1xf32>, %arg373: tensor<512x1x1xf32>, %arg374: tensor<512x1x1xf32>, %arg375: tensor<256x1x1xf32>, %arg376: tensor<256x1x1xf32>, %arg377: tensor<256x1x1xf32>, %arg378: tensor<256x1x1xf32>, %arg379: tensor<256x1x1xf32>, %arg380: tensor<256x1x1xf32>, %arg381: tensor<256x1x1xf32>, %arg382: tensor<256x1x1xf32>, %arg383: tensor<128x1x1xf32>, %arg384: tensor<128x1x1xf32>, %arg385: tensor<128x1x1xf32>, %arg386: tensor<128x1x1xf32>, %arg387: tensor<672x1x1xf32>, %arg388: tensor<672x1x1xf32>, %arg389: tensor<672x1x1xf32>, %arg390: tensor<672x1x1xf32>, %arg391: tensor<480x1x1xf32>, %arg392: tensor<480x1x1xf32>, %arg393: tensor<480x1x1xf32>, %arg394: tensor<480x1x1xf32>, %arg395: tensor<512x1x1xf32>, %arg396: tensor<512x1x1xf32>, %arg397: tensor<512x1x1xf32>, %arg398: tensor<512x1x1xf32>, %arg399: tensor<256x1x1xf32>, %arg400: tensor<256x1x1xf32>, %arg401: tensor<256x1x1xf32>, %arg402: tensor<256x1x1xf32>, %arg403: tensor<256x1x1xf32>, %arg404: tensor<256x1x1xf32>, %arg405: tensor<256x1x1xf32>, %arg406: tensor<256x1x1xf32>, %arg407: tensor<128x1x1xf32>, %arg408: tensor<128x1x1xf32>, %arg409: tensor<128x1x1xf32>, %arg410: tensor<128x1x1xf32>, %arg411: tensor<3234x4xf32>) -> (tensor<1x3234x4xf32>, tensor<1x3234x91xf32>, tensor<3234x4xf32>, tensor<1x3x320x320xf32>) {
    %cst = stablehlo.constant dense<6.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<6> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<1x1xi64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1x16x160x160xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x160x160xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x80x80xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<1x72x80x80xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<1x72x40x40xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<1x24x1x1xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<1x120x40x40xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x1x1xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<1x240x40x40xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<1x240x20x20xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<1x200x20x20xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<1x184x20x20xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<1x480x20x20xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<1x120x1x1xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<1x672x20x20xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<1x168x1x1xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<1x672x10x10xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<1x480x10x10xf32>
    %cst_23 = arith.constant dense<3> : tensor<1xi64>
    %cst_24 = arith.constant dense<6> : tensor<1xi64>
    %cst_25 = arith.constant dense<1600> : tensor<1xi64>
    %cst_26 = arith.constant dense<400> : tensor<1xi64>
    %cst_27 = arith.constant dense<100> : tensor<1xi64>
    %0 = stablehlo.reshape %arg0 : (tensor<1x3x320x320xf32>) -> tensor<3x320x320xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<3x320x320xf32>) -> tensor<3x320x320xf32>
    %2 = stablehlo.broadcast_in_dim %arg127, dims = [0, 1, 2] : (tensor<3x1x1xf32>) -> tensor<3x320x320xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<3x320x320xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<3x320x320xf32>) -> tensor<3x320x320xf32>
    %5 = stablehlo.broadcast_in_dim %arg128, dims = [0, 1, 2] : (tensor<3x1x1xf32>) -> tensor<3x320x320xf32>
    %6 = stablehlo.divide %4, %5 : tensor<3x320x320xf32>
    %7 = stablehlo.reshape %6 : (tensor<3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %8 = stablehlo.transpose %7, dims = [0, 1, 3, 2] : (tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x3x320x320xf32>) -> tensor<3x320x320xf32>
    %10 = stablehlo.broadcast_in_dim %arg129, dims = [0, 1, 2] : (tensor<3x320x320xf32>) -> tensor<3x320x320xf32>
    %11 = stablehlo.dot_general %9, %10, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x320x320xf32>, tensor<3x320x320xf32>) -> tensor<3x320x320xf32>
    %12 = stablehlo.reshape %11 : (tensor<3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %13 = stablehlo.transpose %12, dims = [0, 1, 3, 2] : (tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %14 = stablehlo.reshape %13 : (tensor<1x3x320x320xf32>) -> tensor<3x320x320xf32>
    %15 = stablehlo.broadcast_in_dim %arg130, dims = [0, 1, 2] : (tensor<3x320x320xf32>) -> tensor<3x320x320xf32>
    %16 = stablehlo.dot_general %14, %15, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x320x320xf32>, tensor<3x320x320xf32>) -> tensor<3x320x320xf32>
    %17 = stablehlo.reshape %16 : (tensor<3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x3x320x320xf32>) -> tensor<3x320x320xf32>
    %19 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1x3x320x320xf32>
    %21 = stablehlo.reshape %18 : (tensor<3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %22 = "stablehlo.scatter"(%20, %c_3, %21) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg412: tensor<f32>, %arg413: tensor<f32>):
      stablehlo.return %arg413 : tensor<f32>
    }) : (tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    %23 = stablehlo.convolution(%22, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x320x320xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x160x160xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %25 = stablehlo.broadcast_in_dim %arg131, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %26 = stablehlo.subtract %24, %25 : tensor<1x16x160x160xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %28 = stablehlo.broadcast_in_dim %arg132, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<1x16x160x160xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %31 = stablehlo.broadcast_in_dim %arg133, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<1x16x160x160xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %34 = stablehlo.broadcast_in_dim %arg134, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %35 = stablehlo.add %33, %34 : tensor<1x16x160x160xf32>
    %36 = stablehlo.convert %cst_23 : (tensor<1xi64>) -> tensor<1xf32>
    %37 = stablehlo.reshape %36 : (tensor<1xf32>) -> tensor<f32>
    %38 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x16x160x160xf32>
    %40 = stablehlo.add %38, %39 : tensor<1x16x160x160xf32>
    %41 = stablehlo.maximum %40, %cst_4 : tensor<1x16x160x160xf32>
    %42 = stablehlo.convert %c_1 : (tensor<i64>) -> tensor<f32>
    %43 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x16x160x160xf32>
    %45 = stablehlo.minimum %43, %44 : tensor<1x16x160x160xf32>
    %46 = stablehlo.convert %cst_24 : (tensor<1xi64>) -> tensor<1xf32>
    %47 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
    %48 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x16x160x160xf32>
    %50 = stablehlo.divide %48, %49 : tensor<1x16x160x160xf32>
    %51 = stablehlo.multiply %50, %35 : tensor<1x16x160x160xf32>
    %52 = stablehlo.convolution(%51, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 16 : i64} : (tensor<1x16x160x160xf32>, tensor<16x1x3x3xf32>) -> tensor<1x16x160x160xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %54 = stablehlo.broadcast_in_dim %arg135, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %55 = stablehlo.subtract %53, %54 : tensor<1x16x160x160xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %57 = stablehlo.broadcast_in_dim %arg136, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %58 = stablehlo.multiply %56, %57 : tensor<1x16x160x160xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %60 = stablehlo.broadcast_in_dim %arg137, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %61 = stablehlo.multiply %59, %60 : tensor<1x16x160x160xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %63 = stablehlo.broadcast_in_dim %arg138, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %64 = stablehlo.add %62, %63 : tensor<1x16x160x160xf32>
    %65 = stablehlo.maximum %64, %cst_4 : tensor<1x16x160x160xf32>
    %66 = stablehlo.convolution(%65, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x160x160xf32>, tensor<16x16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %68 = stablehlo.broadcast_in_dim %arg139, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %69 = stablehlo.subtract %67, %68 : tensor<1x16x160x160xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %71 = stablehlo.broadcast_in_dim %arg140, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %72 = stablehlo.multiply %70, %71 : tensor<1x16x160x160xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %74 = stablehlo.broadcast_in_dim %arg141, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %75 = stablehlo.multiply %73, %74 : tensor<1x16x160x160xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x16x160x160xf32>) -> tensor<1x16x160x160xf32>
    %77 = stablehlo.broadcast_in_dim %arg142, dims = [1, 2, 3] : (tensor<16x1x1xf32>) -> tensor<1x16x160x160xf32>
    %78 = stablehlo.add %76, %77 : tensor<1x16x160x160xf32>
    %79 = stablehlo.add %78, %51 : tensor<1x16x160x160xf32>
    %80 = stablehlo.convolution(%79, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x160x160xf32>, tensor<64x16x1x1xf32>) -> tensor<1x64x160x160xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<1x64x160x160xf32>) -> tensor<1x64x160x160xf32>
    %82 = stablehlo.broadcast_in_dim %arg143, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x160x160xf32>
    %83 = stablehlo.subtract %81, %82 : tensor<1x64x160x160xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x64x160x160xf32>) -> tensor<1x64x160x160xf32>
    %85 = stablehlo.broadcast_in_dim %arg144, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x160x160xf32>
    %86 = stablehlo.multiply %84, %85 : tensor<1x64x160x160xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2, 3] : (tensor<1x64x160x160xf32>) -> tensor<1x64x160x160xf32>
    %88 = stablehlo.broadcast_in_dim %arg145, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x160x160xf32>
    %89 = stablehlo.multiply %87, %88 : tensor<1x64x160x160xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x64x160x160xf32>) -> tensor<1x64x160x160xf32>
    %91 = stablehlo.broadcast_in_dim %arg146, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x160x160xf32>
    %92 = stablehlo.add %90, %91 : tensor<1x64x160x160xf32>
    %93 = stablehlo.maximum %92, %cst_5 : tensor<1x64x160x160xf32>
    %94 = stablehlo.convolution(%93, %arg5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<1x64x160x160xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x80x80xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x64x80x80xf32>) -> tensor<1x64x80x80xf32>
    %96 = stablehlo.broadcast_in_dim %arg147, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x80x80xf32>
    %97 = stablehlo.subtract %95, %96 : tensor<1x64x80x80xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x64x80x80xf32>) -> tensor<1x64x80x80xf32>
    %99 = stablehlo.broadcast_in_dim %arg148, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x80x80xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<1x64x80x80xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<1x64x80x80xf32>) -> tensor<1x64x80x80xf32>
    %102 = stablehlo.broadcast_in_dim %arg149, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x80x80xf32>
    %103 = stablehlo.multiply %101, %102 : tensor<1x64x80x80xf32>
    %104 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x64x80x80xf32>) -> tensor<1x64x80x80xf32>
    %105 = stablehlo.broadcast_in_dim %arg150, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x80x80xf32>
    %106 = stablehlo.add %104, %105 : tensor<1x64x80x80xf32>
    %107 = stablehlo.maximum %106, %cst_6 : tensor<1x64x80x80xf32>
    %108 = stablehlo.convolution(%107, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x80x80xf32>, tensor<24x64x1x1xf32>) -> tensor<1x24x80x80xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %110 = stablehlo.broadcast_in_dim %arg151, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %111 = stablehlo.subtract %109, %110 : tensor<1x24x80x80xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %113 = stablehlo.broadcast_in_dim %arg152, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<1x24x80x80xf32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %116 = stablehlo.broadcast_in_dim %arg153, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %117 = stablehlo.multiply %115, %116 : tensor<1x24x80x80xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %119 = stablehlo.broadcast_in_dim %arg154, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %120 = stablehlo.add %118, %119 : tensor<1x24x80x80xf32>
    %121 = stablehlo.convolution(%120, %arg7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x24x80x80xf32>, tensor<72x24x1x1xf32>) -> tensor<1x72x80x80xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %123 = stablehlo.broadcast_in_dim %arg155, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %124 = stablehlo.subtract %122, %123 : tensor<1x72x80x80xf32>
    %125 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %126 = stablehlo.broadcast_in_dim %arg156, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %127 = stablehlo.multiply %125, %126 : tensor<1x72x80x80xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %129 = stablehlo.broadcast_in_dim %arg157, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %130 = stablehlo.multiply %128, %129 : tensor<1x72x80x80xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %132 = stablehlo.broadcast_in_dim %arg158, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %133 = stablehlo.add %131, %132 : tensor<1x72x80x80xf32>
    %134 = stablehlo.maximum %133, %cst_7 : tensor<1x72x80x80xf32>
    %135 = stablehlo.convolution(%134, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 72 : i64} : (tensor<1x72x80x80xf32>, tensor<72x1x3x3xf32>) -> tensor<1x72x80x80xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %137 = stablehlo.broadcast_in_dim %arg159, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %138 = stablehlo.subtract %136, %137 : tensor<1x72x80x80xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %140 = stablehlo.broadcast_in_dim %arg160, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %141 = stablehlo.multiply %139, %140 : tensor<1x72x80x80xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %143 = stablehlo.broadcast_in_dim %arg161, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %144 = stablehlo.multiply %142, %143 : tensor<1x72x80x80xf32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %146 = stablehlo.broadcast_in_dim %arg162, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %147 = stablehlo.add %145, %146 : tensor<1x72x80x80xf32>
    %148 = stablehlo.maximum %147, %cst_7 : tensor<1x72x80x80xf32>
    %149 = stablehlo.convolution(%148, %arg9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x72x80x80xf32>, tensor<24x72x1x1xf32>) -> tensor<1x24x80x80xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %151 = stablehlo.broadcast_in_dim %arg163, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %152 = stablehlo.subtract %150, %151 : tensor<1x24x80x80xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %154 = stablehlo.broadcast_in_dim %arg164, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %155 = stablehlo.multiply %153, %154 : tensor<1x24x80x80xf32>
    %156 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %157 = stablehlo.broadcast_in_dim %arg165, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %158 = stablehlo.multiply %156, %157 : tensor<1x24x80x80xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1, 2, 3] : (tensor<1x24x80x80xf32>) -> tensor<1x24x80x80xf32>
    %160 = stablehlo.broadcast_in_dim %arg166, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x80x80xf32>
    %161 = stablehlo.add %159, %160 : tensor<1x24x80x80xf32>
    %162 = stablehlo.add %161, %120 : tensor<1x24x80x80xf32>
    %163 = stablehlo.convolution(%162, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x24x80x80xf32>, tensor<72x24x1x1xf32>) -> tensor<1x72x80x80xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %165 = stablehlo.broadcast_in_dim %arg167, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %166 = stablehlo.subtract %164, %165 : tensor<1x72x80x80xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %168 = stablehlo.broadcast_in_dim %arg168, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %169 = stablehlo.multiply %167, %168 : tensor<1x72x80x80xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %171 = stablehlo.broadcast_in_dim %arg169, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %172 = stablehlo.multiply %170, %171 : tensor<1x72x80x80xf32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x72x80x80xf32>) -> tensor<1x72x80x80xf32>
    %174 = stablehlo.broadcast_in_dim %arg170, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x80x80xf32>
    %175 = stablehlo.add %173, %174 : tensor<1x72x80x80xf32>
    %176 = stablehlo.maximum %175, %cst_7 : tensor<1x72x80x80xf32>
    %177 = stablehlo.convolution(%176, %arg11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 72 : i64} : (tensor<1x72x80x80xf32>, tensor<72x1x5x5xf32>) -> tensor<1x72x40x40xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<1x72x40x40xf32>) -> tensor<1x72x40x40xf32>
    %179 = stablehlo.broadcast_in_dim %arg171, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x40x40xf32>
    %180 = stablehlo.subtract %178, %179 : tensor<1x72x40x40xf32>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2, 3] : (tensor<1x72x40x40xf32>) -> tensor<1x72x40x40xf32>
    %182 = stablehlo.broadcast_in_dim %arg172, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x40x40xf32>
    %183 = stablehlo.multiply %181, %182 : tensor<1x72x40x40xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2, 3] : (tensor<1x72x40x40xf32>) -> tensor<1x72x40x40xf32>
    %185 = stablehlo.broadcast_in_dim %arg173, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x40x40xf32>
    %186 = stablehlo.multiply %184, %185 : tensor<1x72x40x40xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<1x72x40x40xf32>) -> tensor<1x72x40x40xf32>
    %188 = stablehlo.broadcast_in_dim %arg174, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x40x40xf32>
    %189 = stablehlo.add %187, %188 : tensor<1x72x40x40xf32>
    %190 = stablehlo.maximum %189, %cst_8 : tensor<1x72x40x40xf32>
    %191 = stablehlo.reduce(%190 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x72x40x40xf32>, tensor<f32>) -> tensor<1x72xf32>
    %192 = stablehlo.reshape %191 : (tensor<1x72xf32>) -> tensor<1x72x1x1xf32>
    %193 = stablehlo.convert %cst_25 : (tensor<1xi64>) -> tensor<1xf32>
    %194 = stablehlo.reshape %193 : (tensor<1xf32>) -> tensor<f32>
    %195 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %196 = stablehlo.broadcast_in_dim %194, dims = [] : (tensor<f32>) -> tensor<1x72x1x1xf32>
    %197 = stablehlo.divide %195, %196 : tensor<1x72x1x1xf32>
    %198 = stablehlo.convolution(%197, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x72x1x1xf32>, tensor<24x72x1x1xf32>) -> tensor<1x24x1x1xf32>
    %199 = stablehlo.reshape %arg13 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %200 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2, 3] : (tensor<1x24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %201 = stablehlo.broadcast_in_dim %199, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %202 = stablehlo.add %200, %201 : tensor<1x24x1x1xf32>
    %203 = stablehlo.maximum %202, %cst_10 : tensor<1x24x1x1xf32>
    %204 = stablehlo.convolution(%203, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x24x1x1xf32>, tensor<72x24x1x1xf32>) -> tensor<1x72x1x1xf32>
    %205 = stablehlo.reshape %arg15 : (tensor<72xf32>) -> tensor<72x1x1xf32>
    %206 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %207 = stablehlo.broadcast_in_dim %205, dims = [1, 2, 3] : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %208 = stablehlo.add %206, %207 : tensor<1x72x1x1xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %210 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x72x1x1xf32>
    %211 = stablehlo.add %209, %210 : tensor<1x72x1x1xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %213 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x72x1x1xf32>
    %214 = stablehlo.divide %212, %213 : tensor<1x72x1x1xf32>
    %215 = stablehlo.convert %c : (tensor<i64>) -> tensor<f32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [] : (tensor<f32>) -> tensor<1x72x1x1xf32>
    %217 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %218 = stablehlo.minimum %216, %217 : tensor<1x72x1x1xf32>
    %219 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1x72x1x1xf32>
    %220 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %221 = stablehlo.maximum %219, %220 : tensor<1x72x1x1xf32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2, 3] : (tensor<1x72x1x1xf32>) -> tensor<1x72x40x40xf32>
    %223 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2, 3] : (tensor<1x72x40x40xf32>) -> tensor<1x72x40x40xf32>
    %224 = stablehlo.multiply %222, %223 : tensor<1x72x40x40xf32>
    %225 = stablehlo.convolution(%224, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x72x40x40xf32>, tensor<40x72x1x1xf32>) -> tensor<1x40x40x40xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %227 = stablehlo.broadcast_in_dim %arg175, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %228 = stablehlo.subtract %226, %227 : tensor<1x40x40x40xf32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %230 = stablehlo.broadcast_in_dim %arg176, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %231 = stablehlo.multiply %229, %230 : tensor<1x40x40x40xf32>
    %232 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %233 = stablehlo.broadcast_in_dim %arg177, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %234 = stablehlo.multiply %232, %233 : tensor<1x40x40x40xf32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %236 = stablehlo.broadcast_in_dim %arg178, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %237 = stablehlo.add %235, %236 : tensor<1x40x40x40xf32>
    %238 = stablehlo.convolution(%237, %arg17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x40x40x40xf32>, tensor<120x40x1x1xf32>) -> tensor<1x120x40x40xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %240 = stablehlo.broadcast_in_dim %arg179, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %241 = stablehlo.subtract %239, %240 : tensor<1x120x40x40xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %243 = stablehlo.broadcast_in_dim %arg180, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %244 = stablehlo.multiply %242, %243 : tensor<1x120x40x40xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %246 = stablehlo.broadcast_in_dim %arg181, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %247 = stablehlo.multiply %245, %246 : tensor<1x120x40x40xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %249 = stablehlo.broadcast_in_dim %arg182, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %250 = stablehlo.add %248, %249 : tensor<1x120x40x40xf32>
    %251 = stablehlo.maximum %250, %cst_11 : tensor<1x120x40x40xf32>
    %252 = stablehlo.convolution(%251, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 120 : i64} : (tensor<1x120x40x40xf32>, tensor<120x1x5x5xf32>) -> tensor<1x120x40x40xf32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %254 = stablehlo.broadcast_in_dim %arg183, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %255 = stablehlo.subtract %253, %254 : tensor<1x120x40x40xf32>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %257 = stablehlo.broadcast_in_dim %arg184, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %258 = stablehlo.multiply %256, %257 : tensor<1x120x40x40xf32>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %260 = stablehlo.broadcast_in_dim %arg185, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %261 = stablehlo.multiply %259, %260 : tensor<1x120x40x40xf32>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %263 = stablehlo.broadcast_in_dim %arg186, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %264 = stablehlo.add %262, %263 : tensor<1x120x40x40xf32>
    %265 = stablehlo.maximum %264, %cst_11 : tensor<1x120x40x40xf32>
    %266 = stablehlo.reduce(%265 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x120x40x40xf32>, tensor<f32>) -> tensor<1x120xf32>
    %267 = stablehlo.reshape %266 : (tensor<1x120xf32>) -> tensor<1x120x1x1xf32>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %269 = stablehlo.broadcast_in_dim %194, dims = [] : (tensor<f32>) -> tensor<1x120x1x1xf32>
    %270 = stablehlo.divide %268, %269 : tensor<1x120x1x1xf32>
    %271 = stablehlo.convolution(%270, %arg19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x1x1xf32>, tensor<32x120x1x1xf32>) -> tensor<1x32x1x1xf32>
    %272 = stablehlo.reshape %arg20 : (tensor<32xf32>) -> tensor<32x1x1xf32>
    %273 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2, 3] : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %274 = stablehlo.broadcast_in_dim %272, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %275 = stablehlo.add %273, %274 : tensor<1x32x1x1xf32>
    %276 = stablehlo.maximum %275, %cst_12 : tensor<1x32x1x1xf32>
    %277 = stablehlo.convolution(%276, %arg21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x1x1xf32>, tensor<120x32x1x1xf32>) -> tensor<1x120x1x1xf32>
    %278 = stablehlo.reshape %arg22 : (tensor<120xf32>) -> tensor<120x1x1xf32>
    %279 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %280 = stablehlo.broadcast_in_dim %278, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %281 = stablehlo.add %279, %280 : tensor<1x120x1x1xf32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %283 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x120x1x1xf32>
    %284 = stablehlo.add %282, %283 : tensor<1x120x1x1xf32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %286 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x120x1x1xf32>
    %287 = stablehlo.divide %285, %286 : tensor<1x120x1x1xf32>
    %288 = stablehlo.broadcast_in_dim %215, dims = [] : (tensor<f32>) -> tensor<1x120x1x1xf32>
    %289 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %290 = stablehlo.minimum %288, %289 : tensor<1x120x1x1xf32>
    %291 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1x120x1x1xf32>
    %292 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %293 = stablehlo.maximum %291, %292 : tensor<1x120x1x1xf32>
    %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %295 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %296 = stablehlo.multiply %294, %295 : tensor<1x120x40x40xf32>
    %297 = stablehlo.convolution(%296, %arg23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x40x40xf32>, tensor<40x120x1x1xf32>) -> tensor<1x40x40x40xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %299 = stablehlo.broadcast_in_dim %arg187, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %300 = stablehlo.subtract %298, %299 : tensor<1x40x40x40xf32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %302 = stablehlo.broadcast_in_dim %arg188, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %303 = stablehlo.multiply %301, %302 : tensor<1x40x40x40xf32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %305 = stablehlo.broadcast_in_dim %arg189, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %306 = stablehlo.multiply %304, %305 : tensor<1x40x40x40xf32>
    %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %308 = stablehlo.broadcast_in_dim %arg190, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %309 = stablehlo.add %307, %308 : tensor<1x40x40x40xf32>
    %310 = stablehlo.add %309, %237 : tensor<1x40x40x40xf32>
    %311 = stablehlo.convolution(%310, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x40x40x40xf32>, tensor<120x40x1x1xf32>) -> tensor<1x120x40x40xf32>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %313 = stablehlo.broadcast_in_dim %arg191, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %314 = stablehlo.subtract %312, %313 : tensor<1x120x40x40xf32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %316 = stablehlo.broadcast_in_dim %arg192, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %317 = stablehlo.multiply %315, %316 : tensor<1x120x40x40xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %319 = stablehlo.broadcast_in_dim %arg193, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %320 = stablehlo.multiply %318, %319 : tensor<1x120x40x40xf32>
    %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %322 = stablehlo.broadcast_in_dim %arg194, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %323 = stablehlo.add %321, %322 : tensor<1x120x40x40xf32>
    %324 = stablehlo.maximum %323, %cst_11 : tensor<1x120x40x40xf32>
    %325 = stablehlo.convolution(%324, %arg25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 120 : i64} : (tensor<1x120x40x40xf32>, tensor<120x1x5x5xf32>) -> tensor<1x120x40x40xf32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %327 = stablehlo.broadcast_in_dim %arg195, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %328 = stablehlo.subtract %326, %327 : tensor<1x120x40x40xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %330 = stablehlo.broadcast_in_dim %arg196, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %331 = stablehlo.multiply %329, %330 : tensor<1x120x40x40xf32>
    %332 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %333 = stablehlo.broadcast_in_dim %arg197, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %334 = stablehlo.multiply %332, %333 : tensor<1x120x40x40xf32>
    %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %336 = stablehlo.broadcast_in_dim %arg198, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %337 = stablehlo.add %335, %336 : tensor<1x120x40x40xf32>
    %338 = stablehlo.maximum %337, %cst_11 : tensor<1x120x40x40xf32>
    %339 = stablehlo.reduce(%338 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x120x40x40xf32>, tensor<f32>) -> tensor<1x120xf32>
    %340 = stablehlo.reshape %339 : (tensor<1x120xf32>) -> tensor<1x120x1x1xf32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %342 = stablehlo.divide %341, %269 : tensor<1x120x1x1xf32>
    %343 = stablehlo.convolution(%342, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x1x1xf32>, tensor<32x120x1x1xf32>) -> tensor<1x32x1x1xf32>
    %344 = stablehlo.reshape %arg27 : (tensor<32xf32>) -> tensor<32x1x1xf32>
    %345 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %346 = stablehlo.broadcast_in_dim %344, dims = [1, 2, 3] : (tensor<32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %347 = stablehlo.add %345, %346 : tensor<1x32x1x1xf32>
    %348 = stablehlo.maximum %347, %cst_12 : tensor<1x32x1x1xf32>
    %349 = stablehlo.convolution(%348, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x1x1xf32>, tensor<120x32x1x1xf32>) -> tensor<1x120x1x1xf32>
    %350 = stablehlo.reshape %arg29 : (tensor<120xf32>) -> tensor<120x1x1xf32>
    %351 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %352 = stablehlo.broadcast_in_dim %350, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %353 = stablehlo.add %351, %352 : tensor<1x120x1x1xf32>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %355 = stablehlo.add %354, %283 : tensor<1x120x1x1xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %357 = stablehlo.divide %356, %286 : tensor<1x120x1x1xf32>
    %358 = stablehlo.broadcast_in_dim %357, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %359 = stablehlo.minimum %288, %358 : tensor<1x120x1x1xf32>
    %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %361 = stablehlo.maximum %291, %360 : tensor<1x120x1x1xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x40x40xf32>
    %363 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2, 3] : (tensor<1x120x40x40xf32>) -> tensor<1x120x40x40xf32>
    %364 = stablehlo.multiply %362, %363 : tensor<1x120x40x40xf32>
    %365 = stablehlo.convolution(%364, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x40x40xf32>, tensor<40x120x1x1xf32>) -> tensor<1x40x40x40xf32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %367 = stablehlo.broadcast_in_dim %arg199, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %368 = stablehlo.subtract %366, %367 : tensor<1x40x40x40xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %370 = stablehlo.broadcast_in_dim %arg200, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %371 = stablehlo.multiply %369, %370 : tensor<1x40x40x40xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %373 = stablehlo.broadcast_in_dim %arg201, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %374 = stablehlo.multiply %372, %373 : tensor<1x40x40x40xf32>
    %375 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<1x40x40x40xf32>) -> tensor<1x40x40x40xf32>
    %376 = stablehlo.broadcast_in_dim %arg202, dims = [1, 2, 3] : (tensor<40x1x1xf32>) -> tensor<1x40x40x40xf32>
    %377 = stablehlo.add %375, %376 : tensor<1x40x40x40xf32>
    %378 = stablehlo.add %377, %310 : tensor<1x40x40x40xf32>
    %379 = stablehlo.convolution(%378, %arg31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x40x40x40xf32>, tensor<240x40x1x1xf32>) -> tensor<1x240x40x40xf32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %381 = stablehlo.broadcast_in_dim %arg203, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x40x40xf32>
    %382 = stablehlo.subtract %380, %381 : tensor<1x240x40x40xf32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %384 = stablehlo.broadcast_in_dim %arg204, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x40x40xf32>
    %385 = stablehlo.multiply %383, %384 : tensor<1x240x40x40xf32>
    %386 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %387 = stablehlo.broadcast_in_dim %arg205, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x40x40xf32>
    %388 = stablehlo.multiply %386, %387 : tensor<1x240x40x40xf32>
    %389 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %390 = stablehlo.broadcast_in_dim %arg206, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x40x40xf32>
    %391 = stablehlo.add %389, %390 : tensor<1x240x40x40xf32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %393 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x240x40x40xf32>
    %394 = stablehlo.add %392, %393 : tensor<1x240x40x40xf32>
    %395 = stablehlo.maximum %394, %cst_13 : tensor<1x240x40x40xf32>
    %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %397 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x240x40x40xf32>
    %398 = stablehlo.minimum %396, %397 : tensor<1x240x40x40xf32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<1x240x40x40xf32>) -> tensor<1x240x40x40xf32>
    %400 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x240x40x40xf32>
    %401 = stablehlo.divide %399, %400 : tensor<1x240x40x40xf32>
    %402 = stablehlo.multiply %401, %391 : tensor<1x240x40x40xf32>
    %403 = stablehlo.convolution(%402, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<1x240x40x40xf32>, tensor<240x1x3x3xf32>) -> tensor<1x240x20x20xf32>
    %404 = stablehlo.broadcast_in_dim %403, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %405 = stablehlo.broadcast_in_dim %arg207, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x20x20xf32>
    %406 = stablehlo.subtract %404, %405 : tensor<1x240x20x20xf32>
    %407 = stablehlo.broadcast_in_dim %406, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %408 = stablehlo.broadcast_in_dim %arg208, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x20x20xf32>
    %409 = stablehlo.multiply %407, %408 : tensor<1x240x20x20xf32>
    %410 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %411 = stablehlo.broadcast_in_dim %arg209, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x20x20xf32>
    %412 = stablehlo.multiply %410, %411 : tensor<1x240x20x20xf32>
    %413 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %414 = stablehlo.broadcast_in_dim %arg210, dims = [1, 2, 3] : (tensor<240x1x1xf32>) -> tensor<1x240x20x20xf32>
    %415 = stablehlo.add %413, %414 : tensor<1x240x20x20xf32>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %417 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x240x20x20xf32>
    %418 = stablehlo.add %416, %417 : tensor<1x240x20x20xf32>
    %419 = stablehlo.maximum %418, %cst_14 : tensor<1x240x20x20xf32>
    %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %421 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x240x20x20xf32>
    %422 = stablehlo.minimum %420, %421 : tensor<1x240x20x20xf32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2, 3] : (tensor<1x240x20x20xf32>) -> tensor<1x240x20x20xf32>
    %424 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x240x20x20xf32>
    %425 = stablehlo.divide %423, %424 : tensor<1x240x20x20xf32>
    %426 = stablehlo.multiply %425, %415 : tensor<1x240x20x20xf32>
    %427 = stablehlo.convolution(%426, %arg33) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x240x20x20xf32>, tensor<80x240x1x1xf32>) -> tensor<1x80x20x20xf32>
    %428 = stablehlo.broadcast_in_dim %427, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %429 = stablehlo.broadcast_in_dim %arg211, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %430 = stablehlo.subtract %428, %429 : tensor<1x80x20x20xf32>
    %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %432 = stablehlo.broadcast_in_dim %arg212, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %433 = stablehlo.multiply %431, %432 : tensor<1x80x20x20xf32>
    %434 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %435 = stablehlo.broadcast_in_dim %arg213, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %436 = stablehlo.multiply %434, %435 : tensor<1x80x20x20xf32>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %438 = stablehlo.broadcast_in_dim %arg214, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %439 = stablehlo.add %437, %438 : tensor<1x80x20x20xf32>
    %440 = stablehlo.convolution(%439, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x20x20xf32>, tensor<200x80x1x1xf32>) -> tensor<1x200x20x20xf32>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %442 = stablehlo.broadcast_in_dim %arg215, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %443 = stablehlo.subtract %441, %442 : tensor<1x200x20x20xf32>
    %444 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %445 = stablehlo.broadcast_in_dim %arg216, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %446 = stablehlo.multiply %444, %445 : tensor<1x200x20x20xf32>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %448 = stablehlo.broadcast_in_dim %arg217, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %449 = stablehlo.multiply %447, %448 : tensor<1x200x20x20xf32>
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %451 = stablehlo.broadcast_in_dim %arg218, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %452 = stablehlo.add %450, %451 : tensor<1x200x20x20xf32>
    %453 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %454 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x200x20x20xf32>
    %455 = stablehlo.add %453, %454 : tensor<1x200x20x20xf32>
    %456 = stablehlo.maximum %455, %cst_15 : tensor<1x200x20x20xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %458 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x200x20x20xf32>
    %459 = stablehlo.minimum %457, %458 : tensor<1x200x20x20xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %461 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x200x20x20xf32>
    %462 = stablehlo.divide %460, %461 : tensor<1x200x20x20xf32>
    %463 = stablehlo.multiply %462, %452 : tensor<1x200x20x20xf32>
    %464 = stablehlo.convolution(%463, %arg35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 200 : i64} : (tensor<1x200x20x20xf32>, tensor<200x1x3x3xf32>) -> tensor<1x200x20x20xf32>
    %465 = stablehlo.broadcast_in_dim %464, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %466 = stablehlo.broadcast_in_dim %arg219, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %467 = stablehlo.subtract %465, %466 : tensor<1x200x20x20xf32>
    %468 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %469 = stablehlo.broadcast_in_dim %arg220, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %470 = stablehlo.multiply %468, %469 : tensor<1x200x20x20xf32>
    %471 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %472 = stablehlo.broadcast_in_dim %arg221, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %473 = stablehlo.multiply %471, %472 : tensor<1x200x20x20xf32>
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %475 = stablehlo.broadcast_in_dim %arg222, dims = [1, 2, 3] : (tensor<200x1x1xf32>) -> tensor<1x200x20x20xf32>
    %476 = stablehlo.add %474, %475 : tensor<1x200x20x20xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %478 = stablehlo.add %477, %454 : tensor<1x200x20x20xf32>
    %479 = stablehlo.maximum %478, %cst_15 : tensor<1x200x20x20xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %481 = stablehlo.minimum %480, %458 : tensor<1x200x20x20xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2, 3] : (tensor<1x200x20x20xf32>) -> tensor<1x200x20x20xf32>
    %483 = stablehlo.divide %482, %461 : tensor<1x200x20x20xf32>
    %484 = stablehlo.multiply %483, %476 : tensor<1x200x20x20xf32>
    %485 = stablehlo.convolution(%484, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x200x20x20xf32>, tensor<80x200x1x1xf32>) -> tensor<1x80x20x20xf32>
    %486 = stablehlo.broadcast_in_dim %485, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %487 = stablehlo.broadcast_in_dim %arg223, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %488 = stablehlo.subtract %486, %487 : tensor<1x80x20x20xf32>
    %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %490 = stablehlo.broadcast_in_dim %arg224, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %491 = stablehlo.multiply %489, %490 : tensor<1x80x20x20xf32>
    %492 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %493 = stablehlo.broadcast_in_dim %arg225, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %494 = stablehlo.multiply %492, %493 : tensor<1x80x20x20xf32>
    %495 = stablehlo.broadcast_in_dim %494, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %496 = stablehlo.broadcast_in_dim %arg226, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %497 = stablehlo.add %495, %496 : tensor<1x80x20x20xf32>
    %498 = stablehlo.add %497, %439 : tensor<1x80x20x20xf32>
    %499 = stablehlo.convolution(%498, %arg37) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x20x20xf32>, tensor<184x80x1x1xf32>) -> tensor<1x184x20x20xf32>
    %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %501 = stablehlo.broadcast_in_dim %arg227, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %502 = stablehlo.subtract %500, %501 : tensor<1x184x20x20xf32>
    %503 = stablehlo.broadcast_in_dim %502, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %504 = stablehlo.broadcast_in_dim %arg228, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %505 = stablehlo.multiply %503, %504 : tensor<1x184x20x20xf32>
    %506 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %507 = stablehlo.broadcast_in_dim %arg229, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %508 = stablehlo.multiply %506, %507 : tensor<1x184x20x20xf32>
    %509 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %510 = stablehlo.broadcast_in_dim %arg230, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %511 = stablehlo.add %509, %510 : tensor<1x184x20x20xf32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %513 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x184x20x20xf32>
    %514 = stablehlo.add %512, %513 : tensor<1x184x20x20xf32>
    %515 = stablehlo.maximum %514, %cst_16 : tensor<1x184x20x20xf32>
    %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %517 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x184x20x20xf32>
    %518 = stablehlo.minimum %516, %517 : tensor<1x184x20x20xf32>
    %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %520 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x184x20x20xf32>
    %521 = stablehlo.divide %519, %520 : tensor<1x184x20x20xf32>
    %522 = stablehlo.multiply %521, %511 : tensor<1x184x20x20xf32>
    %523 = stablehlo.convolution(%522, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 184 : i64} : (tensor<1x184x20x20xf32>, tensor<184x1x3x3xf32>) -> tensor<1x184x20x20xf32>
    %524 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %525 = stablehlo.broadcast_in_dim %arg231, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %526 = stablehlo.subtract %524, %525 : tensor<1x184x20x20xf32>
    %527 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %528 = stablehlo.broadcast_in_dim %arg232, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %529 = stablehlo.multiply %527, %528 : tensor<1x184x20x20xf32>
    %530 = stablehlo.broadcast_in_dim %529, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %531 = stablehlo.broadcast_in_dim %arg233, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %532 = stablehlo.multiply %530, %531 : tensor<1x184x20x20xf32>
    %533 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %534 = stablehlo.broadcast_in_dim %arg234, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %535 = stablehlo.add %533, %534 : tensor<1x184x20x20xf32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %537 = stablehlo.add %536, %513 : tensor<1x184x20x20xf32>
    %538 = stablehlo.maximum %537, %cst_16 : tensor<1x184x20x20xf32>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %540 = stablehlo.minimum %539, %517 : tensor<1x184x20x20xf32>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %542 = stablehlo.divide %541, %520 : tensor<1x184x20x20xf32>
    %543 = stablehlo.multiply %542, %535 : tensor<1x184x20x20xf32>
    %544 = stablehlo.convolution(%543, %arg39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x184x20x20xf32>, tensor<80x184x1x1xf32>) -> tensor<1x80x20x20xf32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %546 = stablehlo.broadcast_in_dim %arg235, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %547 = stablehlo.subtract %545, %546 : tensor<1x80x20x20xf32>
    %548 = stablehlo.broadcast_in_dim %547, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %549 = stablehlo.broadcast_in_dim %arg236, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %550 = stablehlo.multiply %548, %549 : tensor<1x80x20x20xf32>
    %551 = stablehlo.broadcast_in_dim %550, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %552 = stablehlo.broadcast_in_dim %arg237, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %553 = stablehlo.multiply %551, %552 : tensor<1x80x20x20xf32>
    %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %555 = stablehlo.broadcast_in_dim %arg238, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %556 = stablehlo.add %554, %555 : tensor<1x80x20x20xf32>
    %557 = stablehlo.add %556, %498 : tensor<1x80x20x20xf32>
    %558 = stablehlo.convolution(%557, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x20x20xf32>, tensor<184x80x1x1xf32>) -> tensor<1x184x20x20xf32>
    %559 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %560 = stablehlo.broadcast_in_dim %arg239, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %561 = stablehlo.subtract %559, %560 : tensor<1x184x20x20xf32>
    %562 = stablehlo.broadcast_in_dim %561, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %563 = stablehlo.broadcast_in_dim %arg240, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %564 = stablehlo.multiply %562, %563 : tensor<1x184x20x20xf32>
    %565 = stablehlo.broadcast_in_dim %564, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %566 = stablehlo.broadcast_in_dim %arg241, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %567 = stablehlo.multiply %565, %566 : tensor<1x184x20x20xf32>
    %568 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %569 = stablehlo.broadcast_in_dim %arg242, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %570 = stablehlo.add %568, %569 : tensor<1x184x20x20xf32>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %572 = stablehlo.add %571, %513 : tensor<1x184x20x20xf32>
    %573 = stablehlo.maximum %572, %cst_16 : tensor<1x184x20x20xf32>
    %574 = stablehlo.broadcast_in_dim %573, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %575 = stablehlo.minimum %574, %517 : tensor<1x184x20x20xf32>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %577 = stablehlo.divide %576, %520 : tensor<1x184x20x20xf32>
    %578 = stablehlo.multiply %577, %570 : tensor<1x184x20x20xf32>
    %579 = stablehlo.convolution(%578, %arg41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 184 : i64} : (tensor<1x184x20x20xf32>, tensor<184x1x3x3xf32>) -> tensor<1x184x20x20xf32>
    %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %581 = stablehlo.broadcast_in_dim %arg243, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %582 = stablehlo.subtract %580, %581 : tensor<1x184x20x20xf32>
    %583 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %584 = stablehlo.broadcast_in_dim %arg244, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %585 = stablehlo.multiply %583, %584 : tensor<1x184x20x20xf32>
    %586 = stablehlo.broadcast_in_dim %585, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %587 = stablehlo.broadcast_in_dim %arg245, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %588 = stablehlo.multiply %586, %587 : tensor<1x184x20x20xf32>
    %589 = stablehlo.broadcast_in_dim %588, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %590 = stablehlo.broadcast_in_dim %arg246, dims = [1, 2, 3] : (tensor<184x1x1xf32>) -> tensor<1x184x20x20xf32>
    %591 = stablehlo.add %589, %590 : tensor<1x184x20x20xf32>
    %592 = stablehlo.broadcast_in_dim %591, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %593 = stablehlo.add %592, %513 : tensor<1x184x20x20xf32>
    %594 = stablehlo.maximum %593, %cst_16 : tensor<1x184x20x20xf32>
    %595 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %596 = stablehlo.minimum %595, %517 : tensor<1x184x20x20xf32>
    %597 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2, 3] : (tensor<1x184x20x20xf32>) -> tensor<1x184x20x20xf32>
    %598 = stablehlo.divide %597, %520 : tensor<1x184x20x20xf32>
    %599 = stablehlo.multiply %598, %591 : tensor<1x184x20x20xf32>
    %600 = stablehlo.convolution(%599, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x184x20x20xf32>, tensor<80x184x1x1xf32>) -> tensor<1x80x20x20xf32>
    %601 = stablehlo.broadcast_in_dim %600, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %602 = stablehlo.broadcast_in_dim %arg247, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %603 = stablehlo.subtract %601, %602 : tensor<1x80x20x20xf32>
    %604 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %605 = stablehlo.broadcast_in_dim %arg248, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %606 = stablehlo.multiply %604, %605 : tensor<1x80x20x20xf32>
    %607 = stablehlo.broadcast_in_dim %606, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %608 = stablehlo.broadcast_in_dim %arg249, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %609 = stablehlo.multiply %607, %608 : tensor<1x80x20x20xf32>
    %610 = stablehlo.broadcast_in_dim %609, dims = [0, 1, 2, 3] : (tensor<1x80x20x20xf32>) -> tensor<1x80x20x20xf32>
    %611 = stablehlo.broadcast_in_dim %arg250, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x20x20xf32>
    %612 = stablehlo.add %610, %611 : tensor<1x80x20x20xf32>
    %613 = stablehlo.add %612, %557 : tensor<1x80x20x20xf32>
    %614 = stablehlo.convolution(%613, %arg43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x20x20xf32>, tensor<480x80x1x1xf32>) -> tensor<1x480x20x20xf32>
    %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %616 = stablehlo.broadcast_in_dim %arg251, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %617 = stablehlo.subtract %615, %616 : tensor<1x480x20x20xf32>
    %618 = stablehlo.broadcast_in_dim %617, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %619 = stablehlo.broadcast_in_dim %arg252, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %620 = stablehlo.multiply %618, %619 : tensor<1x480x20x20xf32>
    %621 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %622 = stablehlo.broadcast_in_dim %arg253, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %623 = stablehlo.multiply %621, %622 : tensor<1x480x20x20xf32>
    %624 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %625 = stablehlo.broadcast_in_dim %arg254, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %626 = stablehlo.add %624, %625 : tensor<1x480x20x20xf32>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %628 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x480x20x20xf32>
    %629 = stablehlo.add %627, %628 : tensor<1x480x20x20xf32>
    %630 = stablehlo.maximum %629, %cst_17 : tensor<1x480x20x20xf32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %632 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x480x20x20xf32>
    %633 = stablehlo.minimum %631, %632 : tensor<1x480x20x20xf32>
    %634 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %635 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x480x20x20xf32>
    %636 = stablehlo.divide %634, %635 : tensor<1x480x20x20xf32>
    %637 = stablehlo.multiply %636, %626 : tensor<1x480x20x20xf32>
    %638 = stablehlo.convolution(%637, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<1x480x20x20xf32>, tensor<480x1x3x3xf32>) -> tensor<1x480x20x20xf32>
    %639 = stablehlo.broadcast_in_dim %638, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %640 = stablehlo.broadcast_in_dim %arg255, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %641 = stablehlo.subtract %639, %640 : tensor<1x480x20x20xf32>
    %642 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %643 = stablehlo.broadcast_in_dim %arg256, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %644 = stablehlo.multiply %642, %643 : tensor<1x480x20x20xf32>
    %645 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %646 = stablehlo.broadcast_in_dim %arg257, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %647 = stablehlo.multiply %645, %646 : tensor<1x480x20x20xf32>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %649 = stablehlo.broadcast_in_dim %arg258, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %650 = stablehlo.add %648, %649 : tensor<1x480x20x20xf32>
    %651 = stablehlo.broadcast_in_dim %650, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %652 = stablehlo.add %651, %628 : tensor<1x480x20x20xf32>
    %653 = stablehlo.maximum %652, %cst_17 : tensor<1x480x20x20xf32>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %655 = stablehlo.minimum %654, %632 : tensor<1x480x20x20xf32>
    %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %657 = stablehlo.divide %656, %635 : tensor<1x480x20x20xf32>
    %658 = stablehlo.multiply %657, %650 : tensor<1x480x20x20xf32>
    %659 = stablehlo.reduce(%658 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x480x20x20xf32>, tensor<f32>) -> tensor<1x480xf32>
    %660 = stablehlo.reshape %659 : (tensor<1x480xf32>) -> tensor<1x480x1x1xf32>
    %661 = stablehlo.convert %cst_26 : (tensor<1xi64>) -> tensor<1xf32>
    %662 = stablehlo.reshape %661 : (tensor<1xf32>) -> tensor<f32>
    %663 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %664 = stablehlo.broadcast_in_dim %662, dims = [] : (tensor<f32>) -> tensor<1x480x1x1xf32>
    %665 = stablehlo.divide %663, %664 : tensor<1x480x1x1xf32>
    %666 = stablehlo.convolution(%665, %arg45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x1x1xf32>, tensor<120x480x1x1xf32>) -> tensor<1x120x1x1xf32>
    %667 = stablehlo.reshape %arg46 : (tensor<120xf32>) -> tensor<120x1x1xf32>
    %668 = stablehlo.broadcast_in_dim %666, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %669 = stablehlo.broadcast_in_dim %667, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %670 = stablehlo.add %668, %669 : tensor<1x120x1x1xf32>
    %671 = stablehlo.maximum %670, %cst_18 : tensor<1x120x1x1xf32>
    %672 = stablehlo.convolution(%671, %arg47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x1x1xf32>, tensor<480x120x1x1xf32>) -> tensor<1x480x1x1xf32>
    %673 = stablehlo.reshape %arg48 : (tensor<480xf32>) -> tensor<480x1x1xf32>
    %674 = stablehlo.broadcast_in_dim %672, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %675 = stablehlo.broadcast_in_dim %673, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %676 = stablehlo.add %674, %675 : tensor<1x480x1x1xf32>
    %677 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %678 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x480x1x1xf32>
    %679 = stablehlo.add %677, %678 : tensor<1x480x1x1xf32>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %681 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x480x1x1xf32>
    %682 = stablehlo.divide %680, %681 : tensor<1x480x1x1xf32>
    %683 = stablehlo.broadcast_in_dim %215, dims = [] : (tensor<f32>) -> tensor<1x480x1x1xf32>
    %684 = stablehlo.broadcast_in_dim %682, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %685 = stablehlo.minimum %683, %684 : tensor<1x480x1x1xf32>
    %686 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1x480x1x1xf32>
    %687 = stablehlo.broadcast_in_dim %685, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %688 = stablehlo.maximum %686, %687 : tensor<1x480x1x1xf32>
    %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x20x20xf32>
    %690 = stablehlo.broadcast_in_dim %658, dims = [0, 1, 2, 3] : (tensor<1x480x20x20xf32>) -> tensor<1x480x20x20xf32>
    %691 = stablehlo.multiply %689, %690 : tensor<1x480x20x20xf32>
    %692 = stablehlo.convolution(%691, %arg49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x20x20xf32>, tensor<112x480x1x1xf32>) -> tensor<1x112x20x20xf32>
    %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %694 = stablehlo.broadcast_in_dim %arg259, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %695 = stablehlo.subtract %693, %694 : tensor<1x112x20x20xf32>
    %696 = stablehlo.broadcast_in_dim %695, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %697 = stablehlo.broadcast_in_dim %arg260, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %698 = stablehlo.multiply %696, %697 : tensor<1x112x20x20xf32>
    %699 = stablehlo.broadcast_in_dim %698, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %700 = stablehlo.broadcast_in_dim %arg261, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %701 = stablehlo.multiply %699, %700 : tensor<1x112x20x20xf32>
    %702 = stablehlo.broadcast_in_dim %701, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %703 = stablehlo.broadcast_in_dim %arg262, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %704 = stablehlo.add %702, %703 : tensor<1x112x20x20xf32>
    %705 = stablehlo.convolution(%704, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x112x20x20xf32>, tensor<672x112x1x1xf32>) -> tensor<1x672x20x20xf32>
    %706 = stablehlo.broadcast_in_dim %705, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %707 = stablehlo.broadcast_in_dim %arg263, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %708 = stablehlo.subtract %706, %707 : tensor<1x672x20x20xf32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %710 = stablehlo.broadcast_in_dim %arg264, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %711 = stablehlo.multiply %709, %710 : tensor<1x672x20x20xf32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %713 = stablehlo.broadcast_in_dim %arg265, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %714 = stablehlo.multiply %712, %713 : tensor<1x672x20x20xf32>
    %715 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %716 = stablehlo.broadcast_in_dim %arg266, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %717 = stablehlo.add %715, %716 : tensor<1x672x20x20xf32>
    %718 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %719 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x672x20x20xf32>
    %720 = stablehlo.add %718, %719 : tensor<1x672x20x20xf32>
    %721 = stablehlo.maximum %720, %cst_19 : tensor<1x672x20x20xf32>
    %722 = stablehlo.broadcast_in_dim %721, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %723 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x672x20x20xf32>
    %724 = stablehlo.minimum %722, %723 : tensor<1x672x20x20xf32>
    %725 = stablehlo.broadcast_in_dim %724, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %726 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x672x20x20xf32>
    %727 = stablehlo.divide %725, %726 : tensor<1x672x20x20xf32>
    %728 = stablehlo.multiply %727, %717 : tensor<1x672x20x20xf32>
    %729 = stablehlo.convolution(%728, %arg51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<1x672x20x20xf32>, tensor<672x1x3x3xf32>) -> tensor<1x672x20x20xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %731 = stablehlo.broadcast_in_dim %arg267, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %732 = stablehlo.subtract %730, %731 : tensor<1x672x20x20xf32>
    %733 = stablehlo.broadcast_in_dim %732, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %734 = stablehlo.broadcast_in_dim %arg268, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %735 = stablehlo.multiply %733, %734 : tensor<1x672x20x20xf32>
    %736 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %737 = stablehlo.broadcast_in_dim %arg269, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %738 = stablehlo.multiply %736, %737 : tensor<1x672x20x20xf32>
    %739 = stablehlo.broadcast_in_dim %738, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %740 = stablehlo.broadcast_in_dim %arg270, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %741 = stablehlo.add %739, %740 : tensor<1x672x20x20xf32>
    %742 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %743 = stablehlo.add %742, %719 : tensor<1x672x20x20xf32>
    %744 = stablehlo.maximum %743, %cst_19 : tensor<1x672x20x20xf32>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %746 = stablehlo.minimum %745, %723 : tensor<1x672x20x20xf32>
    %747 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %748 = stablehlo.divide %747, %726 : tensor<1x672x20x20xf32>
    %749 = stablehlo.multiply %748, %741 : tensor<1x672x20x20xf32>
    %750 = stablehlo.reduce(%749 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x672x20x20xf32>, tensor<f32>) -> tensor<1x672xf32>
    %751 = stablehlo.reshape %750 : (tensor<1x672xf32>) -> tensor<1x672x1x1xf32>
    %752 = stablehlo.broadcast_in_dim %751, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %753 = stablehlo.broadcast_in_dim %662, dims = [] : (tensor<f32>) -> tensor<1x672x1x1xf32>
    %754 = stablehlo.divide %752, %753 : tensor<1x672x1x1xf32>
    %755 = stablehlo.convolution(%754, %arg52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x672x1x1xf32>, tensor<168x672x1x1xf32>) -> tensor<1x168x1x1xf32>
    %756 = stablehlo.reshape %arg53 : (tensor<168xf32>) -> tensor<168x1x1xf32>
    %757 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 2, 3] : (tensor<1x168x1x1xf32>) -> tensor<1x168x1x1xf32>
    %758 = stablehlo.broadcast_in_dim %756, dims = [1, 2, 3] : (tensor<168x1x1xf32>) -> tensor<1x168x1x1xf32>
    %759 = stablehlo.add %757, %758 : tensor<1x168x1x1xf32>
    %760 = stablehlo.maximum %759, %cst_20 : tensor<1x168x1x1xf32>
    %761 = stablehlo.convolution(%760, %arg54) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x168x1x1xf32>, tensor<672x168x1x1xf32>) -> tensor<1x672x1x1xf32>
    %762 = stablehlo.reshape %arg55 : (tensor<672xf32>) -> tensor<672x1x1xf32>
    %763 = stablehlo.broadcast_in_dim %761, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %764 = stablehlo.broadcast_in_dim %762, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %765 = stablehlo.add %763, %764 : tensor<1x672x1x1xf32>
    %766 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %767 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x672x1x1xf32>
    %768 = stablehlo.add %766, %767 : tensor<1x672x1x1xf32>
    %769 = stablehlo.broadcast_in_dim %768, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %770 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x672x1x1xf32>
    %771 = stablehlo.divide %769, %770 : tensor<1x672x1x1xf32>
    %772 = stablehlo.broadcast_in_dim %215, dims = [] : (tensor<f32>) -> tensor<1x672x1x1xf32>
    %773 = stablehlo.broadcast_in_dim %771, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %774 = stablehlo.minimum %772, %773 : tensor<1x672x1x1xf32>
    %775 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1x672x1x1xf32>
    %776 = stablehlo.broadcast_in_dim %774, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %777 = stablehlo.maximum %775, %776 : tensor<1x672x1x1xf32>
    %778 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %779 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %780 = stablehlo.multiply %778, %779 : tensor<1x672x20x20xf32>
    %781 = stablehlo.convolution(%780, %arg56) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x672x20x20xf32>, tensor<112x672x1x1xf32>) -> tensor<1x112x20x20xf32>
    %782 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %783 = stablehlo.broadcast_in_dim %arg271, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %784 = stablehlo.subtract %782, %783 : tensor<1x112x20x20xf32>
    %785 = stablehlo.broadcast_in_dim %784, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %786 = stablehlo.broadcast_in_dim %arg272, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %787 = stablehlo.multiply %785, %786 : tensor<1x112x20x20xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %789 = stablehlo.broadcast_in_dim %arg273, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %790 = stablehlo.multiply %788, %789 : tensor<1x112x20x20xf32>
    %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2, 3] : (tensor<1x112x20x20xf32>) -> tensor<1x112x20x20xf32>
    %792 = stablehlo.broadcast_in_dim %arg274, dims = [1, 2, 3] : (tensor<112x1x1xf32>) -> tensor<1x112x20x20xf32>
    %793 = stablehlo.add %791, %792 : tensor<1x112x20x20xf32>
    %794 = stablehlo.add %793, %704 : tensor<1x112x20x20xf32>
    %795 = stablehlo.convolution(%794, %arg57) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x112x20x20xf32>, tensor<672x112x1x1xf32>) -> tensor<1x672x20x20xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %797 = stablehlo.broadcast_in_dim %arg275, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %798 = stablehlo.subtract %796, %797 : tensor<1x672x20x20xf32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %800 = stablehlo.broadcast_in_dim %arg276, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %801 = stablehlo.multiply %799, %800 : tensor<1x672x20x20xf32>
    %802 = stablehlo.broadcast_in_dim %801, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %803 = stablehlo.broadcast_in_dim %arg277, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %804 = stablehlo.multiply %802, %803 : tensor<1x672x20x20xf32>
    %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %806 = stablehlo.broadcast_in_dim %arg278, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %807 = stablehlo.add %805, %806 : tensor<1x672x20x20xf32>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %809 = stablehlo.add %808, %719 : tensor<1x672x20x20xf32>
    %810 = stablehlo.maximum %809, %cst_19 : tensor<1x672x20x20xf32>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %812 = stablehlo.minimum %811, %723 : tensor<1x672x20x20xf32>
    %813 = stablehlo.broadcast_in_dim %812, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %814 = stablehlo.divide %813, %726 : tensor<1x672x20x20xf32>
    %815 = stablehlo.multiply %814, %807 : tensor<1x672x20x20xf32>
    %816 = stablehlo.convolution(%815, %arg58) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<1x672x20x20xf32>, tensor<672x1x5x5xf32>) -> tensor<1x672x10x10xf32>
    %817 = stablehlo.broadcast_in_dim %816, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %818 = stablehlo.broadcast_in_dim %arg279, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x10x10xf32>
    %819 = stablehlo.subtract %817, %818 : tensor<1x672x10x10xf32>
    %820 = stablehlo.broadcast_in_dim %819, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %821 = stablehlo.broadcast_in_dim %arg280, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x10x10xf32>
    %822 = stablehlo.multiply %820, %821 : tensor<1x672x10x10xf32>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %824 = stablehlo.broadcast_in_dim %arg281, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x10x10xf32>
    %825 = stablehlo.multiply %823, %824 : tensor<1x672x10x10xf32>
    %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %827 = stablehlo.broadcast_in_dim %arg282, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x10x10xf32>
    %828 = stablehlo.add %826, %827 : tensor<1x672x10x10xf32>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %830 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x672x10x10xf32>
    %831 = stablehlo.add %829, %830 : tensor<1x672x10x10xf32>
    %832 = stablehlo.maximum %831, %cst_21 : tensor<1x672x10x10xf32>
    %833 = stablehlo.broadcast_in_dim %832, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %834 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x672x10x10xf32>
    %835 = stablehlo.minimum %833, %834 : tensor<1x672x10x10xf32>
    %836 = stablehlo.broadcast_in_dim %835, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %837 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x672x10x10xf32>
    %838 = stablehlo.divide %836, %837 : tensor<1x672x10x10xf32>
    %839 = stablehlo.multiply %838, %828 : tensor<1x672x10x10xf32>
    %840 = stablehlo.reduce(%839 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x672x10x10xf32>, tensor<f32>) -> tensor<1x672xf32>
    %841 = stablehlo.reshape %840 : (tensor<1x672xf32>) -> tensor<1x672x1x1xf32>
    %842 = stablehlo.convert %cst_27 : (tensor<1xi64>) -> tensor<1xf32>
    %843 = stablehlo.reshape %842 : (tensor<1xf32>) -> tensor<f32>
    %844 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %845 = stablehlo.broadcast_in_dim %843, dims = [] : (tensor<f32>) -> tensor<1x672x1x1xf32>
    %846 = stablehlo.divide %844, %845 : tensor<1x672x1x1xf32>
    %847 = stablehlo.convolution(%846, %arg59) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x672x1x1xf32>, tensor<168x672x1x1xf32>) -> tensor<1x168x1x1xf32>
    %848 = stablehlo.reshape %arg60 : (tensor<168xf32>) -> tensor<168x1x1xf32>
    %849 = stablehlo.broadcast_in_dim %847, dims = [0, 1, 2, 3] : (tensor<1x168x1x1xf32>) -> tensor<1x168x1x1xf32>
    %850 = stablehlo.broadcast_in_dim %848, dims = [1, 2, 3] : (tensor<168x1x1xf32>) -> tensor<1x168x1x1xf32>
    %851 = stablehlo.add %849, %850 : tensor<1x168x1x1xf32>
    %852 = stablehlo.maximum %851, %cst_20 : tensor<1x168x1x1xf32>
    %853 = stablehlo.convolution(%852, %arg61) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x168x1x1xf32>, tensor<672x168x1x1xf32>) -> tensor<1x672x1x1xf32>
    %854 = stablehlo.reshape %arg62 : (tensor<672xf32>) -> tensor<672x1x1xf32>
    %855 = stablehlo.broadcast_in_dim %853, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %856 = stablehlo.broadcast_in_dim %854, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %857 = stablehlo.add %855, %856 : tensor<1x672x1x1xf32>
    %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %859 = stablehlo.add %858, %767 : tensor<1x672x1x1xf32>
    %860 = stablehlo.broadcast_in_dim %859, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %861 = stablehlo.divide %860, %770 : tensor<1x672x1x1xf32>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %863 = stablehlo.minimum %772, %862 : tensor<1x672x1x1xf32>
    %864 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x1x1xf32>
    %865 = stablehlo.maximum %775, %864 : tensor<1x672x1x1xf32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<1x672x1x1xf32>) -> tensor<1x672x10x10xf32>
    %867 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2, 3] : (tensor<1x672x10x10xf32>) -> tensor<1x672x10x10xf32>
    %868 = stablehlo.multiply %866, %867 : tensor<1x672x10x10xf32>
    %869 = stablehlo.convolution(%868, %arg63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x672x10x10xf32>, tensor<80x672x1x1xf32>) -> tensor<1x80x10x10xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %871 = stablehlo.broadcast_in_dim %arg283, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %872 = stablehlo.subtract %870, %871 : tensor<1x80x10x10xf32>
    %873 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %874 = stablehlo.broadcast_in_dim %arg284, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %875 = stablehlo.multiply %873, %874 : tensor<1x80x10x10xf32>
    %876 = stablehlo.broadcast_in_dim %875, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %877 = stablehlo.broadcast_in_dim %arg285, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %878 = stablehlo.multiply %876, %877 : tensor<1x80x10x10xf32>
    %879 = stablehlo.broadcast_in_dim %878, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %880 = stablehlo.broadcast_in_dim %arg286, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %881 = stablehlo.add %879, %880 : tensor<1x80x10x10xf32>
    %882 = stablehlo.convolution(%881, %arg64) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x10x10xf32>, tensor<480x80x1x1xf32>) -> tensor<1x480x10x10xf32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %884 = stablehlo.broadcast_in_dim %arg287, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %885 = stablehlo.subtract %883, %884 : tensor<1x480x10x10xf32>
    %886 = stablehlo.broadcast_in_dim %885, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %887 = stablehlo.broadcast_in_dim %arg288, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %888 = stablehlo.multiply %886, %887 : tensor<1x480x10x10xf32>
    %889 = stablehlo.broadcast_in_dim %888, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %890 = stablehlo.broadcast_in_dim %arg289, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %891 = stablehlo.multiply %889, %890 : tensor<1x480x10x10xf32>
    %892 = stablehlo.broadcast_in_dim %891, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %893 = stablehlo.broadcast_in_dim %arg290, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %894 = stablehlo.add %892, %893 : tensor<1x480x10x10xf32>
    %895 = stablehlo.broadcast_in_dim %894, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %896 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x480x10x10xf32>
    %897 = stablehlo.add %895, %896 : tensor<1x480x10x10xf32>
    %898 = stablehlo.maximum %897, %cst_22 : tensor<1x480x10x10xf32>
    %899 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %900 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x480x10x10xf32>
    %901 = stablehlo.minimum %899, %900 : tensor<1x480x10x10xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %903 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<1x480x10x10xf32>
    %904 = stablehlo.divide %902, %903 : tensor<1x480x10x10xf32>
    %905 = stablehlo.multiply %904, %894 : tensor<1x480x10x10xf32>
    %906 = stablehlo.convolution(%905, %arg65) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<1x480x10x10xf32>, tensor<480x1x5x5xf32>) -> tensor<1x480x10x10xf32>
    %907 = stablehlo.broadcast_in_dim %906, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %908 = stablehlo.broadcast_in_dim %arg291, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %909 = stablehlo.subtract %907, %908 : tensor<1x480x10x10xf32>
    %910 = stablehlo.broadcast_in_dim %909, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %911 = stablehlo.broadcast_in_dim %arg292, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %912 = stablehlo.multiply %910, %911 : tensor<1x480x10x10xf32>
    %913 = stablehlo.broadcast_in_dim %912, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %914 = stablehlo.broadcast_in_dim %arg293, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %915 = stablehlo.multiply %913, %914 : tensor<1x480x10x10xf32>
    %916 = stablehlo.broadcast_in_dim %915, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %917 = stablehlo.broadcast_in_dim %arg294, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %918 = stablehlo.add %916, %917 : tensor<1x480x10x10xf32>
    %919 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %920 = stablehlo.add %919, %896 : tensor<1x480x10x10xf32>
    %921 = stablehlo.maximum %920, %cst_22 : tensor<1x480x10x10xf32>
    %922 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %923 = stablehlo.minimum %922, %900 : tensor<1x480x10x10xf32>
    %924 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %925 = stablehlo.divide %924, %903 : tensor<1x480x10x10xf32>
    %926 = stablehlo.multiply %925, %918 : tensor<1x480x10x10xf32>
    %927 = stablehlo.reduce(%926 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x480x10x10xf32>, tensor<f32>) -> tensor<1x480xf32>
    %928 = stablehlo.reshape %927 : (tensor<1x480xf32>) -> tensor<1x480x1x1xf32>
    %929 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %930 = stablehlo.broadcast_in_dim %843, dims = [] : (tensor<f32>) -> tensor<1x480x1x1xf32>
    %931 = stablehlo.divide %929, %930 : tensor<1x480x1x1xf32>
    %932 = stablehlo.convolution(%931, %arg66) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x1x1xf32>, tensor<120x480x1x1xf32>) -> tensor<1x120x1x1xf32>
    %933 = stablehlo.reshape %arg67 : (tensor<120xf32>) -> tensor<120x1x1xf32>
    %934 = stablehlo.broadcast_in_dim %932, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %935 = stablehlo.broadcast_in_dim %933, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %936 = stablehlo.add %934, %935 : tensor<1x120x1x1xf32>
    %937 = stablehlo.maximum %936, %cst_18 : tensor<1x120x1x1xf32>
    %938 = stablehlo.convolution(%937, %arg68) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x1x1xf32>, tensor<480x120x1x1xf32>) -> tensor<1x480x1x1xf32>
    %939 = stablehlo.reshape %arg69 : (tensor<480xf32>) -> tensor<480x1x1xf32>
    %940 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %941 = stablehlo.broadcast_in_dim %939, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %942 = stablehlo.add %940, %941 : tensor<1x480x1x1xf32>
    %943 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %944 = stablehlo.add %943, %678 : tensor<1x480x1x1xf32>
    %945 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %946 = stablehlo.divide %945, %681 : tensor<1x480x1x1xf32>
    %947 = stablehlo.broadcast_in_dim %946, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %948 = stablehlo.minimum %683, %947 : tensor<1x480x1x1xf32>
    %949 = stablehlo.broadcast_in_dim %948, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %950 = stablehlo.maximum %686, %949 : tensor<1x480x1x1xf32>
    %951 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %952 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %953 = stablehlo.multiply %951, %952 : tensor<1x480x10x10xf32>
    %954 = stablehlo.convolution(%953, %arg70) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x10x10xf32>, tensor<80x480x1x1xf32>) -> tensor<1x80x10x10xf32>
    %955 = stablehlo.broadcast_in_dim %954, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %956 = stablehlo.broadcast_in_dim %arg295, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %957 = stablehlo.subtract %955, %956 : tensor<1x80x10x10xf32>
    %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %959 = stablehlo.broadcast_in_dim %arg296, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %960 = stablehlo.multiply %958, %959 : tensor<1x80x10x10xf32>
    %961 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %962 = stablehlo.broadcast_in_dim %arg297, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %963 = stablehlo.multiply %961, %962 : tensor<1x80x10x10xf32>
    %964 = stablehlo.broadcast_in_dim %963, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %965 = stablehlo.broadcast_in_dim %arg298, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %966 = stablehlo.add %964, %965 : tensor<1x80x10x10xf32>
    %967 = stablehlo.add %966, %881 : tensor<1x80x10x10xf32>
    %968 = stablehlo.convolution(%967, %arg71) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x10x10xf32>, tensor<480x80x1x1xf32>) -> tensor<1x480x10x10xf32>
    %969 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %970 = stablehlo.broadcast_in_dim %arg299, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %971 = stablehlo.subtract %969, %970 : tensor<1x480x10x10xf32>
    %972 = stablehlo.broadcast_in_dim %971, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %973 = stablehlo.broadcast_in_dim %arg300, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %974 = stablehlo.multiply %972, %973 : tensor<1x480x10x10xf32>
    %975 = stablehlo.broadcast_in_dim %974, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %976 = stablehlo.broadcast_in_dim %arg301, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %977 = stablehlo.multiply %975, %976 : tensor<1x480x10x10xf32>
    %978 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %979 = stablehlo.broadcast_in_dim %arg302, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %980 = stablehlo.add %978, %979 : tensor<1x480x10x10xf32>
    %981 = stablehlo.broadcast_in_dim %980, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %982 = stablehlo.add %981, %896 : tensor<1x480x10x10xf32>
    %983 = stablehlo.maximum %982, %cst_22 : tensor<1x480x10x10xf32>
    %984 = stablehlo.broadcast_in_dim %983, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %985 = stablehlo.minimum %984, %900 : tensor<1x480x10x10xf32>
    %986 = stablehlo.broadcast_in_dim %985, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %987 = stablehlo.divide %986, %903 : tensor<1x480x10x10xf32>
    %988 = stablehlo.multiply %987, %980 : tensor<1x480x10x10xf32>
    %989 = stablehlo.convolution(%988, %arg72) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<1x480x10x10xf32>, tensor<480x1x5x5xf32>) -> tensor<1x480x10x10xf32>
    %990 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %991 = stablehlo.broadcast_in_dim %arg303, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %992 = stablehlo.subtract %990, %991 : tensor<1x480x10x10xf32>
    %993 = stablehlo.broadcast_in_dim %992, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %994 = stablehlo.broadcast_in_dim %arg304, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %995 = stablehlo.multiply %993, %994 : tensor<1x480x10x10xf32>
    %996 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %997 = stablehlo.broadcast_in_dim %arg305, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %998 = stablehlo.multiply %996, %997 : tensor<1x480x10x10xf32>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1000 = stablehlo.broadcast_in_dim %arg306, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1001 = stablehlo.add %999, %1000 : tensor<1x480x10x10xf32>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1003 = stablehlo.add %1002, %896 : tensor<1x480x10x10xf32>
    %1004 = stablehlo.maximum %1003, %cst_22 : tensor<1x480x10x10xf32>
    %1005 = stablehlo.broadcast_in_dim %1004, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1006 = stablehlo.minimum %1005, %900 : tensor<1x480x10x10xf32>
    %1007 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1008 = stablehlo.divide %1007, %903 : tensor<1x480x10x10xf32>
    %1009 = stablehlo.multiply %1008, %1001 : tensor<1x480x10x10xf32>
    %1010 = stablehlo.reduce(%1009 init: %cst_9) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x480x10x10xf32>, tensor<f32>) -> tensor<1x480xf32>
    %1011 = stablehlo.reshape %1010 : (tensor<1x480xf32>) -> tensor<1x480x1x1xf32>
    %1012 = stablehlo.broadcast_in_dim %1011, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1013 = stablehlo.divide %1012, %930 : tensor<1x480x1x1xf32>
    %1014 = stablehlo.convolution(%1013, %arg73) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x1x1xf32>, tensor<120x480x1x1xf32>) -> tensor<1x120x1x1xf32>
    %1015 = stablehlo.reshape %arg74 : (tensor<120xf32>) -> tensor<120x1x1xf32>
    %1016 = stablehlo.broadcast_in_dim %1014, dims = [0, 1, 2, 3] : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %1017 = stablehlo.broadcast_in_dim %1015, dims = [1, 2, 3] : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %1018 = stablehlo.add %1016, %1017 : tensor<1x120x1x1xf32>
    %1019 = stablehlo.maximum %1018, %cst_18 : tensor<1x120x1x1xf32>
    %1020 = stablehlo.convolution(%1019, %arg75) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x120x1x1xf32>, tensor<480x120x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1021 = stablehlo.reshape %arg76 : (tensor<480xf32>) -> tensor<480x1x1xf32>
    %1022 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1023 = stablehlo.broadcast_in_dim %1021, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1024 = stablehlo.add %1022, %1023 : tensor<1x480x1x1xf32>
    %1025 = stablehlo.broadcast_in_dim %1024, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1026 = stablehlo.add %1025, %678 : tensor<1x480x1x1xf32>
    %1027 = stablehlo.broadcast_in_dim %1026, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1028 = stablehlo.divide %1027, %681 : tensor<1x480x1x1xf32>
    %1029 = stablehlo.broadcast_in_dim %1028, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1030 = stablehlo.minimum %683, %1029 : tensor<1x480x1x1xf32>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x1x1xf32>
    %1032 = stablehlo.maximum %686, %1031 : tensor<1x480x1x1xf32>
    %1033 = stablehlo.broadcast_in_dim %1032, dims = [0, 1, 2, 3] : (tensor<1x480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1034 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1035 = stablehlo.multiply %1033, %1034 : tensor<1x480x10x10xf32>
    %1036 = stablehlo.convolution(%1035, %arg77) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x10x10xf32>, tensor<80x480x1x1xf32>) -> tensor<1x80x10x10xf32>
    %1037 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %1038 = stablehlo.broadcast_in_dim %arg307, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %1039 = stablehlo.subtract %1037, %1038 : tensor<1x80x10x10xf32>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %1041 = stablehlo.broadcast_in_dim %arg308, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %1042 = stablehlo.multiply %1040, %1041 : tensor<1x80x10x10xf32>
    %1043 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %1044 = stablehlo.broadcast_in_dim %arg309, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %1045 = stablehlo.multiply %1043, %1044 : tensor<1x80x10x10xf32>
    %1046 = stablehlo.broadcast_in_dim %1045, dims = [0, 1, 2, 3] : (tensor<1x80x10x10xf32>) -> tensor<1x80x10x10xf32>
    %1047 = stablehlo.broadcast_in_dim %arg310, dims = [1, 2, 3] : (tensor<80x1x1xf32>) -> tensor<1x80x10x10xf32>
    %1048 = stablehlo.add %1046, %1047 : tensor<1x80x10x10xf32>
    %1049 = stablehlo.add %1048, %967 : tensor<1x80x10x10xf32>
    %1050 = stablehlo.convolution(%1049, %arg78) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x80x10x10xf32>, tensor<480x80x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1051 = stablehlo.broadcast_in_dim %1050, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1052 = stablehlo.broadcast_in_dim %arg311, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1053 = stablehlo.subtract %1051, %1052 : tensor<1x480x10x10xf32>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1055 = stablehlo.broadcast_in_dim %arg312, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1056 = stablehlo.multiply %1054, %1055 : tensor<1x480x10x10xf32>
    %1057 = stablehlo.broadcast_in_dim %1056, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1058 = stablehlo.broadcast_in_dim %arg313, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1059 = stablehlo.multiply %1057, %1058 : tensor<1x480x10x10xf32>
    %1060 = stablehlo.broadcast_in_dim %1059, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1061 = stablehlo.broadcast_in_dim %arg314, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1062 = stablehlo.add %1060, %1061 : tensor<1x480x10x10xf32>
    %1063 = stablehlo.broadcast_in_dim %1062, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1064 = stablehlo.add %1063, %896 : tensor<1x480x10x10xf32>
    %1065 = stablehlo.maximum %1064, %cst_22 : tensor<1x480x10x10xf32>
    %1066 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1067 = stablehlo.minimum %1066, %900 : tensor<1x480x10x10xf32>
    %1068 = stablehlo.broadcast_in_dim %1067, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1069 = stablehlo.divide %1068, %903 : tensor<1x480x10x10xf32>
    %1070 = stablehlo.multiply %1069, %1062 : tensor<1x480x10x10xf32>
    %1071 = stablehlo.convolution(%1070, %arg79) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x10x10xf32>, tensor<256x480x1x1xf32>) -> tensor<1x256x10x10xf32>
    %1072 = stablehlo.broadcast_in_dim %1071, dims = [0, 1, 2, 3] : (tensor<1x256x10x10xf32>) -> tensor<1x256x10x10xf32>
    %1073 = stablehlo.broadcast_in_dim %arg315, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x10x10xf32>
    %1074 = stablehlo.subtract %1072, %1073 : tensor<1x256x10x10xf32>
    %1075 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2, 3] : (tensor<1x256x10x10xf32>) -> tensor<1x256x10x10xf32>
    %1076 = stablehlo.broadcast_in_dim %arg316, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x10x10xf32>
    %1077 = stablehlo.multiply %1075, %1076 : tensor<1x256x10x10xf32>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1, 2, 3] : (tensor<1x256x10x10xf32>) -> tensor<1x256x10x10xf32>
    %1079 = stablehlo.broadcast_in_dim %arg317, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x10x10xf32>
    %1080 = stablehlo.multiply %1078, %1079 : tensor<1x256x10x10xf32>
    %1081 = stablehlo.broadcast_in_dim %1080, dims = [0, 1, 2, 3] : (tensor<1x256x10x10xf32>) -> tensor<1x256x10x10xf32>
    %1082 = stablehlo.broadcast_in_dim %arg318, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x10x10xf32>
    %1083 = stablehlo.add %1081, %1082 : tensor<1x256x10x10xf32>
    %1084 = stablehlo.convert %cst_0 : (tensor<f64>) -> tensor<f32>
    %1085 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2, 3] : (tensor<1x256x10x10xf32>) -> tensor<1x256x10x10xf32>
    %1086 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x256x10x10xf32>
    %1087 = stablehlo.maximum %1085, %1086 : tensor<1x256x10x10xf32>
    %1088 = stablehlo.convert %cst : (tensor<f64>) -> tensor<f32>
    %1089 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x256x10x10xf32>
    %1090 = stablehlo.broadcast_in_dim %1087, dims = [0, 1, 2, 3] : (tensor<1x256x10x10xf32>) -> tensor<1x256x10x10xf32>
    %1091 = stablehlo.minimum %1089, %1090 : tensor<1x256x10x10xf32>
    %1092 = stablehlo.convolution(%1091, %arg80) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x10x10xf32>, tensor<256x1x3x3xf32>) -> tensor<1x256x5x5xf32>
    %1093 = stablehlo.broadcast_in_dim %1092, dims = [0, 1, 2, 3] : (tensor<1x256x5x5xf32>) -> tensor<1x256x5x5xf32>
    %1094 = stablehlo.broadcast_in_dim %arg319, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x5x5xf32>
    %1095 = stablehlo.subtract %1093, %1094 : tensor<1x256x5x5xf32>
    %1096 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2, 3] : (tensor<1x256x5x5xf32>) -> tensor<1x256x5x5xf32>
    %1097 = stablehlo.broadcast_in_dim %arg320, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x5x5xf32>
    %1098 = stablehlo.multiply %1096, %1097 : tensor<1x256x5x5xf32>
    %1099 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2, 3] : (tensor<1x256x5x5xf32>) -> tensor<1x256x5x5xf32>
    %1100 = stablehlo.broadcast_in_dim %arg321, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x5x5xf32>
    %1101 = stablehlo.multiply %1099, %1100 : tensor<1x256x5x5xf32>
    %1102 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2, 3] : (tensor<1x256x5x5xf32>) -> tensor<1x256x5x5xf32>
    %1103 = stablehlo.broadcast_in_dim %arg322, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x5x5xf32>
    %1104 = stablehlo.add %1102, %1103 : tensor<1x256x5x5xf32>
    %1105 = stablehlo.broadcast_in_dim %1104, dims = [0, 1, 2, 3] : (tensor<1x256x5x5xf32>) -> tensor<1x256x5x5xf32>
    %1106 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x256x5x5xf32>
    %1107 = stablehlo.maximum %1105, %1106 : tensor<1x256x5x5xf32>
    %1108 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x256x5x5xf32>
    %1109 = stablehlo.broadcast_in_dim %1107, dims = [0, 1, 2, 3] : (tensor<1x256x5x5xf32>) -> tensor<1x256x5x5xf32>
    %1110 = stablehlo.minimum %1108, %1109 : tensor<1x256x5x5xf32>
    %1111 = stablehlo.convolution(%1110, %arg81) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x5x5xf32>, tensor<512x256x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1112 = stablehlo.broadcast_in_dim %1111, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1113 = stablehlo.broadcast_in_dim %arg323, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1114 = stablehlo.subtract %1112, %1113 : tensor<1x512x5x5xf32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1116 = stablehlo.broadcast_in_dim %arg324, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1117 = stablehlo.multiply %1115, %1116 : tensor<1x512x5x5xf32>
    %1118 = stablehlo.broadcast_in_dim %1117, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1119 = stablehlo.broadcast_in_dim %arg325, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1120 = stablehlo.multiply %1118, %1119 : tensor<1x512x5x5xf32>
    %1121 = stablehlo.broadcast_in_dim %1120, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1122 = stablehlo.broadcast_in_dim %arg326, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1123 = stablehlo.add %1121, %1122 : tensor<1x512x5x5xf32>
    %1124 = stablehlo.broadcast_in_dim %1123, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1125 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x512x5x5xf32>
    %1126 = stablehlo.maximum %1124, %1125 : tensor<1x512x5x5xf32>
    %1127 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x512x5x5xf32>
    %1128 = stablehlo.broadcast_in_dim %1126, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1129 = stablehlo.minimum %1127, %1128 : tensor<1x512x5x5xf32>
    %1130 = stablehlo.convolution(%1129, %arg82) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x5x5xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x5x5xf32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2, 3] : (tensor<1x128x5x5xf32>) -> tensor<1x128x5x5xf32>
    %1132 = stablehlo.broadcast_in_dim %arg327, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x5x5xf32>
    %1133 = stablehlo.subtract %1131, %1132 : tensor<1x128x5x5xf32>
    %1134 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2, 3] : (tensor<1x128x5x5xf32>) -> tensor<1x128x5x5xf32>
    %1135 = stablehlo.broadcast_in_dim %arg328, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x5x5xf32>
    %1136 = stablehlo.multiply %1134, %1135 : tensor<1x128x5x5xf32>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x128x5x5xf32>) -> tensor<1x128x5x5xf32>
    %1138 = stablehlo.broadcast_in_dim %arg329, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x5x5xf32>
    %1139 = stablehlo.multiply %1137, %1138 : tensor<1x128x5x5xf32>
    %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2, 3] : (tensor<1x128x5x5xf32>) -> tensor<1x128x5x5xf32>
    %1141 = stablehlo.broadcast_in_dim %arg330, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x5x5xf32>
    %1142 = stablehlo.add %1140, %1141 : tensor<1x128x5x5xf32>
    %1143 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2, 3] : (tensor<1x128x5x5xf32>) -> tensor<1x128x5x5xf32>
    %1144 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x128x5x5xf32>
    %1145 = stablehlo.maximum %1143, %1144 : tensor<1x128x5x5xf32>
    %1146 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x128x5x5xf32>
    %1147 = stablehlo.broadcast_in_dim %1145, dims = [0, 1, 2, 3] : (tensor<1x128x5x5xf32>) -> tensor<1x128x5x5xf32>
    %1148 = stablehlo.minimum %1146, %1147 : tensor<1x128x5x5xf32>
    %1149 = stablehlo.convolution(%1148, %arg83) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x5x5xf32>, tensor<128x1x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1151 = stablehlo.broadcast_in_dim %arg331, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1152 = stablehlo.subtract %1150, %1151 : tensor<1x128x3x3xf32>
    %1153 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1154 = stablehlo.broadcast_in_dim %arg332, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1155 = stablehlo.multiply %1153, %1154 : tensor<1x128x3x3xf32>
    %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1157 = stablehlo.broadcast_in_dim %arg333, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1158 = stablehlo.multiply %1156, %1157 : tensor<1x128x3x3xf32>
    %1159 = stablehlo.broadcast_in_dim %1158, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1160 = stablehlo.broadcast_in_dim %arg334, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1161 = stablehlo.add %1159, %1160 : tensor<1x128x3x3xf32>
    %1162 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1163 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x128x3x3xf32>
    %1164 = stablehlo.maximum %1162, %1163 : tensor<1x128x3x3xf32>
    %1165 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x128x3x3xf32>
    %1166 = stablehlo.broadcast_in_dim %1164, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1167 = stablehlo.minimum %1165, %1166 : tensor<1x128x3x3xf32>
    %1168 = stablehlo.convolution(%1167, %arg84) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x3x3xf32>, tensor<256x128x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1169 = stablehlo.broadcast_in_dim %1168, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1170 = stablehlo.broadcast_in_dim %arg335, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1171 = stablehlo.subtract %1169, %1170 : tensor<1x256x3x3xf32>
    %1172 = stablehlo.broadcast_in_dim %1171, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1173 = stablehlo.broadcast_in_dim %arg336, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1174 = stablehlo.multiply %1172, %1173 : tensor<1x256x3x3xf32>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1176 = stablehlo.broadcast_in_dim %arg337, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1177 = stablehlo.multiply %1175, %1176 : tensor<1x256x3x3xf32>
    %1178 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1179 = stablehlo.broadcast_in_dim %arg338, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1180 = stablehlo.add %1178, %1179 : tensor<1x256x3x3xf32>
    %1181 = stablehlo.broadcast_in_dim %1180, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1182 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x256x3x3xf32>
    %1183 = stablehlo.maximum %1181, %1182 : tensor<1x256x3x3xf32>
    %1184 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x256x3x3xf32>
    %1185 = stablehlo.broadcast_in_dim %1183, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1186 = stablehlo.minimum %1184, %1185 : tensor<1x256x3x3xf32>
    %1187 = stablehlo.convolution(%1186, %arg85) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x3x3xf32>, tensor<128x256x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1188 = stablehlo.broadcast_in_dim %1187, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1189 = stablehlo.broadcast_in_dim %arg339, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1190 = stablehlo.subtract %1188, %1189 : tensor<1x128x3x3xf32>
    %1191 = stablehlo.broadcast_in_dim %1190, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1192 = stablehlo.broadcast_in_dim %arg340, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1193 = stablehlo.multiply %1191, %1192 : tensor<1x128x3x3xf32>
    %1194 = stablehlo.broadcast_in_dim %1193, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1195 = stablehlo.broadcast_in_dim %arg341, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1196 = stablehlo.multiply %1194, %1195 : tensor<1x128x3x3xf32>
    %1197 = stablehlo.broadcast_in_dim %1196, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1198 = stablehlo.broadcast_in_dim %arg342, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x3x3xf32>
    %1199 = stablehlo.add %1197, %1198 : tensor<1x128x3x3xf32>
    %1200 = stablehlo.broadcast_in_dim %1199, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1201 = stablehlo.maximum %1200, %1163 : tensor<1x128x3x3xf32>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2, 3] : (tensor<1x128x3x3xf32>) -> tensor<1x128x3x3xf32>
    %1203 = stablehlo.minimum %1165, %1202 : tensor<1x128x3x3xf32>
    %1204 = stablehlo.convolution(%1203, %arg86) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x3x3xf32>, tensor<128x1x3x3xf32>) -> tensor<1x128x2x2xf32>
    %1205 = stablehlo.broadcast_in_dim %1204, dims = [0, 1, 2, 3] : (tensor<1x128x2x2xf32>) -> tensor<1x128x2x2xf32>
    %1206 = stablehlo.broadcast_in_dim %arg343, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x2x2xf32>
    %1207 = stablehlo.subtract %1205, %1206 : tensor<1x128x2x2xf32>
    %1208 = stablehlo.broadcast_in_dim %1207, dims = [0, 1, 2, 3] : (tensor<1x128x2x2xf32>) -> tensor<1x128x2x2xf32>
    %1209 = stablehlo.broadcast_in_dim %arg344, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x2x2xf32>
    %1210 = stablehlo.multiply %1208, %1209 : tensor<1x128x2x2xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x128x2x2xf32>) -> tensor<1x128x2x2xf32>
    %1212 = stablehlo.broadcast_in_dim %arg345, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x2x2xf32>
    %1213 = stablehlo.multiply %1211, %1212 : tensor<1x128x2x2xf32>
    %1214 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2, 3] : (tensor<1x128x2x2xf32>) -> tensor<1x128x2x2xf32>
    %1215 = stablehlo.broadcast_in_dim %arg346, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x2x2xf32>
    %1216 = stablehlo.add %1214, %1215 : tensor<1x128x2x2xf32>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2, 3] : (tensor<1x128x2x2xf32>) -> tensor<1x128x2x2xf32>
    %1218 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x128x2x2xf32>
    %1219 = stablehlo.maximum %1217, %1218 : tensor<1x128x2x2xf32>
    %1220 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x128x2x2xf32>
    %1221 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2, 3] : (tensor<1x128x2x2xf32>) -> tensor<1x128x2x2xf32>
    %1222 = stablehlo.minimum %1220, %1221 : tensor<1x128x2x2xf32>
    %1223 = stablehlo.convolution(%1222, %arg87) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x2x2xf32>, tensor<256x128x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1224 = stablehlo.broadcast_in_dim %1223, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1225 = stablehlo.broadcast_in_dim %arg347, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1226 = stablehlo.subtract %1224, %1225 : tensor<1x256x2x2xf32>
    %1227 = stablehlo.broadcast_in_dim %1226, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1228 = stablehlo.broadcast_in_dim %arg348, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1229 = stablehlo.multiply %1227, %1228 : tensor<1x256x2x2xf32>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1231 = stablehlo.broadcast_in_dim %arg349, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1232 = stablehlo.multiply %1230, %1231 : tensor<1x256x2x2xf32>
    %1233 = stablehlo.broadcast_in_dim %1232, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1234 = stablehlo.broadcast_in_dim %arg350, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1235 = stablehlo.add %1233, %1234 : tensor<1x256x2x2xf32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1237 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x256x2x2xf32>
    %1238 = stablehlo.maximum %1236, %1237 : tensor<1x256x2x2xf32>
    %1239 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x256x2x2xf32>
    %1240 = stablehlo.broadcast_in_dim %1238, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1241 = stablehlo.minimum %1239, %1240 : tensor<1x256x2x2xf32>
    %1242 = stablehlo.convolution(%1241, %arg88) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x2x2xf32>, tensor<64x256x1x1xf32>) -> tensor<1x64x2x2xf32>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [0, 1, 2, 3] : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
    %1244 = stablehlo.broadcast_in_dim %arg351, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x2x2xf32>
    %1245 = stablehlo.subtract %1243, %1244 : tensor<1x64x2x2xf32>
    %1246 = stablehlo.broadcast_in_dim %1245, dims = [0, 1, 2, 3] : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
    %1247 = stablehlo.broadcast_in_dim %arg352, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x2x2xf32>
    %1248 = stablehlo.multiply %1246, %1247 : tensor<1x64x2x2xf32>
    %1249 = stablehlo.broadcast_in_dim %1248, dims = [0, 1, 2, 3] : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
    %1250 = stablehlo.broadcast_in_dim %arg353, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x2x2xf32>
    %1251 = stablehlo.multiply %1249, %1250 : tensor<1x64x2x2xf32>
    %1252 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2, 3] : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
    %1253 = stablehlo.broadcast_in_dim %arg354, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x2x2xf32>
    %1254 = stablehlo.add %1252, %1253 : tensor<1x64x2x2xf32>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [0, 1, 2, 3] : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
    %1256 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x64x2x2xf32>
    %1257 = stablehlo.maximum %1255, %1256 : tensor<1x64x2x2xf32>
    %1258 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x64x2x2xf32>
    %1259 = stablehlo.broadcast_in_dim %1257, dims = [0, 1, 2, 3] : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
    %1260 = stablehlo.minimum %1258, %1259 : tensor<1x64x2x2xf32>
    %1261 = stablehlo.convolution(%1260, %arg89) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<1x64x2x2xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x1x1xf32>
    %1262 = stablehlo.broadcast_in_dim %1261, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1263 = stablehlo.broadcast_in_dim %arg355, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1264 = stablehlo.subtract %1262, %1263 : tensor<1x64x1x1xf32>
    %1265 = stablehlo.broadcast_in_dim %1264, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1266 = stablehlo.broadcast_in_dim %arg356, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1267 = stablehlo.multiply %1265, %1266 : tensor<1x64x1x1xf32>
    %1268 = stablehlo.broadcast_in_dim %1267, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1269 = stablehlo.broadcast_in_dim %arg357, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1270 = stablehlo.multiply %1268, %1269 : tensor<1x64x1x1xf32>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1272 = stablehlo.broadcast_in_dim %arg358, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1273 = stablehlo.add %1271, %1272 : tensor<1x64x1x1xf32>
    %1274 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1275 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x64x1x1xf32>
    %1276 = stablehlo.maximum %1274, %1275 : tensor<1x64x1x1xf32>
    %1277 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x64x1x1xf32>
    %1278 = stablehlo.broadcast_in_dim %1276, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %1279 = stablehlo.minimum %1277, %1278 : tensor<1x64x1x1xf32>
    %1280 = stablehlo.convolution(%1279, %arg90) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x1x1xf32>, tensor<128x64x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1281 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1282 = stablehlo.broadcast_in_dim %arg359, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1283 = stablehlo.subtract %1281, %1282 : tensor<1x128x1x1xf32>
    %1284 = stablehlo.broadcast_in_dim %1283, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1285 = stablehlo.broadcast_in_dim %arg360, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1286 = stablehlo.multiply %1284, %1285 : tensor<1x128x1x1xf32>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1288 = stablehlo.broadcast_in_dim %arg361, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1289 = stablehlo.multiply %1287, %1288 : tensor<1x128x1x1xf32>
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1291 = stablehlo.broadcast_in_dim %arg362, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1292 = stablehlo.add %1290, %1291 : tensor<1x128x1x1xf32>
    %1293 = stablehlo.broadcast_in_dim %1292, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1294 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x128x1x1xf32>
    %1295 = stablehlo.maximum %1293, %1294 : tensor<1x128x1x1xf32>
    %1296 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x128x1x1xf32>
    %1297 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1298 = stablehlo.minimum %1296, %1297 : tensor<1x128x1x1xf32>
    %1299 = stablehlo.convolution(%815, %arg91) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<1x672x20x20xf32>, tensor<672x1x3x3xf32>) -> tensor<1x672x20x20xf32>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1301 = stablehlo.broadcast_in_dim %arg363, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1302 = stablehlo.subtract %1300, %1301 : tensor<1x672x20x20xf32>
    %1303 = stablehlo.broadcast_in_dim %1302, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1304 = stablehlo.broadcast_in_dim %arg364, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1305 = stablehlo.multiply %1303, %1304 : tensor<1x672x20x20xf32>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1307 = stablehlo.broadcast_in_dim %arg365, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1308 = stablehlo.multiply %1306, %1307 : tensor<1x672x20x20xf32>
    %1309 = stablehlo.broadcast_in_dim %1308, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1310 = stablehlo.broadcast_in_dim %arg366, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1311 = stablehlo.add %1309, %1310 : tensor<1x672x20x20xf32>
    %1312 = stablehlo.broadcast_in_dim %1311, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1313 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x672x20x20xf32>
    %1314 = stablehlo.maximum %1312, %1313 : tensor<1x672x20x20xf32>
    %1315 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x672x20x20xf32>
    %1316 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1317 = stablehlo.minimum %1315, %1316 : tensor<1x672x20x20xf32>
    %1318 = stablehlo.convolution(%1317, %arg92) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x672x20x20xf32>, tensor<24x672x1x1xf32>) -> tensor<1x24x20x20xf32>
    %1319 = stablehlo.reshape %arg93 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %1320 = stablehlo.broadcast_in_dim %1318, dims = [0, 1, 2, 3] : (tensor<1x24x20x20xf32>) -> tensor<1x24x20x20xf32>
    %1321 = stablehlo.broadcast_in_dim %1319, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x20x20xf32>
    %1322 = stablehlo.add %1320, %1321 : tensor<1x24x20x20xf32>
    %1323 = stablehlo.reshape %1322 : (tensor<1x24x20x20xf32>) -> tensor<1x6x4x20x20xf32>
    %1324 = stablehlo.transpose %1323, dims = [0, 3, 4, 1, 2] : (tensor<1x6x4x20x20xf32>) -> tensor<1x20x20x6x4xf32>
    %1325 = stablehlo.reshape %1324 : (tensor<1x20x20x6x4xf32>) -> tensor<1x2400x4xf32>
    %1326 = stablehlo.convolution(%1070, %arg94) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<1x480x10x10xf32>, tensor<480x1x3x3xf32>) -> tensor<1x480x10x10xf32>
    %1327 = stablehlo.broadcast_in_dim %1326, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1328 = stablehlo.broadcast_in_dim %arg367, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1329 = stablehlo.subtract %1327, %1328 : tensor<1x480x10x10xf32>
    %1330 = stablehlo.broadcast_in_dim %1329, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1331 = stablehlo.broadcast_in_dim %arg368, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1332 = stablehlo.multiply %1330, %1331 : tensor<1x480x10x10xf32>
    %1333 = stablehlo.broadcast_in_dim %1332, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1334 = stablehlo.broadcast_in_dim %arg369, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1335 = stablehlo.multiply %1333, %1334 : tensor<1x480x10x10xf32>
    %1336 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1337 = stablehlo.broadcast_in_dim %arg370, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1338 = stablehlo.add %1336, %1337 : tensor<1x480x10x10xf32>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1340 = stablehlo.broadcast_in_dim %1084, dims = [] : (tensor<f32>) -> tensor<1x480x10x10xf32>
    %1341 = stablehlo.maximum %1339, %1340 : tensor<1x480x10x10xf32>
    %1342 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<1x480x10x10xf32>
    %1343 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1344 = stablehlo.minimum %1342, %1343 : tensor<1x480x10x10xf32>
    %1345 = stablehlo.convolution(%1344, %arg95) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x10x10xf32>, tensor<24x480x1x1xf32>) -> tensor<1x24x10x10xf32>
    %1346 = stablehlo.reshape %arg96 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %1347 = stablehlo.broadcast_in_dim %1345, dims = [0, 1, 2, 3] : (tensor<1x24x10x10xf32>) -> tensor<1x24x10x10xf32>
    %1348 = stablehlo.broadcast_in_dim %1346, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x10x10xf32>
    %1349 = stablehlo.add %1347, %1348 : tensor<1x24x10x10xf32>
    %1350 = stablehlo.reshape %1349 : (tensor<1x24x10x10xf32>) -> tensor<1x6x4x10x10xf32>
    %1351 = stablehlo.transpose %1350, dims = [0, 3, 4, 1, 2] : (tensor<1x6x4x10x10xf32>) -> tensor<1x10x10x6x4xf32>
    %1352 = stablehlo.reshape %1351 : (tensor<1x10x10x6x4xf32>) -> tensor<1x600x4xf32>
    %1353 = stablehlo.convolution(%1129, %arg97) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x5x5xf32>, tensor<512x1x3x3xf32>) -> tensor<1x512x5x5xf32>
    %1354 = stablehlo.broadcast_in_dim %1353, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1355 = stablehlo.broadcast_in_dim %arg371, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1356 = stablehlo.subtract %1354, %1355 : tensor<1x512x5x5xf32>
    %1357 = stablehlo.broadcast_in_dim %1356, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1358 = stablehlo.broadcast_in_dim %arg372, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1359 = stablehlo.multiply %1357, %1358 : tensor<1x512x5x5xf32>
    %1360 = stablehlo.broadcast_in_dim %1359, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1361 = stablehlo.broadcast_in_dim %arg373, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1362 = stablehlo.multiply %1360, %1361 : tensor<1x512x5x5xf32>
    %1363 = stablehlo.broadcast_in_dim %1362, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1364 = stablehlo.broadcast_in_dim %arg374, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1365 = stablehlo.add %1363, %1364 : tensor<1x512x5x5xf32>
    %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1367 = stablehlo.maximum %1366, %1125 : tensor<1x512x5x5xf32>
    %1368 = stablehlo.broadcast_in_dim %1367, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1369 = stablehlo.minimum %1127, %1368 : tensor<1x512x5x5xf32>
    %1370 = stablehlo.convolution(%1369, %arg98) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x5x5xf32>, tensor<24x512x1x1xf32>) -> tensor<1x24x5x5xf32>
    %1371 = stablehlo.reshape %arg99 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %1372 = stablehlo.broadcast_in_dim %1370, dims = [0, 1, 2, 3] : (tensor<1x24x5x5xf32>) -> tensor<1x24x5x5xf32>
    %1373 = stablehlo.broadcast_in_dim %1371, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x5x5xf32>
    %1374 = stablehlo.add %1372, %1373 : tensor<1x24x5x5xf32>
    %1375 = stablehlo.reshape %1374 : (tensor<1x24x5x5xf32>) -> tensor<1x6x4x5x5xf32>
    %1376 = stablehlo.transpose %1375, dims = [0, 3, 4, 1, 2] : (tensor<1x6x4x5x5xf32>) -> tensor<1x5x5x6x4xf32>
    %1377 = stablehlo.reshape %1376 : (tensor<1x5x5x6x4xf32>) -> tensor<1x150x4xf32>
    %1378 = stablehlo.convolution(%1186, %arg100) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x3x3xf32>, tensor<256x1x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1379 = stablehlo.broadcast_in_dim %1378, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1380 = stablehlo.broadcast_in_dim %arg375, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1381 = stablehlo.subtract %1379, %1380 : tensor<1x256x3x3xf32>
    %1382 = stablehlo.broadcast_in_dim %1381, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1383 = stablehlo.broadcast_in_dim %arg376, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1384 = stablehlo.multiply %1382, %1383 : tensor<1x256x3x3xf32>
    %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1386 = stablehlo.broadcast_in_dim %arg377, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1387 = stablehlo.multiply %1385, %1386 : tensor<1x256x3x3xf32>
    %1388 = stablehlo.broadcast_in_dim %1387, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1389 = stablehlo.broadcast_in_dim %arg378, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1390 = stablehlo.add %1388, %1389 : tensor<1x256x3x3xf32>
    %1391 = stablehlo.broadcast_in_dim %1390, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1392 = stablehlo.maximum %1391, %1182 : tensor<1x256x3x3xf32>
    %1393 = stablehlo.broadcast_in_dim %1392, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1394 = stablehlo.minimum %1184, %1393 : tensor<1x256x3x3xf32>
    %1395 = stablehlo.convolution(%1394, %arg101) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x3x3xf32>, tensor<24x256x1x1xf32>) -> tensor<1x24x3x3xf32>
    %1396 = stablehlo.reshape %arg102 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %1397 = stablehlo.broadcast_in_dim %1395, dims = [0, 1, 2, 3] : (tensor<1x24x3x3xf32>) -> tensor<1x24x3x3xf32>
    %1398 = stablehlo.broadcast_in_dim %1396, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x3x3xf32>
    %1399 = stablehlo.add %1397, %1398 : tensor<1x24x3x3xf32>
    %1400 = stablehlo.reshape %1399 : (tensor<1x24x3x3xf32>) -> tensor<1x6x4x3x3xf32>
    %1401 = stablehlo.transpose %1400, dims = [0, 3, 4, 1, 2] : (tensor<1x6x4x3x3xf32>) -> tensor<1x3x3x6x4xf32>
    %1402 = stablehlo.reshape %1401 : (tensor<1x3x3x6x4xf32>) -> tensor<1x54x4xf32>
    %1403 = stablehlo.convolution(%1241, %arg103) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x2x2xf32>, tensor<256x1x3x3xf32>) -> tensor<1x256x2x2xf32>
    %1404 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1405 = stablehlo.broadcast_in_dim %arg379, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1406 = stablehlo.subtract %1404, %1405 : tensor<1x256x2x2xf32>
    %1407 = stablehlo.broadcast_in_dim %1406, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1408 = stablehlo.broadcast_in_dim %arg380, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1409 = stablehlo.multiply %1407, %1408 : tensor<1x256x2x2xf32>
    %1410 = stablehlo.broadcast_in_dim %1409, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1411 = stablehlo.broadcast_in_dim %arg381, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1412 = stablehlo.multiply %1410, %1411 : tensor<1x256x2x2xf32>
    %1413 = stablehlo.broadcast_in_dim %1412, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1414 = stablehlo.broadcast_in_dim %arg382, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1415 = stablehlo.add %1413, %1414 : tensor<1x256x2x2xf32>
    %1416 = stablehlo.broadcast_in_dim %1415, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1417 = stablehlo.maximum %1416, %1237 : tensor<1x256x2x2xf32>
    %1418 = stablehlo.broadcast_in_dim %1417, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1419 = stablehlo.minimum %1239, %1418 : tensor<1x256x2x2xf32>
    %1420 = stablehlo.convolution(%1419, %arg104) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x2x2xf32>, tensor<24x256x1x1xf32>) -> tensor<1x24x2x2xf32>
    %1421 = stablehlo.reshape %arg105 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %1422 = stablehlo.broadcast_in_dim %1420, dims = [0, 1, 2, 3] : (tensor<1x24x2x2xf32>) -> tensor<1x24x2x2xf32>
    %1423 = stablehlo.broadcast_in_dim %1421, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x2x2xf32>
    %1424 = stablehlo.add %1422, %1423 : tensor<1x24x2x2xf32>
    %1425 = stablehlo.reshape %1424 : (tensor<1x24x2x2xf32>) -> tensor<1x6x4x2x2xf32>
    %1426 = stablehlo.transpose %1425, dims = [0, 3, 4, 1, 2] : (tensor<1x6x4x2x2xf32>) -> tensor<1x2x2x6x4xf32>
    %1427 = stablehlo.reshape %1426 : (tensor<1x2x2x6x4xf32>) -> tensor<1x24x4xf32>
    %1428 = stablehlo.convolution(%1298, %arg106) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x1x1xf32>, tensor<128x1x3x3xf32>) -> tensor<1x128x1x1xf32>
    %1429 = stablehlo.broadcast_in_dim %1428, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1430 = stablehlo.broadcast_in_dim %arg383, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1431 = stablehlo.subtract %1429, %1430 : tensor<1x128x1x1xf32>
    %1432 = stablehlo.broadcast_in_dim %1431, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1433 = stablehlo.broadcast_in_dim %arg384, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1434 = stablehlo.multiply %1432, %1433 : tensor<1x128x1x1xf32>
    %1435 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1436 = stablehlo.broadcast_in_dim %arg385, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1437 = stablehlo.multiply %1435, %1436 : tensor<1x128x1x1xf32>
    %1438 = stablehlo.broadcast_in_dim %1437, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1439 = stablehlo.broadcast_in_dim %arg386, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1440 = stablehlo.add %1438, %1439 : tensor<1x128x1x1xf32>
    %1441 = stablehlo.broadcast_in_dim %1440, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1442 = stablehlo.maximum %1441, %1294 : tensor<1x128x1x1xf32>
    %1443 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1444 = stablehlo.minimum %1296, %1443 : tensor<1x128x1x1xf32>
    %1445 = stablehlo.convolution(%1444, %arg107) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x1x1xf32>, tensor<24x128x1x1xf32>) -> tensor<1x24x1x1xf32>
    %1446 = stablehlo.reshape %arg108 : (tensor<24xf32>) -> tensor<24x1x1xf32>
    %1447 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2, 3] : (tensor<1x24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %1448 = stablehlo.broadcast_in_dim %1446, dims = [1, 2, 3] : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %1449 = stablehlo.add %1447, %1448 : tensor<1x24x1x1xf32>
    %1450 = stablehlo.reshape %1449 : (tensor<1x24x1x1xf32>) -> tensor<1x6x4x1x1xf32>
    %1451 = stablehlo.transpose %1450, dims = [0, 3, 4, 1, 2] : (tensor<1x6x4x1x1xf32>) -> tensor<1x1x1x6x4xf32>
    %1452 = stablehlo.reshape %1451 : (tensor<1x1x1x6x4xf32>) -> tensor<1x6x4xf32>
    %1453 = stablehlo.concatenate %1325, %1352, %1377, %1402, %1427, %1452, dim = 1 : (tensor<1x2400x4xf32>, tensor<1x600x4xf32>, tensor<1x150x4xf32>, tensor<1x54x4xf32>, tensor<1x24x4xf32>, tensor<1x6x4xf32>) -> tensor<1x3234x4xf32>
    %1454 = stablehlo.convolution(%815, %arg109) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<1x672x20x20xf32>, tensor<672x1x3x3xf32>) -> tensor<1x672x20x20xf32>
    %1455 = stablehlo.broadcast_in_dim %1454, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1456 = stablehlo.broadcast_in_dim %arg387, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1457 = stablehlo.subtract %1455, %1456 : tensor<1x672x20x20xf32>
    %1458 = stablehlo.broadcast_in_dim %1457, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1459 = stablehlo.broadcast_in_dim %arg388, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1460 = stablehlo.multiply %1458, %1459 : tensor<1x672x20x20xf32>
    %1461 = stablehlo.broadcast_in_dim %1460, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1462 = stablehlo.broadcast_in_dim %arg389, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1463 = stablehlo.multiply %1461, %1462 : tensor<1x672x20x20xf32>
    %1464 = stablehlo.broadcast_in_dim %1463, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1465 = stablehlo.broadcast_in_dim %arg390, dims = [1, 2, 3] : (tensor<672x1x1xf32>) -> tensor<1x672x20x20xf32>
    %1466 = stablehlo.add %1464, %1465 : tensor<1x672x20x20xf32>
    %1467 = stablehlo.broadcast_in_dim %1466, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1468 = stablehlo.maximum %1467, %1313 : tensor<1x672x20x20xf32>
    %1469 = stablehlo.broadcast_in_dim %1468, dims = [0, 1, 2, 3] : (tensor<1x672x20x20xf32>) -> tensor<1x672x20x20xf32>
    %1470 = stablehlo.minimum %1315, %1469 : tensor<1x672x20x20xf32>
    %1471 = stablehlo.convolution(%1470, %arg110) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x672x20x20xf32>, tensor<546x672x1x1xf32>) -> tensor<1x546x20x20xf32>
    %1472 = stablehlo.reshape %arg111 : (tensor<546xf32>) -> tensor<546x1x1xf32>
    %1473 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2, 3] : (tensor<1x546x20x20xf32>) -> tensor<1x546x20x20xf32>
    %1474 = stablehlo.broadcast_in_dim %1472, dims = [1, 2, 3] : (tensor<546x1x1xf32>) -> tensor<1x546x20x20xf32>
    %1475 = stablehlo.add %1473, %1474 : tensor<1x546x20x20xf32>
    %1476 = stablehlo.reshape %1475 : (tensor<1x546x20x20xf32>) -> tensor<1x6x91x20x20xf32>
    %1477 = stablehlo.transpose %1476, dims = [0, 3, 4, 1, 2] : (tensor<1x6x91x20x20xf32>) -> tensor<1x20x20x6x91xf32>
    %1478 = stablehlo.reshape %1477 : (tensor<1x20x20x6x91xf32>) -> tensor<1x2400x91xf32>
    %1479 = stablehlo.convolution(%1070, %arg112) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<1x480x10x10xf32>, tensor<480x1x3x3xf32>) -> tensor<1x480x10x10xf32>
    %1480 = stablehlo.broadcast_in_dim %1479, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1481 = stablehlo.broadcast_in_dim %arg391, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1482 = stablehlo.subtract %1480, %1481 : tensor<1x480x10x10xf32>
    %1483 = stablehlo.broadcast_in_dim %1482, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1484 = stablehlo.broadcast_in_dim %arg392, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1485 = stablehlo.multiply %1483, %1484 : tensor<1x480x10x10xf32>
    %1486 = stablehlo.broadcast_in_dim %1485, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1487 = stablehlo.broadcast_in_dim %arg393, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1488 = stablehlo.multiply %1486, %1487 : tensor<1x480x10x10xf32>
    %1489 = stablehlo.broadcast_in_dim %1488, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1490 = stablehlo.broadcast_in_dim %arg394, dims = [1, 2, 3] : (tensor<480x1x1xf32>) -> tensor<1x480x10x10xf32>
    %1491 = stablehlo.add %1489, %1490 : tensor<1x480x10x10xf32>
    %1492 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1493 = stablehlo.maximum %1492, %1340 : tensor<1x480x10x10xf32>
    %1494 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x480x10x10xf32>) -> tensor<1x480x10x10xf32>
    %1495 = stablehlo.minimum %1342, %1494 : tensor<1x480x10x10xf32>
    %1496 = stablehlo.convolution(%1495, %arg113) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x480x10x10xf32>, tensor<546x480x1x1xf32>) -> tensor<1x546x10x10xf32>
    %1497 = stablehlo.reshape %arg114 : (tensor<546xf32>) -> tensor<546x1x1xf32>
    %1498 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2, 3] : (tensor<1x546x10x10xf32>) -> tensor<1x546x10x10xf32>
    %1499 = stablehlo.broadcast_in_dim %1497, dims = [1, 2, 3] : (tensor<546x1x1xf32>) -> tensor<1x546x10x10xf32>
    %1500 = stablehlo.add %1498, %1499 : tensor<1x546x10x10xf32>
    %1501 = stablehlo.reshape %1500 : (tensor<1x546x10x10xf32>) -> tensor<1x6x91x10x10xf32>
    %1502 = stablehlo.transpose %1501, dims = [0, 3, 4, 1, 2] : (tensor<1x6x91x10x10xf32>) -> tensor<1x10x10x6x91xf32>
    %1503 = stablehlo.reshape %1502 : (tensor<1x10x10x6x91xf32>) -> tensor<1x600x91xf32>
    %1504 = stablehlo.convolution(%1129, %arg115) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<1x512x5x5xf32>, tensor<512x1x3x3xf32>) -> tensor<1x512x5x5xf32>
    %1505 = stablehlo.broadcast_in_dim %1504, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1506 = stablehlo.broadcast_in_dim %arg395, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1507 = stablehlo.subtract %1505, %1506 : tensor<1x512x5x5xf32>
    %1508 = stablehlo.broadcast_in_dim %1507, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1509 = stablehlo.broadcast_in_dim %arg396, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1510 = stablehlo.multiply %1508, %1509 : tensor<1x512x5x5xf32>
    %1511 = stablehlo.broadcast_in_dim %1510, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1512 = stablehlo.broadcast_in_dim %arg397, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1513 = stablehlo.multiply %1511, %1512 : tensor<1x512x5x5xf32>
    %1514 = stablehlo.broadcast_in_dim %1513, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1515 = stablehlo.broadcast_in_dim %arg398, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x5x5xf32>
    %1516 = stablehlo.add %1514, %1515 : tensor<1x512x5x5xf32>
    %1517 = stablehlo.broadcast_in_dim %1516, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1518 = stablehlo.maximum %1517, %1125 : tensor<1x512x5x5xf32>
    %1519 = stablehlo.broadcast_in_dim %1518, dims = [0, 1, 2, 3] : (tensor<1x512x5x5xf32>) -> tensor<1x512x5x5xf32>
    %1520 = stablehlo.minimum %1127, %1519 : tensor<1x512x5x5xf32>
    %1521 = stablehlo.convolution(%1520, %arg116) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x5x5xf32>, tensor<546x512x1x1xf32>) -> tensor<1x546x5x5xf32>
    %1522 = stablehlo.reshape %arg117 : (tensor<546xf32>) -> tensor<546x1x1xf32>
    %1523 = stablehlo.broadcast_in_dim %1521, dims = [0, 1, 2, 3] : (tensor<1x546x5x5xf32>) -> tensor<1x546x5x5xf32>
    %1524 = stablehlo.broadcast_in_dim %1522, dims = [1, 2, 3] : (tensor<546x1x1xf32>) -> tensor<1x546x5x5xf32>
    %1525 = stablehlo.add %1523, %1524 : tensor<1x546x5x5xf32>
    %1526 = stablehlo.reshape %1525 : (tensor<1x546x5x5xf32>) -> tensor<1x6x91x5x5xf32>
    %1527 = stablehlo.transpose %1526, dims = [0, 3, 4, 1, 2] : (tensor<1x6x91x5x5xf32>) -> tensor<1x5x5x6x91xf32>
    %1528 = stablehlo.reshape %1527 : (tensor<1x5x5x6x91xf32>) -> tensor<1x150x91xf32>
    %1529 = stablehlo.convolution(%1186, %arg118) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x3x3xf32>, tensor<256x1x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1530 = stablehlo.broadcast_in_dim %1529, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1531 = stablehlo.broadcast_in_dim %arg399, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1532 = stablehlo.subtract %1530, %1531 : tensor<1x256x3x3xf32>
    %1533 = stablehlo.broadcast_in_dim %1532, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1534 = stablehlo.broadcast_in_dim %arg400, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1535 = stablehlo.multiply %1533, %1534 : tensor<1x256x3x3xf32>
    %1536 = stablehlo.broadcast_in_dim %1535, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1537 = stablehlo.broadcast_in_dim %arg401, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1538 = stablehlo.multiply %1536, %1537 : tensor<1x256x3x3xf32>
    %1539 = stablehlo.broadcast_in_dim %1538, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1540 = stablehlo.broadcast_in_dim %arg402, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x3x3xf32>
    %1541 = stablehlo.add %1539, %1540 : tensor<1x256x3x3xf32>
    %1542 = stablehlo.broadcast_in_dim %1541, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1543 = stablehlo.maximum %1542, %1182 : tensor<1x256x3x3xf32>
    %1544 = stablehlo.broadcast_in_dim %1543, dims = [0, 1, 2, 3] : (tensor<1x256x3x3xf32>) -> tensor<1x256x3x3xf32>
    %1545 = stablehlo.minimum %1184, %1544 : tensor<1x256x3x3xf32>
    %1546 = stablehlo.convolution(%1545, %arg119) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x3x3xf32>, tensor<546x256x1x1xf32>) -> tensor<1x546x3x3xf32>
    %1547 = stablehlo.reshape %arg120 : (tensor<546xf32>) -> tensor<546x1x1xf32>
    %1548 = stablehlo.broadcast_in_dim %1546, dims = [0, 1, 2, 3] : (tensor<1x546x3x3xf32>) -> tensor<1x546x3x3xf32>
    %1549 = stablehlo.broadcast_in_dim %1547, dims = [1, 2, 3] : (tensor<546x1x1xf32>) -> tensor<1x546x3x3xf32>
    %1550 = stablehlo.add %1548, %1549 : tensor<1x546x3x3xf32>
    %1551 = stablehlo.reshape %1550 : (tensor<1x546x3x3xf32>) -> tensor<1x6x91x3x3xf32>
    %1552 = stablehlo.transpose %1551, dims = [0, 3, 4, 1, 2] : (tensor<1x6x91x3x3xf32>) -> tensor<1x3x3x6x91xf32>
    %1553 = stablehlo.reshape %1552 : (tensor<1x3x3x6x91xf32>) -> tensor<1x54x91xf32>
    %1554 = stablehlo.convolution(%1241, %arg121) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<1x256x2x2xf32>, tensor<256x1x3x3xf32>) -> tensor<1x256x2x2xf32>
    %1555 = stablehlo.broadcast_in_dim %1554, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1556 = stablehlo.broadcast_in_dim %arg403, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1557 = stablehlo.subtract %1555, %1556 : tensor<1x256x2x2xf32>
    %1558 = stablehlo.broadcast_in_dim %1557, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1559 = stablehlo.broadcast_in_dim %arg404, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1560 = stablehlo.multiply %1558, %1559 : tensor<1x256x2x2xf32>
    %1561 = stablehlo.broadcast_in_dim %1560, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1562 = stablehlo.broadcast_in_dim %arg405, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1563 = stablehlo.multiply %1561, %1562 : tensor<1x256x2x2xf32>
    %1564 = stablehlo.broadcast_in_dim %1563, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1565 = stablehlo.broadcast_in_dim %arg406, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x2x2xf32>
    %1566 = stablehlo.add %1564, %1565 : tensor<1x256x2x2xf32>
    %1567 = stablehlo.broadcast_in_dim %1566, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1568 = stablehlo.maximum %1567, %1237 : tensor<1x256x2x2xf32>
    %1569 = stablehlo.broadcast_in_dim %1568, dims = [0, 1, 2, 3] : (tensor<1x256x2x2xf32>) -> tensor<1x256x2x2xf32>
    %1570 = stablehlo.minimum %1239, %1569 : tensor<1x256x2x2xf32>
    %1571 = stablehlo.convolution(%1570, %arg122) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x2x2xf32>, tensor<546x256x1x1xf32>) -> tensor<1x546x2x2xf32>
    %1572 = stablehlo.reshape %arg123 : (tensor<546xf32>) -> tensor<546x1x1xf32>
    %1573 = stablehlo.broadcast_in_dim %1571, dims = [0, 1, 2, 3] : (tensor<1x546x2x2xf32>) -> tensor<1x546x2x2xf32>
    %1574 = stablehlo.broadcast_in_dim %1572, dims = [1, 2, 3] : (tensor<546x1x1xf32>) -> tensor<1x546x2x2xf32>
    %1575 = stablehlo.add %1573, %1574 : tensor<1x546x2x2xf32>
    %1576 = stablehlo.reshape %1575 : (tensor<1x546x2x2xf32>) -> tensor<1x6x91x2x2xf32>
    %1577 = stablehlo.transpose %1576, dims = [0, 3, 4, 1, 2] : (tensor<1x6x91x2x2xf32>) -> tensor<1x2x2x6x91xf32>
    %1578 = stablehlo.reshape %1577 : (tensor<1x2x2x6x91xf32>) -> tensor<1x24x91xf32>
    %1579 = stablehlo.convolution(%1298, %arg124) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<1x128x1x1xf32>, tensor<128x1x3x3xf32>) -> tensor<1x128x1x1xf32>
    %1580 = stablehlo.broadcast_in_dim %1579, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1581 = stablehlo.broadcast_in_dim %arg407, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1582 = stablehlo.subtract %1580, %1581 : tensor<1x128x1x1xf32>
    %1583 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1584 = stablehlo.broadcast_in_dim %arg408, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1585 = stablehlo.multiply %1583, %1584 : tensor<1x128x1x1xf32>
    %1586 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1587 = stablehlo.broadcast_in_dim %arg409, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1588 = stablehlo.multiply %1586, %1587 : tensor<1x128x1x1xf32>
    %1589 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1590 = stablehlo.broadcast_in_dim %arg410, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1591 = stablehlo.add %1589, %1590 : tensor<1x128x1x1xf32>
    %1592 = stablehlo.broadcast_in_dim %1591, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1593 = stablehlo.maximum %1592, %1294 : tensor<1x128x1x1xf32>
    %1594 = stablehlo.broadcast_in_dim %1593, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %1595 = stablehlo.minimum %1296, %1594 : tensor<1x128x1x1xf32>
    %1596 = stablehlo.convolution(%1595, %arg125) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x1x1xf32>, tensor<546x128x1x1xf32>) -> tensor<1x546x1x1xf32>
    %1597 = stablehlo.reshape %arg126 : (tensor<546xf32>) -> tensor<546x1x1xf32>
    %1598 = stablehlo.broadcast_in_dim %1596, dims = [0, 1, 2, 3] : (tensor<1x546x1x1xf32>) -> tensor<1x546x1x1xf32>
    %1599 = stablehlo.broadcast_in_dim %1597, dims = [1, 2, 3] : (tensor<546x1x1xf32>) -> tensor<1x546x1x1xf32>
    %1600 = stablehlo.add %1598, %1599 : tensor<1x546x1x1xf32>
    %1601 = stablehlo.reshape %1600 : (tensor<1x546x1x1xf32>) -> tensor<1x6x91x1x1xf32>
    %1602 = stablehlo.transpose %1601, dims = [0, 3, 4, 1, 2] : (tensor<1x6x91x1x1xf32>) -> tensor<1x1x1x6x91xf32>
    %1603 = stablehlo.reshape %1602 : (tensor<1x1x1x6x91xf32>) -> tensor<1x6x91xf32>
    %1604 = stablehlo.concatenate %1478, %1503, %1528, %1553, %1578, %1603, dim = 1 : (tensor<1x2400x91xf32>, tensor<1x600x91xf32>, tensor<1x150x91xf32>, tensor<1x54x91xf32>, tensor<1x24x91xf32>, tensor<1x6x91xf32>) -> tensor<1x3234x91xf32>
    return %1453, %1604, %arg411, %22 : tensor<1x3234x4xf32>, tensor<1x3234x91xf32>, tensor<3234x4xf32>, tensor<1x3x320x320xf32>
  }
}
