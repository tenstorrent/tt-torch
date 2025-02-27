module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1024x3x16x16xbf16>, %arg2: tensor<1024xbf16>, %arg3: tensor<1024xbf16>, %arg4: tensor<1024xbf16>, %arg5: tensor<1024xbf16>, %arg6: tensor<1024xbf16>, %arg7: tensor<1024xbf16>, %arg8: tensor<1024xbf16>, %arg9: tensor<1024xbf16>, %arg10: tensor<1024xbf16>, %arg11: tensor<1024xbf16>, %arg12: tensor<1024xbf16>, %arg13: tensor<1024xbf16>, %arg14: tensor<1024xbf16>, %arg15: tensor<1024xbf16>, %arg16: tensor<1024xbf16>, %arg17: tensor<1024xbf16>, %arg18: tensor<1024xbf16>, %arg19: tensor<1024xbf16>, %arg20: tensor<1024xbf16>, %arg21: tensor<1024xbf16>, %arg22: tensor<1024xbf16>, %arg23: tensor<1024xbf16>, %arg24: tensor<1024xbf16>, %arg25: tensor<1024xbf16>, %arg26: tensor<1024xbf16>, %arg27: tensor<1024xbf16>, %arg28: tensor<1024xbf16>, %arg29: tensor<1024xbf16>, %arg30: tensor<1024xbf16>, %arg31: tensor<1024xbf16>, %arg32: tensor<1024xbf16>, %arg33: tensor<1024xbf16>, %arg34: tensor<1024xbf16>, %arg35: tensor<1024xbf16>, %arg36: tensor<1024xbf16>, %arg37: tensor<1024xbf16>, %arg38: tensor<1024xbf16>, %arg39: tensor<1024xbf16>, %arg40: tensor<1024xbf16>, %arg41: tensor<1024xbf16>, %arg42: tensor<1024xbf16>, %arg43: tensor<1024xbf16>, %arg44: tensor<1024xbf16>, %arg45: tensor<1024xbf16>, %arg46: tensor<1024xbf16>, %arg47: tensor<1024xbf16>, %arg48: tensor<1024xbf16>, %arg49: tensor<1024xbf16>, %arg50: tensor<1024xbf16>, %arg51: tensor<1024xbf16>, %arg52: tensor<1024xbf16>, %arg53: tensor<1024xbf16>, %arg54: tensor<1024xbf16>, %arg55: tensor<1024xbf16>, %arg56: tensor<1024xbf16>, %arg57: tensor<1024xbf16>, %arg58: tensor<1024xbf16>, %arg59: tensor<1024xbf16>, %arg60: tensor<1024xbf16>, %arg61: tensor<1024xbf16>, %arg62: tensor<1024xbf16>, %arg63: tensor<1024xbf16>, %arg64: tensor<1024xbf16>, %arg65: tensor<1024xbf16>, %arg66: tensor<1024xbf16>, %arg67: tensor<1024xbf16>, %arg68: tensor<1024xbf16>, %arg69: tensor<1024xbf16>, %arg70: tensor<1024xbf16>, %arg71: tensor<1024xbf16>, %arg72: tensor<1024xbf16>, %arg73: tensor<1024xbf16>, %arg74: tensor<1024xbf16>, %arg75: tensor<1024xbf16>, %arg76: tensor<1024xbf16>, %arg77: tensor<1024xbf16>, %arg78: tensor<1024xbf16>, %arg79: tensor<1024xbf16>, %arg80: tensor<1024xbf16>, %arg81: tensor<1024xbf16>, %arg82: tensor<1024xbf16>, %arg83: tensor<1024xbf16>, %arg84: tensor<1024xbf16>, %arg85: tensor<1024xbf16>, %arg86: tensor<1024xbf16>, %arg87: tensor<1024xbf16>, %arg88: tensor<1024xbf16>, %arg89: tensor<1024xbf16>, %arg90: tensor<1024xbf16>, %arg91: tensor<1024xbf16>, %arg92: tensor<1024xbf16>, %arg93: tensor<1024xbf16>, %arg94: tensor<1024xbf16>, %arg95: tensor<1024xbf16>, %arg96: tensor<1024xbf16>, %arg97: tensor<1024xbf16>, %arg98: tensor<1024xbf16>, %arg99: tensor<1024xbf16>, %arg100: tensor<1024xbf16>, %arg101: tensor<1024xbf16>, %arg102: tensor<1024xbf16>, %arg103: tensor<1024xbf16>, %arg104: tensor<1024xbf16>, %arg105: tensor<1024xbf16>, %arg106: tensor<1024xbf16>, %arg107: tensor<1024xbf16>, %arg108: tensor<1024xbf16>, %arg109: tensor<1024xbf16>, %arg110: tensor<1024xbf16>, %arg111: tensor<1024xbf16>, %arg112: tensor<1024xbf16>, %arg113: tensor<1024xbf16>, %arg114: tensor<1024xbf16>, %arg115: tensor<1024xbf16>, %arg116: tensor<1024xbf16>, %arg117: tensor<1024xbf16>, %arg118: tensor<1024xbf16>, %arg119: tensor<1024xbf16>, %arg120: tensor<1024xbf16>, %arg121: tensor<1024xbf16>, %arg122: tensor<1024xbf16>, %arg123: tensor<1024xbf16>, %arg124: tensor<1024xbf16>, %arg125: tensor<1024xbf16>, %arg126: tensor<1024xbf16>, %arg127: tensor<1024xbf16>, %arg128: tensor<1024xbf16>, %arg129: tensor<1024xbf16>, %arg130: tensor<1024xbf16>, %arg131: tensor<1024xbf16>, %arg132: tensor<1024xbf16>, %arg133: tensor<1024xbf16>, %arg134: tensor<1024xbf16>, %arg135: tensor<1024xbf16>, %arg136: tensor<1024xbf16>, %arg137: tensor<1024xbf16>, %arg138: tensor<1024xbf16>, %arg139: tensor<1024xbf16>, %arg140: tensor<1024xbf16>, %arg141: tensor<1024xbf16>, %arg142: tensor<1024xbf16>, %arg143: tensor<1024xbf16>, %arg144: tensor<1024xbf16>, %arg145: tensor<1024xbf16>, %arg146: tensor<1024xbf16>, %arg147: tensor<1024xbf16>, %arg148: tensor<1024xbf16>, %arg149: tensor<1x1x1024xbf16>, %arg150: tensor<1024x1024xf32>, %arg151: tensor<1024xf32>, %arg152: tensor<1024x1024xbf16>, %arg153: tensor<1024x1024xf32>, %arg154: tensor<1024xf32>, %arg155: tensor<1x16x197x197xbf16>, %arg156: tensor<1024x1024xf32>, %arg157: tensor<1024xf32>, %arg158: tensor<1024x4096xf32>, %arg159: tensor<4096xf32>, %arg160: tensor<4096x1024xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1024x1024xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<1024x1024xbf16>, %arg165: tensor<1024x1024xf32>, %arg166: tensor<1024xf32>, %arg167: tensor<1x16x197x197xbf16>, %arg168: tensor<1024x1024xf32>, %arg169: tensor<1024xf32>, %arg170: tensor<1024x4096xf32>, %arg171: tensor<4096xf32>, %arg172: tensor<4096x1024xf32>, %arg173: tensor<1024xf32>, %arg174: tensor<1024x1024xf32>, %arg175: tensor<1024xf32>, %arg176: tensor<1024x1024xbf16>, %arg177: tensor<1024x1024xf32>, %arg178: tensor<1024xf32>, %arg179: tensor<1x16x197x197xbf16>, %arg180: tensor<1024x1024xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<1024x4096xf32>, %arg183: tensor<4096xf32>, %arg184: tensor<4096x1024xf32>, %arg185: tensor<1024xf32>, %arg186: tensor<1024x1024xf32>, %arg187: tensor<1024xf32>, %arg188: tensor<1024x1024xbf16>, %arg189: tensor<1024x1024xf32>, %arg190: tensor<1024xf32>, %arg191: tensor<1x16x197x197xbf16>, %arg192: tensor<1024x1024xf32>, %arg193: tensor<1024xf32>, %arg194: tensor<1024x4096xf32>, %arg195: tensor<4096xf32>, %arg196: tensor<4096x1024xf32>, %arg197: tensor<1024xf32>, %arg198: tensor<1024x1024xf32>, %arg199: tensor<1024xf32>, %arg200: tensor<1024x1024xbf16>, %arg201: tensor<1024x1024xf32>, %arg202: tensor<1024xf32>, %arg203: tensor<1x16x197x197xbf16>, %arg204: tensor<1024x1024xf32>, %arg205: tensor<1024xf32>, %arg206: tensor<1024x4096xf32>, %arg207: tensor<4096xf32>, %arg208: tensor<4096x1024xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024x1024xf32>, %arg211: tensor<1024xf32>, %arg212: tensor<1024x1024xbf16>, %arg213: tensor<1024x1024xf32>, %arg214: tensor<1024xf32>, %arg215: tensor<1x16x197x197xbf16>, %arg216: tensor<1024x1024xf32>, %arg217: tensor<1024xf32>, %arg218: tensor<1024x4096xf32>, %arg219: tensor<4096xf32>, %arg220: tensor<4096x1024xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<1024x1024xf32>, %arg223: tensor<1024xf32>, %arg224: tensor<1024x1024xbf16>, %arg225: tensor<1024x1024xf32>, %arg226: tensor<1024xf32>, %arg227: tensor<1x16x197x197xbf16>, %arg228: tensor<1024x1024xf32>, %arg229: tensor<1024xf32>, %arg230: tensor<1024x4096xf32>, %arg231: tensor<4096xf32>, %arg232: tensor<4096x1024xf32>, %arg233: tensor<1024xf32>, %arg234: tensor<1024x1024xf32>, %arg235: tensor<1024xf32>, %arg236: tensor<1024x1024xbf16>, %arg237: tensor<1024x1024xf32>, %arg238: tensor<1024xf32>, %arg239: tensor<1x16x197x197xbf16>, %arg240: tensor<1024x1024xf32>, %arg241: tensor<1024xf32>, %arg242: tensor<1024x4096xf32>, %arg243: tensor<4096xf32>, %arg244: tensor<4096x1024xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024x1024xf32>, %arg247: tensor<1024xf32>, %arg248: tensor<1024x1024xbf16>, %arg249: tensor<1024x1024xf32>, %arg250: tensor<1024xf32>, %arg251: tensor<1x16x197x197xbf16>, %arg252: tensor<1024x1024xf32>, %arg253: tensor<1024xf32>, %arg254: tensor<1024x4096xf32>, %arg255: tensor<4096xf32>, %arg256: tensor<4096x1024xf32>, %arg257: tensor<1024xf32>, %arg258: tensor<1024x1024xf32>, %arg259: tensor<1024xf32>, %arg260: tensor<1024x1024xbf16>, %arg261: tensor<1024x1024xf32>, %arg262: tensor<1024xf32>, %arg263: tensor<1x16x197x197xbf16>, %arg264: tensor<1024x1024xf32>, %arg265: tensor<1024xf32>, %arg266: tensor<1024x4096xf32>, %arg267: tensor<4096xf32>, %arg268: tensor<4096x1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024x1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<1024x1024xbf16>, %arg273: tensor<1024x1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1x16x197x197xbf16>, %arg276: tensor<1024x1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024x4096xf32>, %arg279: tensor<4096xf32>, %arg280: tensor<4096x1024xf32>, %arg281: tensor<1024xf32>, %arg282: tensor<1024x1024xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024x1024xbf16>, %arg285: tensor<1024x1024xf32>, %arg286: tensor<1024xf32>, %arg287: tensor<1x16x197x197xbf16>, %arg288: tensor<1024x1024xf32>, %arg289: tensor<1024xf32>, %arg290: tensor<1024x4096xf32>, %arg291: tensor<4096xf32>, %arg292: tensor<4096x1024xf32>, %arg293: tensor<1024xf32>, %arg294: tensor<1024x1024xf32>, %arg295: tensor<1024xf32>, %arg296: tensor<1024x1024xbf16>, %arg297: tensor<1024x1024xf32>, %arg298: tensor<1024xf32>, %arg299: tensor<1x16x197x197xbf16>, %arg300: tensor<1024x1024xf32>, %arg301: tensor<1024xf32>, %arg302: tensor<1024x4096xf32>, %arg303: tensor<4096xf32>, %arg304: tensor<4096x1024xf32>, %arg305: tensor<1024xf32>, %arg306: tensor<1024x1024xf32>, %arg307: tensor<1024xf32>, %arg308: tensor<1024x1024xbf16>, %arg309: tensor<1024x1024xf32>, %arg310: tensor<1024xf32>, %arg311: tensor<1x16x197x197xbf16>, %arg312: tensor<1024x1024xf32>, %arg313: tensor<1024xf32>, %arg314: tensor<1024x4096xf32>, %arg315: tensor<4096xf32>, %arg316: tensor<4096x1024xf32>, %arg317: tensor<1024xf32>, %arg318: tensor<1024x1024xf32>, %arg319: tensor<1024xf32>, %arg320: tensor<1024x1024xbf16>, %arg321: tensor<1024x1024xf32>, %arg322: tensor<1024xf32>, %arg323: tensor<1x16x197x197xbf16>, %arg324: tensor<1024x1024xf32>, %arg325: tensor<1024xf32>, %arg326: tensor<1024x4096xf32>, %arg327: tensor<4096xf32>, %arg328: tensor<4096x1024xf32>, %arg329: tensor<1024xf32>, %arg330: tensor<1024x1024xf32>, %arg331: tensor<1024xf32>, %arg332: tensor<1024x1024xbf16>, %arg333: tensor<1024x1024xf32>, %arg334: tensor<1024xf32>, %arg335: tensor<1x16x197x197xbf16>, %arg336: tensor<1024x1024xf32>, %arg337: tensor<1024xf32>, %arg338: tensor<1024x4096xf32>, %arg339: tensor<4096xf32>, %arg340: tensor<4096x1024xf32>, %arg341: tensor<1024xf32>, %arg342: tensor<1024x1024xf32>, %arg343: tensor<1024xf32>, %arg344: tensor<1024x1024xbf16>, %arg345: tensor<1024x1024xf32>, %arg346: tensor<1024xf32>, %arg347: tensor<1x16x197x197xbf16>, %arg348: tensor<1024x1024xf32>, %arg349: tensor<1024xf32>, %arg350: tensor<1024x4096xf32>, %arg351: tensor<4096xf32>, %arg352: tensor<4096x1024xf32>, %arg353: tensor<1024xf32>, %arg354: tensor<1024x1024xf32>, %arg355: tensor<1024xf32>, %arg356: tensor<1024x1024xbf16>, %arg357: tensor<1024x1024xf32>, %arg358: tensor<1024xf32>, %arg359: tensor<1x16x197x197xbf16>, %arg360: tensor<1024x1024xf32>, %arg361: tensor<1024xf32>, %arg362: tensor<1024x4096xf32>, %arg363: tensor<4096xf32>, %arg364: tensor<4096x1024xf32>, %arg365: tensor<1024xf32>, %arg366: tensor<1024x1024xf32>, %arg367: tensor<1024xf32>, %arg368: tensor<1024x1024xbf16>, %arg369: tensor<1024x1024xf32>, %arg370: tensor<1024xf32>, %arg371: tensor<1x16x197x197xbf16>, %arg372: tensor<1024x1024xf32>, %arg373: tensor<1024xf32>, %arg374: tensor<1024x4096xf32>, %arg375: tensor<4096xf32>, %arg376: tensor<4096x1024xf32>, %arg377: tensor<1024xf32>, %arg378: tensor<1024x1024xf32>, %arg379: tensor<1024xf32>, %arg380: tensor<1024x1024xbf16>, %arg381: tensor<1024x1024xf32>, %arg382: tensor<1024xf32>, %arg383: tensor<1x16x197x197xbf16>, %arg384: tensor<1024x1024xf32>, %arg385: tensor<1024xf32>, %arg386: tensor<1024x4096xf32>, %arg387: tensor<4096xf32>, %arg388: tensor<4096x1024xf32>, %arg389: tensor<1024xf32>, %arg390: tensor<1024x1024xf32>, %arg391: tensor<1024xf32>, %arg392: tensor<1024x1024xbf16>, %arg393: tensor<1024x1024xf32>, %arg394: tensor<1024xf32>, %arg395: tensor<1x16x197x197xbf16>, %arg396: tensor<1024x1024xf32>, %arg397: tensor<1024xf32>, %arg398: tensor<1024x4096xf32>, %arg399: tensor<4096xf32>, %arg400: tensor<4096x1024xf32>, %arg401: tensor<1024xf32>, %arg402: tensor<1024x1024xf32>, %arg403: tensor<1024xf32>, %arg404: tensor<1024x1024xbf16>, %arg405: tensor<1024x1024xf32>, %arg406: tensor<1024xf32>, %arg407: tensor<1x16x197x197xbf16>, %arg408: tensor<1024x1024xf32>, %arg409: tensor<1024xf32>, %arg410: tensor<1024x4096xf32>, %arg411: tensor<4096xf32>, %arg412: tensor<4096x1024xf32>, %arg413: tensor<1024xf32>, %arg414: tensor<1024x1024xf32>, %arg415: tensor<1024xf32>, %arg416: tensor<1024x1024xbf16>, %arg417: tensor<1024x1024xf32>, %arg418: tensor<1024xf32>, %arg419: tensor<1x16x197x197xbf16>, %arg420: tensor<1024x1024xf32>, %arg421: tensor<1024xf32>, %arg422: tensor<1024x4096xf32>, %arg423: tensor<4096xf32>, %arg424: tensor<4096x1024xf32>, %arg425: tensor<1024xf32>, %arg426: tensor<1024x1024xf32>, %arg427: tensor<1024xf32>, %arg428: tensor<1024x1024xbf16>, %arg429: tensor<1024x1024xf32>, %arg430: tensor<1024xf32>, %arg431: tensor<1x16x197x197xbf16>, %arg432: tensor<1024x1024xf32>, %arg433: tensor<1024xf32>, %arg434: tensor<1024x4096xf32>, %arg435: tensor<4096xf32>, %arg436: tensor<4096x1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<1024x1000xf32>, %arg439: tensor<1000xf32>) -> tensor<1x1000xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<1x197x4096xbf16>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<1x197x4096xbf16>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<1x197x4096xbf16>
    %cst_5 = stablehlo.constant dense<-4.000000e+00> : tensor<1x197x4096xf32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<1x197x4096xf32>
    %cst_7 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x197x4096xf32>
    %cst_8 = stablehlo.constant dense<2.77068146E-8> : tensor<1x197x4096xf32>
    %cst_9 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x197x4096xf32>
    %cst_10 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x197x4096xf32>
    %cst_11 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x197x4096xf32>
    %cst_12 = stablehlo.constant dense<-2.954600e-03> : tensor<1x197x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.0160960332> : tensor<1x197x4096xf32>
    %cst_14 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x197x4096xf32>
    %cst_15 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x197x4096xf32>
    %cst_16 = stablehlo.constant dense<-0.00168282702> : tensor<1x197x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.00737332925> : tensor<1x197x4096xf32>
    %cst_18 = stablehlo.constant dense<-0.0142647391> : tensor<1x197x4096xf32>
    %cst_19 = stablehlo.constant dense<-1.000000e+00> : tensor<1x197x4096xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x197x4096xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_22 = arith.constant dense<1024> : tensor<1xi64>
    %cst_23 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
    %cst_24 = arith.constant dense<1> : tensor<1xi64>
    %cst_25 = arith.constant dense<8.000000e+00> : tensor<1xf64>
    %cst_26 = arith.constant dense<196> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [16, 16], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xbf16>, tensor<1024x3x16x16xbf16>) -> tensor<1x1024x14x14xbf16>
    %1 = stablehlo.reshape %arg2 : (tensor<1024xbf16>) -> tensor<1024x1x1xbf16>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<1024x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %4 = stablehlo.add %2, %3 : tensor<1x1024x14x14xbf16>
    %5 = stablehlo.reshape %4 : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x196xbf16>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1] : (tensor<1x1024x196xbf16>) -> tensor<1x196x1024xbf16>
    %7 = stablehlo.concatenate %arg149, %6, dim = 1 : (tensor<1x1x1024xbf16>, tensor<1x196x1024xbf16>) -> tensor<1x197x1024xbf16>
    %8 = stablehlo.convert %7 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %9 = stablehlo.convert %8 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %11 = stablehlo.reshape %10 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %12 = stablehlo.convert %cst_22 : (tensor<1xi64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %15 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<1x197x1xf64>
    %16 = stablehlo.divide %14, %15 : tensor<1x197x1xf64>
    %17 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %19 = stablehlo.subtract %17, %18 : tensor<1x197x1024xf64>
    %20 = stablehlo.multiply %19, %19 : tensor<1x197x1024xf64>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %22 = stablehlo.reshape %21 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %24 = stablehlo.divide %23, %15 : tensor<1x197x1xf64>
    %25 = stablehlo.convert %24 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %26 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
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
    %39 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %40 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %41 = stablehlo.subtract %39, %40 : tensor<1x197x1024xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %43 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<1x197x1024xf32>
    %45 = stablehlo.convert %arg3 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %48 = stablehlo.multiply %46, %47 : tensor<1x197x1024xf32>
    %49 = stablehlo.convert %arg4 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %51 = stablehlo.broadcast_in_dim %49, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x197x1024xf32>
    %53 = stablehlo.convert %52 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %54 = stablehlo.reshape %53 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %55 = stablehlo.convert %54 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %56 = stablehlo.dot_general %55, %arg150, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %57 = stablehlo.convert %cst_24 : (tensor<1xi64>) -> tensor<1xf32>
    %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
    %59 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %60 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<197x1024xf32>
    %61 = stablehlo.multiply %59, %60 : tensor<197x1024xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %63 = stablehlo.broadcast_in_dim %arg151, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %64 = stablehlo.add %62, %63 : tensor<197x1024xf32>
    %65 = stablehlo.convert %64 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %66 = stablehlo.reshape %65 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %67 = stablehlo.dot_general %54, %arg152, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %68 = stablehlo.reshape %67 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %69 = stablehlo.reshape %68 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %70 = stablehlo.transpose %69, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %71 = stablehlo.dot_general %55, %arg153, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %73 = stablehlo.multiply %72, %60 : tensor<197x1024xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %75 = stablehlo.broadcast_in_dim %arg154, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<197x1024xf32>
    %77 = stablehlo.convert %76 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %78 = stablehlo.reshape %77 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %79 = stablehlo.reshape %78 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %80 = stablehlo.transpose %79, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %81 = stablehlo.reshape %66 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %82 = stablehlo.transpose %81, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %83 = stablehlo.transpose %70, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %84 = stablehlo.reshape %82 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %85 = stablehlo.reshape %83 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %87 = stablehlo.dot_general %84, %86, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %88 = stablehlo.reshape %87 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %89 = stablehlo.convert %cst_25 : (tensor<1xf64>) -> tensor<1xbf16>
    %90 = stablehlo.reshape %89 : (tensor<1xbf16>) -> tensor<bf16>
    %91 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %92 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<bf16>) -> tensor<1x16x197x197xbf16>
    %93 = stablehlo.divide %91, %92 : tensor<1x16x197x197xbf16>
    %94 = stablehlo.add %93, %arg155 : tensor<1x16x197x197xbf16>
    %95 = stablehlo.convert %94 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %96 = stablehlo.reduce(%95 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %97 = stablehlo.reshape %96 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %98 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %99 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %100 = stablehlo.subtract %98, %99 : tensor<1x16x197x197xf32>
    %101 = stablehlo.exponential %100 : tensor<1x16x197x197xf32>
    %102 = stablehlo.reduce(%101 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %103 = stablehlo.reshape %102 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %104 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %105 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %106 = stablehlo.divide %104, %105 : tensor<1x16x197x197xf32>
    %107 = stablehlo.convert %106 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %108 = stablehlo.reshape %107 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %109 = stablehlo.reshape %80 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %111 = stablehlo.dot_general %108, %110, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %112 = stablehlo.reshape %111 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %113 = stablehlo.transpose %112, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %114 = stablehlo.reshape %113 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %115 = stablehlo.reshape %114 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %116 = stablehlo.convert %115 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %117 = stablehlo.dot_general %116, %arg156, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %119 = stablehlo.multiply %118, %60 : tensor<197x1024xf32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %121 = stablehlo.broadcast_in_dim %arg157, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %122 = stablehlo.add %120, %121 : tensor<197x1024xf32>
    %123 = stablehlo.convert %122 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %124 = stablehlo.reshape %123 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %125 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %126 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %127 = stablehlo.multiply %125, %126 : tensor<1x197x1024xbf16>
    %128 = stablehlo.add %127, %7 : tensor<1x197x1024xbf16>
    %129 = stablehlo.convert %128 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %130 = stablehlo.convert %129 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %131 = stablehlo.reduce(%130 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %132 = stablehlo.reshape %131 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %134 = stablehlo.divide %133, %15 : tensor<1x197x1xf64>
    %135 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %136 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %137 = stablehlo.subtract %135, %136 : tensor<1x197x1024xf64>
    %138 = stablehlo.multiply %137, %137 : tensor<1x197x1024xf64>
    %139 = stablehlo.reduce(%138 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %140 = stablehlo.reshape %139 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %141 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %142 = stablehlo.divide %141, %15 : tensor<1x197x1xf64>
    %143 = stablehlo.convert %142 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %144 = stablehlo.reduce(%129 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %145 = stablehlo.reshape %144 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %147 = stablehlo.divide %146, %31 : tensor<1x197x1xf32>
    %148 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %149 = stablehlo.add %148, %36 : tensor<1x197x1xf32>
    %150 = stablehlo.rsqrt %149 : tensor<1x197x1xf32>
    %151 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %152 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %153 = stablehlo.subtract %151, %152 : tensor<1x197x1024xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %155 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %156 = stablehlo.multiply %154, %155 : tensor<1x197x1024xf32>
    %157 = stablehlo.convert %arg6 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %158 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %159 = stablehlo.broadcast_in_dim %157, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %160 = stablehlo.multiply %158, %159 : tensor<1x197x1024xf32>
    %161 = stablehlo.convert %arg7 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %162 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %163 = stablehlo.broadcast_in_dim %161, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %164 = stablehlo.add %162, %163 : tensor<1x197x1024xf32>
    %165 = stablehlo.convert %164 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %166 = stablehlo.reshape %165 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %167 = stablehlo.convert %166 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %168 = stablehlo.dot_general %167, %arg158, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %170 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<197x4096xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<197x4096xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %173 = stablehlo.broadcast_in_dim %arg159, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %174 = stablehlo.add %172, %173 : tensor<197x4096xf32>
    %175 = stablehlo.convert %174 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %176 = stablehlo.reshape %175 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %177 = stablehlo.multiply %176, %cst_4 : tensor<1x197x4096xbf16>
    %178 = stablehlo.rsqrt %cst_3 : tensor<1x197x4096xbf16>
    %179 = stablehlo.multiply %176, %178 : tensor<1x197x4096xbf16>
    %180 = stablehlo.convert %179 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %181 = stablehlo.clamp %cst_5, %180, %cst_6 : tensor<1x197x4096xf32>
    %182 = stablehlo.multiply %181, %181 : tensor<1x197x4096xf32>
    %183 = stablehlo.multiply %cst_7, %182 : tensor<1x197x4096xf32>
    %184 = stablehlo.add %183, %cst_8 : tensor<1x197x4096xf32>
    %185 = stablehlo.multiply %184, %182 : tensor<1x197x4096xf32>
    %186 = stablehlo.add %185, %cst_9 : tensor<1x197x4096xf32>
    %187 = stablehlo.multiply %186, %182 : tensor<1x197x4096xf32>
    %188 = stablehlo.add %187, %cst_10 : tensor<1x197x4096xf32>
    %189 = stablehlo.multiply %188, %182 : tensor<1x197x4096xf32>
    %190 = stablehlo.add %189, %cst_11 : tensor<1x197x4096xf32>
    %191 = stablehlo.multiply %190, %182 : tensor<1x197x4096xf32>
    %192 = stablehlo.add %191, %cst_12 : tensor<1x197x4096xf32>
    %193 = stablehlo.multiply %192, %182 : tensor<1x197x4096xf32>
    %194 = stablehlo.add %193, %cst_13 : tensor<1x197x4096xf32>
    %195 = stablehlo.multiply %cst_14, %182 : tensor<1x197x4096xf32>
    %196 = stablehlo.add %195, %cst_15 : tensor<1x197x4096xf32>
    %197 = stablehlo.multiply %196, %182 : tensor<1x197x4096xf32>
    %198 = stablehlo.add %197, %cst_16 : tensor<1x197x4096xf32>
    %199 = stablehlo.multiply %198, %182 : tensor<1x197x4096xf32>
    %200 = stablehlo.add %199, %cst_17 : tensor<1x197x4096xf32>
    %201 = stablehlo.multiply %200, %182 : tensor<1x197x4096xf32>
    %202 = stablehlo.add %201, %cst_18 : tensor<1x197x4096xf32>
    %203 = stablehlo.multiply %181, %194 : tensor<1x197x4096xf32>
    %204 = stablehlo.divide %203, %202 : tensor<1x197x4096xf32>
    %205 = stablehlo.clamp %cst_19, %204, %cst_20 : tensor<1x197x4096xf32>
    %206 = stablehlo.convert %205 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %207 = stablehlo.add %206, %cst_2 : tensor<1x197x4096xbf16>
    %208 = stablehlo.multiply %207, %177 : tensor<1x197x4096xbf16>
    %209 = stablehlo.reshape %208 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %210 = stablehlo.convert %209 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %211 = stablehlo.dot_general %210, %arg160, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %213 = stablehlo.multiply %212, %60 : tensor<197x1024xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %215 = stablehlo.broadcast_in_dim %arg161, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %216 = stablehlo.add %214, %215 : tensor<197x1024xf32>
    %217 = stablehlo.convert %216 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %218 = stablehlo.reshape %217 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %219 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %220 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %221 = stablehlo.multiply %219, %220 : tensor<1x197x1024xbf16>
    %222 = stablehlo.add %221, %128 : tensor<1x197x1024xbf16>
    %223 = stablehlo.convert %222 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %224 = stablehlo.convert %223 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %225 = stablehlo.reduce(%224 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %226 = stablehlo.reshape %225 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %228 = stablehlo.divide %227, %15 : tensor<1x197x1xf64>
    %229 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %230 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %231 = stablehlo.subtract %229, %230 : tensor<1x197x1024xf64>
    %232 = stablehlo.multiply %231, %231 : tensor<1x197x1024xf64>
    %233 = stablehlo.reduce(%232 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %234 = stablehlo.reshape %233 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %236 = stablehlo.divide %235, %15 : tensor<1x197x1xf64>
    %237 = stablehlo.convert %236 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %238 = stablehlo.reduce(%223 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %239 = stablehlo.reshape %238 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %241 = stablehlo.divide %240, %31 : tensor<1x197x1xf32>
    %242 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %243 = stablehlo.add %242, %36 : tensor<1x197x1xf32>
    %244 = stablehlo.rsqrt %243 : tensor<1x197x1xf32>
    %245 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %246 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %247 = stablehlo.subtract %245, %246 : tensor<1x197x1024xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %249 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %250 = stablehlo.multiply %248, %249 : tensor<1x197x1024xf32>
    %251 = stablehlo.convert %arg9 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %252 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %253 = stablehlo.broadcast_in_dim %251, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %254 = stablehlo.multiply %252, %253 : tensor<1x197x1024xf32>
    %255 = stablehlo.convert %arg10 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %256 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %257 = stablehlo.broadcast_in_dim %255, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %258 = stablehlo.add %256, %257 : tensor<1x197x1024xf32>
    %259 = stablehlo.convert %258 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %260 = stablehlo.reshape %259 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %261 = stablehlo.convert %260 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %262 = stablehlo.dot_general %261, %arg162, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %263 = stablehlo.broadcast_in_dim %262, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %264 = stablehlo.multiply %263, %60 : tensor<197x1024xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %266 = stablehlo.broadcast_in_dim %arg163, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %267 = stablehlo.add %265, %266 : tensor<197x1024xf32>
    %268 = stablehlo.convert %267 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %269 = stablehlo.reshape %268 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %270 = stablehlo.dot_general %260, %arg164, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %271 = stablehlo.reshape %270 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %272 = stablehlo.reshape %271 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %273 = stablehlo.transpose %272, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %274 = stablehlo.dot_general %261, %arg165, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %276 = stablehlo.multiply %275, %60 : tensor<197x1024xf32>
    %277 = stablehlo.broadcast_in_dim %276, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %278 = stablehlo.broadcast_in_dim %arg166, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %279 = stablehlo.add %277, %278 : tensor<197x1024xf32>
    %280 = stablehlo.convert %279 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %281 = stablehlo.reshape %280 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %282 = stablehlo.reshape %281 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %283 = stablehlo.transpose %282, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %284 = stablehlo.reshape %269 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %285 = stablehlo.transpose %284, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %286 = stablehlo.transpose %273, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %287 = stablehlo.reshape %285 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %288 = stablehlo.reshape %286 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %290 = stablehlo.dot_general %287, %289, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %291 = stablehlo.reshape %290 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %292 = stablehlo.broadcast_in_dim %291, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %293 = stablehlo.divide %292, %92 : tensor<1x16x197x197xbf16>
    %294 = stablehlo.add %293, %arg167 : tensor<1x16x197x197xbf16>
    %295 = stablehlo.convert %294 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %296 = stablehlo.reduce(%295 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %297 = stablehlo.reshape %296 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %298 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %299 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %300 = stablehlo.subtract %298, %299 : tensor<1x16x197x197xf32>
    %301 = stablehlo.exponential %300 : tensor<1x16x197x197xf32>
    %302 = stablehlo.reduce(%301 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %303 = stablehlo.reshape %302 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %304 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %305 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %306 = stablehlo.divide %304, %305 : tensor<1x16x197x197xf32>
    %307 = stablehlo.convert %306 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %308 = stablehlo.reshape %307 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %309 = stablehlo.reshape %283 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %311 = stablehlo.dot_general %308, %310, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %312 = stablehlo.reshape %311 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %313 = stablehlo.transpose %312, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %314 = stablehlo.reshape %313 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %315 = stablehlo.reshape %314 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %316 = stablehlo.convert %315 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %317 = stablehlo.dot_general %316, %arg168, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %319 = stablehlo.multiply %318, %60 : tensor<197x1024xf32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %321 = stablehlo.broadcast_in_dim %arg169, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %322 = stablehlo.add %320, %321 : tensor<197x1024xf32>
    %323 = stablehlo.convert %322 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %324 = stablehlo.reshape %323 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %325 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %326 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %327 = stablehlo.multiply %325, %326 : tensor<1x197x1024xbf16>
    %328 = stablehlo.add %327, %222 : tensor<1x197x1024xbf16>
    %329 = stablehlo.convert %328 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %330 = stablehlo.convert %329 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %331 = stablehlo.reduce(%330 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %332 = stablehlo.reshape %331 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %334 = stablehlo.divide %333, %15 : tensor<1x197x1xf64>
    %335 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %336 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %337 = stablehlo.subtract %335, %336 : tensor<1x197x1024xf64>
    %338 = stablehlo.multiply %337, %337 : tensor<1x197x1024xf64>
    %339 = stablehlo.reduce(%338 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %340 = stablehlo.reshape %339 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %342 = stablehlo.divide %341, %15 : tensor<1x197x1xf64>
    %343 = stablehlo.convert %342 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %344 = stablehlo.reduce(%329 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %345 = stablehlo.reshape %344 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %347 = stablehlo.divide %346, %31 : tensor<1x197x1xf32>
    %348 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %349 = stablehlo.add %348, %36 : tensor<1x197x1xf32>
    %350 = stablehlo.rsqrt %349 : tensor<1x197x1xf32>
    %351 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %352 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %353 = stablehlo.subtract %351, %352 : tensor<1x197x1024xf32>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %355 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %356 = stablehlo.multiply %354, %355 : tensor<1x197x1024xf32>
    %357 = stablehlo.convert %arg12 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %358 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %359 = stablehlo.broadcast_in_dim %357, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<1x197x1024xf32>
    %361 = stablehlo.convert %arg13 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %362 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %363 = stablehlo.broadcast_in_dim %361, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %364 = stablehlo.add %362, %363 : tensor<1x197x1024xf32>
    %365 = stablehlo.convert %364 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %366 = stablehlo.reshape %365 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %367 = stablehlo.convert %366 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %368 = stablehlo.dot_general %367, %arg170, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %370 = stablehlo.multiply %369, %170 : tensor<197x4096xf32>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %372 = stablehlo.broadcast_in_dim %arg171, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %373 = stablehlo.add %371, %372 : tensor<197x4096xf32>
    %374 = stablehlo.convert %373 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %375 = stablehlo.reshape %374 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %376 = stablehlo.multiply %375, %cst_4 : tensor<1x197x4096xbf16>
    %377 = stablehlo.multiply %375, %178 : tensor<1x197x4096xbf16>
    %378 = stablehlo.convert %377 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %379 = stablehlo.clamp %cst_5, %378, %cst_6 : tensor<1x197x4096xf32>
    %380 = stablehlo.multiply %379, %379 : tensor<1x197x4096xf32>
    %381 = stablehlo.multiply %cst_7, %380 : tensor<1x197x4096xf32>
    %382 = stablehlo.add %381, %cst_8 : tensor<1x197x4096xf32>
    %383 = stablehlo.multiply %382, %380 : tensor<1x197x4096xf32>
    %384 = stablehlo.add %383, %cst_9 : tensor<1x197x4096xf32>
    %385 = stablehlo.multiply %384, %380 : tensor<1x197x4096xf32>
    %386 = stablehlo.add %385, %cst_10 : tensor<1x197x4096xf32>
    %387 = stablehlo.multiply %386, %380 : tensor<1x197x4096xf32>
    %388 = stablehlo.add %387, %cst_11 : tensor<1x197x4096xf32>
    %389 = stablehlo.multiply %388, %380 : tensor<1x197x4096xf32>
    %390 = stablehlo.add %389, %cst_12 : tensor<1x197x4096xf32>
    %391 = stablehlo.multiply %390, %380 : tensor<1x197x4096xf32>
    %392 = stablehlo.add %391, %cst_13 : tensor<1x197x4096xf32>
    %393 = stablehlo.multiply %cst_14, %380 : tensor<1x197x4096xf32>
    %394 = stablehlo.add %393, %cst_15 : tensor<1x197x4096xf32>
    %395 = stablehlo.multiply %394, %380 : tensor<1x197x4096xf32>
    %396 = stablehlo.add %395, %cst_16 : tensor<1x197x4096xf32>
    %397 = stablehlo.multiply %396, %380 : tensor<1x197x4096xf32>
    %398 = stablehlo.add %397, %cst_17 : tensor<1x197x4096xf32>
    %399 = stablehlo.multiply %398, %380 : tensor<1x197x4096xf32>
    %400 = stablehlo.add %399, %cst_18 : tensor<1x197x4096xf32>
    %401 = stablehlo.multiply %379, %392 : tensor<1x197x4096xf32>
    %402 = stablehlo.divide %401, %400 : tensor<1x197x4096xf32>
    %403 = stablehlo.clamp %cst_19, %402, %cst_20 : tensor<1x197x4096xf32>
    %404 = stablehlo.convert %403 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %405 = stablehlo.add %404, %cst_2 : tensor<1x197x4096xbf16>
    %406 = stablehlo.multiply %405, %376 : tensor<1x197x4096xbf16>
    %407 = stablehlo.reshape %406 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %408 = stablehlo.convert %407 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %409 = stablehlo.dot_general %408, %arg172, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %410 = stablehlo.broadcast_in_dim %409, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %411 = stablehlo.multiply %410, %60 : tensor<197x1024xf32>
    %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %413 = stablehlo.broadcast_in_dim %arg173, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %414 = stablehlo.add %412, %413 : tensor<197x1024xf32>
    %415 = stablehlo.convert %414 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %416 = stablehlo.reshape %415 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %417 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %418 = stablehlo.broadcast_in_dim %416, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %419 = stablehlo.multiply %417, %418 : tensor<1x197x1024xbf16>
    %420 = stablehlo.add %419, %328 : tensor<1x197x1024xbf16>
    %421 = stablehlo.convert %420 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %422 = stablehlo.convert %421 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %423 = stablehlo.reduce(%422 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %424 = stablehlo.reshape %423 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %426 = stablehlo.divide %425, %15 : tensor<1x197x1xf64>
    %427 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %428 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %429 = stablehlo.subtract %427, %428 : tensor<1x197x1024xf64>
    %430 = stablehlo.multiply %429, %429 : tensor<1x197x1024xf64>
    %431 = stablehlo.reduce(%430 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %432 = stablehlo.reshape %431 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %433 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %434 = stablehlo.divide %433, %15 : tensor<1x197x1xf64>
    %435 = stablehlo.convert %434 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %436 = stablehlo.reduce(%421 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %437 = stablehlo.reshape %436 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %439 = stablehlo.divide %438, %31 : tensor<1x197x1xf32>
    %440 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %441 = stablehlo.add %440, %36 : tensor<1x197x1xf32>
    %442 = stablehlo.rsqrt %441 : tensor<1x197x1xf32>
    %443 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %444 = stablehlo.broadcast_in_dim %439, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %445 = stablehlo.subtract %443, %444 : tensor<1x197x1024xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %447 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %448 = stablehlo.multiply %446, %447 : tensor<1x197x1024xf32>
    %449 = stablehlo.convert %arg15 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %450 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %451 = stablehlo.broadcast_in_dim %449, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %452 = stablehlo.multiply %450, %451 : tensor<1x197x1024xf32>
    %453 = stablehlo.convert %arg16 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %454 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %455 = stablehlo.broadcast_in_dim %453, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %456 = stablehlo.add %454, %455 : tensor<1x197x1024xf32>
    %457 = stablehlo.convert %456 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %458 = stablehlo.reshape %457 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %459 = stablehlo.convert %458 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %460 = stablehlo.dot_general %459, %arg174, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %462 = stablehlo.multiply %461, %60 : tensor<197x1024xf32>
    %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %464 = stablehlo.broadcast_in_dim %arg175, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %465 = stablehlo.add %463, %464 : tensor<197x1024xf32>
    %466 = stablehlo.convert %465 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %467 = stablehlo.reshape %466 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %468 = stablehlo.dot_general %458, %arg176, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %469 = stablehlo.reshape %468 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %470 = stablehlo.reshape %469 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %471 = stablehlo.transpose %470, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %472 = stablehlo.dot_general %459, %arg177, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %473 = stablehlo.broadcast_in_dim %472, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %474 = stablehlo.multiply %473, %60 : tensor<197x1024xf32>
    %475 = stablehlo.broadcast_in_dim %474, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %476 = stablehlo.broadcast_in_dim %arg178, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %477 = stablehlo.add %475, %476 : tensor<197x1024xf32>
    %478 = stablehlo.convert %477 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %479 = stablehlo.reshape %478 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %480 = stablehlo.reshape %479 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %481 = stablehlo.transpose %480, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %482 = stablehlo.reshape %467 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %483 = stablehlo.transpose %482, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %484 = stablehlo.transpose %471, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %485 = stablehlo.reshape %483 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %486 = stablehlo.reshape %484 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %488 = stablehlo.dot_general %485, %487, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %489 = stablehlo.reshape %488 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %491 = stablehlo.divide %490, %92 : tensor<1x16x197x197xbf16>
    %492 = stablehlo.add %491, %arg179 : tensor<1x16x197x197xbf16>
    %493 = stablehlo.convert %492 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %494 = stablehlo.reduce(%493 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %495 = stablehlo.reshape %494 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %496 = stablehlo.broadcast_in_dim %493, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %497 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %498 = stablehlo.subtract %496, %497 : tensor<1x16x197x197xf32>
    %499 = stablehlo.exponential %498 : tensor<1x16x197x197xf32>
    %500 = stablehlo.reduce(%499 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %501 = stablehlo.reshape %500 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %502 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %503 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %504 = stablehlo.divide %502, %503 : tensor<1x16x197x197xf32>
    %505 = stablehlo.convert %504 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %506 = stablehlo.reshape %505 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %507 = stablehlo.reshape %481 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %508 = stablehlo.broadcast_in_dim %507, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %509 = stablehlo.dot_general %506, %508, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %510 = stablehlo.reshape %509 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %511 = stablehlo.transpose %510, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %512 = stablehlo.reshape %511 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %513 = stablehlo.reshape %512 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %514 = stablehlo.convert %513 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %515 = stablehlo.dot_general %514, %arg180, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %517 = stablehlo.multiply %516, %60 : tensor<197x1024xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %519 = stablehlo.broadcast_in_dim %arg181, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %520 = stablehlo.add %518, %519 : tensor<197x1024xf32>
    %521 = stablehlo.convert %520 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %522 = stablehlo.reshape %521 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %523 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %524 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %525 = stablehlo.multiply %523, %524 : tensor<1x197x1024xbf16>
    %526 = stablehlo.add %525, %420 : tensor<1x197x1024xbf16>
    %527 = stablehlo.convert %526 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %528 = stablehlo.convert %527 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %529 = stablehlo.reduce(%528 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %530 = stablehlo.reshape %529 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %532 = stablehlo.divide %531, %15 : tensor<1x197x1xf64>
    %533 = stablehlo.broadcast_in_dim %528, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %534 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %535 = stablehlo.subtract %533, %534 : tensor<1x197x1024xf64>
    %536 = stablehlo.multiply %535, %535 : tensor<1x197x1024xf64>
    %537 = stablehlo.reduce(%536 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %538 = stablehlo.reshape %537 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %540 = stablehlo.divide %539, %15 : tensor<1x197x1xf64>
    %541 = stablehlo.convert %540 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %542 = stablehlo.reduce(%527 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %543 = stablehlo.reshape %542 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %545 = stablehlo.divide %544, %31 : tensor<1x197x1xf32>
    %546 = stablehlo.broadcast_in_dim %541, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %547 = stablehlo.add %546, %36 : tensor<1x197x1xf32>
    %548 = stablehlo.rsqrt %547 : tensor<1x197x1xf32>
    %549 = stablehlo.broadcast_in_dim %527, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %550 = stablehlo.broadcast_in_dim %545, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %551 = stablehlo.subtract %549, %550 : tensor<1x197x1024xf32>
    %552 = stablehlo.broadcast_in_dim %551, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %553 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %554 = stablehlo.multiply %552, %553 : tensor<1x197x1024xf32>
    %555 = stablehlo.convert %arg18 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %556 = stablehlo.broadcast_in_dim %554, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %557 = stablehlo.broadcast_in_dim %555, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %558 = stablehlo.multiply %556, %557 : tensor<1x197x1024xf32>
    %559 = stablehlo.convert %arg19 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %560 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %561 = stablehlo.broadcast_in_dim %559, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %562 = stablehlo.add %560, %561 : tensor<1x197x1024xf32>
    %563 = stablehlo.convert %562 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %564 = stablehlo.reshape %563 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %565 = stablehlo.convert %564 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %566 = stablehlo.dot_general %565, %arg182, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %568 = stablehlo.multiply %567, %170 : tensor<197x4096xf32>
    %569 = stablehlo.broadcast_in_dim %568, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %570 = stablehlo.broadcast_in_dim %arg183, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %571 = stablehlo.add %569, %570 : tensor<197x4096xf32>
    %572 = stablehlo.convert %571 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %573 = stablehlo.reshape %572 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %574 = stablehlo.multiply %573, %cst_4 : tensor<1x197x4096xbf16>
    %575 = stablehlo.multiply %573, %178 : tensor<1x197x4096xbf16>
    %576 = stablehlo.convert %575 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %577 = stablehlo.clamp %cst_5, %576, %cst_6 : tensor<1x197x4096xf32>
    %578 = stablehlo.multiply %577, %577 : tensor<1x197x4096xf32>
    %579 = stablehlo.multiply %cst_7, %578 : tensor<1x197x4096xf32>
    %580 = stablehlo.add %579, %cst_8 : tensor<1x197x4096xf32>
    %581 = stablehlo.multiply %580, %578 : tensor<1x197x4096xf32>
    %582 = stablehlo.add %581, %cst_9 : tensor<1x197x4096xf32>
    %583 = stablehlo.multiply %582, %578 : tensor<1x197x4096xf32>
    %584 = stablehlo.add %583, %cst_10 : tensor<1x197x4096xf32>
    %585 = stablehlo.multiply %584, %578 : tensor<1x197x4096xf32>
    %586 = stablehlo.add %585, %cst_11 : tensor<1x197x4096xf32>
    %587 = stablehlo.multiply %586, %578 : tensor<1x197x4096xf32>
    %588 = stablehlo.add %587, %cst_12 : tensor<1x197x4096xf32>
    %589 = stablehlo.multiply %588, %578 : tensor<1x197x4096xf32>
    %590 = stablehlo.add %589, %cst_13 : tensor<1x197x4096xf32>
    %591 = stablehlo.multiply %cst_14, %578 : tensor<1x197x4096xf32>
    %592 = stablehlo.add %591, %cst_15 : tensor<1x197x4096xf32>
    %593 = stablehlo.multiply %592, %578 : tensor<1x197x4096xf32>
    %594 = stablehlo.add %593, %cst_16 : tensor<1x197x4096xf32>
    %595 = stablehlo.multiply %594, %578 : tensor<1x197x4096xf32>
    %596 = stablehlo.add %595, %cst_17 : tensor<1x197x4096xf32>
    %597 = stablehlo.multiply %596, %578 : tensor<1x197x4096xf32>
    %598 = stablehlo.add %597, %cst_18 : tensor<1x197x4096xf32>
    %599 = stablehlo.multiply %577, %590 : tensor<1x197x4096xf32>
    %600 = stablehlo.divide %599, %598 : tensor<1x197x4096xf32>
    %601 = stablehlo.clamp %cst_19, %600, %cst_20 : tensor<1x197x4096xf32>
    %602 = stablehlo.convert %601 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %603 = stablehlo.add %602, %cst_2 : tensor<1x197x4096xbf16>
    %604 = stablehlo.multiply %603, %574 : tensor<1x197x4096xbf16>
    %605 = stablehlo.reshape %604 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %606 = stablehlo.convert %605 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %607 = stablehlo.dot_general %606, %arg184, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %609 = stablehlo.multiply %608, %60 : tensor<197x1024xf32>
    %610 = stablehlo.broadcast_in_dim %609, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %611 = stablehlo.broadcast_in_dim %arg185, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %612 = stablehlo.add %610, %611 : tensor<197x1024xf32>
    %613 = stablehlo.convert %612 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %614 = stablehlo.reshape %613 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %615 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %616 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %617 = stablehlo.multiply %615, %616 : tensor<1x197x1024xbf16>
    %618 = stablehlo.add %617, %526 : tensor<1x197x1024xbf16>
    %619 = stablehlo.convert %618 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %620 = stablehlo.convert %619 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %621 = stablehlo.reduce(%620 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %622 = stablehlo.reshape %621 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %623 = stablehlo.broadcast_in_dim %622, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %624 = stablehlo.divide %623, %15 : tensor<1x197x1xf64>
    %625 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %626 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %627 = stablehlo.subtract %625, %626 : tensor<1x197x1024xf64>
    %628 = stablehlo.multiply %627, %627 : tensor<1x197x1024xf64>
    %629 = stablehlo.reduce(%628 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %630 = stablehlo.reshape %629 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %632 = stablehlo.divide %631, %15 : tensor<1x197x1xf64>
    %633 = stablehlo.convert %632 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %634 = stablehlo.reduce(%619 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %635 = stablehlo.reshape %634 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %636 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %637 = stablehlo.divide %636, %31 : tensor<1x197x1xf32>
    %638 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %639 = stablehlo.add %638, %36 : tensor<1x197x1xf32>
    %640 = stablehlo.rsqrt %639 : tensor<1x197x1xf32>
    %641 = stablehlo.broadcast_in_dim %619, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %642 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %643 = stablehlo.subtract %641, %642 : tensor<1x197x1024xf32>
    %644 = stablehlo.broadcast_in_dim %643, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %645 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %646 = stablehlo.multiply %644, %645 : tensor<1x197x1024xf32>
    %647 = stablehlo.convert %arg21 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %648 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %649 = stablehlo.broadcast_in_dim %647, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %650 = stablehlo.multiply %648, %649 : tensor<1x197x1024xf32>
    %651 = stablehlo.convert %arg22 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %652 = stablehlo.broadcast_in_dim %650, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %653 = stablehlo.broadcast_in_dim %651, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %654 = stablehlo.add %652, %653 : tensor<1x197x1024xf32>
    %655 = stablehlo.convert %654 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %656 = stablehlo.reshape %655 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %657 = stablehlo.convert %656 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %658 = stablehlo.dot_general %657, %arg186, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %659 = stablehlo.broadcast_in_dim %658, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %660 = stablehlo.multiply %659, %60 : tensor<197x1024xf32>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %662 = stablehlo.broadcast_in_dim %arg187, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %663 = stablehlo.add %661, %662 : tensor<197x1024xf32>
    %664 = stablehlo.convert %663 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %665 = stablehlo.reshape %664 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %666 = stablehlo.dot_general %656, %arg188, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %667 = stablehlo.reshape %666 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %668 = stablehlo.reshape %667 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %669 = stablehlo.transpose %668, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %670 = stablehlo.dot_general %657, %arg189, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %672 = stablehlo.multiply %671, %60 : tensor<197x1024xf32>
    %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %674 = stablehlo.broadcast_in_dim %arg190, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %675 = stablehlo.add %673, %674 : tensor<197x1024xf32>
    %676 = stablehlo.convert %675 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %677 = stablehlo.reshape %676 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %678 = stablehlo.reshape %677 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %679 = stablehlo.transpose %678, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %680 = stablehlo.reshape %665 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %681 = stablehlo.transpose %680, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %682 = stablehlo.transpose %669, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %683 = stablehlo.reshape %681 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %684 = stablehlo.reshape %682 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %685 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %686 = stablehlo.dot_general %683, %685, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %687 = stablehlo.reshape %686 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %689 = stablehlo.divide %688, %92 : tensor<1x16x197x197xbf16>
    %690 = stablehlo.add %689, %arg191 : tensor<1x16x197x197xbf16>
    %691 = stablehlo.convert %690 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %692 = stablehlo.reduce(%691 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %693 = stablehlo.reshape %692 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %694 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %695 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %696 = stablehlo.subtract %694, %695 : tensor<1x16x197x197xf32>
    %697 = stablehlo.exponential %696 : tensor<1x16x197x197xf32>
    %698 = stablehlo.reduce(%697 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %699 = stablehlo.reshape %698 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %700 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %701 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %702 = stablehlo.divide %700, %701 : tensor<1x16x197x197xf32>
    %703 = stablehlo.convert %702 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %704 = stablehlo.reshape %703 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %705 = stablehlo.reshape %679 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %706 = stablehlo.broadcast_in_dim %705, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %707 = stablehlo.dot_general %704, %706, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %708 = stablehlo.reshape %707 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %709 = stablehlo.transpose %708, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %710 = stablehlo.reshape %709 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %711 = stablehlo.reshape %710 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %712 = stablehlo.convert %711 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %713 = stablehlo.dot_general %712, %arg192, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %714 = stablehlo.broadcast_in_dim %713, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %715 = stablehlo.multiply %714, %60 : tensor<197x1024xf32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %717 = stablehlo.broadcast_in_dim %arg193, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %718 = stablehlo.add %716, %717 : tensor<197x1024xf32>
    %719 = stablehlo.convert %718 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %720 = stablehlo.reshape %719 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %721 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %722 = stablehlo.broadcast_in_dim %720, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %723 = stablehlo.multiply %721, %722 : tensor<1x197x1024xbf16>
    %724 = stablehlo.add %723, %618 : tensor<1x197x1024xbf16>
    %725 = stablehlo.convert %724 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %726 = stablehlo.convert %725 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %727 = stablehlo.reduce(%726 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %728 = stablehlo.reshape %727 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %729 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %730 = stablehlo.divide %729, %15 : tensor<1x197x1xf64>
    %731 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %732 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %733 = stablehlo.subtract %731, %732 : tensor<1x197x1024xf64>
    %734 = stablehlo.multiply %733, %733 : tensor<1x197x1024xf64>
    %735 = stablehlo.reduce(%734 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %736 = stablehlo.reshape %735 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %737 = stablehlo.broadcast_in_dim %736, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %738 = stablehlo.divide %737, %15 : tensor<1x197x1xf64>
    %739 = stablehlo.convert %738 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %740 = stablehlo.reduce(%725 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %741 = stablehlo.reshape %740 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %742 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %743 = stablehlo.divide %742, %31 : tensor<1x197x1xf32>
    %744 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %745 = stablehlo.add %744, %36 : tensor<1x197x1xf32>
    %746 = stablehlo.rsqrt %745 : tensor<1x197x1xf32>
    %747 = stablehlo.broadcast_in_dim %725, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %748 = stablehlo.broadcast_in_dim %743, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %749 = stablehlo.subtract %747, %748 : tensor<1x197x1024xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %751 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %752 = stablehlo.multiply %750, %751 : tensor<1x197x1024xf32>
    %753 = stablehlo.convert %arg24 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %754 = stablehlo.broadcast_in_dim %752, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %755 = stablehlo.broadcast_in_dim %753, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %756 = stablehlo.multiply %754, %755 : tensor<1x197x1024xf32>
    %757 = stablehlo.convert %arg25 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %758 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %759 = stablehlo.broadcast_in_dim %757, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %760 = stablehlo.add %758, %759 : tensor<1x197x1024xf32>
    %761 = stablehlo.convert %760 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %762 = stablehlo.reshape %761 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %763 = stablehlo.convert %762 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %764 = stablehlo.dot_general %763, %arg194, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %766 = stablehlo.multiply %765, %170 : tensor<197x4096xf32>
    %767 = stablehlo.broadcast_in_dim %766, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %768 = stablehlo.broadcast_in_dim %arg195, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %769 = stablehlo.add %767, %768 : tensor<197x4096xf32>
    %770 = stablehlo.convert %769 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %771 = stablehlo.reshape %770 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %772 = stablehlo.multiply %771, %cst_4 : tensor<1x197x4096xbf16>
    %773 = stablehlo.multiply %771, %178 : tensor<1x197x4096xbf16>
    %774 = stablehlo.convert %773 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %775 = stablehlo.clamp %cst_5, %774, %cst_6 : tensor<1x197x4096xf32>
    %776 = stablehlo.multiply %775, %775 : tensor<1x197x4096xf32>
    %777 = stablehlo.multiply %cst_7, %776 : tensor<1x197x4096xf32>
    %778 = stablehlo.add %777, %cst_8 : tensor<1x197x4096xf32>
    %779 = stablehlo.multiply %778, %776 : tensor<1x197x4096xf32>
    %780 = stablehlo.add %779, %cst_9 : tensor<1x197x4096xf32>
    %781 = stablehlo.multiply %780, %776 : tensor<1x197x4096xf32>
    %782 = stablehlo.add %781, %cst_10 : tensor<1x197x4096xf32>
    %783 = stablehlo.multiply %782, %776 : tensor<1x197x4096xf32>
    %784 = stablehlo.add %783, %cst_11 : tensor<1x197x4096xf32>
    %785 = stablehlo.multiply %784, %776 : tensor<1x197x4096xf32>
    %786 = stablehlo.add %785, %cst_12 : tensor<1x197x4096xf32>
    %787 = stablehlo.multiply %786, %776 : tensor<1x197x4096xf32>
    %788 = stablehlo.add %787, %cst_13 : tensor<1x197x4096xf32>
    %789 = stablehlo.multiply %cst_14, %776 : tensor<1x197x4096xf32>
    %790 = stablehlo.add %789, %cst_15 : tensor<1x197x4096xf32>
    %791 = stablehlo.multiply %790, %776 : tensor<1x197x4096xf32>
    %792 = stablehlo.add %791, %cst_16 : tensor<1x197x4096xf32>
    %793 = stablehlo.multiply %792, %776 : tensor<1x197x4096xf32>
    %794 = stablehlo.add %793, %cst_17 : tensor<1x197x4096xf32>
    %795 = stablehlo.multiply %794, %776 : tensor<1x197x4096xf32>
    %796 = stablehlo.add %795, %cst_18 : tensor<1x197x4096xf32>
    %797 = stablehlo.multiply %775, %788 : tensor<1x197x4096xf32>
    %798 = stablehlo.divide %797, %796 : tensor<1x197x4096xf32>
    %799 = stablehlo.clamp %cst_19, %798, %cst_20 : tensor<1x197x4096xf32>
    %800 = stablehlo.convert %799 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %801 = stablehlo.add %800, %cst_2 : tensor<1x197x4096xbf16>
    %802 = stablehlo.multiply %801, %772 : tensor<1x197x4096xbf16>
    %803 = stablehlo.reshape %802 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %804 = stablehlo.convert %803 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %805 = stablehlo.dot_general %804, %arg196, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %806 = stablehlo.broadcast_in_dim %805, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %807 = stablehlo.multiply %806, %60 : tensor<197x1024xf32>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %809 = stablehlo.broadcast_in_dim %arg197, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %810 = stablehlo.add %808, %809 : tensor<197x1024xf32>
    %811 = stablehlo.convert %810 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %812 = stablehlo.reshape %811 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %813 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %814 = stablehlo.broadcast_in_dim %812, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %815 = stablehlo.multiply %813, %814 : tensor<1x197x1024xbf16>
    %816 = stablehlo.add %815, %724 : tensor<1x197x1024xbf16>
    %817 = stablehlo.convert %816 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %818 = stablehlo.convert %817 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %819 = stablehlo.reduce(%818 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %820 = stablehlo.reshape %819 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %822 = stablehlo.divide %821, %15 : tensor<1x197x1xf64>
    %823 = stablehlo.broadcast_in_dim %818, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %824 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %825 = stablehlo.subtract %823, %824 : tensor<1x197x1024xf64>
    %826 = stablehlo.multiply %825, %825 : tensor<1x197x1024xf64>
    %827 = stablehlo.reduce(%826 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %828 = stablehlo.reshape %827 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %830 = stablehlo.divide %829, %15 : tensor<1x197x1xf64>
    %831 = stablehlo.convert %830 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %832 = stablehlo.reduce(%817 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %833 = stablehlo.reshape %832 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %834 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %835 = stablehlo.divide %834, %31 : tensor<1x197x1xf32>
    %836 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %837 = stablehlo.add %836, %36 : tensor<1x197x1xf32>
    %838 = stablehlo.rsqrt %837 : tensor<1x197x1xf32>
    %839 = stablehlo.broadcast_in_dim %817, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %840 = stablehlo.broadcast_in_dim %835, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %841 = stablehlo.subtract %839, %840 : tensor<1x197x1024xf32>
    %842 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %843 = stablehlo.broadcast_in_dim %838, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %844 = stablehlo.multiply %842, %843 : tensor<1x197x1024xf32>
    %845 = stablehlo.convert %arg27 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %846 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %847 = stablehlo.broadcast_in_dim %845, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %848 = stablehlo.multiply %846, %847 : tensor<1x197x1024xf32>
    %849 = stablehlo.convert %arg28 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %850 = stablehlo.broadcast_in_dim %848, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %851 = stablehlo.broadcast_in_dim %849, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %852 = stablehlo.add %850, %851 : tensor<1x197x1024xf32>
    %853 = stablehlo.convert %852 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %854 = stablehlo.reshape %853 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %855 = stablehlo.convert %854 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %856 = stablehlo.dot_general %855, %arg198, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %857 = stablehlo.broadcast_in_dim %856, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %858 = stablehlo.multiply %857, %60 : tensor<197x1024xf32>
    %859 = stablehlo.broadcast_in_dim %858, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %860 = stablehlo.broadcast_in_dim %arg199, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %861 = stablehlo.add %859, %860 : tensor<197x1024xf32>
    %862 = stablehlo.convert %861 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %863 = stablehlo.reshape %862 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %864 = stablehlo.dot_general %854, %arg200, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %865 = stablehlo.reshape %864 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %866 = stablehlo.reshape %865 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %867 = stablehlo.transpose %866, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %868 = stablehlo.dot_general %855, %arg201, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %869 = stablehlo.broadcast_in_dim %868, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %870 = stablehlo.multiply %869, %60 : tensor<197x1024xf32>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %872 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %873 = stablehlo.add %871, %872 : tensor<197x1024xf32>
    %874 = stablehlo.convert %873 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %875 = stablehlo.reshape %874 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %876 = stablehlo.reshape %875 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %877 = stablehlo.transpose %876, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %878 = stablehlo.reshape %863 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %879 = stablehlo.transpose %878, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %880 = stablehlo.transpose %867, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %881 = stablehlo.reshape %879 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %882 = stablehlo.reshape %880 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %884 = stablehlo.dot_general %881, %883, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %885 = stablehlo.reshape %884 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %886 = stablehlo.broadcast_in_dim %885, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %887 = stablehlo.divide %886, %92 : tensor<1x16x197x197xbf16>
    %888 = stablehlo.add %887, %arg203 : tensor<1x16x197x197xbf16>
    %889 = stablehlo.convert %888 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %890 = stablehlo.reduce(%889 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %891 = stablehlo.reshape %890 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %892 = stablehlo.broadcast_in_dim %889, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %893 = stablehlo.broadcast_in_dim %891, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %894 = stablehlo.subtract %892, %893 : tensor<1x16x197x197xf32>
    %895 = stablehlo.exponential %894 : tensor<1x16x197x197xf32>
    %896 = stablehlo.reduce(%895 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %897 = stablehlo.reshape %896 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %898 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %899 = stablehlo.broadcast_in_dim %897, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %900 = stablehlo.divide %898, %899 : tensor<1x16x197x197xf32>
    %901 = stablehlo.convert %900 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %902 = stablehlo.reshape %901 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %903 = stablehlo.reshape %877 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %905 = stablehlo.dot_general %902, %904, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %906 = stablehlo.reshape %905 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %907 = stablehlo.transpose %906, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %908 = stablehlo.reshape %907 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %909 = stablehlo.reshape %908 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %910 = stablehlo.convert %909 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %911 = stablehlo.dot_general %910, %arg204, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %912 = stablehlo.broadcast_in_dim %911, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %913 = stablehlo.multiply %912, %60 : tensor<197x1024xf32>
    %914 = stablehlo.broadcast_in_dim %913, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %915 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %916 = stablehlo.add %914, %915 : tensor<197x1024xf32>
    %917 = stablehlo.convert %916 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %918 = stablehlo.reshape %917 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %919 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %920 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %921 = stablehlo.multiply %919, %920 : tensor<1x197x1024xbf16>
    %922 = stablehlo.add %921, %816 : tensor<1x197x1024xbf16>
    %923 = stablehlo.convert %922 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %924 = stablehlo.convert %923 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %925 = stablehlo.reduce(%924 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %926 = stablehlo.reshape %925 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %927 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %928 = stablehlo.divide %927, %15 : tensor<1x197x1xf64>
    %929 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %930 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %931 = stablehlo.subtract %929, %930 : tensor<1x197x1024xf64>
    %932 = stablehlo.multiply %931, %931 : tensor<1x197x1024xf64>
    %933 = stablehlo.reduce(%932 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %934 = stablehlo.reshape %933 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %936 = stablehlo.divide %935, %15 : tensor<1x197x1xf64>
    %937 = stablehlo.convert %936 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %938 = stablehlo.reduce(%923 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %939 = stablehlo.reshape %938 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %941 = stablehlo.divide %940, %31 : tensor<1x197x1xf32>
    %942 = stablehlo.broadcast_in_dim %937, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %943 = stablehlo.add %942, %36 : tensor<1x197x1xf32>
    %944 = stablehlo.rsqrt %943 : tensor<1x197x1xf32>
    %945 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %946 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %947 = stablehlo.subtract %945, %946 : tensor<1x197x1024xf32>
    %948 = stablehlo.broadcast_in_dim %947, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %949 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %950 = stablehlo.multiply %948, %949 : tensor<1x197x1024xf32>
    %951 = stablehlo.convert %arg30 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %952 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %953 = stablehlo.broadcast_in_dim %951, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %954 = stablehlo.multiply %952, %953 : tensor<1x197x1024xf32>
    %955 = stablehlo.convert %arg31 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %956 = stablehlo.broadcast_in_dim %954, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %957 = stablehlo.broadcast_in_dim %955, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %958 = stablehlo.add %956, %957 : tensor<1x197x1024xf32>
    %959 = stablehlo.convert %958 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %960 = stablehlo.reshape %959 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %961 = stablehlo.convert %960 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %962 = stablehlo.dot_general %961, %arg206, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %964 = stablehlo.multiply %963, %170 : tensor<197x4096xf32>
    %965 = stablehlo.broadcast_in_dim %964, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %966 = stablehlo.broadcast_in_dim %arg207, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %967 = stablehlo.add %965, %966 : tensor<197x4096xf32>
    %968 = stablehlo.convert %967 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %969 = stablehlo.reshape %968 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %970 = stablehlo.multiply %969, %cst_4 : tensor<1x197x4096xbf16>
    %971 = stablehlo.multiply %969, %178 : tensor<1x197x4096xbf16>
    %972 = stablehlo.convert %971 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %973 = stablehlo.clamp %cst_5, %972, %cst_6 : tensor<1x197x4096xf32>
    %974 = stablehlo.multiply %973, %973 : tensor<1x197x4096xf32>
    %975 = stablehlo.multiply %cst_7, %974 : tensor<1x197x4096xf32>
    %976 = stablehlo.add %975, %cst_8 : tensor<1x197x4096xf32>
    %977 = stablehlo.multiply %976, %974 : tensor<1x197x4096xf32>
    %978 = stablehlo.add %977, %cst_9 : tensor<1x197x4096xf32>
    %979 = stablehlo.multiply %978, %974 : tensor<1x197x4096xf32>
    %980 = stablehlo.add %979, %cst_10 : tensor<1x197x4096xf32>
    %981 = stablehlo.multiply %980, %974 : tensor<1x197x4096xf32>
    %982 = stablehlo.add %981, %cst_11 : tensor<1x197x4096xf32>
    %983 = stablehlo.multiply %982, %974 : tensor<1x197x4096xf32>
    %984 = stablehlo.add %983, %cst_12 : tensor<1x197x4096xf32>
    %985 = stablehlo.multiply %984, %974 : tensor<1x197x4096xf32>
    %986 = stablehlo.add %985, %cst_13 : tensor<1x197x4096xf32>
    %987 = stablehlo.multiply %cst_14, %974 : tensor<1x197x4096xf32>
    %988 = stablehlo.add %987, %cst_15 : tensor<1x197x4096xf32>
    %989 = stablehlo.multiply %988, %974 : tensor<1x197x4096xf32>
    %990 = stablehlo.add %989, %cst_16 : tensor<1x197x4096xf32>
    %991 = stablehlo.multiply %990, %974 : tensor<1x197x4096xf32>
    %992 = stablehlo.add %991, %cst_17 : tensor<1x197x4096xf32>
    %993 = stablehlo.multiply %992, %974 : tensor<1x197x4096xf32>
    %994 = stablehlo.add %993, %cst_18 : tensor<1x197x4096xf32>
    %995 = stablehlo.multiply %973, %986 : tensor<1x197x4096xf32>
    %996 = stablehlo.divide %995, %994 : tensor<1x197x4096xf32>
    %997 = stablehlo.clamp %cst_19, %996, %cst_20 : tensor<1x197x4096xf32>
    %998 = stablehlo.convert %997 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %999 = stablehlo.add %998, %cst_2 : tensor<1x197x4096xbf16>
    %1000 = stablehlo.multiply %999, %970 : tensor<1x197x4096xbf16>
    %1001 = stablehlo.reshape %1000 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %1002 = stablehlo.convert %1001 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %1003 = stablehlo.dot_general %1002, %arg208, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1005 = stablehlo.multiply %1004, %60 : tensor<197x1024xf32>
    %1006 = stablehlo.broadcast_in_dim %1005, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1007 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1008 = stablehlo.add %1006, %1007 : tensor<197x1024xf32>
    %1009 = stablehlo.convert %1008 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1010 = stablehlo.reshape %1009 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1011 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1012 = stablehlo.broadcast_in_dim %1010, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1013 = stablehlo.multiply %1011, %1012 : tensor<1x197x1024xbf16>
    %1014 = stablehlo.add %1013, %922 : tensor<1x197x1024xbf16>
    %1015 = stablehlo.convert %1014 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1016 = stablehlo.convert %1015 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1017 = stablehlo.reduce(%1016 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1018 = stablehlo.reshape %1017 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1019 = stablehlo.broadcast_in_dim %1018, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1020 = stablehlo.divide %1019, %15 : tensor<1x197x1xf64>
    %1021 = stablehlo.broadcast_in_dim %1016, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1022 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1023 = stablehlo.subtract %1021, %1022 : tensor<1x197x1024xf64>
    %1024 = stablehlo.multiply %1023, %1023 : tensor<1x197x1024xf64>
    %1025 = stablehlo.reduce(%1024 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1026 = stablehlo.reshape %1025 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1027 = stablehlo.broadcast_in_dim %1026, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1028 = stablehlo.divide %1027, %15 : tensor<1x197x1xf64>
    %1029 = stablehlo.convert %1028 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1030 = stablehlo.reduce(%1015 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1031 = stablehlo.reshape %1030 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1032 = stablehlo.broadcast_in_dim %1031, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1033 = stablehlo.divide %1032, %31 : tensor<1x197x1xf32>
    %1034 = stablehlo.broadcast_in_dim %1029, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1035 = stablehlo.add %1034, %36 : tensor<1x197x1xf32>
    %1036 = stablehlo.rsqrt %1035 : tensor<1x197x1xf32>
    %1037 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1038 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1039 = stablehlo.subtract %1037, %1038 : tensor<1x197x1024xf32>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1041 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1042 = stablehlo.multiply %1040, %1041 : tensor<1x197x1024xf32>
    %1043 = stablehlo.convert %arg33 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1044 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1045 = stablehlo.broadcast_in_dim %1043, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1046 = stablehlo.multiply %1044, %1045 : tensor<1x197x1024xf32>
    %1047 = stablehlo.convert %arg34 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1048 = stablehlo.broadcast_in_dim %1046, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1049 = stablehlo.broadcast_in_dim %1047, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1050 = stablehlo.add %1048, %1049 : tensor<1x197x1024xf32>
    %1051 = stablehlo.convert %1050 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1052 = stablehlo.reshape %1051 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1053 = stablehlo.convert %1052 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1054 = stablehlo.dot_general %1053, %arg210, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1055 = stablehlo.broadcast_in_dim %1054, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1056 = stablehlo.multiply %1055, %60 : tensor<197x1024xf32>
    %1057 = stablehlo.broadcast_in_dim %1056, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1058 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1059 = stablehlo.add %1057, %1058 : tensor<197x1024xf32>
    %1060 = stablehlo.convert %1059 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1061 = stablehlo.reshape %1060 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1062 = stablehlo.dot_general %1052, %arg212, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %1063 = stablehlo.reshape %1062 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1064 = stablehlo.reshape %1063 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1065 = stablehlo.transpose %1064, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1066 = stablehlo.dot_general %1053, %arg213, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1067 = stablehlo.broadcast_in_dim %1066, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1068 = stablehlo.multiply %1067, %60 : tensor<197x1024xf32>
    %1069 = stablehlo.broadcast_in_dim %1068, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1070 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1071 = stablehlo.add %1069, %1070 : tensor<197x1024xf32>
    %1072 = stablehlo.convert %1071 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1073 = stablehlo.reshape %1072 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1074 = stablehlo.reshape %1073 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1075 = stablehlo.transpose %1074, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1076 = stablehlo.reshape %1061 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1077 = stablehlo.transpose %1076, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1078 = stablehlo.transpose %1065, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %1079 = stablehlo.reshape %1077 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1080 = stablehlo.reshape %1078 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1081 = stablehlo.broadcast_in_dim %1080, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1082 = stablehlo.dot_general %1079, %1081, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %1083 = stablehlo.reshape %1082 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1084 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1085 = stablehlo.divide %1084, %92 : tensor<1x16x197x197xbf16>
    %1086 = stablehlo.add %1085, %arg215 : tensor<1x16x197x197xbf16>
    %1087 = stablehlo.convert %1086 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %1088 = stablehlo.reduce(%1087 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1089 = stablehlo.reshape %1088 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1090 = stablehlo.broadcast_in_dim %1087, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1091 = stablehlo.broadcast_in_dim %1089, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1092 = stablehlo.subtract %1090, %1091 : tensor<1x16x197x197xf32>
    %1093 = stablehlo.exponential %1092 : tensor<1x16x197x197xf32>
    %1094 = stablehlo.reduce(%1093 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1095 = stablehlo.reshape %1094 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1096 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1097 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1098 = stablehlo.divide %1096, %1097 : tensor<1x16x197x197xf32>
    %1099 = stablehlo.convert %1098 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %1100 = stablehlo.reshape %1099 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %1101 = stablehlo.reshape %1075 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1102 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1103 = stablehlo.dot_general %1100, %1102, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1104 = stablehlo.reshape %1103 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1105 = stablehlo.transpose %1104, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %1106 = stablehlo.reshape %1105 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %1107 = stablehlo.reshape %1106 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1108 = stablehlo.convert %1107 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1109 = stablehlo.dot_general %1108, %arg216, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1110 = stablehlo.broadcast_in_dim %1109, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1111 = stablehlo.multiply %1110, %60 : tensor<197x1024xf32>
    %1112 = stablehlo.broadcast_in_dim %1111, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1113 = stablehlo.broadcast_in_dim %arg217, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1114 = stablehlo.add %1112, %1113 : tensor<197x1024xf32>
    %1115 = stablehlo.convert %1114 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1116 = stablehlo.reshape %1115 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1117 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1118 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1119 = stablehlo.multiply %1117, %1118 : tensor<1x197x1024xbf16>
    %1120 = stablehlo.add %1119, %1014 : tensor<1x197x1024xbf16>
    %1121 = stablehlo.convert %1120 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1122 = stablehlo.convert %1121 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1123 = stablehlo.reduce(%1122 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1124 = stablehlo.reshape %1123 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1125 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1126 = stablehlo.divide %1125, %15 : tensor<1x197x1xf64>
    %1127 = stablehlo.broadcast_in_dim %1122, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1128 = stablehlo.broadcast_in_dim %1126, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1129 = stablehlo.subtract %1127, %1128 : tensor<1x197x1024xf64>
    %1130 = stablehlo.multiply %1129, %1129 : tensor<1x197x1024xf64>
    %1131 = stablehlo.reduce(%1130 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1133 = stablehlo.broadcast_in_dim %1132, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1134 = stablehlo.divide %1133, %15 : tensor<1x197x1xf64>
    %1135 = stablehlo.convert %1134 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1136 = stablehlo.reduce(%1121 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1137 = stablehlo.reshape %1136 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1139 = stablehlo.divide %1138, %31 : tensor<1x197x1xf32>
    %1140 = stablehlo.broadcast_in_dim %1135, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1141 = stablehlo.add %1140, %36 : tensor<1x197x1xf32>
    %1142 = stablehlo.rsqrt %1141 : tensor<1x197x1xf32>
    %1143 = stablehlo.broadcast_in_dim %1121, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1144 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1145 = stablehlo.subtract %1143, %1144 : tensor<1x197x1024xf32>
    %1146 = stablehlo.broadcast_in_dim %1145, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1147 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1148 = stablehlo.multiply %1146, %1147 : tensor<1x197x1024xf32>
    %1149 = stablehlo.convert %arg36 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1150 = stablehlo.broadcast_in_dim %1148, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1151 = stablehlo.broadcast_in_dim %1149, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1152 = stablehlo.multiply %1150, %1151 : tensor<1x197x1024xf32>
    %1153 = stablehlo.convert %arg37 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1154 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1155 = stablehlo.broadcast_in_dim %1153, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1156 = stablehlo.add %1154, %1155 : tensor<1x197x1024xf32>
    %1157 = stablehlo.convert %1156 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1158 = stablehlo.reshape %1157 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1159 = stablehlo.convert %1158 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1160 = stablehlo.dot_general %1159, %arg218, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %1161 = stablehlo.broadcast_in_dim %1160, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1162 = stablehlo.multiply %1161, %170 : tensor<197x4096xf32>
    %1163 = stablehlo.broadcast_in_dim %1162, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1164 = stablehlo.broadcast_in_dim %arg219, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %1165 = stablehlo.add %1163, %1164 : tensor<197x4096xf32>
    %1166 = stablehlo.convert %1165 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %1167 = stablehlo.reshape %1166 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %1168 = stablehlo.multiply %1167, %cst_4 : tensor<1x197x4096xbf16>
    %1169 = stablehlo.multiply %1167, %178 : tensor<1x197x4096xbf16>
    %1170 = stablehlo.convert %1169 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %1171 = stablehlo.clamp %cst_5, %1170, %cst_6 : tensor<1x197x4096xf32>
    %1172 = stablehlo.multiply %1171, %1171 : tensor<1x197x4096xf32>
    %1173 = stablehlo.multiply %cst_7, %1172 : tensor<1x197x4096xf32>
    %1174 = stablehlo.add %1173, %cst_8 : tensor<1x197x4096xf32>
    %1175 = stablehlo.multiply %1174, %1172 : tensor<1x197x4096xf32>
    %1176 = stablehlo.add %1175, %cst_9 : tensor<1x197x4096xf32>
    %1177 = stablehlo.multiply %1176, %1172 : tensor<1x197x4096xf32>
    %1178 = stablehlo.add %1177, %cst_10 : tensor<1x197x4096xf32>
    %1179 = stablehlo.multiply %1178, %1172 : tensor<1x197x4096xf32>
    %1180 = stablehlo.add %1179, %cst_11 : tensor<1x197x4096xf32>
    %1181 = stablehlo.multiply %1180, %1172 : tensor<1x197x4096xf32>
    %1182 = stablehlo.add %1181, %cst_12 : tensor<1x197x4096xf32>
    %1183 = stablehlo.multiply %1182, %1172 : tensor<1x197x4096xf32>
    %1184 = stablehlo.add %1183, %cst_13 : tensor<1x197x4096xf32>
    %1185 = stablehlo.multiply %cst_14, %1172 : tensor<1x197x4096xf32>
    %1186 = stablehlo.add %1185, %cst_15 : tensor<1x197x4096xf32>
    %1187 = stablehlo.multiply %1186, %1172 : tensor<1x197x4096xf32>
    %1188 = stablehlo.add %1187, %cst_16 : tensor<1x197x4096xf32>
    %1189 = stablehlo.multiply %1188, %1172 : tensor<1x197x4096xf32>
    %1190 = stablehlo.add %1189, %cst_17 : tensor<1x197x4096xf32>
    %1191 = stablehlo.multiply %1190, %1172 : tensor<1x197x4096xf32>
    %1192 = stablehlo.add %1191, %cst_18 : tensor<1x197x4096xf32>
    %1193 = stablehlo.multiply %1171, %1184 : tensor<1x197x4096xf32>
    %1194 = stablehlo.divide %1193, %1192 : tensor<1x197x4096xf32>
    %1195 = stablehlo.clamp %cst_19, %1194, %cst_20 : tensor<1x197x4096xf32>
    %1196 = stablehlo.convert %1195 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %1197 = stablehlo.add %1196, %cst_2 : tensor<1x197x4096xbf16>
    %1198 = stablehlo.multiply %1197, %1168 : tensor<1x197x4096xbf16>
    %1199 = stablehlo.reshape %1198 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %1200 = stablehlo.convert %1199 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %1201 = stablehlo.dot_general %1200, %arg220, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1203 = stablehlo.multiply %1202, %60 : tensor<197x1024xf32>
    %1204 = stablehlo.broadcast_in_dim %1203, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1205 = stablehlo.broadcast_in_dim %arg221, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1206 = stablehlo.add %1204, %1205 : tensor<197x1024xf32>
    %1207 = stablehlo.convert %1206 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1208 = stablehlo.reshape %1207 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1209 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1210 = stablehlo.broadcast_in_dim %1208, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1211 = stablehlo.multiply %1209, %1210 : tensor<1x197x1024xbf16>
    %1212 = stablehlo.add %1211, %1120 : tensor<1x197x1024xbf16>
    %1213 = stablehlo.convert %1212 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1214 = stablehlo.convert %1213 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1215 = stablehlo.reduce(%1214 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1216 = stablehlo.reshape %1215 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1218 = stablehlo.divide %1217, %15 : tensor<1x197x1xf64>
    %1219 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1220 = stablehlo.broadcast_in_dim %1218, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1221 = stablehlo.subtract %1219, %1220 : tensor<1x197x1024xf64>
    %1222 = stablehlo.multiply %1221, %1221 : tensor<1x197x1024xf64>
    %1223 = stablehlo.reduce(%1222 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1224 = stablehlo.reshape %1223 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1225 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1226 = stablehlo.divide %1225, %15 : tensor<1x197x1xf64>
    %1227 = stablehlo.convert %1226 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1228 = stablehlo.reduce(%1213 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1229 = stablehlo.reshape %1228 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1231 = stablehlo.divide %1230, %31 : tensor<1x197x1xf32>
    %1232 = stablehlo.broadcast_in_dim %1227, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1233 = stablehlo.add %1232, %36 : tensor<1x197x1xf32>
    %1234 = stablehlo.rsqrt %1233 : tensor<1x197x1xf32>
    %1235 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1236 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1237 = stablehlo.subtract %1235, %1236 : tensor<1x197x1024xf32>
    %1238 = stablehlo.broadcast_in_dim %1237, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1239 = stablehlo.broadcast_in_dim %1234, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1240 = stablehlo.multiply %1238, %1239 : tensor<1x197x1024xf32>
    %1241 = stablehlo.convert %arg39 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1242 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1243 = stablehlo.broadcast_in_dim %1241, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1244 = stablehlo.multiply %1242, %1243 : tensor<1x197x1024xf32>
    %1245 = stablehlo.convert %arg40 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1246 = stablehlo.broadcast_in_dim %1244, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1247 = stablehlo.broadcast_in_dim %1245, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1248 = stablehlo.add %1246, %1247 : tensor<1x197x1024xf32>
    %1249 = stablehlo.convert %1248 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1250 = stablehlo.reshape %1249 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1251 = stablehlo.convert %1250 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1252 = stablehlo.dot_general %1251, %arg222, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1253 = stablehlo.broadcast_in_dim %1252, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1254 = stablehlo.multiply %1253, %60 : tensor<197x1024xf32>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1256 = stablehlo.broadcast_in_dim %arg223, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1257 = stablehlo.add %1255, %1256 : tensor<197x1024xf32>
    %1258 = stablehlo.convert %1257 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1259 = stablehlo.reshape %1258 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1260 = stablehlo.dot_general %1250, %arg224, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %1261 = stablehlo.reshape %1260 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1262 = stablehlo.reshape %1261 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1263 = stablehlo.transpose %1262, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1264 = stablehlo.dot_general %1251, %arg225, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1265 = stablehlo.broadcast_in_dim %1264, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1266 = stablehlo.multiply %1265, %60 : tensor<197x1024xf32>
    %1267 = stablehlo.broadcast_in_dim %1266, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1268 = stablehlo.broadcast_in_dim %arg226, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1269 = stablehlo.add %1267, %1268 : tensor<197x1024xf32>
    %1270 = stablehlo.convert %1269 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1271 = stablehlo.reshape %1270 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1272 = stablehlo.reshape %1271 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1273 = stablehlo.transpose %1272, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1274 = stablehlo.reshape %1259 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1275 = stablehlo.transpose %1274, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1276 = stablehlo.transpose %1263, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %1277 = stablehlo.reshape %1275 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1278 = stablehlo.reshape %1276 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1279 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1280 = stablehlo.dot_general %1277, %1279, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %1281 = stablehlo.reshape %1280 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1282 = stablehlo.broadcast_in_dim %1281, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1283 = stablehlo.divide %1282, %92 : tensor<1x16x197x197xbf16>
    %1284 = stablehlo.add %1283, %arg227 : tensor<1x16x197x197xbf16>
    %1285 = stablehlo.convert %1284 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %1286 = stablehlo.reduce(%1285 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1287 = stablehlo.reshape %1286 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1288 = stablehlo.broadcast_in_dim %1285, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1289 = stablehlo.broadcast_in_dim %1287, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1290 = stablehlo.subtract %1288, %1289 : tensor<1x16x197x197xf32>
    %1291 = stablehlo.exponential %1290 : tensor<1x16x197x197xf32>
    %1292 = stablehlo.reduce(%1291 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1293 = stablehlo.reshape %1292 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1294 = stablehlo.broadcast_in_dim %1291, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1295 = stablehlo.broadcast_in_dim %1293, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1296 = stablehlo.divide %1294, %1295 : tensor<1x16x197x197xf32>
    %1297 = stablehlo.convert %1296 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %1298 = stablehlo.reshape %1297 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %1299 = stablehlo.reshape %1273 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1301 = stablehlo.dot_general %1298, %1300, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1302 = stablehlo.reshape %1301 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1303 = stablehlo.transpose %1302, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %1304 = stablehlo.reshape %1303 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %1305 = stablehlo.reshape %1304 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1306 = stablehlo.convert %1305 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1307 = stablehlo.dot_general %1306, %arg228, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1308 = stablehlo.broadcast_in_dim %1307, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1309 = stablehlo.multiply %1308, %60 : tensor<197x1024xf32>
    %1310 = stablehlo.broadcast_in_dim %1309, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1311 = stablehlo.broadcast_in_dim %arg229, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1312 = stablehlo.add %1310, %1311 : tensor<197x1024xf32>
    %1313 = stablehlo.convert %1312 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1314 = stablehlo.reshape %1313 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1315 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1316 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1317 = stablehlo.multiply %1315, %1316 : tensor<1x197x1024xbf16>
    %1318 = stablehlo.add %1317, %1212 : tensor<1x197x1024xbf16>
    %1319 = stablehlo.convert %1318 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1320 = stablehlo.convert %1319 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1321 = stablehlo.reduce(%1320 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1322 = stablehlo.reshape %1321 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1324 = stablehlo.divide %1323, %15 : tensor<1x197x1xf64>
    %1325 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1326 = stablehlo.broadcast_in_dim %1324, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1327 = stablehlo.subtract %1325, %1326 : tensor<1x197x1024xf64>
    %1328 = stablehlo.multiply %1327, %1327 : tensor<1x197x1024xf64>
    %1329 = stablehlo.reduce(%1328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1330 = stablehlo.reshape %1329 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1332 = stablehlo.divide %1331, %15 : tensor<1x197x1xf64>
    %1333 = stablehlo.convert %1332 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1334 = stablehlo.reduce(%1319 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1335 = stablehlo.reshape %1334 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1336 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1337 = stablehlo.divide %1336, %31 : tensor<1x197x1xf32>
    %1338 = stablehlo.broadcast_in_dim %1333, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1339 = stablehlo.add %1338, %36 : tensor<1x197x1xf32>
    %1340 = stablehlo.rsqrt %1339 : tensor<1x197x1xf32>
    %1341 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1342 = stablehlo.broadcast_in_dim %1337, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1343 = stablehlo.subtract %1341, %1342 : tensor<1x197x1024xf32>
    %1344 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1345 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1346 = stablehlo.multiply %1344, %1345 : tensor<1x197x1024xf32>
    %1347 = stablehlo.convert %arg42 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1348 = stablehlo.broadcast_in_dim %1346, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1349 = stablehlo.broadcast_in_dim %1347, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1350 = stablehlo.multiply %1348, %1349 : tensor<1x197x1024xf32>
    %1351 = stablehlo.convert %arg43 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1352 = stablehlo.broadcast_in_dim %1350, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1353 = stablehlo.broadcast_in_dim %1351, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1354 = stablehlo.add %1352, %1353 : tensor<1x197x1024xf32>
    %1355 = stablehlo.convert %1354 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1356 = stablehlo.reshape %1355 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1357 = stablehlo.convert %1356 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1358 = stablehlo.dot_general %1357, %arg230, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %1359 = stablehlo.broadcast_in_dim %1358, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1360 = stablehlo.multiply %1359, %170 : tensor<197x4096xf32>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1362 = stablehlo.broadcast_in_dim %arg231, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %1363 = stablehlo.add %1361, %1362 : tensor<197x4096xf32>
    %1364 = stablehlo.convert %1363 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %1365 = stablehlo.reshape %1364 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %1366 = stablehlo.multiply %1365, %cst_4 : tensor<1x197x4096xbf16>
    %1367 = stablehlo.multiply %1365, %178 : tensor<1x197x4096xbf16>
    %1368 = stablehlo.convert %1367 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %1369 = stablehlo.clamp %cst_5, %1368, %cst_6 : tensor<1x197x4096xf32>
    %1370 = stablehlo.multiply %1369, %1369 : tensor<1x197x4096xf32>
    %1371 = stablehlo.multiply %cst_7, %1370 : tensor<1x197x4096xf32>
    %1372 = stablehlo.add %1371, %cst_8 : tensor<1x197x4096xf32>
    %1373 = stablehlo.multiply %1372, %1370 : tensor<1x197x4096xf32>
    %1374 = stablehlo.add %1373, %cst_9 : tensor<1x197x4096xf32>
    %1375 = stablehlo.multiply %1374, %1370 : tensor<1x197x4096xf32>
    %1376 = stablehlo.add %1375, %cst_10 : tensor<1x197x4096xf32>
    %1377 = stablehlo.multiply %1376, %1370 : tensor<1x197x4096xf32>
    %1378 = stablehlo.add %1377, %cst_11 : tensor<1x197x4096xf32>
    %1379 = stablehlo.multiply %1378, %1370 : tensor<1x197x4096xf32>
    %1380 = stablehlo.add %1379, %cst_12 : tensor<1x197x4096xf32>
    %1381 = stablehlo.multiply %1380, %1370 : tensor<1x197x4096xf32>
    %1382 = stablehlo.add %1381, %cst_13 : tensor<1x197x4096xf32>
    %1383 = stablehlo.multiply %cst_14, %1370 : tensor<1x197x4096xf32>
    %1384 = stablehlo.add %1383, %cst_15 : tensor<1x197x4096xf32>
    %1385 = stablehlo.multiply %1384, %1370 : tensor<1x197x4096xf32>
    %1386 = stablehlo.add %1385, %cst_16 : tensor<1x197x4096xf32>
    %1387 = stablehlo.multiply %1386, %1370 : tensor<1x197x4096xf32>
    %1388 = stablehlo.add %1387, %cst_17 : tensor<1x197x4096xf32>
    %1389 = stablehlo.multiply %1388, %1370 : tensor<1x197x4096xf32>
    %1390 = stablehlo.add %1389, %cst_18 : tensor<1x197x4096xf32>
    %1391 = stablehlo.multiply %1369, %1382 : tensor<1x197x4096xf32>
    %1392 = stablehlo.divide %1391, %1390 : tensor<1x197x4096xf32>
    %1393 = stablehlo.clamp %cst_19, %1392, %cst_20 : tensor<1x197x4096xf32>
    %1394 = stablehlo.convert %1393 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %1395 = stablehlo.add %1394, %cst_2 : tensor<1x197x4096xbf16>
    %1396 = stablehlo.multiply %1395, %1366 : tensor<1x197x4096xbf16>
    %1397 = stablehlo.reshape %1396 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %1398 = stablehlo.convert %1397 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %1399 = stablehlo.dot_general %1398, %arg232, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %1400 = stablehlo.broadcast_in_dim %1399, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1401 = stablehlo.multiply %1400, %60 : tensor<197x1024xf32>
    %1402 = stablehlo.broadcast_in_dim %1401, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1403 = stablehlo.broadcast_in_dim %arg233, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1404 = stablehlo.add %1402, %1403 : tensor<197x1024xf32>
    %1405 = stablehlo.convert %1404 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1406 = stablehlo.reshape %1405 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1407 = stablehlo.broadcast_in_dim %arg44, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1408 = stablehlo.broadcast_in_dim %1406, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1409 = stablehlo.multiply %1407, %1408 : tensor<1x197x1024xbf16>
    %1410 = stablehlo.add %1409, %1318 : tensor<1x197x1024xbf16>
    %1411 = stablehlo.convert %1410 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1412 = stablehlo.convert %1411 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1413 = stablehlo.reduce(%1412 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1414 = stablehlo.reshape %1413 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1416 = stablehlo.divide %1415, %15 : tensor<1x197x1xf64>
    %1417 = stablehlo.broadcast_in_dim %1412, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1418 = stablehlo.broadcast_in_dim %1416, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1419 = stablehlo.subtract %1417, %1418 : tensor<1x197x1024xf64>
    %1420 = stablehlo.multiply %1419, %1419 : tensor<1x197x1024xf64>
    %1421 = stablehlo.reduce(%1420 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1422 = stablehlo.reshape %1421 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1424 = stablehlo.divide %1423, %15 : tensor<1x197x1xf64>
    %1425 = stablehlo.convert %1424 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1426 = stablehlo.reduce(%1411 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1427 = stablehlo.reshape %1426 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1429 = stablehlo.divide %1428, %31 : tensor<1x197x1xf32>
    %1430 = stablehlo.broadcast_in_dim %1425, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1431 = stablehlo.add %1430, %36 : tensor<1x197x1xf32>
    %1432 = stablehlo.rsqrt %1431 : tensor<1x197x1xf32>
    %1433 = stablehlo.broadcast_in_dim %1411, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1434 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1435 = stablehlo.subtract %1433, %1434 : tensor<1x197x1024xf32>
    %1436 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1437 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1438 = stablehlo.multiply %1436, %1437 : tensor<1x197x1024xf32>
    %1439 = stablehlo.convert %arg45 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1440 = stablehlo.broadcast_in_dim %1438, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1441 = stablehlo.broadcast_in_dim %1439, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1442 = stablehlo.multiply %1440, %1441 : tensor<1x197x1024xf32>
    %1443 = stablehlo.convert %arg46 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1444 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1445 = stablehlo.broadcast_in_dim %1443, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1446 = stablehlo.add %1444, %1445 : tensor<1x197x1024xf32>
    %1447 = stablehlo.convert %1446 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1448 = stablehlo.reshape %1447 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1449 = stablehlo.convert %1448 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1450 = stablehlo.dot_general %1449, %arg234, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1451 = stablehlo.broadcast_in_dim %1450, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1452 = stablehlo.multiply %1451, %60 : tensor<197x1024xf32>
    %1453 = stablehlo.broadcast_in_dim %1452, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1454 = stablehlo.broadcast_in_dim %arg235, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1455 = stablehlo.add %1453, %1454 : tensor<197x1024xf32>
    %1456 = stablehlo.convert %1455 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1457 = stablehlo.reshape %1456 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1458 = stablehlo.dot_general %1448, %arg236, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %1459 = stablehlo.reshape %1458 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1460 = stablehlo.reshape %1459 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1461 = stablehlo.transpose %1460, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1462 = stablehlo.dot_general %1449, %arg237, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1464 = stablehlo.multiply %1463, %60 : tensor<197x1024xf32>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1466 = stablehlo.broadcast_in_dim %arg238, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1467 = stablehlo.add %1465, %1466 : tensor<197x1024xf32>
    %1468 = stablehlo.convert %1467 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1469 = stablehlo.reshape %1468 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1470 = stablehlo.reshape %1469 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1471 = stablehlo.transpose %1470, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1472 = stablehlo.reshape %1457 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1473 = stablehlo.transpose %1472, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1474 = stablehlo.transpose %1461, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %1475 = stablehlo.reshape %1473 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1476 = stablehlo.reshape %1474 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1477 = stablehlo.broadcast_in_dim %1476, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1478 = stablehlo.dot_general %1475, %1477, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %1479 = stablehlo.reshape %1478 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1480 = stablehlo.broadcast_in_dim %1479, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1481 = stablehlo.divide %1480, %92 : tensor<1x16x197x197xbf16>
    %1482 = stablehlo.add %1481, %arg239 : tensor<1x16x197x197xbf16>
    %1483 = stablehlo.convert %1482 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %1484 = stablehlo.reduce(%1483 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1485 = stablehlo.reshape %1484 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1486 = stablehlo.broadcast_in_dim %1483, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1487 = stablehlo.broadcast_in_dim %1485, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1488 = stablehlo.subtract %1486, %1487 : tensor<1x16x197x197xf32>
    %1489 = stablehlo.exponential %1488 : tensor<1x16x197x197xf32>
    %1490 = stablehlo.reduce(%1489 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1491 = stablehlo.reshape %1490 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1492 = stablehlo.broadcast_in_dim %1489, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1493 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1494 = stablehlo.divide %1492, %1493 : tensor<1x16x197x197xf32>
    %1495 = stablehlo.convert %1494 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %1496 = stablehlo.reshape %1495 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %1497 = stablehlo.reshape %1471 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1498 = stablehlo.broadcast_in_dim %1497, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1499 = stablehlo.dot_general %1496, %1498, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1500 = stablehlo.reshape %1499 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1501 = stablehlo.transpose %1500, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %1502 = stablehlo.reshape %1501 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %1503 = stablehlo.reshape %1502 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1504 = stablehlo.convert %1503 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1505 = stablehlo.dot_general %1504, %arg240, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1506 = stablehlo.broadcast_in_dim %1505, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1507 = stablehlo.multiply %1506, %60 : tensor<197x1024xf32>
    %1508 = stablehlo.broadcast_in_dim %1507, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1509 = stablehlo.broadcast_in_dim %arg241, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1510 = stablehlo.add %1508, %1509 : tensor<197x1024xf32>
    %1511 = stablehlo.convert %1510 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1512 = stablehlo.reshape %1511 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1513 = stablehlo.broadcast_in_dim %arg47, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1514 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1515 = stablehlo.multiply %1513, %1514 : tensor<1x197x1024xbf16>
    %1516 = stablehlo.add %1515, %1410 : tensor<1x197x1024xbf16>
    %1517 = stablehlo.convert %1516 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1518 = stablehlo.convert %1517 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1519 = stablehlo.reduce(%1518 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1520 = stablehlo.reshape %1519 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1521 = stablehlo.broadcast_in_dim %1520, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1522 = stablehlo.divide %1521, %15 : tensor<1x197x1xf64>
    %1523 = stablehlo.broadcast_in_dim %1518, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1524 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1525 = stablehlo.subtract %1523, %1524 : tensor<1x197x1024xf64>
    %1526 = stablehlo.multiply %1525, %1525 : tensor<1x197x1024xf64>
    %1527 = stablehlo.reduce(%1526 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1528 = stablehlo.reshape %1527 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1529 = stablehlo.broadcast_in_dim %1528, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1530 = stablehlo.divide %1529, %15 : tensor<1x197x1xf64>
    %1531 = stablehlo.convert %1530 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1532 = stablehlo.reduce(%1517 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1533 = stablehlo.reshape %1532 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1534 = stablehlo.broadcast_in_dim %1533, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1535 = stablehlo.divide %1534, %31 : tensor<1x197x1xf32>
    %1536 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1537 = stablehlo.add %1536, %36 : tensor<1x197x1xf32>
    %1538 = stablehlo.rsqrt %1537 : tensor<1x197x1xf32>
    %1539 = stablehlo.broadcast_in_dim %1517, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1540 = stablehlo.broadcast_in_dim %1535, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1541 = stablehlo.subtract %1539, %1540 : tensor<1x197x1024xf32>
    %1542 = stablehlo.broadcast_in_dim %1541, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1543 = stablehlo.broadcast_in_dim %1538, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1544 = stablehlo.multiply %1542, %1543 : tensor<1x197x1024xf32>
    %1545 = stablehlo.convert %arg48 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1546 = stablehlo.broadcast_in_dim %1544, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1547 = stablehlo.broadcast_in_dim %1545, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1548 = stablehlo.multiply %1546, %1547 : tensor<1x197x1024xf32>
    %1549 = stablehlo.convert %arg49 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1550 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1551 = stablehlo.broadcast_in_dim %1549, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1552 = stablehlo.add %1550, %1551 : tensor<1x197x1024xf32>
    %1553 = stablehlo.convert %1552 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1554 = stablehlo.reshape %1553 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1555 = stablehlo.convert %1554 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1556 = stablehlo.dot_general %1555, %arg242, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %1557 = stablehlo.broadcast_in_dim %1556, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1558 = stablehlo.multiply %1557, %170 : tensor<197x4096xf32>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1560 = stablehlo.broadcast_in_dim %arg243, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %1561 = stablehlo.add %1559, %1560 : tensor<197x4096xf32>
    %1562 = stablehlo.convert %1561 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %1563 = stablehlo.reshape %1562 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %1564 = stablehlo.multiply %1563, %cst_4 : tensor<1x197x4096xbf16>
    %1565 = stablehlo.multiply %1563, %178 : tensor<1x197x4096xbf16>
    %1566 = stablehlo.convert %1565 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %1567 = stablehlo.clamp %cst_5, %1566, %cst_6 : tensor<1x197x4096xf32>
    %1568 = stablehlo.multiply %1567, %1567 : tensor<1x197x4096xf32>
    %1569 = stablehlo.multiply %cst_7, %1568 : tensor<1x197x4096xf32>
    %1570 = stablehlo.add %1569, %cst_8 : tensor<1x197x4096xf32>
    %1571 = stablehlo.multiply %1570, %1568 : tensor<1x197x4096xf32>
    %1572 = stablehlo.add %1571, %cst_9 : tensor<1x197x4096xf32>
    %1573 = stablehlo.multiply %1572, %1568 : tensor<1x197x4096xf32>
    %1574 = stablehlo.add %1573, %cst_10 : tensor<1x197x4096xf32>
    %1575 = stablehlo.multiply %1574, %1568 : tensor<1x197x4096xf32>
    %1576 = stablehlo.add %1575, %cst_11 : tensor<1x197x4096xf32>
    %1577 = stablehlo.multiply %1576, %1568 : tensor<1x197x4096xf32>
    %1578 = stablehlo.add %1577, %cst_12 : tensor<1x197x4096xf32>
    %1579 = stablehlo.multiply %1578, %1568 : tensor<1x197x4096xf32>
    %1580 = stablehlo.add %1579, %cst_13 : tensor<1x197x4096xf32>
    %1581 = stablehlo.multiply %cst_14, %1568 : tensor<1x197x4096xf32>
    %1582 = stablehlo.add %1581, %cst_15 : tensor<1x197x4096xf32>
    %1583 = stablehlo.multiply %1582, %1568 : tensor<1x197x4096xf32>
    %1584 = stablehlo.add %1583, %cst_16 : tensor<1x197x4096xf32>
    %1585 = stablehlo.multiply %1584, %1568 : tensor<1x197x4096xf32>
    %1586 = stablehlo.add %1585, %cst_17 : tensor<1x197x4096xf32>
    %1587 = stablehlo.multiply %1586, %1568 : tensor<1x197x4096xf32>
    %1588 = stablehlo.add %1587, %cst_18 : tensor<1x197x4096xf32>
    %1589 = stablehlo.multiply %1567, %1580 : tensor<1x197x4096xf32>
    %1590 = stablehlo.divide %1589, %1588 : tensor<1x197x4096xf32>
    %1591 = stablehlo.clamp %cst_19, %1590, %cst_20 : tensor<1x197x4096xf32>
    %1592 = stablehlo.convert %1591 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %1593 = stablehlo.add %1592, %cst_2 : tensor<1x197x4096xbf16>
    %1594 = stablehlo.multiply %1593, %1564 : tensor<1x197x4096xbf16>
    %1595 = stablehlo.reshape %1594 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %1596 = stablehlo.convert %1595 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %1597 = stablehlo.dot_general %1596, %arg244, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %1598 = stablehlo.broadcast_in_dim %1597, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1599 = stablehlo.multiply %1598, %60 : tensor<197x1024xf32>
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1601 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1602 = stablehlo.add %1600, %1601 : tensor<197x1024xf32>
    %1603 = stablehlo.convert %1602 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1604 = stablehlo.reshape %1603 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1605 = stablehlo.broadcast_in_dim %arg50, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1606 = stablehlo.broadcast_in_dim %1604, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1607 = stablehlo.multiply %1605, %1606 : tensor<1x197x1024xbf16>
    %1608 = stablehlo.add %1607, %1516 : tensor<1x197x1024xbf16>
    %1609 = stablehlo.convert %1608 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1610 = stablehlo.convert %1609 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1611 = stablehlo.reduce(%1610 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1612 = stablehlo.reshape %1611 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1613 = stablehlo.broadcast_in_dim %1612, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1614 = stablehlo.divide %1613, %15 : tensor<1x197x1xf64>
    %1615 = stablehlo.broadcast_in_dim %1610, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1616 = stablehlo.broadcast_in_dim %1614, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1617 = stablehlo.subtract %1615, %1616 : tensor<1x197x1024xf64>
    %1618 = stablehlo.multiply %1617, %1617 : tensor<1x197x1024xf64>
    %1619 = stablehlo.reduce(%1618 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1620 = stablehlo.reshape %1619 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1621 = stablehlo.broadcast_in_dim %1620, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1622 = stablehlo.divide %1621, %15 : tensor<1x197x1xf64>
    %1623 = stablehlo.convert %1622 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1624 = stablehlo.reduce(%1609 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1625 = stablehlo.reshape %1624 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1626 = stablehlo.broadcast_in_dim %1625, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1627 = stablehlo.divide %1626, %31 : tensor<1x197x1xf32>
    %1628 = stablehlo.broadcast_in_dim %1623, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1629 = stablehlo.add %1628, %36 : tensor<1x197x1xf32>
    %1630 = stablehlo.rsqrt %1629 : tensor<1x197x1xf32>
    %1631 = stablehlo.broadcast_in_dim %1609, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1632 = stablehlo.broadcast_in_dim %1627, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1633 = stablehlo.subtract %1631, %1632 : tensor<1x197x1024xf32>
    %1634 = stablehlo.broadcast_in_dim %1633, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1635 = stablehlo.broadcast_in_dim %1630, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1636 = stablehlo.multiply %1634, %1635 : tensor<1x197x1024xf32>
    %1637 = stablehlo.convert %arg51 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1638 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1639 = stablehlo.broadcast_in_dim %1637, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1640 = stablehlo.multiply %1638, %1639 : tensor<1x197x1024xf32>
    %1641 = stablehlo.convert %arg52 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1642 = stablehlo.broadcast_in_dim %1640, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1643 = stablehlo.broadcast_in_dim %1641, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1644 = stablehlo.add %1642, %1643 : tensor<1x197x1024xf32>
    %1645 = stablehlo.convert %1644 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1646 = stablehlo.reshape %1645 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1647 = stablehlo.convert %1646 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1648 = stablehlo.dot_general %1647, %arg246, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1649 = stablehlo.broadcast_in_dim %1648, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1650 = stablehlo.multiply %1649, %60 : tensor<197x1024xf32>
    %1651 = stablehlo.broadcast_in_dim %1650, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1652 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1653 = stablehlo.add %1651, %1652 : tensor<197x1024xf32>
    %1654 = stablehlo.convert %1653 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1655 = stablehlo.reshape %1654 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1656 = stablehlo.dot_general %1646, %arg248, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %1657 = stablehlo.reshape %1656 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1658 = stablehlo.reshape %1657 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1659 = stablehlo.transpose %1658, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1660 = stablehlo.dot_general %1647, %arg249, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1661 = stablehlo.broadcast_in_dim %1660, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1662 = stablehlo.multiply %1661, %60 : tensor<197x1024xf32>
    %1663 = stablehlo.broadcast_in_dim %1662, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1664 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1665 = stablehlo.add %1663, %1664 : tensor<197x1024xf32>
    %1666 = stablehlo.convert %1665 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1667 = stablehlo.reshape %1666 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1668 = stablehlo.reshape %1667 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1669 = stablehlo.transpose %1668, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1670 = stablehlo.reshape %1655 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1671 = stablehlo.transpose %1670, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1672 = stablehlo.transpose %1659, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %1673 = stablehlo.reshape %1671 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1674 = stablehlo.reshape %1672 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1676 = stablehlo.dot_general %1673, %1675, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %1677 = stablehlo.reshape %1676 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1678 = stablehlo.broadcast_in_dim %1677, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1679 = stablehlo.divide %1678, %92 : tensor<1x16x197x197xbf16>
    %1680 = stablehlo.add %1679, %arg251 : tensor<1x16x197x197xbf16>
    %1681 = stablehlo.convert %1680 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %1682 = stablehlo.reduce(%1681 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1683 = stablehlo.reshape %1682 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1684 = stablehlo.broadcast_in_dim %1681, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1685 = stablehlo.broadcast_in_dim %1683, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1686 = stablehlo.subtract %1684, %1685 : tensor<1x16x197x197xf32>
    %1687 = stablehlo.exponential %1686 : tensor<1x16x197x197xf32>
    %1688 = stablehlo.reduce(%1687 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1689 = stablehlo.reshape %1688 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1690 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1691 = stablehlo.broadcast_in_dim %1689, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1692 = stablehlo.divide %1690, %1691 : tensor<1x16x197x197xf32>
    %1693 = stablehlo.convert %1692 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %1694 = stablehlo.reshape %1693 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %1695 = stablehlo.reshape %1669 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1697 = stablehlo.dot_general %1694, %1696, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1698 = stablehlo.reshape %1697 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1699 = stablehlo.transpose %1698, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %1700 = stablehlo.reshape %1699 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %1701 = stablehlo.reshape %1700 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1702 = stablehlo.convert %1701 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1703 = stablehlo.dot_general %1702, %arg252, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1704 = stablehlo.broadcast_in_dim %1703, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1705 = stablehlo.multiply %1704, %60 : tensor<197x1024xf32>
    %1706 = stablehlo.broadcast_in_dim %1705, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1707 = stablehlo.broadcast_in_dim %arg253, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1708 = stablehlo.add %1706, %1707 : tensor<197x1024xf32>
    %1709 = stablehlo.convert %1708 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1710 = stablehlo.reshape %1709 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1711 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1712 = stablehlo.broadcast_in_dim %1710, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1713 = stablehlo.multiply %1711, %1712 : tensor<1x197x1024xbf16>
    %1714 = stablehlo.add %1713, %1608 : tensor<1x197x1024xbf16>
    %1715 = stablehlo.convert %1714 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1716 = stablehlo.convert %1715 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1717 = stablehlo.reduce(%1716 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1718 = stablehlo.reshape %1717 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1719 = stablehlo.broadcast_in_dim %1718, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1720 = stablehlo.divide %1719, %15 : tensor<1x197x1xf64>
    %1721 = stablehlo.broadcast_in_dim %1716, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1722 = stablehlo.broadcast_in_dim %1720, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1723 = stablehlo.subtract %1721, %1722 : tensor<1x197x1024xf64>
    %1724 = stablehlo.multiply %1723, %1723 : tensor<1x197x1024xf64>
    %1725 = stablehlo.reduce(%1724 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1726 = stablehlo.reshape %1725 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1727 = stablehlo.broadcast_in_dim %1726, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1728 = stablehlo.divide %1727, %15 : tensor<1x197x1xf64>
    %1729 = stablehlo.convert %1728 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1730 = stablehlo.reduce(%1715 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1731 = stablehlo.reshape %1730 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1733 = stablehlo.divide %1732, %31 : tensor<1x197x1xf32>
    %1734 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1735 = stablehlo.add %1734, %36 : tensor<1x197x1xf32>
    %1736 = stablehlo.rsqrt %1735 : tensor<1x197x1xf32>
    %1737 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1738 = stablehlo.broadcast_in_dim %1733, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1739 = stablehlo.subtract %1737, %1738 : tensor<1x197x1024xf32>
    %1740 = stablehlo.broadcast_in_dim %1739, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1741 = stablehlo.broadcast_in_dim %1736, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1742 = stablehlo.multiply %1740, %1741 : tensor<1x197x1024xf32>
    %1743 = stablehlo.convert %arg54 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1744 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1745 = stablehlo.broadcast_in_dim %1743, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1746 = stablehlo.multiply %1744, %1745 : tensor<1x197x1024xf32>
    %1747 = stablehlo.convert %arg55 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1748 = stablehlo.broadcast_in_dim %1746, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1749 = stablehlo.broadcast_in_dim %1747, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1750 = stablehlo.add %1748, %1749 : tensor<1x197x1024xf32>
    %1751 = stablehlo.convert %1750 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1752 = stablehlo.reshape %1751 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1753 = stablehlo.convert %1752 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1754 = stablehlo.dot_general %1753, %arg254, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %1755 = stablehlo.broadcast_in_dim %1754, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1756 = stablehlo.multiply %1755, %170 : tensor<197x4096xf32>
    %1757 = stablehlo.broadcast_in_dim %1756, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1758 = stablehlo.broadcast_in_dim %arg255, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %1759 = stablehlo.add %1757, %1758 : tensor<197x4096xf32>
    %1760 = stablehlo.convert %1759 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %1761 = stablehlo.reshape %1760 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %1762 = stablehlo.multiply %1761, %cst_4 : tensor<1x197x4096xbf16>
    %1763 = stablehlo.multiply %1761, %178 : tensor<1x197x4096xbf16>
    %1764 = stablehlo.convert %1763 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %1765 = stablehlo.clamp %cst_5, %1764, %cst_6 : tensor<1x197x4096xf32>
    %1766 = stablehlo.multiply %1765, %1765 : tensor<1x197x4096xf32>
    %1767 = stablehlo.multiply %cst_7, %1766 : tensor<1x197x4096xf32>
    %1768 = stablehlo.add %1767, %cst_8 : tensor<1x197x4096xf32>
    %1769 = stablehlo.multiply %1768, %1766 : tensor<1x197x4096xf32>
    %1770 = stablehlo.add %1769, %cst_9 : tensor<1x197x4096xf32>
    %1771 = stablehlo.multiply %1770, %1766 : tensor<1x197x4096xf32>
    %1772 = stablehlo.add %1771, %cst_10 : tensor<1x197x4096xf32>
    %1773 = stablehlo.multiply %1772, %1766 : tensor<1x197x4096xf32>
    %1774 = stablehlo.add %1773, %cst_11 : tensor<1x197x4096xf32>
    %1775 = stablehlo.multiply %1774, %1766 : tensor<1x197x4096xf32>
    %1776 = stablehlo.add %1775, %cst_12 : tensor<1x197x4096xf32>
    %1777 = stablehlo.multiply %1776, %1766 : tensor<1x197x4096xf32>
    %1778 = stablehlo.add %1777, %cst_13 : tensor<1x197x4096xf32>
    %1779 = stablehlo.multiply %cst_14, %1766 : tensor<1x197x4096xf32>
    %1780 = stablehlo.add %1779, %cst_15 : tensor<1x197x4096xf32>
    %1781 = stablehlo.multiply %1780, %1766 : tensor<1x197x4096xf32>
    %1782 = stablehlo.add %1781, %cst_16 : tensor<1x197x4096xf32>
    %1783 = stablehlo.multiply %1782, %1766 : tensor<1x197x4096xf32>
    %1784 = stablehlo.add %1783, %cst_17 : tensor<1x197x4096xf32>
    %1785 = stablehlo.multiply %1784, %1766 : tensor<1x197x4096xf32>
    %1786 = stablehlo.add %1785, %cst_18 : tensor<1x197x4096xf32>
    %1787 = stablehlo.multiply %1765, %1778 : tensor<1x197x4096xf32>
    %1788 = stablehlo.divide %1787, %1786 : tensor<1x197x4096xf32>
    %1789 = stablehlo.clamp %cst_19, %1788, %cst_20 : tensor<1x197x4096xf32>
    %1790 = stablehlo.convert %1789 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %1791 = stablehlo.add %1790, %cst_2 : tensor<1x197x4096xbf16>
    %1792 = stablehlo.multiply %1791, %1762 : tensor<1x197x4096xbf16>
    %1793 = stablehlo.reshape %1792 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %1794 = stablehlo.convert %1793 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %1795 = stablehlo.dot_general %1794, %arg256, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1797 = stablehlo.multiply %1796, %60 : tensor<197x1024xf32>
    %1798 = stablehlo.broadcast_in_dim %1797, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1799 = stablehlo.broadcast_in_dim %arg257, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1800 = stablehlo.add %1798, %1799 : tensor<197x1024xf32>
    %1801 = stablehlo.convert %1800 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1802 = stablehlo.reshape %1801 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1803 = stablehlo.broadcast_in_dim %arg56, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1804 = stablehlo.broadcast_in_dim %1802, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1805 = stablehlo.multiply %1803, %1804 : tensor<1x197x1024xbf16>
    %1806 = stablehlo.add %1805, %1714 : tensor<1x197x1024xbf16>
    %1807 = stablehlo.convert %1806 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1808 = stablehlo.convert %1807 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1809 = stablehlo.reduce(%1808 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1810 = stablehlo.reshape %1809 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1811 = stablehlo.broadcast_in_dim %1810, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1812 = stablehlo.divide %1811, %15 : tensor<1x197x1xf64>
    %1813 = stablehlo.broadcast_in_dim %1808, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1814 = stablehlo.broadcast_in_dim %1812, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1815 = stablehlo.subtract %1813, %1814 : tensor<1x197x1024xf64>
    %1816 = stablehlo.multiply %1815, %1815 : tensor<1x197x1024xf64>
    %1817 = stablehlo.reduce(%1816 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1818 = stablehlo.reshape %1817 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1820 = stablehlo.divide %1819, %15 : tensor<1x197x1xf64>
    %1821 = stablehlo.convert %1820 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1822 = stablehlo.reduce(%1807 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1823 = stablehlo.reshape %1822 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1824 = stablehlo.broadcast_in_dim %1823, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1825 = stablehlo.divide %1824, %31 : tensor<1x197x1xf32>
    %1826 = stablehlo.broadcast_in_dim %1821, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1827 = stablehlo.add %1826, %36 : tensor<1x197x1xf32>
    %1828 = stablehlo.rsqrt %1827 : tensor<1x197x1xf32>
    %1829 = stablehlo.broadcast_in_dim %1807, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1830 = stablehlo.broadcast_in_dim %1825, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1831 = stablehlo.subtract %1829, %1830 : tensor<1x197x1024xf32>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1833 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1834 = stablehlo.multiply %1832, %1833 : tensor<1x197x1024xf32>
    %1835 = stablehlo.convert %arg57 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1836 = stablehlo.broadcast_in_dim %1834, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1837 = stablehlo.broadcast_in_dim %1835, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1838 = stablehlo.multiply %1836, %1837 : tensor<1x197x1024xf32>
    %1839 = stablehlo.convert %arg58 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1840 = stablehlo.broadcast_in_dim %1838, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1841 = stablehlo.broadcast_in_dim %1839, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1842 = stablehlo.add %1840, %1841 : tensor<1x197x1024xf32>
    %1843 = stablehlo.convert %1842 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1844 = stablehlo.reshape %1843 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1845 = stablehlo.convert %1844 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1846 = stablehlo.dot_general %1845, %arg258, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1847 = stablehlo.broadcast_in_dim %1846, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1848 = stablehlo.multiply %1847, %60 : tensor<197x1024xf32>
    %1849 = stablehlo.broadcast_in_dim %1848, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1850 = stablehlo.broadcast_in_dim %arg259, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1851 = stablehlo.add %1849, %1850 : tensor<197x1024xf32>
    %1852 = stablehlo.convert %1851 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1853 = stablehlo.reshape %1852 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1854 = stablehlo.dot_general %1844, %arg260, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %1855 = stablehlo.reshape %1854 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1856 = stablehlo.reshape %1855 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1857 = stablehlo.transpose %1856, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1858 = stablehlo.dot_general %1845, %arg261, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1859 = stablehlo.broadcast_in_dim %1858, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1860 = stablehlo.multiply %1859, %60 : tensor<197x1024xf32>
    %1861 = stablehlo.broadcast_in_dim %1860, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1862 = stablehlo.broadcast_in_dim %arg262, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1863 = stablehlo.add %1861, %1862 : tensor<197x1024xf32>
    %1864 = stablehlo.convert %1863 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1865 = stablehlo.reshape %1864 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1866 = stablehlo.reshape %1865 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1867 = stablehlo.transpose %1866, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1868 = stablehlo.reshape %1853 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %1869 = stablehlo.transpose %1868, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1870 = stablehlo.transpose %1857, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %1871 = stablehlo.reshape %1869 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1872 = stablehlo.reshape %1870 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1873 = stablehlo.broadcast_in_dim %1872, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %1874 = stablehlo.dot_general %1871, %1873, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %1875 = stablehlo.reshape %1874 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1876 = stablehlo.broadcast_in_dim %1875, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %1877 = stablehlo.divide %1876, %92 : tensor<1x16x197x197xbf16>
    %1878 = stablehlo.add %1877, %arg263 : tensor<1x16x197x197xbf16>
    %1879 = stablehlo.convert %1878 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %1880 = stablehlo.reduce(%1879 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1881 = stablehlo.reshape %1880 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1882 = stablehlo.broadcast_in_dim %1879, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1883 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1884 = stablehlo.subtract %1882, %1883 : tensor<1x16x197x197xf32>
    %1885 = stablehlo.exponential %1884 : tensor<1x16x197x197xf32>
    %1886 = stablehlo.reduce(%1885 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %1887 = stablehlo.reshape %1886 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %1888 = stablehlo.broadcast_in_dim %1885, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %1889 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %1890 = stablehlo.divide %1888, %1889 : tensor<1x16x197x197xf32>
    %1891 = stablehlo.convert %1890 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %1892 = stablehlo.reshape %1891 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %1893 = stablehlo.reshape %1867 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1894 = stablehlo.broadcast_in_dim %1893, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1895 = stablehlo.dot_general %1892, %1894, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %1896 = stablehlo.reshape %1895 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %1897 = stablehlo.transpose %1896, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %1898 = stablehlo.reshape %1897 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %1899 = stablehlo.reshape %1898 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1900 = stablehlo.convert %1899 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1901 = stablehlo.dot_general %1900, %arg264, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %1902 = stablehlo.broadcast_in_dim %1901, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1903 = stablehlo.multiply %1902, %60 : tensor<197x1024xf32>
    %1904 = stablehlo.broadcast_in_dim %1903, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1905 = stablehlo.broadcast_in_dim %arg265, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1906 = stablehlo.add %1904, %1905 : tensor<197x1024xf32>
    %1907 = stablehlo.convert %1906 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %1908 = stablehlo.reshape %1907 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1909 = stablehlo.broadcast_in_dim %arg59, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %1910 = stablehlo.broadcast_in_dim %1908, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %1911 = stablehlo.multiply %1909, %1910 : tensor<1x197x1024xbf16>
    %1912 = stablehlo.add %1911, %1806 : tensor<1x197x1024xbf16>
    %1913 = stablehlo.convert %1912 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %1914 = stablehlo.convert %1913 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %1915 = stablehlo.reduce(%1914 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1916 = stablehlo.reshape %1915 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1917 = stablehlo.broadcast_in_dim %1916, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1918 = stablehlo.divide %1917, %15 : tensor<1x197x1xf64>
    %1919 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %1920 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %1921 = stablehlo.subtract %1919, %1920 : tensor<1x197x1024xf64>
    %1922 = stablehlo.multiply %1921, %1921 : tensor<1x197x1024xf64>
    %1923 = stablehlo.reduce(%1922 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %1924 = stablehlo.reshape %1923 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %1925 = stablehlo.broadcast_in_dim %1924, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %1926 = stablehlo.divide %1925, %15 : tensor<1x197x1xf64>
    %1927 = stablehlo.convert %1926 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %1928 = stablehlo.reduce(%1913 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %1929 = stablehlo.reshape %1928 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %1930 = stablehlo.broadcast_in_dim %1929, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1931 = stablehlo.divide %1930, %31 : tensor<1x197x1xf32>
    %1932 = stablehlo.broadcast_in_dim %1927, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1933 = stablehlo.add %1932, %36 : tensor<1x197x1xf32>
    %1934 = stablehlo.rsqrt %1933 : tensor<1x197x1xf32>
    %1935 = stablehlo.broadcast_in_dim %1913, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1936 = stablehlo.broadcast_in_dim %1931, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1937 = stablehlo.subtract %1935, %1936 : tensor<1x197x1024xf32>
    %1938 = stablehlo.broadcast_in_dim %1937, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1939 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %1940 = stablehlo.multiply %1938, %1939 : tensor<1x197x1024xf32>
    %1941 = stablehlo.convert %arg60 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1942 = stablehlo.broadcast_in_dim %1940, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1943 = stablehlo.broadcast_in_dim %1941, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1944 = stablehlo.multiply %1942, %1943 : tensor<1x197x1024xf32>
    %1945 = stablehlo.convert %arg61 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %1946 = stablehlo.broadcast_in_dim %1944, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %1947 = stablehlo.broadcast_in_dim %1945, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %1948 = stablehlo.add %1946, %1947 : tensor<1x197x1024xf32>
    %1949 = stablehlo.convert %1948 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %1950 = stablehlo.reshape %1949 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %1951 = stablehlo.convert %1950 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %1952 = stablehlo.dot_general %1951, %arg266, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %1953 = stablehlo.broadcast_in_dim %1952, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1954 = stablehlo.multiply %1953, %170 : tensor<197x4096xf32>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %1956 = stablehlo.broadcast_in_dim %arg267, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %1957 = stablehlo.add %1955, %1956 : tensor<197x4096xf32>
    %1958 = stablehlo.convert %1957 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %1959 = stablehlo.reshape %1958 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %1960 = stablehlo.multiply %1959, %cst_4 : tensor<1x197x4096xbf16>
    %1961 = stablehlo.multiply %1959, %178 : tensor<1x197x4096xbf16>
    %1962 = stablehlo.convert %1961 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %1963 = stablehlo.clamp %cst_5, %1962, %cst_6 : tensor<1x197x4096xf32>
    %1964 = stablehlo.multiply %1963, %1963 : tensor<1x197x4096xf32>
    %1965 = stablehlo.multiply %cst_7, %1964 : tensor<1x197x4096xf32>
    %1966 = stablehlo.add %1965, %cst_8 : tensor<1x197x4096xf32>
    %1967 = stablehlo.multiply %1966, %1964 : tensor<1x197x4096xf32>
    %1968 = stablehlo.add %1967, %cst_9 : tensor<1x197x4096xf32>
    %1969 = stablehlo.multiply %1968, %1964 : tensor<1x197x4096xf32>
    %1970 = stablehlo.add %1969, %cst_10 : tensor<1x197x4096xf32>
    %1971 = stablehlo.multiply %1970, %1964 : tensor<1x197x4096xf32>
    %1972 = stablehlo.add %1971, %cst_11 : tensor<1x197x4096xf32>
    %1973 = stablehlo.multiply %1972, %1964 : tensor<1x197x4096xf32>
    %1974 = stablehlo.add %1973, %cst_12 : tensor<1x197x4096xf32>
    %1975 = stablehlo.multiply %1974, %1964 : tensor<1x197x4096xf32>
    %1976 = stablehlo.add %1975, %cst_13 : tensor<1x197x4096xf32>
    %1977 = stablehlo.multiply %cst_14, %1964 : tensor<1x197x4096xf32>
    %1978 = stablehlo.add %1977, %cst_15 : tensor<1x197x4096xf32>
    %1979 = stablehlo.multiply %1978, %1964 : tensor<1x197x4096xf32>
    %1980 = stablehlo.add %1979, %cst_16 : tensor<1x197x4096xf32>
    %1981 = stablehlo.multiply %1980, %1964 : tensor<1x197x4096xf32>
    %1982 = stablehlo.add %1981, %cst_17 : tensor<1x197x4096xf32>
    %1983 = stablehlo.multiply %1982, %1964 : tensor<1x197x4096xf32>
    %1984 = stablehlo.add %1983, %cst_18 : tensor<1x197x4096xf32>
    %1985 = stablehlo.multiply %1963, %1976 : tensor<1x197x4096xf32>
    %1986 = stablehlo.divide %1985, %1984 : tensor<1x197x4096xf32>
    %1987 = stablehlo.clamp %cst_19, %1986, %cst_20 : tensor<1x197x4096xf32>
    %1988 = stablehlo.convert %1987 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %1989 = stablehlo.add %1988, %cst_2 : tensor<1x197x4096xbf16>
    %1990 = stablehlo.multiply %1989, %1960 : tensor<1x197x4096xbf16>
    %1991 = stablehlo.reshape %1990 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %1992 = stablehlo.convert %1991 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %1993 = stablehlo.dot_general %1992, %arg268, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1995 = stablehlo.multiply %1994, %60 : tensor<197x1024xf32>
    %1996 = stablehlo.broadcast_in_dim %1995, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %1997 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %1998 = stablehlo.add %1996, %1997 : tensor<197x1024xf32>
    %1999 = stablehlo.convert %1998 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2000 = stablehlo.reshape %1999 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2001 = stablehlo.broadcast_in_dim %arg62, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2002 = stablehlo.broadcast_in_dim %2000, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2003 = stablehlo.multiply %2001, %2002 : tensor<1x197x1024xbf16>
    %2004 = stablehlo.add %2003, %1912 : tensor<1x197x1024xbf16>
    %2005 = stablehlo.convert %2004 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2006 = stablehlo.convert %2005 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2007 = stablehlo.reduce(%2006 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2008 = stablehlo.reshape %2007 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2009 = stablehlo.broadcast_in_dim %2008, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2010 = stablehlo.divide %2009, %15 : tensor<1x197x1xf64>
    %2011 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2012 = stablehlo.broadcast_in_dim %2010, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2013 = stablehlo.subtract %2011, %2012 : tensor<1x197x1024xf64>
    %2014 = stablehlo.multiply %2013, %2013 : tensor<1x197x1024xf64>
    %2015 = stablehlo.reduce(%2014 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2016 = stablehlo.reshape %2015 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2017 = stablehlo.broadcast_in_dim %2016, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2018 = stablehlo.divide %2017, %15 : tensor<1x197x1xf64>
    %2019 = stablehlo.convert %2018 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2020 = stablehlo.reduce(%2005 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2021 = stablehlo.reshape %2020 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2023 = stablehlo.divide %2022, %31 : tensor<1x197x1xf32>
    %2024 = stablehlo.broadcast_in_dim %2019, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2025 = stablehlo.add %2024, %36 : tensor<1x197x1xf32>
    %2026 = stablehlo.rsqrt %2025 : tensor<1x197x1xf32>
    %2027 = stablehlo.broadcast_in_dim %2005, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2028 = stablehlo.broadcast_in_dim %2023, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2029 = stablehlo.subtract %2027, %2028 : tensor<1x197x1024xf32>
    %2030 = stablehlo.broadcast_in_dim %2029, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2031 = stablehlo.broadcast_in_dim %2026, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2032 = stablehlo.multiply %2030, %2031 : tensor<1x197x1024xf32>
    %2033 = stablehlo.convert %arg63 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2034 = stablehlo.broadcast_in_dim %2032, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2035 = stablehlo.broadcast_in_dim %2033, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2036 = stablehlo.multiply %2034, %2035 : tensor<1x197x1024xf32>
    %2037 = stablehlo.convert %arg64 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2038 = stablehlo.broadcast_in_dim %2036, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2039 = stablehlo.broadcast_in_dim %2037, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2040 = stablehlo.add %2038, %2039 : tensor<1x197x1024xf32>
    %2041 = stablehlo.convert %2040 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2042 = stablehlo.reshape %2041 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2043 = stablehlo.convert %2042 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2044 = stablehlo.dot_general %2043, %arg270, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2045 = stablehlo.broadcast_in_dim %2044, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2046 = stablehlo.multiply %2045, %60 : tensor<197x1024xf32>
    %2047 = stablehlo.broadcast_in_dim %2046, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2048 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2049 = stablehlo.add %2047, %2048 : tensor<197x1024xf32>
    %2050 = stablehlo.convert %2049 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2051 = stablehlo.reshape %2050 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2052 = stablehlo.dot_general %2042, %arg272, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %2053 = stablehlo.reshape %2052 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2054 = stablehlo.reshape %2053 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2055 = stablehlo.transpose %2054, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2056 = stablehlo.dot_general %2043, %arg273, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2057 = stablehlo.broadcast_in_dim %2056, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2058 = stablehlo.multiply %2057, %60 : tensor<197x1024xf32>
    %2059 = stablehlo.broadcast_in_dim %2058, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2060 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2061 = stablehlo.add %2059, %2060 : tensor<197x1024xf32>
    %2062 = stablehlo.convert %2061 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2063 = stablehlo.reshape %2062 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2064 = stablehlo.reshape %2063 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2065 = stablehlo.transpose %2064, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2066 = stablehlo.reshape %2051 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2067 = stablehlo.transpose %2066, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2068 = stablehlo.transpose %2055, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %2069 = stablehlo.reshape %2067 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2070 = stablehlo.reshape %2068 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2071 = stablehlo.broadcast_in_dim %2070, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2072 = stablehlo.dot_general %2069, %2071, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %2073 = stablehlo.reshape %2072 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2074 = stablehlo.broadcast_in_dim %2073, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2075 = stablehlo.divide %2074, %92 : tensor<1x16x197x197xbf16>
    %2076 = stablehlo.add %2075, %arg275 : tensor<1x16x197x197xbf16>
    %2077 = stablehlo.convert %2076 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %2078 = stablehlo.reduce(%2077 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2079 = stablehlo.reshape %2078 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2080 = stablehlo.broadcast_in_dim %2077, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2081 = stablehlo.broadcast_in_dim %2079, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2082 = stablehlo.subtract %2080, %2081 : tensor<1x16x197x197xf32>
    %2083 = stablehlo.exponential %2082 : tensor<1x16x197x197xf32>
    %2084 = stablehlo.reduce(%2083 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2085 = stablehlo.reshape %2084 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2086 = stablehlo.broadcast_in_dim %2083, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2087 = stablehlo.broadcast_in_dim %2085, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2088 = stablehlo.divide %2086, %2087 : tensor<1x16x197x197xf32>
    %2089 = stablehlo.convert %2088 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %2090 = stablehlo.reshape %2089 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %2091 = stablehlo.reshape %2065 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2092 = stablehlo.broadcast_in_dim %2091, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2093 = stablehlo.dot_general %2090, %2092, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2094 = stablehlo.reshape %2093 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2095 = stablehlo.transpose %2094, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %2096 = stablehlo.reshape %2095 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %2097 = stablehlo.reshape %2096 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2098 = stablehlo.convert %2097 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2099 = stablehlo.dot_general %2098, %arg276, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2100 = stablehlo.broadcast_in_dim %2099, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2101 = stablehlo.multiply %2100, %60 : tensor<197x1024xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2103 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2104 = stablehlo.add %2102, %2103 : tensor<197x1024xf32>
    %2105 = stablehlo.convert %2104 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2106 = stablehlo.reshape %2105 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2107 = stablehlo.broadcast_in_dim %arg65, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2108 = stablehlo.broadcast_in_dim %2106, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2109 = stablehlo.multiply %2107, %2108 : tensor<1x197x1024xbf16>
    %2110 = stablehlo.add %2109, %2004 : tensor<1x197x1024xbf16>
    %2111 = stablehlo.convert %2110 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2112 = stablehlo.convert %2111 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2113 = stablehlo.reduce(%2112 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2114 = stablehlo.reshape %2113 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2115 = stablehlo.broadcast_in_dim %2114, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2116 = stablehlo.divide %2115, %15 : tensor<1x197x1xf64>
    %2117 = stablehlo.broadcast_in_dim %2112, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2118 = stablehlo.broadcast_in_dim %2116, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2119 = stablehlo.subtract %2117, %2118 : tensor<1x197x1024xf64>
    %2120 = stablehlo.multiply %2119, %2119 : tensor<1x197x1024xf64>
    %2121 = stablehlo.reduce(%2120 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2122 = stablehlo.reshape %2121 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2123 = stablehlo.broadcast_in_dim %2122, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2124 = stablehlo.divide %2123, %15 : tensor<1x197x1xf64>
    %2125 = stablehlo.convert %2124 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2126 = stablehlo.reduce(%2111 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2127 = stablehlo.reshape %2126 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2128 = stablehlo.broadcast_in_dim %2127, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2129 = stablehlo.divide %2128, %31 : tensor<1x197x1xf32>
    %2130 = stablehlo.broadcast_in_dim %2125, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2131 = stablehlo.add %2130, %36 : tensor<1x197x1xf32>
    %2132 = stablehlo.rsqrt %2131 : tensor<1x197x1xf32>
    %2133 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2134 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2135 = stablehlo.subtract %2133, %2134 : tensor<1x197x1024xf32>
    %2136 = stablehlo.broadcast_in_dim %2135, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2137 = stablehlo.broadcast_in_dim %2132, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2138 = stablehlo.multiply %2136, %2137 : tensor<1x197x1024xf32>
    %2139 = stablehlo.convert %arg66 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2140 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2141 = stablehlo.broadcast_in_dim %2139, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2142 = stablehlo.multiply %2140, %2141 : tensor<1x197x1024xf32>
    %2143 = stablehlo.convert %arg67 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2144 = stablehlo.broadcast_in_dim %2142, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2145 = stablehlo.broadcast_in_dim %2143, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2146 = stablehlo.add %2144, %2145 : tensor<1x197x1024xf32>
    %2147 = stablehlo.convert %2146 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2148 = stablehlo.reshape %2147 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2149 = stablehlo.convert %2148 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2150 = stablehlo.dot_general %2149, %arg278, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %2151 = stablehlo.broadcast_in_dim %2150, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2152 = stablehlo.multiply %2151, %170 : tensor<197x4096xf32>
    %2153 = stablehlo.broadcast_in_dim %2152, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2154 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %2155 = stablehlo.add %2153, %2154 : tensor<197x4096xf32>
    %2156 = stablehlo.convert %2155 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %2157 = stablehlo.reshape %2156 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %2158 = stablehlo.multiply %2157, %cst_4 : tensor<1x197x4096xbf16>
    %2159 = stablehlo.multiply %2157, %178 : tensor<1x197x4096xbf16>
    %2160 = stablehlo.convert %2159 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %2161 = stablehlo.clamp %cst_5, %2160, %cst_6 : tensor<1x197x4096xf32>
    %2162 = stablehlo.multiply %2161, %2161 : tensor<1x197x4096xf32>
    %2163 = stablehlo.multiply %cst_7, %2162 : tensor<1x197x4096xf32>
    %2164 = stablehlo.add %2163, %cst_8 : tensor<1x197x4096xf32>
    %2165 = stablehlo.multiply %2164, %2162 : tensor<1x197x4096xf32>
    %2166 = stablehlo.add %2165, %cst_9 : tensor<1x197x4096xf32>
    %2167 = stablehlo.multiply %2166, %2162 : tensor<1x197x4096xf32>
    %2168 = stablehlo.add %2167, %cst_10 : tensor<1x197x4096xf32>
    %2169 = stablehlo.multiply %2168, %2162 : tensor<1x197x4096xf32>
    %2170 = stablehlo.add %2169, %cst_11 : tensor<1x197x4096xf32>
    %2171 = stablehlo.multiply %2170, %2162 : tensor<1x197x4096xf32>
    %2172 = stablehlo.add %2171, %cst_12 : tensor<1x197x4096xf32>
    %2173 = stablehlo.multiply %2172, %2162 : tensor<1x197x4096xf32>
    %2174 = stablehlo.add %2173, %cst_13 : tensor<1x197x4096xf32>
    %2175 = stablehlo.multiply %cst_14, %2162 : tensor<1x197x4096xf32>
    %2176 = stablehlo.add %2175, %cst_15 : tensor<1x197x4096xf32>
    %2177 = stablehlo.multiply %2176, %2162 : tensor<1x197x4096xf32>
    %2178 = stablehlo.add %2177, %cst_16 : tensor<1x197x4096xf32>
    %2179 = stablehlo.multiply %2178, %2162 : tensor<1x197x4096xf32>
    %2180 = stablehlo.add %2179, %cst_17 : tensor<1x197x4096xf32>
    %2181 = stablehlo.multiply %2180, %2162 : tensor<1x197x4096xf32>
    %2182 = stablehlo.add %2181, %cst_18 : tensor<1x197x4096xf32>
    %2183 = stablehlo.multiply %2161, %2174 : tensor<1x197x4096xf32>
    %2184 = stablehlo.divide %2183, %2182 : tensor<1x197x4096xf32>
    %2185 = stablehlo.clamp %cst_19, %2184, %cst_20 : tensor<1x197x4096xf32>
    %2186 = stablehlo.convert %2185 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %2187 = stablehlo.add %2186, %cst_2 : tensor<1x197x4096xbf16>
    %2188 = stablehlo.multiply %2187, %2158 : tensor<1x197x4096xbf16>
    %2189 = stablehlo.reshape %2188 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %2190 = stablehlo.convert %2189 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %2191 = stablehlo.dot_general %2190, %arg280, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2193 = stablehlo.multiply %2192, %60 : tensor<197x1024xf32>
    %2194 = stablehlo.broadcast_in_dim %2193, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2195 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2196 = stablehlo.add %2194, %2195 : tensor<197x1024xf32>
    %2197 = stablehlo.convert %2196 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2198 = stablehlo.reshape %2197 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2199 = stablehlo.broadcast_in_dim %arg68, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2200 = stablehlo.broadcast_in_dim %2198, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2201 = stablehlo.multiply %2199, %2200 : tensor<1x197x1024xbf16>
    %2202 = stablehlo.add %2201, %2110 : tensor<1x197x1024xbf16>
    %2203 = stablehlo.convert %2202 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2204 = stablehlo.convert %2203 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2205 = stablehlo.reduce(%2204 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2206 = stablehlo.reshape %2205 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2207 = stablehlo.broadcast_in_dim %2206, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2208 = stablehlo.divide %2207, %15 : tensor<1x197x1xf64>
    %2209 = stablehlo.broadcast_in_dim %2204, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2210 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2211 = stablehlo.subtract %2209, %2210 : tensor<1x197x1024xf64>
    %2212 = stablehlo.multiply %2211, %2211 : tensor<1x197x1024xf64>
    %2213 = stablehlo.reduce(%2212 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2214 = stablehlo.reshape %2213 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2215 = stablehlo.broadcast_in_dim %2214, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2216 = stablehlo.divide %2215, %15 : tensor<1x197x1xf64>
    %2217 = stablehlo.convert %2216 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2218 = stablehlo.reduce(%2203 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2219 = stablehlo.reshape %2218 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2220 = stablehlo.broadcast_in_dim %2219, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2221 = stablehlo.divide %2220, %31 : tensor<1x197x1xf32>
    %2222 = stablehlo.broadcast_in_dim %2217, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2223 = stablehlo.add %2222, %36 : tensor<1x197x1xf32>
    %2224 = stablehlo.rsqrt %2223 : tensor<1x197x1xf32>
    %2225 = stablehlo.broadcast_in_dim %2203, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2226 = stablehlo.broadcast_in_dim %2221, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2227 = stablehlo.subtract %2225, %2226 : tensor<1x197x1024xf32>
    %2228 = stablehlo.broadcast_in_dim %2227, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2229 = stablehlo.broadcast_in_dim %2224, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2230 = stablehlo.multiply %2228, %2229 : tensor<1x197x1024xf32>
    %2231 = stablehlo.convert %arg69 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2232 = stablehlo.broadcast_in_dim %2230, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2233 = stablehlo.broadcast_in_dim %2231, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2234 = stablehlo.multiply %2232, %2233 : tensor<1x197x1024xf32>
    %2235 = stablehlo.convert %arg70 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2236 = stablehlo.broadcast_in_dim %2234, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2237 = stablehlo.broadcast_in_dim %2235, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2238 = stablehlo.add %2236, %2237 : tensor<1x197x1024xf32>
    %2239 = stablehlo.convert %2238 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2240 = stablehlo.reshape %2239 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2241 = stablehlo.convert %2240 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2242 = stablehlo.dot_general %2241, %arg282, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2243 = stablehlo.broadcast_in_dim %2242, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2244 = stablehlo.multiply %2243, %60 : tensor<197x1024xf32>
    %2245 = stablehlo.broadcast_in_dim %2244, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2246 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2247 = stablehlo.add %2245, %2246 : tensor<197x1024xf32>
    %2248 = stablehlo.convert %2247 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2249 = stablehlo.reshape %2248 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2250 = stablehlo.dot_general %2240, %arg284, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %2251 = stablehlo.reshape %2250 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2252 = stablehlo.reshape %2251 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2253 = stablehlo.transpose %2252, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2254 = stablehlo.dot_general %2241, %arg285, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2255 = stablehlo.broadcast_in_dim %2254, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2256 = stablehlo.multiply %2255, %60 : tensor<197x1024xf32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2258 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2259 = stablehlo.add %2257, %2258 : tensor<197x1024xf32>
    %2260 = stablehlo.convert %2259 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2261 = stablehlo.reshape %2260 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2262 = stablehlo.reshape %2261 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2263 = stablehlo.transpose %2262, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2264 = stablehlo.reshape %2249 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2265 = stablehlo.transpose %2264, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2266 = stablehlo.transpose %2253, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %2267 = stablehlo.reshape %2265 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2268 = stablehlo.reshape %2266 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2269 = stablehlo.broadcast_in_dim %2268, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2270 = stablehlo.dot_general %2267, %2269, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %2271 = stablehlo.reshape %2270 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2272 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2273 = stablehlo.divide %2272, %92 : tensor<1x16x197x197xbf16>
    %2274 = stablehlo.add %2273, %arg287 : tensor<1x16x197x197xbf16>
    %2275 = stablehlo.convert %2274 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %2276 = stablehlo.reduce(%2275 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2277 = stablehlo.reshape %2276 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2278 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2279 = stablehlo.broadcast_in_dim %2277, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2280 = stablehlo.subtract %2278, %2279 : tensor<1x16x197x197xf32>
    %2281 = stablehlo.exponential %2280 : tensor<1x16x197x197xf32>
    %2282 = stablehlo.reduce(%2281 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2283 = stablehlo.reshape %2282 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2284 = stablehlo.broadcast_in_dim %2281, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2285 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2286 = stablehlo.divide %2284, %2285 : tensor<1x16x197x197xf32>
    %2287 = stablehlo.convert %2286 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %2288 = stablehlo.reshape %2287 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %2289 = stablehlo.reshape %2263 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2290 = stablehlo.broadcast_in_dim %2289, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2291 = stablehlo.dot_general %2288, %2290, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2292 = stablehlo.reshape %2291 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2293 = stablehlo.transpose %2292, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %2294 = stablehlo.reshape %2293 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %2295 = stablehlo.reshape %2294 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2296 = stablehlo.convert %2295 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2297 = stablehlo.dot_general %2296, %arg288, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2298 = stablehlo.broadcast_in_dim %2297, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2299 = stablehlo.multiply %2298, %60 : tensor<197x1024xf32>
    %2300 = stablehlo.broadcast_in_dim %2299, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2301 = stablehlo.broadcast_in_dim %arg289, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2302 = stablehlo.add %2300, %2301 : tensor<197x1024xf32>
    %2303 = stablehlo.convert %2302 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2304 = stablehlo.reshape %2303 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2305 = stablehlo.broadcast_in_dim %arg71, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2306 = stablehlo.broadcast_in_dim %2304, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2307 = stablehlo.multiply %2305, %2306 : tensor<1x197x1024xbf16>
    %2308 = stablehlo.add %2307, %2202 : tensor<1x197x1024xbf16>
    %2309 = stablehlo.convert %2308 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2310 = stablehlo.convert %2309 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2311 = stablehlo.reduce(%2310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2312 = stablehlo.reshape %2311 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2313 = stablehlo.broadcast_in_dim %2312, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2314 = stablehlo.divide %2313, %15 : tensor<1x197x1xf64>
    %2315 = stablehlo.broadcast_in_dim %2310, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2316 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2317 = stablehlo.subtract %2315, %2316 : tensor<1x197x1024xf64>
    %2318 = stablehlo.multiply %2317, %2317 : tensor<1x197x1024xf64>
    %2319 = stablehlo.reduce(%2318 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2320 = stablehlo.reshape %2319 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2321 = stablehlo.broadcast_in_dim %2320, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2322 = stablehlo.divide %2321, %15 : tensor<1x197x1xf64>
    %2323 = stablehlo.convert %2322 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2324 = stablehlo.reduce(%2309 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2325 = stablehlo.reshape %2324 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2326 = stablehlo.broadcast_in_dim %2325, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2327 = stablehlo.divide %2326, %31 : tensor<1x197x1xf32>
    %2328 = stablehlo.broadcast_in_dim %2323, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2329 = stablehlo.add %2328, %36 : tensor<1x197x1xf32>
    %2330 = stablehlo.rsqrt %2329 : tensor<1x197x1xf32>
    %2331 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2332 = stablehlo.broadcast_in_dim %2327, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2333 = stablehlo.subtract %2331, %2332 : tensor<1x197x1024xf32>
    %2334 = stablehlo.broadcast_in_dim %2333, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2335 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2336 = stablehlo.multiply %2334, %2335 : tensor<1x197x1024xf32>
    %2337 = stablehlo.convert %arg72 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2338 = stablehlo.broadcast_in_dim %2336, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2339 = stablehlo.broadcast_in_dim %2337, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2340 = stablehlo.multiply %2338, %2339 : tensor<1x197x1024xf32>
    %2341 = stablehlo.convert %arg73 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2342 = stablehlo.broadcast_in_dim %2340, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2343 = stablehlo.broadcast_in_dim %2341, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2344 = stablehlo.add %2342, %2343 : tensor<1x197x1024xf32>
    %2345 = stablehlo.convert %2344 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2346 = stablehlo.reshape %2345 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2347 = stablehlo.convert %2346 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2348 = stablehlo.dot_general %2347, %arg290, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %2349 = stablehlo.broadcast_in_dim %2348, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2350 = stablehlo.multiply %2349, %170 : tensor<197x4096xf32>
    %2351 = stablehlo.broadcast_in_dim %2350, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2352 = stablehlo.broadcast_in_dim %arg291, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %2353 = stablehlo.add %2351, %2352 : tensor<197x4096xf32>
    %2354 = stablehlo.convert %2353 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %2355 = stablehlo.reshape %2354 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %2356 = stablehlo.multiply %2355, %cst_4 : tensor<1x197x4096xbf16>
    %2357 = stablehlo.multiply %2355, %178 : tensor<1x197x4096xbf16>
    %2358 = stablehlo.convert %2357 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %2359 = stablehlo.clamp %cst_5, %2358, %cst_6 : tensor<1x197x4096xf32>
    %2360 = stablehlo.multiply %2359, %2359 : tensor<1x197x4096xf32>
    %2361 = stablehlo.multiply %cst_7, %2360 : tensor<1x197x4096xf32>
    %2362 = stablehlo.add %2361, %cst_8 : tensor<1x197x4096xf32>
    %2363 = stablehlo.multiply %2362, %2360 : tensor<1x197x4096xf32>
    %2364 = stablehlo.add %2363, %cst_9 : tensor<1x197x4096xf32>
    %2365 = stablehlo.multiply %2364, %2360 : tensor<1x197x4096xf32>
    %2366 = stablehlo.add %2365, %cst_10 : tensor<1x197x4096xf32>
    %2367 = stablehlo.multiply %2366, %2360 : tensor<1x197x4096xf32>
    %2368 = stablehlo.add %2367, %cst_11 : tensor<1x197x4096xf32>
    %2369 = stablehlo.multiply %2368, %2360 : tensor<1x197x4096xf32>
    %2370 = stablehlo.add %2369, %cst_12 : tensor<1x197x4096xf32>
    %2371 = stablehlo.multiply %2370, %2360 : tensor<1x197x4096xf32>
    %2372 = stablehlo.add %2371, %cst_13 : tensor<1x197x4096xf32>
    %2373 = stablehlo.multiply %cst_14, %2360 : tensor<1x197x4096xf32>
    %2374 = stablehlo.add %2373, %cst_15 : tensor<1x197x4096xf32>
    %2375 = stablehlo.multiply %2374, %2360 : tensor<1x197x4096xf32>
    %2376 = stablehlo.add %2375, %cst_16 : tensor<1x197x4096xf32>
    %2377 = stablehlo.multiply %2376, %2360 : tensor<1x197x4096xf32>
    %2378 = stablehlo.add %2377, %cst_17 : tensor<1x197x4096xf32>
    %2379 = stablehlo.multiply %2378, %2360 : tensor<1x197x4096xf32>
    %2380 = stablehlo.add %2379, %cst_18 : tensor<1x197x4096xf32>
    %2381 = stablehlo.multiply %2359, %2372 : tensor<1x197x4096xf32>
    %2382 = stablehlo.divide %2381, %2380 : tensor<1x197x4096xf32>
    %2383 = stablehlo.clamp %cst_19, %2382, %cst_20 : tensor<1x197x4096xf32>
    %2384 = stablehlo.convert %2383 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %2385 = stablehlo.add %2384, %cst_2 : tensor<1x197x4096xbf16>
    %2386 = stablehlo.multiply %2385, %2356 : tensor<1x197x4096xbf16>
    %2387 = stablehlo.reshape %2386 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %2388 = stablehlo.convert %2387 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %2389 = stablehlo.dot_general %2388, %arg292, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %2390 = stablehlo.broadcast_in_dim %2389, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2391 = stablehlo.multiply %2390, %60 : tensor<197x1024xf32>
    %2392 = stablehlo.broadcast_in_dim %2391, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2393 = stablehlo.broadcast_in_dim %arg293, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2394 = stablehlo.add %2392, %2393 : tensor<197x1024xf32>
    %2395 = stablehlo.convert %2394 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2396 = stablehlo.reshape %2395 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2397 = stablehlo.broadcast_in_dim %arg74, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2398 = stablehlo.broadcast_in_dim %2396, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2399 = stablehlo.multiply %2397, %2398 : tensor<1x197x1024xbf16>
    %2400 = stablehlo.add %2399, %2308 : tensor<1x197x1024xbf16>
    %2401 = stablehlo.convert %2400 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2402 = stablehlo.convert %2401 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2403 = stablehlo.reduce(%2402 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2404 = stablehlo.reshape %2403 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2405 = stablehlo.broadcast_in_dim %2404, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2406 = stablehlo.divide %2405, %15 : tensor<1x197x1xf64>
    %2407 = stablehlo.broadcast_in_dim %2402, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2408 = stablehlo.broadcast_in_dim %2406, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2409 = stablehlo.subtract %2407, %2408 : tensor<1x197x1024xf64>
    %2410 = stablehlo.multiply %2409, %2409 : tensor<1x197x1024xf64>
    %2411 = stablehlo.reduce(%2410 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2412 = stablehlo.reshape %2411 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2413 = stablehlo.broadcast_in_dim %2412, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2414 = stablehlo.divide %2413, %15 : tensor<1x197x1xf64>
    %2415 = stablehlo.convert %2414 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2416 = stablehlo.reduce(%2401 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2417 = stablehlo.reshape %2416 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2418 = stablehlo.broadcast_in_dim %2417, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2419 = stablehlo.divide %2418, %31 : tensor<1x197x1xf32>
    %2420 = stablehlo.broadcast_in_dim %2415, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2421 = stablehlo.add %2420, %36 : tensor<1x197x1xf32>
    %2422 = stablehlo.rsqrt %2421 : tensor<1x197x1xf32>
    %2423 = stablehlo.broadcast_in_dim %2401, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2424 = stablehlo.broadcast_in_dim %2419, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2425 = stablehlo.subtract %2423, %2424 : tensor<1x197x1024xf32>
    %2426 = stablehlo.broadcast_in_dim %2425, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2427 = stablehlo.broadcast_in_dim %2422, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2428 = stablehlo.multiply %2426, %2427 : tensor<1x197x1024xf32>
    %2429 = stablehlo.convert %arg75 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2430 = stablehlo.broadcast_in_dim %2428, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2431 = stablehlo.broadcast_in_dim %2429, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2432 = stablehlo.multiply %2430, %2431 : tensor<1x197x1024xf32>
    %2433 = stablehlo.convert %arg76 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2434 = stablehlo.broadcast_in_dim %2432, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2435 = stablehlo.broadcast_in_dim %2433, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2436 = stablehlo.add %2434, %2435 : tensor<1x197x1024xf32>
    %2437 = stablehlo.convert %2436 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2438 = stablehlo.reshape %2437 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2439 = stablehlo.convert %2438 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2440 = stablehlo.dot_general %2439, %arg294, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2441 = stablehlo.broadcast_in_dim %2440, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2442 = stablehlo.multiply %2441, %60 : tensor<197x1024xf32>
    %2443 = stablehlo.broadcast_in_dim %2442, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2444 = stablehlo.broadcast_in_dim %arg295, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2445 = stablehlo.add %2443, %2444 : tensor<197x1024xf32>
    %2446 = stablehlo.convert %2445 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2447 = stablehlo.reshape %2446 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2448 = stablehlo.dot_general %2438, %arg296, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %2449 = stablehlo.reshape %2448 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2450 = stablehlo.reshape %2449 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2451 = stablehlo.transpose %2450, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2452 = stablehlo.dot_general %2439, %arg297, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2453 = stablehlo.broadcast_in_dim %2452, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2454 = stablehlo.multiply %2453, %60 : tensor<197x1024xf32>
    %2455 = stablehlo.broadcast_in_dim %2454, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2456 = stablehlo.broadcast_in_dim %arg298, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2457 = stablehlo.add %2455, %2456 : tensor<197x1024xf32>
    %2458 = stablehlo.convert %2457 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2459 = stablehlo.reshape %2458 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2460 = stablehlo.reshape %2459 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2461 = stablehlo.transpose %2460, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2462 = stablehlo.reshape %2447 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2463 = stablehlo.transpose %2462, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2464 = stablehlo.transpose %2451, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %2465 = stablehlo.reshape %2463 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2466 = stablehlo.reshape %2464 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2467 = stablehlo.broadcast_in_dim %2466, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2468 = stablehlo.dot_general %2465, %2467, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %2469 = stablehlo.reshape %2468 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2470 = stablehlo.broadcast_in_dim %2469, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2471 = stablehlo.divide %2470, %92 : tensor<1x16x197x197xbf16>
    %2472 = stablehlo.add %2471, %arg299 : tensor<1x16x197x197xbf16>
    %2473 = stablehlo.convert %2472 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %2474 = stablehlo.reduce(%2473 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2475 = stablehlo.reshape %2474 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2476 = stablehlo.broadcast_in_dim %2473, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2477 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2478 = stablehlo.subtract %2476, %2477 : tensor<1x16x197x197xf32>
    %2479 = stablehlo.exponential %2478 : tensor<1x16x197x197xf32>
    %2480 = stablehlo.reduce(%2479 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2481 = stablehlo.reshape %2480 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2482 = stablehlo.broadcast_in_dim %2479, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2483 = stablehlo.broadcast_in_dim %2481, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2484 = stablehlo.divide %2482, %2483 : tensor<1x16x197x197xf32>
    %2485 = stablehlo.convert %2484 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %2486 = stablehlo.reshape %2485 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %2487 = stablehlo.reshape %2461 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2488 = stablehlo.broadcast_in_dim %2487, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2489 = stablehlo.dot_general %2486, %2488, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2490 = stablehlo.reshape %2489 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2491 = stablehlo.transpose %2490, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %2492 = stablehlo.reshape %2491 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %2493 = stablehlo.reshape %2492 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2494 = stablehlo.convert %2493 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2495 = stablehlo.dot_general %2494, %arg300, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2496 = stablehlo.broadcast_in_dim %2495, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2497 = stablehlo.multiply %2496, %60 : tensor<197x1024xf32>
    %2498 = stablehlo.broadcast_in_dim %2497, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2499 = stablehlo.broadcast_in_dim %arg301, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2500 = stablehlo.add %2498, %2499 : tensor<197x1024xf32>
    %2501 = stablehlo.convert %2500 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2502 = stablehlo.reshape %2501 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2503 = stablehlo.broadcast_in_dim %arg77, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2504 = stablehlo.broadcast_in_dim %2502, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2505 = stablehlo.multiply %2503, %2504 : tensor<1x197x1024xbf16>
    %2506 = stablehlo.add %2505, %2400 : tensor<1x197x1024xbf16>
    %2507 = stablehlo.convert %2506 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2508 = stablehlo.convert %2507 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2509 = stablehlo.reduce(%2508 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2510 = stablehlo.reshape %2509 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2511 = stablehlo.broadcast_in_dim %2510, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2512 = stablehlo.divide %2511, %15 : tensor<1x197x1xf64>
    %2513 = stablehlo.broadcast_in_dim %2508, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2514 = stablehlo.broadcast_in_dim %2512, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2515 = stablehlo.subtract %2513, %2514 : tensor<1x197x1024xf64>
    %2516 = stablehlo.multiply %2515, %2515 : tensor<1x197x1024xf64>
    %2517 = stablehlo.reduce(%2516 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2518 = stablehlo.reshape %2517 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2519 = stablehlo.broadcast_in_dim %2518, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2520 = stablehlo.divide %2519, %15 : tensor<1x197x1xf64>
    %2521 = stablehlo.convert %2520 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2522 = stablehlo.reduce(%2507 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2523 = stablehlo.reshape %2522 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2524 = stablehlo.broadcast_in_dim %2523, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2525 = stablehlo.divide %2524, %31 : tensor<1x197x1xf32>
    %2526 = stablehlo.broadcast_in_dim %2521, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2527 = stablehlo.add %2526, %36 : tensor<1x197x1xf32>
    %2528 = stablehlo.rsqrt %2527 : tensor<1x197x1xf32>
    %2529 = stablehlo.broadcast_in_dim %2507, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2530 = stablehlo.broadcast_in_dim %2525, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2531 = stablehlo.subtract %2529, %2530 : tensor<1x197x1024xf32>
    %2532 = stablehlo.broadcast_in_dim %2531, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2533 = stablehlo.broadcast_in_dim %2528, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2534 = stablehlo.multiply %2532, %2533 : tensor<1x197x1024xf32>
    %2535 = stablehlo.convert %arg78 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2536 = stablehlo.broadcast_in_dim %2534, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2537 = stablehlo.broadcast_in_dim %2535, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2538 = stablehlo.multiply %2536, %2537 : tensor<1x197x1024xf32>
    %2539 = stablehlo.convert %arg79 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2540 = stablehlo.broadcast_in_dim %2538, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2541 = stablehlo.broadcast_in_dim %2539, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2542 = stablehlo.add %2540, %2541 : tensor<1x197x1024xf32>
    %2543 = stablehlo.convert %2542 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2544 = stablehlo.reshape %2543 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2545 = stablehlo.convert %2544 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2546 = stablehlo.dot_general %2545, %arg302, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %2547 = stablehlo.broadcast_in_dim %2546, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2548 = stablehlo.multiply %2547, %170 : tensor<197x4096xf32>
    %2549 = stablehlo.broadcast_in_dim %2548, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2550 = stablehlo.broadcast_in_dim %arg303, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %2551 = stablehlo.add %2549, %2550 : tensor<197x4096xf32>
    %2552 = stablehlo.convert %2551 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %2553 = stablehlo.reshape %2552 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %2554 = stablehlo.multiply %2553, %cst_4 : tensor<1x197x4096xbf16>
    %2555 = stablehlo.multiply %2553, %178 : tensor<1x197x4096xbf16>
    %2556 = stablehlo.convert %2555 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %2557 = stablehlo.clamp %cst_5, %2556, %cst_6 : tensor<1x197x4096xf32>
    %2558 = stablehlo.multiply %2557, %2557 : tensor<1x197x4096xf32>
    %2559 = stablehlo.multiply %cst_7, %2558 : tensor<1x197x4096xf32>
    %2560 = stablehlo.add %2559, %cst_8 : tensor<1x197x4096xf32>
    %2561 = stablehlo.multiply %2560, %2558 : tensor<1x197x4096xf32>
    %2562 = stablehlo.add %2561, %cst_9 : tensor<1x197x4096xf32>
    %2563 = stablehlo.multiply %2562, %2558 : tensor<1x197x4096xf32>
    %2564 = stablehlo.add %2563, %cst_10 : tensor<1x197x4096xf32>
    %2565 = stablehlo.multiply %2564, %2558 : tensor<1x197x4096xf32>
    %2566 = stablehlo.add %2565, %cst_11 : tensor<1x197x4096xf32>
    %2567 = stablehlo.multiply %2566, %2558 : tensor<1x197x4096xf32>
    %2568 = stablehlo.add %2567, %cst_12 : tensor<1x197x4096xf32>
    %2569 = stablehlo.multiply %2568, %2558 : tensor<1x197x4096xf32>
    %2570 = stablehlo.add %2569, %cst_13 : tensor<1x197x4096xf32>
    %2571 = stablehlo.multiply %cst_14, %2558 : tensor<1x197x4096xf32>
    %2572 = stablehlo.add %2571, %cst_15 : tensor<1x197x4096xf32>
    %2573 = stablehlo.multiply %2572, %2558 : tensor<1x197x4096xf32>
    %2574 = stablehlo.add %2573, %cst_16 : tensor<1x197x4096xf32>
    %2575 = stablehlo.multiply %2574, %2558 : tensor<1x197x4096xf32>
    %2576 = stablehlo.add %2575, %cst_17 : tensor<1x197x4096xf32>
    %2577 = stablehlo.multiply %2576, %2558 : tensor<1x197x4096xf32>
    %2578 = stablehlo.add %2577, %cst_18 : tensor<1x197x4096xf32>
    %2579 = stablehlo.multiply %2557, %2570 : tensor<1x197x4096xf32>
    %2580 = stablehlo.divide %2579, %2578 : tensor<1x197x4096xf32>
    %2581 = stablehlo.clamp %cst_19, %2580, %cst_20 : tensor<1x197x4096xf32>
    %2582 = stablehlo.convert %2581 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %2583 = stablehlo.add %2582, %cst_2 : tensor<1x197x4096xbf16>
    %2584 = stablehlo.multiply %2583, %2554 : tensor<1x197x4096xbf16>
    %2585 = stablehlo.reshape %2584 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %2586 = stablehlo.convert %2585 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %2587 = stablehlo.dot_general %2586, %arg304, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %2588 = stablehlo.broadcast_in_dim %2587, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2589 = stablehlo.multiply %2588, %60 : tensor<197x1024xf32>
    %2590 = stablehlo.broadcast_in_dim %2589, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2591 = stablehlo.broadcast_in_dim %arg305, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2592 = stablehlo.add %2590, %2591 : tensor<197x1024xf32>
    %2593 = stablehlo.convert %2592 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2594 = stablehlo.reshape %2593 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2595 = stablehlo.broadcast_in_dim %arg80, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2596 = stablehlo.broadcast_in_dim %2594, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2597 = stablehlo.multiply %2595, %2596 : tensor<1x197x1024xbf16>
    %2598 = stablehlo.add %2597, %2506 : tensor<1x197x1024xbf16>
    %2599 = stablehlo.convert %2598 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2600 = stablehlo.convert %2599 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2601 = stablehlo.reduce(%2600 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2602 = stablehlo.reshape %2601 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2603 = stablehlo.broadcast_in_dim %2602, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2604 = stablehlo.divide %2603, %15 : tensor<1x197x1xf64>
    %2605 = stablehlo.broadcast_in_dim %2600, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2606 = stablehlo.broadcast_in_dim %2604, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2607 = stablehlo.subtract %2605, %2606 : tensor<1x197x1024xf64>
    %2608 = stablehlo.multiply %2607, %2607 : tensor<1x197x1024xf64>
    %2609 = stablehlo.reduce(%2608 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2610 = stablehlo.reshape %2609 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2611 = stablehlo.broadcast_in_dim %2610, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2612 = stablehlo.divide %2611, %15 : tensor<1x197x1xf64>
    %2613 = stablehlo.convert %2612 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2614 = stablehlo.reduce(%2599 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2615 = stablehlo.reshape %2614 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2616 = stablehlo.broadcast_in_dim %2615, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2617 = stablehlo.divide %2616, %31 : tensor<1x197x1xf32>
    %2618 = stablehlo.broadcast_in_dim %2613, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2619 = stablehlo.add %2618, %36 : tensor<1x197x1xf32>
    %2620 = stablehlo.rsqrt %2619 : tensor<1x197x1xf32>
    %2621 = stablehlo.broadcast_in_dim %2599, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2622 = stablehlo.broadcast_in_dim %2617, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2623 = stablehlo.subtract %2621, %2622 : tensor<1x197x1024xf32>
    %2624 = stablehlo.broadcast_in_dim %2623, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2625 = stablehlo.broadcast_in_dim %2620, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2626 = stablehlo.multiply %2624, %2625 : tensor<1x197x1024xf32>
    %2627 = stablehlo.convert %arg81 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2628 = stablehlo.broadcast_in_dim %2626, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2629 = stablehlo.broadcast_in_dim %2627, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2630 = stablehlo.multiply %2628, %2629 : tensor<1x197x1024xf32>
    %2631 = stablehlo.convert %arg82 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2632 = stablehlo.broadcast_in_dim %2630, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2633 = stablehlo.broadcast_in_dim %2631, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2634 = stablehlo.add %2632, %2633 : tensor<1x197x1024xf32>
    %2635 = stablehlo.convert %2634 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2636 = stablehlo.reshape %2635 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2637 = stablehlo.convert %2636 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2638 = stablehlo.dot_general %2637, %arg306, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2639 = stablehlo.broadcast_in_dim %2638, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2640 = stablehlo.multiply %2639, %60 : tensor<197x1024xf32>
    %2641 = stablehlo.broadcast_in_dim %2640, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2642 = stablehlo.broadcast_in_dim %arg307, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2643 = stablehlo.add %2641, %2642 : tensor<197x1024xf32>
    %2644 = stablehlo.convert %2643 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2645 = stablehlo.reshape %2644 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2646 = stablehlo.dot_general %2636, %arg308, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %2647 = stablehlo.reshape %2646 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2648 = stablehlo.reshape %2647 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2649 = stablehlo.transpose %2648, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2650 = stablehlo.dot_general %2637, %arg309, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2651 = stablehlo.broadcast_in_dim %2650, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2652 = stablehlo.multiply %2651, %60 : tensor<197x1024xf32>
    %2653 = stablehlo.broadcast_in_dim %2652, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2654 = stablehlo.broadcast_in_dim %arg310, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2655 = stablehlo.add %2653, %2654 : tensor<197x1024xf32>
    %2656 = stablehlo.convert %2655 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2657 = stablehlo.reshape %2656 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2658 = stablehlo.reshape %2657 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2659 = stablehlo.transpose %2658, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2660 = stablehlo.reshape %2645 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2661 = stablehlo.transpose %2660, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2662 = stablehlo.transpose %2649, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %2663 = stablehlo.reshape %2661 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2664 = stablehlo.reshape %2662 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2665 = stablehlo.broadcast_in_dim %2664, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2666 = stablehlo.dot_general %2663, %2665, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %2667 = stablehlo.reshape %2666 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2668 = stablehlo.broadcast_in_dim %2667, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2669 = stablehlo.divide %2668, %92 : tensor<1x16x197x197xbf16>
    %2670 = stablehlo.add %2669, %arg311 : tensor<1x16x197x197xbf16>
    %2671 = stablehlo.convert %2670 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %2672 = stablehlo.reduce(%2671 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2673 = stablehlo.reshape %2672 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2674 = stablehlo.broadcast_in_dim %2671, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2675 = stablehlo.broadcast_in_dim %2673, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2676 = stablehlo.subtract %2674, %2675 : tensor<1x16x197x197xf32>
    %2677 = stablehlo.exponential %2676 : tensor<1x16x197x197xf32>
    %2678 = stablehlo.reduce(%2677 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2679 = stablehlo.reshape %2678 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2680 = stablehlo.broadcast_in_dim %2677, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2681 = stablehlo.broadcast_in_dim %2679, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2682 = stablehlo.divide %2680, %2681 : tensor<1x16x197x197xf32>
    %2683 = stablehlo.convert %2682 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %2684 = stablehlo.reshape %2683 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %2685 = stablehlo.reshape %2659 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2686 = stablehlo.broadcast_in_dim %2685, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2687 = stablehlo.dot_general %2684, %2686, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2688 = stablehlo.reshape %2687 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2689 = stablehlo.transpose %2688, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %2690 = stablehlo.reshape %2689 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %2691 = stablehlo.reshape %2690 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2692 = stablehlo.convert %2691 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2693 = stablehlo.dot_general %2692, %arg312, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2694 = stablehlo.broadcast_in_dim %2693, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2695 = stablehlo.multiply %2694, %60 : tensor<197x1024xf32>
    %2696 = stablehlo.broadcast_in_dim %2695, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2697 = stablehlo.broadcast_in_dim %arg313, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2698 = stablehlo.add %2696, %2697 : tensor<197x1024xf32>
    %2699 = stablehlo.convert %2698 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2700 = stablehlo.reshape %2699 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2701 = stablehlo.broadcast_in_dim %arg83, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2702 = stablehlo.broadcast_in_dim %2700, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2703 = stablehlo.multiply %2701, %2702 : tensor<1x197x1024xbf16>
    %2704 = stablehlo.add %2703, %2598 : tensor<1x197x1024xbf16>
    %2705 = stablehlo.convert %2704 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2706 = stablehlo.convert %2705 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2707 = stablehlo.reduce(%2706 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2708 = stablehlo.reshape %2707 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2709 = stablehlo.broadcast_in_dim %2708, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2710 = stablehlo.divide %2709, %15 : tensor<1x197x1xf64>
    %2711 = stablehlo.broadcast_in_dim %2706, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2712 = stablehlo.broadcast_in_dim %2710, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2713 = stablehlo.subtract %2711, %2712 : tensor<1x197x1024xf64>
    %2714 = stablehlo.multiply %2713, %2713 : tensor<1x197x1024xf64>
    %2715 = stablehlo.reduce(%2714 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2716 = stablehlo.reshape %2715 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2717 = stablehlo.broadcast_in_dim %2716, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2718 = stablehlo.divide %2717, %15 : tensor<1x197x1xf64>
    %2719 = stablehlo.convert %2718 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2720 = stablehlo.reduce(%2705 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2721 = stablehlo.reshape %2720 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2723 = stablehlo.divide %2722, %31 : tensor<1x197x1xf32>
    %2724 = stablehlo.broadcast_in_dim %2719, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2725 = stablehlo.add %2724, %36 : tensor<1x197x1xf32>
    %2726 = stablehlo.rsqrt %2725 : tensor<1x197x1xf32>
    %2727 = stablehlo.broadcast_in_dim %2705, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2728 = stablehlo.broadcast_in_dim %2723, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2729 = stablehlo.subtract %2727, %2728 : tensor<1x197x1024xf32>
    %2730 = stablehlo.broadcast_in_dim %2729, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2731 = stablehlo.broadcast_in_dim %2726, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2732 = stablehlo.multiply %2730, %2731 : tensor<1x197x1024xf32>
    %2733 = stablehlo.convert %arg84 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2734 = stablehlo.broadcast_in_dim %2732, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2735 = stablehlo.broadcast_in_dim %2733, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2736 = stablehlo.multiply %2734, %2735 : tensor<1x197x1024xf32>
    %2737 = stablehlo.convert %arg85 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2738 = stablehlo.broadcast_in_dim %2736, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2739 = stablehlo.broadcast_in_dim %2737, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2740 = stablehlo.add %2738, %2739 : tensor<1x197x1024xf32>
    %2741 = stablehlo.convert %2740 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2742 = stablehlo.reshape %2741 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2743 = stablehlo.convert %2742 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2744 = stablehlo.dot_general %2743, %arg314, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %2745 = stablehlo.broadcast_in_dim %2744, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2746 = stablehlo.multiply %2745, %170 : tensor<197x4096xf32>
    %2747 = stablehlo.broadcast_in_dim %2746, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2748 = stablehlo.broadcast_in_dim %arg315, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %2749 = stablehlo.add %2747, %2748 : tensor<197x4096xf32>
    %2750 = stablehlo.convert %2749 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %2751 = stablehlo.reshape %2750 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %2752 = stablehlo.multiply %2751, %cst_4 : tensor<1x197x4096xbf16>
    %2753 = stablehlo.multiply %2751, %178 : tensor<1x197x4096xbf16>
    %2754 = stablehlo.convert %2753 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %2755 = stablehlo.clamp %cst_5, %2754, %cst_6 : tensor<1x197x4096xf32>
    %2756 = stablehlo.multiply %2755, %2755 : tensor<1x197x4096xf32>
    %2757 = stablehlo.multiply %cst_7, %2756 : tensor<1x197x4096xf32>
    %2758 = stablehlo.add %2757, %cst_8 : tensor<1x197x4096xf32>
    %2759 = stablehlo.multiply %2758, %2756 : tensor<1x197x4096xf32>
    %2760 = stablehlo.add %2759, %cst_9 : tensor<1x197x4096xf32>
    %2761 = stablehlo.multiply %2760, %2756 : tensor<1x197x4096xf32>
    %2762 = stablehlo.add %2761, %cst_10 : tensor<1x197x4096xf32>
    %2763 = stablehlo.multiply %2762, %2756 : tensor<1x197x4096xf32>
    %2764 = stablehlo.add %2763, %cst_11 : tensor<1x197x4096xf32>
    %2765 = stablehlo.multiply %2764, %2756 : tensor<1x197x4096xf32>
    %2766 = stablehlo.add %2765, %cst_12 : tensor<1x197x4096xf32>
    %2767 = stablehlo.multiply %2766, %2756 : tensor<1x197x4096xf32>
    %2768 = stablehlo.add %2767, %cst_13 : tensor<1x197x4096xf32>
    %2769 = stablehlo.multiply %cst_14, %2756 : tensor<1x197x4096xf32>
    %2770 = stablehlo.add %2769, %cst_15 : tensor<1x197x4096xf32>
    %2771 = stablehlo.multiply %2770, %2756 : tensor<1x197x4096xf32>
    %2772 = stablehlo.add %2771, %cst_16 : tensor<1x197x4096xf32>
    %2773 = stablehlo.multiply %2772, %2756 : tensor<1x197x4096xf32>
    %2774 = stablehlo.add %2773, %cst_17 : tensor<1x197x4096xf32>
    %2775 = stablehlo.multiply %2774, %2756 : tensor<1x197x4096xf32>
    %2776 = stablehlo.add %2775, %cst_18 : tensor<1x197x4096xf32>
    %2777 = stablehlo.multiply %2755, %2768 : tensor<1x197x4096xf32>
    %2778 = stablehlo.divide %2777, %2776 : tensor<1x197x4096xf32>
    %2779 = stablehlo.clamp %cst_19, %2778, %cst_20 : tensor<1x197x4096xf32>
    %2780 = stablehlo.convert %2779 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %2781 = stablehlo.add %2780, %cst_2 : tensor<1x197x4096xbf16>
    %2782 = stablehlo.multiply %2781, %2752 : tensor<1x197x4096xbf16>
    %2783 = stablehlo.reshape %2782 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %2784 = stablehlo.convert %2783 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %2785 = stablehlo.dot_general %2784, %arg316, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %2786 = stablehlo.broadcast_in_dim %2785, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2787 = stablehlo.multiply %2786, %60 : tensor<197x1024xf32>
    %2788 = stablehlo.broadcast_in_dim %2787, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2789 = stablehlo.broadcast_in_dim %arg317, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2790 = stablehlo.add %2788, %2789 : tensor<197x1024xf32>
    %2791 = stablehlo.convert %2790 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2792 = stablehlo.reshape %2791 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2793 = stablehlo.broadcast_in_dim %arg86, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2794 = stablehlo.broadcast_in_dim %2792, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2795 = stablehlo.multiply %2793, %2794 : tensor<1x197x1024xbf16>
    %2796 = stablehlo.add %2795, %2704 : tensor<1x197x1024xbf16>
    %2797 = stablehlo.convert %2796 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2798 = stablehlo.convert %2797 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2799 = stablehlo.reduce(%2798 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2800 = stablehlo.reshape %2799 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2801 = stablehlo.broadcast_in_dim %2800, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2802 = stablehlo.divide %2801, %15 : tensor<1x197x1xf64>
    %2803 = stablehlo.broadcast_in_dim %2798, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2804 = stablehlo.broadcast_in_dim %2802, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2805 = stablehlo.subtract %2803, %2804 : tensor<1x197x1024xf64>
    %2806 = stablehlo.multiply %2805, %2805 : tensor<1x197x1024xf64>
    %2807 = stablehlo.reduce(%2806 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2808 = stablehlo.reshape %2807 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2809 = stablehlo.broadcast_in_dim %2808, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2810 = stablehlo.divide %2809, %15 : tensor<1x197x1xf64>
    %2811 = stablehlo.convert %2810 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2812 = stablehlo.reduce(%2797 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2813 = stablehlo.reshape %2812 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2814 = stablehlo.broadcast_in_dim %2813, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2815 = stablehlo.divide %2814, %31 : tensor<1x197x1xf32>
    %2816 = stablehlo.broadcast_in_dim %2811, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2817 = stablehlo.add %2816, %36 : tensor<1x197x1xf32>
    %2818 = stablehlo.rsqrt %2817 : tensor<1x197x1xf32>
    %2819 = stablehlo.broadcast_in_dim %2797, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2820 = stablehlo.broadcast_in_dim %2815, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2821 = stablehlo.subtract %2819, %2820 : tensor<1x197x1024xf32>
    %2822 = stablehlo.broadcast_in_dim %2821, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2823 = stablehlo.broadcast_in_dim %2818, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2824 = stablehlo.multiply %2822, %2823 : tensor<1x197x1024xf32>
    %2825 = stablehlo.convert %arg87 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2826 = stablehlo.broadcast_in_dim %2824, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2827 = stablehlo.broadcast_in_dim %2825, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2828 = stablehlo.multiply %2826, %2827 : tensor<1x197x1024xf32>
    %2829 = stablehlo.convert %arg88 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2830 = stablehlo.broadcast_in_dim %2828, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2831 = stablehlo.broadcast_in_dim %2829, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2832 = stablehlo.add %2830, %2831 : tensor<1x197x1024xf32>
    %2833 = stablehlo.convert %2832 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2834 = stablehlo.reshape %2833 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2835 = stablehlo.convert %2834 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2836 = stablehlo.dot_general %2835, %arg318, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2837 = stablehlo.broadcast_in_dim %2836, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2838 = stablehlo.multiply %2837, %60 : tensor<197x1024xf32>
    %2839 = stablehlo.broadcast_in_dim %2838, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2840 = stablehlo.broadcast_in_dim %arg319, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2841 = stablehlo.add %2839, %2840 : tensor<197x1024xf32>
    %2842 = stablehlo.convert %2841 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2843 = stablehlo.reshape %2842 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2844 = stablehlo.dot_general %2834, %arg320, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %2845 = stablehlo.reshape %2844 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2846 = stablehlo.reshape %2845 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2847 = stablehlo.transpose %2846, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2848 = stablehlo.dot_general %2835, %arg321, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2849 = stablehlo.broadcast_in_dim %2848, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2850 = stablehlo.multiply %2849, %60 : tensor<197x1024xf32>
    %2851 = stablehlo.broadcast_in_dim %2850, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2852 = stablehlo.broadcast_in_dim %arg322, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2853 = stablehlo.add %2851, %2852 : tensor<197x1024xf32>
    %2854 = stablehlo.convert %2853 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2855 = stablehlo.reshape %2854 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2856 = stablehlo.reshape %2855 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2857 = stablehlo.transpose %2856, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2858 = stablehlo.reshape %2843 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %2859 = stablehlo.transpose %2858, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2860 = stablehlo.transpose %2847, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %2861 = stablehlo.reshape %2859 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2862 = stablehlo.reshape %2860 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2863 = stablehlo.broadcast_in_dim %2862, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %2864 = stablehlo.dot_general %2861, %2863, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %2865 = stablehlo.reshape %2864 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2866 = stablehlo.broadcast_in_dim %2865, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %2867 = stablehlo.divide %2866, %92 : tensor<1x16x197x197xbf16>
    %2868 = stablehlo.add %2867, %arg323 : tensor<1x16x197x197xbf16>
    %2869 = stablehlo.convert %2868 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %2870 = stablehlo.reduce(%2869 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2871 = stablehlo.reshape %2870 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2872 = stablehlo.broadcast_in_dim %2869, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2873 = stablehlo.broadcast_in_dim %2871, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2874 = stablehlo.subtract %2872, %2873 : tensor<1x16x197x197xf32>
    %2875 = stablehlo.exponential %2874 : tensor<1x16x197x197xf32>
    %2876 = stablehlo.reduce(%2875 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %2877 = stablehlo.reshape %2876 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %2878 = stablehlo.broadcast_in_dim %2875, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %2879 = stablehlo.broadcast_in_dim %2877, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %2880 = stablehlo.divide %2878, %2879 : tensor<1x16x197x197xf32>
    %2881 = stablehlo.convert %2880 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %2882 = stablehlo.reshape %2881 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %2883 = stablehlo.reshape %2857 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2884 = stablehlo.broadcast_in_dim %2883, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2885 = stablehlo.dot_general %2882, %2884, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %2886 = stablehlo.reshape %2885 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %2887 = stablehlo.transpose %2886, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %2888 = stablehlo.reshape %2887 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %2889 = stablehlo.reshape %2888 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2890 = stablehlo.convert %2889 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2891 = stablehlo.dot_general %2890, %arg324, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %2892 = stablehlo.broadcast_in_dim %2891, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2893 = stablehlo.multiply %2892, %60 : tensor<197x1024xf32>
    %2894 = stablehlo.broadcast_in_dim %2893, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2895 = stablehlo.broadcast_in_dim %arg325, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2896 = stablehlo.add %2894, %2895 : tensor<197x1024xf32>
    %2897 = stablehlo.convert %2896 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2898 = stablehlo.reshape %2897 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2899 = stablehlo.broadcast_in_dim %arg89, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2900 = stablehlo.broadcast_in_dim %2898, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2901 = stablehlo.multiply %2899, %2900 : tensor<1x197x1024xbf16>
    %2902 = stablehlo.add %2901, %2796 : tensor<1x197x1024xbf16>
    %2903 = stablehlo.convert %2902 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2904 = stablehlo.convert %2903 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2905 = stablehlo.reduce(%2904 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2906 = stablehlo.reshape %2905 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2907 = stablehlo.broadcast_in_dim %2906, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2908 = stablehlo.divide %2907, %15 : tensor<1x197x1xf64>
    %2909 = stablehlo.broadcast_in_dim %2904, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %2910 = stablehlo.broadcast_in_dim %2908, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %2911 = stablehlo.subtract %2909, %2910 : tensor<1x197x1024xf64>
    %2912 = stablehlo.multiply %2911, %2911 : tensor<1x197x1024xf64>
    %2913 = stablehlo.reduce(%2912 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2914 = stablehlo.reshape %2913 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2915 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %2916 = stablehlo.divide %2915, %15 : tensor<1x197x1xf64>
    %2917 = stablehlo.convert %2916 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %2918 = stablehlo.reduce(%2903 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %2919 = stablehlo.reshape %2918 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %2920 = stablehlo.broadcast_in_dim %2919, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2921 = stablehlo.divide %2920, %31 : tensor<1x197x1xf32>
    %2922 = stablehlo.broadcast_in_dim %2917, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %2923 = stablehlo.add %2922, %36 : tensor<1x197x1xf32>
    %2924 = stablehlo.rsqrt %2923 : tensor<1x197x1xf32>
    %2925 = stablehlo.broadcast_in_dim %2903, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2926 = stablehlo.broadcast_in_dim %2921, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2927 = stablehlo.subtract %2925, %2926 : tensor<1x197x1024xf32>
    %2928 = stablehlo.broadcast_in_dim %2927, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2929 = stablehlo.broadcast_in_dim %2924, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %2930 = stablehlo.multiply %2928, %2929 : tensor<1x197x1024xf32>
    %2931 = stablehlo.convert %arg90 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2932 = stablehlo.broadcast_in_dim %2930, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2933 = stablehlo.broadcast_in_dim %2931, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2934 = stablehlo.multiply %2932, %2933 : tensor<1x197x1024xf32>
    %2935 = stablehlo.convert %arg91 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %2936 = stablehlo.broadcast_in_dim %2934, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %2937 = stablehlo.broadcast_in_dim %2935, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %2938 = stablehlo.add %2936, %2937 : tensor<1x197x1024xf32>
    %2939 = stablehlo.convert %2938 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %2940 = stablehlo.reshape %2939 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %2941 = stablehlo.convert %2940 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %2942 = stablehlo.dot_general %2941, %arg326, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %2943 = stablehlo.broadcast_in_dim %2942, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2944 = stablehlo.multiply %2943, %170 : tensor<197x4096xf32>
    %2945 = stablehlo.broadcast_in_dim %2944, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %2946 = stablehlo.broadcast_in_dim %arg327, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %2947 = stablehlo.add %2945, %2946 : tensor<197x4096xf32>
    %2948 = stablehlo.convert %2947 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %2949 = stablehlo.reshape %2948 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %2950 = stablehlo.multiply %2949, %cst_4 : tensor<1x197x4096xbf16>
    %2951 = stablehlo.multiply %2949, %178 : tensor<1x197x4096xbf16>
    %2952 = stablehlo.convert %2951 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %2953 = stablehlo.clamp %cst_5, %2952, %cst_6 : tensor<1x197x4096xf32>
    %2954 = stablehlo.multiply %2953, %2953 : tensor<1x197x4096xf32>
    %2955 = stablehlo.multiply %cst_7, %2954 : tensor<1x197x4096xf32>
    %2956 = stablehlo.add %2955, %cst_8 : tensor<1x197x4096xf32>
    %2957 = stablehlo.multiply %2956, %2954 : tensor<1x197x4096xf32>
    %2958 = stablehlo.add %2957, %cst_9 : tensor<1x197x4096xf32>
    %2959 = stablehlo.multiply %2958, %2954 : tensor<1x197x4096xf32>
    %2960 = stablehlo.add %2959, %cst_10 : tensor<1x197x4096xf32>
    %2961 = stablehlo.multiply %2960, %2954 : tensor<1x197x4096xf32>
    %2962 = stablehlo.add %2961, %cst_11 : tensor<1x197x4096xf32>
    %2963 = stablehlo.multiply %2962, %2954 : tensor<1x197x4096xf32>
    %2964 = stablehlo.add %2963, %cst_12 : tensor<1x197x4096xf32>
    %2965 = stablehlo.multiply %2964, %2954 : tensor<1x197x4096xf32>
    %2966 = stablehlo.add %2965, %cst_13 : tensor<1x197x4096xf32>
    %2967 = stablehlo.multiply %cst_14, %2954 : tensor<1x197x4096xf32>
    %2968 = stablehlo.add %2967, %cst_15 : tensor<1x197x4096xf32>
    %2969 = stablehlo.multiply %2968, %2954 : tensor<1x197x4096xf32>
    %2970 = stablehlo.add %2969, %cst_16 : tensor<1x197x4096xf32>
    %2971 = stablehlo.multiply %2970, %2954 : tensor<1x197x4096xf32>
    %2972 = stablehlo.add %2971, %cst_17 : tensor<1x197x4096xf32>
    %2973 = stablehlo.multiply %2972, %2954 : tensor<1x197x4096xf32>
    %2974 = stablehlo.add %2973, %cst_18 : tensor<1x197x4096xf32>
    %2975 = stablehlo.multiply %2953, %2966 : tensor<1x197x4096xf32>
    %2976 = stablehlo.divide %2975, %2974 : tensor<1x197x4096xf32>
    %2977 = stablehlo.clamp %cst_19, %2976, %cst_20 : tensor<1x197x4096xf32>
    %2978 = stablehlo.convert %2977 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %2979 = stablehlo.add %2978, %cst_2 : tensor<1x197x4096xbf16>
    %2980 = stablehlo.multiply %2979, %2950 : tensor<1x197x4096xbf16>
    %2981 = stablehlo.reshape %2980 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %2982 = stablehlo.convert %2981 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %2983 = stablehlo.dot_general %2982, %arg328, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %2984 = stablehlo.broadcast_in_dim %2983, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2985 = stablehlo.multiply %2984, %60 : tensor<197x1024xf32>
    %2986 = stablehlo.broadcast_in_dim %2985, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %2987 = stablehlo.broadcast_in_dim %arg329, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %2988 = stablehlo.add %2986, %2987 : tensor<197x1024xf32>
    %2989 = stablehlo.convert %2988 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %2990 = stablehlo.reshape %2989 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2991 = stablehlo.broadcast_in_dim %arg92, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %2992 = stablehlo.broadcast_in_dim %2990, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %2993 = stablehlo.multiply %2991, %2992 : tensor<1x197x1024xbf16>
    %2994 = stablehlo.add %2993, %2902 : tensor<1x197x1024xbf16>
    %2995 = stablehlo.convert %2994 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %2996 = stablehlo.convert %2995 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %2997 = stablehlo.reduce(%2996 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %2998 = stablehlo.reshape %2997 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %2999 = stablehlo.broadcast_in_dim %2998, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3000 = stablehlo.divide %2999, %15 : tensor<1x197x1xf64>
    %3001 = stablehlo.broadcast_in_dim %2996, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3002 = stablehlo.broadcast_in_dim %3000, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3003 = stablehlo.subtract %3001, %3002 : tensor<1x197x1024xf64>
    %3004 = stablehlo.multiply %3003, %3003 : tensor<1x197x1024xf64>
    %3005 = stablehlo.reduce(%3004 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3006 = stablehlo.reshape %3005 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3007 = stablehlo.broadcast_in_dim %3006, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3008 = stablehlo.divide %3007, %15 : tensor<1x197x1xf64>
    %3009 = stablehlo.convert %3008 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3010 = stablehlo.reduce(%2995 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3011 = stablehlo.reshape %3010 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3012 = stablehlo.broadcast_in_dim %3011, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3013 = stablehlo.divide %3012, %31 : tensor<1x197x1xf32>
    %3014 = stablehlo.broadcast_in_dim %3009, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3015 = stablehlo.add %3014, %36 : tensor<1x197x1xf32>
    %3016 = stablehlo.rsqrt %3015 : tensor<1x197x1xf32>
    %3017 = stablehlo.broadcast_in_dim %2995, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3018 = stablehlo.broadcast_in_dim %3013, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3019 = stablehlo.subtract %3017, %3018 : tensor<1x197x1024xf32>
    %3020 = stablehlo.broadcast_in_dim %3019, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3021 = stablehlo.broadcast_in_dim %3016, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3022 = stablehlo.multiply %3020, %3021 : tensor<1x197x1024xf32>
    %3023 = stablehlo.convert %arg93 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3024 = stablehlo.broadcast_in_dim %3022, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3025 = stablehlo.broadcast_in_dim %3023, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3026 = stablehlo.multiply %3024, %3025 : tensor<1x197x1024xf32>
    %3027 = stablehlo.convert %arg94 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3028 = stablehlo.broadcast_in_dim %3026, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3029 = stablehlo.broadcast_in_dim %3027, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3030 = stablehlo.add %3028, %3029 : tensor<1x197x1024xf32>
    %3031 = stablehlo.convert %3030 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3032 = stablehlo.reshape %3031 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3033 = stablehlo.convert %3032 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3034 = stablehlo.dot_general %3033, %arg330, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3035 = stablehlo.broadcast_in_dim %3034, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3036 = stablehlo.multiply %3035, %60 : tensor<197x1024xf32>
    %3037 = stablehlo.broadcast_in_dim %3036, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3038 = stablehlo.broadcast_in_dim %arg331, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3039 = stablehlo.add %3037, %3038 : tensor<197x1024xf32>
    %3040 = stablehlo.convert %3039 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3041 = stablehlo.reshape %3040 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3042 = stablehlo.dot_general %3032, %arg332, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %3043 = stablehlo.reshape %3042 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3044 = stablehlo.reshape %3043 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3045 = stablehlo.transpose %3044, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3046 = stablehlo.dot_general %3033, %arg333, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3047 = stablehlo.broadcast_in_dim %3046, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3048 = stablehlo.multiply %3047, %60 : tensor<197x1024xf32>
    %3049 = stablehlo.broadcast_in_dim %3048, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3050 = stablehlo.broadcast_in_dim %arg334, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3051 = stablehlo.add %3049, %3050 : tensor<197x1024xf32>
    %3052 = stablehlo.convert %3051 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3053 = stablehlo.reshape %3052 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3054 = stablehlo.reshape %3053 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3055 = stablehlo.transpose %3054, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3056 = stablehlo.reshape %3041 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3057 = stablehlo.transpose %3056, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3058 = stablehlo.transpose %3045, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %3059 = stablehlo.reshape %3057 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3060 = stablehlo.reshape %3058 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3061 = stablehlo.broadcast_in_dim %3060, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3062 = stablehlo.dot_general %3059, %3061, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %3063 = stablehlo.reshape %3062 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3064 = stablehlo.broadcast_in_dim %3063, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3065 = stablehlo.divide %3064, %92 : tensor<1x16x197x197xbf16>
    %3066 = stablehlo.add %3065, %arg335 : tensor<1x16x197x197xbf16>
    %3067 = stablehlo.convert %3066 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %3068 = stablehlo.reduce(%3067 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3069 = stablehlo.reshape %3068 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3070 = stablehlo.broadcast_in_dim %3067, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3071 = stablehlo.broadcast_in_dim %3069, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3072 = stablehlo.subtract %3070, %3071 : tensor<1x16x197x197xf32>
    %3073 = stablehlo.exponential %3072 : tensor<1x16x197x197xf32>
    %3074 = stablehlo.reduce(%3073 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3075 = stablehlo.reshape %3074 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3076 = stablehlo.broadcast_in_dim %3073, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3077 = stablehlo.broadcast_in_dim %3075, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3078 = stablehlo.divide %3076, %3077 : tensor<1x16x197x197xf32>
    %3079 = stablehlo.convert %3078 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %3080 = stablehlo.reshape %3079 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %3081 = stablehlo.reshape %3055 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3082 = stablehlo.broadcast_in_dim %3081, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3083 = stablehlo.dot_general %3080, %3082, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3084 = stablehlo.reshape %3083 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3085 = stablehlo.transpose %3084, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %3086 = stablehlo.reshape %3085 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %3087 = stablehlo.reshape %3086 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3088 = stablehlo.convert %3087 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3089 = stablehlo.dot_general %3088, %arg336, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3090 = stablehlo.broadcast_in_dim %3089, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3091 = stablehlo.multiply %3090, %60 : tensor<197x1024xf32>
    %3092 = stablehlo.broadcast_in_dim %3091, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3093 = stablehlo.broadcast_in_dim %arg337, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3094 = stablehlo.add %3092, %3093 : tensor<197x1024xf32>
    %3095 = stablehlo.convert %3094 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3096 = stablehlo.reshape %3095 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3097 = stablehlo.broadcast_in_dim %arg95, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3098 = stablehlo.broadcast_in_dim %3096, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3099 = stablehlo.multiply %3097, %3098 : tensor<1x197x1024xbf16>
    %3100 = stablehlo.add %3099, %2994 : tensor<1x197x1024xbf16>
    %3101 = stablehlo.convert %3100 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3102 = stablehlo.convert %3101 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3103 = stablehlo.reduce(%3102 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3104 = stablehlo.reshape %3103 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3105 = stablehlo.broadcast_in_dim %3104, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3106 = stablehlo.divide %3105, %15 : tensor<1x197x1xf64>
    %3107 = stablehlo.broadcast_in_dim %3102, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3108 = stablehlo.broadcast_in_dim %3106, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3109 = stablehlo.subtract %3107, %3108 : tensor<1x197x1024xf64>
    %3110 = stablehlo.multiply %3109, %3109 : tensor<1x197x1024xf64>
    %3111 = stablehlo.reduce(%3110 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3112 = stablehlo.reshape %3111 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3113 = stablehlo.broadcast_in_dim %3112, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3114 = stablehlo.divide %3113, %15 : tensor<1x197x1xf64>
    %3115 = stablehlo.convert %3114 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3116 = stablehlo.reduce(%3101 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3117 = stablehlo.reshape %3116 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3118 = stablehlo.broadcast_in_dim %3117, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3119 = stablehlo.divide %3118, %31 : tensor<1x197x1xf32>
    %3120 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3121 = stablehlo.add %3120, %36 : tensor<1x197x1xf32>
    %3122 = stablehlo.rsqrt %3121 : tensor<1x197x1xf32>
    %3123 = stablehlo.broadcast_in_dim %3101, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3124 = stablehlo.broadcast_in_dim %3119, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3125 = stablehlo.subtract %3123, %3124 : tensor<1x197x1024xf32>
    %3126 = stablehlo.broadcast_in_dim %3125, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3127 = stablehlo.broadcast_in_dim %3122, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3128 = stablehlo.multiply %3126, %3127 : tensor<1x197x1024xf32>
    %3129 = stablehlo.convert %arg96 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3130 = stablehlo.broadcast_in_dim %3128, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3131 = stablehlo.broadcast_in_dim %3129, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3132 = stablehlo.multiply %3130, %3131 : tensor<1x197x1024xf32>
    %3133 = stablehlo.convert %arg97 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3134 = stablehlo.broadcast_in_dim %3132, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3135 = stablehlo.broadcast_in_dim %3133, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3136 = stablehlo.add %3134, %3135 : tensor<1x197x1024xf32>
    %3137 = stablehlo.convert %3136 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3138 = stablehlo.reshape %3137 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3139 = stablehlo.convert %3138 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3140 = stablehlo.dot_general %3139, %arg338, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %3141 = stablehlo.broadcast_in_dim %3140, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3142 = stablehlo.multiply %3141, %170 : tensor<197x4096xf32>
    %3143 = stablehlo.broadcast_in_dim %3142, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3144 = stablehlo.broadcast_in_dim %arg339, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %3145 = stablehlo.add %3143, %3144 : tensor<197x4096xf32>
    %3146 = stablehlo.convert %3145 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %3147 = stablehlo.reshape %3146 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %3148 = stablehlo.multiply %3147, %cst_4 : tensor<1x197x4096xbf16>
    %3149 = stablehlo.multiply %3147, %178 : tensor<1x197x4096xbf16>
    %3150 = stablehlo.convert %3149 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %3151 = stablehlo.clamp %cst_5, %3150, %cst_6 : tensor<1x197x4096xf32>
    %3152 = stablehlo.multiply %3151, %3151 : tensor<1x197x4096xf32>
    %3153 = stablehlo.multiply %cst_7, %3152 : tensor<1x197x4096xf32>
    %3154 = stablehlo.add %3153, %cst_8 : tensor<1x197x4096xf32>
    %3155 = stablehlo.multiply %3154, %3152 : tensor<1x197x4096xf32>
    %3156 = stablehlo.add %3155, %cst_9 : tensor<1x197x4096xf32>
    %3157 = stablehlo.multiply %3156, %3152 : tensor<1x197x4096xf32>
    %3158 = stablehlo.add %3157, %cst_10 : tensor<1x197x4096xf32>
    %3159 = stablehlo.multiply %3158, %3152 : tensor<1x197x4096xf32>
    %3160 = stablehlo.add %3159, %cst_11 : tensor<1x197x4096xf32>
    %3161 = stablehlo.multiply %3160, %3152 : tensor<1x197x4096xf32>
    %3162 = stablehlo.add %3161, %cst_12 : tensor<1x197x4096xf32>
    %3163 = stablehlo.multiply %3162, %3152 : tensor<1x197x4096xf32>
    %3164 = stablehlo.add %3163, %cst_13 : tensor<1x197x4096xf32>
    %3165 = stablehlo.multiply %cst_14, %3152 : tensor<1x197x4096xf32>
    %3166 = stablehlo.add %3165, %cst_15 : tensor<1x197x4096xf32>
    %3167 = stablehlo.multiply %3166, %3152 : tensor<1x197x4096xf32>
    %3168 = stablehlo.add %3167, %cst_16 : tensor<1x197x4096xf32>
    %3169 = stablehlo.multiply %3168, %3152 : tensor<1x197x4096xf32>
    %3170 = stablehlo.add %3169, %cst_17 : tensor<1x197x4096xf32>
    %3171 = stablehlo.multiply %3170, %3152 : tensor<1x197x4096xf32>
    %3172 = stablehlo.add %3171, %cst_18 : tensor<1x197x4096xf32>
    %3173 = stablehlo.multiply %3151, %3164 : tensor<1x197x4096xf32>
    %3174 = stablehlo.divide %3173, %3172 : tensor<1x197x4096xf32>
    %3175 = stablehlo.clamp %cst_19, %3174, %cst_20 : tensor<1x197x4096xf32>
    %3176 = stablehlo.convert %3175 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %3177 = stablehlo.add %3176, %cst_2 : tensor<1x197x4096xbf16>
    %3178 = stablehlo.multiply %3177, %3148 : tensor<1x197x4096xbf16>
    %3179 = stablehlo.reshape %3178 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %3180 = stablehlo.convert %3179 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %3181 = stablehlo.dot_general %3180, %arg340, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %3182 = stablehlo.broadcast_in_dim %3181, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3183 = stablehlo.multiply %3182, %60 : tensor<197x1024xf32>
    %3184 = stablehlo.broadcast_in_dim %3183, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3185 = stablehlo.broadcast_in_dim %arg341, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3186 = stablehlo.add %3184, %3185 : tensor<197x1024xf32>
    %3187 = stablehlo.convert %3186 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3188 = stablehlo.reshape %3187 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3189 = stablehlo.broadcast_in_dim %arg98, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3190 = stablehlo.broadcast_in_dim %3188, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3191 = stablehlo.multiply %3189, %3190 : tensor<1x197x1024xbf16>
    %3192 = stablehlo.add %3191, %3100 : tensor<1x197x1024xbf16>
    %3193 = stablehlo.convert %3192 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3194 = stablehlo.convert %3193 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3195 = stablehlo.reduce(%3194 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3196 = stablehlo.reshape %3195 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3197 = stablehlo.broadcast_in_dim %3196, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3198 = stablehlo.divide %3197, %15 : tensor<1x197x1xf64>
    %3199 = stablehlo.broadcast_in_dim %3194, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3200 = stablehlo.broadcast_in_dim %3198, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3201 = stablehlo.subtract %3199, %3200 : tensor<1x197x1024xf64>
    %3202 = stablehlo.multiply %3201, %3201 : tensor<1x197x1024xf64>
    %3203 = stablehlo.reduce(%3202 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3204 = stablehlo.reshape %3203 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3205 = stablehlo.broadcast_in_dim %3204, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3206 = stablehlo.divide %3205, %15 : tensor<1x197x1xf64>
    %3207 = stablehlo.convert %3206 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3208 = stablehlo.reduce(%3193 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3209 = stablehlo.reshape %3208 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3210 = stablehlo.broadcast_in_dim %3209, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3211 = stablehlo.divide %3210, %31 : tensor<1x197x1xf32>
    %3212 = stablehlo.broadcast_in_dim %3207, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3213 = stablehlo.add %3212, %36 : tensor<1x197x1xf32>
    %3214 = stablehlo.rsqrt %3213 : tensor<1x197x1xf32>
    %3215 = stablehlo.broadcast_in_dim %3193, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3216 = stablehlo.broadcast_in_dim %3211, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3217 = stablehlo.subtract %3215, %3216 : tensor<1x197x1024xf32>
    %3218 = stablehlo.broadcast_in_dim %3217, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3219 = stablehlo.broadcast_in_dim %3214, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3220 = stablehlo.multiply %3218, %3219 : tensor<1x197x1024xf32>
    %3221 = stablehlo.convert %arg99 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3222 = stablehlo.broadcast_in_dim %3220, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3223 = stablehlo.broadcast_in_dim %3221, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3224 = stablehlo.multiply %3222, %3223 : tensor<1x197x1024xf32>
    %3225 = stablehlo.convert %arg100 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3226 = stablehlo.broadcast_in_dim %3224, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3227 = stablehlo.broadcast_in_dim %3225, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3228 = stablehlo.add %3226, %3227 : tensor<1x197x1024xf32>
    %3229 = stablehlo.convert %3228 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3230 = stablehlo.reshape %3229 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3231 = stablehlo.convert %3230 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3232 = stablehlo.dot_general %3231, %arg342, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3233 = stablehlo.broadcast_in_dim %3232, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3234 = stablehlo.multiply %3233, %60 : tensor<197x1024xf32>
    %3235 = stablehlo.broadcast_in_dim %3234, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3236 = stablehlo.broadcast_in_dim %arg343, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3237 = stablehlo.add %3235, %3236 : tensor<197x1024xf32>
    %3238 = stablehlo.convert %3237 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3239 = stablehlo.reshape %3238 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3240 = stablehlo.dot_general %3230, %arg344, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %3241 = stablehlo.reshape %3240 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3242 = stablehlo.reshape %3241 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3243 = stablehlo.transpose %3242, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3244 = stablehlo.dot_general %3231, %arg345, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3245 = stablehlo.broadcast_in_dim %3244, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3246 = stablehlo.multiply %3245, %60 : tensor<197x1024xf32>
    %3247 = stablehlo.broadcast_in_dim %3246, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3248 = stablehlo.broadcast_in_dim %arg346, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3249 = stablehlo.add %3247, %3248 : tensor<197x1024xf32>
    %3250 = stablehlo.convert %3249 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3251 = stablehlo.reshape %3250 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3252 = stablehlo.reshape %3251 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3253 = stablehlo.transpose %3252, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3254 = stablehlo.reshape %3239 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3255 = stablehlo.transpose %3254, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3256 = stablehlo.transpose %3243, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %3257 = stablehlo.reshape %3255 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3258 = stablehlo.reshape %3256 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3259 = stablehlo.broadcast_in_dim %3258, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3260 = stablehlo.dot_general %3257, %3259, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %3261 = stablehlo.reshape %3260 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3262 = stablehlo.broadcast_in_dim %3261, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3263 = stablehlo.divide %3262, %92 : tensor<1x16x197x197xbf16>
    %3264 = stablehlo.add %3263, %arg347 : tensor<1x16x197x197xbf16>
    %3265 = stablehlo.convert %3264 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %3266 = stablehlo.reduce(%3265 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3267 = stablehlo.reshape %3266 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3268 = stablehlo.broadcast_in_dim %3265, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3269 = stablehlo.broadcast_in_dim %3267, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3270 = stablehlo.subtract %3268, %3269 : tensor<1x16x197x197xf32>
    %3271 = stablehlo.exponential %3270 : tensor<1x16x197x197xf32>
    %3272 = stablehlo.reduce(%3271 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3273 = stablehlo.reshape %3272 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3274 = stablehlo.broadcast_in_dim %3271, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3275 = stablehlo.broadcast_in_dim %3273, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3276 = stablehlo.divide %3274, %3275 : tensor<1x16x197x197xf32>
    %3277 = stablehlo.convert %3276 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %3278 = stablehlo.reshape %3277 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %3279 = stablehlo.reshape %3253 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3280 = stablehlo.broadcast_in_dim %3279, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3281 = stablehlo.dot_general %3278, %3280, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3282 = stablehlo.reshape %3281 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3283 = stablehlo.transpose %3282, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %3284 = stablehlo.reshape %3283 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %3285 = stablehlo.reshape %3284 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3286 = stablehlo.convert %3285 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3287 = stablehlo.dot_general %3286, %arg348, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3288 = stablehlo.broadcast_in_dim %3287, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3289 = stablehlo.multiply %3288, %60 : tensor<197x1024xf32>
    %3290 = stablehlo.broadcast_in_dim %3289, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3291 = stablehlo.broadcast_in_dim %arg349, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3292 = stablehlo.add %3290, %3291 : tensor<197x1024xf32>
    %3293 = stablehlo.convert %3292 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3294 = stablehlo.reshape %3293 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3295 = stablehlo.broadcast_in_dim %arg101, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3296 = stablehlo.broadcast_in_dim %3294, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3297 = stablehlo.multiply %3295, %3296 : tensor<1x197x1024xbf16>
    %3298 = stablehlo.add %3297, %3192 : tensor<1x197x1024xbf16>
    %3299 = stablehlo.convert %3298 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3300 = stablehlo.convert %3299 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3301 = stablehlo.reduce(%3300 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3302 = stablehlo.reshape %3301 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3303 = stablehlo.broadcast_in_dim %3302, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3304 = stablehlo.divide %3303, %15 : tensor<1x197x1xf64>
    %3305 = stablehlo.broadcast_in_dim %3300, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3306 = stablehlo.broadcast_in_dim %3304, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3307 = stablehlo.subtract %3305, %3306 : tensor<1x197x1024xf64>
    %3308 = stablehlo.multiply %3307, %3307 : tensor<1x197x1024xf64>
    %3309 = stablehlo.reduce(%3308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3310 = stablehlo.reshape %3309 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3311 = stablehlo.broadcast_in_dim %3310, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3312 = stablehlo.divide %3311, %15 : tensor<1x197x1xf64>
    %3313 = stablehlo.convert %3312 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3314 = stablehlo.reduce(%3299 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3315 = stablehlo.reshape %3314 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3316 = stablehlo.broadcast_in_dim %3315, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3317 = stablehlo.divide %3316, %31 : tensor<1x197x1xf32>
    %3318 = stablehlo.broadcast_in_dim %3313, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3319 = stablehlo.add %3318, %36 : tensor<1x197x1xf32>
    %3320 = stablehlo.rsqrt %3319 : tensor<1x197x1xf32>
    %3321 = stablehlo.broadcast_in_dim %3299, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3322 = stablehlo.broadcast_in_dim %3317, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3323 = stablehlo.subtract %3321, %3322 : tensor<1x197x1024xf32>
    %3324 = stablehlo.broadcast_in_dim %3323, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3325 = stablehlo.broadcast_in_dim %3320, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3326 = stablehlo.multiply %3324, %3325 : tensor<1x197x1024xf32>
    %3327 = stablehlo.convert %arg102 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3328 = stablehlo.broadcast_in_dim %3326, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3329 = stablehlo.broadcast_in_dim %3327, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3330 = stablehlo.multiply %3328, %3329 : tensor<1x197x1024xf32>
    %3331 = stablehlo.convert %arg103 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3332 = stablehlo.broadcast_in_dim %3330, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3333 = stablehlo.broadcast_in_dim %3331, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3334 = stablehlo.add %3332, %3333 : tensor<1x197x1024xf32>
    %3335 = stablehlo.convert %3334 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3336 = stablehlo.reshape %3335 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3337 = stablehlo.convert %3336 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3338 = stablehlo.dot_general %3337, %arg350, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %3339 = stablehlo.broadcast_in_dim %3338, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3340 = stablehlo.multiply %3339, %170 : tensor<197x4096xf32>
    %3341 = stablehlo.broadcast_in_dim %3340, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3342 = stablehlo.broadcast_in_dim %arg351, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %3343 = stablehlo.add %3341, %3342 : tensor<197x4096xf32>
    %3344 = stablehlo.convert %3343 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %3345 = stablehlo.reshape %3344 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %3346 = stablehlo.multiply %3345, %cst_4 : tensor<1x197x4096xbf16>
    %3347 = stablehlo.multiply %3345, %178 : tensor<1x197x4096xbf16>
    %3348 = stablehlo.convert %3347 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %3349 = stablehlo.clamp %cst_5, %3348, %cst_6 : tensor<1x197x4096xf32>
    %3350 = stablehlo.multiply %3349, %3349 : tensor<1x197x4096xf32>
    %3351 = stablehlo.multiply %cst_7, %3350 : tensor<1x197x4096xf32>
    %3352 = stablehlo.add %3351, %cst_8 : tensor<1x197x4096xf32>
    %3353 = stablehlo.multiply %3352, %3350 : tensor<1x197x4096xf32>
    %3354 = stablehlo.add %3353, %cst_9 : tensor<1x197x4096xf32>
    %3355 = stablehlo.multiply %3354, %3350 : tensor<1x197x4096xf32>
    %3356 = stablehlo.add %3355, %cst_10 : tensor<1x197x4096xf32>
    %3357 = stablehlo.multiply %3356, %3350 : tensor<1x197x4096xf32>
    %3358 = stablehlo.add %3357, %cst_11 : tensor<1x197x4096xf32>
    %3359 = stablehlo.multiply %3358, %3350 : tensor<1x197x4096xf32>
    %3360 = stablehlo.add %3359, %cst_12 : tensor<1x197x4096xf32>
    %3361 = stablehlo.multiply %3360, %3350 : tensor<1x197x4096xf32>
    %3362 = stablehlo.add %3361, %cst_13 : tensor<1x197x4096xf32>
    %3363 = stablehlo.multiply %cst_14, %3350 : tensor<1x197x4096xf32>
    %3364 = stablehlo.add %3363, %cst_15 : tensor<1x197x4096xf32>
    %3365 = stablehlo.multiply %3364, %3350 : tensor<1x197x4096xf32>
    %3366 = stablehlo.add %3365, %cst_16 : tensor<1x197x4096xf32>
    %3367 = stablehlo.multiply %3366, %3350 : tensor<1x197x4096xf32>
    %3368 = stablehlo.add %3367, %cst_17 : tensor<1x197x4096xf32>
    %3369 = stablehlo.multiply %3368, %3350 : tensor<1x197x4096xf32>
    %3370 = stablehlo.add %3369, %cst_18 : tensor<1x197x4096xf32>
    %3371 = stablehlo.multiply %3349, %3362 : tensor<1x197x4096xf32>
    %3372 = stablehlo.divide %3371, %3370 : tensor<1x197x4096xf32>
    %3373 = stablehlo.clamp %cst_19, %3372, %cst_20 : tensor<1x197x4096xf32>
    %3374 = stablehlo.convert %3373 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %3375 = stablehlo.add %3374, %cst_2 : tensor<1x197x4096xbf16>
    %3376 = stablehlo.multiply %3375, %3346 : tensor<1x197x4096xbf16>
    %3377 = stablehlo.reshape %3376 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %3378 = stablehlo.convert %3377 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %3379 = stablehlo.dot_general %3378, %arg352, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %3380 = stablehlo.broadcast_in_dim %3379, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3381 = stablehlo.multiply %3380, %60 : tensor<197x1024xf32>
    %3382 = stablehlo.broadcast_in_dim %3381, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3383 = stablehlo.broadcast_in_dim %arg353, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3384 = stablehlo.add %3382, %3383 : tensor<197x1024xf32>
    %3385 = stablehlo.convert %3384 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3386 = stablehlo.reshape %3385 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3387 = stablehlo.broadcast_in_dim %arg104, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3388 = stablehlo.broadcast_in_dim %3386, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3389 = stablehlo.multiply %3387, %3388 : tensor<1x197x1024xbf16>
    %3390 = stablehlo.add %3389, %3298 : tensor<1x197x1024xbf16>
    %3391 = stablehlo.convert %3390 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3392 = stablehlo.convert %3391 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3393 = stablehlo.reduce(%3392 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3394 = stablehlo.reshape %3393 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3395 = stablehlo.broadcast_in_dim %3394, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3396 = stablehlo.divide %3395, %15 : tensor<1x197x1xf64>
    %3397 = stablehlo.broadcast_in_dim %3392, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3398 = stablehlo.broadcast_in_dim %3396, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3399 = stablehlo.subtract %3397, %3398 : tensor<1x197x1024xf64>
    %3400 = stablehlo.multiply %3399, %3399 : tensor<1x197x1024xf64>
    %3401 = stablehlo.reduce(%3400 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3402 = stablehlo.reshape %3401 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3403 = stablehlo.broadcast_in_dim %3402, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3404 = stablehlo.divide %3403, %15 : tensor<1x197x1xf64>
    %3405 = stablehlo.convert %3404 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3406 = stablehlo.reduce(%3391 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3407 = stablehlo.reshape %3406 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3408 = stablehlo.broadcast_in_dim %3407, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3409 = stablehlo.divide %3408, %31 : tensor<1x197x1xf32>
    %3410 = stablehlo.broadcast_in_dim %3405, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3411 = stablehlo.add %3410, %36 : tensor<1x197x1xf32>
    %3412 = stablehlo.rsqrt %3411 : tensor<1x197x1xf32>
    %3413 = stablehlo.broadcast_in_dim %3391, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3414 = stablehlo.broadcast_in_dim %3409, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3415 = stablehlo.subtract %3413, %3414 : tensor<1x197x1024xf32>
    %3416 = stablehlo.broadcast_in_dim %3415, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3417 = stablehlo.broadcast_in_dim %3412, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3418 = stablehlo.multiply %3416, %3417 : tensor<1x197x1024xf32>
    %3419 = stablehlo.convert %arg105 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3420 = stablehlo.broadcast_in_dim %3418, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3421 = stablehlo.broadcast_in_dim %3419, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3422 = stablehlo.multiply %3420, %3421 : tensor<1x197x1024xf32>
    %3423 = stablehlo.convert %arg106 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3424 = stablehlo.broadcast_in_dim %3422, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3425 = stablehlo.broadcast_in_dim %3423, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3426 = stablehlo.add %3424, %3425 : tensor<1x197x1024xf32>
    %3427 = stablehlo.convert %3426 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3428 = stablehlo.reshape %3427 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3429 = stablehlo.convert %3428 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3430 = stablehlo.dot_general %3429, %arg354, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3431 = stablehlo.broadcast_in_dim %3430, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3432 = stablehlo.multiply %3431, %60 : tensor<197x1024xf32>
    %3433 = stablehlo.broadcast_in_dim %3432, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3434 = stablehlo.broadcast_in_dim %arg355, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3435 = stablehlo.add %3433, %3434 : tensor<197x1024xf32>
    %3436 = stablehlo.convert %3435 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3437 = stablehlo.reshape %3436 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3438 = stablehlo.dot_general %3428, %arg356, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %3439 = stablehlo.reshape %3438 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3440 = stablehlo.reshape %3439 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3441 = stablehlo.transpose %3440, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3442 = stablehlo.dot_general %3429, %arg357, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3443 = stablehlo.broadcast_in_dim %3442, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3444 = stablehlo.multiply %3443, %60 : tensor<197x1024xf32>
    %3445 = stablehlo.broadcast_in_dim %3444, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3446 = stablehlo.broadcast_in_dim %arg358, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3447 = stablehlo.add %3445, %3446 : tensor<197x1024xf32>
    %3448 = stablehlo.convert %3447 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3449 = stablehlo.reshape %3448 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3450 = stablehlo.reshape %3449 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3451 = stablehlo.transpose %3450, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3452 = stablehlo.reshape %3437 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3453 = stablehlo.transpose %3452, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3454 = stablehlo.transpose %3441, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %3455 = stablehlo.reshape %3453 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3456 = stablehlo.reshape %3454 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3457 = stablehlo.broadcast_in_dim %3456, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3458 = stablehlo.dot_general %3455, %3457, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %3459 = stablehlo.reshape %3458 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3460 = stablehlo.broadcast_in_dim %3459, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3461 = stablehlo.divide %3460, %92 : tensor<1x16x197x197xbf16>
    %3462 = stablehlo.add %3461, %arg359 : tensor<1x16x197x197xbf16>
    %3463 = stablehlo.convert %3462 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %3464 = stablehlo.reduce(%3463 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3465 = stablehlo.reshape %3464 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3466 = stablehlo.broadcast_in_dim %3463, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3467 = stablehlo.broadcast_in_dim %3465, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3468 = stablehlo.subtract %3466, %3467 : tensor<1x16x197x197xf32>
    %3469 = stablehlo.exponential %3468 : tensor<1x16x197x197xf32>
    %3470 = stablehlo.reduce(%3469 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3471 = stablehlo.reshape %3470 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3472 = stablehlo.broadcast_in_dim %3469, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3473 = stablehlo.broadcast_in_dim %3471, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3474 = stablehlo.divide %3472, %3473 : tensor<1x16x197x197xf32>
    %3475 = stablehlo.convert %3474 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %3476 = stablehlo.reshape %3475 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %3477 = stablehlo.reshape %3451 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3478 = stablehlo.broadcast_in_dim %3477, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3479 = stablehlo.dot_general %3476, %3478, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3480 = stablehlo.reshape %3479 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3481 = stablehlo.transpose %3480, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %3482 = stablehlo.reshape %3481 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %3483 = stablehlo.reshape %3482 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3484 = stablehlo.convert %3483 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3485 = stablehlo.dot_general %3484, %arg360, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3486 = stablehlo.broadcast_in_dim %3485, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3487 = stablehlo.multiply %3486, %60 : tensor<197x1024xf32>
    %3488 = stablehlo.broadcast_in_dim %3487, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3489 = stablehlo.broadcast_in_dim %arg361, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3490 = stablehlo.add %3488, %3489 : tensor<197x1024xf32>
    %3491 = stablehlo.convert %3490 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3492 = stablehlo.reshape %3491 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3493 = stablehlo.broadcast_in_dim %arg107, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3494 = stablehlo.broadcast_in_dim %3492, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3495 = stablehlo.multiply %3493, %3494 : tensor<1x197x1024xbf16>
    %3496 = stablehlo.add %3495, %3390 : tensor<1x197x1024xbf16>
    %3497 = stablehlo.convert %3496 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3498 = stablehlo.convert %3497 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3499 = stablehlo.reduce(%3498 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3500 = stablehlo.reshape %3499 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3501 = stablehlo.broadcast_in_dim %3500, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3502 = stablehlo.divide %3501, %15 : tensor<1x197x1xf64>
    %3503 = stablehlo.broadcast_in_dim %3498, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3504 = stablehlo.broadcast_in_dim %3502, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3505 = stablehlo.subtract %3503, %3504 : tensor<1x197x1024xf64>
    %3506 = stablehlo.multiply %3505, %3505 : tensor<1x197x1024xf64>
    %3507 = stablehlo.reduce(%3506 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3508 = stablehlo.reshape %3507 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3509 = stablehlo.broadcast_in_dim %3508, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3510 = stablehlo.divide %3509, %15 : tensor<1x197x1xf64>
    %3511 = stablehlo.convert %3510 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3512 = stablehlo.reduce(%3497 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3513 = stablehlo.reshape %3512 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3514 = stablehlo.broadcast_in_dim %3513, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3515 = stablehlo.divide %3514, %31 : tensor<1x197x1xf32>
    %3516 = stablehlo.broadcast_in_dim %3511, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3517 = stablehlo.add %3516, %36 : tensor<1x197x1xf32>
    %3518 = stablehlo.rsqrt %3517 : tensor<1x197x1xf32>
    %3519 = stablehlo.broadcast_in_dim %3497, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3520 = stablehlo.broadcast_in_dim %3515, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3521 = stablehlo.subtract %3519, %3520 : tensor<1x197x1024xf32>
    %3522 = stablehlo.broadcast_in_dim %3521, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3523 = stablehlo.broadcast_in_dim %3518, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3524 = stablehlo.multiply %3522, %3523 : tensor<1x197x1024xf32>
    %3525 = stablehlo.convert %arg108 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3526 = stablehlo.broadcast_in_dim %3524, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3527 = stablehlo.broadcast_in_dim %3525, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3528 = stablehlo.multiply %3526, %3527 : tensor<1x197x1024xf32>
    %3529 = stablehlo.convert %arg109 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3530 = stablehlo.broadcast_in_dim %3528, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3531 = stablehlo.broadcast_in_dim %3529, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3532 = stablehlo.add %3530, %3531 : tensor<1x197x1024xf32>
    %3533 = stablehlo.convert %3532 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3534 = stablehlo.reshape %3533 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3535 = stablehlo.convert %3534 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3536 = stablehlo.dot_general %3535, %arg362, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %3537 = stablehlo.broadcast_in_dim %3536, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3538 = stablehlo.multiply %3537, %170 : tensor<197x4096xf32>
    %3539 = stablehlo.broadcast_in_dim %3538, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3540 = stablehlo.broadcast_in_dim %arg363, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %3541 = stablehlo.add %3539, %3540 : tensor<197x4096xf32>
    %3542 = stablehlo.convert %3541 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %3543 = stablehlo.reshape %3542 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %3544 = stablehlo.multiply %3543, %cst_4 : tensor<1x197x4096xbf16>
    %3545 = stablehlo.multiply %3543, %178 : tensor<1x197x4096xbf16>
    %3546 = stablehlo.convert %3545 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %3547 = stablehlo.clamp %cst_5, %3546, %cst_6 : tensor<1x197x4096xf32>
    %3548 = stablehlo.multiply %3547, %3547 : tensor<1x197x4096xf32>
    %3549 = stablehlo.multiply %cst_7, %3548 : tensor<1x197x4096xf32>
    %3550 = stablehlo.add %3549, %cst_8 : tensor<1x197x4096xf32>
    %3551 = stablehlo.multiply %3550, %3548 : tensor<1x197x4096xf32>
    %3552 = stablehlo.add %3551, %cst_9 : tensor<1x197x4096xf32>
    %3553 = stablehlo.multiply %3552, %3548 : tensor<1x197x4096xf32>
    %3554 = stablehlo.add %3553, %cst_10 : tensor<1x197x4096xf32>
    %3555 = stablehlo.multiply %3554, %3548 : tensor<1x197x4096xf32>
    %3556 = stablehlo.add %3555, %cst_11 : tensor<1x197x4096xf32>
    %3557 = stablehlo.multiply %3556, %3548 : tensor<1x197x4096xf32>
    %3558 = stablehlo.add %3557, %cst_12 : tensor<1x197x4096xf32>
    %3559 = stablehlo.multiply %3558, %3548 : tensor<1x197x4096xf32>
    %3560 = stablehlo.add %3559, %cst_13 : tensor<1x197x4096xf32>
    %3561 = stablehlo.multiply %cst_14, %3548 : tensor<1x197x4096xf32>
    %3562 = stablehlo.add %3561, %cst_15 : tensor<1x197x4096xf32>
    %3563 = stablehlo.multiply %3562, %3548 : tensor<1x197x4096xf32>
    %3564 = stablehlo.add %3563, %cst_16 : tensor<1x197x4096xf32>
    %3565 = stablehlo.multiply %3564, %3548 : tensor<1x197x4096xf32>
    %3566 = stablehlo.add %3565, %cst_17 : tensor<1x197x4096xf32>
    %3567 = stablehlo.multiply %3566, %3548 : tensor<1x197x4096xf32>
    %3568 = stablehlo.add %3567, %cst_18 : tensor<1x197x4096xf32>
    %3569 = stablehlo.multiply %3547, %3560 : tensor<1x197x4096xf32>
    %3570 = stablehlo.divide %3569, %3568 : tensor<1x197x4096xf32>
    %3571 = stablehlo.clamp %cst_19, %3570, %cst_20 : tensor<1x197x4096xf32>
    %3572 = stablehlo.convert %3571 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %3573 = stablehlo.add %3572, %cst_2 : tensor<1x197x4096xbf16>
    %3574 = stablehlo.multiply %3573, %3544 : tensor<1x197x4096xbf16>
    %3575 = stablehlo.reshape %3574 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %3576 = stablehlo.convert %3575 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %3577 = stablehlo.dot_general %3576, %arg364, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %3578 = stablehlo.broadcast_in_dim %3577, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3579 = stablehlo.multiply %3578, %60 : tensor<197x1024xf32>
    %3580 = stablehlo.broadcast_in_dim %3579, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3581 = stablehlo.broadcast_in_dim %arg365, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3582 = stablehlo.add %3580, %3581 : tensor<197x1024xf32>
    %3583 = stablehlo.convert %3582 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3584 = stablehlo.reshape %3583 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3585 = stablehlo.broadcast_in_dim %arg110, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3586 = stablehlo.broadcast_in_dim %3584, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3587 = stablehlo.multiply %3585, %3586 : tensor<1x197x1024xbf16>
    %3588 = stablehlo.add %3587, %3496 : tensor<1x197x1024xbf16>
    %3589 = stablehlo.convert %3588 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3590 = stablehlo.convert %3589 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3591 = stablehlo.reduce(%3590 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3592 = stablehlo.reshape %3591 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3593 = stablehlo.broadcast_in_dim %3592, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3594 = stablehlo.divide %3593, %15 : tensor<1x197x1xf64>
    %3595 = stablehlo.broadcast_in_dim %3590, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3596 = stablehlo.broadcast_in_dim %3594, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3597 = stablehlo.subtract %3595, %3596 : tensor<1x197x1024xf64>
    %3598 = stablehlo.multiply %3597, %3597 : tensor<1x197x1024xf64>
    %3599 = stablehlo.reduce(%3598 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3600 = stablehlo.reshape %3599 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3601 = stablehlo.broadcast_in_dim %3600, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3602 = stablehlo.divide %3601, %15 : tensor<1x197x1xf64>
    %3603 = stablehlo.convert %3602 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3604 = stablehlo.reduce(%3589 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3605 = stablehlo.reshape %3604 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3606 = stablehlo.broadcast_in_dim %3605, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3607 = stablehlo.divide %3606, %31 : tensor<1x197x1xf32>
    %3608 = stablehlo.broadcast_in_dim %3603, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3609 = stablehlo.add %3608, %36 : tensor<1x197x1xf32>
    %3610 = stablehlo.rsqrt %3609 : tensor<1x197x1xf32>
    %3611 = stablehlo.broadcast_in_dim %3589, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3612 = stablehlo.broadcast_in_dim %3607, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3613 = stablehlo.subtract %3611, %3612 : tensor<1x197x1024xf32>
    %3614 = stablehlo.broadcast_in_dim %3613, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3615 = stablehlo.broadcast_in_dim %3610, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3616 = stablehlo.multiply %3614, %3615 : tensor<1x197x1024xf32>
    %3617 = stablehlo.convert %arg111 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3618 = stablehlo.broadcast_in_dim %3616, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3619 = stablehlo.broadcast_in_dim %3617, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3620 = stablehlo.multiply %3618, %3619 : tensor<1x197x1024xf32>
    %3621 = stablehlo.convert %arg112 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3622 = stablehlo.broadcast_in_dim %3620, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3623 = stablehlo.broadcast_in_dim %3621, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3624 = stablehlo.add %3622, %3623 : tensor<1x197x1024xf32>
    %3625 = stablehlo.convert %3624 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3626 = stablehlo.reshape %3625 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3627 = stablehlo.convert %3626 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3628 = stablehlo.dot_general %3627, %arg366, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3629 = stablehlo.broadcast_in_dim %3628, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3630 = stablehlo.multiply %3629, %60 : tensor<197x1024xf32>
    %3631 = stablehlo.broadcast_in_dim %3630, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3632 = stablehlo.broadcast_in_dim %arg367, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3633 = stablehlo.add %3631, %3632 : tensor<197x1024xf32>
    %3634 = stablehlo.convert %3633 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3635 = stablehlo.reshape %3634 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3636 = stablehlo.dot_general %3626, %arg368, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %3637 = stablehlo.reshape %3636 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3638 = stablehlo.reshape %3637 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3639 = stablehlo.transpose %3638, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3640 = stablehlo.dot_general %3627, %arg369, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3641 = stablehlo.broadcast_in_dim %3640, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3642 = stablehlo.multiply %3641, %60 : tensor<197x1024xf32>
    %3643 = stablehlo.broadcast_in_dim %3642, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3644 = stablehlo.broadcast_in_dim %arg370, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3645 = stablehlo.add %3643, %3644 : tensor<197x1024xf32>
    %3646 = stablehlo.convert %3645 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3647 = stablehlo.reshape %3646 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3648 = stablehlo.reshape %3647 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3649 = stablehlo.transpose %3648, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3650 = stablehlo.reshape %3635 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3651 = stablehlo.transpose %3650, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3652 = stablehlo.transpose %3639, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %3653 = stablehlo.reshape %3651 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3654 = stablehlo.reshape %3652 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3655 = stablehlo.broadcast_in_dim %3654, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3656 = stablehlo.dot_general %3653, %3655, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %3657 = stablehlo.reshape %3656 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3658 = stablehlo.broadcast_in_dim %3657, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3659 = stablehlo.divide %3658, %92 : tensor<1x16x197x197xbf16>
    %3660 = stablehlo.add %3659, %arg371 : tensor<1x16x197x197xbf16>
    %3661 = stablehlo.convert %3660 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %3662 = stablehlo.reduce(%3661 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3663 = stablehlo.reshape %3662 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3664 = stablehlo.broadcast_in_dim %3661, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3665 = stablehlo.broadcast_in_dim %3663, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3666 = stablehlo.subtract %3664, %3665 : tensor<1x16x197x197xf32>
    %3667 = stablehlo.exponential %3666 : tensor<1x16x197x197xf32>
    %3668 = stablehlo.reduce(%3667 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3669 = stablehlo.reshape %3668 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3670 = stablehlo.broadcast_in_dim %3667, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3671 = stablehlo.broadcast_in_dim %3669, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3672 = stablehlo.divide %3670, %3671 : tensor<1x16x197x197xf32>
    %3673 = stablehlo.convert %3672 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %3674 = stablehlo.reshape %3673 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %3675 = stablehlo.reshape %3649 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3676 = stablehlo.broadcast_in_dim %3675, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3677 = stablehlo.dot_general %3674, %3676, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3678 = stablehlo.reshape %3677 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3679 = stablehlo.transpose %3678, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %3680 = stablehlo.reshape %3679 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %3681 = stablehlo.reshape %3680 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3682 = stablehlo.convert %3681 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3683 = stablehlo.dot_general %3682, %arg372, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3684 = stablehlo.broadcast_in_dim %3683, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3685 = stablehlo.multiply %3684, %60 : tensor<197x1024xf32>
    %3686 = stablehlo.broadcast_in_dim %3685, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3687 = stablehlo.broadcast_in_dim %arg373, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3688 = stablehlo.add %3686, %3687 : tensor<197x1024xf32>
    %3689 = stablehlo.convert %3688 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3690 = stablehlo.reshape %3689 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3691 = stablehlo.broadcast_in_dim %arg113, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3692 = stablehlo.broadcast_in_dim %3690, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3693 = stablehlo.multiply %3691, %3692 : tensor<1x197x1024xbf16>
    %3694 = stablehlo.add %3693, %3588 : tensor<1x197x1024xbf16>
    %3695 = stablehlo.convert %3694 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3696 = stablehlo.convert %3695 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3697 = stablehlo.reduce(%3696 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3698 = stablehlo.reshape %3697 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3699 = stablehlo.broadcast_in_dim %3698, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3700 = stablehlo.divide %3699, %15 : tensor<1x197x1xf64>
    %3701 = stablehlo.broadcast_in_dim %3696, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3702 = stablehlo.broadcast_in_dim %3700, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3703 = stablehlo.subtract %3701, %3702 : tensor<1x197x1024xf64>
    %3704 = stablehlo.multiply %3703, %3703 : tensor<1x197x1024xf64>
    %3705 = stablehlo.reduce(%3704 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3706 = stablehlo.reshape %3705 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3707 = stablehlo.broadcast_in_dim %3706, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3708 = stablehlo.divide %3707, %15 : tensor<1x197x1xf64>
    %3709 = stablehlo.convert %3708 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3710 = stablehlo.reduce(%3695 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3711 = stablehlo.reshape %3710 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3712 = stablehlo.broadcast_in_dim %3711, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3713 = stablehlo.divide %3712, %31 : tensor<1x197x1xf32>
    %3714 = stablehlo.broadcast_in_dim %3709, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3715 = stablehlo.add %3714, %36 : tensor<1x197x1xf32>
    %3716 = stablehlo.rsqrt %3715 : tensor<1x197x1xf32>
    %3717 = stablehlo.broadcast_in_dim %3695, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3718 = stablehlo.broadcast_in_dim %3713, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3719 = stablehlo.subtract %3717, %3718 : tensor<1x197x1024xf32>
    %3720 = stablehlo.broadcast_in_dim %3719, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3721 = stablehlo.broadcast_in_dim %3716, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3722 = stablehlo.multiply %3720, %3721 : tensor<1x197x1024xf32>
    %3723 = stablehlo.convert %arg114 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3724 = stablehlo.broadcast_in_dim %3722, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3725 = stablehlo.broadcast_in_dim %3723, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3726 = stablehlo.multiply %3724, %3725 : tensor<1x197x1024xf32>
    %3727 = stablehlo.convert %arg115 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3728 = stablehlo.broadcast_in_dim %3726, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3729 = stablehlo.broadcast_in_dim %3727, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3730 = stablehlo.add %3728, %3729 : tensor<1x197x1024xf32>
    %3731 = stablehlo.convert %3730 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3732 = stablehlo.reshape %3731 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3733 = stablehlo.convert %3732 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3734 = stablehlo.dot_general %3733, %arg374, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %3735 = stablehlo.broadcast_in_dim %3734, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3736 = stablehlo.multiply %3735, %170 : tensor<197x4096xf32>
    %3737 = stablehlo.broadcast_in_dim %3736, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3738 = stablehlo.broadcast_in_dim %arg375, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %3739 = stablehlo.add %3737, %3738 : tensor<197x4096xf32>
    %3740 = stablehlo.convert %3739 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %3741 = stablehlo.reshape %3740 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %3742 = stablehlo.multiply %3741, %cst_4 : tensor<1x197x4096xbf16>
    %3743 = stablehlo.multiply %3741, %178 : tensor<1x197x4096xbf16>
    %3744 = stablehlo.convert %3743 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %3745 = stablehlo.clamp %cst_5, %3744, %cst_6 : tensor<1x197x4096xf32>
    %3746 = stablehlo.multiply %3745, %3745 : tensor<1x197x4096xf32>
    %3747 = stablehlo.multiply %cst_7, %3746 : tensor<1x197x4096xf32>
    %3748 = stablehlo.add %3747, %cst_8 : tensor<1x197x4096xf32>
    %3749 = stablehlo.multiply %3748, %3746 : tensor<1x197x4096xf32>
    %3750 = stablehlo.add %3749, %cst_9 : tensor<1x197x4096xf32>
    %3751 = stablehlo.multiply %3750, %3746 : tensor<1x197x4096xf32>
    %3752 = stablehlo.add %3751, %cst_10 : tensor<1x197x4096xf32>
    %3753 = stablehlo.multiply %3752, %3746 : tensor<1x197x4096xf32>
    %3754 = stablehlo.add %3753, %cst_11 : tensor<1x197x4096xf32>
    %3755 = stablehlo.multiply %3754, %3746 : tensor<1x197x4096xf32>
    %3756 = stablehlo.add %3755, %cst_12 : tensor<1x197x4096xf32>
    %3757 = stablehlo.multiply %3756, %3746 : tensor<1x197x4096xf32>
    %3758 = stablehlo.add %3757, %cst_13 : tensor<1x197x4096xf32>
    %3759 = stablehlo.multiply %cst_14, %3746 : tensor<1x197x4096xf32>
    %3760 = stablehlo.add %3759, %cst_15 : tensor<1x197x4096xf32>
    %3761 = stablehlo.multiply %3760, %3746 : tensor<1x197x4096xf32>
    %3762 = stablehlo.add %3761, %cst_16 : tensor<1x197x4096xf32>
    %3763 = stablehlo.multiply %3762, %3746 : tensor<1x197x4096xf32>
    %3764 = stablehlo.add %3763, %cst_17 : tensor<1x197x4096xf32>
    %3765 = stablehlo.multiply %3764, %3746 : tensor<1x197x4096xf32>
    %3766 = stablehlo.add %3765, %cst_18 : tensor<1x197x4096xf32>
    %3767 = stablehlo.multiply %3745, %3758 : tensor<1x197x4096xf32>
    %3768 = stablehlo.divide %3767, %3766 : tensor<1x197x4096xf32>
    %3769 = stablehlo.clamp %cst_19, %3768, %cst_20 : tensor<1x197x4096xf32>
    %3770 = stablehlo.convert %3769 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %3771 = stablehlo.add %3770, %cst_2 : tensor<1x197x4096xbf16>
    %3772 = stablehlo.multiply %3771, %3742 : tensor<1x197x4096xbf16>
    %3773 = stablehlo.reshape %3772 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %3774 = stablehlo.convert %3773 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %3775 = stablehlo.dot_general %3774, %arg376, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %3776 = stablehlo.broadcast_in_dim %3775, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3777 = stablehlo.multiply %3776, %60 : tensor<197x1024xf32>
    %3778 = stablehlo.broadcast_in_dim %3777, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3779 = stablehlo.broadcast_in_dim %arg377, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3780 = stablehlo.add %3778, %3779 : tensor<197x1024xf32>
    %3781 = stablehlo.convert %3780 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3782 = stablehlo.reshape %3781 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3783 = stablehlo.broadcast_in_dim %arg116, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3784 = stablehlo.broadcast_in_dim %3782, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3785 = stablehlo.multiply %3783, %3784 : tensor<1x197x1024xbf16>
    %3786 = stablehlo.add %3785, %3694 : tensor<1x197x1024xbf16>
    %3787 = stablehlo.convert %3786 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3788 = stablehlo.convert %3787 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3789 = stablehlo.reduce(%3788 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3790 = stablehlo.reshape %3789 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3791 = stablehlo.broadcast_in_dim %3790, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3792 = stablehlo.divide %3791, %15 : tensor<1x197x1xf64>
    %3793 = stablehlo.broadcast_in_dim %3788, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3794 = stablehlo.broadcast_in_dim %3792, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3795 = stablehlo.subtract %3793, %3794 : tensor<1x197x1024xf64>
    %3796 = stablehlo.multiply %3795, %3795 : tensor<1x197x1024xf64>
    %3797 = stablehlo.reduce(%3796 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3798 = stablehlo.reshape %3797 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3799 = stablehlo.broadcast_in_dim %3798, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3800 = stablehlo.divide %3799, %15 : tensor<1x197x1xf64>
    %3801 = stablehlo.convert %3800 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3802 = stablehlo.reduce(%3787 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3803 = stablehlo.reshape %3802 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3804 = stablehlo.broadcast_in_dim %3803, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3805 = stablehlo.divide %3804, %31 : tensor<1x197x1xf32>
    %3806 = stablehlo.broadcast_in_dim %3801, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3807 = stablehlo.add %3806, %36 : tensor<1x197x1xf32>
    %3808 = stablehlo.rsqrt %3807 : tensor<1x197x1xf32>
    %3809 = stablehlo.broadcast_in_dim %3787, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3810 = stablehlo.broadcast_in_dim %3805, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3811 = stablehlo.subtract %3809, %3810 : tensor<1x197x1024xf32>
    %3812 = stablehlo.broadcast_in_dim %3811, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3813 = stablehlo.broadcast_in_dim %3808, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3814 = stablehlo.multiply %3812, %3813 : tensor<1x197x1024xf32>
    %3815 = stablehlo.convert %arg117 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3816 = stablehlo.broadcast_in_dim %3814, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3817 = stablehlo.broadcast_in_dim %3815, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3818 = stablehlo.multiply %3816, %3817 : tensor<1x197x1024xf32>
    %3819 = stablehlo.convert %arg118 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3820 = stablehlo.broadcast_in_dim %3818, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3821 = stablehlo.broadcast_in_dim %3819, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3822 = stablehlo.add %3820, %3821 : tensor<1x197x1024xf32>
    %3823 = stablehlo.convert %3822 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3824 = stablehlo.reshape %3823 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3825 = stablehlo.convert %3824 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3826 = stablehlo.dot_general %3825, %arg378, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3827 = stablehlo.broadcast_in_dim %3826, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3828 = stablehlo.multiply %3827, %60 : tensor<197x1024xf32>
    %3829 = stablehlo.broadcast_in_dim %3828, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3830 = stablehlo.broadcast_in_dim %arg379, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3831 = stablehlo.add %3829, %3830 : tensor<197x1024xf32>
    %3832 = stablehlo.convert %3831 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3833 = stablehlo.reshape %3832 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3834 = stablehlo.dot_general %3824, %arg380, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %3835 = stablehlo.reshape %3834 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3836 = stablehlo.reshape %3835 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3837 = stablehlo.transpose %3836, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3838 = stablehlo.dot_general %3825, %arg381, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3839 = stablehlo.broadcast_in_dim %3838, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3840 = stablehlo.multiply %3839, %60 : tensor<197x1024xf32>
    %3841 = stablehlo.broadcast_in_dim %3840, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3842 = stablehlo.broadcast_in_dim %arg382, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3843 = stablehlo.add %3841, %3842 : tensor<197x1024xf32>
    %3844 = stablehlo.convert %3843 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3845 = stablehlo.reshape %3844 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3846 = stablehlo.reshape %3845 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3847 = stablehlo.transpose %3846, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3848 = stablehlo.reshape %3833 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %3849 = stablehlo.transpose %3848, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3850 = stablehlo.transpose %3837, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %3851 = stablehlo.reshape %3849 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3852 = stablehlo.reshape %3850 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3853 = stablehlo.broadcast_in_dim %3852, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %3854 = stablehlo.dot_general %3851, %3853, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %3855 = stablehlo.reshape %3854 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3856 = stablehlo.broadcast_in_dim %3855, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %3857 = stablehlo.divide %3856, %92 : tensor<1x16x197x197xbf16>
    %3858 = stablehlo.add %3857, %arg383 : tensor<1x16x197x197xbf16>
    %3859 = stablehlo.convert %3858 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %3860 = stablehlo.reduce(%3859 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3861 = stablehlo.reshape %3860 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3862 = stablehlo.broadcast_in_dim %3859, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3863 = stablehlo.broadcast_in_dim %3861, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3864 = stablehlo.subtract %3862, %3863 : tensor<1x16x197x197xf32>
    %3865 = stablehlo.exponential %3864 : tensor<1x16x197x197xf32>
    %3866 = stablehlo.reduce(%3865 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %3867 = stablehlo.reshape %3866 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %3868 = stablehlo.broadcast_in_dim %3865, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %3869 = stablehlo.broadcast_in_dim %3867, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %3870 = stablehlo.divide %3868, %3869 : tensor<1x16x197x197xf32>
    %3871 = stablehlo.convert %3870 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %3872 = stablehlo.reshape %3871 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %3873 = stablehlo.reshape %3847 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3874 = stablehlo.broadcast_in_dim %3873, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3875 = stablehlo.dot_general %3872, %3874, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %3876 = stablehlo.reshape %3875 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %3877 = stablehlo.transpose %3876, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %3878 = stablehlo.reshape %3877 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %3879 = stablehlo.reshape %3878 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3880 = stablehlo.convert %3879 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3881 = stablehlo.dot_general %3880, %arg384, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %3882 = stablehlo.broadcast_in_dim %3881, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3883 = stablehlo.multiply %3882, %60 : tensor<197x1024xf32>
    %3884 = stablehlo.broadcast_in_dim %3883, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3885 = stablehlo.broadcast_in_dim %arg385, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3886 = stablehlo.add %3884, %3885 : tensor<197x1024xf32>
    %3887 = stablehlo.convert %3886 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3888 = stablehlo.reshape %3887 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3889 = stablehlo.broadcast_in_dim %arg119, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3890 = stablehlo.broadcast_in_dim %3888, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3891 = stablehlo.multiply %3889, %3890 : tensor<1x197x1024xbf16>
    %3892 = stablehlo.add %3891, %3786 : tensor<1x197x1024xbf16>
    %3893 = stablehlo.convert %3892 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3894 = stablehlo.convert %3893 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3895 = stablehlo.reduce(%3894 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3896 = stablehlo.reshape %3895 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3897 = stablehlo.broadcast_in_dim %3896, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3898 = stablehlo.divide %3897, %15 : tensor<1x197x1xf64>
    %3899 = stablehlo.broadcast_in_dim %3894, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3900 = stablehlo.broadcast_in_dim %3898, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3901 = stablehlo.subtract %3899, %3900 : tensor<1x197x1024xf64>
    %3902 = stablehlo.multiply %3901, %3901 : tensor<1x197x1024xf64>
    %3903 = stablehlo.reduce(%3902 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3904 = stablehlo.reshape %3903 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3905 = stablehlo.broadcast_in_dim %3904, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3906 = stablehlo.divide %3905, %15 : tensor<1x197x1xf64>
    %3907 = stablehlo.convert %3906 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %3908 = stablehlo.reduce(%3893 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %3909 = stablehlo.reshape %3908 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %3910 = stablehlo.broadcast_in_dim %3909, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3911 = stablehlo.divide %3910, %31 : tensor<1x197x1xf32>
    %3912 = stablehlo.broadcast_in_dim %3907, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %3913 = stablehlo.add %3912, %36 : tensor<1x197x1xf32>
    %3914 = stablehlo.rsqrt %3913 : tensor<1x197x1xf32>
    %3915 = stablehlo.broadcast_in_dim %3893, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3916 = stablehlo.broadcast_in_dim %3911, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3917 = stablehlo.subtract %3915, %3916 : tensor<1x197x1024xf32>
    %3918 = stablehlo.broadcast_in_dim %3917, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3919 = stablehlo.broadcast_in_dim %3914, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %3920 = stablehlo.multiply %3918, %3919 : tensor<1x197x1024xf32>
    %3921 = stablehlo.convert %arg120 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3922 = stablehlo.broadcast_in_dim %3920, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3923 = stablehlo.broadcast_in_dim %3921, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3924 = stablehlo.multiply %3922, %3923 : tensor<1x197x1024xf32>
    %3925 = stablehlo.convert %arg121 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %3926 = stablehlo.broadcast_in_dim %3924, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %3927 = stablehlo.broadcast_in_dim %3925, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %3928 = stablehlo.add %3926, %3927 : tensor<1x197x1024xf32>
    %3929 = stablehlo.convert %3928 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %3930 = stablehlo.reshape %3929 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %3931 = stablehlo.convert %3930 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %3932 = stablehlo.dot_general %3931, %arg386, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %3933 = stablehlo.broadcast_in_dim %3932, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3934 = stablehlo.multiply %3933, %170 : tensor<197x4096xf32>
    %3935 = stablehlo.broadcast_in_dim %3934, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %3936 = stablehlo.broadcast_in_dim %arg387, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %3937 = stablehlo.add %3935, %3936 : tensor<197x4096xf32>
    %3938 = stablehlo.convert %3937 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %3939 = stablehlo.reshape %3938 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %3940 = stablehlo.multiply %3939, %cst_4 : tensor<1x197x4096xbf16>
    %3941 = stablehlo.multiply %3939, %178 : tensor<1x197x4096xbf16>
    %3942 = stablehlo.convert %3941 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %3943 = stablehlo.clamp %cst_5, %3942, %cst_6 : tensor<1x197x4096xf32>
    %3944 = stablehlo.multiply %3943, %3943 : tensor<1x197x4096xf32>
    %3945 = stablehlo.multiply %cst_7, %3944 : tensor<1x197x4096xf32>
    %3946 = stablehlo.add %3945, %cst_8 : tensor<1x197x4096xf32>
    %3947 = stablehlo.multiply %3946, %3944 : tensor<1x197x4096xf32>
    %3948 = stablehlo.add %3947, %cst_9 : tensor<1x197x4096xf32>
    %3949 = stablehlo.multiply %3948, %3944 : tensor<1x197x4096xf32>
    %3950 = stablehlo.add %3949, %cst_10 : tensor<1x197x4096xf32>
    %3951 = stablehlo.multiply %3950, %3944 : tensor<1x197x4096xf32>
    %3952 = stablehlo.add %3951, %cst_11 : tensor<1x197x4096xf32>
    %3953 = stablehlo.multiply %3952, %3944 : tensor<1x197x4096xf32>
    %3954 = stablehlo.add %3953, %cst_12 : tensor<1x197x4096xf32>
    %3955 = stablehlo.multiply %3954, %3944 : tensor<1x197x4096xf32>
    %3956 = stablehlo.add %3955, %cst_13 : tensor<1x197x4096xf32>
    %3957 = stablehlo.multiply %cst_14, %3944 : tensor<1x197x4096xf32>
    %3958 = stablehlo.add %3957, %cst_15 : tensor<1x197x4096xf32>
    %3959 = stablehlo.multiply %3958, %3944 : tensor<1x197x4096xf32>
    %3960 = stablehlo.add %3959, %cst_16 : tensor<1x197x4096xf32>
    %3961 = stablehlo.multiply %3960, %3944 : tensor<1x197x4096xf32>
    %3962 = stablehlo.add %3961, %cst_17 : tensor<1x197x4096xf32>
    %3963 = stablehlo.multiply %3962, %3944 : tensor<1x197x4096xf32>
    %3964 = stablehlo.add %3963, %cst_18 : tensor<1x197x4096xf32>
    %3965 = stablehlo.multiply %3943, %3956 : tensor<1x197x4096xf32>
    %3966 = stablehlo.divide %3965, %3964 : tensor<1x197x4096xf32>
    %3967 = stablehlo.clamp %cst_19, %3966, %cst_20 : tensor<1x197x4096xf32>
    %3968 = stablehlo.convert %3967 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %3969 = stablehlo.add %3968, %cst_2 : tensor<1x197x4096xbf16>
    %3970 = stablehlo.multiply %3969, %3940 : tensor<1x197x4096xbf16>
    %3971 = stablehlo.reshape %3970 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %3972 = stablehlo.convert %3971 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %3973 = stablehlo.dot_general %3972, %arg388, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %3974 = stablehlo.broadcast_in_dim %3973, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3975 = stablehlo.multiply %3974, %60 : tensor<197x1024xf32>
    %3976 = stablehlo.broadcast_in_dim %3975, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %3977 = stablehlo.broadcast_in_dim %arg389, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %3978 = stablehlo.add %3976, %3977 : tensor<197x1024xf32>
    %3979 = stablehlo.convert %3978 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %3980 = stablehlo.reshape %3979 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3981 = stablehlo.broadcast_in_dim %arg122, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %3982 = stablehlo.broadcast_in_dim %3980, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %3983 = stablehlo.multiply %3981, %3982 : tensor<1x197x1024xbf16>
    %3984 = stablehlo.add %3983, %3892 : tensor<1x197x1024xbf16>
    %3985 = stablehlo.convert %3984 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %3986 = stablehlo.convert %3985 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %3987 = stablehlo.reduce(%3986 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3988 = stablehlo.reshape %3987 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3989 = stablehlo.broadcast_in_dim %3988, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3990 = stablehlo.divide %3989, %15 : tensor<1x197x1xf64>
    %3991 = stablehlo.broadcast_in_dim %3986, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %3992 = stablehlo.broadcast_in_dim %3990, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %3993 = stablehlo.subtract %3991, %3992 : tensor<1x197x1024xf64>
    %3994 = stablehlo.multiply %3993, %3993 : tensor<1x197x1024xf64>
    %3995 = stablehlo.reduce(%3994 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %3996 = stablehlo.reshape %3995 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %3997 = stablehlo.broadcast_in_dim %3996, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %3998 = stablehlo.divide %3997, %15 : tensor<1x197x1xf64>
    %3999 = stablehlo.convert %3998 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4000 = stablehlo.reduce(%3985 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4001 = stablehlo.reshape %4000 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4002 = stablehlo.broadcast_in_dim %4001, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4003 = stablehlo.divide %4002, %31 : tensor<1x197x1xf32>
    %4004 = stablehlo.broadcast_in_dim %3999, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4005 = stablehlo.add %4004, %36 : tensor<1x197x1xf32>
    %4006 = stablehlo.rsqrt %4005 : tensor<1x197x1xf32>
    %4007 = stablehlo.broadcast_in_dim %3985, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4008 = stablehlo.broadcast_in_dim %4003, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4009 = stablehlo.subtract %4007, %4008 : tensor<1x197x1024xf32>
    %4010 = stablehlo.broadcast_in_dim %4009, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4011 = stablehlo.broadcast_in_dim %4006, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4012 = stablehlo.multiply %4010, %4011 : tensor<1x197x1024xf32>
    %4013 = stablehlo.convert %arg123 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4014 = stablehlo.broadcast_in_dim %4012, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4015 = stablehlo.broadcast_in_dim %4013, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4016 = stablehlo.multiply %4014, %4015 : tensor<1x197x1024xf32>
    %4017 = stablehlo.convert %arg124 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4018 = stablehlo.broadcast_in_dim %4016, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4019 = stablehlo.broadcast_in_dim %4017, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4020 = stablehlo.add %4018, %4019 : tensor<1x197x1024xf32>
    %4021 = stablehlo.convert %4020 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4022 = stablehlo.reshape %4021 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4023 = stablehlo.convert %4022 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4024 = stablehlo.dot_general %4023, %arg390, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4025 = stablehlo.broadcast_in_dim %4024, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4026 = stablehlo.multiply %4025, %60 : tensor<197x1024xf32>
    %4027 = stablehlo.broadcast_in_dim %4026, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4028 = stablehlo.broadcast_in_dim %arg391, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4029 = stablehlo.add %4027, %4028 : tensor<197x1024xf32>
    %4030 = stablehlo.convert %4029 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4031 = stablehlo.reshape %4030 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4032 = stablehlo.dot_general %4022, %arg392, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %4033 = stablehlo.reshape %4032 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4034 = stablehlo.reshape %4033 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4035 = stablehlo.transpose %4034, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4036 = stablehlo.dot_general %4023, %arg393, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4037 = stablehlo.broadcast_in_dim %4036, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4038 = stablehlo.multiply %4037, %60 : tensor<197x1024xf32>
    %4039 = stablehlo.broadcast_in_dim %4038, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4040 = stablehlo.broadcast_in_dim %arg394, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4041 = stablehlo.add %4039, %4040 : tensor<197x1024xf32>
    %4042 = stablehlo.convert %4041 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4043 = stablehlo.reshape %4042 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4044 = stablehlo.reshape %4043 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4045 = stablehlo.transpose %4044, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4046 = stablehlo.reshape %4031 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4047 = stablehlo.transpose %4046, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4048 = stablehlo.transpose %4035, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %4049 = stablehlo.reshape %4047 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4050 = stablehlo.reshape %4048 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4051 = stablehlo.broadcast_in_dim %4050, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4052 = stablehlo.dot_general %4049, %4051, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %4053 = stablehlo.reshape %4052 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4054 = stablehlo.broadcast_in_dim %4053, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4055 = stablehlo.divide %4054, %92 : tensor<1x16x197x197xbf16>
    %4056 = stablehlo.add %4055, %arg395 : tensor<1x16x197x197xbf16>
    %4057 = stablehlo.convert %4056 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %4058 = stablehlo.reduce(%4057 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4059 = stablehlo.reshape %4058 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4060 = stablehlo.broadcast_in_dim %4057, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4061 = stablehlo.broadcast_in_dim %4059, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4062 = stablehlo.subtract %4060, %4061 : tensor<1x16x197x197xf32>
    %4063 = stablehlo.exponential %4062 : tensor<1x16x197x197xf32>
    %4064 = stablehlo.reduce(%4063 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4065 = stablehlo.reshape %4064 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4066 = stablehlo.broadcast_in_dim %4063, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4067 = stablehlo.broadcast_in_dim %4065, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4068 = stablehlo.divide %4066, %4067 : tensor<1x16x197x197xf32>
    %4069 = stablehlo.convert %4068 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %4070 = stablehlo.reshape %4069 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %4071 = stablehlo.reshape %4045 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4072 = stablehlo.broadcast_in_dim %4071, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4073 = stablehlo.dot_general %4070, %4072, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4074 = stablehlo.reshape %4073 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4075 = stablehlo.transpose %4074, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %4076 = stablehlo.reshape %4075 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %4077 = stablehlo.reshape %4076 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4078 = stablehlo.convert %4077 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4079 = stablehlo.dot_general %4078, %arg396, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4080 = stablehlo.broadcast_in_dim %4079, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4081 = stablehlo.multiply %4080, %60 : tensor<197x1024xf32>
    %4082 = stablehlo.broadcast_in_dim %4081, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4083 = stablehlo.broadcast_in_dim %arg397, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4084 = stablehlo.add %4082, %4083 : tensor<197x1024xf32>
    %4085 = stablehlo.convert %4084 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4086 = stablehlo.reshape %4085 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4087 = stablehlo.broadcast_in_dim %arg125, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4088 = stablehlo.broadcast_in_dim %4086, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4089 = stablehlo.multiply %4087, %4088 : tensor<1x197x1024xbf16>
    %4090 = stablehlo.add %4089, %3984 : tensor<1x197x1024xbf16>
    %4091 = stablehlo.convert %4090 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4092 = stablehlo.convert %4091 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4093 = stablehlo.reduce(%4092 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4094 = stablehlo.reshape %4093 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4095 = stablehlo.broadcast_in_dim %4094, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4096 = stablehlo.divide %4095, %15 : tensor<1x197x1xf64>
    %4097 = stablehlo.broadcast_in_dim %4092, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4098 = stablehlo.broadcast_in_dim %4096, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4099 = stablehlo.subtract %4097, %4098 : tensor<1x197x1024xf64>
    %4100 = stablehlo.multiply %4099, %4099 : tensor<1x197x1024xf64>
    %4101 = stablehlo.reduce(%4100 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4102 = stablehlo.reshape %4101 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4103 = stablehlo.broadcast_in_dim %4102, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4104 = stablehlo.divide %4103, %15 : tensor<1x197x1xf64>
    %4105 = stablehlo.convert %4104 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4106 = stablehlo.reduce(%4091 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4107 = stablehlo.reshape %4106 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4108 = stablehlo.broadcast_in_dim %4107, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4109 = stablehlo.divide %4108, %31 : tensor<1x197x1xf32>
    %4110 = stablehlo.broadcast_in_dim %4105, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4111 = stablehlo.add %4110, %36 : tensor<1x197x1xf32>
    %4112 = stablehlo.rsqrt %4111 : tensor<1x197x1xf32>
    %4113 = stablehlo.broadcast_in_dim %4091, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4114 = stablehlo.broadcast_in_dim %4109, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4115 = stablehlo.subtract %4113, %4114 : tensor<1x197x1024xf32>
    %4116 = stablehlo.broadcast_in_dim %4115, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4117 = stablehlo.broadcast_in_dim %4112, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4118 = stablehlo.multiply %4116, %4117 : tensor<1x197x1024xf32>
    %4119 = stablehlo.convert %arg126 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4120 = stablehlo.broadcast_in_dim %4118, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4121 = stablehlo.broadcast_in_dim %4119, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4122 = stablehlo.multiply %4120, %4121 : tensor<1x197x1024xf32>
    %4123 = stablehlo.convert %arg127 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4124 = stablehlo.broadcast_in_dim %4122, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4125 = stablehlo.broadcast_in_dim %4123, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4126 = stablehlo.add %4124, %4125 : tensor<1x197x1024xf32>
    %4127 = stablehlo.convert %4126 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4128 = stablehlo.reshape %4127 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4129 = stablehlo.convert %4128 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4130 = stablehlo.dot_general %4129, %arg398, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %4131 = stablehlo.broadcast_in_dim %4130, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4132 = stablehlo.multiply %4131, %170 : tensor<197x4096xf32>
    %4133 = stablehlo.broadcast_in_dim %4132, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4134 = stablehlo.broadcast_in_dim %arg399, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %4135 = stablehlo.add %4133, %4134 : tensor<197x4096xf32>
    %4136 = stablehlo.convert %4135 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %4137 = stablehlo.reshape %4136 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %4138 = stablehlo.multiply %4137, %cst_4 : tensor<1x197x4096xbf16>
    %4139 = stablehlo.multiply %4137, %178 : tensor<1x197x4096xbf16>
    %4140 = stablehlo.convert %4139 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %4141 = stablehlo.clamp %cst_5, %4140, %cst_6 : tensor<1x197x4096xf32>
    %4142 = stablehlo.multiply %4141, %4141 : tensor<1x197x4096xf32>
    %4143 = stablehlo.multiply %cst_7, %4142 : tensor<1x197x4096xf32>
    %4144 = stablehlo.add %4143, %cst_8 : tensor<1x197x4096xf32>
    %4145 = stablehlo.multiply %4144, %4142 : tensor<1x197x4096xf32>
    %4146 = stablehlo.add %4145, %cst_9 : tensor<1x197x4096xf32>
    %4147 = stablehlo.multiply %4146, %4142 : tensor<1x197x4096xf32>
    %4148 = stablehlo.add %4147, %cst_10 : tensor<1x197x4096xf32>
    %4149 = stablehlo.multiply %4148, %4142 : tensor<1x197x4096xf32>
    %4150 = stablehlo.add %4149, %cst_11 : tensor<1x197x4096xf32>
    %4151 = stablehlo.multiply %4150, %4142 : tensor<1x197x4096xf32>
    %4152 = stablehlo.add %4151, %cst_12 : tensor<1x197x4096xf32>
    %4153 = stablehlo.multiply %4152, %4142 : tensor<1x197x4096xf32>
    %4154 = stablehlo.add %4153, %cst_13 : tensor<1x197x4096xf32>
    %4155 = stablehlo.multiply %cst_14, %4142 : tensor<1x197x4096xf32>
    %4156 = stablehlo.add %4155, %cst_15 : tensor<1x197x4096xf32>
    %4157 = stablehlo.multiply %4156, %4142 : tensor<1x197x4096xf32>
    %4158 = stablehlo.add %4157, %cst_16 : tensor<1x197x4096xf32>
    %4159 = stablehlo.multiply %4158, %4142 : tensor<1x197x4096xf32>
    %4160 = stablehlo.add %4159, %cst_17 : tensor<1x197x4096xf32>
    %4161 = stablehlo.multiply %4160, %4142 : tensor<1x197x4096xf32>
    %4162 = stablehlo.add %4161, %cst_18 : tensor<1x197x4096xf32>
    %4163 = stablehlo.multiply %4141, %4154 : tensor<1x197x4096xf32>
    %4164 = stablehlo.divide %4163, %4162 : tensor<1x197x4096xf32>
    %4165 = stablehlo.clamp %cst_19, %4164, %cst_20 : tensor<1x197x4096xf32>
    %4166 = stablehlo.convert %4165 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %4167 = stablehlo.add %4166, %cst_2 : tensor<1x197x4096xbf16>
    %4168 = stablehlo.multiply %4167, %4138 : tensor<1x197x4096xbf16>
    %4169 = stablehlo.reshape %4168 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %4170 = stablehlo.convert %4169 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %4171 = stablehlo.dot_general %4170, %arg400, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %4172 = stablehlo.broadcast_in_dim %4171, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4173 = stablehlo.multiply %4172, %60 : tensor<197x1024xf32>
    %4174 = stablehlo.broadcast_in_dim %4173, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4175 = stablehlo.broadcast_in_dim %arg401, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4176 = stablehlo.add %4174, %4175 : tensor<197x1024xf32>
    %4177 = stablehlo.convert %4176 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4178 = stablehlo.reshape %4177 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4179 = stablehlo.broadcast_in_dim %arg128, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4180 = stablehlo.broadcast_in_dim %4178, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4181 = stablehlo.multiply %4179, %4180 : tensor<1x197x1024xbf16>
    %4182 = stablehlo.add %4181, %4090 : tensor<1x197x1024xbf16>
    %4183 = stablehlo.convert %4182 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4184 = stablehlo.convert %4183 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4185 = stablehlo.reduce(%4184 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4186 = stablehlo.reshape %4185 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4187 = stablehlo.broadcast_in_dim %4186, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4188 = stablehlo.divide %4187, %15 : tensor<1x197x1xf64>
    %4189 = stablehlo.broadcast_in_dim %4184, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4190 = stablehlo.broadcast_in_dim %4188, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4191 = stablehlo.subtract %4189, %4190 : tensor<1x197x1024xf64>
    %4192 = stablehlo.multiply %4191, %4191 : tensor<1x197x1024xf64>
    %4193 = stablehlo.reduce(%4192 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4194 = stablehlo.reshape %4193 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4195 = stablehlo.broadcast_in_dim %4194, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4196 = stablehlo.divide %4195, %15 : tensor<1x197x1xf64>
    %4197 = stablehlo.convert %4196 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4198 = stablehlo.reduce(%4183 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4199 = stablehlo.reshape %4198 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4200 = stablehlo.broadcast_in_dim %4199, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4201 = stablehlo.divide %4200, %31 : tensor<1x197x1xf32>
    %4202 = stablehlo.broadcast_in_dim %4197, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4203 = stablehlo.add %4202, %36 : tensor<1x197x1xf32>
    %4204 = stablehlo.rsqrt %4203 : tensor<1x197x1xf32>
    %4205 = stablehlo.broadcast_in_dim %4183, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4206 = stablehlo.broadcast_in_dim %4201, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4207 = stablehlo.subtract %4205, %4206 : tensor<1x197x1024xf32>
    %4208 = stablehlo.broadcast_in_dim %4207, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4209 = stablehlo.broadcast_in_dim %4204, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4210 = stablehlo.multiply %4208, %4209 : tensor<1x197x1024xf32>
    %4211 = stablehlo.convert %arg129 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4212 = stablehlo.broadcast_in_dim %4210, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4213 = stablehlo.broadcast_in_dim %4211, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4214 = stablehlo.multiply %4212, %4213 : tensor<1x197x1024xf32>
    %4215 = stablehlo.convert %arg130 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4216 = stablehlo.broadcast_in_dim %4214, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4217 = stablehlo.broadcast_in_dim %4215, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4218 = stablehlo.add %4216, %4217 : tensor<1x197x1024xf32>
    %4219 = stablehlo.convert %4218 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4220 = stablehlo.reshape %4219 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4221 = stablehlo.convert %4220 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4222 = stablehlo.dot_general %4221, %arg402, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4223 = stablehlo.broadcast_in_dim %4222, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4224 = stablehlo.multiply %4223, %60 : tensor<197x1024xf32>
    %4225 = stablehlo.broadcast_in_dim %4224, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4226 = stablehlo.broadcast_in_dim %arg403, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4227 = stablehlo.add %4225, %4226 : tensor<197x1024xf32>
    %4228 = stablehlo.convert %4227 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4229 = stablehlo.reshape %4228 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4230 = stablehlo.dot_general %4220, %arg404, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %4231 = stablehlo.reshape %4230 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4232 = stablehlo.reshape %4231 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4233 = stablehlo.transpose %4232, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4234 = stablehlo.dot_general %4221, %arg405, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4235 = stablehlo.broadcast_in_dim %4234, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4236 = stablehlo.multiply %4235, %60 : tensor<197x1024xf32>
    %4237 = stablehlo.broadcast_in_dim %4236, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4238 = stablehlo.broadcast_in_dim %arg406, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4239 = stablehlo.add %4237, %4238 : tensor<197x1024xf32>
    %4240 = stablehlo.convert %4239 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4241 = stablehlo.reshape %4240 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4242 = stablehlo.reshape %4241 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4243 = stablehlo.transpose %4242, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4244 = stablehlo.reshape %4229 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4245 = stablehlo.transpose %4244, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4246 = stablehlo.transpose %4233, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %4247 = stablehlo.reshape %4245 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4248 = stablehlo.reshape %4246 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4249 = stablehlo.broadcast_in_dim %4248, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4250 = stablehlo.dot_general %4247, %4249, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %4251 = stablehlo.reshape %4250 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4252 = stablehlo.broadcast_in_dim %4251, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4253 = stablehlo.divide %4252, %92 : tensor<1x16x197x197xbf16>
    %4254 = stablehlo.add %4253, %arg407 : tensor<1x16x197x197xbf16>
    %4255 = stablehlo.convert %4254 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %4256 = stablehlo.reduce(%4255 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4257 = stablehlo.reshape %4256 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4258 = stablehlo.broadcast_in_dim %4255, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4259 = stablehlo.broadcast_in_dim %4257, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4260 = stablehlo.subtract %4258, %4259 : tensor<1x16x197x197xf32>
    %4261 = stablehlo.exponential %4260 : tensor<1x16x197x197xf32>
    %4262 = stablehlo.reduce(%4261 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4263 = stablehlo.reshape %4262 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4264 = stablehlo.broadcast_in_dim %4261, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4265 = stablehlo.broadcast_in_dim %4263, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4266 = stablehlo.divide %4264, %4265 : tensor<1x16x197x197xf32>
    %4267 = stablehlo.convert %4266 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %4268 = stablehlo.reshape %4267 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %4269 = stablehlo.reshape %4243 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4270 = stablehlo.broadcast_in_dim %4269, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4271 = stablehlo.dot_general %4268, %4270, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4272 = stablehlo.reshape %4271 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4273 = stablehlo.transpose %4272, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %4274 = stablehlo.reshape %4273 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %4275 = stablehlo.reshape %4274 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4276 = stablehlo.convert %4275 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4277 = stablehlo.dot_general %4276, %arg408, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4278 = stablehlo.broadcast_in_dim %4277, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4279 = stablehlo.multiply %4278, %60 : tensor<197x1024xf32>
    %4280 = stablehlo.broadcast_in_dim %4279, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4281 = stablehlo.broadcast_in_dim %arg409, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4282 = stablehlo.add %4280, %4281 : tensor<197x1024xf32>
    %4283 = stablehlo.convert %4282 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4284 = stablehlo.reshape %4283 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4285 = stablehlo.broadcast_in_dim %arg131, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4286 = stablehlo.broadcast_in_dim %4284, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4287 = stablehlo.multiply %4285, %4286 : tensor<1x197x1024xbf16>
    %4288 = stablehlo.add %4287, %4182 : tensor<1x197x1024xbf16>
    %4289 = stablehlo.convert %4288 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4290 = stablehlo.convert %4289 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4291 = stablehlo.reduce(%4290 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4292 = stablehlo.reshape %4291 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4293 = stablehlo.broadcast_in_dim %4292, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4294 = stablehlo.divide %4293, %15 : tensor<1x197x1xf64>
    %4295 = stablehlo.broadcast_in_dim %4290, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4296 = stablehlo.broadcast_in_dim %4294, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4297 = stablehlo.subtract %4295, %4296 : tensor<1x197x1024xf64>
    %4298 = stablehlo.multiply %4297, %4297 : tensor<1x197x1024xf64>
    %4299 = stablehlo.reduce(%4298 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4300 = stablehlo.reshape %4299 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4301 = stablehlo.broadcast_in_dim %4300, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4302 = stablehlo.divide %4301, %15 : tensor<1x197x1xf64>
    %4303 = stablehlo.convert %4302 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4304 = stablehlo.reduce(%4289 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4305 = stablehlo.reshape %4304 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4306 = stablehlo.broadcast_in_dim %4305, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4307 = stablehlo.divide %4306, %31 : tensor<1x197x1xf32>
    %4308 = stablehlo.broadcast_in_dim %4303, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4309 = stablehlo.add %4308, %36 : tensor<1x197x1xf32>
    %4310 = stablehlo.rsqrt %4309 : tensor<1x197x1xf32>
    %4311 = stablehlo.broadcast_in_dim %4289, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4312 = stablehlo.broadcast_in_dim %4307, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4313 = stablehlo.subtract %4311, %4312 : tensor<1x197x1024xf32>
    %4314 = stablehlo.broadcast_in_dim %4313, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4315 = stablehlo.broadcast_in_dim %4310, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4316 = stablehlo.multiply %4314, %4315 : tensor<1x197x1024xf32>
    %4317 = stablehlo.convert %arg132 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4318 = stablehlo.broadcast_in_dim %4316, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4319 = stablehlo.broadcast_in_dim %4317, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4320 = stablehlo.multiply %4318, %4319 : tensor<1x197x1024xf32>
    %4321 = stablehlo.convert %arg133 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4322 = stablehlo.broadcast_in_dim %4320, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4323 = stablehlo.broadcast_in_dim %4321, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4324 = stablehlo.add %4322, %4323 : tensor<1x197x1024xf32>
    %4325 = stablehlo.convert %4324 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4326 = stablehlo.reshape %4325 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4327 = stablehlo.convert %4326 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4328 = stablehlo.dot_general %4327, %arg410, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %4329 = stablehlo.broadcast_in_dim %4328, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4330 = stablehlo.multiply %4329, %170 : tensor<197x4096xf32>
    %4331 = stablehlo.broadcast_in_dim %4330, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4332 = stablehlo.broadcast_in_dim %arg411, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %4333 = stablehlo.add %4331, %4332 : tensor<197x4096xf32>
    %4334 = stablehlo.convert %4333 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %4335 = stablehlo.reshape %4334 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %4336 = stablehlo.multiply %4335, %cst_4 : tensor<1x197x4096xbf16>
    %4337 = stablehlo.multiply %4335, %178 : tensor<1x197x4096xbf16>
    %4338 = stablehlo.convert %4337 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %4339 = stablehlo.clamp %cst_5, %4338, %cst_6 : tensor<1x197x4096xf32>
    %4340 = stablehlo.multiply %4339, %4339 : tensor<1x197x4096xf32>
    %4341 = stablehlo.multiply %cst_7, %4340 : tensor<1x197x4096xf32>
    %4342 = stablehlo.add %4341, %cst_8 : tensor<1x197x4096xf32>
    %4343 = stablehlo.multiply %4342, %4340 : tensor<1x197x4096xf32>
    %4344 = stablehlo.add %4343, %cst_9 : tensor<1x197x4096xf32>
    %4345 = stablehlo.multiply %4344, %4340 : tensor<1x197x4096xf32>
    %4346 = stablehlo.add %4345, %cst_10 : tensor<1x197x4096xf32>
    %4347 = stablehlo.multiply %4346, %4340 : tensor<1x197x4096xf32>
    %4348 = stablehlo.add %4347, %cst_11 : tensor<1x197x4096xf32>
    %4349 = stablehlo.multiply %4348, %4340 : tensor<1x197x4096xf32>
    %4350 = stablehlo.add %4349, %cst_12 : tensor<1x197x4096xf32>
    %4351 = stablehlo.multiply %4350, %4340 : tensor<1x197x4096xf32>
    %4352 = stablehlo.add %4351, %cst_13 : tensor<1x197x4096xf32>
    %4353 = stablehlo.multiply %cst_14, %4340 : tensor<1x197x4096xf32>
    %4354 = stablehlo.add %4353, %cst_15 : tensor<1x197x4096xf32>
    %4355 = stablehlo.multiply %4354, %4340 : tensor<1x197x4096xf32>
    %4356 = stablehlo.add %4355, %cst_16 : tensor<1x197x4096xf32>
    %4357 = stablehlo.multiply %4356, %4340 : tensor<1x197x4096xf32>
    %4358 = stablehlo.add %4357, %cst_17 : tensor<1x197x4096xf32>
    %4359 = stablehlo.multiply %4358, %4340 : tensor<1x197x4096xf32>
    %4360 = stablehlo.add %4359, %cst_18 : tensor<1x197x4096xf32>
    %4361 = stablehlo.multiply %4339, %4352 : tensor<1x197x4096xf32>
    %4362 = stablehlo.divide %4361, %4360 : tensor<1x197x4096xf32>
    %4363 = stablehlo.clamp %cst_19, %4362, %cst_20 : tensor<1x197x4096xf32>
    %4364 = stablehlo.convert %4363 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %4365 = stablehlo.add %4364, %cst_2 : tensor<1x197x4096xbf16>
    %4366 = stablehlo.multiply %4365, %4336 : tensor<1x197x4096xbf16>
    %4367 = stablehlo.reshape %4366 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %4368 = stablehlo.convert %4367 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %4369 = stablehlo.dot_general %4368, %arg412, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %4370 = stablehlo.broadcast_in_dim %4369, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4371 = stablehlo.multiply %4370, %60 : tensor<197x1024xf32>
    %4372 = stablehlo.broadcast_in_dim %4371, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4373 = stablehlo.broadcast_in_dim %arg413, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4374 = stablehlo.add %4372, %4373 : tensor<197x1024xf32>
    %4375 = stablehlo.convert %4374 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4376 = stablehlo.reshape %4375 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4377 = stablehlo.broadcast_in_dim %arg134, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4378 = stablehlo.broadcast_in_dim %4376, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4379 = stablehlo.multiply %4377, %4378 : tensor<1x197x1024xbf16>
    %4380 = stablehlo.add %4379, %4288 : tensor<1x197x1024xbf16>
    %4381 = stablehlo.convert %4380 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4382 = stablehlo.convert %4381 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4383 = stablehlo.reduce(%4382 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4384 = stablehlo.reshape %4383 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4385 = stablehlo.broadcast_in_dim %4384, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4386 = stablehlo.divide %4385, %15 : tensor<1x197x1xf64>
    %4387 = stablehlo.broadcast_in_dim %4382, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4388 = stablehlo.broadcast_in_dim %4386, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4389 = stablehlo.subtract %4387, %4388 : tensor<1x197x1024xf64>
    %4390 = stablehlo.multiply %4389, %4389 : tensor<1x197x1024xf64>
    %4391 = stablehlo.reduce(%4390 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4392 = stablehlo.reshape %4391 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4393 = stablehlo.broadcast_in_dim %4392, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4394 = stablehlo.divide %4393, %15 : tensor<1x197x1xf64>
    %4395 = stablehlo.convert %4394 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4396 = stablehlo.reduce(%4381 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4397 = stablehlo.reshape %4396 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4398 = stablehlo.broadcast_in_dim %4397, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4399 = stablehlo.divide %4398, %31 : tensor<1x197x1xf32>
    %4400 = stablehlo.broadcast_in_dim %4395, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4401 = stablehlo.add %4400, %36 : tensor<1x197x1xf32>
    %4402 = stablehlo.rsqrt %4401 : tensor<1x197x1xf32>
    %4403 = stablehlo.broadcast_in_dim %4381, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4404 = stablehlo.broadcast_in_dim %4399, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4405 = stablehlo.subtract %4403, %4404 : tensor<1x197x1024xf32>
    %4406 = stablehlo.broadcast_in_dim %4405, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4407 = stablehlo.broadcast_in_dim %4402, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4408 = stablehlo.multiply %4406, %4407 : tensor<1x197x1024xf32>
    %4409 = stablehlo.convert %arg135 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4410 = stablehlo.broadcast_in_dim %4408, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4411 = stablehlo.broadcast_in_dim %4409, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4412 = stablehlo.multiply %4410, %4411 : tensor<1x197x1024xf32>
    %4413 = stablehlo.convert %arg136 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4414 = stablehlo.broadcast_in_dim %4412, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4415 = stablehlo.broadcast_in_dim %4413, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4416 = stablehlo.add %4414, %4415 : tensor<1x197x1024xf32>
    %4417 = stablehlo.convert %4416 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4418 = stablehlo.reshape %4417 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4419 = stablehlo.convert %4418 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4420 = stablehlo.dot_general %4419, %arg414, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4421 = stablehlo.broadcast_in_dim %4420, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4422 = stablehlo.multiply %4421, %60 : tensor<197x1024xf32>
    %4423 = stablehlo.broadcast_in_dim %4422, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4424 = stablehlo.broadcast_in_dim %arg415, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4425 = stablehlo.add %4423, %4424 : tensor<197x1024xf32>
    %4426 = stablehlo.convert %4425 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4427 = stablehlo.reshape %4426 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4428 = stablehlo.dot_general %4418, %arg416, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %4429 = stablehlo.reshape %4428 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4430 = stablehlo.reshape %4429 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4431 = stablehlo.transpose %4430, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4432 = stablehlo.dot_general %4419, %arg417, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4433 = stablehlo.broadcast_in_dim %4432, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4434 = stablehlo.multiply %4433, %60 : tensor<197x1024xf32>
    %4435 = stablehlo.broadcast_in_dim %4434, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4436 = stablehlo.broadcast_in_dim %arg418, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4437 = stablehlo.add %4435, %4436 : tensor<197x1024xf32>
    %4438 = stablehlo.convert %4437 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4439 = stablehlo.reshape %4438 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4440 = stablehlo.reshape %4439 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4441 = stablehlo.transpose %4440, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4442 = stablehlo.reshape %4427 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4443 = stablehlo.transpose %4442, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4444 = stablehlo.transpose %4431, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %4445 = stablehlo.reshape %4443 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4446 = stablehlo.reshape %4444 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4447 = stablehlo.broadcast_in_dim %4446, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4448 = stablehlo.dot_general %4445, %4447, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %4449 = stablehlo.reshape %4448 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4450 = stablehlo.broadcast_in_dim %4449, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4451 = stablehlo.divide %4450, %92 : tensor<1x16x197x197xbf16>
    %4452 = stablehlo.add %4451, %arg419 : tensor<1x16x197x197xbf16>
    %4453 = stablehlo.convert %4452 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %4454 = stablehlo.reduce(%4453 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4455 = stablehlo.reshape %4454 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4456 = stablehlo.broadcast_in_dim %4453, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4457 = stablehlo.broadcast_in_dim %4455, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4458 = stablehlo.subtract %4456, %4457 : tensor<1x16x197x197xf32>
    %4459 = stablehlo.exponential %4458 : tensor<1x16x197x197xf32>
    %4460 = stablehlo.reduce(%4459 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4461 = stablehlo.reshape %4460 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4462 = stablehlo.broadcast_in_dim %4459, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4463 = stablehlo.broadcast_in_dim %4461, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4464 = stablehlo.divide %4462, %4463 : tensor<1x16x197x197xf32>
    %4465 = stablehlo.convert %4464 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %4466 = stablehlo.reshape %4465 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %4467 = stablehlo.reshape %4441 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4468 = stablehlo.broadcast_in_dim %4467, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4469 = stablehlo.dot_general %4466, %4468, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4470 = stablehlo.reshape %4469 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4471 = stablehlo.transpose %4470, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %4472 = stablehlo.reshape %4471 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %4473 = stablehlo.reshape %4472 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4474 = stablehlo.convert %4473 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4475 = stablehlo.dot_general %4474, %arg420, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4476 = stablehlo.broadcast_in_dim %4475, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4477 = stablehlo.multiply %4476, %60 : tensor<197x1024xf32>
    %4478 = stablehlo.broadcast_in_dim %4477, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4479 = stablehlo.broadcast_in_dim %arg421, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4480 = stablehlo.add %4478, %4479 : tensor<197x1024xf32>
    %4481 = stablehlo.convert %4480 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4482 = stablehlo.reshape %4481 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4483 = stablehlo.broadcast_in_dim %arg137, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4484 = stablehlo.broadcast_in_dim %4482, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4485 = stablehlo.multiply %4483, %4484 : tensor<1x197x1024xbf16>
    %4486 = stablehlo.add %4485, %4380 : tensor<1x197x1024xbf16>
    %4487 = stablehlo.convert %4486 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4488 = stablehlo.convert %4487 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4489 = stablehlo.reduce(%4488 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4490 = stablehlo.reshape %4489 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4491 = stablehlo.broadcast_in_dim %4490, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4492 = stablehlo.divide %4491, %15 : tensor<1x197x1xf64>
    %4493 = stablehlo.broadcast_in_dim %4488, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4494 = stablehlo.broadcast_in_dim %4492, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4495 = stablehlo.subtract %4493, %4494 : tensor<1x197x1024xf64>
    %4496 = stablehlo.multiply %4495, %4495 : tensor<1x197x1024xf64>
    %4497 = stablehlo.reduce(%4496 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4498 = stablehlo.reshape %4497 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4499 = stablehlo.broadcast_in_dim %4498, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4500 = stablehlo.divide %4499, %15 : tensor<1x197x1xf64>
    %4501 = stablehlo.convert %4500 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4502 = stablehlo.reduce(%4487 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4503 = stablehlo.reshape %4502 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4504 = stablehlo.broadcast_in_dim %4503, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4505 = stablehlo.divide %4504, %31 : tensor<1x197x1xf32>
    %4506 = stablehlo.broadcast_in_dim %4501, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4507 = stablehlo.add %4506, %36 : tensor<1x197x1xf32>
    %4508 = stablehlo.rsqrt %4507 : tensor<1x197x1xf32>
    %4509 = stablehlo.broadcast_in_dim %4487, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4510 = stablehlo.broadcast_in_dim %4505, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4511 = stablehlo.subtract %4509, %4510 : tensor<1x197x1024xf32>
    %4512 = stablehlo.broadcast_in_dim %4511, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4513 = stablehlo.broadcast_in_dim %4508, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4514 = stablehlo.multiply %4512, %4513 : tensor<1x197x1024xf32>
    %4515 = stablehlo.convert %arg138 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4516 = stablehlo.broadcast_in_dim %4514, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4517 = stablehlo.broadcast_in_dim %4515, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4518 = stablehlo.multiply %4516, %4517 : tensor<1x197x1024xf32>
    %4519 = stablehlo.convert %arg139 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4520 = stablehlo.broadcast_in_dim %4518, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4521 = stablehlo.broadcast_in_dim %4519, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4522 = stablehlo.add %4520, %4521 : tensor<1x197x1024xf32>
    %4523 = stablehlo.convert %4522 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4524 = stablehlo.reshape %4523 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4525 = stablehlo.convert %4524 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4526 = stablehlo.dot_general %4525, %arg422, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %4527 = stablehlo.broadcast_in_dim %4526, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4528 = stablehlo.multiply %4527, %170 : tensor<197x4096xf32>
    %4529 = stablehlo.broadcast_in_dim %4528, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4530 = stablehlo.broadcast_in_dim %arg423, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %4531 = stablehlo.add %4529, %4530 : tensor<197x4096xf32>
    %4532 = stablehlo.convert %4531 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %4533 = stablehlo.reshape %4532 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %4534 = stablehlo.multiply %4533, %cst_4 : tensor<1x197x4096xbf16>
    %4535 = stablehlo.multiply %4533, %178 : tensor<1x197x4096xbf16>
    %4536 = stablehlo.convert %4535 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %4537 = stablehlo.clamp %cst_5, %4536, %cst_6 : tensor<1x197x4096xf32>
    %4538 = stablehlo.multiply %4537, %4537 : tensor<1x197x4096xf32>
    %4539 = stablehlo.multiply %cst_7, %4538 : tensor<1x197x4096xf32>
    %4540 = stablehlo.add %4539, %cst_8 : tensor<1x197x4096xf32>
    %4541 = stablehlo.multiply %4540, %4538 : tensor<1x197x4096xf32>
    %4542 = stablehlo.add %4541, %cst_9 : tensor<1x197x4096xf32>
    %4543 = stablehlo.multiply %4542, %4538 : tensor<1x197x4096xf32>
    %4544 = stablehlo.add %4543, %cst_10 : tensor<1x197x4096xf32>
    %4545 = stablehlo.multiply %4544, %4538 : tensor<1x197x4096xf32>
    %4546 = stablehlo.add %4545, %cst_11 : tensor<1x197x4096xf32>
    %4547 = stablehlo.multiply %4546, %4538 : tensor<1x197x4096xf32>
    %4548 = stablehlo.add %4547, %cst_12 : tensor<1x197x4096xf32>
    %4549 = stablehlo.multiply %4548, %4538 : tensor<1x197x4096xf32>
    %4550 = stablehlo.add %4549, %cst_13 : tensor<1x197x4096xf32>
    %4551 = stablehlo.multiply %cst_14, %4538 : tensor<1x197x4096xf32>
    %4552 = stablehlo.add %4551, %cst_15 : tensor<1x197x4096xf32>
    %4553 = stablehlo.multiply %4552, %4538 : tensor<1x197x4096xf32>
    %4554 = stablehlo.add %4553, %cst_16 : tensor<1x197x4096xf32>
    %4555 = stablehlo.multiply %4554, %4538 : tensor<1x197x4096xf32>
    %4556 = stablehlo.add %4555, %cst_17 : tensor<1x197x4096xf32>
    %4557 = stablehlo.multiply %4556, %4538 : tensor<1x197x4096xf32>
    %4558 = stablehlo.add %4557, %cst_18 : tensor<1x197x4096xf32>
    %4559 = stablehlo.multiply %4537, %4550 : tensor<1x197x4096xf32>
    %4560 = stablehlo.divide %4559, %4558 : tensor<1x197x4096xf32>
    %4561 = stablehlo.clamp %cst_19, %4560, %cst_20 : tensor<1x197x4096xf32>
    %4562 = stablehlo.convert %4561 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %4563 = stablehlo.add %4562, %cst_2 : tensor<1x197x4096xbf16>
    %4564 = stablehlo.multiply %4563, %4534 : tensor<1x197x4096xbf16>
    %4565 = stablehlo.reshape %4564 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %4566 = stablehlo.convert %4565 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %4567 = stablehlo.dot_general %4566, %arg424, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %4568 = stablehlo.broadcast_in_dim %4567, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4569 = stablehlo.multiply %4568, %60 : tensor<197x1024xf32>
    %4570 = stablehlo.broadcast_in_dim %4569, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4571 = stablehlo.broadcast_in_dim %arg425, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4572 = stablehlo.add %4570, %4571 : tensor<197x1024xf32>
    %4573 = stablehlo.convert %4572 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4574 = stablehlo.reshape %4573 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4575 = stablehlo.broadcast_in_dim %arg140, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4576 = stablehlo.broadcast_in_dim %4574, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4577 = stablehlo.multiply %4575, %4576 : tensor<1x197x1024xbf16>
    %4578 = stablehlo.add %4577, %4486 : tensor<1x197x1024xbf16>
    %4579 = stablehlo.convert %4578 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4580 = stablehlo.convert %4579 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4581 = stablehlo.reduce(%4580 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4582 = stablehlo.reshape %4581 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4583 = stablehlo.broadcast_in_dim %4582, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4584 = stablehlo.divide %4583, %15 : tensor<1x197x1xf64>
    %4585 = stablehlo.broadcast_in_dim %4580, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4586 = stablehlo.broadcast_in_dim %4584, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4587 = stablehlo.subtract %4585, %4586 : tensor<1x197x1024xf64>
    %4588 = stablehlo.multiply %4587, %4587 : tensor<1x197x1024xf64>
    %4589 = stablehlo.reduce(%4588 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4590 = stablehlo.reshape %4589 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4591 = stablehlo.broadcast_in_dim %4590, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4592 = stablehlo.divide %4591, %15 : tensor<1x197x1xf64>
    %4593 = stablehlo.convert %4592 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4594 = stablehlo.reduce(%4579 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4595 = stablehlo.reshape %4594 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4596 = stablehlo.broadcast_in_dim %4595, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4597 = stablehlo.divide %4596, %31 : tensor<1x197x1xf32>
    %4598 = stablehlo.broadcast_in_dim %4593, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4599 = stablehlo.add %4598, %36 : tensor<1x197x1xf32>
    %4600 = stablehlo.rsqrt %4599 : tensor<1x197x1xf32>
    %4601 = stablehlo.broadcast_in_dim %4579, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4602 = stablehlo.broadcast_in_dim %4597, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4603 = stablehlo.subtract %4601, %4602 : tensor<1x197x1024xf32>
    %4604 = stablehlo.broadcast_in_dim %4603, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4605 = stablehlo.broadcast_in_dim %4600, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4606 = stablehlo.multiply %4604, %4605 : tensor<1x197x1024xf32>
    %4607 = stablehlo.convert %arg141 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4608 = stablehlo.broadcast_in_dim %4606, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4609 = stablehlo.broadcast_in_dim %4607, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4610 = stablehlo.multiply %4608, %4609 : tensor<1x197x1024xf32>
    %4611 = stablehlo.convert %arg142 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4612 = stablehlo.broadcast_in_dim %4610, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4613 = stablehlo.broadcast_in_dim %4611, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4614 = stablehlo.add %4612, %4613 : tensor<1x197x1024xf32>
    %4615 = stablehlo.convert %4614 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4616 = stablehlo.reshape %4615 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4617 = stablehlo.convert %4616 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4618 = stablehlo.dot_general %4617, %arg426, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4619 = stablehlo.broadcast_in_dim %4618, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4620 = stablehlo.multiply %4619, %60 : tensor<197x1024xf32>
    %4621 = stablehlo.broadcast_in_dim %4620, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4622 = stablehlo.broadcast_in_dim %arg427, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4623 = stablehlo.add %4621, %4622 : tensor<197x1024xf32>
    %4624 = stablehlo.convert %4623 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4625 = stablehlo.reshape %4624 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4626 = stablehlo.dot_general %4616, %arg428, contracting_dims = [1] x [0] : (tensor<197x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<197x1024xbf16>
    %4627 = stablehlo.reshape %4626 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4628 = stablehlo.reshape %4627 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4629 = stablehlo.transpose %4628, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4630 = stablehlo.dot_general %4617, %arg429, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4631 = stablehlo.broadcast_in_dim %4630, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4632 = stablehlo.multiply %4631, %60 : tensor<197x1024xf32>
    %4633 = stablehlo.broadcast_in_dim %4632, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4634 = stablehlo.broadcast_in_dim %arg430, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4635 = stablehlo.add %4633, %4634 : tensor<197x1024xf32>
    %4636 = stablehlo.convert %4635 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4637 = stablehlo.reshape %4636 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4638 = stablehlo.reshape %4637 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4639 = stablehlo.transpose %4638, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4640 = stablehlo.reshape %4625 : (tensor<1x197x1024xbf16>) -> tensor<1x197x16x64xbf16>
    %4641 = stablehlo.transpose %4640, dims = [0, 2, 1, 3] : (tensor<1x197x16x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4642 = stablehlo.transpose %4629, dims = [0, 1, 3, 2] : (tensor<1x16x197x64xbf16>) -> tensor<1x16x64x197xbf16>
    %4643 = stablehlo.reshape %4641 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4644 = stablehlo.reshape %4642 : (tensor<1x16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4645 = stablehlo.broadcast_in_dim %4644, dims = [0, 1, 2] : (tensor<16x64x197xbf16>) -> tensor<16x64x197xbf16>
    %4646 = stablehlo.dot_general %4643, %4645, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x64xbf16>, tensor<16x64x197xbf16>) -> tensor<16x197x197xbf16>
    %4647 = stablehlo.reshape %4646 : (tensor<16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4648 = stablehlo.broadcast_in_dim %4647, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xbf16>
    %4649 = stablehlo.divide %4648, %92 : tensor<1x16x197x197xbf16>
    %4650 = stablehlo.add %4649, %arg431 : tensor<1x16x197x197xbf16>
    %4651 = stablehlo.convert %4650 : (tensor<1x16x197x197xbf16>) -> tensor<1x16x197x197xf32>
    %4652 = stablehlo.reduce(%4651 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4653 = stablehlo.reshape %4652 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4654 = stablehlo.broadcast_in_dim %4651, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4655 = stablehlo.broadcast_in_dim %4653, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4656 = stablehlo.subtract %4654, %4655 : tensor<1x16x197x197xf32>
    %4657 = stablehlo.exponential %4656 : tensor<1x16x197x197xf32>
    %4658 = stablehlo.reduce(%4657 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x16x197x197xf32>, tensor<f32>) -> tensor<1x16x197xf32>
    %4659 = stablehlo.reshape %4658 : (tensor<1x16x197xf32>) -> tensor<1x16x197x1xf32>
    %4660 = stablehlo.broadcast_in_dim %4657, dims = [0, 1, 2, 3] : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xf32>
    %4661 = stablehlo.broadcast_in_dim %4659, dims = [0, 1, 2, 3] : (tensor<1x16x197x1xf32>) -> tensor<1x16x197x197xf32>
    %4662 = stablehlo.divide %4660, %4661 : tensor<1x16x197x197xf32>
    %4663 = stablehlo.convert %4662 : (tensor<1x16x197x197xf32>) -> tensor<1x16x197x197xbf16>
    %4664 = stablehlo.reshape %4663 : (tensor<1x16x197x197xbf16>) -> tensor<16x197x197xbf16>
    %4665 = stablehlo.reshape %4639 : (tensor<1x16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4666 = stablehlo.broadcast_in_dim %4665, dims = [0, 1, 2] : (tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4667 = stablehlo.dot_general %4664, %4666, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x197x197xbf16>, tensor<16x197x64xbf16>) -> tensor<16x197x64xbf16>
    %4668 = stablehlo.reshape %4667 : (tensor<16x197x64xbf16>) -> tensor<1x16x197x64xbf16>
    %4669 = stablehlo.transpose %4668, dims = [0, 2, 1, 3] : (tensor<1x16x197x64xbf16>) -> tensor<1x197x16x64xbf16>
    %4670 = stablehlo.reshape %4669 : (tensor<1x197x16x64xbf16>) -> tensor<1x197x1024xbf16>
    %4671 = stablehlo.reshape %4670 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4672 = stablehlo.convert %4671 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4673 = stablehlo.dot_general %4672, %arg432, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x1024xf32>) -> tensor<197x1024xf32>
    %4674 = stablehlo.broadcast_in_dim %4673, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4675 = stablehlo.multiply %4674, %60 : tensor<197x1024xf32>
    %4676 = stablehlo.broadcast_in_dim %4675, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4677 = stablehlo.broadcast_in_dim %arg433, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4678 = stablehlo.add %4676, %4677 : tensor<197x1024xf32>
    %4679 = stablehlo.convert %4678 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4680 = stablehlo.reshape %4679 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4681 = stablehlo.broadcast_in_dim %arg143, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4682 = stablehlo.broadcast_in_dim %4680, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4683 = stablehlo.multiply %4681, %4682 : tensor<1x197x1024xbf16>
    %4684 = stablehlo.add %4683, %4578 : tensor<1x197x1024xbf16>
    %4685 = stablehlo.convert %4684 : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xf32>
    %4686 = stablehlo.convert %4685 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf64>
    %4687 = stablehlo.reduce(%4686 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4688 = stablehlo.reshape %4687 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4689 = stablehlo.broadcast_in_dim %4688, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4690 = stablehlo.divide %4689, %15 : tensor<1x197x1xf64>
    %4691 = stablehlo.broadcast_in_dim %4686, dims = [0, 1, 2] : (tensor<1x197x1024xf64>) -> tensor<1x197x1024xf64>
    %4692 = stablehlo.broadcast_in_dim %4690, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1024xf64>
    %4693 = stablehlo.subtract %4691, %4692 : tensor<1x197x1024xf64>
    %4694 = stablehlo.multiply %4693, %4693 : tensor<1x197x1024xf64>
    %4695 = stablehlo.reduce(%4694 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf64>, tensor<f64>) -> tensor<1x197xf64>
    %4696 = stablehlo.reshape %4695 : (tensor<1x197xf64>) -> tensor<1x197x1xf64>
    %4697 = stablehlo.broadcast_in_dim %4696, dims = [0, 1, 2] : (tensor<1x197x1xf64>) -> tensor<1x197x1xf64>
    %4698 = stablehlo.divide %4697, %15 : tensor<1x197x1xf64>
    %4699 = stablehlo.convert %4698 : (tensor<1x197x1xf64>) -> tensor<1x197x1xf32>
    %4700 = stablehlo.reduce(%4685 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x197x1024xf32>, tensor<f32>) -> tensor<1x197xf32>
    %4701 = stablehlo.reshape %4700 : (tensor<1x197xf32>) -> tensor<1x197x1xf32>
    %4702 = stablehlo.broadcast_in_dim %4701, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4703 = stablehlo.divide %4702, %31 : tensor<1x197x1xf32>
    %4704 = stablehlo.broadcast_in_dim %4699, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %4705 = stablehlo.add %4704, %36 : tensor<1x197x1xf32>
    %4706 = stablehlo.rsqrt %4705 : tensor<1x197x1xf32>
    %4707 = stablehlo.broadcast_in_dim %4685, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4708 = stablehlo.broadcast_in_dim %4703, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4709 = stablehlo.subtract %4707, %4708 : tensor<1x197x1024xf32>
    %4710 = stablehlo.broadcast_in_dim %4709, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4711 = stablehlo.broadcast_in_dim %4706, dims = [0, 1, 2] : (tensor<1x197x1xf32>) -> tensor<1x197x1024xf32>
    %4712 = stablehlo.multiply %4710, %4711 : tensor<1x197x1024xf32>
    %4713 = stablehlo.convert %arg144 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4714 = stablehlo.broadcast_in_dim %4712, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4715 = stablehlo.broadcast_in_dim %4713, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4716 = stablehlo.multiply %4714, %4715 : tensor<1x197x1024xf32>
    %4717 = stablehlo.convert %arg145 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4718 = stablehlo.broadcast_in_dim %4716, dims = [0, 1, 2] : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xf32>
    %4719 = stablehlo.broadcast_in_dim %4717, dims = [2] : (tensor<1024xf32>) -> tensor<1x197x1024xf32>
    %4720 = stablehlo.add %4718, %4719 : tensor<1x197x1024xf32>
    %4721 = stablehlo.convert %4720 : (tensor<1x197x1024xf32>) -> tensor<1x197x1024xbf16>
    %4722 = stablehlo.reshape %4721 : (tensor<1x197x1024xbf16>) -> tensor<197x1024xbf16>
    %4723 = stablehlo.convert %4722 : (tensor<197x1024xbf16>) -> tensor<197x1024xf32>
    %4724 = stablehlo.dot_general %4723, %arg434, contracting_dims = [1] x [0] : (tensor<197x1024xf32>, tensor<1024x4096xf32>) -> tensor<197x4096xf32>
    %4725 = stablehlo.broadcast_in_dim %4724, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4726 = stablehlo.multiply %4725, %170 : tensor<197x4096xf32>
    %4727 = stablehlo.broadcast_in_dim %4726, dims = [0, 1] : (tensor<197x4096xf32>) -> tensor<197x4096xf32>
    %4728 = stablehlo.broadcast_in_dim %arg435, dims = [1] : (tensor<4096xf32>) -> tensor<197x4096xf32>
    %4729 = stablehlo.add %4727, %4728 : tensor<197x4096xf32>
    %4730 = stablehlo.convert %4729 : (tensor<197x4096xf32>) -> tensor<197x4096xbf16>
    %4731 = stablehlo.reshape %4730 : (tensor<197x4096xbf16>) -> tensor<1x197x4096xbf16>
    %4732 = stablehlo.multiply %4731, %cst_4 : tensor<1x197x4096xbf16>
    %4733 = stablehlo.multiply %4731, %178 : tensor<1x197x4096xbf16>
    %4734 = stablehlo.convert %4733 : (tensor<1x197x4096xbf16>) -> tensor<1x197x4096xf32>
    %4735 = stablehlo.clamp %cst_5, %4734, %cst_6 : tensor<1x197x4096xf32>
    %4736 = stablehlo.multiply %4735, %4735 : tensor<1x197x4096xf32>
    %4737 = stablehlo.multiply %cst_7, %4736 : tensor<1x197x4096xf32>
    %4738 = stablehlo.add %4737, %cst_8 : tensor<1x197x4096xf32>
    %4739 = stablehlo.multiply %4738, %4736 : tensor<1x197x4096xf32>
    %4740 = stablehlo.add %4739, %cst_9 : tensor<1x197x4096xf32>
    %4741 = stablehlo.multiply %4740, %4736 : tensor<1x197x4096xf32>
    %4742 = stablehlo.add %4741, %cst_10 : tensor<1x197x4096xf32>
    %4743 = stablehlo.multiply %4742, %4736 : tensor<1x197x4096xf32>
    %4744 = stablehlo.add %4743, %cst_11 : tensor<1x197x4096xf32>
    %4745 = stablehlo.multiply %4744, %4736 : tensor<1x197x4096xf32>
    %4746 = stablehlo.add %4745, %cst_12 : tensor<1x197x4096xf32>
    %4747 = stablehlo.multiply %4746, %4736 : tensor<1x197x4096xf32>
    %4748 = stablehlo.add %4747, %cst_13 : tensor<1x197x4096xf32>
    %4749 = stablehlo.multiply %cst_14, %4736 : tensor<1x197x4096xf32>
    %4750 = stablehlo.add %4749, %cst_15 : tensor<1x197x4096xf32>
    %4751 = stablehlo.multiply %4750, %4736 : tensor<1x197x4096xf32>
    %4752 = stablehlo.add %4751, %cst_16 : tensor<1x197x4096xf32>
    %4753 = stablehlo.multiply %4752, %4736 : tensor<1x197x4096xf32>
    %4754 = stablehlo.add %4753, %cst_17 : tensor<1x197x4096xf32>
    %4755 = stablehlo.multiply %4754, %4736 : tensor<1x197x4096xf32>
    %4756 = stablehlo.add %4755, %cst_18 : tensor<1x197x4096xf32>
    %4757 = stablehlo.multiply %4735, %4748 : tensor<1x197x4096xf32>
    %4758 = stablehlo.divide %4757, %4756 : tensor<1x197x4096xf32>
    %4759 = stablehlo.clamp %cst_19, %4758, %cst_20 : tensor<1x197x4096xf32>
    %4760 = stablehlo.convert %4759 : (tensor<1x197x4096xf32>) -> tensor<1x197x4096xbf16>
    %4761 = stablehlo.add %4760, %cst_2 : tensor<1x197x4096xbf16>
    %4762 = stablehlo.multiply %4761, %4732 : tensor<1x197x4096xbf16>
    %4763 = stablehlo.reshape %4762 : (tensor<1x197x4096xbf16>) -> tensor<197x4096xbf16>
    %4764 = stablehlo.convert %4763 : (tensor<197x4096xbf16>) -> tensor<197x4096xf32>
    %4765 = stablehlo.dot_general %4764, %arg436, contracting_dims = [1] x [0] : (tensor<197x4096xf32>, tensor<4096x1024xf32>) -> tensor<197x1024xf32>
    %4766 = stablehlo.broadcast_in_dim %4765, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4767 = stablehlo.multiply %4766, %60 : tensor<197x1024xf32>
    %4768 = stablehlo.broadcast_in_dim %4767, dims = [0, 1] : (tensor<197x1024xf32>) -> tensor<197x1024xf32>
    %4769 = stablehlo.broadcast_in_dim %arg437, dims = [1] : (tensor<1024xf32>) -> tensor<197x1024xf32>
    %4770 = stablehlo.add %4768, %4769 : tensor<197x1024xf32>
    %4771 = stablehlo.convert %4770 : (tensor<197x1024xf32>) -> tensor<197x1024xbf16>
    %4772 = stablehlo.reshape %4771 : (tensor<197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4773 = stablehlo.broadcast_in_dim %arg146, dims = [2] : (tensor<1024xbf16>) -> tensor<1x197x1024xbf16>
    %4774 = stablehlo.broadcast_in_dim %4772, dims = [0, 1, 2] : (tensor<1x197x1024xbf16>) -> tensor<1x197x1024xbf16>
    %4775 = stablehlo.multiply %4773, %4774 : tensor<1x197x1024xbf16>
    %4776 = stablehlo.add %4775, %4684 : tensor<1x197x1024xbf16>
    %4777 = stablehlo.slice %4776 [0:1, 1:197, 0:1024] : (tensor<1x197x1024xbf16>) -> tensor<1x196x1024xbf16>
    %4778 = stablehlo.reduce(%4777 init: %cst_21) applies stablehlo.add across dimensions = [1] : (tensor<1x196x1024xbf16>, tensor<bf16>) -> tensor<1x1024xbf16>
    %4779 = stablehlo.convert %cst_26 : (tensor<1xi64>) -> tensor<1xbf16>
    %4780 = stablehlo.reshape %4779 : (tensor<1xbf16>) -> tensor<bf16>
    %4781 = stablehlo.broadcast_in_dim %4778, dims = [0, 1] : (tensor<1x1024xbf16>) -> tensor<1x1024xbf16>
    %4782 = stablehlo.broadcast_in_dim %4780, dims = [] : (tensor<bf16>) -> tensor<1x1024xbf16>
    %4783 = stablehlo.divide %4781, %4782 : tensor<1x1024xbf16>
    %4784 = stablehlo.convert %4783 : (tensor<1x1024xbf16>) -> tensor<1x1024xf32>
    %4785 = stablehlo.convert %4784 : (tensor<1x1024xf32>) -> tensor<1x1024xf64>
    %4786 = stablehlo.reduce(%4785 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x1024xf64>, tensor<f64>) -> tensor<1xf64>
    %4787 = stablehlo.reshape %4786 : (tensor<1xf64>) -> tensor<1x1xf64>
    %4788 = stablehlo.broadcast_in_dim %4787, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %4789 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %4790 = stablehlo.divide %4788, %4789 : tensor<1x1xf64>
    %4791 = stablehlo.broadcast_in_dim %4785, dims = [0, 1] : (tensor<1x1024xf64>) -> tensor<1x1024xf64>
    %4792 = stablehlo.broadcast_in_dim %4790, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x1024xf64>
    %4793 = stablehlo.subtract %4791, %4792 : tensor<1x1024xf64>
    %4794 = stablehlo.multiply %4793, %4793 : tensor<1x1024xf64>
    %4795 = stablehlo.reduce(%4794 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x1024xf64>, tensor<f64>) -> tensor<1xf64>
    %4796 = stablehlo.reshape %4795 : (tensor<1xf64>) -> tensor<1x1xf64>
    %4797 = stablehlo.broadcast_in_dim %4796, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %4798 = stablehlo.divide %4797, %4789 : tensor<1x1xf64>
    %4799 = stablehlo.convert %4798 : (tensor<1x1xf64>) -> tensor<1x1xf32>
    %4800 = stablehlo.reduce(%4784 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<1x1024xf32>, tensor<f32>) -> tensor<1xf32>
    %4801 = stablehlo.reshape %4800 : (tensor<1xf32>) -> tensor<1x1xf32>
    %4802 = stablehlo.broadcast_in_dim %4801, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %4803 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %4804 = stablehlo.divide %4802, %4803 : tensor<1x1xf32>
    %4805 = stablehlo.broadcast_in_dim %4799, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %4806 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %4807 = stablehlo.add %4805, %4806 : tensor<1x1xf32>
    %4808 = stablehlo.rsqrt %4807 : tensor<1x1xf32>
    %4809 = stablehlo.broadcast_in_dim %4784, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %4810 = stablehlo.broadcast_in_dim %4804, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1024xf32>
    %4811 = stablehlo.subtract %4809, %4810 : tensor<1x1024xf32>
    %4812 = stablehlo.broadcast_in_dim %4811, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %4813 = stablehlo.broadcast_in_dim %4808, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1024xf32>
    %4814 = stablehlo.multiply %4812, %4813 : tensor<1x1024xf32>
    %4815 = stablehlo.convert %arg147 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4816 = stablehlo.broadcast_in_dim %4814, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %4817 = stablehlo.broadcast_in_dim %4815, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %4818 = stablehlo.multiply %4816, %4817 : tensor<1x1024xf32>
    %4819 = stablehlo.convert %arg148 : (tensor<1024xbf16>) -> tensor<1024xf32>
    %4820 = stablehlo.broadcast_in_dim %4818, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %4821 = stablehlo.broadcast_in_dim %4819, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %4822 = stablehlo.add %4820, %4821 : tensor<1x1024xf32>
    %4823 = stablehlo.convert %4822 : (tensor<1x1024xf32>) -> tensor<1x1024xbf16>
    %4824 = stablehlo.convert %4823 : (tensor<1x1024xbf16>) -> tensor<1x1024xf32>
    %4825 = stablehlo.dot_general %4824, %arg438, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x1000xf32>) -> tensor<1x1000xf32>
    %4826 = stablehlo.broadcast_in_dim %4825, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %4827 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<1x1000xf32>
    %4828 = stablehlo.multiply %4826, %4827 : tensor<1x1000xf32>
    %4829 = stablehlo.broadcast_in_dim %4828, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %4830 = stablehlo.broadcast_in_dim %arg439, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %4831 = stablehlo.add %4829, %4830 : tensor<1x1000xf32>
    %4832 = stablehlo.convert %4831 : (tensor<1x1000xf32>) -> tensor<1x1000xbf16>
    return %4832 : tensor<1x1000xbf16>
  }
}
