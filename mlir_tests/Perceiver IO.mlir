module {
  func.func @main(%arg0: tensor<1x2048xi64>, %arg1: tensor<1x2048xi64>, %arg2: tensor<262x768xbf16>, %arg3: tensor<768xbf16>, %arg4: tensor<768xbf16>, %arg5: tensor<1280xbf16>, %arg6: tensor<1280xbf16>, %arg7: tensor<1280xbf16>, %arg8: tensor<1280xbf16>, %arg9: tensor<1280xbf16>, %arg10: tensor<1280xbf16>, %arg11: tensor<1280xbf16>, %arg12: tensor<1280xbf16>, %arg13: tensor<1280xbf16>, %arg14: tensor<1280xbf16>, %arg15: tensor<1280xbf16>, %arg16: tensor<1280xbf16>, %arg17: tensor<1280xbf16>, %arg18: tensor<1280xbf16>, %arg19: tensor<1280xbf16>, %arg20: tensor<1280xbf16>, %arg21: tensor<1280xbf16>, %arg22: tensor<1280xbf16>, %arg23: tensor<1280xbf16>, %arg24: tensor<1280xbf16>, %arg25: tensor<1280xbf16>, %arg26: tensor<1280xbf16>, %arg27: tensor<1280xbf16>, %arg28: tensor<1280xbf16>, %arg29: tensor<1280xbf16>, %arg30: tensor<1280xbf16>, %arg31: tensor<1280xbf16>, %arg32: tensor<1280xbf16>, %arg33: tensor<1280xbf16>, %arg34: tensor<1280xbf16>, %arg35: tensor<1280xbf16>, %arg36: tensor<1280xbf16>, %arg37: tensor<1280xbf16>, %arg38: tensor<1280xbf16>, %arg39: tensor<1280xbf16>, %arg40: tensor<1280xbf16>, %arg41: tensor<1280xbf16>, %arg42: tensor<1280xbf16>, %arg43: tensor<1280xbf16>, %arg44: tensor<1280xbf16>, %arg45: tensor<1280xbf16>, %arg46: tensor<1280xbf16>, %arg47: tensor<1280xbf16>, %arg48: tensor<1280xbf16>, %arg49: tensor<1280xbf16>, %arg50: tensor<1280xbf16>, %arg51: tensor<1280xbf16>, %arg52: tensor<1280xbf16>, %arg53: tensor<1280xbf16>, %arg54: tensor<1280xbf16>, %arg55: tensor<1280xbf16>, %arg56: tensor<1280xbf16>, %arg57: tensor<1280xbf16>, %arg58: tensor<1280xbf16>, %arg59: tensor<1280xbf16>, %arg60: tensor<1280xbf16>, %arg61: tensor<1280xbf16>, %arg62: tensor<1280xbf16>, %arg63: tensor<1280xbf16>, %arg64: tensor<1280xbf16>, %arg65: tensor<1280xbf16>, %arg66: tensor<1280xbf16>, %arg67: tensor<1280xbf16>, %arg68: tensor<1280xbf16>, %arg69: tensor<1280xbf16>, %arg70: tensor<1280xbf16>, %arg71: tensor<1280xbf16>, %arg72: tensor<1280xbf16>, %arg73: tensor<1280xbf16>, %arg74: tensor<1280xbf16>, %arg75: tensor<1280xbf16>, %arg76: tensor<1280xbf16>, %arg77: tensor<1280xbf16>, %arg78: tensor<1280xbf16>, %arg79: tensor<1280xbf16>, %arg80: tensor<1280xbf16>, %arg81: tensor<1280xbf16>, %arg82: tensor<1280xbf16>, %arg83: tensor<1280xbf16>, %arg84: tensor<1280xbf16>, %arg85: tensor<1280xbf16>, %arg86: tensor<1280xbf16>, %arg87: tensor<1280xbf16>, %arg88: tensor<1280xbf16>, %arg89: tensor<1280xbf16>, %arg90: tensor<1280xbf16>, %arg91: tensor<1280xbf16>, %arg92: tensor<1280xbf16>, %arg93: tensor<1280xbf16>, %arg94: tensor<1280xbf16>, %arg95: tensor<1280xbf16>, %arg96: tensor<1280xbf16>, %arg97: tensor<1280xbf16>, %arg98: tensor<1280xbf16>, %arg99: tensor<1280xbf16>, %arg100: tensor<1280xbf16>, %arg101: tensor<1280xbf16>, %arg102: tensor<1280xbf16>, %arg103: tensor<1280xbf16>, %arg104: tensor<1280xbf16>, %arg105: tensor<1280xbf16>, %arg106: tensor<1280xbf16>, %arg107: tensor<1280xbf16>, %arg108: tensor<1280xbf16>, %arg109: tensor<1280xbf16>, %arg110: tensor<1280xbf16>, %arg111: tensor<1280xbf16>, %arg112: tensor<1280xbf16>, %arg113: tensor<768xbf16>, %arg114: tensor<768xbf16>, %arg115: tensor<262xbf16>, %arg116: tensor<2048x768xbf16>, %arg117: tensor<768x256xf32>, %arg118: tensor<256xf32>, %arg119: tensor<768x1280xf32>, %arg120: tensor<1280xf32>, %arg121: tensor<8x256x32xbf16>, %arg122: tensor<1280x1280xf32>, %arg123: tensor<1280xf32>, %arg124: tensor<1x256x1280xbf16>, %arg125: tensor<1280x1280xf32>, %arg126: tensor<1280xf32>, %arg127: tensor<1280x1280xf32>, %arg128: tensor<1280xf32>, %arg129: tensor<1280x256xf32>, %arg130: tensor<256xf32>, %arg131: tensor<1280x256xf32>, %arg132: tensor<256xf32>, %arg133: tensor<1280x1280xf32>, %arg134: tensor<1280xf32>, %arg135: tensor<1280x1280xf32>, %arg136: tensor<1280xf32>, %arg137: tensor<1280x1280xf32>, %arg138: tensor<1280xf32>, %arg139: tensor<1280x1280xf32>, %arg140: tensor<1280xf32>, %arg141: tensor<1280x256xf32>, %arg142: tensor<256xf32>, %arg143: tensor<1280x256xf32>, %arg144: tensor<256xf32>, %arg145: tensor<1280x1280xf32>, %arg146: tensor<1280xf32>, %arg147: tensor<1280x1280xf32>, %arg148: tensor<1280xf32>, %arg149: tensor<1280x1280xf32>, %arg150: tensor<1280xf32>, %arg151: tensor<1280x1280xf32>, %arg152: tensor<1280xf32>, %arg153: tensor<1280x256xf32>, %arg154: tensor<256xf32>, %arg155: tensor<1280x256xf32>, %arg156: tensor<256xf32>, %arg157: tensor<1280x1280xf32>, %arg158: tensor<1280xf32>, %arg159: tensor<1280x1280xf32>, %arg160: tensor<1280xf32>, %arg161: tensor<1280x1280xf32>, %arg162: tensor<1280xf32>, %arg163: tensor<1280x1280xf32>, %arg164: tensor<1280xf32>, %arg165: tensor<1280x256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<1280x256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<1280x1280xf32>, %arg170: tensor<1280xf32>, %arg171: tensor<1280x1280xf32>, %arg172: tensor<1280xf32>, %arg173: tensor<1280x1280xf32>, %arg174: tensor<1280xf32>, %arg175: tensor<1280x1280xf32>, %arg176: tensor<1280xf32>, %arg177: tensor<1280x256xf32>, %arg178: tensor<256xf32>, %arg179: tensor<1280x256xf32>, %arg180: tensor<256xf32>, %arg181: tensor<1280x1280xf32>, %arg182: tensor<1280xf32>, %arg183: tensor<1280x1280xf32>, %arg184: tensor<1280xf32>, %arg185: tensor<1280x1280xf32>, %arg186: tensor<1280xf32>, %arg187: tensor<1280x1280xf32>, %arg188: tensor<1280xf32>, %arg189: tensor<1280x256xf32>, %arg190: tensor<256xf32>, %arg191: tensor<1280x256xf32>, %arg192: tensor<256xf32>, %arg193: tensor<1280x1280xf32>, %arg194: tensor<1280xf32>, %arg195: tensor<1280x1280xf32>, %arg196: tensor<1280xf32>, %arg197: tensor<1280x1280xf32>, %arg198: tensor<1280xf32>, %arg199: tensor<1280x1280xf32>, %arg200: tensor<1280xf32>, %arg201: tensor<1280x256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<1280x256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<1280x1280xf32>, %arg206: tensor<1280xf32>, %arg207: tensor<1280x1280xf32>, %arg208: tensor<1280xf32>, %arg209: tensor<1280x1280xf32>, %arg210: tensor<1280xf32>, %arg211: tensor<1280x1280xf32>, %arg212: tensor<1280xf32>, %arg213: tensor<1280x256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1280x256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<1280x1280xf32>, %arg218: tensor<1280xf32>, %arg219: tensor<1280x1280xf32>, %arg220: tensor<1280xf32>, %arg221: tensor<1280x1280xf32>, %arg222: tensor<1280xf32>, %arg223: tensor<1280x1280xf32>, %arg224: tensor<1280xf32>, %arg225: tensor<1280x256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1280x256xf32>, %arg228: tensor<256xf32>, %arg229: tensor<1280x1280xf32>, %arg230: tensor<1280xf32>, %arg231: tensor<1280x1280xf32>, %arg232: tensor<1280xf32>, %arg233: tensor<1280x1280xf32>, %arg234: tensor<1280xf32>, %arg235: tensor<1280x1280xf32>, %arg236: tensor<1280xf32>, %arg237: tensor<1280x256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1280x256xf32>, %arg240: tensor<256xf32>, %arg241: tensor<1280x1280xf32>, %arg242: tensor<1280xf32>, %arg243: tensor<1280x1280xf32>, %arg244: tensor<1280xf32>, %arg245: tensor<1280x1280xf32>, %arg246: tensor<1280xf32>, %arg247: tensor<1280x1280xf32>, %arg248: tensor<1280xf32>, %arg249: tensor<1280x256xf32>, %arg250: tensor<256xf32>, %arg251: tensor<1280x256xf32>, %arg252: tensor<256xf32>, %arg253: tensor<1280x1280xf32>, %arg254: tensor<1280xf32>, %arg255: tensor<1280x1280xf32>, %arg256: tensor<1280xf32>, %arg257: tensor<1280x1280xf32>, %arg258: tensor<1280xf32>, %arg259: tensor<1280x1280xf32>, %arg260: tensor<1280xf32>, %arg261: tensor<1280x256xf32>, %arg262: tensor<256xf32>, %arg263: tensor<1280x256xf32>, %arg264: tensor<256xf32>, %arg265: tensor<1280x1280xf32>, %arg266: tensor<1280xf32>, %arg267: tensor<1280x1280xf32>, %arg268: tensor<1280xf32>, %arg269: tensor<1280x1280xf32>, %arg270: tensor<1280xf32>, %arg271: tensor<1280x1280xf32>, %arg272: tensor<1280xf32>, %arg273: tensor<1280x256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<1280x256xf32>, %arg276: tensor<256xf32>, %arg277: tensor<1280x1280xf32>, %arg278: tensor<1280xf32>, %arg279: tensor<1280x1280xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280x1280xf32>, %arg282: tensor<1280xf32>, %arg283: tensor<1280x1280xf32>, %arg284: tensor<1280xf32>, %arg285: tensor<1280x256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<1280x256xf32>, %arg288: tensor<256xf32>, %arg289: tensor<1280x1280xf32>, %arg290: tensor<1280xf32>, %arg291: tensor<1280x1280xf32>, %arg292: tensor<1280xf32>, %arg293: tensor<1280x1280xf32>, %arg294: tensor<1280xf32>, %arg295: tensor<1280x1280xf32>, %arg296: tensor<1280xf32>, %arg297: tensor<1280x256xf32>, %arg298: tensor<256xf32>, %arg299: tensor<1280x256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<1280x1280xf32>, %arg302: tensor<1280xf32>, %arg303: tensor<1280x1280xf32>, %arg304: tensor<1280xf32>, %arg305: tensor<1280x1280xf32>, %arg306: tensor<1280xf32>, %arg307: tensor<1280x1280xf32>, %arg308: tensor<1280xf32>, %arg309: tensor<1280x256xf32>, %arg310: tensor<256xf32>, %arg311: tensor<1280x256xf32>, %arg312: tensor<256xf32>, %arg313: tensor<1280x1280xf32>, %arg314: tensor<1280xf32>, %arg315: tensor<1280x1280xf32>, %arg316: tensor<1280xf32>, %arg317: tensor<1280x1280xf32>, %arg318: tensor<1280xf32>, %arg319: tensor<1280x1280xf32>, %arg320: tensor<1280xf32>, %arg321: tensor<1280x256xf32>, %arg322: tensor<256xf32>, %arg323: tensor<1280x256xf32>, %arg324: tensor<256xf32>, %arg325: tensor<1280x1280xf32>, %arg326: tensor<1280xf32>, %arg327: tensor<1280x1280xf32>, %arg328: tensor<1280xf32>, %arg329: tensor<1280x1280xf32>, %arg330: tensor<1280xf32>, %arg331: tensor<1280x1280xf32>, %arg332: tensor<1280xf32>, %arg333: tensor<1280x256xf32>, %arg334: tensor<256xf32>, %arg335: tensor<1280x256xf32>, %arg336: tensor<256xf32>, %arg337: tensor<1280x1280xf32>, %arg338: tensor<1280xf32>, %arg339: tensor<1280x1280xf32>, %arg340: tensor<1280xf32>, %arg341: tensor<1280x1280xf32>, %arg342: tensor<1280xf32>, %arg343: tensor<1280x1280xf32>, %arg344: tensor<1280xf32>, %arg345: tensor<1280x256xf32>, %arg346: tensor<256xf32>, %arg347: tensor<1280x256xf32>, %arg348: tensor<256xf32>, %arg349: tensor<1280x1280xf32>, %arg350: tensor<1280xf32>, %arg351: tensor<1280x1280xf32>, %arg352: tensor<1280xf32>, %arg353: tensor<1280x1280xf32>, %arg354: tensor<1280xf32>, %arg355: tensor<1280x1280xf32>, %arg356: tensor<1280xf32>, %arg357: tensor<1280x256xf32>, %arg358: tensor<256xf32>, %arg359: tensor<1280x256xf32>, %arg360: tensor<256xf32>, %arg361: tensor<1280x1280xf32>, %arg362: tensor<1280xf32>, %arg363: tensor<1280x1280xf32>, %arg364: tensor<1280xf32>, %arg365: tensor<1280x1280xf32>, %arg366: tensor<1280xf32>, %arg367: tensor<1280x1280xf32>, %arg368: tensor<1280xf32>, %arg369: tensor<1280x256xf32>, %arg370: tensor<256xf32>, %arg371: tensor<1280x256xf32>, %arg372: tensor<256xf32>, %arg373: tensor<1280x1280xf32>, %arg374: tensor<1280xf32>, %arg375: tensor<1280x1280xf32>, %arg376: tensor<1280xf32>, %arg377: tensor<1280x1280xf32>, %arg378: tensor<1280xf32>, %arg379: tensor<1280x1280xf32>, %arg380: tensor<1280xf32>, %arg381: tensor<1280x256xf32>, %arg382: tensor<256xf32>, %arg383: tensor<1280x256xf32>, %arg384: tensor<256xf32>, %arg385: tensor<1280x1280xf32>, %arg386: tensor<1280xf32>, %arg387: tensor<1280x1280xf32>, %arg388: tensor<1280xf32>, %arg389: tensor<1280x1280xf32>, %arg390: tensor<1280xf32>, %arg391: tensor<1280x1280xf32>, %arg392: tensor<1280xf32>, %arg393: tensor<1280x256xf32>, %arg394: tensor<256xf32>, %arg395: tensor<1280x256xf32>, %arg396: tensor<256xf32>, %arg397: tensor<1280x1280xf32>, %arg398: tensor<1280xf32>, %arg399: tensor<1280x1280xf32>, %arg400: tensor<1280xf32>, %arg401: tensor<1280x1280xf32>, %arg402: tensor<1280xf32>, %arg403: tensor<1280x1280xf32>, %arg404: tensor<1280xf32>, %arg405: tensor<1280x256xf32>, %arg406: tensor<256xf32>, %arg407: tensor<1280x256xf32>, %arg408: tensor<256xf32>, %arg409: tensor<1280x1280xf32>, %arg410: tensor<1280xf32>, %arg411: tensor<1280x1280xf32>, %arg412: tensor<1280xf32>, %arg413: tensor<1280x1280xf32>, %arg414: tensor<1280xf32>, %arg415: tensor<1280x1280xf32>, %arg416: tensor<1280xf32>, %arg417: tensor<1280x256xf32>, %arg418: tensor<256xf32>, %arg419: tensor<1280x256xf32>, %arg420: tensor<256xf32>, %arg421: tensor<1280x1280xf32>, %arg422: tensor<1280xf32>, %arg423: tensor<1280x1280xf32>, %arg424: tensor<1280xf32>, %arg425: tensor<1280x1280xf32>, %arg426: tensor<1280xf32>, %arg427: tensor<1280x1280xf32>, %arg428: tensor<1280xf32>, %arg429: tensor<1280x256xf32>, %arg430: tensor<256xf32>, %arg431: tensor<1280x256xf32>, %arg432: tensor<256xf32>, %arg433: tensor<1280x1280xf32>, %arg434: tensor<1280xf32>, %arg435: tensor<1280x1280xf32>, %arg436: tensor<1280xf32>, %arg437: tensor<1280x1280xf32>, %arg438: tensor<1280xf32>, %arg439: tensor<1280x1280xf32>, %arg440: tensor<1280xf32>, %arg441: tensor<1280x256xf32>, %arg442: tensor<256xf32>, %arg443: tensor<1280x768xf32>, %arg444: tensor<768xf32>, %arg445: tensor<8x2048x32xbf16>, %arg446: tensor<768x768xf32>, %arg447: tensor<768xf32>, %arg448: tensor<768x768xf32>, %arg449: tensor<768xf32>, %arg450: tensor<768x768xf32>, %arg451: tensor<768xf32>, %arg452: tensor<768x262xbf16>) -> tensor<1x2048x262xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<1x256x1280xbf16>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<1x256x1280xbf16>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<1x256x1280xbf16>
    %cst_5 = stablehlo.constant dense<-4.000000e+00> : tensor<1x256x1280xf32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<1x256x1280xf32>
    %cst_7 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x256x1280xf32>
    %cst_8 = stablehlo.constant dense<2.77068146E-8> : tensor<1x256x1280xf32>
    %cst_9 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x256x1280xf32>
    %cst_10 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x256x1280xf32>
    %cst_11 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x256x1280xf32>
    %cst_12 = stablehlo.constant dense<-2.954600e-03> : tensor<1x256x1280xf32>
    %cst_13 = stablehlo.constant dense<-0.0160960332> : tensor<1x256x1280xf32>
    %cst_14 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x256x1280xf32>
    %cst_15 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x256x1280xf32>
    %cst_16 = stablehlo.constant dense<-0.00168282702> : tensor<1x256x1280xf32>
    %cst_17 = stablehlo.constant dense<-0.00737332925> : tensor<1x256x1280xf32>
    %cst_18 = stablehlo.constant dense<-0.0142647391> : tensor<1x256x1280xf32>
    %cst_19 = stablehlo.constant dense<-1.000000e+00> : tensor<1x256x1280xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<1x256x1280xf32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<1x2048x768xbf16>
    %cst_22 = stablehlo.constant dense<2.000000e+00> : tensor<1x2048x768xbf16>
    %cst_23 = stablehlo.constant dense<5.000000e-01> : tensor<1x2048x768xbf16>
    %cst_24 = stablehlo.constant dense<-4.000000e+00> : tensor<1x2048x768xf32>
    %cst_25 = stablehlo.constant dense<4.000000e+00> : tensor<1x2048x768xf32>
    %cst_26 = stablehlo.constant dense<-2.72614237E-10> : tensor<1x2048x768xf32>
    %cst_27 = stablehlo.constant dense<2.77068146E-8> : tensor<1x2048x768xf32>
    %cst_28 = stablehlo.constant dense<-2.10102394E-6> : tensor<1x2048x768xf32>
    %cst_29 = stablehlo.constant dense<-5.69250624E-5> : tensor<1x2048x768xf32>
    %cst_30 = stablehlo.constant dense<-7.34990637E-4> : tensor<1x2048x768xf32>
    %cst_31 = stablehlo.constant dense<-2.954600e-03> : tensor<1x2048x768xf32>
    %cst_32 = stablehlo.constant dense<-0.0160960332> : tensor<1x2048x768xf32>
    %cst_33 = stablehlo.constant dense<-1.45660715E-5> : tensor<1x2048x768xf32>
    %cst_34 = stablehlo.constant dense<-2.13374049E-4> : tensor<1x2048x768xf32>
    %cst_35 = stablehlo.constant dense<-0.00168282702> : tensor<1x2048x768xf32>
    %cst_36 = stablehlo.constant dense<-0.00737332925> : tensor<1x2048x768xf32>
    %cst_37 = stablehlo.constant dense<-0.0142647391> : tensor<1x2048x768xf32>
    %cst_38 = stablehlo.constant dense<-1.000000e+00> : tensor<1x2048x768xf32>
    %cst_39 = stablehlo.constant dense<1.000000e+00> : tensor<1x2048x768xf32>
    %cst_40 = arith.constant dense<1.000000e+00> : tensor<1xf64>
    %cst_41 = arith.constant dense<-3.3895313892515355E+38> : tensor<1xf64>
    %cst_42 = arith.constant dense<768> : tensor<1xi64>
    %cst_43 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_44 = arith.constant dense<1> : tensor<1xi64>
    %cst_45 = arith.constant dense<5.6568542494923806> : tensor<1xf64>
    %cst_46 = arith.constant dense<1280> : tensor<1xi64>
    %0 = "stablehlo.gather"(%arg2, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<262x768xbf16>, tensor<1x2048xi64>) -> tensor<1x2048x768xbf16>
    %1 = stablehlo.convert %0 : tensor<1x2048x768xbf16>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x2048x768xbf16>) -> tensor<1x2048x768xbf16>
    %3 = stablehlo.broadcast_in_dim %arg116, dims = [1, 2] : (tensor<2048x768xbf16>) -> tensor<1x2048x768xbf16>
    %4 = stablehlo.add %2, %3 : tensor<1x2048x768xbf16>
    %5 = stablehlo.reshape %arg1 : (tensor<1x2048xi64>) -> tensor<1x1x2048xi64>
    %6 = stablehlo.reshape %5 : (tensor<1x1x2048xi64>) -> tensor<1x1x1x2048xi64>
    %7 = stablehlo.convert %6 : (tensor<1x1x1x2048xi64>) -> tensor<1x1x1x2048xbf16>
    %8 = stablehlo.convert %cst_40 : (tensor<1xf64>) -> tensor<1xbf16>
    %9 = stablehlo.reshape %8 : (tensor<1xbf16>) -> tensor<bf16>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<bf16>) -> tensor<1x1x1x2048xbf16>
    %11 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    %12 = stablehlo.subtract %10, %11 : tensor<1x1x1x2048xbf16>
    %13 = stablehlo.convert %cst_41 : (tensor<1xf64>) -> tensor<1xbf16>
    %14 = stablehlo.reshape %13 : (tensor<1xbf16>) -> tensor<bf16>
    %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    %16 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<bf16>) -> tensor<1x1x1x2048xbf16>
    %17 = stablehlo.multiply %15, %16 : tensor<1x1x1x2048xbf16>
    %18 = stablehlo.convert %4 : (tensor<1x2048x768xbf16>) -> tensor<1x2048x768xf32>
    %19 = stablehlo.convert %18 : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf64>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x2048x768xf64>, tensor<f64>) -> tensor<1x2048xf64>
    %21 = stablehlo.reshape %20 : (tensor<1x2048xf64>) -> tensor<1x2048x1xf64>
    %22 = stablehlo.convert %cst_42 : (tensor<1xi64>) -> tensor<1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
    %24 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x2048x1xf64>) -> tensor<1x2048x1xf64>
    %25 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f64>) -> tensor<1x2048x1xf64>
    %26 = stablehlo.divide %24, %25 : tensor<1x2048x1xf64>
    %27 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x2048x768xf64>) -> tensor<1x2048x768xf64>
    %28 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x2048x1xf64>) -> tensor<1x2048x768xf64>
    %29 = stablehlo.subtract %27, %28 : tensor<1x2048x768xf64>
    %30 = stablehlo.multiply %29, %29 : tensor<1x2048x768xf64>
    %31 = stablehlo.reduce(%30 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x2048x768xf64>, tensor<f64>) -> tensor<1x2048xf64>
    %32 = stablehlo.reshape %31 : (tensor<1x2048xf64>) -> tensor<1x2048x1xf64>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x2048x1xf64>) -> tensor<1x2048x1xf64>
    %34 = stablehlo.divide %33, %25 : tensor<1x2048x1xf64>
    %35 = stablehlo.convert %34 : (tensor<1x2048x1xf64>) -> tensor<1x2048x1xf32>
    %36 = stablehlo.reduce(%18 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x2048x768xf32>, tensor<f32>) -> tensor<1x2048xf32>
    %37 = stablehlo.reshape %36 : (tensor<1x2048xf32>) -> tensor<1x2048x1xf32>
    %38 = stablehlo.convert %cst_42 : (tensor<1xi64>) -> tensor<1xf32>
    %39 = stablehlo.reshape %38 : (tensor<1xf32>) -> tensor<f32>
    %40 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x1xf32>
    %41 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<1x2048x1xf32>
    %42 = stablehlo.divide %40, %41 : tensor<1x2048x1xf32>
    %43 = stablehlo.convert %cst_43 : (tensor<1xf64>) -> tensor<1xf32>
    %44 = stablehlo.reshape %43 : (tensor<1xf32>) -> tensor<f32>
    %45 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x1xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f32>) -> tensor<1x2048x1xf32>
    %47 = stablehlo.add %45, %46 : tensor<1x2048x1xf32>
    %48 = stablehlo.rsqrt %47 : tensor<1x2048x1xf32>
    %49 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %50 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x768xf32>
    %51 = stablehlo.subtract %49, %50 : tensor<1x2048x768xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %53 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x768xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<1x2048x768xf32>
    %55 = stablehlo.convert %arg3 : (tensor<768xbf16>) -> tensor<768xf32>
    %56 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %57 = stablehlo.broadcast_in_dim %55, dims = [2] : (tensor<768xf32>) -> tensor<1x2048x768xf32>
    %58 = stablehlo.multiply %56, %57 : tensor<1x2048x768xf32>
    %59 = stablehlo.convert %arg4 : (tensor<768xbf16>) -> tensor<768xf32>
    %60 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %61 = stablehlo.broadcast_in_dim %59, dims = [2] : (tensor<768xf32>) -> tensor<1x2048x768xf32>
    %62 = stablehlo.add %60, %61 : tensor<1x2048x768xf32>
    %63 = stablehlo.convert %62 : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xbf16>
    %64 = stablehlo.reshape %63 : (tensor<1x2048x768xbf16>) -> tensor<2048x768xbf16>
    %65 = stablehlo.convert %64 : (tensor<2048x768xbf16>) -> tensor<2048x768xf32>
    %66 = stablehlo.dot_general %65, %arg117, contracting_dims = [1] x [0] : (tensor<2048x768xf32>, tensor<768x256xf32>) -> tensor<2048x256xf32>
    %67 = stablehlo.convert %cst_44 : (tensor<1xi64>) -> tensor<1xf32>
    %68 = stablehlo.reshape %67 : (tensor<1xf32>) -> tensor<f32>
    %69 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<2048x256xf32>) -> tensor<2048x256xf32>
    %70 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<2048x256xf32>
    %71 = stablehlo.multiply %69, %70 : tensor<2048x256xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<2048x256xf32>) -> tensor<2048x256xf32>
    %73 = stablehlo.broadcast_in_dim %arg118, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
    %74 = stablehlo.add %72, %73 : tensor<2048x256xf32>
    %75 = stablehlo.convert %74 : (tensor<2048x256xf32>) -> tensor<2048x256xbf16>
    %76 = stablehlo.reshape %75 : (tensor<2048x256xbf16>) -> tensor<1x2048x256xbf16>
    %77 = stablehlo.dot_general %65, %arg119, contracting_dims = [1] x [0] : (tensor<2048x768xf32>, tensor<768x1280xf32>) -> tensor<2048x1280xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1] : (tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
    %79 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<2048x1280xf32>
    %80 = stablehlo.multiply %78, %79 : tensor<2048x1280xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1] : (tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
    %82 = stablehlo.broadcast_in_dim %arg120, dims = [1] : (tensor<1280xf32>) -> tensor<2048x1280xf32>
    %83 = stablehlo.add %81, %82 : tensor<2048x1280xf32>
    %84 = stablehlo.convert %83 : (tensor<2048x1280xf32>) -> tensor<2048x1280xbf16>
    %85 = stablehlo.reshape %84 : (tensor<2048x1280xbf16>) -> tensor<1x2048x1280xbf16>
    %86 = stablehlo.reshape %76 : (tensor<1x2048x256xbf16>) -> tensor<1x2048x8x32xbf16>
    %87 = stablehlo.transpose %86, dims = [0, 2, 1, 3] : (tensor<1x2048x8x32xbf16>) -> tensor<1x8x2048x32xbf16>
    %88 = stablehlo.reshape %85 : (tensor<1x2048x1280xbf16>) -> tensor<1x2048x8x160xbf16>
    %89 = stablehlo.transpose %88, dims = [0, 2, 1, 3] : (tensor<1x2048x8x160xbf16>) -> tensor<1x8x2048x160xbf16>
    %90 = stablehlo.transpose %87, dims = [0, 1, 3, 2] : (tensor<1x8x2048x32xbf16>) -> tensor<1x8x32x2048xbf16>
    %91 = stablehlo.reshape %90 : (tensor<1x8x32x2048xbf16>) -> tensor<8x32x2048xbf16>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2] : (tensor<8x32x2048xbf16>) -> tensor<8x32x2048xbf16>
    %93 = stablehlo.dot_general %arg121, %92, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x2048xbf16>) -> tensor<8x256x2048xbf16>
    %94 = stablehlo.reshape %93 : (tensor<8x256x2048xbf16>) -> tensor<1x8x256x2048xbf16>
    %95 = stablehlo.convert %cst_45 : (tensor<1xf64>) -> tensor<1xbf16>
    %96 = stablehlo.reshape %95 : (tensor<1xbf16>) -> tensor<bf16>
    %97 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x8x256x2048xbf16>) -> tensor<1x8x256x2048xbf16>
    %98 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<bf16>) -> tensor<1x8x256x2048xbf16>
    %99 = stablehlo.divide %97, %98 : tensor<1x8x256x2048xbf16>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2, 3] : (tensor<1x8x256x2048xbf16>) -> tensor<1x8x256x2048xbf16>
    %101 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xbf16>) -> tensor<1x8x256x2048xbf16>
    %102 = stablehlo.add %100, %101 : tensor<1x8x256x2048xbf16>
    %103 = stablehlo.convert %102 : (tensor<1x8x256x2048xbf16>) -> tensor<1x8x256x2048xf32>
    %104 = stablehlo.reduce(%103 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x2048xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %105 = stablehlo.reshape %104 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %106 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x8x256x2048xf32>) -> tensor<1x8x256x2048xf32>
    %107 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x2048xf32>
    %108 = stablehlo.subtract %106, %107 : tensor<1x8x256x2048xf32>
    %109 = stablehlo.exponential %108 : tensor<1x8x256x2048xf32>
    %110 = stablehlo.reduce(%109 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x2048xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %111 = stablehlo.reshape %110 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %112 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2, 3] : (tensor<1x8x256x2048xf32>) -> tensor<1x8x256x2048xf32>
    %113 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x2048xf32>
    %114 = stablehlo.divide %112, %113 : tensor<1x8x256x2048xf32>
    %115 = stablehlo.convert %114 : (tensor<1x8x256x2048xf32>) -> tensor<1x8x256x2048xbf16>
    %116 = stablehlo.reshape %115 : (tensor<1x8x256x2048xbf16>) -> tensor<8x256x2048xbf16>
    %117 = stablehlo.reshape %89 : (tensor<1x8x2048x160xbf16>) -> tensor<8x2048x160xbf16>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<8x2048x160xbf16>) -> tensor<8x2048x160xbf16>
    %119 = stablehlo.dot_general %116, %118, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x2048xbf16>, tensor<8x2048x160xbf16>) -> tensor<8x256x160xbf16>
    %120 = stablehlo.reshape %119 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %121 = stablehlo.transpose %120, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %122 = stablehlo.reshape %121 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %123 = stablehlo.reshape %122 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %124 = stablehlo.convert %123 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %125 = stablehlo.dot_general %124, %arg122, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %127 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<256x1280xf32>
    %128 = stablehlo.multiply %126, %127 : tensor<256x1280xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %130 = stablehlo.broadcast_in_dim %arg123, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %131 = stablehlo.add %129, %130 : tensor<256x1280xf32>
    %132 = stablehlo.convert %131 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %133 = stablehlo.reshape %132 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %134 = stablehlo.add %133, %arg124 : tensor<1x256x1280xbf16>
    %135 = stablehlo.convert %134 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %136 = stablehlo.convert %135 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %137 = stablehlo.reduce(%136 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %138 = stablehlo.reshape %137 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %139 = stablehlo.convert %cst_46 : (tensor<1xi64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %142 = stablehlo.broadcast_in_dim %140, dims = [] : (tensor<f64>) -> tensor<1x256x1xf64>
    %143 = stablehlo.divide %141, %142 : tensor<1x256x1xf64>
    %144 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %145 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %146 = stablehlo.subtract %144, %145 : tensor<1x256x1280xf64>
    %147 = stablehlo.multiply %146, %146 : tensor<1x256x1280xf64>
    %148 = stablehlo.reduce(%147 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %149 = stablehlo.reshape %148 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %151 = stablehlo.divide %150, %142 : tensor<1x256x1xf64>
    %152 = stablehlo.convert %151 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %153 = stablehlo.reduce(%135 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %154 = stablehlo.reshape %153 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %155 = stablehlo.convert %cst_46 : (tensor<1xi64>) -> tensor<1xf32>
    %156 = stablehlo.reshape %155 : (tensor<1xf32>) -> tensor<f32>
    %157 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %158 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %159 = stablehlo.divide %157, %158 : tensor<1x256x1xf32>
    %160 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %161 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f32>) -> tensor<1x256x1xf32>
    %162 = stablehlo.add %160, %161 : tensor<1x256x1xf32>
    %163 = stablehlo.rsqrt %162 : tensor<1x256x1xf32>
    %164 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %165 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %166 = stablehlo.subtract %164, %165 : tensor<1x256x1280xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %168 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %169 = stablehlo.multiply %167, %168 : tensor<1x256x1280xf32>
    %170 = stablehlo.convert %arg5 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %171 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %172 = stablehlo.broadcast_in_dim %170, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %173 = stablehlo.multiply %171, %172 : tensor<1x256x1280xf32>
    %174 = stablehlo.convert %arg6 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %175 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %176 = stablehlo.broadcast_in_dim %174, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %177 = stablehlo.add %175, %176 : tensor<1x256x1280xf32>
    %178 = stablehlo.convert %177 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %179 = stablehlo.reshape %178 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %180 = stablehlo.convert %179 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %181 = stablehlo.dot_general %180, %arg125, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %183 = stablehlo.multiply %182, %127 : tensor<256x1280xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %185 = stablehlo.broadcast_in_dim %arg126, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %186 = stablehlo.add %184, %185 : tensor<256x1280xf32>
    %187 = stablehlo.convert %186 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %188 = stablehlo.reshape %187 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %189 = stablehlo.multiply %188, %cst_4 : tensor<1x256x1280xbf16>
    %190 = stablehlo.rsqrt %cst_3 : tensor<1x256x1280xbf16>
    %191 = stablehlo.multiply %188, %190 : tensor<1x256x1280xbf16>
    %192 = stablehlo.convert %191 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %193 = stablehlo.clamp %cst_5, %192, %cst_6 : tensor<1x256x1280xf32>
    %194 = stablehlo.multiply %193, %193 : tensor<1x256x1280xf32>
    %195 = stablehlo.multiply %cst_7, %194 : tensor<1x256x1280xf32>
    %196 = stablehlo.add %195, %cst_8 : tensor<1x256x1280xf32>
    %197 = stablehlo.multiply %196, %194 : tensor<1x256x1280xf32>
    %198 = stablehlo.add %197, %cst_9 : tensor<1x256x1280xf32>
    %199 = stablehlo.multiply %198, %194 : tensor<1x256x1280xf32>
    %200 = stablehlo.add %199, %cst_10 : tensor<1x256x1280xf32>
    %201 = stablehlo.multiply %200, %194 : tensor<1x256x1280xf32>
    %202 = stablehlo.add %201, %cst_11 : tensor<1x256x1280xf32>
    %203 = stablehlo.multiply %202, %194 : tensor<1x256x1280xf32>
    %204 = stablehlo.add %203, %cst_12 : tensor<1x256x1280xf32>
    %205 = stablehlo.multiply %204, %194 : tensor<1x256x1280xf32>
    %206 = stablehlo.add %205, %cst_13 : tensor<1x256x1280xf32>
    %207 = stablehlo.multiply %cst_14, %194 : tensor<1x256x1280xf32>
    %208 = stablehlo.add %207, %cst_15 : tensor<1x256x1280xf32>
    %209 = stablehlo.multiply %208, %194 : tensor<1x256x1280xf32>
    %210 = stablehlo.add %209, %cst_16 : tensor<1x256x1280xf32>
    %211 = stablehlo.multiply %210, %194 : tensor<1x256x1280xf32>
    %212 = stablehlo.add %211, %cst_17 : tensor<1x256x1280xf32>
    %213 = stablehlo.multiply %212, %194 : tensor<1x256x1280xf32>
    %214 = stablehlo.add %213, %cst_18 : tensor<1x256x1280xf32>
    %215 = stablehlo.multiply %193, %206 : tensor<1x256x1280xf32>
    %216 = stablehlo.divide %215, %214 : tensor<1x256x1280xf32>
    %217 = stablehlo.clamp %cst_19, %216, %cst_20 : tensor<1x256x1280xf32>
    %218 = stablehlo.convert %217 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %219 = stablehlo.add %218, %cst_2 : tensor<1x256x1280xbf16>
    %220 = stablehlo.multiply %219, %189 : tensor<1x256x1280xbf16>
    %221 = stablehlo.reshape %220 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %222 = stablehlo.convert %221 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %223 = stablehlo.dot_general %222, %arg127, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %225 = stablehlo.multiply %224, %127 : tensor<256x1280xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %227 = stablehlo.broadcast_in_dim %arg128, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %228 = stablehlo.add %226, %227 : tensor<256x1280xf32>
    %229 = stablehlo.convert %228 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %230 = stablehlo.reshape %229 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %231 = stablehlo.add %230, %134 : tensor<1x256x1280xbf16>
    %232 = stablehlo.convert %231 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %233 = stablehlo.convert %232 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %234 = stablehlo.reduce(%233 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %235 = stablehlo.reshape %234 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %236 = stablehlo.broadcast_in_dim %235, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %237 = stablehlo.divide %236, %142 : tensor<1x256x1xf64>
    %238 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %239 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %240 = stablehlo.subtract %238, %239 : tensor<1x256x1280xf64>
    %241 = stablehlo.multiply %240, %240 : tensor<1x256x1280xf64>
    %242 = stablehlo.reduce(%241 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %243 = stablehlo.reshape %242 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %245 = stablehlo.divide %244, %142 : tensor<1x256x1xf64>
    %246 = stablehlo.convert %245 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %247 = stablehlo.reduce(%232 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %248 = stablehlo.reshape %247 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %250 = stablehlo.divide %249, %158 : tensor<1x256x1xf32>
    %251 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %252 = stablehlo.add %251, %161 : tensor<1x256x1xf32>
    %253 = stablehlo.rsqrt %252 : tensor<1x256x1xf32>
    %254 = stablehlo.broadcast_in_dim %232, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %255 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %256 = stablehlo.subtract %254, %255 : tensor<1x256x1280xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %258 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %259 = stablehlo.multiply %257, %258 : tensor<1x256x1280xf32>
    %260 = stablehlo.convert %arg7 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %261 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %262 = stablehlo.broadcast_in_dim %260, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %263 = stablehlo.multiply %261, %262 : tensor<1x256x1280xf32>
    %264 = stablehlo.convert %arg8 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %265 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %266 = stablehlo.broadcast_in_dim %264, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %267 = stablehlo.add %265, %266 : tensor<1x256x1280xf32>
    %268 = stablehlo.convert %267 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %269 = stablehlo.reshape %268 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %270 = stablehlo.convert %269 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %271 = stablehlo.dot_general %270, %arg129, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %273 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %274 = stablehlo.multiply %272, %273 : tensor<256x256xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %276 = stablehlo.broadcast_in_dim %arg130, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %277 = stablehlo.add %275, %276 : tensor<256x256xf32>
    %278 = stablehlo.convert %277 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %279 = stablehlo.reshape %278 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %280 = stablehlo.dot_general %270, %arg131, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %282 = stablehlo.multiply %281, %273 : tensor<256x256xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %284 = stablehlo.broadcast_in_dim %arg132, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %285 = stablehlo.add %283, %284 : tensor<256x256xf32>
    %286 = stablehlo.convert %285 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %287 = stablehlo.reshape %286 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %288 = stablehlo.dot_general %270, %arg133, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %290 = stablehlo.multiply %289, %127 : tensor<256x1280xf32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %292 = stablehlo.broadcast_in_dim %arg134, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %293 = stablehlo.add %291, %292 : tensor<256x1280xf32>
    %294 = stablehlo.convert %293 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %295 = stablehlo.reshape %294 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %296 = stablehlo.reshape %279 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %297 = stablehlo.transpose %296, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %298 = stablehlo.reshape %287 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %300 = stablehlo.reshape %295 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %302 = stablehlo.transpose %299, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %303 = stablehlo.reshape %297 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %304 = stablehlo.reshape %302 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %305 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %306 = stablehlo.dot_general %303, %305, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %307 = stablehlo.reshape %306 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %309 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<bf16>) -> tensor<1x8x256x256xbf16>
    %310 = stablehlo.divide %308, %309 : tensor<1x8x256x256xbf16>
    %311 = stablehlo.convert %310 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %312 = stablehlo.reduce(%311 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %313 = stablehlo.reshape %312 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %314 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %315 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %316 = stablehlo.subtract %314, %315 : tensor<1x8x256x256xf32>
    %317 = stablehlo.exponential %316 : tensor<1x8x256x256xf32>
    %318 = stablehlo.reduce(%317 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %319 = stablehlo.reshape %318 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %320 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %321 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %322 = stablehlo.divide %320, %321 : tensor<1x8x256x256xf32>
    %323 = stablehlo.convert %322 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %324 = stablehlo.reshape %323 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %325 = stablehlo.reshape %301 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %327 = stablehlo.dot_general %324, %326, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %328 = stablehlo.reshape %327 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %329 = stablehlo.transpose %328, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %330 = stablehlo.reshape %329 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %331 = stablehlo.reshape %330 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %332 = stablehlo.convert %331 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %333 = stablehlo.dot_general %332, %arg135, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %335 = stablehlo.multiply %334, %127 : tensor<256x1280xf32>
    %336 = stablehlo.broadcast_in_dim %335, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %337 = stablehlo.broadcast_in_dim %arg136, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %338 = stablehlo.add %336, %337 : tensor<256x1280xf32>
    %339 = stablehlo.convert %338 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %340 = stablehlo.reshape %339 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %341 = stablehlo.add %340, %231 : tensor<1x256x1280xbf16>
    %342 = stablehlo.convert %341 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %343 = stablehlo.convert %342 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %344 = stablehlo.reduce(%343 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %345 = stablehlo.reshape %344 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %347 = stablehlo.divide %346, %142 : tensor<1x256x1xf64>
    %348 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %349 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %350 = stablehlo.subtract %348, %349 : tensor<1x256x1280xf64>
    %351 = stablehlo.multiply %350, %350 : tensor<1x256x1280xf64>
    %352 = stablehlo.reduce(%351 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %353 = stablehlo.reshape %352 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %355 = stablehlo.divide %354, %142 : tensor<1x256x1xf64>
    %356 = stablehlo.convert %355 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %357 = stablehlo.reduce(%342 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %358 = stablehlo.reshape %357 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %359 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %360 = stablehlo.divide %359, %158 : tensor<1x256x1xf32>
    %361 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %362 = stablehlo.add %361, %161 : tensor<1x256x1xf32>
    %363 = stablehlo.rsqrt %362 : tensor<1x256x1xf32>
    %364 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %365 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %366 = stablehlo.subtract %364, %365 : tensor<1x256x1280xf32>
    %367 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %368 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %369 = stablehlo.multiply %367, %368 : tensor<1x256x1280xf32>
    %370 = stablehlo.convert %arg9 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %371 = stablehlo.broadcast_in_dim %369, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %372 = stablehlo.broadcast_in_dim %370, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %373 = stablehlo.multiply %371, %372 : tensor<1x256x1280xf32>
    %374 = stablehlo.convert %arg10 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %375 = stablehlo.broadcast_in_dim %373, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %376 = stablehlo.broadcast_in_dim %374, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %377 = stablehlo.add %375, %376 : tensor<1x256x1280xf32>
    %378 = stablehlo.convert %377 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %379 = stablehlo.reshape %378 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %380 = stablehlo.convert %379 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %381 = stablehlo.dot_general %380, %arg137, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %383 = stablehlo.multiply %382, %127 : tensor<256x1280xf32>
    %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %385 = stablehlo.broadcast_in_dim %arg138, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %386 = stablehlo.add %384, %385 : tensor<256x1280xf32>
    %387 = stablehlo.convert %386 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %388 = stablehlo.reshape %387 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %389 = stablehlo.multiply %388, %cst_4 : tensor<1x256x1280xbf16>
    %390 = stablehlo.multiply %388, %190 : tensor<1x256x1280xbf16>
    %391 = stablehlo.convert %390 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %392 = stablehlo.clamp %cst_5, %391, %cst_6 : tensor<1x256x1280xf32>
    %393 = stablehlo.multiply %392, %392 : tensor<1x256x1280xf32>
    %394 = stablehlo.multiply %cst_7, %393 : tensor<1x256x1280xf32>
    %395 = stablehlo.add %394, %cst_8 : tensor<1x256x1280xf32>
    %396 = stablehlo.multiply %395, %393 : tensor<1x256x1280xf32>
    %397 = stablehlo.add %396, %cst_9 : tensor<1x256x1280xf32>
    %398 = stablehlo.multiply %397, %393 : tensor<1x256x1280xf32>
    %399 = stablehlo.add %398, %cst_10 : tensor<1x256x1280xf32>
    %400 = stablehlo.multiply %399, %393 : tensor<1x256x1280xf32>
    %401 = stablehlo.add %400, %cst_11 : tensor<1x256x1280xf32>
    %402 = stablehlo.multiply %401, %393 : tensor<1x256x1280xf32>
    %403 = stablehlo.add %402, %cst_12 : tensor<1x256x1280xf32>
    %404 = stablehlo.multiply %403, %393 : tensor<1x256x1280xf32>
    %405 = stablehlo.add %404, %cst_13 : tensor<1x256x1280xf32>
    %406 = stablehlo.multiply %cst_14, %393 : tensor<1x256x1280xf32>
    %407 = stablehlo.add %406, %cst_15 : tensor<1x256x1280xf32>
    %408 = stablehlo.multiply %407, %393 : tensor<1x256x1280xf32>
    %409 = stablehlo.add %408, %cst_16 : tensor<1x256x1280xf32>
    %410 = stablehlo.multiply %409, %393 : tensor<1x256x1280xf32>
    %411 = stablehlo.add %410, %cst_17 : tensor<1x256x1280xf32>
    %412 = stablehlo.multiply %411, %393 : tensor<1x256x1280xf32>
    %413 = stablehlo.add %412, %cst_18 : tensor<1x256x1280xf32>
    %414 = stablehlo.multiply %392, %405 : tensor<1x256x1280xf32>
    %415 = stablehlo.divide %414, %413 : tensor<1x256x1280xf32>
    %416 = stablehlo.clamp %cst_19, %415, %cst_20 : tensor<1x256x1280xf32>
    %417 = stablehlo.convert %416 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %418 = stablehlo.add %417, %cst_2 : tensor<1x256x1280xbf16>
    %419 = stablehlo.multiply %418, %389 : tensor<1x256x1280xbf16>
    %420 = stablehlo.reshape %419 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %421 = stablehlo.convert %420 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %422 = stablehlo.dot_general %421, %arg139, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %424 = stablehlo.multiply %423, %127 : tensor<256x1280xf32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %426 = stablehlo.broadcast_in_dim %arg140, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %427 = stablehlo.add %425, %426 : tensor<256x1280xf32>
    %428 = stablehlo.convert %427 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %429 = stablehlo.reshape %428 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %430 = stablehlo.add %429, %341 : tensor<1x256x1280xbf16>
    %431 = stablehlo.convert %430 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %432 = stablehlo.convert %431 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %433 = stablehlo.reduce(%432 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %434 = stablehlo.reshape %433 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %436 = stablehlo.divide %435, %142 : tensor<1x256x1xf64>
    %437 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %438 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %439 = stablehlo.subtract %437, %438 : tensor<1x256x1280xf64>
    %440 = stablehlo.multiply %439, %439 : tensor<1x256x1280xf64>
    %441 = stablehlo.reduce(%440 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %442 = stablehlo.reshape %441 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %444 = stablehlo.divide %443, %142 : tensor<1x256x1xf64>
    %445 = stablehlo.convert %444 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %446 = stablehlo.reduce(%431 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %447 = stablehlo.reshape %446 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %448 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %449 = stablehlo.divide %448, %158 : tensor<1x256x1xf32>
    %450 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %451 = stablehlo.add %450, %161 : tensor<1x256x1xf32>
    %452 = stablehlo.rsqrt %451 : tensor<1x256x1xf32>
    %453 = stablehlo.broadcast_in_dim %431, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %454 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %455 = stablehlo.subtract %453, %454 : tensor<1x256x1280xf32>
    %456 = stablehlo.broadcast_in_dim %455, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %457 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %458 = stablehlo.multiply %456, %457 : tensor<1x256x1280xf32>
    %459 = stablehlo.convert %arg11 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %460 = stablehlo.broadcast_in_dim %458, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %461 = stablehlo.broadcast_in_dim %459, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %462 = stablehlo.multiply %460, %461 : tensor<1x256x1280xf32>
    %463 = stablehlo.convert %arg12 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %464 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %465 = stablehlo.broadcast_in_dim %463, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %466 = stablehlo.add %464, %465 : tensor<1x256x1280xf32>
    %467 = stablehlo.convert %466 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %468 = stablehlo.reshape %467 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %469 = stablehlo.convert %468 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %470 = stablehlo.dot_general %469, %arg141, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %471 = stablehlo.broadcast_in_dim %470, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %472 = stablehlo.multiply %471, %273 : tensor<256x256xf32>
    %473 = stablehlo.broadcast_in_dim %472, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %474 = stablehlo.broadcast_in_dim %arg142, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %475 = stablehlo.add %473, %474 : tensor<256x256xf32>
    %476 = stablehlo.convert %475 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %477 = stablehlo.reshape %476 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %478 = stablehlo.dot_general %469, %arg143, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %480 = stablehlo.multiply %479, %273 : tensor<256x256xf32>
    %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %482 = stablehlo.broadcast_in_dim %arg144, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %483 = stablehlo.add %481, %482 : tensor<256x256xf32>
    %484 = stablehlo.convert %483 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %485 = stablehlo.reshape %484 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %486 = stablehlo.dot_general %469, %arg145, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %488 = stablehlo.multiply %487, %127 : tensor<256x1280xf32>
    %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %490 = stablehlo.broadcast_in_dim %arg146, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %491 = stablehlo.add %489, %490 : tensor<256x1280xf32>
    %492 = stablehlo.convert %491 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %493 = stablehlo.reshape %492 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %494 = stablehlo.reshape %477 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %495 = stablehlo.transpose %494, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %496 = stablehlo.reshape %485 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %497 = stablehlo.transpose %496, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %498 = stablehlo.reshape %493 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %499 = stablehlo.transpose %498, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %500 = stablehlo.transpose %497, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %501 = stablehlo.reshape %495 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %502 = stablehlo.reshape %500 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %503 = stablehlo.broadcast_in_dim %502, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %504 = stablehlo.dot_general %501, %503, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %505 = stablehlo.reshape %504 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %506 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %507 = stablehlo.divide %506, %309 : tensor<1x8x256x256xbf16>
    %508 = stablehlo.convert %507 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %509 = stablehlo.reduce(%508 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %510 = stablehlo.reshape %509 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %511 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %512 = stablehlo.broadcast_in_dim %510, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %513 = stablehlo.subtract %511, %512 : tensor<1x8x256x256xf32>
    %514 = stablehlo.exponential %513 : tensor<1x8x256x256xf32>
    %515 = stablehlo.reduce(%514 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %516 = stablehlo.reshape %515 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %517 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %518 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %519 = stablehlo.divide %517, %518 : tensor<1x8x256x256xf32>
    %520 = stablehlo.convert %519 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %521 = stablehlo.reshape %520 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %522 = stablehlo.reshape %499 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %523 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %524 = stablehlo.dot_general %521, %523, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %525 = stablehlo.reshape %524 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %526 = stablehlo.transpose %525, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %527 = stablehlo.reshape %526 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %528 = stablehlo.reshape %527 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %529 = stablehlo.convert %528 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %530 = stablehlo.dot_general %529, %arg147, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %532 = stablehlo.multiply %531, %127 : tensor<256x1280xf32>
    %533 = stablehlo.broadcast_in_dim %532, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %534 = stablehlo.broadcast_in_dim %arg148, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %535 = stablehlo.add %533, %534 : tensor<256x1280xf32>
    %536 = stablehlo.convert %535 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %537 = stablehlo.reshape %536 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %538 = stablehlo.add %537, %430 : tensor<1x256x1280xbf16>
    %539 = stablehlo.convert %538 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %540 = stablehlo.convert %539 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %541 = stablehlo.reduce(%540 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %542 = stablehlo.reshape %541 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %543 = stablehlo.broadcast_in_dim %542, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %544 = stablehlo.divide %543, %142 : tensor<1x256x1xf64>
    %545 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %546 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %547 = stablehlo.subtract %545, %546 : tensor<1x256x1280xf64>
    %548 = stablehlo.multiply %547, %547 : tensor<1x256x1280xf64>
    %549 = stablehlo.reduce(%548 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %550 = stablehlo.reshape %549 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %551 = stablehlo.broadcast_in_dim %550, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %552 = stablehlo.divide %551, %142 : tensor<1x256x1xf64>
    %553 = stablehlo.convert %552 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %554 = stablehlo.reduce(%539 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %555 = stablehlo.reshape %554 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %556 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %557 = stablehlo.divide %556, %158 : tensor<1x256x1xf32>
    %558 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %559 = stablehlo.add %558, %161 : tensor<1x256x1xf32>
    %560 = stablehlo.rsqrt %559 : tensor<1x256x1xf32>
    %561 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %562 = stablehlo.broadcast_in_dim %557, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %563 = stablehlo.subtract %561, %562 : tensor<1x256x1280xf32>
    %564 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %565 = stablehlo.broadcast_in_dim %560, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %566 = stablehlo.multiply %564, %565 : tensor<1x256x1280xf32>
    %567 = stablehlo.convert %arg13 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %568 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %569 = stablehlo.broadcast_in_dim %567, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %570 = stablehlo.multiply %568, %569 : tensor<1x256x1280xf32>
    %571 = stablehlo.convert %arg14 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %572 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %573 = stablehlo.broadcast_in_dim %571, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %574 = stablehlo.add %572, %573 : tensor<1x256x1280xf32>
    %575 = stablehlo.convert %574 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %576 = stablehlo.reshape %575 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %577 = stablehlo.convert %576 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %578 = stablehlo.dot_general %577, %arg149, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %579 = stablehlo.broadcast_in_dim %578, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %580 = stablehlo.multiply %579, %127 : tensor<256x1280xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %582 = stablehlo.broadcast_in_dim %arg150, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %583 = stablehlo.add %581, %582 : tensor<256x1280xf32>
    %584 = stablehlo.convert %583 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %585 = stablehlo.reshape %584 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %586 = stablehlo.multiply %585, %cst_4 : tensor<1x256x1280xbf16>
    %587 = stablehlo.multiply %585, %190 : tensor<1x256x1280xbf16>
    %588 = stablehlo.convert %587 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %589 = stablehlo.clamp %cst_5, %588, %cst_6 : tensor<1x256x1280xf32>
    %590 = stablehlo.multiply %589, %589 : tensor<1x256x1280xf32>
    %591 = stablehlo.multiply %cst_7, %590 : tensor<1x256x1280xf32>
    %592 = stablehlo.add %591, %cst_8 : tensor<1x256x1280xf32>
    %593 = stablehlo.multiply %592, %590 : tensor<1x256x1280xf32>
    %594 = stablehlo.add %593, %cst_9 : tensor<1x256x1280xf32>
    %595 = stablehlo.multiply %594, %590 : tensor<1x256x1280xf32>
    %596 = stablehlo.add %595, %cst_10 : tensor<1x256x1280xf32>
    %597 = stablehlo.multiply %596, %590 : tensor<1x256x1280xf32>
    %598 = stablehlo.add %597, %cst_11 : tensor<1x256x1280xf32>
    %599 = stablehlo.multiply %598, %590 : tensor<1x256x1280xf32>
    %600 = stablehlo.add %599, %cst_12 : tensor<1x256x1280xf32>
    %601 = stablehlo.multiply %600, %590 : tensor<1x256x1280xf32>
    %602 = stablehlo.add %601, %cst_13 : tensor<1x256x1280xf32>
    %603 = stablehlo.multiply %cst_14, %590 : tensor<1x256x1280xf32>
    %604 = stablehlo.add %603, %cst_15 : tensor<1x256x1280xf32>
    %605 = stablehlo.multiply %604, %590 : tensor<1x256x1280xf32>
    %606 = stablehlo.add %605, %cst_16 : tensor<1x256x1280xf32>
    %607 = stablehlo.multiply %606, %590 : tensor<1x256x1280xf32>
    %608 = stablehlo.add %607, %cst_17 : tensor<1x256x1280xf32>
    %609 = stablehlo.multiply %608, %590 : tensor<1x256x1280xf32>
    %610 = stablehlo.add %609, %cst_18 : tensor<1x256x1280xf32>
    %611 = stablehlo.multiply %589, %602 : tensor<1x256x1280xf32>
    %612 = stablehlo.divide %611, %610 : tensor<1x256x1280xf32>
    %613 = stablehlo.clamp %cst_19, %612, %cst_20 : tensor<1x256x1280xf32>
    %614 = stablehlo.convert %613 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %615 = stablehlo.add %614, %cst_2 : tensor<1x256x1280xbf16>
    %616 = stablehlo.multiply %615, %586 : tensor<1x256x1280xbf16>
    %617 = stablehlo.reshape %616 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %618 = stablehlo.convert %617 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %619 = stablehlo.dot_general %618, %arg151, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %620 = stablehlo.broadcast_in_dim %619, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %621 = stablehlo.multiply %620, %127 : tensor<256x1280xf32>
    %622 = stablehlo.broadcast_in_dim %621, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %623 = stablehlo.broadcast_in_dim %arg152, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %624 = stablehlo.add %622, %623 : tensor<256x1280xf32>
    %625 = stablehlo.convert %624 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %626 = stablehlo.reshape %625 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %627 = stablehlo.add %626, %538 : tensor<1x256x1280xbf16>
    %628 = stablehlo.convert %627 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %629 = stablehlo.convert %628 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %630 = stablehlo.reduce(%629 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %631 = stablehlo.reshape %630 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %632 = stablehlo.broadcast_in_dim %631, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %633 = stablehlo.divide %632, %142 : tensor<1x256x1xf64>
    %634 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %635 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %636 = stablehlo.subtract %634, %635 : tensor<1x256x1280xf64>
    %637 = stablehlo.multiply %636, %636 : tensor<1x256x1280xf64>
    %638 = stablehlo.reduce(%637 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %639 = stablehlo.reshape %638 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %640 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %641 = stablehlo.divide %640, %142 : tensor<1x256x1xf64>
    %642 = stablehlo.convert %641 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %643 = stablehlo.reduce(%628 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %644 = stablehlo.reshape %643 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %645 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %646 = stablehlo.divide %645, %158 : tensor<1x256x1xf32>
    %647 = stablehlo.broadcast_in_dim %642, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %648 = stablehlo.add %647, %161 : tensor<1x256x1xf32>
    %649 = stablehlo.rsqrt %648 : tensor<1x256x1xf32>
    %650 = stablehlo.broadcast_in_dim %628, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %651 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %652 = stablehlo.subtract %650, %651 : tensor<1x256x1280xf32>
    %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %654 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %655 = stablehlo.multiply %653, %654 : tensor<1x256x1280xf32>
    %656 = stablehlo.convert %arg15 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %657 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %658 = stablehlo.broadcast_in_dim %656, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %659 = stablehlo.multiply %657, %658 : tensor<1x256x1280xf32>
    %660 = stablehlo.convert %arg16 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %661 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %662 = stablehlo.broadcast_in_dim %660, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %663 = stablehlo.add %661, %662 : tensor<1x256x1280xf32>
    %664 = stablehlo.convert %663 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %665 = stablehlo.reshape %664 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %666 = stablehlo.convert %665 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %667 = stablehlo.dot_general %666, %arg153, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %668 = stablehlo.broadcast_in_dim %667, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %669 = stablehlo.multiply %668, %273 : tensor<256x256xf32>
    %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %671 = stablehlo.broadcast_in_dim %arg154, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %672 = stablehlo.add %670, %671 : tensor<256x256xf32>
    %673 = stablehlo.convert %672 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %674 = stablehlo.reshape %673 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %675 = stablehlo.dot_general %666, %arg155, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %677 = stablehlo.multiply %676, %273 : tensor<256x256xf32>
    %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %679 = stablehlo.broadcast_in_dim %arg156, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %680 = stablehlo.add %678, %679 : tensor<256x256xf32>
    %681 = stablehlo.convert %680 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %682 = stablehlo.reshape %681 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %683 = stablehlo.dot_general %666, %arg157, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %684 = stablehlo.broadcast_in_dim %683, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %685 = stablehlo.multiply %684, %127 : tensor<256x1280xf32>
    %686 = stablehlo.broadcast_in_dim %685, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %687 = stablehlo.broadcast_in_dim %arg158, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %688 = stablehlo.add %686, %687 : tensor<256x1280xf32>
    %689 = stablehlo.convert %688 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %690 = stablehlo.reshape %689 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %691 = stablehlo.reshape %674 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %692 = stablehlo.transpose %691, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %693 = stablehlo.reshape %682 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %694 = stablehlo.transpose %693, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %695 = stablehlo.reshape %690 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %696 = stablehlo.transpose %695, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %697 = stablehlo.transpose %694, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %698 = stablehlo.reshape %692 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %699 = stablehlo.reshape %697 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %700 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %701 = stablehlo.dot_general %698, %700, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %702 = stablehlo.reshape %701 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %703 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %704 = stablehlo.divide %703, %309 : tensor<1x8x256x256xbf16>
    %705 = stablehlo.convert %704 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %706 = stablehlo.reduce(%705 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %707 = stablehlo.reshape %706 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %708 = stablehlo.broadcast_in_dim %705, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %709 = stablehlo.broadcast_in_dim %707, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %710 = stablehlo.subtract %708, %709 : tensor<1x8x256x256xf32>
    %711 = stablehlo.exponential %710 : tensor<1x8x256x256xf32>
    %712 = stablehlo.reduce(%711 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %713 = stablehlo.reshape %712 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %714 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %715 = stablehlo.broadcast_in_dim %713, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %716 = stablehlo.divide %714, %715 : tensor<1x8x256x256xf32>
    %717 = stablehlo.convert %716 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %718 = stablehlo.reshape %717 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %719 = stablehlo.reshape %696 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %720 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %721 = stablehlo.dot_general %718, %720, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %722 = stablehlo.reshape %721 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %723 = stablehlo.transpose %722, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %724 = stablehlo.reshape %723 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %725 = stablehlo.reshape %724 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %726 = stablehlo.convert %725 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %727 = stablehlo.dot_general %726, %arg159, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %729 = stablehlo.multiply %728, %127 : tensor<256x1280xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %731 = stablehlo.broadcast_in_dim %arg160, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %732 = stablehlo.add %730, %731 : tensor<256x1280xf32>
    %733 = stablehlo.convert %732 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %734 = stablehlo.reshape %733 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %735 = stablehlo.add %734, %627 : tensor<1x256x1280xbf16>
    %736 = stablehlo.convert %735 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %737 = stablehlo.convert %736 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %738 = stablehlo.reduce(%737 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %739 = stablehlo.reshape %738 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %741 = stablehlo.divide %740, %142 : tensor<1x256x1xf64>
    %742 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %743 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %744 = stablehlo.subtract %742, %743 : tensor<1x256x1280xf64>
    %745 = stablehlo.multiply %744, %744 : tensor<1x256x1280xf64>
    %746 = stablehlo.reduce(%745 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %747 = stablehlo.reshape %746 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %748 = stablehlo.broadcast_in_dim %747, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %749 = stablehlo.divide %748, %142 : tensor<1x256x1xf64>
    %750 = stablehlo.convert %749 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %751 = stablehlo.reduce(%736 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %752 = stablehlo.reshape %751 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %753 = stablehlo.broadcast_in_dim %752, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %754 = stablehlo.divide %753, %158 : tensor<1x256x1xf32>
    %755 = stablehlo.broadcast_in_dim %750, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %756 = stablehlo.add %755, %161 : tensor<1x256x1xf32>
    %757 = stablehlo.rsqrt %756 : tensor<1x256x1xf32>
    %758 = stablehlo.broadcast_in_dim %736, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %759 = stablehlo.broadcast_in_dim %754, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %760 = stablehlo.subtract %758, %759 : tensor<1x256x1280xf32>
    %761 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %762 = stablehlo.broadcast_in_dim %757, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %763 = stablehlo.multiply %761, %762 : tensor<1x256x1280xf32>
    %764 = stablehlo.convert %arg17 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %765 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %766 = stablehlo.broadcast_in_dim %764, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %767 = stablehlo.multiply %765, %766 : tensor<1x256x1280xf32>
    %768 = stablehlo.convert %arg18 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %769 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %770 = stablehlo.broadcast_in_dim %768, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %771 = stablehlo.add %769, %770 : tensor<1x256x1280xf32>
    %772 = stablehlo.convert %771 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %773 = stablehlo.reshape %772 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %774 = stablehlo.convert %773 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %775 = stablehlo.dot_general %774, %arg161, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %776 = stablehlo.broadcast_in_dim %775, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %777 = stablehlo.multiply %776, %127 : tensor<256x1280xf32>
    %778 = stablehlo.broadcast_in_dim %777, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %779 = stablehlo.broadcast_in_dim %arg162, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %780 = stablehlo.add %778, %779 : tensor<256x1280xf32>
    %781 = stablehlo.convert %780 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %782 = stablehlo.reshape %781 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %783 = stablehlo.multiply %782, %cst_4 : tensor<1x256x1280xbf16>
    %784 = stablehlo.multiply %782, %190 : tensor<1x256x1280xbf16>
    %785 = stablehlo.convert %784 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %786 = stablehlo.clamp %cst_5, %785, %cst_6 : tensor<1x256x1280xf32>
    %787 = stablehlo.multiply %786, %786 : tensor<1x256x1280xf32>
    %788 = stablehlo.multiply %cst_7, %787 : tensor<1x256x1280xf32>
    %789 = stablehlo.add %788, %cst_8 : tensor<1x256x1280xf32>
    %790 = stablehlo.multiply %789, %787 : tensor<1x256x1280xf32>
    %791 = stablehlo.add %790, %cst_9 : tensor<1x256x1280xf32>
    %792 = stablehlo.multiply %791, %787 : tensor<1x256x1280xf32>
    %793 = stablehlo.add %792, %cst_10 : tensor<1x256x1280xf32>
    %794 = stablehlo.multiply %793, %787 : tensor<1x256x1280xf32>
    %795 = stablehlo.add %794, %cst_11 : tensor<1x256x1280xf32>
    %796 = stablehlo.multiply %795, %787 : tensor<1x256x1280xf32>
    %797 = stablehlo.add %796, %cst_12 : tensor<1x256x1280xf32>
    %798 = stablehlo.multiply %797, %787 : tensor<1x256x1280xf32>
    %799 = stablehlo.add %798, %cst_13 : tensor<1x256x1280xf32>
    %800 = stablehlo.multiply %cst_14, %787 : tensor<1x256x1280xf32>
    %801 = stablehlo.add %800, %cst_15 : tensor<1x256x1280xf32>
    %802 = stablehlo.multiply %801, %787 : tensor<1x256x1280xf32>
    %803 = stablehlo.add %802, %cst_16 : tensor<1x256x1280xf32>
    %804 = stablehlo.multiply %803, %787 : tensor<1x256x1280xf32>
    %805 = stablehlo.add %804, %cst_17 : tensor<1x256x1280xf32>
    %806 = stablehlo.multiply %805, %787 : tensor<1x256x1280xf32>
    %807 = stablehlo.add %806, %cst_18 : tensor<1x256x1280xf32>
    %808 = stablehlo.multiply %786, %799 : tensor<1x256x1280xf32>
    %809 = stablehlo.divide %808, %807 : tensor<1x256x1280xf32>
    %810 = stablehlo.clamp %cst_19, %809, %cst_20 : tensor<1x256x1280xf32>
    %811 = stablehlo.convert %810 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %812 = stablehlo.add %811, %cst_2 : tensor<1x256x1280xbf16>
    %813 = stablehlo.multiply %812, %783 : tensor<1x256x1280xbf16>
    %814 = stablehlo.reshape %813 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %815 = stablehlo.convert %814 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %816 = stablehlo.dot_general %815, %arg163, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %817 = stablehlo.broadcast_in_dim %816, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %818 = stablehlo.multiply %817, %127 : tensor<256x1280xf32>
    %819 = stablehlo.broadcast_in_dim %818, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %820 = stablehlo.broadcast_in_dim %arg164, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %821 = stablehlo.add %819, %820 : tensor<256x1280xf32>
    %822 = stablehlo.convert %821 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %823 = stablehlo.reshape %822 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %824 = stablehlo.add %823, %735 : tensor<1x256x1280xbf16>
    %825 = stablehlo.convert %824 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %826 = stablehlo.convert %825 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %827 = stablehlo.reduce(%826 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %828 = stablehlo.reshape %827 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %830 = stablehlo.divide %829, %142 : tensor<1x256x1xf64>
    %831 = stablehlo.broadcast_in_dim %826, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %832 = stablehlo.broadcast_in_dim %830, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %833 = stablehlo.subtract %831, %832 : tensor<1x256x1280xf64>
    %834 = stablehlo.multiply %833, %833 : tensor<1x256x1280xf64>
    %835 = stablehlo.reduce(%834 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %836 = stablehlo.reshape %835 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %838 = stablehlo.divide %837, %142 : tensor<1x256x1xf64>
    %839 = stablehlo.convert %838 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %840 = stablehlo.reduce(%825 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %841 = stablehlo.reshape %840 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %842 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %843 = stablehlo.divide %842, %158 : tensor<1x256x1xf32>
    %844 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %845 = stablehlo.add %844, %161 : tensor<1x256x1xf32>
    %846 = stablehlo.rsqrt %845 : tensor<1x256x1xf32>
    %847 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %848 = stablehlo.broadcast_in_dim %843, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %849 = stablehlo.subtract %847, %848 : tensor<1x256x1280xf32>
    %850 = stablehlo.broadcast_in_dim %849, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %851 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %852 = stablehlo.multiply %850, %851 : tensor<1x256x1280xf32>
    %853 = stablehlo.convert %arg19 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %854 = stablehlo.broadcast_in_dim %852, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %855 = stablehlo.broadcast_in_dim %853, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %856 = stablehlo.multiply %854, %855 : tensor<1x256x1280xf32>
    %857 = stablehlo.convert %arg20 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %858 = stablehlo.broadcast_in_dim %856, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %859 = stablehlo.broadcast_in_dim %857, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %860 = stablehlo.add %858, %859 : tensor<1x256x1280xf32>
    %861 = stablehlo.convert %860 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %862 = stablehlo.reshape %861 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %863 = stablehlo.convert %862 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %864 = stablehlo.dot_general %863, %arg165, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %865 = stablehlo.broadcast_in_dim %864, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %866 = stablehlo.multiply %865, %273 : tensor<256x256xf32>
    %867 = stablehlo.broadcast_in_dim %866, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %868 = stablehlo.broadcast_in_dim %arg166, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %869 = stablehlo.add %867, %868 : tensor<256x256xf32>
    %870 = stablehlo.convert %869 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %871 = stablehlo.reshape %870 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %872 = stablehlo.dot_general %863, %arg167, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %873 = stablehlo.broadcast_in_dim %872, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %874 = stablehlo.multiply %873, %273 : tensor<256x256xf32>
    %875 = stablehlo.broadcast_in_dim %874, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %876 = stablehlo.broadcast_in_dim %arg168, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %877 = stablehlo.add %875, %876 : tensor<256x256xf32>
    %878 = stablehlo.convert %877 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %879 = stablehlo.reshape %878 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %880 = stablehlo.dot_general %863, %arg169, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %882 = stablehlo.multiply %881, %127 : tensor<256x1280xf32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %884 = stablehlo.broadcast_in_dim %arg170, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %885 = stablehlo.add %883, %884 : tensor<256x1280xf32>
    %886 = stablehlo.convert %885 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %887 = stablehlo.reshape %886 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %888 = stablehlo.reshape %871 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %889 = stablehlo.transpose %888, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %890 = stablehlo.reshape %879 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %891 = stablehlo.transpose %890, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %892 = stablehlo.reshape %887 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %893 = stablehlo.transpose %892, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %894 = stablehlo.transpose %891, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %895 = stablehlo.reshape %889 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %896 = stablehlo.reshape %894 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %897 = stablehlo.broadcast_in_dim %896, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %898 = stablehlo.dot_general %895, %897, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %899 = stablehlo.reshape %898 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %900 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %901 = stablehlo.divide %900, %309 : tensor<1x8x256x256xbf16>
    %902 = stablehlo.convert %901 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %903 = stablehlo.reduce(%902 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %904 = stablehlo.reshape %903 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %905 = stablehlo.broadcast_in_dim %902, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %906 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %907 = stablehlo.subtract %905, %906 : tensor<1x8x256x256xf32>
    %908 = stablehlo.exponential %907 : tensor<1x8x256x256xf32>
    %909 = stablehlo.reduce(%908 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %910 = stablehlo.reshape %909 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %911 = stablehlo.broadcast_in_dim %908, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %912 = stablehlo.broadcast_in_dim %910, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %913 = stablehlo.divide %911, %912 : tensor<1x8x256x256xf32>
    %914 = stablehlo.convert %913 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %915 = stablehlo.reshape %914 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %916 = stablehlo.reshape %893 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %917 = stablehlo.broadcast_in_dim %916, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %918 = stablehlo.dot_general %915, %917, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %919 = stablehlo.reshape %918 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %920 = stablehlo.transpose %919, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %921 = stablehlo.reshape %920 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %922 = stablehlo.reshape %921 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %923 = stablehlo.convert %922 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %924 = stablehlo.dot_general %923, %arg171, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %926 = stablehlo.multiply %925, %127 : tensor<256x1280xf32>
    %927 = stablehlo.broadcast_in_dim %926, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %928 = stablehlo.broadcast_in_dim %arg172, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %929 = stablehlo.add %927, %928 : tensor<256x1280xf32>
    %930 = stablehlo.convert %929 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %931 = stablehlo.reshape %930 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %932 = stablehlo.add %931, %824 : tensor<1x256x1280xbf16>
    %933 = stablehlo.convert %932 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %934 = stablehlo.convert %933 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %935 = stablehlo.reduce(%934 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %936 = stablehlo.reshape %935 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %937 = stablehlo.broadcast_in_dim %936, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %938 = stablehlo.divide %937, %142 : tensor<1x256x1xf64>
    %939 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %940 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %941 = stablehlo.subtract %939, %940 : tensor<1x256x1280xf64>
    %942 = stablehlo.multiply %941, %941 : tensor<1x256x1280xf64>
    %943 = stablehlo.reduce(%942 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %944 = stablehlo.reshape %943 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %945 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %946 = stablehlo.divide %945, %142 : tensor<1x256x1xf64>
    %947 = stablehlo.convert %946 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %948 = stablehlo.reduce(%933 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %949 = stablehlo.reshape %948 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %950 = stablehlo.broadcast_in_dim %949, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %951 = stablehlo.divide %950, %158 : tensor<1x256x1xf32>
    %952 = stablehlo.broadcast_in_dim %947, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %953 = stablehlo.add %952, %161 : tensor<1x256x1xf32>
    %954 = stablehlo.rsqrt %953 : tensor<1x256x1xf32>
    %955 = stablehlo.broadcast_in_dim %933, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %956 = stablehlo.broadcast_in_dim %951, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %957 = stablehlo.subtract %955, %956 : tensor<1x256x1280xf32>
    %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %959 = stablehlo.broadcast_in_dim %954, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %960 = stablehlo.multiply %958, %959 : tensor<1x256x1280xf32>
    %961 = stablehlo.convert %arg21 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %962 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %963 = stablehlo.broadcast_in_dim %961, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %964 = stablehlo.multiply %962, %963 : tensor<1x256x1280xf32>
    %965 = stablehlo.convert %arg22 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %966 = stablehlo.broadcast_in_dim %964, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %967 = stablehlo.broadcast_in_dim %965, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %968 = stablehlo.add %966, %967 : tensor<1x256x1280xf32>
    %969 = stablehlo.convert %968 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %970 = stablehlo.reshape %969 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %971 = stablehlo.convert %970 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %972 = stablehlo.dot_general %971, %arg173, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %973 = stablehlo.broadcast_in_dim %972, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %974 = stablehlo.multiply %973, %127 : tensor<256x1280xf32>
    %975 = stablehlo.broadcast_in_dim %974, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %976 = stablehlo.broadcast_in_dim %arg174, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %977 = stablehlo.add %975, %976 : tensor<256x1280xf32>
    %978 = stablehlo.convert %977 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %979 = stablehlo.reshape %978 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %980 = stablehlo.multiply %979, %cst_4 : tensor<1x256x1280xbf16>
    %981 = stablehlo.multiply %979, %190 : tensor<1x256x1280xbf16>
    %982 = stablehlo.convert %981 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %983 = stablehlo.clamp %cst_5, %982, %cst_6 : tensor<1x256x1280xf32>
    %984 = stablehlo.multiply %983, %983 : tensor<1x256x1280xf32>
    %985 = stablehlo.multiply %cst_7, %984 : tensor<1x256x1280xf32>
    %986 = stablehlo.add %985, %cst_8 : tensor<1x256x1280xf32>
    %987 = stablehlo.multiply %986, %984 : tensor<1x256x1280xf32>
    %988 = stablehlo.add %987, %cst_9 : tensor<1x256x1280xf32>
    %989 = stablehlo.multiply %988, %984 : tensor<1x256x1280xf32>
    %990 = stablehlo.add %989, %cst_10 : tensor<1x256x1280xf32>
    %991 = stablehlo.multiply %990, %984 : tensor<1x256x1280xf32>
    %992 = stablehlo.add %991, %cst_11 : tensor<1x256x1280xf32>
    %993 = stablehlo.multiply %992, %984 : tensor<1x256x1280xf32>
    %994 = stablehlo.add %993, %cst_12 : tensor<1x256x1280xf32>
    %995 = stablehlo.multiply %994, %984 : tensor<1x256x1280xf32>
    %996 = stablehlo.add %995, %cst_13 : tensor<1x256x1280xf32>
    %997 = stablehlo.multiply %cst_14, %984 : tensor<1x256x1280xf32>
    %998 = stablehlo.add %997, %cst_15 : tensor<1x256x1280xf32>
    %999 = stablehlo.multiply %998, %984 : tensor<1x256x1280xf32>
    %1000 = stablehlo.add %999, %cst_16 : tensor<1x256x1280xf32>
    %1001 = stablehlo.multiply %1000, %984 : tensor<1x256x1280xf32>
    %1002 = stablehlo.add %1001, %cst_17 : tensor<1x256x1280xf32>
    %1003 = stablehlo.multiply %1002, %984 : tensor<1x256x1280xf32>
    %1004 = stablehlo.add %1003, %cst_18 : tensor<1x256x1280xf32>
    %1005 = stablehlo.multiply %983, %996 : tensor<1x256x1280xf32>
    %1006 = stablehlo.divide %1005, %1004 : tensor<1x256x1280xf32>
    %1007 = stablehlo.clamp %cst_19, %1006, %cst_20 : tensor<1x256x1280xf32>
    %1008 = stablehlo.convert %1007 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1009 = stablehlo.add %1008, %cst_2 : tensor<1x256x1280xbf16>
    %1010 = stablehlo.multiply %1009, %980 : tensor<1x256x1280xbf16>
    %1011 = stablehlo.reshape %1010 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1012 = stablehlo.convert %1011 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1013 = stablehlo.dot_general %1012, %arg175, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1014 = stablehlo.broadcast_in_dim %1013, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1015 = stablehlo.multiply %1014, %127 : tensor<256x1280xf32>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1017 = stablehlo.broadcast_in_dim %arg176, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1018 = stablehlo.add %1016, %1017 : tensor<256x1280xf32>
    %1019 = stablehlo.convert %1018 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1020 = stablehlo.reshape %1019 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1021 = stablehlo.add %1020, %932 : tensor<1x256x1280xbf16>
    %1022 = stablehlo.convert %1021 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1023 = stablehlo.convert %1022 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1024 = stablehlo.reduce(%1023 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1025 = stablehlo.reshape %1024 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1026 = stablehlo.broadcast_in_dim %1025, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1027 = stablehlo.divide %1026, %142 : tensor<1x256x1xf64>
    %1028 = stablehlo.broadcast_in_dim %1023, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1029 = stablehlo.broadcast_in_dim %1027, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1030 = stablehlo.subtract %1028, %1029 : tensor<1x256x1280xf64>
    %1031 = stablehlo.multiply %1030, %1030 : tensor<1x256x1280xf64>
    %1032 = stablehlo.reduce(%1031 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1033 = stablehlo.reshape %1032 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1035 = stablehlo.divide %1034, %142 : tensor<1x256x1xf64>
    %1036 = stablehlo.convert %1035 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1037 = stablehlo.reduce(%1022 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1038 = stablehlo.reshape %1037 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1039 = stablehlo.broadcast_in_dim %1038, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1040 = stablehlo.divide %1039, %158 : tensor<1x256x1xf32>
    %1041 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1042 = stablehlo.add %1041, %161 : tensor<1x256x1xf32>
    %1043 = stablehlo.rsqrt %1042 : tensor<1x256x1xf32>
    %1044 = stablehlo.broadcast_in_dim %1022, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1045 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1046 = stablehlo.subtract %1044, %1045 : tensor<1x256x1280xf32>
    %1047 = stablehlo.broadcast_in_dim %1046, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1048 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1049 = stablehlo.multiply %1047, %1048 : tensor<1x256x1280xf32>
    %1050 = stablehlo.convert %arg23 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1051 = stablehlo.broadcast_in_dim %1049, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1052 = stablehlo.broadcast_in_dim %1050, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1053 = stablehlo.multiply %1051, %1052 : tensor<1x256x1280xf32>
    %1054 = stablehlo.convert %arg24 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1055 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1056 = stablehlo.broadcast_in_dim %1054, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1057 = stablehlo.add %1055, %1056 : tensor<1x256x1280xf32>
    %1058 = stablehlo.convert %1057 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1059 = stablehlo.reshape %1058 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1060 = stablehlo.convert %1059 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1061 = stablehlo.dot_general %1060, %arg177, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1062 = stablehlo.broadcast_in_dim %1061, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1063 = stablehlo.multiply %1062, %273 : tensor<256x256xf32>
    %1064 = stablehlo.broadcast_in_dim %1063, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1065 = stablehlo.broadcast_in_dim %arg178, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1066 = stablehlo.add %1064, %1065 : tensor<256x256xf32>
    %1067 = stablehlo.convert %1066 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1068 = stablehlo.reshape %1067 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1069 = stablehlo.dot_general %1060, %arg179, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1070 = stablehlo.broadcast_in_dim %1069, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1071 = stablehlo.multiply %1070, %273 : tensor<256x256xf32>
    %1072 = stablehlo.broadcast_in_dim %1071, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1073 = stablehlo.broadcast_in_dim %arg180, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1074 = stablehlo.add %1072, %1073 : tensor<256x256xf32>
    %1075 = stablehlo.convert %1074 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1076 = stablehlo.reshape %1075 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1077 = stablehlo.dot_general %1060, %arg181, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1079 = stablehlo.multiply %1078, %127 : tensor<256x1280xf32>
    %1080 = stablehlo.broadcast_in_dim %1079, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1081 = stablehlo.broadcast_in_dim %arg182, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1082 = stablehlo.add %1080, %1081 : tensor<256x1280xf32>
    %1083 = stablehlo.convert %1082 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1084 = stablehlo.reshape %1083 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1085 = stablehlo.reshape %1068 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1086 = stablehlo.transpose %1085, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1087 = stablehlo.reshape %1076 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1088 = stablehlo.transpose %1087, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1089 = stablehlo.reshape %1084 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %1090 = stablehlo.transpose %1089, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1091 = stablehlo.transpose %1088, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %1092 = stablehlo.reshape %1086 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1093 = stablehlo.reshape %1091 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1094 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1095 = stablehlo.dot_general %1092, %1094, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %1096 = stablehlo.reshape %1095 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1097 = stablehlo.broadcast_in_dim %1096, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1098 = stablehlo.divide %1097, %309 : tensor<1x8x256x256xbf16>
    %1099 = stablehlo.convert %1098 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %1100 = stablehlo.reduce(%1099 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1101 = stablehlo.reshape %1100 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1102 = stablehlo.broadcast_in_dim %1099, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1103 = stablehlo.broadcast_in_dim %1101, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1104 = stablehlo.subtract %1102, %1103 : tensor<1x8x256x256xf32>
    %1105 = stablehlo.exponential %1104 : tensor<1x8x256x256xf32>
    %1106 = stablehlo.reduce(%1105 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1107 = stablehlo.reshape %1106 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1108 = stablehlo.broadcast_in_dim %1105, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1109 = stablehlo.broadcast_in_dim %1107, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1110 = stablehlo.divide %1108, %1109 : tensor<1x8x256x256xf32>
    %1111 = stablehlo.convert %1110 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %1112 = stablehlo.reshape %1111 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %1113 = stablehlo.reshape %1090 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1114 = stablehlo.broadcast_in_dim %1113, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1115 = stablehlo.dot_general %1112, %1114, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1116 = stablehlo.reshape %1115 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1117 = stablehlo.transpose %1116, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %1118 = stablehlo.reshape %1117 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %1119 = stablehlo.reshape %1118 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1120 = stablehlo.convert %1119 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1121 = stablehlo.dot_general %1120, %arg183, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1122 = stablehlo.broadcast_in_dim %1121, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1123 = stablehlo.multiply %1122, %127 : tensor<256x1280xf32>
    %1124 = stablehlo.broadcast_in_dim %1123, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1125 = stablehlo.broadcast_in_dim %arg184, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1126 = stablehlo.add %1124, %1125 : tensor<256x1280xf32>
    %1127 = stablehlo.convert %1126 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1128 = stablehlo.reshape %1127 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1129 = stablehlo.add %1128, %1021 : tensor<1x256x1280xbf16>
    %1130 = stablehlo.convert %1129 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1131 = stablehlo.convert %1130 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1132 = stablehlo.reduce(%1131 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1133 = stablehlo.reshape %1132 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1134 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1135 = stablehlo.divide %1134, %142 : tensor<1x256x1xf64>
    %1136 = stablehlo.broadcast_in_dim %1131, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1137 = stablehlo.broadcast_in_dim %1135, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1138 = stablehlo.subtract %1136, %1137 : tensor<1x256x1280xf64>
    %1139 = stablehlo.multiply %1138, %1138 : tensor<1x256x1280xf64>
    %1140 = stablehlo.reduce(%1139 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1141 = stablehlo.reshape %1140 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1142 = stablehlo.broadcast_in_dim %1141, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1143 = stablehlo.divide %1142, %142 : tensor<1x256x1xf64>
    %1144 = stablehlo.convert %1143 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1145 = stablehlo.reduce(%1130 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1146 = stablehlo.reshape %1145 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1147 = stablehlo.broadcast_in_dim %1146, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1148 = stablehlo.divide %1147, %158 : tensor<1x256x1xf32>
    %1149 = stablehlo.broadcast_in_dim %1144, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1150 = stablehlo.add %1149, %161 : tensor<1x256x1xf32>
    %1151 = stablehlo.rsqrt %1150 : tensor<1x256x1xf32>
    %1152 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1153 = stablehlo.broadcast_in_dim %1148, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1154 = stablehlo.subtract %1152, %1153 : tensor<1x256x1280xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1156 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1157 = stablehlo.multiply %1155, %1156 : tensor<1x256x1280xf32>
    %1158 = stablehlo.convert %arg25 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1159 = stablehlo.broadcast_in_dim %1157, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1160 = stablehlo.broadcast_in_dim %1158, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1161 = stablehlo.multiply %1159, %1160 : tensor<1x256x1280xf32>
    %1162 = stablehlo.convert %arg26 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1163 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1164 = stablehlo.broadcast_in_dim %1162, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1165 = stablehlo.add %1163, %1164 : tensor<1x256x1280xf32>
    %1166 = stablehlo.convert %1165 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1167 = stablehlo.reshape %1166 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1168 = stablehlo.convert %1167 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1169 = stablehlo.dot_general %1168, %arg185, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1170 = stablehlo.broadcast_in_dim %1169, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1171 = stablehlo.multiply %1170, %127 : tensor<256x1280xf32>
    %1172 = stablehlo.broadcast_in_dim %1171, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1173 = stablehlo.broadcast_in_dim %arg186, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1174 = stablehlo.add %1172, %1173 : tensor<256x1280xf32>
    %1175 = stablehlo.convert %1174 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1176 = stablehlo.reshape %1175 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1177 = stablehlo.multiply %1176, %cst_4 : tensor<1x256x1280xbf16>
    %1178 = stablehlo.multiply %1176, %190 : tensor<1x256x1280xbf16>
    %1179 = stablehlo.convert %1178 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1180 = stablehlo.clamp %cst_5, %1179, %cst_6 : tensor<1x256x1280xf32>
    %1181 = stablehlo.multiply %1180, %1180 : tensor<1x256x1280xf32>
    %1182 = stablehlo.multiply %cst_7, %1181 : tensor<1x256x1280xf32>
    %1183 = stablehlo.add %1182, %cst_8 : tensor<1x256x1280xf32>
    %1184 = stablehlo.multiply %1183, %1181 : tensor<1x256x1280xf32>
    %1185 = stablehlo.add %1184, %cst_9 : tensor<1x256x1280xf32>
    %1186 = stablehlo.multiply %1185, %1181 : tensor<1x256x1280xf32>
    %1187 = stablehlo.add %1186, %cst_10 : tensor<1x256x1280xf32>
    %1188 = stablehlo.multiply %1187, %1181 : tensor<1x256x1280xf32>
    %1189 = stablehlo.add %1188, %cst_11 : tensor<1x256x1280xf32>
    %1190 = stablehlo.multiply %1189, %1181 : tensor<1x256x1280xf32>
    %1191 = stablehlo.add %1190, %cst_12 : tensor<1x256x1280xf32>
    %1192 = stablehlo.multiply %1191, %1181 : tensor<1x256x1280xf32>
    %1193 = stablehlo.add %1192, %cst_13 : tensor<1x256x1280xf32>
    %1194 = stablehlo.multiply %cst_14, %1181 : tensor<1x256x1280xf32>
    %1195 = stablehlo.add %1194, %cst_15 : tensor<1x256x1280xf32>
    %1196 = stablehlo.multiply %1195, %1181 : tensor<1x256x1280xf32>
    %1197 = stablehlo.add %1196, %cst_16 : tensor<1x256x1280xf32>
    %1198 = stablehlo.multiply %1197, %1181 : tensor<1x256x1280xf32>
    %1199 = stablehlo.add %1198, %cst_17 : tensor<1x256x1280xf32>
    %1200 = stablehlo.multiply %1199, %1181 : tensor<1x256x1280xf32>
    %1201 = stablehlo.add %1200, %cst_18 : tensor<1x256x1280xf32>
    %1202 = stablehlo.multiply %1180, %1193 : tensor<1x256x1280xf32>
    %1203 = stablehlo.divide %1202, %1201 : tensor<1x256x1280xf32>
    %1204 = stablehlo.clamp %cst_19, %1203, %cst_20 : tensor<1x256x1280xf32>
    %1205 = stablehlo.convert %1204 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1206 = stablehlo.add %1205, %cst_2 : tensor<1x256x1280xbf16>
    %1207 = stablehlo.multiply %1206, %1177 : tensor<1x256x1280xbf16>
    %1208 = stablehlo.reshape %1207 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1209 = stablehlo.convert %1208 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1210 = stablehlo.dot_general %1209, %arg187, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1212 = stablehlo.multiply %1211, %127 : tensor<256x1280xf32>
    %1213 = stablehlo.broadcast_in_dim %1212, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1214 = stablehlo.broadcast_in_dim %arg188, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1215 = stablehlo.add %1213, %1214 : tensor<256x1280xf32>
    %1216 = stablehlo.convert %1215 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1217 = stablehlo.reshape %1216 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1218 = stablehlo.add %1217, %1129 : tensor<1x256x1280xbf16>
    %1219 = stablehlo.convert %1218 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1220 = stablehlo.convert %1219 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1221 = stablehlo.reduce(%1220 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1222 = stablehlo.reshape %1221 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1224 = stablehlo.divide %1223, %142 : tensor<1x256x1xf64>
    %1225 = stablehlo.broadcast_in_dim %1220, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1226 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1227 = stablehlo.subtract %1225, %1226 : tensor<1x256x1280xf64>
    %1228 = stablehlo.multiply %1227, %1227 : tensor<1x256x1280xf64>
    %1229 = stablehlo.reduce(%1228 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1230 = stablehlo.reshape %1229 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1231 = stablehlo.broadcast_in_dim %1230, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1232 = stablehlo.divide %1231, %142 : tensor<1x256x1xf64>
    %1233 = stablehlo.convert %1232 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1234 = stablehlo.reduce(%1219 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1235 = stablehlo.reshape %1234 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1237 = stablehlo.divide %1236, %158 : tensor<1x256x1xf32>
    %1238 = stablehlo.broadcast_in_dim %1233, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1239 = stablehlo.add %1238, %161 : tensor<1x256x1xf32>
    %1240 = stablehlo.rsqrt %1239 : tensor<1x256x1xf32>
    %1241 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1242 = stablehlo.broadcast_in_dim %1237, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1243 = stablehlo.subtract %1241, %1242 : tensor<1x256x1280xf32>
    %1244 = stablehlo.broadcast_in_dim %1243, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1245 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1246 = stablehlo.multiply %1244, %1245 : tensor<1x256x1280xf32>
    %1247 = stablehlo.convert %arg27 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1248 = stablehlo.broadcast_in_dim %1246, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1249 = stablehlo.broadcast_in_dim %1247, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1250 = stablehlo.multiply %1248, %1249 : tensor<1x256x1280xf32>
    %1251 = stablehlo.convert %arg28 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1252 = stablehlo.broadcast_in_dim %1250, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1253 = stablehlo.broadcast_in_dim %1251, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1254 = stablehlo.add %1252, %1253 : tensor<1x256x1280xf32>
    %1255 = stablehlo.convert %1254 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1256 = stablehlo.reshape %1255 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1257 = stablehlo.convert %1256 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1258 = stablehlo.dot_general %1257, %arg189, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1259 = stablehlo.broadcast_in_dim %1258, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1260 = stablehlo.multiply %1259, %273 : tensor<256x256xf32>
    %1261 = stablehlo.broadcast_in_dim %1260, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1262 = stablehlo.broadcast_in_dim %arg190, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1263 = stablehlo.add %1261, %1262 : tensor<256x256xf32>
    %1264 = stablehlo.convert %1263 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1265 = stablehlo.reshape %1264 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1266 = stablehlo.dot_general %1257, %arg191, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1267 = stablehlo.broadcast_in_dim %1266, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1268 = stablehlo.multiply %1267, %273 : tensor<256x256xf32>
    %1269 = stablehlo.broadcast_in_dim %1268, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1270 = stablehlo.broadcast_in_dim %arg192, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1271 = stablehlo.add %1269, %1270 : tensor<256x256xf32>
    %1272 = stablehlo.convert %1271 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1273 = stablehlo.reshape %1272 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1274 = stablehlo.dot_general %1257, %arg193, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1275 = stablehlo.broadcast_in_dim %1274, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1276 = stablehlo.multiply %1275, %127 : tensor<256x1280xf32>
    %1277 = stablehlo.broadcast_in_dim %1276, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1278 = stablehlo.broadcast_in_dim %arg194, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1279 = stablehlo.add %1277, %1278 : tensor<256x1280xf32>
    %1280 = stablehlo.convert %1279 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1281 = stablehlo.reshape %1280 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1282 = stablehlo.reshape %1265 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1283 = stablehlo.transpose %1282, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1284 = stablehlo.reshape %1273 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1285 = stablehlo.transpose %1284, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1286 = stablehlo.reshape %1281 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %1287 = stablehlo.transpose %1286, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1288 = stablehlo.transpose %1285, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %1289 = stablehlo.reshape %1283 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1290 = stablehlo.reshape %1288 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1291 = stablehlo.broadcast_in_dim %1290, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1292 = stablehlo.dot_general %1289, %1291, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %1293 = stablehlo.reshape %1292 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1294 = stablehlo.broadcast_in_dim %1293, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1295 = stablehlo.divide %1294, %309 : tensor<1x8x256x256xbf16>
    %1296 = stablehlo.convert %1295 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %1297 = stablehlo.reduce(%1296 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1298 = stablehlo.reshape %1297 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1299 = stablehlo.broadcast_in_dim %1296, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1300 = stablehlo.broadcast_in_dim %1298, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1301 = stablehlo.subtract %1299, %1300 : tensor<1x8x256x256xf32>
    %1302 = stablehlo.exponential %1301 : tensor<1x8x256x256xf32>
    %1303 = stablehlo.reduce(%1302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1304 = stablehlo.reshape %1303 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1305 = stablehlo.broadcast_in_dim %1302, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1306 = stablehlo.broadcast_in_dim %1304, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1307 = stablehlo.divide %1305, %1306 : tensor<1x8x256x256xf32>
    %1308 = stablehlo.convert %1307 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %1309 = stablehlo.reshape %1308 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %1310 = stablehlo.reshape %1287 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1311 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1312 = stablehlo.dot_general %1309, %1311, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1313 = stablehlo.reshape %1312 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1314 = stablehlo.transpose %1313, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %1315 = stablehlo.reshape %1314 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %1316 = stablehlo.reshape %1315 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1317 = stablehlo.convert %1316 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1318 = stablehlo.dot_general %1317, %arg195, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1319 = stablehlo.broadcast_in_dim %1318, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1320 = stablehlo.multiply %1319, %127 : tensor<256x1280xf32>
    %1321 = stablehlo.broadcast_in_dim %1320, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1322 = stablehlo.broadcast_in_dim %arg196, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1323 = stablehlo.add %1321, %1322 : tensor<256x1280xf32>
    %1324 = stablehlo.convert %1323 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1325 = stablehlo.reshape %1324 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1326 = stablehlo.add %1325, %1218 : tensor<1x256x1280xbf16>
    %1327 = stablehlo.convert %1326 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1328 = stablehlo.convert %1327 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1329 = stablehlo.reduce(%1328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1330 = stablehlo.reshape %1329 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1332 = stablehlo.divide %1331, %142 : tensor<1x256x1xf64>
    %1333 = stablehlo.broadcast_in_dim %1328, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1334 = stablehlo.broadcast_in_dim %1332, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1335 = stablehlo.subtract %1333, %1334 : tensor<1x256x1280xf64>
    %1336 = stablehlo.multiply %1335, %1335 : tensor<1x256x1280xf64>
    %1337 = stablehlo.reduce(%1336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1338 = stablehlo.reshape %1337 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1340 = stablehlo.divide %1339, %142 : tensor<1x256x1xf64>
    %1341 = stablehlo.convert %1340 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1342 = stablehlo.reduce(%1327 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1343 = stablehlo.reshape %1342 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1344 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1345 = stablehlo.divide %1344, %158 : tensor<1x256x1xf32>
    %1346 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1347 = stablehlo.add %1346, %161 : tensor<1x256x1xf32>
    %1348 = stablehlo.rsqrt %1347 : tensor<1x256x1xf32>
    %1349 = stablehlo.broadcast_in_dim %1327, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1350 = stablehlo.broadcast_in_dim %1345, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1351 = stablehlo.subtract %1349, %1350 : tensor<1x256x1280xf32>
    %1352 = stablehlo.broadcast_in_dim %1351, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1353 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1354 = stablehlo.multiply %1352, %1353 : tensor<1x256x1280xf32>
    %1355 = stablehlo.convert %arg29 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1356 = stablehlo.broadcast_in_dim %1354, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1357 = stablehlo.broadcast_in_dim %1355, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1358 = stablehlo.multiply %1356, %1357 : tensor<1x256x1280xf32>
    %1359 = stablehlo.convert %arg30 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1360 = stablehlo.broadcast_in_dim %1358, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1361 = stablehlo.broadcast_in_dim %1359, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1362 = stablehlo.add %1360, %1361 : tensor<1x256x1280xf32>
    %1363 = stablehlo.convert %1362 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1364 = stablehlo.reshape %1363 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1365 = stablehlo.convert %1364 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1366 = stablehlo.dot_general %1365, %arg197, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1367 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1368 = stablehlo.multiply %1367, %127 : tensor<256x1280xf32>
    %1369 = stablehlo.broadcast_in_dim %1368, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1370 = stablehlo.broadcast_in_dim %arg198, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1371 = stablehlo.add %1369, %1370 : tensor<256x1280xf32>
    %1372 = stablehlo.convert %1371 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1373 = stablehlo.reshape %1372 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1374 = stablehlo.multiply %1373, %cst_4 : tensor<1x256x1280xbf16>
    %1375 = stablehlo.multiply %1373, %190 : tensor<1x256x1280xbf16>
    %1376 = stablehlo.convert %1375 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1377 = stablehlo.clamp %cst_5, %1376, %cst_6 : tensor<1x256x1280xf32>
    %1378 = stablehlo.multiply %1377, %1377 : tensor<1x256x1280xf32>
    %1379 = stablehlo.multiply %cst_7, %1378 : tensor<1x256x1280xf32>
    %1380 = stablehlo.add %1379, %cst_8 : tensor<1x256x1280xf32>
    %1381 = stablehlo.multiply %1380, %1378 : tensor<1x256x1280xf32>
    %1382 = stablehlo.add %1381, %cst_9 : tensor<1x256x1280xf32>
    %1383 = stablehlo.multiply %1382, %1378 : tensor<1x256x1280xf32>
    %1384 = stablehlo.add %1383, %cst_10 : tensor<1x256x1280xf32>
    %1385 = stablehlo.multiply %1384, %1378 : tensor<1x256x1280xf32>
    %1386 = stablehlo.add %1385, %cst_11 : tensor<1x256x1280xf32>
    %1387 = stablehlo.multiply %1386, %1378 : tensor<1x256x1280xf32>
    %1388 = stablehlo.add %1387, %cst_12 : tensor<1x256x1280xf32>
    %1389 = stablehlo.multiply %1388, %1378 : tensor<1x256x1280xf32>
    %1390 = stablehlo.add %1389, %cst_13 : tensor<1x256x1280xf32>
    %1391 = stablehlo.multiply %cst_14, %1378 : tensor<1x256x1280xf32>
    %1392 = stablehlo.add %1391, %cst_15 : tensor<1x256x1280xf32>
    %1393 = stablehlo.multiply %1392, %1378 : tensor<1x256x1280xf32>
    %1394 = stablehlo.add %1393, %cst_16 : tensor<1x256x1280xf32>
    %1395 = stablehlo.multiply %1394, %1378 : tensor<1x256x1280xf32>
    %1396 = stablehlo.add %1395, %cst_17 : tensor<1x256x1280xf32>
    %1397 = stablehlo.multiply %1396, %1378 : tensor<1x256x1280xf32>
    %1398 = stablehlo.add %1397, %cst_18 : tensor<1x256x1280xf32>
    %1399 = stablehlo.multiply %1377, %1390 : tensor<1x256x1280xf32>
    %1400 = stablehlo.divide %1399, %1398 : tensor<1x256x1280xf32>
    %1401 = stablehlo.clamp %cst_19, %1400, %cst_20 : tensor<1x256x1280xf32>
    %1402 = stablehlo.convert %1401 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1403 = stablehlo.add %1402, %cst_2 : tensor<1x256x1280xbf16>
    %1404 = stablehlo.multiply %1403, %1374 : tensor<1x256x1280xbf16>
    %1405 = stablehlo.reshape %1404 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1406 = stablehlo.convert %1405 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1407 = stablehlo.dot_general %1406, %arg199, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1408 = stablehlo.broadcast_in_dim %1407, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1409 = stablehlo.multiply %1408, %127 : tensor<256x1280xf32>
    %1410 = stablehlo.broadcast_in_dim %1409, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1411 = stablehlo.broadcast_in_dim %arg200, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1412 = stablehlo.add %1410, %1411 : tensor<256x1280xf32>
    %1413 = stablehlo.convert %1412 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1414 = stablehlo.reshape %1413 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1415 = stablehlo.add %1414, %1326 : tensor<1x256x1280xbf16>
    %1416 = stablehlo.convert %1415 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1417 = stablehlo.convert %1416 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1418 = stablehlo.reduce(%1417 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1419 = stablehlo.reshape %1418 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1420 = stablehlo.broadcast_in_dim %1419, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1421 = stablehlo.divide %1420, %142 : tensor<1x256x1xf64>
    %1422 = stablehlo.broadcast_in_dim %1417, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1423 = stablehlo.broadcast_in_dim %1421, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1424 = stablehlo.subtract %1422, %1423 : tensor<1x256x1280xf64>
    %1425 = stablehlo.multiply %1424, %1424 : tensor<1x256x1280xf64>
    %1426 = stablehlo.reduce(%1425 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1427 = stablehlo.reshape %1426 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1429 = stablehlo.divide %1428, %142 : tensor<1x256x1xf64>
    %1430 = stablehlo.convert %1429 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1431 = stablehlo.reduce(%1416 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1432 = stablehlo.reshape %1431 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1434 = stablehlo.divide %1433, %158 : tensor<1x256x1xf32>
    %1435 = stablehlo.broadcast_in_dim %1430, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1436 = stablehlo.add %1435, %161 : tensor<1x256x1xf32>
    %1437 = stablehlo.rsqrt %1436 : tensor<1x256x1xf32>
    %1438 = stablehlo.broadcast_in_dim %1416, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1439 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1440 = stablehlo.subtract %1438, %1439 : tensor<1x256x1280xf32>
    %1441 = stablehlo.broadcast_in_dim %1440, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1442 = stablehlo.broadcast_in_dim %1437, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1443 = stablehlo.multiply %1441, %1442 : tensor<1x256x1280xf32>
    %1444 = stablehlo.convert %arg31 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1445 = stablehlo.broadcast_in_dim %1443, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1446 = stablehlo.broadcast_in_dim %1444, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1447 = stablehlo.multiply %1445, %1446 : tensor<1x256x1280xf32>
    %1448 = stablehlo.convert %arg32 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1449 = stablehlo.broadcast_in_dim %1447, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1450 = stablehlo.broadcast_in_dim %1448, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1451 = stablehlo.add %1449, %1450 : tensor<1x256x1280xf32>
    %1452 = stablehlo.convert %1451 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1453 = stablehlo.reshape %1452 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1454 = stablehlo.convert %1453 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1455 = stablehlo.dot_general %1454, %arg201, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1456 = stablehlo.broadcast_in_dim %1455, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1457 = stablehlo.multiply %1456, %273 : tensor<256x256xf32>
    %1458 = stablehlo.broadcast_in_dim %1457, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1459 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1460 = stablehlo.add %1458, %1459 : tensor<256x256xf32>
    %1461 = stablehlo.convert %1460 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1462 = stablehlo.reshape %1461 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1463 = stablehlo.dot_general %1454, %arg203, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1464 = stablehlo.broadcast_in_dim %1463, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1465 = stablehlo.multiply %1464, %273 : tensor<256x256xf32>
    %1466 = stablehlo.broadcast_in_dim %1465, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1467 = stablehlo.broadcast_in_dim %arg204, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1468 = stablehlo.add %1466, %1467 : tensor<256x256xf32>
    %1469 = stablehlo.convert %1468 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1470 = stablehlo.reshape %1469 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1471 = stablehlo.dot_general %1454, %arg205, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1472 = stablehlo.broadcast_in_dim %1471, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1473 = stablehlo.multiply %1472, %127 : tensor<256x1280xf32>
    %1474 = stablehlo.broadcast_in_dim %1473, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1475 = stablehlo.broadcast_in_dim %arg206, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1476 = stablehlo.add %1474, %1475 : tensor<256x1280xf32>
    %1477 = stablehlo.convert %1476 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1478 = stablehlo.reshape %1477 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1479 = stablehlo.reshape %1462 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1480 = stablehlo.transpose %1479, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1481 = stablehlo.reshape %1470 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1482 = stablehlo.transpose %1481, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1483 = stablehlo.reshape %1478 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %1484 = stablehlo.transpose %1483, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1485 = stablehlo.transpose %1482, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %1486 = stablehlo.reshape %1480 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1487 = stablehlo.reshape %1485 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1488 = stablehlo.broadcast_in_dim %1487, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1489 = stablehlo.dot_general %1486, %1488, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %1490 = stablehlo.reshape %1489 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1491 = stablehlo.broadcast_in_dim %1490, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1492 = stablehlo.divide %1491, %309 : tensor<1x8x256x256xbf16>
    %1493 = stablehlo.convert %1492 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %1494 = stablehlo.reduce(%1493 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1495 = stablehlo.reshape %1494 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1496 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1497 = stablehlo.broadcast_in_dim %1495, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1498 = stablehlo.subtract %1496, %1497 : tensor<1x8x256x256xf32>
    %1499 = stablehlo.exponential %1498 : tensor<1x8x256x256xf32>
    %1500 = stablehlo.reduce(%1499 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1501 = stablehlo.reshape %1500 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1502 = stablehlo.broadcast_in_dim %1499, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1503 = stablehlo.broadcast_in_dim %1501, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1504 = stablehlo.divide %1502, %1503 : tensor<1x8x256x256xf32>
    %1505 = stablehlo.convert %1504 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %1506 = stablehlo.reshape %1505 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %1507 = stablehlo.reshape %1484 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1508 = stablehlo.broadcast_in_dim %1507, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1509 = stablehlo.dot_general %1506, %1508, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1510 = stablehlo.reshape %1509 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1511 = stablehlo.transpose %1510, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %1512 = stablehlo.reshape %1511 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %1513 = stablehlo.reshape %1512 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1514 = stablehlo.convert %1513 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1515 = stablehlo.dot_general %1514, %arg207, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1516 = stablehlo.broadcast_in_dim %1515, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1517 = stablehlo.multiply %1516, %127 : tensor<256x1280xf32>
    %1518 = stablehlo.broadcast_in_dim %1517, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1519 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1520 = stablehlo.add %1518, %1519 : tensor<256x1280xf32>
    %1521 = stablehlo.convert %1520 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1522 = stablehlo.reshape %1521 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1523 = stablehlo.add %1522, %1415 : tensor<1x256x1280xbf16>
    %1524 = stablehlo.convert %1523 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1525 = stablehlo.convert %1524 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1526 = stablehlo.reduce(%1525 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1527 = stablehlo.reshape %1526 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1528 = stablehlo.broadcast_in_dim %1527, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1529 = stablehlo.divide %1528, %142 : tensor<1x256x1xf64>
    %1530 = stablehlo.broadcast_in_dim %1525, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1531 = stablehlo.broadcast_in_dim %1529, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1532 = stablehlo.subtract %1530, %1531 : tensor<1x256x1280xf64>
    %1533 = stablehlo.multiply %1532, %1532 : tensor<1x256x1280xf64>
    %1534 = stablehlo.reduce(%1533 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1535 = stablehlo.reshape %1534 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1536 = stablehlo.broadcast_in_dim %1535, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1537 = stablehlo.divide %1536, %142 : tensor<1x256x1xf64>
    %1538 = stablehlo.convert %1537 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1539 = stablehlo.reduce(%1524 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1540 = stablehlo.reshape %1539 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1541 = stablehlo.broadcast_in_dim %1540, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1542 = stablehlo.divide %1541, %158 : tensor<1x256x1xf32>
    %1543 = stablehlo.broadcast_in_dim %1538, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1544 = stablehlo.add %1543, %161 : tensor<1x256x1xf32>
    %1545 = stablehlo.rsqrt %1544 : tensor<1x256x1xf32>
    %1546 = stablehlo.broadcast_in_dim %1524, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1547 = stablehlo.broadcast_in_dim %1542, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1548 = stablehlo.subtract %1546, %1547 : tensor<1x256x1280xf32>
    %1549 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1550 = stablehlo.broadcast_in_dim %1545, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1551 = stablehlo.multiply %1549, %1550 : tensor<1x256x1280xf32>
    %1552 = stablehlo.convert %arg33 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1553 = stablehlo.broadcast_in_dim %1551, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1554 = stablehlo.broadcast_in_dim %1552, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1555 = stablehlo.multiply %1553, %1554 : tensor<1x256x1280xf32>
    %1556 = stablehlo.convert %arg34 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1557 = stablehlo.broadcast_in_dim %1555, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1558 = stablehlo.broadcast_in_dim %1556, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1559 = stablehlo.add %1557, %1558 : tensor<1x256x1280xf32>
    %1560 = stablehlo.convert %1559 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1561 = stablehlo.reshape %1560 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1562 = stablehlo.convert %1561 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1563 = stablehlo.dot_general %1562, %arg209, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1564 = stablehlo.broadcast_in_dim %1563, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1565 = stablehlo.multiply %1564, %127 : tensor<256x1280xf32>
    %1566 = stablehlo.broadcast_in_dim %1565, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1567 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1568 = stablehlo.add %1566, %1567 : tensor<256x1280xf32>
    %1569 = stablehlo.convert %1568 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1570 = stablehlo.reshape %1569 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1571 = stablehlo.multiply %1570, %cst_4 : tensor<1x256x1280xbf16>
    %1572 = stablehlo.multiply %1570, %190 : tensor<1x256x1280xbf16>
    %1573 = stablehlo.convert %1572 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1574 = stablehlo.clamp %cst_5, %1573, %cst_6 : tensor<1x256x1280xf32>
    %1575 = stablehlo.multiply %1574, %1574 : tensor<1x256x1280xf32>
    %1576 = stablehlo.multiply %cst_7, %1575 : tensor<1x256x1280xf32>
    %1577 = stablehlo.add %1576, %cst_8 : tensor<1x256x1280xf32>
    %1578 = stablehlo.multiply %1577, %1575 : tensor<1x256x1280xf32>
    %1579 = stablehlo.add %1578, %cst_9 : tensor<1x256x1280xf32>
    %1580 = stablehlo.multiply %1579, %1575 : tensor<1x256x1280xf32>
    %1581 = stablehlo.add %1580, %cst_10 : tensor<1x256x1280xf32>
    %1582 = stablehlo.multiply %1581, %1575 : tensor<1x256x1280xf32>
    %1583 = stablehlo.add %1582, %cst_11 : tensor<1x256x1280xf32>
    %1584 = stablehlo.multiply %1583, %1575 : tensor<1x256x1280xf32>
    %1585 = stablehlo.add %1584, %cst_12 : tensor<1x256x1280xf32>
    %1586 = stablehlo.multiply %1585, %1575 : tensor<1x256x1280xf32>
    %1587 = stablehlo.add %1586, %cst_13 : tensor<1x256x1280xf32>
    %1588 = stablehlo.multiply %cst_14, %1575 : tensor<1x256x1280xf32>
    %1589 = stablehlo.add %1588, %cst_15 : tensor<1x256x1280xf32>
    %1590 = stablehlo.multiply %1589, %1575 : tensor<1x256x1280xf32>
    %1591 = stablehlo.add %1590, %cst_16 : tensor<1x256x1280xf32>
    %1592 = stablehlo.multiply %1591, %1575 : tensor<1x256x1280xf32>
    %1593 = stablehlo.add %1592, %cst_17 : tensor<1x256x1280xf32>
    %1594 = stablehlo.multiply %1593, %1575 : tensor<1x256x1280xf32>
    %1595 = stablehlo.add %1594, %cst_18 : tensor<1x256x1280xf32>
    %1596 = stablehlo.multiply %1574, %1587 : tensor<1x256x1280xf32>
    %1597 = stablehlo.divide %1596, %1595 : tensor<1x256x1280xf32>
    %1598 = stablehlo.clamp %cst_19, %1597, %cst_20 : tensor<1x256x1280xf32>
    %1599 = stablehlo.convert %1598 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1600 = stablehlo.add %1599, %cst_2 : tensor<1x256x1280xbf16>
    %1601 = stablehlo.multiply %1600, %1571 : tensor<1x256x1280xbf16>
    %1602 = stablehlo.reshape %1601 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1603 = stablehlo.convert %1602 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1604 = stablehlo.dot_general %1603, %arg211, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1605 = stablehlo.broadcast_in_dim %1604, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1606 = stablehlo.multiply %1605, %127 : tensor<256x1280xf32>
    %1607 = stablehlo.broadcast_in_dim %1606, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1608 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1609 = stablehlo.add %1607, %1608 : tensor<256x1280xf32>
    %1610 = stablehlo.convert %1609 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1611 = stablehlo.reshape %1610 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1612 = stablehlo.add %1611, %1523 : tensor<1x256x1280xbf16>
    %1613 = stablehlo.convert %1612 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1614 = stablehlo.convert %1613 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1615 = stablehlo.reduce(%1614 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1616 = stablehlo.reshape %1615 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1617 = stablehlo.broadcast_in_dim %1616, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1618 = stablehlo.divide %1617, %142 : tensor<1x256x1xf64>
    %1619 = stablehlo.broadcast_in_dim %1614, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1620 = stablehlo.broadcast_in_dim %1618, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1621 = stablehlo.subtract %1619, %1620 : tensor<1x256x1280xf64>
    %1622 = stablehlo.multiply %1621, %1621 : tensor<1x256x1280xf64>
    %1623 = stablehlo.reduce(%1622 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1624 = stablehlo.reshape %1623 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1626 = stablehlo.divide %1625, %142 : tensor<1x256x1xf64>
    %1627 = stablehlo.convert %1626 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1628 = stablehlo.reduce(%1613 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1629 = stablehlo.reshape %1628 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1630 = stablehlo.broadcast_in_dim %1629, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1631 = stablehlo.divide %1630, %158 : tensor<1x256x1xf32>
    %1632 = stablehlo.broadcast_in_dim %1627, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1633 = stablehlo.add %1632, %161 : tensor<1x256x1xf32>
    %1634 = stablehlo.rsqrt %1633 : tensor<1x256x1xf32>
    %1635 = stablehlo.broadcast_in_dim %1613, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1636 = stablehlo.broadcast_in_dim %1631, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1637 = stablehlo.subtract %1635, %1636 : tensor<1x256x1280xf32>
    %1638 = stablehlo.broadcast_in_dim %1637, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1639 = stablehlo.broadcast_in_dim %1634, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1640 = stablehlo.multiply %1638, %1639 : tensor<1x256x1280xf32>
    %1641 = stablehlo.convert %arg35 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1642 = stablehlo.broadcast_in_dim %1640, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1643 = stablehlo.broadcast_in_dim %1641, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1644 = stablehlo.multiply %1642, %1643 : tensor<1x256x1280xf32>
    %1645 = stablehlo.convert %arg36 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1646 = stablehlo.broadcast_in_dim %1644, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1647 = stablehlo.broadcast_in_dim %1645, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1648 = stablehlo.add %1646, %1647 : tensor<1x256x1280xf32>
    %1649 = stablehlo.convert %1648 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1650 = stablehlo.reshape %1649 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1651 = stablehlo.convert %1650 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1652 = stablehlo.dot_general %1651, %arg213, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1653 = stablehlo.broadcast_in_dim %1652, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1654 = stablehlo.multiply %1653, %273 : tensor<256x256xf32>
    %1655 = stablehlo.broadcast_in_dim %1654, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1656 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1657 = stablehlo.add %1655, %1656 : tensor<256x256xf32>
    %1658 = stablehlo.convert %1657 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1659 = stablehlo.reshape %1658 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1660 = stablehlo.dot_general %1651, %arg215, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1661 = stablehlo.broadcast_in_dim %1660, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1662 = stablehlo.multiply %1661, %273 : tensor<256x256xf32>
    %1663 = stablehlo.broadcast_in_dim %1662, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1664 = stablehlo.broadcast_in_dim %arg216, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1665 = stablehlo.add %1663, %1664 : tensor<256x256xf32>
    %1666 = stablehlo.convert %1665 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1667 = stablehlo.reshape %1666 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1668 = stablehlo.dot_general %1651, %arg217, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1669 = stablehlo.broadcast_in_dim %1668, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1670 = stablehlo.multiply %1669, %127 : tensor<256x1280xf32>
    %1671 = stablehlo.broadcast_in_dim %1670, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1672 = stablehlo.broadcast_in_dim %arg218, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1673 = stablehlo.add %1671, %1672 : tensor<256x1280xf32>
    %1674 = stablehlo.convert %1673 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1675 = stablehlo.reshape %1674 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1676 = stablehlo.reshape %1659 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1677 = stablehlo.transpose %1676, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1678 = stablehlo.reshape %1667 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1679 = stablehlo.transpose %1678, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1680 = stablehlo.reshape %1675 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %1681 = stablehlo.transpose %1680, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1682 = stablehlo.transpose %1679, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %1683 = stablehlo.reshape %1677 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1684 = stablehlo.reshape %1682 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1685 = stablehlo.broadcast_in_dim %1684, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1686 = stablehlo.dot_general %1683, %1685, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %1687 = stablehlo.reshape %1686 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1689 = stablehlo.divide %1688, %309 : tensor<1x8x256x256xbf16>
    %1690 = stablehlo.convert %1689 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %1691 = stablehlo.reduce(%1690 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1692 = stablehlo.reshape %1691 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1693 = stablehlo.broadcast_in_dim %1690, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1694 = stablehlo.broadcast_in_dim %1692, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1695 = stablehlo.subtract %1693, %1694 : tensor<1x8x256x256xf32>
    %1696 = stablehlo.exponential %1695 : tensor<1x8x256x256xf32>
    %1697 = stablehlo.reduce(%1696 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1698 = stablehlo.reshape %1697 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1699 = stablehlo.broadcast_in_dim %1696, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1700 = stablehlo.broadcast_in_dim %1698, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1701 = stablehlo.divide %1699, %1700 : tensor<1x8x256x256xf32>
    %1702 = stablehlo.convert %1701 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %1703 = stablehlo.reshape %1702 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %1704 = stablehlo.reshape %1681 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1705 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1706 = stablehlo.dot_general %1703, %1705, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1707 = stablehlo.reshape %1706 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1708 = stablehlo.transpose %1707, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %1709 = stablehlo.reshape %1708 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %1710 = stablehlo.reshape %1709 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1711 = stablehlo.convert %1710 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1712 = stablehlo.dot_general %1711, %arg219, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1713 = stablehlo.broadcast_in_dim %1712, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1714 = stablehlo.multiply %1713, %127 : tensor<256x1280xf32>
    %1715 = stablehlo.broadcast_in_dim %1714, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1716 = stablehlo.broadcast_in_dim %arg220, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1717 = stablehlo.add %1715, %1716 : tensor<256x1280xf32>
    %1718 = stablehlo.convert %1717 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1719 = stablehlo.reshape %1718 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1720 = stablehlo.add %1719, %1612 : tensor<1x256x1280xbf16>
    %1721 = stablehlo.convert %1720 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1722 = stablehlo.convert %1721 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1723 = stablehlo.reduce(%1722 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1724 = stablehlo.reshape %1723 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1725 = stablehlo.broadcast_in_dim %1724, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1726 = stablehlo.divide %1725, %142 : tensor<1x256x1xf64>
    %1727 = stablehlo.broadcast_in_dim %1722, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1728 = stablehlo.broadcast_in_dim %1726, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1729 = stablehlo.subtract %1727, %1728 : tensor<1x256x1280xf64>
    %1730 = stablehlo.multiply %1729, %1729 : tensor<1x256x1280xf64>
    %1731 = stablehlo.reduce(%1730 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1732 = stablehlo.reshape %1731 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1733 = stablehlo.broadcast_in_dim %1732, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1734 = stablehlo.divide %1733, %142 : tensor<1x256x1xf64>
    %1735 = stablehlo.convert %1734 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1736 = stablehlo.reduce(%1721 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1737 = stablehlo.reshape %1736 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1738 = stablehlo.broadcast_in_dim %1737, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1739 = stablehlo.divide %1738, %158 : tensor<1x256x1xf32>
    %1740 = stablehlo.broadcast_in_dim %1735, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1741 = stablehlo.add %1740, %161 : tensor<1x256x1xf32>
    %1742 = stablehlo.rsqrt %1741 : tensor<1x256x1xf32>
    %1743 = stablehlo.broadcast_in_dim %1721, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1744 = stablehlo.broadcast_in_dim %1739, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1745 = stablehlo.subtract %1743, %1744 : tensor<1x256x1280xf32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1747 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1748 = stablehlo.multiply %1746, %1747 : tensor<1x256x1280xf32>
    %1749 = stablehlo.convert %arg37 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1750 = stablehlo.broadcast_in_dim %1748, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1751 = stablehlo.broadcast_in_dim %1749, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1752 = stablehlo.multiply %1750, %1751 : tensor<1x256x1280xf32>
    %1753 = stablehlo.convert %arg38 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1754 = stablehlo.broadcast_in_dim %1752, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1755 = stablehlo.broadcast_in_dim %1753, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1756 = stablehlo.add %1754, %1755 : tensor<1x256x1280xf32>
    %1757 = stablehlo.convert %1756 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1758 = stablehlo.reshape %1757 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1759 = stablehlo.convert %1758 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1760 = stablehlo.dot_general %1759, %arg221, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1761 = stablehlo.broadcast_in_dim %1760, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1762 = stablehlo.multiply %1761, %127 : tensor<256x1280xf32>
    %1763 = stablehlo.broadcast_in_dim %1762, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1764 = stablehlo.broadcast_in_dim %arg222, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1765 = stablehlo.add %1763, %1764 : tensor<256x1280xf32>
    %1766 = stablehlo.convert %1765 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1767 = stablehlo.reshape %1766 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1768 = stablehlo.multiply %1767, %cst_4 : tensor<1x256x1280xbf16>
    %1769 = stablehlo.multiply %1767, %190 : tensor<1x256x1280xbf16>
    %1770 = stablehlo.convert %1769 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1771 = stablehlo.clamp %cst_5, %1770, %cst_6 : tensor<1x256x1280xf32>
    %1772 = stablehlo.multiply %1771, %1771 : tensor<1x256x1280xf32>
    %1773 = stablehlo.multiply %cst_7, %1772 : tensor<1x256x1280xf32>
    %1774 = stablehlo.add %1773, %cst_8 : tensor<1x256x1280xf32>
    %1775 = stablehlo.multiply %1774, %1772 : tensor<1x256x1280xf32>
    %1776 = stablehlo.add %1775, %cst_9 : tensor<1x256x1280xf32>
    %1777 = stablehlo.multiply %1776, %1772 : tensor<1x256x1280xf32>
    %1778 = stablehlo.add %1777, %cst_10 : tensor<1x256x1280xf32>
    %1779 = stablehlo.multiply %1778, %1772 : tensor<1x256x1280xf32>
    %1780 = stablehlo.add %1779, %cst_11 : tensor<1x256x1280xf32>
    %1781 = stablehlo.multiply %1780, %1772 : tensor<1x256x1280xf32>
    %1782 = stablehlo.add %1781, %cst_12 : tensor<1x256x1280xf32>
    %1783 = stablehlo.multiply %1782, %1772 : tensor<1x256x1280xf32>
    %1784 = stablehlo.add %1783, %cst_13 : tensor<1x256x1280xf32>
    %1785 = stablehlo.multiply %cst_14, %1772 : tensor<1x256x1280xf32>
    %1786 = stablehlo.add %1785, %cst_15 : tensor<1x256x1280xf32>
    %1787 = stablehlo.multiply %1786, %1772 : tensor<1x256x1280xf32>
    %1788 = stablehlo.add %1787, %cst_16 : tensor<1x256x1280xf32>
    %1789 = stablehlo.multiply %1788, %1772 : tensor<1x256x1280xf32>
    %1790 = stablehlo.add %1789, %cst_17 : tensor<1x256x1280xf32>
    %1791 = stablehlo.multiply %1790, %1772 : tensor<1x256x1280xf32>
    %1792 = stablehlo.add %1791, %cst_18 : tensor<1x256x1280xf32>
    %1793 = stablehlo.multiply %1771, %1784 : tensor<1x256x1280xf32>
    %1794 = stablehlo.divide %1793, %1792 : tensor<1x256x1280xf32>
    %1795 = stablehlo.clamp %cst_19, %1794, %cst_20 : tensor<1x256x1280xf32>
    %1796 = stablehlo.convert %1795 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1797 = stablehlo.add %1796, %cst_2 : tensor<1x256x1280xbf16>
    %1798 = stablehlo.multiply %1797, %1768 : tensor<1x256x1280xbf16>
    %1799 = stablehlo.reshape %1798 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1800 = stablehlo.convert %1799 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1801 = stablehlo.dot_general %1800, %arg223, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1802 = stablehlo.broadcast_in_dim %1801, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1803 = stablehlo.multiply %1802, %127 : tensor<256x1280xf32>
    %1804 = stablehlo.broadcast_in_dim %1803, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1805 = stablehlo.broadcast_in_dim %arg224, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1806 = stablehlo.add %1804, %1805 : tensor<256x1280xf32>
    %1807 = stablehlo.convert %1806 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1808 = stablehlo.reshape %1807 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1809 = stablehlo.add %1808, %1720 : tensor<1x256x1280xbf16>
    %1810 = stablehlo.convert %1809 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1811 = stablehlo.convert %1810 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1812 = stablehlo.reduce(%1811 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1813 = stablehlo.reshape %1812 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1814 = stablehlo.broadcast_in_dim %1813, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1815 = stablehlo.divide %1814, %142 : tensor<1x256x1xf64>
    %1816 = stablehlo.broadcast_in_dim %1811, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1817 = stablehlo.broadcast_in_dim %1815, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1818 = stablehlo.subtract %1816, %1817 : tensor<1x256x1280xf64>
    %1819 = stablehlo.multiply %1818, %1818 : tensor<1x256x1280xf64>
    %1820 = stablehlo.reduce(%1819 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1821 = stablehlo.reshape %1820 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1822 = stablehlo.broadcast_in_dim %1821, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1823 = stablehlo.divide %1822, %142 : tensor<1x256x1xf64>
    %1824 = stablehlo.convert %1823 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1825 = stablehlo.reduce(%1810 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1826 = stablehlo.reshape %1825 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1827 = stablehlo.broadcast_in_dim %1826, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1828 = stablehlo.divide %1827, %158 : tensor<1x256x1xf32>
    %1829 = stablehlo.broadcast_in_dim %1824, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1830 = stablehlo.add %1829, %161 : tensor<1x256x1xf32>
    %1831 = stablehlo.rsqrt %1830 : tensor<1x256x1xf32>
    %1832 = stablehlo.broadcast_in_dim %1810, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1833 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1834 = stablehlo.subtract %1832, %1833 : tensor<1x256x1280xf32>
    %1835 = stablehlo.broadcast_in_dim %1834, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1836 = stablehlo.broadcast_in_dim %1831, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1837 = stablehlo.multiply %1835, %1836 : tensor<1x256x1280xf32>
    %1838 = stablehlo.convert %arg39 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1839 = stablehlo.broadcast_in_dim %1837, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1840 = stablehlo.broadcast_in_dim %1838, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1841 = stablehlo.multiply %1839, %1840 : tensor<1x256x1280xf32>
    %1842 = stablehlo.convert %arg40 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1843 = stablehlo.broadcast_in_dim %1841, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1844 = stablehlo.broadcast_in_dim %1842, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1845 = stablehlo.add %1843, %1844 : tensor<1x256x1280xf32>
    %1846 = stablehlo.convert %1845 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1847 = stablehlo.reshape %1846 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1848 = stablehlo.convert %1847 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1849 = stablehlo.dot_general %1848, %arg225, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1850 = stablehlo.broadcast_in_dim %1849, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1851 = stablehlo.multiply %1850, %273 : tensor<256x256xf32>
    %1852 = stablehlo.broadcast_in_dim %1851, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1853 = stablehlo.broadcast_in_dim %arg226, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1854 = stablehlo.add %1852, %1853 : tensor<256x256xf32>
    %1855 = stablehlo.convert %1854 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1856 = stablehlo.reshape %1855 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1857 = stablehlo.dot_general %1848, %arg227, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %1858 = stablehlo.broadcast_in_dim %1857, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1859 = stablehlo.multiply %1858, %273 : tensor<256x256xf32>
    %1860 = stablehlo.broadcast_in_dim %1859, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1861 = stablehlo.broadcast_in_dim %arg228, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1862 = stablehlo.add %1860, %1861 : tensor<256x256xf32>
    %1863 = stablehlo.convert %1862 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %1864 = stablehlo.reshape %1863 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %1865 = stablehlo.dot_general %1848, %arg229, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1866 = stablehlo.broadcast_in_dim %1865, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1867 = stablehlo.multiply %1866, %127 : tensor<256x1280xf32>
    %1868 = stablehlo.broadcast_in_dim %1867, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1869 = stablehlo.broadcast_in_dim %arg230, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1870 = stablehlo.add %1868, %1869 : tensor<256x1280xf32>
    %1871 = stablehlo.convert %1870 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1872 = stablehlo.reshape %1871 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1873 = stablehlo.reshape %1856 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1874 = stablehlo.transpose %1873, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1875 = stablehlo.reshape %1864 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %1876 = stablehlo.transpose %1875, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %1877 = stablehlo.reshape %1872 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %1878 = stablehlo.transpose %1877, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1879 = stablehlo.transpose %1876, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %1880 = stablehlo.reshape %1874 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %1881 = stablehlo.reshape %1879 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1882 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %1883 = stablehlo.dot_general %1880, %1882, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %1884 = stablehlo.reshape %1883 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %1886 = stablehlo.divide %1885, %309 : tensor<1x8x256x256xbf16>
    %1887 = stablehlo.convert %1886 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %1888 = stablehlo.reduce(%1887 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1889 = stablehlo.reshape %1888 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1890 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1891 = stablehlo.broadcast_in_dim %1889, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1892 = stablehlo.subtract %1890, %1891 : tensor<1x8x256x256xf32>
    %1893 = stablehlo.exponential %1892 : tensor<1x8x256x256xf32>
    %1894 = stablehlo.reduce(%1893 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %1895 = stablehlo.reshape %1894 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %1896 = stablehlo.broadcast_in_dim %1893, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %1897 = stablehlo.broadcast_in_dim %1895, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %1898 = stablehlo.divide %1896, %1897 : tensor<1x8x256x256xf32>
    %1899 = stablehlo.convert %1898 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %1900 = stablehlo.reshape %1899 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %1901 = stablehlo.reshape %1878 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1902 = stablehlo.broadcast_in_dim %1901, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1903 = stablehlo.dot_general %1900, %1902, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %1904 = stablehlo.reshape %1903 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %1905 = stablehlo.transpose %1904, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %1906 = stablehlo.reshape %1905 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %1907 = stablehlo.reshape %1906 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1908 = stablehlo.convert %1907 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1909 = stablehlo.dot_general %1908, %arg231, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1910 = stablehlo.broadcast_in_dim %1909, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1911 = stablehlo.multiply %1910, %127 : tensor<256x1280xf32>
    %1912 = stablehlo.broadcast_in_dim %1911, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1913 = stablehlo.broadcast_in_dim %arg232, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1914 = stablehlo.add %1912, %1913 : tensor<256x1280xf32>
    %1915 = stablehlo.convert %1914 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1916 = stablehlo.reshape %1915 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1917 = stablehlo.add %1916, %1809 : tensor<1x256x1280xbf16>
    %1918 = stablehlo.convert %1917 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1919 = stablehlo.convert %1918 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %1920 = stablehlo.reduce(%1919 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1921 = stablehlo.reshape %1920 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1922 = stablehlo.broadcast_in_dim %1921, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1923 = stablehlo.divide %1922, %142 : tensor<1x256x1xf64>
    %1924 = stablehlo.broadcast_in_dim %1919, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %1925 = stablehlo.broadcast_in_dim %1923, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %1926 = stablehlo.subtract %1924, %1925 : tensor<1x256x1280xf64>
    %1927 = stablehlo.multiply %1926, %1926 : tensor<1x256x1280xf64>
    %1928 = stablehlo.reduce(%1927 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %1929 = stablehlo.reshape %1928 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %1930 = stablehlo.broadcast_in_dim %1929, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %1931 = stablehlo.divide %1930, %142 : tensor<1x256x1xf64>
    %1932 = stablehlo.convert %1931 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %1933 = stablehlo.reduce(%1918 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %1934 = stablehlo.reshape %1933 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1935 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1936 = stablehlo.divide %1935, %158 : tensor<1x256x1xf32>
    %1937 = stablehlo.broadcast_in_dim %1932, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1938 = stablehlo.add %1937, %161 : tensor<1x256x1xf32>
    %1939 = stablehlo.rsqrt %1938 : tensor<1x256x1xf32>
    %1940 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1941 = stablehlo.broadcast_in_dim %1936, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1942 = stablehlo.subtract %1940, %1941 : tensor<1x256x1280xf32>
    %1943 = stablehlo.broadcast_in_dim %1942, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1944 = stablehlo.broadcast_in_dim %1939, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1945 = stablehlo.multiply %1943, %1944 : tensor<1x256x1280xf32>
    %1946 = stablehlo.convert %arg41 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1947 = stablehlo.broadcast_in_dim %1945, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1948 = stablehlo.broadcast_in_dim %1946, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1949 = stablehlo.multiply %1947, %1948 : tensor<1x256x1280xf32>
    %1950 = stablehlo.convert %arg42 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %1951 = stablehlo.broadcast_in_dim %1949, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1952 = stablehlo.broadcast_in_dim %1950, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %1953 = stablehlo.add %1951, %1952 : tensor<1x256x1280xf32>
    %1954 = stablehlo.convert %1953 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1955 = stablehlo.reshape %1954 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1956 = stablehlo.convert %1955 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1957 = stablehlo.dot_general %1956, %arg233, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1958 = stablehlo.broadcast_in_dim %1957, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1959 = stablehlo.multiply %1958, %127 : tensor<256x1280xf32>
    %1960 = stablehlo.broadcast_in_dim %1959, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1961 = stablehlo.broadcast_in_dim %arg234, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %1962 = stablehlo.add %1960, %1961 : tensor<256x1280xf32>
    %1963 = stablehlo.convert %1962 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %1964 = stablehlo.reshape %1963 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %1965 = stablehlo.multiply %1964, %cst_4 : tensor<1x256x1280xbf16>
    %1966 = stablehlo.multiply %1964, %190 : tensor<1x256x1280xbf16>
    %1967 = stablehlo.convert %1966 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %1968 = stablehlo.clamp %cst_5, %1967, %cst_6 : tensor<1x256x1280xf32>
    %1969 = stablehlo.multiply %1968, %1968 : tensor<1x256x1280xf32>
    %1970 = stablehlo.multiply %cst_7, %1969 : tensor<1x256x1280xf32>
    %1971 = stablehlo.add %1970, %cst_8 : tensor<1x256x1280xf32>
    %1972 = stablehlo.multiply %1971, %1969 : tensor<1x256x1280xf32>
    %1973 = stablehlo.add %1972, %cst_9 : tensor<1x256x1280xf32>
    %1974 = stablehlo.multiply %1973, %1969 : tensor<1x256x1280xf32>
    %1975 = stablehlo.add %1974, %cst_10 : tensor<1x256x1280xf32>
    %1976 = stablehlo.multiply %1975, %1969 : tensor<1x256x1280xf32>
    %1977 = stablehlo.add %1976, %cst_11 : tensor<1x256x1280xf32>
    %1978 = stablehlo.multiply %1977, %1969 : tensor<1x256x1280xf32>
    %1979 = stablehlo.add %1978, %cst_12 : tensor<1x256x1280xf32>
    %1980 = stablehlo.multiply %1979, %1969 : tensor<1x256x1280xf32>
    %1981 = stablehlo.add %1980, %cst_13 : tensor<1x256x1280xf32>
    %1982 = stablehlo.multiply %cst_14, %1969 : tensor<1x256x1280xf32>
    %1983 = stablehlo.add %1982, %cst_15 : tensor<1x256x1280xf32>
    %1984 = stablehlo.multiply %1983, %1969 : tensor<1x256x1280xf32>
    %1985 = stablehlo.add %1984, %cst_16 : tensor<1x256x1280xf32>
    %1986 = stablehlo.multiply %1985, %1969 : tensor<1x256x1280xf32>
    %1987 = stablehlo.add %1986, %cst_17 : tensor<1x256x1280xf32>
    %1988 = stablehlo.multiply %1987, %1969 : tensor<1x256x1280xf32>
    %1989 = stablehlo.add %1988, %cst_18 : tensor<1x256x1280xf32>
    %1990 = stablehlo.multiply %1968, %1981 : tensor<1x256x1280xf32>
    %1991 = stablehlo.divide %1990, %1989 : tensor<1x256x1280xf32>
    %1992 = stablehlo.clamp %cst_19, %1991, %cst_20 : tensor<1x256x1280xf32>
    %1993 = stablehlo.convert %1992 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %1994 = stablehlo.add %1993, %cst_2 : tensor<1x256x1280xbf16>
    %1995 = stablehlo.multiply %1994, %1965 : tensor<1x256x1280xbf16>
    %1996 = stablehlo.reshape %1995 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %1997 = stablehlo.convert %1996 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %1998 = stablehlo.dot_general %1997, %arg235, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %1999 = stablehlo.broadcast_in_dim %1998, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2000 = stablehlo.multiply %1999, %127 : tensor<256x1280xf32>
    %2001 = stablehlo.broadcast_in_dim %2000, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2002 = stablehlo.broadcast_in_dim %arg236, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2003 = stablehlo.add %2001, %2002 : tensor<256x1280xf32>
    %2004 = stablehlo.convert %2003 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2005 = stablehlo.reshape %2004 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2006 = stablehlo.add %2005, %1917 : tensor<1x256x1280xbf16>
    %2007 = stablehlo.convert %2006 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2008 = stablehlo.convert %2007 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2009 = stablehlo.reduce(%2008 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2010 = stablehlo.reshape %2009 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2011 = stablehlo.broadcast_in_dim %2010, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2012 = stablehlo.divide %2011, %142 : tensor<1x256x1xf64>
    %2013 = stablehlo.broadcast_in_dim %2008, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2014 = stablehlo.broadcast_in_dim %2012, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2015 = stablehlo.subtract %2013, %2014 : tensor<1x256x1280xf64>
    %2016 = stablehlo.multiply %2015, %2015 : tensor<1x256x1280xf64>
    %2017 = stablehlo.reduce(%2016 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2018 = stablehlo.reshape %2017 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2019 = stablehlo.broadcast_in_dim %2018, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2020 = stablehlo.divide %2019, %142 : tensor<1x256x1xf64>
    %2021 = stablehlo.convert %2020 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2022 = stablehlo.reduce(%2007 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2023 = stablehlo.reshape %2022 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2024 = stablehlo.broadcast_in_dim %2023, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2025 = stablehlo.divide %2024, %158 : tensor<1x256x1xf32>
    %2026 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2027 = stablehlo.add %2026, %161 : tensor<1x256x1xf32>
    %2028 = stablehlo.rsqrt %2027 : tensor<1x256x1xf32>
    %2029 = stablehlo.broadcast_in_dim %2007, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2030 = stablehlo.broadcast_in_dim %2025, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2031 = stablehlo.subtract %2029, %2030 : tensor<1x256x1280xf32>
    %2032 = stablehlo.broadcast_in_dim %2031, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2033 = stablehlo.broadcast_in_dim %2028, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2034 = stablehlo.multiply %2032, %2033 : tensor<1x256x1280xf32>
    %2035 = stablehlo.convert %arg43 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2036 = stablehlo.broadcast_in_dim %2034, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2037 = stablehlo.broadcast_in_dim %2035, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2038 = stablehlo.multiply %2036, %2037 : tensor<1x256x1280xf32>
    %2039 = stablehlo.convert %arg44 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2040 = stablehlo.broadcast_in_dim %2038, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2041 = stablehlo.broadcast_in_dim %2039, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2042 = stablehlo.add %2040, %2041 : tensor<1x256x1280xf32>
    %2043 = stablehlo.convert %2042 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2044 = stablehlo.reshape %2043 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2045 = stablehlo.convert %2044 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2046 = stablehlo.dot_general %2045, %arg237, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2047 = stablehlo.broadcast_in_dim %2046, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2048 = stablehlo.multiply %2047, %273 : tensor<256x256xf32>
    %2049 = stablehlo.broadcast_in_dim %2048, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2050 = stablehlo.broadcast_in_dim %arg238, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2051 = stablehlo.add %2049, %2050 : tensor<256x256xf32>
    %2052 = stablehlo.convert %2051 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2053 = stablehlo.reshape %2052 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2054 = stablehlo.dot_general %2045, %arg239, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2055 = stablehlo.broadcast_in_dim %2054, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2056 = stablehlo.multiply %2055, %273 : tensor<256x256xf32>
    %2057 = stablehlo.broadcast_in_dim %2056, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2058 = stablehlo.broadcast_in_dim %arg240, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2059 = stablehlo.add %2057, %2058 : tensor<256x256xf32>
    %2060 = stablehlo.convert %2059 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2061 = stablehlo.reshape %2060 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2062 = stablehlo.dot_general %2045, %arg241, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2063 = stablehlo.broadcast_in_dim %2062, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2064 = stablehlo.multiply %2063, %127 : tensor<256x1280xf32>
    %2065 = stablehlo.broadcast_in_dim %2064, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2066 = stablehlo.broadcast_in_dim %arg242, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2067 = stablehlo.add %2065, %2066 : tensor<256x1280xf32>
    %2068 = stablehlo.convert %2067 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2069 = stablehlo.reshape %2068 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2070 = stablehlo.reshape %2053 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2071 = stablehlo.transpose %2070, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2072 = stablehlo.reshape %2061 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2073 = stablehlo.transpose %2072, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2074 = stablehlo.reshape %2069 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %2075 = stablehlo.transpose %2074, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2076 = stablehlo.transpose %2073, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %2077 = stablehlo.reshape %2071 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2078 = stablehlo.reshape %2076 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2079 = stablehlo.broadcast_in_dim %2078, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2080 = stablehlo.dot_general %2077, %2079, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %2081 = stablehlo.reshape %2080 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2082 = stablehlo.broadcast_in_dim %2081, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2083 = stablehlo.divide %2082, %309 : tensor<1x8x256x256xbf16>
    %2084 = stablehlo.convert %2083 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %2085 = stablehlo.reduce(%2084 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2086 = stablehlo.reshape %2085 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2087 = stablehlo.broadcast_in_dim %2084, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2088 = stablehlo.broadcast_in_dim %2086, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2089 = stablehlo.subtract %2087, %2088 : tensor<1x8x256x256xf32>
    %2090 = stablehlo.exponential %2089 : tensor<1x8x256x256xf32>
    %2091 = stablehlo.reduce(%2090 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2092 = stablehlo.reshape %2091 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2093 = stablehlo.broadcast_in_dim %2090, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2094 = stablehlo.broadcast_in_dim %2092, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2095 = stablehlo.divide %2093, %2094 : tensor<1x8x256x256xf32>
    %2096 = stablehlo.convert %2095 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %2097 = stablehlo.reshape %2096 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %2098 = stablehlo.reshape %2075 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2099 = stablehlo.broadcast_in_dim %2098, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2100 = stablehlo.dot_general %2097, %2099, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2101 = stablehlo.reshape %2100 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2102 = stablehlo.transpose %2101, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %2103 = stablehlo.reshape %2102 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %2104 = stablehlo.reshape %2103 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2105 = stablehlo.convert %2104 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2106 = stablehlo.dot_general %2105, %arg243, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2107 = stablehlo.broadcast_in_dim %2106, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2108 = stablehlo.multiply %2107, %127 : tensor<256x1280xf32>
    %2109 = stablehlo.broadcast_in_dim %2108, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2110 = stablehlo.broadcast_in_dim %arg244, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2111 = stablehlo.add %2109, %2110 : tensor<256x1280xf32>
    %2112 = stablehlo.convert %2111 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2113 = stablehlo.reshape %2112 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2114 = stablehlo.add %2113, %2006 : tensor<1x256x1280xbf16>
    %2115 = stablehlo.convert %2114 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2116 = stablehlo.convert %2115 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2117 = stablehlo.reduce(%2116 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2118 = stablehlo.reshape %2117 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2119 = stablehlo.broadcast_in_dim %2118, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2120 = stablehlo.divide %2119, %142 : tensor<1x256x1xf64>
    %2121 = stablehlo.broadcast_in_dim %2116, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2122 = stablehlo.broadcast_in_dim %2120, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2123 = stablehlo.subtract %2121, %2122 : tensor<1x256x1280xf64>
    %2124 = stablehlo.multiply %2123, %2123 : tensor<1x256x1280xf64>
    %2125 = stablehlo.reduce(%2124 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2126 = stablehlo.reshape %2125 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2127 = stablehlo.broadcast_in_dim %2126, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2128 = stablehlo.divide %2127, %142 : tensor<1x256x1xf64>
    %2129 = stablehlo.convert %2128 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2130 = stablehlo.reduce(%2115 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2131 = stablehlo.reshape %2130 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2132 = stablehlo.broadcast_in_dim %2131, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2133 = stablehlo.divide %2132, %158 : tensor<1x256x1xf32>
    %2134 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2135 = stablehlo.add %2134, %161 : tensor<1x256x1xf32>
    %2136 = stablehlo.rsqrt %2135 : tensor<1x256x1xf32>
    %2137 = stablehlo.broadcast_in_dim %2115, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2138 = stablehlo.broadcast_in_dim %2133, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2139 = stablehlo.subtract %2137, %2138 : tensor<1x256x1280xf32>
    %2140 = stablehlo.broadcast_in_dim %2139, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2141 = stablehlo.broadcast_in_dim %2136, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2142 = stablehlo.multiply %2140, %2141 : tensor<1x256x1280xf32>
    %2143 = stablehlo.convert %arg45 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2144 = stablehlo.broadcast_in_dim %2142, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2145 = stablehlo.broadcast_in_dim %2143, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2146 = stablehlo.multiply %2144, %2145 : tensor<1x256x1280xf32>
    %2147 = stablehlo.convert %arg46 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2148 = stablehlo.broadcast_in_dim %2146, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2149 = stablehlo.broadcast_in_dim %2147, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2150 = stablehlo.add %2148, %2149 : tensor<1x256x1280xf32>
    %2151 = stablehlo.convert %2150 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2152 = stablehlo.reshape %2151 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2153 = stablehlo.convert %2152 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2154 = stablehlo.dot_general %2153, %arg245, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2155 = stablehlo.broadcast_in_dim %2154, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2156 = stablehlo.multiply %2155, %127 : tensor<256x1280xf32>
    %2157 = stablehlo.broadcast_in_dim %2156, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2158 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2159 = stablehlo.add %2157, %2158 : tensor<256x1280xf32>
    %2160 = stablehlo.convert %2159 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2161 = stablehlo.reshape %2160 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2162 = stablehlo.multiply %2161, %cst_4 : tensor<1x256x1280xbf16>
    %2163 = stablehlo.multiply %2161, %190 : tensor<1x256x1280xbf16>
    %2164 = stablehlo.convert %2163 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2165 = stablehlo.clamp %cst_5, %2164, %cst_6 : tensor<1x256x1280xf32>
    %2166 = stablehlo.multiply %2165, %2165 : tensor<1x256x1280xf32>
    %2167 = stablehlo.multiply %cst_7, %2166 : tensor<1x256x1280xf32>
    %2168 = stablehlo.add %2167, %cst_8 : tensor<1x256x1280xf32>
    %2169 = stablehlo.multiply %2168, %2166 : tensor<1x256x1280xf32>
    %2170 = stablehlo.add %2169, %cst_9 : tensor<1x256x1280xf32>
    %2171 = stablehlo.multiply %2170, %2166 : tensor<1x256x1280xf32>
    %2172 = stablehlo.add %2171, %cst_10 : tensor<1x256x1280xf32>
    %2173 = stablehlo.multiply %2172, %2166 : tensor<1x256x1280xf32>
    %2174 = stablehlo.add %2173, %cst_11 : tensor<1x256x1280xf32>
    %2175 = stablehlo.multiply %2174, %2166 : tensor<1x256x1280xf32>
    %2176 = stablehlo.add %2175, %cst_12 : tensor<1x256x1280xf32>
    %2177 = stablehlo.multiply %2176, %2166 : tensor<1x256x1280xf32>
    %2178 = stablehlo.add %2177, %cst_13 : tensor<1x256x1280xf32>
    %2179 = stablehlo.multiply %cst_14, %2166 : tensor<1x256x1280xf32>
    %2180 = stablehlo.add %2179, %cst_15 : tensor<1x256x1280xf32>
    %2181 = stablehlo.multiply %2180, %2166 : tensor<1x256x1280xf32>
    %2182 = stablehlo.add %2181, %cst_16 : tensor<1x256x1280xf32>
    %2183 = stablehlo.multiply %2182, %2166 : tensor<1x256x1280xf32>
    %2184 = stablehlo.add %2183, %cst_17 : tensor<1x256x1280xf32>
    %2185 = stablehlo.multiply %2184, %2166 : tensor<1x256x1280xf32>
    %2186 = stablehlo.add %2185, %cst_18 : tensor<1x256x1280xf32>
    %2187 = stablehlo.multiply %2165, %2178 : tensor<1x256x1280xf32>
    %2188 = stablehlo.divide %2187, %2186 : tensor<1x256x1280xf32>
    %2189 = stablehlo.clamp %cst_19, %2188, %cst_20 : tensor<1x256x1280xf32>
    %2190 = stablehlo.convert %2189 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2191 = stablehlo.add %2190, %cst_2 : tensor<1x256x1280xbf16>
    %2192 = stablehlo.multiply %2191, %2162 : tensor<1x256x1280xbf16>
    %2193 = stablehlo.reshape %2192 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2194 = stablehlo.convert %2193 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2195 = stablehlo.dot_general %2194, %arg247, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2196 = stablehlo.broadcast_in_dim %2195, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2197 = stablehlo.multiply %2196, %127 : tensor<256x1280xf32>
    %2198 = stablehlo.broadcast_in_dim %2197, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2199 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2200 = stablehlo.add %2198, %2199 : tensor<256x1280xf32>
    %2201 = stablehlo.convert %2200 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2202 = stablehlo.reshape %2201 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2203 = stablehlo.add %2202, %2114 : tensor<1x256x1280xbf16>
    %2204 = stablehlo.convert %2203 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2205 = stablehlo.convert %2204 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2206 = stablehlo.reduce(%2205 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2207 = stablehlo.reshape %2206 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2208 = stablehlo.broadcast_in_dim %2207, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2209 = stablehlo.divide %2208, %142 : tensor<1x256x1xf64>
    %2210 = stablehlo.broadcast_in_dim %2205, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2211 = stablehlo.broadcast_in_dim %2209, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2212 = stablehlo.subtract %2210, %2211 : tensor<1x256x1280xf64>
    %2213 = stablehlo.multiply %2212, %2212 : tensor<1x256x1280xf64>
    %2214 = stablehlo.reduce(%2213 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2215 = stablehlo.reshape %2214 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2216 = stablehlo.broadcast_in_dim %2215, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2217 = stablehlo.divide %2216, %142 : tensor<1x256x1xf64>
    %2218 = stablehlo.convert %2217 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2219 = stablehlo.reduce(%2204 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2220 = stablehlo.reshape %2219 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2221 = stablehlo.broadcast_in_dim %2220, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2222 = stablehlo.divide %2221, %158 : tensor<1x256x1xf32>
    %2223 = stablehlo.broadcast_in_dim %2218, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2224 = stablehlo.add %2223, %161 : tensor<1x256x1xf32>
    %2225 = stablehlo.rsqrt %2224 : tensor<1x256x1xf32>
    %2226 = stablehlo.broadcast_in_dim %2204, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2227 = stablehlo.broadcast_in_dim %2222, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2228 = stablehlo.subtract %2226, %2227 : tensor<1x256x1280xf32>
    %2229 = stablehlo.broadcast_in_dim %2228, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2230 = stablehlo.broadcast_in_dim %2225, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2231 = stablehlo.multiply %2229, %2230 : tensor<1x256x1280xf32>
    %2232 = stablehlo.convert %arg47 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2233 = stablehlo.broadcast_in_dim %2231, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2234 = stablehlo.broadcast_in_dim %2232, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2235 = stablehlo.multiply %2233, %2234 : tensor<1x256x1280xf32>
    %2236 = stablehlo.convert %arg48 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2237 = stablehlo.broadcast_in_dim %2235, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2238 = stablehlo.broadcast_in_dim %2236, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2239 = stablehlo.add %2237, %2238 : tensor<1x256x1280xf32>
    %2240 = stablehlo.convert %2239 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2241 = stablehlo.reshape %2240 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2242 = stablehlo.convert %2241 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2243 = stablehlo.dot_general %2242, %arg249, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2244 = stablehlo.broadcast_in_dim %2243, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2245 = stablehlo.multiply %2244, %273 : tensor<256x256xf32>
    %2246 = stablehlo.broadcast_in_dim %2245, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2247 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2248 = stablehlo.add %2246, %2247 : tensor<256x256xf32>
    %2249 = stablehlo.convert %2248 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2250 = stablehlo.reshape %2249 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2251 = stablehlo.dot_general %2242, %arg251, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2252 = stablehlo.broadcast_in_dim %2251, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2253 = stablehlo.multiply %2252, %273 : tensor<256x256xf32>
    %2254 = stablehlo.broadcast_in_dim %2253, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2255 = stablehlo.broadcast_in_dim %arg252, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2256 = stablehlo.add %2254, %2255 : tensor<256x256xf32>
    %2257 = stablehlo.convert %2256 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2258 = stablehlo.reshape %2257 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2259 = stablehlo.dot_general %2242, %arg253, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2260 = stablehlo.broadcast_in_dim %2259, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2261 = stablehlo.multiply %2260, %127 : tensor<256x1280xf32>
    %2262 = stablehlo.broadcast_in_dim %2261, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2263 = stablehlo.broadcast_in_dim %arg254, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2264 = stablehlo.add %2262, %2263 : tensor<256x1280xf32>
    %2265 = stablehlo.convert %2264 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2266 = stablehlo.reshape %2265 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2267 = stablehlo.reshape %2250 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2268 = stablehlo.transpose %2267, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2269 = stablehlo.reshape %2258 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2270 = stablehlo.transpose %2269, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2271 = stablehlo.reshape %2266 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %2272 = stablehlo.transpose %2271, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2273 = stablehlo.transpose %2270, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %2274 = stablehlo.reshape %2268 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2275 = stablehlo.reshape %2273 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2276 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2277 = stablehlo.dot_general %2274, %2276, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %2278 = stablehlo.reshape %2277 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2279 = stablehlo.broadcast_in_dim %2278, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2280 = stablehlo.divide %2279, %309 : tensor<1x8x256x256xbf16>
    %2281 = stablehlo.convert %2280 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %2282 = stablehlo.reduce(%2281 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2283 = stablehlo.reshape %2282 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2284 = stablehlo.broadcast_in_dim %2281, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2285 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2286 = stablehlo.subtract %2284, %2285 : tensor<1x8x256x256xf32>
    %2287 = stablehlo.exponential %2286 : tensor<1x8x256x256xf32>
    %2288 = stablehlo.reduce(%2287 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2289 = stablehlo.reshape %2288 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2290 = stablehlo.broadcast_in_dim %2287, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2291 = stablehlo.broadcast_in_dim %2289, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2292 = stablehlo.divide %2290, %2291 : tensor<1x8x256x256xf32>
    %2293 = stablehlo.convert %2292 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %2294 = stablehlo.reshape %2293 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %2295 = stablehlo.reshape %2272 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2296 = stablehlo.broadcast_in_dim %2295, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2297 = stablehlo.dot_general %2294, %2296, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2298 = stablehlo.reshape %2297 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2299 = stablehlo.transpose %2298, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %2300 = stablehlo.reshape %2299 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %2301 = stablehlo.reshape %2300 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2302 = stablehlo.convert %2301 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2303 = stablehlo.dot_general %2302, %arg255, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2304 = stablehlo.broadcast_in_dim %2303, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2305 = stablehlo.multiply %2304, %127 : tensor<256x1280xf32>
    %2306 = stablehlo.broadcast_in_dim %2305, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2307 = stablehlo.broadcast_in_dim %arg256, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2308 = stablehlo.add %2306, %2307 : tensor<256x1280xf32>
    %2309 = stablehlo.convert %2308 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2310 = stablehlo.reshape %2309 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2311 = stablehlo.add %2310, %2203 : tensor<1x256x1280xbf16>
    %2312 = stablehlo.convert %2311 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2313 = stablehlo.convert %2312 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2314 = stablehlo.reduce(%2313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2315 = stablehlo.reshape %2314 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2316 = stablehlo.broadcast_in_dim %2315, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2317 = stablehlo.divide %2316, %142 : tensor<1x256x1xf64>
    %2318 = stablehlo.broadcast_in_dim %2313, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2319 = stablehlo.broadcast_in_dim %2317, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2320 = stablehlo.subtract %2318, %2319 : tensor<1x256x1280xf64>
    %2321 = stablehlo.multiply %2320, %2320 : tensor<1x256x1280xf64>
    %2322 = stablehlo.reduce(%2321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2323 = stablehlo.reshape %2322 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2324 = stablehlo.broadcast_in_dim %2323, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2325 = stablehlo.divide %2324, %142 : tensor<1x256x1xf64>
    %2326 = stablehlo.convert %2325 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2327 = stablehlo.reduce(%2312 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2328 = stablehlo.reshape %2327 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2329 = stablehlo.broadcast_in_dim %2328, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2330 = stablehlo.divide %2329, %158 : tensor<1x256x1xf32>
    %2331 = stablehlo.broadcast_in_dim %2326, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2332 = stablehlo.add %2331, %161 : tensor<1x256x1xf32>
    %2333 = stablehlo.rsqrt %2332 : tensor<1x256x1xf32>
    %2334 = stablehlo.broadcast_in_dim %2312, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2335 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2336 = stablehlo.subtract %2334, %2335 : tensor<1x256x1280xf32>
    %2337 = stablehlo.broadcast_in_dim %2336, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2338 = stablehlo.broadcast_in_dim %2333, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2339 = stablehlo.multiply %2337, %2338 : tensor<1x256x1280xf32>
    %2340 = stablehlo.convert %arg49 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2341 = stablehlo.broadcast_in_dim %2339, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2342 = stablehlo.broadcast_in_dim %2340, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2343 = stablehlo.multiply %2341, %2342 : tensor<1x256x1280xf32>
    %2344 = stablehlo.convert %arg50 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2345 = stablehlo.broadcast_in_dim %2343, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2346 = stablehlo.broadcast_in_dim %2344, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2347 = stablehlo.add %2345, %2346 : tensor<1x256x1280xf32>
    %2348 = stablehlo.convert %2347 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2349 = stablehlo.reshape %2348 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2350 = stablehlo.convert %2349 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2351 = stablehlo.dot_general %2350, %arg257, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2352 = stablehlo.broadcast_in_dim %2351, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2353 = stablehlo.multiply %2352, %127 : tensor<256x1280xf32>
    %2354 = stablehlo.broadcast_in_dim %2353, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2355 = stablehlo.broadcast_in_dim %arg258, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2356 = stablehlo.add %2354, %2355 : tensor<256x1280xf32>
    %2357 = stablehlo.convert %2356 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2358 = stablehlo.reshape %2357 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2359 = stablehlo.multiply %2358, %cst_4 : tensor<1x256x1280xbf16>
    %2360 = stablehlo.multiply %2358, %190 : tensor<1x256x1280xbf16>
    %2361 = stablehlo.convert %2360 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2362 = stablehlo.clamp %cst_5, %2361, %cst_6 : tensor<1x256x1280xf32>
    %2363 = stablehlo.multiply %2362, %2362 : tensor<1x256x1280xf32>
    %2364 = stablehlo.multiply %cst_7, %2363 : tensor<1x256x1280xf32>
    %2365 = stablehlo.add %2364, %cst_8 : tensor<1x256x1280xf32>
    %2366 = stablehlo.multiply %2365, %2363 : tensor<1x256x1280xf32>
    %2367 = stablehlo.add %2366, %cst_9 : tensor<1x256x1280xf32>
    %2368 = stablehlo.multiply %2367, %2363 : tensor<1x256x1280xf32>
    %2369 = stablehlo.add %2368, %cst_10 : tensor<1x256x1280xf32>
    %2370 = stablehlo.multiply %2369, %2363 : tensor<1x256x1280xf32>
    %2371 = stablehlo.add %2370, %cst_11 : tensor<1x256x1280xf32>
    %2372 = stablehlo.multiply %2371, %2363 : tensor<1x256x1280xf32>
    %2373 = stablehlo.add %2372, %cst_12 : tensor<1x256x1280xf32>
    %2374 = stablehlo.multiply %2373, %2363 : tensor<1x256x1280xf32>
    %2375 = stablehlo.add %2374, %cst_13 : tensor<1x256x1280xf32>
    %2376 = stablehlo.multiply %cst_14, %2363 : tensor<1x256x1280xf32>
    %2377 = stablehlo.add %2376, %cst_15 : tensor<1x256x1280xf32>
    %2378 = stablehlo.multiply %2377, %2363 : tensor<1x256x1280xf32>
    %2379 = stablehlo.add %2378, %cst_16 : tensor<1x256x1280xf32>
    %2380 = stablehlo.multiply %2379, %2363 : tensor<1x256x1280xf32>
    %2381 = stablehlo.add %2380, %cst_17 : tensor<1x256x1280xf32>
    %2382 = stablehlo.multiply %2381, %2363 : tensor<1x256x1280xf32>
    %2383 = stablehlo.add %2382, %cst_18 : tensor<1x256x1280xf32>
    %2384 = stablehlo.multiply %2362, %2375 : tensor<1x256x1280xf32>
    %2385 = stablehlo.divide %2384, %2383 : tensor<1x256x1280xf32>
    %2386 = stablehlo.clamp %cst_19, %2385, %cst_20 : tensor<1x256x1280xf32>
    %2387 = stablehlo.convert %2386 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2388 = stablehlo.add %2387, %cst_2 : tensor<1x256x1280xbf16>
    %2389 = stablehlo.multiply %2388, %2359 : tensor<1x256x1280xbf16>
    %2390 = stablehlo.reshape %2389 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2391 = stablehlo.convert %2390 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2392 = stablehlo.dot_general %2391, %arg259, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2393 = stablehlo.broadcast_in_dim %2392, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2394 = stablehlo.multiply %2393, %127 : tensor<256x1280xf32>
    %2395 = stablehlo.broadcast_in_dim %2394, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2396 = stablehlo.broadcast_in_dim %arg260, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2397 = stablehlo.add %2395, %2396 : tensor<256x1280xf32>
    %2398 = stablehlo.convert %2397 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2399 = stablehlo.reshape %2398 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2400 = stablehlo.add %2399, %2311 : tensor<1x256x1280xbf16>
    %2401 = stablehlo.convert %2400 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2402 = stablehlo.convert %2401 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2403 = stablehlo.reduce(%2402 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2404 = stablehlo.reshape %2403 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2405 = stablehlo.broadcast_in_dim %2404, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2406 = stablehlo.divide %2405, %142 : tensor<1x256x1xf64>
    %2407 = stablehlo.broadcast_in_dim %2402, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2408 = stablehlo.broadcast_in_dim %2406, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2409 = stablehlo.subtract %2407, %2408 : tensor<1x256x1280xf64>
    %2410 = stablehlo.multiply %2409, %2409 : tensor<1x256x1280xf64>
    %2411 = stablehlo.reduce(%2410 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2412 = stablehlo.reshape %2411 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2413 = stablehlo.broadcast_in_dim %2412, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2414 = stablehlo.divide %2413, %142 : tensor<1x256x1xf64>
    %2415 = stablehlo.convert %2414 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2416 = stablehlo.reduce(%2401 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2417 = stablehlo.reshape %2416 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2418 = stablehlo.broadcast_in_dim %2417, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2419 = stablehlo.divide %2418, %158 : tensor<1x256x1xf32>
    %2420 = stablehlo.broadcast_in_dim %2415, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2421 = stablehlo.add %2420, %161 : tensor<1x256x1xf32>
    %2422 = stablehlo.rsqrt %2421 : tensor<1x256x1xf32>
    %2423 = stablehlo.broadcast_in_dim %2401, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2424 = stablehlo.broadcast_in_dim %2419, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2425 = stablehlo.subtract %2423, %2424 : tensor<1x256x1280xf32>
    %2426 = stablehlo.broadcast_in_dim %2425, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2427 = stablehlo.broadcast_in_dim %2422, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2428 = stablehlo.multiply %2426, %2427 : tensor<1x256x1280xf32>
    %2429 = stablehlo.convert %arg51 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2430 = stablehlo.broadcast_in_dim %2428, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2431 = stablehlo.broadcast_in_dim %2429, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2432 = stablehlo.multiply %2430, %2431 : tensor<1x256x1280xf32>
    %2433 = stablehlo.convert %arg52 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2434 = stablehlo.broadcast_in_dim %2432, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2435 = stablehlo.broadcast_in_dim %2433, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2436 = stablehlo.add %2434, %2435 : tensor<1x256x1280xf32>
    %2437 = stablehlo.convert %2436 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2438 = stablehlo.reshape %2437 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2439 = stablehlo.convert %2438 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2440 = stablehlo.dot_general %2439, %arg261, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2441 = stablehlo.broadcast_in_dim %2440, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2442 = stablehlo.multiply %2441, %273 : tensor<256x256xf32>
    %2443 = stablehlo.broadcast_in_dim %2442, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2444 = stablehlo.broadcast_in_dim %arg262, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2445 = stablehlo.add %2443, %2444 : tensor<256x256xf32>
    %2446 = stablehlo.convert %2445 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2447 = stablehlo.reshape %2446 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2448 = stablehlo.dot_general %2439, %arg263, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2449 = stablehlo.broadcast_in_dim %2448, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2450 = stablehlo.multiply %2449, %273 : tensor<256x256xf32>
    %2451 = stablehlo.broadcast_in_dim %2450, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2452 = stablehlo.broadcast_in_dim %arg264, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2453 = stablehlo.add %2451, %2452 : tensor<256x256xf32>
    %2454 = stablehlo.convert %2453 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2455 = stablehlo.reshape %2454 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2456 = stablehlo.dot_general %2439, %arg265, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2457 = stablehlo.broadcast_in_dim %2456, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2458 = stablehlo.multiply %2457, %127 : tensor<256x1280xf32>
    %2459 = stablehlo.broadcast_in_dim %2458, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2460 = stablehlo.broadcast_in_dim %arg266, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2461 = stablehlo.add %2459, %2460 : tensor<256x1280xf32>
    %2462 = stablehlo.convert %2461 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2463 = stablehlo.reshape %2462 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2464 = stablehlo.reshape %2447 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2465 = stablehlo.transpose %2464, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2466 = stablehlo.reshape %2455 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2467 = stablehlo.transpose %2466, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2468 = stablehlo.reshape %2463 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %2469 = stablehlo.transpose %2468, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2470 = stablehlo.transpose %2467, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %2471 = stablehlo.reshape %2465 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2472 = stablehlo.reshape %2470 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2473 = stablehlo.broadcast_in_dim %2472, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2474 = stablehlo.dot_general %2471, %2473, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %2475 = stablehlo.reshape %2474 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2477 = stablehlo.divide %2476, %309 : tensor<1x8x256x256xbf16>
    %2478 = stablehlo.convert %2477 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %2479 = stablehlo.reduce(%2478 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2480 = stablehlo.reshape %2479 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2481 = stablehlo.broadcast_in_dim %2478, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2482 = stablehlo.broadcast_in_dim %2480, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2483 = stablehlo.subtract %2481, %2482 : tensor<1x8x256x256xf32>
    %2484 = stablehlo.exponential %2483 : tensor<1x8x256x256xf32>
    %2485 = stablehlo.reduce(%2484 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2486 = stablehlo.reshape %2485 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2487 = stablehlo.broadcast_in_dim %2484, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2488 = stablehlo.broadcast_in_dim %2486, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2489 = stablehlo.divide %2487, %2488 : tensor<1x8x256x256xf32>
    %2490 = stablehlo.convert %2489 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %2491 = stablehlo.reshape %2490 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %2492 = stablehlo.reshape %2469 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2493 = stablehlo.broadcast_in_dim %2492, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2494 = stablehlo.dot_general %2491, %2493, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2495 = stablehlo.reshape %2494 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2496 = stablehlo.transpose %2495, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %2497 = stablehlo.reshape %2496 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %2498 = stablehlo.reshape %2497 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2499 = stablehlo.convert %2498 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2500 = stablehlo.dot_general %2499, %arg267, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2501 = stablehlo.broadcast_in_dim %2500, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2502 = stablehlo.multiply %2501, %127 : tensor<256x1280xf32>
    %2503 = stablehlo.broadcast_in_dim %2502, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2504 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2505 = stablehlo.add %2503, %2504 : tensor<256x1280xf32>
    %2506 = stablehlo.convert %2505 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2507 = stablehlo.reshape %2506 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2508 = stablehlo.add %2507, %2400 : tensor<1x256x1280xbf16>
    %2509 = stablehlo.convert %2508 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2510 = stablehlo.convert %2509 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2511 = stablehlo.reduce(%2510 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2512 = stablehlo.reshape %2511 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2513 = stablehlo.broadcast_in_dim %2512, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2514 = stablehlo.divide %2513, %142 : tensor<1x256x1xf64>
    %2515 = stablehlo.broadcast_in_dim %2510, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2516 = stablehlo.broadcast_in_dim %2514, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2517 = stablehlo.subtract %2515, %2516 : tensor<1x256x1280xf64>
    %2518 = stablehlo.multiply %2517, %2517 : tensor<1x256x1280xf64>
    %2519 = stablehlo.reduce(%2518 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2520 = stablehlo.reshape %2519 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2521 = stablehlo.broadcast_in_dim %2520, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2522 = stablehlo.divide %2521, %142 : tensor<1x256x1xf64>
    %2523 = stablehlo.convert %2522 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2524 = stablehlo.reduce(%2509 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2525 = stablehlo.reshape %2524 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2526 = stablehlo.broadcast_in_dim %2525, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2527 = stablehlo.divide %2526, %158 : tensor<1x256x1xf32>
    %2528 = stablehlo.broadcast_in_dim %2523, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2529 = stablehlo.add %2528, %161 : tensor<1x256x1xf32>
    %2530 = stablehlo.rsqrt %2529 : tensor<1x256x1xf32>
    %2531 = stablehlo.broadcast_in_dim %2509, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2532 = stablehlo.broadcast_in_dim %2527, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2533 = stablehlo.subtract %2531, %2532 : tensor<1x256x1280xf32>
    %2534 = stablehlo.broadcast_in_dim %2533, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2535 = stablehlo.broadcast_in_dim %2530, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2536 = stablehlo.multiply %2534, %2535 : tensor<1x256x1280xf32>
    %2537 = stablehlo.convert %arg53 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2538 = stablehlo.broadcast_in_dim %2536, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2539 = stablehlo.broadcast_in_dim %2537, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2540 = stablehlo.multiply %2538, %2539 : tensor<1x256x1280xf32>
    %2541 = stablehlo.convert %arg54 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2542 = stablehlo.broadcast_in_dim %2540, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2543 = stablehlo.broadcast_in_dim %2541, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2544 = stablehlo.add %2542, %2543 : tensor<1x256x1280xf32>
    %2545 = stablehlo.convert %2544 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2546 = stablehlo.reshape %2545 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2547 = stablehlo.convert %2546 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2548 = stablehlo.dot_general %2547, %arg269, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2549 = stablehlo.broadcast_in_dim %2548, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2550 = stablehlo.multiply %2549, %127 : tensor<256x1280xf32>
    %2551 = stablehlo.broadcast_in_dim %2550, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2552 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2553 = stablehlo.add %2551, %2552 : tensor<256x1280xf32>
    %2554 = stablehlo.convert %2553 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2555 = stablehlo.reshape %2554 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2556 = stablehlo.multiply %2555, %cst_4 : tensor<1x256x1280xbf16>
    %2557 = stablehlo.multiply %2555, %190 : tensor<1x256x1280xbf16>
    %2558 = stablehlo.convert %2557 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2559 = stablehlo.clamp %cst_5, %2558, %cst_6 : tensor<1x256x1280xf32>
    %2560 = stablehlo.multiply %2559, %2559 : tensor<1x256x1280xf32>
    %2561 = stablehlo.multiply %cst_7, %2560 : tensor<1x256x1280xf32>
    %2562 = stablehlo.add %2561, %cst_8 : tensor<1x256x1280xf32>
    %2563 = stablehlo.multiply %2562, %2560 : tensor<1x256x1280xf32>
    %2564 = stablehlo.add %2563, %cst_9 : tensor<1x256x1280xf32>
    %2565 = stablehlo.multiply %2564, %2560 : tensor<1x256x1280xf32>
    %2566 = stablehlo.add %2565, %cst_10 : tensor<1x256x1280xf32>
    %2567 = stablehlo.multiply %2566, %2560 : tensor<1x256x1280xf32>
    %2568 = stablehlo.add %2567, %cst_11 : tensor<1x256x1280xf32>
    %2569 = stablehlo.multiply %2568, %2560 : tensor<1x256x1280xf32>
    %2570 = stablehlo.add %2569, %cst_12 : tensor<1x256x1280xf32>
    %2571 = stablehlo.multiply %2570, %2560 : tensor<1x256x1280xf32>
    %2572 = stablehlo.add %2571, %cst_13 : tensor<1x256x1280xf32>
    %2573 = stablehlo.multiply %cst_14, %2560 : tensor<1x256x1280xf32>
    %2574 = stablehlo.add %2573, %cst_15 : tensor<1x256x1280xf32>
    %2575 = stablehlo.multiply %2574, %2560 : tensor<1x256x1280xf32>
    %2576 = stablehlo.add %2575, %cst_16 : tensor<1x256x1280xf32>
    %2577 = stablehlo.multiply %2576, %2560 : tensor<1x256x1280xf32>
    %2578 = stablehlo.add %2577, %cst_17 : tensor<1x256x1280xf32>
    %2579 = stablehlo.multiply %2578, %2560 : tensor<1x256x1280xf32>
    %2580 = stablehlo.add %2579, %cst_18 : tensor<1x256x1280xf32>
    %2581 = stablehlo.multiply %2559, %2572 : tensor<1x256x1280xf32>
    %2582 = stablehlo.divide %2581, %2580 : tensor<1x256x1280xf32>
    %2583 = stablehlo.clamp %cst_19, %2582, %cst_20 : tensor<1x256x1280xf32>
    %2584 = stablehlo.convert %2583 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2585 = stablehlo.add %2584, %cst_2 : tensor<1x256x1280xbf16>
    %2586 = stablehlo.multiply %2585, %2556 : tensor<1x256x1280xbf16>
    %2587 = stablehlo.reshape %2586 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2588 = stablehlo.convert %2587 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2589 = stablehlo.dot_general %2588, %arg271, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2590 = stablehlo.broadcast_in_dim %2589, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2591 = stablehlo.multiply %2590, %127 : tensor<256x1280xf32>
    %2592 = stablehlo.broadcast_in_dim %2591, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2593 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2594 = stablehlo.add %2592, %2593 : tensor<256x1280xf32>
    %2595 = stablehlo.convert %2594 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2596 = stablehlo.reshape %2595 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2597 = stablehlo.add %2596, %2508 : tensor<1x256x1280xbf16>
    %2598 = stablehlo.convert %2597 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2599 = stablehlo.convert %2598 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2600 = stablehlo.reduce(%2599 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2601 = stablehlo.reshape %2600 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2602 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2603 = stablehlo.divide %2602, %142 : tensor<1x256x1xf64>
    %2604 = stablehlo.broadcast_in_dim %2599, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2605 = stablehlo.broadcast_in_dim %2603, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2606 = stablehlo.subtract %2604, %2605 : tensor<1x256x1280xf64>
    %2607 = stablehlo.multiply %2606, %2606 : tensor<1x256x1280xf64>
    %2608 = stablehlo.reduce(%2607 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2609 = stablehlo.reshape %2608 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2610 = stablehlo.broadcast_in_dim %2609, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2611 = stablehlo.divide %2610, %142 : tensor<1x256x1xf64>
    %2612 = stablehlo.convert %2611 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2613 = stablehlo.reduce(%2598 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2614 = stablehlo.reshape %2613 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2615 = stablehlo.broadcast_in_dim %2614, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2616 = stablehlo.divide %2615, %158 : tensor<1x256x1xf32>
    %2617 = stablehlo.broadcast_in_dim %2612, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2618 = stablehlo.add %2617, %161 : tensor<1x256x1xf32>
    %2619 = stablehlo.rsqrt %2618 : tensor<1x256x1xf32>
    %2620 = stablehlo.broadcast_in_dim %2598, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2621 = stablehlo.broadcast_in_dim %2616, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2622 = stablehlo.subtract %2620, %2621 : tensor<1x256x1280xf32>
    %2623 = stablehlo.broadcast_in_dim %2622, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2624 = stablehlo.broadcast_in_dim %2619, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2625 = stablehlo.multiply %2623, %2624 : tensor<1x256x1280xf32>
    %2626 = stablehlo.convert %arg55 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2627 = stablehlo.broadcast_in_dim %2625, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2628 = stablehlo.broadcast_in_dim %2626, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2629 = stablehlo.multiply %2627, %2628 : tensor<1x256x1280xf32>
    %2630 = stablehlo.convert %arg56 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2631 = stablehlo.broadcast_in_dim %2629, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2632 = stablehlo.broadcast_in_dim %2630, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2633 = stablehlo.add %2631, %2632 : tensor<1x256x1280xf32>
    %2634 = stablehlo.convert %2633 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2635 = stablehlo.reshape %2634 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2636 = stablehlo.convert %2635 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2637 = stablehlo.dot_general %2636, %arg273, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2638 = stablehlo.broadcast_in_dim %2637, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2639 = stablehlo.multiply %2638, %273 : tensor<256x256xf32>
    %2640 = stablehlo.broadcast_in_dim %2639, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2641 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2642 = stablehlo.add %2640, %2641 : tensor<256x256xf32>
    %2643 = stablehlo.convert %2642 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2644 = stablehlo.reshape %2643 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2645 = stablehlo.dot_general %2636, %arg275, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2646 = stablehlo.broadcast_in_dim %2645, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2647 = stablehlo.multiply %2646, %273 : tensor<256x256xf32>
    %2648 = stablehlo.broadcast_in_dim %2647, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2649 = stablehlo.broadcast_in_dim %arg276, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2650 = stablehlo.add %2648, %2649 : tensor<256x256xf32>
    %2651 = stablehlo.convert %2650 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2652 = stablehlo.reshape %2651 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2653 = stablehlo.dot_general %2636, %arg277, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2654 = stablehlo.broadcast_in_dim %2653, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2655 = stablehlo.multiply %2654, %127 : tensor<256x1280xf32>
    %2656 = stablehlo.broadcast_in_dim %2655, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2657 = stablehlo.broadcast_in_dim %arg278, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2658 = stablehlo.add %2656, %2657 : tensor<256x1280xf32>
    %2659 = stablehlo.convert %2658 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2660 = stablehlo.reshape %2659 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2661 = stablehlo.reshape %2644 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2662 = stablehlo.transpose %2661, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2663 = stablehlo.reshape %2652 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2664 = stablehlo.transpose %2663, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2665 = stablehlo.reshape %2660 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %2666 = stablehlo.transpose %2665, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2667 = stablehlo.transpose %2664, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %2668 = stablehlo.reshape %2662 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2669 = stablehlo.reshape %2667 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2670 = stablehlo.broadcast_in_dim %2669, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2671 = stablehlo.dot_general %2668, %2670, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %2672 = stablehlo.reshape %2671 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2673 = stablehlo.broadcast_in_dim %2672, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2674 = stablehlo.divide %2673, %309 : tensor<1x8x256x256xbf16>
    %2675 = stablehlo.convert %2674 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %2676 = stablehlo.reduce(%2675 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2677 = stablehlo.reshape %2676 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2678 = stablehlo.broadcast_in_dim %2675, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2679 = stablehlo.broadcast_in_dim %2677, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2680 = stablehlo.subtract %2678, %2679 : tensor<1x8x256x256xf32>
    %2681 = stablehlo.exponential %2680 : tensor<1x8x256x256xf32>
    %2682 = stablehlo.reduce(%2681 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2683 = stablehlo.reshape %2682 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2684 = stablehlo.broadcast_in_dim %2681, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2685 = stablehlo.broadcast_in_dim %2683, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2686 = stablehlo.divide %2684, %2685 : tensor<1x8x256x256xf32>
    %2687 = stablehlo.convert %2686 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %2688 = stablehlo.reshape %2687 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %2689 = stablehlo.reshape %2666 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2690 = stablehlo.broadcast_in_dim %2689, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2691 = stablehlo.dot_general %2688, %2690, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2692 = stablehlo.reshape %2691 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2693 = stablehlo.transpose %2692, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %2694 = stablehlo.reshape %2693 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %2695 = stablehlo.reshape %2694 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2696 = stablehlo.convert %2695 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2697 = stablehlo.dot_general %2696, %arg279, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2698 = stablehlo.broadcast_in_dim %2697, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2699 = stablehlo.multiply %2698, %127 : tensor<256x1280xf32>
    %2700 = stablehlo.broadcast_in_dim %2699, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2701 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2702 = stablehlo.add %2700, %2701 : tensor<256x1280xf32>
    %2703 = stablehlo.convert %2702 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2704 = stablehlo.reshape %2703 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2705 = stablehlo.add %2704, %2597 : tensor<1x256x1280xbf16>
    %2706 = stablehlo.convert %2705 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2707 = stablehlo.convert %2706 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2708 = stablehlo.reduce(%2707 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2709 = stablehlo.reshape %2708 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2710 = stablehlo.broadcast_in_dim %2709, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2711 = stablehlo.divide %2710, %142 : tensor<1x256x1xf64>
    %2712 = stablehlo.broadcast_in_dim %2707, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2713 = stablehlo.broadcast_in_dim %2711, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2714 = stablehlo.subtract %2712, %2713 : tensor<1x256x1280xf64>
    %2715 = stablehlo.multiply %2714, %2714 : tensor<1x256x1280xf64>
    %2716 = stablehlo.reduce(%2715 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2717 = stablehlo.reshape %2716 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2718 = stablehlo.broadcast_in_dim %2717, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2719 = stablehlo.divide %2718, %142 : tensor<1x256x1xf64>
    %2720 = stablehlo.convert %2719 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2721 = stablehlo.reduce(%2706 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2722 = stablehlo.reshape %2721 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2723 = stablehlo.broadcast_in_dim %2722, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2724 = stablehlo.divide %2723, %158 : tensor<1x256x1xf32>
    %2725 = stablehlo.broadcast_in_dim %2720, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2726 = stablehlo.add %2725, %161 : tensor<1x256x1xf32>
    %2727 = stablehlo.rsqrt %2726 : tensor<1x256x1xf32>
    %2728 = stablehlo.broadcast_in_dim %2706, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2729 = stablehlo.broadcast_in_dim %2724, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2730 = stablehlo.subtract %2728, %2729 : tensor<1x256x1280xf32>
    %2731 = stablehlo.broadcast_in_dim %2730, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2732 = stablehlo.broadcast_in_dim %2727, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2733 = stablehlo.multiply %2731, %2732 : tensor<1x256x1280xf32>
    %2734 = stablehlo.convert %arg57 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2735 = stablehlo.broadcast_in_dim %2733, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2736 = stablehlo.broadcast_in_dim %2734, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2737 = stablehlo.multiply %2735, %2736 : tensor<1x256x1280xf32>
    %2738 = stablehlo.convert %arg58 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2739 = stablehlo.broadcast_in_dim %2737, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2740 = stablehlo.broadcast_in_dim %2738, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2741 = stablehlo.add %2739, %2740 : tensor<1x256x1280xf32>
    %2742 = stablehlo.convert %2741 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2743 = stablehlo.reshape %2742 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2744 = stablehlo.convert %2743 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2745 = stablehlo.dot_general %2744, %arg281, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2746 = stablehlo.broadcast_in_dim %2745, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2747 = stablehlo.multiply %2746, %127 : tensor<256x1280xf32>
    %2748 = stablehlo.broadcast_in_dim %2747, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2749 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2750 = stablehlo.add %2748, %2749 : tensor<256x1280xf32>
    %2751 = stablehlo.convert %2750 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2752 = stablehlo.reshape %2751 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2753 = stablehlo.multiply %2752, %cst_4 : tensor<1x256x1280xbf16>
    %2754 = stablehlo.multiply %2752, %190 : tensor<1x256x1280xbf16>
    %2755 = stablehlo.convert %2754 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2756 = stablehlo.clamp %cst_5, %2755, %cst_6 : tensor<1x256x1280xf32>
    %2757 = stablehlo.multiply %2756, %2756 : tensor<1x256x1280xf32>
    %2758 = stablehlo.multiply %cst_7, %2757 : tensor<1x256x1280xf32>
    %2759 = stablehlo.add %2758, %cst_8 : tensor<1x256x1280xf32>
    %2760 = stablehlo.multiply %2759, %2757 : tensor<1x256x1280xf32>
    %2761 = stablehlo.add %2760, %cst_9 : tensor<1x256x1280xf32>
    %2762 = stablehlo.multiply %2761, %2757 : tensor<1x256x1280xf32>
    %2763 = stablehlo.add %2762, %cst_10 : tensor<1x256x1280xf32>
    %2764 = stablehlo.multiply %2763, %2757 : tensor<1x256x1280xf32>
    %2765 = stablehlo.add %2764, %cst_11 : tensor<1x256x1280xf32>
    %2766 = stablehlo.multiply %2765, %2757 : tensor<1x256x1280xf32>
    %2767 = stablehlo.add %2766, %cst_12 : tensor<1x256x1280xf32>
    %2768 = stablehlo.multiply %2767, %2757 : tensor<1x256x1280xf32>
    %2769 = stablehlo.add %2768, %cst_13 : tensor<1x256x1280xf32>
    %2770 = stablehlo.multiply %cst_14, %2757 : tensor<1x256x1280xf32>
    %2771 = stablehlo.add %2770, %cst_15 : tensor<1x256x1280xf32>
    %2772 = stablehlo.multiply %2771, %2757 : tensor<1x256x1280xf32>
    %2773 = stablehlo.add %2772, %cst_16 : tensor<1x256x1280xf32>
    %2774 = stablehlo.multiply %2773, %2757 : tensor<1x256x1280xf32>
    %2775 = stablehlo.add %2774, %cst_17 : tensor<1x256x1280xf32>
    %2776 = stablehlo.multiply %2775, %2757 : tensor<1x256x1280xf32>
    %2777 = stablehlo.add %2776, %cst_18 : tensor<1x256x1280xf32>
    %2778 = stablehlo.multiply %2756, %2769 : tensor<1x256x1280xf32>
    %2779 = stablehlo.divide %2778, %2777 : tensor<1x256x1280xf32>
    %2780 = stablehlo.clamp %cst_19, %2779, %cst_20 : tensor<1x256x1280xf32>
    %2781 = stablehlo.convert %2780 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2782 = stablehlo.add %2781, %cst_2 : tensor<1x256x1280xbf16>
    %2783 = stablehlo.multiply %2782, %2753 : tensor<1x256x1280xbf16>
    %2784 = stablehlo.reshape %2783 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2785 = stablehlo.convert %2784 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2786 = stablehlo.dot_general %2785, %arg283, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2787 = stablehlo.broadcast_in_dim %2786, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2788 = stablehlo.multiply %2787, %127 : tensor<256x1280xf32>
    %2789 = stablehlo.broadcast_in_dim %2788, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2790 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2791 = stablehlo.add %2789, %2790 : tensor<256x1280xf32>
    %2792 = stablehlo.convert %2791 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2793 = stablehlo.reshape %2792 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2794 = stablehlo.add %2793, %2705 : tensor<1x256x1280xbf16>
    %2795 = stablehlo.convert %2794 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2796 = stablehlo.convert %2795 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2797 = stablehlo.reduce(%2796 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2798 = stablehlo.reshape %2797 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2799 = stablehlo.broadcast_in_dim %2798, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2800 = stablehlo.divide %2799, %142 : tensor<1x256x1xf64>
    %2801 = stablehlo.broadcast_in_dim %2796, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2802 = stablehlo.broadcast_in_dim %2800, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2803 = stablehlo.subtract %2801, %2802 : tensor<1x256x1280xf64>
    %2804 = stablehlo.multiply %2803, %2803 : tensor<1x256x1280xf64>
    %2805 = stablehlo.reduce(%2804 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2806 = stablehlo.reshape %2805 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2807 = stablehlo.broadcast_in_dim %2806, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2808 = stablehlo.divide %2807, %142 : tensor<1x256x1xf64>
    %2809 = stablehlo.convert %2808 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2810 = stablehlo.reduce(%2795 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2811 = stablehlo.reshape %2810 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2812 = stablehlo.broadcast_in_dim %2811, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2813 = stablehlo.divide %2812, %158 : tensor<1x256x1xf32>
    %2814 = stablehlo.broadcast_in_dim %2809, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2815 = stablehlo.add %2814, %161 : tensor<1x256x1xf32>
    %2816 = stablehlo.rsqrt %2815 : tensor<1x256x1xf32>
    %2817 = stablehlo.broadcast_in_dim %2795, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2818 = stablehlo.broadcast_in_dim %2813, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2819 = stablehlo.subtract %2817, %2818 : tensor<1x256x1280xf32>
    %2820 = stablehlo.broadcast_in_dim %2819, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2821 = stablehlo.broadcast_in_dim %2816, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2822 = stablehlo.multiply %2820, %2821 : tensor<1x256x1280xf32>
    %2823 = stablehlo.convert %arg59 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2824 = stablehlo.broadcast_in_dim %2822, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2825 = stablehlo.broadcast_in_dim %2823, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2826 = stablehlo.multiply %2824, %2825 : tensor<1x256x1280xf32>
    %2827 = stablehlo.convert %arg60 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2828 = stablehlo.broadcast_in_dim %2826, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2829 = stablehlo.broadcast_in_dim %2827, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2830 = stablehlo.add %2828, %2829 : tensor<1x256x1280xf32>
    %2831 = stablehlo.convert %2830 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2832 = stablehlo.reshape %2831 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2833 = stablehlo.convert %2832 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2834 = stablehlo.dot_general %2833, %arg285, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2835 = stablehlo.broadcast_in_dim %2834, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2836 = stablehlo.multiply %2835, %273 : tensor<256x256xf32>
    %2837 = stablehlo.broadcast_in_dim %2836, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2838 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2839 = stablehlo.add %2837, %2838 : tensor<256x256xf32>
    %2840 = stablehlo.convert %2839 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2841 = stablehlo.reshape %2840 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2842 = stablehlo.dot_general %2833, %arg287, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %2843 = stablehlo.broadcast_in_dim %2842, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2844 = stablehlo.multiply %2843, %273 : tensor<256x256xf32>
    %2845 = stablehlo.broadcast_in_dim %2844, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2846 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %2847 = stablehlo.add %2845, %2846 : tensor<256x256xf32>
    %2848 = stablehlo.convert %2847 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %2849 = stablehlo.reshape %2848 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %2850 = stablehlo.dot_general %2833, %arg289, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2851 = stablehlo.broadcast_in_dim %2850, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2852 = stablehlo.multiply %2851, %127 : tensor<256x1280xf32>
    %2853 = stablehlo.broadcast_in_dim %2852, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2854 = stablehlo.broadcast_in_dim %arg290, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2855 = stablehlo.add %2853, %2854 : tensor<256x1280xf32>
    %2856 = stablehlo.convert %2855 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2857 = stablehlo.reshape %2856 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2858 = stablehlo.reshape %2841 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2859 = stablehlo.transpose %2858, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2860 = stablehlo.reshape %2849 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %2861 = stablehlo.transpose %2860, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %2862 = stablehlo.reshape %2857 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %2863 = stablehlo.transpose %2862, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2864 = stablehlo.transpose %2861, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %2865 = stablehlo.reshape %2859 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %2866 = stablehlo.reshape %2864 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2867 = stablehlo.broadcast_in_dim %2866, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %2868 = stablehlo.dot_general %2865, %2867, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %2869 = stablehlo.reshape %2868 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2870 = stablehlo.broadcast_in_dim %2869, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %2871 = stablehlo.divide %2870, %309 : tensor<1x8x256x256xbf16>
    %2872 = stablehlo.convert %2871 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %2873 = stablehlo.reduce(%2872 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2874 = stablehlo.reshape %2873 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2875 = stablehlo.broadcast_in_dim %2872, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2876 = stablehlo.broadcast_in_dim %2874, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2877 = stablehlo.subtract %2875, %2876 : tensor<1x8x256x256xf32>
    %2878 = stablehlo.exponential %2877 : tensor<1x8x256x256xf32>
    %2879 = stablehlo.reduce(%2878 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %2880 = stablehlo.reshape %2879 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %2881 = stablehlo.broadcast_in_dim %2878, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %2882 = stablehlo.broadcast_in_dim %2880, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %2883 = stablehlo.divide %2881, %2882 : tensor<1x8x256x256xf32>
    %2884 = stablehlo.convert %2883 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %2885 = stablehlo.reshape %2884 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %2886 = stablehlo.reshape %2863 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2887 = stablehlo.broadcast_in_dim %2886, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2888 = stablehlo.dot_general %2885, %2887, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %2889 = stablehlo.reshape %2888 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %2890 = stablehlo.transpose %2889, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %2891 = stablehlo.reshape %2890 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %2892 = stablehlo.reshape %2891 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2893 = stablehlo.convert %2892 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2894 = stablehlo.dot_general %2893, %arg291, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2895 = stablehlo.broadcast_in_dim %2894, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2896 = stablehlo.multiply %2895, %127 : tensor<256x1280xf32>
    %2897 = stablehlo.broadcast_in_dim %2896, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2898 = stablehlo.broadcast_in_dim %arg292, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2899 = stablehlo.add %2897, %2898 : tensor<256x1280xf32>
    %2900 = stablehlo.convert %2899 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2901 = stablehlo.reshape %2900 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2902 = stablehlo.add %2901, %2794 : tensor<1x256x1280xbf16>
    %2903 = stablehlo.convert %2902 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2904 = stablehlo.convert %2903 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2905 = stablehlo.reduce(%2904 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2906 = stablehlo.reshape %2905 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2907 = stablehlo.broadcast_in_dim %2906, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2908 = stablehlo.divide %2907, %142 : tensor<1x256x1xf64>
    %2909 = stablehlo.broadcast_in_dim %2904, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2910 = stablehlo.broadcast_in_dim %2908, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %2911 = stablehlo.subtract %2909, %2910 : tensor<1x256x1280xf64>
    %2912 = stablehlo.multiply %2911, %2911 : tensor<1x256x1280xf64>
    %2913 = stablehlo.reduce(%2912 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2914 = stablehlo.reshape %2913 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2915 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2916 = stablehlo.divide %2915, %142 : tensor<1x256x1xf64>
    %2917 = stablehlo.convert %2916 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %2918 = stablehlo.reduce(%2903 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %2919 = stablehlo.reshape %2918 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %2920 = stablehlo.broadcast_in_dim %2919, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2921 = stablehlo.divide %2920, %158 : tensor<1x256x1xf32>
    %2922 = stablehlo.broadcast_in_dim %2917, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2923 = stablehlo.add %2922, %161 : tensor<1x256x1xf32>
    %2924 = stablehlo.rsqrt %2923 : tensor<1x256x1xf32>
    %2925 = stablehlo.broadcast_in_dim %2903, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2926 = stablehlo.broadcast_in_dim %2921, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2927 = stablehlo.subtract %2925, %2926 : tensor<1x256x1280xf32>
    %2928 = stablehlo.broadcast_in_dim %2927, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2929 = stablehlo.broadcast_in_dim %2924, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2930 = stablehlo.multiply %2928, %2929 : tensor<1x256x1280xf32>
    %2931 = stablehlo.convert %arg61 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2932 = stablehlo.broadcast_in_dim %2930, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2933 = stablehlo.broadcast_in_dim %2931, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2934 = stablehlo.multiply %2932, %2933 : tensor<1x256x1280xf32>
    %2935 = stablehlo.convert %arg62 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %2936 = stablehlo.broadcast_in_dim %2934, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2937 = stablehlo.broadcast_in_dim %2935, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %2938 = stablehlo.add %2936, %2937 : tensor<1x256x1280xf32>
    %2939 = stablehlo.convert %2938 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2940 = stablehlo.reshape %2939 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2941 = stablehlo.convert %2940 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2942 = stablehlo.dot_general %2941, %arg293, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2943 = stablehlo.broadcast_in_dim %2942, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2944 = stablehlo.multiply %2943, %127 : tensor<256x1280xf32>
    %2945 = stablehlo.broadcast_in_dim %2944, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2946 = stablehlo.broadcast_in_dim %arg294, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2947 = stablehlo.add %2945, %2946 : tensor<256x1280xf32>
    %2948 = stablehlo.convert %2947 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2949 = stablehlo.reshape %2948 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2950 = stablehlo.multiply %2949, %cst_4 : tensor<1x256x1280xbf16>
    %2951 = stablehlo.multiply %2949, %190 : tensor<1x256x1280xbf16>
    %2952 = stablehlo.convert %2951 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2953 = stablehlo.clamp %cst_5, %2952, %cst_6 : tensor<1x256x1280xf32>
    %2954 = stablehlo.multiply %2953, %2953 : tensor<1x256x1280xf32>
    %2955 = stablehlo.multiply %cst_7, %2954 : tensor<1x256x1280xf32>
    %2956 = stablehlo.add %2955, %cst_8 : tensor<1x256x1280xf32>
    %2957 = stablehlo.multiply %2956, %2954 : tensor<1x256x1280xf32>
    %2958 = stablehlo.add %2957, %cst_9 : tensor<1x256x1280xf32>
    %2959 = stablehlo.multiply %2958, %2954 : tensor<1x256x1280xf32>
    %2960 = stablehlo.add %2959, %cst_10 : tensor<1x256x1280xf32>
    %2961 = stablehlo.multiply %2960, %2954 : tensor<1x256x1280xf32>
    %2962 = stablehlo.add %2961, %cst_11 : tensor<1x256x1280xf32>
    %2963 = stablehlo.multiply %2962, %2954 : tensor<1x256x1280xf32>
    %2964 = stablehlo.add %2963, %cst_12 : tensor<1x256x1280xf32>
    %2965 = stablehlo.multiply %2964, %2954 : tensor<1x256x1280xf32>
    %2966 = stablehlo.add %2965, %cst_13 : tensor<1x256x1280xf32>
    %2967 = stablehlo.multiply %cst_14, %2954 : tensor<1x256x1280xf32>
    %2968 = stablehlo.add %2967, %cst_15 : tensor<1x256x1280xf32>
    %2969 = stablehlo.multiply %2968, %2954 : tensor<1x256x1280xf32>
    %2970 = stablehlo.add %2969, %cst_16 : tensor<1x256x1280xf32>
    %2971 = stablehlo.multiply %2970, %2954 : tensor<1x256x1280xf32>
    %2972 = stablehlo.add %2971, %cst_17 : tensor<1x256x1280xf32>
    %2973 = stablehlo.multiply %2972, %2954 : tensor<1x256x1280xf32>
    %2974 = stablehlo.add %2973, %cst_18 : tensor<1x256x1280xf32>
    %2975 = stablehlo.multiply %2953, %2966 : tensor<1x256x1280xf32>
    %2976 = stablehlo.divide %2975, %2974 : tensor<1x256x1280xf32>
    %2977 = stablehlo.clamp %cst_19, %2976, %cst_20 : tensor<1x256x1280xf32>
    %2978 = stablehlo.convert %2977 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %2979 = stablehlo.add %2978, %cst_2 : tensor<1x256x1280xbf16>
    %2980 = stablehlo.multiply %2979, %2950 : tensor<1x256x1280xbf16>
    %2981 = stablehlo.reshape %2980 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %2982 = stablehlo.convert %2981 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %2983 = stablehlo.dot_general %2982, %arg295, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %2984 = stablehlo.broadcast_in_dim %2983, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2985 = stablehlo.multiply %2984, %127 : tensor<256x1280xf32>
    %2986 = stablehlo.broadcast_in_dim %2985, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2987 = stablehlo.broadcast_in_dim %arg296, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %2988 = stablehlo.add %2986, %2987 : tensor<256x1280xf32>
    %2989 = stablehlo.convert %2988 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %2990 = stablehlo.reshape %2989 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %2991 = stablehlo.add %2990, %2902 : tensor<1x256x1280xbf16>
    %2992 = stablehlo.convert %2991 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %2993 = stablehlo.convert %2992 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %2994 = stablehlo.reduce(%2993 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %2995 = stablehlo.reshape %2994 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %2996 = stablehlo.broadcast_in_dim %2995, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %2997 = stablehlo.divide %2996, %142 : tensor<1x256x1xf64>
    %2998 = stablehlo.broadcast_in_dim %2993, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %2999 = stablehlo.broadcast_in_dim %2997, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3000 = stablehlo.subtract %2998, %2999 : tensor<1x256x1280xf64>
    %3001 = stablehlo.multiply %3000, %3000 : tensor<1x256x1280xf64>
    %3002 = stablehlo.reduce(%3001 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3003 = stablehlo.reshape %3002 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3004 = stablehlo.broadcast_in_dim %3003, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3005 = stablehlo.divide %3004, %142 : tensor<1x256x1xf64>
    %3006 = stablehlo.convert %3005 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3007 = stablehlo.reduce(%2992 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3008 = stablehlo.reshape %3007 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3009 = stablehlo.broadcast_in_dim %3008, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3010 = stablehlo.divide %3009, %158 : tensor<1x256x1xf32>
    %3011 = stablehlo.broadcast_in_dim %3006, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3012 = stablehlo.add %3011, %161 : tensor<1x256x1xf32>
    %3013 = stablehlo.rsqrt %3012 : tensor<1x256x1xf32>
    %3014 = stablehlo.broadcast_in_dim %2992, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3015 = stablehlo.broadcast_in_dim %3010, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3016 = stablehlo.subtract %3014, %3015 : tensor<1x256x1280xf32>
    %3017 = stablehlo.broadcast_in_dim %3016, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3018 = stablehlo.broadcast_in_dim %3013, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3019 = stablehlo.multiply %3017, %3018 : tensor<1x256x1280xf32>
    %3020 = stablehlo.convert %arg63 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3021 = stablehlo.broadcast_in_dim %3019, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3022 = stablehlo.broadcast_in_dim %3020, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3023 = stablehlo.multiply %3021, %3022 : tensor<1x256x1280xf32>
    %3024 = stablehlo.convert %arg64 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3025 = stablehlo.broadcast_in_dim %3023, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3026 = stablehlo.broadcast_in_dim %3024, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3027 = stablehlo.add %3025, %3026 : tensor<1x256x1280xf32>
    %3028 = stablehlo.convert %3027 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3029 = stablehlo.reshape %3028 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3030 = stablehlo.convert %3029 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3031 = stablehlo.dot_general %3030, %arg297, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3032 = stablehlo.broadcast_in_dim %3031, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3033 = stablehlo.multiply %3032, %273 : tensor<256x256xf32>
    %3034 = stablehlo.broadcast_in_dim %3033, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3035 = stablehlo.broadcast_in_dim %arg298, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3036 = stablehlo.add %3034, %3035 : tensor<256x256xf32>
    %3037 = stablehlo.convert %3036 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3038 = stablehlo.reshape %3037 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3039 = stablehlo.dot_general %3030, %arg299, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3040 = stablehlo.broadcast_in_dim %3039, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3041 = stablehlo.multiply %3040, %273 : tensor<256x256xf32>
    %3042 = stablehlo.broadcast_in_dim %3041, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3043 = stablehlo.broadcast_in_dim %arg300, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3044 = stablehlo.add %3042, %3043 : tensor<256x256xf32>
    %3045 = stablehlo.convert %3044 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3046 = stablehlo.reshape %3045 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3047 = stablehlo.dot_general %3030, %arg301, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3048 = stablehlo.broadcast_in_dim %3047, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3049 = stablehlo.multiply %3048, %127 : tensor<256x1280xf32>
    %3050 = stablehlo.broadcast_in_dim %3049, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3051 = stablehlo.broadcast_in_dim %arg302, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3052 = stablehlo.add %3050, %3051 : tensor<256x1280xf32>
    %3053 = stablehlo.convert %3052 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3054 = stablehlo.reshape %3053 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3055 = stablehlo.reshape %3038 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3056 = stablehlo.transpose %3055, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3057 = stablehlo.reshape %3046 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3058 = stablehlo.transpose %3057, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3059 = stablehlo.reshape %3054 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %3060 = stablehlo.transpose %3059, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3061 = stablehlo.transpose %3058, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %3062 = stablehlo.reshape %3056 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %3063 = stablehlo.reshape %3061 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3064 = stablehlo.broadcast_in_dim %3063, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3065 = stablehlo.dot_general %3062, %3064, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %3066 = stablehlo.reshape %3065 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3067 = stablehlo.broadcast_in_dim %3066, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3068 = stablehlo.divide %3067, %309 : tensor<1x8x256x256xbf16>
    %3069 = stablehlo.convert %3068 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %3070 = stablehlo.reduce(%3069 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3071 = stablehlo.reshape %3070 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3072 = stablehlo.broadcast_in_dim %3069, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3073 = stablehlo.broadcast_in_dim %3071, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3074 = stablehlo.subtract %3072, %3073 : tensor<1x8x256x256xf32>
    %3075 = stablehlo.exponential %3074 : tensor<1x8x256x256xf32>
    %3076 = stablehlo.reduce(%3075 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3077 = stablehlo.reshape %3076 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3078 = stablehlo.broadcast_in_dim %3075, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3079 = stablehlo.broadcast_in_dim %3077, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3080 = stablehlo.divide %3078, %3079 : tensor<1x8x256x256xf32>
    %3081 = stablehlo.convert %3080 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %3082 = stablehlo.reshape %3081 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %3083 = stablehlo.reshape %3060 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3084 = stablehlo.broadcast_in_dim %3083, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3085 = stablehlo.dot_general %3082, %3084, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3086 = stablehlo.reshape %3085 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3087 = stablehlo.transpose %3086, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %3088 = stablehlo.reshape %3087 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %3089 = stablehlo.reshape %3088 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3090 = stablehlo.convert %3089 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3091 = stablehlo.dot_general %3090, %arg303, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3092 = stablehlo.broadcast_in_dim %3091, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3093 = stablehlo.multiply %3092, %127 : tensor<256x1280xf32>
    %3094 = stablehlo.broadcast_in_dim %3093, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3095 = stablehlo.broadcast_in_dim %arg304, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3096 = stablehlo.add %3094, %3095 : tensor<256x1280xf32>
    %3097 = stablehlo.convert %3096 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3098 = stablehlo.reshape %3097 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3099 = stablehlo.add %3098, %2991 : tensor<1x256x1280xbf16>
    %3100 = stablehlo.convert %3099 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3101 = stablehlo.convert %3100 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3102 = stablehlo.reduce(%3101 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3103 = stablehlo.reshape %3102 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3104 = stablehlo.broadcast_in_dim %3103, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3105 = stablehlo.divide %3104, %142 : tensor<1x256x1xf64>
    %3106 = stablehlo.broadcast_in_dim %3101, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3107 = stablehlo.broadcast_in_dim %3105, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3108 = stablehlo.subtract %3106, %3107 : tensor<1x256x1280xf64>
    %3109 = stablehlo.multiply %3108, %3108 : tensor<1x256x1280xf64>
    %3110 = stablehlo.reduce(%3109 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3111 = stablehlo.reshape %3110 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3112 = stablehlo.broadcast_in_dim %3111, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3113 = stablehlo.divide %3112, %142 : tensor<1x256x1xf64>
    %3114 = stablehlo.convert %3113 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3115 = stablehlo.reduce(%3100 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3116 = stablehlo.reshape %3115 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3117 = stablehlo.broadcast_in_dim %3116, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3118 = stablehlo.divide %3117, %158 : tensor<1x256x1xf32>
    %3119 = stablehlo.broadcast_in_dim %3114, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3120 = stablehlo.add %3119, %161 : tensor<1x256x1xf32>
    %3121 = stablehlo.rsqrt %3120 : tensor<1x256x1xf32>
    %3122 = stablehlo.broadcast_in_dim %3100, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3123 = stablehlo.broadcast_in_dim %3118, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3124 = stablehlo.subtract %3122, %3123 : tensor<1x256x1280xf32>
    %3125 = stablehlo.broadcast_in_dim %3124, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3126 = stablehlo.broadcast_in_dim %3121, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3127 = stablehlo.multiply %3125, %3126 : tensor<1x256x1280xf32>
    %3128 = stablehlo.convert %arg65 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3129 = stablehlo.broadcast_in_dim %3127, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3130 = stablehlo.broadcast_in_dim %3128, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3131 = stablehlo.multiply %3129, %3130 : tensor<1x256x1280xf32>
    %3132 = stablehlo.convert %arg66 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3133 = stablehlo.broadcast_in_dim %3131, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3134 = stablehlo.broadcast_in_dim %3132, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3135 = stablehlo.add %3133, %3134 : tensor<1x256x1280xf32>
    %3136 = stablehlo.convert %3135 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3137 = stablehlo.reshape %3136 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3138 = stablehlo.convert %3137 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3139 = stablehlo.dot_general %3138, %arg305, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3140 = stablehlo.broadcast_in_dim %3139, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3141 = stablehlo.multiply %3140, %127 : tensor<256x1280xf32>
    %3142 = stablehlo.broadcast_in_dim %3141, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3143 = stablehlo.broadcast_in_dim %arg306, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3144 = stablehlo.add %3142, %3143 : tensor<256x1280xf32>
    %3145 = stablehlo.convert %3144 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3146 = stablehlo.reshape %3145 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3147 = stablehlo.multiply %3146, %cst_4 : tensor<1x256x1280xbf16>
    %3148 = stablehlo.multiply %3146, %190 : tensor<1x256x1280xbf16>
    %3149 = stablehlo.convert %3148 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3150 = stablehlo.clamp %cst_5, %3149, %cst_6 : tensor<1x256x1280xf32>
    %3151 = stablehlo.multiply %3150, %3150 : tensor<1x256x1280xf32>
    %3152 = stablehlo.multiply %cst_7, %3151 : tensor<1x256x1280xf32>
    %3153 = stablehlo.add %3152, %cst_8 : tensor<1x256x1280xf32>
    %3154 = stablehlo.multiply %3153, %3151 : tensor<1x256x1280xf32>
    %3155 = stablehlo.add %3154, %cst_9 : tensor<1x256x1280xf32>
    %3156 = stablehlo.multiply %3155, %3151 : tensor<1x256x1280xf32>
    %3157 = stablehlo.add %3156, %cst_10 : tensor<1x256x1280xf32>
    %3158 = stablehlo.multiply %3157, %3151 : tensor<1x256x1280xf32>
    %3159 = stablehlo.add %3158, %cst_11 : tensor<1x256x1280xf32>
    %3160 = stablehlo.multiply %3159, %3151 : tensor<1x256x1280xf32>
    %3161 = stablehlo.add %3160, %cst_12 : tensor<1x256x1280xf32>
    %3162 = stablehlo.multiply %3161, %3151 : tensor<1x256x1280xf32>
    %3163 = stablehlo.add %3162, %cst_13 : tensor<1x256x1280xf32>
    %3164 = stablehlo.multiply %cst_14, %3151 : tensor<1x256x1280xf32>
    %3165 = stablehlo.add %3164, %cst_15 : tensor<1x256x1280xf32>
    %3166 = stablehlo.multiply %3165, %3151 : tensor<1x256x1280xf32>
    %3167 = stablehlo.add %3166, %cst_16 : tensor<1x256x1280xf32>
    %3168 = stablehlo.multiply %3167, %3151 : tensor<1x256x1280xf32>
    %3169 = stablehlo.add %3168, %cst_17 : tensor<1x256x1280xf32>
    %3170 = stablehlo.multiply %3169, %3151 : tensor<1x256x1280xf32>
    %3171 = stablehlo.add %3170, %cst_18 : tensor<1x256x1280xf32>
    %3172 = stablehlo.multiply %3150, %3163 : tensor<1x256x1280xf32>
    %3173 = stablehlo.divide %3172, %3171 : tensor<1x256x1280xf32>
    %3174 = stablehlo.clamp %cst_19, %3173, %cst_20 : tensor<1x256x1280xf32>
    %3175 = stablehlo.convert %3174 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3176 = stablehlo.add %3175, %cst_2 : tensor<1x256x1280xbf16>
    %3177 = stablehlo.multiply %3176, %3147 : tensor<1x256x1280xbf16>
    %3178 = stablehlo.reshape %3177 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3179 = stablehlo.convert %3178 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3180 = stablehlo.dot_general %3179, %arg307, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3181 = stablehlo.broadcast_in_dim %3180, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3182 = stablehlo.multiply %3181, %127 : tensor<256x1280xf32>
    %3183 = stablehlo.broadcast_in_dim %3182, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3184 = stablehlo.broadcast_in_dim %arg308, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3185 = stablehlo.add %3183, %3184 : tensor<256x1280xf32>
    %3186 = stablehlo.convert %3185 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3187 = stablehlo.reshape %3186 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3188 = stablehlo.add %3187, %3099 : tensor<1x256x1280xbf16>
    %3189 = stablehlo.convert %3188 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3190 = stablehlo.convert %3189 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3191 = stablehlo.reduce(%3190 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3192 = stablehlo.reshape %3191 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3193 = stablehlo.broadcast_in_dim %3192, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3194 = stablehlo.divide %3193, %142 : tensor<1x256x1xf64>
    %3195 = stablehlo.broadcast_in_dim %3190, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3196 = stablehlo.broadcast_in_dim %3194, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3197 = stablehlo.subtract %3195, %3196 : tensor<1x256x1280xf64>
    %3198 = stablehlo.multiply %3197, %3197 : tensor<1x256x1280xf64>
    %3199 = stablehlo.reduce(%3198 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3200 = stablehlo.reshape %3199 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3201 = stablehlo.broadcast_in_dim %3200, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3202 = stablehlo.divide %3201, %142 : tensor<1x256x1xf64>
    %3203 = stablehlo.convert %3202 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3204 = stablehlo.reduce(%3189 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3205 = stablehlo.reshape %3204 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3206 = stablehlo.broadcast_in_dim %3205, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3207 = stablehlo.divide %3206, %158 : tensor<1x256x1xf32>
    %3208 = stablehlo.broadcast_in_dim %3203, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3209 = stablehlo.add %3208, %161 : tensor<1x256x1xf32>
    %3210 = stablehlo.rsqrt %3209 : tensor<1x256x1xf32>
    %3211 = stablehlo.broadcast_in_dim %3189, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3212 = stablehlo.broadcast_in_dim %3207, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3213 = stablehlo.subtract %3211, %3212 : tensor<1x256x1280xf32>
    %3214 = stablehlo.broadcast_in_dim %3213, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3215 = stablehlo.broadcast_in_dim %3210, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3216 = stablehlo.multiply %3214, %3215 : tensor<1x256x1280xf32>
    %3217 = stablehlo.convert %arg67 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3218 = stablehlo.broadcast_in_dim %3216, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3219 = stablehlo.broadcast_in_dim %3217, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3220 = stablehlo.multiply %3218, %3219 : tensor<1x256x1280xf32>
    %3221 = stablehlo.convert %arg68 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3222 = stablehlo.broadcast_in_dim %3220, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3223 = stablehlo.broadcast_in_dim %3221, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3224 = stablehlo.add %3222, %3223 : tensor<1x256x1280xf32>
    %3225 = stablehlo.convert %3224 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3226 = stablehlo.reshape %3225 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3227 = stablehlo.convert %3226 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3228 = stablehlo.dot_general %3227, %arg309, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3229 = stablehlo.broadcast_in_dim %3228, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3230 = stablehlo.multiply %3229, %273 : tensor<256x256xf32>
    %3231 = stablehlo.broadcast_in_dim %3230, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3232 = stablehlo.broadcast_in_dim %arg310, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3233 = stablehlo.add %3231, %3232 : tensor<256x256xf32>
    %3234 = stablehlo.convert %3233 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3235 = stablehlo.reshape %3234 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3236 = stablehlo.dot_general %3227, %arg311, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3237 = stablehlo.broadcast_in_dim %3236, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3238 = stablehlo.multiply %3237, %273 : tensor<256x256xf32>
    %3239 = stablehlo.broadcast_in_dim %3238, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3240 = stablehlo.broadcast_in_dim %arg312, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3241 = stablehlo.add %3239, %3240 : tensor<256x256xf32>
    %3242 = stablehlo.convert %3241 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3243 = stablehlo.reshape %3242 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3244 = stablehlo.dot_general %3227, %arg313, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3245 = stablehlo.broadcast_in_dim %3244, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3246 = stablehlo.multiply %3245, %127 : tensor<256x1280xf32>
    %3247 = stablehlo.broadcast_in_dim %3246, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3248 = stablehlo.broadcast_in_dim %arg314, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3249 = stablehlo.add %3247, %3248 : tensor<256x1280xf32>
    %3250 = stablehlo.convert %3249 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3251 = stablehlo.reshape %3250 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3252 = stablehlo.reshape %3235 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3253 = stablehlo.transpose %3252, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3254 = stablehlo.reshape %3243 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3255 = stablehlo.transpose %3254, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3256 = stablehlo.reshape %3251 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %3257 = stablehlo.transpose %3256, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3258 = stablehlo.transpose %3255, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %3259 = stablehlo.reshape %3253 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %3260 = stablehlo.reshape %3258 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3261 = stablehlo.broadcast_in_dim %3260, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3262 = stablehlo.dot_general %3259, %3261, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %3263 = stablehlo.reshape %3262 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3264 = stablehlo.broadcast_in_dim %3263, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3265 = stablehlo.divide %3264, %309 : tensor<1x8x256x256xbf16>
    %3266 = stablehlo.convert %3265 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %3267 = stablehlo.reduce(%3266 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3268 = stablehlo.reshape %3267 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3269 = stablehlo.broadcast_in_dim %3266, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3270 = stablehlo.broadcast_in_dim %3268, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3271 = stablehlo.subtract %3269, %3270 : tensor<1x8x256x256xf32>
    %3272 = stablehlo.exponential %3271 : tensor<1x8x256x256xf32>
    %3273 = stablehlo.reduce(%3272 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3274 = stablehlo.reshape %3273 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3275 = stablehlo.broadcast_in_dim %3272, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3276 = stablehlo.broadcast_in_dim %3274, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3277 = stablehlo.divide %3275, %3276 : tensor<1x8x256x256xf32>
    %3278 = stablehlo.convert %3277 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %3279 = stablehlo.reshape %3278 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %3280 = stablehlo.reshape %3257 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3281 = stablehlo.broadcast_in_dim %3280, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3282 = stablehlo.dot_general %3279, %3281, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3283 = stablehlo.reshape %3282 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3284 = stablehlo.transpose %3283, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %3285 = stablehlo.reshape %3284 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %3286 = stablehlo.reshape %3285 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3287 = stablehlo.convert %3286 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3288 = stablehlo.dot_general %3287, %arg315, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3289 = stablehlo.broadcast_in_dim %3288, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3290 = stablehlo.multiply %3289, %127 : tensor<256x1280xf32>
    %3291 = stablehlo.broadcast_in_dim %3290, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3292 = stablehlo.broadcast_in_dim %arg316, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3293 = stablehlo.add %3291, %3292 : tensor<256x1280xf32>
    %3294 = stablehlo.convert %3293 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3295 = stablehlo.reshape %3294 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3296 = stablehlo.add %3295, %3188 : tensor<1x256x1280xbf16>
    %3297 = stablehlo.convert %3296 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3298 = stablehlo.convert %3297 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3299 = stablehlo.reduce(%3298 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3300 = stablehlo.reshape %3299 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3301 = stablehlo.broadcast_in_dim %3300, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3302 = stablehlo.divide %3301, %142 : tensor<1x256x1xf64>
    %3303 = stablehlo.broadcast_in_dim %3298, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3304 = stablehlo.broadcast_in_dim %3302, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3305 = stablehlo.subtract %3303, %3304 : tensor<1x256x1280xf64>
    %3306 = stablehlo.multiply %3305, %3305 : tensor<1x256x1280xf64>
    %3307 = stablehlo.reduce(%3306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3308 = stablehlo.reshape %3307 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3309 = stablehlo.broadcast_in_dim %3308, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3310 = stablehlo.divide %3309, %142 : tensor<1x256x1xf64>
    %3311 = stablehlo.convert %3310 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3312 = stablehlo.reduce(%3297 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3313 = stablehlo.reshape %3312 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3314 = stablehlo.broadcast_in_dim %3313, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3315 = stablehlo.divide %3314, %158 : tensor<1x256x1xf32>
    %3316 = stablehlo.broadcast_in_dim %3311, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3317 = stablehlo.add %3316, %161 : tensor<1x256x1xf32>
    %3318 = stablehlo.rsqrt %3317 : tensor<1x256x1xf32>
    %3319 = stablehlo.broadcast_in_dim %3297, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3320 = stablehlo.broadcast_in_dim %3315, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3321 = stablehlo.subtract %3319, %3320 : tensor<1x256x1280xf32>
    %3322 = stablehlo.broadcast_in_dim %3321, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3323 = stablehlo.broadcast_in_dim %3318, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3324 = stablehlo.multiply %3322, %3323 : tensor<1x256x1280xf32>
    %3325 = stablehlo.convert %arg69 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3326 = stablehlo.broadcast_in_dim %3324, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3327 = stablehlo.broadcast_in_dim %3325, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3328 = stablehlo.multiply %3326, %3327 : tensor<1x256x1280xf32>
    %3329 = stablehlo.convert %arg70 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3330 = stablehlo.broadcast_in_dim %3328, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3331 = stablehlo.broadcast_in_dim %3329, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3332 = stablehlo.add %3330, %3331 : tensor<1x256x1280xf32>
    %3333 = stablehlo.convert %3332 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3334 = stablehlo.reshape %3333 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3335 = stablehlo.convert %3334 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3336 = stablehlo.dot_general %3335, %arg317, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3337 = stablehlo.broadcast_in_dim %3336, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3338 = stablehlo.multiply %3337, %127 : tensor<256x1280xf32>
    %3339 = stablehlo.broadcast_in_dim %3338, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3340 = stablehlo.broadcast_in_dim %arg318, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3341 = stablehlo.add %3339, %3340 : tensor<256x1280xf32>
    %3342 = stablehlo.convert %3341 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3343 = stablehlo.reshape %3342 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3344 = stablehlo.multiply %3343, %cst_4 : tensor<1x256x1280xbf16>
    %3345 = stablehlo.multiply %3343, %190 : tensor<1x256x1280xbf16>
    %3346 = stablehlo.convert %3345 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3347 = stablehlo.clamp %cst_5, %3346, %cst_6 : tensor<1x256x1280xf32>
    %3348 = stablehlo.multiply %3347, %3347 : tensor<1x256x1280xf32>
    %3349 = stablehlo.multiply %cst_7, %3348 : tensor<1x256x1280xf32>
    %3350 = stablehlo.add %3349, %cst_8 : tensor<1x256x1280xf32>
    %3351 = stablehlo.multiply %3350, %3348 : tensor<1x256x1280xf32>
    %3352 = stablehlo.add %3351, %cst_9 : tensor<1x256x1280xf32>
    %3353 = stablehlo.multiply %3352, %3348 : tensor<1x256x1280xf32>
    %3354 = stablehlo.add %3353, %cst_10 : tensor<1x256x1280xf32>
    %3355 = stablehlo.multiply %3354, %3348 : tensor<1x256x1280xf32>
    %3356 = stablehlo.add %3355, %cst_11 : tensor<1x256x1280xf32>
    %3357 = stablehlo.multiply %3356, %3348 : tensor<1x256x1280xf32>
    %3358 = stablehlo.add %3357, %cst_12 : tensor<1x256x1280xf32>
    %3359 = stablehlo.multiply %3358, %3348 : tensor<1x256x1280xf32>
    %3360 = stablehlo.add %3359, %cst_13 : tensor<1x256x1280xf32>
    %3361 = stablehlo.multiply %cst_14, %3348 : tensor<1x256x1280xf32>
    %3362 = stablehlo.add %3361, %cst_15 : tensor<1x256x1280xf32>
    %3363 = stablehlo.multiply %3362, %3348 : tensor<1x256x1280xf32>
    %3364 = stablehlo.add %3363, %cst_16 : tensor<1x256x1280xf32>
    %3365 = stablehlo.multiply %3364, %3348 : tensor<1x256x1280xf32>
    %3366 = stablehlo.add %3365, %cst_17 : tensor<1x256x1280xf32>
    %3367 = stablehlo.multiply %3366, %3348 : tensor<1x256x1280xf32>
    %3368 = stablehlo.add %3367, %cst_18 : tensor<1x256x1280xf32>
    %3369 = stablehlo.multiply %3347, %3360 : tensor<1x256x1280xf32>
    %3370 = stablehlo.divide %3369, %3368 : tensor<1x256x1280xf32>
    %3371 = stablehlo.clamp %cst_19, %3370, %cst_20 : tensor<1x256x1280xf32>
    %3372 = stablehlo.convert %3371 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3373 = stablehlo.add %3372, %cst_2 : tensor<1x256x1280xbf16>
    %3374 = stablehlo.multiply %3373, %3344 : tensor<1x256x1280xbf16>
    %3375 = stablehlo.reshape %3374 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3376 = stablehlo.convert %3375 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3377 = stablehlo.dot_general %3376, %arg319, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3378 = stablehlo.broadcast_in_dim %3377, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3379 = stablehlo.multiply %3378, %127 : tensor<256x1280xf32>
    %3380 = stablehlo.broadcast_in_dim %3379, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3381 = stablehlo.broadcast_in_dim %arg320, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3382 = stablehlo.add %3380, %3381 : tensor<256x1280xf32>
    %3383 = stablehlo.convert %3382 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3384 = stablehlo.reshape %3383 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3385 = stablehlo.add %3384, %3296 : tensor<1x256x1280xbf16>
    %3386 = stablehlo.convert %3385 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3387 = stablehlo.convert %3386 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3388 = stablehlo.reduce(%3387 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3389 = stablehlo.reshape %3388 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3390 = stablehlo.broadcast_in_dim %3389, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3391 = stablehlo.divide %3390, %142 : tensor<1x256x1xf64>
    %3392 = stablehlo.broadcast_in_dim %3387, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3393 = stablehlo.broadcast_in_dim %3391, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3394 = stablehlo.subtract %3392, %3393 : tensor<1x256x1280xf64>
    %3395 = stablehlo.multiply %3394, %3394 : tensor<1x256x1280xf64>
    %3396 = stablehlo.reduce(%3395 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3397 = stablehlo.reshape %3396 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3398 = stablehlo.broadcast_in_dim %3397, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3399 = stablehlo.divide %3398, %142 : tensor<1x256x1xf64>
    %3400 = stablehlo.convert %3399 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3401 = stablehlo.reduce(%3386 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3402 = stablehlo.reshape %3401 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3403 = stablehlo.broadcast_in_dim %3402, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3404 = stablehlo.divide %3403, %158 : tensor<1x256x1xf32>
    %3405 = stablehlo.broadcast_in_dim %3400, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3406 = stablehlo.add %3405, %161 : tensor<1x256x1xf32>
    %3407 = stablehlo.rsqrt %3406 : tensor<1x256x1xf32>
    %3408 = stablehlo.broadcast_in_dim %3386, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3409 = stablehlo.broadcast_in_dim %3404, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3410 = stablehlo.subtract %3408, %3409 : tensor<1x256x1280xf32>
    %3411 = stablehlo.broadcast_in_dim %3410, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3412 = stablehlo.broadcast_in_dim %3407, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3413 = stablehlo.multiply %3411, %3412 : tensor<1x256x1280xf32>
    %3414 = stablehlo.convert %arg71 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3415 = stablehlo.broadcast_in_dim %3413, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3416 = stablehlo.broadcast_in_dim %3414, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3417 = stablehlo.multiply %3415, %3416 : tensor<1x256x1280xf32>
    %3418 = stablehlo.convert %arg72 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3419 = stablehlo.broadcast_in_dim %3417, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3420 = stablehlo.broadcast_in_dim %3418, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3421 = stablehlo.add %3419, %3420 : tensor<1x256x1280xf32>
    %3422 = stablehlo.convert %3421 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3423 = stablehlo.reshape %3422 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3424 = stablehlo.convert %3423 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3425 = stablehlo.dot_general %3424, %arg321, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3426 = stablehlo.broadcast_in_dim %3425, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3427 = stablehlo.multiply %3426, %273 : tensor<256x256xf32>
    %3428 = stablehlo.broadcast_in_dim %3427, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3429 = stablehlo.broadcast_in_dim %arg322, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3430 = stablehlo.add %3428, %3429 : tensor<256x256xf32>
    %3431 = stablehlo.convert %3430 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3432 = stablehlo.reshape %3431 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3433 = stablehlo.dot_general %3424, %arg323, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3434 = stablehlo.broadcast_in_dim %3433, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3435 = stablehlo.multiply %3434, %273 : tensor<256x256xf32>
    %3436 = stablehlo.broadcast_in_dim %3435, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3437 = stablehlo.broadcast_in_dim %arg324, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3438 = stablehlo.add %3436, %3437 : tensor<256x256xf32>
    %3439 = stablehlo.convert %3438 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3440 = stablehlo.reshape %3439 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3441 = stablehlo.dot_general %3424, %arg325, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3442 = stablehlo.broadcast_in_dim %3441, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3443 = stablehlo.multiply %3442, %127 : tensor<256x1280xf32>
    %3444 = stablehlo.broadcast_in_dim %3443, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3445 = stablehlo.broadcast_in_dim %arg326, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3446 = stablehlo.add %3444, %3445 : tensor<256x1280xf32>
    %3447 = stablehlo.convert %3446 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3448 = stablehlo.reshape %3447 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3449 = stablehlo.reshape %3432 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3450 = stablehlo.transpose %3449, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3451 = stablehlo.reshape %3440 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3452 = stablehlo.transpose %3451, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3453 = stablehlo.reshape %3448 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %3454 = stablehlo.transpose %3453, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3455 = stablehlo.transpose %3452, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %3456 = stablehlo.reshape %3450 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %3457 = stablehlo.reshape %3455 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3458 = stablehlo.broadcast_in_dim %3457, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3459 = stablehlo.dot_general %3456, %3458, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %3460 = stablehlo.reshape %3459 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3461 = stablehlo.broadcast_in_dim %3460, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3462 = stablehlo.divide %3461, %309 : tensor<1x8x256x256xbf16>
    %3463 = stablehlo.convert %3462 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %3464 = stablehlo.reduce(%3463 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3465 = stablehlo.reshape %3464 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3466 = stablehlo.broadcast_in_dim %3463, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3467 = stablehlo.broadcast_in_dim %3465, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3468 = stablehlo.subtract %3466, %3467 : tensor<1x8x256x256xf32>
    %3469 = stablehlo.exponential %3468 : tensor<1x8x256x256xf32>
    %3470 = stablehlo.reduce(%3469 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3471 = stablehlo.reshape %3470 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3472 = stablehlo.broadcast_in_dim %3469, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3473 = stablehlo.broadcast_in_dim %3471, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3474 = stablehlo.divide %3472, %3473 : tensor<1x8x256x256xf32>
    %3475 = stablehlo.convert %3474 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %3476 = stablehlo.reshape %3475 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %3477 = stablehlo.reshape %3454 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3478 = stablehlo.broadcast_in_dim %3477, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3479 = stablehlo.dot_general %3476, %3478, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3480 = stablehlo.reshape %3479 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3481 = stablehlo.transpose %3480, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %3482 = stablehlo.reshape %3481 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %3483 = stablehlo.reshape %3482 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3484 = stablehlo.convert %3483 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3485 = stablehlo.dot_general %3484, %arg327, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3486 = stablehlo.broadcast_in_dim %3485, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3487 = stablehlo.multiply %3486, %127 : tensor<256x1280xf32>
    %3488 = stablehlo.broadcast_in_dim %3487, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3489 = stablehlo.broadcast_in_dim %arg328, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3490 = stablehlo.add %3488, %3489 : tensor<256x1280xf32>
    %3491 = stablehlo.convert %3490 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3492 = stablehlo.reshape %3491 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3493 = stablehlo.add %3492, %3385 : tensor<1x256x1280xbf16>
    %3494 = stablehlo.convert %3493 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3495 = stablehlo.convert %3494 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3496 = stablehlo.reduce(%3495 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3497 = stablehlo.reshape %3496 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3498 = stablehlo.broadcast_in_dim %3497, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3499 = stablehlo.divide %3498, %142 : tensor<1x256x1xf64>
    %3500 = stablehlo.broadcast_in_dim %3495, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3501 = stablehlo.broadcast_in_dim %3499, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3502 = stablehlo.subtract %3500, %3501 : tensor<1x256x1280xf64>
    %3503 = stablehlo.multiply %3502, %3502 : tensor<1x256x1280xf64>
    %3504 = stablehlo.reduce(%3503 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3505 = stablehlo.reshape %3504 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3506 = stablehlo.broadcast_in_dim %3505, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3507 = stablehlo.divide %3506, %142 : tensor<1x256x1xf64>
    %3508 = stablehlo.convert %3507 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3509 = stablehlo.reduce(%3494 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3510 = stablehlo.reshape %3509 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3511 = stablehlo.broadcast_in_dim %3510, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3512 = stablehlo.divide %3511, %158 : tensor<1x256x1xf32>
    %3513 = stablehlo.broadcast_in_dim %3508, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3514 = stablehlo.add %3513, %161 : tensor<1x256x1xf32>
    %3515 = stablehlo.rsqrt %3514 : tensor<1x256x1xf32>
    %3516 = stablehlo.broadcast_in_dim %3494, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3517 = stablehlo.broadcast_in_dim %3512, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3518 = stablehlo.subtract %3516, %3517 : tensor<1x256x1280xf32>
    %3519 = stablehlo.broadcast_in_dim %3518, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3520 = stablehlo.broadcast_in_dim %3515, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3521 = stablehlo.multiply %3519, %3520 : tensor<1x256x1280xf32>
    %3522 = stablehlo.convert %arg73 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3523 = stablehlo.broadcast_in_dim %3521, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3524 = stablehlo.broadcast_in_dim %3522, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3525 = stablehlo.multiply %3523, %3524 : tensor<1x256x1280xf32>
    %3526 = stablehlo.convert %arg74 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3527 = stablehlo.broadcast_in_dim %3525, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3528 = stablehlo.broadcast_in_dim %3526, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3529 = stablehlo.add %3527, %3528 : tensor<1x256x1280xf32>
    %3530 = stablehlo.convert %3529 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3531 = stablehlo.reshape %3530 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3532 = stablehlo.convert %3531 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3533 = stablehlo.dot_general %3532, %arg329, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3534 = stablehlo.broadcast_in_dim %3533, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3535 = stablehlo.multiply %3534, %127 : tensor<256x1280xf32>
    %3536 = stablehlo.broadcast_in_dim %3535, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3537 = stablehlo.broadcast_in_dim %arg330, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3538 = stablehlo.add %3536, %3537 : tensor<256x1280xf32>
    %3539 = stablehlo.convert %3538 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3540 = stablehlo.reshape %3539 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3541 = stablehlo.multiply %3540, %cst_4 : tensor<1x256x1280xbf16>
    %3542 = stablehlo.multiply %3540, %190 : tensor<1x256x1280xbf16>
    %3543 = stablehlo.convert %3542 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3544 = stablehlo.clamp %cst_5, %3543, %cst_6 : tensor<1x256x1280xf32>
    %3545 = stablehlo.multiply %3544, %3544 : tensor<1x256x1280xf32>
    %3546 = stablehlo.multiply %cst_7, %3545 : tensor<1x256x1280xf32>
    %3547 = stablehlo.add %3546, %cst_8 : tensor<1x256x1280xf32>
    %3548 = stablehlo.multiply %3547, %3545 : tensor<1x256x1280xf32>
    %3549 = stablehlo.add %3548, %cst_9 : tensor<1x256x1280xf32>
    %3550 = stablehlo.multiply %3549, %3545 : tensor<1x256x1280xf32>
    %3551 = stablehlo.add %3550, %cst_10 : tensor<1x256x1280xf32>
    %3552 = stablehlo.multiply %3551, %3545 : tensor<1x256x1280xf32>
    %3553 = stablehlo.add %3552, %cst_11 : tensor<1x256x1280xf32>
    %3554 = stablehlo.multiply %3553, %3545 : tensor<1x256x1280xf32>
    %3555 = stablehlo.add %3554, %cst_12 : tensor<1x256x1280xf32>
    %3556 = stablehlo.multiply %3555, %3545 : tensor<1x256x1280xf32>
    %3557 = stablehlo.add %3556, %cst_13 : tensor<1x256x1280xf32>
    %3558 = stablehlo.multiply %cst_14, %3545 : tensor<1x256x1280xf32>
    %3559 = stablehlo.add %3558, %cst_15 : tensor<1x256x1280xf32>
    %3560 = stablehlo.multiply %3559, %3545 : tensor<1x256x1280xf32>
    %3561 = stablehlo.add %3560, %cst_16 : tensor<1x256x1280xf32>
    %3562 = stablehlo.multiply %3561, %3545 : tensor<1x256x1280xf32>
    %3563 = stablehlo.add %3562, %cst_17 : tensor<1x256x1280xf32>
    %3564 = stablehlo.multiply %3563, %3545 : tensor<1x256x1280xf32>
    %3565 = stablehlo.add %3564, %cst_18 : tensor<1x256x1280xf32>
    %3566 = stablehlo.multiply %3544, %3557 : tensor<1x256x1280xf32>
    %3567 = stablehlo.divide %3566, %3565 : tensor<1x256x1280xf32>
    %3568 = stablehlo.clamp %cst_19, %3567, %cst_20 : tensor<1x256x1280xf32>
    %3569 = stablehlo.convert %3568 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3570 = stablehlo.add %3569, %cst_2 : tensor<1x256x1280xbf16>
    %3571 = stablehlo.multiply %3570, %3541 : tensor<1x256x1280xbf16>
    %3572 = stablehlo.reshape %3571 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3573 = stablehlo.convert %3572 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3574 = stablehlo.dot_general %3573, %arg331, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3575 = stablehlo.broadcast_in_dim %3574, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3576 = stablehlo.multiply %3575, %127 : tensor<256x1280xf32>
    %3577 = stablehlo.broadcast_in_dim %3576, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3578 = stablehlo.broadcast_in_dim %arg332, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3579 = stablehlo.add %3577, %3578 : tensor<256x1280xf32>
    %3580 = stablehlo.convert %3579 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3581 = stablehlo.reshape %3580 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3582 = stablehlo.add %3581, %3493 : tensor<1x256x1280xbf16>
    %3583 = stablehlo.convert %3582 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3584 = stablehlo.convert %3583 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3585 = stablehlo.reduce(%3584 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3586 = stablehlo.reshape %3585 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3587 = stablehlo.broadcast_in_dim %3586, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3588 = stablehlo.divide %3587, %142 : tensor<1x256x1xf64>
    %3589 = stablehlo.broadcast_in_dim %3584, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3590 = stablehlo.broadcast_in_dim %3588, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3591 = stablehlo.subtract %3589, %3590 : tensor<1x256x1280xf64>
    %3592 = stablehlo.multiply %3591, %3591 : tensor<1x256x1280xf64>
    %3593 = stablehlo.reduce(%3592 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3594 = stablehlo.reshape %3593 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3595 = stablehlo.broadcast_in_dim %3594, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3596 = stablehlo.divide %3595, %142 : tensor<1x256x1xf64>
    %3597 = stablehlo.convert %3596 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3598 = stablehlo.reduce(%3583 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3599 = stablehlo.reshape %3598 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3600 = stablehlo.broadcast_in_dim %3599, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3601 = stablehlo.divide %3600, %158 : tensor<1x256x1xf32>
    %3602 = stablehlo.broadcast_in_dim %3597, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3603 = stablehlo.add %3602, %161 : tensor<1x256x1xf32>
    %3604 = stablehlo.rsqrt %3603 : tensor<1x256x1xf32>
    %3605 = stablehlo.broadcast_in_dim %3583, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3606 = stablehlo.broadcast_in_dim %3601, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3607 = stablehlo.subtract %3605, %3606 : tensor<1x256x1280xf32>
    %3608 = stablehlo.broadcast_in_dim %3607, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3609 = stablehlo.broadcast_in_dim %3604, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3610 = stablehlo.multiply %3608, %3609 : tensor<1x256x1280xf32>
    %3611 = stablehlo.convert %arg75 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3612 = stablehlo.broadcast_in_dim %3610, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3613 = stablehlo.broadcast_in_dim %3611, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3614 = stablehlo.multiply %3612, %3613 : tensor<1x256x1280xf32>
    %3615 = stablehlo.convert %arg76 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3616 = stablehlo.broadcast_in_dim %3614, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3617 = stablehlo.broadcast_in_dim %3615, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3618 = stablehlo.add %3616, %3617 : tensor<1x256x1280xf32>
    %3619 = stablehlo.convert %3618 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3620 = stablehlo.reshape %3619 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3621 = stablehlo.convert %3620 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3622 = stablehlo.dot_general %3621, %arg333, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3623 = stablehlo.broadcast_in_dim %3622, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3624 = stablehlo.multiply %3623, %273 : tensor<256x256xf32>
    %3625 = stablehlo.broadcast_in_dim %3624, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3626 = stablehlo.broadcast_in_dim %arg334, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3627 = stablehlo.add %3625, %3626 : tensor<256x256xf32>
    %3628 = stablehlo.convert %3627 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3629 = stablehlo.reshape %3628 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3630 = stablehlo.dot_general %3621, %arg335, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3631 = stablehlo.broadcast_in_dim %3630, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3632 = stablehlo.multiply %3631, %273 : tensor<256x256xf32>
    %3633 = stablehlo.broadcast_in_dim %3632, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3634 = stablehlo.broadcast_in_dim %arg336, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3635 = stablehlo.add %3633, %3634 : tensor<256x256xf32>
    %3636 = stablehlo.convert %3635 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3637 = stablehlo.reshape %3636 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3638 = stablehlo.dot_general %3621, %arg337, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3639 = stablehlo.broadcast_in_dim %3638, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3640 = stablehlo.multiply %3639, %127 : tensor<256x1280xf32>
    %3641 = stablehlo.broadcast_in_dim %3640, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3642 = stablehlo.broadcast_in_dim %arg338, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3643 = stablehlo.add %3641, %3642 : tensor<256x1280xf32>
    %3644 = stablehlo.convert %3643 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3645 = stablehlo.reshape %3644 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3646 = stablehlo.reshape %3629 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3647 = stablehlo.transpose %3646, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3648 = stablehlo.reshape %3637 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3649 = stablehlo.transpose %3648, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3650 = stablehlo.reshape %3645 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %3651 = stablehlo.transpose %3650, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3652 = stablehlo.transpose %3649, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %3653 = stablehlo.reshape %3647 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %3654 = stablehlo.reshape %3652 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3655 = stablehlo.broadcast_in_dim %3654, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3656 = stablehlo.dot_general %3653, %3655, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %3657 = stablehlo.reshape %3656 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3658 = stablehlo.broadcast_in_dim %3657, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3659 = stablehlo.divide %3658, %309 : tensor<1x8x256x256xbf16>
    %3660 = stablehlo.convert %3659 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %3661 = stablehlo.reduce(%3660 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3662 = stablehlo.reshape %3661 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3663 = stablehlo.broadcast_in_dim %3660, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3664 = stablehlo.broadcast_in_dim %3662, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3665 = stablehlo.subtract %3663, %3664 : tensor<1x8x256x256xf32>
    %3666 = stablehlo.exponential %3665 : tensor<1x8x256x256xf32>
    %3667 = stablehlo.reduce(%3666 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3668 = stablehlo.reshape %3667 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3669 = stablehlo.broadcast_in_dim %3666, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3670 = stablehlo.broadcast_in_dim %3668, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3671 = stablehlo.divide %3669, %3670 : tensor<1x8x256x256xf32>
    %3672 = stablehlo.convert %3671 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %3673 = stablehlo.reshape %3672 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %3674 = stablehlo.reshape %3651 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3675 = stablehlo.broadcast_in_dim %3674, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3676 = stablehlo.dot_general %3673, %3675, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3677 = stablehlo.reshape %3676 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3678 = stablehlo.transpose %3677, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %3679 = stablehlo.reshape %3678 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %3680 = stablehlo.reshape %3679 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3681 = stablehlo.convert %3680 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3682 = stablehlo.dot_general %3681, %arg339, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3683 = stablehlo.broadcast_in_dim %3682, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3684 = stablehlo.multiply %3683, %127 : tensor<256x1280xf32>
    %3685 = stablehlo.broadcast_in_dim %3684, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3686 = stablehlo.broadcast_in_dim %arg340, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3687 = stablehlo.add %3685, %3686 : tensor<256x1280xf32>
    %3688 = stablehlo.convert %3687 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3689 = stablehlo.reshape %3688 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3690 = stablehlo.add %3689, %3582 : tensor<1x256x1280xbf16>
    %3691 = stablehlo.convert %3690 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3692 = stablehlo.convert %3691 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3693 = stablehlo.reduce(%3692 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3694 = stablehlo.reshape %3693 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3695 = stablehlo.broadcast_in_dim %3694, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3696 = stablehlo.divide %3695, %142 : tensor<1x256x1xf64>
    %3697 = stablehlo.broadcast_in_dim %3692, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3698 = stablehlo.broadcast_in_dim %3696, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3699 = stablehlo.subtract %3697, %3698 : tensor<1x256x1280xf64>
    %3700 = stablehlo.multiply %3699, %3699 : tensor<1x256x1280xf64>
    %3701 = stablehlo.reduce(%3700 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3702 = stablehlo.reshape %3701 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3703 = stablehlo.broadcast_in_dim %3702, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3704 = stablehlo.divide %3703, %142 : tensor<1x256x1xf64>
    %3705 = stablehlo.convert %3704 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3706 = stablehlo.reduce(%3691 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3707 = stablehlo.reshape %3706 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3708 = stablehlo.broadcast_in_dim %3707, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3709 = stablehlo.divide %3708, %158 : tensor<1x256x1xf32>
    %3710 = stablehlo.broadcast_in_dim %3705, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3711 = stablehlo.add %3710, %161 : tensor<1x256x1xf32>
    %3712 = stablehlo.rsqrt %3711 : tensor<1x256x1xf32>
    %3713 = stablehlo.broadcast_in_dim %3691, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3714 = stablehlo.broadcast_in_dim %3709, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3715 = stablehlo.subtract %3713, %3714 : tensor<1x256x1280xf32>
    %3716 = stablehlo.broadcast_in_dim %3715, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3717 = stablehlo.broadcast_in_dim %3712, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3718 = stablehlo.multiply %3716, %3717 : tensor<1x256x1280xf32>
    %3719 = stablehlo.convert %arg77 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3720 = stablehlo.broadcast_in_dim %3718, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3721 = stablehlo.broadcast_in_dim %3719, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3722 = stablehlo.multiply %3720, %3721 : tensor<1x256x1280xf32>
    %3723 = stablehlo.convert %arg78 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3724 = stablehlo.broadcast_in_dim %3722, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3725 = stablehlo.broadcast_in_dim %3723, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3726 = stablehlo.add %3724, %3725 : tensor<1x256x1280xf32>
    %3727 = stablehlo.convert %3726 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3728 = stablehlo.reshape %3727 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3729 = stablehlo.convert %3728 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3730 = stablehlo.dot_general %3729, %arg341, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3731 = stablehlo.broadcast_in_dim %3730, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3732 = stablehlo.multiply %3731, %127 : tensor<256x1280xf32>
    %3733 = stablehlo.broadcast_in_dim %3732, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3734 = stablehlo.broadcast_in_dim %arg342, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3735 = stablehlo.add %3733, %3734 : tensor<256x1280xf32>
    %3736 = stablehlo.convert %3735 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3737 = stablehlo.reshape %3736 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3738 = stablehlo.multiply %3737, %cst_4 : tensor<1x256x1280xbf16>
    %3739 = stablehlo.multiply %3737, %190 : tensor<1x256x1280xbf16>
    %3740 = stablehlo.convert %3739 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3741 = stablehlo.clamp %cst_5, %3740, %cst_6 : tensor<1x256x1280xf32>
    %3742 = stablehlo.multiply %3741, %3741 : tensor<1x256x1280xf32>
    %3743 = stablehlo.multiply %cst_7, %3742 : tensor<1x256x1280xf32>
    %3744 = stablehlo.add %3743, %cst_8 : tensor<1x256x1280xf32>
    %3745 = stablehlo.multiply %3744, %3742 : tensor<1x256x1280xf32>
    %3746 = stablehlo.add %3745, %cst_9 : tensor<1x256x1280xf32>
    %3747 = stablehlo.multiply %3746, %3742 : tensor<1x256x1280xf32>
    %3748 = stablehlo.add %3747, %cst_10 : tensor<1x256x1280xf32>
    %3749 = stablehlo.multiply %3748, %3742 : tensor<1x256x1280xf32>
    %3750 = stablehlo.add %3749, %cst_11 : tensor<1x256x1280xf32>
    %3751 = stablehlo.multiply %3750, %3742 : tensor<1x256x1280xf32>
    %3752 = stablehlo.add %3751, %cst_12 : tensor<1x256x1280xf32>
    %3753 = stablehlo.multiply %3752, %3742 : tensor<1x256x1280xf32>
    %3754 = stablehlo.add %3753, %cst_13 : tensor<1x256x1280xf32>
    %3755 = stablehlo.multiply %cst_14, %3742 : tensor<1x256x1280xf32>
    %3756 = stablehlo.add %3755, %cst_15 : tensor<1x256x1280xf32>
    %3757 = stablehlo.multiply %3756, %3742 : tensor<1x256x1280xf32>
    %3758 = stablehlo.add %3757, %cst_16 : tensor<1x256x1280xf32>
    %3759 = stablehlo.multiply %3758, %3742 : tensor<1x256x1280xf32>
    %3760 = stablehlo.add %3759, %cst_17 : tensor<1x256x1280xf32>
    %3761 = stablehlo.multiply %3760, %3742 : tensor<1x256x1280xf32>
    %3762 = stablehlo.add %3761, %cst_18 : tensor<1x256x1280xf32>
    %3763 = stablehlo.multiply %3741, %3754 : tensor<1x256x1280xf32>
    %3764 = stablehlo.divide %3763, %3762 : tensor<1x256x1280xf32>
    %3765 = stablehlo.clamp %cst_19, %3764, %cst_20 : tensor<1x256x1280xf32>
    %3766 = stablehlo.convert %3765 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3767 = stablehlo.add %3766, %cst_2 : tensor<1x256x1280xbf16>
    %3768 = stablehlo.multiply %3767, %3738 : tensor<1x256x1280xbf16>
    %3769 = stablehlo.reshape %3768 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3770 = stablehlo.convert %3769 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3771 = stablehlo.dot_general %3770, %arg343, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3772 = stablehlo.broadcast_in_dim %3771, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3773 = stablehlo.multiply %3772, %127 : tensor<256x1280xf32>
    %3774 = stablehlo.broadcast_in_dim %3773, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3775 = stablehlo.broadcast_in_dim %arg344, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3776 = stablehlo.add %3774, %3775 : tensor<256x1280xf32>
    %3777 = stablehlo.convert %3776 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3778 = stablehlo.reshape %3777 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3779 = stablehlo.add %3778, %3690 : tensor<1x256x1280xbf16>
    %3780 = stablehlo.convert %3779 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3781 = stablehlo.convert %3780 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3782 = stablehlo.reduce(%3781 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3783 = stablehlo.reshape %3782 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3784 = stablehlo.broadcast_in_dim %3783, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3785 = stablehlo.divide %3784, %142 : tensor<1x256x1xf64>
    %3786 = stablehlo.broadcast_in_dim %3781, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3787 = stablehlo.broadcast_in_dim %3785, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3788 = stablehlo.subtract %3786, %3787 : tensor<1x256x1280xf64>
    %3789 = stablehlo.multiply %3788, %3788 : tensor<1x256x1280xf64>
    %3790 = stablehlo.reduce(%3789 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3791 = stablehlo.reshape %3790 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3792 = stablehlo.broadcast_in_dim %3791, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3793 = stablehlo.divide %3792, %142 : tensor<1x256x1xf64>
    %3794 = stablehlo.convert %3793 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3795 = stablehlo.reduce(%3780 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3796 = stablehlo.reshape %3795 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3797 = stablehlo.broadcast_in_dim %3796, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3798 = stablehlo.divide %3797, %158 : tensor<1x256x1xf32>
    %3799 = stablehlo.broadcast_in_dim %3794, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3800 = stablehlo.add %3799, %161 : tensor<1x256x1xf32>
    %3801 = stablehlo.rsqrt %3800 : tensor<1x256x1xf32>
    %3802 = stablehlo.broadcast_in_dim %3780, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3803 = stablehlo.broadcast_in_dim %3798, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3804 = stablehlo.subtract %3802, %3803 : tensor<1x256x1280xf32>
    %3805 = stablehlo.broadcast_in_dim %3804, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3806 = stablehlo.broadcast_in_dim %3801, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3807 = stablehlo.multiply %3805, %3806 : tensor<1x256x1280xf32>
    %3808 = stablehlo.convert %arg79 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3809 = stablehlo.broadcast_in_dim %3807, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3810 = stablehlo.broadcast_in_dim %3808, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3811 = stablehlo.multiply %3809, %3810 : tensor<1x256x1280xf32>
    %3812 = stablehlo.convert %arg80 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3813 = stablehlo.broadcast_in_dim %3811, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3814 = stablehlo.broadcast_in_dim %3812, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3815 = stablehlo.add %3813, %3814 : tensor<1x256x1280xf32>
    %3816 = stablehlo.convert %3815 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3817 = stablehlo.reshape %3816 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3818 = stablehlo.convert %3817 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3819 = stablehlo.dot_general %3818, %arg345, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3820 = stablehlo.broadcast_in_dim %3819, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3821 = stablehlo.multiply %3820, %273 : tensor<256x256xf32>
    %3822 = stablehlo.broadcast_in_dim %3821, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3823 = stablehlo.broadcast_in_dim %arg346, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3824 = stablehlo.add %3822, %3823 : tensor<256x256xf32>
    %3825 = stablehlo.convert %3824 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3826 = stablehlo.reshape %3825 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3827 = stablehlo.dot_general %3818, %arg347, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %3828 = stablehlo.broadcast_in_dim %3827, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3829 = stablehlo.multiply %3828, %273 : tensor<256x256xf32>
    %3830 = stablehlo.broadcast_in_dim %3829, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3831 = stablehlo.broadcast_in_dim %arg348, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %3832 = stablehlo.add %3830, %3831 : tensor<256x256xf32>
    %3833 = stablehlo.convert %3832 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %3834 = stablehlo.reshape %3833 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %3835 = stablehlo.dot_general %3818, %arg349, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3836 = stablehlo.broadcast_in_dim %3835, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3837 = stablehlo.multiply %3836, %127 : tensor<256x1280xf32>
    %3838 = stablehlo.broadcast_in_dim %3837, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3839 = stablehlo.broadcast_in_dim %arg350, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3840 = stablehlo.add %3838, %3839 : tensor<256x1280xf32>
    %3841 = stablehlo.convert %3840 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3842 = stablehlo.reshape %3841 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3843 = stablehlo.reshape %3826 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3844 = stablehlo.transpose %3843, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3845 = stablehlo.reshape %3834 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %3846 = stablehlo.transpose %3845, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %3847 = stablehlo.reshape %3842 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %3848 = stablehlo.transpose %3847, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3849 = stablehlo.transpose %3846, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %3850 = stablehlo.reshape %3844 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %3851 = stablehlo.reshape %3849 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3852 = stablehlo.broadcast_in_dim %3851, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %3853 = stablehlo.dot_general %3850, %3852, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %3854 = stablehlo.reshape %3853 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3855 = stablehlo.broadcast_in_dim %3854, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %3856 = stablehlo.divide %3855, %309 : tensor<1x8x256x256xbf16>
    %3857 = stablehlo.convert %3856 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %3858 = stablehlo.reduce(%3857 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3859 = stablehlo.reshape %3858 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3860 = stablehlo.broadcast_in_dim %3857, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3861 = stablehlo.broadcast_in_dim %3859, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3862 = stablehlo.subtract %3860, %3861 : tensor<1x8x256x256xf32>
    %3863 = stablehlo.exponential %3862 : tensor<1x8x256x256xf32>
    %3864 = stablehlo.reduce(%3863 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %3865 = stablehlo.reshape %3864 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %3866 = stablehlo.broadcast_in_dim %3863, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %3867 = stablehlo.broadcast_in_dim %3865, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %3868 = stablehlo.divide %3866, %3867 : tensor<1x8x256x256xf32>
    %3869 = stablehlo.convert %3868 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %3870 = stablehlo.reshape %3869 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %3871 = stablehlo.reshape %3848 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3872 = stablehlo.broadcast_in_dim %3871, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3873 = stablehlo.dot_general %3870, %3872, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %3874 = stablehlo.reshape %3873 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %3875 = stablehlo.transpose %3874, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %3876 = stablehlo.reshape %3875 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %3877 = stablehlo.reshape %3876 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3878 = stablehlo.convert %3877 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3879 = stablehlo.dot_general %3878, %arg351, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3880 = stablehlo.broadcast_in_dim %3879, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3881 = stablehlo.multiply %3880, %127 : tensor<256x1280xf32>
    %3882 = stablehlo.broadcast_in_dim %3881, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3883 = stablehlo.broadcast_in_dim %arg352, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3884 = stablehlo.add %3882, %3883 : tensor<256x1280xf32>
    %3885 = stablehlo.convert %3884 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3886 = stablehlo.reshape %3885 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3887 = stablehlo.add %3886, %3779 : tensor<1x256x1280xbf16>
    %3888 = stablehlo.convert %3887 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3889 = stablehlo.convert %3888 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3890 = stablehlo.reduce(%3889 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3891 = stablehlo.reshape %3890 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3892 = stablehlo.broadcast_in_dim %3891, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3893 = stablehlo.divide %3892, %142 : tensor<1x256x1xf64>
    %3894 = stablehlo.broadcast_in_dim %3889, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3895 = stablehlo.broadcast_in_dim %3893, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3896 = stablehlo.subtract %3894, %3895 : tensor<1x256x1280xf64>
    %3897 = stablehlo.multiply %3896, %3896 : tensor<1x256x1280xf64>
    %3898 = stablehlo.reduce(%3897 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3899 = stablehlo.reshape %3898 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3900 = stablehlo.broadcast_in_dim %3899, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3901 = stablehlo.divide %3900, %142 : tensor<1x256x1xf64>
    %3902 = stablehlo.convert %3901 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3903 = stablehlo.reduce(%3888 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3904 = stablehlo.reshape %3903 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3905 = stablehlo.broadcast_in_dim %3904, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3906 = stablehlo.divide %3905, %158 : tensor<1x256x1xf32>
    %3907 = stablehlo.broadcast_in_dim %3902, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3908 = stablehlo.add %3907, %161 : tensor<1x256x1xf32>
    %3909 = stablehlo.rsqrt %3908 : tensor<1x256x1xf32>
    %3910 = stablehlo.broadcast_in_dim %3888, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3911 = stablehlo.broadcast_in_dim %3906, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3912 = stablehlo.subtract %3910, %3911 : tensor<1x256x1280xf32>
    %3913 = stablehlo.broadcast_in_dim %3912, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3914 = stablehlo.broadcast_in_dim %3909, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3915 = stablehlo.multiply %3913, %3914 : tensor<1x256x1280xf32>
    %3916 = stablehlo.convert %arg81 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3917 = stablehlo.broadcast_in_dim %3915, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3918 = stablehlo.broadcast_in_dim %3916, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3919 = stablehlo.multiply %3917, %3918 : tensor<1x256x1280xf32>
    %3920 = stablehlo.convert %arg82 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %3921 = stablehlo.broadcast_in_dim %3919, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3922 = stablehlo.broadcast_in_dim %3920, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %3923 = stablehlo.add %3921, %3922 : tensor<1x256x1280xf32>
    %3924 = stablehlo.convert %3923 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3925 = stablehlo.reshape %3924 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3926 = stablehlo.convert %3925 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3927 = stablehlo.dot_general %3926, %arg353, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3928 = stablehlo.broadcast_in_dim %3927, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3929 = stablehlo.multiply %3928, %127 : tensor<256x1280xf32>
    %3930 = stablehlo.broadcast_in_dim %3929, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3931 = stablehlo.broadcast_in_dim %arg354, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3932 = stablehlo.add %3930, %3931 : tensor<256x1280xf32>
    %3933 = stablehlo.convert %3932 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3934 = stablehlo.reshape %3933 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3935 = stablehlo.multiply %3934, %cst_4 : tensor<1x256x1280xbf16>
    %3936 = stablehlo.multiply %3934, %190 : tensor<1x256x1280xbf16>
    %3937 = stablehlo.convert %3936 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3938 = stablehlo.clamp %cst_5, %3937, %cst_6 : tensor<1x256x1280xf32>
    %3939 = stablehlo.multiply %3938, %3938 : tensor<1x256x1280xf32>
    %3940 = stablehlo.multiply %cst_7, %3939 : tensor<1x256x1280xf32>
    %3941 = stablehlo.add %3940, %cst_8 : tensor<1x256x1280xf32>
    %3942 = stablehlo.multiply %3941, %3939 : tensor<1x256x1280xf32>
    %3943 = stablehlo.add %3942, %cst_9 : tensor<1x256x1280xf32>
    %3944 = stablehlo.multiply %3943, %3939 : tensor<1x256x1280xf32>
    %3945 = stablehlo.add %3944, %cst_10 : tensor<1x256x1280xf32>
    %3946 = stablehlo.multiply %3945, %3939 : tensor<1x256x1280xf32>
    %3947 = stablehlo.add %3946, %cst_11 : tensor<1x256x1280xf32>
    %3948 = stablehlo.multiply %3947, %3939 : tensor<1x256x1280xf32>
    %3949 = stablehlo.add %3948, %cst_12 : tensor<1x256x1280xf32>
    %3950 = stablehlo.multiply %3949, %3939 : tensor<1x256x1280xf32>
    %3951 = stablehlo.add %3950, %cst_13 : tensor<1x256x1280xf32>
    %3952 = stablehlo.multiply %cst_14, %3939 : tensor<1x256x1280xf32>
    %3953 = stablehlo.add %3952, %cst_15 : tensor<1x256x1280xf32>
    %3954 = stablehlo.multiply %3953, %3939 : tensor<1x256x1280xf32>
    %3955 = stablehlo.add %3954, %cst_16 : tensor<1x256x1280xf32>
    %3956 = stablehlo.multiply %3955, %3939 : tensor<1x256x1280xf32>
    %3957 = stablehlo.add %3956, %cst_17 : tensor<1x256x1280xf32>
    %3958 = stablehlo.multiply %3957, %3939 : tensor<1x256x1280xf32>
    %3959 = stablehlo.add %3958, %cst_18 : tensor<1x256x1280xf32>
    %3960 = stablehlo.multiply %3938, %3951 : tensor<1x256x1280xf32>
    %3961 = stablehlo.divide %3960, %3959 : tensor<1x256x1280xf32>
    %3962 = stablehlo.clamp %cst_19, %3961, %cst_20 : tensor<1x256x1280xf32>
    %3963 = stablehlo.convert %3962 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %3964 = stablehlo.add %3963, %cst_2 : tensor<1x256x1280xbf16>
    %3965 = stablehlo.multiply %3964, %3935 : tensor<1x256x1280xbf16>
    %3966 = stablehlo.reshape %3965 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %3967 = stablehlo.convert %3966 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %3968 = stablehlo.dot_general %3967, %arg355, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %3969 = stablehlo.broadcast_in_dim %3968, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3970 = stablehlo.multiply %3969, %127 : tensor<256x1280xf32>
    %3971 = stablehlo.broadcast_in_dim %3970, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3972 = stablehlo.broadcast_in_dim %arg356, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %3973 = stablehlo.add %3971, %3972 : tensor<256x1280xf32>
    %3974 = stablehlo.convert %3973 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %3975 = stablehlo.reshape %3974 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %3976 = stablehlo.add %3975, %3887 : tensor<1x256x1280xbf16>
    %3977 = stablehlo.convert %3976 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %3978 = stablehlo.convert %3977 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %3979 = stablehlo.reduce(%3978 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3980 = stablehlo.reshape %3979 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3981 = stablehlo.broadcast_in_dim %3980, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3982 = stablehlo.divide %3981, %142 : tensor<1x256x1xf64>
    %3983 = stablehlo.broadcast_in_dim %3978, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %3984 = stablehlo.broadcast_in_dim %3982, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %3985 = stablehlo.subtract %3983, %3984 : tensor<1x256x1280xf64>
    %3986 = stablehlo.multiply %3985, %3985 : tensor<1x256x1280xf64>
    %3987 = stablehlo.reduce(%3986 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %3988 = stablehlo.reshape %3987 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %3989 = stablehlo.broadcast_in_dim %3988, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %3990 = stablehlo.divide %3989, %142 : tensor<1x256x1xf64>
    %3991 = stablehlo.convert %3990 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %3992 = stablehlo.reduce(%3977 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %3993 = stablehlo.reshape %3992 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %3994 = stablehlo.broadcast_in_dim %3993, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3995 = stablehlo.divide %3994, %158 : tensor<1x256x1xf32>
    %3996 = stablehlo.broadcast_in_dim %3991, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3997 = stablehlo.add %3996, %161 : tensor<1x256x1xf32>
    %3998 = stablehlo.rsqrt %3997 : tensor<1x256x1xf32>
    %3999 = stablehlo.broadcast_in_dim %3977, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4000 = stablehlo.broadcast_in_dim %3995, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4001 = stablehlo.subtract %3999, %4000 : tensor<1x256x1280xf32>
    %4002 = stablehlo.broadcast_in_dim %4001, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4003 = stablehlo.broadcast_in_dim %3998, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4004 = stablehlo.multiply %4002, %4003 : tensor<1x256x1280xf32>
    %4005 = stablehlo.convert %arg83 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4006 = stablehlo.broadcast_in_dim %4004, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4007 = stablehlo.broadcast_in_dim %4005, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4008 = stablehlo.multiply %4006, %4007 : tensor<1x256x1280xf32>
    %4009 = stablehlo.convert %arg84 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4010 = stablehlo.broadcast_in_dim %4008, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4011 = stablehlo.broadcast_in_dim %4009, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4012 = stablehlo.add %4010, %4011 : tensor<1x256x1280xf32>
    %4013 = stablehlo.convert %4012 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4014 = stablehlo.reshape %4013 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4015 = stablehlo.convert %4014 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4016 = stablehlo.dot_general %4015, %arg357, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4017 = stablehlo.broadcast_in_dim %4016, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4018 = stablehlo.multiply %4017, %273 : tensor<256x256xf32>
    %4019 = stablehlo.broadcast_in_dim %4018, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4020 = stablehlo.broadcast_in_dim %arg358, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4021 = stablehlo.add %4019, %4020 : tensor<256x256xf32>
    %4022 = stablehlo.convert %4021 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4023 = stablehlo.reshape %4022 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4024 = stablehlo.dot_general %4015, %arg359, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4025 = stablehlo.broadcast_in_dim %4024, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4026 = stablehlo.multiply %4025, %273 : tensor<256x256xf32>
    %4027 = stablehlo.broadcast_in_dim %4026, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4028 = stablehlo.broadcast_in_dim %arg360, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4029 = stablehlo.add %4027, %4028 : tensor<256x256xf32>
    %4030 = stablehlo.convert %4029 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4031 = stablehlo.reshape %4030 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4032 = stablehlo.dot_general %4015, %arg361, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4033 = stablehlo.broadcast_in_dim %4032, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4034 = stablehlo.multiply %4033, %127 : tensor<256x1280xf32>
    %4035 = stablehlo.broadcast_in_dim %4034, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4036 = stablehlo.broadcast_in_dim %arg362, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4037 = stablehlo.add %4035, %4036 : tensor<256x1280xf32>
    %4038 = stablehlo.convert %4037 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4039 = stablehlo.reshape %4038 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4040 = stablehlo.reshape %4023 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4041 = stablehlo.transpose %4040, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4042 = stablehlo.reshape %4031 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4043 = stablehlo.transpose %4042, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4044 = stablehlo.reshape %4039 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %4045 = stablehlo.transpose %4044, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4046 = stablehlo.transpose %4043, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %4047 = stablehlo.reshape %4041 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %4048 = stablehlo.reshape %4046 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4049 = stablehlo.broadcast_in_dim %4048, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4050 = stablehlo.dot_general %4047, %4049, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %4051 = stablehlo.reshape %4050 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4052 = stablehlo.broadcast_in_dim %4051, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4053 = stablehlo.divide %4052, %309 : tensor<1x8x256x256xbf16>
    %4054 = stablehlo.convert %4053 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %4055 = stablehlo.reduce(%4054 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4056 = stablehlo.reshape %4055 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4057 = stablehlo.broadcast_in_dim %4054, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4058 = stablehlo.broadcast_in_dim %4056, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4059 = stablehlo.subtract %4057, %4058 : tensor<1x8x256x256xf32>
    %4060 = stablehlo.exponential %4059 : tensor<1x8x256x256xf32>
    %4061 = stablehlo.reduce(%4060 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4062 = stablehlo.reshape %4061 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4063 = stablehlo.broadcast_in_dim %4060, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4064 = stablehlo.broadcast_in_dim %4062, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4065 = stablehlo.divide %4063, %4064 : tensor<1x8x256x256xf32>
    %4066 = stablehlo.convert %4065 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %4067 = stablehlo.reshape %4066 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %4068 = stablehlo.reshape %4045 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4069 = stablehlo.broadcast_in_dim %4068, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4070 = stablehlo.dot_general %4067, %4069, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4071 = stablehlo.reshape %4070 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4072 = stablehlo.transpose %4071, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %4073 = stablehlo.reshape %4072 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %4074 = stablehlo.reshape %4073 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4075 = stablehlo.convert %4074 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4076 = stablehlo.dot_general %4075, %arg363, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4077 = stablehlo.broadcast_in_dim %4076, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4078 = stablehlo.multiply %4077, %127 : tensor<256x1280xf32>
    %4079 = stablehlo.broadcast_in_dim %4078, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4080 = stablehlo.broadcast_in_dim %arg364, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4081 = stablehlo.add %4079, %4080 : tensor<256x1280xf32>
    %4082 = stablehlo.convert %4081 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4083 = stablehlo.reshape %4082 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4084 = stablehlo.add %4083, %3976 : tensor<1x256x1280xbf16>
    %4085 = stablehlo.convert %4084 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4086 = stablehlo.convert %4085 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4087 = stablehlo.reduce(%4086 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4088 = stablehlo.reshape %4087 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4089 = stablehlo.broadcast_in_dim %4088, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4090 = stablehlo.divide %4089, %142 : tensor<1x256x1xf64>
    %4091 = stablehlo.broadcast_in_dim %4086, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4092 = stablehlo.broadcast_in_dim %4090, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4093 = stablehlo.subtract %4091, %4092 : tensor<1x256x1280xf64>
    %4094 = stablehlo.multiply %4093, %4093 : tensor<1x256x1280xf64>
    %4095 = stablehlo.reduce(%4094 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4096 = stablehlo.reshape %4095 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4097 = stablehlo.broadcast_in_dim %4096, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4098 = stablehlo.divide %4097, %142 : tensor<1x256x1xf64>
    %4099 = stablehlo.convert %4098 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4100 = stablehlo.reduce(%4085 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4101 = stablehlo.reshape %4100 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4102 = stablehlo.broadcast_in_dim %4101, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4103 = stablehlo.divide %4102, %158 : tensor<1x256x1xf32>
    %4104 = stablehlo.broadcast_in_dim %4099, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4105 = stablehlo.add %4104, %161 : tensor<1x256x1xf32>
    %4106 = stablehlo.rsqrt %4105 : tensor<1x256x1xf32>
    %4107 = stablehlo.broadcast_in_dim %4085, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4108 = stablehlo.broadcast_in_dim %4103, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4109 = stablehlo.subtract %4107, %4108 : tensor<1x256x1280xf32>
    %4110 = stablehlo.broadcast_in_dim %4109, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4111 = stablehlo.broadcast_in_dim %4106, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4112 = stablehlo.multiply %4110, %4111 : tensor<1x256x1280xf32>
    %4113 = stablehlo.convert %arg85 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4114 = stablehlo.broadcast_in_dim %4112, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4115 = stablehlo.broadcast_in_dim %4113, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4116 = stablehlo.multiply %4114, %4115 : tensor<1x256x1280xf32>
    %4117 = stablehlo.convert %arg86 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4118 = stablehlo.broadcast_in_dim %4116, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4119 = stablehlo.broadcast_in_dim %4117, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4120 = stablehlo.add %4118, %4119 : tensor<1x256x1280xf32>
    %4121 = stablehlo.convert %4120 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4122 = stablehlo.reshape %4121 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4123 = stablehlo.convert %4122 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4124 = stablehlo.dot_general %4123, %arg365, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4125 = stablehlo.broadcast_in_dim %4124, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4126 = stablehlo.multiply %4125, %127 : tensor<256x1280xf32>
    %4127 = stablehlo.broadcast_in_dim %4126, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4128 = stablehlo.broadcast_in_dim %arg366, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4129 = stablehlo.add %4127, %4128 : tensor<256x1280xf32>
    %4130 = stablehlo.convert %4129 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4131 = stablehlo.reshape %4130 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4132 = stablehlo.multiply %4131, %cst_4 : tensor<1x256x1280xbf16>
    %4133 = stablehlo.multiply %4131, %190 : tensor<1x256x1280xbf16>
    %4134 = stablehlo.convert %4133 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4135 = stablehlo.clamp %cst_5, %4134, %cst_6 : tensor<1x256x1280xf32>
    %4136 = stablehlo.multiply %4135, %4135 : tensor<1x256x1280xf32>
    %4137 = stablehlo.multiply %cst_7, %4136 : tensor<1x256x1280xf32>
    %4138 = stablehlo.add %4137, %cst_8 : tensor<1x256x1280xf32>
    %4139 = stablehlo.multiply %4138, %4136 : tensor<1x256x1280xf32>
    %4140 = stablehlo.add %4139, %cst_9 : tensor<1x256x1280xf32>
    %4141 = stablehlo.multiply %4140, %4136 : tensor<1x256x1280xf32>
    %4142 = stablehlo.add %4141, %cst_10 : tensor<1x256x1280xf32>
    %4143 = stablehlo.multiply %4142, %4136 : tensor<1x256x1280xf32>
    %4144 = stablehlo.add %4143, %cst_11 : tensor<1x256x1280xf32>
    %4145 = stablehlo.multiply %4144, %4136 : tensor<1x256x1280xf32>
    %4146 = stablehlo.add %4145, %cst_12 : tensor<1x256x1280xf32>
    %4147 = stablehlo.multiply %4146, %4136 : tensor<1x256x1280xf32>
    %4148 = stablehlo.add %4147, %cst_13 : tensor<1x256x1280xf32>
    %4149 = stablehlo.multiply %cst_14, %4136 : tensor<1x256x1280xf32>
    %4150 = stablehlo.add %4149, %cst_15 : tensor<1x256x1280xf32>
    %4151 = stablehlo.multiply %4150, %4136 : tensor<1x256x1280xf32>
    %4152 = stablehlo.add %4151, %cst_16 : tensor<1x256x1280xf32>
    %4153 = stablehlo.multiply %4152, %4136 : tensor<1x256x1280xf32>
    %4154 = stablehlo.add %4153, %cst_17 : tensor<1x256x1280xf32>
    %4155 = stablehlo.multiply %4154, %4136 : tensor<1x256x1280xf32>
    %4156 = stablehlo.add %4155, %cst_18 : tensor<1x256x1280xf32>
    %4157 = stablehlo.multiply %4135, %4148 : tensor<1x256x1280xf32>
    %4158 = stablehlo.divide %4157, %4156 : tensor<1x256x1280xf32>
    %4159 = stablehlo.clamp %cst_19, %4158, %cst_20 : tensor<1x256x1280xf32>
    %4160 = stablehlo.convert %4159 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4161 = stablehlo.add %4160, %cst_2 : tensor<1x256x1280xbf16>
    %4162 = stablehlo.multiply %4161, %4132 : tensor<1x256x1280xbf16>
    %4163 = stablehlo.reshape %4162 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4164 = stablehlo.convert %4163 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4165 = stablehlo.dot_general %4164, %arg367, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4166 = stablehlo.broadcast_in_dim %4165, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4167 = stablehlo.multiply %4166, %127 : tensor<256x1280xf32>
    %4168 = stablehlo.broadcast_in_dim %4167, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4169 = stablehlo.broadcast_in_dim %arg368, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4170 = stablehlo.add %4168, %4169 : tensor<256x1280xf32>
    %4171 = stablehlo.convert %4170 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4172 = stablehlo.reshape %4171 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4173 = stablehlo.add %4172, %4084 : tensor<1x256x1280xbf16>
    %4174 = stablehlo.convert %4173 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4175 = stablehlo.convert %4174 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4176 = stablehlo.reduce(%4175 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4177 = stablehlo.reshape %4176 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4178 = stablehlo.broadcast_in_dim %4177, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4179 = stablehlo.divide %4178, %142 : tensor<1x256x1xf64>
    %4180 = stablehlo.broadcast_in_dim %4175, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4181 = stablehlo.broadcast_in_dim %4179, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4182 = stablehlo.subtract %4180, %4181 : tensor<1x256x1280xf64>
    %4183 = stablehlo.multiply %4182, %4182 : tensor<1x256x1280xf64>
    %4184 = stablehlo.reduce(%4183 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4185 = stablehlo.reshape %4184 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4186 = stablehlo.broadcast_in_dim %4185, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4187 = stablehlo.divide %4186, %142 : tensor<1x256x1xf64>
    %4188 = stablehlo.convert %4187 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4189 = stablehlo.reduce(%4174 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4190 = stablehlo.reshape %4189 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4191 = stablehlo.broadcast_in_dim %4190, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4192 = stablehlo.divide %4191, %158 : tensor<1x256x1xf32>
    %4193 = stablehlo.broadcast_in_dim %4188, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4194 = stablehlo.add %4193, %161 : tensor<1x256x1xf32>
    %4195 = stablehlo.rsqrt %4194 : tensor<1x256x1xf32>
    %4196 = stablehlo.broadcast_in_dim %4174, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4197 = stablehlo.broadcast_in_dim %4192, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4198 = stablehlo.subtract %4196, %4197 : tensor<1x256x1280xf32>
    %4199 = stablehlo.broadcast_in_dim %4198, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4200 = stablehlo.broadcast_in_dim %4195, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4201 = stablehlo.multiply %4199, %4200 : tensor<1x256x1280xf32>
    %4202 = stablehlo.convert %arg87 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4203 = stablehlo.broadcast_in_dim %4201, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4204 = stablehlo.broadcast_in_dim %4202, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4205 = stablehlo.multiply %4203, %4204 : tensor<1x256x1280xf32>
    %4206 = stablehlo.convert %arg88 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4207 = stablehlo.broadcast_in_dim %4205, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4208 = stablehlo.broadcast_in_dim %4206, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4209 = stablehlo.add %4207, %4208 : tensor<1x256x1280xf32>
    %4210 = stablehlo.convert %4209 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4211 = stablehlo.reshape %4210 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4212 = stablehlo.convert %4211 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4213 = stablehlo.dot_general %4212, %arg369, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4214 = stablehlo.broadcast_in_dim %4213, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4215 = stablehlo.multiply %4214, %273 : tensor<256x256xf32>
    %4216 = stablehlo.broadcast_in_dim %4215, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4217 = stablehlo.broadcast_in_dim %arg370, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4218 = stablehlo.add %4216, %4217 : tensor<256x256xf32>
    %4219 = stablehlo.convert %4218 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4220 = stablehlo.reshape %4219 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4221 = stablehlo.dot_general %4212, %arg371, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4222 = stablehlo.broadcast_in_dim %4221, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4223 = stablehlo.multiply %4222, %273 : tensor<256x256xf32>
    %4224 = stablehlo.broadcast_in_dim %4223, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4225 = stablehlo.broadcast_in_dim %arg372, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4226 = stablehlo.add %4224, %4225 : tensor<256x256xf32>
    %4227 = stablehlo.convert %4226 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4228 = stablehlo.reshape %4227 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4229 = stablehlo.dot_general %4212, %arg373, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4230 = stablehlo.broadcast_in_dim %4229, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4231 = stablehlo.multiply %4230, %127 : tensor<256x1280xf32>
    %4232 = stablehlo.broadcast_in_dim %4231, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4233 = stablehlo.broadcast_in_dim %arg374, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4234 = stablehlo.add %4232, %4233 : tensor<256x1280xf32>
    %4235 = stablehlo.convert %4234 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4236 = stablehlo.reshape %4235 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4237 = stablehlo.reshape %4220 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4238 = stablehlo.transpose %4237, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4239 = stablehlo.reshape %4228 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4240 = stablehlo.transpose %4239, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4241 = stablehlo.reshape %4236 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %4242 = stablehlo.transpose %4241, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4243 = stablehlo.transpose %4240, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %4244 = stablehlo.reshape %4238 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %4245 = stablehlo.reshape %4243 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4246 = stablehlo.broadcast_in_dim %4245, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4247 = stablehlo.dot_general %4244, %4246, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %4248 = stablehlo.reshape %4247 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4249 = stablehlo.broadcast_in_dim %4248, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4250 = stablehlo.divide %4249, %309 : tensor<1x8x256x256xbf16>
    %4251 = stablehlo.convert %4250 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %4252 = stablehlo.reduce(%4251 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4253 = stablehlo.reshape %4252 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4254 = stablehlo.broadcast_in_dim %4251, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4255 = stablehlo.broadcast_in_dim %4253, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4256 = stablehlo.subtract %4254, %4255 : tensor<1x8x256x256xf32>
    %4257 = stablehlo.exponential %4256 : tensor<1x8x256x256xf32>
    %4258 = stablehlo.reduce(%4257 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4259 = stablehlo.reshape %4258 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4260 = stablehlo.broadcast_in_dim %4257, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4261 = stablehlo.broadcast_in_dim %4259, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4262 = stablehlo.divide %4260, %4261 : tensor<1x8x256x256xf32>
    %4263 = stablehlo.convert %4262 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %4264 = stablehlo.reshape %4263 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %4265 = stablehlo.reshape %4242 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4266 = stablehlo.broadcast_in_dim %4265, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4267 = stablehlo.dot_general %4264, %4266, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4268 = stablehlo.reshape %4267 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4269 = stablehlo.transpose %4268, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %4270 = stablehlo.reshape %4269 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %4271 = stablehlo.reshape %4270 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4272 = stablehlo.convert %4271 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4273 = stablehlo.dot_general %4272, %arg375, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4274 = stablehlo.broadcast_in_dim %4273, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4275 = stablehlo.multiply %4274, %127 : tensor<256x1280xf32>
    %4276 = stablehlo.broadcast_in_dim %4275, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4277 = stablehlo.broadcast_in_dim %arg376, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4278 = stablehlo.add %4276, %4277 : tensor<256x1280xf32>
    %4279 = stablehlo.convert %4278 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4280 = stablehlo.reshape %4279 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4281 = stablehlo.add %4280, %4173 : tensor<1x256x1280xbf16>
    %4282 = stablehlo.convert %4281 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4283 = stablehlo.convert %4282 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4284 = stablehlo.reduce(%4283 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4285 = stablehlo.reshape %4284 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4286 = stablehlo.broadcast_in_dim %4285, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4287 = stablehlo.divide %4286, %142 : tensor<1x256x1xf64>
    %4288 = stablehlo.broadcast_in_dim %4283, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4289 = stablehlo.broadcast_in_dim %4287, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4290 = stablehlo.subtract %4288, %4289 : tensor<1x256x1280xf64>
    %4291 = stablehlo.multiply %4290, %4290 : tensor<1x256x1280xf64>
    %4292 = stablehlo.reduce(%4291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4293 = stablehlo.reshape %4292 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4294 = stablehlo.broadcast_in_dim %4293, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4295 = stablehlo.divide %4294, %142 : tensor<1x256x1xf64>
    %4296 = stablehlo.convert %4295 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4297 = stablehlo.reduce(%4282 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4298 = stablehlo.reshape %4297 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4299 = stablehlo.broadcast_in_dim %4298, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4300 = stablehlo.divide %4299, %158 : tensor<1x256x1xf32>
    %4301 = stablehlo.broadcast_in_dim %4296, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4302 = stablehlo.add %4301, %161 : tensor<1x256x1xf32>
    %4303 = stablehlo.rsqrt %4302 : tensor<1x256x1xf32>
    %4304 = stablehlo.broadcast_in_dim %4282, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4305 = stablehlo.broadcast_in_dim %4300, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4306 = stablehlo.subtract %4304, %4305 : tensor<1x256x1280xf32>
    %4307 = stablehlo.broadcast_in_dim %4306, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4308 = stablehlo.broadcast_in_dim %4303, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4309 = stablehlo.multiply %4307, %4308 : tensor<1x256x1280xf32>
    %4310 = stablehlo.convert %arg89 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4311 = stablehlo.broadcast_in_dim %4309, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4312 = stablehlo.broadcast_in_dim %4310, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4313 = stablehlo.multiply %4311, %4312 : tensor<1x256x1280xf32>
    %4314 = stablehlo.convert %arg90 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4315 = stablehlo.broadcast_in_dim %4313, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4316 = stablehlo.broadcast_in_dim %4314, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4317 = stablehlo.add %4315, %4316 : tensor<1x256x1280xf32>
    %4318 = stablehlo.convert %4317 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4319 = stablehlo.reshape %4318 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4320 = stablehlo.convert %4319 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4321 = stablehlo.dot_general %4320, %arg377, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4322 = stablehlo.broadcast_in_dim %4321, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4323 = stablehlo.multiply %4322, %127 : tensor<256x1280xf32>
    %4324 = stablehlo.broadcast_in_dim %4323, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4325 = stablehlo.broadcast_in_dim %arg378, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4326 = stablehlo.add %4324, %4325 : tensor<256x1280xf32>
    %4327 = stablehlo.convert %4326 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4328 = stablehlo.reshape %4327 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4329 = stablehlo.multiply %4328, %cst_4 : tensor<1x256x1280xbf16>
    %4330 = stablehlo.multiply %4328, %190 : tensor<1x256x1280xbf16>
    %4331 = stablehlo.convert %4330 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4332 = stablehlo.clamp %cst_5, %4331, %cst_6 : tensor<1x256x1280xf32>
    %4333 = stablehlo.multiply %4332, %4332 : tensor<1x256x1280xf32>
    %4334 = stablehlo.multiply %cst_7, %4333 : tensor<1x256x1280xf32>
    %4335 = stablehlo.add %4334, %cst_8 : tensor<1x256x1280xf32>
    %4336 = stablehlo.multiply %4335, %4333 : tensor<1x256x1280xf32>
    %4337 = stablehlo.add %4336, %cst_9 : tensor<1x256x1280xf32>
    %4338 = stablehlo.multiply %4337, %4333 : tensor<1x256x1280xf32>
    %4339 = stablehlo.add %4338, %cst_10 : tensor<1x256x1280xf32>
    %4340 = stablehlo.multiply %4339, %4333 : tensor<1x256x1280xf32>
    %4341 = stablehlo.add %4340, %cst_11 : tensor<1x256x1280xf32>
    %4342 = stablehlo.multiply %4341, %4333 : tensor<1x256x1280xf32>
    %4343 = stablehlo.add %4342, %cst_12 : tensor<1x256x1280xf32>
    %4344 = stablehlo.multiply %4343, %4333 : tensor<1x256x1280xf32>
    %4345 = stablehlo.add %4344, %cst_13 : tensor<1x256x1280xf32>
    %4346 = stablehlo.multiply %cst_14, %4333 : tensor<1x256x1280xf32>
    %4347 = stablehlo.add %4346, %cst_15 : tensor<1x256x1280xf32>
    %4348 = stablehlo.multiply %4347, %4333 : tensor<1x256x1280xf32>
    %4349 = stablehlo.add %4348, %cst_16 : tensor<1x256x1280xf32>
    %4350 = stablehlo.multiply %4349, %4333 : tensor<1x256x1280xf32>
    %4351 = stablehlo.add %4350, %cst_17 : tensor<1x256x1280xf32>
    %4352 = stablehlo.multiply %4351, %4333 : tensor<1x256x1280xf32>
    %4353 = stablehlo.add %4352, %cst_18 : tensor<1x256x1280xf32>
    %4354 = stablehlo.multiply %4332, %4345 : tensor<1x256x1280xf32>
    %4355 = stablehlo.divide %4354, %4353 : tensor<1x256x1280xf32>
    %4356 = stablehlo.clamp %cst_19, %4355, %cst_20 : tensor<1x256x1280xf32>
    %4357 = stablehlo.convert %4356 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4358 = stablehlo.add %4357, %cst_2 : tensor<1x256x1280xbf16>
    %4359 = stablehlo.multiply %4358, %4329 : tensor<1x256x1280xbf16>
    %4360 = stablehlo.reshape %4359 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4361 = stablehlo.convert %4360 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4362 = stablehlo.dot_general %4361, %arg379, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4363 = stablehlo.broadcast_in_dim %4362, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4364 = stablehlo.multiply %4363, %127 : tensor<256x1280xf32>
    %4365 = stablehlo.broadcast_in_dim %4364, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4366 = stablehlo.broadcast_in_dim %arg380, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4367 = stablehlo.add %4365, %4366 : tensor<256x1280xf32>
    %4368 = stablehlo.convert %4367 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4369 = stablehlo.reshape %4368 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4370 = stablehlo.add %4369, %4281 : tensor<1x256x1280xbf16>
    %4371 = stablehlo.convert %4370 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4372 = stablehlo.convert %4371 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4373 = stablehlo.reduce(%4372 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4374 = stablehlo.reshape %4373 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4375 = stablehlo.broadcast_in_dim %4374, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4376 = stablehlo.divide %4375, %142 : tensor<1x256x1xf64>
    %4377 = stablehlo.broadcast_in_dim %4372, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4378 = stablehlo.broadcast_in_dim %4376, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4379 = stablehlo.subtract %4377, %4378 : tensor<1x256x1280xf64>
    %4380 = stablehlo.multiply %4379, %4379 : tensor<1x256x1280xf64>
    %4381 = stablehlo.reduce(%4380 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4382 = stablehlo.reshape %4381 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4383 = stablehlo.broadcast_in_dim %4382, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4384 = stablehlo.divide %4383, %142 : tensor<1x256x1xf64>
    %4385 = stablehlo.convert %4384 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4386 = stablehlo.reduce(%4371 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4387 = stablehlo.reshape %4386 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4388 = stablehlo.broadcast_in_dim %4387, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4389 = stablehlo.divide %4388, %158 : tensor<1x256x1xf32>
    %4390 = stablehlo.broadcast_in_dim %4385, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4391 = stablehlo.add %4390, %161 : tensor<1x256x1xf32>
    %4392 = stablehlo.rsqrt %4391 : tensor<1x256x1xf32>
    %4393 = stablehlo.broadcast_in_dim %4371, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4394 = stablehlo.broadcast_in_dim %4389, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4395 = stablehlo.subtract %4393, %4394 : tensor<1x256x1280xf32>
    %4396 = stablehlo.broadcast_in_dim %4395, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4397 = stablehlo.broadcast_in_dim %4392, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4398 = stablehlo.multiply %4396, %4397 : tensor<1x256x1280xf32>
    %4399 = stablehlo.convert %arg91 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4400 = stablehlo.broadcast_in_dim %4398, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4401 = stablehlo.broadcast_in_dim %4399, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4402 = stablehlo.multiply %4400, %4401 : tensor<1x256x1280xf32>
    %4403 = stablehlo.convert %arg92 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4404 = stablehlo.broadcast_in_dim %4402, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4405 = stablehlo.broadcast_in_dim %4403, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4406 = stablehlo.add %4404, %4405 : tensor<1x256x1280xf32>
    %4407 = stablehlo.convert %4406 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4408 = stablehlo.reshape %4407 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4409 = stablehlo.convert %4408 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4410 = stablehlo.dot_general %4409, %arg381, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4411 = stablehlo.broadcast_in_dim %4410, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4412 = stablehlo.multiply %4411, %273 : tensor<256x256xf32>
    %4413 = stablehlo.broadcast_in_dim %4412, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4414 = stablehlo.broadcast_in_dim %arg382, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4415 = stablehlo.add %4413, %4414 : tensor<256x256xf32>
    %4416 = stablehlo.convert %4415 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4417 = stablehlo.reshape %4416 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4418 = stablehlo.dot_general %4409, %arg383, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4419 = stablehlo.broadcast_in_dim %4418, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4420 = stablehlo.multiply %4419, %273 : tensor<256x256xf32>
    %4421 = stablehlo.broadcast_in_dim %4420, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4422 = stablehlo.broadcast_in_dim %arg384, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4423 = stablehlo.add %4421, %4422 : tensor<256x256xf32>
    %4424 = stablehlo.convert %4423 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4425 = stablehlo.reshape %4424 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4426 = stablehlo.dot_general %4409, %arg385, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4427 = stablehlo.broadcast_in_dim %4426, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4428 = stablehlo.multiply %4427, %127 : tensor<256x1280xf32>
    %4429 = stablehlo.broadcast_in_dim %4428, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4430 = stablehlo.broadcast_in_dim %arg386, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4431 = stablehlo.add %4429, %4430 : tensor<256x1280xf32>
    %4432 = stablehlo.convert %4431 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4433 = stablehlo.reshape %4432 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4434 = stablehlo.reshape %4417 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4435 = stablehlo.transpose %4434, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4436 = stablehlo.reshape %4425 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4437 = stablehlo.transpose %4436, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4438 = stablehlo.reshape %4433 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %4439 = stablehlo.transpose %4438, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4440 = stablehlo.transpose %4437, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %4441 = stablehlo.reshape %4435 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %4442 = stablehlo.reshape %4440 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4443 = stablehlo.broadcast_in_dim %4442, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4444 = stablehlo.dot_general %4441, %4443, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %4445 = stablehlo.reshape %4444 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4446 = stablehlo.broadcast_in_dim %4445, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4447 = stablehlo.divide %4446, %309 : tensor<1x8x256x256xbf16>
    %4448 = stablehlo.convert %4447 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %4449 = stablehlo.reduce(%4448 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4450 = stablehlo.reshape %4449 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4451 = stablehlo.broadcast_in_dim %4448, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4452 = stablehlo.broadcast_in_dim %4450, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4453 = stablehlo.subtract %4451, %4452 : tensor<1x8x256x256xf32>
    %4454 = stablehlo.exponential %4453 : tensor<1x8x256x256xf32>
    %4455 = stablehlo.reduce(%4454 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4456 = stablehlo.reshape %4455 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4457 = stablehlo.broadcast_in_dim %4454, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4458 = stablehlo.broadcast_in_dim %4456, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4459 = stablehlo.divide %4457, %4458 : tensor<1x8x256x256xf32>
    %4460 = stablehlo.convert %4459 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %4461 = stablehlo.reshape %4460 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %4462 = stablehlo.reshape %4439 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4463 = stablehlo.broadcast_in_dim %4462, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4464 = stablehlo.dot_general %4461, %4463, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4465 = stablehlo.reshape %4464 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4466 = stablehlo.transpose %4465, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %4467 = stablehlo.reshape %4466 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %4468 = stablehlo.reshape %4467 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4469 = stablehlo.convert %4468 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4470 = stablehlo.dot_general %4469, %arg387, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4471 = stablehlo.broadcast_in_dim %4470, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4472 = stablehlo.multiply %4471, %127 : tensor<256x1280xf32>
    %4473 = stablehlo.broadcast_in_dim %4472, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4474 = stablehlo.broadcast_in_dim %arg388, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4475 = stablehlo.add %4473, %4474 : tensor<256x1280xf32>
    %4476 = stablehlo.convert %4475 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4477 = stablehlo.reshape %4476 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4478 = stablehlo.add %4477, %4370 : tensor<1x256x1280xbf16>
    %4479 = stablehlo.convert %4478 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4480 = stablehlo.convert %4479 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4481 = stablehlo.reduce(%4480 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4482 = stablehlo.reshape %4481 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4483 = stablehlo.broadcast_in_dim %4482, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4484 = stablehlo.divide %4483, %142 : tensor<1x256x1xf64>
    %4485 = stablehlo.broadcast_in_dim %4480, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4486 = stablehlo.broadcast_in_dim %4484, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4487 = stablehlo.subtract %4485, %4486 : tensor<1x256x1280xf64>
    %4488 = stablehlo.multiply %4487, %4487 : tensor<1x256x1280xf64>
    %4489 = stablehlo.reduce(%4488 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4490 = stablehlo.reshape %4489 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4491 = stablehlo.broadcast_in_dim %4490, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4492 = stablehlo.divide %4491, %142 : tensor<1x256x1xf64>
    %4493 = stablehlo.convert %4492 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4494 = stablehlo.reduce(%4479 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4495 = stablehlo.reshape %4494 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4496 = stablehlo.broadcast_in_dim %4495, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4497 = stablehlo.divide %4496, %158 : tensor<1x256x1xf32>
    %4498 = stablehlo.broadcast_in_dim %4493, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4499 = stablehlo.add %4498, %161 : tensor<1x256x1xf32>
    %4500 = stablehlo.rsqrt %4499 : tensor<1x256x1xf32>
    %4501 = stablehlo.broadcast_in_dim %4479, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4502 = stablehlo.broadcast_in_dim %4497, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4503 = stablehlo.subtract %4501, %4502 : tensor<1x256x1280xf32>
    %4504 = stablehlo.broadcast_in_dim %4503, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4505 = stablehlo.broadcast_in_dim %4500, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4506 = stablehlo.multiply %4504, %4505 : tensor<1x256x1280xf32>
    %4507 = stablehlo.convert %arg93 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4508 = stablehlo.broadcast_in_dim %4506, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4509 = stablehlo.broadcast_in_dim %4507, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4510 = stablehlo.multiply %4508, %4509 : tensor<1x256x1280xf32>
    %4511 = stablehlo.convert %arg94 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4512 = stablehlo.broadcast_in_dim %4510, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4513 = stablehlo.broadcast_in_dim %4511, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4514 = stablehlo.add %4512, %4513 : tensor<1x256x1280xf32>
    %4515 = stablehlo.convert %4514 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4516 = stablehlo.reshape %4515 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4517 = stablehlo.convert %4516 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4518 = stablehlo.dot_general %4517, %arg389, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4519 = stablehlo.broadcast_in_dim %4518, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4520 = stablehlo.multiply %4519, %127 : tensor<256x1280xf32>
    %4521 = stablehlo.broadcast_in_dim %4520, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4522 = stablehlo.broadcast_in_dim %arg390, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4523 = stablehlo.add %4521, %4522 : tensor<256x1280xf32>
    %4524 = stablehlo.convert %4523 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4525 = stablehlo.reshape %4524 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4526 = stablehlo.multiply %4525, %cst_4 : tensor<1x256x1280xbf16>
    %4527 = stablehlo.multiply %4525, %190 : tensor<1x256x1280xbf16>
    %4528 = stablehlo.convert %4527 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4529 = stablehlo.clamp %cst_5, %4528, %cst_6 : tensor<1x256x1280xf32>
    %4530 = stablehlo.multiply %4529, %4529 : tensor<1x256x1280xf32>
    %4531 = stablehlo.multiply %cst_7, %4530 : tensor<1x256x1280xf32>
    %4532 = stablehlo.add %4531, %cst_8 : tensor<1x256x1280xf32>
    %4533 = stablehlo.multiply %4532, %4530 : tensor<1x256x1280xf32>
    %4534 = stablehlo.add %4533, %cst_9 : tensor<1x256x1280xf32>
    %4535 = stablehlo.multiply %4534, %4530 : tensor<1x256x1280xf32>
    %4536 = stablehlo.add %4535, %cst_10 : tensor<1x256x1280xf32>
    %4537 = stablehlo.multiply %4536, %4530 : tensor<1x256x1280xf32>
    %4538 = stablehlo.add %4537, %cst_11 : tensor<1x256x1280xf32>
    %4539 = stablehlo.multiply %4538, %4530 : tensor<1x256x1280xf32>
    %4540 = stablehlo.add %4539, %cst_12 : tensor<1x256x1280xf32>
    %4541 = stablehlo.multiply %4540, %4530 : tensor<1x256x1280xf32>
    %4542 = stablehlo.add %4541, %cst_13 : tensor<1x256x1280xf32>
    %4543 = stablehlo.multiply %cst_14, %4530 : tensor<1x256x1280xf32>
    %4544 = stablehlo.add %4543, %cst_15 : tensor<1x256x1280xf32>
    %4545 = stablehlo.multiply %4544, %4530 : tensor<1x256x1280xf32>
    %4546 = stablehlo.add %4545, %cst_16 : tensor<1x256x1280xf32>
    %4547 = stablehlo.multiply %4546, %4530 : tensor<1x256x1280xf32>
    %4548 = stablehlo.add %4547, %cst_17 : tensor<1x256x1280xf32>
    %4549 = stablehlo.multiply %4548, %4530 : tensor<1x256x1280xf32>
    %4550 = stablehlo.add %4549, %cst_18 : tensor<1x256x1280xf32>
    %4551 = stablehlo.multiply %4529, %4542 : tensor<1x256x1280xf32>
    %4552 = stablehlo.divide %4551, %4550 : tensor<1x256x1280xf32>
    %4553 = stablehlo.clamp %cst_19, %4552, %cst_20 : tensor<1x256x1280xf32>
    %4554 = stablehlo.convert %4553 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4555 = stablehlo.add %4554, %cst_2 : tensor<1x256x1280xbf16>
    %4556 = stablehlo.multiply %4555, %4526 : tensor<1x256x1280xbf16>
    %4557 = stablehlo.reshape %4556 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4558 = stablehlo.convert %4557 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4559 = stablehlo.dot_general %4558, %arg391, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4560 = stablehlo.broadcast_in_dim %4559, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4561 = stablehlo.multiply %4560, %127 : tensor<256x1280xf32>
    %4562 = stablehlo.broadcast_in_dim %4561, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4563 = stablehlo.broadcast_in_dim %arg392, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4564 = stablehlo.add %4562, %4563 : tensor<256x1280xf32>
    %4565 = stablehlo.convert %4564 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4566 = stablehlo.reshape %4565 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4567 = stablehlo.add %4566, %4478 : tensor<1x256x1280xbf16>
    %4568 = stablehlo.convert %4567 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4569 = stablehlo.convert %4568 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4570 = stablehlo.reduce(%4569 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4571 = stablehlo.reshape %4570 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4572 = stablehlo.broadcast_in_dim %4571, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4573 = stablehlo.divide %4572, %142 : tensor<1x256x1xf64>
    %4574 = stablehlo.broadcast_in_dim %4569, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4575 = stablehlo.broadcast_in_dim %4573, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4576 = stablehlo.subtract %4574, %4575 : tensor<1x256x1280xf64>
    %4577 = stablehlo.multiply %4576, %4576 : tensor<1x256x1280xf64>
    %4578 = stablehlo.reduce(%4577 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4579 = stablehlo.reshape %4578 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4580 = stablehlo.broadcast_in_dim %4579, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4581 = stablehlo.divide %4580, %142 : tensor<1x256x1xf64>
    %4582 = stablehlo.convert %4581 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4583 = stablehlo.reduce(%4568 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4584 = stablehlo.reshape %4583 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4585 = stablehlo.broadcast_in_dim %4584, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4586 = stablehlo.divide %4585, %158 : tensor<1x256x1xf32>
    %4587 = stablehlo.broadcast_in_dim %4582, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4588 = stablehlo.add %4587, %161 : tensor<1x256x1xf32>
    %4589 = stablehlo.rsqrt %4588 : tensor<1x256x1xf32>
    %4590 = stablehlo.broadcast_in_dim %4568, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4591 = stablehlo.broadcast_in_dim %4586, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4592 = stablehlo.subtract %4590, %4591 : tensor<1x256x1280xf32>
    %4593 = stablehlo.broadcast_in_dim %4592, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4594 = stablehlo.broadcast_in_dim %4589, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4595 = stablehlo.multiply %4593, %4594 : tensor<1x256x1280xf32>
    %4596 = stablehlo.convert %arg95 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4597 = stablehlo.broadcast_in_dim %4595, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4598 = stablehlo.broadcast_in_dim %4596, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4599 = stablehlo.multiply %4597, %4598 : tensor<1x256x1280xf32>
    %4600 = stablehlo.convert %arg96 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4601 = stablehlo.broadcast_in_dim %4599, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4602 = stablehlo.broadcast_in_dim %4600, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4603 = stablehlo.add %4601, %4602 : tensor<1x256x1280xf32>
    %4604 = stablehlo.convert %4603 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4605 = stablehlo.reshape %4604 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4606 = stablehlo.convert %4605 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4607 = stablehlo.dot_general %4606, %arg393, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4608 = stablehlo.broadcast_in_dim %4607, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4609 = stablehlo.multiply %4608, %273 : tensor<256x256xf32>
    %4610 = stablehlo.broadcast_in_dim %4609, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4611 = stablehlo.broadcast_in_dim %arg394, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4612 = stablehlo.add %4610, %4611 : tensor<256x256xf32>
    %4613 = stablehlo.convert %4612 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4614 = stablehlo.reshape %4613 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4615 = stablehlo.dot_general %4606, %arg395, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4616 = stablehlo.broadcast_in_dim %4615, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4617 = stablehlo.multiply %4616, %273 : tensor<256x256xf32>
    %4618 = stablehlo.broadcast_in_dim %4617, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4619 = stablehlo.broadcast_in_dim %arg396, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4620 = stablehlo.add %4618, %4619 : tensor<256x256xf32>
    %4621 = stablehlo.convert %4620 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4622 = stablehlo.reshape %4621 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4623 = stablehlo.dot_general %4606, %arg397, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4624 = stablehlo.broadcast_in_dim %4623, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4625 = stablehlo.multiply %4624, %127 : tensor<256x1280xf32>
    %4626 = stablehlo.broadcast_in_dim %4625, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4627 = stablehlo.broadcast_in_dim %arg398, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4628 = stablehlo.add %4626, %4627 : tensor<256x1280xf32>
    %4629 = stablehlo.convert %4628 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4630 = stablehlo.reshape %4629 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4631 = stablehlo.reshape %4614 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4632 = stablehlo.transpose %4631, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4633 = stablehlo.reshape %4622 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4634 = stablehlo.transpose %4633, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4635 = stablehlo.reshape %4630 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %4636 = stablehlo.transpose %4635, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4637 = stablehlo.transpose %4634, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %4638 = stablehlo.reshape %4632 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %4639 = stablehlo.reshape %4637 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4640 = stablehlo.broadcast_in_dim %4639, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4641 = stablehlo.dot_general %4638, %4640, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %4642 = stablehlo.reshape %4641 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4643 = stablehlo.broadcast_in_dim %4642, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4644 = stablehlo.divide %4643, %309 : tensor<1x8x256x256xbf16>
    %4645 = stablehlo.convert %4644 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %4646 = stablehlo.reduce(%4645 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4647 = stablehlo.reshape %4646 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4648 = stablehlo.broadcast_in_dim %4645, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4649 = stablehlo.broadcast_in_dim %4647, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4650 = stablehlo.subtract %4648, %4649 : tensor<1x8x256x256xf32>
    %4651 = stablehlo.exponential %4650 : tensor<1x8x256x256xf32>
    %4652 = stablehlo.reduce(%4651 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4653 = stablehlo.reshape %4652 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4654 = stablehlo.broadcast_in_dim %4651, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4655 = stablehlo.broadcast_in_dim %4653, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4656 = stablehlo.divide %4654, %4655 : tensor<1x8x256x256xf32>
    %4657 = stablehlo.convert %4656 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %4658 = stablehlo.reshape %4657 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %4659 = stablehlo.reshape %4636 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4660 = stablehlo.broadcast_in_dim %4659, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4661 = stablehlo.dot_general %4658, %4660, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4662 = stablehlo.reshape %4661 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4663 = stablehlo.transpose %4662, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %4664 = stablehlo.reshape %4663 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %4665 = stablehlo.reshape %4664 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4666 = stablehlo.convert %4665 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4667 = stablehlo.dot_general %4666, %arg399, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4668 = stablehlo.broadcast_in_dim %4667, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4669 = stablehlo.multiply %4668, %127 : tensor<256x1280xf32>
    %4670 = stablehlo.broadcast_in_dim %4669, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4671 = stablehlo.broadcast_in_dim %arg400, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4672 = stablehlo.add %4670, %4671 : tensor<256x1280xf32>
    %4673 = stablehlo.convert %4672 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4674 = stablehlo.reshape %4673 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4675 = stablehlo.add %4674, %4567 : tensor<1x256x1280xbf16>
    %4676 = stablehlo.convert %4675 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4677 = stablehlo.convert %4676 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4678 = stablehlo.reduce(%4677 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4679 = stablehlo.reshape %4678 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4680 = stablehlo.broadcast_in_dim %4679, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4681 = stablehlo.divide %4680, %142 : tensor<1x256x1xf64>
    %4682 = stablehlo.broadcast_in_dim %4677, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4683 = stablehlo.broadcast_in_dim %4681, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4684 = stablehlo.subtract %4682, %4683 : tensor<1x256x1280xf64>
    %4685 = stablehlo.multiply %4684, %4684 : tensor<1x256x1280xf64>
    %4686 = stablehlo.reduce(%4685 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4687 = stablehlo.reshape %4686 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4688 = stablehlo.broadcast_in_dim %4687, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4689 = stablehlo.divide %4688, %142 : tensor<1x256x1xf64>
    %4690 = stablehlo.convert %4689 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4691 = stablehlo.reduce(%4676 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4692 = stablehlo.reshape %4691 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4693 = stablehlo.broadcast_in_dim %4692, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4694 = stablehlo.divide %4693, %158 : tensor<1x256x1xf32>
    %4695 = stablehlo.broadcast_in_dim %4690, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4696 = stablehlo.add %4695, %161 : tensor<1x256x1xf32>
    %4697 = stablehlo.rsqrt %4696 : tensor<1x256x1xf32>
    %4698 = stablehlo.broadcast_in_dim %4676, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4699 = stablehlo.broadcast_in_dim %4694, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4700 = stablehlo.subtract %4698, %4699 : tensor<1x256x1280xf32>
    %4701 = stablehlo.broadcast_in_dim %4700, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4702 = stablehlo.broadcast_in_dim %4697, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4703 = stablehlo.multiply %4701, %4702 : tensor<1x256x1280xf32>
    %4704 = stablehlo.convert %arg97 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4705 = stablehlo.broadcast_in_dim %4703, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4706 = stablehlo.broadcast_in_dim %4704, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4707 = stablehlo.multiply %4705, %4706 : tensor<1x256x1280xf32>
    %4708 = stablehlo.convert %arg98 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4709 = stablehlo.broadcast_in_dim %4707, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4710 = stablehlo.broadcast_in_dim %4708, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4711 = stablehlo.add %4709, %4710 : tensor<1x256x1280xf32>
    %4712 = stablehlo.convert %4711 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4713 = stablehlo.reshape %4712 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4714 = stablehlo.convert %4713 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4715 = stablehlo.dot_general %4714, %arg401, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4716 = stablehlo.broadcast_in_dim %4715, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4717 = stablehlo.multiply %4716, %127 : tensor<256x1280xf32>
    %4718 = stablehlo.broadcast_in_dim %4717, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4719 = stablehlo.broadcast_in_dim %arg402, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4720 = stablehlo.add %4718, %4719 : tensor<256x1280xf32>
    %4721 = stablehlo.convert %4720 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4722 = stablehlo.reshape %4721 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4723 = stablehlo.multiply %4722, %cst_4 : tensor<1x256x1280xbf16>
    %4724 = stablehlo.multiply %4722, %190 : tensor<1x256x1280xbf16>
    %4725 = stablehlo.convert %4724 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4726 = stablehlo.clamp %cst_5, %4725, %cst_6 : tensor<1x256x1280xf32>
    %4727 = stablehlo.multiply %4726, %4726 : tensor<1x256x1280xf32>
    %4728 = stablehlo.multiply %cst_7, %4727 : tensor<1x256x1280xf32>
    %4729 = stablehlo.add %4728, %cst_8 : tensor<1x256x1280xf32>
    %4730 = stablehlo.multiply %4729, %4727 : tensor<1x256x1280xf32>
    %4731 = stablehlo.add %4730, %cst_9 : tensor<1x256x1280xf32>
    %4732 = stablehlo.multiply %4731, %4727 : tensor<1x256x1280xf32>
    %4733 = stablehlo.add %4732, %cst_10 : tensor<1x256x1280xf32>
    %4734 = stablehlo.multiply %4733, %4727 : tensor<1x256x1280xf32>
    %4735 = stablehlo.add %4734, %cst_11 : tensor<1x256x1280xf32>
    %4736 = stablehlo.multiply %4735, %4727 : tensor<1x256x1280xf32>
    %4737 = stablehlo.add %4736, %cst_12 : tensor<1x256x1280xf32>
    %4738 = stablehlo.multiply %4737, %4727 : tensor<1x256x1280xf32>
    %4739 = stablehlo.add %4738, %cst_13 : tensor<1x256x1280xf32>
    %4740 = stablehlo.multiply %cst_14, %4727 : tensor<1x256x1280xf32>
    %4741 = stablehlo.add %4740, %cst_15 : tensor<1x256x1280xf32>
    %4742 = stablehlo.multiply %4741, %4727 : tensor<1x256x1280xf32>
    %4743 = stablehlo.add %4742, %cst_16 : tensor<1x256x1280xf32>
    %4744 = stablehlo.multiply %4743, %4727 : tensor<1x256x1280xf32>
    %4745 = stablehlo.add %4744, %cst_17 : tensor<1x256x1280xf32>
    %4746 = stablehlo.multiply %4745, %4727 : tensor<1x256x1280xf32>
    %4747 = stablehlo.add %4746, %cst_18 : tensor<1x256x1280xf32>
    %4748 = stablehlo.multiply %4726, %4739 : tensor<1x256x1280xf32>
    %4749 = stablehlo.divide %4748, %4747 : tensor<1x256x1280xf32>
    %4750 = stablehlo.clamp %cst_19, %4749, %cst_20 : tensor<1x256x1280xf32>
    %4751 = stablehlo.convert %4750 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4752 = stablehlo.add %4751, %cst_2 : tensor<1x256x1280xbf16>
    %4753 = stablehlo.multiply %4752, %4723 : tensor<1x256x1280xbf16>
    %4754 = stablehlo.reshape %4753 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4755 = stablehlo.convert %4754 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4756 = stablehlo.dot_general %4755, %arg403, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4757 = stablehlo.broadcast_in_dim %4756, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4758 = stablehlo.multiply %4757, %127 : tensor<256x1280xf32>
    %4759 = stablehlo.broadcast_in_dim %4758, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4760 = stablehlo.broadcast_in_dim %arg404, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4761 = stablehlo.add %4759, %4760 : tensor<256x1280xf32>
    %4762 = stablehlo.convert %4761 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4763 = stablehlo.reshape %4762 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4764 = stablehlo.add %4763, %4675 : tensor<1x256x1280xbf16>
    %4765 = stablehlo.convert %4764 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4766 = stablehlo.convert %4765 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4767 = stablehlo.reduce(%4766 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4768 = stablehlo.reshape %4767 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4769 = stablehlo.broadcast_in_dim %4768, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4770 = stablehlo.divide %4769, %142 : tensor<1x256x1xf64>
    %4771 = stablehlo.broadcast_in_dim %4766, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4772 = stablehlo.broadcast_in_dim %4770, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4773 = stablehlo.subtract %4771, %4772 : tensor<1x256x1280xf64>
    %4774 = stablehlo.multiply %4773, %4773 : tensor<1x256x1280xf64>
    %4775 = stablehlo.reduce(%4774 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4776 = stablehlo.reshape %4775 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4777 = stablehlo.broadcast_in_dim %4776, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4778 = stablehlo.divide %4777, %142 : tensor<1x256x1xf64>
    %4779 = stablehlo.convert %4778 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4780 = stablehlo.reduce(%4765 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4781 = stablehlo.reshape %4780 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4782 = stablehlo.broadcast_in_dim %4781, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4783 = stablehlo.divide %4782, %158 : tensor<1x256x1xf32>
    %4784 = stablehlo.broadcast_in_dim %4779, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4785 = stablehlo.add %4784, %161 : tensor<1x256x1xf32>
    %4786 = stablehlo.rsqrt %4785 : tensor<1x256x1xf32>
    %4787 = stablehlo.broadcast_in_dim %4765, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4788 = stablehlo.broadcast_in_dim %4783, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4789 = stablehlo.subtract %4787, %4788 : tensor<1x256x1280xf32>
    %4790 = stablehlo.broadcast_in_dim %4789, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4791 = stablehlo.broadcast_in_dim %4786, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4792 = stablehlo.multiply %4790, %4791 : tensor<1x256x1280xf32>
    %4793 = stablehlo.convert %arg99 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4794 = stablehlo.broadcast_in_dim %4792, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4795 = stablehlo.broadcast_in_dim %4793, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4796 = stablehlo.multiply %4794, %4795 : tensor<1x256x1280xf32>
    %4797 = stablehlo.convert %arg100 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4798 = stablehlo.broadcast_in_dim %4796, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4799 = stablehlo.broadcast_in_dim %4797, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4800 = stablehlo.add %4798, %4799 : tensor<1x256x1280xf32>
    %4801 = stablehlo.convert %4800 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4802 = stablehlo.reshape %4801 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4803 = stablehlo.convert %4802 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4804 = stablehlo.dot_general %4803, %arg405, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4805 = stablehlo.broadcast_in_dim %4804, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4806 = stablehlo.multiply %4805, %273 : tensor<256x256xf32>
    %4807 = stablehlo.broadcast_in_dim %4806, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4808 = stablehlo.broadcast_in_dim %arg406, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4809 = stablehlo.add %4807, %4808 : tensor<256x256xf32>
    %4810 = stablehlo.convert %4809 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4811 = stablehlo.reshape %4810 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4812 = stablehlo.dot_general %4803, %arg407, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %4813 = stablehlo.broadcast_in_dim %4812, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4814 = stablehlo.multiply %4813, %273 : tensor<256x256xf32>
    %4815 = stablehlo.broadcast_in_dim %4814, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %4816 = stablehlo.broadcast_in_dim %arg408, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %4817 = stablehlo.add %4815, %4816 : tensor<256x256xf32>
    %4818 = stablehlo.convert %4817 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %4819 = stablehlo.reshape %4818 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %4820 = stablehlo.dot_general %4803, %arg409, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4821 = stablehlo.broadcast_in_dim %4820, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4822 = stablehlo.multiply %4821, %127 : tensor<256x1280xf32>
    %4823 = stablehlo.broadcast_in_dim %4822, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4824 = stablehlo.broadcast_in_dim %arg410, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4825 = stablehlo.add %4823, %4824 : tensor<256x1280xf32>
    %4826 = stablehlo.convert %4825 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4827 = stablehlo.reshape %4826 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4828 = stablehlo.reshape %4811 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4829 = stablehlo.transpose %4828, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4830 = stablehlo.reshape %4819 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %4831 = stablehlo.transpose %4830, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %4832 = stablehlo.reshape %4827 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %4833 = stablehlo.transpose %4832, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4834 = stablehlo.transpose %4831, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %4835 = stablehlo.reshape %4829 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %4836 = stablehlo.reshape %4834 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4837 = stablehlo.broadcast_in_dim %4836, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %4838 = stablehlo.dot_general %4835, %4837, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %4839 = stablehlo.reshape %4838 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4840 = stablehlo.broadcast_in_dim %4839, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %4841 = stablehlo.divide %4840, %309 : tensor<1x8x256x256xbf16>
    %4842 = stablehlo.convert %4841 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %4843 = stablehlo.reduce(%4842 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4844 = stablehlo.reshape %4843 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4845 = stablehlo.broadcast_in_dim %4842, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4846 = stablehlo.broadcast_in_dim %4844, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4847 = stablehlo.subtract %4845, %4846 : tensor<1x8x256x256xf32>
    %4848 = stablehlo.exponential %4847 : tensor<1x8x256x256xf32>
    %4849 = stablehlo.reduce(%4848 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %4850 = stablehlo.reshape %4849 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %4851 = stablehlo.broadcast_in_dim %4848, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %4852 = stablehlo.broadcast_in_dim %4850, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %4853 = stablehlo.divide %4851, %4852 : tensor<1x8x256x256xf32>
    %4854 = stablehlo.convert %4853 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %4855 = stablehlo.reshape %4854 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %4856 = stablehlo.reshape %4833 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4857 = stablehlo.broadcast_in_dim %4856, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4858 = stablehlo.dot_general %4855, %4857, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %4859 = stablehlo.reshape %4858 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %4860 = stablehlo.transpose %4859, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %4861 = stablehlo.reshape %4860 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %4862 = stablehlo.reshape %4861 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4863 = stablehlo.convert %4862 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4864 = stablehlo.dot_general %4863, %arg411, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4865 = stablehlo.broadcast_in_dim %4864, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4866 = stablehlo.multiply %4865, %127 : tensor<256x1280xf32>
    %4867 = stablehlo.broadcast_in_dim %4866, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4868 = stablehlo.broadcast_in_dim %arg412, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4869 = stablehlo.add %4867, %4868 : tensor<256x1280xf32>
    %4870 = stablehlo.convert %4869 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4871 = stablehlo.reshape %4870 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4872 = stablehlo.add %4871, %4764 : tensor<1x256x1280xbf16>
    %4873 = stablehlo.convert %4872 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4874 = stablehlo.convert %4873 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4875 = stablehlo.reduce(%4874 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4876 = stablehlo.reshape %4875 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4877 = stablehlo.broadcast_in_dim %4876, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4878 = stablehlo.divide %4877, %142 : tensor<1x256x1xf64>
    %4879 = stablehlo.broadcast_in_dim %4874, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4880 = stablehlo.broadcast_in_dim %4878, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4881 = stablehlo.subtract %4879, %4880 : tensor<1x256x1280xf64>
    %4882 = stablehlo.multiply %4881, %4881 : tensor<1x256x1280xf64>
    %4883 = stablehlo.reduce(%4882 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4884 = stablehlo.reshape %4883 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4885 = stablehlo.broadcast_in_dim %4884, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4886 = stablehlo.divide %4885, %142 : tensor<1x256x1xf64>
    %4887 = stablehlo.convert %4886 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4888 = stablehlo.reduce(%4873 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4889 = stablehlo.reshape %4888 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4890 = stablehlo.broadcast_in_dim %4889, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4891 = stablehlo.divide %4890, %158 : tensor<1x256x1xf32>
    %4892 = stablehlo.broadcast_in_dim %4887, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4893 = stablehlo.add %4892, %161 : tensor<1x256x1xf32>
    %4894 = stablehlo.rsqrt %4893 : tensor<1x256x1xf32>
    %4895 = stablehlo.broadcast_in_dim %4873, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4896 = stablehlo.broadcast_in_dim %4891, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4897 = stablehlo.subtract %4895, %4896 : tensor<1x256x1280xf32>
    %4898 = stablehlo.broadcast_in_dim %4897, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4899 = stablehlo.broadcast_in_dim %4894, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4900 = stablehlo.multiply %4898, %4899 : tensor<1x256x1280xf32>
    %4901 = stablehlo.convert %arg101 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4902 = stablehlo.broadcast_in_dim %4900, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4903 = stablehlo.broadcast_in_dim %4901, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4904 = stablehlo.multiply %4902, %4903 : tensor<1x256x1280xf32>
    %4905 = stablehlo.convert %arg102 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4906 = stablehlo.broadcast_in_dim %4904, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4907 = stablehlo.broadcast_in_dim %4905, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4908 = stablehlo.add %4906, %4907 : tensor<1x256x1280xf32>
    %4909 = stablehlo.convert %4908 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4910 = stablehlo.reshape %4909 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4911 = stablehlo.convert %4910 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4912 = stablehlo.dot_general %4911, %arg413, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4913 = stablehlo.broadcast_in_dim %4912, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4914 = stablehlo.multiply %4913, %127 : tensor<256x1280xf32>
    %4915 = stablehlo.broadcast_in_dim %4914, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4916 = stablehlo.broadcast_in_dim %arg414, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4917 = stablehlo.add %4915, %4916 : tensor<256x1280xf32>
    %4918 = stablehlo.convert %4917 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4919 = stablehlo.reshape %4918 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4920 = stablehlo.multiply %4919, %cst_4 : tensor<1x256x1280xbf16>
    %4921 = stablehlo.multiply %4919, %190 : tensor<1x256x1280xbf16>
    %4922 = stablehlo.convert %4921 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4923 = stablehlo.clamp %cst_5, %4922, %cst_6 : tensor<1x256x1280xf32>
    %4924 = stablehlo.multiply %4923, %4923 : tensor<1x256x1280xf32>
    %4925 = stablehlo.multiply %cst_7, %4924 : tensor<1x256x1280xf32>
    %4926 = stablehlo.add %4925, %cst_8 : tensor<1x256x1280xf32>
    %4927 = stablehlo.multiply %4926, %4924 : tensor<1x256x1280xf32>
    %4928 = stablehlo.add %4927, %cst_9 : tensor<1x256x1280xf32>
    %4929 = stablehlo.multiply %4928, %4924 : tensor<1x256x1280xf32>
    %4930 = stablehlo.add %4929, %cst_10 : tensor<1x256x1280xf32>
    %4931 = stablehlo.multiply %4930, %4924 : tensor<1x256x1280xf32>
    %4932 = stablehlo.add %4931, %cst_11 : tensor<1x256x1280xf32>
    %4933 = stablehlo.multiply %4932, %4924 : tensor<1x256x1280xf32>
    %4934 = stablehlo.add %4933, %cst_12 : tensor<1x256x1280xf32>
    %4935 = stablehlo.multiply %4934, %4924 : tensor<1x256x1280xf32>
    %4936 = stablehlo.add %4935, %cst_13 : tensor<1x256x1280xf32>
    %4937 = stablehlo.multiply %cst_14, %4924 : tensor<1x256x1280xf32>
    %4938 = stablehlo.add %4937, %cst_15 : tensor<1x256x1280xf32>
    %4939 = stablehlo.multiply %4938, %4924 : tensor<1x256x1280xf32>
    %4940 = stablehlo.add %4939, %cst_16 : tensor<1x256x1280xf32>
    %4941 = stablehlo.multiply %4940, %4924 : tensor<1x256x1280xf32>
    %4942 = stablehlo.add %4941, %cst_17 : tensor<1x256x1280xf32>
    %4943 = stablehlo.multiply %4942, %4924 : tensor<1x256x1280xf32>
    %4944 = stablehlo.add %4943, %cst_18 : tensor<1x256x1280xf32>
    %4945 = stablehlo.multiply %4923, %4936 : tensor<1x256x1280xf32>
    %4946 = stablehlo.divide %4945, %4944 : tensor<1x256x1280xf32>
    %4947 = stablehlo.clamp %cst_19, %4946, %cst_20 : tensor<1x256x1280xf32>
    %4948 = stablehlo.convert %4947 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4949 = stablehlo.add %4948, %cst_2 : tensor<1x256x1280xbf16>
    %4950 = stablehlo.multiply %4949, %4920 : tensor<1x256x1280xbf16>
    %4951 = stablehlo.reshape %4950 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %4952 = stablehlo.convert %4951 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %4953 = stablehlo.dot_general %4952, %arg415, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %4954 = stablehlo.broadcast_in_dim %4953, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4955 = stablehlo.multiply %4954, %127 : tensor<256x1280xf32>
    %4956 = stablehlo.broadcast_in_dim %4955, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4957 = stablehlo.broadcast_in_dim %arg416, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %4958 = stablehlo.add %4956, %4957 : tensor<256x1280xf32>
    %4959 = stablehlo.convert %4958 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %4960 = stablehlo.reshape %4959 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %4961 = stablehlo.add %4960, %4872 : tensor<1x256x1280xbf16>
    %4962 = stablehlo.convert %4961 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %4963 = stablehlo.convert %4962 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %4964 = stablehlo.reduce(%4963 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4965 = stablehlo.reshape %4964 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4966 = stablehlo.broadcast_in_dim %4965, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4967 = stablehlo.divide %4966, %142 : tensor<1x256x1xf64>
    %4968 = stablehlo.broadcast_in_dim %4963, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %4969 = stablehlo.broadcast_in_dim %4967, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %4970 = stablehlo.subtract %4968, %4969 : tensor<1x256x1280xf64>
    %4971 = stablehlo.multiply %4970, %4970 : tensor<1x256x1280xf64>
    %4972 = stablehlo.reduce(%4971 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %4973 = stablehlo.reshape %4972 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %4974 = stablehlo.broadcast_in_dim %4973, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %4975 = stablehlo.divide %4974, %142 : tensor<1x256x1xf64>
    %4976 = stablehlo.convert %4975 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %4977 = stablehlo.reduce(%4962 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %4978 = stablehlo.reshape %4977 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %4979 = stablehlo.broadcast_in_dim %4978, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4980 = stablehlo.divide %4979, %158 : tensor<1x256x1xf32>
    %4981 = stablehlo.broadcast_in_dim %4976, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4982 = stablehlo.add %4981, %161 : tensor<1x256x1xf32>
    %4983 = stablehlo.rsqrt %4982 : tensor<1x256x1xf32>
    %4984 = stablehlo.broadcast_in_dim %4962, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4985 = stablehlo.broadcast_in_dim %4980, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4986 = stablehlo.subtract %4984, %4985 : tensor<1x256x1280xf32>
    %4987 = stablehlo.broadcast_in_dim %4986, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4988 = stablehlo.broadcast_in_dim %4983, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4989 = stablehlo.multiply %4987, %4988 : tensor<1x256x1280xf32>
    %4990 = stablehlo.convert %arg103 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4991 = stablehlo.broadcast_in_dim %4989, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4992 = stablehlo.broadcast_in_dim %4990, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4993 = stablehlo.multiply %4991, %4992 : tensor<1x256x1280xf32>
    %4994 = stablehlo.convert %arg104 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %4995 = stablehlo.broadcast_in_dim %4993, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4996 = stablehlo.broadcast_in_dim %4994, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %4997 = stablehlo.add %4995, %4996 : tensor<1x256x1280xf32>
    %4998 = stablehlo.convert %4997 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %4999 = stablehlo.reshape %4998 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5000 = stablehlo.convert %4999 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5001 = stablehlo.dot_general %5000, %arg417, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %5002 = stablehlo.broadcast_in_dim %5001, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5003 = stablehlo.multiply %5002, %273 : tensor<256x256xf32>
    %5004 = stablehlo.broadcast_in_dim %5003, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5005 = stablehlo.broadcast_in_dim %arg418, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %5006 = stablehlo.add %5004, %5005 : tensor<256x256xf32>
    %5007 = stablehlo.convert %5006 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %5008 = stablehlo.reshape %5007 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %5009 = stablehlo.dot_general %5000, %arg419, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %5010 = stablehlo.broadcast_in_dim %5009, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5011 = stablehlo.multiply %5010, %273 : tensor<256x256xf32>
    %5012 = stablehlo.broadcast_in_dim %5011, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5013 = stablehlo.broadcast_in_dim %arg420, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %5014 = stablehlo.add %5012, %5013 : tensor<256x256xf32>
    %5015 = stablehlo.convert %5014 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %5016 = stablehlo.reshape %5015 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %5017 = stablehlo.dot_general %5000, %arg421, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5018 = stablehlo.broadcast_in_dim %5017, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5019 = stablehlo.multiply %5018, %127 : tensor<256x1280xf32>
    %5020 = stablehlo.broadcast_in_dim %5019, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5021 = stablehlo.broadcast_in_dim %arg422, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5022 = stablehlo.add %5020, %5021 : tensor<256x1280xf32>
    %5023 = stablehlo.convert %5022 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5024 = stablehlo.reshape %5023 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5025 = stablehlo.reshape %5008 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %5026 = stablehlo.transpose %5025, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %5027 = stablehlo.reshape %5016 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %5028 = stablehlo.transpose %5027, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %5029 = stablehlo.reshape %5024 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %5030 = stablehlo.transpose %5029, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %5031 = stablehlo.transpose %5028, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %5032 = stablehlo.reshape %5026 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %5033 = stablehlo.reshape %5031 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %5034 = stablehlo.broadcast_in_dim %5033, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %5035 = stablehlo.dot_general %5032, %5034, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %5036 = stablehlo.reshape %5035 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %5037 = stablehlo.broadcast_in_dim %5036, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %5038 = stablehlo.divide %5037, %309 : tensor<1x8x256x256xbf16>
    %5039 = stablehlo.convert %5038 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %5040 = stablehlo.reduce(%5039 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %5041 = stablehlo.reshape %5040 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %5042 = stablehlo.broadcast_in_dim %5039, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %5043 = stablehlo.broadcast_in_dim %5041, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %5044 = stablehlo.subtract %5042, %5043 : tensor<1x8x256x256xf32>
    %5045 = stablehlo.exponential %5044 : tensor<1x8x256x256xf32>
    %5046 = stablehlo.reduce(%5045 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %5047 = stablehlo.reshape %5046 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %5048 = stablehlo.broadcast_in_dim %5045, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %5049 = stablehlo.broadcast_in_dim %5047, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %5050 = stablehlo.divide %5048, %5049 : tensor<1x8x256x256xf32>
    %5051 = stablehlo.convert %5050 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %5052 = stablehlo.reshape %5051 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %5053 = stablehlo.reshape %5030 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %5054 = stablehlo.broadcast_in_dim %5053, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %5055 = stablehlo.dot_general %5052, %5054, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %5056 = stablehlo.reshape %5055 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %5057 = stablehlo.transpose %5056, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %5058 = stablehlo.reshape %5057 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %5059 = stablehlo.reshape %5058 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5060 = stablehlo.convert %5059 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5061 = stablehlo.dot_general %5060, %arg423, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5062 = stablehlo.broadcast_in_dim %5061, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5063 = stablehlo.multiply %5062, %127 : tensor<256x1280xf32>
    %5064 = stablehlo.broadcast_in_dim %5063, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5065 = stablehlo.broadcast_in_dim %arg424, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5066 = stablehlo.add %5064, %5065 : tensor<256x1280xf32>
    %5067 = stablehlo.convert %5066 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5068 = stablehlo.reshape %5067 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5069 = stablehlo.add %5068, %4961 : tensor<1x256x1280xbf16>
    %5070 = stablehlo.convert %5069 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %5071 = stablehlo.convert %5070 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %5072 = stablehlo.reduce(%5071 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5073 = stablehlo.reshape %5072 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5074 = stablehlo.broadcast_in_dim %5073, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5075 = stablehlo.divide %5074, %142 : tensor<1x256x1xf64>
    %5076 = stablehlo.broadcast_in_dim %5071, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %5077 = stablehlo.broadcast_in_dim %5075, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %5078 = stablehlo.subtract %5076, %5077 : tensor<1x256x1280xf64>
    %5079 = stablehlo.multiply %5078, %5078 : tensor<1x256x1280xf64>
    %5080 = stablehlo.reduce(%5079 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5081 = stablehlo.reshape %5080 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5082 = stablehlo.broadcast_in_dim %5081, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5083 = stablehlo.divide %5082, %142 : tensor<1x256x1xf64>
    %5084 = stablehlo.convert %5083 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %5085 = stablehlo.reduce(%5070 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %5086 = stablehlo.reshape %5085 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %5087 = stablehlo.broadcast_in_dim %5086, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5088 = stablehlo.divide %5087, %158 : tensor<1x256x1xf32>
    %5089 = stablehlo.broadcast_in_dim %5084, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5090 = stablehlo.add %5089, %161 : tensor<1x256x1xf32>
    %5091 = stablehlo.rsqrt %5090 : tensor<1x256x1xf32>
    %5092 = stablehlo.broadcast_in_dim %5070, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5093 = stablehlo.broadcast_in_dim %5088, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5094 = stablehlo.subtract %5092, %5093 : tensor<1x256x1280xf32>
    %5095 = stablehlo.broadcast_in_dim %5094, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5096 = stablehlo.broadcast_in_dim %5091, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5097 = stablehlo.multiply %5095, %5096 : tensor<1x256x1280xf32>
    %5098 = stablehlo.convert %arg105 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5099 = stablehlo.broadcast_in_dim %5097, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5100 = stablehlo.broadcast_in_dim %5098, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5101 = stablehlo.multiply %5099, %5100 : tensor<1x256x1280xf32>
    %5102 = stablehlo.convert %arg106 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5103 = stablehlo.broadcast_in_dim %5101, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5104 = stablehlo.broadcast_in_dim %5102, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5105 = stablehlo.add %5103, %5104 : tensor<1x256x1280xf32>
    %5106 = stablehlo.convert %5105 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %5107 = stablehlo.reshape %5106 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5108 = stablehlo.convert %5107 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5109 = stablehlo.dot_general %5108, %arg425, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5110 = stablehlo.broadcast_in_dim %5109, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5111 = stablehlo.multiply %5110, %127 : tensor<256x1280xf32>
    %5112 = stablehlo.broadcast_in_dim %5111, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5113 = stablehlo.broadcast_in_dim %arg426, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5114 = stablehlo.add %5112, %5113 : tensor<256x1280xf32>
    %5115 = stablehlo.convert %5114 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5116 = stablehlo.reshape %5115 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5117 = stablehlo.multiply %5116, %cst_4 : tensor<1x256x1280xbf16>
    %5118 = stablehlo.multiply %5116, %190 : tensor<1x256x1280xbf16>
    %5119 = stablehlo.convert %5118 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %5120 = stablehlo.clamp %cst_5, %5119, %cst_6 : tensor<1x256x1280xf32>
    %5121 = stablehlo.multiply %5120, %5120 : tensor<1x256x1280xf32>
    %5122 = stablehlo.multiply %cst_7, %5121 : tensor<1x256x1280xf32>
    %5123 = stablehlo.add %5122, %cst_8 : tensor<1x256x1280xf32>
    %5124 = stablehlo.multiply %5123, %5121 : tensor<1x256x1280xf32>
    %5125 = stablehlo.add %5124, %cst_9 : tensor<1x256x1280xf32>
    %5126 = stablehlo.multiply %5125, %5121 : tensor<1x256x1280xf32>
    %5127 = stablehlo.add %5126, %cst_10 : tensor<1x256x1280xf32>
    %5128 = stablehlo.multiply %5127, %5121 : tensor<1x256x1280xf32>
    %5129 = stablehlo.add %5128, %cst_11 : tensor<1x256x1280xf32>
    %5130 = stablehlo.multiply %5129, %5121 : tensor<1x256x1280xf32>
    %5131 = stablehlo.add %5130, %cst_12 : tensor<1x256x1280xf32>
    %5132 = stablehlo.multiply %5131, %5121 : tensor<1x256x1280xf32>
    %5133 = stablehlo.add %5132, %cst_13 : tensor<1x256x1280xf32>
    %5134 = stablehlo.multiply %cst_14, %5121 : tensor<1x256x1280xf32>
    %5135 = stablehlo.add %5134, %cst_15 : tensor<1x256x1280xf32>
    %5136 = stablehlo.multiply %5135, %5121 : tensor<1x256x1280xf32>
    %5137 = stablehlo.add %5136, %cst_16 : tensor<1x256x1280xf32>
    %5138 = stablehlo.multiply %5137, %5121 : tensor<1x256x1280xf32>
    %5139 = stablehlo.add %5138, %cst_17 : tensor<1x256x1280xf32>
    %5140 = stablehlo.multiply %5139, %5121 : tensor<1x256x1280xf32>
    %5141 = stablehlo.add %5140, %cst_18 : tensor<1x256x1280xf32>
    %5142 = stablehlo.multiply %5120, %5133 : tensor<1x256x1280xf32>
    %5143 = stablehlo.divide %5142, %5141 : tensor<1x256x1280xf32>
    %5144 = stablehlo.clamp %cst_19, %5143, %cst_20 : tensor<1x256x1280xf32>
    %5145 = stablehlo.convert %5144 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %5146 = stablehlo.add %5145, %cst_2 : tensor<1x256x1280xbf16>
    %5147 = stablehlo.multiply %5146, %5117 : tensor<1x256x1280xbf16>
    %5148 = stablehlo.reshape %5147 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5149 = stablehlo.convert %5148 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5150 = stablehlo.dot_general %5149, %arg427, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5151 = stablehlo.broadcast_in_dim %5150, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5152 = stablehlo.multiply %5151, %127 : tensor<256x1280xf32>
    %5153 = stablehlo.broadcast_in_dim %5152, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5154 = stablehlo.broadcast_in_dim %arg428, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5155 = stablehlo.add %5153, %5154 : tensor<256x1280xf32>
    %5156 = stablehlo.convert %5155 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5157 = stablehlo.reshape %5156 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5158 = stablehlo.add %5157, %5069 : tensor<1x256x1280xbf16>
    %5159 = stablehlo.convert %5158 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %5160 = stablehlo.convert %5159 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %5161 = stablehlo.reduce(%5160 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5162 = stablehlo.reshape %5161 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5163 = stablehlo.broadcast_in_dim %5162, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5164 = stablehlo.divide %5163, %142 : tensor<1x256x1xf64>
    %5165 = stablehlo.broadcast_in_dim %5160, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %5166 = stablehlo.broadcast_in_dim %5164, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %5167 = stablehlo.subtract %5165, %5166 : tensor<1x256x1280xf64>
    %5168 = stablehlo.multiply %5167, %5167 : tensor<1x256x1280xf64>
    %5169 = stablehlo.reduce(%5168 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5170 = stablehlo.reshape %5169 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5171 = stablehlo.broadcast_in_dim %5170, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5172 = stablehlo.divide %5171, %142 : tensor<1x256x1xf64>
    %5173 = stablehlo.convert %5172 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %5174 = stablehlo.reduce(%5159 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %5175 = stablehlo.reshape %5174 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %5176 = stablehlo.broadcast_in_dim %5175, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5177 = stablehlo.divide %5176, %158 : tensor<1x256x1xf32>
    %5178 = stablehlo.broadcast_in_dim %5173, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5179 = stablehlo.add %5178, %161 : tensor<1x256x1xf32>
    %5180 = stablehlo.rsqrt %5179 : tensor<1x256x1xf32>
    %5181 = stablehlo.broadcast_in_dim %5159, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5182 = stablehlo.broadcast_in_dim %5177, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5183 = stablehlo.subtract %5181, %5182 : tensor<1x256x1280xf32>
    %5184 = stablehlo.broadcast_in_dim %5183, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5185 = stablehlo.broadcast_in_dim %5180, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5186 = stablehlo.multiply %5184, %5185 : tensor<1x256x1280xf32>
    %5187 = stablehlo.convert %arg107 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5188 = stablehlo.broadcast_in_dim %5186, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5189 = stablehlo.broadcast_in_dim %5187, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5190 = stablehlo.multiply %5188, %5189 : tensor<1x256x1280xf32>
    %5191 = stablehlo.convert %arg108 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5192 = stablehlo.broadcast_in_dim %5190, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5193 = stablehlo.broadcast_in_dim %5191, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5194 = stablehlo.add %5192, %5193 : tensor<1x256x1280xf32>
    %5195 = stablehlo.convert %5194 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %5196 = stablehlo.reshape %5195 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5197 = stablehlo.convert %5196 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5198 = stablehlo.dot_general %5197, %arg429, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %5199 = stablehlo.broadcast_in_dim %5198, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5200 = stablehlo.multiply %5199, %273 : tensor<256x256xf32>
    %5201 = stablehlo.broadcast_in_dim %5200, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5202 = stablehlo.broadcast_in_dim %arg430, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %5203 = stablehlo.add %5201, %5202 : tensor<256x256xf32>
    %5204 = stablehlo.convert %5203 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %5205 = stablehlo.reshape %5204 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %5206 = stablehlo.dot_general %5197, %arg431, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %5207 = stablehlo.broadcast_in_dim %5206, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5208 = stablehlo.multiply %5207, %273 : tensor<256x256xf32>
    %5209 = stablehlo.broadcast_in_dim %5208, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5210 = stablehlo.broadcast_in_dim %arg432, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %5211 = stablehlo.add %5209, %5210 : tensor<256x256xf32>
    %5212 = stablehlo.convert %5211 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %5213 = stablehlo.reshape %5212 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %5214 = stablehlo.dot_general %5197, %arg433, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5215 = stablehlo.broadcast_in_dim %5214, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5216 = stablehlo.multiply %5215, %127 : tensor<256x1280xf32>
    %5217 = stablehlo.broadcast_in_dim %5216, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5218 = stablehlo.broadcast_in_dim %arg434, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5219 = stablehlo.add %5217, %5218 : tensor<256x1280xf32>
    %5220 = stablehlo.convert %5219 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5221 = stablehlo.reshape %5220 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5222 = stablehlo.reshape %5205 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %5223 = stablehlo.transpose %5222, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %5224 = stablehlo.reshape %5213 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %5225 = stablehlo.transpose %5224, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %5226 = stablehlo.reshape %5221 : (tensor<1x256x1280xbf16>) -> tensor<1x256x8x160xbf16>
    %5227 = stablehlo.transpose %5226, dims = [0, 2, 1, 3] : (tensor<1x256x8x160xbf16>) -> tensor<1x8x256x160xbf16>
    %5228 = stablehlo.transpose %5225, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %5229 = stablehlo.reshape %5223 : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %5230 = stablehlo.reshape %5228 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %5231 = stablehlo.broadcast_in_dim %5230, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %5232 = stablehlo.dot_general %5229, %5231, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %5233 = stablehlo.reshape %5232 : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %5234 = stablehlo.broadcast_in_dim %5233, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %5235 = stablehlo.divide %5234, %309 : tensor<1x8x256x256xbf16>
    %5236 = stablehlo.convert %5235 : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xf32>
    %5237 = stablehlo.reduce(%5236 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %5238 = stablehlo.reshape %5237 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %5239 = stablehlo.broadcast_in_dim %5236, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %5240 = stablehlo.broadcast_in_dim %5238, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %5241 = stablehlo.subtract %5239, %5240 : tensor<1x8x256x256xf32>
    %5242 = stablehlo.exponential %5241 : tensor<1x8x256x256xf32>
    %5243 = stablehlo.reduce(%5242 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %5244 = stablehlo.reshape %5243 : (tensor<1x8x256xf32>) -> tensor<1x8x256x1xf32>
    %5245 = stablehlo.broadcast_in_dim %5242, dims = [0, 1, 2, 3] : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %5246 = stablehlo.broadcast_in_dim %5244, dims = [0, 1, 2, 3] : (tensor<1x8x256x1xf32>) -> tensor<1x8x256x256xf32>
    %5247 = stablehlo.divide %5245, %5246 : tensor<1x8x256x256xf32>
    %5248 = stablehlo.convert %5247 : (tensor<1x8x256x256xf32>) -> tensor<1x8x256x256xbf16>
    %5249 = stablehlo.reshape %5248 : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %5250 = stablehlo.reshape %5227 : (tensor<1x8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %5251 = stablehlo.broadcast_in_dim %5250, dims = [0, 1, 2] : (tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %5252 = stablehlo.dot_general %5249, %5251, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x256x256xbf16>, tensor<8x256x160xbf16>) -> tensor<8x256x160xbf16>
    %5253 = stablehlo.reshape %5252 : (tensor<8x256x160xbf16>) -> tensor<1x8x256x160xbf16>
    %5254 = stablehlo.transpose %5253, dims = [0, 2, 1, 3] : (tensor<1x8x256x160xbf16>) -> tensor<1x256x8x160xbf16>
    %5255 = stablehlo.reshape %5254 : (tensor<1x256x8x160xbf16>) -> tensor<1x256x1280xbf16>
    %5256 = stablehlo.reshape %5255 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5257 = stablehlo.convert %5256 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5258 = stablehlo.dot_general %5257, %arg435, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5259 = stablehlo.broadcast_in_dim %5258, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5260 = stablehlo.multiply %5259, %127 : tensor<256x1280xf32>
    %5261 = stablehlo.broadcast_in_dim %5260, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5262 = stablehlo.broadcast_in_dim %arg436, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5263 = stablehlo.add %5261, %5262 : tensor<256x1280xf32>
    %5264 = stablehlo.convert %5263 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5265 = stablehlo.reshape %5264 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5266 = stablehlo.add %5265, %5158 : tensor<1x256x1280xbf16>
    %5267 = stablehlo.convert %5266 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %5268 = stablehlo.convert %5267 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %5269 = stablehlo.reduce(%5268 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5270 = stablehlo.reshape %5269 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5271 = stablehlo.broadcast_in_dim %5270, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5272 = stablehlo.divide %5271, %142 : tensor<1x256x1xf64>
    %5273 = stablehlo.broadcast_in_dim %5268, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %5274 = stablehlo.broadcast_in_dim %5272, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %5275 = stablehlo.subtract %5273, %5274 : tensor<1x256x1280xf64>
    %5276 = stablehlo.multiply %5275, %5275 : tensor<1x256x1280xf64>
    %5277 = stablehlo.reduce(%5276 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5278 = stablehlo.reshape %5277 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5279 = stablehlo.broadcast_in_dim %5278, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5280 = stablehlo.divide %5279, %142 : tensor<1x256x1xf64>
    %5281 = stablehlo.convert %5280 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %5282 = stablehlo.reduce(%5267 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %5283 = stablehlo.reshape %5282 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %5284 = stablehlo.broadcast_in_dim %5283, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5285 = stablehlo.divide %5284, %158 : tensor<1x256x1xf32>
    %5286 = stablehlo.broadcast_in_dim %5281, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5287 = stablehlo.add %5286, %161 : tensor<1x256x1xf32>
    %5288 = stablehlo.rsqrt %5287 : tensor<1x256x1xf32>
    %5289 = stablehlo.broadcast_in_dim %5267, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5290 = stablehlo.broadcast_in_dim %5285, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5291 = stablehlo.subtract %5289, %5290 : tensor<1x256x1280xf32>
    %5292 = stablehlo.broadcast_in_dim %5291, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5293 = stablehlo.broadcast_in_dim %5288, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5294 = stablehlo.multiply %5292, %5293 : tensor<1x256x1280xf32>
    %5295 = stablehlo.convert %arg109 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5296 = stablehlo.broadcast_in_dim %5294, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5297 = stablehlo.broadcast_in_dim %5295, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5298 = stablehlo.multiply %5296, %5297 : tensor<1x256x1280xf32>
    %5299 = stablehlo.convert %arg110 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5300 = stablehlo.broadcast_in_dim %5298, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5301 = stablehlo.broadcast_in_dim %5299, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5302 = stablehlo.add %5300, %5301 : tensor<1x256x1280xf32>
    %5303 = stablehlo.convert %5302 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %5304 = stablehlo.reshape %5303 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5305 = stablehlo.convert %5304 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5306 = stablehlo.dot_general %5305, %arg437, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5307 = stablehlo.broadcast_in_dim %5306, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5308 = stablehlo.multiply %5307, %127 : tensor<256x1280xf32>
    %5309 = stablehlo.broadcast_in_dim %5308, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5310 = stablehlo.broadcast_in_dim %arg438, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5311 = stablehlo.add %5309, %5310 : tensor<256x1280xf32>
    %5312 = stablehlo.convert %5311 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5313 = stablehlo.reshape %5312 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5314 = stablehlo.multiply %5313, %cst_4 : tensor<1x256x1280xbf16>
    %5315 = stablehlo.multiply %5313, %190 : tensor<1x256x1280xbf16>
    %5316 = stablehlo.convert %5315 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %5317 = stablehlo.clamp %cst_5, %5316, %cst_6 : tensor<1x256x1280xf32>
    %5318 = stablehlo.multiply %5317, %5317 : tensor<1x256x1280xf32>
    %5319 = stablehlo.multiply %cst_7, %5318 : tensor<1x256x1280xf32>
    %5320 = stablehlo.add %5319, %cst_8 : tensor<1x256x1280xf32>
    %5321 = stablehlo.multiply %5320, %5318 : tensor<1x256x1280xf32>
    %5322 = stablehlo.add %5321, %cst_9 : tensor<1x256x1280xf32>
    %5323 = stablehlo.multiply %5322, %5318 : tensor<1x256x1280xf32>
    %5324 = stablehlo.add %5323, %cst_10 : tensor<1x256x1280xf32>
    %5325 = stablehlo.multiply %5324, %5318 : tensor<1x256x1280xf32>
    %5326 = stablehlo.add %5325, %cst_11 : tensor<1x256x1280xf32>
    %5327 = stablehlo.multiply %5326, %5318 : tensor<1x256x1280xf32>
    %5328 = stablehlo.add %5327, %cst_12 : tensor<1x256x1280xf32>
    %5329 = stablehlo.multiply %5328, %5318 : tensor<1x256x1280xf32>
    %5330 = stablehlo.add %5329, %cst_13 : tensor<1x256x1280xf32>
    %5331 = stablehlo.multiply %cst_14, %5318 : tensor<1x256x1280xf32>
    %5332 = stablehlo.add %5331, %cst_15 : tensor<1x256x1280xf32>
    %5333 = stablehlo.multiply %5332, %5318 : tensor<1x256x1280xf32>
    %5334 = stablehlo.add %5333, %cst_16 : tensor<1x256x1280xf32>
    %5335 = stablehlo.multiply %5334, %5318 : tensor<1x256x1280xf32>
    %5336 = stablehlo.add %5335, %cst_17 : tensor<1x256x1280xf32>
    %5337 = stablehlo.multiply %5336, %5318 : tensor<1x256x1280xf32>
    %5338 = stablehlo.add %5337, %cst_18 : tensor<1x256x1280xf32>
    %5339 = stablehlo.multiply %5317, %5330 : tensor<1x256x1280xf32>
    %5340 = stablehlo.divide %5339, %5338 : tensor<1x256x1280xf32>
    %5341 = stablehlo.clamp %cst_19, %5340, %cst_20 : tensor<1x256x1280xf32>
    %5342 = stablehlo.convert %5341 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %5343 = stablehlo.add %5342, %cst_2 : tensor<1x256x1280xbf16>
    %5344 = stablehlo.multiply %5343, %5314 : tensor<1x256x1280xbf16>
    %5345 = stablehlo.reshape %5344 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5346 = stablehlo.convert %5345 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5347 = stablehlo.dot_general %5346, %arg439, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x1280xf32>) -> tensor<256x1280xf32>
    %5348 = stablehlo.broadcast_in_dim %5347, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5349 = stablehlo.multiply %5348, %127 : tensor<256x1280xf32>
    %5350 = stablehlo.broadcast_in_dim %5349, dims = [0, 1] : (tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %5351 = stablehlo.broadcast_in_dim %arg440, dims = [1] : (tensor<1280xf32>) -> tensor<256x1280xf32>
    %5352 = stablehlo.add %5350, %5351 : tensor<256x1280xf32>
    %5353 = stablehlo.convert %5352 : (tensor<256x1280xf32>) -> tensor<256x1280xbf16>
    %5354 = stablehlo.reshape %5353 : (tensor<256x1280xbf16>) -> tensor<1x256x1280xbf16>
    %5355 = stablehlo.add %5354, %5266 : tensor<1x256x1280xbf16>
    %5356 = stablehlo.convert %5355 : (tensor<1x256x1280xbf16>) -> tensor<1x256x1280xf32>
    %5357 = stablehlo.convert %5356 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf64>
    %5358 = stablehlo.reduce(%5357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5359 = stablehlo.reshape %5358 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5360 = stablehlo.broadcast_in_dim %5359, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5361 = stablehlo.divide %5360, %142 : tensor<1x256x1xf64>
    %5362 = stablehlo.broadcast_in_dim %5357, dims = [0, 1, 2] : (tensor<1x256x1280xf64>) -> tensor<1x256x1280xf64>
    %5363 = stablehlo.broadcast_in_dim %5361, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1280xf64>
    %5364 = stablehlo.subtract %5362, %5363 : tensor<1x256x1280xf64>
    %5365 = stablehlo.multiply %5364, %5364 : tensor<1x256x1280xf64>
    %5366 = stablehlo.reduce(%5365 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf64>, tensor<f64>) -> tensor<1x256xf64>
    %5367 = stablehlo.reshape %5366 : (tensor<1x256xf64>) -> tensor<1x256x1xf64>
    %5368 = stablehlo.broadcast_in_dim %5367, dims = [0, 1, 2] : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
    %5369 = stablehlo.divide %5368, %142 : tensor<1x256x1xf64>
    %5370 = stablehlo.convert %5369 : (tensor<1x256x1xf64>) -> tensor<1x256x1xf32>
    %5371 = stablehlo.reduce(%5356 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x256x1280xf32>, tensor<f32>) -> tensor<1x256xf32>
    %5372 = stablehlo.reshape %5371 : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %5373 = stablehlo.broadcast_in_dim %5372, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5374 = stablehlo.divide %5373, %158 : tensor<1x256x1xf32>
    %5375 = stablehlo.broadcast_in_dim %5370, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %5376 = stablehlo.add %5375, %161 : tensor<1x256x1xf32>
    %5377 = stablehlo.rsqrt %5376 : tensor<1x256x1xf32>
    %5378 = stablehlo.broadcast_in_dim %5356, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5379 = stablehlo.broadcast_in_dim %5374, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5380 = stablehlo.subtract %5378, %5379 : tensor<1x256x1280xf32>
    %5381 = stablehlo.broadcast_in_dim %5380, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5382 = stablehlo.broadcast_in_dim %5377, dims = [0, 1, 2] : (tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %5383 = stablehlo.multiply %5381, %5382 : tensor<1x256x1280xf32>
    %5384 = stablehlo.convert %arg111 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5385 = stablehlo.broadcast_in_dim %5383, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5386 = stablehlo.broadcast_in_dim %5384, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5387 = stablehlo.multiply %5385, %5386 : tensor<1x256x1280xf32>
    %5388 = stablehlo.convert %arg112 : (tensor<1280xbf16>) -> tensor<1280xf32>
    %5389 = stablehlo.broadcast_in_dim %5387, dims = [0, 1, 2] : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %5390 = stablehlo.broadcast_in_dim %5388, dims = [2] : (tensor<1280xf32>) -> tensor<1x256x1280xf32>
    %5391 = stablehlo.add %5389, %5390 : tensor<1x256x1280xf32>
    %5392 = stablehlo.convert %5391 : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xbf16>
    %5393 = stablehlo.reshape %5392 : (tensor<1x256x1280xbf16>) -> tensor<256x1280xbf16>
    %5394 = stablehlo.convert %5393 : (tensor<256x1280xbf16>) -> tensor<256x1280xf32>
    %5395 = stablehlo.dot_general %5394, %arg441, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x256xf32>) -> tensor<256x256xf32>
    %5396 = stablehlo.broadcast_in_dim %5395, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5397 = stablehlo.multiply %5396, %273 : tensor<256x256xf32>
    %5398 = stablehlo.broadcast_in_dim %5397, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5399 = stablehlo.broadcast_in_dim %arg442, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %5400 = stablehlo.add %5398, %5399 : tensor<256x256xf32>
    %5401 = stablehlo.convert %5400 : (tensor<256x256xf32>) -> tensor<256x256xbf16>
    %5402 = stablehlo.reshape %5401 : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %5403 = stablehlo.dot_general %5394, %arg443, contracting_dims = [1] x [0] : (tensor<256x1280xf32>, tensor<1280x768xf32>) -> tensor<256x768xf32>
    %5404 = stablehlo.broadcast_in_dim %5403, dims = [0, 1] : (tensor<256x768xf32>) -> tensor<256x768xf32>
    %5405 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<256x768xf32>
    %5406 = stablehlo.multiply %5404, %5405 : tensor<256x768xf32>
    %5407 = stablehlo.broadcast_in_dim %5406, dims = [0, 1] : (tensor<256x768xf32>) -> tensor<256x768xf32>
    %5408 = stablehlo.broadcast_in_dim %arg444, dims = [1] : (tensor<768xf32>) -> tensor<256x768xf32>
    %5409 = stablehlo.add %5407, %5408 : tensor<256x768xf32>
    %5410 = stablehlo.convert %5409 : (tensor<256x768xf32>) -> tensor<256x768xbf16>
    %5411 = stablehlo.reshape %5410 : (tensor<256x768xbf16>) -> tensor<1x256x768xbf16>
    %5412 = stablehlo.reshape %5402 : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %5413 = stablehlo.transpose %5412, dims = [0, 2, 1, 3] : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %5414 = stablehlo.reshape %5411 : (tensor<1x256x768xbf16>) -> tensor<1x256x8x96xbf16>
    %5415 = stablehlo.transpose %5414, dims = [0, 2, 1, 3] : (tensor<1x256x8x96xbf16>) -> tensor<1x8x256x96xbf16>
    %5416 = stablehlo.transpose %5413, dims = [0, 1, 3, 2] : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %5417 = stablehlo.reshape %5416 : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %5418 = stablehlo.broadcast_in_dim %5417, dims = [0, 1, 2] : (tensor<8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %5419 = stablehlo.dot_general %arg445, %5418, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x2048x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x2048x256xbf16>
    %5420 = stablehlo.reshape %5419 : (tensor<8x2048x256xbf16>) -> tensor<1x8x2048x256xbf16>
    %5421 = stablehlo.broadcast_in_dim %5420, dims = [0, 1, 2, 3] : (tensor<1x8x2048x256xbf16>) -> tensor<1x8x2048x256xbf16>
    %5422 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<bf16>) -> tensor<1x8x2048x256xbf16>
    %5423 = stablehlo.divide %5421, %5422 : tensor<1x8x2048x256xbf16>
    %5424 = stablehlo.convert %5423 : (tensor<1x8x2048x256xbf16>) -> tensor<1x8x2048x256xf32>
    %5425 = stablehlo.reduce(%5424 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x2048x256xf32>, tensor<f32>) -> tensor<1x8x2048xf32>
    %5426 = stablehlo.reshape %5425 : (tensor<1x8x2048xf32>) -> tensor<1x8x2048x1xf32>
    %5427 = stablehlo.broadcast_in_dim %5424, dims = [0, 1, 2, 3] : (tensor<1x8x2048x256xf32>) -> tensor<1x8x2048x256xf32>
    %5428 = stablehlo.broadcast_in_dim %5426, dims = [0, 1, 2, 3] : (tensor<1x8x2048x1xf32>) -> tensor<1x8x2048x256xf32>
    %5429 = stablehlo.subtract %5427, %5428 : tensor<1x8x2048x256xf32>
    %5430 = stablehlo.exponential %5429 : tensor<1x8x2048x256xf32>
    %5431 = stablehlo.reduce(%5430 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x8x2048x256xf32>, tensor<f32>) -> tensor<1x8x2048xf32>
    %5432 = stablehlo.reshape %5431 : (tensor<1x8x2048xf32>) -> tensor<1x8x2048x1xf32>
    %5433 = stablehlo.broadcast_in_dim %5430, dims = [0, 1, 2, 3] : (tensor<1x8x2048x256xf32>) -> tensor<1x8x2048x256xf32>
    %5434 = stablehlo.broadcast_in_dim %5432, dims = [0, 1, 2, 3] : (tensor<1x8x2048x1xf32>) -> tensor<1x8x2048x256xf32>
    %5435 = stablehlo.divide %5433, %5434 : tensor<1x8x2048x256xf32>
    %5436 = stablehlo.convert %5435 : (tensor<1x8x2048x256xf32>) -> tensor<1x8x2048x256xbf16>
    %5437 = stablehlo.reshape %5436 : (tensor<1x8x2048x256xbf16>) -> tensor<8x2048x256xbf16>
    %5438 = stablehlo.reshape %5415 : (tensor<1x8x256x96xbf16>) -> tensor<8x256x96xbf16>
    %5439 = stablehlo.broadcast_in_dim %5438, dims = [0, 1, 2] : (tensor<8x256x96xbf16>) -> tensor<8x256x96xbf16>
    %5440 = stablehlo.dot_general %5437, %5439, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x2048x256xbf16>, tensor<8x256x96xbf16>) -> tensor<8x2048x96xbf16>
    %5441 = stablehlo.reshape %5440 : (tensor<8x2048x96xbf16>) -> tensor<1x8x2048x96xbf16>
    %5442 = stablehlo.transpose %5441, dims = [0, 2, 1, 3] : (tensor<1x8x2048x96xbf16>) -> tensor<1x2048x8x96xbf16>
    %5443 = stablehlo.reshape %5442 : (tensor<1x2048x8x96xbf16>) -> tensor<1x2048x768xbf16>
    %5444 = stablehlo.reshape %5443 : (tensor<1x2048x768xbf16>) -> tensor<2048x768xbf16>
    %5445 = stablehlo.convert %5444 : (tensor<2048x768xbf16>) -> tensor<2048x768xf32>
    %5446 = stablehlo.dot_general %5445, %arg446, contracting_dims = [1] x [0] : (tensor<2048x768xf32>, tensor<768x768xf32>) -> tensor<2048x768xf32>
    %5447 = stablehlo.broadcast_in_dim %5446, dims = [0, 1] : (tensor<2048x768xf32>) -> tensor<2048x768xf32>
    %5448 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<2048x768xf32>
    %5449 = stablehlo.multiply %5447, %5448 : tensor<2048x768xf32>
    %5450 = stablehlo.broadcast_in_dim %5449, dims = [0, 1] : (tensor<2048x768xf32>) -> tensor<2048x768xf32>
    %5451 = stablehlo.broadcast_in_dim %arg447, dims = [1] : (tensor<768xf32>) -> tensor<2048x768xf32>
    %5452 = stablehlo.add %5450, %5451 : tensor<2048x768xf32>
    %5453 = stablehlo.convert %5452 : (tensor<2048x768xf32>) -> tensor<2048x768xbf16>
    %5454 = stablehlo.reshape %5453 : (tensor<2048x768xbf16>) -> tensor<1x2048x768xbf16>
    %5455 = stablehlo.convert %5454 : (tensor<1x2048x768xbf16>) -> tensor<1x2048x768xf32>
    %5456 = stablehlo.convert %5455 : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf64>
    %5457 = stablehlo.reduce(%5456 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x2048x768xf64>, tensor<f64>) -> tensor<1x2048xf64>
    %5458 = stablehlo.reshape %5457 : (tensor<1x2048xf64>) -> tensor<1x2048x1xf64>
    %5459 = stablehlo.broadcast_in_dim %5458, dims = [0, 1, 2] : (tensor<1x2048x1xf64>) -> tensor<1x2048x1xf64>
    %5460 = stablehlo.divide %5459, %25 : tensor<1x2048x1xf64>
    %5461 = stablehlo.broadcast_in_dim %5456, dims = [0, 1, 2] : (tensor<1x2048x768xf64>) -> tensor<1x2048x768xf64>
    %5462 = stablehlo.broadcast_in_dim %5460, dims = [0, 1, 2] : (tensor<1x2048x1xf64>) -> tensor<1x2048x768xf64>
    %5463 = stablehlo.subtract %5461, %5462 : tensor<1x2048x768xf64>
    %5464 = stablehlo.multiply %5463, %5463 : tensor<1x2048x768xf64>
    %5465 = stablehlo.reduce(%5464 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x2048x768xf64>, tensor<f64>) -> tensor<1x2048xf64>
    %5466 = stablehlo.reshape %5465 : (tensor<1x2048xf64>) -> tensor<1x2048x1xf64>
    %5467 = stablehlo.broadcast_in_dim %5466, dims = [0, 1, 2] : (tensor<1x2048x1xf64>) -> tensor<1x2048x1xf64>
    %5468 = stablehlo.divide %5467, %25 : tensor<1x2048x1xf64>
    %5469 = stablehlo.convert %5468 : (tensor<1x2048x1xf64>) -> tensor<1x2048x1xf32>
    %5470 = stablehlo.reduce(%5455 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<1x2048x768xf32>, tensor<f32>) -> tensor<1x2048xf32>
    %5471 = stablehlo.reshape %5470 : (tensor<1x2048xf32>) -> tensor<1x2048x1xf32>
    %5472 = stablehlo.broadcast_in_dim %5471, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x1xf32>
    %5473 = stablehlo.divide %5472, %41 : tensor<1x2048x1xf32>
    %5474 = stablehlo.broadcast_in_dim %5469, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x1xf32>
    %5475 = stablehlo.add %5474, %46 : tensor<1x2048x1xf32>
    %5476 = stablehlo.rsqrt %5475 : tensor<1x2048x1xf32>
    %5477 = stablehlo.broadcast_in_dim %5455, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %5478 = stablehlo.broadcast_in_dim %5473, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x768xf32>
    %5479 = stablehlo.subtract %5477, %5478 : tensor<1x2048x768xf32>
    %5480 = stablehlo.broadcast_in_dim %5479, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %5481 = stablehlo.broadcast_in_dim %5476, dims = [0, 1, 2] : (tensor<1x2048x1xf32>) -> tensor<1x2048x768xf32>
    %5482 = stablehlo.multiply %5480, %5481 : tensor<1x2048x768xf32>
    %5483 = stablehlo.convert %arg113 : (tensor<768xbf16>) -> tensor<768xf32>
    %5484 = stablehlo.broadcast_in_dim %5482, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %5485 = stablehlo.broadcast_in_dim %5483, dims = [2] : (tensor<768xf32>) -> tensor<1x2048x768xf32>
    %5486 = stablehlo.multiply %5484, %5485 : tensor<1x2048x768xf32>
    %5487 = stablehlo.convert %arg114 : (tensor<768xbf16>) -> tensor<768xf32>
    %5488 = stablehlo.broadcast_in_dim %5486, dims = [0, 1, 2] : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xf32>
    %5489 = stablehlo.broadcast_in_dim %5487, dims = [2] : (tensor<768xf32>) -> tensor<1x2048x768xf32>
    %5490 = stablehlo.add %5488, %5489 : tensor<1x2048x768xf32>
    %5491 = stablehlo.convert %5490 : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xbf16>
    %5492 = stablehlo.reshape %5491 : (tensor<1x2048x768xbf16>) -> tensor<2048x768xbf16>
    %5493 = stablehlo.convert %5492 : (tensor<2048x768xbf16>) -> tensor<2048x768xf32>
    %5494 = stablehlo.dot_general %5493, %arg448, contracting_dims = [1] x [0] : (tensor<2048x768xf32>, tensor<768x768xf32>) -> tensor<2048x768xf32>
    %5495 = stablehlo.broadcast_in_dim %5494, dims = [0, 1] : (tensor<2048x768xf32>) -> tensor<2048x768xf32>
    %5496 = stablehlo.multiply %5495, %5448 : tensor<2048x768xf32>
    %5497 = stablehlo.broadcast_in_dim %5496, dims = [0, 1] : (tensor<2048x768xf32>) -> tensor<2048x768xf32>
    %5498 = stablehlo.broadcast_in_dim %arg449, dims = [1] : (tensor<768xf32>) -> tensor<2048x768xf32>
    %5499 = stablehlo.add %5497, %5498 : tensor<2048x768xf32>
    %5500 = stablehlo.convert %5499 : (tensor<2048x768xf32>) -> tensor<2048x768xbf16>
    %5501 = stablehlo.reshape %5500 : (tensor<2048x768xbf16>) -> tensor<1x2048x768xbf16>
    %5502 = stablehlo.multiply %5501, %cst_23 : tensor<1x2048x768xbf16>
    %5503 = stablehlo.rsqrt %cst_22 : tensor<1x2048x768xbf16>
    %5504 = stablehlo.multiply %5501, %5503 : tensor<1x2048x768xbf16>
    %5505 = stablehlo.convert %5504 : (tensor<1x2048x768xbf16>) -> tensor<1x2048x768xf32>
    %5506 = stablehlo.clamp %cst_24, %5505, %cst_25 : tensor<1x2048x768xf32>
    %5507 = stablehlo.multiply %5506, %5506 : tensor<1x2048x768xf32>
    %5508 = stablehlo.multiply %cst_26, %5507 : tensor<1x2048x768xf32>
    %5509 = stablehlo.add %5508, %cst_27 : tensor<1x2048x768xf32>
    %5510 = stablehlo.multiply %5509, %5507 : tensor<1x2048x768xf32>
    %5511 = stablehlo.add %5510, %cst_28 : tensor<1x2048x768xf32>
    %5512 = stablehlo.multiply %5511, %5507 : tensor<1x2048x768xf32>
    %5513 = stablehlo.add %5512, %cst_29 : tensor<1x2048x768xf32>
    %5514 = stablehlo.multiply %5513, %5507 : tensor<1x2048x768xf32>
    %5515 = stablehlo.add %5514, %cst_30 : tensor<1x2048x768xf32>
    %5516 = stablehlo.multiply %5515, %5507 : tensor<1x2048x768xf32>
    %5517 = stablehlo.add %5516, %cst_31 : tensor<1x2048x768xf32>
    %5518 = stablehlo.multiply %5517, %5507 : tensor<1x2048x768xf32>
    %5519 = stablehlo.add %5518, %cst_32 : tensor<1x2048x768xf32>
    %5520 = stablehlo.multiply %cst_33, %5507 : tensor<1x2048x768xf32>
    %5521 = stablehlo.add %5520, %cst_34 : tensor<1x2048x768xf32>
    %5522 = stablehlo.multiply %5521, %5507 : tensor<1x2048x768xf32>
    %5523 = stablehlo.add %5522, %cst_35 : tensor<1x2048x768xf32>
    %5524 = stablehlo.multiply %5523, %5507 : tensor<1x2048x768xf32>
    %5525 = stablehlo.add %5524, %cst_36 : tensor<1x2048x768xf32>
    %5526 = stablehlo.multiply %5525, %5507 : tensor<1x2048x768xf32>
    %5527 = stablehlo.add %5526, %cst_37 : tensor<1x2048x768xf32>
    %5528 = stablehlo.multiply %5506, %5519 : tensor<1x2048x768xf32>
    %5529 = stablehlo.divide %5528, %5527 : tensor<1x2048x768xf32>
    %5530 = stablehlo.clamp %cst_38, %5529, %cst_39 : tensor<1x2048x768xf32>
    %5531 = stablehlo.convert %5530 : (tensor<1x2048x768xf32>) -> tensor<1x2048x768xbf16>
    %5532 = stablehlo.add %5531, %cst_21 : tensor<1x2048x768xbf16>
    %5533 = stablehlo.multiply %5532, %5502 : tensor<1x2048x768xbf16>
    %5534 = stablehlo.reshape %5533 : (tensor<1x2048x768xbf16>) -> tensor<2048x768xbf16>
    %5535 = stablehlo.convert %5534 : (tensor<2048x768xbf16>) -> tensor<2048x768xf32>
    %5536 = stablehlo.dot_general %5535, %arg450, contracting_dims = [1] x [0] : (tensor<2048x768xf32>, tensor<768x768xf32>) -> tensor<2048x768xf32>
    %5537 = stablehlo.broadcast_in_dim %5536, dims = [0, 1] : (tensor<2048x768xf32>) -> tensor<2048x768xf32>
    %5538 = stablehlo.multiply %5537, %5448 : tensor<2048x768xf32>
    %5539 = stablehlo.broadcast_in_dim %5538, dims = [0, 1] : (tensor<2048x768xf32>) -> tensor<2048x768xf32>
    %5540 = stablehlo.broadcast_in_dim %arg451, dims = [1] : (tensor<768xf32>) -> tensor<2048x768xf32>
    %5541 = stablehlo.add %5539, %5540 : tensor<2048x768xf32>
    %5542 = stablehlo.convert %5541 : (tensor<2048x768xf32>) -> tensor<2048x768xbf16>
    %5543 = stablehlo.reshape %5542 : (tensor<2048x768xbf16>) -> tensor<1x2048x768xbf16>
    %5544 = stablehlo.add %5543, %5454 : tensor<1x2048x768xbf16>
    %5545 = stablehlo.reshape %5544 : (tensor<1x2048x768xbf16>) -> tensor<2048x768xbf16>
    %5546 = stablehlo.dot_general %5545, %arg452, contracting_dims = [1] x [0] : (tensor<2048x768xbf16>, tensor<768x262xbf16>) -> tensor<2048x262xbf16>
    %5547 = stablehlo.broadcast_in_dim %5546, dims = [0, 1] : (tensor<2048x262xbf16>) -> tensor<2048x262xbf16>
    %5548 = stablehlo.broadcast_in_dim %arg115, dims = [1] : (tensor<262xbf16>) -> tensor<2048x262xbf16>
    %5549 = stablehlo.add %5547, %5548 : tensor<2048x262xbf16>
    %5550 = stablehlo.reshape %5549 : (tensor<2048x262xbf16>) -> tensor<1x2048x262xbf16>
    return %5550 : tensor<1x2048x262xbf16>
  }
}
