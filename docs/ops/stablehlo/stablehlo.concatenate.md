
### stablehlo.concatenate::ttnn.concat


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,32,64]>,<br>Tensor<[1,32,64]>,<br>dim: 2<br>|ttnn.concat|aten::cat|5|
|1|Tensor<[1,32,32,64]>,<br>Tensor<[1,32,32,64]>,<br>dim: 3<br>|ttnn.concat|aten::cat|5|
|2|Tensor<[1,1]>,<br>Tensor<[1,1]>,<br>dim: 1<br>|ttnn.concat|aten::index.Tensor|4|
|3|Tensor<[1,128,28,28]>,<br>Tensor<[1,19,28,28]>,<br>Tensor<[1,38,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|4|Tensor<[1,23,40,128]>,<br>Tensor<[1,23,40,128]>,<br>dim: 3<br>|ttnn.concat|aten::cat|5|
|5|Tensor<[1,1,23,40,1]>,<br>Tensor<[1,1,23,40,1]>,<br>Tensor<[1,1,23,40,1]>,<br>Tensor<[1,1,23,40,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|6|Tensor<[1,23,40,64,1]>,<br>Tensor<[1,23,40,64,1]>,<br>dim: 4<br>|ttnn.concat|aten::stack|5|
|7|Tensor<[1,100,1,256]>,<br>Tensor<[1,100,1,256]>,<br>Tensor<[1,100,1,256]>,<br>Tensor<[1,100,1,256]>,<br>Tensor<[1,100,1,256]>,<br>Tensor<[1,100,1,256]>,<br>dim: 0<br>|ttnn.concat|aten::stack|5|
|8|Tensor<[1,160]>,<br>Tensor<[1,160]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|9|Tensor<[1,1280,8,8]>,<br>Tensor<[1,1280,8,8]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|10|Tensor<[1,1280,16,16]>,<br>Tensor<[1,1280,16,16]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|11|Tensor<[1,1280,16,16]>,<br>Tensor<[1,640,16,16]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|12|Tensor<[1,1280,32,32]>,<br>Tensor<[1,640,32,32]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|13|Tensor<[1,640,32,32]>,<br>Tensor<[1,640,32,32]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|14|Tensor<[1,640,32,32]>,<br>Tensor<[1,320,32,32]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|15|Tensor<[1,640,64,64]>,<br>Tensor<[1,320,64,64]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|16|Tensor<[1,320,64,64]>,<br>Tensor<[1,320,64,64]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|17|Tensor<[1,1280,16,16,1]>,<br>Tensor<[1,1280,16,16,1]>,<br>Tensor<[1,1280,16,16,1]>,<br>Tensor<[1,1280,16,16,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|18|Tensor<[1,1280,32,32,1]>,<br>Tensor<[1,1280,32,32,1]>,<br>Tensor<[1,1280,32,32,1]>,<br>Tensor<[1,1280,32,32,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|19|Tensor<[1,640,64,64,1]>,<br>Tensor<[1,640,64,64,1]>,<br>Tensor<[1,640,64,64,1]>,<br>Tensor<[1,640,64,64,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|20|Tensor<[1,1,192]>,<br>Tensor<[1,1344,192]>,<br>Tensor<[1,100,192]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|21|Tensor<[1,8,768]>,<br>Tensor<[1,193,768]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|22|Tensor<[1,8]>,<br>Tensor<[1,193]>,<br>dim: 1<br>|ttnn.concat|aten::cat|4|
|23|Tensor<[1,1,12,16,1]>,<br>Tensor<[1,1,12,16,1]>,<br>Tensor<[1,1,12,16,1]>,<br>Tensor<[1,1,12,16,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|24|Tensor<[12,16,1]>,<br>Tensor<[12,16,1]>,<br>dim: 2<br>|ttnn.concat|aten::stack|4|
|25|Tensor<[19,1,1]>,<br>Tensor<[19,1,1]>,<br>dim: 2<br>|ttnn.concat|aten::gather|4|
|26|Tensor<[1,14,56,56]>,<br>Tensor<[1,64,56,56]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|27|Tensor<[1,14,56,56]>,<br>Tensor<[1,24,56,56]>,<br>Tensor<[1,64,56,56]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|28|Tensor<[1,14,56,56]>,<br>Tensor<[1,40,56,56]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|29|Tensor<[1,14,56,56]>,<br>Tensor<[1,24,56,56]>,<br>Tensor<[1,40,56,56]>,<br>Tensor<[1,64,56,56]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|30|Tensor<[1,14,56,56]>,<br>Tensor<[1,14,56,56]>,<br>Tensor<[1,14,56,56]>,<br>Tensor<[1,14,56,56]>,<br>Tensor<[1,68,56,56]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|31|Tensor<[1,16,28,28]>,<br>Tensor<[1,128,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|32|Tensor<[1,16,28,28]>,<br>Tensor<[1,28,28,28]>,<br>Tensor<[1,128,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|33|Tensor<[1,16,28,28]>,<br>Tensor<[1,46,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|34|Tensor<[1,16,28,28]>,<br>Tensor<[1,28,28,28]>,<br>Tensor<[1,46,28,28]>,<br>Tensor<[1,128,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|35|Tensor<[1,16,28,28]>,<br>Tensor<[1,78,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|36|Tensor<[1,16,28,28]>,<br>Tensor<[1,28,28,28]>,<br>Tensor<[1,78,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|37|Tensor<[1,16,28,28]>,<br>Tensor<[1,28,28,28]>,<br>Tensor<[1,46,28,28]>,<br>Tensor<[1,78,28,28]>,<br>Tensor<[1,128,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|38|Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>Tensor<[1,134,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|39|Tensor<[1,20,28,28]>,<br>Tensor<[1,256,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|40|Tensor<[1,20,28,28]>,<br>Tensor<[1,34,28,28]>,<br>Tensor<[1,256,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|41|Tensor<[1,20,28,28]>,<br>Tensor<[1,58,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|42|Tensor<[1,20,28,28]>,<br>Tensor<[1,34,28,28]>,<br>Tensor<[1,58,28,28]>,<br>Tensor<[1,256,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|43|Tensor<[1,20,28,28]>,<br>Tensor<[1,98,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|44|Tensor<[1,20,28,28]>,<br>Tensor<[1,34,28,28]>,<br>Tensor<[1,98,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|45|Tensor<[1,20,28,28]>,<br>Tensor<[1,34,28,28]>,<br>Tensor<[1,58,28,28]>,<br>Tensor<[1,98,28,28]>,<br>Tensor<[1,256,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|46|Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>Tensor<[1,168,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|47|Tensor<[1,40,14,14]>,<br>Tensor<[1,320,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|48|Tensor<[1,40,14,14]>,<br>Tensor<[1,68,14,14]>,<br>Tensor<[1,320,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|49|Tensor<[1,40,14,14]>,<br>Tensor<[1,116,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|50|Tensor<[1,40,14,14]>,<br>Tensor<[1,68,14,14]>,<br>Tensor<[1,116,14,14]>,<br>Tensor<[1,320,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|51|Tensor<[1,40,14,14]>,<br>Tensor<[1,196,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|52|Tensor<[1,40,14,14]>,<br>Tensor<[1,68,14,14]>,<br>Tensor<[1,196,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|53|Tensor<[1,40,14,14]>,<br>Tensor<[1,68,14,14]>,<br>Tensor<[1,116,14,14]>,<br>Tensor<[1,196,14,14]>,<br>Tensor<[1,320,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|54|Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>Tensor<[1,334,14,14]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|55|Tensor<[1,160,7,7]>,<br>Tensor<[1,640,7,7]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|56|Tensor<[1,160,7,7]>,<br>Tensor<[1,272,7,7]>,<br>Tensor<[1,640,7,7]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|57|Tensor<[1,160,7,7]>,<br>Tensor<[1,160,7,7]>,<br>Tensor<[1,462,7,7]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|58|Tensor<[1,256,32,32]>,<br>Tensor<[1,512,32,32]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|59|Tensor<[1,128,64,64]>,<br>Tensor<[1,256,64,64]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|60|Tensor<[1,256,32,32,1]>,<br>Tensor<[1,256,32,32,1]>,<br>Tensor<[1,256,32,32,1]>,<br>Tensor<[1,256,32,32,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|61|Tensor<[1,128,64,64,1]>,<br>Tensor<[1,128,64,64,1]>,<br>Tensor<[1,128,64,64,1]>,<br>Tensor<[1,128,64,64,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|62|Tensor<[1,256,32,32]>,<br>Tensor<[1,256,32,32]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|63|Tensor<[1,128,64,64]>,<br>Tensor<[1,128,64,64]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|64|Tensor<[1,64,128,128]>,<br>Tensor<[1,64,128,128]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|65|Tensor<[1,32,256,256]>,<br>Tensor<[1,32,256,256]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|66|Tensor<[1,512,28,28]>,<br>Tensor<[1,512,28,28]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|67|Tensor<[1,256,56,56]>,<br>Tensor<[1,256,56,56]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|68|Tensor<[1,128,112,112]>,<br>Tensor<[1,128,112,112]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|69|Tensor<[1,64,224,224]>,<br>Tensor<[1,64,224,224]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|70|Tensor<[1,64,30,40]>,<br>Tensor<[1,64,30,40]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|71|Tensor<[1,64,60,80]>,<br>Tensor<[1,64,60,80]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|72|Tensor<[1,64,120,160]>,<br>Tensor<[1,64,120,160]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|73|Tensor<[1,64,30,40,1]>,<br>Tensor<[1,64,30,40,1]>,<br>Tensor<[1,64,30,40,1]>,<br>Tensor<[1,64,30,40,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|74|Tensor<[1,64,60,80,1]>,<br>Tensor<[1,64,60,80,1]>,<br>Tensor<[1,64,60,80,1]>,<br>Tensor<[1,64,60,80,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|75|Tensor<[1,64,120,160,1]>,<br>Tensor<[1,64,120,160,1]>,<br>Tensor<[1,64,120,160,1]>,<br>Tensor<[1,64,120,160,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|76|Tensor<[1,64,240,320,1]>,<br>Tensor<[1,64,240,320,1]>,<br>Tensor<[1,64,240,320,1]>,<br>Tensor<[1,64,240,320,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|77|Tensor<[1,64,480,640,1]>,<br>Tensor<[1,64,480,640,1]>,<br>Tensor<[1,64,480,640,1]>,<br>Tensor<[1,64,480,640,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|78|Tensor<[1,1,768]>,<br>Tensor<[1,196,768]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|79|Tensor<[1,256,128,128]>,<br>Tensor<[1,256,128,128]>,<br>Tensor<[1,256,128,128]>,<br>Tensor<[1,256,128,128]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|80|Tensor<[1,256,128,128,1]>,<br>Tensor<[1,256,128,128,1]>,<br>Tensor<[1,256,128,128,1]>,<br>Tensor<[1,256,128,128,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|81|Tensor<[1,7,32]>,<br>Tensor<[1,7,32]>,<br>dim: 2<br>|ttnn.concat|aten::cat|5|
|82|Tensor<[1,71,7,32]>,<br>Tensor<[1,71,7,32]>,<br>dim: 3<br>|ttnn.concat|aten::cat|5|
|83|Tensor<[1,1,7,32]>,<br>Tensor<[1,1,7,32]>,<br>dim: 3<br>|ttnn.concat|aten::cat|5|
|84|Tensor<[1,7,1,64,1]>,<br>Tensor<[1,7,1,64,1]>,<br>Tensor<[1,7,1,64,1]>,<br>Tensor<[1,7,1,64,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|85|Tensor<[1,1,768]>,<br>Tensor<[1,49,768]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|86|Tensor<[2,1]>,<br>Tensor<[2,1]>,<br>dim: 1<br>|ttnn.concat|aten::index.Tensor|4|
|87|Tensor<[1,1,1024]>,<br>Tensor<[1,196,1024]>,<br>dim: 1<br>|ttnn.concat|aten::cat|5|
|88|Tensor<[729,16]>,<br>Tensor<[3,16]>,<br>dim: 0<br>|ttnn.concat|aten::cat|5|
|89|Tensor<[1,16,27,27,1]>,<br>Tensor<[1,16,27,27,1]>,<br>Tensor<[1,16,27,27,1]>,<br>Tensor<[1,16,27,27,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|
|90|Tensor<[1,14,14]>,<br>Tensor<[1,14,14]>,<br>dim: 0<br>|ttnn.concat|aten::stack|4|
|91|Tensor<[729,12]>,<br>Tensor<[3,12]>,<br>dim: 0<br>|ttnn.concat|aten::cat|5|
|92|Tensor<[1,12,27,27,1]>,<br>Tensor<[1,12,27,27,1]>,<br>Tensor<[1,12,27,27,1]>,<br>Tensor<[1,12,27,27,1]>,<br>dim: 4<br>|ttnn.concat|aten::index.Tensor|4|