
### stablehlo.slice::ttnn.slice


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,32,32,128]>,<br>indices: [0:1, 0:32, 0:32, 0:64]<br>|ttnn.reshape|aten::slice.Tensor|4|
|1|Tensor<[1,32,32,128]>,<br>indices: [0:1, 0:32, 0:32, 64:128]<br>|ttnn.reshape|aten::slice.Tensor|4|
|2|Tensor<[1,7,2304]>,<br>indices: [0:1, 0:7, 0:768]<br>|ttnn.reshape|aten::slice.Tensor|4|
|3|Tensor<[1,7,2304]>,<br>indices: [0:1, 0:7, 768:1536]<br>|ttnn.reshape|aten::slice.Tensor|4|
|4|Tensor<[1,7,2304]>,<br>indices: [0:1, 0:7, 1536:2304]<br>|ttnn.reshape|aten::slice.Tensor|4|
|5|Tensor<[1,185,28,28]>,<br>indices: [0:1, 0:128, 0:28, 0:28]<br>|ttnn.reshape|aten::slice.Tensor|4|
|6|Tensor<[1,185,28,28]>,<br>indices: [0:1, 128:185, 0:28, 0:28]<br>|ttnn.reshape|aten::slice.Tensor|4|
|7|Tensor<[6,1,100,4]>,<br>indices: [5:6, 0:1, 0:100, 0:4]<br>|ttnn.reshape|aten::select.int|4|
|8|Tensor<[6,1,100,92]>,<br>indices: [5:6, 0:1, 0:100, 0:92]<br>|ttnn.reshape|aten::select.int|4|
|9|Tensor<[1,23,40]>,<br>indices: [0:1, 22:23, 0:40]<br>|ttnn.reshape|aten::slice.Tensor|4|
|10|Tensor<[1,23,40]>,<br>indices: [0:1, 0:23, 39:40]<br>|ttnn.reshape|aten::slice.Tensor|4|
|11|Tensor<[1,23,40,128]>,<br>indices: [0:1, 0:23, 0:40, 0:128:2]<br>|ttnn.reshape|aten::slice.Tensor|4|
|12|Tensor<[1,23,40,128]>,<br>indices: [0:1, 0:23, 0:40, 1:128:2]<br>|ttnn.reshape|aten::slice.Tensor|4|
|13|Tensor<[768,256]>,<br>indices: [0:256, 0:256]<br>|ttnn.reshape|aten::slice.Tensor|4|
|14|Tensor<[768,256]>,<br>indices: [256:512, 0:256]<br>|ttnn.reshape|aten::slice.Tensor|4|
|15|Tensor<[768,256]>,<br>indices: [512:768, 0:256]<br>|ttnn.reshape|aten::slice.Tensor|4|
|16|Tensor<[768]>,<br>indices: [0:256]<br>|ttnn.reshape|aten::slice.Tensor|4|
|17|Tensor<[768]>,<br>indices: [256:512]<br>|ttnn.reshape|aten::slice.Tensor|4|
|18|Tensor<[768]>,<br>indices: [512:768]<br>|ttnn.reshape|aten::slice.Tensor|4|
|19|Tensor<[1,514]>,<br>indices: [0:1, 0:10]<br>|ttnn.reshape|aten::slice.Tensor|4|
|20|Tensor<[1,320]>,<br>indices: [0:1, 160:320]<br>|ttnn.reshape|aten::slice.Tensor|4|
|21|Tensor<[1,320]>,<br>indices: [0:1, 0:160]<br>|ttnn.reshape|aten::slice.Tensor|4|
|22|Tensor<[1,4096,2560]>,<br>indices: [0:1, 0:4096, 0:1280]<br>|ttnn.reshape|aten::slice.Tensor|4|
|23|Tensor<[1,4096,2560]>,<br>indices: [0:1, 0:4096, 1280:2560]<br>|ttnn.reshape|aten::slice.Tensor|4|
|24|Tensor<[1,1024,5120]>,<br>indices: [0:1, 0:1024, 0:2560]<br>|ttnn.reshape|aten::slice.Tensor|4|
|25|Tensor<[1,1024,5120]>,<br>indices: [0:1, 0:1024, 2560:5120]<br>|ttnn.reshape|aten::slice.Tensor|4|
|26|Tensor<[1,256,10240]>,<br>indices: [0:1, 0:256, 0:5120]<br>|ttnn.reshape|aten::slice.Tensor|4|
|27|Tensor<[1,256,10240]>,<br>indices: [0:1, 0:256, 5120:10240]<br>|ttnn.reshape|aten::slice.Tensor|4|
|28|Tensor<[1,64,10240]>,<br>indices: [0:1, 0:64, 0:5120]<br>|ttnn.reshape|aten::slice.Tensor|4|
|29|Tensor<[1,64,10240]>,<br>indices: [0:1, 0:64, 5120:10240]<br>|ttnn.reshape|aten::slice.Tensor|4|
|30|Tensor<[1,25,768]>,<br>indices: [0:1, 0:1, 0:768]<br>|ttnn.reshape|aten::select.int|4|
|31|Tensor<[1,512]>,<br>indices: [0:1, 0:25]<br>|ttnn.reshape|aten::slice.Tensor|4|
|32|Tensor<[1,25,2]>,<br>indices: [0:1, 0:25, 0:1]<br>|ttnn.reshape|aten::slice.Tensor|4|
|33|Tensor<[1,25,2]>,<br>indices: [0:1, 0:25, 1:2]<br>|ttnn.reshape|aten::slice.Tensor|4|
|34|Tensor<[1,4251,192]>,<br>indices: [0:1, 0:1, 0:192]<br>|ttnn.reshape|aten::select.int|4|
|35|Tensor<[1,4251,192]>,<br>indices: [0:1, 4151:4251, 0:192]<br>|ttnn.reshape|aten::slice.Tensor|4|
|36|Tensor<[1,4251,192]>,<br>indices: [0:1, 1:4151, 0:192]<br>|ttnn.reshape|aten::slice.Tensor|4|
|37|Tensor<[1,1445,192]>,<br>indices: [0:1, 1345:1445, 0:192]<br>|ttnn.reshape|aten::slice.Tensor|4|
|38|Tensor<[1,8,768]>,<br>indices: [0:1, 0:1, 0:768]<br>|ttnn.reshape|aten::select.int|4|
|39|Tensor<[1,512]>,<br>indices: [0:1, 0:8]<br>|ttnn.reshape|aten::slice.Tensor|4|
|40|Tensor<[1,16]>,<br>indices: [0:1, 0:1]<br>|ttnn.reshape|aten::select.int|4|
|41|Tensor<[1,12]>,<br>indices: [0:1, 0:1]<br>|ttnn.reshape|aten::select.int|4|
|42|Tensor<[192,2]>,<br>indices: [0:192, 0:1]<br>|ttnn.reshape|aten::select.int|4|
|43|Tensor<[1,201,768]>,<br>indices: [0:1, 0:1, 0:768]<br>|ttnn.reshape|aten::select.int|4|
|44|Tensor<[1,40]>,<br>indices: [0:1, 0:8]<br>|ttnn.reshape|aten::slice.Tensor|4|
|45|Tensor<[1,145,768]>,<br>indices: [0:1, 1:145, 0:768]<br>|ttnn.reshape|aten::slice.Tensor|4|
|46|Tensor<[1,19]>,<br>indices: [0:1, 18:19]<br>|ttnn.reshape|aten::select.int|4|
|47|Tensor<[1,19]>,<br>indices: [0:1, 1:19]<br>|ttnn.reshape|aten::slice.Tensor|4|
|48|Tensor<[1,19]>,<br>indices: [0:1, 0:18]<br>|ttnn.reshape|aten::slice.Tensor|4|
|49|Tensor<[1,32,16,3,96]>,<br>indices: [0:1, 0:32, 0:16, 0:1, 0:96]<br>|ttnn.reshape|aten::select.int|4|
|50|Tensor<[1,32,16,3,96]>,<br>indices: [0:1, 0:32, 0:16, 1:2, 0:96]<br>|ttnn.reshape|aten::select.int|4|
|51|Tensor<[1,32,16,3,96]>,<br>indices: [0:1, 0:32, 0:16, 2:3, 0:96]<br>|ttnn.reshape|aten::select.int|4|
|52|Tensor<[1,512]>,<br>indices: [0:1, 0:16]<br>|ttnn.reshape|aten::slice.Tensor|4|
|53|Tensor<[1,2,30,40]>,<br>indices: [0:1, 0:1, 0:30, 0:40]<br>|ttnn.reshape|aten::select.int|4|
|54|Tensor<[1,2,30,40]>,<br>indices: [0:1, 1:2, 0:30, 0:40]<br>|ttnn.reshape|aten::select.int|4|
|55|Tensor<[1,2,60,80]>,<br>indices: [0:1, 0:1, 0:60, 0:80]<br>|ttnn.reshape|aten::select.int|4|
|56|Tensor<[1,2,60,80]>,<br>indices: [0:1, 1:2, 0:60, 0:80]<br>|ttnn.reshape|aten::select.int|4|
|57|Tensor<[1,2,120,160]>,<br>indices: [0:1, 0:1, 0:120, 0:160]<br>|ttnn.reshape|aten::select.int|4|
|58|Tensor<[1,2,120,160]>,<br>indices: [0:1, 1:2, 0:120, 0:160]<br>|ttnn.reshape|aten::select.int|4|
|59|Tensor<[1,197,768]>,<br>indices: [0:1, 0:1, 0:768]<br>|ttnn.reshape|aten::select.int|4|
|60|Tensor<[1,7,73,64]>,<br>indices: [0:1, 0:7, 0:71, 0:64]<br>|ttnn.reshape|aten::slice.Tensor|4|
|61|Tensor<[1,71,7,64]>,<br>indices: [0:1, 0:71, 0:7, 0:32]<br>|ttnn.reshape|aten::slice.Tensor|4|
|62|Tensor<[1,71,7,64]>,<br>indices: [0:1, 0:71, 0:7, 32:64]<br>|ttnn.reshape|aten::slice.Tensor|4|
|63|Tensor<[1,1,7,64]>,<br>indices: [0:1, 0:1, 0:7, 0:32]<br>|ttnn.reshape|aten::slice.Tensor|4|
|64|Tensor<[1,1,7,64]>,<br>indices: [0:1, 0:1, 0:7, 32:64]<br>|ttnn.reshape|aten::slice.Tensor|4|
|65|Tensor<[1,512]>,<br>indices: [0:1, 0:12]<br>|ttnn.reshape|aten::slice.Tensor|4|
|66|Tensor<[1,512]>,<br>indices: [0:1, 0:9]<br>|ttnn.reshape|aten::slice.Tensor|4|
|67|Tensor<[1,9,768]>,<br>indices: [0:1, 0:1, 0:768]<br>|ttnn.reshape|aten::select.int|4|
|68|Tensor<[1,512]>,<br>indices: [0:1, 0:14]<br>|ttnn.reshape|aten::slice.Tensor|4|
|69|Tensor<[1,14,2]>,<br>indices: [0:1, 0:14, 0:1]<br>|ttnn.reshape|aten::slice.Tensor|4|
|70|Tensor<[1,14,2]>,<br>indices: [0:1, 0:14, 1:2]<br>|ttnn.reshape|aten::slice.Tensor|4|
|71|Tensor<[1,50,768]>,<br>indices: [0:1, 0:1, 0:768]<br>|ttnn.reshape|aten::select.int|4|
|72|Tensor<[1,77]>,<br>indices: [0:1, 0:7]<br>|ttnn.reshape|aten::slice.Tensor|4|
|73|Tensor<[196,196,2]>,<br>indices: [0:196, 0:196, 0:1]<br>|ttnn.reshape|aten::select.int|4|
|74|Tensor<[196,196,2]>,<br>indices: [0:196, 0:196, 1:2]<br>|ttnn.reshape|aten::select.int|4|
|75|Tensor<[197,197]>,<br>indices: [0:1, 0:197]<br>|ttnn.reshape|aten::select.int|4|
|76|Tensor<[197,197]>,<br>indices: [0:197, 0:1]<br>|ttnn.reshape|aten::select.int|4|
|77|Tensor<[197]>,<br>indices: [0:1]<br>|ttnn.reshape|aten::select.int|4|
|78|Tensor<[732,16]>,<br>indices: [0:729, 0:16]<br>|ttnn.reshape|aten::slice.Tensor|4|
|79|Tensor<[732,16]>,<br>indices: [729:732, 0:16]<br>|ttnn.reshape|aten::slice.Tensor|4|
|80|Tensor<[197,197]>,<br>indices: [1:197, 0:197]<br>|ttnn.reshape|aten::slice.Tensor|4|
|81|Tensor<[196,197]>,<br>indices: [0:196, 1:197]<br>|ttnn.reshape|aten::slice.Tensor|4|
|82|Tensor<[1,197,1024]>,<br>indices: [0:1, 1:197, 0:1024]<br>|ttnn.reshape|aten::slice.Tensor|4|
|83|Tensor<[732,12]>,<br>indices: [0:729, 0:12]<br>|ttnn.reshape|aten::slice.Tensor|4|
|84|Tensor<[732,12]>,<br>indices: [729:732, 0:12]<br>|ttnn.reshape|aten::slice.Tensor|4|
|85|Tensor<[1,197,768]>,<br>indices: [0:1, 1:197, 0:768]<br>|ttnn.reshape|aten::slice.Tensor|4|
