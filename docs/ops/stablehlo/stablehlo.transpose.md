
### stablehlo.transpose::ttnn.permute


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,64,32]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|1|Tensor<[4096,4096]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|2|Tensor<[1,32,32,128]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|3|Tensor<[1,32,32,128]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|4|Tensor<[11008,4096]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|5|Tensor<[4096,11008]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|6|Tensor<[32000,4096]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|7|Tensor<[1,7,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|8|Tensor<[1,12,7,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|9|Tensor<[1,12,7,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|10|Tensor<[2,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|11|Tensor<[1,3,16,16,16,16]>,<br>dims: [0, 2, 4, 3, 5, 1]<br>|ttnn.permute|aten::permute|4|
|12|Tensor<[1,256,512]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|13|Tensor<[512,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|14|Tensor<[256,512]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|15|Tensor<[512,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|16|Tensor<[1000,512]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|17|Tensor<[1,23,40,256]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|18|Tensor<[1,256,920]>,<br>dims: [2, 0, 1]<br>|ttnn.permute|aten::permute|4|
|19|Tensor<[256,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|20|Tensor<[920,8,32]>,<br>dims: [1, 0, 2]<br>|ttnn.permute|aten::transpose.int|4|
|21|Tensor<[8,920,32]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|22|Tensor<[8,920,32]>,<br>dims: [1, 0, 2]<br>|ttnn.permute|aten::transpose.int|4|
|23|Tensor<[2048,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|24|Tensor<[256,2048]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|25|Tensor<[100,8,32]>,<br>dims: [1, 0, 2]<br>|ttnn.permute|aten::transpose.int|4|
|26|Tensor<[8,100,32]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|27|Tensor<[8,100,32]>,<br>dims: [1, 0, 2]<br>|ttnn.permute|aten::transpose.int|4|
|28|Tensor<[6,100,1,256]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|29|Tensor<[92,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|30|Tensor<[4,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|31|Tensor<[1,10,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|32|Tensor<[768,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|33|Tensor<[1,12,10,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|34|Tensor<[1,12,10,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|35|Tensor<[3072,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|36|Tensor<[768,3072]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|37|Tensor<[250002,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|38|Tensor<[1,320,64,64]>,<br>dims: [0, 2, 3, 1]<br>|ttnn.permute|aten::permute|4|
|39|Tensor<[1,64,64,320]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|40|Tensor<[1,640,32,32]>,<br>dims: [0, 2, 3, 1]<br>|ttnn.permute|aten::permute|4|
|41|Tensor<[1,32,32,640]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|42|Tensor<[1,1280,16,16]>,<br>dims: [0, 2, 3, 1]<br>|ttnn.permute|aten::permute|4|
|43|Tensor<[1,16,16,1280]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|44|Tensor<[1,1280,8,8]>,<br>dims: [0, 2, 3, 1]<br>|ttnn.permute|aten::permute|4|
|45|Tensor<[1,8,8,1280]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|46|Tensor<[1280,320]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|47|Tensor<[1280,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|48|Tensor<[320,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|49|Tensor<[320,320]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|50|Tensor<[1,4096,8,40]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|51|Tensor<[1,8,4096,40]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|52|Tensor<[1,8,4096,40]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|53|Tensor<[320,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|54|Tensor<[1,9,8,40]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|55|Tensor<[1,8,9,40]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|56|Tensor<[2560,320]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|57|Tensor<[640,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|58|Tensor<[640,640]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|59|Tensor<[1,1024,8,80]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|60|Tensor<[1,8,1024,80]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|61|Tensor<[1,8,1024,80]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|62|Tensor<[640,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|63|Tensor<[1,9,8,80]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|64|Tensor<[1,8,9,80]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|65|Tensor<[5120,640]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|66|Tensor<[640,2560]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|67|Tensor<[1,256,8,160]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|68|Tensor<[1,8,256,160]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|69|Tensor<[1,8,256,160]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|70|Tensor<[1280,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|71|Tensor<[1,9,8,160]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|72|Tensor<[1,8,9,160]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|73|Tensor<[10240,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|74|Tensor<[1280,5120]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|75|Tensor<[1,64,8,160]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|76|Tensor<[1,8,64,160]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|77|Tensor<[1,8,64,160]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|78|Tensor<[1,25,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|79|Tensor<[1,12,25,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|80|Tensor<[1,12,25,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|81|Tensor<[1,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|82|Tensor<[1,1445,3,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|83|Tensor<[1,3,1445,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|84|Tensor<[1,192,1344]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|85|Tensor<[1,4150,192]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|86|Tensor<[192,192]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|87|Tensor<[1,3,1445,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|88|Tensor<[768,192]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|89|Tensor<[192,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|90|Tensor<[92,192]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|91|Tensor<[4,192]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|92|Tensor<[1,8,768]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|93|Tensor<[1,12,64,8]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::permute|4|
|94|Tensor<[1,12,8,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::permute|4|
|95|Tensor<[1,768,8]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|96|Tensor<[3,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|97|Tensor<[1,256,8,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|98|Tensor<[1,2048,8,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|99|Tensor<[1,2048,8,160]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|100|Tensor<[1,256,8,96]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|101|Tensor<[1,8,2048,96]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|102|Tensor<[256,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|103|Tensor<[256,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|104|Tensor<[1,8,2048,32]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|105|Tensor<[1,8,256,32]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|106|Tensor<[768,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|107|Tensor<[262,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|108|Tensor<[1000,2048]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|109|Tensor<[1,201,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|110|Tensor<[1,12,201,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|111|Tensor<[1,144,768]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|112|Tensor<[1,768,192]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|113|Tensor<[1,12,201,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|114|Tensor<[1536,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|115|Tensor<[3129,1536]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|116|Tensor<[128,9216]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|117|Tensor<[10,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|118|Tensor<[1024,1024]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|119|Tensor<[1,19,16,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|120|Tensor<[16,19,64]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|121|Tensor<[1,16,19,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|122|Tensor<[4096,1024]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|123|Tensor<[1024,4096]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|124|Tensor<[256008,1024]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|125|Tensor<[1000,1024]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|126|Tensor<[512,256,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|127|Tensor<[256,128,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|128|Tensor<[128,64,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|129|Tensor<[64,32,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|130|Tensor<[4,16,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|131|Tensor<[16,1,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|132|Tensor<[1,16,32,96]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|133|Tensor<[4608,1536]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|134|Tensor<[1,32,16,96]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|135|Tensor<[16,32,96]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|136|Tensor<[1536,1536]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|137|Tensor<[6144,1536]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|138|Tensor<[1536,6144]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|139|Tensor<[250880,1536]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|140|Tensor<[1,16,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|141|Tensor<[1,12,16,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|142|Tensor<[1,12,16,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|143|Tensor<[1024,512,2,2]>,<br>dims: [2, 3, 1, 0]<br>|ttnn.permute|aten::convolution|4|
|144|Tensor<[1,19200,1,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|145|Tensor<[1,19200,64]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|146|Tensor<[1,64,300]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|147|Tensor<[1,300,1,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|148|Tensor<[1,1,19200,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|149|Tensor<[1,120,160,64]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|150|Tensor<[1,4800,2,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|151|Tensor<[1,4800,128]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|152|Tensor<[1,128,300]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|153|Tensor<[1,300,2,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|154|Tensor<[1,2,4800,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|155|Tensor<[1,60,80,128]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|156|Tensor<[1,1200,5,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|157|Tensor<[1,1200,320]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|158|Tensor<[1,320,300]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|159|Tensor<[1,300,5,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|160|Tensor<[1,5,1200,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|161|Tensor<[1,30,40,320]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|162|Tensor<[1,300,8,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|163|Tensor<[1,8,300,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|164|Tensor<[1,15,20,512]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|165|Tensor<[1,64,19200]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|166|Tensor<[64,64]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|167|Tensor<[1,1,300,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|168|Tensor<[256,64]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|169|Tensor<[1,19200,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|170|Tensor<[1,256,19200]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|171|Tensor<[64,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|172|Tensor<[1,128,4800]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|173|Tensor<[128,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|174|Tensor<[1,2,300,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|175|Tensor<[512,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|176|Tensor<[1,4800,512]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|177|Tensor<[1,512,4800]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|178|Tensor<[128,512]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|179|Tensor<[1,320,1200]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|180|Tensor<[1,5,300,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|181|Tensor<[1,1200,1280]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|182|Tensor<[1,1280,1200]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|183|Tensor<[1,512,300]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|184|Tensor<[512,512]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|185|Tensor<[1,8,300,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|186|Tensor<[2048,512]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|187|Tensor<[1,300,2048]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|188|Tensor<[1,2048,300]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|189|Tensor<[512,2048]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|190|Tensor<[1,197,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|191|Tensor<[1,12,197,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|192|Tensor<[1,768,196]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|193|Tensor<[1,12,197,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|194|Tensor<[1000,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|195|Tensor<[1,16384,1,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|196|Tensor<[1,16384,32]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|197|Tensor<[1,32,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|198|Tensor<[1,256,1,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|199|Tensor<[1,1,16384,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|200|Tensor<[1,128,128,32]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|201|Tensor<[1,4096,2,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|202|Tensor<[1,4096,64]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|203|Tensor<[1,64,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|204|Tensor<[1,256,2,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|205|Tensor<[1,2,4096,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|206|Tensor<[1,64,64,64]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|207|Tensor<[1,1024,5,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|208|Tensor<[1,1024,160]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|209|Tensor<[1,160,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|210|Tensor<[1,256,5,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|211|Tensor<[1,5,1024,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|212|Tensor<[1,32,32,160]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|213|Tensor<[1,8,256,32]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|214|Tensor<[1,16,16,256]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|215|Tensor<[1,16384,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|216|Tensor<[1,4096,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|217|Tensor<[1,1024,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|218|Tensor<[1,256,256]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::permute|4|
|219|Tensor<[1,32,16384]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|220|Tensor<[32,32]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|221|Tensor<[1,1,256,32]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|222|Tensor<[128,32]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|223|Tensor<[1,16384,128]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|224|Tensor<[1,128,16384]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|225|Tensor<[32,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|226|Tensor<[1,64,4096]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|227|Tensor<[1,2,256,32]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|228|Tensor<[1,256,4096]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|229|Tensor<[1,160,1024]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|230|Tensor<[160,160]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|231|Tensor<[1,5,256,32]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|232|Tensor<[640,160]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|233|Tensor<[1,1024,640]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|234|Tensor<[1,640,1024]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|235|Tensor<[160,640]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|236|Tensor<[1024,256]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|237|Tensor<[1,256,1024]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|238|Tensor<[256,1024]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|239|Tensor<[256,32]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|240|Tensor<[256,160]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|241|Tensor<[4672,4544]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::permute|5|
|242|Tensor<[1,71,7,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|243|Tensor<[4544,4544]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::permute|5|
|244|Tensor<[18176,4544]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::permute|5|
|245|Tensor<[4544,18176]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::permute|5|
|246|Tensor<[1,32,7]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|247|Tensor<[1,7,71,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|248|Tensor<[1,7,1,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|249|Tensor<[1,1,7,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|250|Tensor<[65024,4544]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|251|Tensor<[1000,1280]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|252|Tensor<[1,12,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|253|Tensor<[768,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|254|Tensor<[1,12,12,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|255|Tensor<[1,9,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|256|Tensor<[1,12,9,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|257|Tensor<[1,12,9,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|258|Tensor<[128,768]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|259|Tensor<[30000,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|260|Tensor<[1,9,16,128]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|261|Tensor<[2048,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|262|Tensor<[2048,2048]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|263|Tensor<[1,16,9,128]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|264|Tensor<[1,16,9,128]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|265|Tensor<[8192,2048]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|266|Tensor<[2048,8192]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|267|Tensor<[128,2048]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|268|Tensor<[1,9,16,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|269|Tensor<[1024,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|270|Tensor<[1,16,9,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|271|Tensor<[1,16,9,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|272|Tensor<[128,1024]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|273|Tensor<[1,9,64,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|274|Tensor<[4096,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|275|Tensor<[1,64,9,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|276|Tensor<[1,64,9,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|277|Tensor<[16384,4096]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|278|Tensor<[4096,16384]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|279|Tensor<[128,4096]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|280|Tensor<[1,14,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|281|Tensor<[1,12,14,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|282|Tensor<[1,12,14,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|283|Tensor<[1,768,49]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|284|Tensor<[1,50,12,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|285|Tensor<[1,12,50,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|286|Tensor<[1,12,50,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|287|Tensor<[2,7,8,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|288|Tensor<[2,8,7,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|289|Tensor<[2,8,7,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::transpose.int|4|
|290|Tensor<[1,512]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|291|Tensor<[2,1]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|292|Tensor<[1,197,16,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|293|Tensor<[1,27,27,16]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|294|Tensor<[1,16,27,27]>,<br>dims: [0, 2, 3, 1]<br>|ttnn.permute|aten::permute|4|
|295|Tensor<[2,196,196]>,<br>dims: [1, 2, 0]<br>|ttnn.permute|aten::permute|4|
|296|Tensor<[197,197,16]>,<br>dims: [2, 0, 1]<br>|ttnn.permute|aten::permute|4|
|297|Tensor<[1,16,197,64]>,<br>dims: [0, 2, 1, 3]<br>|ttnn.permute|aten::permute|4|
|298|Tensor<[1,1024,196]>,<br>dims: [0, 2, 1]<br>|ttnn.permute|aten::transpose.int|4|
|299|Tensor<[1,16,197,64]>,<br>dims: [0, 1, 3, 2]<br>|ttnn.permute|aten::transpose.int|4|
|300|Tensor<[1,27,27,12]>,<br>dims: [0, 3, 1, 2]<br>|ttnn.permute|aten::permute|4|
|301|Tensor<[1,12,27,27]>,<br>dims: [0, 2, 3, 1]<br>|ttnn.permute|aten::permute|4|
|302|Tensor<[197,197,12]>,<br>dims: [2, 0, 1]<br>|ttnn.permute|aten::permute|4|
|303|Tensor<[128,784]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|304|Tensor<[64,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|305|Tensor<[12,64]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|306|Tensor<[3,12]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|307|Tensor<[12,3]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|308|Tensor<[64,12]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|309|Tensor<[128,64]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
|310|Tensor<[784,128]>,<br>dims: [1, 0]<br>|ttnn.permute|aten::transpose.int|5|
