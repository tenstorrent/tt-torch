
### stablehlo.multiply::ttnn.multiply


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[32]>,<br>Tensor<[32]>,<br>|ttnn.multiply|aten::arange|4|
|1|Tensor<[1,32,32,128]>,<br>Tensor<[1,32,32,128]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|2|Tensor<[1,32,128,32]>,<br>Tensor<[1,32,128,32]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|3|Tensor<[32,32]>,<br>Tensor<[32,32]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|4|Tensor<[1,32,128]>,<br>Tensor<[1,32,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|5|Tensor<[1,32,4096]>,<br>Tensor<[1,32,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|6|Tensor<[1,32,11008]>,<br>Tensor<[1,32,11008]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|7|Tensor<[7]>,<br>Tensor<[7]>,<br>|ttnn.multiply|aten::arange|4|
|8|Tensor<[1]>,<br>Tensor<[1]>,<br>|ttnn.multiply|aten::arange|4|
|9|Tensor<[1,12,7,64]>,<br>Tensor<[1,12,7,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|10|Tensor<[1,12,64,7]>,<br>Tensor<[1,12,64,7]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|11|Tensor<[1,7,768]>,<br>Tensor<[1,7,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|12|Tensor<[7,2304]>,<br>Tensor<[7,2304]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|13|Tensor<[2304]>,<br>Tensor<[2304]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|14|Tensor<[7,768]>,<br>Tensor<[7,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|15|Tensor<[768]>,<br>Tensor<[768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|16|Tensor<[7,3072]>,<br>Tensor<[7,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|17|Tensor<[3072]>,<br>Tensor<[3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|18|Tensor<[1,7,3072]>,<br>Tensor<[1,7,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|19|Tensor<[1,128,28,28]>,<br>Tensor<[1,128,28,28]>,<br>|ttnn.multiply|aten::elu|4|
|20|Tensor<[1,32,112,112]>,<br>Tensor<[1,32,112,112]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|21|Tensor<[64]>,<br>Tensor<[64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|22|Tensor<[1,64,112,112]>,<br>Tensor<[1,64,112,112]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|23|Tensor<[1,64,56,56]>,<br>Tensor<[1,64,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|24|Tensor<[128]>,<br>Tensor<[128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|25|Tensor<[1,128,56,56]>,<br>Tensor<[1,128,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|26|Tensor<[256]>,<br>Tensor<[256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|27|Tensor<[1,256,28,28]>,<br>Tensor<[1,256,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|28|Tensor<[512]>,<br>Tensor<[512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|29|Tensor<[1,512,28,28]>,<br>Tensor<[1,512,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|30|Tensor<[1,1024,512]>,<br>Tensor<[1,1024,512]>,<br>|ttnn.multiply|aten::gelu|4|
|31|Tensor<[1,256,256]>,<br>Tensor<[1,256,256]>,<br>|ttnn.multiply|aten::gelu|4|
|32|Tensor<[256,512]>,<br>Tensor<[256,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|33|Tensor<[1,256,512]>,<br>Tensor<[1,256,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|34|Tensor<[256,256]>,<br>Tensor<[256,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|35|Tensor<[1,1000]>,<br>Tensor<[1,1000]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|36|Tensor<[1000]>,<br>Tensor<[1000]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|37|Tensor<[23]>,<br>Tensor<[23]>,<br>|ttnn.multiply|aten::arange|4|
|38|Tensor<[40]>,<br>Tensor<[40]>,<br>|ttnn.multiply|aten::arange|4|
|39|Tensor<[8,920,920]>,<br>Tensor<[8,920,920]>,<br>|ttnn.multiply|aten::baddbmm|4|
|40|Tensor<[8,100,920]>,<br>Tensor<[8,100,920]>,<br>|ttnn.multiply|aten::baddbmm|4|
|41|Tensor<[1,64,1,1]>,<br>Tensor<[1,64,1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|42|Tensor<[1,64,360,640]>,<br>Tensor<[1,64,360,640]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|43|Tensor<[1,64,180,320]>,<br>Tensor<[1,64,180,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|44|Tensor<[1,256,1,1]>,<br>Tensor<[1,256,1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|45|Tensor<[1,256,180,320]>,<br>Tensor<[1,256,180,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|46|Tensor<[1,128,1,1]>,<br>Tensor<[1,128,1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|47|Tensor<[1,128,180,320]>,<br>Tensor<[1,128,180,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|48|Tensor<[1,128,90,160]>,<br>Tensor<[1,128,90,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|49|Tensor<[1,512,1,1]>,<br>Tensor<[1,512,1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|50|Tensor<[1,512,90,160]>,<br>Tensor<[1,512,90,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|51|Tensor<[1,256,90,160]>,<br>Tensor<[1,256,90,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|52|Tensor<[1,256,45,80]>,<br>Tensor<[1,256,45,80]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|53|Tensor<[1,1024,1,1]>,<br>Tensor<[1,1024,1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|54|Tensor<[1,1024,45,80]>,<br>Tensor<[1,1024,45,80]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|55|Tensor<[1,512,45,80]>,<br>Tensor<[1,512,45,80]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|56|Tensor<[1,512,23,40]>,<br>Tensor<[1,512,23,40]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|57|Tensor<[1,2048,1,1]>,<br>Tensor<[1,2048,1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|58|Tensor<[1,2048,23,40]>,<br>Tensor<[1,2048,23,40]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|59|Tensor<[1,23,40]>,<br>Tensor<[1,23,40]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|60|Tensor<[8,920,32]>,<br>Tensor<[8,920,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|61|Tensor<[920,256]>,<br>Tensor<[920,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|62|Tensor<[920,1,256]>,<br>Tensor<[920,1,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|63|Tensor<[920,2048]>,<br>Tensor<[920,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|64|Tensor<[2048]>,<br>Tensor<[2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|65|Tensor<[100,256]>,<br>Tensor<[100,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|66|Tensor<[8,100,32]>,<br>Tensor<[8,100,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|67|Tensor<[100,1,256]>,<br>Tensor<[100,1,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|68|Tensor<[100,2048]>,<br>Tensor<[100,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|69|Tensor<[1,10,3072]>,<br>Tensor<[1,10,3072]>,<br>|ttnn.multiply|aten::gelu|4|
|70|Tensor<[1,10,768]>,<br>Tensor<[1,10,768]>,<br>|ttnn.multiply|aten::gelu|4|
|71|Tensor<[1,12,10,64]>,<br>Tensor<[1,12,10,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|72|Tensor<[1,12,64,10]>,<br>Tensor<[1,12,64,10]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|73|Tensor<[1,10]>,<br>Tensor<[1,10]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|74|Tensor<[10,768]>,<br>Tensor<[10,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|75|Tensor<[10,3072]>,<br>Tensor<[10,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|76|Tensor<[10,250002]>,<br>Tensor<[10,250002]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|77|Tensor<[250002]>,<br>Tensor<[250002]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|78|Tensor<[16]>,<br>Tensor<[16]>,<br>|ttnn.multiply|aten::arange|4|
|79|Tensor<[160]>,<br>Tensor<[160]>,<br>|ttnn.multiply|aten::arange.start|4|
|80|Tensor<[1,4096,1280]>,<br>Tensor<[1,4096,1280]>,<br>|ttnn.multiply|aten::gelu|4|
|81|Tensor<[1,1024,2560]>,<br>Tensor<[1,1024,2560]>,<br>|ttnn.multiply|aten::gelu|4|
|82|Tensor<[1,256,5120]>,<br>Tensor<[1,256,5120]>,<br>|ttnn.multiply|aten::gelu|4|
|83|Tensor<[1,64,5120]>,<br>Tensor<[1,64,5120]>,<br>|ttnn.multiply|aten::gelu|4|
|84|Tensor<[1280]>,<br>Tensor<[1280]>,<br>|ttnn.multiply|aten::index.Tensor|4|
|85|Tensor<[640]>,<br>Tensor<[640]>,<br>|ttnn.multiply|aten::index.Tensor|4|
|86|Tensor<[1,8,4096,40]>,<br>Tensor<[1,8,4096,40]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|87|Tensor<[1,8,40,4096]>,<br>Tensor<[1,8,40,4096]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|88|Tensor<[1,8,40,9]>,<br>Tensor<[1,8,40,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|89|Tensor<[1,8,1024,80]>,<br>Tensor<[1,8,1024,80]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|90|Tensor<[1,8,80,1024]>,<br>Tensor<[1,8,80,1024]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|91|Tensor<[1,8,80,9]>,<br>Tensor<[1,8,80,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|92|Tensor<[1,8,256,160]>,<br>Tensor<[1,8,256,160]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|93|Tensor<[1,8,160,256]>,<br>Tensor<[1,8,160,256]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|94|Tensor<[1,8,160,9]>,<br>Tensor<[1,8,160,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|95|Tensor<[1,8,64,160]>,<br>Tensor<[1,8,64,160]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|96|Tensor<[1,8,160,64]>,<br>Tensor<[1,8,160,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|97|Tensor<[1,160]>,<br>Tensor<[1,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|98|Tensor<[1,1280]>,<br>Tensor<[1,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|99|Tensor<[1,32,10,4096]>,<br>Tensor<[1,32,10,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|100|Tensor<[1,320,64,64]>,<br>Tensor<[1,320,64,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|101|Tensor<[1,320]>,<br>Tensor<[1,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|102|Tensor<[320]>,<br>Tensor<[320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|103|Tensor<[1,4096,320]>,<br>Tensor<[1,4096,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|104|Tensor<[4096,320]>,<br>Tensor<[4096,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|105|Tensor<[4096,2560]>,<br>Tensor<[4096,2560]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|106|Tensor<[2560]>,<br>Tensor<[2560]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|107|Tensor<[1,32,10,1024]>,<br>Tensor<[1,32,10,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|108|Tensor<[1,320,32,32]>,<br>Tensor<[1,320,32,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|109|Tensor<[1,640]>,<br>Tensor<[1,640]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|110|Tensor<[1,32,20,1024]>,<br>Tensor<[1,32,20,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|111|Tensor<[1,640,32,32]>,<br>Tensor<[1,640,32,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|112|Tensor<[1,1024,640]>,<br>Tensor<[1,1024,640]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|113|Tensor<[1024,640]>,<br>Tensor<[1024,640]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|114|Tensor<[1024,5120]>,<br>Tensor<[1024,5120]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|115|Tensor<[5120]>,<br>Tensor<[5120]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|116|Tensor<[1,32,20,256]>,<br>Tensor<[1,32,20,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|117|Tensor<[1,640,16,16]>,<br>Tensor<[1,640,16,16]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|118|Tensor<[1,32,40,256]>,<br>Tensor<[1,32,40,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|119|Tensor<[1,1280,16,16]>,<br>Tensor<[1,1280,16,16]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|120|Tensor<[1,256,1280]>,<br>Tensor<[1,256,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|121|Tensor<[256,1280]>,<br>Tensor<[256,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|122|Tensor<[256,10240]>,<br>Tensor<[256,10240]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|123|Tensor<[10240]>,<br>Tensor<[10240]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|124|Tensor<[1,32,40,64]>,<br>Tensor<[1,32,40,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|125|Tensor<[1,1280,8,8]>,<br>Tensor<[1,1280,8,8]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|126|Tensor<[1,64,1280]>,<br>Tensor<[1,64,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|127|Tensor<[64,1280]>,<br>Tensor<[64,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|128|Tensor<[64,10240]>,<br>Tensor<[64,10240]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|129|Tensor<[1,32,80,64]>,<br>Tensor<[1,32,80,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|130|Tensor<[1,2560,8,8]>,<br>Tensor<[1,2560,8,8]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|131|Tensor<[1,32,80,256]>,<br>Tensor<[1,32,80,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|132|Tensor<[1,2560,16,16]>,<br>Tensor<[1,2560,16,16]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|133|Tensor<[1,32,60,256]>,<br>Tensor<[1,32,60,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|134|Tensor<[1,1920,16,16]>,<br>Tensor<[1,1920,16,16]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|135|Tensor<[1,32,60,1024]>,<br>Tensor<[1,32,60,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|136|Tensor<[1,1920,32,32]>,<br>Tensor<[1,1920,32,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|137|Tensor<[1,32,40,1024]>,<br>Tensor<[1,32,40,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|138|Tensor<[1,1280,32,32]>,<br>Tensor<[1,1280,32,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|139|Tensor<[1,32,30,1024]>,<br>Tensor<[1,32,30,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|140|Tensor<[1,960,32,32]>,<br>Tensor<[1,960,32,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|141|Tensor<[1,32,30,4096]>,<br>Tensor<[1,32,30,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|142|Tensor<[1,960,64,64]>,<br>Tensor<[1,960,64,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|143|Tensor<[1,32,20,4096]>,<br>Tensor<[1,32,20,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|144|Tensor<[1,640,64,64]>,<br>Tensor<[1,640,64,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|145|Tensor<[1,25,3072]>,<br>Tensor<[1,25,3072]>,<br>|ttnn.multiply|aten::gelu|4|
|146|Tensor<[1,12,25,64]>,<br>Tensor<[1,12,25,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|147|Tensor<[1,12,64,25]>,<br>Tensor<[1,12,64,25]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|148|Tensor<[1,25,768]>,<br>Tensor<[1,25,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|149|Tensor<[25,768]>,<br>Tensor<[25,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|150|Tensor<[25,3072]>,<br>Tensor<[25,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|151|Tensor<[25,2]>,<br>Tensor<[25,2]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|152|Tensor<[2]>,<br>Tensor<[2]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|153|Tensor<[1,1]>,<br>Tensor<[1,1]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|154|Tensor<[1,1445,768]>,<br>Tensor<[1,1445,768]>,<br>|ttnn.multiply|aten::gelu|4|
|155|Tensor<[1,3,1445,64]>,<br>Tensor<[1,3,1445,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|156|Tensor<[1,3,64,1445]>,<br>Tensor<[1,3,64,1445]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|157|Tensor<[1,1445,192]>,<br>Tensor<[1,1445,192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|158|Tensor<[1445,192]>,<br>Tensor<[1445,192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|159|Tensor<[192]>,<br>Tensor<[192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|160|Tensor<[1445,768]>,<br>Tensor<[1445,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|161|Tensor<[100,192]>,<br>Tensor<[100,192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|162|Tensor<[100,92]>,<br>Tensor<[100,92]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|163|Tensor<[92]>,<br>Tensor<[92]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|164|Tensor<[100,4]>,<br>Tensor<[100,4]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|165|Tensor<[4]>,<br>Tensor<[4]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|166|Tensor<[1,256,14,14]>,<br>Tensor<[1,256,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|167|Tensor<[1,512,7,7]>,<br>Tensor<[1,512,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|168|Tensor<[1,3072,8]>,<br>Tensor<[1,3072,8]>,<br>|ttnn.multiply|aten::gelu|4|
|169|Tensor<[1,1,1,8]>,<br>Tensor<[1,1,1,8]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|170|Tensor<[1,8,768]>,<br>Tensor<[1,8,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|171|Tensor<[1,768]>,<br>Tensor<[1,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|172|Tensor<[1,3]>,<br>Tensor<[1,3]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|173|Tensor<[3]>,<br>Tensor<[3]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|174|Tensor<[1,2048,768]>,<br>Tensor<[1,2048,768]>,<br>|ttnn.multiply|aten::gelu|4|
|175|Tensor<[1,1,1,2048]>,<br>Tensor<[1,1,1,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|176|Tensor<[2048,256]>,<br>Tensor<[2048,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|177|Tensor<[2048,1280]>,<br>Tensor<[2048,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|178|Tensor<[256,768]>,<br>Tensor<[256,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|179|Tensor<[2048,768]>,<br>Tensor<[2048,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|180|Tensor<[1,256,56,56]>,<br>Tensor<[1,256,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|181|Tensor<[1024]>,<br>Tensor<[1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|182|Tensor<[1,1024,14,14]>,<br>Tensor<[1,1024,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|183|Tensor<[1,512,14,14]>,<br>Tensor<[1,512,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|184|Tensor<[1,2048,7,7]>,<br>Tensor<[1,2048,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|185|Tensor<[12]>,<br>Tensor<[12]>,<br>|ttnn.multiply|aten::arange|4|
|186|Tensor<[1,201,3072]>,<br>Tensor<[1,201,3072]>,<br>|ttnn.multiply|aten::gelu|4|
|187|Tensor<[1,1536]>,<br>Tensor<[1,1536]>,<br>|ttnn.multiply|aten::gelu|4|
|188|Tensor<[1,1,1,201]>,<br>Tensor<[1,1,1,201]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|189|Tensor<[1,201,768]>,<br>Tensor<[1,201,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|190|Tensor<[201,768]>,<br>Tensor<[201,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|191|Tensor<[201,3072]>,<br>Tensor<[201,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|192|Tensor<[1536]>,<br>Tensor<[1536]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|193|Tensor<[1,3129]>,<br>Tensor<[1,3129]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|194|Tensor<[3129]>,<br>Tensor<[3129]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|195|Tensor<[1,128]>,<br>Tensor<[1,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|196|Tensor<[10]>,<br>Tensor<[10]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|197|Tensor<[19]>,<br>Tensor<[19]>,<br>|ttnn.multiply|aten::arange|4|
|198|Tensor<[1,19,4096]>,<br>Tensor<[1,19,4096]>,<br>|ttnn.multiply|aten::gelu|4|
|199|Tensor<[1,19,1024]>,<br>Tensor<[1,19,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|200|Tensor<[19,1024]>,<br>Tensor<[19,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|201|Tensor<[19,4096]>,<br>Tensor<[19,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|202|Tensor<[4096]>,<br>Tensor<[4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|203|Tensor<[14]>,<br>Tensor<[14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|204|Tensor<[1,14,56,56]>,<br>Tensor<[1,14,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|205|Tensor<[24]>,<br>Tensor<[24]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|206|Tensor<[1,24,56,56]>,<br>Tensor<[1,24,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|207|Tensor<[1,40,56,56]>,<br>Tensor<[1,40,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|208|Tensor<[68]>,<br>Tensor<[68]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|209|Tensor<[1,68,56,56]>,<br>Tensor<[1,68,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|210|Tensor<[1,16,28,28]>,<br>Tensor<[1,16,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|211|Tensor<[28]>,<br>Tensor<[28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|212|Tensor<[1,28,28,28]>,<br>Tensor<[1,28,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|213|Tensor<[46]>,<br>Tensor<[46]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|214|Tensor<[1,46,28,28]>,<br>Tensor<[1,46,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|215|Tensor<[78]>,<br>Tensor<[78]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|216|Tensor<[1,78,28,28]>,<br>Tensor<[1,78,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|217|Tensor<[134]>,<br>Tensor<[134]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|218|Tensor<[1,134,28,28]>,<br>Tensor<[1,134,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|219|Tensor<[20]>,<br>Tensor<[20]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|220|Tensor<[1,20,28,28]>,<br>Tensor<[1,20,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|221|Tensor<[34]>,<br>Tensor<[34]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|222|Tensor<[1,34,28,28]>,<br>Tensor<[1,34,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|223|Tensor<[58]>,<br>Tensor<[58]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|224|Tensor<[1,58,28,28]>,<br>Tensor<[1,58,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|225|Tensor<[98]>,<br>Tensor<[98]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|226|Tensor<[1,98,28,28]>,<br>Tensor<[1,98,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|227|Tensor<[168]>,<br>Tensor<[168]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|228|Tensor<[1,168,28,28]>,<br>Tensor<[1,168,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|229|Tensor<[1,320,28,28]>,<br>Tensor<[1,320,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|230|Tensor<[1,40,14,14]>,<br>Tensor<[1,40,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|231|Tensor<[1,68,14,14]>,<br>Tensor<[1,68,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|232|Tensor<[116]>,<br>Tensor<[116]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|233|Tensor<[1,116,14,14]>,<br>Tensor<[1,116,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|234|Tensor<[196]>,<br>Tensor<[196]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|235|Tensor<[1,196,14,14]>,<br>Tensor<[1,196,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|236|Tensor<[334]>,<br>Tensor<[334]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|237|Tensor<[1,334,14,14]>,<br>Tensor<[1,334,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|238|Tensor<[1,640,14,14]>,<br>Tensor<[1,640,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|239|Tensor<[1,160,7,7]>,<br>Tensor<[1,160,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|240|Tensor<[272]>,<br>Tensor<[272]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|241|Tensor<[1,272,7,7]>,<br>Tensor<[1,272,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|242|Tensor<[462]>,<br>Tensor<[462]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|243|Tensor<[1,462,7,7]>,<br>Tensor<[1,462,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|244|Tensor<[1,1024,7,7]>,<br>Tensor<[1,1024,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|245|Tensor<[1,32,512,512]>,<br>Tensor<[1,32,512,512]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|246|Tensor<[1,64,256,256]>,<br>Tensor<[1,64,256,256]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|247|Tensor<[1,32,256,256]>,<br>Tensor<[1,32,256,256]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|248|Tensor<[1,128,128,128]>,<br>Tensor<[1,128,128,128]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|249|Tensor<[1,64,128,128]>,<br>Tensor<[1,64,128,128]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|250|Tensor<[1,256,64,64]>,<br>Tensor<[1,256,64,64]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|251|Tensor<[1,128,64,64]>,<br>Tensor<[1,128,64,64]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|252|Tensor<[1,512,32,32]>,<br>Tensor<[1,512,32,32]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|253|Tensor<[1,256,32,32]>,<br>Tensor<[1,256,32,32]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|254|Tensor<[1,1024,16,16]>,<br>Tensor<[1,1024,16,16]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|255|Tensor<[1,512,16,16]>,<br>Tensor<[1,512,16,16]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|256|Tensor<[1,256,16,16]>,<br>Tensor<[1,256,16,16]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|257|Tensor<[1,128,32,32]>,<br>Tensor<[1,128,32,32]>,<br>|ttnn.multiply|aten::leaky_relu|4|
|258|Tensor<[16,32,32]>,<br>Tensor<[16,32,32]>,<br>|ttnn.multiply|aten::baddbmm|4|
|259|Tensor<[1,32,1536]>,<br>Tensor<[1,32,1536]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|260|Tensor<[1,32]>,<br>Tensor<[1,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|261|Tensor<[1,16,32]>,<br>Tensor<[1,16,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|262|Tensor<[32,4608]>,<br>Tensor<[32,4608]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|263|Tensor<[4608]>,<br>Tensor<[4608]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|264|Tensor<[32,1536]>,<br>Tensor<[32,1536]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|265|Tensor<[32,6144]>,<br>Tensor<[32,6144]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|266|Tensor<[6144]>,<br>Tensor<[6144]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|267|Tensor<[1,32,6144]>,<br>Tensor<[1,32,6144]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|268|Tensor<[1,16,3072]>,<br>Tensor<[1,16,3072]>,<br>|ttnn.multiply|aten::gelu|4|
|269|Tensor<[1,12,16,64]>,<br>Tensor<[1,12,16,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|270|Tensor<[1,12,64,16]>,<br>Tensor<[1,12,64,16]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|271|Tensor<[1,16,768]>,<br>Tensor<[1,16,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|272|Tensor<[16,768]>,<br>Tensor<[16,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|273|Tensor<[16,3072]>,<br>Tensor<[16,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|274|Tensor<[1,64,224,224]>,<br>Tensor<[1,64,224,224]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|275|Tensor<[1,128,112,112]>,<br>Tensor<[1,128,112,112]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|276|Tensor<[30]>,<br>Tensor<[30]>,<br>|ttnn.multiply|aten::arange|4|
|277|Tensor<[60]>,<br>Tensor<[60]>,<br>|ttnn.multiply|aten::arange|4|
|278|Tensor<[80]>,<br>Tensor<[80]>,<br>|ttnn.multiply|aten::arange|4|
|279|Tensor<[120]>,<br>Tensor<[120]>,<br>|ttnn.multiply|aten::arange|4|
|280|Tensor<[240]>,<br>Tensor<[240]>,<br>|ttnn.multiply|aten::arange|4|
|281|Tensor<[480]>,<br>Tensor<[480]>,<br>|ttnn.multiply|aten::arange|4|
|282|Tensor<[1,19200,256]>,<br>Tensor<[1,19200,256]>,<br>|ttnn.multiply|aten::gelu|4|
|283|Tensor<[1,4800,512]>,<br>Tensor<[1,4800,512]>,<br>|ttnn.multiply|aten::gelu|4|
|284|Tensor<[1,1200,1280]>,<br>Tensor<[1,1200,1280]>,<br>|ttnn.multiply|aten::gelu|4|
|285|Tensor<[1,300,2048]>,<br>Tensor<[1,300,2048]>,<br>|ttnn.multiply|aten::gelu|4|
|286|Tensor<[1,19200,64]>,<br>Tensor<[1,19200,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|287|Tensor<[19200,64]>,<br>Tensor<[19200,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|288|Tensor<[1,300,64]>,<br>Tensor<[1,300,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|289|Tensor<[300,64]>,<br>Tensor<[300,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|290|Tensor<[19200,256]>,<br>Tensor<[19200,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|291|Tensor<[1,4800,128]>,<br>Tensor<[1,4800,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|292|Tensor<[4800,128]>,<br>Tensor<[4800,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|293|Tensor<[1,300,128]>,<br>Tensor<[1,300,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|294|Tensor<[300,128]>,<br>Tensor<[300,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|295|Tensor<[4800,512]>,<br>Tensor<[4800,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|296|Tensor<[1,1200,320]>,<br>Tensor<[1,1200,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|297|Tensor<[1200,320]>,<br>Tensor<[1200,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|298|Tensor<[1,300,320]>,<br>Tensor<[1,300,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|299|Tensor<[300,320]>,<br>Tensor<[300,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|300|Tensor<[1200,1280]>,<br>Tensor<[1200,1280]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|301|Tensor<[1,300,512]>,<br>Tensor<[1,300,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|302|Tensor<[300,512]>,<br>Tensor<[300,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|303|Tensor<[300,2048]>,<br>Tensor<[300,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|304|Tensor<[1,64,30,40]>,<br>Tensor<[1,64,30,40]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|305|Tensor<[1,32,30,40]>,<br>Tensor<[1,32,30,40]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|306|Tensor<[1,64,60,80]>,<br>Tensor<[1,64,60,80]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|307|Tensor<[1,32,60,80]>,<br>Tensor<[1,32,60,80]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|308|Tensor<[1,64,120,160]>,<br>Tensor<[1,64,120,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|309|Tensor<[1,32,120,160]>,<br>Tensor<[1,32,120,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|310|Tensor<[1,64,240,320]>,<br>Tensor<[1,64,240,320]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|311|Tensor<[1,64,480,640]>,<br>Tensor<[1,64,480,640]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|312|Tensor<[1,1,480,640]>,<br>Tensor<[1,1,480,640]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|313|Tensor<[1,197,3072]>,<br>Tensor<[1,197,3072]>,<br>|ttnn.multiply|aten::gelu|4|
|314|Tensor<[1,12,197,64]>,<br>Tensor<[1,12,197,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|315|Tensor<[1,12,64,197]>,<br>Tensor<[1,12,64,197]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|316|Tensor<[1,197,768]>,<br>Tensor<[1,197,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|317|Tensor<[197,768]>,<br>Tensor<[197,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|318|Tensor<[197,3072]>,<br>Tensor<[197,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|319|Tensor<[1,16384,128]>,<br>Tensor<[1,16384,128]>,<br>|ttnn.multiply|aten::gelu|4|
|320|Tensor<[1,4096,256]>,<br>Tensor<[1,4096,256]>,<br>|ttnn.multiply|aten::gelu|4|
|321|Tensor<[1,256,1024]>,<br>Tensor<[1,256,1024]>,<br>|ttnn.multiply|aten::gelu|4|
|322|Tensor<[1,16384,32]>,<br>Tensor<[1,16384,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|323|Tensor<[16384,32]>,<br>Tensor<[16384,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|324|Tensor<[1,256,32]>,<br>Tensor<[1,256,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|325|Tensor<[256,32]>,<br>Tensor<[256,32]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|326|Tensor<[16384,128]>,<br>Tensor<[16384,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|327|Tensor<[1,4096,64]>,<br>Tensor<[1,4096,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|328|Tensor<[4096,64]>,<br>Tensor<[4096,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|329|Tensor<[1,256,64]>,<br>Tensor<[1,256,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|330|Tensor<[256,64]>,<br>Tensor<[256,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|331|Tensor<[4096,256]>,<br>Tensor<[4096,256]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|332|Tensor<[1,1024,160]>,<br>Tensor<[1,1024,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|333|Tensor<[1024,160]>,<br>Tensor<[1024,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|334|Tensor<[1,256,160]>,<br>Tensor<[1,256,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|335|Tensor<[256,160]>,<br>Tensor<[256,160]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|336|Tensor<[256,1024]>,<br>Tensor<[256,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|337|Tensor<[1,256,128,128]>,<br>Tensor<[1,256,128,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|338|Tensor<[1,7,18176]>,<br>Tensor<[1,7,18176]>,<br>|ttnn.multiply|aten::gelu|4|
|339|Tensor<[1,71,7,64]>,<br>Tensor<[1,71,7,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|340|Tensor<[1,1,64,7]>,<br>Tensor<[1,1,64,7]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|341|Tensor<[7,7]>,<br>Tensor<[7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|342|Tensor<[1,7,64]>,<br>Tensor<[1,7,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|343|Tensor<[1,7,4544]>,<br>Tensor<[1,7,4544]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|344|Tensor<[1,1,7,64]>,<br>Tensor<[1,1,7,64]>,<br>|ttnn.multiply|aten::mul.Tensor|5|
|345|Tensor<[1,16,112,112]>,<br>Tensor<[1,16,112,112]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|346|Tensor<[96]>,<br>Tensor<[96]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|347|Tensor<[1,96,112,112]>,<br>Tensor<[1,96,112,112]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|348|Tensor<[1,96,56,56]>,<br>Tensor<[1,96,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|349|Tensor<[144]>,<br>Tensor<[144]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|350|Tensor<[1,144,56,56]>,<br>Tensor<[1,144,56,56]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|351|Tensor<[1,144,28,28]>,<br>Tensor<[1,144,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|352|Tensor<[1,32,28,28]>,<br>Tensor<[1,32,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|353|Tensor<[1,192,28,28]>,<br>Tensor<[1,192,28,28]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|354|Tensor<[1,192,14,14]>,<br>Tensor<[1,192,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|355|Tensor<[1,64,14,14]>,<br>Tensor<[1,64,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|356|Tensor<[384]>,<br>Tensor<[384]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|357|Tensor<[1,384,14,14]>,<br>Tensor<[1,384,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|358|Tensor<[1,96,14,14]>,<br>Tensor<[1,96,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|359|Tensor<[576]>,<br>Tensor<[576]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|360|Tensor<[1,576,14,14]>,<br>Tensor<[1,576,14,14]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|361|Tensor<[1,576,7,7]>,<br>Tensor<[1,576,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|362|Tensor<[960]>,<br>Tensor<[960]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|363|Tensor<[1,960,7,7]>,<br>Tensor<[1,960,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|364|Tensor<[1,320,7,7]>,<br>Tensor<[1,320,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|365|Tensor<[1,1280,7,7]>,<br>Tensor<[1,1280,7,7]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|366|Tensor<[1,12,12,64]>,<br>Tensor<[1,12,12,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|367|Tensor<[1,12,64,12]>,<br>Tensor<[1,12,64,12]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|368|Tensor<[1,12,128]>,<br>Tensor<[1,12,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|369|Tensor<[12,768]>,<br>Tensor<[12,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|370|Tensor<[1,12,768]>,<br>Tensor<[1,12,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|371|Tensor<[12,3072]>,<br>Tensor<[12,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|372|Tensor<[1,12,3072]>,<br>Tensor<[1,12,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|373|Tensor<[12,2]>,<br>Tensor<[12,2]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|374|Tensor<[1,12,9,64]>,<br>Tensor<[1,12,9,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|375|Tensor<[1,12,64,9]>,<br>Tensor<[1,12,64,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|376|Tensor<[1,9,128]>,<br>Tensor<[1,9,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|377|Tensor<[9,768]>,<br>Tensor<[9,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|378|Tensor<[1,9,768]>,<br>Tensor<[1,9,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|379|Tensor<[9,3072]>,<br>Tensor<[9,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|380|Tensor<[1,9,3072]>,<br>Tensor<[1,9,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|381|Tensor<[9,128]>,<br>Tensor<[9,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|382|Tensor<[9,30000]>,<br>Tensor<[9,30000]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|383|Tensor<[30000]>,<br>Tensor<[30000]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|384|Tensor<[1,16,9,128]>,<br>Tensor<[1,16,9,128]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|385|Tensor<[1,16,128,9]>,<br>Tensor<[1,16,128,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|386|Tensor<[9,2048]>,<br>Tensor<[9,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|387|Tensor<[1,9,2048]>,<br>Tensor<[1,9,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|388|Tensor<[9,8192]>,<br>Tensor<[9,8192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|389|Tensor<[8192]>,<br>Tensor<[8192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|390|Tensor<[1,9,8192]>,<br>Tensor<[1,9,8192]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|391|Tensor<[1,16,9,64]>,<br>Tensor<[1,16,9,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|392|Tensor<[1,16,64,9]>,<br>Tensor<[1,16,64,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|393|Tensor<[9,1024]>,<br>Tensor<[9,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|394|Tensor<[1,9,1024]>,<br>Tensor<[1,9,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|395|Tensor<[9,4096]>,<br>Tensor<[9,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|396|Tensor<[1,9,4096]>,<br>Tensor<[1,9,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|397|Tensor<[1,64,9,64]>,<br>Tensor<[1,64,9,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|398|Tensor<[1,64,64,9]>,<br>Tensor<[1,64,64,9]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|399|Tensor<[9,16384]>,<br>Tensor<[9,16384]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|400|Tensor<[16384]>,<br>Tensor<[16384]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|401|Tensor<[1,9,16384]>,<br>Tensor<[1,9,16384]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|402|Tensor<[1,2]>,<br>Tensor<[1,2]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|403|Tensor<[1,12,14,64]>,<br>Tensor<[1,12,14,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|404|Tensor<[1,12,64,14]>,<br>Tensor<[1,12,64,14]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|405|Tensor<[1,14,128]>,<br>Tensor<[1,14,128]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|406|Tensor<[14,768]>,<br>Tensor<[14,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|407|Tensor<[1,14,768]>,<br>Tensor<[1,14,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|408|Tensor<[14,3072]>,<br>Tensor<[14,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|409|Tensor<[1,14,3072]>,<br>Tensor<[1,14,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|410|Tensor<[14,2]>,<br>Tensor<[14,2]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|411|Tensor<[1,12,50,64]>,<br>Tensor<[1,12,50,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|412|Tensor<[1,12,64,50]>,<br>Tensor<[1,12,64,50]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|413|Tensor<[2,8,7,64]>,<br>Tensor<[2,8,7,64]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|414|Tensor<[2,8,64,7]>,<br>Tensor<[2,8,64,7]>,<br>|ttnn.multiply|aten::mul.Scalar|4|
|415|Tensor<[1,50,768]>,<br>Tensor<[1,50,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|416|Tensor<[50,768]>,<br>Tensor<[50,768]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|417|Tensor<[50,3072]>,<br>Tensor<[50,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|418|Tensor<[1,50,3072]>,<br>Tensor<[1,50,3072]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|419|Tensor<[2,7,512]>,<br>Tensor<[2,7,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|420|Tensor<[14,512]>,<br>Tensor<[14,512]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|421|Tensor<[14,2048]>,<br>Tensor<[14,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|422|Tensor<[2,7,2048]>,<br>Tensor<[2,7,2048]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|423|Tensor<[2,1]>,<br>Tensor<[2,1]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|424|Tensor<[27]>,<br>Tensor<[27]>,<br>|ttnn.multiply|aten::arange|4|
|425|Tensor<[197]>,<br>Tensor<[197]>,<br>|ttnn.multiply|aten::arange|4|
|426|Tensor<[1,197,4096]>,<br>Tensor<[1,197,4096]>,<br>|ttnn.multiply|aten::gelu|4|
|427|Tensor<[1,197,1024]>,<br>Tensor<[1,197,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|428|Tensor<[197,1024]>,<br>Tensor<[197,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|429|Tensor<[1,16,27,27]>,<br>Tensor<[1,16,27,27]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|430|Tensor<[196,196]>,<br>Tensor<[196,196]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|431|Tensor<[197,4096]>,<br>Tensor<[197,4096]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|432|Tensor<[1,1024]>,<br>Tensor<[1,1024]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|433|Tensor<[1,12,27,27]>,<br>Tensor<[1,12,27,27]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|434|Tensor<[1,64]>,<br>Tensor<[1,64]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|435|Tensor<[1,12]>,<br>Tensor<[1,12]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|436|Tensor<[1,784]>,<br>Tensor<[1,784]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
|437|Tensor<[784]>,<br>Tensor<[784]>,<br>|ttnn.multiply|aten::mul.Tensor|4|
