
### stablehlo.dot_general::ttnn.matmul


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,64,1]>,Tensor<[1,1,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|1|Tensor<[32,32,128]>,Tensor<[32,128,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|2|Tensor<[32,32,32]>,Tensor<[32,32,128]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|3|Tensor<[32,4096]>,Tensor<[4096,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|4|Tensor<[32,4096]>,Tensor<[4096,11008]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|5|Tensor<[32,11008]>,Tensor<[11008,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|6|Tensor<[32,4096]>,Tensor<[4096,32000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|7|Tensor<[12,7,64]>,Tensor<[12,64,7]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|8|Tensor<[12,7,7]>,Tensor<[12,7,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|9|Tensor<[7,768]>,Tensor<[768,2304]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|10|Tensor<[7,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|11|Tensor<[7,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|12|Tensor<[7,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|13|Tensor<[7,768]>,Tensor<[768,2]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|14|Tensor<[256,768]>,Tensor<[768,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|15|Tensor<[256,512]>,Tensor<[512,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|16|Tensor<[256,256]>,Tensor<[256,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|17|Tensor<[1,512]>,Tensor<[512,1000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|18|Tensor<[8,920,32]>,Tensor<[8,32,920]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::baddbmm|4|
|19|Tensor<[8,100,32]>,Tensor<[8,32,920]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::baddbmm|4|
|20|Tensor<[920,1,256]>,Tensor<[920,256,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|21|Tensor<[8,920,920]>,Tensor<[8,920,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|22|Tensor<[8,100,32]>,Tensor<[8,32,100]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|23|Tensor<[8,100,100]>,Tensor<[8,100,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|24|Tensor<[8,100,920]>,Tensor<[8,920,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|25|Tensor<[6,100,256]>,Tensor<[6,256,92]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|26|Tensor<[6,100,256]>,Tensor<[6,256,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|27|Tensor<[920,256]>,Tensor<[256,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|28|Tensor<[920,256]>,Tensor<[256,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|29|Tensor<[920,2048]>,Tensor<[2048,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|30|Tensor<[100,256]>,Tensor<[256,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|31|Tensor<[100,256]>,Tensor<[256,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|32|Tensor<[100,2048]>,Tensor<[2048,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|33|Tensor<[600,256]>,Tensor<[256,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|34|Tensor<[600,256]>,Tensor<[256,4]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|35|Tensor<[12,10,64]>,Tensor<[12,64,10]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|36|Tensor<[12,10,10]>,Tensor<[12,10,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|37|Tensor<[10,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|38|Tensor<[10,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|39|Tensor<[10,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|40|Tensor<[10,768]>,Tensor<[768,250002]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|41|Tensor<[8,4096,40]>,Tensor<[8,40,4096]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|42|Tensor<[8,4096,4096]>,Tensor<[8,4096,40]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|43|Tensor<[8,4096,40]>,Tensor<[8,40,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|44|Tensor<[8,4096,9]>,Tensor<[8,9,40]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|45|Tensor<[8,1024,80]>,Tensor<[8,80,1024]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|46|Tensor<[8,1024,1024]>,Tensor<[8,1024,80]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|47|Tensor<[8,1024,80]>,Tensor<[8,80,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|48|Tensor<[8,1024,9]>,Tensor<[8,9,80]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|49|Tensor<[8,256,160]>,Tensor<[8,160,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|50|Tensor<[8,256,256]>,Tensor<[8,256,160]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|51|Tensor<[8,256,160]>,Tensor<[8,160,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|52|Tensor<[8,256,9]>,Tensor<[8,9,160]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|53|Tensor<[8,64,160]>,Tensor<[8,160,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|54|Tensor<[8,64,64]>,Tensor<[8,64,160]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|55|Tensor<[8,64,160]>,Tensor<[8,160,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|56|Tensor<[8,64,9]>,Tensor<[8,9,160]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|57|Tensor<[1,320]>,Tensor<[320,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|58|Tensor<[1,1280]>,Tensor<[1280,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|59|Tensor<[1,1280]>,Tensor<[1280,320]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|60|Tensor<[4096,320]>,Tensor<[320,320]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|61|Tensor<[9,768]>,Tensor<[768,320]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|62|Tensor<[4096,320]>,Tensor<[320,2560]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|63|Tensor<[4096,1280]>,Tensor<[1280,320]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|64|Tensor<[1,1280]>,Tensor<[1280,640]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|65|Tensor<[1024,640]>,Tensor<[640,640]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|66|Tensor<[9,768]>,Tensor<[768,640]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|67|Tensor<[1024,640]>,Tensor<[640,5120]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|68|Tensor<[1024,2560]>,Tensor<[2560,640]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|69|Tensor<[256,1280]>,Tensor<[1280,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|70|Tensor<[9,768]>,Tensor<[768,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|71|Tensor<[256,1280]>,Tensor<[1280,10240]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|72|Tensor<[256,5120]>,Tensor<[5120,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|73|Tensor<[64,1280]>,Tensor<[1280,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|74|Tensor<[64,1280]>,Tensor<[1280,10240]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|75|Tensor<[64,5120]>,Tensor<[5120,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|76|Tensor<[12,25,64]>,Tensor<[12,64,25]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|77|Tensor<[12,25,25]>,Tensor<[12,25,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|78|Tensor<[25,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|79|Tensor<[25,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|80|Tensor<[25,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|81|Tensor<[25,768]>,Tensor<[768,2]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|82|Tensor<[1,768]>,Tensor<[768,1]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|83|Tensor<[3,1445,64]>,Tensor<[3,64,1445]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|84|Tensor<[3,1445,1445]>,Tensor<[3,1445,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|85|Tensor<[1445,192]>,Tensor<[192,192]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|86|Tensor<[1445,192]>,Tensor<[192,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|87|Tensor<[1445,768]>,Tensor<[768,192]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|88|Tensor<[100,192]>,Tensor<[192,192]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|89|Tensor<[100,192]>,Tensor<[192,92]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|90|Tensor<[100,192]>,Tensor<[192,4]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|91|Tensor<[12,8,64]>,Tensor<[12,64,8]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|92|Tensor<[12,8,8]>,Tensor<[12,8,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|93|Tensor<[1,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|94|Tensor<[1,768]>,Tensor<[768,3]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|95|Tensor<[8,256,32]>,Tensor<[8,32,2048]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|96|Tensor<[8,256,2048]>,Tensor<[8,2048,160]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|97|Tensor<[8,256,32]>,Tensor<[8,32,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|98|Tensor<[8,2048,32]>,Tensor<[8,32,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|99|Tensor<[8,2048,256]>,Tensor<[8,256,96]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|100|Tensor<[256,1280]>,Tensor<[1280,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|101|Tensor<[2048,768]>,Tensor<[768,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|102|Tensor<[2048,768]>,Tensor<[768,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|103|Tensor<[256,1280]>,Tensor<[1280,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|104|Tensor<[2048,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|105|Tensor<[2048,768]>,Tensor<[768,262]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|106|Tensor<[1,2048]>,Tensor<[2048,1000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|107|Tensor<[12,201,64]>,Tensor<[12,64,201]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|108|Tensor<[12,201,201]>,Tensor<[12,201,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|109|Tensor<[201,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|110|Tensor<[201,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|111|Tensor<[201,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|112|Tensor<[1,768]>,Tensor<[768,1536]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|113|Tensor<[1,1536]>,Tensor<[1536,3129]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|114|Tensor<[1,9216]>,Tensor<[9216,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|115|Tensor<[1,128]>,Tensor<[128,10]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|116|Tensor<[16,19,64]>,Tensor<[16,64,19]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|117|Tensor<[16,19,19]>,Tensor<[16,19,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|118|Tensor<[19,1024]>,Tensor<[1024,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|119|Tensor<[19,1024]>,Tensor<[1024,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|120|Tensor<[19,4096]>,Tensor<[4096,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|121|Tensor<[19,1024]>,Tensor<[1024,256008]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|122|Tensor<[1,1024]>,Tensor<[1024,1000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|123|Tensor<[16,32,96]>,Tensor<[16,96,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::baddbmm|4|
|124|Tensor<[16,32,32]>,Tensor<[16,32,96]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|125|Tensor<[32,1536]>,Tensor<[1536,4608]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|126|Tensor<[32,1536]>,Tensor<[1536,1536]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|127|Tensor<[32,1536]>,Tensor<[1536,6144]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|128|Tensor<[32,6144]>,Tensor<[6144,1536]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|129|Tensor<[32,1536]>,Tensor<[1536,250880]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|130|Tensor<[12,16,64]>,Tensor<[12,64,16]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|131|Tensor<[12,16,16]>,Tensor<[12,16,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|132|Tensor<[16,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|133|Tensor<[16,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|134|Tensor<[16,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|135|Tensor<[1,19200,64]>,Tensor<[1,64,300]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|136|Tensor<[1,19200,300]>,Tensor<[1,300,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|137|Tensor<[1,19200,256]>,Tensor<[1,256,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|138|Tensor<[2,4800,64]>,Tensor<[2,64,300]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|139|Tensor<[2,4800,300]>,Tensor<[2,300,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|140|Tensor<[1,4800,512]>,Tensor<[1,512,128]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|141|Tensor<[5,1200,64]>,Tensor<[5,64,300]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|142|Tensor<[5,1200,300]>,Tensor<[5,300,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|143|Tensor<[1,1200,1280]>,Tensor<[1,1280,320]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|144|Tensor<[8,300,64]>,Tensor<[8,64,300]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|145|Tensor<[8,300,300]>,Tensor<[8,300,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|146|Tensor<[1,300,2048]>,Tensor<[1,2048,512]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|147|Tensor<[19200,64]>,Tensor<[64,64]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|148|Tensor<[300,64]>,Tensor<[64,64]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|149|Tensor<[19200,64]>,Tensor<[64,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|150|Tensor<[4800,128]>,Tensor<[128,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|151|Tensor<[300,128]>,Tensor<[128,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|152|Tensor<[4800,128]>,Tensor<[128,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|153|Tensor<[1200,320]>,Tensor<[320,320]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|154|Tensor<[300,320]>,Tensor<[320,320]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|155|Tensor<[1200,320]>,Tensor<[320,1280]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|156|Tensor<[300,512]>,Tensor<[512,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|157|Tensor<[300,512]>,Tensor<[512,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|158|Tensor<[12,197,64]>,Tensor<[12,64,197]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|159|Tensor<[12,197,197]>,Tensor<[12,197,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|160|Tensor<[197,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|161|Tensor<[197,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|162|Tensor<[197,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|163|Tensor<[1,768]>,Tensor<[768,1000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|164|Tensor<[1,16384,32]>,Tensor<[1,32,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|165|Tensor<[1,16384,256]>,Tensor<[1,256,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|166|Tensor<[1,16384,128]>,Tensor<[1,128,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|167|Tensor<[2,4096,32]>,Tensor<[2,32,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|168|Tensor<[2,4096,256]>,Tensor<[2,256,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|169|Tensor<[1,4096,256]>,Tensor<[1,256,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|170|Tensor<[5,1024,32]>,Tensor<[5,32,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|171|Tensor<[5,1024,256]>,Tensor<[5,256,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|172|Tensor<[1,1024,640]>,Tensor<[1,640,160]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|173|Tensor<[8,256,256]>,Tensor<[8,256,32]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|174|Tensor<[1,256,1024]>,Tensor<[1,1024,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|175|Tensor<[1,4096,64]>,Tensor<[1,64,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|176|Tensor<[1,1024,160]>,Tensor<[1,160,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|177|Tensor<[1,256,256]>,Tensor<[1,256,256]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|178|Tensor<[16384,32]>,Tensor<[32,32]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|179|Tensor<[256,32]>,Tensor<[32,32]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|180|Tensor<[16384,32]>,Tensor<[32,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|181|Tensor<[4096,64]>,Tensor<[64,64]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|182|Tensor<[256,64]>,Tensor<[64,64]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|183|Tensor<[4096,64]>,Tensor<[64,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|184|Tensor<[1024,160]>,Tensor<[160,160]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|185|Tensor<[256,160]>,Tensor<[160,160]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|186|Tensor<[1024,160]>,Tensor<[160,640]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|187|Tensor<[256,256]>,Tensor<[256,256]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|188|Tensor<[256,256]>,Tensor<[256,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|189|Tensor<[1,32,1]>,Tensor<[1,1,7]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|190|Tensor<[71,7,64]>,Tensor<[71,64,7]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|191|Tensor<[71,7,7]>,Tensor<[71,7,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|192|Tensor<[7,4544]>,Tensor<[4544,4672]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|193|Tensor<[7,4544]>,Tensor<[4544,4544]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|194|Tensor<[7,4544]>,Tensor<[4544,18176]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|195|Tensor<[7,18176]>,Tensor<[18176,4544]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|196|Tensor<[7,4544]>,Tensor<[4544,65024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|197|Tensor<[1,1280]>,Tensor<[1280,1000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|198|Tensor<[12,12,64]>,Tensor<[12,64,12]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|199|Tensor<[12,12,12]>,Tensor<[12,12,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|200|Tensor<[12,128]>,Tensor<[128,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|201|Tensor<[12,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|202|Tensor<[12,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|203|Tensor<[12,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|204|Tensor<[12,768]>,Tensor<[768,2]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|205|Tensor<[12,9,64]>,Tensor<[12,64,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|206|Tensor<[12,9,9]>,Tensor<[12,9,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|207|Tensor<[9,128]>,Tensor<[128,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|208|Tensor<[9,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|209|Tensor<[9,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|210|Tensor<[9,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|211|Tensor<[9,768]>,Tensor<[768,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|212|Tensor<[9,128]>,Tensor<[128,30000]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|213|Tensor<[16,9,128]>,Tensor<[16,128,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|214|Tensor<[16,9,9]>,Tensor<[16,9,128]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|215|Tensor<[9,128]>,Tensor<[128,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|216|Tensor<[9,2048]>,Tensor<[2048,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|217|Tensor<[9,2048]>,Tensor<[2048,8192]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|218|Tensor<[9,8192]>,Tensor<[8192,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|219|Tensor<[9,2048]>,Tensor<[2048,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|220|Tensor<[16,9,64]>,Tensor<[16,64,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|221|Tensor<[16,9,9]>,Tensor<[16,9,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|222|Tensor<[9,128]>,Tensor<[128,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|223|Tensor<[9,1024]>,Tensor<[1024,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|224|Tensor<[9,1024]>,Tensor<[1024,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|225|Tensor<[9,4096]>,Tensor<[4096,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|226|Tensor<[9,1024]>,Tensor<[1024,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|227|Tensor<[64,9,64]>,Tensor<[64,64,9]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|228|Tensor<[64,9,9]>,Tensor<[64,9,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|229|Tensor<[9,128]>,Tensor<[128,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|230|Tensor<[9,4096]>,Tensor<[4096,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|231|Tensor<[9,4096]>,Tensor<[4096,16384]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|232|Tensor<[9,16384]>,Tensor<[16384,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|233|Tensor<[9,4096]>,Tensor<[4096,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|234|Tensor<[1,768]>,Tensor<[768,2]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|235|Tensor<[12,14,64]>,Tensor<[12,64,14]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|236|Tensor<[12,14,14]>,Tensor<[12,14,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|237|Tensor<[14,128]>,Tensor<[128,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|238|Tensor<[14,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|239|Tensor<[14,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|240|Tensor<[14,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|241|Tensor<[14,768]>,Tensor<[768,2]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|242|Tensor<[12,50,64]>,Tensor<[12,64,50]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|243|Tensor<[12,50,50]>,Tensor<[12,50,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|244|Tensor<[16,7,64]>,Tensor<[16,64,7]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|245|Tensor<[16,7,7]>,Tensor<[16,7,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|246|Tensor<[50,768]>,Tensor<[768,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|247|Tensor<[50,768]>,Tensor<[768,3072]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|248|Tensor<[50,3072]>,Tensor<[3072,768]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|249|Tensor<[14,512]>,Tensor<[512,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|250|Tensor<[14,512]>,Tensor<[512,2048]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|251|Tensor<[14,2048]>,Tensor<[2048,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|252|Tensor<[1,768]>,Tensor<[768,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|253|Tensor<[2,512]>,Tensor<[512,512]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|254|Tensor<[2,512]>,Tensor<[512,1]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|255|Tensor<[16,197,64]>,Tensor<[16,64,197]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|256|Tensor<[16,197,197]>,Tensor<[16,197,64]>,batching_dims: [0] x [0]contracting_dims: [2] x [1]|ttnn.matmul|aten::bmm|4|
|257|Tensor<[197,1024]>,Tensor<[1024,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|258|Tensor<[197,1024]>,Tensor<[1024,4096]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|259|Tensor<[197,4096]>,Tensor<[4096,1024]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|260|Tensor<[1,784]>,Tensor<[784,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|261|Tensor<[1,128]>,Tensor<[128,64]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|262|Tensor<[1,64]>,Tensor<[64,12]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|263|Tensor<[1,12]>,Tensor<[12,3]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|264|Tensor<[1,3]>,Tensor<[3,12]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|265|Tensor<[1,12]>,Tensor<[12,64]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|266|Tensor<[1,64]>,Tensor<[64,128]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
|267|Tensor<[1,128]>,Tensor<[128,784]>,contracting_dims: [1] x [0]|ttnn.matmul|aten::mm|5|
