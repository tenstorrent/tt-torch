
### stablehlo.broadcast_in_dim


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,32,32,32]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|1|Tensor<[1,32,32,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|2|Scalar,dims: []||aten::_safe_softmax|4|
|3|Tensor<[1,1,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|4|Tensor<[1,1,1,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|5|Tensor<[1,32,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|6|Tensor<[32]>,dims: [0]||aten::arange|4|
|7|Tensor<[1,1,32]>,dims: [0, 1, 2]||aten::bmm|4|
|8|Tensor<[32,128,32]>,dims: [0, 1, 2]||aten::bmm|4|
|9|Tensor<[32,32,128]>,dims: [0, 1, 2]||aten::bmm|4|
|10|Tensor<[32]>,dims: [1]||aten::gt.Tensor|4|
|11|Tensor<[32,1]>,dims: [0, 1]||aten::gt.Tensor|4|
|12|Tensor<[1,32,32,128]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|13|Tensor<[1,32,128,32]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|14|Tensor<[1,32,128]>,dims: [0, 1, 2]||aten::mul.Tensor|4|
|15|Tensor<[1,32,4096]>,dims: [0, 1, 2]||aten::mul.Tensor|4|
|16|Tensor<[4096]>,dims: [2]||aten::mul.Tensor|4|
|17|Tensor<[1,1,32,128]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|18|Tensor<[1,32]>,dims: [0, 1]||aten::triu|4|
|19|Tensor<[32,32]>,dims: [0, 1]||aten::triu|4|
|20|Tensor<[1,12,7,7]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|21|Tensor<[1,12,7,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|22|Tensor<[7]>,dims: [0]||aten::add.Tensor|4|
|23|Tensor<[1,7,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|24|Tensor<[1,7,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|25|Tensor<[768]>,dims: [2]||aten::add.Tensor|4|
|26|Tensor<[7,2304]>,dims: [0, 1]||aten::add.Tensor|4|
|27|Tensor<[2304]>,dims: [1]||aten::add.Tensor|4|
|28|Tensor<[1,1,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|29|Tensor<[7,768]>,dims: [0, 1]||aten::add.Tensor|4|
|30|Tensor<[768]>,dims: [1]||aten::add.Tensor|4|
|31|Tensor<[7,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|32|Tensor<[3072]>,dims: [1]||aten::add.Tensor|4|
|33|Tensor<[1,7,3072]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|34|Tensor<[1]>,dims: [0]||aten::arange|4|
|35|Tensor<[12,64,7]>,dims: [0, 1, 2]||aten::bmm|4|
|36|Tensor<[12,7,64]>,dims: [0, 1, 2]||aten::bmm|4|
|37|Tensor<[1,7]>,dims: [0, 1]||aten::eq.Scalar|4|
|38|Tensor<[1,1,1,7]>,dims: [0, 1, 2, 3]||aten::expand|4|
|39|Tensor<[7]>,dims: [1]||aten::lt.Tensor|4|
|40|Tensor<[7,1]>,dims: [0, 1]||aten::lt.Tensor|4|
|41|Tensor<[1,12,7,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|42|Tensor<[1,12,64,7]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|43|Tensor<[2304]>,dims: [0]||aten::mul.Tensor|4|
|44|Tensor<[768]>,dims: [0]||aten::mul.Tensor|4|
|45|Tensor<[3072]>,dims: [0]||aten::mul.Tensor|4|
|46|Tensor<[7,7]>,dims: [0, 1]||aten::where.self|4|
|47|Tensor<[1,32,112,112]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|48|Tensor<[32,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|49|Tensor<[64]>,dims: [0]||aten::add.Tensor|4|
|50|Tensor<[1,64,112,112]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|51|Tensor<[64,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|52|Tensor<[1,64,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|53|Tensor<[128]>,dims: [0]||aten::add.Tensor|4|
|54|Tensor<[1,128,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|55|Tensor<[128,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|56|Tensor<[1,128,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|57|Tensor<[256]>,dims: [0]||aten::add.Tensor|4|
|58|Tensor<[1,256,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|59|Tensor<[256,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|60|Tensor<[512]>,dims: [0]||aten::add.Tensor|4|
|61|Tensor<[1,512,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|62|Tensor<[512,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|63|Tensor<[1,19,28,28]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|64|Tensor<[19,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|65|Tensor<[1,38,28,28]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|66|Tensor<[38,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|67|Tensor<[256,512]>,dims: [0, 1]||aten::add.Tensor|4|
|68|Tensor<[512]>,dims: [1]||aten::add.Tensor|4|
|69|Tensor<[1,256,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|70|Tensor<[1,256,512]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|71|Tensor<[512]>,dims: [2]||aten::add.Tensor|4|
|72|Tensor<[256,256]>,dims: [0, 1]||aten::add.Tensor|4|
|73|Tensor<[256]>,dims: [1]||aten::add.Tensor|4|
|74|Tensor<[1,1000]>,dims: [0, 1]||aten::add.Tensor|4|
|75|Tensor<[1000]>,dims: [1]||aten::add.Tensor|4|
|76|Tensor<[1,1024,512]>,dims: [0, 1, 2]||aten::convolution|4|
|77|Tensor<[1024,1]>,dims: [1, 2]||aten::convolution|4|
|78|Tensor<[256,1]>,dims: [1, 2]||aten::convolution|4|
|79|Tensor<[1,512]>,dims: [0, 1]||aten::mean.dim|4|
|80|Tensor<[1000]>,dims: [0]||aten::mul.Tensor|4|
|81|Tensor<[8,920,920]>,dims: [0, 1, 2]||aten::_softmax|4|
|82|Tensor<[8,920,1]>,dims: [0, 1, 2]||aten::_softmax|4|
|83|Tensor<[8,100,100]>,dims: [0, 1, 2]||aten::_softmax|4|
|84|Tensor<[8,100,1]>,dims: [0, 1, 2]||aten::_softmax|4|
|85|Tensor<[8,100,920]>,dims: [0, 1, 2]||aten::_softmax|4|
|86|Tensor<[1,64,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|87|Tensor<[1,64,360,640]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|88|Tensor<[1,64,180,320]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|89|Tensor<[1,256,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|90|Tensor<[1,256,180,320]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|91|Tensor<[1,128,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|92|Tensor<[1,128,180,320]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|93|Tensor<[1,128,90,160]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|94|Tensor<[1,512,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|95|Tensor<[1,512,90,160]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|96|Tensor<[1,256,90,160]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|97|Tensor<[1,256,45,80]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|98|Tensor<[1,1024,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|99|Tensor<[1,1024,45,80]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|100|Tensor<[1,512,45,80]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|101|Tensor<[1,512,23,40]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|102|Tensor<[1,2048,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|103|Tensor<[1,2048,23,40]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|104|Tensor<[23]>,dims: [0]||aten::add.Tensor|4|
|105|Tensor<[40]>,dims: [0]||aten::add.Tensor|4|
|106|Tensor<[1,1,40]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|107|Tensor<[1,23,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|108|Tensor<[920,1,256]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|109|Tensor<[256]>,dims: [2]||aten::add.Tensor|4|
|110|Tensor<[920,256]>,dims: [0, 1]||aten::add.Tensor|4|
|111|Tensor<[920,1,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|112|Tensor<[920,2048]>,dims: [0, 1]||aten::add.Tensor|4|
|113|Tensor<[2048]>,dims: [1]||aten::add.Tensor|4|
|114|Tensor<[100,256]>,dims: [0, 1]||aten::add.Tensor|4|
|115|Tensor<[100,1,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|116|Tensor<[100,1,256]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|117|Tensor<[100,2048]>,dims: [0, 1]||aten::add.Tensor|4|
|118|Tensor<[6,1,100,92]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|119|Tensor<[92]>,dims: [3]||aten::add.Tensor|4|
|120|Tensor<[6,1,100,256]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|121|Tensor<[256]>,dims: [3]||aten::add.Tensor|4|
|122|Tensor<[6,1,100,4]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|123|Tensor<[4]>,dims: [3]||aten::add.Tensor|4|
|124|Tensor<[8,32,920]>,dims: [0, 1, 2]||aten::baddbmm|4|
|125|Tensor<[8,1,920]>,dims: [0, 1, 2]||aten::baddbmm|4|
|126|Tensor<[920,256,256]>,dims: [0, 1, 2]||aten::bmm|4|
|127|Tensor<[8,920,32]>,dims: [0, 1, 2]||aten::bmm|4|
|128|Tensor<[8,32,100]>,dims: [0, 1, 2]||aten::bmm|4|
|129|Tensor<[8,100,32]>,dims: [0, 1, 2]||aten::bmm|4|
|130|Tensor<[6,256,92]>,dims: [0, 1, 2]||aten::bmm|4|
|131|Tensor<[6,256,256]>,dims: [0, 1, 2]||aten::bmm|4|
|132|Tensor<[1,256,23,40]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|133|Tensor<[1,23,40]>,dims: [0, 1, 2]||aten::div.Tensor|4|
|134|Tensor<[1,23,40,1]>,dims: [0, 1, 2, 3]||aten::div.Tensor|4|
|135|Tensor<[128]>,dims: [3]||aten::div.Tensor|4|
|136|Tensor<[256,256]>,dims: [1, 2]||aten::expand|5|
|137|Tensor<[1,1,1,920]>,dims: [0, 1, 2, 3]||aten::expand|5|
|138|Tensor<[256,92]>,dims: [2, 3]||aten::expand|5|
|139|Tensor<[256,256]>,dims: [2, 3]||aten::expand|5|
|140|Tensor<[1,1,1,1]>,dims: [0, 1, 2, 3]||aten::index.Tensor|4|
|141|Tensor<[1,1,1]>,dims: [1, 2, 3]||aten::index.Tensor|4|
|142|Tensor<[23,1]>,dims: [2, 3]||aten::index.Tensor|4|
|143|Tensor<[40]>,dims: [3]||aten::index.Tensor|4|
|144|Tensor<[2048]>,dims: [0]||aten::mul.Tensor|4|
|145|Tensor<[1,920]>,dims: [0, 1]||aten::where.self|4|
|146|Tensor<[1,12,10,10]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|147|Tensor<[1,12,10,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|148|Tensor<[1,10]>,dims: [0, 1]||aten::add.Tensor|5|
|149|Tensor<[1,10,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|150|Tensor<[1,10,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|151|Tensor<[10,768]>,dims: [0, 1]||aten::add.Tensor|4|
|152|Tensor<[1,1,10,10]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|153|Tensor<[10,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|154|Tensor<[10,250002]>,dims: [0, 1]||aten::add.Tensor|4|
|155|Tensor<[250002]>,dims: [1]||aten::add.Tensor|4|
|156|Tensor<[12,64,10]>,dims: [0, 1, 2]||aten::bmm|4|
|157|Tensor<[12,10,64]>,dims: [0, 1, 2]||aten::bmm|4|
|158|Tensor<[1,1,1,10]>,dims: [0, 1, 2, 3]||aten::expand|4|
|159|Tensor<[1,12,10,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|160|Tensor<[1,12,64,10]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|161|Tensor<[250002]>,dims: [0]||aten::mul.Tensor|4|
|162|Tensor<[1,8,4096,4096]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|163|Tensor<[1,8,4096,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|164|Tensor<[1,8,4096,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|165|Tensor<[1,8,1024,1024]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|166|Tensor<[1,8,1024,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|167|Tensor<[1,8,1024,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|168|Tensor<[1,8,256,256]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|169|Tensor<[1,8,256,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|170|Tensor<[1,8,256,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|171|Tensor<[1,8,64,64]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|172|Tensor<[1,8,64,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|173|Tensor<[1,8,64,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|174|Tensor<[1,1280]>,dims: [0, 1]||aten::add.Tensor|4|
|175|Tensor<[1280]>,dims: [1]||aten::add.Tensor|4|
|176|Tensor<[1,32,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|177|Tensor<[1,320,64,64]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|178|Tensor<[1,320,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|179|Tensor<[1,320]>,dims: [0, 1]||aten::add.Tensor|4|
|180|Tensor<[320]>,dims: [1]||aten::add.Tensor|4|
|181|Tensor<[1,4096,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|182|Tensor<[1,4096,320]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|183|Tensor<[320]>,dims: [2]||aten::add.Tensor|4|
|184|Tensor<[4096,320]>,dims: [0, 1]||aten::add.Tensor|4|
|185|Tensor<[4096,2560]>,dims: [0, 1]||aten::add.Tensor|4|
|186|Tensor<[2560]>,dims: [1]||aten::add.Tensor|4|
|187|Tensor<[1,320,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|188|Tensor<[1,640]>,dims: [0, 1]||aten::add.Tensor|4|
|189|Tensor<[640]>,dims: [1]||aten::add.Tensor|4|
|190|Tensor<[1,640,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|191|Tensor<[1,640,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|192|Tensor<[1,1024,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|193|Tensor<[1,1024,640]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|194|Tensor<[640]>,dims: [2]||aten::add.Tensor|4|
|195|Tensor<[1024,640]>,dims: [0, 1]||aten::add.Tensor|4|
|196|Tensor<[1024,5120]>,dims: [0, 1]||aten::add.Tensor|4|
|197|Tensor<[5120]>,dims: [1]||aten::add.Tensor|4|
|198|Tensor<[1,640,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|199|Tensor<[1,1280,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|200|Tensor<[1,1280,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|201|Tensor<[1,256,1280]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|202|Tensor<[1280]>,dims: [2]||aten::add.Tensor|4|
|203|Tensor<[256,1280]>,dims: [0, 1]||aten::add.Tensor|4|
|204|Tensor<[256,10240]>,dims: [0, 1]||aten::add.Tensor|4|
|205|Tensor<[10240]>,dims: [1]||aten::add.Tensor|4|
|206|Tensor<[1,1280,8,8]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|207|Tensor<[1,64,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|208|Tensor<[1,64,1280]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|209|Tensor<[64,1280]>,dims: [0, 1]||aten::add.Tensor|4|
|210|Tensor<[64,10240]>,dims: [0, 1]||aten::add.Tensor|4|
|211|Tensor<[1,2560,8,8]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|212|Tensor<[1,2560,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|213|Tensor<[16]>,dims: [0]||aten::add.Tensor|4|
|214|Tensor<[1,2560,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|215|Tensor<[1,1920,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|216|Tensor<[1,1920,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|217|Tensor<[1,1920,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|218|Tensor<[1,1280,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|219|Tensor<[1,960,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|220|Tensor<[1,960,1,1]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|221|Tensor<[1,960,64,64]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|222|Tensor<[1,640,64,64]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|223|Tensor<[160]>,dims: [0]||aten::arange.start|4|
|224|Tensor<[8,40,4096]>,dims: [0, 1, 2]||aten::bmm|4|
|225|Tensor<[8,4096,40]>,dims: [0, 1, 2]||aten::bmm|4|
|226|Tensor<[8,40,9]>,dims: [0, 1, 2]||aten::bmm|4|
|227|Tensor<[8,9,40]>,dims: [0, 1, 2]||aten::bmm|4|
|228|Tensor<[8,80,1024]>,dims: [0, 1, 2]||aten::bmm|4|
|229|Tensor<[8,1024,80]>,dims: [0, 1, 2]||aten::bmm|4|
|230|Tensor<[8,80,9]>,dims: [0, 1, 2]||aten::bmm|4|
|231|Tensor<[8,9,80]>,dims: [0, 1, 2]||aten::bmm|4|
|232|Tensor<[8,160,256]>,dims: [0, 1, 2]||aten::bmm|4|
|233|Tensor<[8,256,160]>,dims: [0, 1, 2]||aten::bmm|4|
|234|Tensor<[8,160,9]>,dims: [0, 1, 2]||aten::bmm|4|
|235|Tensor<[8,9,160]>,dims: [0, 1, 2]||aten::bmm|4|
|236|Tensor<[8,160,64]>,dims: [0, 1, 2]||aten::bmm|4|
|237|Tensor<[8,64,160]>,dims: [0, 1, 2]||aten::bmm|4|
|238|Tensor<[320,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|239|Tensor<[640,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|240|Tensor<[1280,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|241|Tensor<[1,4,64,64]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|242|Tensor<[4,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|243|Tensor<[1280]>,dims: [0]||aten::index.Tensor|4|
|244|Tensor<[16,1]>,dims: [2, 3]||aten::index.Tensor|4|
|245|Tensor<[16]>,dims: [3]||aten::index.Tensor|4|
|246|Tensor<[32,1]>,dims: [2, 3]||aten::index.Tensor|4|
|247|Tensor<[32]>,dims: [3]||aten::index.Tensor|4|
|248|Tensor<[640]>,dims: [0]||aten::index.Tensor|4|
|249|Tensor<[64,1]>,dims: [2, 3]||aten::index.Tensor|4|
|250|Tensor<[64]>,dims: [3]||aten::index.Tensor|4|
|251|Tensor<[1,8,4096,40]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|252|Tensor<[1,8,40,4096]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|253|Tensor<[1,8,40,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|254|Tensor<[1,8,1024,80]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|255|Tensor<[1,8,80,1024]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|256|Tensor<[1,8,80,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|257|Tensor<[1,8,256,160]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|258|Tensor<[1,8,160,256]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|259|Tensor<[1,8,160,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|260|Tensor<[1,8,64,160]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|261|Tensor<[1,8,160,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|262|Tensor<[1,1]>,dims: [0, 1]||aten::mul.Tensor|4|
|263|Tensor<[1,160]>,dims: [0, 1]||aten::mul.Tensor|4|
|264|Tensor<[1,32,10,4096]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|265|Tensor<[320]>,dims: [0]||aten::mul.Tensor|4|
|266|Tensor<[2560]>,dims: [0]||aten::mul.Tensor|4|
|267|Tensor<[1,32,10,1024]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|268|Tensor<[1,32,20,1024]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|269|Tensor<[5120]>,dims: [0]||aten::mul.Tensor|4|
|270|Tensor<[1,32,20,256]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|271|Tensor<[1,32,40,256]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|272|Tensor<[10240]>,dims: [0]||aten::mul.Tensor|4|
|273|Tensor<[1,32,40,64]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|274|Tensor<[1,32,80,64]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|275|Tensor<[1,32,80,256]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|276|Tensor<[1,32,60,256]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|277|Tensor<[1,32,60,1024]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|278|Tensor<[1,32,40,1024]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|279|Tensor<[1,32,30,1024]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|280|Tensor<[1,32,30,4096]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|281|Tensor<[1,32,20,4096]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|282|Tensor<[1,12,25,25]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|283|Tensor<[1,12,25,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|284|Tensor<[1,25,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|285|Tensor<[1,25,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|286|Tensor<[25,768]>,dims: [0, 1]||aten::add.Tensor|4|
|287|Tensor<[1,1,25,25]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|288|Tensor<[25,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|289|Tensor<[25,2]>,dims: [0, 1]||aten::add.Tensor|4|
|290|Tensor<[2]>,dims: [1]||aten::add.Tensor|4|
|291|Tensor<[1]>,dims: [1]||aten::add.Tensor|4|
|292|Tensor<[12,64,25]>,dims: [0, 1, 2]||aten::bmm|4|
|293|Tensor<[12,25,64]>,dims: [0, 1, 2]||aten::bmm|4|
|294|Tensor<[1,1,1,25]>,dims: [0, 1, 2, 3]||aten::expand|4|
|295|Tensor<[1,12,25,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|296|Tensor<[1,12,64,25]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|297|Tensor<[2]>,dims: [0]||aten::mul.Tensor|4|
|298|Tensor<[1,3,1445,1445]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|299|Tensor<[1,3,1445,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|300|Tensor<[1,1445,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|301|Tensor<[1,1445,192]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|302|Tensor<[192]>,dims: [2]||aten::add.Tensor|4|
|303|Tensor<[1445,192]>,dims: [0, 1]||aten::add.Tensor|4|
|304|Tensor<[192]>,dims: [1]||aten::add.Tensor|4|
|305|Tensor<[1445,768]>,dims: [0, 1]||aten::add.Tensor|4|
|306|Tensor<[100,192]>,dims: [0, 1]||aten::add.Tensor|4|
|307|Tensor<[100,92]>,dims: [0, 1]||aten::add.Tensor|4|
|308|Tensor<[92]>,dims: [1]||aten::add.Tensor|4|
|309|Tensor<[100,4]>,dims: [0, 1]||aten::add.Tensor|4|
|310|Tensor<[4]>,dims: [1]||aten::add.Tensor|4|
|311|Tensor<[3,64,1445]>,dims: [0, 1, 2]||aten::bmm|4|
|312|Tensor<[3,1445,64]>,dims: [0, 1, 2]||aten::bmm|4|
|313|Tensor<[1,192,32,42]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|314|Tensor<[192,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|315|Tensor<[1,3,1445,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|316|Tensor<[1,3,64,1445]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|317|Tensor<[192]>,dims: [0]||aten::mul.Tensor|4|
|318|Tensor<[92]>,dims: [0]||aten::mul.Tensor|4|
|319|Tensor<[4]>,dims: [0]||aten::mul.Tensor|4|
|320|Tensor<[1,256,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|321|Tensor<[1,512,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|322|Tensor<[1,12,8,8]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|323|Tensor<[1,12,8,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|324|Tensor<[1,8,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|325|Tensor<[1,8,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|326|Tensor<[1,1,1,8]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|327|Tensor<[1,768]>,dims: [0, 1]||aten::add.Tensor|4|
|328|Tensor<[1,3]>,dims: [0, 1]||aten::add.Tensor|4|
|329|Tensor<[3]>,dims: [1]||aten::add.Tensor|4|
|330|Tensor<[12,64,8]>,dims: [0, 1, 2]||aten::bmm|4|
|331|Tensor<[12,8,64]>,dims: [0, 1, 2]||aten::bmm|4|
|332|Tensor<[1,768,8]>,dims: [0, 1, 2]||aten::convolution|4|
|333|Tensor<[768,1]>,dims: [1, 2]||aten::convolution|4|
|334|Tensor<[1,3072,8]>,dims: [0, 1, 2]||aten::convolution|4|
|335|Tensor<[3072,1]>,dims: [1, 2]||aten::convolution|4|
|336|Tensor<[3]>,dims: [0]||aten::mul.Tensor|4|
|337|Tensor<[1,8,256,2048]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|338|Tensor<[1,8,2048,256]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|339|Tensor<[1,8,2048,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|340|Tensor<[1,2048,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|341|Tensor<[2048,768]>,dims: [1, 2]||aten::add.Tensor|4|
|342|Tensor<[1,2048,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|343|Tensor<[2048,256]>,dims: [0, 1]||aten::add.Tensor|4|
|344|Tensor<[2048,1280]>,dims: [0, 1]||aten::add.Tensor|4|
|345|Tensor<[1,1,1,2048]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|346|Tensor<[256,768]>,dims: [0, 1]||aten::add.Tensor|4|
|347|Tensor<[2048,768]>,dims: [0, 1]||aten::add.Tensor|4|
|348|Tensor<[2048,262]>,dims: [0, 1]||aten::add.Tensor|4|
|349|Tensor<[262]>,dims: [1]||aten::add.Tensor|4|
|350|Tensor<[8,32,2048]>,dims: [0, 1, 2]||aten::bmm|4|
|351|Tensor<[8,2048,160]>,dims: [0, 1, 2]||aten::bmm|4|
|352|Tensor<[8,32,256]>,dims: [0, 1, 2]||aten::bmm|4|
|353|Tensor<[8,256,96]>,dims: [0, 1, 2]||aten::bmm|4|
|354|Tensor<[256,1280]>,dims: [1, 2]||aten::expand|5|
|355|Tensor<[1,256,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|356|Tensor<[1024]>,dims: [0]||aten::add.Tensor|4|
|357|Tensor<[1,1024,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|358|Tensor<[1024,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|359|Tensor<[1,512,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|360|Tensor<[1,2048,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|361|Tensor<[2048,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|362|Tensor<[1,12,201,201]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|363|Tensor<[1,12,201,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|364|Tensor<[12]>,dims: [0]||aten::add.Tensor|4|
|365|Tensor<[1,201,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|366|Tensor<[1,201,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|367|Tensor<[201,768]>,dims: [0, 1]||aten::add.Tensor|4|
|368|Tensor<[1,1,1,201]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|369|Tensor<[201,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|370|Tensor<[1,1536]>,dims: [0, 1]||aten::add.Tensor|4|
|371|Tensor<[1536]>,dims: [1]||aten::add.Tensor|4|
|372|Tensor<[1,3129]>,dims: [0, 1]||aten::add.Tensor|4|
|373|Tensor<[3129]>,dims: [1]||aten::add.Tensor|4|
|374|Tensor<[12,64,201]>,dims: [0, 1, 2]||aten::bmm|4|
|375|Tensor<[12,201,64]>,dims: [0, 1, 2]||aten::bmm|4|
|376|Tensor<[1,768,12,16]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|377|Tensor<[768,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|378|Tensor<[12,1]>,dims: [0, 1]||aten::expand|4|
|379|Tensor<[1,16]>,dims: [0, 1]||aten::expand|4|
|380|Tensor<[12,1]>,dims: [2, 3]||aten::index.Tensor|4|
|381|Tensor<[1536]>,dims: [0]||aten::mul.Tensor|4|
|382|Tensor<[3129]>,dims: [0]||aten::mul.Tensor|4|
|383|Tensor<[1,192]>,dims: [0, 1]||aten::rsub.Scalar|4|
|384|Tensor<[1,128]>,dims: [0, 1]||aten::add.Tensor|4|
|385|Tensor<[128]>,dims: [1]||aten::add.Tensor|4|
|386|Tensor<[10]>,dims: [1]||aten::add.Tensor|4|
|387|Tensor<[1,32,26,26]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|388|Tensor<[1,64,24,24]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|389|Tensor<[10]>,dims: [0]||aten::mul.Tensor|4|
|390|Tensor<[16,19,19]>,dims: [0, 1, 2]||aten::_softmax|4|
|391|Tensor<[16,19,1]>,dims: [0, 1, 2]||aten::_softmax|4|
|392|Tensor<[19]>,dims: [0]||aten::add.Tensor|4|
|393|Tensor<[1,19]>,dims: [0, 1]||aten::add.Tensor|4|
|394|Tensor<[1,19,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|395|Tensor<[1,19,1024]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|396|Tensor<[1024]>,dims: [2]||aten::add.Tensor|4|
|397|Tensor<[19,1024]>,dims: [0, 1]||aten::add.Tensor|4|
|398|Tensor<[1024]>,dims: [1]||aten::add.Tensor|4|
|399|Tensor<[1,16,19,19]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|400|Tensor<[1,1,19,19]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|401|Tensor<[19,4096]>,dims: [0, 1]||aten::add.Tensor|4|
|402|Tensor<[4096]>,dims: [1]||aten::add.Tensor|4|
|403|Tensor<[16,64,19]>,dims: [0, 1, 2]||aten::bmm|4|
|404|Tensor<[16,19,64]>,dims: [0, 1, 2]||aten::bmm|4|
|405|Tensor<[1,1,1,19]>,dims: [0, 1, 2, 3]||aten::expand|4|
|406|Tensor<[19]>,dims: [1]||aten::lt.Tensor|4|
|407|Tensor<[19,1]>,dims: [0, 1]||aten::lt.Tensor|4|
|408|Tensor<[4096]>,dims: [0]||aten::mul.Tensor|4|
|409|Tensor<[19,256008]>,dims: [0, 1]||aten::sub.Tensor|4|
|410|Tensor<[19,19]>,dims: [0, 1]||aten::where.self|4|
|411|Tensor<[14]>,dims: [0]||aten::add.Tensor|4|
|412|Tensor<[1,14,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|413|Tensor<[14,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|414|Tensor<[24]>,dims: [0]||aten::add.Tensor|4|
|415|Tensor<[1,24,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|416|Tensor<[24,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|417|Tensor<[1,40,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|418|Tensor<[40,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|419|Tensor<[68]>,dims: [0]||aten::add.Tensor|4|
|420|Tensor<[1,68,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|421|Tensor<[68,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|422|Tensor<[1,16,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|423|Tensor<[16,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|424|Tensor<[28]>,dims: [0]||aten::add.Tensor|4|
|425|Tensor<[1,28,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|426|Tensor<[28,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|427|Tensor<[46]>,dims: [0]||aten::add.Tensor|4|
|428|Tensor<[1,46,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|429|Tensor<[46,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|430|Tensor<[78]>,dims: [0]||aten::add.Tensor|4|
|431|Tensor<[1,78,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|432|Tensor<[78,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|433|Tensor<[134]>,dims: [0]||aten::add.Tensor|4|
|434|Tensor<[1,134,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|435|Tensor<[134,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|436|Tensor<[20]>,dims: [0]||aten::add.Tensor|4|
|437|Tensor<[1,20,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|438|Tensor<[20,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|439|Tensor<[34]>,dims: [0]||aten::add.Tensor|4|
|440|Tensor<[1,34,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|441|Tensor<[34,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|442|Tensor<[58]>,dims: [0]||aten::add.Tensor|4|
|443|Tensor<[1,58,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|444|Tensor<[58,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|445|Tensor<[98]>,dims: [0]||aten::add.Tensor|4|
|446|Tensor<[1,98,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|447|Tensor<[98,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|448|Tensor<[168]>,dims: [0]||aten::add.Tensor|4|
|449|Tensor<[1,168,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|450|Tensor<[168,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|451|Tensor<[1,320,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|452|Tensor<[1,40,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|453|Tensor<[1,68,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|454|Tensor<[116]>,dims: [0]||aten::add.Tensor|4|
|455|Tensor<[1,116,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|456|Tensor<[116,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|457|Tensor<[196]>,dims: [0]||aten::add.Tensor|4|
|458|Tensor<[1,196,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|459|Tensor<[196,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|460|Tensor<[334]>,dims: [0]||aten::add.Tensor|4|
|461|Tensor<[1,334,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|462|Tensor<[334,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|463|Tensor<[1,640,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|464|Tensor<[1,160,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|465|Tensor<[160,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|466|Tensor<[272]>,dims: [0]||aten::add.Tensor|4|
|467|Tensor<[1,272,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|468|Tensor<[272,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|469|Tensor<[462]>,dims: [0]||aten::add.Tensor|4|
|470|Tensor<[1,462,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|471|Tensor<[462,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|472|Tensor<[1,1024,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|473|Tensor<[1,32,512,512]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|474|Tensor<[1,64,256,256]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|475|Tensor<[1,32,256,256]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|476|Tensor<[1,128,128,128]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|477|Tensor<[1,64,128,128]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|478|Tensor<[1,256,64,64]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|479|Tensor<[1,128,64,64]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|480|Tensor<[1,512,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|481|Tensor<[1,256,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|482|Tensor<[1,1024,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|483|Tensor<[1,512,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|484|Tensor<[1,256,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|485|Tensor<[1,128,32,32]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|486|Tensor<[1,255,16,16]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|487|Tensor<[255,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|488|Tensor<[1,255,32,32]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|489|Tensor<[1,255,64,64]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|490|Tensor<[1,1,256,256]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|491|Tensor<[1,4,14,14]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|492|Tensor<[1,16,14,14]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|493|Tensor<[1,1,28,28]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|494|Tensor<[1,16,32,32]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|495|Tensor<[1,16,32,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|496|Tensor<[1,32,1536]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|497|Tensor<[1536]>,dims: [2]||aten::add.Tensor|4|
|498|Tensor<[32,4608]>,dims: [0, 1]||aten::add.Tensor|4|
|499|Tensor<[4608]>,dims: [1]||aten::add.Tensor|4|
|500|Tensor<[32,1536]>,dims: [0, 1]||aten::add.Tensor|4|
|501|Tensor<[32,6144]>,dims: [0, 1]||aten::add.Tensor|4|
|502|Tensor<[6144]>,dims: [1]||aten::add.Tensor|4|
|503|Tensor<[1,32,6144]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|504|Tensor<[16,96,32]>,dims: [0, 1, 2]||aten::baddbmm|4|
|505|Tensor<[16,32,32]>,dims: [0, 1, 2]||aten::baddbmm|4|
|506|Tensor<[16,1,32]>,dims: [0, 1, 2]||aten::baddbmm|4|
|507|Tensor<[16,32,96]>,dims: [0, 1, 2]||aten::bmm|4|
|508|Tensor<[16,1]>,dims: [1, 2]||aten::mul.Tensor|4|
|509|Tensor<[4608]>,dims: [0]||aten::mul.Tensor|4|
|510|Tensor<[6144]>,dims: [0]||aten::mul.Tensor|4|
|511|Tensor<[1,12,16,16]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|512|Tensor<[1,12,16,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|513|Tensor<[1,16,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|514|Tensor<[1,16,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|515|Tensor<[16,768]>,dims: [0, 1]||aten::add.Tensor|4|
|516|Tensor<[1,1,16,16]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|517|Tensor<[16,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|518|Tensor<[12,64,16]>,dims: [0, 1, 2]||aten::bmm|4|
|519|Tensor<[12,16,64]>,dims: [0, 1, 2]||aten::bmm|4|
|520|Tensor<[1,1,1,16]>,dims: [0, 1, 2, 3]||aten::expand|4|
|521|Tensor<[1,12,16,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|522|Tensor<[1,12,64,16]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|523|Tensor<[1,64,224,224]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|524|Tensor<[1,128,112,112]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|525|Tensor<[1,1,224,224]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|526|Tensor<[1,1,19200,300]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|527|Tensor<[1,1,19200,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|528|Tensor<[1,2,4800,300]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|529|Tensor<[1,2,4800,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|530|Tensor<[1,5,1200,300]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|531|Tensor<[1,5,1200,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|532|Tensor<[1,8,300,300]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|533|Tensor<[1,8,300,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|534|Tensor<[1,19200,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|535|Tensor<[1,19200,64]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|536|Tensor<[64]>,dims: [2]||aten::add.Tensor|4|
|537|Tensor<[19200,64]>,dims: [0, 1]||aten::add.Tensor|4|
|538|Tensor<[64]>,dims: [1]||aten::add.Tensor|4|
|539|Tensor<[1,300,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|540|Tensor<[1,300,64]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|541|Tensor<[300,64]>,dims: [0, 1]||aten::add.Tensor|4|
|542|Tensor<[19200,256]>,dims: [0, 1]||aten::add.Tensor|4|
|543|Tensor<[1,4800,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|544|Tensor<[1,4800,128]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|545|Tensor<[128]>,dims: [2]||aten::add.Tensor|4|
|546|Tensor<[4800,128]>,dims: [0, 1]||aten::add.Tensor|4|
|547|Tensor<[1,300,128]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|548|Tensor<[300,128]>,dims: [0, 1]||aten::add.Tensor|4|
|549|Tensor<[4800,512]>,dims: [0, 1]||aten::add.Tensor|4|
|550|Tensor<[1,1200,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|551|Tensor<[1,1200,320]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|552|Tensor<[1200,320]>,dims: [0, 1]||aten::add.Tensor|4|
|553|Tensor<[1,300,320]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|554|Tensor<[300,320]>,dims: [0, 1]||aten::add.Tensor|4|
|555|Tensor<[1200,1280]>,dims: [0, 1]||aten::add.Tensor|4|
|556|Tensor<[1,300,512]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|557|Tensor<[300,512]>,dims: [0, 1]||aten::add.Tensor|4|
|558|Tensor<[300,2048]>,dims: [0, 1]||aten::add.Tensor|4|
|559|Tensor<[30]>,dims: [0]||aten::add.Tensor|4|
|560|Tensor<[30,1]>,dims: [0, 1]||aten::add.Tensor|4|
|561|Tensor<[1,64,30,40]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|562|Tensor<[1,32,30,40]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|563|Tensor<[60]>,dims: [0]||aten::add.Tensor|4|
|564|Tensor<[60,1]>,dims: [0, 1]||aten::add.Tensor|4|
|565|Tensor<[80]>,dims: [0]||aten::add.Tensor|4|
|566|Tensor<[1,64,60,80]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|567|Tensor<[1,32,60,80]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|568|Tensor<[120]>,dims: [0]||aten::add.Tensor|4|
|569|Tensor<[120,1]>,dims: [0, 1]||aten::add.Tensor|4|
|570|Tensor<[1,64,120,160]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|571|Tensor<[1,32,120,160]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|572|Tensor<[240]>,dims: [0]||aten::add.Tensor|4|
|573|Tensor<[240,1]>,dims: [0, 1]||aten::add.Tensor|4|
|574|Tensor<[480]>,dims: [0]||aten::add.Tensor|4|
|575|Tensor<[480,1]>,dims: [0, 1]||aten::add.Tensor|4|
|576|Tensor<[1,64,300]>,dims: [0, 1, 2]||aten::bmm|4|
|577|Tensor<[1,256,64]>,dims: [0, 1, 2]||aten::bmm|4|
|578|Tensor<[2,64,300]>,dims: [0, 1, 2]||aten::bmm|4|
|579|Tensor<[2,300,64]>,dims: [0, 1, 2]||aten::bmm|4|
|580|Tensor<[1,512,128]>,dims: [0, 1, 2]||aten::bmm|4|
|581|Tensor<[5,64,300]>,dims: [0, 1, 2]||aten::bmm|4|
|582|Tensor<[5,300,64]>,dims: [0, 1, 2]||aten::bmm|4|
|583|Tensor<[1,1280,320]>,dims: [0, 1, 2]||aten::bmm|4|
|584|Tensor<[8,64,300]>,dims: [0, 1, 2]||aten::bmm|4|
|585|Tensor<[8,300,64]>,dims: [0, 1, 2]||aten::bmm|4|
|586|Tensor<[1,2048,512]>,dims: [0, 1, 2]||aten::bmm|4|
|587|Tensor<[1,64,15,20]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|588|Tensor<[1,256,120,160]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|589|Tensor<[1,128,60,80]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|590|Tensor<[1,128,15,20]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|591|Tensor<[1,512,60,80]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|592|Tensor<[1,320,30,40]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|593|Tensor<[1,320,15,20]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|594|Tensor<[1,1280,30,40]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|595|Tensor<[1,512,15,20]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|596|Tensor<[1,2048,15,20]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|597|Tensor<[1,2,30,40]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|598|Tensor<[2,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|599|Tensor<[1,2,60,80]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|600|Tensor<[1,2,120,160]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|601|Tensor<[1,64,480,640]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|602|Tensor<[1,1,480,640]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|603|Tensor<[256,64]>,dims: [1, 2]||aten::expand|5|
|604|Tensor<[512,128]>,dims: [1, 2]||aten::expand|5|
|605|Tensor<[1280,320]>,dims: [1, 2]||aten::expand|5|
|606|Tensor<[2048,512]>,dims: [1, 2]||aten::expand|5|
|607|Tensor<[30,1]>,dims: [2, 3]||aten::index.Tensor|4|
|608|Tensor<[60,1]>,dims: [2, 3]||aten::index.Tensor|4|
|609|Tensor<[80]>,dims: [3]||aten::index.Tensor|4|
|610|Tensor<[120,1]>,dims: [2, 3]||aten::index.Tensor|4|
|611|Tensor<[160]>,dims: [3]||aten::index.Tensor|4|
|612|Tensor<[240,1]>,dims: [2, 3]||aten::index.Tensor|4|
|613|Tensor<[320]>,dims: [3]||aten::index.Tensor|4|
|614|Tensor<[480,1]>,dims: [2, 3]||aten::index.Tensor|4|
|615|Tensor<[640]>,dims: [3]||aten::index.Tensor|4|
|616|Tensor<[1,1,30,40]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|617|Tensor<[1,1,60,80]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|618|Tensor<[1,1,120,160]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|619|Tensor<[1,64,240,320]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|620|Tensor<[1,12,197,197]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|621|Tensor<[1,12,197,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|622|Tensor<[1,197,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|623|Tensor<[1,197,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|624|Tensor<[197,768]>,dims: [0, 1]||aten::add.Tensor|4|
|625|Tensor<[197,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|626|Tensor<[12,64,197]>,dims: [0, 1, 2]||aten::bmm|4|
|627|Tensor<[12,197,64]>,dims: [0, 1, 2]||aten::bmm|4|
|628|Tensor<[1,768,14,14]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|629|Tensor<[1,12,197,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|630|Tensor<[1,12,64,197]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|631|Tensor<[1,1,16384,256]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|632|Tensor<[1,1,16384,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|633|Tensor<[1,2,4096,256]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|634|Tensor<[1,2,4096,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|635|Tensor<[1,5,1024,256]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|636|Tensor<[1,5,1024,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|637|Tensor<[1,16384,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|638|Tensor<[1,16384,32]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|639|Tensor<[32]>,dims: [2]||aten::add.Tensor|4|
|640|Tensor<[16384,32]>,dims: [0, 1]||aten::add.Tensor|4|
|641|Tensor<[1,256,32]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|642|Tensor<[256,32]>,dims: [0, 1]||aten::add.Tensor|4|
|643|Tensor<[16384,128]>,dims: [0, 1]||aten::add.Tensor|4|
|644|Tensor<[1,4096,64]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|645|Tensor<[4096,64]>,dims: [0, 1]||aten::add.Tensor|4|
|646|Tensor<[256,64]>,dims: [0, 1]||aten::add.Tensor|4|
|647|Tensor<[4096,256]>,dims: [0, 1]||aten::add.Tensor|4|
|648|Tensor<[1,1024,160]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|649|Tensor<[160]>,dims: [2]||aten::add.Tensor|4|
|650|Tensor<[1024,160]>,dims: [0, 1]||aten::add.Tensor|4|
|651|Tensor<[160]>,dims: [1]||aten::add.Tensor|4|
|652|Tensor<[1,256,160]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|653|Tensor<[256,160]>,dims: [0, 1]||aten::add.Tensor|4|
|654|Tensor<[1,256,256]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|655|Tensor<[256,1024]>,dims: [0, 1]||aten::add.Tensor|4|
|656|Tensor<[1,16384,256]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|657|Tensor<[128,1]>,dims: [0, 1]||aten::add.Tensor|4|
|658|Tensor<[1,4096,256]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|659|Tensor<[1,1024,256]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|660|Tensor<[1,256,128,128]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|661|Tensor<[1,32,256]>,dims: [0, 1, 2]||aten::bmm|4|
|662|Tensor<[1,128,32]>,dims: [0, 1, 2]||aten::bmm|4|
|663|Tensor<[2,32,256]>,dims: [0, 1, 2]||aten::bmm|4|
|664|Tensor<[2,256,32]>,dims: [0, 1, 2]||aten::bmm|4|
|665|Tensor<[5,32,256]>,dims: [0, 1, 2]||aten::bmm|4|
|666|Tensor<[5,256,32]>,dims: [0, 1, 2]||aten::bmm|4|
|667|Tensor<[1,640,160]>,dims: [0, 1, 2]||aten::bmm|4|
|668|Tensor<[8,256,32]>,dims: [0, 1, 2]||aten::bmm|4|
|669|Tensor<[1,64,256]>,dims: [0, 1, 2]||aten::bmm|4|
|670|Tensor<[1,160,256]>,dims: [0, 1, 2]||aten::bmm|4|
|671|Tensor<[1,32,128,128]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|672|Tensor<[1,32,16,16]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|673|Tensor<[1,64,64,64]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|674|Tensor<[1,64,16,16]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|675|Tensor<[1,160,32,32]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|676|Tensor<[1,160,16,16]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|677|Tensor<[1,150,128,128]>,dims: [0, 1, 2, 3]||aten::convolution|4|
|678|Tensor<[150,1,1]>,dims: [1, 2, 3]||aten::convolution|4|
|679|Tensor<[128,32]>,dims: [1, 2]||aten::expand|5|
|680|Tensor<[640,160]>,dims: [1, 2]||aten::expand|5|
|681|Tensor<[1024,256]>,dims: [1, 2]||aten::expand|5|
|682|Tensor<[32,256]>,dims: [1, 2]||aten::expand|5|
|683|Tensor<[64,256]>,dims: [1, 2]||aten::expand|5|
|684|Tensor<[160,256]>,dims: [1, 2]||aten::expand|5|
|685|Tensor<[128,1]>,dims: [2, 3]||aten::index.Tensor|4|
|686|Tensor<[1,71,7,7]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|687|Tensor<[1,71,7,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|688|Tensor<[1,7,4544]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|689|Tensor<[4544]>,dims: [2]||aten::add.Tensor|4|
|690|Tensor<[1,1,7]>,dims: [0, 1, 2]||aten::bmm|4|
|691|Tensor<[71,64,7]>,dims: [0, 1, 2]||aten::bmm|4|
|692|Tensor<[71,7,64]>,dims: [0, 1, 2]||aten::bmm|4|
|693|Tensor<[1,1,64,7]>,dims: [0, 1, 2, 3]||aten::expand|5|
|694|Tensor<[1,1,7,64]>,dims: [0, 1, 2, 3]||aten::expand|5|
|695|Tensor<[7,1,1]>,dims: [1, 2, 3]||aten::index.Tensor|4|
|696|Tensor<[1,1]>,dims: [2, 3]||aten::index.Tensor|4|
|697|Tensor<[1,71,7,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|698|Tensor<[1,7,64]>,dims: [0, 1, 2]||aten::mul.Tensor|4|
|699|Tensor<[1,16,112,112]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|700|Tensor<[96]>,dims: [0]||aten::add.Tensor|4|
|701|Tensor<[1,96,112,112]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|702|Tensor<[96,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|703|Tensor<[1,96,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|704|Tensor<[144]>,dims: [0]||aten::add.Tensor|4|
|705|Tensor<[1,144,56,56]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|706|Tensor<[144,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|707|Tensor<[1,144,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|708|Tensor<[1,32,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|709|Tensor<[1,192,28,28]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|710|Tensor<[1,192,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|711|Tensor<[1,64,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|712|Tensor<[384]>,dims: [0]||aten::add.Tensor|4|
|713|Tensor<[1,384,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|714|Tensor<[384,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|715|Tensor<[1,96,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|716|Tensor<[576]>,dims: [0]||aten::add.Tensor|4|
|717|Tensor<[1,576,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|718|Tensor<[576,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|719|Tensor<[1,576,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|720|Tensor<[960]>,dims: [0]||aten::add.Tensor|4|
|721|Tensor<[1,960,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|722|Tensor<[960,1,1]>,dims: [1, 2, 3]||aten::add.Tensor|4|
|723|Tensor<[1,320,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|724|Tensor<[1,1280,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|725|Tensor<[1,12,12,12]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|726|Tensor<[1,12,12,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|727|Tensor<[1,12,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|728|Tensor<[1,12,128]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|729|Tensor<[12,768]>,dims: [0, 1]||aten::add.Tensor|4|
|730|Tensor<[1,1,12,12]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|731|Tensor<[1,12,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|732|Tensor<[12,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|733|Tensor<[1,12,3072]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|734|Tensor<[12,2]>,dims: [0, 1]||aten::add.Tensor|4|
|735|Tensor<[12,64,12]>,dims: [0, 1, 2]||aten::bmm|4|
|736|Tensor<[12,12,64]>,dims: [0, 1, 2]||aten::bmm|4|
|737|Tensor<[1,1,1,12]>,dims: [0, 1, 2, 3]||aten::expand|4|
|738|Tensor<[1,12,12,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|739|Tensor<[1,12,64,12]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|740|Tensor<[1,12,9,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|741|Tensor<[1,12,9,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|742|Tensor<[1,9,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|743|Tensor<[1,9,128]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|744|Tensor<[9,768]>,dims: [0, 1]||aten::add.Tensor|4|
|745|Tensor<[1,1,9,9]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|746|Tensor<[1,9,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|747|Tensor<[9,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|748|Tensor<[1,9,3072]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|749|Tensor<[9,128]>,dims: [0, 1]||aten::add.Tensor|4|
|750|Tensor<[9,30000]>,dims: [0, 1]||aten::add.Tensor|4|
|751|Tensor<[30000]>,dims: [1]||aten::add.Tensor|4|
|752|Tensor<[12,64,9]>,dims: [0, 1, 2]||aten::bmm|4|
|753|Tensor<[12,9,64]>,dims: [0, 1, 2]||aten::bmm|4|
|754|Tensor<[1,1,1,9]>,dims: [0, 1, 2, 3]||aten::expand|4|
|755|Tensor<[1,12,9,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|756|Tensor<[1,12,64,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|757|Tensor<[30000]>,dims: [0]||aten::mul.Tensor|4|
|758|Tensor<[1,16,9,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|759|Tensor<[1,16,9,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|760|Tensor<[9,2048]>,dims: [0, 1]||aten::add.Tensor|4|
|761|Tensor<[1,9,2048]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|762|Tensor<[2048]>,dims: [2]||aten::add.Tensor|4|
|763|Tensor<[9,8192]>,dims: [0, 1]||aten::add.Tensor|4|
|764|Tensor<[8192]>,dims: [1]||aten::add.Tensor|4|
|765|Tensor<[1,9,8192]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|766|Tensor<[16,128,9]>,dims: [0, 1, 2]||aten::bmm|4|
|767|Tensor<[16,9,128]>,dims: [0, 1, 2]||aten::bmm|4|
|768|Tensor<[1,16,9,128]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|769|Tensor<[1,16,128,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|770|Tensor<[8192]>,dims: [0]||aten::mul.Tensor|4|
|771|Tensor<[9,1024]>,dims: [0, 1]||aten::add.Tensor|4|
|772|Tensor<[1,9,1024]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|773|Tensor<[9,4096]>,dims: [0, 1]||aten::add.Tensor|4|
|774|Tensor<[1,9,4096]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|775|Tensor<[16,64,9]>,dims: [0, 1, 2]||aten::bmm|4|
|776|Tensor<[16,9,64]>,dims: [0, 1, 2]||aten::bmm|4|
|777|Tensor<[1,16,9,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|778|Tensor<[1,16,64,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|779|Tensor<[1,64,9,9]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|780|Tensor<[1,64,9,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|781|Tensor<[9,16384]>,dims: [0, 1]||aten::add.Tensor|4|
|782|Tensor<[16384]>,dims: [1]||aten::add.Tensor|4|
|783|Tensor<[1,9,16384]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|784|Tensor<[64,64,9]>,dims: [0, 1, 2]||aten::bmm|4|
|785|Tensor<[64,9,64]>,dims: [0, 1, 2]||aten::bmm|4|
|786|Tensor<[1,64,9,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|787|Tensor<[1,64,64,9]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|788|Tensor<[16384]>,dims: [0]||aten::mul.Tensor|4|
|789|Tensor<[1,2]>,dims: [0, 1]||aten::add.Tensor|4|
|790|Tensor<[1,12,14,14]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|791|Tensor<[1,12,14,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|792|Tensor<[1,14,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|793|Tensor<[1,14,128]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|794|Tensor<[14,768]>,dims: [0, 1]||aten::add.Tensor|4|
|795|Tensor<[1,1,14,14]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|796|Tensor<[1,14,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|797|Tensor<[14,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|798|Tensor<[1,14,3072]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|799|Tensor<[14,2]>,dims: [0, 1]||aten::add.Tensor|4|
|800|Tensor<[12,64,14]>,dims: [0, 1, 2]||aten::bmm|4|
|801|Tensor<[12,14,64]>,dims: [0, 1, 2]||aten::bmm|4|
|802|Tensor<[1,1,1,14]>,dims: [0, 1, 2, 3]||aten::expand|4|
|803|Tensor<[1,12,14,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|804|Tensor<[1,12,64,14]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|805|Tensor<[1,12,50,50]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|806|Tensor<[1,12,50,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|807|Tensor<[2,8,7,7]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|808|Tensor<[2,8,7,1]>,dims: [0, 1, 2, 3]||aten::_safe_softmax|4|
|809|Tensor<[1,50,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|810|Tensor<[1,50,768]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|811|Tensor<[50,768]>,dims: [0, 1]||aten::add.Tensor|4|
|812|Tensor<[50,3072]>,dims: [0, 1]||aten::add.Tensor|4|
|813|Tensor<[2,7,512]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|814|Tensor<[1,7,512]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|815|Tensor<[2,7,1]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|816|Tensor<[14,512]>,dims: [0, 1]||aten::add.Tensor|4|
|817|Tensor<[2,1,7,7]>,dims: [0, 1, 2, 3]||aten::add.Tensor|4|
|818|Tensor<[14,2048]>,dims: [0, 1]||aten::add.Tensor|4|
|819|Tensor<[12,64,50]>,dims: [0, 1, 2]||aten::bmm|4|
|820|Tensor<[12,50,64]>,dims: [0, 1, 2]||aten::bmm|4|
|821|Tensor<[16,64,7]>,dims: [0, 1, 2]||aten::bmm|4|
|822|Tensor<[16,7,64]>,dims: [0, 1, 2]||aten::bmm|4|
|823|Tensor<[2,512]>,dims: [0, 1]||aten::div.Tensor|4|
|824|Tensor<[2,1]>,dims: [0, 1]||aten::div.Tensor|4|
|825|Tensor<[2,1,1,7]>,dims: [0, 1, 2, 3]||aten::expand|4|
|826|Tensor<[1,12,50,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|827|Tensor<[1,12,64,50]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|828|Tensor<[2,8,7,64]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|829|Tensor<[2,8,64,7]>,dims: [0, 1, 2, 3]||aten::mul.Scalar|4|
|830|Tensor<[1,50,3072]>,dims: [0, 1, 2]||aten::mul.Tensor|4|
|831|Tensor<[2,7,2048]>,dims: [0, 1, 2]||aten::mul.Tensor|4|
|832|Tensor<[1,16,197,197]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|833|Tensor<[1,16,197,1]>,dims: [0, 1, 2, 3]||aten::_softmax|4|
|834|Tensor<[1,197,1024]>,dims: [0, 1, 2]||aten::add.Tensor|4|
|835|Tensor<[197,1024]>,dims: [0, 1]||aten::add.Tensor|4|
|836|Tensor<[27]>,dims: [0]||aten::add.Tensor|4|
|837|Tensor<[27,1]>,dims: [0, 1]||aten::add.Tensor|4|
|838|Tensor<[196,196]>,dims: [0, 1]||aten::add.Tensor|4|
|839|Tensor<[197,4096]>,dims: [0, 1]||aten::add.Tensor|4|
|840|Tensor<[1,1024]>,dims: [0, 1]||aten::add.Tensor|4|
|841|Tensor<[197]>,dims: [0]||aten::arange|4|
|842|Tensor<[16,64,197]>,dims: [0, 1, 2]||aten::bmm|4|
|843|Tensor<[16,197,64]>,dims: [0, 1, 2]||aten::bmm|4|
|844|Tensor<[14,1]>,dims: [0, 1]||aten::expand|4|
|845|Tensor<[1,14]>,dims: [0, 1]||aten::expand|4|
|846|Tensor<[27,1]>,dims: [2, 3]||aten::index.Tensor|4|
|847|Tensor<[27]>,dims: [3]||aten::index.Tensor|4|
|848|Tensor<[1,16,27,27]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|849|Tensor<[2,196,1]>,dims: [0, 1, 2]||aten::sub.Tensor|4|
|850|Tensor<[2,1,196]>,dims: [0, 1, 2]||aten::sub.Tensor|4|
|851|Tensor<[1,197]>,dims: [0, 1]||aten::where.self|4|
|852|Tensor<[196,197]>,dims: [0, 1]||aten::where.self|4|
|853|Tensor<[197,1]>,dims: [0, 1]||aten::where.self|4|
|854|Tensor<[197,197]>,dims: [0, 1]||aten::where.self|4|
|855|Tensor<[12,1,1]>,dims: [1, 2, 3]||aten::index.Tensor|4|
|856|Tensor<[1,12,27,27]>,dims: [0, 1, 2, 3]||aten::mul.Tensor|4|
|857|Tensor<[1,64]>,dims: [0, 1]||aten::add.Tensor|4|
|858|Tensor<[1,12]>,dims: [0, 1]||aten::add.Tensor|4|
|859|Tensor<[12]>,dims: [1]||aten::add.Tensor|4|
|860|Tensor<[1,784]>,dims: [0, 1]||aten::add.Tensor|4|
|861|Tensor<[784]>,dims: [1]||aten::add.Tensor|4|
|862|Tensor<[784]>,dims: [0]||aten::mul.Tensor|4|
