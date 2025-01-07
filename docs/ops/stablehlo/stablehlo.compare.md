
### stablehlo.compare::ttnn.?


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,32,32,32]>,Tensor<[1,32,32,32]>,|ttnn.eq|aten::_safe_softmax|4|
|1|Tensor<[1,1,32,32]>,Tensor<[1,1,32,32]>,|ttnn.eq|aten::eq.Scalar|4|
|2|Tensor<[32,32]>,Tensor<[32,32]>,|ttnn.gt|aten::gt.Tensor|4|
|3|Tensor<[1,12,7,7]>,Tensor<[1,12,7,7]>,|ttnn.eq|aten::_safe_softmax|4|
|4|Tensor<[1,7]>,Tensor<[1,7]>,|ttnn.eq|aten::eq.Scalar|4|
|5|Tensor<[7,7]>,Tensor<[7,7]>,|ttnn.lt|aten::lt.Tensor|4|
|6|Tensor<[1,12,10,10]>,Tensor<[1,12,10,10]>,|ttnn.eq|aten::_safe_softmax|4|
|7|Tensor<[1,10]>,Tensor<[1,10]>,|ttnn.ne|aten::ne.Scalar|4|
|8|Tensor<[1,8,4096,4096]>,Tensor<[1,8,4096,4096]>,|ttnn.eq|aten::_safe_softmax|4|
|9|Tensor<[1,8,4096,9]>,Tensor<[1,8,4096,9]>,|ttnn.eq|aten::_safe_softmax|4|
|10|Tensor<[1,8,1024,1024]>,Tensor<[1,8,1024,1024]>,|ttnn.eq|aten::_safe_softmax|4|
|11|Tensor<[1,8,1024,9]>,Tensor<[1,8,1024,9]>,|ttnn.eq|aten::_safe_softmax|4|
|12|Tensor<[1,8,256,256]>,Tensor<[1,8,256,256]>,|ttnn.eq|aten::_safe_softmax|4|
|13|Tensor<[1,8,256,9]>,Tensor<[1,8,256,9]>,|ttnn.eq|aten::_safe_softmax|4|
|14|Tensor<[1,8,64,64]>,Tensor<[1,8,64,64]>,|ttnn.eq|aten::_safe_softmax|4|
|15|Tensor<[1,8,64,9]>,Tensor<[1,8,64,9]>,|ttnn.eq|aten::_safe_softmax|4|
|16|Tensor<[1,12,25,25]>,Tensor<[1,12,25,25]>,|ttnn.eq|aten::_safe_softmax|4|
|17|Tensor<[1,3,1445,1445]>,Tensor<[1,3,1445,1445]>,|ttnn.eq|aten::_safe_softmax|4|
|18|Tensor<[19]>,Tensor<[19]>,|ttnn.lt|aten::lt.Scalar|4|
|19|Tensor<[19,19]>,Tensor<[19,19]>,|ttnn.lt|aten::lt.Tensor|4|
|20|Tensor<[1,12,16,16]>,Tensor<[1,12,16,16]>,|ttnn.eq|aten::_safe_softmax|4|
|21|Tensor<[1,12,197,197]>,Tensor<[1,12,197,197]>,|ttnn.eq|aten::_safe_softmax|4|
|22|Tensor<[1,71,7,7]>,Tensor<[1,71,7,7]>,|ttnn.eq|aten::_safe_softmax|4|
|23|Tensor<[1,1,7,7]>,Tensor<[1,1,7,7]>,|ttnn.eq|aten::eq.Scalar|4|
|24|Tensor<[1,12,12,12]>,Tensor<[1,12,12,12]>,|ttnn.eq|aten::_safe_softmax|4|
|25|Tensor<[1,12,9,9]>,Tensor<[1,12,9,9]>,|ttnn.eq|aten::_safe_softmax|4|
|26|Tensor<[1,16,9,9]>,Tensor<[1,16,9,9]>,|ttnn.eq|aten::_safe_softmax|4|
|27|Tensor<[1,64,9,9]>,Tensor<[1,64,9,9]>,|ttnn.eq|aten::_safe_softmax|4|
|28|Tensor<[1,12,14,14]>,Tensor<[1,12,14,14]>,|ttnn.eq|aten::_safe_softmax|4|
|29|Tensor<[1,12,50,50]>,Tensor<[1,12,50,50]>,|ttnn.eq|aten::_safe_softmax|4|
|30|Tensor<[2,8,7,7]>,Tensor<[2,8,7,7]>,|ttnn.eq|aten::_safe_softmax|4|
|31|Tensor<[197]>,Tensor<[197]>,|ttnn.ge|aten::ge.Scalar|4|
