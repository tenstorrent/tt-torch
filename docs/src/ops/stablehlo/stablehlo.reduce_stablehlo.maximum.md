
### stablehlo.reduce_stablehlo.maximum::ttnn.max


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,32,32,32]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|1|Tensor<[1,12,7,7]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|2|Tensor<[8,920,920]>,<br>Scalar,<br>dim: [2]<br>|ttnn.max|aten::_softmax|4|
|3|Tensor<[8,100,100]>,<br>Scalar,<br>dim: [2]<br>|ttnn.max|aten::_softmax|4|
|4|Tensor<[8,100,920]>,<br>Scalar,<br>dim: [2]<br>|ttnn.max|aten::_softmax|4|
|5|Tensor<[1,12,10,10]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|6|Tensor<[1,8,4096,4096]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|7|Tensor<[1,8,4096,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|8|Tensor<[1,8,1024,1024]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|9|Tensor<[1,8,1024,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|10|Tensor<[1,8,256,256]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|11|Tensor<[1,8,256,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|12|Tensor<[1,8,64,64]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|13|Tensor<[1,8,64,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|14|Tensor<[1,12,25,25]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|15|Tensor<[1,3,1445,1445]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|16|Tensor<[1,12,8,8]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|17|Tensor<[1,8,256,2048]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|18|Tensor<[1,8,2048,256]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|19|Tensor<[1,12,201,201]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|20|Tensor<[1]>,<br>Scalar,<br>dim: [0]<br>|ttnn.max|aten::max|4|
|21|Tensor<[1,10]>,<br>Scalar,<br>dim: [1]<br>|ttnn.max|aten::amax|5|
|22|Tensor<[16,19,19]>,<br>Scalar,<br>dim: [2]<br>|ttnn.max|aten::_softmax|4|
|23|Tensor<[19,256008]>,<br>Scalar,<br>dim: [1]<br>|ttnn.max|aten::amax|5|
|24|Tensor<[1,16,32,32]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|25|Tensor<[1,12,16,16]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|26|Tensor<[1,1,19200,300]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|27|Tensor<[1,2,4800,300]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|28|Tensor<[1,5,1200,300]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|29|Tensor<[1,8,300,300]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|30|Tensor<[1,12,197,197]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|31|Tensor<[1,1,16384,256]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|32|Tensor<[1,2,4096,256]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|33|Tensor<[1,5,1024,256]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
|34|Tensor<[1,71,7,7]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|35|Tensor<[1,12,12,12]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|36|Tensor<[1,12,9,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|37|Tensor<[1,16,9,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|38|Tensor<[1,64,9,9]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|39|Tensor<[1,12,14,14]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|40|Tensor<[1,12,50,50]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|41|Tensor<[2,8,7,7]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_safe_softmax|4|
|42|Tensor<[1,16,197,197]>,<br>Scalar,<br>dim: [3]<br>|ttnn.max|aten::_softmax|4|
