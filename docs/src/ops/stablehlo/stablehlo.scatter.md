
### stablehlo.scatter::ttnn.scatter


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,3,720,1280]>,<br>Tensor<[1,1]>,<br>Tensor<[1,3,720,1280]>,<br>update_window_dims: [1, 2, 3]<br>inserted_window_dims: [0]<br>scatter_dims_to_operand_dims: [0]<br>index_vector_dim: 1><br>|ttnn.scatter|aten::select_scatter|4|
|1|Tensor<[1,720,1280]>,<br>Tensor<[1,1]>,<br>Tensor<[1,720,1280]>,<br>update_window_dims: [1, 2]<br>inserted_window_dims: [0]<br>scatter_dims_to_operand_dims: [0]<br>index_vector_dim: 1><br>|ttnn.scatter|aten::select_scatter|4|
|2|Tensor<[196,196,2]>,<br>Tensor<[1,1]>,<br>Tensor<[196,196,1]>,<br>update_window_dims: [0, 1]<br>inserted_window_dims: [2]<br>scatter_dims_to_operand_dims: [2]<br>index_vector_dim: 1><br>|ttnn.scatter|aten::select_scatter|4|
|3|Tensor<[197,197]>,<br>Tensor<[1,1]>,<br>Tensor<[1,197]>,<br>update_window_dims: [1]<br>inserted_window_dims: [0]<br>scatter_dims_to_operand_dims: [0]<br>index_vector_dim: 1><br>|ttnn.scatter|aten::select_scatter|4|
|4|Tensor<[197,197]>,<br>Tensor<[1,1]>,<br>Tensor<[197,1]>,<br>update_window_dims: [0]<br>inserted_window_dims: [1]<br>scatter_dims_to_operand_dims: [1]<br>index_vector_dim: 1><br>|ttnn.scatter|aten::select_scatter|4|
|5|Tensor<[197]>,<br>Tensor<[1,1]>,<br>Tensor<[1]>,<br>inserted_window_dims: [0]<br>scatter_dims_to_operand_dims: [0]<br>index_vector_dim: 1><br>|ttnn.scatter|aten::select_scatter|4|
