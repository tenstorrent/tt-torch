
### stablehlo.scatter::ttnn.scatter


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[1,3,720,1280]>,Tensor<[1,1]>,Tensor<[1,3,720,1280]>,update_window_dims: [1, 2, 3]inserted_window_dims: [0]scatter_dims_to_operand_dims: [0]index_vector_dim: 1>|ttnn.scatter|aten::select_scatter|4|
|1|Tensor<[1,720,1280]>,Tensor<[1,1]>,Tensor<[1,720,1280]>,update_window_dims: [1, 2]inserted_window_dims: [0]scatter_dims_to_operand_dims: [0]index_vector_dim: 1>|ttnn.scatter|aten::select_scatter|4|
|2|Tensor<[196,196,2]>,Tensor<[1,1]>,Tensor<[196,196,1]>,update_window_dims: [0, 1]inserted_window_dims: [2]scatter_dims_to_operand_dims: [2]index_vector_dim: 1>|ttnn.scatter|aten::select_scatter|4|
|3|Tensor<[197,197]>,Tensor<[1,1]>,Tensor<[1,197]>,update_window_dims: [1]inserted_window_dims: [0]scatter_dims_to_operand_dims: [0]index_vector_dim: 1>|ttnn.scatter|aten::select_scatter|4|
|4|Tensor<[197,197]>,Tensor<[1,1]>,Tensor<[197,1]>,update_window_dims: [0]inserted_window_dims: [1]scatter_dims_to_operand_dims: [1]index_vector_dim: 1>|ttnn.scatter|aten::select_scatter|4|
|5|Tensor<[197]>,Tensor<[1,1]>,Tensor<[1]>,inserted_window_dims: [0]scatter_dims_to_operand_dims: [0]index_vector_dim: 1>|ttnn.scatter|aten::select_scatter|4|
