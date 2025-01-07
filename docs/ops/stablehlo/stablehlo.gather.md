
### stablehlo.gather::ttnn.embedding


||STABLE HLO Input Variations|ttnn op|Torch Name|Status|
| :--- | :--- | :--- | :--- | :--- |
|0|Tensor<[32000,4096]>,Tensor<[1,32]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|1|Tensor<[50257,768]>,Tensor<[1,7]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|2|Tensor<[1024,768]>,Tensor<[1,7]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|3|Tensor<[1,7,2]>,Tensor<[1,2]>,offset_dims: [1]collapsed_slice_dims: [0, 1]start_index_map: [0, 1]index_vector_dim: 1indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|4|Tensor<[1,1,720,1280]>,Tensor<[1,1,23,40,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|5|Tensor<[250002,768]>,Tensor<[1,10]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|6|Tensor<[1,768]>,Tensor<[1,10]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|7|Tensor<[514,768]>,Tensor<[1,10]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|8|Tensor<[1,1280,8,8]>,Tensor<[1,1280,16,16,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|9|Tensor<[1,1280,16,16]>,Tensor<[1,1280,32,32,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|10|Tensor<[1,640,32,32]>,Tensor<[1,640,64,64,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|11|Tensor<[30522,768]>,Tensor<[1,25]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|12|Tensor<[2,768]>,Tensor<[1,25]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|13|Tensor<[512,768]>,Tensor<[1,25]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|14|Tensor<[30528,768]>,Tensor<[1,8]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|15|Tensor<[512,768]>,Tensor<[1,8]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|16|Tensor<[2,768]>,Tensor<[1,8]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|17|Tensor<[262,768]>,Tensor<[1,2048]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|18|Tensor<[2048,768]>,Tensor<[2048]>,offset_dims: [1]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 1indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|19|Tensor<[30522,768]>,Tensor<[1,8]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|20|Tensor<[40,768]>,Tensor<[1,8]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|21|Tensor<[2,768]>,Tensor<[1,193]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|22|Tensor<[1,1,384,512]>,Tensor<[1,1,12,16,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|23|Tensor<[256008,1024]>,Tensor<[1,19]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|24|Tensor<[19,256008]>,Tensor<[19,1,2]>,collapsed_slice_dims: [0, 1]start_index_map: [0, 1]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::gather|4|
|25|Tensor<[2050,1024]>,Tensor<[19]>,offset_dims: [1]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 1indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index_select|4|
|26|Tensor<[1,256,16,16]>,Tensor<[1,256,32,32,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|27|Tensor<[1,128,32,32]>,Tensor<[1,128,64,64,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|28|Tensor<[250880,1536]>,Tensor<[1,32]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|29|Tensor<[30522,768]>,Tensor<[1,16]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|30|Tensor<[512,768]>,Tensor<[1,16]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|31|Tensor<[1,64,15,20]>,Tensor<[1,64,30,40,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|32|Tensor<[1,64,30,40]>,Tensor<[1,64,60,80,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|33|Tensor<[1,64,60,80]>,Tensor<[1,64,120,160,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|34|Tensor<[1,64,120,160]>,Tensor<[1,64,240,320,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|35|Tensor<[1,64,240,320]>,Tensor<[1,64,480,640,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|36|Tensor<[1,256,128,128]>,Tensor<[1,256,128,128,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|37|Tensor<[1,256,64,64]>,Tensor<[1,256,128,128,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|38|Tensor<[1,256,32,32]>,Tensor<[1,256,128,128,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|39|Tensor<[1,256,16,16]>,Tensor<[1,256,128,128,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|40|Tensor<[65024,4544]>,Tensor<[1,7]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|41|Tensor<[1,7,73,64]>,Tensor<[1,7,1,64,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|42|Tensor<[30000,128]>,Tensor<[1,12]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|43|Tensor<[2,128]>,Tensor<[1,12]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|44|Tensor<[512,128]>,Tensor<[1,12]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|45|Tensor<[30000,128]>,Tensor<[1,9]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|46|Tensor<[2,128]>,Tensor<[1,9]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|47|Tensor<[512,128]>,Tensor<[1,9]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|48|Tensor<[30000,128]>,Tensor<[1,14]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|49|Tensor<[2,128]>,Tensor<[1,14]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|50|Tensor<[512,128]>,Tensor<[1,14]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|51|Tensor<[50,768]>,Tensor<[1,50]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|52|Tensor<[49408,512]>,Tensor<[2,7]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|53|Tensor<[77,512]>,Tensor<[1,7]>,offset_dims: [2]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 2indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::embedding|4|
|54|Tensor<[2,7,512]>,Tensor<[2,2]>,offset_dims: [1]collapsed_slice_dims: [0, 1]start_index_map: [0, 1]index_vector_dim: 1indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|55|Tensor<[1,16,27,27]>,Tensor<[1,16,27,27,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|56|Tensor<[732,16]>,Tensor<[38809,1]>,offset_dims: [1]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 1indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|57|Tensor<[1,12,27,27]>,Tensor<[1,12,27,27,4]>,collapsed_slice_dims: [0, 1, 2, 3]start_index_map: [0, 1, 2, 3]index_vector_dim: 4indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
|58|Tensor<[732,12]>,Tensor<[38809,1]>,offset_dims: [1]collapsed_slice_dims: [0]start_index_map: [0]index_vector_dim: 1indices_are_sorted: falseslice_sizes: array<i64|ttnn.embedding|aten::index.Tensor|4|
