module @SyncTensorsGraph.11 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg1: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}) -> tensor<16384x784xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<8192x784xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<8192x784xf32>
    %2 = "stablehlo.all_gather"(%1) <{all_gather_dim = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<8192x784xf32>) -> tensor<16384x784xf32>
    return %2 : tensor<16384x784xf32>
  }
}

sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg1: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}) -> (tensor<16384x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0", ?}, {?}]>}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : (tensor<f32>) -> tensor<8192x784xf32>
    %1 = stablehlo.add %arg1, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : tensor<8192x784xf32>
    %2 = "stablehlo.all_gather"(%1) <{all_gather_dim = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : (tensor<8192x784xf32>) -> tensor<16384x784xf32>
    return %2 : tensor<16384x784xf32>
  }
}