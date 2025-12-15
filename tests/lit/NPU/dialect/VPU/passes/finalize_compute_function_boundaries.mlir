//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-disable-threading --finalize-compute-function-boundaries %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @StaticEltwiseNHWC {
    func.func private @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>,
                                  %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        return %0 : tensor<1x16x90x1000xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16, {order = #NHWC}> {
        %c90 = arith.constant 90 : index
        %c720 = arith.constant 720 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
        %1 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %0) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
            %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16, {order = #NHWC}> to tensor<1x16x90x1000xf16, {order = #NHWC}>
            %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16, {order = #NHWC}> to tensor<1x16x90x1000xf16, {order = #NHWC}>
            %2 = func.call @main_func0(%extracted_slice, %extracted_slice_0) : (tensor<1x16x90x1000xf16, {order = #NHWC}>, tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}>
            %inserted_slice = tensor.insert_slice %2 into %arg3[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x90x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
            scf.yield %inserted_slice : tensor<1x16x720x1000xf16, {order = #NHWC}>
        }
        return %1 : tensor<1x16x720x1000xf16, {order = #NHWC}>
    }
}

// CHECK: #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseNHWC
// CHECK: func.func private @main_func0
// CHECK-SAME: (%{{.*}}: tensor<1x90x1000x16xf16>, %{{.*}}: tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16>
// CHECK: Core.ReinterpretCast{{.*}} : tensor<1x90x1000x16xf16> -> tensor<1x16x90x1000xf16, {order = #NHWC}>
// CHECK: Core.ReinterpretCast{{.*}} : tensor<1x90x1000x16xf16> -> tensor<1x16x90x1000xf16, {order = #NHWC}>
// CHECK: VPU.NCE.Eltwise
// CHECK: Core.ReinterpretCast{{.*}} : tensor<1x16x90x1000xf16, {order = #NHWC}> -> tensor<1x90x1000x16xf16>
// CHECK: return{{.*}} : tensor<1x90x1000x16xf16>

// CHECK: func.func @main
// CHECK-SAME: (%{{.*}}: tensor<1x720x1000x16xf16>, %{{.*}}: tensor<1x720x1000x16xf16>) -> tensor<1x720x1000x16xf16>
// CHECK: tensor.empty() : tensor<1x720x1000x16xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice{{.*}}[0, %{{.*}}, 0, 0] [1, 90, 1000, 16]
// CHECK: func.call @main_func0
// CHECK: return{{.*}} : tensor<1x720x1000x16xf16>


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @StaticEltwiseMultipleOps {
    func.func private @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>)
        -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        %1 = VPU.NCE.Eltwise(%0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        %2 = VPU.NCE.Eltwise(%1, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %2 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
    func.func @main(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>)
              -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = func.call @main_func0(%arg0, %arg1)
           : (tensor<1x16x90x1000xf16, {order = #NHWC}>, tensor<1x16x90x1000xf16, {order = #NHWC}>)
           -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseMultipleOps
// CHECK: func.func private @main_func0([[ARG0:%.+]]: tensor<1x90x1000x16xf16>, [[ARG1:%.+]]: tensor<1x90x1000x16xf16>)
// CHECK-DAG: [[CAST0:%.+]] = Core.ReinterpretCast([[ARG0]]) : tensor<1x90x1000x16xf16> -> tensor<1x16x90x1000xf16, {order = #NHWC}>
// CHECK-DAG: [[CAST1:%.+]] = Core.ReinterpretCast([[ARG1]]) : tensor<1x90x1000x16xf16> -> tensor<1x16x90x1000xf16, {order = #NHWC}>
// CHECK:     [[ADD0:%.+]] = VPU.NCE.Eltwise([[CAST0]], [[CAST1]])
// CHECK:     [[ADD1:%.+]] = VPU.NCE.Eltwise([[ADD0]], [[CAST1]])
// CHECK:     [[ADD2:%.+]] = VPU.NCE.Eltwise([[ADD1]], [[CAST1]])
// CHECK:     [[CAST2:%.+]] = Core.ReinterpretCast([[ADD2]]) : tensor<1x16x90x1000xf16, {order = #NCHW}> -> tensor<1x16x90x1000xf16>
// CHECK:     return [[CAST2]] : tensor<1x16x90x1000xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x90x1000x16xf16>, [[ARG1:%.+]]: tensor<1x90x1000x16xf16>)
// CHECK:    [[CALL:%.+]] = call @main_func0([[ARG0]], [[ARG1]])
// CHECK:    return [[CALL]] : tensor<1x16x90x1000xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @StaticEltwiseNCHW {
    func.func private @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NCHW}>) -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = VPU.ReLU(%arg0) : tensor<1x16x90x1000xf16, {order = #NCHW}> -> tensor<1x16x90x1000xf16, {order = #NCHW}>
        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }

    func.func @main(%arg0: tensor<1x16x90x1000xf16, {order = #NCHW}>) -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = func.call @main_func0(%arg0)
           : (tensor<1x16x90x1000xf16, {order = #NCHW}>)
           -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @StaticEltwiseNCHW
// CHECK: func.func private @main_func0(%{{.*}}: tensor<1x16x90x1000xf16>)
// CHECK:     Core.ReinterpretCast{{.*}} : tensor<1x16x90x1000xf16> -> tensor<1x16x90x1000xf16, {order = #NCHW}>
// CHECK:     VPU.ReLU
// CHECK:     Core.ReinterpretCast{{.*}} : tensor<1x16x90x1000xf16, {order = #NCHW}> -> tensor<1x16x90x1000xf16>
// CHECK:     return{{.*}} : tensor<1x16x90x1000xf16>

// CHECK: func.func @main(%{{.*}}: tensor<1x16x90x1000xf16>)
// CHECK:    call @main_func0({{.*}})
// CHECK:    return {{.*}} : tensor<1x16x90x1000xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (240, -d0 + s0)>

// Check combined case with dynamic shapes + bounds and NHWC layout
module @DynamicBoundedEltwiseNHWC {
  func.func private @main_func0(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                                 %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
                                  -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c3_0 = arith.constant 3 : index
    %dim_1 = tensor.dim %arg0, %c3_0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %c240 = arith.constant 240 : index
    %1 = scf.for %arg2 = %c0 to %dim_1 step %c240 iter_args(%arg3 = %0) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim_1]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_2 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice, %extracted_slice_2) : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
  }
}

// Expected that information about bounds and layout is preserved only inside compute function,
//  and not part of the main function or compute function signature.

// CHECK-LABEL: @DynamicBoundedEltwiseNHWC
// CHECK: func.func private @main_func0(
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x256x?x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<1x256x?x16xf16>)
// CHECK-DAG:   Core.ReinterpretCast([[ARG0]]) : tensor<1x256x?x16xf16> -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:   Core.ReinterpretCast([[ARG1]]) : tensor<1x256x?x16xf16> -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       VPU.NCE.Eltwise{{.*}} -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       Core.ReinterpretCast({{.*}}) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x256x?x16xf16>

// CHECK:       return {{.*}} : tensor<1x256x?x16xf16>

// CHECK: func.func @main(
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x256x?x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<1x256x?x16xf16>)
// CHECK-NOT:   bounds
// CHECK:       return {{.*}} : tensor<1x256x?x16xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TensorDimForDynamicDim(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> index {
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    return %dim : index
    // CHECK: func.func @TensorDimForDynamicDim([[ARG:%.+]]: tensor<1x256x?x16xf16>)
    // CHECK:   [[C2:%.+]] = arith.constant 2 : index
    // CHECK:   [[DIM:%.+]] = tensor.dim [[ARG]], [[C2]] : tensor<1x256x?x16xf16>
    // CHECK:   return [[DIM]] : index
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @TensorDimForTwoDynamicDims(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> (index, index) {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim_0 = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %dim_1 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    return %dim_0, %dim_1 : index, index
    // CHECK: func.func @TensorDimForTwoDynamicDims([[ARG:%.+]]: tensor<1x?x?x16xf16>)
    // CHECK:   [[C1:%.+]] = arith.constant 1 : index
    // CHECK:   [[DIM0:%.+]] = tensor.dim [[ARG]], [[C1]] : tensor<1x?x?x16xf16>
    // CHECK:   [[C2:%.+]] = arith.constant 2 : index
    // CHECK:   [[DIM1:%.+]] = tensor.dim [[ARG]], [[C2]] : tensor<1x?x?x16xf16>
    // CHECK:   return [[DIM0]], [[DIM1]] : index, index
}
// -----

#WHCN = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>

func.func @TensorDimForDynamicDimWHCN(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #WHCN}>) -> index {
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #WHCN}>
    return %dim : index
    // CHECK: func.func @TensorDimForDynamicDimWHCN([[ARG:%.+]]: tensor<?x256x16x1xf16>)
    // CHECK:   [[C0:%.+]] = arith.constant 0 : index
    // CHECK:   [[DIM:%.+]] = tensor.dim [[ARG]], [[C0]] : tensor<?x256x16x1xf16>
    // CHECK:   return [[DIM]] : index
}
