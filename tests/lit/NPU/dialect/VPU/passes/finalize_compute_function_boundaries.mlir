//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --split-input-file --mlir-disable-threading --finalize-compute-function-boundaries --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @StaticEltwiseNHWC {
    func.func nested @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>,
                                  %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        return %0 : tensor<1x16x90x1000xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16, {order = #NHWC}> attributes {config.pureHostCompileFunc} {
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
// CHECK: func.func nested @main_func0
// CHECK-SAME: ({{.+}}: tensor<1x90x1000x16xf16>, {{.+}}: tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16>
// CHECK: Core.ReinterpretCast{{.+}} : tensor<1x90x1000x16xf16> -> tensor<1x16x90x1000xf16, {order = #NHWC}>
// CHECK: Core.ReinterpretCast{{.+}} : tensor<1x90x1000x16xf16> -> tensor<1x16x90x1000xf16, {order = #NHWC}>
// CHECK: VPU.NCE.Eltwise
// CHECK: Core.ReinterpretCast{{.+}} : tensor<1x16x90x1000xf16, {order = #NHWC}> -> tensor<1x90x1000x16xf16>
// CHECK: return{{.+}} : tensor<1x90x1000x16xf16>

// CHECK: func.func @main
// CHECK-SAME: ({{.+}}: tensor<1x720x1000x16xf16>, {{.+}}: tensor<1x720x1000x16xf16>) -> tensor<1x720x1000x16xf16>
// CHECK: tensor.empty() : tensor<1x720x1000x16xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice{{.+}}[0, {{.+}}, 0, 0] [1, 90, 1000, 16]
// CHECK: func.call @main_func0
// CHECK: return{{.+}} : tensor<1x720x1000x16xf16>


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @StaticEltwiseMultipleOps {
    func.func nested @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>)
        -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        %1 = VPU.NCE.Eltwise(%0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        %2 = VPU.NCE.Eltwise(%1, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %2 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
    func.func @main(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>)
              -> tensor<1x16x90x1000xf16, {order = #NCHW}> attributes {config.pureHostCompileFunc} {
        %0 = func.call @main_func0(%arg0, %arg1)
           : (tensor<1x16x90x1000xf16, {order = #NHWC}>, tensor<1x16x90x1000xf16, {order = #NHWC}>)
           -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseMultipleOps
// CHECK: func.func nested @main_func0([[ARG0:%.+]]: tensor<1x90x1000x16xf16>, [[ARG1:%.+]]: tensor<1x90x1000x16xf16>)
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
    func.func nested @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NCHW}>) -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = VPU.ReLU(%arg0) : tensor<1x16x90x1000xf16, {order = #NCHW}> -> tensor<1x16x90x1000xf16, {order = #NCHW}>
        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }

    func.func @main(%arg0: tensor<1x16x90x1000xf16, {order = #NCHW}>) -> tensor<1x16x90x1000xf16, {order = #NCHW}> attributes {config.pureHostCompileFunc} {
        %0 = func.call @main_func0(%arg0)
           : (tensor<1x16x90x1000xf16, {order = #NCHW}>)
           -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @StaticEltwiseNCHW
// CHECK: func.func nested @main_func0({{.+}}: tensor<1x16x90x1000xf16>)
// CHECK:     Core.ReinterpretCast{{.+}} : tensor<1x16x90x1000xf16> -> tensor<1x16x90x1000xf16, {order = #NCHW}>
// CHECK:     VPU.ReLU
// CHECK:     Core.ReinterpretCast{{.+}} : tensor<1x16x90x1000xf16, {order = #NCHW}> -> tensor<1x16x90x1000xf16>
// CHECK:     return{{.+}} : tensor<1x16x90x1000xf16>

// CHECK: func.func @main({{.+}}: tensor<1x16x90x1000xf16>)
// CHECK:    call @main_func0({{.+}})
// CHECK:    return {{.+}} : tensor<1x16x90x1000xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (240, -d0 + s0)>

// Check combined case with dynamic shapes + bounds and NHWC layout
module @DynamicBoundedEltwiseNHWC {
  func.func nested @main_func0(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                                 %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
                                  -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> attributes {config.pureHostCompileFunc} {
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
// CHECK: func.func nested @main_func0(
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x256x?x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<1x256x?x16xf16>)
// CHECK-DAG:   Core.ReinterpretCast([[ARG0]]) : tensor<1x256x?x16xf16> -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:   Core.ReinterpretCast([[ARG1]]) : tensor<1x256x?x16xf16> -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       VPU.NCE.Eltwise{{.+}} -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       Core.ReinterpretCast({{.+}}) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x256x?x16xf16>

// CHECK:       return {{.+}} : tensor<1x256x?x16xf16>

// CHECK: func.func @main(
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x256x?x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<1x256x?x16xf16>)
// CHECK-NOT:   bounds
// CHECK:       return {{.+}} : tensor<1x256x?x16xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
#map1 = affine_map<(d0)[s0] -> (s0 - 100, d0)>

// Sub-tile dispatch produced by AdjustBlockSizeForScfTiling: scf.if with i1 condition and tensor results.
// Then-branch: untouched dynamic loop calling the dedicated @main_func0_sub_tile kernel.
// Else-branch: adjusted loop with static tiles calling @main_func0.
// Bounds and layout are kept only inside compute functions, as for every other case in this file.
module @SubTileDispatchIf {
  func.func nested @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>)
                                -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func nested @main_func0_sub_tile(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>)
                                         -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> attributes {sub_tile_fallback} {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> attributes {config.pureHostCompileFunc} {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = arith.cmpi slt, %dim, %c100 : index
    %2 = scf.if %1 -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %3 = scf.for %arg2 = %c0 to %dim step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
        %4 = affine.min #map(%arg2)[%dim]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %4] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
        %5 = func.call @main_func0_sub_tile(%extracted_slice) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %5 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %4] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      } {sub_tile_fallback}
      scf.yield %3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    } else {
      %3 = scf.for %arg2 = %c0 to %dim step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
        %4 = affine.min #map1(%arg2)[%dim]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %4] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {order = #NHWC}>
        %cast = tensor.cast %extracted_slice : tensor<1x16x720x100xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
        %5 = func.call @main_func0(%cast) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
        %cast_0 = tensor.cast %5 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast_0 into %arg3[0, 0, 0, %4] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x100xf16, {order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %2 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }
}

// CHECK-LABEL: @SubTileDispatchIf

// Static-tile kernel: bounds and layout only inside the body
// CHECK:       func.func nested @main_func0([[K_ARG0:%.+]]: tensor<1x720x?x16xf16>) -> tensor<1x720x?x16xf16>
// CHECK:         [[K_IN:%.+]] = Core.ReinterpretCast([[K_ARG0]]) : tensor<1x720x?x16xf16> -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[K_ELTWISE:%.+]] = VPU.NCE.Eltwise([[K_IN]], [[K_IN]]) {{.*}} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[K_OUT:%.+]] = Core.ReinterpretCast([[K_ELTWISE]]) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x720x?x16xf16>
// CHECK:         return [[K_OUT]] : tensor<1x720x?x16xf16>

// Sub-tile kernel: same boundary handling as the static-tile kernel, marker attribute preserved
// CHECK:       func.func nested @main_func0_sub_tile([[ST_ARG0:%.+]]: tensor<1x720x?x16xf16>) -> tensor<1x720x?x16xf16> attributes {sub_tile_fallback}
// CHECK:         [[ST_IN:%.+]] = Core.ReinterpretCast([[ST_ARG0]]) : tensor<1x720x?x16xf16> -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[ST_ELTWISE:%.+]] = VPU.NCE.Eltwise([[ST_IN]], [[ST_IN]]) {{.*}} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[ST_OUT:%.+]] = Core.ReinterpretCast([[ST_ELTWISE]]) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x720x?x16xf16>
// CHECK:         return [[ST_OUT]] : tensor<1x720x?x16xf16>

// Main function: no bounds anywhere, scf.if dispatch preserved with memory-order types
// CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x720x?x16xf16>) -> tensor<1x720x?x16xf16>
// CHECK-NOT:     bounds
// CHECK:         [[DIM:%.+]] = tensor.dim [[ARG0]], {{%.+}} : tensor<1x720x?x16xf16>
// CHECK:         [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x720x?x16xf16>
// CHECK:         [[COND:%.+]] = arith.cmpi slt, [[DIM]], {{%.+}} : index
// CHECK:         [[IF:%.+]] = scf.if [[COND]] -> (tensor<1x720x?x16xf16>) {
// CHECK:           [[SUB_FOR:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[EMPTY]]) -> (tensor<1x720x?x16xf16>) {
// CHECK:             [[SUB_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, {{%[^,]+}}, 0] [1, 720, {{%[^,]+}}, 16] [1, 1, 1, 1] : tensor<1x720x?x16xf16> to tensor<1x720x?x16xf16>
// CHECK:             [[SUB_CALL:%.+]] = func.call @main_func0_sub_tile([[SUB_SLICE]]) : (tensor<1x720x?x16xf16>) -> tensor<1x720x?x16xf16>
// CHECK:             tensor.insert_slice [[SUB_CALL]] into {{.*}}[0, 0, {{%[^,]+}}, 0] [1, 720, {{%[^,]+}}, 16] [1, 1, 1, 1]
// CHECK:           } {sub_tile_fallback}
// CHECK:           scf.yield [[SUB_FOR]] : tensor<1x720x?x16xf16>
// CHECK:         } else {
// CHECK:           [[FOR:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[EMPTY]]) -> (tensor<1x720x?x16xf16>) {
// CHECK:             [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, {{%[^,]+}}, 0] [1, 720, 100, 16] [1, 1, 1, 1] : tensor<1x720x?x16xf16> to tensor<1x720x100x16xf16>
// CHECK:             [[CAST:%.+]] = tensor.cast [[SLICE]] : tensor<1x720x100x16xf16> to tensor<1x720x?x16xf16>
// CHECK:             [[CALL:%.+]] = func.call @main_func0([[CAST]]) : (tensor<1x720x?x16xf16>) -> tensor<1x720x?x16xf16>
// CHECK:             [[CAST_OUT:%.+]] = tensor.cast [[CALL]] : tensor<1x720x?x16xf16> to tensor<1x720x100x16xf16>
// CHECK:             tensor.insert_slice [[CAST_OUT]] into {{.*}}[0, 0, {{%[^,]+}}, 0] [1, 720, 100, 16] [1, 1, 1, 1]
// CHECK:           }
// CHECK:           scf.yield [[FOR]] : tensor<1x720x?x16xf16>
// CHECK:         }
// CHECK:         return [[IF]] : tensor<1x720x?x16xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TensorDimForDynamicDim(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> index attributes {config.pureHostCompileFunc} {
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
func.func @TensorDimForTwoDynamicDims(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> (index, index) attributes {config.pureHostCompileFunc} {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim_0 = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %dim_1 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    return %dim_0, %dim_1 : index, index
    // CHECK: func.func @TensorDimForTwoDynamicDims([[ARG:%.+]]: tensor<1x?x?x16xf16>)
    // CHECK-DAG:   [[C2:%.+]] = arith.constant 2 : index
    // CHECK-DAG:   [[C1:%.+]] = arith.constant 1 : index
    // CHECK:   [[DIM0:%.+]] = tensor.dim [[ARG]], [[C1]] : tensor<1x?x?x16xf16>
    // CHECK:   [[DIM1:%.+]] = tensor.dim [[ARG]], [[C2]] : tensor<1x?x?x16xf16>
    // CHECK:   return [[DIM0]], [[DIM1]] : index, index
}

// -----

#WHCN = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>

func.func @TensorDimForDynamicDimWHCN(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #WHCN}>) -> index attributes {config.pureHostCompileFunc} {
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #WHCN}>
    return %dim : index
    // CHECK: func.func @TensorDimForDynamicDimWHCN([[ARG:%.+]]: tensor<?x256x16x1xf16>)
    // CHECK:   [[C0:%.+]] = arith.constant 0 : index
    // CHECK:   [[DIM:%.+]] = tensor.dim [[ARG]], [[C0]] : tensor<?x256x16x1xf16>
    // CHECK:   return [[DIM]] : index
}

// -----

// Test validates that quantized types are converted to storage types in function signatures

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @StaticQuantizedNHWC {
    func.func nested @main_func0(%arg0: tensor<1x16x90x1000x!qElemType, {order = #NHWC}>,
                                  %arg1: tensor<1x16x90x1000x!qElemType, {order = #NHWC}>)
                                  -> tensor<1x16x90x1000x!qElemType, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
        return %0 : tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
    }

    func.func @main(%arg0: tensor<1x16x720x1000x!qElemType, {order = #NHWC}>,
                    %arg1: tensor<1x16x720x1000x!qElemType, {order = #NHWC}>)
                    -> tensor<1x16x720x1000x!qElemType, {order = #NHWC}> attributes {config.pureHostCompileFunc}  {
        %c90 = arith.constant 90 : index
        %c720 = arith.constant 720 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<1x16x720x1000x!qElemType, {order = #NHWC}>
        %1 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %0) -> (tensor<1x16x720x1000x!qElemType, {order = #NHWC}>) {
            %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000x!qElemType, {order = #NHWC}> to tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
            %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000x!qElemType, {order = #NHWC}> to tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
            %2 = func.call @main_func0(%extracted_slice, %extracted_slice_0) : (tensor<1x16x90x1000x!qElemType, {order = #NHWC}>, tensor<1x16x90x1000x!qElemType, {order = #NHWC}>) -> tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
            %inserted_slice = tensor.insert_slice %2 into %arg3[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x90x1000x!qElemType, {order = #NHWC}> into tensor<1x16x720x1000x!qElemType, {order = #NHWC}>
            scf.yield %inserted_slice : tensor<1x16x720x1000x!qElemType, {order = #NHWC}>
        }
        return %1 : tensor<1x16x720x1000x!qElemType, {order = #NHWC}>
    }
}

// CHECK-LABEL: @StaticQuantizedNHWC
// CHECK: func.func nested @main_func0([[ARG0:%.+]]: tensor<1x90x1000x16xi8>, [[ARG1:%.+]]: tensor<1x90x1000x16xi8>) -> tensor<1x90x1000x16xi8>
// CHECK-DAG:   [[CAST0:%.+]] = Core.ReinterpretCast([[ARG1]]) : tensor<1x90x1000x16xi8> -> tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
// CHECK-DAG:   [[CAST1:%.+]] = Core.ReinterpretCast([[ARG0]]) : tensor<1x90x1000x16xi8> -> tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
// CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CAST1]], [[CAST0]]) {{.*}} -> tensor<1x16x90x1000x!qElemType, {order = #NHWC}>
// CHECK:       [[CAST2:%.+]] = Core.ReinterpretCast([[ELTWISE]]) : tensor<1x16x90x1000x!qElemType, {order = #NHWC}> -> tensor<1x90x1000x16xi8>
// CHECK:       return [[CAST2]] : tensor<1x90x1000x16xi8>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x720x1000x16xi8>, [[ARG1:%.+]]: tensor<1x720x1000x16xi8>) -> tensor<1x720x1000x16xi8>
// CHECK:       tensor.empty() : tensor<1x720x1000x16xi8>
// CHECK:       scf.for
// CHECK:         tensor.extract_slice{{.*}}[0, {{%[^,]+}}, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:         tensor.extract_slice{{.*}}[0, {{%[^,]+}}, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:         func.call @main_func0
// CHECK:       return{{.*}} : tensor<1x720x1000x16xi8>
