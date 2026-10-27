//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --scf-vertical-fusion="vf-merge-configuration=GREEDY vf-merge-cost-mode=power" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!q = !quant.uniform<u8:f16, 0.0033894703668706557:146>

module @executors {
  config.Resources 3 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1571840 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }


// CHECK-LABEL: @SCFVerticalFusionWithPowerMode
// CHECK-SAME:      ([[ARG0:%arg[0-9]+]]: tensor<1x32x800x1280x[[QTYPE:![a-zA-Z0-9]+]]
func.func @SCFVerticalFusionWithPowerMode(%arg0: tensor<1x32x800x1280x!q, {order = #NHWC}>) -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
    %weights3x3 = const.Declare tensor<32x32x3x3x!q, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<!q>, #const.Reorder<#NHWC>]
    %weights1x1 = const.Declare tensor<32x32x1x1x!q, {order = #NHWC}> = dense<1> : tensor<32x32x1x1xui8>, [#const.CastElemType<!q>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %4 = VPU.NCE.Eltwise(%2, %3) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %6 = VPU.NCE.Eltwise(%4, %5) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %8 = VPU.NCE.Eltwise(%6, %7) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %9 = VPU.NCE.Convolution(%8, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %10 = VPU.NCE.Eltwise(%8, %9) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %11 = VPU.NCE.Convolution(%10, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %12 = VPU.NCE.Eltwise(%10, %11) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %13 = VPU.NCE.Convolution(%12, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %14 = VPU.NCE.Convolution(%0, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %15 = VPU.NCE.Convolution(%14, %weights1x1) rawFilterShape [32, 32, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 28]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x1x1x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %16 = VPU.NCE.Convolution(%13, %weights3x3) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 17, 4]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    %17 = VPU.NCE.Eltwise(%16, %15) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
    return %17 : tensor<1x32x800x1280x!q, {order = #NHWC}>

    // The fused region is tiled over W in chunks of 4. Intermediate 3x3 convolution tiles have
    // dynamic widths because halo sizes vary at the boundaries; the final output tile is static.
    // CHECK:       [[TILE_WIDTH:%c[0-9]+]] = arith.constant 4 : index
    // CHECK:       [[FULL_WIDTH:%c[0-9]+]] = arith.constant 1280 : index
    // CHECK:       [[ZERO:%c[0-9]+]] = arith.constant 0 : index
    // CHECK:       [[EMPTY:%[0-9]+]] = tensor.empty() : tensor<1x32x800x1280x[[QTYPE]]
    // CHECK:       [[LOOP:%[0-9]+]] = scf.for [[TILE_OFFSET:%arg[0-9]+]] = [[ZERO]] to [[FULL_WIDTH]] step [[TILE_WIDTH]] iter_args([[LOOP_OUT:%arg[0-9]+]] = [[EMPTY]])
    // CHECK:         [[INPUT_SLICE:%.*]] = tensor.extract_slice [[ARG0]]
    // CHECK-SAME:        to tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV0:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV1:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[SKIP0:%.*]] = tensor.extract_slice [[CONV0]]{{.*}}skip_connection_slice
    // CHECK:         [[ELTWISE0:%.*]] = VPU.NCE.Eltwise([[SKIP0]], [[CONV1]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV2:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[SKIP1:%.*]] = tensor.extract_slice [[ELTWISE0]]{{.*}}skip_connection_slice
    // CHECK:         [[ELTWISE1:%.*]] = VPU.NCE.Eltwise([[SKIP1]], [[CONV2]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV3:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[SKIP2:%.*]] = tensor.extract_slice [[ELTWISE1]]{{.*}}skip_connection_slice
    // CHECK:         [[ELTWISE2:%.*]] = VPU.NCE.Eltwise([[SKIP2]], [[CONV3]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV4:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[SKIP3:%.*]] = tensor.extract_slice [[ELTWISE2]]{{.*}}skip_connection_slice
    // CHECK:         [[ELTWISE3:%.*]] = VPU.NCE.Eltwise([[SKIP3]], [[CONV4]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV5:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[SKIP4:%.*]] = tensor.extract_slice [[ELTWISE3]]{{.*}}skip_connection_slice
    // CHECK:         [[ELTWISE4:%.*]] = VPU.NCE.Eltwise([[SKIP4]], [[CONV5]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV6:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[SKIP5:%.*]] = tensor.extract_slice [[ELTWISE4]]{{.*}}skip_connection_slice
    // CHECK:         [[ELTWISE5:%.*]] = VPU.NCE.Eltwise([[SKIP5]], [[CONV6]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[BRANCH_SLICE:%.*]] = tensor.extract_slice [[CONV0]]{{.*}}skip_connection_slice
    // CHECK:         [[BRANCH_CONV0:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[BRANCH_TILE:%.*]] = tensor.cast [[BRANCH_CONV0]]
    // CHECK-SAME:        to tensor<1x32x800x4x[[QTYPE]]
    // CHECK:         [[BRANCH_CONV1:%.*]] = VPU.NCE.Convolution([[BRANCH_TILE]]
    // CHECK-SAME:        rawFilterShape [32, 32, 1, 1]
    // CHECK-SAME:        -> tensor<1x32x800x4x[[QTYPE]]
    // CHECK:         [[CONV7:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[CONV8:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:        rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE]]
    // CHECK:         [[MAIN_TILE:%.*]] = tensor.cast [[CONV8]]
    // CHECK-SAME:        to tensor<1x32x800x4x[[QTYPE]]
    // CHECK:         [[RESULT_TILE:%.*]] = VPU.NCE.Eltwise([[MAIN_TILE]], [[BRANCH_CONV1]])
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        -> tensor<1x32x800x4x[[QTYPE]]
    // CHECK:         [[INSERT:%.*]] = tensor.insert_slice [[RESULT_TILE]] into [[LOOP_OUT]][0, 0, 0, [[TILE_OFFSET]]] [1, 32, 800, 4]
    // CHECK:         scf.yield [[INSERT]] : tensor<1x32x800x1280x[[QTYPE]]
    // CHECK:       return [[LOOP]] : tensor<1x32x800x1280x[[QTYPE]]
}
}
