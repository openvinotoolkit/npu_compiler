//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvReduceOutToSubtract
module @ConvReduceOutToSubtract {
func.func @main(%arg0: tensor<1x1024x4x48xf16>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
  %bias_table = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>
  %scale_table = const.Declare tensor<4096x1x1x1xf32> = dense<1.000000e+00> : tensor<4096x1x1x1xf32>
  %weights = const.Declare tensor<4096x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4096x48x1x1xf16>, [#const.Reorder<#NHWC>]
  %perm_cast = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW}
    : tensor<1x1024x4x48xf16> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
  %vf:2 = VPU.VerticalFusion (%perm_cast as %arg1: tensor<1x48x1024x4xf16, {order = #NHWC}>,
                              %weights as %arg2: tensor<4096x48x1x1xf16, {order = #NHWC}>,
                              %scale_table as %arg3: tensor<4096x1x1x1xf32>,
                              %bias_table as %arg4: tensor<4096x1x1x1xf32>) attributes {tilingStrategy = [1, 1, 32, 1]}
        -> (tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x1x1024x4xf16, {order = #NHWC}>) {
    %out, %reduce = VPU.NCE.Convolution(%arg1, %arg2, %arg3, %arg4) rawFilterShape [4096, 48, 1, 1] {
      axes_value = [1], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>, strides = [1, 1]
    } : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32>
      -> tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x1x1024x4xf16, {order = #NHWC}>
    VPU.Yield %out, %reduce
  }

  %subtract_vf = VPU.VerticalFusion (%vf#0 as %arg1: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
                                    %vf#1 as %arg2: tensor<1x1x1024x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 19, 1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %sub = VPU.Subtract(%arg1, %arg2)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x1x1024x4xf16, {order = #NHWC}>
      -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    VPU.Yield %sub
  }
    //  CHECK-DAG: [[CST:%.+]] = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00>
    //  CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<4096x1x1x1xf32> = dense<1.000000e+00>
    //  CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<4096x48x1x1xf16
    //  CHECK: [[PERM:%.+]] = VPU.PermuteCast
    //  CHECK: [[VF:%.+]] = VPU.VerticalFusion ([[PERM]] as {{%.+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[CST_1]] as {{%.+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[CST_0]] as {{%.+}}: tensor<4096x1x1x1xf32>, [[CST]] as {{%.+}}: tensor<4096x1x1x1xf32>)
    //  CHECK-SAME: scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    //  CHECK-SAME: -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    //  CHECK-NEXT: [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
    //  CHECK-NEXT: [[SUB:%.+]] = VPU.Subtract([[CONV_OUT]], [[REDUCE_OUT]])
    //  CHECK-NEXT: VPU.Yield [[SUB]]
    //  CHECK: return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
    return %subtract_vf : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolReduceOutToSubtract
module @MaxPoolReduceOutToSubtract {
func.func @main(%arg0: tensor<1x256x1024x4xf16, {order = #NHWC}>) -> tensor<1x256x1024x4xf16, {order = #NHWC}> {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x1024x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 16, 1]}
        -> (tensor<1x256x1024x4xf16, {order = #NHWC}>, tensor<1x1x1024x4xf16, {order = #NHWC}>) {
    %out, %reduce = VPU.NCE.MaxPool(%arg1) {
      axes_value = [1],
      kernel_size = [1, 1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>, strides = [1, 1]
    } : tensor<1x256x1024x4xf16, {order = #NHWC}>
      -> tensor<1x256x1024x4xf16, {order = #NHWC}>, tensor<1x1x1024x4xf16, {order = #NHWC}>
    VPU.Yield %out, %reduce
  }

  %subtract_vf = VPU.VerticalFusion (%vf#0 as %arg1: tensor<1x256x1024x4xf16, {order = #NHWC}>,
                                     %vf#1 as %arg2: tensor<1x1x1024x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 16, 1, 1]} -> tensor<1x256x1024x4xf16, {order = #NHWC}> {
    %sub = VPU.Subtract(%arg1, %arg2)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x256x1024x4xf16, {order = #NHWC}>, tensor<1x1x1024x4xf16, {order = #NHWC}>
      -> tensor<1x256x1024x4xf16, {order = #NHWC}>
    VPU.Yield %sub
  }
  //  CHECK: [[VF:%.+]] = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x256x1024x4xf16, {order = #NHWC}>)
  //  CHECK-SAME: -> tensor<1x256x1024x4xf16, {order = #NHWC}>
  //  CHECK-NEXT: [[POOL_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.MaxPool
  //  CHECK-NEXT: [[SUB:%.+]] = VPU.Subtract([[POOL_OUT]], [[REDUCE_OUT]])
  //  CHECK-NEXT: VPU.Yield [[SUB]]
  //  CHECK: return [[VF]] : tensor<1x256x1024x4xf16, {order = #NHWC}>
  return %subtract_vf : tensor<1x256x1024x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvBothOutputsToConvertNotMerged
module @ConvBothOutputsToConvertNotMerged {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    -> (tensor<1x32x160x160xf32, {order = #NHWC}>, tensor<1x1x160x160xf32, {order = #NHWC}>) {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmin = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 0, 1, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmin
  }

  %cvt_main_vf = VPU.VerticalFusion (%vf#0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x32x160x160xf32, {order = #NHWC}> {
    %cvt_main = VPU.Convert(%arg2) {dstElemType = f32,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf16, {order = #NHWC}> -> tensor<1x32x160x160xf32, {order = #NHWC}>
    VPU.Yield %cvt_main
  }
  %cvt_red_vf = VPU.VerticalFusion (%vf#1 as %arg2: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x1x160x160xf32, {order = #NHWC}> {
    %cvt_red = VPU.Convert(%arg2) {dstElemType = f32,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x1x160x160xf16, {order = #NHWC}> -> tensor<1x1x160x160xf32, {order = #NHWC}>
    VPU.Yield %cvt_red
  }
  //  CHECK: [[VF:%.+]]:2 = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME:        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-NEXT:   [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT:   VPU.Yield [[CONV_OUT]], [[REDUCE_OUT]]

  //  CHECK: [[CVT_MAIN_VF:%.+]] = VPU.VerticalFusion ([[VF]]#0 as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>)
  //  CHECK-SAME:         -> tensor<1x32x160x160xf32, {order = #NHWC}>
  //  CHECK-NEXT:   [[CVT_MAIN:%.+]] = VPU.Convert({{%.+}})
  //  CHECK-NEXT:   VPU.Yield [[CVT_MAIN]]

  //  CHECK: [[CVT_RED_VF:%.+]] = VPU.VerticalFusion ([[VF]]#1 as {{%.+}}: tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-SAME:         -> tensor<1x1x160x160xf32, {order = #NHWC}>
  //  CHECK-NEXT:   [[CVT_RED:%.+]] = VPU.Convert({{%.+}})
  //  CHECK-NEXT:   VPU.Yield [[CVT_RED]]

  //  CHECK: return [[CVT_MAIN_VF]], [[CVT_RED_VF]]
  return %cvt_main_vf, %cvt_red_vf : tensor<1x32x160x160xf32, {order = #NHWC}>, tensor<1x1x160x160xf32, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvMainOutputToConvertNotMerged
module @ConvMainOutputToConvertNotMerged {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    -> (tensor<1x32x160x160xf32, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmin = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 0, 1, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmin
  }

  %cvt_vf = VPU.VerticalFusion (%vf#0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x32x160x160xf32, {order = #NHWC}> {
    %cvt = VPU.Convert(%arg2) {dstElemType = f32,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf16, {order = #NHWC}> -> tensor<1x32x160x160xf32, {order = #NHWC}>
    VPU.Yield %cvt
  }
  //  CHECK: [[VF:%.+]]:2 = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME:         -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-NEXT:     [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT:     VPU.Yield [[CONV_OUT]], [[REDUCE_OUT]]

  //  CHECK: [[CVT_VF:%.+]] = VPU.VerticalFusion ([[VF]]#0 as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>)
  //  CHECK-SAME:         -> tensor<1x32x160x160xf32, {order = #NHWC}>
  //  CHECK-NEXT:     [[CVT:%.+]] = VPU.Convert({{%.+}})
  //  CHECK-NEXT:     VPU.Yield [[CVT]]

  //  CHECK: return [[CVT_VF]], [[VF]]#1
  return %cvt_vf, %vf#1 : tensor<1x32x160x160xf32, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvReduceOutputToConvertNotMerged
module @ConvReduceOutputToConvertNotMerged {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf32, {order = #NHWC}>) {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmin = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 0, 1, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmin
  }

  %cvt_vf = VPU.VerticalFusion (%vf#1 as %arg2: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x1x160x160xf32, {order = #NHWC}> {
    %cvt = VPU.Convert(%arg2) {dstElemType = f32,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x1x160x160xf16, {order = #NHWC}> -> tensor<1x1x160x160xf32, {order = #NHWC}>
    VPU.Yield %cvt
  }
  //  CHECK: [[VF:%.+]]:2 = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME:         -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-NEXT:     [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT:     VPU.Yield [[CONV_OUT]], [[REDUCE_OUT]]

  //  CHECK: [[CVT_VF:%.+]] = VPU.VerticalFusion ([[VF]]#1 as {{%.+}}: tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-SAME:         -> tensor<1x1x160x160xf32, {order = #NHWC}>
  //  CHECK-NEXT:     [[CVT:%.+]] = VPU.Convert({{%.+}})
  //  CHECK-NEXT:     VPU.Yield [[CVT]]

  //  CHECK: return [[VF]]#0, [[CVT_VF]]
  return %vf#0, %cvt_vf : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf32, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvReduceOutPermuteCastToSubtractUnalignedMCs
module @ConvReduceOutPermuteCastToSubtractUnalignedMCs {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>) -> tensor<1x160x160x32xf16> {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmin = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 0, 1, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmin
  }

  %subtract_vf = VPU.VerticalFusion (%vf#0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                                     %vf#1 as %arg3: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x160x160x32xf16> {
    %permCastAct = VPU.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x32x160x160xf16, {order = #NHWC}> -> tensor<1x160x160x32xf16>
    %permCastRed = VPU.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x1x160x160xf16, {order = #NHWC}> -> tensor<1x160x160x1xf16>
    %sub = VPU.Subtract(%permCastAct, %permCastRed)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x160x160x32xf16>, tensor<1x160x160x1xf16> -> tensor<1x160x160x32xf16>
    VPU.Yield %sub
  }

  //  CHECK: [[VF:%.+]]:2 = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME:          -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-NEXT:     [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT:     VPU.Yield [[CONV_OUT]], [[REDUCE_OUT]]

  //  CHECK: [[SUB_VF:%.+]] = VPU.VerticalFusion ([[VF]]#0 as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, [[VF]]#1 as {{%.+}}: tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-SAME:          -> tensor<1x160x160x32xf16>
  //  CHECK-NEXT:     [[PERM_ACT:%.+]] = VPU.PermuteCast({{%.+}})
  //  CHECK-NEXT:     [[PERM_RED:%.+]] = VPU.PermuteCast({{%.+}})
  //  CHECK-NEXT:     [[SUB:%.+]] = VPU.Subtract([[PERM_ACT]], [[PERM_RED]])
  //  CHECK-NEXT: VPU.Yield [[SUB]]

  //  CHECK: return [[SUB_VF]] : tensor<1x160x160x32xf16>
  return %subtract_vf : tensor<1x160x160x32xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvReduceOutPermuteCastToSubtract
module @ConvReduceOutPermuteCastToSubtract {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>) -> tensor<1x160x160x32xf16> {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmin = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 0, 1, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmin
  }

  %subtract_vf = VPU.VerticalFusion (%vf#0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                                     %vf#1 as %arg3: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x160x160x32xf16> {
    %permCastAct = VPU.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x32x160x160xf16, {order = #NHWC}> -> tensor<1x160x160x32xf16>
    %permCastRed = VPU.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x1x160x160xf16, {order = #NHWC}> -> tensor<1x160x160x1xf16>
    %sub = VPU.Subtract(%permCastAct, %permCastRed)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
      : tensor<1x160x160x32xf16>, tensor<1x160x160x1xf16> -> tensor<1x160x160x32xf16>
    VPU.Yield %sub
  }
  //  CHECK: [[VF:%.+]] = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME: -> tensor<1x160x160x32xf16>
  //  CHECK-NEXT: [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT: [[PERM_ACT:%.+]] = VPU.PermuteCast([[CONV_OUT]])
  //  CHECK-NEXT: [[PERM_RED:%.+]] = VPU.PermuteCast([[REDUCE_OUT]])
  //  CHECK-NEXT: [[SUB:%.+]] = VPU.Subtract([[PERM_ACT]], [[PERM_RED]])
  //  CHECK-NEXT: VPU.Yield [[SUB]]
  //  CHECK: return [[VF]] : tensor<1x160x160x32xf16>
  return %subtract_vf : tensor<1x160x160x32xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvReduceOutToSoftMaxWithMax
module @ConvReduceOutToSoftMaxWithMax {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmax = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmax
  }

  %softmax_vf = VPU.VerticalFusion (%vf#0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                                    %vf#1 as %arg3: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x160x160xf16, {order = #NHWC}> {
    %softmax = VPU.SoftMax(%arg2, %arg3) {axisInd = 1 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>
    VPU.Yield %softmax
  }
  //  CHECK: [[VF:%.+]] = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME: -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //  CHECK-NEXT: [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_OUT]], [[REDUCE_OUT]])
  //  CHECK-NEXT: VPU.Yield [[SOFTMAX]]
  //  CHECK: return [[VF]] : tensor<1x32x160x160xf16, {order = #NHWC}>
  return %softmax_vf : tensor<1x32x160x160xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertToConvReduceOut
module @ConvertToConvReduceOut {
func.func @main(%arg0: tensor<1x32x160x160xf32, {order = #NHWC}>,
                %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>,
            tensor<1x1x160x160xf16, {order = #NHWC}>) {
  %cvt_vf = VPU.VerticalFusion (
        %arg0 as %arg2: tensor<1x32x160x160xf32, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x32x160x160xf16, {order = #NHWC}> {
    %cvt = VPU.Convert(%arg2) {
        dstElemType = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf32, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
    VPU.Yield %cvt
  }

  %conv_vf:2 = VPU.VerticalFusion (
        %cvt_vf as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
        %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>,
            tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmax = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmax
  }
  //  CHECK: VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf32, {order = #NHWC}>,
  //  CHECK-SAME: {{%.+}} as {{%.+}}: tensor<32x32x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME: -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>)
  //  CHECK-NEXT: [[CVT:%.+]] = VPU.Convert
  //  CHECK-NEXT: [[CONV_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Convolution([[CVT]]
  //  CHECK-NEXT: VPU.Yield [[CONV_OUT]], [[REDUCE_OUT]]
  //  CHECK: return {{%.+}}#0, {{%.+}}#1
  return %conv_vf#0, %conv_vf#1
    : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseReduceOutToSubtract
// VF with NCE.Eltwise producing reduce_xy_max, followed by a VF consuming both
// the primary output and the reduce output via Subtract. The two VFs share the
// same H-tiling strategy and are expected to be merged into one.
module @EltwiseReduceOutToSubtract {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
                %arg1: tensor<1x32x160x160xf16, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %vf:2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %arg1 as %arg3: tensor<1x32x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmax = VPU.NCE.Eltwise(%arg2, %arg3) {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      op_type = #VPU.eltwise_type<ADD>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x32x160x160xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmax
  }

  %subtract_vf = VPU.VerticalFusion (%vf#0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
                                     %vf#1 as %arg3: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x160x160xf16, {order = #NHWC}> {
    %sub = VPU.Subtract(%arg2, %arg3)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>
    VPU.Yield %sub
  }

  //  CHECK: [[VF:%.+]] = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, {{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>)
  //  CHECK-SAME: -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //  CHECK-NEXT: [[ELT_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.Eltwise
  //  CHECK-NEXT: [[SUB:%.+]] = VPU.Subtract([[ELT_OUT]], [[REDUCE_OUT]])
  //  CHECK-NEXT: VPU.Yield [[SUB]]
  //  CHECK: return [[VF]] : tensor<1x32x160x160xf16, {order = #NHWC}>
  return %subtract_vf : tensor<1x32x160x160xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvReduceOutToSubtract
// VF with NCE.DepthConvolution producing reduce_xy_max, followed by a VF consuming
// both the primary output and the reduce output via Subtract. The two VFs share
// the same H-tiling strategy and are expected to be merged into one.
module @DepthConvReduceOutToSubtract {
func.func @main(%arg0: tensor<1x32x160x160xf16, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %filter = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

  %vf:2 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x160x160xf16, {order = #NHWC}>,
                              %filter as %arg2: tensor<32x16x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %out, %rmax = VPU.NCE.DepthConvolution(%arg1, %arg2) rawFilterShape [32, 1, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>, strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmax
  }

  %subtract_vf = VPU.VerticalFusion (%vf#0 as %arg1: tensor<1x32x160x160xf16, {order = #NHWC}>,
                                     %vf#1 as %arg2: tensor<1x1x160x160xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x160x160xf16, {order = #NHWC}> {
    %sub = VPU.Subtract(%arg1, %arg2)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>
    VPU.Yield %sub
  }

  //  CHECK-DAG: [[FILTER:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
  //  CHECK: [[VF:%.+]] = VPU.VerticalFusion ({{%.+}} as {{%.+}}: tensor<1x32x160x160xf16, {order = #NHWC}>, [[FILTER]] as {{%.+}}: tensor<32x16x1x1xf16, {order = #NHWC}>)
  //  CHECK-SAME: -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //  CHECK-NEXT: [[DW_OUT:%.+]], [[REDUCE_OUT:%.+]] = VPU.NCE.DepthConvolution
  //  CHECK-NEXT: [[SUB:%.+]] = VPU.Subtract([[DW_OUT]], [[REDUCE_OUT]])
  //  CHECK-NEXT: VPU.Yield [[SUB]]
  //  CHECK: return [[VF]] : tensor<1x32x160x160xf16, {order = #NHWC}>
  return %subtract_vf : tensor<1x32x160x160xf16, {order = #NHWC}>
}
}
