//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED enable-dynamic-dim-alignment=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDynamicWithDimAlignment
func.func @ConvertDynamicWithDimAlignment(%arg0: tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>)
    -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       VPU.Convert
    // CHECK-SAME:  tilingStrategy = [1, 1, [[TS_D2:[0-9]+]], [[TS_D3:[0-9]+]]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertDynamicNCHWWithDimAlignment
func.func @ConvertDynamicNCHWWithDimAlignment(%arg0: tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 4320, 7680]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4320, 7680]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 4320, 7680]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4320, 7680]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4320, 7680]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       VPU.Convert
    // CHECK-SAME:  tilingStrategy = [1, 1, [[TS_D2:[0-9]+]], [[TS_D3:[0-9]+]]]
}
