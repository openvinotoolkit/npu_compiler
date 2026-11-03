//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 46341, 46341]> : tensor<4xsi64>, order = #NCHW}>
func.func @spatial_count_overflow(%input: !input, %aux: tensor<65536xui8>) -> tensor<1x1x1x1xf16> {
    // expected-error@+1 {{DMA Mean requires positive spatial bounds with an int32 reduction count}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<65536xui8> -> tensor<1x1x1x1xf16>
    return %result : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x2147483648x1x1xf16>
func.func @channel_count_overflow(%input: !input, %aux: tensor<65536xui8>) -> tensor<1x2147483648x1x1xf16> {
    // expected-error@+1 {{DMA Mean requires NCHW [1,C,H,W] with positive int32 static C}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<65536xui8> -> tensor<1x2147483648x1x1xf16>
    return %result : tensor<1x2147483648x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
func.func @dynamic_channels(%input: !input, %aux: tensor<65536xui8>) -> tensor<1x7x1x1xf16> {
    // expected-error@+1 {{DMA Mean requires NCHW [1,C,H,W] with positive int32 static C}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<65536xui8> -> tensor<1x7x1x1xf16>
    return %result : tensor<1x7x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
func.func @small_workspace(%input: !input, %aux: tensor<1xui8>) -> tensor<1x7x1x1xf16> {
    // expected-error@+1 {{DMA Mean requires an adequate unsigned byte workspace}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<1xui8> -> tensor<1x7x1x1xf16>
    return %result : tensor<1x7x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
func.func @wrong_workspace_type(%input: !input, %aux: tensor<65536xf16>) -> tensor<1x7x1x1xf16> {
    // expected-error@+1 {{DMA Mean requires an adequate unsigned byte workspace}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<65536xf16> -> tensor<1x7x1x1xf16>
    return %result : tensor<1x7x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<2x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[2, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
func.func @batch_not_one(%input: !input, %aux: tensor<65536xui8>) -> tensor<2x7x1x1xf16> {
    // expected-error@+1 {{DMA Mean requires NCHW [1,C,H,W] input}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<65536xui8> -> tensor<2x7x1x1xf16>
    return %result : tensor<2x7x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!input = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
func.func @wrong_output_layout(%input: !input, %aux: tensor<65536xui8>) -> tensor<1x7x1x1xf16, {order = #NHWC}> {
    // expected-error@+1 {{DMA Mean requires static NCHW [1,C,1,1] output}}
    %result = VPU.ReduceMeanDMA(%input, %aux) : !input, tensor<65536xui8> -> tensor<1x7x1x1xf16, {order = #NHWC}>
    return %result : tensor<1x7x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
func.func @unsupported_strategy(%input: !input, %aux: tensor<65536xui8>) -> tensor<1x7x1x1xf16> {
    // expected-error@+1 {{DMA Mean runs on a single tile and takes no multi-cluster strategy}}
    %result = VPU.ReduceMeanDMA(%input, %aux) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : !input, tensor<65536xui8> -> tensor<1x7x1x1xf16>
    return %result : tensor<1x7x1x1xf16>
}
