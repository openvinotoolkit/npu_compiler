//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-upsampling-dma %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @NoUnrollUpsamplingDMAWithNCHW
func.func @NoUnrollUpsamplingDMAWithNCHW() -> memref<1x4x33554432x60xf16, #NCHW, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x16777216x30xf16, #NCHW, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x4x33554432x60xf16, #NCHW, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x4x16777216x30xf16, #NCHW, @DDR>)
                        outputs(%output : memref<1x4x33554432x60xf16, #NCHW, @DDR>) -> memref<1x4x33554432x60xf16, #NCHW, @DDR>
    }
    return %output: memref<1x4x33554432x60xf16, #NCHW, @DDR>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x67108864x30xf16, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x4x33554432x60xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x1x134217728x60xf16, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 67108864 : i64
    // CHECK-SAME:              len = 60 : i64
    // CHECK-SAME:              srcWidth = 60 : i64
    // CHECK-SAME:              srcStride = 60 : i64
    // CHECK-SAME:              srcPlaneStride = 60 : i64
    // CHECK-SAME:              dstWidth = 2 : i64
    // CHECK-SAME:              dstStride = 4 : i64
    // CHECK-SAME:              dstPlaneStride = 240 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT_0]] : memref<1x1x67108864x30xf16, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_0]] : memref<1x1x134217728x60xf16, @DDR>) -> memref<1x1x134217728x60xf16, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x4x33554432x60xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNCHW
func.func @UnrollUpsamplingDMAWithNCHW() -> memref<1x4x33554432x64xf16, #NCHW, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x16777216x32xf16, #NCHW, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x4x33554432x64xf16, #NCHW, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x4x16777216x32xf16, #NCHW, @DDR>)
                        outputs(%output : memref<1x4x33554432x64xf16, #NCHW, @DDR>) -> memref<1x4x33554432x64xf16, #NCHW, @DDR>
    }
    return %output: memref<1x4x33554432x64xf16, #NCHW, @DDR>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <4294967232> -> memref<1x1x1x32xf16, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x67108863x32xf16, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x4x33554432x64xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <DDR> <17180131072> -> memref<1x1x2x64xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x1x134217726x64xf16, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 67108863 : i64
    // CHECK-SAME:              len = 64 : i64
    // CHECK-SAME:              srcWidth = 64 : i64
    // CHECK-SAME:              srcStride = 64 : i64
    // CHECK-SAME:              srcPlaneStride = 64 : i64
    // CHECK-SAME:              dstWidth = 2 : i64
    // CHECK-SAME:              dstStride = 4 : i64
    // CHECK-SAME:              dstPlaneStride = 256 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT_1]] : memref<1x1x67108863x32xf16, [@DDR, 0]>
    // CHECK-SAME:      outputs([[OUTPUT_1]] : memref<1x1x134217726x64xf16, @DDR>) -> memref<1x1x134217726x64xf16, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 1 : i64
    // CHECK-SAME:              len = 64 : i64
    // CHECK-SAME:              srcWidth = 64 : i64
    // CHECK-SAME:              srcStride = 64 : i64
    // CHECK-SAME:              srcPlaneStride = 64 : i64
    // CHECK-SAME:              dstWidth = 2 : i64
    // CHECK-SAME:              dstStride = 4 : i64
    // CHECK-SAME:              dstPlaneStride = 256 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          port = 1 : i64,
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT_0]] : memref<1x1x1x32xf16, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_0]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x4x33554432x64xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNHWC
func.func @UnrollUpsamplingDMAWithNHWC() -> memref<1x32x8x33554432xf16, #NHWC, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x4x16777216xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x32x8x33554432xf16, #NHWC, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x32x4x16777216xf16, #NHWC, @DDR>)
                        outputs(%output : memref<1x32x8x33554432xf16, #NHWC, @DDR>) -> memref<1x32x8x33554432xf16, #NHWC, @DDR>
    }

    return %output: memref<1x32x8x33554432xf16, #NHWC, @DDR>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <3221225472> -> memref<1x32x1x16777216xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x3x16777216xf16, #NHWC, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x32x8x33554432xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <DDR> <12885426176> -> memref<1x32x2x33554432xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x32x6x33554432xf16, #NHWC, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 3 : i64
    // CHECK-SAME:              len = 1073741824 : i64
    // CHECK-SAME:              srcWidth = 1073741824 : i64
    // CHECK-SAME:              srcStride = 1073741824 : i64
    // CHECK-SAME:              srcPlaneStride = 1073741824 : i64
    // CHECK-SAME:              dstWidth = 64 : i64
    // CHECK-SAME:              dstStride = 128 : i64
    // CHECK-SAME:              dstPlaneStride = 4294967296 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT_1]] : memref<1x32x3x16777216xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_1]] : memref<1x32x6x33554432xf16, #NHWC, @DDR>) -> memref<1x32x6x33554432xf16, #NHWC, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 1 : i64
    // CHECK-SAME:              len = 1073741824 : i64
    // CHECK-SAME:              srcWidth = 1073741824 : i64
    // CHECK-SAME:              srcStride = 1073741824 : i64
    // CHECK-SAME:              srcPlaneStride = 1073741824 : i64
    // CHECK-SAME:              dstWidth = 64 : i64
    // CHECK-SAME:              dstStride = 128 : i64
    // CHECK-SAME:              dstPlaneStride = 4294967296 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          port = 1 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT_0]] : memref<1x32x1x16777216xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_0]] : memref<1x32x2x33554432xf16, #NHWC, @DDR>) -> memref<1x32x2x33554432xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x32x8x33554432xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithExpandAttrWithNHWC
func.func @UnrollUpsamplingDMAWithExpandAttrWithNHWC() -> memref<1x40x8x33554432xf16, #NHWC, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x4x16777216xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x40x8x33554432xf16, #NHWC, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %3 = VPUIP.UpsamplingDMAOp {expand = [0, 8, 0, 0], port = 1 : i64, upsampling_factor = [1, 1, 2, 2]}
        inputs(%input : memref<1x32x4x16777216xf16, #NHWC, @DDR>)
        outputs(%output : memref<1x40x8x33554432xf16, #NHWC, @DDR>) -> memref<1x40x8x33554432xf16, #NHWC, @DDR>
    }

    return %output: memref<1x40x8x33554432xf16, #NHWC, @DDR>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <3221225472> -> memref<1x32x1x16777216xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x3x16777216xf16, #NHWC, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x40x8x33554432xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <DDR> <16106651648> -> memref<1x40x2x33554432xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x40x6x33554432xf16, #NHWC, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 3 : i64
    // CHECK-SAME:              len = 1073741824 : i64
    // CHECK-SAME:              srcWidth = 1073741824 : i64
    // CHECK-SAME:              srcStride = 1073741824 : i64
    // CHECK-SAME:              srcPlaneStride = 1073741824 : i64
    // CHECK-SAME:              dstWidth = 64 : i64
    // CHECK-SAME:              dstStride = 160 : i64
    // CHECK-SAME:              dstPlaneStride = 5368709120 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          expand = [0, 8, 0, 0]
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT_1]] : memref<1x32x3x16777216xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_1]] : memref<1x40x6x33554432xf16, #NHWC, @DDR>) -> memref<1x40x6x33554432xf16, #NHWC, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 1 : i64
    // CHECK-SAME:              len = 1073741824 : i64
    // CHECK-SAME:              srcWidth = 1073741824 : i64
    // CHECK-SAME:              srcStride = 1073741824 : i64
    // CHECK-SAME:              srcPlaneStride = 1073741824 : i64
    // CHECK-SAME:              dstWidth = 64 : i64
    // CHECK-SAME:              dstStride = 160 : i64
    // CHECK-SAME:              dstPlaneStride = 5368709120 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          expand = [0, 8, 0, 0]
    // CHECK-SAME:          port = 1 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT_0]] : memref<1x32x1x16777216xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_0]] : memref<1x40x2x33554432xf16, #NHWC, @DDR>) -> memref<1x40x2x33554432xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x40x8x33554432xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithCstInputWithNHWC
func.func @UnrollUpsamplingDMAWithCstInputWithNHWC() -> memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cst = const.Declare memref<1x32x4x16777216xf16, #NHWC> = dense<1.0> : tensor<1x32x4x16777216xf16>, [#const.Reorder<#NHWC>]
    %output  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %3 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 1, 2, 2]} inputs(%cst : memref<1x32x4x16777216xf16, #NHWC>)
        outputs(%output : memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %output: memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK-DAG:    [[CST_0:%.*]] = const.Declare memref<1x32x1x16777216xf16, {order = #NHWC, strides = [2147483648, 1, 536870912, 32]}> = dense<1.000000e+00> : tensor<1x32x4x16777216xf16>, [#const.SubView<[0, 0, 3, 0], [1, 32, 1, 16777216]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[CST_1:%.*]] = const.Declare memref<1x32x3x16777216xf16, {order = #NHWC, strides = [2147483648, 1, 536870912, 32]}> = dense<1.000000e+00> : tensor<1x32x4x16777216xf16>, [#const.SubView<[0, 0, 0, 0], [1, 32, 3, 16777216]>, #const.Reorder<#NHWC>]

    // CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <12884901888> -> memref<1x32x2x33554432xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x6x33554432xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:     VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:        dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:            numPlanes = 3 : i64,
    // CHECK-SAME:            len = 1073741824 : i64,
    // CHECK-SAME:            srcWidth = 1073741824 : i64,
    // CHECK-SAME:            srcStride = 1073741824 : i64,
    // CHECK-SAME:            srcPlaneStride = 1073741824 : i64,
    // CHECK-SAME:            dstWidth = 64 : i64,
    // CHECK-SAME:            dstStride = 128 : i64,
    // CHECK-SAME:            dstPlaneStride = 4294967296 : i64
    // CHECK-SAME:        >,
    // CHECK-SAME:        upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[CST_1]] : memref<1x32x3x16777216xf16, {order = #NHWC, strides = [2147483648, 1, 536870912, 32]}>)
    // CHECK-SAME:      outputs([[OUTPUT_1]] : memref<1x32x6x33554432xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x6x33554432xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:      dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 1 : i64,
    // CHECK-SAME:          len = 1073741824 : i64,
    // CHECK-SAME:          srcWidth = 1073741824 : i64,
    // CHECK-SAME:          srcStride = 1073741824 : i64,
    // CHECK-SAME:          srcPlaneStride = 1073741824 : i64,
    // CHECK-SAME:          dstWidth = 64 : i64,
    // CHECK-SAME:          dstStride = 128 : i64,
    // CHECK-SAME:          dstPlaneStride = 4294967296 : i64
    // CHECK-SAME:      >,
    // CHECK-SAME:      port = 1 : i64,
    // CHECK-SAME:      upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[CST_0]] : memref<1x32x1x16777216xf16, {order = #NHWC, strides = [2147483648, 1, 536870912, 32]}>)
    // CHECK-SAME:      outputs([[OUTPUT_0]] : memref<1x32x2x33554432xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x2x33554432xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x32x8x33554432xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithCstInputWithNCHW
func.func @UnrollUpsamplingDMAWithCstInputWithNCHW() -> memref<1x4x33554432x64xf16, #NCHW, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cst = const.Declare memref<1x4x16777216x32xf16> = dense<1.0> : tensor<1x4x16777216x32xf16>, [#const.CastElemType<f16>]
    %output  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x33554432x64xf16, [@CMX_NN, 0]>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %3 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 1, 2, 2]} inputs(%cst : memref<1x4x16777216x32xf16>)
        outputs(%output : memref<1x4x33554432x64xf16, [@CMX_NN, 0]>) -> memref<1x4x33554432x64xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x4x33554432x64xf16, #NCHW, [@CMX_NN, 0]>

    // CHECK-DAG:    [[CST_0:%.*]] = const.Declare memref<1x1x1x32xf16, {order = #NCHW, strides = [2147483648, 2147483648, 32, 1]}> = dense<1.000000e+00> : tensor<1x4x16777216x32xf16>, [#const.Reshape<[1, 1, 67108864, 32]>, #const.SubView<[0, 0, 67108863, 0], [1, 1, 1, 32]>, #const.CastElemType<f16>]
    // CHECK-DAG:    [[CST_1:%.*]] =  const.Declare memref<1x1x67108863x32xf16, {order = #NCHW, strides = [2147483648, 2147483648, 32, 1]}> = dense<1.000000e+00> : tensor<1x4x16777216x32xf16>, [#const.Reshape<[1, 1, 67108864, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 67108863, 32]>, #const.CastElemType<f16>]
    // CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x33554432x64xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17179868928> -> memref<1x1x2x64xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x134217726x64xf16, [@CMX_NN, 0]>

    // CHECK:     VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:            VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:            dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:            numPlanes = 67108863 : i64,
    // CHECK-SAME:            len = 64 : i64,
    // CHECK-SAME:            srcWidth = 64 : i64,
    // CHECK-SAME:            srcStride = 64 : i64,
    // CHECK-SAME:            srcPlaneStride = 64 : i64,
    // CHECK-SAME:            dstWidth = 2 : i64,
    // CHECK-SAME:            dstStride = 4 : i64,
    // CHECK-SAME:            dstPlaneStride = 256 : i64
    // CHECK-SAME:        >,
    // CHECK-SAME:        upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:        }
    // CHECK-SAME:        inputs([[CST_1]] : memref<1x1x67108863x32xf16, {order = #NCHW, strides = [2147483648, 2147483648, 32, 1]}>)
    // CHECK-SAME:        outputs([[OUTPUT_1]] : memref<1x1x134217726x64xf16, [@CMX_NN, 0]>) -> memref<1x1x134217726x64xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:     VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 1 : i64,
    // CHECK-SAME:          len = 64 : i64,
    // CHECK-SAME:          srcWidth = 64 : i64,
    // CHECK-SAME:          srcStride = 64 : i64,
    // CHECK-SAME:          srcPlaneStride = 64 : i64,
    // CHECK-SAME:          dstWidth = 2 : i64,
    // CHECK-SAME:          dstStride = 4 : i64,
    // CHECK-SAME:          dstPlaneStride = 256 : i64
    // CHECK-SAME:        >,
    // CHECK-SAME:        port = 1 : i64,
    // CHECK-SAME:        upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:        }
    // CHECK-SAME:        inputs([[CST_0]] : memref<1x1x1x32xf16, {order = #NCHW, strides = [2147483648, 2147483648, 32, 1]}>)
    // CHECK-SAME:        outputs([[OUTPUT_0]] : memref<1x1x2x64xf16, [@CMX_NN, 0]>) -> memref<1x1x2x64xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:     return [[OUTPUT]] : memref<1x4x33554432x64xf16, [@CMX_NN, 0]>
}

// -----

// CHECK-LABEL: @UnrollUpsamplingDMANCHWWithExpandAttr
func.func @UnrollUpsamplingDMANCHWWithExpandAttr() -> memref<1x16x60x1xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x30x1xf16, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x60x1xf16, @DDR>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %upsDMA = VPUIP.UpsamplingDMAOp {expand = [0, 13, 0, 0], port = 1 : i64, upsampling_factor = [1, 1, 2, 1]}
          inputs(%input : memref<1x3x30x1xf16, @DDR>)
          outputs(%output : memref<1x16x60x1xf16, @DDR>) -> memref<1x16x60x1xf16, @DDR>
    }

    return %output : memref<1x16x60x1xf16, @DDR>

    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[INPUT:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x90x1xf16, [@DDR, 0]>
    // CHECK:   [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x60x1xf16, @DDR>
    // CHECK:   [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x60x1xf16, @DDR>
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 90 : i64,
    // CHECK-SAME:          len = 2 : i64,
    // CHECK-SAME:          srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 2 : i64,
    // CHECK-SAME:          dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 4 : i64>,
    // CHECK-SAME:          expand = [0, 13, 0, 0],
    // CHECK-SAME:          port = 0 : i64,
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:        inputs([[INPUT]] : memref<1x1x90x1xf16, [@DDR, 0]>)
    // CHECK-SAME:        outputs([[OUTPUT0]] : memref<1x16x60x1xf16, @DDR>) -> memref<1x16x60x1xf16, @DDR>
    // CHECK:       }
    // CHECK:   return [[OUTPUT]] : memref<1x16x60x1xf16, @DDR>
}
