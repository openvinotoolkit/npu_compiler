//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --unroll-upsampling-dma %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNCHW
func.func @UnrollUpsamplingDMAWithNCHW() -> memref<1x32x32x64xf16, #NCHW, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x16x32xf16, #NCHW, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x32x32x64xf16, #NCHW, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x32x16x32xf16, #NCHW, @DDR>)
                        outputs(%output : memref<1x32x32x64xf16, #NCHW, @DDR>) -> memref<1x32x32x64xf16, #NCHW, @DDR>
    }
    return %output: memref<1x32x32x64xf16, #NCHW, @DDR>

    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x256x32xf16, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <16384> -> memref<1x1x256x32xf16, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x32x32x64xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <262144> -> memref<1x1x512x64xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <327680> -> memref<1x1x512x64xf16, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 256 : i64
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
    // CHECK-SAME:      inputs([[INPUT0]] : memref<1x1x256x32xf16, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT0]] : memref<1x1x512x64xf16, @DDR>) -> memref<1x1x512x64xf16, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 256 : i64
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
    // CHECK-SAME:      inputs([[INPUT1]] : memref<1x1x256x32xf16, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT1]] : memref<1x1x512x64xf16, @DDR>) -> memref<1x1x512x64xf16, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x32x32x64xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNHWC
func.func @UnrollUpsamplingDMAWithNHWC() -> memref<1x16x1024x32xf16, #NHWC, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x512x32xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x16x1024x32xf16, #NHWC, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x16x512x32xf16, #NHWC, @DDR>)
                        outputs(%output : memref<1x16x1024x32xf16, #NHWC, @DDR>) -> memref<1x16x1024x32xf16, #NHWC, @DDR>
    }

    return %output: memref<1x16x1024x32xf16, #NHWC, @DDR>

    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <262144> -> memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x16x1024x32xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <1572864> -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              len = 1024 : i64
    // CHECK-SAME:              srcWidth = 1024 : i64
    // CHECK-SAME:              srcStride = 1024 : i64
    // CHECK-SAME:              srcPlaneStride = 1024 : i64
    // CHECK-SAME:              dstWidth = 32 : i64
    // CHECK-SAME:              dstStride = 64 : i64
    // CHECK-SAME:              dstPlaneStride = 4096 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT0]] : memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT0]] : memref<1x16x512x64xf16, #NHWC, @DDR>) -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              len = 1024 : i64
    // CHECK-SAME:              srcWidth = 1024 : i64
    // CHECK-SAME:              srcStride = 1024 : i64
    // CHECK-SAME:              srcPlaneStride = 1024 : i64
    // CHECK-SAME:              dstWidth = 32 : i64
    // CHECK-SAME:              dstStride = 64 : i64
    // CHECK-SAME:              dstPlaneStride = 4096 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          port = 1 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT1]] : memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT1]] : memref<1x16x512x64xf16, #NHWC, @DDR>) -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x16x1024x32xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNHWCWithSizeBiggerThan16MB
func.func @UnrollUpsamplingDMAWithNHWCWithSizeBiggerThan16MB() -> memref<1x1024x256x384xf16, #NHWC, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1024x128x192xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <33554432> -> memref<1x1024x256x384xf16, #NHWC, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x1024x128x192xf16, #NHWC, @DDR>)
                        outputs(%output : memref<1x1024x256x384xf16, #NHWC, @DDR>) -> memref<1x1024x256x384xf16, #NHWC, @DDR>
    }

    return %output: memref<1x1024x256x384xf16, #NHWC, @DDR>

    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <49545216> -> memref<1x1024x2x192xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <33030144> -> memref<1x1024x42x192xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT2:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <16515072> -> memref<1x1024x42x192xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT3:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1024x42x192xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <33554432> -> memref<1x1024x256x384xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <231735296> -> memref<1x1024x4x384xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <165675008> -> memref<1x1024x84x384xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <DDR> <99614720> -> memref<1x1024x84x384xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <DDR> <33554432> -> memref<1x1024x84x384xf16, #NHWC, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:         VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:               numPlanes = 42 : i64
    // CHECK-SAME:               len = 393216 : i64
    // CHECK-SAME:               srcWidth = 393216 : i64
    // CHECK-SAME:               srcStride = 393216 : i64
    // CHECK-SAME:               srcPlaneStride = 393216 : i64
    // CHECK-SAME:               dstWidth = 2048 : i64
    // CHECK-SAME:               dstStride = 4096 : i64
    // CHECK-SAME:               dstPlaneStride = 1572864 : i64
    // CHECK-SAME:           >
    // CHECK-SAME:           upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    inputs([[INPUT3]] : memref<1x1024x42x192xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:    outputs([[OUTPUT3]] : memref<1x1024x84x384xf16, #NHWC, @DDR>) -> memref<1x1024x84x384xf16, #NHWC, @DDR>
    // CHECK:  }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:               numPlanes = 42 : i64,
    // CHECK-SAME:               len = 393216 : i64
    // CHECK-SAME:               srcWidth = 393216 : i64
    // CHECK-SAME:               srcStride = 393216 : i64
    // CHECK-SAME:               srcPlaneStride = 393216 : i64
    // CHECK-SAME:               dstWidth = 2048 : i64
    // CHECK-SAME:               dstStride = 4096 : i64
    // CHECK-SAME:               dstPlaneStride = 1572864 : i64
    // CHECK-SAME:           >
    // CHECK-SAME:           port = 1 : i64
    // CHECK-SAME:           upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    inputs([[INPUT2]] : memref<1x1024x42x192xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:    outputs([[OUTPUT2]] : memref<1x1024x84x384xf16, #NHWC, @DDR>) -> memref<1x1024x84x384xf16, #NHWC, @DDR>
    // CHECK:  }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:               numPlanes = 42 : i64
    // CHECK-SAME:               len = 393216 : i64
    // CHECK-SAME:               srcWidth = 393216 : i64
    // CHECK-SAME:               srcStride = 393216 : i64
    // CHECK-SAME:               srcPlaneStride = 393216 : i64
    // CHECK-SAME:               dstWidth = 2048 : i64
    // CHECK-SAME:               dstStride = 4096 : i64
    // CHECK-SAME:               dstPlaneStride = 1572864 : i64
    // CHECK-SAME:           >
    // CHECK-SAME:           upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    inputs([[INPUT1]] : memref<1x1024x42x192xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:    outputs([[OUTPUT1]] : memref<1x1024x84x384xf16, #NHWC, @DDR>) -> memref<1x1024x84x384xf16, #NHWC, @DDR>
    // CHECK:  }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:               numPlanes = 2 : i64
    // CHECK-SAME:               len = 393216 : i64
    // CHECK-SAME:               srcWidth = 393216 : i64
    // CHECK-SAME:               srcStride = 393216 : i64
    // CHECK-SAME:               srcPlaneStride = 393216 : i64
    // CHECK-SAME:               dstWidth = 2048 : i64
    // CHECK-SAME:               dstStride = 4096 : i64
    // CHECK-SAME:               dstPlaneStride = 1572864 : i64
    // CHECK-SAME:           >
    // CHECK-SAME:           port = 1 : i64
    // CHECK-SAME:           upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    inputs([[INPUT0]] : memref<1x1024x2x192xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:    outputs([[OUTPUT0]] : memref<1x1024x4x384xf16, #NHWC, @DDR>) -> memref<1x1024x4x384xf16, #NHWC, @DDR>
    // CHECK:  }
    // CHECK:    return [[OUTPUT]] : memref<1x1024x256x384xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithExpandAttr
func.func @UnrollUpsamplingDMAWithExpandAttr() -> memref<1x32x640x640xf16, #NHWC, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x24x320x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x32x640x640xf16, #NHWC, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %3 = VPUIP.UpsamplingDMAOp {expand = [0, 8, 0, 0], port = 1 : i64, upsampling_factor = [1, 1, 2, 2]}
        inputs(%input : memref<1x24x320x320xf16, #NHWC, @DDR>)
        outputs(%output : memref<1x32x640x640xf16, #NHWC, @DDR>) -> memref<1x32x640x640xf16, #NHWC, @DDR>
    }

    return %output: memref<1x32x640x640xf16, #NHWC, @DDR>

    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <3932160> -> memref<1x24x64x320xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x24x256x320xf16, #NHWC, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x32x640x640xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <21495808> -> memref<1x32x128x640xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <524288> -> memref<1x32x512x640xf16, #NHWC, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              len = 15360 : i64
    // CHECK-SAME:              srcWidth = 15360 : i64
    // CHECK-SAME:              srcStride = 15360 : i64
    // CHECK-SAME:              srcPlaneStride = 15360 : i64
    // CHECK-SAME:              dstWidth = 48 : i64
    // CHECK-SAME:              dstStride = 128 : i64
    // CHECK-SAME:              dstPlaneStride = 81920 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          expand = [0, 8, 0, 0]
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT1]] : memref<1x24x256x320xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT1]] : memref<1x32x512x640xf16, #NHWC, @DDR>) -> memref<1x32x512x640xf16, #NHWC, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:              numPlanes = 64 : i64
    // CHECK-SAME:              len = 15360 : i64
    // CHECK-SAME:              srcWidth = 15360 : i64
    // CHECK-SAME:              srcStride = 15360 : i64
    // CHECK-SAME:              srcPlaneStride = 15360 : i64
    // CHECK-SAME:              dstWidth = 48 : i64
    // CHECK-SAME:              dstStride = 128 : i64
    // CHECK-SAME:              dstPlaneStride = 81920 : i64
    // CHECK-SAME:          >
    // CHECK-SAME:          expand = [0, 8, 0, 0]
    // CHECK-SAME:          port = 1 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT0]] : memref<1x24x64x320xf16, #NHWC, [@DDR, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT0]] : memref<1x32x128x640xf16, #NHWC, @DDR>) -> memref<1x32x128x640xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x32x640x640xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @UnrollUpsamplingDMAWithCstInputNHWC {
config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 17179869184 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
// CHECK-LABEL: @UnrollUpsamplingDMAWithCstInputNHWC
func.func @UnrollUpsamplingDMAWithCstInputNHWC() -> memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cst = const.Declare memref<1x96x512x16xf16, #NHWC> = dense<1.0> : tensor<1x96x512x16xf16>, [#const.Reorder<#NHWC>]
    %output  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %3 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 1, 2, 2]} inputs(%cst : memref<1x96x512x16xf16, #NHWC>)
        outputs(%output : memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %output: memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[CST:%.+]] = const.Declare memref<1x96x512x16xf16, #NHWC> = dense<1.000000e+00> : tensor<1x96x512x16xf16>, [#const.Reorder<#NHWC>]
    // CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <3145728> -> memref<1x96x512x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x512x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CST]] [0, 0, 0, 0] [1, 96, 256, 16] : memref<1x96x512x16xf16, #NHWC> to memref<1x96x256x16xf16, {order = #NHWC, strides = [786432, 1, 1536, 96]}>

    // CHECK:     VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:        dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:            numPlanes = 256 : i64,
    // CHECK-SAME:            len = 3072 : i64,
    // CHECK-SAME:            srcWidth = 3072 : i64,
    // CHECK-SAME:            srcStride = 3072 : i64,
    // CHECK-SAME:            srcPlaneStride = 3072 : i64,
    // CHECK-SAME:            dstWidth = 192 : i64,
    // CHECK-SAME:            dstStride = 384 : i64,
    // CHECK-SAME:            dstPlaneStride = 12288 : i64
    // CHECK-SAME:        >,
    // CHECK-SAME:        upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[SUBVIEW_0]] : memref<1x96x256x16xf16, {order = #NHWC, strides = [786432, 1, 1536, 96]}>)
    // CHECK-SAME:      outputs([[OUTPUT1]] : memref<1x96x512x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x96x512x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CST]]  [0, 0, 256, 0] [1, 96, 256, 16] : memref<1x96x512x16xf16, #NHWC> to memref<1x96x256x16xf16, {order = #NHWC, strides = [786432, 1, 1536, 96]}>
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:      dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 256 : i64,
    // CHECK-SAME:          len = 3072 : i64,
    // CHECK-SAME:          srcWidth = 3072 : i64,
    // CHECK-SAME:          srcStride = 3072 : i64,
    // CHECK-SAME:          srcPlaneStride = 3072 : i64,
    // CHECK-SAME:          dstWidth = 192 : i64,
    // CHECK-SAME:          dstStride = 384 : i64,
    // CHECK-SAME:          dstPlaneStride = 12288 : i64
    // CHECK-SAME:      >,
    // CHECK-SAME:      port = 1 : i64,
    // CHECK-SAME:      upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[SUBVIEW_1]] : memref<1x96x256x16xf16, {order = #NHWC, strides = [786432, 1, 1536, 96]}>)
    // CHECK-SAME:      outputs([[OUTPUT0]] : memref<1x96x512x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x96x512x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x96x1024x32xf16, #NHWC, [@CMX_NN, 0]>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithCstInputNCHW
func.func @UnrollUpsamplingDMAWithCstInputNCHW() -> memref<1x96x10x10xf16, #NCHW, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cst = const.Declare memref<1x96x5x5xf16> = dense<1.0> : tensor<1x96x5x5xf32>, [#const.CastElemType<f16>]
    %output  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x10x10xf16, [@CMX_NN, 0]>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %3 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 1, 2, 2]} inputs(%cst : memref<1x96x5x5xf16>)
        outputs(%output : memref<1x96x10x10xf16, [@CMX_NN, 0]>) -> memref<1x96x10x10xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x96x10x10xf16, #NCHW, [@CMX_NN, 0]>

    // CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x10x10xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <10240> -> memref<1x1x448x10xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x512x10xf16, [@CMX_NN, 0]>
    // CHECK:    [[CST:%.+]] = const.Declare memref<1x1x480x5xf16> = dense<1.000000e+00> : tensor<1x96x5x5xf32>, [#const.Reshape<[1, 1, 480, 5]>, #const.CastElemType<f16>]
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CST]] [0, 0, 0, 0] [1, 1, 256, 5] : memref<1x1x480x5xf16> to memref<1x1x256x5xf16, {order = #NCHW, strides = [2400, 2400, 5, 1]}>

    // CHECK:     VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:            VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:            dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:            numPlanes = 256 : i64,
    // CHECK-SAME:            len = 10 : i64,
    // CHECK-SAME:            srcWidth = 10 : i64,
    // CHECK-SAME:            srcStride = 10 : i64,
    // CHECK-SAME:            srcPlaneStride = 10 : i64,
    // CHECK-SAME:            dstWidth = 2 : i64,
    // CHECK-SAME:            dstStride = 4 : i64,
    // CHECK-SAME:            dstPlaneStride = 40 : i64
    // CHECK-SAME:        >,
    // CHECK-SAME:        upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:        }
    // CHECK-SAME:        inputs([[SUBVIEW_0]] : memref<1x1x256x5xf16, {order = #NCHW, strides = [2400, 2400, 5, 1]}>)
    // CHECK-SAME:        outputs([[OUTPUT1]] : memref<1x1x512x10xf16, [@CMX_NN, 0]>) -> memref<1x1x512x10xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CST]] [0, 0, 256, 0] [1, 1, 224, 5] : memref<1x1x480x5xf16> to memref<1x1x224x5xf16, {order = #NCHW, strides = [2400, 2400, 5, 1]}>
    // CHECK:     VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 224 : i64,
    // CHECK-SAME:          len = 10 : i64,
    // CHECK-SAME:          srcWidth = 10 : i64,
    // CHECK-SAME:          srcStride = 10 : i64,
    // CHECK-SAME:          srcPlaneStride = 10 : i64,
    // CHECK-SAME:          dstWidth = 2 : i64,
    // CHECK-SAME:          dstStride = 4 : i64,
    // CHECK-SAME:          dstPlaneStride = 40 : i64
    // CHECK-SAME:        >,
    // CHECK-SAME:        port = 1 : i64,
    // CHECK-SAME:        upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:        }
    // CHECK-SAME:        inputs([[SUBVIEW_1]] : memref<1x1x224x5xf16, {order = #NCHW, strides = [2400, 2400, 5, 1]}>)
    // CHECK-SAME:        outputs([[OUTPUT0]] : memref<1x1x448x10xf16, [@CMX_NN, 0]>) -> memref<1x1x448x10xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:     return [[OUTPUT]] : memref<1x96x10x10xf16, [@CMX_NN, 0]>
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
