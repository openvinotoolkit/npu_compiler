//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-permute-dma  %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNHWCToNCHW
func.func @PermuteToDMAWithNHWCToNCHW() -> memref<1x8x16x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<128x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<128x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4352> -> memref<8x128xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x128xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 128 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 512 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_1]] : memref<128x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<8x128xf16, [@CMX_NN, 0]>) -> memref<8x128xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 128 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 512 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_0]] : memref<128x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<8x128xf16, [@CMX_NN, 0]>) -> memref<8x128xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWC
func.func @PermuteToDMAWithNCHWToNHWC() -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<64x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4224> -> memref<16x64xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<16x64xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 256 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_1]] : memref<64x16xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<16x64xf16, [@CMX_NN, 0]>) -> memref<16x64xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 256 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_0]] : memref<64x16xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<16x64xf16, [@CMX_NN, 0]>) -> memref<16x64xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAFromTranspose
func.func @PermuteToDMAFromTranspose() -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<16x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4128> -> memref<8x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x16xf16, [@CMX_NN, 0]>


    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_1]] : memref<16x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<8x16xf16, [@CMX_NN, 0]>) -> memref<8x16xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_0]] : memref<16x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<8x16xf16, [@CMX_NN, 0]>) -> memref<8x16xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithLargePlaneNumber
func.func @PermuteToDMAWithLargePlaneNumber() -> memref<1x8x32x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x32x16xf16, [@CMX_NN, 0]>) -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x8x32x16xf16, [@CMX_NN, 0]>


    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 1024 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER0]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER0]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 1024 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER1]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER1]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x32x16xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
  func.func private @builtin_Convert(%input : memref<*xf16,  [@CMX_NN, 0]>, %output : memref<*xf16,  [@CMX_NN, 0]>) attributes { VPU.kernel_code = "convert_fp16.cpp", VPU.kernel_entry = "convert_fp16", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollDistributedPermuteDMA
func.func @UnrollDistributedPermuteDMA() -> memref<1x3x24x24xf16, #NHWC, @DDR> {
    %result = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x3x24x24xf16, #NHWC, @DDR>
    %cst = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %12 = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x3x24x24xui8, @DDR>
    %13 = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x3x12x24xf16, #NHWC, @DDR>
    %14 = VPURT.DeclareBuffer <NetworkOutput> <1728> -> memref<1x3x12x24xf16, #NHWC, @DDR>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <3456> -> memref<1x3x24x24xui8, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x24x24xf16, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x24x24xf16, @DDR>
    %18 = VPURT.DeclareBuffer <DDR> <3456> -> memref<1x16x24x24xf16, @DDR>
    %19 = VPURT.DeclareBuffer <CMX_NN> <5440> -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <5440> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <5440> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <5184> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [1] <5184> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <5184> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %26 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 1]>
    %27 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %28 = VPURT.DeclareBuffer <CMX_NN> <14656> -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <14656> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <14656> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <DDR> <3456> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %32 = VPURT.DeclareBuffer <DDR> <6912> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %33 = VPURT.DeclareBuffer <DDR> <10368> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %34 = VPURT.DeclareBuffer <DDR> <13824> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %35 = VPURT.DeclareBuffer <DDR> <17280> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %36 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x24x24xf16, {order = #NCHW, strides = [1728, 576, 24, 1]}, @DDR>
    %37 = VPURT.DeclareBuffer <DDR> <20736> -> memref<1x1x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %38 = VPURT.DeclareBuffer <CMX_NN> [0] <14656> -> memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>
    %39 = VPURT.DeclareBuffer <CMX_NN> [1] <14656> -> memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 1]>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%12 : memref<1x3x24x24xui8, @DDR>) outputs(%15 : memref<1x3x24x24xui8, [@CMX_NN, 0]>) -> memref<1x3x24x24xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%cst_0 : memref<16x1x1x4xsi32>) outputs(%24 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%15 as %arg2: memref<1x3x24x24xui8, [@CMX_NN, 0]>) outputs(%16 as %arg3: memref<1x3x24x24xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x24x24xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x3x24x24xui8, [@CMX_NN, 0]>, memref<1x3x24x24xf16, [@CMX_NN, 0]>
      }
    }

    // expand input
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%16 : memref<1x3x24x24xf16, [@CMX_NN, 0]>) outputs(%17 : memref<1x3x24x24xf16, @DDR>) -> memref<1x3x24x24xf16, @DDR>
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%31 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%32 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%33 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%34 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%35 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%36 : memref<1x1x24x24xf16, {order = #NCHW, strides = [1728, 576, 24, 1]}, @DDR>) outputs(%37 : memref<1x1x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x1x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    // permute
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %41 = VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64, src_plane_stride = 0 : i64} inputs(%18 : memref<1x16x24x24xf16, @DDR>) outputs(%19 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }

    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%cst : memref<1x1x1x16xui8>) outputs(%27 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }

    // NCE task
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%20 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%22 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%28 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%29 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%21 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%23 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%28 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%30 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 12, 0]}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // copy result
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA inputs(%38 : memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>) outputs(%13 : memref<1x3x12x24xf16, #NHWC, @DDR>) -> memref<1x3x12x24xf16, #NHWC, @DDR>
    }
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 1 : i64} inputs(%39 : memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 1]>) outputs(%14 : memref<1x3x12x24xf16, #NHWC, @DDR>) -> memref<1x3x12x24xf16, #NHWC, @DDR>
    }
    return %result : memref<1x3x24x24xf16, #NHWC, @DDR>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR7:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR8:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR9:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR10:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR11:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUF_0:%.*]] = VPURT.DeclareBuffer <DDR> <12672> -> memref<8x288xf16, @DDR>
    //CHECK:    [[INPUT_BUF_1:%.*]] = VPURT.DeclareBuffer <DDR> <3456> -> memref<8x288xf16, @DDR>
    //CHECK:    [[INPUT_BUF_2:%.*]] = VPURT.DeclareBuffer <DDR> <13248> -> memref<8x288xf16, @DDR>
    //CHECK:    [[INPUT_BUF_3:%.*]] = VPURT.DeclareBuffer <DDR> <4032> -> memref<8x288xf16, @DDR>
    //CHECK:    [[CMX_BUF_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <5456> -> memref<288x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[CMX_BUF_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <5440> -> memref<288x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[CMX_BUF_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <5456> -> memref<288x8xf16, [@CMX_NN, 1]>
    //CHECK:    [[CMX_BUF_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <5440> -> memref<288x8xf16, [@CMX_NN, 1]>




    //CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR7]] : !VPURT.Barrier) updates([[BAR8]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }


    //CHECK:    VPURT.Task waits([[BAR8]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:     dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 8 : i64, len = 576 : i64, srcWidth = 576 : i64, srcStride = 2 : i64, srcPlaneStride = 1152 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:     }
    //CHECK-SAME:     inputs([[INPUT_BUF_1]] : memref<8x288xf16, @DDR>) outputs([[CMX_BUF_1]] : memref<288x8xf16, [@CMX_NN, 0]>) -> memref<288x8xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR8]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:     dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 8 : i64, len = 576 : i64, srcWidth = 576 : i64, srcStride = 2 : i64, srcPlaneStride = 1152 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:     }
    //CHECK-SAME:     inputs([[INPUT_BUF_0]] : memref<8x288xf16, @DDR>) outputs([[CMX_BUF_0]] : memref<288x8xf16, [@CMX_NN, 0]>) -> memref<288x8xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR8]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:     dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 8 : i64, len = 576 : i64, srcWidth = 576 : i64, srcStride = 2 : i64, srcPlaneStride = 1152 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:     }
    //CHECK-SAME:     inputs([[INPUT_BUF_3]] : memref<8x288xf16, @DDR>) outputs([[CMX_BUF_3]] : memref<288x8xf16, [@CMX_NN, 1]>) -> memref<288x8xf16, [@CMX_NN, 1]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR8]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:     dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 8 : i64, len = 576 : i64, srcWidth = 576 : i64, srcStride = 2 : i64, srcPlaneStride = 1152 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:     }
    //CHECK-SAME:     inputs([[INPUT_BUF_2]] : memref<8x288xf16, @DDR>) outputs([[CMX_BUF_2]] : memref<288x8xf16, [@CMX_NN, 1]>) -> memref<288x8xf16, [@CMX_NN, 1]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR9]] : !VPURT.Barrier) updates([[BAR10]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA inputs([[CST:%.+]] : memref<1x1x1x16xui8>) outputs(%33 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    }

    // nce task
    //CHECK:    VPURT.Task waits([[BAR10]] : !VPURT.Barrier) updates([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR10]] : !VPURT.Barrier) updates([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:    }

    // copy back
    //CHECK:    VPURT.Task waits([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA {port = 1 : i64}
    //CHECK:    }
    //CHECK:    return %0 : memref<1x3x24x24xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>


// CHECK-LABEL: @PermuteToDMAWithHWCToWHC
func.func @PermuteToDMAWithHWCToWHC() -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x76xf16, #map, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = #map1}
                inputs(%input: memref<1x16x4x76xf16, #map, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %output: memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[VAR0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4864> -> memref<8x4x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<8x4x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[VAR3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <10944> -> memref<4x8x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR4:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <9728> -> memref<4x8x76xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:         VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 8 : i64, len = 608 : i64, srcWidth = 608 : i64, srcStride = 2 : i64, srcPlaneStride = 608 : i64, dstWidth = 152 : i64, dstStride = 2432 : i64, dstPlaneStride = 152 : i64>, port = 0 : i64}
    //CHECK-SAME:    inputs([[VAR1]] : memref<8x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<4x8x76xf16, [@CMX_NN, 0]>) -> memref<4x8x76xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:         {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 8 : i64, len = 608 : i64, srcWidth = 608 : i64, srcStride = 2 : i64, srcPlaneStride = 608 : i64, dstWidth = 152 : i64, dstStride = 2432 : i64, dstPlaneStride = 152 : i64>, port = 1 : i64}
    //CHECK-SAME:    inputs([[VAR0]] : memref<8x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<4x8x76xf16, [@CMX_NN, 0]>) -> memref<4x8x76xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[VAR2]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-LABEL: @PermuteToDMAWithHWCToHCW
func.func @PermuteToDMAWithHWCToHCW() -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = #map}
                inputs(%input: memref<1x16x4x76xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %output: memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[VAR0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <304> -> memref<16x2x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x2x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[VAR3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <9732> -> memref<16x76x2xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR4:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <9728> -> memref<16x76x2xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 2432 : i64, srcWidth = 152 : i64, srcStride = 608 : i64, srcPlaneStride = 152 : i64, dstWidth = 2 : i64, dstStride = 8 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    //CHECK-SAME: inputs([[VAR1]] : memref<16x2x76xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<16x76x2xf16, [@CMX_NN, 0]>) -> memref<16x76x2xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 2432 : i64, srcWidth = 152 : i64, srcStride = 608 : i64, srcPlaneStride = 152 : i64, dstWidth = 2 : i64, dstStride = 8 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    //CHECK-SAME: inputs([[VAR0]] : memref<16x2x76xf16, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<16x76x2xf16, [@CMX_NN, 0]>) -> memref<16x76x2xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[VAR2]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteSEGMENTED
func.func @ClusterPermuteDMAWithExpandAndPermuteSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x32x32x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : memref<1x3x32x32x!qElemType, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <1024> -> memref<2x512x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x!qElemType, @DDR>
    //CHECK:    [[INPUT2:%.*]] = VPURT.DeclareBuffer <DDR> <1536> -> memref<2x512x!qElemType, @DDR>
    //CHECK:    [[INPUT3:%.*]] = VPURT.DeclareBuffer <DDR> <512> -> memref<1x512x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1> -> memref<512x16x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<512x16x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <1> -> memref<512x16x!qElemType, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64, dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<512x16x!qElemType, [@CMX_NN, 0]>) -> memref<512x16x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64, dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<2x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<512x16x!qElemType, [@CMX_NN, 0]>) -> memref<512x16x!qElemType, [@CMX_NN, 0]>


    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64, dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<1x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT3]] : memref<512x16x!qElemType, [@CMX_NN, 1]>) -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64, dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<2x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT2]] : memref<512x16x!qElemType, [@CMX_NN, 1]>) -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteExplicitSEGMENTED
func.func @ClusterPermuteDMAWithExpandAndPermuteExplicitSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x32x32x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #NHWC}
              inputs(%0 : memref<1x3x32x32x!qElemType, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <1024> -> memref<2x512x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x!qElemType, @DDR>
    //CHECK:    [[INPUT2:%.*]] = VPURT.DeclareBuffer <DDR> <1536> -> memref<2x512x!qElemType, @DDR>
    //CHECK:    [[INPUT3:%.*]] = VPURT.DeclareBuffer <DDR> <512> -> memref<1x512x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1> -> memref<512x16x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<512x16x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <1> -> memref<512x16x!qElemType, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:     numPlanes = 1 : i64, len = 512 : i64,
    //CHECK-SAME:     srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64,
    //CHECK-SAME:     dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<512x16x!qElemType, [@CMX_NN, 0]>)
    //CHECK-SAME:   -> memref<512x16x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:   numPlanes = 2 : i64, len = 512 : i64,
    //CHECK-SAME:   srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64,
    //CHECK-SAME:   dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<2x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<512x16x!qElemType, [@CMX_NN, 0]>)
    //CHECK-SAME:   -> memref<512x16x!qElemType, [@CMX_NN, 0]>


    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:     numPlanes = 1 : i64, len = 512 : i64,
    //CHECK-SAME:     srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64,
    //CHECK-SAME:     dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<1x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT3]] : memref<512x16x!qElemType, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:   numPlanes = 2 : i64, len = 512 : i64,
    //CHECK-SAME:   srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64,
    //CHECK-SAME:   dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<2x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT2]] : memref<512x16x!qElemType, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:                             {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}:                     compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    //CHECK-SAME{LITERAL}:                     memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}:                     memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    strides = [2, 2],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteOVERLAPPED
func.func @ClusterPermuteDMAWithExpandAndPermuteOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : memref<1x3x224x224x!qElemType, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <50176> -> memref<2x25536x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x25536x!qElemType, @DDR>
    //CHECK:    [[INPUT2:%.*]] = VPURT.DeclareBuffer <DDR> <74592> -> memref<2x25760x!qElemType, @DDR>
    //CHECK:    [[INPUT3:%.*]] = VPURT.DeclareBuffer <DDR> <24416> -> memref<1x25760x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1> -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <1> -> memref<25760x4x!qElemType, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 25536 : i64, srcWidth = 25536 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64, dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x25536x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<25536x4x!qElemType, [@CMX_NN, 0]>) -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 25536 : i64, srcWidth = 25536 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64, dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<2x25536x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<25536x4x!qElemType, [@CMX_NN, 0]>) -> memref<25536x4x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 25760 : i64, srcWidth = 25760 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64, dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<1x25760x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT3]] : memref<25760x4x!qElemType, [@CMX_NN, 1]>) -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 25760 : i64, srcWidth = 25760 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64, dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<2x25760x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT2]] : memref<25760x4x!qElemType, [@CMX_NN, 1]>) -> memref<25760x4x!qElemType, [@CMX_NN, 1]>


    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    memory_shapes = [[1, 4, 114, 224], [1, 4, 115, 224]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 109, 0]]
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteExplicitOVERLAPPED
func.func @ClusterPermuteDMAWithExpandAndPermuteExplicitOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : memref<1x3x224x224x!qElemType, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <50176> -> memref<2x25536x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x25536x!qElemType, @DDR>
    //CHECK:    [[INPUT2:%.*]] = VPURT.DeclareBuffer <DDR> <74592> -> memref<2x25760x!qElemType, @DDR>
    //CHECK:    [[INPUT3:%.*]] = VPURT.DeclareBuffer <DDR> <24416> -> memref<1x25760x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 114, 224], [1, 4, 115, 224]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 109, 0]]}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1> -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <1> -> memref<25760x4x!qElemType, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:     numPlanes = 1 : i64, len = 25536 : i64,
    //CHECK-SAME:     srcWidth = 25536 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64,
    //CHECK-SAME:     dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x25536x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<25536x4x!qElemType, [@CMX_NN, 0]>)
    //CHECK-SAME:     -> memref<25536x4x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:     numPlanes = 2 : i64, len = 25536 : i64,
    //CHECK-SAME:     srcWidth = 25536 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64,
    //CHECK-SAME:     dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<2x25536x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<25536x4x!qElemType, [@CMX_NN, 0]>)
    //CHECK-SAME:   -> memref<25536x4x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:     numPlanes = 1 : i64, len = 25760 : i64,
    //CHECK-SAME:     srcWidth = 25760 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64,
    //CHECK-SAME:     dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<1x25760x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT3]] : memref<25760x4x!qElemType, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:   numPlanes = 2 : i64, len = 25760 : i64,
    //CHECK-SAME:   srcWidth = 25760 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64,
    //CHECK-SAME:   dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<2x25760x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT2]] : memref<25760x4x!qElemType, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<25760x4x!qElemType, [@CMX_NN, 1]>


    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:                             {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    //CHECK-SAME{LITERAL}:                     compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    //CHECK-SAME{LITERAL}:                     memory_shapes = [[1, 4, 114, 224], [1, 4, 115, 224]],
    //CHECK-SAME{LITERAL}:                     memory_offsets = [[0, 0, 0, 0], [0, 0, 109, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWCForNetworkOutput
func.func @PermuteToDMAWithNCHWToNHWCForNetworkOutput() -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x14x7xf16, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
      VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}
            inputs(%input : memref<1x32x14x7xf16, @DDR>)
            outputs(%output : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
   }

    return %output: memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <3136> -> memref<16x98xf16, @DDR>
    //CHECK:    [[INPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<16x98xf16, @DDR>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6304> -> memref<98x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<98x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 196 : i64, srcWidth = 196 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_1]] : memref<16x98xf16, @DDR>)
    //CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<98x16xf16, [@CMX_NN, 0]>) -> memref<98x16xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 196 : i64, srcWidth = 196 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_0]] : memref<16x98xf16, @DDR>)
    //CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<98x16xf16, [@CMX_NN, 0]>) -> memref<98x16xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWCForNetworkInput
func.func @PermuteToDMAWithNCHWToNHWCForNetworkInput() -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x32x14x7xf16, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
      VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}
            inputs(%input : memref<1x32x14x7xf16, @DDR>)
            outputs(%output : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
   }

    return %output: memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> <3136> -> memref<16x98xf16, @DDR>
    //CHECK:    [[INPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<16x98xf16, @DDR>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6304> -> memref<98x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<98x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 196 : i64, srcWidth = 196 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_1]] : memref<16x98xf16, @DDR>)
    //CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<98x16xf16, [@CMX_NN, 0]>) -> memref<98x16xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 196 : i64, srcWidth = 196 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER_0]] : memref<16x98xf16, @DDR>)
    //CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<98x16xf16, [@CMX_NN, 0]>) -> memref<98x16xf16, [@CMX_NN, 0]>
    //CHECK:    }


    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x14x7xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteToDMAWithNCHWToNHWCForNetworkOutput
func.func @ClusterPermuteToDMAWithNCHWToNHWCForNetworkOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x14x7xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #NHWC}
              inputs(%0 : memref<1x32x14x7xf16, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <3136> -> memref<16x49xf16, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<16x49xf16, @DDR>
    //CHECK:    [[INPUT2:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <3234> -> memref<16x49xf16, @DDR>
    //CHECK:    [[INPUT3:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <98> -> memref<16x49xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<49x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<49x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32> -> memref<49x16xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<49x16xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<49x16xf16, [@CMX_NN, 0]>) -> memref<49x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<49x16xf16, [@CMX_NN, 0]>) -> memref<49x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT3]] : memref<49x16xf16, [@CMX_NN, 1]>) -> memref<49x16xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT2]] : memref<49x16xf16, [@CMX_NN, 1]>) -> memref<49x16xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x14x7xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteToDMAWithNCHWToNHWCForNetworkInput
func.func @ClusterPermuteToDMAWithNCHWToNHWCForNetworkInput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x32x14x7xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #NHWC}
              inputs(%0 : memref<1x32x14x7xf16, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> <3136> -> memref<16x49xf16, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<16x49xf16, @DDR>
    //CHECK:    [[INPUT2:%.*]] = VPURT.DeclareBuffer <NetworkInput> <3234> -> memref<16x49xf16, @DDR>
    //CHECK:    [[INPUT3:%.*]] = VPURT.DeclareBuffer <NetworkInput> <98> -> memref<16x49xf16, @DDR>

    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<49x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<49x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32> -> memref<49x16xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<49x16xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<49x16xf16, [@CMX_NN, 0]>) -> memref<49x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<49x16xf16, [@CMX_NN, 0]>) -> memref<49x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT3]] : memref<49x16xf16, [@CMX_NN, 1]>) -> memref<49x16xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 98 : i64, srcWidth = 98 : i64, srcStride = 2 : i64, srcPlaneStride = 196 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<16x49xf16, @DDR>)
    //CHECK:                outputs([[OUTPUT2]] : memref<49x16xf16, [@CMX_NN, 1]>) -> memref<49x16xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithDistributedInputAndOutput
func.func @ClusterPermuteDMAWithDistributedInputAndOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <2000> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : !InputDistributed)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed
    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<32x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <2000>
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[BUFF_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2032>
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[BUFF_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000>
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 32 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64, dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHECK-SAME:           inputs([[INPUT_1]] : memref<32x4x!qElemType, [@CMX_NN, 0]>) outputs([[BUFF_1]] : !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    }

    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 32 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64, dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHECK-SAME:           inputs([[INPUT_0]] : memref<32x4x!qElemType, [@CMX_NN, 0]>) outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    }

    // CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @ClusterPermuteDMAWithDistributedExplicitInputAndOutput
func.func @ClusterPermuteDMAWithDistributedExplicitInputAndOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <2000> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : !InputDistributed)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed
    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<32x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <2000>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:    [[BUFF_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2032>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[4, 32], [4, 32]], compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[4, 32], [4, 32]], memory_offsets = [[0, 0], [0, 0]]

    // CHECK:    [[BUFF_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000>
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[4, 32], [4, 32]], compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[4, 32], [4, 32]], memory_offsets = [[0, 0], [0, 0]]

    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:      dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:        numPlanes = 32 : i64, len = 4 : i64,
    // CHECK-SAME:        srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64,
    // CHECK-SAME:        dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHECK-SAME:           inputs([[INPUT_1]] : memref<32x4x!qElemType, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[BUFF_1]] : !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                         compute_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:                         compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:                         memory_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:                         memory_offsets = [[0, 0], [0, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0], [0, 0]]
    // CHECK:    }

    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.PermuteDMA {
    // CHECK-SAME:      dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:        numPlanes = 32 : i64, len = 4 : i64,
    // CHECK-SAME:        srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64,
    // CHECK-SAME:        dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHECK-SAME:           inputs([[INPUT_0]] : memref<32x4x!qElemType, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                         compute_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:                         compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:                         memory_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:                         memory_offsets = [[0, 0], [0, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<4x32x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[4, 32], [4, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0], [0, 0]]}>
    // CHECK:    }

    // CHECK:    return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x72x2x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWC2D
func.func @PermuteToDMAWithNCHWToNHWC2D() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x72x2x1xf16, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %18 = VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64} inputs(%input : memref<1x72x2x1xf16, @DDR>) outputs(%output : !OutputDistributed) -> !OutputDistributed
    }
    return %output: !VPUIP.DistributedBuffer<1x72x2x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> <144> -> memref<36x1xf16, @DDR>
    // CHECK:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<36x1xf16, @DDR>
    // CHECK:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer <NetworkInput> <146> -> memref<36x1xf16, @DDR>
    // CHECK:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer <NetworkInput> <2> -> memref<36x1xf16, @DDR>

    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x72x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[BUFF_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1x36xf16, [@CMX_NN, 0]>
    // CHECK:    [[BUFF_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x36xf16, [@CMX_NN, 0]>
    // CHECK:    [[BUFF_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <72> -> memref<1x36xf16, [@CMX_NN, 1]>
    // CHECK:    [[BUFF_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x36xf16, [@CMX_NN, 1]>

    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:    VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 36 : i64, len = 2 : i64, srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 144 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    // CHECK-SAME:  inputs([[INPUT_1]] : memref<36x1xf16, @DDR>) outputs([[BUFF_1]] : memref<1x36xf16, [@CMX_NN, 0]>) -> memref<1x36xf16, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:    VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 36 : i64, len = 2 : i64, srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 144 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}
    // CHECK-SAME:  inputs([[INPUT_0]] : memref<36x1xf16, @DDR>) outputs([[BUFF_0]] : memref<1x36xf16, [@CMX_NN, 0]>) -> memref<1x36xf16, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:    VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 36 : i64, len = 2 : i64, srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 144 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    // CHECK-SAME:  inputs([[INPUT_3]] : memref<36x1xf16, @DDR>) outputs([[BUFF_3]] : memref<1x36xf16, [@CMX_NN, 1]>) -> memref<1x36xf16, [@CMX_NN, 1]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:    VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 36 : i64, len = 2 : i64, srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 144 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64}
    // CHECK-SAME:  inputs([[INPUT_2]] : memref<36x1xf16, @DDR>) outputs([[BUFF_2]] : memref<1x36xf16, [@CMX_NN, 1]>) -> memref<1x36xf16, [@CMX_NN, 1]>
    // CHECK:    }


    // CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x72x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniformPermuteDMAPlaneSizeRequiresTwoDMAs
func.func @UniformPermuteDMAPlaneSizeRequiresTwoDMAs() -> memref<1x16x1x261xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x261xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x1x261xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x16x1x261xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x1x261xf16, [@CMX_NN, 0]>) -> memref<1x16x1x261xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x16x1x261xf16, [@CMX_NN, 0]>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4160> -> memref<131x16xf16, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<130x16xf16, [@CMX_NN, 0]>
    // CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x1x261xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFFER_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4356> -> memref<16x131xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFFER_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<16x130xf16, [@CMX_NN, 0]>

    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:         dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 130 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 522 : i64, dstPlaneStride = 2 : i64>
    // CHECK-SAME:         port = 0 : i64
    // CHECK:            inputs([[INPUT_BUFFER_1]] : memref<130x16xf16, [@CMX_NN, 0]>)
    // CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<16x130xf16, [@CMX_NN, 0]>) -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:         dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 131 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 522 : i64, dstPlaneStride = 2 : i64>
    // CHECK-SAME:         port = 1 : i64
    // CHECK:            inputs([[INPUT_BUFFER_0]] : memref<131x16xf16, [@CMX_NN, 0]>)
    // CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<16x131xf16, [@CMX_NN, 0]>) -> memref<16x131xf16, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    return [[RETURN_BUFFER]] : memref<1x16x1x261xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniformPermuteDMAPlaneSizeRequiresFourDMAs
func.func @UniformPermuteDMAPlaneSizeRequiresFourDMAs() -> memref<1x16x1x520xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x520xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x1x520xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x16x1x520xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x1x520xf16, [@CMX_NN, 0]>) -> memref<1x16x1x520xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x16x1x520xf16, [@CMX_NN, 0]>

    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12480> -> memref<130x16xf16, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8320> -> memref<130x16xf16, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4160> -> memref<130x16xf16, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<130x16xf16, [@CMX_NN, 0]>
    // CHECK:    [[RETURN_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x1x520xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4876> -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4616> -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4356> -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<16x130xf16, [@CMX_NN, 0]>

    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:         dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 130 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 1040 : i64, dstPlaneStride = 2 : i64>
    // CHECK-SAME:         port = 0 : i64
    // CHECK:            inputs([[INPUT_BUFFER_3]] : memref<130x16xf16, [@CMX_NN, 0]>)
    // CHECK:            outputs([[OUTPUT_BUFFER_3]] : memref<16x130xf16, [@CMX_NN, 0]>) -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:         dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 130 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 1040 : i64, dstPlaneStride = 2 : i64>
    // CHECK-SAME:         port = 1 : i64
    // CHECK:            inputs([[INPUT_BUFFER_2]] : memref<130x16xf16, [@CMX_NN, 0]>)
    // CHECK:            outputs([[OUTPUT_BUFFER_2]] : memref<16x130xf16, [@CMX_NN, 0]>) -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:         dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 130 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 1040 : i64, dstPlaneStride = 2 : i64>
    // CHECK-SAME:         port = 0 : i64
    // CHECK:            inputs([[INPUT_BUFFER_1]] : memref<130x16xf16, [@CMX_NN, 0]>)
    // CHECK:            outputs([[OUTPUT_BUFFER_1]] : memref<16x130xf16, [@CMX_NN, 0]>) -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.PermuteDMA {
    // CHECK-SAME:         dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 130 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 1040 : i64, dstPlaneStride = 2 : i64>
    // CHECK-SAME:         port = 1 : i64
    // CHECK:            inputs([[INPUT_BUFFER_0]] : memref<130x16xf16, [@CMX_NN, 0]>)
    // CHECK:            outputs([[OUTPUT_BUFFER_0]] : memref<16x130xf16, [@CMX_NN, 0]>) -> memref<16x130xf16, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    return [[RETURN_BUFFER]] : memref<1x16x1x520xf16, [@CMX_NN, 0]>
}
