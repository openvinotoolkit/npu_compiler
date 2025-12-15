//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --unroll-sw-kernel="enable-sw-kernel-fifo-per-shave-engine=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @UnrollSwKernel()
        -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) {

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%2 as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, %4 as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 as %arg2: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, %5 as %arg3: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>){
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg2) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
        }
    }
    return %3, %5: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>


    // CHECK:   [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[TILE0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[TILE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[TILE0]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT0]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[TILE1]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT1]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 list 1 -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   return [[OUTPUT0]], [[OUTPUT1]] : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
}

// -----
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// Same as UnrollSwKernel, checks only for correct profiling unrolling
func.func @UnrollSwKernelWithProfiling()
        -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) {

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<8xui32, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results:2, %profiling_output = VPUIP.SW.Kernel {
            profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 2 : i64, dataIndex = 0 : i64>,
            resultSegmentSizes = array<i32: 2, 0, 1>
            } @VPU.SW::@builtin_MVN
              inputs(%2 as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, %4 as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>)
              outputs(%3 as %arg2: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, %5 as %arg3: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>)
              profiling_data(%6 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg2) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
        }
    }
    return %3, %5: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>


    // CHECK:   [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[TILE0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[TILE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[PROF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:                  profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 2 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>,
    // CHECK-SAME:                  resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                  inputs([[TILE0]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  outputs([[OUTPUT0]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  profiling_data([[PROF0]] : memref<8xui32, [@CMX_NN, 0]>)
    // CHECK-SAME:                  on tile 0 list 0 -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:   }

    // CHECK:   [[PROF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <160> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:   VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:                  profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 2 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 0 : i64>,
    // CHECK-SAME:                  resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                  inputs([[TILE1]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  outputs([[OUTPUT1]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  profiling_data([[PROF1]] : memref<8xui32, [@CMX_NN, 0]>)
    // CHECK-SAME:                  on tile 0 list 1 -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   return [[OUTPUT0]], [[OUTPUT1]] : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>


!DistributedT = !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @DistributedBufferUnrollSwKernel() -> (!DistributedT, !DistributedT) {

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <663616> -> !DistributedT
    %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <925760> -> !DistributedT
    %4 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1187904> -> !DistributedT
    %5 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1450048> -> !DistributedT

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%2 as %arg0: !DistributedT, %4 as %arg1: !DistributedT) outputs(%3 as %arg2: !DistributedT, %5 as %arg3: !DistributedT) outputStrides([[131072, 1, 64, 2048], [131072, 1, 64, 2048]]) on tile 0 -> (!DistributedT, !DistributedT){
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg2) : !DistributedT, !DistributedT
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : !DistributedT, !DistributedT
      }
    }
    return %3, %5: !DistributedT, !DistributedT


    // CHECK:   [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[TILE0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <663616> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <925760> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[TILE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1187904> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1450048> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[TILE0]] as %arg0: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs([[OUTPUT0]] as %arg1: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputStrides({{\[\[}}131072, 1, 64, 2048], [131072, 1, 64, 2048]]) on tile 0 list 0
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[TILE1]] as %arg0: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs([[OUTPUT1]] as %arg1: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputStrides({{\[\[}}131072, 1, 64, 2048], [131072, 1, 64, 2048]]) on tile 0 list 1 -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   return [[OUTPUT0]], [[OUTPUT1]] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>


!DistributedT = !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!DistributedProfType = !VPUIP.DistributedBuffer<16xui32, {order = affine_map<(d0) -> (d0)>, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


// Same as @DistributedBufferUnrollSwKernel, but with profiling metadata
func.func @DistributedBufferUnrollSwKernelWithProfiling() -> (!DistributedT, !DistributedT) {

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <663616> -> !DistributedT
    %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <925760> -> !DistributedT
    %4 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1187904> -> !DistributedT
    %5 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1450048> -> !DistributedT
    %6 = VPURT.DeclareBuffer <CMX_NN> <128> -> !DistributedProfType

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %results:2, %profiling_buffer = VPUIP.SW.Kernel {
            profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 2 : i64, dataIndex = 0 : i64>,
            resultSegmentSizes = array<i32: 2, 0, 1>
        } @VPU.SW::@builtin_MVN
            inputs(%2 as %arg0: !DistributedT, %4 as %arg1: !DistributedT)
            outputs(%3 as %arg2: !DistributedT, %5 as %arg3: !DistributedT)
            profiling_data(%6 : !DistributedProfType)
            outputStrides([[131072, 1, 64, 2048], [131072, 1, 64, 2048]])
            on tile 0 -> (!DistributedT, !DistributedT, !DistributedProfType){
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg2) : !DistributedT, !DistributedT
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : !DistributedT, !DistributedT
      }
    }
    return %3, %5: !DistributedT, !DistributedT


    // CHECK:   [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[TILE0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <663616> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <925760> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[TILE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1187904> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1450048> -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[PROF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> <128> -> !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 2 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>}
    // CHECK-SAME:          @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[TILE0]] as %arg0: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[OUTPUT0]] as %arg1: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          profiling_data([[PROF0]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    // CHECK-SAME:          outputStrides({{\[\[}}131072, 1, 64, 2048], [131072, 1, 64, 2048]]) on tile 0 list 0
    // CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           }
    // CHECK:   }

    // CHECK:   [[PROF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> <160> -> !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    // CHECK:   VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 2 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 0 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>}
    // CHECK-SAME:          @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[TILE1]] as %arg0: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[OUTPUT1]] as %arg1: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          profiling_data([[PROF1]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    // CHECK-SAME:          outputStrides({{\[\[}}131072, 1, 64, 2048], [131072, 1, 64, 2048]]) on tile 0 list 1
    // CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   return [[OUTPUT0]], [[OUTPUT1]] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @UnrollDynamicSwKernel
func.func @UnrollDynamicSwKernel() -> (memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x35x512xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <35840> -> memref<4xsi32, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <330752> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <331008> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <163840> -> memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <331456> -> memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <294912> -> memref<1x1x35x128xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <331328> -> memref<4xsi32, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <303872> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <304128> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>

    %10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %12 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %13 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %14 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %15 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task waits(%10, %11, %12, %13, %14 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%15 : !VPURT.Barrier) {
      %results:6, %dynamicOutputShapes = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 0, -1, -1, -1, -1>,
            dynamicOutputShapesMap = array<i32: 0, -1, -1>,
            resultSegmentSizes = array<i32: 6, 1, 0>
        } @VPU.SW::@builtin_LSTMSequence
            inputs(
              %0 as %arg8: memref<1x1x35x512xf16, [@CMX_NN, 0]>,
              %2 as %arg9: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %3 as %arg10: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %4 as %arg11: memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>,
              %5 as %arg12: memref<1x1x1x2xsi32, [@CMX_NN, 0]>,
              %0 as %arg13: memref<1x1x35x512xf16, [@CMX_NN, 0]>,
              %2 as %arg14: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %3 as %arg15: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %4 as %arg16: memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>,
              %5 as %arg17: memref<1x1x1x2xsi32, [@CMX_NN, 0]>
            )
            dynamicInputShapes(%1 : memref<4xsi32, [@CMX_NN, 0]>)
            outputs(
              %6 as %arg18: memref<1x1x35x128xf16, [@CMX_NN, 0]>,
              %8 as %arg19: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %9 as %arg20: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %6 as %arg21: memref<1x1x35x128xf16, [@CMX_NN, 0]>,
              %8 as %arg22: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
              %9 as %arg23: memref<1x1x1x128xf16, [@CMX_NN, 0]>
            )
            dynamicOutputShapes(%7 : memref<4xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg8, %arg9, %arg10, %arg11, %arg12, %arg18, %arg19, %arg20) :
            memref<1x1x35x512xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>,
            memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>,
            memref<1x1x1x2xsi32, [@CMX_NN, 0]>,
            memref<1x1x35x128xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg13, %arg14, %arg15, %arg16, %arg17, %arg21, %arg22, %arg23) :
            memref<1x1x35x512xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>,
            memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>,
            memref<1x1x1x2xsi32, [@CMX_NN, 0]>,
            memref<1x1x35x128xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>,
            memref<1x1x1x128xf16, [@CMX_NN, 0]>
      }
    }

    return %6, %8, %9 : memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>


    // CHECK:   [[BUFF_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x35x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <35840> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <330752> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <331008> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_4:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <163840> -> memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_5:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <331456> -> memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_6:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <294912> -> memref<1x1x35x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_7:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <331328> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_8:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <303872> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_9:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <304128> -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[BAR_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR_3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR_4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR_5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task waits([[BAR_0]], [[BAR_1]], [[BAR_2:%.*]], [[BAR_3]], [[BAR_4]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%15 : !VPURT.Barrier) {
    // CHECK:     [[RESULTS:%.+]]:3, [[DYN_OUT_SHAPES:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1, -1, -1, -1>, dynamicOutputShapesMap = array<i32: 0, -1, -1>, resultSegmentSizes = array<i32: 3, 1, 0>} @VPU.SW::@builtin_LSTMSequence
    // CHECK:       inputs(
    // CHECK:         [[BUFF_0]] as %arg0: memref<1x1x35x512xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_2]] as %arg1: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_3]] as %arg2: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_4]] as %arg3: memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_5]] as %arg4: memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    // CHECK:       )
    // CHECK:       dynamicInputShapes([[BUFF_1]] : memref<4xsi32, [@CMX_NN, 0]>)
    // CHECK:       outputs(
    // CHECK:         [[BUFF_6]] as %arg5: memref<1x1x35x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_8]] as %arg6: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_9]] as %arg7: memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       )
    // CHECK:       dynamicOutputShapes([[BUFF_7]] : memref<4xsi32, [@CMX_NN, 0]>) on tile 0 list 0 -> (memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [0, 0]}(
    // CHECK:         : memref<1x1x35x512xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:     }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BAR_0]], [[BAR_1]], [[BAR_2:%.*]], [[BAR_3]], [[BAR_4]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates([[BAR_5]] : !VPURT.Barrier) {
    // CHECK:     [[RESULTS:%.+]]:3, [[DYN_OUT_SHAPES:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1, -1, -1, -1>, dynamicOutputShapesMap = array<i32: 0, -1, -1>, resultSegmentSizes = array<i32: 3, 1, 0>} @VPU.SW::@builtin_LSTMSequence
    // CHECK:       inputs(
    // CHECK:         [[BUFF_0]] as %arg0: memref<1x1x35x512xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_2]] as %arg1: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_3]] as %arg2: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_4]] as %arg3: memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_5]] as %arg4: memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    // CHECK:       )
    // CHECK:       dynamicInputShapes([[BUFF_1]] : memref<4xsi32, [@CMX_NN, 0]>)
    // CHECK:       outputs(
    // CHECK:         [[BUFF_6]] as %arg5: memref<1x1x35x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_8]] as %arg6: memref<1x1x1x128xf16, [@CMX_NN, 0]>,
    // CHECK:         [[BUFF_9]] as %arg7: memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       )
    // CHECK:       dynamicOutputShapes([[BUFF_7]] : memref<4xsi32, [@CMX_NN, 0]>) on tile 0 list 1 -> (memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [0, 0]}(
    // CHECK:         : memref<1x1x35x512xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x4x128x128xf16, #NWHC, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:     }
    // CHECK:   }
    // CHECK:   return [[BUFF_6]], [[BUFF_8]], [[BUFF_9]] : memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
}
