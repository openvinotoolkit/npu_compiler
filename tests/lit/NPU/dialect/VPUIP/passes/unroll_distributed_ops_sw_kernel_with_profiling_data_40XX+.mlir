//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistBuf = !VPUIP.DistributedBuffer<1x5x262144x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 5, 1, 1],
    num_clusters = 5 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 262144, 1], [1, 1, 262144, 1], [1, 1, 262144, 1], [1, 1, 262144, 1], [1, 1, 262144, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0], [0, 4, 0, 0]],
    memory_shapes = [[1, 1, 262144, 1], [1, 1, 262144, 1], [1, 1, 262144, 1], [1, 1, 262144, 1], [1, 1, 262144, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0], [0, 4, 0, 0]]
}>

!ProfDistBuf = !VPUIP.DistributedBuffer<48xui32, {order = affine_map<(d0) -> (d0)>, strides = [1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [6],
    num_clusters = 6 : i64,
    uniform_distributed_segments
}>

!DummyT = memref<1x3x224x224xf16, @DDR>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollDistributedOpsWithProfilingData
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x3x224x224xf16, @DDR>)
func.func @UnrollDistributedOpsWithProfilingData(%arg0: !DummyT) -> !DummyT {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <CMX_NN> <524544> -> !DistBuf
    %1 = VPURT.DeclareBuffer <CMX_NN> <256> -> !DistBuf
    %2 = VPURT.DeclareBuffer <CMX_NN> <160> -> !ProfDistBuf

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %results, %profiling_output = VPUIP.SW.Kernel {
            profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 0 : i64>,
            resultSegmentSizes = array<i32: 1, 0, 1>
        } @VPU.SW::@builtin_MVN
            inputs(%0 as %arg1: memref<1x5x262144x1xf16, @CMX_NN>)
            outputs(%1 as %arg2: memref<1x5x262144x1xf16, @CMX_NN>)
            profiling_data(%2 : !ProfDistBuf) on tile 0 -> (!DistBuf, !ProfDistBuf) {
                VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999999747524271E-7]}(%arg1, %arg2) : memref<1x5x262144x1xf16, @CMX_NN>, memref<1x5x262144x1xf16, @CMX_NN>
            }
    }

    return %arg0 : !DummyT

    // CHECK:       [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       [[CMX_0_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <524544> -> memref<1x1x262144x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[CMX_1_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <524544> -> memref<1x1x262144x1xf16, [@CMX_NN, 1]>
    // CHECK:       [[CMX_2_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <524544> -> memref<1x1x262144x1xf16, [@CMX_NN, 2]>
    // CHECK:       [[CMX_3_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <524544> -> memref<1x1x262144x1xf16, [@CMX_NN, 3]>
    // CHECK:       [[CMX_4_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <524544> -> memref<1x1x262144x1xf16, [@CMX_NN, 4]>

    // CHECK:       [[CMX_0_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x1x262144x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[CMX_1_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <256> -> memref<1x1x262144x1xf16, [@CMX_NN, 1]>
    // CHECK:       [[CMX_2_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <256> -> memref<1x1x262144x1xf16, [@CMX_NN, 2]>
    // CHECK:       [[CMX_3_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <256> -> memref<1x1x262144x1xf16, [@CMX_NN, 3]>
    // CHECK:       [[CMX_4_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <256> -> memref<1x1x262144x1xf16, [@CMX_NN, 4]>

    // CHECK:       [[PROF_BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <160> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[PROF_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <160> -> memref<8xui32, [@CMX_NN, 1]>
    // CHECK:       [[PROF_BUF_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <160> -> memref<8xui32, [@CMX_NN, 2]>
    // CHECK:       [[PROF_BUF_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <160> -> memref<8xui32, [@CMX_NN, 3]>
    // CHECK:       [[PROF_BUF_4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <160> -> memref<8xui32, [@CMX_NN, 4]>

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 0 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_0_IN]] as [[ARG0:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[CMX_0_OUT]] as [[ARG1:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_0]] : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x262144x1xf16, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999999747524271E-7]}([[ARG0]], [[ARG1]]) : memref<1x1x262144x1xf16, [@CMX_NN, 0]>, memref<1x1x262144x1xf16, [@CMX_NN, 0]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 1 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_1_IN]] as [[ARG2:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 1]>)
    // CHECK-SAME:          outputs([[CMX_1_OUT]] as [[ARG3:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 1]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_1]] : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x262144x1xf16, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999999747524271E-7]}([[ARG2]], [[ARG3]]) : memref<1x1x262144x1xf16, [@CMX_NN, 1]>, memref<1x1x262144x1xf16, [@CMX_NN, 1]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 2 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_2_IN]] as [[ARG4:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 2]>)
    // CHECK-SAME:          outputs([[CMX_2_OUT]] as [[ARG5:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 2]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_2]] : memref<8xui32, [@CMX_NN, 2]>) on tile 2 -> (memref<1x1x262144x1xf16, [@CMX_NN, 2]>, memref<8xui32, [@CMX_NN, 2]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999999747524271E-7]}([[ARG4]], [[ARG5]]) : memref<1x1x262144x1xf16, [@CMX_NN, 2]>, memref<1x1x262144x1xf16, [@CMX_NN, 2]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 3 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_3_IN]] as [[ARG6:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 3]>)
    // CHECK-SAME:          outputs([[CMX_3_OUT]] as [[ARG7:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 3]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_3]] : memref<8xui32, [@CMX_NN, 3]>) on tile 3 -> (memref<1x1x262144x1xf16, [@CMX_NN, 3]>, memref<8xui32, [@CMX_NN, 3]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999999747524271E-7]}([[ARG6]], [[ARG7]]) : memref<1x1x262144x1xf16, [@CMX_NN, 3]>, memref<1x1x262144x1xf16, [@CMX_NN, 3]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 4 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_4_IN]] as [[ARG8:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 4]>)
    // CHECK-SAME:          outputs([[CMX_4_OUT]] as [[ARG9:[^:]+]]: memref<1x1x262144x1xf16, [@CMX_NN, 4]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_4]] : memref<8xui32, [@CMX_NN, 4]>) on tile 4 -> (memref<1x1x262144x1xf16, [@CMX_NN, 4]>, memref<8xui32, [@CMX_NN, 4]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999999747524271E-7]}([[ARG8]], [[ARG9]]) : memref<1x1x262144x1xf16, [@CMX_NN, 4]>, memref<1x1x262144x1xf16, [@CMX_NN, 4]>
    // CHECK:               }
    // CHECK:       }
    // CHECK:  return [[ARG_0]] : memref<1x3x224x224xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuf = !VPUIP.DistributedBuffer<1x192x16x48xf16, {order = #NHWC, strides = [294912, 1, 18432, 384]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 16, 48], [1, 64, 16, 48], [1, 64, 16, 48]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 64, 16, 48], [1, 64, 16, 48], [1, 64, 16, 48]],
    memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0]]
}>


!ProfDistBuf = !VPUIP.DistributedBuffer<24xui32, affine_map<(d0) -> (d0)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3],
    num_clusters = 3 : i64,
    uniform_distributed_segments
}>


!DummyT = memref<1x192x16x48xf16, @DDR>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollDistributedOpsWithProfilingDataAndOutputStrides
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x192x16x48xf16, @DDR>)
func.func @UnrollDistributedOpsWithProfilingDataAndOutputStrides(%arg0: !DummyT) -> !DummyT {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <CMX_NN> <589888> -> !DistBuf
    %1 = VPURT.DeclareBuffer <CMX_NN> <64> -> !DistBuf
    %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !ProfDistBuf

    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
        %results, %profiling_output = VPUIP.SW.Kernel {
            profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 0 : i64>,
            resultSegmentSizes = array<i32: 1, 0, 1>
        } @VPU.SW::@builtin_MVN
            inputs(%0 as %arg1: !DistBuf)
            outputs(%1 as %arg2: !DistBuf)
            profiling_data(%2 : !ProfDistBuf)
            outputStrides([[98304, 1, 6144, 128], [98304, 1, 6144, 128], [98304, 1, 6144, 128]]) on tile 0 -> (!DistBuf, !ProfDistBuf) {
                VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999997473787516E-6]}(%arg1, %arg2) : !DistBuf, !DistBuf
            }
    }

    // CHECK:       [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       [[CMX_0_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589888> -> memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>
    // CHECK:       [[CMX_1_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <589888> -> memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>
    // CHECK:       [[CMX_2_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <589888> -> memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>

    // CHECK:       [[CMX_0_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>
    // CHECK:       [[CMX_1_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <64> -> memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>
    // CHECK:       [[CMX_2_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <64> -> memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>

    // CHECK:       [[PROF_BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[PROF_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<8xui32, [@CMX_NN, 1]>
    // CHECK:       [[PROF_BUF_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<8xui32, [@CMX_NN, 2]>

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           %results, %profiling_output = VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 0 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_0_IN]] as [[ARG0:[^:]+]]: memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[CMX_0_OUT]] as [[ARG1:[^:]+]]: memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_0]] : memref<8xui32, [@CMX_NN, 0]>) outputStrides({{\[\[}}98304, 1, 6144, 128], [98304, 1, 6144, 128], [98304, 1, 6144, 128]]) on tile 0 -> (memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999997473787516E-6]}([[ARG0]], [[ARG1]]) : memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>, memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 0]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           %results, %profiling_output = VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 1 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_1_IN]] as [[ARG2:[^:]+]]: memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>)
    // CHECK-SAME:          outputs([[CMX_1_OUT]] as [[ARG3:[^:]+]]: memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_1]] : memref<8xui32, [@CMX_NN, 1]>) outputStrides({{\[\[}}98304, 1, 6144, 128], [98304, 1, 6144, 128], [98304, 1, 6144, 128]]) on tile 1 -> (memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999997473787516E-6]}([[ARG2]], [[ARG3]]) : memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>, memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 1]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           %results, %profiling_output = VPUIP.SW.Kernel {
    // CHECK-SAME:          profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 2 : i64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1>
    // CHECK-SAME:      } @VPU.SW::@builtin_MVN
    // CHECK-SAME:          inputs([[CMX_2_IN]] as [[ARG4:[^:]+]]: memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          outputs([[CMX_2_OUT]] as [[ARG5:[^:]+]]: memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          profiling_data([[PROF_BUF_2]] : memref<8xui32, [@CMX_NN, 2]>) outputStrides({{\[\[}}98304, 1, 6144, 128], [98304, 1, 6144, 128], [98304, 1, 6144, 128]]) on tile 2 -> (memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>, memref<8xui32, [@CMX_NN, 2]>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999997473787516E-6]}([[ARG4]], [[ARG5]]) : memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>, memref<1x64x16x48xf16, {order = #NHWC, strides = [98304, 1, 6144, 128]}, [@CMX_NN, 2]>
    // CHECK:               }
    // CHECK:       }

    // CHECK:  return [[ARG_0]] : memref<1x192x16x48xf16, @DDR>

    return %arg0 : !DummyT
}
