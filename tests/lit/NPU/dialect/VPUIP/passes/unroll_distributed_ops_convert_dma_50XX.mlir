//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollConvertDMAWithInputSegmented
func.func @UnrollConvertDMAWithInputSegmented(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <512> -> !InputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_buffer : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%cmx_buffer : !InputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:    [[INPUT_DDR0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[INPUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <1568> -> memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[PARENT_OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:    [[OUTPUT_DDR0:%.+]] = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[OUTPUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <10192> -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[NNCMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x3x14x28xf32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[NNCMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x3x14x28xf32, [@CMX_NN, 1]>

    //CHECK:  [[ARG:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[INPUT_DDR0]] : memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x14x28xf32, [@CMX_NN, 0]>) -> memref<1x3x14x28xf32, [@CMX_NN, 0]>
    //CHECK:  [[ARG:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[INPUT_DDR1]] : memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x14x28xf32, [@CMX_NN, 1]>) -> memref<1x3x14x28xf32, [@CMX_NN, 1]>
    //CHECK:  [[ARG:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x14x28xf32, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR0]] : memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK:  [[ARG:%.+]] = VPUIP.ConvertDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x14x28xf32, [@CMX_NN, 1]>) outputs([[OUTPUT_DDR1]] : memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>

    //CHECK:    return [[PARENT_OUTPUT_DDR]] : memref<1x3x28x28xf16, @DDR>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollConvertDMAWithInputDuplicated
func.func @UnrollConvertDMAWithInputDuplicated(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <512> -> !InputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_buffer : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%cmx_buffer : !InputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:  [[INPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    //CHECK-DAG:  [[OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:  [[CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x3x28x28xf32, [@CMX_NN, 0]>
    //CHECK-DAG:  [[DUPL_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <512> -> !VPUIP.DistributedBuffer<1x3x28x28xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:  [[ARG:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[INPUT_DDR]] : memref<1x3x28x28xf32, @DDR>) outputs([[DUPL_CMX]] : !VPUIP.DistributedBuffer<1x3x28x28xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x28x28xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:  [[ARG:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[CMX_0]] : memref<1x3x28x28xf32, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    //CHECK:  return [[OUTPUT_DDR]] : memref<1x3x28x28xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollConvertDMAWithOutputSegmented
func.func @UnrollConvertDMAWithOutputSegmented(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <512> -> !OutputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_buffer : !OutputDistributed) -> !OutputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cmx_buffer : !OutputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:    [[INPUT_DDR0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[INPUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <1568> -> memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[PARENT_OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:    [[OUTPUT_DDR0:%.+]] = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[OUTPUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <10192> -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK-DAG:    [[NNCMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x3x14x28xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[NNCMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x3x14x28xf16, [@CMX_NN, 1]>

    //CHECK:  [[ARG:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[INPUT_DDR0]] : memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x14x28xf16, [@CMX_NN, 0]>) -> memref<1x3x14x28xf16, [@CMX_NN, 0]>
    //CHECK:  [[ARG:%.+]] = VPUIP.ConvertDMA <{port = 1 : i64}> inputs([[INPUT_DDR1]] : memref<1x3x14x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x14x28xf16, [@CMX_NN, 1]>) -> memref<1x3x14x28xf16, [@CMX_NN, 1]>
    //CHECK:  [[ARG:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x14x28xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR0]] : memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK:  [[ARG:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x14x28xf16, [@CMX_NN, 1]>) outputs([[OUTPUT_DDR1]] : memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x14x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>

    //CHECK:    return [[PARENT_OUTPUT_DDR]] : memref<1x3x28x28xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollConvertDMAWithOutputDuplicated
func.func @UnrollConvertDMAWithOutputDuplicated(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <512> -> !OutputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_buffer : !OutputDistributed) -> !OutputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cmx_buffer : !OutputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:  [[INPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    //CHECK-DAG:  [[OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:  [[CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x3x28x28xf16, [@CMX_NN, 0]>
    //CHECK-DAG:  [[DUPL_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <512> -> !VPUIP.DistributedBuffer<1x3x28x28xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:  [[ARG:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[INPUT_DDR]] : memref<1x3x28x28xf32, @DDR>) outputs([[DUPL_CMX]] : !VPUIP.DistributedBuffer<1x3x28x28xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x28x28xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:  [[ARG:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[CMX_0]] : memref<1x3x28x28xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    //CHECK:  return [[OUTPUT_DDR]] : memref<1x3x28x28xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollConvertDMAWithInputOutputSegmented
func.func @UnrollConvertDMAWithInputOutputSegmented(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_input_buffer = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %cmx_output_buffer = VPURT.DeclareBuffer <CMX_NN> <9408> -> !OutputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_input_buffer : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%cmx_input_buffer : !InputDistributed) outputs(%cmx_output_buffer : !OutputDistributed) -> !OutputDistributed
    }

    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cmx_output_buffer : !OutputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>

    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x14x28xf32, [@CMX_NN, 0]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x14x28xf32, [@CMX_NN, 1]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <9408> -> memref<1x3x14x28xf16, [@CMX_NN, 0]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <9408> -> memref<1x3x14x28xf16, [@CMX_NN, 1]>

    //CHECK: [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x14x28xf32, [@CMX_NN, 0]>) outputs([[ARG:%.+]] : memref<1x3x14x28xf16, [@CMX_NN, 0]>) -> memref<1x3x14x28xf16, [@CMX_NN, 0]>
    //CHECK: [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x14x28xf32, [@CMX_NN, 1]>) outputs([[ARG:%.+]] : memref<1x3x14x28xf16, [@CMX_NN, 1]>) -> memref<1x3x14x28xf16, [@CMX_NN, 1]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollConvertDMAWithInputOutputDuplicated
func.func @UnrollConvertDMAWithInputOutputDuplicated(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_input_buffer = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %cmx_output_buffer = VPURT.DeclareBuffer <CMX_NN> <9408> -> !OutputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_input_buffer : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%cmx_input_buffer : !InputDistributed) outputs(%cmx_output_buffer : !OutputDistributed) -> !OutputDistributed
    }

    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cmx_output_buffer : !OutputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG:  [[INPUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x28x28xf32, [@CMX_NN, 0]>
    //CHECK-DAG:  [[OUTPUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <9408> -> !VPUIP.DistributedBuffer<1x3x28x28xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:  [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[INPUT_CMX]] : memref<1x3x28x28xf32, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX]] : !VPUIP.DistributedBuffer<1x3x28x28xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x28x28xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf32, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollConvertDMAWithInputOutputOverlapped
func.func @UnrollConvertDMAWithInputOutputOverlapped(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_input_buffer = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %cmx_output_buffer = VPURT.DeclareBuffer <CMX_NN> <9408> -> !OutputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_input_buffer : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%cmx_input_buffer : !InputDistributed) outputs(%cmx_output_buffer : !OutputDistributed) -> !OutputDistributed
    }

    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cmx_output_buffer : !OutputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>


    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x15x28xf32, [@CMX_NN, 0]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x15x28xf32, [@CMX_NN, 1]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <9408> -> memref<1x3x15x28xf16, [@CMX_NN, 0]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <9408> -> memref<1x3x15x28xf16, [@CMX_NN, 1]>

    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 0]>) -> memref<1x3x15x28xf32, [@CMX_NN, 0]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 1]>) -> memref<1x3x15x28xf32, [@CMX_NN, 1]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 0]>) outputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 0]>) -> memref<1x3x15x28xf16, [@CMX_NN, 0]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 1]>) outputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 1]>) -> memref<1x3x15x28xf16, [@CMX_NN, 1]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 0]>) outputs([[ARG:%.+]] : memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 1]>) outputs([[ARG:%.+]]: memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf32, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 3, 14, 28], [1, 3, 14, 28]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]],
    memory_shapes = [[1, 3, 15, 28], [1, 3, 15, 28]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x28x28xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 3, 14, 28], [1, 3, 14, 28]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]],
    memory_shapes = [[1, 3, 15, 28], [1, 3, 15, 28]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]]
}>

// CHECK-LABEL: @UnrollConvertDMAWithInputOutputOverlappedExplicitShapesOffsets
func.func @UnrollConvertDMAWithInputOutputOverlappedExplicitShapesOffsets(%arg: memref<1x3x28x28xf32, @DDR>) -> memref<1x3x28x28xf16, @DDR> {
    %input_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x28x28xf32, @DDR>
    %output_ddr = VPURT.DeclareBuffer <DDR> <9408> -> memref<1x3x28x28xf16, @DDR>
    %cmx_input_buffer = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %cmx_output_buffer = VPURT.DeclareBuffer <CMX_NN> <9408> -> !OutputDistributed

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_ddr : memref<1x3x28x28xf32, @DDR>) outputs(%cmx_input_buffer : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.ConvertDMA <{port = 0 : i64}> inputs(%cmx_input_buffer : !InputDistributed) outputs(%cmx_output_buffer : !OutputDistributed) -> !OutputDistributed
    }

    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cmx_output_buffer : !OutputDistributed) outputs(%output_ddr : memref<1x3x28x28xf16, @DDR>) -> memref<1x3x28x28xf16, @DDR>
    }

    return %output_ddr: memref<1x3x28x28xf16, @DDR>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x15x28xf32, [@CMX_NN, 0]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x15x28xf32, [@CMX_NN, 1]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <9408> -> memref<1x3x15x28xf16, [@CMX_NN, 0]>
    //CHECK-DAG: [[ARG:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <9408> -> memref<1x3x15x28xf16, [@CMX_NN, 1]>

    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 0]>) -> memref<1x3x15x28xf32, [@CMX_NN, 0]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) outputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 1]>) -> memref<1x3x15x28xf32, [@CMX_NN, 1]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 0]>) outputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 0]>) -> memref<1x3x15x28xf16, [@CMX_NN, 0]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.ConvertDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf32, [@CMX_NN, 1]>) outputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 1]>) -> memref<1x3x15x28xf16, [@CMX_NN, 1]>
    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 0]>) outputs([[ARG:%.+]] : memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
    //CHECK:  [[RESULT:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[ARG:%.+]] : memref<1x3x15x28xf16, [@CMX_NN, 1]>) outputs([[ARG:%.+]]: memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>) -> memref<1x3x15x28xf16, {order = #NCHW, strides = [2352, 784, 28, 1]}, @DDR>
}
