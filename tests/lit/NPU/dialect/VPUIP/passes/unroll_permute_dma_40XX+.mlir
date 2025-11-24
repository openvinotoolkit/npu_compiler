//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-permute-dma  %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1x960x4xf16, #NHWC, @CMX_NN,
    {
      mode = "DUPLICATED",
      num_clusters = 4 : i64,
      uniform_distributed_segments,                
      compute_shapes = [[1, 1, 960, 4], [1, 1, 960, 4], [1, 1, 960, 4], [1, 1, 960, 4]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 1, 960, 4], [1, 1, 960, 4], [1, 1, 960, 4], [1, 1, 960, 4]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }
>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x960x4x1xf16, #NHWC, @CMX_NN,
    {
      mode = "DUPLICATED",
      num_clusters = 4 : i64,
      uniform_distributed_segments,                
      compute_shapes = [[1, 960, 4, 1], [1, 960, 4, 1], [1, 960, 4, 1], [1, 960, 4, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 960, 4, 1], [1, 960, 4, 1], [1, 960, 4, 1], [1, 960, 4, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }
>

// CHECK-LABEL: @PermuteDMAForShapeWithDuplicatedOutputWithExplicitShapesAndOffsets
func.func @PermuteDMAForShapeWithDuplicatedOutputWithExplicitShapesAndOffsets() -> !OutputDistributed {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer <CMX_NN> <16384> -> !OutputDistributed

    VPURT.Task updates(%BAR_0 : !VPURT.Barrier) {
      %80 = VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64} 
              inputs(%input : !InputDistributed)
              outputs(%output : !OutputDistributed)
              -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <16384> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <16384> -> [[OUTPUT_TYPE_0:.+VPUIP.DistributedBuffer.+strides =.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <17344> -> [[OUTPUT_TYPE_1:.+VPUIP.DistributedBuffer.+strides =.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NHWC
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NHWC
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    64x64x3x3xf16, #NCHW, @CMX_NN,
    {
      mode = "DUPLICATED", 
      num_clusters = 4 : i64, 
      uniform_distributed_segments, 
      compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]], 
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
      memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]], 
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }
>

!OutputDistributed = !VPUIP.DistributedBuffer<
    64x64x3x3xf16, #NHWC, @CMX_NN,
    {
      mode = "DUPLICATED", 
      num_clusters = 4 : i64, 
      uniform_distributed_segments, 
      compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]], 
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
      memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]], 
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }
>

// CHECK-LABEL: @PermuteDMAForLayoutWithDuplicatedOutputWithExplicitShapesAndOffsets
func.func @PermuteDMAForLayoutWithDuplicatedOutputWithExplicitShapesAndOffsets() -> !OutputDistributed {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer <CMX_NN> <689152> -> !OutputDistributed

    VPURT.Task updates(%BAR_0 : !VPURT.Barrier) {
      %1 = VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64} 
              inputs(%input : !InputDistributed)
              outputs(%output : !OutputDistributed)
              -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <689152> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <689152> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <726016> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:3, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128,
                                        8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128,
                                       8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128}>

// CHECK: !qElemType = !quant.uniform<u8:f16:3, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128,
// CHECK-SAME: 8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128}>
// CHECK: !qElemType1 = !quant.uniform<u8:f16:1, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128,
// CHECK-SAME: 8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128}>

// CHECK-LABEL: @PermuteDMAWithTileOverQuantAxisAndTransposeOfQuantAxis
func.func @PermuteDMAWithTileOverQuantAxisAndTransposeOfQuantAxis() -> memref<1x32x1x8x!qElemType, #NHWC, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x32x!qElemType1, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8x!qElemType, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x1x32x!qElemType1, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x32x1x8x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x8x!qElemType, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x32x1x8x!qElemType, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+!qElemType1.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> [[INPUT_TYPE_1:.+!qElemType1.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+!qElemType.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+!qElemType.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4112> -> [[OUTPUT_TYPE_1:.+!qElemType.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NWHC
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NWHC
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAWithNoSplitForNonByteAlignedElementType
func.func @PermuteDMAWithNoSplitForNonByteAlignedElementType() -> memref<4x3x3x3xui4, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4x3x3x3xui4, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<4x3x3x3xui4, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
                inputs(%input : memref<4x3x3x3xui4, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<4x3x3x3xui4, [@CMX_NN, 0]>) -> memref<4x3x3x3xui4, [@CMX_NN, 0]>
    }

    return %output: memref<4x3x3x3xui4, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+ui4.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+ui4.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+ui4.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:0, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128}>

// CHECK: !qElemType = !quant.uniform<u8:f16:0, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128,9.8173718388174099E-5:128}>
// CHECK: !qElemType1 = !quant.uniform<u8:f16:0, {8.1867772136248799E-5:128,7.2379066955809501E-5:128}>
// CHECK: !qElemType2 = !quant.uniform<u8:f16:0, {8.9968320931874069E-5:128,9.8173718388174099E-5:128}>

// CHECK-LABEL: @PermuteDMAWithTileOverQuantAxis
func.func @PermuteDMAWithTileOverQuantAxis() -> memref<4x3x3x3x!qElemType, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4x3x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
                inputs(%input : memref<4x3x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>) -> memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>
    }

    return %output: memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+!qElemType1.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54> -> [[INPUT_TYPE_1:.+!qElemType2.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+!qElemType.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+!qElemType1.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4150> -> [[OUTPUT_TYPE_1:.+!qElemType2.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:1, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128}>

// CHECK: !qElemType = !quant.uniform<u8:f16:1, {8.1867772136248799E-5:128,7.2379066955809501E-5:128,8.9968320931874069E-5:128}>

// CHECK-LABEL: @PermuteDMAWithTileOverNonQuantAxis
func.func @PermuteDMAWithTileOverNonQuantAxis() -> memref<4x3x3x3x!qElemType, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4x3x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
                inputs(%input : memref<4x3x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>) -> memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>
    }

    return %output: memref<4x3x3x3x!qElemType, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+!qElemType.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54> -> [[INPUT_TYPE_1:.+!qElemType.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+!qElemType.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+!qElemType.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4150> -> [[OUTPUT_TYPE_1:.+!qElemType.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAWithNHWCToNCHW
func.func @PermuteDMAWithNHWCToNCHW() -> memref<1x3x3x3xf16, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x3x3x3xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x3x3x3xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x3x3x3xf16, [@CMX_NN, 0]>) -> memref<1x3x3x3xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x3x3x3xf16, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAWithNHWCToNCHW
func.func @PermuteDMAWithNHWCToNCHW() -> memref<1x8x16x16xf16, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4352> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAFromTranspose
func.func @PermuteDMAFromTranspose() -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4128> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NWHC
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NWHC
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAWithLargePlaneNumber
func.func @PermuteDMAWithLargePlaneNumber() -> memref<1x8x32x16xf16, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x32x16xf16, [@CMX_NN, 0]>) -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x8x32x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
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

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_6:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_7:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_8:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_9:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_10:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_11:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <3456> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <DDR> <12672> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <DDR> <4032> -> [[INPUT_TYPE_2:.+]]
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <DDR> <13248> -> [[INPUT_TYPE_3:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <5440> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <5456> -> [[OUTPUT_TYPE_1:.+]]
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <5440> -> [[OUTPUT_TYPE_2:.+]]
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <5456> -> [[OUTPUT_TYPE_3:.+]]


    //CHECK:   VPURT.Task waits([[BAR_1]] : !VPURT.Barrier) updates([[BAR_2]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_2]] : !VPURT.Barrier) updates([[BAR_3]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_3]] : !VPURT.Barrier) updates([[BAR_4]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_4]] : !VPURT.Barrier) updates([[BAR_5]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_5]] : !VPURT.Barrier) updates([[BAR_6]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_6]] : !VPURT.Barrier) updates([[BAR_7]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_7]] : !VPURT.Barrier) updates([[BAR_8]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }


    //CHECK:    VPURT.Task waits([[BAR_8]] : !VPURT.Barrier) updates([[BAR_9]] : !VPURT.Barrier)
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_8]] : !VPURT.Barrier) updates([[BAR_9]] : !VPURT.Barrier)
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_8]] : !VPURT.Barrier) updates([[BAR_9]] : !VPURT.Barrier)
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_2]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_2]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_2]]
    //CHECK:            outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_8]] : !VPURT.Barrier) updates([[BAR_9]] : !VPURT.Barrier)
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_3]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_3]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_3]]
    //CHECK:            outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_9]] : !VPURT.Barrier) updates([[BAR_10]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA inputs(%cst_0 : memref<1x1x1x16xui8>) outputs(%33 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    }

    // nce task
    //CHECK:    VPURT.Task waits([[BAR_10]] : !VPURT.Barrier) updates([[BAR_11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_10]] : !VPURT.Barrier) updates([[BAR_11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:    }

    // copy back
    //CHECK:    VPURT.Task waits([[BAR_11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA {port = 1 : i64}
    //CHECK:    }
    //CHECK:    return %0 : memref<1x3x24x24xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAWithNCHWToNHWCForNetworkOutput
func.func @PermuteDMAWithNCHWToNHWCForNetworkOutput() -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x14x7xf16, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0 : !VPURT.Barrier)  {
      VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}
            inputs(%input : memref<1x32x14x7xf16, @DDR>)
            outputs(%output : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
   }

    return %output: memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <3136> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6304> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteDMAWithNCHWToNHWCForNetworkInput
func.func @PermuteDMAWithNCHWToNHWCForNetworkInput() -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x32x14x7xf16, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0 : !VPURT.Barrier)  {
      VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}
            inputs(%input : memref<1x32x14x7xf16, @DDR>)
            outputs(%output : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
   }

    return %output: memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> <3136> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6272> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6304> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x14x7xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithNCHWToNHWCForNetworkOutput
func.func @ClusterPermuteDMAWithNCHWToNHWCForNetworkOutput() -> !OutputDistributed {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %BAR_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x14x7xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%BAR_0 : !VPURT.Barrier) updates(%BAR_1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #NHWC}
              inputs(%0 : memref<1x32x14x7xf16, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <3136> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <98> -> [[INPUT_TYPE_2:.+]]
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <3234> -> [[INPUT_TYPE_3:.+]]

    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[RETURN_TYPE_0:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> [[OUTPUT_TYPE_1:.+]]
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_2:.+]]
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32> -> [[OUTPUT_TYPE_3:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_2]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_2]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_2]]
    //CHECK:            outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_3]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_3]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_3]]
    //CHECK:            outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x14x7xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithNCHWToNHWCForNetworkInput
func.func @ClusterPermuteDMAWithNCHWToNHWCForNetworkInput() -> !OutputDistributed {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %BAR_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x32x14x7xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%BAR_0 : !VPURT.Barrier) updates(%BAR_1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #NHWC}
              inputs(%0 : memref<1x32x14x7xf16, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> <3136> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <NetworkInput> <98> -> [[INPUT_TYPE_2:.+]]
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <NetworkInput> <3234> -> [[INPUT_TYPE_3:.+]]

    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[RETURN_TYPE_0:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> [[OUTPUT_TYPE_1:.+]]
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_2:.+]]
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32> -> [[OUTPUT_TYPE_3:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_2]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_2]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_2]]
    //CHECK:            outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_3]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_3]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_3]]
    //CHECK:            outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
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
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %BAR_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <2000> -> !OutputDistributed

    VPURT.Task waits(%BAR_0 : !VPURT.Barrier) updates(%BAR_1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : !InputDistributed)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed
 
    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <2000> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2032> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
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

// CHECK-LABEL: @PermuteDMAWithNCHWToNHWC2D
func.func @PermuteDMAWithNCHWToNHWC2D() -> !OutputDistributed {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x72x2x1xf16, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%BAR_0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %18 = VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64} inputs(%input : memref<1x72x2x1xf16, @DDR>) outputs(%output : !OutputDistributed) -> !OutputDistributed
    }
    return %output: !VPUIP.DistributedBuffer<1x72x2x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> <144> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <NetworkInput> <2> -> [[INPUT_TYPE_2:.+]]
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <NetworkInput> <146> -> [[INPUT_TYPE_3:.+]]

    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[RETURN_TYPE_0:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> [[OUTPUT_TYPE_1:.+]]
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_2:.+]]
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <72> -> [[OUTPUT_TYPE_3:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_2]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_2]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_2]]
    //CHECK:            outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_3]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_3]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NCHW
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_3]]
    //CHECK:            outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniformPermuteDMAPlaneSizeRequiresTwoDMAs
func.func @UniformPermuteDMAPlaneSizeRequiresTwoDMAs() -> memref<1x16x1x261xf16, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x261xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <8352> -> memref<1x16x1x261xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x16x1x261xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x1x261xf16, [@CMX_NN, 0]>) -> memref<1x16x1x261xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x16x1x261xf16, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <16> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8352> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8352> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12528> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniformPermuteDMAPlaneSizeDoesNotRequireFourDMAs
func.func @UniformPermuteDMAPlaneSizeDoesNotRequireFourDMAs() -> memref<1x16x1x520xf16, [@CMX_NN, 0]> {
    %BAR_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x520xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <16640> -> memref<1x16x1x520xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%BAR_0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x16x1x520xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x1x520xf16, [@CMX_NN, 0]>) -> memref<1x16x1x520xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x16x1x520xf16, [@CMX_NN, 0]>

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8320> -> [[INPUT_TYPE_1:.+]]
    //CHECK:    [[RETURN_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <16640> -> [[RETURN_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <16640> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17160> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_0]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_0]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 0
    //CHECK:            inputs([[INPUT_BUFFER_0]]
    //CHECK:            outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:           inputType = [[INPUT_TYPE_1]]
    //CHECK-SAME:           outputType = [[OUTPUT_TYPE_1]]
    //CHECK-SAME:           mappingOrder = #NCHW
    //CHECK-SAME:           loopOrder = #NHWC
    //CHECK-SAME:       port = 1
    //CHECK:            inputs([[INPUT_BUFFER_1]]
    //CHECK:            outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER_0]] : [[RETURN_TYPE_0]]
}
