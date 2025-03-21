//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --patch-populate-weight-table-with-shave --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX
!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<1x1x1x4096xf32, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputDistributedFloat16 = !VPUIP.DistributedBuffer<1x1x1x4096xf16, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputDistributedReshaped = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputDistributedPermuted = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!PopulateWeightTableDistributed = !VPUIP.DistributedBuffer<512x1x1x4xsi32, { order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
      memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>

!WTDistributed = !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
      compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
      memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

!WeightsDistributed = !VPUIP.DistributedBuffer<1024x4096x1x1x!qElemType, #NHWC, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
      compute_shapes = [[256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
      memory_shapes = [[256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

!ScaleDistributed = !VPUIP.DistributedBuffer<512x1x1x4xf16, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
      memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>

!ConvInputDistributed = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!ConvOutputDistributed = !VPUIP.DistributedBuffer<1x1024x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64,
      alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 256, 0, 0], [0, 512, 0, 0], [0, 768, 0, 0]],
      memory_shapes = [[1, 1024, 1, 1], [1, 1024, 1, 1], [1, 1024, 1, 1], [1, 1024, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

module @VPU.SW {
    func.func private @builtin_PopulateWeightTable(memref<*xsi32, @CMX_NN>, i64, i64)
            attributes {VPU.kernel_code = "populate_weight_table.cpp",
                        VPU.kernel_entry = "populate_weight_table", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @PatchSWKernelModeSegmented
func.func @PatchSWKernelModeSegmented(%arg0: memref<1x1x4096xf32, @DDR>, %scale: !ScaleDistributed)
           -> !ConvOutputDistributed {

    %0 = VPURT.DeclareBuffer <CMX_NN> <8192> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedFloat16
    %2 = VPURT.DeclareBuffer <CMX_NN> <24576> -> !WeightsDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <1073152> -> !WTDistributed
    %4 = VPURT.DeclareBuffer <CMX_NN> <1081344> -> !ConvOutputDistributed

    %cst_2 = const.Declare memref<1024x4096x1x1x!qElemType, #NHWC> = dense<1.0> : tensor<1024x4096x1x1xf16>,
              [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %token, %bodyResults:2 = async.execute ->
        (!async.value<!PopulateWeightTableDistributed>, !async.value<!PopulateWeightTableDistributed>)
        attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {

      %sub_view0 = VPUIP.SubView %3 [0, 0, 0, 0] [512, 1, 1, 4] {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
            !WTDistributed to !PopulateWeightTableDistributed

      %sub_view1 = VPUIP.SubView %3 [512, 0, 0, 0] [512, 1, 1, 4] {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
            !WTDistributed to !PopulateWeightTableDistributed

      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>, weightsPtrsPerClusterAttr = [0, 0, 0, 0]}
        @VPU.SW::@builtin_PopulateWeightTable inputs(%scale as %arg3: !ScaleDistributed)
        outputs(%sub_view0 as %arg5: memref<512x1x1x4xsi32, @CMX_NN>, %sub_view1 as %arg6: memref<512x1x1x4xsi32, @CMX_NN>)
            strides([[4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1]]) on tile 0 -> (
            !PopulateWeightTableDistributed, !PopulateWeightTableDistributed)
            {
                    VPUIP.SW.Kernel.run {attrs = [8192, 2048]}(%arg5) : memref<512x1x1x4xsi32, @CMX_NN>
                    VPUIP.SW.Kernel.run {attrs = [270336, 2048]}(%arg6) : memref<512x1x1x4xsi32, @CMX_NN>
      }

      async.yield %results#0, %results#1 :
          !PopulateWeightTableDistributed, !PopulateWeightTableDistributed
    }

     %token_3, %bodyResults_4 = async.execute -> !async.value<!InputDistributed>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64,
                                cycleBegin = 0 : i64, cycleCost = 2072 : i64, cycleEnd = 2072 : i64} {
      %18 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x4096xf32, @DDR>) -> memref<1x1x1x4096xf32, @DDR>
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<1x1x1x4096xf32, @DDR>)
            outputs(%0 : !InputDistributed)
            -> !InputDistributed
      async.yield %19 : !InputDistributed
    }

    %token_7, %bodyResults_8 = async.execute [%token_3] (%bodyResults_4 as %arg2: !async.value<!InputDistributed>)
            -> !async.value<!InputDistributedFloat16>
                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64,
                cycleBegin = 2072 : i64, cycleCost = 474 : i64, cycleEnd = 2546 : i64} {
      %18 = VPUIP.ConvertDMA {port = 0 : i64} inputs(%arg2 : !InputDistributed)
            outputs(%1 : !InputDistributedFloat16)
            -> !InputDistributedFloat16
      async.yield %18 : !InputDistributedFloat16
    }
    %token_9, %bodyResults_10 = async.execute -> !async.value<!WeightsDistributed>
                  attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0, 1], "async-deps-index" = 4 : i64,
                  cycleBegin = 2072 : i64, cycleCost = 458697 : i64, cycleEnd = 460769 : i64} {
      %18 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_2 : memref<1024x4096x1x1x!qElemType, #NHWC>) outputs(%2 : !WeightsDistributed)
            -> !WeightsDistributed
      async.yield %18 : !WeightsDistributed
    }
    %token_11, %bodyResults_12 = async.execute [%token, %token_7, %token_9] (%bodyResults_8 as %arg2: !async.value<!InputDistributedFloat16>,
        %bodyResults_10 as %arg3: !async.value<!WeightsDistributed>,
        %bodyResults#0 as %arg4: !async.value<!PopulateWeightTableDistributed>,
        %bodyResults#1 as %arg5: !async.value<!PopulateWeightTableDistributed>)
        -> !async.value<!ConvOutputDistributed>
          attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 460769 : i64, cycleCost = 1 : i64, cycleEnd = 460770 : i64} {
      %18 = VPUIP.GenericReshape inputs(%arg2 : !InputDistributedFloat16)
          -> !InputDistributedReshaped
      %19 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%18 : !InputDistributedReshaped)
          -> !InputDistributedPermuted
      %20 = VPUIP.DistributedCast inputs(%19 : !InputDistributedPermuted)
          -> !ConvInputDistributed
      %21 = VPUIP.ConcatView inputs(%arg4, %arg5 : !PopulateWeightTableDistributed, !PopulateWeightTableDistributed)
        outputs(%3 : !WTDistributed) -> !WTDistributed
      %22 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                kernel_strides = [1, 1], minimumHardwareExecutionCost = 4294967295 : i64, populateWeightTable = true, task_type = #VPUIP.nce_task_type<CONV>}
        input(%20 : !ConvInputDistributed)
        weights(%arg3 : !WeightsDistributed)
        weight_table(%21 : !WTDistributed)
        parent_input(%20 : !ConvInputDistributed)
        parent_output(%4 : !ConvOutputDistributed)
        outputs(%4 : !ConvOutputDistributed)
        -> !ConvOutputDistributed variants : {
              DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 255], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 256],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 2 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 767], outStart = [0, 0, 512],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 3 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1023], outStart = [0, 0, 768],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
      async.yield %22 : !ConvOutputDistributed
    }

    %result = async.await %bodyResults_12 : !async.value<!ConvOutputDistributed>

    return %result : !ConvOutputDistributed

    // CHECK:       [[IN_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1073152> -> !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[TOKEN:%.+]], [[BODY_RESULT:%.+]]:2 = async.execute
    // CHECK-SAME:       -> (!async.value<!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>>,
    // CHECK-SAME:      !async.value<!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>>)
    // CHECK-SAME:      attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {

    // CHECK:              [[SUBVIEW0:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [512, 1, 1, 4]
    // CHECK:              [[SUBVIEW1:%.+]] = VPUIP.SubView [[IN_BUF]] [512, 0, 0, 0] [512, 1, 1, 4]
    // CHECK:              [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>, weightsPtrsPerClusterAttr = [0, 0, 0, 0]}
    // CHECK-SAME:            @VPU.SW::@builtin_PopulateWeightTable
    // CHECK-SAME:            inputs({{[^:]+}} as {{[^:]+}}: !VPUIP.DistributedBuffer<512x1x1x4xf16, #NCHW, @CMX_NN
    // CHECK-SAME:            outputs([[SUBVIEW0]] as [[INNER_ARG0:[^:]+]]: memref<512x1x1x4xsi32, @CMX_NN>,
    // CHECK-SAME:            [[SUBVIEW1]] as [[INNER_ARG1:[^:]+]]: memref<512x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME{LITERAL}:            strides([[4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1]]) on tile 0 ->
    // CHECK-SAME:            (!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:              compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:              memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>,
    // CHECK-SAME:            !VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:              compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:              memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>){

    // CHECK:                VPUIP.SW.Kernel.run {attrs = [24576, 2048]}([[INNER_ARG0]]) : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK:                VPUIP.SW.Kernel.run {attrs = [1073152, 2048]}([[INNER_ARG1]]) : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK:              }
    // CHECK:              async.yield [[RESULTS]]#0, [[RESULTS]]#1
    // CHECK:       }
}

// -----

!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<1x1x1x4096xf32, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputDistributedFloat16 = !VPUIP.DistributedBuffer<1x1x1x4096xf16, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputDistributedReshaped = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputDistributedPermuted = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!PopulateWeightTableDistributed = !VPUIP.DistributedBuffer<512x1x1x4xsi32, { order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
      memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>

!PopulateWeightTableDistributedDDR = !VPUIP.DistributedBuffer<512x1x1x4xsi32, { order = #NCHW, strides = [4, 4, 4, 1]}, @DDR,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
      memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>

!WTDistributed = !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
      compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
      memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

!WeightsDistributed = !VPUIP.DistributedBuffer<1024x4096x1x1x!qElemType, #NHWC, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
      compute_shapes = [[256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
      memory_shapes = [[256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1], [256, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

!BiasDistributed = !VPUIP.DistributedBuffer<512x1x1x4xi4, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
      memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>

!ScaleDistributed = !VPUIP.DistributedBuffer<512x1x1x4xf16, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
      compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
      memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
      memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>

!ConvInputDistributed = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      memory_shapes = [[1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1], [1, 4096, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!ConvOutputDistributed = !VPUIP.DistributedBuffer<1x1024x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64,
      alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1]],
      compute_offsets = [[0, 0, 0, 0], [0, 256, 0, 0], [0, 512, 0, 0], [0, 768, 0, 0]],
      memory_shapes = [[1, 1024, 1, 1], [1, 1024, 1, 1], [1, 1024, 1, 1], [1, 1024, 1, 1]],
      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

module @VPU.SW {
    func.func private @builtin_PopulateWeightTable(memref<*xsi32, @CMX_NN>, i64, i64)
            attributes {VPU.kernel_code = "populate_weight_table.cpp",
                        VPU.kernel_entry = "populate_weight_table", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @PatchSWKernelClusteredWithDMASpill
func.func @PatchSWKernelClusteredWithDMASpill(%arg0: memref<1x1x4096xf32, @DDR>, %scale: !ScaleDistributed)
           -> !ConvOutputDistributed {

    %0 = VPURT.DeclareBuffer <CMX_NN> <8192> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedFloat16
    %2 = VPURT.DeclareBuffer <CMX_NN> <24576> -> !WeightsDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <1073152> -> !WTDistributed
    %4 = VPURT.DeclareBuffer <CMX_NN> <1081344> -> !ConvOutputDistributed

    %5 = VPURT.DeclareBuffer <CMX_NN> <3416064> -> !BiasDistributed
    %6 = VPURT.DeclareBuffer <DDR> <0> -> !PopulateWeightTableDistributedDDR
    %7 = VPURT.DeclareBuffer <DDR> <2048> -> !PopulateWeightTableDistributedDDR
    %8 = VPURT.DeclareBuffer <CMX_NN> <2000000> -> !PopulateWeightTableDistributed
    %9 = VPURT.DeclareBuffer <CMX_NN> <2002048> -> !PopulateWeightTableDistributed

    %bias = const.Declare memref<512x1x1x4xi4> = dense<1.0> : memref<512x1x1x4xf16>,
            [#const.CastElemType<i4>]
    %cst_2 = const.Declare memref<1024x4096x1x1x!qElemType, #NHWC> = dense<1.0> : tensor<1024x4096x1x1xf16>,
              [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %token_13, %bodyResults_14 = async.execute -> !async.value<!BiasDistributed>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 1637 : i64, cycleCost = 329408 : i64, cycleEnd = 331045 : i64} {
      %21 = VPUIP.NNDMA {port = 0 : i64} inputs(%bias : memref<512x1x1x4xi4>) outputs(%5 : !BiasDistributed)
        -> !BiasDistributed
      async.yield %21 : !BiasDistributed
    }

    %token, %bodyResults:2 = async.execute (%bodyResults_14 as %arg2: !async.value<!BiasDistributed>) ->
        (!async.value<!PopulateWeightTableDistributed>, !async.value<!PopulateWeightTableDistributed>)
        attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {

      %sub_view0 = VPUIP.SubView %3 [0, 0, 0, 0] [512, 1, 1, 4] {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
            !WTDistributed to !PopulateWeightTableDistributed

      %sub_view1 = VPUIP.SubView %3 [512, 0, 0, 0] [512, 1, 1, 4] {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
            !WTDistributed to !PopulateWeightTableDistributed

      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>, weightsPtrsPerClusterAttr = [0, 0, 0, 0]}
        @VPU.SW::@builtin_PopulateWeightTable inputs(%scale as %arg3: !ScaleDistributed, %arg2 as %arg4: !BiasDistributed)
        outputs(%sub_view0 as %arg5: memref<512x1x1x4xsi32, @CMX_NN>, %sub_view1 as %arg6: memref<512x1x1x4xsi32, @CMX_NN>)
            strides([[4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1]]) on tile 0 -> (
            !PopulateWeightTableDistributed, !PopulateWeightTableDistributed)
            {
                    VPUIP.SW.Kernel.run {attrs = [8192, 2048]}(%arg5) : memref<512x1x1x4xsi32, @CMX_NN>
                    VPUIP.SW.Kernel.run {attrs = [270336, 2048]}(%arg6) : memref<512x1x1x4xsi32, @CMX_NN>
      }

      async.yield %results#0, %results#1 :
          !PopulateWeightTableDistributed, !PopulateWeightTableDistributed
    }

    %token_15, %bodyResults_16:2 = async.execute [%token]
        (%bodyResults#0 as %arg2: !async.value<!PopulateWeightTableDistributed>, %bodyResults#1 as %arg3: !async.value<!PopulateWeightTableDistributed>)
        -> (!async.value<!PopulateWeightTableDistributedDDR>, !async.value<!PopulateWeightTableDistributedDDR>)
        attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64,
          cycleBegin = 1935 : i64, cycleCost = 115893 : i64, cycleEnd = 117828 : i64} {
      %18 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !PopulateWeightTableDistributed) outputs(%6 : !PopulateWeightTableDistributedDDR)
      -> !PopulateWeightTableDistributedDDR
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg3 : !PopulateWeightTableDistributed) outputs(%7 : !PopulateWeightTableDistributedDDR)
      -> !PopulateWeightTableDistributedDDR
      async.yield %18, %19 : !PopulateWeightTableDistributedDDR, !PopulateWeightTableDistributedDDR
    }

    %token_17, %bodyResults_18:2 = async.execute [%token_15]
        (%bodyResults_16#0 as %arg2: !async.value<!PopulateWeightTableDistributedDDR>, %bodyResults_16#1 as %arg3: !async.value<!PopulateWeightTableDistributedDDR>)
        -> (!async.value<!PopulateWeightTableDistributed>, !async.value<!PopulateWeightTableDistributed>)
        attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64,
          cycleBegin = 1935 : i64, cycleCost = 115893 : i64, cycleEnd = 117828 : i64} {
      %18 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !PopulateWeightTableDistributedDDR) outputs(%8 : !PopulateWeightTableDistributed)
      -> !PopulateWeightTableDistributed
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg3 : !PopulateWeightTableDistributedDDR) outputs(%9 : !PopulateWeightTableDistributed)
      -> !PopulateWeightTableDistributed
      async.yield %18, %19 : !PopulateWeightTableDistributed, !PopulateWeightTableDistributed
    }


    %token_3, %bodyResults_4 = async.execute -> !async.value<!InputDistributed>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64,
                                cycleBegin = 0 : i64, cycleCost = 2072 : i64, cycleEnd = 2072 : i64} {
      %18 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x4096xf32, @DDR>) -> memref<1x1x1x4096xf32, @DDR>
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<1x1x1x4096xf32, @DDR>)
            outputs(%0 : !InputDistributed)
            -> !InputDistributed
      async.yield %19 : !InputDistributed
    }

    %token_7, %bodyResults_8 = async.execute [%token_3] (%bodyResults_4 as %arg2: !async.value<!InputDistributed>)
            -> !async.value<!InputDistributedFloat16>
                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64,
                cycleBegin = 2072 : i64, cycleCost = 474 : i64, cycleEnd = 2546 : i64} {
      %18 = VPUIP.ConvertDMA {port = 0 : i64} inputs(%arg2 : !InputDistributed)
            outputs(%1 : !InputDistributedFloat16)
            -> !InputDistributedFloat16
      async.yield %18 : !InputDistributedFloat16
    }
    %token_9, %bodyResults_10 = async.execute -> !async.value<!WeightsDistributed>
                  attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0, 1], "async-deps-index" = 4 : i64,
                  cycleBegin = 2072 : i64, cycleCost = 458697 : i64, cycleEnd = 460769 : i64} {
      %18 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_2 : memref<1024x4096x1x1x!qElemType, #NHWC>) outputs(%2 : !WeightsDistributed)
            -> !WeightsDistributed
      async.yield %18 : !WeightsDistributed
    }
    %token_11, %bodyResults_12 = async.execute [%token, %token_7, %token_9] (%bodyResults_8 as %arg2: !async.value<!InputDistributedFloat16>,
        %bodyResults_10 as %arg3: !async.value<!WeightsDistributed>,
        %bodyResults_18#0 as %arg4: !async.value<!PopulateWeightTableDistributed>,
        %bodyResults_18#1 as %arg5: !async.value<!PopulateWeightTableDistributed>)
        -> !async.value<!ConvOutputDistributed>
          attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 460769 : i64, cycleCost = 1 : i64, cycleEnd = 460770 : i64} {
      %18 = VPUIP.GenericReshape inputs(%arg2 : !InputDistributedFloat16)
          -> !InputDistributedReshaped
      %19 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%18 : !InputDistributedReshaped)
          -> !InputDistributedPermuted
      %20 = VPUIP.DistributedCast inputs(%19 : !InputDistributedPermuted)
          -> !ConvInputDistributed
      %21 = VPUIP.ConcatView inputs(%arg4, %arg5 : !PopulateWeightTableDistributed, !PopulateWeightTableDistributed)
        outputs(%3 : !WTDistributed) -> !WTDistributed
      %22 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                kernel_strides = [1, 1], minimumHardwareExecutionCost = 4294967295 : i64, populateWeightTable = true, task_type = #VPUIP.nce_task_type<CONV>}
        input(%20 : !ConvInputDistributed)
        weights(%arg3 : !WeightsDistributed)
        weight_table(%21 : !WTDistributed)
        parent_input(%20 : !ConvInputDistributed)
        parent_output(%4 : !ConvOutputDistributed)
        outputs(%4 : !ConvOutputDistributed)
        -> !ConvOutputDistributed variants : {
              DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 255], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 256],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 2 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 767], outStart = [0, 0, 512],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 3 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1023], outStart = [0, 0, 768],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
      async.yield %22 : !ConvOutputDistributed
    }

    %result = async.await %bodyResults_12 : !async.value<!ConvOutputDistributed>

    return %result : !ConvOutputDistributed

    // CHECK:       [[IN_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1073152> -> !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[TOKEN:%.+]], [[BODY_RESULT:%.+]]:2 = async.execute
    // CHECK-SAME:       {{[^:]+}} as {{[^:]+}}: !async.value<!VPUIP.DistributedBuffer<512x1x1x4xi4, #NCHW, @CMX_NN,
    // CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>>)
    // CHECK-SAME:       -> (!async.value<!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>>,
    // CHECK-SAME:      !async.value<!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>>)
    // CHECK-SAME:      attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {

    // CHECK:              [[SUBVIEW0:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [512, 1, 1, 4]
    // CHECK:              [[SUBVIEW1:%.+]] = VPUIP.SubView [[IN_BUF]] [512, 0, 0, 0] [512, 1, 1, 4]
    // CHECK:              [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>, weightsPtrsPerClusterAttr = [0, 0, 0, 0]}
    // CHECK-SAME:            @VPU.SW::@builtin_PopulateWeightTable
    // CHECK-SAME:            inputs({{[^:]+}} as {{[^:]+}}: !VPUIP.DistributedBuffer<512x1x1x4xf16, #NCHW, @CMX_NN
    // CHECK-SAME:            , {{[^:]+}} as {{[^:]+}}: !VPUIP.DistributedBuffer<512x1x1x4xi4, #NCHW, @CMX_NN,
    // CHECK-SAME:            outputs([[SUBVIEW0]] as [[INNER_ARG0:[^:]+]]: memref<512x1x1x4xsi32, @CMX_NN>,
    // CHECK-SAME:            [[SUBVIEW1]] as [[INNER_ARG1:[^:]+]]: memref<512x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME{LITERAL}:            strides([[4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1]]) on tile 0 ->
    // CHECK-SAME:            (!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:              compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:              memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>,
    // CHECK-SAME:            !VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:              compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:              memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>){

    // CHECK:                VPUIP.SW.Kernel.run {attrs = [24576, 2048]}([[INNER_ARG0]]) : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK:                VPUIP.SW.Kernel.run {attrs = [1073152, 2048]}([[INNER_ARG1]]) : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK:              }
    // CHECK:              async.yield [[RESULTS]]#0, [[RESULTS]]#1
    // CHECK:       }
}
