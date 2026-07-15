//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!InputStub_CMX = memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>

func.func @NotFuseCMXCopyToTheFrontOfTillingCopyDueToCMXSizeLimitation() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.Copy inputs(%0: !InputDistributedType) outputs(%1: !SpilledOutput_DDR) -> !SpilledOutput_DDR

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
  // CHECK:   [[BUF_1:%.+]] = memref.alloc() : memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs([[BUF_1]] : memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_DynamicTile(memref<*xsi32>, memref<*xsi32>) attributes {VPU.kernel_code = "dynamic_tile.cpp", VPU.kernel_entry = "dynamic_tile"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @NotEraseCMX2CMXCopyForDynamicTile
// CHECK-SAME:      ([[INPUT_0:%.+]]: memref<1x100xsi32, [@CMX_NN, 0]>, [[INPUT_1:%.+]]: memref<2xsi32, [@CMX_NN, 0]>, [[INPUT_2:%.+]]: memref<2xsi32, [@CMX_NN, 0]>)
func.func @NotEraseCMX2CMXCopyForDynamicTile(%arg0 : memref<1x100xsi32, [@CMX_NN, 0]>, %arg1 : memref<2xsi32, [@CMX_NN, 0]>, %arg2 : memref<2xsi32, [@CMX_NN, 0]>) -> (memref<1x100xsi32, [@CMX_NN, 0]>) {
  %alloc_0 = memref.alloc() : memref<1x100xsi32, [@CMX_NN, 0]>
  %0 = VPUIP.Copy inputs(%arg0 : memref<1x100xsi32, [@CMX_NN, 0]>) outputs(%alloc_0 : memref<1x100xsi32, [@CMX_NN, 0]>) -> memref<1x100xsi32, [@CMX_NN, 0]>
  %alloc_1 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
  %1 = VPUIP.Copy inputs(%arg1 : memref<2xsi32, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>

  %alloc_2 = memref.alloc() : memref<1x100xsi32, [@CMX_NN, 0]>
  %alloc_3 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
  %results, %dynamicOutputShapes = VPUIP.SW.Kernel {
    dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_DynamicTile
    inputs(%0 as %arg3: memref<1x100xsi32, [@CMX_NN, 0]>, %1 as %arg4: memref<2xsi32, [@CMX_NN, 0]>)
    dynamicInputShapes(%arg2 : memref<2xsi32, [@CMX_NN, 0]>)
    outputs(%alloc_2 as %arg5: memref<1x100xsi32, [@CMX_NN, 0]>)
    dynamicOutputShapes(%alloc_3 : memref<2xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>) {
      VPUIP.SW.Kernel.run {attrs = [2, [1, 1]]}(%arg3, %arg4, %arg5) : memref<1x100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x100xsi32, [@CMX_NN, 0]>
  }

  return %results : memref<1x100xsi32, [@CMX_NN, 0]>

  // CHECK:      [[ALLOC_0:%.+]] = memref.alloc() : memref<1x100xsi32, [@CMX_NN, 0]>
  // CHECK:      [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x100xsi32, [@CMX_NN, 0]>) outputs([[ALLOC_0]] : memref<1x100xsi32, [@CMX_NN, 0]>) -> memref<1x100xsi32, [@CMX_NN, 0]>
  // CHECK:      [[ALLOC_1:%.+]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
  // CHECK:      [[COPY_1:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<2xsi32, [@CMX_NN, 0]>) outputs([[ALLOC_1]] : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
  // CHECK:      [[ALLOC_2:%.+]] = memref.alloc() : memref<1x100xsi32, [@CMX_NN, 0]>
  // CHECK:      [[ALLOC_3:%.+]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
  // CHECK:      [[RESULTS:%.+]], [[DYNAMIC_OUTPUT_SHAPES:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_DynamicTile
  // CHECK:          inputs([[COPY_0]] as {{[^:]+}}: memref<1x100xsi32, [@CMX_NN, 0]>, [[COPY_1]] as {{[^:]+}}: memref<2xsi32, [@CMX_NN, 0]>)
  // CHECK:          dynamicInputShapes([[INPUT_2]] : memref<2xsi32, [@CMX_NN, 0]>)
  // CHECK:          outputs([[ALLOC_2]] as {{[^:]+}}: memref<1x100xsi32, [@CMX_NN, 0]>)
  // CHECK:          dynamicOutputShapes([[ALLOC_3]] : memref<2xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>){
  // CHECK:              VPUIP.SW.Kernel.run {attrs = [2, [1, 1]]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x100xsi32, [@CMX_NN, 0]>

  // CHECK:      return [[RESULTS]] : memref<1x100xsi32, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.014214559629851697>

// CHECK-LABEL: @SkipCopySeqDistributedCMXSubViewIntoConcatView
// CHECK-SAME:      [[SPLIT_SRC:%.+]]: !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
// CHECK-SAME:      [[CST:%.+]]: memref<1x144x20x20x!qElemType, {order = #NHWC}>
func.func @SkipCopySeqDistributedCMXSubViewIntoConcatView(
        %split_src: !VPUIP.DistributedBuffer<
            1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
        %cst: memref<1x144x20x20x!qElemType, {order = #NHWC}>)
        -> !VPUIP.DistributedBuffer<
            1x288x20x20x!qElemType, #NHWC, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

    // parentCopyOp: CMX(Distributed) -> DDR intermediate buffer
    %ddr_alloc = memref.alloc() : memref<1x144x20x20x!qElemType, {order = #NHWC}>
    %parent_copy = VPUIP.Copy
        inputs(%split_src : !VPUIP.DistributedBuffer<
            1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%ddr_alloc : memref<1x144x20x20x!qElemType, {order = #NHWC}>)
        -> memref<1x144x20x20x!qElemType, {order = #NHWC}>

    // CMX ConcatView output buffer (SEGMENTED)
    %cmx_concat_buf = VPURT.AllocDistributed ->
        !VPUIP.DistributedBuffer<1x288x20x20x!qElemType, #NHWC, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CMX SubView [0,0,0,0] -- copyOp writes DDR -> this distributed CMX SubView -> ConcatView
    %cmx_subview0 = VPUIP.SubView %cmx_concat_buf [0, 0, 0, 0] [1, 144, 20, 20] :
        !VPUIP.DistributedBuffer<1x288x20x20x!qElemType, #NHWC, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        to !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // copyOp (DDR -> CMX SubView, distributed): CopyOpSequence candidate with parent_copy
    // guard must block fusion: fusing would use split_src (CMX SubView) as source directly
    %copy0 = VPUIP.Copy
        inputs(%parent_copy : memref<1x144x20x20x!qElemType, {order = #NHWC}>)
        outputs(%cmx_subview0 : !VPUIP.DistributedBuffer<
            1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CMX SubView [0,144,0,0] -- const -> CMX SubView -> ConcatView
    %cmx_subview1 = VPUIP.SubView %cmx_concat_buf [0, 144, 0, 0] [1, 144, 20, 20] :
        !VPUIP.DistributedBuffer<1x288x20x20x!qElemType, #NHWC, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        to !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %copy1 = VPUIP.Copy
        inputs(%cst : memref<1x144x20x20x!qElemType, {order = #NHWC}>)
        outputs(%cmx_subview1 : !VPUIP.DistributedBuffer<
            1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
            {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %concat = VPUIP.ConcatView
        inputs(%copy0, %copy1 :
            !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
                {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
                {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
            !VPUIP.DistributedBuffer<1x144x20x20x!qElemType,
                {order = #NHWC, strides = [115200, 1, 5760, 288]}, @CMX_NN,
                {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%cmx_concat_buf : !VPUIP.DistributedBuffer<
            1x288x20x20x!qElemType, #NHWC, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x288x20x20x!qElemType, #NHWC, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    return %concat : !VPUIP.DistributedBuffer<
        1x288x20x20x!qElemType, #NHWC, @CMX_NN,
        {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // DDR intermediate buffer must be preserved (not fused away)
    // CHECK:      [[DDR_ALLOC:%.+]] = memref.alloc() : memref<1x144x20x20x!qElemType, {order = #NHWC}>
    // CHECK:      [[PARENT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SPLIT_SRC]]
    // CHECK-SAME:     outputs([[DDR_ALLOC]]
    // CHECK:      [[CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[CMX_SV0:%.+]] = VPUIP.SubView [[CMX_BUF]] [0, 0, 0, 0] [1, 144, 20, 20]
    // CHECK:      [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[PARENT_COPY]]
    // CHECK-SAME:     outputs([[CMX_SV0]]
    // CHECK:      [[CMX_SV1:%.+]] = VPUIP.SubView [[CMX_BUF]] [0, 144, 0, 0] [1, 144, 20, 20]
    // CHECK:      [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]]
    // CHECK-SAME:     outputs([[CMX_SV1]]
    // CHECK:      [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY0]], [[COPY1]]
    // CHECK-SAME:     outputs([[CMX_BUF]]
    // CHECK:      return [[CONCAT]]
}
