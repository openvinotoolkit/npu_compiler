//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>

func.func @NotFuseCMXCopyToTheFrontOfTillingCopyDueToCMXSizeLimitation() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.Copy inputs(%0: !InputDistributedType) outputs(%1: !SpilledOutput_DDR) -> !SpilledOutput_DDR

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs([[BUF_1]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_DynamicTile(memref<*xsi32>, memref<*xsi32>) attributes {VPU.kernel_code = "dynamic_tile.cpp", VPU.kernel_entry = "dynamic_tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
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
