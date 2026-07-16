//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-to-dma --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: func.func @ConvertMemPermute
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x12x12xf16, @DDR>)
func.func @ConvertMemPermute(%arg0: memref<1x16x12x12xf16, @DDR>)
        -> memref<1x16x12x12xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x12x12xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x12x12xf16, @DDR>) outputs(%0 : memref<1x16x12x12xf16, [@CMX_NN, 0]>) -> memref<1x16x12x12xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x12x12xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg2, %arg3) : memref<1x16x12x12xf16, [@CMX_NN, 0]>, memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x12x12xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x12x12xf16, {order = #NHWC}, @DDR>) -> memref<1x16x12x12xf16, {order = #NHWC}, @DDR>
    return %4: memref<1x16x12x12xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x12x12xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x12x12xf16, @DDR>) outputs([[VAR0]] : memref<1x16x12x12xf16, [@CMX_NN, 0]>) -> memref<1x16x12x12xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHWC}> inputs([[VAR1]] : memref<1x16x12x12xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x16x12x12xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR5:%.+]]  = VPUIP.Copy inputs([[VAR3]] : memref<1x16x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x12x12xf16, {order = #NHWC}, @DDR>) -> memref<1x16x12x12xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR5:%.+]] : memref<1x16x12x12xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSWDepthToSpaceToDMA_BLOCKS_FIRST
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @ConvertSWDepthToSpaceToDMA_BLOCKS_FIRST(%arg0: memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[DepthToSpaceDMAOUT:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}>
    //CHECK:            inputs([[ARG_0]] : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSWDepthToSpaceToDMA_BLOCKS_FIRST_LARGE_HEIGHT
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @ConvertSWDepthToSpaceToDMA_BLOCKS_FIRST_LARGE_HEIGHT(%arg0: memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[DepthToSpaceDMAOUT:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}>
    //CHECK:            inputs([[ARG_0]] : memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSWDepthToSpaceToDMA_DEPTH_FIRST
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @ConvertSWDepthToSpaceToDMA_DEPTH_FIRST(%arg0: memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 1]}(%arg1, %arg2) : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[DepthToSpaceDMAOUT:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}>
    //CHECK:            inputs([[ARG_0]] : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSWDepthToSpaceToDMA_DEPTH_FIRST_LARGE_HEIGHT
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @ConvertSWDepthToSpaceToDMA_DEPTH_FIRST_LARGE_HEIGHT(%arg0: memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 1]}(%arg1, %arg2) : memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[DepthToSpaceDMAOUT:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}>
    //CHECK:            inputs([[ARG_0]] : memref<1x8x800x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x1600x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotConvertSWDepthToSpaceToDMAIfNotBeneficial
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x128x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @NotConvertSWDepthToSpaceToDMAIfNotBeneficial(%arg0: memref<1x128x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x128x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg1, %arg2) : memref<1x128x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:        [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:        [[D2S:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
    // CHECK-SAME:           inputs([[ARG_0]] as [[ARG_1:%[^:]+]]: memref<1x128x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[OUTBUFFER]] as [[ARG_2:%[^:]+]]: memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>{
    // CHECK:            VPUIP.SW.Kernel.run {attrs = [4, 1]}([[ARG_1]], [[ARG_2]]) : memref<1x128x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:        }

    // CHECK:        return [[D2S]] : memref<1x8x8x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotConvertSWDepthToSpaceToDMAIfNotBeneficialForBS2
// CHECK-SAME: [[INPUT:%.+]]: memref<1x1024x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>
func.func @NotConvertSWDepthToSpaceToDMAIfNotBeneficialForBS2(%arg0: memref<1x1024x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x1024x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 1]}(%arg1, %arg2) : memref<1x1024x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:        [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:        [[D2S:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
    // CHECK-SAME:           inputs([[INPUT]] as [[ARG_1:%[^:]+]]: memref<1x1024x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[OUTBUFFER]] as [[ARG_2:%[^:]+]]: memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>{
    // CHECK:            VPUIP.SW.Kernel.run {attrs = [2, 1]}([[ARG_1]], [[ARG_2]]) : memref<1x1024x12x12xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:        }

    // CHECK:        return [[D2S]] : memref<1x256x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteWithThreeAxis
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x128xf16, @DDR>)
func.func @ConvertMemPermuteWithThreeAxis(%arg0: memref<1x16x4x128xf16, @DDR>)
        -> memref<1x4x16x128xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x128xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs(%0 : memref<1x16x4x128xf16, [@CMX_NN, 0]>) -> memref<1x16x4x128xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x128xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[0, 2, 1, 3]]}(%arg2, %arg3) : memref<1x16x4x128xf16, [@CMX_NN, 0]>, memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x4x16x128xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x4x16x128xf16, {order = #NHWC}, @DDR>) -> memref<1x4x16x128xf16, {order = #NHWC}, @DDR>
    return %4: memref<1x4x16x128xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x128xf16, @DDR>) outputs([[VAR0]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) -> memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[VAR1]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x4x16x128xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR5:%.+]]  = VPUIP.Copy inputs([[VAR3]] : memref<1x4x16x128xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x4x16x128xf16, {order = #NHWC}, @DDR>) -> memref<1x4x16x128xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR5:%.+]] : memref<1x4x16x128xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteHWCToWHC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x76xf16, {order = #map}, @DDR>)
func.func @ConvertMemPermuteHWCToWHC(%arg0: memref<1x16x4x76xf16, {order = #map}, @DDR>)
        -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, {order = #map}, @DDR>) outputs(%0 : memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[1, 0, 3, 2]]}(%arg2, %arg3) : memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>, memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    return %4: memref<1x16x4x76xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x76xf16, {order = #map}, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = VPUIP.PermuteDMA <{mem_perm = #map1}> inputs([[VAR1]] : memref<1x16x4x76xf16, {order = #map}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR3]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR5]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteHWCToHCW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x76xf16, @DDR>)
func.func @ConvertMemPermuteHWCToHCW(%arg0: memref<1x16x4x76xf16, @DDR>)
        -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, @DDR>) outputs(%0 : memref<1x16x4x76xf16, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[1, 0, 3, 2]]}(%arg2, %arg3) : memref<1x16x4x76xf16, [@CMX_NN, 0]>, memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    return %4: memref<1x16x4x76xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x76xf16, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = VPUIP.PermuteDMA <{mem_perm = #map}> inputs([[VAR1]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR3]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR5]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteWHCToCHW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x76xf16, {order = #NWHC}, @DDR>)
func.func @ConvertMemPermuteWHCToCHW(%arg0: memref<1x16x4x76xf16, {order = #NWHC}, @DDR>)
        -> memref<1x16x4x76xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, {order = #NWHC}, @DDR>) outputs(%0 : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg2, %arg3) : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>, memref<1x16x4x76xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, @DDR>) -> memref<1x16x4x76xf16, @DDR>
    return %4: memref<1x16x4x76xf16, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x76xf16, {order = #NWHC}, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHWC}> inputs([[VAR1]] : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[VAR5]] : memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR7:%.+]] = memref.alloc() : memref<1x16x4x76xf16, @DDR>
    // CHECK:   [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR7]] : memref<1x16x4x76xf16, @DDR>) -> memref<1x16x4x76xf16, @DDR>
    // CHECK:   return [[VAR8]] : memref<1x16x4x76xf16, @DDR>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteWCHToCHW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x76xf16, {order = #NWCH}, @DDR>)
func.func @ConvertMemPermuteWCHToCHW(%arg0: memref<1x16x4x76xf16, {order = #NWCH}, @DDR>)
        -> memref<1x16x4x76xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, {order = #NWCH}, @DDR>) outputs(%0 : memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg2, %arg3) : memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>, memref<1x16x4x76xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, @DDR>) -> memref<1x16x4x76xf16, @DDR>
    return %4: memref<1x16x4x76xf16, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x76xf16, {order = #NWCH}, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = VPUIP.PermuteCast {dst_order = #NWHC, mem_perm = #NCHW} inputs([[VAR1]] : memref<1x16x4x76xf16, {order = #NWCH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHWC}> inputs([[VAR3]] : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[VAR5]] : memref<1x16x4x76xf16, {order = #NHCW}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR7:%.+]] = memref.alloc() : memref<1x16x4x76xf16, @DDR>
    // CHECK:   [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR7]] : memref<1x16x4x76xf16, @DDR>) -> memref<1x16x4x76xf16, @DDR>
    // CHECK:   return [[VAR8]] : memref<1x16x4x76xf16, @DDR>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteCWHToHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x76xf16, {order = #NCWH}, @DDR>)
func.func @ConvertMemPermuteCWHToHWC(%arg0: memref<1x16x4x76xf16, {order = #NCWH}, @DDR>)
        -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, {order = #NCWH}, @DDR>) outputs(%0 : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg2, %arg3) : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>, memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    return %4: memref<1x16x4x76xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x76xf16, {order = #NCWH}, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHWC}> inputs([[VAR1]] : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[VAR5]] : memref<1x16x4x76xf16, {order = #NWHC}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR7:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR7]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR8]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteWithMemPermHNWC
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x15x2x128xf16, @DDR>
func.func @ConvertMemPermuteWithMemPermHNWC(%arg0: memref<1x15x2x128xf16, @DDR>)
        -> memref<2x1x128x15xf16, @DDR> {
    %0 = memref.alloc() : memref<1x15x2x128xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x15x2x128xf16, @DDR>) outputs(%0 : memref<1x15x2x128xf16, [@CMX_NN, 0]>) -> memref<1x15x2x128xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<2x1x128x15xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
        inputs(%1 as %arg2: memref<1x15x2x128xf16, [@CMX_NN, 0]>)
        outputs(%2 as %arg3: memref<2x1x128x15xf16, [@CMX_NN, 0]>) on tile 0 -> memref<2x1x128x15xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[2, 0, 3, 1]]}(%arg2, %arg3) : memref<1x15x2x128xf16, [@CMX_NN, 0]>, memref<2x1x128x15xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<2x1x128x15xf16, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<2x1x128x15xf16, [@CMX_NN, 0]>) outputs(%3 : memref<2x1x128x15xf16, @DDR>) -> memref<2x1x128x15xf16, @DDR>
    return %4: memref<2x1x128x15xf16, @DDR>

    // CHECK:   [[BUFF_IN:%.+]] = memref.alloc() : memref<1x15x2x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_IN:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x15x2x128xf16, @DDR>) outputs([[BUFF_IN]] : memref<1x15x2x128xf16, [@CMX_NN, 0]>) -> memref<1x15x2x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTE_OUT:%.+]] = memref.alloc() : memref<2x1x128x15xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTE:%.+]] = VPUIP.PermuteDMA <{mem_perm = #map}> inputs([[COPY_IN]] : memref<1x15x2x128xf16, [@CMX_NN, 0]>) outputs([[PERMUTE_OUT]] : memref<2x1x128x15xf16, [@CMX_NN, 0]>) -> memref<2x1x128x15xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFF_OUT:%.+]] =  memref.alloc() : memref<2x1x128x15xf16, @DDR>
    // CHECK:   [[COPY_OUT:%.+]] = VPUIP.Copy inputs([[PERMUTE]] : memref<2x1x128x15xf16, [@CMX_NN, 0]>) outputs([[BUFF_OUT]] : memref<2x1x128x15xf16, @DDR>) -> memref<2x1x128x15xf16, @DDR>
    // CHECK:   return [[COPY_OUT]] : memref<2x1x128x15xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

// CHECK-LABEL: @WrapExpandandPermuteWithoutDistributedOp
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x24x24x!qElemType>)
func.func @WrapExpandandPermuteWithoutDistributedOp(%arg0: memref<1x3x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType> {
   %0 = memref.alloc() : memref<1x16x24x24x!qElemType>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs(%0 : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>

   return %1 : memref<1x16x24x24x!qElemType>

   //CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x24x24x!qElemType>
   //CHECK:   [[VAR1:%.+]] = VPUIP.ExpandDMA <{pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}> inputs([[ARG_0]] : memref<1x3x24x24x!qElemType>) outputs([[VAR0]] : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   //CHECK:   return [[VAR1]] : memref<1x16x24x24x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertPerAxisTileToDMA
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x1x1x1xf16, {order = #NHWC}, @DDR>)
func.func @ConvertPerAxisTileToDMA(%arg0: memref<1x1x1x1xf16, {order = #NHWC}, @DDR>)
        -> memref<1x512x1x1xf16, {order = #NHWC}, @DDR> {
    %cst_0 = const.Declare memref<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi32>
    %0 = memref.alloc() : memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst_0 : memref<4xsi32>) outputs(%2 : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
          inputs(%1 as %arg3: memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, %3 as %arg4: memref<4xsi32, [@CMX_NN, 0]>)
          outputs(%4 as %arg5: memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %6 = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%6 : memref<1x512x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x512x1x1xf16, {order = #NHWC}, @DDR>
    return %7: memref<1x512x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi32>
    // CHECK:   [[VAR1:%.+]] = memref.alloc() : memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x1x1x1xf16, {order = #NHWC}, @DDR>) outputs([[VAR1]] : memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = VPUIP.Copy inputs([[CST]] : memref<4xsi32>) outputs([[VAR3]] : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.+]] = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.+]] = VPUIP.PerAxisTileDMA <{axis = 1 : i64, tiles = 512 : i64}> inputs([[VAR2]] : memref<1x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR7:%.+]] = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x512x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR7]] : memref<1x512x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x512x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR8]] : memref<1x512x1x1xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_Tile(memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSI32PerAxisTileToDMA
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x1x1x1xsi32, {order = #NHWC}, @DDR>)
func.func @ConvertSI32PerAxisTileToDMA(%arg0: memref<1x1x1x1xsi32, {order = #NHWC}, @DDR>)
        -> memref<1x512x1x1xsi32, {order = #NHWC}, @DDR> {
    %cst_0 = const.Declare memref<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi32>
    %0 = memref.alloc() : memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xsi32, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst_0 : memref<4xsi32>) outputs(%2 : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
          inputs(%1 as %arg3: memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %3 as %arg4: memref<4xsi32, [@CMX_NN, 0]>)
          outputs(%4 as %arg5: memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %6 = memref.alloc() : memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%6 : memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>) -> memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>
    return %7: memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi32>
    // CHECK:   [[VAR1:%.+]] = memref.alloc() : memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x1x1x1xsi32, {order = #NHWC}, @DDR>) outputs([[VAR1]] : memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = VPUIP.Copy inputs([[CST]] : memref<4xsi32>) outputs([[VAR3]] : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.+]] = memref.alloc() : memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.+]] = VPUIP.PerAxisTileDMA <{axis = 1 : i64, tiles = 512 : i64}> inputs([[VAR2]] : memref<1x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR7:%.+]] = memref.alloc() : memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>
    // CHECK:   [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x512x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR7]] : memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>) -> memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>
    // CHECK:   return [[VAR8]] : memref<1x512x1x1xsi32, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertTileToDMAWithThreeAxisExpansion
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x2x3x4xf16, {order = #NHWC}, @DDR>)
func.func @ConvertTileToDMAWithThreeAxisExpansion(%arg0: memref<1x2x3x4xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x9x16xf16, {order = #NHWC}, @DDR> {
    %cst_0 = const.Declare memref<4xsi32> = dense<[1, 2, 3, 4]> : tensor<4xsi32>
    %0 = memref.alloc() : memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x2x3x4xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst_0 : memref<4xsi32>) outputs(%2 : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
          inputs(%1 as %arg3: memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %3 as %arg4: memref<4xsi32, [@CMX_NN, 0]>)
          outputs(%4 as %arg5: memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %6 = memref.alloc() : memref<1x4x9x16xf16, {order = #NHWC}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%6 : memref<1x4x9x16xf16, {order = #NHWC}, @DDR>) -> memref<1x4x9x16xf16, {order = #NHWC}, @DDR>
    return %7: memref<1x4x9x16xf16, {order = #NHWC}, @DDR>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<4xsi32> = dense<[1, 2, 3, 4]> : tensor<4xsi32>
    // CHECK:   [[VAR1:%.+]] = memref.alloc() : memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x2x3x4xf16, {order = #NHWC}, @DDR>) outputs([[VAR1]] : memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.+]] = VPUIP.Copy inputs([[CST]] : memref<4xsi32>) outputs([[VAR3]] : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>

    // CHECK:   [[OUTBUFFER_0:%.+]] = memref.alloc() : memref<1x4x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[PERAXISTILE_0:%.+]] = VPUIP.PerAxisTileDMA <{axis = 1 : i64, tiles = 2 : i64}>
    // CHECK:       inputs([[VAR2]] : memref<1x2x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTBUFFER_0]] : memref<1x4x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:   [[OUTBUFFER_1:%.+]] = memref.alloc() : memref<1x4x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[PERAXISTILE_1:%.+]] = VPUIP.PerAxisTileDMA <{axis = 3 : i64, tiles = 4 : i64}>
    // CHECK:       inputs([[PERAXISTILE_0]] : memref<1x4x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTBUFFER_1]] : memref<1x4x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:   [[OUTBUFFER_2:%.+]] = memref.alloc() : memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[PERAXISTILE_2:%.+]] = VPUIP.PerAxisTileDMA <{axis = 2 : i64, tiles = 3 : i64}>
    // CHECK:       inputs([[PERAXISTILE_1]] : memref<1x4x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTBUFFER_2]] : memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:   [[OUTBUFFER:%.+]] = memref.alloc() : memref<1x4x9x16xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[OUTCOPY:%.+]] = VPUIP.Copy inputs([[PERAXISTILE_2]] : memref<1x4x9x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTBUFFER]] : memref<1x4x9x16xf16, {order = #NHWC}, @DDR>)
    // CHECK:   return [[OUTCOPY]] : memref<1x4x9x16xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @convertUpsampling2DMANoMemSpaceWithNHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x256x16x32xf16, {order = #NHWC}>)
func.func @convertUpsampling2DMANoMemSpaceWithNHWC(%arg0: memref<1x256x16x32xf16, {order = #NHWC}>) -> memref<1x256x32x64xf16, {order = #NHWC}> {
    %0 = memref.alloc() : memref<1x256x32x64xf16, {order = #NHWC}>
    %1 = VPUIP.Upsampling {
                pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
                upsampling_factor = [2, 2, 1]}
            inputs(%arg0 : memref<1x256x16x32xf16, {order = #NHWC}>)
            outputs(%0 : memref<1x256x32x64xf16, {order = #NHWC}>) -> memref<1x256x32x64xf16, {order = #NHWC}>
    return %1 : memref<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare memref<1x256x32x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x256x32x64xf16, {order = #NHWC}>
    // CHECK:       [[DDR_BUFF:%.+]] = memref.alloc() : memref<1x256x32x64xf16, {order = #NHWC}>
    // CHECK:       [[CMX_BUFF:%.+]] = memref.alloc() : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[COPYZERO:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[CST]] : memref<1x256x32x64xf16, {order = #NHWC}>)
    // CHECK:              outputs([[CMX_BUFF]] : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       [[DMA:%.+]] = VPUIP.UpsamplingDMAOp <{port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}>
    // CHECK:              inputs([[ARG_0]] : memref<1x256x16x32xf16, {order = #NHWC}>)
    // CHECK:              outputs([[COPYZERO]] : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       [[COPYOUT:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[DMA]] : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:              outputs([[DDR_BUFF]] : memref<1x256x32x64xf16, {order = #NHWC}>)

    // CHECK:       return [[COPYOUT]] : memref<1x256x32x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @convertUpsampling2DMAHasMemSpaceWithNHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x256x16x32xf16, {order = #NHWC}>)
func.func @convertUpsampling2DMAHasMemSpaceWithNHWC(%arg0: memref<1x256x16x32xf16, {order = #NHWC}>) -> memref<1x256x32x64xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x256x32x64xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.Upsampling {
                pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
                upsampling_factor = [2, 2, 1]}
            inputs(%arg0 : memref<1x256x16x32xf16, {order = #NHWC}>)
            outputs(%0 : memref<1x256x32x64xf16, {order = #NHWC}, @DDR>) -> memref<1x256x32x64xf16, {order = #NHWC}, @DDR>
    return %1 : memref<1x256x32x64xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[CST:%.+]] = const.Declare memref<1x256x32x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x256x32x64xf16, {order = #NHWC}>
    // CHECK:       [[DDR_BUFF:%.+]] = memref.alloc() : memref<1x256x32x64xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[CMX_BUFF:%.+]] = memref.alloc() : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[COPYZERO:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[CST]] : memref<1x256x32x64xf16, {order = #NHWC}>)
    // CHECK:              outputs([[CMX_BUFF]] : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       [[DMA:%.+]] = VPUIP.UpsamplingDMAOp <{port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}>
    // CHECK:              inputs([[ARG_0]] : memref<1x256x16x32xf16, {order = #NHWC}>)
    // CHECK:              outputs([[COPYZERO]] : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       [[COPYOUT:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[DMA]] : memref<1x256x32x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:              outputs([[DDR_BUFF]] : memref<1x256x32x64xf16, {order = #NHWC}, @DDR>)

    // CHECK:       return [[COPYOUT]] : memref<1x256x32x64xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @convertUpsampling2DMANoMemSpaceWithNCHW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x256x16x32xf16>)
func.func @convertUpsampling2DMANoMemSpaceWithNCHW(%arg0: memref<1x256x16x32xf16>) -> memref<1x256x32x64xf16> {
    %0 = memref.alloc() : memref<1x256x32x64xf16>
    %1 = VPUIP.Upsampling {
                pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
                upsampling_factor = [2, 2, 1]}
            inputs(%arg0 : memref<1x256x16x32xf16>)
            outputs(%0 : memref<1x256x32x64xf16>) -> memref<1x256x32x64xf16>
    return %1 : memref<1x256x32x64xf16>

    // CHECK:       [[CST:%.+]] = const.Declare memref<1x256x32x64xf16> = dense<0.000000e+00> : tensor<1x256x32x64xf16>
    // CHECK:       [[DDR_BUFF:%.+]] = memref.alloc() : memref<1x256x32x64xf16>
    // CHECK:       [[CMX_BUFF:%.+]] = memref.alloc() : memref<1x256x32x64xf16, [@CMX_NN, 0]>

    // CHECK:       [[COPYZERO:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[CST]] : memref<1x256x32x64xf16>)
    // CHECK:              outputs([[CMX_BUFF]] : memref<1x256x32x64xf16, [@CMX_NN, 0]>)
    // CHECK:       [[DMA:%.+]] = VPUIP.UpsamplingDMAOp <{port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}>
    // CHECK:              inputs([[ARG_0]] : memref<1x256x16x32xf16>)
    // CHECK:              outputs([[COPYZERO]] : memref<1x256x32x64xf16, [@CMX_NN, 0]>)
    // CHECK:       [[COPYOUT:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[DMA]] : memref<1x256x32x64xf16, [@CMX_NN, 0]>)
    // CHECK:              outputs([[DDR_BUFF]] : memref<1x256x32x64xf16>)

    // CHECK:       return [[COPYOUT]] : memref<1x256x32x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotMoveUpsamplingDMAInCMXWithLargeSize
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x64x128x128xf16>)
func.func @NotMoveUpsamplingDMAInCMXWithLargeSize(%arg0: memref<1x64x128x128xf16>) -> memref<1x64x256x256xf16> {
    %0 = memref.alloc() : memref<1x64x256x256xf16>
    %1 = VPUIP.Upsampling {
                pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
                upsampling_factor = [2, 2, 1]}
            inputs(%arg0 : memref<1x64x128x128xf16>)
            outputs(%0 : memref<1x64x256x256xf16>) -> memref<1x64x256x256xf16>
    return %1 : memref<1x64x256x256xf16>

    // CHECK:       [[CST:%.+]] = const.Declare memref<1x64x256x256xf16> = dense<0.000000e+00> : tensor<1x64x256x256xf16>
    // CHECK:       [[DDR_BUFF:%.+]] = memref.alloc() : memref<1x64x256x256xf16>

    // CHECK:       [[COPYZERO:%.+]] = VPUIP.Copy
    // CHECK:              inputs([[CST]] : memref<1x64x256x256xf16>)
    // CHECK:              outputs([[DDR_BUFF]] : memref<1x64x256x256xf16>)
    // CHECK:       [[DMA:%.+]] = VPUIP.UpsamplingDMAOp <{port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}>
    // CHECK:              inputs([[ARG_0]] : memref<1x64x128x128xf16>)
    // CHECK:              outputs([[COPYZERO]] : memref<1x64x256x256xf16>)

    // CHECK:       return [[DMA]] : memref<1x64x256x256xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotConvertD0IsInPermutationCase0
// CHECK-SAME:    [[INPUT:%.+]]:  memref<10x16x4x76xf16, {order = #NCWH}, @DDR>
func.func @NotConvertD0IsInPermutationCase0(%arg0: memref<10x16x4x76xf16, {order = #NCWH}, @DDR>)
        -> memref<10x16x4x76xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<10x16x4x76xf16, {order = #NCWH}, @DDR>) outputs(%0 : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[3, 1, 2, 0]]}(%arg2, %arg3) : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>, memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    return %4: memref<10x16x4x76xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<10x16x4x76xf16, {order = #NCWH}, @DDR>) outputs([[VAR0]] : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[COPY0]] : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<10x4x16x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[PERMUTE_CAST]] : memref<10x4x16x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x10x1216xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = memref.alloc() : memref<1x4x1216x10xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTE_DMA0:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[GENERIC_RESHAPE0]] : memref<1x4x10x1216xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR1]] : memref<1x4x1216x10xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x1216x10xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTE_DMA0]] : memref<1x4x1216x10xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x4864x10xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x1x10x4864xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTE_DMA1:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[GENERIC_RESHAPE1]] : memref<1x1x4864x10xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x1x10x4864xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x10x4864xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTE_DMA1]] : memref<1x1x10x4864xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.+]] = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.Copy inputs([[GENERIC_RESHAPE2]] : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[COPY1]] : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotConvertD0IsInPermutationCase1
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x76xf16, {order = #NCWH}, @DDR>)
func.func @NotConvertD0IsInPermutationCase1(%arg0: memref<1x16x4x76xf16, {order = #NCWH}, @DDR>)
        -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, {order = #NCWH}, @DDR>) outputs(%0 : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 3, 1, 0]]}(%arg2, %arg3) : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>, memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    return %4: memref<1x16x4x76xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x4x76xf16, {order = #NCWH}, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[RESULTS:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK-SAME: inputs([[COPY0]] as [[ARG_1:%[^:]+]]: memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[VAR1]] as [[ARG_2:%[^:]+]]: memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [
    // CHECK:     [2, 3, 1, 0]
    // CHECK:     ]}([[ARG_1]], [[ARG_2]]) : memref<1x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>, memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.Copy inputs([[RESULTS]] : memref<1x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[COPY1]] : memref<1x16x4x76xf16, {order = #NHWC}, @DDR>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotConvertD0IsInPermutationCase2
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<10x16x4x76xf16, {order = #NCWH}, @DDR>)
func.func @NotConvertD0IsInPermutationCase2(%arg0: memref<10x16x4x76xf16, {order = #NCWH}, @DDR>)
        -> memref<10x16x4x76xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<10x16x4x76xf16, {order = #NCWH}, @DDR>) outputs(%0 : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}(%arg2, %arg3) : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>, memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    return %4: memref<10x16x4x76xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<10x16x4x76xf16, {order = #NCWH}, @DDR>) outputs([[VAR0]] : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>) -> memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.+]] = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[RESULTS:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK-SAME: inputs([[COPY0]] as [[ARG_1:%[^:]+]]: memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[VAR1]] as [[ARG_2:%[^:]+]]: memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>{
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [
    // CHECK:     [3, 2, 0, 1]
    // CHECK:     ]}([[ARG_1]], [[ARG_2]]) : memref<10x16x4x76xf16, {order = #NCWH}, [@CMX_NN, 0]>, memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[VAR2:%.+]] = memref.alloc() : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.Copy inputs([[RESULTS]] : memref<10x16x4x76xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>) -> memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[COPY1]] : memref<10x16x4x76xf16, {order = #NHWC}, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToCNWH
// CHECK-SAME: ([[INPUT:%.+]]: memref<4x86x256x4xf16, @DDR>)
func.func @ConvertMemPermuteNCHWToCNWH(%arg0: memref<4x86x256x4xf16, @DDR>)
        -> memref<86x4x4x256xf16, @DDR> {
    %0 = memref.alloc() : memref<4x86x256x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<4x86x256x4xf16, @DDR>) outputs(%0 : memref<4x86x256x4xf16, [@CMX_NN, 0]>) -> memref<4x86x256x4xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<86x4x4x256xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
        inputs(%1 as %arg2: memref<4x86x256x4xf16, [@CMX_NN, 0]>)
        outputs(%2 as %arg3: memref<86x4x4x256xf16, [@CMX_NN, 0]>) on tile 0 -> memref<86x4x4x256xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[1, 0, 3, 2]]}(%arg2, %arg3) : memref<4x86x256x4xf16, [@CMX_NN, 0]>, memref<86x4x4x256xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<86x4x4x256xf16, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<86x4x4x256xf16, [@CMX_NN, 0]>) outputs(%3 : memref<86x4x4x256xf16, @DDR>) -> memref<86x4x4x256xf16, @DDR>
    return %4: memref<86x4x4x256xf16, @DDR>

    // CHECK:   [[COPY_BUFF_0:%.+]] = memref.alloc() : memref<4x86x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<4x86x256x4xf16, @DDR>) outputs([[COPY_BUFF_0]] : memref<4x86x256x4xf16, [@CMX_NN, 0]>) -> memref<4x86x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[RESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<4x86x256x4xf16, [@CMX_NN, 0]>) -> memref<1x344x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERM_DMA_BUFF_0:%.+]] = memref.alloc() : memref<1x344x4x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERM_DMA_0:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NCWH}> inputs([[RESHAPE_0]] : memref<1x344x256x4xf16, [@CMX_NN, 0]>) outputs([[PERM_DMA_BUFF_0]] : memref<1x344x4x256xf16, [@CMX_NN, 0]>) -> memref<1x344x4x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[RESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[PERM_DMA_0]] : memref<1x344x4x256xf16, [@CMX_NN, 0]>) -> memref<1x4x86x1024xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERM_DMA_BUFF_1:%.+]] = memref.alloc() : memref<1x86x4x1024xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERM_DMA_1:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}> inputs([[RESHAPE_1]] : memref<1x4x86x1024xf16, [@CMX_NN, 0]>) outputs([[PERM_DMA_BUFF_1]] : memref<1x86x4x1024xf16, [@CMX_NN, 0]>) -> memref<1x86x4x1024xf16, [@CMX_NN, 0]>
    // CHECK:   [[RESHAPE_2:%.+]] = VPUIP.GenericReshape inputs([[PERM_DMA_1]] : memref<1x86x4x1024xf16, [@CMX_NN, 0]>) -> memref<86x4x4x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_BUFF_1:%.+]] = memref.alloc() : memref<86x4x4x256xf16, @DDR>
    // CHECK:   [[COPY_1:%.+]] = VPUIP.Copy inputs([[RESHAPE_2]] : memref<86x4x4x256xf16, [@CMX_NN, 0]>) outputs([[COPY_BUFF_1]] : memref<86x4x4x256xf16, @DDR>) -> memref<86x4x4x256xf16, @DDR>
    // CHECK:   return [[COPY_1]] : memref<86x4x4x256xf16, @DDR>
}
