//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --convert-to-dma --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToNHCW
// CHECK-SAME:    [[INPUT:%.+]]: memref<6x4x8x512xf16, @DDR>
func.func @ConvertMemPermuteNCHWToNHCW(%arg0: memref<6x4x8x512xf16, @DDR>)
                                       -> memref<6x8x4x512xf16, @DDR> {
    %0 = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<6x4x8x512xf16, @DDR>)
            outputs(%0 as %arg3: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                -> memref<6x4x8x512xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x4x8x512xf16, @DDR>)
                    outputs(%arg3 : memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                        -> memref<6x4x8x512xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<6x8x4x512xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<6x8x4x512xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<6x8x4x512xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[0, 2, 1, 3]]}(%arg2, %arg3)
                : memref<6x4x8x512xf16, [@CMX_NN, 0]>, memref<6x8x4x512xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<6x8x4x512xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<6x8x4x512xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<6x8x4x512xf16, @DDR>)
                -> memref<6x8x4x512xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x8x4x512xf16, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<6x8x4x512xf16, @DDR>)
                        -> memref<6x8x4x512xf16, @DDR>
    }

    return %5: memref<6x8x4x512xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<6x4x8x512xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<6x8x4x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[COPY0]] : memref<6x4x8x512xf16, [@CMX_NN, 0]>) -> memref<1x24x8x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x8x24x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x24x8x512xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x8x24x512xf16, [@CMX_NN, 0]>) -> memref<1x8x24x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x8x24x512xf16, [@CMX_NN, 0]>) -> memref<1x8x6x2048xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x8x6x2048xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<6x8x4x512xf16, [@CMX_NN, 0]>) -> memref<6x8x4x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_BUFF3:%.+]] = memref.alloc() : memref<6x8x4x512xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[PERMUTEDMA1]] as {{[^:]+}}: memref<6x8x4x512xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[COPY_BUFF3]] as {{[^:]+}}: memref<6x8x4x512xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<6x8x4x512xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToNHCWWithDifferentDimsOrder
// CHECK-SAME:    [[INPUT:%.+]]: memref<6x4x8x512xf16, @DDR>
func.func @ConvertMemPermuteNCHWToNHCWWithDifferentDimsOrder(%arg0: memref<6x4x8x512xf16, @DDR>)
                                       -> memref<6x512x8x4xf16, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<6x4x8x512xf16, @DDR>)
            outputs(%0 as %arg3: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                -> memref<6x4x8x512xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x4x8x512xf16, @DDR>)
                    outputs(%arg3 : memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                        -> memref<6x4x8x512xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>)
                on tile 0 -> memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[0, 2, 1, 3]]}(%arg2, %arg3)
                : memref<6x4x8x512xf16, [@CMX_NN, 0]>, memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<6x512x8x4xf16, #NHWC, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<6x512x8x4xf16, #NHWC, @DDR>)
                -> memref<6x512x8x4xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<6x512x8x4xf16, #NHWC, @DDR>)
                        -> memref<6x512x8x4xf16, #NHWC, @DDR>
    }

    return %5: memref<6x512x8x4xf16, #NHWC, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<6x4x8x512xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[COPY0]] : memref<6x4x8x512xf16, [@CMX_NN, 0]>) -> memref<6x512x4x8xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[PERMUTECAST]] : memref<6x512x4x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x24x8xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x512x8x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x512x24x8xf16, #NHWC, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x512x8x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x8x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x512x8x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2048x8x6xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x2048x8x6xf16, #NHWC, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[COPY_BUFF3:%.+]] = memref.alloc() : memref<6x512x8x4xf16, #NHWC, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[PERMUTEDMA1]] as {{[^:]+}}: memref<6x512x8x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[COPY_BUFF3]] as {{[^:]+}}: memref<6x512x8x4xf16, #NHWC, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<6x512x8x4xf16, #NHWC, @DDR>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToHCNW
// CHECK-SAME:    [[INPUT:%.+]]: memref<6x4x8x512xf16, @DDR>
func.func @ConvertMemPermuteNCHWToHCNW(%arg0: memref<6x4x8x512xf16, @DDR>)
                                       -> memref<8x4x6x512xf16, @DDR> {
    %0 = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<6x4x8x512xf16, @DDR>)
            outputs(%0 as %arg3: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                -> memref<6x4x8x512xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x4x8x512xf16, @DDR>)
                    outputs(%arg3 : memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                        -> memref<6x4x8x512xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<8x4x6x512xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<8x4x6x512xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<8x4x6x512xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[0, 3, 2, 1]]}(%arg2, %arg3)
                : memref<6x4x8x512xf16, [@CMX_NN, 0]>, memref<8x4x6x512xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<8x4x6x512xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<8x4x6x512xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<8x4x6x512xf16, @DDR>)
                -> memref<8x4x6x512xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<8x4x6x512xf16, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<8x4x6x512xf16, @DDR>)
                        -> memref<8x4x6x512xf16, @DDR>
    }

    return %5: memref<8x4x6x512xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<6x4x8x512xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<8x4x6x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[COPY0]] : memref<6x4x8x512xf16, [@CMX_NN, 0]>) -> memref<1x24x8x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x8x24x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x24x8x512xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x8x24x512xf16, [@CMX_NN, 0]>) -> memref<1x8x24x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x8x24x512xf16, [@CMX_NN, 0]>) -> memref<1x48x4x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF3:%.+]] = memref.alloc() : memref<1x4x48x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x48x4x512xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF3]] : memref<1x4x48x512xf16, [@CMX_NN, 0]>) -> memref<1x4x48x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA1]] : memref<1x4x48x512xf16, [@CMX_NN, 0]>) -> memref<1x4x8x3072xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA2:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE2]] : memref<1x4x8x3072xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<8x4x6x512xf16, [@CMX_NN, 0]>) -> memref<8x4x6x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_BUFF4:%.+]] = memref.alloc() : memref<8x4x6x512xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[PERMUTEDMA2]] as {{[^:]+}}: memref<8x4x6x512xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[COPY_BUFF4]] as {{[^:]+}}: memref<8x4x6x512xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<8x4x6x512xf16, @DDR>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToNWHC
// CHECK-SAME:    [[INPUT:%.+]]: memref<6x4x8x512xf16, @DDR>
func.func @ConvertMemPermuteNCHWToNWHC(%arg0: memref<6x4x8x512xf16, @DDR>)
                                       -> memref<6x512x8x4xf16, @DDR> {
    %0 = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<6x4x8x512xf16, @DDR>)
            outputs(%0 as %arg3: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                -> memref<6x4x8x512xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x4x8x512xf16, @DDR>)
                    outputs(%arg3 : memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                        -> memref<6x4x8x512xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<6x512x8x4xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<6x512x8x4xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<6x512x8x4xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg2, %arg3)
                : memref<6x4x8x512xf16, [@CMX_NN, 0]>, memref<6x512x8x4xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<6x512x8x4xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<6x512x8x4xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<6x512x8x4xf16, @DDR>)
                -> memref<6x512x8x4xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<6x512x8x4xf16, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<6x512x8x4xf16, @DDR>)
                        -> memref<6x512x8x4xf16, @DDR>
    }

    return %5: memref<6x512x8x4xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<6x4x8x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<6x4x8x512xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<6x4x8x512xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<6x512x8x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[COPY0]] : memref<6x4x8x512xf16, [@CMX_NN, 0]>) -> memref<1x6x4x4096xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x6x4096x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NCWH} inputs([[GENERIC_RESHAPE0]] : memref<1x6x4x4096xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x6x4096x4xf16, [@CMX_NN, 0]>) -> memref<1x6x4096x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x6x4096x4xf16, [@CMX_NN, 0]>) -> memref<1x48x512x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF3:%.+]] = memref.alloc() : memref<1x512x48x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x48x512x4xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF3]] : memref<1x512x48x4xf16, [@CMX_NN, 0]>) -> memref<1x512x48x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA1]] : memref<1x512x48x4xf16, [@CMX_NN, 0]>) -> memref<1x512x6x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA2:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE2]] : memref<1x512x6x32xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<6x512x8x4xf16, [@CMX_NN, 0]>) -> memref<6x512x8x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_BUFF4:%.+]] = memref.alloc() : memref<6x512x8x4xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[PERMUTEDMA2]] as {{[^:]+}}: memref<6x512x8x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[COPY_BUFF4]] as {{[^:]+}}: memref<6x512x8x4xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<6x512x8x4xf16, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToCWNH
// CHECK-SAME:    [[INPUT:%.+]]: memref<86x4x256x4xf16, @DDR>
func.func @ConvertMemPermuteNCHWToCWNH(%arg0: memref<86x4x256x4xf16, @DDR>)
                                       -> memref<4x4x86x256xf16, @DDR> {
    %0 = memref.alloc() : memref<86x4x256x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<86x4x256x4xf16, @DDR>)
            outputs(%0 as %arg3: memref<86x4x256x4xf16, [@CMX_NN, 0]>)
                -> memref<86x4x256x4xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<86x4x256x4xf16, @DDR>)
                    outputs(%arg3 : memref<86x4x256x4xf16, [@CMX_NN, 0]>)
                        -> memref<86x4x256x4xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<4x4x86x256xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<86x4x256x4xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<4x4x86x256xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<4x4x86x256xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[1, 3, 0, 2]]}(%arg2, %arg3)
                : memref<86x4x256x4xf16, [@CMX_NN, 0]>, memref<4x4x86x256xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<4x4x86x256xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<4x4x86x256xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<4x4x86x256xf16, @DDR>)
                -> memref<4x4x86x256xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<4x4x86x256xf16, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<4x4x86x256xf16, @DDR>)
                        -> memref<4x4x86x256xf16, @DDR>
    }

    return %5: memref<4x4x86x256xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<86x4x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<86x4x256x4xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<86x4x256x4xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<86x256x4x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[COPY0]] : memref<86x4x256x4xf16, [@CMX_NN, 0]>) -> memref<1x344x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x256x344x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x344x256x4xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x256x344x4xf16, [@CMX_NN, 0]>) -> memref<1x256x344x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x256x344x4xf16, [@CMX_NN, 0]>) -> memref<1x256x86x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x256x86x16xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<86x256x4x4xf16, [@CMX_NN, 0]>) -> memref<86x256x4x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA1]] : memref<86x256x4x4xf16, [@CMX_NN, 0]>) -> memref<1x22016x16x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF3:%.+]] = memref.alloc() : memref<1x16x22016x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA2:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE2]] : memref<1x22016x16x1xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF3]] : memref<1x16x22016x1xf16, [@CMX_NN, 0]>) -> memref<1x16x22016x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE3:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA2]] : memref<1x16x22016x1xf16, [@CMX_NN, 0]>) -> memref<4x4x86x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF4:%.+]] = memref.alloc() : memref<4x4x86x256xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[GENERIC_RESHAPE3]] as {{[^:]+}}: memref<4x4x86x256xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[PERMUTEDMA_BUFF4]] as {{[^:]+}}: memref<4x4x86x256xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<4x4x86x256xf16, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToHNWC
// CHECK-SAME:    [[INPUT:%.+]]: memref<4x4x86x256xf16, @DDR>
func.func @ConvertMemPermuteNCHWToHNWC(%arg0: memref<4x4x86x256xf16, @DDR>)
                                       -> memref<86x4x256x4xf16, @DDR> {
    %0 = memref.alloc() : memref<4x4x86x256xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<4x4x86x256xf16, @DDR>)
            outputs(%0 as %arg3: memref<4x4x86x256xf16, [@CMX_NN, 0]>)
                -> memref<4x4x86x256xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<4x4x86x256xf16, @DDR>)
                    outputs(%arg3 : memref<4x4x86x256xf16, [@CMX_NN, 0]>)
                        -> memref<4x4x86x256xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<86x4x256x4xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<4x4x86x256xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<86x4x256x4xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<86x4x256x4xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[2, 0, 3, 1]]}(%arg2, %arg3)
                : memref<4x4x86x256xf16, [@CMX_NN, 0]>, memref<86x4x256x4xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<86x4x256x4xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<86x4x256x4xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<86x4x256x4xf16, @DDR>)
                -> memref<86x4x256x4xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<86x4x256x4xf16, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<86x4x256x4xf16, @DDR>)
                        -> memref<86x4x256x4xf16, @DDR>
    }

    return %5: memref<86x4x256x4xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<4x4x86x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<4x4x86x256xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<4x4x86x256xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[COPY0]] : memref<4x4x86x256xf16, [@CMX_NN, 0]>) -> memref<1x16x86x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<1x86x16x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x16x86x256xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<1x86x16x256xf16, [@CMX_NN, 0]>) -> memref<1x86x16x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x86x16x256xf16, [@CMX_NN, 0]>) -> memref<1x344x4x256xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x344x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NCWH} inputs([[GENERIC_RESHAPE1]] : memref<1x344x4x256xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x344x256x4xf16, [@CMX_NN, 0]>) -> memref<1x344x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA1]] : memref<1x344x256x4xf16, [@CMX_NN, 0]>) -> memref<86x4x256x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF3:%.+]] = memref.alloc() : memref<86x4x256x4xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[GENERIC_RESHAPE2]] as {{[^:]+}}: memref<86x4x256x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[PERMUTEDMA_BUFF3]] as {{[^:]+}}: memref<86x4x256x4xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<86x4x256x4xf16, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xui4, [@CMX_NN, 0]>, memref<*xui4, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToCWNH
// CHECK-SAME:    [[INPUT:%.+]]: memref<86x4x256x4xui4, @DDR>
func.func @ConvertMemPermuteNCHWToCWNH(%arg0: memref<86x4x256x4xui4, @DDR>)
                                       -> memref<4x4x86x256xui4, @DDR> {
    %0 = memref.alloc() : memref<86x4x256x4xui4, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg2: memref<86x4x256x4xui4, @DDR>)
            outputs(%0 as %arg3: memref<86x4x256x4xui4, [@CMX_NN, 0]>)
                -> memref<86x4x256x4xui4, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<86x4x256x4xui4, @DDR>)
                    outputs(%arg3 : memref<86x4x256x4xui4, [@CMX_NN, 0]>)
                        -> memref<86x4x256x4xui4, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<4x4x86x256xui4, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg2: memref<86x4x256x4xui4, [@CMX_NN, 0]>)
                outputs(%2 as %arg3: memref<4x4x86x256xui4, [@CMX_NN, 0]>)
                on tile 0 -> memref<4x4x86x256xui4, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[1, 3, 0, 2]]}(%arg2, %arg3)
                : memref<86x4x256x4xui4, [@CMX_NN, 0]>, memref<4x4x86x256xui4, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<4x4x86x256xui4, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<4x4x86x256xui4, [@CMX_NN, 0]>)
            outputs(%4 as %arg3: memref<4x4x86x256xui4, @DDR>)
                -> memref<4x4x86x256xui4, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg2 : memref<4x4x86x256xui4, [@CMX_NN, 0]>)
                    outputs(%arg3 : memref<4x4x86x256xui4, @DDR>)
                        -> memref<4x4x86x256xui4, @DDR>
    }

    return %5: memref<4x4x86x256xui4, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK-NOT:   VPUIP.PermuteDMA
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToWCHN
// CHECK-SAME:    [[INPUT:%.+]]: memref<4x2x121x3xf16, @DDR>
func.func @ConvertMemPermuteNCHWToWCHN(%arg0: memref<4x2x121x3xf16, @DDR>)
                                       -> memref<3x2x121x4xf16, @DDR> {
    %0 = memref.alloc() : memref<4x2x121x3xf16, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg1: memref<4x2x121x3xf16, @DDR>)
            outputs(%0 as %arg2: memref<4x2x121x3xf16, [@CMX_NN, 0]>)
                -> memref<4x2x121x3xf16, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg1 : memref<4x2x121x3xf16, @DDR>)
                    outputs(%arg2 : memref<4x2x121x3xf16, [@CMX_NN, 0]>)
                        -> memref<4x2x121x3xf16, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<3x2x121x4xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg1: memref<4x2x121x3xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg2: memref<3x2x121x4xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<3x2x121x4xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[3, 1, 2, 0]]}(%arg1, %arg2)
                : memref<4x2x121x3xf16, [@CMX_NN, 0]>, memref<3x2x121x4xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<3x2x121x4xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg1: memref<3x2x121x4xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg2: memref<3x2x121x4xf16, @DDR>)
                -> memref<3x2x121x4xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg1 : memref<3x2x121x4xf16, [@CMX_NN, 0]>)
                    outputs(%arg2 : memref<3x2x121x4xf16, @DDR>)
                        -> memref<3x2x121x4xf16, @DDR>
    }

    return %5: memref<3x2x121x4xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<4x2x121x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<4x2x121x3xf16, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<4x2x121x3xf16, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[COPY0]] : memref<4x2x121x3xf16, [@CMX_NN, 0]>) -> memref<1x4x242x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<1x242x4x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x4x242x3xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<1x242x4x3xf16, [@CMX_NN, 0]>) -> memref<1x242x4x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x242x4x3xf16, [@CMX_NN, 0]>) -> memref<1x968x3x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x3x968x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x968x3x1xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x3x968x1xf16, [@CMX_NN, 0]>) -> memref<1x3x968x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA1]] : memref<1x3x968x1xf16, [@CMX_NN, 0]>) -> memref<3x2x121x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF3:%.+]] = memref.alloc() : memref<3x2x121x4xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[GENERIC_RESHAPE2]] as {{[^:]+}}: memref<3x2x121x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[PERMUTEDMA_BUFF3]] as {{[^:]+}}: memref<3x2x121x4xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<3x2x121x4xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertMemPermuteNCHWToWCHNWithDifferentDimsOrder
// CHECK-SAME:    [[INPUT:%.+]]: memref<4x3x2x121xf16, #NHWC, @DDR>
func.func @ConvertMemPermuteNCHWToWCHNWithDifferentDimsOrder(%arg0: memref<4x3x2x121xf16, #NHWC, @DDR>)
                                       -> memref<3x2x121x4xf16, @DDR> {
    %0 = memref.alloc() : memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling
            inputs(%arg0 as %arg1: memref<4x3x2x121xf16, #NHWC, @DDR>)
            outputs(%0 as %arg2: memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]> {
        %inner = VPUIP.Copy
                    inputs(%arg1 : memref<4x3x2x121xf16, #NHWC, @DDR>)
                    outputs(%arg2 : memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>)
                        -> memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>
    }

    %2 = memref.alloc() : memref<3x2x121x4xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_MemPermute
                inputs(%1 as %arg1: memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%2 as %arg2: memref<3x2x121x4xf16, [@CMX_NN, 0]>)
                on tile 0 -> memref<3x2x121x4xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [[3, 1, 2, 0]]}(%arg1, %arg2)
                : memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>, memref<3x2x121x4xf16, [@CMX_NN, 0]>
    }

    %4 = memref.alloc() : memref<3x2x121x4xf16, @DDR>
    %5 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg1: memref<3x2x121x4xf16, [@CMX_NN, 0]>)
            outputs(%4 as %arg2: memref<3x2x121x4xf16, @DDR>)
                -> memref<3x2x121x4xf16, @DDR> {
        %inner = VPUIP.Copy
                    inputs(%arg1 : memref<3x2x121x4xf16, [@CMX_NN, 0]>)
                    outputs(%arg2 : memref<3x2x121x4xf16, @DDR>)
                        -> memref<3x2x121x4xf16, @DDR>
    }

    return %5: memref<3x2x121x4xf16, @DDR>

    // CHECK:   [[COPY_BUFF0:%.+]] = memref.alloc() : memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as {{[^:]+}}: memref<4x3x2x121xf16, #NHWC, @DDR>)
    // CHECK-SAME:    outputs([[COPY_BUFF0]] as {{[^:]+}}: memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[PERMUTE_CAST0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[COPY0]] : memref<4x3x2x121xf16, #NHWC, [@CMX_NN, 0]>) -> memref<4x2x121x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[PERMUTE_CAST0]] : memref<4x2x121x3xf16, [@CMX_NN, 0]>) -> memref<1x4x242x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF1:%.+]] = memref.alloc() : memref<1x242x4x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA0:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE0]] : memref<1x4x242x3xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF1]] : memref<1x242x4x3xf16, [@CMX_NN, 0]>) -> memref<1x242x4x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA0]] : memref<1x242x4x3xf16, [@CMX_NN, 0]>) -> memref<1x968x3x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF2:%.+]] = memref.alloc() : memref<1x3x968x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA1:%.+]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[GENERIC_RESHAPE1]] : memref<1x968x3x1xf16, [@CMX_NN, 0]>) outputs([[PERMUTEDMA_BUFF2]] : memref<1x3x968x1xf16, [@CMX_NN, 0]>) -> memref<1x3x968x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[PERMUTEDMA1]] : memref<1x3x968x1xf16, [@CMX_NN, 0]>) -> memref<3x2x121x4xf16, [@CMX_NN, 0]>
    // CHECK:   [[PERMUTEDMA_BUFF3:%.+]] = memref.alloc() : memref<3x2x121x4xf16, @DDR>
    // CHECK:   [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[GENERIC_RESHAPE2]] as {{[^:]+}}: memref<3x2x121x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[PERMUTEDMA_BUFF3]] as {{[^:]+}}: memref<3x2x121x4xf16, @DDR>)
    // CHECK:         [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[COPY1]] : memref<3x2x121x4xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

// CHECK-LABEL: @ConvertExpandWithBlockArgumentAsOutput
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x3x24x24x!qElemType>
// CHECK-SAME:    [[OUTPUT:%.+]]: memref<1x16x24x24x!qElemType>
func.func @ConvertExpandWithBlockArgumentAsOutput(%arg0: memref<1x3x24x24x!qElemType>, %arg1: memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType> {
   %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs(%arg1 : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>

   return %0 : memref<1x16x24x24x!qElemType>

   //CHECK:   [[VAR0:%.*]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs([[INPUT]] : memref<1x3x24x24x!qElemType>) outputs([[OUTPUT]] : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   //CHECK:   return [[VAR0]] : memref<1x16x24x24x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @convertUpsamplingWithBlockArgumentAsOutput
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x256x16x32xf16>
// CHECK-SAME:    [[OUTPUT:%.+]]: memref<1x256x32x64xf16>
func.func @convertUpsamplingWithBlockArgumentAsOutput(%arg0: memref<1x256x16x32xf16>, %arg1: memref<1x256x32x64xf16>) -> memref<1x256x32x64xf16> {
    %0 = VPUIP.Upsampling {
                pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
                upsampling_factor = [2, 2, 1]}
            inputs(%arg0 : memref<1x256x16x32xf16>)
            outputs(%arg1 : memref<1x256x32x64xf16>) -> memref<1x256x32x64xf16>
    return %0 : memref<1x256x32x64xf16>

    // CHECK:       [[CST:%.*]] = const.Declare memref<1x256x32x64xf16> = dense<0.000000e+00> : tensor<1x256x32x64xf16>
    // CHECK:       [[CMX_BUFF:%.*]] = memref.alloc() : memref<1x256x32x64xf16, [@CMX_NN, 0]>

    // CHECK:       [[COPYZERO:%.*]] = VPUIP.Copy
    // CHECK:              inputs([[CST]] : memref<1x256x32x64xf16>)
    // CHECK:              outputs([[CMX_BUFF]] : memref<1x256x32x64xf16, [@CMX_NN, 0]>)
    // CHECK:       [[DMA:%.*]] = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
    // CHECK:              inputs([[INPUT]] : memref<1x256x16x32xf16>)
    // CHECK:              outputs([[COPYZERO]] : memref<1x256x32x64xf16, [@CMX_NN, 0]>)
    // CHECK:       [[COPYOUT:%.*]] = VPUIP.Copy
    // CHECK:              inputs([[DMA]] : memref<1x256x32x64xf16, [@CMX_NN, 0]>)
    // CHECK:              outputs([[OUTPUT]] : memref<1x256x32x64xf16>)

    // CHECK:       return [[COPYOUT]] : memref<1x256x32x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotMoveUpsamplingDMAInCMXWithLargeSize
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x64x128x128xf16>
// CHECK-SAME:    [[OUTPUT:%.+]]: memref<1x64x256x256xf16>
func.func @NotMoveUpsamplingDMAInCMXWithLargeSize(%arg0: memref<1x64x128x128xf16>, %arg1: memref<1x64x256x256xf16>) -> memref<1x64x256x256xf16> {
    %0 = VPUIP.Upsampling {
                pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
                upsampling_factor = [2, 2, 1]}
            inputs(%arg0 : memref<1x64x128x128xf16>)
            outputs(%arg1 : memref<1x64x256x256xf16>) -> memref<1x64x256x256xf16>
    return %0 : memref<1x64x256x256xf16>

    // CHECK:       [[CST:%.*]] = const.Declare memref<1x64x256x256xf16> = dense<0.000000e+00> : tensor<1x64x256x256xf16>

    // CHECK:       [[COPYZERO:%.*]] = VPUIP.Copy
    // CHECK:              inputs([[CST]] : memref<1x64x256x256xf16>)
    // CHECK:              outputs([[OUTPUT]] : memref<1x64x256x256xf16>)
    // CHECK:       [[DMA:%.*]] = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
    // CHECK:              inputs([[INPUT]] : memref<1x64x128x128xf16>)
    // CHECK:              outputs([[COPYZERO]] : memref<1x64x256x256xf16>)

    // CHECK:       return [[DMA]] : memref<1x64x256x256xf16>
}

// -----

IE.TileResource 3 of @NCE {
IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertPerAxisTileToDMAOutputDDR
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x1x1x2000xf16, @DDR>
// CHECK-SAME:    [[INPUT_1:%.+]]: memref<1x22x16x2000xf16, @DDR>
func.func @ConvertPerAxisTileToDMAOutputDDR(%arg0: memref<1x1x1x2000xf16, @DDR>, %arg1: memref<1x22x16x2000xf16, @DDR>) -> memref<1x22x16x2000xf16, @DDR> {
    %cmx_buf_0 = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x2000xf16, @DDR>) outputs(%cmx_buf_0 : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    %cmx_buf_1 = memref.alloc() : memref<1x22x16x2000xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
        inputs(%0 as %arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>)
        outputs(%cmx_buf_1 as %arg3: memref<1x22x16x2000xf16, [@CMX_NN, 0]>) on tile 0
        -> memref<1x22x16x2000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, [1, 22, 16, 1]]}(%arg2, %arg3) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x22x16x2000xf16, [@CMX_NN, 0]>
    }
    %1 = VPUIP.Copy inputs(%results : memref<1x22x16x2000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x22x16x2000xf16, @DDR>) -> memref<1x22x16x2000xf16, @DDR>
    return %1 : memref<1x22x16x2000xf16, @DDR>

    // CHECK:   [[CMX_BUF_0:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_IN:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x1x1x2000xf16, @DDR>) outputs([[CMX_BUF_0]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK:   [[DDR_BUF:%.+]] = memref.alloc() : memref<1x22x1x2000xf16, @DDR>
    // CHECK:   [[PER_AXIS_TILE_DMA_0:%.+]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, tiles = 22 : i64}
    // CHECK-SAME:    inputs([[COPY_IN]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[DDR_BUF]] : memref<1x22x1x2000xf16, @DDR>) -> memref<1x22x1x2000xf16, @DDR>
    // CHECK:   [[CMX_BUF_1:%.+]] = memref.alloc() : memref<1x22x16x2000xf16, [@CMX_NN, 0]>
    // CHECK:   [[PER_AXIS_TILE_DMA_1:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 16 : i64}
    // CHECK-SAME:    inputs([[PER_AXIS_TILE_DMA_0]] : memref<1x22x1x2000xf16, @DDR>)
    // CHECK-SAME:    outputs([[CMX_BUF_1]] : memref<1x22x16x2000xf16, [@CMX_NN, 0]>) -> memref<1x22x16x2000xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_OUT:%.+]] = VPUIP.Copy inputs([[PER_AXIS_TILE_DMA_1]] : memref<1x22x16x2000xf16, [@CMX_NN, 0]>) outputs([[INPUT_1]] : memref<1x22x16x2000xf16, @DDR>) -> memref<1x22x16x2000xf16, @DDR>

    // CHECK:   return [[COPY_OUT]] : memref<1x22x16x2000xf16, @DDR>
}
