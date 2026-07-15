//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --pass-pipeline="builtin.module(builtin.module(set-memory-space{memory-space=DDR set-memory-space-for-function-boundaries=false}))" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

//CHECK-LABEL: @NestedFunction
module @NestedFunction {
  //CHECK-LABEL: @Module0
  module @Module0  {
    //CHECK-LABEL: func.func @main_func0
    //CHECK-SAME: ([[SUBMAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
    func.func @main_func0(%arg0: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
      %alloc = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %alloc2 = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %0 = VPUIP.Copy inputs(%alloc2 : memref<1x89x1000x16xf16>) outputs(%alloc : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ALLOC1]] : memref<1x89x1000x16xf16, @DDR>) outputs([[ALLOC0]] : memref<1x89x1000x16xf16, @DDR>) -> memref<1x89x1000x16xf16, @DDR>
      %1 = VPUIP.Copy inputs(%0 : memref<1x89x1000x16xf16>) outputs(%arg0 : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x89x1000x16xf16, @DDR>) outputs([[SUBMAIN_ARG0]] : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      return %1 : memref<1x89x1000x16xf16>
      //CHECK: return [[COPY1]] : memref<1x89x1000x16xf16>
    }
  }

  //CHECK-LABEL: func.func @main
  //CHECK-SAME: ([[MAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) {
  func.func @main(%arg0 : memref<1x89x1000x16xf16>) {
    %0 = Core.NestedCall @Module0::@main_func0(%arg0) : (memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
    //CHECK: Core.NestedCall @Module0::@main_func0([[MAIN_ARG0]]) : (memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
    return
  }
}

// -----

// CHECK-LABEL: @TopModule
module @TopModule {
    // CHECK-LABEL: @NestedModule
    module @NestedModule {
        // CHECK-LABEL: func.func @StridedLayoutPreservedOnMemSpaceAssignment
        // CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>,
        // CHECK-SAME: [[OUTPUT:%.+]]: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
        func.func @StridedLayoutPreservedOnMemSpaceAssignment(
                %input: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>,
                %output: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>> {
            %buf = memref.alloc() : memref<1x16x48x640xf16>
            // CHECK: [[BUF:%.+]] = memref.alloc() : memref<1x16x48x640xf16, @DDR>
            %copy_to_buf = VPUIP.Copy inputs(%input: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                                      outputs(%buf : memref<1x16x48x640xf16>) -> memref<1x16x48x640xf16>
            // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
            // CHECK-SAME: outputs([[BUF]] : memref<1x16x48x640xf16, @DDR>) -> memref<1x16x48x640xf16, @DDR>

            %reinterp = Core.ReinterpretCast(%copy_to_buf) : memref<1x16x48x640xf16>
                      -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[REINTERP:%.+]] = Core.ReinterpretCast([[COPY0]]) : memref<1x16x48x640xf16, @DDR>
            // CHECK-SAME: -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            %last_copy = VPUIP.Copy inputs(%reinterp : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                                    outputs(%output : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                       -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[REINTERP]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>)
            // CHECK-SAME: outputs([[OUTPUT]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
            // CHECK-SAME: -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>

            return %last_copy : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[COPY1]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
        }
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0038725490663565842>

// CHECK-LABEL: @HostCompileBoundaryAliasesModule
module @HostCompileBoundaryAliasesModule {
    // CHECK-LABEL: @HostCompileBoundaryAliasesNestedModule
    module @HostCompileBoundaryAliasesNestedModule {
        // CHECK-LABEL: func.func @HostCompileBoundaryAliases
        // CHECK-SAME: ([[ARG0:%[^:]+]]: memref<1x16x16x3xui8>, [[ARG1:%[^:]+]]: memref<1x4x16x16x!qElemType>) -> memref<1x4x16x16x!qElemType>
        func.func @HostCompileBoundaryAliases(%arg0: memref<1x16x16x3xui8>, %arg1: memref<1x4x16x16x!qElemType>) -> memref<1x4x16x16x!qElemType> {
            %0 = Core.ReinterpretCast(%arg0) : memref<1x16x16x3xui8> -> memref<1x16x16x3xui8>
            %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}
                    inputs(%0 : memref<1x16x16x3xui8>) -> memref<1x3x16x16xui8, {order = #NHWC}>
            %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}>
            %alloc = memref.alloc() : memref<1x4x16x16x!qElemType, {order = #NHWC}>
            %3 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
                    inputs(%2 : memref<1x3x16x16x!qElemType, {order = #NHWC}>)
                    outputs(%alloc : memref<1x4x16x16x!qElemType, {order = #NHWC}>) -> memref<1x4x16x16x!qElemType, {order = #NHWC}>
            %4 = VPUIP.Copy inputs(%3 : memref<1x4x16x16x!qElemType, {order = #NHWC}>) outputs(%arg1 : memref<1x4x16x16x!qElemType>) -> memref<1x4x16x16x!qElemType>
            return %arg1 : memref<1x4x16x16x!qElemType>

            // CHECK: [[CAST:%[^ ]+]] = Core.ReinterpretCast([[ARG0]]) : memref<1x16x16x3xui8> -> memref<1x16x16x3xui8, @DDR>
            // CHECK: [[PERMUTE:%[^ ]+]] = VPUIP.PermuteCast
            // CHECK-SAME: inputs([[CAST]] : memref<1x16x16x3xui8, @DDR>) -> memref<1x3x16x16xui8, {order = #NHWC}, @DDR>
            // CHECK: [[QUANT:%[^ ]+]] = VPUIP.QuantizeCast inputs([[PERMUTE]] : memref<1x3x16x16xui8, {order = #NHWC}, @DDR>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}, @DDR>
            // CHECK: [[ALLOC:%[^ ]+]] = memref.alloc() : memref<1x4x16x16x!qElemType, {order = #NHWC}, @DDR>
            // CHECK: [[EXPAND:%[^ ]+]] = VPUIP.Expand
            // CHECK-SAME: inputs([[QUANT]] : memref<1x3x16x16x!qElemType, {order = #NHWC}, @DDR>)
            // CHECK-SAME: outputs([[ALLOC]] : memref<1x4x16x16x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x4x16x16x!qElemType, {order = #NHWC}, @DDR>
            // CHECK: return [[ARG1]] : memref<1x4x16x16x!qElemType>
        }
    }
}

// -----

// In HostCompile mode the outlined kernel function has device-info-free block arguments
// (the function_type must stay consistent with the call site). When a
// VPUIP.GroupBoundedBuffer has a data operand that carries @DDR (here from
// updateFunctionBoundaryAliases - the data arg is accessed via a Core.ReinterpretCast
// pure-view alias) and a shape operand that is a device-info-free block arg,
// SetMemorySpace must NOT mutate the block arg type. Instead it inserts a
// Core.ReinterpretCast at the GroupBoundedBuffer use site so that all operand types
// are consistent while the function signature stays unchanged.

// CHECK-LABEL: @HostCompileDeviceInfoFreeArgReinterpretCastData
module @HostCompileDeviceInfoFreeArgReinterpretCastData attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
    // CHECK-LABEL: @Module0
    module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
        // The function signature must remain unchanged: both block args stay device-info-free
        // CHECK-LABEL: func.func @main_func0
        // CHECK-SAME: ([[ARG_DATA:%.+]]: memref<1x3x8x8xf16, strided<[?, ?, ?, ?], offset: ?>>,
        // CHECK-SAME:  [[ARG_SHAPE:%.+]]: memref<4xsi32>)
        func.func @main_func0(
            %arg_data: memref<1x3x8x8xf16, strided<[?, ?, ?, ?], offset: ?>>,
            %arg_shape: memref<4xsi32>
        ) -> memref<4xsi32> {
            // Pure-view alias of %arg_data -> gets @DDR via updateFunctionBoundaryAliases
            %data_cast = Core.ReinterpretCast(%arg_data) : memref<1x3x8x8xf16, strided<[?, ?, ?, ?], offset: ?>> -> memref<1x3x8x8xf16>
            // CHECK: [[DATA_CAST:%.+]] = Core.ReinterpretCast([[ARG_DATA]])
            // CHECK-SAME: -> memref<1x3x8x8xf16, @DDR>

            // groupOpCallback fires: data has @DDR, shape is a device-info-free block arg
            // Fix: Core.ReinterpretCast([[ARG_SHAPE]]) inserted; block arg type unchanged
            // CHECK: [[SHAPE_CAST:%.+]] = Core.ReinterpretCast([[ARG_SHAPE]])
            // CHECK-SAME: memref<4xsi32> -> memref<4xsi32, @DDR>
            %grouped = VPUIP.GroupBoundedBuffer(%data_cast, %arg_shape)
                    : memref<1x3x8x8xf16>, memref<4xsi32>
                    -> !VPUIP.BoundedBuffer<data=memref<1x3x8x8xf16>, dynamic_shape=memref<4xsi32>>
            // CHECK: VPUIP.GroupBoundedBuffer([[DATA_CAST]], [[SHAPE_CAST]])
            // CHECK-SAME: !VPUIP.BoundedBuffer<data=memref<1x3x8x8xf16, @DDR>, dynamic_shape=memref<4xsi32, @DDR>>

            // %arg_shape is returned as-is: the function signature is unchanged
            // CHECK: return [[ARG_SHAPE]] : memref<4xsi32>
            return %arg_shape : memref<4xsi32>
        }
    }
}
