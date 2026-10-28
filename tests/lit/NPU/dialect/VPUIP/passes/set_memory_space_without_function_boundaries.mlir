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
    net.NetworkInfo entryPoint : @main_func0 inputsInfo :  {
        DataInfo "input" : tensor<1x89x1000x16xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x89x1000x16xf16>
    }

    //CHECK-LABEL: func.func @main_func0
    //CHECK-SAME: ([[SUBMAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
    func.func @main_func0(%arg0: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
      // CHECK: [[ARG0_CAST:%.+]] = Core.ReinterpretCast([[SUBMAIN_ARG0]]) {{.*}} -> memref<1x89x1000x16xf16, @DDR>
      %alloc = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %alloc2 = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %0 = VPUIP.Copy inputs(%alloc2 : memref<1x89x1000x16xf16>) outputs(%alloc : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ALLOC1]] : memref<1x89x1000x16xf16, @DDR>) outputs([[ALLOC0]] : memref<1x89x1000x16xf16, @DDR>) -> memref<1x89x1000x16xf16, @DDR>
      %1 = VPUIP.Copy inputs(%0 : memref<1x89x1000x16xf16>) outputs(%arg0 : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x89x1000x16xf16, @DDR>) outputs([[ARG0_CAST]] : memref<1x89x1000x16xf16, @DDR>)
      // CHECK-SAME:    -> memref<1x89x1000x16xf16, @DDR>
      return %1 : memref<1x89x1000x16xf16>
      // CHECK: [[RES_CAST:%.+]] = Core.ReinterpretCast([[COPY1]]) {{.*}} -> memref<1x89x1000x16xf16>
      //CHECK: return [[RES_CAST]]
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
        net.NetworkInfo entryPoint : @StridedLayoutPreservedOnMemSpaceAssignment inputsInfo :  {
            DataInfo "input" : tensor<1x16x48x640xf16>
        } outputsInfo :  {
            DataInfo "output" : tensor<1x16x48x640xf16>
        }

        // CHECK-LABEL: func.func @StridedLayoutPreservedOnMemSpaceAssignment
        // CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>,
        // CHECK-SAME: [[OUTPUT:%.+]]: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
        func.func @StridedLayoutPreservedOnMemSpaceAssignment(
                %input: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>,
                %output: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>> {
            // CHECK: [[IN_CAST:%.+]] = Core.ReinterpretCast([[INPUT]]) {{.*}} -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            // CHECK: [[OUT_CAST:%.+]] = Core.ReinterpretCast([[OUTPUT]]) {{.*}} -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            %buf = memref.alloc() : memref<1x16x48x640xf16>
            // CHECK: [[BUF:%.+]] = memref.alloc() : memref<1x16x48x640xf16, @DDR>
            %copy_to_buf = VPUIP.Copy inputs(%input: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                                      outputs(%buf : memref<1x16x48x640xf16>) -> memref<1x16x48x640xf16>
            // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[IN_CAST]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>)
            // CHECK-SAME: outputs([[BUF]] : memref<1x16x48x640xf16, @DDR>) -> memref<1x16x48x640xf16, @DDR>

            %reinterp = Core.ReinterpretCast(%copy_to_buf) : memref<1x16x48x640xf16>
                      -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[REINTERP:%.+]] = Core.ReinterpretCast([[COPY0]]) : memref<1x16x48x640xf16, @DDR>
            // CHECK-SAME: -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            %last_copy = VPUIP.Copy inputs(%reinterp : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                                    outputs(%output : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                       -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[REINTERP]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>)
            // CHECK-SAME: outputs([[OUT_CAST]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>)
            // CHECK-SAME: -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>

            return %last_copy : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[RES_CAST:%.+]] = Core.ReinterpretCast([[COPY1]]) {{.*}} -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[RES_CAST]]
        }
    }
}

// -----

// In HostCompile mode, the NPU kernel signature must be free from any
// NPU-specific details, including the memory space information. Thus, in such a
// case, an alternative approach to memory space assignment is used, that relies on
// introducing the Core.ReinterpreetCast operations around input/output arguments
// and return types of the NPU kernel.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0038725490663565842>

// CHECK-LABEL: @HostCompileBoundaryAliasesModule
module @HostCompileBoundaryAliasesModule {
    // CHECK-LABEL: @HostCompileBoundaryAliasesNestedModule
    module @HostCompileBoundaryAliasesNestedModule {
        net.NetworkInfo entryPoint : @HostCompileBoundaryAliases inputsInfo :  {
            DataInfo "input" : tensor<1x16x16x3xui8>
        } outputsInfo :  {
            DataInfo "output" : tensor<1x4x16x16xui8>
        }

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
            return %4 : memref<1x4x16x16x!qElemType>

            // CHECK: [[ARG0_CAST:%[^ ]+]] = Core.ReinterpretCast([[ARG0]]) {{.*}} -> memref<1x16x16x3xui8, @DDR>
            // CHECK: [[ARG1_CAST:%[^ ]+]] = Core.ReinterpretCast([[ARG1]]) {{.*}} -> memref<1x4x16x16x!qElemType, @DDR>
            // CHECK: [[DUMMY_CAST:%[^ ]+]] = Core.ReinterpretCast([[ARG0_CAST]]) {{.*}} -> memref<1x16x16x3xui8, @DDR>
            // CHECK: [[PERMUTE:%[^ ]+]] = VPUIP.PermuteCast
            // CHECK-SAME: inputs([[DUMMY_CAST]] : memref<1x16x16x3xui8, @DDR>) -> memref<1x3x16x16xui8, {order = #NHWC}, @DDR>
            // CHECK: [[QUANT:%[^ ]+]] = VPUIP.QuantizeCast inputs([[PERMUTE]] : memref<1x3x16x16xui8, {order = #NHWC}, @DDR>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}, @DDR>
            // CHECK: [[ALLOC:%[^ ]+]] = memref.alloc() : memref<1x4x16x16x!qElemType, {order = #NHWC}, @DDR>
            // CHECK: [[EXPAND:%[^ ]+]] = VPUIP.Expand
            // CHECK-SAME: inputs([[QUANT]] : memref<1x3x16x16x!qElemType, {order = #NHWC}, @DDR>)
            // CHECK-SAME: outputs([[ALLOC]] : memref<1x4x16x16x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x4x16x16x!qElemType, {order = #NHWC}, @DDR>
            // CHECK: [[COPY:%[^ ]+]] = VPUIP.Copy inputs([[EXPAND]] {{.*}} outputs([[ARG1_CAST]] {{.*}}
            // CHECK: [[RES_CAST:%[^ ]+]] = Core.ReinterpretCast([[COPY]]) {{.*}} -> memref<1x4x16x16x!qElemType>
            // CHECK: return [[RES_CAST]]
        }
    }
}
// -----

// Note: This is a fairly dumb example, but it would suffice for a test: only
// an entry point "public" function is expected to preserve the boundaries,
// while inner functions are free to have memory space at the signature level.


// CHECK-LABEL: @HostCompileWithNpuKernelAndInnerFunctionCall
module @HostCompileWithNpuKernelAndInnerFunctionCall {
    // CHECK-LABEL: @NpuModule
    module @NpuModule {
        net.NetworkInfo entryPoint : @main_func0 inputsInfo :  {
            DataInfo "input" : tensor<1x3x?x?xf16>
        } outputsInfo :  {
            DataInfo "output" : tensor<1x3x?x?xf16>
        }

        // CHECK: func.func nested @main_part([[IN:%.+]]: memref<1x3x368x432xf16, @DDR>, [[OUT:%.+]]: memref<1x3x368x432xf16, @DDR>)
        // CHECK-SAME:  -> memref<1x3x368x432xf16, @DDR>
        func.func nested @main_part(%in: memref<1x3x368x432xf16>, %out: memref<1x3x368x432xf16>)
                -> memref<1x3x368x432xf16> {
            %alloc = memref.alloc() : memref<1x3x368x432xf16, @CMX_NN>
            %cmx_copy = VPUIP.Copy inputs(%in: memref<1x3x368x432xf16>)
                outputs(%alloc : memref<1x3x368x432xf16, @CMX_NN>)
                -> memref<1x3x368x432xf16, @CMX_NN>
            %ddr_copy = VPUIP.Copy inputs(%cmx_copy: memref<1x3x368x432xf16, @CMX_NN>)
                outputs(%out : memref<1x3x368x432xf16>)
                -> memref<1x3x368x432xf16>
            return %ddr_copy : memref<1x3x368x432xf16>
        }

        // CHECK: func.func @main_func0([[IN:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>,
        // CHECK-SAME: [[OUT:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
        // CHECK-SAME:  -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        func.func @main_func0(%in: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>,
                              %out: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
            // CHECK: [[IN_CAST:%.+]] = Core.ReinterpretCast([[IN]]) {{.*}} -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            // CHECK: [[OUT_CAST:%.+]] = Core.ReinterpretCast([[OUT]]) {{.*}} -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>

            %in_static = Core.ReinterpretCast(%in) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
                -> memref<1x3x368x432xf16>
            %out_static = Core.ReinterpretCast(%out) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
                -> memref<1x3x368x432xf16>
            // CHECK: [[IN_STATIC:%.+]] = Core.ReinterpretCast([[IN_CAST]]) {{.*}} -> memref<1x3x368x432xf16, @DDR>
            // CHECK: [[OUT_STATIC:%.+]] = Core.ReinterpretCast([[OUT_CAST]]) {{.*}} -> memref<1x3x368x432xf16, @DDR>

            %call = func.call @main_part(%in_static, %out_static) : (memref<1x3x368x432xf16>, memref<1x3x368x432xf16>)
                -> memref<1x3x368x432xf16>
            // CHECK: [[CALL:%.+]] = call @main_part([[IN_STATIC]], [[OUT_STATIC]])
            // CHECK-SAME: : (memref<1x3x368x432xf16, @DDR>, memref<1x3x368x432xf16, @DDR>)
            // CHECK-SAME: -> memref<1x3x368x432xf16, @DDR>

            %call_dynamic = Core.ReinterpretCast(%call) : memref<1x3x368x432xf16>
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            return %call_dynamic : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[CALL_DYNAMIC:%.+]] = Core.ReinterpretCast([[CALL]]) {{.*}} -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            // CHECK: [[RES_CAST:%.+]] = Core.ReinterpretCast([[CALL_DYNAMIC]]) {{.*}} -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[RES_CAST]]
        }
    }
}
