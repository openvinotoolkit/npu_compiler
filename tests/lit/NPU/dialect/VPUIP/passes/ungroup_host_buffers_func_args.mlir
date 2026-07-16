//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --ungroup-host-buffers-as-func-args --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: module @simple
// CHECK:       module @Module0
// CHECK:       net.NetworkInfo entryPoint : @main_func0 inputsInfo
// CHECK:         DataInfo "in_0" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
// CHECK:         DataInfo "vpux_ie_shape_in_0" : tensor<3xsi32>
// CHECK:       } outputsInfo : {
// CHECK:         DataInfo "out_0" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
// CHECK:         DataInfo "vpux_ie_shape_out_0" : tensor<3xsi32>
module @simple attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_13" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
  } outputsInfo : {
    DataInfo "Softmax_14" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
  }

  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
      func.func nested @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.kernel_name = "softmax", VPU.task_type = @COMPUTE}
      func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }
    net.NetworkInfo entryPoint : @main_func0 inputsInfo : {
      DataInfo "in_0" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
    } outputsInfo : {
      DataInfo "out_0" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
    }

    // CHECK:       func.func nested @main_func0([[IN:%.+]]: memref<?x1x16xf16>, [[IN_SHAPE:%.+]]: memref<3xsi32>) -> (memref<?x1x16xf16>, memref<3xsi32>)
    // CHECK:         [[IN_DATA:%.+]] = Core.ReinterpretCast([[IN]]) : memref<?x1x16xf16> -> memref<8x1x16xf16>
    // CHECK:         [[IN_BB:%.+]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])
    // CHECK:         [[OUT_DATA:%.+]], [[OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer
    // CHECK:         [[OUT:%.+]] = Core.ReinterpretCast([[OUT_DATA]]) : memref<8x1x16xf16> -> memref<?x1x16xf16>
    // CHECK:         return [[OUT]], [[OUT_SHAPE]] : memref<?x1x16xf16>, memref<3xsi32>
    func.func nested @main_func0(%main: memref<?x1x16xf16>) -> memref<?x1x16xf16> {
      // Input hidden boundary: dynamic arg -> static cast -> GroupBoundedBuffer
      %main_data = Core.ReinterpretCast(%main) : memref<?x1x16xf16> -> memref<8x1x16xf16>
      %main_shape = memref.alloc() : memref<3xsi32>
      %in_bounded = VPUIP.GroupBoundedBuffer(%main_data, %main_shape) : memref<8x1x16xf16>, memref<3xsi32> -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>

      // DDR -> CMX copy
      %cmx_in_data = memref.alloc() : memref<8x1x16xf16, [@CMX_NN, 0]>
      %cmx_in_shape = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
      %cmx_in_bounded = VPUIP.GroupBoundedBuffer(%cmx_in_data, %cmx_in_shape) : memref<8x1x16xf16, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
      %copy_in = VPUIP.Copy inputs(%in_bounded : !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>) outputs(%cmx_in_bounded : !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>

      // Compute: SoftMax on CMX
      %cmx_out_data = memref.alloc() : memref<8x1x16xf16, [@CMX_NN, 0]>
      %cmx_out_shape = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
      %cmx_out_bounded = VPUIP.GroupBoundedBuffer(%cmx_out_data, %cmx_out_shape) : memref<8x1x16xf16, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
      %compute = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
          inputs(%copy_in as %softmax_in: !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>)
          outputs(%cmx_out_bounded as %softmax_out: !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>)
          on tile 0 -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>> {
        VPUIP.SW.Kernel.run {attrs = [2, 0]}(%softmax_in, %softmax_out)
            : !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>,
              !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
      }

      // CMX -> DDR copy
      %ddr_out_data = memref.alloc() : memref<8x1x16xf16>
      %ddr_out_shape = memref.alloc() : memref<3xsi32>
      %ddr_out_bounded = VPUIP.GroupBoundedBuffer(%ddr_out_data, %ddr_out_shape) : memref<8x1x16xf16>, memref<3xsi32> -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>
      %copy_out = VPUIP.Copy inputs(%compute : !VPUIP.BoundedBuffer<data=memref<8x1x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) outputs(%ddr_out_bounded : !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>

      // Output hidden boundary: UngroupBoundedBuffer -> static -> dynamic cast
      %out_data, %out_shape = VPUIP.UngroupBoundedBuffer(%copy_out) : !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>> -> memref<8x1x16xf16>, memref<3xsi32>
      %ret = Core.ReinterpretCast(%out_data) : memref<8x1x16xf16> -> memref<?x1x16xf16>
      return %ret : memref<?x1x16xf16>
    }
  }

  // CHECK:       func.func @main([[MAIN_IN:%.+]]: memref<?x1x16xf16>) -> memref<?x1x16xf16>
  // CHECK:         [[TMP_SHAPE:%.+]] = memref.alloc() : memref<3xsi32>
  // CHECK:         [[CALL_RES:%.+]]:2 = Core.NestedCall @Module0::@main_func0([[MAIN_IN]], [[TMP_SHAPE]]) : (memref<?x1x16xf16>, memref<3xsi32>) -> (memref<?x1x16xf16>, memref<3xsi32>)
  // CHECK:         return [[CALL_RES]]#0 : memref<?x1x16xf16>
  func.func @main(%arg: memref<?x1x16xf16>) -> memref<?x1x16xf16> {
    %res = Core.NestedCall @Module0::@main_func0(%arg) : (memref<?x1x16xf16>) -> memref<?x1x16xf16>
    return %res : memref<?x1x16xf16>
  }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @output_cast_chain
// CHECK:       module @Module0
// CHECK:       net.NetworkInfo entryPoint : @main_func0 inputsInfo
// CHECK:         DataInfo "in_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:         DataInfo "vpux_ie_shape_in_0" : tensor<4xsi32>
// CHECK:       } outputsInfo : {
// CHECK:         DataInfo "out_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:         DataInfo "vpux_ie_shape_out_0" : tensor<4xsi32>
module @output_cast_chain attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "Result_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
  }

  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
    net.NetworkInfo entryPoint : @main_func0 inputsInfo : {
      DataInfo "in_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
    }

    // CHECK:       func.func nested @main_func0([[IN:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, [[IN_SHAPE:%.+]]: memref<4xsi32>) -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>) {
    // CHECK:         [[IN_DATA:%.+]] = Core.ReinterpretCast([[IN]]) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> -> memref<1x3x50x50xf16>
    // CHECK:         [[IN_BB:%.+]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])
    // CHECK:         [[OUT_DATA:%.+]], [[OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[IN_BB]])
    // CHECK:         [[OUT_DYN0:%.+]] = Core.ReinterpretCast([[OUT_DATA]]) : memref<1x3x50x50xf16> -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    // CHECK:         return [[OUT_DYN0]], [[OUT_SHAPE]] : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>
    func.func nested @main_func0(%main: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
      // Input hidden boundary: two chained ReinterpretCasts (strided -> dynamic -> static) mirror the
      // real model IR; canonicalization folds the chained ReinterpretCasts into a single ReinterpretCast before UngroupHostBuffersAsFuncArgs runs
      %main_1 = Core.ReinterpretCast(%main) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> -> memref<1x3x?x?xf16>
      %main_data = Core.ReinterpretCast(%main_1) : memref<1x3x?x?xf16> -> memref<1x3x50x50xf16>
      %main_shape = memref.alloc() : memref<4xsi32>
      %in_bounded = VPUIP.GroupBoundedBuffer(%main_data, %main_shape) : memref<1x3x50x50xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x3x50x50xf16>, dynamic_shape=memref<4xsi32>>

      // Output hidden boundary with casts at the end of the reinterpret chain
      %out_data, %out_shape = VPUIP.UngroupBoundedBuffer(%in_bounded) : !VPUIP.BoundedBuffer<data=memref<1x3x50x50xf16>, dynamic_shape=memref<4xsi32>> -> memref<1x3x50x50xf16>, memref<4xsi32>
      %out_dyn = Core.ReinterpretCast(%out_data) : memref<1x3x50x50xf16> -> memref<1x3x?x?xf16>
      %out_dyn_alias = Core.ReinterpretCast(%out_dyn) : memref<1x3x?x?xf16> -> memref<1x3x?x?xf16>
      %ret = memref.cast %out_dyn_alias : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
      return %ret : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    }
  }

  // CHECK:       func.func @main([[MAIN_IN:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
  // CHECK:         [[TMP_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>
  // CHECK:         [[CALL_RES:%.+]]:2 = Core.NestedCall @Module0::@main_func0([[MAIN_IN]], [[TMP_SHAPE]]) : (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>) -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>)
  // CHECK:         return [[CALL_RES]]#0 : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
  func.func @main(%arg: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
    %res = Core.NestedCall @Module0::@main_func0(%arg) : (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    return %res : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
  }
}
