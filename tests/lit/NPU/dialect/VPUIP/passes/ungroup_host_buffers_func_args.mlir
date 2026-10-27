//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --ungroup-bounded-buffers-as-func-args --canonicalize %s | FileCheck %s
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
      func.func nested @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.kernel_name = "softmax", VPU.task_type = @COMPUTE}
      func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }
    net.NetworkInfo entryPoint : @main_func0 inputsInfo : {
      DataInfo "in_0" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
    } outputsInfo : {
      DataInfo "out_0" : tensor<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
    }

    // CHECK:       func.func @main_func0([[IN:%.+]]: memref<?x1x16xf16>, [[IN_SHAPE:%.+]]: memref<3xsi32>) -> (memref<?x1x16xf16>, memref<3xsi32>)
    // CHECK:         [[IN_DATA:%.+]] = Core.ReinterpretCast([[IN]]) : memref<?x1x16xf16> -> memref<8x1x16xf16>
    // CHECK:         [[IN_BB:%.+]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])
    // CHECK:         [[OUT_DATA:%.+]], [[OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer
    // CHECK:         [[OUT:%.+]] = Core.ReinterpretCast([[OUT_DATA]]) : memref<8x1x16xf16> -> memref<?x1x16xf16>
    // CHECK:         return [[OUT]], [[OUT_SHAPE]] : memref<?x1x16xf16>, memref<3xsi32>
    func.func @main_func0(%main: memref<?x1x16xf16>) -> memref<?x1x16xf16> {
      %main_data = Core.ReinterpretCast(%main) : memref<?x1x16xf16>
        -> memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>

      // DDR -> CMX copy
      %c8 = arith.constant 8 : index
      %cmx_in_data = memref.alloc(%c8) : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>
      %copy_in = VPUIP.Copy
        inputs(%main_data : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>)
        outputs(%cmx_in_data : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>)
        -> memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>

      // Compute: SoftMax on CMX
      %cmx_out_data = memref.alloc() : memref<8x1x16xf16, [@CMX_NN, 0]>
      %cmx_out_shape = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
      %cmx_out_bounded = memref.alloc(%c8) : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>
      %compute = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
          inputs(%copy_in as %softmax_in: memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>)
          outputs(%cmx_out_bounded as %softmax_out: memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>)
          on tile 0 -> memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]> {
        VPUIP.SW.Kernel.run {attrs = [2, 0]}(%softmax_in, %softmax_out)
            : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>,
              memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>
      }

      // CMX -> DDR copy
      %ddr_out_data = memref.alloc(%c8) : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
      %copy_out = VPUIP.Copy
        inputs(%compute : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>)
        outputs(%ddr_out_data : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>)
        -> memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>

      %ret = Core.ReinterpretCast(%copy_out) : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
        -> memref<?x1x16xf16>
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

    // CHECK:       func.func @main_func0([[IN:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, [[IN_SHAPE:%.+]]: memref<4xsi32>) -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>) {
    // CHECK:         [[IN_DATA:%.+]] = Core.ReinterpretCast([[IN]]) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> -> memref<1x3x50x50xf16>
    // CHECK:         [[IN_BB:%.+]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])
    // CHECK:         [[COPY:%.+]] = VPUIP.Copy inputs([[IN_BB]]
    // CHECK:         [[OUT_DATA:%.+]], [[OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[COPY]])
    // CHECK:         [[OUT_DYN0:%.+]] = Core.ReinterpretCast([[OUT_DATA]]) : memref<1x3x50x50xf16> -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    // CHECK:         return [[OUT_DYN0]], [[OUT_SHAPE]] : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>
    func.func @main_func0(%main: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
      %in_dyn = Core.ReinterpretCast(%main) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> -> memref<1x3x?x?xf16>
      %in_bounded = Core.ReinterpretCast(%in_dyn) : memref<1x3x?x?xf16>
        -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>

      %c50 = arith.constant 50 : index
      %copy_alloc = memref.alloc(%c50, %c50) : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
      %copy = VPUIP.Copy
        inputs(%in_bounded : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>)
        outputs(%copy_alloc : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>)
        -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>

      %out_dyn = Core.ReinterpretCast(%copy) : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
        -> memref<1x3x?x?xf16>
      %ret = Core.ReinterpretCast(%out_dyn) : memref<1x3x?x?xf16> -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistBuff = !VPUIP.DistributedBuffer<1x1x1x1325593xui8, #NCHW, @CMX_NN,
  {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
   compute_shapes = [[1, 1, 1, 1325593], [1, 1, 1, 1325593], [1, 1, 1, 1325593], [1, 1, 1, 1325593]],
   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
   memory_shapes = [[1, 1, 1, 1325593], [1, 1, 1, 1325593], [1, 1, 1, 1325593], [1, 1, 1, 1325593]],
   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>


// Note: this is a test case that mainly focuses on making sure that multiple
// kernels are handled correctly by the pass

// CHECK-LABEL: module @multiple_kernels
module @multiple_kernels attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {

    // CHECK-LABEL: module @Module0
    module @Module0 {
        net.NetworkInfo entryPoint : @main_func0 inputsInfo : {
            DataInfo "in_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "in_1" : tensor<2xf32>
        } outputsInfo : {
            DataInfo "out_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>
        }
        // CHECK: net.NetworkInfo entryPoint : @main_func0 inputsInfo : {
        // CHECK-NEXT:  tensor<1x3x?x?xf16
        // CHECK-NEXT:  tensor<2xf32>
        // CHECK-NEXT:  tensor<4xsi32>
        // CHECK-NEXT: } outputsInfo : {
        // CHECK-NEXT:  tensor<1x3x?x?xf16
        // CHECK-NEXT:  tensor<4xsi32>
        // CHECK-NEXT: }

        // CHECK: func.func @main_func0([[IN0:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, {{%.+}}: memref<2xf32, strided<[?], offset: ?>>,
        // CHECK-SAME:  [[SHAPE:%.+]]: memref<4xsi32>)
        // CHECK-SAME: -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>)
        func.func @main_func0(%arg0: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<2xf32, strided<[?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
            %arg0_bounded = Core.ReinterpretCast(%arg0) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
                -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>
            // CHECK: [[IN_CAST:%.+]] = Core.ReinterpretCast([[IN0]])
            // CHECK-SAME: -> memref<1x3x368x432xf16>

            // CHECK: [[IN_BB:%.+]] = VPUIP.GroupBoundedBuffer([[IN_CAST]], [[SHAPE]])

            %dist_buff = VPURT.AllocDistributed -> !DistBuff
            %c2658 = arith.constant 2658 : index
            %c3120 = arith.constant 3120 : index
            %out_alloc = memref.alloc(%c2658, %c3120)
                : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>
            %sw_res:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
                @VPU.SW::@builtin_InterpolateDMA
                inputs(%arg0_bounded as %in0: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>,
                    %arg1 as %in1: memref<2xf32, strided<[?], offset: ?>>,
                    %dist_buff as %in2: !DistBuff)
                outputs(%out_alloc as %out0: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>,
                        %dist_buff as %out1: !DistBuff)
                on tile 0 -> (memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>, !DistBuff)
            {
                VPUIP.SW.Kernel.run {attrs = [1, 0, 2, 0, [2, 3], -7.500000e-01, 0.000000e+00, 8589934596]}
                (%in0, %in1, %in2, %out0, %out1)
                    : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>,
                    memref<2xf32, strided<[?], offset: ?>>,
                    !DistBuff,
                    memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>,
                    !DistBuff
            }

            // CHECK: [[SW_RES:%.+]]:2 = VPUIP.SW.Kernel
            // CHECK-SAME: @VPU.SW::@builtin_InterpolateDMA
            // CHECK-SAME: inputs([[IN_BB]]

            // CHECK: [[OUT_DATA:%.+]], [[OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[SW_RES]]#0)

            %res_dyn = Core.ReinterpretCast(%sw_res#0) : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>
                -> memref<1x3x?x?xf16>
            %res = Core.ReinterpretCast(%res_dyn) : memref<1x3x?x?xf16> -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[OUT_CAST:%.+]] = Core.ReinterpretCast([[OUT_DATA]])
            // CHECK: -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

            return %res : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[OUT_CAST]], [[OUT_SHAPE]]
        }
    }

    // CHECK-LABEL: module @Module1
    module @Module1 {
        net.NetworkInfo entryPoint : @main_func1 inputsInfo : {
            DataInfo "in_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "in_1" : tensor<2xf32>
        } outputsInfo : {
            DataInfo "out_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>
        }
        // CHECK: net.NetworkInfo entryPoint : @main_func1 inputsInfo : {
        // CHECK-NEXT:  tensor<1x3x?x?xf16
        // CHECK-NEXT:  tensor<2xf32>
        // CHECK-NEXT:  tensor<4xsi32>
        // CHECK-NEXT: } outputsInfo : {
        // CHECK-NEXT:  tensor<1x3x?x?xf16
        // CHECK-NEXT:  tensor<4xsi32>
        // CHECK-NEXT: }

        // CHECK: func.func @main_func1([[IN0:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, {{%.+}}: memref<2xf32, strided<[?], offset: ?>>,
        // CHECK-SAME:  [[SHAPE:%.+]]: memref<4xsi32>)
        // CHECK-SAME: -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>)
        func.func @main_func1(%arg0: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<2xf32, strided<[?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
            %arg0_bounded = Core.ReinterpretCast(%arg0) : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
                -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>
            // CHECK: [[IN_CAST:%.+]] = Core.ReinterpretCast([[IN0]])
            // CHECK-SAME: -> memref<1x3x368x432xf16>

            // CHECK: [[IN_BB:%.+]] = VPUIP.GroupBoundedBuffer([[IN_CAST]], [[SHAPE]])

            %dist_buff = VPURT.AllocDistributed -> !DistBuff
            %c2658 = arith.constant 2658 : index
            %c3120 = arith.constant 3120 : index
            %out_alloc = memref.alloc(%c2658, %c3120)
                : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>
            %sw_res:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
                @VPU.SW::@builtin_InterpolateDMA
                inputs(%arg0_bounded as %in0: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>,
                    %arg1 as %in1: memref<2xf32, strided<[?], offset: ?>>,
                    %dist_buff as %in2: !DistBuff)
                outputs(%out_alloc as %out0: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>,
                        %dist_buff as %out1: !DistBuff)
                on tile 0 -> (memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>, !DistBuff)
            {
                VPUIP.SW.Kernel.run {attrs = [1, 0, 2, 0, [2, 3], -7.500000e-01, 0.000000e+00, 8589934596]}
                (%in0, %in1, %in2, %out0, %out1)
                    : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 368, 432]> : tensor<4xsi64>, order = #NCHW}>,
                    memref<2xf32, strided<[?], offset: ?>>,
                    !DistBuff,
                    memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>,
                    !DistBuff
            }

            // CHECK: [[SW_RES:%.+]]:2 = VPUIP.SW.Kernel
            // CHECK-SAME: @VPU.SW::@builtin_InterpolateDMA
            // CHECK-SAME: inputs([[IN_BB]]

            // CHECK: [[OUT_DATA:%.+]], [[OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[SW_RES]]#0)

            %res_dyn = Core.ReinterpretCast(%sw_res#0) : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2658, 3120]> : tensor<4xsi64>, order = #NCHW}>
                -> memref<1x3x?x?xf16>
            %res = Core.ReinterpretCast(%res_dyn) : memref<1x3x?x?xf16> -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[OUT_CAST:%.+]] = Core.ReinterpretCast([[OUT_DATA]])
            // CHECK: -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

            return %res : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[OUT_CAST]], [[OUT_SHAPE]]
        }
    }

    // CHECK: func.func @main
    // CHECK-SAME: ([[IN0:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, [[IN1:%.+]]: memref<2xf32, strided<[?], offset: ?>>
    // CHECK-SAME:  -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    func.func @main(%arg0: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>,
                    %arg1: memref<2xf32, strided<[?], offset: ?>>, %cond: i1)
            -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
        %res = scf.if %cond -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
            %call = Core.NestedCall @Module0::@main_func0(%arg0, %arg1)
                : (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<2xf32, strided<[?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[SHAPE_ALLOC0:%.+]] = memref.alloc() : memref<4xsi32>
            // CHECK: {{%.+}} = Core.NestedCall @Module0::@main_func0([[IN0]], [[IN1]], [[SHAPE_ALLOC0]])
            scf.yield %call : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        } else {
            %call = Core.NestedCall @Module1::@main_func1(%arg0, %arg1)
                : (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<2xf32, strided<[?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[SHAPE_ALLOC1:%.+]] = memref.alloc() : memref<4xsi32>
            // CHECK: {{%.+}} = Core.NestedCall @Module1::@main_func1([[IN0]], [[IN1]], [[SHAPE_ALLOC1]])
            scf.yield %call : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        }
        return %res : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    }
}
