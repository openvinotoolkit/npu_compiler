//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --ungroup-bounded-buffers --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @TestCopy attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  // CHECK-LABEL: main
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
    // CHECK: DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
    // CHECK: DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
  }

  // CHECK-LABEL: main
  func.func @main(%arg0: memref<2x4x20x20xf16>, %arg1: memref<4xsi32>, %arg2: memref<2x4x20x20xf16>, %arg3: memref<4xsi32>) -> (memref<2x4x20x20xf16>, memref<4xsi32>) {
    // CHECK-SAME: [[IN_DATA:%.+]]: memref<2x4x20x20xf16>, [[IN_SHAPE:%.+]]: memref<4xsi32>,
    // CHECK-SAME: [[OUT_DATA:%.+]]: memref<2x4x20x20xf16>, [[OUT_SHAPE:%.+]]: memref<4xsi32>

    %DATA = memref.alloc() : memref<2x4x20x20xf16>
    %SHAPE = memref.alloc() : memref<4xsi32>
    // CHECK: [[DATA:%.*]] = memref.alloc
    // CHECK: [[SHAPE:%.*]] = memref.alloc

    %IN_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%arg0, %arg1) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECKA: [[IN_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])
    %BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%DATA, %SHAPE) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECKA: [[BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[DATA]], [[SHAPE]])

    %COPY = VPUIP.Copy inputs(%IN_BOUNDED_BUFFER: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                       outputs (%BOUNDED_BUFFER: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                       -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECKA: [[DATA1:%.*]], [[SHAPE1:%.*]] = VPUIP.UngroupBoundedBuffer([[IN_BOUNDED_BUFFER]])
    // CHECKA: [[DATA2:%.*]], [[SHAPE2:%.*]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFFER]])
    // CHECK: [[DATA_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_DATA]]
    // CHECK-SAME: outputs([[DATA]]
    // CHECK: [[SHAPE_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_SHAPE]]
    // CHECK-SAME: outputs([[SHAPE]]
    // CHECKA: [[COPY_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[DATA_COPY]], [[SHAPE_COPY]])

    %OUT_DATA, %OUT_SHAPE = VPUIP.UngroupBoundedBuffer(%COPY) :
        !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<2x4x20x20xf16>, memref<4xsi32>
    // CHECKA: [[DDR_OUT_DATA:%.*]], [[DDR_OUT_SHAPE:%.*]] = VPUIP.UngroupBoundedBuffer

    %RESULT_DATA = VPUIP.Copy inputs(%OUT_DATA: memref<2x4x20x20xf16>) outputs(%arg2 : memref<2x4x20x20xf16>) -> memref<2x4x20x20xf16>
    %RESULT_SHAPE = VPUIP.Copy inputs(%OUT_SHAPE: memref<4xsi32>) outputs(%arg3 : memref<4xsi32>) -> memref<4xsi32>
    // CHECK: [[DATA_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[DATA_COPY]]
    // CHECK-SAME: outputs([[OUT_DATA]]
    // CHECK: [[SHAPE_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SHAPE_COPY]]
    // CHECK-SAME: outputs([[OUT_SHAPE]]

    return %RESULT_DATA, %RESULT_SHAPE: memref<2x4x20x20xf16>, memref<4xsi32>
    // CHECK: return [[DATA_RESULT]], [[SHAPE_RESULT]]
  }
}

// -----

module @TestSwKernel attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]

  module @VPU.SW {
    func.func private @builtin_ReLU(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "relu_fp16.cpp", VPU.kernel_entry = "relu_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  // CHECK-LABEL: main
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
    // CHECK: DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
    // CHECK: DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
  }

  // CHECK-LABEL: main
  func.func @main(%arg0: memref<2x4x20x20xf16>, %arg1: memref<4xsi32>, %arg2: memref<2x4x20x20xf16>, %arg3: memref<4xsi32>) -> (memref<2x4x20x20xf16>, memref<4xsi32>) {
    // CHECK-SAME: [[IN_DATA:%.+]]: memref<2x4x20x20xf16>, [[IN_SHAPE:%.+]]: memref<4xsi32>,
    // CHECK-SAME: [[OUT_DATA:%.+]]: memref<2x4x20x20xf16>, [[OUT_SHAPE:%.+]]: memref<4xsi32>

    %IN_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%arg0, %arg1) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>

    // CMX Input
    %ALLOC0 = memref.alloc() : memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %ALLOC1 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[ALLOC0:%.*]] = memref.alloc
    // CHECK: [[ALLOC1:%.*]] = memref.alloc
    %CMX_IN_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%ALLOC0, %ALLOC1) :
        memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<4xsi32,[@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    %COPY_IN = VPUIP.Copy inputs(%IN_BOUNDED_BUFFER : !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                          outputs(%CMX_IN_BOUNDED_BUFFER : !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
                          -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK: [[COPY_IN:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_DATA]]
    // CHECK-SAME: outputs([[ALLOC0]]
    // CHECK: [[SHAPE_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_SHAPE]]
    // CHECK-SAME: outputs([[ALLOC1]]

    %ALLOC2 = memref.alloc() : memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %ALLOC3 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[ALLOC2:%.*]] = memref.alloc
    // CHECK: [[ALLOC3:%.*]] = memref.alloc
    %CMX_OUT_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%ALLOC2, %ALLOC3) :
        memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    %KERNEL_OUT = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReLU
        inputs(%CMX_IN_BOUNDED_BUFFER as %arg4: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
        outputs(%CMX_OUT_BOUNDED_BUFFER as %arg5: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) on tile 0
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>{
            VPUIP.SW.Kernel.run(%arg4, %arg5) : !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>,
                                                !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
        }
    // CHECK: [[KERNEL_OUT:%.*]], [[OUTPUT_DIMS:%.*]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_ReLU
    // CHECK-SAME: inputs([[ALLOC0]] as %arg4
    // CHECK-SAME: outputs([[ALLOC2]] as %arg5
    // CHECK-SAME: -> (memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run(%arg4, %arg5) : memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<2x4x20x20xf16, [@CMX_NN, 0]>
    // CHECK: }

    %ALLOC4 = memref.alloc() : memref<2x4x20x20xf16>
    %ALLOC5 = memref.alloc() : memref<4xsi32>
    // CHECK: [[ALLOC4:%.*]] = memref.alloc
    // CHECK: [[ALLOC5:%.*]] = memref.alloc
    %OUTPUT = VPUIP.GroupBoundedBuffer(%ALLOC4, %ALLOC5) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    %COPY_OUTPUT  = VPUIP.Copy inputs(%KERNEL_OUT: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
                          outputs(%OUTPUT: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                          -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK: [[DATA_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[KERNEL_OUT]]
    // CHECK-SAME: outputs([[ALLOC4]]
    // CHECK: [[SHAPE_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[OUTPUT_DIMS]]
    // CHECK-SAME: outputs([[ALLOC5]]

    %OUT_DATA, %OUT_SHAPE = VPUIP.UngroupBoundedBuffer(%COPY_OUTPUT) :
        !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<2x4x20x20xf16>, memref<4xsi32>

    %RESULT_DATA = VPUIP.Copy inputs(%OUT_DATA: memref<2x4x20x20xf16>) outputs(%arg2 : memref<2x4x20x20xf16>) -> memref<2x4x20x20xf16>
    %RESULT_SHAPE = VPUIP.Copy inputs(%OUT_SHAPE: memref<4xsi32>) outputs(%arg3 : memref<4xsi32>) -> memref<4xsi32>
    // CHECK: [[DATA_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[DATA_COPY]]
    // CHECK-SAME: outputs([[OUT_DATA]]
    // CHECK: [[SHAPE_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SHAPE_COPY]]
    // CHECK-SAME: outputs([[OUT_SHAPE]]

    return %RESULT_DATA, %RESULT_SHAPE: memref<2x4x20x20xf16>, memref<4xsi32>
    // CHECK: return [[DATA_RESULT]], [[SHAPE_RESULT]]
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @DynamicReshape attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_DynamicReshape(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "dynamic_reshape.cpp", VPU.kernel_entry = "dynamic_reshape", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  IE.PipelineOptions @Options {
    IE.Option @VPU.FP16CompressedConv : false
    IE.Option @VPU.ReduceSupported : false
    IE.Option @VPU.AutoPaddingODU : false
    IE.Option @VPU.AutoPaddingIDU : false
    IE.Option @VPU.BarrierMaxVariantSum : 256
    IE.Option @VPU.BarrierMaxVariantCount : 256
    IE.Option @VPU.MaxKernelSize : 11
  }
  IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

  // CHECK-LABEL: main
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_7" : tensor<32x1x548xf16>
    DataInfo "vpux_ie_shape_Parameter_7" : tensor<3xsi32>
    DataInfo "NewShape" : tensor<3xsi32>
  } outputsInfo : {
    DataInfo "Softmax_8" friendlyName = "Result_9" : tensor<32x1x548xf16>
    DataInfo "vpux_ie_shape_Softmax_8" : tensor<3xsi32>
  }

  // CHECK-LABEL: main
  // CHECK-SAME:    [[IN_DATA:%[a-z0-9]+]]: memref<32x1x548xf16, @DDR>,
  // CHECK-SAME:    [[IN_SHAPE:%[a-z0-9]+]]: memref<3xsi32, @DDR>,
  // CHECK-SAME:    [[NEW_SHAPE:%[a-z0-9]+]]: memref<3xsi32, @DDR>,
  // CHECK-SAME:    [[OUT_DATA:%[a-z0-9]+]]: memref<32x1x548xf16, @DDR>,
  // CHECK-SAME:    [[OUT_SHAPE:%[a-z0-9]+]]: memref<3xsi32, @DDR>)
  func.func @main(%arg0: memref<32x1x548xf16, @DDR>, %arg1: memref<3xsi32, @DDR>, %arg2: memref<3xsi32, @DDR>, %arg3: memref<32x1x548xf16, @DDR>, %arg4: memref<3xsi32, @DDR>) -> (memref<32x1x548xf16, @DDR>, memref<3xsi32, @DDR>) {
    %2 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<32x1x548xf16, @DDR>, memref<3xsi32, @DDR> -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>

    %alloc_15 = memref.alloc() : memref<32x1x548xf16, [@CMX_NN, 0]>
    %alloc_16 = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
    %20 = VPUIP.GroupBoundedBuffer(%alloc_15, %alloc_16) : memref<32x1x548xf16, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    %21 = VPUIP.Copy inputs(%2 : !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>) outputs(%20 : !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    // CHECK:       [[IN_DATA_COPY:%.+]] = VPUIP.Copy inputs([[IN_DATA]]
    // CHECK:       [[IN_SHAPE_COPY:%.+]] = VPUIP.Copy inputs([[IN_SHAPE]]

    %alloc_17 = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
    %22 = VPUIP.Copy inputs(%arg2 : memref<3xsi32, @DDR>) outputs(%alloc_17 : memref<3xsi32, [@CMX_NN, 0]>) -> memref<3xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NEW_SHAPE_COPY:%.+]] = VPUIP.Copy inputs([[NEW_SHAPE]]

    %alloc_18 = memref.alloc() : memref<32x1x548xf16, [@CMX_NN, 0]>
    %alloc_19 = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
    %23 = VPUIP.GroupBoundedBuffer(%alloc_18, %alloc_19) : memref<32x1x548xf16, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>

    %results_20 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicReshape inputs(%21 as %arg5: !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>, %22 as %arg6: memref<3xsi32, [@CMX_NN, 0]>) outputs(%23 as %arg7: !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) on tile 0 -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>{
      VPUIP.SW.Kernel.run {attrs = [1]}(%arg5, %arg6, %arg7) : !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>, memref<3xsi32, [@CMX_NN, 0]>, !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    }
    // CHECK:       {{%.+}}, [[PROPAGATED_SHAPE:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:      @builtin_DynamicReshape
    // CHECK-SAME:      inputs(
    // CHECK-SAME:      [[IN_DATA_COPY]]
    // CHECK-SAME:      [[NEW_SHAPE_COPY]]

    %alloc_21 = memref.alloc() : memref<32x1x548xf16, @DDR>
    %alloc_22 = memref.alloc() : memref<3xsi32, @DDR>
    %24 = VPUIP.GroupBoundedBuffer(%alloc_21, %alloc_22) : memref<32x1x548xf16, @DDR>, memref<3xsi32, @DDR> -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>
    %25 = VPUIP.Copy inputs(%results_20 : !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) outputs(%24 : !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>) -> !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>
    // CHECK:       [[RESULT_DATA:%.+]] = VPUIP.Copy inputs([[IN_DATA_COPY]]
    // CHECK:       [[RESULT_SHAPE:%.+]] = VPUIP.Copy inputs([[PROPAGATED_SHAPE]]

    %data, %dynamicShape = VPUIP.UngroupBoundedBuffer(%25) : !VPUIP.BoundedBuffer<data=memref<32x1x548xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>> -> memref<32x1x548xf16, @DDR>, memref<3xsi32, @DDR>
    %26 = VPUIP.Copy inputs(%data : memref<32x1x548xf16, @DDR>) outputs(%arg3 : memref<32x1x548xf16, @DDR>) -> memref<32x1x548xf16, @DDR>
    %27 = VPUIP.Copy inputs(%dynamicShape : memref<3xsi32, @DDR>) outputs(%arg4 : memref<3xsi32, @DDR>) -> memref<3xsi32, @DDR>
    // CHECK:       [[RETURN_DATA:%.+]] = VPUIP.Copy inputs([[RESULT_DATA]]
    // CHECK:       [[RETURN_SHAPE:%.+]] = VPUIP.Copy inputs([[RESULT_SHAPE]]

    return %26, %27 : memref<32x1x548xf16, @DDR>, memref<3xsi32, @DDR>
    // CHECK:       return [[RETURN_DATA]], [[RETURN_SHAPE]]

  }
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.EnableSEPtrsOperations : false
    IE.Option @VPU.EnableExperimentalSEPtrsOperations : false
    IE.Option @VPU.FP16CompressedConv : false
    IE.Option @VPU.ReduceSupported : false
    IE.Option @VPU.AutoPaddingODU : false
    IE.Option @VPU.AutoPaddingIDU : false
    IE.Option @VPU.SprLUTEnabled : false
    IE.Option @VPU.BarrierMaxVariantSum : 64
    IE.Option @VPU.BarrierMaxVariantCount : 128
    IE.Option @VPU.MaxKernelSize : 11
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  module @VPU.SW {
    func.func private @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  IE.TileResource 4 of @NCE at 1.850000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  func.func @TileDynamicLSTMSequence(%arg0: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, %arg1: memref<1x1x1x128xf16>, %arg2: memref<1x1x1x128xf16>, %arg3: memref<1x4x128x128xf16, #NWHC>, %arg4: memref<1x1x1x2xsi32>) -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) {
    %alloc = memref.alloc() : memref<1x1x35x512xf16>
    %alloc_0 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc, %alloc_0) : memref<1x1x35x512xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_1 = memref.alloc() : memref<1x1x1x128xf16>
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x128xf16>) outputs(%alloc_1 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_2 = memref.alloc() : memref<1x1x1x128xf16>
    %2 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x128xf16>) outputs(%alloc_2 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_3 = memref.alloc() : memref<1x4x128x128xf16, #NWHC>
    %3 = VPUIP.Copy inputs(%arg3 : memref<1x4x128x128xf16, #NWHC>) outputs(%alloc_3 : memref<1x4x128x128xf16, #NWHC>) -> memref<1x4x128x128xf16, #NWHC>
    %alloc_4 = memref.alloc() : memref<1x1x1x2xsi32>
    %4 = VPUIP.Copy inputs(%arg4 : memref<1x1x1x2xsi32>) outputs(%alloc_4 : memref<1x1x1x2xsi32>) -> memref<1x1x1x2xsi32>
    %alloc_5 = memref.alloc() : memref<1x1x35x128xf16>
    %alloc_6 = memref.alloc() : memref<4xsi32>
    %5 = VPUIP.GroupBoundedBuffer(%alloc_5, %alloc_6) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_7 = memref.alloc() : memref<1x1x1x128xf16>
    %alloc_8 = memref.alloc() : memref<1x1x1x128xf16>
    %results:6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 6, 0, 0>} @VPU.SW::@builtin_LSTMSequence inputs(%0 as %arg5: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, %1 as %arg6: memref<1x1x1x128xf16>, %2 as %arg7: memref<1x1x1x128xf16>, %3 as %arg8: memref<1x4x128x128xf16, #NWHC>, %4 as %arg9: memref<1x1x1x2xsi32>, %0 as %arg10: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, %1 as %arg11: memref<1x1x1x128xf16>, %2 as %arg12: memref<1x1x1x128xf16>, %3 as %arg13: memref<1x4x128x128xf16, #NWHC>, %4 as %arg14: memref<1x1x1x2xsi32>) outputs(%5 as %arg15: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, %alloc_7 as %arg16: memref<1x1x1x128xf16>, %alloc_8 as %arg17: memref<1x1x1x128xf16>, %5 as %arg18: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, %alloc_7 as %arg19: memref<1x1x1x128xf16>, %alloc_8 as %arg20: memref<1x1x1x128xf16>) on tile 0 -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>){
      VPUIP.SW.Kernel.run {attrs = [0]}(%arg5, %arg6, %arg7, %arg8, %arg9, %arg15, %arg16, %arg17) : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x4x128x128xf16, #NWHC>, memref<1x1x1x2xsi32>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
      VPUIP.SW.Kernel.run {attrs = [0]}(%arg10, %arg11, %arg12, %arg13, %arg14, %arg18, %arg19, %arg20) : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x4x128x128xf16, #NWHC>, memref<1x1x1x2xsi32>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
    }
    %6 = VPUIP.ConcatView inputs(%results#0, %results#3 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs(%5 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %7 = VPUIP.ConcatView inputs(%results#1, %results#4 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs(%alloc_7 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %8 = VPUIP.ConcatView inputs(%results#2, %results#5 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs(%alloc_8 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_9 = memref.alloc() : memref<1x1x35x128xf16>
    %alloc_10 = memref.alloc() : memref<4xsi32>
    %9 = VPUIP.GroupBoundedBuffer(%alloc_9, %alloc_10) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %10 = VPUIP.Copy inputs(%6 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs(%9 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_11 = memref.alloc() : memref<1x1x1x128xf16>
    %11 = VPUIP.Copy inputs(%7 : memref<1x1x1x128xf16>) outputs(%alloc_11 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_12 = memref.alloc() : memref<1x1x1x128xf16>
    %12 = VPUIP.Copy inputs(%8 : memref<1x1x1x128xf16>) outputs(%alloc_12 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    return %10, %11, %12 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>

    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x35x128xf16>
    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK: [[ALLOC_2:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
    // CHECK: [[ALLOC_3:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
    // CHECK: [[RESULTS:%.+]]:6, [[DYN_OUTPUT_SHAPES:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1, -1, -1, -1>, dynamicOutputShapesMap = array<i32: 0, -1, -1>, resultSegmentSizes = array<i32: 6, 1, 0>} @VPU.SW::@builtin_LSTMSequence
    // CHECK:   inputs(
    // CHECK:     memref<1x1x35x512xf16>,
    // CHECK:     memref<1x1x1x128xf16>,
    // CHECK:     memref<1x1x1x128xf16>,
    // CHECK:     memref<1x4x128x128xf16, #NWHC>,
    // CHECK:     memref<1x1x1x2xsi32>,
    // CHECK:     memref<1x1x35x512xf16>,
    // CHECK:      memref<1x1x1x128xf16>,
    // CHECK:     memref<1x1x1x128xf16>,
    // CHECK:     memref<1x4x128x128xf16, #NWHC>,
    // CHECK:     memref<1x1x1x2xsi32>
    // CHECK:   ) dynamicInputShapes(
    // CHECK:     memref<4xsi32>
    // CHECK:   ) outputs(
    // CHECK:     memref<1x1x35x128xf16>,
    // CHECK:     memref<1x1x1x128xf16>,
    // CHECK:     memref<1x1x1x128xf16>,
    // CHECK:     memref<1x1x35x128xf16>,
    // CHECK:     memref<1x1x1x128xf16>,
    // CHECK:     memref<1x1x1x128xf16>
    // CHECK:   ) dynamicOutputShapes(
    // CHECK:     memref<4xsi32>
    // CHECK:   ) on tile 0 -> (memref<1x1x35x128xf16>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x1x35x128xf16>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<4xsi32>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}(
    // CHECK:       memref<1x1x35x512xf16>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x4x128x128xf16, #NWHC>, memref<1x1x1x2xsi32>, memref<1x1x35x128xf16>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}(
    // CHECK:       memref<1x1x35x512xf16>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x4x128x128xf16, #NWHC>, memref<1x1x1x2xsi32>, memref<1x1x35x128xf16>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
    // CHECK: }
    // CHECK: [[CONCAT_VIEW_0:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#3 : memref<1x1x35x128xf16>, memref<1x1x35x128xf16>) outputs([[ALLOC_0]] : memref<1x1x35x128xf16>) -> memref<1x1x35x128xf16>
    // CHECK: [[CONCAT_VIEW_1:%.+]] = VPUIP.ConcatView inputs([[DYN_OUTPUT_SHAPES]], [[DYN_OUTPUT_SHAPES]] : memref<4xsi32>, memref<4xsi32>) outputs([[ALLOC_1]] : memref<4xsi32>) -> memref<4xsi32>
    // CHECK: [[CONCAT_VIEW_2:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#1, [[RESULTS]]#4 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs([[ALLOC_2]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK: [[CONCAT_VIEW_3:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#2, [[RESULTS]]#5 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs([[ALLOC_3]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK: [[ALLOC_4:%.+]] = memref.alloc() : memref<1x1x35x128xf16>
    // CHECK: [[ALLOC_5:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_0]] : memref<1x1x35x128xf16>) outputs([[ALLOC_4]] : memref<1x1x35x128xf16>) -> memref<1x1x35x128xf16>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_1]] : memref<4xsi32>) outputs([[ALLOC_5]] : memref<4xsi32>) -> memref<4xsi32>
    // CHECK: [[GROUP_BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[COPY_0]], [[COPY_1]]) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK: [[ALLOC_6:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_2]] : memref<1x1x1x128xf16>) outputs([[ALLOC_6]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK: [[ALLOC_7:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_3]] : memref<1x1x1x128xf16>) outputs([[ALLOC_7]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK: return [[GROUP_BUFF]], [[COPY_2]], [[COPY_3]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
  }
}
