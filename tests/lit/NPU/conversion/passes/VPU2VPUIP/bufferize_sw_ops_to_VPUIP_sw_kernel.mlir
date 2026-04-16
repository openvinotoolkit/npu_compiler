//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convert(memref<*xi8>, memref<*xf32>) attributes {
// CHECK-SAME:                                       VPU.kernel_name = "convert", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @builtin_Equal(memref<*xf16>, memref<*xf16>, memref<*xi8>) attributes {VPU.kernel_code = "eltwise_equal.cpp", VPU.kernel_entry = "eltwise_equal", VPU.kernel_name = "eltwise_equal", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @EqualOpSWLayer
// CHECK-SAME:     ([[INPUT1:%.+]]: memref<1x1x1x5xf16>, [[INPUT2:%.+]]: memref<1x1x1x1xf16>)
func.func @EqualOpSWLayer(%input1: tensor<1x1x1x5xf16>, %input2: tensor<1x1x1x1xf16>) -> tensor<1x1x1x5xf32> {
    %equalop = VPU.Equal(%input1, %input2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x5xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x5xi8>
    %output = VPU.Convert(%equalop) {dstElemType = f32} : tensor<1x1x1x5xi8> -> tensor<1x1x1x5xf32>
    return %output : tensor<1x1x1x5xf32>

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1x5xi8>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal inputs([[INPUT1]] as {{[^:]+}}: memref<1x1x1x5xf16>, [[INPUT2]] as {{[^:]+}}: memref<1x1x1x1xf16>) outputs([[ALLOC1]] as {{[^:]+}}: memref<1x1x1x5xi8>) on tile 0 -> memref<1x1x1x5xi8>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x1x5xf16>, memref<1x1x1x1xf16>, memref<1x1x1x5xi8>
    // CHECK: }

    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x1x5xf32>
    // CHECK: [[OUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[RES]] as {{[^:]+}}: memref<1x1x1x5xi8>) outputs([[ALLOC2]] as {{[^:]+}}: memref<1x1x1x5xf32>) on tile 0 -> memref<1x1x1x5xf32>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x5xi8>, memref<1x1x1x5xf32>
    // CHECK: }

    // CHECK: return [[OUT]]
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convert(memref<*xi8, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>) attributes {
// CHECK-SAME:                                       VPU.kernel_name = "convert", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @builtin_Equal(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xi8, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "eltwise_equal.cpp", VPU.kernel_entry = "eltwise_equal", VPU.kernel_name = "eltwise_equal", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @EqualOpSWLayerCMX
// CHECK-SAME:     ([[INPUT1:%.+]]: memref<1x1x1x5xf16, [@CMX_NN, 0]>, [[INPUT2:%.+]]: memref<1x1x1x1xf16, [@CMX_NN, 0]>)
func.func @EqualOpSWLayerCMX(%input1: tensor<1x1x1x5xf16, {mem_space = [@CMX_NN, 0]}>, %input2: tensor<1x1x1x1xf16, {mem_space = [@CMX_NN, 0]}>) -> tensor<1x1x1x5xf32, {mem_space = [@CMX_NN, 0]}> {
    %equalop = VPU.Equal(%input1, %input2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x5xf16, {mem_space = [@CMX_NN, 0]}>, tensor<1x1x1x1xf16, {mem_space = [@CMX_NN, 0]}> -> tensor<1x1x1x5xi8, {mem_space = [@CMX_NN, 0]}>
    %output = VPU.Convert(%equalop) {dstElemType = f32} : tensor<1x1x1x5xi8, {mem_space = [@CMX_NN, 0]}> -> tensor<1x1x1x5xf32, {mem_space = [@CMX_NN, 0]}>
    return %output : tensor<1x1x1x5xf32, {mem_space = [@CMX_NN, 0]}>

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1x5xi8, [@CMX_NN, 0]>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal inputs([[INPUT1]] as {{[^:]+}}: memref<1x1x1x5xf16, [@CMX_NN, 0]>, [[INPUT2]] as {{[^:]+}}: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC1]] as {{[^:]+}}: memref<1x1x1x5xi8, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x5xi8, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x1x5xf16, [@CMX_NN, 0]>, memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x5xi8, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x1x5xf32, [@CMX_NN, 0]>
    // CHECK: [[OUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[RES]] as {{[^:]+}}: memref<1x1x1x5xi8, [@CMX_NN, 0]>) outputs([[ALLOC2]] as {{[^:]+}}: memref<1x1x1x5xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x5xf32, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x5xi8, [@CMX_NN, 0]>, memref<1x1x1x5xf32, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: return [[OUT]]
}

// -----

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_ConditionalCopyOp(memref<*xsi8>, memref<*xf16>, memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "conditional_copy.cpp", VPU.kernel_entry = "conditional_copy", VPU.kernel_name = "conditional_copy", VPU.task_type = @COMPUTE}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @ConditionalCopySWLayer
// CHECK-SAME:     ([[COND:%.+]]: memref<1xsi8>, [[INPUT1:%.+]]: memref<1x1x4x4xf16>, [[INPUT2:%.+]]: memref<1x1x4x4xf16>)
func.func @ConditionalCopySWLayer(%cond: tensor<1xsi8>, %input1: tensor<1x1x4x4xf16>, %input2: tensor<1x1x4x4xf16>) -> (tensor<1x1x4x4xf16>) {
    %output = VPU.ConditionalCopyOp(%cond, %input1, %input2) : tensor<1xsi8>, tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    return %output : tensor<1x1x4x4xf16>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1x4x4xf16>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ConditionalCopyOp inputs([[COND]] as {{[^:]+}}: memref<1xsi8>, [[INPUT1]] as {{[^:]+}}: memref<1x1x4x4xf16>, [[INPUT2]] as {{[^:]+}}: memref<1x1x4x4xf16>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x1x4x4xf16>) on tile 0 -> memref<1x1x4x4xf16>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1xsi8>, memref<1x1x4x4xf16>, memref<1x1x4x4xf16>, memref<1x1x4x4xf16>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x1x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:      func.func private @builtin_ReduceSum(memref<*xf16>, memref<*xf16>, i64, i64, none) attributes {VPU.kernel_code = "reduce_sum.cpp", VPU.kernel_entry = "reduce_sum", VPU.kernel_name = "reduce_sum", VPU.task_type = @COMPUTE}
// CHECK-NEXT:      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ReduceSumSWLayer
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x7x2x3xf16, #NHWC>)
func.func @ReduceSumSWLayer(%input: tensor<1x7x2x3xf16, {order = #NHWC}>) -> tensor<1x1x2x3xf16, {order = #NHWC}> {
    %output = VPU.ReduceSum(%input) {axes_value = [1], keep_dims} : tensor<1x7x2x3xf16, {order = #NHWC}> -> tensor<1x1x2x3xf16, {order = #NHWC}>
    return %output : tensor<1x1x2x3xf16, {order = #NHWC}>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1x2x3xf16, #NHWC>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs([[INPUT]] as {{[^:]+}}: memref<1x7x2x3xf16, #NHWC>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x1x2x3xf16, #NHWC>) on tile 0 -> memref<1x1x2x3xf16, #NHWC>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [1, 1, [0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x7x2x3xf16, #NHWC>, memref<1x1x2x3xf16, #NHWC>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x1x2x3xf16, #NHWC>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Log(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_log.cpp", VPU.kernel_entry = "activation_log", VPU.kernel_name = "activation_log", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ActivationLog
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x50x1x1xf16>)
func.func @ActivationLog(%input: tensor<1x50x1x1xf16>) -> tensor<1x50x1x1xf16> {
    %output = VPU.Log(%input) : tensor<1x50x1x1xf16> -> tensor<1x50x1x1xf16>
    return %output : tensor<1x50x1x1xf16>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x50x1x1xf16>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Log inputs([[INPUT]] as {{[^:]+}}: memref<1x50x1x1xf16>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x50x1x1xf16>

}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convert(memref<*xf16>, memref<*xf32>) attributes {
// CHECK-SAME:                                       VPU.kernel_name = "convert", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ConvertFP16ToFP32UsingSW
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ConvertFP16ToFP32UsingSW(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.Convert(%input) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %output : tensor<1x3x4x4xf32>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[CONVERT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[ARG]] as {{[^:]+}}: memref<1x3x4x4xf16>) outputs([[CONVERT_BUFFER_CMX]] as {{[^:]+}}: memref<1x3x4x4xf32>) on tile 0 -> memref<1x3x4x4xf32>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x3x4x4xf16>, memref<1x3x4x4xf32>
    // CHECK: }
    // CHECK: return [[OUTPUT]] : memref<1x3x4x4xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.kernel_name = "softmax", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @SingleSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x1x1x1000xf16>)
func.func @SingleSWLayer(%input: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3, padSize = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %output: tensor<1x1x1x1000xf16>

    // CHECK: [[SOFTMAX_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[ARG]] as {{[^:]+}}: memref<1x1x1x1000xf16>) outputs([[SOFTMAX_BUFFER_CMX]] as {{[^:]+}}: memref<1x1x1x1000xf16>) on tile 0 -> memref<1x1x1x1000xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 3]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
    // CHECK:  }
    // CHECK: return [[OUTPUT]] : memref<1x1x1x1000xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @VPU.SW  {
// CHECK-NEXT: func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid", VPU.kernel_name = "activation_sigmoid", VPU.task_type = @COMPUTE}
// CHECK-NEXT: func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.kernel_name = "softmax", VPU.task_type = @COMPUTE}
// CHECK-NEXT: func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @ThreeSWLayers
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x1x1x2000xf16>)
func.func @ThreeSWLayers(%input: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x2000xf16> {
    %softmax = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %sigmoid = VPU.Sigmoid(%softmax) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %output = VPU.SoftMax(%sigmoid) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>

    return %output : tensor<1x1x1x2000xf16>

    // CHECK: [[SOFTMAX1_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SOFTMAX1_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[ARG]] as {{[^:]+}}: memref<1x1x1x2000xf16>) outputs([[SOFTMAX1_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16>) on tile 0 -> memref<1x1x1x2000xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16>, memref<1x1x1x2000xf16>
    // CHECK: }

    // CHECK: [[SIGMOID_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SIGMOID_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[SOFTMAX1_SW_OUTPUT]] as {{[^:]+}}: memref<1x1x1x2000xf16>) outputs([[SIGMOID_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16>) on tile 0 -> memref<1x1x1x2000xf16>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16>, memref<1x1x1x2000xf16>
    // CHECK: }

    // CHECK: [[SOFTMAX2_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SOFTMAX2_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[SIGMOID_SW_OUTPUT]] as {{[^:]+}}: memref<1x1x1x2000xf16>) outputs([[SOFTMAX2_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16>) on tile 0 -> memref<1x1x1x2000xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16>, memref<1x1x1x2000xf16>
    // CHECK: }

    // CHECK: return [[SOFTMAX2_SW_OUTPUT]] : memref<1x1x1x2000xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_ReduceMean(memref<*xf16>, memref<*xf16>, i64, i64, none) attributes {VPU.kernel_code = "reduce_mean.cpp", VPU.kernel_entry = "reduce_mean", VPU.kernel_name = "reduce_mean", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ReduceMean
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x512x7x7xf16>, [[ARG1:%.+]]: memref<1x512x7xf16>)
func.func @ReduceMean(%input0: tensor<1x512x7x7xf16>, %input1: tensor<1x512x7xf16>) -> tensor<1x512x7xf16> {
    %output = VPU.ReduceMean(%input0) {axes_value = [2]} : tensor<1x512x7x7xf16> -> tensor<1x512x7xf16>
    return %output : tensor<1x512x7xf16>

    // CHECK: [[REDUCEMEAN_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x512x7xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMean inputs([[ARG0]] as {{[^:]+}}: memref<1x512x7x7xf16>) outputs([[REDUCEMEAN_BUFFER_CMX]] as {{[^:]+}}: memref<1x512x7xf16>) on tile 0 -> memref<1x512x7xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 1, [1]]}({{[^:]+}}, {{[^:]+}}) : memref<1x512x7x7xf16>, memref<1x512x7xf16>
    // CHECK: }
    // CHECK: return [[OUTPUT]] : memref<1x512x7xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Interpolate(memref<*xf16>, memref<*xf16>, i64, i64, i64, i64, i64, none, none, none, none, f64, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate", VPU.kernel_name = "interpolate", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @InterpolateSWLayerWithUnnecessaryScalingAxes
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x128x1x1xf16>)
func.func @InterpolateSWLayerWithUnnecessaryScalingAxes(%input: tensor<1x128x1x1xf16>) -> tensor<1x128x32x32xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 1, 1], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 32, 32], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 3.200000e+00, 3.200000e+00], sizes_attr = [1, 128, 32, 32], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x1x1xf16> -> tensor<1x128x32x32xf16>

    return %output : tensor<1x128x32x32xf16>

    // CHECK: [[INTERPOLATE_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x32x32xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[ARG]] as {{[^:]+}}: memref<1x128x1x1xf16>) outputs([[INTERPOLATE_BUFFER_CMX]] as {{[^:]+}}: memref<1x128x32x32xf16>) on tile 0 -> memref<1x128x32x32xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1, 1, 128, 1], [32, 32, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x1x1xf16>, memref<1x128x32x32xf16>
    // CHECK: }
    // CHECK: return [[OUTPUT]] : memref<1x128x32x32xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convolution(memref<*xf16>, memref<*xf16>, memref<*xf16>, none, none, none, none, i64) attributes {VPU.kernel_code = "convolution.cpp", VPU.kernel_entry = "convolution", VPU.kernel_name = "convolution", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @Convolution
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x32x64x64xf16>, [[ARG1:%.+]]: memref<64x32x3x3xf16>)
func.func @Convolution(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<64x32x3x3xf16>)
        -> tensor<1x64x62x62xf16> {
    %output = VPU.Convolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<64x32x3x3xf16> -> tensor<1x64x62x62xf16>
    return %output : tensor<1x64x62x62xf16>

    // CHECK: [[CONV_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x64x62x62xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convolution inputs([[ARG0]] as {{[^:]+}}: memref<1x32x64x64xf16>, [[ARG1]] as {{[^:]+}}: memref<64x32x3x3xf16>) outputs([[CONV_BUFFER_CMX]] as {{[^:]+}}: memref<1x64x62x62xf16>) on tile 0 -> memref<1x64x62x62xf16>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [0, 0], [0, 0], [1, 1], 1]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x32x64x64xf16>, memref<64x32x3x3xf16>, memref<1x64x62x62xf16>
    // CHECK: }
    // CHECK: return [[OUTPUT]] : memref<1x64x62x62xf16>
}

// -----
// Neither of SW Kernel's input and output buffers fit in NNCMX, so both of them should be placed in DDR
// but they will later be converted from SW Kernel to VPUIP.PermuteDMA operations.
// Leave input and output buffers in NNCMX to not add a performance hit for DMA for working with DDR.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @MemPermuteSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x3x1024x1024xf16>)
func.func @MemPermuteSWLayer(%input: tensor<1x3x1024x1024xf16, {order = #NCHW}>) -> tensor<1x1024x3x1024xf16, {order = #NHWC}> {
    %memPermute = VPU.MemPermute(%input) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x1024x1024xf16, {order = #NCHW}> -> tensor<1x1024x3x1024xf16, {order = #NHWC}>
    return %memPermute: tensor<1x1024x3x1024xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x1024x1024xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) -> memref<1x3x1024x1024xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) outputs([[MEMPERMUTE_BUFFER_CMX]] as {{[^:]+}}: memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[0, 1, 2, 3]]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}) : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>, memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1024x3x1024xf16, #NHWC>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x1024x3x1024xf16, #NHWC>) -> memref<1x1024x3x1024xf16, #NHWC>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x1024x3x1024xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @MemPermuteSWLayerTooLargeForCMXButDMAConvertible
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x3x1024x1024xf16>)
func.func @MemPermuteSWLayerTooLargeForCMXButDMAConvertible(%arg0: tensor<1x3x1024x1024xf16>) -> tensor<1x3x1024x1024xf16, {order = #NHWC}> {
    %output = VPU.MemPermute(%arg0) {mem_perm = #NHWC, dst_order = #NHWC} : tensor<1x3x1024x1024xf16> -> tensor<1x3x1024x1024xf16, {order = #NHWC}>
    return %output: tensor<1x3x1024x1024xf16, {order = #NHWC}>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x1024x1024xf16>) outputs([[ALLOC0]] : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) -> memref<1x3x1024x1024xf16, [@CMX_NN, 0]>

    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[COPY0]] as {{[^:]+}}: memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[2, 0, 1, 3]]}
    // CHECK: ({{[^:]+}}, {{[^:]+}}) : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>, memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, #NHWC>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x3x1024x1024xf16, #NHWC>) -> memref<1x3x1024x1024xf16, #NHWC>
    // CHECK: return [[COPY1]] : memref<1x3x1024x1024xf16, #NHWC>
}

// -----
// CHECK-LABEL:  func.func @ReverseSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x1x3x2xf16>)
func.func @ReverseSWLayer(%input0: tensor<2x1x3x2xf16>) -> tensor<2x1x3x2xf16> {
    %output = VPU.Reverse(%input0) {axis_value = [0, 2, 3], mode = #IE.reverse_mode<INDEX>} :  tensor<2x1x3x2xf16> -> tensor<2x1x3x2xf16>
    return %output : tensor<2x1x3x2xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Reverse
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x1x3x2xf16>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<2x1x3x2xf16>) on tile 0
    // CHECK-SAME:  -> memref<2x1x3x2xf16>{

    // CHECK:  VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:  {attrs = [3, 0, [0, 2, 3]]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) :  memref<2x1x3x2xf16>, memref<2x1x3x2xf16>
    // CHECK:  }
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_ROIAlign(memref<*xf16>, memref<*xf16>, memref<*xsi32>, memref<*xf16>, i64, i64, i64, f64, i64, i64) attributes {VPU.kernel_code = "roi_align.cpp", VPU.kernel_entry = "roi_align", VPU.kernel_name = "roi_align", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ROIAlignSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x22x20x20xf16>)
func.func @ROIAlignSWLayer(%input0: tensor<2x22x20x20xf16>) -> tensor<2x22x8x8xf16> {
    %cst = const.Declare tensor<2x4xf16> = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 3.500000e+00], [0.000000e+00, 3.781250e+00, 0.000000e+00, 3.906250e+00]]> : tensor<2x4xf16>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 1]> : tensor<2xsi32>
    %output = VPU.ROIAlign(%input0, %cst, %cst_0) {alignedMode = #IE.roi_align_aligned_method<ASYMMETRIC>, pooled_h = 8 : i64, pooled_w = 8 : i64, poolingMode = #IE.roi_align_method<AVG>, sampling_ratio = 2 : i64, spatial_scale = 3.125000e-02 : f64} : tensor<2x22x20x20xf16>, tensor<2x4xf16>, tensor<2xsi32> -> tensor<2x22x8x8xf16>
    return %output : tensor<2x22x8x8xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ROIAlign
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x22x20x20xf16>, {{[^:]+}} as {{[^:]+}}: memref<2x4xf16>, {{[^:]+}} as {{[^:]+}}: memref<2xsi32>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<2x22x8x8xf16>) on tile 0
    // CHECK-SAME:  -> memref<2x22x8x8xf16>{

    // CHECK:  VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:  {attrs = [8, 8, 2, 3.125000e-02, 0, 0]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<2x22x20x20xf16>, memref<2x4xf16>, memref<2xsi32>, memref<2x22x8x8xf16>
    // CHECK:  }
}

// -----
// CHECK-LABEL:  func.func @SpaceToBatchSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x8x8x3x3xf16>)
func.func @SpaceToBatchSWLayer(%input0: tensor<2x8x8x3x3xf16>) -> tensor<48x2x2x3x3xf16> {
    %output = VPU.SpaceToBatch(%input0) {block_shape_value = [1, 6, 4, 1, 1], pads_begin_value = [0, 1, 0, 0, 0], pads_end_value = [0, 3, 0, 0, 0]} : tensor<2x8x8x3x3xf16> -> tensor<48x2x2x3x3xf16>
    return %output : tensor<48x2x2x3x3xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SpaceToBatch
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x8x8x3x3xf16>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<48x2x2x3x3xf16>) on tile 0
    // CHECK-SAME:  -> memref<48x2x2x3x3xf16>{

    // CHECK:  VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:  {attrs = [[1, 6, 4, 1, 1], [0, 1, 0, 0, 0], [0, 3, 0, 0, 0]]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<2x8x8x3x3xf16>, memref<48x2x2x3x3xf16>
    // CHECK:  }
}

// -----

// CHECK-LABEL: func.func @GroupNormalization
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x4x16x16xf16>, [[ARG1:%.+]]: memref<4xf16>, [[ARG2:%.+]]: memref<4xf16>)
func.func @GroupNormalization(%arg0: tensor<1x4x16x16xf16>, %arg1: tensor<4xf16>, %arg2: tensor<4xf16>) -> tensor<1x4x16x16xf16> {
    %0 = VPU.GroupNormalization(%arg0, %arg1, %arg2) {epsilon = 9.9999997473787516E-5 : f32, num_groups = 2 : i32} : tensor<1x4x16x16xf16>, tensor<4xf16>, tensor<4xf16> -> tensor<1x4x16x16xf16>
    return %0 : tensor<1x4x16x16xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GroupNormalization
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<1x4x16x16xf16>, {{[^:]+}} as {{[^:]+}}: memref<4xf16>, {{[^:]+}} as {{[^:]+}}: memref<4xf16>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<1x4x16x16xf16>) on tile 0
    // CHECK-SAME:  -> memref<1x4x16x16xf16>{

    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK-SAME: {attrs = [9.9999997473787516E-5, 2]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x4x16x16xf16>, memref<4xf16>, memref<4xf16>, memref<1x4x16x16xf16>
    // CHECK:  }
}

// -----
// CHECK-LABEL:  func.func @AdaptiveMaxPoolSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x3x7xf16>)
func.func @AdaptiveMaxPoolSWLayer(%arg0: tensor<2x3x7xf16>) -> (tensor<2x3x1xf16>, tensor<2x3x1xsi32>) {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %output, %output_index = VPU.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<2x3x7xf16>, tensor<1xsi32> -> tensor<2x3x1xf16>, tensor<2x3x1xsi32>
    return %output, %output_index : tensor<2x3x1xf16>, tensor<2x3x1xsi32>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_AdaptiveMaxPool
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x3x7xf16>, {{[^:]+}} as {{[^:]+}}: memref<1xsi32>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<2x3x1xf16>, {{[^:]+}} as {{[^:]+}}: memref<2x3x1xsi32>) on tile 0
    // CHECK-SAME:  -> (memref<2x3x1xf16>, memref<2x3x1xsi32>){

    // CHECK:  VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<2x3x7xf16>, memref<1xsi32>, memref<2x3x1xf16>, memref<2x3x1xsi32>
    // CHECK:  }
}


// -----
// CHECK-LABEL: func.func @MaxPool8SWLayer
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x3x30x30xf16>) -> (memref<1x3x13x26xf16>, memref<1x3x13x26xsi32>)
func.func @MaxPool8SWLayer(%arg0: tensor<1x3x30x30xf16>) -> (tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>) {
    %output, %output_index = VPU.MaxPool8(%arg0) {axis = 0 : i64, dilations = [2, 2], index_element_type = si32, kernel_size = [3, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
    return %output, %output_index : tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>


    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x3x13x26xf16>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x3x13x26xsi32>

    // CHECK: VPUIP.SW.Kernel
    // CHECK-SAME: {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
    // CHECK-SAME: inputs([[COPY0:%.+]] as {{[^:]+}}: memref<1x3x30x30xf16>)
    // CHECK-SAME: outputs([[ALLOC0]] as {{[^:]+}}: memref<1x3x13x26xf16>, [[ALLOC1]] as {{[^:]+}}: memref<1x3x13x26xsi32>) on tile 0 -> (memref<1x3x13x26xf16>, memref<1x3x13x26xsi32>){

    // CHECK: VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 3, 5], [1, 2, 1], [1, 2, 2], [0, 0, 2], [0, 0, 2], 1, [1, 1, 3, 30, 30], [1, 1, 3, 13, 26], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}
    // CHECK-SAME: memref<1x3x30x30xf16>, memref<1x3x13x26xf16>, memref<1x3x13x26xsi32>
    // CHECK: }
}

// -----
// CHECK-LABEL:  func.func @BucketizeSWLayer
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x20x20xf16>, [[ARG1:%.+]]: memref<100xf16>)
func.func @BucketizeSWLayer(%input0: tensor<1x20x20xf16>, %input1: tensor<100xf16>) -> tensor<1x20x20xsi32> {
    %output = VPU.Bucketize(%input0, %input1) {output_type = si32, with_right_bound} : tensor<1x20x20xf16>, tensor<100xf16> -> tensor<1x20x20xsi32>
    return %output : tensor<1x20x20xsi32>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Bucketize
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<1x20x20xf16>, {{[^:]+}} as {{[^:]+}}: memref<100xf16>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<1x20x20xsi32>) on tile 0
    // CHECK-SAME:  -> memref<1x20x20xsi32>{

    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [1]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x20x20xf16>, memref<100xf16>, memref<1x20x20xsi32>
    // CHECK:  }
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TensorsWithBounds
// CHECK-SAME:          ([[ARG:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) ->
// CHECK-SAME:          !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>
func.func @TensorsWithBounds(%arg0: tensor<1x18x3x3xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NHWC}>) -> tensor<1x18x3x3xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.ReLU(%arg0) : tensor<1x18x3x3xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NHWC}> -> tensor<1x18x3x3xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[SW_OP_RESULT_DATA:%.+]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC>
// CHECK:       [[SW_OP_RESULT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[SW_OP_RESULT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[SW_OP_RESULT_DATA]], [[SW_OP_RESULT_SHAPE]]) : memref<1x18x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
// CHECK:       [[SW_OP_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReLU
// CHECK-SAME:    inputs([[ARG]] as [[ARG_0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) outputs([[SW_OP_RESULT_BOUNDED_BUFFER]] as [[ARG_1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) on tile 0
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>{
// CHECK:         VPUIP.SW.Kernel.run([[ARG_0]], [[ARG_1]]) : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>, !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
// CHECK:       }

    return %0 : tensor<1x18x3x3xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NHWC}>
    // CHECK: return [[SW_OP_RESULT]]
    // CHECK-SAME: !VPUIP.BoundedBuffer<
    // CHECK-SAME:  data=memref<1x18x3x3xf32, #NHWC>,
    // CHECK-SAME:  dynamic_shape=memref<4xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ShapeOf
// CHECK-SAME:          ([[ARG:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x8x48x48xf16>, dynamic_shape=memref<4xsi32>>)
func.func @ShapeOf(%DATA: tensor<1x8x48x48xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>) -> tensor<4xsi32> {

    %SHAPE_OF = VPU.ShapeOf(%DATA) :
        tensor<1x8x48x48xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>

    // CHECK:       {{%.+}}, [[SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[ARG]])

    // CHECK:       [[ALLOC_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[COPY_SHAPE:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SHAPE]]
    // CHECK-SAME:      outputs([[ALLOC_SHAPE]]

    // CHECK: [[OUT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ShapeOf
    // CHECK-SAME: inputs([[COPY_SHAPE]]
    // CHECK-SAME: outputs([[OUT_SHAPE]]

    // CHECK: [[RES_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>

    // CHECK: [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[OUT]]
    // CHECK-SAME: outputs([[RES_SHAPE]]
    // CHECK-SAME: -> memref<4xsi32>

    return %SHAPE_OF: tensor<4xsi32>
    // CHECK:   return [[COPY_OUT]] : memref<4xsi32>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipStaticPermuteCast
// CHECK-SAME:         ([[ARG0:%.+]]: memref<1x32x32x16xf16>)
func.func @SkipStaticPermuteCast(%arg0: tensor<1x32x32x16xf16, {order = #NCHW}>)
    -> tensor<1x16x32x32xf16, {order = #NHWC}> {

    %PERMUTE_CAST = VPU.PermuteCast(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NCHW
    } : tensor<1x32x32x16xf16, {order = #NCHW}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    return %PERMUTE_CAST : tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK-NOT:   VPUIP.SW.Kernel
    // CHECK:   [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME: }
    // CHECK-SAME: inputs([[ARG0]]
    // CHECK-SAME:      -> memref<1x16x32x32xf16, #NHWC>

    // CHECK:       return [[PERMUTE_CAST]] : memref<1x16x32x32xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DynamicPermuteCast
// CHECK-SAME:         ([[ARG:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x32x64x16xf16>, dynamic_shape=memref<4xsi32>>)
func.func @DynamicPermuteCast(%arg: tensor<1x32x64x16xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 1, 0]>: tensor<4xsi64>, order = #NCHW}>)
   -> (tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NHWC}>) {

    %permute_cast = VPU.PermuteCast(%arg) {
        dst_order = #NHWC,
        mem_perm = #NCHW
    } : tensor<1x32x64x16xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 1, 0]>: tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[OUT_DATA:%.+]] = memref.alloc() : memref<1x16x32x64xf16, #NHWC>
    // CHECK: [[OUT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK: [[OUT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[OUT_DATA]], [[OUT_SHAPE]])

    // CHECK: [[SW_OP_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PermuteCast
    // CHECK-SAME: inputs([[ARG]]
    // CHECK-SAME: outputs([[OUT_BOUNDED_BUFFER]]

    return %permute_cast : tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NHWC}>
    // CHECK: return [[SW_OP_RESULT]] : !VPUIP.BoundedBuffer<data=memref<1x16x32x64xf16, #NHWC>, dynamic_shape=memref<4xsi32>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Gather(memref<*xf16>, memref<*xsi32>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather", VPU.kernel_name = "gather", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GatherWithDDRAccessOutputAtCMX
// CHECK-SAME:      ([[INPUT:%.+]]: memref<51865x512xf16>)
func.func @GatherWithDDRAccessOutputAtCMX(%arg0: tensor<51865x512xf16>) -> tensor<1x16x512xf16, {mem_space = [@CMX_NN, 0]}> {
    %cst = const.Declare tensor<1x16xsi32> = dense<1> : tensor<1x16xsi64>, [#const.CastElemType<si32>]
    %output = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<51865x512xf16>, tensor<1x16xsi32> -> tensor<1x16x512xf16, {mem_space = [@CMX_NN, 0]}>
    return %output: tensor<1x16x512xf16, {mem_space = [@CMX_NN, 0]}>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare memref<1x16xsi32> = dense<1> : tensor<1x16xsi64>, [#const.CastElemType<si32>]

    // CHECK: [[OUTPUT_CMX:%.+]] = memref.alloc() : memref<1x16x512xf16, [@CMX_NN, 0]>
    // CHECK: [[GATHER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs([[INPUT]] as {{[^:]+}}: memref<51865x512xf16>, [[INDICES]] as {{[^:]+}}: memref<1x16xsi32>) outputs([[OUTPUT_CMX]] as {{[^:]+}}: memref<1x16x512xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x512xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [1, 0, 2]}
    // CHECK: ({{[^:]+}}, {{[^:]+}}) : memref<51865x512xf16>, memref<1x16xsi32>, memref<1x16x512xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: return [[GATHER]] : memref<1x16x512xf16, [@CMX_NN, 0]>
}

// -----
// Using DDR Access for GatherOp with the output buffer in DDR leads to suboptimal performance and should be avoided

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Gather(memref<*xf16>, memref<*xsi32>, memref<*xf16>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather", VPU.kernel_name = "gather", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GatherWithDDRAccessOutputAtDDR
// CHECK-SAME:      ([[INPUT:%.+]]: memref<51865x512xf16>)
func.func @GatherWithDDRAccessOutputAtDDR(%arg0: tensor<51865x512xf16>) -> tensor<1x2000x512xf16> {
    %cst = const.Declare tensor<1x2000xsi32> = dense<1> : tensor<1x2000xsi64>, [#const.CastElemType<si32>]
    %output = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<51865x512xf16>, tensor<1x2000xsi32> -> tensor<1x2000x512xf16>
    return %output: tensor<1x2000x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare memref<1x2000xsi32> = dense<1> : tensor<1x2000xsi64>, [#const.CastElemType<si32>]

    // CHECK: [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x2000x512xf16>
    // CHECK: [[GATHER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs([[INPUT]] as {{[^:]+}}: memref<51865x512xf16>, [[INDICES]] as {{[^:]+}}: memref<1x2000xsi32>) outputs([[OUTPUT_DDR]] as {{[^:]+}}: memref<1x2000x512xf16>) on tile 0 -> memref<1x2000x512xf16>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [1, 0, 2]}
    // CHECK: ({{[^:]+}}, {{[^:]+}}) : memref<51865x512xf16>, memref<1x2000xsi32>, memref<1x2000x512xf16>
    // CHECK: }

    // CHECK: return [[GATHER]] : memref<1x2000x512xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.0>

// CHECK-LABEL: func.func @DynamicDequantizeSWLayer
// CHECK-SAME:     [[INPUT:%.+]]: memref<16x7x3x51xi4>, [[SCALE:%.+]]: memref<16x1x3x1xf16>, [[ZP:%.+]]: memref<16x7x1x51xi4>
func.func @DynamicDequantizeSWLayer(%input: tensor<16x7x3x51xi4>, %scale: tensor<16x1x3x1xf16>, %zp: tensor<16x7x1x51xi4>) -> tensor<16x7x3x51xf16> {
    %0 = VPU.QuantizeCast(%input) {dstElemType = !qElemType} : tensor<16x7x3x51xi4> -> tensor<16x7x3x51x!qElemType>
    %1 = VPU.DynamicDequantize(%0, %scale, %zp) {dstElemType = f16} : tensor<16x7x3x51x!qElemType>, tensor<16x1x3x1xf16>, tensor<16x7x1x51xi4> -> tensor<16x7x3x51xf16>
    return %1 : tensor<16x7x3x51xf16>

    // CHECK: [[IN_CAST:%.+]]  = VPUIP.QuantizeCast inputs([[INPUT]] : memref<16x7x3x51xi4>) -> memref<16x7x3x51x!qElemType>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<16x7x3x51xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK-SAME:                inputs([[IN_CAST]] as [[ARG_0:%.+]]: memref<16x7x3x51x!qElemType>, [[SCALE]] as [[ARG_1:%.+]]: memref<16x1x3x1xf16>, [[ZP]] as [[ARG_2:%.+]]: memref<16x7x1x51xi4>)
    // CHECK-SAME:                outputs([[OUT_ALLOC]] as [[ARG_3:%.+]]: memref<16x7x3x51xf16>) on tile 0 -> memref<16x7x3x51xf16>
    // CHECK-SAME:            {
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}([[ARG_0]], [[ARG_1]], [[ARG_2]], [[ARG_3]]) : memref<16x7x3x51x!qElemType>, memref<16x1x3x1xf16>, memref<16x7x1x51xi4>, memref<16x7x3x51xf16>
    // CHECK:                 }
    // CHECK: return [[OUTPUT]] : memref<16x7x3x51xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_GRUSequence(memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, i64, i64, i64, i64, f64) attributes {VPU.kernel_code = "gru_sequence.cpp", VPU.kernel_entry = "gru_sequence", VPU.kernel_name = "gru_sequence", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GRUSequenceWithDDRAccess
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x1x200xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: memref<1x1x1024xf16>
func.func @GRUSequenceWithDDRAccess(%arg0: tensor<1x1x200xf16>, %arg1: tensor<1x1x1024xf16>) -> (tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>) {
    %cst = const.Declare tensor<1x3072x200xf16> = dense<1.000000e+00> : tensor<1x3072x200xf16>
    %cst_0 = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    %cst_1 = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {__inplace_operands_attr__ = ["true", "true", "true", "true", "true"], clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x200xf16>, tensor<1x1x1024xf16>, tensor<1x3072x200xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    return {__inplace_operands_attr__ = ["true", "true"]} %middle_hidden_state, %output_hidden_state : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>

    // CHECK: [[CST:%.+]] = const.Declare memref<1x3072x200xf16> = dense<1.000000e+00> : tensor<1x3072x200xf16>
    // CHECK: [[CST0:%.+]] = const.Declare memref<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    // CHECK: [[CST1:%.+]] = const.Declare memref<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    // CHECK: [[ALLOC4:%.+]] = memref.alloc() : memref<1x1x1x1024xf16>
    // CHECK: [[ALLOC5:%.+]] = memref.alloc() : memref<1x1x1024xf16>
    // CHECK: [[OUT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GRUSequence
    // CHECK-SAME:      inputs([[INPUT0]] as {{[^:]+}}: memref<1x1x200xf16>, [[INPUT1]] as {{[^:]+}}: memref<1x1x1024xf16>,
    // CHECK-SAME:      [[CST]] as {{[^:]+}}: memref<1x3072x200xf16>, [[CST0]] as {{[^:]+}}: memref<1x3072x1024xf16>,
    // CHECK-SAME:      [[CST1]] as {{[^:]+}}: memref<1x4096xf16>) outputs([[ALLOC4]] as {{[^:]+}}: memref<1x1x1x1024xf16>,
    // CHECK-SAME:      [[ALLOC5]] as {{[^:]+}}: memref<1x1x1024xf16>) on tile 0 -> (memref<1x1x1x1024xf16>, memref<1x1x1024xf16>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [1024, 0, 1, 1, 0.000000e+00]}
    // CHECK-SAME:          : memref<1x1x200xf16>, memref<1x1x1024xf16>, memref<1x3072x200xf16>, memref<1x3072x1024xf16>,
    // CHECK-SAME:              memref<1x4096xf16>, memref<1x1x1x1024xf16>, memref<1x1x1024xf16>
    // CHECK:}
    // CHECK: return [[OUT]]#0, [[OUT]]#1 : memref<1x1x1x1024xf16>, memref<1x1x1024xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_GRUSequenceLastPart(memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, i64, i64, i64, i64, f64) attributes {VPU.kernel_code = "gru_sequence_last_part.cpp", VPU.kernel_entry = "gru_sequence_last_part", VPU.kernel_name = "gru_sequence_last_part", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GRUSequenceLastPartWithDDRAccess
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x1x1x3072xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: memref<1x1x1024xf16>
func.func @GRUSequenceLastPartWithDDRAccess_(%arg0: tensor<1x1x1x3072xf16>, %arg1: tensor<1x1x1024xf16>) -> (tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    %cst_1 = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    %middle_hidden_state, %output_hidden_state = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst_0, %cst_1) {__inplace_operands_attr__ = ["true", "true", "true", "true"], clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x1x3072xf16>, tensor<1x1x1024xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    return {__inplace_operands_attr__ = ["true", "true"]} %middle_hidden_state, %output_hidden_state : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>

    // CHECK: [[CST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK: [[CST0:%.+]] = const.Declare memref<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    // CHECK: [[CST1:%.+]] = const.Declare memref<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x1x1x1024xf16>
    // CHECK: [[ALLOC4:%.+]] = memref.alloc() : memref<1x1x1024xf16>
    // CHECK: [[RESULT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GRUSequenceLastPart
    // CHECK-SAME:         inputs([[INPUT0]] as {{[^:]+}}: memref<1x1x1x3072xf16>, [[INPUT1]] as {{[^:]+}}: memref<1x1x1024xf16>,
    // CHECK-SAME:          [[CST0]] as {{[^:]+}}: memref<1x3072x1024xf16>, [[CST1]] as{{[^:]+}}: memref<1x4096xf16>)
    // CHECK-SAME:          outputs([[ALLOC3]] as {{[^:]+}}: memref<1x1x1x1024xf16>, [[ALLOC4]] as {{[^:]+}}: memref<1x1x1024xf16>) on tile 0
    // CHECK-SAME:          -> (memref<1x1x1x1024xf16>, memref<1x1x1024xf16>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [1024, 0, 1, 1, 0.000000e+00]}
    // CHECK-SAME:      : memref<1x1x1x3072xf16>, memref<1x1x1024xf16>, memref<1x3072x1024xf16>,
    // CHECK-SAME:      memref<1x4096xf16>, memref<1x1x1x1024xf16>, memref<1x1x1024xf16>
    // CHECK: }
    // CHECK: return [[RESULT]]#0, [[RESULT]]#1 : memref<1x1x1x1024xf16>, memref<1x1x1024xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Concat(memref<*xf16>, memref<*xsi32>, memref<*xf16>, memref<*xsi32>, memref<*xf16>, memref<*xsi32>, none, none) attributes {VPU.kernel_code = "concat.cpp", VPU.kernel_entry = "concat", VPU.kernel_name = "concat", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:   func.func @ConcatSWLayer

// CHECK-SAME:         ([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>,
// CHECK-SAME:         [[ARG1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:         -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>
func.func @ConcatSWLayer(%arg0: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}>,
                                %arg1: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}>)
                                -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}> {

    %0 = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}>, tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}> -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}>

    // CHECK:  [[ALLOC1:%.+]] = memref.alloc() : memref<1x4x3x8xf16>
    // CHECK:  [[ALLOC2:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:  [[BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC1]], [[ALLOC2]])
    // CHECK:           : memref<1x4x3x8xf16>, memref<4xsi32>
    // CHECK:           -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>

    // CHECK:  [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
    // CHECK:           @VPU.SW::@builtin_Concat inputs([[ARG0]] as {{[^:]+}}:
    // CHECK:               !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:               [[ARG1]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>)
    // CHECK:               outputs([[BUFFER]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>) on tile 0
    // CHECK:               -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>{
    // CHECK:                       VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:             {attrs = [[0, 0, 0, 0], [0, 0, 2, 0]]}
    // CHECK:                           ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:                                   !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:                                   !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:  }
    // CHECK:  return [[RES]] : !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_RMS(memref<*xf32>, memref<*xf16>, memref<*xf32>, f64) attributes {VPU.kernel_code = "rms_norm.cpp", VPU.kernel_entry = "rms_norm", VPU.kernel_name = "rms_norm", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @RMSNorm
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x2x6xf32>
func.func @RMSNorm(%input: tensor<1x2x6xf32>) -> tensor<1x2x6xf32> {
    %cst = const.Declare tensor<6xf16> = dense<[2.900000e-02, 1.400000e-02, 3.000000e-03, 1.300000e-02, 1.500000e-02, 0.00899999961]> : tensor<6xf32>, [#const.CastElemType<f16>]
    %rmsop = VPU.RMS(%input, %cst) {eps = 9.9999997473787516E-6 : f64} : tensor<1x2x6xf32>, tensor<6xf16> -> tensor<1x2x6xf32>
    return %rmsop : tensor<1x2x6xf32>

// CHECK:    [[CST:%.+]] = const.Declare memref<6xf16> = dense<[2.900000e-02, 1.400000e-02, 3.000000e-03, 1.300000e-02, 1.500000e-02, 0.00899999961]> : tensor<6xf32>, [#const.CastElemType<f16>]
// CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x2x6xf32>
// CHECK:    [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RMS inputs([[INPUT]] as {{[^:]+}}: memref<1x2x6xf32>, [[CST]] as {{[^:]+}}: memref<6xf16>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x2x6xf32>) on tile 0 -> memref<1x2x6xf32>{
// CHECK:      VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x6xf32>, memref<6xf16>, memref<1x2x6xf32>
// CHECK:    }
// CHECK:    return [[RES]] : memref<1x2x6xf32>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Inverse(memref<*xf32>, memref<*xf32>, i64) attributes {VPU.kernel_code = "inverse.cpp", VPU.kernel_entry = "inverse", VPU.kernel_name = "inverse", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @Inverse
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x10x2x2xf32>
func.func @Inverse(%input: tensor<1x10x2x2xf32>) -> tensor<1x10x2x2xf32> {
    %inverseop = VPU.Inverse(%input) {__inplace_operands_attr__ = ["true"], adjoint} : tensor<1x10x2x2xf32> -> tensor<1x10x2x2xf32>
    return %inverseop : tensor<1x10x2x2xf32>

// CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x10x2x2xf32>
// CHECK:    [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Inverse inputs([[INPUT]] as {{[^:]+}}: memref<1x10x2x2xf32>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x10x2x2xf32>) on tile 0 -> memref<1x10x2x2xf32>{
// CHECK:      VPUIP.SW.Kernel.run {attrs = [1]}({{[^:]+}}, {{[^:]+}}) : memref<1x10x2x2xf32>, memref<1x10x2x2xf32>
// CHECK:    }
// CHECK:    return [[RES]] : memref<1x10x2x2xf32>
}

// -----

// CHECK-LABEL: func.func @DeformableConvolutionSWLayer
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x128x19x19xf16>, [[ARG1:%.+]]: memref<1x18x19x19xf16>, [[ARG2:%.+]]: memref<128x128x3x3xf16>, [[ARG3:%.+]]: memref<1x9x19x19xf16>) -> memref<1x128x19x19xf16>
func.func @DeformableConvolutionSWLayer(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x18x19x19xf16>, %arg2: tensor<128x128x3x3xf16>, %arg3: tensor<1x9x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %output = VPU.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x18x19x19xf16>, tensor<128x128x3x3xf16>, tensor<1x9x19x19xf16> -> tensor<1x128x19x19xf16>
    return %output : tensor<1x128x19x19xf16>

    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x128x19x19xf16>

    // CHECK:   VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DeformableConvolution
    // CHECK-SAME:  inputs([[ARG0]] as {{[^:]+}}: memref<1x128x19x19xf16>, [[ARG1]] as {{[^:]+}}: memref<1x18x19x19xf16>,
    // CHECK-SAME:  [[ARG2]] as {{[^:]+}}: memref<128x128x3x3xf16>, [[ARG3]] as {{[^:]+}}: memref<1x9x19x19xf16>)
    // CHECK-SAME:  outputs([[ALLOC]] as {{[^:]+}}: memref<1x128x19x19xf16>) on tile 0 -> memref<1x128x19x19xf16>{

    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [1, 1], [1, 1], [1, 1], 1, 1, 1, [0, 0, 0, 0]]}
    // CHECK-SAME:  memref<1x128x19x19xf16>, memref<1x18x19x19xf16>, memref<128x128x3x3xf16>, memref<1x9x19x19xf16>, memref<1x128x19x19xf16>
    // CHECK:   }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_1" : tensor<4x1x1xf16>
    DataInfo "input_0" : tensor<1x4x5x5xf16>
  } outputsInfo : {
    DataInfo "Broadcast_63" friendlyName = "Result_67" : tensor<1x4x5x5xf16>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: memref<4x1x1xf16>, [[ARG1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>> {
  func.func @main(%arg0: tensor<4x1x1xf16>, %arg1: tensor<1x4x5x5xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>) -> tensor<1x4x5x5xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.ShapeOf(%arg1) : tensor<1x4x5x5xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>
    %1 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 4, 1, 1]} : tensor<4x1x1xf16> -> tensor<1x4x1x1xf16>
    %2 = VPU.DynamicTile(%1, %0) {bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>, output_bounds = [1, 4, 5, 5], output_shape = [1, 4, -9223372036854775808, -9223372036854775808]} : tensor<1x4x1x1xf16>, tensor<4xsi32> -> tensor<1x4x5x5xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>
    return %2 : tensor<1x4x5x5xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>

    // CHECK:       {{%.+}}, [[SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[ARG1]])
    // CHECK:       [[ALLOC_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[COPY_SHAPE:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SHAPE]]
    // CHECK-SAME:      outputs([[ALLOC_SHAPE]]
    // CHECK:    [[ALLOC_1:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ShapeOf
    // CHECK-SAME:      inputs([[COPY_SHAPE]] as {{[^:]+}}: memref<4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[ALLOC_1]] as {{[^:]+}}: memref<4xsi32, [@CMX_NN, 0]>) on tile 0 -> memref<4xsi32, [@CMX_NN, 0]>{
    // CHECK:      VPUIP.SW.Kernel.run
    // CHECK-SAME:  : memref<4xsi32, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[ALLOC_2:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy inputs([[RESULT]] : memref<4xsi32, [@CMX_NN, 0]>) outputs([[ALLOC_2]] : memref<4xsi32>) -> memref<4xsi32>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<4x1x1xf16>) -> memref<1x4x1x1xf16>
    // CHECK:    [[ALLOC_3:%.+]] = memref.alloc() : memref<1x4x5x5xf16>
    // CHECK:    [[ALLOC_4:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:    [[BUFF_0:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_3]], [[ALLOC_4]]) : memref<1x4x5x5xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:    [[RESULT_0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicTile
    // CHECK-SAME:      inputs([[RESHAPE]] as {{[^:]+}}: memref<1x4x1x1xf16>, [[COPY_0]] as{{[^:]+}}: memref<4xsi32>)
    // CHECK-SAME:      outputs([[BUFF_0]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>) on tile 0
    // CHECK-SAME:      -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>{
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 1, 1]]}
    // CHECK-SAME:      : memref<1x4x1x1xf16>, memref<4xsi32>,
    // CHECK-SAME:      !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:    }
    // CHECK:    return [[RESULT_0]] : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>
  }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func @DynamicTileFromBroadcast([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64>, dynamic_shape=memref<3xsi32>>, [[ARG1:%.+]]: memref<4xsi32>) -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>> {
func.func @DynamicTileFromBroadcast(%arg0: tensor<1x1x10xsi64, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1]>: tensor<3xsi64>, order = #CHW}>, %arg1: tensor<4xsi32>) -> tensor<1x1x10x5xsi64, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.DynamicTile(%arg0, %arg1) {bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>, output_bounds =[1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x10xsi64, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1]>: tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x10x5xsi64, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x10x5xsi64, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>

    // CHECK:    [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x10x5xsi64>
    // CHECK:    [[ALLOC_1:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:    [[BUFF_0:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_0]], [[ALLOC_1]]) : memref<1x1x10x5xsi64>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>
    // CHECK:    [[RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicTile
    // CHECK-SAME:      inputs([[ARG0]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64>, dynamic_shape=memref<3xsi32>>,
    // CHECK-SAME:      [[ARG1]] as {{[^:]+}}: memref<4xsi32>)
    // CHECK-SAME:      outputs([[BUFF_0]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>) on tile 0
    // CHECK-SAME:      -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>{
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 1, 1]]}
    // CHECK-SAME:      : !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64>, dynamic_shape=memref<3xsi32>>, memref<4xsi32>,
    // CHECK-SAME:      !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>
    // CHECK:    }
    // CHECK:    return [[RESULT]] : !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>
}

// -----

#C = affine_map<(d0) -> (d0)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Range(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xsi32>) attributes {VPU.kernel_code = "range.cpp", VPU.kernel_entry = "range", VPU.kernel_name = "range", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @Range
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1xf32>, [[INPUT1:%.+]]: memref<1xf32>, [[INPUT2:%.+]]: memref<1xf32>
func.func @Range(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1024xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[1]>: tensor<1xsi64>, order = #C}> {
    %0 = VPU.Range(%arg0, %arg1, %arg2) {bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>, __inplace_operands_attr__ = ["true", "true", "true"], dstElemType = f32} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1024xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[1]>: tensor<1xsi64>, order = #C}>
    return {__inplace_operands_attr__ = ["true"]} %0 : tensor<1024xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[1]>: tensor<1xsi64>, order = #C}>

    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1024xf32>
    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1xsi32>
    // CHECK:    [[GROUPBOUNDEDBUFFER0:%.+]]  = VPUIP.GroupBoundedBuffer([[ALLOC0]], [[ALLOC1]]) : memref<1024xf32>, memref<1xsi32> -> !VPUIP.BoundedBuffer<data=memref<1024xf32>, dynamic_shape=memref<1xsi32>>
    // CHECK:    [[RESULTS:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Range inputs([[INPUT0]] as [[INNER_ARG3:[^:]+]]: memref<1xf32>, [[INPUT1]] as [[INNER_ARG4:[^:]+]]: memref<1xf32>, [[INPUT2]] as [[INNER_ARG5:[^:]+]]: memref<1xf32>) outputs([[GROUPBOUNDEDBUFFER0]] as [[INNER_ARG6:[^:]+]]: !VPUIP.BoundedBuffer<data=memref<1024xf32>, dynamic_shape=memref<1xsi32>>) on tile 0 -> !VPUIP.BoundedBuffer<data=memref<1024xf32>, dynamic_shape=memref<1xsi32>>{
    // CHECK:    VPUIP.SW.Kernel.run([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]], [[INNER_ARG6]]) : memref<1xf32>, memref<1xf32>, memref<1xf32>, !VPUIP.BoundedBuffer<data=memref<1024xf32>, dynamic_shape=memref<1xsi32>>
    // CHECK:    }
    // CHECK:    return [[RESULTS]] : !VPUIP.BoundedBuffer<data=memref<1024xf32>, dynamic_shape=memref<1xsi32>>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_DynamicExpand(memref<*xf16>, memref<*xsi32>, memref<*xf16>) attributes {VPU.kernel_code = "dynamic_expand.cpp", VPU.kernel_entry = "dynamic_expand", VPU.kernel_name = "dynamic_expand", VPU.task_type = @COMPUTE}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @DynamicExpandSWLayer
// CHECK-SAME:     ([[INPUT:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x3x20x20xf16>, dynamic_shape=memref<4xsi32>>) -> memref<1x3x20x20xf16>

func.func @DynamicExpandSWLayer(%input: tensor<1x3x20x20xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x20x20xf16> {
    %output = VPU.DynamicExpand(%input) : tensor<1x3x20x20xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x20x20xf16>
    return %output : tensor<1x3x20x20xf16>

// CHECK: [[ALLOC_RESULT:%.+]] = memref.alloc() : memref<1x3x20x20xf16>
// CHECK: [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicExpand inputs([[INPUT]] as [[ARG_1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x3x20x20xf16>, dynamic_shape=memref<4xsi32>>) outputs([[ALLOC_RESULT]] as [[ARG_2:%.+]]: memref<1x3x20x20xf16>) on tile 0 -> memref<1x3x20x20xf16>{
// CHECK: VPUIP.SW.Kernel.run([[ARG_1]], [[ARG_2]]) : !VPUIP.BoundedBuffer<data=memref<1x3x20x20xf16>, dynamic_shape=memref<4xsi32>>, memref<1x3x20x20xf16>
// CHECK: return [[SW_KERNEL]] : memref<1x3x20x20xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_PopulateWeightTable(memref<*xf16>, memref<*xsi32>, i64, i64) attributes {VPU.kernel_code = "populate_weight_table.cpp", VPU.kernel_entry = "populate_weight_table", VPU.kernel_name = "populate_weight_table", VPU.task_type = @COMPUTE}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @PopulateWeightTableSWLayer
// CHECK-SAME:     ([[INPUT:%.+]]: memref<4096x1x1x1xf16, #NHWC>) -> memref<4096x1x1x4xsi32, #NHWC>

func.func @PopulateWeightTableSWLayer(%input: tensor<4096x1x1x1xf16, {order = #NHWC}>)
                                    -> tensor<4096x1x1x4xsi32, {order = #NHWC}> {
    %output = VPU.PopulateWeightTable(%input) {base = 0 : i64, dstType = tensor<4096x1x1x4xsi32, {order = #NHWC}>, step = 0 : i64}
        : tensor<4096x1x1x1xf16, {order = #NHWC}> -> tensor<4096x1x1x4xsi32, {order = #NHWC}>
    return %output : tensor<4096x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[ALLOC_RESULT:%.+]] = memref.alloc() : memref<4096x1x1x4xsi32, #NHWC>
    // CHECK:       [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable
    // CHECK-SAME:      inputs([[INPUT]] as {{[^:]+}}: memref<4096x1x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_RESULT]] as {{[^:]+}}: memref<4096x1x1x4xsi32, #NHWC>) on tile 0
    // CHECK-SAME:      -> memref<4096x1x1x4xsi32, #NHWC>{
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) :
    // CHECK-SAME:              memref<4096x1x1x1xf16, #NHWC>, memref<4096x1x1x4xsi32, #NHWC>
    // CHECK:       }
    // CHECK:       return [[SW_KERNEL]] : memref<4096x1x1x4xsi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_DynamicExpand(memref<*xf32>, memref<*xsi32>, memref<*xf32>) attributes {VPU.kernel_code = "dynamic_expand.cpp", VPU.kernel_entry = "dynamic_expand", VPU.kernel_name = "dynamic_expand", VPU.task_type = @COMPUTE}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @DynamicExpandSWLayerFP32
// CHECK-SAME:     ([[INPUT:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x3x20x20xf32>, dynamic_shape=memref<4xsi32>>) -> memref<1x3x20x20xf32>

func.func @DynamicExpandSWLayerFP32(%input: tensor<1x3x20x20xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x20x20xf32> {
    %output = VPU.DynamicExpand(%input) : tensor<1x3x20x20xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x20x20xf32>
    return %output : tensor<1x3x20x20xf32>

// CHECK: [[ALLOC_RESULT:%.+]] = memref.alloc() : memref<1x3x20x20xf32>
// CHECK: [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicExpand inputs([[INPUT]] as [[ARG_1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x3x20x20xf32>, dynamic_shape=memref<4xsi32>>) outputs([[ALLOC_RESULT]] as [[ARG_2:%.+]]: memref<1x3x20x20xf32>) on tile 0 -> memref<1x3x20x20xf32>{
// CHECK: VPUIP.SW.Kernel.run([[ARG_1]], [[ARG_2]]) : !VPUIP.BoundedBuffer<data=memref<1x3x20x20xf32>, dynamic_shape=memref<4xsi32>>, memref<1x3x20x20xf32>
// CHECK: return [[SW_KERNEL]] : memref<1x3x20x20xf32>
}

// -----

// CHECK-LABEL: @ExperimentalDetectronROIFeatureExtractor
// CHECK-SAME:  ([[ARG0:%.+]]: memref<100x4xf32>, [[ARG1:%.+]]: memref<1x64x192x320xf32>, [[ARG2:%.+]]: memref<1x64x96x160xf32>, [[ARG3:%.+]]: memref<1x64x48x80xf32>)
func.func @ExperimentalDetectronROIFeatureExtractor(%arg0: tensor<100x4xf32>, %arg1: tensor<1x64x192x320xf32>, %arg2: tensor<1x64x96x160xf32>, %arg3: tensor<1x64x48x80xf32>) -> (tensor<100x64x14x14xf32>, tensor<100x4xf32>) {
    %aux_reordered_rois = const.Declare tensor<400xf32> = dense<0.000000e+00> : tensor<400xf32>
    %aux_original_roi_map = const.Declare tensor<100xui32> = dense<0> : tensor<100xui32>
    %aux_output_rois_features = const.Declare tensor<1254400xf32> = dense<0.000000e+00> : tensor<1254400xf32>
    %aux_levels = const.Declare tensor<100xui32> = dense<0> : tensor<100xui32>
    %output, %outputROIs = VPU.ExperimentalDetectronROIFeatureExtractor(%arg0, %arg1, %arg2, %arg3, %aux_reordered_rois, %aux_original_roi_map, %aux_output_rois_features, %aux_levels) {
        attr = #IE.ExperimentalDetectronROIFeatureExtractor<output_size = 14 : i64, sampling_ratio = 2 : i64, aligned = false, pyramid_scales = [4, 8, 16]>
    } : tensor<100x4xf32>, tensor<1x64x192x320xf32>, tensor<1x64x96x160xf32>, tensor<1x64x48x80xf32>, tensor<400xf32>, tensor<100xui32>, tensor<1254400xf32>, tensor<100xui32>
      -> tensor<100x64x14x14xf32>, tensor<100x4xf32>
    return %output, %outputROIs : tensor<100x64x14x14xf32>, tensor<100x4xf32>

    // CHECK:       [[AUX_REORDERED_ROIS:%.+]] = const.Declare memref<400xf32> = dense<0.000000e+00> : tensor<400xf32>
    // CHECK:       [[AUX_ORIGINAL_ROI_MAP:%.+]] = const.Declare memref<100xui32> = dense<0> : tensor<100xui32>
    // CHECK:       [[AUX_OUTPUT_ROIS_FEATURES:%.+]] = const.Declare memref<1254400xf32> = dense<0.000000e+00> : tensor<1254400xf32>
    // CHECK:       [[AUX_LEVELS:%.+]] = const.Declare memref<100xui32> = dense<0> : tensor<100xui32>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<100x64x14x14xf32>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<100x4xf32>
    // CHECK:       [[SW:%.+]]:6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 6, 0, 0>} @VPU.SW::@builtin_ExperimentalDetectronROIFeatureExtractor
    // CHECK-SAME:    inputs([[ARG0]] as [[INNER_ARG0:%[^:]+]]: memref<100x4xf32>,
    // CHECK-SAME:           [[ARG1]] as [[INNER_ARG1:%[^:]+]]: memref<1x64x192x320xf32>,
    // CHECK-SAME:           [[ARG2]] as [[INNER_ARG2:%[^:]+]]: memref<1x64x96x160xf32>,
    // CHECK-SAME:           [[ARG3]] as [[INNER_ARG3:%[^:]+]]: memref<1x64x48x80xf32>,
    // CHECK-SAME:           [[AUX_REORDERED_ROIS]] as [[INNER_AUX_REORDERED_ROIS:%[^:]+]]: memref<400xf32>,
    // CHECK-SAME:           [[AUX_ORIGINAL_ROI_MAP]] as [[INNER_AUX_ORIGINAL_ROI_MAP:%[^:]+]]: memref<100xui32>,
    // CHECK-SAME:           [[AUX_OUTPUT_ROIS_FEATURES]] as [[INNER_AUX_OUTPUT_ROIS_FEATURES:%[^:]+]]: memref<1254400xf32>,
    // CHECK-SAME:           [[AUX_LEVELS]] as [[INNER_AUX_LEVELS:%[^:]+]]: memref<100xui32>)
    // CHECK-SAME:    outputs([[ALLOC_0]] as [[INNER_ALLOC_0:%[^:]+]]: memref<100x64x14x14xf32>,
    // CHECK-SAME:            [[ALLOC_1]] as [[INNER_ALLOC_1:%[^:]+]]: memref<100x4xf32>
    // CHECK-SAME:            [[AUX_REORDERED_ROIS]] as [[INNER_AUX_REORDERED_ROIS_OUT:%[^:]+]]: memref<400xf32>,
    // CHECK-SAME:            [[AUX_ORIGINAL_ROI_MAP]] as [[INNER_AUX_ORIGINAL_ROI_MAP_OUT:%[^:]+]]: memref<100xui32>,
    // CHECK-SAME:            [[AUX_OUTPUT_ROIS_FEATURES]] as [[INNER_AUX_OUTPUT_ROIS_FEATURES_OUT:%[^:]+]]: memref<1254400xf32>,
    // CHECK-SAME:            [[AUX_LEVELS]] as [[INNER_AUX_LEVELS_OUT:%[^:]+]]: memref<100xui32>)
    // CHECK-SAME:    on tile 0 -> (memref<100x64x14x14xf32>, memref<100x4xf32>, memref<400xf32>, memref<100xui32>, memref<1254400xf32>, memref<100xui32>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 14, 2, 0, [4, 8, 16]]}(
    // CHECK-SAME:            [[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]], [[INNER_ARG3]],
    // CHECK-SAME:            [[INNER_AUX_REORDERED_ROIS]], [[INNER_AUX_ORIGINAL_ROI_MAP]], [[INNER_AUX_OUTPUT_ROIS_FEATURES]], [[INNER_AUX_LEVELS]],
    // CHECK-SAME:            [[INNER_ALLOC_0]], [[INNER_ALLOC_1]],
    // CHECK-SAME:            [[INNER_AUX_REORDERED_ROIS_OUT]], [[INNER_AUX_ORIGINAL_ROI_MAP_OUT]], [[INNER_AUX_OUTPUT_ROIS_FEATURES_OUT]], [[INNER_AUX_LEVELS_OUT]])
    // CHECK:       [[SW]]#0, [[SW]]#1 : memref<100x64x14x14xf32>, memref<100x4xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_RoPE(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, i64) attributes {VPU.kernel_code = "rope.cpp", VPU.kernel_entry = "rope", VPU.kernel_name = "rope", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:    }

// CHECK-LABEL:  func.func @RoPE
// CHECK-SAME:      ([[INPUT:%.+]]: memref<1x32x1x64xf32>, [[INPUT_COS:%.+]]: memref<1x1x1x64xf32>, [[INPUT_SIN:%.+]]: memref<1x1x1x64xf32>)
func.func @RoPE(%input: tensor<1x32x1x64xf32>, %input_cos: tensor<1x1x1x64xf32>, %input_sin: tensor<1x1x1x64xf32>) -> tensor<1x32x1x64xf32> {
    %ropeop = VPU.RoPE(%input, %input_cos, %input_sin) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x32x1x64xf32>
    return %ropeop : tensor<1x32x1x64xf32>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x1x64xf32>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RoPE inputs([[INPUT]] as {{[^:]+}}: memref<1x32x1x64xf32>, [[INPUT_COS]] as {{[^:]+}}: memref<1x1x1x64xf32>, [[INPUT_SIN]] as {{[^:]+}}: memref<1x1x1x64xf32>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x32x1x64xf32>) on tile 0 -> memref<1x32x1x64xf32>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x32x1x64xf32>, memref<1x1x1x64xf32>, memref<1x1x1x64xf32>, memref<1x32x1x64xf32>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x32x1x64xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_RoPE(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, i64) attributes {VPU.kernel_code = "rope_ilv.cpp", VPU.kernel_entry = "rope_ilv", VPU.kernel_name = "rope_ilv", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:    }

// CHECK-LABEL:  func.func @RoPEInterleaved
// CHECK-SAME:      ([[INPUT:%.+]]: memref<1x32x1x64xf32>, [[INPUT_COS:%.+]]: memref<1x1x1x64xf32>, [[INPUT_SIN:%.+]]: memref<1x1x1x64xf32>)
func.func @RoPEInterleaved(%input: tensor<1x32x1x64xf32>, %input_cos: tensor<1x1x1x64xf32>, %input_sin: tensor<1x1x1x64xf32>) -> tensor<1x32x1x64xf32> {
    %ropeop = VPU.RoPE(%input, %input_cos, %input_sin) {mode = #IE.rope_mode<INTERLEAVED>} : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x32x1x64xf32>
    return %ropeop : tensor<1x32x1x64xf32>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x1x64xf32>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RoPE inputs([[INPUT]] as {{[^:]+}}: memref<1x32x1x64xf32>, [[INPUT_COS]] as {{[^:]+}}: memref<1x1x1x64xf32>, [[INPUT_SIN]] as {{[^:]+}}: memref<1x1x1x64xf32>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x32x1x64xf32>) on tile 0 -> memref<1x32x1x64xf32>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x32x1x64xf32>, memref<1x1x1x64xf32>, memref<1x1x1x64xf32>, memref<1x32x1x64xf32>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x32x1x64xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_RoPE(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, i64) attributes {VPU.kernel_code = "rope_pairwise.cpp", VPU.kernel_entry = "rope_pairwise", VPU.kernel_name = "rope_pairwise", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:    }

// CHECK-LABEL:  func.func @RoPEPairwise
// CHECK-SAME:      ([[INPUT:%.+]]: memref<1x32x1x64xf32>, [[INPUT_COS:%.+]]: memref<1x1x1x64xf32>, [[INPUT_SIN:%.+]]: memref<1x1x1x64xf32>)
func.func @RoPEPairwise(%input: tensor<1x32x1x64xf32>, %input_cos: tensor<1x1x1x64xf32>, %input_sin: tensor<1x1x1x64xf32>) -> tensor<1x32x1x64xf32> {
    %ropeop = VPU.RoPE(%input, %input_cos, %input_sin) {mode = #IE.rope_mode<PAIRWISE>} : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x32x1x64xf32>
    return %ropeop : tensor<1x32x1x64xf32>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x1x64xf32>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RoPE inputs([[INPUT]] as {{[^:]+}}: memref<1x32x1x64xf32>, [[INPUT_COS]] as {{[^:]+}}: memref<1x1x1x64xf32>, [[INPUT_SIN]] as {{[^:]+}}: memref<1x1x1x64xf32>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x32x1x64xf32>) on tile 0 -> memref<1x32x1x64xf32>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x32x1x64xf32>, memref<1x1x1x64xf32>, memref<1x1x1x64xf32>, memref<1x32x1x64xf32>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x32x1x64xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// module @executors {
// config.Resources 6 of @NCE at 1.700000e+03 MHz

!InputTensor0 = !VPU.DistributedTensor<
    1x32x44x44xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]
}>

!InputTensor1 = !VPU.DistributedTensor<
    1x1x44x44xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]
}>

!OutputTensor = !VPU.DistributedTensor<
    1x32x44x44xi8, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]
}>

// CHECK-LABEL:   @EqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x32x44x44xf16, @CMX_NN>, [[INPUT_1:%.+]]: memref<1x1x44x44xf16, @CMX_NN>
 func.func @EqualSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, %arg1: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> {

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> !InputTensor0
    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> !InputTensor1

    %2 = VPU.Equal(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : !InputTensor0, !InputTensor1 -> !OutputTensor

    %3 = VPU.Copy(%2) {out_mem_space = @CMX_NN} : !OutputTensor -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    return %3 : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

// CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed
// CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

// CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x32x44x44xf16, @CMX_NN>)
// CHECK-SAME:    outputs([[ALLOC0]] : !VPUIP.DistributedBuffer<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>)
// CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

// CHECK:       [[ALLOC1:%.+]] = VPURT.AllocDistributed
// CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

// CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x1x44x44xf16, @CMX_NN>)
// CHECK-SAME:    outputs([[ALLOC1]] : !VPUIP.DistributedBuffer<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>)
// CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

// CHECK:    [[ALLOC_OUT:%.+]] = VPURT.AllocDistributed
// CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

// CHECK:         [[SW_OP_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal
// CHECK-SAME:    inputs([[COPY0]] as [[INNER0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x44x44xf16, #NCHW, @CMX_NN,
// CHECK-SAME:    {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>,
// CHECK-SAME:    [[COPY1]] as [[INNER1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x44x44xf16, #NCHW, @CMX_NN,
// CHECK-SAME:    {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>)
// CHECK-SAME:    outputs([[ALLOC_OUT]] as [[INNER_OUT0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>)
// CHECK-SAME:    on tile 0 -> !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>{

// CHECK:           VPUIP.SW.Kernel.run([[INNER0]], [[INNER1]], [[INNER_OUT0]])
// CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:        compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:        memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

// CHECK:    [[OUT_MEMREF:%.+]] = memref.alloc() : memref<1x32x44x44xi8, @CMX_NN>

// CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy
// CHECK-SAME:             inputs(%results : !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>)
// CHECK-SAME:             outputs([[OUT_MEMREF]] : memref<1x32x44x44xi8, @CMX_NN>) -> memref<1x32x44x44xi8, @CMX_NN>

// CHECK:    return [[OUT_COPY]] : memref<1x32x44x44xi8, @CMX_NN>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_DynamicDataMask(memref<*xsi32>, memref<*xf16>) attributes {VPU.kernel_code = "dynamic_data_mask.cpp", VPU.kernel_entry = "dynamic_data_mask", VPU.kernel_name = "dynamic_data_mask", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:    }

// CHECK-LABEL:  func.func @DynamicDataMask
// CHECK-SAME:      ([[INPUT:%.+]]: memref<4xsi32>
func.func @DynamicDataMask(%arg0: tensor<4xsi32>) -> tensor<1x3x32x32xf16> {
    %0 = VPU.DynamicDataMask(%arg0) {outputTensorType = tensor<1x3x32x32xf16>} : tensor<4xsi32> -> tensor<1x3x32x32xf16>
    return %0 : tensor<1x3x32x32xf16>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x3x32x32xf16>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDataMask inputs([[INPUT]] as {{[^:]+}}: memref<4xsi32>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x3x32x32xf16>) on tile 0 -> memref<1x3x32x32xf16>
    // CHECK:      VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<4xsi32>, memref<1x3x32x32xf16>
    // CHECK: }
    // CHECK: return [[RES]] : memref<1x3x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$NHWC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   module @VPU.SW {
// CHECK:           func.func private @builtin_dummy_kernel(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
// CHECK-SAME:      attributes {VPU.kernel_code = "dummy_kernel.cpp", VPU.kernel_entry = "dummy_kernel", VPU.kernel_name = "dummy_kernel", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}

// CHECK-LABEL:   func.func @ExternalKernelOneInputOneOutputCMX(
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>) -> memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>
func.func @ExternalKernelOneInputOneOutputCMX(%arg0: tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) -> tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
    %0 = VPU.ExternalKernel "dummy_kernel" inputs(%arg0 : tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) attrs({}) -> tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    return %0 : tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

// CHECK:         [[ALLOC:%.+]] = memref.alloc() : memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:         [[SW:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_dummy_kernel inputs([[INPUT]] as [[INNER_ARG0:[^:]+]]: memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>) outputs([[ALLOC]] as [[INNER_ARG1:[^:]+]]: memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>) on tile 0
// CHECK-SAME:    -> memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>{
// CHECK:           VPUIP.SW.Kernel.run([[INNER_ARG0]], [[INNER_ARG1]]) : memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>, memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:         }
// CHECK:         return [[SW]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$NHWC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   module @VPU.SW {
// CHECK:           func.func private @builtin_dummy_kernel(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>)
// CHECK-SAME:      attributes {VPU.kernel_code = "dummy_kernel.cpp", VPU.kernel_entry = "dummy_kernel", VPU.kernel_name = "dummy_kernel", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}

// CHECK-LABEL:   func.func @ExternalKernelOneInputCMXOneOutputDDR(
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>) -> memref<1x512x512x10xf16, #[[$NHWC]]>
func.func @ExternalKernelOneInputCMXOneOutputDDR(%arg0: tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) -> tensor<1x512x512x10xf16, {order = #NHWC}> {
    %0 = VPU.ExternalKernel "dummy_kernel" inputs(%arg0 : tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) attrs({}) -> tensor<1x512x512x10xf16, {order = #NHWC}>
    return %0 : tensor<1x512x512x10xf16, {order = #NHWC}>

// CHECK:         [[ALLOC:%.+]] = memref.alloc() : memref<1x512x512x10xf16, #[[$NHWC]]>
// CHECK:         [[SW:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_dummy_kernel
// CHECK-SAME:    inputs([[INPUT]] as [[INNER_ARG0:[^:]+]]: memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>)
// CHECK-SAME:    outputs([[ALLOC]] as [[INNER_ARG1:[^:]+]]: memref<1x512x512x10xf16, #[[$NHWC]]>) on tile 0
// CHECK-SAME:    -> memref<1x512x512x10xf16, #[[$NHWC]]>{
// CHECK:           VPUIP.SW.Kernel.run([[INNER_ARG0]], [[INNER_ARG1]]) : memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>, memref<1x512x512x10xf16, #[[$NHWC]]>
// CHECK:         }
// CHECK:         return [[SW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$NHWC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   module @VPU.SW {
// CHECK:           func.func private @builtin_dummy_kernel(memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>)
// CHECK-SAME:      attributes {VPU.kernel_code = "dummy_kernel.cpp", VPU.kernel_entry = "dummy_kernel", VPU.kernel_name = "dummy_kernel", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}

// CHECK-LABEL:   func.func @ExternalKernelOneInputDDROneOutputCMX(
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x512x512x10xf16, #[[$NHWC]]>) -> memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>
func.func @ExternalKernelOneInputDDROneOutputCMX(%arg0: tensor<1x512x512x10xf16, {order = #NHWC}>) -> tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0],order = #NHWC}> {
    %0 = VPU.ExternalKernel "dummy_kernel" inputs(%arg0 : tensor<1x512x512x10xf16, {order = #NHWC}>) attrs({}) -> tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    return %0 : tensor<1x512x512x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

// CHECK:         [[ALLOC:%.+]] = memref.alloc() : memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:         [[SW:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_dummy_kernel
// CHECK-SAME:    inputs([[INPUT]] as [[INNER_ARG0:[^:]+]]: memref<1x512x512x10xf16, #[[$NHWC]]>)
// CHECK-SAME:    outputs([[ALLOC]] as [[INNER_ARG1:[^:]+]]: memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>) on tile 0
// CHECK-SAME:    -> memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>{
// CHECK:           VPUIP.SW.Kernel.run([[INNER_ARG0]], [[INNER_ARG1]]) : memref<1x512x512x10xf16, #[[$NHWC]]>, memref<1x512x512x1xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:         }
// CHECK:         return [[SW]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$NHWC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   module @VPU.SW {
// CHECK:           func.func private @builtin_dummy_kernel
// CHECK-SAME:      (memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
// CHECK-SAME:      attributes {VPU.kernel_code = "dummy_kernel.cpp", VPU.kernel_entry = "dummy_kernel", VPU.kernel_name = "dummy_kernel", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}

// CHECK-LABEL:   func.func @ExternalKernelMultipleIOCMX(
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT1:%.+]]: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT2:%.+]]: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>) -> (memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>) {
func.func @ExternalKernelMultipleIOCMX(%arg0: tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
                                       %arg1: tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
                                       %arg2: tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>)
                                       -> (tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) {
    %0, %1 = VPU.ExternalKernel "dummy_kernel" inputs(%arg0, %arg1, %arg2 : tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) attrs({}) -> tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    return %0, %1 : tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

// CHECK:         [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:         [[ALLOC1:%.+]] = memref.alloc() : memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:         [[SW:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_dummy_kernel inputs([[INPUT0]] as {{[^:]+}}: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT1]] as {{[^:]+}}: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT2]] as {{[^:]+}}: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>)
// CHECK:         return [[SW]]#0, [[SW]]#1
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$NHWC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   module @VPU.SW {
// CHECK:           func.func private @builtin_dummy_kernel
// CHECK-SAME:      (memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>)
// CHECK-SAME:      attributes {VPU.kernel_code = "dummy_kernel.cpp", VPU.kernel_entry = "dummy_kernel", VPU.kernel_name = "dummy_kernel", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}

// CHECK-LABEL:   func.func @ExternalKernelMultipleIOHybrid(
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT1:%.+]]: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT2:%.+]]: memref<1x512x512x10xf16, #[[$NHWC]]>) -> (memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, memref<1x512x512x10xf16, #[[$NHWC]]>) {
func.func @ExternalKernelMultipleIOHybrid(%arg0: tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
                                          %arg1: tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
                                          %arg2: tensor<1x512x512x10xf16, {order = #NHWC}>)
                                       -> (tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x512x512x10xf16, {order = #NHWC}>) {
    %0, %1 = VPU.ExternalKernel "dummy_kernel" inputs(%arg0, %arg1, %arg2 : tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x512x512x10xf16, {order = #NHWC}>) attrs({}) -> tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x512x512x10xf16, {order = #NHWC}>
    return %0, %1 : tensor<1x64x64x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x512x512x10xf16, {order = #NHWC}>

// CHECK:           [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>
// CHECK:           [[ALLOC1:%.+]] = memref.alloc() : memref<1x512x512x10xf16, #[[$NHWC]]>
// CHECK:           [[SW:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_dummy_kernel inputs([[INPUT0]] as {{[^:]+}}: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT1]] as {{[^:]+}}: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[INPUT2]] as {{[^:]+}}: memref<1x512x512x10xf16, #[[$NHWC]]>) outputs([[ALLOC0]] as {{[^:]+}}: memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, [[ALLOC1]] as {{[^:]+}}: memref<1x512x512x10xf16, #[[$NHWC]]>) on tile 0
// CHECK-SAME:      -> (memref<1x64x64x3xf16, #[[$NHWC]], [@CMX_NN, 0]>, memref<1x512x512x10xf16, #[[$NHWC]]>)
// CHECK:           return [[SW]]#0, [[SW]]#1
}

// -----

// CHECK-LABEL: func.func @AvgPool16SWLayer
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x2x5x5xf16>) -> memref<1x2x1x1xf16>
func.func @AvgPool16SWLayer(%arg0: tensor<1x2x5x5xf16>) -> tensor<1x2x1x1xf16> {
    %0 = VPU.AvgPool16(%arg0) {dilations = [2, 2], exclude_pads, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2x5x5xf16> -> tensor<1x2x1x1xf16>
    return %0 : tensor<1x2x1x1xf16>

// CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1x2x1x1xf16>
// CHECK:    [[RES:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME: {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_AvgPool16
// CHECK-SAME: inputs([[ARG0]] as {{[^:]+}}: memref<1x2x5x5xf16>)
// CHECK-SAME: outputs([[ALLOC0]] as {{[^:]+}}: memref<1x2x1x1xf16>) on tile 0 -> memref<1x2x1x1xf16>{
// CHECK:      VPUIP.SW.Kernel.run
// CHECK-SAME{LITERAL}: {attrs = [[1, 3, 3], [1, 1, 1], [1, 2, 2], [0, 0, 0], [0, 0, 0], 1]}
// CHECK-SAME: memref<1x2x5x5xf16>, memref<1x2x1x1xf16>
// CHECK:    }
// CHECK:    return [[RES]] : memref<1x2x1x1xf16>
}

// -----

#C = affine_map<(d0) -> (d0)>

!Distributed = !VPU.DistributedTensor<100xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

func.func @EmptyOp() -> (tensor<1000xf16>, tensor<1000xf16, {mem_space = [@CMX_NN, 0], order = #C}>, !Distributed) {
    %ddr = VPU.Empty : tensor<1000xf16>
    %cmx = VPU.Empty : tensor<1000xf16, {mem_space = [@CMX_NN, 0], order = #C}>
    %distributed = VPU.Empty : !Distributed
    return %ddr, %cmx, %distributed : tensor<1000xf16>, tensor<1000xf16, {mem_space = [@CMX_NN, 0], order = #C}>, !Distributed

    // CHECK: [[DDR:%.+]] = memref.alloc() : memref<1000xf16>
    // CHECK: [[CMX:%.+]] = memref.alloc() : memref<1000xf16, [@CMX_NN, 0]>
    // CHECK: [[DISTRIBUTED:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<100xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // return [[DDR]], [[CMX]], [[DISTRIBUTED]]
}
