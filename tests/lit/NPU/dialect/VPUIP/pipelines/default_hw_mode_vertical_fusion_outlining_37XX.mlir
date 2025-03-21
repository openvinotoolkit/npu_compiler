//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VerticalFusionOutlining {
  module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x128x128xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x128x128xf16, {order = #NHWC}>
  }
  // CHECK: DataInfo "input" : tensor<1x16x128x128xf16, {order = #NHWC}>
  // CHECK: DataInfo "output" : tensor<1x16x128x128xf16, {order = #NHWC}>

  // CHECK-NOT: func.func private @main_vf1
  func.func private @main_vf1(%arg0: memref<1x16x128x128xf16, #NHWC>, %arg1: memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x128x128xf16, #NHWC>) outputs(%arg1 : memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC>
    return %0 : memref<1x16x128x128xf16, #NHWC>
  }

  // CHECK-NOT: func.func private @main_vf2
  func.func private @main_vf2(%arg0: memref<1x16x128x128xf16, #NHWC>, %arg1: memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC> {
    %alloc = memref.alloc() : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x128x128xf16, #NHWC>) outputs(%alloc : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%0 as %arg4: memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_1 as %arg5: memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg4, %arg5) : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    }
    %2 = VPUIP.Copy inputs(%1 : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC>
    return %2 : memref<1x16x128x128xf16, #NHWC>
  }

  // CHECK: func.func @main({{[^:]+}}: memref<1x16x128x128xf16, #NHWC, @DDR>, [[OUT:%.+]]: memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>
  func.func @main(%arg0: memref<1x16x128x128xf16, #NHWC>, %arg1: memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC> {
    %alloc = memref.alloc() : memref<1x16x128x128xf16, #NHWC>
    %0 = call @main_vf1(%arg0, %alloc) : (memref<1x16x128x128xf16, #NHWC>, memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC>
    %alloc_0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC>
    %1 = call @main_vf2(%0, %alloc_0) : (memref<1x16x128x128xf16, #NHWC>, memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC>
    %2 = VPUIP.Copy inputs(%1 : memref<1x16x128x128xf16, #NHWC>) outputs(%arg1 : memref<1x16x128x128xf16, #NHWC>) -> memref<1x16x128x128xf16, #NHWC>
    return %2 : memref<1x16x128x128xf16, #NHWC>
  }

    // CHECK:      [[NET_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK-NEXT: [[DDR0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]> -> memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK-NEXT: [[DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]> -> memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK-NEXT: [[NET_OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK-NEXT: [[CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT: [[CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT: [[CMX2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>
    // CHECK-NEXT: [[CMX3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>
    // CHECK-NEXT: [[CMX4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>
    // CHECK-NEXT: [[CMX5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[ADDR0:[0-9]+]]> -> memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>

    // CHECK: VPUIP.NNDMA {{.*}} inputs([[NET_IN]] {{.*}} outputs([[DDR0]]
    // CHECK: VPUIP.NNDMA {{.*}} inputs([[DDR1]] {{.*}} outputs([[CMX0]]

    // CHECK: VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[CMX2]] as {{[^:]+}}: memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>) outputs([[CMX3]] as {{[^:]+}}: memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>
    // CHECK: VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[CMX4]] as {{[^:]+}}: memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>) outputs([[CMX5]] as {{[^:]+}}: memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x64x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, [@CMX_NN, 0]>

    // CHECK: VPUIP.NNDMA {{.*}} inputs([[CMX1]] {{.*}} outputs([[NET_OUT]]

    // CHECK: return [[OUT]]
}
