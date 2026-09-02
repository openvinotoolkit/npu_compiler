//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK: func.func @SingleInput({{[^:]+}}: memref<1x1x1x1000xf16>)
func.func @SingleInput(%input: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3, padSize = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %output: tensor<1x1x1x1000xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x1x1x1000xf16>
}

// -----

// CHECK: func.func @OnlyOneOutput() -> memref<1x2x2x2xf16> {
func.func @OnlyOneOutput() -> tensor<1x2x2x2xf16> {
    %output = const.Declare tensor<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.CastElemType<f16>]
    return %output : tensor<1x2x2x2xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x2x2x2xf16>
}

// -----

// CHECK: func.func @TwoInputs({{[^:]+}}: memref<1x16x16x16xf16>, {{[^:]+}}: memref<1x16x16x16xf16>) -> memref<1x32x16x16xf16> {
func.func @TwoInputs(%input0: tensor<1x16x16x16xf16>, %input1: tensor<1x16x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %output = VPU.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x32x16x16xf16>
    return %output : tensor<1x32x16x16xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x32x16x16xf16>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedTensor = !VPU.DistributedTensor<
    1x2x3x4xf16, #NHWC, @DDR, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!DistributedTensorCmx = !VPU.DistributedTensor<
    1x2x3x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// CHECK: func.func @DistributedTensors(
// CHECK-SAME:  !VPUIP.DistributedBuffer<1x2x3x4xf16, #NHWC, @DDR, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK-SAME: ) -> !VPUIP.DistributedBuffer<1x2x3x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
func.func @DistributedTensors(%input: !DistributedTensor) -> !DistributedTensorCmx {
    %output = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !DistributedTensor -> !DistributedTensorCmx
    return %output : !DistributedTensorCmx

    // CHECK: return
    // CHECK-SAME: !VPUIP.DistributedBuffer<1x2x3x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseTensor = !VPU.SparseTensor<
    data=tensor<1x4x8x16xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x1x1x512xi1, {order = #NHWC}>
>

!SparseTensorCmx = !VPU.SparseTensor<
    data=tensor<1x4x8x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x1x1x512xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

// CHECK: func.func @SparseTensors(
// CHECK-SAME:  !VPUIP.SparseBuffer<
// CHECK-SAME:      data=memref<1x4x8x16xf16, {order = #NHWC}>
// CHECK-SAME:      sparsity_map=memref<1x1x1x512xi1, {order = #NHWC}>>
// CHECK-SAME: ) -> !VPUIP.SparseBuffer<
// CHECK-SAME:          data=memref<1x4x8x16xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:          sparsity_map=memref<1x1x1x512xi1, {order = #NHWC}, @CMX_NN>>
func.func @SparseTensors(%input: !SparseTensor) -> !SparseTensorCmx {
    %output = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !SparseTensor -> !SparseTensorCmx
    return %output : !SparseTensorCmx

    // CHECK: return
    // CHECK-SAME: !VPUIP.SparseBuffer<
    // CHECK-SAME:  data=memref<1x4x8x16xf16, {order = #NHWC}, @CMX_NN>,
    // CHECK-SAME:  sparsity_map=memref<1x1x1x512xi1, {order = #NHWC}, @CMX_NN>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TensorsWithBounds
// CHECK-SAME: ([[ARG:%.+]]: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:  -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
func.func @TensorsWithBounds(%arg0: tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}>
    return %0 : tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK: return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TensorsWithBounds
// CHECK-SAME:  ([[ARG:%.+]]: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:  -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
func.func @TensorsWithBounds(%arg0: tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.ReLU(%arg0) : tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_ReLU inputs([[ARG]]
    // CHECK: return [[OUTPUT]]
}

// -----

module @Convolution {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG:[^:]+]]: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
    func.func @foo1(%arg0: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
        %0 = VPU.Slice %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
        %1 = VPU.Slice %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        return %0, %1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

        // CHECK: return
        // CHECK-SAME: memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG:[^:]+]]: memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
    func.func @foo2(%input: tensor<1x4x60x60xf16>) -> tensor<1x4x60x60xf16> {
        return %input: tensor<1x4x60x60xf16>
        // CHECK: return [[ARG]] : memref<1x4x60x60xf16>
    }

    // CHECK: func.func @main([[ARG:[^:]+]]: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
    func.func @main(%arg0: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
        %0:2 = call @foo1(%arg0) : (tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>)
        %1 = call @foo2(%0#0) : (tensor<1x4x60x60xf16>) -> tensor<1x4x60x60xf16>
        return %1, %0#1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

        // CHECK: [[FOO1_RES:%.+]]:2 = call @foo1([[ARG]]) : (memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0) : (memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: return [[FOO2_RES]], [[FOO1_RES]]#1 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }
}

// -----

// CHECK: func.func @ReinterpretCast([[ARG0:%.+]]: memref<1x1000xf16>) -> memref<2x1000xi8> {
func.func @ReinterpretCast(%arg0: tensor<1x1000xf16>) -> tensor<2x1000xi8> {
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1000xf16> -> tensor<2x1000xi8>
    return %0 : tensor<2x1000xi8>

    // CHECK:  [[VAR0:%.+]] = Core.ReinterpretCast([[ARG0]]) : memref<1x1000xf16> -> memref<2x1000xi8>
    // CHECK:  return [[VAR0]] : memref<2x1000xi8>
}

// -----

// CHECK: func.func @main() -> memref<128x256xf16, [@CMX_NN, 1]>
func.func @main() -> tensor<128x256xf16, {mem_space = [@CMX_NN, 1]}> {
    %tensor = tensor.empty() : tensor<128x256xf16, {mem_space = [@CMX_NN, 1]}>
    return %tensor : tensor<128x256xf16, {mem_space = [@CMX_NN, 1]}>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xf16, [@CMX_NN, 1]>
    // CHECK: return [[ALLOC]]
}
