//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --encapsulate-codegen-ops %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @SingleCosLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res2 = IE.Cos(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[VAR1]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosWithExceptionLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    %sig = IE.Sigmoid(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16> // not supported by ShaveCodeGen, at least for now

    %cos_res2 = IE.Cos(%sig) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // Unsupported layer should stay unwrapped
    // CHECK: [[SIG_RES:%.+]] = IE.Sigmoid([[VAR1]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[SIG_RES]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @BinaryEltwiseLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "max" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %max_res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %div_res = IE.Divide(%max_res, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %div_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[MAX_RES:%.+]] = IE.Maximum([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[MAX_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[DIV_RES:%.+]] = IE.Divide([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[DIV_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR1]] : tensor<1x1x1x1000xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @ChainWithReshapesAndSlice {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1008x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1008x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1000xf32> {
    %0 = IE.Log(%arg0) : tensor<1x1008x1x1xf16, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = #NHWC}> to tensor<1x1000x1x1xf16, {order = #NHWC}>
    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf16, {order = #NHWC}> -> tensor<1x1000x1x1xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    return %4 : tensor<1x1x1x1000xf32>

// CHECK:    [[LOGC:%.+]] = IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1008x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[LOG:%.+]] = IE.Log([[ARG1]]) : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:      IE.CGCYield [[LOG]] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    } -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[SLICEC:%.+]] = IE.CodeGenCapsule inputs([[LOGC]] as [[ARG1:%.+]]: tensor<1x1008x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[SLICE:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> to tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:      IE.CGCYield [[SLICE]] : tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:    } -> tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[PCC:%.+]] = IE.CodeGenCapsule inputs([[SLICEC]] as [[ARG1:%.+]]: tensor<1x1000x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[PC:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = [[NCHW]], mem_perm = [[NWCH]]} : tensor<1x1000x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1000x1x1xf16>
// CHECK:      IE.CGCYield [[PC]] : tensor<1x1000x1x1xf16>
// CHECK:    } -> tensor<1x1000x1x1xf16>
// CHECK:    [[ARC:%.+]] = IE.CodeGenCapsule inputs([[PCC]] as [[ARG1:%.+]]: tensor<1x1000x1x1xf16>) {
// CHECK:      [[AR:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3{{\]\]}}, shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
// CHECK:      IE.CGCYield [[AR]] : tensor<1x1x1x1000xf16>
// CHECK:    } -> tensor<1x1x1x1000xf16>
// CHECK:    [[CONVC:%.+]] = IE.CodeGenCapsule inputs([[ARC]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK:      [[CONV:%.+]] = IE.Convert([[ARG1]]) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
// CHECK:      IE.CGCYield [[CONV]] : tensor<1x1x1x1000xf32>
// CHECK:    } -> tensor<1x1x1x1000xf32>
// CHECK:    return [[CONVC]] : tensor<1x1x1x1000xf32>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @EdgeReshapesAndSlice {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1008xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1000x1x1xf32, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x1008x1x1xf16, {order = #NHWC}>) -> tensor<1x1000x1x1xf32, {order = #NHWC}> {
    %0 = IE.Log(%arg0) : tensor<1x1008x1x1xf16, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = #NHWC}> to tensor<1x1000x1x1xf16, {order = #NHWC}>
    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf16, {order = #NHWC}> -> tensor<1x1000x1x1xf16>
    %3 = IE.Sigmoid(%2) : tensor<1x1000x1x1xf16> -> tensor<1x1000x1x1xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
    %5 = IE.Convert(%4) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x1000xf32> -> tensor<1x1000x1x1xf32, {order = #NHWC}>
    return %6 : tensor<1x1000x1x1xf32, {order = #NHWC}>

// CHECK:    [[LOGC:%.+]] = IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1008x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[LOG:%.+]] = IE.Log([[ARG1]]) : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:      IE.CGCYield [[LOG]] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    } -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[SLICE:%.+]] = IE.Slice [[LOGC]] [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> to tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[PC:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = [[NCHW]], mem_perm = [[NWCH]]} : tensor<1x1000x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1000x1x1xf16>
// CHECK:    [[SIGM:%.+]] = IE.Sigmoid([[PC]]) : tensor<1x1000x1x1xf16> -> tensor<1x1000x1x1xf16>
// CHECK:    [[AR:%.+]] = IE.AffineReshape([[SIGM]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3{{\]\]}}, shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
// CHECK:    [[CONVC:%.+]] = IE.CodeGenCapsule inputs([[AR]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK:      [[CONV:%.+]] = IE.Convert([[ARG1:%.+]]) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
// CHECK:      IE.CGCYield [[CONV:%.+]] : tensor<1x1x1x1000xf32>
// CHECK:    } -> tensor<1x1x1x1000xf32>
// CHECK:    [[PC1:%.+]] = IE.PermuteCast([[CONVC]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x1x1x1000xf32> -> tensor<1x1000x1x1xf32, {order = [[NHWC]]}>
// CHECK:    return [[PC1]] : tensor<1x1000x1x1xf32, {order = [[NHWC]]}>
  }
}


// -----

module @RejectDynamicShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<?x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<?x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<?x1x1x1000xf16>) -> tensor<?x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<?x1x1x1000xf16> -> tensor<?x1x1x1000xf16>
    return %cos_res : tensor<?x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<?x1x1x1000xf16>) -> tensor<?x1x1x1000xf16> {
    // CHECK-NOT: [[VAR0:%.+]] = IE.CodeGenCapsule
  }
}

// -----

module @RejectBf16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x10x100xbf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x10x100xbf16>
  }

  func.func @main(%arg0: tensor<1x1x10x100xbf16>) -> tensor<1x1x10x100xbf16> {
    %cos_res = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x1x10x100xbf16> -> tensor<1x1x10x100xbf16>
    return %cos_res : tensor<1x1x10x100xbf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x10x100xbf16>) -> tensor<1x1x10x100xbf16>
    // CHECK-NOT: [[VAR0:%.+]] = IE.CodeGenCapsule
  }
}

// -----

module @RejectNonUnitDimReshape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x4x8x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x4x128x1xf16>
  }

  func.func @main(%arg0: tensor<1x4x8x16xf16>) -> tensor<1x4x128x1xf16> {
    %0 = IE.Log(%arg0) : tensor<1x4x8x16xf16> -> tensor<1x4x8x16xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4, 128, 1]} : tensor<1x4x8x16xf16> -> tensor<1x4x128x1xf16>
    %2 = IE.Cos(%1) : tensor<1x4x128x1xf16> -> tensor<1x4x128x1xf16>
    return %2 : tensor<1x4x128x1xf16>

    // CHECK:      func.func @main([[ARG0:%.+]]: tensor<1x4x8x16xf16>) -> tensor<1x4x128x1xf16> {
    // CHECK-NEXT:   [[LOGC:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x4x8x16xf16>) {
    // CHECK-NEXT:     [[LOG:%.+]] = IE.Log([[CAPSULE_ARG]]) : tensor<1x4x8x16xf16> -> tensor<1x4x8x16xf16>
    // CHECK-NEXT:     IE.CGCYield [[LOG]] : tensor<1x4x8x16xf16>
    // CHECK-NEXT:   } -> tensor<1x4x8x16xf16>

    // CHECK-NEXT:   [[AR:%.+]] = IE.AffineReshape([[LOGC]]) {dim_mapping = {{\[\[}}0], [1], [2], [2, 3{{\]\]}}, shape_value = [1, 4, 128, 1]} : tensor<1x4x8x16xf16> -> tensor<1x4x128x1xf16>

    // CHECK-NEXT:   [[COSC:%.+]] = IE.CodeGenCapsule inputs([[AR]] as [[CAPSULE_ARG:%.+]]: tensor<1x4x128x1xf16>) {
    // CHECK-NEXT:     [[COS:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x4x128x1xf16> -> tensor<1x4x128x1xf16>
    // CHECK-NEXT:     IE.CGCYield [[COS]] : tensor<1x4x128x1xf16>
    // CHECK-NEXT:   } -> tensor<1x4x128x1xf16>
    // CHECK-NEXT:   return [[COSC]] : tensor<1x4x128x1xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-DAG: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[MAP:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

module @AcceptUnitDimMemShapeReshape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x8x4x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x4xf16>
  }

  func.func @main(%arg0: tensor<1x8x4x1xf16, {order = #NHWC}>) -> tensor<1x8x4xf16, {order = #map}> {
    %0 = IE.Log(%arg0) : tensor<1x8x4x1xf16, {order = #NHWC}> -> tensor<1x8x4x1xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 8, 4]} : tensor<1x8x4x1xf16, {order = #NHWC}> -> tensor<1x8x4xf16, {order = #map}>
    %2 = IE.Cos(%1) : tensor<1x8x4xf16, {order = #map}> -> tensor<1x8x4xf16, {order = #map}>
    return %2 : tensor<1x8x4xf16, {order = #map}>

    // CHECK:      func.func @main([[ARG0:%.+]]: tensor<1x8x4x1xf16, {order = [[NHWC]]}>) -> tensor<1x8x4xf16, {order = [[MAP]]}> {
    // CHECK-NEXT:   [[LOGC:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x8x4x1xf16, {order = [[NHWC]]}>) {
    // CHECK-NEXT:     [[LOG:%.+]] = IE.Log([[CAPSULE_ARG]]) : tensor<1x8x4x1xf16, {order = [[NHWC]]}> -> tensor<1x8x4x1xf16, {order = [[NHWC]]}>
    // CHECK-NEXT:     IE.CGCYield [[LOG]] : tensor<1x8x4x1xf16, {order = [[NHWC]]}>
    // CHECK-NEXT:   } -> tensor<1x8x4x1xf16, {order = [[NHWC]]}>

    // CHECK-NEXT:   [[ARC:%.+]] = IE.CodeGenCapsule inputs([[LOGC]] as [[CAPSULE_ARG:%.+]]: tensor<1x8x4x1xf16, {order = [[NHWC]]}>) {
    // CHECK-NEXT:     [[AR:%.+]] = IE.AffineReshape([[CAPSULE_ARG]]) {dim_mapping = {{\[\[}}0], [1], [2], [2{{\]\]}}, shape_value = [1, 8, 4]} : tensor<1x8x4x1xf16, {order = [[NHWC]]}> -> tensor<1x8x4xf16, {order = [[MAP]]}>
    // CHECK-NEXT:     IE.CGCYield [[AR]] : tensor<1x8x4xf16, {order = [[MAP]]}>
    // CHECK-NEXT:   } -> tensor<1x8x4xf16, {order = [[MAP]]}>

    // CHECK-NEXT:   [[COSC:%.+]] = IE.CodeGenCapsule inputs([[ARC]] as [[CAPSULE_ARG:%.+]]: tensor<1x8x4xf16, {order = [[MAP]]}>) {
    // CHECK-NEXT:     [[COS:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x8x4xf16, {order = [[MAP]]}> -> tensor<1x8x4xf16, {order = [[MAP]]}>
    // CHECK-NEXT:     IE.CGCYield [[COS]] : tensor<1x8x4xf16, {order = [[MAP]]}>
    // CHECK-NEXT:   } -> tensor<1x8x4xf16, {order = [[MAP]]}>
    // CHECK-NEXT:   return [[COSC]] : tensor<1x8x4xf16, {order = [[MAP]]}>
  }
}
