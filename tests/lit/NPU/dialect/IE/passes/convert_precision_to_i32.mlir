//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-precision-to-i32 --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @GatherConvertIndices
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<100xf16>
func.func @GatherConvertIndices(%arg0: tensor<100xf16>) -> tensor<10xf16> {
  %0 = const.Declare tensor<10xsi64> = dense<1> : tensor<10xsi64>

  %prob = IE.Gather(%arg0, %0) {axis_value = 0, batch_dims = 0} : tensor<100xf16>,tensor<10xsi64> -> tensor<10xf16>

  return %prob : tensor<10xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<10xsi32> = dense<1> : tensor<10xsi64>, [#const.CastElemType<si32>]
  //CHECK: [[VAL1:%.+]] = IE.Gather([[ARG_0]], [[VAL0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<100xf16>, tensor<10xsi32> -> tensor<10xf16>
  //CHECK: return [[VAL1]]
}

// -----

// CHECK-LABEL: @EqualConvert
func.func @EqualConvert(%arg0: tensor<1x10x1xsi64>) -> tensor<1x10x1xi8> {
  %0 = const.Declare tensor<1x1x1xsi64> = dense<0> : tensor<1x1x1xsi64>
  %1 = IE.Equal(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1xsi64>, tensor<1x1x1xsi64> -> tensor<1x10x1xi8>
  return %1 : tensor<1x10x1xi8>

  //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1xsi32> = dense<0> : tensor<1x1x1xsi64>, [#const.CastElemType<si32>]
  //CHECK: [[VAL0:%.+]] = IE.Equal({{[^:]+}}, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1xsi32>, tensor<1x1x1xsi32> -> tensor<1x10x1xi8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @OneHotConvert
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x100xsi32>
func.func @OneHotConvert(%arg0: tensor<1x100xsi64>) -> tensor<1x30x100xsi64> {
  %1 = IE.OneHot(%arg0) {axis_attr = 1 : i64, depth_attr = 30 : i64, mode = #IE.one_hot_mode<IGNORE_NEGATIVE>, off_value_attr = 0.000000e+00 : f64, on_value_attr = 1.000000e+00 : f64, operandSegmentSizes = array<i32: 1, 0, 0, 0>, outputType = si64} : tensor<1x100xsi64> -> tensor<1x30x100xsi64>
  return %1 : tensor<1x30x100xsi64>

  //CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[ARG_0]])
  //CHECK-SAME: shape_value = [1, 1, 100]
  //CHECK-SAME: tensor<1x100xsi32> -> tensor<1x1x100xsi32>
  //CHECK: [[VAL0:%.+]] = IE.OneHot([[RESHAPE]])
  //CHECK-SAME: axis_attr = 1 : i64
  //CHECK-SAME: depth_attr = 30 : i64
  //CHECK-SAME: outputType = si32
  //CHECK-SAME: rank_expanded = true
  //CHECK-SAME: tensor<1x1x100xsi32> -> tensor<1x30x100xsi32>
  //CHECK: return [[VAL0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ShapeOf
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]>
func.func @ShapeOf(%arg0: tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<4xsi32> {
    %SHAPE_OF = IE.ShapeOf(%arg0) {
        dstElemType = si32
    } : tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>

    // CHECK: [[SHAPE_OF:%.+]] = IE.ShapeOf([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = si32
    // CHECK-SAME:  } : tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<4xsi32>

    return %SHAPE_OF : tensor<4xsi32>

    // CHECK:   return [[SHAPE_OF]] : tensor<4xsi32>
}

// -----

// CHECK-LABEL: @AddOp
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x5x16x32xui32>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x5x16x32xui32>
func.func @AddOp(%arg0: tensor<1x5x16x32xui64>, %arg1: tensor<1x5x16x32xui64>) -> tensor<1x5x16x32xui64> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x16x32xui64>, tensor<1x5x16x32xui64> -> tensor<1x5x16x32xui64>
    return %0 : tensor<1x5x16x32xui64>

    // CHECK: [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x16x32xui32>, tensor<1x5x16x32xui32> -> tensor<1x5x16x32xui32>
    // CHECK: return [[ADD]] : tensor<1x5x16x32xui32>
}

// -----

// CHECK-LABEL: @SelectOpSi64ConvertToSi32
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1024xsi32>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x1024xsi32>
func.func @SelectOpSi64ConvertToSi32(%arg0: tensor<1x1024xsi64>, %arg1: tensor<1x1024xsi64>) -> tensor<1x1024xsi64> {
    %cond = const.Declare tensor<1x1024xi8> = dense<1> : tensor<1x1024xi8>
    %0 = IE.Select(%cond, %arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xsi64>, tensor<1x1024xsi64> -> tensor<1x1024xsi64>
    return %0 : tensor<1x1024xsi64>

    // CHECK-DAG: [[COND:%.+]] = const.Declare tensor<1x1024xsi32> = dense<1> : tensor<1x1024xi8>, [#const.CastElemType<si32>]
    // CHECK:     [[SELECT:%.+]] = IE.Select([[COND]], [[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK:     return [[SELECT]] : tensor<1x1024xsi32>
}

// -----

// CHECK-LABEL: @SelectI8CondSi64Data
// CHECK-SAME:    [[COND:%[^:]+]]: tensor<1x1024xi8>
// CHECK-SAME:    [[IN1:%[^:]+]]: tensor<1x1024xsi32>
// CHECK-SAME:    [[IN2:%[^:]+]]: tensor<1x1024xsi32>
func.func @SelectI8CondSi64Data(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xsi64>, %arg2: tensor<1x1024xsi64>) -> tensor<1x1024xsi64> {
    %0 = IE.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xsi64>, tensor<1x1024xsi64> -> tensor<1x1024xsi64>
    return %0 : tensor<1x1024xsi64>

    // CHECK: [[COND_CAST:%.+]] = IE.Convert([[COND]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK: [[SELECT:%.+]] = IE.Select([[COND_CAST]], [[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK: return [[SELECT]] : tensor<1x1024xsi32>
}

// -----

// CHECK-LABEL: @SelectI8DataNonSi32Cond
// CHECK-SAME:    [[COND:%[^:]+]]: tensor<1x1024xi8>
// CHECK-SAME:    [[IN1:%[^:]+]]: tensor<1x1024xi8>
// CHECK-SAME:    [[IN2:%[^:]+]]: tensor<1x1024xi8>
func.func @SelectI8DataNonSi32Cond(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xi8>, %arg2: tensor<1x1024xi8>) -> tensor<1x1024xi8> {
    %0 = IE.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    return %0 : tensor<1x1024xi8>

    // CHECK-DAG: [[IN2_CAST:%.+]] = IE.Convert([[IN2]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK-DAG: [[IN1_CAST:%.+]] = IE.Convert([[IN1]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK-DAG: [[COND_CAST:%.+]] = IE.Convert([[COND]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK: [[SELECT:%.+]] = IE.Select([[COND_CAST]], [[IN1_CAST]], [[IN2_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK: [[OUT_CAST:%.+]] = IE.Convert([[SELECT]]) {dstElemType = i8} : tensor<1x1024xsi32> -> tensor<1x1024xi8>
    // CHECK: return [[OUT_CAST]] : tensor<1x1024xi8>
}

// -----

// CHECK-LABEL: @SelectSi16DataNonSi32Cond
// CHECK-SAME:    [[COND:%[^:]+]]: tensor<1x1024xi8>
// CHECK-SAME:    [[IN1:%[^:]+]]: tensor<1x1024xsi16>
// CHECK-SAME:    [[IN2:%[^:]+]]: tensor<1x1024xsi16>
func.func @SelectSi16DataNonSi32Cond(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xsi16>, %arg2: tensor<1x1024xsi16>) -> tensor<1x1024xsi16> {
    %0 = IE.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xsi16>, tensor<1x1024xsi16> -> tensor<1x1024xsi16>
    return %0 : tensor<1x1024xsi16>

    // CHECK-DAG: [[IN2_CAST:%.+]] = IE.Convert([[IN2]]) {dstElemType = si32} : tensor<1x1024xsi16> -> tensor<1x1024xsi32>
    // CHECK-DAG: [[IN1_CAST:%.+]] = IE.Convert([[IN1]]) {dstElemType = si32} : tensor<1x1024xsi16> -> tensor<1x1024xsi32>
    // CHECK-DAG: [[COND_CAST:%.+]] = IE.Convert([[COND]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK: [[SELECT:%.+]] = IE.Select([[COND_CAST]], [[IN1_CAST]], [[IN2_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK: [[OUT_CAST:%.+]] = IE.Convert([[SELECT]]) {dstElemType = si16} : tensor<1x1024xsi32> -> tensor<1x1024xsi16>
    // CHECK: return [[OUT_CAST]] : tensor<1x1024xsi16>
}

// -----

// CHECK-LABEL: @SelectI8CondSi32Data
// CHECK-SAME:    [[COND:%[^:]+]]: tensor<1x1024xi8>
// CHECK-SAME:    [[IN1:%[^:]+]]: tensor<1x1024xsi32>
// CHECK-SAME:    [[IN2:%[^:]+]]: tensor<1x1024xsi32>
func.func @SelectI8CondSi32Data(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xsi32>, %arg2: tensor<1x1024xsi32>) -> tensor<1x1024xsi32> {
    %0 = IE.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    return %0 : tensor<1x1024xsi32>

    // CHECK: [[COND_CAST:%.+]] = IE.Convert([[COND]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK: [[SELECT:%.+]] = IE.Select([[COND_CAST]], [[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK: return [[SELECT]] : tensor<1x1024xsi32>
}

// -----

// CHECK-LABEL: @SelectI8DataSi32Cond
// CHECK-SAME:    [[COND:%[^:]+]]: tensor<1x1024xsi32>
// CHECK-SAME:    [[IN1:%[^:]+]]: tensor<1x1024xi8>
// CHECK-SAME:    [[IN2:%[^:]+]]: tensor<1x1024xi8>
func.func @SelectI8DataSi32Cond(%arg0: tensor<1x1024xsi32>, %arg1: tensor<1x1024xi8>, %arg2: tensor<1x1024xi8>) -> tensor<1x1024xi8> {
    %0 = IE.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    return %0 : tensor<1x1024xi8>

    // CHECK-DAG: [[IN2_CAST:%.+]] = IE.Convert([[IN2]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK-DAG: [[IN1_CAST:%.+]] = IE.Convert([[IN1]]) {dstElemType = si32} : tensor<1x1024xi8> -> tensor<1x1024xsi32>
    // CHECK: [[SELECT:%.+]] = IE.Select([[COND]], [[IN1_CAST]], [[IN2_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi32>, tensor<1x1024xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK: [[OUT_CAST:%.+]] = IE.Convert([[SELECT]]) {dstElemType = i8} : tensor<1x1024xsi32> -> tensor<1x1024xi8>
    // CHECK: return [[OUT_CAST]] : tensor<1x1024xi8>
}

// -----

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        // CHECK: DataInfo "input" : tensor<1x48x60x60xsi64>
        DataInfo "input" : tensor<1x48x60x60xsi64>
    } outputsInfo : {
        // CHECK: DataInfo "output" : tensor<1x48x60x60xsi64>
        DataInfo "output" : tensor<1x48x60x60xsi64>
    }

    // CHECK: func.func @foo1({{[^:]+}}: tensor<1x48x60x60xsi32>) -> tensor<1x48x60x60xsi32>
    func.func @foo1(%arg0: tensor<1x48x60x60xsi64>) -> tensor<1x48x60x60xsi64> {
        %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xsi64>, tensor<1x48x60x60xsi64> -> tensor<1x48x60x60xsi64>
        return %0 : tensor<1x48x60x60xsi64>
    }

    // CHECK: func.func @foo2({{[^:]+}}: tensor<1x48x60x60xsi32>) -> tensor<1x48x60x60xsi32>
    func.func @foo2(%arg0: tensor<1x48x60x60xsi64>) -> tensor<1x48x60x60xsi64> {
        %0 = IE.Negative(%arg0) : tensor<1x48x60x60xsi64> -> tensor<1x48x60x60xsi64>
        return %0 : tensor<1x48x60x60xsi64>
    }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x48x60x60xsi32>) -> tensor<1x48x60x60xsi32>
    func.func @main(%arg0: tensor<1x48x60x60xsi64>) -> tensor<1x48x60x60xsi64> {
        %0 = call @foo1(%arg0) : (tensor<1x48x60x60xsi64>) -> tensor<1x48x60x60xsi64>
        %1 = call @foo2(%0) : (tensor<1x48x60x60xsi64>) -> tensor<1x48x60x60xsi64>
        return %1 : tensor<1x48x60x60xsi64>

        // CHECK: [[OUT1:%.+]] = call @foo1([[ARG0]]) : (tensor<1x48x60x60xsi32>) -> tensor<1x48x60x60xsi32>
        // CHECK: [[OUT2:%.+]] = call @foo2([[OUT1]]) : (tensor<1x48x60x60xsi32>) -> tensor<1x48x60x60xsi32>
        // CHECK: return [[OUT2]] : tensor<1x48x60x60xsi32>
    }
}
