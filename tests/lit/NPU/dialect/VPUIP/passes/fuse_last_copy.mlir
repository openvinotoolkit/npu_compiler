//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-last-copy %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NotFuseLastCopyWithQuantizeCastOpWithMultipleUsers(%arg0: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>,
                                                          %arg1: memref<1x32x56x56xui8, #NHWC>, %arg2: memref<1x16x56x56xui8, #NHWC>)
                                                          -> (memref<1x32x56x56xui8, #NHWC>, memref<1x16x56x56xui8, #NHWC>) {
    %ddr_buf = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %0 = VPUIP.SubView %ddr_buf [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56x!qElemType, #NHWC, @DDR> to
        memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %2 = VPUIP.SubView %ddr_buf [0, 16, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56x!qElemType, #NHWC, @DDR> to
        memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>, memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        outputs(%ddr_buf : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %5 = VPUIP.QuantizeCast
        inputs(%4 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>)
        -> memref<1x32x56x56xui8, #NHWC, @DDR>

    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56xui8, #NHWC, @DDR> to
        memref<1x16x56x56xui8, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %7 = memref.alloc(): memref<1x16x56x56xui8, #NHWC, @CMX_NN>
    %8 = VPUIP.Copy
        inputs(%6 : memref<1x16x56x56xui8, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        outputs(%7 : memref<1x16x56x56xui8, #NHWC, @CMX_NN>)  ->  memref<1x16x56x56xui8, #NHWC, @CMX_NN>

    %9 = VPUIP.Copy
        inputs(%5 : memref<1x32x56x56xui8, #NHWC, @DDR>)
        outputs(%arg1 : memref<1x32x56x56xui8, #NHWC>)
        -> memref<1x32x56x56xui8, #NHWC>
    %10 = VPUIP.Copy
        inputs(%8 : memref<1x16x56x56xui8, #NHWC, @CMX_NN>)
        outputs(%arg2 : memref<1x16x56x56xui8, #NHWC>)
        -> memref<1x16x56x56xui8, #NHWC>
    return %9, %10 : memref<1x32x56x56xui8, #NHWC>, memref<1x16x56x56xui8, #NHWC>

    // CHECK:       [[BUFFER0:%.+]] = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[BUFFER0]] [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>) outputs([[SUBVIEW]] : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
    // CHECK-SAME:      -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]] [0, 16, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
    // CHECK-SAME:      -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:       [[VAL5:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY0]], [[COPY1]] : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>, memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER0]] : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:       [[QUANTIZECAST0:%.+]] = VPUIP.QuantizeCast inputs([[VAL5]] : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56xui8, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[QUANTIZECAST0]] [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56xui8, #NHWC, @DDR> to memref<1x16x56x56xui8, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:       [[BUFFER1:%.+]] = memref.alloc() : memref<1x16x56x56xui8, #NHWC, @CMX_NN>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : memref<1x16x56x56xui8, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER1]] : memref<1x16x56x56xui8, #NHWC, @CMX_NN>)  ->  memref<1x16x56x56xui8, #NHWC, @CMX_NN>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[QUANTIZECAST0]] : memref<1x32x56x56xui8, #NHWC, @DDR>) outputs(%arg1 : memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC>
    // CHECK:       [[COPY4:%.+]] = VPUIP.Copy inputs([[COPY2]] : memref<1x16x56x56xui8, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x56x56xui8, #NHWC>) -> memref<1x16x56x56xui8, #NHWC>
    // CHECK:       return [[COPY3]], [[COPY4]] : memref<1x32x56x56xui8, #NHWC>, memref<1x16x56x56xui8, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FuseLastCopyWithMultiConcatView
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: memref<1x16x2x2xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: memref<1x16x2x2xf16>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
func.func @FuseLastCopyWithMultiConcatView(%arg0: memref<1x16x2x2xf16>, %arg1: memref<1x16x2x2xf16>, %arg2: memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16> {
    %0 = memref.alloc() : memref<1x64x2x2xf16>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16>) outputs(%1 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %3 = VPUIP.SubView %0 [0, 16, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %4 = VPUIP.Copy inputs(%arg1 : memref<1x16x2x2xf16>) outputs(%3 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    %5 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
    %6 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) outputs(%5 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>

    %7 = VPUIP.SubView %0 [0, 32, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %8 = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16>) outputs(%7 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %9 = VPUIP.SubView %0 [0, 48, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %10 = VPUIP.Copy inputs(%arg1 : memref<1x16x2x2xf16>) outputs(%9 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    %11 = VPUIP.SubView %0 [0, 32, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to
            memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
    %12 = VPUIP.ConcatView inputs(%8, %10 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
            outputs(%11 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>

    %13 = VPUIP.ConcatView inputs(%6, %12 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>)
            outputs(%0 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
    %14 = VPUIP.Copy inputs(%13 : memref<1x64x2x2xf16>) outputs(%arg2 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>

    return %14 : memref<1x64x2x2xf16>

    // CHECK-NOT:   memref.alloc() : memref<1x64x2x2xf16>

    // CHECK:   [[VAR0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK:   [[VAR2:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 16, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK:   [[VAR3:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK:   [[VAR4:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK:   [[VAR5:%.+]] = VPUIP.ConcatView inputs([[VAR1]], [[VAR3]] :
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    // CHECK:   [[VAR6:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 32, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK:   [[VAR7:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR6]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK:   [[VAR8:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 48, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK:   [[VAR9:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR8]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK:   [[VAR10:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 32, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK:   [[VAR11:%.+]] = VPUIP.ConcatView inputs([[VAR7]], [[VAR9]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK-SAME:      outputs([[VAR10]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK:   [[VAR12:%.+]] = VPUIP.ConcatView inputs([[VAR5]], [[VAR11]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK-SAME:      outputs([[OUTPUT]] : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:   return [[VAR12]] : memref<1x64x2x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseLastCopyWithConcatAndQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC>
func.func @FuseLastCopyWithConcatAndQuantizeCast(%arg0: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>,
                                                 %arg1: memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC> {
    %alloc = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
                    outputs(%0 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %2 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
                    outputs(%2 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>, memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
                          outputs(%alloc : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %5 = VPUIP.QuantizeCast inputs(%4 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56xui8, #NHWC, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x32x56x56xui8, #NHWC, @DDR>)
                    outputs(%arg1 : memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC>

    return %6 : memref<1x32x56x56xui8, #NHWC>

    // CHECK:   [[VAL0:%.+]] = VPUIP.QuantizeCast inputs([[OUTPUT]] : memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:   [[VAL1:%.+]] = VPUIP.SubView [[VAL0]] [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:   [[VAL2:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAL1]] : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    // CHECK:   [[VAL3:%.+]] = VPUIP.SubView [[VAL0]] [0, 16, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:   [[VAL4:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAL3]] : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    // CHECK-NOT:   VPUIP.ConcatView
    // CHECK-NOT:   VPUIP.QuantizeCast
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:   return [[OUTPUT]] : memref<1x32x56x56xui8, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseLastCopyWithQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC>
func.func @FuseLastCopyWithQuantizeCast(%arg0: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>,
                                        %arg1: memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC> {
    %0 = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
                       outputs(%0 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56xui8, #NHWC, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<1x32x56x56xui8, #NHWC, @DDR>)
                    outputs(%arg1 : memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC>

    return %3 : memref<1x32x56x56xui8, #NHWC>

    // CHECK:   [[VAL0:%.+]] = VPUIP.QuantizeCast  inputs([[OUTPUT]] : memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:   [[VAL1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.QuantizeCast
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[OUTPUT]] : memref<1x32x56x56xui8, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @FuseLastCopyWithPermuteCast
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: memref<1x131584x11x1xf16, @DDR>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: memref<1x131585x11x1xf16, @DDR>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x11x1x263169xf16, #NHWC, @DDR>
func.func @FuseLastCopyWithPermuteCast(%arg0: memref<1x131584x11x1xf16, @DDR>,
                                       %arg1: memref<1x131585x11x1xf16, @DDR>,
                                       %arg2: memref<1x11x1x263169xf16, #NHWC, @DDR>) -> (memref<1x11x1x263169xf16, #NHWC, @DDR>) {
    %0 = memref.alloc() : memref<1x263169x11x1xf16, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 131584, 11, 1] : memref<1x263169x11x1xf16, @DDR> to memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x131584x11x1xf16, @DDR>)
                    outputs(%1 : memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>) -> memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 131585, 11, 1] : memref<1x263169x11x1xf16, @DDR> to memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%arg1 : memref<1x131585x11x1xf16, @DDR>)
                    outputs(%3 : memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>) -> memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>, memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>)
                          outputs(%0 : memref<1x263169x11x1xf16, @DDR>) -> memref<1x263169x11x1xf16, @DDR>
    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%5 : memref<1x263169x11x1xf16, @DDR>) -> memref<1x11x1x263169xf16, #NHWC, @DDR>
    %7 = VPUIP.Copy inputs(%6 : memref<1x11x1x263169xf16, #NHWC, @DDR>)
                    outputs(%arg2 : memref<1x11x1x263169xf16, #NHWC, @DDR>) -> memref<1x11x1x263169xf16, #NHWC, @DDR>

    return %7 : memref<1x11x1x263169xf16, #NHWC, @DDR>

    // CHECK:   [[VAL0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[OUTPUT]] : memref<1x11x1x263169xf16, #NHWC, @DDR>) -> memref<1x263169x11x1xf16, @DDR>
    // CHECK:   [[VAL1:%.+]] = VPUIP.SubView [[VAL0]] [0, 0, 0, 0] [1, 131584, 11, 1] : memref<1x263169x11x1xf16, @DDR> to memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>
    // CHECK:   [[VAL2:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x131584x11x1xf16, @DDR>)
    // CHECK-SAME:      outputs([[VAL1]] : memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>) -> memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>

    // CHECK:   [[VAL3:%.+]] = VPUIP.SubView [[VAL0]] [0, 0, 0, 0] [1, 131585, 11, 1] : memref<1x263169x11x1xf16, @DDR> to memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>
    // CHECK:   [[VAL4:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x131585x11x1xf16, @DDR>)
    // CHECK-SAME:      outputs([[VAL3]] : memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>) -> memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>

    // CHECK:   return [[OUTPUT]] : memref<1x11x1x263169xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseLastCopyWithConcatAndGenericReshape
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x64x28x56x!qElemType, #NHWC>
func.func @FuseLastCopyWithConcatAndGenericReshape(%arg0: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>,
                                                   %arg1: memref<1x64x28x56x!qElemType, #NHWC>) -> memref<1x64x28x56x!qElemType, #NHWC> {
    %alloc = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
                    outputs(%0 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %2 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
                    outputs(%2 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>, memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
                          outputs(%alloc : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %5 = VPUIP.GenericReshape inputs(%4 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x64x28x56x!qElemType, #NHWC, @DDR>)
                    outputs(%arg1 : memref<1x64x28x56x!qElemType, #NHWC>) -> memref<1x64x28x56x!qElemType, #NHWC>
    return %6 : memref<1x64x28x56x!qElemType, #NHWC>

    // CHECK:   [[VAL0:%.+]] = VPUIP.GenericReshape inputs([[OUTPUT]] : memref<1x64x28x56x!qElemType, #NHWC>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:   [[VAL1:%.+]] = VPUIP.SubView [[VAL0]] [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:   [[VAL2:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%1 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    // CHECK:   [[VAL3:%.+]] = VPUIP.SubView [[VAL0]] [0, 16, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56x!qElemType, #NHWC, @DDR> to memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    // CHECK:   [[VAL4:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%3 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>) -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    // CHECK-NOT:   VPUIP.ConcatView
    // CHECK-NOT:   VPUIP.GenericReshape
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:   return [[OUTPUT]] : memref<1x64x28x56x!qElemType, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseLastCopyWithGenericReshape
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x64x28x56x!qElemType, #NHWC>
func.func @FuseLastCopyWithGenericReshape(%arg0: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>,
                                          %arg1: memref<1x64x28x56x!qElemType, #NHWC>) -> memref<1x64x28x56x!qElemType, #NHWC> {
    %0 = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<1x64x28x56x!qElemType, #NHWC, @DDR>)
                    outputs(%arg1 : memref<1x64x28x56x!qElemType, #NHWC>) -> memref<1x64x28x56x!qElemType, #NHWC>

    return %3 : memref<1x64x28x56x!qElemType, #NHWC>

    // CHECK:   [[VAL0:%.+]] = VPUIP.GenericReshape inputs([[OUTPUT]] : memref<1x64x28x56x!qElemType, #NHWC>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:   [[VAL1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.GenericReshape
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[OUTPUT]] : memref<1x64x28x56x!qElemType, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseLastCopyWithShapeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x64x28x56x!qElemType, #NHWC, @DDR>
func.func @FuseLastCopyWithShapeCast(%arg0: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>,
                                          %arg1: memref<1x64x28x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %2 = VPUIP.ShapeCast {shape = [1, 64, 28, 56]} inputs(%1 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<1x64x28x56x!qElemType, #NHWC, @DDR>)
                    outputs(%arg1 : memref<1x64x28x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR>

    return %3 : memref<1x64x28x56x!qElemType, #NHWC, @DDR>

    // CHECK:   [[VAL0:%.+]] = VPUIP.ShapeCast {shape = [1, 32, 56, 56]} inputs([[OUTPUT]] : memref<1x64x28x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:   [[VAL1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.ShapeCast
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[OUTPUT]] : memref<1x64x28x56x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x63x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: func.func @FuseLastCopyWithMultipleCastOps
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x63xf16, @DDR>
func.func @FuseLastCopyWithMultipleCastOps(%arg0 : !InputDistributedType, %arg1: memref<1x63xf16, @DDR>) -> memref<1x63xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 63, 1, 1] : !InputDistributedType to !OutputDistributedType
    %1 = memref.alloc() : memref<1x63x1x1xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy inputs(%0: !OutputDistributedType)
                                outputs(%1: memref<1x63x1x1xf16, #NHWC, @DDR>) -> memref<1x63x1x1xf16, #NHWC, @DDR>

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%2 : memref<1x63x1x1xf16, #NHWC, @DDR>) -> memref<1x63x1x1xf16, @DDR>
    %4 = VPUIP.GenericReshape inputs(%3 : memref<1x63x1x1xf16, @DDR>) -> memref<1x63xf16, @DDR>
    %5 = VPUIP.Copy inputs(%4 : memref<1x63xf16, @DDR>) outputs(%arg1 : memref<1x63xf16, @DDR>) -> memref<1x63xf16, @DDR>

    return %5 : memref<1x63xf16, @DDR>

    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 63, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x63x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[OUTPUT]] : memref<1x63xf16, @DDR>) -> memref<1x63x1x1xf16, @DDR>
    // CHECK:   [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<1x63x1x1xf16, @DDR>) -> memref<1x63x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_OP:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x63x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[PERMUTE_CAST]] : memref<1x63x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      -> memref<1x63x1x1xf16, #NHWC, @DDR>

    // CHECK:   return [[OUTPUT]] : memref<1x63xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: func.func @FuseLastCopyWithPermuteAndQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x1024x2048x1x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<2048x1024x1x1xui8, #NHWC, @DDR>) -> memref<2048x1024x1x1xui8, #NHWC, @DDR>
func.func @FuseLastCopyWithPermuteAndQuantizeCast(%arg0: memref<1x1024x2048x1x!qElemType, #NHWC, @CMX_NN>,
                                        %arg1: memref<2048x1024x1x1xui8, #NHWC, @DDR>) -> memref<2048x1024x1x1xui8, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1024x2048x1x!qElemType, #NHWC, @CMX_NN>)
                            outputs(%alloc : memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>)
                            -> memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>}
                            inputs(%0 : memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>) -> memref<2048x1024x1x1x!qElemType, #NHWC, @DDR>
    %2 = VPUIP.QuantizeCast inputs(%1 : memref<2048x1024x1x1x!qElemType, #NHWC, @DDR>) -> memref<2048x1024x1x1xui8, #NHWC, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<2048x1024x1x1xui8, #NHWC, @DDR>) outputs(%arg1 : memref<2048x1024x1x1xui8, #NHWC, @DDR>) -> memref<2048x1024x1x1xui8, #NHWC, @DDR>

    return %3 : memref<2048x1024x1x1xui8, #NHWC, @DDR>

    // CHECK:       [[QUANT_CAST:%.+]] = VPUIP.QuantizeCast inputs([[OUTPUT]] : memref<2048x1024x1x1xui8, #NHWC, @DDR>) -> memref<2048x1024x1x1x!qElemType, #NHWC, @DDR>
    // CHECK:       [[PERM_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #map} inputs([[QUANT_CAST]] :
    // CHECK-SAME:                               memref<2048x1024x1x1x!qElemType, #NHWC, @DDR>) -> memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>
    // CHECK:                            VPUIP.Copy inputs([[INPUT]] : memref<1x1024x2048x1x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:                              outputs([[PERM_CAST]] : memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>) -> memref<1x1024x2048x1x!qElemType, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.PermuteCast
    // CHECK-NOT:   VPUIP.QuantizeCast
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[OUTPUT]] : memref<2048x1024x1x1xui8, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseLastCopyWithShapeCastAndQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x64x28x56xui8, #NHWC, @DDR>
func.func @FuseLastCopyWithShapeCastAndQuantizeCast(%arg0: memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>,
                                          %arg1: memref<1x64x28x56xui8, #NHWC, @DDR>) -> memref<1x64x28x56xui8, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %2 = VPUIP.ShapeCast {shape = [1, 64, 28, 56]} inputs(%1 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.QuantizeCast inputs(%2 : memref<1x64x28x56x!qElemType, #NHWC, @DDR>) -> memref<1x64x28x56xui8, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x64x28x56xui8, #NHWC, @DDR>)
                    outputs(%arg1 : memref<1x64x28x56xui8, #NHWC, @DDR>) -> memref<1x64x28x56xui8, #NHWC, @DDR>

    return %4 : memref<1x64x28x56xui8, #NHWC, @DDR>

    // CHECK:       [[QUANT_CAST:%.+]] = VPUIP.QuantizeCast inputs([[OUTPUT]] : memref<1x64x28x56xui8, #NHWC, @DDR>) -> memref<1x64x28x56x!qElemType, #NHWC, @DDR>
    // CHECK:       [[PERM_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 32, 56, 56]} inputs([[QUANT_CAST]] : memref<1x64x28x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:                           VPUIP.Copy inputs([[INPUT]] : memref<1x32x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:                          outputs([[PERM_CAST]] : memref<1x32x56x56x!qElemType, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.ShapeCast
    // CHECK-NOT:   VPUIP.QuantizeCast
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[OUTPUT]] : memref<1x64x28x56xui8, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FuseLastCopyWithTwoCopies
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x8x2x2xf16, @DDR>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x32x2x2xf16, @DDR>
func.func @FuseLastCopyWithTwoCopies(%arg0: memref<1x8x2x2xf16, @DDR>, %arg1: memref<1x32x2x2xf16, @DDR>) -> (memref<1x8x2x2xf16, @DDR>, memref<1x32x2x2xf16, @DDR>) {
    %cst0 = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]
    %cst1 = const.Declare memref<1x32x2x2xf16, @DDR> = dense<2.000000e+00> : tensor<1x32x2x2xf16>

    %1 = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    %2 = VPUIP.Copy inputs(%cst0 : memref<1x8x2x2xf16, @DDR>) outputs(%1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<1x8x2x2xf16, @DDR>) outputs(%arg0 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    %4 = memref.alloc() : memref<1x32x2x2xf16, @DDR>
    %5 = VPUIP.Copy inputs(%cst1 : memref<1x32x2x2xf16, @DDR>) outputs(%4 : memref<1x32x2x2xf16, @DDR>) -> memref<1x32x2x2xf16, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x32x2x2xf16, @DDR>) outputs(%arg1 : memref<1x32x2x2xf16, @DDR>) -> memref<1x32x2x2xf16, @DDR>

    return %3, %6 : memref<1x8x2x2xf16, @DDR>, memref<1x32x2x2xf16, @DDR>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare memref<1x32x2x2xf16, @DDR> = dense<2.000000e+00> : tensor<1x32x2x2xf16>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]

    // CHECK:   [[VAR0:%.+]] = VPUIP.Copy inputs([[CST1]] : memref<1x8x2x2xf16, @DDR>) outputs([[INPUT]] : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[CST0]] : memref<1x32x2x2xf16, @DDR>) outputs([[OUTPUT]] : memref<1x32x2x2xf16, @DDR>) -> memref<1x32x2x2xf16, @DDR>

    // CHECK:   return [[VAR0]], [[VAR1]] : memref<1x8x2x2xf16, @DDR>, memref<1x32x2x2xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:       func.func @DoNotOptimizeLastCopies
// CHECK-SAME:      ([[IN:%.+]]: memref<6x6x12x24xf16, #NHWC, [@CMX_NN, 0]>, [[OUT_0:%.+]]: memref<3x12x24x6xf16, @DDR>, [[OUT_1:%.+]]: memref<3x12x24x6xf16, @DDR>)
// CHECK-SAME:      -> (memref<3x12x24x6xf16, @DDR>, memref<3x12x24x6xf16, @DDR>) {
func.func @DoNotOptimizeLastCopies(%in: memref<6x6x12x24xf16, #NHWC, [@CMX_NN, 0]>, %out0: memref<3x12x24x6xf16, @DDR>, %out1: memref<3x12x24x6xf16, @DDR>)
        -> (memref<3x12x24x6xf16, @DDR>, memref<3x12x24x6xf16, @DDR>) {
    %in_alloc = memref.alloc() : memref<6x6x12x24xf16, #NHWC, @DDR>
    %in_copy = VPUIP.Copy inputs(%in : memref<6x6x12x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%in_alloc : memref<6x6x12x24xf16, #NHWC, @DDR>) -> memref<6x6x12x24xf16, #NHWC, @DDR>

    %in_subview0 = VPUIP.SubView %in_copy [0, 0, 0, 0] [3, 6, 12, 24] : memref<6x6x12x24xf16, #NHWC, @DDR> to memref<3x6x12x24xf16, #NHWC, @DDR>
    %in_subview1 = VPUIP.SubView %in_copy [3, 0, 0, 0] [3, 6, 12, 24] : memref<6x6x12x24xf16, #NHWC, @DDR> to memref<3x6x12x24xf16, #NHWC, @DDR>

    %in_permute_cast0 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%in_subview0 : memref<3x6x12x24xf16, #NHWC, @DDR>) -> memref<3x12x24x6xf16, @DDR>
    %in_permute_cast1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%in_subview1 : memref<3x6x12x24xf16, #NHWC, @DDR>) -> memref<3x12x24x6xf16, @DDR>

    %out_copy0 = VPUIP.Copy inputs(%in_permute_cast0 : memref<3x12x24x6xf16, @DDR>) outputs(%out0 : memref<3x12x24x6xf16, @DDR>) -> memref<3x12x24x6xf16, @DDR>
    %out_copy1 = VPUIP.Copy inputs(%in_permute_cast1 : memref<3x12x24x6xf16, @DDR>) outputs(%out1 : memref<3x12x24x6xf16, @DDR>) -> memref<3x12x24x6xf16, @DDR>

    return %out_copy0, %out_copy1 : memref<3x12x24x6xf16, @DDR>, memref<3x12x24x6xf16, @DDR>

    // CHECK:       [[IN_ALLOC:%.+]] = memref.alloc() : memref<6x6x12x24xf16, #NHWC, @DDR>
    // CHECK:       [[IN_COPY:%.+]] = VPUIP.Copy inputs([[IN]]
    // CHECK-SAME:                               outputs([[IN_ALLOC]]

    // CHECK:       [[IN_SUBVIEW0:%.+]] = VPUIP.SubView [[IN_COPY]]
    // CHECK:       [[IN_SUBVIEW1:%.+]] = VPUIP.SubView [[IN_COPY]]

    // CHECK:       [[IN_PERMUTE_CAST0:%.+]] = VPUIP.PermuteCast
    // CHECK-SAME:      inputs([[IN_SUBVIEW0]]
    // CHECK:       [[IN_PERMUTE_CAST1:%.+]] = VPUIP.PermuteCast
    // CHECK-SAME:      inputs([[IN_SUBVIEW1]]

    // CHECK:       [[OUT_COPY0:%.+]] = VPUIP.Copy inputs([[IN_PERMUTE_CAST0]]
    // CHECK-SAME:      outputs([[OUT_0]]
    // CHECK:       [[OUT_COPY1:%.+]] = VPUIP.Copy inputs([[IN_PERMUTE_CAST1]]
    // CHECK-SAME:      outputs([[OUT_1]]

    // CHECK:       return [[OUT_COPY0]], [[OUT_COPY1]]
}

// -----

func.func @NoChangesSourceIsConstantOp(%arg0: memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16> {
    %0 = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x2x4x4xf16>) outputs(%arg0 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    return %1 : memref<1x2x4x4xf16>

    // CHECK-DAG: [[VAR0:%.+]] = const.Declare
    // CHECK: [[VAR1:%.+]] = VPUIP.Copy
    // CHECK: return [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NoChangesDifferentMemSpace(%arg0: memref<1x16x4x4xf16, #NHWC>, %arg1 : memref<16x1x1x4xsi32, @CMX_NN>,
                                 %arg2 : memref<16x1x1x16xui8, @CMX_NN>, %arg3: memref<1x16x2x2xf16, #NHWC>) -> memref<1x16x2x2xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16, #NHWC>) outputs(%0 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>) -> memref<1x16x4x4xf16, #NHWC, @CMX_NN>

    %2 = memref.alloc() : memref<1x16x2x2xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [2, 2],
            kernel_strides = [2, 2],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%1 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>)
        weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%1 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>)
        parent_output(%2 : memref<1x16x2x2xf16, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x2x2xf16, #NHWC, @CMX_NN>) -> memref<1x16x2x2xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 2, 2], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }

    %4 = VPUIP.Copy inputs(%3 : memref<1x16x2x2xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x2x2xf16, #NHWC>) -> memref<1x16x2x2xf16, #NHWC>
    return %4 : memref<1x16x2x2xf16, #NHWC>

    // CHECK: VPUIP.Copy

    // CHECK: [[VAR0:%.+]] = VPUIP.NCEClusterTask
    // CHECK: [[VAR1:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x16x2x2xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                      outputs(%arg3 : memref<1x16x2x2xf16, #NHWC>)

    // CHECK: return [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataCMXType = memref<1x16x4x4xf16, #NHWC, @CMX_NN>
!ISMCMXType = memref<1x16x4x4xi1, #NHWC, @CMX_NN>

// CHECK-LABEL: @NoChangesDifferentMemSpaceSparse
// CHECK-SAME: ([[IN_DATA:%.+]]: memref<1x16x4x4xf16, #NHWC>, [[IN_SM:%.+]]: memref<1x16x4x4xi1, #NHWC>,
// CHECK-SAME: {{%.+}}: memref<16x1x1x4xsi32, @CMX_NN>, {{%.+}}: memref<16x1x1x16xui8, @CMX_NN>,
// CHECK-SAME: [[OUT_DATA:%.+]]: memref<1x16x4x4xf16, #NHWC>, [[OUT_SM:%.+]]: memref<1x16x4x4xi1, #NHWC>)
func.func @NoChangesDifferentMemSpaceSparse(
        %arg0data: memref<1x16x4x4xf16, #NHWC>, %arg0sm: memref<1x16x4x4xi1, #NHWC>,
        %arg1 : memref<16x1x1x4xsi32, @CMX_NN>,
        %arg2 : memref<16x1x1x16xui8, @CMX_NN>,
        %arg3data: memref<1x16x4x4xf16, #NHWC>, %arg3sm: memref<1x16x4x4xi1, #NHWC>)
        -> (memref<1x16x4x4xf16, #NHWC>, memref<1x16x4x4xi1, #NHWC>) {
    %data_buff = memref.alloc() : !IDataCMXType
    %sm_buff = memref.alloc() : !ISMCMXType
    %in_data_0 = VPUIP.Copy
        inputs(%arg0data: memref<1x16x4x4xf16, #NHWC>)
        outputs(%data_buff: !IDataCMXType)
        -> !IDataCMXType
    %in_sm_0 = VPUIP.Copy
        inputs(%arg0sm: memref<1x16x4x4xi1, #NHWC>)
        outputs(%sm_buff: !ISMCMXType)
        -> !ISMCMXType

    %out_data_0 = memref.alloc() : !IDataCMXType
    %out_sm_0 = memref.alloc() : !ISMCMXType
    %mp:2 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
            kernel_size = [2, 2],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%in_data_0 : !IDataCMXType)
        input_sparsity_map(%in_sm_0 : !ISMCMXType)
        weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data_0 : !IDataCMXType)
        parent_input_sparsity_map(%in_sm_0 : !ISMCMXType)
        parent_output(%out_data_0 : !IDataCMXType)
        parent_output_sparsity_map(%out_sm_0 : !ISMCMXType)
        outputs(%out_data_0 : !IDataCMXType)
        output_sparsity_map(%out_sm_0 : !ISMCMXType) -> !IDataCMXType, !ISMCMXType
        variants :
        {
            DPUTask { outEnd = [16, 2, 2], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }

    %3 = VPUIP.Copy inputs(%mp#0 : !IDataCMXType) outputs(%arg3data : memref<1x16x4x4xf16, #NHWC>)
        -> memref<1x16x4x4xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%mp#1 : !ISMCMXType) outputs(%arg3sm : memref<1x16x4x4xi1, #NHWC>)
        -> memref<1x16x4x4xi1, #NHWC>
    return %3, %4 : memref<1x16x4x4xf16, #NHWC>, memref<1x16x4x4xi1, #NHWC>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x4x4xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x4x4xi1, #NHWC, @CMX_NN>

    // CHECK:       [[IN_COPY_DATA:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[IN_DATA]]
    // CHECK-SAME:      outputs([[BUFF_0_DATA]]
    // CHECK:       [[IN_COPY_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[IN_SM]]
    // CHECK-SAME:      outputs([[BUFF_0_SM]]

    // CHECK:       [[NCE_OUT_DATA:%.+]] = memref.alloc() : memref<1x16x4x4xf16, #NHWC, @CMX_NN>
    // CHECK:       [[NCE_OUT_SM:%.+]] = memref.alloc() : memref<1x16x4x4xi1, #NHWC, @CMX_NN>

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[IN_COPY_DATA]]
    // CHECK-SAME:          input_sparsity_map([[IN_COPY_SM]]
    // CHECK-SAME:          outputs([[NCE_OUT_DATA]]
    // CHECK-SAME:          output_sparsity_map([[NCE_OUT_SM]]

    // CHECK:       [[COPY_DATA:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCE_0]]#0
    // CHECK-SAME:      outputs([[OUT_DATA]]

    // CHECK:       [[COPY_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCE_0]]#1
    // CHECK-SAME:      outputs([[OUT_SM]]

    // CHECK:       return [[COPY_DATA]], [[COPY_SM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotFuseCopyWithPermuteCastOpWithMultipleUsers(%arg0: memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>,
                                                         %arg1: memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>,
                                                         %out0: memref<1x11x513x513xf16, #NHWC, @DDR>,
                                                         %out1: memref<1x11x256x513xf16, #NHWC, @DDR>)
                                                         -> (memref<1x11x513x513xf16, #NHWC, @DDR>, memref<1x11x256x513xf16, #NHWC, @DDR>) {
    %0 = memref.alloc() : memref<1x263169x11x1xf16, @DDR>
    %1 = VPUIP.ConcatView inputs(%arg0, %arg1 : memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>, memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>)
            outputs(%0 : memref<1x263169x11x1xf16, @DDR>) -> memref<1x263169x11x1xf16, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x263169x11x1xf16, @DDR>) -> memref<1x513x513x11xf16, @DDR>
    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%2 : memref<1x513x513x11xf16, @DDR>) -> memref<1x11x513x513xf16, #NHWC, @DDR>
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 11, 256, 513] : memref<1x11x513x513xf16, #NHWC, @DDR> to memref<1x11x256x513xf16, {order = #NHWC, strides = [2894859, 1, 5643, 11]}, @DDR>
    %5 = memref.alloc(): memref<1x11x256x513xf16, #NHWC, @CMX_NN>
    %6 =  VPUIP.Copy inputs(%4 : memref<1x11x256x513xf16, {order = #NHWC, strides = [2894859, 1, 5643, 11]}, @DDR>) outputs(%5 : memref<1x11x256x513xf16, #NHWC, @CMX_NN>) -> memref<1x11x256x513xf16, #NHWC, @CMX_NN>
    %7 = VPUIP.Copy inputs(%3 : memref<1x11x513x513xf16, #NHWC, @DDR>) outputs(%out0 : memref<1x11x513x513xf16, #NHWC, @DDR>) -> memref<1x11x513x513xf16, #NHWC, @DDR>
    %8 = VPUIP.Copy inputs(%6 : memref<1x11x256x513xf16, #NHWC, @CMX_NN>) outputs(%out1 : memref<1x11x256x513xf16, #NHWC, @DDR>) -> memref<1x11x256x513xf16, #NHWC, @DDR>
    return %7, %8: memref<1x11x513x513xf16, #NHWC, @DDR>, memref<1x11x256x513xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x263169x11x1xf16, @DDR>
    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs(%arg0, %arg1 : memref<1x131584x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>, memref<1x131585x11x1xf16, {order = #NCHW, strides = [2894859, 11, 1, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x263169x11x1xf16, @DDR>) -> memref<1x263169x11x1xf16, @DDR>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCATVIEW]] : memref<1x263169x11x1xf16, @DDR>) -> memref<1x513x513x11xf16, @DDR>
    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[RESHAPE]] : memref<1x513x513x11xf16, @DDR>) -> memref<1x11x513x513xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[PERMUTECAST]] [0, 0, 0, 0] [1, 11, 256, 513]
    // CHECK-SAME:      memref<1x11x513x513xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x11x256x513xf16, {order = #NHWC, strides = [2894859, 1, 5643, 11]}, @DDR>
    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x11x256x513xf16, #NHWC, @CMX_NN>
    // CHECK:       [[DISTRIBUTEDOP:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x11x256x513xf16, {order = #NHWC, strides = [2894859, 1, 5643, 11]}, @DDR>) outputs([[ALLOC1]] : memref<1x11x256x513xf16, #NHWC, @CMX_NN>) -> memref<1x11x256x513xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[PERMUTECAST]] : memref<1x11x513x513xf16, #NHWC, @DDR>) outputs(%arg2 : memref<1x11x513x513xf16, #NHWC, @DDR>) -> memref<1x11x513x513xf16, #NHWC, @DDR>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[DISTRIBUTEDOP]] : memref<1x11x256x513xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x11x256x513xf16, #NHWC, @DDR>) -> memref<1x11x256x513xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY0]], [[COPY1]] : memref<1x11x513x513xf16, #NHWC, @DDR>, memref<1x11x256x513xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @NotFuseCopyWitSubView
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x16x254x254xf16, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%.+]]: memref<1x16x128x128xf16, @DDR>
func.func @NotFuseCopyWitSubView(%arg0:  memref<1x16x254x254xf16, @CMX_NN>, %arg1: memref<1x16x128x128xf16, @DDR>) -> memref<1x16x128x128xf16, @DDR> {
    %alloc = memref.alloc() : memref<1x16x254x254xf16, @DDR>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x254x254xf16, @CMX_NN>) outputs(%alloc : memref<1x16x254x254xf16, @DDR>) -> memref<1x16x254x254xf16, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 63, 63] [1, 16, 128, 128] : memref<1x16x254x254xf16, @DDR> to memref<1x16x128x128xf16, {order = #NCHW, strides = [1032256, 64516, 254, 1]}, @DDR>
    %2 = VPUIP.Copy inputs(%1 : memref<1x16x128x128xf16, {order = #NCHW, strides = [1032256, 64516, 254, 1]}, @DDR>) outputs(%arg1 : memref<1x16x128x128xf16, @DDR>) -> memref<1x16x128x128xf16, @DDR>
    return %2 : memref<1x16x128x128xf16, @DDR>

    // CHECK:           [[ALLOC:%.+]] = memref.alloc() : memref<1x16x254x254xf16, @DDR>
    // CHECK:           [[COPY_RESULT:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x254x254xf16, @CMX_NN>) outputs([[ALLOC]] : memref<1x16x254x254xf16, @DDR>) -> memref<1x16x254x254xf16, @DDR>
    // CHECK:           [[SUB_VIEW:%.+]] = VPUIP.SubView [[COPY_RESULT]] [0, 0, 63, 63] [1, 16, 128, 128] :
    // CHECK-SAME:                          memref<1x16x254x254xf16, @DDR> to memref<1x16x128x128xf16, {order = #NCHW, strides = [1032256, 64516, 254, 1]}, @DDR>
    // CHECK:           [[DDR_COPY:%.+]] = VPUIP.Copy inputs([[SUB_VIEW]] : memref<1x16x128x128xf16, {order = #NCHW, strides = [1032256, 64516, 254, 1]}, @DDR>)
    // CHECK-SAME:                                      outputs([[OUTPUT]] : memref<1x16x128x128xf16, @DDR>) -> memref<1x16x128x128xf16, @DDR>
    // CHECK:           return [[DDR_COPY]] : memref<1x16x128x128xf16, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @FuseLastCopyWithoutViewOp
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_0:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_1:%arg[0-9]]]: memref<1x2x4x4xf16>) -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>)
func.func @FuseLastCopyWithoutViewOp(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>, %arg2: memref<1x2x4x4xf16>)
                            -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>
    %1 = memref.alloc() : memref<1x2x4x4xf16>
    %2 = memref.alloc() : memref<1x2x4x4xf16>

    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg3: memref<1x2x4x4xf16>) outputs(%1 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<1x2x4x4xf16>) outputs(%2 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }

    %5 = VPUIP.Copy inputs(%3 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    %6 = VPUIP.Copy inputs(%4 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %5, %6 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK-DAG:   [[VAR0:%.+]] = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>

    // CHECK-NOT:   memref.alloc() : memref<1x2x4x4xf16>
    // CHECK-NOT:   memref.alloc() : memref<1x2x4x4xf16>

    // CHECK:   [[VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[INPUT]] as {{[^:]+}}: memref<1x2x4x4xf16>) outputs([[OUTPUT_0]] as {{[^:]+}}: memref<1x2x4x4xf16>)
    // CHECK:   [[VAR2:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[VAR0]] as {{[^:]+}}: memref<1x2x4x4xf16>) outputs([[OUTPUT_1]] as {{[^:]+}}: memref<1x2x4x4xf16>)
    // CHECK:   return [[VAR1]], [[VAR2]] : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @NotFuseLastCopyChangesTypeMismatch
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x50x1x1xf16>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x50xf16>) -> memref<1x50xf16>
func.func @NotFuseLastCopyChangesTypeMismatch(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50xf16>) -> memref<1x50xf16> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg2: memref<1x50x1x1xf16>) outputs(%0 as %arg3: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
        }
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50xf16>) outputs(%arg1 : memref<1x50xf16>) -> memref<1x50xf16>
    return %3 : memref<1x50xf16>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[OUTPUT]] : memref<1x50xf16>) -> memref<1x50x1x1xf16>
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[INPUT]] as {{[^:]+}}: memref<1x50x1x1xf16>)
    // CHECK-SAME:                                                          outputs([[RESHAPE]] as {{[^:]+}}: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0]}({{[^:]+}}, {{[^:]+}}) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>

    // CHECK:       return [[OUTPUT]] : memref<1x50xf16>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @NotFuseLastCopyChangesInputIsBlockArgument
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_0:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_1:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_2:%arg[0-9]]]: memref<1x2x4x4xf16>) -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>)
func.func @NotFuseLastCopyChangesInputIsBlockArgument(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>,
                                    %arg2: memref<1x2x4x4xf16>, %arg3: memref<1x2x4x4xf16>) ->
                                    (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg4: memref<1x2x4x4xf16>) outputs(%arg2 as %arg5: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %2 = VPUIP.Copy inputs(%1 : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %0, %1, %2 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy
    // CHECK:       [[VAR1:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy
    // CHECK:       return [[VAR0]], [[VAR1]], [[VAR2]]
}
