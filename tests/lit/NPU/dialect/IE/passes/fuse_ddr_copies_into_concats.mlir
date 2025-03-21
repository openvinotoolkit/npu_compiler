//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-ddr-copies-into-concats %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.019874431572708431:128>

// CHECK-LABEL: @FuseTwoDDR2DDRCopies
func.func @FuseTwoDDR2DDRCopies(%LHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
                                %RHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    -> memref<1x64x250x250x!qElemType, #NHWC, @DDR> {
    // CMX2DDR transaction for the first input
    %LHS_DDR_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    %LHS_CMX2DDR_COPY = VPUIP.Copy
        inputs(%LHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%LHS_DDR_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @DDR>) -> memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // CMX2DDR transaction for the second input
    %RHS_DDR_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    %RHS_CMX2DDR_COPY = VPUIP.Copy
        inputs(%RHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%RHS_DDR_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @DDR>) -> memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // Concatenation
    %CONCAT_ALLOC = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
    %LHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 0, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %LHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%LHS_CMX2DDR_COPY : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%LHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 125, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%RHS_CMX2DDR_COPY : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%RHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %CONCAT = VPUIP.ConcatView
        inputs(%LHS_DDR2DDR_COPY,
               %RHS_DDR2DDR_COPY :
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>,
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
        outputs(%CONCAT_ALLOC : memref<1x64x250x250x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    return %CONCAT : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:        ([[LHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:    [[RHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:   [[CONCAT_ALLOC:%.*]] = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:   [[LHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [1, 64, 125, 250]
    // CHECK:   [[RHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 125, 0] [1, 64, 125, 250]

    // CHECK:   [[LHS_CMX2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[LHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[LHS_SUBVIEW]] : memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]}, @DDR>
    // CHECK-SAME:      ) -> memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]
    // CHECK-SAME:      }, @DDR>

    // CHECK:   [[RHS_CMX2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[RHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[RHS_SUBVIEW]] : memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]}, @DDR>
    // CHECK-SAME:      ) -> memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]
    // CHECK-SAME:      }, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[LHS_CMX2DDR_COPY]], [[RHS_CMX2DDR_COPY]]
    // CHECK:   return [[CONCAT]] : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.019874431572708431:128>

// CHECK-LABEL: @FuseLeftDDR2DDRCopy
func.func @FuseLeftDDR2DDRCopy(%LHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
                               %RHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
    -> memref<1x64x250x250x!qElemType, #NHWC, @DDR> {

    // Copy left input from CMX to DDR. Right input goes from DDR directly to VPUIP.ConcatView.
    // CMX2DDR transaction for the first input
    %LHS_DDR_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    %LHS_CMX2DDR_COPY = VPUIP.Copy
        inputs(%LHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%LHS_DDR_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @DDR>) -> memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // Concatenation
    %CONCAT_ALLOC = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
    %LHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 0, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %LHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%LHS_CMX2DDR_COPY : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%LHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 125, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%RHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%RHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %CONCAT = VPUIP.ConcatView
        inputs(%LHS_DDR2DDR_COPY,
               %RHS_DDR2DDR_COPY :
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>,
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
        outputs(%CONCAT_ALLOC : memref<1x64x250x250x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    return %CONCAT : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:        ([[LHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,

    // CHECK:   [[CONCAT_ALLOC:%.*]] = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:   [[LHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [1, 64, 125, 250]
    // CHECK:   [[RHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 125, 0] [1, 64, 125, 250]

    // CHECK:   [[RHS_INPUT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[RHS_SUBVIEW]]

    // CHECK:   [[LHS_CMX2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[LHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[LHS_SUBVIEW]] : memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]}, @DDR>
    // CHECK-SAME:      ) -> memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]
    // CHECK-SAME:      }, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[LHS_CMX2DDR_COPY]], [[RHS_INPUT]]
    // CHECK:   return [[CONCAT]] : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.019874431572708431:128>

// CHECK-LABEL: @FuseRightDDR2DDRCopy
func.func @FuseRightDDR2DDRCopy(%LHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @DDR>,
                                %RHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    -> memref<1x64x250x250x!qElemType, #NHWC, @DDR> {

    // Copy right input from CMX to DDR. Left input goes from DDR to directly VPUIP.ConcatView.
    // CMX2DDR transaction for the first input
    %RHS_DDR_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    %RHS_CMX2DDR_COPY = VPUIP.Copy
        inputs(%RHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%RHS_DDR_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @DDR>) -> memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // Concatenation
    %CONCAT_ALLOC = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    %LHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 0, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %LHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%LHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%LHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>
    %RHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 125, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%RHS_CMX2DDR_COPY : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%RHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %CONCAT = VPUIP.ConcatView
        inputs(%LHS_DDR2DDR_COPY,
               %RHS_DDR2DDR_COPY :
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>,
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
        outputs(%CONCAT_ALLOC : memref<1x64x250x250x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    return %CONCAT : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:        ([[LHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @DDR>,
    // CHECK-SAME:    [[RHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:   [[CONCAT_ALLOC:%.*]] = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:   [[LHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [1, 64, 125, 250]
    // CHECK:   [[LHS_INPUT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[LHS_SUBVIEW]]

    // CHECK:   [[RHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 125, 0] [1, 64, 125, 250]

    // CHECK:   [[RHS_CMX2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[RHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[RHS_SUBVIEW]] : memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]}, @DDR>
    // CHECK-SAME:      ) -> memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]
    // CHECK-SAME:      }, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[LHS_INPUT]], [[RHS_CMX2DDR_COPY]]
    // CHECK:   return [[CONCAT]] : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.019874431572708431:128>

// CHECK-LABEL: @SkipDDR2DDRClusterCopy
func.func @SkipDDR2DDRClusterCopy(%LHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @DDR>,
                                  %RHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
    -> memref<1x64x250x250x!qElemType, #NHWC, @DDR> {
    // Concatenation
    %CONCAT_ALLOC = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    %LHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 0, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 125, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    // Copy the first input directly to VPUIP.ConcatView
    %LHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%LHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%LHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    // Copy the second input directly to VPUIP.ConcatView
    %RHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%RHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%RHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %CONCAT = VPUIP.ConcatView
        inputs(%LHS_DDR2DDR_COPY,
               %RHS_DDR2DDR_COPY :
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>,
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
        outputs(%CONCAT_ALLOC : memref<1x64x250x250x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    return %CONCAT : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:   [[CONCAT_ALLOC:%.*]] = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
    // CHECK:   [[LHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [1, 64, 125, 250]
    // CHECK:   [[RHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 125, 0] [1, 64, 125, 250]

    // CHECK:   [[LHS_DDR2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[LHS_SUBVIEW]] : memref<1x64x125x250x!qElemType,

    // CHECK:   [[RHS_DDR2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[RHS_SUBVIEW]] : memref<1x64x125x250x!qElemType,

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[LHS_DDR2DDR_COPY]], [[RHS_DDR2DDR_COPY]]
    // CHECK:   return [[CONCAT]] : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.019874431572708431:128>

// CHECK-LABEL: @SkipCMX2CMXCopy
func.func @SkipCMX2CMXCopy(%LHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
                           %RHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    -> memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN> {

    // CMX2CMX transaction for the first input
    %LHS_CMX_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>
    %LHS_CMX2CMX_CLUSTER_COPY = VPUIP.Copy
        inputs(%LHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%LHS_CMX_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>

    // CMX2CMX transaction for the second input
    %RHS_CMX_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>
    %RHS_CMX2CMX_CLUSTER_COPY = VPUIP.Copy
        inputs(%RHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%RHS_CMX_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>

    // Concatenation
    %CONCAT_ALLOC = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>
    %LHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 0, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>

    %LHS_CMX2CMX_COPY = VPUIP.Copy
        inputs(%LHS_CMX2CMX_CLUSTER_COPY : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%LHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>

    %RHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 125, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>

    %RHS_CMX2CMX_COPY = VPUIP.Copy
        inputs(%RHS_CMX2CMX_CLUSTER_COPY : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%RHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>

    %CONCAT = VPUIP.ConcatView
        inputs(%LHS_CMX2CMX_COPY,
               %RHS_CMX2CMX_COPY :
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>,
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @CMX_NN>)
        outputs(%CONCAT_ALLOC : memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>)
            -> memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>

    return %CONCAT : memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>

    // CHECK:        ([[LHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:    [[RHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:   [[LHS_CMX_ALLOC:%.*]] = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[LHS_CMX2CMX_CLUSTER_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[LHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[LHS_CMX_ALLOC]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:   [[RHS_CMX_ALLOC:%.*]] = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[RHS_CMX2CMX_CLUSTER_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[RHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[RHS_CMX_ALLOC]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:   [[CONCAT_ALLOC:%.*]] = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[LHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [1, 64, 125, 250]

    // CHECK:   [[LHS_CMX2CMX_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[LHS_CMX2CMX_CLUSTER_COPY]]
    // CHECK-SAME:  outputs([[LHS_SUBVIEW]]

    // CHECK:   [[RHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 125, 0] [1, 64, 125, 250]

    // CHECK:   [[RHS_CMX2CMX_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[RHS_CMX2CMX_CLUSTER_COPY]]
    // CHECK-SAME:  outputs([[RHS_SUBVIEW]]

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[LHS_CMX2CMX_COPY]], [[RHS_CMX2CMX_COPY]]
    // CHECK:   return [[CONCAT]] : memref<1x64x250x250x!qElemType, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.019874431572708431:128>

// CHECK-LABEL: @DoNotEraseClusterTask
func.func @DoNotEraseClusterTask(%LHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
                                 %RHS_ARG: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    -> (memref<1x64x250x250x!qElemType, #NHWC, @DDR>, memref<1x64x125x250x!qElemType, #NHWC, @DDR>) {

    // CMX2DDR transaction for the first input
    %LHS_DDR_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    %LHS_CMX2DDR_COPY = VPUIP.Copy
        inputs(%LHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%LHS_DDR_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @DDR>) -> memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // CMX2DDR transaction for the second input
    %RHS_DDR_ALLOC = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    %RHS_CMX2DDR_COPY = VPUIP.Copy
        inputs(%RHS_ARG : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
        outputs(%RHS_DDR_ALLOC : memref<1x64x125x250x!qElemType, #NHWC, @DDR>) -> memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // Concatenation
    %CONCAT_ALLOC = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>
    %LHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 0, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %LHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%LHS_CMX2DDR_COPY : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%LHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_SUBVIEW = VPUIP.SubView %CONCAT_ALLOC [0, 0, 125, 0] [1, 64, 125, 250] :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>
        to memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %RHS_DDR2DDR_COPY = VPUIP.Copy
        inputs(%RHS_CMX2DDR_COPY : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)
        outputs(%RHS_SUBVIEW : memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
            -> memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %CONCAT = VPUIP.ConcatView
        inputs(%LHS_DDR2DDR_COPY,
               %RHS_DDR2DDR_COPY :
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>,
               memref<1x64x125x250x!qElemType, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
        outputs(%CONCAT_ALLOC : memref<1x64x250x250x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    return %CONCAT, %LHS_CMX2DDR_COPY :
        memref<1x64x250x250x!qElemType, #NHWC, @DDR>,
        memref<1x64x125x250x!qElemType, #NHWC, @DDR>

    // CHECK:        ([[LHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:    [[RHS_INPUT:%arg.*]]: memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:   [[LHS_DDR_ALLOC:%.*]] = memref.alloc() : memref<1x64x125x250x!qElemType, #NHWC, @DDR>
    // CHECK:   [[LHS_CMX2DDR_ORIG_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[LHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[LHS_DDR_ALLOC]] : memref<1x64x125x250x!qElemType, #NHWC, @DDR>)

    // CHECK:   [[CONCAT_ALLOC:%.*]] = memref.alloc() : memref<1x64x250x250x!qElemType, #NHWC, @DDR>

    // CHECK:   [[LHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [1, 64, 125, 250]
    // CHECK:   [[RHS_SUBVIEW:%.*]] = VPUIP.SubView [[CONCAT_ALLOC]] [0, 0, 125, 0] [1, 64, 125, 250]

    // CHECK:   [[LHS_CMX2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[LHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[LHS_SUBVIEW]] : memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]}, @DDR>
    // CHECK-SAME:      ) -> memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]
    // CHECK-SAME:      }, @DDR>

    // CHECK:   [[RHS_CMX2DDR_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[RHS_INPUT]] : memref<1x64x125x250x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[RHS_SUBVIEW]] : memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]}, @DDR>
    // CHECK-SAME:      ) -> memref<1x64x125x250x!qElemType, {
    // CHECK-SAME:          order = #NHWC,
    // CHECK-SAME:          strides = [4000000, 1, 16000, 64]
    // CHECK-SAME:      }, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[LHS_CMX2DDR_COPY]], [[RHS_CMX2DDR_COPY]]
    // CHECK:   return [[CONCAT]], [[LHS_CMX2DDR_ORIG_COPY]]
}
