//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-strided-dmas %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x24x1x1xui8, @DDR>
!FlatCMXType = memref<1x24x1x1xui8, @CMX_NN>

net.NetworkInfo entryPoint : @LegalizeStridedDmas inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x6xui8> {dynamicStrides}
    DataInfo "Parameter_2" : tensor<4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: func.func @LegalizeStridedDmas
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME:  [[ARG_2:%[^:]+]]: memref<4x6xui8, @DDR>
func.func @LegalizeStridedDmas(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.GenericReshape inputs(%arg1 : !DDRType) -> !FlatDDRType
    %2 = memref.alloc() : !FlatCMXType
    %3 = VPUIP.NNDMA inputs(%0 : !FlatDDRType) outputs(%2 : !FlatCMXType) -> !FlatCMXType
    %4 = memref.alloc() : !FlatCMXType
    %5 = VPUIP.NNDMA inputs(%1 : !FlatDDRType) outputs(%4 : !FlatCMXType) -> !FlatCMXType

    %7 = VPUIP.GenericReshape inputs(%arg2 : !DDRType) -> !FlatDDRType
    %8 = VPUIP.NNDMA inputs(%2 : !FlatCMXType) outputs(%7 : !FlatDDRType) -> !FlatDDRType
    return %arg2 : !DDRType

    // CHECK:   [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_2:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs([[ARG_2]]
    // CHECK-SAME:  outputs([[ALLOC_OUTPUT]]

    // CHECK:   [[ALLOC_INPUT1:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_1:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs([[ARG_1]] : memref<4x6xui8, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT1]] : memref<4x6xui8, @DDR>)

    // CHECK:   [[ALLOC_INPUT0:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_0:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs([[ARG_0]] : memref<4x6xui8, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT0]] : memref<4x6xui8, @DDR>)

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = VPUIP.GenericReshape inputs([[NNDMA_2]] : memref<4x6xui8, @DDR>) -> memref<1x24x1x1xui8, @DDR>
    // CHECK-NEXT:   [[INCOMPATIBLE_OUT_DMA:%.+]] = VPUIP.NNDMA
    // CHECK:   [[CONCAT_VIEW_OUT:%.+]] = VPUIP.ConcatView inputs([[ARG_2]]
    // CHECK-SAME:                                                [[INCOMPATIBLE_OUT_DMA]]
    // CHECK-SAME:                                                [[NNDMA_2]]
    // CHECK-SAME:                                                outputs([[ALLOC_OUTPUT]]
    // CHECK-NEXT:   VPUIP.NNDMA {stridedOutput} inputs([[CONCAT_VIEW_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x24x1x1xui8, @DDR>
!FlatCMXType = memref<1x24x1x1xui8, @CMX_NN>

net.NetworkInfo entryPoint : @LegalizeStridedDmasOneInput inputsInfo : {
    DataInfo "Parameter_1" : tensor<1x4x6xui8>
    DataInfo "Parameter_2" : tensor<1x4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<1x4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: func.func @LegalizeStridedDmasOneInput
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME:  [[ARG_2:%[^:]+]]: memref<4x6xui8, @DDR>
func.func @LegalizeStridedDmasOneInput(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.GenericReshape inputs(%arg1 : !DDRType) -> !FlatDDRType
    %2 = memref.alloc() : !FlatCMXType
    %3 = VPUIP.NNDMA inputs(%0 : !FlatDDRType) outputs(%2 : !FlatCMXType) -> !FlatCMXType
    %4 = memref.alloc() : !FlatCMXType
    %5 = VPUIP.NNDMA inputs(%1 : !FlatDDRType) outputs(%4 : !FlatCMXType) -> !FlatCMXType

    %7 = VPUIP.GenericReshape inputs(%arg2 : !DDRType) -> !FlatDDRType
    %8 = VPUIP.NNDMA inputs(%2 : !FlatCMXType) outputs(%7 : !FlatDDRType) -> !FlatDDRType
    return %arg2 : !DDRType

    // CHECK:   [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_2:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs([[ARG_2]]
    // CHECK-SAME:  outputs([[ALLOC_OUTPUT]]

    // CHECK:   [[ALLOC_INPUT0:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_0:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs([[ARG_1]] : memref<4x6xui8, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT0]] : memref<4x6xui8, @DDR>)

    // CHECK:   [[ARG0_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]]
    // CHECK-NOT: VPUIP.NNDMA {stridedInput} inputs([[ARG0_RESHAPE]]

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = VPUIP.GenericReshape inputs([[NNDMA_2]] : memref<4x6xui8, @DDR>) -> memref<1x24x1x1xui8, @DDR>
    // CHECK:   [[INCOMPATIBLE_OUT_DMA:%.+]] = VPUIP.NNDMA
    // CHECK:   [[CONCAT_VIEW_OUT:%.+]] = VPUIP.ConcatView inputs([[ARG_2]]
    // CHECK-SAME:                                                [[INCOMPATIBLE_OUT_DMA]]
    // CHECK-SAME:                                                [[NNDMA_2]]
    // CHECK-SAME:                                                outputs([[ALLOC_OUTPUT]]
    // CHECK:   VPUIP.NNDMA {stridedOutput} inputs([[CONCAT_VIEW_OUT]] : memref<4x6xui8, @DDR>) outputs([[ARG_2]] : memref<4x6xui8, @DDR>) -> memref<4x6xui8, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x24x1x1xui8, @DDR>
!FlatCMXType = memref<1x24x1x1xui8, @CMX_NN>

net.NetworkInfo entryPoint : @NoStridedCopies inputsInfo : {
    DataInfo "Parameter_1" : tensor<1x4x6xui8>
    DataInfo "Parameter_2" : tensor<1x4x6xui8>
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<1x4x6xui8>
}

// CHECK-LABEL: @NoStridedCopies
func.func @NoStridedCopies(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.GenericReshape inputs(%arg1 : !DDRType) -> !FlatDDRType
    %2 = memref.alloc() : !FlatCMXType
    %3 = VPUIP.NNDMA inputs(%0 : !FlatDDRType) outputs(%2 : !FlatCMXType) -> !FlatCMXType
    %4 = memref.alloc() : !FlatCMXType
    %5 = VPUIP.NNDMA inputs(%1 : !FlatDDRType) outputs(%4 : !FlatCMXType) -> !FlatCMXType

    %7 = VPUIP.GenericReshape inputs(%arg2 : !DDRType) -> !FlatDDRType
    %8 = VPUIP.NNDMA inputs(%2 : !FlatCMXType) outputs(%7 : !FlatDDRType) -> !FlatDDRType
    return %arg2 : !DDRType

    // CHECK:       VPUIP.NNDMA
    // CHECK-NOT:       {stridedInput} inputs({{%.+}} : memref<1x4x6xui8,    @DDR>)
    // CHECK:                                 inputs({{%.+}} : memref<1x24x1x1xui8, @DDR>)

    // CHECK:       VPUIP.NNDMA
    // CHECK-NOT:       {stridedInput} inputs({{%.+}} : memref<1x4x6xui8,    @DDR>)
    // CHECK:                                 inputs({{%.+}} : memref<1x24x1x1xui8, @DDR>)

    // CHECK:       VPUIP.NNDMA
    // CHECK-NOT:       {stridedOutput} inputs({{%.*}} : memref<1x4x6xui8,    @DDR>) outputs({{%.+}} : memref<4x6xui8, @DDR>)
    // CHECK:                                  inputs({{%.*}} : memref<1x24x1x1xui8, @CMX_NN>
    // CHECK:                                  outputs({{%.+}} : memref<1x24x1x1xui8, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!DDRType = memref<1x4x64x320xf32, @DDR>
!DDRTypeOut = memref<1x4x64x320xf16, @DDR>
!CMXType = memref<1x4x64x320xf16, @CMX_NN>

net.NetworkInfo entryPoint : @LabelStridedCopies inputsInfo : {
    DataInfo "Parameter_233593" : tensor<1x4x64x320xf32> {dynamicStrides}
    DataInfo "Parameter_233594" : tensor<1x4x64x320xf32> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

// CHECK-LABEL: @LabelStridedCopies
func.func @LabelStridedCopies(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRTypeOut) -> !DDRTypeOut {
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.ConvertDMA  inputs(%arg0 : !DDRType) outputs(%1 : !CMXType) -> !CMXType
    %3 = memref.alloc() : !CMXType
    %4 = VPUIP.ConvertDMA  inputs(%arg1 :!DDRType) outputs(%3 : !CMXType) -> !CMXType

    %11 = VPUIP.NNDMA inputs(%2 : !CMXType) outputs(%arg2 : !DDRTypeOut) -> !DDRTypeOut
    return %11 : !DDRTypeOut

    // CHECK: VPUIP.ConvertDMA {stridedInput}
    // CHECK: VPUIP.ConvertDMA {stridedInput}
    // CHECK: VPUIP.NNDMA {stridedOutput}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x12x1x1xui8, @DDR>
!FlatCMXType = memref<1x12x1x1xui8, @CMX_NN>

net.NetworkInfo entryPoint : @LegalizeStridedDmasMultipleIncompatibleOutputDmas inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x6xui8>
    DataInfo "Parameter_2" : tensor<4x6xui8>
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: func.func @LegalizeStridedDmasMultipleIncompatibleOutputDmas
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME:  [[ARG_2:%[^:]+]]: memref<4x6xui8, @DDR>
func.func @LegalizeStridedDmasMultipleIncompatibleOutputDmas(%arg0: memref<4x6xui8, @DDR>, %arg1: memref<4x6xui8, @DDR>, %arg2: memref<4x6xui8, @DDR>) -> memref<4x6xui8, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %1 = VPUIP.SubView %arg0 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %2 = VPUIP.SubView %arg1 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %3 = VPUIP.SubView %arg1 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %4 = VPUIP.SubView %arg2 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %5 = VPUIP.SubView %arg2 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %6 = VPUIP.GenericReshape inputs(%0 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %7 = VPUIP.GenericReshape inputs(%1 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %8 = VPUIP.GenericReshape inputs(%2 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %9 = VPUIP.GenericReshape inputs(%3 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %10 = VPUIP.GenericReshape inputs(%4 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %11 = VPUIP.GenericReshape inputs(%5 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %alloc = memref.alloc() : !FlatCMXType
    %alloc_1 = memref.alloc() : !FlatCMXType
    %alloc_2 = memref.alloc() : !FlatCMXType
    %alloc_3 = memref.alloc() : !FlatCMXType
    %12 = VPUIP.NNDMA  inputs(%6 : !FlatDDRType) outputs(%alloc :!FlatCMXType) -> !FlatCMXType
    %13 = VPUIP.NNDMA  inputs(%7 : !FlatDDRType) outputs(%alloc_1 : !FlatCMXType) -> !FlatCMXType
    %14 = VPUIP.NNDMA  inputs(%8 : !FlatDDRType) outputs(%alloc_2 : !FlatCMXType) -> !FlatCMXType
    %15 = VPUIP.NNDMA  inputs(%9 : !FlatDDRType) outputs(%alloc_3 : !FlatCMXType) -> !FlatCMXType

    %20 = VPUIP.NNDMA  inputs(%alloc :!FlatCMXType) outputs(%10 : memref<1x12x1x1xui8, @DDR>) -> !FlatDDRType
    %21 = VPUIP.NNDMA  inputs(%alloc : !FlatCMXType) outputs(%11 : memref<1x12x1x1xui8, @DDR>) -> !FlatDDRType
    return %arg2 : memref<4x6xui8, @DDR>

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ARG_2]] [0, 0]
    // CHECK-NEXT: [[ALLOC_OUT0:%.+]] = memref.alloc() : memref<2x6xui8, @DDR>
    // CHECK-NEXT: [[NNDMA_0:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:      inputs([[SUBVIEW_0]]
    // CHECK-SAME:      outputs([[ALLOC_OUT0]]

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ARG_2]] [2, 0]
    // CHECK-NEXT: [[ALLOC_OUT1:%.+]] = memref.alloc() : memref<2x6xui8, @DDR>
    // CHECK-NEXT: [[NNDMA_1:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:      inputs([[SUBVIEW_1]]
    // CHECK-SAME:      outputs([[ALLOC_OUT1]]

    // CHECK: [[RESHAPED_OUT0:%.+]] = VPUIP.GenericReshape inputs([[NNDMA_0]]
    // CHECK: [[RESHAPED_OUT1:%.+]] = VPUIP.GenericReshape inputs([[NNDMA_1]]
    // CHECK: [[CMX_ALLOC0:%.+]] = memref.alloc() : memref<1x12x1x1xui8, @CMX_NN>
    // CHECK:      [[INCOMPATIBLE_OUT_DMA0:%.+]] = VPUIP.NNDMA inputs([[CMX_ALLOC0]]
    // CHECK-SAME:                                             outputs([[RESHAPED_OUT0]]
    // CHECK-NEXT: [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[INCOMPATIBLE_OUT_DMA0]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT0]]
    // CHECK-NEXT: [[INCOMPATIBLE_OUT_DMA1:%.+]] = VPUIP.NNDMA inputs([[CMX_ALLOC0]]
    // CHECK-SAME:                                             outputs([[RESHAPED_OUT1]]
    // CHECK-NEXT: [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[INCOMPATIBLE_OUT_DMA1]]
    // CHECK-SAME:                                    outputs([[ALLOC_OUT1]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT1]]
  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!CMXSlice = memref<2x6xui8, @CMX_NN>
!DDRSlice = memref<2x6xui8, @DDR>

net.NetworkInfo entryPoint : @LegalizeStridedDmasWithConcatInBetween inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<2x6xui8> {dynamicStrides}
}

// Below test case checks if a ViewOp that can be reached from function argument in 2 different ways is handled correctly
// CHECK-LABEL: func.func @LegalizeStridedDmasWithConcatInBetween
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<2x6xui8, @DDR>
func.func @LegalizeStridedDmasWithConcatInBetween(%arg0: memref<4x6xui8, @DDR>, %arg1: !DDRSlice) -> !DDRSlice {
    %sub0 = VPUIP.SubView %arg0 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to !DDRSlice
    %sub1 = VPUIP.SubView %arg0 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to !DDRSlice
    %concat = VPUIP.ConcatView inputs(%sub0, %sub1 : !DDRSlice, !DDRSlice)
                               outputs(%sub0 : !DDRSlice) -> !DDRSlice
    %cmx_0 = memref.alloc() : !CMXSlice
    %cmx_1 = memref.alloc() : !CMXSlice
    %0 = VPUIP.NNDMA inputs(%concat : !DDRSlice) outputs(%cmx_0 : !CMXSlice) -> !CMXSlice
    %1 = VPUIP.NNDMA inputs(%concat : !DDRSlice) outputs(%cmx_1 : !CMXSlice) -> !CMXSlice

    %2 = VPUIP.NNDMA inputs(%cmx_0 : !CMXSlice) outputs(%arg1 :!DDRSlice) -> !DDRSlice
    return %arg1 : !DDRSlice

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK: [[ALLOC_CMX:%.+]] = memref.alloc() : memref<2x6xui8, @CMX_NN>
    // CHECK: VPUIP.NNDMA {stridedInput}  inputs([[CONCAT]] : memref<2x6xui8, @DDR>) outputs({{%[^:]+}} : memref<2x6xui8, @CMX_NN>) -> memref<2x6xui8, @CMX_NN>
    // CHECK: VPUIP.NNDMA {stridedInput}  inputs([[CONCAT]] : memref<2x6xui8, @DDR>) outputs({{%[^:]+}} : memref<2x6xui8, @CMX_NN>) -> memref<2x6xui8, @CMX_NN>
    // CHECK: VPUIP.NNDMA {stridedOutput}  inputs({{%[^:]+}} : memref<2x6xui8, @CMX_NN>) outputs([[ARG_1]] : memref<2x6xui8, @DDR>) -> memref<2x6xui8, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!DDRType = memref<4x6x1xui8, @DDR>
!ReshapedDDRType = memref<1x4x1x6xui8, @DDR>
!CMXType = memref<1x4x1x6xui8, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesDmasUnitExpandContract inputsInfo : {
    DataInfo "Parameter_233593" : tensor<4x6x1xui8> {dynamicStrides}
    DataInfo "Parameter_233594" : tensor<4x6x1xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<4x6x1xui8> {dynamicStrides}
}

// CHECK-LABEL: func.func @DynamicStridesDmasUnitExpandContract
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<4x6x1xui8, @DDR>
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<4x6x1xui8, @DDR>
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<4x6x1xui8, @DDR>
func.func @DynamicStridesDmasUnitExpandContract(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %1 = memref.alloc() : !CMXType
    %reshaped_in0 = VPUIP.GenericReshape inputs(%arg0: !DDRType) -> !ReshapedDDRType
    %2 = VPUIP.NNDMA inputs(%reshaped_in0 : !ReshapedDDRType) outputs(%1 : !CMXType) -> !CMXType
    %3 = memref.alloc() : !CMXType
    %reshaped_in1 = VPUIP.GenericReshape inputs(%arg1: !DDRType) -> !ReshapedDDRType
    %4 = VPUIP.NNDMA  inputs(%reshaped_in1 :!ReshapedDDRType) outputs(%3 : !CMXType) -> !CMXType

    %reshaped_out = VPUIP.GenericReshape inputs(%arg2: !DDRType) -> !ReshapedDDRType
    %11 = VPUIP.NNDMA inputs(%3 : !CMXType) outputs(%reshaped_out : !ReshapedDDRType) -> !ReshapedDDRType
    return %arg2 : !DDRType

    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<1x4x1x6xui8, @CMX_NN>
    // CHECK-NEXT: [[RESHAPED_IN0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN0]]
    // CHECK-SAME:                              outputs([[ALLOC]]
    // CHECK-NEXT: [[ALLOC_0:%.+]] = memref.alloc() : memref<1x4x1x6xui8, @CMX_NN>
    // CHECK-NEXT: [[RESHAPED_IN1:%.+]] = VPUIP.GenericReshape inputs([[ARG_1]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN1]]
    // CHECK-SAME:                              outputs([[ALLOC_0]]
    // CHECK-NEXT: [[RESHAPED_OUT:%.+]] = VPUIP.GenericReshape inputs([[ARG_2]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[ALLOC_0]]
    // CHECK-SAME:                              outputs([[RESHAPED_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!DDRType = memref<4x6xui8, @DDR>
!ReshapedDDRType = memref<1x1x4x6xui8>
!CMXType = memref<1x1x4x6xui8, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesDmasUnitExpandOnly inputsInfo : {
    DataInfo "Parameter_233593" : tensor<4x6xui8> {dynamicStrides}
    DataInfo "Parameter_233594" : tensor<4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: func.func @DynamicStridesDmasUnitExpandOnly
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<4x6xui8, @DDR>
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<4x6xui8, @DDR>
func.func @DynamicStridesDmasUnitExpandOnly(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %reshaped0 = VPUIP.GenericReshape inputs(%arg0: !DDRType) -> !ReshapedDDRType
    %reshaped1 = VPUIP.GenericReshape inputs(%arg1: !DDRType) -> !ReshapedDDRType
    %reshapedOut = VPUIP.GenericReshape inputs(%arg2: !DDRType) -> !ReshapedDDRType
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.NNDMA inputs(%reshaped0 : !ReshapedDDRType) outputs(%1 : !CMXType) -> !CMXType
    %3 = memref.alloc() : !CMXType
    %4 = VPUIP.NNDMA  inputs(%reshaped1 :!ReshapedDDRType) outputs(%3 : !CMXType) -> !CMXType

    %11 = VPUIP.NNDMA inputs(%3 : !CMXType) outputs(%reshapedOut : !ReshapedDDRType) -> !ReshapedDDRType
    return %arg2 : !DDRType

    // CHECK:      [[RESHAPED_IN0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]]
    // CHECK-NEXT: [[RESHAPED_IN1:%.+]] = VPUIP.GenericReshape inputs([[ARG_1]]
    // CHECK-NEXT: [[RESHAPED_OUT:%.+]] = VPUIP.GenericReshape inputs([[ARG_2]]
    // CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc()
    // CHECK-SAME: @CMX_NN
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN0]]
    // CHECK-SAME:                          outputs([[ALLOC]]
    // CHECK-NEXT: [[ALLOC1:%.+]] = memref.alloc()
    // CHECK-SAME: @CMX_NN
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN1]]
    // CHECK-SAME:                              outputs([[ALLOC1]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[ALLOC1]]
    // CHECK-SAME:                              outputs([[RESHAPED_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!DDRType = memref<1x1x4x6xui8, @DDR>
!ReshapedDDRType = memref<4x6xui8>
!CMXType = memref<4x6xui8, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesDmasUnitContractOnly inputsInfo : {
    DataInfo "Parameter_233593" : tensor<1x1x4x6xui8> {dynamicStrides}
    DataInfo "Parameter_233594" : tensor<1x1x4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x1x4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: func.func @DynamicStridesDmasUnitContractOnly
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x1x4x6xui8, @DDR>
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x1x4x6xui8, @DDR>
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<1x1x4x6xui8, @DDR>
func.func @DynamicStridesDmasUnitContractOnly(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %reshaped0 = VPUIP.GenericReshape inputs(%arg0: !DDRType) -> !ReshapedDDRType
    %reshaped1 = VPUIP.GenericReshape inputs(%arg1: !DDRType) -> !ReshapedDDRType
    %reshapedOut = VPUIP.GenericReshape inputs(%arg2: !DDRType) -> !ReshapedDDRType
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.NNDMA inputs(%reshaped0 : !ReshapedDDRType) outputs(%1 : !CMXType) -> !CMXType
    %3 = memref.alloc() : !CMXType
    %4 = VPUIP.NNDMA  inputs(%reshaped1 :!ReshapedDDRType) outputs(%3 : !CMXType) -> !CMXType

    %11 = VPUIP.NNDMA inputs(%3 : !CMXType) outputs(%reshapedOut : !ReshapedDDRType) -> !ReshapedDDRType
    return %arg2 : !DDRType

    // CHECK:      [[RESHAPED_IN0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]]
    // CHECK-NEXT: [[RESHAPED_IN1:%.+]] = VPUIP.GenericReshape inputs([[ARG_1]]
    // CHECK-NEXT: [[RESHAPED_OUT:%.+]] = VPUIP.GenericReshape inputs([[ARG_2]]
    // CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc()
    // CHECK-SAME: @CMX_NN
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN0]]
    // CHECK-SAME:                          outputs([[ALLOC]]
    // CHECK-NEXT: [[ALLOC1:%.+]] = memref.alloc()
    // CHECK-SAME: @CMX_NN
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN1]]
    // CHECK-SAME:                              outputs([[ALLOC1]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[ALLOC1]]
    // CHECK-SAME:                              outputs([[RESHAPED_OUT]]
}

// -----

!DDRType = memref<1xui8, @DDR>
!ReshapedDDRType = memref<1x1x1xui8, @DDR>
!CMXType = memref<1x1x1xui8, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesDmasUnitShapeExpand inputsInfo : {
    DataInfo "Parameter_233593" : tensor<1xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1xui8>
}

// CHECK-LABEL: func.func @DynamicStridesDmasUnitShapeExpand
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1xui8, @DDR>
func.func @DynamicStridesDmasUnitShapeExpand(%arg0: !DDRType, %arg2: !DDRType) -> !DDRType {
    %reshaped0 = VPUIP.GenericReshape inputs(%arg0: !DDRType) -> !ReshapedDDRType
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.NNDMA inputs(%reshaped0 : !ReshapedDDRType) outputs(%1 : !CMXType) -> !CMXType

    return %arg2 : !DDRType

    // CHECK:      [[RESHAPED_IN0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]]
    // CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc()
    // CHECK-SAME: @CMX_NN
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[RESHAPED_IN0]]
    // CHECK-SAME:                          outputs([[ALLOC]]
}

// -----

!DDRType = memref<4x1x2x4xui8, @DDR>
!DDRSlice = memref<1x1x2x4xui8, @DDR>
!CMXType = memref<1x1x2x4xui8, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesDmasTilingOnLastDim inputsInfo : {
    DataInfo "Parameter_233593" : tensor<4x1x2x4xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<4x1x2x4xui8>
}

// CHECK-LABEL: func.func @DynamicStridesDmasTilingOnLastDim
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<4x1x2x4xui8, @DDR>
func.func @DynamicStridesDmasTilingOnLastDim(%arg0: !DDRType, %arg2: !DDRType) -> !DDRType {
    %subView = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 2, 4] : !DDRType to !DDRSlice
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.NNDMA inputs(%subView : !DDRSlice) outputs(%1 : !CMXType) -> !CMXType

    return %arg2 : !DDRType

    // CHECK: [[DDR_SLICE:%.+]] = VPUIP.SubView [[ARG_0]]
    // CHECK-NEXT: [[CMX:%.+]] = memref.alloc
    // CHECK-SAME:                  @CMX_NN
    // CHECK: VPUIP.NNDMA {stridedInput} inputs([[DDR_SLICE]]
    // CHECK-SAME:                       outputs([[CMX]]
}

// -----

!DDRType = memref<8x1x1x1xui8, @DDR>
!DDRSlice = memref<4x1x1x1xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [2, 1, 1, 1]}, @DDR>
!CMXType = memref<4x1x1x1xui8, @CMX_NN>

net.NetworkInfo entryPoint : @LegalizeDynamicStridesDmasStridedTilingOnLastDim inputsInfo : {
    DataInfo "Parameter_233593" : tensor<8x1x1x1xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<8x1x1x1xui8>
}

func.func @LegalizeDynamicStridesDmasStridedTilingOnLastDim(%arg0: !DDRType, %arg2: !DDRType) -> !DDRType {
    %subView = VPUIP.SubView %arg0 [0, 0, 0, 0] [4, 1, 1, 1] [2, 1, 1, 1] : !DDRType to !DDRSlice
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.NNDMA inputs(%subView : !DDRSlice) outputs(%1 : !CMXType) -> !CMXType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<8x1x1x1xui8, @DDR>
    // CHECK: VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:                      outputs([[ALLOC]]

    return %arg2 : !DDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesReadOutputWriteOutput inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

// CHECK-LABEL: func.func @DynamicStridesReadOutputWriteOutput
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x4x64x320xf16, @DDR>
func.func @DynamicStridesReadOutputWriteOutput(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %1 = VPUIP.NNDMA inputs(%0: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %2 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %3 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%2: !FlatDDRType) -> !FlatDDRType

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: [[NNDMA0:%.+]] = VPUIP.NNDMA {stridedInput} inputs([[ARG_0]]
    // CHECK-SAME:                            outputs([[ALLOC0]]
    // CHECK-NEXT: [[CMX_ALLOC:%.+]] = memref.alloc() : memref<1x81920xf16, @CMX_NN>
    // CHECK-NEXT: [[RESHAPED_ALLOC:%.+]] = VPUIP.GenericReshape inputs([[NNDMA0]]
    // CHECK-NEXT: [[IN_DMA:%.+]] = VPUIP.NNDMA inputs([[RESHAPED_ALLOC]]
    // CHECK-SAME:                              outputs([[ALLOC0]]
    // CHECK-NEXT: [[RESHAPED_ALLOC1:%.+]] = VPUIP.GenericReshape inputs([[NNDMA0]]
    // CHECK-NEXT: [[OUT_DMA:%.+]] = VPUIP.NNDMA inputs([[CMX_ALLOC]]
    // CHECK-SAME:                               outputs([[RESHAPED_ALLOC1]]
    // CHECK-NEXT: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[ARG_0]]
    // CHECK-SAME:                                          [[OUT_DMA]]
    // CHECK-SAME:                                          [[RESHAPED_ALLOC]]
    // CHECK-SAME:                                          [[NNDMA0]]
    // CHECK-SAME:                                   outputs([[ALLOC0]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]
    // CHECK-SAME:                             outputs([[ARG_0]]

    return %arg : !DDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesRwOutputDirectly inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

// CHECK-LABEL: func.func @DynamicStridesRwOutputDirectly
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x4x64x320xf16, @DDR>
func.func @DynamicStridesRwOutputDirectly(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %1 = VPUIP.NNDMA inputs(%0: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %3 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%0: !FlatDDRType) -> !FlatDDRType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: [[LEGALIZATION_DMA:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:                                                 outputs([[ALLOC]]
    // CHECK-NEXT: [[CMX_ALLOC:%.+]] = memref.alloc() : memref<1x81920xf16, @CMX_NN>
    // CHECK-NEXT: [[INCOMPATIBLE_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[LEGALIZATION_DMA]]
    // CHECK: [[INCOMPATIBLE_DMA:%.+]] = VPUIP.NNDMA inputs([[CMX_ALLOC]]
    // CHECK-SAME:                               outputs([[INCOMPATIBLE_RESHAPE]]
    // CHECK-NEXT: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[ARG_0]]
    // CHECK-SAME:                                          [[INCOMPATIBLE_DMA]]
    // CHECK-SAME:                                          [[INCOMPATIBLE_RESHAPE]]
    // CHECK-SAME:                                          [[LEGALIZATION_DMA]]
    // CHECK-SAME:                                   outputs([[ALLOC]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]

    return %arg : !DDRType
}

// -----

!FullDDRType = memref<2x4x64x320xf16, @DDR>
!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesRwOutputMultipleReads inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<2x4x64x320xf32> {dynamicStrides}
}

func.func @DynamicStridesRwOutputMultipleReads(%arg: !FullDDRType) -> !FullDDRType {
    %cmx = memref.alloc() : !CMXType
    %subView0 = VPUIP.SubView %arg [0, 0, 0, 0] [1, 4, 64, 320] : !FullDDRType to !DDRType
    %subView1 = VPUIP.SubView %arg [1, 0, 0, 0] [1, 4, 64, 320] : !FullDDRType to !DDRType
    %0 = VPUIP.GenericReshape inputs(%subView0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.NNDMA inputs(%0: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %2 = VPUIP.GenericReshape inputs(%subView1 : !DDRType) -> !FlatDDRType
    %3 = VPUIP.NNDMA inputs(%2: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType

    %4 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%0: !FlatDDRType) -> !FlatDDRType
    %5 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%2: !FlatDDRType) -> !FlatDDRType

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:                             outputs([[ALLOC0]]
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:                            outputs([[ALLOC1]]
    // CHECK: [[CONCAT0:%.+]] = VPUIP.ConcatView inputs
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT0]]
    // CHECK: [[CONCAT1:%.+]] = VPUIP.ConcatView inputs
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT1]]

    return %arg : !FullDDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesReadThenWrite inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

func.func @DynamicStridesReadThenWrite(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %1 = VPUIP.NNDMA inputs(%0: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %3 = VPUIP.NNDMA inputs(%1: !CMXType) outputs(%0: !FlatDDRType) -> !FlatDDRType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs(%arg0
    // CHECK-SAME:                            outputs([[ALLOC]]
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]

    return %arg : !DDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesSimpleMultiFunc inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

func.func @Outlined(%arg: !FlatDDRType) -> !FlatDDRType {
    %cmx = memref.alloc() : !CMXType
    %1 = VPUIP.NNDMA inputs(%arg: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %2 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%arg: !FlatDDRType) -> !FlatDDRType
    return %arg: !FlatDDRType
}

// CHECK-LABEL: func.func @DynamicStridesSimpleMultiFunc
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x4x64x320xf16, @DDR>
func.func @DynamicStridesSimpleMultiFunc(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %1 = func.call @Outlined(%0) : (!FlatDDRType) -> !FlatDDRType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: [[IN_DMA:%.+]] = VPUIP.NNDMA {stridedInput} inputs([[ARG_0]]
    // CHECK-SAME:                            outputs([[ALLOC]]
    // CHECK: [[INCOMPATIBLE_RESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK: [[FUNC:%.+]] = call
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[ARG_0]]
    // CHECK-SAME:                                     [[FUNC]]
    // CHECK-SAME:                                     [[IN_DMA]]
    // CHECK-SAME:                              outputs([[ALLOC]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]

    return %arg : !DDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesMultipleNestedFunctions inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

func.func @OutlinedNested(%arg: !FlatDDRType) -> !FlatDDRType {
    %cmx = memref.alloc() : !CMXType
    %1 = VPUIP.NNDMA inputs(%arg: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %2 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%arg: !FlatDDRType) -> !FlatDDRType
    return %arg: !FlatDDRType
}

func.func @Outlined(%arg: !FlatDDRType) -> !FlatDDRType {
    %cmx = memref.alloc() : !CMXType
    %1 = VPUIP.NNDMA inputs(%arg: !FlatDDRType) outputs(%cmx: !CMXType) -> !CMXType
    %2 = VPUIP.NNDMA inputs(%cmx: !CMXType) outputs(%arg: !FlatDDRType) -> !FlatDDRType
    %3 = func.call @OutlinedNested(%arg) : (!FlatDDRType) -> !FlatDDRType
    return %arg: !FlatDDRType
}

// CHECK-LABEL: func.func @DynamicStridesMultipleNestedFunctions
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x4x64x320xf16, @DDR>
func.func @DynamicStridesMultipleNestedFunctions(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %1 = func.call @Outlined(%0) : (!FlatDDRType) -> !FlatDDRType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput} inputs([[ARG_0]]
    // CHECK-SAME:                            outputs([[ALLOC]]
    // CHECK: [[INCOMPATIBLE_RESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]

    return %arg : !DDRType
}

// -----

!DDRType = memref<4x64x320xf16, @DDR>
!DDRSliceType = memref<2x64x320xf16, @DDR>
!CMXType = memref<2x64x320xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesMultiFuncIoAlias inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

func.func @FuncNested(%arg : !DDRSliceType) -> !DDRSliceType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.NNDMA inputs(%arg : !DDRSliceType) outputs(%cmx : !CMXType) -> !CMXType
    %1 = VPUIP.NNDMA inputs(%cmx : !CMXType) outputs(%arg : !DDRSliceType) -> !DDRSliceType
    // CHECK: VPUIP.NNDMA {stridedInput}
    // CHECK: VPUIP.NNDMA {stridedOutput}
    return %1 : !DDRSliceType
}

func.func @TileFunc(%arg: !DDRType) -> (!DDRSliceType, !DDRSliceType) {
    %cmx = memref.alloc() : !CMXType
    %1 = VPUIP.SubView %arg [0, 0, 0] [2, 64, 320] : !DDRType to !DDRSliceType
    %2 = VPUIP.SubView %arg [2, 0, 0] [2, 64, 320] : !DDRType to !DDRSliceType
    %3 = VPUIP.NNDMA inputs(%1 : !DDRSliceType) outputs(%cmx : !CMXType) -> !CMXType
    %4 = VPUIP.NNDMA inputs(%cmx : !CMXType) outputs(%2 : !DDRSliceType) -> !DDRSliceType
    // CHECK: VPUIP.NNDMA {stridedInput}
    // CHECK: VPUIP.NNDMA {stridedOutput}
    %5 = func.call @FuncNested(%1) : (!DDRSliceType) -> !DDRSliceType
    return %5, %2 : !DDRSliceType, !DDRSliceType
}

func.func @DynamicStridesMultiFuncIoAlias(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %1:2 = func.call @TileFunc(%arg) : (!DDRType) -> (!DDRSliceType, !DDRSliceType)
    %2 = VPUIP.NNDMA inputs(%1#0 : !DDRSliceType) outputs(%cmx : !CMXType) -> !CMXType
    %3 = VPUIP.NNDMA inputs(%cmx : !CMXType) outputs(%1#0 : !DDRSliceType) -> !DDRSliceType
    %4 = VPUIP.NNDMA inputs(%1#1 : !DDRSliceType) outputs(%cmx : !CMXType) -> !CMXType
    %5 = VPUIP.NNDMA inputs(%cmx : !CMXType) outputs(%1#1 : !DDRSliceType) -> !DDRSliceType

    // CHECK: VPUIP.NNDMA {stridedInput}
    // CHECK: VPUIP.NNDMA {stridedOutput}
    // CHECK: VPUIP.NNDMA {stridedInput}
    // CHECK: VPUIP.NNDMA {stridedOutput}
    return %arg : !DDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<81920xf16, @DDR>
!FlatDDRSliceType = memref<40960xf16, @DDR>
!CMXType = memref<40960xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesMultiFuncIncompatibleIoAlias inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

func.func @TileFunc(%arg: !FlatDDRType) -> (!FlatDDRSliceType, !FlatDDRSliceType) {
    %1 = VPUIP.SubView %arg [0] [40960] : !FlatDDRType to !FlatDDRSliceType
    %2 = VPUIP.SubView %arg [40960] [40960] : !FlatDDRType to !FlatDDRSliceType
    return %1, %2 : !FlatDDRSliceType, !FlatDDRSliceType
}

func.func @DynamicStridesMultiFuncIncompatibleIoAlias(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %0 = VPUIP.GenericReshape inputs(%arg : !DDRType) -> !FlatDDRType
    %1:2 = func.call @TileFunc(%0) : (!FlatDDRType) -> (!FlatDDRSliceType, !FlatDDRSliceType)
    %2 = VPUIP.NNDMA inputs(%1#0 : !FlatDDRSliceType) outputs(%cmx : !CMXType) -> !CMXType
    %3 = VPUIP.NNDMA inputs(%cmx : !CMXType) outputs(%1#0 : !FlatDDRSliceType) -> !FlatDDRSliceType
    %4 = VPUIP.NNDMA inputs(%1#1 : !FlatDDRSliceType) outputs(%cmx : !CMXType) -> !CMXType
    %5 = VPUIP.NNDMA inputs(%cmx : !CMXType) outputs(%1#1 : !FlatDDRSliceType) -> !FlatDDRSliceType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK-NEXT: VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:                          outputs([[ALLOC]]
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:                          outputs([[ALLOC]]
    // CHECK-NEXT: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]

    return %arg : !DDRType
}

// -----

!DDRType = memref<1x4x64x320xf16, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesIncompatibleReadDmas inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

func.func @DynamicStridesIncompatibleReadDmas(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %indices = memref.alloc() : memref<1xi64, @CMX_NN>
    %0 = VPUIP.GatherDMA <{addressingMode = 1 : i64,
                        port = 0 : i64, elementSize = 16, padding = 0}>
            inputs(%arg : !DDRType)
            indices(%indices : memref<1xi64, @CMX_NN>)
            outputs(%cmx : !CMXType) -> !CMXType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, @DDR>
    // CHECK: VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:                      outputs([[ALLOC]]

    return %arg : !DDRType
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DDRType = memref<1x4x64x320xf16,  {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
!FlatDDRType = memref<1x81920xf16, @DDR>
!CMXType = memref<1x81920xf16, @CMX_NN>

net.NetworkInfo entryPoint : @DynamicStridesIncompatibleWriteDmas inputsInfo : {
} outputsInfo : {
    DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

// CHECK-LABEL: func.func @DynamicStridesIncompatibleWriteDmas
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x4x64x320xf16,  {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
func.func @DynamicStridesIncompatibleWriteDmas(%arg: !DDRType) -> !DDRType {
    %cmx = memref.alloc() : !CMXType
    %act_compression_size = memref.alloc() : memref<32xui8, @CMX_NN>
    %1 = VPUIP.CompressDMAOp inputs(%cmx : !CMXType)
                            outputs(%arg : !DDRType)
                            act_compression_size_entry(%act_compression_size : memref<32xui8, @CMX_NN>)
                            -> !DDRType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x64x320xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
    // CHECK: [[IN_DMA:%.+]] = VPUIP.NNDMA {stridedInput} inputs([[ARG_0]]
    // CHECK-SAME:                                      outputs([[ALLOC]]
    // CHECK: [[INCOMPATIBLE_DMA:%.+]] = VPUIP.CompressDMAOp inputs(
    // CHECK-SAME:                                           outputs([[IN_DMA]]
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[ARG_0]]
    // CHECK-SAME:                                     [[INCOMPATIBLE_DMA]]
    // CHECK-SAME:                                     [[IN_DMA]]
    // CHECK-SAME:                              outputs([[ALLOC]]
    // CHECK: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT]]
    // CHECK-SAME:                        outputs([[ARG_0]]

    return %arg : !DDRType
}
