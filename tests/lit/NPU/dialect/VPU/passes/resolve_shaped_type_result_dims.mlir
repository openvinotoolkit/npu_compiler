//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --resolve-shaped-type-result-dims %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @Conv
func.func @Conv(%arg0: tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 8]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x4x8xf16, {order = #NHWC}>) -> (tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 8]> : tensor<4xsi64>, order = #NHWC}>
    %C2 = arith.constant 2 : index
    // CHECK: [[C2:%.+]] = arith.constant 2 : index
    %cst_0 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[WEIGHT:%.+]] = const.Declare
    %CONV = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,  strides = [2, 2]} : tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 8]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[WEIGHT]])
    %DIM = tensor.dim %CONV, %C2 : tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[VAL:%.+]] = tensor.dim [[IN]], [[C2]]
    // CHECK: [[DIM:%.+]] = arith.divsi [[VAL]], [[C2]] : index
    return %CONV, %DIM : tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>, index
    // CHECK: return [[CONV]], [[DIM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @Eltwise
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>
func.func @Eltwise(%arg0: tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>) -> (tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>, index) {
    %C2 = arith.constant 2 : index
    %ELTWISE = VPU.NCE.Eltwise(%arg0, %arg0) {
            input_padding = [0, 0, 0, 0],
            op_type = #VPU.eltwise_type<SUBTRACT>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            tilingStrategy = [1, 1, 2, 1]
        } -> tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>
    %DIM = tensor.dim %ELTWISE, %C2 : tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>
    return %ELTWISE,%DIM : tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>, index

    // CHECK:       [[C2:%.+]] = arith.constant 2 : index
    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]])
    // CHECK:       [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]]
    // CHECK:       return [[OUTPUT]], [[DIM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NCEMaxPool
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x16x?x15xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
func.func @NCEMaxPool(%arg0: tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NHWC}>,
                 %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
                 ) -> (tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NCHW}>, index) {
    %C2 = arith.constant 2 : index
    %MAXPOOL = VPU.NCE.MaxPool(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] pad [0, 0, 0, 13] #VPU.mpe_mode<CUBOID_16x16>
    }
    %DIM = tensor.dim %arg0, %C2 : tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NHWC}>
    return %MAXPOOL,%DIM : tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NCHW}>, index

    // CHECK:       [[C2:%.+]] = arith.constant 2 : index
    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT0]], [[INPUT1]] )
    // CHECK:       [[DIM:%.+]] = tensor.dim [[INPUT0]], [[C2]]
    // CHECK:       return [[OUTPUT]], [[DIM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @test {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

// CHECK-LABEL: @ConvWithFlattenedFilter
func.func @ConvWithFlattenedFilter(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>, index) {
    // CHECK: [[IN:%.+]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>
    %C2 = arith.constant 2 : index
    // CHECK: [[C2:%.+]] = arith.constant 2 : index
    %cst_0 = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x144xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[WEIGHT:%.+]] = const.Declare
    %CONV = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [32, 3, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,  strides = [2, 2]} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[WEIGHT]])
    %DIM = tensor.dim %CONV, %C2 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[VAL:%.+]] = tensor.dim [[IN]], [[C2]]
    // CHECK: [[DIM:%.+]] = arith.divsi [[VAL]], [[C2]] : index
    return %CONV, %DIM : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>, index
    // CHECK: return [[CONV]], [[DIM]]
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InterpDMAInType = tensor<1x3x4x6xf16>
!InterpDMAOutType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @ReifyInterpolateDMAOpShape
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x4x6xf16>, [[ARG1:%.+]]: tensor<2xf16>, [[ARG2:%.+]]: tensor<1x1x1x1024xui8>
func.func @ReifyInterpolateDMAOpShape(%IN: !InterpDMAInType, %SCALES: tensor<2xf16>, %AUX: tensor<1x1x1x1024xui8>) -> (!InterpDMAOutType, index, index) {
    %IDX_2 = arith.constant 2 : index
    %IDX_3 = arith.constant 3 : index
    // CHECK: [[CST_W:%.+]] = arith.constant 6.000000e+00 : f64
    // CHECK: [[C1:%.+]] = arith.constant 1 : index
    // CHECK: [[CST_H:%.+]] = arith.constant 4.000000e+00 : f64
    // CHECK: [[C0:%.+]] = arith.constant 0 : index

    %INTERP = VPU.InterpolateDMA(%IN, %SCALES, %AUX) {
        attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3]
    } : !InterpDMAInType, tensor<2xf16>, tensor<1x1x1x1024xui8> -> !InterpDMAOutType
    // CHECK: [[INTERP:%.+]] = VPU.InterpolateDMA([[ARG0]], [[ARG1]], [[ARG2]])

    %DIM_2 = tensor.dim %INTERP, %IDX_2 : !InterpDMAOutType
    %DIM_3 = tensor.dim %INTERP, %IDX_3 : !InterpDMAOutType
    // CHECK: [[SCALE_H:%.+]] = tensor.extract [[ARG1]]{{\[}}[[C0]]{{\]}} : tensor<2xf16>
    // CHECK: [[SCALE_H_F64:%.+]] = arith.extf [[SCALE_H]] : f16 to f64
    // CHECK: [[H_MUL:%.+]] = arith.mulf [[SCALE_H_F64]], [[CST_H]] : f64
    // CHECK: [[H_I64:%.+]] = arith.fptosi [[H_MUL]] : f64 to i64
    // CHECK: [[DIM_2_REIFIED:%.+]] = arith.index_cast [[H_I64]] : i64 to index
    // CHECK: [[SCALE_W:%.+]] = tensor.extract [[ARG1]]{{\[}}[[C1]]{{\]}} : tensor<2xf16>
    // CHECK: [[SCALE_W_F64:%.+]] = arith.extf [[SCALE_W]] : f16 to f64
    // CHECK: [[W_MUL:%.+]] = arith.mulf [[SCALE_W_F64]], [[CST_W]] : f64
    // CHECK: [[W_I64:%.+]] = arith.fptosi [[W_MUL]] : f64 to i64
    // CHECK: [[DIM_3_REIFIED:%.+]] = arith.index_cast [[W_I64]] : i64 to index

    return %INTERP, %DIM_2, %DIM_3 : !InterpDMAOutType, index, index
    // CHECK: return [[INTERP]], [[DIM_2_REIFIED]], [[DIM_3_REIFIED]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @QuantizeCastReifyDynamicHW
// CHECK-SAME:  [[INPUT:%arg[0-9]+]]: tensor<1x3x?x?x!qElemType
func.func @QuantizeCastReifyDynamicHW(
    %arg0: tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
) -> (tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>, index, index) {
    %C2 = arith.constant 2 : index
    %C3 = arith.constant 3 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    %OUT = VPU.QuantizeCast(%arg0) {dstElemType = ui8}
        : tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
       -> tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[OUT:%.+]] = VPU.QuantizeCast([[INPUT]])
    %DIM_H = tensor.dim %OUT, %C2 : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[C2]]
    %DIM_W = tensor.dim %OUT, %C3 : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[C3]]
    return %OUT, %DIM_H, %DIM_W : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>, index, index
    // CHECK: return [[OUT]], [[DIM_H]], [[DIM_W]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @QuantizeCastReifyStaticDim
// CHECK-SAME:  [[INPUT:%arg[0-9]+]]: tensor<1x3x720x1280x!qElemType
func.func @QuantizeCastReifyStaticDim(
    %arg0: tensor<1x3x720x1280x!qElemType, {order = #NHWC}>
) -> (tensor<1x3x720x1280xui8, {order = #NHWC}>, index) {
    %C2 = arith.constant 2 : index
    // CHECK: [[DIM_H:%.+]] = arith.constant 720 : index
    %OUT = VPU.QuantizeCast(%arg0) {dstElemType = ui8}
        : tensor<1x3x720x1280x!qElemType, {order = #NHWC}>
       -> tensor<1x3x720x1280xui8, {order = #NHWC}>
    // CHECK: [[OUT:%.+]] = VPU.QuantizeCast([[INPUT]])
    %DIM_H = tensor.dim %OUT, %C2 : tensor<1x3x720x1280xui8, {order = #NHWC}>
    return %OUT, %DIM_H : tensor<1x3x720x1280xui8, {order = #NHWC}>, index
    // CHECK: return [[OUT]], [[DIM_H]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ExpandInputC3HW = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
!ExpandOutputC16HW = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @ExpandReifyNonPaddedDynamicDims
// CHECK-SAME:  [[INPUT:%arg[0-9]+]]: tensor<1x3x?x?xf16
func.func @ExpandReifyNonPaddedDynamicDims(%arg0: !ExpandInputC3HW) -> (!ExpandOutputC16HW, index, index) {
    %C2 = arith.constant 2 : index
    %C3 = arith.constant 3 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    %EXPAND = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : !ExpandInputC3HW -> !ExpandOutputC16HW
    // CHECK: [[EXPAND:%.+]] = VPU.Expand([[INPUT]])
    %DIM_H = tensor.dim %EXPAND, %C2 : !ExpandOutputC16HW
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[C2]]
    %DIM_W = tensor.dim %EXPAND, %C3 : !ExpandOutputC16HW
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[C3]]
    return %EXPAND, %DIM_H, %DIM_W : !ExpandOutputC16HW, index, index
    // CHECK: return [[EXPAND]], [[DIM_H]], [[DIM_W]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ExpandInputC3HW = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
!ExpandOutputC3HPaddedW = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2568]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @ExpandReifyPaddedDynamicDim
// CHECK-SAME:  [[INPUT:%arg[0-9]+]]: tensor<1x3x?x?xf16
func.func @ExpandReifyPaddedDynamicDim(%arg0: !ExpandInputC3HW) -> (!ExpandOutputC3HPaddedW, index) {
    %C3 = arith.constant 3 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[PAD:%.+]] = arith.constant 8 : index
    %EXPAND = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 1], pads_end = [0, 0, 0, 7]} : !ExpandInputC3HW -> !ExpandOutputC3HPaddedW
    // CHECK: [[EXPAND:%.+]] = VPU.Expand([[INPUT]])
    %DIM_W = tensor.dim %EXPAND, %C3 : !ExpandOutputC3HPaddedW
    // CHECK: [[INPUT_W:%.+]] = tensor.dim [[INPUT]], [[C3]]
    // CHECK: [[DIM_W:%.+]] = arith.addi [[INPUT_W]], [[PAD]] : index
    return %EXPAND, %DIM_W : !ExpandOutputC3HPaddedW, index
    // CHECK: return [[EXPAND]], [[DIM_W]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ExpandInputStatic = tensor<1x3x720x1280xf16, {order = #NHWC}>
!ExpandOutputStatic = tensor<1x16x720x1280xf16, {order = #NHWC}>

// CHECK-LABEL: @ExpandReifyStaticDim
func.func @ExpandReifyStaticDim(%arg0: !ExpandInputStatic) -> (!ExpandOutputStatic, index) {
    %C1 = arith.constant 1 : index
    // CHECK: [[DIM_C:%.+]] = arith.constant 16 : index
    %EXPAND = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : !ExpandInputStatic -> !ExpandOutputStatic
    // CHECK: [[EXPAND:%.+]] = VPU.Expand
    %DIM_C = tensor.dim %EXPAND, %C1 : !ExpandOutputStatic
    return %EXPAND, %DIM_C : !ExpandOutputStatic, index
    // CHECK: return [[EXPAND]], [[DIM_C]]
}
