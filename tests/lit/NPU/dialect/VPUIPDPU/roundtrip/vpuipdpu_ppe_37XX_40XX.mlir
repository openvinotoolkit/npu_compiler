//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% %s | FileCheck %s --match-full-lines
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1a { // Use case #1a: u8 DPU in, u8 DPU out - with activation scaling
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_WeightTable !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, weight_table=@DeclareBuffer_WeightTable, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
                VPUIPDPU.PPEFpConvert convert_mode(NONE)
                VPUIPDPU.PPEIntBiasAdd %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntScaleMult %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                VPUIPDPU.PPEIntScaleShift %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                VPUIPDPU.PPEIntRound round_mode(RNE)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
                VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
                VPUIPDPU.PPEIntConvert convert_mode(NONE)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, weight_table = @DeclareBuffer_WeightTable}> DPUCfg : {
// CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.PPECfg {
// CHECK:       VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
// CHECK:       VPUIPDPU.PPEFpConvert convert_mode(NONE)
// CHECK:       VPUIPDPU.PPEIntBiasAdd [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEIntScaleMult [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:       VPUIPDPU.PPEIntScaleShift [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:       VPUIPDPU.PPEIntRound round_mode(RNE)
// CHECK:       VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
// CHECK:       VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
// CHECK:       VPUIPDPU.PPEIntConvert convert_mode(NONE)
// CHECK:     }
// CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1b { // Use case #1b: u8 DPU in, u8 DPU out - with activation truncation
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
                VPUIPDPU.PPEFpConvert convert_mode(NONE)
                VPUIPDPU.PPEIntBiasAdd bias_static(0)
                VPUIPDPU.PPEIntScaleMult scale_static(1)
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                VPUIPDPU.PPEIntScaleShift shift_static(0)
                VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                VPUIPDPU.PPEIntRound round_mode(NONE)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
                VPUIPDPU.PPEIntConvert convert_mode(NONE)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0}> DPUCfg : {
// CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.PPECfg {
// CHECK:       VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
// CHECK:       VPUIPDPU.PPEFpConvert convert_mode(NONE)
// CHECK:       VPUIPDPU.PPEIntBiasAdd bias_static(0)
// CHECK:       VPUIPDPU.PPEIntScaleMult scale_static(1)
// CHECK:       VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:       VPUIPDPU.PPEIntScaleShift shift_static(0)
// CHECK:       VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:       VPUIPDPU.PPEIntRound round_mode(NONE)
// CHECK:       VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
// CHECK:       VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
// CHECK:       VPUIPDPU.PPEIntConvert convert_mode(NONE)
// CHECK:     }
// CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_2 { // Use case #2: u8 DPU in, fp16 DPU out
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_WeightTable !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, weight_table=@DeclareBuffer_WeightTable, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
                VPUIPDPU.PPEFpConvert convert_mode(NONE)
                VPUIPDPU.PPEIntBiasAdd %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntScaleMult %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(0)
                VPUIPDPU.PPEIntScaleShift %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntConvert convert_mode(FP16)
                VPUIPDPU.PPEIntClamp clamp_high(70) // 70 corresponds to RELU6
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, weight_table = @DeclareBuffer_WeightTable}> DPUCfg : {
// CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.PPECfg {
// CHECK:       VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
// CHECK:       VPUIPDPU.PPEFpConvert convert_mode(NONE)
// CHECK:       VPUIPDPU.PPEIntBiasAdd [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEIntScaleMult [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEIntPreluMult prelu_mult_static(0)
// CHECK:       VPUIPDPU.PPEIntScaleShift [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEIntConvert convert_mode(FP16)
// CHECK:       VPUIPDPU.PPEIntClamp clamp_high(70)
// CHECK:     }
// CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_3 { // Use case #3: fp16 DPU in, fp16 DPU out
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_WeightTable !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, weight_table=@DeclareBuffer_WeightTable, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScalePreluMult %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> prelu_alpha(0.1)
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
                VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON) ftz_mode(OFF)
                VPUIPDPU.PPEIntBiasAdd bias_static(0)
                VPUIPDPU.PPEIntScaleMult scale_static(1)
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                VPUIPDPU.PPEIntScaleShift shift_static(0)
                VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                VPUIPDPU.PPEIntRound round_mode(NONE)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647) // (MIN_I32, MAX_I32)
                VPUIPDPU.PPEIntConvert convert_mode(NONE)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, weight_table = @DeclareBuffer_WeightTable}> DPUCfg : {
// CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.PPECfg {
// CHECK:       VPUIPDPU.PPEFpBiasAdd [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEFpScalePreluMult [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> prelu_alpha(1.000000e-01)
// CHECK:       VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
// CHECK:       VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON) ftz_mode(OFF)
// CHECK:       VPUIPDPU.PPEIntBiasAdd bias_static(0)
// CHECK:       VPUIPDPU.PPEIntScaleMult scale_static(1)
// CHECK:       VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:       VPUIPDPU.PPEIntScaleShift shift_static(0)
// CHECK:       VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:       VPUIPDPU.PPEIntRound round_mode(NONE)
// CHECK:       VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
// CHECK:       VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
// CHECK:       VPUIPDPU.PPEIntConvert convert_mode(NONE)
// CHECK:     }
// CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_4 { // Use case #4: fp16 DPU in, u8 DPU out
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_WeightTable !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, weight_table=@DeclareBuffer_WeightTable, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScalePreluMult %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> prelu_alpha(0.1)
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
                VPUIPDPU.PPEFpConvert convert_mode(I32)
                VPUIPDPU.PPEIntBiasAdd bias_static(0)
                VPUIPDPU.PPEIntScaleMult scale_static(1)
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                VPUIPDPU.PPEIntScaleShift shift_static(0)
                VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                VPUIPDPU.PPEIntRound round_mode(NONE)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
                VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
                VPUIPDPU.PPEIntConvert convert_mode(NONE)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, weight_table = @DeclareBuffer_WeightTable}> DPUCfg : {
// CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.PPECfg {
// CHECK:       VPUIPDPU.PPEFpBiasAdd [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK:       VPUIPDPU.PPEFpScalePreluMult [[ARG_1]] : memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]> prelu_alpha(1.000000e-01)
// CHECK:       VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
// CHECK:       VPUIPDPU.PPEFpConvert convert_mode(I32)
// CHECK:       VPUIPDPU.PPEIntBiasAdd bias_static(0)
// CHECK:       VPUIPDPU.PPEIntScaleMult scale_static(1)
// CHECK:       VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:       VPUIPDPU.PPEIntScaleShift shift_static(0)
// CHECK:       VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:       VPUIPDPU.PPEIntRound round_mode(NONE)
// CHECK:       VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
// CHECK:       VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
// CHECK:       VPUIPDPU.PPEIntConvert convert_mode(NONE)
// CHECK:     }
// CHECK:   }
