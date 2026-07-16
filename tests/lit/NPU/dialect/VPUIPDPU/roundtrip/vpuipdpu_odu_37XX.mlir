//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% %s | FileCheck %s --match-full-lines
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_1 idx(!VPURegMapped.Index<0:0:1>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out_cast1: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out_cast2: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_type(ODU_DTYPE_FP16)
                VPUIPDPU.ODUCast cast_output(%act_out_cast1: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.ODUCast cast_output(%act_out_cast2: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    }

    VPUIPDPU.DPUVariant @DPUVariant_1 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_1, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    }
}

//CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0}> DPUCfg : {
//CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_3:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_4:%.+]]: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
//CHECK:     VPUIPDPU.ODUCfg {
//CHECK:       VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
//CHECK:       VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
//CHECK:       VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
//CHECK:       VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
//CHECK:       VPUIPDPU.ODUSparsity [[ARG_4]] : memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>
//CHECK:       VPUIPDPU.ODUOutActivations out_activations([[ARG_1]] : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_type(ODU_DTYPE_FP16)
//CHECK:       VPUIPDPU.ODUCast cast_output([[ARG_2]] : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
//CHECK:       VPUIPDPU.ODUCast cast_output([[ARG_3]] : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
//CHECK:     }
//CHECK:   }
//CHECK:   VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:     VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:   }
//CHECK:   VPUIPDPU.DPUVariant @DPUVariant_1 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_1, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:     VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_2 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
                VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(0)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    }
}

//CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0}> DPUCfg : {
//CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
//CHECK:     VPUIPDPU.ODUCfg {
//CHECK:       VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
//CHECK:       VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
//CHECK:       VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(0)
//CHECK:       VPUIPDPU.ODUOutActivations out_activations([[ARG_1]] : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>)
//CHECK:     }
//CHECK:   }
//CHECK:   VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:     VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_3 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]> sparse_value(0)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_type(ODU_DTYPE_FP16)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    }
}

//CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 <{input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0}> DPUCfg : {
//CHECK:   ^bb0([[ARG_0:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_2:%.+]]: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
//CHECK:     VPUIPDPU.ODUCfg {
//CHECK:       VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
//CHECK:       VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
//CHECK:       VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
//CHECK:       VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
//CHECK:       VPUIPDPU.ODUSparsity [[ARG_2]] : memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]> sparse_value(0)
//CHECK:       VPUIPDPU.ODUOutActivations out_activations([[ARG_1]] : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_type(ODU_DTYPE_FP16)
//CHECK:     }
//CHECK:   }
//CHECK:   VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:     VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:   }
