//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% %s | FileCheck %s --match-full-lines
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_1 idx(!VPURegMapped.Index<0:0:1>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
        VPUIPDPU.ODUHaloCfg {
            VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(3) end_coord_y(3)
                                    activations_offset(0) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_1|DPU_TILE_2")
            VPUIPDPU.ODUHaloRegion begin_coord_x(60) begin_coord_y(60) end_coord_x(63) end_coord_y(63)
                                    activations_offset(0) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_1|DPU_TILE_2")
        }
    }

    VPUIPDPU.DPUVariant @DPUVariant_1 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_1, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
        VPUIPDPU.ODUHaloCfg {
            VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x (63) end_coord_y(63)
                                    activations_offset(0) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_2|DPU_TILE_3")
        }
    }
}

//CHECK:    VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:    ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>, %arg2: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
//CHECK:      VPUIPDPU.ODUCfg {
//CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
//CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
//CHECK:        VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
//CHECK:        VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
//CHECK:        VPUIPDPU.ODUSparsity %arg2 : memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>
//CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
//CHECK:        VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
//CHECK:      }
//CHECK:    }
//CHECK:    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:      VPUIPDPU.ODUHaloCfg {
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(3) end_coord_y(3) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(60) begin_coord_y(60) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
//CHECK:      }
//CHECK:    }
//CHECK:    VPUIPDPU.DPUVariant @DPUVariant_1 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_1, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:      VPUIPDPU.ODUHaloCfg {
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_2|DPU_TILE_3)
//CHECK:      }
//CHECK:    }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_2 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
                VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(0)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    }
}

//CHECK:    VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:    ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>, %arg2: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
//CHECK:      VPUIPDPU.ODUCfg {
//CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
//CHECK:        VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
//CHECK:        VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(0)
//CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK:        VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS)
//CHECK:      }
//CHECK:    }
//CHECK:    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:    }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_3 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]> sparse_value(0)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
    DPUCfg: {
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
        VPUIPDPU.ODUHaloCfg {
            VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(4) end_coord_y(4)
                                    activations_offset(0) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_1|DPU_TILE_2")
            VPUIPDPU.ODUHaloRegion begin_coord_x(59) begin_coord_y(59) end_coord_x(63) end_coord_y(63)
                                    activations_offset(0) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_1|DPU_TILE_2")
            VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(63) end_coord_y(63)
                                    activations_offset(0) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_3|DPU_TILE_4")
            VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(100) end_coord_y(100)
                                    activations_offset(131072) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_1|DPU_TILE_2|DPU_TILE_3")
            VPUIPDPU.ODUHaloRegion begin_coord_x(29) begin_coord_y(29) end_coord_x(35) end_coord_y(35)
                                    activations_offset(262144) sparsity_offset(0) target_width(0)
                                    cast_to_tile("DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4")
        }
    }
}

//CHECK:    VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:    ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>, %arg2: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
//CHECK:      VPUIPDPU.ODUCfg {
//CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
//CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
//CHECK:        VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
//CHECK:        VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
//CHECK:        VPUIPDPU.ODUSparsity %arg2 : memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]> sparse_value(0)
//CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
//CHECK:        VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
//CHECK:      }
//CHECK:    }
//CHECK:    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
//CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
//CHECK:      VPUIPDPU.ODUHaloCfg {
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(4) end_coord_y(4) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(59) begin_coord_y(59) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_3|DPU_TILE_4)
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(100) end_coord_y(100) activations_offset(131072) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_3)
//CHECK:        VPUIPDPU.ODUHaloRegion begin_coord_x(29) begin_coord_y(29) end_coord_x(35) end_coord_y(35) activations_offset(262144) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4)
//CHECK:      }
//CHECK:    }
