//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% %s | FileCheck %s --match-full-lines
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
    VPUASM.DeclareBuffer @DeclareBuffer_WeightTable !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <196736> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_SEIn !VPUASM.Buffer< "CMX_NN"[0] <204928> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_SparseIn !VPUASM.Buffer< "CMX_NN"[0] <213120> : memref<1x16x16x16xi1, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, input_sparsity_map = @DeclareBuffer_SparseIn, input_storage_element_table = @DeclareBuffer_SEIn, weight_table = @DeclareBuffer_WeightTable, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<ELTWISE>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_in_seg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_in_seg1: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_in_seg2: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_in_seg3: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg1: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg2: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg3: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_in_seg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_in_seg1: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_in_seg2: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_in_seg3: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUStorageElement se_size(32)
                VPUIPDPU.IDUKernel kernel_x(1) kernel_y(2)
                VPUIPDPU.IDUStride stride_x(0) stride_y(0)
                VPUIPDPU.IDUInputLayerCfg sparsity_pattern(7) {input_compressed}
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
                VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
            }
        }

    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, weight_table = @DeclareBuffer_WeightTable, nce_task_type = #VPUIP.nce_task_type<ELTWISE>}
    DPUCfg: {
        VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_1)
        VPUIPDPU.IDUWeightSwizzle wt_swizzle_key(SWIZZLE_KEY_1)
        VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_8_8)
        VPUIPDPU.IDUSEDense
        VPUIPDPU.IDUConvContinue
        VPUIPDPU.IDUWorkloadSet start_x(3) start_y(3) start_z(3) size_x(32) size_y(32) size_z(32)
        VPUIPDPU.IDUPadding pad_count(<left = 3, right = 3, top = 3, bottom = 3>)
        VPUIPDPU.IDUPerOutputChannelScaling {tensor2_act_sparse}
        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    }
}

// CHECK:    VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, input_sparsity_map = @DeclareBuffer_SparseIn, input_storage_element_table = @DeclareBuffer_SEIn, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>, weight_table = @DeclareBuffer_WeightTable} DPUCfg : {
// CHECK:    ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg2: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg3: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg4: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg5: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg6: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg7: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg8: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg9: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg10: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg11: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg12: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg13: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
// CHECK:      VPUIPDPU.IDUCfg {
// CHECK:        VPUIPDPU.IDUStorageElement se_size(32)
// CHECK:        VPUIPDPU.IDUKernel kernel_x(1) kernel_y(2)
// CHECK:        VPUIPDPU.IDUStride stride_x(0) stride_y(0)
// CHECK:        VPUIPDPU.IDUInputLayerCfg sparsity_pattern(7) {input_compressed}
// CHECK:        VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
// CHECK:        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
// CHECK:        VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
// CHECK:        VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
// CHECK:      }
// CHECK:    }
// CHECK:    VPUIPDPU.DPUVariant @DPUVariant_0 invariant(@DeclareTaskBuffer_DPUInvariant_0) {nce_task_type = #VPUIP.nce_task_type<ELTWISE>, taskLocation = @DeclareTaskBuffer_DPUVariant_0, task_index = !VPURegMapped.Index<0:0:0>, weight_table = @DeclareBuffer_WeightTable} DPUCfg : {
// CHECK:      VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_1)
// CHECK:      VPUIPDPU.IDUWeightSwizzle wt_swizzle_key(SWIZZLE_KEY_1)
// CHECK:      VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_8_8)
// CHECK:      VPUIPDPU.IDUSEDense
// CHECK:      VPUIPDPU.IDUConvContinue
// CHECK:      VPUIPDPU.IDUWorkloadSet start_x(3) start_y(3) start_z(3) size_x(32) size_y(32) size_z(32)
// CHECK:      VPUIPDPU.IDUPadding pad_count(<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>)
// CHECK:      VPUIPDPU.IDUPerOutputChannelScaling {tensor2_act_sparse}
// CHECK:    }
