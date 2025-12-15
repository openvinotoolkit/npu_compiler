//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% %s | FileCheck %s --match-full-lines
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            // Note1: After IDU update, make sure DTZ is only used on f16 input
            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEDenormalOperandsFTZ
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
// CHECK:   ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.MPECfg {
// CHECK:       VPUIPDPU.MPEDenormalOperandsFTZ
// CHECK:     }
// CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_2 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.MPECfg {
                // Note2: The bias needs to be I8 for I8 input and U8 for U8 input. Check will be added once IDU is done.
                VPUIPDPU.MPEActivationBias act_bias(-12)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
// CHECK:   ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.MPECfg {
// CHECK:       VPUIPDPU.MPEActivationBias act_bias(-12)
// CHECK:     }
// CHECK:   }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_3 {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEWeightsBias weights_bias(-10)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
            }
        }

}

// CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
// CHECK:   ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
// CHECK:     VPUIPDPU.MPECfg {
// CHECK:       VPUIPDPU.MPEWeightsBias weights_bias(-10)
// CHECK:     }
// CHECK:   }
