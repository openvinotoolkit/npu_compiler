//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIPDPU-to-NPUReg50XX="npu5-ppe-backwards-compatibility-mode=ENABLED" %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xf16>
      DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x16x64x64xf16>
  }
  func.func @main() {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_1 idx(!VPURegMapped.Index<0:0:1>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_2 idx(!VPURegMapped.Index<0:0:2>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_3 idx(!VPURegMapped.Index<0:0:3>) <DPUInvariant>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_4 idx(!VPURegMapped.Index<0:0:4>) <DPUInvariant>
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    // Use case #1a: u8 DPU in, u8 DPU out - with activation scaling
        VPUIPDPU.DPUInvariant @DPUInvariant1a
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg1: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg2: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %se_in_seg3: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.IDUStorageElement se_size(32) num_ses_in_z_dir(2)
                VPUIPDPU.IDUKernel kernel_x(1) kernel_y(2)
                VPUIPDPU.IDUStride stride_x(1) stride_y(2)
                VPUIPDPU.IDUInputLayerCfg sparsity_pattern(7)
                VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
                VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true) dw_opt_offset(16)
            }
            // CHECK:  UINT num_ses_in_z_dir = 2
            // CHECK:  UINT cm_sp_pattern = 7
            // CHECK:  UINT npo2_se_z_split_en = 0
            // CHECK:  UINT kernel_y = 2
            // CHECK:  UINT kernel_x = 1
            // CHECK:  UINT act_dense = 1
            // CHECK:  UINT wt_dense = 1
            // CHECK:  UINT stride_y_en = 1
            // CHECK:  UINT stride_y = 1
            // CHECK:  UINT layer1_cmp_en = 0
            // CHECK:  UINT pool_opt_en = 1
            // CHECK:  UINT tensor_size_x = 0x10
            // CHECK:  UINT tensor_size_y = 0x10
            // CHECK:  UINT tensor_size_z = 0x10
            // CHECK:  UINT amode = 0
            // CHECK:  UINT stride = 0
            // CHECK:  UINT zm_input = 1
            // CHECK:  UINT workload_operation = 2
            // CHECK:  UINT pool_wt_data = 0
            // CHECK:  UINT dw_opt_offset = 0x10
            // CHECK:  UINT dw_opt_en = 1
            // CHECK:  UINT dw_3x3s1_opt_dis = 1

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
	            VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }
            // CHECK:  UINT dtype = 4
            // CHECK:  UINT wcb_ac_mode = 1
            // CHECK:  UINT wcb_sp_mode = 0
            // CHECK:  UINT sp_value = 6
            // CHECK:  UINT sp_out_en = 1
            // CHECK:  UINT cmx_port_muxing_disable = 0
            // CHECK:  UINT write_sp = 0
            // CHECK:  UINT write_ac = 1
            // CHECK:  UINT mode = 0
            // CHECK:  UINT swizzle_key = 1
            // CHECK:  UINT nthw = 2
            // CHECK:  UINT permutation = 1
            // CHECK:  UINT wcb_bypass = 0
            // CHECK:  UINT te_dim_y = 0x40
            // CHECK:  UINT te_dim_z = 0x10
            // CHECK:  UINT te_dim_x = 0x40

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
                VPUIPDPU.PPEFpConvert convert_mode(NONE)
                VPUIPDPU.PPEIntBiasAdd %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntScaleMult %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                VPUIPDPU.PPEIntScaleShift %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                VPUIPDPU.PPEIntRound round_mode(RNE)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
                VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
                VPUIPDPU.PPEIntConvert convert_mode(NONE)

            }
            // CHECK:  SINT ppe_g8_bias_c = 0x180
            // Note:              0x180(Hex) = 384(Dec:uint64_t) is 2's complement representation of -128(Dec:int64_t)
            // CHECK:  UINT ppe_scale_round = 0
            // CHECK:  UINT ppe_scale_override = 0
            // CHECK:  UINT ppe_prelu_shift = 0
            // CHECK:  UINT ppe_prelu_mult = 1
            // CHECK:  SINT ppe_int_scale_hclamp = 0xFF
            // CHECK:  SINT ppe_int_scale_lclamp = 0
            // CHECK:  UINT ppe_i32_convert = 0
            // CHECK:  UINT ppe_fp_convert = 0
            // CHECK:  UINT ppe_fp_bypass = 1

            VPUIPDPU.BarrierCfg waits([3 : ui8, 5 : ui8]) updates([1 : ui8, 7 : ui8, 8 : ui8]) start_after(0) clean_after(0)
            // CHECK:  UINT barriers_wait_mask_hi_ = 0
            // CHECK:  barriers_wait_mask_lo_ = UINT 0x28
            // CHECK:  UINT barriers_post_mask_hi_ = 0
            // CHECK:  barriers_post_mask_lo_ = UINT 0x182
            // CHECK:  UINT group_ = 1
            // CHECK:  UINT mask_ = 0x28
        }

    // Use case #1b: u8 DPU in, u8 DPU out - with activation truncation
        VPUIPDPU.DPUInvariant @DPUInvariant1b
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:1>, task_location = @DeclareTaskBuffer_DPUInvariant_1,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %out_activations: memref<1x16x1x1xi8, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.IDUStorageElement se_size(32)
                VPUIPDPU.IDUKernel kernel_x(11) kernel_y(11)
                VPUIPDPU.IDUStride stride_x(8) stride_y(8)
                VPUIPDPU.IDUInputLayerCfg sparsity_pattern(31) {input_compressed}
                VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            }
            // CHECK:  UINT cm_sp_pattern = 0x1F
            // CHECK:  UINT npo2_se_z_split_en = 0
            // CHECK:  UINT kernel_y = 0xB
            // CHECK:  UINT kernel_x = 0xB
            // CHECK:  UINT act_dense = 1
            // CHECK:  UINT wt_dense = 1
            // CHECK:  UINT layer1_cmp_en = 1
            // CHECK:  UINT tensor_size_x = 0x10
            // CHECK:  UINT tensor_size_y = 0x10
            // CHECK:  UINT tensor_size_z = 0x10
            // CHECK:  UINT amode = 0
            // CHECK:  UINT stride = 7
            // CHECK:  UINT dw_input = 1
            // CHECK:  UINT workload_operation = 2
            // CHECK:  UINT pool_wt_data = 0

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(33) dim_z(17)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_5)
                VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x1x1xi8, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
	            VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }

            // CHECK:  UINT dtype = 3
            // CHECK:  UINT wcb_ac_mode = 0
            // CHECK:  UINT wcb_sp_mode = 0
            // CHECK:  UINT sp_value = 6
            // CHECK:  UINT sp_out_en = 1
            // CHECK:  UINT cmx_port_muxing_disable = 0
            // CHECK:  UINT write_sp = 0
            // CHECK:  UINT write_ac = 1
            // CHECK:  UINT mode = 0
            // CHECK:  UINT swizzle_key = 5
            // CHECK:  UINT nthw = 3
            // CHECK:  UINT permutation = 3
            // CHECK:  UINT wcb_bypass = 0
            // CHECK:  UINT te_dim_y = 0x20
            // CHECK:  UINT te_dim_z = 0x10
            // CHECK:  UINT te_dim_x = 0x40

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
            // CHECK:  SINT ppe_g8_bias_c = 0
            // CHECK:  ppe_bias = SINT 0
            // CHECK:  UINT ppe_scale_shift = 0
            // CHECK:  UINT ppe_scale_round = 3
            // CHECK:  SINT ppe_scale_mult = 1
            // CHECK:  UINT ppe_scale_override = 1
            // CHECK:  UINT ppe_prelu_shift = 0
            // CHECK:  SINT ppe_int_scale_hclamp = 0xFF
            // CHECK:  SINT ppe_int_scale_lclamp = 0
            // CHECK:  UINT ppe_i32_convert = 0
            // CHECK:  UINT ppe_fp_convert = 0
            // CHECK:  UINT ppe_fp_bypass = 1

            VPUIPDPU.BarrierCfg waits([0 : ui8]) updates([]) start_after(0) clean_after(0)
            // CHECK:  UINT barriers_wait_mask_hi_ = 0
            // CHECK:  barriers_wait_mask_lo_ = UINT 1
            // CHECK:  UINT barriers_post_mask_hi_ = 0
            // CHECK:  barriers_post_mask_lo_ = UINT 0
            // CHECK:  UINT group_ = 1
            // CHECK:  UINT mask_ = 1
        }

    // Use case #2: u8 DPU in, fp16 DPU out
        VPUIPDPU.DPUInvariant @DPUInvariant2
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:2>, task_location = @DeclareTaskBuffer_DPUInvariant_2,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %out_activations: memref<1x16x64x64xi32, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi32, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_ADDRESS)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
	            VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }
            // CHECK:  UINT dtype = 5
            // CHECK:  UINT wcb_ac_mode = 1
            // CHECK:  UINT wcb_sp_mode = 1
            // CHECK:  UINT sp_value = 6
            // CHECK:  UINT sp_out_en = 1
            // CHECK:  UINT cmx_port_muxing_disable = 0
            // CHECK:  UINT write_sp = 1
            // CHECK:  UINT write_ac = 1
            // CHECK:  UINT mode = 0
            // CHECK:  UINT swizzle_key = 1
            // CHECK:  UINT nthw = 2
            // CHECK:  UINT permutation = 1
            // CHECK:  UINT wcb_bypass = 0
            // CHECK:  UINT te_dim_y = 0x40
            // CHECK:  UINT te_dim_z = 0x10
            // CHECK:  UINT te_dim_x = 0x40

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEDenormalOperandsFTZ
            }
            // CHECK:  UINT mpe_daz = 1

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)

                VPUIPDPU.PPEFpConvert convert_mode(NONE)
                VPUIPDPU.PPEIntBiasAdd %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntScaleMult %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(0)
                VPUIPDPU.PPEIntScaleShift %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEIntConvert convert_mode(FP16)
                VPUIPDPU.PPEIntClamp clamp_high(70) // 70 corresponds to RELU6
            }
            // CHECK:  UINT ppe_scale_override = 0
            // CHECK:  UINT ppe_prelu_mult = 0
            // CHECK:  SINT ppe_int_scale_hclamp = 0x46
            // CHECK:  UINT ppe_i32_convert = 1
            // CHECK:  UINT ppe_fp_convert = 0
            // CHECK:  UINT ppe_fp_bypass = 1
        }

    // Use case #3: fp16 DPU in, fp16 DPU out
        VPUIPDPU.DPUInvariant @DPUInvariant3
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_3,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %out_activations: memref<1x16x64x64xi8, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
	            VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }
            // CHECK:  UINT dtype = 3
            // CHECK:  UINT wcb_ac_mode = 1
            // CHECK:  UINT wcb_sp_mode = 0
            // CHECK:  UINT sp_value = 6
            // CHECK:  UINT sp_out_en = 1
            // CHECK:  UINT cmx_port_muxing_disable = 0
            // CHECK:  UINT write_sp = 1
            // CHECK:  UINT write_ac = 1
            // CHECK:  UINT mode = 0
            // CHECK:  UINT swizzle_key = 1
            // CHECK:  UINT nthw = 2
            // CHECK:  UINT permutation = 1
            // CHECK:  UINT wcb_bypass = 0
            // CHECK:  UINT te_dim_y = 0x40
            // CHECK:  UINT te_dim_z = 0x1FFF
            // CHECK:  UINT te_dim_x = 0x40

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }
            // CHECK:  UINT mpe_actbias = 0xC
            // CHECK:  UINT mpe_daz = 0

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScalePreluMult %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]> prelu_alpha(0.1)
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
                VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON) ftz_mode(OFF)
                VPUIPDPU.PPEIntBiasAdd bias_static(0)
                VPUIPDPU.PPEIntScaleMult scale_static(1)
                VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                VPUIPDPU.PPEIntScaleShift shift_static(40)
                VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                VPUIPDPU.PPEIntRound round_mode(NONE)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647) // (MIN_I32, MAX_I32)
                VPUIPDPU.PPEIntConvert convert_mode(NONE)
            }
            // CHECK:  SINT ppe_g8_bias_c = 0
            // CHECK:  ppe_bias = SINT 0
            // CHECK:  UINT ppe_scale_shift = 0x28
            // CHECK:  UINT ppe_scale_round = 3
            // CHECK:  SINT ppe_scale_mult = 1
            // CHECK:  UINT ppe_scale_override = 1
            // CHECK:  UINT ppe_fp_scale_override = 0
            // CHECK:  UINT ppe_prelu_shift = 0
            // CHECK:  UINT ppe_prelu_mult = 1
            // CHECK:  SINT ppe_int_scale_hclamp = 0x7FFFFFFF
            // CHECK:  SINT ppe_int_scale_lclamp = 0x80000000
            // Note:              0x80000000(Hex) = 2147483648(Dec:uint64_t) is 2's complement representation of = -2147483648(Dec:int32_t)
            // CHECK:  UINT ppe_fp16_ftz = 0
            // CHECK:  UINT ppe_fp16_clamp = 1
            // CHECK:  UINT ppe_i32_convert = 0
            // CHECK:  ppe_fp_prelu = FP 0x3DCCCCCD
            // Note:              0x3DCCCCCD(Hex) = 1036831949(Dec) = 0.1 in IEEE floating-point format
            // CHECK:  UINT ppe_fp_convert = 1
            // CHECK:  UINT ppe_fp_bypass = 0
            // CHECK:  UINT ppe_fp_prelu_en = 1
        }

    // Use case #4: fp16 DPU in, u8 DPU out
        VPUIPDPU.DPUInvariant @DPUInvariant4
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:4>, task_location = @DeclareTaskBuffer_DPUInvariant_4,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
	            VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }
            // CHECK:  UINT dtype = 5
            // CHECK:  UINT wcb_ac_mode = 0
            // CHECK:  UINT wcb_sp_mode = 0
            // CHECK:  UINT sp_value = 0
            // CHECK:  UINT sp_out_en = 0
            // CHECK:  UINT cmx_port_muxing_disable = 0
            // CHECK:  UINT write_sp = 0
            // CHECK:  UINT write_ac = 1
            // CHECK:  UINT mode = 0
            // CHECK:  UINT swizzle_key = 1
            // CHECK:  UINT nthw = 1
            // CHECK:  UINT permutation = 1
            // CHECK:  UINT wcb_bypass = 0
            // CHECK:  UINT te_dim_y = 0x40
            // CHECK:  UINT te_dim_z = 0x10
            // CHECK:  UINT te_dim_x = 0x40

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEWeightsBias weights_bias(10)
            }
            // CHECK:  UINT mpe_wtbias = 0xA
            // CHECK:  UINT mpe_daz = 0

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScalePreluMult %weight_table:memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]> prelu_alpha(0.1)
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
            // CHECK:  SINT ppe_g8_bias_c = 0x180
            // Note:              0x180(Hex) = 384(Dec:uint64_t) is 2's complement representation of -128(Dec:int64_t)
            // CHECK:  ppe_bias = SINT 0
            // CHECK:  UINT ppe_scale_shift = 0
            // CHECK:  UINT ppe_scale_round = 3
            // CHECK:  SINT ppe_scale_mult = 1
            // CHECK:  UINT ppe_scale_override = 1
            // CHECK:  UINT ppe_fp_scale_override = 0
            // CHECK:  UINT ppe_prelu_shift = 0
            // CHECK:  UINT ppe_prelu_mult = 1
            // CHECK:  SINT ppe_int_scale_hclamp = 0xFF
            // CHECK:  SINT ppe_int_scale_lclamp = 0
            // CHECK:  UINT ppe_fp_convert = 4
            // CHECK:  UINT ppe_fp_bypass = 0
        }

    // Use case #5: default initialized values
        VPUIPDPU.DPUInvariant @DPUInvariant5
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:5>, task_location = @DeclareTaskBuffer_DPUInvariant_5,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            // CHECK:  cmx_slice0_low_addr = UINT 0x4000000
            // CHECK:  cmx_slice1_low_addr = UINT 0x4000000
            // CHECK:  cmx_slice2_low_addr = UINT 0x4000000
            // CHECK:  cmx_slice3_low_addr = UINT 0x4000000
            // CHECK:  cmx_slice_size = UINT 0x18000
            // CHECK:  UINT cm_sp_pattern = 0
            // CHECK:  UINT pool_wt_data = 0
            // CHECK:  UINT ppe_scale_round = 3
            // CHECK:  UINT ppe_scale_override = 0
            // CHECK:  UINT ppe_fp_scale_override = 0
            // CHECK:  UINT ppe_prelu_mult = 1
            // CHECK:  SINT ppe_int_scale_hclamp = 0x7FFFFFFF
            // CHECK:  SINT ppe_int_scale_lclamp = 0x80000000
            // Note:              0x80000000(Hex) = 2147483648(Dec:uint64_t) is 2's complement representation of = -2147483648(Dec:int32_t)
        }

        return
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ProfilingTest {
  net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xf16>
      DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x16x64x64xf16>
  } profilingOutputsInfo : {
      DataInfo "profilingOutput" : tensor<32xui32>
  }
  func.func @main() {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>

    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ProfilingData !VPUASM.Buffer< "CMX_NN"[0] <139392> : memref<4xui64, [@CMX_NN, 0]> :  swizzling(0)>

    // Use case: default initialized values + profiling
        VPUIPDPU.DPUInvariant @DPUInvariantProf
        // CHECK-NOT:   VPUIPDPU.DPUInvariant
        // CHECK:       NPUReg50XX.DPUInvariant
            {task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, profiling_data = @DeclareBuffer_ProfilingData,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            // CHECK:  hwp_ctrl {
            // CHECK:  UINT hwp_en = 1
            // CHECK:  UINT hwp_stat_mode = 7
        }

        return
    }
}
