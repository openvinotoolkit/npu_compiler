//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUASM-to-NPUReg60XX --convert-VPUIPDPU-to-NPUReg60XX  %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xf16>
      DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x16x64x64xf16>
  }
  func.func @main() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_1 idx(!VPURegMapped.Index<0:0:1>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_2 idx(!VPURegMapped.Index<0:0:2>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_3 idx(!VPURegMapped.Index<0:0:3>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_4 idx(!VPURegMapped.Index<0:0:4>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_5 idx(!VPURegMapped.Index<0:0:4>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_6 idx(!VPURegMapped.Index<0:0:4>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            // Use case #1a: u8 DPU in, u8 DPU out - with activation scaling
            VPUIPDPU.DPUInvariant @DPUInvariant1a
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %se_in_seg0: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %se_in_seg1: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %se_in_seg2: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %se_in_seg3: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUStorageElement se_size(32) num_ses_in_z_dir(2)
                    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(2)
                    VPUIPDPU.IDUStride stride_x(1) stride_y(2)
                    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(7)
                    VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
                    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
                    VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true) dw_opt_offset(16)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(0.0) clamp_high(255.0)
                    VPUIPDPU.PPEFpConvert convert_mode(I32)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
                }
                VPUIPDPU.BarrierCfg waits([3 : ui16, 5 : ui16]) updates([1 : ui16, 7 : ui16, 8 : ui16]) start_after(0) clean_after(0)
            }
            VPUIPDPU.DPUInvariant @DPUInvariant1b

                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:7>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>}>
                DPUCfg  : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUMSQUARE)
                    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEActivationBias act_bias(12)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38) // (MIN_FP32, MAX_FP32)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                }
            }
            // Use case #1c: u8 DPU in, u8 DPU out - with activation truncation
            VPUIPDPU.DPUInvariant @DPUInvariant1c
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:1>, task_location = @DeclareTaskBuffer_DPUInvariant_1,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUStorageElement se_size(32)
                    VPUIPDPU.IDUKernel kernel_x(11) kernel_y(11)
                    VPUIPDPU.IDUStride stride_x(8) stride_y(8)
                    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(31) <{input_compressed}>
                    VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
                    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(33) dim_z(17)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_5)
                    VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpClamp clamp_low(0.0) clamp_high(255.0)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpConvert convert_mode(I32)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                }
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([]) start_after(0) clean_after(0)
            }

            // Use case #2: u8 DPU in, fp16 DPU out
            VPUIPDPU.DPUInvariant @DPUInvariant2
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:2>, task_location = @DeclareTaskBuffer_DPUInvariant_2,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_ADDRESS)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEDenormalOperandsFTZ
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(70.0) // 70 corresponds to RELU6
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                }
            }

            // Use case #3: fp16 DPU in, fp16 DPU out
            VPUIPDPU.DPUInvariant @DPUInvariant3
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_3,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEActivationBias act_bias(12)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38) // (MIN_FP32, MAX_FP32)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                }
            }
            // Use case #4: fp16 DPU in, u8 DPU out + reduce output: min_per_xy, max_per_xy, min_max_per_tensor
            VPUIPDPU.DPUInvariant @DPUInvariant4
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:4>, task_location = @DeclareTaskBuffer_DPUInvariant_4,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    min_per_xy = @DeclareBuffer_MinPerXy,
                    max_per_xy = @DeclareBuffer_MaxPerXy,
                    min_max_per_tensor = [@DeclareBuffer_MinMaxPerTensor],
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %min_per_xy: memref<1x1x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %max_per_xy:  memref<1x1x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %min_max_per_tensor: memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                }

                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                    VPUIPDPU.ODUReduceLayer reduce_dtype(ODU_DTYPE_U8) <{reduce_max, reduce_min, reduce_total}>
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEWeightsBias weights_bias(10)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                    VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(0.0) clamp_high(255.0)
                    VPUIPDPU.PPEFpConvert convert_mode(I32)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
                }
            }

            // Use case #5: default initialized values
            VPUIPDPU.DPUInvariant @DPUInvariant5
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:5>, task_location = @DeclareTaskBuffer_DPUInvariant_5,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                    VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                }
            }

            // Use case #6: default values, PPE sprLUT mode ON
            VPUIPDPU.DPUInvariant @DPUInvariant6
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:6>, task_location = @DeclareTaskBuffer_DPUInvariant_6,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                DPUCfg : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(17)
                    VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(ON)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                }
            }
            // Use case #6b: fp16 DPU in, fp16 DPU out + reduce output: min_per_xy, max_per_xy, min_max_per_tensor_2_tiles
            VPUIPDPU.DPUInvariant @DPUInvariant6b
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64, min_max_per_tensor0 = @DeclareBuffer_MinMaxPerTensor0, min_max_per_tensor1 = @DeclareBuffer_MinMaxPerTensor1}
                <{task_index = !VPURegMapped.Index<0:0:6>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    min_per_xy = @DeclareBuffer_MinPerXy,
                    max_per_xy = @DeclareBuffer_MaxPerXy,
                    nce_task_type = #VPUIP.nce_task_type<REDUCEMEAN>}>
                DPUCfg  : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>,
                     %min_per_xy: memref<1x1x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %max_per_xy:  memref<1x1x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %min_max_per_tensor0: memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 1]>,
                     %min_max_per_tensor1: memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCEMEAN)
                    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                    VPUIPDPU.ODUReduceLayer reduce_dtype(ODU_DTYPE_FP16) tile_select(2) <{reduce_max, reduce_min, reduce_total}>
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEActivationBias act_bias(12)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38) // (MIN_FP32, MAX_FP32)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                }
            }

            // Use case #7: fp16 DPU in, fp16 DPU out
            VPUIPDPU.DPUInvariant @DPUInvariant7
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                <{task_index = !VPURegMapped.Index<0:0:7>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>}>
                DPUCfg  : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUMSQUARE)
                    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                }
                 VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEActivationBias act_bias(12)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38) // (MIN_FP32, MAX_FP32)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                }
            }
            // Use case #8: fp16 DPU in, fp16 DPU out + reduce output: min_per_xy, max_per_xy, min_max_per_tensor_2_tiles
            VPUIPDPU.DPUInvariant @DPUInvariant8
                {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64, min_max_per_tensor0 = @DeclareBuffer_MinMaxPerTensor0, min_max_per_tensor1 = @DeclareBuffer_MinMaxPerTensor1}
                <{task_index = !VPURegMapped.Index<0:0:6>, task_location = @DeclareTaskBuffer_DPUInvariant_0,
                    input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                    min_per_xy = @DeclareBuffer_MinPerXy,
                    max_per_xy = @DeclareBuffer_MaxPerXy,
                    nce_task_type = #VPUIP.nce_task_type<REDUCESUM>}>
                DPUCfg  : {
                ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>,
                     %min_per_xy: memref<1x1x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %max_per_xy:  memref<1x1x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                     %min_max_per_tensor0: memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 1]>,
                     %min_max_per_tensor1: memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUM)
                    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                }
                 VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                    VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                    VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                    VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                    VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
                    VPUIPDPU.ODUReduceLayer reduce_dtype(ODU_DTYPE_FP16) tile_select(2) <{reduce_max, reduce_min, reduce_total}>
                }
                VPUIPDPU.MPECfg {
                    VPUIPDPU.MPEActivationBias act_bias(12)
                }
                VPUIPDPU.PPECfg {
                    VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                    VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38) // (MIN_FP32, MAX_FP32)
                    VPUIPDPU.PPEFpConvert convert_mode(FP16)
                    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                }
            }
        }
      ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
          VPUASM.DpuPVP @DpuPVP_0_0_0
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)
      }
    }
    return
    }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0xA9A86CF,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_NoWtReads inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_output_f16_wt_split_NoWtReads() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0
            <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
                task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
                VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38)
                VPUIPDPU.PPEFpConvert convert_mode(FP16)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_WeightsOnly inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_output_f16_wt_split_WeightsOnly() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <98304> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtScale !VPUASM.Buffer< "CMX_NN"[0] <98560> : memref<64x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtBias !VPUASM.Buffer< "CMX_NN"[0] <98816> : memref<64x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0
            <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
                task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0,
                weight_table_scale = @buffer.CMX_NN.0::@DeclareBuffer_WtScale, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                VPUIPDPU.WeightTableSplit wt_split_mode(WEIGHTS_ONLY) scale_dtype(FP32)
            }

                VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
                VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38)
                VPUIPDPU.PPEFpConvert convert_mode(FP16)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x21020000,

// -----

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0
            <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
                task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                VPUIPDPU.WeightTableSplit wt_split_mode(ZERO_POINTS_ONLY)
                VPUIPDPU.IDUSubtractMpeWtBias
            }

                VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38)
                VPUIPDPU.PPEFpConvert convert_mode(FP16)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_WeightsAndZeroPoints inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_output_f16_wt_split_WeightsAndZeroPoints() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0
            <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
             task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                VPUIPDPU.WeightTableSplit wt_split_mode(WEIGHTS_AND_ZPS)
            }

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpPreluMult prelu_alpha(0.1)
                VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38)
                VPUIPDPU.PPEFpConvert convert_mode(FP16)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_legacy_wt_Weights inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_output_f16_legacy_wt_Weights() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0
            <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
             task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                VPUIPDPU.WeightTableLegacyZp wt_legacy_zp_mode(WEIGHTS_PTR_AND_ZPS)
            }

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_legacy_wt_WeightsAndZeroPoints inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_output_f16_legacy_wt_WeightsAndZeroPoints() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0
            <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
             task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(0x3c00)
                VPUIPDPU.WeightTableLegacyZp wt_legacy_zp_mode(WEIGHTS_PTR_AND_ZPS)
            }

            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZYX)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> compression_enabled(true) sparse_value(6)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
                VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
                VPUIPDPU.ODUMemoryMode mem_mode(MODE_DENSE)
                VPUIPDPU.ODUCmxPorts cmx_ports(CMX_PORTS_ALL)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

// Palletized weights (4 bit -> fp16)

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_u4_pallet_fp16 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_weights_u4_pallet_fp16() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <45312> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_fp16_in_u4_fp16_pallet_weights
            {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_fp16_in_u4_fp16_pallet_weights,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %pallet_lookup_table: memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(FOUR_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 0.69999999999999996, 1.000000e+00])
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

// Palletized weights (1 bit -> fp16)

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_u1_pallet_fp16 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_weights_u1_pallet_fp16() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <45312> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_fp16_in_u2_fp16_pallet_weights
            {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_fp16_in_u1_fp16_pallet_weights,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %pallet_lookup_table: memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(ONE_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01])
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

// Palletized weights (2 bit -> fp16)

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_u2_pallet_fp16 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_weights_u2_pallet_fp16() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <45312> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_fp16_in_u2_fp16_pallet_weights
            {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_fp16_in_u2_fp16_pallet_weights,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %pallet_lookup_table: memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(TWO_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01])
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

// Palletized weights (3 bit -> fp16)

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_u3_pallet_fp16 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_weights_u3_pallet_fp16() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <45312> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_fp16_in_u3_fp16_pallet_weights
            {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_fp16_in_u3_fp16_pallet_weights,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %pallet_lookup_table: memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(THREE_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00])
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

// Palletized weights (5 bit -> fp16)

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_u5_pallet_fp16 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_weights_u5_pallet_fp16() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <45312> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_fp16_in_u5_fp16_pallet_weights
            {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_fp16_in_u5_fp16_pallet_weights,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %pallet_lookup_table: memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(FIVE_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00])
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)

            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

// Palletized weights (6 bit -> fp16)

!qElemType = !quant.uniform<u8:f32:1, {1.000000e+00:1,1.000000e+00:255,1.000000e+00}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_u6_pallet_fp16 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x8x8xf16>
  }
  func.func @IDU_CONV_input_f16_weights_u6_pallet_fp16() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <45312> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_fp16_in_u5_fp16_pallet_weights
            {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            <{task_index = !VPURegMapped.Index<0:0:3>, task_location = @DeclareTaskBuffer_DPUInvariant_fp16_in_u6_fp16_pallet_weights,
                input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                    %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %pallet_lookup_table: memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>,
                    %out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(SIX_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, -1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00])
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(65) dim_y(65) dim_z(8192)
                VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<1x16x64x64xi8, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
            }

            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEActivationBias act_bias(12)
            }

            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult  %weight_table:memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpClamp clamp_low(-3.4028234e+38) clamp_high(3.4028234e+38) // (MIN_FP32, MAX_FP32)
            }
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x2102000,

// -----

!qElemType = !quant.uniform<i8:f16, 1.0000152590218967>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @CONV_input_I8_weights_I8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }
  func.func @CONV_input_I8_weights_I8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <99328> : memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <82944> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <81920> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, input_storage_element_table = @buffer.CMX_NN.0::@DeclareBuffer_InputSETable,
        nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0,
        weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
        DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>, %arg4: memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{in_sparse}>
            VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUStride stride_x(2) stride_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
            VPUIPDPU.ODUOutActivations out_activations(%arg4 : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT)
          }
          VPUIPDPU.MPECfg {
            VPUIPDPU.MPEWeightsBias weights_bias(0)
            VPUIPDPU.MPEActivationBias act_bias(-22)
            VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpBiasAdd %arg2 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
            VPUIPDPU.PPEFpScaleMult %arg2 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
            VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e-01)
            VPUIPDPU.PPEFpClamp clamp_low(-3.276800e+04) clamp_high(3.276700e+04)
            VPUIPDPU.PPEFpConvert convert_mode(I32)
            VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-2)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x1000000,

// -----

!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2222>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @CONV_input_I16_weights_I8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }
  func.func @CONV_input_I16_weights_I8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <197632> : memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, input_storage_element_table = @buffer.CMX_NN.0::@DeclareBuffer_InputSETable, nce_task_type = #VPUIP.nce_task_type<CONV>,
         output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0,
         weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
         DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>, %arg4: memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{in_sparse}>
            VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUStride stride_x(2) stride_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
            VPUIPDPU.ODUOutActivations out_activations(%arg4 : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.MPECfg {
            VPUIPDPU.MPEWeightsBias weights_bias(0)
            VPUIPDPU.MPEActivationBias act_bias(2222)
            VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpBiasAdd %arg2 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
            VPUIPDPU.PPEFpScaleMult %arg2 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
            VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e-01)
            VPUIPDPU.PPEFpClamp clamp_low(-3.276800e+04) clamp_high(3.276700e+04)
            VPUIPDPU.PPEFpConvert convert_mode(I32)
            VPUIPDPU.PPEIntZeroPointOffset zero_point_static(2222)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
          VPUASM.DpuPVP @DpuPVP_0_0_0
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)
      }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x1000000,


// -----

!qElemType = !quant.uniform<u16:f16, 5.000000e-01:2222>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_CONV_input_I16_weights_I8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }
  func.func @IDU_CONV_input_I16_weights_I8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <197632> : memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, input_storage_element_table = @buffer.CMX_NN.0::@DeclareBuffer_InputSETable,
        nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0,
         weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
         DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>, %arg4: memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{in_sparse}>
            VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUStride stride_x(2) stride_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
            VPUIPDPU.ODUOutActivations out_activations(%arg4 : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.MPECfg {
            VPUIPDPU.MPEWeightsBias weights_bias(0)
            VPUIPDPU.MPEActivationBias act_bias(2222)
            VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpBiasAdd %arg2 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
            VPUIPDPU.PPEFpScaleMult %arg2 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
            VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e-01)
            VPUIPDPU.PPEFpClamp clamp_low(0.000000e+00) clamp_high(6.553500e+04)
            VPUIPDPU.PPEFpConvert convert_mode(I32)
            VPUIPDPU.PPEIntZeroPointOffset zero_point_static(2222)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
          VPUASM.DpuPVP @DpuPVP_0_0_0
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)
      }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x1000000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
    net.NetworkInfo entryPoint : @dynamic_shape_invar inputsInfo : {
        DataInfo "input_0" : tensor<1x64x16x16xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x64x8x8xf16>
    }
    func.func @dynamic_shape_invar() {
        ELF.Main {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant1 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_1 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant2 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_2 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant3 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_3 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUIPDPU.DPUInvariant @DPUInvariant_0 {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64} <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0,
                input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %weight_table: memref<16x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                    }
                        VPUIPDPU.MPECfg {
                        VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
                    }
                        VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
                        VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
                        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
                        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
                        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
                        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(8) dim_y(8) dim_z(64)
                        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    }

                    VPUIPDPU.DynScaleCfgInv  dyn_primitive_id(8) dyn_primitive_id2(4) dyn_ppe_data(0x7777) dynamic_skip_mode(#VPUIP.dynamic_skip_mode<SPATIAL_AND_OUTPUT_CHANNELS>) dynamic_adjust_mode(#VPUIP.dynamic_adjust_mode<OUTPUT_CHANNELS>)
               }
                VPUIPDPU.DPUInvariant @DPUInvariant_1 {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64} <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant1::@DeclareTaskBuffer_DPUInvariant_1,
                 input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %weight_table: memref<16x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                    }
                        VPUIPDPU.MPECfg {
                        VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
                    }
                        VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
                        VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
                        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
                        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
                        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
                        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(8) dim_y(8) dim_z(64)
                        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    }

                    VPUIPDPU.DynScaleCfgInv dyn_primitive_id(8) dyn_primitive_id2(4) dynamic_skip_mode(#VPUIP.dynamic_skip_mode<SPATIAL_AND_INPUT_CHANNELS>) dynamic_adjust_mode(#VPUIP.dynamic_adjust_mode<INPUT_CHANNELS>) <{dyn_conv_cont_mod_en}>
               }

               VPUIPDPU.DPUInvariant @DPUInvariant_2 {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64} <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant2::@DeclareTaskBuffer_DPUInvariant_2,
               input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %weight_table: memref<16x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                    }
                        VPUIPDPU.MPECfg {
                        VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
                    }
                        VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
                        VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
                        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
                        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
                        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
                        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(8) dim_y(8) dim_z(64)
                        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    }

                    VPUIPDPU.DynScaleCfgInv dyn_primitive_id(8) dynamic_skip_mode(#VPUIP.dynamic_skip_mode<SPATIAL_ONLY>) dynamic_adjust_mode(#VPUIP.dynamic_adjust_mode<DISABLED>)
               }

                VPUIPDPU.DPUInvariant @DPUInvariant_3 {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64} <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant3::@DeclareTaskBuffer_DPUInvariant_3,
                input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<CONV>}>
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %weight_table: memref<16x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                    }
                        VPUIPDPU.MPECfg {
                        VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
                    }
                        VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
                        VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
                        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
                        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
                        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
                        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(8) dim_y(8) dim_z(64)
                        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    }

                    VPUIPDPU.DynScaleCfgInv dyn_primitive_id(8) dyn_primitive_id2(4) dyn_ppe_data(-256) dynamic_skip_mode(#VPUIP.dynamic_skip_mode<SPATIAL_AND_OUTPUT_CHANNELS>) dynamic_adjust_mode(#VPUIP.dynamic_adjust_mode<DISABLED>)
               }
            }

            ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUIPDPU.DPUVariant @DPUVariant1 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<CONV>}
                DPUCfg: {
                        VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                        VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(64)
                        VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                        VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                        VPUIPDPU.IDUSEDense
                        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                        VPUIPDPU.BarrierCfg waits([2 : ui16]) updates([3 : ui16]) start_after(0) clean_after(0)
                        VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
                  }
            }
            ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUASM.DpuPVP @DpuPVP_0_0_0
            }
            ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
                ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
                ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
                ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

            }
        }

        return
    }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x40000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_ACCU_STORE_FP32 inputsInfo : {
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x16x16xf32>
  }
  func.func @IDU_ACCU_STORE_FP32() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>,
            task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}> DPUCfg : {
            ^bb0(%arg0: memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_STORE_FP32)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUOutActivations out_activations(%arg0 : memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
            }
            VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
            VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
            }

            VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
                VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                VPUIPDPU.IDUSEDense
                VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
                VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)
      }
    }
    return
    }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x4000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_ACCU_STORE_I32 inputsInfo : {
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x16x16xsi32>
  }
  func.func @IDU_ACCU_STORE_I32() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
            task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}> DPUCfg : {
            ^bb0(%arg0: memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_STORE_I32)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUOutActivations out_activations(%arg0 : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
            }
            VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
            VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
            }

            VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
                VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                VPUIPDPU.IDUSEDense
                VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
                VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
    }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x4000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_ACCU_LOAD_FP32 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xf32>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x16x16xf32>
  }
  func.func @IDU_ACCU_LOAD_FP32() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
                task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}> DPUCfg : {
            ^bb0(%arg0: memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>, %arg1 : memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUWeights wmode(f32) wt_plt_cfg(NO_PLT)
                    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
                    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
                    VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_LOAD_FP32)
                    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                    VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<1x64x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                }
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
                VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
            }

            VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
                VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                VPUIPDPU.IDUSEDense
                VPUIPDPU.IDUConvContinue
                VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
                VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
    }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0x100000,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x100000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @IDU_ACCU_LOAD_I32 inputsInfo : {
      DataInfo "input_0" : tensor<1x64x16x16xsi32>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x16x16xsi32>
  }
  func.func @IDU_ACCU_LOAD_I32() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
            task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}>
            DPUCfg : {
            ^bb0(%arg0: memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg1 : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>):
                VPUIPDPU.IDUCfg {
                    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>)
                    VPUIPDPU.IDUWeights wmode(f32) wt_plt_cfg(NO_PLT)
                    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
                    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
                    VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_LOAD_I32)
                    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
                }
                VPUIPDPU.ODUCfg {
                    VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
                    VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                    VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_32BIT)
                }
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
                VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
            }

            VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
                VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                VPUIPDPU.IDUSEDense
                VPUIPDPU.IDUConvContinue
                VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
                VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
    }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x100000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @ODU_LOCAL_REGION_NO_SUPPRESS inputsInfo : {
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x16x16xsi32>
  }
  func.func @ODU_LOCAL_REGION_NO_SUPPRESS() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
            task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}>
            DPUCfg : {

            ^bb0(%act_in: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
            %weight_table: memref<16x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
            %act_out: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                VPUIPDPU.ODULocalRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(32) end_coord_y(32)
            }
            VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
            VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)

            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x100000,

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @ODU_LOCAL_REGION_SUPPRESS inputsInfo : {
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x64x16x16xsi32>
  }
  func.func @ODU_LOCAL_REGION_SUPPRESS() {
    ELF.Main {
        ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        }
        ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
            VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        }
        ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
            task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}> DPUCfg : {

            ^bb0(%act_in: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
            %weight_table: memref<16x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
            %act_out: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                VPUIPDPU.ODUOutActivations out_activations(%act_out : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                VPUIPDPU.ODULocalRegion begin_coord_x(16) begin_coord_y(16) end_coord_x(48) end_coord_y(48) <{suppress_output}>
            }
            VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
            VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)

            }
        }
        ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
            VPUASM.DpuPVP @DpuPVP_0_0_0
        }
        ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)

        }
    }
    return
  }
}

// CHECK: ELF.CreateSection @dpu.pvp
// CHECK:       VpuDpuPVP_opFPCount = UINT 0,
// CHECK:       VpuDpuPVP_totalOpCount = UINT 0x100000
