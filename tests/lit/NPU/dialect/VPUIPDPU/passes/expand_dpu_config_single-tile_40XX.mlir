//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @maxpool_f16_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    //CHECK:    VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0}> DPUCfg : {
    //CHECK:    ^bb0([[ARG_0:%.+]]: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, [[ARG_1:%.+]]: memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>):
    //CHECK:      VPUIPDPU.IDUCfg {
    //CHECK:        VPUIPDPU.IDUInActivations in_activations([[ARG_0]] : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        VPUIPDPU.IDUWeights wmode(f16)
    //CHECK:        VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    //CHECK:        VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    //CHECK:        VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
    //CHECK:        VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true)
    //CHECK:      }
    //CHECK:      VPUIPDPU.MPECfg {
    //CHECK:      }
    //CHECK:      VPUIPDPU.PPECfg {
    //CHECK:        VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    //CHECK:        VPUIPDPU.PPEFpConvert convert_mode(NONE)
    //CHECK:        VPUIPDPU.PPEIntBiasAdd bias_static(0)
    //CHECK:        VPUIPDPU.PPEIntScaleMult scale_static(1)
    //CHECK:        VPUIPDPU.PPEIntScaleShift shift_static(0)
    //CHECK:        VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
    //CHECK:        VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    //CHECK:        VPUIPDPU.PPEIntRound round_mode(NONE)
    //CHECK:        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    //CHECK:        VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
    //CHECK:        VPUIPDPU.PPEIntConvert convert_mode(NONE)
    //CHECK:      }
    //CHECK:      VPUIPDPU.ODUCfg {
    //CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(8) dim_y(8) dim_z(64)
    //CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
    //CHECK:        VPUIPDPU.ODUOutActivations out_activations([[ARG_1]] : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:      }
    //CHECK:      VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(0)
    //CHECK:      VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
    //CHECK:    }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0
        <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    //CHECK:       VPUIPDPU.DPUVariant
    //CHECK:      VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
    //CHECK:      VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
    //CHECK:      VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    //CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
    //CHECK:      VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(0)
    //CHECK:      VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)

    }
    return
  }
}
