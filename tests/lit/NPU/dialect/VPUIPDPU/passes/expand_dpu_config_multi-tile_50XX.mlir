//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16>
  } outputsInfo : {
  DataInfo "output_0" : tensor<1x64x16x32xf16>
  DataInfo "output_1" : tensor<1x64x16x32xf16>
  }

  func.func @multiple_clusters_dpu_soh_f16_f16_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_1_0_0 idx(!VPURegMapped.Index<1:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_1_0_0 idx(!VPURegMapped.Index<1:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <69632> : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <102400> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <4096> : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.1 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[1] <69632> : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[1] <0> : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[1] <102400> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[1] <4096> : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
        kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
        }
      }

    // CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
    // CHECK:    ^bb0(%arg0: memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>):
    // CHECK:      VPUIPDPU.IDUCfg {
    // CHECK:        VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:        VPUIPDPU.IDUWeights wmode(f16)
    // CHECK:        VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK:        VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK:        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK:      }
    // CHECK:      VPUIPDPU.MPECfg {
    // CHECK:      }
    // CHECK:      VPUIPDPU.PPECfg {
    // CHECK:        VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:        VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:        VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK:        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK:        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
    // CHECK:        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK:      }
    // CHECK:      VPUIPDPU.ODUCfg {
    // CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(32) dim_y(16) dim_z(64)
    // CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
    // CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:      }
    // CHECK:    }

      ELF.CreateSection @task.dpu.invariant.1.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_1_0 idx(!VPURegMapped.Index<1:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_1_0_0) input(@buffer.CMX_NN.1::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.1::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.1::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.1::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
        kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
        }
      }

    // CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_1_0 <{input = @buffer.CMX_NN.1::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.1::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<1:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_1_0_0, weight_table = @buffer.CMX_NN.1::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.1::@DeclareBuffer_Weights}> DPUCfg : {
    // CHECK:    ^bb0(%arg0: memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>, %arg2: memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>, %arg3: memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>):
    // CHECK:      VPUIPDPU.IDUCfg {
    // CHECK:        VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK:        VPUIPDPU.IDUWeights wmode(f16)
    // CHECK:        VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK:        VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK:        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK:      }
    // CHECK:      VPUIPDPU.MPECfg {
    // CHECK:      }
    // CHECK:      VPUIPDPU.PPECfg {
    // CHECK:        VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:        VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:        VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK:        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK:        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
    // CHECK:        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK:      }
    // CHECK:      VPUIPDPU.ODUCfg {
    // CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(32) dim_y(16) dim_z(64)
    // CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
    // CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK:      }
    // CHECK:    }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [31, 15, 63], inEnd = [31, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:   VPUIPDPU.DPUVariant
    // CHECK:      VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(16) size_z(32)
    // CHECK:      VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(32)
    // CHECK:      VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK:      VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
    // CHECK:      VPUIPDPU.IDUSEDense
    // CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(31) end_coord_y(15) end_coord_z(63)
    // CHECK:    }

      ELF.CreateSection @task.dpu.variant.1.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_1_0 idx(!VPURegMapped.Index<1:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_1_0_0) invariant @task.dpu.invariant.1.0::@DPUInvariant_1_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_1_0_0 weights(@buffer.CMX_NN.1::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.1::@DeclareBuffer_WeightsTable)
        <{end = [31, 31, 63], inEnd = [31, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]}>
      }

    // CHECK:   VPUIPDPU.DPUVariant
    // CHECK:      VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(16) size_z(32)
    // CHECK:      VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(32)
    // CHECK:      VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK:      VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
    // CHECK:      VPUIPDPU.IDUSEDense
    // CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(16) begin_coord_z(0) end_coord_x(31) end_coord_y(31) end_coord_z(63)
    // CHECK:    }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @multiple_clusters_dpu_sok_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
  DataInfo "output_0" : tensor<1x64x16x16xf16>
  DataInfo "output_1" : tensor<1x64x16x16xf16>
  }

  func.func @multiple_clusters_dpu_sok_f16_f16_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_1_0_0 idx(!VPURegMapped.Index<1:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_1_0_0 idx(!VPURegMapped.Index<1:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <34816> : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <51200> : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <2048> : memref<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.1 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[1] <34816> : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[1] <0> : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[1] <51200> : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[1] <2048> : memref<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
        }
      }

    // CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
    // CHECK:    ^bb0(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>):
    // CHECK:      VPUIPDPU.IDUCfg {
    // CHECK:        VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:        VPUIPDPU.IDUWeights wmode(f16)
    // CHECK:        VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK:        VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK:        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK:      }
    // CHECK:      VPUIPDPU.MPECfg {
    // CHECK:      }
    // CHECK:      VPUIPDPU.PPECfg {
    // CHECK:        VPUIPDPU.PPEFpBiasAdd %arg1 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:        VPUIPDPU.PPEFpScaleMult %arg1 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:        VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK:        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK:        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
    // CHECK:        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK:      }
    // CHECK:      VPUIPDPU.ODUCfg {
    // CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
    // CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
    // CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>)
    // CHECK:      }
    // CHECK:    }

      ELF.CreateSection @task.dpu.invariant.1.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_1_0 idx(!VPURegMapped.Index<1:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_1_0_0) input(@buffer.CMX_NN.1::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.1::@DeclareBuffer_Weights)
        weight_table(@buffer.CMX_NN.1::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.1::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
        }
      }

    // CHECK:   VPUIPDPU.DPUInvariant @DPUInvariant_1_0 <{input = @buffer.CMX_NN.1::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.1::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<1:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_1_0_0, weight_table = @buffer.CMX_NN.1::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.1::@DeclareBuffer_Weights}> DPUCfg : {
    // CHECK:    ^bb0(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>, %arg1: memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>, %arg2: memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>, %arg3: memref<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>):
    // CHECK:      VPUIPDPU.IDUCfg {
    // CHECK:        VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK:        VPUIPDPU.IDUWeights wmode(f16)
    // CHECK:        VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK:        VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK:        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK:      }
    // CHECK:      VPUIPDPU.MPECfg {
    // CHECK:      }
    // CHECK:      VPUIPDPU.PPECfg {
    // CHECK:        VPUIPDPU.PPEFpBiasAdd %arg1 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:        VPUIPDPU.PPEFpScaleMult %arg1 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:        VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
    // CHECK:        VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK:        VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK:        VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
    // CHECK:        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK:      }
    // CHECK:      VPUIPDPU.ODUCfg {
    // CHECK:        VPUIPDPU.ODUOutTensorSize dim_x(16) dim_y(16) dim_z(64)
    // CHECK:        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
    // CHECK:        VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>)
    // CHECK:      }
    // CHECK:    }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [15, 15, 31], inEnd = [31, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:   VPUIPDPU.DPUVariant
    // CHECK:      VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(16) size_z(32)
    // CHECK:      VPUIPDPU.IDUWeightSet weight_start(0) weight_num(32) weight_size(32)
    // CHECK:      VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK:      VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
    // CHECK:      VPUIPDPU.IDUSEDense
    // CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(15) end_coord_y(15) end_coord_z(31)
    // CHECK:    }

      ELF.CreateSection @task.dpu.variant.1.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_1_0 idx(!VPURegMapped.Index<1:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_1_0_0) invariant @task.dpu.invariant.1.0::@DPUInvariant_1_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_1_0_0 weights(@buffer.CMX_NN.1::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.1::@DeclareBuffer_WeightsTable)
        <{end = [15, 15, 63], inEnd = [31, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 32]}>
      }

    // CHECK:   VPUIPDPU.DPUVariant
    // CHECK:      VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(16) size_z(32)
    // CHECK:      VPUIPDPU.IDUWeightSet weight_start(512) weight_num(32) weight_size(32)
    // CHECK:      VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK:      VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
    // CHECK:      VPUIPDPU.IDUSEDense
    // CHECK:      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(32) end_coord_x(15) end_coord_y(15) end_coord_z(63)
    // CHECK:    }

    }
    return
  }
}
