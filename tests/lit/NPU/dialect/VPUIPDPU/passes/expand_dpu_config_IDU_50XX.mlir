//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_fp16_se_table_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_fp16_se_table_weights_fp16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <197632> : memref<1x16x64x64xi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>)  {in_sparse}{{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEOnly{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_SUB_input_fp16_sparse_se_table_weights_fp16_sparse inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_ELTWISE_SUB_input_fp16_sparse_se_table_weights_fp16_sparse() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <57360> : memref<1x64x16x16xi1, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <59408> : memref<1x1x16x16xi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <50192> : memref<64x1x1x896xi1, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActSparsityMap) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weights_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_WeightsSparsityMap) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, eltwise_type = #VPU.eltwise_type<SUBTRACT>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>)  {in_sparse}{{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) {wt_sparse}{{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(SUBTRACT){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}

        VPUASM.DPUVariant @DPUVariant_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_1) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(16384){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEOnly{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPerOutputChannelScaling {tensor2_act_sparse}{{$}}

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(16384){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEOnly{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPerOutputChannelScaling {tensor2_act_sparse}{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
!wqElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_bf8_output_bf8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf8E5M2>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf8E5M2>
  }

  func.func @IDU_CONV_input_bf8_output_bf8() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2x!wqElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E5M2) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16)
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
!wqElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_hf8_output_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf8E4M3FN>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_hf8_output_f16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2x!wqElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E4M3FN) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16)
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
!wqElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_hf8_output_hf8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf8E4M3FN>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf8E4M3FN>
  }

  func.func @IDU_CONV_input_hf8_output_hf8() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2x!wqElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E4M3FN) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16)
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @IDU_CONV_autopad inputsInfo : {
    DataInfo "input_0" : tensor<1x3x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x3x16x16xf16>
  }

  func.func @IDU_CONV_autopad() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<3x3x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {start = [0, 0, 0], end = [15, 15, 2], inStart = [0, 0, 0], inEnd = [15, 15, 2], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(3)
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(16)
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16)
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_ADD_input_f16_weights_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x256x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x16x16xf16>
  }

  func.func @IDU_ELTWISE_ADD_input_f16_weights_f16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <262144> : memref<1x256x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x256x16x16xf16, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) {start = [0, 0, 0], end = [15, 15, 255], inStart = [0, 0, 0], inEnd = [15, 15, 255], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(256) weight_size(65536){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_MULT_input_bf16_weights_bf16 inputsInfo : {
    DataInfo "input_0" : tensor<1x256x16x16xbf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x16x16xbf16>
  }

  func.func @IDU_ELTWISE_MULT_input_bf16_weights_bf16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <262144> : memref<1x256x16x16xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, eltwise_type = #VPU.eltwise_type<MULTIPLY>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x256x16x16xbf16, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(NO_PLT){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64){{$}}
    // CHECK-NEXT:     VPUIPDPU.IDUEltWiseMode eltwise_type(MULT){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) {start = [0, 0, 0], end = [15, 15, 255], inStart = [0, 0, 0], inEnd = [15, 15, 255], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(256) weight_size(65536){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_REDUCEMEAN_input_f16_output_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x32x32xf16>
  }

  func.func @IDU_REDUCEMEAN_input_f16_output_f16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 0 : ui64, cm_sp_pattern = 0 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCEMEAN>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(15360){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCEMEAN){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 {start = [0, 0, 0], end = [31, 31, 15], inStart = [0, 0, 0], inEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCEMEAN>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(32) size_z(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_REDUCESUMSQUARE_input_bf16_output_bf16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xbf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x32x32xbf16>
  }

  func.func @IDU_REDUCESUMSQUARE_input_bf16_output_bf16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x32x32xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 0 : ui64, cm_sp_pattern = 0 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x16x32x32xbf16, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(NO_PLT) pool_wt_data(16256){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUMSQUARE){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 {start = [0, 0, 0], end = [31, 31, 15], inStart = [0, 0, 0], inEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(32) size_z(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_REDUCESUM_input_f16_output_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x32x32xf16>
  }

  func.func @IDU_REDUCESUM_input_f16_output_f16() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 0 : ui64, cm_sp_pattern = 0 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUM>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(15360){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUM){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 {start = [0, 0, 0], end = [31, 31, 15], inStart = [0, 0, 0], inEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUM>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(32) size_z(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}
