//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_SUB_input_fp16_sparse_se_table_weights_fp16_sparse inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_ELTWISE_SUB_input_fp16_sparse_se_table_weights_fp16_sparse() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <57360> : memref<1x64x16x16xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <59408> : memref<1x1x16x16xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <50192> : memref<64x1x1x896xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActSparsityMap)
        input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        weights_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_WeightsSparsityMap) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>,
        start_after = 0 : ui64, variant_count = 1 : ui64, eltwise_type = #VPU.eltwise_type<SUBTRACT>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)  <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(SUBTRACT)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>

        VPUASM.DPUVariant @DPUVariant_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_1) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(16384){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEOnly{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPerOutputChannelScaling <{tensor2_act_sparse}>{{$}}

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(16384){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEOnly{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPerOutputChannelScaling <{tensor2_act_sparse}>{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_legacy_wt_Weights inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_legacy_wt_Weights() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
         nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_legacy_wt_WeightsAndZeroPoints inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_legacy_wt_WeightsAndZeroPoints() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64,
        first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ACCU_STORE_FP32 inputsInfo : {
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf32>
  }

  func.func @IDU_ACCU_STORE_FP32() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64,
        first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
        kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_STORE_FP32)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ACCU_STORE_I32 inputsInfo : {
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xsi32>
  }

  func.func @IDU_ACCU_STORE_I32() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
        last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_STORE_I32)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ACCU_LOAD_FP32 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf32>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf32>
  }

  func.func @IDU_ACCU_LOAD_FP32() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, is_continued,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
         last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         output_type_continued = !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x32x16x16xf32, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f32) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_LOAD_FP32)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{end = [7, 7, 31], inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(32){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(32) weight_size(32){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUConvContinue{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ACCU_LOAD_I32 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xsi32>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xsi32>
  }

  func.func @IDU_ACCU_LOAD_I32() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64,
        first_variant_index = 0 : ui32, is_continued, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
         nce_task_type = #VPUIP.nce_task_type<CONV>,
         output_type_continued = !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x32x16x16xsi32, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si32) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ACCU_LOAD_I32)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{end = [7, 7, 31], inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(32){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(32) weight_size(32){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:   VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUConvContinue{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_I16_weights_I8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
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
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
         output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
         <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
         kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)  <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
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
!qElemType = !quant.uniform<u16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_U16_weights_U8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_CONV_input_U16_weights_U8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <197632> : memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
         output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
          <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}>
           PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)  <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
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
  net.NetworkInfo entryPoint : @IDU_CONV_input_fp16_se_table_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_fp16_se_table_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <197632> : memref<1x16x64x64xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
        kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}>
        PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)  <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <57360> : memref<1x64x16x16xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_InputSETable !VPUASM.Buffer< "CMX_NN"[0] <59408> : memref<1x1x16x16xi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <50192> : memref<64x1x1x896xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActSparsityMap)
        input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        weights_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_WeightsSparsityMap) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>,
        start_after = 0 : ui64, variant_count = 1 : ui64, eltwise_type = #VPU.eltwise_type<SUBTRACT>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)  <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(SUBTRACT)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>

        VPUASM.DPUVariant @DPUVariant_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_1) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(16384){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEOnly{{$}}

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(16384){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NOT:  VPUIPDPU.IDUSEOnly{{$}}
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}>
         PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E5M2) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}>
         PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E4M3FN) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}>
         PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E4M3FN) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
  net.NetworkInfo entryPoint : @IDU_ELTWISE_ADD_input_f16_weights_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x256x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x16x16xf16>
  }

  func.func @IDU_ELTWISE_ADD_input_f16_weights_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <262144> : memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}>
        PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias= 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{start = [0, 0, 0], end = [15, 15, 255], inStart = [0, 0, 0], inEnd = [15, 15, 255], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <262144> : memref<1x256x16x16xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
         weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
         <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>,
         start_after = 0 : ui64, variant_count = 1 : ui64, eltwise_type = #VPU.eltwise_type<MULTIPLY>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias= 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x256x16x16xbf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(MULT)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{start = [0, 0, 0], end = [15, 15, 255], inStart = [0, 0, 0], inEnd = [15, 15, 255], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 0 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCEMEAN>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(15360)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCEMEAN)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{start = [0, 0, 0], end = [31, 31, 15], inStart = [0, 0, 0], inEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCEMEAN>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 0 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
         nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(NO_PLT) pool_wt_data(16256)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUMSQUARE)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{start = [0, 0, 0], end = [31, 31, 15], inStart = [0, 0, 0], inEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 0 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUM>,
          start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(15360)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(REDUCESUM)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 <{start = [0, 0, 0], end = [31, 31, 15], inStart = [0, 0, 0], inEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUM>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
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
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_NoWtReads inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_NoWtReads() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
        kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
!wqElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_bf8_output_bf8_wt_split_WeightsOnly inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf8E5M2>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf8E5M2>
  }

  func.func @IDU_CONV_input_bf8_output_bf8_wt_split_WeightsOnly() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <81920> : memref<64x64x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <98304> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WtScale !VPUASM.Buffer< "CMX_NN"[0] <98560> : memref<64x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WtBias !VPUASM.Buffer< "CMX_NN"[0] <98816> : memref<64x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WtDataPtr)
        weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WtScale) weight_table_bias(@buffer.CMX_NN.0::@DeclareBuffer_WtBias) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], mpe_engine = #VPU.MPEEngine60XX<mode = <NONE>, weight_zp = 0 : i64, activation_zp = 0 : i64>, last_variant_index = 0 : ui32,
        mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f8E5M2) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(WEIGHTS_ONLY) scale_dtype(FP32)
    // CHECK-NEXT:  }
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints)
         output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
         kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32,
         mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(ZERO_POINTS_ONLY)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u4:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly_4B inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly_4B() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32,
         mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui4) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(ZERO_POINTS_ONLY)
    // CHECK-NEXT:    VPUIPDPU.IDUSubtractMpeWtBias
    // CHECK-NEXT:  }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.WeightsTableZpBitwidth zp_bitwidth(ZP_4b)
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u4:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly_4B inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly_4B() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
        kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui4) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(ZERO_POINTS_ONLY)
    // CHECK-NEXT:    VPUIPDPU.IDUSubtractMpeWtBias
    // CHECK-NEXT:  }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.WeightsTableZpBitwidth zp_bitwidth(ZP_4b)
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_subtrat_mpe_wt_bias inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_subtrat_mpe_wt_bias() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32,
         mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si4) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(ZERO_POINTS_ONLY)
    // CHECK-NEXT:    VPUIPDPU.IDUSubtractMpeWtBias
    // CHECK-NEXT:  }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.WeightsTableZpBitwidth zp_bitwidth(ZP_4b)
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!wqElemType = !quant.uniform<!quant.quantile<ui4:si8, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}>:f16:2, {1.000000e-01:0,2.000000e-01:0,3.000000e-01}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_subtrat_mpe_wt_bias_plt inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_subtrat_mpe_wt_bias_plt() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x16x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
         waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32,
         mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(FOUR_BIT_PLT) quantiles_lut([-8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00])
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:    VPUIPDPU.IDUSubtractMpeWtBias
    // CHECK-NEXT:  }

    // CHECK:    VPUIPDPU.DPUVariant
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_WeightsAndZps inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_WeightsAndZps() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WtDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<64x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WtDataPtr)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32,
        mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(WEIGHTS_AND_ZPS)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>:f16, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16_FourbitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16_FourbitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(FOUR_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 0.69999999999999996, 1.000000e+00])
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:    VPUIPDPU.IDUSubtractMpeWtBias
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense
    // CHECK-NEXT:  VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01:10>
!wqElemType = !quant.uniform<!quant.quantile<ui4:si8, {-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}>:f32, 1.600000e+01:16>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_i8_weights_i8_FourBitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi8>
  }

  func.func @IDU_CONV_input_i8_weights_i8_FourBitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <84992> : memref<64x16x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
         pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
         <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<VECTOR>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(FOUR_BIT_PLT) quantiles_lut([-8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00])
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(NO_WT_READS)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<VECTOR>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUNthwNtk
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense
    // CHECK-NEXT:  VPUIPDPU.IDUpalletLUTRead
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui4:bf16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>:f16, 0.02211761474609375>
module {
  net.NetworkInfo entryPoint : @IDU_DWCONV_input_bf16_weights_bf16_FourBitPallet_output_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xbf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x30x29xf16>
  }

  func.func @IDU_DWCONV_input_bf16_weights_bf16_FourBitPallet_output_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <27840> : memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <60608> : memref<16x1x1x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTableDataPtr !VPUASM.Buffer< "CMX_NN"[0] <61120> : memref<16x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x30x29xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <61632> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTableDataPtr)
        pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        kernel_size = [4, 4], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine60XX<mode = <NONE>, weight_zp = 0 : i64>, last_variant_index = 0 : ui32,
        mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<DWCONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(FOUR_BIT_PLT) quantiles_lut([-1.000000e+00, -8.000000e-01, -0.69999999999999996, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, 0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 0.69999999999999996, 1.000000e+00])
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(4) kernel_y(4)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(DWCONV)
    // CHECK-NEXT:    VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(WEIGHTS_ONLY)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [28, 29, 15], inEnd = [31, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<DWCONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(32) size_z(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui1:f16, {2.000000e+00,0.000000e+00}>:f32, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16_OnebitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16_OnebitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
         pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
         <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:         VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(ONE_BIT_PLT) quantiles_lut([2.000000e+00, 0.000000e+00])

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK:       VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui2:f16, {2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00}>:f32, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16_TwobitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16_TwobitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:         VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(TWO_BIT_PLT) quantiles_lut([2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00])

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK:       VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui3:f16, {2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00}>:f32, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16_ThreebitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16_ThreebitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:         VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(THREE_BIT_PLT) quantiles_lut([2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00])

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK:       VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui5:f16, {2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00}>:f32, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16_FivebitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16_FivebitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:         VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(FIVE_BIT_PLT) quantiles_lut([2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00])

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK:       VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui6:f16, {2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00, 2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00,2.000000e+00,0.000000e+00}>:f32, 1.000000e+00>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16_SixbitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16_SixbitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_palletLUT !VPUASM.Buffer< "CMX_NN"[0] <166912> : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        pallet_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_palletLUT) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)  waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:         VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(SIX_BIT_PLT) quantiles_lut([2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00])

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK:       VPUIPDPU.IDUpalletLUTRead

    }
    return
  }
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_no_partial_IC_CONV_input_f16_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_no_partial_IC_CONV_input_f16_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64,
        first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:       VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK:       VPUIPDPU.IDUWorkloadCfg workload_type(CONV)

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK:       VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NOT:   VPUIPDPU.IDUGroupConvOp
    // CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
    }
    return
  }
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_partial_32_IC_CONV_first_half_input_f16_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_partial_32_IC_CONV_first_half_input_f16_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32,
        mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:       VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK:       VPUIPDPU.IDUWorkloadCfg workload_type(CONV)

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(32){{$}}
    // CHECK:       VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(128){{$}}
    // CHECK:       VPUIPDPU.IDUGroupConvOp
    // CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_partial_16_IC_CONV_offset_group_input_f16_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_partial_16_IC_CONV_offset_group_input_f16_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32,
         mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK:       VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK:       VPUIPDPU.IDUWorkloadCfg workload_type(CONV)

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 31], inStart = [0, 0, 16], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(16) size_x(16) size_y(16) size_z(16)
    // CHECK:       VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(64){{$}}
    // CHECK:       VPUIPDPU.IDUGroupConvOp
    // CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_wzp8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_wt_offset_wzp8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 16], pallet_lut_read, weight_table_offset = 256 : ui64}>
      }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64, sp_ptr_offset = 16 : i64}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32:1, {1.0:1,1.0:255,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_spptr inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_spptr() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableSpPtr !VPUASM.Buffer< "CMX_NN"[0] <45184> : memref<32x1x1x1xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_sp_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableSpPtr) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_sp_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableSpPtr) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read, weight_table_offset = 256 : ui64}>
      }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64, sp_ptr_offset = 8 : i64}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_bias inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_bias() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableBias !VPUASM.Buffer< "CMX_NN"[0] <45184> : memref<32x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_bias(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableBias) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_bias(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableBias) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 16], pallet_lut_read, weight_table_offset = 256 : ui64}>
      }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.IDUWeightTableOffsets {bias_offset = 2 : i64, data_ptr_zp_offset = 8 : i64}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_scale_f32 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_scale_f32() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableScale !VPUASM.Buffer< "CMX_NN"[0] <45184> : memref<32x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableScale) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableScale) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 16], pallet_lut_read, weight_table_offset = 256 : ui64}>
      }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64, scale_offset = 2 : i64}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_scale_bf8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_scale_bf8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableScale !VPUASM.Buffer< "CMX_NN"[0] <45184> : memref<32x1x1x1xf8E5M2, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableScale) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableScale) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 32], pallet_lut_read, weight_table_offset = 256 : ui64}>
      }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64, scale_offset = 1 : i64}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_scale_hf8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_scale_hf8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableDataPtr !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightTableScale !VPUASM.Buffer< "CMX_NN"[0] <45184> : memref<32x1x1x1xf8E4M3FN, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableScale) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table_data_ptr(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableDataPtr) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WeightTableScale) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 32], pallet_lut_read, weight_table_offset = 256 : ui64}>
      }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64, scale_offset = 1 : i64}
    }
    return
  }
}

// -----

// Verify that u2 (2-bit unsigned) weights produce wmode(ui2) and ZP_4b bitwidth,
// consistent with u4 handling (both 2-bit and 4-bit ZP are packed into ZP_4b slots).
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u2:f32:1, {1.0:1,1.0:3,1.0:0}>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly_2B inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_CONV_input_f16_output_f16_wt_split_ZeroPointsOnly_2B() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <45056> : memref<32x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
        kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], pallet_lut_read}>
      }

    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui2) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseMode eltwise_type(ADD)
    // CHECK-NEXT:    VPUIPDPU.WeightTableSplit wt_split_mode(ZERO_POINTS_ONLY)
    // CHECK-NEXT:  }

    // CHECK:    VPUIPDPU.DPUVariant
    // CHECK:    VPUIPDPU.WeightsTableZpBitwidth zp_bitwidth(ZP_4b)
    }
    return
  }
}
