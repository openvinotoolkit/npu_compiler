//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>,
        nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0
         weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16){{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!QuantileType.quantile<ui4:f16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>:f16, 0.02211761474609375>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_fp16FourbitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_weights_fp16FourbitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
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
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16){{$}}
    }
    return
  }
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
!wqElemType = !quant.uniform<u8:f32, 1.600000e+01:16>
module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_input_u8_weights_u8 inputsInfo : {
    DataInfo "input_0" : tensor<1x256x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x16x16xui8>
  }

  func.func @IDU_ELTWISE_input_u8_weights_u8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, input_channels_compression, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32) <{input_compressed}>
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        <{start = [0, 0, 0], end = [15, 15, 255], inStart = [0, 0, 0], inEnd = [15, 15, 255], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(256) weight_size(65536){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_8_8)
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01:10>
!wqElemType = !quant.uniform<i8:f32, 1.600000e+01:16>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_i8_weights_i8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi8>
  }

  func.func @IDU_CONV_input_i8_weights_i8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <82944> : memref<64x16x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <81920> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
        kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<VECTOR>, nce_task_type = #VPUIP.nce_task_type<CONV>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<VECTOR>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUNthwNtk
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01:10>
!wqElemType = !quant.uniform<!QuantileType.quantile<ui4:si8, {-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}>:f32, 1.600000e+01:16>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_i8_weights_i8FourBitPallet inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi8>
  }

  func.func @IDU_CONV_input_i8_weights_i8FourBitPallet() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <82944> : memref<64x16x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <81920> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
         weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16])
         updates([1 : ui16]) <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<VECTOR>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(FOUR_BIT_PLT) quantiles_lut([-8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00])
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<VECTOR>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUNthwNtk
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense
    }
  return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_weights_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x8x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x15x15xf16>
  }

  func.func @IDU_CONV_input_f16_weights_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <36992> : memref<1x8x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<64x8x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <41088> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x15x15xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 255 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, out_channel_offset = 0 : i64,
         start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x8x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(255)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{start = [0, 0, 0], end = [14, 14, 63], inStart = [0, 0, 0], inEnd = [15, 15, 7], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(8){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUActSwizzle swizzle_key
    // CHECK-NOT:  VPUIPDPU.IDUWeightSwizzle wt_swizzle_key
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_AVEPOOL_input_f16_weights_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_AVEPOOL_input_f16_weights_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(1)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <41984> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(1)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
         kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}>
         PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) pool_wt_data(15360)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(AVEPOOL)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{start = [0, 0, 0], end = [63, 63, 63], inStart = [0, 0, 0], inEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_1)
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSwizzle wt_swizzle_key(SWIZZLE_KEY_1)
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_AVEPOOL_input_bf16_weights_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xbf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xbf16>
  }

  func.func @IDU_AVEPOOL_input_bf16_weights_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <41984> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
        nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xbf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(NO_PLT) pool_wt_data(16256)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(AVEPOOL)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @IDU_AVEPOOL_input_u8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @IDU_AVEPOOL_input_u8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <4096> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
        nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, quant_mult = [12288], quant_shift = [9], quant_post_shift = 0>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT) pool_wt_data(257)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(AVEPOOL)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @IDU_AVEPOOL_input_i8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xi8>
  }

  func.func @IDU_AVEPOOL_input_i8() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <4096> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
        nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, quant_mult = [12288], quant_shift = [9], quant_post_shift = 0>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT) pool_wt_data(257)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(AVEPOOL)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_MAXPOOL_input_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_MAXPOOL_input_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
        kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
    // CHECK-NEXT:    VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_MAXPOOL_input_f16_small_kernel_opt inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @IDU_MAXPOOL_input_f16_small_kernel_opt() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, is_small_kernel_optimized,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32,
        mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
    // CHECK-NEXT:    VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!wqElemType = !quant.uniform<i8:f32, 1.600000e+01:16>
module {
  net.NetworkInfo entryPoint : @IDU_CONV_input_f16_sparse_weights_i8_sparse_output_cont inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @IDU_CONV_input_f16_sparse_weights_i8_sparse_output_cont() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x64x16x16xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <34816> : memref<64x16x2x2x!wqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <38912> : memref<64x16x2x2xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <39424> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        input_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActSparsityMap) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weights_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_WeightsSparsityMap)
        weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, is_continued,
         kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32,
         mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, output_type_continued = !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>,
         start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(2) stride_y(2)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
    // CHECK-NOT:  VPUIPDPU.IDUNthwNtk
    // CHECK:       VPUIPDPU.IDUConvContinue
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_input_f16_sparse_weights_f16_se_size inputsInfo : {
    DataInfo "input_0" : tensor<1x256x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x16x16xf16>
  }

  func.func @IDU_ELTWISE_input_f16_sparse_weights_f16_se_size() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <139264> : memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <270336> : memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        input_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActSparsityMap) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
        waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, input_se_size = 17 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>,
        start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x256x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{in_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUStorageElement se_size(17) num_ses_in_z_dir(15)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        <{start = [0, 0, 0], end = [15, 15, 255], inStart = [0, 0, 0], inEnd = [15, 15, 255], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(256) weight_size(65536){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NOT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_8_8)
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.01918038901160745:88>
!oqElemType = !quant.uniform<u8:f16, 0.027439035153856463:128>
module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_input_u8_weights_u8_ppe_quant_mult inputsInfo : {
    DataInfo "input_0" : tensor<1x256x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x16x16xui8>
  }

  func.func @IDU_ELTWISE_input_u8_weights_u8_ppe_quant_mult() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x256x16x16x!oqElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, in1_quant_mult = [29455], in2_quant_mult = [40224], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [18659], quant_post_shift = 0 : i64, quant_shift = [30]>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x256x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(ui8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(29455 : i64) elop_scale_b(40224 : i64)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32:1, {1.5118216247002567,1.9504636963259352,1.1441596127196338,1.9486494471372438,1.3118314520104855,1.4233264489725757,1.8277025938204416,1.4091991363691614,1.5495936876730596,1.0275591132430684,1.7535131086748066,1.5381433132192783,1.3297317164990923,1.7884287034284043,1.3031948292916451,1.4534978894806514}>
module {
  net.NetworkInfo entryPoint : @IDU_ELTWISE_input_i8_weights_i8_quant_per_axis inputsInfo : {
    DataInfo "input_0" : tensor<1x16x10x10xsi8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x10x10xsi8>
  }

  func.func @IDU_ELTWISE_input_i8_weights_i8_quant_per_axis() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <1600> : memref<1x16x10x10x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <3200> : memref<1x16x10x10x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x10x10x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>,
        nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, quant_scale = [0.000000e+00]>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x16x10x10x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(si8) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(ELTWISE)
    // CHECK-NEXT:    VPUIPDPU.IDUEltWiseCfg elop_scale_a(1 : i64) elop_scale_b(1 : i64)
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_DWCONV_input_bf16_weights_bf16_output_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xbf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x30x29xf16>
  }

  func.func @IDU_DWCONV_input_bf16_weights_bf16_output_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <27840> : memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <60608> : memref<16x1x1x16xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <61120> : memref<16x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x30x29xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [4, 4],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<DWCONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, quant_scale = [1.000000e+00]>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations({{%[^:]+}}: memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(bf16) wt_plt_cfg(NO_PLT)
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(4) kernel_y(4)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(DWCONV)
    // CHECK-NEXT:    VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true)
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [28, 29, 15], inEnd = [31, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<DWCONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(32) size_z(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!QuantileType.quantile<ui4:bf16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>:f16, 0.02211761474609375>
module {
  net.NetworkInfo entryPoint : @IDU_DWCONV_input_bf16_weights_bf16FourBitPallet_output_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xbf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x30x29xf16>
  }

  func.func @IDU_DWCONV_input_bf16_weights_bf16FourBitPallet_output_f16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <27840> : memref<1x16x32x32xbf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <60608> : memref<16x1x1x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <61120> : memref<16x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x30x29xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 0 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [4, 4], kernel_strides = [1, 1],
        last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<DWCONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, quant_scale = [1.000000e+00]>}
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
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0
        calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable)
        <{end = [28, 29, 15], inEnd = [31, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<DWCONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(32) size_y(32) size_z(16){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(16) weight_size(256){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    }
    return
  }
}
