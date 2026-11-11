//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @MPE_CONV_input_I16_weights_I8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }

  func.func @MPE_CONV_input_I16_weights_I8() {
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
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
        weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         start_after = 0 : ui64, variant_count = 1 : ui64, mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 0 : i64, activation_zp = 2888 : i64>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.276800e+04 : f64, clamp_high = 3.276700e+04 : f64, prelu_alpha = [1.000000e-01], adder = 2.888000e+03 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.MPECfg {
    // CHECK:         VPUIPDPU.MPEActivationBias act_bias(2888)
    // CHECK-NEXT:    VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
    // CHECK:       }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:3}>
module {
  net.NetworkInfo entryPoint : @MPE_CONV_input_F16_weights_I8_ZeroPointsOnly inputsInfo : {
    DataInfo "input_0" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x3x3xf16>
  }

  func.func @MPE_CONV_input_F16_weights_I8_ZeroPointsOnly() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<16x16x2x2x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsZeroPoints !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_zero_points(@buffer.CMX_NN.0::@DeclareBuffer_WeightsZeroPoints) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.276800e+04 : f64, clamp_high = 3.276700e+04 : f64, prelu_alpha = [1.000000e-01], scale = 2.500000e-01 : f64, adder = 2.888000e+03 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.MPECfg {
    // CHECK:         VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
    // CHECK-NOT:     VPUIPDPU.MPEWeightsBias
    // CHECK:       }
    }
    return
  }
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @MPE_CONV_input_I16_weights_I8 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @MPE_CONV_input_I16_weights_I8() {
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
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) input_storage_element_table(@buffer.CMX_NN.0::@DeclareBuffer_InputSETable) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
         weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
         <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         start_after = 0 : ui64, variant_count = 1 : ui64, mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 0 : i64, activation_zp = 2888 : i64>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.553500e+04 : f64, prelu_alpha = [1.000000e-01], adder = 2.888000e+03 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.MPECfg {
    // CHECK:         VPUIPDPU.MPEWeightsBias weights_bias(0)
    // CHECK:         VPUIPDPU.MPEActivationBias act_bias(2888)
    // CHECK-NEXT:    VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
    // CHECK:       }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @MPE_CONV_input_u8_output_u8_wt_split_NoWtReads_SCL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @MPE_CONV_input_u8_output_u8_wt_split_NoWtReads_SCL() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <4096> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <20480> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut)
         waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32,
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32,
          mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64,
          mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 10 : i64, activation_zp = 10 : i64>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.MPECfg {
    // CHECK:         VPUIPDPU.MPEActivationBias act_bias(10)
    // CHECK:         VPUIPDPU.MPEWeightsBias weights_bias(10)
    // CHECK:         VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
    // CHECK:       }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @MPE_CONV_input_u8_output_u8_wt_split_NoWtReads_DCIM inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @MPE_CONV_input_u8_output_u8_wt_split_NoWtReads_DCIM() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <4096> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <20480> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
         kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64,
         mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 10 : i64, activation_zp = 10 : i64>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.MPECfg {
    // CHECK:         VPUIPDPU.MPEActivationBias act_bias(10)
    // CHECK:         VPUIPDPU.MPEWeightsBias weights_bias(10)
    // CHECK:         VPUIPDPU.MPEEngineMode mpe_engine_mode(<DCIM>)
    // CHECK:       }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01:-118>
module {
  net.NetworkInfo entryPoint : @MPE_CONV_input_F16_weight_i8_output_F16_wt_SCL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @MPE_CONV_input_F16_weight_i8_output_F16_wt_SCL() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <4096> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <20480> : memref<64x64x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
         start_after = 0 : ui64, variant_count = 1 : ui64, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = -118 : i64>}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 2.500000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg  {
    // CHECK:         VPUIPDPU.IDUSubtractMpeWtBias
    // CHECK:       }
    // CHECK:       VPUIPDPU.MPECfg {
    // CHECK:         VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
    // CHECK:         VPUIPDPU.MPEWeightsBias weights_bias(138)
    // CHECK:       }
    }
    return
  }
}
