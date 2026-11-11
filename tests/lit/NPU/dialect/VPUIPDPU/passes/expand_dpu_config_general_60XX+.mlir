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
  net.NetworkInfo entryPoint : @dyn_primitive_id_var_0 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }

  func.func @dyn_primitive_id_var_0() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias = 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], variant_primitive_id = [4, 6]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.DynPrimitiveIdVar dyn_primitive_id_var(4) dyn_primitive_id2_var(6)
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @dyn_primitive_id_var_1 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }

  func.func @dyn_primitive_id_var_1() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], variant_primitive_id = [6]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.DynPrimitiveIdVar dyn_primitive_id_var(6)
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @dyn_scale_cfg_inv_0 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }

  func.func @dyn_scale_cfg_inv_0() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, dynamic_scale_config = #VPUIP.DynamicScaleConfigAttr<dynamicSkipMode = <SPATIAL_AND_INPUT_CHANNELS>, dynamicAdjustMode = <INPUT_CHANNELS>>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}

              // CHECK:       VPUIPDPU.DPUInvariant
              // CHECK:       VPUIPDPU.DynScaleCfgInv dynamic_skip_mode(<SPATIAL_AND_INPUT_CHANNELS>) dynamic_adjust_mode(<INPUT_CHANNELS>)
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @dyn_scale_cfg_inv_1 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }

  func.func @dyn_scale_cfg_inv_1() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, dynamic_scale_config = #VPUIP.DynamicScaleConfigAttr<dynamicSkipMode = <SPATIAL_AND_INPUT_CHANNELS>, dynamicAdjustMode = <INPUT_CHANNELS>, ppeOverwriteData = 0x7777, invariantPrimitiveId = [4, 6]>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}

              // CHECK:       VPUIPDPU.DPUInvariant
              // CHECK:       VPUIPDPU.DynScaleCfgInv dyn_primitive_id(4) dyn_primitive_id2(6) dyn_ppe_data(30583) dynamic_skip_mode(<SPATIAL_AND_INPUT_CHANNELS>) dynamic_adjust_mode(<INPUT_CHANNELS>)
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>
module {
  net.NetworkInfo entryPoint : @dyn_scale_cfg_inv_3 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xi16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xi16>
  }

  func.func @dyn_scale_cfg_inv_3() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, dynamic_scale_config = #VPUIP.DynamicScaleConfigAttr<dynamicSkipMode = <SPATIAL_ONLY>, dynamicAdjustMode = <DISABLED>, invariantPrimitiveId = [4]>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64, scale = 1.000000e+00 : f64, bias= 0.000000e+00 : f64>}

              // CHECK:       VPUIPDPU.DPUInvariant
              // CHECK:       VPUIPDPU.DynScaleCfgInv dyn_primitive_id(4) dynamic_skip_mode(<SPATIAL_ONLY>) dynamic_adjust_mode(<DISABLED>)
        }
      }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) <{end = [7, 7, 63], inEnd = [15, 15, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }
    }
    return
  }
}
