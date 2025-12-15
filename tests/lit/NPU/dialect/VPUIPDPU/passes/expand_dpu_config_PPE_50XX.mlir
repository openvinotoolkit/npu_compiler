//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --expand-dpu-config="npu5-ppe-backwards-compatibility-mode=DISABLED" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282306e+38 : f64, clamp_high = 3.40282306e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_CONV_LRELU inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV_LRELU() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [-0.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(-0.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_AVEPOOL_LRELUX inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_AVEPOOL_LRELUX() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = -3.000000e+01 : f64, clamp_high = 2.000000e+01 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(1.000000e+01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.000000e+01) clamp_high(2.000000e+01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_CONV_LPRELU inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV_LPRELU() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.250000e-01], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.250000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_BFP16_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xbf16>
  }

  func.func @PPE_FP16_BFP16_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(BF16) bf16_round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_MAXPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_MAXPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_ELTWISE inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_ELTWISE() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, scale = 2.500000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 2.500000e+01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(2.500000e+01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(2.500000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    PUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_AVEPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_AVEPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(1.000000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_U8_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_FP16_U8_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_U8_FP16_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_U8_FP16_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282347E+38 : f64, clamp_high = 3.40282306e+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282306E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_U8_U8_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282306e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_U8_U8_ELTWISE_ADD inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_ELTWISE_ADD() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <ADD>, clamp_low = -3.40282306e+38 : f64, clamp_high = 3.40282306e+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_U8_U8_ELTWISE_ADD_CHANNEL_QDQ inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_ELTWISE_ADD_CHANNEL_QDQ() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <ADD>, clamp_low = -3.40282306e+38 : f64, clamp_high = 3.40282306e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282306E+38) clamp_high(3.40282306E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @PPE_U8_U8_AVEPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_AVEPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282347e+38 : f64, clamp_high = 3.40282347e+38 : f64, scale = 5.000000e+01 : f64, prelu_alpha = [1.000000e+00], bias = 5.000000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(5.000000e-01)
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(5.000000e+01)
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @PPE_U8_U8_AVEPOOL_CHANNEL_QDQ inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_AVEPOOL_CHANNEL_QDQ() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282347e+38 : f64, clamp_high = 3.40282347e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @PPE_U8_U8_MAXPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_MAXPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282347e+38 : f64, clamp_high = 3.40282347e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_U8_FP16_MAXPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_U8_FP16_MAXPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282347e+38 : f64, clamp_high = 3.40282347e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @PPE_U8_FP16_MAXPOOL_CHANNEL_QDQ inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_U8_FP16_MAXPOOL_CHANNEL_QDQ() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.40282347e+38 : f64, clamp_high = 3.40282347e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_CONV_sqrLUT inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV_sqrLUT() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_SprLookupTable !VPUASM.Buffer< "CMX_NN"[0] <74768> : memref<1x1x1x256xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) spr_lookup_table(@buffer.CMX_NN.0::@DeclareBuffer_SprLookupTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402820e+38 : f64, clamp_high = 3.402820e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.402820e+38) clamp_high(3.402820e+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) {end = [7, 7, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], spr_lut_read}
      }
    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK:       VPUIPDPU.PPEsprLUTRead

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_REDUCEMEAN inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
  }

  func.func @PPE_FP16_FP16_REDUCEMEAN() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<REDUCEMEAN>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402820e+38 : f64, clamp_high = 3.402820e+38 : f64, scale = 1.562500e-02 : f64, prelu_alpha = [1.000000e+00], bias = 5.000000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(5.000000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.562500e-02){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.402820e+38) clamp_high(3.402820e+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_REDUCESUMSQUARE inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
  }

  func.func @PPE_FP16_FP16_REDUCESUMSQUARE() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUMSQUARE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402820e+38 : f64, clamp_high = 3.402820e+38 : f64, prelu_alpha = [1.000000e+00], bias = 4.000000e-01 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(4.000000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.562500e-02){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.402820e+38) clamp_high(3.402820e+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @PPE_FP16_FP16_REDUCESUM inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
  }

  func.func @PPE_FP16_FP16_REDUCESUM() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<REDUCESUM>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402820e+38 : f64, clamp_high = 3.402820e+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScaleMult scale_static(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpClamp clamp_low(-3.402820e+38) clamp_high(3.402820e+38){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}
