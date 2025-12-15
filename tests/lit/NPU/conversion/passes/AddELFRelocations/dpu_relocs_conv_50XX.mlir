//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUIPDPU-to-NPUReg50XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DPURelocWeightTableReuseTest
module @DPURelocWeightTableReuseTest {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input_0" : tensor<1x16x16x16xf16>
        DataInfo "input_1" : tensor<16x1x1x1xi64>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x64x64xf16>
    }
    func.func @main() {
        ELF.Main @ELFMain {
          ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
              VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_22 idx(!VPURegMapped.Index<0:0:22>) <DPUInvariant>
          }
          ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
              VPUASM.DeclareBuffer @DeclareBuffer117 !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<160x1280x1x1xf16,
                  {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]> :  swizzling(5)>
              VPUASM.DeclareBuffer @DeclareBuffer125 !VPUASM.Buffer< "CMX_NN"[0] <1376256> : memref<160x1x1x4xsi32,
                  {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]> :  swizzling(5)>
              VPUASM.DeclareBuffer @DeclareBuffer396 !VPUASM.Buffer< "CMX_NN"[0] <868352> : memref<1x1280x4x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
              VPUASM.DeclareBuffer @DeclareBuffer444 !VPUASM.Buffer< "CMX_NN"[0] <878592> : memref<1x640x4x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
          }
          ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
              VPUIPDPU.DPUInvariant @DPUInvariant_0_2
                  {input = @buffer.CMX_NN.0::@DeclareBuffer396, is_zero_offset_weights_table, nce_task_type = #VPUIP.nce_task_type<CONV>,
                  output = @buffer.CMX_NN.0::@DeclareBuffer444, task_index = !VPURegMapped.Index<0:0:2>,
                  task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_22,
                  weight_table = @buffer.CMX_NN.0::@DeclareBuffer125,
                  weights = @buffer.CMX_NN.0::@DeclareBuffer117}
              DPUCfg :
              {^bb0(%arg0: memref<1x1280x4x1xf16, #NHWC, [@CMX_NN, 0]>,
                  %arg1: memref<160x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>,
                  %arg2: memref<160x1280x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>,
                  %arg3: memref<1x640x4x1xf16, #NHWC, [@CMX_NN, 0]>):
              VPUIPDPU.IDUCfg {
                  VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x1280x4x1xf16, #NHWC, [@CMX_NN, 0]>)
                  VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
                  VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
                  VPUIPDPU.IDUStride stride_x(1) stride_y(1)
                  VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
              }
              VPUIPDPU.PPECfg {
                  VPUIPDPU.PPEFpBiasAdd %arg1 : memref<160x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
                  VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<160x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
                  VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
                  VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
                  VPUIPDPU.PPEIntBiasAdd bias_static(0)
                  VPUIPDPU.PPEIntScaleMult scale_static(1)
                  VPUIPDPU.PPEIntScaleShift shift_static(0)
                  VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                  VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                  VPUIPDPU.PPEIntRound round_mode(RNE)
                  VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                  VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
                  VPUIPDPU.PPEIntConvert convert_mode(NONE)
              }
              VPUIPDPU.ODUCfg {
                  VPUIPDPU.ODUOutTensorSize dim_x(1) dim_y(4) dim_z(640)
                  VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
                  VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x640x4x1xf16, #NHWC, [@CMX_NN, 0]>)
              }
              VPUIPDPU.BarrierCfg waits([13 : ui8]) updates([14 : ui8]) start_after(0) clean_after(0)
              VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:2>) variantCount(1)
              }
          }
          ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>)  size(82944) value(1075854336)
            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(1474560) value(1075937280)
            ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)
          }
        }
      return
    }
}

// CHECK:       VPUASM.DeclareBuffer @DeclareBuffer117 !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<160x1280x1x1xf16

// CHECK:     ELF.CreateSection @task.dpu.invariant.0.0
// CHECK:       act_offset0 = UINT 0xEC000,
// CHECK:       act_offset1 = UINT 0xEC000,
// CHECK:       act_offset2 = UINT 0xEC000,
// CHECK:       act_offset3 = UINT 0xEC000,
// CHECK:       wt_offset = UINT 0x20000,
// CHECK:       odu_ac_base {
// CHECK:         UINT ac_base = 0xEE80,

// CHECK-NOT:       ELF.CreateRelocationSection @rela.task.dpu.invariant.0.0.symtab

// Weights Relocs (relocation generated because nce_task_type has is_zero_offset_weights_table set and WT_OFFSET points to the weights buffer)
//      wt_offset:
// CHECK-NOT:           ELF.Reloc offset({{[0-9]+}}) sourceSym(@symtab::@elfsym.buffer.CMX_NN.0)
