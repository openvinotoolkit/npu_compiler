//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @OneM2IWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x256x256x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x256x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.M2I0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_M2I_0 idx(!VPURegMapped.Index<0:0:0>) <M2I>
      }
      ELF.CreateSection @text.dma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        "NPUReg40XX.M2I"() <{input = @DeclareBuffer0, descriptor = #NPUReg40XX.VpuMediaTask<
          VpuMediaTask {
            inAddr0 {
              UINT inAddr = 0
            },
            inSize0 {
              UINT width = 0xFF,
              UINT height = 0xA9,
              UINT ls = 0x100,
              UINT pid = 0
            },
            inAddr1 {
              UINT inAddr = 0
            },
            inSize1 {
              UINT width = 0x7F,
              UINT height = 0x54,
              UINT ls = 0x80,
              UINT HWPEN = 0,
              UINT ExtHDR = 0 ,
              UINT Reserved_InSize1 = 0
            },
            inAddr2 {
              UINT inAddr = 0
            },
            inSize2 {
              UINT width = 0x7F,
              UINT height = 0x54,
              UINT ls = 0x80,
              UINT Reserved_InSize2 = 0
            },
            IOCfg {
              UINT inFormat = 1,
              UINT outFormat = 0x1A,
              UINT numRois = 1,
              UINT sampleType = 0,
              UINT operations = 0xB,
              UINT IFC = 1,
              UINT IRQMask = 0x8000
            },
            NormFactor_0 {
              UINT NormFact0 = 0x4900,
              UINT NormFact1 = 0x4980,
              UINT NormFact2 = 0x4A00,
              UINT NormFact3 = 0x4A80
            },
            NormFactor_1 {
              UINT NormFact0 = 0x4D00,
              UINT NormFact1 = 0x4D40,
              UINT NormFact2 = 0x4D80,
              UINT NormFact3 = 0x4DC0
            },
            NormFactor_2 {
              UINT NormFact0 = 0x4F80,
              UINT NormFact1 = 0x4FC0,
              UINT NormFact2 = 0x5000,
              UINT NormFact3 = 0x5020
            },
            PSOB {
              UINT inPS = 0xAA00,
              UINT outBase = 0,
              UINT HWPAddrLO = 0
            },
            nextDesc = UINT 0,
            HWPAddrHI = UINT 0,
            RoiDef {
              UINT roiBase = 0,
              UINT outFormatLocal = 0x1A,
              UINT samlingTypeLocal = 0,
              UINT OFC = 2,
              UINT IRQLocal = 0,
              UINT HWProfEN = 0,
              UINT RoiDef_RESERVED = 0
            },
            RoiCfg {
              UINT X_coord = 0,
              UINT Y_coord = 0,
              UINT roiWidth = 0xFF,
              UINT roiHeight = 0xA9
            },
            OutScaleSize {
              UINT outScale0_width = 0xFF,
              UINT outScale0_height = 0xFF,
              UINT outScale1_width = 0,
              UINT outScale1_height = 0
            },
            ScPSY {
              UINT psSc0Y = 0x30000,
              UINT psSc1Y = 0
            },
            ScPSUV {
              UINT psSc0UV = 0,
              UINT psSc1UV = 0
            },
            OutLS {
              UINT lsSc0Y = 0x300,
              UINT lsSc1Y = 0,
              UINT lsSc0UV = 0,
              UINT lsSc1UV = 0
            },
            ScOffset {
              UINT vSc_offset = 0,
              UINT hSc_offset = 0
            },
            ScFactor {
              UINT vSc_factor = 0x20000,
              UINT hSc_factor = 0x20000
            },
            barGateMaskLO = UINT 0,
            barGateMaskHI = UINT 0,
            barUpdateLO = UINT 0,
            barUpdateHI = UINT 0,
            media_barriers_sched_ {
              UINT start_after_ = 1,
              UINT clean_after_ = 2
            },
            pad8_0 = UINT 0,
            pad8_1 = UINT 0
          } requires 11:4:10
        >, output_buff = @DeclareBuffer1, sym_name = "M2I_0_0"}> : () -> ()
      }
    }
    return
  }
}

// CHECK: descriptor = #NPUReg40XX.VpuMediaTask<
// CHECK: VpuMediaTask {
// CHECK:   inAddr0 {
// CHECK:     UINT inAddr = 0
// CHECK:   },
// CHECK:   inSize0 {
// CHECK:     UINT width = 0xFF,
// CHECK:     UINT height = 0xA9,
// CHECK:     UINT ls = 0x100,
// CHECK:     UINT pid = 0
// CHECK:   },
// CHECK:   inAddr1 {
// CHECK:     UINT inAddr = 0
// CHECK:   },
// CHECK:   inSize1 {
// CHECK:     UINT width = 0x7F,
// CHECK:     UINT height = 0x54,
// CHECK:     UINT ls = 0x80,
// CHECK:     UINT HWPEN = 0,
// CHECK:     UINT ExtHDR = 0,
// CHECK:     UINT Reserved_InSize1 = 0
// CHECK:   },
// CHECK:   inAddr2 {
// CHECK:     UINT inAddr = 0
// CHECK:   },
// CHECK:   inSize2 {
// CHECK:     UINT width = 0x7F,
// CHECK:     UINT height = 0x54,
// CHECK:     UINT ls = 0x80,
// CHECK:     UINT Reserved_InSize2 = 0
// CHECK:   },
// CHECK:   IOCfg {
// CHECK:     UINT inFormat = 1,
// CHECK:     UINT outFormat = 0x1A,
// CHECK:     UINT numRois = 1,
// CHECK:     UINT sampleType = 0,
// CHECK:     UINT operations = 0xB,
// CHECK:     UINT IFC = 1,
// CHECK:     UINT IRQMask = 0x8000
// CHECK:   },
// CHECK:   NormFactor_0 {
// CHECK:     UINT NormFact0 = 0x4900,
// CHECK:     UINT NormFact1 = 0x4980,
// CHECK:     UINT NormFact2 = 0x4A00,
// CHECK:     UINT NormFact3 = 0x4A80
// CHECK:   },
// CHECK:   NormFactor_1 {
// CHECK:     UINT NormFact0 = 0x4D00,
// CHECK:     UINT NormFact1 = 0x4D40,
// CHECK:     UINT NormFact2 = 0x4D80,
// CHECK:     UINT NormFact3 = 0x4DC0
// CHECK:   },
// CHECK:   NormFactor_2 {
// CHECK:     UINT NormFact0 = 0x4F80,
// CHECK:     UINT NormFact1 = 0x4FC0,
// CHECK:     UINT NormFact2 = 0x5000,
// CHECK:     UINT NormFact3 = 0x5020
// CHECK:   },
// CHECK:   PSOB {
// CHECK:     UINT inPS = 0xAA00,
// CHECK:     UINT outBase = 0,
// CHECK:     UINT HWPAddrLO = 0
// CHECK:   },
// CHECK:   nextDesc = UINT 0,
// CHECK:   HWPAddrHI = UINT 0,
// CHECK:   RoiDef {
// CHECK:     UINT roiBase = 0,
// CHECK:     UINT outFormatLocal = 0x1A,
// CHECK:     UINT samlingTypeLocal = 0,
// CHECK:     UINT OFC = 2,
// CHECK:     UINT IRQLocal = 0,
// CHECK:     UINT HWProfEN = 0,
// CHECK:     UINT RoiDef_RESERVED = 0
// CHECK:   },
// CHECK:   RoiCfg {
// CHECK:     UINT X_coord = 0,
// CHECK:     UINT Y_coord = 0,
// CHECK:     UINT roiWidth = 0xFF,
// CHECK:     UINT roiHeight = 0xA9
// CHECK:   },
// CHECK:   OutScaleSize {
// CHECK:     UINT outScale0_width = 0xFF,
// CHECK:     UINT outScale0_height = 0xFF,
// CHECK:     UINT outScale1_width = 0,
// CHECK:     UINT outScale1_height = 0
// CHECK:   },
// CHECK:   ScPSY {
// CHECK:     UINT psSc0Y = 0x30000,
// CHECK:     UINT psSc1Y = 0
// CHECK:   },
// CHECK:   ScPSUV {
// CHECK:     UINT psSc0UV = 0,
// CHECK:     UINT psSc1UV = 0
// CHECK:   },
// CHECK:   OutLS {
// CHECK:     UINT lsSc0Y = 0x300,
// CHECK:     UINT lsSc1Y = 0,
// CHECK:     UINT lsSc0UV = 0,
// CHECK:     UINT lsSc1UV = 0
// CHECK:   },
// CHECK:   ScOffset {
// CHECK:     UINT vSc_offset = 0,
// CHECK:     UINT hSc_offset = 0
// CHECK:   },
// CHECK:   ScFactor {
// CHECK:     UINT vSc_factor = 0x20000,
// CHECK:     UINT hSc_factor = 0x20000
// CHECK:   },
// CHECK:   barGateMaskLO = UINT 0,
// CHECK:   barGateMaskHI = UINT 0,
// CHECK:   barUpdateLO = UINT 0,
// CHECK:   barUpdateHI = UINT 0,
// CHECK:   media_barriers_sched_ {
// CHECK:     UINT start_after_ = 1,
// CHECK:     UINT clean_after_ = 2
// CHECK:   },
// CHECK:   pad8_0 = UINT 0,
// CHECK:   pad8_1 = UINT 0
// CHECK: } requires 11:4:10
// CHECK: >
