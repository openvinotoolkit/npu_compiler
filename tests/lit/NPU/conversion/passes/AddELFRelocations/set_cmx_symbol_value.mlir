//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --set-cmx-symbol-value %s | FileCheck %s --check-prefixes=CHECK,CHECK-%platform%
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU5020

module @setCMXSymbols {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() attributes {inliner_dispatch = #VPUIP.VPUIPInlinerDispatch} {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
      }
    }
    return
  }
}

//CHECK:               ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE")
//CHECK-NEXT:            ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(82944) value(1075854336)
//CHECK-NEXT:            ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>)
//CHECK-NPU4000-SAME:      size(1571840)
//CHECK-NPU5010-SAME:      size(1571840)
//CHECK-NPU5020-SAME:      size(2096128)
//CHECK-SAME:              value(1075838976)
