//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --create-elf-symbol-table %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

net.NetworkInfo entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}

func.func @oneDma() {
  ELF.Main {
    ELF.CreateSection @dsec1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    }
    ELF.CreateSection @dsec2 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    }
    ELF.CreateSection @dsec3 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    }
    ELF.CreateLogicalSection @lsec1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
    }
    ELF.CreateLogicalSection @lsec2 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
    }
    ELF.CreateLogicalSection @lsec3 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
    }
  }
  return
}

//CHECK: ELF.Main

//CHECK: ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
//CHECK-NEXT: ELF.Symbol @elfsym.dsec1 of(@dsec1) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.dsec2 of(@dsec2) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.dsec3 of(@dsec3) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec1 of(@lsec1) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec2 of(@lsec2) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec3 of(@lsec3) type(<STT_SECTION>)

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Test that IO symbol table sections for an outlined dynamic kernel use the
// declared upper bounds from net.NetworkInfo when computing symbol sizes:
//   - static inputs/outputs use getTotalAllocSize()
//   - dynamic inputs/outputs substitute upper bounds before computing size

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "in_0" : tensor<2xf32>
  DataInfo "in_1" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
  DataInfo "vpux_ie_shape_in_1" : tensor<4xsi32>
} outputsInfo : {
  DataInfo "out_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 400, 400]> : tensor<4xsi64>, order = #NCHW}>
  DataInfo "vpux_ie_shape_out_0" : tensor<4xsi32>
}

VPUASM.InputBindings inputDeclarations : {
  VPUASM.DeclareBuffer @in_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<2xf32, strided<[?], offset: ?>> :  swizzling(0)>
  VPUASM.DeclareBuffer @in_1_buffDecl !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> :  swizzling(0)>
  VPUASM.DeclareBuffer @vpux_ie_shape_in_1_buffDecl !VPUASM.Buffer< "NetworkInput"[2] <0> : memref<4xsi32> :  swizzling(0)>
}
VPUASM.OutputBindings outputDeclarations : {
  VPUASM.DeclareBuffer @out_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> :  swizzling(0)>
  VPUASM.DeclareBuffer @vpux_ie_shape_out_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<4xsi32> :  swizzling(0)>
}
VPUASM.ProfilingBindings profilingDeclarations : {
}

func.func @main() {
  ELF.Main {
    ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<2xf32, strided<[?], offset: ?>> :  swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkInput.1 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_1 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> :  swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkInput.2 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_2 !VPUASM.Buffer< "NetworkInput"[2] <0> : memref<4xsi32> :  swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkOutput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT") secLocation(<NetworkOutput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> :  swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkOutput.1 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT") secLocation(<NetworkOutput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_4 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<4xsi32> :  swizzling(0)>
    }
  }
  return
}

//CHECK: ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
//CHECK-NEXT: ELF.Symbol @elfsym.io.NetworkInput.0 of(@io.NetworkInput.0) type(<STT_SECTION>) size(8)
//CHECK-NEXT: ELF.Symbol @elfsym.io.NetworkInput.1 of(@io.NetworkInput.1) type(<STT_SECTION>) size(15000)
//CHECK-NEXT: ELF.Symbol @elfsym.io.NetworkInput.2 of(@io.NetworkInput.2) type(<STT_SECTION>) size(16)
//CHECK-NEXT: }
//CHECK: ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") {
//CHECK-NEXT: ELF.Symbol @elfsym.io.NetworkOutput.0 of(@io.NetworkOutput.0) type(<STT_SECTION>) size(960000)
//CHECK-NEXT: ELF.Symbol @elfsym.io.NetworkOutput.1 of(@io.NetworkOutput.1) type(<STT_SECTION>) size(16)
//CHECK-NEXT: }
