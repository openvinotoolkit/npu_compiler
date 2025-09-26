//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @act_shave inputsInfo : {
DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x2x3x4xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func private @act_shave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}

  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}

  VPUMI40XX.OpRanges
}
}

//CHECK: ELF.CreateLogicalSection @[[BuffersSection:.+]] aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE")
//CHECK:   VPUASM.DeclareBuffer @[[DeclareBuffer0:.+]] !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
//CHECK:   VPUASM.DeclareBuffer @[[DeclareBuffer1:.+]] !VPUASM.Buffer< "CMX_NN"[0] <2000> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>

//CHECK: ELF.CreateSection @[[TextSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.DeclareKernelText @[[DeclareKernelText:.+]] : [[KernelName:.+]]

//CHECK: VPUASM.DeclareKernelEntry @[[DeclareKernelEntry:.+]] : [[KernelName]]

//CHECK: ELF.CreateSection @[[DataSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags("SHF_WRITE|SHF_ALLOC")
//CHECK:   VPUASM.DeclareKernelData @[[DeclareKernelArgs:.+]] : [[KernelName]]

//CHECK: ELF.CreateSection @[[ParamsSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.KernelParams @[[KernelParams:.+]] inputs([@[[BuffersSection]]::@[[DeclareBuffer0]]]) outputs([@[[BuffersSection]]::@[[DeclareBuffer1]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type([[KernelName]]) < {kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]}>

//CHECK: ELF.CreateLogicalSection @[[MetadataSection:.+]] aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK:   VPUASM.DeclareTaskBuffer @[[RangeTaskLocation:.+]] idx(!VPURegMapped.Index<[[RTLI:.+]]>) <ActKernelRange>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation:.+]] idx(!VPURegMapped.Index<[[ITLI:.+]]>) <ActKernelInvocation>

//CHECK: ELF.CreateSection @[[RangeSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelRange
//CHECK-DAG: @[[RangeSymbol:.+]] idx(!VPURegMapped.Index<[[RTLI]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[RangeTaskLocation]])
//CHECK-DAG: kernelTaskType(@COMPUTE)
//CHECK-DAG: calls @[[TextSection]]::@[[DeclareKernelText]]
//CHECK-DAG: @[[DeclareKernelEntry]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[InvoSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol:.+]] idx(!VPURegMapped.Index<[[ITLI]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.MappedInference
//CHECK-DAG: actKernelRanges([@[[RangeSection]]::@[[RangeSymbol]]])
//CHECK-DAG: actKernelInvocations([@[[InvoSection]]::@[[InvoSymbol]]])
//CHECK-DAG: actKernelRangesCount([1, 0, 0, 0, 0, 0])
//CHECK-DAG: actKernelInvocationsCount([1, 0, 0, 0, 0, 0])

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @act_shave inputsInfo : {
DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x2x3x4xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func private @act_shave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
  %itl1 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:1>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

  %i1 = VPUMI40XX.ActKernelInvocation taskLocation(%itl1 : !VPURegMapped.Index<0:0:1>) previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}

  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}

  VPUMI40XX.OpRanges
}
}

//CHECK: ELF.CreateLogicalSection @[[BuffersSection:.+]] aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE")
//CHECK:   VPUASM.DeclareBuffer @[[DeclareBuffer0:.+]] !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
//CHECK:   VPUASM.DeclareBuffer @[[DeclareBuffer1:.+]] !VPUASM.Buffer< "CMX_NN"[0] <2000> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>

//CHECK: ELF.CreateSection @[[TextSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.DeclareKernelText @[[DeclareKernelText:.+]] : [[KernelName:.+]]

//CHECK: VPUASM.DeclareKernelEntry @[[DeclareKernelEntry:.+]] : [[KernelName]]

//CHECK: ELF.CreateSection @[[DataSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags("SHF_WRITE|SHF_ALLOC")
//CHECK:   VPUASM.DeclareKernelData @[[DeclareKernelArgs:.+]] : [[KernelName]]

//CHECK: ELF.CreateSection @[[ParamsSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.KernelParams @[[KernelParams:.+]] inputs([@[[BuffersSection]]::@[[DeclareBuffer0]]]) outputs([@[[BuffersSection]]::@[[DeclareBuffer1]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type([[KernelName]]) < {kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]}>

//CHECK: ELF.CreateLogicalSection @[[MetadataSection:.+]] aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK:   VPUASM.DeclareTaskBuffer @[[RangeTaskLocation:.+]] idx(!VPURegMapped.Index<[[RTLI:.+]]>) <ActKernelRange>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation:.+]] idx(!VPURegMapped.Index<[[ITLI:.+]]>) <ActKernelInvocation>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation1:.+]] idx(!VPURegMapped.Index<[[ITLI1:.+]]>) <ActKernelInvocation>

//CHECK: ELF.CreateSection @[[RangeSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelRange
//CHECK-DAG: @[[RangeSymbol:.+]] idx(!VPURegMapped.Index<[[RTLI]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[RangeTaskLocation]])
//CHECK-DAG: kernelTaskType(@COMPUTE)
//CHECK-DAG: calls @[[TextSection]]::@[[DeclareKernelText]]
//CHECK-DAG: @[[DeclareKernelEntry]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[InvoSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol:.+]] idx(!VPURegMapped.Index<[[ITLI]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-NOT: next_link

//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol1:.+]] idx(!VPURegMapped.Index<[[ITLI1]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation1]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.MappedInference
//CHECK-DAG: actKernelRanges([@[[RangeSection]]::@[[RangeSymbol]]])
//CHECK-DAG: actKernelInvocations([@[[InvoSection]]::@[[InvoSymbol]]])
//CHECK-DAG: actKernelRangesCount([1, 0, 0, 0, 0, 0])
//CHECK-DAG: actKernelInvocationsCount([2, 0, 0, 0, 0, 0])

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @act_shave inputsInfo : {
DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x2x3x4xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func private @act_shave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %rtl1 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:0>
  %rtl2 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:1:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
  %itl1 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:1>
  %itl2 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:2>
  %itl3 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:3>
  %itl4 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:0>
  %itl5 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:1:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %r1 = VPUMI40XX.ActKernelRange taskLocation(%rtl1 : !VPURegMapped.Index<0:1:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>

  %r2 = VPUMI40XX.ActKernelRange taskLocation(%rtl2 : !VPURegMapped.Index<1:1:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

  %i1 = VPUMI40XX.ActKernelInvocation {taskLinkAttrName = #VPURegMapped.IndexType<<0:0:0>>} taskLocation(%itl1 : !VPURegMapped.Index<0:0:1>) previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

  %i2 = VPUMI40XX.ActKernelInvocation {taskLinkAttrName = #VPURegMapped.IndexType<<0:0:1>>} taskLocation(%itl2 : !VPURegMapped.Index<0:0:2>) previousTask(%i1 : !VPURegMapped.Index<0:0:1>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>

  %i3 = VPUMI40XX.ActKernelInvocation {taskLinkAttrName = #VPURegMapped.IndexType<<0:0:2>>} taskLocation(%itl3 : !VPURegMapped.Index<0:0:3>) previousTask(%i2 : !VPURegMapped.Index<0:0:2>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>

  %i4 = VPUMI40XX.ActKernelInvocation taskLocation(%itl4 : !VPURegMapped.Index<0:1:0>) range_index(%r1 : <0:1:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>

  %i5 = VPUMI40XX.ActKernelInvocation taskLocation(%itl5 : !VPURegMapped.Index<1:1:0>) range_index(%r2 : <1:1:0>) kernel_params(%7 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:0>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0, %r1), (%r2) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:1:0>)) actKernelInvocations((%i0, %i4), (%i5) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:1:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[4, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}

  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}

  VPUMI40XX.OpRanges
}
}

//CHECK: ELF.CreateLogicalSection @[[BuffersSection:.+]] aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE")
//CHECK:   VPUASM.DeclareBuffer @[[DeclareBuffer0:.+]] !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
//CHECK:   VPUASM.DeclareBuffer @[[DeclareBuffer1:.+]] !VPUASM.Buffer< "CMX_NN"[0] <2000> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>

//CHECK: ELF.CreateSection @[[TextSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.DeclareKernelText @[[DeclareKernelText:.+]] : [[KernelName:.+]]

//CHECK: VPUASM.DeclareKernelEntry @[[DeclareKernelEntry:.+]] : [[KernelName]]

//CHECK: ELF.CreateSection @[[DataSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags("SHF_WRITE|SHF_ALLOC")
//CHECK:   VPUASM.DeclareKernelData @[[DeclareKernelArgs:.+]] : [[KernelName]]

//CHECK: ELF.CreateSection @[[ParamsSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.KernelParams @[[KernelParams:.+]] inputs([@[[BuffersSection]]::@[[DeclareBuffer0]]]) outputs([@[[BuffersSection]]::@[[DeclareBuffer1]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type([[KernelName]]) < {kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]}>

//CHECK: ELF.CreateLogicalSection @[[MetadataSection:.+]] aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK:   VPUASM.DeclareTaskBuffer @[[RangeTaskLocation:.+]] idx(!VPURegMapped.Index<[[RTLI:.+]]>) <ActKernelRange>
//CHECK:   VPUASM.DeclareTaskBuffer @[[RangeTaskLocation1:.+]] idx(!VPURegMapped.Index<[[RTLI1:.+]]>) <ActKernelRange>
//CHECK:   VPUASM.DeclareTaskBuffer @[[RangeTaskLocation2:.+]] idx(!VPURegMapped.Index<[[RTLI2:.+]]>) <ActKernelRange>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation:.+]] idx(!VPURegMapped.Index<[[ITLI:.+]]>) <ActKernelInvocation>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation1:.+]] idx(!VPURegMapped.Index<[[ITLI1:.+]]>) <ActKernelInvocation>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation2:.+]] idx(!VPURegMapped.Index<[[ITLI2:.+]]>) <ActKernelInvocation>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation3:.+]] idx(!VPURegMapped.Index<[[ITLI3:.+]]>) <ActKernelInvocation>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation4:.+]] idx(!VPURegMapped.Index<[[ITLI4:.+]]>) <ActKernelInvocation>
//CHECK:   VPUASM.DeclareTaskBuffer @[[InvoTaskLocation5:.+]] idx(!VPURegMapped.Index<[[ITLI5:.+]]>) <ActKernelInvocation>

//CHECK: ELF.CreateSection @[[RangeSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelRange
//CHECK-DAG: @[[RangeSymbol:.+]] idx(!VPURegMapped.Index<[[RTLI]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[RangeTaskLocation]])
//CHECK-DAG: kernelTaskType(@COMPUTE)
//CHECK-DAG: calls @[[TextSection]]::@[[DeclareKernelText]]
//CHECK-DAG: @[[DeclareKernelEntry]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[RangeSection1:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelRange
//CHECK-DAG: @[[RangeSymbol1:.+]] idx(!VPURegMapped.Index<[[RTLI1]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[RangeTaskLocation1]])
//CHECK-DAG: kernelTaskType(@COMPUTE)
//CHECK-DAG: calls @[[TextSection]]::@[[DeclareKernelText]]
//CHECK-DAG: @[[DeclareKernelEntry]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[RangeSection2:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelRange
//CHECK-DAG: @[[RangeSymbol2:.+]] idx(!VPURegMapped.Index<[[RTLI2]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[RangeTaskLocation2]])
//CHECK-DAG: kernelTaskType(@COMPUTE)
//CHECK-DAG: calls @[[TextSection]]::@[[DeclareKernelText]]
//CHECK-DAG: @[[DeclareKernelEntry]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[InvoSection:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol:.+]] idx(!VPURegMapped.Index<[[ITLI]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-DAG: next_link = @[[MetadataSection]]::@[[InvoTaskLocation1]]

//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol1:.+]] idx(!VPURegMapped.Index<[[ITLI1]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation1]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-DAG: next_link = @[[MetadataSection]]::@[[InvoTaskLocation2]]

//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol2:.+]] idx(!VPURegMapped.Index<[[ITLI2]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation2]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-DAG: next_link = @[[MetadataSection]]::@[[InvoTaskLocation3]]

//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol3:.+]] idx(!VPURegMapped.Index<[[ITLI3]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation3]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[InvoSection1:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol4:.+]] idx(!VPURegMapped.Index<[[ITLI4]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation4]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation1]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @[[InvoSection2:.+]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.ActKernelInvocation
//CHECK-DAG: @[[InvoSymbol5:.+]] idx(!VPURegMapped.Index<[[ITLI5]]>)
//CHECK-DAG: taskLocation(@[[MetadataSection]]::@[[InvoTaskLocation5]])
//CHECK-DAG: -> @[[MetadataSection]]::@[[RangeTaskLocation2]]
//CHECK-DAG: kernel_data : @[[DataSection]]::@[[DeclareKernelArgs]]
//CHECK-DAG: kernel_params : @[[ParamsSection]]::@[[KernelParams]]
//CHECK-NOT: next_link

//CHECK: ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.MappedInference
//CHECK-DAG: actKernelRanges([@[[RangeSection]]::@[[RangeSymbol]]])
//CHECK-DAG: actKernelInvocations([@[[InvoSection]]::@[[InvoSymbol]]])
//CHECK-DAG: actKernelRangesCount([1, 0, 0, 0, 0, 0])
//CHECK-DAG: actKernelInvocationsCount([4, 0, 0, 0, 0, 0])

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @act_shave_only_list_index_1 inputsInfo : {
DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x2x3x4xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func private @act_shave_only_list_index_1() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:1:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:1:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:1:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:1:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:1:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:1:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:1:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:1:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:1:0>) range_index(%r0 : <0:1:0>) kernel_params(%7 : <0:1:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:1:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:1:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}

  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}

  VPUMI40XX.OpRanges
}
}

//CHECK: ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.MappedInference
//CHECK-DAG: actKernelRangesCount([0, 0, 0, 0, 0, 0])
//CHECK-DAG: actKernelInvocationsCount([0, 0, 0, 0, 0, 0])
//CHECK-NOT: actKernelRanges
//CHECK-NOT: actKernelInvocations
