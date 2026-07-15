//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @act_shave inputsInfo : {
DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x2x3x4xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func nested @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @act_shave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion

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
//CHECK:   VPUASM.KernelParams @[[KernelParams:.+]] inputs([@[[BuffersSection]]::@[[DeclareBuffer0]]]) outputs([@[[BuffersSection]]::@[[DeclareBuffer1]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type([[KernelName]]) <{
//CHECK: inputDimsBinaryVector = [232, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0],
//CHECK:  kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
//CHECK:  outputDimsBinaryVector = [232, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0]}>

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
  func.func nested @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @act_shave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
  %itl1 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:1>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

  %i1 = VPUMI40XX.ActKernelInvocation taskLocation(%itl1 : !VPURegMapped.Index<0:0:1>) previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion

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
//CHECK:   VPUASM.KernelParams @[[KernelParams:.+]] inputs([@[[BuffersSection]]::@[[DeclareBuffer0]]]) outputs([@[[BuffersSection]]::@[[DeclareBuffer1]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type([[KernelName]])
//CHECK: <{inputDimsBinaryVector = [232, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0],
//CHECK:     kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
//CHECK: outputDimsBinaryVector = [232, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0]}>

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
  func.func nested @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @act_shave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

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

  ELF.ABIVersion

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
//CHECK:   VPUASM.KernelParams @[[KernelParams:.+]] inputs([@[[BuffersSection]]::@[[DeclareBuffer0]]]) outputs([@[[BuffersSection]]::@[[DeclareBuffer1]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type([[KernelName]]) <{
//CHECK: inputDimsBinaryVector = [232, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0],
//CHECK: kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
//CHECK: outputDimsBinaryVector = [232, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 0, 0, 0, 0]}>

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
  func.func nested @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @act_shave_only_list_index_1() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:1:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:1:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:1:0>
  %7 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:1:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:1:0>) kernel_text_index(%4 : !VPURegMapped.Index<0:1:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:1:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:1:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>

  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:1:0>) range_index(%r0 : <0:1:0>) kernel_params(%7 : <0:1:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:1:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:1:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion

  VPUMI40XX.OpRanges
}
}

//CHECK: ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK:   VPUASM.MappedInference
//CHECK-DAG: actKernelRangesCount([0, 0, 0, 0, 0, 0])
//CHECK-DAG: actKernelInvocationsCount([0, 0, 0, 0, 0, 0])
//CHECK-NOT: actKernelRanges
//CHECK-NOT: actKernelInvocations

// -----

// Verify that a non-stride-aware kernel (not in SW_KERNELS_SUPPORTING_STRIDE) has unit-size
// dimension strides normalized when serialized into kernel parameter binary vectors.
//
// Input:  1x1x1x8192 NHWC, strides=[16384,1,16384,1] elems — stride_H inherited from parent.
// Output: 1x1x1x8192 NHWC, strides=[8192,1,8192,1] elems — compact.
//
// NHWC memory dim order (outermost→innermost): N, H, W, C
//   MemDim(0)=N, MemDim(1)=H, MemDim(2)=W, MemDim(3)=C
//   memShape (N,H,W,C) = [1, 1, 8192, 1]
//
// Normalized strides for both buffers (innermost-first):
//   C=16, W=16, H=131072 (0x20000), N=131072 bits
//   inputStridesBinaryVector  = [16,0,..., 16,0,..., 0,0,2,0,..., 0,0,2,0,...]
//   outputStridesBinaryVector = [16,0,..., 16,0,..., 0,0,2,0,..., 0,0,2,0,...]
//
// dims binary vector (innermost-first): C=1, W=8192 (0x2000), H=1, N=1
//   = [1,0,0,0, 0,32,0,0, 1,0,0,0, 1,0,0,0]

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @unit_dim_stride_norm inputsInfo : {
DataInfo "input" : tensor<1x1x1x8192xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x1x1x8192xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func nested @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @unit_dim_stride_norm() {
  // Input: stride_H inherited from parent (1x1x1x16384); compact stride_H = 8192 elems.
  %in  = VPURT.DeclareBuffer <CMX_NN> [0] <0>     -> memref<1x1x1x8192xf16, {order = #NHWC, strides = [16384, 1, 16384, 1]}, [@CMX_NN, 0]>
  %out = VPURT.DeclareBuffer <CMX_NN> [0] <16384>  -> memref<1x1x1x8192xf16, {order = #NHWC, strides = [8192,  1,  8192, 1]}, [@CMX_NN, 0]>

  %ktext  = VPUMI40XX.DeclareKernelText  kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %kentry = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %kargs  = VPUMI40XX.DeclareKernelArgs  kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %kp = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%in : memref<1x1x1x8192xf16, {order = #NHWC, strides = [16384, 1, 16384, 1]}, [@CMX_NN, 0]>) outputs(%out : memref<1x1x1x8192xf16, {order = #NHWC, strides = [8192, 1, 8192, 1]}, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange>      -> !VPURegMapped.Index<0:0:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%ktext : !VPURegMapped.Index<0:0:0>) kernel_args_index(%kargs : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%kentry : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%kp : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion

  VPUMI40XX.OpRanges
}
}

//CHECK: VPUASM.KernelParams @{{.+}} inputs([@{{.+}}::@{{.+}}]) outputs([@{{.+}}::@{{.+}}]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_hswish") <{
//CHECK-SAME: inputDimsBinaryVector = [1, 0, 0, 0, 0, 32, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
//CHECK-SAME: outputDimsBinaryVector = [1, 0, 0, 0, 0, 32, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]}>

// -----

// Verify that a stride-aware kernel (in SW_KERNELS_SUPPORTING_STRIDE, e.g. "mvn1") has
// unit-size dimension strides preserved as-is when serialized into kernel parameter binary
// vectors. Such kernels derive dimension sizes from stride ratios at runtime, so normalizing
// inherited subview strides would corrupt those ratios and produce wrong results.
//
// Input:  1x1x1x8192 NHWC, strides=[16384,1,16384,1] elems — stride_H inherited from parent.
// Output: 1x1x1x8192 NHWC, strides=[8192,1,8192,1] elems — compact.
//
// NHWC memory dim order (outermost→innermost): N, H, W, C
//   MemDim(0)=N, MemDim(1)=H, MemDim(2)=W, MemDim(3)=C
//   memShape (N,H,W,C) = [1, 1, 8192, 1]
//
// No normalization — raw strides are passed through (innermost-first):
//   input:  C=16, W=16, H=262144 (0x40000), N=262144 bits  [inflated, preserved]
//   output: C=16, W=16, H=131072 (0x20000), N=131072 bits  [compact, unchanged]

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
net.NetworkInfo entryPoint : @stride_aware_kernel_no_norm inputsInfo : {
DataInfo "input" : tensor<1x1x1x8192xf16>
} outputsInfo : {
DataInfo "output" : tensor<1x1x1x8192xf16>
}

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @stride_aware_kernel_no_norm() {
  // Input: stride_H inherited from parent (1x1x1x16384); compact stride_H = 8192 elems.
  %in  = VPURT.DeclareBuffer <CMX_NN> [0] <0>     -> memref<1x1x1x8192xf16, {order = #NHWC, strides = [16384, 1, 16384, 1]}, [@CMX_NN, 0]>
  %out = VPURT.DeclareBuffer <CMX_NN> [0] <16384>  -> memref<1x1x1x8192xf16, {order = #NHWC, strides = [8192,  1,  8192, 1]}, [@CMX_NN, 0]>

  %ktext  = VPUMI40XX.DeclareKernelText  kernel_path("mvn1") -> !VPURegMapped.Index<0:0:0>
  %kentry = VPUMI40XX.DeclareKernelEntry kernel_path("mvn1") -> !VPURegMapped.Index<0:0:0>
  %kargs  = VPUMI40XX.DeclareKernelArgs  kernel_path("mvn1") -> !VPURegMapped.Index<0:0:0>
  %kp = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%in : memref<1x1x1x8192xf16, {order = #NHWC, strides = [16384, 1, 16384, 1]}, [@CMX_NN, 0]>) outputs(%out : memref<1x1x1x8192xf16, {order = #NHWC, strides = [8192, 1, 8192, 1]}, [@CMX_NN, 0]>) kernel_type("mvn1") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %rtl = VPUMI40XX.DeclareTaskBuffer <ActKernelRange>      -> !VPURegMapped.Index<0:0:0>
  %itl = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange taskLocation(%rtl : !VPURegMapped.Index<0:0:0>) kernel_text_index(%ktext : !VPURegMapped.Index<0:0:0>) kernel_args_index(%kargs : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%kentry : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %i0 = VPUMI40XX.ActKernelInvocation taskLocation(%itl : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%kp : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  ELF.ABIVersion

  VPUMI40XX.OpRanges
}
}

// Input strides: raw inflated (H=262144=0x40000, N=262144 bits), NOT normalized to 0x20000.
// Output strides: compact (H=131072=0x20000, N=131072 bits), unchanged.
//CHECK: VPUASM.KernelParams @{{.+}} inputs([@{{.+}}::@{{.+}}]) outputs([@{{.+}}::@{{.+}}]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("mvn1") <{
//CHECK-SAME: inputDimsBinaryVector = [1, 0, 0, 0, 0, 32, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
//CHECK-SAME: outputDimsBinaryVector = [1, 0, 0, 0, 0, 32, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]}>
