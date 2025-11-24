//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_0d_to_0d inputsInfo : {
    DataInfo "input" : tensor<f16>
  } outputsInfo : {
    DataInfo "output" : tensor<f16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<f16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<f16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_0d_to_0d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<f16, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<f16, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<f16, @DDR>)
        outputs(%2 : memref<f16, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 0 : i32, len = 2 : i32, srcWidth = 2 : i32, srcStride = 2 : i32, srcPlaneStride = 0 : i32, dstWidth = 2 : i32, dstStride = 2 : i32, dstPlaneStride = 0 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_1d_to_1d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_1d_to_1d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 0 : i32, len = 801360 : i32, srcWidth = 801360 : i32, srcStride = 801360 : i32, srcPlaneStride = 0 : i32, dstWidth = 801360 : i32, dstStride = 801360 : i32, dstPlaneStride = 0 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_1d_to_3d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_1d_to_3d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 8904 : i32, srcStride = 8904 : i32, srcPlaneStride = 8904 : i32, dstWidth = 168 : i32, dstStride = 360 : i32, dstPlaneStride = 115200 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_2d_to_3d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_2d_to_3d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
      %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
          inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>)
          outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 168 : i32, srcStride = 192 : i32, srcPlaneStride = 10176 : i32, dstWidth = 168 : i32, dstStride = 360 : i32, dstPlaneStride = 115200 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_2d_to_3d_with_single_shape inputsInfo : {
    DataInfo "input" : tensor<1x3x1x232xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x1x232xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x3x1x232xf16, {order = #NCHW, strides = [155904, 51968, 232, 1]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x1x232xf16, {order = #NCHW, strides = [167040, 55680, 240, 1]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_2d_to_3d_with_single_shape() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x3x1x232xf16, {order = #NCHW, strides = [155904, 51968, 232, 1]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x3x1x232xf16, {order = #NCHW, strides = [167040, 55680, 240, 1]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x3x1x232xf16, {order = #NCHW, strides = [155904, 51968, 232, 1]}, @DDR>)
        outputs(%2 : memref<1x3x1x232xf16, {order = #NCHW, strides = [167040, 55680, 240, 1]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 3 : i32, len = 464 : i32, srcWidth = 464 : i32, srcStride = 464 : i32, srcPlaneStride = 103936 : i32, dstWidth = 464 : i32, dstStride = 480 : i32, dstPlaneStride = 111360 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_3d_to_2d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_3d_to_2d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <20368000> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 168 : i32, srcStride = 360 : i32, srcPlaneStride = 115200 : i32, dstWidth = 168 : i32, dstStride = 192 : i32, dstPlaneStride = 10176 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_3d_to_2d_with_single_shape inputsInfo : {
    DataInfo "input" : tensor<1x3x1x232xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x1x232xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x3x1x232xf16, {order = #NCHW, strides = [167040, 55680, 240, 1]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x1x232xf16, {order = #NCHW, strides = [155904, 51968, 232, 1]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_3d_to_2d_with_single_shape() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <20368000> {swizzlingKey = 0 : i64} -> memref<1x3x1x232xf16, {order = #NCHW, strides = [167040, 55680, 240, 1]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x3x1x232xf16, {order = #NCHW, strides = [155904, 51968, 232, 1]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x3x1x232xf16, {order = #NCHW, strides = [167040, 55680, 240, 1]}, @DDR>)
        outputs(%2 : memref<1x3x1x232xf16, {order = #NCHW, strides = [155904, 51968, 232, 1]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 3 : i32, len = 464 : i32, srcWidth = 464 : i32, srcStride = 480 : i32, srcPlaneStride = 111360 : i32, dstWidth = 464 : i32, dstStride = 464 : i32, dstPlaneStride = 103936 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_3d_to_1d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_3d_to_1d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 168 : i32, srcStride = 360 : i32, srcPlaneStride = 115200 : i32, dstWidth = 8904 : i32, dstStride = 8904 : i32, dstPlaneStride = 8904 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.038143169178682212:128>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_2d_to_3d_input_stride_on_the_highest_dim inputsInfo : {
    DataInfo "input" : tensor<1x512x23x20xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x512x23x20xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [2826240, 1, 61440, 1024]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_2d_to_3d_input_stride_on_the_highest_dim() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <512> {swizzlingKey = 0 : i64} -> memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <11345920> {swizzlingKey = 0 : i64} -> memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [2826240, 1, 61440, 1024]}, @DDR>

    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @DDR>)
        outputs(%2 : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [2826240, 1, 61440, 1024]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 23 : i32, len = 10240 : i32, srcWidth = 10240 : i32, srcStride = 10240 : i32, srcPlaneStride = 10240 : i32, dstWidth = 512 : i32, dstStride = 1024 : i32, dstPlaneStride = 61440 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.038143169178682212:128>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_3d_to_2d_output_stride_on_the_highest_dim inputsInfo : {
    DataInfo "input" : tensor<1x512x23x20xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x512x23x20xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [2826240, 1, 61440, 1024]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_3d_to_2d_output_stride_on_the_highest_dim() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <512> {swizzlingKey = 0 : i64} -> memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [2826240, 1, 61440, 1024]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <11345920> {swizzlingKey = 0 : i64} -> memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @DDR>

    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [2826240, 1, 61440, 1024]}, @DDR>)
        outputs(%2 : memref<1x512x23x20x!qElemType, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   dma_descriptor(<numPlanes = 23 : i32, len = 10240 : i32, srcWidth = 512 : i32, srcStride = 1024 : i32, srcPlaneStride = 61440 : i32, dstWidth = 10240 : i32, dstStride = 10240 : i32, dstPlaneStride = 10240 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_nf4 inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53x!quantileFloatType, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53x!quantileFloatType, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53x!quantileFloatType, {order = #NHWC}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53x!quantileFloatType, {order = #NHWC}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_nf4() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53x!quantileFloatType, {order = #NHWC}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53x!quantileFloatType, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53x!quantileFloatType, {order = #NHWC}, @DDR>)
        outputs(%2 : memref<1x84x90x53x!quantileFloatType, {order = #NHWC}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 0 : i32, len = 200340 : i32, srcWidth = 200340 : i32, srcStride = 200340 : i32, srcPlaneStride = 0 : i32, dstWidth = 200340 : i32, dstStride = 200340 : i32, dstPlaneStride = 0 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @dma_writing_to_register inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @dma_writing_to_register() {
    %cst = const.Declare memref<1024xui8> = dense<1> : tensor<1024xui8>
    %reg_out = VPURT.DeclareBuffer <Register> <788594688> -> memref<1024xui8, @Register>

    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 1024 : i64, srcWidth = 1024 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 16 : i64, dstStride = 32 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%cst : memref<1024xui8>)
        outputs(%reg_out : memref<1024xui8, @Register>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }

  // CHECK: ELF.CreateSection @buffer.Constant.0.constant aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
  // CHECK:   VPUASM.ConstBuffer @Declare_0 !VPUASM.Buffer< "Constant"[0] <0> : memref<1024xui8> :  swizzling(0)> = dense<1> : tensor<1024xui8>
  // CHECK: ELF.CreateLogicalSection @reg.Register.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_NONE") secLocation(<Register>) {
  // CHECK:   VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "Register"[0] <788594688> : memref<1024xui8, @Register> :  swizzling(0)>

  // CHECK: ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
  // CHECK:   VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_f32tof16 inputsInfo : {
    DataInfo "input" : tensor<1x3x4x5xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x4x5xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x3x4x5xf32, {order = #NCHW}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x4x5xf16, {order = #NCHW, strides = [288, 96, 16, 1]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_f32tof16() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x4x5xf32, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x3x4x5xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [288, 96, 16, 1]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x3x4x5xf32, @DDR>)
        outputs(%2 : memref<1x3x4x5xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [288, 96, 16, 1]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 3 : i32, len = 80 : i32, srcWidth = 80 : i32, srcStride = 80 : i32, srcPlaneStride = 80 : i32, dstWidth = 10 : i32, dstStride = 32 : i32, dstPlaneStride = 192 : i32>)

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}
