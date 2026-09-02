//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --make-distributed-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0236605052270141E-4:128>

// CHECK-LABEL: @UnrolledTypeSimpleConversion
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x3x112x112xf16>
func.func @UnrolledTypeSimpleConversion(%arg0: tensor<1x3x112x112xf16>) -> tensor<1x4x112x112x!qElemType, {order = #NHWC}> {
    %0 = VPU.UnrolledType(%arg0 : tensor<1x3x112x112xf16>) -> !VPU.DistributedTensor<1x3x112x112xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %1 = VPU.NCE.Permute(%0) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 49.895641326904297 : f64>} -> !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %2 = VPU.UnrolledType(%1 : !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) -> tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    return %2 : tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    //CHECK: [[COPY_0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x3x112x112xf16> -> !VPU.DistributedTensor<1x3x112x112xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    //CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_0]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 49.895641326904297 : f64>} -> !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    //CHECK: [[COPY_1:%.+]] = VPU.Copy([[PERMUTE]]) : !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}> -> tensor<1x4x112x112x!qElemType, {order = #NHWC}>
    //CHECK: return [[COPY_1]] : tensor<1x4x112x112x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0236605052270141E-4:128>

// CHECK-LABEL: @DeleteUnrolledType
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x3x112x112xf16>
func.func @DeleteUnrolledType(%arg0: tensor<1x3x112x112xf16>) -> tensor<1x4x112x112x!qElemType, {order = #NHWC}> {
    %0 = VPU.UnrolledType(%arg0 : tensor<1x3x112x112xf16>) -> tensor<1x3x112x112xf16>
    %1 = VPU.NCE.Permute(%0) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 49.895641326904297 : f64>} -> tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    //CHECK-NOT: VPU.UnrolledType
}

// -----

#C = affine_map<(d0) -> (d0)>

!Distributed = !VPU.DistributedTensor<100xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

// CHECK-LABEL: @UnrollEmptyOp
func.func @UnrollEmptyOp() -> !Distributed {
    %empty = VPU.Empty : tensor<100xf16>
    %unroll = VPU.UnrolledType(%empty : tensor<100xf16>) -> !Distributed
    return %unroll : !Distributed

    // CHECK:     [[EMPTY:%.+]] = VPU.Empty : !VPU.DistributedTensor<100xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-NOT: VPU.UnrolledType
    // CHECK:     return [[EMPTY]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SCF_Pad_Dim_Yield_Cast_With_NonIdentityCast
// CHECK-SAME:  ([[IN:%.+]]: tensor<1x12x?x?xf16, {order = #NHWC}>) -> tensor<1x12x?x?xf16, {order = #NHWC}>
func.func @SCF_Pad_Dim_Yield_Cast_With_NonIdentityCast(
    %arg0: tensor<1x12x?x?xf16, {order = #NHWC}>
) -> tensor<1x12x?x?xf16, {order = #NHWC}> {

  // Non-identity cast: precise -> bounded/opaque
  // CHECK: [[BOUNDED:%.+]] = builtin.unrealized_conversion_cast {{.+}} : tensor<1x12x?x?xf16, {order = #NHWC}> to tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
  %bounded = builtin.unrealized_conversion_cast %arg0
      : tensor<1x12x?x?xf16, {order = #NHWC}>
      to tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK: tensor.dim
  %c2 = arith.constant 2 : index
  %h_raw = tensor.dim %bounded, %c2 : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK: tensor.dim
  %c3 = arith.constant 3 : index
  %w_raw = tensor.dim %bounded, %c3 : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

  // Output buffer
  %out = tensor.empty(%h_raw, %w_raw)
      : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

  // Loop constants
  %zero = arith.constant 0 : index
  %hstep = arith.constant 30 : index
  %wstep = arith.constant 192 : index
  %padv = arith.constant 0.0 : f16

  // Outer loop
  // CHECK: scf.for
  %hloop = scf.for %hi = %zero to %h_raw step %hstep iter_args(%hout = %out)
      -> (tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    // Inner loop
    // CHECK: scf.for
    %wloop = scf.for %wi = %zero to %w_raw step %wstep iter_args(%wout = %hout)
        -> (tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

      %h_rem = arith.subi %h_raw, %hi : index
      %w_rem = arith.subi %w_raw, %wi : index
      %h_sz = arith.minui %h_rem, %hstep : index
      %w_sz = arith.minui %w_rem, %wstep : index

      // CHECK: tensor.extract_slice
      %sl = tensor.extract_slice %bounded[0, 0, %hi, %wi] [1, 12, %h_sz, %w_sz] [1, 1, 1, 1]
          : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
            to tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 30, 192]> : tensor<4xsi64>, order = #NHWC}>

      // CHECK: tensor.pad
      // CHECK: ^bb0
      // CHECK: tensor.yield
      %pd = tensor.pad %sl low[0, 0, 0, 0] high[0, 0, 1, 1] {
      ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
        tensor.yield %padv : f16
      } : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 30, 192]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 31, 193]> : tensor<4xsi64>, order = #NHWC}>

      // CHECK: tensor.insert_slice
      %ins = tensor.insert_slice %pd into %wout[0, 0, %hi, %wi] [1, 12, %h_sz, %w_sz] [1, 1, 1, 1]
          : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 31, 193]> : tensor<4xsi64>, order = #NHWC}>
            into tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

      scf.yield %ins : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    }
    scf.yield %wloop : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
  }

  // Cast back to precise tensor for return
  // CHECK: [[BACK:%.+]] = builtin.unrealized_conversion_cast {{.+}} : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x12x?x?xf16, {order = #NHWC}>
  %back = builtin.unrealized_conversion_cast %hloop
      : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x12x?x?xf16, {order = #NHWC}>

  // CHECK: return [[BACK]] : tensor<1x12x?x?xf16, {order = #NHWC}>
  return %back : tensor<1x12x?x?xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ShapeCastWithNonDistributedUnrolledType
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x16x8x8xf16>)
func.func @ShapeCastWithNonDistributedUnrolledType(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x64x1xf16> {
    %0 = VPU.UnrolledType(%arg0 : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
    %1 = VPU.ShapeCast {shape = [1, 16, 64, 1]} inputs(%0 : tensor<1x16x8x8xf16>) -> tensor<1x16x64x1xf16>
    %2 = VPU.UnrolledType(%1 : tensor<1x16x64x1xf16>) -> tensor<1x16x64x1xf16>
    return %2 : tensor<1x16x64x1xf16>

    // ShapeCast remains legal because UnrolledTypeOp types are not DistributedTensorType
    // CHECK-NOT: VPU.Copy
    // CHECK: [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 16, 64, 1]} inputs([[ARG0]] : tensor<1x16x8x8xf16>) -> tensor<1x16x64x1xf16>
    // CHECK: return [[SHAPECAST]] : tensor<1x16x64x1xf16>
}
