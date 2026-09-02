//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-sw-kernels %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Note this would have incorrectly bufferized with the usual VPU to VPUIP
// bufferization because of the read-after-write conflicts caused by the
// DestinationPassingStyle operations.
//
// Since the VPU to VPUIP bufferization does not insert tensor copies on
// read-after-write conflicts and the DestinationPassingStyle bufferization
// model assumes read-after-write conflicts have been solved, we end up directly
// bufferizing as-is, with the output of %1 being overwritten by %2. The pre-bufferization
// semantics of the IR were to compute exp(x) + log(x), however without inserting
// tensor copies we emit code that computes log(x) + log(x).

// If we have DestinationPassingStyle ops in the IR there is no way to guard
// against read-after-write conflicts occurring, as various optimizations can
// introduce them (e.g. CSE). This is why --one-shot-bufferize-sw-kernels
// needs to use full one-shot bufferization and is expected to correctly bufferize
// this IR.

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @foo
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @foo {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x1x16x1000xf16>, %arg1: memref<1x1x16x1000xf16>) {
      %0 = bufferization.to_tensor %arg1 restrict writable : memref<1x1x16x1000xf16> to tensor<1x1x16x1000xf16>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x16x1000xf16>) outs(%0 : tensor<1x1x16x1000xf16>) {
      ^bb0(%in: f16, %out: f16):
        %4 = math.exp %in : f16
        linalg.yield %4 : f16
      } -> tensor<1x1x16x1000xf16>
      %2 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x16x1000xf16>) outs(%0 : tensor<1x1x16x1000xf16>) {
      ^bb0(%in: f16, %out: f16):
        %4 = math.log %in : f16
        linalg.yield %4 : f16
      } -> tensor<1x1x16x1000xf16>
      %3 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %2 : tensor<1x1x16x1000xf16>, tensor<1x1x16x1000xf16>) outs(%0 : tensor<1x1x16x1000xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %4 = arith.addf %in, %in_0 : f16
        linalg.yield %4 : f16
      } -> tensor<1x1x16x1000xf16>
      bufferization.materialize_in_destination %3 in writable %arg1 : (tensor<1x1x16x1000xf16>, memref<1x1x16x1000xf16>) -> ()
      return

// CHECK: func.func @generated_0(
// CHECK-SAME: [[ARG0:%.+]]: memref<1x1x16x1000xf16>, [[ARG1:%.+]]: memref<1x1x16x1000xf16>) {
// CHECK: [[ALLOC:%.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x1x16x1000xf16>
// CHECK: linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : memref<1x1x16x1000xf16>) outs([[ALLOC]] : memref<1x1x16x1000xf16>)
// CHECK: linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : memref<1x1x16x1000xf16>) outs([[ARG1]] : memref<1x1x16x1000xf16>)
// CHECK: linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ALLOC]], [[ARG1]] : memref<1x1x16x1000xf16>, memref<1x1x16x1000xf16>) outs([[ARG1]] : memref<1x1x16x1000xf16>)
    }
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }
  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
    return %0 : tensor<1x1x16x1000xf16>
  }
}
