//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @SingleCosLayer {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
      %0 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf16>) outs(%arg1 : tensor<1x1x1x1000xf16>) {
      ^bb0(%in: f16, %out: f16):
        %1 = math.cos %in : f16
        linalg.yield %1 : f16
      } -> tensor<1x1x1x1000xf16>
      return %0 : tensor<1x1x1x1000xf16>
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0) {callee = @VPU.SW::@generated_0} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
  }
}

  // CHECK: module @VPU.SW
  // CHECK: func.func @generated
  // CHECK-SAME: [[VAR0:%.+]]: memref<1x1x1x1000xf16>,
  // CHECK-SAME: [[VAR1:%.+]]: memref<1x1x1x1000xf16>
  // CHECK-SAME: -> memref<1x1x1x1000xf16>

  // CHECK: linalg.generic
  // CHECK-SAME: ins([[VAR0]]
  // CHECK-SAME: outs([[VAR1]]

  // CHECK: linalg.yield
  // CHECK-NEXT: }{{[[:space:]]}}
  // CHECK: return [[VAR1]] : memref<1x1x1x1000xf16>

  // CHECK: func.func @main
  // CHECK-SAME: [[ARG0:%.+]]: memref<1x1x1x1000xf16>
  // CHECK-SAME: -> memref<1x1x1x1000xf16>

  // CHECK: [[ALLOC_CMX_INPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  // CHECK: [[CMX_INPUT:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[ARG0]] : memref<1x1x1x1000xf16>)
  // CHECK-SAME: outputs([[ALLOC_CMX_INPUT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>)

  // CHECK: [[ALLOC_CMX_OUTPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

  // CHECK: [[SHAVE_RES:%.+]] = VPUIP.SW.Kernel
  // CHECK-SAME: @VPU.SW::@generated_0
  // CHECK-SAME: inputs([[CMX_INPUT]] as [[KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[ALLOC_CMX_OUTPUT]] as [[KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
  // CHECK-SAME: -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  // CHECK-NEXT: VPUIP.SW.Kernel.run([[KERNEL_INPUT]], [[KERNEL_OUTPUT]])

  // CHECK: [[DDR_OUTOUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>
  // CHECK: [[DDR_OUTPUT:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[SHAVE_RES]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
  // CHECK-SAME: outputs([[DDR_OUTOUT_ALLOC]] : memref<1x1x1x1000xf16>)

  // CHECK: return [[DDR_OUTPUT]] : memref<1x1x1x1000xf16>
