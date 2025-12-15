//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --pass-pipeline='builtin.module(clone-reserved-resources-from-top-module)' %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @TopModule attributes {config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  //CHECK-NOT: config.PipelineOptions
  config.PipelineOptions @Options {
    config.Option @VPU.EnableExtraStaticShapeOps : true
  }

  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @SampleReservedMemory {
            config.MemoryResource 512 bytes of @CMX_NN offset 1474048
        }
    }
  }


  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
    // CHECK: config.PipelineOptions @Options {
    // CHECK-NEXT:     config.Option @VPU.EnableExtraStaticShapeOps : true


    // CHECK: config.Resources 1 of @NCE at 1.300000e+03 MHz {
    // CHECK-NEXT:     config.MemoryResource 1473536 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    // CHECK-NEXT: builtin.module @ReservedMemory
    // CHECK-NEXT: module @SampleReservedMemory
    // CHECK-NEXT: config.MemoryResource 512 bytes of @CMX_NN offset 1474048




    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    } outputsInfo : {
        DataInfo "prob" : tensor<1x16x4x4xf16>
    }
    func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
        %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
        return %results: tensor<1x16x4x4xf16>
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

}
