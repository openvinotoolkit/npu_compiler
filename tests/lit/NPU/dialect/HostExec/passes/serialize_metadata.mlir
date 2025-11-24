//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile allow-custom-values=true" --serialize-network-metadata  %s | FileCheck %s
// REQUIRES: arch-NPU40XX


#map = affine_map<(d0) -> (-d0 + 720, 44)>
module @Add attributes {HostExec.numSubgraphs = 3 : i64, config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "Add_3" friendlyName = "output" : tensor<1x16x720x1000xf16>
  }
  func.func @main(%arg0: memref<1x16x720x1000xf16>, %arg1: memref<1x16x720x1000xf16>, %arg2: memref<1x16x720x1000xf16>, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: llvm.func internal @_mlir_ciface_get_network_metadata
// CHECK: llvm.call @npu_level_zero_get_network_metadata
// CHECK-NOT: config.PipelineOptions
// CHECK-NOT: IE.Resource
// CHECK-NOT: config.MemoryResource
// CHECK-NOT: IE.(a-z|A-Z)+Resource
// CHECK-NOT: net.NetworkInfo

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
module @MaxPool attributes {HostExec.numSubgraphs = 1 : i64, config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" tensorNames = ["input1"] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "MaxPool_2.0" friendlyName = "output" : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }
  func.func @main(%arg0: memref<1x16x720x?xf16>, %arg1: memref<1x16x720x?xf16>, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: llvm.func internal @_mlir_ciface_get_network_metadata
// CHECK: llvm.call @npu_level_zero_get_network_metadata
// CHECK-NOT: config.PipelineOptions
// CHECK-NOT: IE.Resource
// CHECK-NOT: config.MemoryResource
// CHECK-NOT: net.NetworkInfo
