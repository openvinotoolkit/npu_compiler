//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --mlir-elide-elementsattrs-if-larger 8 --generate-execution-context-funcs  %s | FileCheck %s
// REQUIRES: platform-NPU4000

module {
  // The module can be empty; the pass should add the functions.
}

// CHECK: llvm.func @_mlir_ciface_update_mutable_command_list
// CHECK: llvm.call @npu_level_zero_update_mutable_command_list
// CHECK: llvm.func @_mlir_ciface_destroy_execution_context
// CHECK: llvm.call @npu_level_zero_destroy_execution_context
// CHECK: llvm.func @_mlir_ciface_reset_execution_context
// CHECK: llvm.call @npu_level_zero_reset_execution_context
// CHECK: llvm.func @_mlir_ciface_create_execution_context
// CHECK: llvm.call @npu_level_zero_create_execution_context
