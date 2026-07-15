//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --init-compiler="platform=%platform%" --convert-intermediate-bytecode-ops %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {
  bytecode.func_section @func_section {
    bytecode.ext.func @dynamic_offset_arg (memref<2x3xf32, strided<[3, 1], offset: ?>>) -> () {
      bytecode.ret
    }
  }
}

// CHECK: Bytecode buffer_type cannot encode dynamic offset for memref
