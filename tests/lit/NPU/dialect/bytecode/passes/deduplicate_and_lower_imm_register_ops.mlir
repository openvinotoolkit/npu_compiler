//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --deduplicate-and-lower-imm-register-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// ===--------------------------------------------------------------------=== //
// Case 1: Single imm_register is lowered to VirtualGeneralRegisterOp + SetImmOp.
// ===--------------------------------------------------------------------=== //
module {
bytecode.func_section @func_section {
    bytecode.ext.func @lower_single () -> () {
        %dst = bytecode.virtual_general_register
        %c42 = bytecode.imm_register 42
        bytecode.set %dst, %c42
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.ext.func @lower_single
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[C42:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.set_imm [[C42]], 42
// CHECK:         bytecode.set [[DST]], [[C42]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 2: Two identical imm_register ops in the same block are deduplicated by
// DeduplicateAndLowerImmRegisterOps to a single VirtualGeneralRegisterOp + SetImmOp.
// ===--------------------------------------------------------------------=== //
module {
bytecode.func_section @func_section {
    bytecode.ext.func @same_block_dedup () -> () {
        %dst1 = bytecode.virtual_general_register
        %dst2 = bytecode.virtual_general_register
        %c1a = bytecode.imm_register 1
        %c1b = bytecode.imm_register 1
        bytecode.set %dst1, %c1a
        bytecode.set %dst2, %c1b
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.ext.func @same_block_dedup
// CHECK:         [[DST1:%.+]] = bytecode.virtual_general_register
// CHECK:         [[DST2:%.+]] = bytecode.virtual_general_register
// CHECK:         [[C1:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.set_imm [[C1]], 1
// CHECK:         bytecode.set [[DST1]], [[C1]]
// CHECK:         bytecode.set [[DST2]], [[C1]]
// CHECK:         bytecode.ret
// CHECK-NOT:     bytecode.set_imm

// -----

// ===--------------------------------------------------------------------=== //
// Case 3: %c1_dst_src is used as the destination (operand 0) of neg.f64 and
// also as the source (operand 1) of set. %c1_src is used as the source
// (operand 1) of neg.f64. The destination use always gets its own fresh
// (VGR + SetImmOp); the two source uses share a single (VGR + SetImmOp).
// Result: two set_imm 1 ops emitted — no deduplication across the src/dst
// boundary. The shared VGR retains its value after neg.f64 writes to [[FRESH]].
// ===--------------------------------------------------------------------=== //
module {
bytecode.func_section @func_section {
    bytecode.ext.func @dst_use_isolated () -> () {
        %c1_dst_src = bytecode.imm_register 1
        %c1_src = bytecode.imm_register 1
        %result = bytecode.virtual_general_register
        bytecode.neg.f64 %c1_dst_src, %c1_src
        bytecode.set %result, %c1_dst_src
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.ext.func @dst_use_isolated
// CHECK:         [[SHARED:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[SHARED]], 1
// CHECK:         [[FRESH:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[FRESH]], 1
// CHECK:         [[RESULT:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.neg.f64 [[FRESH]], [[SHARED]]
// CHECK:         bytecode.set [[RESULT]], [[SHARED]]
// CHECK:         bytecode.ret
// CHECK-NOT:     bytecode.imm_register

// -----

// ===--------------------------------------------------------------------=== //
// Case 4: Negative immediate (-1) is lowered and deduplicated correctly,
// verifying no sign-related issues in the dedup map keying.
// ===--------------------------------------------------------------------=== //
module {
bytecode.func_section @func_section {
    bytecode.ext.func @negative_immediate () -> () {
        %dst = bytecode.virtual_general_register
        %n1a = bytecode.imm_register -1
        %n1b = bytecode.imm_register -1
        bytecode.set %dst, %n1a
        bytecode.set %dst, %n1b
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.ext.func @negative_immediate
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[N1:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[N1]], -1
// CHECK:         bytecode.set [[DST]], [[N1]]
// CHECK:         bytecode.set [[DST]], [[N1]]
// CHECK:         bytecode.ret
// CHECK-NOT:     bytecode.set_imm
