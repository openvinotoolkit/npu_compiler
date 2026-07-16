//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --allocate-bytecode-registers %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// ===--------------------------------------------------------------------=== //
// Case 1: Empty body - no virtuals, no params. G = 0.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "empty_body"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @empty_body @fn_name @fn_type {
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @empty_body
// CHECK-NEXT:    bytecode.ret
// CHECK-NEXT:  }

// -----

// ===--------------------------------------------------------------------=== //
// Case 2: Single virtual with a single use. Allocated to reg 0. G = 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "single_use"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @single_use @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 42
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @single_use
// CHECK:         [[R0:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[R0]], 42
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 3: Single virtual with multiple uses spans all uses in reg 0. G = 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "multi_use"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @multi_use @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 1
        bytecode.set_imm %0, 2
        bytecode.set_imm %0, 3
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @multi_use
// CHECK:         [[R0:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[R0]], 1
// CHECK:         bytecode.set_imm [[R0]], 2
// CHECK:         bytecode.set_imm [[R0]], 3
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 4: Dead virtual (no uses) is erased. G = 0.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "dead_virtual"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @dead_virtual @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @dead_virtual
// CHECK-NOT:     bytecode.virtual_general_register
// CHECK-NOT:     bytecode.general_register
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 5: Two non-overlapping virtuals reuse the same register. G = 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "reuse_after_free"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @reuse_after_free @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 1
        %1 = bytecode.virtual_general_register
        bytecode.set_imm %1, 2
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @reuse_after_free
// CHECK:         [[R0A:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[R0A]], 1
// CHECK:         [[R0B:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[R0B]], 2
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 6: Two virtuals overlapping at the same op get distinct registers. G = 2.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "overlap_two"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @overlap_two @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        %1 = bytecode.virtual_general_register
        bytecode.set %0, %1
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @overlap_two
// CHECK:         [[R0:%.+]] = bytecode.general_register 0
// CHECK:         [[R1:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set [[R0]], [[R1]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 7: Nested live ranges - outer lives across the entire inner range. G = 2.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "nested_ranges"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @nested_ranges @fn_name @fn_type {
        %outer = bytecode.virtual_general_register
        bytecode.set_imm %outer, 10
        %inner = bytecode.virtual_general_register
        bytecode.set_imm %inner, 20
        bytecode.set %outer, %inner
        bytecode.set_imm %outer, 30
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @nested_ranges
// CHECK:         [[OUTER:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[OUTER]], 10
// CHECK:         [[INNER:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set_imm [[INNER]], 20
// CHECK:         bytecode.set [[OUTER]], [[INNER]]
// CHECK:         bytecode.set_imm [[OUTER]], 30
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 8: Chain of 3 virtuals with disjoint lifetimes - all reuse reg 0. G = 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "chain_reuse"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @chain_reuse @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 1
        %1 = bytecode.virtual_general_register
        bytecode.set_imm %1, 2
        %2 = bytecode.virtual_general_register
        bytecode.set_imm %2, 3
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @chain_reuse
// CHECK:         [[RA:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[RA]], 1
// CHECK:         [[RB:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[RB]], 2
// CHECK:         [[RC:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[RC]], 3
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 9: Free pool ordering - when two registers are freed, the next
// allocation picks the lowest number first.
// Interval traces:
//   v0: [1,1]  v1: [3,3]  v2: [5,5]  v3: [7,7]
// After v0, v1 die (by op index 5), freePool = {0, 1}.
// v2 gets 0 (lowest); v3 gets 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "free_pool_order"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @free_pool_order @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        %1 = bytecode.virtual_general_register
        bytecode.set %0, %1
        %2 = bytecode.virtual_general_register
        %3 = bytecode.virtual_general_register
        bytecode.set %2, %3
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @free_pool_order
// CHECK:         [[V0:%.+]] = bytecode.general_register 0
// CHECK:         [[V1:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set [[V0]], [[V1]]
// CHECK:         [[V2:%.+]] = bytecode.general_register 0
// CHECK:         [[V3:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set [[V2]], [[V3]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 10: Three simultaneously live virtuals (3-operand add). G = 3.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "three_live"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @three_live @fn_name @fn_type {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.add.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @three_live
// CHECK:         [[DST:%.+]] = bytecode.general_register 0
// CHECK:         [[LHS:%.+]] = bytecode.general_register 1
// CHECK:         [[RHS:%.+]] = bytecode.general_register 2
// CHECK:         bytecode.add.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 11: Def order differs from first-use order.
// v0 is defined first, but v1 is used first and its range overlaps v0's range.
// Linear scan sorts by interval start (first use): v1 [2,3], v0 [3,3].
// v1 grabs reg 0 first; v0 cannot reuse reg 0 (v1 still alive at op 3), so
// it gets reg 1 even though it was defined earlier in IR.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "def_vs_use_order"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @def_vs_use_order @fn_name @fn_type {
        %v0 = bytecode.virtual_general_register
        %v1 = bytecode.virtual_general_register
        bytecode.set_imm %v1, 1
        bytecode.set %v1, %v0
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @def_vs_use_order
// CHECK:         [[V0:%.+]] = bytecode.general_register 1
// CHECK:         [[V1:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[V1]], 1
// CHECK:         bytecode.set [[V1]], [[V0]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 12: Single-point interval (def and only use at the same op).
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "single_point"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @single_point @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 7
        %1 = bytecode.virtual_general_register
        bytecode.set_imm %1, 8
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @single_point
// CHECK:         [[A:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[A]], 7
// CHECK:         [[B:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[B]], 8
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 13: Parameters only, no general virtuals. G = 0, params at 0, 1.
// Uses a buffer-typed first parameter to demonstrate how
// `#bytecode.buffer_type<...>` entries are wired through the type section
// and referenced from the function signature. The allocator is agnostic to
// parameter element types, so the emitted registers are identical to the
// scalar-only case.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "params_only"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@buf, @i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @params_only @fn_name @fn_type {
        %p0 = bytecode.virtual_parameter_register 0
        %p1 = bytecode.virtual_parameter_register 1
        bytecode.set %p0, %p1
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @params_only
// CHECK:         [[P0:%.+]] = bytecode.general_register 0
// CHECK:         [[P1:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set [[P0]], [[P1]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 14: Non-contiguous paramIndex. Only param 2 is used; params 0, 1 are unused.
// Emits `general_register G+2`; unused slots produce no ops.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "sparse_param"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64, @i64, @i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @sparse_param @fn_name @fn_type {
        %p2 = bytecode.virtual_parameter_register 2
        bytecode.set_imm %p2, 100
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @sparse_param
// CHECK:         [[P2:%.+]] = bytecode.general_register 2
// CHECK:         bytecode.set_imm [[P2]], 100
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 15: Same paramIndex referenced twice - collapsed to a single
// general_register via CSE.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "dup_param"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @dup_param @fn_name @fn_type {
        %p0a = bytecode.virtual_parameter_register 0
        %p0b = bytecode.virtual_parameter_register 0
        bytecode.set %p0a, %p0b
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @dup_param
// CHECK:         [[P0:%.+]] = bytecode.general_register 0
// CHECK-NOT:     bytecode.general_register 0
// CHECK:         bytecode.set [[P0]], [[P0]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 16: Mixed general and parameter virtuals. G = 2, params at 2, 3.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "mixed"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64, @i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @mixed @fn_name @fn_type {
        %a = bytecode.virtual_general_register
        %b = bytecode.virtual_general_register
        %p0 = bytecode.virtual_parameter_register 0
        %p1 = bytecode.virtual_parameter_register 1
        bytecode.set %a, %p0
        bytecode.set %b, %p1
        bytecode.add.i64 %a, %a, %b
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @mixed
// CHECK:         [[A:%.+]] = bytecode.general_register 0
// CHECK:         [[B:%.+]] = bytecode.general_register 1
// CHECK:         [[P0:%.+]] = bytecode.general_register 2
// CHECK:         [[P1:%.+]] = bytecode.general_register 3
// CHECK:         bytecode.set [[A]], [[P0]]
// CHECK:         bytecode.set [[B]], [[P1]]
// CHECK:         bytecode.add.i64 [[A]], [[A]], [[B]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 17: Parameter interleaved with general virtual definitions and uses.
// The param doesn't consume any slot in [0..G-1]; it occupies G+0 only.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "interleaved"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @interleaved @fn_name @fn_type {
        %a = bytecode.virtual_general_register
        bytecode.set_imm %a, 1
        %p0 = bytecode.virtual_parameter_register 0
        bytecode.set %a, %p0
        %b = bytecode.virtual_general_register
        bytecode.set %b, %p0
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @interleaved
// CHECK:         [[A:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[A]], 1
// CHECK:         [[P0:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set [[A]], [[P0]]
// CHECK:         [[B:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set [[B]], [[P0]]
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 18: Two functions in one section. Each function gets its own allocation.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fna_name "func_a"
    bytecode.string @fnb_name "func_b"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @func_a @fna_name @fn_type {
        %0 = bytecode.virtual_general_register
        %1 = bytecode.virtual_general_register
        bytecode.set %0, %1
        bytecode.ret
    }
    bytecode.func @func_b @fnb_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 5
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @func_a
// CHECK:         bytecode.general_register 0
// CHECK:         bytecode.general_register 1
// CHECK:         bytecode.set
// CHECK:         bytecode.ret
//
// CHECK-LABEL: bytecode.func @func_b
// CHECK:         bytecode.general_register 0
// CHECK:         bytecode.set_imm
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 19: Signature declares parameters but none are referenced. The pass must
// still reserve the full calling-convention frame (G + P) by materializing a
// `general_register` at the highest parameter slot (G + P - 1), so that the
// serialized `num_general_registers` matches `G + P`.
// Here G = 1 (one general virtual) and P = 2, so the placeholder lands at
// slot 1 + 2 - 1 = 2.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "unused_params"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64, @i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @unused_params @fn_name @fn_type {
        %0 = bytecode.virtual_general_register
        bytecode.set_imm %0, 7
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @unused_params
// CHECK:         [[R0:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[R0]], 7
// CHECK:         bytecode.general_register 2
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 20: Only a non-last parameter is referenced (param 0 of 2). The pass
// still reserves the final parameter slot so that `num_general_registers`
// covers all P parameters. Here G = 0 and P = 2, so param 0 is emitted at
// slot 0 and the trailing placeholder lands at slot 0 + 2 - 1 = 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "unused_trailing_param"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64, @i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @unused_trailing_param @fn_name @fn_type {
        %p0 = bytecode.virtual_parameter_register 0
        bytecode.set_imm %p0, 1
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @unused_trailing_param
// CHECK:         [[P0:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[P0]], 1
// CHECK:         bytecode.general_register 1
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 21: Parameters declared but function body is empty (no virtuals at all).
// Only the tail-slot placeholder is emitted, pinning `num_general_registers`
// to P. Here G = 0 and P = 3, so the placeholder lands at slot 2.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "params_never_used"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64, @i64, @i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @params_never_used @fn_name @fn_type {
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @params_never_used
// CHECK-NOT:     bytecode.virtual_parameter_register
// CHECK:         bytecode.general_register 2
// CHECK-NEXT:    bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 22: Intra-block VGR in a non-entry block is allocated. G = 1.
// The entry block has no VGRs; the successor defines and uses one locally.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "intra_nonentry"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @intra_nonentry @fn_name @fn_type {
        bytecode.jmp ^bb1
    ^bb1:
        %vgr = bytecode.virtual_general_register
        bytecode.set_imm %vgr, 1
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @intra_nonentry
// CHECK:         bytecode.jmp ^bb1
// CHECK:       ^bb1:
// CHECK:         [[R0:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[R0]], 1
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 23: Cross-block VGR (entry → bb1) and intra-block VGR both in bb1.
// Cross-block VGR interval covers all of bb1, so both use different registers.
// G = 2.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "cross_and_intra"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @cross_and_intra @fn_name @fn_type {
        %cross = bytecode.virtual_general_register
        bytecode.jmp ^bb1
    ^bb1:
        bytecode.set_imm %cross, 1
        %intra = bytecode.virtual_general_register
        bytecode.set_imm %intra, 2
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @cross_and_intra
// CHECK:         [[CROSS:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.jmp ^bb1
// CHECK:       ^bb1:
// CHECK:         bytecode.set_imm [[CROSS]], 1
// CHECK:         [[INTRA:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set_imm [[INTRA]], 2
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 24: Two cross-block VGRs with non-overlapping live ranges reuse reg 0.
// %a lives in {entry, bb1}: interval ends at first op of bb2.
// %b lives in {bb2, bb3}: interval starts at first op of bb2.
// end(%a) <= start(%b) → %a expires → %b reuses reg 0. G = 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "cross_reuse"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @cross_reuse @fn_name @fn_type {
        %a = bytecode.virtual_general_register
        bytecode.jmp ^bb1
    ^bb1:
        bytecode.set_imm %a, 1
        bytecode.jmp ^bb2
    ^bb2:
        %b = bytecode.virtual_general_register
        bytecode.jmp ^bb3
    ^bb3:
        bytecode.set_imm %b, 2
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @cross_reuse
// CHECK:         [[A:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.jmp ^bb1
// CHECK:       ^bb1:
// CHECK:         bytecode.set_imm [[A]], 1
// CHECK:         bytecode.jmp ^bb2
// CHECK:       ^bb2:
// CHECK:         [[B:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.jmp ^bb3
// CHECK:       ^bb3:
// CHECK:         bytecode.set_imm [[B]], 2
// CHECK:         bytecode.ret

// -----

// ===--------------------------------------------------------------------=== //
// Case 25: Back-edge extends %a's live range across the loop body.
// %b is defined in loop_body AFTER %a's last use there. Without back-edge
// liveness %a would appear to expire before %b starts; with liveness %a is
// liveOut(loop_body) so its interval covers loop_body entirely, preventing
// %b from reusing %a's register. G = 2.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "loop_backedge"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @loop_backedge @fn_name @fn_type {
        %a = bytecode.virtual_general_register
        bytecode.set_imm %a, 42
        bytecode.jmp ^loop_hdr
    ^loop_hdr:
        %cond = bytecode.virtual_general_register
        bytecode.set_imm %cond, 0
        bytecode.je %cond, %cond, ^exit, ^loop_body
    ^loop_body:
        bytecode.set_imm %a, 99
        %b = bytecode.virtual_general_register
        bytecode.set_imm %b, 1
        bytecode.jmp ^loop_hdr
    ^exit:
        bytecode.ret
    }
}
}

// CHECK-LABEL: bytecode.func @loop_backedge
// CHECK:         [[A:%.+]] = bytecode.general_register 0
// CHECK:         bytecode.set_imm [[A]], 42
// CHECK:         bytecode.jmp ^bb1
// CHECK:       ^bb1:
// CHECK:         [[COND:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set_imm [[COND]], 0
// CHECK:         bytecode.je [[COND]], [[COND]], ^bb3, ^bb2
// CHECK:       ^bb2:
// CHECK:         bytecode.set_imm [[A]], 99
// CHECK:         [[B:%.+]] = bytecode.general_register 1
// CHECK:         bytecode.set_imm [[B]], 1
// CHECK:         bytecode.jmp ^bb1
// CHECK:       ^bb3:
// CHECK:         bytecode.ret
