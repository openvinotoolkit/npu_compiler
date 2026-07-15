//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-intermediate-bytecode-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test ext.assert -> assert conversion and ext.func -> func conversion
module {

bytecode.func_section @func_section {
  bytecode.ext.func @fn () -> () {
    %0 = bytecode.virtual_general_register
    bytecode.set_imm %0, 0
    bytecode.ext.assert %0, "Assertion failed"
    bytecode.ret
  }
}

// CHECK:  bytecode.string_section @string_section {
// CHECK:    bytecode.string @function_name_0 "fn"
// CHECK:    bytecode.string @assert_msg_1 "Assertion failed"
// CHECK:  }
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @function_type_0 #bytecode.function_type<arguments = [], results = []>
// CHECK:  }
// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.func @fn @function_name_0 @function_type_0 {
// CHECK:      [[REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG]], 0
// CHECK:      bytecode.assert [[REG]], @assert_msg_1
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

// Test ext.func -> func conversion with type decomposition
module {

bytecode.func_section @func_section {
  bytecode.ext.func @add (i64, i64) -> (i64) {
    %0 = bytecode.virtual_general_register
    bytecode.set_imm %0, 42
    bytecode.ret
  }
}

// CHECK:  bytecode.string_section @string_section {
// CHECK:    bytecode.string @function_name_0 "add"
// CHECK:  }
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@i64, @i64], results = [@i64]>
// CHECK:  }
// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.func @add @function_name_0 @function_type_{{[0-9]+}} {
// CHECK:      [[REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG]], 42
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

// Test type deduplication: same type used multiple times results in a single type section entry
module {

bytecode.func_section @func_section {
  bytecode.ext.func @identity (i64) -> (i64) {
    bytecode.ret
  }
}

// The i64 type should only appear once despite being used as both argument and result
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@i64], results = [@i64]>
// CHECK:  }

}

// -----

// Test memref with float element type decomposition into buffer type
module {

bytecode.func_section @func_section {
  bytecode.ext.func @float_buffer_fn (memref<2x3x4xf32>) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE754>
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @f32, rank = 3, shape = [2, 3, 4]
// CHECK-SAME: strides = [12, 4, 1]>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@buffer_type_{{[0-9]+}}], results = []>
// CHECK:  }

}

// -----

// Test E4M3 float format mapping
module {

bytecode.func_section @func_section {
  bytecode.ext.func @e4m3_fn (f8E4M3FN) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f8_e4m3 #bytecode.float_type<width = 8, format = E4M3>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@f8_e4m3], results = []>
// CHECK:  }

}

// -----

// Test E5M2 float format mapping
module {

bytecode.func_section @func_section {
  bytecode.ext.func @e5m2_fn (f8E5M2) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f8_e5m2 #bytecode.float_type<width = 8, format = E5M2>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@f8_e5m2], results = []>
// CHECK:  }

}

// -----

// Test memref type decomposition into buffer type
module {

bytecode.func_section @func_section {
  bytecode.ext.func @buffer_fn (memref<1x16x32x32xi64>) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32]
// CHECK-SAME: strides = [16384, 1024, 32, 1]>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@buffer_type_{{[0-9]+}}], results = []>
// CHECK:  }

}

// -----

module {

bytecode.func_section @func_section {
  bytecode.ext.func @callee_buf (memref<2x3xf32>) -> () {
    bytecode.ret
  }

  bytecode.ext.func @caller_buf (memref<2x3xf32>) -> () {
    %func_idx = bytecode.virtual_general_register
    %buf = bytecode.virtual_parameter_register 0
    bytecode.set_imm %func_idx, 0
    bytecode.call %func_idx, results(), args(%buf : !bytecode.Register)
    bytecode.ret
  }
}

// CHECK:  bytecode.string_section @string_section {
// CHECK:    bytecode.string @function_name_0 "callee_buf"
// CHECK:    bytecode.string @function_name_1 "caller_buf"
// CHECK:  }
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE754>
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @f32, rank = 2, shape = [2, 3]
// CHECK-SAME: strides = [3, 1]>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@buffer_type_{{[0-9]+}}], results = []>
// CHECK:  }
// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.func @callee_buf @function_name_0 @function_type_{{[0-9]+}} {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.func @caller_buf @function_name_1 @function_type_{{[0-9]+}} {
// CHECK:      [[FUNC_IDX:%.+]] = bytecode.virtual_general_register
// CHECK:      [[BUF:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      bytecode.set_imm [[FUNC_IDX]], 0
// CHECK:      bytecode.call [[FUNC_IDX]], results(), args([[BUF]] : !bytecode.Register)
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

// Test ext.buffer.create lowering: float memref in the body of an ext.func
module {

bytecode.func_section @func_section {
  bytecode.ext.func @alloc_fn () -> () {
    %0 = bytecode.virtual_general_register
    bytecode.ext.buffer.create %0, memref<2x3xf32>
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [], results = []>
// CHECK:    bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE754>
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @f32, rank = 2, shape = [2, 3]
// CHECK-SAME: strides = [3, 1]>
// CHECK:  }
// CHECK-LABEL: bytecode.func @alloc_fn
// CHECK:        [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:        [[S0:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.set_imm [[S0]], 2
// CHECK:        [[S1:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.set_imm [[S1]], 3
// CHECK:        [[ST0:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.set_imm [[ST0]], 3
// CHECK:        [[ST1:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.set_imm [[ST1]], 1
// CHECK:        bytecode.buffer.create [[DST]], @f32 shape([[S0]], [[S1]]) strides([[ST0]], [[ST1]])
// CHECK:        bytecode.ret

}

// -----

// Test section insertion
module {
bytecode.string_section @string_section {
  bytecode.string @str_network_name "Conversion"
  bytecode.string @str_parameter_10 "Parameter_10"
  bytecode.string @str_convert_11 "Convert_11"
}
bytecode.type_section @type_section {
  bytecode.type @type_f16 #bytecode.float_type<width = 16, format = IEEE754>
}
bytecode.func_section @func_section {
  bytecode.ext.func @alloc_fn () -> () {
    %0 = bytecode.virtual_general_register
    bytecode.ext.buffer.create %0, memref<2x3xf32>
    bytecode.ret
  }
}
// CHECK:  bytecode.string_section @string_section {
// CHECK:    bytecode.string @str_network_name "Conversion"
// CHECK:    bytecode.string @str_parameter_10 "Parameter_10"
// CHECK:    bytecode.string @str_convert_11 "Convert_11"
// CHECK:    bytecode.string @function_name_0 "alloc_fn"
// CHECK:  }
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @type_f16 #bytecode.float_type<width = 16, format = IEEE754>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [], results = []>
// CHECK:    bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE754>
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @f32, rank = 2, shape = [2, 3], strides = [3, 1]>
// CHECK:  }
// CHECK:  bytecode.constant_section @constant_section {
// CHECK:  }
}
