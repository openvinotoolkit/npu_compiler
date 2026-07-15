//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

module @execute_returns_value {
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn() -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'async.execute' that was explicitly marked illegal}}
    %token, %val = async.execute -> !async.value<i32> {
      %c = arith.constant 42 : i32
      async.yield %c : i32
    }
    return
  }
}

// -----

module @no_nested_call {
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn() -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'async.execute' that was explicitly marked illegal}}
    %token = async.execute {
      async.yield
    }
    return
  }
}

// -----

module @no_kernel_section {
  module @SubModule {
    func.func nested @compute(memref<16xf32>) -> ()
  }
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: memref<16xf32>) -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'async.execute' that was explicitly marked illegal}}
    %token = async.execute {
      Core.NestedCall @SubModule::@compute(%arg0) : (memref<16xf32>) -> ()
      async.yield
    }
    return
  }
}

// -----

module @kernel_not_in_section {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @other_kernel "\00\01\02\03"
  }
  module @SubModule {
    func.func nested @compute(memref<16xf32>) -> ()
  }
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: memref<16xf32>) -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'async.execute' that was explicitly marked illegal}}
    %token = async.execute {
      Core.NestedCall @SubModule::@compute(%arg0) : (memref<16xf32>) -> ()
      async.yield
    }
    return
  }
}

// -----

// Negative test: async.execute body contains more than one non-terminator op.
// Only a single Core.NestedCallOp besides the terminator is accepted.

module @extra_op_before_nested_call {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @compute "\00\01\02\03"
  }
  module @SubModule {
    func.func nested @compute(memref<16xf32>) -> ()
  }
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: memref<16xf32>) -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'async.execute' that was explicitly marked illegal}}
    %token = async.execute {
      %c = arith.constant 0 : i32
      Core.NestedCall @SubModule::@compute(%arg0) : (memref<16xf32>) -> ()
      async.yield
    }
    return
  }
}

// -----

// Negative test: memref.store value type must be convertible to a bytecode register.

module @memref_store_complex_value {
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: complex<f32>, %idx: index) -> () attributes {config.pureHostCompileFunc} {
    %0 = memref.alloc() : memref<4xcomplex<f32>>
    // expected-error @+1 {{failed to legalize operation 'memref.store' that was explicitly marked illegal}}
    memref.store %arg0, %0[%idx] : memref<4xcomplex<f32>>
    return
  }
}

// -----

// Negative test: vector store payloads are not register-backed scalar values.

module @memref_store_vector_value {
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: vector<4xi32>, %idx: index) -> () attributes {config.pureHostCompileFunc} {
    %0 = memref.alloc() : memref<4xvector<4xi32>>
    // expected-error @+1 {{failed to legalize operation 'memref.store' that was explicitly marked illegal}}
    memref.store %arg0, %0[%idx] : memref<4xvector<4xi32>>
    return
  }
}

// -----

// Negative test: memref.reinterpret_cast with a non-zero static offset cannot be lowered to
// buffer.view because converting an element-level offset to a byte offset is not implemented.

module @reinterpret_cast_nonzero_static_offset {
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: memref<8xf32>) -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'memref.reinterpret_cast' that was explicitly marked illegal}}
    %1 = memref.reinterpret_cast %arg0 to offset: [4], sizes: [4], strides: [1]
        : memref<8xf32> to memref<4xf32, strided<[1], offset: 4>>
    return
  }
}

// -----

// Negative test: memref.reinterpret_cast with a dynamic offset cannot be lowered to buffer.view.

module @reinterpret_cast_dynamic_offset {
  // expected-error @+1 {{Failed to apply conversion patterns}}
  func.func @fn(%arg0: memref<8xf32>, %off: index) -> () attributes {config.pureHostCompileFunc} {
    // expected-error @+1 {{failed to legalize operation 'memref.reinterpret_cast' that was explicitly marked illegal}}
    %1 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [4], strides: [1]
        : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    return
  }
}
