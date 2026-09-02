//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics --allow-unregistered-dialect
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

func.func @BlockArgCountMismatch(%arg: f32) -> f32 {
  // expected-error@+1 {{'Shave.KernelRegion' op entry block has 0 argument(s) but op has 1 input(s)}}
  %out = Shave.KernelRegion(%arg : f32) -> f32 {
  ^bb0:
    %undef = "test.undefined_val"() : () -> f32
    Shave.KernelRegion.Yield %undef : f32
  }
  return %out : f32
}

// -----

func.func @BlockArgTypeMismatch(%arg: i32) -> i32 {
  // expected-error@+1 {{'Shave.KernelRegion' op type mismatch at input 0}}
  %out = Shave.KernelRegion(%arg : i32) -> i32 {
  ^bb0(%barg: f32):
    %undef = "test.undefined_val"() : () -> i32
    Shave.KernelRegion.Yield %undef : i32
  }
  return %out : i32
}

// -----

func.func @YieldCountMismatch(%a: f32, %b: f32) -> f32 {
  // expected-error@+1 {{'Shave.KernelRegion' op yields 2 value(s) but op has 1 result(s)}}
  %out = Shave.KernelRegion(%a, %b : f32, f32) -> f32 {
  ^bb0(%ba: f32, %bb: f32):
    Shave.KernelRegion.Yield %ba, %bb : f32, f32
  }
  return %out : f32
}

// -----

func.func @YieldTypeMismatch(%arg: f32) -> i32 {
  // expected-error@+1 {{'Shave.KernelRegion' op type mismatch at result 0}}
  %out = Shave.KernelRegion(%arg : f32) -> i32 {
  ^bb0(%barg: f32):
    Shave.KernelRegion.Yield %barg : f32
  }
  return %out : i32
}
