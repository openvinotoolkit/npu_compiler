//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-memref-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

func.func private @main_func0(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                              %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x90x1000x16xf16>) outputs(%arg1 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>

    return %1 : memref<1x90x1000x16xf16>
}

func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>,
                %arg2: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
    %c90 = arith.constant 90 : index
    %c720 = arith.constant 720 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x720x1000x16xf16>

    scf.for %arg3 = %c0 to %c720 step %c90 {
      %subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16>
               to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16>
                 to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %0 = builtin.unrealized_conversion_cast %subview
         : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %1 = builtin.unrealized_conversion_cast %subview_0
         : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %alloc_1 = memref.alloc() : memref<1x90x1000x16xf16>
      %2 = func.call @main_func0(%0, %1, %alloc_1)
         : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
      %subview_2 = memref.subview %alloc[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                 : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      memref.copy %2, %subview_2 : memref<1x90x1000x16xf16>
          to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
    }

    memref.copy %alloc, %arg2 : memref<1x720x1000x16xf16> to memref<1x720x1000x16xf16>
    return %arg2 : memref<1x720x1000x16xf16>
}

// CHECK: func.func private [[MAIN_FUNC0:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x720x1000x16xf16>, [[ARG1:%.+]]: memref<1x720x1000x16xf16>, [[ARG2:%.+]]: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
// CHECK:   [[C90:%.+]] = arith.constant 90 : index
// CHECK:   [[C720:%.+]] = arith.constant 720 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index

// CHECK:   scf.for [[ARG3:%.+]] = [[C0]] to [[C720]] step [[C90]] {
// CHECK:     [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:     [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:     [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]]
// CHECK:     [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]]
// CHECK:     [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:     [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]]
// CHECK:     [[CALL:%.+]] = func.call [[MAIN_FUNC0]]([[CAST0]], [[CAST1]], [[CAST2]])

// -----

func.func private @main_func0(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                              %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x90x1000x16xf16>) outputs(%arg1 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>

    return %1 : memref<1x90x1000x16xf16>
}

func.func @main(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x90x1000x16xf16>

    %call = func.call @main_func0(%arg0, %arg1, %alloc)
       : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>

    memref.copy %alloc, %arg2 : memref<1x90x1000x16xf16> to memref<1x90x1000x16xf16>
    return %arg2 : memref<1x90x1000x16xf16>
}

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x90x1000x16xf16>, [[ARG1:%.+]]: memref<1x90x1000x16xf16>, [[ARG2:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
// CHECK:    [[CALL:%.+]] = call @main_func0([[ARG0]], [[ARG1]], [[ARG2]])
// CHECK:    return [[ARG2]] : memref<1x90x1000x16xf16>

// -----

module @ScheduleCopyFunction {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x?x1000xf16>
    DataInfo "input2" : tensor<1x16x?x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x?x1000xf16>
  }
  func.func @main_func0_static(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>, %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
    return %1 : memref<1x90x1000x16xf16>
  }
  func.func @main(%arg0: memref<1x?x1000x16xf16>, %arg1: memref<1x?x1000x16xf16>, %arg2: memref<1x?x1000x16xf16>) -> memref<1x?x1000x16xf16> {
    %c90 = arith.constant 90 : index
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1000x1000x16xf16>
    scf.for %arg3 = %c0 to %c1000 step %c90 {
      %input0_subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
      %input1_subview = memref.subview %arg1[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
      %3 = builtin.unrealized_conversion_cast %input0_subview : memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %4 = builtin.unrealized_conversion_cast %input1_subview : memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %tmp_buffer = memref.alloc() : memref<1x90x1000x16xf16>
      %5 = func.call @main_func0_static(%3, %4, %tmp_buffer) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
      %output_buf_subview = memref.subview %alloc[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x1000x1000x16xf16> to memref<1x90x1000x16xf16, strided<[16000000, 16000, 16, 1], offset: ?>>
      memref.copy %5, %output_buf_subview : memref<1x90x1000x16xf16> to memref<1x90x1000x16xf16, strided<[16000000, 16000, 16, 1], offset: ?>>
    }
    memref.copy %alloc, %arg2 : memref<1x1000x1000x16xf16> to memref<1x?x1000x16xf16>
    return %arg2 : memref<1x?x1000x16xf16>
  }

   // CHECK: func.func [[MAIN_FUNC0:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {

   // CHECK: func.func @main([[ARG0:%.+]]: memref<1x?x1000x16xf16>, [[ARG1:%.+]]: memref<1x?x1000x16xf16>, [[ARG2:%.+]]: memref<1x?x1000x16xf16>) -> memref<1x?x1000x16xf16> {
   // CHECK:   [[STEP:%.+]] = arith.constant 90 : index
   // CHECK:   [[LOOP_START:%.+]] = arith.constant 0 : index
   // CHECK:   [[LOOP_END:%.+]] = arith.constant 1000 : index

   // CHECK:   scf.for [[ITER:%.+]] = [[LOOP_START]] to [[LOOP_END]] step [[STEP]] {
   // CHECK:     [[SUBVIEW0:%.+]] = memref.subview [[ARG0]]
   // CHECK:     [[SUBVIEW1:%.+]] = memref.subview [[ARG1]]
   // CHECK:     [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]]
   // CHECK:     [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]]
   // CHECK:     [[SUBVIEW2:%.+]] = memref.subview [[ARG2]]
   // CHECK:     [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]]
   // CHECK:     [[CALL:%.+]] = func.call [[MAIN_FUNC0]]([[CAST0]], [[CAST1]], [[CAST2]])
}

// -----

module @ScheduleCopyAndEltwiseFunctions {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x?x1000xf16>
    DataInfo "input2" : tensor<1x16x?x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x?x1000xf16>
  }
  func.func @main_func0_static(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>, %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
    return %1 : memref<1x90x1000x16xf16>
  }
  func.func @main_func1_static(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>, %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
        inputs(%arg0 as %input_0: memref<1x90x1000x16xf16>, %arg1 as %input_1: memref<1x90x1000x16xf16>)
        outputs(%arg2 as %output: memref<1x90x1000x16xf16>) on tile 0 -> memref<1x90x1000x16xf16> {
        VPUIP.SW.Kernel.run (%input_0, %input_1, %output) : memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>
    }
    return %1 : memref<1x90x1000x16xf16>
  }
  func.func @main(%arg0: memref<1x?x1000x16xf16>, %arg1: memref<1x?x1000x16xf16>, %arg2: memref<1x?x1000x16xf16>) -> memref<1x?x1000x16xf16> {
    %c90 = arith.constant 90 : index
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1000x1000x16xf16>
    scf.for %arg3 = %c0 to %c1000 step %c90 {
      %input0_subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
      %input1_subview = memref.subview %arg1[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
      %3 = builtin.unrealized_conversion_cast %input0_subview : memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %4 = builtin.unrealized_conversion_cast %input1_subview : memref<1x90x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %tmp_buffer = memref.alloc() : memref<1x90x1000x16xf16>
      %5 = func.call @main_func0_static(%3, %4, %tmp_buffer) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
      %eltwise_input = memref.subview %alloc[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x1000x1000x16xf16> to memref<1x90x1000x16xf16, strided<[16000000, 16000, 16, 1], offset: ?>>
      %eltwise_input_static = builtin.unrealized_conversion_cast %eltwise_input : memref<1x90x1000x16xf16, strided<[16000000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %dim = memref.dim %alloc, %c0 : memref<1x1000x1000x16xf16>
      %6 = func.call @main_func1_static(%5, %eltwise_input_static, %tmp_buffer) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
      %subview_3 = memref.subview %alloc[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x1000x1000x16xf16> to memref<1x90x1000x16xf16, strided<[16000000, 16000, 16, 1], offset: ?>>
      memref.copy %6, %subview_3 : memref<1x90x1000x16xf16> to memref<1x90x1000x16xf16, strided<[16000000, 16000, 16, 1], offset: ?>>
    }

    memref.copy %alloc, %arg2 : memref<1x1000x1000x16xf16> to memref<1x?x1000x16xf16>
    return %arg2 : memref<1x?x1000x16xf16>
  }

   // CHECK: func.func [[MAIN_FUNC0:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {

   // CHECK: func.func [[MAIN_FUNC1:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {

   // CHECK: func.func @main([[ARG0:%.+]]: memref<1x?x1000x16xf16>, [[ARG1:%.+]]: memref<1x?x1000x16xf16>, [[ARG2:%.+]]: memref<1x?x1000x16xf16>) -> memref<1x?x1000x16xf16> {
   // CHECK:   [[STEP:%.+]] = arith.constant 90 : index
   // CHECK:   [[LOOP_START:%.+]] = arith.constant 0 : index
   // CHECK:   [[LOOP_END:%.+]] = arith.constant 1000 : index

   // CHECK:   scf.for [[ITER:%.+]] = [[LOOP_START]] to [[LOOP_END]] step [[STEP]] {
   // CHECK:     [[IN0_SUBVIEW:%.+]] = memref.subview [[ARG0]][0, [[ITER]], 0, 0]
   // CHECK:     [[IN1_SUBVIEW:%.+]] = memref.subview [[ARG1]][0, [[ITER]], 0, 0]
   // CHECK:     [[IN0_SUBVIEW_STATIC:%.+]] = builtin.unrealized_conversion_cast [[IN0_SUBVIEW]]
   // CHECK:     [[IN1_SUBVIEW_STATIC:%.+]] = builtin.unrealized_conversion_cast [[IN1_SUBVIEW]]
   // CHECK:     [[ALLOC:%.+]] = memref.alloc()
   // CHECK:     [[COPY_OUTPUT:%.+]] = func.call [[MAIN_FUNC0]]([[IN0_SUBVIEW_STATIC]], [[IN1_SUBVIEW_STATIC]], [[ALLOC]])
   // CHECK:     [[ELTWISE_IN0_SUBVIEW:%.+]] = memref.subview [[ARG2]][0, [[ITER]], 0, 0]
   // CHECK:     [[ELTWISE_IN0_STATIC:%.+]] = builtin.unrealized_conversion_cast [[ELTWISE_IN0_SUBVIEW]]
   // CHECK:     [[DIM:%.+]] = memref.dim [[ARG2]]
   // CHECK:     [[OUTPUT_SUBVIEW:%.+]] = memref.subview [[ARG2]][0, [[ITER]], 0, 0]
   // CHECK:     [[OUTPUT_STATIC:%.+]] = builtin.unrealized_conversion_cast [[OUTPUT_SUBVIEW]]
   // CHECK:     [[CALL0:%.+]] = func.call [[MAIN_FUNC1]]([[COPY_OUTPUT]], [[ELTWISE_IN0_STATIC]], [[OUTPUT_STATIC]])
}

// -----

// CHECK: module [[NESTED_CALL:@.+]] {
module @NestedCall {
  //CHECK: func.func private [[MAIN_FUNC0:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
  func.func private @main_func0(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                                %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x90x1000x16xf16>) outputs(%arg1 : memref<1x90x1000x16xf16>)
        -> memref<1x90x1000x16xf16>
      %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>)
        -> memref<1x90x1000x16xf16>

      return %1 : memref<1x90x1000x16xf16>
  }
}

func.func @main(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x90x1000x16xf16>

    %call = Core.NestedCall @NestedCall::@main_func0(%arg0, %arg1, %alloc)
       : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>

    memref.copy %alloc, %arg2 : memref<1x90x1000x16xf16> to memref<1x90x1000x16xf16>
    return %arg2 : memref<1x90x1000x16xf16>
}

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x90x1000x16xf16>, [[ARG1:%.+]]: memref<1x90x1000x16xf16>, [[ARG2:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
// CHECK-NOT: memref.alloc()
// CHECK:    [[CALL:%.+]] = Core.NestedCall [[NESTED_CALL]]::[[MAIN_FUNC0]]([[ARG0]], [[ARG1]], [[ARG2]])
// CHECK-NOT: memref.copy
// CHECK:    return [[ARG2]] : memref<1x90x1000x16xf16>

// -----

func.func @main_func0_dims_H_cases_0_static(%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
    return %arg1 : memref<1x16x28x1280xf16>
  }
func.func @main_func0_dims_H_cases_2_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
    return %arg1 : memref<1x16x28x1280xf16>
  }
func.func @main_func0_dims_H_cases_1_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
    return %arg1 : memref<1x16x28x1280xf16>
  }
func.func @CopiesInCasesOfIndexSwitch(%arg0: memref<1x16x?x1280xf16>, %arg1: memref<1x16x?x1280xf16>, %arg2: index, %arg3: index) -> memref<1x16x?x1280xf16> {
  %false = arith.constant false
  %c2 = arith.constant 2 : index
  %c28 = arith.constant 28 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c2 : memref<1x16x?x1280xf16>
  %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<1x16x?x1280xf16>
  scf.for %arg4 = %c0 to %dim step %c28 {
    %13 = scf.index_switch %arg2 -> memref<1x16x28x1280xf16>
    case 0 {
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x30x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = func.call @main_func0_dims_H_cases_0_static(%14, %alloc_2) : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    case 1 {
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x29x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = func.call @main_func0_dims_H_cases_1_static(%14, %alloc_2) : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    case 2 {
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x29x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = func.call @main_func0_dims_H_cases_2_static(%14, %alloc_2) : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    default {
      cf.assert %false, "Unsupported case"
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x30x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = func.call @main_func0_dims_H_cases_0_static(%14, %alloc_2) : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    %subview = memref.subview %alloc[0, 0, %arg4, 0] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
    memref.copy %13, %subview : memref<1x16x28x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
  }
  memref.copy %alloc, %arg1 : memref<1x16x?x1280xf16> to memref<1x16x?x1280xf16>
  return %arg1 : memref<1x16x?x1280xf16>

  // CHECK: func.func [[FUNC0:@.+]](%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
  // CHECK: func.func [[FUNC1:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
  // CHECK: func.func [[FUNC2:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)

  // CHECK: func.func @CopiesInCasesOfIndexSwitch([[ARG0:%.+]]: memref<1x16x?x1280xf16>, [[ARG1:%.+]]: memref<1x16x?x1280xf16>, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index)
  // CHECK: [[FALSE:%.+]] = arith.constant false
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[C28:%.+]] = arith.constant 28 : index
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM:%.+]] = memref.dim [[ARG0]], [[C2]]

  // CHECK: scf.for [[ARG4:%.+]] = [[C0]] to [[DIM]] step [[C28]] {

  // CHECK: scf.index_switch [[ARG2]]

  // CHECK: case 0 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK: case 1 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK: case 2 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK: default {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK-NOT: memref.copy
  // CHECK: return [[ARG1]]
}

// -----

module @Module0 {
    func.func @main_func0_dims_H_cases_0_static(%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
        -> memref<1x16x28x1280xf16> {
        return %arg1 : memref<1x16x28x1280xf16>
    }
}

module @Module1 {
    func.func @main_func0_dims_H_cases_1_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
        -> memref<1x16x28x1280xf16> {
        return %arg1 : memref<1x16x28x1280xf16>
    }
}


module @Module2 {
    func.func @main_func0_dims_H_cases_2_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
        -> memref<1x16x28x1280xf16> {
        return %arg1 : memref<1x16x28x1280xf16>
    }
}

func.func @CopiesInCasesOfIndexSwitchWithNestedCall(%arg0: memref<1x16x?x1280xf16>, %arg1: memref<1x16x?x1280xf16>, %arg2: index, %arg3: index) -> memref<1x16x?x1280xf16> {
  %false = arith.constant false
  %c2 = arith.constant 2 : index
  %c28 = arith.constant 28 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c2 : memref<1x16x?x1280xf16>
  %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<1x16x?x1280xf16>
  scf.for %arg4 = %c0 to %dim step %c28 {
    %13 = scf.index_switch %arg2 -> memref<1x16x28x1280xf16>
    case 0 {
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x30x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = Core.NestedCall @Module0::@main_func0_dims_H_cases_0_static(%14, %alloc_2)
          : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    case 1 {
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x29x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = Core.NestedCall @Module1::@main_func0_dims_H_cases_1_static(%14, %alloc_2)
          : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    case 2 {
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x29x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = Core.NestedCall @Module2::@main_func0_dims_H_cases_2_static(%14, %alloc_2)
          : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    default {
      cf.assert %false, "Unsupported case"
      %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x30x1280xf16>
      %alloc_2 = memref.alloc() : memref<1x16x28x1280xf16>
      %15 = Core.NestedCall @Module0::@main_func0_dims_H_cases_0_static(%14, %alloc_2)
          : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
      scf.yield %15 : memref<1x16x28x1280xf16>
    }
    %subview = memref.subview %alloc[0, 0, %arg4, 0] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
    memref.copy %13, %subview : memref<1x16x28x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
  }
  memref.copy %alloc, %arg1 : memref<1x16x?x1280xf16> to memref<1x16x?x1280xf16>
  return %arg1 : memref<1x16x?x1280xf16>

  // CHECK: module [[MODULE0:@.+]] {
  // CHECK: func.func [[FUNC0:@.+]](%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>)

  // CHECK: module [[MODULE1:@.+]] {
  // CHECK: func.func [[FUNC1:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)

  // CHECK: module [[MODULE2:@.+]] {
  // CHECK: func.func [[FUNC2:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)

  // CHECK: func.func @CopiesInCasesOfIndexSwitchWithNestedCall([[ARG0:%.+]]: memref<1x16x?x1280xf16>, [[ARG1:%.+]]: memref<1x16x?x1280xf16>, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index)
  // CHECK: [[FALSE:%.+]] = arith.constant false
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[C28:%.+]] = arith.constant 28 : index
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM:%.+]] = memref.dim [[ARG0]], [[C2]]

  // CHECK: scf.for [[ARG4:%.+]] = [[C0]] to [[DIM]] step [[C28]] {

  // CHECK: scf.index_switch [[ARG2]]

  // CHECK: case 0 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK: case 1 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK: case 2 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK: default {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy

  // CHECK-NOT: memref.copy
  // CHECK: return [[ARG1]]
}
