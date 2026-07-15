//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter" --canonicalize --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: bytecode.ext.func @mul_add_optimization_test
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @main_func0_static "\7FELF\02\01"
  }
  net.NetworkInfo entryPoint : @mul_add_optimization_test inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 3]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "shape" : tensor<1xi64>
  } outputsInfo : {
    DataInfo "Convert_11" friendlyName = "Result_12" : tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 3]> : tensor<4xsi64>, order = #NCHW}>
  }
  func.func @mul_add_optimization_test(%input: memref<1x?x?x3xui8>, %shape: memref<1xi64>, %output: memref<1x?x?x3xui8>) -> memref<1x?x?x3xui8> attributes {config.pureHostCompileFunc} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %dim = memref.dim %input, %c1 : memref<1x?x?x3xui8>
    %1 = arith.divsi %dim, %c2 : index
    %2 = arith.muli %1, %c-1 : index
    %3 = arith.addi %2, %c2 : index
    %result_i64 = arith.index_cast %3 : index to i64
    memref.store %result_i64, %shape[%c0] : memref<1xi64>
    return %output : memref<1x?x?x3xui8>
  }
  // CHECK:      [[OUTPUT:%.+]] = bytecode.virtual_parameter_register 2
  // CHECK:      [[SHAPE:%.+]] = bytecode.virtual_parameter_register 1
  // CHECK:      [[INPUT:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[C0:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.set_imm [[C0]], 0
  // CHECK:      [[C1:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.set_imm [[C1]], 1
  // CHECK:      [[C2:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.set_imm [[C2]], 2
  // CHECK-NOT:  bytecode.set_imm {{.+}}, -1
  // CHECK:      [[DIM:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.buffer.get_dim [[DIM]], [[INPUT]], [[C1]]
  // CHECK:      [[DIV:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.div.i64 [[DIV]], [[DIM]], [[C2]]
  // CHECK-NOT:  bytecode.mul.i64
  // CHECK:      [[SUB:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.sub.i64 [[SUB]], [[C2]], [[DIV]]
  // CHECK:      bytecode.buffer.store [[SHAPE]], [[SUB]]
  // CHECK:      bytecode.retv
}

// -----

module {
  func.func @mul_neg_one_to_sub(%arg0: i64, %arg1: i64) -> (i64, i64) attributes {config.pureHostCompileFunc} {
    %c_neg1 = arith.constant -1 : i64
    %mul0 = arith.muli %arg0, %c_neg1 : i64
    %add0 = arith.addi %mul0, %arg1 : i64
    %mul1 = arith.muli %arg1, %c_neg1 : i64
    %add1 = arith.addi %arg0, %mul1 : i64
    return %add0, %add1 : i64, i64
  }
}
// CHECK-LABEL: bytecode.ext.func @mul_neg_one_to_sub
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-NOT:  bytecode.set_imm {{.+}}, -1
// CHECK-NOT:  bytecode.mul.i64
// CHECK:      [[SUB0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.sub.i64 [[SUB0]], [[PARAM1]], [[PARAM0]]
// CHECK:      [[SUB1:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.sub.i64 [[SUB1]], [[PARAM0]], [[PARAM1]]
// CHECK:      bytecode.retv
