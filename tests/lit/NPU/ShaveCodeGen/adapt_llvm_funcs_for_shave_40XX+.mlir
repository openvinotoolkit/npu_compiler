//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adapt-LLVM-funcs-for-shave --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @SingleCosLayer {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    llvm.func @generated_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32) -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)> {
      %0 = llvm.mlir.constant(1000 : index) : i32
      %1 = llvm.mlir.constant(1 : index) : i32
      %2 = llvm.mlir.constant(0 : index) : i32
      %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %4 = llvm.insertvalue %arg11, %3[0] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %5 = llvm.insertvalue %arg12, %4[1] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %6 = llvm.insertvalue %arg13, %5[2] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %7 = llvm.insertvalue %arg14, %6[3, 0] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %8 = llvm.insertvalue %arg18, %7[4, 0] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %9 = llvm.insertvalue %arg15, %8[3, 1] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %10 = llvm.insertvalue %arg19, %9[4, 1] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %11 = llvm.insertvalue %arg16, %10[3, 2] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %12 = llvm.insertvalue %arg20, %11[4, 2] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %13 = llvm.insertvalue %arg17, %12[3, 3] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %14 = llvm.insertvalue %arg21, %13[4, 3] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      llvm.br ^bb1(%2 : i32)
    ^bb1(%15: i32):  // 2 preds: ^bb0, ^bb11
      %16 = llvm.icmp "slt" %15, %1 : i32
      llvm.cond_br %16, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      llvm.br ^bb3(%2 : i32)
    ^bb3(%17: i32):  // 2 preds: ^bb2, ^bb10
      %18 = llvm.icmp "slt" %17, %1 : i32
      llvm.cond_br %18, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      llvm.br ^bb5(%2 : i32)
    ^bb5(%19: i32):  // 2 preds: ^bb4, ^bb9
      %20 = llvm.icmp "slt" %19, %1 : i32
      llvm.cond_br %20, ^bb6, ^bb10
    ^bb6:  // pred: ^bb5
      llvm.br ^bb7(%2 : i32)
    ^bb7(%21: i32):  // 2 preds: ^bb6, ^bb8
      %22 = llvm.icmp "slt" %21, %0 : i32
      llvm.cond_br %22, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %23 = llvm.mul %15, %0  : i32
      %24 = llvm.mul %17, %0  : i32
      %25 = llvm.add %23, %24  : i32
      %26 = llvm.mul %19, %0  : i32
      %27 = llvm.add %25, %26  : i32
      %28 = llvm.add %27, %21  : i32
      %29 = llvm.getelementptr %arg1[%28] : (!llvm.ptr, i32) -> !llvm.ptr, f16
      %30 = llvm.load %29 : !llvm.ptr -> f16
      %31 = llvm.intr.cos(%30)  : (f16) -> f16
      %32 = llvm.mul %15, %0  : i32
      %33 = llvm.mul %17, %0  : i32
      %34 = llvm.add %32, %33  : i32
      %35 = llvm.mul %19, %0  : i32
      %36 = llvm.add %34, %35  : i32
      %37 = llvm.add %36, %21  : i32
      %38 = llvm.getelementptr %arg12[%37] : (!llvm.ptr, i32) -> !llvm.ptr, f16
      llvm.store %31, %38 : f16, !llvm.ptr
      %39 = llvm.add %21, %1  : i32
      llvm.br ^bb7(%39 : i32)
    ^bb9:  // pred: ^bb7
      %40 = llvm.add %19, %1  : i32
      llvm.br ^bb5(%40 : i32)
    ^bb10:  // pred: ^bb5
      %41 = llvm.add %17, %1  : i32
      llvm.br ^bb3(%41 : i32)
    ^bb11:  // pred: ^bb3
      %42 = llvm.add %15, %1  : i32
      llvm.br ^bb1(%42 : i32)
    ^bb12:  // pred: ^bb1
      llvm.return %14 : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%0 as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_0 as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %alloc_1 = memref.alloc() : memref<1x1x1x1000xf16>
    %1 = VPUIP.Copy inputs(%results : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %2 : memref<1x1x1x1000xf16>
  }
}

// CHECK: module @VPU.SW
// CHECK: llvm.func @generated_0(
// CHECK-SAME: [[ARG0:%.+]]: !llvm.ptr) attributes {dso_local} {

// CHECK: [[INPUT_PTR:%.+]] = llvm.getelementptr [[ARG0]][4] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: [[INPUT_ADDR:%.+]] = llvm.load [[INPUT_PTR]] : !llvm.ptr -> !llvm.ptr

// CHECK: [[OUTPUT_PTR:%.+]] = llvm.getelementptr [[ARG0]][48] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: [[OUTPUT_ADDR:%.+]] = llvm.load [[OUTPUT_PTR]] : !llvm.ptr -> !llvm.ptr

// CHECK: llvm.call @__impl_generated_0({{.*}}, [[INPUT_ADDR]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[OUTPUT_ADDR]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>

// CHECK: llvm.return


// CHECK: llvm.func internal @__impl_generated_0({{.*}}: !llvm.ptr, [[ARG1:%.*]]: !llvm.ptr, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: !llvm.ptr, [[ARG12:%.*]]: !llvm.ptr, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32) -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)> {

// CHECK: [[INPUT_VALUE_PTR:%.+]] = llvm.getelementptr [[ARG1]][{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
// CHECK: [[INPUT_VALUE:%.+]] = llvm.load [[INPUT_VALUE_PTR]] : !llvm.ptr -> f16
// CHECK: [[COMPUTATION_RES:%.+]] = llvm.intr.cos([[INPUT_VALUE]])  : (f16) -> f16

// CHECK: [[OUTPUT_VALUE_PTR:%.+]] = llvm.getelementptr [[ARG12]][{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
// CHECK: llvm.store [[COMPUTATION_RES]], [[OUTPUT_VALUE_PTR]] : f16, !llvm.ptr
