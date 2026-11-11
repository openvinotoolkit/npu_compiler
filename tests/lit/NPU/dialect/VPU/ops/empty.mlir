//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @OptimizeEmptySlice
func.func @OptimizeEmptySlice() -> tensor<50xf16> {
   %empty = VPU.Empty : tensor<100xf16>
   %slice = VPU.Slice %empty [0] [50] : tensor<100xf16> to tensor<50xf16>
   return %slice : tensor<50xf16>

   // CHECK:      [[EMPTY:%.+]] = VPU.Empty : tensor<50xf16>
   // CHECK-NOT:  VPU.Slice
   // CHECK:      return [[EMPTY]]
}

// -----

#C = affine_map<(d0) -> (d0)>

!Distributed100 = !VPU.DistributedTensor<100xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed50 = !VPU.DistributedTensor<50xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

// CHECK-LABEL: @OptimizeEmptySliceDistributed
func.func @OptimizeEmptySliceDistributed() -> !Distributed50 {
   %empty = VPU.Empty : !Distributed100
   %slice = VPU.Slice %empty [0] [50] : !Distributed100 to !Distributed50
   return %slice : !Distributed50

   // CHECK:      [[EMPTY:%.+]] = VPU.Empty : !VPU.DistributedTensor<50xf16, #C, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
   // CHECK-NOT:  VPU.Slice
   // CHECK:      return [[EMPTY]]
}

// -----

// CHECK-LABEL: @OptimizeEmptyMultiplySlices
func.func @OptimizeEmptyMultiplySlices() -> (tensor<50xf16>, tensor<20xf16>, tensor<10xf16>) {
   %empty = VPU.Empty : tensor<100xf16>
   %slice1 = VPU.Slice %empty [0] [50] : tensor<100xf16> to tensor<50xf16>
   %slice2 = VPU.Slice %empty [10] [20] : tensor<100xf16> to tensor<20xf16>
   %slice3 = VPU.Slice %empty [90] [10] : tensor<100xf16> to tensor<10xf16>
   return %slice1, %slice2, %slice3 : tensor<50xf16>, tensor<20xf16>, tensor<10xf16>

   // CHECK-DAG:  [[EMPTY1:%.+]] = VPU.Empty : tensor<50xf16>
   // CHECK-DAG:  [[EMPTY2:%.+]] = VPU.Empty : tensor<20xf16>
   // CHECK-DAG:  [[EMPTY3:%.+]] = VPU.Empty : tensor<10xf16>
   // CHECK-NOT:  VPU.Slice
   // CHECK:      return [[EMPTY1]], [[EMPTY2]], [[EMPTY3]]
}
