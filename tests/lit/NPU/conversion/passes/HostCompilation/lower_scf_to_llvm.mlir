//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scf-to-cf --convert-cf-to-llvm %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#map = affine_map<(d0) -> (-d0 + 720, 44)>
module {
  llvm.func @npu_level_zero_get_network_metadata(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func internal @_mlir_ciface_get_network_metadata(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.getelementptr %arg1[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %0, %2 : i64, !llvm.ptr
    %3 = llvm.mlir.addressof @HostExec.networkMetadata : !llvm.ptr
    %4 = llvm.mlir.constant(29504 : i64) : i64
    llvm.call @npu_level_zero_get_network_metadata(%3, %4, %arg0, %arg2, %arg3) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.mlir.global internal constant @HostExec.networkMetadata(" \00\00\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.func @npu_level_zero_append_memory_copy(i64, i64, i64, !llvm.ptr)
  llvm.mlir.global internal constant @main_func0_kernel("\7FELF\02\01\00\00\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @main_func2_static_kernel("\7FELF\02\01\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.func @npu_level_zero_submit_commandlist(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @npu_level_zero_execute_graph(!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.mlir.global internal constant @main_func1_static_kernel("\7FELF\02\01\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.func @npu_level_zero_alloc(i64, !llvm.ptr) -> !llvm.ptr
  func.func @main(%arg0: memref<1x16x720x1000xf16>, %arg1: memref<1x16x720x1000xf16>, %arg2: memref<1x16x720x1000xf16>, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<1x16x720x1000xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.mlir.constant(8 : i64) : i64
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.alloca %2 x !llvm.ptr : (i32) -> !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.ptr : (i32) -> !llvm.ptr
    %6 = llvm.mlir.constant(44 : index) : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.mlir.constant(720 : index) : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(720 : index) : i64
    %14 = llvm.mlir.constant(1000 : index) : i64
    %15 = llvm.mlir.constant(16 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(16000 : index) : i64
    %18 = llvm.mlir.constant(11520000 : index) : i64
    %19 = llvm.mlir.constant(11520000 : index) : i64
    %20 = llvm.mlir.zero : !llvm.ptr
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.call @npu_level_zero_alloc(%22, %arg3) : (i64, !llvm.ptr) -> !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %25 = llvm.insertvalue %23, %24[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %12, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %13, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %14, %30[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %15, %31[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %18, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %17, %33[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %15, %34[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.insertvalue %16, %35[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %37 = builtin.unrealized_conversion_cast %36 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x720x1000x16xf16>
    %38 = llvm.sdiv %8, %6  : i64
    %39 = builtin.unrealized_conversion_cast %38 : i64 to index
    %40 = llvm.ptrtoint %arg6 : !llvm.ptr to i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %42 = llvm.sdiv %8, %6  : i64
    %43 = builtin.unrealized_conversion_cast %42 : i64 to index
    scf.for %arg11 = %11 to %9 step %7 {
      %123 = builtin.unrealized_conversion_cast %arg11 : index to i64
      %124 = affine.min #map(%arg11)
      %125 = builtin.unrealized_conversion_cast %124 : index to i64
      %126 = llvm.icmp "ne" %125, %6 : i64
      %127 = scf.if %126 -> (index) {
        %215 = llvm.sub %6, %125 : i64
        %216 = llvm.icmp "sgt" %123, %215 : i64
        cf.assert %216, "Not enough elements to backtrack in scf.for loop"
        %217 = llvm.sub %123, %215 : i64
        %218 = builtin.unrealized_conversion_cast %217 : i64 to index
        scf.yield %218 : index
      } else {
        scf.yield %arg11 : index
      }
      %subview = memref.subview %arg0[0, 0, %127, 0] [1, 16, 44, 1000] [1, 1, 1, 1] : memref<1x16x720x1000xf16> to memref<1x16x44x1000xf16, strided<[11520000, 720000, 1000, 1], offset: ?>>
      %128 = builtin.unrealized_conversion_cast %subview : memref<1x16x44x1000xf16, strided<[11520000, 720000, 1000, 1], offset: ?>> to memref<1x16x44x1000xf16>
      %129 = llvm.mlir.constant(1 : index) : i64
      %130 = llvm.mlir.constant(44 : index) : i64
      %131 = llvm.mlir.constant(1000 : index) : i64
      %132 = llvm.mlir.constant(16 : index) : i64
      %133 = llvm.mlir.constant(1 : index) : i64
      %134 = llvm.mlir.constant(16000 : index) : i64
      %135 = llvm.mlir.constant(704000 : index) : i64
      %136 = llvm.mlir.constant(704000 : index) : i64
      %137 = llvm.mlir.zero : !llvm.ptr
      %138 = llvm.getelementptr %137[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %139 = llvm.ptrtoint %138 : !llvm.ptr to i64
      %140 = llvm.call @npu_level_zero_alloc(%139, %arg3) : (i64, !llvm.ptr) -> !llvm.ptr
      %141 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %142 = llvm.insertvalue %140, %141[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %143 = llvm.insertvalue %140, %142[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %144 = llvm.mlir.constant(0 : index) : i64
      %145 = llvm.insertvalue %144, %143[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %146 = llvm.insertvalue %129, %145[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %147 = llvm.insertvalue %130, %146[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %148 = llvm.insertvalue %131, %147[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %149 = llvm.insertvalue %132, %148[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %150 = llvm.insertvalue %135, %149[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %151 = llvm.insertvalue %134, %150[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %152 = llvm.insertvalue %132, %151[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %153 = llvm.insertvalue %133, %152[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %154 = builtin.unrealized_conversion_cast %153 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x44x1000x16xf16>
      %155 = llvm.mlir.constant(27656 : i64) : i64
      %156 = llvm.mlir.addressof @main_func1_static_kernel : !llvm.ptr
      %157 = llvm.getelementptr %156[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<27656 x i8>
      %158 = llvm.mlir.constant(1 : i32) : i32
      %159 = llvm.mlir.constant(1 : i32) : i32
      %160 = llvm.intr.stacksave : !llvm.ptr
      %161 = llvm.mlir.constant(0 : i64) : i64
      %162 = llvm.getelementptr %4[%161] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %163 = builtin.unrealized_conversion_cast %128 : memref<1x16x44x1000xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %164 = llvm.extractvalue %163[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %165 = llvm.extractvalue %163[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %166 = llvm.ptrtoint %164 : !llvm.ptr to i64
      %167 = llvm.mlir.constant(2 : i64) : i64
      %168 = llvm.mul %167, %165 : i64
      %169 = llvm.add %166, %168 : i64
      llvm.store %169, %162 : i64, !llvm.ptr
      %170 = llvm.mlir.constant(0 : i64) : i64
      %171 = llvm.getelementptr %5[%170] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %172 = builtin.unrealized_conversion_cast %154 : memref<1x44x1000x16xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %173 = llvm.extractvalue %172[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %174 = llvm.extractvalue %172[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %175 = llvm.ptrtoint %173 : !llvm.ptr to i64
      %176 = llvm.mlir.constant(2 : i64) : i64
      %177 = llvm.mul %176, %174 : i64
      %178 = llvm.add %175, %177 : i64
      llvm.store %178, %171 : i64, !llvm.ptr
      llvm.call @npu_level_zero_execute_graph(%4, %158, %5, %159, %157, %155, %arg3, %arg4, %arg5, %arg6, %arg8) : (!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      llvm.intr.stackrestore %160 : !llvm.ptr
      %179 = llvm.mlir.zero : !llvm.ptr
      llvm.call @npu_level_zero_submit_commandlist(%arg6, %arg8, %179, %179) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      %subview_0 = memref.subview %arg1[0, 0, %127, 0] [1, 16, 44, 1000] [1, 1, 1, 1] : memref<1x16x720x1000xf16> to memref<1x16x44x1000xf16, strided<[11520000, 720000, 1000, 1], offset: ?>>
      %180 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x44x1000xf16, strided<[11520000, 720000, 1000, 1], offset: ?>> to memref<1x16x44x1000xf16>
      %subview_1 = memref.subview %37[0, %127, 0, 0] [1, 44, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x44x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %181 = builtin.unrealized_conversion_cast %subview_1 : memref<1x44x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x44x1000x16xf16>
      %182 = llvm.mlir.constant(47656 : i64) : i64
      %183 = llvm.mlir.addressof @main_func2_static_kernel : !llvm.ptr
      %184 = llvm.getelementptr %183[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<47656 x i8>
      %185 = llvm.mlir.constant(2 : i32) : i32
      %186 = llvm.mlir.constant(1 : i32) : i32
      %187 = llvm.intr.stacksave : !llvm.ptr
      %188 = llvm.mlir.constant(0 : i64) : i64
      %189 = llvm.getelementptr %4[%188] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %190 = builtin.unrealized_conversion_cast %180 : memref<1x16x44x1000xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %191 = llvm.extractvalue %190[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %192 = llvm.extractvalue %190[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %193 = llvm.ptrtoint %191 : !llvm.ptr to i64
      %194 = llvm.mlir.constant(2 : i64) : i64
      %195 = llvm.mul %194, %192 : i64
      %196 = llvm.add %193, %195 : i64
      llvm.store %196, %189 : i64, !llvm.ptr
      %197 = llvm.mlir.constant(1 : i64) : i64
      %198 = llvm.getelementptr %4[%197] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %199 = builtin.unrealized_conversion_cast %154 : memref<1x44x1000x16xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %200 = llvm.extractvalue %199[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %201 = llvm.extractvalue %199[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %202 = llvm.ptrtoint %200 : !llvm.ptr to i64
      %203 = llvm.mlir.constant(2 : i64) : i64
      %204 = llvm.mul %203, %201 : i64
      %205 = llvm.add %202, %204 : i64
      llvm.store %205, %198 : i64, !llvm.ptr
      %206 = llvm.mlir.constant(0 : i64) : i64
      %207 = llvm.getelementptr %5[%206] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %208 = builtin.unrealized_conversion_cast %181 : memref<1x44x1000x16xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %209 = llvm.extractvalue %208[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %210 = llvm.extractvalue %208[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %211 = llvm.ptrtoint %209 : !llvm.ptr to i64
      %212 = llvm.mlir.constant(2 : i64) : i64
      %213 = llvm.mul %212, %210 : i64
      %214 = llvm.add %211, %213 : i64
      llvm.store %214, %207 : i64, !llvm.ptr
      llvm.call @npu_level_zero_execute_graph(%4, %185, %5, %186, %184, %182, %arg3, %arg4, %arg5, %arg6, %arg8) : (!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      llvm.intr.stackrestore %187 : !llvm.ptr
    }
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.mlir.constant(16 : index) : i64
    %46 = llvm.mlir.constant(720 : index) : i64
    %47 = llvm.mlir.constant(1000 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(720000 : index) : i64
    %50 = llvm.mlir.constant(11520000 : index) : i64
    %51 = llvm.mlir.constant(11520000 : index) : i64
    %52 = llvm.mlir.zero : !llvm.ptr
    %53 = llvm.getelementptr %52[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.call @npu_level_zero_alloc(%54, %arg3) : (i64, !llvm.ptr) -> !llvm.ptr
    %56 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %57 = llvm.insertvalue %55, %56[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %55, %57[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.mlir.constant(0 : index) : i64
    %60 = llvm.insertvalue %59, %58[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.insertvalue %44, %60[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %62 = llvm.insertvalue %45, %61[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.insertvalue %46, %62[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %64 = llvm.insertvalue %47, %63[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %65 = llvm.insertvalue %50, %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %66 = llvm.insertvalue %49, %65[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %67 = llvm.insertvalue %47, %66[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.insertvalue %48, %67[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = builtin.unrealized_conversion_cast %68 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x16x720x1000xf16>
    %70 = llvm.add %40, %1 : i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    %72 = llvm.mlir.constant(27656 : i64) : i64
    %73 = llvm.mlir.addressof @main_func0_kernel : !llvm.ptr
    %74 = llvm.getelementptr %73[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<27656 x i8>
    %75 = llvm.mlir.constant(1 : i32) : i32
    %76 = llvm.mlir.constant(1 : i32) : i32
    %77 = llvm.intr.stacksave : !llvm.ptr
    %78 = llvm.mlir.constant(0 : i64) : i64
    %79 = llvm.getelementptr %4[%78] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %80 = builtin.unrealized_conversion_cast %37 : memref<1x720x1000x16xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %81 = llvm.extractvalue %80[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %82 = llvm.extractvalue %80[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %83 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %84 = llvm.mlir.constant(2 : i64) : i64
    %85 = llvm.mul %84, %82 : i64
    %86 = llvm.add %83, %85 : i64
    llvm.store %86, %79 : i64, !llvm.ptr
    %87 = llvm.mlir.constant(0 : i64) : i64
    %88 = llvm.getelementptr %5[%87] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %89 = builtin.unrealized_conversion_cast %69 : memref<1x16x720x1000xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %90 = llvm.extractvalue %89[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.extractvalue %89[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %92 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %93 = llvm.mlir.constant(2 : i64) : i64
    %94 = llvm.mul %93, %91 : i64
    %95 = llvm.add %92, %94 : i64
    llvm.store %95, %88 : i64, !llvm.ptr
    llvm.call @npu_level_zero_execute_graph(%4, %75, %5, %76, %74, %72, %arg3, %arg4, %arg5, %71, %arg8) : (!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %77 : !llvm.ptr
    %96 = llvm.mlir.zero : !llvm.ptr
    llvm.call @npu_level_zero_submit_commandlist(%71, %arg8, %96, %96) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %97 = llvm.mlir.constant(1 : index) : i64
    %98 = llvm.mlir.constant(16 : index) : i64
    %99 = llvm.mlir.constant(720 : index) : i64
    %100 = llvm.mlir.constant(1000 : index) : i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.mlir.constant(720000 : index) : i64
    %103 = llvm.mlir.constant(11520000 : index) : i64
    %104 = llvm.mlir.constant(11520000 : index) : i64
    %105 = llvm.mlir.zero : !llvm.ptr
    %106 = llvm.getelementptr %105[%104] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %107 = llvm.ptrtoint %106 : !llvm.ptr to i64
    %108 = llvm.extractvalue %68[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %109 = llvm.extractvalue %68[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %110 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %111 = llvm.mlir.constant(2 : i64) : i64
    %112 = llvm.mul %111, %109 : i64
    %113 = llvm.add %110, %112 : i64
    %114 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %115 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %116 = llvm.ptrtoint %114 : !llvm.ptr to i64
    %117 = llvm.mlir.constant(2 : i64) : i64
    %118 = llvm.mul %117, %115 : i64
    %119 = llvm.add %116, %118 : i64
    %120 = llvm.add %70, %1 : i64
    %121 = llvm.inttoptr %120 : i64 to !llvm.ptr
    llvm.call @npu_level_zero_append_memory_copy(%113, %119, %107, %121) : (i64, i64, i64, !llvm.ptr) -> ()
    %122 = llvm.mlir.zero : !llvm.ptr
    llvm.call @npu_level_zero_submit_commandlist(%121, %arg8, %arg9, %arg10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}

// CHECK-NOT: scf.for
// CHECK: llvm.br
