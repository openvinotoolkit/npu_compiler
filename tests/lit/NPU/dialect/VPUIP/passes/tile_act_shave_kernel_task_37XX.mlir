//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileStridedMVN(%arg0: memref<1x128x64x32xf16, #NWHC>)
        -> memref<1x128x64x32xf16, #NWHC> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>) outputs(%0 : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%1 as %arg1: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    return %4: memref<1x128x64x32xf16, #NWHC>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>

    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[MVN:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16, #NWHC>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileStridedMVNWithDifferentTileSize(%arg0: memref<1x129x64x32xf16, #NWHC>)
        -> memref<1x129x64x32xf16, #NWHC> {
    %0 = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x129x64x32xf16, #NWHC>) outputs(%0 : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%1 as %arg1: memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x129x64x32xf16, #NWHC>
    %4 = VPUIP.Copy inputs(%results : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 : memref<1x129x64x32xf16, #NWHC>) -> memref<1x129x64x32xf16, #NWHC>
    return %4: memref<1x129x64x32xf16, #NWHC>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x129x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_CMX:%.*]] = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 65, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 65, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 65, 0, 0] [1, 64, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 65, 0, 0] [1, 64, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[MVN:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs([[SUBVIEW0]] as %arg1: memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>){
    // CHECK:                       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:                       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX]] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x129x64x32xf16, #NWHC>
    // CHECK:   [[COPY_BACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x129x64x32xf16, #NWHC>) -> memref<1x129x64x32xf16, #NWHC>
    // CHECK:   return [[COPY_BACK]] : memref<1x129x64x32xf16, #NWHC>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, HALF_PIXEL = 0
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[INBUF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[COPY1]] : memref<1x1x192x320xf16>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileHalfPixelInterpolateNotOnOuterMostDim(%arg0: memref<1x3x96x160xf16, [@CMX_NN, 0]>) -> memref<1x3x192x320xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, HALF_PIXEL = 0
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg2: memref<1x3x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x3x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x3x96x160xf16, [@CMX_NN, 0]>, memref<1x3x192x320xf16, [@CMX_NN, 0]>
    }
    return %results : memref<1x3x192x320xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 49, 160] : memref<1x3x96x160xf16, [@CMX_NN, 0]> to memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF_0:%.*]] = memref.alloc() : memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[INPUT_BUF_0]] : memref<1x3x49x160xf16, [@CMX_NN, 0]>) -> memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFF_0:%.*]] = memref.alloc() : memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 3, 49, 160] : memref<1x3x96x160xf16, [@CMX_NN, 0]> to memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF_1:%.*]] = memref.alloc() : memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[INPUT_BUF_1]] : memref<1x3x49x160xf16, [@CMX_NN, 0]>) -> memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFF_1:%.*]] = memref.alloc() : memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[COPY_0]] as %arg1: memref<1x3x49x160xf16, [@CMX_NN, 0]>, [[COPY_1]] as %arg2: memref<1x3x49x160xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFF_0]] as %arg3: memref<1x3x96x320xf16, [@CMX_NN, 0]>, [[OUTPUT_BUFF_1]] as %arg4: memref<1x3x96x320xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x96x320xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x3x49x160xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x3x49x160xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUFF:%.*]] = memref.alloc() : memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 3, 96, 320] : memref<1x3x192x320xf16, [@CMX_NN, 0]> to memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy inputs([[INTERP]]#0 : memref<1x3x96x320xf16, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) -> memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 96, 0] [1, 3, 96, 320] : memref<1x3x192x320xf16, [@CMX_NN, 0]> to memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COPY_3:%.*]] = VPUIP.Copy inputs([[INTERP]]#1 : memref<1x3x96x320xf16, [@CMX_NN, 0]>) outputs([[SUBVIEW_3]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) -> memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFF]] : memref<1x3x192x320xf16, [@CMX_NN, 0]>) -> memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x3x192x320xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Maximum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_max.cpp", VPU.kernel_entry = "eltwise_max"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMaximum(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Maximum inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.*]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.*]] = VPUIP.SubView %arg1 [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[MAXIMUM:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Maximum inputs([[SUBVIEW_0]] as %arg2: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as %arg3: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as %arg4: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as %arg5: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as %arg6: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as %arg7: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3, %arg6) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg4, %arg5, %arg7) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MAXIMUM]]#0, [[MAXIMUM]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileAlignCornersInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, ALIGN_CORNERS = 4
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[INBUF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[COPY1]] : memref<1x1x192x320xf16>
}

func.func @TilePytorchHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:  [[IN_BUFF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[COPY0:%.*]]  = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[IN_BUFF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:  [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUT_BUF]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:  return [[COPY1]] : memref<1x1x192x320xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TilePytorchHalfPixelInterpolateWithInitialTileOffsetOnNonScalingDim(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 16, 0], [0, 0, 16, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:  [[IN_BUFF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[COPY0:%.*]]  = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[IN_BUFF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // initial offset c(16) keep value unchanged after tiling
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 16, 0], [0, 0, 16, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 16, 0], [0, 96, 16, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUT_BUFF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:  [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUT_BUF]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:  return [[COPY1]] : memref<1x1x192x320xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGelu(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[GELU:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_HardSigmoid(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_hardsigmoid.cpp", VPU.kernel_entry = "activation_hardsigmoid"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileHardSigmoid(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_HardSigmoid inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[HARDSIGMOID:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_HardSigmoid inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[HARDSIGMOID]]#0, [[HARDSIGMOID]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSoftmax(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>

    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SOFTMAX:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSoftmaxWhenAxisIsHighestDim(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 32, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 32, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SOFTMAX:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[SUBVIEW0]] as %arg1: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg3) : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [2]}(%arg2, %arg4) : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPYBACK:%.*]]  = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPYBACK]]  : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileSoftmaxForUnsupportedAxis(%arg0: memref<1x128x1x1xf16>)
        -> memref<1x128x1x1xf16> {
    %0 = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x1x1xf16>) outputs(%0 : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x1x1xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x1x1xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x1x1xf16>) -> memref<1x128x1x1xf16>
    return %4: memref<1x128x1x1xf16>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x1x1xf16>) outputs([[INPUT_CMX]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>

    // CHECK:   [[SOFTMAX:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[COPY_CMX]] as %arg1: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[OUTPUT]] as %arg2: memref<1x128x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x1x1xf16, [@CMX_NN, 0]>{
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x1x1xf16>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[SOFTMAX]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x1x1xf16>) -> memref<1x128x1x1xf16>
    // CHECK:   return [[COPYBACK]] : memref<1x128x1x1xf16>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileHalfPixelInterpolateForCMXSizeRequirement(%arg0: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) -> memref<1x16x12x4096xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg2: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x16x12x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x4096xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2048, 6, 16, 1], [4096, 12, 16, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x16x6x2048xf16, [@CMX_NN, 0]>, memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    }
    return %results : memref<1x16x12x4096xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg1: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF]] as %arg2: memref<1x16x12x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x4096xf16, [@CMX_NN, 0]>{
    // CHECK:                         VPUIP.SW.Kernel.run
    // CHECK-NOT:                     VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    return [[INTERP]] : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}
func.func @ConvertOpTest(%arg0: memref<1x64x16x16xf32>) -> memref<1x64x16x16xf16> {
    %0 = memref.alloc() : memref<1x64x16x16xf32, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x64x16x16xf32>) outputs(%0 : memref<1x64x16x16xf32, [@CMX_NN, 0]>) -> memref<1x64x16x16xf32, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x64x16x16xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%1 as %arg1: memref<1x64x16x16xf32, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x64x16x16xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x16x16xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x64x16x16xf32, [@CMX_NN, 0]>, memref<1x64x16x16xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x64x16x16xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x64x16x16xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x16x16xf16>) -> memref<1x64x16x16xf16>
    return %4: memref<1x64x16x16xf16>

    // CHECK:    [[MEMREF1:%.*]] = memref.alloc()
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x16x16xf32>)
    // CHECK:    [[MEMREF2:%.*]] = memref.alloc()

    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 32, 16, 16]
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[MEMREF2]] [0, 0, 0, 0] [1, 32, 16, 16]
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[COPY1]] [0, 32, 0, 0] [1, 32, 16, 16]
    // CHECK:    [[SUBVIEW4:%.*]] = VPUIP.SubView [[MEMREF2]] [0, 32, 0, 0] [1, 32, 16, 16]

    // CHECK:     [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Convert inputs([[SUBVIEW1]] as %arg1: memref<1x32x16x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg2: memref<1x32x16x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW2]] as %arg3: memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>, [[SUBVIEW4]] as %arg4: memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg3)
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg2, %arg4) : memref<1x32x16x16xf32
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}
func.func @NotTileConvertOpTest(%arg0: memref<1x64x10x10xf32>) -> memref<1x64x10x10xf16> {
    %0 = memref.alloc() : memref<1x64x10x10xf32, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x64x10x10xf32>) outputs(%0 : memref<1x64x10x10xf32, [@CMX_NN, 0]>) -> memref<1x64x10x10xf32, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x64x10x10xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%1 as %arg1: memref<1x64x10x10xf32, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x64x10x10xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x10x10xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x64x10x10xf32, [@CMX_NN, 0]>, memref<1x64x10x10xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x64x10x10xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x64x10x10xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x10x10xf16>) -> memref<1x64x10x10xf16>
    return %4: memref<1x64x10x10xf16>


    // CHECK:   [[MEMREF0:%.*]]  = memref.alloc() : memref<1x64x10x10xf32, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x10x10xf32>) outputs([[MEMREF0]] : memref<1x64x10x10xf32, [@CMX_NN, 0]>) -> memref<1x64x10x10xf32, [@CMX_NN, 0]>
    // CHECK:   [[MEMREF1:%.*]] = memref.alloc() : memref<1x64x10x10xf16, [@CMX_NN, 0]>
    // CHECK:   [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[COPY0]] as %arg1: memref<1x64x10x10xf32, [@CMX_NN, 0]>) outputs([[MEMREF1]] as %arg2: memref<1x64x10x10xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x10x10xf16, [@CMX_NN, 0]>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x64x10x10xf32, [@CMX_NN, 0]>, memref<1x64x10x10xf16, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[MEMREF2:%.*]] = memref.alloc() : memref<1x64x10x10xf16>
    // CHECK:   [[COPY1:%.*]] = VPUIP.Copy inputs([[RESULT]] : memref<1x64x10x10xf16, [@CMX_NN, 0]>) outputs([[MEMREF2]] : memref<1x64x10x10xf16>) -> memref<1x64x10x10xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileTanh(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tanh inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[TANH:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Tanh inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[TANH]]#0, [[TANH]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGather(%arg0: memref<30522x26xf16>, %arg1: memref<1x512xsi32>)
        -> memref<1x512x26xf16> {
    %0 = memref.alloc() : memref<30522x26xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<30522x26xf16>) outputs(%0 : memref<30522x26xf16, [@CMX_NN, 0]>) -> memref<30522x26xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x512xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x512xsi32>) outputs(%2 : memref<1x512xsi32, [@CMX_NN, 0]>) -> memref<1x512xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x512x26xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs(%1 as %arg2: memref<30522x26xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<1x512xsi32, [@CMX_NN, 0]>) outputs(%4 as %arg4: memref<1x512x26xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x26xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg2, %arg3, %arg4) : memref<30522x26xf16, [@CMX_NN, 0]>, memref<1x512xsi32, [@CMX_NN, 0]>, memref<1x512x26xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x512x26xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x512x26xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x512x26xf16>) -> memref<1x512x26xf16>
    return %6: memref<1x512x26xf16>

    // CHECK:    [[VAR0:%.*]] = memref.alloc() : memref<30522x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<30522x26xf16>) outputs([[VAR0]] : memref<30522x26xf16, [@CMX_NN, 0]>) -> memref<30522x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR2:%.*]] = memref.alloc() : memref<1x512xsi32, [@CMX_NN, 0]>
    // CHECK:    [[VAR3:%.*]] = VPUIP.Copy inputs(%arg1 : memref<1x512xsi32>) outputs([[VAR2]] : memref<1x512xsi32, [@CMX_NN, 0]>) -> memref<1x512xsi32, [@CMX_NN, 0]>
    // CHECK:    [[VAR4:%.*]] = memref.alloc() : memref<1x512x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR5:%.*]] = VPUIP.SubView [[VAR3]] [0, 0] [1, 256] : memref<1x512xsi32, [@CMX_NN, 0]> to memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR6:%.*]] = VPUIP.SubView [[VAR4]] [0, 0, 0] [1, 256, 26] : memref<1x512x26xf16, [@CMX_NN, 0]> to memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR7:%.*]] = VPUIP.SubView [[VAR3]] [0, 256] [1, 256] : memref<1x512xsi32, [@CMX_NN, 0]> to memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR8:%.*]] = VPUIP.SubView [[VAR4]] [0, 256, 0] [1, 256, 26] : memref<1x512x26xf16, [@CMX_NN, 0]> to memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather inputs([[VAR1]] as %arg2: memref<30522x26xf16, [@CMX_NN, 0]>, [[VAR5]] as %arg3: memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>, [[VAR1]] as %arg4: memref<30522x26xf16, [@CMX_NN, 0]>, [[VAR7]] as %arg5: memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>) outputs([[VAR6]] as %arg6: memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>, [[VAR8]] as %arg7: memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg2, %arg3, %arg6) : memref<30522x26xf16, [@CMX_NN, 0]>, memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg4, %arg5, %arg7) : memref<30522x26xf16, [@CMX_NN, 0]>, memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    [[VAR9:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x512x26xf16, [@CMX_NN, 0]>) -> memref<1x512x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR10:%.*]] = memref.alloc() : memref<1x512x26xf16>
    // CHECK:    [[VAR11:%.*]] = VPUIP.Copy inputs([[VAR9]] : memref<1x512x26xf16, [@CMX_NN, 0]>) outputs([[VAR10]] : memref<1x512x26xf16>) -> memref<1x512x26xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Sigmoid(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSigmoid(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_DepthToSpace(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSWDepthToSpace(%arg0: memref<1x128x12x270xf16, #NHWC>)
        -> memref<1x8x48x1080xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs(%0 : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%1 as %arg1: memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg1, %arg2) : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    return %4: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs([[INBUF]] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 128, 6, 270] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]> to memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 8, 24, 1080] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]> to memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 6, 0] [1, 128, 6, 270] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]> to memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 24, 0] [1, 8, 24, 1080] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]> to memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs([[IN_SUBVIEW0]] as %arg1: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg1, %arg3) : memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg2, %arg4) : memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    return [[COPY1]] : memref<1x8x48x1080xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_DepthToSpace(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileDMADepthToSpace(%arg0: memref<1x128x12x270xf16, #NHWC>)
        -> memref<1x8x48x1080xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs(%0 : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%1 as %arg1: memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, 0]}(%arg1, %arg2) : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    return %4: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs([[INBUF]] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[RES:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs([[COPY0]] as %arg1: memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF]] as %arg2: memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 0]}(%arg1, %arg2) : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[RES]] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    return [[COPY1]] : memref<1x8x48x1080xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Clamp(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_clamp.cpp", VPU.kernel_entry = "activation_clamp"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClamp(%arg0: memref<1x5x34x60xf16, #NHWC>)
        -> memref<1x5x34x60xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs(%0 : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Clamp inputs(%1 as %arg1: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg1, %arg2) : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>, memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
      }
    %3 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    return %4: memref<1x5x34x60xf16, #NHWC>

    // CHECK:   [[INBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[INCOPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs([[INBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB0:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB1:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Clamp
    // CHECK-SAME:    inputs([[IN_SUB0]] as %arg1: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[IN_SUB1]] as %arg2: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUT_SUB0]] as %arg3: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[OUT_SUB1]] as %arg4: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:    -> (memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg1, %arg3) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg2, %arg4) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 :
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUTBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_DDR:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:   [[COPY:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUT_DDR]] : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
}

// -----

module @VPU.SW {
  func.func private @builtin_GRUSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "gru_sequence.cpp", VPU.kernel_entry = "gru_sequence"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileGRUSequence
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: memref<2x5x10xf16>
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: memref<2x1x4xf16>
// CHECK-SAME:        [[OUTPUT0:%arg[0-9]]]: memref<2x1x5x4xf16>
// CHECK-SAME:        [[OUTPUT1:%arg[0-9]]]: memref<2x1x4xf16>
func.func @TileGRUSequence(%arg0: memref<2x5x10xf16>, %arg1: memref<2x1x4xf16>, %arg2: memref<2x1x5x4xf16>, %arg3: memref<2x1x4xf16>) -> (memref<2x1x5x4xf16>, memref<2x1x4xf16>) {
    %cst = const.Declare memref<1x16xf16> = dense<1.000000e+00> : tensor<1x16xf16>
    %cst_0 = const.Declare memref<1x12x4xf16> = dense<1.000000e+00> : tensor<1x12x4xf16>
    %cst_1 = const.Declare memref<1x12x10xf16> = dense<1.000000e+00> : tensor<1x12x10xf16>
    %alloc = memref.alloc() : memref<2x5x10xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<2x5x10xf16>) outputs(%alloc : memref<2x5x10xf16, [@CMX_NN, 0]>) -> memref<2x5x10xf16, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg1 : memref<2x1x4xf16>) outputs(%alloc_2 : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<1x12x10xf16, [@CMX_NN, 0]>
    %2 = VPUIP.Copy inputs(%cst_1 : memref<1x12x10xf16>) outputs(%alloc_3 : memref<1x12x10xf16, [@CMX_NN, 0]>) -> memref<1x12x10xf16, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x12x4xf16, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst_0 : memref<1x12x4xf16>) outputs(%alloc_4 : memref<1x12x4xf16, [@CMX_NN, 0]>) -> memref<1x12x4xf16, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x16xf16, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%cst : memref<1x16xf16>) outputs(%alloc_5 : memref<1x16xf16, [@CMX_NN, 0]>) -> memref<1x16xf16, [@CMX_NN, 0]>
    %alloc_6 = memref.alloc() : memref<2x1x5x4xf16, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>
    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GRUSequence inputs(%0 as %arg4: memref<2x5x10xf16, [@CMX_NN, 0]>, %1 as %arg5: memref<2x1x4xf16, [@CMX_NN, 0]>, %2 as %arg6: memref<1x12x10xf16, [@CMX_NN, 0]>, %3 as %arg7: memref<1x12x4xf16, [@CMX_NN, 0]>, %4 as %arg8: memref<1x16xf16, [@CMX_NN, 0]>) outputs(%alloc_6 as %arg9: memref<2x1x5x4xf16, [@CMX_NN, 0]>, %alloc_7 as %arg10: memref<2x1x4xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<2x1x5x4xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [4, 0, 5, 1, 0.000000e+00]}(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10) : memref<2x5x10xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>, memref<1x12x10xf16, [@CMX_NN, 0]>, memref<1x12x4xf16, [@CMX_NN, 0]>, memref<1x16xf16, [@CMX_NN, 0]>, memref<2x1x5x4xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>
    }
    %alloc_8 = memref.alloc() : memref<2x1x5x4xf16>
    %5 = VPUIP.Copy inputs(%results#0 : memref<2x1x5x4xf16, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>
    %alloc_9 = memref.alloc() : memref<2x1x4xf16>
    %6 = VPUIP.Copy inputs(%results#1 : memref<2x1x4xf16, [@CMX_NN, 0]>) outputs(%alloc_9 : memref<2x1x4xf16>) -> memref<2x1x4xf16>
    %7 = VPUIP.Copy inputs(%5 : memref<2x1x5x4xf16>) outputs(%arg2 : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>
    %8 = VPUIP.Copy inputs(%6 : memref<2x1x4xf16>) outputs(%arg3 : memref<2x1x4xf16>) -> memref<2x1x4xf16>
    return %7, %8 : memref<2x1x5x4xf16>, memref<2x1x4xf16>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x16xf16> = dense<1.000000e+00> : tensor<1x16xf16>
    // CHECK-DAG:   [[CST0:%.*]] = const.Declare memref<1x12x4xf16> = dense<1.000000e+00> : tensor<1x12x4xf16>
    // CHECK-DAG:   [[CST1:%.*]] = const.Declare memref<1x12x10xf16> = dense<1.000000e+00> : tensor<1x12x10xf16>

    // CHECK:       [[ALLOC1:%.*]] = memref.alloc() : memref<2x5x10xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY1:%.*]] = VPUIP.Copy inputs([[INPUT0]] : memref<2x5x10xf16>) outputs([[ALLOC1]] : memref<2x5x10xf16, [@CMX_NN, 0]>) -> memref<2x5x10xf16, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC2:%.*]] = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY2:%.*]] = VPUIP.Copy inputs([[INPUT1]] : memref<2x1x4xf16>) outputs([[ALLOC2]] : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC3:%.*]] = memref.alloc() : memref<1x12x10xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY3:%.*]] = VPUIP.Copy inputs([[CST1]] : memref<1x12x10xf16>) outputs([[ALLOC3]] : memref<1x12x10xf16, [@CMX_NN, 0]>) -> memref<1x12x10xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC4:%.*]] = memref.alloc() : memref<1x12x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY4:%.*]] = VPUIP.Copy inputs([[CST0]] : memref<1x12x4xf16>) outputs([[ALLOC4]] : memref<1x12x4xf16, [@CMX_NN, 0]>) -> memref<1x12x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC5:%.*]] = memref.alloc() : memref<1x16xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY5:%.*]] = VPUIP.Copy inputs([[CST]] : memref<1x16xf16>) outputs([[ALLOC5]] : memref<1x16xf16, [@CMX_NN, 0]>) -> memref<1x16xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC6:%.*]] = memref.alloc() : memref<2x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC7:%.*]] = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBCOPY1_1:%.*]] = VPUIP.SubView [[COPY1]] [0, 0, 0] [1, 5, 10] : memref<2x5x10xf16, [@CMX_NN, 0]> to memref<1x5x10xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBCOPY2_1:%.*]] = VPUIP.SubView [[COPY2]] [0, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBALLOC6_1:%.*]] = VPUIP.SubView [[ALLOC6]] [0, 0, 0, 0] [1, 1, 5, 4] : memref<2x1x5x4xf16, [@CMX_NN, 0]> to memref<1x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBALLOC7_1:%.*]] = VPUIP.SubView [[ALLOC7]] [0, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBCOPY1_2:%.*]] = VPUIP.SubView [[COPY1]] [1, 0, 0] [1, 5, 10] : memref<2x5x10xf16, [@CMX_NN, 0]> to memref<1x5x10xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBCOPY2_2:%.*]] = VPUIP.SubView [[COPY2]] [1, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBALLOC6_2:%.*]] = VPUIP.SubView [[ALLOC6]] [1, 0, 0, 0] [1, 1, 5, 4] : memref<2x1x5x4xf16, [@CMX_NN, 0]> to memref<1x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBALLOC7_2:%.*]] = VPUIP.SubView [[ALLOC7]] [1, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[RESULT:%.*]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_GRUSequence
    // CHECK-SAME:  inputs([[SUBCOPY1_1]] as [[ARG4:%.*]]: memref<1x5x10xf16, [@CMX_NN, 0]>, [[SUBCOPY2_1]] as [[ARG5:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>, [[COPY3]] as [[ARG6:%.*]]: memref<1x12x10xf16, [@CMX_NN, 0]>, [[COPY4]] as [[ARG7:%.*]]: memref<1x12x4xf16, [@CMX_NN, 0]>, [[COPY5]] as [[ARG8:%.*]]: memref<1x16xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[SUBCOPY1_2]] as [[ARG9:%.*]]: memref<1x5x10xf16, [@CMX_NN, 0]>, [[SUBCOPY2_2]] as [[ARG10:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>, [[COPY3]] as [[ARG11:%.*]]: memref<1x12x10xf16, [@CMX_NN, 0]>, [[COPY4]] as [[ARG12:%.*]]: memref<1x12x4xf16, [@CMX_NN, 0]>, [[COPY5]] as [[ARG13:%.*]]: memref<1x16xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[SUBALLOC6_1]] as [[ARG14:%.*]]: memref<1x1x5x4xf16, [@CMX_NN, 0]>, [[SUBALLOC7_1]] as [[ARG15:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:          [[SUBALLOC6_2]] as [[ARG16:%.*]]: memref<1x1x5x4xf16, [@CMX_NN, 0]>, [[SUBALLOC7_2]] as [[ARG17:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  on tile 0 -> (memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [4, 0, 5, 1, 0.000000e+00]}([[ARG4]], [[ARG5]], [[ARG6]], [[ARG7]], [[ARG8]], [[ARG14]], [[ARG15]]) : memref<1x5x10xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x12x10xf16, [@CMX_NN, 0]>, memref<1x12x4xf16, [@CMX_NN, 0]>, memref<1x16xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [4, 0, 5, 1, 0.000000e+00]}([[ARG9]], [[ARG10]], [[ARG11]], [[ARG12]], [[ARG13]], [[ARG16]], [[ARG17]]) : memref<1x5x10xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x12x10xf16, [@CMX_NN, 0]>, memref<1x12x4xf16, [@CMX_NN, 0]>, memref<1x16xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       [[RESULT0:%.*]] = VPUIP.ConcatView inputs([[RESULT]]#0, [[RESULT]]#2 : memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC6]] : memref<2x1x5x4xf16, [@CMX_NN, 0]>) -> memref<2x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[RESULT1:%.*]] = VPUIP.ConcatView inputs([[RESULT]]#1, [[RESULT]]#3 : memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC7]] : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC8:%.*]] = memref.alloc() : memref<2x1x5x4xf16>
    // CHECK:       [[COPY8:%.*]] = VPUIP.Copy inputs([[RESULT0]] : memref<2x1x5x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC8]] : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>

    // CHECK:       [[ALLOC9:%.*]] = memref.alloc() : memref<2x1x4xf16>
    // CHECK:       [[COPY9:%.*]] = VPUIP.Copy inputs([[RESULT1]] : memref<2x1x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC9]] : memref<2x1x4xf16>) -> memref<2x1x4xf16>

    // CHECK:       [[COPY10:%.*]] = VPUIP.Copy inputs([[COPY8]] : memref<2x1x5x4xf16>) outputs([[OUTPUT0]] : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>
    // CHECK:       [[COPY11:%.*]] = VPUIP.Copy inputs([[COPY9]] : memref<2x1x4xf16>) outputs([[OUTPUT1]] : memref<2x1x4xf16>) -> memref<2x1x4xf16>

    // CHECK:       return [[COPY10]], [[COPY11]] : memref<2x1x5x4xf16>, memref<2x1x4xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_GRUSequenceLastPart(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "gru_sequence_last_part.cpp", VPU.kernel_entry = "gru_sequence_last_part"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileGRUSequenceLastPart
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: memref<2x1x5x12xf16>
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: memref<2x1x4xf16>
// CHECK-SAME:        [[OUTPUT0:%arg[0-9]]]: memref<2x1x5x4xf16>
// CHECK-SAME:        [[OUTPUT1:%arg[0-9]]]: memref<2x1x4xf16>
func.func @TileGRUSequenceLastPart(%arg0: memref<2x1x5x12xf16>, %arg1: memref<2x1x4xf16>, %arg2: memref<2x1x5x4xf16>, %arg3: memref<2x1x4xf16>) -> (memref<2x1x5x4xf16>, memref<2x1x4xf16>) {
    %cst = const.Declare memref<1x16xf16> = dense<1.000000e+00> : tensor<1x16xf16>
    %cst_0 = const.Declare memref<1x12x4xf16> = dense<1.000000e+00> : tensor<1x12x4xf16>
    %alloc = memref.alloc() : memref<2x1x5x12xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<2x1x5x12xf16>) outputs(%alloc : memref<2x1x5x12xf16, [@CMX_NN, 0]>) -> memref<2x1x5x12xf16, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg1 : memref<2x1x4xf16>) outputs(%alloc_2 : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<1x12x4xf16, [@CMX_NN, 0]>
    %2 = VPUIP.Copy inputs(%cst_0 : memref<1x12x4xf16>) outputs(%alloc_3 : memref<1x12x4xf16, [@CMX_NN, 0]>) -> memref<1x12x4xf16, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x16xf16, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst : memref<1x16xf16>) outputs(%alloc_4 : memref<1x16xf16, [@CMX_NN, 0]>) -> memref<1x16xf16, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<2x1x5x4xf16, [@CMX_NN, 0]>
    %alloc_6 = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>
    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GRUSequenceLastPart inputs(%0 as %arg4: memref<2x1x5x12xf16, [@CMX_NN, 0]>, %1 as %arg5: memref<2x1x4xf16, [@CMX_NN, 0]>, %2 as %arg6: memref<1x12x4xf16, [@CMX_NN, 0]>, %3 as %arg7: memref<1x16xf16, [@CMX_NN, 0]>) outputs(%alloc_5 as %arg8: memref<2x1x5x4xf16, [@CMX_NN, 0]>, %alloc_6 as %arg9: memref<2x1x4xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<2x1x5x4xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [4, 0, 5, 1, 0.000000e+00]}(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : memref<2x1x5x12xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>, memref<1x12x4xf16, [@CMX_NN, 0]>, memref<1x16xf16, [@CMX_NN, 0]>, memref<2x1x5x4xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>
    }
    %alloc_7 = memref.alloc() : memref<2x1x5x4xf16>
    %4 = VPUIP.Copy inputs(%results#0 : memref<2x1x5x4xf16, [@CMX_NN, 0]>) outputs(%alloc_7 : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>
    %alloc_8 = memref.alloc() : memref<2x1x4xf16>
    %5 = VPUIP.Copy inputs(%results#1 : memref<2x1x4xf16, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<2x1x4xf16>) -> memref<2x1x4xf16>
    %6 = VPUIP.Copy inputs(%4 : memref<2x1x5x4xf16>) outputs(%arg2 : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>
    %7 = VPUIP.Copy inputs(%5 : memref<2x1x4xf16>) outputs(%arg3 : memref<2x1x4xf16>) -> memref<2x1x4xf16>
    return %6, %7 : memref<2x1x5x4xf16>, memref<2x1x4xf16>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x16xf16> = dense<1.000000e+00> : tensor<1x16xf16>
    // CHECK-DAG:   [[CST0:%.*]] = const.Declare memref<1x12x4xf16> = dense<1.000000e+00> : tensor<1x12x4xf16>

    // CHECK:       [[ALLOC1:%.*]] = memref.alloc() : memref<2x1x5x12xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY1:%.*]] = VPUIP.Copy inputs([[INPUT0]] : memref<2x1x5x12xf16>) outputs([[ALLOC1]] : memref<2x1x5x12xf16, [@CMX_NN, 0]>) -> memref<2x1x5x12xf16, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC2:%.*]] = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY2:%.*]] = VPUIP.Copy inputs([[INPUT1]] : memref<2x1x4xf16>) outputs([[ALLOC2]] : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC3:%.*]] = memref.alloc() : memref<1x12x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY3:%.*]] = VPUIP.Copy inputs([[CST0]] : memref<1x12x4xf16>) outputs([[ALLOC3]] : memref<1x12x4xf16, [@CMX_NN, 0]>) -> memref<1x12x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC4:%.*]] = memref.alloc() : memref<1x16xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY4:%.*]] = VPUIP.Copy inputs([[CST]] : memref<1x16xf16>) outputs([[ALLOC4]] : memref<1x16xf16, [@CMX_NN, 0]>) -> memref<1x16xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC5:%.*]] = memref.alloc() : memref<2x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC6:%.*]] = memref.alloc() : memref<2x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBCOPY1_1:%.*]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 1, 5, 12] : memref<2x1x5x12xf16, [@CMX_NN, 0]> to memref<1x1x5x12xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBCOPY2_1:%.*]] = VPUIP.SubView [[COPY2]] [0, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBALLOC5_1:%.*]] = VPUIP.SubView [[ALLOC5]] [0, 0, 0, 0] [1, 1, 5, 4] : memref<2x1x5x4xf16, [@CMX_NN, 0]> to memref<1x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBALLOC6_1:%.*]] = VPUIP.SubView [[ALLOC6]] [0, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBCOPY1_2:%.*]] = VPUIP.SubView [[COPY1]] [1, 0, 0, 0] [1, 1, 5, 12] : memref<2x1x5x12xf16, [@CMX_NN, 0]> to memref<1x1x5x12xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBCOPY2_2:%.*]] = VPUIP.SubView [[COPY2]] [1, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[SUBALLOC5_2:%.*]] = VPUIP.SubView [[ALLOC5]] [1, 0, 0, 0] [1, 1, 5, 4] : memref<2x1x5x4xf16, [@CMX_NN, 0]> to memref<1x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBALLOC6_2:%.*]] = VPUIP.SubView [[ALLOC6]] [1, 0, 0] [1, 1, 4] : memref<2x1x4xf16, [@CMX_NN, 0]> to memref<1x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[RESULT:%.*]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_GRUSequenceLastPart
    // CHECK-SAME:  inputs([[SUBCOPY1_1]] as [[ARG4:%.*]]: memref<1x1x5x12xf16, [@CMX_NN, 0]>, [[SUBCOPY2_1]] as [[ARG5:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>, [[COPY3]] as [[ARG6:%.*]]: memref<1x12x4xf16, [@CMX_NN, 0]>, [[COPY4]] as [[ARG7:%.*]]: memref<1x16xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[SUBCOPY1_2]] as [[ARG8:%.*]]: memref<1x1x5x12xf16, [@CMX_NN, 0]>, [[SUBCOPY2_2]] as [[ARG9:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>, [[COPY3]] as [[ARG10:%.*]]: memref<1x12x4xf16, [@CMX_NN, 0]>, [[COPY4]] as [[ARG11:%.*]]: memref<1x16xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[SUBALLOC5_1]] as [[ARG12:%.*]]: memref<1x1x5x4xf16, [@CMX_NN, 0]>, [[SUBALLOC6_1]] as [[ARG13:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:          [[SUBALLOC5_2]] as [[ARG14:%.*]]: memref<1x1x5x4xf16, [@CMX_NN, 0]>, [[SUBALLOC6_2]] as [[ARG15:%.*]]: memref<1x1x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  on tile 0 -> (memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [4, 0, 5, 1, 0.000000e+00]}([[ARG4]], [[ARG5]], [[ARG6]], [[ARG7]], [[ARG12]], [[ARG13]]) : memref<1x1x5x12xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x12x4xf16, [@CMX_NN, 0]>, memref<1x16xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [4, 0, 5, 1, 0.000000e+00]}([[ARG8]], [[ARG9]], [[ARG10]], [[ARG11]], [[ARG14]], [[ARG15]]) : memref<1x1x5x12xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x12x4xf16, [@CMX_NN, 0]>, memref<1x16xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       [[RESULT0:%.*]] = VPUIP.ConcatView inputs([[RESULT]]#0, [[RESULT]]#2 : memref<1x1x5x4xf16, [@CMX_NN, 0]>, memref<1x1x5x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC5]] : memref<2x1x5x4xf16, [@CMX_NN, 0]>) -> memref<2x1x5x4xf16, [@CMX_NN, 0]>
    // CHECK:       [[RESULT1:%.*]] = VPUIP.ConcatView inputs([[RESULT]]#1, [[RESULT]]#3 : memref<1x1x4xf16, [@CMX_NN, 0]>, memref<1x1x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC6]] : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC7:%.*]] = memref.alloc() : memref<2x1x5x4xf16>
    // CHECK:       [[COPY7:%.*]] = VPUIP.Copy inputs([[RESULT0]] : memref<2x1x5x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC7]] : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>

    // CHECK:       [[ALLOC8:%.*]] = memref.alloc() : memref<2x1x4xf16>
    // CHECK:       [[COPY8:%.*]] = VPUIP.Copy inputs([[RESULT1]] : memref<2x1x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC8]] : memref<2x1x4xf16>) -> memref<2x1x4xf16>

    // CHECK:       [[COPY9:%.*]] = VPUIP.Copy inputs([[COPY7]] : memref<2x1x5x4xf16>) outputs([[OUTPUT0]] : memref<2x1x5x4xf16>) -> memref<2x1x5x4xf16>
    // CHECK:       [[COPY10:%.*]] = VPUIP.Copy inputs([[COPY8]] : memref<2x1x4xf16>) outputs([[OUTPUT1]] : memref<2x1x4xf16>) -> memref<2x1x4xf16>

    //CHECK:        return [[COPY9]], [[COPY10]] : memref<2x1x5x4xf16>, memref<2x1x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Abs(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_abs.cpp", VPU.kernel_entry = "activation_abs"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileAbs(%arg0: memref<1x5x34x60xf16, #NHWC>)
        -> memref<1x5x34x60xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs(%0 : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Abs inputs(%1 as %arg1: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run (%arg1, %arg2) : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>, memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
      }
    %3 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    return %4: memref<1x5x34x60xf16, #NHWC>

    // CHECK:   [[INBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[INCOPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs([[INBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB0:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB1:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Abs
    // CHECK-SAME:    inputs([[IN_SUB0]] as %arg1: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[IN_SUB1]] as %arg2: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUT_SUB0]] as %arg3: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[OUT_SUB1]] as %arg4: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:    -> (memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 :
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUTBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_DDR:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:   [[COPY:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUT_DDR]] : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
}

// -----

module @VPU.SW {
  func.func private @builtin_GatherND(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, none) attributes {VPU.kernel_code = "gatherND.cpp", VPU.kernel_entry = "gatherND"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileGatherNDOpAtC
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x16x180x16xf16, [@CMX_NN, 0]>
// CHECK-SAME:    [[INPUT_1:%.+]]: memref<1x16x1458x2xsi32, [@CMX_NN, 0]>
func.func @TileGatherNDOpAtC(%arg0: memref<1x16x180x16xf16, [@CMX_NN, 0]>, %arg1: memref<1x16x1458x2xsi32, [@CMX_NN, 0]>)
              -> memref<1x16x1458x16xf16, [@CMX_NN, 0]> {
    %alloc_out = memref.alloc() : memref<1x16x1458x16xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherND
          inputs(%arg0 as %arg2: memref<1x16x180x16xf16, [@CMX_NN, 0]>, %arg1 as %arg3: memref<1x16x1458x2xsi32, [@CMX_NN, 0]>)
          outputs(%alloc_out as %arg4: memref<1x16x1458x16xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x1458x16xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, [68719476741, 77309411338, 4294967312]]}(%arg2, %arg3, %arg4)
          : memref<1x16x180x16xf16, [@CMX_NN, 0]>, memref<1x16x1458x2xsi32, [@CMX_NN, 0]>, memref<1x16x1458x16xf16, [@CMX_NN, 0]>
    }

    // Note: 68719476741: 0x_10_00000005 (16, 5); 77309411338: 0x_12_0000000a (18, 10); 4294967312: 0x_1_00000010 (1, 16)
    // Original Shape: [1, 16, 18, 10, 16]

    return %results: memref<1x16x1458x16xf16, [@CMX_NN, 0]>

    // CHECK:   [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[DATA_0:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 0, 0] [1, 8, 180, 16] : memref<1x16x180x16xf16, [@CMX_NN, 0]> to memref<1x8x180x16xf16, {order = #NCHW, strides = [46080, 2880, 16, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_0:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 8, 1458, 2] : memref<1x16x1458x2xsi32, [@CMX_NN, 0]> to memref<1x8x1458x2xsi32, {order = #NCHW, strides = [46656, 2916, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_0:%.+]] = VPUIP.SubView [[ALLOC_OUT]] [0, 0, 0, 0] [1, 8, 1458, 16] : memref<1x16x1458x16xf16, [@CMX_NN, 0]> to memref<1x8x1458x16xf16, {order = #NCHW, strides = [373248, 23328, 16, 1]}, [@CMX_NN, 0]>

    // CHECK:   [[DATA_1:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 8, 0, 0] [1, 8, 180, 16] : memref<1x16x180x16xf16, [@CMX_NN, 0]> to memref<1x8x180x16xf16, {order = #NCHW, strides = [46080, 2880, 16, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_1:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 8, 0, 0] [1, 8, 1458, 2] : memref<1x16x1458x2xsi32, [@CMX_NN, 0]> to memref<1x8x1458x2xsi32, {order = #NCHW, strides = [46656, 2916, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_1:%.+]] = VPUIP.SubView [[ALLOC_OUT]] [0, 8, 0, 0] [1, 8, 1458, 16] : memref<1x16x1458x16xf16, [@CMX_NN, 0]> to memref<1x8x1458x16xf16, {order = #NCHW, strides = [373248, 23328, 16, 1]}, [@CMX_NN, 0]>

    // CHECK:   [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GatherND
    // CHECK-SAME:              inputs([[DATA_0]] as [[DATA_0_INNER:%.+]]: memref<1x8x180x16xf16, {order = #NCHW, strides = [46080, 2880, 16, 1]}, [@CMX_NN, 0]>, [[INDICES_0]] as [[INDICES_0_INNER:%.+]]: memref<1x8x1458x2xsi32, {order = #NCHW, strides = [46656, 2916, 2, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                     [[DATA_1]] as [[DATA_1_INNER:%.+]]: memref<1x8x180x16xf16, {order = #NCHW, strides = [46080, 2880, 16, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                     [[INDICES_1]] as [[INDICES_1_INNER:%.+]]: memref<1x8x1458x2xsi32, {order = #NCHW, strides = [46656, 2916, 2, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:              outputs([[OUT_0]] as [[OUT_0_INNER:%.+]]: memref<1x8x1458x16xf16, {order = #NCHW, strides = [373248, 23328, 16, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                     [[OUT_1]] as [[OUT_1_INNER:%.+]]: memref<1x8x1458x16xf16, {order = #NCHW, strides = [373248, 23328, 16, 1]}, [@CMX_NN, 0]>)
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [68719476741, 77309411338, 4294967304]]
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [68719476741, 77309411338, 4294967304]]

    // Note: 68719476741: 0x_10_00000005 (16, 5); 77309411338: 0x_12_0000000a (18, 10); 4294967304: 0x_1_00000008 (1, 8)
    // Original Shape: [1, 8, 18, 10, 16]

    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1 : memref<1x8x1458x16xf16, {order = #NCHW, strides = [373248, 23328, 16, 1]}, [@CMX_NN, 0]>, memref<1x8x1458x16xf16, {order = #NCHW, strides = [373248, 23328, 16, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                outputs([[ALLOC_OUT]] : memref<1x16x1458x16xf16, [@CMX_NN, 0]>) -> memref<1x16x1458x16xf16, [@CMX_NN, 0]>

    // CHECK:   return [[CONCAT]] : memref<1x16x1458x16xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
  func.func private @builtin_GatherND(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, none) attributes {VPU.kernel_code = "gatherND.cpp", VPU.kernel_entry = "gatherND"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileGatherNDOpAtH
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x1x180x16xf16, [@CMX_NN, 0]>
// CHECK-SAME:    [[INPUT_1:%.+]]: memref<1x1x1458x2xsi32, [@CMX_NN, 0]>
func.func @TileGatherNDOpAtH(%arg0: memref<1x1x180x16xf16, [@CMX_NN, 0]>, %arg1: memref<1x1x1458x2xsi32, [@CMX_NN, 0]>)
              -> memref<1x1x1458x16xf16, [@CMX_NN, 0]> {
    %alloc_out = memref.alloc() : memref<1x1x1458x16xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherND
          inputs(%arg0 as %arg2: memref<1x1x180x16xf16, [@CMX_NN, 0]>, %arg1 as %arg3: memref<1x1x1458x2xsi32, [@CMX_NN, 0]>)
          outputs(%alloc_out as %arg4: memref<1x1x1458x16xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1458x16xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, [68719476741, 77309411338, 4294967297]]}(%arg2, %arg3, %arg4)
          : memref<1x1x180x16xf16, [@CMX_NN, 0]>, memref<1x1x1458x2xsi32, [@CMX_NN, 0]>, memref<1x1x1458x16xf16, [@CMX_NN, 0]>
    }

    // Note: 68719476741: 0x_10_00000005 (16, 5); 77309411338: 0x_12_0000000a (18, 10); 4294967297: 0x_1_00000001 (1, 1)
    // Original Shape: [1, 1, 18, 10, 16]

    return %results: memref<1x1x1458x16xf16, [@CMX_NN, 0]>

    // CHECK:   [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x1x1458x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_0:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 1, 729, 2] : memref<1x1x1458x2xsi32, [@CMX_NN, 0]> to memref<1x1x729x2xsi32, {order = #NCHW, strides = [2916, 2916, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_0:%.+]] = VPUIP.SubView [[ALLOC_OUT]] [0, 0, 0, 0] [1, 1, 729, 16] : memref<1x1x1458x16xf16, [@CMX_NN, 0]> to memref<1x1x729x16xf16, {order = #NCHW, strides = [23328, 23328, 16, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_1:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 729, 0] [1, 1, 729, 2] : memref<1x1x1458x2xsi32, [@CMX_NN, 0]> to memref<1x1x729x2xsi32, {order = #NCHW, strides = [2916, 2916, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_1:%.+]] = VPUIP.SubView [[ALLOC_OUT]] [0, 0, 729, 0] [1, 1, 729, 16] : memref<1x1x1458x16xf16, [@CMX_NN, 0]> to memref<1x1x729x16xf16, {order = #NCHW, strides = [23328, 23328, 16, 1]}, [@CMX_NN, 0]>

    // CHECK:   [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GatherND
    // CHECK-SAME:              inputs([[INPUT_0]] as [[DATA_INNER:%.+]]: memref<1x1x180x16xf16, [@CMX_NN, 0]>, [[INDICES_0]] as [[INDICES_0_INNER:%.+]]: memref<1x1x729x2xsi32, {order = #NCHW, strides = [2916, 2916, 2, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                     [[INPUT_0]] as [[DATA_INNER:%.+]]: memref<1x1x180x16xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:                     [[INDICES_1]] as [[INDICES_1_INNER:%.+]]: memref<1x1x729x2xsi32, {order = #NCHW, strides = [2916, 2916, 2, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:              outputs([[OUT_0]] as [[ARG6:%.+]]: memref<1x1x729x16xf16, {order = #NCHW, strides = [23328, 23328, 16, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                      [[OUT_1]] as [[ARG7:%.+]]: memref<1x1x729x16xf16, {order = #NCHW, strides = [23328, 23328, 16, 1]}, [@CMX_NN, 0]>)
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [68719476741, 77309411338, 4294967297]]
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [68719476741, 77309411338, 4294967297]]

    // Note: 68719476741: 0x_10_00000005 (16, 5); 77309411338: 0x_12_0000000a (18, 10); 4294967297: 0x_1_00000001 (1, 1)
    // Original Shape: [1, 1, 18, 10, 16]

    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1 : memref<1x1x729x16xf16, {order = #NCHW, strides = [23328, 23328, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x729x16xf16, {order = #NCHW, strides = [23328, 23328, 16, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                outputs([[ALLOC_OUT]] : memref<1x1x1458x16xf16, [@CMX_NN, 0]>) -> memref<1x1x1458x16xf16, [@CMX_NN, 0]>

    // CHECK:   return [[CONCAT]] : memref<1x1x1458x16xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Clamp(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "activation_clamp.cpp", VPU.kernel_entry = "activation_clamp", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @SWKernelTilingNotApplied
// CHECK-SAME: [[INPUT:%.+]]: memref<1x527x2x1xf16>, [[OUTPUT:%.+]]: memref<1x527x2x1xf16>
func.func @SWKernelTilingNotApplied(%arg0: memref<1x527x2x1xf16>, %arg1: memref<1x527x2x1xf16>) -> memref<1x527x2x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x527x2x1xf16>) outputs(%0 : !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x527x2x1xf16>) outputs(%2 : !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %results_0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Clamp inputs(%1 as %arg2: !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%3 as %arg3: !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
    VPUIP.SW.Kernel.run {attrs = [1.1920928955078125E-7, 1.000000e+00]}(%arg2, %arg3) : !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }
    %alloc = memref.alloc() : memref<1x527x2x1xf16>
    %4 = VPUIP.Copy inputs(%results_0 : !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%alloc : memref<1x527x2x1xf16>) -> memref<1x527x2x1xf16>
    return %4 : memref<1x527x2x1xf16>

    // CHECK:   [[IN_BUFFER:%.+]] = VPUIP.Copy inputs([[INPUT]]
    // CHECK:   [[OUT_BUFFER:%.+]] = VPUIP.Copy inputs([[OUTPUT]]
    
    // CHECK:       [[RESULTS:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Clamp inputs([[IN_BUFFER]] as [[KERNEL_INPUT:%.+]]: !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUFFER]] as [[KERNEL_OUTPUT:%.+]]: !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [1.1920928955078125E-7, 1.000000e+00]}([[KERNEL_INPUT]], [[KERNEL_OUTPUT]]) : !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x527x2x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[DDR_RESULT:%.+]] = memref.alloc() : memref<1x527x2x1xf16>

    // CHECK:       [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[RESULTS]]
    // CHECK-SAME:      outputs([[DDR_RESULT]]

    // CHECK:       return [[COPY_OUT]] : memref<1x527x2x1xf16>
}
