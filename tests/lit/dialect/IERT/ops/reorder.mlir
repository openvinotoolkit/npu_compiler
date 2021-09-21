// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

func @FoldReorder(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = memref.alloc() : memref<1x8x4x2xf16>
    %1 = IERT.Reorder inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%0 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %2 = memref.alloc() : memref<1x8x4x2xf16>
    %3 = IERT.SoftMax {axisInd = 1} inputs(%1 : memref<1x8x4x2xf16>) outputs(%2 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %4 = memref.alloc() : memref<1x8x4x2xf16>
    %5 = IERT.Reorder inputs(%3 : memref<1x8x4x2xf16>) outputs(%4 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %6 = memref.alloc() : memref<1x8x4x2xf16>
    %7 = IERT.SoftMax {axisInd = 1} inputs(%5 : memref<1x8x4x2xf16>) outputs(%6 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %8 = IERT.Reorder inputs(%7 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %8 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR3:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs([[VAR2]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: return [[VAR4]] : memref<1x8x4x2xf16>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @FuseReorders {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : tensor<1x8x4x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : tensor<1x8x4x2xf16, {order = #map0}>
    }

func @main(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16, #map0>) -> memref<1x8x4x2xf16, #map0> {
    %0 = memref.alloc() : memref<1x8x4x2xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%0 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %2 = memref.alloc() : memref<1x8x4x2xf16, #map1>
    %3 = IERT.Reorder inputs(%1 : memref<1x8x4x2xf16>) outputs(%2 : memref<1x8x4x2xf16, #map1>) -> memref<1x8x4x2xf16, #map1>
    %4 = IERT.Reorder inputs(%3 : memref<1x8x4x2xf16, #map1>) outputs(%arg1 : memref<1x8x4x2xf16, #map0>) -> memref<1x8x4x2xf16, #map0>

    return %4 : memref<1x8x4x2xf16, #map0>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.Reorder inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16, #map>) -> memref<1x8x4x2xf16, #map>
    // CHECK: return [[VAR2]] : memref<1x8x4x2xf16, #map>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 1000 + d1 * 1000 + d2 * 1000 + d3)>

func @ConvertTrivialReorder(%arg0: memref<1x1000x1x1xf16, #NHWC, #map>) -> memref<1x1000x1x1xf16> {
    %0 = memref.alloc() : memref<1x1000x1x1xf16>
    %1 = IERT.Reorder inputs(%arg0 : memref<1x1000x1x1xf16, #NHWC, #map>) outputs(%0 : memref<1x1000x1x1xf16>) -> memref<1x1000x1x1xf16>

    // CHECK: [[REORDER:%.*]] = IERT.ImplicitReorder
    // CHECK-SAME:  {dstOrder = #NCHW}
    // CHECK-SAME:  inputs(%arg0 : memref<1x1000x1x1xf16, #NHWC, #map>)
    // CHECK-SAME:  -> memref<1x1000x1x1xf16>

    return %1 : memref<1x1000x1x1xf16>

    // CHECK: return [[REORDER]] : memref<1x1000x1x1xf16>
}
