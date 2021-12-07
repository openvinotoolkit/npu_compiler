// RUN: vpux-opt --split-input-file --optimize-copies %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @OptimizeCopy(
        %arg0: memref<1x16x112x112xf16, #NHWC, "CMX_NN">,
        %arg1: memref<1x16x112x112xf16, #NHWC, "CMX_NN">,
        %arg2: memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC>
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC>

    %2 = IERT.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, "CMX_NN">)
        outputs(%1 : memref<1x16x112x112xf16, #NHWC>)
        -> memref<1x16x112x112xf16, #NHWC>

    %3 = IERT.SubView %0 [0, 0, 0, 0] [1, 16, 112, 112] :
        memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    %4 = IERT.Copy inputs(%2 : memref<1x16x112x112xf16, #NHWC>)
        outputs(%3 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)
        -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>

    %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC>

    %6 = IERT.Copy inputs(%arg1 : memref<1x16x112x112xf16, #NHWC, "CMX_NN">)
        outputs(%5 : memref<1x16x112x112xf16, #NHWC>)
        -> memref<1x16x112x112xf16, #NHWC>

    %7 = IERT.SubView %0 [0, 16, 0, 0] [1, 16, 112, 112] :
        memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    %8 = IERT.Copy inputs(%6 : memref<1x16x112x112xf16, #NHWC>)
        outputs(%7 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)
        -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>

    %9 = IERT.ConcatView
        inputs(%4, %8 :
            memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>,
            memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        )
        outputs(%0 : memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC>

    %10 = IERT.Copy inputs(%9 : memref<1x32x112x112xf16, #NHWC>)
        outputs(%arg2 : memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC>

    return %10 : memref<1x32x112x112xf16, #NHWC>

    // CHECK-NOT:   memref.alloc() : memref<1x32x112x112xf16, #NHWC>

    // CHECK-NOT:   memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    // CHECK:       [[VAR0:%.*]] = IERT.SubView %arg2 [0, 0, 0, 0] [1, 16, 112, 112] :
    // CHECK-SAME:      memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       [[VAR1:%.*]] = IERT.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, "CMX_NN">)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK-NOT:   memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    // CHECK:       [[VAR2:%.*]] = IERT.SubView %arg2 [0, 16, 0, 0] [1, 16, 112, 112] :
    // CHECK-SAME:      memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       [[VAR3:%.*]] = IERT.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, "CMX_NN">)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK:       [[VAR4:%.*]] = IERT.ConcatView inputs([[VAR1]], [[VAR3]] :
    // CHECK-SAME:      outputs(%arg2 : memref<1x32x112x112xf16, #NHWC>)

    // CHECK: return [[VAR4]] : memref<1x32x112x112xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @OptimizeLastCopyForPureViewOps(%arg0: memref<1x16x2x2xf16>, %arg1: memref<1x16x2x2xf16>, %arg2: memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16> {
    %0 = memref.alloc() : memref<1x64x2x2xf16>

    %1 = IERT.SubView %0 [0, 0, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %2 = IERT.Copy inputs(%arg0 : memref<1x16x2x2xf16>) outputs(%1 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %3 = IERT.SubView %0 [0, 16, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %4 = IERT.Copy inputs(%arg1 : memref<1x16x2x2xf16>) outputs(%3 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    %5 = IERT.SubView %0 [0, 0, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
    %6 = IERT.ConcatView inputs(%2, %4 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) outputs(%5 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>

    %7 = IERT.SubView %0 [0, 32, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %8 = IERT.Copy inputs(%arg0 : memref<1x16x2x2xf16>) outputs(%7 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %9 = IERT.SubView %0 [0, 48, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %10 = IERT.Copy inputs(%arg1 : memref<1x16x2x2xf16>) outputs(%9 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    %11 = IERT.SubView %0 [0, 32, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to
            memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
    %12 = IERT.ConcatView inputs(%8, %10 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
            outputs(%11 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>

    %13 = IERT.ConcatView inputs(%6, %12 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>)
            outputs(%0 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
    %14 = IERT.Copy inputs(%13 : memref<1x64x2x2xf16>) outputs(%arg2 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>

    return %14 : memref<1x64x2x2xf16>

    // CHECK-NOT: memref.alloc() : memref<1x64x2x2xf16>

    // CHECK: [[VAR0:%.*]] = IERT.SubView %arg2 [0, 0, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR1:%.*]] = IERT.Copy inputs(%arg0 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK: [[VAR2:%.*]] = IERT.SubView %arg2 [0, 16, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR3:%.*]] = IERT.Copy inputs(%arg1 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK: [[VAR4:%.*]] = IERT.SubView %arg2 [0, 0, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR5:%.*]] = IERT.ConcatView inputs([[VAR1]], [[VAR3]] :
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    // CHECK: [[VAR6:%.*]] = IERT.SubView %arg2 [0, 32, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR7:%.*]] = IERT.Copy inputs(%arg0 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR6]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK: [[VAR8:%.*]] = IERT.SubView %arg2 [0, 48, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR9:%.*]] = IERT.Copy inputs(%arg1 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR8]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK: [[VAR10:%.*]] = IERT.SubView %arg2 [0, 32, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR11:%.*]] = IERT.ConcatView inputs([[VAR7]], [[VAR9]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK-SAME:      outputs([[VAR10]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK: [[VAR12:%.*]] = IERT.ConcatView inputs([[VAR5]], [[VAR11]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) outputs(%arg2 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
    // CHECK-NOT: IERT.Copy

    // CHECK: return [[VAR12]] : memref<1x64x2x2xf16>
}

// -----

func @OptimizeLastCopy(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>,
                        %arg2: memref<1x2x4x4xf16>, %arg3: memref<1x2x4x4xf16>)
                            -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = const.Declare memref<1x2x4x4xf16> = #const.Content<dense<1.000000e+00> : tensor<1x2x4x4xf16>>
    %1 = memref.alloc() : memref<1x2x4x4xf16>
    %2 = memref.alloc() : memref<1x2x4x4xf16>

    %3 = IERT.And
            inputs(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>)
            outputs(%1 : memref<1x2x4x4xf16>)
            -> memref<1x2x4x4xf16>
    %4 = IERT.And
            inputs(%arg0: memref<1x2x4x4xf16>, %0: memref<1x2x4x4xf16>)
            outputs(%2 : memref<1x2x4x4xf16>)
            -> memref<1x2x4x4xf16>

    %5 = IERT.Copy inputs(%3 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    %6 = IERT.Copy inputs(%4 : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %5, %6 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK: [[VAR0:%.*]] = const.Declare

    // CHECK-NOT: memref.alloc() : memref<1x2x4x4xf16>
    // CHECK-NOT: memref.alloc() : memref<1x2x4x4xf16>

    // CHECK: [[VAR1:%.*]] = IERT.And inputs(%arg0 : memref<1x2x4x4xf16>, %arg1 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    // CHECK: [[VAR2:%.*]] = IERT.And inputs(%arg0 : memref<1x2x4x4xf16>, [[VAR0]] : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    // CHECK: return [[VAR1]], [[VAR2]] : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 50 + d1 * 50 + d2 * 50 + d3)>

func @NoChangesTypeMismatch(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50x1x1xf16, #NHWC, #map>) -> memref<1x50x1x1xf16, #NHWC, #map> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = IERT.Sigmoid inputs(%arg0 : memref<1x50x1x1xf16>) outputs(%0 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16>
    %2 = IERT.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16, #NHWC, #map>
    %3 = IERT.Copy inputs(%2 : memref<1x50x1x1xf16, #NHWC, #map>) outputs(%arg1 : memref<1x50x1x1xf16, #NHWC, #map>) -> memref<1x50x1x1xf16, #NHWC, #map>
    return %3 : memref<1x50x1x1xf16, #NHWC, #map>

    // CHECK: memref.alloc()
    // CHECK: IERT.Sigmoid
    // CHECK: IERT.PermuteCast
    // CHECK: [[VAR0:%.*]] = IERT.Copy
    // CHECK: return [[VAR0]]
}


// -----

func @NoChangesSourceIsConstantOp(%arg0: memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16> {
    %0 = const.Declare memref<1x2x4x4xf16> = #const.Content<dense<1.000000e+00> : tensor<1x2x4x4xf16>>
    %1 = IERT.Copy inputs(%0 : memref<1x2x4x4xf16>) outputs(%arg0 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    return %1 : memref<1x2x4x4xf16>

    // CHECK: [[VAR0:%.*]] = const.Declare
    // CHECK: [[VAR1:%.*]] = IERT.Copy
    // CHECK: return [[VAR1]]
}

// -----

func @NoChangesInputIsBlockArgument(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>,
                                    %arg2: memref<1x2x4x4xf16>, %arg3: memref<1x2x4x4xf16>) ->
                                    (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = IERT.Copy inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    %1 = IERT.ReLU
            inputs(%arg0: memref<1x2x4x4xf16>)
            outputs(%arg2 : memref<1x2x4x4xf16>)
            -> memref<1x2x4x4xf16>
    %2 = IERT.Copy inputs(%1 : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %0, %1, %2 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK: [[VAR0:%.*]] = IERT.Copy
    // CHECK: [[VAR1:%.*]] = IERT.ReLU
    // CHECK: [[VAR2:%.*]] = IERT.Copy
    // CHECK: return [[VAR0]], [[VAR1]], [[VAR2]]
}

// -----

func @NoChangesDifferentMemSpace(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16> {
    %0 = memref.alloc() : memref<1x2x4x4xf16, "CMX_NN">
    %1 = IERT.Copy inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%0 : memref<1x2x4x4xf16, "CMX_NN">) -> memref<1x2x4x4xf16, "CMX_NN">

    %2 = memref.alloc() : memref<1x2x4x4xf16, "CMX_NN">
    %3 = IERT.And
            inputs(%1 : memref<1x2x4x4xf16, "CMX_NN">, %1 : memref<1x2x4x4xf16, "CMX_NN">)
            outputs(%2 : memref<1x2x4x4xf16, "CMX_NN">)
            -> memref<1x2x4x4xf16, "CMX_NN">

    %4 = IERT.Copy inputs(%3 : memref<1x2x4x4xf16, "CMX_NN">) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    return %4 : memref<1x2x4x4xf16>

    // CHECK: IERT.Copy

    // CHECK: [[VAR0:%.*]] = IERT.And
    // CHECK: [[VAR1:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x2x4x4xf16, "CMX_NN">)
    // CHECK-SAME:                     outputs(%arg1 : memref<1x2x4x4xf16>)

    // CHECK: return [[VAR1]]
}
