//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-adaptive-stripping=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="quant-dequant-removal=true fuse-outstanding-dequant=true fuse-outstanding-quant=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @MatMulScaleShiftedU16FQ
module @MatMulScaleShiftedU16FQ {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x64x64xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x64x4096xf32>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x64x64xf32>
    func.func @main(%arg0: tensor<1x64x64x64xf32>) -> tensor<1x64x64x4096xf32> {
        %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
        %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
        %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-15.0> : tensor<1x1x1x1xf32>
        %output_high = const.Declare tensor<1x1x1x1xf32> = dense<25.0> : tensor<1x1x1x1xf32>

        %weight = const.Declare tensor<1x1x4096x64xf32> = dense<1.0> : tensor<1x1x4096x64xf32>

        %matmul = IE.MatMul(%arg0, %weight) {
            transpose_b
        } : tensor<1x64x64x64xf32>, tensor<1x1x4096x64xf32> -> tensor<1x64x64x4096xf32>
        %fq = IE.FakeQuantize(%matmul, %input_low, %input_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 65536 : i64
        } : tensor<1x64x64x4096xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x64x4096xf32>

        return %fq : tensor<1x64x64x4096xf32>

        // CHECK:       [[BIAS:%.+]] = const.Declare tensor<1x4096x1x1xf16> = dense<5.000000e+00>
        // CHECK-SAME:      #const.Rescale<5.000000e-01 : f64>
        // CHECK:       IE.Convolution([[INPUT:%.+]], [[WEIGHT:%.+]], [[BIAS]])
        // CHECK-SAME:      static_scale = 2.000000e+00 : f32

        // CHECK-NOT:   IE.FakeQuantize
        // CHECK-NOT:   IE.Multiply
        // CHECK-NOT:   IE.Add
        // CHECK-NOT:   IE.ScaleShift
    }
}

// -----

// CHECK-LABEL: @MultiplyFQFusion
module @MultiplyFQFusion {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x256x64xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x256x64xf32>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x256x64xf32>
    func.func @main(%arg0: tensor<1x64x256x64xf32>) -> tensor<1x64x256x64xf32> {
        %low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
        %high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>

        %bias = const.Declare tensor<1x64x256x64xf32> = dense<2.0> : tensor<1x64x256x64xf32>
        %biasfq = IE.FakeQuantize(%bias, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x256x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x256x64xf32>
        %scale = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
        %scalefq = IE.FakeQuantize(%scale, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

        %add1 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x256x64xf32>, tensor<1x64x256x64xf32> -> tensor<1x64x256x64xf32>
        %add1fq = IE.FakeQuantize(%add1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x256x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x256x64xf32>
        %mul = IE.Multiply(%add1fq, %scalefq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x256x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x256x64xf32>
        %mulfq = IE.FakeQuantize(%mul, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x256x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x256x64xf32>
        %add2 = IE.Add(%mulfq, %biasfq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x256x64xf32>, tensor<1x64x256x64xf32> -> tensor<1x64x256x64xf32>

        return %add2 : tensor<1x64x256x64xf32>

        // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x256x64x{{[^:]+}}, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[ARG0]])
        // CHECK-NEXT:  [[PERMUTE_QUANT:%.+]] = IE.PermuteQuantize([[CONVERT1]])

        // CHECK-NEXT:  [[ADD1:%.+]] = IE.Add([[PERMUTE_QUANT]], [[PERMUTE_QUANT]])
        // CHECK-NEXT:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[ADD1]])
        // CHECK-NOT:   IE.AvgPool
        // CHECK-NOT:   IE.GroupConvolution
        // CHECK-NEXT:  [[ADD2:%.+]] = IE.Add([[QUANT_CAST]], [[BIAS]])
    }
}
