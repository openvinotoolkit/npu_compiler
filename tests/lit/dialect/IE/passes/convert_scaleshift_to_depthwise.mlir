// RUN: vpux-opt --split-input-file --convert-scale-shift-depthwise %s | FileCheck %s

// CHECK-LABEL: @ConvertScaleShiftToDepthwise
func @ConvertScaleShiftToDepthwise(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = #const.Content<dense<-1.000000e+00> : tensor<1x3x1x1xf16>>
    %bias = const.Declare tensor<1x3x1x1xf16> = #const.Content<dense<7.843020e-03> : tensor<1x3x1x1xf16>>
    %0 = IE.ScaleShift(%arg0, %weights, %bias) {operand_segment_sizes = dense<1> : vector<3xi32>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %0 : tensor<1x3x224x224xf16>

    // CHECK-NOT:   IE.ScaleShift
    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<3x1x1x1xf16> = #const.Content<dense<-1.000000e+00> : tensor<1x3x1x1xf16>, [#const.Reshape<[3, 1, 1, 1]>]>
    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = #const.Content<dense<7.843020e-03> : tensor<1x3x1x1xf16>>
    // CHECK:       %[[GROUPCONV:.*]] = IE.GroupConvolution(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 3 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[GROUPCONV]] 
}
