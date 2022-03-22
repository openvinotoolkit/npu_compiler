// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s
// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = #const.Content<dense<[20, 15]> : tensor<2xsi64>>
    %1 = const.Declare tensor<2xf32>  = #const.Content<dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>>
    %2 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>
    // CHECK-NOT:   const.Declare
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {attr = {antialias = false, coord_mode = "half_pixel", cube_coeff = -7.500000e-01, mode = "nearest", nearest_mode = "round_prefer_floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"}, operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "half_pixel", cube_coeff = -7.500000e-01 : f64, mode = "nearest", nearest_mode = "round_prefer_floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"},
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %3 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// CHECK-LABEL: @OvPreprocessingInterpolateU8
func @OvPreprocessingInterpolateU8(%arg0: tensor<1x10x30x30xui8>) -> tensor<1x10x40x40xui8> {
    %cst = const.Declare tensor<2xsi64> = #const.Content<dense<40> : tensor<2xsi64>>
    %cst_0 = const.Declare tensor<2xf32> = #const.Content<dense<1.333330e+00> : tensor<2xf32>>
    %cst_1 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = {antialias = false, coord_mode = "half_pixel", cube_coeff = -7.500000e-01 : f64, mode = "nearest", nearest_mode = "round_prefer_floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"}, operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x10x30x30xui8>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x10x40x40xui8>

    return %0 : tensor<1x10x40x40xui8>
    // CHECK:       [[VAL0:%.*]] = IE.Convert
    // CHECK-SAME:      dstElemType = f16
    // CHECK:       [[VAL1:%.*]] = IE.Interpolate([[VAL0]]
    // CHECK:       IE.Convert([[VAL1]]
    // CHECK-SAME:      dstElemType = ui8
}
