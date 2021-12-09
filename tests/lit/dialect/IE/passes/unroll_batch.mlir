// RUN: vpux-opt --split-input-file --unroll-batch %s | FileCheck %s

// CHECK-LABEL: @UnrollFullyConnectedBatch
func @UnrollFullyConnectedBatch(%arg0: tensor<2x16xf32>) -> tensor<2x64xf32> {
    %cst = const.Declare tensor<64x16xf16> = #const.Content<dense<1.0> : tensor<64x16xf32>, [#const.ConvertElemType<f16>]>
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    %1 = IE.FullyConnected(%0, %cst) : tensor<2x16xf16>, tensor<64x16xf16> -> tensor<2x64xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>

    return %2 : tensor<2x64xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf16> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16xf32>, [#const.ConvertElemType<f16>]>
    // CHECK:       %[[INPUT:.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    // CHECK:       %[[INPUT_SLICE_1:.*]] = IE.Slice %[[INPUT]] [0, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_1:.*]] = IE.FullyConnected(%[[INPUT_SLICE_1]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[INPUT_SLICE_2:.*]] = IE.Slice %[[INPUT]] [1, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_2:.*]] = IE.FullyConnected(%[[INPUT_SLICE_2]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[FC_CONCAT:.*]] = IE.Concat(%[[FC_1]], %[[FC_2]])
    // CHECK-SAME:      {per_axis = {axis = 0 : i64}} : tensor<1x64xf16>, tensor<1x64xf16> -> tensor<2x64xf16>
    // CHECK:       %[[OUT:.*]] = IE.Convert(%[[FC_CONCAT]]) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>
    // CHECK:       return %[[OUT]] : tensor<2x64xf32>
}

!qElemType = type !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @UnrollEltwiseAndBatch
func @UnrollEltwiseAndBatch(%arg0: tensor<2x128x40x8xf16>) -> tensor<2x128x40x8xf16> {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<2x128x40x8xf16>, tensor<2x128x40x8xf16> -> tensor<2x128x40x8x!qElemType>
    %1 = IE.And(%0, %0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<2x128x40x8x!qElemType>, tensor<2x128x40x8x!qElemType> -> tensor<2x128x40x8xf16>
    return %1 : tensor<2x128x40x8xf16>

    // CHECK: %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: %2 = IE.And(%0, %1) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<1x128x40x8x!qElemType>
    // CHECK: %3 = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: %4 = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: %5 = IE.And(%3, %4) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<1x128x40x8x!qElemType>
    // CHECK: %6 = IE.Concat(%2, %5) {per_axis = {axis = 0 : i64}} : tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<2x128x40x8x!qElemType>
    // CHECK: %7 = IE.Slice %6 [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: %8 = IE.Slice %6 [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: %9 = IE.And(%7, %8) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<1x128x40x8xf16>
    // CHECK: %10 = IE.Slice %6 [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: %11 = IE.Slice %6 [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: %12 = IE.And(%10, %11) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<1x128x40x8xf16>
    // CHECK: %13 = IE.Concat(%9, %12) {per_axis = {axis = 0 : i64}} : tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<2x128x40x8xf16>
    // CHECK: return %13 : tensor<2x128x40x8xf16>
}
