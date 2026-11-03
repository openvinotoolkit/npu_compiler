//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

module {
  net.NetworkInfo entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x3x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x3x16x16xf16>) -> (tensor<1x3x16x16xf16>) {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<205.0> : tensor<1x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
      { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i32 } :
      tensor<1x3x16x16xf16>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x16x16xf16>

    return %fq : tensor<1x3x16x16xf16>
  }
}
