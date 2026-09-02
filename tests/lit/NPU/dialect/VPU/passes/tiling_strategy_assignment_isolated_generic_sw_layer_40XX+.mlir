//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK: module @IsolatedGenericSwLayerNCHW
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x80x960xf16>, [[ARG1:%.+]]: tensor<1x8x1x1xf16>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 1, 2]
module @IsolatedGenericSwLayerNCHW {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5, %c1, %arg0, %c1, %c1, %c0, %arg3, %c0, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x?x1x1xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x8x80x960xf16>
    DataInfo "input1" : tensor<1x8x1x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x80x960xf16>
  }
  func.func @main(%arg0: tensor<1x8x80x960xf16>, %arg1: tensor<1x8x1x1xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x8x80x960xf16>, tensor<1x8x1x1xf16>) @VPU.SW::@generated_0 tiling(sizes = [8, 80, 960], offsets = [0, 0, 0]) -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @IsolatedGenericSwLayerNHWC
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x80x960xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 1, 2]
module @IsolatedGenericSwLayerNHWC {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5, %c1, %c1, %c1, %arg2, %c0, %c0, %c0, %arg5 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x8x80x960xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x8x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x80x960xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x8x80x960xf16, {order = #NHWC}>, %arg1: tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<1x8x80x960xf16, {order = #NHWC}> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x8x80x960xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [80, 960, 8], offsets = [0, 0, 0]) -> tensor<1x8x80x960xf16, {order = #NHWC}>
    return %0 : tensor<1x8x80x960xf16, {order = #NHWC}>
  }
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: module @IsolatedGenericSwLayerNWCH
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x80x960xf16, {order = #NWCH}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NWCH}>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 1, 2]
module @IsolatedGenericSwLayerNWCH {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5, %c1, %c1, %arg1, %c1, %c0, %c0, %arg4, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x1x?x1xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x8x80x960xf16, {order = #NWCH}>
    DataInfo "input1" : tensor<1x8x1x1xf16, {order = #NWCH}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x80x960xf16, {order = #NWCH}>
  }
  func.func @main(%arg0: tensor<1x8x80x960xf16, {order = #NWCH}>, %arg1: tensor<1x8x1x1xf16, {order = #NWCH}>) -> tensor<1x8x80x960xf16, {order = #NWCH}> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x8x80x960xf16, {order = #NWCH}>, tensor<1x8x1x1xf16, {order = #NWCH}>) @VPU.SW::@generated_0 tiling(sizes = [960, 8, 80], offsets = [0, 0, 0]) -> tensor<1x8x80x960xf16, {order = #NWCH}>
    return %0 : tensor<1x8x80x960xf16, {order = #NWCH}>
  }
}

// -----

// CHECK: module @IsolatedGenericSwLayerNCHW
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x960x80xf16>, [[ARG1:%.+]]: tensor<1x8x1x1xf16>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
module @IsolatedGenericSwLayerNCHW {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5, %c1, %arg0, %c1, %c1, %c0, %arg3, %c0, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x?x1x1xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x8x960x80xf16>
    DataInfo "input1" : tensor<1x8x1x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x960x80xf16>
  }
  func.func @main(%arg0: tensor<1x8x960x80xf16>, %arg1: tensor<1x8x1x1xf16>) -> tensor<1x8x960x80xf16> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x8x960x80xf16>, tensor<1x8x1x1xf16>) @VPU.SW::@generated_0 tiling(sizes = [8, 960, 80], offsets = [0, 0, 0]) -> tensor<1x8x960x80xf16>
    return %0 : tensor<1x8x960x80xf16>
  }
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @IsolatedGenericSwLayerNHWC
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x960x80xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
module @IsolatedGenericSwLayerNHWC {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5, %c1, %c1, %c1, %arg2, %c0, %c0, %c0, %arg5 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x8x960x80xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x8x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x960x80xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x8x960x80xf16, {order = #NHWC}>, %arg1: tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<1x8x960x80xf16, {order = #NHWC}> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x8x960x80xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [960, 80, 8], offsets = [0, 0, 0]) -> tensor<1x8x960x80xf16, {order = #NHWC}>
    return %0 : tensor<1x8x960x80xf16, {order = #NHWC}>
  }
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: module @IsolatedGenericSwLayerNWCH
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x960x80xf16, {order = #NWCH}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NWCH}>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
module @IsolatedGenericSwLayerNWCH {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5, %c1, %c1, %arg1, %c1, %c0, %c0, %arg4, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x1x?x1xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x8x960x80xf16, {order = #NWCH}>
    DataInfo "input1" : tensor<1x8x1x1xf16, {order = #NWCH}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x960x80xf16, {order = #NWCH}>
  }
  func.func @main(%arg0: tensor<1x8x960x80xf16, {order = #NWCH}>, %arg1: tensor<1x8x1x1xf16, {order = #NWCH}>) -> tensor<1x8x960x80xf16, {order = #NWCH}> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x8x960x80xf16, {order = #NWCH}>, tensor<1x8x1x1xf16, {order = #NWCH}>) @VPU.SW::@generated_0 tiling(sizes = [80, 8, 960], offsets = [0, 0, 0]) -> tensor<1x8x960x80xf16, {order = #NWCH}>
    return %0 : tensor<1x8x960x80xf16, {order = #NWCH}>
  }
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @IsolatedGenericSwLayerNHWCLargeC
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x640x1x960xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 1, 1, 2]
module @IsolatedGenericSwLayerNHWCLargeC {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x640x1x960xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x640x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x640x1x960xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1: tensor<1x640x1x1xf16, {order = #NHWC}>) -> tensor<1x640x1x960xf16, {order = #NHWC}> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) -> tensor<1x640x1x960xf16, {order = #NHWC}>
    return %0 : tensor<1x640x1x960xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @IsolatedGenericSwLayerNHWCLargeH
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x960x1x640xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x960x1x1xf16, {order = #NHWC}>)
// CHECK: VPU.GenericSwLayer([[ARG0]], [[ARG1]]
// CHECK-SAME: tilingStrategy = [1, 2, 1, 1]
module @IsolatedGenericSwLayerNHWCLargeH {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
      >
    }
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x960x1x640xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x960x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x960x1x640xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1: tensor<1x960x1x1xf16, {order = #NHWC}>) -> tensor<1x960x1x640xf16, {order = #NHWC}> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x960x1x640xf16, {order = #NHWC}>, tensor<1x960x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [640, 960], offsets = [0, 0]) -> tensor<1x960x1x640xf16, {order = #NHWC}>
    return %0 : tensor<1x960x1x640xf16, {order = #NHWC}>
  }
}
