//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --unpack-nested-modules="mode=entry-point" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @mainModuleWithNestedModule
module @mainModuleWithNestedModule {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : false
    config.Option @config.AutoPaddingIDU : false
  }
  config.Resources 1 of @global {
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
  }
  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = Core.NestedCall @NPUModule::@nestedModule::@nestedFunc(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }
  module @NPUModule {
    config.PipelineOptions @Options {
      config.Option @config.AutoPaddingODU : false
      config.Option @config.AutoPaddingIDU : false
    }
    config.Resources 1 of @global {
      config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
      config.ExecutorResource 1 of @M2I
      config.ExecutorResource 2 of @DMA_NN
    }
    net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input" : tensor<1x3x60x60xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x60x60xf16>
    }
    module @nestedModule {
      func.func nested @nestedFunc(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
      %callee = Core.NestedCall @nestedModule::@nestedFunc(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
      return %callee : tensor<1x3x60x60xf16>
    }
  }

  // CHECK:       net.NetworkInfo entryPoint : @main inputsInfo
  // CHECK:       func.func @func0
  // CHECK:         Core.NestedCall @nestedModule::@nestedFunc

  // CHECK-NOT:   module @NPUModule

  // CHECK:       module @nestedModule {
  // CHECK:         func.func nested @nestedFunc
  // CHECK:       }

  // CHECK:       func.func @main
  // CHECK:         Core.NestedCall @nestedModule::@nestedFunc
}

// -----

// CHECK-LABEL: @mainModuleWithOtherFunc
module @mainModuleWithOtherFunc {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : false
    config.Option @config.AutoPaddingIDU : false
  }
  config.Resources 1 of @global {
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
  }
  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = Core.NestedCall @NPUModule::@func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }
  module @NPUModule {
    config.PipelineOptions @Options {
      config.Option @config.AutoPaddingODU : false
      config.Option @config.AutoPaddingIDU : false
    }
    config.Resources 1 of @global {
      config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
      config.ExecutorResource 1 of @M2I
      config.ExecutorResource 2 of @DMA_NN
    }
    net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input" : tensor<1x3x60x60xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x60x60xf16>
    }
    func.func nested @func1(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
      %callee = call @func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
      return %callee : tensor<1x3x60x60xf16>
    }
  }

  // CHECK:       net.NetworkInfo entryPoint : @main inputsInfo
  // CHECK:       func.func @func0
  // CHECK:         call @func1

  // CHECK-NOT:   module @NPUModule

  // CHECK:       func.func nested @func1

  // CHECK:       func.func @main
  // CHECK:         call @func1
}

// -----

// CHECK-LABEL: @mainModuleDoubleNesting
module @mainModuleDoubleNesting {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : false
    config.Option @config.AutoPaddingIDU : false
  }
  config.Resources 1 of @global {
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
  }
  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = Core.NestedCall @NPUModule::@nestedModule1::@nestedModule2::@nestedFunc(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }
  module @NPUModule {
    config.PipelineOptions @Options {
      config.Option @config.AutoPaddingODU : false
      config.Option @config.AutoPaddingIDU : false
    }
    config.Resources 1 of @global {
      config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
      config.ExecutorResource 1 of @M2I
      config.ExecutorResource 2 of @DMA_NN
    }
    net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input" : tensor<1x3x60x60xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x60x60xf16>
    }
    module @nestedModule1 {
      module @nestedModule2 {
        func.func nested @nestedFunc(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
      }
    }
    func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
      %callee = Core.NestedCall @nestedModule1::@nestedModule2::@nestedFunc(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
      return %callee : tensor<1x3x60x60xf16>
    }
  }

  // CHECK:       net.NetworkInfo entryPoint : @main inputsInfo
  // CHECK:       func.func @func0
  // CHECK:         Core.NestedCall @nestedModule1::@nestedModule2::@nestedFunc

  // CHECK-NOT:   module @NPUModule

  // CHECK:       module @nestedModule1 {
  // CHECK:         module @nestedModule2 {
  // CHECK:           func.func nested @nestedFunc
  // CHECK:         }
  // CHECK:       }

  // CHECK:       func.func @main
  // CHECK:         Core.NestedCall @nestedModule1::@nestedModule2::@nestedFunc
}
