//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --pack-nested-modules="mode=entry-point" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @mainModule
module @mainModule {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }

  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    return %arg0 : tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    return %arg0 : tensor<1x3x60x60xf16>
  }

  // CHECK:       func.func @func0

  // CHECK:       module @NPUModule {{.+}} {
  // CHECK:         config.PipelineOptions @Options {
  // CHECK:         config.Resources {{.+}} of @NCE at {{.+}} MHz
  // CHECK:         config.Resources {{.+}} of @global
  // CHECK:         net.NetworkInfo entryPoint : @main

  // CHECK:         func.func @main
  // CHECK:       }
}

// -----

// CHECK-LABEL: @mainModuleCalleeFuncs
module @mainModuleCalleeFuncs {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }

  func.func nested @func0(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = call @func0(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }

  // CHECK:       module @NPUModule {{.+}} {
  // CHECK:         config.PipelineOptions @Options {
  // CHECK:         config.Resources {{.+}} of @NCE at {{.+}} MHz
  // CHECK:         config.Resources {{.+}} of @global
  // CHECK:         net.NetworkInfo entryPoint : @main

  // CHECK:         func.func nested @func0

  // CHECK:         func.func @main
  // CHECK:           call @func0
  // CHECK:       }
}

// -----

// CHECK-LABEL: @mainModuleMultipleCalleeFuncs
module @mainModuleMultipleCalleeFuncs {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }

  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee_1 = call @func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    %callee_2 = call @func2(%callee_1) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee_2 : tensor<1x3x60x60xf16>
  }

  func.func nested @func1(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
  func.func nested @func2(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = call @func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }

  // CHECK:       func.func @func
  // CHECK:         Core.NestedCall @NPUModule::@func1
  // CHECK:         call @func2

  // CHECK:       func.func nested @func2

  // CHECK:       module @NPUModule {{.+}} {
  // CHECK:         config.PipelineOptions @Options {
  // CHECK:         config.Resources {{.+}} of @NCE at {{.+}} MHz
  // CHECK:         config.Resources {{.+}} of @global
  // CHECK:         net.NetworkInfo entryPoint : @main

  // CHECK:         func.func nested @func1

  // CHECK:         func.func @main
  // CHECK:           call @func1
  // CHECK:       }
}

// -----

// CHECK-LABEL: @mainModuleMultipleCalleeFuncsNested
module @mainModuleMultipleCalleeFuncsNested {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }
  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee_1 = Core.NestedCall @nestedModule::@func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    %callee_2 = Core.NestedCall @nestedModule::@func2(%callee_1) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee_2 : tensor<1x3x60x60xf16>
  }

  module @nestedModule {
    func.func nested @func1(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    func.func nested @func2(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = Core.NestedCall @nestedModule::@func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }

  // CHECK:       func.func @func0
  // CHECK:         Core.NestedCall @NPUModule::@nestedModule::@func1
  // CHECK:         Core.NestedCall @NPUModule::@nestedModule::@func2

  // CHECK:       module @NPUModule {{.+}} {
  // CHECK:         config.PipelineOptions @Options {
  // CHECK:         config.Resources {{.+}} of @NCE at {{.+}} MHz
  // CHECK:         config.Resources {{.+}} of @global
  // CHECK:         net.NetworkInfo entryPoint : @main

  // CHECK:         module @nestedModule {
  // CHECK:           func.func nested @func1
  // CHECK:           func.func nested @func2
  // CHECK:         }

  // CHECK:         func.func @main
  // CHECK:           Core.NestedCall @nestedModule::@func1
  // CHECK:       }
}

// -----

// CHECK-LABEL: @mainModuleMultipleCalleeFuncsNestedModule
module @mainModuleMultipleCalleeFuncsNestedModule {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }
  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee_1 = Core.NestedCall @nestedModule2::@func2(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    %callee_2 = Core.NestedCall @nestedModule2::@func3(%callee_1) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee_2 : tensor<1x3x60x60xf16>
  }

  module @nestedModule1 {
    func.func nested @func1(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
  }

  module @nestedModule2 {
    func.func nested @func2(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    func.func nested @func3(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = Core.NestedCall @nestedModule2::@func2(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }

  // CHECK:      func.func @func0
  // CHECK:         Core.NestedCall @NPUModule::@nestedModule2::@func2
  // CHECK:         Core.NestedCall @NPUModule::@nestedModule2::@func3

  // CHECK:       module @nestedModule1 {
  // CHECK:         func.func nested @func1
  // CHECK:       }

  // CHECK:       module @NPUModule {{.+}} {
  // CHECK:         config.PipelineOptions @Options {
  // CHECK:         config.Resources {{.+}} of @NCE at {{.+}} MHz
  // CHECK:         config.Resources {{.+}} of @global
  // CHECK:         net.NetworkInfo entryPoint : @main

  // CHECK:         module @nestedModule2 {
  // CHECK:           func.func nested @func2
  // CHECK:           func.func nested @func3
  // CHECK:         }

  // CHECK:         func.func @main
  // CHECK:           Core.NestedCall @nestedModule2::@func2
  // CHECK:       }
}

// -----

module @mainModuleCalleeChain {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }

  module @nestedModule {
    func.func nested @func1(tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
  }

  func.func @func0(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = Core.NestedCall @nestedModule::@func1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %callee = call @func0(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %callee : tensor<1x3x60x60xf16>
  }

  // CHECK:       module @NPUModule {{.+}} {
  // CHECK:         config.PipelineOptions @Options {
  // CHECK:         config.Resources {{.+}} of @NCE at {{.+}} MHz
  // CHECK:         config.Resources {{.+}} of @global
  // CHECK:         net.NetworkInfo entryPoint : @main

  // CHECK:         module @nestedModule {
  // CHECK:           func.func nested @func1
  // CHECK:         }

  // CHECK:         func.func @func0
  // CHECK:           Core.NestedCall @nestedModule::@func1

  // CHECK:         func.func @main
  // CHECK:           call @func0
  // CHECK:       }
}
