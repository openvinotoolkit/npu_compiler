//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true" --inject-bytecode-metadata %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true" --inject-bytecode-metadata --inject-bytecode-metadata %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


// CHECK-LABEL: module @StaticEltwiseNHWC{{.*}}{
module @StaticEltwiseNHWC attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x1000xf16>
  }

  bytecode.func_section @function_section {
    bytecode.ext.func @main (memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>) -> () {
      %cl0 = bytecode.virtual_general_register
      bytecode.cmd_list.create %cl0
      bytecode.ret
    }
  }

  // CHECK:      bytecode.string_section @string_section {
  // CHECK:        bytecode.string [[STATIC_NET_NAME:@[A-Za-z0-9_.$-]+]] "StaticEltwiseNHWC"
  // CHECK:        bytecode.string [[STATIC_IN_0_NAME:@[A-Za-z0-9_.$-]+]] "input1"
  // CHECK:        bytecode.string [[STATIC_IN_1_NAME:@[A-Za-z0-9_.$-]+]] "input2"
  // CHECK:        bytecode.string [[STATIC_OUT_0_NAME:@[A-Za-z0-9_.$-]+]] "output"
  // CHECK:      }
  // CHECK:      bytecode.type_section @type_section {
  // CHECK:        bytecode.type [[STATIC_F16_TYPE:@[A-Za-z0-9_.$-]+]] #bytecode.float_type<width = 16, format = IEEE754>
  // CHECK:      }
  // CHECK:      bytecode.constant_section @constant_section {
  // CHECK:        bytecode.constant [[STATIC_SHAPE:@[A-Za-z0-9_.$-]+]] dense<[1, 16, 720, 1000]> : tensor<4xi64>
  // CHECK:      }
  // CHECK:      bytecode.metadata_section @metadata_section {
  // CHECK:        bytecode.network_metadata [[STATIC_NET_NAME]] 0 1
  // CHECK:        bytecode.input_metadata [[STATIC_IN_0_NAME]] {nodeFriendlyName = [[STATIC_IN_0_NAME]], shapeFromIRModel = [[STATIC_SHAPE]]} [[STATIC_F16_TYPE]] [[STATIC_SHAPE]] index_used_by_driver(0) has_dynamic_strides(false)
  // CHECK:        bytecode.input_metadata [[STATIC_IN_1_NAME]] {nodeFriendlyName = [[STATIC_IN_1_NAME]], shapeFromIRModel = [[STATIC_SHAPE]]} [[STATIC_F16_TYPE]] [[STATIC_SHAPE]] index_used_by_driver(1) has_dynamic_strides(false)
  // CHECK:        bytecode.output_metadata [[STATIC_OUT_0_NAME]] {nodeFriendlyName = [[STATIC_OUT_0_NAME]], shapeFromIRModel = [[STATIC_SHAPE]]} [[STATIC_F16_TYPE]] [[STATIC_SHAPE]] index_used_by_driver(0) has_dynamic_strides(false)
  // CHECK:      }
}

// -----

// CHECK-LABEL: module @DynamicEltwiseNHWC{{.*}}{
module @DynamicEltwiseNHWC attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_dynamic" : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>}>
  } outputsInfo : {
    DataInfo "output_dynamic" : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>}>
  }

  bytecode.func_section @function_section {
    bytecode.ext.func @main (memref<1x720x?x16xf16>, memref<1x720x?x16xf16>) -> () {
      %cl0 = bytecode.virtual_general_register
      bytecode.cmd_list.create %cl0
      bytecode.ret
    }
  }

  // CHECK:      bytecode.string_section @string_section {
  // CHECK:        bytecode.string [[DYN_NET_NAME:@[A-Za-z0-9_.$-]+]] "DynamicEltwiseNHWC"
  // CHECK:        bytecode.string [[DYN_IN_0_NAME:@[A-Za-z0-9_.$-]+]] "input_dynamic"
  // CHECK:        bytecode.string [[DYN_OUT_0_NAME:@[A-Za-z0-9_.$-]+]] "output_dynamic"
  // CHECK:      }
  // CHECK:      bytecode.type_section @type_section {
  // CHECK:        bytecode.type [[DYN_F16_TYPE:@[A-Za-z0-9_.$-]+]] #bytecode.float_type<width = 16, format = IEEE754>
  // CHECK:      }
  // CHECK:      bytecode.constant_section @constant_section {
  // CHECK:        bytecode.constant [[DYN_BOUND:@[A-Za-z0-9_.$-]+]] dense<[1, 16, 720, 1280]> : tensor<4xi64>
  // CHECK:        bytecode.constant [[DYN_SHAPE:@[A-Za-z0-9_.$-]+]] {{.*-9223372036854775808.*}}
  // CHECK:      }
  // CHECK:      bytecode.metadata_section @metadata_section {
  // CHECK:        bytecode.network_metadata [[DYN_NET_NAME]] 0 1
  // CHECK:        bytecode.input_metadata [[DYN_IN_0_NAME]] {nodeFriendlyName = [[DYN_IN_0_NAME]], shapeFromIRModel = [[DYN_SHAPE]]} [[DYN_F16_TYPE]] [[DYN_BOUND]] index_used_by_driver(0) has_dynamic_strides(false)
  // CHECK:        bytecode.output_metadata [[DYN_OUT_0_NAME]] {nodeFriendlyName = [[DYN_OUT_0_NAME]], shapeFromIRModel = [[DYN_SHAPE]]} [[DYN_F16_TYPE]] [[DYN_BOUND]] index_used_by_driver(0) has_dynamic_strides(false)
  // CHECK:      }
}

// -----

// Verify that num_streams equals the number of bytecode.cmd_list.create ops in the module.
// The config.compilationMode attribute must be present before any pass runs so that the
// net.NetworkInfo verifySymbolUses escape hatch for HostCompile_Interpreter fires correctly.
// CHECK-LABEL: module @PipelinedWithThreeStreams{{.*}}{
module @PipelinedWithThreeStreams attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x8x8xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x8x8xf16>
  }

  bytecode.func_section @function_section {
    bytecode.ext.func @main (memref<1x16x8x8xf16>, memref<1x16x8x8xf16>) -> () {
      %cl0 = bytecode.virtual_general_register
      %cl1 = bytecode.virtual_general_register
      %cl2 = bytecode.virtual_general_register
      bytecode.cmd_list.create %cl0
      bytecode.cmd_list.create %cl1
      bytecode.cmd_list.create %cl2
      bytecode.ret
    }
  }

  // CHECK:      bytecode.metadata_section @metadata_section {
  // CHECK:        bytecode.network_metadata [[PIPE_NET_NAME:@[A-Za-z0-9_.$-]+]] 0 3
  // CHECK:      }
}
