//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true" --inject-bytecode-metadata --verify-diagnostics %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Verify that InjectBytecodeMetadataPass fails when no CmdListCreateOp is present in the module,
// which indicates that ConvertHostcodeToBytecodePass has not run before this pass.
// expected-error @+1 {{No CmdListCreateOp found; ConvertHostcodeToBytecodePass must run before InjectBytecodeMetadataPass}}
module @NoCmdLists attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x8x8xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x8x8xf16>
  }
}

// -----

// Verify that the check fires even when a func_section is present but contains no cmd_list.create.
// expected-error @+1 {{No CmdListCreateOp found; ConvertHostcodeToBytecodePass must run before InjectBytecodeMetadataPass}}
module @FuncSectionNoCmdList attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x8x8xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x8x8xf16>
  }

  bytecode.func_section @function_section {
    bytecode.ext.func @main (memref<1x8x8x16xf16>, memref<1x8x8x16xf16>) -> () {
      bytecode.ret
    }
  }
}
