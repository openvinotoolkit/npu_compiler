//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUREG50XX_PASSES
#define VPUX_COMPILER_DIALECT_NPUREG50XX_PASSES

include "mlir/Pass/PassBase.td"

//
// Setup IduCmxMuxMode
//

def SetupIduCmxMuxMode : PassBase<"setup-idu-cmx-mux-mode", "vpux::FunctionPass"> {
    let summary = "Set up idu_cmx_mux_mode filed for DPUInvariant";

    let description = [{
        Set up idu_cmx_mux_mode field for DPUInvariant.
        Possible values for idu_cmx_mux_mode[1:0]:
            2'b00 = default RTL behaviour
            2'b01 = force mode0
            2'b10 = force mode1
            2'b11 = force mode2
    }];

    let constructor = "vpux::NPUReg50XX::createSetupIduCmxMuxModePass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect",
        "vpux::NPUReg50XX::NPUReg50XXDialect",
    ];

    let options = [
        Option<
            "iduCmxMuxMode", "idu-cmx-mux-mode",
            "std::string", [{"MODE0"}],
            "[Optional] Set tiling mode as `MODE0` or `MODE1` or `MODE3`. `MODE0` is set by default"
        >
    ];
}

#endif
