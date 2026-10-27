//
// Copyright (C) 2023-2026 Intel Corporation
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

//
// AddShaveDpuDescriptors
//

def AddShaveDpuDescriptors : PassBase<"add-shave-dpu-descriptors", "vpux::FunctionPass"> {
    let summary = "Preload a dummy DPU parameters buffer into a dedicated CMX location of every tile";

    let description = [{
        SHAVE kernels which drive the DPU (see VPUIP::SW_KERNELS_USE_DPU) expect a DPU
        parameters block to be already present at a dedicated CMX address of every tile.
        The block has the following layout:

            uint32_t                 weightTable[16 * 4];   // 256 bytes
            VpuDPUInvariantRegisters invariant;             // 288 bytes
            VpuDPUVariantRegisters   variant[2];            // 2 x 192 bytes

        i.e. 928 bytes in total.

        This pass creates a constant holding a valid, but trivial, version of that block
        and one bootstrap NNDMA per tile which copies it to the dedicated CMX offset.
        The DMAs carry no wait/update barriers and are prepended to the DMA list, so they
        are executed before any other task of the inference.

        The pass is a no-op if the network contains no SHAVE task which uses the DPU.

        The pass operates on VPUMI40XX IR (it runs before the VPUASM/NPUReg50XX lowering),
        but lives in the NPUReg50XX dialect because it authors the DPU parameters block
        directly through the NPUReg50XX descriptor definitions
        (NPUReg50XX::Descriptors::DpuInvariantRegister / DpuVariantRegister).
    }];

    let constructor = "vpux::NPUReg50XX::createAddShaveDpuDescriptors()";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
    ];
}

#endif
