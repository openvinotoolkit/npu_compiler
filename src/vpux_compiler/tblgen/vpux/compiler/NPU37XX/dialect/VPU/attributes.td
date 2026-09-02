//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_NPU37XX_DIALECT_VPU_ATTRIBUTES
#define VPUX_COMPILER_NPU37XX_DIALECT_VPU_ATTRIBUTES

//
// MPEEngine37XXMode
//

def VPU_MPEEngine37XXMode :
        VPU_I64EnumAttr<
            "MPEEngine37XXMode",
            "MPE Engine Type for NPU37XX",
            [
                I64EnumAttrCase<"NONE", 0>,
                I64EnumAttrCase<"SCL",  1>
            ]
        > {
}

def VPU_MPEEngine37XXModeAttr : VPU_EnumAttr<VPU_MPEEngine37XXMode, "mpe_engine_37xx_mode">;

// MPEEngine37XXAttr

def VPU_MPEEngine37XXAttr : VPU_Attr<"MPEEngine37XX", [
        DeclareAttrInterfaceMethods<VPU_MPEEngineAttrInterface>
    ] > {
    let description = [{
        Storing parameters for NPU37XX, NPU40XX and NPU50XX MPEEngine operations.
    }];
    let parameters = (ins
        "vpux::VPU::MPEEngine37XXModeAttr":$mode,
        OptionalParameter<"mlir::Attribute">:$weight_zp,
        OptionalParameter<"mlir::Attribute">:$activation_zp
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // VPUX_COMPILER_NPU37XX_DIALECT_VPU_ATTRIBUTES
