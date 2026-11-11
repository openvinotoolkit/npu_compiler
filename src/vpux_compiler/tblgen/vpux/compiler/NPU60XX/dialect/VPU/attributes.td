//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_NPU60XX_DIALECT_VPU_ATTRIBUTES
#define VPUX_COMPILER_NPU60XX_DIALECT_VPU_ATTRIBUTES

//
// MPEEngine60XXMode
//

def VPU_MPEEngine60XXMode :
        VPU_I64EnumAttr<
            "MPEEngine60XXMode",
            "MPE Engine Type for NPU60XX",
            [
                I64EnumAttrCase<"NONE", 0>,
                I64EnumAttrCase<"SCL",  1>,
                I64EnumAttrCase<"DCIM", 2>
            ]
        > {
}

def VPU_MPEEngine60XXModeAttr : VPU_EnumAttr<VPU_MPEEngine60XXMode, "mpe_engine_60xx_mode">;

// MPEEngine60XXAttr

def VPU_MPEEngine60XXAttr : VPU_Attr<"MPEEngine60XX", [
        DeclareAttrInterfaceMethods<VPU_MPEEngineAttrInterface>
    ] > {
    let description = [{
        Storing parameters only for NPU60XX MPEEngine operations.
    }];
    let parameters = (ins
        "vpux::VPU::MPEEngine60XXModeAttr":$mode,
        OptionalParameter<"mlir::Attribute">:$weight_zp,
        OptionalParameter<"mlir::Attribute">:$activation_zp
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // VPUX_COMPILER_NPU60XX_DIALECT_VPU_ATTRIBUTES
