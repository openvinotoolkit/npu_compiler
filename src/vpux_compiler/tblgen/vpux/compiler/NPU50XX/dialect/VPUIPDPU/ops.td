//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/config/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPDPU/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"
include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.td"

//
// Base classes
//

class VPUIPDPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIPDPU_Dialect,
            mnemonic,
            !listconcat(
                traits,
                [
                    DefinedInArch<"vpux::config::ArchKind::NPU50XX">
                ]
            )
        >;

//
// IDUEltWiseMode
//

def VPUIPDPU_IDUEltWiseModeOp :
        VPUIPDPU_Op<"IDUEltWiseMode",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU eltwise mode configuration";

    let arguments =(ins
        VPUIPDPU_IDUEltwiseType:$eltwise_type
        );

    let assemblyFormat = [{
        `eltwise_type`  `(` $eltwise_type `)`
        attr-dict
    }];
}

//
// IDUSEOnly
//

def VPUIPDPU_IDUSEOnlyOp :
        VPUIPDPU_Op<"IDUSEOnly",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU set Storage Element table in se only mode";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// IDUPerOutputChannelScaling
//

def VPUIPDPU_IDUPerOutputChannelScalingOp :
        VPUIPDPU_Op<"IDUPerOutputChannelScaling",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU per output channel scaling configuration";

    let description = [{
This ops is relevant only in case of maxpool, avepool and dual elops with per output channel quantization scale.

tensor2_act_sparse - set if eltwise tensor2 is sparse
         }];

    let arguments =(ins
        UnitAttr:$tensor2_act_sparse
        );

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// PPEFpScaleMultOp
//

def VPUIPDPU_PPEFpScaleMultOp :
        VPUIPDPU_Op<"PPEFpScaleMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline scale only multiplication";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<F32Attr>:$scale_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`scale_static` `(` $scale_static^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEFpSprLUTModeOp
//

def VPUIPDPU_PPEFpSprLUTModeOp :
        VPUIPDPU_Op<"PPEFpSprLUTMode",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline sprLUT (Software Programmable Register based Look-Up Table) bypass mode";

    let arguments = (ins
        VPUIPDPU_PPEsprLUTMode:$sprlut_mode
    );

    let assemblyFormat = [{
        `sprlut_mode` `(` $sprlut_mode `)`
        attr-dict
    }];
}

//
// PPEFpPreluMultOp
//

def VPUIPDPU_PPEFpPreluMultOp :
        VPUIPDPU_Op<"PPEFpPreluMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline pReLU scale only multiplication";

    let arguments = (ins
        F32Attr:$prelu_alpha
    );

    let assemblyFormat = [{
        `prelu_alpha` `(` $prelu_alpha `)`
        attr-dict
    }];
}

//
// PPEFpClampOp
//

def VPUIPDPU_PPEFpClampOp :
        VPUIPDPU_Op<"PPEFpClamp",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline clamping (FP32 high and low clamp respectively)";

    let arguments = (ins
        F32Attr:$clamp_low,
        F32Attr:$clamp_high
    );

    let assemblyFormat = [{
        `clamp_low` `(` $clamp_low `)`
        `clamp_high` `(` $clamp_high `)`
        attr-dict
    }];
}

//
// PPEsprLUTReadOp
//

def VPUIPDPU_PPEsprLUTReadOp :
        VPUIPDPU_Op<"PPEsprLUTRead",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "PPE read the sprLUT config from CMX";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// MPEMode
//

def VPUIPDPU_MPEBf16ModeOp :
        VPUIPDPU_Op<"MPEBf16Mode",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE mode BF16 enabled";

    let assemblyFormat = [{
        attr-dict
    }];
}


#endif
