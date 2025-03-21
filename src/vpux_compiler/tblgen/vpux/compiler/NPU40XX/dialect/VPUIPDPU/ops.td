//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/dialect/VPU/ops_interfaces.td"
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
                    DefinedInArch<"vpux::VPU::ArchKind::NPU40XX">
                ]
            )
        >;

//
// IDUWeights
//

def VPUIPDPU_IDUWeightsOp :
        VPUIPDPU_Op<"IDUWeights",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU Weight offset";

    let description = [{
wt_offset - This is added to the weight sparsity and weight data address
 read by the IDU and allows an offest to be specified from the addresses
 in the weight control pointer.

pool_wt_data - A custom constant weight that is not read from CMX.
This is required when doing Average pool, where the data is expected to be set to 1.
Setting this value will trigger pool_wt_rd_dis and disable reading of weights from CMX.

wt_plt_cfg - This config allows to specify the weight palletization in the IDU.

quantiles_lut - Optiona attribute that stores the weight palletization table values, when palletization is enabled.
         }];

    let arguments =(ins
        TypeAttr:$wmode,
        OptionalAttr<IntAttr>:$pool_wt_data,
        UnitAttr:$wt_sparse,
        VPUIPDPU_IDUWeightPalletMode:$wt_plt_cfg,
        OptionalAttr<F64ArrayAttr>:$quantiles_lut
        );

    let assemblyFormat = [{
        `wmode`         `(` $wmode `)`
        `wt_plt_cfg`    `(` $wt_plt_cfg `)`
        (`pool_wt_data` `(` $pool_wt_data^ `)`)?
        (`quantiles_lut` `(` $quantiles_lut^ `)`)?
        attr-dict
    }];
}

//
// ODUCmxPortsOp
//

def VPUIPDPU_ODUCmxPortsOp :
        VPUIPDPU_Op<"ODUCmxPorts",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU CMX ports config.";

    let arguments = (ins
        VPUIPDPU_ODUCmxPorts:$cmx_ports
    );

    let assemblyFormat = [{
        `cmx_ports` `(` $cmx_ports `)`
        attr-dict
    }];
}

//
// ODUWriteCombineBufferOp
//

def VPUIPDPU_ODUWriteCombineBufferOp :
        VPUIPDPU_Op<"ODUWriteCombineBuffer",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU Write Combine Buffer";

    let arguments = (ins
        VPUIPDPU_ODUWcbCombineMode:$activations_mode,
        OptionalAttr<VPUIPDPU_ODUWcbCombineMode>:$sparsity_mode
    );

    let assemblyFormat = [{
        `activations_mode` `(` $activations_mode `)`
        (`sparsity_mode` `(` $sparsity_mode^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// ODUHaloCfg
//

def VPUIPDPU_ODUHaloCfgOp :
        VPUIPDPU_Op<"ODUHaloCfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">,
                NoTerminator
            ]
        > {
    let summary = "ODU Halo configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict-with-keyword
        $region
    }];

    let hasVerifier = 1;
}

//
// ODUHaloRegionOp
//

def VPUIPDPU_ODUHaloRegionOp :
        VPUIPDPU_Op<"ODUHaloRegion",
            [
                HasParent<"vpux::VPUIPDPU::ODUHaloCfgOp">
            ]
        > {
    let summary = "ODU define halo region configuration";

    let arguments = (ins
        IntAttr:$begin_coord_x,
        IntAttr:$begin_coord_y,
        IntAttr:$end_coord_x,
        IntAttr:$end_coord_y,
        IntAttr:$activations_offset,
        OptionalAttr<IntAttr>:$sparsity_offset,
        IntAttr:$target_width,
        VPUIPDPU_DPUTiles:$cast_to_tile
    );

    let assemblyFormat = [{
        `begin_coord_x` `(` $begin_coord_x `)`
        `begin_coord_y` `(` $begin_coord_y `)`
        `end_coord_x` `(` $end_coord_x `)`
        `end_coord_y` `(` $end_coord_y `)`
        `activations_offset` `(` $activations_offset `)`
        (`sparsity_offset` `(` $sparsity_offset^ `)`)?
        `target_width` `(` $target_width `)`
        `cast_to_tile` `(` $cast_to_tile `)`
        attr-dict
    }];
}

//
// ForceInvReadOp
//

def VPUIPDPU_ForceInvReadOp :
        VPUIPDPU_Op<"ForceInvRead",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "Force read of the Invariant descriptor";

    let assemblyFormat = [{
        attr-dict
    }];
}

#endif
