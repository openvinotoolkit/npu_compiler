//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/config/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
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
                    DefinedInArch<"vpux::config::ArchKind::NPU60XX">
                ]
            )
        >;

//
// ODUReduceLayerOp
//

def VPUIPDPU_ODUReduceLayerOp :
        VPUIPDPU_Op<"ODUReduceLayer",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU reduce layer op is used for ReduceMax/ReduceMin operations";

    let arguments = (ins
        UnitProp:$reduce_max,
        UnitProp:$reduce_min,
        UnitProp:$reduce_total,
        VPUIPDPU_ODUReduceDataType:$reduce_dtype,
        OptionalAttr<IntAttr>:$tile_select
    );

    let assemblyFormat = [{
        `reduce_dtype` `(` $reduce_dtype `)`
        (`tile_select` `(` $tile_select^ `)`)?
        prop-dict
        attr-dict
    }];
}

//
// ODULocalRegionOp
//

def VPUIPDPU_ODULocalRegionOp :
        VPUIPDPU_Op<"ODULocalRegion",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU define of the local region - x/y coordinates of the partial compute output for the local tile or full local output suppression enable";

    let arguments = (ins
        IntAttr:$begin_coord_x,
        IntAttr:$begin_coord_y,
        IntAttr:$end_coord_x,
        IntAttr:$end_coord_y,
        UnitProp:$suppress_output
    );

    let assemblyFormat = [{
        `begin_coord_x` `(` $begin_coord_x `)`
        `begin_coord_y` `(` $begin_coord_y `)`
        `end_coord_x` `(` $end_coord_x `)`
        `end_coord_y` `(` $end_coord_y `)`
        prop-dict
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// WeightTableSplit
//

def VPUIPDPU_WeightTableSplitOp :
        VPUIPDPU_Op<"WeightTableSplit",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "Weight table split format configuration";

    let arguments =(ins
        VPUIPDPU_WtSplitMode:$wt_split_mode,
        OptionalAttr<VPUIPDPU_IDUWorkloadType>:$workload_type,
        OptionalAttr<VPUIPDPU_PPEScaleDataType>:$scale_dtype
        );

    let assemblyFormat = [{
        `wt_split_mode` `(` $wt_split_mode `)`
        (`workload_type` `(` $workload_type^ `)`)?
        (`scale_dtype` `(` $scale_dtype^ `)`)?
        attr-dict
    }];
}

//
// WeightTableLegacyZp
//

def VPUIPDPU_WeightTableLegacyZpOp :
        VPUIPDPU_Op<"WeightTableLegacyZp",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">,
                LimitedToArch<["vpux::config::ArchKind::NPU60XX"]>
            ]
        > {
    let summary = "Weight table legacy zero point configuration";

    let arguments =(ins
        VPUIPDPU_WtLegacyZpMode:$wt_legacy_zp_mode
        );

    let assemblyFormat = [{
        `wt_legacy_zp_mode` `(` $wt_legacy_zp_mode `)`
        attr-dict
    }];
}

//
// MPEEngineModeOp
//

def VPUIPDPU_MPEEngineModeOp :
        VPUIPDPU_Op<"MPEEngineMode",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE engine mode: DCIM (Digital Compute-in-Memory) or SCL (Sparse Cell)";

    let arguments = (ins
        VPU_MPEEngine60XXModeAttr:$mpe_engine_mode
    );

    let assemblyFormat = [{
        `mpe_engine_mode` `(` $mpe_engine_mode `)`
        attr-dict
    }];
}

//
// IDUpalletLUTReadOp
//

def VPUIPDPU_IDUpalletLUTReadOp :
        VPUIPDPU_Op<"IDUpalletLUTRead",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU reads the palletization table from CMX";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// WeightsTableZpBitwidthOp
//

def VPUIPDPU_WeightsTableZpBitwidthOp :
        VPUIPDPU_Op<"WeightsTableZpBitwidth",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "Weights table ZP bitwidth";

    let arguments = (ins
        VPUIPDPU_WeightsZpBitwidthType:$zp_bitwidth
    );

    let assemblyFormat = [{
        `zp_bitwidth` `(` $zp_bitwidth `)`
        attr-dict
    }];
}

//
// IDUSubtractMpeWtBiasOp
//

def VPUIPDPU_IDUSubtractMpeWtBiasOp :
        VPUIPDPU_Op<"IDUSubtractMpeWtBias",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "Subtract MPE Weight Bias in IDU for I8 weigths upconverted to FP16";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// DynPrimitiveIdVarOp
//

def VPUIPDPU_DynPrimitiveIdVarOp :
        VPUIPDPU_Op<"DynPrimitiveIdVar",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "Dynamic Primitive ID variant";

    let arguments =(ins
        IntAttr:$dyn_primitive_id_var,
        OptionalAttr<IntAttr>:$dyn_primitive_id2_var
        );

    let assemblyFormat = [{
        `dyn_primitive_id_var`  `(` $dyn_primitive_id_var `)`
        (`dyn_primitive_id2_var` `(` $dyn_primitive_id2_var^ `)`)?
        attr-dict
    }];
}

//
// DynScaleCfgInvOp
//

def VPUIPDPU_DynScaleCfgInvOp :
        VPUIPDPU_Op<"DynScaleCfgInv",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariantOp">
            ]
        > {
    let summary = "Dynamic scale config invariant";

    let arguments =(ins
        OptionalAttr<IntAttr>:$dyn_primitive_id,
        OptionalAttr<IntAttr>:$dyn_primitive_id2,
        OptionalAttr<IntAttr>:$dyn_ppe_data,
        UnitProp:$dyn_conv_cont_mod_en,
        VPUIP_DynamicSkipModeAttr:$dynamic_skip_mode,
        VPUIP_DynamicAdjustModeAttr:$dynamic_adjust_mode
        );

    let assemblyFormat = [{
        (`dyn_primitive_id`     `(` $dyn_primitive_id^ `)`)?
        (`dyn_primitive_id2`    `(` $dyn_primitive_id2^ `)`)?
        (`dyn_ppe_data`         `(` $dyn_ppe_data^ `)`)?
        `dynamic_skip_mode`     `(` $dynamic_skip_mode `)`
        `dynamic_adjust_mode`   `(` $dynamic_adjust_mode `)`
        attr-dict
        prop-dict
    }];
}

//
// IDUGroupConvOp
//

def VPUIPDPU_IDUGroupConvOp :
        VPUIPDPU_Op<"IDUGroupConvOp",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {

    let summary = "Enable partial IC (group) conv for variant";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// IDUWeightTableOffsetsOp
//

def VPUIPDPU_IDUWeightTableOffsetsOp :
        VPUIPDPU_Op<"IDUWeightTableOffsets",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU Weight settings";

    let description = [{
    Each variant that produces a slice of the output on z-axis needs to offset the weight table buffers.
    The offsets correspond to the output channels it works on.
        data_ptr_zp_offset - offset in the data pointers tables, resp. in the zero points only buffer, whichever is mapped to weight_d_ptr_start
        sp_ptr_offset - offset in sparsity pointers table, mapped to weight_sp_ptr_start
        scale_offset - offset in scale table weight_start[31:16]
        bias_offset - offset in bias table weight_start[15:0]
         }];

    let arguments =(ins
        OptionalAttr<IntAttr>:$data_ptr_zp_offset,
        OptionalAttr<IntAttr>:$sp_ptr_offset,
        OptionalAttr<IntAttr>:$scale_offset,
        OptionalAttr<IntAttr>:$bias_offset
        );

    let assemblyFormat = [{
        attr-dict
    }];
}

#endif
