//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUASM/attributes.td"
include "vpux/compiler/dialect/VPUASM/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPDPU/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

//
// Base classes
//

class VPUIPDPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIPDPU_Dialect,
            mnemonic,
            traits
        >;

//
// DPU Invariant
//

def VPUIPDPU_DPUInvariantOp :
        VPUIPDPU_Op<"DPUInvariant",
            [
                VPUASM_IndexedTaskInterface,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "DPU Invariant configuration";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$input,
        OptionalAttr<SymbolRefAttr>:$input_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$input_storage_element_table,

        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weights_sparsity_map,

        OptionalAttr<SymbolRefAttr>:$weight_table,
        OptionalAttr<SymbolRefAttr>:$weight_table_data_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_scale,

        OptionalAttr<SymbolRefAttr>:$spr_lookup_table,
        OptionalAttr<SymbolRefAttr>:$pallet_lookup_table,

        OptionalAttr<SymbolRefAttr>:$output,
        OptionalAttr<SymbolRefAttr>:$output_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$profiling_data,
        UnitAttr:$is_zero_offset_weights_table,

        OptionalAttr<SymbolRefAttr>:$max_per_xy,
        OptionalAttr<SymbolRefAttr>:$min_per_xy,
        OptionalAttr<SymbolRefArrayAttr>:$min_max_per_tensor,

        OptionalAttr<SymbolRefAttr>:$dynamic_sequence_length,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        UnitAttr:$is_continued
    );

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        $sym_name
        (`profiling_data` `(` $profiling_data^ `)`)?
        attr-dict
        `DPUCfg` `:` $region
    }];

    let hasVerifier = 1;
}

//
// DPU Variant
//

def VPUIPDPU_DPUVariantOp :
        VPUIPDPU_Op<"DPUVariant",
            [
                Symbol,
                NoTerminator
            ]
        > {
    let summary = "DPU Variant configuration";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$invariant_task_location,

        OptionalAttr<SymbolRefAttr>:$weights,

        OptionalAttr<SymbolRefAttr>:$weight_table,
        OptionalAttr<SymbolRefAttr>:$weight_table_data_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_sp_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_scale,
        OptionalAttr<SymbolRefAttr>:$weight_table_bias,
        OptionalAttr<SymbolRefAttr>:$weight_zero_points,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        OptionalAttr<IntAttr>:$workload_id,
        OptionalAttr<I64ArrayAttr>:$variant_primitive_id // IDs for variant primitives A and B
    );

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        $sym_name
        (`nextLink` `(` $next_link^ `)`)?
        `invariant` `(` $invariant_task_location `)`
        attr-dict
        `DPUCfg` `:` $region
    }];

    let hasVerifier = 1;
}

//
// IDUCfg
//

def VPUIPDPU_IDUCfgOp :
        VPUIPDPU_Op<"IDUCfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariantOp">,
                NoTerminator
            ]
        > {
    let summary = "IDU configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict-with-keyword
        $region
    }];
}

//
// MPECfg
//

def VPUIPDPU_MPECfgOp :
        VPUIPDPU_Op<"MPECfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariantOp">,
                NoTerminator
            ]
        > {
    let summary = "MPE configuration";

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
// PPECfg
//

def VPUIPDPU_PPECfgOp :
        VPUIPDPU_Op<"PPECfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariantOp">,
                NoTerminator
            ]
        > {
    let summary = "PPE configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict-with-keyword
        $region
    }];
}

//
// ODUCfg
//

def VPUIPDPU_ODUCfgOp :
        VPUIPDPU_Op<"ODUCfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariantOp">,
                NoTerminator
            ]
        > {
    let summary = "ODU configuration";

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
// BarrierCfgOp
//

def VPUIPDPU_BarrierCfgOp:
    VPUIPDPU_Op<"BarrierCfg",
        [
            ParentOneOf<["vpux::VPUIPDPU::DPUVariantOp", "vpux::VPUIPDPU::DPUInvariantOp"]>
        ]
    > {

    let summary = "Barrier Configuration op";

    let arguments = (ins
        UI8ArrayAttr:$waitBarriers,
        UI8ArrayAttr:$updateBarriers,
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let assemblyFormat = [{
        `waits` `(` $waitBarriers `)`
        `updates` `(` $updateBarriers `)`
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        attr-dict
    }];
}

//
// DPUGroupOp
//

def VPUIPDPU_DPUGroupOp:
    VPUIPDPU_Op<"DPUGroup",
        [
            ParentOneOf<["vpux::VPUIPDPU::DPUVariantOp", "vpux::VPUIPDPU::DPUInvariantOp"]>
        ]
    > {

    let summary = "DPU Group Op";

    let description = [{In some views it is necessary to view a "logical" dpu task as a pair of Invariant and Variant
                            group size. This OP's purpose is to keep that information in the form of
                            list identifier of the invariant, and variant group size }];

    let arguments = (ins
        VPUASM_IndexTypeAttr:$invariantIdx,
        UI64Attr:$variantCount,
        UnitAttr:$isFirstVariant,
        UnitAttr:$isLastVariant
    );

    let assemblyFormat = [{
        `invariantIdx` `(` $invariantIdx `)`
        `variantCount` `(` $variantCount `)`
        attr-dict
    }];
}

#endif
