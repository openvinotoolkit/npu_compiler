//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_SHAVE_ATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "vpux/compiler/dialect/Shave/dialect.td"

//
// Base classes
//

class Shave_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::Shave";
    let genSpecializedAttr = 0;
}

class Shave_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<Shave_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class Shave_Attr<string name, list<Trait> traits = []> :
        AttrDef<Shave_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// IOType
//

def Shave_IOType :
        Shave_I64EnumAttr<
            "IOType",
            "IO Type",
            [
                I64EnumAttrCase<"Input", 0>,
                I64EnumAttrCase<"Output", 1>
            ]
        > {
}

def Shave_IOTypeAttr : Shave_EnumAttr<Shave_IOType, "io_type">;

//
// KernelType
//

def Shave_KernelType :
        Shave_I64EnumAttr<
            "KernelType",
            "Type of an external shave Kernel",
            [
                I64EnumAttrCase<"PREBUILT", 0>,
                I64EnumAttrCase<"TRITON", 1>
            ]
        > {
}

def Shave_KernelTypeAttr : Shave_EnumAttr<Shave_KernelType, "kernel_type">;

//
// ScratchLocation
//

def Shave_ScratchLocation :
        Shave_I64EnumAttr<
            "ScratchLocation",
            "Location of the Scratch buffer",
            [
                I64EnumAttrCase<"CMX", 0>,
                I64EnumAttrCase<"DDR", 1>,
                I64EnumAttrCase<"AUTO", 2>
            ]
        > {
}

def Shave_ScratchLocationAttr : Shave_EnumAttr<Shave_ScratchLocation, "scratch_location">;

def Shave_ScratchLocationArrayAttr : TypedArrayAttrBase<Shave_ScratchLocationAttr,
                                      "Array of ScratchLocation values, one per scratch buffer"> {
}

//
// TensorRank
//

def Shave_TensorRank :
        Shave_I64EnumAttr<
            "TensorRank",
            "Rank of a Tensor",
            [
                I64EnumAttrCase<"_ND", 0>,
                I64EnumAttrCase<"_1D", 1>,
                I64EnumAttrCase<"_2D", 2>,
                I64EnumAttrCase<"_3D", 3>,
                I64EnumAttrCase<"_4D", 4>,
                I64EnumAttrCase<"_5D", 5>
            ]
        > {
}

def Shave_TensorRankAttr : Shave_EnumAttr<Shave_TensorRank, "tensor_rank">;

def Shave_TensorRankArrayAttr : TypedArrayAttrBase<Shave_TensorRankAttr,
                                      "Array of Shave_TensorRankAttrs"> {
}

//
// TensorDataType
//

// Reflects Shave DataType enum defined in sw_runtime_kernels/kernels/inc/common_types.h
def Shave_TensorDataType :
        Shave_I64EnumAttr<
            "TensorDataType",
            "DataType of a Tensor",
            [
                I64EnumAttrCase<"FP64", 0>,
                I64EnumAttrCase<"FP32", 1>,
                I64EnumAttrCase<"FP16", 2>,
                I64EnumAttrCase<"FP8", 3>,
                I64EnumAttrCase<"U64", 4>,
                I64EnumAttrCase<"U32", 5>,
                I64EnumAttrCase<"U16", 6>,
                I64EnumAttrCase<"U8", 7>,
                I64EnumAttrCase<"I64", 8>,
                I64EnumAttrCase<"I32", 9>,
                I64EnumAttrCase<"I16", 10>,
                I64EnumAttrCase<"I8", 11>,
                I64EnumAttrCase<"I4", 12>,
                I64EnumAttrCase<"I2", 13>,
                I64EnumAttrCase<"BIN", 14>,
                I64EnumAttrCase<"BF16", 15>,
                I64EnumAttrCase<"UNDEFINED", 16>,
                I64EnumAttrCase<"U4", 17>,
                I64EnumAttrCase<"NF4", 18>,
                I64EnumAttrCase<"BF8", 19>, // f8E5M2
                I64EnumAttrCase<"HF8", 20>, // f8E4M3
                I64EnumAttrCase<"U2", 21>
            ]
        > {
}

def Shave_TensorDataTypeAttr : Shave_EnumAttr<Shave_TensorDataType, "tensor_datatype">;

def Shave_TensorDataTypeArrayAttr : TypedArrayAttrBase<Shave_TensorDataTypeAttr,
                                      "Array of Shave_TensorDataTypeAttrs"> {
}

//
// TilingMode
//

def Shave_TilingMode :
        Shave_I64EnumAttr<
            "TilingMode",
            "Tiling strategy to be used for a CustomOp",
            [
                I64EnumAttrCase<"OFF",      0>,
                I64EnumAttrCase<"REPEATED", 1>
            ]
        > {
}

def Shave_TilingModeAttr : Shave_EnumAttr<Shave_TilingMode, "tiling_mode">;

//
// TensorLayout
//

def Shave_TensorLayout :
        Shave_I64EnumAttr<
            "TensorLayout",
            "Tensor Layout",
            [
                I64EnumAttrCase<"DEFAULT", 0x0>, // Keeps the original layout for any rank
                I64EnumAttrCase<"NHWC", 0x2431>,
                I64EnumAttrCase<"NHCW", 0x4231>,
                I64EnumAttrCase<"NCHW", 0x4321>,
                I64EnumAttrCase<"NCWH", 0x3421>,
                I64EnumAttrCase<"NWHC", 0x2341>,
                I64EnumAttrCase<"NWCH", 0x3241>,
                I64EnumAttrCase<"HWC", 0x132>,
                I64EnumAttrCase<"CHW", 0x321>,
                I64EnumAttrCase<"WHC", 0x123>,
                I64EnumAttrCase<"HCW", 0x312>,
                I64EnumAttrCase<"WCH", 0x213>,
                I64EnumAttrCase<"CWH", 0x231>,
                I64EnumAttrCase<"NC", 0x21>,
                I64EnumAttrCase<"CN", 0x12>,
                I64EnumAttrCase<"W", 0x1>,
                I64EnumAttrCase<"FULL_ORDER", 0xFEDCBA987654321>,
                I64EnumAttrCase<"FULL_NHWC", 0xFEDCBA987652431>,
                I64EnumAttrCase<"FULL_ND_ORDER", 0x123456789ABCDEF>,
                I64EnumAttrCase<"FULL_ND_NHWC", 0x123456789ABCEFD>
            ]
        > {
}

def Shave_TensorLayoutAttr : Shave_EnumAttr<Shave_TensorLayout, "tensor_layout">;

def Shave_TensorLayoutArrayAttr : TypedArrayAttrBase<Shave_TensorLayoutAttr,
                                      "Array of Shave_TensorLayoutAttrs"> {
}

//
// DistributionAttr
//

def Shave_KernelDistributionAttr : Shave_Attr<"KernelDistribution"> {
    let description = [{
    Represents the distribution of a Repeated Custom Kernel across multiple shaves/tiles
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$num_shaves,
        "mlir::IntegerAttr":$num_tiles
    );
}

#endif
