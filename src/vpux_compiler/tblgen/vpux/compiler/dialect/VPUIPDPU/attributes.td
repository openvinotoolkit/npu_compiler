//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIPDPU_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"

//
// PPEBypassMode
//

def VPUIPDPU_PPEBypassMode :
        I32EnumAttr<
            "PPEBypassMode",
            "PPE bypass mode",
            [
                I32EnumAttrCase<"OFF", 0>,
                I32EnumAttrCase<"ON", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvertMode
//

def VPUIPDPU_PPEFpConvertMode :
        I32EnumAttr<
            "PPEFpConvertMode",
            "PPE FP convert mode",
            [
                I32EnumAttrCase<"NONE", 0>,
                I32EnumAttrCase<"FP16", 1>,
                I32EnumAttrCase<"BF16", 2>,
                I32EnumAttrCase<"BF8", 3>,
                I32EnumAttrCase<"I32", 4>,
                I32EnumAttrCase<"HF8", 5>,
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvClampMode
//

def VPUIPDPU_PPEFpConvClampMode :
        I32EnumAttr<
            "PPEFpConvClampMode",
            "PPE FP convert clamp mode",
            [
                I32EnumAttrCase<"OFF", 0>,
                I32EnumAttrCase<"ON", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvFTZMode
//

def VPUIPDPU_PPEFpConvFTZMode :
        I32EnumAttr<
            "PPEFpConvFTZMode",
            "PPE FP convert force-to-zero mode",
            [
                I32EnumAttrCase<"OFF", 0>,
                I32EnumAttrCase<"ON", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvBf16RoundMode
//

def VPUIPDPU_PPEFpConvBf16RoundMode :
        I32EnumAttr<
            "PPEFpConvBf16RoundMode",
            "PPE FP convert to BF16 round mode",
            [
                I32EnumAttrCase<"TRUNC", 0>,
                I32EnumAttrCase<"RNE", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEIntRoundMode
//

def VPUIPDPU_PPEIntRoundMode :
        I32EnumAttr<
            "PPEIntRoundMode",
            "PPE Int round mode",
            [
                I32EnumAttrCase<"RNE", 0>,
                I32EnumAttrCase<"RNAZ", 2>,
                I32EnumAttrCase<"NONE", 3>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEIntConvertMode
//

def VPUIPDPU_PPEIntConvertMode :
        I32EnumAttr<
            "PPEIntConvertMode",
            "PPE Int convert mode",
            [
                I32EnumAttrCase<"NONE", 0>,
                I32EnumAttrCase<"FP16", 1>,
                I32EnumAttrCase<"FP8", 2>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEsprLUTMode
//

def VPUIPDPU_PPEsprLUTMode :
        I32EnumAttr<
            "PPEsprLUTMode",
            "PPE sprLUT (Software Programmable Register based Look-Up Table) mode",
            [
                I32EnumAttrCase<"OFF", 1>,
                I32EnumAttrCase<"ON", 2>
            ]
        > {
    let summary = "PPE sprLUT mode of operation: 1 - Not NPU40XX blob compatible, PPE config conversions disabled (sprlut disabled) 2 - Not NPU40XX blob compatible, PPE config conversions disabled (sprlut enabled)";
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUDataTypes
//

def VPUIPDPU_ODUDataTypes :
        I32EnumAttr<
            "ODUDataTypes",
            "ODU output data types - VPU2.7",
            [
                I32EnumAttrCase<"ODU_DTYPE_FP16", 0>,
                I32EnumAttrCase<"ODU_DTYPE_U8F",  1>,
                I32EnumAttrCase<"ODU_DTYPE_G8",   2>,
                I32EnumAttrCase<"ODU_DTYPE_I8",   3>,
                I32EnumAttrCase<"ODU_DTYPE_I32",  4>,
                I32EnumAttrCase<"ODU_DTYPE_I4",   5>,
                I32EnumAttrCase<"ODU_DTYPE_I2",   6>,
                I32EnumAttrCase<"ODU_DTYPE_LOG",  7>,
                I32EnumAttrCase<"ODU_DTYPE_BIN",  8>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUDataBitWidth
//

def VPUIPDPU_ODUDataBitWidth :
        I32EnumAttr<
            "ODUDataBitWidth",
            "ODU output data bit width - NPU40XX",
            [
                I32EnumAttrCase<"ODU_DTYPE_1BIT",  0>,
                I32EnumAttrCase<"ODU_DTYPE_2BIT",  1>,
                I32EnumAttrCase<"ODU_DTYPE_4BIT",  2>,
                I32EnumAttrCase<"ODU_DTYPE_8BIT",  3>,
                I32EnumAttrCase<"ODU_DTYPE_16BIT", 4>,
                I32EnumAttrCase<"ODU_DTYPE_32BIT", 5>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUOutTensorDataMode
//

def VPUIPDPU_ODUActivationReuseMode :
        I32EnumAttr<
            "ODUActivationReuseMode",
            "ODU activation reuse mode",
            [
                I32EnumAttrCase<"NTHW_1",   0>,
                I32EnumAttrCase<"NTHW_4",   1>,
                I32EnumAttrCase<"NTHW_8",   2>,
                I32EnumAttrCase<"NTHW_16",  3>,
                I32EnumAttrCase<"NTHW_CIM", 4>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUPermuteDataMode
//

def VPUIPDPU_ODUPermuteDataMode :
        I32EnumAttr<
            "ODUPermuteDataMode",
            "ODU data permutation mode",
            [
                I32EnumAttrCase<"PERMUTE_ZXY",  0>,
                I32EnumAttrCase<"PERMUTE_ZYX",  1>,
                I32EnumAttrCase<"PERMUTE_YZX",  2>,
                I32EnumAttrCase<"PERMUTE_YXZ",  3>,
                I32EnumAttrCase<"PERMUTE_XZY",  4>,
                I32EnumAttrCase<"PERMUTE_XYZ",  5>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUMemoryMode
//

def VPUIPDPU_ODUMemoryMode :
        I32EnumAttr<
            "ODUMemoryMode",
            "ODU memory data mode",
            [
                I32EnumAttrCase<"MODE_DENSE",  0>,
                I32EnumAttrCase<"MODE_SUPERDENSE",  1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUWcbCombineMode
//

def VPUIPDPU_ODUWcbCombineMode :
        I32EnumAttr<
            "ODUWcbCombineMode",
            "ODU WCB combination mode",
            [
                I32EnumAttrCase<"WCB_COMBINE_BY_CONTEXT", 0>,
                I32EnumAttrCase<"WCB_COMBINE_BY_ADDRESS", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// DPUSwizzleKey
//

def VPUIPDPU_DPUSwizzleKey :
        I32EnumAttr<
            "DPUSwizzleKey",
            "DPU data swizzle key config to be used for IDU and ODU",
            [
                I32EnumAttrCase<"SWIZZLE_OFF",   0>,
                I32EnumAttrCase<"SWIZZLE_KEY_1", 1>,
                I32EnumAttrCase<"SWIZZLE_KEY_2", 2>,
                I32EnumAttrCase<"SWIZZLE_KEY_3", 3>,
                I32EnumAttrCase<"SWIZZLE_KEY_4", 4>,
                I32EnumAttrCase<"SWIZZLE_KEY_5", 5>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUCmxPorts
//

def VPUIPDPU_ODUCmxPorts :
        I32EnumAttr<
            "ODUCmxPorts",
            "ODU out data CMX ports config.",
            [
                I32EnumAttrCase<"CMX_PORTS_ALL", 0>,
                I32EnumAttrCase<"CMX_PORTS_01",  1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUHaloAugmentatorDPUTile
//

def VPUIPDPU_DPUTiles :
        I64BitEnumAttr<
            "DPUTiles",
            "DPU tile IDs",
            [
                I64BitEnumAttrCase<"DPU_TILE_0", 1>,
                I64BitEnumAttrCase<"DPU_TILE_1", 2>,
                I64BitEnumAttrCase<"DPU_TILE_2", 4>,
                I64BitEnumAttrCase<"DPU_TILE_3", 8>,
                I64BitEnumAttrCase<"DPU_TILE_4", 16>,
                I64BitEnumAttrCase<"DPU_TILE_5", 32>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUReduceDataType
//

def VPUIPDPU_ODUReduceDataType :
        I32EnumAttr<
            "ODUReduceDataType",
            "ODU output data types supported by reduce operation",
            [
                I32EnumAttrCase<"ODU_DTYPE_U8",     0>,
                I32EnumAttrCase<"ODU_DTYPE_I8",     1>,
                I32EnumAttrCase<"ODU_DTYPE_BF8",    2>,
                I32EnumAttrCase<"ODU_DTYPE_HF8",    3>,
                I32EnumAttrCase<"ODU_DTYPE_BF16",   4>,
                I32EnumAttrCase<"ODU_DTYPE_FP16",   5>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// IDUWorkloadType
//

def VPUIPDPU_IDUWorkloadType :
        I32EnumAttr<
            "IDUWorkloadType",
            "Suported tensor workload types",
            [
                I32EnumAttrCase<"CONV",                    0>,
                I32EnumAttrCase<"DWCONV",                  1>,
                I32EnumAttrCase<"MAXPOOL",                 2>,
                I32EnumAttrCase<"AVEPOOL",                 3>,
                I32EnumAttrCase<"ELTWISE",                 4>,
                I32EnumAttrCase<"REDUCEMEAN",              5>,
                I32EnumAttrCase<"REDUCESUMSQUARE",         6>,
                I32EnumAttrCase<"REDUCESUM",               7>,
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// IDUWeightPalletMode
//
def VPUIPDPU_IDUWeightPalletMode :
        I32EnumAttr<
            "IDUWeightPalletMode",
            "Suported weight palletization modes",
            [
                I32EnumAttrCase<"NO_PLT",                  0>,
                I32EnumAttrCase<"ONE_BIT_PLT",             1>,
                I32EnumAttrCase<"TWO_BIT_PLT",             2>,
                I32EnumAttrCase<"THREE_BIT_PLT",           3>,
                I32EnumAttrCase<"FOUR_BIT_PLT",            4>,
                I32EnumAttrCase<"FIVE_BIT_PLT",            5>,
                I32EnumAttrCase<"SIX_BIT_PLT",             6>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// IDUEltwiseType
//

def VPUIPDPU_IDUEltwiseType :
        I32EnumAttr<
            "IDUEltwiseType",
            "Elementwise operation type (only valid when elop_wload=1'b1)",
            [
                I32EnumAttrCase<"ADD",                    0>,
                I32EnumAttrCase<"SUBTRACT",               1>,
                I32EnumAttrCase<"MULT",                   2>,
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// IDUNthwNtk
//

def VPUIPDPU_IDUNthwNtk :
        I32EnumAttr<
            "IDUNthwNtk",
            "Suported NTHW_NTK setting",
            [
                I32EnumAttrCase<"NTHW_NTK_8_8",         0>,
                I32EnumAttrCase<"NTHW_NTK_4_16",        1>,
                I32EnumAttrCase<"NTHW_NTK_16_4",        3>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// WtSplitMode
//

def VPUIPDPU_WtSplitMode :
        I32EnumAttr<
            "WtSplitMode",
            "Weight table split modes",
            [
                I32EnumAttrCase<"NO_WT_READS",          0>,
                I32EnumAttrCase<"WEIGHTS_ONLY",         1>,
                I32EnumAttrCase<"ZERO_POINTS_ONLY",     2>,
                I32EnumAttrCase<"WEIGHTS_AND_ZPS",      3>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// WtSplitMode
//

def VPUIPDPU_PPEScaleDataType :
        I32EnumAttr<
            "PPEScaleDataType",
            "PPE scale data type",
            [
                I32EnumAttrCase<"FP32", 0>,
                I32EnumAttrCase<"HF8",  1>,
                I32EnumAttrCase<"BF8",  2>,
                I32EnumAttrCase<"E8M0", 3>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

#endif
