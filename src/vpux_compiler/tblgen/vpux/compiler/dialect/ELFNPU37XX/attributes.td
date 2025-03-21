//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_ELF_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/ELFNPU37XX/dialect.td"

//
// Base classes
//

class ELFNPU37XX_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::ELFNPU37XX";
    let genSpecializedAttr = 0;
}

class ELFNPU37XX_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<ELFNPU37XX_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

// For further development, please refer to the ticket #29279.

//
// SectionTypeAttr
//
// Refer to https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html
//   for the detailed description of the case values.

// I64EnumAttr has uint64_t as the underlying type, which is OK for all cases.

def ELFNPU37XX_CMXMappingSymbol :
        ELFNPU37XX_I64EnumAttr<
            "CMXMappingSymbol",
            "Enum for describing VPU special CMX Mapping names and values",
            [
                I64EnumAttrCase<"VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR",  0>,
                I64EnumAttrCase<"VPU_NNRD_SYM_RTM_IVAR",               1>,
                I64EnumAttrCase<"VPU_NNRD_SYM_RTM_ACT",                2>,
                I64EnumAttrCase<"VPU_NNRD_SYM_RTM_DMA0",               3>,
                I64EnumAttrCase<"VPU_NNRD_SYM_RTM_DMA1",               4>,
                I64EnumAttrCase<"VPU_NNRD_SYM_FIFO_BASE",              5>,
                I64EnumAttrCase<"VPU_NNRD_SYM_BARRIERS_START",         6>,
                I64EnumAttrCase<"VPU_NNRD_SYM_HW_REGISTER",            7>
            ]
        > {
}

def ELFNPU37XX_CMXMappingSymbolAttr : ELFNPU37XX_EnumAttr<ELFNPU37XX_CMXMappingSymbol, "cmx_mapping_symbol">;

//
// SectionTypeAttr
//
// Refer to https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html
//   for the detailed description of the case values.

// I64EnumAttr has uint64_t as the underlying type, which is OK for all cases.
def ELFNPU37XX_SectionTypeAttr :
        I64EnumAttr<
            "SectionTypeAttr",
            "Enum for describing ELF section header types",
            [
                I64EnumAttrCase<"SHT_NULL",              0>,
                I64EnumAttrCase<"SHT_PROGBITS",          1>,
                I64EnumAttrCase<"SHT_SYMTAB",            2>,
                I64EnumAttrCase<"SHT_STRTAB",            3>,
                I64EnumAttrCase<"SHT_RELA",              4>,
                I64EnumAttrCase<"SHT_HASH",              5>,
                I64EnumAttrCase<"SHT_DYNAMIC",           6>,
                I64EnumAttrCase<"SHT_NOTE",              7>,
                I64EnumAttrCase<"SHT_NOBITS",            8>,
                I64EnumAttrCase<"SHT_REL",               9>,
                I64EnumAttrCase<"SHT_SHLIB",             10>,
                I64EnumAttrCase<"SHT_DYNSYM",            11>,
                I64EnumAttrCase<"SHT_LOPROC",            0x70000000>,
                I64EnumAttrCase<"SHT_HIPROC",            0x7FFFFFFF>,
                I64EnumAttrCase<"SHT_LOUSER",            0x80000000>,
                I64EnumAttrCase<"VPU_SHT_NETDESC",       0x8AAAAAAA>,
                I64EnumAttrCase<"VPU_SHT_PROF",          0x8AAAAAAB>,
                I64EnumAttrCase<"VPU_SHT_PERF_METRICS",  0x8AAAAAAE>,
                I64EnumAttrCase<"VPU_SHT_PLATFORM_INFO", 0x8AAAAAAF>,
                I64EnumAttrCase<"SHT_HIUSER",            0xFFFFFFFF>
            ]
        > {
    let cppNamespace = "vpux::ELFNPU37XX";
    let genSpecializedAttr = 1;
}

//
// SymbolTypeAttr
//
// Refer to https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-79797.html
// for the detailed description of the case values

// The suffix "Enum" was added here because otherwise this name conflicts with the SymbolType type
def ELFNPU37XX_SymbolTypeEnum:
    ELFNPU37XX_I64EnumAttr<
        "SymbolTypeEnum",
        "Enum to represent symbol types",
        [
            I64EnumAttrCase<"STT_NOTYPE"     , 0>,
            I64EnumAttrCase<"STT_OBJECT"     , 1>,
            I64EnumAttrCase<"STT_FUNC"       , 2>,
            I64EnumAttrCase<"STT_SECTION"    , 3>,
            I64EnumAttrCase<"STT_FILE"       , 4>,
            I64EnumAttrCase<"STT_COMMON"     , 5>,
            I64EnumAttrCase<"STT_TLS"        , 6>,
            I64EnumAttrCase<"VPU_STT_ENTRY"  , 10>,
            I64EnumAttrCase<"VPU_STT_INPUT"  , 11>,
            I64EnumAttrCase<"VPU_STT_OUTPUT" , 12>,
            I64EnumAttrCase<"STT_LOPROC"     , 13>,
            I64EnumAttrCase<"STT_HIPROC"     , 15>,
        ]
    > {
}

def ELFNPU37XX_SymbolTypeAttr : ELFNPU37XX_EnumAttr<ELFNPU37XX_SymbolTypeEnum, "symbol_type">;

//
// SectionFlagsAttr
//
// Refer to
// https://docs.oracle.com/cd/E19120-01/open.solaris/819-0690/6n33n7fcj/index.html
// (or https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html)
// for the detailed description of the case values

def ELFNPU37XX_SectionFlagsAttr :
        I64BitEnumAttr<
            "SectionFlagsAttr",
            "Enum for describing ELF section header flags (we can use also the | operator)",
            [
                I64BitEnumAttrCase<"SHF_NONE",                0x0>,
                I64BitEnumAttrCase<"SHF_WRITE",               0x1>,
                I64BitEnumAttrCase<"SHF_ALLOC",               0x2>,
                I64BitEnumAttrCase<"SHF_EXECINSTR",           0x4>,
                I64BitEnumAttrCase<"SHF_INFO_LINK",           0x40>,
                I64BitEnumAttrCase<"VPU_SHF_JIT",             0x100000>,
                I64BitEnumAttrCase<"VPU_SHF_USERINPUT",       0x200000>,
                I64BitEnumAttrCase<"VPU_SHF_USEROUTPUT",      0x400000>,
                I64BitEnumAttrCase<"VPU_SHF_PROFOUTPUT",      0x800000>,
                I64BitEnumAttrCase<"SHF_MASKOS",              0xFF00000>,
                I64BitEnumAttrCase<"VPU_SHF_PROC_DPU",        0x10000000>,
                I64BitEnumAttrCase<"VPU_SHF_PROC_DMA",        0x20000000>,
                I64BitEnumAttrCase<"VPU_SHF_PROC_SHAVE",      0x40000000>,
                I64BitEnumAttrCase<"SHF_MASKPROC",            0xF0000000>,
            ]
        > {
    let cppNamespace = "vpux::ELFNPU37XX";
}

//
// RelocationTypeAttr
//

def ELFNPU37XX_RelocationType :
        ELFNPU37XX_I64EnumAttr<
            "RelocationType",
            "Enum for describing ELF relocation types",
            [
                I64EnumAttrCase<"R_VPU_64",                                  0>,
                I64EnumAttrCase<"R_VPU_64_OR",                               1>,
                I64EnumAttrCase<"R_VPU_DISP40_RTM",                          2>,
                I64EnumAttrCase<"R_VPU_64_LSHIFT",                           3>,
                I64EnumAttrCase<"R_VPU_32",                                  4>,
                I64EnumAttrCase<"R_VPU_32_RTM",                              5>,
                I64EnumAttrCase<"R_VPU_32_SUM",                              6>,
                I64EnumAttrCase<"R_VPU_32_MULTICAST_BASE",                   7>,
                I64EnumAttrCase<"R_VPU_32_MULTICAST_BASE_SUB",               8>,
                I64EnumAttrCase<"R_VPU_DISP28_MULTICAST_OFFSET",             9>,
                I64EnumAttrCase<"R_VPU_DISP4_MULTICAST_OFFSET_CMP",         10>,
                I64EnumAttrCase<"R_VPU_N_CONSECUTIVE_32",                   11>,
                I64EnumAttrCase<"R_VPU_16_SUM",                           1011>,
                I64EnumAttrCase<"R_VPU_64_MULT",                          1012>,
                I64EnumAttrCase<"R_VPU_64_MULT_SUB",                      1013>
            ]
        > {
}

def ELFNPU37XX_RelocationTypeAttr : ELFNPU37XX_EnumAttr<ELFNPU37XX_RelocationType, "relocation_type">;

#endif
