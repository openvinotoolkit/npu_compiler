//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_ELF_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/dialect.td"

//
// Base classes
//

class ELF_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 0;
}

class ELF_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<ELF_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

// For further development, please refer to the ticket #29279.

//
// SectionTypeAttr
//
// Refer to https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html
//   for the detailed description of the case values.

// I64EnumAttr has uint64_t as the underlying type, which is OK for all cases.
def ELF_SectionTypeAttr :
        I64EnumAttr<
            "SectionTypeAttr",
            "Enum for describing ELF section header types",
            [
                I64EnumAttrCase<"SHT_NULL",               0>,
                I64EnumAttrCase<"SHT_PROGBITS",           1>,
                I64EnumAttrCase<"SHT_SYMTAB",             2>,
                I64EnumAttrCase<"SHT_STRTAB",             3>,
                I64EnumAttrCase<"SHT_RELA",               4>,
                I64EnumAttrCase<"SHT_HASH",               5>,
                I64EnumAttrCase<"SHT_DYNAMIC",            6>,
                I64EnumAttrCase<"SHT_NOTE",               7>,
                I64EnumAttrCase<"SHT_NOBITS",             8>,
                I64EnumAttrCase<"SHT_REL",                9>,
                I64EnumAttrCase<"SHT_SHLIB",              10>,
                I64EnumAttrCase<"SHT_DYNSYM",             11>,
                I64EnumAttrCase<"SHT_LOPROC",             0x70000000>,
                I64EnumAttrCase<"SHT_HIPROC",             0x7FFFFFFF>,
                I64EnumAttrCase<"SHT_LOUSER",             0x80000000>,
                I64EnumAttrCase<"VPU_SHT_NETDESC",        0x8AAAAAAA>,
                I64EnumAttrCase<"VPU_SHT_PROF",           0x8AAAAAAB>,
                I64EnumAttrCase<"VPU_SHT_CMX_METADATA",   0x8AAAAAAC>,
                I64EnumAttrCase<"VPU_SHT_CMX_WORKSPACE",  0x8AAAAAAD>,
                I64EnumAttrCase<"VPU_SHT_PERF_METRICS",   0x8AAAAAAE>,
                I64EnumAttrCase<"VPU_SHT_PLATFORM_INFO",  0x8AAAAAAF>,
                I64EnumAttrCase<"SHT_HIUSER",             0xFFFFFFFF>
            ]
        > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}

//
// SymbolTypeAttr
//
// Refer to https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-79797.html
// for the detailed description of the case values

def ELF_SymbolType :
    ELF_I64EnumAttr<
        "SymbolType",
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

def ELF_SymbolTypeAttr : ELF_EnumAttr<ELF_SymbolType, "symbol_type">;

//
// SectionFlagsAttr
//
// Refer to
// https://docs.oracle.com/cd/E19120-01/open.solaris/819-0690/6n33n7fcj/index.html
// (or https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html)
// for the detailed description of the case values

def ELF_SectionFlagsAttr :
        I64BitEnumAttr<
            "SectionFlagsAttr",
            "Enum for describing ELF section header flags (we can use also the | operator)",
            [
                I64BitEnumAttrCase<"SHF_NONE",                0x00000000>,
                I64BitEnumAttrCase<"SHF_WRITE",               0x00000001>,  // write access at inference execution time
                I64BitEnumAttrCase<"SHF_ALLOC",               0x00000002>,  // for DDR only (declare buffers)
                I64BitEnumAttrCase<"SHF_EXECINSTR",           0x00000004>,  // data in DDR to be accessed by RT

                I64BitEnumAttrCase<"SHF_MASKOS",              0x0FF00000>,  // all bits that are included in this mask are reserved for operating system-specific semantics
                I64BitEnumAttrCase<"VPU_SHF_JIT",             0x00100000>,  // sections of JIT relocations (network input, output and profiling)
                I64BitEnumAttrCase<"VPU_SHF_USERINPUT",       0x00200000>,  // tables of symbols for sections that contain declare buffers for network input
                I64BitEnumAttrCase<"VPU_SHF_USEROUTPUT",      0x00400000>,  // tables of symbols for sections that contain declare buffers for network output
                I64BitEnumAttrCase<"VPU_SHF_PROFOUTPUT",      0x00800000>,  // tables of symbols for sections that contain declare buffers for network profiling outputs

                I64BitEnumAttrCase<"SHF_MASKPROC",            0xF0000000>,  // all bits that are included in this mask are reserved for operating system-specific semantics
                I64BitEnumAttrCase<"VPU_SHF_PROC_DMA",        0x20000000>,  // sections to be accessed by DMA
                I64BitEnumAttrCase<"VPU_SHF_PROC_SHAVE",      0x40000000>,  // sections to be accessed by Activation Shaves
            ]
        > {
    let cppNamespace = "vpux::ELF";
}

//
// RelocationTypeAttr
//

def ELF_RelocationType :
        ELF_I64EnumAttr<
            "RelocationType",
            "Enum for describing ELF relocation types",
            [
                I64EnumAttrCase<"R_VPU_64",                                             0>,
                I64EnumAttrCase<"R_VPU_64_OR",                                          1>,
                I64EnumAttrCase<"R_VPU_DISP40_RTM",                                     2>,
                I64EnumAttrCase<"R_VPU_64_LSHIFT",                                      3>,
                I64EnumAttrCase<"R_VPU_32",                                             4>,
                I64EnumAttrCase<"R_VPU_32_RTM",                                         5>,
                I64EnumAttrCase<"R_VPU_32_SUM",                                         6>,
                I64EnumAttrCase<"R_VPU_32_MULTICAST_BASE",                              7>,
                I64EnumAttrCase<"R_VPU_32_MULTICAST_BASE_SUB",                          8>,
                I64EnumAttrCase<"R_VPU_DISP28_MULTICAST_OFFSET",                        9>,
                I64EnumAttrCase<"R_VPU_DISP4_MULTICAST_OFFSET_CMP",                     10>,
                I64EnumAttrCase<"R_VPU_LO_21",                                          11>,
                I64EnumAttrCase<"R_VPU_LO_21_SUM",                                      12>,
                I64EnumAttrCase<"R_VPU_LO_21_MULTICAST_BASE",                           13>,
                I64EnumAttrCase<"R_VPU_16_LSB_21_RSHIFT_5",                             14>,
                I64EnumAttrCase<"R_VPU_LO_21_RSHIFT_4",                                 15>,
                I64EnumAttrCase<"R_VPU_CMX_LOCAL_RSHIFT_5",                             16>,
                I64EnumAttrCase<"R_VPU_32_BIT_OR_B21_B26_UNSET",                        17>,
                I64EnumAttrCase<"R_VPU_64_BIT_OR_B21_B26_UNSET",                        18>,
                I64EnumAttrCase<"R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16",                   19>,
                I64EnumAttrCase<"R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_CUSTOM",               20>,
                I64EnumAttrCase<"R_VPU_32_BIT_OR_B21_B26_UNSET_HIGH_16",                21>,
                I64EnumAttrCase<"R_VPU_32_BIT_OR_B21_B26_UNSET_LOW_16",                 22>,
                I64EnumAttrCase<"R_VPU_HIGH_27_BIT_OR",                                 23>,
            ]
        > {
}

def ELF_RelocationTypeAttr : ELF_EnumAttr<ELF_RelocationType, "relocation_type">;

#endif
