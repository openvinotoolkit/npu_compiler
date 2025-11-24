//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_OPS
#define VPUX_COMPILER_DIALECT_CONFIG_OPS

include "vpux/compiler/dialect/config/dialect.td"
include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/config/ops_interfaces.td"

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Config_Op<string mnemonic, list<Trait> traits = []> :
    Op<Config_Dialect, mnemonic, traits>;

def Config_PipelineOptionsOp :
        Config_Op<
            "PipelineOptions",
            [
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Parallel to config options used in the compiler to be exposed globally";

    let description = [{
        Options can include another PipelineOptionsOp or OptionsOp
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$options
    );

    let assemblyFormat = [{
        attr-dict
        $sym_name
        custom<OptionalBlockRegion>($options)
    }];
}

def Config_OptionOp :
        Config_Op<
            "Option",
            [
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Option to be added to the model/IR";

    let description = [{
        Concrete option values that will be added to IR, must have a sym_name for lookup
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        AnyAttrOf<[IntAttr, F32Attr, BoolAttr, StrAttr, ArrayAttr]>:$optionValue
    );

    let assemblyFormat = [{
        $sym_name `:` $optionValue
        attr-dict
    }];
}

def Config_MemoryResourceOp :
        Config_Op<
            "MemoryResource",
            [
                Config_ResourceOpInterface,
                Symbol
            ]
        > {
    let summary = "Information about memory resource";

    let description = [{
        The memory resource is defined by the following attributes:

          * sym_name - kind of memory space.
          * byteSize - size in bytes of memory space.
          * offset - optional offset attribute if this resource is part of larger memory space
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$byteSize,
        OptionalAttr<IntAttr>:$offset
    );

    let extraClassDeclaration = [{
        vpux::Byte size() { return vpux::Byte(getByteSize()); }

        static mlir::StringRef getDefaultDialect() {
            return "config";
        }
    }];

    let assemblyFormat = [{
        $byteSize `bytes` `of` $sym_name (`offset` $offset^)?
        attr-dict
    }];
}

def Config_ExecutorResourceOp :
        Config_Op<
            "ExecutorResource",
            [
                Config_ResourceOpInterface,
                Config_ComputeResourceOpInterface,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about executor resource";

    let description = [{
        The executor resource is defined by the following attributes:

          * sym_name - kind of the executor.
          * count - number of executor units.
          * activity_factor - power activity factor.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$count,
        OptionalAttr<F64Attr>:$activity_factor
    );

    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() {
            return "config";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name
    }];

    let hasVerifier = 1;
}

def Config_ResourcesOp :
        Config_Op<
            "Resources",
            [
                Config_ResourceOpInterface,
                Config_ComputeResourceOpInterface,
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Wrapper for resources: memory and executors";

    let description = [{
        config.ResourcesOp is a container for resources:
          * Available memory
          * Reserved memory
          * Executors

        Typically 2 instances of this operation are present in IR:
          * tile resources - tile-specific items, e.g. DPUs, CMX
          * global resources - everything else, e.g. DDR memory

        Attributes:
          * sym_name - kind of the resource.
          * count - number of executor units.
          * proc_freq - processor frequency (in MHz).
          * activity_factor - power activity factor.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DefaultValuedAttr<IntAttr, "1">:$count,
        OptionalAttr<F64Attr>:$proc_freq,
        OptionalAttr<F64Attr>:$activity_factor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<
            (ins "::llvm::StringRef":$sym_name, CArg<"int64_t", "1">:$count, CArg<"std::optional<double>", "std::nullopt">:$proc_freq, CArg<"std::optional<double>", "std::nullopt">:$activity_factor), [{
                auto& properties = $_state.getOrAddProperties<Properties>();
                properties.sym_name = $_builder.getStringAttr(sym_name);
                properties.count = $_builder.getIntegerAttr($_builder.getI64Type(), count);

                if (proc_freq) {
                    properties.proc_freq = ::mlir::FloatAttr::get($_builder.getF64Type(), proc_freq.value());
                }

                if (activity_factor) {
                    properties.activity_factor = ::mlir::FloatAttr::get($_builder.getF64Type(), activity_factor.value());
                }

                // SymbolTable requires exactly one block
                $_state.addRegion()->emplaceBlock();
            }]
        >
    ];

    let regions = (region
        SizedRegion<1>:$subExecutors
    );

    let extraClassDeclaration = [{
        mlir::FloatAttr getProcessorFrequency() {
            return getProcFreqAttr();
        }

        void setProcessorFrequency(mlir::FloatAttr procFreqAttr) {
            setProcFreqAttr(procFreqAttr);
        }

        bool hasProcessorFrequency() {
            return getProcFreqAttr() != nullptr;
        }

        vpux::config::ExecutorResourceOp addSubExecutor(mlir::SymbolRefAttr executor, size_t count);
        bool hasSubExecutor(mlir::SymbolRefAttr executor);
        vpux::config::ExecutorResourceOp getSubExecutor(mlir::SymbolRefAttr executor);
        vpux::config::MemoryResourceOp addAvailableMemory(mlir::SymbolRefAttr memSpace, Byte size);

        bool hasAvailableMemory(mlir::SymbolRefAttr memSpace);

        vpux::config::MemoryResourceOp getAvailableMemory(mlir::SymbolRefAttr memSpace);

        template <typename Enum, typename OutT = vpux::config::ExecutorResourceOp>
        using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum, typename OutT = vpux::config::MemoryResourceOp>
        using mem_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        exec_resource_if<Enum> addSubExecutor(Enum kind, size_t count) {
            return addSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)), count);
        }

        template <typename Enum, typename OutT = bool>
        using bool_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        bool_if<Enum> hasSubExecutor(Enum kind) {
            return hasSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        exec_resource_if<Enum> getSubExecutor(Enum kind) {
            return getSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        mem_resource_if<Enum> addAvailableMemory(Enum kind, Byte size) {
            return addAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)), size);
        }

        template <typename Enum>
        bool_if<Enum> hasAvailableMemory(Enum kind) {
            return hasAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        mem_resource_if<Enum> getAvailableMemory(Enum kind) {
            return getAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        static mlir::StringRef getDefaultDialect() {
            return "config";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name (`at` $proc_freq^`MHz`)?
        custom<OptionalBlockRegion>($subExecutors)
    }];

    let hasVerifier = 1;
}

#endif
