//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/JSON.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

#include <cstdint>

namespace vpux::VPU {

/// Attribute name (UnitAttr) set on every op whose strategy was explicitly pinned by the
/// precomputed strategy table.  Pass skip-guards key on this marker rather than on the
/// functional strategy attributes so that any later pass that default-initialises
/// multiClusterStrategy or tilingStrategy does not accidentally suppress assignment.
constexpr StringLiteral pinnedStrategy = "pinnedStrategy";

/// Per-context holder for the architecture-specific built-in binary precomputed strategy table.
///
/// One instance lives inside SingletonCache (a per-MLIRContext dialect interface),
/// so concurrent compilations for different architectures each have their own isolated
/// copy and cannot overwrite each other's builtin table.
class PrecomputedStrategyTable {
public:
    // Defined out-of-line so Coverity recognises it as user-provided (C.21).
    PrecomputedStrategyTable() = default;
    ~PrecomputedStrategyTable();
    PrecomputedStrategyTable(const PrecomputedStrategyTable&) = delete;
    PrecomputedStrategyTable& operator=(const PrecomputedStrategyTable&) = delete;
    PrecomputedStrategyTable(PrecomputedStrategyTable&&) = delete;
    PrecomputedStrategyTable& operator=(PrecomputedStrategyTable&&) = delete;

    /// Registers an architecture-specific built-in binary precomputed strategy table that
    /// is embedded into the binary at build time.  Called once per context from the
    /// pipeline builder before any pass runs.
    void setBuiltinTable(const uint8_t* data, std::size_t size);

    /// Returns the built-in table registered via setBuiltinTable(), or an empty ArrayRef if
    /// none has been registered.  The returned ArrayRef is valid for the lifetime of the process.
    ArrayRef<uint8_t> getBuiltinTable() const;

private:
    // Non-owning view into the embedded static blob set by setBuiltinTable().
    // Empty when no builtin table has been registered.
    ArrayRef<uint8_t> _builtinTable;
};

// Keys used in the precomputed strategy table JSON schema.
constexpr StringLiteral ptcSpatialTiling = "spatial_tiling";
constexpr StringLiteral ptcTemporalTiling = "temporal_tiling";
constexpr StringLiteral ptcPipelineMode = "pipeline_mode";

/// Applies strategies from a binary PTC table to all NCE/SW ops in func.
void applyPrecomputedStrategyTableBinary(ArrayRef<uint8_t> buf, mlir::func::FuncOp func, Logger log);

/// Populates json with one entry per NCE/SW op in func that was decided to need temporal tiling
/// (i.e. has a tilingStrategy attribute). Each entry key is the 16-character hex FNV-1a descriptor
/// hash; the value holds the full descriptor (shapes, layout, element type, strides, padding) plus
/// spatial_tiling, temporal_tiling, and pipeline_mode fields.  Merges into any existing entries
/// already present in json so incremental population across multiple functions is supported.
void createPrecomputedStrategyTableJSON(llvm::json::Value& json, mlir::func::FuncOp func);

/// Reads the existing JSON file at path (if any), merges entries from func, and writes the
/// result back.  No-op unless built with VPUX_DEVELOPER_BUILD.
/// Reads the output path from VPUX_PTC_WRITE_LOCATION (default: "precomputed_strategy_table.json")
/// and only runs if VPUX_PTC_WRITE_ENABLED=1.
void writePrecomputedStrategyTableJSON(mlir::func::FuncOp func);

}  // namespace vpux::VPU
