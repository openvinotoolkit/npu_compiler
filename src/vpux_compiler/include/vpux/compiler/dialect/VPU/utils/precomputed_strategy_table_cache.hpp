//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

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

}  // namespace vpux::VPU

namespace vpux::VPU {

/// Applies strategies from a binary PTC table to all NCE/SW ops in func.
void applyPrecomputedStrategyTableBinary(ArrayRef<uint8_t> buf, mlir::func::FuncOp func, Logger log);

}  // namespace vpux::VPU
