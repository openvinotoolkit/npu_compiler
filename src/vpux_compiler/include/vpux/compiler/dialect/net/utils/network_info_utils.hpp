//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <utility>

namespace vpux::net {

/** @brief Get NetworkInfoOp from the immediate parent module of an operation.

    Finds the nearest parent ModuleOp and returns the NetworkInfoOp within it.
    The NetworkInfoOp must exist in the same module as the operation - this function
    does NOT search parent modules. For operations inside nested modules (e.g., @VPU.SW)
    that don't contain their own NetworkInfoOp, callers should pass an operation from
    the appropriate module level.

    @param op Operation inside the target module, or the ModuleOp itself.
    @throws VPUX_THROW if op is nullptr, parent ModuleOp not found, or exactly one
            NetworkInfoOp is not present in the module.
 */
net::NetworkInfoOp getNetworkInfo(mlir::Operation* op);

/** @brief Get the main entry point function from the immediate parent module.

    Finds the nearest parent ModuleOp and returns the entry point function defined
    in the NetworkInfoOp. The NetworkInfoOp and entry point must exist in the same
    module as the operation.

    @param op Operation inside the target module, or the ModuleOp itself.
    @throws VPUX_THROW if NetworkInfoOp or entry point function is not found.
 */
mlir::func::FuncOp getMainFunc(mlir::Operation* op);

/** @brief Get both NetworkInfoOp and main function from the immediate parent module.

    Enables structured binding: auto [netInfo, netFunc] = net::getFromModule(op);

    Finds the nearest parent ModuleOp and returns both the NetworkInfoOp and entry
    point function. Both must exist in the same module as the operation.

    @param op Operation inside the target module, or the ModuleOp itself.
    @throws VPUX_THROW if NetworkInfoOp or entry point function is not found.
 */
std::pair<net::NetworkInfoOp, mlir::func::FuncOp> getFromModule(mlir::Operation* op);

/** @brief Utility that sets up basic sections of the net::NetworkInfoOp.

    The function creates net::NetworkInfoOp sections if they are not created
    already. The sections are: inputsInfo, outputsInfo, profilingOutputsInfo.
 */
void setupSections(net::NetworkInfoOp netInfo, bool enableProfiling = false);

/** @brief Utility that erases entries from the net::NetworkInfoOp section.

    The function erases the entries (namely, net::DataInfoOp operations) within
    the specified section. The removal starts from @a begin.
 */
void eraseSectionEntries(mlir::Region& section, size_t begin = 0);

// Remove the utility function after pipeline issues are resolved in E#168311

/** @brief Utility function for HostCompile pipeline to fetch entry point function.

    The function safely returns an entry point function from network info object.
    If the entry point function is not found, it returns nullptr.
    Track: E#168311
 */
mlir::func::FuncOp findEntryPointFunc(mlir::Operation* op, Logger& log);

bool isArgStrided(mlir::ModuleOp moduleOp, size_t argIndex);

/** @brief Extract per-dimension upper bounds from a DataInfoOp list entry.

    Looks up the DataInfoOp at the given index, inspects its declared
    Core::BoundedTensorType user type, and returns the bounds array.
    Returns an empty vector when the entry is missing or has no BoundedTensorType
    (i.e. the buffer is purely static).

    @param dataInfos  Ordered list of DataInfoOp values from one I/O section.
    @param index      Zero-based position within the list.
 */
SmallVector<int64_t> getBoundsFromDataInfo(ArrayRef<net::DataInfoOp> dataInfos, size_t index);

}  // namespace vpux::net
