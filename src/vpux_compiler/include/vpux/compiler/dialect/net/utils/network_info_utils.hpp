//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::net {

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

}  // namespace vpux::net
