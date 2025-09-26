//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Pass/PassOptions.h>
#include "vpux/utils/core/string_ref.hpp"

#include <string>

namespace vpux {

using IntOption = mlir::detail::PassOptions::Option<int>;
using Int64Option = mlir::detail::PassOptions::Option<int64_t>;
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;
using DoubleOption = mlir::detail::PassOptions::Option<double>;
enum class WorkloadManagementMode { PWLM_V0_LCA = 0, PWLM_V1_BARRIER_FIFO = 1, PWLM_V2_PAGES = 2 };
enum class AllocateShaveStackFrames { ENABLED = 0, DISABLED = 1 };
enum class WorkloadManagementBarrierProgrammingMode {
    LEGACY = 0,
    NO_BARRIER_DMAS_SCHEDULED = 1,
    INITIAL_BARRIER_DMAS_SCHEDULED = 2,
    ALL_BARRIER_DMAS_SCHEDULED = 3,
    UNKNOWN = 255
};
enum class DMAFifoType { SW = 0, HW = 1 };

/**
 * @brief This enum is used to specify the mode of weights table reuse.
 *
 * It can be set to ENABLED, VF_ENABLED, or DISABLED.
 * ENABLED means that the weights table can be reused for all operations that support it.
 * VF_ENABLED means that the weights table can be reused for operations in pure-vertical-fusion region, to avoid
 * possible memory fragmentation. DISABLED means that the weights table cannot be reused.
 */
enum class WeightsTableReuseMode { ENABLED = 0, VF_ENABLED = 1, DISABLED = 2 };

StringLiteral stringifyEnum(WorkloadManagementBarrierProgrammingMode val);
StringLiteral stringifyEnum(DMAFifoType val);
StringLiteral stringifyEnum(WeightsTableReuseMode val);
std::optional<std::string> convertToOptional(const StrOption& strOption);
StringLiteral stringifyEnum(WorkloadManagementMode val);
bool isOptionEnabled(const BoolOption& option);
}  // namespace vpux
