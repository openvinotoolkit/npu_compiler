//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Pass/PassOptions.h>

#include <string>

namespace vpux {

using IntOption = mlir::detail::PassOptions::Option<int>;
using Int64Option = mlir::detail::PassOptions::Option<int64_t>;
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;
using DoubleOption = mlir::detail::PassOptions::Option<double>;
enum class WorkloadManagementMode { PWLM_V0_LCA = 0, PWLM_V1_BARRIER_FIFO = 1 };
enum class AllocateShaveStackFrames { ENABLED = 0, DISABLED = 1 };

std::optional<int> convertToOptional(const IntOption& intOption);
std::optional<std::string> convertToOptional(const StrOption& strOption);
bool isOptionEnabled(const BoolOption& option);

}  // namespace vpux
