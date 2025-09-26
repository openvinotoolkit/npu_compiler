//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/expand_activation_channels_strategy.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

std::unique_ptr<IExpandActivationChannelsStrategy> createExpandActivationChannelsStrategy(mlir::func::FuncOp funcOp,
                                                                                          bool _seOpsEnabled,
                                                                                          Logger& log);

}  // namespace vpux::IE
