
//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <optional>

using namespace vpux;

namespace {

using S = vpux::VPU::MultiClusterStrategy;

};
std::optional<mlir::DenseMap<size_t, std::optional<VPU::MultiClusterStrategy>>> vpux::maybeGetStrategyFor(StringRef) {
    return std::nullopt;
}

bool vpux::isStrategyPreConfigured(StringRef) {
    return false;
}
