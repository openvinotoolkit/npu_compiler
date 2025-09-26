//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/expand_activation_channels_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/expand_activation_channels_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE {

std::unique_ptr<IExpandActivationChannelsStrategy> createExpandActivationChannelsStrategy(mlir::func::FuncOp funcOp,
                                                                                          bool _seOpsEnabled,
                                                                                          Logger& _log) {
    const auto arch = config::getArch(funcOp);

    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
        return std::make_unique<arch37xx::ExpandActivationChannelsStrategy>(_seOpsEnabled, _log);
    default:
        _log.error("Unsupported architecture: {0}", arch);
        VPUX_THROW("Unable to get ExpandActivationChannelsStrategy for architecture {0}", arch);
    }
}

}  // namespace vpux::IE
