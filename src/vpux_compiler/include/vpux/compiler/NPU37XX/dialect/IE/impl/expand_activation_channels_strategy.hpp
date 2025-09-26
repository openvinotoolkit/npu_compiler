//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/expand_activation_channels_strategy.hpp"

namespace vpux::IE::arch37xx {

class ExpandActivationChannelsStrategy final : public vpux::IE::IExpandActivationChannelsStrategy {
public:
    ExpandActivationChannelsStrategy(bool _seOpsEnabled, const Logger& log)
            : IExpandActivationChannelsStrategy(_seOpsEnabled, log) {
    }

    void addTargets(mlir::ConversionTarget& target) override;

    void addPatterns(mlir::RewritePatternSet& patterns) override;

};  // class ExpandActivationChannelsStrategy

}  // namespace vpux::IE::arch37xx
