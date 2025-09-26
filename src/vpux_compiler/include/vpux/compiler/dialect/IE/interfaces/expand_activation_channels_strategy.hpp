//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {

class IExpandActivationChannelsStrategy {
public:
    IExpandActivationChannelsStrategy(bool _seOpsEnabled, const Logger& log): _seOpsEnabled(_seOpsEnabled), _log(log) {
    }

    virtual ~IExpandActivationChannelsStrategy() = default;

    virtual void addTargets(mlir::ConversionTarget& target) = 0;

    virtual void addPatterns(mlir::RewritePatternSet& patterns) = 0;

protected:
    bool _seOpsEnabled;
    const Logger& _log;

};  // class IExpandActivationChannelsStrategy

}  // namespace vpux::IE
