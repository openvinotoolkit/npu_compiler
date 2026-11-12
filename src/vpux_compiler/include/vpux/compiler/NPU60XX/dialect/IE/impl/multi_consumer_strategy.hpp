//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/multi_consumer_strategy.hpp"

namespace vpux::IE::arch60xx {

class MultiConsumerStrategy final : public IE::IMultiConsumerStrategy {
public:
    MPEConsumerAnalysis getMPEEngineMappableConsumers(mlir::Operation* op) const override;
};

}  // namespace vpux::IE::arch60xx
