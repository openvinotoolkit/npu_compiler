//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/counters_strategy.hpp"

namespace vpux::VPUIP::arch60xx {

class CountersStrategy : public VPUIP::ICountersStrategy {
public:
    virtual void appendCounters(std::vector<vpux::utils::OpCounterTree::Node>& counters) override;
};

}  // namespace vpux::VPUIP::arch60xx
