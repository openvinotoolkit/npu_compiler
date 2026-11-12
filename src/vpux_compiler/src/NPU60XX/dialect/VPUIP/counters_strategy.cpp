//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIP/counters_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/counters_category.hpp"

namespace vpux::VPUIP::arch60xx {
void CountersStrategy::appendCounters(std::vector<vpux::utils::OpCounterTree::Node>& counters) {
    auto makeModeChecker = [](VPU::MPEEngine60XXMode mode) {
        return [mode](mlir::Operation* op) {
            if (auto nceOp = mlir::dyn_cast_if_present<VPUIP::NCEClusterTaskOp>(op)) {
                if (auto mpeEngine = nceOp.getMpeEngine()) {
                    if (auto attr = mlir::dyn_cast<VPU::MPEEngine60XXAttr>(*mpeEngine)) {
                        return attr.getMode().getValue() == mode;
                    }
                }
            }
            return false;
        };
    };

    CountersVec nestedCountersDCIM, nestedCountersSCL;
    nestedCountersDCIM.push_back(VPUIP::makeCounterNode("opCount", [&](mlir::Operation* op) {
        return VPUIP::isOpCounterSupported(op);
    }));
    counters.push_back(VPUIP::makeCounterNode("DCIM Task Op", makeModeChecker(VPU::MPEEngine60XXMode::DCIM),
                                              std::move(nestedCountersDCIM)));

    nestedCountersSCL.push_back(VPUIP::makeCounterNode("opCount", [&](mlir::Operation* op) {
        return VPUIP::isOpCounterSupported(op);
    }));
    counters.push_back(VPUIP::makeCounterNode("SCL Task Op", makeModeChecker(VPU::MPEEngine60XXMode::SCL),
                                              std::move(nestedCountersSCL)));
}
}  // namespace vpux::VPUIP::arch60xx
