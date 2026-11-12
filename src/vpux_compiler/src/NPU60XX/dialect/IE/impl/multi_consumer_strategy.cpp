//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/multi_consumer_strategy.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/utils/mpe_engine_info_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"

using namespace vpux;

IE::MPEConsumerAnalysis IE::arch60xx::MultiConsumerStrategy::getMPEEngineMappableConsumers(mlir::Operation* op) const {
    if (op == nullptr) {
        return {};
    }

    return IE::collectMPEEngineMappableConsumersThroughViewLikes(op, [](mlir::Operation* terminalOp) {
        auto mpeEngineInfo = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(terminalOp);
        if (mpeEngineInfo == nullptr || !mpeEngineInfo.supportsSpecializedMPE()) {
            return false;
        }
        return arch60xx::isMappableToDCIM(terminalOp, /*checkChannelAlignment=*/true);
    });
}
