//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Operation.h>

#include <cstdint>

namespace mlir {
class PatternRewriter;
}  // namespace mlir

namespace vpux::IE {

enum class MPEConsumerEligibility { ALL, NONE, MIXED };

struct MPEConsumerAnalysis final {
    MPEConsumerEligibility eligibility = MPEConsumerEligibility::NONE;
    SmallVector<mlir::Operation*> mappableDirectUsers;
    SmallVector<mlir::Operation*> mappableTerminalOps;
    int64_t terminalMappableUserCount = 0;
    bool suppressViewLikePathSplitting = false;
};

class IMultiConsumerStrategy {
public:
    virtual ~IMultiConsumerStrategy() = default;

    // Collects the direct users of the given op that the target's specialized MPE engine can map, plus the number of
    // terminal mappable users found through transparent view-like chains.
    virtual MPEConsumerAnalysis getMPEEngineMappableConsumers(mlir::Operation* op) const = 0;
};

MPEConsumerAnalysis collectMPEEngineMappableConsumersThroughViewLikes(
        mlir::Operation* op, llvm::function_ref<bool(mlir::Operation*)> isTerminalMappable);

bool canFakeQuantizeFoldUpstream(IE::FakeQuantizeOp fqOp, bool seOpsEnabled = false, Logger log = Logger::global());

bool splitMPEEngineMappableConsumers(mlir::PatternRewriter& rewriter, mlir::Operation* op, bool seOpsEnabled = false,
                                     Logger log = Logger::global());

MPEConsumerAnalysis classifyMPEEngineMappableConsumers(mlir::Operation* op, bool seOpsEnabled = false,
                                                       Logger log = Logger::global());

}  // namespace vpux::IE
