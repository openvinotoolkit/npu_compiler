//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <memory>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTTILEWITHELTWISETONCEPOOL
#define GEN_PASS_DEF_CONVERTTILEWITHELTWISETONCEPOOL
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class ConvertTileWithEltwiseToNCEPoolPass final :
        public VPU::impl::ConvertTileWithEltwiseToNCEPoolBase<ConvertTileWithEltwiseToNCEPoolPass> {
public:
    explicit ConvertTileWithEltwiseToNCEPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto func = getOperation();
        auto& ctx = getContext();

        const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
        auto strategy = strategyFactory->getConvertTileWithEltwiseToNCEPoolStrategy(&ctx);

        SmallVector<mlir::RewritePatternSet> patternSets;
        strategy->addPatterns(patternSets, _log);
        for (auto& patternSet : patternSets) {
            collectOpsAndApplyPatterns(func, std::move(patternSet));
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertTileWithEltwiseToNCEPoolPass(Logger log) {
    return std::make_unique<ConvertTileWithEltwiseToNCEPoolPass>(log);
}
