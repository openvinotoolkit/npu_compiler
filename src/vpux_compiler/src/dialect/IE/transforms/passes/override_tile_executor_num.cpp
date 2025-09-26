//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/batch.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OVERRIDETILEEXECUTORNUM
#define GEN_PASS_DEF_OVERRIDETILEEXECUTORNUM
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// OverrideTileExecutorNumPass
//

class OverrideTileExecutorNumPass final : public IE::impl::OverrideTileExecutorNumBase<OverrideTileExecutorNumPass> {
public:
    explicit OverrideTileExecutorNumPass(const vpux::IE::DebatcherOpReorderingOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);
    }

    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

private:
    void safeRunOnModule() final;
    const std::vector<std::string> _supportedModes = {"apply", "revert"};
};

mlir::LogicalResult OverrideTileExecutorNumPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    if (!overrideToTilesPerBatchMode.hasValue()) {
        _log.error("Missing overriding mode.");
        return mlir::failure();
    }

    const auto modeStr = overrideToTilesPerBatchMode.getValue();
    if (std::find(_supportedModes.begin(), _supportedModes.end(), modeStr) == _supportedModes.end()) {
        _log.error("Unknown overriding mode: '{0}'", modeStr);
        return mlir::failure();
    }

    return mlir::success();
}

//
// safeRunOnModule
//

void OverrideTileExecutorNumPass::safeRunOnModule() {
    _log.debug("{0}::safeRunOnModule", getName());
    _log.nest(1).debug("Mode: {0}", overrideToTilesPerBatchMode);

    auto module = getOperation();

    VPUX_THROW_UNLESS(config::hasTileExecutor(module), "Expected module to have 'IE::TileResourceOp'.");

    auto tileOp = config::getTileExecutor(module);
    const auto origTileCount = static_cast<DebatchedCallOpData::ValueType>(tileOp.getCount());

    std::vector<DebatchedCallOpData::ValueType> valuesToCheck;

    const auto areAllValuesSame = [](const std::vector<DebatchedCallOpData::ValueType>& vec) -> bool {
        return std::adjacent_find(vec.begin(), vec.end(), std::not_equal_to<>()) == vec.end();
    };

    if (overrideToTilesPerBatchMode == "apply") {
        module.walk([&](mlir::func::CallOp callOp) {
            VPUX_THROW_UNLESS(!DebatchedCallOpAttributeView::hasAvailableTilesAttr(callOp),
                              "Detected existing overridden process, unable to override without reverting.");

            const auto debatchedCallOpAttrView = DebatchedCallOpAttributeView::extract(callOp);

            if (debatchedCallOpAttrView.has_value()) {
                const auto batchNum = debatchedCallOpAttrView.value().getCallData().getBatchSize();

                VPUX_THROW_UNLESS(batchNum > 1, "Expected batch number to be greater than 1, but got '{0}'.", batchNum);
                VPUX_THROW_UNLESS(
                        origTileCount % batchNum == 0,
                        "Unsupported configuration: The tile count '{0}' is not a multiple of the batch number '{1}'.",
                        origTileCount, batchNum);

                // Track original 'count' with 'available_tiles' attribute
                DebatchedCallOpAttributeView::setAvailableTilesAttr(callOp, origTileCount);

                valuesToCheck.push_back(batchNum);
            }
        });

        // Skip the pass if there is no batching detected
        if (valuesToCheck.empty()) {
            _log.nest(1).debug("No batching detected, skipping the pass.");
            return;
        }

        // NOTE: It's assumed that currently only supports a single or same batch number from all the 'CallOp's
        VPUX_THROW_UNLESS(areAllValuesSame(valuesToCheck), "Expected all 'CallOp's to have the same batch number.");

        const auto singleBatchNum = valuesToCheck[0];

        // Set 'count' to 'tilesPerBatch'
        tileOp.setCount(static_cast<int64_t>(origTileCount / singleBatchNum));

        _log.nest(2).debug(
                "Updated 'TileResourceOp' tile count from old: {0}, to new: {1} - based on batch request count: {2}",
                origTileCount, tileOp.getCount(), singleBatchNum);
    } else if (overrideToTilesPerBatchMode == "revert") {
        module.walk([&](mlir::func::CallOp callOp) {
            if (DebatchedCallOpAttributeView::hasAvailableTilesAttr(callOp)) {
                valuesToCheck.push_back(DebatchedCallOpAttributeView::getAvailableTilesVal(callOp));

                // Remove to allow the next overriding process
                DebatchedCallOpAttributeView::removeAvailableTilesAttr(callOp);
            }
        });

        // Skip the pass if there is no batching detected
        if (valuesToCheck.empty()) {
            _log.nest(1).debug("No batching detected, skipping the pass.");
            return;
        }

        VPUX_THROW_UNLESS(areAllValuesSame(valuesToCheck),
                          "Expected all 'CallOp's to have the same 'available_tiles' values.");

        // Reset 'count' to 'available_tiles'
        tileOp.setCount(static_cast<int64_t>(valuesToCheck[0]));

        _log.nest(2).debug("Reverted 'TileResourceOp' tile count from old: {0}, to new: {1}", origTileCount,
                           tileOp.getCount());
    }
}

}  // namespace

//
// createOverrideTileExecutorNumPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOverrideTileExecutorNumPass(
        const vpux::IE::DebatcherOpReorderingOptions& options, Logger log) {
    return std::make_unique<OverrideTileExecutorNumPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createOverrideTileExecutorNumPass(Logger log) {
    return createOverrideTileExecutorNumPass(vpux::IE::DebatcherOpReorderingOptions{}, log);
}

//
// createRevertTileExecutorNumPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRevertTileExecutorNumPass(Logger log) {
    vpux::IE::DebatcherOpReorderingOptions options{};
    options.overideToTilesPerBatchMode = "revert";
    return createOverrideTileExecutorNumPass(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createRevertTileExecutorNumPass(
        const vpux::IE::DebatcherOpReorderingOptions& options, Logger log) {
    auto revertOptions = options;
    revertOptions.overideToTilesPerBatchMode = "revert";
    return createOverrideTileExecutorNumPass(revertOptions, log);
}
