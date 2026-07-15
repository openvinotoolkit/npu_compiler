//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <set>
#include <utility>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLSHAVECACHEOPS
#define GEN_PASS_DEF_UNROLLSHAVECACHEOPS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

constexpr StringLiteral cacheFlushInvalidateFuncName{"cache_flush_invalidate"};
constexpr StringLiteral cacheFlushFuncName{"cache_flush"};

namespace {
class UnrollShaveCacheOps final : public VPUIP::impl::UnrollShaveCacheOpsBase<UnrollShaveCacheOps> {
public:
    explicit UnrollShaveCacheOps(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollShaveCacheOps::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::SwKernelOp origOp) {
        auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();
        mlir::OpBuilder builder(origOp);
        OpBuilderLogger builderLog(_log);
        auto ctx = builder.getContext();

        auto kernelFuncSym = origOp.getKernelFunction();

        if (auto swKernelFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(kernelFuncSym)) {
            auto swKernelTaskType = swKernelFuncOp->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");

            if (!VPUIP::isCacheOpTaskType(swKernelTaskType)) {
                return;
            }

            _log.trace("Start unrolling cache op: {0}", origOp);
            auto vpurtTask = origOp->getParentOfType<VPURT::TaskOp>();
            auto updateBarriers = vpurtTask.getUpdateBarriers();
            auto waitBarriers = vpurtTask.getWaitBarriers();
            std::set<std::pair<size_t, size_t>> unrolledIndices;

            auto origCacgheOpTileIndex = origOp.getTileIndex();
            auto origCacheOpListIndex = origOp.getListIndex().value_or(0);
            VPUX_THROW_UNLESS(origCacgheOpTileIndex.has_value(), "CacheOp {0} has no tile index.", origOp);
            // check ops before and after cache to find shave indices which require shave cache insertion
            // need to check users of wait barriers and if we have SwKernelOp user we need to store tile index and
            // list index
            auto collectSwKernelIndices = [&](auto barriers) {
                for (auto barrier : barriers) {
                    for (auto user : barrier.getUsers()) {
                        if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(user)) {
                            if (auto swKernel = mlir::dyn_cast<VPUIP::SwKernelOp>(userTaskOp.getInnerTaskOp())) {
                                auto swKernelTileIndex = swKernel.getTileIndex();
                                auto swKernelListIndex = swKernel.getListIndex().value_or(0);
                                VPUX_THROW_UNLESS(swKernelTileIndex.has_value(), "CacheOp {0} has no tile index.",
                                                  swKernel);
                                unrolledIndices.insert({swKernelTileIndex.value(), swKernelListIndex});
                            }
                        }
                    }
                }
            };

            collectSwKernelIndices(waitBarriers);
            collectSwKernelIndices(updateBarriers);

            unrolledIndices.erase(
                    {origCacgheOpTileIndex.value(), origCacheOpListIndex});  // remove original cache op indices
            auto insertAfter = vpurtTask;
            for (auto& [tileIndex, listIndex] : unrolledIndices) {
                auto taskTypeVal = VPU::symbolizeActShaveTaskType(swKernelTaskType.getLeafReference().strref());
                VPUX_THROW_UNLESS(taskTypeVal.has_value(), "VPU::ActShaveTaskType has no value.");
                auto funcName = cacheFlushInvalidateFuncName;
                switch (taskTypeVal.value()) {
                case VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE:
                    funcName = cacheFlushInvalidateFuncName;
                    break;
                case VPU::ActShaveTaskType::CACHE_FLUSH:
                    funcName = cacheFlushFuncName;
                    break;
                case VPU::ActShaveTaskType::CACHE_INVALIDATE:
                    VPUX_THROW("Cache invalidate op is not supported for unrolling");
                default:
                    VPUX_THROW("Unsupported VPU::ActShaveTaskType '{0}'", taskTypeVal.value());
                }
                builder.setInsertionPointAfter(insertAfter);
                SmallVector<mlir::Value> buffers = {};
                const auto buffersRange = mlir::ValueRange(buffers);
                auto newLoc = appendLoc(origOp->getLoc(), "unrolled_cache_tile_{0}_list_{1}", tileIndex, listIndex);
                auto functionSymbol =
                        createCacheHandlingFunction(ctx, builderLog, _log, origOp, funcName, taskTypeVal.value());
                auto cachePrefetchSwKernel = vpux::VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                        builder, waitBarriers, updateBarriers, newLoc, buffersRange, buffersRange, nullptr,
                        functionSymbol, getIntAttr(builder, tileIndex));

                cachePrefetchSwKernel.setListIndexAttr(getIntAttr(builder, listIndex));

                const mlir::SmallVector<mlir::Attribute> args = {};
                vpux::VPUIP::initSwKernel(cachePrefetchSwKernel, buffersRange, buffersRange, args, _log.nest(),
                                          /*swKernelRunOp=*/nullptr);
                insertAfter = cachePrefetchSwKernel->getParentOfType<VPURT::TaskOp>();
                _log.trace("Insert new cacheOp: {0} for tile = {1}, list = {2}", cachePrefetchSwKernel, tileIndex,
                           listIndex);
            }
        }
    });
}
}  // namespace

//
// createSplitDMAToBalanceLoadPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollShaveCacheOpsPass(Logger log) {
    return std::make_unique<UnrollShaveCacheOps>(log);
}
