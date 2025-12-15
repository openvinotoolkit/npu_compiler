//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_CONVERTFETCHDMASTOFETCHTASKOPS
#define GEN_PASS_DEF_CONVERTFETCHDMASTOFETCHTASKOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

struct FetchDMAKey {
    int64_t tile;
    int64_t list;
    int64_t group;
    VPURegMapped::TaskType taskType;

    bool operator==(const FetchDMAKey& other) const {
        return tile == other.tile && list == other.list && group == other.group && taskType == other.taskType;
    }
};

namespace llvm {
template <>
struct DenseMapInfo<FetchDMAKey> {
    static inline FetchDMAKey getEmptyKey() {
        return {DenseMapInfo<int64_t>::getEmptyKey(), DenseMapInfo<int64_t>::getEmptyKey(),
                DenseMapInfo<int64_t>::getEmptyKey(), static_cast<VPURegMapped::TaskType>(-1)};
    }

    static inline FetchDMAKey getTombstoneKey() {
        return {DenseMapInfo<int64_t>::getTombstoneKey(), DenseMapInfo<int64_t>::getTombstoneKey(),
                DenseMapInfo<int64_t>::getTombstoneKey(), static_cast<VPURegMapped::TaskType>(-2)};
    }

    static unsigned getHashValue(const FetchDMAKey& key) {
        return hash_combine(key.tile, key.list, key.group,
                            static_cast<std::underlying_type_t<VPURegMapped::TaskType>>(key.taskType));
    }

    static bool isEqual(const FetchDMAKey& lhs, const FetchDMAKey& rhs) {
        return lhs == rhs;
    }
};
}  // namespace llvm

namespace {
class ConvertFetchDmasToFetchTaskOpsPass :
        public VPUMI40XX::impl::ConvertFetchDmasToFetchTaskOpsBase<ConvertFetchDmasToFetchTaskOpsPass> {
public:
    explicit ConvertFetchDmasToFetchTaskOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    llvm::DenseMap<FetchDMAKey, VPUMI40XX::NNDMAOp> _placeHolderFetchDMAMap;
};

VPURegMapped::TaskType convertTargetToTaskType(VPU::ExecutorKind kind) {
    VPURegMapped::TaskType returnType;
    switch (kind) {
    case VPU::ExecutorKind::DPU:
        returnType = VPURegMapped::TaskType::DPUInvariant;
        break;
    case VPU::ExecutorKind::SHAVE_ACT:
        returnType = VPURegMapped::TaskType::ActKernelRange;
        break;
    default:
        VPUX_THROW("Unsupported executor kind passed for FetchTask");
    }

    return returnType;
}

mlir::LogicalResult addFetchTasks(VPUMI40XX::MappedInferenceOp mpi, const VPURegMapped::TaskType taskType,
                                  const int64_t tilesCount,
                                  llvm::DenseMap<FetchDMAKey, VPUMI40XX::NNDMAOp>& placeHolderFetchDMAMap,
                                  SmallVector<VPURegMapped::FetchTaskOp>& fetchTasks, const int64_t listsCount = 1) {
    auto ctx = mpi.getContext();
    auto builder = mlir::OpBuilder(mpi);

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < listsCount; listIdx++) {
            auto startingInvValue = mpi.getListHead(taskType, tileIdx, listIdx);
            // theoretically there can be cases where we run for 6 tiles, but only 4 tiles have Variants associated
            if (!startingInvValue) {
                continue;
            }

            auto currentGroup =
                    mlir::dyn_cast_or_null<VPURegMapped::ExecutionGroupOp>(startingInvValue.getDefiningOp());
            if (!currentGroup) {
                continue;
            }

            int64_t groupIdx = 0;

            while (currentGroup) {
                FetchDMAKey searchKey{tileIdx, listIdx, groupIdx, taskType};
                if (!placeHolderFetchDMAMap.contains(searchKey)) {
                    VPUX_THROW("Placeholder FetchDMA not found for {0} {1} {2} {3}", tileIdx, listIdx, groupIdx,
                               taskType);
                }
                auto insertionDma = placeHolderFetchDMAMap[searchKey];
                builder.setInsertionPointAfter(insertionDma.getOperation());

                auto wlmPageAttr = groupIdx < 2 ? mlir::IntegerAttr::get(getInt64Type(ctx), static_cast<uint64_t>(-1))
                                                : insertionDma.getWlmPageAttr();

                auto fetchTaskOp = builder.create<VPURegMapped::FetchTaskOp>(
                        currentGroup.getLoc(), insertionDma.getIndexType(), insertionDma.getWaitBarriers(),
                        insertionDma.getUpdateBarriers(), insertionDma.getPreviousTask(),
                        currentGroup.getStartIndexes()[0], currentGroup.getEndIndexes()[0],
                        currentGroup.getStartIndexes()[1], currentGroup.getEndIndexes()[1],
                        VPURegMapped::TaskTypeAttr::get(ctx, taskType),
                        mlir::IntegerAttr::get(getUInt64Type(ctx), tileIdx),
                        mlir::IntegerAttr::get(getUInt64Type(ctx), groupIdx), wlmPageAttr);

                // set the previousIdx to the fetchOp
                insertionDma.getResult().replaceAllUsesWith(fetchTaskOp.getResult());
                if (insertionDma->use_empty()) {
                    insertionDma->erase();
                }

                fetchTasks.push_back(fetchTaskOp);
                currentGroup = VPUMI40XX::getNextGroup(currentGroup);
                ++groupIdx;
            }
        }
    }

    return mlir::success();
}

void ConvertFetchDmasToFetchTaskOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();
    const auto shavesCountPerTile = config::getAvailableExecutor(parentModule, VPU::ExecutorKind::SHAVE_ACT).getCount();

    auto mpi = VPUMI40XX::getMPI(netFunc);

    const size_t DMA_DDR2CMX_LISTIDX = 0;
    const size_t DMA_WLM_TILEIDX = 0;  // all WLM dma's should be on tile0 for now;

    auto dmaTaskOps = netFunc.getOps<VPUMI40XX::NNDMAOp>();

    _log.trace("Get placeholder Fetch DMAs");
    for (auto dmaOp : llvm::make_early_inc_range(llvm::make_filter_range(dmaTaskOps, [](auto dma) {
             return dma.getFetchDmaAttr() != nullptr;
         }))) {
        auto fetchAttr = dmaOp.getFetchDmaAttr();

        const auto tileIdx = fetchAttr.getTileIdx().getValue().getSExtValue();
        const auto listIdx = fetchAttr.getListIdx().getValue().getSExtValue();
        const auto groupIdx = fetchAttr.getExecGroupIdx().getValue().getSExtValue();
        const auto targetExecutorKind = fetchAttr.getTargetExecutorKindAttr();

        FetchDMAKey key{tileIdx, listIdx, groupIdx, convertTargetToTaskType(targetExecutorKind.getValue())};
        _placeHolderFetchDMAMap[key] = dmaOp;
    }

    _log.trace("Add Fetch Tasks");
    SmallVector<VPURegMapped::FetchTaskOp> fetchTasks;
    if (mlir::failed(addFetchTasks(mpi, VPURegMapped::TaskType::DPUInvariant, tilesCount, _placeHolderFetchDMAMap,
                                   fetchTasks))) {
        config::setWorkloadManagementStatus(parentModule, WorkloadManagementStatus::FAILED);
        signalPassFailure();
        return;
    }
    if (mlir::failed(addFetchTasks(mpi, VPURegMapped::TaskType::ActKernelRange, tilesCount, _placeHolderFetchDMAMap,
                                   fetchTasks, shavesCountPerTile))) {
        config::setWorkloadManagementStatus(parentModule, WorkloadManagementStatus::FAILED);
        signalPassFailure();
        return;
    }

    _log.trace("Reindex list");
    auto firstFetchIt =
            std::min_element(fetchTasks.begin(), fetchTasks.end(), [](mlir::Operation* lhs, mlir::Operation* rhs) {
                auto lhsDma = mlir::cast<VPURegMapped::FetchTaskOp>(lhs);
                auto rhsDma = mlir::cast<VPURegMapped::FetchTaskOp>(rhs);
                return lhsDma.getType().getValue() < rhsDma.getType().getValue();
            });

    if (firstFetchIt == fetchTasks.end()) {
        return;
    }

    auto listHead = mpi.getListHead(VPURegMapped::TaskType::DMA, DMA_WLM_TILEIDX, DMA_DDR2CMX_LISTIDX);
    if (!listHead) {
        return;
    }

    VPUMI40XX::reindexList(mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp()));
    return;
}

}  // namespace

//
// createConvertFetchDmasToFetchTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createConvertFetchDmasToFetchTaskOpsPass(Logger log) {
    return std::make_unique<ConvertFetchDmasToFetchTaskOpsPass>(log);
}
