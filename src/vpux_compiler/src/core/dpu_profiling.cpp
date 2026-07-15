//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/dpu_profiling.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <numeric>
#include <sstream>

namespace vpux {

// Return number of used clusters
unsigned getClustersNumber(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    std::set<uint64_t> clusterIds;
    for (auto dpuTask : nceClusterTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        const auto clusterId = dpuTask.getClusterId().value_or(0);
        clusterIds.insert(clusterId);
    }
    return clusterIds.size();
}

template <class T>
unsigned countDpuTasks(SmallVector<std::pair<T, unsigned>> vector) {
    return std::accumulate(vector.begin(), vector.end(), 0, [](const auto& a, const auto& b) {
        return a + b.second;
    });
}

mlir::Type ClusterBufferScheduler::getTimestampType(unsigned dpuTasksAmount) {
    return getMemRefType({static_cast<int64_t>(_profilingElementSize) * dpuTasksAmount}, getUInt64Type(_ctx),
                         DimsOrder::C, _memKindAttr);
}

unsigned ClusterBufferScheduler::getNextBufferId() {
    return uniqBufferId++;
}

void ClusterBufferScheduler::resetBufferIdCounter() {
    uniqBufferId = 0;
}

ClusterBufferScheduler::ClusterBufferScheduler(unsigned clustersNum, unsigned profilingWorkloadSize,
                                               mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                               IndexedSymbolAttr memKindAttr, mlir::func::FuncOp netFunc,
                                               std::shared_ptr<NameUniqifier> uniqifier)
        : _clustersNum(clustersNum),
          _profilingWorkloadSize(profilingWorkloadSize),
          _profilingElementSize(profilingWorkloadSize /
                                sizeof(uint64_t)),  // How many words are need to store one workload
          _profilingBufferSizes({0}),
          _builder(builder),
          _ctx(ctx),
          _netFunc(netFunc),
          _memKindAttr(memKindAttr),
          _uniqifier(std::move(uniqifier)) {
}

unsigned ClusterBufferScheduler::getRequiredDdrMemory() const {
    unsigned dpuTasksAmount =
            std::accumulate(_nceTaskSignatures.begin(), _nceTaskSignatures.end(), 0, [](const auto& a, const auto& b) {
                return a + b._maxSubTasks;
            });
    return dpuTasksAmount * _clustersNum * _profilingElementSize;
}

void ClusterBufferScheduler::scheduleNceTask(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    const auto taskSignature = getTaskSignature(nceClusterTaskOp);
    const auto maxDpuTasks = taskSignature._maxSubTasks;
    const auto requiredMemory = maxDpuTasks * _profilingWorkloadSize;

    const auto arch = config::getArch(nceClusterTaskOp);
    const auto dpuProfMaxBufferSize = VPUIP::getDPUProfMaxBufferSize(arch);
    VPUX_THROW_WHEN(requiredMemory > dpuProfMaxBufferSize,
                    "NCEClusterTask at '{0}' requires more memory {1} than currently supported. Change  "
                    "DPU profiling max buffer size.",
                    nceClusterTaskOp->getLoc(), requiredMemory);
    _nceTaskSignatures.push_back(taskSignature);
    // Trying to reuse last profiling buffer
    const auto currentBufferSize = _profilingBufferSizes.back();
    const auto newBufferSize = currentBufferSize + maxDpuTasks;
    // If we can store profiling result of current task in last buffer without exceeding
    // max size - reuse it, otherwise - scheduling one more
    if (newBufferSize * _profilingWorkloadSize > dpuProfMaxBufferSize) {
        _profilingBufferSizes.push_back(maxDpuTasks);
    } else {
        _profilingBufferSizes.pop_back();
        _profilingBufferSizes.push_back(newBufferSize);
    }
}

void ClusterBufferScheduler::addProfilingOps(unsigned& currentDDROffset, SmallVector<mlir::Value>& clusterResults,
                                             mlir::BlockArgument& profilingResult, unsigned& tasksCounter) {
    if (getRequiredDdrMemory() == 0) {
        return;
    }
    // Contains profiling_output of individual nceTaskOp and amount of profiled DPU tasks
    SmallVector<std::pair<mlir::Value, unsigned>> nceProfilingOutputs;
    mlir::Operation* currentProfilingBuffer = nullptr;
    unsigned currentBufferId;
    const auto allocateProfilingBufferCMX = [&]() {
        if (_profilingBufferSizes.empty()) {
            return;
        }

        const auto currentBufferSize = _profilingBufferSizes.front();
        VPUX_THROW_WHEN(currentBufferSize == 0, "Empty CMXBuffers is not allowed");

        _profilingBufferSizes.pop_front();

        currentBufferId = getNextBufferId();
        const auto locationName =
                std::to_string(_clustersNum) + "_dpuProfilingSubviewBuffer_" + std::to_string(currentBufferId);

        mlir::OpBuilder::InsertPoint lastInsertionPoint = _builder.saveInsertionPoint();
        _builder.setInsertionPointAfter(&_netFunc.getBody().front().front());

        const unsigned totalSizeCMXElements = currentBufferSize * _profilingElementSize * _clustersNum;
        currentProfilingBuffer = createAllocationOp(totalSizeCMXElements, locationName);

        _builder.restoreInsertionPoint(lastInsertionPoint);
    };

    const auto flushCMX2DDR = [&]() {
        if (nceProfilingOutputs.empty() || currentProfilingBuffer == nullptr) {
            return;
        }

        const auto flushedTasksAmount = countDpuTasks(nceProfilingOutputs);
        SmallVector<mlir::Value> profilingOutputs;
        std::transform(nceProfilingOutputs.begin(), nceProfilingOutputs.end(), std::back_inserter(profilingOutputs),
                       [](const auto& x) {
                           return x.first;
                       });

        clusterResults.push_back(copyToDDR(profilingResult, currentProfilingBuffer, profilingOutputs,
                                           flushedTasksAmount, currentDDROffset, "dpu"));

        profilingOutputs.clear();
        nceProfilingOutputs.clear();
        currentDDROffset += flushedTasksAmount;
    };

    // Allocate first buffer for storing profiling results
    allocateProfilingBufferCMX();
    for (auto& nceTaskSignature : _nceTaskSignatures) {
        auto nceTaskOp = nceTaskSignature._task;
        auto* insertionPoint = nceTaskOp.getOperation();
        _builder.setInsertionPoint(insertionPoint);

        const unsigned dpuTasksAmount = nceTaskSignature._maxSubTasks * _clustersNum;
        auto profilingSamplesInCMX = countDpuTasks(nceProfilingOutputs);
        const auto expectedCMXMemoryUsage = (profilingSamplesInCMX + dpuTasksAmount) * _profilingWorkloadSize;
        const auto arch = config::getArch(nceTaskOp);
        const auto dpuProfMaxBufferSize = VPUIP::getDPUProfMaxBufferSize(arch);
        // If couldnt place current task in the end of cmx buffer flushing all previous tasks to DDR
        // expectedCMXMemoryUsage counts size for all clusters, while dpuProfMaxBufferSize only for one
        // so, need to align them for comparison
        if (expectedCMXMemoryUsage > dpuProfMaxBufferSize * _clustersNum) {
            flushCMX2DDR();  // Flush current CMX content to DDR
            profilingSamplesInCMX = 0;
            allocateProfilingBufferCMX();  // Allocate next CMX buffer
        }

        const SmallVector<int64_t> sizes(
                {static_cast<int64_t>(dpuTasksAmount) * static_cast<int64_t>(_profilingElementSize)});
        auto subView = getViewToBuffer(currentProfilingBuffer, profilingSamplesInCMX, sizes);
        bool isDistributed = vpux::VPUIP::hasDistributedOperand(nceTaskOp);
        mlir::Type timestampType = isDistributed ? subView.getType() : getTimestampType(dpuTasksAmount);

        const auto profAttr = nceTaskSignature.dpuSignature(_ctx, currentBufferId, ++tasksCounter);
        const auto uniqLoc = _uniqifier->getUniqueLoc(nceTaskOp->getLoc());

        _builder.setInsertionPointAfter(nceTaskOp);

        auto newCluster = _builder.create<VPUIP::NCEClusterTaskOp>(uniqLoc, nceTaskOp, timestampType);
        newCluster.setProfilingMetadataAttr(profAttr);

        if (nceTaskOp->hasAttr(DPUCost)) {
            newCluster->setAttr(DPUCost, nceTaskOp->getAttr(DPUCost));
        }
        copyLoopAttributes(nceTaskOp, newCluster);

        for (const auto& region : llvm::enumerate(nceTaskOp.getRegions())) {
            newCluster.getRegion(static_cast<unsigned>(region.index())).takeBody(*region.value());
        }
        newCluster.getProfilingDataMutable().assign(subView);
        SmallVector<mlir::Value> newUses{newCluster.getOutput()};
        if (newCluster.getOutputSparsityMap() != nullptr) {
            newUses.push_back(newCluster.getOutputSparsityMap());
        }
        nceTaskOp->replaceAllUsesWith(mlir::ValueRange(newUses));
        nceTaskOp->erase();
        nceProfilingOutputs.push_back({newCluster.getProfilingOutput(), dpuTasksAmount});
    }
    flushCMX2DDR();
}

NCETaskSignature ClusterBufferScheduler::getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    SmallVector<unsigned> dpuTasksPerCluster(_clustersNum, 0);
    unsigned maxTasksInCluster = 0;
    for (auto dpuTask : nceClusterTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        const auto clusterId = dpuTask.getClusterId().value_or(0);
        maxTasksInCluster = std::max(maxTasksInCluster, ++dpuTasksPerCluster[clusterId]);
    }
    return {nceClusterTaskOp, maxTasksInCluster, std::move(dpuTasksPerCluster)};
}

VPUIP::DistributedBufferType getDistributedBufferType(vpux::IndexedSymbolAttr memKindAttr, mlir::MLIRContext* ctx,
                                                      unsigned clusterNum, unsigned totalElements) {
    const auto layout = mlir::AffineMapAttr::get(DimsOrder::C.toAffineMap(ctx));

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    const SmallVector<uint64_t> tiles = {clusterNum};
    const auto tilesArrayAttribute = getIntArrayAttr(ctx, tiles);
    const auto numClusters = getIntAttr(ctx, clusterNum);
    auto distributedTensorAttr = VPU::DistributionInfoAttr::get(ctx, distributionModeAttr, tilesArrayAttribute, nullptr,
                                                                nullptr, nullptr, numClusters, nullptr,
                                                                /*uniformDistributedSegments=*/mlir::UnitAttr::get(ctx),
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return VPUIP::DistributedBufferType::get(ctx, {totalElements}, getUInt64Type(ctx), layout, memKindAttr,
                                             distributedTensorAttr);
}

mlir::Operation* ClusterBufferScheduler::createAllocationOp(unsigned totalSizeCMXElements,
                                                            const std::string& location) {
    auto alignmentAttr = _builder.getI64IntegerAttr(_profilingWorkloadSize);
    if (_clustersNum > 1) {
        const auto bufferType = getDistributedBufferType(_memKindAttr, _ctx, _clustersNum, totalSizeCMXElements);
        return _builder.create<VPURT::AllocDistributed>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                        bufferType, alignmentAttr, nullptr);
    } else {
        const auto cmxMemType =
                getMemRefType(ShapeRef(totalSizeCMXElements), getUInt64Type(_ctx), DimsOrder::C, _memKindAttr);
        return _builder.create<mlir::memref::AllocOp>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                      cmxMemType, alignmentAttr);
    }
}

mlir::Value ClusterBufferScheduler::copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                              SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements,
                                              unsigned offset, StringRef name) {
    const auto memorySize = numElements * _profilingElementSize;
    const auto resultType = vpux::getMemRefType({static_cast<int64_t>(memorySize)}, getUInt64Type(_ctx));

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "DDR" + std::to_string(offset))), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * _profilingElementSize}), resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(_ctx, name + profiling::PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset)));
    const auto concatLoc =
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "ProfilingConcat" + std::to_string(offset)));
    VPUIP::ConcatViewOp concatView;
    if (_clustersNum > 1) {
        const auto resultTypeDistributed = getDistributedBufferType(_memKindAttr, _ctx, _clustersNum, memorySize);
        concatView = _builder.create<VPUIP::ConcatViewOp>(concatLoc, resultTypeDistributed, dpuProfilingOutputs,
                                                          cmxMemOp->getResult(0));
    } else {
        concatView = _builder.create<VPUIP::ConcatViewOp>(concatLoc, dpuProfilingOutputs, cmxMemOp->getResult(0));
    }

    auto dmaOp = _builder.create<VPUIP::NNDMAOp>(copyLoc, concatView.getOutput(), subDDR.getResult());
    dmaOp.setProfilingBufferMgmt(true);
    return dmaOp;
}

mlir::Value ClusterBufferScheduler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                    unsigned profilingSamplesInCMX, ArrayRef<int64_t> sizes) {
    return _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "dpuProfilingSubview")),
            currentProfilingBuffer->getResult(0),
            SmallVector<int64_t>({static_cast<int64_t>(profilingSamplesInCMX) * _profilingElementSize / _clustersNum}),
            sizes);
}

}  // namespace vpux
