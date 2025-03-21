//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/dpu_profiling.hpp"

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/utils/profiling/common.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <numeric>
#include <sstream>

namespace vpux {

using namespace vpux;

// Return number of used clusters
unsigned getClustersNumber(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    std::set<uint64_t> clusterIds;
    for (auto dpuTask : nceClusterTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        const auto clusterId = dpuTask.getClusterId().value_or(0);
        clusterIds.insert(clusterId);
    }
    return static_cast<unsigned>(clusterIds.size());
}

template <class T>
unsigned countDpuTasks(SmallVector<std::pair<T, unsigned>> vector) {
    return std::accumulate(vector.begin(), vector.end(), 0, [](const auto& a, const auto& b) {
        return a + b.second;
    });
}

mlir::Type BaseClusterBufferScheduler::getTimestampType(unsigned dpuTasksAmount) {
    return getMemRefType({static_cast<int64_t>(_profilingElementSize) * dpuTasksAmount}, getUInt64Type(_ctx),
                         DimsOrder::C, _memKindAttr);
}

unsigned BaseClusterBufferScheduler::getNextBufferId() {
    return uniqBufferId++;
}

void BaseClusterBufferScheduler::resetBufferIdCounter() {
    uniqBufferId = 0;
}

BaseClusterBufferScheduler::BaseClusterBufferScheduler(unsigned clustersNum, unsigned profilingWorkloadSize,
                                                       mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                                       vpux::VPU::MemoryKind memKind, mlir::func::FuncOp netFunc,
                                                       std::shared_ptr<NameUniqifier> uniqifier)
        : _clustersNum(clustersNum),
          _profilingWorkloadSize(profilingWorkloadSize),
          _profilingElementSize(profilingWorkloadSize /
                                sizeof(uint64_t)),  // How many words are need to store one workload
          _profilingBufferSizes({0}),
          _builder(builder),
          _ctx(ctx),
          _netFunc(netFunc),
          _memKindAttr(IndexedSymbolAttr::get(ctx, stringifyEnum(memKind))),
          _uniqifier(std::move(uniqifier)) {
}

unsigned BaseClusterBufferScheduler::getRequiredDdrMemory() const {
    unsigned dpuTasksAmount =
            std::accumulate(_nceTaskSignatures.begin(), _nceTaskSignatures.end(), 0, [](const auto& a, const auto& b) {
                return a + b._maxSubTasks;
            });
    return dpuTasksAmount * _clustersNum * _profilingElementSize;
}

void BaseClusterBufferScheduler::scheduleNceTask(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    const auto taskSignature = getTaskSignature(nceClusterTaskOp);
    const auto maxDpuTasks = taskSignature._maxSubTasks;

    const auto requiredMemory = maxDpuTasks * _profilingWorkloadSize;
    VPUX_THROW_WHEN(requiredMemory > VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE,
                    "NCEClusterTask at '{0}' requires more memory {1} than currently supported. Change  "
                    "HW_DPU_PROFILING_MAX_BUFFER_SIZE.",
                    nceClusterTaskOp->getLoc(), requiredMemory);
    _nceTaskSignatures.push_back(taskSignature);
    // Trying to reuse last profiling buffer
    const auto currentBufferSize = _profilingBufferSizes.back();
    const auto newBufferSize = currentBufferSize + maxDpuTasks;
    // If we can store profiling result of current task in last buffer without exceeding
    // max size - reuse it, otherwise - scheduling one more
    if (newBufferSize * _profilingWorkloadSize > VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE) {
        _profilingBufferSizes.push_back(maxDpuTasks);
    } else {
        _profilingBufferSizes.pop_back();
        _profilingBufferSizes.push_back(newBufferSize);
    }
}

void BaseClusterBufferScheduler::addProfilingOps(unsigned& currentDDROffset, SmallVector<mlir::Value>& clusterResults,
                                                 mlir::BlockArgument& profilingResult) {
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
    unsigned tasksCounter = 0;  // Needed to sort tasks in ascending order. Profiling buffer(i.e. address) of task with
                                // bigger ID goes after task with smaller
    for (auto& nceTaskSignature : _nceTaskSignatures) {
        auto nceTaskOp = nceTaskSignature._task;
        auto* insertionPoint = nceTaskOp.getOperation();
        _builder.setInsertionPoint(insertionPoint);

        const unsigned dpuTasksAmount = nceTaskSignature._maxSubTasks * _clustersNum;
        auto profilingSamplesInCMX = countDpuTasks(nceProfilingOutputs);
        const auto expectedCMXMemoryUsage = (profilingSamplesInCMX + dpuTasksAmount) * _profilingWorkloadSize;
        // If couldnt place current task in the end of cmx buffer flushing all previous tasks to DDR
        // expectedCMXMemoryUsage counts size for all clusters, while HW_DPU_PROFILING_MAX_BUFFER_SIZE only for one
        // so, need to align them for comparison
        if (expectedCMXMemoryUsage > VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE * _clustersNum) {
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

SingleClusterScheduler::SingleClusterScheduler(unsigned profilingWorkloadSize, mlir::OpBuilder& builder,
                                               mlir::MLIRContext* ctx, vpux::VPU::MemoryKind memKind,
                                               mlir::func::FuncOp netFunc, std::shared_ptr<NameUniqifier> uniqifier)
        : BaseClusterBufferScheduler(1, profilingWorkloadSize, builder, ctx, memKind, netFunc, std::move(uniqifier)) {
    if (memKind == VPU::MemoryKind::CMX_NN) {
        _memKindAttr = IndexedSymbolAttr::get(_ctx, stringifyEnum(memKind), 0);
    } else {
        _memKindAttr = IndexedSymbolAttr::get(_ctx, stringifyEnum(memKind));
    }
}

NCETaskSignature SingleClusterScheduler::getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    const auto dpuIt = nceClusterTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>();
    const auto maxDpuTasks = static_cast<unsigned>(std::distance(dpuIt.begin(), dpuIt.end()));
    return {nceClusterTaskOp, maxDpuTasks, {maxDpuTasks}};
}

mlir::Operation* SingleClusterScheduler::createAllocationOp(unsigned totalSizeCMXElements,
                                                            const std::string& location) {
    const auto cmxMemType =
            getMemRefType(ShapeRef(totalSizeCMXElements), getUInt64Type(_ctx), DimsOrder::C, _memKindAttr);
    auto alignmentAttr = _builder.getI64IntegerAttr(_profilingWorkloadSize);
    return _builder.create<mlir::memref::AllocOp>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)), cmxMemType,
                                                  alignmentAttr);
}

mlir::Value SingleClusterScheduler::copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                              SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements,
                                              unsigned offset, StringRef name) {
    const auto resultType = mlir::MemRefType::get(
            {static_cast<int64_t>(numElements) * static_cast<int64_t>(_profilingElementSize)}, getUInt64Type(_ctx));

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "DDR" + std::to_string(offset))), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * _profilingElementSize}), resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc2 = mlir::NameLoc::get(
            mlir::StringAttr::get(_ctx, name + profiling::PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "ProfilingConcat" + std::to_string(offset))),
            dpuProfilingOutputs, cmxMemOp->getResult(0));

    auto dmaOp = _builder.create<VPUIP::NNDMAOp>(copyLoc2, concatview.getOutput(), subDDR);
    dmaOp.setProfilingBufferMgmt(true);
    return dmaOp.getOutput();
}

mlir::Value SingleClusterScheduler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                    unsigned profilingSamplesInCMX, ArrayRef<int64_t> sizes) {
    return _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "dpuProfilingSubview")),
            currentProfilingBuffer->getResult(0),
            SmallVector<int64_t>({static_cast<int64_t>(profilingSamplesInCMX) * _profilingElementSize}), sizes);
}

VPUIP::DistributedBufferType MultiClusterScheduler::getDistributedBufferType(unsigned totalElements) {
    const auto layout = mlir::AffineMapAttr::get(DimsOrder::C.toAffineMap(_ctx));

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(_ctx, VPU::DistributionMode::SEGMENTED);
    const SmallVector<uint64_t> tiles = {_clustersNum};
    const auto numTiles = getIntArrayAttr(_ctx, tiles);
    const auto numClusters = getIntAttr(_ctx, _clustersNum);
    auto distributedTensorAttr = VPU::DistributionInfoAttr::get(
            _ctx, distributionModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr,
            /*uniformDistributedSegments=*/mlir::UnitAttr::get(_ctx), nullptr, nullptr, nullptr, nullptr, nullptr);
    return VPUIP::DistributedBufferType::get(_ctx, {totalElements}, getUInt64Type(_ctx), layout, _memKindAttr,
                                             distributedTensorAttr);
}

mlir::Type MultiClusterScheduler::getDistributedTimestampType(unsigned dpuTasksAmount) {
    return getDistributedBufferType(dpuTasksAmount * _profilingElementSize);
}

NCETaskSignature MultiClusterScheduler::getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    SmallVector<unsigned> dpuTasksPerCluster(_clustersNum, 0);
    unsigned maxTasksInCluster = 0;
    for (auto dpuTask : nceClusterTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        const auto clusterId = dpuTask.getClusterId().value();
        maxTasksInCluster = std::max(maxTasksInCluster, ++dpuTasksPerCluster[clusterId]);
    }
    return {nceClusterTaskOp, maxTasksInCluster, dpuTasksPerCluster};
}

mlir::Operation* MultiClusterScheduler::createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) {
    const auto bufferType = getDistributedBufferType(totalSizeCMXElements);
    auto alignmentAttr = _builder.getI64IntegerAttr(_profilingWorkloadSize);
    return _builder.create<VPURT::AllocDistributed>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                    bufferType, alignmentAttr, nullptr);
}

mlir::Value MultiClusterScheduler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                   unsigned profilingSamplesInCMX, ArrayRef<int64_t> sizes) {
    return _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "dpuProfilingSubview")),
            currentProfilingBuffer->getResult(0),
            SmallVector<int64_t>({static_cast<int64_t>(profilingSamplesInCMX) * _profilingElementSize / _clustersNum}),
            sizes);
}

mlir::Value MultiClusterScheduler::copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                             SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements,
                                             unsigned offset, StringRef name) {
    const auto memorySize = numElements * _profilingElementSize;
    const auto resultTypeDDR = mlir::MemRefType::get({static_cast<int64_t>(memorySize)}, getUInt64Type(_ctx));
    const auto resultTypeDistributed = getDistributedBufferType(memorySize);

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "DDR" + std::to_string(offset))), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * _profilingElementSize}), resultTypeDDR.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(_ctx, name + profiling::PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset)));
    const auto concatLoc =
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "ProfilingConcat" + std::to_string(offset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(concatLoc, resultTypeDistributed, dpuProfilingOutputs,
                                                           cmxMemOp->getResult(0));

    auto dmaOp = _builder.create<VPUIP::NNDMAOp>(copyLoc, concatview.getOutput(), subDDR.getResult());
    dmaOp.setProfilingBufferMgmt(true);
    return dmaOp;
}

}  // namespace vpux
