//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/common.hpp"

#include "vpux/compiler/core/act_profiling.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <deque>
#include <iterator>
#include <sstream>
#include <string>

namespace vpux {

mlir::IntegerType getActShaveProfilingElementType(mlir::MLIRContext* ctx) {
    return getUInt32Type(ctx);
}

unsigned ActShaveProfiler::getNextBufferId() {
    return uniqBufferId++;
}

void ActShaveProfiler::resetBufferIdCounter() {
    uniqBufferId = 0;
}

ActShaveProfiler::ActShaveProfiler(unsigned clustersNum, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                   vpux::IndexedSymbolAttr memKindAttr, mlir::func::FuncOp netFunc, vpux::Logger& log,
                                   std::shared_ptr<NameUniqifier> uniqifier)
        : _clustersNum(clustersNum),
          _profilingWorkloadSize(VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES),
          _profilingElementSize(VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES /
                                sizeof(uint32_t)),  // How many DWords are needed to store one workload
          _profilingBufferSizes({0}),
          _builder(builder),
          _ctx(ctx),
          _netFunc(netFunc),
          _memKindAttr(memKindAttr),
          _log(log),
          _uniqifier(std::move(uniqifier)) {
}

// Get count of memory needed to store profiling data of all ActShave tasks in the model
unsigned ActShaveProfiler::getRequiredDdrMemory() const {
    unsigned swTasksCount =
            std::accumulate(_swTaskSignatures.begin(), _swTaskSignatures.end(), 0, [](const auto& a, const auto& b) {
                return a + b._maxSubTasks;
            });
    return swTasksCount * _clustersNum * _profilingElementSize;
}

// Go over all SwKernelOps and store required information about those tasks like required size of
// profiling buffer or size of profiling buffer instances
void ActShaveProfiler::scheduleTask(VPUIP::SwKernelOp swOp) {
    const auto taskSignature = getTaskSignature(swOp);

    // How many elements are needed to store profiling data of one task
    const auto maxSwTasks = taskSignature._maxSubTasks;
    const auto requiredMemory = maxSwTasks * _profilingWorkloadSize;
    VPUX_THROW_WHEN(requiredMemory > VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE,
                    "SwKernelOp at '{0}' requires more memory {1} than currently supported. Change  "
                    "HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE.",
                    swOp->getLoc(), requiredMemory);
    _swTaskSignatures.push_back(taskSignature);
    // Trying to reuse last profiling buffer
    const auto currentBufferSize = _profilingBufferSizes.back();
    const auto newBufferSize = currentBufferSize + maxSwTasks;
    _log.trace("Schedule '{0}' operation with '{1}' subtask, op: '{2}'",
               (this->_clustersNum > 1 ? "distributed " : "non-distributed "), maxSwTasks, swOp->getLoc());
    // If we can store profiling result of current task in last buffer without exceeding
    // max size - reuse it, otherwise - scheduling one more
    if (newBufferSize * _profilingWorkloadSize > VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE) {
        _profilingBufferSizes.push_back(maxSwTasks);
    } else {
        _profilingBufferSizes.pop_back();
        _profilingBufferSizes.push_back(newBufferSize);
    }
}

// Main function which goes through all identified ActShave ops and based on gathered data recreates
// those operations to have profiling output with proper slot in profiling buffer instance. When profiling
// buffer is full it also inserts CMX2DDR DMA and allocates new profiling buffer
void ActShaveProfiler::addProfilingOps(mlir::BlockArgument& profilingDdrResult,
                                       SmallVector<mlir::Value>& clusterResults) {
    // Contains profiling_output of individual swTaskOp and count of profiled tiles
    ProfilingResults nceProfilingOutputs;
    size_t currentDDROffset = 0;
    mlir::Operation* currentProfilingBuffer = nullptr;
    unsigned currentBufferSize = 0;
    unsigned currentBufferId = 0;
    const auto allocateProfilingBufferCMX = [&]() {
        if (_profilingBufferSizes.empty()) {
            return;
        }

        currentBufferId = getNextBufferId();
        currentBufferSize = _profilingBufferSizes.front();
        VPUX_THROW_WHEN(currentBufferSize == 0, "Empty CMXBuffers is not allowed");

        _profilingBufferSizes.pop_front();

        const unsigned totalSizeCMXElements = currentBufferSize * _profilingElementSize * _clustersNum;
        const auto locationName =
                std::to_string(_clustersNum) + "_actProfilingSubviewBuffer_" + std::to_string(currentBufferId);

        mlir::OpBuilder::InsertPoint lastInsertionPoint = _builder.saveInsertionPoint();
        _builder.setInsertionPointAfter(&_netFunc.getBody().front().front());

        currentProfilingBuffer = createAllocationOp(totalSizeCMXElements, locationName);

        _builder.restoreInsertionPoint(lastInsertionPoint);
    };

    const auto flushCMX2DDR = [&]() {
        if (nceProfilingOutputs.empty() || currentProfilingBuffer == nullptr) {
            return;
        }
        auto copyToDDRResult =
                copyToDdr(nceProfilingOutputs, currentProfilingBuffer, currentDDROffset, profilingDdrResult);
        clusterResults.push_back(copyToDDRResult);

        auto flushedTasksCount = countTasks(nceProfilingOutputs);
        currentDDROffset += flushedTasksCount;

        nceProfilingOutputs.clear();
    };

    size_t inClusterOffset = 0;
    // Allocate first buffer for storing profiling results
    allocateProfilingBufferCMX();
    for (auto& swTaskSignature : _swTaskSignatures) {
        auto swTaskOp = swTaskSignature._task;

        bool isSingleCluster = !vpux::VPUIP::hasDistributedOperand(swTaskOp);
        _builder.setInsertionPoint(swTaskOp.getOperation());

        const unsigned tasksCount = swTaskSignature._maxSubTasks * _clustersNum;
        auto profilingSamplesInCMX = countTasks(nceProfilingOutputs);
        const auto expectedCMXMemoryUsage = (profilingSamplesInCMX + tasksCount) * _profilingWorkloadSize;
        // If couldnt place current task in the end of cmx buffer flushing all previous tasks to DDR
        // expectedCMXMemoryUsage counts size for all clusters, while HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE only
        // for one so, need to align them for comparison
        if (expectedCMXMemoryUsage > VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE * _clustersNum) {
            flushCMX2DDR();  // Flush current CMX content to DDR
            profilingSamplesInCMX = 0;
            inClusterOffset = 0;
            allocateProfilingBufferCMX();  // Allocate next CMX buffer
        }

        auto subView = getViewToBuffer(currentProfilingBuffer, profilingSamplesInCMX, tasksCount);

        // If we have only one tile - we already know his index, otherwise setting std::nullopt
        std::optional<size_t> maybeTileId = swTaskSignature._maxSubTasks == 1 ? 0 : std::optional<size_t>();
        std::optional<size_t> maybeClusterId = isSingleCluster ? 0 : std::optional<size_t>();
        const auto profilingMeta = getSwProfilingMetaAttr(_ctx, currentBufferId, currentDDROffset, currentBufferSize,
                                                          inClusterOffset, maybeTileId, maybeClusterId);
        const auto uniqLoc = _uniqifier->getUniqueLoc(swTaskOp->getLoc());

        auto profilingOutput = replaceOpWithProfiledOp(swTaskOp, subView, uniqLoc, profilingMeta);

        inClusterOffset += swTaskSignature._maxSubTasks;

        nceProfilingOutputs.push_back({profilingOutput, tasksCount});
    }
    flushCMX2DDR();
}

SWTaskSignature ActShaveProfiler::getTaskSignature(VPUIP::SwKernelOp swOp) const {
    auto numOfProfiledTasks = getNumProfiledTasks(swOp);
    return {swOp, numOfProfiledTasks, {numOfProfiledTasks}};
}

mlir::Type ActShaveProfiler::getTimestampType(int64_t tasksCount) {
    return getMemRefType({_profilingElementSize * tasksCount}, getActShaveProfilingElementType(_ctx), DimsOrder::C,
                         _memKindAttr);
}

// Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
mlir::Value ActShaveProfiler::getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                              int64_t numTasks) {
    const SmallVector<int64_t> sizes({numTasks * _profilingElementSize});
    int offset = profilingSamplesInCMX * _profilingElementSize / _clustersNum;

    _log.trace("Get view to profiling buffer, offset '{0}', size '{1}'", offset, sizes[0]);

    auto subViewLoc =
            appendLoc(currentProfilingBuffer->getLoc(), formatv("_actshaveProfilingSubview_{0}", offset).str());

    auto sub = _builder.create<VPUIP::SubViewOp>(subViewLoc, currentProfilingBuffer->getResult(0),
                                                 SmallVector<int64_t>({static_cast<int>(offset)}), sizes);

    return sub.getResult();
}

// Replace a Actshave task with new one that has profiling output set
mlir::Value ActShaveProfiler::replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask, mlir::Value profilingBuffer,
                                                      mlir::Location loc, VPUIP::SwProfilingMetadataAttr profMeta) {
    _log.trace("Replace op with new profiled task '{0}'", loc);

    auto profilingSlot = profilingBuffer;
    bool isCurrentSwTaskDistributed = vpux::VPUIP::hasDistributedOperand(origSwTask);
    if (this->_clustersNum > 1 && !isCurrentSwTaskDistributed) {
        auto viewOpName = appendLoc(loc, "_view_cast");
        auto viewOp = _builder.create<VPUIP::ViewOp>(viewOpName, getTimestampType(1), profilingBuffer);
        profilingSlot = viewOp.getResult();
    }

    auto swTask = _builder.create<VPUIP::SwKernelOp>(loc, origSwTask, profilingSlot);
    swTask.getProfilingDataMutable().assign(profilingSlot);
    swTask.setProfilingMetadataAttr(profMeta);

    swTask.getRegion().takeBody(origSwTask.getRegion());

    // can't use origSwTask.replaceAllUsesWith() because swTask has 1 more result than origSwTask
    for (unsigned i = 0; i < origSwTask->getNumResults(); i++) {
        origSwTask->getResult(i).replaceAllUsesWith(swTask->getResult(i));
    }

    return swTask.getProfilingOutput();
}

VPUIP::DistributedBufferType ActShaveProfiler::getDistributedBufferType(unsigned totalElements) {
    const auto layout = mlir::AffineMapAttr::get(DimsOrder::C.toAffineMap(_ctx));

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(_ctx, VPU::DistributionMode::SEGMENTED);
    const SmallVector<uint64_t> tiles = {_clustersNum};
    const auto numTiles = getIntArrayAttr(_ctx, tiles);
    const auto numClusters = getIntAttr(_ctx, _clustersNum);
    const auto memKindAttr = IndexedSymbolAttr::get(_memKindAttr.getLeafNameAttr());
    auto distributedTensorAttr = VPU::DistributionInfoAttr::get(
            _ctx, distributionModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr,
            /*uniformDistributedSegments=*/mlir::UnitAttr::get(_ctx), nullptr, nullptr, nullptr, nullptr, nullptr);
    return VPUIP::DistributedBufferType::get(_ctx, {totalElements}, getActShaveProfilingElementType(_ctx), layout,
                                             memKindAttr, distributedTensorAttr);
}

// Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
// new one needs to be allocated. Type of this alloc is a DistributedBufferType
mlir::Operation* ActShaveProfiler::createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) {
    if (this->_clustersNum > 1) {
        const auto bufferType = getDistributedBufferType(totalSizeCMXElements);
        _log.trace("Create new distributed allocation op of type - '{0}'", bufferType);
        return _builder.create<VPURT::AllocDistributed>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                        bufferType, nullptr, nullptr);
    } else {
        auto profBuffType = getMemRefType({totalSizeCMXElements}, getActShaveProfilingElementType(_ctx), DimsOrder::C,
                                          _memKindAttr);

        _log.trace("Create new non-distributed allocation op of type - '{0}'", profBuffType);
        return _builder.create<mlir::memref::AllocOp>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                      profBuffType);
    }
}

// Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
// profiling buffer instance is full or there are no more tasks to profile
mlir::Value ActShaveProfiler::copyToDdr(const ProfilingResults& profilingResults, mlir::Operation* cmxMemOp,
                                        size_t& currentDDROffset, mlir::BlockArgument& profilingDdrResult) {
    SmallVector<mlir::Value> concatInputs;
    int64_t totalNumElements = 0;

    _log.trace("Insert chunk copy to DDR offset '{0}'", currentDDROffset);
    for (auto& profRes : profilingResults) {
        auto profResult = profRes.first;

        totalNumElements += profRes.second;

        const auto isCurrentResultMemref = mlir::isa<mlir::MemRefType>(profResult.getType());
        if (this->_clustersNum > 1 && isCurrentResultMemref) {
            // Result is a plain memref, need to cast back to DistributedBuffer
            auto distType = getDistributedBufferType(profRes.second * _profilingElementSize);
            auto viewLoc = appendLoc(profResult.getLoc(), "_view_cast_to_distributed");
            auto viewOp = _builder.create<VPUIP::ViewOp>(viewLoc, distType, profResult);
            concatInputs.push_back(viewOp.getResult());
        } else {
            concatInputs.push_back(profResult);
        }
    }

    const auto resultType =
            mlir::MemRefType::get({totalNumElements * _profilingElementSize}, getActShaveProfilingElementType(_ctx));

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "actshaveDDR" + std::to_string(currentDDROffset))),
            profilingDdrResult, SmallVector<int64_t>({static_cast<int64_t>(currentDDROffset * _profilingElementSize)}),
            resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(mlir::StringAttr::get(
            _ctx,
            mlir::StringRef("actshave") + profiling::PROFILING_CMX_2_DDR_OP_NAME + std::to_string(currentDDROffset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(
                    _ctx, mlir::StringRef("actshaveProfilingConcat") + std::to_string(currentDDROffset))),
            concatInputs, cmxMemOp->getResult(0));

    auto dmaOp = _builder.create<VPUIP::NNDMAOp>(copyLoc, concatview.getOutput(), subDDR.getResult());
    dmaOp.setProfilingBufferMgmt(true);
    return dmaOp;
}

}  // namespace vpux
