//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/IRMapping.h>
#include <npu_40xx_nnrt.hpp>
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_UPDATEENQUEUEDMAINPUTANDOUTPUT
#define GEN_PASS_DEF_UPDATEENQUEUEDMAINPUTANDOUTPUT
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;
namespace {
class UpdateEnqueueDMAInputAndOutput :
        public VPUMI40XX::impl::UpdateEnqueueDMAInputAndOutputBase<UpdateEnqueueDMAInputAndOutput> {
public:
    explicit UpdateEnqueueDMAInputAndOutput(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    mlir::Value getOrCreateRegisterBuffer(mlir::OpBuilder& builder, mlir::Operation* bufferInsertionPoint,
                                          mlir::MemRefType memType, uint32_t fifoAddr);
    Const::DeclareOp createEnqueueConstant(mlir::OpBuilder& builder, mlir::Operation* insertionPoint,
                                           const uint32_t& val);

    void updateInputAndOutput(mlir::OpBuilder& builder, VPUMI40XX::NNDMAOp enqueueDma,
                              VPURegMapped::TaskOpInterface taskOp, uint32_t fifoAddr,
                              mlir::Operation* bufferInsertionPoint, mlir::Operation* cstInsertionPoint);

    uint32_t getFifoAddr(VPURegMapped::TaskType type, size_t tile, size_t list);

    llvm::DenseMap<std::pair<mlir::Type, uint32_t>, mlir::Value> _regBufferCache;
    SmallVector<uint32_t> _shvFIFOAddrs;
    SmallVector<uint32_t> _dpuFIFOAddrs;

    uint32_t _shavesCountPerTile = 0;
    uint32_t _tilesCount = 0;
};

uint32_t UpdateEnqueueDMAInputAndOutput::getFifoAddr(VPURegMapped::TaskType type, size_t tile, size_t list) {
    VPUX_THROW_WHEN(tile >= _tilesCount || list > _shavesCountPerTile - 1, "Invalid tile index {0} or list index {1}",
                    tile, list);
    return type == VPURegMapped::TaskType::DPUVariant ? _dpuFIFOAddrs[tile]
                                                      : _shvFIFOAddrs[(_shavesCountPerTile * tile) + list];
}

Const::DeclareOp UpdateEnqueueDMAInputAndOutput::createEnqueueConstant(mlir::OpBuilder& builder,
                                                                       mlir::Operation* insertionPoint,
                                                                       const uint32_t& val) {
    const Shape valShape = {1};
    const auto dataStorageType = mlir::RankedTensorType::get(valShape.raw(), getUInt32Type(builder.getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(val));

    auto memType = mlir::MemRefType::get(dataStorageType.getShape(), dataStorageType.getElementType());
    builder.setInsertionPoint(insertionPoint);
    auto configurationConstOp =
            builder.create<Const::DeclareOp>(builder.getUnknownLoc(), memType, Const::ContentAttr::get(dataAttr));

    return configurationConstOp;
}

mlir::Value UpdateEnqueueDMAInputAndOutput::getOrCreateRegisterBuffer(mlir::OpBuilder& builder,
                                                                      mlir::Operation* bufferInsertionPoint,
                                                                      mlir::MemRefType memType, uint32_t fifoAddr) {
    std::pair<mlir::Type, uint32_t> key = {memType, fifoAddr};

    auto it = _regBufferCache.find(key);
    if (it != _regBufferCache.end()) {
        return it->second;
    }

    builder.setInsertionPoint(bufferInsertionPoint);
    auto declBuf = builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), memType,
                                                          VPURT::BufferSection::Register, fifoAddr);

    mlir::Value buffer = declBuf.getBuffer();
    _regBufferCache[key] = buffer;
    return buffer;
}

// For given enqueue DMA operation create a new enqueue DMA that will have correct input and output buffers and will
// replace original op
void UpdateEnqueueDMAInputAndOutput::updateInputAndOutput(mlir::OpBuilder& builder, VPUMI40XX::NNDMAOp enqueueDma,
                                                          VPURegMapped::TaskOpInterface taskOp, uint32_t fifoAddr,
                                                          mlir::Operation* bufferInsertionPoint,
                                                          mlir::Operation* cstInsertionPoint) {
    auto ctx = builder.getContext();
    auto cmxTaskLocationBuf = mlir::cast<VPUMI40XX::DeclareTaskBufferOp>(taskOp.getTaskLocation().getDefiningOp());

    // ----------------------------------------
    // Step 1: Prepare DMA-specific Attributes
    // ----------------------------------------

    // Convert CMX offset to metadata-relative offset in 32-byte units.
    // 15360 = VPU_METADATA_OFFSET = start of metadata region in CMX.
    const auto descriptorOffsetInCMX =
            (cmxTaskLocationBuf.getOffset().value() + npu40xx::nn_public::VPU_METADATA_OFFSET) >> 5;
    auto enqueueConstOp = createEnqueueConstant(builder, cstInsertionPoint, descriptorOffsetInCMX);
    const auto constOutputType = mlir::cast<vpux::NDTypeInterface>(enqueueConstOp.getOutput().getType());

    const auto layout = mlir::MemRefLayoutAttrInterface{};
    const auto regMemSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::Register));
    auto outputType = constOutputType.changeMemSpace(regMemSpace);

    // ----------------------------------------
    // Step 2: Declare Buffers for Register
    // ----------------------------------------

    builder.setInsertionPoint(bufferInsertionPoint);
    auto memType = mlir::MemRefType::get(outputType.getShape().raw(), outputType.getElementType(), layout,
                                         outputType.getMemSpace());
    mlir::Value dstBuffer = getOrCreateRegisterBuffer(builder, bufferInsertionPoint, memType, fifoAddr);

    // ----------------------------------------
    // Step 3: Create new DMA operation with proper input and output
    // to replace original op that had dummy buffers
    // ----------------------------------------
    mlir::IRMapping mapper;
    mapper.map(enqueueDma.getInput(), enqueueConstOp.getOutput());
    mapper.map(enqueueDma.getOutputBuffs().front(), dstBuffer);

    builder.setInsertionPointAfter(enqueueDma);
    auto newOp = builder.clone(*enqueueDma, mapper);
    auto newEnqueueDma = mlir::cast<VPUMI40XX::NNDMAOp>(newOp);

    // TODO: Check if setting DMA descriptor explicitly is really needed
    const auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx,
                                                                 /*numPlane*/ getIntAttr(ctx, 0),
                                                                 /*len*/ getIntAttr(ctx, 4),
                                                                 /*srcWidth*/ getIntAttr(ctx, 4),
                                                                 /*srcStride*/ getIntAttr(ctx, 0),
                                                                 /*srcPlaneStride*/ getIntAttr(ctx, 0),
                                                                 /*dstWidth*/ getIntAttr(ctx, 4),
                                                                 /*dstStride*/ getIntAttr(ctx, 0),
                                                                 /*dstPlaneStride*/ getIntAttr(ctx, 0));

    newEnqueueDma.setDmaDescriptorAttr(dmaDescriptorAttr);

    enqueueDma.replaceAllUsesWith(newEnqueueDma.getResult());
    enqueueDma.erase();
}

void UpdateEnqueueDMAInputAndOutput::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    _shvFIFOAddrs = config::getConstraint<llvm::SmallVector<uint32_t>>(parentModule, config::SHV_FIFO_ADDRS);
    _dpuFIFOAddrs = config::getConstraint<llvm::SmallVector<uint32_t>>(parentModule, config::DPU_FIFO_ADDRS);

    _tilesCount = config::getTileExecutor(parentModule).getCount();
    _shavesCountPerTile = config::getAvailableExecutor(parentModule, VPU::ExecutorKind::SHAVE_ACT).getCount();

    auto dmaTile0List0Head = mpi.getListHead(VPURegMapped::TaskType::DMA, 0, 0);
    if (!dmaTile0List0Head) {
        return;
    }

    auto builder = mlir::OpBuilder(mpi.getOperation());

    // Set insertion point where new buffers representing HW FIFO register address will be placed
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    // Set insertion point where new constant ops storing task descriptor location will be placed
    auto declOps = netFunc.getOps<Const::DeclareOp>();
    auto cstInsertionPoint = !declOps.empty() ? *declOps.begin() : &netFunc.getBody().front().front();

    // Gather data about all enqueue DMAs which are always on DMA tile 0 list 0 (Port 0, channel DDR)
    auto firstDmaTile0List0Op = dmaTile0List0Head.getDefiningOp<VPUMI40XX::NNDMAOp>();
    auto enqueueDmasPerHwQueue = VPUMI40XX::getEnqueueDmaData(firstDmaTile0List0Op, _log);

    const mlir::DenseSet<std::pair<VPURegMapped::TaskType, uint32_t>> taskTypesWithListCountPerTile = {
            {{VPURegMapped::TaskType::DPUVariant, 1},
             {VPURegMapped::TaskType::ActKernelInvocation, _shavesCountPerTile}}};

    // Iterate over DPU/SHV tasks on each tile and list and check if task is the head of the enqueued task by the
    // enqueue DMA. If yes update input and output buffer of the enqueue DMA so that it will push task
    // descriptor to HW FIFO register.
    for (uint32_t tileIdx = 0; tileIdx < _tilesCount; tileIdx++) {
        for (const auto& [taskType, listCount] : taskTypesWithListCountPerTile) {
            for (uint32_t listIdx = 0; listIdx < listCount; listIdx++) {
                auto listHead = mpi.getListHead(taskType, tileIdx, listIdx);
                if (!listHead) {
                    continue;
                }

                _log.trace("Check enqueue DMAs for task type {0} on tile {1}, list {2}", taskType, tileIdx, listIdx);
                auto taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp());

                auto hwQueue = VPUMI40XX::HwQueueType{taskType, tileIdx, listIdx};
                auto fifoAddr = getFifoAddr(taskType, tileIdx, listIdx);

                VPUX_THROW_WHEN(enqueueDmasPerHwQueue.find(hwQueue) == enqueueDmasPerHwQueue.end(),
                                "No Enqueue DMAs available for task type {0} on tile {1}, list {2}", taskType, tileIdx,
                                listIdx);
                // Initial tasks must be enqueued by first enqueue DMA for given HW queue type
                size_t curEnqueueIndex = 0;
                // Get start and end task range enqueued by enqueue DMA to understand what range of tasks
                // are processed by a single enqueue DMA
                auto [enqueueDmaStartIdx, enqueueDmaEndIdx, enqueueDmaOp] =
                        enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex];
                _log.trace("Enqueue DMA task range: {0} - {1}", enqueueDmaStartIdx, enqueueDmaEndIdx);

                // Iterate over tasks in the list and check if enabling link to previous is possible
                do {
                    auto taskInd = mlir::cast<VPURegMapped::IndexType>(taskOp.getResult().getType()).getValue();

                    // If current task index is greater than end index of current enqueue DMA it means that
                    // this task is enqueued by next enqueue DMA -> switch to next enqueue DMA
                    if (taskInd > enqueueDmaEndIdx) {
                        _log.trace("Task {0} is after end task {1} of current enqueue DMA. Move to next enqueue DMA op",
                                   taskInd, enqueueDmaEndIdx);
                        // Move to next enqueue DMA and get start and end indexes
                        curEnqueueIndex++;
                        VPUX_THROW_UNLESS(
                                curEnqueueIndex < enqueueDmasPerHwQueue[hwQueue].size(),
                                "No enqueue DMAs available for task type {0} on tile {1}, list {2} at index {3}",
                                taskType, tileIdx, listIdx, curEnqueueIndex);
                        enqueueDmaStartIdx = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].startTaskIdx;
                        enqueueDmaEndIdx = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].endTaskIdx;
                        enqueueDmaOp = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].enqDmaOp;

                        _log.trace("Enqueue DMA task range: {0} - {1}", enqueueDmaStartIdx, enqueueDmaEndIdx);
                    }

                    // If task index is same as first task enqueued by a DMA then update enqueue DMA input and
                    // output buffers so that this DMA will push task descriptor to HW FIFO register
                    if (taskInd == enqueueDmaStartIdx) {
                        _log.trace("Update input and output data of enqueue DMA for task {0} ", taskInd);
                        updateInputAndOutput(builder, enqueueDmaOp, taskOp, fifoAddr, bufferInsertionPoint,
                                             cstInsertionPoint);
                    }

                    taskOp = taskOp.getNextTask();
                } while (taskOp);
            }
        }
    }
}

}  // namespace

//
// createUpdateEnqueueDMAInputAndOutput
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUpdateEnqueueDMAInputAndOutput(Logger log) {
    return std::make_unique<UpdateEnqueueDMAInputAndOutput>(log);
}
