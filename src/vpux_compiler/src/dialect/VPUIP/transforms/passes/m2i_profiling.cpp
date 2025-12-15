//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include "vpux/compiler/core/profiling.hpp"

#include <deque>
#include <memory>
#include <string>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_M2IPROFILING
#define GEN_PASS_DEF_M2IPROFILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

// M2I Profiling for 50XX consists of a 64 byte structure
constexpr uint16_t M2I_PROFILING_SIZE_BYTES_50XX = 64;

//
// M2IProfilingPass
//

class M2IProfilingPass final : public VPUIP::impl::M2IProfilingBase<M2IProfilingPass> {
public:
    explicit M2IProfilingPass(Logger log)
            : _profilingBufferSizes({0}), _uniqifier(log), _currentBufferId(0), _currentBufferSize(0) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(::mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPURT::VPURTDialect>();
    }

private:
    void safeRunOnModule() final;

    // Generate next buffer id
    unsigned getNextBufferId();

    // Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
    // new one needs to be allocated. Type of this alloc is a memref
    mlir::Operation* createAllocationOp(mlir::OpBuilder& builder, unsigned totalSizeCMXElements,
                                        const std::string& location);

    // Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
    // profiling buffer instance is full or there are no more tasks to profile
    mlir::Value copyToDdr(mlir::OpBuilder& builder, ArrayRef<mlir::Value> profilingResults, mlir::Operation* cmxMemOp,
                          size_t& currentDDROffset, mlir::BlockArgument& profilingDdrResult);

    // Get a SubView of profiling buffer instance so that given M2I task is given required chunk of it
    mlir::Value getViewToBuffer(mlir::OpBuilder& builder, mlir::Operation* currentProfilingBuffer,
                                unsigned profilingSamplesInCMX);

    // Replace an M2I task with new one that has profiling output set
    mlir::Value replaceOpWithProfiledOp(mlir::OpBuilder& builder, VPUIP::M2ITaskOp origM2ITask,
                                        mlir::Value profilingBuffer, mlir::Location loc,
                                        VPUIP::M2IProfilingMetadataAttr profMeta);

    void allocateProfilingBufferCMX();

    void flushCMX2DDR(mlir::OpBuilder& builder, SmallVector<mlir::Value>& profilingOutputs,
                      mlir::Operation* currentProfilingBuffer, size_t& currentDDROffset,
                      mlir::BlockArgument profilingResult, SmallVector<mlir::Value>& concatResults);

    std::deque<unsigned> _profilingBufferSizes;
    NameUniqifier _uniqifier;
    unsigned _uniqBufferId = 0;
    unsigned _currentBufferId;
    unsigned _currentBufferSize;
};

void M2IProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(module, netInfo, netFunc);
    const auto arch = config::getArch(module);

    if (arch != config::ArchKind::NPU50XX) {
        return;
    }

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&(netFunc.getFunctionBody()), &builderLog);

    SmallVector<VPUIP::M2ITaskOp> m2iTasks;
    netFunc.walk([&](VPUIP::M2ITaskOp m2iTaskOp) {
        m2iTasks.push_back(m2iTaskOp);

        // Trying to reuse last profiling buffer
        const auto currentBufferSize = _profilingBufferSizes.back();
        const auto newBufferSize = currentBufferSize + 1;
        // If we can store profiling result of current task in last buffer without exceeding
        // max size - reuse it, otherwise - scheduling one more
        if (newBufferSize * M2I_PROFILING_SIZE_BYTES_50XX > VPUIP::HW_M2I_PROFILING_MAX_BUFFER_SIZE) {
            _profilingBufferSizes.push_back(1);
        } else {
            _profilingBufferSizes.pop_back();
            _profilingBufferSizes.push_back(newBufferSize);
        }
    });

    // No M2I tasks in the network, nothing to profile
    if (m2iTasks.empty()) {
        return;
    }

    // Declare and create additional output from network
    const unsigned outputDdrSize = m2iTasks.size() * M2I_PROFILING_SIZE_BYTES_50XX;
    const auto outputResultDdr = mlir::MemRefType::get({outputDdrSize}, getUInt8Type(ctx));
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netInfo, outputResultDdr, profiling::ExecutorType::M2I);

    SmallVector<mlir::Value> concatResults;
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(&netFunc.getBody().front().front());

    // Contains profiling_output of individual taskOp and count of profiled tiles
    SmallVector<mlir::Value> profilingOutputs;
    size_t currentDDROffset = 0;
    mlir::Operation* currentProfilingBuffer = nullptr;
    _currentBufferSize = 0;
    _currentBufferId = 0;

    // Allocate first buffer for storing profiling results
    allocateProfilingBufferCMX();
    currentProfilingBuffer = createAllocationOp(builder, _currentBufferSize * M2I_PROFILING_SIZE_BYTES_50XX,
                                                "m2iProfilingSubviewBuffer_" + std::to_string(_currentBufferId));

    for (auto& m2iTaskOp : m2iTasks) {
        builder.setInsertionPoint(m2iTaskOp);

        auto profilingSamplesInCMX = profilingOutputs.size();
        const auto expectedCMXMemoryUsage = (profilingSamplesInCMX + 1) * M2I_PROFILING_SIZE_BYTES_50XX;
        // If we can't place the current task at the end of cmx buffer flushing all previous tasks to DDR
        // expectedCMXMemoryUsage counts size for all clusters, while HW_M2I_PROFILING_MAX_BUFFER_SIZE only
        // for one so, need to align them for comparison
        if (expectedCMXMemoryUsage > VPUIP::HW_M2I_PROFILING_MAX_BUFFER_SIZE) {
            // Flush current CMX content to DDR
            flushCMX2DDR(builder, profilingOutputs, currentProfilingBuffer, currentDDROffset, profilingResult,
                         concatResults);
            profilingSamplesInCMX = 0;

            // Allocate next CMX buffer
            allocateProfilingBufferCMX();
            currentProfilingBuffer =
                    createAllocationOp(builder, _currentBufferSize * M2I_PROFILING_SIZE_BYTES_50XX,
                                       "m2iProfilingSubviewBuffer_" + std::to_string(_currentBufferId));
        }

        auto subView = getViewToBuffer(builder, currentProfilingBuffer, profilingSamplesInCMX);
        const auto profilingMeta =
                getM2IProfilingMetaAttr(builder.getContext(), _currentBufferId, profilingSamplesInCMX);

        const auto uniqLoc = _uniqifier.getUniqueLoc(m2iTaskOp->getLoc());

        auto profilingOutput = replaceOpWithProfiledOp(builder, m2iTaskOp, subView, uniqLoc, profilingMeta);
        profilingOutputs.push_back(profilingOutput);
    }
    flushCMX2DDR(builder, profilingOutputs, currentProfilingBuffer, currentDDROffset, profilingResult, concatResults);

    builder.restoreInsertionPoint(lastInsertionPoint);

    mlir::func::ReturnOp returnOp =
            mlir::dyn_cast_or_null<mlir::func::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "m2iDDRProfiling")), concatResults, profilingResult);
    returnOp.getOperandsMutable().append(concatview.getOutput());

    // After profiling buffers and M2I tasks with profiling outputs were created - remove old operations
    for (auto& m2iTask : m2iTasks) {
        m2iTask.erase();
    }
}

unsigned M2IProfilingPass::getNextBufferId() {
    return _uniqBufferId++;
}

mlir::Operation* M2IProfilingPass::createAllocationOp(mlir::OpBuilder& builder, unsigned totalSizeCMXElements,
                                                      const std::string& location) {
    auto* ctx = builder.getContext();
    auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    auto profBuffType = getMemRefType({totalSizeCMXElements}, getUInt8Type(ctx), DimsOrder::C, memKindAttr);
    return builder.create<mlir::memref::AllocOp>(mlir::NameLoc::get(mlir::StringAttr::get(ctx, location)),
                                                 profBuffType);
}

mlir::Value M2IProfilingPass::copyToDdr(mlir::OpBuilder& builder, ArrayRef<mlir::Value> profilingResults,
                                        mlir::Operation* cmxMemOp, size_t& currentDDROffset,
                                        mlir::BlockArgument& profilingDdrResult) {
    SmallVector<mlir::Value> concatInputs;
    int64_t totalNumElements = profilingResults.size();
    for (auto& profRes : profilingResults) {
        concatInputs.push_back(profRes);
    }

    auto* ctx = builder.getContext();
    const auto resultType = mlir::MemRefType::get(
            {static_cast<int64_t>(totalNumElements) * M2I_PROFILING_SIZE_BYTES_50XX}, getUInt8Type(ctx));

    auto subDDR = builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "m2iDDR" + std::to_string(currentDDROffset))),
            profilingDdrResult,
            SmallVector<int64_t>({static_cast<int64_t>(currentDDROffset * M2I_PROFILING_SIZE_BYTES_50XX)}),
            resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(mlir::StringAttr::get(
            ctx, mlir::StringRef("m2i") + profiling::PROFILING_CMX_2_DDR_OP_NAME + std::to_string(currentDDROffset)));
    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(
                    ctx, mlir::StringRef("m2iProfilingConcat") + std::to_string(currentDDROffset))),
            concatInputs, cmxMemOp->getResult(0));

    auto dmaOp = builder.create<VPUIP::NNDMAOp>(copyLoc, concatview.getOutput(), subDDR.getResult());
    dmaOp.setProfilingBufferMgmt(true);
    return dmaOp;
}

mlir::Value M2IProfilingPass::getViewToBuffer(mlir::OpBuilder& builder, mlir::Operation* currentProfilingBuffer,
                                              unsigned profilingSamplesInCMX) {
    const SmallVector<int64_t> sizes({M2I_PROFILING_SIZE_BYTES_50XX});
    int offset = profilingSamplesInCMX * M2I_PROFILING_SIZE_BYTES_50XX;

    _log.trace("Get view to profiling buffer, offset '{0}', size '{1}'", offset, sizes[0]);

    auto subViewLoc = appendLoc(currentProfilingBuffer->getLoc(), formatv("_m2iProfilingSubview_{0}", offset).str());

    auto sub = builder.create<VPUIP::SubViewOp>(subViewLoc, currentProfilingBuffer->getResult(0),
                                                SmallVector<int64_t>({static_cast<int>(offset)}), sizes);

    return sub.getResult();
}

mlir::Value M2IProfilingPass::replaceOpWithProfiledOp(mlir::OpBuilder& builder, VPUIP::M2ITaskOp m2iTask,
                                                      mlir::Value profilingBuffer, mlir::Location loc,
                                                      VPUIP::M2IProfilingMetadataAttr profMeta) {
    _log.trace("Replace op with new profiled task '{0}'", loc);

    SmallVector<mlir::Type> newResultTypes(m2iTask.getResultTypes());
    newResultTypes.push_back(profilingBuffer.getType());

    auto newM2iTask = builder.create<VPUIP::M2ITaskOp>(
            loc, newResultTypes, m2iTask.getInput(), m2iTask.getOutputBuff(), m2iTask.getProfilingData(),
            m2iTask.getDoCscAttr(), m2iTask.getDoNormAttr(), m2iTask.getInFmtAttr(), m2iTask.getOutFmtAttr(),
            m2iTask.getChromaInReverseChannelsAttr(), m2iTask.getChromaOutReverseChannelsAttr(),
            m2iTask.getLumaInReverseChannelsAttr(), m2iTask.getLumaOutReverseChannelsAttr(),
            m2iTask.getScaleFactorXAttr(), m2iTask.getScaleFactorYAttr(), m2iTask.getNormAttr(),
            m2iTask.getTileOffsetXAttr(), m2iTask.getTileOffsetYAttr(), m2iTask.getProfilingMetadataAttr(),
            m2iTask.getInterpAttr());

    newM2iTask.setProfilingMetadataAttr(profMeta);
    newM2iTask.getProfilingDataMutable().assign(profilingBuffer);

    m2iTask->getResult(0).replaceAllUsesWith(newM2iTask->getResult(0));

    return newM2iTask.getProfilingOutput();
}

void M2IProfilingPass::allocateProfilingBufferCMX() {
    if (_profilingBufferSizes.empty()) {
        return;
    }

    _currentBufferId = getNextBufferId();
    _currentBufferSize = _profilingBufferSizes.front();
    VPUX_THROW_WHEN(_currentBufferSize == 0, "Empty CMXBuffers is not allowed");

    _profilingBufferSizes.pop_front();
}

void M2IProfilingPass::flushCMX2DDR(mlir::OpBuilder& builder, SmallVector<mlir::Value>& profilingOutputs,
                                    mlir::Operation* currentProfilingBuffer, size_t& currentDDROffset,
                                    mlir::BlockArgument profilingResult, SmallVector<mlir::Value>& concatResults) {
    if (profilingOutputs.empty() || currentProfilingBuffer == nullptr) {
        return;
    }
    auto copyToDDRResult =
            copyToDdr(builder, profilingOutputs, currentProfilingBuffer, currentDDROffset, profilingResult);
    concatResults.push_back(copyToDDRResult);

    auto flushedTasksCount = profilingOutputs.size();
    currentDDROffset += flushedTasksCount;

    profilingOutputs.clear();
}

}  // namespace

//
// createM2IProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createM2IProfilingPass(Logger log) {
    return std::make_unique<M2IProfilingPass>(log);
}
