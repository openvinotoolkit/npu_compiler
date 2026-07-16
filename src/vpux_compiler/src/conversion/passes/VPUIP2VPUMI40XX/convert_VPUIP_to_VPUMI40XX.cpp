//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/nce_cluster_task_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/sw_kernel_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/task_rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FileSystem.h>

#include <cstdint>
#include <vector>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUIP2VPUMI40XX
#define GEN_PASS_DEF_CONVERTVPUIP2VPUMI40XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpux::vpuip2vpumi40xx;

namespace {

// Enumerate ops with sequential indices and chain TaskOpInterface ops in a single pass.
// Merges the previously separate enumerateOperations and chainTasksInLists to avoid
// two full block traversals. E#146741
void enumerateAndChainOperations(mlir::func::FuncOp funcOp) {
    llvm::SmallDenseMap<std::tuple<mlir::OperationName, uint32_t, uint32_t>, uint32_t> counters;
    llvm::SmallDenseMap<std::tuple<VPURegMapped::TaskType, uint32_t, uint32_t>, mlir::Value> lastTaskInList;

    // take op by l-value non-const reference as single "auto"
    // deduces mlir::Operation that calls deleted copy ctor
    for (auto& op : funcOp.getOps()) {
        if (!op.hasTrait<VPUMI40XX::SingleOutputAsIndexOp>()) {
            continue;
        }
        auto result = op.getResult(0);

        auto originalIndex = mlir::cast<VPURegMapped::IndexType>(result.getType());
        assert(originalIndex.getValue() == 0);

        auto enumKey = std::make_tuple(op.getName(), originalIndex.getTileIdx(), originalIndex.getListIdx());
        auto newIndex = VPURegMapped::IndexType::get(op.getContext(), originalIndex.getTileIdx(),
                                                     originalIndex.getListIdx(), counters[enumKey]++);
        result.setType(newIndex);

        // Chain tasks in-place during enumeration
        if (auto task = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(&op)) {
            assert(!task.getPreviousTask());
            auto chainKey = std::make_tuple(task.getTaskType(), originalIndex.getTileIdx(), originalIndex.getListIdx());
            auto& previousTask = lastTaskInList[chainKey];
            if (previousTask) {
                task.setPreviousTask(previousTask);
            }
            previousTask = result;
        }
    }
}

void finalizeBarriersLegalization(mlir::func::FuncOp funcOp) {
    struct BarrierCounts {
        uint8_t producerCount = 0;
        uint8_t consumerCount = 0;
    };

    mlir::DenseMap<mlir::Value, BarrierCounts> counts;

    // Iterate tasks directly; wait/update operand lists already classify barrier role,
    // so no dyn_cast, enqueue filtering, or llvm::is_contained scans are needed
    for (auto task : funcOp.getOps<VPUMI40XX::ExecutableTaskOpInterface>()) {
        const auto increment = task.getBarrierHitsCount();
        for (auto barrier : task.waitBarriers()) {
            counts[barrier].consumerCount += increment;
        }
        for (auto barrier : task.updateBarriers()) {
            counts[barrier].producerCount += increment;
        }
    }

    for (auto barrier : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        const auto& [producerCount, consumerCount] = counts[barrier.getResult()];
        assert(producerCount > 0 || consumerCount > 0);
        barrier.setProducerCount(producerCount);
        barrier.setConsumerCount(consumerCount);
    }
}

void replaceReturnOpWithOpRanges(mlir::func::FuncOp funcOp) {
    auto* context = funcOp->getContext();

    struct RangeInfo {
        mlir::Value begin;
        mlir::Value end;
        mlir::Attribute taskTypeAttr;
    };

    using RangeKey = std::tuple<VPURegMapped::TaskType, uint32_t, uint32_t>;
    llvm::MapVector<RangeKey, RangeInfo> ranges;

    for (auto taskOp : funcOp.getOps<VPURegMapped::TaskOpInterface>()) {
        const auto index = taskOp.getIndexType();
        auto key = std::make_tuple(taskOp.getTaskType(), index.getTileIdx(), index.getListIdx());
        auto result = taskOp.getResult();

        auto [it, inserted] = ranges.try_emplace(key, RangeInfo{});
        auto& [begin, end, taskTypeAttr] = it->second;
        if (inserted) {
            begin = result;
            taskTypeAttr = VPURegMapped::TaskTypeAttr::get(context, taskOp.getTaskType());
        }
        // Always update end; iteration order matches index order after enumerateOperations,
        // so the last task seen per range has the highest index
        end = result;
    }

    SmallVector<mlir::Attribute> rangesTaskTypesAttrs;
    SmallVector<mlir::Value> rangesBegins;
    SmallVector<mlir::Value> rangesEnds;
    rangesTaskTypesAttrs.reserve(ranges.size());
    rangesBegins.reserve(ranges.size());
    rangesEnds.reserve(ranges.size());

    for (const auto& [_, info] : ranges) {
        rangesTaskTypesAttrs.push_back(info.taskTypeAttr);
        rangesBegins.push_back(info.begin);
        rangesEnds.push_back(info.end);
    }

    assert(funcOp.getBlocks().size() == 1);
    auto* returnOp = funcOp.getBlocks().front().getTerminator();
    assert(returnOp);

    mlir::OpBuilder builder(returnOp);
    builder.create<VPUMI40XX::OpRanges>(returnOp->getLoc(), mlir::ArrayRef(rangesBegins), mlir::ArrayRef(rangesEnds),
                                        mlir::ArrayAttr::get(context, rangesTaskTypesAttrs));

    returnOp->erase();
}

void createProfilingMetadataOp(mlir::func::FuncOp funcOp, Logger log) {
    auto ctx = funcOp.getContext();
    auto moduleOp = getModuleOp(funcOp);

    auto netInfo = net::getNetworkInfo(moduleOp);

    if (netInfo.getProfilingOutputsInfo().empty()) {
        return;
    }

    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    auto buffer = vpux::buildProfilingMetadataBuffer(netInfo, funcOp, log);
    llvm::ArrayRef<char> rawMetadata{reinterpret_cast<const char*>(buffer.data()), buffer.size()};
    long int bufferSize = buffer.size();

    auto vectorType = mlir::VectorType::get({bufferSize}, getUInt8Type(ctx));
    const auto elemAttr = mlir::DenseElementsAttr::getFromRawBuffer(vectorType, rawMetadata);
    auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);
    builderFunc.create<VPUMI40XX::ProfilingMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType, elemAttr);
}

struct MappedInferenceTaskInfo {
    SmallVector<SmallVector<mlir::Value>> dmaHeads;
    SmallVector<SmallVector<mlir::Value>> actKernelRangeHeads;
    SmallVector<SmallVector<mlir::Value>> actKernelInvocationHeads;
    SmallVector<mlir::Value> invariantHeads;
    SmallVector<mlir::Value> variantHeads;
    SmallVector<SmallVector<int64_t>> dmaCount;
    SmallVector<int64_t> invariantCount;
    SmallVector<int64_t> variantCount;
    SmallVector<SmallVector<int64_t>> rangeCount;
    SmallVector<SmallVector<int64_t>> invocationCount;
    mlir::Value barrierTasks = nullptr;
    int64_t barrierCount = 0;
    bool hasInvocations = false;
};

MappedInferenceTaskInfo collectMappedInferenceTaskInfo(mlir::func::FuncOp funcOp, size_t tileCount, size_t dmaTileCount,
                                                       size_t shavesPerTileCount, size_t dmaDirectionRank) {
    MappedInferenceTaskInfo info;
    info.dmaHeads.assign(dmaTileCount, SmallVector<mlir::Value>(dmaDirectionRank));
    info.actKernelRangeHeads.assign(tileCount, SmallVector<mlir::Value>(shavesPerTileCount));
    info.actKernelInvocationHeads.assign(tileCount, SmallVector<mlir::Value>(shavesPerTileCount));
    info.invariantHeads.assign(tileCount, mlir::Value());
    info.variantHeads.assign(tileCount, mlir::Value());
    info.dmaCount.assign(dmaTileCount, SmallVector<int64_t>(dmaDirectionRank, 0));
    info.invariantCount.assign(tileCount, 0);
    info.variantCount.assign(tileCount, 0);
    info.rangeCount.assign(tileCount, SmallVector<int64_t>(shavesPerTileCount, 0));
    info.invocationCount.assign(tileCount, SmallVector<int64_t>(shavesPerTileCount, 0));

    // Use getOps instead of walk to avoid recursing into nested regions (e.g. DPUInvariantOp).
    // Dispatch via TaskType enum switch instead of TypeSwitch dyn_cast chain.
    for (auto taskOp : funcOp.getOps<VPURegMapped::TaskOpInterface>()) {
        const auto indexType = taskOp.getIndexType();
        const auto tileIdx = indexType.getTileIdx();
        const auto listIdx = indexType.getListIdx();
        auto result = taskOp.getResult();

        switch (taskOp.getTaskType()) {
        case VPURegMapped::TaskType::DMA:
            assert(tileIdx < dmaTileCount && listIdx < dmaDirectionRank);
            info.dmaCount[tileIdx][listIdx]++;
            if (!info.dmaHeads[tileIdx][listIdx]) {
                info.dmaHeads[tileIdx][listIdx] = result;
            }
            break;
        case VPURegMapped::TaskType::DPUInvariant:
            assert(tileIdx < tileCount);
            info.invariantCount[tileIdx]++;
            if (!info.invariantHeads[tileIdx]) {
                info.invariantHeads[tileIdx] = result;
            }
            break;
        case VPURegMapped::TaskType::DPUVariant:
            assert(tileIdx < tileCount);
            info.variantCount[tileIdx]++;
            if (!info.variantHeads[tileIdx]) {
                info.variantHeads[tileIdx] = result;
            }
            break;
        case VPURegMapped::TaskType::ActKernelRange:
            assert(tileIdx < tileCount && listIdx < shavesPerTileCount);
            info.rangeCount[tileIdx][listIdx]++;
            if (!info.actKernelRangeHeads[tileIdx][listIdx]) {
                info.actKernelRangeHeads[tileIdx][listIdx] = result;
            }
            break;
        case VPURegMapped::TaskType::ActKernelInvocation:
            assert(tileIdx < tileCount && listIdx < shavesPerTileCount);
            info.invocationCount[tileIdx][listIdx]++;
            info.hasInvocations = true;
            if (!info.actKernelInvocationHeads[tileIdx][listIdx]) {
                info.actKernelInvocationHeads[tileIdx][listIdx] = result;
            }
            break;
        default:
            break;
        }
    }

    // ConfigureBarrierOp does not implement TaskOpInterface, count separately
    for (auto barrier : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        if (!info.barrierTasks) {
            info.barrierTasks = barrier.getResult();
        }
        info.barrierCount++;
    }

    return info;
}

std::pair<mlir::Value, SmallVector<mlir::Value>> setupActKernelRt(
        mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::OpBuilder& builderFunc,
        AllocateDDRStackFrames createDDRStacks = AllocateDDRStackFrames::DISABLED) {
    constexpr auto ACT_RT_CODE_BUFFER_SIZE = (1_MB).to<vpux::Byte>().count();

    // check for actShaveRt info
    mlir::Value actShvRt;
    auto vpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");
    VPUX_THROW_UNLESS(vpuSwModuleOp != nullptr, "setupActKernelConfig: @VPU.SW module missing.");
    auto runtimeKernelFunction = vpuSwModuleOp.lookupSymbol<mlir::func::FuncOp>("runtime");

    // check for actShave stacks info
    auto swRtOpRange = moduleOp.getOps<VPURT::SWRunTimeOp>();
    SmallVector<mlir::Value> shaveStacks;
    if (!swRtOpRange.empty() && createDDRStacks == AllocateDDRStackFrames::ENABLED) {
        VPUX_THROW_WHEN(std::distance(swRtOpRange.begin(), swRtOpRange.end()) > 1,
                        "More than 1 instance of VPURT.SW.Runtime");
        auto swRtOp = *(swRtOpRange.begin());

        auto stackSizes = mlir::extractFromIntegerArrayAttr<int64_t>(swRtOp.getStacks());
        VPUX_THROW_UNLESS(!stackSizes.empty(), "VPURT.SW.Runtime op should have non-empty 'stacks' attribute");
        VPUX_THROW_UNLESS(std::equal(stackSizes.begin() + 1, stackSizes.end(), stackSizes.begin()),
                          "Expected all stacks to be equal!");

        shaveStacks.reserve(stackSizes.size());
        for (auto idx : irange(stackSizes.size())) {
            auto indexType = VPURegMapped::IndexType::get(ctx, 0, idx);
            // TODO: use the computed size when E#147157 is implemented
            // for now only set a hardcoded value to the stack to enable DDR stack allocation
            // and to be able to correctly test E2E functionality.
            static constexpr size_t countSize = 16;
            static constexpr size_t overrideDefaultStackSize = countSize * Byte(1_KB).count();
            auto stack = builderFunc.create<VPUMI40XX::ShaveStackFrameBuffOp>(builderFunc.getUnknownLoc(), indexType,
                                                                              overrideDefaultStackSize);

            shaveStacks.push_back(stack.getResult());
        }
    }

    if (runtimeKernelFunction) {
        auto kernelElf = runtimeKernelFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
        VPUX_THROW_UNLESS(kernelElf, "Expected 'VPU.kernel_code' attribute in runtime kernel function");

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        auto actShvRtOp =
                builderFunc.create<VPUMI40XX::ActShaveRtOp>(builderFunc.getUnknownLoc(), trivialIndexType, kernelElf);

        actShvRt = actShvRtOp.getResult();
    } else {
        auto actRtCodeBufferMemrefType = vpux::ELF::getLinearMemrefType(ctx, ACT_RT_CODE_BUFFER_SIZE,
                                                                        vpux::getInt8Type(ctx), VPU::MemoryKind::DDR);

        auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                          actRtCodeBufferMemrefType,  // Type
                                                                          VPURT::BufferSection::DDR,  // Buffer Type
                                                                          0                           // byteOffset
        );

        actShvRt = declareBufferOp.getResult();
    }
    return std::make_pair(actShvRt, shaveStacks);
}

void createMappedInferenceOp(mlir::func::FuncOp funcOp, AllocateDDRStackFrames allocateDDRStackFrames) {
    // hardcoded, to be replaced with proper HW capabilities
    constexpr auto dmaDirectionRank = size_t{2};

    auto ctx = funcOp.getContext();
    auto moduleOp = getModuleOp(funcOp);

    const auto tileCount = static_cast<size_t>(config::getTileExecutor(moduleOp).getCount());
    const auto dmaTileCount =
            static_cast<size_t>(config::getAvailableExecutor(moduleOp, config::ExecutorKind::DMA_NN).getCount());
    const auto shavesPerTileCount =
            static_cast<size_t>(config::getAvailableExecutor(moduleOp, config::ExecutorKind::SHAVE_ACT).getCount());

    auto taskInfo =
            collectMappedInferenceTaskInfo(funcOp, tileCount, dmaTileCount, shavesPerTileCount, dmaDirectionRank);

    SmallVector<SmallVector<mlir::Value>> dmaTasks(dmaTileCount);
    SmallVector<mlir::ValueRange> dmaTasksArg(dmaTileCount);
    size_t dmaTasksArgLength = 0;
    for (size_t tileIdx = 0; tileIdx < dmaTileCount; ++tileIdx) {
        // dmaTasks
        for (size_t srcType = 0; srcType < dmaDirectionRank; ++srcType) {
            if (taskInfo.dmaHeads[tileIdx][srcType]) {
                dmaTasks[tileIdx].push_back(taskInfo.dmaHeads[tileIdx][srcType]);
            }
        }
        if (!dmaTasks[tileIdx].empty()) {
            dmaTasksArg[tileIdx] = mlir::ValueRange(dmaTasks[tileIdx]);
            dmaTasksArgLength = tileIdx + 1;
        }
    }

    SmallVector<mlir::Value> invariantTasks;
    SmallVector<mlir::Value> variantTasks;
    invariantTasks.reserve(tileCount);
    variantTasks.reserve(tileCount);
    SmallVector<SmallVector<mlir::Value>> actKernelRanges(tileCount), actKernelInvocations(tileCount);
    SmallVector<mlir::ValueRange> actKernelRangesArgs(tileCount), actKernelInvocationsArgs(tileCount);
    size_t actKernRangesTasksArgLength = 0;
    size_t actKernInvocationsTasksArgLength = 0;
    for (size_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
        if (taskInfo.invariantHeads[tileIdx]) {
            invariantTasks.push_back(taskInfo.invariantHeads[tileIdx]);
        }
        if (taskInfo.variantHeads[tileIdx]) {
            variantTasks.push_back(taskInfo.variantHeads[tileIdx]);
        }

        for (size_t shaveIdx = 0; shaveIdx < shavesPerTileCount; ++shaveIdx) {
            if (taskInfo.actKernelRangeHeads[tileIdx][shaveIdx]) {
                actKernelRanges[tileIdx].push_back(taskInfo.actKernelRangeHeads[tileIdx][shaveIdx]);
            }
            if (taskInfo.actKernelInvocationHeads[tileIdx][shaveIdx]) {
                actKernelInvocations[tileIdx].push_back(taskInfo.actKernelInvocationHeads[tileIdx][shaveIdx]);
            }
        }
        if (!actKernelRanges[tileIdx].empty()) {
            actKernelRangesArgs[tileIdx] = mlir::ValueRange(actKernelRanges[tileIdx]);
            actKernRangesTasksArgLength = tileIdx + 1;
        }
        if (!actKernelInvocations[tileIdx].empty()) {
            actKernelInvocationsArgs[tileIdx] = mlir::ValueRange(actKernelInvocations[tileIdx]);
            actKernInvocationsTasksArgLength = tileIdx + 1;
        }
    }

    mlir::Value barrierTasks = taskInfo.barrierTasks;
    auto barrierCount = taskInfo.barrierCount;
    auto hasInvocations = taskInfo.hasInvocations;

    mlir::Value actShvRt;
    SmallVector<mlir::Value> actShaveStacks;

    // create MappedInferenceOp
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    // create ActShaveRtOp
    if (hasInvocations) {
        std::tie(actShvRt, actShaveStacks) = setupActKernelRt(ctx, moduleOp, builderFunc, allocateDDRStackFrames);
    }

    auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);
    builderFunc.create<VPUMI40XX::MappedInferenceOp>(
            mlir::UnknownLoc::get(ctx), trivialIndexType,
            ArrayRef(dmaTasksArg.data(), dmaTasksArgLength),  // llvm::ArrayRef<::mlir::ValueRange> dmaTasks
            invariantTasks,                                   // mlir::ValueRange invariantTasks
            variantTasks,                                     // mlir::ValueRange variantTasks
            ArrayRef(actKernelRangesArgs.data(),
                     actKernRangesTasksArgLength),  // llvm::ArrayRef<::mlir::ValueRange> actKernelRanges
            ArrayRef(actKernelInvocationsArgs.data(),
                     actKernInvocationsTasksArgLength),  // llvm::ArrayRef<::mlir::ValueRange> actKernelInvocations
            nullptr,                                     // mlir::Value mediaTasks
            barrierTasks,                                // mlir::Value barrierTasks
            nullptr,                                     // mlir::Value workItemTasks
            nullptr,                                     // mlir::Value bootstrapBarriers
            actShvRt,                                    // mlir::Value actShaveRt
            mlir::ValueRange(actShaveStacks),            // mlir::ValueRange actShaveStacks
            nullptr,                                     // mlir::Value dmaHwpBase
            nullptr,                                     // mlir::Value hwpWorkpointCfg
            getIntArrayOfArray(ctx, taskInfo.dmaCount),  // mlir::ArrayAttr dmaCount
            builderFunc.getI64ArrayAttr(ArrayRef(taskInfo.invariantCount)),  // mlir::ArrayAttr invariantCount
            builderFunc.getI64ArrayAttr(ArrayRef(taskInfo.variantCount)),    // mlir::ArrayAttr variantCount
            getIntArrayOfArray(ctx, taskInfo.rangeCount),                    // mlir::ArrayAttr actKernelRangesCount
            getIntArrayOfArray(ctx, taskInfo.invocationCount),  // mlir::ArrayAttr actKernelInvocationsCount
            0,                                                  // mlir::IntegerAttr mediaCount
            barrierCount,                                       // mlir::IntegerAttr barrierCount
            nullptr,                                            // mlir::IntegerAttr workItemCount
            nullptr,                                            // mlir::IntegerAttr bootstrapBarriersCount
            nullptr,                                            // mlir::IntegerAttr bootstrapWorkItemTasksCount
            nullptr,                                            // mlir::IntegerAttr finalBarrierId
            nullptr,                                            // mlir::AnyMemRef barrierConfigurationTasks
            nullptr,                                            // mlir::IntegerAttr barrierConfigurationTasksCount
            nullptr,                                            // mlir::Value numOfBarrierReprogrammings
            nullptr,                                            // mlir::Value mappedInferenceVersion
            nullptr);                                           // VPURegMapped::BarrierProgrammingModeAttr
}

void foldActKernelTextAndEntry(mlir::func::FuncOp funcOp) {
    using TextAndEntry = std::pair<VPUMI40XX::DeclareKernelTextOp, VPUMI40XX::DeclareKernelEntryOp>;
    mlir::DenseMap<mlir::StringRef, TextAndEntry> visited;

    const auto replaceUses = [](auto op, auto& visitedOp) {
        if (!visitedOp) {
            visitedOp = op;
            return;
        }
        op.replaceAllUsesWith(visitedOp.getResult());
        op.erase();
    };

    funcOp.walk([&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*>(op)
                .Case<VPUMI40XX::DeclareKernelTextOp>([&](auto text) {
                    replaceUses(text, visited[text.getKernelPath()].first);
                })
                .Case<VPUMI40XX::DeclareKernelEntryOp>([&](auto entry) {
                    replaceUses(entry, visited[entry.getKernelPath()].second);
                });
    });
}

class ConvertVPUIP2VPUMI40XXPass final : public impl::ConvertVPUIP2VPUMI40XXBase<ConvertVPUIP2VPUMI40XXPass> {
public:
    ConvertVPUIP2VPUMI40XXPass(Logger log, bool enableMemorySideCache, AllocateDDRStackFrames allocateDDRStackFrames)
            : _enableMemorySideCacheOption(enableMemorySideCache), _allocateDDRStackFrames(allocateDDRStackFrames) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }

        if (allocateDDRStackFrames.hasValue()) {
            _log.trace("Allocate DDR shave stack frames has value of {0}",
                       allocateDDRStackFrames.getValue() ? "ENABLED" : "DISABLED");
            _allocateDDRStackFrames = allocateDDRStackFrames.getValue() ? AllocateDDRStackFrames::ENABLED
                                                                        : AllocateDDRStackFrames::DISABLED;
        }

        return mlir::success();
    }

private:
    bool _enableMemorySideCacheOption;
    AllocateDDRStackFrames _allocateDDRStackFrames;

    void safeRunOnFunc() final {
        auto& ctx = getContext();
        auto funcOp = getOperation();

        // E#145158: move to a dedicated pass
        createProfilingMetadataOp(funcOp, _log);

        // on VPUIP level IR contains VPURT::TaskOps with region populated with actual tasks
        // e.g. VPURT::TaskOp with VPUIP::NNDMAOp inside
        //
        // VPURT::TaskOp contains barrier data (wait barriers, update barriers, enqueue barrier)
        // separately from its content (VPUIP::NNDMAOp doesn't have any data about barriers itself)
        //
        // on VPUMI40XX level IR contains tasks directly with all associated information
        // e.g. VPUMI40XX::NNDMAOp replaces both VPURT::TaskOp and its internal VPUIP::NNDMAOp
        // and stores both DMA-specific data and barriers (wait, update, enqueue), so requires
        // both VPURT.TaskOp's and its content data to complete conversion
        //
        // if we match against VPURT::TaskOp and will manually inspect its content in rewriter
        // e.g. check task type inside: DMA or SW kernel, etc.
        // then we won't have access to task's operands through rewriter's OpAdaptor argument
        // as adaptor gives access to operands of operation we matched against (VPURT.TaskOp)
        //
        // to simplify rewriters match against internal tasks directly (VPUIP.NNDMAOp,
        // VPUIP.NCEClusterTask and etc.) instead of VPURT.TaskOp; replace content of
        // VPURT.TaskOp with VPUMI40XX tasks as 1st stage and rewrite VPURT.TaskOps and
        // VPURT.ConfigureBarrierOp together later (2nd/final stage)
        //
        // motivation is to keep rewriters as simple and local as possible, as accessing
        // IR content outside of matched operation is generally-unsafe in MLIR dialect
        // conversion
        //
        // double staged approach is based on assumption it's safe to leave VPURT.TaskOp
        // in incorrect intermediate state (immediately after 1st stage):
        // 1) VPURT.TaskOp requires its content to implement MemorySideEffects,
        //    which isn't done by VPUMI40XX;
        // 2) VPURT.TaskOp may be left with more than 1 op inside
        //
        // since VPURT.TaskOps would be soon (2nd stage) removed
        //
        // Note: assumption above is incorrect in case of 1-N dialect conversion as in contrast
        // with 1-1 version after each rewriter application it checks if there're trivially-dead
        // operations in IR that triggers VPURT.TaskOp MemorySideEffects evaluation
        // insertion of VPUMI40XX tasks ops outside VPURT.TaskOp doesn't work in 1-N infra anyway,
        // because:
        // 1) if you still remove original tasks from VPURT.TaskOp's body - it's
        //    invalid intermediate state again
        // 2) if you leave original tasks inside VPURT.TaskOp they will be triggered recursively
        //    and eventually fail conversion

        // during VPUIP -> VPUMI40XX there're 2 type conversions to happen:
        // 1) VPURT.DeclareBufferOps with DistributedBufferType; these ops should be
        // unrolled into multiple VPURT.DeclareBufferOps with single memref as output type
        // 2) VPURT.DeclareBufferOps with ITIBufferType; these ops should be converted
        // to a single VPURT.DeclareBufferOp with memref
        //
        // if type converter is provided DialectConversion would insert unrealized casts
        // that are expected to be canceled-out by the end of conversion
        //
        // since we match against internals of VPURT.TaskOp these unrealized casts would be
        // inserted into VPURT.TaskOp body; this way they won't cancel out as possible
        // "reverse" unrealized cast from previous op would be outside of VPURT.TaskOp
        // and since its region is IsolatedFromAbove they won't connect; remaining unrealized
        // cast ops would fail conversion
        //
        // 1-1 dialect conversion infra doesn't support 1-N (unrolling case)
        // 1-N dialect conversion infra is unusable due to reasons explained above
        //
        // conversion of VPURT.DeclareBufferOps is handled by rewriters themselve via
        // adding new required buffer ops to IR; original buffers (expected to be unused
        // by the end of conversion) are preserved and erased separately at the end of the pass
        //
        // so, overall expectation from rewriters:
        // 1) stay local and don't set anything for converted op that requires "external"
        //    context: barriers, indexing, merging operations
        // 2) handle VPURT.DeclareBufferOps conversion
        // 3) don't accept type converter

        mlir::RewritePatternSet tasksConverters(&ctx);
        tasksConverters.add<NNDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<PermuteDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<ExpandDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<ConvertDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<SpaceToDepthDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<DepthToSpaceDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<UpsamplingDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<PerAxisTileDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<DecompressDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<CompressDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<GatherDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<SyncDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<BarrierProgDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<FetchDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<EnqueueDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<ReadOnlyDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<SkipDMARewriter>(&ctx, _enableMemorySideCacheOption);
        tasksConverters.add<NCEClusterTaskRewriter>(&ctx);
        tasksConverters.add<SWKernelRewriter>(&ctx);

        mlir::ConversionTarget irWithMITasksInsideVPURTTaskOp(ctx);
        irWithMITasksInsideVPURTTaskOp.addIllegalDialect<VPUIP::VPUIPDialect>();
        irWithMITasksInsideVPURTTaskOp.addLegalDialect<VPUMI40XX::VPUMI40XXDialect>();

        // add operations that are inserted by rewriters as explicitly legal
        // otherwise conversion will fail; it's fine to keep VPURT::DeclareBufferOp
        // unconditionally legal as cases with DistributedBufferType & ITIBufferType
        // are handled by rewriters per explanation above
        irWithMITasksInsideVPURTTaskOp.addLegalOp<VPURT::DeclareBufferOp>();

        if (mlir::failed(
                    mlir::applyPartialConversion(funcOp, irWithMITasksInsideVPURTTaskOp, std::move(tasksConverters)))) {
            return signalPassFailure();
        }

        mlir::ConversionTarget finalConversionTarget(ctx);
        finalConversionTarget.addLegalDialect<VPUMI40XX::VPUMI40XXDialect>();
        finalConversionTarget.addLegalDialect<Const::ConstDialect>();
        finalConversionTarget.addLegalOp<mlir::func::FuncOp>();
        finalConversionTarget.addLegalOp<mlir::func::ReturnOp>();
        finalConversionTarget.addLegalOp<VPURT::DeclareBufferOp>();

        // if type converter is provided it needs to cover all the types processed
        // add trivial type converter 1st to signal no conversion for all types
        // except listed afterwards (they are searched in reversed order)
        mlir::TypeConverter typeConverter;
        typeConverter.addConversion([](mlir::Type type) {
            return type;
        });
        typeConverter.addConversion([&ctx](VPURT::BarrierType) {
            return VPURegMapped::IndexType::get(&ctx, 0);
        });

        mlir::RewritePatternSet finalConverters(&ctx);
        finalConverters.add<BarrierRewriter>(typeConverter, &ctx);
        finalConverters.add<VPURTTaskRewriter>(typeConverter, &ctx);

        if (mlir::failed(mlir::applyFullConversion(funcOp, finalConversionTarget, std::move(finalConverters)))) {
            signalPassFailure();
        }

        // even though DeclareBufferOp is Pure and will be removed by canonicalizer
        // if it doesn't have users, assert here we don't have dangling DistributedBuffers
        // and ITIBuffers and erase
        funcOp.walk([](VPURT::DeclareBufferOp bufferOp) {
            if (mlir::isa<VPUIP::DistributedBufferType, VPUIP::ITIBufferType>(bufferOp.getType())) {
                assert(bufferOp.getResult().getUsers().empty());
                bufferOp.erase();
            }
        });

        // finalize IR outside of DialectConversion when IR traversal is required
        finalizeBarriersLegalization(funcOp);

        enumerateAndChainOperations(funcOp);
        // requires enumerateAndChainOperations to happen first
        // as it relies on indexes to be valid
        replaceReturnOpWithOpRanges(funcOp);

        createMappedInferenceOp(funcOp, _allocateDDRStackFrames);

        foldActKernelTextAndEntry(funcOp);
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUMI40XXPass(Logger log, bool enableMemorySideCache,
                                                                   AllocateDDRStackFrames allocateDDRStackFrames) {
    return std::make_unique<ConvertVPUIP2VPUMI40XXPass>(log, enableMemorySideCache, allocateDDRStackFrames);
}
