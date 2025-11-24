//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/m2i_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/nce_cluster_task_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/sw_kernel_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/task_rewriter.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/Support/FileSystem.h>

#include <vector>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUIP2VPUMI40XX
#define GEN_PASS_DEF_CONVERTVPUIP2VPUMI40XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpux::vpuip2vpumi40xx;

namespace {

void enumerateOperations(mlir::func::FuncOp funcOp) {
    // Note: it's not the 1st time type + tile & list part of index come up as key to distinguish a list
    // of tasks, the same is used in VPUMI40XX::OpRanges. Consider reusing logic
    // E#146741
    llvm::SmallDenseMap<std::tuple<mlir::OperationName, uint32_t, uint32_t>, uint32_t> counters;
    // take op by l-value non-const reference as single "auto"
    // deduces mlir::Operation that calls deleted copy ctor
    for (auto& op : funcOp.getOps()) {
        if (!op.hasTrait<VPUMI40XX::SingleOutputAsIndexOp>()) {
            continue;
        }
        auto result = op.getResult(0);

        auto originalIndex = mlir::cast<VPURegMapped::IndexType>(result.getType());
        assert(originalIndex.getValue() == 0);

        auto key = std::make_tuple(op.getName(), originalIndex.getTileIdx(), originalIndex.getListIdx());
        auto newIndex = VPURegMapped::IndexType::get(op.getContext(), originalIndex.getTileIdx(),
                                                     originalIndex.getListIdx(), counters[key]++);

        result.setType(newIndex);
    }
}

void chainTasksInLists(mlir::func::FuncOp funcOp) {
    // E#146741
    llvm::SmallDenseMap<std::tuple<VPURegMapped::TaskType, uint32_t, uint32_t>, mlir::Value> lastTaskInListResult;
    for (auto task : funcOp.getOps<VPURegMapped::TaskOpInterface>()) {
        assert(!task.getPreviousTask());

        auto index = task.getIndexType();
        auto key = std::make_tuple(task.getTaskType(), index.getTileIdx(), index.getListIdx());
        auto& previousTask = lastTaskInListResult[key];
        if (previousTask) {
            task.setPreviousTask(previousTask);
        }
        previousTask = task.getResult();
    }
}

void finalizeBarriersLegalization(mlir::func::FuncOp funcOp) {
    for (auto barrier : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        uint8_t producerCount = 0;
        uint8_t consumerCount = 0;
        auto result = barrier.getResult();
        for (auto user : result.getUsers()) {
            auto executableTask = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(user);
            if (!executableTask) {
                continue;
            }

            // Enqueue barrier can't be wait or update barrier otherwise we have a cycle
            auto enqueueTarget = executableTask.getEnqueueBarrier();
            if (enqueueTarget && enqueueTarget == result) {
                continue;
            }

            const auto increment = executableTask.getBarrierHitsCount();
            if (llvm::is_contained(executableTask.waitBarriers(), result)) {
                consumerCount += increment;
            } else {
                assert(llvm::is_contained(executableTask.updateBarriers(), result));
                producerCount += increment;
            }
        }

        assert(producerCount > 0 || consumerCount > 0);
        barrier.setProducerCount(producerCount);
        barrier.setConsumerCount(consumerCount);
    }
}

void replaceReturnOpWithOpRanges(mlir::func::FuncOp funcOp) {
    auto context = funcOp->getContext();
    auto taskOps = funcOp.getOps<VPURegMapped::TaskOpInterface>();

    using Range = std::tuple<VPURegMapped::TaskType, uint32_t, uint32_t>;
    const auto getRange = [](auto taskOp) {
        const auto index = taskOp.getIndexType();
        return std::make_tuple(taskOp.getTaskType(), index.getTileIdx(), index.getListIdx());
    };

    mlir::DenseMap<Range, size_t> rangesSizes;
    const auto getRangeSize = [&](auto taskOp) {
        auto& rangeSize = rangesSizes[getRange(taskOp)];
        if (rangeSize != 0) {
            return rangeSize;
        }

        const auto isOpFromTheSameRange = [taskOp, getRange](auto op) {
            return getRange(taskOp) == getRange(op);
        };

        return rangeSize = std::count_if(std::begin(taskOps), std::end(taskOps), isOpFromTheSameRange);
    };

    const auto isFirst = [](auto taskOp) {
        return taskOp.getIndexType().getValue() == 0;
    };

    const auto isLast = [getRangeSize](auto taskOp) {
        return taskOp.getIndexType().getValue() == getRangeSize(taskOp) - 1;
    };

    mlir::DenseMap<Range, size_t> rangesIndexes;
    mlir::SmallVector<mlir::Attribute> rangesTaskTypesAttrs;
    mlir::SmallVector<mlir::Value> rangesBegins;
    mlir::SmallVector<mlir::Value> rangesEnds;

    for (auto taskOp : taskOps) {
        const auto range = getRange(taskOp);
        const auto result = taskOp.getResult();

        if (isFirst(taskOp)) {
            rangesTaskTypesAttrs.push_back(VPURegMapped::TaskTypeAttr::get(context, taskOp.getTaskType()));
            rangesBegins.push_back(result);
            rangesEnds.push_back({});
            rangesIndexes[range] = rangesBegins.size() - 1;
        }

        if (isLast(taskOp)) {
            rangesEnds[rangesIndexes[range]] = result;
        }
    }

    assert(rangesSizes.size() == rangesIndexes.size());
    assert(rangesIndexes.size() == rangesTaskTypesAttrs.size());
    assert(rangesTaskTypesAttrs.size() == rangesBegins.size());
    assert(rangesBegins.size() == rangesEnds.size());

    assert(funcOp.getBlocks().size() == 1);
    auto returnOp = funcOp.getBlocks().front().getTerminator();
    assert(returnOp);

    mlir::OpBuilder builder(returnOp);
    builder.create<VPUMI40XX::OpRanges>(returnOp->getLoc(), mlir::ArrayRef(rangesBegins), mlir::ArrayRef(rangesEnds),
                                        mlir::ArrayAttr::get(context, rangesTaskTypesAttrs));

    returnOp->erase();
}

void createProfilingMetadataOp(mlir::func::FuncOp funcOp, Logger log) {
    auto ctx = funcOp.getContext();
    auto moduleOp = getModuleOp(funcOp);

    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);

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

template <typename TaskType>
bool noCond(TaskType) {
    return true;
}

template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
size_t countTasksIf(mlir::func::FuncOp& funcOp, Condition&& condition = noCond) {
    auto tasks = funcOp.template getOps<TaskType>();
    return std::count_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
}

template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
mlir::Value findTaskIf(mlir::func::FuncOp& funcOp, Condition&& condition = noCond) {
    auto tasks = funcOp.template getOps<TaskType>();
    auto target = std::find_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
    return target != tasks.end() ? (*target).getResult() : mlir::Value();
}

template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
int64_t gatherTasks(mlir::SmallVector<mlir::Value>& taskValues, mlir::func::FuncOp& funcOp, uint32_t tileIdx,
                    uint32_t listIdx) {
    auto indexCond = [tileIdx, listIdx](auto op) {
        auto type = mlir::dyn_cast<vpux::VPURegMapped::IndexType>(op.getIndex().getType());
        return (type.getTileIdx() == tileIdx) && (type.getListIdx() == listIdx);
    };

    auto head = findTaskIf<TaskType>(funcOp, indexCond);
    if (head) {
        taskValues.push_back(head);
    }
    return countTasksIf<TaskType>(funcOp, indexCond);
}

std::pair<mlir::Value, SmallVector<mlir::Value>> setupActKernelRt(
        mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::OpBuilder& builderFunc,
        AllocateShaveStackFrames createStacks = AllocateShaveStackFrames::DISABLED) {
    constexpr auto ACT_RT_CODE_BUFFER_SIZE = (1_MB).to<vpux::Byte>().count();

    // check for actShaveRt info
    mlir::Value actShvRt;
    auto vpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");
    VPUX_THROW_UNLESS(vpuSwModuleOp != nullptr, "setupActKernelConfig: @VPU.SW module missing.");
    auto runtimeKernelFunction = vpuSwModuleOp.lookupSymbol<mlir::func::FuncOp>("runtime");

    // check for actShave stacks info
    auto swRtOpRange = moduleOp.getOps<VPURT::SWRunTimeOp>();
    SmallVector<mlir::Value> shaveStacks;
    if (!swRtOpRange.empty() && createStacks == AllocateShaveStackFrames::ENABLED) {
        VPUX_THROW_WHEN(std::distance(swRtOpRange.begin(), swRtOpRange.end()) > 1,
                        "More than 1 instance of VPURT.SW.Runtime");
        auto swRtOp = *(swRtOpRange.begin());

        auto stackSizes = mlir::extractFromIntegerArrayAttr<int64_t>(swRtOp.getStacks());
        VPUX_THROW_UNLESS(std::adjacent_find(stackSizes.begin(), stackSizes.end()) != stackSizes.end(),
                          "Were expecting all stacks to be equal!");

        shaveStacks.reserve(stackSizes.size());
        for (auto idx : irange(stackSizes.size())) {
            auto indexType = VPURegMapped::IndexType::get(ctx, 0, idx);
            // TODO: use the computed size when E#147157 is implemented
            // for now only set a hardcoded value to the stack to enable DDR stack allocation
            // and to be able to correctly test E2E functionality.
            static constexpr size_t countSize = 16;
            static constexpr size_t overrideDefaultStackSize = countSize * Byte(1_KB).count();
            auto stack = builderFunc.create<VPUMI40XX::ShaveStackFrameOp>(builderFunc.getUnknownLoc(), indexType,
                                                                          overrideDefaultStackSize);

            shaveStacks.push_back(stack.getResult());
        }
    }

    if (runtimeKernelFunction) {
        const auto kernelElf =
                std::string(runtimeKernelFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_code").getValue());

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        auto actShvRtOp = builderFunc.create<VPUMI40XX::ActShaveRtOp>(builderFunc.getUnknownLoc(), trivialIndexType,
                                                                      mlir::StringAttr::get(ctx, kernelElf));

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

void createMappedInferenceOp(mlir::func::FuncOp funcOp, AllocateShaveStackFrames allocateShaveStackFrames) {
    // hardcoded, to be replaced with proper HW capabilities
    constexpr auto dmaDirectionRank = size_t{2};

    auto ctx = funcOp.getContext();
    auto moduleOp = getModuleOp(funcOp);

    const auto tileCount = static_cast<size_t>(config::getTileExecutor(moduleOp).getCount());
    const auto dmaTileCount =
            static_cast<size_t>(config::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN).getCount());
    const auto shavesPerTileCount =
            static_cast<size_t>(config::getAvailableExecutor(moduleOp, VPU::ExecutorKind::SHAVE_ACT).getCount());

    mlir::SmallVector<mlir::SmallVector<mlir::Value>> dmaTasks(dmaTileCount);
    mlir::SmallVector<mlir::ValueRange> dmaTasksArg(dmaTileCount);
    size_t dmaTasksArgLength = 0;
    mlir::SmallVector<mlir::Value> invariantTasks, variantTasks;
    mlir::SmallVector<mlir::SmallVector<mlir::Value>> actKernelRanges(tileCount), actKernelInvocations(tileCount);
    mlir::SmallVector<mlir::ValueRange> actKernelRangesArgs(tileCount), actKernelInvocationsArgs(tileCount);
    size_t actKernRangesTasksArgLength = 0;
    size_t actKernInvocationsTasksArgLength = 0;
    mlir::Value barrierTasks;
    mlir::Value mediaTasks;
    mlir::Value actShvRt;
    SmallVector<mlir::Value> actShaveStacks;

    mlir::SmallVector<mlir::SmallVector<int64_t>> dmaCount(dmaTileCount,
                                                           mlir::SmallVector<int64_t>(dmaDirectionRank, 0));
    mlir::SmallVector<int64_t> invariantCount(tileCount, 0), variantCount(tileCount, 0);
    mlir::SmallVector<mlir::SmallVector<int64_t>> rangeCount(tileCount,
                                                             mlir::SmallVector<int64_t>(shavesPerTileCount, 0));
    mlir::SmallVector<mlir::SmallVector<int64_t>> invoCount(tileCount,
                                                            mlir::SmallVector<int64_t>(shavesPerTileCount, 0));

    int64_t barrierCount = 0;
    int64_t mediaCount = 0;
    bool hasInvocations = false;

    for (size_t tileIdx = 0; tileIdx < dmaTileCount; ++tileIdx) {
        // dmaTasks
        for (size_t srcType = 0; srcType < dmaDirectionRank; ++srcType) {
            dmaCount[tileIdx][srcType] = gatherTasks<VPUMI40XX::NNDMAOp>(dmaTasks[tileIdx], funcOp, tileIdx, srcType);
        }
        if (!dmaTasks[tileIdx].empty()) {
            dmaTasksArg[tileIdx] = mlir::ValueRange(dmaTasks[tileIdx]);
            dmaTasksArgLength = tileIdx + 1;
        }
    }

    for (size_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
        // invariantTasks
        invariantCount[tileIdx] = gatherTasks<VPUMI40XX::DPUInvariantOp>(invariantTasks, funcOp, tileIdx, 0);

        // variantTasks
        variantCount[tileIdx] = gatherTasks<VPUMI40XX::DPUVariantOp>(variantTasks, funcOp, tileIdx, 0);

        for (size_t shaveIdx = 0; shaveIdx < shavesPerTileCount; ++shaveIdx) {
            rangeCount[tileIdx][shaveIdx] =
                    gatherTasks<VPUMI40XX::ActKernelRangeOp>(actKernelRanges[tileIdx], funcOp, tileIdx, shaveIdx);
            invoCount[tileIdx][shaveIdx] = gatherTasks<VPUMI40XX::ActKernelInvocationOp>(actKernelInvocations[tileIdx],
                                                                                         funcOp, tileIdx, shaveIdx);
        }
        if (!actKernelRanges[tileIdx].empty()) {
            actKernelRangesArgs[tileIdx] = mlir::ValueRange(actKernelRanges[tileIdx]);
            actKernRangesTasksArgLength = tileIdx + 1;
        }
        if (!actKernelInvocations[tileIdx].empty()) {
            actKernelInvocationsArgs[tileIdx] = mlir::ValueRange(actKernelInvocations[tileIdx]);
            actKernInvocationsTasksArgLength = tileIdx + 1;
        }

        auto tileInvoCount =
                std::accumulate(invoCount[tileIdx].begin(), invoCount[tileIdx].end(), static_cast<int64_t>(0));
        if (tileInvoCount != 0) {
            hasInvocations = true;
        }
    }

    // barrierTasks
    barrierTasks = findTaskIf<VPUMI40XX::ConfigureBarrierOp>(funcOp);
    barrierCount = countTasksIf<VPUMI40XX::ConfigureBarrierOp>(funcOp);

    // mediaTasks
    mediaTasks = findTaskIf<VPUMI40XX::M2IOp>(funcOp);
    mediaCount = countTasksIf<VPUMI40XX::M2IOp>(funcOp);

    // create MappedInferenceOp
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    // create ActShaveRtOp
    if (hasInvocations) {
        std::tie(actShvRt, actShaveStacks) = setupActKernelRt(ctx, moduleOp, builderFunc, allocateShaveStackFrames);
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
            mediaTasks,                                  // mlir::Value mediaTasks
            barrierTasks,                                // mlir::Value barrierTasks
            nullptr,                                     // mlir::Value workItemTasks
            nullptr,                                     // mlir::Value bootstrapBarriers
            actShvRt,                                    // mlir::Value actShaveRt
            mlir::ValueRange(actShaveStacks),            // mlir::ValueRange actShaveStacks
            nullptr,                                     // mlir::Value dmaHwpBase
            nullptr,                                     // mlir::Value hwpWorkpointCfg
            getIntArrayOfArray(ctx, dmaCount),           // mlir::ArrayAttr dmaCount
            builderFunc.getI64ArrayAttr(ArrayRef(invariantCount)),  // mlir::ArrayAttr invariantCount
            builderFunc.getI64ArrayAttr(ArrayRef(variantCount)),    // mlir::ArrayAttr variantCount
            getIntArrayOfArray(ctx, rangeCount),                    // mlir::ArrayAttr actKernelRangesCount
            getIntArrayOfArray(ctx, invoCount),                     // mlir::ArrayAttr actKernelInvocationsCount
            mediaCount,                                             // mlir::IntegerAttr mediaCount
            barrierCount,                                           // mlir::IntegerAttr barrierCount
            nullptr,                                                // mlir::IntegerAttr workItemCount
            nullptr,                                                // mlir::IntegerAttr bootstrapBarriersCount
            nullptr,                                                // mlir::IntegerAttr bootstrapWorkItemTasksCount
            nullptr,                                                // mlir::IntegerAttr finalBarrierId
            nullptr,                                                // mlir::AnyMemRef barrierConfigurationTasks
            nullptr,                                                // mlir::IntegerAttr barrierConfigurationTasksCount
            nullptr,                                                // mlir::Value numOfBarrierReprogrammings
            nullptr,                                                // mlir::Value mappedInferenceVersion
            nullptr                                                 // VPURegMapped::BarrierProgrammingModeAttr
    );
}

void foldActKernelTextAndEntry(mlir::func::FuncOp funcOp) {
    using TextAndEntry = std::pair<VPUMI40XX::DeclareKernelTextOp, VPUMI40XX::DeclareKernelEntryOp>;
    mlir::DenseMap<mlir::StringRef, TextAndEntry> visited;

    funcOp.walk([&visited](VPUMI40XX::DeclareKernelTextOp text) {
        auto kernel = text.getKernelPath();
        auto& [visitedText, _] = visited[kernel];
        if (!visitedText) {
            visitedText = text;
            return;
        }
        text.replaceAllUsesWith(visitedText.getResult());
        text.erase();
    });

    funcOp.walk([&visited](VPUMI40XX::DeclareKernelEntryOp entry) {
        auto kernel = entry.getKernelPath();
        auto& [_, visitedEntry] = visited[kernel];
        if (!visitedEntry) {
            visitedEntry = entry;
            return;
        }
        entry.replaceAllUsesWith(visitedEntry.getResult());
        entry.erase();
    });
}

class ConvertVPUIP2VPUMI40XXPass final : public impl::ConvertVPUIP2VPUMI40XXBase<ConvertVPUIP2VPUMI40XXPass> {
public:
    ConvertVPUIP2VPUMI40XXPass(Logger log, bool enableMemorySideCache, AllocateShaveStackFrames allocateShaveStack)
            : _enableMemorySideCacheOption(enableMemorySideCache), _allocateShaveStackFrames(allocateShaveStack) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }

        if (allocateShaveStackFrames.hasValue()) {
            _log.trace("Allocate stack shave has velue of {0}",
                       allocateShaveStackFrames.getValue() ? "ENABLED" : "DISABLED");
            _allocateShaveStackFrames = allocateShaveStackFrames.getValue() ? AllocateShaveStackFrames::ENABLED
                                                                            : AllocateShaveStackFrames::DISABLED;
        }

        return mlir::success();
    }

private:
    bool _enableMemorySideCacheOption;
    AllocateShaveStackFrames _allocateShaveStackFrames;

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
        tasksConverters.add<NCEClusterTaskRewriter>(&ctx);
        tasksConverters.add<SWKernelRewriter>(&ctx);
        tasksConverters.add<M2IRewriter>(&ctx);

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
        chainTasksInLists(funcOp);

        enumerateOperations(funcOp);
        // requires enumerateOperations to happen first
        // as it relies on indexes to be valid
        replaceReturnOpWithOpRanges(funcOp);

        createMappedInferenceOp(funcOp, _allocateShaveStackFrames);

        foldActKernelTextAndEntry(funcOp);
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUMI40XXPass(Logger log, bool enableMemorySideCache,
                                                                   AllocateShaveStackFrames allocateShaveStackFrames) {
    return std::make_unique<ConvertVPUIP2VPUMI40XXPass>(log, enableMemorySideCache, allocateShaveStackFrames);
}
