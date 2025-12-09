//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/cache_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADDSWKERNELINSTRUCTIONPREFETCH
#define GEN_PASS_DEF_ADDSWKERNELINSTRUCTIONPREFETCH
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

static const SmallVector<StringLiteral> SW_DUMMY_KERNELS_PREFETCH_SUPPORTED = {
        "activation_swish", "eltwise_mul",   "softmax",        "convert",        "rms_norm", "activation_swish",
        "activation_sin",   "eltwise_equal", "activation_cos", "eltwise_select", "topk"};

//
// AddSwKernelInstructionPrefetch
//
class AddSwKernelInstructionPrefetch final :
        public VPUIP::impl::AddSwKernelInstructionPrefetchBase<AddSwKernelInstructionPrefetch> {
public:
    explicit AddSwKernelInstructionPrefetch(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }

        if (minimumShaveStartTimeForPrefetch.hasValue()) {
            _minFreeCyclesHasValue = true;
            _minimumFreeCyclesForPrefetch = minimumShaveStartTimeForPrefetch.getValue();
        }

        return mlir::success();
    }

private:
    void safeRunOnFunc() final;

    mlir::SymbolRefAttr getPrefetchSymbol(mlir::Operation* funcOp);
    size_t getNumTiles(mlir::ModuleOp moduleOp);
    size_t getAvailableCacheSize(mlir::ModuleOp moduleOp);
    VPUIP::SwKernelOp insertPrefetchOpBeforeFirstKernelTask(mlir::Operation* firstSwTask, mlir::Value updateBarrier,
                                                            size_t clusterIdx, std::string& kernelName,
                                                            mlir::SymbolRefAttr functionSymbol);

    VPUIP::SwKernelOp insertDummyKernelOpBeforeFirstKernelTask(mlir::Operation* firstSwTask,
                                                               mlir::ValueRange updateBarrier, size_t clusterIdx,
                                                               std::string& kernelName);
    mlir::Operation* getFirstSwTaskInIRWaitingForBarrier(mlir::Value waitBarrier);
    std::pair<std::string, size_t> getKernelNameAndSize(VPUIP::SwKernelOp swKernelOp);

    using SwKernelPrefetchVec = std::vector<std::tuple<std::string, size_t, size_t>>;
    std::pair<SwKernelPrefetchVec, size_t> getPrefetchCandidatesAndFirstSwTask(mlir::Operation* funcOp,
                                                                               VPURT::TaskConfigVec& allTasks);
    std::tuple<mlir::Operation*, mlir::Value, size_t> getFirstSwTaskInIRAndBestUpdateBarrier(
            VPURT::InferenceExecutionSimulator& infSim, VPURT::TaskConfigVec& allTasks, size_t firstShvTaskIndex);
    std::vector<VPUIP::SwKernelOp> insertPrefetchTasks(mlir::Operation* funcOp, SwKernelPrefetchVec& kernelsToPrefetch,
                                                       mlir::Operation* firstShaveTaskInIR,
                                                       mlir::Value bestUpdateBarrier);
    std::vector<VPUIP::SwKernelOp> insertPrefetchTasksDuringExec(
            mlir::Operation* funcOp, AddSwKernelInstructionPrefetch::SwKernelPrefetchVec& kernelsToPrefetch,
            VPURT::TaskConfigVec& allTasks);

    bool hasVPUSWModule(mlir::Operation* funcOp);
    size_t getOffsetReservedMem(const mlir::ModuleOp module);

    std::map<std::string, mlir::SymbolRefAttr> kernelNameToSymbol;
    std::map<std::string, mlir::ArrayAttr> kernelNameToArgs;
    std::map<std::string, VPUIP::SwKernelOp> kernelNameToOps;

    static constexpr StringLiteral vpuTaskTypeAttrName{"VPU.task_type"};
    static constexpr StringLiteral vpuKernelEntryAttrName{"VPU.kernel_entry"};
    static constexpr size_t CACHE_LINE_SIZE = 64ul;

    bool _minFreeCyclesHasValue = false;
    size_t _minimumFreeCyclesForPrefetch = 250000;
    bool _useDummyKernelForInstructionPrefetch = false;
    size_t _dynamicPrefetchTileCounter = 0;
};

bool AddSwKernelInstructionPrefetch::hasVPUSWModule(mlir::Operation* funcOp) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};
    auto innerModule = moduleOp.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    return innerModule;
}

size_t AddSwKernelInstructionPrefetch::getOffsetReservedMem(const mlir::ModuleOp module) {
    auto cachePrefetchMem =
            config::getDummySwKernelsForInstructionPrefetchReservedMemory(module, VPU::MemoryKind::CMX_NN);
    auto offsetCachePrefetch = cachePrefetchMem.getOffset();
    VPUX_THROW_WHEN(!offsetCachePrefetch.has_value(),
                    "DummySwKernelsForInstructionPrefetchReservedMemory offset is not set!");
    return offsetCachePrefetch.value();
}

mlir::SymbolRefAttr AddSwKernelInstructionPrefetch::getPrefetchSymbol(mlir::Operation* funcOp) {
    auto ctx = funcOp->getContext();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    auto vpuswModule = vpux::VPUIP::getVPUSWModule(moduleOp, _log);

    const auto cacheOpType = VPU::ActShaveTaskType::CACHE_PREFETCH;
    const std::string functionName = "cache_prefetch";
    auto functionNameSymbol = mlir::SymbolRefAttr::get(ctx, functionName);
    auto functionSymbol = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {functionNameSymbol});

    // check if this function was already created
    auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(functionName);
    if (prebuiltFunction == nullptr) {
        OpBuilderLogger builderLog(_log.nest());
        auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);

        const auto funcType = mlir::FunctionType::get(ctx, {}, {});
        auto newFuncOp =
                innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), functionName, funcType);

        // modify attributes
        newFuncOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));
        newFuncOp->setAttr(vpuTaskTypeAttrName,
                           mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(cacheOpType)));
    }

    return functionSymbol;
}

size_t AddSwKernelInstructionPrefetch::getNumTiles(mlir::ModuleOp moduleOp) {
    auto tileOp = config::getTileExecutor(moduleOp);
    return tileOp.getCount();
}

size_t AddSwKernelInstructionPrefetch::getAvailableCacheSize(mlir::ModuleOp moduleOp) {
    const auto numTiles = getNumTiles(moduleOp);

    const Byte BASE_CACHE_SIZE = 64_KB;
    const uint32_t TILE_CACHE_MASK = 0b1110;
    // NOTE: 14KB is subtracted to account for ActShave runtime. It would be better
    // to use getKernelELF but at this point in compilation VPU.kernel_entry is not set
    // for runtime.
    const Byte totalCacheSize = BASE_CACHE_SIZE * (numTiles & TILE_CACHE_MASK) - Byte(14_KB);
    return totalCacheSize.count();
}

VPUIP::SwKernelOp AddSwKernelInstructionPrefetch::insertPrefetchOpBeforeFirstKernelTask(
        mlir::Operation* firstSwTask, mlir::Value updateBarrier, size_t clusterIdx, std::string& kernelName,
        mlir::SymbolRefAttr functionSymbol) {
    mlir::OpBuilder builder(firstSwTask);
    mlir::SmallVector<mlir::Value> buffers = {};
    const auto buffersRange = mlir::ValueRange(buffers);
    auto updateBarriers = updateBarrier;
    auto waitBarriers = mlir::ValueRange(buffers);
    auto newLoc = appendLoc(firstSwTask->getLoc(), "prefetch_{0}", kernelName);
    if (stringifyPrimaryLocation(newLoc).find("/cluster_") == std::string::npos) {
        newLoc = appendLoc(newLoc, "cluster_{0}", clusterIdx);
    }

    auto cachePrefetchSwKernel = vpux::VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
            builder, waitBarriers, updateBarriers, newLoc, buffersRange, buffersRange, nullptr, functionSymbol,
            getIntAttr(builder, clusterIdx));

    auto ctx = firstSwTask->getContext();
    cachePrefetchSwKernel->setAttr("kernelElfName", mlir::StringAttr::get(ctx, kernelName));

    const mlir::SmallVector<mlir::Attribute> args = {};
    vpux::VPUIP::initSwKernel(cachePrefetchSwKernel, buffersRange, buffersRange, args, _log.nest(),
                              /*swKernelRunOp=*/nullptr);
    _log.trace("cachePrefetchSwKernel {0}", cachePrefetchSwKernel);
    return cachePrefetchSwKernel;
}

// For LNL, Shave kernel instruction prefetch needs to insert a dummy kernel instead of prefetch kernel
VPUIP::SwKernelOp AddSwKernelInstructionPrefetch::insertDummyKernelOpBeforeFirstKernelTask(
        mlir::Operation* firstSwTask, mlir::ValueRange updateBarrier, size_t clusterIdx, std::string& kernelName) {
    mlir::OpBuilder builder(firstSwTask);
    auto kernelOp = kernelNameToOps[kernelName];
    auto moduleOp = kernelOp->getParentOfType<mlir::ModuleOp>();
    auto reservedMemOffset = getOffsetReservedMem(moduleOp);
    auto offsetAttr = getIntAttr(moduleOp->getContext(), reservedMemOffset);
    auto tileIndexAttr = kernelOp.getTileIndexAttr();
    VPUX_THROW_UNLESS(tileIndexAttr, "SwKernelOp '{0}' does not have a tileIndex attribute", kernelOp->getLoc());
    const int64_t tileIndex = static_cast<int64_t>(clusterIdx);

    auto createBuffer = [&](mlir::Value io, StringRef suffix, mlir::SmallVector<mlir::Value>& buffers) {
        if (auto bufOp = io.getDefiningOp<VPURT::DeclareBufferOp>()) {
            auto origType = mlir::cast<NDTypeInterface>(io.getType());
            auto newMemSpaceAttr = vpux::IndexedSymbolAttr::get(moduleOp->getContext(),
                                                                stringifyEnum(VPU::MemoryKind::CMX_NN), tileIndex);
            auto newSectionIndexAttr = builder.getI64ArrayAttr({tileIndex});
            auto newType = origType.changeShape({1, 1, 1, 1}).changeMemSpace(newMemSpaceAttr);
            auto newBuff = builder.create<VPURT::DeclareBufferOp>(appendLoc(bufOp->getLoc(), suffix), newType,
                                                                  bufOp.getSectionAttr(), newSectionIndexAttr,
                                                                  offsetAttr, bufOp.getSwizzlingKeyAttr());
            buffers.push_back(newBuff);
            return true;
        }
        return false;
    };

    mlir::SmallVector<mlir::Value> srcBuffers, dstBuffers;
    for (auto input : kernelOp.getInputs()) {
        if (createBuffer(input, "prefetch_src", srcBuffers)) {
            break;
        }
    }
    for (auto output : kernelOp.getOutputBuffs()) {
        if (createBuffer(output, "prefetch_dst", dstBuffers)) {
            break;
        }
    }

    VPUX_THROW_WHEN(srcBuffers.empty() || dstBuffers.empty(),
                    "Got empty buffers during dummy shave kernel collecting I/O for instruction prefetch.");

    auto newLoc = appendLoc(firstSwTask->getLoc(), "prefetch_{0}", kernelName);
    if (stringifyPrimaryLocation(newLoc).find("/cluster_") == std::string::npos) {
        newLoc = appendLoc(newLoc, "cluster_{0}", clusterIdx);
    }

    auto cachePrefetchSwKernel = vpux::VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
            builder, mlir::ValueRange(), updateBarrier, newLoc, mlir::ValueRange(srcBuffers),
            mlir::ValueRange(dstBuffers), nullptr, kernelNameToSymbol[kernelName], builder.getI64IntegerAttr(tileIndex),
            kernelOp.getInputStridesAttr(), kernelOp.getOutputStridesAttr());
    // The dummy kernels here are generated after ActShaveProfilingPass,
    // so we need to add skipProfiling as attribute to avoid capturing their metadata
    cachePrefetchSwKernel->setAttr("skipProfiling", mlir::UnitAttr::get(firstSwTask->getContext()));

    auto args = (kernelName == "convert" || kernelName == "eltwise_mul" || kernelName == "activation_cos" ||
                 kernelName == "activation_sin" || kernelName == "eltwise_equal" || kernelName == "eltwise_select" ||
                 kernelName == "rms_norm")
                        ? mlir::ArrayAttr::get(moduleOp->getContext(), {})
                        : kernelNameToArgs[kernelName];

    vpux::VPUIP::initSwKernel(cachePrefetchSwKernel, mlir::ValueRange(srcBuffers), mlir::ValueRange(dstBuffers), args,
                              _log.nest(), /*swKernelRunOp=*/nullptr);

    _log.trace("cachePrefetchSwKernel {0}", cachePrefetchSwKernel);
    return cachePrefetchSwKernel;
}

mlir::Operation* AddSwKernelInstructionPrefetch::getFirstSwTaskInIRWaitingForBarrier(mlir::Value waitBarrier) {
    mlir::Operation* firstKernelOpInIR = nullptr;
    for (auto user : waitBarrier.getUsers()) {
        if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(user)) {
            if (!mlir::isa<VPUIP::SwKernelOp>(userTaskOp.getInnerTaskOp())) {
                continue;
            }
            bool waitsForTargetBarrier = false;
            for (auto userWaitBarrier : userTaskOp.getWaitBarriers()) {
                if (userWaitBarrier == waitBarrier) {
                    waitsForTargetBarrier = true;
                }
            }
            if (waitsForTargetBarrier) {
                if ((firstKernelOpInIR == nullptr) || user->isBeforeInBlock(firstKernelOpInIR)) {
                    firstKernelOpInIR = user;
                }
            }
        }
    }
    return firstKernelOpInIR;
}

std::pair<std::string, size_t> AddSwKernelInstructionPrefetch::getKernelNameAndSize(VPUIP::SwKernelOp swKernelOp) {
    auto moduleOp = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto swKernelSymbol = swKernelOp.getKernelFunction();
    auto kernelInfoFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(swKernelSymbol);
    auto kernelName = std::string(kernelInfoFuncOp->getAttrOfType<mlir::StringAttr>(vpuKernelEntryAttrName).getValue());
    auto kernelSize = alignValUp(ELF::getKernelELF(moduleOp, kernelName, {".text"}).size(), CACHE_LINE_SIZE);
    return std::make_pair(kernelName, kernelSize);
}

std::pair<AddSwKernelInstructionPrefetch::SwKernelPrefetchVec, size_t>
AddSwKernelInstructionPrefetch::getPrefetchCandidatesAndFirstSwTask(mlir::Operation* funcOp,
                                                                    VPURT::TaskConfigVec& allTasks) {
    AddSwKernelInstructionPrefetch::SwKernelPrefetchVec kernelsToPrefetch{};
    size_t firstShvTaskIndex = 0;

    const size_t availableCacheSize = getAvailableCacheSize(funcOp->getParentOfType<mlir::ModuleOp>());
    VPUIP::ShaveL2CacheSimulator cache(availableCacheSize);
    _log.trace("Available Act Shave L2 cache size - {0}", availableCacheSize);

    for (size_t shvTaskIndex = 0; shvTaskIndex < allTasks.size(); shvTaskIndex++) {
        if (auto kernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(allTasks[shvTaskIndex].taskOp.getInnerTaskOp())) {
            // Break on JIT'ed kernels since we don't have any kind of timing information for them.
            // This invalidates the heuristic. Additionally they are not in binary cache at
            // the moment so we don't have the size available.
            if (vpux::VPUIP::isJitKernelOp(kernelOp) || vpux::VPUIP::isCacheHandlingOp(kernelOp)) {
                break;
            }

            auto kernelNameAndSize = getKernelNameAndSize(kernelOp);
            auto [kernelName, kernelSize] = kernelNameAndSize;
            _log.trace("Name - {0} Size - {1} free cache size - {2}", kernelName, kernelSize, cache.getFreeSize());
            if (kernelSize > cache.getFreeSize()) {
                break;
            }

            if (firstShvTaskIndex == 0) {
                firstShvTaskIndex = shvTaskIndex;
            }

            if (_useDummyKernelForInstructionPrefetch && llvm::find(SW_DUMMY_KERNELS_PREFETCH_SUPPORTED, kernelName) ==
                                                                 SW_DUMMY_KERNELS_PREFETCH_SUPPORTED.end()) {
                // If there are Shave kernels out of targets, we still need to consider them into account of cache.
                // Because they will be fetched into Shave engine firstly during inference.
                cache.loadKernel(kernelName, kernelSize);
                continue;
            }

            if (!cache.isLoaded(kernelName)) {
                kernelsToPrefetch.push_back(std::make_tuple(kernelName, kernelSize, shvTaskIndex));
            }
            cache.loadKernel(kernelName, kernelSize);

            if (_useDummyKernelForInstructionPrefetch && kernelNameToSymbol.count(kernelName) == 0) {
                auto swKernelSymbol = kernelOp.getKernelFunction();
                auto swKernelRun = *kernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
                kernelNameToSymbol[kernelName] = swKernelSymbol;
                kernelNameToArgs[kernelName] = swKernelRun.getAttrsAttr();
                kernelNameToOps[kernelName] = kernelOp;
            }
        }
    }

    _log.trace("kernelsToPrefetch - {0} firstShvTaskIndex - {1}", kernelsToPrefetch, firstShvTaskIndex);
    return std::make_pair(kernelsToPrefetch, firstShvTaskIndex);
}

std::tuple<mlir::Operation*, mlir::Value, size_t>
AddSwKernelInstructionPrefetch::getFirstSwTaskInIRAndBestUpdateBarrier(VPURT::InferenceExecutionSimulator& infSim,
                                                                       VPURT::TaskConfigVec& allTasks,
                                                                       size_t firstShvTaskIndex) {
    auto firstKernelTask = allTasks[firstShvTaskIndex];
    if (!mlir::isa<VPUIP::SwKernelOp>(firstKernelTask.taskOp.getInnerTaskOp())) {
        return std::make_tuple(nullptr, nullptr, 0);
    }

    int64_t bestVirtUpdateBarrier = 0;
    size_t bestReleaseCycle = 0;
    for (auto virtWaitBarrier : firstKernelTask.virtBarrierWaits) {
        auto virtBarrierConfig = infSim.getVirtBarrierConfig(virtWaitBarrier);
        if (virtBarrierConfig.getReleaseCycle() > bestReleaseCycle) {
            bestReleaseCycle = virtBarrierConfig.getReleaseCycle();
            bestVirtUpdateBarrier = virtWaitBarrier;
        }
    }
    _log.trace("First SW kernel start time {0}, best barrier release time {1}", firstKernelTask.cycleStart,
               bestReleaseCycle);
    if (bestReleaseCycle < _minimumFreeCyclesForPrefetch) {
        _log.info("bestReleaseCycle: {0} is smaller than _minimumFreeCyclesForPrefetch {1}, skipping prefetching",
                  bestReleaseCycle, _minimumFreeCyclesForPrefetch);
        return std::make_tuple(nullptr, nullptr, 0);
    }

    auto bestUpdateBarrier = infSim.getDeclareBarrierOp(bestVirtUpdateBarrier)->getResult(0);
    // firstShvTaskIndex here is a first SW task in the simulated inference order but we
    // must make sure to insert prefetch tasks before any other SW kernels in IR order
    // to avoid potential deadlocks. Mismatch between simulation order and IR order
    // can arise due to sorting of SW tasks which depend on the same barrier and have the
    // same startCycle. For instance following IR order:
    // VPUIP.SWKernelOp waits : %0 clusterIdx = 0
    // VPUIP.SWKernelOp waits : %0 clusterIdx = 1
    // VPUIP.SwKernelOp waits : %0 clusterIdx = 2
    // could be sorted into
    // VPUIP.SwKernelOp waits : %0 clusterIdx = 2
    // VPUIP.SWKernelOp waits : %0 clusterIdx = 1
    // VPUIP.SWKernelOp waits : %0 clusterIdx = 0
    // Inserting task before SWKernelOp on cluster 2 would lead into deadlock since prefetch
    // task would be updating barrier %0.
    auto firstShaveTaskInIR = getFirstSwTaskInIRWaitingForBarrier(bestUpdateBarrier);

    return std::make_tuple(firstShaveTaskInIR, bestUpdateBarrier, bestReleaseCycle);
}

std::vector<VPUIP::SwKernelOp> AddSwKernelInstructionPrefetch::insertPrefetchTasks(
        mlir::Operation* funcOp, AddSwKernelInstructionPrefetch::SwKernelPrefetchVec& kernelsToPrefetch,
        mlir::Operation* firstShaveTaskInIR, mlir::Value bestUpdateBarrier) {
    auto functionSymbol = getPrefetchSymbol(funcOp);
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = getNumTiles(moduleOp);
    const auto noOfShavesPerCluster =
            config::getTileExecutor(moduleOp).getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();
    _log.info("numClusters {0}, noOfShavesPerCluster: {1}", numClusters, noOfShavesPerCluster);
    std::vector<VPUIP::SwKernelOp> prefetchedKernels{};
    prefetchedKernels.reserve(numClusters * noOfShavesPerCluster);
    for (size_t shaveIdx = 0; (shaveIdx < numClusters * noOfShavesPerCluster) && (shaveIdx < kernelsToPrefetch.size());
         shaveIdx++) {
        auto clusterIdx = shaveIdx / noOfShavesPerCluster;
        auto [kernelName, kernelSize, shvTaskIndex] = kernelsToPrefetch[shaveIdx];
        _log.trace("Prefetching kernel {0} on cluster {1}", kernelName, clusterIdx);
        auto newPrefetchKernel =
                _useDummyKernelForInstructionPrefetch
                        ? insertDummyKernelOpBeforeFirstKernelTask(firstShaveTaskInIR, bestUpdateBarrier, clusterIdx,
                                                                   kernelName)
                        : insertPrefetchOpBeforeFirstKernelTask(firstShaveTaskInIR, bestUpdateBarrier, clusterIdx,
                                                                kernelName, functionSymbol);
        prefetchedKernels.push_back(newPrefetchKernel);
    }

    _log.info("Inserted {0} prefetch kernels", prefetchedKernels.size());

    return prefetchedKernels;
}

uint64_t findNextSaturationStart(size_t startIndex, vpux::VPURT::TaskConfigVec& allTasks, size_t numClusters,
                                 std::map<uint64_t, size_t>& swKernelCountsCache) {
    // Saturation is defined as 2x the number of clusters (e.g., 4 clusters -> 8 SW kernels)
    const size_t saturationThreshold = numClusters * 2;

    // Iterate through tasks strictly AFTER the startIndex
    for (size_t i = startIndex + 1; i < allTasks.size(); ++i) {
        uint64_t currentStartTime = static_cast<uint64_t>(allTasks[i].cycleStart);

        if (swKernelCountsCache.find(currentStartTime) == swKernelCountsCache.end()) {
            size_t swKernelCount = 0;
            // Count all SW Kernels that start at this specific time
            for (auto& task : allTasks) {
                if (static_cast<uint64_t>(task.cycleStart) == currentStartTime) {
                    if (mlir::isa<VPUIP::SwKernelOp>(task.taskOp.getInnerTaskOp())) {
                        swKernelCount++;
                    }
                }
                if (static_cast<uint64_t>(task.cycleStart) > currentStartTime) {
                    break;
                }
            }
            swKernelCountsCache[currentStartTime] = swKernelCount;
        }

        if (swKernelCountsCache[currentStartTime] >= saturationThreshold) {
            return currentStartTime;
        }
    }

    return std::numeric_limits<uint64_t>::max();
}

struct GapCandidate {
    uint64_t lookaheadGap = 0;
    int64_t insertionPointTaskIndex = -1;

    // used for sort
    bool operator>(const GapCandidate& other) const {
        return lookaheadGap > other.lookaheadGap;
    }
};

size_t getSwKernelCountAtTime(uint64_t startTime, VPURT::TaskConfigVec& allTasks) {
    size_t count = 0;
    for (auto& taskConfig : allTasks) {
        if (static_cast<uint64_t>(taskConfig.cycleStart) == startTime) {
            if (mlir::isa<VPUIP::SwKernelOp>(taskConfig.taskOp.getInnerTaskOp())) {
                count++;
            }
        }
        if (static_cast<uint64_t>(taskConfig.cycleStart) > startTime) {
            break;
        }
    }
    return count;
}

std::optional<GapCandidate> findBestInsertionGap(const std::string& kernelName, uint64_t targetKernelGroupStartTime,
                                                 VPURT::TaskConfigVec& allTasks, size_t numClusters, Logger& log) {
    const int64_t targetInsertTile = 1;
    const uint64_t GAP_THRESHOLD = 50000;
    const size_t saturationThreshold = numClusters * 2;

    // <LookaheadGapSize, GapCandidate>
    std::map<uint64_t, GapCandidate, std::greater<uint64_t>> validGaps;
    std::map<uint64_t, size_t> swKernelCountsCache;  // local cache

    int64_t previousT1TaskIndex = -1;
    uint64_t previousT1TaskStartTime = 0;

    // find the largest gap between a non-saturated SW task and a saturated SW task / the kernel to be prefetched
    for (size_t i = 0; i < allTasks.size(); ++i) {
        auto& currentTaskConfig = allTasks[i];
        uint64_t currentTaskStartTime = static_cast<uint64_t>(currentTaskConfig.cycleStart);
        if (currentTaskStartTime > targetKernelGroupStartTime) {
            break;
        }

        bool isT1Task = false;
        if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(currentTaskConfig.taskOp.getInnerTaskOp()); swOp != nullptr) {
            isT1Task = (swOp.getTileIndexAttr().getInt() == targetInsertTile);
        }

        if (previousT1TaskIndex != -1 && isT1Task) {
            auto& insertionPointTask = allTasks[previousT1TaskIndex];
            auto insertionPointStartTime = static_cast<uint64_t>(insertionPointTask.cycleStart);

            size_t simultaneousSwKernels = getSwKernelCountAtTime(insertionPointStartTime, allTasks);

            if (simultaneousSwKernels < saturationThreshold) {
                uint64_t nextSaturationStart =
                        findNextSaturationStart(previousT1TaskIndex, allTasks, numClusters, swKernelCountsCache);
                uint64_t gapEnd = std::min(nextSaturationStart, targetKernelGroupStartTime);
                uint64_t lookaheadGap = 0;
                if (gapEnd > previousT1TaskStartTime) {
                    lookaheadGap = gapEnd - previousT1TaskStartTime;
                }

                if (lookaheadGap >= GAP_THRESHOLD) {
                    GapCandidate gap;
                    gap.lookaheadGap = lookaheadGap;
                    gap.insertionPointTaskIndex = previousT1TaskIndex;
                    validGaps[lookaheadGap] = gap;
                }
            }
        }

        if (isT1Task) {
            previousT1TaskIndex = static_cast<int64_t>(i);
            previousT1TaskStartTime = currentTaskStartTime;
        }
    }

    if (validGaps.empty()) {
        log.trace("Kernel '{0}': No suitable insertion point found.", kernelName);
        return std::nullopt;
    }

    return validGaps.begin()->second;
}

std::vector<VPUIP::SwKernelOp> AddSwKernelInstructionPrefetch::insertPrefetchTasksDuringExec(
        mlir::Operation* funcOp, AddSwKernelInstructionPrefetch::SwKernelPrefetchVec& kernelsToPrefetch,
        VPURT::TaskConfigVec& allTasks) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = getNumTiles(moduleOp);
    VPUX_THROW_WHEN(numClusters == 0, "Number of tiles is zero.");

    std::vector<VPUIP::SwKernelOp> prefetchedKernels{};

    for (auto& kernelInfo : kernelsToPrefetch) {
        std::string kernelName = std::get<0>(kernelInfo);
        size_t firstAppearanceIndex = std::get<2>(kernelInfo);

        if (firstAppearanceIndex >= allTasks.size()) {
            _log.trace("Skipping kernel '{0}': Invalid firstAppearanceIndex {1}", kernelName, firstAppearanceIndex);
            continue;
        }
        if (kernelNameToOps.count(kernelName) == 0) {
            _log.trace("Skipping kernel '{0}': Missing dependencies (kernelNameToOps)", kernelName);
            continue;
        }

        auto targetKernelGroupStartTime = static_cast<uint64_t>(allTasks[firstAppearanceIndex].cycleStart);

        auto bestGapOpt = findBestInsertionGap(kernelName, targetKernelGroupStartTime, allTasks, numClusters, _log);

        if (!bestGapOpt.has_value()) {
            _log.trace("Kernel '{0}': No valid gap found.", kernelName);
            continue;
        }

        GapCandidate bestGap = bestGapOpt.value();
        _log.trace("Kernel '{0}': Found best gap of {1} cycles. Inserting relative to task {2}.", kernelName,
                   bestGap.lookaheadGap, bestGap.insertionPointTaskIndex);

        if (bestGap.insertionPointTaskIndex < 0 ||
            static_cast<size_t>(bestGap.insertionPointTaskIndex) >= allTasks.size()) {
            _log.error("Kernel '{0}': Invalid insertionPointTaskIndex {1}. Skipping insertion.", kernelName,
                       bestGap.insertionPointTaskIndex);
            continue;
        }

        auto insertBeforeOp = allTasks[bestGap.insertionPointTaskIndex].taskOp;
        size_t dynamicExecTile = _dynamicPrefetchTileCounter % numClusters;
        _dynamicPrefetchTileCounter++;

        auto newPrefetchKernel = insertDummyKernelOpBeforeFirstKernelTask(insertBeforeOp, mlir::ValueRange(),
                                                                          dynamicExecTile, kernelName);

        prefetchedKernels.push_back(newPrefetchKernel);
    }

    return prefetchedKernels;
}

void AddSwKernelInstructionPrefetch::safeRunOnFunc() {
    auto funcOp = getOperation();
    if (!hasVPUSWModule(funcOp)) {
        _log.trace("No SW kernels in schedule");
        return;
    }

    auto simLogger = vpux::Logger("InfSim", _log.level());
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);
    CycleCostInfo cycleCostInfo(std::move(costModel), funcOp);

    VPURT::InferenceExecutionSimulator infSim(simLogger, funcOp, cycleCostInfo);
    _log.trace("Start inference schedule simulation and update cycles");
    infSim.runSim();
    _log.trace("Finished simulation");

    auto allTasks = infSim.getTaskCycleConfig();
    std::stable_sort(allTasks.begin(), allTasks.end(),
                     [](const VPURT::TaskConfig& first, const VPURT::TaskConfig& second) {
                         return first.cycleStart < second.cycleStart;
                     });

    auto [useDummyKernelForInstructionPrefetch, minimumShaveStartTimeForPrefetch] =
            vpux::VPUIP::getSwKernelInstructionPrefetchConfig(arch);
    _useDummyKernelForInstructionPrefetch = useDummyKernelForInstructionPrefetch;
    _minimumFreeCyclesForPrefetch =
            _minFreeCyclesHasValue ? _minimumFreeCyclesForPrefetch : minimumShaveStartTimeForPrefetch;
    _log.trace("_useDummyKernelForInstructionPrefetch is {0}", _useDummyKernelForInstructionPrefetch);
    auto [kernelsToPrefetch, firstShvTaskIndex] = getPrefetchCandidatesAndFirstSwTask(funcOp, allTasks);
    auto [firstShaveTaskInIR, bestUpdateBarrier, bestReleaseCycle] =
            getFirstSwTaskInIRAndBestUpdateBarrier(infSim, allTasks, firstShvTaskIndex);

    if (_useDummyKernelForInstructionPrefetch) {
        auto memSpaceAttr = mlir::SymbolRefAttr::get(module->getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
        auto dummyKernelResMem = config::getDummySwKernelsForInstructionPrefetchReservedMemory(module, memSpaceAttr);
        VPUX_THROW_WHEN(dummyKernelResMem == nullptr,
                        "Cannot find DummySWKernelsForInstructionPrefetchReservedMemory!");
    }
    if (kernelsToPrefetch.empty()) {
        return;
    }
    _log.trace("insertPoint: {0}, bestReleaseCycle: {1}", *firstShaveTaskInIR, bestReleaseCycle);

    auto newPrefetchKernels =
            (firstShaveTaskInIR == nullptr)
                    ? insertPrefetchTasksDuringExec(funcOp, kernelsToPrefetch, allTasks)
                    : insertPrefetchTasks(funcOp, kernelsToPrefetch, firstShaveTaskInIR, bestUpdateBarrier);

    // Update dependencies for cache handling operations to meet requirements of control graph split.
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);
    BarrierInfo::TaskSet prefetchTasks;
    for (auto swKernelOp : newPrefetchKernels) {
        auto taskOp = swKernelOp->getParentOfType<VPURT::TaskOp>();
        auto taskInd = barrierInfo.getIndex(taskOp);
        prefetchTasks.insert(taskInd);
        _log.trace("New prefetch op: {0} task index: {1}", taskOp, taskInd);
    }

    // Check and correct dependencies of the prefetch tasks
    bool dependenciesChanged = barrierInfo.adjustTasksDependenciesToGraphSplitConstraints(prefetchTasks);
    if (dependenciesChanged) {
        _log.trace("Dependencies changed - updating IR");
        barrierInfo.updateIR();
    }

    VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo.clearAttributes();
}

}  // namespace

//
// createAddSwKernelInstructionPrefetchPass
//
std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddSwKernelInstructionPrefetchPass(Logger log) {
    return std::make_unique<AddSwKernelInstructionPrefetch>(log);
}
