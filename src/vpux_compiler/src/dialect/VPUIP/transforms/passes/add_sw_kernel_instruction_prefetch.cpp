//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/cache_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/STLExtras.h>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADDSWKERNELINSTRUCTIONPREFETCH
#define GEN_PASS_DEF_ADDSWKERNELINSTRUCTIONPREFETCH
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

static const std::unordered_set<std::string> supportedDummyKernels = {"activation_exp",
                                                                      "activation_tanh",
                                                                      "activation_hswish",
                                                                      "activation_hsigmoid",
                                                                      "activation_sigmoid",
                                                                      "activation_softsign",
                                                                      "activation_sqrt",
                                                                      "activation_erf",
                                                                      "activation_ceil",
                                                                      "activation_relu",
                                                                      "activation_negative",
                                                                      "activation_sin",
                                                                      "activation_cos",
                                                                      "activation_sinh",
                                                                      "activation_cosh",
                                                                      "activation_floor",
                                                                      "activation_sign",
                                                                      "activation_tan",
                                                                      "activation_asin",
                                                                      "activation_acos",
                                                                      "activation_atan",
                                                                      "activation_asinh",
                                                                      "activation_atanh",
                                                                      "activation_softplus",
                                                                      "activation_abs",
                                                                      "activation_gelu",

                                                                      "eltwise_mul",
                                                                      "eltwise_add",
                                                                      "eltwise_sub",
                                                                      "eltwise_min",
                                                                      "eltwise_max",
                                                                      "eltwise_equal",
                                                                      "eltwise_greater",
                                                                      "eltwise_greater_equal",
                                                                      "eltwise_less",
                                                                      "eltwise_isinf",
                                                                      "eltwise_isfinite",
                                                                      "eltwise_logical_or",
                                                                      "eltwise_logical_xor",
                                                                      "eltwise_logical_not",
                                                                      "eltwise_and",
                                                                      "eltwise_bitwise_and",
                                                                      "eltwise_bitwise_or",
                                                                      "eltwise_bitwise_xor",
                                                                      "eltwise_bitwise_not",
                                                                      "eltwise_bitwise_left_shift",
                                                                      "eltwise_bitwise_right_shift",
                                                                      "eltwise_not_equal",
                                                                      "activation_mish",

                                                                      "adaptive_pool",
                                                                      "adaptive_max_pool",

                                                                      "dynamic_quantize",
                                                                      "dynamic_dequantize",

                                                                      "convert_p00",
                                                                      "convert_p01",
                                                                      "convert",

                                                                      "softmax",

                                                                      "rms_norm",
                                                                      "rms_norm_p00"};
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

        if (minimumShaveIdleTimeForPrefetch.hasValue()) {
            _minimumFreeCyclesForPrefetch = minimumShaveIdleTimeForPrefetch.getValue();
        }

        if (minimumShaveInitTimeDummyKernels.hasValue()) {
            _minimumTimeForShaveInitDummyKernels = minimumShaveInitTimeDummyKernels.getValue();
        }

        if (minimumShaveInitTime.hasValue()) {
            _minimumTimeForShaveInit = minimumShaveInitTime.getValue();
        }

        return mlir::success();
    }

private:
    void safeRunOnFunc() final;

    void updateBarrierInfo(mlir::func::FuncOp funcOp, std::vector<VPUIP::SwKernelOp>& prefetchKernels);

    mlir::SymbolRefAttr getPrefetchSymbol(mlir::Operation* funcOp);
    VPUIP::SwKernelOp insertPrefetchTask(mlir::Operation* firstSwTask, mlir::Value waitBarrier,
                                         mlir::Value updateBarrier, size_t clusterIdx, std::string& kernelName,
                                         mlir::SymbolRefAttr functionSymbol);

    bool dummyKernelPrefetchSupported(const std::string& kernelName);
    std::optional<size_t> getDummyKernelOffsetReservedMem(const mlir::ModuleOp module);
    SmallVector<mlir::Attribute> getDummyKernelAttrs(const std::string& kernelName);
    VPUIP::SwKernelOp insertDummyKernelPrefetchTask(mlir::Operation* firstSwTask, VPUIP::SwKernelOp kernelOp,
                                                    mlir::Value waitBarrier, mlir::Value updateBarrier,
                                                    size_t clusterIdx, std::string& kernelName);

    mlir::Value getBestWaitBarrier(VPURT::InferenceExecutionSimulator& infSim, VPURT::TaskConfigVec& allTasks,
                                   size_t prefetchIntervalStartTaskIdx);
    mlir::Operation* getFirstSwTaskInIRWaitingForBarrier(mlir::Value waitBarrier);
    static std::string getKernelName(VPUIP::SwKernelOp swKernelOp);

    struct KernelPrefetchDesc {
        std::string kernelName;
        size_t kernelIndex;
    };
    using SwKernelPrefetchVec = std::vector<KernelPrefetchDesc>;

    // Used when finding all shave idle intervals.
    // Init - state of shaves at the start of inference. Separate from Start to indicate
    //        that taskIdx is not a valid shave kernel.
    // Start - one of the SHAVEs started executing kernel. SHAVEs no longer idle
    // End - all shaves finished executing kernels. All SHAVEs are idle
    enum class ShaveEventType { Init, Start, End };

    struct ShaveEvent {
        size_t time;
        size_t taskIdx;
        ShaveEventType eventType;
        long tileId;
    };

    struct PrefetchInterval {
        ShaveEvent start;
        ShaveEvent end;
        SwKernelPrefetchVec kernelsToPrefetch;
    };

    std::vector<PrefetchInterval> getPrefetchIntervals(VPURT::TaskConfigVec& allTasks, size_t numTiles,
                                                       bool useDummyKernels);

    std::tuple<mlir::Operation*, mlir::Value, size_t> getFirstSwTaskInIRAndBestUpdateBarrier(
            VPURT::InferenceExecutionSimulator& infSim, VPURT::TaskConfigVec& allTasks, size_t firstShvTaskIndex);

    std::vector<VPUIP::SwKernelOp> insertPrefetchTasks(mlir::Operation* funcOp, VPURT::TaskConfigVec& allTasks,
                                                       SwKernelPrefetchVec& kernelsToPrefetch,
                                                       mlir::Operation* firstShaveTaskInIR, mlir::Value bestWaitBarrier,
                                                       mlir::Value bestUpdateBarrier, bool useDummyKernels);

    bool hasVPUSWModule(mlir::Operation* funcOp);
    size_t getNumTiles(mlir::ModuleOp moduleOp);

    static constexpr StringLiteral vpuKernelEntryAttrName{"VPU.kernel_entry"};

    size_t _minimumFreeCyclesForPrefetch = 50000;
    // Below variables are unlikely to actually represent shave init time
    // but they are used by this pass to penalize prefetch in first few
    // intervals. This has been experimentally found to produce good results
    // and part of the component of this time is shave init time but it is not
    // clear what percentage of below values is taken by actual shave init.
    size_t _minimumTimeForShaveInit = 200000;
    size_t _minimumTimeForShaveInitDummyKernels = 100000;
};

bool AddSwKernelInstructionPrefetch::dummyKernelPrefetchSupported(const std::string& kernelName) {
    return supportedDummyKernels.find(kernelName) != supportedDummyKernels.end();
}

SmallVector<mlir::Attribute> AddSwKernelInstructionPrefetch::getDummyKernelAttrs(const std::string& kernelName) {
    if (kernelName == "softmax") {
        auto zeroAttr = getIntAttr(&getContext(), 0);
        return {zeroAttr, zeroAttr};
    } else if (kernelName == "rms_norm" || kernelName == "rms_norm_p00") {
        auto epsAttr = mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), 0.1);
        return {epsAttr};
    }

    return {};
}

std::optional<size_t> AddSwKernelInstructionPrefetch::getDummyKernelOffsetReservedMem(const mlir::ModuleOp module) {
    auto cachePrefetchMem =
            config::getDummySwKernelsForInstructionPrefetchReservedMemory(module, VPU::MemoryKind::CMX_NN);
    if (!cachePrefetchMem) {
        return std::nullopt;
    }
    auto offsetCachePrefetch = cachePrefetchMem.getOffset();
    VPUX_THROW_WHEN(!offsetCachePrefetch.has_value(),
                    "DummySwKernelsForInstructionPrefetchReservedMemory offset is not set!");
    return offsetCachePrefetch.value();
}

// For LNL, Shave kernel instruction prefetch needs to insert a dummy kernel instead of prefetch kernel
VPUIP::SwKernelOp AddSwKernelInstructionPrefetch::insertDummyKernelPrefetchTask(
        mlir::Operation* firstSwTask, VPUIP::SwKernelOp kernelOp, mlir::Value waitBarrier, mlir::Value updateBarrier,
        size_t clusterIdx, std::string& kernelName) {
    mlir::OpBuilder builder(firstSwTask);
    auto moduleOp = firstSwTask->getParentOfType<mlir::ModuleOp>();
    auto maybeReservedMemOffset = getDummyKernelOffsetReservedMem(moduleOp);
    if (!maybeReservedMemOffset) {
        _log.warning("Can't insert dummy kernel prefetch due to missing memory allocation");
        return nullptr;
    }
    auto reservedMemOffset = maybeReservedMemOffset.value();
    auto offsetAttrIn = getIntAttr(moduleOp->getContext(), reservedMemOffset);
    auto offsetAttrOut = getIntAttr(moduleOp->getContext(), reservedMemOffset + 8);

    const auto memSpaceCMX =
            IndexedSymbolAttr::get(&getContext(), stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), clusterIdx);
    const auto sectionCMX = VPURT::BufferSectionAttr::get(&getContext(), VPURT::BufferSection::CMX_NN);
    auto createBuffer = [&](mlir::Value io, StringRef suffix, mlir::IntegerAttr offset,
                            mlir::SmallVector<mlir::Value>& buffers) {
        if (auto bufOp = io.getDefiningOp<VPURT::DeclareBufferOp>()) {
            auto newType =
                    mlir::cast<NDTypeInterface>(io.getType()).changeMemSpace(memSpaceCMX).changeShape({1, 1, 1, 1});
            auto newBuff = builder.create<VPURT::DeclareBufferOp>(
                    appendLoc(bufOp->getLoc(), suffix), newType, sectionCMX,
                    getIntArrayAttr(&getContext(), mlir::SmallVector<int>{static_cast<int>(clusterIdx)}), offset,
                    bufOp.getSwizzlingKeyAttr());
            buffers.push_back(newBuff);
        }
    };

    mlir::SmallVector<mlir::Value> srcBuffers, dstBuffers;

    for (auto input : kernelOp.getInputs()) {
        createBuffer(input, "prefetch_src", offsetAttrIn, srcBuffers);
    }
    for (auto output : kernelOp.getOutputBuffs()) {
        createBuffer(output, "prefetch_dst", offsetAttrOut, dstBuffers);
    }

    VPUX_THROW_WHEN(srcBuffers.empty() || dstBuffers.empty(),
                    "Got empty buffers during dummy shave kernel collecting I/O for instruction prefetch.");

    auto newLoc = appendLoc(firstSwTask->getLoc(), "prefetch_{0}", kernelName);
    if (stringifyPrimaryLocation(newLoc).find("/cluster_") == std::string::npos) {
        newLoc = appendLoc(newLoc, "cluster_{0}", clusterIdx);
    }

    mlir::SmallVector<mlir::Value> buffers;
    if (waitBarrier) {
        buffers.push_back(waitBarrier);
    }
    const auto buffersRange = mlir::ValueRange(buffers);

    auto swKernelSymbol = kernelOp.getKernelFunction();

    auto cachePrefetchSwKernel = vpux::VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
            builder, buffersRange, updateBarrier, newLoc, mlir::ValueRange(srcBuffers), mlir::ValueRange(dstBuffers),
            nullptr, swKernelSymbol, getIntAttr(builder, clusterIdx));

    auto args = getDummyKernelAttrs(kernelName);
    vpux::VPUIP::initSwKernel(cachePrefetchSwKernel, mlir::ValueRange(srcBuffers), mlir::ValueRange(dstBuffers), args,
                              _log.nest(), /*swKernelRunOp=*/nullptr);

    _log.trace("cachePrefetchSwKernel {0}", cachePrefetchSwKernel);
    return cachePrefetchSwKernel;
}

mlir::SymbolRefAttr AddSwKernelInstructionPrefetch::getPrefetchSymbol(mlir::Operation* funcOp) {
    auto ctx = funcOp->getContext();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    auto vpuswModule = vpux::VPUIP::getVPUSWModule(moduleOp, _log);

    const std::string functionName = "cache_prefetch";
    auto functionNameSymbol = mlir::SymbolRefAttr::get(ctx, functionName);
    VPUX_THROW_UNLESS(vpuswModule.lookupSymbol<mlir::func::FuncOp>(functionName), "No prefetch kernel found");
    return mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {functionNameSymbol});
}

size_t AddSwKernelInstructionPrefetch::getNumTiles(mlir::ModuleOp moduleOp) {
    auto tileOp = config::getTileExecutor(moduleOp);
    return tileOp.getCount();
}

VPUIP::SwKernelOp AddSwKernelInstructionPrefetch::insertPrefetchTask(mlir::Operation* firstSwTask,
                                                                     mlir::Value waitBarrier, mlir::Value updateBarrier,
                                                                     size_t clusterIdx, std::string& kernelName,
                                                                     mlir::SymbolRefAttr functionSymbol) {
    mlir::OpBuilder builder(firstSwTask);
    mlir::SmallVector<mlir::Value> buffers = {};
    const auto buffersRange = mlir::ValueRange(buffers);
    auto updateBarriers = updateBarrier;
    mlir::SmallVector<mlir::Value> waitBarriersVec;
    if (waitBarrier) {
        waitBarriersVec.push_back(waitBarrier);
    }
    auto waitBarriers = mlir::ValueRange(waitBarriersVec);
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

std::string AddSwKernelInstructionPrefetch::getKernelName(VPUIP::SwKernelOp swKernelOp) {
    auto moduleOp = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto swKernelSymbol = swKernelOp.getKernelFunction();
    auto kernelInfoFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(swKernelSymbol);
    return std::string(kernelInfoFuncOp->getAttrOfType<mlir::StringAttr>(vpuKernelEntryAttrName).getValue());
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

mlir::Value AddSwKernelInstructionPrefetch::getBestWaitBarrier(VPURT::InferenceExecutionSimulator& infSim,
                                                               VPURT::TaskConfigVec& allTasks,
                                                               size_t prefetchIntervalStartTaskIdx) {
    auto prefetchIntervalStartTask = allTasks[prefetchIntervalStartTaskIdx];
    if (!mlir::isa<VPUIP::SwKernelOp>(prefetchIntervalStartTask.taskOp.getInnerTaskOp())) {
        return nullptr;
    }

    size_t bestReleaseCycle = std::numeric_limits<size_t>::max();
    int64_t bestVirtWaitBarrier = 0;
    for (auto virtUpdateBarrier : prefetchIntervalStartTask.virtBarrierUpdates) {
        auto virtBarrierConfig = infSim.getVirtBarrierConfig(virtUpdateBarrier);
        if (virtBarrierConfig.getReleaseCycle() < bestReleaseCycle) {
            bestReleaseCycle = virtBarrierConfig.getReleaseCycle();
            bestVirtWaitBarrier = virtUpdateBarrier;
        }
    }

    if (bestReleaseCycle == std::numeric_limits<size_t>::max()) {
        return nullptr;
    }

    auto bestWaitBarrier = infSim.getDeclareBarrierOp(bestVirtWaitBarrier)->getResult(0);
    return bestWaitBarrier;
}

std::vector<VPUIP::SwKernelOp> AddSwKernelInstructionPrefetch::insertPrefetchTasks(
        mlir::Operation* funcOp, VPURT::TaskConfigVec& allTasks,
        AddSwKernelInstructionPrefetch::SwKernelPrefetchVec& kernelsToPrefetch, mlir::Operation* firstShaveTaskInIR,
        mlir::Value bestWaitBarrier, mlir::Value bestUpdateBarrier, bool useDummyKernels) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = getNumTiles(moduleOp);
    const auto noOfShavesPerCluster =
            config::getTileExecutor(moduleOp).getSubExecutor(config::ExecutorKind::SHAVE_ACT).getCount();
    _log.info("numClusters {0}, noOfShavesPerCluster: {1}", numClusters, noOfShavesPerCluster);

    std::vector<VPUIP::SwKernelOp> prefetchKernels{};
    prefetchKernels.reserve(numClusters * noOfShavesPerCluster);
    for (size_t shaveIdx = 0; (shaveIdx < numClusters * noOfShavesPerCluster) && (shaveIdx < kernelsToPrefetch.size());
         shaveIdx++) {
        auto clusterIdx = shaveIdx / noOfShavesPerCluster;
        auto kernelName = kernelsToPrefetch[shaveIdx].kernelName;
        auto kernelOp = mlir::cast<VPUIP::SwKernelOp>(
                allTasks[kernelsToPrefetch[shaveIdx].kernelIndex].taskOp.getInnerTaskOp());
        _log.trace("Prefetching kernel {0} on cluster {1}", kernelName, clusterIdx);
        if (useDummyKernels) {
            auto newDummyKernel = insertDummyKernelPrefetchTask(firstShaveTaskInIR, kernelOp, bestWaitBarrier,
                                                                bestUpdateBarrier, clusterIdx, kernelName);
            if (newDummyKernel) {
                prefetchKernels.push_back(newDummyKernel);
            }
        } else {
            auto functionSymbol = getPrefetchSymbol(funcOp);
            auto newPrefetchKernel = insertPrefetchTask(firstShaveTaskInIR, bestWaitBarrier, bestUpdateBarrier,
                                                        clusterIdx, kernelName, functionSymbol);
            if (newPrefetchKernel) {
                prefetchKernels.push_back(newPrefetchKernel);
            }
        }
    }

    _log.info("Inserted {0} prefetch kernels", prefetchKernels.size());
    return prefetchKernels;
}

std::vector<AddSwKernelInstructionPrefetch::PrefetchInterval> AddSwKernelInstructionPrefetch::getPrefetchIntervals(
        VPURT::TaskConfigVec& allTasks, size_t numTiles, bool useDummyKernels) {
    std::vector<ShaveEvent> shaveEvents;
    std::vector<PrefetchInterval> prefetchIntervals;
    for (size_t taskIdx = 0; taskIdx < allTasks.size(); taskIdx++) {
        if (auto kernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(allTasks[taskIdx].taskOp.getInnerTaskOp())) {
            auto maybeTileIdx = kernelOp.getTileIndex();
            VPUX_THROW_UNLESS(maybeTileIdx.has_value(), "SW kernel op has no tile idx");
            auto tileIdx = maybeTileIdx.value();
            size_t taskEnd = allTasks[taskIdx].cycleStart + allTasks[taskIdx].cycleCost;
            size_t taskStart = allTasks[taskIdx].cycleStart;
            shaveEvents.push_back(
                    {taskStart, static_cast<size_t>(taskIdx), ShaveEventType::Start, static_cast<long>(tileIdx)});
            shaveEvents.push_back(
                    {taskEnd, static_cast<size_t>(taskIdx), ShaveEventType::End, static_cast<long>(tileIdx)});
        }
    }
    std::stable_sort(shaveEvents.begin(), shaveEvents.end(), [](const ShaveEvent& event1, const ShaveEvent& event2) {
        return event1.time < event2.time;
    });

    ShaveEvent lastEventAllIdle{0, 0, ShaveEventType::Init, 0};
    std::vector<bool> engineIdle(numTiles, true);
    auto minimumTimeForShaveInit = useDummyKernels ? _minimumTimeForShaveInitDummyKernels : _minimumTimeForShaveInit;
    auto identity = [](bool x) {
        return x;
    };
    // Find intervals in schedule where all SHAVE engines are idle. The idea behind ensuring all shaves
    // are idle is to make sure instruction prefetch does not collide with actual kernels fetching their instructions
    // and data from DDR(since all SHAVEs share interface to the SOC fabric).
    // It also makes algorithm slightly simpler since we don't have to track intervals per-SHAVE instance.
    // Note that it was never tested if prefetch will collide with kernel instruction and data fetch so it is
    // a potential room for improvement.
    for (auto event : shaveEvents) {
        if (event.eventType == ShaveEventType::Start) {
            if (llvm::all_of(engineIdle, identity)) {
                size_t intervalSize = event.time - lastEventAllIdle.time;
                if (lastEventAllIdle.time < minimumTimeForShaveInit) {
                    auto remainingTimeForShaveInit = minimumTimeForShaveInit - lastEventAllIdle.time;
                    intervalSize =
                            intervalSize > remainingTimeForShaveInit ? intervalSize - remainingTimeForShaveInit : 0;
                }
                if (intervalSize > _minimumFreeCyclesForPrefetch) {
                    prefetchIntervals.push_back({lastEventAllIdle, event, {}});
                }
                // Only prefetch in first interval. This is due to unpredictable
                // impact on WLM schedule when prefetch tasks are inserted in the middle of the schedule.
                break;
            }
            engineIdle[event.tileId] = false;
        } else {
            engineIdle[event.tileId] = true;
            if (llvm::all_of(engineIdle, identity)) {
                lastEventAllIdle = event;
            }
        }
    }

    if (prefetchIntervals.empty()) {
        return prefetchIntervals;
    }
    std::unordered_set<std::string> seenKernels;
    auto activeInterval = prefetchIntervals.begin();
    auto nextInterval = activeInterval + 1;
    for (size_t taskIdx = 0; taskIdx < allTasks.size(); taskIdx++) {
        auto task = allTasks[taskIdx];
        if (activeInterval == prefetchIntervals.end()) {
            break;
        }
        if (auto kernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(task.taskOp.getInnerTaskOp())) {
            if (vpux::VPUIP::isJitKernelOp(kernelOp) || vpux::VPUIP::isCacheHandlingOp(kernelOp)) {
                activeInterval = nextInterval;
                if (nextInterval != prefetchIntervals.end()) {
                    nextInterval++;
                }
                continue;
            }

            // If the task is located after nextInterval switch to it if it is also larger then
            // the current interval. If it is not stay at the current interval to allow for maximum
            // margin of error.
            if (nextInterval != prefetchIntervals.end()) {
                if (static_cast<size_t>(task.cycleStart) > nextInterval->end.time) {
                    auto activeIntervalDuration = activeInterval->end.time - activeInterval->start.time;
                    auto nextIntervalDuration = nextInterval->end.time - nextInterval->start.time;
                    if (activeIntervalDuration < nextIntervalDuration) {
                        activeInterval = nextInterval;
                    }
                    nextInterval++;
                }
            }

            auto name = getKernelName(kernelOp);
            // For networks that have SW kernel at the start it is possible that first activeInterval
            // is located after the first few SW tasks. Below check skips those.
            if (static_cast<size_t>(task.cycleStart) < activeInterval->start.time) {
                seenKernels.insert(std::move(name));
                continue;
            }
            if (seenKernels.find(name) != seenKernels.end()) {
                continue;
            }
            if (useDummyKernels && !dummyKernelPrefetchSupported(name)) {
                continue;
            }
            activeInterval->kernelsToPrefetch.push_back(KernelPrefetchDesc{name, taskIdx});
            seenKernels.insert(std::move(name));
        }
    }

    return prefetchIntervals;
}

bool AddSwKernelInstructionPrefetch::hasVPUSWModule(mlir::Operation* funcOp) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};
    auto innerModule = moduleOp.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    return innerModule;
}

void AddSwKernelInstructionPrefetch::updateBarrierInfo(mlir::func::FuncOp funcOp,
                                                       std::vector<VPUIP::SwKernelOp>& prefetchKernels) {
    // Update dependencies for cache handling operations to meet requirements of control graph split.
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);
    BarrierInfo::TaskSet prefetchTasks;
    for (auto swKernelOp : prefetchKernels) {
        auto taskOp = swKernelOp->getParentOfType<VPURT::TaskOp>();
        auto taskInd = barrierInfo.getIndex(taskOp);
        prefetchTasks.insert(taskInd);
        _log.trace("New prefetch op: {0} task index: {1}", taskOp, taskInd);
    }

    bool dependenciesChanged = barrierInfo.adjustTasksDependenciesToGraphSplitConstraints(prefetchTasks);

    if (dependenciesChanged) {
        _log.trace("Dependencies changed - updating IR");
        barrierInfo.updateIR();
    }

    VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo.clearAttributes();
}

void AddSwKernelInstructionPrefetch::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    if (!hasVPUSWModule(funcOp)) {
        _log.trace("No SW kernels in schedule");
        return;
    }

    bool useDummyKernels = config::getArch(module) == config::ArchKind::NPU40XX;
    if (useDummyKernels) {
        _log.info("using dummy kernels for prefetch");
    }
    _log.trace("minimum free cycles for prefetch {0}", _minimumFreeCyclesForPrefetch);

    auto simLogger = vpux::Logger("InfSim", _log.level());
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, &getContext(), _log);
    CycleCostInfo cycleCostInfo(std::move(costModel), funcOp);

    VPURT::InferenceExecutionSimulator infSim(simLogger, funcOp, cycleCostInfo);
    infSim.runSim();

    auto allTasks = infSim.getTaskCycleConfig();
    std::stable_sort(allTasks.begin(), allTasks.end(),
                     [](const VPURT::TaskConfig& first, const VPURT::TaskConfig& second) {
                         return first.cycleStart < second.cycleStart;
                     });

    auto prefetchIntervals = getPrefetchIntervals(allTasks, getNumTiles(module), useDummyKernels);
    std::vector<VPUIP::SwKernelOp> prefetchKernels;

    for (auto& prefetchInterval : prefetchIntervals) {
        if (prefetchInterval.kernelsToPrefetch.empty()) {
            continue;
        }

        mlir::Value bestWaitBarrier = nullptr;
        if (prefetchInterval.start.eventType == ShaveEventType::End) {
            bestWaitBarrier = getBestWaitBarrier(infSim, allTasks, prefetchInterval.start.taskIdx);
        }
        auto [firstShaveTaskInIR, bestUpdateBarrier, bestReleaseCycle] =
                getFirstSwTaskInIRAndBestUpdateBarrier(infSim, allTasks, prefetchInterval.end.taskIdx);
        if (!firstShaveTaskInIR) {
            continue;
        }

        _log.trace("insertPoint: {0}, bestReleaseCycle: {1}", *firstShaveTaskInIR, bestReleaseCycle);
        auto newPrefetchKernels =
                insertPrefetchTasks(funcOp, allTasks, prefetchInterval.kernelsToPrefetch, firstShaveTaskInIR,
                                    bestWaitBarrier, bestUpdateBarrier, useDummyKernels);
        prefetchKernels.insert(prefetchKernels.end(), newPrefetchKernels.begin(), newPrefetchKernels.end());
    }

    if (!prefetchKernels.empty()) {
        updateBarrierInfo(funcOp, prefetchKernels);
    }
}

}  // namespace

//
// createAddSwKernelInstructionPrefetchPass
//
std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddSwKernelInstructionPrefetchPass(Logger log) {
    return std::make_unique<AddSwKernelInstructionPrefetch>(log);
}
