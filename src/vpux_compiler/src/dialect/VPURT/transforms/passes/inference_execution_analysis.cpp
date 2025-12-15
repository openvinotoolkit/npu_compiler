//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

#include <fstream>

namespace vpux::VPURT {
#define GEN_PASS_DECL_INFERENCEEXECUTIONANALYSIS
#define GEN_PASS_DEF_INFERENCEEXECUTIONANALYSIS
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

size_t getClusterId(VPUIP::DPUTaskOp op) {
    return op.getClusterId().value_or(0);
}

size_t getClusterId(VPUIP::SwKernelOp op) {
    return op.getTileIndex().value_or(0);
}

size_t getClusterId(VPUIP::NCEClusterTaskOp nceOp) {
    const auto dpuTasks = nceOp.getVariants().getOps<VPUIP::DPUTaskOp>();
    return dpuTasks.empty() ? 0 : getClusterId(*dpuTasks.begin());
}

profiling::TaskInfo makeTaskInfo(VPURT::TaskOp taskOp, double startTimeNs, double durationNs, Logger log,
                                 std::string suffix = "") {
    profiling::TaskInfo taskInfo = {};
    auto execKind = taskOp.getExecutorKind();
    switch (execKind) {
    case VPU::ExecutorKind::DMA_NN:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::DMA;
        break;
    case VPU::ExecutorKind::DPU:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::DPU;
        taskInfo.clusterId = getClusterId(mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp()));
        taskInfo.isSubtask = !suffix.empty();  // variant
        break;
    case VPU::ExecutorKind::SHAVE_ACT:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::SW;
        taskInfo.clusterId = getClusterId(mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp()));
        break;

    default:
        log.warning("Not supported executor type - '{0}", execKind);
        taskInfo.exec_type = profiling::TaskInfo::ExecType::NONE;
        break;
    }

    auto op = taskOp.getInnerTaskOp();
    VPUX_THROW_WHEN(op == nullptr, "TaskOp with no op inside - '{0}'", taskOp->getLoc());
    auto loc = op->getLoc();

    taskInfo.name = stringifyPrimaryLocation(loc) + suffix;
    taskInfo.layer_type = getLayerTypeFromLocation(loc);

    taskInfo.start_time_ns = startTimeNs;
    taskInfo.duration_ns = durationNs;

    return taskInfo;
}

/// @brief Calculate compiled activity factor for runtime NPU power estimation
/// @param totalEnergy - includes dpu and shave energy for all workloads
/// @param totalCycles - model inference cycles by DPU freq and the parallalism among different executors are considered
/// @param numTils - number of tiles in NCE, to get a perTile AF
/// @param arch - architecture kind
double getActivityFactor(double totalEnergy, size_t totalCycles, size_t numTiles, config::ArchKind arch) {
    VPUX_THROW_WHEN(totalCycles == 0 || numTiles == 0, "Divide zero value as totalCycles = {0} or numTiles = {1}",
                    totalCycles, numTiles);
    // A statistical ratio for total activity factor, to estimate final NPU activity factor considering DMA running.
    // For newer architectures than NPU40XX, the ratio is 1 (no change) as VPUNN itself can process it well.
    const auto npu_ratio = (arch <= config::ArchKind::NPU40XX ? 0.85 : 1.0);
    auto energyPerTile = totalEnergy / numTiles;
    auto afPerTile = energyPerTile / totalCycles;
    auto af = afPerTile * npu_ratio;  // reduce activity factor
    return af;
}

double convertCyclesToNanoSeconds(size_t cycles, double freqInMHz) {
    return (cycles * 1000.0) / freqInMHz;
}

void createScheduleTraceEventFile(const SmallVector<VPURT::TaskConfig, 1>& tasksCycleConfig, double freqInMHz,
                                  StringRef fileName, Logger log) {
    std::ofstream out_stream(fileName.str());
    VPUX_THROW_UNLESS(out_stream.good(), "File for schedule traces not created correctly");

    VPUX_THROW_WHEN(tasksCycleConfig.empty(), "Empty cycle config array");

    std::vector<profiling::TaskInfo> tasks;
    for (auto& taskConfig : tasksCycleConfig) {
        auto taskOp = taskConfig.taskOp;
        auto cycleBegin = taskConfig.cycleStart;
        auto cycleCost = taskConfig.cycleCost;

        VPUX_THROW_UNLESS(cycleCost > 0, "Invalid cycle setting (cycleBegin - '{0}', cycleCost - '{1}') for op - '{2}'",
                          cycleBegin, cycleCost, taskOp->getLoc());

        tasks.push_back(makeTaskInfo(taskOp, convertCyclesToNanoSeconds(cycleBegin, freqInMHz),
                                     convertCyclesToNanoSeconds(cycleCost, freqInMHz), log));

        // Represent in trace file DPU tasks per variant
        if (taskOp.getExecutorKind() == VPU::ExecutorKind::DPU) {
            VPUX_THROW_WHEN(taskConfig.subTasksCycleCost.size() != taskConfig.subTasksCycleStart.size(),
                            "Incorrect config of sub task cycle start and cost");
            for (size_t i = 0; i < taskConfig.subTasksCycleStart.size(); i++) {
                std::string suffix = "/variant_" + std::to_string(i);
                tasks.push_back(makeTaskInfo(taskOp,
                                             convertCyclesToNanoSeconds(taskConfig.subTasksCycleStart[i], freqInMHz),
                                             convertCyclesToNanoSeconds(taskConfig.subTasksCycleCost[i], freqInMHz),
                                             log, std::move(suffix)));
            }
        }
    }

    auto layers = getLayerInfo(tasks);
    printProfilingAsTraceEvent(tasks, layers, /*dpuFreq=*/{freqInMHz, profiling::FreqStatus::SIM}, out_stream, log);
}

class InferenceExecutionAnalysisPass final :
        public VPURT::impl::InferenceExecutionAnalysisBase<InferenceExecutionAnalysisPass> {
public:
    explicit InferenceExecutionAnalysisPass(const std::string& compileSchedTraceFileName, bool dumpToJson,
                                            bool enableActivityFactor, Logger log)
            : _compileSchedTraceFileName(compileSchedTraceFileName),
              _dumpToJson(dumpToJson),
              _enableActivityFactor(enableActivityFactor) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::string _compileSchedTraceFileName;
    bool _dumpToJson;
    bool _enableActivityFactor;
};

void InferenceExecutionAnalysisPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(moduleOp);
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(moduleOp);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);
    CycleCostInfo cycleCostInfo(std::move(costModel), funcOp);

    VPURT::InferenceExecutionSimulator infSim(_log, funcOp, cycleCostInfo);

    _log.trace("Start inference schedule simulation and update cycles");

    infSim.runSim();

    // Get total dpu cycles for model inference time
    auto totalCycles = infSim.getInferenceLatencyInCycles();
    _log.trace("Inference total cycles - {0}", totalCycles);

    // All cycles returned from VPUNN cost model are provided with respect to DPU clock
    // Get frequency information to allow translation to time units
    double freqInMHz = 0;
    auto tileOp = config::getTileExecutor(moduleOp);
    if (tileOp != nullptr) {
        freqInMHz = tileOp.getProcessorFrequency().getValueAsDouble();
    }
    VPUX_THROW_WHEN(freqInMHz == 0, "Frequency was not configured");

    auto estimatedLatencyInUs = convertCyclesToNanoSeconds(totalCycles, freqInMHz) / 1000;
    _log.info("Estimated inference latency: {0} us in dpu frequency {1} MHz", estimatedLatencyInUs, freqInMHz);
    if (auto tasksCountWithInvalidCost = cycleCostInfo.getNumberOfTasksWithInvalidCost()) {
        _log.warning("There are {0} tasks with invalid cost, estimation might not be valid", tasksCountWithInvalidCost);

        _log.debug("Invalid cost for:");
        for (auto& layerWithInvalidCost : cycleCostInfo.getLayersWithInvalidCost()) {
            _log.nest().debug("{0}", layerWithInvalidCost);
        }
    }

    // Calculate AF & inference time and store them into attributes
    if (_enableActivityFactor) {
        // Get dpu total energy for all dpuTasks
        auto dpuTotalEnergy = infSim.getDPUTotalEnergy();
        _log.trace("[Energy] dpu total energy - {0}", dpuTotalEnergy);

        // Get shave total energy for sw ops
        auto shaveTotalEnergy = infSim.getSHAVETotalEnergy();
        _log.trace("[Energy] shave total energy - {0}", shaveTotalEnergy);

        // Set compiled Activity Factor (AF) attribute to TileResource op for NPU Energy feature
        if (tileOp != nullptr) {
            auto numTiles = tileOp.getCount();
            auto activityFactor = getActivityFactor(dpuTotalEnergy + shaveTotalEnergy, totalCycles, numTiles, arch);
            auto activityFactorAttr = mlir::FloatAttr::get(mlir::Float64Type::get(funcOp.getContext()), activityFactor);
            tileOp.setActivityFactorAttr(activityFactorAttr);
            _log.info("[Energy] compiled Activity Factor - {0}", activityFactor);
        }

        // Set inferenceTiming attribute to NetworkInfoOp for NPU Energy feature
        auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
        VPUX_THROW_UNLESS(netInfoOps.size() == 1,
                          "Can't have more than one 'net::NetworkInfoOp' Operation in Module, got '{0}'",
                          netInfoOps.size());
        auto netInfo = netInfoOps.front();
        netInfo.setInferenceTiming(std::optional<int64_t>(totalCycles));
        _log.info("[Energy] inferenceTiming {0} cycles (DPU cycle unit)", totalCycles);
    }

    if (_dumpToJson) {
        VPUX_THROW_WHEN(_compileSchedTraceFileName.empty(), "Empty compile time schedule trace file");
        auto tasksCycleConfig = infSim.getTaskCycleConfig();
        createScheduleTraceEventFile(tasksCycleConfig, freqInMHz, _compileSchedTraceFileName, _log);
    }
}

}  // namespace

//
// createInferenceExecutionAnalysisPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createInferenceExecutionAnalysisPass(std::string compileSchedTraceFileName,
                                                                              bool dumpToJson,
                                                                              bool enableActivityFactor, Logger log) {
    return std::make_unique<InferenceExecutionAnalysisPass>(compileSchedTraceFileName, dumpToJson, enableActivityFactor,
                                                            log);
}
