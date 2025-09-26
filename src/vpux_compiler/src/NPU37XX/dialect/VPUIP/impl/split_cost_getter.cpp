//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/split_cost_getter.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"

#include <vpu_cost_model.h>

using namespace vpux;

int64_t VPUIP::arch37xx::computeSplitCost(const WorkloadSplit& split, const WorkloadCostParams& params,
                                          VPUNN::VPUCostModel& costModel, LogCb logCb) {
    std::vector<int64_t> workloadCost;
    workloadCost.reserve(split.size());

    std::string vpunnInputCheckInfo;

    // Correct invalid input channels for depthwise workload before passing to VPUNN
    // split to produce more small and valid workloads
    const SmallVector<int64_t> supportedChannelsDW = {64, 32, 16};
    auto correctDepthwiseWorkloadChannel = [=](const WorkloadTile& wl) {
        auto wlChannel = std::get<0>(wl).shape[Dims4D::Act::C];
        SmallVector<int64_t> validWorkloadChannels;
        std::vector<WorkloadTile> newWorkloads;
        auto newWl = wl;
        validWorkloadChannels = splitWorkloadChannel(wlChannel, supportedChannelsDW);
        VPUX_THROW_WHEN(validWorkloadChannels.size() == 0,
                        "splitWorkloadChannel failed please check wlChannel - {0}, supportedChannelsDW - {1}",
                        wlChannel, supportedChannelsDW);
        for (auto validChannel : validWorkloadChannels) {
            std::get<0>(newWl).shape[Dims4D::Act::C] = validChannel;
            newWorkloads.push_back(newWl);
        }
        return newWorkloads;
    };

    std::vector<WorkloadTile> correctWls;
    for (const auto& wl : split) {
        correctWls.push_back(wl);
        // Split workload channel to satisfy HW limit for depthwise ops before passing to VPUNN
        if (params.nceTaskType == NCETaskType::DWCONV || params.nceTaskType == NCETaskType::MAXPOOL ||
            params.nceTaskType == NCETaskType::AVEPOOL) {
            auto wlChannel = std::get<0>(wl).shape[Dims4D::Act::C];
            if (std::find(supportedChannelsDW.begin(), supportedChannelsDW.end(), wlChannel) ==
                supportedChannelsDW.end()) {
                correctWls = correctDepthwiseWorkloadChannel(wl);
            }
        }

        for (const auto& correctWl : correctWls) {
            const auto vpunnWorkload = VPU::getDPUWorkload(params, correctWl);
            auto wlCost =
                    VPU::checkAndReturnCost(costModel.DPU(vpunnWorkload, vpunnInputCheckInfo), Logger::global(), true);
            if (wlCost >= VPU::INVALID_COST_BASE) {
                logCb(formatv("[VPUNN LOG] INVALID_COST is caught. Please check possible VPUNN debug info: {0}",
                              vpunnInputCheckInfo));
                VPU::printVPUNNWorkloadConfig(vpunnWorkload, logCb);
            }
            workloadCost.push_back(static_cast<int64_t>(wlCost));
        }

        correctWls.clear();
    }

    return VPUNN::dpu_schedule(checked_cast<unsigned int>(params.numDPU), workloadCost);
}
