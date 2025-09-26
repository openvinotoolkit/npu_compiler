//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/reports/api.hpp"

#include "vpux/utils/profiling/taskinfo.hpp"

#include <iomanip>
#include <ostream>
#include <string>

void vpux::profiling::printProfilingAsText(const std::vector<TaskInfo>& tasks, const std::vector<LayerInfo>& layers,
                                           std::ostream& output) {
    uint64_t last_time_ns = 0;
    std::ios::fmtflags origFlags(output.flags());
    output << std::left << std::setprecision(2) << std::fixed;
    for (auto& task : tasks) {
        std::string exec_type_str;
        std::string taskName(task.name);

        switch (task.exec_type) {
        case TaskInfo::ExecType::DMA:
            exec_type_str = "DMA";
            output << "Task(" << exec_type_str << "): " << std::setw(60) << taskName << "\tTime(us): " << std::setw(8)
                   << (float)task.duration_ns / 1000 << "\tStart(us): " << std::setw(8)
                   << (float)task.start_time_ns / 1000 << std::endl;
            break;
        case TaskInfo::ExecType::DPU:
            exec_type_str = "DPU";
            output << "Task(" << exec_type_str << "): " << std::setw(60) << taskName << "\tTime(us): " << std::setw(8)
                   << (float)task.duration_ns / 1000 << "\tStart(us): " << std::setw(8)
                   << (float)task.start_time_ns / 1000 << std::endl;
            break;
        case TaskInfo::ExecType::SW:
            exec_type_str = "SW";
            output << "Task(" << exec_type_str << "): " << std::setw(60) << taskName << "\tTime(us): " << std::setw(8)
                   << (float)task.duration_ns / 1000 << "\tCycles:" << task.active_cycles << "(" << task.stall_cycles
                   << ")"
                   << "\tStart(us): " << std::setw(8) << (float)task.start_time_ns / 1000 << std::endl;
            break;
        case TaskInfo::ExecType::M2I:
            exec_type_str = "M2I";
            output << "Task(" << exec_type_str << "): " << std::setw(60) << taskName << "\tTime(us): " << std::setw(8)
                   << (float)task.duration_ns / 1000 << "\tStart(us): " << std::setw(8)
                   << (float)task.start_time_ns / 1000 << std::endl;
            break;
        default:
            break;
        }

        uint64_t task_end_time_ns = task.start_time_ns + task.duration_ns;
        if (last_time_ns < task_end_time_ns) {
            last_time_ns = task_end_time_ns;
        }
    }

    uint64_t total_time = 0;
    for (auto& layer : layers) {
        output << "Layer: " << std::setw(40) << layer.name << " Type: " << std::setw(20) << layer.layer_type
               << " DPU: " << std::setw(8) << (float)layer.dpu_ns / 1000 << " SW: " << std::setw(8)
               << (float)layer.sw_ns / 1000 << " DMA: " << std::setw(8) << (float)layer.dma_ns / 1000
               << "\tStart: " << (float)layer.start_time_ns / 1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    output << "Total time: " << (float)total_time / 1000 << "us, Real: " << (float)last_time_ns / 1000 << "us"
           << std::endl;
    output.flags(origFlags);
}
