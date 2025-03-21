
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include <algorithm>
#include <vector>

namespace vpux::profiling {

/**
 * Converts task profiling info to OV layer info
 */
std::vector<LayerInfo> getLayerInfo(const std::vector<TaskInfo>& taskInfo) {
    std::vector<LayerInfo> layerInfo;
    for (const auto& task : taskInfo) {
        LayerInfo* layer;
        if (task.isSubtask) {
            continue;  // Skip variant info
        }

        std::string layerName = getLayerName(task.name);
        auto result = std::find_if(begin(layerInfo), end(layerInfo), [&](const LayerInfo& item) {
            return layerName == item.name;
        });
        if (result == end(layerInfo)) {
            layer = &layerInfo.emplace_back();
            layer->status = LayerInfo::layer_status_t::EXECUTED;
            layer->start_time_ns = task.start_time_ns;
            layer->duration_ns = 0;

            const auto nameLen = layerName.copy(layer->name, sizeof(layer->name) - 1);
            layer->name[nameLen] = 0;

            const std::string layerTypeStr(task.layer_type);
            const auto typeLen = layerTypeStr.copy(layer->layer_type, sizeof(layer->layer_type) - 1);
            layer->layer_type[typeLen] = 0;
        } else {
            layer = &(*result);
        }
        if (task.start_time_ns < layer->start_time_ns) {
            layer->duration_ns += layer->start_time_ns - task.start_time_ns;
            layer->start_time_ns = task.start_time_ns;
        }
        auto duration = (int64_t)task.start_time_ns + task.duration_ns - layer->start_time_ns;
        if (duration > layer->duration_ns) {
            layer->duration_ns = duration;
        }

        if (task.exec_type == TaskInfo::ExecType::DPU) {
            layer->dpu_ns += task.duration_ns;
        } else if (task.exec_type == TaskInfo::ExecType::SW) {
            layer->sw_ns += task.duration_ns;
        } else {
            layer->dma_ns += task.duration_ns;
        }
    }

    std::sort(layerInfo.begin(), layerInfo.end(), profilingTaskStartTimeComparator<LayerInfo>);
    return layerInfo;
}

}  // namespace vpux::profiling
