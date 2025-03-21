#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace vpux {
namespace VPURegMapped {

size_t calcMinBitsRequirement(uint64_t value) {
    if (value == 0) {
        return 1;
    }
    if (value == std::numeric_limits<uint64_t>::max()) {
        return sizeof(uint64_t) * CHAR_BIT;
    }
    return checked_cast<size_t>(std::ceil(log2(value + 1)));
}

void updateRegMappedInitializationValues(std::map<std::string, std::map<std::string, RegFieldValue>>& values,
                                         const std::map<std::string, std::map<std::string, RegFieldValue>>& newValues) {
    for (auto newRegisterIter = newValues.begin(); newRegisterIter != newValues.end(); ++newRegisterIter) {
        auto correspondingRegisterIter = values.find(newRegisterIter->first);
        VPUX_THROW_UNLESS(correspondingRegisterIter != values.end(),
                          "updateRegMappedInitializationValues: Register with name {0} not found in provided values",
                          newRegisterIter->first);

        for (auto newFieldIter = newRegisterIter->second.begin(); newFieldIter != newRegisterIter->second.end();
             ++newFieldIter) {
            auto correspondingFieldIter = correspondingRegisterIter->second.find(newFieldIter->first);
            VPUX_THROW_UNLESS(correspondingFieldIter != correspondingRegisterIter->second.end(),
                              "updateRegMappedInitializationValues: Field with name {0} not found in provided values",
                              newFieldIter->first);

            // update field value
            correspondingFieldIter->second = newFieldIter->second;
        }
    }
}

std::optional<TaskBufferLayoutOp> getTaskBufferLayoutOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op->hasTrait<mlir::OpTrait::OneRegion>(),
                      "VPURegMapped::TaskBufferLayoutOp should only exist under OneRegion-type ops");

    auto taskBufferOpsRange = op->getRegion(0).getOps<TaskBufferLayoutOp>();
    if (taskBufferOpsRange.empty()) {
        return std::nullopt;
    }

    VPUX_THROW_WHEN(std::distance(taskBufferOpsRange.begin(), taskBufferOpsRange.end()) > 1,
                    "Only one VPURegMapped::TaskBufferLayoutOp should exist");
    return *(taskBufferOpsRange.begin());
}

template <typename T>
struct TaskTypeMapper;

template <>
struct TaskTypeMapper<VPURegMapped::TaskType> {
    static VPU::TaskType map(VPURegMapped::TaskType regMappedType) {
        return static_cast<VPU::TaskType>(
                static_cast<std::underlying_type<VPURegMapped::TaskType>::type>(regMappedType));
    }
};

uint32_t getDefaultTaskListCount(VPURegMapped::TaskType taskType, VPU::ArchKind archKind) {
    VPU::TaskType vpuTaskType = TaskTypeMapper<VPURegMapped::TaskType>::map(taskType);
    return VPU::getDefaultTaskListCount(vpuTaskType, archKind);
}

}  // namespace VPURegMapped
}  // namespace vpux
