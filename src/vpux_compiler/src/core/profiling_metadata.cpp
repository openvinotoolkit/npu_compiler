//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/core/profiling_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/device.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/metadata.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include <flatbuffers/flatbuffers.h>
#include "schema/profiling_generated.h"

#include <mlir/IR/Visitors.h>

using namespace vpux;

namespace {

VPUIP::TargetDevice mapTargetDevice(config::ArchKind kind) {
    switch (kind) {
    case config::ArchKind::NPU37XX:
        return VPUIP::TargetDevice::TargetDevice_VPUX37XX;
    case config::ArchKind::NPU40XX:
        return VPUIP::TargetDevice::TargetDevice_VPUX40XX;
    case config::ArchKind::NPU50XX:
        return VPUIP::TargetDevice::TargetDevice_VPUX50XX;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

bool isCacheHandlingOp(mlir::Operation* op) {
    if (auto swKernel = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        return VPUIP::isCacheHandlingOp(swKernel);
    } else if (auto kernelInvocation = mlir::dyn_cast<VPUMI37XX::ActKernelInvocationOp>(op)) {
        auto kernelRange = kernelInvocation.getRangeIndex().getDefiningOp<VPUMI37XX::ActKernelRangeOp>();
        return VPUMI37XX::isSwKernelCacheOp(kernelRange);
    } else {
        return false;
    }
}

std::string stringifyPrimaryLocationChecked(mlir::Location loc) {
    VPUX_THROW_WHEN(mlir::isa<mlir::UnknownLoc>(loc), "Trying to serialize UnknownLoc");
    const auto str = stringifyPrimaryLocation(loc);
    VPUX_THROW_WHEN(str.empty(), "Trying to serialize empty operation name");
    return str;
}

net::DataInfoOp getProfilingOutputInfoOp(net::NetworkInfoOp netInfo) {
    auto profilingOutputsInfo = netInfo.getProfilingOutputsInfo();
    VPUX_THROW_WHEN(profilingOutputsInfo.size() != 1, "Unexpected number of profiling outputs");
    return *profilingOutputsInfo.front().getOps<net::DataInfoOp>().begin();
}

size_t getProfilingOutputSize(net::DataInfoOp profilingOutputInfo) {
    const auto profilingType = mlir::cast<NDTypeInterface>(profilingOutputInfo.getUserType());
    const auto shape = profilingType.getShape();
    VPUX_THROW_WHEN(shape.size() != 1, "Invalid profiling output shape");
    VPUX_THROW_UNLESS(profilingType.getElementType().isInteger(CHAR_BIT * sizeof(uint32_t)),
                      "Invalid profiling output type");

    return shape[DimsOrder::C.dimAt(0)] * sizeof(uint32_t);
}

SmallVector<VPUIP::ProfilingSectionOp> getProfilingSections(net::DataInfoOp profilingOutputInfo) {
    SmallVector<VPUIP::ProfilingSectionOp> profilingSections;
    const auto sectionsRange = profilingOutputInfo.getSections().front().front().getOps<VPUIP::ProfilingSectionOp>();
    return SmallVector<VPUIP::ProfilingSectionOp>(sectionsRange.begin(), sectionsRange.end());
}

profiling::ExecutorType getSectionType(VPUIP::ProfilingSectionOp sectionOp) {
    return static_cast<profiling::ExecutorType>(sectionOp.getSectionType());
}

using BarrierMap = DenseMap<mlir::Value, uint32_t>;
using TaskBarriers = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

BarrierMap getBarriers(mlir::func::FuncOp funcOp) {
    BarrierMap barriersIds;
    for (auto barrierOp : funcOp.getOps<VPURT::ConfigureBarrierOp>()) {
        auto val = barrierOp.getBarrier();
        VPUX_THROW_UNLESS(barriersIds.count(val) == 0, "Value {0} was already serialized", val);
        barriersIds.insert({val, checked_cast<uint32_t>(barriersIds.size())});
    }
    return barriersIds;
}

flatbuffers::Offset<ProfilingFB::DPUTask> createDPUTaskMeta(
        flatbuffers::FlatBufferBuilder& builder, VPUIP::DpuProfilingMetadataAttr metaAttr, const std::string& name,
        const std::vector<uint32_t>& waitBarriers, const std::vector<uint32_t>& updateBarriers,
        const std::vector<uint32_t>& workloadIds, const profiling::TensorInfo& tensorInfo,
        const VariantInfoArray& variantArray) {
    VPUX_THROW_WHEN(metaAttr.getNumVariants() == nullptr, "Missed numVariants information for DpuMetaSerialization");
    VPUX_THROW_WHEN(metaAttr.getClusterId() == nullptr, "Missed clusterId information for DpuMetaSerialization");

    const auto bufferId = metaAttr.getBufferId().getInt();
    const auto clusterId = metaAttr.getClusterId().getInt();
    const auto taskId = metaAttr.getTaskId().getInt();
    const auto numVariants = metaAttr.getNumVariants().getInt();
    const auto maxVariants = metaAttr.getMaxVariants().getInt();

    VPUX_THROW_WHEN(workloadIds.size() != 0 && workloadIds.size() != static_cast<size_t>(numVariants),
                    "Expected {0} workloads, but got {1}", numVariants, workloadIds.size());

    auto fbTensorInfo = ProfilingFB::CreateTensorInfo(builder, &tensorInfo);
    auto variants = to_std_vector(variantArray | transformed([&builder](const profiling::DPUVariantInfo& variant) {
                                      return ProfilingFB::CreateDPUVariantInfo(builder, &variant);
                                  }));
    return ProfilingFB::CreateDPUTaskDirect(builder, name.c_str(), bufferId, clusterId, taskId, numVariants,
                                            maxVariants, &waitBarriers, &updateBarriers, &workloadIds, fbTensorInfo,
                                            &variants);
}

TaskBarriers getOpBarriersImpl(const BarrierMap& virtBarriers, const mlir::ValueRange waitBarriers,
                               const mlir::ValueRange updateBarriers) {
    const auto extractBarriersIDs = [&virtBarriers](const mlir::ValueRange barriers) -> std::vector<uint32_t> {
        std::vector<uint32_t> ids;
        ids.reserve(barriers.size());
        for (const auto& bar : barriers) {
            const auto it = virtBarriers.find(bar);
            VPUX_THROW_UNLESS(it != virtBarriers.end(), "Value {0} wasn't serialized yet", bar);
            ids.push_back(it->second);
        }
        return ids;
    };

    std::vector<uint32_t> waitIds = extractBarriersIDs(waitBarriers);
    std::vector<uint32_t> updateIds = extractBarriersIDs(updateBarriers);

    return std::make_pair(waitIds, updateIds);
}

template <class TargetOp>
auto extractOp(mlir::func::FuncOp funcOp) {
    SmallVector<TargetOp> ops;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (auto innerOp = mlir::dyn_cast<TargetOp>(taskOp.getInnerTaskOp())) {
            ops.push_back(innerOp);
        }
    });
    return ops;
}

// Specialization to skip cache handling SW ops
template <>
auto extractOp<VPUIP::SwKernelOp>(mlir::func::FuncOp funcOp) {
    SmallVector<VPUIP::SwKernelOp> ops;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (auto innerOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp())) {
            if (!isCacheHandlingOp(taskOp.getInnerTaskOp())) {
                ops.push_back(innerOp);
            }
        }
    });
    return ops;
}

TaskBarriers getOpBarriers(const BarrierMap& virtBarriers, mlir::Operation* op) {
    auto parentOp = op->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(op == nullptr, "Parent must be VPURT::TaskOp");
    return getOpBarriersImpl(virtBarriers, parentOp.getWaitBarriers(), parentOp.getUpdateBarriers());
}

std::vector<uint32_t> getWorkloadIds(VPUIP::NCEClusterTaskOp dpuOp) {
    std::vector<uint32_t> workloadIds;
    for (auto variant : dpuOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        if (variant.getIsDummy()) {
            continue;
        }
        if (variant.getWorkloadId().has_value()) {
            workloadIds.push_back(variant.getWorkloadId().value());
        }
    }
    return workloadIds;
}

unsigned short getDmaHwpId(VPUIP::DMATypeOpInterface dmaOp) {
    if (auto hwpIdAttr = dmaOp.getDmaHwpIdAttr()) {
        return static_cast<unsigned short>(hwpIdAttr.getSInt());
    }
    return 0;
}

std::optional<ProfilingFB::MemoryKind> getMemKind(mlir::Value value) {
    auto memKind = mlir::cast<NDTypeInterface>(value.getType()).getMemoryKind();
    switch (memKind) {
    case VPU::MemoryKind::DDR:
        return ProfilingFB::MemoryKind::DDR;
    case VPU::MemoryKind::CMX_NN:
        return ProfilingFB::MemoryKind::CMX;
    default:
        break;
    }
    return std::nullopt;
}

ProfilingFB::DMAChannelType getDmaChannelType(VPUIP::DMATypeOpInterface dmaOp) {
    // \ref dmaOp.getChannelType
    auto srcMemKind = mlir::cast<NDTypeInterface>(dmaOp->getOperand(0).getType()).getMemoryKind();

    switch (srcMemKind) {
    case VPU::MemoryKind::DDR:
        return ProfilingFB::DMAChannelType::DDR;
    case VPU::MemoryKind::CMX_NN:
    case VPU::MemoryKind::Register:
        return ProfilingFB::DMAChannelType::CMX;
    default:
        VPUX_THROW("Unknown DMA channel type");
    }
}

template <typename TaskType>
using FbVector = flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TaskType>>>;

std::string cleanSwTaskType(const std::string& origType) {
    const std::vector<std::pair<std::string, std::string>> REPLACE_PAIRS = {{"VPUIP.", ""}};
    return std::accumulate(REPLACE_PAIRS.cbegin(), REPLACE_PAIRS.cend(), origType,
                           [](std::string a, const auto& replacement) {
                               const auto pos = a.find(replacement.first);
                               if (pos == std::string::npos) {
                                   return a;
                               }
                               return a.replace(pos, replacement.first.size(), replacement.second);
                           });
}

mlir::Operation* getOriginalDmaOp(mlir::Operation* endTimestampDmaOp) {
    if (auto parentTaskOp = endTimestampDmaOp->getParentOp()) {
        if (auto origTaskOp = parentTaskOp->getPrevNode()) {
            if (auto& region = origTaskOp->getRegion(0); !region.empty() && !region.front().empty()) {
                return &region.front().front();
            }
        }
    }
    return nullptr;
}

FbVector<ProfilingFB::DMATask> getDmaTasksOffset(flatbuffers::FlatBufferBuilder& builder,
                                                 const SmallVector<VPUIP::DMATypeOpInterface>& dmaTasks,
                                                 const BarrierMap& barriers, config::ArchKind arch) {
    std::vector<flatbuffers::Offset<ProfilingFB::DMATask>> dmaOffsets;

    const bool isDmaHwpSupported = (arch == config::ArchKind::NPU37XX) ? false : true;

    for (auto dmaTask : dmaTasks) {
        const auto metadata = dmaTask.getProfilingMetadata().value_or(nullptr);
        if (!metadata) {
            continue;
        }
        VPUX_THROW_UNLESS(!!metadata.getDataIndex() ^ !!metadata.getProfBegin(), "Invalid DMA metadata '{0}'");

        const unsigned short hwpId = getDmaHwpId(dmaTask);
        // Do not serialize task with hwpId = 0 when HWP enabled
        if (isDmaHwpSupported && hwpId == 0) {
            continue;
        }

        const auto name = stringifyPrimaryLocationChecked(dmaTask->getLoc());
        const bool isProfBeginDma = metadata.getProfBegin() != nullptr;
        const bool isProfEndDma = !(isDmaHwpSupported || isProfBeginDma);
        const unsigned dataIndex = isProfBeginDma ? 0 : metadata.getDataIndex().getInt();
        const auto [waitBarriers, updateBarriers] = getOpBarriers(barriers, dmaTask);
        auto origDmaTask =
                isProfEndDma
                        ? mlir::dyn_cast_or_null<VPUIP::DMATypeOpInterface>(getOriginalDmaOp(dmaTask.getOperation()))
                        : dmaTask;
        VPUX_THROW_WHEN(!origDmaTask, "Invalid DMA task {0}", dmaTask);

        std::optional<uint16_t> portId = std::nullopt;
        std::optional<ProfilingFB::DMAChannelType> channelType = std::nullopt;
        std::optional<ProfilingFB::MemoryKind> sourceMemoryKind = std::nullopt;
        std::optional<ProfilingFB::MemoryKind> destinationMemoryKind = std::nullopt;
        if (!isProfBeginDma) {
            portId = origDmaTask.getPortVal();
            channelType = getDmaChannelType(origDmaTask);
            sourceMemoryKind = getMemKind(origDmaTask.getInput());
            destinationMemoryKind = getMemKind(origDmaTask.getOutput());
        }
        auto [tensorShapeInfo, tensorStrideInfo] = extractTensorInfoFromOp(dmaTask);
        auto fbShapeTypeInfo = CreateTensorInfo(builder, &tensorShapeInfo);
        auto fbStrideTypeInfo = CreateTensorInfo(builder, &tensorStrideInfo);
        auto dynamicStridesInput = dmaTask->hasAttr(stridedInputAttrName);
        auto dynamicStridesOutput = dmaTask->hasAttr(stridedOutputAttrName);

        unsigned short gatherIndices = 0;
        if (auto gatherDma = mlir::dyn_cast<VPUIP::GatherDMAOp>(dmaTask.getOperation())) {
            gatherIndices = mlir::cast<NDTypeInterface>(gatherDma.getIndices().getType()).getShape().front();
        }

        dmaOffsets.push_back(ProfilingFB::CreateDMATaskDirect(
                builder, name.c_str(), &waitBarriers, &updateBarriers, hwpId, dataIndex, isProfBeginDma, portId,
                channelType, sourceMemoryKind, destinationMemoryKind, fbShapeTypeInfo, fbStrideTypeInfo, gatherIndices,
                dynamicStridesInput, dynamicStridesOutput));
    }
    return builder.CreateVector(dmaOffsets);
}

FbVector<ProfilingFB::M2ITask> getM2iTasksOffset(flatbuffers::FlatBufferBuilder& builder,
                                                 const SmallVector<VPUIP::M2ITaskOp>& m2iTasks,
                                                 const BarrierMap& barriers) {
    std::vector<flatbuffers::Offset<ProfilingFB::M2ITask>> m2iOffsets;
    for (auto m2iTask : m2iTasks) {
        const auto name = stringifyPrimaryLocationChecked(m2iTask->getLoc());
        const auto [waitBarriers, updateBarriers] = getOpBarriers(barriers, m2iTask);

        auto profMeta = m2iTask.getProfilingMetadata();
        VPUX_THROW_UNLESS(profMeta.has_value(), "Empty profiling metadata at '{0}'", m2iTask);

        m2iOffsets.push_back(ProfilingFB::CreateM2ITaskDirect(builder, name.c_str(), &waitBarriers, &updateBarriers));
    }
    return builder.CreateVector(m2iOffsets);
}

FbVector<ProfilingFB::DPUTask> getDpuTasksOffset(flatbuffers::FlatBufferBuilder& builder,
                                                 const SmallVector<VPUIP::NCEClusterTaskOp>& dpuTasks,
                                                 const BarrierMap& barriers) {
    // Build a map from split_id -> ordered NCEClusterTaskOps sharing that ID.
    // Only the first op in each group carries profiling metadata; the rest are split chunks.
    DenseMap<int64_t, SmallVector<VPUIP::NCEClusterTaskOp>> splitGroups;
    for (auto dpuTask : dpuTasks) {
        if (auto splitIdAttr = dpuTask.getSplitIdAttr()) {
            splitGroups[splitIdAttr.getInt()].push_back(dpuTask);
        }
    }

    std::vector<flatbuffers::Offset<ProfilingFB::DPUTask>> dpuOffsets;
    for (auto dpuInvariant : dpuTasks) {
        auto profMeta = dpuInvariant.getProfilingMetadata();
        if (!profMeta.has_value()) {
            // Split chunk without profiling metadata – accounted for via the first chunk's split group.
            continue;
        }

        const auto name = stringifyPrimaryLocationChecked(dpuInvariant->getLoc());
        const auto [waitBarriers, updateBarriers] = getOpBarriers(barriers, dpuInvariant);

        auto tensorInfo = extractTensorInfoFromOp(dpuInvariant);
        auto variantArray = extractVariantInfoFromOp(dpuInvariant);
        auto workloadIds = getWorkloadIds(dpuInvariant);

        // Append workload IDs and variant info from the remaining chunks of the same split group.
        if (auto splitIdAttr = dpuInvariant.getSplitIdAttr()) {
            const auto& group = splitGroups[splitIdAttr.getInt()];
            // group[0] is dpuInvariant itself (the chunk carrying profiling metadata).
            for (size_t i = 1; i < group.size(); ++i) {
                auto sibling = group[i];
                VPUX_THROW_WHEN(sibling.getProfilingMetadata().has_value(),
                                "Unexpected profiling metadata on split chunk '{0}'", sibling);
                auto moreWorkloadIds = getWorkloadIds(sibling);
                workloadIds.insert(workloadIds.end(), moreWorkloadIds.begin(), moreWorkloadIds.end());
                auto moreVariantInfo = extractVariantInfoFromOp(sibling);
                variantArray.insert(variantArray.end(), moreVariantInfo.begin(), moreVariantInfo.end());
            }
        }

        dpuOffsets.push_back(createDPUTaskMeta(builder, profMeta.value(), name, waitBarriers, updateBarriers,
                                               workloadIds, tensorInfo, variantArray));
    }
    return builder.CreateVector(dpuOffsets);
}

FbVector<ProfilingFB::SWTask> getSwTasksOffset(flatbuffers::FlatBufferBuilder& builder,
                                               const SmallVector<VPUIP::SwKernelOp>& swTasks,
                                               const BarrierMap& barriers) {
    std::vector<flatbuffers::Offset<ProfilingFB::SWTask>> swTaskOffsets;
    for (auto swTask : swTasks) {
        auto maybeMetadata = swTask.getProfilingMetadata();
        if (!maybeMetadata) {
            continue;
        }
        auto name = stringifyPrimaryLocationChecked(swTask->getLoc());
        std::string swTaskType;
        // ActShave store kernel as attribute, so for all task same operation used.
        const auto taskType = swTask->getName().getStringRef().str();
        if (VPUIP::SwKernelOp::getOperationName().str() != taskType) {
            swTaskType = cleanSwTaskType(taskType);
        }
        const auto [waitBarriers, updateBarriers] = getOpBarriers(barriers, swTask);
        const auto metadata = maybeMetadata.value();
        const auto bufferId = metadata.getBufferId().getInt();
        const auto bufferOffset = metadata.getBufferOffset().getInt();
        const auto clusterSize = metadata.getClusterSize().getInt();
        const auto dataIndex = metadata.getDataIndex().getInt();
        const auto tileId = metadata.getTileId().getInt();
        const auto fifoId = swTask.getListIndex();
        const auto clusterId = metadata.getClusterId().getInt();
        auto tensorInfo = extractTensorInfoFromOp(swTask);
        auto fbTypeInfo = CreateTensorInfo(builder, &tensorInfo);

        swTaskOffsets.push_back(ProfilingFB::CreateSWTaskDirect(builder, name.c_str(), &waitBarriers, &updateBarriers,
                                                                swTaskType.c_str(), bufferId, bufferOffset, clusterSize,
                                                                dataIndex, tileId, clusterId, fbTypeInfo, fifoId));
    }
    return builder.CreateVector(swTaskOffsets);
}

flatbuffers::Offset<ProfilingFB::ProfilingSection> createProfilingSectionOffset(flatbuffers::FlatBufferBuilder& builder,
                                                                                VPUIP::ProfilingSectionOp sectionOp) {
    const auto secType = sectionOp.getSectionType();
    const auto offset = sectionOp.getOffset();
    const auto size = sectionOp.getSize();
    const auto secTypeLabel = profiling::convertExecTypeToName(static_cast<profiling::ExecutorType>(secType));
    return ProfilingFB::CreateProfilingSectionDirect(builder, secType, offset, size, secTypeLabel.c_str());
}

flatbuffers::Offset<ProfilingFB::Platform> createPlatformOffset(config::ArchKind arch,
                                                                flatbuffers::FlatBufferBuilder& builder) {
    auto targetDevice = mapTargetDevice(arch);
    return ProfilingFB::CreatePlatform(builder, (int8_t)targetDevice);
}

flatbuffers::DetachedBuffer buildProfilingMeta(net::NetworkInfoOp netInfo, mlir::func::FuncOp funcOp) {
    flatbuffers::FlatBufferBuilder builder;

    const auto arch = config::getArch(funcOp);
    const auto barriers = getBarriers(funcOp);
    auto profilingOutputInfoOp = getProfilingOutputInfoOp(netInfo);

    std::vector<flatbuffers::Offset<ProfilingFB::ProfilingSection>> profilingSectionsOffsets;
    FbVector<ProfilingFB::DMATask> dmaOffset;
    FbVector<ProfilingFB::DPUTask> dpuOffset;
    FbVector<ProfilingFB::SWTask> swTaskOffset;
    FbVector<ProfilingFB::M2ITask> m2iOffset;

    for (auto sectionOp : getProfilingSections(profilingOutputInfoOp)) {
        profilingSectionsOffsets.push_back(createProfilingSectionOffset(builder, sectionOp));
        switch (getSectionType(sectionOp)) {
        case profiling::ExecutorType::DMA_HW:
        case profiling::ExecutorType::DMA_SW:
            dmaOffset = getDmaTasksOffset(builder, extractOp<VPUIP::DMATypeOpInterface>(funcOp), barriers, arch);
            break;
        case profiling::ExecutorType::DPU:
            dpuOffset = getDpuTasksOffset(builder, extractOp<VPUIP::NCEClusterTaskOp>(funcOp), barriers);
            break;
        case profiling::ExecutorType::ACTSHAVE:
            swTaskOffset = getSwTasksOffset(builder, extractOp<VPUIP::SwKernelOp>(funcOp), barriers);
            break;
        case profiling::ExecutorType::M2I:
            m2iOffset = getM2iTasksOffset(builder, extractOp<VPUIP::M2ITaskOp>(funcOp), barriers);
            break;
        case profiling::ExecutorType::WORKPOINT:
            // no metadata
            break;
        }
    }
    const auto platformOffset = createPlatformOffset(arch, builder);
    const auto sectionTotalSizeBytes = getProfilingOutputSize(profilingOutputInfoOp);
    const auto profilingBufferOffset =
            ProfilingFB::CreateProfilingBufferDirect(builder, &profilingSectionsOffsets, sectionTotalSizeBytes);
    const auto metadataOffset = ProfilingFB::CreateProfilingMeta(
            builder, profiling::PROFILING_METADATA_VERSION_MAJOR, profiling::PROFILING_METADATA_VERSION_MINOR,
            platformOffset, profilingBufferOffset, dmaOffset, dpuOffset, swTaskOffset, m2iOffset);
    builder.Finish(metadataOffset);
    return builder.Release();
}

};  // namespace

std::vector<uint8_t> vpux::buildProfilingMetadataBuffer(net::NetworkInfoOp netInfo, mlir::func::FuncOp funcOp,
                                                        Logger log) {
    log.trace("Building profiling metadata");
    flatbuffers::DetachedBuffer buffer = buildProfilingMeta(netInfo, funcOp);
    return profiling::constructProfilingSectionWithHeader(std::move(buffer));
}
