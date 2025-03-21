//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/device.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/metadata.hpp"

#include "schema/profiling_generated.h"

#include <mlir/IR/Visitors.h>

using namespace vpux;

namespace {

VPUIP::TargetDevice mapTargetDevice(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::NPU37XX:
        return VPUIP::TargetDevice::TargetDevice_VPUX37XX;
    case VPU::ArchKind::NPU40XX:
        return VPUIP::TargetDevice::TargetDevice_VPUX40XX;
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
    VPUX_THROW_WHEN(loc.isa<mlir::UnknownLoc>(), "Trying to serialize UnknownLoc");
    const auto str = stringifyPrimaryLocation(loc);
    VPUX_THROW_WHEN(str.empty(), "Trying to serialize empty operation name");
    return str;
}

struct ProfilingConfiguration {
    ProfilingConfiguration(IE::CNNNetworkOp netOp) {
        using profiling::ExecutorType;

        auto profilingOutputsInfo = netOp.getProfilingOutputsInfo();
        VPUX_THROW_WHEN(profilingOutputsInfo.size() != 1,
                        "Unexpected number of profiling outputs (expected 1, got {0})", profilingOutputsInfo.size());

        IE::DataInfoOp profilingOutputInfo = *profilingOutputsInfo.front().getOps<IE::DataInfoOp>().begin();
        totalOutputSize = getProfilingOutputSize(profilingOutputInfo);

        const auto& sectionsRange =
                profilingOutputInfo.getSections().front().front().getOps<VPUIP::ProfilingSectionOp>();

        sections = SmallVector<VPUIP::ProfilingSectionOp>(sectionsRange.begin(), sectionsRange.end());

        isDmaProfEnabled = hasSectionOfType<ExecutorType::DMA_HW, ExecutorType::DMA_SW>();
        isDpuProfEnabled = hasSectionOfType<ExecutorType::DPU>();
        isSwProfEnabled = hasSectionOfType<ExecutorType::ACTSHAVE>();
        isM2iProfEnabled = hasSectionOfType<ExecutorType::M2I>();
    }

    SmallVector<VPUIP::ProfilingSectionOp> sections;
    size_t totalOutputSize;  // size of output in bytes

    bool isDmaProfEnabled;
    bool isDpuProfEnabled;
    bool isSwProfEnabled;
    bool isM2iProfEnabled;

private:
    template <profiling::ExecutorType... execTypes>
    bool hasSectionOfType() {
        return llvm::any_of(sections, [](VPUIP::ProfilingSectionOp section) {
            VPUX_THROW_WHEN(section.getSize() == 0, "Section of type {0} is empty", section.getSectionType());

            const auto sectionType = static_cast<profiling::ExecutorType>(section.getSectionType());
            return ((sectionType == execTypes) || ...);
        });
    }

    // returns total profiling output size in bytes
    size_t getProfilingOutputSize(IE::DataInfoOp profilingOutputInfo) {
        const auto profilingType = profilingOutputInfo.getUserType().cast<vpux::NDTypeInterface>();
        const auto shape = profilingType.getShape();

        VPUX_THROW_WHEN(shape.size() != 1, "Invalid profiling output shape '{0}'. Must be 1D tensor");
        VPUX_THROW_UNLESS(profilingType.getElementType().isInteger(CHAR_BIT * sizeof(uint32_t)),
                          "Profiling tensor type must be ui32, but got '{0}'", profilingType.getElementType());

        const size_t totalSize = shape[DimsOrder::C.dimAt(0)] * sizeof(uint32_t);
        return totalSize;
    }
};

using BarrierMap = DenseMap<mlir::Value, uint32_t>;
using TaskBarriers = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

template <class BarrierOp>
BarrierMap getBarriers(mlir::func::FuncOp funcOp) {
    BarrierMap barriersIds;
    for (BarrierOp barrierOp : funcOp.getOps<BarrierOp>()) {
        auto val = barrierOp.getBarrier();
        VPUX_THROW_UNLESS(barriersIds.count(val) == 0, "Value {0} was already serialized", val);
        barriersIds.insert({val, checked_cast<uint32_t>(barriersIds.size())});
    }
    return barriersIds;
}

flatbuffers::Offset<ProfilingFB::DPUTask> createDPUTaskMeta(flatbuffers::FlatBufferBuilder& builder,
                                                            VPUIP::DpuProfilingMetadataAttr metaAttr,
                                                            const std::string& name,
                                                            const std::vector<uint32_t>& waitBarriers,
                                                            const std::vector<uint32_t>& updateBarriers,
                                                            const std::vector<uint32_t>& workloadIds) {
    VPUX_THROW_WHEN(metaAttr.getNumVariants() == nullptr, "Missed numVariants information for DpuMetaSerialization");
    VPUX_THROW_WHEN(metaAttr.getClusterId() == nullptr, "Missed clusterId information for DpuMetaSerialization");

    const auto bufferId = metaAttr.getBufferId().getInt();
    const auto clusterId = metaAttr.getClusterId().getInt();
    const auto taskId = metaAttr.getTaskId().getInt();
    const auto numVariants = metaAttr.getNumVariants().getInt();
    const auto maxVariants = metaAttr.getMaxVariants().getInt();

    VPUX_THROW_WHEN(workloadIds.size() != 0 && workloadIds.size() != static_cast<size_t>(numVariants),
                    "Expected {0} workloads, but got {1}", numVariants, workloadIds.size());

    const auto nameOffset = builder.CreateString(name);
    const auto waitBarriersOffset = builder.CreateVector(waitBarriers);
    const auto updateBarriersOffset = builder.CreateVector(updateBarriers);
    const auto workloadIdsOffset = builder.CreateVector(workloadIds);

    return ProfilingFB::CreateDPUTask(builder, nameOffset, bufferId, clusterId, taskId, numVariants, maxVariants,
                                      waitBarriersOffset, updateBarriersOffset, workloadIdsOffset);
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

struct RtDialectProvider {
    static inline bool IS_DMA_HWP_SUPPORTED = false;

    template <class TargetOp>
    static auto extractOp(mlir::func::FuncOp funcOp) {
        SmallVector<TargetOp> ops;
        funcOp->walk([&](VPURT::TaskOp taskOp) {
            if (auto innerOp = mlir::dyn_cast<TargetOp>(taskOp.getInnerTaskOp())) {
                ops.push_back(innerOp);
            }
        });
        return ops;
    }

    static TaskBarriers getOpBarriers(const BarrierMap& virtBarriers, mlir::Operation* op) {
        auto parentOp = op->getParentOfType<VPURT::TaskOp>();
        VPUX_THROW_WHEN(op == nullptr, "Parent must be VPURT::TaskOp");
        return getOpBarriersImpl(virtBarriers, parentOp.getWaitBarriers(), parentOp.getUpdateBarriers());
    }

    template <class TargetOp>
    static auto extractComputeSwOp(mlir::func::FuncOp funcOp) {
        SmallVector<TargetOp> ops;
        funcOp->walk([&](VPURT::TaskOp taskOp) {
            if (auto innerOp = mlir::dyn_cast<TargetOp>(taskOp.getInnerTaskOp())) {
                if (!isCacheHandlingOp(taskOp.getInnerTaskOp())) {
                    ops.push_back(innerOp);
                }
            }
        });
        return ops;
    }

    template <class DPUInvariant, class DPUVariant>
    static std::vector<uint32_t> getWorkloadIds(DPUInvariant dpuOp) {
        std::vector<uint32_t> workloadIds;
        for (DPUVariant variant : dpuOp.getVariants().template getOps<DPUVariant>()) {
            if (variant.getWorkloadId().has_value()) {
                workloadIds.push_back(variant.getWorkloadId().value());
            }
        }
        return workloadIds;
    }

    static unsigned short getDmaHwpId(VPUIP::DMATypeOpInterface dmaOp) {
        if (auto hwpIdAttr = dmaOp.getDmaHwpIdAttr()) {
            return static_cast<unsigned short>(hwpIdAttr.getSInt());
        }
        return 0;
    }

    static std::optional<VPUIP::SwProfilingMetadataAttr> getSwProfilingMetadata(VPUIP::SwKernelOp op) {
        return op.getProfilingMetadata();
    }
};

using RtDialectProvider37XX = RtDialectProvider;
struct RtDialectProvider40XX : public RtDialectProvider {
    static inline bool IS_DMA_HWP_SUPPORTED = true;
};

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

template <class DialectProvider, class DmaType, class Iterable>
FbVector<ProfilingFB::DMATask> getDmaTasksOffset(const ProfilingConfiguration& profilingCfg,
                                                 flatbuffers::FlatBufferBuilder& builder, const Iterable& dmaTasks,
                                                 const BarrierMap& barriers) {
    if (!profilingCfg.isDmaProfEnabled) {
        return {};
    }
    std::vector<flatbuffers::Offset<ProfilingFB::DMATask>> dmaOffsets;
    for (const auto& dmaTask : dmaTasks) {
        // TableGen generate interface methods without const specifier, so can't be called from const DmaType&.
        // In the same moment, coverity force to use const auto&
        DmaType& mutDmaTask = const_cast<DmaType&>(dmaTask);

        const auto maybeMetadata = mutDmaTask.getProfilingMetadata();
        if (!maybeMetadata.has_value()) {
            continue;
        }

        const auto metadata = maybeMetadata.value();
        const auto numNullAttrs = static_cast<int>(metadata.getDataIndex() == nullptr) +
                                  static_cast<int>(metadata.getProfBegin() == nullptr);
        VPUX_THROW_UNLESS(numNullAttrs == 1, "Invalid DMA metadata '{0}'. Only one attribute must be set. {1}",
                          metadata, numNullAttrs);

        const unsigned short hwpId = DialectProvider::getDmaHwpId(dmaTask);
        // Do not serialize task with hwpId = 0 when HWP enabled
        if (DialectProvider::IS_DMA_HWP_SUPPORTED && hwpId == 0) {
            continue;
        }

        const auto name = stringifyPrimaryLocationChecked(dmaTask->getLoc());
        const bool isProfBegin = metadata.getProfBegin() != nullptr;

        unsigned dataIndex = 0;
        if (!isProfBegin) {
            dataIndex = metadata.getDataIndex().getInt();
        }

        const auto opBarriers = DialectProvider::getOpBarriers(barriers, dmaTask);
        const auto nameOffset = builder.CreateString(name);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);
        const auto taskOffset = ProfilingFB::CreateDMATask(builder, nameOffset, waitBarriersOffset,
                                                           updateBarriersOffset, hwpId, dataIndex, isProfBegin);
        dmaOffsets.push_back(taskOffset);
    }
    return builder.CreateVector(dmaOffsets);
}

template <class DialectProvider, class M2iType, class Iterable>
FbVector<ProfilingFB::M2ITask> getM2iTasksOffset(const ProfilingConfiguration& profilingCfg,
                                                 flatbuffers::FlatBufferBuilder& builder, const Iterable& m2iTasks,
                                                 const BarrierMap& barriers) {
    if (!profilingCfg.isM2iProfEnabled) {
        return {};
    }
    std::vector<flatbuffers::Offset<ProfilingFB::M2ITask>> m2iOffsets;
    for (const auto& m2iTask : m2iTasks) {
        const auto name = stringifyPrimaryLocationChecked(m2iTask->getLoc());
        const auto opBarriers = DialectProvider::getOpBarriers(barriers, m2iTask);
        const auto nameOffset = builder.CreateString(name);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);

        auto profMeta = const_cast<M2iType&>(m2iTask).getProfilingMetadata();
        VPUX_THROW_UNLESS(profMeta.has_value(), "Empty profiling metadata at '{0}'", m2iTask);

        const auto taskOffset =
                ProfilingFB::CreateM2ITask(builder, nameOffset, waitBarriersOffset, updateBarriersOffset);
        m2iOffsets.push_back(taskOffset);
    }
    return builder.CreateVector(m2iOffsets);
}

template <class DialectProvider, class DPUInvariantType, class DPUVariantType, class Iterable>
FbVector<ProfilingFB::DPUTask> getDpuTasksOffset(const ProfilingConfiguration& profilingCfg,
                                                 flatbuffers::FlatBufferBuilder& builder, const Iterable& dpuTasks,
                                                 const BarrierMap& barriers) {
    if (!profilingCfg.isDpuProfEnabled) {
        return {};
    }
    std::vector<flatbuffers::Offset<ProfilingFB::DPUTask>> dpuOffsets;
    for (const auto& dpuInvariant : dpuTasks) {
        auto name = stringifyPrimaryLocationChecked(dpuInvariant->getLoc());

        const auto opBarriers = DialectProvider::getOpBarriers(barriers, dpuInvariant);
        std::vector<uint32_t> workloadIds =
                DialectProvider::template getWorkloadIds<DPUInvariantType, DPUVariantType>(dpuInvariant);

        // TableGen generate interface methods without const specifier, so can't be called from const DpuType&.
        // In the same moment, coverity force to use const auto&
        auto profMeta = const_cast<DPUInvariantType&>(dpuInvariant).getProfilingMetadata();
        VPUX_THROW_UNLESS(profMeta.has_value(), "Empty profiling metadata at '{0}'", dpuInvariant);

        const auto taskOffset =
                createDPUTaskMeta(builder, profMeta.value(), name, opBarriers.first, opBarriers.second, workloadIds);
        dpuOffsets.push_back(taskOffset);
    }
    size_t dpuTaskCount = std::distance(dpuTasks.begin(), dpuTasks.end());
    VPUX_THROW_WHEN(dpuOffsets.size() != dpuTaskCount,
                    "Number of DPU tasks in profiling metadata ({0}) doesn't match the number of DPU invariants ({1})",
                    dpuOffsets.size(), dpuTaskCount);
    return builder.CreateVector(dpuOffsets);
}

template <class DialectProvider, class SwType, class Iterable>
FbVector<ProfilingFB::SWTask> getSwTasksOffset(const ProfilingConfiguration& profilingCfg,
                                               flatbuffers::FlatBufferBuilder& builder, const Iterable& swTasks,
                                               const BarrierMap& barriers) {
    if (!profilingCfg.isSwProfEnabled) {
        return {};
    }
    std::vector<flatbuffers::Offset<ProfilingFB::SWTask>> swTaskOffsets;
    for (const auto& swTask : swTasks) {
        auto name = stringifyPrimaryLocationChecked(swTask->getLoc());
        std::string swTaskType;
        // ActShave store kernel as attribute, so for all task same operation used.
        const auto taskType = swTask->getName().getStringRef().str();
        if (SwType::getOperationName().str() != taskType) {
            swTaskType = cleanSwTaskType(taskType);
        }

        const auto opBarriers = DialectProvider::getOpBarriers(barriers, swTask);

        const auto nameOffset = builder.CreateString(name);
        const auto typeOffset = builder.CreateString(swTaskType);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);

        auto maybeMetadata = DialectProvider::getSwProfilingMetadata(swTask);
        VPUX_THROW_UNLESS(maybeMetadata.has_value(), "Missed metadata for '{0}'", swTask->getLoc());

        const auto metadata = maybeMetadata.value();
        const auto bufferId = metadata.getBufferId().getInt();
        const auto bufferOffset = metadata.getBufferOffset().getInt();
        const auto clusterSize = metadata.getClusterSize().getInt();
        const auto dataIndex = metadata.getDataIndex().getInt();
        const auto tileId = metadata.getTileId().getInt();
        const auto clusterId = metadata.getClusterId().getInt();

        const auto taskOffset =
                ProfilingFB::CreateSWTask(builder, nameOffset, waitBarriersOffset, updateBarriersOffset, typeOffset,
                                          bufferId, bufferOffset, clusterSize, dataIndex, tileId, clusterId);

        swTaskOffsets.push_back(taskOffset);
    }
    size_t swTaskCount = std::distance(swTasks.begin(), swTasks.end());
    VPUX_THROW_WHEN(swTaskOffsets.size() != swTaskCount,
                    "Number of SW tasks in profiling metadata ({0}) doesn't match the number of SW tasks ({1})",
                    swTaskOffsets.size(), swTaskCount);

    return builder.CreateVector(swTaskOffsets);
}

flatbuffers::Offset<ProfilingFB::ProfilingBuffer> createProfilingBufferOffset(ProfilingConfiguration& profilingCfg,
                                                                              flatbuffers::FlatBufferBuilder& builder) {
    std::vector<flatbuffers::Offset<ProfilingFB::ProfilingSection>> profilingSectionsOffsets;
    for (auto& section : profilingCfg.sections) {
        const auto secType = section.getSectionType();
        const auto offset = section.getOffset();
        const auto size = section.getSize();

        const auto secTypeLabel =
                builder.CreateString(profiling::convertExecTypeToName(static_cast<profiling::ExecutorType>(secType)));

        const auto sectionOffset = ProfilingFB::CreateProfilingSection(builder, secType, offset, size, secTypeLabel);
        profilingSectionsOffsets.push_back(sectionOffset);
    }

    const auto sectionsOffset = builder.CreateVector(profilingSectionsOffsets);
    const auto sectionTotalSizeBytes = profilingCfg.totalOutputSize;
    return ProfilingFB::CreateProfilingBuffer(builder, sectionsOffset, sectionTotalSizeBytes);
}

flatbuffers::Offset<ProfilingFB::Platform> createPlatformOffset(VPU::ArchKind arch,
                                                                flatbuffers::FlatBufferBuilder& builder) {
    auto targetDevice = mapTargetDevice(arch);
    return ProfilingFB::CreatePlatform(builder, (int8_t)targetDevice);
}

template <typename DialectProvider, class BarrierOp, class DmaType, class DpuInvariantType, class DpuVariantType,
          class SwType, class M2iType>
flatbuffers::DetachedBuffer buildProfilingMetaGeneric(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger log) {
    log.trace("building Profiling Metadata");

    flatbuffers::FlatBufferBuilder builder;

    const auto barriers = getBarriers<BarrierOp>(funcOp);
    ProfilingConfiguration profilingCfg(netOp);
    const auto arch = VPU::getArch(funcOp);

    auto dmaOffset = getDmaTasksOffset<DialectProvider, DmaType>(
            profilingCfg, builder, DialectProvider::template extractOp<DmaType>(funcOp), barriers);
    auto dpuOffset = getDpuTasksOffset<DialectProvider, DpuInvariantType, DpuVariantType>(
            profilingCfg, builder, DialectProvider::template extractOp<DpuInvariantType>(funcOp), barriers);
    auto swTaskOffset = getSwTasksOffset<DialectProvider, SwType>(
            profilingCfg, builder, DialectProvider::template extractComputeSwOp<SwType>(funcOp), barriers);
    auto m2iOffset = getM2iTasksOffset<DialectProvider, M2iType>(
            profilingCfg, builder, DialectProvider::template extractOp<M2iType>(funcOp), barriers);
    auto profilingBufferOffset = createProfilingBufferOffset(profilingCfg, builder);
    auto platformOffset = createPlatformOffset(arch, builder);

    auto metadataOffset =
            ProfilingFB::CreateProfilingMeta(builder, vpux::profiling::PROFILING_METADATA_VERSION_MAJOR,
                                             vpux::profiling::PROFILING_METADATA_VERSION_MINOR, platformOffset,
                                             profilingBufferOffset, dmaOffset, dpuOffset, swTaskOffset, m2iOffset);
    builder.Finish(metadataOffset);

    return builder.Release();
}

template <typename VPURTDialectProvider>
flatbuffers::DetachedBuffer buildProfilingMetaVPURTGeneral(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp,
                                                           Logger log) {
    return buildProfilingMetaGeneric<VPURTDialectProvider, VPURT::ConfigureBarrierOp, VPUIP::DMATypeOpInterface,
                                     VPUIP::NCEClusterTaskOp, VPUIP::DPUTaskOp, VPUIP::SwKernelOp, VPUIP::M2ITaskOp>(
            netOp, funcOp, log);
}

flatbuffers::DetachedBuffer buildProfilingMeta(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger log) {
    const auto arch = VPU::getArch(funcOp);
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return ::buildProfilingMetaVPURTGeneral<RtDialectProvider37XX>(netOp, funcOp, log);
    default:
        return ::buildProfilingMetaVPURTGeneral<RtDialectProvider40XX>(netOp, funcOp, log);
    }
}

};  // namespace

std::vector<uint8_t> vpux::buildProfilingMetadataBuffer(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger log) {
    flatbuffers::DetachedBuffer buffer = buildProfilingMeta(netOp, funcOp, log);
    return vpux::profiling::constructProfilingSectionWithHeader(std::move(buffer));
}
