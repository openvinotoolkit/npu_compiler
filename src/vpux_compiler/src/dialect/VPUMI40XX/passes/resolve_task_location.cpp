//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_RESOLVETASKLOCATION
#define GEN_PASS_DEF_RESOLVETASKLOCATION
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX
namespace vpux {

namespace {

class ResolveTaskLocationPass final : public VPUMI40XX::impl::ResolveTaskLocationBase<ResolveTaskLocationPass> {
public:
    ResolveTaskLocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        // Needs to be in the order that RT expects
        _supportedTaskTypes = {VPURegMapped::TaskType::DPUInvariant,
                               VPURegMapped::TaskType::DPUVariant,
                               VPURegMapped::TaskType::ActKernelRange,
                               VPURegMapped::TaskType::ActKernelInvocation,
                               VPURegMapped::TaskType::DMA,
                               VPURegMapped::TaskType::M2I};
    }

private:
    struct MaxTileInfo {
        std::unordered_map<VPURegMapped::TaskType, size_t> maxTilePerTaskType;
        size_t maxUsedTile;
    };

    template <VPURegMapped::TaskType type>
    std::array<VPURegMapped::TaskBufferSize, VPUMI40XX::MetadataBufferSize<type>::listCount>
    getOptimalTaskCountsPerList(llvm::ArrayRef<size_t> defaultTaskCounts,
                                VPUMI40XX::MappedInferenceOp mappedInferenceOp, const MaxTileInfo& maxTileInfo) {
        VPUX_THROW_UNLESS(mappedInferenceOp != nullptr,
                          "Mapped Inference Op Interface member needs to be initialized first.");
        std::array<std::vector<size_t>, VPUMI40XX::MetadataBufferSize<type>::listCount> sizeCountPerListAndTile;

        for (auto [listIdx, sizeCountPerTile] : sizeCountPerListAndTile | indexed) {
            sizeCountPerTile.resize(maxTileInfo.maxUsedTile);
            for (size_t tileIdx = 0; tileIdx < maxTileInfo.maxTilePerTaskType.at(type); tileIdx++) {
                sizeCountPerTile[tileIdx] =
                        std::min(mappedInferenceOp.getTaskCount(type, tileIdx, listIdx), defaultTaskCounts[listIdx]);
            }
        }

        std::array<VPURegMapped::TaskBufferSize, VPUMI40XX::MetadataBufferSize<type>::listCount> maxTaskCountsPerList;

        switch (type) {
        case VPURegMapped::TaskType::DMA: {
            constexpr auto DDR_INDEX = static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR);
            constexpr auto CMX_INDEX = static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN);

            auto& ddr_counts = maxTaskCountsPerList[DDR_INDEX];
            auto& cmx_counts = maxTaskCountsPerList[CMX_INDEX];

            std::tie(ddr_counts.staticSize, cmx_counts.staticSize) =
                    VPUMI40XX::compute_dma_split(std::accumulate(sizeCountPerListAndTile[DDR_INDEX].begin(),
                                                                 sizeCountPerListAndTile[DDR_INDEX].end(), 0),
                                                 std::accumulate(sizeCountPerListAndTile[CMX_INDEX].begin(),
                                                                 sizeCountPerListAndTile[CMX_INDEX].end(), 0));

            auto getDMAListDynamicSize = [&](size_t listIndex) {
                size_t staticSize = 0;
                if (listIndex == DDR_INDEX) {
                    staticSize = ddr_counts.staticSize;
                } else if (listIndex == CMX_INDEX) {
                    staticSize = cmx_counts.staticSize;
                } else {
                    VPUX_THROW("Invalid index for DMA list. Only 0 for DDR and 1 for CMX is accepted");
                }
                return std::min(*(std::max_element(sizeCountPerListAndTile[listIndex].begin(),
                                                   sizeCountPerListAndTile[listIndex].end())),
                                staticSize);
            };

            ddr_counts.dynamicSize = getDMAListDynamicSize(DDR_INDEX);
            cmx_counts.dynamicSize = getDMAListDynamicSize(CMX_INDEX);

            break;
        }
        default: {
            for (size_t listIdx = 0; listIdx < VPUMI40XX::MetadataBufferSize<type>::listCount; listIdx++) {
                maxTaskCountsPerList[listIdx].dynamicSize = *(std::max_element(sizeCountPerListAndTile[listIdx].begin(),
                                                                               sizeCountPerListAndTile[listIdx].end()));
                maxTaskCountsPerList[listIdx].staticSize = defaultTaskCounts[listIdx];
            }
            break;
        }
        }

        return maxTaskCountsPerList;
    }

    template <VPURegMapped::TaskType type>
    void populate(MetadataBuffersContainer& metadataBuffers, VPUMI40XX::MappedInferenceOp mappedInferenceOp,
                  MaxTileInfo& maxTileInfo) {
        auto optimalTaskCountsPerList = getOptimalTaskCountsPerList<type>(
                VPUMI40XX::MetadataBufferSize<type>::defaultTaskCount, mappedInferenceOp, maxTileInfo);

        for (auto [tileIdx, sizesPerTile] : metadataBuffers.sizes | indexed) {
            auto& sizesPerList = sizesPerTile[type];
            sizesPerList.resize(VPUMI40XX::MetadataBufferSize<type>::listCount);
            for (auto [listIdx, size] : sizesPerList | indexed) {
                auto& taskBufferSize = size;
                taskBufferSize =
                        tileIdx < maxTileInfo.maxTilePerTaskType[type]
                                ? optimalTaskCountsPerList[listIdx]
                                : VPURegMapped::TaskBufferSize(0, optimalTaskCountsPerList[listIdx].staticSize);
            }
        }
    }

    void safeRunOnFunc() final;
};

namespace {
const std::unordered_map<VPURegMapped::TaskType, size_t> taskBinarySize40XX = {
        {VPURegMapped::TaskType::DPUInvariant, sizeof(npu40xx::nn_public::VpuDPUInvariant)},
        {VPURegMapped::TaskType::DPUVariant, sizeof(npu40xx::nn_public::VpuDPUVariant)},
        {VPURegMapped::TaskType::ActKernelRange, sizeof(npu40xx::nn_public::VpuActKernelRange)},
        {VPURegMapped::TaskType::ActKernelInvocation, sizeof(npu40xx::nn_public::VpuActKernelInvocation)},
        {VPURegMapped::TaskType::DMA, sizeof(npu40xx::nn_public::VpuDMATask)},
        {VPURegMapped::TaskType::M2I, sizeof(npu40xx::nn_public::VpuMediaTask)}};
}  // namespace

// TODO: E#121934 Add method for VPURegMapped TaskType to be able to directly return its binary size in an
// arch-specific way
size_t getTaskBinarySize(VPURegMapped::TaskType taskType, [[maybe_unused]] VPU::ArchKind arch) {
    return taskBinarySize40XX.at(taskType);
}

void ResolveTaskLocationPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    const auto arch = VPU::getArch(funcOp);

    MetadataBuffersContainer metadataBuffers;
    MaxTileInfo maxTileInfo;
    maxTileInfo.maxUsedTile = 0;

    auto mappedInferenceRange = funcOp.getOps<VPUMI40XX::MappedInferenceOp>();
    VPUX_THROW_WHEN(std::distance(mappedInferenceRange.begin(), mappedInferenceRange.end()) != 1,
                    "There should be only one MappedInferenceOp");
    auto mappedInferenceOp = *(mappedInferenceRange.begin());

    // resize the container to the number of max used tiles - no need to create task layout for tiles that are not used
    for (auto& taskType : _supportedTaskTypes) {
        maxTileInfo.maxTilePerTaskType[taskType] = mappedInferenceOp.getMaxTaskTile(taskType);
        maxTileInfo.maxUsedTile = std::max(maxTileInfo.maxUsedTile, maxTileInfo.maxTilePerTaskType[taskType]);
    }
    metadataBuffers.sizes.resize(maxTileInfo.maxUsedTile);

    populate<VPURegMapped::TaskType::DPUInvariant>(metadataBuffers, mappedInferenceOp, maxTileInfo);
    populate<VPURegMapped::TaskType::DPUVariant>(metadataBuffers, mappedInferenceOp, maxTileInfo);
    populate<VPURegMapped::TaskType::ActKernelRange>(metadataBuffers, mappedInferenceOp, maxTileInfo);
    populate<VPURegMapped::TaskType::ActKernelInvocation>(metadataBuffers, mappedInferenceOp, maxTileInfo);
    populate<VPURegMapped::TaskType::DMA>(metadataBuffers, mappedInferenceOp, maxTileInfo);
    populate<VPURegMapped::TaskType::M2I>(metadataBuffers, mappedInferenceOp, maxTileInfo);

    auto builder = mlir::OpBuilder::atBlockBegin(&funcOp.getBody().front());

    // Construct map for TaskBufferLayout
    // DictionaryAttr has be constructed respecting the structure presented at TaskBufferLayoutOp tblgen definition
    llvm::SmallVector<mlir::NamedAttribute> taskList;
    size_t taskOffset = 0, intraTileOffset = 0;
    auto u64Type = vpux::getUInt64Type(builder.getContext());

    for (auto& taskType : _supportedTaskTypes) {
        auto taskTypeStrAttr = mlir::StringAttr::get(builder.getContext(), VPURegMapped::stringifyTaskType(taskType));

        llvm::SmallVector<mlir::Attribute> sizeForTileAndList;
        for (auto& tile : metadataBuffers.sizes) {
            llvm::SmallVector<mlir::Attribute> sizesPerList;
            intraTileOffset = 0;
            for (auto& list : tile[taskType]) {
                auto taskGroup = VPURegMapped::TaskGroupAttr::get(
                        builder.getContext(), mlir::IntegerAttr::get(u64Type, list.dynamicSize),
                        mlir::IntegerAttr::get(u64Type, list.staticSize),
                        mlir::IntegerAttr::get(u64Type, taskOffset + intraTileOffset),
                        mlir::IntegerAttr::get(u64Type, getTaskBinarySize(taskType, arch)));
                sizesPerList.push_back(taskGroup);

                intraTileOffset += list.staticSize * getTaskBinarySize(taskType, arch);
            }
            auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), sizesPerList);
            sizeForTileAndList.push_back(arrayAttr);
        }
        taskOffset += intraTileOffset;
        auto sizesForTaskTypeAttr = mlir::ArrayAttr::get(builder.getContext(), sizeForTileAndList);

        auto namedTaskAttr = mlir::NamedAttribute(taskTypeStrAttr, sizesForTaskTypeAttr);
        taskList.push_back(namedTaskAttr);
    }

    auto dictAttr = mlir::DictionaryAttr::get(builder.getContext(), taskList);
    auto taskLayoutOp = builder.create<VPURegMapped::TaskBufferLayoutOp>(builder.getUnknownLoc(), dictAttr);

    createTaskLocationBuffers(taskLayoutOp, metadataBuffers);
}

}  // namespace

std::unique_ptr<mlir::Pass> VPUMI40XX::createResolveTaskLocationPass(Logger log) {
    return std::make_unique<ResolveTaskLocationPass>(log);
}

}  // namespace vpux
