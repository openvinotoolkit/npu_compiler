//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_RESOLVETASKLOCATION
#define GEN_PASS_DEF_RESOLVETASKLOCATION
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX
namespace vpux {

namespace {

class ResolveTaskLocationPass final : public VPUMI40XX::impl::ResolveTaskLocationBase<ResolveTaskLocationPass> {
public:
    explicit ResolveTaskLocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    struct TaskBufferSize {
        TaskBufferSize(size_t dynamicSize, size_t staticSize): dynamicSize(dynamicSize), staticSize(staticSize) {};
        TaskBufferSize() = default;

        size_t dynamicSize = 0;
        size_t staticSize = 0;
    };
    template <typename Content>
    using MetadataBuffersContainerType =
            llvm::SmallVector<llvm::DenseMap<VPURegMapped::TaskType, llvm::SmallVector<Content>>>;
    struct MetadataBuffersContainer {
        MetadataBuffersContainerType<llvm::SmallVector<mlir::Value>> data;
        MetadataBuffersContainerType<TaskBufferSize> sizes;
    };

    llvm::SmallVector<VPURegMapped::TaskType> _supportedTaskTypes = {VPURegMapped::TaskType::DPUInvariant,
                                                                     VPURegMapped::TaskType::DPUVariant,
                                                                     VPURegMapped::TaskType::ActKernelRange,
                                                                     VPURegMapped::TaskType::ActKernelInvocation,
                                                                     VPURegMapped::TaskType::DMA,
                                                                     VPURegMapped::TaskType::M2I};
    struct MaxTileInfo {
        std::unordered_map<VPURegMapped::TaskType, size_t> maxTilePerTaskType;
        size_t maxUsedTile;
    };

    template <VPURegMapped::TaskType type>
    std::array<TaskBufferSize, VPUMI40XX::MetadataBufferSize<type>::listCount> getOptimalTaskCountsPerList(
            llvm::ArrayRef<size_t> defaultTaskCounts, VPUMI40XX::MappedInferenceOp mappedInferenceOp,
            const MaxTileInfo& maxTileInfo) {
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

        std::array<TaskBufferSize, VPUMI40XX::MetadataBufferSize<type>::listCount> maxTaskCountsPerList;

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
                taskBufferSize = tileIdx < maxTileInfo.maxTilePerTaskType[type]
                                         ? optimalTaskCountsPerList[listIdx]
                                         : TaskBufferSize(0, optimalTaskCountsPerList[listIdx].staticSize);
            }
        }
    }

    void createTaskLocationBuffers(VPURegMapped::TaskBufferLayoutOp taskLayoutOp,
                                   MetadataBuffersContainer& metadataBuffers);

    void safeRunOnFunc() final;
};

void ResolveTaskLocationPass::createTaskLocationBuffers(VPURegMapped::TaskBufferLayoutOp taskLayoutOp,
                                                        MetadataBuffersContainer& metadataBuffers) {
    auto function = getOperation();
    auto builder = mlir::OpBuilder::atBlockBegin(&function.getBody().front());
    auto context = function.getContext();

    auto populateTaskBuffers = [&](size_t tile, VPURegMapped::TaskType type, const auto& sizesPerTaskType) {
        // order of DeclareTaskBuffer is important as it must be aligned with firmware expectations
        // tile0: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
        // tile1: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
        // ...
        const auto sizesPerList = sizesPerTaskType.lookup(type);
        auto& metadataBuffersPerTaskType = metadataBuffers.data[tile][type];
        metadataBuffersPerTaskType.resize(sizesPerList.size());
        for (const auto& entryPerList : llvm::enumerate(sizesPerList)) {
            const auto list = entryPerList.index();
            const auto sizePerList =
                    entryPerList.value().dynamicSize;  // can be modified from "dynamicSize" to "staticSize" if
                                                       // generating all task buffers is ever needed

            for (auto i : irange(sizePerList)) {
                auto offsetAttr = mlir::IntegerAttr::get(vpux::getUInt64Type(context),
                                                         taskLayoutOp.getTaskBufferOffset(type, tile, list, i));
                auto declareTaskBufferOp = builder.create<VPUMI40XX::DeclareTaskBufferOp>(
                        function.getLoc(),
                        vpux::VPURegMapped::IndexType::get(context, checked_cast<uint32_t>(tile),
                                                           checked_cast<uint32_t>(list), checked_cast<uint32_t>(i)),
                        type, offsetAttr);
                VPUX_THROW_UNLESS(list < metadataBuffersPerTaskType.size(),
                                  "Incorrect size of metadata buffers ({0}) for list index {1}",
                                  metadataBuffersPerTaskType.size(), list);
                metadataBuffersPerTaskType[list].push_back(declareTaskBufferOp);
            }
        }
    };

    metadataBuffers.data.resize(metadataBuffers.sizes.size());
    VPUX_THROW_WHEN(_supportedTaskTypes.empty(), "The _supportedTaskTypes was not populated by the arch-specificpass");
    for (const auto& entryPerTile : llvm::enumerate(metadataBuffers.sizes)) {
        const auto tile = entryPerTile.index();
        const auto& sizesPerTaskType = entryPerTile.value();
        for (auto& taskType : _supportedTaskTypes) {
            populateTaskBuffers(tile, taskType, sizesPerTaskType);
        }
    }

    for (auto task : function.getOps<VPURegMapped::TaskOpInterface>()) {
        const auto type = task.getTaskType();
        const auto index = task.getIndexType();
        const auto& taskBuffers = metadataBuffers.data[index.getTileIdx()][type][index.getListIdx()];

        task.setTaskLocation(taskBuffers[index.getValue() % taskBuffers.size()]);
    }
}

void ResolveTaskLocationPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    VPUX_THROW_WHEN(config::isFifoPerShaveEngineEnabled(funcOp),
                    "Dedicated Shave FIFOs for non-Wlm are not supported.");
    const auto arch = config::getArch(funcOp);

    MetadataBuffersContainer metadataBuffers;
    MaxTileInfo maxTileInfo;
    maxTileInfo.maxUsedTile = 0;

    auto mappedInferenceOp = VPUMI40XX::getMPI(funcOp);

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
                        mlir::IntegerAttr::get(u64Type, VPUMI40XX::getTaskBinarySize(taskType, arch)));
                sizesPerList.push_back(taskGroup);

                intraTileOffset += list.staticSize * VPUMI40XX::getTaskBinarySize(taskType, arch);
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
