//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

#include <numeric>

using namespace vpux;

size_t countEmptyArray(mlir::ArrayAttr array, int64_t limit) {
    const auto emptyPredicate = [](const auto& item) -> bool {
        return mlir::cast<mlir::ArrayAttr>(item).empty();
    };
    return std::count_if(array.begin(), array.begin() + limit, emptyPredicate);
}

size_t countZeroes(mlir::ArrayAttr array, int64_t limit) {
    const auto zeroPredicate = [](const int val) -> bool {
        return val == 0;
    };
    const auto arrAttr = parseIntArrayAttr<int64_t>(array);
    return std::count_if(arrAttr.begin(), arrAttr.begin() + limit, zeroPredicate);
}

mlir::ArrayAttr subArray(mlir::ArrayAttr attr, int64_t idx) {
    return mlir::cast<mlir::ArrayAttr>(attr[checked_cast<unsigned>(idx)]);
}

mlir::Value vpux::VPUMI40XX::MappedInferenceOp::getListHead(VPURegMapped::TaskType taskType, int64_t tileIdx,
                                                            int64_t listIdx) {
    auto mutableRange = getListHeadMutable(taskType, tileIdx, listIdx);
    if (mutableRange.size() > 0) {
        size_t emptyTiles, emptyLists, majorOperand, minorOperand = 0;
        switch (taskType) {
        case VPURegMapped::TaskType::DMA:
            emptyTiles = countEmptyArray(getDmaCount(), tileIdx);
            majorOperand = tileIdx - emptyTiles;

            emptyLists = countZeroes(subArray(getDmaCount(), tileIdx), listIdx);
            minorOperand = listIdx - emptyLists;

            return getDmaTasks()[majorOperand].slice(minorOperand, 1)[0];
            break;
        case VPURegMapped::TaskType::DPUInvariant:
            emptyTiles = countZeroes(getInvariantCount(), tileIdx);
            return getInvariantTasks().slice(tileIdx - emptyTiles, 1)[0];
            break;
        case VPURegMapped::TaskType::DPUVariant:
            emptyTiles = countZeroes(getVariantCount(), tileIdx);
            return getVariantTasks().slice(tileIdx - emptyTiles, 1)[0];
            break;
        case VPURegMapped::TaskType::ActKernelInvocation:
            emptyTiles = countEmptyArray(getActKernelInvocationsCount(), tileIdx);
            majorOperand = tileIdx - emptyTiles;

            emptyLists = countZeroes(subArray(getActKernelInvocationsCount(), tileIdx), listIdx);
            minorOperand = listIdx - emptyLists;

            return getActKernelInvocations()[majorOperand].slice(minorOperand, 1)[0];
            break;
        case VPURegMapped::TaskType::ActKernelRange:
            emptyTiles = countEmptyArray(getActKernelRangesCount(), tileIdx);
            majorOperand = tileIdx - emptyTiles;

            emptyLists = countZeroes(subArray(getActKernelRangesCount(), tileIdx), listIdx);
            minorOperand = listIdx - emptyLists;

            return getActKernelRanges()[majorOperand].slice(minorOperand, 1)[0];
            break;
        default:
            return nullptr;
            break;
        };
    } else {
        return nullptr;
    }
}

mlir::MutableOperandRange vpux::VPUMI40XX::MappedInferenceOp::getListHeadMutable(VPURegMapped::TaskType taskType,
                                                                                 int64_t tileIdx, int64_t listIdx) {
    auto arrayIdx = [](mlir::ArrayAttr attr, int64_t idx) -> int64_t {
        return mlir::cast<mlir::IntegerAttr>(attr[checked_cast<unsigned>(idx)]).getInt();
    };

    auto taskListSizeIsNotValid = [&arrayIdx](mlir::ArrayAttr array, int64_t tileIdx) -> bool {
        if (tileIdx >= static_cast<int64_t>(array.size()) || arrayIdx(array, tileIdx) == 0) {
            return true;
        }
        return false;
    };

    auto taskSubListSizeIsNotValid = [&arrayIdx](mlir::ArrayAttr array, int64_t tileIdx, int64_t listIdx) -> bool {
        if (tileIdx >= static_cast<int64_t>(array.size()) || subArray(array, tileIdx).size() == 0 ||
            listIdx >= static_cast<int64_t>(subArray(array, tileIdx).size()) ||
            arrayIdx(subArray(array, tileIdx), listIdx) == 0) {
            return true;
        }

        return false;
    };

    size_t emptyTiles, emptyLists, majorOperand, minorOperand = 0;
    auto emptyOperandRange = mlir::MutableOperandRange(getOperation(), 0, 0);

    switch (taskType) {
    case VPURegMapped::TaskType::DMA:
        if (taskSubListSizeIsNotValid(getDmaCount(), tileIdx, listIdx)) {
            return emptyOperandRange;
        }

        emptyTiles = countEmptyArray(getDmaCount(), tileIdx);
        majorOperand = tileIdx - emptyTiles;

        emptyLists = countZeroes(subArray(getDmaCount(), tileIdx), listIdx);
        minorOperand = listIdx - emptyLists;

        return getDmaTasksMutable()[majorOperand].slice(checked_cast<unsigned int>(minorOperand), 1);
        break;
    case VPURegMapped::TaskType::DPUInvariant:
        if (taskListSizeIsNotValid(getInvariantCount(), tileIdx)) {
            return emptyOperandRange;
        }

        emptyTiles = countZeroes(getInvariantCount(), tileIdx);
        return getInvariantTasksMutable().slice(checked_cast<unsigned int>(tileIdx - emptyTiles), 1);
        break;
    case VPURegMapped::TaskType::DPUVariant:
        if (taskListSizeIsNotValid(getVariantCount(), tileIdx)) {
            return emptyOperandRange;
        }

        emptyTiles = countZeroes(getVariantCount(), tileIdx);
        return getVariantTasksMutable().slice(checked_cast<unsigned int>(tileIdx - emptyTiles), 1);
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        if (taskSubListSizeIsNotValid(getActKernelInvocationsCount(), tileIdx, listIdx)) {
            return emptyOperandRange;
        }

        emptyTiles = countEmptyArray(getActKernelInvocationsCount(), tileIdx);
        majorOperand = tileIdx - emptyTiles;

        emptyLists = countZeroes(subArray(getActKernelInvocationsCount(), tileIdx), listIdx);
        minorOperand = listIdx - emptyLists;

        return getActKernelInvocationsMutable()[majorOperand].slice(checked_cast<unsigned int>(minorOperand), 1);
        break;
    case VPURegMapped::TaskType::ActKernelRange:
        if (taskSubListSizeIsNotValid(getActKernelRangesCount(), tileIdx, listIdx)) {
            return emptyOperandRange;
        }

        emptyTiles = countEmptyArray(getActKernelRangesCount(), tileIdx);
        majorOperand = tileIdx - emptyTiles;

        emptyLists = countZeroes(subArray(getActKernelRangesCount(), tileIdx), listIdx);
        minorOperand = listIdx - emptyLists;

        return getActKernelRangesMutable()[majorOperand].slice(checked_cast<unsigned int>(minorOperand), 1);
        break;
    default:
        return emptyOperandRange;
        break;
    };
}

//
// Dot Printer
//

DotNodeColor VPUMI40XX::MappedInferenceOp::getNodeColor() {
    return DotNodeColor::RED;
}

bool VPUMI40XX::MappedInferenceOp::printAttributes(llvm::raw_ostream&, llvm::StringRef, llvm::StringRef,
                                                   llvm::StringRef) {
    return true;
}

DOT::EdgeDir VPUMI40XX::MappedInferenceOp::getEdgeDirection(mlir::Operation*) {
    return DOT::EdgeDir::EDGE_REVERSE;
}

size_t vpux::VPUMI40XX::MappedInferenceOp::getTaskCount(vpux::VPURegMapped::TaskType taskType, size_t tileIdx,
                                                        size_t listIdx) {
    VPUX_THROW_WHEN((taskType != vpux::VPURegMapped::TaskType::DMA &&
                     taskType != vpux::VPURegMapped::TaskType::ActKernelInvocation &&
                     taskType != vpux::VPURegMapped::TaskType::ActKernelRange) &&
                            listIdx != 0,
                    "Only DMA or SW tasks can have non-zero list index. Encountered task type: {0} and list index: {1}",
                    taskType, listIdx);
    VPUX_THROW_WHEN((taskType == vpux::VPURegMapped::TaskType::DMA ||
                     taskType == vpux::VPURegMapped::TaskType::ActKernelInvocation ||
                     taskType == vpux::VPURegMapped::TaskType::ActKernelRange) &&
                            (listIdx > 1),
                    "Invalid list index value > 1. Task type: {0}, list index: {1}", taskType, listIdx);

    switch (taskType) {
    case vpux::VPURegMapped::TaskType::DMA: {
        auto dmaCounts = parseIntArrayOfArrayAttr<int64_t>(getDmaCount());
        return tileIdx > dmaCounts.size() ? 0 : dmaCounts[tileIdx][listIdx];
    }
    case vpux::VPURegMapped::TaskType::ActKernelInvocation: {
        auto kernelInvoCounts = parseIntArrayOfArrayAttr<int64_t>(getActKernelInvocationsCount());
        return tileIdx > kernelInvoCounts.size() ? 0 : kernelInvoCounts[tileIdx][listIdx];
    }
    case vpux::VPURegMapped::TaskType::ActKernelRange: {
        auto kernelRangeCounts = parseIntArrayOfArrayAttr<int64_t>(getActKernelRangesCount());
        return tileIdx > kernelRangeCounts.size() ? 0 : kernelRangeCounts[tileIdx][listIdx];
    }
    case vpux::VPURegMapped::TaskType::DPUInvariant: {
        auto dpuInvariantCounts = parseIntArrayAttr<int64_t>(getInvariantCount());
        return tileIdx > dpuInvariantCounts.size() ? 0 : dpuInvariantCounts[tileIdx];
    }
    case vpux::VPURegMapped::TaskType::DPUVariant: {
        auto dpuVariantCounts = parseIntArrayAttr<int64_t>(getVariantCount());
        return tileIdx > dpuVariantCounts.size() ? 0 : dpuVariantCounts[tileIdx];
    }
    default: {
        VPUX_THROW("Unrecognized task type");
        break;
    }
    }
}

size_t vpux::VPUMI40XX::MappedInferenceOp::getMaxTaskTile(vpux::VPURegMapped::TaskType taskType) {
    auto getMaxUsedTile = [](llvm::SmallVector<int64_t>& taskCountsVec) {
        auto lastPositiveCount = std::find_if(taskCountsVec.rbegin(), taskCountsVec.rend(), [](int64_t count) {
            return count > 0;
        });

        return std::abs(std::distance(taskCountsVec.rend(), lastPositiveCount));
    };

    auto getMaxUsedTileForShave = [](size_t listIdx,
                                     llvm::SmallVector<llvm::SmallVector<int64_t>>& taskCountsPerTilesAndListsVec) {
        auto lastPositiveCount =
                std::find_if(taskCountsPerTilesAndListsVec.rbegin(), taskCountsPerTilesAndListsVec.rend(),
                             [listIdx](const llvm::SmallVector<int64_t>& perTileVec) {
                                 return perTileVec[listIdx] > 0;
                             });

        // Note: tile indexes start with 1. Return 0 when no tasks were found.
        return std::abs(std::distance(taskCountsPerTilesAndListsVec.rend(), lastPositiveCount));
    };

    switch (taskType) {
    case vpux::VPURegMapped::TaskType::DMA: {
        auto dmaCounts = parseIntArrayOfArrayAttr<int64_t>(getDmaCount());

        llvm::SmallVector<int64_t> ddrDmaCounts{}, cmxDmaCounts{};
        llvm::for_each(dmaCounts, [&ddrDmaCounts, &cmxDmaCounts](auto& vec) {
            ddrDmaCounts.push_back(vec[static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]);
            cmxDmaCounts.push_back(vec[static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]);
        });

        auto maxTileDDR = getMaxUsedTile(ddrDmaCounts);
        auto maxTileCMX = getMaxUsedTile(cmxDmaCounts);

        return std::max(maxTileDDR, maxTileCMX);
    }
    case vpux::VPURegMapped::TaskType::ActKernelInvocation: {
        auto kernelInvoCounts = parseIntArrayOfArrayAttr<int64_t>(getActKernelInvocationsCount());
        auto maxTileForShave0 = getMaxUsedTileForShave(0, kernelInvoCounts);
        auto maxTileForShave1 = getMaxUsedTileForShave(1, kernelInvoCounts);
        VPUX_THROW_WHEN(maxTileForShave1 > 0,
                        "In non-WLM compilation no ActKernelInvocation should be assigned to SHAVE1 (found "
                        "maxTileForShave1 = {0}).",
                        maxTileForShave1);
        return maxTileForShave0;
    }
    case vpux::VPURegMapped::TaskType::ActKernelRange: {
        auto kernelRangeCounts = parseIntArrayOfArrayAttr<int64_t>(getActKernelRangesCount());
        auto maxTileForShave0 = getMaxUsedTileForShave(0, kernelRangeCounts);
        auto maxTileForShave1 = getMaxUsedTileForShave(1, kernelRangeCounts);
        VPUX_THROW_WHEN(
                maxTileForShave1 > 0,
                "In non-WLM compilation no ActKernelRange should be assigned to SHAVE1 (found maxTileForShave1 = {0}).",
                maxTileForShave1);
        return maxTileForShave0;
    }
    case vpux::VPURegMapped::TaskType::DPUInvariant: {
        auto dpuInvariantCounts = parseIntArrayAttr<int64_t>(getInvariantCount());
        return getMaxUsedTile(dpuInvariantCounts);
    }
    case vpux::VPURegMapped::TaskType::DPUVariant: {
        auto dpuVariantCounts = parseIntArrayAttr<int64_t>(getVariantCount());
        return getMaxUsedTile(dpuVariantCounts);
    }
    default: {
        VPUX_THROW("Unrecognized task type");
        break;
    }
    }
}
