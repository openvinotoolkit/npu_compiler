//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CREATENEWWEIGHTTABLESDATA
#define GEN_PASS_DEF_CREATENEWWEIGHTTABLESDATA
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

struct WeightTableKey {
    SmallVector<int32_t> data;
    SmallVector<int32_t> workloadChannels;

    bool operator==(const WeightTableKey& other) const {
        return data == other.data && workloadChannels == other.workloadChannels;
    }
};

}  // namespace

namespace llvm {
template <>
struct DenseMapInfo<WeightTableKey> {
    static inline WeightTableKey getEmptyKey() {
        return {{llvm::DenseMapInfo<int32_t>::getEmptyKey()}, {}};
    }
    static inline WeightTableKey getTombstoneKey() {
        return {{llvm::DenseMapInfo<int32_t>::getTombstoneKey()}, {}};
    }
    static unsigned getHashValue(const WeightTableKey& key) {
        auto hash = llvm::hash_combine_range(key.data.begin(), key.data.end());
        hash = llvm::hash_combine(hash,
                                  llvm::hash_combine_range(key.workloadChannels.begin(), key.workloadChannels.end()));
        return hash;
    }
    static bool isEqual(const WeightTableKey& lhs, const WeightTableKey& rhs) {
        return lhs == rhs;
    }
};
}  // namespace llvm

namespace {

inline Dim getActChannelDim(size_t rank) {
    return (rank == 4) ? Dims4D::Act::C : DimsGroups5D::Act::C;
}

inline Dim getFilterChannelDim(size_t rank) {
    return (rank == 4) ? Dims4D::Filter::OC : DimsGroups5D::Filter::OC;
}

SmallVector<SmallVector<int32_t>> extractWorkloadChannels(VPU::NCEOpInterface nceOp, bool& shouldSwitchToSegmented) {
    auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
    VPUX_THROW_UNLESS(!workloads.empty(), "No workloads were retrieved from '{0}' at '{1}'", nceOp->getName(),
                      nceOp->getLoc());

    const auto weightTableOperand = nceOp.getWeightZeroPointsOperand() ? nceOp.getWeightZeroPointsOperand()
                                                                       : nceOp.getWeightTableDataPtrOperand();
    VPUX_THROW_UNLESS(weightTableOperand != nullptr, "Can't get weight table operand for '{0}' at '{1}'",
                      nceOp->getName(), nceOp->getLoc());

    const auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(weightTableOperand.getType());

    // Default to 1 cluster if there's no distribution
    int64_t numClusters = 1;
    bool isDuplicatedMode = false;
    if (distributedType != nullptr) {
        numClusters = distributedType.getDistribution().getNumClusters().getInt();
        isDuplicatedMode = distributedType.getDistribution().getMode().getValue() == VPU::DistributionMode::DUPLICATED;
    }

    // Group workloads by cluster
    SmallVector<SmallVector<int32_t>> workloadSizes;
    workloadSizes.resize(numClusters);

    for (auto workload : workloads) {
        auto outSizes = workload.getConstOutputSizes();
        auto clusterId = workload.getClusterId().has_value() ? workload.getClusterId().value() : 0;
        VPUX_THROW_UNLESS(clusterId < numClusters, "Invalid cluster_id for workload in NCE op '{0}' at '{1}'",
                          nceOp->getName(), nceOp->getLoc());

        const auto channelDimIdx = getActChannelDim(outSizes.size()).ind();
        workloadSizes[clusterId].push_back(static_cast<int32_t>(outSizes[channelDimIdx]));
    }

    shouldSwitchToSegmented = false;

    // For DUPLICATED mode, check if workloads are identical across all clusters
    if (isDuplicatedMode && numClusters > 1) {
        // Compare all clusters against the first cluster's workload pattern
        const auto& firstClusterWorkloads = workloadSizes[0];
        bool allIdentical = true;

        for (int64_t i = 1; i < numClusters; ++i) {
            // Check if size and content match
            if (workloadSizes[i].size() != firstClusterWorkloads.size()) {
                allIdentical = false;
                break;
            }

            // Check each workload size in the pattern
            for (size_t j = 0; j < firstClusterWorkloads.size(); ++j) {
                if (workloadSizes[i][j] != firstClusterWorkloads[j]) {
                    allIdentical = false;
                    break;
                }
            }

            if (!allIdentical) {
                break;
            }
        }

        if (allIdentical) {
            // Workloads are identical across all clusters, keep DUPLICATED mode
            // Return single cluster workloads since they're all the same
            return {firstClusterWorkloads};
        } else {
            // Workloads differ across clusters, need to switch to SEGMENTED mode
            shouldSwitchToSegmented = true;
        }
    }

    return workloadSizes;
}

template <typename WeightTableType>
mlir::Value sliceZeroPoints(mlir::IRRewriter& rewriter, WeightTableType weightTableOp, Const::DeclareOp& zpConstOp,
                            VPU::SliceOp sliceOp) {
    zpConstOp = weightTableOp.getZeroPoints().template getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(zpConstOp != nullptr,
                      "ZeroPoints operand of '{0}' at '{1}' is not a const.Declare, cannot perform slicing",
                      weightTableOp->getName(), weightTableOp->getLoc());

    const auto sliceOffsets = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets()));
    const auto sliceShape = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes()));

    const auto zpOrigNDType = mlir::cast<vpux::NDTypeInterface>(zpConstOp.getOutput().getType());
    const auto zpElemType = zpOrigNDType.getElementType();

    rewriter.setInsertionPoint(weightTableOp);

    // Sub-byte ZP constants are not stored in packed form, so the ContentAttr subview transform - which requires packed
    // sub-byte storage fails. Read values as int8/uint8 and build a new constant directly from the sliced values.
    if (zpElemType.isInteger(4) || zpElemType.isInteger(2)) {
        const auto content = zpConstOp.getContent();

        // Verify that the constant is stored unpacked (1 byte per element).
        // getValues<int8_t/uint8_t>() below depends on this assumption.
        const auto numElements = content.getType().getNumElements();
        VPUX_THROW_UNLESS(content.getRawStorageBuf().size() == static_cast<size_t>(numElements),
                          "Expected 4-bit or 2-bit ZP constant at '{0}' to be stored unpacked (1 byte per element), "
                          "but got {1} bytes raw storage for {2} elements",
                          zpConstOp->getLoc(), content.getRawStorageBuf().size(), numElements);

        const auto channelDim = getFilterChannelDim(sliceShape.size());
        const auto ocOffset = sliceOffsets[channelDim];
        const auto ocSize = sliceShape[channelDim];
        const auto zpIntElemType = mlir::cast<mlir::IntegerType>(zpElemType);
        const auto bitWidth = zpIntElemType.getWidth();
        const auto zpSlicedRankedType = mlir::RankedTensorType::get(sliceShape.raw(), zpElemType);

        SmallVector<llvm::APInt> slicedValues;
        slicedValues.reserve(ocSize);

        auto createSlicedValues = [&](auto storageType) {
            using StorageT = decltype(storageType);

            const auto zpValues = to_small_vector(content.getValues<StorageT>());
            for (int64_t i = ocOffset; i < ocOffset + ocSize; ++i) {
                slicedValues.push_back(llvm::APInt(bitWidth, zpValues[i], zpIntElemType.isSigned()));
            }
        };

        zpIntElemType.isSigned() ? createSlicedValues(int8_t{}) : createSlicedValues(uint8_t{});

        return Const::createConst<llvm::APInt>(rewriter, zpConstOp.getLoc(), zpSlicedRankedType, slicedValues);
    }

    auto zpType = zpOrigNDType.changeShape(sliceShape);
    auto newContentAttr = zpConstOp.transformContentAttr().subview(sliceOffsets, sliceShape).get();
    return rewriter.create<Const::DeclareOp>(zpConstOp.getLoc(), zpType, newContentAttr).getOutput();
}

template <typename WeightTableType>
WeightTableType updateWeightTableOp(mlir::IRRewriter& rewriter, VPU::NCEOpInterface nceOp,
                                    WeightTableType weightTableOp, ArrayRef<SmallVector<int32_t>> workloadChannels,
                                    bool shouldSwitchToSegmented,
                                    llvm::DenseMap<WeightTableKey, mlir::Operation*>& createdTables,
                                    mlir::Value adjustedZeroPoints, Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();

    // Calculate output channels
    const auto channelDim = getActChannelDim(outputShape.size());
    int64_t outChannels = outputType.getShape()[channelDim];
    if (shouldSwitchToSegmented) {
        // When switching from DUPLICATED to SEGMENTED, calculate as sum of all workload channels
        int64_t totalChannels = 0;
        for (const auto& clusterWorkloads : workloadChannels) {
            for (auto channelSize : clusterWorkloads) {
                totalChannels += channelSize;
            }
        }
        outChannels = totalChannels;
    }

    // Create the weight table data with correct workload sizes and weights element type information.
    auto weightsElemType = mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getElementType();

    SmallVector<int32_t> weightTableData;
    mlir::RankedTensorType newOutputType;

    // Flatten workload channels to append as an attribute of weight table operation and use for materialization of
    // zero-point table.
    SmallVector<int32_t> flatWorkloadChannels;
    for (const auto& clusterWorkloads : workloadChannels) {
        flatWorkloadChannels.append(clusterWorkloads.begin(), clusterWorkloads.end());
    }

    // Create the table data based on the weight table type.
    if (mlir::isa<VPU::ZeroPointTableOp>(weightTableOp)) {
        weightTableData =
                VPU::materializeZeroPointTable(weightsElemType, outChannels, flatWorkloadChannels, adjustedZeroPoints);

        // Use appropriate shape based on rank
        // For 5D table, the Group dim is set to 1. We can re-use the same table for each group.
        const auto zeroPointDataShape =
                (outputShape.size() == 4)
                        ? VPU::NCESparsity::inferWeightsTableShape(static_cast<int64_t>(weightTableData.size()),
                                                                   /*newFormat=*/true)
                        : VPU::NCESparsity::infer5DWeightsTableShape(static_cast<int64_t>(weightTableData.size()),
                                                                     /*groups =*/1,
                                                                     /*newFormat=*/true);

        newOutputType = mlir::RankedTensorType::get(zeroPointDataShape.raw(), rewriter.getI8Type());

    } else if (mlir::isa<VPU::DataPointerTableOp>(weightTableOp)) {
        weightTableData = VPU::materializeDataPointerTable(
                rewriter.getContext(), workloadChannels, nceOp.getWeightsOperand(), 0, outChannels, adjustedZeroPoints);

        const auto dataPointerDataShape = VPU::NCESparsity::inferWeightsTableShape(
                static_cast<int64_t>(weightTableData.size()), /*newFormat=*/true);
        newOutputType = mlir::RankedTensorType::get(dataPointerDataShape.raw(), getSInt32Type(rewriter.getContext()));
    } else {
        VPUX_THROW("Unsupported weight table op '{0}' at '{1}'", weightTableOp->getName(), weightTableOp->getLoc());
    }

    // Check if we already created a table with this exact data and attributes
    WeightTableKey key{weightTableData, flatWorkloadChannels};
    auto it = createdTables.find(key);
    if (it != createdTables.end()) {
        log.trace("Reusing previously created weight table with matching data and attributes");
        return mlir::cast<WeightTableType>(it->second);
    }

    rewriter.setInsertionPoint(weightTableOp);
    auto newWeightTableOp =
            rewriter.create<WeightTableType>(weightTableOp->getLoc(), newOutputType, adjustedZeroPoints,
                                             getIntArrayAttr(rewriter.getContext(), flatWorkloadChannels),
                                             getIntArrayAttr(rewriter.getContext(), weightTableData));

    // Track this newly created table
    createdTables[key] = newWeightTableOp;

    log.trace("Updated weights table: {0}", newWeightTableOp);
    return newWeightTableOp;
}

template <typename WeightTableType>
void updateCopyOp(mlir::IRRewriter& rewriter, VPU::NCEOpInterface nceOp, VPU::CopyOp oldCopyOp,
                  WeightTableType newWeightTableOp, ArrayRef<SmallVector<int32_t>> workloadChannels,
                  bool shouldSwitchToSegmented, Logger log) {
    auto oldOutputType = oldCopyOp.getOutput().getType();

    auto oldDistType = mlir::dyn_cast<VPU::DistributedTensorType>(oldOutputType);
    if (oldDistType == nullptr) {
        // No distribution, just update CopyOp with corrected input/output sizes and its attributes
        auto newWeightTableType = mlir::cast<vpux::NDTypeInterface>(newWeightTableOp.getOutput().getType());
        auto oldNDType = mlir::cast<vpux::NDTypeInterface>(oldOutputType);

        auto newOutputType =
                newWeightTableType
                        .changeShapeElemType(newWeightTableType.getShape(), newWeightTableType.getElementType())
                        .changeMemSpace(oldNDType.getMemSpace())
                        .changeDimsOrder(oldNDType.getDimsOrder());

        rewriter.setInsertionPoint(oldCopyOp);
        auto newCopyOp = rewriter.replaceOpWithNewOp<VPU::CopyOp>(
                oldCopyOp, newOutputType, newWeightTableOp.getOutput(), oldCopyOp.getOutMemSpaceAttr());
        log.trace("Updated CopyOp for weight table: {0}", newCopyOp);
        return;
    }

    auto oldDistribution = oldDistType.getDistribution();

    auto oldNumTiles = oldDistribution.getNumTiles();
    auto newNumTiles = oldNumTiles ? parseIntArrayAttr<int64_t>(oldNumTiles) : SmallVector<int64_t>{};

    auto oldAlignment = oldDistribution.getAlignment();
    auto newAlignment = oldAlignment ? parseIntArrayAttr<int64_t>(oldAlignment) : SmallVector<int64_t>{};

    auto newWeightTableType = mlir::cast<vpux::NDTypeInterface>(newWeightTableOp.getOutput().getType());

    bool isZeroPointSubByte = false;
    if (mlir::isa<VPU::ZeroPointTableOp>(newWeightTableOp)) {
        const auto zps = mlir::cast<vpux::NDTypeInterface>(newWeightTableOp.getZeroPoints().getType());

        isZeroPointSubByte = zps.getElementType().isInteger(4) || zps.getElementType().isInteger(2);
    }

    auto newMemoryShapes = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getMemoryShapes());
    auto newComputeShapes = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getComputeShapes());

    auto newMemoryOffsets = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getMemoryOffsets());
    auto newComputeOffsets = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getComputeOffsets());

    bool isDuplicatedMode = oldDistribution.getMode().getValue() == VPU::DistributionMode::DUPLICATED;
    auto numClusters = oldDistribution.getNumClusters().getInt();

    // 4D: [OC, 1, 1, 1]
    // 5D: [Groups, OC, 1, 1, 1]
    const auto rank = newWeightTableType.getShape().size();
    const auto channelDim = getFilterChannelDim(rank);
    const auto channelDimIdx = channelDim.ind();

    // Determine final distribution mode based on workload analysis
    VPU::DistributionMode finalMode = oldDistribution.getMode().getValue();
    if (shouldSwitchToSegmented) {
        finalMode = VPU::DistributionMode::SEGMENTED;
        isDuplicatedMode = false;

        // Update num_tiles for SEGMENTED mode: segmented on OC dimension
        if (!newNumTiles.empty()) {
            newNumTiles[channelDimIdx] = numClusters;
        } else {
            // If num_tiles was not set, create it with segmentation on OC dimension
            auto shape = newWeightTableType.getShape();
            newNumTiles = SmallVector<int64_t>(shape.size(), 1);
            newNumTiles[channelDimIdx] = numClusters;
        }

        log.trace("Switching from DUPLICATED to SEGMENTED mode due to different workloads across clusters");
    }

    auto getAlignedSize = [&](int64_t workloadSize) {
        if (mlir::isa<VPU::ZeroPointTableOp>(newWeightTableOp)) {
            return VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointTableAlignmentForWorkload(
                    isZeroPointSubByte, static_cast<int32_t>(workloadSize));
        } else if (mlir::isa<VPU::DataPointerTableOp>(newWeightTableOp)) {
            return VPU::NCESparsity::NewWeightsTableFormatMapper::getNewPointerTableLogicalAlignmentForWorkload(
                    static_cast<int32_t>(workloadSize));
        } else {
            VPUX_THROW("Unsupported weight table op '{0}' at '{1}'", newWeightTableOp->getName(),
                       newWeightTableOp->getLoc());
        }
    };

    if (isDuplicatedMode) {
        // In DUPLICATED mode, each tile gets the same weight table
        auto totalOC = newWeightTableType.getShape()[channelDim];
        for (auto i = 0; i < numClusters; i++) {
            newMemoryShapes[i][channelDimIdx] = totalOC;
            newComputeShapes[i][channelDimIdx] = totalOC;
            // Offsets remain zero
        }
    } else {
        // In SEGMENTED mode, each tile gets its own piece from one weight table based on workload channels
        int32_t cumulativeOffset = 0;
        for (auto i = 0; i < numClusters; i++) {
            // Calculate total aligned size for all workloads in this cluster
            int32_t totalAlignedSize = 0;
            for (auto workloadSize : workloadChannels[i]) {
                totalAlignedSize += getAlignedSize(workloadSize);
            }

            newMemoryShapes[i][channelDimIdx] = totalAlignedSize;
            newComputeShapes[i][channelDimIdx] = totalAlignedSize;
            newMemoryOffsets[i][channelDimIdx] = cumulativeOffset;
            newComputeOffsets[i][channelDimIdx] = cumulativeOffset;

            cumulativeOffset += totalAlignedSize;
        }
    }

    auto overlapParams = VPU::OverlapDistributionParams(std::move(newMemoryShapes), std::move(newMemoryOffsets),
                                                        std::move(newComputeShapes), std::move(newComputeOffsets));

    // Determine if segments are uniform after potential mode switch
    bool hasUniformSegments = oldDistribution.getUniformDistributedSegments() != nullptr;
    if (shouldSwitchToSegmented) {
        // When switching to SEGMENTED, segments are non-uniform (different workloads per cluster)
        hasUniformSegments = false;
    }

    // Create new distributed type with manually set distribution parameters
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    auto newDistributedType = VPU::createExplicitDistributedTensorType(
            clusteredOp, newWeightTableType, finalMode, newNumTiles, oldDistribution.getNumClusters().getInt(),
            newAlignment, hasUniformSegments, overlapParams, std::nullopt);

    rewriter.setInsertionPoint(oldCopyOp);
    auto newCopyOp = rewriter.replaceOpWithNewOp<VPU::CopyOp>(
            oldCopyOp, newDistributedType, newWeightTableOp.getOutput(), oldCopyOp.getOutMemSpaceAttr());

    log.trace("Updated CopyOp for weight table: {0}", newCopyOp);
}

mlir::Operation* findWeightTableOp(mlir::Value value) {
    auto parentOp = value.getDefiningOp();
    VPUX_THROW_WHEN(parentOp == nullptr, "Unexpected NCE parent operation");

    return llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(parentOp)
            .Case<VPU::ZeroPointTableOp, VPU::DataPointerTableOp>([](mlir::Operation* op) {
                return op;
            })
            .Case<VPU::CopyOp>([&](VPU::CopyOp copyOp) {
                return findWeightTableOp(copyOp.getInput());
            })
            .Case<VPU::SliceOp>([&](VPU::SliceOp sliceOp) {
                return findWeightTableOp(sliceOp.getInput());
            })
            .Default([](mlir::Operation* op) -> mlir::Operation* {
                VPUX_THROW("Unexpected operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

template <typename WeightTableType>
void processWeightTableOp(mlir::IRRewriter& rewriter, VPU::NCEOpInterface nceOp, mlir::Value tableOperand,
                          llvm::DenseMap<WeightTableKey, mlir::Operation*>& createdTables, Logger log) {
    auto oldWeightTableOp = mlir::cast<WeightTableType>(findWeightTableOp(tableOperand));

    // 1. Extract workload channels and determine if we need to switch from DUPLICATED to SEGMENTED mode on the table.
    bool shouldSwitchToSegmented = false;
    const auto workloadChannels = extractWorkloadChannels(nceOp, shouldSwitchToSegmented);

    // 2. If table was sliced, remember a slice related to this current nceOp.
    VPU::SliceOp oldSliceOp;
    if (auto oldCopyOp = tableOperand.getDefiningOp<VPU::CopyOp>()) {
        oldSliceOp = oldCopyOp.getInput().getDefiningOp<VPU::SliceOp>();
    }

    // 3. If tableOperand (zero-points) is routed through a SliceOp, the zero-points input of the new op must cover only
    // the sliced channel range, not the full tensor. Create a new const.Declare with the relevant sub-range.
    mlir::Value adjustedZeroPoints = oldWeightTableOp.getZeroPoints();
    Const::DeclareOp zpConstOp;
    if (oldSliceOp != nullptr && oldWeightTableOp.getZeroPoints() != nullptr) {
        adjustedZeroPoints = sliceZeroPoints(rewriter, oldWeightTableOp, zpConstOp, oldSliceOp);
    }

    // 4. Materialize a new weight table based on workloads and append it to a set of created tables to reuse if the
    // same table is needed again.
    auto newWeightTableOp = updateWeightTableOp(rewriter, nceOp, oldWeightTableOp, workloadChannels,
                                                shouldSwitchToSegmented, createdTables, adjustedZeroPoints, log);

    // 5. Update CopyOp and its distribution information of a new table
    if (auto oldCopyOp = tableOperand.getDefiningOp<VPU::CopyOp>()) {
        updateCopyOp(rewriter, nceOp, oldCopyOp, newWeightTableOp, workloadChannels, shouldSwitchToSegmented, log);
        if (oldSliceOp != nullptr) {
            rewriter.replaceOp(oldSliceOp, newWeightTableOp.getResult());
        }
    }

    // 6. Remove old table and old zero-points constant.
    if (oldWeightTableOp->use_empty()) {
        rewriter.eraseOp(oldWeightTableOp);
    }

    if (zpConstOp != nullptr && zpConstOp->use_empty()) {
        rewriter.eraseOp(zpConstOp);
    }
}

//
// CreateNewWeightTablesData
//

class CreateNewWeightTablesDataPass final :
        public VPU::impl::CreateNewWeightTablesDataBase<CreateNewWeightTablesDataPass> {
public:
    explicit CreateNewWeightTablesDataPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CreateNewWeightTablesDataPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::IRRewriter rewriter(&getContext());

    // Track created weight tables by their data and attributes to avoid duplicates
    llvm::DenseMap<WeightTableKey, mlir::Operation*> createdTables;

    // Process weight table operations connected to NCE operations
    func->walk([&](VPU::NCEOpInterface nceOp) {
        // If we find any table, it needs to be updated
        auto dataPointerTable = nceOp.getWeightTableDataPtrOperand();
        auto zeroPointTable = nceOp.getWeightZeroPointsOperand();

        if (dataPointerTable == nullptr && zeroPointTable == nullptr) {
            return;
        }

        if (dataPointerTable) {
            processWeightTableOp<VPU::DataPointerTableOp>(rewriter, nceOp, dataPointerTable, createdTables, _log);
        } else if (zeroPointTable) {
            processWeightTableOp<VPU::ZeroPointTableOp>(rewriter, nceOp, zeroPointTable, createdTables, _log);
        }
    });
}

}  // namespace

//
// createCreateNewWeightTablesDataPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCreateNewWeightTablesDataPass(Logger log) {
    return std::make_unique<CreateNewWeightTablesDataPass>(log);
}
