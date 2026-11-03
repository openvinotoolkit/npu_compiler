//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_BATCHMATMULTOMATMUL
#define GEN_PASS_DEF_BATCHMATMULTOMATMUL
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

// Drops the outermost group dimension from a 5-d shape or offset array attribute.
mlir::ArrayAttr dimArrayTo4D(const mlir::ArrayAttr dimArray) {
    VPUX_THROW_WHEN(dimArray == nullptr, "dimArrayTo4D expects a non-null attribute.");
    const auto dims = parseIntArrayAttr<int64_t>(dimArray);
    VPUX_THROW_WHEN(dims.size() != DimsGroups5D::Act::numDims, "dimArrayTo4D expects only 5-d arrays as inputs.");
    return getIntArrayAttr(dimArray.getContext(), ArrayRef<int64_t>(dims).drop_front());
}

VPUIP::HaloRegionAttr haloRegionTo4D(const VPUIP::HaloRegionAttr halo) {
    return VPUIP::HaloRegionAttr::get(halo.getContext(), dimArrayTo4D(halo.getShape()), dimArrayTo4D(halo.getOffset()),
                                      halo.getClusterId());
}

VPUIP::OutwardHaloRegionAttr outwardHaloRegionTo4D(const VPUIP::OutwardHaloRegionAttr halo) {
    SmallVector<mlir::Attribute> targetHalos;
    targetHalos.reserve(halo.getInwardHaloRegions().size());
    for (const auto targetHalo : halo.getInwardHaloRegions()) {
        targetHalos.push_back(haloRegionTo4D(mlir::cast<VPUIP::HaloRegionAttr>(targetHalo)));
    }

    return VPUIP::OutwardHaloRegionAttr::get(halo.getContext(), dimArrayTo4D(halo.getShape()),
                                             dimArrayTo4D(halo.getOffset()), halo.getClusterId(),
                                             mlir::ArrayAttr::get(halo.getContext(), targetHalos));
}

// ITIBufferType does not implement changeTypeComponents, so the 4-d layout must be rebuilt explicitly.
mlir::MemRefLayoutAttrInterface layoutTo4D(const mlir::MemRefLayoutAttrInterface origLayout, const DimsOrder newOrder,
                                           StridesRef newStrides, const Bit elemSize, mlir::MLIRContext* ctx) {
    const auto orderAttr = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));

    const auto memRefAttr = mlir::dyn_cast_or_null<vpux::MemRefAttr>(origLayout);
    if (memRefAttr == nullptr) {
        return vpux::MemRefAttr::get(orderAttr, /*optionalStrides=*/nullptr, /*optionalAllocSize=*/nullptr, ctx);
    }

    // The allocation size of the 5-d buffer does not describe a single group, thus it cannot be reused.
    VPUX_THROW_WHEN(memRefAttr.allocSize() != nullptr, "Explicit allocation size is not supported.");

    mlir::ArrayAttr newStridesAttr = nullptr;
    if (memRefAttr.strides() != nullptr) {
        const auto elemStrides = to_small_vector(newStrides | transformed([&](const Bit stride) {
                                                     return stride.count() / elemSize.count();
                                                 }));
        newStridesAttr = getIntArrayAttr(ctx, elemStrides);
    }

    return vpux::MemRefAttr::get(orderAttr, newStridesAttr, /*optionalAllocSize=*/nullptr, /*optionalBounds=*/{},
                                 memRefAttr.hwSpecificFields(), ctx);
}

mlir::Type typeTo4D(const mlir::Type type) {
    auto origType = mlir::cast<vpux::NDTypeInterface>(type);
    const ShapeRef origShape = origType.getShape();
    VPUX_THROW_WHEN(origShape.size() != DimsGroups5D::Act::numDims, "typeTo4D expects only 5-d shapes as inputs.");

    const Shape newShape(origShape.begin() + 1, origShape.end());
    const Strides origStrides = origType.getStrides();
    const Strides newStrides(origStrides.begin() + 1, origStrides.end());

    // Remove the first dimension from the permutation, then shift all remaining dimensions.
    // [d0, d1, d2, d3, d4] -> erase -> [d1, d2, d3, d4] -> shift -> [d0, d1, d2, d3]
    // [d0, d1, d3, d4, d2] -> erase -> [d1, d3, d4, d2] -> shift -> [d0, d2, d3, d1]
    auto permutation = origType.getDimsOrder().toPermutation();
    VPUX_THROW_WHEN(permutation.front() != DimsGroups5D::Act::G, "Groups must be the outermost dimension.");
    permutation.erase(permutation.begin());
    for (auto& dim : permutation) {
        dim = Dim(dim.ind() - 1);
    }

    const auto newOrder = DimsOrder::fromPermutation(permutation);

    // Update axis in per-axis quantization
    auto elemType = origType.getElementType();
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto origQuantAxis = perAxisType.getQuantizedDimension();
        VPUX_THROW_WHEN(origQuantAxis <= 0, "Quantization over batch is not supported.");
        const auto newQuantAxis = origQuantAxis - 1;
        if (const auto quantileStorage = mlir::dyn_cast<mlir::quant::QuantileType>(perAxisType.getStorageType())) {
            elemType = mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxisType.getFlags(), quantileStorage, perAxisType.getExpressedType(), perAxisType.getScales(),
                    perAxisType.getZeroPoints(), newQuantAxis, perAxisType.getStorageTypeMin(),
                    perAxisType.getStorageTypeMax());
        } else {
            elemType = mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxisType.getFlags(), perAxisType.getStorageType(), perAxisType.getExpressedType(),
                    perAxisType.getScales(), perAxisType.getZeroPoints(), newQuantAxis, perAxisType.getStorageTypeMin(),
                    perAxisType.getStorageTypeMax());
        }
    }

    // ITI buffers carry halo regions whose shapes and offsets have the same rank as the buffer itself.
    // NDTypeInterface::changeTypeComponents is not implemented for ITIBufferType, so the type is rebuilt here.
    if (auto itiType = mlir::dyn_cast<VPUIP::ITIBufferType>(type)) {
        auto* ctx = itiType.getContext();
        const auto newLayout = layoutTo4D(itiType.getLayout(), newOrder, newStrides, origType.getElemTypeSize(), ctx);

        SmallVector<VPUIP::HaloRegionAttr> newInwardHalos;
        newInwardHalos.reserve(itiType.getInwardHaloRegions().size());
        for (const auto halo : itiType.getInwardHaloRegions()) {
            newInwardHalos.push_back(haloRegionTo4D(halo));
        }

        SmallVector<VPUIP::OutwardHaloRegionAttr> newOutwardHalos;
        newOutwardHalos.reserve(itiType.getOutwardHaloRegions().size());
        for (const auto halo : itiType.getOutwardHaloRegions()) {
            newOutwardHalos.push_back(outwardHaloRegionTo4D(halo));
        }

        return VPUIP::ITIBufferType::get(ctx, newShape.raw(), elemType, newLayout, itiType.getMemSpace(),
                                         itiType.getIduSegmentation(), newInwardHalos, newOutwardHalos);
    }

    const auto newTypeComponents =
            TypeComponents().setShape(newShape).setDimsOrder(newOrder).setStrides(newStrides).setElementType(elemType);
    return origType.changeTypeComponents(newTypeComponents);
}

// Helper class for keeping track of buffer slices created by unrolling. Main purpose is
// to ensure that buffers that point to the same slice are deduplicated. Passes below expect
// unique DeclareBufferOp per buffer slice.
class BufferSlicer final {
public:
    VPURT::DeclareBufferOp getOrCreate(const mlir::Value val, const int64_t step) {
        auto origProducer = val.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_WHEN(origProducer == nullptr, "MemRef producer must be a VPURT::DeclareBufferOp");

        const auto key = BufferSliceKey{origProducer.getOperation(), step};

        const auto cached = _slices.find(key);
        if (cached != _slices.end()) {
            return cached->second;
        }

        const auto sliceType = typeTo4D(origProducer.getBuffer().getType());
        const auto byteOffset = origProducer.getByteOffset() + step;

        mlir::OpBuilder builder(origProducer);
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(origProducer);

        auto producer4d = mlir::cast<VPURT::DeclareBufferOp>(builder.clone(*origProducer));
        producer4d.setByteOffset(byteOffset);
        producer4d.getBuffer().setType(sliceType);

        _slices.insert({key, producer4d});

        return producer4d;
    }

private:
    using BufferSliceKey = std::pair<mlir::Operation* /*origProducer*/, int64_t /*step*/>;
    DenseMap<BufferSliceKey, VPURT::DeclareBufferOp> _slices;
};

int64_t getStep(const mlir::Type type) {
    auto origType = mlir::cast<vpux::NDTypeInterface>(type);
    auto batchSize = origType.getShape()[DimsGroups5D::Act::G];
    return origType.getTotalAllocSize().count() / batchSize;
}

void unrollBatchMatMul(VPURT::TaskOp vpurtTask, BufferSlicer& slicer) {
    auto nceOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
    const auto origType = mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType());
    const auto batchSize = origType.getShape()[DimsGroups5D::Act::G];

    const auto inputStep = getStep(nceOp.getInput().getType());
    const auto weightsStep = getStep(nceOp.getWeights().getType());
    const auto weightTableStep = nceOp.getWeightTable() == nullptr ? 0 : getStep(nceOp.getWeightTable().getType());

    // Scale, bias and zero-point tables are always created with G=1 and are shared across all unrolled MatMul tasks.
    // Create each 4D buffer once and reuse it instead of slicing per group.
    const auto checkSharedTableHasSingleGroup = [](mlir::Value table, llvm::StringRef tableName) {
        if (table == nullptr) {
            return;
        }
        VPUX_THROW_UNLESS(mlir::cast<vpux::NDTypeInterface>(table.getType()).getShape()[DimsGroups5D::Act::G] == 1,
                          "{0} must have G=1", tableName);
    };
    const auto scaleTable = nceOp.getWeightTableScale();
    const auto biasTable = nceOp.getWeightTableBias();
    const auto zeroPointTable = nceOp.getWeightZeroPoints();
    checkSharedTableHasSingleGroup(scaleTable, "Scale table");
    checkSharedTableHasSingleGroup(biasTable, "Bias table");
    checkSharedTableHasSingleGroup(zeroPointTable, "Zero-point table");

    auto sharedScaleTable4d = scaleTable != nullptr ? slicer.getOrCreate(scaleTable, /*step=*/0) : nullptr;
    auto sharedBiasTable4d = biasTable != nullptr ? slicer.getOrCreate(biasTable, /*step=*/0) : nullptr;
    auto sharedZeroPointTable4d = zeroPointTable != nullptr ? slicer.getOrCreate(zeroPointTable, /*step=*/0) : nullptr;

    // When reuse is enabled, the legacy weight table has G=1 and is shared across all groups.
    const auto weightTable = nceOp.getWeightTable();
    const bool isSharedWeightTable =
            weightTable != nullptr &&
            mlir::cast<vpux::NDTypeInterface>(weightTable.getType()).getShape()[DimsGroups5D::Filter::G] == 1;
    auto sharedWeightTable4d = isSharedWeightTable ? slicer.getOrCreate(weightTable, /*step=*/0) : nullptr;

    const auto parentInputStep = getStep(nceOp.getParentInput().getType());
    const auto parentOutputStep = getStep(nceOp.getParentOutput().getType());
    const auto outputProducerStep = getStep(nceOp.getOutputBuff().getType());

    // Output ITI buffers are the broadcast destinations in neighbouring tiles. Each of them stores the same 5-d
    // tensor, so every group occupies the same byte step as in the local output buffer.
    const SmallVector<mlir::Value> outputItiBuffs(nceOp.getOutput_ITIBuff().begin(), nceOp.getOutput_ITIBuff().end());
    SmallVector<int64_t> outputItiSteps;
    outputItiSteps.reserve(outputItiBuffs.size());
    for (const auto itiBuff : outputItiBuffs) {
        outputItiSteps.push_back(getStep(itiBuff.getType()));
    }

    mlir::OpBuilder builder(vpurtTask);
    for (const auto& idx : irange(batchSize)) {
        builder.setInsertionPointAfter(vpurtTask);
        auto newVpurtTask = mlir::cast<VPURT::TaskOp>(builder.clone(*vpurtTask));
        auto innerTask = newVpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
        VPUX_THROW_WHEN(innerTask == nullptr, "Cloned operation must have inner NCEClusterTaskOp");

        // Reverse the order of operations.
        const auto pos = batchSize - idx - 1;
        auto inputProducer4d = slicer.getOrCreate(innerTask.getInput(), inputStep * pos);
        auto weightProducer4d = slicer.getOrCreate(innerTask.getWeights(), weightsStep * pos);
        auto parentInputProducer4d = slicer.getOrCreate(innerTask.getParentInput(), parentInputStep * pos);
        auto parentOutputProducer4d = slicer.getOrCreate(innerTask.getParentOutput(), parentOutputStep * pos);
        auto outputProducer4d = slicer.getOrCreate(innerTask.getOutputBuff(), outputProducerStep * pos);

        mlir::IRMapping mapper;
        mapper.map(innerTask.getInput(), inputProducer4d);
        mapper.map(innerTask.getWeights(), weightProducer4d);
        if (innerTask.getWeightTable() != nullptr) {
            if (isSharedWeightTable) {
                mapper.map(innerTask.getWeightTable(), sharedWeightTable4d);
            } else {
                auto weightTableProducer4d = slicer.getOrCreate(innerTask.getWeightTable(), weightTableStep * pos);
                mapper.map(innerTask.getWeightTable(), weightTableProducer4d);
            }
        }
        if (innerTask.getWeightTableScale() != nullptr) {
            mapper.map(innerTask.getWeightTableScale(), sharedScaleTable4d);
        }
        if (innerTask.getWeightTableBias() != nullptr) {
            mapper.map(innerTask.getWeightTableBias(), sharedBiasTable4d);
        }
        if (innerTask.getWeightZeroPoints() != nullptr) {
            mapper.map(innerTask.getWeightZeroPoints(), sharedZeroPointTable4d);
        }
        mapper.map(innerTask.getParentInput(), parentInputProducer4d);
        mapper.map(innerTask.getParentOutput(), parentOutputProducer4d);
        mapper.map(innerTask.getOutputBuff(), outputProducer4d);

        for (const auto& itiBuff : innerTask.getOutput_ITIBuff() | indexed) {
            auto itiProducer4d = slicer.getOrCreate(itiBuff.value(), outputItiSteps[itiBuff.index()] * pos);
            mapper.map(itiBuff.value(), itiProducer4d);
        }

        builder.setInsertionPointAfter(innerTask);

        auto newOp = mlir::cast<VPUIP::NCEClusterTaskOp>(builder.clone(*innerTask, mapper));
        newOp.getOutput().setType(typeTo4D(newOp.getOutput().getType()));
        innerTask->replaceAllUsesWith(newOp.getOutputs());
        innerTask->erase();
    }
}

class BatchMatMulToMatMul final : public VPUIP::impl::BatchMatMulToMatMulBase<BatchMatMulToMatMul> {
public:
    explicit BatchMatMulToMatMul(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BatchMatMulToMatMul::safeRunOnFunc() {
    auto func = getOperation();

    const auto is5DTask = [](VPURT::TaskOp vpurtTask) -> bool {
        auto nceOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
        if (nceOp == nullptr) {
            return false;
        }
        if (getShape(nceOp.getOutput()).size() != DimsGroups5D::Act::numDims) {
            return false;
        }
        return true;
    };

    // Collect the tasks up front, the IR is modified while unrolling.
    llvm::SmallVector<VPURT::TaskOp> tasksToUnroll;
    for (auto vpurtTask : func.getOps<VPURT::TaskOp>()) {
        if (is5DTask(vpurtTask)) {
            tasksToUnroll.push_back(vpurtTask);
        }
    }

    if (tasksToUnroll.empty()) {
        return;
    }

    BufferSlicer slicer;

    SmallVector<VPURT::TaskOp> deadTasks;
    SmallVector<mlir::Operation*> staleBuffers;
    for (auto& vpurtTask : tasksToUnroll) {
        _log.trace("Unrolling '{0}'", vpurtTask->getLoc());
        unrollBatchMatMul(vpurtTask, slicer);
        deadTasks.push_back(vpurtTask);

        const auto collectProducer = [&staleBuffers](mlir::Value val) {
            if (val == nullptr) {
                return;
            }
            if (auto producer = val.getDefiningOp()) {
                staleBuffers.push_back(producer);
            }
        };

        auto nceOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
        const SmallVector<mlir::Value> outputItiBuffs(nceOp.getOutput_ITIBuff().begin(),
                                                      nceOp.getOutput_ITIBuff().end());
        collectProducer(nceOp.getInput());
        collectProducer(nceOp.getWeights());
        collectProducer(nceOp.getWeightTable());
        collectProducer(nceOp.getWeightTableScale());
        collectProducer(nceOp.getWeightTableBias());
        collectProducer(nceOp.getWeightZeroPoints());
        collectProducer(nceOp.getParentInput());
        collectProducer(nceOp.getParentOutput());
        collectProducer(nceOp.getOutputBuff());
        for (const auto itiBuff : outputItiBuffs) {
            collectProducer(itiBuff);
        }
    }

    for (auto deadTask : deadTasks) {
        deadTask->erase();
    }

    llvm::DenseSet<mlir::Operation*> uniqueProducers;
    for (auto* producer : staleBuffers) {
        if (!uniqueProducers.insert(producer).second) {
            continue;
        }
        if (producer->use_empty()) {
            producer->erase();
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createBatchMatMulToMatMulPass(Logger log) {
    return std::make_unique<BatchMatMulToMatMul>(log);
}
