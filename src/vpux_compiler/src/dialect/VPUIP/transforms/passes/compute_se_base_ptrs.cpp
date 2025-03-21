//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/loop.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_COMPUTESEBASEPTRS
#define GEN_PASS_DEF_COMPUTESEBASEPTRS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

void computeBasePtrs(VPUIP::StorageElementTableOp seTableOp, vpux::NDTypeInterface inputType, const Logger& log) {
    log.trace("Computing base pointers for '{0}' at '{1}'", seTableOp->getName(), seTableOp->getLoc());

    if (seTableOp.getBasePtrs().has_value()) {
        log.nest().trace("Operation already has base pointers computed. Skipping");
        return;
    }

    const auto seDataShape = parseIntArrayAttr<int64_t>(seTableOp.getDataShape());
    const auto inputShape = to_small_vector(inputType.getShape());
    VPUX_THROW_UNLESS(seDataShape == inputShape,
                      "Mismatch between storage element table's data shape '{0}' and input shape '{1}'", seDataShape,
                      inputShape);

    SmallVector<Shape> perClusterOffsets{};
    SmallVector<Shape> perClusterShapes{};
    Dim distributedTilingAxis(0);

    if (auto inputDistType = inputType.dyn_cast<VPUIP::DistributedBufferType>()) {
        perClusterOffsets = inputDistType.getPerClusterMemoryShapeOffsets();
        perClusterShapes = inputDistType.getPerClusterMemoryShapes();
        VPUX_THROW_UNLESS(perClusterOffsets.size() == perClusterShapes.size(),
                          "Mismatch between per cluster offsets '{0}' and shapes '{1}", perClusterOffsets.size(),
                          perClusterShapes.size());

        auto distributionAttr = inputDistType.getDistribution();
        if (auto numTiles = distributionAttr.getNumTiles()) {
            distributedTilingAxis = Dim(VPU::getDistributedTilingAxis(parseIntArrayAttr<int64_t>(numTiles)));
        }
    }

    int64_t padLeft = 0;
    int64_t padTop = 0;
    const auto sePaddingAttr = mlir::dyn_cast_or_null<VPU::SEPaddingAttr>(seTableOp.getSeAttr().value_or(nullptr));
    if (sePaddingAttr != nullptr) {
        const auto padding = parseIntArrayAttr<int64_t>(sePaddingAttr.getPadding());
        padLeft = padding[VPU::SE_PAD_LEFT];
        padTop = padding[VPU::SE_PAD_TOP];
    }

    // SEPaddingAttr has a specific scenario compared to other SEAttr
    // Example: Pad Op with 'Reflect' mode and padLeft/padRight is 2
    //
    // Input Data:                             |  1  |  2  |  3  |  4  |
    // Effective Data:             |  3  |  2  |  1  |  2  |  3  |  4  |  3  |  2  |
    //
    // If equally tiled into two clusters, each cluster has size 4
    // Input Data in Cluster 0:                |  1  |  2  |  3  |
    // Input Data in Cluster 1:                      |  2  |  3  |  4  |
    //
    // If searching clusters from front to end according to the input data shape and offsets
    // BASE_PTR in Cluster 0:      |  0  |  0  |  0  |  0  |
    // BASE_PTR in Cluster 1:                              |  0  |  1  |  0  |  0  |
    //
    // BASE_PTR in Cluster 1 has an issue, expected value: |  0  |  1  |  1  |  1  |
    // Update logic to use the last cluster index when the coordinate is at the SEPadding end
    // and input spans multiple clusters

    const auto findCluster = [&](const int64_t inH, const int64_t inW, const int64_t outH,
                                 const int64_t outW) -> int32_t {
        if (perClusterOffsets.empty() || perClusterShapes.empty()) {
            return static_cast<int32_t>(0);
        }

        const auto isCoordAtSEPaddingEnd =
                (sePaddingAttr != nullptr) &&
                ((distributedTilingAxis == Dims4D::Act::H && outH >= padTop + inputShape[Dims4D::Act::H.ind()]) ||
                 (distributedTilingAxis == Dims4D::Act::W && outW >= padLeft + inputShape[Dims4D::Act::W.ind()]));

        int32_t clusterIdx = -1;
        for (const auto& [index, value] : zip(perClusterOffsets, perClusterShapes) | indexed) {
            const auto& [offsets, shape] = value;

            auto containsH = offsets[Dims4D::Act::H] <= inH && inH < (offsets[Dims4D::Act::H] + shape[Dims4D::Act::H]);
            auto containsW = offsets[Dims4D::Act::W] <= inW && inW < (offsets[Dims4D::Act::W] + shape[Dims4D::Act::W]);

            if (containsH && containsW) {
                clusterIdx = checked_cast<int32_t>(index);
                if (isCoordAtSEPaddingEnd) {
                    continue;
                }
                return clusterIdx;
            }
        }

        return clusterIdx;
    };

    const auto outputNDType = seTableOp.getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputNDType.getShape();
    const auto outputH = outputShape[Dims4D::Act::H];
    const auto outputW = outputShape[Dims4D::Act::W];

    const auto seAttr = seTableOp.getSeAttr().value_or(nullptr);
    const auto seSize = seTableOp.getSeSize();
    const auto seDepth = seTableOp.getSeDepth();

    const auto numElements = outputNDType.getNumElements();
    std::vector<int32_t> basePtrs(numElements, 0);

    SmallVector<int64_t> outputOffsets{0, 0, 0, 0};

    if (seAttr != nullptr) {
        auto tileInfo = seAttr.getTileInfo();

        if (tileInfo.has_value() && tileInfo.value().offsets != nullptr && !tileInfo.value().offsets.empty()) {
            outputOffsets = parseIntArrayAttr<int64_t>(tileInfo.value().offsets);
        }
    }

    loop_3d(LoopExecPolicy::Parallel, seTableOp.getContext(), outputH, outputW, seDepth,
            [&](int64_t h, int64_t w, int64_t se) {
                const Shape outputCoord{
                        outputOffsets[Dims4D::Act::N.ind()] + 0, outputOffsets[Dims4D::Act::C.ind()] + se * seSize,
                        outputOffsets[Dims4D::Act::H.ind()] + h, outputOffsets[Dims4D::Act::W.ind()] + w};

                auto inputCoord =
                        (seAttr != nullptr) ? seAttr.backInferInputCoord(outputCoord, Shape(inputShape)) : outputCoord;

                const auto seOffset = (h * outputW + w) * seDepth + se;  // HWC

                basePtrs[seOffset] = findCluster(inputCoord[Dims4D::Act::H], inputCoord[Dims4D::Act::W],
                                                 outputCoord[Dims4D::Act::H], outputCoord[Dims4D::Act::W]);
            });

    const auto basePtrType =
            mlir::RankedTensorType::get({static_cast<int64_t>(basePtrs.size())}, getInt32Type(seTableOp.getContext()));

    const auto basePtrsElems = mlir::DenseIntElementsAttr::get(basePtrType, basePtrs);

    seTableOp.setBasePtrsAttr(basePtrsElems);
}

//
// ComputeSEBasePtrsPass
//

class ComputeSEBasePtrsPass final : public VPUIP::impl::ComputeSEBasePtrsBase<ComputeSEBasePtrsPass> {
public:
    explicit ComputeSEBasePtrsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ComputeSEBasePtrsPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.getInputStorageElementTable() == nullptr) {
            return;
        }

        _log.trace("Found NCE operation with an input SE table at '{0}'", nceOp->getLoc());

        VPUX_THROW_WHEN(nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE,
                        "Eltwise operations with input storage element tables are not yet supported");

        auto inputOperand = nceOp.getInput();
        auto seTableOperand = nceOp.getInputStorageElementTable();
        if (auto parentTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
            auto inputArg = nceOp.getInput().dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(inputArg == nullptr, "Input is not a block argument");
            inputOperand = parentTilingOp.getOperand(inputArg.getArgNumber());

            auto seTableBlockArg = nceOp.getInputStorageElementTable().dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(seTableBlockArg == nullptr, "Input storage element table is not a block argument");
            seTableOperand = parentTilingOp.getOperand(seTableBlockArg.getArgNumber());
        }

        auto seTable = VPUIP::findSETableOp(seTableOperand);
        VPUX_THROW_WHEN(seTable == nullptr, "Unable to find the storage element table");
        if (mlir::isa<Const::DeclareOp>(seTable)) {
            _log.nest().trace("Storage element table was already converted to a constant");
            return;
        }

        auto seTableOp = mlir::cast<VPUIP::StorageElementTableOp>(seTable);
        auto inputType = inputOperand.getType().cast<vpux::NDTypeInterface>();
        computeBasePtrs(seTableOp, inputType, _log);
    });
}

}  // namespace

//
// createComputeSEBasePtrsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createComputeSEBasePtrsPass(Logger log) {
    return std::make_unique<ComputeSEBasePtrsPass>(log);
}
