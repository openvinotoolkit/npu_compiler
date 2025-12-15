//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Dialect/Utils/IndexingUtils.h"

namespace vpux::VPU {

//
// SCFViewLikeTilingModelOp
//

template <typename ConcreteModel, typename ConcreteOp>
class SCFViewLikeTilingModelOp : public mlir::TilingInterface::ExternalModel<ConcreteModel, ConcreteOp> {
protected:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        return static_cast<const ConcreteModel*>(this)->backInferSCFTileInfo(operation, builder, outputTile);
    }

public:
    SmallVector<mlir::Range> getIterationDomain(mlir::Operation*, mlir::OpBuilder&) const {
        // return empty list to prevent tiling without fusion
        return {};
    }

    mlir::FailureOr<mlir::TilingResult> getTiledImplementation(mlir::Operation*, mlir::OpBuilder&,
                                                               ArrayRef<mlir::OpFoldResult>,
                                                               ArrayRef<mlir::OpFoldResult>) const {
        // return failure to prevent tiling view like ops without fusion
        return mlir::failure();
    }

    mlir::FailureOr<mlir::TilingResult> generateResultTileValue(mlir::Operation* operation, mlir::OpBuilder& builder,
                                                                unsigned resultNumber,
                                                                ArrayRef<mlir::OpFoldResult> offsets,
                                                                ArrayRef<mlir::OpFoldResult> sizes) const {
        Bounds resultBounds;
        if (IE::hasDynamicTensors(operation) && operation->hasAttr(tilingStrategy)) {
            const auto strategy =
                    Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
            auto tilingDims = getSCFTilingOrderedDims(operation, strategy);
            if (mlir::failed(getResultTileBounds(operation, resultNumber, tilingDims, sizes, resultBounds))) {
                return mlir::failure();
            }
        }
        auto outputTile = SCFTileInfo(sizes, offsets, SCFShape(offsets.size(), builder.getIndexAttr(1)), resultBounds);
        auto inputTiling = backInferSCFTileInfo(operation, builder, outputTile);

        SmallVector<mlir::Value> tiledOperands;
        SmallVector<mlir::Operation*> generatedSlices;
        tiledOperands.reserve(operation->getNumOperands());

        for (auto p : operation->getOperands() | indexed) {
            auto origInput = p.value();
            auto inputIdx = p.index();

            if (inputTiling.tiles.size() <= inputIdx) {
                tiledOperands.emplace_back(origInput);
                continue;
            }

            auto inputTileInfo = inputTiling.tiles[inputIdx];
            auto tiledInput = generateTile(operation->getLoc(), builder, origInput, inputTileInfo, generatedSlices);

            tiledOperands.emplace_back(tiledInput);
        }

        auto resultDenseTile = extractResultType(operation->getResult(0).getType(), sizes, resultBounds);
        auto* tiledOp = mlir::cloneWithoutRegions(builder, operation, {resultDenseTile}, tiledOperands);
        vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::SHAPE);
        tiledOp->removeAttr(tilingStrategy);

        return mlir::TilingResult{{tiledOp}, {tiledOp->getResult(resultNumber)}, std::move(generatedSlices)};
    }

    mlir::LogicalResult getResultTilePosition(mlir::Operation*, mlir::OpBuilder&, unsigned,
                                              ArrayRef<mlir::OpFoldResult>, ArrayRef<mlir::OpFoldResult>,
                                              SmallVector<mlir::OpFoldResult>&,
                                              SmallVector<mlir::OpFoldResult>&) const {
        return mlir::failure();
    }
};

template <typename ConcreteOp>
class SCFGenericViewLikeTilingModelOp :
        public SCFViewLikeTilingModelOp<SCFGenericViewLikeTilingModelOp<ConcreteOp>, ConcreteOp> {
public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation*, mlir::OpBuilder&, const SCFTileInfo& outputTile) const {
        return SCFTilingInfo{SmallVector<SCFTileInfo, 1>{outputTile}};
    }
};

class SCFPermuteCastTilingModelOp : public SCFViewLikeTilingModelOp<SCFPermuteCastTilingModelOp, VPU::PermuteCastOp> {
public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* op, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto permuteCastOp = mlir::cast<VPU::PermuteCastOp>(op);

        const auto srcType = mlir::cast<vpux::NDTypeInterface>(permuteCastOp.getInput().getType());

        const auto toIntPermutation = [](auto dimsPermutation) {
            return to_small_vector(dimsPermutation | transformed([](Dim dim) {
                                       return checked_cast<int64_t>(dim.ind());
                                   }));
        };
        auto inputTile = outputTile;
        auto dstPerm = DimsOrder::fromAffineMap(permuteCastOp.getDstOrder());
        auto inversePerm = DimsOrder::fromAffineMap(mlir::inversePermutation(permuteCastOp.getMemPerm()));
        auto inverseSrcPerm = DimsOrder::fromAffineMap(
                mlir::inversePermutation(srcType.getDimsOrder().toAffineMap(builder.getContext())));

        auto permutation = toIntPermutation(
                applyPermutation(applyPermutation(dstPerm, inversePerm), inverseSrcPerm).toPermutation());

        mlir::applyPermutationToVector(inputTile.shape, permutation);
        mlir::applyPermutationToVector(inputTile.offsets, permutation);
        mlir::applyPermutationToVector(inputTile.axis, permutation);

        if (!inputTile.bounds.empty()) {
            inputTile.bounds = Bounds(mlir::applyPermutation(inputTile.bounds.raw(), permutation));
        }

        return SCFTilingInfo{{std::move(inputTile)}};
    }
};

}  // namespace vpux::VPU
