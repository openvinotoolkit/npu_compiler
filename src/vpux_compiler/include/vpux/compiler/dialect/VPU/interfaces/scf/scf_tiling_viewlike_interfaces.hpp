//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

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
        auto outputTile = SCFTileInfo(sizes, offsets, SCFShape(offsets.size(), builder.getIndexAttr(1)));
        auto inputTiling = backInferSCFTileInfo(operation, builder, outputTile);

        SmallVector<mlir::Value> tiledOperands;
        tiledOperands.reserve(operation->getNumOperands());

        for (auto p : operation->getOperands() | indexed) {
            auto origInput = p.value();
            auto inputIdx = p.index();

            if (inputTiling.tiles.size() <= inputIdx) {
                tiledOperands.emplace_back(origInput);
                continue;
            }

            auto inputTileInfo = inputTiling.tiles[inputIdx];
            auto tiledInput = generateTile(operation->getLoc(), builder, origInput, inputTileInfo);

            tiledOperands.emplace_back(tiledInput);
        }

        auto resultDenseTile = extractResultType(operation->getResult(0).getType(), sizes, {});
        auto* tiledOp = mlir::cloneWithoutRegions(builder, operation, {resultDenseTile}, tiledOperands);
        tiledOp->removeAttr(tilingStrategy);

        return mlir::TilingResult{{tiledOp}, {tiledOp->getResult(resultNumber)}};
    }

    mlir::LogicalResult getResultTilePosition(mlir::Operation*, mlir::OpBuilder&, unsigned,
                                              ArrayRef<mlir::OpFoldResult>, ArrayRef<mlir::OpFoldResult>,
                                              SmallVector<mlir::OpFoldResult>&,
                                              SmallVector<mlir::OpFoldResult>&) const {
        return mlir::failure();
    }
};

class SCFLayoutCastTilingModelOp : public SCFViewLikeTilingModelOp<SCFLayoutCastTilingModelOp, VPU::LayoutCastOp> {
public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation*, mlir::OpBuilder&, const SCFTileInfo& outputTile) const {
        return SCFTilingInfo{{outputTile}};
    }
};

}  // namespace vpux::VPU
