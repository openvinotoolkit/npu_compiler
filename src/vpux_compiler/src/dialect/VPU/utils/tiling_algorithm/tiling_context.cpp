//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Interfaces/TilingInterface.h"

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/multicluster_tiling_scf_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_general_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_scf_algorithm.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Interfaces/InferTypeOpInterface.h>

using namespace vpux;
using namespace VPU;

TilingContext::TilingContext(mlir::Operation* operation): _operation(operation) {
}

void TilingContext::setTiling(std::unique_ptr<ITilingAlgorithm> tilingAlgorithm) {
    _tilingAlgorithm = std::move(tilingAlgorithm);
}

mlir::LogicalResult TilingContext::applyTiling(mlir::RewriterBase& builder, Logger log) {
    VPUX_THROW_WHEN(_tilingAlgorithm == nullptr, "Tiling algorithm is not specified");

    return _tilingAlgorithm->applyTiling(_operation, builder, log);
}

SmallVector<mlir::Operation*> TilingContext::applySCFTilingAndFusion(mlir::RewriterBase& builder,
                                                                     const MergeConfiguration& mergeConfig,
                                                                     Logger log) {
    VPUX_THROW_WHEN(_tilingAlgorithm == nullptr, "Tiling algorithm is not specified");

    return _tilingAlgorithm->applySCFTilingAndFusion(_operation, builder, mergeConfig, log);
}

namespace {

// Scales and zero points of a per-axis quantized type are part of the element type, which is shared by
// every iteration of the generated scf.for, so a tile cannot carry its own slice of them. Such tiling is
// handed over to the non-SCF algorithm, which materializes a separate tile type per slice.
bool tilesPerAxisQuantDim(mlir::Operation* operation, ShapeRef strategy) {
    const auto getPerAxisQuantDim = [](mlir::Value value) -> std::optional<int64_t> {
        const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(value.getType());
        if (ndType == nullptr) {
            return std::nullopt;
        }
        const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(ndType.getElementType());
        if (perAxisQType == nullptr) {
            return std::nullopt;
        }
        return perAxisQType.getQuantizedDimension();
    };

    const auto isTiledDim = [&](int64_t dimInd) {
        return dimInd >= 0 && checked_cast<size_t>(dimInd) < strategy.size() && strategy[Dim(dimInd)] > 1;
    };

    for (auto result : operation->getResults()) {
        const auto quantDim = getPerAxisQuantDim(result);
        if (quantDim.has_value() && isTiledDim(quantDim.value())) {
            return true;
        }
    }

    auto nceOperation = mlir::dyn_cast<VPU::NCEOpInterface>(operation);
    const auto channelDim = getShape(operation->getResult(0)).size() == 5 ? DimsGroups5D::Act::C : Dims4D::Act::C;

    for (auto operand : operation->getOperands()) {
        const auto quantDim = getPerAxisQuantDim(operand);
        if (!quantDim.has_value()) {
            continue;
        }
        // NCE weights hold one entry per output channel, so their quantized axis follows the output
        // channel dimension instead of the operand dimension with the same index.
        const auto tiledDim = (nceOperation != nullptr && nceOperation.getWeightsOperand() == operand)
                                      ? channelDim.ind()
                                      : quantDim.value();
        if (isTiledDim(tiledDim)) {
            return true;
        }
    }

    return false;
}

}  // namespace

bool isSCFSupported(mlir::Operation* operation) {
    const auto outShape = getShape(operation->getResult(0));

    if (outShape.isDynamic() && !mlir::isa<mlir::ReifyRankedShapedTypeOpInterface>(operation)) {
        return false;
    }

    // E-172335 add sparse tensors support
    if (operation->hasAttr(tilingStrategy)) {
        auto sparseCheckOperandInd = 0;
        auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        if (auto nceOperation = mlir::dyn_cast<VPU::NCEOpInterface>(operation)) {
            if (nceOperation.getWeightsOperand() != 0 && strategy[Dims4D::Act::C] > 1) {
                sparseCheckOperandInd = 1;
            }
        }
        if (mlir::isa<VPU::SparseTensorType>(operation->getOperand(sparseCheckOperandInd).getType())) {
            return false;
        }

        if (tilesPerAxisQuantDim(operation, strategy)) {
            return false;
        }
    }

    return true;
}

TilingContext vpux::VPU::createTilingContext(mlir::Operation* operation, const TilingContextOptions& options) {
    TilingContext context(operation);

    std::unique_ptr<ITilingAlgorithm> algorithm;

    if (options.enableSCFTiling && mlir::isa<mlir::TilingInterface>(operation) && isSCFSupported(operation)) {
        if (options.type == TilingContextOptions::ContextType::MULTICLUSTERING) {
            algorithm = std::make_unique<MulticlusterTilingSCFAlgorithm>();
        } else {
            algorithm = std::make_unique<TilingSCFAlgorithm>();
        }
    } else {
        VPUX_THROW_WHEN(options.type != TilingContextOptions::ContextType::TILING,
                        "TilingContext without scf can only be of TILING type.");
        algorithm = std::make_unique<TilingGeneralAlgorithm>();
    }

    context.setTiling(std::move(algorithm));

    return context;
}
