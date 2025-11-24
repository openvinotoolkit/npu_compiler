//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Interfaces/TilingInterface.h"

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_general_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_scf_algorithm.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"

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

mlir::FailureOr<SmallVector<mlir::Operation*>> TilingContext::applyVerticalFusion(mlir::RewriterBase& builder,
                                                                                  Logger log) {
    VPUX_THROW_WHEN(_tilingAlgorithm == nullptr, "Tiling algorithm is not specified");

    return _tilingAlgorithm->applyVerticalFusion(_operation, builder, log);
}

bool isSCFSupported(mlir::Operation* operation) {
    // E-162801 extend for operations with > 1 output
    if (operation->getNumResults() > 1) {
        return false;
    }

    const auto outShape = getShape(operation->getResult(0));

    if (outShape.isDynamic() && !mlir::isa<mlir::ReifyRankedShapedTypeOpInterface>(operation)) {
        return false;
    }

    // E-172335 add sparse tensors support
    if (mlir::isa<VPU::SparseTensorType>(operation->getOperand(0).getType())) {
        return false;
    }
    return true;
}

TilingContext vpux::VPU::createTilingContext(mlir::Operation* operation, bool enableSCFTiling) {
    TilingContext context(operation);

    std::unique_ptr<ITilingAlgorithm> algorithm;

    if (enableSCFTiling && mlir::isa<mlir::TilingInterface>(operation) && isSCFSupported(operation)) {
        algorithm = std::make_unique<TilingSCFAlgorithm>();
    } else {
        algorithm = std::make_unique<TilingGeneralAlgorithm>();
    }

    context.setTiling(std::move(algorithm));

    return context;
}
