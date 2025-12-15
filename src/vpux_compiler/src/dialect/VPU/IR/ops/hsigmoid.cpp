//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::HSigmoidOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::HSigmoidOpAdaptor hsigmoid(operands, attrs, prop);
    if (mlir::failed(hsigmoid.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = hsigmoid.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
