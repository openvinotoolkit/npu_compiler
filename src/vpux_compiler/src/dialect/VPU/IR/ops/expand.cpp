//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExpandOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExpandOpAdaptor expand(operands, attrs, prop);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(expand.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(expand.getPadsEnd());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(expand.getInput().getType());

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    inferredReturnTypes.push_back(newType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::ExpandOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    auto operands = adaptor.getOperands();
    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto padsBefore = Shape(parseIntArrayAttr<int64_t>(getPadsBegin()));
        const auto padsAfter = Shape(parseIntArrayAttr<int64_t>(getPadsEnd()));
        return static_cast<Const::ContentAttr>(attr).transform().padWithZero(padsBefore, padsAfter).get();
    }

    return nullptr;
}
