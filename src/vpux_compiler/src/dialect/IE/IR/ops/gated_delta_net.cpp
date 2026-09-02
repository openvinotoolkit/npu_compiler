//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatedDeltaNetOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatedDeltaNetOpAdaptor gdn(operands, attrs, prop);
    if (mlir::failed(gdn.verify(loc))) {
        return mlir::failure();
    }

    const auto queryType = mlir::cast<mlir::RankedTensorType>(gdn.getQuery().getType());
    const auto valueType = mlir::cast<mlir::RankedTensorType>(gdn.getValue().getType());
    const auto stateType = mlir::cast<mlir::RankedTensorType>(gdn.getRecurrentState().getType());

    inferredReturnShapes.emplace_back(valueType.getShape(), queryType.getElementType(), vpux::getTensorAttr(valueType));
    inferredReturnShapes.emplace_back(stateType.getShape(), stateType.getElementType(), vpux::getTensorAttr(stateType));

    return mlir::success();
}

mlir::LogicalResult vpux::IE::GatedDeltaNetOp::verify() {
    const auto queryShape = mlir::cast<mlir::RankedTensorType>(getQuery().getType()).getShape();
    const auto keyShape = mlir::cast<mlir::RankedTensorType>(getKey().getType()).getShape();
    const auto valueShape = mlir::cast<mlir::RankedTensorType>(getValue().getType()).getShape();
    const auto stateShape = mlir::cast<mlir::RankedTensorType>(getRecurrentState().getType()).getShape();
    const auto gateShape = mlir::cast<mlir::RankedTensorType>(getGate().getType()).getShape();
    const auto betaShape = mlir::cast<mlir::RankedTensorType>(getBeta().getType()).getShape();

    if (queryShape.size() != 4 || keyShape.size() != 4 || valueShape.size() != 4 || stateShape.size() != 4) {
        return errorAt(*this, "GatedDeltaNet expects 4D query/key/value/recurrent_state, got {0}/{1}/{2}/{3}",
                       queryShape.size(), keyShape.size(), valueShape.size(), stateShape.size());
    }
    if (gateShape.size() != 3 || betaShape.size() != 3) {
        return errorAt(*this, "GatedDeltaNet expects 3D gate/beta, got {0} and {1}", gateShape.size(),
                       betaShape.size());
    }

    constexpr int64_t MAX_STATE_DIM = 256;
    const auto D = queryShape.back();
    const auto Dv = valueShape.back();
    if ((!mlir::ShapedType::isDynamic(D) && D <= 0) || (!mlir::ShapedType::isDynamic(Dv) && Dv <= 0)) {
        return errorAt(*this, "GatedDeltaNet requires positive qk head size ({0}) and v head size ({1})", D, Dv);
    }
    if (!mlir::ShapedType::isDynamic(D) && D > MAX_STATE_DIM) {
        return errorAt(*this, "GatedDeltaNet qk head size {0} exceeds the maximum supported state dim {1}", D,
                       MAX_STATE_DIM);
    }

    const auto qkH = queryShape[queryShape.size() - 2];
    const auto vH = valueShape[valueShape.size() - 2];
    if (!mlir::ShapedType::isDynamic(qkH) && !mlir::ShapedType::isDynamic(vH) &&
        (qkH <= 0 || vH <= 0 || vH % qkH != 0)) {
        return errorAt(*this, "GatedDeltaNet requires positive qk_H ({0}) and v_H ({1}) with v_H divisible by qk_H",
                       qkH, vH);
    }

    const auto dimEq = [](int64_t a, int64_t b) {
        return mlir::ShapedType::isDynamic(a) || mlir::ShapedType::isDynamic(b) || a == b;
    };
    const auto B = queryShape[0];
    const auto S = queryShape[1];
    bool shapesOk = dimEq(keyShape[0], B) && dimEq(keyShape[1], S) && dimEq(keyShape[2], qkH) && dimEq(keyShape[3], D);
    shapesOk = shapesOk && dimEq(valueShape[0], B) && dimEq(valueShape[1], S);
    shapesOk = shapesOk && dimEq(stateShape[0], B) && dimEq(stateShape[1], vH) && dimEq(stateShape[2], D) &&
               dimEq(stateShape[3], valueShape[3]);
    shapesOk = shapesOk && dimEq(gateShape[0], B) && dimEq(gateShape[1], S) && dimEq(gateShape[2], vH) &&
               dimEq(betaShape[0], B) && dimEq(betaShape[1], S) && dimEq(betaShape[2], vH);
    if (!shapesOk) {
        return errorAt(*this, "GatedDeltaNet input shapes are inconsistent: expect key==query [B,S,qk_H,D], "
                              "value [B,S,v_H,Dv], recurrent_state [B,v_H,D,Dv], gate/beta [B,S,v_H]");
    }

    const auto queryElemType = mlir::cast<mlir::RankedTensorType>(getQuery().getType()).getElementType();
    for (const auto& operand : {getKey(), getValue(), getGate(), getBeta()}) {
        const auto elemType = mlir::cast<mlir::RankedTensorType>(operand.getType()).getElementType();
        if (elemType != queryElemType) {
            return errorAt(*this,
                           "GatedDeltaNet requires query/key/value/gate/beta to share the query element type {0}, "
                           "got {1}",
                           queryElemType, elemType);
        }
    }

    if (getQL2NormEps().convertToDouble() < 0.0 || getKL2NormEps().convertToDouble() < 0.0) {
        return errorAt(*this, "GatedDeltaNet requires non-negative l2-norm epsilons, got q={0} k={1}",
                       getQL2NormEps().convertToDouble(), getKL2NormEps().convertToDouble());
    }

    return mlir::success();
}
