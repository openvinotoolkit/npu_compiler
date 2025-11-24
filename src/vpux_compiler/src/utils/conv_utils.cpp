//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

using namespace vpux;

mlir::Value vpux::buildDwWeights(const mlir::Location& loc, int64_t OC, const mlir::Type& elementType,
                                 mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    if (auto quantizeType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
        const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vpux::type::float16(1.f)));
        const auto contentAttr = Const::ContentAttr::get(baseAttr);
        auto quantWeightsConstAttr = contentAttr.transform().castElemType(quantizeType).get();
        const auto weightsType = mlir::cast<vpux::NDTypeInterface>(contentAttr.getType()).changeElemType(quantizeType);
        return rewriter.create<Const::DeclareOp>(loc, weightsType, std::move(quantWeightsConstAttr));
    }
    if (elementType.isF16()) {
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
        return Const::createConst(rewriter, loc, baseType, ArrayRef(vpux::type::float16(1.f)));
    }
    if (elementType.isF32()) {
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float32Type::get(ctx));
        return Const::createConst(rewriter, loc, baseType, ArrayRef(1.f));
    }
    VPUX_THROW("buildDwWeights: other types are not supported");
}
