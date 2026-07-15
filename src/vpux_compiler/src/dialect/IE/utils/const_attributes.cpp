//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"

namespace vpux {
namespace IE {

mlir::ArrayAttr getIntArrayAttrValue(mlir::Value operand) {
    if (operand == nullptr) {
        return nullptr;
    }
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return nullptr;
    }
    const auto content = constOp.getContent();
    return getIntArrayAttr(operand.getContext(), content.getValues<int32_t>());
}

mlir::ArrayAttr getFloatArrayAttrValue(mlir::Value operand) {
    if (operand == nullptr) {
        return nullptr;
    }
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return nullptr;
    }
    const auto content = constOp.getContent();
    return getFPArrayAttr(operand.getContext(), content.getValues<double>());
}

mlir::IntegerAttr getIntAttrValue(mlir::Value operand, mlir::PatternRewriter& rewriter) {
    if (operand == nullptr) {
        return nullptr;
    }
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    if (const auto& attr = constOp.getContentAttr(); !attr.isSplat()) {
        return nullptr;
    }
    const auto content = constOp.getContent();
    const auto attrValue = content.getSplatValue<int32_t>();
    return rewriter.getI32IntegerAttr(attrValue);
}

mlir::FailureOr<Const::DeclareOp> getConstParentOp(mlir::Value input) {
    auto parent = input.getDefiningOp();
    while (parent && mlir::isa<IE::FakeQuantizeOp, IE::TransposeOp, IE::NegativeOp, IE::ReshapeOp, IE::BroadcastOp,
                               IE::AffineReshapeOp>(parent)) {
        parent = parent->getOperand(0).getDefiningOp();
    }
    if (parent && mlir::isa<Const::DeclareOp>(parent)) {
        return mlir::cast<Const::DeclareOp>(parent);
    }
    return mlir::failure();
}

mlir::FailureOr<int64_t> getBaseContentNumElements(Const::DeclareOp constOp) {
    if (constOp == nullptr) {
        return mlir::failure();
    }
    const auto& contentAttr = constOp.getContentAttr();
    if (contentAttr == nullptr) {
        return mlir::failure();
    }
    const auto& baseContent = contentAttr.getBaseContent();
    if (baseContent != nullptr) {
        return baseContent.getShapedType().getNumElements();
    }
    return mlir::failure();
}

bool isBaseContentSplat(Const::DeclareOp constOp) {
    if (constOp == nullptr) {
        return false;
    }

    const auto& contentAttr = constOp.getContentAttr();
    if (contentAttr == nullptr) {
        return false;
    }
    const auto& baseContent = contentAttr.getBaseContent();

    return baseContent.isSplat();
}

// Read all values from a constant DeclareOp as doubles.
SmallVector<double> readConstAsDoubles(Const::DeclareOp constOp) {
    const auto content = constOp.getContentAttr().fold();
    const auto numElements = content.getType().getNumElements();
    SmallVector<double> result;
    result.reserve(numElements);
    content.read([&](auto values) {
        for (const auto v : values) {
            result.push_back(static_cast<double>(v));
        }
    });
    return result;
}

// Read all values from a constant DeclareOp as int64_t.
SmallVector<int64_t> readConstAsInt64(Const::DeclareOp constOp) {
    const auto content = constOp.getContentAttr().fold();
    const auto numElements = content.getType().getNumElements();
    SmallVector<int64_t> result;
    result.reserve(numElements);
    content.read([&](auto values) {
        for (const auto v : values) {
            result.push_back(static_cast<int64_t>(v));
        }
    });
    return result;
}

}  // namespace IE
}  // namespace vpux
