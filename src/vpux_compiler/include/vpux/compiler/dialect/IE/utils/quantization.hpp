//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace IE {

static constexpr float QUANT_RANGE_RATIO = 5.0;

std::optional<int64_t> getFQAxisIndex(IE::FakeQuantizeOp fq, Logger log = Logger::global());
std::optional<int64_t> getQuantAxisIndex(mlir::Operation* fq, Logger log = Logger::global());
bool areAnyUserQuantizeOps(mlir::Operation* op);
bool areAllUsersQuantized(mlir::Operation* op);
bool isPerAxisQuant(mlir::Value val);
bool checkQuantApproximation(mlir::Operation* op);
bool isPerTensorFQ(ArrayRef<IE::FakeQuantizeOp> fqOps);
bool hasStaticLowAndHighValues(IE::FakeQuantizeOp fakeQuantizeOp);
IE::FakeQuantizeOp createFQ(mlir::PatternRewriter& rewriter, mlir::Value inputOp, IE::FakeQuantizeOp fq,
                            mlir::Location loc);
Const::DeclareOp createFQConst(mlir::MLIRContext* ctx, mlir::Location loc, float val, mlir::RankedTensorType argType,
                               mlir::PatternRewriter& rewriter);
mlir::Value createFQScaling(mlir::Location loc, mlir::Value input, float scaleFactor, mlir::Type elemType,
                            std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                            vpux::IE::AutoBroadcastTypeAttr autoBroadcast, mlir::PatternRewriter& rewriter);
SmallVector<float> getConst(Const::DeclareOp declOp);
mlir::Value findQuantizedInput(mlir::Value opInput, bool allowPerAxisQuantize);
bool isSymmetricQuantType(mlir::quant::QuantizedType type);
bool hasLeakyReLUPostOp(mlir::Operation* op);
bool hasReLUPostOp(mlir::Operation* op);
bool hasNegativeScales(mlir::quant::QuantizedType type);
mlir::quant::UniformQuantizedType getQuantizedTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp);
bool hasFQSameZeroPoint(IE::FakeQuantizeOp fqOp);

bool checkRescaledQuantApproximationForConvBasedOp(mlir::Operation* op);

mlir::Type composeWeightsExpressedType(const mlir::Type convolutionInputType);

/*
 *  Bias will be rescaled for mixed precision and written in weight table later, so need to check whether the
 *  rescaled bias range exceeds or not
 */
template <class ConcreteOp>
mlir::LogicalResult checkRescaledBiasRange(ConcreteOp op) {
    auto inputDequantizeOp = op.getInput().template getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = op.getFilter().template getDefiningOp<IE::DequantizeOp>();
    if (!inputDequantizeOp || !filterDequantizeOp) {
        return mlir::failure();
    }

    if (auto biasAttr = op.getBias()) {
        const auto inElemType =
                mlir::cast<vpux::NDTypeInterface>(inputDequantizeOp.getInput().getType()).getElementType();
        const auto filterElemType =
                mlir::cast<vpux::NDTypeInterface>(filterDequantizeOp.getInput().getType()).getElementType();

        Const::ContentAttr bias;
        if (auto biasConstOp = biasAttr.template getDefiningOp<Const::DeclareOp>()) {
            bias = biasConstOp.getContentAttr();
        } else {
            auto biasDequantOp = biasAttr.template getDefiningOp<IE::DequantizeOp>();
            if (!biasDequantOp) {
                return mlir::failure();
            }
            if (auto inputConst = biasDequantOp.getInput().template getDefiningOp<Const::DeclareOp>()) {
                bias = inputConst.transformContentAttr().dequantize().get();
            } else {
                return mlir::failure();
            }
        }
        const auto OC = getShape(op.getFilter())[Dims4D::Filter::OC];
        if (mlir::failed(VPU::NCESparsity::getRescaledBias(bias, inElemType, filterElemType, OC))) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

// Parses the IR upwards looking for a possibly quantized splat constant and returns its folded dequantized value.
mlir::FailureOr<double> getQuantizedSplatConstant(mlir::Value input);

bool isNCEOpCandidatesWithWeights(mlir::Operation* op);
bool keepIntTypeForSIWeightsAsInput(mlir::Operation* op);
}  // namespace IE
}  // namespace vpux
