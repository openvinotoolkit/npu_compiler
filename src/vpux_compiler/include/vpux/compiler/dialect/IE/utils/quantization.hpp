//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_capability.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace IE {

// Broadcasting

template <typename T>
void broadcastRange(SmallVectorImpl<T>& lowVals, SmallVectorImpl<T>& highVals, IE::AutoBroadcastType broadcast) {
    if (lowVals.size() == highVals.size()) {
        return;
    }
    if (broadcast == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        return;
    }

    const auto numpyBroadcast = [](SmallVectorImpl<T>& smaller, SmallVectorImpl<T>& larger) {
        VPUX_THROW_UNLESS(smaller.size() == 1, "One of the dimensions should be 1 for broadcasting.");
        return SmallVector<T>(larger.size(), smaller[0]);
    };

    if (broadcast == IE::AutoBroadcastType::NUMPY) {
        if (lowVals.size() < highVals.size()) {
            lowVals = numpyBroadcast(lowVals, highVals);
        } else {
            highVals = numpyBroadcast(highVals, lowVals);
        }
        return;
    }

    VPUX_THROW("Unsupported broadcast type '{0}'", broadcast);
}

//
// Derive new UniformQuantizedType. Multiply scale by specified factor.
//

mlir::Type rescaleUniformQuantizedType(const mlir::Type tensorType, const double factor);

mlir::quant::UniformQuantizedPerAxisType rescaleUniformQuantizedPerAxisType(
        const mlir::quant::UniformQuantizedPerAxisType tensorType, ArrayRef<float> factors);

void getFakeQuantParams(vpux::NDTypeInterface qType, int64_t& levels, mlir::RankedTensorType& attrType,
                        mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr);

mlir::quant::QuantizedType getQuantizedType(const Const::ContentAttr& lowConst, const Const::ContentAttr& highConst,
                                            std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                                            mlir::FloatType expressedType, bool isSigned, mlir::Location loc,
                                            IE::AutoBroadcastType broadcast = IE::AutoBroadcastType::NONE_OR_EXPLICIT,
                                            bool ignoreZPCheck = false, const Logger& log = Logger::global());

// Check whether the given FQ input and output range constants can convert to a valid signed QuantizedType.
bool canConvertToSignedQuantizedType(const Const::ContentAttr& inLowConst, const Const::ContentAttr& inHighConst,
                                     const Const::ContentAttr& outLowConst, const Const::ContentAttr& outHighConst,
                                     std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                                     IE::AutoBroadcastType broadcast = IE::AutoBroadcastType::NONE_OR_EXPLICIT,
                                     const Logger& log = Logger::global());

mlir::FailureOr<int32_t> getQuantizedDimension(ShapeRef lowShape, ShapeRef highShape, IE::AutoBroadcastType broadcast,
                                               mlir::Location loc, const Logger& log);

mlir::FailureOr<std::tuple<SmallVector<double>, SmallVector<int64_t>>> getScalesAndZeroPointsFromContentAttr(
        const Const::ContentAttr& lowContentAttr, const Const::ContentAttr& highContentAttr,
        IE::AutoBroadcastType broadcast, const std::optional<int64_t> levels, const std::optional<mlir::Type> lowFpType,
        bool isSigned, const Logger& log = Logger::global());

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
mlir::Operation* findDynDequantized(mlir::Value opInput, bool allowPerAxisQuantize);
bool isSymmetricQuantType(mlir::quant::QuantizedType type);
bool areAllQuantTypeZeroPointsEqualToZero(mlir::quant::QuantizedType type);
bool hasLeakyReLUPostOp(mlir::Operation* op);
bool hasReLUPostOp(mlir::Operation* op);
bool hasNegativeScales(mlir::quant::QuantizedType type);
mlir::quant::UniformQuantizedType getQuantizedTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp);
bool hasFQSameZeroPoint(IE::FakeQuantizeOp fqOp);

bool checkRescaledQuantApproximationForConvBasedOp(mlir::Operation* op);

mlir::Type composeWeightsExpressedType(const mlir::Type convolutionInputType);

/*
 *  Bias will be rescaled for mixed precision and written in weight table later, so need to check whether the
 *  rescaled bias range exceeds or not.
 *  The int32 overflow check is delegated to IPPECapability::getBiasStorageType: it returns i32 for
 *  NPU37XX/40XX (int32 constraint applies) and f32 for NPU50XX+ (any value is valid).
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

        const auto* ppeCapability = VPU::tryGetPPECapability(op.getOperation()->getContext());
        const bool checkInt32Range =
                !ppeCapability || mlir::isa<mlir::IntegerType>(ppeCapability->getBiasStorageType(inElemType));

        const auto OC = getShape(op.getFilter())[Dims4D::Filter::OC];
        if (mlir::failed(VPU::NCESparsity::getRescaledBias(bias, inElemType, filterElemType, OC, checkInt32Range))) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

// Parses the IR upwards looking for a possibly quantized splat constant and returns its folded dequantized value.
mlir::FailureOr<double> getQuantizedSplatConstant(mlir::Value input);
int64_t getMaximumQuantizationLevels(int64_t currentLevels, mlir::Operation* op);

bool isNCEOpCandidatesWithWeights(mlir::Operation* op);
bool keepIntTypeForSIWeightsAsInputOrConst(mlir::Operation* op);
bool isQuantizationSupported(IE::QuantizeOp quantizeOp, mlir::Operation* mainOp,
                             IE::TypeComparisonMode elemComparisonMode);
bool isInputQuantizationSupported(mlir::Value activationInput, mlir::Value filterInput);
// Checks whether activationStorageType and filterStorageType match any of the supplied {actBits, wtBits} pairs
// using the same signedness (signed, unsigned, or signless).
bool checkQuantizePattern(mlir::Type activationStorageType, mlir::Type filterStorageType,
                          llvm::ArrayRef<std::pair<unsigned, unsigned>> patterns);

bool hasIdentityQuantizationParams(mlir::Type quantizedElemType);
}  // namespace IE
}  // namespace vpux
