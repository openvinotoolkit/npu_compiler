//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/range.hpp"

#include <cstdint>

namespace vpux {
namespace IE {

namespace {
template <typename Transform>
mlir::LogicalResult applyTransformationInplace(mlir::MLIRContext* ctx, FqData& data, Const::Content&& transform,
                                               Transform transformCb, const Logger& log) {
    auto& [inLow, inHigh] = data;

    // must hold by definition and uniformity of transformations
    VPUX_THROW_UNLESS(inLow.getType() == inHigh.getType(), "FQ's input low and input high types differ: {0} vs {1}",
                      inLow.getType(), inHigh.getType());
    if (mlir::failed(vpux::IE::broadcastAlignShapes(ctx, inLow, transform, log))) {
        log.trace("Didn't manage to broadcast const content attributes");
        return mlir::failure();
    }
    // Note: technically, if first succeeded, the second must also succeed.
    if (mlir::failed(vpux::IE::broadcastAlignShapes(ctx, inHigh, transform, log))) {
        log.trace("Didn't manage to broadcast const content attributes");
        return mlir::failure();
    }

    // must hold by construction and type-preserving transformations
    VPUX_THROW_UNLESS(inLow.getStorageElemType().isF32() && inLow.getStorageElemType() == inHigh.getStorageElemType(),
                      "Unexpected storage element type: {0}", inLow.getStorageElemType());

    const auto inLowValues = to_small_vector(inLow.getValues<float>());
    const auto inHighValues = to_small_vector(inHigh.getValues<float>());
    const auto transformValues = to_small_vector(transform.getValues<float>());

    const auto commonType = inLow.getType();
    const bool commonSplat = inLow.isSplat() && transform.isSplat();
    auto outLowContent = Const::Content::allocTempBuffer(commonType, inLow.getStorageElemType(), commonSplat);
    auto outHighContent = Const::Content::allocTempBuffer(commonType, inLow.getStorageElemType(), commonSplat);

    // Apply transformation
    auto outLowValues = outLowContent.getTempBuf<float>();
    auto outHighValues = outHighContent.getTempBuf<float>();
    // E#131318: it is not clear whether this has to be run in parallel or if
    // sequential computation is enough.
    loop_1d(LoopExecPolicy::Parallel, ctx, outLowValues.size(), [&](size_t i) {
        outLowValues[i] = transformCb(inLowValues[i], transformValues[i]);
        outHighValues[i] = transformCb(inHighValues[i], transformValues[i]);
    });

    data.low = Const::Content::moveBuffer(commonType, std::move(outLowContent));
    data.high = Const::Content::moveBuffer(commonType, std::move(outHighContent));
    return mlir::success();
}

Const::Content splatToContent(mlir::MLIRContext* ctx, vpux::NDTypeInterface inType, float splat) {
    // Note: unfortunately, allocTempBuffer() would always allocate here even
    // for 1 element! ideally, it would be able to store a single splat value
    // without any allocation whatsoever.
    auto content = Const::Content::allocTempBuffer(mlir::cast<mlir::RankedTensorType>(inType),
                                                   mlir::Float32Type::get(ctx), true);
    content.getTempBuf<float>()[0] = splat;
    return content;
}
}  // namespace

mlir::FailureOr<FqData> applyScaleShift(mlir::MLIRContext* ctx, const Const::ContentAttr& scale,
                                        const Const::ContentAttr& shift, float low, float high,
                                        vpux::NDTypeInterface storageType, const Logger& log) {
    // Applies X * (1/scale) + shift to the given low and high values
    FqData data{splatToContent(ctx, storageType, low), splatToContent(ctx, storageType, high)};

    // Apply scale (if given)
    if (scale != nullptr) {
        if (mlir::failed(applyTransformationInplace(ctx, data, scale.fold(), std::divides<float>(), log))) {
            return mlir::failure();
        }
    }

    // Apply shift (if given)
    if (shift != nullptr) {
        if (mlir::failed(applyTransformationInplace(ctx, data, shift.fold(), std::plus<float>(), log))) {
            return mlir::failure();
        }
    }

    return data;
}

mlir::FailureOr<FqData> revertScaleShift(mlir::MLIRContext* ctx, const Const::ContentAttr& scale,
                                         const Const::ContentAttr& shift, float low, float high,
                                         vpux::NDTypeInterface storageType, const Logger& log) {
    // Applies (X - shift) * scale to the given low and high tensors
    FqData data{splatToContent(ctx, storageType, low), splatToContent(ctx, storageType, high)};

    // Apply shift (if given)
    if (shift != nullptr) {
        if (mlir::failed(applyTransformationInplace(ctx, data, shift.fold(), std::minus<float>(), log))) {
            return mlir::failure();
        }
    }

    // Apply scale (if given)
    if (scale != nullptr) {
        if (mlir::failed(applyTransformationInplace(ctx, data, scale.fold(), std::multiplies<float>(), log))) {
            return mlir::failure();
        }
    }

    return data;
}

bool isWeightOfUserOp(mlir::Operation* curOp, mlir::Operation* userOp) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MatMulOp, IE::FullyConnectedOp>(userOp)) {
        return false;
    }

    if (curOp->getResult(0) != userOp->getOperand(1)) {
        return false;
    }

    return true;
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::MultiplyOp& multiplyOp) {
    opChain.push_back(multiplyOp.getOperation());

    // Retrieve scale
    auto scaleCst = multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (scaleCst == nullptr) {
        auto scaleBlockArg = multiplyOp.getInput2();
        if (!mlir::isa<mlir::BlockArgument>(scaleBlockArg)) {
            log.trace("Match failed: Got non-const and non-blockArgument scale");
            return mlir::failure();
        }
        log.trace("Got blockArgument scale");
        dynamicScale = scaleBlockArg;
        return mlir::success();
    }
    scale = scaleCst.getContentAttr();

    // We don't need to check following ops for constant input because the new created FakeQuantize will be folded to
    // constant even if it can't be folded to users
    if (inputIsConst) {
        return mlir::success();
    }

    // Check following ops
    if (!multiplyOp->hasOneUse()) {
        // We decided to only treat the single-use case for now
        log.trace("Match failed: Got MultiplyOp with 0 or multiple users");
        return mlir::failure();
    }
    auto opUser = multiplyOp->user_begin();

    if (isWeightOfUserOp(multiplyOp, *opUser)) {
        return mlir::success();
    }

    if (auto affineReshape = mlir::dyn_cast<IE::AffineReshapeOp>(*opUser)) {
        return this->initializeStructure(affineReshape);
    }
    if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(*opUser)) {
        return this->initializeStructure(convertOp);
    }
    if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*opUser)) {
        return this->initializeStructure(transposeOp);
    }

    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::SubtractOp& subtractOp) {
    opChain.push_back(subtractOp.getOperation());

    // Retrieve shift
    auto shiftCst = subtractOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (shiftCst == nullptr) {
        auto zpBlockArg = subtractOp.getInput2();
        if (!mlir::isa<mlir::BlockArgument>(zpBlockArg)) {
            log.trace("Match failed: Got non-const and non-blockArgument zp");
            return mlir::failure();
        }
        log.trace("Got blockArgument zp");
        dynamicShift = zpBlockArg;
        return mlir::success();
    }

    log.trace("Got Const zp");
    shift = shiftCst.getContentAttr();

    // Check following ops
    const auto opUser = subtractOp->user_begin();
    if (opUser == subtractOp->user_end()) {
        return mlir::failure();
    }

    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    if (isWeightOfUserOp(subtractOp.getOperation(), *opUser)) {
        return mlir::success();
    }

    // We don't need to check following ops for constant input because the new created FakeQuantize will be folded to
    // constant even if it can't be folded to users
    if (inputIsConst) {
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::ConvertOp& convertOp) {
    opChain.push_back(convertOp.getOperation());

    // Check following ops
    if (!convertOp->hasOneUse()) {
        // We decided to only treat the single-use case for now
        log.trace("Match failed: Got ConvertOp with 0 or multiple users");
        return mlir::failure();
    }
    auto opUser = convertOp->user_begin();

    // Retrieve non-const input properties
    const auto inputBlock = mlir::dyn_cast_or_null<mlir::BlockArgument>(convertOp.getInput());
    if (inputBlock != nullptr) {
        log.trace("Got block argument input: {0}", inputBlock);
        inputValue = convertOp.getOutput();

        if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*opUser)) {
            inputValue = transposeOp.getOutput();
            return this->initializeStructure(transposeOp);
        }
    }

    if (isWeightOfUserOp(convertOp.getOperation(), *opUser)) {
        log.trace("Match succeed: {0} is the weight of {1}, match succeed", convertOp->getLoc(), (*opUser)->getLoc());
        return mlir::success();
    }

    // Prevent rematching already processed ConvertOps (they aren't deleted by the WDtoFQ pass)
    if (auto fakeQuantOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*opUser)) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }

    if (auto affineReshape = mlir::dyn_cast<IE::AffineReshapeOp>(*opUser)) {
        return this->initializeStructure(affineReshape);
    }
    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(*opUser)) {
        return this->initializeStructure(subtractOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    log.trace("Match failed: ConvertOp with no following AffineReshapeOp or SubractOp or MultiplyOp, match failed");
    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::AffineReshapeOp& affineReshapeOp) {
    opChain.push_back(affineReshapeOp.getOperation());

    // Check following ops
    if (!affineReshapeOp->hasOneUse()) {
        // We decided to only treat the single-use case for now
        log.trace("Match failed: Got affineReshapeOp with 0 or multiple users");
        return mlir::failure();
    }
    auto opUser = affineReshapeOp->user_begin();

    if (isWeightOfUserOp(affineReshapeOp.getOperation(), *opUser)) {
        log.trace("Match succeed: {0} is the weight of {1}, match succeed", affineReshapeOp->getLoc(),
                  (*opUser)->getLoc());
        return mlir::success();
    }

    // Prevent rematching already processed AffineReshapeOps (they aren't deleted by the WDtoFQ pass)
    if (auto fakeQuantOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*opUser)) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }

    if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(*opUser)) {
        return this->initializeStructure(convertOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }
    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(*opUser)) {
        return this->initializeStructure(subtractOp);
    }
    if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*opUser)) {
        return this->initializeStructure(transposeOp);
    }

    log.trace("Match failed: affineReshapeOp with no following ConvertOp or SubractOp or MultiplyOp or TransposeOp, "
              "match failed");
    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::TransposeOp& transposeOp) {
    opChain.push_back(transposeOp.getOperation());

    // Check following ops
    if (!transposeOp->hasOneUse()) {
        // We decided to only treat the single-use case for now
        log.trace("Match failed: Got transposeOp with 0 or multiple users");
        return mlir::failure();
    }
    auto opUser = transposeOp->user_begin();

    if (isWeightOfUserOp(transposeOp.getOperation(), *opUser)) {
        log.trace("Match succeed: {0} is the weight of {1}, match succeed", transposeOp->getLoc(), (*opUser)->getLoc());
        return mlir::success();
    }

    // Prevent rematching already processed TransposeOps (they aren't deleted by the WDtoFQ pass)
    if (auto fakeQuantOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*opUser)) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }

    if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*opUser)) {
        return this->initializeStructure(affineReshapeOp);
    }
    if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(*opUser)) {
        return this->initializeStructure(convertOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }
    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(*opUser)) {
        return this->initializeStructure(subtractOp);
    }

    log.trace("Match failed: transposeOp with no following AffineReshapeOp or ConvertOp or MultiplyOp or SubtractOp, "
              "match failed");
    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(Const::DeclareOp& declareOp) {
    opChain.push_back(declareOp.getOperation());

    const auto& inputAttr = declareOp.getContentAttr();
    inputValue = declareOp.getOutput();

    const auto baseContentElemType = inputAttr.getBaseContent().getShapedType().getElementType();

    // Note: reject non-floating-point inputs as the semantics of the
    // transformation expects weights of FP type. in case of explicit Convert,
    // expect it to be fused into the constant first, afterwards the output type
    // of the declare op would be floating-point.
    if (!mlir::isa<mlir::FloatType>(inputAttr.getType().getElementType())) {
        log.trace("Match failed: non-float DeclareOp is not suitable for FQ");
        return mlir::failure();
    }

    // Check following ops
    const auto users = declareOp->getUsers();
    if (users.empty()) {
        return mlir::failure();
    }

    const auto nonFqOp = llvm::find_if_not(users, [](const mlir::OpOperand& use) {
        return mlir::isa<IE::FakeQuantizeOp>(use.getOwner());
    });
    // Prevent matching ops that were already quantized
    if (nonFqOp == users.end()) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }

    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(*nonFqOp)) {
        return this->initializeStructure(subtractOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*nonFqOp)) {
        return this->initializeStructure(multiplyOp);
    }

    if (baseContentElemType == inputAttr.getType().getElementType()) {
        // A DeclareOp is still considered a WD if it has at least one CastElemType transformation
        // The WDtoFQ pass must remove CastElemType from processed constants, otherwise this case results in an
        // infinite loop
        log.trace("Match failed: DeclareOp without conversions, shifting or scaling");
        return mlir::failure();
    }

    return mlir::success();
}

WeightsDequantizeStructureInfo::WeightsDequantizeStructureInfo(bool constInput, const Logger& log)
        : inputIsConst(constInput), log(log) {
}

mlir::FailureOr<WeightsDequantizeStructureInfo> WeightsDequantizeStructureInfo::create(Const::DeclareOp origOp,
                                                                                       const Logger& log) {
    WeightsDequantizeStructureInfo info(true, log);
    const auto status = info.initializeStructure(origOp);
    if (mlir::succeeded(status)) {
        return info;
    }
    return mlir::failure();
}

mlir::FailureOr<WeightsDequantizeStructureInfo> WeightsDequantizeStructureInfo::create(IE::ConvertOp origOp,
                                                                                       const Logger& log) {
    WeightsDequantizeStructureInfo info(false, log);
    const auto status = info.initializeStructure(origOp);
    if (mlir::succeeded(status)) {
        return info;
    }
    return mlir::failure();
}

mlir::Operation* WeightsDequantizeStructureInfo::getLastOp() const {
    VPUX_THROW_UNLESS(opChain.size() >= 1, "WD info is not initialized");
    // traverse bottom-up to find the last Multiply op
    for (auto first = opChain.rbegin(), last = opChain.rend(); first != last; ++first) {
        auto op = *first;
        if (mlir::isa<IE::MultiplyOp>(op)) {
            return op;
        }
    }

    // For the pattern of weights as input with dynamic scale, etraverse bottom-up to find the last Multiply op
    return opChain.back();
}

mlir::Value WeightsDequantizeStructureInfo::getInput() const {
    return inputValue;
}

void WeightsDequantizeStructureInfo::cleanUpCurrentWdChain(mlir::PatternRewriter& rewriter) const {
    // traverse bottom-up to remove as many operations as possible
    for (auto first = opChain.rbegin(), last = opChain.rend(); first != last; ++first) {
        auto op = *first;
        if (bool operationIsStillUsed = !op->getUsers().empty(); operationIsStillUsed) {
            break;
        }
        rewriter.eraseOp(op);
    }
}

NDTypeInterface WeightsDequantizeStructureInfo::getInputType() const {
    return mlir::cast<NDTypeInterface>(inputValue.getType());
}

mlir::Type getTrueElemTypeOfWeights(Const::DeclareOp op) {
    return mlir::cast<NDTypeInterface>(op.getContentAttr().getBaseContent().getType()).getElementType();
}
mlir::Type getTrueElemTypeOfWeights(IE::ConvertOp op) {
    return mlir::cast<NDTypeInterface>(op.getInput().getType()).getElementType();
}

int64_t getQuantizationLevels(mlir::Type inputElemType) {
    // Note: universally use fixed quantization levels. For activations, we
    // cannot know real values, so it's impossible to adjust this anyhow. For
    // weights, we do not need to know real values, because it does not affect
    // accuracy (or, should not, at least).
    if (inputElemType.isInteger(4) || mlir::isa<vpux::type::NF4Type>(inputElemType)) {
        return 16;
    }
    if (inputElemType.isInteger(8)) {
        return 256;
    }

    VPUX_THROW("Got unsupported type when trying to compute levels: {0}", inputElemType);
}

std::pair<mlir::Value, mlir::Value> WeightsDequantizeStructureInfo::getInputQuantizationInterval(
        mlir::OpBuilder& builder, mlir::Location loc, float low, float high) const {
    const auto inType = getInputType();
    const auto inStorageType =
            mlir::RankedTensorType::get(SmallVector<int64_t>(inType.getRank(), 1), inType.getElementType());
    // Note: it might be better to do optional CastElemType<f16> instead of
    // using createFloatConst.
    return {Const::createFloatConst(builder, loc, inStorageType, ArrayRef(low)),
            Const::createFloatConst(builder, loc, inStorageType, ArrayRef(high))};
}

std::pair<mlir::Value, mlir::Value> WeightsDequantizeStructureInfo::getOutputQuantizationInterval(
        mlir::OpBuilder& builder, mlir::Location loc, float low, float high) const {
    const auto inType = getInputType();
    const auto inStorageType =
            mlir::RankedTensorType::get(SmallVector<int64_t>(inType.getRank(), 1), inType.getElementType());
    const auto reverted = IE::revertScaleShift(builder.getContext(), scale, shift, low, high, inStorageType, log);
    VPUX_THROW_WHEN(mlir::failed(reverted), "Failed to revert scale-shift");
    const auto& [outLow, outHigh] = reverted.value();

    // Note: shape could've changed due to scale / shift and broadcasting.
    const auto outStorageType = mlir::cast<mlir::RankedTensorType>(outLow.getType());
    const auto outLowValues = to_small_vector(outLow.getValues<float>());
    const auto outHighValues = to_small_vector(outHigh.getValues<float>());
    return {Const::createFloatConst(builder, loc, outStorageType, ArrayRef(outLowValues)),
            Const::createFloatConst(builder, loc, outStorageType, ArrayRef(outHighValues))};
}

// findAxes returns the positions of quantization axes
// For FQ in_low = in_high = out_low = out_high = 1x1x1x1 the set is empty
// For FQ in_low = in_high = out_low = out_high = 1x3x1x1 the set contains only one value = 1
// For FQ in_low = in_high = 1x1x1x1, out_low = out_high = 1x3x1x1 the set contains only one value = 1
// For FQ in_low = in_high = out_low = out_high = 1x3x1x16 the set contains positions 1 and 3
std::set<int64_t> findAxes(IE::FakeQuantizeOp origOp) {
    const auto operandShapes = SmallVector<ShapeRef>{
            getShape(origOp.getInputLow()),
            getShape(origOp.getInputHigh()),
            getShape(origOp.getOutputLow()),
            getShape(origOp.getOutputHigh()),
    };
    std::set<int64_t> axes;
    for (const auto& shape : operandShapes) {
        for (const auto& axis : irange(shape.size())) {
            if (shape[Dim(axis)] != 1) {
                axes.insert(axis);
            }
        }
    }
    return axes;
}

std::set<int64_t> findAxes(IE::DynamicDequantizeOp origOp) {
    auto operandShapes = SmallVector<ShapeRef>{getShape(origOp.getScale())};
    if (origOp.getZp() != nullptr) {
        operandShapes.push_back(getShape(origOp.getZp()));
    }
    std::set<int64_t> axes;
    for (const auto& shape : operandShapes) {
        for (const auto& axis : irange(shape.size())) {
            if (shape[Dim(axis)] != 1) {
                axes.insert(axis);
            }
        }
    }
    return axes;
}

mlir::Value WeightsDequantizeStructureInfo::getDynamicScale() const {
    return dynamicScale;
}

mlir::Value WeightsDequantizeStructureInfo::getDynamicShift() const {
    return dynamicShift;
}

Const::ContentAttr WeightsDequantizeStructureInfo::getScale() const {
    return scale;
}

Const::ContentAttr WeightsDequantizeStructureInfo::getShift() const {
    return shift;
}

}  // namespace IE
}  // namespace vpux
