//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux {
namespace IE {

//
// applyScaleShift / revertScaleShift
//

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

//
// WeightsDequantizeStructureInfo
//
namespace {

// Checks if the given operation leads to the weights input of a "weighted" operation, through a sequence of ViewLike
// and Transpose ops.
bool isOnWeightsAsInputPath(mlir::Operation* op, mlir::Type lowPrecisionType, const Logger& log) {
    // If low precision type is 16 bits integer type then consider the pattern is weights dequantize.
    // Without this a Convert from U16 to FP16 can be propagated until the end which can produce invalid numbers because
    // the U16 (0:65535) range is not fully contained by FP16 (-65504.0:+65504.0) range
    if (lowPrecisionType != nullptr && lowPrecisionType.isInteger(16)) {
        return true;
    }
    static constexpr uint8_t MAX_CHAIN_LENGTH = 8;  // Safeguard, increase if the throw occurs.
    for (uint8_t i = 0; i < MAX_CHAIN_LENGTH; ++i) {
        if (!op->hasOneUse()) {
            log.trace("Match failed: Got {0} with 0 or multiple users", op->getName());
            return false;
        }

        auto opUser = *op->user_begin();
        if (mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MatMulOp, IE::FullyConnectedOp>(opUser)) {
            if (op->getResult(0) != opUser->getOperand(1)) {
                log.trace("Match failed: Pattern is not on the weights path");
                return false;
            }
            return true;
        }

        if (mlir::isa<mlir::func::ReturnOp>(opUser)) {
            // Allows functional tests to validate isolated DynamicDequantize layers.
            return true;
        }

        if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp, IE::TransposeOp, IE::ConvertOp>(opUser)) {
            log.trace("Match failed: Got invalid intermediate op: {0}", opUser->getName());
            return false;
        }
        op = opUser;
    }

    VPUX_THROW("Weights dequantize op-chain exceeded the pattern match limit: {0}", MAX_CHAIN_LENGTH);
    return false;
}

}  // namespace

mlir::LogicalResult WeightsDequantizeStructureInfo::checkAndSet(mlir::Value& out, mlir::Value value,
                                                                bool allowConstant) const {
    // Checks if the given Subtract/Multiply input `value` is a valid shift/scale. On success `out` is set to the
    // `mlir::Value` of the shift/scale.
    auto prevOp = opChain.back();

    if (mlir::isa<mlir::BlockArgument>(value)) {
        out = value;
        return mlir::success();
    }

    const auto definingOp = value.getDefiningOp();
    if (definingOp == prevOp) {
        return mlir::failure();
    }

    if (mlir::isa<Const::DeclareOp>(definingOp)) {
        if (!allowConstant) {
            return mlir::failure();
        }
        out = value;
        return mlir::success();
    }

    if (auto convert = mlir::dyn_cast<IE::ConvertOp>(definingOp)) {
        const auto convertInput = convert.getInput();
        if (mlir::isa<mlir::BlockArgument>(convertInput)) {
            out = convertInput;
            return mlir::success();
        }
    }

    if (auto stridedSlice = mlir::dyn_cast<IE::StridedSliceOp>(definingOp)) {
        const auto stridedSliceInput = stridedSlice.getInput();
        if (mlir::isa<mlir::BlockArgument>(stridedSliceInput)) {
            out = stridedSlice.getResult();
            return mlir::success();
        }
    }

    if (auto gather = mlir::dyn_cast<IE::GatherOp>(definingOp)) {
        const auto gatherInput = gather.getInput();
        if (mlir::isa<mlir::BlockArgument>(gatherInput)) {
            out = gather.getResult();
            return mlir::success();
        }
    }

    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::MultiplyOp& multiplyOp) {
    // Retrieve scale
    if (mlir::failed(checkAndSet(scale, multiplyOp.getInput2(), /*allowConstant=*/true)) &&
        mlir::failed(checkAndSet(scale, multiplyOp.getInput1(), /*allowConstant=*/false))) {
        log.trace("Match failed: Failed to retrieve scale from {0}", multiplyOp->getName());
        return mlir::failure();
    }

    // To avoid unwanted matching on WAI cases, we must check if the pattern is on the weights path.
    if (!hasConstWeights() && !isOnWeightsAsInputPath(multiplyOp, lowPrecisionType, log)) {
        return mlir::failure();
    }

    opChain.push_back(multiplyOp.getOperation());
    return mlir::success();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::SubtractOp& subtractOp) {
    // Retrieve shift. Default shift to the second operand of Subtract to avoid confusion with weights
    if (mlir::failed(checkAndSet(shift, subtractOp.getInput2(), /*allowConstant=*/true))) {
        log.trace("Match failed: Failed to retrieve shift from {0}", subtractOp->getName());
        return mlir::failure();
    }

    // Check following ops
    if (subtractOp->user_begin() == subtractOp->user_end()) {
        return mlir::failure();
    }
    const auto opUser = *subtractOp->user_begin();

    opChain.push_back(subtractOp.getOperation());

    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    // To avoid unwanted matching on WAI cases, we must check if the pattern is on the weights path.
    return hasConstWeights() || isOnWeightsAsInputPath(subtractOp, lowPrecisionType, log) ? mlir::success()
                                                                                          : mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::ConvertOp& convertOp) {
    opChain.push_back(convertOp.getOperation());

    // Check following ops
    if (!convertOp->hasOneUse()) {
        // We decided to only treat the single-use case for now
        log.trace("Match failed: Got {0} with 0 or multiple users", convertOp->getName());
        return mlir::failure();
    }
    auto opUser = *convertOp->user_begin();

    inputValue = convertOp.getOutput();
    if (const auto inputBlock = mlir::dyn_cast<mlir::BlockArgument>(convertOp.getInput())) {
        log.trace("Got block argument input: {0}", inputBlock);

        lowPrecisionType = IE::getTrueElemType(convertOp);
        // There could be a Transpose after the Convert:
        // WAI -> Convert -> Transpose -> Subtract/Multiply -> ...
        if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(opUser)) {
            inputValue = transposeOp.getOutput();
            opChain.push_back(transposeOp.getOperation());

            if (!transposeOp->hasOneUse()) {
                log.trace("Match failed: Got {0} with 0 or multiple users", transposeOp->getName());
                return mlir::failure();
            }
            opUser = *transposeOp->user_begin();
        }
    }

    // Prevent rematching already processed ConvertOps (they aren't deleted by the WDtoFQ pass)
    if (mlir::isa<IE::FakeQuantizeOp>(opUser)) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }

    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(opUser)) {
        return this->initializeStructure(subtractOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    log.trace("Match failed: ConvertOp with no following AffineReshapeOp or SubractOp or MultiplyOp, match failed");
    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(Const::DeclareOp& declareOp) {
    opChain.push_back(declareOp.getOperation());

    const auto& inputAttr = declareOp.getContentAttr();
    inputValue = declareOp.getOutput();
    lowPrecisionType = IE::getTrueElemType(declareOp);
    if (lowPrecisionType.isInteger(16)) {
        log.trace("Match failed: 16 bits weights as constant is not suitable for FQ");
        return mlir::failure();
    }

    const auto castedElemType = inputAttr.getType().getElementType();
    if (!mlir::isa<mlir::FloatType>(castedElemType)) {
        // Note: reject non-floating-point inputs as the semantics of the
        // transformation expects weights of FP type. in case of explicit Convert,
        // expect it to be fused into the constant first, afterwards the output type
        // of the declare op would be floating-point.
        log.trace("Match failed: non-float DeclareOp is not suitable for FQ");
        return mlir::failure();
    }

    const auto trueElemType = getTrueElemType(declareOp);
    if (trueElemType == castedElemType) {
        // WD structure must contain at least one CastElemType transformation (converting low-precision weights to
        // high-precision).
        log.trace("Match failed: Got {0} without quantization type casting", declareOp->getName());
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

    return mlir::success();
}

WeightsDequantizeStructureInfo::WeightsDequantizeStructureInfo(const Logger& log): log(log) {
}

mlir::FailureOr<WeightsDequantizeStructureInfo> WeightsDequantizeStructureInfo::create(Const::DeclareOp origOp,
                                                                                       const Logger& log) {
    WeightsDequantizeStructureInfo info(log);
    const auto status = info.initializeStructure(origOp);
    if (mlir::succeeded(status)) {
        return info;
    }
    return mlir::failure();
}

mlir::FailureOr<WeightsDequantizeStructureInfo> WeightsDequantizeStructureInfo::create(IE::ConvertOp origOp,
                                                                                       const Logger& log) {
    WeightsDequantizeStructureInfo info(log);
    const auto status = info.initializeStructure(origOp);
    if (mlir::succeeded(status)) {
        return info;
    }
    return mlir::failure();
}

mlir::Operation* WeightsDequantizeStructureInfo::getLastOp() const {
    VPUX_THROW_UNLESS(opChain.size() >= 1, "WD info is not initialized");
    return opChain.back();
}

SmallVector<mlir::Operation*> WeightsDequantizeStructureInfo::getOpChain() const {
    return opChain;
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

mlir::Value getTrueInputValue(mlir::Operation* op, mlir::PatternRewriter& rewriter) {
    if (auto declareOp = mlir::dyn_cast_or_null<Const::DeclareOp>(op)) {
        auto contentAttr = declareOp.getContentAttr();
        auto transformations = contentAttr.getTransformations().vec();
        auto elemType = IE::getTrueElemType(declareOp);
        transformations.push_back(Const::CastElemTypeAttr::get(elemType));
        auto newType = mlir::cast<vpux::NDTypeInterface>(declareOp.getOutput().getType()).changeElemType(elemType);
        auto baseContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), transformations);
        return rewriter.create<Const::DeclareOp>(appendLoc(op->getLoc(), "base_shift"), newType, baseContentAttr)
                .getOutput();
    } else if (auto convertOp = mlir::dyn_cast_or_null<IE::ConvertOp>(op)) {
        // Historically, assume that convert op's input is weights. This assumption holds when
        // WeightsDequantizeStructureInfo is constructed successfully.
        return convertOp.getInput();
    } else {
        VPUX_THROW("Got unsupported op type: {0}", op->getName());
    }
}

mlir::Type getTrueElemType(mlir::Operation* op) {
    if (auto declareOp = mlir::dyn_cast_or_null<Const::DeclareOp>(op)) {
        return mlir::cast<NDTypeInterface>(declareOp.getContentAttr().getBaseContent().getType()).getElementType();
    } else if (auto convertOp = mlir::dyn_cast_or_null<IE::ConvertOp>(op)) {
        // Historically, assume that convert op's input is weights and their type is the
        // real type. This assumption holds when WeightsDequantizeStructureInfo is
        // constructed successfully.
        return mlir::cast<NDTypeInterface>(convertOp.getInput().getType()).getElementType();
    } else {
        VPUX_THROW("Got unsupported op type: {0}", op->getName());
    }
}

int64_t getQuantizationLevels(mlir::Type inputElemType) {
    // Note: universally use fixed quantization levels. For activations, we
    // cannot know real values, so it's impossible to adjust this anyhow. For
    // weights, we do not need to know real values, because it does not affect
    // accuracy (or, should not, at least).
    if (inputElemType.isInteger(2)) {
        return 4;
    }
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
    const auto reverted = IE::revertScaleShift(builder.getContext(), getStaticScaleAttr(), getStaticShiftAttr(), low,
                                               high, inStorageType, log);
    VPUX_THROW_WHEN(mlir::failed(reverted), "Failed to revert scale-shift");
    const auto& [outLow, outHigh] = reverted.value();

    // Note: shape could've changed due to scale / shift and broadcasting.
    const auto outStorageType = mlir::cast<mlir::RankedTensorType>(outLow.getType());
    const auto outLowValues = to_small_vector(outLow.getValues<float>());
    const auto outHighValues = to_small_vector(outHigh.getValues<float>());
    return {Const::createFloatConst(builder, loc, outStorageType, ArrayRef(outLowValues)),
            Const::createFloatConst(builder, loc, outStorageType, ArrayRef(outHighValues))};
}

bool WeightsDequantizeStructureInfo::hasConstWeights() const {
    return inputValue.getDefiningOp<Const::DeclareOp>() != nullptr;
}

bool WeightsDequantizeStructureInfo::hasScale() const {
    return scale != nullptr;
}

bool WeightsDequantizeStructureInfo::hasShift() const {
    return shift != nullptr;
}

bool WeightsDequantizeStructureInfo::isKVcachedPattern() const {
    auto lastOp = getLastOp();
    while (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::ConvertOp, IE::AffineReshapeOp, IE::ReshapeOp>(lastOp)) {
        if (!lastOp->getResult(0).hasOneUse()) {
            return false;
        }

        lastOp = *lastOp->user_begin();
    }

    if (mlir::isa<IE::MatMulOp, IE::FullyConnectedOp>(lastOp)) {
        const auto countNonTrivialDims = [](const auto& shape) {
            return std::count_if(shape.begin(), shape.end(), [](const auto& d) {
                return d != 1;
            });
        };
        return countNonTrivialDims(getShape(lastOp->getOperand(0))) == 1;
    }

    if (mlir::isa<IE::ConvolutionOp>(lastOp)) {
        return getShape(lastOp->getOperand(0))[Dim(0)] == 1;
    }

    return false;
}

mlir::Type WeightsDequantizeStructureInfo::getInputElemType() const {
    return getInputType().getElementType();
}

mlir::Type WeightsDequantizeStructureInfo::getLowPrecisionElemType() const {
    return lowPrecisionType;
}

Const::ContentAttr WeightsDequantizeStructureInfo::getStaticScaleAttr() const {
    if (scale == nullptr) {
        return {};
    }
    auto declareOp = scale.getDefiningOp<Const::DeclareOp>();
    return declareOp != nullptr ? declareOp.getContentAttr() : Const::ContentAttr{};
}

Const::ContentAttr WeightsDequantizeStructureInfo::getStaticShiftAttr() const {
    if (shift == nullptr) {
        return {};
    }
    auto declareOp = shift.getDefiningOp<Const::DeclareOp>();
    return declareOp != nullptr ? declareOp.getContentAttr() : Const::ContentAttr{};
}

mlir::Value WeightsDequantizeStructureInfo::getStaticScale() const {
    return scale == nullptr || scale.getDefiningOp<Const::DeclareOp>() == nullptr ? nullptr : scale;
}

mlir::Value WeightsDequantizeStructureInfo::getStaticShift() const {
    return shift == nullptr || shift.getDefiningOp<Const::DeclareOp>() == nullptr ? nullptr : shift;
}

mlir::Value WeightsDequantizeStructureInfo::getDynamicScale() const {
    return scale == nullptr || scale.getDefiningOp<Const::DeclareOp>() != nullptr ? nullptr : scale;
}

mlir::Value WeightsDequantizeStructureInfo::getDynamicShift() const {
    return shift == nullptr || shift.getDefiningOp<Const::DeclareOp>() != nullptr ? nullptr : shift;
}

NDTypeInterface WeightsDequantizeStructureInfo::getScaleType() const {
    return scale != nullptr ? mlir::cast<vpux::NDTypeInterface>(scale.getType()) : nullptr;
}

NDTypeInterface WeightsDequantizeStructureInfo::getShiftType() const {
    return shift != nullptr ? mlir::cast<vpux::NDTypeInterface>(shift.getType()) : nullptr;
}

int64_t WeightsDequantizeStructureInfo::getQuantizedAxisCount() const {
    const auto countNonTrivialDims = [](const auto& shape) {
        return std::count_if(shape.begin(), shape.end(), [](const auto& d) {
            return d != 1;
        });
    };

    const auto scaleType = getScaleType();
    const auto shiftType = getShiftType();
    if (scaleType != nullptr) {
        const auto scaleShape = scaleType.getShape();
        if (shiftType == nullptr) {
            return countNonTrivialDims(scaleShape.raw());
        }

        const auto shiftShape = shiftType.getShape();
        const auto bcastOrFail = IE::broadcastEltwiseShape({scaleShape.raw(), shiftShape.raw()},
                                                           AutoBroadcastType::NUMPY, getLastOp()->getLoc());
        VPUX_THROW_WHEN(mlir::failed(bcastOrFail), "Failed to broadcast scale: {0} and shift: {1} shapes.", scaleShape,
                        shiftShape);

        return countNonTrivialDims(*bcastOrFail);
    }
    if (shiftType != nullptr) {
        const auto shiftShape = shiftType.getShape();
        return countNonTrivialDims(shiftShape.raw());
    }
    return 0;
}

//
// findAxes
//

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

template <>
type::QuantileFloatType tryParsingNF4(Const::DeclareOp constOp) {
    // Note: NF4 is special: the raw data is 4-bit int, but - due to quantiles -
    // its range is not standard quantization range - it is instead deduced from
    // quantiles.
    const auto& contentAttr = constOp.getContentAttr();
    const bool baseTypeIsInt4 = mlir::isa<mlir::IntegerType>(contentAttr.getBaseContent().getElementType()) &&
                                contentAttr.getBaseContent().getElementType().getIntOrFloatBitWidth() == 4;
    if (!baseTypeIsInt4) {
        return nullptr;
    }

    for (const auto& transform : contentAttr.getTransformations()) {
        // if there is a cast to NF4, it means the constant is NF4.
        if (auto cast = mlir::dyn_cast<Const::CastElemTypeAttr>(transform);
            cast && mlir::isa<type::QuantileFloatType>(cast.getElemType())) {
            return mlir::cast<type::QuantileFloatType>(cast.getElemType());
        }
    }
    return nullptr;
}

bool WeightsDequantizeStructureInfo::isI4ConsumedByGather() const {
    auto* lastOp = getLastOp();
    if (!lastOp->getResult(0).hasOneUse()) {
        return false;
    }
    if (!mlir::isa<IE::GatherOp>(*lastOp->getResult(0).user_begin())) {
        return false;
    }
    auto elemType = getLowPrecisionElemType();
    return elemType.isInteger(4);
}

}  // namespace IE
}  // namespace vpux
