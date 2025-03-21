//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Support/LogicalResult.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTBRANCHESCONCATTOCONV
#define GEN_PASS_DEF_CONVERTBRANCHESCONCATTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// OptimizeGroupConvConcat
//

/*
    Convert below pattern:

        Root(1x16x144x144)             Weights(16x1x3x3)
       /               \                  /
      |                GroupConv(1x16x144x144)
       \                    /
        Concat(1x32x144x144)

    to:

        Root(1x16x144x144)             Weights(32x16x3x3)
                    \                     /
                    Convolution(1x32x144x144)

    When root tensor is ReLu or layer with fused ReLu post op
    below conversion is still appliable because applying ReLu for serveral times does not change the results

       ReLu(1x16x144x144)             Weights(16x1x3x3)
       /               \                  /
      |               GroupConv + ReLu(1x16x144x144)
      |                         |
       \                       /
          Concat(1x32x144x144)

    to:

       ReLu(1x16x144x144)             Weights(32x16x3x3)
                       \                  /
                      Conv + ReLu(1x32x144x144)
*/

constexpr size_t SUPPORTED_RANK = 4;
struct GroupConvParameters {
    bool isPopulated = false;
    int64_t kx{};
    int64_t ky{};
    SmallVector<int64_t> stride;
    Shape padBegin;
    Shape padEnd;
    SmallVector<int64_t> dilations;
    bool withNonConstBias{};
    bool withPostOp{};
    bool withClamp{};

    bool operator==(const GroupConvParameters& other) const {
        return std::tie(isPopulated, kx, ky, stride, padBegin, padEnd, dilations, withNonConstBias, withPostOp,
                        withClamp) == std::tie(other.isPopulated, other.kx, other.ky, other.stride, other.padBegin,
                                               other.padEnd, other.dilations, other.withNonConstBias, other.withPostOp,
                                               other.withClamp);
    }
};

GroupConvParameters extractGroupConvParameters(IE::GroupConvolutionOp op) {
    GroupConvParameters parameters;

    auto filterShape = getShape(op.getFilter());
    parameters.kx = filterShape[Dims4D::Filter::KX];
    parameters.ky = filterShape[Dims4D::Filter::KY];

    parameters.stride = parseIntArrayAttr<int64_t>(op.getStrides());
    parameters.padBegin = Shape(parseIntArrayAttr<int64_t>(op.getPadsBegin()));
    parameters.padEnd = Shape(parseIntArrayAttr<int64_t>(op.getPadsEnd()));
    parameters.dilations = parseIntArrayAttr<int64_t>(op.getDilations());

    auto bias = op.getBias();
    parameters.withNonConstBias = bias != nullptr && !mlir::isa<Const::DeclareOp>(bias.getDefiningOp());
    parameters.withPostOp = op.getPostOpAttr() != nullptr;
    parameters.withClamp = op.getClampAttr() != nullptr;

    parameters.isPopulated = true;

    return parameters;
}

class OptimizeGroupConvConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
    struct InputPattern {
        size_t index;
        IE::GroupConvolutionOp groupConv = nullptr;

        InputPattern(size_t id, IE::GroupConvolutionOp groupConvOp): index(id), groupConv(groupConvOp) {
        }
    };

public:
    OptimizeGroupConvConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(std::move(log)) {
        setDebugName("OptimizeGroupConvConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::FailureOr<SmallVector<InputPattern>> getValidConcatInputs(IE::ConcatOp concatOp,
                                                                    GroupConvParameters& parameters) const;

    bool hasReLUPostOp(mlir::Operation* op) const;
    bool isSupportedParameters(const GroupConvParameters& parameters, mlir::Operation* rootOp) const;

    mlir::Value convertGroupConvWeights(IE::GroupConvolutionOp groupConv, mlir::PatternRewriter& rewriter) const;

    mlir::Value createWeights(mlir::Value activation, const GroupConvParameters& parameters,
                              mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

bool OptimizeGroupConvConcat::hasReLUPostOp(mlir::Operation* op) const {
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op)) {
        const auto postOpName = layerWithPostOp.getPostOp();
        return postOpName.has_value() && postOpName.value().getStringRef() == IE::ReLUOp::getOperationName();
    }
    return false;
}

bool OptimizeGroupConvConcat::isSupportedParameters(const GroupConvParameters& parameters,
                                                    mlir::Operation* rootOp) const {
    auto isGreaterThanOne = [](int64_t size) {
        return size > 1;
    };
    auto hasNonOneDilation = llvm::any_of(parameters.dilations, isGreaterThanOne);
    if (hasNonOneDilation) {
        return false;
    }

    auto hasNonOneStride = llvm::any_of(parameters.stride, isGreaterThanOne);
    if (hasNonOneStride) {
        return false;
    }

    // Odd & even kernels have different weights structures, right now only odd kernel is supported
    if (parameters.kx != parameters.ky || (parameters.kx - 1) % 2 != 0) {
        return false;
    }

    if (parameters.padBegin[Dims4D::PadsBegin::Left] != parameters.padEnd[Dims4D::PadsEnd::Right] ||
        parameters.padBegin[Dims4D::PadsBegin::Top] != parameters.padEnd[Dims4D::PadsEnd::Bottom] ||
        parameters.kx / 2 != parameters.padBegin[Dims4D::PadsBegin::Left]) {
        return false;
    }

    if (parameters.withClamp || parameters.withNonConstBias) {
        return false;
    }

    // It can be supported when root tensor is a ReLU layer or layer with ReLU postOp.
    // The new Convolution with ReLU postOp can produce the identical result, because
    // multiple cascading ReLu ops can produce the same result with single ReLu.
    if (parameters.withPostOp && !hasReLUPostOp(rootOp) && !mlir::isa<IE::ReLUOp>(rootOp)) {
        return false;
    }

    return true;
}

SmallVector<Dim> getDimsWithDifferentDimSize(ShapeRef inShape, ShapeRef outShape) {
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Got unexpected input and output shape");
    SmallVector<Dim> concatDims;
    auto ioShapes = zip(inShape, outShape);
    for (const auto& ioShape : ioShapes | indexed) {
        const auto inSize = std::get<0>(ioShape.value());
        const auto outSize = std::get<1>(ioShape.value());
        if (inSize != outSize) {
            concatDims.push_back(Dim(ioShape.index()));
        }
    }
    return concatDims;
}

mlir::FailureOr<SmallVector<OptimizeGroupConvConcat::InputPattern>> OptimizeGroupConvConcat::getValidConcatInputs(
        IE::ConcatOp concatOp, GroupConvParameters& parameters) const {
    auto inputNum = concatOp.getInputs().size();
    SmallVector<InputPattern> concatInputs;
    concatInputs.reserve(inputNum);

    const auto concatOutType = concatOp.getOutput().getType().cast<NDTypeInterface>();
    const auto concatOutShape = concatOutType.getShape();

    // Inputs of Concat should have the same root
    mlir::Operation* maybeRootOp = nullptr;
    auto hasTheSameParentOp = [&maybeRootOp](mlir::Operation* parentOp) {
        if (maybeRootOp == nullptr) {
            maybeRootOp = parentOp;
            return true;
        }

        return maybeRootOp == parentOp;
    };

    // Inputs of Concat with GroupConvolution should have the same parameters
    GroupConvParameters prevParameters;
    auto hasTheSameParameters = [&prevParameters](const GroupConvParameters& currParameters) {
        if (!prevParameters.isPopulated) {
            prevParameters = currParameters;
            return true;
        }

        return prevParameters == currParameters;
    };

    for (const auto& input : concatOp.getInputs() | indexed) {
        auto inputIndex = input.index();
        auto inputValue = input.value();

        auto concatDims = getDimsWithDifferentDimSize(getShape(inputValue), concatOutShape);
        if (concatDims.size() != 1 || concatDims.front() != Dims4D::Act::C) {
            _log.trace("Unsupported concat dimension");
            return mlir::failure();
        }

        if (inputValue.hasOneUse()) {
            auto groupConv = inputValue.getDefiningOp<IE::GroupConvolutionOp>();
            if (groupConv == nullptr) {
                return mlir::failure();
            }

            if (groupConv.getFilter().getDefiningOp<Const::DeclareOp>() == nullptr) {
                return mlir::failure();
            }

            auto ic = getShape(groupConv.getInput())[Dims4D::Act::C];
            auto groupNum = groupConv.getGroups().value();
            if (groupNum != ic) {
                _log.trace("Only support DWConv for now");
                return mlir::failure();
            }

            auto currParameters = extractGroupConvParameters(groupConv);
            auto parentOp = groupConv.getInput().getDefiningOp();
            if (!isSupportedParameters(currParameters, parentOp)) {
                return mlir::failure();
            }

            if (!hasTheSameParameters(currParameters)) {
                return mlir::failure();
            }

            if (!hasTheSameParentOp(parentOp)) {
                return mlir::failure();
            }

            concatInputs.push_back(InputPattern(inputIndex, groupConv));
        } else if (std::distance(inputValue.getUsers().begin(), inputValue.getUsers().end()) >=
                   checked_cast<int64_t>(inputNum)) {
            auto parentOp = inputValue.getDefiningOp();
            if (!hasTheSameParentOp(parentOp)) {
                return mlir::failure();
            }

            concatInputs.push_back(InputPattern(inputIndex, nullptr));
        } else {
            _log.trace("[{0}] unknown input at '{1}'", concatOp->getName(), concatOp->getLoc());
            return mlir::failure();
        }
    }

    if (!prevParameters.isPopulated) {
        _log.trace("[{0}] No GroupConv is found at concat inputs '{1}'", concatOp->getName(), concatOp->getLoc());
        return mlir::failure();
    }

    if (maybeRootOp == nullptr) {
        return mlir::failure();
    }
    parameters = std::move(prevParameters);
    return concatInputs;
}

// Create weights to convert GroupConvolution to Convolution
mlir::Value OptimizeGroupConvConcat::convertGroupConvWeights(IE::GroupConvolutionOp groupConv,
                                                             mlir::PatternRewriter& rewriter) const {
    auto origFilter = groupConv.getFilter();
    auto origFilterOp = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(origFilterOp == nullptr, "Unable to find filter constant operation");
    auto origContent = origFilterOp.getContentAttr().fold();
    auto origContentType = origContent.getType();

    const auto origFiliterShape = getShape(origFilter);
    const auto kx = origFiliterShape[Dims4D::Filter::KX];
    const auto ky = origFiliterShape[Dims4D::Filter::KY];

    const auto groupNum = groupConv.getGroups().value();
    const auto groupInSize = origFiliterShape[Dims4D::Filter::IC];
    const auto groupOutSize = origFiliterShape[Dims4D::Filter::OC] / groupNum;
    const auto groupInChannel = getShape(groupConv.getInput())[Dims4D::Act::C] / groupNum;
    const auto groupOutChannel = getShape(groupConv.getOutput())[Dims4D::Act::C] / groupNum;
    VPUX_THROW_UNLESS(groupInSize == groupInChannel && groupOutSize == groupOutChannel,
                      "groupInSize '{0}' not equal with input channel '{1}' or groupOutSize '{2}' not equal with "
                      "output channel '{3}' ",
                      groupInSize, groupInChannel, groupOutSize, groupOutChannel);

    const auto oc = getShape(groupConv.getOutput())[Dims4D::Act::C];
    const auto ic = getShape(groupConv.getInput())[Dims4D::Act::C];
    auto newFilterShape = Shape{oc, ic, ky, kx};
    std::vector<float> weights(newFilterShape.totalSize(), .0f);

    const auto inValues = origContent.getValues<float>();

    const auto weightsSizePerOC = ic * ky * kx;
    const auto weightsSizePerGroup = kx * ky * groupInSize * groupOutSize;
    for (int64_t groupIndex = 0; groupIndex < groupNum; groupIndex++) {
        auto innerOffset = weightsSizePerGroup * groupIndex;
        for (auto idx = 0; idx < weightsSizePerGroup; idx++) {
            weights[weightsSizePerOC * groupIndex + innerOffset + idx] = inValues[innerOffset + idx];
        }
    }

    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType =
            mlir::RankedTensorType::get(newFilterShape.raw(), origContentType.getElementType(),
                                        getTensorAttr(rewriter.getContext(), weightsOrder, nullptr, nullptr));

    return Const::buildWeightsConst(rewriter, origFilter.getLoc(), weightsType, ArrayRef(weights));
}

// Create weights for Convolution that can generate the same result with input
mlir::Value OptimizeGroupConvConcat::createWeights(mlir::Value activation, const GroupConvParameters& parameters,
                                                   mlir::PatternRewriter& rewriter) const {
    const auto kx = parameters.kx;
    const auto ky = parameters.ky;
    const auto actShape = getShape(activation);
    const auto ic = actShape[Dims4D::Act::C];
    const auto oc = actShape[Dims4D::Act::C];

    const Shape weightsShape = {oc, ic, ky, kx};
    std::vector<float> weights(weightsShape.totalSize(), .0f);

    // assign values
    const auto weightsSizePerOC = ic * ky * kx;
    for (int64_t ocIndex = 0; ocIndex < oc; ocIndex++) {
        auto currOffsetPerOC = ocIndex * weightsSizePerOC;
        auto innerOffset = kx * ky * ocIndex + kx * ky / 2;
        weights[currOffsetPerOC + innerOffset] = 1.0f;
    }

    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType = mlir::RankedTensorType::get(
            weightsShape.raw(), mlir::cast<NDTypeInterface>(activation.getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightsOrder, nullptr, nullptr));
    return Const::buildWeightsConst(rewriter, activation.getLoc(), weightsType, ArrayRef(weights));
}

mlir::LogicalResult OptimizeGroupConvConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    if (outputShape.size() != SUPPORTED_RANK) {
        return mlir::failure();
    }

    // Skip conversion when channel number is large for below reasons:
    // 1.Performance of DMA is extremely poor only when the number of concat channels is small and the concat dimension
    // is on the innermost dimension.
    // 2.Too large number of channels would cause large constant size of new filter.
    // 64 is an experimental value to ensure the conversion is beneficial.
    const int64_t CHANNEL_NUM_TO_PREVENT_LARGE_FILTER = 64;
    if (outputShape[Dims4D::Act::C] > CHANNEL_NUM_TO_PREVENT_LARGE_FILTER) {
        return mlir::failure();
    }

    GroupConvParameters parameters;
    auto getInputs = getValidConcatInputs(origOp, parameters);
    if (mlir::failed(getInputs)) {
        _log.trace("[{0}] Failed to get valid input pattern at '{1}'", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Rewriting '{1}'", origOp->getName(), origOp->getLoc());

    auto inputs = getInputs.value();
    mlir::Value root = nullptr;
    IE::GroupConvolutionOp origGroupConv = nullptr;
    int64_t groupConvIndex = 0;
    // Create filter
    SmallVector<mlir::Value> newWeights;
    for (const auto& input : inputs) {
        auto groupConv = input.groupConv;
        if (groupConv != nullptr) {
            origGroupConv = groupConv;
            // Find root by groupConv's input
            if (root == nullptr) {
                root = groupConv.getInput();
            }
            newWeights.push_back(convertGroupConvWeights(groupConv, rewriter));
            groupConvIndex = input.index;
        } else {
            // Find root by concat's input
            if (root == nullptr) {
                auto inputIndex = input.index;
                root = origOp.getInputs()[inputIndex];
            }
            newWeights.push_back(createWeights(root, parameters, rewriter));
        }
    }
    VPUX_THROW_WHEN(origGroupConv == nullptr, "Can't find GroupConv");
    auto concatWeights = rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), newWeights, Dims4D::Filter::OC);
    // update bias value
    mlir::Value newBiasValue = origGroupConv.getBias();
    auto biasConst = newBiasValue != nullptr ? newBiasValue.getDefiningOp<Const::DeclareOp>() : nullptr;
    if (biasConst) {
        int64_t padding = outputShape[Dims4D::Act::C] - getShape(origGroupConv.getBias()).totalSize();
        int64_t padBefore = 0;
        int64_t padEnd = 0;
        if (groupConvIndex == 0) {
            padEnd = padding;
        } else {
            padBefore = padding;
        }
        auto biasContentAttr =
                biasConst.transformContentAttr().padWithZero({0, padBefore, 0, 0}, {0, padEnd, 0, 0}).get();
        newBiasValue = rewriter.create<Const::DeclareOp>(origOp.getLoc(), biasContentAttr.getType(),
                                                         std::move(biasContentAttr))
                               .getResult();
    }

    // GroupConv has no scale parameter, so it's nullptr when creating 'IE::ConvolutionOp'
    auto newConv = rewriter.create<IE::ConvolutionOp>(
            origOp.getLoc(), root, concatWeights, newBiasValue, origGroupConv.getStrides(),
            origGroupConv.getPadsBegin(), origGroupConv.getPadsEnd(), origGroupConv.getDilations(),
            origGroupConv.getPostOpAttr(), origGroupConv.getClampAttr(), nullptr, origGroupConv.getOutputChannelsAttr(),
            origGroupConv.getInputChannelsAttr());

    rewriter.replaceOp(origOp, newConv.getOutput());
    return mlir::success();
}

//
// OptimizeConvConcat
//

class OptimizeConvConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    OptimizeConvConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(std::move(log)) {
        setDebugName("OptimizeConvConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::FailureOr<SmallVector<mlir::Value>> getValidConcatInputs(IE::ConcatOp concatOp) const;

private:
    Logger _log;
};

mlir::FailureOr<SmallVector<mlir::Value>> OptimizeConvConcat::getValidConcatInputs(IE::ConcatOp concatOp) const {
    SmallVector<mlir::Value> concatInputs;
    const auto concatOutShape = getShape(concatOp.getOutput());

    IE::ConvolutionOp firstConvOp = nullptr;
    mlir::Value firstParentInput = nullptr;
    int64_t firstKx = 0;
    int64_t firstKy = 0;
    for (const auto& inputValue : concatOp.getInputs()) {
        auto concatDims = getDimsWithDifferentDimSize(getShape(inputValue), concatOutShape);
        if (concatDims.size() != 1 || concatDims.front() != Dims4D::Act::C) {
            _log.trace("Unsupported concat dimension");
            return mlir::failure();
        }

        if (!inputValue.hasOneUse()) {
            return mlir::failure();
        }

        auto convOp = inputValue.getDefiningOp<IE::ConvolutionOp>();
        if (convOp == nullptr) {
            return mlir::failure();
        }

        auto filter = convOp.getFilter();
        if (!mlir::isa<Const::DeclareOp>(filter.getDefiningOp())) {
            return mlir::failure();
        }

        auto parentInput = convOp.getInput();

        if (firstConvOp == nullptr) {
            firstConvOp = convOp;
            firstParentInput = parentInput;
            auto firstFilterShape = getShape(firstConvOp.getFilter());
            firstKx = firstFilterShape[Dims4D::Filter::KX];
            firstKy = firstFilterShape[Dims4D::Filter::KY];
        } else {
            auto filterShape = getShape(filter);
            auto kx = filterShape[Dims4D::Filter::KX];
            auto ky = filterShape[Dims4D::Filter::KY];
            if (kx != firstKx || ky != firstKy) {
                return mlir::failure();
            }

            if (firstConvOp.getPadsBegin() != convOp.getPadsBegin() ||
                firstConvOp.getPadsEnd() != convOp.getPadsEnd() ||
                firstConvOp.getDilations() != convOp.getDilations() ||
                firstConvOp.getPostOpAttr() != convOp.getPostOpAttr() ||
                firstConvOp.getClampAttr() != convOp.getClampAttr() ||
                firstConvOp.getStaticScaleAttr() != convOp.getStaticScaleAttr() ||
                firstConvOp.getOutputChannelsAttr() != convOp.getOutputChannelsAttr() ||
                firstConvOp.getInputChannelsAttr() != convOp.getInputChannelsAttr()) {
                return mlir::failure();
            }
        }

        // Inputs of Concat should have the same parent
        if (firstParentInput != parentInput) {
            return mlir::failure();
        }

        concatInputs.push_back(inputValue);
    }

    return concatInputs;
}

mlir::LogicalResult OptimizeConvConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    if (outputShape.size() != SUPPORTED_RANK) {
        return mlir::failure();
    }

    auto getInputs = getValidConcatInputs(origOp);
    if (mlir::failed(getInputs)) {
        _log.trace("[{0}] Failed to get valid input pattern at '{1}'", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Rewriting '{1}'", origOp->getName(), origOp->getLoc());

    auto inputs = getInputs.value();
    mlir::Value root = nullptr;
    IE::ConvolutionOp convOp = nullptr;
    // Create filter
    SmallVector<mlir::Value> newWeights;
    size_t emptyBiasNum = 0;
    for (const auto& input : inputs) {
        convOp = input.getDefiningOp<IE::ConvolutionOp>();

        if (root == nullptr) {
            root = convOp.getInput();
        }

        newWeights.push_back(convOp.getFilter());

        if (convOp.getBias() == nullptr) {
            emptyBiasNum++;
        }
    }
    auto concatWeights = rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), newWeights, Dims4D::Filter::OC);

    // Create Bias
    SmallVector<mlir::Value> newBias;
    mlir::Value concatBias = nullptr;
    if (emptyBiasNum != inputs.size()) {
        for (const auto& input : inputs) {
            convOp = input.getDefiningOp<IE::ConvolutionOp>();
            if (auto bias = convOp.getBias()) {
                newBias.push_back(bias);
            } else {
                auto inputShape = getShape(input);
                auto oc = inputShape[Dims4D::Act::C];
                const Shape biasShape = {1, oc, 1, 1};
                std::vector<float> biasValue(oc, .0f);

                const DimsOrder biasOrder = DimsOrder::NCHW;
                const auto biasType = mlir::RankedTensorType::get(
                        biasShape.raw(), input.getType().cast<NDTypeInterface>().getElementType(),
                        getTensorAttr(rewriter.getContext(), biasOrder, nullptr, nullptr));

                newBias.push_back(Const::buildWeightsConst(rewriter, convOp.getLoc(), biasType, ArrayRef(biasValue)));
            }
        }
        concatBias = rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), newBias, Dims4D::Act::C);
    }

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            origOp.getLoc(), outputType, root, concatWeights, concatBias, convOp.getStrides(), convOp.getPadsBegin(),
            convOp.getPadsEnd(), convOp.getDilations(), convOp.getPostOpAttr(), convOp.getClampAttr(),
            convOp.getStaticScaleAttr(), convOp.getOutputChannelsAttr(), convOp.getInputChannelsAttr());

    rewriter.replaceOp(origOp, newConvOp.getOutput());
    return mlir::success();
}

//
// OptimizeSliceMultiplyConcat
//

/*
    Convert below pattern:

             Root(1x24x1x64)
            /               \
    Slice(1x24x1x32)    Slice(1x24x1x32)    Constant(1x1x1x1)
            |                   \               /
            |                   Multiply(1x24x1x32)
            \                   /
              Concat(1x24x1x64)

    to:

        Root(1x24x1x64)
                |
    PermuteCast(1x64x24x1@NHWC)     Weights(64x64x1x1)
                    \               /
                    Convolution(1x64x24x1@NHWC)
                            |
                        PermuteCast(1x24x1x64)
*/

class OptimizeSliceMultiplyConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
    struct InputPattern {
        IE::SliceOp slice = nullptr;
        IE::MultiplyOp multiply = nullptr;

        InputPattern(IE::SliceOp sliceOp, IE::MultiplyOp multiplyOp): slice(sliceOp), multiply(multiplyOp) {
        }
    };

public:
    OptimizeSliceMultiplyConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(std::move(log)) {
        setDebugName("OptimizeSliceMultiplyConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::FailureOr<SmallVector<InputPattern>> getValidConcatInputs(IE::ConcatOp concatOp) const;

    mlir::Value createWeightsForScale(mlir::Location loc, ShapeRef weightsShape, mlir::Type elemType, int64_t offset,
                                      float scale, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::FailureOr<SmallVector<OptimizeSliceMultiplyConcat::InputPattern>>
OptimizeSliceMultiplyConcat::getValidConcatInputs(IE::ConcatOp concatOp) const {
    auto inputNum = concatOp.getInputs().size();
    SmallVector<InputPattern> concatInputs;
    concatInputs.reserve(inputNum);

    const auto concatOutType = concatOp.getOutput().getType().cast<NDTypeInterface>();
    const auto SUPPORTED_LAYOUT = DimsOrder::NCHW;
    const auto SUPPORTED_SLICE_CONCAT_DIM = Dims4D::Act::W;
    if (concatOutType.getDimsOrder() != SUPPORTED_LAYOUT) {
        return mlir::failure();
    }
    const auto concatOutShape = concatOutType.getShape();

    // Inputs of Concat should have the same root
    mlir::Value maybeRoot = nullptr;
    auto hasTheSameParent = [&maybeRoot](mlir::Value parent) {
        if (maybeRoot == nullptr) {
            maybeRoot = parent;
            return true;
        }

        return maybeRoot == parent;
    };

    // MultiplyOp on concat input should have single scale value
    auto hasSingleScaleValue = [](IE::MultiplyOp multiplyOp) {
        auto scale = multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (scale == nullptr) {
            return false;
        }
        auto scaleShape = getShape(multiplyOp.getInput2());
        return scaleShape.totalSize() == 1;
    };

    size_t numOfMultiplyOps = 0;
    for (const auto& input : concatOp.getInputs()) {
        // Check if concat on dim W
        auto concatDims = getDimsWithDifferentDimSize(getShape(input), concatOutShape);
        if (concatDims.size() != 1 || concatDims.front() != SUPPORTED_SLICE_CONCAT_DIM) {
            _log.trace("Unsupported concat dimension");
            return mlir::failure();
        }

        // Find SliceOp on input chain and check if slice on dim W
        // MultiplyOp is optional on input chain
        auto sliceOp = input.getDefiningOp<IE::SliceOp>();
        IE::MultiplyOp multiplyOp = nullptr;
        if (sliceOp == nullptr) {
            multiplyOp = input.getDefiningOp<IE::MultiplyOp>();
            if (multiplyOp == nullptr) {
                return mlir::failure();
            }
            sliceOp = multiplyOp.getInput1().getDefiningOp<IE::SliceOp>();
        }

        if (sliceOp == nullptr || !sliceOp->hasOneUse()) {
            _log.trace("Can't find SliceOp");
            return mlir::failure();
        }

        auto sliceDims = getDimsWithDifferentDimSize(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
        if (sliceDims.size() != 1 || sliceDims.front() != SUPPORTED_SLICE_CONCAT_DIM) {
            _log.trace("Unsupported slice dimension");
            return mlir::failure();
        }

        // Check if sibling SliceOps have the same parent
        auto parent = sliceOp.getSource();
        if (!hasTheSameParent(parent)) {
            return mlir::failure();
        }

        if (multiplyOp == nullptr) {
            // case when input chain doesn't include MultiplyOp
            concatInputs.push_back(InputPattern(sliceOp, nullptr));
            continue;
        }

        // case when input chain includes MultiplyOp, should have single scale value
        if (!multiplyOp->hasOneUse()) {
            return mlir::failure();
        }

        if (!hasSingleScaleValue(multiplyOp)) {
            return mlir::failure();
        }

        numOfMultiplyOps++;
        concatInputs.push_back(InputPattern(sliceOp, multiplyOp));
    }

    if (maybeRoot == nullptr) {
        return mlir::failure();
    }

    // Expect at least one input chain has MultiplyOp
    if (numOfMultiplyOps == 0) {
        return mlir::failure();
    }

    return concatInputs;
}

mlir::Value OptimizeSliceMultiplyConcat::createWeightsForScale(mlir::Location loc, ShapeRef weightsShape,
                                                               mlir::Type elemType, int64_t offset, float scale,
                                                               mlir::PatternRewriter& rewriter) const {
    const auto kx = weightsShape[Dims4D::Filter::KX];
    const auto ky = weightsShape[Dims4D::Filter::KY];
    const auto ic = weightsShape[Dims4D::Filter::IC];
    const auto oc = weightsShape[Dims4D::Filter::OC];

    std::vector<float> weights(weightsShape.totalSize(), .0f);
    // assign values
    const auto weightsSizePerOC = ic * ky * kx;
    for (int64_t ocIndex = 0; ocIndex < oc; ocIndex++) {
        auto currOffsetPerOC = ocIndex * weightsSizePerOC;
        auto innerOffset = offset + ocIndex;
        weights[currOffsetPerOC + innerOffset] = scale;
    }
    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType = mlir::RankedTensorType::get(
            weightsShape.raw(), elemType, getTensorAttr(rewriter.getContext(), weightsOrder, nullptr, nullptr));

    return Const::buildWeightsConst(rewriter, loc, weightsType, ArrayRef(weights));
}

mlir::LogicalResult OptimizeSliceMultiplyConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    const auto elemType = outputType.getElementType();

    auto getInputs = getValidConcatInputs(origOp);
    if (mlir::failed(getInputs)) {
        _log.trace("[{0}] Failed to get valid input pattern at '{1}'", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Rewriting '{1}'", origOp->getName(), origOp->getLoc());

    auto inputs = getInputs.value();
    mlir::Value root = inputs.front().slice.getSource();
    // Cast DimsOrder from NCHW to NHWC
    auto ctx = rewriter.getContext();
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(outputType.getRank()), ctx);
    auto targetInOrderMap = DimsOrder::NHWC.toAffineMap(ctx);
    auto inputCast = rewriter.create<IE::PermuteCastOp>(
            appendLoc(origOp.getLoc(), "branches_concat_to_conv_permute_cast_in"), root, targetInOrderMap, identityMap);

    // Create filter for new Conv
    SmallVector<mlir::Value> newWeights;
    for (auto input : inputs | indexed) {
        auto inputValue = input.value();
        auto inputIndex = input.index();
        auto sliceOp = inputValue.slice;
        auto multiply = inputValue.multiply;

        const auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
        const auto sliceShape = getShape(sliceOp.getResult());

        const auto numIC = getShape(inputCast.getOutput())[Dims4D::Act::C];
        const auto numOC = sliceShape[Dims4D::Act::W];
        const Shape weightShape = Shape{numOC, numIC, 1, 1};
        const auto offset = sliceOffset[Dims4D::Act::W.ind()];
        if (multiply != nullptr) {
            auto constOp = multiply.getInput2().getDefiningOp<Const::DeclareOp>();
            VPUX_THROW_WHEN(constOp == nullptr, "Multiply input is not a constant");
            auto constContent = constOp.getContentAttr().fold();
            auto scale = constContent.getSplatValue<float>();
            newWeights.push_back(createWeightsForScale(
                    appendLoc(origOp.getLoc(), "branches_concat_to_conv_weights_slice_{0}", inputIndex), weightShape,
                    elemType, offset, scale, rewriter));
        } else {
            auto scale = 1.0f;
            newWeights.push_back(createWeightsForScale(
                    appendLoc(origOp.getLoc(), "branches_concat_to_conv_weights_slice_{0}", inputIndex), weightShape,
                    elemType, offset, scale, rewriter));
        }
    }

    auto concatWeights = rewriter.createOrFold<IE::ConcatOp>(
            appendLoc(origOp.getLoc(), "branches_concat_to_conv_weights"), newWeights, Dims4D::Filter::OC);

    // Create new Conv
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto newConv = rewriter.create<IE::ConvolutionOp>(appendLoc(origOp.getLoc(), "branches_concat_to_conv"), inputCast,
                                                      concatWeights, nullptr, stridesAttr, padBeginAttr, padEndAttr,
                                                      dilationsAttr, nullptr, nullptr, nullptr, nullptr, nullptr);
    changeDimsOrder(newConv, DimsOrder::NHWC, _log.nest());

    // Cast to the original DimsOrder
    auto targetOutOrderMap = DimsOrder::NCHW.toAffineMap(ctx);
    auto outputCast =
            rewriter.create<IE::PermuteCastOp>(appendLoc(origOp.getLoc(), "branches_concat_to_conv_permute_cast_out"),
                                               newConv.getOutput(), targetOutOrderMap, identityMap);

    rewriter.replaceOp(origOp, outputCast.getOutput());

    return mlir::success();
}

//
// ConvertBranchesConcatToConvPass
//

class ConvertBranchesConcatToConvPass final :
        public IE::impl::ConvertBranchesConcatToConvBase<ConvertBranchesConcatToConvPass> {
public:
    explicit ConvertBranchesConcatToConvPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertBranchesConcatToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeGroupConvConcat>(&ctx, _log);
    patterns.add<OptimizeConvConcat>(&ctx, _log);
    patterns.add<OptimizeSliceMultiplyConcat>(&ctx, _log);
    IE::ConcatOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBranchesConcatToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBranchesConcatToConvPass(Logger log) {
    return std::make_unique<ConvertBranchesConcatToConvPass>(log);
}
