//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>

#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEEXPAND
#define GEN_PASS_DEF_PROPAGATEEXPAND
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t UNEXPANDED_CHANNELS = 3;
constexpr int64_t EXPANDED_CHANNELS = 4;
// E-69860 - Implement D2S on DPU - Supported values for DPU implementation.
constexpr int64_t D2SCHANNEL_CUTOFF = 16;
constexpr int64_t BLOCK_SIZE = 2;
//
// EltwiseShapeCastRewriter
//

class EltwiseShapeCastRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    EltwiseShapeCastRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("EltwiseShapeCastRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkInput(mlir::Value tensor, int64_t origChannels) const;
    mlir::Value getExpandedInput(mlir::PatternRewriter& rewriter, IE::ShapeCastOp shapeCast, IE::ExpandOp expandOp,
                                 Shape expandedShape) const;

    Logger _log;
};

mlir::Value EltwiseShapeCastRewriter::getExpandedInput(mlir::PatternRewriter& rewriter, IE::ShapeCastOp shapeCast,
                                                       IE::ExpandOp expandOp, Shape expandedShape) const {
    auto tensor = shapeCast.getSource();
    if (auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(tensor.getDefiningOp())) {
        rewriter.setInsertionPointAfter(permuteQuantize);

        auto pqUsers = permuteQuantize.getOutput().getUsers();
        SmallVector<mlir::Operation*> otherUsers;
        llvm::copy_if(pqUsers, std::back_inserter(otherUsers), [&](auto user) {
            return user != shapeCast.getOperation();
        });

        auto expand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), permuteQuantize.getOutput(),
                                                    expandOp.getPadsBegin(), expandOp.getPadsEnd());

        if (!otherUsers.empty()) {
            auto sliceOp = rewriter.create<IE::SliceOp>(
                    expandOp.getLoc(), expand.getOutput(), expandOp.getPadsBegin(),
                    getIntArrayAttr(rewriter.getContext(), getShape(permuteQuantize.getOutput()).raw()));
            for (auto user : otherUsers) {
                user->replaceUsesOfWith(permuteQuantize.getOutput(), sliceOp.getResult());
            }
        }

        auto newShapeCast = rewriter.create<IE::ShapeCastOp>(
                expandOp.getLoc(), expand.getOutput(), getIntArrayAttr(expandOp.getContext(), expandedShape.raw()));

        return newShapeCast;
    } else if (auto depthToSpace = mlir::dyn_cast_or_null<IE::DepthToSpaceOp>(tensor.getDefiningOp())) {
        rewriter.setInsertionPoint(expandOp);
        auto expand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), depthToSpace.getOutput(),
                                                    expandOp.getPadsBegin(), expandOp.getPadsEnd());

        return rewriter.create<IE::ShapeCastOp>(expandOp.getLoc(), expand.getOutput(),
                                                getIntArrayAttr(expandOp.getContext(), expandedShape.raw()));
    }

    VPUX_THROW("Unsupported input {0}", *tensor.getDefiningOp());
}

// Check here that there are layers before ShapeCast
// which might be fused with Expand
// For now following cases are supported:
// 1.  PermuteQuantize -> Expand -> ShapeCase
//  in this case PermuteQuantize is fused with Expand later on
// 2. Slice -> DepthToSpace -> Expand -> ShapeCast
//  here Expand might be fused with DepthToSpace which has padded descriptor
// if we fuse these layers with Expand, ShapeCast uses already expanded tensor
// and reshape it back after Expand

bool EltwiseShapeCastRewriter::checkInput(mlir::Value tensor, int64_t origChannels) const {
    if (auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(tensor.getDefiningOp())) {
        // PermuteQuantize is going to be fused with Expand only in that case
        if (getShape(permuteQuantize.getOutput())[Dims4D::Act::C] != UNEXPANDED_CHANNELS &&
            origChannels != EXPANDED_CHANNELS) {
            return false;
        }

        return true;
    } else if (auto depthToSpace = mlir::dyn_cast_or_null<IE::DepthToSpaceOp>(tensor.getDefiningOp())) {
        if (!depthToSpace.getOutput().hasOneUse()) {
            return false;
        }

        auto slice = depthToSpace.getInput().getDefiningOp<IE::SliceOp>();

        if (slice == nullptr || !slice.getResult().hasOneUse()) {
            return false;
        }

        const auto blockSize = depthToSpace.getBlockSize();
        VPUX_THROW_WHEN(blockSize <= 0, "blockSize must be greater than zero");
        if (getShape(slice.getSource())[Dims4D::Act::C] / (blockSize * blockSize) != origChannels) {
            return false;
        }

        return true;
    }

    return false;
}

mlir::LogicalResult EltwiseShapeCastRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    // F16 case would be beneficial to be executed as Concat + Const.
    auto inputTensor = expandOp.getInput();
    if (mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(
                mlir::cast<vpux::NDTypeInterface>(inputTensor.getType()).getElementType()) == nullptr) {
        return mlir::failure();
    }

    auto lastShapeCastOp = inputTensor.getDefiningOp<IE::ShapeCastOp>();

    if (lastShapeCastOp == nullptr) {
        return mlir::failure();
    }

    auto addOp = lastShapeCastOp.getSource().getDefiningOp<IE::AddOp>();

    if (addOp == nullptr) {
        return mlir::failure();
    }

    auto firstInput = addOp.getInput1().getDefiningOp<IE::ShapeCastOp>();
    auto secondInput = addOp.getInput2().getDefiningOp<IE::ShapeCastOp>();

    if (firstInput == nullptr || secondInput == nullptr) {
        return mlir::failure();
    }

    auto origChannels = getShape(expandOp)[Dims4D::Act::C];
    bool needExplicitExpand = false;
    if (origChannels != EXPANDED_CHANNELS) {
        // In case the channels are not expanded to 4, but they can be expanded to this value, an explicit pair of
        // Expand->Slice can be introduced in order to support this optimization
        if (getShape(expandOp.getInput())[Dims4D::Act::C] < EXPANDED_CHANNELS) {
            needExplicitExpand = true;
            origChannels = EXPANDED_CHANNELS;
        }
    }

    bool equalInputs = firstInput == secondInput;

    if (!checkInput(firstInput.getSource(), origChannels) || !checkInput(secondInput.getSource(), origChannels)) {
        return mlir::failure();
    }

    Shape newUnexpandedShape = getShape(firstInput.getSource()).toValues();
    newUnexpandedShape[Dims4D::Act::C] = origChannels;
    Shape newExpandedShape = getShape(firstInput.getSource()).toValues();
    newExpandedShape[Dims4D::Act::C] = getShape(addOp)[Dims4D::Act::C];

    auto expandedShape = vpux::IE::getShapeCastExpandedShape(addOp.getOperation(), newExpandedShape, newUnexpandedShape,
                                                             _log.nest());

    if (mlir::failed(expandedShape)) {
        return mlir::failure();
    }

    if (needExplicitExpand) {
        auto shape = getShape(expandOp.getInput());
        SmallVector<int64_t> padBegin(shape.size(), 0);
        SmallVector<int64_t> padEnd(shape.size(), 0);
        padEnd[Dims4D::Act::C.ind()] = EXPANDED_CHANNELS - shape[Dims4D::Act::C];
        auto newExpandOp = rewriter.create<IE::ExpandOp>(appendLoc(expandOp->getLoc(), "_expand"), expandOp.getInput(),
                                                         getIntArrayAttr(getContext(), padBegin),
                                                         getIntArrayAttr(getContext(), padEnd));
        SmallVector<int64_t> staticOffsets(shape.size(), 0);
        SmallVector<int64_t> staticSizes(shape.raw());
        auto newSliceOp = rewriter.create<IE::SliceOp>(appendLoc(expandOp->getLoc(), "_slice"), newExpandOp.getOutput(),
                                                       getIntArrayAttr(getContext(), staticOffsets),
                                                       getIntArrayAttr(getContext(), staticSizes));
        rewriter.replaceAllUsesExcept(expandOp.getInput(), newSliceOp.getOutput(), newExpandOp);
        expandOp = newExpandOp;
    }

    auto expandedInput1 = getExpandedInput(rewriter, firstInput, expandOp, expandedShape.value());
    auto expandedInput2 =
            !equalInputs ? getExpandedInput(rewriter, secondInput, expandOp, expandedShape.value()) : expandedInput1;
    rewriter.setInsertionPoint(expandOp);

    auto outputType =
            mlir::cast<vpux::NDTypeInterface>(expandedInput1.getType())
                    .changeElemType(mlir::cast<vpux::NDTypeInterface>(addOp.getOutput().getType()).getElementType());
    auto newEltwise = rewriter.create<IE::AddOp>(addOp.getLoc(), outputType, expandedInput1, expandedInput2,
                                                 addOp.getAutoBroadcast(), nullptr, nullptr, nullptr, nullptr);

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(expandOp, newEltwise.getOutput(),
                                                 getIntArrayAttr(expandOp.getContext(), getShape(expandOp).raw()));

    return mlir::success();
}

//
// SliceShapeCastRewriter
// Convert pattern:
//      Slice                            Slice
//        |                                |
//      ShapeCast         ->             Expand
//        |                                |
//      Expand                           ShapeCast
//

class SliceShapeCastRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    SliceShapeCastRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("SliceShapeCastRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

    Logger _log;
};

mlir::LogicalResult SliceShapeCastRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    auto shapeCastOp = mlir::dyn_cast_or_null<IE::ShapeCastOp>(expandOp.getInput().getDefiningOp());
    if (shapeCastOp == nullptr || !shapeCastOp->hasOneUse()) {
        return mlir::failure();
    }

    auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(shapeCastOp.getInput().getDefiningOp());
    if (sliceOp == nullptr || !sliceOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto sliceInShape = getShape(sliceOp.getSource());
    const auto sliceOutShape = getShape(sliceOp.getResult());
    const auto sliceAxisData = IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
    if (!sliceAxisData.has_value()) {
        return mlir::failure();
    }

    const auto expandAxisData = IE::getExpandAxis(expandOp);
    if (!expandAxisData.has_value()) {
        return mlir::failure();
    }

    // Check the slice and expand on same axis
    const auto sliceAxis = sliceAxisData.value();
    const auto expandAxis = expandAxisData.value();
    if (sliceAxis != expandAxis) {
        return mlir::failure();
    }

    const auto shapeCastInType = mlir::cast<vpux::NDTypeInterface>(shapeCastOp.getInput().getType());
    const auto shapeCastOutType = mlir::cast<vpux::NDTypeInterface>(shapeCastOp.getResult().getType());
    const auto shapeCastInShape = shapeCastInType.getShape();
    const auto shapeCastOutShape = shapeCastOutType.getShape();
    const auto shapeCastInorder = shapeCastInType.getDimsOrder();
    auto getInnermostDim = [](const vpux::DimsOrder& order) {
        return order.toDim(MemDim(order.numDims() - 1));
    };
    auto getHighestDim = [](const vpux::DimsOrder& order) {
        return order.toDim(MemDim(0));
    };
    const auto shapeCastInnermostDim = getInnermostDim(shapeCastInorder);
    const auto shapeCastHighestDim = getHighestDim(shapeCastInorder);

    // Check the expand dim is the innermost dim or highest dim of shapeCast and the dim size is not reshaped
    if (shapeCastInnermostDim != expandAxis && shapeCastHighestDim != expandAxis) {
        return mlir::failure();
    }

    if (shapeCastInShape[expandAxis] != shapeCastOutShape[expandAxis]) {
        return mlir::failure();
    }

    rewriter.setInsertionPoint(expandOp);
    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), shapeCastOp.getInput(), expandOp.getPadsBegin(),
                                                     expandOp.getPadsEnd());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(expandOp, newExpandOp.getOutput(),
                                                 getIntArrayAttr(rewriter.getContext(), getShape(expandOp)));
    _log.trace("Successfully propagate Expand through ShapeCast after Slice.");

    return mlir::success();
}

//
// DepthToSpaceSliceRewriter
//

class DepthToSpaceSliceRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    DepthToSpaceSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("DepthToSpaceSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value getConcatResult(mlir::PatternRewriter& rewriter, mlir::Value input, int64_t blockSize) const;
    mlir::Value createConvforD2S(mlir::PatternRewriter& rewriter, mlir::Value input, int64_t blockSize,
                                 IE::ExpandOp expandOp) const;
    mlir::Value createConvFilter(mlir::PatternRewriter& rewriter, mlir::Value inputTensor, int64_t blockSize) const;

private:
    Logger _log;
};

mlir::LogicalResult DepthToSpaceSliceRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    auto depthToSpace = expandOp.getInput().getDefiningOp<IE::DepthToSpaceOp>();

    if (depthToSpace == nullptr) {
        return mlir::failure();
    }

    // For now only block_first mode supported, depth_first has AC issue.
    // E#108049 for depth_first mode.
    if (depthToSpace.getMode() != IE::DepthToSpaceMode::BLOCKS_FIRST) {
        return mlir::failure();
    }

    auto slice = depthToSpace.getInput().getDefiningOp<IE::SliceOp>();

    if (slice == nullptr) {
        return mlir::failure();
    }
    auto sliceOffsets = parseIntArrayAttr<int64_t>(slice.getStaticOffsets());
    auto hasNonZeroOffsets = llvm::any_of(sliceOffsets, [](auto offset) {
        return offset != 0;
    });
    if (hasNonZeroOffsets) {
        return mlir::failure();
    }

    if (!mlir::isa_and_nonnull<IE::AlignedChannelsOpInterface>(slice.getSource().getDefiningOp())) {
        return mlir::failure();
    }
    const auto sliceElemType = mlir::cast<vpux::NDTypeInterface>(slice.getSource().getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(sliceElemType)) {
        return mlir::failure();
    }

    const auto blockSize = depthToSpace.getBlockSize();
    if (blockSize <= 0) {
        return matchFailed(rewriter, depthToSpace, "Unsupported block size: {0}", blockSize);
    }
    auto blockSizeSquare = blockSize * blockSize;
    auto sliceInChannels = getShape(slice.getSource())[Dims4D::Act::C];
    if (sliceInChannels / blockSizeSquare != getShape(expandOp)[Dims4D::Act::C]) {
        return mlir::failure();
    }

    if (sliceInChannels < D2SCHANNEL_CUTOFF || depthToSpace.getBlockSize() != BLOCK_SIZE) {
        auto paddedIC = sliceInChannels - getShape(slice.getResult())[Dims4D::Act::C];
        auto paddedOC = paddedIC / blockSizeSquare;

        auto paddedChannels =
                IE::ChannelPaddingAttr::get(expandOp.getContext(), getIntAttr(expandOp.getContext(), paddedIC),
                                            getIntAttr(expandOp.getContext(), paddedOC));

        rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(expandOp, slice.getSource(), depthToSpace.getBlockSize(),
                                                        depthToSpace.getMode(), paddedChannels);
    } else {
        auto paddedInput = getConcatResult(rewriter, slice.getSource(), depthToSpace.getBlockSize());
        auto d2sTensor = createConvforD2S(rewriter, paddedInput, depthToSpace.getBlockSize(), expandOp);
        auto outputShape = getShape(d2sTensor);
        Shape newOutShape(outputShape.size(), 1);

        newOutShape[Dims4D::Act::N] = outputShape[Dims4D::Act::N];
        newOutShape[Dims4D::Act::C] = outputShape[Dims4D::Act::C] / blockSizeSquare;
        newOutShape[Dims4D::Act::H] = outputShape[Dims4D::Act::H];
        newOutShape[Dims4D::Act::W] = outputShape[Dims4D::Act::W] * blockSizeSquare;

        SmallVector<SmallVector<int64_t>> reassociationMap(newOutShape.size());
        for (size_t dimIdx = 0; dimIdx < newOutShape.size(); dimIdx++) {
            reassociationMap[dimIdx].push_back(dimIdx);
        }
        auto outputDimAttr = getIntArrayOfArray(getContext(), reassociationMap);

        const auto outShapeAttr = getIntArrayAttr(getContext(), newOutShape);

        rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(expandOp, d2sTensor, outputDimAttr, outShapeAttr);
    }

    return mlir::success();
}
/*
Convert the following subgraph in case no of channels to the input of D2S are > 16.
   Conv (1x16x256x256)                         Input(1x16x256x256)     Const (1x16x256x256)
      |                                                 \        /
Slice (1x12x256x256)                                    Concat (H-axis) (1x16x512x256)
      |                                                       |         Weights (16x16x2x2)
DepthToSpace (1x3x512x512)                                    |         /
      |                                                  Conv (Top pad = 1) (1x16x512x128)
Expand (1x4x512x512)                                        |
                                                        AffineReshape (1x4x512x512)

    */

mlir::Value DepthToSpaceSliceRewriter::getConcatResult(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                       int64_t blockSize) const {
    auto loc = input.getLoc();
    auto inputType = input.getType();
    auto inputShape = getShape(input);
    float zeroValue = 0.f;

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(inputType).getElementType();

    if (auto qInputElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
        const auto zeroPoint = qInputElemType.getZeroPoint();
        zeroValue = static_cast<float>(zeroPoint);
    }

    auto filterTensorAttr = vpux::getTensorAttr(getContext(), DimsOrder::NHWC, nullptr);
    auto filterType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(inputShape.raw(), elemType, filterTensorAttr));
    auto dataStorageType =
            mlir::RankedTensorType::get(inputShape.raw(), mlir::Float32Type::get(filterType.getContext()));

    auto dataAttr = Const::createConstContent(dataStorageType, ArrayRef(zeroValue));
    Const::ContentSetup contentAttrSetup(dataStorageType);

    if (auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(qElemType);
    } else if (mlir::isa<mlir::Float16Type>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(mlir::Float16Type::get(filterType.getContext()));
    } else {
        VPUX_THROW("Unsupported type {0}", elemType);
    }
    auto contentAttr = Const::ContentAttr::get(dataAttr, contentAttrSetup.reorder(filterType.getDimsOrder()));
    auto constTensor = rewriter.create<Const::DeclareOp>(loc, filterType, std::move(contentAttr)).getOutput();
    // Concat along H axis and blockSize as stride
    auto axis = Dims4D::Act::H;
    auto offset = 1;
    SmallVector<mlir::Value> interlacedInputs;
    interlacedInputs.push_back(input);
    interlacedInputs.push_back(constTensor);
    return rewriter.create<IE::ConcatOp>(loc, mlir::ValueRange(interlacedInputs), axis, offset, blockSize).getOutput();
}
/*
Creates filter with following patter for each channel :
- non-zero bits are incremented per nth channel, along Z axis. i.e in 0th weight set bit[2] and bit[24] =1. In 1st
weight set, bit[6] and bit[30]= 1.
- Every 3,4,7,11, and 15th weight sets (sets corresponding to the padded zero descriptor in input tensor after conv) are
set to 0. For the rest of set :
- For the first half :
        | bit0    | bit1 | ///////  |bit24 =1 | bit25 |  ///////
        | bit2 =1 | bit3 | ///////  |bit26     | bit27 |  ///////
- For the second half :
        | bit0 | bit1    | ////////  |bit24 | bit25=1 | ///////
        | bit2 | bit3 =1 | ///////   |bit26 | bit27   |  ///////
*/
mlir::Value DepthToSpaceSliceRewriter::createConvFilter(mlir::PatternRewriter& rewriter, mlir::Value inputTensor,
                                                        int64_t blockSize) const {
    const auto IC = getShape(inputTensor)[Dims4D::Act::C];
    const auto KX = blockSize;
    const auto KY = blockSize;
    const auto OC = IC;
    const Shape weightShape = {OC, IC, KX, KY};

    SmallVector<float> weights(weightShape.totalSize(), .0f);

    auto channelStride = KX * KY;
    const auto dimsOrder = mlir::cast<vpux::NDTypeInterface>(inputTensor.getType()).getDimsOrder();
    auto outMemShape = dimsOrder.toMemoryOrder(ShapeRef(weightShape.raw()));
    const auto halfOC = OC / 2;
    for (auto i = 0; i < OC; ++i) {
        const auto outMemIndND = dimsOrder.toMemoryOrder(ShapeRef({i, 0, 0, 0}));
        const auto outMemInd1D = getMemIndex1D(outMemIndND, outMemShape);
        const auto check = i % halfOC;
        const auto offset = (check < 4) ? channelStride * check : channelStride * (check - 1);
        const auto index = outMemInd1D + offset;
        if (i % channelStride == 3) {
            // Condition : All elements are zero for O = 3, 7, 11, 15
            continue;
        }
        if (i < halfOC && i % channelStride != 3) {
            // Condition : 2nd bit and 24th bit of each tensor in first half is 1
            weights[index + 2] = 1.0f;
            weights[index + 24] = 1.0f;
        } else if (i >= halfOC && i % channelStride != 3) {
            // Condition : 3rd bit and 25th bit of each tensor in second half is 1
            weights[index + 3] = 1.0f;
            weights[index + 25] = 1.0f;
        }
    }

    const DimsOrder weightOrder = DimsOrder::OYXI;
    const auto weightType = mlir::RankedTensorType::get(
            weightShape.raw(), mlir::cast<NDTypeInterface>(inputTensor.getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    return Const::buildWeightsConst(rewriter, inputTensor.getLoc(), weightType, ArrayRef(weights));
}

mlir::Value DepthToSpaceSliceRewriter::createConvforD2S(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                        int64_t blockSize, IE::ExpandOp expandOp) const {
    const auto convFilter = createConvFilter(rewriter, input, blockSize);
    auto newStrides = getIntArrayAttr(
            getContext(), ov::Strides{checked_cast<size_t>(blockSize) / 2, checked_cast<size_t>(blockSize)});
    auto newPadsBegin = getIntArrayAttr(getContext(), ov::CoordinateDiff{1, 0});  // Top Pad = 1
    auto newPadsEnd = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newDilations = getIntArrayAttr(getContext(), ov::Strides{1, 1});
    auto outputType = mlir::cast<vpux::NDTypeInterface>(expandOp.getType());
    auto newInputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape().toValues().raw();
    auto filterShape = mlir::cast<mlir::ShapedType>(convFilter.getType()).getShape();

    const auto op =
            ov::op::v1::Convolution(std::make_shared<ov::op::v0::Parameter>(
                                            ov::element::i32, ov::Shape(newInputShape.begin(), newInputShape.end())),
                                    std::make_shared<ov::op::v0::Parameter>(
                                            ov::element::i32, ov::Shape(filterShape.begin(), filterShape.end())),
                                    ov::Strides{checked_cast<size_t>(blockSize) / 2, checked_cast<size_t>(blockSize)},
                                    ov::CoordinateDiff{1, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

    const auto& outputShape = op.get_output_partial_shape(0);
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    auto output = outputType.changeShape(ShapeRef(shapeI64)).changeDimsOrder(DimsOrder::NHWC);
    return rewriter.create<IE::ConvolutionOp>(input.getLoc(), output, input, convFilter, nullptr, newStrides,
                                              newPadsBegin, newPadsEnd, newDilations, nullptr, nullptr, nullptr,
                                              nullptr, nullptr);
}
//
// SpaceToDepthSliceRewriter
//

class SpaceToDepthSliceRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    SpaceToDepthSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("SpaceToDepthSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createDPUOperation(mlir::PatternRewriter& rewriter, mlir::Value input, IE::SpaceToDepthOp s2dOp) const;
    void createPaddedConvolution(mlir::PatternRewriter& rewriter, mlir::Value input, IE::ConvolutionOp origConv) const;

    mlir::Value createConvFilter(mlir::PatternRewriter& rewriter, mlir::Value activation, int64_t blockSize) const;
    bool checkSliceExpand(IE::SliceOp sliceOp, IE::ExpandOp expandOp) const;
    Logger _log;
};

mlir::Value SpaceToDepthSliceRewriter::createConvFilter(mlir::PatternRewriter& rewriter, mlir::Value activation,
                                                        int64_t blockSize) const {
    const auto IC = getShape(activation)[Dims4D::Act::C];  // 16
    const auto KX = blockSize;                             // 4
    const auto KY = blockSize;                             // 4
    const auto OC = IC * blockSize * blockSize;            // 256

    const Shape weightShape = {OC, IC, KX, KY};

    SmallVector<float> weights(weightShape.totalSize(), .0f);

    const auto outChannelStride = (IC * KX * KY);
    const auto inChannelStride = (KX * KY);
    const auto heightStride = KX;

    // set 1 to each filter so that it calculates the same as S2D
    // multiply coordinates on strides accordingly for each filter
    for (auto i = 0; i < OC; ++i) {
        const auto index = i * outChannelStride + (i % IC) * inChannelStride + (i / (KY * IC)) * heightStride +
                           ((i % (KY * IC)) / IC);

        weights[index] = 1.0f;
    }

    const DimsOrder weightOrder = DimsOrder::OYXI;
    const auto weightType = mlir::RankedTensorType::get(
            weightShape.raw(), mlir::cast<NDTypeInterface>(activation.getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    return Const::buildWeightsConst(rewriter, activation.getLoc(), weightType, ArrayRef(weights));
}

mlir::Value SpaceToDepthSliceRewriter::createDPUOperation(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                          IE::SpaceToDepthOp s2dOp) const {
    const auto blockSize = s2dOp.getBlockSize();
    const auto inputShape = getShape(input);
    const auto s2dOutShape = getShape(s2dOp.getOutput());

    const auto convFilter = createConvFilter(rewriter, input, blockSize);
    auto newStrides = getIntArrayAttr(getContext(),
                                      ov::Strides{checked_cast<size_t>(blockSize), checked_cast<size_t>(blockSize)});
    auto newPadsBegin = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newPadsEnd = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newDilations = getIntArrayAttr(getContext(), ov::Strides{1, 1});

    const Shape outPadBefore(checked_cast<size_t>(s2dOp.getType().getRank()), 0);

    Shape outPadAfter(checked_cast<size_t>(s2dOp.getType().getRank()), 0);
    outPadAfter[Dims4D::Act::C] = inputShape[Dims4D::Act::C] * blockSize * blockSize - s2dOutShape[Dims4D::Act::C];

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(s2dOp.getType());
    const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

    return rewriter.create<IE::ConvolutionOp>(s2dOp.getLoc(), newOutputType, input, convFilter, nullptr, newStrides,
                                              newPadsBegin, newPadsEnd, newDilations, nullptr, nullptr, nullptr,
                                              nullptr, nullptr);
}

void SpaceToDepthSliceRewriter::createPaddedConvolution(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                        IE::ConvolutionOp origConv) const {
    auto filter = origConv.getFilter();

    auto filterConst = filter.getDefiningOp<Const::DeclareOp>();

    const auto filterShape = getShape(filter);
    const auto channelsPad = getShape(input)[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];

    const auto newFilterShape = Shape({filterShape[Dims4D::Filter::OC], getShape(input)[Dims4D::Act::C],
                                       filterShape[Dims4D::Filter::KX], filterShape[Dims4D::Filter::KY]});
    auto newContentAttr =
            filterConst.getContentAttr()
                    .transform()
                    .reshape({filterShape[Dims4D::Filter::OC], filterShape[Dims4D::Filter::IC] / channelsPad,
                              filterShape[Dims4D::Filter::KX] * channelsPad, filterShape[Dims4D::Filter::KY]})
                    .padWithZero({0, 0, 0, 0}, {0, 1, 0, 0})
                    .reshape(newFilterShape)
                    .get();

    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(filter.getType());
    const auto outAllocType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(to_small_vector(newFilterShape), origFilterType.getElementType()));
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto paddedFilter =
            rewriter.create<Const::DeclareOp>(filterConst.getLoc(), outAllocTypeNHWC, std::move(newContentAttr));

    mlir::IRMapping mapper;
    mapper.map(origConv.getOperands(), ArrayRef({input, paddedFilter.getOutput()}));
    auto* newOperand = rewriter.clone(*origConv, mapper);

    rewriter.replaceOp(origConv, newOperand->getResult(0));
}

// Check if we have case when
// PermuteQuantizeOp -> ExpandOp (4 channels) -> SliceOp (3 channels) -> SpaceToDepth
bool SpaceToDepthSliceRewriter::checkSliceExpand(IE::SliceOp sliceOp, IE::ExpandOp expandOp) const {
    if (expandOp.getInput().getDefiningOp<IE::PermuteQuantizeOp>() == nullptr) {
        return false;
    }

    const auto expandShape = getShape(expandOp.getOutput());
    const auto sliceShape = getShape(sliceOp.getResult());

    if (expandShape[Dims4D::Act::C] != EXPANDED_CHANNELS || sliceShape[Dims4D::Act::C] != UNEXPANDED_CHANNELS) {
        return false;
    }

    if (expandOp.getInput().getType() != sliceOp.getResult().getType()) {
        return false;
    }

    auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    auto hasNonZeroOffsets = llvm::any_of(sliceOffsets, [](auto offset) {
        return offset != 0;
    });
    if (hasNonZeroOffsets) {
        return false;
    }

    return true;
}

// Convert pattern
//   PermuteQuantize                                 PermuteQuantize
//        |                                                 |
//     Expand (4 channels)                              Expand (4 channels)
//        |                                                 |
//      Slice (3 channels)           ->              Convolution (S2D on DPU)
//        |                                                 |
//    SpaceToDepth                                   Convolution (padded filter)
//        |
//    Convolution
mlir::LogicalResult SpaceToDepthSliceRewriter::matchAndRewrite(IE::ConvolutionOp convOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", convOp->getName(), convOp->getLoc());

    auto spaceToDepth = convOp.getInput().getDefiningOp<IE::SpaceToDepthOp>();

    if (spaceToDepth == nullptr || !spaceToDepth.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    // for now only block_first case supported
    // E#69685 for depth_first mode
    if (spaceToDepth.getMode() != IE::SpaceToDepthMode::BLOCKS_FIRST) {
        return mlir::failure();
    }

    auto slice = spaceToDepth.getInput().getDefiningOp<IE::SliceOp>();

    if (slice == nullptr) {
        return mlir::failure();
    }

    auto expandOp = slice.getSource().getDefiningOp<IE::ExpandOp>();

    if (expandOp == nullptr) {
        return mlir::failure();
    }

    if (!checkSliceExpand(slice, expandOp)) {
        return mlir::failure();
    }

    auto s2dDPU = createDPUOperation(rewriter, expandOp.getOutput(), spaceToDepth);

    createPaddedConvolution(rewriter, s2dDPU, convOp);

    return mlir::success();
}

//
// PropagateExpandPass
//

class PropagateExpandPass final : public IE::impl::PropagateExpandBase<PropagateExpandPass> {
public:
    explicit PropagateExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void PropagateExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EltwiseShapeCastRewriter>(&ctx, _log);
    patterns.add<SliceShapeCastRewriter>(&ctx, _log);
    patterns.add<DepthToSpaceSliceRewriter>(&ctx, _log);
    patterns.add<SpaceToDepthSliceRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateExpandPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateExpandPass(Logger log) {
    return std::make_unique<PropagateExpandPass>(log);
}
