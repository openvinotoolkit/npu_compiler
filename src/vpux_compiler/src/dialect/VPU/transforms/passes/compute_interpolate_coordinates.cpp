//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/utils/core/error.hpp>
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <utility>

namespace vpux::VPU {
#define GEN_PASS_DECL_COMPUTEINTERPOLATECOORDINATES
#define GEN_PASS_DEF_COMPUTEINTERPOLATECOORDINATES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class ComputeInterpolateCoordinates final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    ComputeInterpolateCoordinates(mlir::MLIRContext* ctx, bool enableExplicitDistributionInfoAttr, Logger log)
            : mlir::OpRewritePattern<VPU::InterpolateOp>(ctx),
              _enableExplicitDistributionInfoAttr(enableExplicitDistributionInfoAttr),
              _log(std::move(log)) {
        setDebugName("ComputeInterpolateCoordinates");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp interpolateOp, mlir::PatternRewriter& rewriter) const final;

private:
    template <typename ValueType>
    mlir::Value interpolateConstant(VPU::InterpolateOp interpolateOp, mlir::PatternRewriter& rewriter,
                                    mlir::Location loc, mlir::RankedTensorType type, ArrayRef<ValueType> values) const;

    bool _enableExplicitDistributionInfoAttr = false;
    Logger _log;
};

const float ZERO_FIVE = 0.5;

float mapCoordHalfPixel(int x, float scale, int64_t, int64_t) {
    const auto xF = static_cast<float>(x);
    return ((xF + ZERO_FIVE) / scale) - ZERO_FIVE;
}

float mapCoordPytorchHalfPixel(int x, float scale, int64_t lenghtResized, int64_t) {
    const auto xF = static_cast<float>(x);
    return (lenghtResized > 1) ? (xF + ZERO_FIVE) / scale - ZERO_FIVE : 0.0f;
}

float mapCoordAsymmetric(int x, float scale, int64_t, int64_t) {
    const auto xF = static_cast<float>(x);
    return xF / scale;
}

float mapCoordTfHalfPixelForNN(int x, float scale, int64_t, int64_t) {
    const auto xF = static_cast<float>(x);
    return (xF + ZERO_FIVE) / scale;
}

float mapCoordAlignCorners(int x, float, int64_t lenghtResized, int64_t lenghtOriginal) {
    const auto xF = static_cast<float>(x);
    return (lenghtResized == 1) ? 0
                                : xF * static_cast<float>(lenghtOriginal - 1) / static_cast<float>(lenghtResized - 1);
}

using MapCoordFuncT = std::function<float(int, float, int64_t, int64_t)>;

MapCoordFuncT getMapCoordFunc(IE::InterpolateCoordMode coordMode) {
    switch (coordMode) {
    case IE::InterpolateCoordMode::HALF_PIXEL: {
        return mapCoordHalfPixel;
    }
    case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL: {
        return mapCoordPytorchHalfPixel;
    }
    case IE::InterpolateCoordMode::ASYMMETRIC: {
        return mapCoordAsymmetric;
    }
    case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN: {
        return mapCoordTfHalfPixelForNN;
    }
    case IE::InterpolateCoordMode::ALIGN_CORNERS: {
        return mapCoordAlignCorners;
    }
    }
    return nullptr;
}

template <typename ValueType>
mlir::Value ComputeInterpolateCoordinates::interpolateConstant(VPU::InterpolateOp interpolateOp,
                                                               mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                               mlir::RankedTensorType type,
                                                               ArrayRef<ValueType> values) const {
    mlir::Value newConst = Const::createConst(rewriter, loc, type, values);

    auto inputType = mlir::dyn_cast_or_null<VPU::DistributedTypeInterface>(interpolateOp.getInput().getType());
    if (inputType != nullptr && inputType.containsDistributedTypes()) {
        const auto constType = mlir::cast<vpux::NDTypeInterface>(newConst.getType());
        auto inputDistrType = mlir::cast<VPU::DistributedTensorType>(inputType.getDistributedTypes().front());
        auto uniformSegments = inputDistrType.getDistribution().getUniformDistributedSegments() != nullptr;

        auto constDistributedType = VPU::createDistributedTensorType(
                mlir::cast<VPU::ClusteredOpInterface>(interpolateOp.getOperation()), constType,
                VPU::DistributionMode::DUPLICATED, {1, 1, 1, 1},
                inputDistrType.getDistribution().getNumClusters().getInt(), ArrayRef<int64_t>{}, uniformSegments,
                _enableExplicitDistributionInfoAttr, VPU::OverlapDistributionParams());
        newConst = rewriter.createOrFold<VPU::UnrolledTypeOp>(loc, constDistributedType, newConst);
    }

    return newConst;
}

mlir::LogicalResult ComputeInterpolateCoordinates::matchAndRewrite(VPU::InterpolateOp interpolateOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    const auto coordinates = interpolateOp.getCoordinates();
    const auto lambdas = interpolateOp.getLambdas();

    if (coordinates != nullptr && lambdas != nullptr) {
        return mlir::failure();
    }

    const auto interpolateAttr = interpolateOp.getAttr();
    const auto interpolateMode = interpolateAttr.getMode().getValue();

    // Computing coordinates at compile time is a feature supported only for linear interpolate modes.
    // The ticket for adding support for all interpolate modes is E#132985.
    if (interpolateMode != IE::InterpolateMode::LINEAR && interpolateMode != IE::InterpolateMode::LINEAR_ONNX) {
        return mlir::failure();
    }

    const auto loc = interpolateOp.getLoc();
    const auto ctx = interpolateOp.getContext();

    const auto inType = mlir::cast<NDTypeInterface>(interpolateOp.getInput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape().raw();
    const auto outShape = getShape(interpolateOp.getOutput()).raw();
    const auto coordMode = interpolateAttr.getCoordMode().getValue();

    const auto axesResult = IE::extractIntVector(loc, interpolateOp.getAxes(), interpolateOp.getAxesAttrAttr());
    VPUX_THROW_WHEN(mlir::failed(axesResult), "Failed to extract axes");
    const auto innermostAxisResult = IE::getInnermostAxis(loc, inOrder, axesResult.value());
    VPUX_THROW_WHEN(mlir::failed(innermostAxisResult), "Failed to get the innermost axis");
    const auto innermostAxis = innermostAxisResult.value();

    const auto parseIntArrayAttrOr = [](std::optional<mlir::ArrayAttr> attribute,
                                        const SmallVector<int64_t>& defaultValue) {
        return attribute.has_value() ? parseIntArrayAttr<int64_t>(attribute.value()) : defaultValue;
    };

    const auto initialInputShape =
            parseIntArrayAttrOr(interpolateOp.getInitialInputDimsAttr(), to_small_vector(inShape));
    const auto initialOutputShape =
            parseIntArrayAttrOr(interpolateOp.getInitialOutputDimsAttr(), to_small_vector(outShape));
    const auto initialInputOffset =
            parseIntArrayAttrOr(interpolateOp.getInitialInputOffsetAttr(), SmallVector<int64_t>(inShape.size(), 0));
    const auto initialOutputOffset =
            parseIntArrayAttrOr(interpolateOp.getInitialOutputOffsetAttr(), SmallVector<int64_t>(outShape.size(), 0));

    const auto innermostAxisInputDim = outShape[innermostAxis];
    const auto innermostAxisOuputDim = inShape[innermostAxis];
    const auto innermostAxisInitialInputDim = initialInputShape[innermostAxis];
    const auto innermostAxisInitialOutputDim = initialOutputShape[innermostAxis];
    const auto innermostAxisInputOffset = checked_cast<float>(initialInputOffset[innermostAxis]);
    const auto innermostAxisOutputOffset = checked_cast<int>(initialOutputOffset[innermostAxis]);
    const auto scale =
            static_cast<float>(innermostAxisInitialOutputDim) / static_cast<float>(innermostAxisInitialInputDim);

    const auto coordinatesSize = IE::getInterpCoordinatesSize(interpolateOp.getOutput(), innermostAxis);
    const auto lambdasSize = IE::getInterpLambdasSize(interpolateOp.getOutput(), innermostAxis);
    std::vector<int> coordinatesVec(coordinatesSize);
    std::vector<type::float16> lambdasVec(lambdasSize);

    const auto inMemStrides = inOrder.toLogicalOrder(inType.getMemStrides()).raw();
    const auto innermostAxisStride = inMemStrides[innermostAxis].to<Byte>().count();

    const auto mapCoord = getMapCoordFunc(coordMode);

    for (int i = 0; i < innermostAxisInputDim; i++) {
        float inCoord = mapCoord(i + innermostAxisOutputOffset, scale, innermostAxisInitialOutputDim,
                                 innermostAxisInitialInputDim) -
                        innermostAxisInputOffset;
        inCoord = std::clamp(inCoord, 0.0f, static_cast<float>(innermostAxisOuputDim - 1));
        const auto inCoord1 = static_cast<int64_t>(inCoord);
        const auto inCoord2 = std::clamp(inCoord1 + 1, int64_t(0), innermostAxisOuputDim - 1);

        auto lambda1 = inCoord - static_cast<float>(inCoord1);
        auto lambda2 = 1.f - lambda1;
        if (interpolateMode == IE::InterpolateMode::LINEAR_ONNX && inCoord1 == inCoord2) {
            lambda1 = lambda2 = ZERO_FIVE;
        }

        coordinatesVec[i] = checked_cast<int>(inCoord1 * innermostAxisStride);
        lambdasVec[i * 2] = lambda1;
        lambdasVec[i * 2 + 1] = lambda2;
    }

    const auto coordinatesConstType = mlir::RankedTensorType::get({1, 1, 1, coordinatesSize}, getSInt32Type(ctx));
    const auto coordinatesConst =
            interpolateConstant(interpolateOp, rewriter, loc, coordinatesConstType, ArrayRef(coordinatesVec));

    const auto lambdasConstType = mlir::RankedTensorType::get({1, 1, 1, lambdasSize}, mlir::Float16Type::get(ctx));
    const auto lambdasConst = interpolateConstant(interpolateOp, rewriter, loc, lambdasConstType, ArrayRef(lambdasVec));

    rewriter.modifyOpInPlace(interpolateOp, [&] {
        interpolateOp.getCoordinatesMutable().assign(coordinatesConst);
        interpolateOp.getLambdasMutable().assign(lambdasConst);
    });

    return mlir::success();
}

class ComputeInterpolateCoordinatesPass final :
        public VPU::impl::ComputeInterpolateCoordinatesBase<ComputeInterpolateCoordinatesPass> {
public:
    explicit ComputeInterpolateCoordinatesPass(bool enableExplicitDistributionInfoAttr, Logger log)
            : _enableExplicitDistributionInfoAttr(enableExplicitDistributionInfoAttr) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableExplicitDistributionInfoAttr = false;
    void safeRunOnFunc() final;
};

mlir::LogicalResult ComputeInterpolateCoordinatesPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableExplicitDistributionInfoAttr.hasValue()) {
        _enableExplicitDistributionInfoAttr = enableExplicitDistributionInfoAttr.getValue();
        return mlir::success();
    }

    return mlir::success();
}

void ComputeInterpolateCoordinatesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<ComputeInterpolateCoordinates>(&ctx, _enableExplicitDistributionInfoAttr, _log);
    collectOpsAndApplyPatterns(func, std::move(greedyPatterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createComputeInterpolateCoordinatesPass(bool enableExplicitDistributionInfoAttr,
                                                                               Logger log) {
    return std::make_unique<ComputeInterpolateCoordinatesPass>(enableExplicitDistributionInfoAttr, std::move(log));
}
