//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSHAPETO4D
#define GEN_PASS_DEF_CONVERTSHAPETO4D
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t TARGET_TENSOR_DIM = 4;

using MergeMapItem = SmallVector<int64_t>;
using MergeMap = SmallVector<MergeMapItem>;

void alignShapeToReferenceShapeSize(size_t refSize, SmallVector<int64_t>& shape, bool extendOnH) {
    VPUX_THROW_UNLESS(refSize >= shape.size(), "The reference shape size({0}) < shape size({1})", refSize,
                      shape.size());
    const size_t diff = refSize - shape.size();
    if (diff) {
        if (extendOnH) {
            VPUX_THROW_UNLESS(diff < 3 && diff >= 1,
                              "Extend on H does not support reference shape size({0}) and shape size({1})", refSize,
                              shape.size());
            if (diff == 2) {
                shape.insert(shape.end(), 1, 1);
            }
            shape.insert(shape.begin() + 2, 1, 1);
        } else {
            shape.insert(shape.begin(), diff, 1);
        }
    }
}

int64_t getBalancedDimIndexFromShape(SmallVector<int64_t> shape) {
    int64_t dimH = 1;
    int64_t dimW = 1;
    int64_t dimIndex = 0;
    while (!shape.empty()) {
        if (dimW < dimH) {
            dimW *= shape.back();
            shape.pop_back();
        } else {
            dimH *= shape.front();
            shape.erase(shape.begin());
            dimIndex++;
        }
    }
    return dimIndex;
}

SmallVector<int64_t> alignShapeWithDimMap(ArrayRef<int64_t> originShape, const MergeMap& mapper) {
    SmallVector<int64_t> retNewShape;
    for (const auto& dims : mapper) {
        int64_t dimSize = 1;
        for (auto i : dims) {
            dimSize *= originShape[i];
        }
        retNewShape.push_back(dimSize);
    }
    return retNewShape;
}

SmallVector<int64_t> alignShapeTo4D(ArrayRef<int64_t> originShape, const MergeMap& mapper, bool extendOnH) {
    auto newShape = extendOnH ? SmallVector<int64_t>(originShape) : alignShapeWithDimMap(originShape, mapper);
    alignShapeToReferenceShapeSize(TARGET_TENSOR_DIM, newShape, extendOnH);
    return newShape;
}

MergeMap getTrivialMap(size_t size) {
    auto mapper = MergeMap(size);
    std::generate(mapper.begin(), mapper.end(), [counter = 0]() mutable {
        return SmallVector<int64_t>{counter++};
    });
    return mapper;
}

MergeMap getDimMapWithFirstGreater1DimAsC(SmallVector<int64_t> shape) {
    const int64_t maxDim = checked_cast<int64_t>(shape.size());
    // Try to convert great than 4D shape to 3D.
    // In this way, to promise
    //   N always = 1
    //   C always > 1 unless the shape size is 1.
    // eg.
    //   1x1x1x1x1  -> 1x1x1
    //   1x3x9x16x1 -> 3x9x16
    //   3x9x16x1x1 -> 3x9x16
    //   3x9x1x1x16 -> 3x9x16
    //   2x3x4x5    -> 2x12x5
    //   2x3x4x5x6  -> 2x12x30
    //   2x3x4x5x6x7-> 2x60x42
    const auto moreThanOnePredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    const auto firstMoreThanOneIt = std::find_if(shape.begin(), shape.end(), moreThanOnePredicate);
    if (firstMoreThanOneIt == shape.end()) {
        return {};
    }

    MergeMap retMapper;
    const int64_t nextDimCIndex = std::distance(shape.begin(), firstMoreThanOneIt) + 1;
    retMapper.push_back(irange(nextDimCIndex));

    shape.erase(shape.begin(), shape.begin() + nextDimCIndex);
    // Convert shape to 2D, and make the value of 2 Dims close to each other
    const auto splitDimIndex = getBalancedDimIndexFromShape(std::move(shape)) + nextDimCIndex;
    retMapper.push_back(irange(nextDimCIndex, splitDimIndex));
    retMapper.push_back(irange(splitDimIndex, maxDim));
    return retMapper;
}

MergeMap getDimMapGeneric(ArrayRef<int64_t> shape) {
    MergeMap dimMapper;
    if (shape.size() > TARGET_TENSOR_DIM) {
        return getDimMapWithFirstGreater1DimAsC(to_small_vector(shape));
    }
    return getTrivialMap(shape.size());
}

MergeMap getDimMergeMapWith2Inputs(ArrayRef<int64_t> input1, ArrayRef<int64_t> input2) {
    auto shapeSize1 = std::accumulate(input1.begin(), input1.end(), (int64_t)1, std::multiplies<int64_t>());
    auto shapeSize2 = std::accumulate(input2.begin(), input2.end(), (int64_t)1, std::multiplies<int64_t>());
    // Find the origin input and broadcast shape
    //  The large size shape is the origin input
    //  The small size shape is the shape that needs to be broadcast in some planes
    auto maxShape = (shapeSize1 > shapeSize2) ? input1 : input2;
    auto planeShape = (shapeSize1 > shapeSize2) ? input2 : input1;

    auto getMergeMap = [](ArrayRef<int64_t> fullShape, ArrayRef<int64_t> planeShape, auto condition) {
        MergeMap dimMap;
        SmallVector<int64_t> inputDimsTmp;
        for (size_t i = 0; i < fullShape.size(); i++) {
            auto compareVal = condition(i, fullShape);
            if (compareVal == planeShape[i]) {
                inputDimsTmp.push_back(i);
            } else {
                if (inputDimsTmp.size() > 1) {
                    dimMap.push_back(inputDimsTmp);
                }
                inputDimsTmp.clear();
            }
        }
        if (inputDimsTmp.size() > 1) {
            dimMap.push_back(inputDimsTmp);
        }
        return dimMap;
    };

    auto sameDimCondition = [](size_t i, ArrayRef<int64_t> shape) {
        return shape[i];
    };
    auto planeDimCondition = [](size_t, ArrayRef<int64_t>) {
        return 1;
    };

    // Examples:
    //  Merge in plane:
    //      Inputs: tensor<4x3x13x13x2xf16>, tensor<1x1x1x1x1xf16>
    //       Dim(0, 1, 2, 3, 4) can merge together.
    //  Merge in same Dim:
    //      Inputs: tensor<4x3x13x13x2xf16>, tensor<4x3x13x13x2xf16>
    //       Dim(0, 1, 2, 3, 4) can merge together.
    //  Mixed:
    //      Inputs: tensor<4x3x13x13x2xf16>, tensor<1x1x13x13x2xf16>
    //       Dim(0, 1) 4x3 and Dim(2, 3, 4) 13x13x2 can merge together.
    //      Inputs: tensor<1x2x3x4x5x6xf16>, tensor<1x2x1x4x5x1xf16>
    //       Dim(0, 1) 1x2,  Dim(2) 3, Dim(3, 4) 4x5 and Dim(5) 6 can merge together.
    auto calculateMergeMap = [&](ArrayRef<int64_t> fullShape, ArrayRef<int64_t> planeShape) {
        auto mergeInSameDims = getMergeMap(fullShape, planeShape, sameDimCondition);
        auto mergeInPlaneDims = getMergeMap(fullShape, planeShape, planeDimCondition);
        MergeMap dimsCanMerge;
        auto fullShapeSize = checked_cast<int64_t>(fullShape.size());
        for (int64_t dimIndex = 0; dimIndex < fullShapeSize; dimIndex++) {
            auto minIndex = fullShapeSize;
            MergeMap* minVector = nullptr;

            auto getMinimumIndex = [&](MergeMap& dimMapper) {
                if (!dimMapper.empty()) {
                    if (dimMapper.front()[0] < minIndex) {
                        minVector = &dimMapper;
                        minIndex = dimMapper.front()[0];
                    }
                }
            };
            getMinimumIndex(mergeInPlaneDims);
            getMinimumIndex(mergeInSameDims);

            if (dimIndex < minIndex) {
                dimsCanMerge.push_back({dimIndex});
            } else {
                auto& currentDims = minVector->front();
                while (!currentDims.empty() && (currentDims.front() < dimIndex)) {
                    currentDims.erase(currentDims.begin());
                }
                if (!currentDims.empty()) {
                    dimsCanMerge.push_back(currentDims);
                    dimIndex = currentDims.back();
                }
                minVector->erase(minVector->begin());
            }
        }
        return dimsCanMerge;
    };

    auto getSubShape = [](ArrayRef<int64_t> shape, ArrayRef<int64_t> map) {
        SmallVector<int64_t> retShape;
        for (auto& dims : map) {
            retShape.push_back(shape[dims]);
        }
        return retShape;
    };

    MergeMap dimsCanMerge;
    // Corner case:
    //  %4 = IE.Operator(%3, %cst) : tensor<f16>, tensor<f16> -> tensor<f16>
    //  The shape size is 0, and the empty merge map will be 1.
    if (maxShape.empty() && planeShape.empty()) {
        dimsCanMerge.resize(4);
        return dimsCanMerge;
    }

    if (maxShape == planeShape) {
        dimsCanMerge.push_back(irange(static_cast<int64_t>(maxShape.size())));
    } else {
        dimsCanMerge = calculateMergeMap(maxShape, planeShape);
    }
    switch (dimsCanMerge.size()) {
    case 1: {
        dimsCanMerge = getDimMapGeneric(maxShape);
        break;
    }
    case 2: {
        auto expandMapTo3D = [&](auto mapIt) {
            auto newReshapeDim = getBalancedDimIndexFromShape(getSubShape(maxShape, *mapIt));
            SmallVector<int64_t> dimTmp(mapIt->begin(), mapIt->begin() + newReshapeDim);
            mapIt->erase(mapIt->begin(), mapIt->begin() + newReshapeDim);
            dimsCanMerge.insert(mapIt, dimTmp);
        };
        // N always 1 to avoid unroll
        if (dimsCanMerge[1].size() > 1) {
            expandMapTo3D(dimsCanMerge.begin() + 1);
        } else {
            expandMapTo3D(dimsCanMerge.begin());
        }
        break;
    }
    case 4:
        // Direct convert
        break;
    case 3:
        // Add 1 at dim N
        break;
    default:
        VPUX_THROW("The input shape {0}, {1} can't convert to 4D", input1, input2);
        break;
    }
    return dimsCanMerge;
}

MergeMap getDimMergeMapWith3Inputs(ArrayRef<int64_t> input1, ArrayRef<int64_t> inputLow, ArrayRef<int64_t> outLow) {
    // Handle 3 input shapes
    //  input:   AxBxCxDxF
    //  in_low:  1xBx1x1x1
    //  out_low: 1x1xCx1x1
    //  To: (A, B, C, [DxF])
    // vs
    //  input:   AxBxCxDxF
    //  in_low:  1xBx1x1x1
    //  out_low: 1x1x1xDx1
    //  To: (A, B, C, D, F) can't convert to 4D, unsupported.
    const auto moreThanOnePredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };

    auto getDimIdx = [&](ArrayRef<int64_t> dims) -> int64_t {
        auto firstMoreThanOneIt = std::find_if(dims.begin(), dims.end(), moreThanOnePredicate);
        VPUX_THROW_WHEN(firstMoreThanOneIt == dims.end(), "The shape size is 1, should not enter this case.");
        return std::distance(dims.begin(), firstMoreThanOneIt);
    };
    int64_t inDimIndex = getDimIdx(inputLow);
    int64_t outDimIndex = getDimIdx(outLow);

    auto generateDimMap = [](int64_t minIndex, int64_t maxIndex, int64_t size) {
        MergeMap mergeMap;
        if (minIndex > 0) {
            mergeMap.push_back(irange(minIndex));
        }
        mergeMap.push_back({minIndex});
        minIndex++;
        if (minIndex < maxIndex) {
            mergeMap.push_back(irange(minIndex, maxIndex));
        }
        mergeMap.push_back({maxIndex});
        maxIndex++;
        if (maxIndex < size) {
            mergeMap.push_back(irange(maxIndex, size));
        }
        return mergeMap;
    };

    auto fullShapeSize = checked_cast<int64_t>(input1.size());
    MergeMap mergeMapTmp;
    if (inDimIndex < outDimIndex) {
        mergeMapTmp = generateDimMap(inDimIndex, outDimIndex, fullShapeSize);
    } else {
        mergeMapTmp = generateDimMap(outDimIndex, inDimIndex, fullShapeSize);
    }
    auto newShape = alignShapeWithDimMap(input1, mergeMapTmp);
    MergeMap mergeMapRet;
    MergeMapItem item;
    for (int64_t dimIdx = 0; dimIdx < checked_cast<int64_t>(newShape.size()); dimIdx++) {
        item.append(mergeMapTmp[dimIdx]);
        if (newShape[dimIdx] > 1) {
            mergeMapRet.push_back(item);
            item.clear();
        }
    }
    if (!item.empty()) {
        mergeMapRet.back().append(item);
    }
    VPUX_THROW_WHEN(mergeMapRet.size() > 4, "Can't convert the shape to 4D, the converted shape is {0}D",
                    mergeMapRet.size());
    return mergeMapRet;
}

MergeMap extendInputShapeTo4D(IE::FakeQuantizeOp origOp) {
    auto inputLowScaleShape = to_small_vector(getShape(origOp.getInputLow()));
    auto outputLowScaleShape = to_small_vector(getShape(origOp.getOutputLow()));
    const auto inputShape = to_small_vector(getShape(origOp.getInput()));
    const auto ref1ElemShape = SmallVector<int64_t>(inputShape.size(), 1);

    alignShapeToReferenceShapeSize(inputShape.size(), inputLowScaleShape, false);
    alignShapeToReferenceShapeSize(inputShape.size(), outputLowScaleShape, false);

    if (inputLowScaleShape == outputLowScaleShape) {
        return getDimMergeMapWith2Inputs(inputShape, inputLowScaleShape);
    }
    if (ref1ElemShape == inputLowScaleShape) {
        return getDimMergeMapWith2Inputs(inputShape, outputLowScaleShape);
    }
    if (ref1ElemShape == outputLowScaleShape) {
        return getDimMergeMapWith2Inputs(inputShape, inputLowScaleShape);
    }
    return getDimMergeMapWith3Inputs(inputShape, inputLowScaleShape, outputLowScaleShape);
}

mlir::Value reshapeInputWithMergeMap(mlir::PatternRewriter& rewriter, mlir::Location loc, size_t referenceShapeSize,
                                     mlir::Value origInput, const MergeMap& map, bool extendOnH) {
    auto inShape = to_small_vector(getShape(origInput));

    // Note: ensure the rank of the current shape is aligned to the "reference"
    // shape (the shape that was used to calculate the merge map). this
    // guarantees we don't have buffer overflows due to mege map using indices
    // outside of current shape's rank.
    alignShapeToReferenceShapeSize(referenceShapeSize, inShape, extendOnH);

    auto constInputShape = alignShapeTo4D(inShape, map, extendOnH);
    const auto constInputShapeAttr = getIntArrayAttr(rewriter.getContext(), constInputShape);

    return rewriter.createOrFold<IE::ReshapeOp>(loc, origInput, nullptr, false, constInputShapeAttr);
}

void tryAndConvert2NCEShape(SmallVector<int64_t>& shape1, SmallVector<int64_t>& shape2, MergeMap& map) {
    // 4D Multiply shape 1x1x1xM need convert Shape to 1xMx1x1
    //
    // TODO:
    // This logic is a litte same as AdaptShapesForScaleShiftPass.
    // May combine them into 1 pass and abandon the AdaptShapesForScaleShiftPass
    const auto nonTrivialDimPredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    const auto nonTrivialShape1Dims = std::count_if(shape1.begin(), shape1.end(), nonTrivialDimPredicate);
    const auto nonTrivialShape2Dims = std::count_if(shape2.begin(), shape2.end(), nonTrivialDimPredicate);
    // Filter out the Shape 1x1x1x1 and nonTrivialDims > 1 cases
    if ((nonTrivialShape1Dims > 1 || nonTrivialShape2Dims > 1) ||
        (nonTrivialShape1Dims == 0 && nonTrivialShape2Dims == 0)) {
        return;
    }
    auto findFirstNonTrivialIndex = [&](auto shape) {
        const auto firstIt = std::find_if(shape.begin(), shape.end(), nonTrivialDimPredicate);
        return std::distance(shape.begin(), firstIt);
    };
    int64_t firstNonTrivialIndex;
    // Find the first non-trivial index from 2 input shapes
    firstNonTrivialIndex = (findFirstNonTrivialIndex(shape1) <= findFirstNonTrivialIndex(shape2))
                                   ? findFirstNonTrivialIndex(shape1)
                                   : findFirstNonTrivialIndex(shape2);

    // Already at DimC
    if (firstNonTrivialIndex == 1) {
        return;
    }
    if (map.size() < 4) {
        map.insert(map.begin(), 4 - map.size(), {});
    }
    std::swap(shape1[1], shape1[firstNonTrivialIndex]);
    std::swap(shape2[1], shape2[firstNonTrivialIndex]);
    std::swap(map[1], map[firstNonTrivialIndex]);
}

// Merge all adjacent axis and non-axis dimensions
std::pair<SmallVector<int64_t>, SmallVector<int64_t>> getMergedShapeAndAxes(const SmallVector<int64_t>& inputShape,
                                                                            const SmallVector<int64_t>& axes) {
    SmallVector<int64_t> newShape;
    SmallVector<int64_t> newAxes;

    SmallVector<bool> isAxis(inputShape.size(), false);
    for (auto axis : axes) {
        isAxis[axis] = true;
    }

    newShape.push_back(inputShape[0]);
    if (isAxis[0]) {
        newAxes.push_back(0);
    }

    for (size_t i = 1; i < inputShape.size(); i++) {
        if (isAxis[i - 1] == isAxis[i]) {
            newShape.back() *= inputShape[i];
        } else {
            newShape.push_back(inputShape[i]);
            if (isAxis[i]) {
                newAxes.push_back(newShape.size() - 1);
            }
        }
    }

    return {std::move(newShape), std::move(newAxes)};
}

// For TileOp, align the shape or repeats to 4D
SmallVector<int64_t> alignTileShapeRepeatsTo4D(SmallVector<int64_t> origShape) {
    const auto origRank = static_cast<int64_t>(origShape.size());
    SmallVector<int64_t> newShape;

    if (origRank > TARGET_TENSOR_DIM) {
        for (int64_t i = 0; i < origRank - TARGET_TENSOR_DIM; i++) {
            if (origShape[i] != 1) {
                VPUX_THROW("The dims from range [0, origRank - TARGET_TENSOR_DIM] are not equal to 1");
            }

            if (i == origRank - TARGET_TENSOR_DIM - 1) {
                newShape.append(origShape.begin() + i + 1, origShape.end());
            }
        }
    } else {
        newShape.append(origShape.begin(), origShape.end());

        for (int64_t i = 0; i < TARGET_TENSOR_DIM - origRank; i++) {
            newShape.insert(newShape.begin(), 1);
        }
    }

    return newShape;
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>> squeezeRankForTile(IE::TileOp origOp) {
    // Squeeze the input/repeat-value ranks to support more cases, for example:
    // tensor<1x1x8xf16>, [16, 1, 1] after squeezed tensor<1x8xf16>, [16, 1]
    // tensor<1x1x1x1x8xf16>, [1, 1, 1, 16, 1] after squeezed tensor<1x8xf16>, [16, 1]
    // tensor<1024x1x1x1x8xf16>, [1, 1, 1, 16, 1] after squeezed tensor<1024x1x8xf16>, [1, 16, 1]
    // tensor<1024x128x1x1x8xf16>, [1, 1, 1, 16, 1] after squeezed tensor<1024x128x1x8xf16>, [1, 1, 16, 1]
    auto inputType = origOp.getInput().getType().cast<NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    auto repeatValues = parseIntArrayAttr<int64_t>(origOp.getRepeatsValuesAttr());
    auto squeezeVec = [](const SmallVector<int64_t>& arr) {
        auto it = std::find_if(arr.begin(), arr.end(), [](int64_t x) {
            return x != 1;
        });
        return SmallVector<int64_t>(it, arr.end());
    };
    SmallVector<int64_t> removedRepeatValue = squeezeVec(repeatValues);

    SmallVector<int64_t> newInputShape;
    auto inputSize = inputShape.size();
    auto outputSize = outputShape.size();
    auto inputOutputSizeGap = outputSize - inputSize;
    for (size_t i = 0; i < outputSize; ++i) {
        auto inDimVal = i < inputOutputSizeGap ? 1 : inputShape[Dim(i - inputOutputSizeGap)];
        if (outputShape[Dim(i)] == 1) {
            if (removedRepeatValue.size() > inputSize) {
                removedRepeatValue.erase(removedRepeatValue.begin() + i);
            }
            continue;
        }
        newInputShape.push_back(inDimVal);
    }

    return {std::move(newInputShape), std::move(removedRepeatValue)};
}

//
// ConvertShapeTo4DPass
//

class ConvertShapeTo4DPass final : public IE::impl::ConvertShapeTo4DBase<ConvertShapeTo4DPass> {
public:
    explicit ConvertShapeTo4DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// GenericConverter
//

mlir::LogicalResult convertGeneric(mlir::Operation* origOp, mlir::ValueRange operands,
                                   mlir::ConversionPatternRewriter& rewriter, const mlir::TypeConverter& typeConverter,
                                   Logger log) {
    log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::IRMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(typeConverter.convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

template <class ConcreteOp>
class GenericConverter final : public mlir::OpConversionPattern<ConcreteOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<ConcreteOp>::OpAdaptor;

public:
    GenericConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ConcreteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto* typeConverter = this->getTypeConverter();
        VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

        if (origOp->getOperands().size() == 2) {
            return convertWith2Inputs(origOp, newArgs.getOperands(), rewriter);
        }
        return convertGeneric(origOp, newArgs.getOperands(), rewriter, *typeConverter, _log);
    }

private:
    mlir::LogicalResult convertWith2Inputs(ConcreteOp origOp, mlir::ValueRange operands,
                                           mlir::ConversionPatternRewriter& rewriter) const;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::convertWith2Inputs(ConcreteOp origOp, mlir::ValueRange operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::Value input1 = origOp->getOperand(0);
    mlir::Value input2 = origOp->getOperand(1);

    const auto shapeOne = input1.getType().template cast<vpux::NDTypeInterface>().getShape();
    const auto shapeTwo = input2.getType().template cast<vpux::NDTypeInterface>().getShape();

    auto shapeOneVector = to_small_vector(shapeOne);
    auto shapeTwoVector = to_small_vector(shapeTwo);
    auto origOutputShape = getShape(origOp.getOutput());

    const auto elemType = origOp.getOutput().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(elemType);
    const auto nonTrivialDimPredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    const auto nonTrivialOrigOutputShapeDims =
            std::count_if(origOutputShape.begin(), origOutputShape.end(), nonTrivialDimPredicate);
    auto findFirstNonTrivialIndex = [&](auto shape) {
        const auto firstIt = std::find_if(shape.begin(), shape.end(), nonTrivialDimPredicate);
        return std::distance(shape.begin(), firstIt);
    };

    auto firstNonTrivialIndex = findFirstNonTrivialIndex(origOutputShape);
    auto extendOnH = false;
    // If the dim on firstNonTrivalIndex is aligned, extending on H is more friendly to NCE ops.
    // Examples:
    // 1x512x28 -> 1x512x1x28
    // 1x512 -> 1x512x1x1
    if (origOutputShape.size() < 4 && nonTrivialOrigOutputShapeDims > 0 && firstNonTrivialIndex == 1 &&
        origOutputShape[Dim(firstNonTrivialIndex)] % alignment == 0) {
        extendOnH = true;
    }
    if (mlir::isa<IE::AddOp>(origOp) && nonTrivialOrigOutputShapeDims > 1) {
        extendOnH = false;
    }

    // Align dims
    if (shapeOneVector.size() != shapeTwoVector.size()) {
        extendOnH = false;
        auto maxSize = std::max(shapeOneVector.size(), shapeTwoVector.size());
        auto& smallShape = (shapeOneVector.size() > shapeTwoVector.size()) ? shapeTwoVector : shapeOneVector;
        auto& bigShape = (shapeOneVector.size() > shapeTwoVector.size()) ? shapeOneVector : shapeTwoVector;
        SmallVector<int64_t> expanedShape(maxSize, 1);
        if (origOp->hasAttr("auto_broadcast")) {
            alignShapeToReferenceShapeSize(bigShape.size(), smallShape, false);
        } else {
            // Some operations need to map their channels first. e.g. PRelu
            if ((smallShape.size() == 1) && (smallShape[0] == bigShape[1])) {
                expanedShape[1] = smallShape[0];
                smallShape.swap(expanedShape);
            } else {
                alignShapeToReferenceShapeSize(bigShape.size(), smallShape, false);
            }
        }
    }

    auto dimsCanMerge = getDimMergeMapWith2Inputs(shapeOneVector, shapeTwoVector);
    auto newInputShape1 = alignShapeTo4D(shapeOneVector, dimsCanMerge, extendOnH);
    auto newInputShape2 = alignShapeTo4D(shapeTwoVector, dimsCanMerge, extendOnH);

    if (std::is_same<IE::MultiplyOp, ConcreteOp>::value) {
        tryAndConvert2NCEShape(newInputShape1, newInputShape2, dimsCanMerge);
    }
    auto newIn1 = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_lhs"), operands[0], nullptr, false,
                                                       getIntArrayAttr(this->getContext(), newInputShape1));
    auto newIn2 = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_rhs"), operands[1], nullptr, false,
                                                       getIntArrayAttr(this->getContext(), newInputShape2));

    SmallVector<mlir::Value> newOperands;
    newOperands.push_back(newIn1);
    newOperands.push_back(newIn2);
    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), newOperands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    SmallVector<mlir::Value> newResults;
    for (auto result : newOp->getResults()) {
        auto resultNDI = result.getType().template cast<vpux::NDTypeInterface>();
        auto resultShape = to_small_vector(resultNDI.getShape());
        result.setType(resultNDI.changeShape(ShapeRef(alignShapeTo4D(resultShape, dimsCanMerge, extendOnH))));
        const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), resultShape);
        auto resultReshapeOp = rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, StringLiteral("reshape_out_{0}"), newResults.size()), result, nullptr, false,
                outputShapeAttr);
        if (result == resultReshapeOp) {
            newResults.push_back(result);
        } else {
            newResults.push_back(resultReshapeOp.template getDefiningOp<IE::ReshapeOp>().getOutput());
        }
    }

    rewriter.replaceOp(origOp, newResults);
    return mlir::success();
}

//
// FakeQuantizeConverter
//

class FakeQuantizeConverter final : public mlir::OpConversionPattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::FakeQuantizeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantizeConverter::matchAndRewrite(IE::FakeQuantizeOp origOp, OpAdaptor,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto mergeMap = extendInputShapeTo4D(origOp);

    const auto referenceShapeSize = getShape(origOp.getInput()).size();
    const auto inputLow = reshapeInputWithMergeMap(rewriter, takeOpLoc(origOp, "reshape_in_low"), referenceShapeSize,
                                                   origOp.getInputLow(), mergeMap, false);
    const auto inputHigh = reshapeInputWithMergeMap(rewriter, takeOpLoc(origOp, "reshape_in_high"), referenceShapeSize,
                                                    origOp.getInputHigh(), mergeMap, false);
    const auto outputLow = reshapeInputWithMergeMap(rewriter, takeOpLoc(origOp, "reshape_out_low"), referenceShapeSize,
                                                    origOp.getOutputLow(), mergeMap, false);
    const auto outputHigh = reshapeInputWithMergeMap(rewriter, takeOpLoc(origOp, "reshape_out_high"),
                                                     referenceShapeSize, origOp.getOutputHigh(), mergeMap, false);

    auto inputReshape = reshapeInputWithMergeMap(rewriter, takeOpLoc(origOp, "reshape_in"), referenceShapeSize,
                                                 origOp.getInput(), mergeMap, false);

    auto newFakeQuantizeOp = rewriter.create<IE::FakeQuantizeOp>(
            takeOpLoc(origOp, "fq_in"), inputReshape, inputLow, inputHigh, outputLow, outputHigh,
            origOp.getLevelsAttr(), origOp.getLowFpTypeAttr(), origOp.getAutoBroadcastAttr());

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newFakeQuantizeOp.getOutput(), nullptr, false,
                                                                 outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::FakeQuantize'", getDebugName());

    return mlir::success();
}

//
// TopKOpConverter
//

class TopKOpConverter final : public mlir::OpConversionPattern<IE::TopKOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<IE::TopKOp>::OpAdaptor;

public:
    TopKOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::TopKOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKOpConverter::matchAndRewrite(IE::TopKOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    const auto origInType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const int64_t origInRank = origInType.getRank();
    int64_t axis = origOp.getAxis();
    if (axis < 0) {
        axis += origInRank;
    }

    // Deduce the new TopK aix from map table
    const auto inShape = to_small_vector(getShape(origOp.getInput()));

    MergeMap mergeMap;
    SmallVector<int64_t> tempMap;
    int64_t newAxis = 0;
    if (axis > 0) {
        mergeMap.push_back(irange(axis));
        newAxis = 1;
    }
    mergeMap.push_back({axis});
    if (axis < origInRank - 1) {
        mergeMap.push_back(irange(axis + 1, origInRank));
    }
    // The mergeMap's Max Size is 3
    auto delta4D = 4 - mergeMap.size();
    mergeMap.insert(mergeMap.begin(), delta4D, {});
    newAxis += delta4D;

    const auto newAxisAttr = getIntAttr(origOp->getContext(), newAxis);

    const auto newInShapeAttr = getIntArrayAttr(this->getContext(), alignShapeTo4D(inShape, mergeMap, false));
    const auto newInReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                                   nullptr, false, newInShapeAttr);

    auto newTopKOp = rewriter.create<IE::TopKOp>(origOp->getLoc(), newInReshape, origOp.getK(), origOp.getKValueAttr(),
                                                 newAxisAttr, origOp.getModeAttr(), origOp.getSortAttr(),
                                                 origOp.getElementTypeAttr());

    for (auto indexResult : origOp->getResults() | indexed) {
        auto idx = checked_cast<unsigned>(indexResult.index());
        auto origResult = indexResult.value();
        const auto outputShapeAttr = getIntArrayAttr(this->getContext(), getShape(origResult));
        const auto newOutputReshape =
                rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("reshape_out_{0}"), idx),
                                                     newTopKOp->getResult(idx), nullptr, false, outputShapeAttr);
        origResult.replaceAllUsesWith(newOutputReshape);
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// Mvn6OpConverter
//

class Mvn6Converter final : public mlir::OpConversionPattern<IE::MVN6Op> {
public:
    Mvn6Converter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::MVN6Op>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MVN6Op origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult Mvn6Converter::matchAndRewrite(IE::MVN6Op origOp, OpAdaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::MVN6Op Operation '{1}'", getDebugName(), origOp->getLoc());
    const auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = SmallVector<int64_t>(inType.getShape().raw());
    const auto inRank = inShape.size();
    auto inAxes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());

    SmallVector<int64_t> newShape;
    SmallVector<int64_t> newAxes;
    SmallVector<int64_t> actShape;     // optional scale/bias
    SmallVector<int64_t> newActShape;  // 4D version of 'actShape'

    if (origOp.getScale()) {
        actShape = SmallVector<int64_t>(origOp.getScale().getType().getShape());
    } else if (origOp.getBias()) {
        actShape = SmallVector<int64_t>(origOp.getBias().getType().getShape());
    }

    if (inRank < 4) {
        // insert leading 1s up to 4D and ajust axes accordingly
        auto newDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inRank);
        auto to4DShape = [=](ArrayRef<int64_t> iShape, SmallVector<int64_t>& oShape) {
            oShape.insert(oShape.end(), newDims, 1);
            oShape.append(iShape.begin(), iShape.end());
        };
        to4DShape(inShape, newShape);  // main input
        if (!actShape.empty()) {       // optional inputs
            to4DShape(actShape, newActShape);
        }
        // increment 'axes'
        newAxes = std::move(inAxes);
        std::for_each(newAxes.begin(), newAxes.end(), [newDims](int64_t& axis) {
            axis += newDims;
        });
    } else if (inRank == 5) {
        VPUX_THROW_WHEN(origOp.getScale() || origOp.getBias(), "Unimplemented 5D->4D convert of MVN6 with scale/bias");
        // Find and merge two nearby axes of same type (either NORM or non-NORM)
        auto isNormAxis = [inAxes](auto curDim) {
            return std::find(inAxes.begin(), inAxes.end(), curDim) != inAxes.end();
        };
        SmallVector<int64_t> axes5D(inRank);
        std::iota(axes5D.begin(), axes5D.end(), 0);

        auto checkSame = [&](auto curDim, auto nxtDim) {
            auto curType = isNormAxis(curDim);
            auto nxtType = isNormAxis(nxtDim);
            return (curType == nxtType);
        };
        const auto mergeIt = std::adjacent_find(axes5D.begin(), axes5D.end(), checkSame);
        VPUX_THROW_WHEN(mergeIt == axes5D.end(), "MVN6 5D->4D failed : cannot find 2 adjacent dims of same type");
        const auto mergeDim = checked_cast<int64_t>(std::distance(axes5D.begin(), mergeIt));

        //=> new 'shape'
        newShape = decltype(newShape){inShape.begin(), inShape.end()};
        newShape[mergeDim] *= newShape[mergeDim + 1];
        newShape.erase(newShape.begin() + mergeDim + 1);

        // => new 'axes'
        newAxes = std::move(inAxes);
        newAxes.erase(std::remove(newAxes.begin(), newAxes.end(), mergeDim + 1), newAxes.end());
        std::for_each(newAxes.begin(), newAxes.end(), [mergeDim](auto& axis) {
            axis = axis > mergeDim ? (axis - 1) : axis;
        });

        VPUX_THROW_UNLESS(newShape.size() == TARGET_TENSOR_DIM, "MVN6 5D->4D conversion failed");
    } else {
        VPUX_THROW("Unimplemented {0}D->4D convert", inRank);
    }

    const auto newShapeAttr = getIntArrayAttr(getContext(), newShape);
    const auto newActShapeAttr = getIntArrayAttr(getContext(), newActShape);
    auto inReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(), nullptr,
                                                          false, newShapeAttr);

    // reshape optional inputs if present
    mlir::Value scale = origOp.getScale();
    mlir::Value bias = origOp.getBias();
    if (scale) {
        scale = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_scale"), scale, nullptr, false,
                                                     newActShapeAttr);
    }
    if (bias) {
        bias = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_bias"), bias, nullptr, false,
                                                    newActShapeAttr);
    }

    const auto axisAttr = getIntArrayAttr(getContext(), newAxes);
    auto newMvnOp = rewriter.create<IE::MVN6Op>(origOp->getLoc(), inReshape, scale, bias, origOp.getAxes(), axisAttr,
                                                origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr(),
                                                origOp.getEpsModeAttr());

    const auto outShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newMvnOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::MVN6Op'", getDebugName());

    return mlir::success();
}

//
// RMSOpConverter
//

class RMSOpConverter final : public mlir::OpConversionPattern<IE::RMSOp> {
public:
    RMSOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::RMSOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::RMSOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RMSOpConverter::matchAndRewrite(IE::RMSOp origOp, OpAdaptor,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto gammaType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(1).getType());

    // input
    auto newInShape = to_small_vector(inType.getShape());
    auto inRank = newInShape.size();
    const int64_t newInDims = TARGET_TENSOR_DIM - inRank;
    newInShape.insert(newInShape.begin(), newInDims, 1);

    // gamma
    auto newGammaShape = to_small_vector(gammaType.getShape());
    auto gammaRank = newGammaShape.size();
    const int64_t newGammaDims = TARGET_TENSOR_DIM - gammaRank;
    newGammaShape.insert(newGammaShape.begin(), newGammaDims, 1);

    const auto newInShapeAttr = getIntArrayAttr(origOp->getContext(), newInShape);
    const auto newGammaShapeAttr = getIntArrayAttr(origOp->getContext(), newGammaShape);
    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));

    const auto inReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                                nullptr, false, newInShapeAttr);
    const auto gammaReshape = rewriter.createOrFold<IE::ReshapeOp>(
            takeOpLoc(origOp, "reshape_gamma"), origOp.getGamma(), nullptr, false, newGammaShapeAttr);
    auto newRMSOp = rewriter.create<IE::RMSOp>(origOp->getLoc(), inReshape, gammaReshape, origOp.getEpsilonAttr());
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newRMSOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

//
// ReduceConverter
//

template <class ReduceOp>
class ReduceConverter final : public mlir::OpConversionPattern<ReduceOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<ReduceOp>::OpAdaptor;

public:
    ReduceConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ReduceOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ReduceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ReduceOp>
mlir::LogicalResult ReduceConverter<ReduceOp>::matchAndRewrite(ReduceOp origOp, OpAdaptor,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto inType = origOp->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
    auto newShape = to_small_vector(inType.getShape());
    auto newAxes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    auto inRank = newShape.size();

    if (inRank > TARGET_TENSOR_DIM) {
        std::tie(newShape, newAxes) = getMergedShapeAndAxes(newShape, newAxes);
        inRank = newShape.size();
    }

    if (inRank < TARGET_TENSOR_DIM) {
        const int64_t newDims = TARGET_TENSOR_DIM - inRank;
        newShape.insert(newShape.begin(), newDims, 1);
        for (auto& axis : newAxes) {
            axis += newDims;
        }
    }

    const auto newShapeAttr = getIntArrayAttr(origOp->getContext(), newShape);
    const auto axisValueAttr = getIntArrayAttr(origOp->getContext(), newAxes);
    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));

    const auto inReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                                nullptr, false, newShapeAttr);
    auto newReduceOp =
            rewriter.create<ReduceOp>(origOp->getLoc(), inReshape, /*axes*/ nullptr, axisValueAttr, /*keepDims*/ true);
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newReduceOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

template <class ReduceOp>
auto isLegalReduceOp(ReduceOp reduceOp) {
    const auto inShape = reduceOp.getOperand(0).getType().template cast<vpux::NDTypeInterface>().getShape();
    const auto outShape = reduceOp.getResult().getType().template cast<vpux::NDTypeInterface>().getShape();
    if (inShape.size() == TARGET_TENSOR_DIM && outShape.size() == TARGET_TENSOR_DIM) {
        return true;
    }

    const auto axes = parseIntArrayAttr<int64_t>(reduceOp.getAxesValue().value());
    const auto mergedInputShape = getMergedShapeAndAxes(to_small_vector(inShape), axes).first;
    return mergedInputShape.size() > TARGET_TENSOR_DIM;
};

//
// StridedSliceConverter
//

class StridedSliceConverter final : public mlir::OpConversionPattern<IE::StridedSliceOp> {
public:
    StridedSliceConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::StridedSliceOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StridedSliceConverter::matchAndRewrite(IE::StridedSliceOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::StridedSliceOp Operation '{1}'", getDebugName(), origOp->getLoc());

    SmallVector<int64_t> newInputShape;

    auto begins = parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value());
    auto ends = parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value());
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value());
    auto beginMask = parseIntArrayAttr<int64_t>(origOp.getBeginMask());
    auto endMask = parseIntArrayAttr<int64_t>(origOp.getEndMask());

    SmallVector<int64_t> newAxisMask;
    SmallVector<int64_t> shrinkAxisMask;
    SmallVector<int64_t> ellipsisMask;

    if ((!origOp.getNewAxisMask().empty()) && (!origOp.getShrinkAxisMask().empty()) &&
        (!origOp.getEllipsisMask().empty())) {  // in the < 4D cases, if newAxisMask, shrinkAxisMask,
                                                // ellipsisMask are nullptr, they are filled with zeros in
                                                // ResolveStridedSlice pass, but this is not happening for 5D cases.
        newAxisMask = parseIntArrayAttr<int64_t>(origOp.getNewAxisMask());
        shrinkAxisMask = parseIntArrayAttr<int64_t>(origOp.getShrinkAxisMask());
        ellipsisMask = parseIntArrayAttr<int64_t>(origOp.getEllipsisMask());
    }

    const auto origType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto origRank = origType.getRank();
    const auto origShape = origType.getShape();

    if (origRank > TARGET_TENSOR_DIM) {
        SmallVector<int64_t> newBeginAttrShape;
        SmallVector<int64_t> newEndAttrShape;
        SmallVector<int64_t> newStridesAttrShape;
        SmallVector<int64_t> newBeginMaskAttrShape;
        SmallVector<int64_t> newEndMaskAttrShape;
        SmallVector<int64_t> newAxisAttrShape;
        SmallVector<int64_t> newShrinkAxisAttrShape;
        SmallVector<int64_t> newEllipsisAttrShape;

        for (int i = 0; i < origRank - TARGET_TENSOR_DIM; i++) {
            if (origRank > TARGET_TENSOR_DIM && origShape[Dim(i)] == 1) {
                if (i == origRank - TARGET_TENSOR_DIM - 1) {
                    newInputShape.append(origShape.begin() + i + 1, origShape.end());
                    std::copy(begins.begin() + i + 1, begins.end(), std::back_inserter(newBeginAttrShape));
                    std::copy(ends.begin() + i + 1, ends.end(), std::back_inserter(newEndAttrShape));
                    std::copy(strides.begin() + i + 1, strides.end(), std::back_inserter(newStridesAttrShape));
                    std::copy(beginMask.begin() + i + 1, beginMask.end(), std::back_inserter(newBeginMaskAttrShape));
                    std::copy(endMask.begin() + i + 1, endMask.end(), std::back_inserter(newEndMaskAttrShape));
                    if ((!origOp.getNewAxisMask().empty()) && (!origOp.getShrinkAxisMask().empty()) &&
                        (!origOp.getEllipsisMask().empty())) {
                        std::copy(newAxisMask.begin() + i + 1, newAxisMask.end(), std::back_inserter(newAxisAttrShape));
                        std::copy(shrinkAxisMask.begin() + i + 1, shrinkAxisMask.end(),
                                  std::back_inserter(newShrinkAxisAttrShape));
                        std::copy(ellipsisMask.begin() + i + 1, ellipsisMask.end(),
                                  std::back_inserter(newEllipsisAttrShape));
                    } else {
                        newAxisAttrShape = {0, 0, 0, 0};
                        newShrinkAxisAttrShape = {0, 0, 0, 0};
                        newEllipsisAttrShape = {0, 0, 0, 0};
                    }
                }
            } else {
                VPUX_THROW("The dims from range [0, origRank - TARGET_TENSOR_DIM] are not equal to 1");
            }
        }

        origType.changeShape(ShapeRef(newInputShape));

        const auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);
        const auto newBeginAttrShapeAttr = getIntArrayAttr(getContext(), newBeginAttrShape);
        const auto newEndAttrShapeAttr = getIntArrayAttr(getContext(), newEndAttrShape);
        const auto newStridesAttrShapeAttr = getIntArrayAttr(getContext(), newStridesAttrShape);
        const auto newBeginMaskAttrShapeAttr = getIntArrayAttr(getContext(), newBeginMaskAttrShape);
        const auto newEndMaskAttrShapeAttr = getIntArrayAttr(getContext(), newEndMaskAttrShape);

        auto newAxisAttrShapeAttr = getIntArrayAttr(getContext(), newAxisAttrShape);
        auto newShrinkAttrShapeAttr = getIntArrayAttr(getContext(), newShrinkAxisAttrShape);
        auto newEllipsisAttrShapeAttr = getIntArrayAttr(getContext(), newEllipsisAttrShape);

        auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), newArgs.getInput(),
                                                                 nullptr, false, newInputShapeAttr);

        auto newStridedSliceOp = rewriter.create<IE::StridedSliceOp>(
                takeOpLoc(origOp, "as_strided_slice"), inputReshape, origOp.getBegins(), origOp.getEnds(),
                origOp.getStrides(), newBeginAttrShapeAttr, newEndAttrShapeAttr, newStridesAttrShapeAttr,
                newBeginMaskAttrShapeAttr, newEndMaskAttrShapeAttr, newAxisAttrShapeAttr, newShrinkAttrShapeAttr,
                newEllipsisAttrShapeAttr);

        const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
        auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newStridedSliceOp.getOutput(), nullptr,
                                                                     false, outputShapeAttr);
        extendOpLoc(outReshape, "reshape_out");

    } else {
        newInputShape.append(origShape.begin(), origShape.end());

        for (int64_t i = 0; i < TARGET_TENSOR_DIM - origRank; ++i) {
            newInputShape.insert(newInputShape.end(), 1);
            begins.insert(begins.end(), 0);
            ends.insert(ends.end(), 1);
            strides.insert(strides.end(), 1);
            beginMask.insert(beginMask.end(), 0);
            endMask.insert(endMask.end(), 0);
            newAxisMask.insert(newAxisMask.end(), 0);
            shrinkAxisMask.insert(shrinkAxisMask.end(), 0);
            ellipsisMask.insert(ellipsisMask.end(), 0);
        }

        const auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);
        const auto newBeginAttrShapeAttr = getIntArrayAttr(getContext(), begins);
        const auto newEndAttrShapeAttr = getIntArrayAttr(getContext(), ends);
        const auto newStridesAttrShapeAttr = getIntArrayAttr(getContext(), strides);
        const auto newBeginMaskAttrShapeAttr = getIntArrayAttr(getContext(), beginMask);
        const auto newEndMaskAttrShapeAttr = getIntArrayAttr(getContext(), endMask);

        auto newAxisAttrShapeAttr = getIntArrayAttr(getContext(), newAxisMask);
        auto newShrinkAttrShapeAttr = getIntArrayAttr(getContext(), shrinkAxisMask);
        auto newEllipsisAttrShapeAttr = getIntArrayAttr(getContext(), ellipsisMask);

        auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                                 nullptr, false, newInputShapeAttr);

        auto newStridedSliceOp = rewriter.create<IE::StridedSliceOp>(
                takeOpLoc(origOp, "as_strided_slice"), inputReshape, origOp.getBegins(), origOp.getEnds(),
                origOp.getStrides(), newBeginAttrShapeAttr, newEndAttrShapeAttr, newStridesAttrShapeAttr,
                newBeginMaskAttrShapeAttr, newEndMaskAttrShapeAttr, newAxisAttrShapeAttr, newShrinkAttrShapeAttr,
                newEllipsisAttrShapeAttr);

        const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
        auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newStridedSliceOp.getOutput(), nullptr,
                                                                     false, outputShapeAttr);
        extendOpLoc(outReshape, "reshape_out");
    }

    _log.trace("[{0}] Replaced with 'IE::StridedSlice'", getDebugName());

    return mlir::success();
}

//
// TileConverter
//

class TileConverter final : public mlir::OpConversionPattern<IE::TileOp> {
public:
    TileConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::TileOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileConverter::matchAndRewrite(IE::TileOp origOp, OpAdaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::TileOp Operation '{1}'", getDebugName(), origOp->getLoc());

    SmallVector<int64_t> squeezedInShape, squeezedValue;
    std::tie(squeezedInShape, squeezedValue) = squeezeRankForTile(origOp);

    auto newInputShape = alignTileShapeRepeatsTo4D(std::move(squeezedInShape));

    // Build input ReshapeOp
    const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), std::move(newInputShape));
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);
    auto repeatsOnNewShape = alignTileShapeRepeatsTo4D(std::move(squeezedValue));

    // Update the TileOp
    const auto repeatsOnNewShapeAttr = getIntArrayAttr(rewriter.getContext(), repeatsOnNewShape);
    auto newTileOp = rewriter.create<IE::TileOp>(origOp.getLoc(), inputReshape, nullptr, repeatsOnNewShapeAttr);

    // Reshape to original output shape
    const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), getShape(origOp.getOutput()));
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newTileOp.getOutput(), nullptr, false, outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::TileOp'", getDebugName());

    return mlir::success();
}

//
// LSTMGatesConverter
//

class LSTMGatesConverter final : public mlir::OpConversionPattern<IE::LSTMGatesOp> {
public:
    LSTMGatesConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::LSTMGatesOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMGatesOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMGatesConverter::matchAndRewrite(IE::LSTMGatesOp origOp, OpAdaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::LSTMGatesOp Operation '{1}'", getDebugName(), origOp->getLoc());

    // Build input ReshapeOp
    SmallVector<mlir::Value> newInputs;
    for (const auto& origInput : origOp.getInputs()) {
        const auto origInputType = origInput.getType().cast<vpux::NDTypeInterface>();
        SmallVector<int64_t> origInputShape = to_small_vector(origInputType.getShape());
        const auto newInputShape = alignTileShapeRepeatsTo4D(std::move(origInputShape));
        const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newInputShape);

        auto inputReshape =
                rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origInput, nullptr, false, newInputShapeAttr);

        newInputs.emplace_back(inputReshape);
    }

    // Update the LSTMGatesOp
    auto newLSTMGatesOp = rewriter.create<IE::LSTMGatesOp>(origOp.getLoc(), newInputs[0], newInputs[1]);

    // Reshape to original output shape
    for (const auto& output : origOp.getOutputs() | indexed) {
        const auto idx = checked_cast<unsigned>(output.index());
        auto origOutput = output.value();
        const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), getShape(origOutput));

        auto newOutputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), newLSTMGatesOp.getOutputs()[idx],
                                                                     nullptr, false, outputShapeAttr);
        origOutput.replaceAllUsesWith(newOutputReshape);
    }

    rewriter.eraseOp(origOp);

    _log.trace("[{0}] Replaced with 'IE::LSTMGatesOp'", getDebugName());

    return mlir::success();
}

//
// LSTMCellConverter
//

class LSTMCellConverter final : public mlir::OpConversionPattern<IE::LSTMCellOp> {
public:
    LSTMCellConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::LSTMCellOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellConverter::matchAndRewrite(IE::LSTMCellOp origOp, OpAdaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::LSTMCellOp Operation '{1}'", getDebugName(), origOp->getLoc());

    // Build input ReshapeOp
    SmallVector<mlir::Value> newInputs;
    for (const auto& origInput : origOp.getInputs()) {
        const auto origInputType = origInput.getType().cast<vpux::NDTypeInterface>();
        SmallVector<int64_t> origInputShape = to_small_vector(origInputType.getShape());
        const auto newInputShape = alignTileShapeRepeatsTo4D(std::move(origInputShape));
        const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newInputShape);

        auto inputReshape =
                rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origInput, nullptr, false, newInputShapeAttr);

        newInputs.emplace_back(inputReshape);
    }

    // Update the LSTMCellOp
    auto newLSTMCellOp =
            rewriter.create<IE::LSTMCellOp>(origOp.getLoc(), newInputs[0], newInputs[1], newInputs[2], newInputs[3],
                                            newInputs[4], newInputs[5], origOp.getHiddenSizeAttr());

    // Reshape to original output shape
    for (const auto& output : origOp.getOutputs() | indexed) {
        const auto idx = checked_cast<unsigned>(output.index());
        auto origOutput = output.value();
        const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), getShape(origOutput));

        auto newOutputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), newLSTMCellOp.getOutputs()[idx],
                                                                     nullptr, false, outputShapeAttr);
        origOutput.replaceAllUsesWith(newOutputReshape);
    }

    rewriter.eraseOp(origOp);

    _log.trace("[{0}] Replaced with 'IE::LSTMCellOp'", getDebugName());

    return mlir::success();
}

//
// LSTMSequenceConverter
//

class LSTMSequenceConverter final : public mlir::OpConversionPattern<IE::LSTMSequenceOp> {
public:
    LSTMSequenceConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::LSTMSequenceOp>(typeConverter, ctx), _log(std::move(log)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMSequenceConverter::matchAndRewrite(IE::LSTMSequenceOp origOp, OpAdaptor,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();

    auto createDynamicReshape = [&](mlir::Value value, ShapeRef newShape, std::string_view suffix) -> mlir::Value {
        const auto valueShape = getShape(value);
        auto newInputDataShape = to_small_vector(valueShape);
        auto newInputDataBounds = parseIntArrayAttr<int64_t>(getBounds(value));

        const auto newInputDataShapeAttr = getIntArrayAttr(ctx, newInputDataShape);
        const auto newInputDataBoundsAttr = getIntArrayAttr(ctx, newInputDataBounds);

        const auto newInputDataShapeRank = checked_cast<int64_t>(newShape.size());
        const auto dataType = mlir::RankedTensorType::get({newInputDataShapeRank}, getSInt32Type(ctx));
        auto newInputDataShapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(newShape), -1);

        const auto shapeTensor =
                Const::createConst(rewriter, value.getLoc(), dataType, ArrayRef(newInputDataShapeValues));
        return rewriter.createOrFold<IE::DynamicReshapeOp>(appendLoc(value.getLoc(), suffix), value, shapeTensor,
                                                           newInputDataShapeAttr, newInputDataBoundsAttr);
    };

    auto reshapeValue = [&](mlir::Value value, ShapeRef newShape, const std::string& suffix) -> mlir::Value {
        const auto valueShape = getShape(value);
        if (valueShape == newShape) {
            return value;
        }

        if (valueShape.isDynamic()) {
            return createDynamicReshape(value, newShape, suffix);
        } else {
            return rewriter.createOrFold<IE::ReshapeOp>(appendLoc(value.getLoc(), suffix), value, nullptr, false,
                                                        getIntArrayAttr(ctx, newShape));
        }
    };

    const auto initialHiddenStateShape = getShape(origOp.getInitialHiddenState());
    const auto batchSize = initialHiddenStateShape[Dim(0)];
    const auto numDirections = initialHiddenStateShape[Dim(1)];
    const auto hiddenSize = initialHiddenStateShape[Dim(2)];
    auto sequenceLength = origOp.getSequenceLength().has_value() ? origOp.getSequenceLength().value() : 1;

    const Shape newInitialHiddenStateShape{batchSize, numDirections, 1, hiddenSize};
    const Shape newInitialCellStateShape{batchSize, numDirections, 1, hiddenSize};
    const Shape newRecurrenceWeightsShape{numDirections, 4, hiddenSize, hiddenSize};
    Shape newInputDataShape{};

    const auto inputDataShape = getShape(origOp.getInputData());

    if (inputDataShape.isDynamic()) {
        sequenceLength = to_small_vector(inputDataShape)[inputDataShape.size() - 2];
    }

    mlir::Value newWeights;
    if (const auto weights = origOp.getWeights(); weights) {
        const auto inputSize = getShape(weights).back();
        const Shape newWeightsShape{1, numDirections, 4 * hiddenSize, inputSize};
        newWeights = reshapeValue(weights, newWeightsShape, "_reshapeWeights");
        newInputDataShape = Shape{batchSize, 1, sequenceLength, inputSize};
    } else {
        newInputDataShape = Shape{batchSize, numDirections, sequenceLength, 4 * hiddenSize};
    }

    mlir::Value newBiases;
    if (const auto biases = origOp.getBiases(); biases) {
        const Shape newBiasesShape{1, numDirections, 1, 4 * hiddenSize};
        newBiases = reshapeValue(biases, newBiasesShape, "_reshapeBiases");
    }

    const mlir::Value newInputData = reshapeValue(origOp.getInputData(), newInputDataShape, "_reshapeInputData");
    const mlir::Value newInitialHiddenState =
            reshapeValue(origOp.getInitialHiddenState(), newInitialHiddenStateShape, "_reshapeInitialHiddenState");
    const mlir::Value newInitialCellState =
            reshapeValue(origOp.getInitialCellState(), newInitialCellStateShape, "_reshapeInitialCellState");
    const mlir::Value newRecurrenceWeights =
            reshapeValue(origOp.getReccurenceWeights(), newRecurrenceWeightsShape, "_reshapeRecurrenceWeights");

    auto newOp = rewriter.create<IE::LSTMSequenceOp>(origOp.getLoc(), newInputData, newInitialHiddenState,
                                                     newInitialCellState, newWeights, newRecurrenceWeights, newBiases,
                                                     origOp.getSequenceLengthAttr(), origOp.getDirectionAttr());

    SmallVector<mlir::Value> reshapedResultsVec;
    for (const auto& [origOpResult, newOpResult] : llvm::zip(origOp.getResults(), newOp.getResults())) {
        const auto origOpResultShape = getShape(origOpResult);
        mlir::Value resultReshape = reshapeValue(newOpResult, origOpResultShape, "_reshapeOut");

        reshapedResultsVec.push_back(resultReshape);
    }

    rewriter.replaceOp(origOp, reshapedResultsVec);
    return mlir::success();
}

//
// ConcatConverter
//

class ConcatConverter final : public mlir::OpConversionPattern<IE::ConcatOp> {
public:
    ConcatConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConcatOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatConverter::matchAndRewrite(IE::ConcatOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::ConcatOp Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        return mlir::failure();
    }

    const auto concatAxis = (*axis.begin()).ind();
    const auto origOutputShape = getShape(origOp.getOutput());
    const auto shapeRank = checked_cast<int32_t>(origOutputShape.size());

    // The reason for placing the axis of concat in the third dimension is:
    // 1. We need to ensure that the batch dimension after conversion is 1.
    // 2. The axis for concatenation cannot be split or merged.
    // So a concat will be converted to 1x (axis before concat axis) x (concat axis) x (axis after concat axis)

    // For inputRank > TARGET_TENSOR_DIM case:
    //      tensor<axbxcxdxexfxf16>,       tensor<axbxcxdxexfxf16> ->      tensor<axbxcx2dxexfxf16>
    //             \|/   |  \/                    \|/   |  \/                      \|/   |  \/
    //  tensor<1x(a*b*c)xdx(e*f)xf16>, tensor<1x(a*b*c)xdx(e*f)xf16> -> tensor<1x(a*b*c)x2dx(e*f)xf16>

    // For inputRank < TARGET_TENSOR_DIM case:
    //     tensor<axbxf16>,    tensor<axbxf16> ->     tensor<ax2bxf16>
    //            | |                 | |                    |  |
    //   tensor<1xaxbx1xf16>,tensor<1xaxbx1xf16> -> tensor<1xax2bx1xf16>
    // Special pattern: The axis is in the second dim and the output c value is 16 aligned. The extension on H
    // is more friendly to NCE ops.
    //     tensor<1xa1xbxf16>,  tensor<1xa2xbxf16> ->     tensor<1xcxbxf16>
    //              |   \                |   \                    /  |
    //     tensor<1xa1x1xbxf16>,tensor<1xa2x1xbxf16> -> tensor<1xcx1xbxf16>
    const auto elemType = origOp.getOutput().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(elemType);
    auto extendOnH = false;
    if (origOutputShape[Dim(0)] == 1 && concatAxis == 1 && origOutputShape[Dim(concatAxis)] % alignment == 0 &&
        shapeRank <= TARGET_TENSOR_DIM) {
        extendOnH = true;
    }

    MergeMap mergeMap;
    mergeMap.push_back(irange(concatAxis));
    mergeMap.push_back({concatAxis});
    mergeMap.push_back(irange(concatAxis + 1, shapeRank));

    const auto inputs = origOp.getInputs();
    SmallVector<mlir::Value> newInputs;
    for (const auto& input : inputs) {
        const auto inputReshape =
                reshapeInputWithMergeMap(rewriter, takeOpLoc(origOp, StringLiteral("reshape_in_{0}"), newInputs.size()),
                                         shapeRank, input, mergeMap, extendOnH);
        newInputs.emplace_back(inputReshape);
    }

    auto offsetsAttr = origOp.getStaticOffsetsAttr();
    if (!offsetsAttr) {
        auto axis = origOp.getPerAxisAttr().getAxis().getValue().getSExtValue();
        offsetsAttr = inferOffsetsAttrWithAxis(origOp, axis);
    }
    const auto totalOffset = parseIntArrayOfArrayAttr<int64_t>(offsetsAttr);
    SmallVector<SmallVector<int64_t>> newTotalOffset;
    const auto outShape = getShape(origOp.getOutput());

    for (const auto& offset : totalOffset) {
        SmallVector<int64_t> newOffset(TARGET_TENSOR_DIM, 0);
        if (extendOnH) {
            newOffset[1] = offset[concatAxis];
        } else {
            // The concat will be convert to 1x (axis before concat axis) x (concat axis) x (axis after concat axis),so
            // the concat axis must in the third dimension.
            newOffset[2] = offset[concatAxis];
        }
        newTotalOffset.emplace_back(newOffset);
    }

    const auto newStaticOffsetsAttr = getIntArrayOfArray(this->getContext(), newTotalOffset);

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), newInputs, nullptr, newStaticOffsetsAttr);

    const auto outputShapeAttr = getIntArrayAttr(this->getContext(), outShape);
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConcat.getOutput(), nullptr, false, outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::ConcatOp'", getDebugName());

    return mlir::success();
}

//
// TransposeConverter
//

class TransposeConverter final : public mlir::OpConversionPattern<IE::TransposeOp> {
public:
    TransposeConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::TransposeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TransposeConverter::matchAndRewrite(IE::TransposeOp origOp, OpAdaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Transpose Operation '{1}'", getDebugName(), origOp->getLoc());
    const auto origType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();

    auto mergedPermAndShape =
            vpux::getMergedPermutationAndShape(origType, origOp.getOrderValue().value(), TARGET_TENSOR_DIM);
    auto mergedPermutation = mergedPermAndShape.first;
    auto mergedShape = mergedPermAndShape.second;

    extendPermutationAndShape(mergedPermutation, mergedShape, TARGET_TENSOR_DIM);
    auto reducedPermutation = mlir::AffineMap::getPermutationMap(ArrayRef(mergedPermutation), rewriter.getContext());

    // Build input reshape operation
    auto reducedShapeAttr = getIntArrayAttr(rewriter.getContext(), mergedShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             /*shape=*/nullptr, false, reducedShapeAttr);

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(origOp->getLoc(), inputReshape, nullptr,
                                                           mlir::AffineMapAttr::get(reducedPermutation));

    // Reshape to original output shape
    auto outputShape = getShape(origOp.getOutput());
    auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), outputShape);
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newTransposeOp.getOutput(), /*shape=*/nullptr,
                                                                 false, outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::Tranpose'", getDebugName());
    return mlir::success();
}

mlir::FailureOr<std::tuple<SmallVector<int64_t>, int64_t>> getNewSoftmaxParam(vpux::NDTypeInterface origType,
                                                                              int64_t axis) {
    const auto inputShape = origType.getShape().raw();
    // Only support dimension expansion
    if (origType.getRank() >= TARGET_TENSOR_DIM) {
        return mlir::failure();
    }

    // Support two cases:
    // when the meaningful shape (not 1) is 1 and axis is on that dimension,
    //      put the dimension to W to keep the original method
    // for other cases, put the non-1 dimensions to C and H
    //      to increase the possibility of multi-cluster and tiling
    // e.g. [32, 10] -> [1, 32, 10, 1]
    //      [1, 51] -> [1, 1, 1, 51]
    //      [1, 32, 10] -> [1, 32, 10, 1]
    if (axis < 0) {
        axis += origType.getRank();
    }

    auto isSingleDimSoftMax = [&]() {
        return llvm::all_of(irange(origType.getRank()), [&](int64_t ind) {
            return inputShape[ind] == 1 || ind == axis;
        });
    };

    // Optimization for softmax kernel should make axis last dim.
    // Maintain axis last dim after being reshaped to 4D.
    auto isTwoDimAxisLastSoftMax = [&]() {
        const auto rank = origType.getRank();
        return rank == 2 && axis == rank - 1;
    };

    SmallVector<int64_t> newInputShape;
    auto addDims = static_cast<int32_t>(TARGET_TENSOR_DIM - origType.getRank());
    int64_t newAxis = axis;
    if (isSingleDimSoftMax() || isTwoDimAxisLastSoftMax()) {
        newInputShape = SmallVector<int64_t>(addDims, 1);
        for (auto i = 0; i < origType.getRank(); i++) {
            newInputShape.push_back(inputShape[i]);
        }
        newAxis = axis + addDims;
    } else {
        // set batch = 1 and enable more axis to split
        if (inputShape[0] != 1) {
            newInputShape.push_back(1);
            addDims--;
            newAxis++;
        }

        for (auto i = 0; i < origType.getRank(); i++) {
            newInputShape.push_back(inputShape[i]);
        }

        for (auto i = 0; i < addDims; i++) {
            newInputShape.push_back(1);
        }
    }

    return std::tuple<SmallVector<int64_t>, int64_t>(newInputShape, newAxis);
}

//
// SoftmaxConverter
//

class SoftmaxConverter final : public mlir::OpConversionPattern<IE::SoftMaxOp> {
public:
    SoftmaxConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SoftMaxOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SoftmaxConverter::matchAndRewrite(IE::SoftMaxOp origOp, OpAdaptor,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Softmax Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto origType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    int64_t axis = origOp.getAxisInd();

    const auto newSoftmaxParam = getNewSoftmaxParam(origType, axis);
    if (mlir::failed(newSoftmaxParam)) {
        _log.trace("Only support dimension expansion");
        return mlir::failure();
    }
    const auto newSoftmaxParamVal = newSoftmaxParam.value();
    const auto newInputShapeAttr = getIntArrayAttr(getContext(), std::get<0>(newSoftmaxParamVal));
    const auto axisAttr = getIntAttr(getContext(), std::get<1>(newSoftmaxParamVal));

    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);

    auto newSoftmaxOp =
            rewriter.create<IE::SoftMaxOp>(origOp->getLoc(), inputReshape, axisAttr, origOp.getPadSizeAttr());

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newSoftmaxOp.getOutput(), nullptr, false,
                                                                 outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::SoftMaxOp'", getDebugName());

    return mlir::success();
}

class ScaleShiftConverter final : public mlir::OpConversionPattern<IE::ScaleShiftOp> {
public:
    ScaleShiftConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ScaleShiftOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value reshapeOperand(mlir::Value operand, mlir::Value newOperand, mlir::ConversionPatternRewriter& rewriter) {
    if (operand == nullptr) {
        return nullptr;
    }
    ShapeRef targetShape = getShape(newOperand);
    const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), targetShape);
    return rewriter.createOrFold<IE::ReshapeOp>(appendLoc(operand.getLoc(), "reshape"), operand, nullptr, false,
                                                newInputShapeAttr);
}

mlir::LogicalResult ScaleShiftConverter::matchAndRewrite(IE::ScaleShiftOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::ScaleShiftOp Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto reshapedData = reshapeOperand(origOp.getInput(), newArgs.getInput(), rewriter);
    const auto reshapedScales = reshapeOperand(origOp.getWeights(), newArgs.getWeights(), rewriter);
    const auto reshapedBiases = reshapeOperand(origOp.getBiases(), newArgs.getBiases(), rewriter);

    auto newScaleShift =
            rewriter.create<IE::ScaleShiftOp>(origOp->getLoc(), reshapedData, reshapedScales, reshapedBiases);

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newScaleShift.getOutput(), nullptr, false,
                                                                 outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::ScaleShiftOp'", getDebugName());

    return mlir::success();
}

//
// LogSoftmaxConverter
//

class LogSoftmaxConverter final : public mlir::OpConversionPattern<IE::LogSoftmaxOp> {
public:
    LogSoftmaxConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::LogSoftmaxOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LogSoftmaxOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LogSoftmaxConverter::matchAndRewrite(IE::LogSoftmaxOp origOp, OpAdaptor,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::LogSoftmaxOp Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto origType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    int64_t axis = origOp.getAxisInd();

    const auto newSoftmaxParam = getNewSoftmaxParam(origType, axis);
    if (mlir::failed(newSoftmaxParam)) {
        _log.trace("Only support dimension expansion");
        return mlir::failure();
    }
    const auto newSoftmaxParamVal = newSoftmaxParam.value();
    const auto newInputShapeAttr = getIntArrayAttr(getContext(), std::get<0>(newSoftmaxParamVal));
    const auto axisAttr = getIntAttr(getContext(), std::get<1>(newSoftmaxParamVal));

    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), origOp.getInput(), nullptr, false,
                                                             newInputShapeAttr);

    auto newSoftmaxOp = rewriter.create<IE::LogSoftmaxOp>(origOp->getLoc(), inputReshape, axisAttr);

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newSoftmaxOp.getOutput(), nullptr, false, outputShapeAttr);

    _log.trace("[{0}] Replaced with 'IE::LogSoftmaxOp'", getDebugName());

    return mlir::success();
}

//
// InterpolateConverter
//

class InterpolateConverter final : public mlir::OpConversionPattern<IE::InterpolateOp> {
public:
    InterpolateConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::InterpolateOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InterpolateConverter::matchAndRewrite(IE::InterpolateOp origOp, OpAdaptor,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Interpolate Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto inputShape = getShape(origOp.getInput()).raw();
    const auto inputRank = inputShape.size();

    VPUX_THROW_WHEN(inputRank > TARGET_TENSOR_DIM, "Tensors with rank > 4 are not supported");

    const auto addDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inputRank);

    const auto createAxesAttr = [&](std::optional<mlir::ArrayAttr> axesAttr) {
        if (axesAttr.has_value()) {
            auto intArray = parseIntArrayAttr<int64_t>(axesAttr.value());
            for (auto& val : intArray) {
                val += addDims;
            }
            SmallVector<unsigned> sortIndexArray(addDims);
            std::iota(sortIndexArray.begin(), sortIndexArray.end(), 0);
            intArray.insert(intArray.begin(), sortIndexArray.begin(), sortIndexArray.end());
            return getIntArrayAttr(this->getContext(), intArray);
        }
        return mlir::ArrayAttr();
    };

    const auto extendShapeWithValue = [&](std::optional<mlir::ArrayAttr> attr, int64_t value) {
        if (attr.has_value()) {
            auto intArray = parseIntArrayAttr<int64_t>(attr.value());
            intArray.insert(intArray.begin(), addDims, value);
            return getIntArrayAttr(this->getContext(), intArray);
        }
        return mlir::ArrayAttr();
    };

    const auto extendShapeWithFloatValue = [&](std::optional<mlir::ArrayAttr> attr, double value) {
        if (attr.has_value()) {
            auto fpArray = parseFPArrayAttr<double>(attr.value());
            fpArray.insert(fpArray.begin(), addDims, value);
            return getFPArrayAttr(this->getContext(), fpArray);
        }
        return mlir::ArrayAttr();
    };

    SmallVector<int64_t> newInputShape(addDims, 1);
    newInputShape.insert(newInputShape.end(), inputShape.begin(), inputShape.end());
    const auto newInputShapeAttr = getIntArrayAttr(this->getContext(), newInputShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);

    const auto attrs = origOp.getAttr();
    const auto newPadsBeginAttr = extendShapeWithValue(attrs.getPadsBegin(), 0);
    const auto newPadsEndAttr = extendShapeWithValue(attrs.getPadsEnd(), 0);
    const auto newAttr = IE::InterpolateAttr::get(this->getContext(), attrs.getMode(), attrs.getShapeCalcMode(),
                                                  attrs.getCoordMode(), attrs.getNearestMode(), attrs.getAntialias(),
                                                  newPadsBeginAttr, newPadsEndAttr, attrs.getCubeCoeff());

    const auto newAxesAttr = createAxesAttr(origOp.getAxesAttr());
    const auto newSizesAttr = extendShapeWithValue(origOp.getSizesAttr(), 1);
    const auto newScalesAttr = extendShapeWithFloatValue(origOp.getScalesAttr(), 1.0);
    const auto newOffsetAttr = extendShapeWithValue(origOp.getTileOffsetAttr(), 0);
    const auto newInitInputDimAttr = extendShapeWithValue(origOp.getInitialInputDimsAttr(), 1);
    const auto newInitOutputDimAttr = extendShapeWithValue(origOp.getInitialOutputDimsAttr(), 1);
    auto newInterpOp = rewriter.create<IE::InterpolateOp>(
            origOp->getLoc(), inputReshape, nullptr, nullptr, nullptr, newSizesAttr, newScalesAttr, newAxesAttr,
            newOffsetAttr, newInitInputDimAttr, newInitOutputDimAttr, newAttr, origOp.getOutputChannelsAttr());

    const auto outShape = getShape(origOp.getOutput());
    const auto outputShapeAttr = getIntArrayAttr(this->getContext(), outShape);
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newInterpOp.getOutput(), nullptr, false,
                                                                 outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::InterpolateOp'", getDebugName());

    return mlir::success();
}

//
// GatherConverter
//

class GatherConverter final : public mlir::OpConversionPattern<IE::GatherOp> {
public:
    GatherConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::GatherOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GatherConverter::matchAndRewrite(IE::GatherOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    _log.trace("[{0}] Found Gather Operation at '{1}'", getDebugName(), origOp->getLoc());

    const auto axis = origOp.getAxisValue().value();
    const auto inType = origOp.getInput().getType().cast<NDTypeInterface>();
    const auto inShape = inType.getShape();
    const auto indicesType = origOp.getIndices().getType().cast<NDTypeInterface>();
    const auto batchDim = origOp.getBatchDims();

    auto fuseDims = [](auto begin, auto end) {
        return std::accumulate(begin, end, int64_t(1), std::multiplies<int64_t>());
    };

    // Convert Gather Op to a 4D tensor using the following dimensional rules:
    // [BatchDimsRange, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    // Example transformations:
    // 1. Original: Input: 5x6x7, Indices: 5x3, Axis: 1, Batch_dim: 1, Output: 5x3x7
    //    Transformed: Input: 5x1x6x7, Indices: 5x3, Axis: 2, Batch_dim: 1, Output: 5x1x3x7
    // 2. Original: Input: 1x5x7, Indices: 6x3, Axis: 2, Batch_dim: 0, Output: 1x5x6x3
    //    Transformed: Input: 1x5x7x1, Indices: 18, Axis: 2, Batch_dim: 0, Output: 1x5x18x1
    // 3. Original: Input: 1x5x1x1x6x7, Indices: 4x5, Axis: 4, Batch_dim: 0, Output: 1x5x1x1x4x5x7
    //    Transformed: Input: 1x5x6x7, Indices: 4x5, Axis: 2, Batch_dim: 0, Output: 1x5x20x7
    auto fusedBatchDimSize = batchDim > 0 ? fuseDims(inShape.begin(), inShape.begin() + batchDim) : 1;
    auto fusedBeforeAxisDimSize = fuseDims(inShape.begin() + batchDim, inShape.begin() + axis);
    auto fusedIndicesDimSize = fuseDims(indicesType.getShape().begin() + batchDim, indicesType.getShape().end());
    auto fusedAfterAxisDimSize = fuseDims(inShape.begin() + axis + 1, inShape.end());

    SmallVector<int64_t> newInShape{fusedBatchDimSize, fusedBeforeAxisDimSize, inShape[Dim(axis)],
                                    fusedAfterAxisDimSize};
    SmallVector<int64_t> newOutShape{fusedBatchDimSize, fusedBeforeAxisDimSize, fusedIndicesDimSize,
                                     fusedAfterAxisDimSize};

    // Support Multi Cluster feature. The Indices must be 4D to comply with requirements
    // Two dimensions of size '1' are appended after the actual indices
    // The attribute 'indicesRank' is used to retrieve the actual indices
    const auto indicesRank = 2;
    SmallVector<int64_t> newIndicesShape{fusedBatchDimSize, fusedIndicesDimSize, 1, 1};

    auto newAxis = Dims4D::Act::H.ind();
    auto newBatchDim = 1;

    auto createReshapeOp = [&](mlir::Value input, ShapeRef shape, StringRef locSuffix) -> mlir::Value {
        auto shapeAttr = getIntArrayAttr(ctx, shape);
        return rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("reshape_{0}"), locSuffix), input,
                                                    nullptr, false, shapeAttr);
    };

    auto inputReshape = createReshapeOp(origOp.getInput(), Shape(newInShape), "in");
    auto indicesReshape = createReshapeOp(origOp.getIndices(), Shape(newIndicesShape), "indices");
    auto newGatherOp =
            rewriter.create<IE::GatherOp>(origOp.getLoc(), inputReshape, indicesReshape, nullptr,
                                          getIntAttr(ctx, newAxis), newBatchDim, getIntAttr(ctx, indicesRank));
    auto outputReshape = createReshapeOp(newGatherOp.getOutput(), getShape(origOp.getOutput()), "out");

    _log.trace("Replaced {0} with 4D tensor", origOp.getLoc());
    rewriter.replaceOp(origOp, outputReshape);

    return mlir::success();
}

//
// GatherElementsConverter
//

class GatherElementsConverter final : public mlir::OpConversionPattern<IE::GatherElementsOp> {
public:
    GatherElementsConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::GatherElementsOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherElementsOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GatherElementsConverter::matchAndRewrite(IE::GatherElementsOp origOp, OpAdaptor,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    _log.trace("[{0}] Found GatherElements Operation at '{1}'", getDebugName(), origOp->getLoc());

    const auto axis = origOp.getAxis();
    const auto inType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inShape = inType.getShape();
    const auto indicesType = mlir::cast<NDTypeInterface>(origOp.getIndices().getType());

    auto fuseDims = [](auto begin, auto end) {
        return std::accumulate(begin, end, int64_t(1), std::multiplies<int64_t>());
    };

    // Convert GatherElements Op to a 4D tensor using the following dimensional rules:
    // [1, DataBeforeAxisRange, AxisRange, DataAfterAxisRange]
    // Example transformations:
    // 1. Original: Input: 4x3x4, Indices: 2x3x4, Axis: 0, Output: 2x3x4
    //    Transformed: Input: 1x1x4x12, Indices: 1x1x2x12, Axis: 2, Output: 1x1x2x12
    // 2. Original: Input: 2x6x4, Indices: 2x3x4, Axis: 1, Output: 2x3x4
    //    Transformed: Input: 1x2x6x4, Indices: 1x2x3x4, Axis: 2, , Output: 1x2x3x4
    // 3. Original: Input: 6x8x8x8x8, Indices: 2x3x4x4x4, Axis: 2, Output: 1x5x1x1x4x5x7
    //    Transformed: Input: 1x48x8x64, Indices: 1x6x4x16, Axis: 2, Output: 1x6x4x16
    auto getFusedShape = [&](ShapeRef shape) {
        auto fusedDimSizeBeforeAxis = fuseDims(shape.begin(), shape.begin() + axis);
        auto fusedDimSizeAtAxis = shape[Dim(axis)];
        auto fusedDimSizeAfterAxis = fuseDims(shape.begin() + axis + 1, shape.end());
        return SmallVector<int64_t>{1, fusedDimSizeBeforeAxis, fusedDimSizeAtAxis, fusedDimSizeAfterAxis};
    };
    auto newInputShape = getFusedShape(inShape);
    auto newIndicesShape = getFusedShape(indicesType.getShape());
    auto newOutputShape = newIndicesShape;

    auto newAxis = Dims4D::Act::H.ind();

    auto createReshapeOp = [&](mlir::Value input, ShapeRef shape, StringRef locSuffix) -> mlir::Value {
        auto shapeAttr = getIntArrayAttr(ctx, shape);
        return rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("reshape_{0}"), locSuffix), input,
                                                    nullptr, false, shapeAttr);
    };

    auto inputReshape = createReshapeOp(origOp.getInput(), Shape(newInputShape), "in");
    auto indicesReshape = createReshapeOp(origOp.getIndices(), Shape(newIndicesShape), "indices");
    auto newGatherElementsOp =
            rewriter.create<IE::GatherElementsOp>(origOp.getLoc(), inputReshape, indicesReshape, newAxis);
    auto outputReshape = createReshapeOp(newGatherElementsOp.getOutput(), getShape(origOp.getOutput()), "out");

    _log.trace("Replaced {0} with 4D tensor", origOp.getLoc());
    rewriter.replaceOp(origOp, outputReshape);
    return mlir::success();
}
class AccumulateConverter final : public mlir::OpConversionPattern<IE::AccumulateOp> {
public:
    AccumulateConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::AccumulateOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AccumulateOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AccumulateConverter::matchAndRewrite(IE::AccumulateOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Accumulate Operation '{1}'", getDebugName(), origOp.getLoc());

    auto ctx = rewriter.getContext();
    // Transpose 1x1xHxW into 1xWxHx1.
    // VPU.Accumulate kernel expects the scales to apply over the innermost dimension.
    // VPU.Accumulate also requires NHWC layout, thus the scales must apply over channel axis.
    // The producer of IE.Accumulate operation is a MatMul with 1x1xHxW output.
    // In 1x1xHxW * 1x1x1xW case the scales apply over the width of the tensor.
    // The transposition is required in order to scale over channels instead of scaling over width.
    // 1xWxHx1 * 1xWx1x1 -> 1xWxHx1
    const SmallVector<unsigned> transposition = {0, 3, 2, 1};
    const auto affineMap = mlir::AffineMap::getPermutationMap(ArrayRef(transposition), ctx);
    const auto affineMapAttr = mlir::AffineMapAttr::get(affineMap);
    const auto newOperands = newArgs.getOperands();
    size_t counter = 0;
    const auto transposeOperand = [&](const mlir::Value val) -> mlir::Value {
        const auto loc = takeOpLoc(origOp, StringLiteral("in_{0}_to_NWHC"), counter++);
        auto transposedVal = rewriter.create<IE::TransposeOp>(loc, val, nullptr, affineMapAttr);
        return transposedVal.getOutput();
    };

    // transposedOperands is a placeholder for lhs, rhs, lhsScale and rhsScale.
    // When newOperands don't provide scales, transposedOperands[2:3] contain nullptr values
    SmallVector<mlir::Value> transposedOperands(4, nullptr);
    std::transform(newOperands.begin(), newOperands.end(), transposedOperands.begin(), transposeOperand);

    auto newAcc = rewriter.create<IE::AccumulateOp>(origOp.getLoc(),
                                                    /*lhs=*/transposedOperands[0],
                                                    /*rhs=*/transposedOperands[1],
                                                    /*lhsScale=*/transposedOperands[2],
                                                    /*rhsScale=*/transposedOperands[3]);

    auto transposeOut = rewriter.create<IE::TransposeOp>(origOp.getLoc(), newAcc.getOutput(), nullptr, affineMapAttr);
    extendOpLoc(transposeOut, "transpose_out");

    rewriter.replaceOp(origOp, transposeOut.getOutput());

    return mlir::success();
}

//
// BroadcastConverter
//

class BroadcastConverter final : public mlir::OpConversionPattern<IE::BroadcastOp> {
public:
    BroadcastConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::BroadcastOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::BroadcastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BroadcastConverter::matchAndRewrite(IE::BroadcastOp origOp, OpAdaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Broadcast Operation '{1}'", getDebugName(), origOp->getLoc());

    auto inShape = origOp.getInput().getType().cast<NDTypeInterface>().getShape();
    auto outShape = origOp.getOutput().getType().cast<NDTypeInterface>().getShape();

    SmallVector<int64_t> newInputShape;
    SmallVector<int64_t> newOutputShape;

    for (size_t i = 1; i < inShape.size(); i++) {
        newInputShape.push_back(inShape.raw()[i]);
        newOutputShape.push_back(outShape.raw()[i]);
    }

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);

    auto newBroadcastOp = rewriter.create<IE::BroadcastOp>(
            origOp->getLoc(), inputReshape,
            vpux::IE::createShapeConstForBroadCast(rewriter, getContext(), origOp->getLoc(), ShapeRef(newOutputShape)),
            origOp.getAxesMapping(), origOp.getModeAttr());

    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newBroadcastOp.getOutput(), nullptr, false,
                                                                 getIntArrayAttr(getContext(), outShape.raw()));
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::BroadcastOp'", getDebugName());
    return mlir::success();
}

//
// safeRunOnFunc
//

auto buildReshapeMaterializer(StringRef locSuffix) {
    const auto reshapeFunc = [=](mlir::OpBuilder& builder, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                                 mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());

        const auto outShapeAttr = builder.getI64ArrayAttr(dstType.getShape());
        return builder.createOrFold<IE::ReshapeOp>(appendLoc(loc, locSuffix), inputs.front(), nullptr, false,
                                                   outShapeAttr);
    };
    return reshapeFunc;
}

void ConvertShapeTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface type) {
        SmallVector<int64_t> shape = to_small_vector(type.getShape());
        auto dimMapper = getDimMapGeneric(shape);
        return type.changeShape(ShapeRef(alignShapeTo4D(shape, dimMapper, false)));
    });
    typeConverter.addSourceMaterialization(buildReshapeMaterializer("source"));
    typeConverter.addTargetMaterialization(buildReshapeMaterializer("target"));
    typeConverter.addArgumentMaterialization(buildReshapeMaterializer("argument"));

    mlir::TypeConverter scaleShiftTypeConverter;
    // TODO: #143748 consider change ConvertScaleShiftToDWPass
    scaleShiftTypeConverter.addConversion([](vpux::NDTypeInterface type) {
        SmallVector<int64_t> shape = to_small_vector(type.getShape());
        if (shape.size() == 3 && shape[0] == 1 && shape[1] > 1) {
            return type.changeShape(Shape{shape[0], shape[1], 1, shape[2]});
        }
        auto dimMapper = getDimMapGeneric(shape);
        return type.changeShape(ShapeRef(alignShapeTo4D(shape, dimMapper, false)));
    });
    scaleShiftTypeConverter.addSourceMaterialization(buildReshapeMaterializer("scale_shift_source"));
    scaleShiftTypeConverter.addTargetMaterialization(buildReshapeMaterializer("scale_shift_target"));
    scaleShiftTypeConverter.addArgumentMaterialization(buildReshapeMaterializer("scale_shift_arg"));

    // TODO(E#117111): the below checks are organized in a way to skip the pass
    // if op has operands/results with dynamic shapes. Converting dynamically-shaped tensor to 4D
    // will be addressed separately
    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op) || IE::hasDynamicTensors(op);
    };

    const auto isLegalFqOp = [&](IE::FakeQuantizeOp op) {
        const auto inShape = op.getInput().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto outShape = op.getOutput().getType().cast<vpux::NDTypeInterface>().getShape();

        VPUX_THROW_WHEN(inShape != outShape,
                        "FakeQuantize must have the same shape for input and output. Got: {0} != {1}", inShape,
                        outShape);

        return inShape.size() == TARGET_TENSOR_DIM;
    };

    const auto allOperandsAre4D = [](mlir::Operation* op) {
        return llvm::all_of(op->getOperands(), [](const auto& value) {
            auto shape = getShape(value);
            return shape.size() == TARGET_TENSOR_DIM;
        });
    };

    const auto isLegalEltwiseOp = [&](mlir::Operation* op) {
        if (op->getNumOperands() < 2) {
            return true;
        }
        return allOperandsAre4D(op);
    };

    const auto is4DLegalOp = [&](mlir::Operation* op) {
        const auto inShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
        const auto outShape = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
        return inShape.size() == TARGET_TENSOR_DIM || outShape.isDynamic();
    };

    const auto isLegalTransposeOp = [&](IE::TransposeOp op) {
        const auto origType = op.getInput().getType().cast<vpux::NDTypeInterface>();
        // Cannot handle shape after been reduced is still bigger than TARGET_TENSOR_DIM now.
        // Will insert 1 before mergedShape, so mergedShape should be smaller than TARGET_TENSOR_DIM.
        auto mergedShape =
                vpux::getMergedPermutationAndShape(origType, op.getOrderValue().value(), TARGET_TENSOR_DIM).second;
        const auto inShape = getShape(op.getInput());

        return mergedShape.size() >= TARGET_TENSOR_DIM || origType.getRank() == TARGET_TENSOR_DIM ||
               inShape.isDynamic();
    };

    const auto isLegalBroadcastOp = [&](IE::BroadcastOp op) {
        auto inType = op.getInput().getType().cast<NDTypeInterface>();
        auto outShape = op.getOutput().getType().cast<NDTypeInterface>().getShape();

        return !(op.getMode() == IE::BroadcastType::BIDIRECTIONAL && inType.getRank() == 5 &&
                 inType.getShape()[Dims4D::Act::N] == 1 && outShape[Dims4D::Act::N] == 1);
    };

    const auto isLegalGatherOp = [&](IE::GatherOp op) {
        if (!op.getAxisValue().has_value()) {
            return true;
        }

        const auto axis = op.getAxisValue().value();
        const auto inShape = op.getInput().getType().cast<vpux::NDTypeInterface>().getShape();
        // The purpose of converting the Gather Op to 4D is to enable Multi Cluster execution
        // There are already several optimizations for the Gather Op, such as DDR Access and GatherDMA
        // The Gather software kernel is optimized and performs well when the axis is the highest dimension
        // Here, only the cases where the axis is not the highest dimension will undergo 4D conversion
        // This is because performance regressions were observed in CI, requiring further debugging
        const auto areDimsBeforeAxisOne = std::all_of(inShape.begin(), inShape.begin() + axis, [](int dim) {
            return dim == 1;
        });
        if (areDimsBeforeAxisOne) {
            return true;
        }

        const auto indicesShape = op.getIndices().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto outShape = op.getOutput().getType().cast<vpux::NDTypeInterface>().getShape();

        return inShape.size() == TARGET_TENSOR_DIM && outShape.size() == TARGET_TENSOR_DIM &&
               indicesShape.size() == TARGET_TENSOR_DIM;
    };

    const auto isLegalGatherElementsOp = [&](IE::GatherElementsOp op) {
        // The purpose of converting the GatherElements Op to 4D is to enable Multi Cluster execution here. Since the
        // input, output and indices have same rank, so only check the input shape here.
        const auto inShape = getShape(op.getInput());
        if (inShape.size() != TARGET_TENSOR_DIM) {
            return false;
        }

        // Check if the shape adheres to the following configurations:
        // [1, DataBeforeAxisRange, AxisRange, DataAfterAxisRange]
        if (inShape[Dim(0)] != 1) {
            return false;
        }

        auto axis = op.getAxis();
        return axis == 2;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IE::IEDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addDynamicallyLegalOp<IE::ClampOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReLUOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SigmoidOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::HSwishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SwishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::CosOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SqrtOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SinhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::CoshOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AsinhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AcoshOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AtanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ExpOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GeluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::DivideOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::MinimumOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::MaximumOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::PowerOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::AndOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::ScaleShiftOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::EqualOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::NotEqualOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>(isLegalFqOp);
    target.addDynamicallyLegalOp<IE::LessOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::SelectOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::LessEqualOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::GreaterOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::GreaterEqualOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::LogicalNotOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalOrOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::LogicalXorOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::BitwiseNotOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BitwiseAndOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::BitwiseOrOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::BitwiseXorOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::AbsOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AtanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AsinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AcosOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::RoundOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::PReluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LeakyReluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::MatMulOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::TopKOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::MVN6Op>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::FloorModOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::ModOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::StridedSliceOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::TransposeOp>(isLegalTransposeOp);
    target.addDynamicallyLegalOp<IE::SoftMaxOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::LogSoftmaxOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::InterpolateOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::FloorOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SquaredDifferenceOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ConvertOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ConcatOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AccumulateOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::BroadcastOp>(isLegalBroadcastOp);
    target.addDynamicallyLegalOp<IE::ReduceL1Op>(isLegalReduceOp<IE::ReduceL1Op>);
    target.addDynamicallyLegalOp<IE::ReduceL2Op>(isLegalReduceOp<IE::ReduceL2Op>);
    target.addDynamicallyLegalOp<IE::ReduceLogicalAndOp>(isLegalReduceOp<IE::ReduceLogicalAndOp>);
    target.addDynamicallyLegalOp<IE::ReduceLogicalOrOp>(isLegalReduceOp<IE::ReduceLogicalOrOp>);
    target.addDynamicallyLegalOp<IE::ReduceMaxOp>(isLegalReduceOp<IE::ReduceMaxOp>);
    target.addDynamicallyLegalOp<IE::ReduceMeanOp>(isLegalReduceOp<IE::ReduceMeanOp>);
    target.addDynamicallyLegalOp<IE::ReduceMinOp>(isLegalReduceOp<IE::ReduceMinOp>);
    target.addDynamicallyLegalOp<IE::ReduceProdOp>(isLegalReduceOp<IE::ReduceProdOp>);
    target.addDynamicallyLegalOp<IE::ReduceSumOp>(isLegalReduceOp<IE::ReduceSumOp>);
    target.addDynamicallyLegalOp<IE::TileOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::LSTMGatesOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::LSTMCellOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::LSTMSequenceOp>(allOperandsAre4D);
    target.addDynamicallyLegalOp<IE::GatherOp>(isLegalGatherOp);
    target.addDynamicallyLegalOp<IE::GatherElementsOp>(isLegalGatherElementsOp);
    target.addDynamicallyLegalOp<IE::ErfOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::DynamicDequantizeOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::RMSOp>(allOperandsAre4D);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter<IE::ClampOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::EluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ReLUOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SigmoidOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::HSwishOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SwishOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MishOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::TanhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SinOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::CosOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SqrtOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SinhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::CoshOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AsinhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AcoshOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AtanhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ExpOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GeluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::DivideOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MinimumOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MaximumOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::PowerOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AndOp>>(typeConverter, &ctx, _log);
    patterns.add<ScaleShiftConverter>(scaleShiftTypeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::EqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LessOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SelectOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LessEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::NotEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GreaterOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GreaterEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalNotOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalOrOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalXorOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::BitwiseAndOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::BitwiseOrOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::BitwiseXorOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::BitwiseNotOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AbsOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AtanOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AsinOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AcosOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::PReluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::RoundOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ConvertOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LeakyReluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::FloorOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::FloorModOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ModOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AddOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MultiplyOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MatMulOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SubtractOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SquaredDifferenceOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ErfOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::DynamicDequantizeOp>>(typeConverter, &ctx, _log);

    patterns.add<GatherConverter>(typeConverter, &ctx, _log);
    patterns.add<GatherElementsConverter>(typeConverter, &ctx, _log);
    patterns.add<FakeQuantizeConverter>(typeConverter, &ctx, _log);
    patterns.add<TopKOpConverter>(typeConverter, &ctx, _log);
    patterns.add<Mvn6Converter>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceL1Op>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceL2Op>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceLogicalAndOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceLogicalOrOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceMaxOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceMeanOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceMinOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceProdOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceSumOp>>(typeConverter, &ctx, _log);
    patterns.add<StridedSliceConverter>(typeConverter, &ctx, _log);
    patterns.add<ConcatConverter>(typeConverter, &ctx, _log);
    patterns.add<TransposeConverter>(typeConverter, &ctx, _log);
    patterns.add<SoftmaxConverter>(typeConverter, &ctx, _log);
    patterns.add<LogSoftmaxConverter>(typeConverter, &ctx, _log);
    patterns.add<InterpolateConverter>(typeConverter, &ctx, _log);
    patterns.add<AccumulateConverter>(typeConverter, &ctx, _log);
    patterns.add<BroadcastConverter>(typeConverter, &ctx, _log);
    patterns.add<TileConverter>(typeConverter, &ctx, _log);
    patterns.add<LSTMGatesConverter>(typeConverter, &ctx, _log);
    patterns.add<LSTMCellConverter>(typeConverter, &ctx, _log);
    patterns.add<LSTMSequenceConverter>(typeConverter, &ctx, _log);
    patterns.add<RMSOpConverter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}  // namespace

}  // namespace

//
// createConvertShapeTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShapeTo4DPass(Logger log) {
    return std::make_unique<ConvertShapeTo4DPass>(log);
}
