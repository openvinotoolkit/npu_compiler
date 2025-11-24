//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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

bool isTrivialDim(int64_t dim) {
    // Note: Dynamic dims are marked by `mlir::ShapedType::kDynamic`.
    return dim == 1;
}

template <typename ShapeType>
void alignShapeToReferenceShapeSize(size_t refSize, ShapeType& shape, bool extendOnH) {
    VPUX_THROW_UNLESS(refSize >= shape.size(), "The reference shape size({0}) < shape size({1})", refSize,
                      shape.size());
    const size_t diff = refSize - shape.size();
    if (diff) {
        if (extendOnH) {
            VPUX_THROW_UNLESS(diff < 3 && diff >= 1,
                              "Extend on H does not support reference shape size({0}) and shape size({1})", refSize,
                              shape.size());
            if (diff == 2) {
                shape.push_back(1);
            }
            shape.insert(shape.begin() + 2, 1, 1);
        } else {
            shape.insert(shape.begin(), diff, 1);
        }
    }
}

template <typename ShapeType1, typename ShapeType2>
void alignShapes(ShapeType1& smallShape, ShapeType2& largeShape, bool autoBroadcast) {
    if (autoBroadcast) {
        alignShapeToReferenceShapeSize(largeShape.size(), smallShape, false);
    } else {
        if (smallShape.size() == 1 && smallShape[Dim(0)] == largeShape[Dim(1)]) {
            // Some operations need to map their channels first. e.g. PRelu
            smallShape.insert(smallShape.begin(), 1);
            smallShape.append(largeShape.size() - 2, 1);
        } else {
            alignShapeToReferenceShapeSize(largeShape.size(), smallShape, false);
        }
    }
}

int64_t getBalancedDimIndex(ShapeRef shape) {
    int64_t dimH = 1;
    int64_t dimW = 1;
    size_t front = 0;
    size_t back = shape.size();

    while (front < back) {
        if (dimW < dimH) {
            --back;
            dimW *= shape[Dim(back)];
        } else {
            dimH *= shape[Dim(front)];
            ++front;
        }
    }
    return front;
}

template <typename ShapeType>
auto alignShapeWithDimMap(const ShapeType& originShape, const MergeMap& mapper) {
    auto retNewShape = makeShape(originShape, 0, 1);
    for (const auto& dims : mapper) {
        typename ShapeType::ValueType dimSize = 1;
        for (auto i : dims) {
            dimSize *= originShape[Dim(i)];
        }
        retNewShape.push_back(dimSize);
    }
    return retNewShape;
}

template <typename ShapeType>
auto alignShapeTo4D(const ShapeType& originShape, const MergeMap& mapper, bool extendOnH) {
    auto newShape = extendOnH ? copyShape(originShape) : alignShapeWithDimMap(originShape, mapper);
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

MergeMap getDimMapWithFirstGreater1DimAsC(Shape shape) {
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
    const auto splitDimIndex = getBalancedDimIndex(shape) + nextDimCIndex;
    retMapper.push_back(irange(nextDimCIndex, splitDimIndex));
    retMapper.push_back(irange(splitDimIndex, maxDim));
    return retMapper;
}

MergeMap getDimMapGeneric(ShapeRef shape) {
    MergeMap dimMapper;
    if (shape.size() > TARGET_TENSOR_DIM) {
        return getDimMapWithFirstGreater1DimAsC(Shape(shape));
    }
    return getTrivialMap(shape.size());
}

MergeMap getDimMergeMapWith2Inputs(ShapeRef input1, ShapeRef input2) {
    const auto shapeSize1 = input1.totalSize();
    const auto shapeSize2 = input2.totalSize();
    // Find the origin input and broadcast shape
    //  The large size shape is the origin input
    //  The small size shape is the shape that needs to be broadcast in some planes
    auto maxShape = (shapeSize1 > shapeSize2) ? input1 : input2;
    auto planeShape = (shapeSize1 > shapeSize2) ? input2 : input1;

    auto getMergeMap = [](ShapeRef fullShape, ShapeRef planeShape, auto condition) {
        MergeMap dimMap;
        SmallVector<int64_t> inputDimsTmp;
        for (size_t i = 0; i < fullShape.size(); i++) {
            const auto d = Dim(i);
            auto compareVal = condition(d, fullShape);
            if (compareVal == planeShape[d]) {
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

    auto sameDimCondition = [](Dim d, ShapeRef shape) {
        return shape[d];
    };
    auto planeDimCondition = [](Dim, ShapeRef) {
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
    auto calculateMergeMap = [&](ShapeRef fullShape, ShapeRef planeShape) {
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
                } else {
                    dimsCanMerge.push_back({dimIndex});
                }
                minVector->erase(minVector->begin());
            }
        }
        return dimsCanMerge;
    };

    auto getSubShape = [](ShapeRef shape, ArrayRef<int64_t> map) {
        Shape retShape;
        retShape.reserve(map.size());
        for (auto& dim : map) {
            retShape.push_back(shape[Dim(dim)]);
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

    auto isAllOne = [](const MergeMapItem& item, ShapeRef planeShape) {
        return std::all_of(item.begin(), item.end(), [&](int64_t dim) {
            return planeShape[Dim(dim)] == 1;
        });
    };
    switch (dimsCanMerge.size()) {
    case 1: {
        dimsCanMerge = getDimMapGeneric(maxShape);
        break;
    }
    case 2: {
        auto expandMapTo3D = [&](auto mapIt) {
            const auto subShape = getSubShape(maxShape, *mapIt);
            auto newReshapeDim = getBalancedDimIndex(subShape);
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
    case 3:
        // Add 1 at dim N to convert to 4D
        break;
    case 4:
        // Direct return 4D merge map
        break;
    case 5:
        // If the mergeMap is 5D but some merged dims are trivial (all sizes are 1), we can convert it to 4D.
        for (auto it = dimsCanMerge.begin() + 1; it != dimsCanMerge.end();) {
            if (isAllOne(*it, maxShape)) {
                // Merge with the previous element
                auto prevIt = it - 1;
                prevIt->insert(prevIt->end(), it->begin(), it->end());
                it = dimsCanMerge.erase(it);
            } else {
                ++it;
            }
        }
        // If no trivial merged dim is found, we throw an exception as default.
        if (dimsCanMerge.size() > 4) {
            VPUX_THROW("The input shape {0}, {1} can't convert to 4D", input1, input2);
        }
        break;
    default:
        VPUX_THROW("The input shape {0}, {1} can't convert to 4D", input1, input2);
        break;
    }
    return dimsCanMerge;
}

MergeMap getDimMergeMapWith3Inputs(ShapeRef input1, ShapeRef inputLow, ShapeRef outLow) {
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

    auto getDimIdx = [&](ShapeRef dims) -> int64_t {
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
        if (newShape[Dim(dimIdx)] > 1) {
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
    auto inputLowShape = Shape(getShape(origOp.getInputLow()));
    auto outputLowShape = Shape(getShape(origOp.getOutputLow()));

    const auto inputShape = callOnShapeOf(origOp.getInput().getType(), [](const auto& shape) {
        return reifyShape(shape);
    });
    const auto ref1ElemShape = makeShape(inputShape, inputShape.size(), 1);

    alignShapeToReferenceShapeSize(inputShape.size(), inputLowShape, false);
    alignShapeToReferenceShapeSize(inputShape.size(), outputLowShape, false);

    if (inputLowShape == outputLowShape) {
        return getDimMergeMapWith2Inputs(inputShape, inputLowShape);
    }
    if (ref1ElemShape == inputLowShape) {
        return getDimMergeMapWith2Inputs(inputShape, outputLowShape);
    }
    if (ref1ElemShape == outputLowShape) {
        return getDimMergeMapWith2Inputs(inputShape, inputLowShape);
    }
    return getDimMergeMapWith3Inputs(inputShape, inputLowShape, outputLowShape);
}

template <typename ShapeType>
mlir::Value createReshape(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                          const ShapeType& outputShape) {
    const auto ctx = builder.getContext();
    if constexpr (isStaticShape<ShapeType>) {
        return builder.createOrFold<IE::ReshapeOp>(loc, input, /*shape=*/nullptr, /*special_zero=*/false,
                                                   getIntArrayAttr(ctx, outputShape.raw()));
    } else {
        return IE::createDynamicReshape(builder, loc, input, shapeCast<BoundedShape>(outputShape));
    }
}

mlir::Value reshapeInputWithMergeMap(mlir::PatternRewriter& rewriter, mlir::Location loc, size_t referenceShapeSize,
                                     mlir::Value origInput, const MergeMap& map, bool extendOnH) {
    return callOnShapeOf(origInput.getType(), [&](const auto& shape) {
        auto outShape = copyShape(shape);
        // Note: ensure the rank of the current shape is aligned to the "reference"
        // shape (the shape that was used to calculate the merge map). this
        // guarantees we don't have buffer overflows due to mege map using indices
        // outside of current shape's rank.
        alignShapeToReferenceShapeSize(referenceShapeSize, outShape, extendOnH);

        auto constInputShape = alignShapeTo4D(outShape, map, extendOnH);
        return createReshape(rewriter, loc, origInput, constInputShape);
    });
}

template <typename ShapeType1, typename ShapeType2>
void tryAndConvert2NCEShape(ShapeType1& shape1, ShapeType2& shape2, MergeMap& map) {
    // 4D Multiply shape 1x1x1xM need convert Shape to 1xMx1x1
    //
    // TODO:
    // This logic is a litte same as AdaptShapesForScaleShiftPass.
    // May combine them into 1 pass and abandon the AdaptShapesForScaleShiftPass
    const auto nonTrivialDimPredicate = [](const auto dim) -> bool {
        return dim > 1;
    };
    const auto nonTrivialShape1Dims = std::count_if(shape1.begin(), shape1.end(), nonTrivialDimPredicate);
    const auto nonTrivialShape2Dims = std::count_if(shape2.begin(), shape2.end(), nonTrivialDimPredicate);
    // Filter out the Shape 1x1x1x1 and nonTrivialDims > 1 cases
    if ((nonTrivialShape1Dims > 1 || nonTrivialShape2Dims > 1) ||
        (nonTrivialShape1Dims == 0 && nonTrivialShape2Dims == 0)) {
        return;
    }

    auto findFirstNonTrivialDim = [&](const auto& shape) {
        const auto firstIt = std::find_if(shape.begin(), shape.end(), nonTrivialDimPredicate);
        return Dim(std::distance(shape.begin(), firstIt));
    };
    // Find the first non-trivial index from 2 input shapes
    const auto firstNonTrivialDim = std::min(findFirstNonTrivialDim(shape1), findFirstNonTrivialDim(shape2));

    // Already at DimC
    if (firstNonTrivialDim.ind() == 1) {
        return;
    }
    if (map.size() < 4) {
        map.insert(map.begin(), 4 - map.size(), {});
    }
    std::swap(shape1[Dim(1)], shape1[firstNonTrivialDim]);
    std::swap(shape2[Dim(1)], shape2[firstNonTrivialDim]);
    std::swap(map[1], map[firstNonTrivialDim.ind()]);
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

// Adjust input or repeat for tileOp
// First we will align the input rank and repeat rank to the same as output rank. We have the logic in TileOp
// canonicalize pass but may still have mismatch case in convert shape to 4D. Then for outputRank < TARGET_TENSOR_DIM,
// insert 1 to the front. For outputRank > TARGET_TENSOR_DIM, try to merge un-repeat Dim.
std::optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> getAdjustedShapeForTile(IE::TileOp origOp) {
    SmallVector<int64_t> inputShape(getShape(origOp.getInput()).raw());
    auto repeatValues = parseIntArrayAttr<int64_t>(origOp.getRepeatsValuesAttr());
    const auto origInRank = static_cast<int64_t>(inputShape.size());
    const auto origOutRank = static_cast<int64_t>(getShape(origOp.getOutput()).size());
    const auto repeatRank = static_cast<int64_t>(repeatValues.size());

    if (origInRank == TARGET_TENSOR_DIM && origOutRank == TARGET_TENSOR_DIM && repeatRank == TARGET_TENSOR_DIM) {
        return std::nullopt;
    }

    if (origInRank < origOutRank) {
        inputShape.insert(inputShape.begin(), origOutRank - origInRank, 1);
    }

    if (repeatRank < origOutRank) {
        repeatValues.insert(repeatValues.begin(), origOutRank - repeatRank, 1);
    }

    SmallVector<int64_t> newInshape(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> newRepeats(repeatValues.begin(), repeatValues.end());
    if (origOutRank == TARGET_TENSOR_DIM) {
        return std::pair{std::move(newInshape), std::move(newRepeats)};
    } else if (origOutRank < TARGET_TENSOR_DIM) {
        for (int64_t i = 0; i < TARGET_TENSOR_DIM - origOutRank; i++) {
            newInshape.insert(newInshape.begin(), 1);
            newRepeats.insert(newRepeats.begin(), 1);
        }
        return std::pair{std::move(newInshape), std::move(newRepeats)};
    } else {
        int64_t mergeCnt = 0;
        for (int64_t i = 0; i < origOutRank - 1; i++) {
            if (repeatValues[i] == 1 && repeatValues[i + 1] == 1) {
                newInshape[i + 1 - mergeCnt] *= newInshape[i - mergeCnt];
                newInshape.erase(newInshape.begin() + i - mergeCnt);
                newRepeats.erase(newRepeats.begin() + i - mergeCnt);
                mergeCnt++;
                if (mergeCnt == origOutRank - TARGET_TENSOR_DIM) {
                    break;
                }
            }
        }

        if (newInshape.size() == TARGET_TENSOR_DIM) {
            return std::pair{std::move(newInshape), std::move(newRepeats)};
        }

        if (origInRank == DimsGroups5D::Act::numDims && origOutRank == DimsGroups5D::Act::numDims &&
            repeatRank == DimsGroups5D::Act::numDims) {
            return std::nullopt;
        }

        if (newInshape.size() == DimsGroups5D::Act::numDims && newRepeats.size() == DimsGroups5D::Act::numDims) {
            return std::pair{std::move(newInshape), std::move(newRepeats)};
        }
    }

    return std::nullopt;
}

//
// ConvertShapeTo4DPass
//

class ConvertShapeTo4DPass final : public IE::impl::ConvertShapeTo4DBase<ConvertShapeTo4DPass> {
public:
    explicit ConvertShapeTo4DPass(bool forceConvertGatherTo4D, Logger log)
            : _forceConvertGatherTo4D(forceConvertGatherTo4D) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _forceConvertGatherTo4D;
};

//
// GenericConverter
//

// Returns true if any of the operands or result values has a per-axis quantized type.
bool isPerAxisQuantizedOp(mlir::Operation* op) {
    const auto isPerAxisQuantized = [](mlir::Type type) {
        return mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(mlir::cast<NDTypeInterface>(type).getElementType());
    };
    const auto isAnyOperandPerAxis = llvm::any_of(op->getOperandTypes(), isPerAxisQuantized);
    const auto isAnyResultPerAxis = llvm::any_of(op->getResultTypes(), isPerAxisQuantized);
    return isAnyOperandPerAxis || isAnyResultPerAxis;
}

// This is analogous to convertGeneric but has the additional ability to update the required destination type attribute
// for certain operations.
mlir::LogicalResult convertLowPrecisionConversionOp(mlir::Operation* origOp, mlir::ValueRange operands,
                                                    mlir::ConversionPatternRewriter& rewriter,
                                                    const mlir::TypeConverter& typeConverter) {
    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::IRMapping mapper;
    mapper.map(origOperands, operands);

    assert(origOp->getNumResults() == 1 && "Element type conversion operations should have exactly one result");
    const auto newType = typeConverter.convertType(origOp->getResult(0).getType());
    const auto newElementType = mlir::cast<NDTypeInterface>(newType).getElementType();

    assert(origOp->getNumOperands() == 1 && "Element type conversion operations should have exactly one operand");
    const auto input = mapper.lookup(origOp->getOperand(0));

    mlir::Operation* newOp = nullptr;
    if (mlir::isa<IE::QuantizeOp>(origOp)) {
        newOp = rewriter.create<IE::QuantizeOp>(origOp->getLoc(), input, newElementType);
    } else if (mlir::isa<IE::DequantizeOp>(origOp)) {
        newOp = rewriter.create<IE::DequantizeOp>(origOp->getLoc(), input, newElementType);
    } else if (mlir::isa<IE::QuantizeCastOp>(origOp)) {
        newOp = rewriter.create<IE::QuantizeCastOp>(origOp->getLoc(), input, newElementType);
    } else {
        return matchFailed(rewriter, origOp, "Unsupported conversion operation: {0}", origOp->getName());
    }

    assert(newOp != nullptr);
    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

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

        // TODO: E#-171827 Support operands with 2 inputs as well.
        // If we have any operand or result that uses per-axis quantized types, we have to do some special handling.
        // Also for now, only operations with a single input are handles.
        if (isPerAxisQuantizedOp(origOp.getOperation())) {
            // These operations need to some of their attributes to be updated, that's why they're special.
            const auto isLowPresicsionConversionOp =
                    mlir::isa<IE::QuantizeOp, IE::DequantizeOp, IE::QuantizeCastOp>(origOp.getOperation());
            if (isLowPresicsionConversionOp) {
                return convertLowPrecisionConversionOp(origOp, newArgs.getOperands(), rewriter, *typeConverter);
            }

            // For now, we use the generic converter for all other operations.
            return convertGeneric(origOp, newArgs.getOperands(), rewriter, *typeConverter, _log);
        }

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

    const auto shapeOne = mlir::cast<vpux::NDTypeInterface>(input1.getType()).getShape();
    const auto shapeTwo = mlir::cast<vpux::NDTypeInterface>(input2.getType()).getShape();

    auto shapeOneVector = to_small_vector(shapeOne);
    auto shapeTwoVector = to_small_vector(shapeTwo);
    auto origOutputShape = getShape(origOp.getOutput());

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
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

    const auto [newIn1, newIn2, dimsCanMerge] = callOnShapeOf(input1.getType(), [&](const auto& inShapeRef1) {
        return callOnShapeOf(input2.getType(), [&](const auto& inShapeRef2) {
            auto inShape1 = copyShape(inShapeRef1);
            auto inShape2 = copyShape(inShapeRef2);

            // Align dims
            if (inShape1.size() != inShape2.size()) {
                extendOnH = false;
                const auto autoBroadcast = origOp->hasAttr("auto_broadcast");
                if (inShape1.size() < inShape2.size()) {
                    alignShapes(inShape1, inShape2, autoBroadcast);
                } else {
                    alignShapes(inShape2, inShape1, autoBroadcast);
                }
            }

            const auto reifiedShape1 = reifyShape(inShape1);
            const auto reifiedShape2 = reifyShape(inShape2);
            auto dimsCanMerge = getDimMergeMapWith2Inputs(reifiedShape1, reifiedShape2);
            auto newInShape1 = alignShapeTo4D(inShape1, dimsCanMerge, extendOnH);
            auto newInShape2 = alignShapeTo4D(inShape2, dimsCanMerge, extendOnH);

            if constexpr (std::is_same_v<IE::MultiplyOp, ConcreteOp>) {
                tryAndConvert2NCEShape(newInShape1, newInShape2, dimsCanMerge);
            }

            auto newIn1 = operands[0];
            auto newIn2 = operands[1];
            if (newInShape1 != inShapeRef1) {
                newIn1 = createReshape(rewriter, takeOpLoc(origOp, "reshape_lhs"), newIn1, newInShape1);
            }
            if (newInShape2 != inShapeRef2) {
                newIn2 = createReshape(rewriter, takeOpLoc(origOp, "reshape_rhs"), newIn2, newInShape2);
            }

            return std::make_tuple(newIn1, newIn2, std::move(dimsCanMerge));
        });
    });

    SmallVector<mlir::Value> newOperands;
    newOperands.push_back(newIn1);
    newOperands.push_back(newIn2);
    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), newOperands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    SmallVector<mlir::Value> newResults;
    newResults.reserve(newOp->getResults().size());

    for (auto result : newOp->getResults()) {
        const auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
        auto resultReshapeOp = callOnShapeOf(
                resultType,
                [&](const auto& shape, const MergeMap& dimsCanMerge) {
                    auto newOutShape = alignShapeTo4D(copyShape(shape), dimsCanMerge, extendOnH);
                    result.setType(resultType.changeTypeComponents(
                            TypeComponents()
                                    .setShapeWithRepresentation(std::move(newOutShape))
                                    .setDimsOrder(DimsOrder::fromNumDims(TARGET_TENSOR_DIM))));

                    return createReshape(rewriter,
                                         takeOpLoc(origOp, StringLiteral("reshape_out_{0}"), newResults.size()), result,
                                         shape);
                },
                dimsCanMerge);

        const auto newResult = resultReshapeOp == result ? result : resultReshapeOp.getDefiningOp()->getResult(0);
        newResults.push_back(newResult);
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

    auto outReshape = callOnShapeOf(origOp.getOutput().getType(), [&](const auto& shape) {
        return createReshape(rewriter, takeOpLoc(origOp, "reshape_out"), newFakeQuantizeOp.getOutput(), shape);
    });

    rewriter.replaceOp(origOp, outReshape);

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

    const auto origInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const int64_t origInRank = origInType.getRank();
    int64_t axis = origOp.getAxis();
    if (axis < 0) {
        axis += origInRank;
    }

    // Deduce the new TopK aix from map table
    const auto inShape = getShape(origOp.getInput());

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

    const auto newInShapeAttr = getIntArrayAttr(this->getContext(), alignShapeTo4D(inShape, mergeMap, false).raw());
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
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inShape = SmallVector<int64_t>(inType.getShape().raw());
    const auto inRank = inShape.size();
    auto inAxes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    mlir::Value scale = origOp.getScale();
    mlir::Value bias = origOp.getBias();

    SmallVector<int64_t> newShape;
    SmallVector<int64_t> newAxes;
    SmallVector<int64_t> actMulShape;     // optional scale
    SmallVector<int64_t> actAddShape;     // optional bias
    SmallVector<int64_t> newActMulShape;  // 4D version of 'actMulShape'
    SmallVector<int64_t> newActAddShape;  // 4D version of 'actAddShape'

    if (scale) {
        actMulShape = SmallVector<int64_t>(origOp.getScale().getType().getShape());
    }
    if (bias) {
        actAddShape = SmallVector<int64_t>(origOp.getBias().getType().getShape());
    }

    if ((inRank < TARGET_TENSOR_DIM) || (scale && (actMulShape.size() < TARGET_TENSOR_DIM)) ||
        ((bias && (actAddShape.size() < TARGET_TENSOR_DIM)))) {
        // insert leading 1s up to 4D and ajust axes accordingly
        auto to4DShape = [=](ArrayRef<int64_t> iShape, SmallVector<int64_t>& oShape) {
            auto inRank = iShape.size();
            if (inRank == TARGET_TENSOR_DIM) {
                oShape = SmallVector<int64_t>(iShape);
                return;
            }
            auto newDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inRank);
            oShape.insert(oShape.end(), newDims, 1);
            oShape.append(iShape.begin(), iShape.end());
        };
        to4DShape(inShape, newShape);  // main input
        if (scale) {                   // optional scale
            to4DShape(actMulShape, newActMulShape);
        }
        if (bias) {  // optional bias
            to4DShape(actAddShape, newActAddShape);
        }
        // increment 'axes'
        newAxes = std::move(inAxes);
        auto mvnNewDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inRank);
        std::for_each(newAxes.begin(), newAxes.end(), [mvnNewDims](int64_t& axis) {
            axis += mvnNewDims;
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
    auto inReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(), nullptr,
                                                          false, newShapeAttr);

    // reshape optional inputs if present
    if (scale) {
        const auto newActShapeAttr = getIntArrayAttr(getContext(), newActMulShape);
        scale = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_scale"), scale, nullptr, false,
                                                     newActShapeAttr);
    }
    if (bias) {
        const auto newActShapeAttr = getIntArrayAttr(getContext(), newActAddShape);
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
// ReverseSequenceOpConverter
//

class ReverseSequenceOpConverter final : public mlir::OpConversionPattern<IE::ReverseSequenceOp> {
public:
    ReverseSequenceOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ReverseSequenceOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReverseSequenceOpConverter::matchAndRewrite(IE::ReverseSequenceOp origOp, OpAdaptor,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto seqLensType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(1).getType());

    auto seqAxis = origOp.getSeqAxis();
    auto batchAxis = origOp.getBatchAxis();

    // input-data
    auto newInShape = to_small_vector(inType.getShape());
    auto inRank = newInShape.size();
    if (inRank <= TARGET_TENSOR_DIM) {
        const int64_t padDims = TARGET_TENSOR_DIM - inRank;
        newInShape.insert(newInShape.begin(), padDims, 1);
        batchAxis += padDims;
        seqAxis += padDims;
    } else if (inRank == 5) {
        SmallVector<int64_t> axes5D(inRank);
        std::iota(axes5D.begin(), axes5D.end(), 0);
        auto checkSame = [&](auto curDim, auto nxtDim) {
            auto curNonAxis = (curDim != batchAxis) && (curDim != seqAxis);
            auto nxtNonAxis = (nxtDim != batchAxis) && (nxtDim != seqAxis);
            return (curNonAxis && nxtNonAxis);
        };
        const auto mergeIt = std::adjacent_find(axes5D.begin(), axes5D.end(), checkSame);
        VPUX_THROW_WHEN(mergeIt == axes5D.end(),
                        "ReverseSequence 5D->4D failed : cannot find 2 adjacent dims which are not batch/seq axes");
        const auto mergeDim = checked_cast<int64_t>(std::distance(axes5D.begin(), mergeIt));
        if (seqAxis > mergeDim) {
            seqAxis = seqAxis - 1;
        }
        if (batchAxis > mergeDim) {
            batchAxis = batchAxis - 1;
        }
        newInShape[mergeDim] *= newInShape[mergeDim + 1];
        newInShape.erase(newInShape.begin() + mergeDim + 1);
    } else {
        VPUX_THROW("Unimplemented {0}D->4D ReverseSequence convert", inRank);
    }

    // sequence-lengths
    auto newSeqLensShape = SmallVector(TARGET_TENSOR_DIM, 1);
    newSeqLensShape[batchAxis] = seqLensType.getShape().totalSize();

    const auto newInShapeAttr = getIntArrayAttr(origOp->getContext(), newInShape);
    const auto newSeqLenShapeAttr = getIntArrayAttr(origOp->getContext(), newSeqLensShape);
    const auto newBatchAxisAttr = getIntAttr(origOp->getContext(), batchAxis);
    const auto newSeqAxisAttr = getIntAttr(origOp->getContext(), seqAxis);
    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));

    const auto in1Reshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getData(),
                                                                 nullptr, false, newInShapeAttr);
    const auto in2Reshape = rewriter.createOrFold<IE::ReshapeOp>(
            takeOpLoc(origOp, "reshape_seq_lens"), origOp.getSeqLength(), nullptr, false, newSeqLenShapeAttr);

    auto newRevSeqOp = rewriter.create<IE::ReverseSequenceOp>(origOp->getLoc(), in1Reshape, in2Reshape, newSeqAxisAttr,
                                                              newBatchAxisAttr);

    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newRevSeqOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

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
// SDPAOpConverter
//

auto fillWithOnes(vpux::NDTypeInterface inType, int64_t targetRank) {
    auto newInShape = to_small_vector(inType.getShape());
    int64_t inRank = newInShape.size();
    VPUX_THROW_UNLESS(targetRank >= inRank, "Input {0} has target rank {1}, less than current rank {2}", inType,
                      targetRank, inRank);
    const int64_t newInDims = targetRank - inRank;
    newInShape.insert(newInShape.begin(), newInDims, 1);
    return newInShape;
}

int getMaxInRank(mlir::Operation* origOp) {
    int maxRank = 0;
    for (size_t i = 0; i < origOp->getNumOperands(); i++) {
        if (origOp->getOperand(i)) {
            auto inType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(i).getType());
            auto inRank = inType.getRank();
            if (inRank > maxRank) {
                maxRank = inRank;
            }
        }
    }
    return maxRank;
}

class SDPAOpConverter final : public mlir::OpConversionPattern<IE::SDPAOp> {
public:
    SDPAOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SDPAOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SDPAOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SDPAOpConverter::matchAndRewrite(IE::SDPAOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    int maxInRank = getMaxInRank(origOp);
    if (maxInRank > 4) {
        VPUX_THROW("Unimplemented {0}D->4D convert", maxInRank);
    }
    const auto inQType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto inKType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(1).getType());
    const auto inVType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(2).getType());

    auto newInQShape = fillWithOnes(inQType, TARGET_TENSOR_DIM);
    auto newInKShape = fillWithOnes(inKType, TARGET_TENSOR_DIM);
    auto newInVShape = fillWithOnes(inVType, TARGET_TENSOR_DIM);

    const auto newInQShapeAttr = getIntArrayAttr(origOp->getContext(), newInQShape);
    const auto newInKShapeAttr = getIntArrayAttr(origOp->getContext(), newInKShape);
    const auto newInVShapeAttr = getIntArrayAttr(origOp->getContext(), newInVShape);

    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));

    const auto inQReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inQ"), origOp.getInputQ(),
                                                                 nullptr, false, newInQShapeAttr);
    const auto inKReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inK"), origOp.getInputK(),
                                                                 nullptr, false, newInKShapeAttr);
    const auto inVReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inV"), origOp.getInputV(),
                                                                 nullptr, false, newInVShapeAttr);
    mlir::Value maskReshape = nullptr;
    if (origOp.getInputMask()) {
        const auto inMType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputMask().getType());
        const auto newInMShape = fillWithOnes(inMType, TARGET_TENSOR_DIM);
        const auto newInMShapeAttr = getIntArrayAttr(origOp->getContext(), newInMShape);
        maskReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inM"), origOp.getInputMask(),
                                                           nullptr, false, newInMShapeAttr);
    }

    mlir::Value scaleReshape = nullptr;
    if (origOp.getInputScale()) {
        const auto inSType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputScale().getType());
        const auto newInSShape = fillWithOnes(inSType, TARGET_TENSOR_DIM);
        const auto newInSShapeAttr = getIntArrayAttr(origOp->getContext(), newInSShape);
        scaleReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inS"), origOp.getInputScale(),
                                                            nullptr, false, newInSShapeAttr);
    }

    auto newSDPAOp = rewriter.create<IE::SDPAOp>(origOp->getLoc(), inQReshape, inKReshape, inVReshape, maskReshape,
                                                 scaleReshape, origOp.getCausalAttr());
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newSDPAOp.getOutput(), nullptr, false, outShapeAttr);

    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

//
// SDPAExtendConverter
//

class SDPAExtendedOpConverter final : public mlir::OpConversionPattern<IE::SDPAExtendedOp> {
public:
    SDPAExtendedOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SDPAExtendedOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SDPAExtendedOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SDPAExtendedOpConverter::matchAndRewrite(IE::SDPAExtendedOp origOp, OpAdaptor,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
    int maxInRank = getMaxInRank(origOp);
    if (maxInRank > 4) {
        VPUX_THROW("Unimplemented {0}D->4D convert", maxInRank);
    }
    const auto inQType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto inKType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(1).getType());
    const auto inVType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(2).getType());

    auto newInQShape = fillWithOnes(inQType, TARGET_TENSOR_DIM);
    auto newInKShape = fillWithOnes(inKType, TARGET_TENSOR_DIM);
    auto newInVShape = fillWithOnes(inVType, TARGET_TENSOR_DIM);

    const auto newInQShapeAttr = getIntArrayAttr(origOp->getContext(), newInQShape);
    const auto newInKShapeAttr = getIntArrayAttr(origOp->getContext(), newInKShape);
    const auto newInVShapeAttr = getIntArrayAttr(origOp->getContext(), newInVShape);

    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));

    const auto inQReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inQ"), origOp.getInputQ(),
                                                                 nullptr, false, newInQShapeAttr);
    const auto inKReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inK"), origOp.getInputK(),
                                                                 nullptr, false, newInKShapeAttr);
    const auto inVReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inV"), origOp.getInputV(),
                                                                 nullptr, false, newInVShapeAttr);
    mlir::Value maskReshape = nullptr;
    if (origOp.getInputMask()) {
        const auto inMType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputMask().getType());
        const auto newInMShape = fillWithOnes(inMType, TARGET_TENSOR_DIM);
        const auto newInMShapeAttr = getIntArrayAttr(origOp->getContext(), newInMShape);
        maskReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inM"), origOp.getInputMask(),
                                                           nullptr, false, newInMShapeAttr);
    }

    mlir::Value scaleReshape = nullptr;
    if (origOp.getInputScale()) {
        const auto inSType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputScale().getType());
        const auto newInSShape = fillWithOnes(inSType, TARGET_TENSOR_DIM);
        const auto newInSShapeAttr = getIntArrayAttr(origOp->getContext(), newInSShape);
        scaleReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_inS"), origOp.getInputScale(),
                                                            nullptr, false, newInSShapeAttr);
    }

    auto newSDPAExtendedOp = rewriter.create<IE::SDPAExtendedOp>(origOp->getLoc(), inQReshape, inKReshape, inVReshape,
                                                                 maskReshape, scaleReshape, origOp.getPadSizeSAttr());
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newSDPAExtendedOp.getOutput(), nullptr, false,
                                                                 outShapeAttr);

    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

//
// RandomUniformConverter
//

class RandomUniformConverter final : public mlir::OpConversionPattern<IE::RandomUniformOp> {
public:
    RandomUniformConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::RandomUniformOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::RandomUniformOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RandomUniformConverter::matchAndRewrite(IE::RandomUniformOp origOp, OpAdaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::RandomUniformOp Operation '{1}'", getDebugName(), origOp->getLoc());

    // Build input ReshapeOp
    SmallVector<mlir::Value> newInputs;
    for (const auto& origInput : origOp.getInputs()) {
        const auto origInputType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());
        SmallVector<int64_t> origInputShape = to_small_vector(origInputType.getShape());
        const auto newInputShape = alignTileShapeRepeatsTo4D(std::move(origInputShape));
        const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newInputShape);

        auto inputReshape =
                rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origInput, nullptr, false, newInputShapeAttr);

        newInputs.emplace_back(inputReshape);
    }

    // Update output shape attr
    const auto origOutputShape = parseIntArrayAttr<int64_t>(origOp.getOutputShapeAttr());
    const auto newOutputShape = alignTileShapeRepeatsTo4D(std::move(origOutputShape));
    const auto newOutputShapeAttr = getIntArrayAttr(rewriter.getContext(), newOutputShape);

    // Update the RandomUniformOp
    auto newRandomUniformOp = rewriter.create<IE::RandomUniformOp>(origOp.getLoc(), newInputs[0], newInputs[1],
                                                                   newOutputShapeAttr, origOp.getOutputTypeAttr(),
                                                                   origOp.getGlobalSeedAttr(), origOp.getOpSeedAttr());

    // Reshape to original output shape
    const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), getShape(origOp.getOutput()));
    auto newOutputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), newRandomUniformOp.getOutput(),
                                                                 nullptr, false, outputShapeAttr);
    origOp.getOutput().replaceAllUsesWith(newOutputReshape);

    rewriter.eraseOp(origOp);

    _log.trace("[{0}] Replaced with 'IE::RandomUniformOp'", getDebugName());

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

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    const auto canBeRunOnNCE = config::isReduceOpSupportedOnNCE(origOp) && VPU::isNCEReduceSupported(origOp, logCb);

    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    auto newShape = to_small_vector(inType.getShape());
    auto newAxes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    auto inRank = newShape.size();

    if (inRank > TARGET_TENSOR_DIM) {
        std::tie(newShape, newAxes) = getMergedShapeAndAxes(newShape, newAxes);
        inRank = newShape.size();
    }

    if (inRank < TARGET_TENSOR_DIM) {
        // In case the operation could be execute on the NCE, append the new dimensions to the end, to maintain
        // compatibility (e.g. the reduction axis to remain unchanged)
        const int64_t newDims = TARGET_TENSOR_DIM - inRank;
        if (canBeRunOnNCE) {
            newShape.insert(newShape.end(), newDims, 1);
        } else {
            newShape.insert(newShape.begin(), newDims, 1);
            for (auto& axis : newAxes) {
                axis += newDims;
            }
        }
    }

    const auto newShapeAttr = getIntArrayAttr(origOp->getContext(), newShape);
    const auto axisValueAttr = getIntArrayAttr(origOp->getContext(), newAxes);
    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));

    const auto inReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                                nullptr, false, newShapeAttr);
    auto newReduceOp = rewriter.create<ReduceOp>(origOp->getLoc(), inReshape, /*axes*/ nullptr, axisValueAttr,
                                                 /*keepDims*/ mlir::UnitAttr::get(origOp.getContext()));
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newReduceOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

template <class ReduceOp>
auto isLegalReduceOp(ReduceOp reduceOp) {
    const auto inShape = mlir::cast<vpux::NDTypeInterface>(reduceOp.getOperand(0).getType()).getShape();
    const auto outShape = mlir::cast<vpux::NDTypeInterface>(reduceOp.getResult().getType()).getShape();
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

    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
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

    SmallVector<int64_t> adjustedInShape, adjustedRepeatValue;
    auto validAdjustedShape = getAdjustedShapeForTile(origOp);
    if (!validAdjustedShape.has_value()) {
        return mlir::failure();
    }
    std::tie(adjustedInShape, adjustedRepeatValue) = validAdjustedShape.value();

    // Build input ReshapeOp
    const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), std::move(adjustedInShape));
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);
    // Update the TileOp
    const auto repeatsOnNewShapeAttr = getIntArrayAttr(rewriter.getContext(), adjustedRepeatValue);
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
        const auto origInputType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());
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
// GRUGatesConverter
//

class GRUGatesConverter final : public mlir::OpConversionPattern<IE::GRUGatesOp> {
public:
    GRUGatesConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::GRUGatesOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GRUGatesOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GRUGatesConverter::matchAndRewrite(IE::GRUGatesOp origOp, OpAdaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::GRUGatesOp Operation '{1}'", getDebugName(), origOp->getLoc());

    // Build input ReshapeOp
    SmallVector<mlir::Value> newInputs;
    for (const auto& origInput : origOp.getInputs()) {
        const auto origInputType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());
        SmallVector<int64_t> origInputShape = to_small_vector(origInputType.getShape());
        const auto newInputShape = alignTileShapeRepeatsTo4D(std::move(origInputShape));
        const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newInputShape);

        auto inputReshape =
                rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origInput, nullptr, false, newInputShapeAttr);

        newInputs.emplace_back(inputReshape);
    }

    // Update the GRUGatesOp
    auto newGRUGatesOp =
            rewriter.create<IE::GRUGatesOp>(origOp.getLoc(), newInputs[0], newInputs[1], newInputs[2], newInputs[3]);

    // Reshape to original output shape
    for (const auto& output : origOp.getOutputs() | indexed) {
        const auto idx = checked_cast<unsigned>(output.index());
        auto origOutput = output.value();
        const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), getShape(origOutput));

        auto newOutputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), newGRUGatesOp.getOutputs()[idx],
                                                                     nullptr, false, outputShapeAttr);
        origOutput.replaceAllUsesWith(newOutputReshape);
    }

    rewriter.eraseOp(origOp);

    _log.trace("[{0}] Replaced with 'IE::GRUGatesOp'", getDebugName());

    return mlir::success();
}

//
// SplitConverter
//

class SplitConverter final : public mlir::OpConversionPattern<IE::SplitOp> {
public:
    SplitConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SplitOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitConverter::matchAndRewrite(IE::SplitOp origOp, OpAdaptor,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found '{1}' Operation '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto splitAxis = origOp.getAxisValue().value();
    const auto origInputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutputs()[0].getType());
    VPUX_THROW_UNLESS(origInputType.getRank() == origOutputType.getRank(),
                      "Split Op should has same input rank {0} and output rank {1}", origInputType.getRank(),
                      origOutputType.getRank());

    auto newInputShape = to_small_vector(origInputType.getShape());
    auto newOutputShape = to_small_vector(origOutputType.getShape());
    auto newSplitAxis = splitAxis;

    auto inItr = newInputShape.begin();
    auto outItr = newOutputShape.begin();
    while (inItr != newInputShape.end() && outItr != newOutputShape.end()) {
        auto idx = std::distance(newInputShape.begin(), inItr);
        if (idx != newSplitAxis && *inItr == 1) {
            if (idx < newSplitAxis) {
                newSplitAxis -= 1;
            }
            inItr = newInputShape.erase(inItr);
            outItr = newOutputShape.erase(outItr);
        } else {
            ++inItr;
            ++outItr;
        }

        if (newInputShape.size() == TARGET_TENSOR_DIM) {
            break;
        }
    }

    VPUX_THROW_UNLESS(newInputShape.size() == TARGET_TENSOR_DIM,
                      "Got illegal Split Op to 4D with in Shape {0} and Axis {1}", Shape(origInputType.getShape()),
                      splitAxis);

    // Build input ReshapeOp
    const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newInputShape);
    auto inputReshape =
            rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origOp.getInput(), nullptr, false, newInputShapeAttr);

    // Update the SplitOp
    const auto newAxisAttr = getIntAttr(rewriter.getContext(), newSplitAxis);
    auto newSplitOp = rewriter.create<IE::SplitOp>(origOp.getLoc(), inputReshape, nullptr, origOp.getNumSplitsAttr(),
                                                   newAxisAttr);

    // Reshape to original output shape
    for (const auto& output : origOp.getOutputs() | indexed) {
        const auto idx = checked_cast<unsigned>(output.index());
        auto origOutput = output.value();
        const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), to_small_vector(getShape(origOutput)));

        auto newOutputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), newSplitOp.getOutputs()[idx],
                                                                     nullptr, false, outputShapeAttr);
        origOutput.replaceAllUsesWith(newOutputReshape);
    }

    rewriter.eraseOp(origOp);

    _log.trace("[{0}] Replaced with 4D 'IE::SplitOp'", getDebugName());

    return mlir::success();
}

//
// RollConverter
//

class RollConverter final : public mlir::OpConversionPattern<IE::RollOp> {
public:
    RollConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::RollOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::RollOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RollConverter::matchAndRewrite(IE::RollOp origOp, OpAdaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found '{1}' Operation '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    const auto ctx = rewriter.getContext();
    const auto dataRank = static_cast<int64_t>(getShape(origOp.getData()).size());
    if (dataRank > TARGET_TENSOR_DIM) {
        _log.trace("cannot convert RollOp with rank > TARGET_TENSOR_DIM");
        return mlir::failure();
    }

    // For dataRank < TARGET_TENSOR_DIM, we expand the shape of data to 4D by
    // adding trivial axes from the left side, e.g., (X,Y) to (1,1,X,Y).
    // Then, the axes and shift values (already converted to positive) will be adjust for keep accurate.
    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getData().getType());
    auto shiftAndAxesOrFail =
            IE::getShiftAndAxesForRollOp(origOp.getLoc(), origOp.getShift(), origOp.getAxes(), origType.getShape());
    if (mlir::failed(shiftAndAxesOrFail)) {
        _log.trace("cannot convert RollOp without shift and axes");
        return mlir::failure();
    }
    auto shiftAndAxes = shiftAndAxesOrFail.value();
    auto shift = shiftAndAxes.shift;
    auto axes = shiftAndAxes.axes;
    if (shift.size() != axes.size()) {
        _log.trace("cannot convert RollOp with different size of shift and axes");
        return mlir::failure();
    }

    // create reshaped data input
    int64_t expandDimNum = TARGET_TENSOR_DIM - dataRank;
    const auto dataInput = origOp.getData();
    const auto dataInputType = mlir::cast<vpux::NDTypeInterface>(dataInput.getType());
    SmallVector<int64_t> dataInputShape = to_small_vector(dataInputType.getShape());
    SmallVector<int64_t> newDataInputShape;
    newDataInputShape.append(dataInputShape.begin(), dataInputShape.end());
    for (int64_t i = 0; i < expandDimNum; i++) {
        newDataInputShape.insert(newDataInputShape.begin(), 1);
    }
    const auto newDataInputShapeAttr = getIntArrayAttr(ctx, newDataInputShape);
    auto dataInputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_data"), origOp.getData(),
                                                                 nullptr, false, newDataInputShapeAttr);

    // adjust the axes value to the new data shape
    for (size_t i = 0; i < axes.size(); i++) {
        axes[i] += expandDimNum;
    }

    // create new shift and axes inputs
    const auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const SmallVector<int64_t> newShiftAndAxesShape = {1, 1, 1, static_cast<int64_t>(shift.size())};
    const auto newShiftAndAxesType =
            mlir::RankedTensorType::get(newShiftAndAxesShape, si32Type, getTensorAttr(ctx, /*order=*/nullptr, nullptr));

    auto createNewConst = [&](mlir::Location loc, ArrayRef<int64_t> data) {
        SmallVector<int32_t> newData;
        newData.reserve(data.size());
        for (const auto& val : data) {
            newData.push_back(static_cast<int32_t>(val));
        }
        return Const::createConst(rewriter, loc, newShiftAndAxesType, ArrayRef(newData));
    };

    const auto newShiftConst = createNewConst(origOp.getShift().getLoc(), shift);
    const auto newAxesConst = createNewConst(origOp.getAxes().getLoc(), axes);

    // create new RollOp and output reshape
    auto newRollOp = rewriter.create<IE::RollOp>(origOp->getLoc(), dataInputReshape, newShiftConst, newAxesConst);

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newRollOp.getOutput(), nullptr, false, outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

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
        const auto origInputType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());
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

    auto reshapeValue = [&](mlir::Value value, ShapeRef newShape, const std::string& suffix) -> mlir::Value {
        const auto valueShape = getShape(value);
        if (valueShape == newShape) {
            return value;
        }

        if (valueShape.isDynamic()) {
            const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(value.getType());
            VPUX_THROW_WHEN(boundedType == nullptr, "Expected to get BoundedTensorType at {0}", value.getLoc());

            return IE::createDynamicReshape(rewriter, appendLoc(value.getLoc(), suffix), value,
                                            boundedType.getDynamicShape());
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
        const Shape newBiasesShape{1, numDirections, 4, hiddenSize};
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
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
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
    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());

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

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), inputReshape, nullptr,
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

auto expandSoftmaxTo4DShape(const ArrayRef<int64_t>& inputShape, const int64_t& axis) {
    // Expand 2D/3D Softmax to 4D
    // 1. if Softmax has only 1 non-trivial dim and axis is on that dimension,
    //      put the dimension to W to keep the original method
    //    e.g. [1, 51] -> [1, 1, 1, 51]
    // 2. for other cases, put the non-1 dimensions to C and H
    //      to increase the possibility of multi-cluster and tiling
    //    e.g. [32, 10] -> [1, 32, 10, 1]
    //         [1, 32, 10] -> [1, 32, 10, 1]
    auto rank = static_cast<int64_t>(inputShape.size());
    auto isSingleDimSoftMax = [&]() {
        return llvm::all_of(irange(rank), [&](int64_t ind) {
            return inputShape[ind] == 1 || ind == axis;
        });
    };

    // Optimization for softmax kernel should make axis last dim.
    // Maintain axis last dim after being reshaped to 4D.
    auto isTwoDimAxisLastSoftMax = [&]() {
        return rank == 2 && axis == rank - 1;
    };

    SmallVector<int64_t> newInputShape;
    int64_t newAxis = axis;
    auto addDims = static_cast<int32_t>(TARGET_TENSOR_DIM - rank);
    if (isSingleDimSoftMax() || isTwoDimAxisLastSoftMax()) {
        // if original Softmax gets only 1 non-trivial axis or get 2D tensor with axis on the last dim,
        // keep the axis on the last dim after reshaping
        newInputShape = SmallVector<int64_t>(addDims, 1);
        for (auto i = 0; i < rank; i++) {
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

        for (auto i = 0; i < rank; i++) {
            newInputShape.push_back(inputShape[i]);
        }

        for (auto i = 0; i < addDims; i++) {
            newInputShape.push_back(1);
        }
    }
    return std::tuple<SmallVector<int64_t>, int64_t>(newInputShape, newAxis);
}

mlir::FailureOr<std::tuple<SmallVector<int64_t>, int64_t>> getNewSoftmaxParam(vpux::NDTypeInterface origType,
                                                                              int64_t axis) {
    // parse negative axis into positive value
    if (axis < 0) {
        axis += origType.getRank();
    }

    const auto inputShape = origType.getShape().raw();

    // for rank < TARGET_TENSOR_DIM, we expand it to 4D for new param
    if (origType.getRank() < TARGET_TENSOR_DIM) {
        return expandSoftmaxTo4DShape(inputShape, axis);
    }
    // for rank >= TARGET_TENSOR_DIM, we first merge axes to make it has no more than 3 non-trivial axes,
    // then we can use the same extension function to expand the softmax to 4D
    SmallVector<int64_t> nonTrivialShape = {1, 1, 1};
    int64_t newAxis = 1;  // Initialize newAxis to 1
    for (int i = 0; i < origType.getRank(); i++) {
        if (i < axis) {
            nonTrivialShape[0] *= inputShape[i];
        } else if (i == axis) {
            nonTrivialShape[1] = inputShape[i];
        } else {
            nonTrivialShape[2] *= inputShape[i];
        }
    }
    // Remove unnecessary dim from nonTrivialShape
    if (nonTrivialShape[0] == 1 && axis != 0) {
        nonTrivialShape.erase(nonTrivialShape.begin());
        newAxis--;
    }
    if (nonTrivialShape.back() == 1 && axis != origType.getRank() - 1) {
        nonTrivialShape.pop_back();
    }
    return expandSoftmaxTo4DShape(nonTrivialShape, newAxis);
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

bool is5DSoftmaxGroupBiggerThanTileCount(IE::SoftMaxOp origOp) {
    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    if (origType.getRank() != 5) {
        return false;
    }

    const auto module = getModuleOp(origOp);
    auto tileOp = config::getTileExecutor(module);
    const auto numOfTiles = tileOp.getCount();

    const auto inputShape = origType.getShape();
    auto batchSize = inputShape[DimsGroups5D::Act::G];

    return batchSize >= numOfTiles;
}

mlir::LogicalResult SoftmaxConverter::matchAndRewrite(IE::SoftMaxOp origOp, OpAdaptor,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Softmax Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    int64_t axis = origOp.getAxisInd();

    if (is5DSoftmaxGroupBiggerThanTileCount(origOp)) {
        return mlir::success();
    }

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

    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
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
    int64_t addDims = (inputRank < 5) ? static_cast<int64_t>(TARGET_TENSOR_DIM - inputRank) : 0;

    const auto createAxesAttr = [&](std::optional<mlir::ArrayAttr> axesAttr, bool isInput5D) {
        if (axesAttr.has_value()) {
            auto intArray = parseIntArrayAttr<int64_t>(axesAttr.value());
            for (auto& val : intArray) {
                if (isInput5D) {
                    if (val >= 1) {
                        val -= 1;
                    }
                }
                val += addDims;
            }

            if (isInput5D) {
                return getIntArrayAttr(this->getContext(), intArray);
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
            if (inputRank == 5) {
                SmallVector<int64_t> newSizes(intArray.begin(), intArray.end());
                return getIntArrayAttr(this->getContext(), newSizes);
            }
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

    SmallVector<int64_t> newInputShape;
    if (inputRank < 5) {
        newInputShape = SmallVector<int64_t>(addDims, 1);
        newInputShape.insert(newInputShape.end(), inputShape.begin(), inputShape.end());
    } else {
        const auto axesAttr = origOp.getAxesAttr();
        if (axesAttr.has_value()) {
            const auto axes = parseIntArrayAttr<int64_t>(axesAttr.value());
            for (const auto& axis : axes) {
                if (axis == 0 || axis == 1) {
                    VPUX_THROW("Unsupported 5D case: Scaling on axes 0 or 1 is not supported.");
                }
            }
            VPUX_THROW_UNLESS(axes.size() <= 3, "Unsupported 5D case: Scaling on more than 3 axes is not supported.");
        }
        newInputShape = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
        newInputShape.front() *= newInputShape[1];
        newInputShape.erase(newInputShape.begin() + 1);
        VPUX_THROW_UNLESS(newInputShape.size() == TARGET_TENSOR_DIM, "Interpolate 5D->4D conversion failed");
    }

    const auto newInputShapeAttr = getIntArrayAttr(this->getContext(), newInputShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);

    const auto attrs = origOp.getAttr();
    const auto newPadsBeginAttr = extendShapeWithValue(attrs.getPadsBegin(), 0);
    const auto newPadsEndAttr = extendShapeWithValue(attrs.getPadsEnd(), 0);
    const auto newAttr = IE::InterpolateAttr::get(this->getContext(), attrs.getMode(), attrs.getShapeCalcMode(),
                                                  attrs.getCoordMode(), attrs.getNearestMode(), attrs.getAntialias(),
                                                  newPadsBeginAttr, newPadsEndAttr, attrs.getCubeCoeff());

    const auto newAxesAttr = createAxesAttr(origOp.getAxesAttr(), inputRank == 5);
    const auto newSizesAttr = extendShapeWithValue(origOp.getSizesAttr(), 1);
    const auto newScalesAttr = extendShapeWithFloatValue(origOp.getScalesAttr(), 1.0);
    const auto newOffsetAttr = extendShapeWithValue(origOp.getTileOffsetAttr(), 0);
    const auto newInitInputDimAttr = extendShapeWithValue(origOp.getInitialInputDimsAttr(), 1);
    const auto newInitOutputDimAttr = extendShapeWithValue(origOp.getInitialOutputDimsAttr(), 1);

    auto newInterpOp = rewriter.create<IE::InterpolateOp>(origOp->getLoc(), inputReshape, nullptr, nullptr, nullptr,
                                                          newSizesAttr, newScalesAttr, newAxesAttr, newOffsetAttr,
                                                          newInitInputDimAttr, newInitOutputDimAttr, newAttr,
                                                          origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

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
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inShape = inType.getShape();
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(origOp.getIndices().getType());
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

    auto inputReshape = createReshapeOp(origOp.getInput(), ShapeRef(newInShape), "in");
    auto indicesReshape = createReshapeOp(origOp.getIndices(), ShapeRef(newIndicesShape), "indices");
    auto newGatherOp =
            rewriter.create<IE::GatherOp>(origOp.getLoc(), inputReshape, indicesReshape, nullptr,
                                          getIntAttr(ctx, newAxis), newBatchDim, getIntAttr(ctx, indicesRank));
    auto outputReshape = createReshapeOp(newGatherOp.getOutput(), getShape(origOp.getOutput()), "out");

    _log.trace("Replaced {0} with 4D tensor", origOp.getLoc());
    rewriter.replaceOp(origOp, outputReshape);

    return mlir::success();
}

//
// GatherNDConverter
//

class GatherNDConverter final : public mlir::OpConversionPattern<IE::GatherNDOp> {
public:
    GatherNDConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::GatherNDOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherNDOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GatherNDConverter::matchAndRewrite(IE::GatherNDOp origOp, OpAdaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    _log.trace("[{0}] Found GatherND Operation at '{1}'", getDebugName(), origOp->getLoc());

    const auto inType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inShape = inType.getShape();
    const auto indicesType = mlir::cast<NDTypeInterface>(origOp.getIndices().getType());
    const auto indicesShape = indicesType.getShape();
    const auto batchDim = origOp.getBatchDims();
    const auto coordRank = indicesShape.back();

    auto fuseDims = [](auto begin, auto end) -> int64_t {
        return std::accumulate(begin, end, int64_t(1), std::multiplies<int64_t>());
    };

    // Convert GatherND Op to a 4D tensor using the following dimensional rules:
    //    Input Data: [fixed_dim, batch_dim_size, coord_dim_size, after_coord_dim_size]
    //    Indice:     [fixed_dim, batch_dim_size, indices_dim_size, after_indices_dim_size]
    //    Batch_dims: 2
    // Example transformations:
    // 1. Original:    Input: 1000x256x10x15, Indices: 25x125x3, Batch_dims: 0, Output: 25x125x15
    //    Transformed: Input: 1x1x2560000x15x16, Indices: 1x1x3125x3, Output: 1x1x3125x15
    //                 Batch_dims: 2, original_shape: [1, 1, 1000, 256, 10, 15]
    // 2. Original:    Input: 5x6x7, Indices: 5x3x1, Batch_dims: 1, Output: 5x3x7
    //    Transformed: Input: 1x5x6x7, Indices: 1x5x3x1, Batch_dims: 2, Output: 1x5x3x7
    //                 Batch_dims: 2, original_shape: [1, 5, 6, 7]
    // 3. Original:    Input: 1x16x32x56x16, Indices: 1x16x14580x2, Batch_dims: 2, Output: 1x16x14580x16
    //    Transformed: Input: 1x16x1792x16, Indices: 1x16x14580x2, Batch_dims: 2, Output: 1x16x14580x16
    //                 Batch_dims: 2, original_shape: [1, 16, 32, 56, 16]

    const auto fixedDim = int64_t(1);
    const auto batchDimSize = fuseDims(inShape.begin(), inShape.begin() + batchDim);
    const auto coordDimSize = fuseDims(inShape.begin() + batchDim, inShape.begin() + batchDim + coordRank);
    const auto afterCoordDimSize = fuseDims(inShape.begin() + batchDim + coordRank, inShape.end());
    SmallVector<int64_t> newInShape{fixedDim, batchDimSize, coordDimSize, afterCoordDimSize};

    const auto afterIndicesDimSize = fuseDims(indicesShape.begin() + batchDim, indicesShape.end() - 1);
    SmallVector<int64_t> newIndicesShape{fixedDim, batchDimSize, afterIndicesDimSize, coordRank};

    const auto newBatchDim = 2;

    auto createReshapeOp = [&](mlir::Value input, ShapeRef shape, StringRef locSuffix) -> mlir::Value {
        auto shapeAttr = getIntArrayAttr(ctx, shape);
        return rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("reshape_{0}"), locSuffix), input,
                                                    nullptr, false, shapeAttr);
    };

    SmallVector<int64_t> originalFusedShape{fixedDim, batchDimSize};
    originalFusedShape.append(inShape.begin() + batchDim, inShape.begin() + batchDim + coordRank);
    originalFusedShape.push_back(afterCoordDimSize);

    auto inputReshape = createReshapeOp(origOp.getInput(), ShapeRef(newInShape), "in");
    auto indicesReshape = createReshapeOp(origOp.getIndices(), ShapeRef(newIndicesShape), "indices");
    auto newGatherNDOp =
            rewriter.create<IE::GatherNDOp>(origOp.getLoc(), inputReshape, indicesReshape, getIntAttr(ctx, newBatchDim),
                                            getIntArrayAttr(ctx, originalFusedShape));
    auto outputReshape = createReshapeOp(newGatherNDOp.getOutput(), getShape(origOp.getOutput()), "out");

    _log.trace("Replaced {0} with 4D tensor", newGatherNDOp->getLoc());
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

    auto inputReshape = createReshapeOp(origOp.getInput(), ShapeRef(newInputShape), "in");
    auto indicesReshape = createReshapeOp(origOp.getIndices(), ShapeRef(newIndicesShape), "indices");
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

    auto inShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getShape();
    auto outShape = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getShape();

    Shape newInputShape;
    Shape newOutputShape;

    for (size_t i = 1; i < inShape.size(); i++) {
        newInputShape.push_back(inShape.raw()[i]);
        newOutputShape.push_back(outShape.raw()[i]);
    }

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);

    auto newBroadcast = IE::createBroadcast(rewriter, origOp->getLoc(), inputReshape, newOutputShape,
                                            origOp.getAxesMapping(), origOp.getModeAttr());
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newBroadcast, nullptr, false,
                                                                 getIntArrayAttr(getContext(), outShape.raw()));
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::BroadcastOp'", getDebugName());
    return mlir::success();
}

//
// SelectOpConverter
//
class SelectConverter final : public mlir::OpConversionPattern<IE::SelectOp> {
public:
    SelectConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SelectOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SelectOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SelectConverter::matchAndRewrite(IE::SelectOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::SelectOp Operation '{1}'", getDebugName(), origOp->getLoc());
    const auto inType1 = mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType());
    const auto inType2 = mlir::cast<vpux::NDTypeInterface>(origOp.getInput2().getType());
    const auto inType3 = mlir::cast<vpux::NDTypeInterface>(origOp.getInput3().getType());
    const auto inShape1 = SmallVector<int64_t>(inType1.getShape().raw());
    const auto inShape2 = SmallVector<int64_t>(inType2.getShape().raw());
    const auto inShape3 = SmallVector<int64_t>(inType3.getShape().raw());

    const auto inRank1 = inShape1.size();

    const auto inRank2 = inShape2.size();
    const auto inRank3 = inShape3.size();

    SmallVector<int64_t> newShape1;
    SmallVector<int64_t> newShape2;
    SmallVector<int64_t> newShape3;

    if (inRank1 <= 4) {
        // insert leading 1s up to 4D
        auto newDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inRank1);
        auto to4DShape = [=](ArrayRef<int64_t> iShape, SmallVector<int64_t>& oShape) {
            oShape.insert(oShape.end(), newDims, 1);
            oShape.append(iShape.begin(), iShape.end());
        };
        to4DShape(inShape1, newShape1);

    } else if (inRank1 == 5) {
        auto mergeDim = 0;
        //=> new 'shape'
        newShape1 = decltype(newShape1){inShape1.begin(), inShape1.end()};
        newShape1[mergeDim] *= newShape1[mergeDim + 1];
        newShape1.erase(newShape1.begin() + mergeDim + 1);

        VPUX_THROW_UNLESS(newShape1.size() == TARGET_TENSOR_DIM, "Select 5D->4D conversion failed");
    } else {
        VPUX_THROW("Unimplemented {0}D->4D convert", inRank1);
    }

    if (inRank2 <= 4) {
        // insert leading 1s up to 4D
        auto newDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inRank2);
        auto to4DShape = [=](ArrayRef<int64_t> iShape, SmallVector<int64_t>& oShape) {
            oShape.insert(oShape.end(), newDims, 1);
            oShape.append(iShape.begin(), iShape.end());
        };
        to4DShape(inShape2, newShape2);

    } else if (inRank2 == 5) {
        auto mergeDim = 0;
        //=> new 'shape'
        newShape2 = decltype(newShape2){inShape2.begin(), inShape2.end()};
        newShape2[mergeDim] *= newShape2[mergeDim + 1];
        newShape2.erase(newShape2.begin() + mergeDim + 1);

        VPUX_THROW_UNLESS(newShape2.size() == TARGET_TENSOR_DIM, "Select 5D->4D conversion failed");
    } else {
        VPUX_THROW("Unimplemented {0}D->4D convert", inRank2);
    }

    if (inRank3 <= 4) {
        // insert leading 1s up to 4D
        auto newDims = static_cast<int64_t>(TARGET_TENSOR_DIM - inRank3);
        auto to4DShape = [=](ArrayRef<int64_t> iShape, SmallVector<int64_t>& oShape) {
            oShape.insert(oShape.end(), newDims, 1);
            oShape.append(iShape.begin(), iShape.end());
        };
        to4DShape(inShape3, newShape3);

    } else if (inRank3 == 5) {
        auto mergeDim = 0;
        //=> new 'shape'
        newShape3 = decltype(newShape3){inShape3.begin(), inShape3.end()};
        newShape3[mergeDim] *= newShape3[mergeDim + 1];
        newShape3.erase(newShape3.begin() + mergeDim + 1);

        VPUX_THROW_UNLESS(newShape3.size() == TARGET_TENSOR_DIM, "Select 5D->4D conversion failed");
    } else {
        VPUX_THROW("Unimplemented {0}D->4D convert", inRank3);
    }

    const auto newShapeAttr1 = getIntArrayAttr(getContext(), newShape1);
    const auto newShapeAttr2 = getIntArrayAttr(getContext(), newShape2);
    const auto newShapeAttr3 = getIntArrayAttr(getContext(), newShape3);
    auto inReshape1 = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput1(), nullptr,
                                                           false, newShapeAttr1);
    auto inReshape2 = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput2(), nullptr,
                                                           false, newShapeAttr2);
    auto inReshape3 = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput3(), nullptr,
                                                           false, newShapeAttr3);

    auto newSelectOp = rewriter.create<IE::SelectOp>(origOp->getLoc(), inReshape1, inReshape2, inReshape3,
                                                     origOp.getAutoBroadcastAttr());

    const auto outShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.getOutput()));
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newSelectOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("[{0}] Replaced with 'IE::SelectOp'", getDebugName());

    return mlir::success();
}

//
// InterleavedDimsRewriter
//

class InterleavedDimsRewriter final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    InterleavedDimsRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx), _log(log) {
        this->setDebugName("InterleavedDimsRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InterleavedDimsRewriter::matchAndRewrite(mlir::Operation* eltwiseOp,
                                                             mlir::PatternRewriter& rewriter) const {
    // Treat the case where:
    // Shape 1: N x C x D x H x W  Could be converted to 4D
    // Shape 2: N x 1 x D x 1 x W  Can't be converted to 4D because the trivial dims aren't adjacent.
    // A simple solution for Eltwise ops is to broadcast Shape 2 and then proceed as usual.
    _log.trace("Found '{0}' Operation at '{1}'", eltwiseOp->getName(), eltwiseOp->getLoc());

    if (const auto inputCount = eltwiseOp->getOperands().size(); inputCount != 2) {
        // If needed, this could be relaxed for specific ops with >2 inputs.
        _log.nest().trace("Match failed: Expected operation with 2 operands but got {0} operands", inputCount);
        return mlir::failure();
    }

    auto input1 = eltwiseOp->getOperand(0);
    auto input2 = eltwiseOp->getOperand(1);

    const auto input1Type = mlir::cast<vpux::NDTypeInterface>(input1.getType());
    const auto input2Type = mlir::cast<vpux::NDTypeInterface>(input2.getType());
    const auto input1Rank = input1Type.getRank();
    const auto input2Rank = input2Type.getRank();

    if (input1Rank != input2Rank) {
        _log.nest().trace("Match failed: Expected inputs with the same rank but got {0} and {1}", input1Rank,
                          input2Rank);
        return mlir::failure();
    }

    if (input1Rank <= 4) {
        _log.nest().trace("Match failed: No reason to broadcast");
        return mlir::failure();
    }

    // Counts the number of unfuseable regions: N x 1 x D x 1 x W -> 5 regions
    static const auto getInterleavedRegionCount = [](ArrayRef<int64_t> shape) {
        size_t regionCount = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            if (isTrivialDim(shape[i]) != isTrivialDim(shape[i + 1])) {
                ++regionCount;
            }
        }
        return regionCount;
    };

    const auto loc = eltwiseOp->getLoc();
    const auto targetShape = getShape(eltwiseOp->getResult(0));

    auto modified = false;
    if (getInterleavedRegionCount(input1Type.getShape().raw()) > 4) {
        input1 = IE::createBroadcast(rewriter, appendLoc(loc, "broadcast_lhs"), input1, targetShape);
        _log.nest().trace("Broadcast input 1 to {0}", targetShape);
        modified = true;
    }
    if (getInterleavedRegionCount(input2Type.getShape().raw()) > 4) {
        input2 = IE::createBroadcast(rewriter, appendLoc(loc, "broadcast_rhs"), input2, targetShape);
        _log.nest().trace("Broadcast input 2 to {0}", targetShape);
        modified = true;
    }

    if (modified == false) {
        _log.nest().trace("Match failed: No reason to broadcast");
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(eltwiseOp->getOperands(), SmallVector{input1, input2});
    auto newEltwiseOp = rewriter.clone(*eltwiseOp, mapper);
    rewriter.replaceOp(eltwiseOp, newEltwiseOp->getResults());

    _log.nest().trace("Successful rewrite");
    return mlir::success();
}

//
// ReverseConverter
//

class ReverseConverter final : public mlir::OpConversionPattern<IE::ReverseOp> {
public:
    ReverseConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ReverseOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::pair<Shape, Shape> mergeAxes(ShapeRef inputShape, ShapeRef axes) {
    const auto inputRank = inputShape.size();
    VPUX_THROW_UNLESS(std::all_of(axes.begin(), axes.end(),
                                  [&](int64_t idx) {
                                      return idx >= 0 && idx < checked_cast<int64_t>(inputRank);
                                  }),
                      "Axis value out of range in inputShape");

    SmallVector<int64_t> mergedShape;
    SmallVector<int64_t> updatedAxes;
    int64_t currentProduct = 1;
    bool isMerging = false;

    std::unordered_set<int64_t> axesSet(axes.begin(), axes.end());
    for (size_t idx = 0; idx < inputRank; ++idx) {
        if (axesSet.find(idx) != axesSet.end()) {
            if (isMerging) {
                mergedShape.push_back(currentProduct);
                currentProduct = 1;
                isMerging = false;
            }
            mergedShape.push_back(inputShape[Dim(idx)]);
            updatedAxes.push_back(mergedShape.size() - 1);
        } else {
            currentProduct *= inputShape[Dim(idx)];
            isMerging = true;
        }
    }

    if (isMerging) {
        mergedShape.push_back(currentProduct);
    }

    return std::make_pair(Shape(mergedShape), Shape(updatedAxes));
}

mlir::LogicalResult ReverseConverter::matchAndRewrite(IE::ReverseOp origOp, OpAdaptor,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found '{1}' Operation '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ctx = rewriter.getContext();
    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto inputRank = inputType.getRank();
    const auto axes = parseIntArrayAttr<int64_t>(origOp.getAxisValueAttr());

    Shape targetShape = Shape(inputShape.raw());
    Shape targetAxes = Shape(axes);
    if (inputRank > 4) {
        auto mergedInfo = mergeAxes(targetShape, targetAxes);
        targetShape = mergedInfo.first;
        targetAxes = mergedInfo.second;
        VPUX_THROW_UNLESS(targetShape.size() <= TARGET_TENSOR_DIM, "Cannot convert ReverseOp to 4D");
    }

    int64_t expandDimNum = TARGET_TENSOR_DIM - targetShape.size();
    SmallVector<int64_t> newInputShape(targetShape.begin(), targetShape.end());
    newInputShape.resize(newInputShape.size() + expandDimNum, 1);

    const auto newInputShapeAttr = getIntArrayAttr(ctx, newInputShape);
    const auto newAxesAttr = getIntArrayAttr(ctx, targetAxes);

    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_input"), origOp.getInput(),
                                                             nullptr, false, newInputShapeAttr);

    auto newReverseOp =
            rewriter.create<IE::ReverseOp>(origOp->getLoc(), inputReshape, nullptr, newAxesAttr, origOp.getModeAttr());

    const auto outputShapeAttr = getIntArrayAttr(ctx, getShape(origOp.getOutput()));
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newReverseOp.getOutput(), nullptr, false,
                                                                 outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

//
// NormalizeL2Converter
//

class NormalizeL2Converter final : public mlir::OpConversionPattern<IE::NormalizeL2Op> {
    using OpAdaptor = typename mlir::OpConversionPattern<IE::NormalizeL2Op>::OpAdaptor;

public:
    NormalizeL2Converter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::NormalizeL2Op>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::NormalizeL2Op origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NormalizeL2Converter::matchAndRewrite(IE::NormalizeL2Op origOp, OpAdaptor,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
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
    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getResult()));

    const auto inReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getOperand(0),
                                                                nullptr, false, newShapeAttr);

    auto newNormalizeOp = rewriter.create<IE::NormalizeL2Op>(
            origOp->getLoc(), inReshape, /*axes*/ nullptr, axisValueAttr, origOp.getEpsAttr(), origOp.getEpsModeAttr());

    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newNormalizeOp.getResult(), nullptr, false,
                                                                 outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");
    return mlir::success();
}

//
// safeRunOnFunc
//

auto buildReshapeMaterializer(StringRef locSuffix) {
    const auto reshapeFunc = [=](mlir::OpBuilder& builder, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                                 mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        const auto newLoc = appendLoc(loc, locSuffix);

        // TODO: E#-171827 It might be beneficial to use AffineReshape for all cases, as it has well defined semantics
        // for per-axis quantized types and layout information propagation.
        const auto inputType = mlir::cast<NDTypeInterface>(inputs[0].getType());
        const auto isPerAxisQuantized = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType());
        const auto outShapeAttr = builder.getI64ArrayAttr(dstType.getShape());

        if (!isPerAxisQuantized) {
            if (const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(dstType)) {
                const auto ctx = builder.getContext();
                const auto outputShape = boundedType.getDynamicShape();
                const auto shape = extractShape(outputShape);
                const auto bounds = outputShape.toRepresentation();
                const auto outputRank = checked_cast<int64_t>(outputShape.size());
                const auto shapeType = mlir::RankedTensorType::get({outputRank}, getSInt32Type(ctx));
                const auto shapeValues = IE::replaceDynamicDimsWithValue<int32_t>(shape.raw(), -1);

                const auto shapeTensor =
                        Const::createConst<int32_t>(builder, appendLoc(loc, "shape"), shapeType, shapeValues);

                auto ndType = mlir::cast<vpux::NDTypeInterface>(dstType);
                auto newType = Core::BoundedTensorType::get(
                        vpux::getTensorType(shape, ndType.getElementType(), vpux::DimsOrder::fromNumDims(shape.size()),
                                            ndType.getMemSpace(), /*Bounds=*/{}, {}),
                        vpux::BoundsRef(bounds));

                return builder.createOrFold<IE::DynamicReshapeOp>(newLoc, newType, inputs.front(), shapeTensor,
                                                                  builder.getI64ArrayAttr(shape.raw()),
                                                                  builder.getI64ArrayAttr(bounds.raw()), nullptr);
            }
            return builder.createOrFold<IE::ReshapeOp>(newLoc, inputs.front(), nullptr, false, outShapeAttr);
        }

        // If we have a per-axis quantized type, we need to use AffineReshapeOp because it can properly handle the axis
        // change.
        SmallVector<SmallVector<int64_t>> dimMapping;

        const auto dimOffset = dstType.getRank() - inputType.getRank();
        if (dimOffset >= 0) {
            dimMapping.push_back({});
            // map dimension s0 to 1 x ... x 1 x s0
            for (int64_t i = 0; i <= dimOffset; ++i) {
                dimMapping.front().push_back(i);
            }
            // map the remaining dimensions si to s(i + dimOffset)
            for (int64_t i = 1; i < inputType.getRank(); ++i) {
                dimMapping.push_back({i + dimOffset});
            }
        } else {
            // map dimensions s0, ..., s(-dimOffset - 1) to s0
            for (int64_t i = 0; i < -dimOffset; ++i) {
                dimMapping.push_back({0});
            }
            // map dimensions s(-dimOffset), ..., s(inputType.getRank() - 1) to s0, s1, ...
            for (int64_t i = -dimOffset; i < inputType.getRank(); ++i) {
                dimMapping.push_back({i + dimOffset});
            }
        }

        const auto dimMappingAttr = getIntArrayOfArray(dstType.getContext(), dimMapping);
        return builder.createOrFold<IE::AffineReshapeOp>(newLoc, inputs.front(), dimMappingAttr, outShapeAttr);
    };
    return reshapeFunc;
}

mlir::LogicalResult ConvertShapeTo4DPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (forceConvertGatherTo4D.hasValue()) {
        _forceConvertGatherTo4D = forceConvertGatherTo4D.getValue();
    }

    return mlir::success();
}

void ConvertShapeTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface type) {
        return callOnShapeOf(type, [&](const auto& shape) {
            const auto reifiedShape = reifyShape(shape);
            const auto dimMapper = getDimMapGeneric(reifiedShape);
            auto newShape = alignShapeTo4D(shape, dimMapper, false);
            if (shape == newShape) {
                return type;
            }

            auto elemType = type.getElementType();
            if (const auto qElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
                VPUX_THROW_UNLESS(newShape.size() >= shape.size(), "Unexpected rank reduction: {0} -> {1}", shape,
                                  newShape);
                const auto rankDiff = newShape.size() - shape.size();
                elemType = changeAxis(qElemType, qElemType.getQuantizedDimension() + rankDiff);
            }

            return type.changeTypeComponents(TypeComponents()
                                                     .setShapeWithRepresentation(std::move(newShape))
                                                     .setElementType(elemType)
                                                     .setDimsOrder(DimsOrder::fromNumDims(TARGET_TENSOR_DIM)));
        });
    });
    typeConverter.addSourceMaterialization(buildReshapeMaterializer("source"));
    typeConverter.addTargetMaterialization(buildReshapeMaterializer("target"));
    typeConverter.addArgumentMaterialization(buildReshapeMaterializer("argument"));

    mlir::TypeConverter scaleShiftTypeConverter;
    // TODO: #143748 consider change ConvertScaleShiftToDWPass
    scaleShiftTypeConverter.addConversion([](vpux::NDTypeInterface type) {
        const auto shape = type.getShape();
        if (shape.size() == 3 && shape[Dim(0)] == 1 && shape[Dim(1)] > 1) {
            auto newShape = copyShape(shape);
            newShape.insert(newShape.begin() + 2, 1);  // 1xHxW -> 1xHx1xW
            return type.changeShape(newShape);
        }
        const auto reifiedShape = reifyShape(shape);
        auto dimMapper = getDimMapGeneric(reifiedShape);
        const auto alignedShape = alignShapeTo4D(shape, dimMapper, false);
        return type.changeShape(alignedShape);
    });
    scaleShiftTypeConverter.addSourceMaterialization(buildReshapeMaterializer("scale_shift_source"));
    scaleShiftTypeConverter.addTargetMaterialization(buildReshapeMaterializer("scale_shift_target"));
    scaleShiftTypeConverter.addArgumentMaterialization(buildReshapeMaterializer("scale_shift_arg"));

    // TODO(E#117111): the below checks are organized in a way to skip the pass
    // if op has operands/results with dynamic shapes. Converting dynamically-shaped tensor to 4D
    // will be addressed separately
    const auto isLegalOp = [&](mlir::Operation* op) {
        return IE::hasDynamicTensors(op) || typeConverter.isLegal(op);
    };

    const auto isLegalDynamicOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    // TODO: E#-171827 Ideally we want to use `isLegalOp` also for quantized ops, but for now we don't modify per-tensor
    // quant ops, only per-axis.
    const auto isLegalQuantOp = [&](mlir::Operation* op) {
        if (!isPerAxisQuantizedOp(op)) {
            return true;
        }

        return IE::hasDynamicTensors(op) || typeConverter.isLegal(op);
    };

    const auto isLegalFqOp = [&](IE::FakeQuantizeOp op) {
        const auto inShape = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getShape();
        const auto outShape = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getShape();

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

    const auto isLegalSDPAOp = [&](IE::SDPAOp op) {
        const auto inQRank = mlir::cast<vpux::NDTypeInterface>(op.getInputQ().getType()).getRank();
        const auto inKRank = mlir::cast<vpux::NDTypeInterface>(op.getInputK().getType()).getRank();
        const auto inVRank = mlir::cast<vpux::NDTypeInterface>(op.getInputV().getType()).getRank();
        bool isLegalSDPA = inQRank == TARGET_TENSOR_DIM && inKRank == TARGET_TENSOR_DIM && inVRank == TARGET_TENSOR_DIM;
        if (op.getInputMask()) {
            const auto inMRank = mlir::cast<vpux::NDTypeInterface>(op.getInputMask().getType()).getRank();
            isLegalSDPA = isLegalSDPA && inMRank == TARGET_TENSOR_DIM;
        }
        if (op.getInputScale()) {
            const auto inSRank = mlir::cast<vpux::NDTypeInterface>(op.getInputScale().getType()).getRank();
            isLegalSDPA = isLegalSDPA && inSRank == TARGET_TENSOR_DIM;
        }
        return isLegalSDPA;
    };

    const auto isLegalSDPAExtendedOp = [&](IE::SDPAExtendedOp op) {
        const auto inQRank = mlir::cast<vpux::NDTypeInterface>(op.getInputQ().getType()).getRank();
        const auto inKRank = mlir::cast<vpux::NDTypeInterface>(op.getInputK().getType()).getRank();
        const auto inVRank = mlir::cast<vpux::NDTypeInterface>(op.getInputV().getType()).getRank();
        bool isLegalSDPA = inQRank == TARGET_TENSOR_DIM && inKRank == TARGET_TENSOR_DIM && inVRank == TARGET_TENSOR_DIM;
        if (op.getInputMask()) {
            const auto inMRank = mlir::cast<vpux::NDTypeInterface>(op.getInputMask().getType()).getRank();
            isLegalSDPA = isLegalSDPA && inMRank == TARGET_TENSOR_DIM;
        }
        if (op.getInputScale()) {
            const auto inSRank = mlir::cast<vpux::NDTypeInterface>(op.getInputScale().getType()).getRank();
            isLegalSDPA = isLegalSDPA && inSRank == TARGET_TENSOR_DIM;
        }
        return isLegalSDPA;
    };

    const auto isLegalEltwiseOp = [&](mlir::Operation* op) {
        if (op->getNumOperands() < 2) {
            return true;
        }
        return allOperandsAre4D(op);
    };

    const auto is4DLegalOp = [&](mlir::Operation* op) {
        const auto inShape = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getShape();
        const auto outShape = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape();
        return inShape.size() == TARGET_TENSOR_DIM || outShape.isDynamic();
    };

    const auto isLegalSoftMaxOp = [&](IE::SoftMaxOp op) {
        if (is5DSoftmaxGroupBiggerThanTileCount(op)) {
            return true;
        }
        return is4DLegalOp(op);
    };

    const auto isLegalTransposeOp = [&](IE::TransposeOp op) {
        const auto origType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
        // Cannot handle shape after been reduced is still bigger than TARGET_TENSOR_DIM now.
        // Will insert 1 before mergedShape, so mergedShape should be smaller than TARGET_TENSOR_DIM.
        auto mergedShape =
                vpux::getMergedPermutationAndShape(origType, op.getOrderValue().value(), TARGET_TENSOR_DIM).second;
        const auto inShape = getShape(op.getInput());

        return mergedShape.size() >= TARGET_TENSOR_DIM || origType.getRank() == TARGET_TENSOR_DIM ||
               inShape.isDynamic();
    };

    const auto isLegalBroadcastOp = [&](IE::BroadcastOp op) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
        auto outShape = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getShape();

        return !(op.getMode() == IE::BroadcastType::BIDIRECTIONAL && inType.getRank() == 5 &&
                 inType.getShape()[Dims4D::Act::N] == 1 && outShape[Dims4D::Act::N] == 1);
    };

    const auto isLegalGatherOp = [&](IE::GatherOp op) {
        if (!op.getAxisValue().has_value()) {
            return true;
        }

        const auto axis = op.getAxisValue().value();
        const auto inShape = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getShape();
        // The purpose of converting the Gather Op to 4D is to enable Multi Cluster execution
        // There are already several optimizations for the Gather Op, such as DDR Access and GatherDMA
        // The Gather software kernel is optimized and performs well when the axis is the highest dimension
        // Here, only the cases where the axis is not the highest dimension will undergo 4D conversion
        // This is because performance regressions were observed in CI, requiring further debugging
        const auto areDimsBeforeAxisOne = std::all_of(inShape.begin(), inShape.begin() + axis, [](int dim) {
            return dim == 1;
        });
        // E#180631: This pass is supposed to convert every op to 4D shape, however, this is not the case for Gather.
        //           In WS, GatherOp has to be 4D for the IR to be legal while DefaultHW can manage without because
        //           these ops get folded in the context of constants, after being turned into transformations.
        //           The proper solution is to make it universally 4D and fix any performance regressions.
        if (areDimsBeforeAxisOne && !_forceConvertGatherTo4D) {
            return true;
        }

        const auto indicesShape = mlir::cast<vpux::NDTypeInterface>(op.getIndices().getType()).getShape();
        const auto outShape = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getShape();

        return inShape.size() == TARGET_TENSOR_DIM && outShape.size() == TARGET_TENSOR_DIM &&
               indicesShape.size() == TARGET_TENSOR_DIM;
    };

    const auto isLegalGatherNDOp = [&](IE::GatherNDOp op) {
        auto inShape = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getShape();
        auto indicesShape = mlir::cast<vpux::NDTypeInterface>(op.getIndices().getType()).getShape();
        auto outShape = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getShape();
        auto batchDims = op.getBatchDims();

        const auto isDynamicShape = outShape.isDynamic();
        const auto isTargetTensorRank = inShape.size() == TARGET_TENSOR_DIM &&
                                        indicesShape.size() == TARGET_TENSOR_DIM &&
                                        outShape.size() == TARGET_TENSOR_DIM;
        const auto isSingleIndicesDim = indicesShape.size() == checked_cast<size_t>(batchDims + 2);
        const auto isSingleBatchSize = inShape.front() == 1;

        // - Scenario where Indices dimension is not single:
        //   Shape needs to change even if it is already 4D for simple tiling logic
        //   Input: 1x2x3x4, Indices: 1x5x6x2, Batch_dims: 1, Output: 1x5x6x4
        // - Scenario where batch dimension is not single:
        //   Shape needs to change even if it is already 4D to enable MC (SOK)
        //   Input: 2x3x4x5, Indices: 2x3x7x1, Batch_dims: 2, Output: 2x3x7x5
        return isDynamicShape || (isTargetTensorRank && isSingleIndicesDim && isSingleBatchSize);
    };

    const auto isLegalGatherElementsOp = [&](IE::GatherElementsOp op) {
        // The purpose of converting the GatherElements Op to 4D is to enable Multi Cluster execution here. Since
        // the input, output and indices have same rank, so only check the input shape here.
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

    const auto isLegalSplitOp = [&](IE::SplitOp op) {
        if (!op.getAxisValue().has_value()) {
            return true;
        }

        const auto inShape = getShape(op.getInput());
        const auto inShapeRank = static_cast<int64_t>(inShape.size());
        if (inShapeRank <= TARGET_TENSOR_DIM) {
            return true;
        }

        auto splitAxis = op.getAxisValue().value();
        int64_t numOfSingleDim = 0;
        for (auto idx = 0; idx < inShapeRank; ++idx) {
            if (idx != splitAxis && inShape[Dim(idx)] == 1) {
                ++numOfSingleDim;
            }
        }

        // Only process original ranks larger than TARGET_TENSOR_DIM
        // and only remove dimensions with a size of one to convert it to TARGET_TENSOR_DIM
        return (inShapeRank - numOfSingleDim) > TARGET_TENSOR_DIM;
    };

    const auto isLegalRollOp = [&](IE::RollOp op) {
        // The purpose of converting RollOp to 4D is to enable Multi Cluster execution here.
        // Currently we only support case whose dataRank == TARGET_TENSOR_DIM
        const auto dataRank = static_cast<int64_t>(getShape(op.getData()).size());
        const auto shiftRank = static_cast<int64_t>(getShape(op.getShift()).size());
        const auto axesRank = static_cast<int64_t>(getShape(op.getAxes()).size());

        // If the rank of data/shift/axes are all TARGET_TENSOR_DIM, the op is legal
        return (dataRank == TARGET_TENSOR_DIM && shiftRank == TARGET_TENSOR_DIM && axesRank == TARGET_TENSOR_DIM);
    };

    const auto isLegalTileOp = [](IE::TileOp op) {
        auto validAdjustedShape = getAdjustedShapeForTile(op);
        return !validAdjustedShape.has_value();
    };

    const auto isLegalReverseOp = [&](IE::ReverseOp op) {
        const auto inputType = mlir::cast<NDTypeInterface>(op.getInput().getType());

        if (inputType.getRank() < TARGET_TENSOR_DIM) {
            return false;
        } else if (inputType.getRank() == TARGET_TENSOR_DIM) {
            return true;
        }

        const auto axesShape = Shape(parseIntArrayAttr<int64_t>(op.getAxisValueAttr()));
        const auto [mergedShape, _] = mergeAxes(inputType.getShape(), axesShape);
        return mergedShape.size() > TARGET_TENSOR_DIM;
    };

    auto isLegalNormalizeL2Op = [&](IE::NormalizeL2Op op) {
        const auto inShape = mlir::cast<vpux::NDTypeInterface>(op.getOperand(0).getType()).getShape();
        const auto outShape = mlir::cast<vpux::NDTypeInterface>(op.getResult().getType()).getShape();
        if (inShape.size() == TARGET_TENSOR_DIM && outShape.size() == TARGET_TENSOR_DIM) {
            return true;
        }

        const auto axes = parseIntArrayAttr<int64_t>(op.getAxesValueAttr());
        const auto mergedInputShape = getMergedShapeAndAxes(to_small_vector(inShape), axes).first;
        return mergedInputShape.size() > TARGET_TENSOR_DIM;
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
    target.addDynamicallyLegalOp<IE::GeluOp>(isLegalDynamicOp);
    target.addDynamicallyLegalOp<IE::DynamicQuantizeOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SignOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SoftPlusOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::FlashSDPAOp>(isLegalOp);

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
    target.addDynamicallyLegalOp<IE::SoftMaxOp>(isLegalSoftMaxOp);
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
    target.addDynamicallyLegalOp<IE::TileOp>(isLegalTileOp);
    target.addDynamicallyLegalOp<IE::LSTMGatesOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::LSTMCellOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::LSTMSequenceOp>(allOperandsAre4D);
    target.addDynamicallyLegalOp<IE::FakeConvertOp>(allOperandsAre4D);
    target.addDynamicallyLegalOp<IE::GatherOp>(isLegalGatherOp);
    target.addDynamicallyLegalOp<IE::GatherNDOp>(isLegalGatherNDOp);
    target.addDynamicallyLegalOp<IE::GatherElementsOp>(isLegalGatherElementsOp);
    target.addDynamicallyLegalOp<IE::ErfOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::DynamicDequantizeOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReverseSequenceOp>(allOperandsAre4D);
    target.addDynamicallyLegalOp<IE::RMSOp>(allOperandsAre4D);
    target.addDynamicallyLegalOp<IE::SDPAOp>(isLegalSDPAOp);
    target.addDynamicallyLegalOp<IE::SDPAExtendedOp>(isLegalSDPAExtendedOp);
    target.addDynamicallyLegalOp<IE::RandomUniformOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::GRUGatesOp>(is4DLegalOp);
    target.addDynamicallyLegalOp<IE::SplitOp>(isLegalSplitOp);
    target.addDynamicallyLegalOp<IE::RollOp>(isLegalRollOp);
    target.addDynamicallyLegalOp<IE::ReverseOp>(isLegalReverseOp);
    target.addDynamicallyLegalOp<IE::QuantizeOp>(isLegalQuantOp);
    target.addDynamicallyLegalOp<IE::DequantizeOp>(isLegalQuantOp);
    target.addDynamicallyLegalOp<IE::QuantizeCastOp>(isLegalQuantOp);
    target.addDynamicallyLegalOp<IE::NormalizeL2Op>(isLegalNormalizeL2Op);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InterleavedDimsRewriter>(&ctx, _log);
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
    patterns.add<GenericConverter<IE::DynamicQuantizeOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::FakeConvertOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SignOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SoftPlusOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::QuantizeCastOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::QuantizeOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::DequantizeOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::FlashSDPAOp>>(typeConverter, &ctx, _log);

    patterns.add<GatherConverter>(typeConverter, &ctx, _log);
    patterns.add<GatherNDConverter>(typeConverter, &ctx, _log);
    patterns.add<GatherElementsConverter>(typeConverter, &ctx, _log);
    patterns.add<FakeQuantizeConverter>(typeConverter, &ctx, _log);
    patterns.add<TopKOpConverter>(typeConverter, &ctx, _log);
    patterns.add<Mvn6Converter>(typeConverter, &ctx, _log);
    patterns.add<SelectConverter>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceL1Op>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceL2Op>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceLogicalAndOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceLogicalOrOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceMaxOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceMeanOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceMinOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceProdOp>>(typeConverter, &ctx, _log);
    patterns.add<ReduceConverter<IE::ReduceSumOp>>(typeConverter, &ctx, _log);
    patterns.add<ReverseSequenceOpConverter>(typeConverter, &ctx, _log);
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
    patterns.add<SDPAOpConverter>(typeConverter, &ctx, _log);
    patterns.add<RandomUniformConverter>(typeConverter, &ctx, _log);
    patterns.add<GRUGatesConverter>(typeConverter, &ctx, _log);
    patterns.add<SplitConverter>(typeConverter, &ctx, _log);
    patterns.add<RollConverter>(typeConverter, &ctx, _log);
    patterns.add<ReverseConverter>(typeConverter, &ctx, _log);
    patterns.add<NormalizeL2Converter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertShapeTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShapeTo4DPass(Logger log) {
    return std::make_unique<ConvertShapeTo4DPass>(false, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShapeTo4DPass(bool forceConvertGatherTo4D, Logger log) {
    return std::make_unique<ConvertShapeTo4DPass>(forceConvertGatherTo4D, log);
}
