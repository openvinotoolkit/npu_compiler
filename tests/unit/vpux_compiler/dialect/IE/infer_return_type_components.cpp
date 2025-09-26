//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

enum class OpType { MatMul, Convolution, ActivationFunction, Add, MaxPool, LSTMSequence };

//
// Base parameter class
//

struct BaseParams {
    BaseParams(OpType opType, SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo)
            : m_opType(opType),
              m_inputsShapeInfo(std::move(inputsShapeInfo)),
              m_expectedShapesInfo(std::move(expectedShapesInfo)) {
    }
    virtual ~BaseParams() = default;
    OpType m_opType;
    SmallVector<ShapeInfo> m_inputsShapeInfo;
    SmallVector<ShapeInfo> m_expectedShapesInfo;
};

//
// Specialized parameter classes for IE dialect operations
//

struct MatMulParams : public BaseParams {
    MatMulParams(SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo, bool transpose_a,
                 bool transpose_b)
            : BaseParams(OpType::MatMul, std::move(inputsShapeInfo), std::move(expectedShapesInfo)),
              m_transpose_a(transpose_a),
              m_transpose_b(transpose_b) {
    }
    bool m_transpose_a;
    bool m_transpose_b;
};

struct ConvParams : public BaseParams {
    ConvParams(SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo,
               SmallVector<int64_t> strides, SmallVector<int64_t> padsBegin, SmallVector<int64_t> padsEnd,
               SmallVector<int64_t> dilations)
            : BaseParams(OpType::Convolution, std::move(inputsShapeInfo), std::move(expectedShapesInfo)),
              m_strides(std::move(strides)),
              m_padsBegin(std::move(padsBegin)),
              m_padsEnd(std::move(padsEnd)),
              m_dilations(std::move(dilations)) {
    }
    SmallVector<int64_t> m_strides;
    SmallVector<int64_t> m_padsBegin;
    SmallVector<int64_t> m_padsEnd;
    SmallVector<int64_t> m_dilations;
};

struct ActivationFunctionParams : public BaseParams {
    enum ActivationFunction { ReLU, SoftMax };
    ActivationFunctionParams(SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo,
                             ActivationFunction activationFunc)
            : BaseParams(OpType::ActivationFunction, std::move(inputsShapeInfo), std::move(expectedShapesInfo)),
              m_activationFunc(activationFunc) {
    }
    ActivationFunction m_activationFunc;
};

struct AddParams : public BaseParams {
    AddParams(SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo,
              IE::AutoBroadcastType broadcastType)
            : BaseParams(OpType::Add, std::move(inputsShapeInfo), std::move(expectedShapesInfo)),
              m_broadcastType(broadcastType) {
    }
    IE::AutoBroadcastType m_broadcastType;
};

struct MaxPoolParams : public BaseParams {
    MaxPoolParams(SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo,
                  SmallVector<int64_t> strides, SmallVector<int64_t> kernelSize, SmallVector<int64_t> padsBegin,
                  SmallVector<int64_t> padsEnd, IE::RoundingType roundingType)
            : BaseParams(OpType::MaxPool, std::move(inputsShapeInfo), std::move(expectedShapesInfo)),
              m_strides(strides),
              m_kernelSize(kernelSize),
              m_padsBegin(padsBegin),
              m_padsEnd(padsEnd),
              m_roundingType(roundingType) {
    }
    SmallVector<int64_t> m_strides;
    SmallVector<int64_t> m_kernelSize;
    SmallVector<int64_t> m_padsBegin;
    SmallVector<int64_t> m_padsEnd;
    IE::RoundingType m_roundingType;
};

struct LSTMSequenceParams : public BaseParams {
    LSTMSequenceParams(SmallVector<ShapeInfo> inputsShapeInfo, SmallVector<ShapeInfo> expectedShapesInfo)
            : BaseParams(OpType::LSTMSequence, std::move(inputsShapeInfo), std::move(expectedShapesInfo)) {
    }
};

//
// Base builder class
//

class OpBuilder {
public:
    virtual ~OpBuilder() {
    }
    virtual std::optional<mlir::tensor::EmptyOp> prepareInput(mlir::OpBuilder& builder,
                                                              std::shared_ptr<BaseParams> baseParams,
                                                              int64_t index = 0) {
        const auto ctx = builder.getContext();
        const auto loc = mlir::UnknownLoc::get(ctx);
        const SmallVector<int64_t>& shape = baseParams->m_inputsShapeInfo[index].shape;
        const SmallVector<int64_t>& inBounds = baseParams->m_inputsShapeInfo[index].bounds;

        const auto elemType = mlir::Float16Type::get(ctx);
        const auto affineMap = mlir::AffineMapAttr::get(DimsOrder::fromNumDims(shape.size()).toAffineMap(ctx));

        const auto tensorAttr = vpux::TensorAttr::get(ctx, affineMap, /*memSpace=*/nullptr, BoundsRef(inBounds), {});
        const auto tensorType = mlir::RankedTensorType::get(shape, elemType, tensorAttr);
        return builder.create<mlir::tensor::EmptyOp>(loc, tensorType, mlir::ValueRange{});
    }
    virtual std::optional<mlir::tensor::EmptyOp> prepareWeights(mlir::OpBuilder& builder,
                                                                std::shared_ptr<BaseParams> baseParams) {
        const auto ctx = builder.getContext();
        const auto loc = mlir::UnknownLoc::get(ctx);
        const SmallVector<int64_t>& shapeWeights = baseParams->m_inputsShapeInfo[1].shape;
        const auto elemType = mlir::Float16Type::get(ctx);
        const auto tensorTypeWeights = mlir::RankedTensorType::get(shapeWeights, elemType, nullptr);
        return builder.create<mlir::tensor::EmptyOp>(loc, tensorTypeWeights, mlir::ValueRange{});
    }
    virtual mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) = 0;
    virtual std::string getOpName(std::shared_ptr<BaseParams> baseParams) = 0;
};

//
// Specialized builder classes for IE dialect operations
//

class MatmulBuilder : public OpBuilder {
public:
    mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::MatMul);
        if (auto matmulParams = std::dynamic_pointer_cast<MatMulParams>(baseParams)) {
            const auto loc = builder.getUnknownLoc();
            auto input = prepareInput(builder, baseParams);
            auto weights = prepareWeights(builder, baseParams);
            return builder.create<IE::MatMulOp>(loc, input.value(), weights.value(), matmulParams->m_transpose_a,
                                                matmulParams->m_transpose_b, /*post_op=*/nullptr);
        }
        return nullptr;
    }
    std::string getOpName(std::shared_ptr<BaseParams> /*baseParams*/) override {
        return IE::MatMulOp::getOperationName().str();
    }
};

class ConvolutionBuilder : public OpBuilder {
public:
    mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::Convolution);
        if (auto convParams = std::dynamic_pointer_cast<ConvParams>(baseParams)) {
            const auto ctx = builder.getContext();
            const auto loc = builder.getUnknownLoc();

            auto input = prepareInput(builder, baseParams);
            auto weights = prepareWeights(builder, baseParams);

            const auto attrStride = getIntArrayAttr(ctx, convParams->m_strides);
            const auto attrPadsBegin = getIntArrayAttr(ctx, convParams->m_padsBegin);
            const auto attrPadsEnd = getIntArrayAttr(ctx, convParams->m_padsEnd);
            const auto attrDilation = getIntArrayAttr(ctx, convParams->m_dilations);

            return builder.create<IE::ConvolutionOp>(loc, input.value(), weights.value(), nullptr, attrStride,
                                                     attrPadsBegin, attrPadsEnd, attrDilation, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr);
        }
        return nullptr;
    }
    std::string getOpName(std::shared_ptr<BaseParams> /*baseParams*/) override {
        return IE::ConvolutionOp::getOperationName().str();
    }
};

class ActivationFunctionBuilder : public OpBuilder {
public:
    std::optional<mlir::tensor::EmptyOp> prepareWeights(mlir::OpBuilder& /*builder*/,
                                                        std::shared_ptr<BaseParams> /*baseParams*/) override {
        return std::nullopt;
    }
    mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::ActivationFunction);
        if (auto actFuncParams = std::dynamic_pointer_cast<ActivationFunctionParams>(baseParams)) {
            const auto ctx = builder.getContext();
            const auto loc = builder.getUnknownLoc();
            auto input = prepareInput(builder, baseParams);

            switch (actFuncParams->m_activationFunc) {
            case ActivationFunctionParams::ReLU:
                return builder.create<IE::ReLUOp>(loc, input.value());
            case ActivationFunctionParams::SoftMax:
                return builder.create<IE::SoftMaxOp>(loc, input.value(), /*axisInd=*/getIntAttr(ctx, 1),
                                                     /*padSize=*/nullptr);
            default:
                return nullptr;
            }
        }
        return nullptr;
    }
    std::string getOpName(std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::ActivationFunction);
        if (auto actFuncParams = std::dynamic_pointer_cast<ActivationFunctionParams>(baseParams)) {
            switch (actFuncParams->m_activationFunc) {
            case ActivationFunctionParams::ReLU:
                return IE::ReLUOp::getOperationName().str();
            case ActivationFunctionParams::SoftMax:
                return IE::SoftMaxOp::getOperationName().str();
            default:
                break;
            }
        }
        return "Unknown Activation function";
    }
};

class AddBuilder : public OpBuilder {
public:
    mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::Add);
        if (auto addParams = std::dynamic_pointer_cast<AddParams>(baseParams)) {
            const auto ctx = builder.getContext();
            const auto loc = builder.getUnknownLoc();
            auto lhsOp = prepareInput(builder, baseParams);
            auto rhsOp = prepareWeights(builder, baseParams);

            return builder.create<IE::AddOp>(loc, lhsOp.value(), rhsOp.value(),
                                             IE::AutoBroadcastTypeAttr::get(ctx, addParams->m_broadcastType),
                                             /*post_op=*/nullptr,
                                             /*clamp=*/nullptr,
                                             /*outputPadding=*/nullptr,
                                             /*inputPadding=*/nullptr);
        }
        return nullptr;
    }
    std::string getOpName(std::shared_ptr<BaseParams> /*baseParams*/) override {
        return IE::AddOp::getOperationName().str();
    }
};

class MaxPoolBuilder : public OpBuilder {
public:
    mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::MaxPool);
        if (auto maxPoolParams = std::dynamic_pointer_cast<MaxPoolParams>(baseParams)) {
            const auto ctx = builder.getContext();
            const auto loc = builder.getUnknownLoc();
            auto input = prepareInput(builder, baseParams);

            return builder.create<IE::MaxPoolOp>(loc, input.value(), getIntArrayAttr(ctx, maxPoolParams->m_kernelSize),
                                                 getIntArrayAttr(ctx, maxPoolParams->m_strides),
                                                 getIntArrayAttr(ctx, maxPoolParams->m_padsBegin),
                                                 getIntArrayAttr(ctx, maxPoolParams->m_padsEnd),
                                                 vpux::IE::RoundingTypeAttr::get(ctx, maxPoolParams->m_roundingType),
                                                 nullptr, nullptr, nullptr, nullptr);
        }
        return nullptr;
    }
    std::string getOpName(std::shared_ptr<BaseParams> /*baseParams*/) override {
        return IE::MaxPoolOp::getOperationName().str();
    }
};

class LSTMSequenceBuilder : public OpBuilder {
public:
    mlir::Operation* buildFunction(mlir::OpBuilder& builder, std::shared_ptr<BaseParams> baseParams) override {
        assert(baseParams->m_opType == OpType::LSTMSequence);
        const auto ctx = builder.getContext();
        const auto loc = mlir::UnknownLoc::get(ctx);

        // Prepare input tensors
        auto inputData = prepareInput(builder, baseParams, 0);
        auto initialHiddenState = prepareInput(builder, baseParams, 1);
        auto initialCellState = prepareInput(builder, baseParams, 2);
        auto weights = prepareInput(builder, baseParams, 3);
        auto recurrenceWeights = prepareInput(builder, baseParams, 4);
        auto bias = prepareInput(builder, baseParams, 5);

        // Create LSTMSequence operation
        return builder.create<IE::LSTMSequenceOp>(
                loc, mlir::Value(inputData->getResult()), mlir::Value(initialHiddenState->getResult()),
                mlir::Value(initialCellState->getResult()), mlir::Value(weights->getResult()),
                mlir::Value(recurrenceWeights->getResult()), mlir::Value(bias->getResult()),
                nullptr /*sequenceLengthAttr*/, IE::RNNSequenceDirection::FORWARD);
    }
    std::string getOpName(std::shared_ptr<BaseParams> /*baseParams*/) override {
        return IE::LSTMSequenceOp::getOperationName().str();
    }
};

//
// define opBuilderMap
//

const std::map<OpType, std::shared_ptr<OpBuilder>> opBuilderMap{
        {OpType::MatMul, std::make_shared<MatmulBuilder>()},
        {OpType::Convolution, std::make_shared<ConvolutionBuilder>()},
        {OpType::ActivationFunction, std::make_shared<ActivationFunctionBuilder>()},
        {OpType::Add, std::make_shared<AddBuilder>()},
        {OpType::MaxPool, std::make_shared<MaxPoolBuilder>()},
        {OpType::LSTMSequence, std::make_shared<LSTMSequenceBuilder>()},
};

std::string OpTypeToString(const testing::TestParamInfo<std::shared_ptr<BaseParams>>& info) {
    auto opBuilderIter = opBuilderMap.find(info.param->m_opType);
    // ASSERT_NE(opBuilderIter, opBuilderMap.end());
    std::string name = opBuilderIter->second->getOpName(info.param);
    return name + "_" + std::to_string(info.index);
}

}  // namespace

//
// InferReturnTypeComponents.PropagatesBounds test Definition
//

using InferReturnTypeComponents = testing::TestWithParam<std::shared_ptr<BaseParams>>;

TEST_P(InferReturnTypeComponents, PropagatesBounds) {
    // find builder function
    auto opBuilderIter = opBuilderMap.find(GetParam()->m_opType);
    ASSERT_NE(opBuilderIter, opBuilderMap.end());

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx{registry};
    ctx.loadDialect<IE::IEDialect>();

    mlir::OpBuilder builder(&ctx);

    // call build function
    mlir::OwningOpRef<mlir::Operation*> testedOp = opBuilderIter->second->buildFunction(builder, GetParam());

    // check the results
    ASSERT_NE(testedOp.get(), nullptr);
    for (auto result : testedOp->getResults() | indexed) {
        if (result.index() >= GetParam()->m_expectedShapesInfo.size()) {
            break;
        }
        // check shape
        const auto outputNDType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(result.value().getType());
        ASSERT_NE(outputNDType, nullptr);
        const auto outputShape = outputNDType.getShape().toValues();

        EXPECT_EQ(outputShape, Shape{GetParam()->m_expectedShapesInfo[result.index()].shape});

        // check bounds
        const auto tensorType = mlir::cast<mlir::RankedTensorType>(outputNDType);
        ASSERT_NE(tensorType, nullptr);

        const auto outputBoundedType = mlir::dyn_cast<Core::BoundedTensorType>(tensorType);
        EXPECT_TRUE(outputBoundedType != nullptr);

        const auto outputBounds = to_small_vector(outputBoundedType.getBounds());
        EXPECT_EQ(outputBounds, GetParam()->m_expectedShapesInfo[result.index()].bounds);
    }
}

//
// InferReturnTypeComponents.PropagatesBounds test cases Instantiation
//

std::vector<std::shared_ptr<BaseParams>> InferReturnTypeComponentsData = {
        // MatMulParams
        std::make_shared<MatMulParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                                               SmallVector<int64_t>{mlir::ShapedType::kDynamic, 128},  // input1Shape
                                               SmallVector<int64_t>{8000, 128}                         // input1Bounds
                                       },
                                       vpux::ShapeInfo{
                                               SmallVector<int64_t>{128, 128},  // input2Shape
                                               SmallVector<int64_t>{}           // input2Bounds
                                       }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{mlir::ShapedType::kDynamic, 128},  // expectedOutShape
                        SmallVector<int64_t>{8000, 128}                         // expectedOutBounds
                }},
                // ExtendedParams
                false,  // transpose_a
                false   // transpose_a
                ),
        std::make_shared<MatMulParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                                               SmallVector<int64_t>{128, mlir::ShapedType::kDynamic},  // input1Shape
                                               SmallVector<int64_t>{128, 8000}                         // input1Bounds
                                       },
                                       ShapeInfo{
                                               SmallVector<int64_t>{128, 128},  // input2Shape
                                               SmallVector<int64_t>{}           // input2Bounds
                                       }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{mlir::ShapedType::kDynamic, 128},  // expectedOutShape
                        SmallVector<int64_t>{8000, 128}                         // expectedOutBounds
                }},
                // ExtendedParams
                true,  // transpose_a
                false  // transpose_a
                ),
        // ConvParams
        std::make_shared<ConvParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{
                        ShapeInfo{
                                SmallVector<int64_t>{1, 3, 64, mlir::ShapedType::kDynamic},  // inputShape
                                SmallVector<int64_t>{1, 3, 64, 64}                           // inputBounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{3, 3, 2, 2},  // weightsShape
                                SmallVector<int64_t>{}             // weightsBounds
                        }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 3, 32, mlir::ShapedType::kDynamic},  // expectedOutShape
                        SmallVector<int64_t>{1, 3, 32, 32}                           // expectedOutBounds
                }},
                // ExtendedParams
                SmallVector<int64_t>{2, 2},  // strides
                SmallVector<int64_t>{0, 0},  // padsBegin
                SmallVector<int64_t>{0, 0},  // padsEnd
                SmallVector<int64_t>{1, 1}   // dilations
                ),
        std::make_shared<ConvParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                                               SmallVector<int64_t>{1, 3, mlir::ShapedType::kDynamic,
                                                                    mlir::ShapedType::kDynamic},  // inputShape
                                               SmallVector<int64_t>{1, 3, 128, 32}                // inputBounds
                                       },
                                       ShapeInfo{
                                               SmallVector<int64_t>{3, 3, 5, 5},  // weightsShape
                                               SmallVector<int64_t>{}             // weightsBounds
                                       }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 3, mlir::ShapedType::kDynamic,
                                             mlir::ShapedType::kDynamic},  // expectedOutShape
                        SmallVector<int64_t>{1, 3, 42, 10}                 // expectedOutBounds
                }},
                // ExtendedParams
                SmallVector<int64_t>{3, 3},  // strides
                SmallVector<int64_t>{2, 0},  // padsBegin
                SmallVector<int64_t>{0, 2},  // padsEnd
                SmallVector<int64_t>{1, 1}   // dilations
                ),
        // ActivationFunctionParams
        std::make_shared<ActivationFunctionParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // inputShape
                        SmallVector<int64_t>{1, 16, 32, 320}                          // inputBounds
                }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // expectedOutShape
                        SmallVector<int64_t>{1, 16, 32, 320}                          // expectedOutBounds
                }},
                // ExtendedParams
                ActivationFunctionParams::ReLU),  // activationFunc
        std::make_shared<ActivationFunctionParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // inputShape
                        SmallVector<int64_t>{1, 16, 32, 320}                          // inputBounds
                }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // expectedOutShape
                        SmallVector<int64_t>{1, 16, 32, 320}                          // expectedOutBounds
                }},
                // ExtendedParams
                ActivationFunctionParams::SoftMax),  // activationFunc
        // Add
        std::make_shared<AddParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{
                        ShapeInfo{
                                SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // input1Shape
                                SmallVector<int64_t>{1, 16, 32, 320}                          // input1Bounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{1, 16, 1, 1},  // input2Shape
                                SmallVector<int64_t>{}              // input2Bounds
                        }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // expectedOutShape
                        SmallVector<int64_t>{1, 16, 32, 320}                          // expectedOutBounds
                }},
                // ExtendedParams
                IE::AutoBroadcastType::NUMPY),  // broadcastType
        // MaxPool
        std::make_shared<MaxPoolParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 32, mlir::ShapedType::kDynamic},  // inputShape
                        SmallVector<int64_t>{1, 16, 32, 320}                          // inputBounds
                }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 16, 16, mlir::ShapedType::kDynamic},  // expectedOutShape
                        SmallVector<int64_t>{1, 16, 16, 160}                          // expectedOutBounds
                }},
                // ExtendedParams
                SmallVector<int64_t>{2, 2},  // strides
                SmallVector<int64_t>{2, 2},  // kernelSize
                SmallVector<int64_t>{0, 0},  // padsBegin
                SmallVector<int64_t>{0, 0},  // padsEnd
                IE::RoundingType::FLOOR      // m_roundingType
                ),
        // LSTMSequence
        std::make_shared<LSTMSequenceParams>(
                // BaseParams
                // inputShapeInfo
                SmallVector<ShapeInfo>{
                        ShapeInfo{
                                SmallVector<int64_t>{1, mlir::ShapedType::kDynamic, 512},  // inputDataShape
                                SmallVector<int64_t>{1, 10, 512}                           // inputDataBounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{1, 1, 128},  // initialHiddenStateShape
                                SmallVector<int64_t>{}            // initialHiddenStateBounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{1, 1, 128},  // initialCellStateShape
                                SmallVector<int64_t>{}            // initialCellStateBounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{1, 4 * 128, 512},  // weightsShape
                                SmallVector<int64_t>{}                  // weightsBounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{1, 4 * 128, 128},  // recurrenceWeightsShape
                                SmallVector<int64_t>{}                  // recurrenceWeightsBounds
                        },
                        ShapeInfo{
                                SmallVector<int64_t>{1, 4 * 128},  // biasShape
                                SmallVector<int64_t>{}             // biasBounds
                        }},
                // expectedShapeInfo
                SmallVector<ShapeInfo>{ShapeInfo{
                        SmallVector<int64_t>{1, 1, mlir::ShapedType::kDynamic, 128},  // expectedOutputShape
                        SmallVector<int64_t>{1, 1, 10, 128}                           // expectedOutputBounds
                }}),
};

INSTANTIATE_TEST_SUITE_P(CheckBounds, InferReturnTypeComponents, testing::ValuesIn(InferReturnTypeComponentsData),
                         OpTypeToString);
