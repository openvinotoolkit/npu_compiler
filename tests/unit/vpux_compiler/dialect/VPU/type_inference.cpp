//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/public_options.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

enum class ElemType : uint64_t {
    F16 = 0,
    F32 = 1,
    BOOL8 = 2,
    SI32 = 3,
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ElemType elemType) {
    switch (elemType) {
    case ElemType::F16: {
        return os << "F16";
    }
    case ElemType::F32: {
        return os << "F32";
    }
    case ElemType::BOOL8: {
        return os << "BOOL8";
    }
    case ElemType::SI32: {
        return os << "SI32";
    }
    default: {
        return os;
    }
    }
}

struct TypeInfo {
    std::vector<int64_t> shape;
    ElemType elemType;
    DimsOrder order;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeInfo& typeInfo) {
    os << "{";
    for (auto dim : typeInfo.shape) {
        if (dim == mlir::ShapedType::kDynamic) {
            os << "?" << "x";
        } else {
            os << dim << "x";
        }
    }
    os << typeInfo.elemType << ",";
    typeInfo.order.printFormat(os);
    os << "}";
    return os;
}

struct TypeInferParam {
    std::vector<TypeInfo> inputs;
    std::optional<ElemType> outElemType;
    TypeInfo expectedResult;
};

[[maybe_unused]] std::ostream& operator<<(std::ostream& stream, const TypeInferParam& param) {
    const std::string sep = "_";
    llvm::raw_os_ostream result(stream);
    result << "inputs=";
    for (const auto& input : param.inputs) {
        result << input;
    }
    result << sep << "outElemType=" << param.outElemType;
    result << sep << "expectedResult=" << param.expectedResult;
    return stream;
};

class MLIR_VPUTypeInferenceTest : public MLIR_UnitBase, public ::testing::WithParamInterface<TypeInferParam> {
public:
    MLIR_VPUTypeInferenceTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect, mlir::func::FuncDialect, VPU::VPUDialect, mlir::tensor::TensorDialect>();
        builder = std::make_unique<mlir::OpBuilder>(&ctx);
    }

    MLIR_VPUTypeInferenceTest(const MLIR_VPUTypeInferenceTest&) = delete;
    MLIR_VPUTypeInferenceTest& operator=(const MLIR_VPUTypeInferenceTest&) = delete;

    mlir::OwningOpRef<mlir::tensor::EmptyOp> createOperand(const TypeInfo& typeInfo) {
        SmallVector<mlir::Value> dynValues;
        for (auto dim : typeInfo.shape | indexed) {
            if (dim.value() == mlir::ShapedType::kDynamic) {
                auto dyn = builder->create<mlir::arith::ConstantOp>(builder->getUnknownLoc(), builder->getIndexType(),
                                                                    builder->getIndexAttr(dim.index()));
                dynValues.push_back(dyn);
            }
        }
        return builder->create<mlir::tensor::EmptyOp>(builder->getUnknownLoc(), typeInfo.shape,
                                                      getElemType(typeInfo.elemType), dynValues,
                                                      getTensorAttr(&ctx, typeInfo.order, nullptr));
    }

    mlir::Type getElemType(ElemType elemType) {
        switch (elemType) {
        case ElemType::F16: {
            return mlir::Float16Type::get(&ctx);
        }
        case ElemType::F32: {
            return mlir::Float32Type::get(&ctx);
        }
        case ElemType::BOOL8: {
            return getBool8Type(&ctx);
        }
        case ElemType::SI32: {
            return getSInt32Type(&ctx);
        }
        default: {
            return nullptr;
        }
        }
    }

public:
    mlir::MLIRContext ctx;
    std::unique_ptr<mlir::OpBuilder> builder;
};

class MLIR_VPUTypeInferenceUtilsTest : public MLIR_VPUTypeInferenceTest {};
class MLIR_VPUTypeInferenceEltwiseOpTest : public MLIR_VPUTypeInferenceTest {};
class MLIR_VPUTypeInferenceGatherOpTest : public MLIR_VPUTypeInferenceTest {};
class MLIR_VPUTypeInferenceEyeOpTest : public MLIR_VPUTypeInferenceTest {};

}  // namespace

TEST_P(MLIR_VPUTypeInferenceUtilsTest, inferEltwiseReturnTypes) {
    const auto& params = GetParam();
    ASSERT_EQ(params.inputs.size(), 2) << "Invalid test parameters";

    SmallVector<mlir::Type> inferredReturnTypes;
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input1 = createOperand(params.inputs[0]);
    auto input2 = createOperand(params.inputs[1]);
    auto broadcast = IE::AutoBroadcastType::NUMPY;
    auto outElemType =
            params.outElemType.has_value() ? std::optional(getElemType(params.outElemType.value())) : std::nullopt;
    ASSERT_TRUE(mlir::succeeded(VPU::inferEltwiseReturnTypes(inferredReturnTypes, loc, input1->getResult(),
                                                             input2->getResult(), broadcast, outElemType)));
    ASSERT_EQ(inferredReturnTypes.size(), 1);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
    ASSERT_TRUE(type != nullptr);
    EXPECT_EQ(type.getShape().vec(), params.expectedResult.shape);
    EXPECT_EQ(type.getElementType(), getElemType(params.expectedResult.elemType));
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(type.getEncoding());
    if (params.expectedResult.shape.empty() ||
        params.expectedResult.order == DimsOrder::fromNumDims(params.expectedResult.shape.size())) {
        ASSERT_TRUE(tensorAttr == nullptr);
    } else {
        ASSERT_TRUE(tensorAttr != nullptr);
        auto expectedOrder = mlir::AffineMapAttr::get(params.expectedResult.order.toAffineMap(&ctx));
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

INSTANTIATE_TEST_SUITE_P(
        Broadcast, MLIR_VPUTypeInferenceUtilsTest,
        ::testing::Values(TypeInferParam{/*inputs=*/{TypeInfo{{}, ElemType::F32, DimsOrder::C},
                                                     TypeInfo{{}, ElemType::F32, DimsOrder::C}},
                                         /*outElemType=*/std::nullopt,
                                         /*expectedResult=*/TypeInfo{{}, ElemType::F32, DimsOrder::C}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{10}, ElemType::F32, DimsOrder::C},
                                                     TypeInfo{{10}, ElemType::F32, DimsOrder::C}},
                                         /*outElemType=*/std::nullopt,
                                         /*expectedResult=*/TypeInfo{{10}, ElemType::F32, DimsOrder::C}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW},
                                                     TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                                         /*outElemType=*/std::nullopt,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW},
                                                     TypeInfo{{3, 4}, ElemType::F16, DimsOrder::NC}},
                                         /*outElemType=*/std::nullopt,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{4}, ElemType::F16, DimsOrder::C},
                                                     TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                                         /*outElemType=*/std::nullopt,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{4}, ElemType::F16, DimsOrder::C},
                                                     TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NHWC}},
                                         /*outElemType=*/std::nullopt,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NHWC}}));

INSTANTIATE_TEST_SUITE_P(
        OutElemType, MLIR_VPUTypeInferenceUtilsTest,
        ::testing::Values(TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW},
                                                     TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                                         /*outElemType=*/ElemType::F16,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW},
                                                     TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                                         /*outElemType=*/ElemType::F32,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F32, DimsOrder::NCHW}},
                          TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F32, DimsOrder::NCHW},
                                                     TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                                         /*outElemType=*/ElemType::BOOL8,
                                         /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::BOOL8, DimsOrder::NCHW}}));

INSTANTIATE_TEST_SUITE_P(
        Dynamic, MLIR_VPUTypeInferenceUtilsTest,
        ::testing::Values(
                TypeInferParam{
                        /*inputs=*/{TypeInfo{{1, 2, 3, mlir::ShapedType::kDynamic}, ElemType::F16, DimsOrder::NCHW},
                                    TypeInfo{{1, 2, 3, mlir::ShapedType::kDynamic}, ElemType::F16, DimsOrder::NCHW}},
                        /*outElemType=*/std::nullopt,
                        /*expectedResult=*/
                        TypeInfo{{1, 2, 3, mlir::ShapedType::kDynamic}, ElemType::F16, DimsOrder::NCHW}},
                TypeInferParam{/*inputs=*/{TypeInfo{{1, mlir::ShapedType::kDynamic, 3, mlir::ShapedType::kDynamic},
                                                    ElemType::F16,
                                                    DimsOrder::NHWC},
                                           TypeInfo{{1, mlir::ShapedType::kDynamic, 3, mlir::ShapedType::kDynamic},
                                                    ElemType::F16,
                                                    DimsOrder::NHWC}},
                               /*outElemType=*/std::nullopt,
                               /*expectedResult=*/
                               TypeInfo{{1, mlir::ShapedType::kDynamic, 3, mlir::ShapedType::kDynamic},
                                        ElemType::F16,
                                        DimsOrder::NHWC}}));

TEST_P(MLIR_VPUTypeInferenceEltwiseOpTest, AddOp) {
    const auto& params = GetParam();
    ASSERT_EQ(params.inputs.size(), 2) << "Invalid test parameters";

    SmallVector<mlir::Type> inferredReturnTypes;
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input1 = createOperand(params.inputs[0]);
    auto input2 = createOperand(params.inputs[1]);
    VPU::AddOp::Properties properties{};
    properties.auto_broadcast = IE::AutoBroadcastTypeAttr::get(&ctx, IE::AutoBroadcastType::NUMPY);
    ASSERT_TRUE(mlir::succeeded(VPU::AddOp::inferReturnTypes(&ctx, loc, {input1->getResult(), input2->getResult()},
                                                             /*attrs=*/nullptr, &properties,
                                                             /*regions=*/{}, inferredReturnTypes)));
    ASSERT_EQ(inferredReturnTypes.size(), 1);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
    ASSERT_TRUE(type != nullptr);
    EXPECT_EQ(type.getShape().vec(), params.expectedResult.shape);
    EXPECT_EQ(type.getElementType(), getElemType(params.expectedResult.elemType));
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(type.getEncoding());
    if (params.expectedResult.order == DimsOrder::fromNumDims(params.expectedResult.shape.size())) {
        ASSERT_TRUE(tensorAttr == nullptr);
    } else {
        ASSERT_TRUE(tensorAttr != nullptr);
        auto expectedOrder = mlir::AffineMapAttr::get(params.expectedResult.order.toAffineMap(&ctx));
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

INSTANTIATE_TEST_SUITE_P(
        Add, MLIR_VPUTypeInferenceEltwiseOpTest,
        ::testing::Values(
                TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW},
                                           TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                               /*outElemType=*/std::nullopt,
                               /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NCHW}},
                TypeInferParam{/*inputs=*/{TypeInfo{{4}, ElemType::F16, DimsOrder::C},
                                           TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NHWC}},
                               /*outElemType=*/std::nullopt,
                               /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F16, DimsOrder::NHWC}},
                TypeInferParam{/*inputs=*/{TypeInfo{{1, 2, 3, 4}, ElemType::F32, DimsOrder::NCHW},
                                           TypeInfo{{1, 2, 3, 4}, ElemType::F32, DimsOrder::NCHW}},
                               /*outElemType=*/std::nullopt,
                               /*expectedResult=*/TypeInfo{{1, 2, 3, 4}, ElemType::F32, DimsOrder::NCHW}},
                TypeInferParam{
                        /*inputs=*/{TypeInfo{{1, 2, 3, mlir::ShapedType::kDynamic}, ElemType::F16, DimsOrder::NCHW},
                                    TypeInfo{{1, 2, 3, mlir::ShapedType::kDynamic}, ElemType::F16, DimsOrder::NCHW}},
                        /*outElemType=*/std::nullopt,
                        /*expectedResult=*/
                        TypeInfo{{1, 2, 3, mlir::ShapedType::kDynamic}, ElemType::F16, DimsOrder::NCHW}}));

TEST_P(MLIR_VPUTypeInferenceGatherOpTest, GatherOp) {
    const auto& params = GetParam();
    ASSERT_EQ(params.inputs.size(), 2) << "Invalid test parameters";

    SmallVector<mlir::Type> inferredReturnTypes;
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input1 = createOperand(params.inputs[0]);
    auto input2 = createOperand(params.inputs[1]);
    VPU::GatherOp::Properties properties{};
    properties.axis_value = getIntAttr(&ctx, 0);
    properties.batch_dims = getIntAttr(&ctx, 0);
    properties.indices_rank = getIntAttr(&ctx, params.inputs[1].shape.size());
    ASSERT_TRUE(mlir::succeeded(VPU::GatherOp::inferReturnTypes(&ctx, loc, {input1->getResult(), input2->getResult()},
                                                                /*attrs=*/nullptr, &properties,
                                                                /*regions=*/{}, inferredReturnTypes)));
    ASSERT_EQ(inferredReturnTypes.size(), 1);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
    ASSERT_TRUE(type != nullptr);
    EXPECT_EQ(type.getShape().vec(), params.expectedResult.shape);
    EXPECT_EQ(type.getElementType(), getElemType(params.expectedResult.elemType));
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(type.getEncoding());
    if (params.expectedResult.order == DimsOrder::fromNumDims(params.expectedResult.shape.size())) {
        ASSERT_TRUE(tensorAttr == nullptr);
    } else {
        ASSERT_TRUE(tensorAttr != nullptr);
        auto expectedOrder = mlir::AffineMapAttr::get(params.expectedResult.order.toAffineMap(&ctx));
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

INSTANTIATE_TEST_SUITE_P(Gather, MLIR_VPUTypeInferenceGatherOpTest,
                         ::testing::Values(TypeInferParam{
                                 /*inputs=*/{TypeInfo{{51865, 512}, ElemType::F16, DimsOrder::NC},
                                             TypeInfo{{1, 16}, ElemType::SI32, DimsOrder::NC}},
                                 /*outElemType=*/std::nullopt,
                                 /*expectedResult=*/TypeInfo{{1, 16, 512}, ElemType::F16, DimsOrder::CHW}}));

TEST_P(MLIR_VPUTypeInferenceEyeOpTest, EyeOp) {
    const auto& params = GetParam();
    ASSERT_EQ(params.inputs.size(), 1) << "Invalid test parameters";

    SmallVector<mlir::Type> inferredReturnTypes;
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input = createOperand(params.inputs[0]);
    VPU::EyeOp::Properties properties{};
    properties.batch_shape_value = getIntArrayAttr(&ctx, SmallVector<int64_t>{0});
    properties.num_columns_value = getIntAttr(&ctx, 128);
    properties.num_rows_value = getIntAttr(&ctx, 128);
    properties.outputType = mlir::TypeAttr::get(getElemType(params.outElemType.value()));
    ASSERT_TRUE(mlir::succeeded(VPU::EyeOp::inferReturnTypes(&ctx, loc, {input->getResult()},
                                                             /*attrs=*/nullptr, &properties,
                                                             /*regions=*/{}, inferredReturnTypes)));
    ASSERT_EQ(inferredReturnTypes.size(), 1);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
    ASSERT_TRUE(type != nullptr);
    EXPECT_EQ(type.getShape().vec(), params.expectedResult.shape);
    EXPECT_EQ(type.getElementType(), getElemType(params.expectedResult.elemType));
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(type.getEncoding());
    if (params.expectedResult.order == DimsOrder::fromNumDims(params.expectedResult.shape.size())) {
        ASSERT_TRUE(tensorAttr == nullptr);
    } else {
        ASSERT_TRUE(tensorAttr != nullptr);
        auto expectedOrder = mlir::AffineMapAttr::get(params.expectedResult.order.toAffineMap(&ctx));
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

INSTANTIATE_TEST_SUITE_P(Eye, MLIR_VPUTypeInferenceEyeOpTest,
                         ::testing::Values(TypeInferParam{
                                 /*inputs=*/{TypeInfo{{1}, ElemType::SI32, DimsOrder::C}},
                                 /*outElemType=*/ElemType::F16,
                                 /*expectedResult=*/TypeInfo{{128, 128}, ElemType::F16, DimsOrder::NC}}));
