//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

struct InferStridedSliceData {
    std::vector<int64_t> inDataShape;
    std::vector<int64_t> begins;
    std::vector<int64_t> ends;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginsShape;
    std::vector<int64_t> endsShape;
    std::vector<int64_t> stridesShape;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisMask;
    std::vector<int64_t> outDataShape;
};

class InferStridedSliceTests : public testing::TestWithParam<InferStridedSliceData> {};

TEST_P(InferStridedSliceTests, InferOutputShapeStridedSlice) {
    const auto params = GetParam();
    const auto inShapeInfo = ShapeInfo{to_small_vector(params.inDataShape), SmallVector<int64_t>{}};
    const auto shapeInfo =
            inferStridedSliceOutputShape(inShapeInfo, params.begins, params.ends, params.strides, params.beginsShape,
                                         params.endsShape, params.stridesShape, params.beginMask, params.endMask,
                                         params.newAxisMask, params.shrinkAxisMask, params.ellipsisMask);
    EXPECT_EQ(to_std_vector(shapeInfo.shape), params.outDataShape);
}

// clang-format off

std::vector<InferStridedSliceData> inferStridedSliceData = {{{1, 32, 12, 64}, /* inDataShape */
                                                             {0, 0, 1, 0},    // begins
                                                             {1, 32, 12, 64}, // ends
                                                             {1, 1, 1, 1},    // strides
                                                             {},              // beginsShape
                                                             {},              // endsShape
                                                             {},              // stridesShape
                                                             {1, 1, 0, 1},    // beginMask
                                                             {1, 1, 1, 1},    // endMask
                                                             {0, 0, 0, 0},    // newAxisMask
                                                             {0, 0, 0, 0},    // shrinkAxisMask
                                                             {0, 0, 0, 0},    // ellipsisMask
                                                             {1, 32, 11, 64}},// outDataShape
                                                            {{1, 32, 64, 128},/* inDataShape */
                                                             {0, 0, 53, 0},   // begins
                                                             {1, 32, 64, 128},// ends
                                                             {1, 1, 1, 1},    // strides
                                                             {},              // beginsShape
                                                             {},              // endsShape
                                                             {},              // stridesShape
                                                             {1, 1, 0, 1},    // beginMask
                                                             {1, 1, 1, 1},    // endMask
                                                             {0, 0, 0, 0},    // newAxisMask
                                                             {0, 0, 0, 0},    // shrinkAxisMask
                                                             {0, 0, 0, 0},    // ellipsisMask
                                                             {1, 32, 11, 128}},// outDataShape
                                                            {{1, 32, 64, 256},/* inDataShape */
                                                             {0, 0, 54, 0},   // begins
                                                             {1, 32, 64, 128},// ends
                                                             {1, 1, 1, 1},    // strides
                                                             {},              // beginsShape
                                                             {},              // endsShape
                                                             {},              // stridesShape
                                                             {1, 1, 0, 1},    // beginMask
                                                             {1, 1, 1, 1},    // endMask
                                                             {0, 0, 0, 0},    // newAxisMask
                                                             {0, 0, 0, 0},    // shrinkAxisMask
                                                             {0, 0, 0, 0},    // ellipsisMask
                                                             {1, 32, 10, 256}},// outDataShape
                                                            {{1, 32, 64, 512},/* inDataShape */
                                                             {0, 0, 55, 0},   // begins
                                                             {1, 32, 64, 128},// ends
                                                             {1, 1, 1, 1},    // strides
                                                             {},              // beginsShape
                                                             {},              // endsShape
                                                             {},              // stridesShape
                                                             {1, 1, 0, 1},    // beginMask
                                                             {1, 1, 1, 1},    // endMask
                                                             {0, 0, 0, 0},    // newAxisMask
                                                             {0, 0, 0, 0},    // shrinkAxisMask
                                                             {0, 0, 0, 0},    // ellipsisMask
                                                             {1, 32, 9, 512}},// outDataShape
                                                            {{9},             /* inDataShape */
                                                             {0},             // begins
                                                             {},              // ends
                                                             {1},             // strides
                                                             {},              // beginsShape
                                                             {1},             // endsShape
                                                             {},              // stridesShape
                                                             {0},             // beginMask
                                                             {0},             // endMask
                                                             {0},             // newAxisMask
                                                             {0},             // shrinkAxisMask
                                                             {0},             // ellipsisMask
                                                             {mlir::ShapedType::kDynamic}},// outDataShape
                                                            {{9},             /* inDataShape */
                                                             {},              // begins
                                                             {0},             // ends
                                                             {1},             // strides
                                                             {1},             // beginsShape
                                                             {},              // endsShape
                                                             {},              // stridesShape
                                                             {0},             // beginMask
                                                             {0},             // endMask
                                                             {0},             // newAxisMask
                                                             {0},             // shrinkAxisMask
                                                             {0},             // ellipsisMask
                                                             {mlir::ShapedType::kDynamic}}};// outDataShape

// clang-format on

INSTANTIATE_TEST_SUITE_P(StridedSlice, InferStridedSliceTests, testing::ValuesIn(inferStridedSliceData));

struct InferPoolingData {
    std::vector<int64_t> inDataShape;
    std::vector<int64_t> padsBegin;
    std::vector<int64_t> padsEnd;
    std::vector<int64_t> windowShape;
    std::vector<int64_t> windowStrides;
    std::vector<int64_t> outDataShape;
};

class InferMaxPoolTests : public testing::TestWithParam<InferPoolingData> {};

TEST_P(InferMaxPoolTests, InferOutputShapePooling) {
    const auto params = GetParam();
    const auto inShapeInfo = ShapeInfo{to_small_vector(params.inDataShape), SmallVector<int64_t>{}};

    const auto outputShapeInfo = inferMaxPoolOutputShape(inShapeInfo, params.windowStrides, params.padsBegin,
                                                         params.padsEnd, params.windowShape);

    EXPECT_EQ(to_std_vector(outputShapeInfo.shape), params.outDataShape);
}

// clang-format off

std::vector<InferPoolingData> inferMaxPoolData = {
        // inData      padsBegin  padsEnd windowShape windowStrides outDataShape
        {{1, 2048, 23, 30}, {0, 0}, {0, 0}, {23, 30}, {23, 30}, {1, 2048, 1, 1}},
        {{1, 16, 1, 14}, {0, 0}, {0, 0}, {1, 14}, {1, 1}, {1, 16, 1, 1}},
        {{1, 16, 14, 1}, {0, 0}, {0, 0}, {14, 1}, {1, 1}, {1, 16, 1, 1}},
        {{1, 4, 54, 54}, {0, 0}, {2, 2}, {3, 3}, {2, 2}, {1, 4, 27, 27}},
        {{1, 8, 32, 32}, {1, 1}, {23, 30}, {3, 3}, {1, 1}, {1, 8, 54, 61}},
        {{1, 16, 24, 24}, {1, 1}, {1, 1}, {3, 3}, {1, 1}, {1, 16, 24, 24}},
        {{1, 24, 16, 16}, {1, 1}, {1, 1}, {5, 5}, {2, 2}, {1, 24, 7, 7}},
        {{1, 32, 8, 8}, {0, 0}, {0, 0}, {5, 5}, {2, 2}, {1, 32, 2, 2}},
        {{1, 16, 27, 27}, {1, 1}, {1, 1}, {4, 4}, {1, 1}, {1, 16, 26, 26}},
        {{1, 16, 64, 64}, {2, 2}, {2, 2}, {3, 3}, {1, 1}, {1, 16, 66, 66}}};

// clang-format on

INSTANTIATE_TEST_SUITE_P(MaxPool, InferMaxPoolTests, testing::ValuesIn(inferMaxPoolData));

class InferAvgPoolTests : public testing::TestWithParam<InferPoolingData> {};

TEST_P(InferAvgPoolTests, InferOutputShapePooling) {
    const auto params = GetParam();
    ShapeInfo shapeInfo;
    shapeInfo.shape = to_small_vector(params.inDataShape);
    const auto shapeI64 = inferAvgPoolOutputShape(shapeInfo, params.windowStrides, params.padsBegin, params.padsEnd,
                                                  params.windowShape);
    EXPECT_EQ(to_std_vector(shapeI64.shape), params.outDataShape);
}

// clang-format off

std::vector<InferPoolingData> inferAvgPoolData = {
        // inData     padsBegin  padsEnd windowShape windowStrides outDataShape
        {{1, 8, 32, 32}, {1, 1}, {23, 30}, {3, 3}, {1, 1}, {1, 8, 54, 61}},
        {{1, 16, 24, 24}, {1, 1}, {1, 1}, {3, 3}, {1, 1}, {1, 16, 24, 24}},
        {{1, 24, 16, 16}, {1, 1}, {1, 1}, {5, 5}, {2, 2}, {1, 24, 7, 7}},
        {{1, 32, 8, 8}, {0, 0}, {0, 0}, {5, 5}, {2, 2}, {1, 32, 2, 2}},
        {{1, 2048, 23, 30}, {0, 0}, {0, 0}, {23, 30}, {23, 30}, {1, 2048, 1, 1}},
        {{1, 16, 1, 14}, {0, 0}, {0, 0}, {1, 14}, {1, 1}, {1, 16, 1, 1}},
        {{1, 16, 14, 1}, {0, 0}, {0, 0}, {14, 1}, {1, 1}, {1, 16, 1, 1}},
        {{1, 4, 54, 54}, {0, 0}, {2, 2}, {3, 3}, {2, 2}, {1, 4, 27, 27}},
        {{1, 16, 27, 27}, {1, 1}, {1, 1}, {4, 4}, {1, 1}, {1, 16, 26, 26}},
        {{1, 16, 64, 64}, {2, 2}, {2, 2}, {3, 3}, {1, 1}, {1, 16, 66, 66}}};

// clang-format on

INSTANTIATE_TEST_SUITE_P(AvgPool, InferAvgPoolTests, testing::ValuesIn(inferAvgPoolData));

struct InferConvBackpropData {
    std::vector<int64_t> inputShape;
    std::vector<int64_t> filterShape;
    std::vector<int64_t> windowStrides;
    std::vector<int64_t> dataPaddingBelow;
    std::vector<int64_t> dataPaddingAbove;
    std::vector<int64_t> windowDilations;
    std::vector<int64_t> outputPadding;
    std::vector<int64_t> outDataShape;
};

class InferConvBackpropDataTests : public testing::TestWithParam<InferConvBackpropData> {};

TEST_P(InferConvBackpropDataTests, InferConvBackpropData) {
    const auto params = GetParam();
    const auto shapeI64 = inferConvBackpropOutputShape(params.inputShape, params.filterShape, params.windowStrides,
                                                       params.dataPaddingBelow, params.dataPaddingAbove,
                                                       params.windowDilations, params.outputPadding);

    EXPECT_EQ(to_std_vector(shapeI64), params.outDataShape);
}

// clang-format off

std::vector<InferConvBackpropData> inferConvBackpropData = {{{1, 3, 64, 64},      /* inputShape */
                                                             {3, 16, 2, 2},       // filterShape
                                                             {2, 2},              // strides
                                                             {0, 0},              // dataPaddingBelow
                                                             {0, 0},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {1, 1},              // outputPadding
                                                             {1, 16, 129, 129}},  // outDataShape
                                                            {{1,3,300,300},       /* inputShape */
                                                             {16,3,3,3},          // filterShape
                                                             {1, 1},              // strides
                                                             {1, 1},              // dataPaddingBelow
                                                             {1, 1},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {0, 0},              // outputPadding
                                                             {1,3,300,300}},      // outDataShape
                                                            {{1,64,64,157},       /* inputShape */
                                                             {64,64,1,3},         // filterShape
                                                             {1, 1},              // strides
                                                             {0, 2},              // dataPaddingBelow
                                                             {0, 0},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {0, 0},              // outputPadding
                                                             {1,64,64,157}},      // outDataShape
                                                            {{1,16,23,30},        /* inputShape */
                                                             {16,32,2,1},         // filterShape
                                                             {2, 2},              // strides
                                                             {0, 0},              // dataPaddingBelow
                                                             {0, 0},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {0, 0},              // outputPadding
                                                             {1,32,46,59}}};      // outDataShape

// clang-format on

INSTANTIATE_TEST_SUITE_P(ConvBackpropData, InferConvBackpropDataTests, testing::ValuesIn(inferConvBackpropData));

struct InferGroupConvBackpropData {
    std::vector<int64_t> inputShape;
    std::vector<int64_t> filterShape;
    std::vector<int64_t> windowStrides;
    std::vector<int64_t> dataPaddingBelow;
    std::vector<int64_t> dataPaddingAbove;
    std::vector<int64_t> windowDilations;
    std::vector<int64_t> outputPadding;
    std::vector<int64_t> outDataShape;
};

class InferGroupConvBackpropDataTests : public testing::TestWithParam<InferGroupConvBackpropData> {};

TEST_P(InferGroupConvBackpropDataTests, InferGroupConvBackpropData) {
    const auto params = GetParam();
    const auto shapeI64 = inferGroupConvBackpropOutputShape(params.inputShape, params.filterShape, params.windowStrides,
                                                            params.dataPaddingBelow, params.dataPaddingAbove,
                                                            params.windowDilations, params.outputPadding);

    EXPECT_EQ(to_std_vector(shapeI64), params.outDataShape);
}

// clang-format off

std::vector<InferGroupConvBackpropData> inferGroupConvBackpropData = {
                                                            {{1,16,300,300},    /* inputShape */
                                                             {16,1,1,3,3},      // filterShape
                                                             {1, 1},            // strides
                                                             {1, 1},            // dataPaddingBelow
                                                             {1, 1},            // dataPaddingAbove
                                                             {1, 1},            // windowDilations
                                                             {0, 0},            // outputPadding
                                                             {1,16,300,300}},   // outDataShape
                                                            {{1,32,23},         /* inputShape */
                                                             {2,16,32,2},       // filterShape
                                                             {2},               // strides
                                                             {0},               // dataPaddingBelow
                                                             {0},               // dataPaddingAbove
                                                             {1},               // windowDilations
                                                             {0},               // outputPadding
                                                             {1,64,46}},        // outDataShape
                                                            {{1,96,96,96},      /* inputShape */
                                                             {3,32,32,3,3},     // filterShape
                                                             {1, 1},            // strides
                                                             {1, 1},            // dataPaddingBelow
                                                             {1, 1},            // dataPaddingAbove
                                                             {1, 1},            // windowDilations
                                                             {0, 0},            // outputPadding
                                                             {1,96,96,96}},     // outDataShape
                                                            {{1,32,23,30},      /* inputShape */
                                                             {2,16,32,2,1},     // filterShape
                                                             {2, 2},            // strides
                                                             {0, 0},            // dataPaddingBelow
                                                             {0, 0},            // dataPaddingAbove
                                                             {1, 1},            // windowDilations
                                                             {0, 0},            // outputPadding
                                                             {1,64,46,59}}};    // outDataShape

// clang-format on

INSTANTIATE_TEST_SUITE_P(GroupConvBackpropData, InferGroupConvBackpropDataTests,
                         testing::ValuesIn(inferGroupConvBackpropData));

struct InferTransposedConvBackpropData {
    std::vector<int64_t> inputShape;
    std::vector<int64_t> filterShape;
    std::vector<int64_t> windowStrides;
    std::vector<int64_t> dataPaddingBelow;
    std::vector<int64_t> dataPaddingAbove;
    std::vector<int64_t> windowDilations;
    std::vector<int64_t> outputPadding;
    std::vector<int64_t> outDataShape;
};

class InferTransposedConvBackpropDataTests : public testing::TestWithParam<InferTransposedConvBackpropData> {};

TEST_P(InferTransposedConvBackpropDataTests, InferTransposedConvBackpropData) {
    const auto params = GetParam();
    const auto shapeI64 = inferTransposedConvBackpropOutputShape(
            params.inputShape, params.filterShape, params.windowStrides, params.dataPaddingBelow,
            params.dataPaddingAbove, params.windowDilations, params.outputPadding);

    EXPECT_EQ(to_std_vector(shapeI64), params.outDataShape);
}

// clang-format off

std::vector<InferTransposedConvBackpropData> inferTransposedConvBackpropData = {
                                                            {{1,3,64,64},       /* inputShape */
                                                             {16,3,2,2},        // filterShape
                                                             {2, 2},            // strides
                                                             {0, 0},            // dataPaddingBelow
                                                             {0, 0},            // dataPaddingAbove
                                                             {1, 1},            // windowDilations
                                                             {1, 1},            // outputPadding
                                                             {1,16,129,129}},   // outDataShape
                                                            {{1,16,23,30},      /* inputShape */
                                                             {32,16,2,1},       // filterShape
                                                             {2, 2},            // strides
                                                             {0, 0},            // dataPaddingBelow
                                                             {0, 0},            // dataPaddingAbove
                                                             {1, 1},            // windowDilations
                                                             {0, 0},            // outputPadding
                                                             {1,32,46,59}},     // outDataShape
                                                            {{1,16,23},         /* inputShape */
                                                             {32,16,2},         // filterShape
                                                             {2},               // strides
                                                             {0},               // dataPaddingBelow
                                                             {0},               // dataPaddingAbove
                                                             {1},               // windowDilations
                                                             {0},               // outputPadding
                                                             {1,32,46}},        // outDataShape
                                                            {{1,16,23,30},      /* inputShape */
                                                             {16,16,2,2},       // filterShape
                                                             {2, 2},            // strides
                                                             {0, 0},            // dataPaddingBelow
                                                             {0, 0},            // dataPaddingAbove
                                                             {1, 1},            // windowDilations
                                                             {0, 0},            // outputPadding
                                                             {1,16,46,60}}};    // outDataShape

// clang-format on

INSTANTIATE_TEST_SUITE_P(TransposedConvBackpropData, InferTransposedConvBackpropDataTests,
                         testing::ValuesIn(inferTransposedConvBackpropData));

struct InferTransposedGroupConvBackpropData {
    std::vector<int64_t> inputShape;
    std::vector<int64_t> filterShape;
    std::vector<int64_t> windowStrides;
    std::vector<int64_t> dataPaddingBelow;
    std::vector<int64_t> dataPaddingAbove;
    std::vector<int64_t> windowDilations;
    std::vector<int64_t> outputPadding;
    std::vector<int64_t> outDataShape;
};

class InferTransposedGroupConvBackpropDataTests :
        public testing::TestWithParam<InferTransposedGroupConvBackpropData> {};

TEST_P(InferTransposedGroupConvBackpropDataTests, InferTransposedGroupConvBackpropData) {
    const auto params = GetParam();
    const auto shapeI64 = inferTransposedGroupConvBackpropOutputShape(
            params.inputShape, params.filterShape, params.windowStrides, params.dataPaddingBelow,
            params.dataPaddingAbove, params.windowDilations, params.outputPadding);

    EXPECT_EQ(to_std_vector(shapeI64), params.outDataShape);
}

// clang-format off

std::vector<InferTransposedGroupConvBackpropData> inferTransposedGroupConvBackpropData = {
                                                            {{1,32,23,30},        /* inputShape */
                                                             {2,32,16,2,1},       // filterShape
                                                             {2, 2},              // strides
                                                             {0, 0},              // dataPaddingBelow
                                                             {0, 0},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {0, 0},              // outputPadding
                                                             {1,64,46,59}},       // outDataShape
                                                            {{1,32,23},           /* inputShape */
                                                             {2,32,16,2},         // filterShape
                                                             {2},                 // strides
                                                             {0},                 // dataPaddingBelow
                                                             {0},                 // dataPaddingAbove
                                                             {1},                 // windowDilations
                                                             {0},                 // outputPadding
                                                             {1,64,46}},          // outDataShape
                                                            {{1,64,64,64},        /* inputShape */
                                                             {64,1,1,4,4},        // filterShape
                                                             {2, 2},              // strides
                                                             {0, 0},              // dataPaddingBelow
                                                             {0, 0},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {0, 0},              // outputPadding
                                                             {1, 64, 130, 130}},  // outDataShape
                                                            {{1,32,23,30},        /* inputShape */
                                                             {2,16,32,2,1},       // filterShape
                                                             {2, 2},              // strides
                                                             {0, 0},              // dataPaddingBelow
                                                             {0, 0},              // dataPaddingAbove
                                                             {1, 1},              // windowDilations
                                                             {0, 0},              // outputPadding
                                                             {1,32,46,59}}};      // outDataShape

// clang-format on

INSTANTIATE_TEST_SUITE_P(TransposedGroupConvBackpropData, InferTransposedGroupConvBackpropDataTests,
                         testing::ValuesIn(inferTransposedGroupConvBackpropData));

struct ReifyDimTestData {
    std::vector<int64_t> shape;
    std::vector<int64_t> bounds;
};

class ReifyDimTests : public testing::TestWithParam<ReifyDimTestData> {};

TEST_P(ReifyDimTests, ReifyDimTest) {
    mlir::MLIRContext context(vpux::createDialectRegistry());
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<IE::IEDialect>();

    mlir::OpBuilder builder(&context);

    auto loc = builder.getUnknownLoc();

    const auto params = GetParam();

    EXPECT_EQ(params.shape.size(), params.bounds.size());

    auto rank = params.shape.size();

    auto order = DimsOrder::fromNumDims(rank);
    const auto desc = vpux::getTensorAttr(
            &context, order.toAffineMap(&context),
            vpux::IndexedSymbolAttr::get(&context, stringifyEnum(VPU::MemoryKind::CMX_NN)), BoundsRef(params.bounds));
    auto inType = mlir::RankedTensorType::get(params.shape, builder.getF16Type(), desc);

    auto funcType = builder.getFunctionType({inType}, {});
    mlir::OwningOpRef<mlir::func::FuncOp> func = builder.create<mlir::func::FuncOp>(loc, "TestFunc", funcType);
    auto& entryBlock = *func->addEntryBlock();

    builder.setInsertionPointToStart(&entryBlock);

    auto inArg = entryBlock.getArgument(0);

    for (auto i : irange(rank)) {
        auto result = reifyDim(builder, inArg, i);
        if (params.shape[i] == mlir::ShapedType::kDynamic) {
            EXPECT_TRUE(result.is<mlir::Value>());
            EXPECT_TRUE(mlir::isa<mlir::tensor::DimOp>(result.get<mlir::Value>().getDefiningOp()));
            auto dimOp = mlir::cast<mlir::tensor::DimOp>(result.get<mlir::Value>().getDefiningOp());
            EXPECT_TRUE(dimOp.getConstantIndex().has_value());
            EXPECT_EQ(dimOp.getConstantIndex().value(), i);
        } else {
            EXPECT_TRUE(result.is<mlir::Attribute>());
            EXPECT_TRUE(mlir::isa<mlir::IntegerAttr>(result.get<mlir::Attribute>()));
            EXPECT_EQ(mlir::cast<mlir::IntegerAttr>(result.get<mlir::Attribute>()).getInt(), params.shape[i]);
        }
    }
    builder.create<mlir::func::ReturnOp>(loc);
}

std::vector<ReifyDimTestData> reifyDimData = {
        {{mlir::ShapedType::kDynamic, 23, 30, 10}, {2048, 23, 30, 10}},
        {{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, 30, 10}, {2048, 23, 30, 10}},
        {{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, 10}, {2048, 23, 30, 10}},
        {{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic,
          mlir::ShapedType::kDynamic},
         {2048, 23, 30, 10}}};

INSTANTIATE_TEST_SUITE_P(ReifyDim, ReifyDimTests, testing::ValuesIn(reifyDimData));
