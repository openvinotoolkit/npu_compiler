//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

class MLIR_IETypeInferenceTest : public MLIR_UnitBase {
public:
    MLIR_IETypeInferenceTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect, mlir::func::FuncDialect, IE::IEDialect, mlir::tensor::TensorDialect>();
        listener = std::make_unique<mlir::OpBuilder::Listener>();
        builder = std::make_unique<mlir::OpBuilder>(&ctx, listener.get());
    }

    MLIR_IETypeInferenceTest(const MLIR_IETypeInferenceTest&) = delete;
    MLIR_IETypeInferenceTest& operator=(const MLIR_IETypeInferenceTest&) = delete;

    mlir::OwningOpRef<mlir::tensor::EmptyOp> createOperand(ArrayRef<int64_t> shape, DimsOrder order) {
        return mlir::tensor::EmptyOp::create(*builder, builder->getUnknownLoc(), shape, mlir::Float32Type::get(&ctx),
                                             getTensorAttr(&ctx, order, nullptr));
    }

    mlir::OwningOpRef<mlir::tensor::EmptyOp> createOperand(ArrayRef<int64_t> shape, DimsOrder order,
                                                           ArrayRef<int64_t> bounds) {
        auto tensorAttr = getTensorAttr(&ctx, order, nullptr, BoundsRef(bounds));
        auto emptyOp = mlir::tensor::EmptyOp::create(*builder, builder->getUnknownLoc(), bounds,
                                                     mlir::Float32Type::get(&ctx), tensorAttr);
        auto tensorType = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx), tensorAttr);
        emptyOp->getResult(0).setType(tensorType);
        return emptyOp;
    }

    mlir::DictionaryAttr createAttributes(ArrayRef<mlir::NamedAttribute> attributes) {
        return mlir::DictionaryAttr::get(&ctx, attributes);
    }

    mlir::MLIRContext ctx;
    std::unique_ptr<mlir::OpBuilder::Listener> listener;
    std::unique_ptr<mlir::OpBuilder> builder;
};

}  // namespace

TEST_F(MLIR_IETypeInferenceTest, PadOp) {
    SmallVector<int64_t> shape{2, 2, 2, 2};
    auto operand = createOperand({2, 2, 2, 2}, DimsOrder::NHCW);

    IE::PadOp::Properties properties{};
    properties.pads_begin_attr = getIntArrayAttr<ArrayRef<int32_t>>(&ctx, {0, 0, 0, 0});
    properties.pads_end_attr = getIntArrayAttr<ArrayRef<int32_t>>(&ctx, {4, 0, 0, 0});
    properties.pad_value_attr = getFPAttr(&ctx, 0.0);
    properties.mode = IE::PadModeAttr::get(&ctx, IE::PadMode::CONSTANT);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::PadOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedShape{6, 2, 2, 2};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedShape));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr);
    auto expectedOrder = mlir::AffineMapAttr::get(DimsOrder::NHCW.toAffineMap(&ctx));
    EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
}

using BinaryOpTestParams = std::tuple<SmallVector<int64_t>, DimsOrder, SmallVector<int64_t>, DimsOrder>;

// This workaround is needed because using DimsOrder::NWHC to initialize global variables (as needed for gtest test
// instances) is undefined behaviour.
std::vector<BinaryOpTestParams> getBinaryOpTestCases() {
    return {BinaryOpTestParams{{2, 2, 2, 2}, vpux::DimsOrder::NWHC, {2, 2}, vpux::DimsOrder::CN},
            BinaryOpTestParams{{2, 2}, vpux::DimsOrder::CN, {2, 2, 2, 2}, vpux::DimsOrder::NWHC},
            BinaryOpTestParams{{9, 9, 9, 9}, vpux::DimsOrder::HWCN, {9, 9, 9, 9}, vpux::DimsOrder::NCWH}};
}

TEST_F(MLIR_IETypeInferenceTest, MultiplyOp) {
    for (auto [lhsShape, lhsOrder, rhsShape, rhsOrder] : getBinaryOpTestCases()) {
        bool takeLeft = lhsShape.size() > rhsShape.size();
        auto expectedDims = takeLeft ? lhsShape : rhsShape;
        auto expectedOrder = mlir::AffineMapAttr::get((takeLeft ? lhsOrder : rhsOrder).toAffineMap(&ctx));

        auto lhs = createOperand(lhsShape, lhsOrder);
        auto rhs = createOperand(rhsShape, rhsOrder);

        IE::MultiplyOp::Properties properties{};
        properties.auto_broadcast = IE::AutoBroadcastTypeAttr::get(&ctx, IE::AutoBroadcastType::NUMPY);

        SmallVector<mlir::ShapedTypeComponents> typeComponents;
        ASSERT_TRUE(mlir::succeeded(IE::MultiplyOp::inferReturnTypeComponents(&ctx, builder->getUnknownLoc(),
                                                                              {lhs->getResult(), rhs->getResult()}, {},
                                                                              &properties, {}, typeComponents)));
        ASSERT_EQ(typeComponents.size(), 1);
        ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

        auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
        ASSERT_TRUE(tensorAttr != nullptr);
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

TEST_F(MLIR_IETypeInferenceTest, MultiplyOp_DynamicSecondInput) {
    auto lhs = createOperand({1, 32, 1, 1}, vpux::DimsOrder::NHWC);
    auto rhs = createOperand({1, 32, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, vpux::DimsOrder::NHWC,
                             {1, 32, 800, 1280});

    IE::MultiplyOp::Properties properties{};
    properties.auto_broadcast = IE::AutoBroadcastTypeAttr::get(&ctx, IE::AutoBroadcastType::NUMPY);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::MultiplyOp::inferReturnTypeComponents(&ctx, builder->getUnknownLoc(),
                                                                          {lhs->getResult(), rhs->getResult()}, {},
                                                                          &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 32, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr);

    auto bounds = tensorAttr.getBounds();
    ASSERT_FALSE(bounds.empty()) << "Expected bounds to be present in TensorAttr";
    SmallVector<int64_t> expectedBounds = {1, 32, 800, 1280};
    ASSERT_EQ(bounds.raw(), ArrayRef<int64_t>(expectedBounds));
}

TEST_F(MLIR_IETypeInferenceTest, DivideOp) {
    for (auto [lhsShape, lhsOrder, rhsShape, rhsOrder] : getBinaryOpTestCases()) {
        bool takeLeft = lhsShape.size() > rhsShape.size();
        auto expectedDims = takeLeft ? lhsShape : rhsShape;
        auto expectedOrder = mlir::AffineMapAttr::get((takeLeft ? lhsOrder : rhsOrder).toAffineMap(&ctx));

        auto lhs = createOperand(lhsShape, lhsOrder);
        auto rhs = createOperand(rhsShape, rhsOrder);

        IE::DivideOp::Properties properties{};
        properties.auto_broadcast = IE::AutoBroadcastTypeAttr::get(&ctx, IE::AutoBroadcastType::NUMPY);

        SmallVector<mlir::ShapedTypeComponents> typeComponents;
        ASSERT_TRUE(mlir::succeeded(IE::DivideOp::inferReturnTypeComponents(&ctx, builder->getUnknownLoc(),
                                                                            {lhs->getResult(), rhs->getResult()}, {},
                                                                            &properties, {}, typeComponents)));
        ASSERT_EQ(typeComponents.size(), 1);
        ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

        auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
        ASSERT_TRUE(tensorAttr != nullptr);
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

//
// Reduce bounds propagation. The rule is in the comment on IE::inferReduceReturnTypeComponents, in
// reduce_infer.cpp. All ten IE Reduce operations delegate to that helper, so ReduceMean stands for
// all of them here.
//

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_KeepDimsKeepsBounds) {
    auto operand = createOperand({1, 16, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, vpux::DimsOrder::NCHW,
                                 {1, 16, 48, 400});

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {1});
    properties.keep_dims = mlir::UnitAttr::get(&ctx);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 1, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr) << "The result keeps two dynamic dimensions, so it needs bounds";

    auto bounds = tensorAttr.getBounds();
    ASSERT_FALSE(bounds.empty()) << "Expected bounds to be present in TensorAttr";
    // The bounds follow the same rule as the shape. The reduced axis holds 1, and the rest carry the
    // input bounds.
    SmallVector<int64_t> expectedBounds = {1, 1, 48, 400};
    ASSERT_EQ(bounds.raw(), ArrayRef<int64_t>(expectedBounds));
}

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_WithoutKeepDimsKeepsBounds) {
    auto operand = createOperand({1, 16, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, vpux::DimsOrder::NCHW,
                                 {1, 16, 48, 400});

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {1});

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    // Without keep_dims the reduced axis leaves the shape, and the bounds lose the same entry.
    SmallVector<int64_t> expectedDims = {1, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr);

    auto bounds = tensorAttr.getBounds();
    ASSERT_FALSE(bounds.empty()) << "Expected bounds to be present in TensorAttr";
    SmallVector<int64_t> expectedBounds = {1, 48, 400};
    ASSERT_EQ(bounds.raw(), ArrayRef<int64_t>(expectedBounds));
}

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_ReducingEveryDynamicAxisGivesAStaticResult) {
    auto operand = createOperand({1, 16, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, vpux::DimsOrder::NCHW,
                                 {1, 16, 48, 400});

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {2, 3});
    properties.keep_dims = mlir::UnitAttr::get(&ctx);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 16, 1, 1};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    // Nothing dynamic is left, so the result needs no bounds, and getTensorAttr gives back a null
    // attribute for an identity order with no bounds and no memory space.
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    EXPECT_TRUE(tensorAttr == nullptr) << "A fully static result must not carry bounds";
}

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_StaticInputKeepsANullDescriptor) {
    auto operand = createOperand({1, 16, 48, 400}, vpux::DimsOrder::NCHW);

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {1});
    properties.keep_dims = mlir::UnitAttr::get(&ctx);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 1, 48, 400};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    // A static input gives a plain result type. That means a null descriptor, and not an identity
    // order spelled out, because the two are different MLIR types. Spelling it out would move every
    // static Reduce in every model.
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    EXPECT_TRUE(tensorAttr == nullptr) << "A Reduce over a static input must give a plain result type";
}

//
// Reduce layout propagation. The result carries the layout the input came in with, and not the
// identity order the inference used to write out. With keep_dims the rank does not move, so the
// order carries over as it stands. Without keep_dims the reduced axes leave the shape, and
// IE::calculateReducedOutputLayout gives the order for what is left. VPU::inferReduceReturnTypes
// applies the same two rules after lowering.
//

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_KeepDimsKeepsTheInputLayout) {
    auto operand = createOperand({1, 16, 48, 400}, vpux::DimsOrder::NHWC);

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {1});
    properties.keep_dims = mlir::UnitAttr::get(&ctx);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 1, 48, 400};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr) << "An NHWC input gives a non-identity order, so the result needs a descriptor";
    auto expectedOrder = mlir::AffineMapAttr::get(vpux::DimsOrder::NHWC.toAffineMap(&ctx));
    EXPECT_EQ(tensorAttr.getOrder(), expectedOrder) << "keep_dims leaves the rank alone, so NHWC carries over";
}

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_WithoutKeepDimsUsesTheReducedOrder) {
    auto operand = createOperand({1, 16, 48, 400}, vpux::DimsOrder::NHWC);

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {1});

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 48, 400};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr) << "The reduced NHWC order is not the identity, so it must be written out";
    // calculateReducedOutputLayout matches the axes against the digits of the order code, and those
    // digits hold the dimension index plus one. Axis 1 therefore takes out the digit 1, which is N:
    // NHWC 1342 leaves 342 and renumbers to 231, which is HWC. The axes arrive as plain dimension
    // indices, so the two numberings do not line up, and the table on calculateReducedOutputLayout
    // spells this same case out. VPU::verifyReduceLayoutInfo holds every VPU Reduce to the result.
    auto expectedOrder = mlir::AffineMapAttr::get(vpux::DimsOrder::HWC.toAffineMap(&ctx));
    EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
}

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_WithoutKeepDimsCanReachTheIdentityOrder) {
    auto operand = createOperand({1, 16, 48, 400}, vpux::DimsOrder::NHWC);

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {2});

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedDims = {1, 16, 400};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

    // A reduced order can land on the identity, and then the result type has to stay plain.
    // NHWC 1342 with axis 2 takes out the digit 2, which leaves 134 and renumbers to 123, the
    // identity order for rank 3. getTensorAttr gives back nothing for an identity order with no
    // bounds and no memory space, so an implementation that always writes the order out fails here.
    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    EXPECT_TRUE(tensorAttr == nullptr) << "A reduced order that is the identity must give a plain result type";
}

TEST_F(MLIR_IETypeInferenceTest, ReduceMeanOp_BoundsAndLayoutTravelTogether) {
    auto operand = createOperand({1, 16, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, vpux::DimsOrder::NHWC,
                                 {1, 16, 48, 400});

    IE::ReduceMeanOp::Properties properties{};
    properties.axes_value = getIntArrayAttr<ArrayRef<int64_t>>(&ctx, {1});
    properties.keep_dims = mlir::UnitAttr::get(&ctx);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::ReduceMeanOp::inferReturnTypeComponents(
            &ctx, builder->getUnknownLoc(), {operand->getResult()}, {}, &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr);

    auto expectedOrder = mlir::AffineMapAttr::get(vpux::DimsOrder::NHWC.toAffineMap(&ctx));
    EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);

    auto bounds = tensorAttr.getBounds();
    ASSERT_FALSE(bounds.empty()) << "Expected bounds to be present in TensorAttr";
    SmallVector<int64_t> expectedBounds = {1, 1, 48, 400};
    EXPECT_EQ(bounds.raw(), ArrayRef<int64_t>(expectedBounds));
}
