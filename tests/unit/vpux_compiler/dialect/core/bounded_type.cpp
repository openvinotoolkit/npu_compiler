//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/dialect.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace {

class MLIR_BoundedTensorTypeTest : public MLIR_UnitBase {  // NOLINT case style
public:
    MLIR_BoundedTensorTypeTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<mlir::tensor::TensorDialect, vpux::Core::CoreDialect, vpux::Const::ConstDialect>();
        listener = std::make_unique<mlir::OpBuilder::Listener>();
        builder = std::make_unique<mlir::OpBuilder>(&ctx, listener.get());
    }

    MLIR_BoundedTensorTypeTest(const MLIR_BoundedTensorTypeTest&) = delete;
    MLIR_BoundedTensorTypeTest& operator=(const MLIR_BoundedTensorTypeTest&) = delete;
    MLIR_BoundedTensorTypeTest(MLIR_BoundedTensorTypeTest&&) = delete;
    MLIR_BoundedTensorTypeTest& operator=(MLIR_BoundedTensorTypeTest&&) = delete;

    mlir::Type getStaticTensorType() {
        const auto shape = SmallVector<int64_t>{1, 2, 10, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr);

        return mlir::RankedTensorType::get(shape, elemType, tensorAttr);
    }

    std::pair<mlir::Type, SmallVector<int64_t>> getTensorWithBoundsType() {
        const auto shape = SmallVector<int64_t>{1, 2, mlir::ShapedType::kDynamic, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto bounds = SmallVector<int64_t>{1, 2, 10, 20};
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr, BoundsRef(bounds));
        const auto type = mlir::RankedTensorType::get(shape, elemType, tensorAttr);

        return std::make_pair(type, bounds);
    };

    mlir::MLIRContext ctx;
    std::unique_ptr<mlir::OpBuilder::Listener> listener;
    std::unique_ptr<mlir::OpBuilder> builder;
};

}  // namespace

TEST_F(MLIR_BoundedTensorTypeTest, NoBounds) {
    auto type = getStaticTensorType();

    const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    ASSERT_TRUE(boundedType == nullptr);
}

TEST_F(MLIR_BoundedTensorTypeTest, ThrowWithStaticShape) {
    auto type = getStaticTensorType();

    const auto bounds = Bounds{1, 2, 10, 20};
    EXPECT_THROW(Core::BoundedTensorType::get(type, bounds), std::exception);
}

TEST_F(MLIR_BoundedTensorTypeTest, Get) {
    auto type = getStaticTensorType();
    auto ndType = mlir::cast<NDTypeInterface>(type);
    const Shape shape(ndType.getShape().size(), mlir::ShapedType::kDynamic);
    auto dynShapeType = ndType.changeShape(shape);

    const auto bounds = Bounds{1, 2, 10, 20};
    const auto boundedType = Core::BoundedTensorType::get(dynShapeType, bounds);
    ASSERT_TRUE(boundedType != nullptr);
    ASSERT_TRUE(llvm::equal(boundedType.getBounds(), bounds));
}

TEST_F(MLIR_BoundedTensorTypeTest, GetBounds) {
    auto [type, bounds] = getTensorWithBoundsType();

    const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    ASSERT_TRUE(boundedType != nullptr);

    const auto typeBounds = boundedType.getBounds();
    ASSERT_TRUE(llvm::equal(typeBounds, bounds));
}

TEST_F(MLIR_BoundedTensorTypeTest, ChangeBounds) {
    auto [type, bounds] = getTensorWithBoundsType();

    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    ASSERT_TRUE(boundedType != nullptr);

    const auto newBounds = mlir::SmallVector<int64_t>{1, 2, 20, 20};
    ASSERT_TRUE(bounds != newBounds);

    const auto newType = boundedType.changeBounds(BoundsRef(newBounds));
    ASSERT_TRUE(llvm::equal(newType.getBounds(), newBounds));
}

TEST_F(MLIR_BoundedTensorTypeTest, CallOnStaticShape) {
    const auto type = getStaticTensorType();
    const int64_t argument = 4;
    const int64_t capture = 4;

    const auto size = callOnShapeOf(
            type,
            [capture](const auto& shape, int64_t argument) {
                EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(shape)>, ShapeRef>));

                auto shapeCopy = copyShape(shape);
                EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(shapeCopy)>, Shape>));
                EXPECT_EQ(llvm::formatv("{0}", shapeCopy).str(), "[1, 2, 10, 20]");

                shapeCopy[Dims4D::Act::H] += shapeCopy[Dims4D::Act::W] * argument + capture;
                EXPECT_EQ(llvm::formatv("{0}", shapeCopy).str(), "[1, 2, 94, 20]");

                std::swap(shapeCopy[Dims4D::Act::H], shapeCopy[Dims4D::Act::C]);
                EXPECT_EQ(llvm::formatv("{0}", shapeCopy).str(), "[1, 94, 2, 20]");

                return shapeCopy.totalSize();
            },
            argument);
    EXPECT_EQ(size, 1 * 94 * 2 * 20);
}

TEST_F(MLIR_BoundedTensorTypeTest, CallOnShape) {
    const auto [type, _] = getTensorWithBoundsType();
    const int64_t argument = 4;
    const int64_t capture = 4;

    const auto size = callOnShapeOf(
            type,
            [capture](const auto& shape, int64_t argument) {
                EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(shape)>, BoundedShape>));

                auto shapeCopy = copyShape(shape);
                EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(shapeCopy)>, BoundedShape>));
                EXPECT_EQ(llvm::formatv("{0}", shapeCopy).str(), "[1, 2, 1..10, 20]");

                shapeCopy[Dims4D::Act::H] += shapeCopy[Dims4D::Act::W] * argument + capture;
                EXPECT_EQ(llvm::formatv("{0}", shapeCopy).str(), "[1, 2, 1..94, 20]");

                std::swap(shapeCopy[Dims4D::Act::H], shapeCopy[Dims4D::Act::C]);
                EXPECT_EQ(llvm::formatv("{0}", shapeCopy).str(), "[1, 1..94, 2, 20]");

                return shapeCopy.totalSize();
            },
            argument);
    EXPECT_EQ(size, 1 * 94 * 2 * 20);
}
