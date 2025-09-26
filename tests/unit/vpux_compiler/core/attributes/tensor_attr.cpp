//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/BuiltinTypeInterfaces.h>

#include <gtest/gtest.h>
#include <exception>

using namespace vpux;

using MLIR_TensorAttr = ::testing::Test;

TEST_F(MLIR_TensorAttr, ThrowWhenBothBoundsAndDynamicDimsMaskSet) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    auto shape = Shape{1, 1, mlir::ShapedType::kDynamic};
    auto bounds = Bounds{1, 1, 4};
    auto dynamicDimsMask = DynamicDimsMask{0, 0, 1};

    EXPECT_NO_THROW(getTensorAttr(&ctx, DimsOrder::CHW, nullptr, bounds, /*DynamicDimsMask=*/{}));
    EXPECT_NO_THROW(getTensorAttr(&ctx, DimsOrder::CHW, nullptr, /*Bounds=*/{}, dynamicDimsMask));
    EXPECT_THROW(getTensorAttr(&ctx, DimsOrder::CHW, nullptr, bounds, dynamicDimsMask), std::exception);
}
