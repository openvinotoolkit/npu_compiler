//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_MemRefAttr = MLIR_UnitBase;

TEST_F(MLIR_MemRefAttr, ImplicitConversionToMemRefLayoutAttrInterface) {
    mlir::MLIRContext ctx(registry);

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, &ctx);

    // implicit conversion to interface must succeed
    mlir::MemRefLayoutAttrInterface interface = memRef;
    ASSERT_NE(interface, nullptr);
    // roundtrip also succeeds
    ASSERT_EQ(mlir::cast<vpux::MemRefAttr>(interface), memRef);
}

TEST_F(MLIR_MemRefAttr, ImplicitConversionWorksOnNullptrHwSpecificFields) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, &ctx);

    ASSERT_TRUE(memRef.hwSpecificFields().empty());

    // implicit conversion must not fail, even though the fields are not set
    auto implicitlyConvertedSwizzling = memRef.hwSpecificField<VPUIP::SwizzlingSchemeAttr>();
    auto implicitlyConvertedCompression = memRef.hwSpecificField<VPUIP::SparsityCompressionAttr>();
    ASSERT_EQ(implicitlyConvertedSwizzling, nullptr);
    ASSERT_EQ(implicitlyConvertedCompression, nullptr);
}

TEST_F(MLIR_MemRefAttr, BoundsRoundTripAndHwSpecificFieldsRemainEmpty) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const SmallVector<int64_t> expectedBounds = {16, 32, 64, 128};

    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, BoundsRef(expectedBounds), &ctx);
    ASSERT_NE(memRef, nullptr);

    const auto bounds = memRef.bounds();
    ASSERT_EQ(bounds.size(), expectedBounds.size());
    for (size_t idx = 0; idx < expectedBounds.size(); ++idx) {
        EXPECT_EQ(bounds[Dim(idx)], expectedBounds[idx]);
    }

    ASSERT_TRUE(memRef.hwSpecificFields().empty());
}

TEST_F(MLIR_MemRefAttr, BoundsAreEmptyWhenNotSet) {
    mlir::MLIRContext ctx(registry);

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, &ctx);

    ASSERT_TRUE(memRef.bounds().empty());
}

TEST_F(MLIR_MemRefAttr, ClassofRejectsNonOpaqueBoundsType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto i64Type = mlir::IntegerType::get(&ctx, 64);
    const auto badBounds = mlir::ArrayAttr::get(&ctx, {mlir::IntegerAttr::get(i64Type, 42)});

    const auto dict = mlir::DictionaryAttr::get(
            &ctx, {
                          mlir::NamedAttribute(mlir::StringAttr::get(&ctx, "order"), orderAttr),
                          mlir::NamedAttribute(mlir::StringAttr::get(&ctx, "bounds"), badBounds),
                  });

    ASSERT_EQ(mlir::dyn_cast<vpux::MemRefAttr>(dict), nullptr);
}

TEST_F(MLIR_MemRefAttr, GetStridesAndOffsetFailsWithoutBoundsOrStridesOnDynamicShape) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, &ctx);

    const SmallVector<int64_t> shape{1, mlir::ShapedType::kDynamic, 32, 32};
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    const auto layoutIface = mlir::cast<mlir::MemRefLayoutAttrInterface>(memRef);
    ASSERT_TRUE(mlir::failed(layoutIface.getStridesAndOffset(shape, strides, offset)));
}

TEST_F(MLIR_MemRefAttr, GetStridesAndOffsetDerivesCompactStridesFromBoundsOnDynamicShape) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const SmallVector<int64_t> boundsValues{1, 16, 32, 32};
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, BoundsRef(boundsValues), &ctx);

    // Dynamic on dim 1; strides aren't explicit on the attribute, so compact strides must be
    // derived from the bound (16), not left dynamic. The offset is a static zero: bounds bound
    // the shape, not the view's start position.
    const SmallVector<int64_t> shape{1, mlir::ShapedType::kDynamic, 32, 32};
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    const auto layoutIface = mlir::cast<mlir::MemRefLayoutAttrInterface>(memRef);
    ASSERT_TRUE(mlir::succeeded(layoutIface.getStridesAndOffset(shape, strides, offset)));

    EXPECT_EQ(offset, 0);

    const SmallVector<int64_t> expectedStrides{16 * 32 * 32, 32 * 32, 32, 1};
    EXPECT_EQ(strides, expectedStrides);
}

TEST_F(MLIR_MemRefAttr, GetStridesAndOffsetPrefersExplicitStridesOverBounds) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const SmallVector<int64_t> boundsValues{1, 16, 32, 32};
    const SmallVector<int64_t> explicitStrides{2048, 1024, 32, 1};
    const auto stridesAttr = getIntArrayAttr(&ctx, explicitStrides);
    const auto memRef = vpux::MemRefAttr::get(orderAttr, stridesAttr, nullptr, BoundsRef(boundsValues), {}, &ctx);

    const SmallVector<int64_t> shape{1, mlir::ShapedType::kDynamic, 32, 32};
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    const auto layoutIface = mlir::cast<mlir::MemRefLayoutAttrInterface>(memRef);
    ASSERT_TRUE(mlir::succeeded(layoutIface.getStridesAndOffset(shape, strides, offset)));

    EXPECT_EQ(offset, 0);
    EXPECT_EQ(strides, explicitStrides);
}

TEST_F(MLIR_MemRefAttr, GetStridesAndOffsetReturnsDynamicOffset) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const SmallVector<int64_t> boundsValues{1, 16, 32, 32};
    const auto baseMemRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, BoundsRef(boundsValues), &ctx);
    // The common case for bufferized tensor.extract_slice/insert_slice with a runtime-computed
    // start position: the offset itself is not known at compile time. getOffsetOrZero() has no
    // special-casing for this value, so it also stands in for "any non-zero offset propagates
    // unchanged on the bounds-derived-strides path".
    const auto memRef = baseMemRef.addOffset(mlir::ShapedType::kDynamic);

    const SmallVector<int64_t> shape{1, mlir::ShapedType::kDynamic, 32, 32};
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    const auto layoutIface = mlir::cast<mlir::MemRefLayoutAttrInterface>(memRef);
    ASSERT_TRUE(mlir::succeeded(layoutIface.getStridesAndOffset(shape, strides, offset)));

    EXPECT_TRUE(mlir::ShapedType::isDynamic(offset));

    const SmallVector<int64_t> expectedStrides{16 * 32 * 32, 32 * 32, 32, 1};
    EXPECT_EQ(strides, expectedStrides);
}

TEST_F(MLIR_MemRefAttr, GetStridesAndOffsetPermutesCompactStridesForNonIdentityOrder) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NHWC;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const SmallVector<int64_t> boundsValues{1, 16, 32, 32};
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, BoundsRef(boundsValues), &ctx);

    // Dynamic on dim 1 (C); compact strides must be computed in NHWC memory order (N, H, W, C) and
    // converted back to logical (N, C, H, W) order, not simply treated as if order were identity.
    const SmallVector<int64_t> shape{1, mlir::ShapedType::kDynamic, 32, 32};
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    const auto layoutIface = mlir::cast<mlir::MemRefLayoutAttrInterface>(memRef);
    ASSERT_TRUE(mlir::succeeded(layoutIface.getStridesAndOffset(shape, strides, offset)));

    EXPECT_EQ(offset, 0);

    // Memory order is N, H, W, C: C has stride 1, W has stride C-size (16), H has stride W*C (512),
    // N has stride H*W*C (16384). Expressed in logical (N, C, H, W) order below.
    const SmallVector<int64_t> expectedStrides{16 * 32 * 32, 1, 32 * 16, 16};
    EXPECT_EQ(strides, expectedStrides);
}

TEST_F(MLIR_MemRefAttr, IRRoundTripPreservesOrderAndBounds) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NHWC;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const SmallVector<int64_t> expectedBounds = {1, 18, 3, 3};

    // Parse full MemRef type from IR string
    const char* irType = "memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, "
                         "order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>";
    const auto parsedType = mlir::parseType(irType, &ctx);
    ASSERT_NE(parsedType, nullptr) << "Failed to parse MemRef type IR: " << irType;

    const auto parsedMemRefType = mlir::dyn_cast<mlir::MemRefType>(parsedType);
    ASSERT_NE(parsedMemRefType, nullptr) << "Parsed type is not MemRefType";

    const auto parsedMemRef = mlir::dyn_cast<vpux::MemRefAttr>(parsedMemRefType.getLayout());
    ASSERT_NE(parsedMemRef, nullptr) << "MemRef layout is not vpux::MemRefAttr";

    // Verify order is preserved
    EXPECT_EQ(parsedMemRef.order(), orderAttr);
    const auto parsedOrder = DimsOrder::fromAffineMap(parsedMemRef.order().getValue());
    EXPECT_EQ(parsedOrder, order);
    EXPECT_FALSE(parsedMemRef.order().getValue().isIdentity());

    // Verify bounds are preserved
    const auto parsedBounds = parsedMemRef.bounds();
    ASSERT_EQ(parsedBounds.size(), expectedBounds.size());
    for (size_t idx = 0; idx < expectedBounds.size(); ++idx) {
        EXPECT_EQ(parsedBounds[Dim(idx)], expectedBounds[idx]);
    }
}
