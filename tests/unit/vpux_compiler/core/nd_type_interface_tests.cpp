//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

constexpr StringRef CMX_NAME = "CMX_NN";
constexpr StringRef DDR_NAME = "DDR";

using MLIR_NDTypeInterface = MLIR_UnitBase;

TEST_F(MLIR_NDTypeInterface, RankedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const Shape shape{1, 16, 32, 32};
    const auto tensorType = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(tensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 16 * 32 * 32);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({262144_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 32768);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 32768);

    const SmallVector<int64_t> newShape({1, 16, 64, 16});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));

    const auto newElemType = mlir::Float32Type::get(&ctx);
    const auto changedElemType = ndType.changeElemType(newElemType);
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    const SmallVector<int64_t> newShape2({1, 32, 32, 16});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    const auto newDimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = IndexedSymbolAttr::get(&ctx, CMX_NAME);
    auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);

    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef({})));

    const auto newTypeComponents = TypeComponents()
                                           .setShape(ShapeRef(newShape))
                                           .setElementType(newElemType)
                                           .setDimsOrder(newDimsOrder)
                                           .setMemSpace(newMemSpace);
    const auto changedTypeComponents = ndType.changeTypeComponents(newTypeComponents);
    EXPECT_EQ(changedTypeComponents.getShape(), ShapeRef(newShape));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedTypeComponents.getElementType()));
    EXPECT_EQ(changedTypeComponents.getDimsOrder(), newDimsOrder);
    EXPECT_EQ(changedTypeComponents.getMemSpace().getLeafName(), CMX_NAME);

    const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
    const SmallVector<int64_t> tileShape({1, 8, 32, 32});
    const SmallVector<Bit> tileStrides({131072_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const auto outputTile = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(outputTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(outputTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    const SmallVector<int64_t> paddedShape({1, 16, 34, 34});
    const auto paddedOutput = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedOutput.getShape(), ShapeRef(paddedShape));
}

TEST_F(MLIR_NDTypeInterface, UnrankedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const Shape shape{};
    const auto tensorType = mlir::UnrankedTensorType::get(mlir::Float16Type::get(&ctx));

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(tensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_ANY_THROW(ndType.getMemShape());

    EXPECT_FALSE(ndType.hasRank());
    EXPECT_ANY_THROW(ndType.getRank());
    EXPECT_ANY_THROW(ndType.getNumElements());

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_ANY_THROW(ndType.getDimsOrder());

    EXPECT_ANY_THROW(ndType.getMemSpace());
    EXPECT_ANY_THROW(ndType.getMemoryKind());

    EXPECT_ANY_THROW(ndType.getStrides());
    EXPECT_ANY_THROW(ndType.getMemStrides());

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_ANY_THROW(ndType.getTotalAllocSize());
    EXPECT_ANY_THROW(ndType.getCompactAllocSize());

    const SmallVector<int64_t> newShape({1, 16, 64, 16});
    EXPECT_ANY_THROW(ndType.changeShape(ShapeRef(newShape)));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef({})));

    EXPECT_ANY_THROW(ndType.changeShapeElemType(ShapeRef(newShape), mlir::Float32Type::get(&ctx)));
    EXPECT_ANY_THROW(ndType.changeDimsOrder(DimsOrder::NHWC));
    EXPECT_ANY_THROW(ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME)));
    EXPECT_ANY_THROW(ndType.changeMemSpace(VPU::MemoryKind::CMX_NN));
    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
    const SmallVector<int64_t> tileShape({1, 8, 32, 32});
    EXPECT_ANY_THROW(ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape)));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    EXPECT_ANY_THROW(ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter)));
}

TEST_F(MLIR_NDTypeInterface, MemRefType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const Shape shape{1, 16, 32, 32};
    const DimsOrder order = DimsOrder::NCHW;
    const mlir::AffineMapAttr layout = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto memrefType = mlir::MemRefType::get(shape.raw(), mlir::Float16Type::get(&ctx), layout, memSpace);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(memrefType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 16 * 32 * 32);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), order);

    EXPECT_EQ(ndType.getMemSpace(), memSpace);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({262144_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 32768);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 32768);

    const SmallVector<int64_t> newShape({1, 16, 64, 16});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));

    const auto newElemType = mlir::Float32Type::get(&ctx);
    const auto changedElemType = ndType.changeElemType(newElemType);
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    const SmallVector<int64_t> newShape2({1, 32, 32, 16});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    const auto newDimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = IndexedSymbolAttr::get(&ctx, CMX_NAME);
    auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);

    const SmallVector<Bit> newStrides({1048576_Bit, 32768_Bit, 1024_Bit, 16_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
    const SmallVector<int64_t> tileShape({1, 8, 32, 32});
    const SmallVector<Bit> tileStrides({131072_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const auto outputTile = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(outputTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(outputTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> tileStrides2({524288_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), tileStrides2);

    const auto erasedTiledInfo = changedStrides.eraseTiledInfo();
    EXPECT_EQ(erasedTiledInfo.getStrides().raw(), strides);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    const SmallVector<int64_t> paddedShape({1, 16, 34, 34});
    const auto paddedOutput = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedOutput.getShape(), ShapeRef(paddedShape));
}

TEST_F(MLIR_NDTypeInterface, UnrankedMemRefType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const Shape shape{};
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto memrefType = mlir::UnrankedMemRefType::get(mlir::Float16Type::get(&ctx), memSpace);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(memrefType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_ANY_THROW(ndType.getMemShape());

    EXPECT_FALSE(ndType.hasRank());
    EXPECT_ANY_THROW(ndType.getRank());
    EXPECT_ANY_THROW(ndType.getNumElements());

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_ANY_THROW(ndType.getDimsOrder());

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), DDR_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    EXPECT_ANY_THROW(ndType.getStrides());
    EXPECT_ANY_THROW(ndType.getMemStrides());

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_ANY_THROW(ndType.getTotalAllocSize());
    EXPECT_ANY_THROW(ndType.getCompactAllocSize());

    const SmallVector<int64_t> newShape({1, 16, 64, 16});
    EXPECT_ANY_THROW(ndType.changeShape(ShapeRef(newShape)));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    EXPECT_ANY_THROW(ndType.changeShapeElemType(ShapeRef(newShape), mlir::Float32Type::get(&ctx)));
    EXPECT_ANY_THROW(ndType.changeDimsOrder(DimsOrder::NHWC));

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);

    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef({})));

    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
    const SmallVector<int64_t> tileShape({1, 8, 32, 32});
    EXPECT_ANY_THROW(ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape)));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    EXPECT_ANY_THROW(ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter)));
}

TEST_F(MLIR_NDTypeInterface, CompressedMemRefType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, 16, 32, 32};
    const DimsOrder order = DimsOrder::NCHW;
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);

    SmallVector<int64_t> numElems(16);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({16}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const int64_t compressionAxis = 1;
    const int64_t alignment = 16;
    const auto sparsityCompression = VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                         numElemsAttr, getIntAttr(&ctx, alignment));
    const auto memrefType = getMemRefType(shape, mlir::Float16Type::get(&ctx), order, memSpace, StridesRef(),
                                          BoundsRef(), nullptr, sparsityCompression);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(memrefType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));

    int64_t totalByteSize = 0;
    for (auto elems : numElems) {
        totalByteSize += alignValUp<int64_t>(elems * sizeof(vpux::type::float16), alignment);
    }
    EXPECT_EQ(ndType.getTotalAllocSize().count(), totalByteSize);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), totalByteSize);

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({1, 8, 32, 32});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto memrefTiledType = mlir::dyn_cast<mlir::MemRefType>(tiledType);
    ASSERT_TRUE(memrefTiledType != nullptr);
    auto layout = mlir::dyn_cast_or_null<vpux::MemRefAttr>(memrefTiledType.getLayout());
    ASSERT_TRUE(layout != nullptr);
    auto tiledNumElems = layout.hwSpecificField<VPUIP::SparsityCompressionAttr>().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);
}

TEST_F(MLIR_NDTypeInterface, ExplicitSizeMemRefType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, 1, 1, 512};
    const DimsOrder order = DimsOrder::NCHW;

    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));

    const int64_t explicitSize = 1024;
    mlir::IntegerAttr allocSizeAttr = getIntAttr(&ctx, explicitSize);

    const auto newLayoutAttr = vpux::MemRefAttr::get(orderAttr, nullptr, allocSizeAttr, &ctx);

    mlir::MemRefType::Builder builder(shape.raw(), mlir::IntegerType::get(&ctx, 8));
    builder.setLayout(mlir::cast<mlir::MemRefLayoutAttrInterface>(newLayoutAttr));

    auto memrefType = static_cast<mlir::MemRefType>(builder);
    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(memrefType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), explicitSize);
}

// Strides of a shape without an upper bound are unknown at compile time, so they are rejected, and a type rebuilt
// from such a type is given none.
TEST_F(MLIR_NDTypeInterface, UnboundedTypeStridesAreRejected) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, mlir::ShapedType::kDynamic, 16, 16};
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto tensorType = mlir::cast<vpux::NDTypeInterface>(getTensorType(shape, elemType, DimsOrder::NCHW, nullptr));
    EXPECT_ANY_THROW(tensorType.getStrides());
    EXPECT_ANY_THROW(tensorType.getMemStrides());
    EXPECT_TRUE(getStridesUnlessUnbounded(tensorType).empty());

    const auto memrefType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, elemType, DimsOrder::NCHW));
    EXPECT_ANY_THROW(memrefType.getStrides());
    EXPECT_ANY_THROW(memrefType.getMemStrides());
    EXPECT_TRUE(getStridesUnlessUnbounded(memrefType).empty());
}

// A shape without an upper bound has only the identity layout, which MLIR reads as dense for the runtime shape.
TEST_F(MLIR_NDTypeInterface, UnboundedMemRefTypeLayout) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, mlir::ShapedType::kDynamic, 16, 16};
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto memrefType = getMemRefType(shape, elemType, DimsOrder::NCHW);
    EXPECT_TRUE(memrefType.getLayout().isIdentity());
    const auto [mlirStrides, mlirOffset] = memrefType.getStridesAndOffset();
    EXPECT_EQ(mlirStrides, SmallVector<int64_t>({mlir::ShapedType::kDynamic, 256, 16, 1}));
    EXPECT_EQ(mlirOffset, 0);

    // Explicit strides are rejected, even the ones a dense layout would have.
    const auto unknownStride = Bit(mlir::ShapedType::kDynamic);
    const Strides denseStrides({unknownStride, 4096_Bit, 256_Bit, 16_Bit});
    EXPECT_ANY_THROW(getMemRefType(shape, elemType, DimsOrder::NCHW, nullptr, denseStrides));

    // vpux::MemRefAttr requires bounds for a dynamic shape, so a non-identity order is rejected, and so is any layout
    // extra on the identity order.
    EXPECT_ANY_THROW(getMemRefType(shape, elemType, DimsOrder::NHWC));
    EXPECT_ANY_THROW(
            getMemRefType(shape, elemType, DimsOrder::NCHW, nullptr, {}, {}, nullptr, nullptr, getIntAttr(&ctx, 8192)));
    const auto swizzlingScheme = VPUIP::SwizzlingSchemeAttr::get(&ctx, getIntAttr(&ctx, 5), getIntAttr(&ctx, 512));
    EXPECT_ANY_THROW(getMemRefType(shape, elemType, DimsOrder::NCHW, nullptr, {}, {}, swizzlingScheme));

    const auto numElemsType = mlir::RankedTensorType::get({16}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(SmallVector<int64_t>(16, 1)));
    const auto sparsityCompression =
            VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, 1), numElemsAttr, getIntAttr(&ctx, 16));
    EXPECT_ANY_THROW(VPUIP::setSparsityCompressionAttr(memrefType, sparsityCompression));
}

// Changing the memory space keeps the identity layout.
TEST_F(MLIR_NDTypeInterface, UnboundedMemRefTypeChangeMemSpace) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, mlir::ShapedType::kDynamic, 16, 16};
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, elemType, DimsOrder::NCHW));

    const auto memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto ddrType = ndType.changeMemSpace(memSpace);
    EXPECT_EQ(ddrType, mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, elemType, DimsOrder::NCHW, memSpace)));
    EXPECT_TRUE(mlir::cast<mlir::MemRefType>(ddrType).getLayout().isIdentity());
}

// An unbounded type passes no strides on, so the new type is dense for its own components.
TEST_F(MLIR_NDTypeInterface, UnboundedMemRefTypeChangeTypeComponents) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, mlir::ShapedType::kDynamic, 16, 16};
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, f16Type, DimsOrder::NCHW));

    const auto memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    EXPECT_EQ(ndType.changeTypeComponents(TypeComponents().setMemSpace(memSpace)),
              mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, f16Type, DimsOrder::NCHW, memSpace)));

    const Shape staticShape{1, 128, 16, 16};
    EXPECT_EQ(ndType.changeTypeComponents(TypeComponents().setShape(staticShape)),
              mlir::cast<vpux::NDTypeInterface>(getMemRefType(staticShape, f16Type, DimsOrder::NCHW)));

    const Shape otherUnboundedShape{1, mlir::ShapedType::kDynamic, 8, 16};
    EXPECT_EQ(ndType.changeTypeComponents(TypeComponents().setShape(otherUnboundedShape)),
              mlir::cast<vpux::NDTypeInterface>(getMemRefType(otherUnboundedShape, f16Type, DimsOrder::NCHW)));

    const Bounds bounds{1, 32, 16, 16};
    EXPECT_EQ(ndType.changeTypeComponents(TypeComponents().setBounds(Bounds(bounds))),
              mlir::cast<vpux::NDTypeInterface>(
                      getMemRefType(shape, f16Type, DimsOrder::NCHW, nullptr, StridesRef(), bounds)));

    const auto f32Type = mlir::Float32Type::get(&ctx);
    EXPECT_EQ(ndType.changeTypeComponents(TypeComponents().setElementType(f32Type)),
              mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, f32Type, DimsOrder::NCHW)));
}

// An explicit strided layout with unknown strides cannot be rebuilt, only kept as is.
TEST_F(MLIR_NDTypeInterface, UnboundedStridedMemRefType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto dyn = mlir::ShapedType::kDynamic;
    const auto layout = mlir::StridedLayoutAttr::get(&ctx, dyn, {dyn, dyn, dyn, dyn});
    const auto memrefType = mlir::MemRefType::get({1, dyn, 16, 16}, mlir::Float16Type::get(&ctx), layout);
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(memrefType);

    EXPECT_ANY_THROW(ndType.getStrides());
    EXPECT_ANY_THROW(getStridesUnlessUnbounded(ndType));

    const Shape staticShape{1, 128, 16, 16};
    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(staticShape)));

    const auto memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    EXPECT_EQ(mlir::cast<mlir::MemRefType>(ndType.changeMemSpace(memSpace)).getLayout(), layout);
}

// No shared layout can be proven without strides, so an in-place op is refused instead of throwing.
TEST_F(MLIR_NDTypeInterface, UnboundedTypeIsNotCompatibleForInplaceOp) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, mlir::ShapedType::kDynamic, 16, 16};
    const auto elemType = mlir::Float16Type::get(&ctx);
    const Bounds bounds{1, 32, 16, 16};
    const auto unboundedType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(shape, elemType, DimsOrder::NCHW));
    const auto boundedType = mlir::cast<vpux::NDTypeInterface>(
            getMemRefType(shape, elemType, DimsOrder::NCHW, nullptr, StridesRef(), bounds));
    const auto log = Logger::global();

    EXPECT_FALSE(isCompatibleForInplaceOp(unboundedType, unboundedType, log));
    EXPECT_FALSE(isCompatibleForInplaceOp(unboundedType, boundedType, log));
    EXPECT_FALSE(isCompatibleForInplaceOp(boundedType, unboundedType, log));
    EXPECT_TRUE(isCompatibleForInplaceOp(boundedType, boundedType, log));
}

TEST_F(MLIR_NDTypeInterface, BoundedMemRefTypeStrides) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, mlir::ShapedType::kDynamic, 16, 16};
    const Bounds bounds{1, 32, 16, 16};
    const auto memrefType = getMemRefType(shape, mlir::Float16Type::get(&ctx), DimsOrder::NCHW, nullptr, {}, bounds);
    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(memrefType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // A dynamic shape with an upper bound keeps the compact strides of its bounds.
    const SmallVector<Bit> strides({131072_Bit, 4096_Bit, 256_Bit, 16_Bit});
    EXPECT_FALSE(ndType.getStrides().isDynamic());
    EXPECT_EQ(ndType.getStrides().raw(), strides);
}
