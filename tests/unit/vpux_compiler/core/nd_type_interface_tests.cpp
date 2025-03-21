//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/utils.hpp"

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

    const auto ndType = tensorType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 16 * 32 * 32);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

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
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());

    const SmallVector<int64_t> newShape2({1, 32, 32, 16});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::IntegerType>());

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
    EXPECT_TRUE(changedTypeComponents.getElementType().isa<mlir::Float32Type>());
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

    const auto ndType = tensorType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_ANY_THROW(ndType.getMemShape());

    EXPECT_FALSE(ndType.hasRank());
    EXPECT_ANY_THROW(ndType.getRank());
    EXPECT_ANY_THROW(ndType.getNumElements());

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

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
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());

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

    const auto ndType = memrefType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 16 * 32 * 32);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

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
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());

    const SmallVector<int64_t> newShape2({1, 32, 32, 16});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::IntegerType>());

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

    const auto ndType = memrefType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_ANY_THROW(ndType.getMemShape());

    EXPECT_FALSE(ndType.hasRank());
    EXPECT_ANY_THROW(ndType.getRank());
    EXPECT_ANY_THROW(ndType.getNumElements());

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

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
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());

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
    const auto memrefType = getMemRefType(shape, mlir::Float16Type::get(&ctx), order, memSpace, StridesRef(), nullptr,
                                          sparsityCompression);

    const auto ndType = memrefType.dyn_cast<vpux::NDTypeInterface>();
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
    const auto memrefTiledType = tiledType.dyn_cast<mlir::MemRefType>();
    ASSERT_TRUE(memrefTiledType != nullptr);
    auto layout = memrefTiledType.getLayout().dyn_cast_or_null<vpux::MemRefAttr>();
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
    builder.setLayout(newLayoutAttr.cast<mlir::MemRefLayoutAttrInterface>());

    auto memrefType = static_cast<mlir::MemRefType>(builder);
    const auto ndType = memrefType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), explicitSize);
}
