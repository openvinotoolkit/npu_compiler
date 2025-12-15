//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>
#include <numeric>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";

}  // namespace

using MLIR_NDTypeInterface = MLIR_UnitBase;

TEST_F(MLIR_NDTypeInterface, SparseTensorType_Weights) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const Shape weightsShape{64, 16, 3, 3};
    const Shape sparsityMapShape{64, 1, 1, 256};
    const auto data = mlir::RankedTensorType::get(weightsShape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMap = mlir::RankedTensorType::get(sparsityMapShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto isWeights = mlir::UnitAttr::get(&ctx);
    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto sparsityCompression = VPU::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                       numElemsAttr, getIntAttr(&ctx, alignment));
    const auto sparseTensorType =
            VPU::SparseTensorType::get(data, sparsityMap, nullptr, isWeights, sparsityCompression);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(weightsShape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(weightsShape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({2304_Bit, 144_Bit, 48_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);

    int64_t dataByteSize = 0;
    for (auto elems : numElems) {
        dataByteSize += alignValUp<int64_t>(elems * sizeof(vpux::type::float16), alignment);
    }
    EXPECT_EQ(ndType.getTotalAllocSize().count(), dataByteSize + 64 * 256 / CHAR_BIT);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), dataByteSize + 64 * 256 / CHAR_BIT);

    const SmallVector<int64_t> newShape({32, 16, 3, 3});
    const SmallVector<int64_t> newSMShape({32, 1, 1, 256});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShape);
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getData()).getShape(), ShapeRef(newShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getSparsityMap()).getShape(), ShapeRef(newSMShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));
    const auto sparseChangedElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedElemType);
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getSparsityMap()).getElementType()));

    const SmallVector<int64_t> newShape2({32, 16, 5, 5});
    const SmallVector<int64_t> newSMShape2({32, 1, 1, 512});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(changedShapeElemType.getElementType()));
    const auto sparseChangedShapeElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShapeElemType);
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getShape(), ShapeRef(newShape2));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getShape(),
              ShapeRef(newSMShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getElementType()));

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedDimsOrder);
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getData()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getSparsityMap()).getDimsOrder(),
              DimsOrder::NCHW);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);

    const SmallVector<Bit> newStrides({4608_Bit, 144_Bit, 48_Bit, 16_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 16, 3, 3});
    const SmallVector<int64_t> smTileShape({32, 1, 1, 256});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), ShapeRef(tileShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(smTileShape));
    auto tiledNumElems = sparseTiledType.getSparsityCompression().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    const SmallVector<int64_t> paddedShape({64, 16, 5, 5});
    const SmallVector<int64_t> paddedSMShape({64, 1, 1, 512});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(paddedType);
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getData()).getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getSparsityMap()).getShape(), ShapeRef(paddedSMShape));
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_Activation) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape seTableShape{1, 1, 32, 32};
    const auto data = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMap = mlir::RankedTensorType::get(shape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto storageElementTable = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));
    const auto sparseTensorType = VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 1 * 64 * 32 * 32);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 143360);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 143360);

    const SmallVector<int64_t> newShape({1, 64, 32, 16});
    const SmallVector<int64_t> newSETableShape({1, 1, 32, 16});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShape);
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getData()).getShape(), ShapeRef(newShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getSparsityMap()).getShape(), ShapeRef(newShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));
    const auto sparseChangedElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedElemType);
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getSparsityMap()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getStorageElementTable()).getElementType()));

    const SmallVector<int64_t> newShape2({1, 32, 32, 32});
    const SmallVector<int64_t> newSETableShape2({1, 1, 32, 32});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(changedShapeElemType.getElementType()));
    const auto sparseChangedShapeElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShapeElemType);
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getShape(), ShapeRef(newShape2));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getShape(),
              ShapeRef(newShape2));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getStorageElementTable()).getElementType()));

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedDimsOrder);
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getData()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getSparsityMap()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getStorageElementTable()).getDimsOrder(),
              DimsOrder::NCHW);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getStorageElementTable())
                      .getMemSpace()
                      .getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getStorageElementTable())
                      .getMemSpace()
                      .getLeafName(),
              CMX_NAME);

    const SmallVector<Bit> newStrides({2097152_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({1, 64, 16, 32});
    const SmallVector<int64_t> tileSETableShape({1, 1, 16, 32});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), ShapeRef(tileShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(tileShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 2, 1, 1});
    const SmallVector<int64_t> padAfter({0, 2, 1, 1});
    const SmallVector<int64_t> paddedShape({1, 68, 34, 34});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 34, 34});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(paddedType);
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getData()).getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getSparsityMap()).getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getStorageElementTable()).getShape(),
              ShapeRef(paddedSETableShape));
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_SETable_Interp_BILINEAR) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // Original data shape
    const Shape inputDataShape{1, 64, 5, 10};
    // Interpolated data shape
    const Shape interpolatedShape{1, 64, 11, 21};
    // SET shape
    const Shape seTableShape{1, 1, 11, 21};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);

    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);

    const SmallVector<int64_t> sizes({1, 64, 11, 21});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    // Create data types
    const auto dataType = mlir::RankedTensorType::get(inputDataShape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMapType = mlir::RankedTensorType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto setType = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));

    auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::BILINEAR);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto SEInterpolateAttr =
            VPU::SEInterpolateAttr::get(&ctx, modeAttr, coordTransformModeAttr, scaleAttr,
                                        /*nearestMode=*/nullptr, offsetsAttr, sizesAttr,
                                        /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);
    const auto sparseTensorType =
            VPU::SparseTensorType::get(dataType, sparsityMapType, setType, nullptr, nullptr, SEInterpolateAttr);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // #1 Get methods
    EXPECT_EQ(ndType.getShape(), ShapeRef(interpolatedShape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(interpolatedShape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 1 * 64 * 11 * 21);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({236544_Bit, 3696_Bit, 336_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 9172);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 9172);

    // #2 Change shape
    const SmallVector<int64_t> newShape({1, 64, 17, 13});
    const SmallVector<int64_t> newSMShape({1, 64, 17, 13});
    const SmallVector<int64_t> newSETableShape({1, 1, 17, 13});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShapeByComp = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShapeByComp.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShape);
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getData()).getShape(), Shape({1, 64, 8, 6}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getSparsityMap()).getShape(), ShapeRef(newSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape));

    // #3 Change element type
    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));
    const auto sparseChangedElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedElemType);
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getSparsityMap()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getStorageElementTable()).getElementType()));

    // #4 Change shape and element type
    const SmallVector<int64_t> newShape2({1, 64, 17, 11});
    const SmallVector<int64_t> newSMShape2({1, 64, 17, 11});
    const SmallVector<int64_t> newSETableShape2({1, 1, 17, 11});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(changedShapeElemType.getElementType()));
    const auto sparseChangedShapeElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShapeElemType);
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getShape(), Shape({1, 64, 8, 5}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getShape(),
              ShapeRef(newSMShape2));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getStorageElementTable()).getElementType()));

    // #5 Change dims order
    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedDimsOrder);
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getData()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getSparsityMap()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getStorageElementTable()).getDimsOrder(),
              DimsOrder::NCHW);

    // #6 Change mem space
    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getStorageElementTable())
                      .getMemSpace()
                      .getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getStorageElementTable())
                      .getMemSpace()
                      .getLeafName(),
              CMX_NAME);

    // #7 Change strides
    const SmallVector<Bit> newStrides({2097152_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    // #8 pad
    const SmallVector<int64_t> padBefore({0, 2, 2, 2});
    const SmallVector<int64_t> padAfter({0, 2, 2, 2});
    const SmallVector<int64_t> paddedShape({1, 68, 15, 25});
    const SmallVector<int64_t> paddedSMShape({1, 68, 15, 25});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 15, 25});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(paddedType);
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getData()).getShape(), Shape({1, 68, 7, 12}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getSparsityMap()).getShape(), ShapeRef(paddedSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getStorageElementTable()).getShape(),
              ShapeRef(paddedSETableShape));

    // #9 Extract tile
    // Zero offsets
    SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    SmallVector<int64_t> tileShape({1, 64, 5, 5});
    SmallVector<int64_t> tileSMShape({1, 64, 5, 5});
    SmallVector<int64_t> tileSETableShape({1, 1, 5, 5});
    auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), ShapeRef({1, 64, 3, 3}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(tileSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));
    const auto seAttrUpdated = sparseTiledType.getSeAttr();
    ASSERT_TRUE(seAttrUpdated != nullptr);
    const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
    ASSERT_TRUE(seInterpAttrUpdated != nullptr);
    auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 0, 0});
    EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);

    // Non-zero offsets, offsets are specified in output coordinates
    tileOffsets = SmallVector<int64_t>({0, 0, 3, 3});
    auto tiledTypeOffsets = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledTypeOffsets.getShape(), ShapeRef(tileShape));
    const auto sparseTiledTypeOffsets = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledTypeOffsets);
    ASSERT_TRUE(sparseTiledTypeOffsets != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getData()).getShape(), ShapeRef({1, 64, 3, 3}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getSparsityMap()).getShape(),
              ShapeRef(tileSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));
    const auto seAttrWithOffsets = sparseTiledTypeOffsets.getSeAttr();
    ASSERT_TRUE(seAttrWithOffsets != nullptr);
    const auto seInterpAttrWithOffsets = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrWithOffsets);
    ASSERT_TRUE(seInterpAttrWithOffsets != nullptr);
    outputTileRelOffsets = SmallVector<int64_t>({0, 0, 1, 1});
    EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrWithOffsets.getOffsets()), outputTileRelOffsets);

    SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_SETable_Interp_BILINEAR_Depth2_Tiling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // Original data shape
    const Shape inputDataShape{1, 64, 5, 10};
    // Interpolated data shape
    const Shape interpolatedShape{1, 64, 11, 21};
    // SET shape
    const Shape seTableShape{1, 2, 11, 21};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);

    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);

    const SmallVector<int64_t> sizes({1, 64, 11, 21});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    // Create data types
    const auto dataType = mlir::RankedTensorType::get(inputDataShape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMapType = mlir::RankedTensorType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto setType = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));

    auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::BILINEAR);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto SEInterpolateAttr =
            VPU::SEInterpolateAttr::get(&ctx, modeAttr, coordTransformModeAttr, scaleAttr,
                                        /*nearestMode=*/nullptr, offsetsAttr, sizesAttr,
                                        /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);
    const auto sparseTensorType =
            VPU::SparseTensorType::get(dataType, sparsityMapType, setType, nullptr, nullptr, SEInterpolateAttr);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // Change shape
    const SmallVector<int64_t> newShape({1, 64, 17, 13});
    const SmallVector<int64_t> newSMShape({1, 64, 17, 13});
    const SmallVector<int64_t> newSETableShape({1, 2, 17, 13});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShapeByComp = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShapeByComp.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShape);
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getData()).getShape(), Shape({1, 64, 8, 6}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getSparsityMap()).getShape(), ShapeRef(newSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape));

    // Extract tile
    {  // Zero offsets - on spatial dims
        SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
        SmallVector<int64_t> tileShape({1, 64, 5, 5});
        SmallVector<int64_t> tileSMShape({1, 64, 5, 5});
        SmallVector<int64_t> tileSETableShape({1, 2, 5, 5});
        auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
        EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
        const auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
        ASSERT_TRUE(sparseTiledType != nullptr);
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), ShapeRef({1, 64, 3, 3}));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(),
                  ShapeRef(tileSMShape));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
                  ShapeRef(tileSETableShape));
        const auto seAttrUpdated = sparseTiledType.getSeAttr();
        ASSERT_TRUE(seAttrUpdated != nullptr);
        const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
        ASSERT_TRUE(seInterpAttrUpdated != nullptr);
        auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 0, 0});
        EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);

        // Non-zero offsets, offsets are specified in output coordinates - on spatial dims
        tileOffsets = SmallVector<int64_t>({0, 0, 3, 3});
        auto tiledTypeOffsets = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
        EXPECT_EQ(tiledTypeOffsets.getShape(), ShapeRef(tileShape));
        const auto sparseTiledTypeOffsets = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledTypeOffsets);
        ASSERT_TRUE(sparseTiledTypeOffsets != nullptr);
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getData()).getShape(),
                  ShapeRef({1, 64, 3, 3}));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getSparsityMap()).getShape(),
                  ShapeRef(tileSMShape));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getStorageElementTable()).getShape(),
                  ShapeRef(tileSETableShape));
        const auto seAttrWithOffsets = sparseTiledTypeOffsets.getSeAttr();
        ASSERT_TRUE(seAttrWithOffsets != nullptr);
        const auto seInterpAttrWithOffsets = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrWithOffsets);
        ASSERT_TRUE(seInterpAttrWithOffsets != nullptr);
        outputTileRelOffsets = SmallVector<int64_t>({0, 0, 1, 1});
        EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrWithOffsets.getOffsets()), outputTileRelOffsets);

        SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));
        EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    }

    {  // Zero offsets - on channels dims
        SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
        SmallVector<int64_t> tileShape({1, 32, 11, 21});
        SmallVector<int64_t> tileSMShape({1, 32, 11, 21});
        SmallVector<int64_t> tileSETableShape({1, 1, 11, 21});
        auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
        EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
        const auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
        ASSERT_TRUE(sparseTiledType != nullptr);
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), ShapeRef({1, 32, 5, 10}));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(),
                  ShapeRef(tileSMShape));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
                  ShapeRef(tileSETableShape));
        const auto seAttrUpdated = sparseTiledType.getSeAttr();
        ASSERT_TRUE(seAttrUpdated != nullptr);
        const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
        ASSERT_TRUE(seInterpAttrUpdated != nullptr);
        auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 0, 0});
        EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);

        // Non-zero offsets, offsets are specified in output coordinates - on channels dims
        tileOffsets = SmallVector<int64_t>({0, 32, 0, 0});
        auto tiledTypeOffsets = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
        EXPECT_EQ(tiledTypeOffsets.getShape(), ShapeRef(tileShape));
        const auto sparseTiledTypeOffsets = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledTypeOffsets);
        ASSERT_TRUE(sparseTiledTypeOffsets != nullptr);
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getData()).getShape(),
                  ShapeRef({1, 32, 5, 10}));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getSparsityMap()).getShape(),
                  ShapeRef(tileSMShape));
        EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledTypeOffsets.getStorageElementTable()).getShape(),
                  ShapeRef(tileSETableShape));
        const auto seAttrWithOffsets = sparseTiledTypeOffsets.getSeAttr();
        ASSERT_TRUE(seAttrWithOffsets != nullptr);
        const auto seInterpAttrWithOffsets = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrWithOffsets);
        ASSERT_TRUE(seInterpAttrWithOffsets != nullptr);
        outputTileRelOffsets = SmallVector<int64_t>({0, 0, 0, 0});
        EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrWithOffsets.getOffsets()), outputTileRelOffsets);

        SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));
        EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    }
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_SETable_Interp_NEAREST_PREFER_FLOOR) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // Original data shape
    const Shape shape{1, 64, 5, 10};
    // Interpolated data shape
    const Shape interpolatedShape{1, 64, 10, 20};
    // SET shape
    const Shape seTableShape{1, 1, 10, 20};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);

    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);

    const SmallVector<int64_t> sizes({1, 64, 10, 20});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    // Create data types
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMapType = mlir::RankedTensorType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto setType = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr =
            IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::ROUND_PREFER_FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);
    const auto sparseTensorType =
            VPU::SparseTensorType::get(dataType, sparsityMapType, setType, nullptr, nullptr, SEInterpolateAttr);

    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // #1 Get methods
    EXPECT_EQ(ndType.getShape(), ShapeRef(interpolatedShape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(interpolatedShape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 1 * 64 * 10 * 20);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({204800_Bit, 3200_Bit, 320_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 8800);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 8800);

    // #2 Change shape
    const SmallVector<int64_t> newShape({1, 64, 2, 50});
    const SmallVector<int64_t> newSMShape({1, 64, 2, 50});
    const SmallVector<int64_t> newSETableShape({1, 1, 2, 50});
    auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    auto changedShapeByComp = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShapeByComp.getShape(), ShapeRef(newShape));

    auto sparseChangedShape = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShape);
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getData()).getShape(), Shape({1, 64, 1, 25}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getSparsityMap()).getShape(), ShapeRef(newSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShape.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape));

    // #3 Change element type
    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));
    const auto sparseChangedElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedElemType);
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getSparsityMap()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedElemType.getStorageElementTable()).getElementType()));

    // #4 Change shape and element type
    const SmallVector<int64_t> newShape2({1, 64, 2, 50});
    const SmallVector<int64_t> newSMShape2({1, 64, 2, 50});
    const SmallVector<int64_t> newSETableShape2({1, 1, 2, 50});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(changedShapeElemType.getElementType()));
    const auto sparseChangedShapeElemType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedShapeElemType);
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getShape(),
              Shape({1, 64, 1, 25}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getShape(),
              ShapeRef(newSMShape2));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getStorageElementTable()).getShape(),
              ShapeRef(newSETableShape2));
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getData()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getSparsityMap()).getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(sparseChangedShapeElemType.getStorageElementTable()).getElementType()));

    // #5 Change dims order
    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedDimsOrder);
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getData()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getSparsityMap()).getDimsOrder(), dimsOrder);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedDimsOrder.getStorageElementTable()).getDimsOrder(),
              DimsOrder::NCHW);

    // #6 Change mem space
    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getStorageElementTable())
                      .getMemSpace()
                      .getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = mlir::dyn_cast<vpux::VPU::SparseTensorType>(changedMemSpace);
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getData()).getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getSparsityMap()).getMemSpace().getLeafName(),
              CMX_NAME);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseChangedMemSpace.getStorageElementTable())
                      .getMemSpace()
                      .getLeafName(),
              CMX_NAME);

    // #7 Change strides
    const SmallVector<Bit> newStrides({2097152_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    // #8 pad
    const SmallVector<int64_t> padBefore({0, 2, 2, 2});
    const SmallVector<int64_t> padAfter({0, 2, 2, 2});
    const SmallVector<int64_t> paddedShape({1, 68, 14, 24});
    const SmallVector<int64_t> paddedSMShape({1, 68, 14, 24});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 14, 24});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(paddedType);
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getData()).getShape(), ShapeRef({1, 68, 7, 12}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getSparsityMap()).getShape(), ShapeRef(paddedSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparsePaddedType.getStorageElementTable()).getShape(),
              ShapeRef(paddedSETableShape));

    // #9 Extract tile
    SmallVector<int64_t> tileOffsets({0, 0, 2, 0});
    SmallVector<int64_t> tileShape({1, 64, 4, 4});
    SmallVector<int64_t> tileSMShape({1, 64, 4, 4});
    SmallVector<int64_t> tileSETableShape({1, 1, 4, 4});
    auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), Shape({1, 64, 2, 2}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(tileSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));

    const auto seAttrUpdated = sparseTiledType.getSeAttr();
    ASSERT_TRUE(seAttrUpdated != nullptr);
    const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
    ASSERT_TRUE(seInterpAttrUpdated != nullptr);
    auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 0, 0});
    EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);

    SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_SETable_Interp_NEAREST_FLOOR) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // Original data shape
    const Shape shape{1, 16, 3, 3};
    // Interpolated data shape
    const Shape interpolatedShape{1, 16, 9, 9};
    // SET shape
    const Shape seTableShape{1, 1, 9, 9};

    const SmallVector<float> scale({1, 1, 3, 3});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);

    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);

    const SmallVector<int64_t> sizes({1, 16, 9, 9});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    // Create data types
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMapType = mlir::RankedTensorType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto setType = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);
    const auto sparseTensorType =
            VPU::SparseTensorType::get(dataType, sparsityMapType, setType, nullptr, nullptr, SEInterpolateAttr);

    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // Extract tile
    SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    SmallVector<int64_t> tileShape({1, 16, 6, 9});
    SmallVector<int64_t> tileSMShape({1, 16, 6, 9});
    SmallVector<int64_t> tileSETableShape({1, 1, 6, 9});
    auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), Shape({1, 16, 2, 3}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(tileSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));

    const auto seAttrUpdated = sparseTiledType.getSeAttr();
    ASSERT_TRUE(seAttrUpdated != nullptr);
    const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
    ASSERT_TRUE(seInterpAttrUpdated != nullptr);
    auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 0, 0});
    EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_SETable_Interp_BILINEAR_OFFSETS) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // Original data shape
    const Shape shape{1, 16, 3, 3};
    // Interpolated data shape
    const Shape interpolatedShape{1, 16, 7, 7};
    // SET shape
    const Shape seTableShape{1, 1, 7, 7};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);

    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);

    const SmallVector<int64_t> sizes({1, 16, 7, 7});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    // Create data types
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMapType = mlir::RankedTensorType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto setType = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::BILINEAR);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto SEInterpolateAttr =
            VPU::SEInterpolateAttr::get(&ctx, modeAttr, coordTransformModeAttr, scaleAttr,
                                        /*nearestModeAttr=*/nullptr, offsetsAttr, sizesAttr,
                                        /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);
    const auto sparseTensorType =
            VPU::SparseTensorType::get(dataType, sparsityMapType, setType, nullptr, nullptr, SEInterpolateAttr);

    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // Extract tile
    SmallVector<int64_t> tileOffsets({0, 0, 6, 6});
    SmallVector<int64_t> tileShape({1, 16, 1, 1});
    SmallVector<int64_t> tileSMShape({1, 16, 1, 1});
    SmallVector<int64_t> tileSETableShape({1, 1, 1, 1});
    auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), Shape({1, 16, 1, 1}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(tileSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));

    const auto seAttrUpdated = sparseTiledType.getSeAttr();
    ASSERT_TRUE(seAttrUpdated != nullptr);
    const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
    ASSERT_TRUE(seInterpAttrUpdated != nullptr);
    auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 2, 2});
    EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_SETable_Interp_NEAREST_PREFER_FLOOR_OFFSETS) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // Original data shape
    const Shape shape{1, 16, 3, 3};
    // Interpolated data shape
    const Shape interpolatedShape{1, 16, 9, 9};
    // SET shape
    const Shape seTableShape{1, 1, 9, 9};

    const SmallVector<float> scale({1, 1, 3, 3});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);

    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);

    const SmallVector<int64_t> sizes({1, 16, 9, 9});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    // Create data types
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMapType = mlir::RankedTensorType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto setType = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);
    const auto sparseTensorType =
            VPU::SparseTensorType::get(dataType, sparsityMapType, setType, nullptr, nullptr, SEInterpolateAttr);

    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(sparseTensorType);
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    // Extract tile
    SmallVector<int64_t> tileOffsets({0, 0, 5, 5});
    SmallVector<int64_t> tileShape({1, 16, 2, 2});
    SmallVector<int64_t> tileSMShape({1, 16, 2, 2});
    SmallVector<int64_t> tileSETableShape({1, 1, 2, 2});
    auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    auto sparseTiledType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tiledType);
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getData()).getShape(), Shape({1, 16, 2, 2}));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getSparsityMap()).getShape(), ShapeRef(tileSMShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(sparseTiledType.getStorageElementTable()).getShape(),
              ShapeRef(tileSETableShape));

    const auto seAttrUpdated = sparseTiledType.getSeAttr();
    ASSERT_TRUE(seAttrUpdated != nullptr);
    const auto seInterpAttrUpdated = mlir::dyn_cast<vpux::VPU::SEInterpolateAttr>(seAttrUpdated);
    ASSERT_TRUE(seInterpAttrUpdated != nullptr);
    auto outputTileRelOffsets = SmallVector<int64_t>({0, 0, 2, 2});
    EXPECT_EQ(parseIntArrayAttr<int64_t>(seInterpAttrUpdated.getOffsets()), outputTileRelOffsets);
}

TEST_F(MLIR_NDTypeInterface, SparseTensorType_Extract_DenseTile) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const Shape shape{1, 16, 32, 32};
    const Shape sparsityMapShape{1, 16, 64, 64};
    const Shape seTableShape{1, 1, 64, 64};
    const auto data = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMap = mlir::RankedTensorType::get(sparsityMapShape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto seTable = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));
    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto sparsityCompression = VPU::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                       numElemsAttr, getIntAttr(&ctx, alignment));
    const auto sparseTensorType = VPU::SparseTensorType::get(data, sparsityMap, seTable, nullptr, sparsityCompression);

    const auto effectiveSparseOutputType =
            mlir::cast<NDTypeInterface>(VPU::getEffectiveSparseOutputType(sparseTensorType));

    const Shape tileOffset{0, 0, 16, 16};
    const Shape tileShape{1, 16, 16, 16};

    auto tiledSparseShape = sparseTensorType.extractDenseTile(tileOffset, tileShape).getShape();
    auto tiledEffectiveSparseShape = effectiveSparseOutputType.extractDenseTile(tileOffset, tileShape).getShape();
    EXPECT_EQ(tiledSparseShape, tiledEffectiveSparseShape);
}
