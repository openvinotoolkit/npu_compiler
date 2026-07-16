//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "common/utils.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_SubByteTest : public MLIR_UnitBase {
public:
    mlir::MLIRContext ctx;

public:
    MLIR_SubByteTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }
};

TEST_F(MLIR_SubByteTest, SubViewI1) {
    const auto baseType = mlir::RankedTensorType::get({1, 7, 5}, getInt1Type(&ctx));

    // 35 elements -> ceil(35/8) = 5 bytes (partial last byte)
    const std::vector<char> inputVals = {
            static_cast<char>(0xAA),  // elements 0-7
            static_cast<char>(0xAA),  // elements 8-15
            static_cast<char>(0xAA),  // elements 16-23
            static_cast<char>(0xAA),  // elements 24-31
            0x02                      // elements 32-34 (3 valid bits)
    };

    auto offsetArr = std::vector<int64_t>{0, 1, 0};
    auto shapeArr = std::vector<int64_t>{1, 1, 1};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    // Element at (0,1,0) has linear index 5 -> value = 5 % 2 = 1
    const std::vector<char> expectedVal = {0x01};
    const auto contentVals = content.getRawStorageBuf();
    EXPECT_EQ(contentVals.size(), expectedVal.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVal[i]);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_1D_Splat) {
    const auto baseType = mlir::RankedTensorType::get({8}, getUInt4Type(&ctx));
    const char splatVal = 0x66;
    const std::vector<char> inputVals = {splatVal};

    auto offsetArr = std::vector<int64_t>{3};
    auto shapeArr = std::vector<int64_t>{7};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const auto contentVals = content.getRawStorageBuf();
    // In the case of a splat sub byte type tensor, we keep the first byte.
    // Therefore a 0x0f is needed to get the first element
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i] & 0x0f, 0x06);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_1D) {
    const auto baseType = mlir::RankedTensorType::get({8}, getUInt4Type(&ctx));
    const std::vector<char> inputVals = {0x10, 0x32, 0x54, 0x76};

    auto offsetArr = std::vector<int64_t>{3};
    auto shapeArr = std::vector<int64_t>{4};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const std::vector<char> expectedVal = {0x43, 0x65};
    const auto contentVals = content.getRawStorageBuf();
    EXPECT_EQ(contentVals.size(), expectedVal.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVal[i]);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_2D_Splat) {
    const auto baseType = mlir::RankedTensorType::get({2, 4}, getUInt4Type(&ctx));
    const char splatVal = 0x66;
    const std::vector<char> inputVals = {splatVal};

    auto offsetArr = std::vector<int64_t>{1, 2};
    auto shapeArr = std::vector<int64_t>{1, 2};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const auto contentVals = content.getRawStorageBuf();
    // In the case of a splat sub byte type tensor, we keep the first byte.
    // Therefore a 0x0f is needed to get the first element
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i] & 0x0f, 0x06);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_2D) {
    const auto baseType = mlir::RankedTensorType::get({2, 4}, getUInt4Type(&ctx));
    const std::vector<char> inputVals = {0x66, 0x7f, 0x10, 0x21};

    auto offsetArr = std::vector<int64_t>{1, 2};
    auto shapeArr = std::vector<int64_t>{1, 2};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const std::vector<char> expectedVal = {0x21};
    const auto contentVals = content.getRawStorageBuf();
    EXPECT_EQ(contentVals.size(), expectedVal.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVal[i]);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_3D_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 4}, getUInt4Type(&ctx));
    const char splatVal = 0x66;
    const std::vector<char> inputVals = {splatVal};

    auto offsetArr = std::vector<int64_t>{0, 1, 2};
    auto shapeArr = std::vector<int64_t>{1, 1, 2};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const auto contentVals = content.getRawStorageBuf();
    // In the case of a splat sub byte type tensor, we keep the first byte.
    // Therefore a 0x0f is needed to get the first element
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i] & 0x0f, 0x06);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_3D) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 4}, getUInt4Type(&ctx));
    const std::vector<char> inputVals = {0x66, 0x7f, 0x10, 0x21};

    auto offsetArr = std::vector<int64_t>{0, 1, 2};
    auto shapeArr = std::vector<int64_t>{1, 1, 2};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const std::vector<char> expectedVal = {0x21};
    const auto contentVals = content.getRawStorageBuf();
    EXPECT_EQ(contentVals.size(), expectedVal.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVal[i]);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_4D_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, getUInt4Type(&ctx));
    const char splatVal = 0x66;
    const std::vector<char> inputVals = {splatVal};

    auto offsetArr = std::vector<int64_t>{0, 1, 1, 1};
    auto shapeArr = std::vector<int64_t>{1, 1, 2, 3};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const auto contentVals = content.getRawStorageBuf();
    // In the case of a splat sub byte type tensor, we keep the first byte.
    // Therefore a 0x0f is needed to get the first element
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i] & 0x0f, 0x06);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_4D) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, getUInt4Type(&ctx));
    const std::vector<char> inputVals = {0x10, 0x32, 0x54, 0x76, 0x67, 0x45, 0x23, 0x01, 0x10, 0x32, 0x54, 0x76};

    auto offsetArr = std::vector<int64_t>{0, 1, 1, 1};
    auto shapeArr = std::vector<int64_t>{1, 1, 2, 3};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const std::vector<char> expectedVal = {0x21, 0x53, 0x76};
    const auto contentVals = content.getRawStorageBuf();
    EXPECT_EQ(contentVals.size(), expectedVal.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVal[i]);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_Generic_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4, 1}, getUInt4Type(&ctx));
    const char splatVal = 0x66;
    const std::vector<char> inputVals = {splatVal};

    auto offsetArr = std::vector<int64_t>{0, 0, 1, 2, 0};
    auto shapeArr = std::vector<int64_t>{1, 1, 1, 2, 1};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const auto contentVals = content.getRawStorageBuf();
    // In the case of a splat sub byte type tensor, we keep the first byte.
    // Therefore a 0x0f is needed to get the first element
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i] & 0x0f, 0x06);
    }
}

TEST_F(MLIR_SubByteTest, SubViewI4_General) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4, 1}, getUInt4Type(&ctx));
    const std::vector<char> inputVals = {0x66, 0x7f, 0x10, 0x21};

    auto offsetArr = std::vector<int64_t>{0, 0, 1, 2, 0};
    auto shapeArr = std::vector<int64_t>{1, 1, 1, 2, 1};

    const auto baseAttr = Const::createExternalConstContent(baseType, ArrayRef(inputVals), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);

    const auto offset = Shape(to_small_vector(offsetArr));
    const auto shape = Shape(to_small_vector(shapeArr));
    contentSetup = contentSetup.subview(offset, shape);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());

    auto contentShape = content.getType().getShape();
    auto contentShapeArr = contentShape.raw();
    EXPECT_EQ(shapeArr.size(), contentShapeArr.size());
    for (size_t i = 0; i < shapeArr.size(); i++) {
        EXPECT_EQ(shapeArr[i], contentShapeArr[i]);
    }

    const std::vector<char> expectedVal = {0x21};
    const auto contentVals = content.getRawStorageBuf();
    EXPECT_EQ(contentVals.size(), expectedVal.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVal[i]);
    }
}
