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

namespace {
template <typename T>
ArrayRef<char> convertArrayRef(ArrayRef<T> typed) {
    return ArrayRef<char>(reinterpret_cast<const char*>(typed.data()), typed.size() * sizeof(T));
}
}  // namespace

TEST_F(MLIR_SubByteTest, ConvertElemTypeI4ToI8_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getInt4Type(&ctx));
    const unsigned char splatVal = 0xdd;
    const std::vector<unsigned char> inputVals = {splatVal};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSignless() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<unsigned char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    EXPECT_EQ(content.getSplatValue<unsigned char>(), 0x0d);
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], 0x0d);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeI4ToI8) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getInt4Type(&ctx));
    std::vector<unsigned char> inputVals{0x66, 0x85, 0x2f, 0xde};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSignless() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<unsigned char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    std::vector<unsigned char> expectedVals{0x06, 0x06, 0x05, 0x08, 0x0f, 0x02, 0x0e, 0x0d};
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeU4ToU8_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getUInt4Type(&ctx));
    const unsigned char splatVal = 0xdd;
    const std::vector<unsigned char> inputVals = {splatVal};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getUInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isUnsigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<unsigned char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    EXPECT_EQ(content.getSplatValue<unsigned char>(), 0x0d);
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], 0x0d);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeU4ToU8) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getUInt4Type(&ctx));
    std::vector<unsigned char> inputVals{0x66, 0x85, 0x2f, 0xde};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getUInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isUnsigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<unsigned char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    std::vector<unsigned char> expectedVals{0x06, 0x06, 0x05, 0x08, 0x0f, 0x02, 0x0e, 0x0d};
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeSI4ToSI8_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getSInt4Type(&ctx));
    const unsigned char splatVal = 0xdd;
    const std::vector<unsigned char> inputVals = {splatVal};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    EXPECT_EQ(content.getSplatValue<char>(), static_cast<char>(0xfd));
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<char>(0xfd));
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeSI4ToSI8) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getSInt4Type(&ctx));
    std::vector<unsigned char> inputVals{0x66, 0x85, 0x2f, 0xde};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    std::vector<unsigned char> expectedVals{0x06, 0x06, 0x05, 0xf8, 0xff, 0x02, 0xfe, 0xfd};
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<char>(expectedVals[i]));
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeI1ToSI8_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 8}, getInt1Type(&ctx));
    const unsigned char splatVal = 0xff;
    const std::vector<unsigned char> inputVals = {splatVal};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(content.getSplatValue<char>(), 0x01);
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], 0x01);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeI1ToSI8) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 8}, getInt1Type(&ctx));
    std::vector<unsigned char> inputVals{0x0d, 0x82};

    const auto baseAttr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef(inputVals)), "testConst");

    auto contentSetup = Const::ContentSetup(baseAttr, baseType);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentSetup));
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    std::vector<char> expectedVals{0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                                   0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}
