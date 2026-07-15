//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include "common/utils.hpp"

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/Dialect/Quant/IR/Quant.h>

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>
#include <vpux/compiler/utils/quantization.hpp>

#include <cassert>
#include <memory>

using namespace vpux;

namespace {
template <typename T>
std::vector<T> generateValues(size_t n) {
    std::vector<T> vals(n);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<T>(i);
    }

    return vals;
}

template <typename T>
std::unique_ptr<T[]> generateValuesPointer(size_t n) {
    std::unique_ptr<T[]> data = std::make_unique<T[]>(n);
    T* vals = data.get();
    for (size_t i = 0; i < n; ++i) {
        vals[i] = static_cast<T>(i);
    }

    return data;
}

template <typename T>
Const::ContentAttr getContentAttr(
        mlir::RankedTensorType rankedType,
        FuncRef<Const::ContentSetup(Const::ContentSetup&)> transform = [](Const::ContentSetup& setup) {
            return std::move(setup);
        }) {
    auto data = generateValues<T>(rankedType.getNumElements());
    auto baseAttr = Const::createConstContent(rankedType, ArrayRef(data));

    Const::ContentSetup setup(baseAttr, baseAttr.getType());
    setup = transform(setup);

    return Const::ContentAttr::get(baseAttr, setup);
}

template <typename T>
void checkPaddedBuffer(const Const::Content& actual, const std::vector<T>& expVals, ShapeRef buf, ShapeRef pad, T zp,
                       size_t actOffset = 0, size_t originOffset = 0) {
    const int64_t IC = buf[Dim(0)];
    const int64_t IH = buf[Dim(1)];
    const int64_t IW = buf[Dim(2)];

    const int64_t PC = pad[Dim(0)];
    const int64_t PH = pad[Dim(1)];
    const int64_t PW = pad[Dim(2)];

    const auto actVals = actual.getValues<T>();
    for (int64_t c = 0; c < IC + 2 * PC; ++c) {
        for (int64_t h = 0; h < IH + 2 * PH; ++h) {
            for (int64_t w = 0; w < IW + 2 * PW; ++w) {
                const auto newIndex = w + h * (IW + 2 * PW) + c * (IW + 2 * PW) * (IH + 2 * PH) + actOffset;
                if (c < PC || c >= IC + PC || h < PH || h >= IH + PH || w < PW || w >= IW + PW) {
                    EXPECT_EQ(zp, actVals[newIndex]) << c << " " << h << " " << w;
                } else {
                    const auto origIndex = (w - PW) + (h - PH) * IW + (c - PC) * IW * IH + originOffset;
                    EXPECT_EQ(expVals[origIndex], actVals[newIndex]) << c << " " << h << " " << w;
                }
            }
        }
    }
}

mlir::ArrayAttr getArrayAttr(mlir::MLIRContext* ctx, std::initializer_list<int64_t> list) {
    const auto int64Type = mlir::IntegerType::get(ctx, 64);
    auto array = to_small_vector(list | transformed([int64Type](int64_t value) -> mlir::Attribute {
                                     return mlir::IntegerAttr::get(int64Type, value);
                                 }));
    return mlir::ArrayAttr::get(ctx, array);
}

mlir::ArrayAttr getArrayOfArrayAttr(mlir::MLIRContext* ctx,
                                    std::initializer_list<std::initializer_list<int64_t>> listOfLists) {
    SmallVector<mlir::Attribute> elements;
    for (auto list : listOfLists) {
        elements.push_back(getArrayAttr(ctx, list));
    }
    return mlir::ArrayAttr::get(ctx, elements);
}

template <typename T>
Const::ContentAttr getContentAttr(ArrayRef<int64_t> shape, mlir::Type elemType, ArrayRef<T> data) {
    auto rankedType = mlir::RankedTensorType::get(shape, elemType);
    auto baseAttr = Const::createConstContent(rankedType, data);
    return Const::ContentAttr::get(baseAttr);
}
}  // namespace

class MLIR_ConstContentAttrTest : public MLIR_UnitBase {
public:
    mlir::MLIRContext ctx;

public:
    MLIR_ConstContentAttrTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
        ctx.loadDialect<vpux::type::QuantileDialect>();
    }
};

TEST_F(MLIR_ConstContentAttrTest, FromDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    ASSERT_NE(static_cast<const void*>(baseAttr.getRawData().data()), static_cast<const void*>(vals.data()))
            << "Local data has to be copied inside DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }

    std::vector<float> readVals;
    content.read([&](auto values) {
        for (auto x : values) {
            readVals.push_back(static_cast<float>(x));
        }
    });
    EXPECT_EQ(readVals, vals);
}

TEST_F(MLIR_ConstContentAttrTest, FromIRParsing) {
    const auto inputIR = R"(
        module @test {
            func.func @main() -> tensor<2x3x1x1xf32> {
                %cst = const.Declare tensor<2x3x1x1xf32> = dense_resource<splat_blob> : tensor<2x3x1x1xf32> isSplat

                return %cst : tensor<2x3x1x1xf32>
            }
        }

        {-#
        dialect_resources: {
            builtin: {
            splat_blob: "0x0400000001000000"
            }
        }
        #-}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](Const::DeclareOp declareOp) {
        auto contentAttr = declareOp.getContentAttr();
        EXPECT_TRUE(contentAttr.isSplat());
    });
}

// Note: some networks (in tests at least) provide 0-element constants
TEST_F(MLIR_ConstContentAttrTest, FromEmptyDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({0}, mlir::Float32Type::get(&ctx));

    const std::vector<char> empty{};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(empty.data(), empty.size()));
    ASSERT_TRUE(baseAttr.empty());

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    ASSERT_TRUE(content.getValues<float>().empty());
    content.read([](auto values) {
        ASSERT_TRUE(values.empty());
    });
}

TEST_F(MLIR_ConstContentAttrTest, FromSplatDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }

    std::vector<float> readVals;
    content.read([&](auto values) {
        for (auto x : values) {
            readVals.push_back(static_cast<float>(x));
        }
    });
    EXPECT_EQ(readVals.size(), 1);
    EXPECT_EQ(readVals[0], splatVal);
}

namespace {
template <typename T>
ArrayRef<char> convertArrayRef(ArrayRef<T> typed) {
    return ArrayRef<char>(reinterpret_cast<const char*>(typed.data()), typed.size() * sizeof(T));
}
}  // namespace

TEST_F(MLIR_ConstContentAttrTest, FromDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    std::unique_ptr<float[]> data = generateValuesPointer<float>(baseType.getNumElements());

    const auto baseAttr = Const::createExternalConstContent(
            baseType, convertArrayRef(ArrayRef<float>(data.get(), baseType.getNumElements())),
            "FromDenseResourceElementsAttr");

    float* dataPtr = data.get();
    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(dataPtr))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], dataPtr[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromDenseResourceElementsAttrNonOwning) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                            "FromDenseResourceElementsAttrNonOwning");

    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(vals.data()))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromSplatDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const float splatVal = 4.0f;
    const std::vector<float> vals = {splatVal};
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                            "FromSplatDenseResourceElementsAttr");

    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(vals.data()))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

// Note: some networks (in tests at least) provide 0-element constants
TEST_F(MLIR_ConstContentAttrTest, FromEmptyDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({0}, mlir::Float32Type::get(&ctx));
    const std::vector<float> empty{};
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(empty)),
                                                            "FromEmptyDenseResourceElementsAttr");

    ASSERT_TRUE(baseAttr.empty());

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    ASSERT_TRUE(content.getValues<float>().empty());
}

TEST_F(MLIR_ConstContentAttrTest, FromEqualElementsSplatDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const float splatVal = -4.0f;
    const std::vector<float> vals(baseType.getNumElements(), splatVal);
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                            "FromEqualElementsSplatDenseResourceElementsAttr");

    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(vals.data()))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ExpositionOnlyDenseResourceDuplicate) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());

    const auto attr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                        "ExpositionOnlyDenseResourceDuplicate");
    const auto attrDuplicate = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                                 "ExpositionOnlyDenseResourceDuplicate");

    static_assert(!std::is_copy_constructible_v<mlir::AsmResourceBlob> &&
                  !std::is_copy_assignable_v<mlir::AsmResourceBlob>);

    ASSERT_NE(attrDuplicate.getRawHandle().getKey(), attr.getRawHandle().getKey())
            << "Two separately created DenseResourceElementsAttr objects could not share the same key - otherwise, "
               "revise nGraph constant sharing in NGraphImporter::parseNode()";

    const auto attrCopy = attr;
    ASSERT_EQ(attrCopy.getRawHandle().getKey(), attr.getRawHandle().getKey());
}

TEST_F(MLIR_ConstContentAttrTest, ExpositionOnlyDenseResourceDuplicate_AlternativeBehaviour) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());

    Const::ExternalConstContentCreationOptions options;
    options.allowDuplicatesForTheSameResourceName = true;

    const auto attr =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                              "ExpositionOnlyDenseResourceDuplicate_AlternativeBehaviour", options);
    const auto attrDuplicate =
            Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                              "ExpositionOnlyDenseResourceDuplicate_AlternativeBehaviour", options);

    static_assert(!std::is_copy_constructible_v<mlir::AsmResourceBlob> &&
                  !std::is_copy_assignable_v<mlir::AsmResourceBlob>);

    ASSERT_EQ(attrDuplicate.getRawHandle().getKey(), attr.getRawHandle().getKey())
            << "Two separately created DenseResourceElementsAttr objects can share the same key as this is explicitly "
               "allowed";
    ASSERT_EQ(attrDuplicate.getRawHandle().getBlob(), attr.getRawHandle().getBlob())
            << "Two separately created DenseResourceElementsAttr objects must share the same blob as this is "
               "explicitly allowed";
    ASSERT_EQ(attrDuplicate, attr) << "MLIR uniquifier would re-use already-created dense_resource<>";

    const auto baseType2 = mlir::RankedTensorType::get({2, 2, 3, 4}, mlir::Float16Type::get(&ctx));
    const auto attrDuplicateWithAnotherType =
            Const::createExternalConstContent(baseType2, convertArrayRef(ArrayRef<float>(vals)),
                                              "ExpositionOnlyDenseResourceDuplicate_AlternativeBehaviour", options);

    ASSERT_EQ(attrDuplicateWithAnotherType.getRawHandle().getKey(), attr.getRawHandle().getKey())
            << "Two separately created DenseResourceElementsAttr objects can share the same key as this is explicitly "
               "allowed";
    ASSERT_EQ(attrDuplicateWithAnotherType.getRawHandle().getBlob(), attr.getRawHandle().getBlob())
            << "Two separately created DenseResourceElementsAttr objects must share the same blob as this is "
               "explicitly allowed";
    ASSERT_NE(attrDuplicateWithAnotherType, attr) << "MLIR uniquifier cannot re-use already-created dense_resource<>";
}

TEST_F(MLIR_ConstContentAttrTest, DuplicateDenseResources_SplatAndNonSplat) {
    const auto baseTypeSingleInt64 = mlir::RankedTensorType::get({1}, getSInt64Type(&ctx));
    const auto baseTypeTwoInt32 = mlir::RankedTensorType::get({2}, getSInt32Type(&ctx));
    const std::vector<int32_t> vals = {10, 42};  // Note: two *distinct* values

    Const::ExternalConstContentCreationOptions options;
    options.allowDuplicatesForTheSameResourceName = true;

    const auto attr = Const::createExternalConstContent(baseTypeSingleInt64, convertArrayRef(ArrayRef(vals)),
                                                        "DuplicateDenseResources_SplatAndNonSplat", options);
    const auto attrDuplicate = Const::createExternalConstContent(baseTypeTwoInt32, convertArrayRef(ArrayRef(vals)),
                                                                 "DuplicateDenseResources_SplatAndNonSplat", options);

    ASSERT_NE(attrDuplicate, attr) << "Attributes differ, because they have different types";
    ASSERT_EQ(attrDuplicate.getRawHandle().getKey(), attr.getRawHandle().getKey());
    ASSERT_EQ(attrDuplicate.getRawHandle().getBlob(), attr.getRawHandle().getBlob());

    const auto contentAttr = Const::ContentAttr::get(attr);
    const auto content = contentAttr.fold();
    ASSERT_TRUE(content.isSplat()) << "Single int64 is a splat";
    ASSERT_EQ(*reinterpret_cast<const int64_t*>(vals.data()), content.getSplatValue<int64_t>())
            << "Single splat value should match the array of int32 byte-wise";

    const auto contentAttrDuplicate = Const::ContentAttr::get(attrDuplicate);
    const auto contentDuplicate = contentAttrDuplicate.fold();
    ASSERT_FALSE(contentDuplicate.isSplat()) << "Two *distinct* int32 values - not a splat";
    ASSERT_EQ(vals, to_std_vector(contentDuplicate.getValues<int32_t>())) << "Two int32 values should match the array";
}

TEST_F(MLIR_ConstContentAttrTest, DuplicateDenseResources_SplatAndSplat) {
    const auto baseTypeSingleInt64 = mlir::RankedTensorType::get({1}, getSInt64Type(&ctx));
    const auto baseTypeTwoInt32 = mlir::RankedTensorType::get({2}, getSInt32Type(&ctx));
    const std::vector<int32_t> vals = {10, 10};  // Note: two *identical* values

    Const::ExternalConstContentCreationOptions options;
    options.allowDuplicatesForTheSameResourceName = true;

    const auto attr = Const::createExternalConstContent(baseTypeSingleInt64, convertArrayRef(ArrayRef(vals)),
                                                        "DuplicateDenseResources_SplatAndNonSplat", options);
    const auto attrDuplicate = Const::createExternalConstContent(baseTypeTwoInt32, convertArrayRef(ArrayRef(vals)),
                                                                 "DuplicateDenseResources_SplatAndNonSplat", options);

    ASSERT_NE(attrDuplicate, attr) << "Attributes differ, because they have different types";
    ASSERT_EQ(attrDuplicate.getRawHandle().getKey(), attr.getRawHandle().getKey());
    ASSERT_EQ(attrDuplicate.getRawHandle().getBlob(), attr.getRawHandle().getBlob());

    const auto contentAttr = Const::ContentAttr::get(attr);
    const auto content = contentAttr.fold();
    ASSERT_TRUE(content.isSplat()) << "Single int64 is a splat";
    ASSERT_EQ(*reinterpret_cast<const int64_t*>(vals.data()), content.getSplatValue<int64_t>())
            << "Single splat value should match the array of int32 byte-wise";

    const auto contentAttrDuplicate = Const::ContentAttr::get(attrDuplicate);
    const auto contentDuplicate = contentAttrDuplicate.fold();
    ASSERT_TRUE(contentDuplicate.isSplat()) << "Two *identical* int32 values - splat";
    ASSERT_EQ(vals, to_std_vector(contentDuplicate.getValues<int32_t>())) << "Two int32 values should match the array";
}

TEST_F(MLIR_ConstContentAttrTest, ReadAndConvertInCpp) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, getSInt64Type(&ctx));

    const auto vals = generateValues<int64_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    ASSERT_NE(static_cast<const void*>(baseAttr.getRawData().data()), static_cast<const void*>(vals.data()))
            << "Local data has to be copied inside DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    std::vector<int16_t> convertedVals;
    content.read(getSInt16Type(&ctx), [&](auto values, auto dummy) {
        if constexpr (!std::is_same_v<ArrayRef<int64_t>, decltype(values)>) {
            GTEST_FAIL() << "Wrong type dispatch on internal type";
        }

        using cast_type = decltype(dummy);
        if constexpr (!std::is_same_v<int16_t, cast_type>) {
            GTEST_FAIL() << "Wrong type dispatch on other type";
        }

        std::transform(values.begin(), values.end(), std::back_inserter(convertedVals), [](int64_t value) -> cast_type {
            return static_cast<cast_type>(value);
        });
    });
    const auto expected = generateValues<int16_t>(baseType.getNumElements());
    EXPECT_EQ(expected, convertedVals);
}

TEST_F(MLIR_ConstContentAttrTest, ConvertStorageElemType) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_FP32) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    const auto bufSize = bufSizeBytes / sizeof(float);
    std::vector<float> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(vals.size(), bufSize);
    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(vals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_U8) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::IntegerType::get(&ctx, 8));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(vals.size(), tempBuf.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(vals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_U2) {
    const auto baseType =
            mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned));

    const std::vector<uint8_t> vals = {0,   // 0x0 b'00
                                       1,   // 0x1 b'01
                                       2,   // 0x2 b'10
                                       3};  // 0x3 b'11

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Unsigned));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xE4};  // b'1110_0100
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I4) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1,    // 0x1
                                       7,    // 0x7
                                       10,   // 0xA
                                       15};  // 0xF
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x71, 0xFA};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I2) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed));

    const std::vector<int8_t> vals = {-2,  // 0x2 b'10
                                      -1,  // 0x3 b'11
                                      0,   // 0x0 b'00
                                      1};  // 0x1 b'01

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Signed));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x4E};  // b'0100_1110
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I1) {
    const auto baseType = mlir::RankedTensorType::get({16}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {0, 0, 0, 1,   // 0x1 -> 0x8 packed
                                       0, 0, 1, 1,   // 0x3 -> 0xC packed
                                       1, 1, 1, 1,   // 0xF -> 0xF packed
                                       1, 1, 1, 0};  // 0xE -> 0x7 packed
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xC8, 0x7F};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_FP32) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&ctx));

    const std::vector<float> vals = {1.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    auto contentAttr = Const::ContentAttr::get(baseAttr);
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    const auto bufSize = bufSizeBytes / sizeof(float);
    std::vector<float> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(bufSize, baseType.getNumElements());
    for (size_t i = 0; i < tempBuf.size(); ++i) {
        EXPECT_EQ(tempBuf[i], vals[0]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_U8) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    auto contentAttr = Const::ContentAttr::get(baseAttr);
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(bufSizeBytes, baseType.getNumElements());
    for (size_t i = 0; i < tempBuf.size(); ++i) {
        EXPECT_EQ(tempBuf[i], vals[0]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_U2) {
    const auto baseType =
            mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned));

    const std::vector<uint8_t> vals = {3};  // 0x2 b'11
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Unsigned));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xFF};  // b'1111_1111
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_I4) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xAA, 0xAA};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_I2) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed));

    const std::vector<int8_t> vals = {-2};  // 0x2 b'10
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Signed));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xAA};  // b'1010_1010
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_I1) {
    const auto baseType = mlir::RankedTensorType::get({16}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xFF, 0xFF};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I1_With_Enough_Buffer) {
    const auto baseType = mlir::RankedTensorType::get({16}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {0, 0, 0, 1,   // 0x1 -> 0x8 packed
                                       0, 0, 1, 1,   // 0x3 -> 0xC packed
                                       1, 1, 1, 1,   // 0xF -> 0xF packed
                                       1, 1, 1, 0};  // 0xE -> 0x7 packed
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    // Offer a large enough buffer to copy the data. The data not will be packed.
    int bufSizeBytes = 512;
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> UnpackedVals = {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

    for (size_t i = 0; i < UnpackedVals.size(); ++i) {
        EXPECT_EQ(UnpackedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CastElemType) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(getSInt32Type(&ctx));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CastElemTypeSplat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(getSInt32Type(&ctx));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<int32_t>(), static_cast<int32_t>(splatVal));

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<int32_t>(splatVal));
    }
}

TEST_F(MLIR_ConstContentAttrTest, CastElemTypeSubByte) {
    const auto baseType =
            mlir::RankedTensorType::get({3}, mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<bool>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], (i == 0) ? false : true);
    }
}

struct MLIR_ConstContentAttrTest_Rescale : testing::TestWithParam<std::tuple<std::vector<float>, std::vector<float>>> {
    MLIR_ConstContentAttrTest_Rescale() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

protected:
    mlir::DialectRegistry registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx;
};

TEST_P(MLIR_ConstContentAttrTest_Rescale, Fold) {
    auto lhsVals = std::get<0>(GetParam());
    auto rhsVals = std::get<1>(GetParam());

    const SmallVector<int64_t> commonShape({1, 2, 2, 1});
    auto lhsAttr = getContentAttr<float>(commonShape, mlir::Float32Type::get(&ctx), lhsVals);
    auto rhsAttr = getContentAttr<float>(commonShape, mlir::Float32Type::get(&ctx), rhsVals);

    auto combined = lhsAttr.transform().rescale(rhsAttr).get();
    EXPECT_EQ(combined.isSplat(), lhsAttr.isSplat() && rhsAttr.isSplat());
    EXPECT_EQ(combined.getType(), lhsAttr.getType());
    EXPECT_EQ(combined.getType(), rhsAttr.getType());

    // Note: for simplicity in this test (to ignore splat vs non-splat problem),
    // convert splat data to non-splat by "broadcasting".
    const auto broadcastToTensor = [&](std::vector<float>& data) {
        if (data.size() != 1) {
            return;
        }
        const auto totalSize = std::accumulate(commonShape.begin(), commonShape.end(), 1, std::multiplies<int64_t>{});
        const auto firstElement = data.front();
        data.resize(totalSize, firstElement);
    };
    broadcastToTensor(lhsVals);
    broadcastToTensor(rhsVals);

    auto content = combined.fold();
    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(content.isSplat(), combined.isSplat());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        const float expected = lhsVals[i] * rhsVals[i];
        EXPECT_FLOAT_EQ(contentVals[i], expected)
                << "mismatch at index " << i << ": got " << contentVals[i] << ", expected " << expected;
    }
}

INSTANTIATE_TEST_SUITE_P(
        all, MLIR_ConstContentAttrTest_Rescale,
        testing::Combine(testing::Values(std::vector<float>({1.0}), std::vector<float>({1.0, 2.0, 3.0, 4.0})),
                         testing::Values(std::vector<float>({42.0}), std::vector<float>({42.0, 43.0, 44.0, 45.0}))));

TEST_F(MLIR_ConstContentAttrTest, Add) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto bias = 10;
    const auto vals = generateValues<float>(baseType.getNumElements());
    std::vector<float> expectedVals(vals.size());
    std::transform(vals.begin(), vals.end(), expectedVals.begin(), [&](float item) {
        return item + bias;
    });

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.add(bias);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, QuantCast) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getUInt8Type(&ctx));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, 128, 0, 255);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<uint8_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<uint8_t>(i));
    }
}

TEST_F(MLIR_ConstContentAttrTest, Dequantize) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getSInt8Type(&ctx));

    std::vector<int8_t> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        // -127, 0, 127
        const auto choice = (static_cast<int>(i) % 3) - 1;
        vals[i] = static_cast<int8_t>(choice * 127);
    }

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    const double scale = 2.0 / 254.0;
    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), scale, 0, -127, 127);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(quantType).dequantize();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        const auto choice = (static_cast<int>(i) % 3) - 1;
        EXPECT_FLOAT_EQ(contentVals[i], static_cast<float>(choice));
    }
}

TEST_F(MLIR_ConstContentAttrTest, Reshape) {
    const auto baseType = mlir::RankedTensorType::get({1, 9, 2}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.reshape({1, 3, 3, 2});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ReverseCWise) {
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.reverse(Dim(1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = (W - w - 1) + (H - h - 1) * W + c * W * H + n * W * H * C;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, ReverseNWise) {
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.reverse(Dim(0));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = (W - w - 1) + (H - h - 1) * W + (C - c - 1) * W * H + n * W * H * C;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, Reverse_Splat) {
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    constexpr float splatVal = 42.f;
    const std::vector<float> vals = {splatVal};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.reverse(Dim(0));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<float>(), splatVal);
}

TEST_F(MLIR_ConstContentAttrTest, Reorder) {
    const int64_t N = 1;
    const int64_t C = 2;
    const int64_t H = 2;
    const int64_t W = 2;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::NHWC);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = c + w * C + h * C * W + n * C * W * H;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, ReorderAfterReshape) {
    const int64_t N = 1;
    const int64_t C = 2;
    const int64_t H = 2;
    const int64_t W = 2;
    const auto baseType = mlir::RankedTensorType::get({N, C * H * W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.reshape({N, C, H, W}).reorder(DimsOrder::NHWC);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = c + w * C + h * C * W + n * C * W * H;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, Pad) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const int64_t PC = 1;
    const int64_t PH = 1;
    const int64_t PW = 1;

    auto contentAttrSetup = baseContentAttrSetup.padWithZero({PC, PH, PW}, {PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    checkPaddedBuffer<int32_t>(content, vals, {IC, IH, IW}, {PC, PH, PW}, 0);
}

TEST_F(MLIR_ConstContentAttrTest, PadSplat) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const int32_t splatVal = 42;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const int64_t PC = 1;
    const int64_t PH = 1;
    const int64_t PW = 1;

    auto contentAttrSetup = baseContentAttrSetup.padWithZero({PC, PH, PW}, {PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    std::vector<int32_t> vals(baseType.getNumElements(), splatVal);
    checkPaddedBuffer<int32_t>(content, vals, {IC, IH, IW}, {PC, PH, PW}, 0);
}

TEST_F(MLIR_ConstContentAttrTest, PadUniformQuant) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const int64_t OC = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto zp = 128;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, zp, 0, 255);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    const int64_t PC = 2;
    const int64_t PH = 2;
    const int64_t PW = 2;

    auto contentAttrSetup = quantContentAttrSetup.padWithZero({0, PC, PH, PW}, {0, PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    for (int64_t oc = 0; oc < OC; ++oc) {
        checkPaddedBuffer<float>(content, vals, {IC, IH, IW}, {PC, PH, PW}, zp,
                                 oc * (IC + 2 * PC) * (IW + 2 * PW) * (IH + 2 * PH), oc * IC * IW * IH);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PadPerAxisQuant) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const int64_t OC = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto zp = 127;
    std::vector<double> scales(2, 0.5);
    std::vector<int64_t> zeroPoints{zp, zp};
    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales, zeroPoints, 0, 0, 255);

    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    const int64_t POC = 2;
    const int64_t PIC = 2;
    const int64_t PH = 2;
    const int64_t PW = 2;

    auto contentAttrSetup = quantContentAttrSetup.padWithZero({POC, PIC, PH, PW}, {POC, PIC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    std::vector<int64_t> expZP(POC, zp);
    expZP.insert(expZP.end(), zeroPoints.begin(), zeroPoints.end());
    expZP.insert(expZP.end(), POC, zp);

    const auto channelSize = IC * IW * IH;
    std::vector<float> expVals(channelSize * POC, zp);
    expVals.insert(expVals.end(), vals.begin(), vals.end());
    expVals.insert(expVals.end(), channelSize * POC, zp);

    for (int64_t oc = 0; oc < OC + 2 * POC; ++oc) {
        checkPaddedBuffer<float>(content, expVals, {IC, IH, IW}, {PIC, PH, PW}, expZP[oc],
                                 oc * (IC + 2 * PIC) * (IW + 2 * PW) * (IH + 2 * PH), oc * channelSize);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PadUniformQuantileI4) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const int64_t OC = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, getSInt8Type(&ctx));
    const auto vals = std::vector<int8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const double scale = 1.0;
    const int8_t zeroPoint = 0;
    const int64_t storageTypeMin = -8;
    const int64_t storageTypeMax = 7;

    std::vector<double> quantileLUT = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                       8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

    const auto newQuantileTypeStorage =
            vpux::type::QuantileType::get(&ctx, getSInt8Type(&ctx), getUInt4Type(&ctx), quantileLUT);
    const auto quantileType = mlir::quant::UniformQuantizedType::get(
            mlir::quant::QuantizationFlags::Signed, newQuantileTypeStorage, mlir::Float32Type::get(&ctx), scale,
            zeroPoint, storageTypeMin, storageTypeMax);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantileType);

    const int64_t PC = 0;
    const int64_t PH = 2;
    const int64_t PW = 0;

    auto contentAttrSetup = quantContentAttrSetup.padWithZero({0, PC, PH, PW}, {0, PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    for (int64_t oc = 0; oc < OC; ++oc) {
        checkPaddedBuffer<int8_t>(content, vals, {IC, IH, IW}, {PC, PH, PW}, zeroPoint,
                                  oc * (IC + 2 * PC) * (IW + 2 * PW) * (IH + 2 * PH), oc * IC * IW * IH);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PadPerAxisQuantileI4) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const int64_t OC = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, getSInt8Type(&ctx));
    const auto vals = std::vector<int8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto zp = 0;
    std::vector<double> scales(1, 1);
    std::vector<int64_t> zeroPoints{zp};

    const int64_t storageTypeMin = -8;
    const int64_t storageTypeMax = 7;

    std::vector<double> quantileLUT = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                       8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

    const auto newQuantileTypeStorage =
            vpux::type::QuantileType::get(&ctx, getSInt8Type(&ctx), getUInt4Type(&ctx), quantileLUT);
    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            mlir::quant::QuantizationFlags::Signed, newQuantileTypeStorage, mlir::Float32Type::get(&ctx), scales,
            zeroPoints, 0, storageTypeMin, storageTypeMax);

    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    const int64_t POC = 0;
    const int64_t PIC = 0;
    const int64_t PH = 2;
    const int64_t PW = 0;

    auto contentAttrSetup = quantContentAttrSetup.padWithZero({POC, PIC, PH, PW}, {POC, PIC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    std::vector<int64_t> expZP(POC, zp);
    expZP.insert(expZP.end(), zeroPoints.begin(), zeroPoints.end());
    expZP.insert(expZP.end(), POC, zp);

    const auto channelSize = IC * IW * IH;
    std::vector<float> expVals(channelSize * POC, zp);
    expVals.insert(expVals.end(), vals.begin(), vals.end());
    expVals.insert(expVals.end(), channelSize * POC, zp);

    for (int64_t oc = 0; oc < OC + 2 * POC; ++oc) {
        checkPaddedBuffer<float>(content, expVals, {IC, IH, IW}, {PIC, PH, PW}, expZP[oc],
                                 oc * (IC + 2 * PIC) * (IW + 2 * PW) * (IH + 2 * PH), oc * channelSize);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SubView) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    auto contentAttrSetup = baseContentAttrSetup.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_LT(contentVals.size(), vals.size());

    for (int64_t c = 0; c < OC; ++c) {
        for (int64_t h = 0; h < OH; ++h) {
            for (int64_t w = 0; w < OW; ++w) {
                const auto newIndex = w + h * OW + c * OW * OH;
                const auto origIndex = (w + OFF_W) + (h + OFF_H) * IW + (c + OFF_C) * IW * IH;
                EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << c << " " << h << " " << w;
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, SubViewSplat) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const int32_t splatVal = 42;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    auto contentAttrSetup = baseContentAttrSetup.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<int32_t>(), splatVal);
}

TEST_F(MLIR_ConstContentAttrTest, Transpose) {
    const int64_t N = 512;
    const int64_t C = 40;
    const auto baseType = mlir::RankedTensorType::get({N, C}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto permutationMap = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0}, &ctx);
    const auto orderAttr = DimsOrder::fromAffineMap(permutationMap);
    auto contentAttrSetup = baseContentAttrSetup.transpose(orderAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const auto origIndex = n * C + c * 1;
            const auto newIndex = n * 1 + c * N;
            EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " ";
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, MemPermute) {
    const int64_t N = 512;
    const int64_t C = 40;
    const auto baseType = mlir::RankedTensorType::get({N, C}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto permutationMap = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0}, &ctx);
    const auto memPermAttr = DimsOrder::fromAffineMap(permutationMap);

    const auto dstOrderMap = mlir::AffineMap::getMultiDimIdentityMap(permutationMap.getNumDims(), &ctx);
    const auto dstOrderAttr = DimsOrder::fromAffineMap(dstOrderMap);

    auto contentAttrSetup = baseContentAttrSetup.memPermute(dstOrderAttr, memPermAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const auto origIndex = n * C + c * 1;
            const auto newIndex = n * 1 + c * N;
            EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " ";
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, ExpandDilated) {
    const int64_t OC = 2;
    const int64_t IC = 2;
    const int64_t KY = 5;
    const int64_t KX = 5;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, KY, KX}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const int64_t dilY = 3;
    const int64_t dilX = 3;

    auto contentAttrSetup = baseContentAttrSetup.expandDilated({dilY, dilX});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const int64_t dKY = KY + (KY - 1) * (dilY - 1);
    const int64_t dKX = KX + (KX - 1) * (dilX - 1);
    std::vector<int8_t> expectedVals(OC * IC * dKY * dKX, 0);

    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t ky = 0; ky < KY; ++ky) {
                for (int64_t kx = 0; kx < KX; ++kx) {
                    const auto dky = ky + (dilY - 1) * ky;
                    const auto dkx = kx + (dilX - 1) * kx;
                    const auto expectedValsInd = dkx + dky * dKX + ic * dKX * dKY + oc * dKX * dKY * IC;
                    const auto valsInd = kx + ky * KX + ic * KX * KY + oc * KX * KY * IC;
                    expectedVals[expectedValsInd] = vals[valsInd];
                }
            }
        }
    }

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ExpandDilated_Splat) {
    const int64_t OC = 2;
    const int64_t IC = 2;
    const int64_t KY = 5;
    const int64_t KX = 5;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, KY, KX}, getSInt32Type(&ctx));

    const auto vals = std::vector<int32_t>(baseType.getNumElements(), 42);
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const int64_t dilY = 3;
    const int64_t dilX = 3;

    auto contentAttrSetup = baseContentAttrSetup.expandDilated({dilY, dilX});
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));

    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const int64_t dKY = KY + (KY - 1) * (dilY - 1);
    const int64_t dKX = KX + (KX - 1) * (dilX - 1);
    std::vector<int8_t> expectedVals(OC * IC * dKY * dKX, 0);

    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t ky = 0; ky < KY; ++ky) {
                for (int64_t kx = 0; kx < KX; ++kx) {
                    const auto dky = ky + (dilY - 1) * ky;
                    const auto dkx = kx + (dilX - 1) * kx;
                    const auto expectedValsInd = dkx + dky * dKX + ic * dKX * dKY + oc * dKX * dKY * IC;
                    const auto valsInd = kx + ky * KX + ic * KX * KY + oc * KX * KY * IC;
                    expectedVals[expectedValsInd] = vals[valsInd];
                }
            }
        }
    }

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, GetSparsityMap) {
    const int64_t OC = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, getUInt8Type(&ctx));

    const auto vals = std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 0, 13, 14, 15};
    // expected result binary form:        0  1  1  1  1  1  1  1 |1  1  0   1   0  1   1   1
    // expected result HEX form:               E            F     |     B             E
    const auto expectedResult = std::vector<uint8_t>{0xFE, 0xEB};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.getSparsityMap();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    const auto ndBaseType = mlir::cast<vpux::NDTypeInterface>(baseType);
    const auto expectedType = ndBaseType.changeShapeElemType(
            ShapeRef({OC, 1, 1, 128}), mlir::IntegerType::get(ndBaseType.getContext(), 1, mlir::IntegerType::Signless));
    EXPECT_EQ(content.getType(), expectedType);
    EXPECT_EQ(content.getType(), contentAttr.getType());

    const auto valsSize = static_cast<size_t>(vals.size() / 8);
    const auto alignment = static_cast<size_t>(16);
    const auto alignedValsSize = vpux::alignValUp(valsSize, alignment);
    std::vector<uint8_t> actVals(alignedValsSize, 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.begin() + valsSize, expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, GetSparsityMapQuantized) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const int64_t OC = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, getUInt8Type(&ctx));

    // source float values: {0, -7, -6, -5, -4, -3, -2, -1,  0, 1, 0, 3, 0, 5, 6, 7};
    const double scale = 1.0;
    const int64_t zeroPoint = 7;
    const int64_t storageTypeMin = 0;
    const int64_t storageTypeMax = 14;
    // apply quantization to src values
    const auto vals = std::vector<uint8_t>{7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 10, 7, 12, 13, 14};

    // expected result binary form:        0  1  1  1  1  1  1  1 |0  1  0  1   0   1   1   1
    // expected result HEX form:                E          F      |     A             E
    const auto expectedResult = std::vector<uint8_t>{0xFE, 0xEA};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(0, baseType.getElementType(), mlir::Float32Type::get(&ctx), scale,
                                                   zeroPoint, storageTypeMin, storageTypeMax);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    auto contentAttrSetup = quantContentAttrSetup.getSparsityMap();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    const auto ndBaseType = mlir::cast<vpux::NDTypeInterface>(baseType);
    const auto expectedType = ndBaseType.changeShapeElemType(
            ShapeRef({OC, 1, 1, 128}), mlir::IntegerType::get(ndBaseType.getContext(), 1, mlir::IntegerType::Signless));
    EXPECT_EQ(content.getType(), expectedType);
    EXPECT_EQ(content.getType(), contentAttr.getType());

    const auto valsSize = static_cast<size_t>(vals.size() / 8);
    const auto alignment = static_cast<size_t>(16);
    const auto alignedValsSize = vpux::alignValUp(valsSize, alignment);
    std::vector<uint8_t> actVals(alignedValsSize, 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.begin() + valsSize, expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, Sparsify) {
    const int64_t IN = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getUInt8Type(&ctx));

    const auto vals = std::vector<uint8_t>{0,  1, 2,  3,  4,  5, 6,  7,  8,  9,  0,  11, 0, 13, 14, 15,
                                           16, 0, 18, 19, 20, 0, 22, 23, 24, 25, 26, 0,  0, 29, 30, 31};
    const auto expectedResult = std::vector<uint8_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 13, 14, 15, 0, 0, 0,
                                                     16, 18, 19, 20, 22, 23, 24, 25, 26, 29, 30, 31, 0,  0, 0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.sparsify(false);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), contentAttr.getType()) << "these types are different, this is by design";

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));

    EXPECT_ANY_THROW(contentAttr.getTransformationHash());
}

TEST_F(MLIR_ConstContentAttrTest, Sparsify_True) {
    const int64_t IN = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getUInt8Type(&ctx));

    const auto vals = std::vector<uint8_t>{0,  1, 2,  3,  4,  5, 6,  7,  8,  9,  0,  11, 0, 13, 14, 15,
                                           16, 0, 18, 19, 20, 0, 22, 23, 24, 25, 26, 0,  0, 29, 30, 31};
    const auto expectedResult = std::vector<uint8_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 13, 14, 15, 0, 0, 0,
                                                     16, 18, 19, 20, 22, 23, 24, 25, 26, 29, 30, 31, 0,  0, 0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto numElemsPerOC =
            vpux::countNonSparseElementsPerOC(Const::ContentAttr::get(baseAttr).fold(), baseType.getElementType());
    const auto numElemsPerOCType =
            mlir::RankedTensorType::get({static_cast<int64_t>(numElemsPerOC.size())}, getInt64Type(&ctx));
    const auto numElemsAttr = Const::createConstContent(numElemsPerOCType, ArrayRef(numElemsPerOC));
    const auto contentAttr = Const::ContentAttr::get(baseAttr, baseContentAttrSetup.sparsify(true, numElemsAttr));
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType()) << "these types are the same";

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));

    EXPECT_ANY_THROW(contentAttr.getTransformationHash());
}

TEST_F(MLIR_ConstContentAttrTest, SparsifyQuantized) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const int64_t IN = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;

    // source float values:{0, -15, -14,  -13,  -12,  -11, -10,  -9,  -8,  -7,  0,  -5, 0, -3, -2, -1,
    //                      0,   0,   2,    3,    4,    5,   6,   7,   8,   9, 10,   0, 0, 13, 14, 15};
    const double scale = 1.0;
    const int64_t zeroPoint = 16;
    const int64_t storageTypeMin = 0;
    const int64_t storageTypeMax = 31;
    // apply quantization to src values
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getUInt8Type(&ctx));
    const auto vals = std::vector<uint8_t>{16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  16, 11, 16, 13, 14, 15,
                                           16, 16, 18, 19, 20, 16, 22, 23, 24, 25, 26, 16, 16, 29, 30, 31};
    const auto expectedResult = std::vector<uint8_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 13, 14, 15, 0, 0, 0,
                                                     18, 19, 20, 22, 23, 24, 25, 26, 29, 30, 31, 0,  0,  0, 0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(0, baseType.getElementType(), mlir::Float32Type::get(&ctx), scale,
                                                   zeroPoint, storageTypeMin, storageTypeMax);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    auto contentAttrSetup = quantContentAttrSetup.sparsify(false);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), contentAttr.getType()) << "these types are different, this is by design";

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_EQ(actVals.size(), expectedResult.size());
    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, PositionRequirement) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 3;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup baseContentAttrSetup(nullptr, baseType);

    // Inserting a transformation that has no position requirement
    auto contentAttrSetup1 = baseContentAttrSetup.rescale(10.0);

    // Inserting a transformation that has the LAST position requirement
    auto contentAttrSetup2 = contentAttrSetup1.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    // Inserting a transformation that has the PREFERRED_LAST position requirement
    auto contentAttrSetup3 = contentAttrSetup2.sparsify(false);

    // Inserting another transformation that has no position requirement
    auto contentAttrSetup4 = contentAttrSetup3.castElemType(mlir::Float16Type::get(&ctx));

    const auto finalTransformations = contentAttrSetup4.getTransformations();
    EXPECT_EQ(finalTransformations.size(), 4);
    EXPECT_EQ(finalTransformations[0].getTransformationName(), "Rescale");
    EXPECT_EQ(finalTransformations[1].getTransformationName(), "CastElemType");
    EXPECT_EQ(finalTransformations[2].getTransformationName(), "Sparsify");
    EXPECT_EQ(finalTransformations[3].getTransformationName(), "SwizzleConstant");
}

TEST_F(MLIR_ConstContentAttrTest, ChangeShapeAndElemType) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 1, 1, 8}, getUInt8Type(&ctx));
    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, 128, 0, 255);
    auto quantContentAttrSetup = baseContentAttrSetup.changeShapeAndElemType({1, 2, 1, 8}, quantType);

    auto quantContentAttr = Const::ContentAttr::get(baseAttr, std::move(quantContentAttrSetup));
    const auto content = quantContentAttr.fold();
    EXPECT_EQ(content.getType(), quantContentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(quantContentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<uint8_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ChangeShapeAndElemTypePerAxisQuant) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 1, 1, 8}, getUInt8Type(&ctx));
    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const auto zp = 127;
    std::vector<double> scales(2, 0.5);
    std::vector<int64_t> zeroPoints{zp, zp};
    int32_t quantizedDim1 = 0;
    const auto quantElemType1 = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), scales, zeroPoints, quantizedDim1, 0, 255);
    auto quantContentAttrSetup1 = baseContentAttrSetup.castElemType(quantElemType1);

    int32_t quantizedDim2 = 1;
    const auto quantElemType2 = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), scales, zeroPoints, quantizedDim2, 0, 255);
    auto quantContentAttrSetup2 = quantContentAttrSetup1.clone().changeShapeAndElemType({1, 2, 1, 8}, quantElemType2);

    auto quantContentAttr1 = Const::ContentAttr::get(baseAttr, std::move(quantContentAttrSetup1));
    auto quantContentAttr2 = Const::ContentAttr::get(baseAttr, std::move(quantContentAttrSetup2));
    const auto content1 = quantContentAttr1.fold();
    const auto content2 = quantContentAttr2.fold();
    EXPECT_EQ(content1.getType(), quantContentAttr1.getType());
    EXPECT_EQ(content2.getType(), quantContentAttr2.getType());
    EXPECT_NE(content1.getType(), content2.getType());
    EXPECT_FALSE(content1.isSplat());
    EXPECT_EQ(quantContentAttr1.isSplat(), content1.isSplat());
    EXPECT_FALSE(content2.isSplat());
    EXPECT_EQ(quantContentAttr2.isSplat(), content2.isSplat());

    const auto contentVals1 = content1.getValues<uint8_t>();
    const auto contentVals2 = content2.getValues<uint8_t>();
    EXPECT_EQ(contentVals1.size(), vals.size());
    EXPECT_EQ(contentVals2.size(), vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(contentVals1[i], vals[i]);
        EXPECT_EQ(contentVals2[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ChangeShapeAndElemTypeFloat) {
    const auto baseType = mlir::RankedTensorType::get({2, 1, 1, 8}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto newContentAttrSetup = baseContentAttrSetup.changeShapeAndElemType({1, 2, 1, 8}, getSInt32Type(&ctx));

    auto newContentAttr = Const::ContentAttr::get(baseAttr, std::move(newContentAttrSetup));
    const auto content = newContentAttr.fold();
    EXPECT_EQ(content.getType(), newContentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(newContentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

#ifdef BACKGROUND_FOLDING_ENABLED

TEST_F(MLIR_ConstContentAttrTest, GetTransformationsRange) {
    const auto baseType = mlir::RankedTensorType::get({10}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup contentAttrSetup(nullptr, baseType);

    contentAttrSetup = contentAttrSetup.reshape({2, 5}).subview({0, 0}, {1, 5}).add(1.0);

    auto transformations = contentAttrSetup.getTransformations();
    ASSERT_EQ(transformations.size(), 3);

    auto subviewTransformation = transformations[1];
    ASSERT_NE(subviewTransformation, nullptr);

    auto headTransformations = vpux::Const::BackgroundConstantFolding::stripTransformationsFrom(
            contentAttrSetup.getTransformations(), subviewTransformation);
    ASSERT_EQ(headTransformations.size(), 1);
    EXPECT_EQ(headTransformations[0].getTransformationName(), "Reshape");

    auto tailTransformations = vpux::Const::BackgroundConstantFolding::getLastTransformationsFrom(
            contentAttrSetup.getTransformations(), subviewTransformation);
    ASSERT_EQ(tailTransformations.size(), 2);
    EXPECT_EQ(tailTransformations[0].getTransformationName(), "SubView");
    EXPECT_EQ(tailTransformations[1].getTransformationName(), "Add");
}

#endif

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_I1) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {0, 0, 0, 1,   // 0x1 -> 0x8 packed
                                       0, 0, 1, 1,   // 0x3 -> 0xC packed
                                       1, 1, 1, 1,   // 0xF -> 0xF packed
                                       1, 1, 1, 0};  // 0xE -> 0x7 packed
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));
    auto contentAttrSetup1 =
            contentAttrSetup.clone().swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));
    const std::vector<uint8_t> expectedVals = {0xC8, 0x7F};

    EXPECT_EQ(acheAlignSize, static_cast<size_t>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_I4) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const auto vals = std::vector<uint8_t>{1, 7,     // 0x17
                                           10, 15};  // 0xAF
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    auto contentAttrSetup1 =
            contentAttrSetup.clone().swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x71, 0xFA};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_U8) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const auto vals = std::vector<uint8_t>{255, 100, 0, 1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));
    const std::vector<uint8_t> expectedVals = {255, 100, 0, 1};

    EXPECT_EQ(acheAlignSize, static_cast<size_t>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_FP32) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = std::vector<float>{700, 800, 900, 900};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<float> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));
    const std::vector<float> expectedVals = {700, 800, 900, 900};

    EXPECT_EQ(acheAlignSize, static_cast<size_t>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_Splat_I1) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    const auto contentType = Const::inferFinalType(baseType, contentAttrSetup.getTransformations());
    ASSERT_NE(contentType, nullptr);

    auto contentAttrSetup1 = contentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr1.isSplat(), content.isSplat());

    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xFF, 0xFF};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_Splat_I4) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    const auto contentType = Const::inferFinalType(baseType, contentAttrSetup.getTransformations());
    ASSERT_NE(contentType, nullptr);

    auto contentAttrSetup1 = contentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr1.isSplat(), content.isSplat());

    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_Splat_U8) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x0A, 0x0A, 0x0A, 0x0A};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_Splat_FP32) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const std::vector<float> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(config::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    config::ArchKind archKind = static_cast<config::ArchKind>(config::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<float> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<float> expectedVals = {10, 10, 10, 10};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));

    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ScalarMultInverse) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = [&]() {
        auto values = generateValues<float>(baseType.getNumElements());
        values[0] = 42.f;  // Note: change values[0] (that is 0.f) to avoid divide-by-zero
        return values;
    }();
    std::vector<float> expectedVals(vals.size());
    std::transform(vals.begin(), vals.end(), expectedVals.begin(), [&](float item) {
        return 1.f / item;
    });

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.scalarMultInverse();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ScalarMultInverse_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const std::vector<float> vals{42.f};
    const std::vector<float> expectedVals{1.f / vals[0]};

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.scalarMultInverse();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), static_cast<size_t>(baseType.getNumElements()));

    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], contentVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FuseConstants) {
    auto tensorType = getInt8Type(&ctx);
    auto i1Type = getInt1Type(&ctx);

    const auto baseType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1}, tensorType);
    auto baseContentAttrSetup = Const::ContentSetup(nullptr, baseType);

    auto cst1 = getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 2, 3, 4});
    auto cst2 = getContentAttr<uint8_t>(ArrayRef<int64_t>{2, 1, 1, 1}, tensorType, SmallVector<uint8_t>{5, 6});
    auto cst3 = getContentAttr<bool>(ArrayRef<int64_t>{8, 1, 1, 1}, i1Type, SmallVector<bool>{1, 1, 1, 1, 0, 0, 0, 1});

    std::vector<uint8_t> expectedVals{1, 2, 3, 4, 5, 6, 0b10001111};

    auto fusedType = mlir::RankedTensorType::get({7, 1, 1, 1}, tensorType);
    std::vector<Const::ContentAttr> constants{cst1, cst2, cst3};
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, constants);

    auto fakeBaseContent = cst1.getBaseContent();
    auto contentAttr = Const::ContentAttr::get(fakeBaseContent, std::move(fusedConstantSetup));
    const auto content = contentAttr.fold();
    const auto contentVals = content.getValues<uint8_t>();
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

namespace {
struct QuantTypeParams {
    unsigned quantFlags = 0;
    mlir::Type storageType;
    double scale = 1.;
    int64_t zeroPoint = 0;
    int64_t storageMin = 0;
    int64_t storageMax = 0;
};
struct ConvertElemTypeParams {
    std::vector<float> inputValues;
    mlir::ShapedType baseType;
    QuantTypeParams inType;
    QuantTypeParams outType;
};
using GetConvertElemTypeParams = std::function<ConvertElemTypeParams(mlir::MLIRContext*)>;

std::tuple<mlir::quant::QuantizedType, mlir::quant::QuantizedType> getDefaultTypes(mlir::MLIRContext* ctx,
                                                                                   QuantTypeParams paramsIn,
                                                                                   QuantTypeParams paramsOut) {
    return std::make_tuple(
            mlir::quant::UniformQuantizedType::get(paramsIn.quantFlags, paramsIn.storageType,
                                                   /*expressed type*/ mlir::Float16Type::get(ctx), paramsIn.scale,
                                                   paramsIn.zeroPoint, paramsIn.storageMin, paramsIn.storageMax),
            mlir::quant::UniformQuantizedType::get(paramsOut.quantFlags, paramsOut.storageType,
                                                   /*expressed type*/ mlir::Float16Type::get(ctx), paramsOut.scale,
                                                   paramsOut.zeroPoint, paramsOut.storageMin, paramsOut.storageMax));
}
}  // namespace
class MLIR_ConstContentAttrTest_ConvertElemType :
        public MLIR_ConstContentAttrTest,
        public ::testing::WithParamInterface<GetConvertElemTypeParams> {
public:
    MLIR_ConstContentAttrTest_ConvertElemType() {
        ctx.loadDialect<mlir::quant::QuantDialect>();
    }
};

TEST_P(MLIR_ConstContentAttrTest_ConvertElemType, Roundtrip) {
    auto [inputVals, baseType, qParamsIn, qParamsOut] = GetParam()(&ctx);
    const auto [qTypeIn, qTypeOut] = getDefaultTypes(&ctx, qParamsIn, qParamsOut);
    const auto offsetBetweenRanges = qParamsOut.storageMax - qParamsIn.storageMax;

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(inputVals));
    const auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(baseAttr, baseType).castElemType(qTypeIn).convertElemType(qTypeOut));
    const auto roundtripContentAttr = contentAttr.transform().convertElemType(qTypeIn).get();

    std::vector<float> shiftedVals;
    std::transform(inputVals.begin(), inputVals.end(), std::back_inserter(shiftedVals), [&](float x) {
        return x + offsetBetweenRanges;
    });
    const auto content = contentAttr.fold();
    auto contentVals = to_std_vector(content.getValues<float>());
    ASSERT_EQ(ArrayRef(contentVals).take_front(shiftedVals.size()), ArrayRef(shiftedVals));

    const auto roundtripContent = roundtripContentAttr.fold();
    auto roundtripVals = to_std_vector(roundtripContent.getValues<float>());
    ASSERT_EQ(ArrayRef(roundtripVals).take_front(inputVals.size()), ArrayRef(inputVals));
}

INSTANTIATE_TEST_SUITE_P(
        I8_to_U8, MLIR_ConstContentAttrTest_ConvertElemType,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    // Note: from the transformation's perspective, zero-points
                    // could be anything, but from NPU perspective, de-facto
                    // values are 0 and 128.
                    QuantTypeParams i8Params{
                            mlir::quant::QuantizationFlags::Signed, getInt8Type(ctx), 0.5, 0, -128, 127};
                    QuantTypeParams u8Params{0, getUInt8Type(ctx), 0.5, 128, 0, 255};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals{-5.f};
                    return {inputVals, baseType, i8Params, u8Params};
                },

                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    QuantTypeParams i8Params{
                            mlir::quant::QuantizationFlags::Signed, getInt8Type(ctx), 0.5, 0, -128, 127};
                    QuantTypeParams u8Params{0, getUInt8Type(ctx), 0.5, 128, 0, 255};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals;
                    for (int64_t i = 0; i < baseType.getNumElements(); ++i) {
                        int64_t res = 0;
                        if (i % 2 == 0) {
                            res = i8Params.storageMin + i;
                        } else {
                            res = i8Params.storageMax - i;
                        }
                        inputVals.push_back(static_cast<float>(res));
                    }
                    return {inputVals, baseType, i8Params, u8Params};
                }));

INSTANTIATE_TEST_SUITE_P(
        U4_to_I4, MLIR_ConstContentAttrTest_ConvertElemType,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    QuantTypeParams u4Params{0, getUInt4Type(ctx), 1.0, 8, 0, 15};
                    QuantTypeParams i4Params{mlir::quant::QuantizationFlags::Signed, getInt4Type(ctx), 1.0, 0, -8, 7};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals{5.f};
                    return {inputVals, baseType, u4Params, i4Params};
                },

                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    QuantTypeParams u4Params{0, getUInt4Type(ctx), 1.0, 8, 0, 15};
                    QuantTypeParams i4Params{mlir::quant::QuantizationFlags::Signed, getInt4Type(ctx), 1.0, 0, -8, 7};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals;
                    for (int64_t i = 0; i < baseType.getNumElements(); ++i) {
                        uint64_t res = (u4Params.storageMin + i) % u4Params.storageMax;
                        inputVals.push_back(static_cast<float>(res));
                    }
                    return {inputVals, baseType, u4Params, i4Params};
                }));

using CreateElementsAttr = std::function<mlir::ElementsAttr(mlir::MLIRContext*)>;
class MLIR_ConstContentAttrTypedTest :
        public MLIR_ConstContentAttrTest,
        public ::testing::WithParamInterface<CreateElementsAttr> {
    static const char* dataAddressImpl(mlir::ElementsAttr attr) {
        // support most probable candidates
        if (const auto content = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
            return content.getRawData().data();
        }
        if (const auto content = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(attr)) {
            return content.getRawHandle().getBlob()->getData().data();
        }
        assert(false && "Extend this function with extra types");
        return nullptr;
    }

protected:
    mlir::ElementsAttr baseContent() {
        return GetParam()(&ctx);
    }
    static const void* dataAddress(mlir::ElementsAttr attr) {
        // return a void* instead to compare pointers instead of strings
        return static_cast<const void*>(dataAddressImpl(attr));
    }

    SmallVector<Const::TransformAttrInterface> getTransformations() {
        SmallVector<Const::TransformAttrInterface> randomTransformations = {
                Const::CastElemTypeAttr::get(mlir::Float16Type::get(&ctx)),
        };
        return randomTransformations;
    }
};

namespace {
// workaround for ElementsAttr::getTypeID() not being marked const
mlir::TypeID getTypeID(mlir::ElementsAttr attr) {
    return attr.getTypeID();
}
}  // namespace

// Note: expect copy behavior to be identical, regardless of the base content type
TEST_P(MLIR_ConstContentAttrTypedTest, CopyContentAttr) {
    const auto baseAttr = baseContent();
    const auto contentAttr = Const::ContentAttr::get(baseAttr, getTransformations());

    const auto copy = contentAttr;
    ASSERT_EQ(copy.getType(), contentAttr.getType());
    ASSERT_EQ(copy.getTransformations(), contentAttr.getTransformations());
    ASSERT_EQ(getTypeID(copy.getBaseContent()), getTypeID(contentAttr.getBaseContent()));
    ASSERT_EQ(dataAddress(copy.getBaseContent()), dataAddress(contentAttr.getBaseContent()))
            << "ContentAttr copy should not deepcopy data";
}

TEST_P(MLIR_ConstContentAttrTypedTest, CopyContentAttrIndirectly) {
    const auto baseAttr = baseContent();
    const auto contentAttr = Const::ContentAttr::get(baseAttr, getTransformations());

    const auto indirectCopy = Const::ContentAttr::get(contentAttr.getBaseContent());
    ASSERT_NE(indirectCopy.getType(), contentAttr.getType()) << "Content-only copy does not copy type";
    ASSERT_NE(indirectCopy.getTransformations(), contentAttr.getTransformations())
            << "Content-only copy does not copy transformations";
    ASSERT_EQ(getTypeID(indirectCopy.getBaseContent()), getTypeID(contentAttr.getBaseContent()));
    ASSERT_EQ(dataAddress(indirectCopy.getBaseContent()), dataAddress(contentAttr.getBaseContent()))
            << "ContentAttr::get() should not deepcopy data";
}

INSTANTIATE_TEST_SUITE_P(
        CommonElementsAttrImplementations, MLIR_ConstContentAttrTypedTest,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> mlir::ElementsAttr {
                    const auto type = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(ctx));
                    const auto vals = generateValues<float>(type.getNumElements());
                    return Const::createConstContent(type, ArrayRef(vals));
                },

                [](mlir::MLIRContext* ctx) -> mlir::ElementsAttr {
                    const auto type = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(ctx));

                    // create owning blob
                    static std::unique_ptr<float[]> data = std::make_unique<float[]>(type.getNumElements());
                    auto range = convertArrayRef(ArrayRef<float>(data.get(), type.getNumElements()));

                    return Const::createExternalConstContent(type, range, "MLIR_ConstContentAttrTypedTest_resource");
                }));

TEST_F(MLIR_ConstContentAttrTest, Quantize) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    // Quantize to 1/127;56 type => [-1.4409448818897637; 0.5590551181102362]
    const double scale = 1.0 / 127.0;
    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), scale, 56, -127, 127);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {-127, -127, -127, -103, -71, -39, -7, 24, 56, 88, 120, 127, 127, 127, 127, 127};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_s4) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    // Quantize to 1/7;0 type => [-8; 7]
    const double scale = 1.0 / 7.0;
    const auto quantType = mlir::quant::UniformQuantizedType::get(
            mlir::quant::QuantizationFlags::Signed, getSInt4Type(&ctx), mlir::Float32Type::get(&ctx), scale, 0, -8, 7);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {-8, -8, -8, -8, -7, -5, -3, -2, 0, 2, 4, 5, 7, 7, 7, 7};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_u4) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const double scale = 1.0 / 7.0;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0 /* unsigned*/, getUInt4Type(&ctx),
                                                                  mlir::Float32Type::get(&ctx), scale, 7, 0, 15);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {0, 0, 0, 0, 0, 2, 3, 5, 7, 9, 10, 12, 14, 15, 15, 15};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_u6) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const double scale = 0.1;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0 /* unsigned*/, getUInt6Type(&ctx),
                                                                  mlir::Float32Type::get(&ctx), scale, 40, 0, 63);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55, 58};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_u3) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    // Quantize to 1/3;2 type => [-0.6(6); 1.6(6)]
    const double scale = 1.0 / 3.0;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0 /* unsigned*/, getUInt3Type(&ctx),
                                                                  mlir::Float32Type::get(&ctx), scale, 2, 0, 7);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_s2) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const double scale = 0.5;
    const auto quantType = mlir::quant::UniformQuantizedType::get(
            mlir::quant::QuantizationFlags::Signed, getSInt2Type(&ctx), mlir::Float32Type::get(&ctx), scale, 0, -2, 1);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {-2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        auto x = contentVals[i];
        EXPECT_EQ(x, target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_u2) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const double scale = 0.5;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0 /* unsigned*/, getUInt2Type(&ctx),
                                                                  mlir::Float32Type::get(&ctx), scale, 2, 0, 3);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Quantize_Subbyte_u1) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    const double scale = 1.0;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0 /* unsigned*/, getUInt1Type(&ctx),
                                                                  mlir::Float32Type::get(&ctx), scale, 0, 0, 1);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PerAxisQuantize_s8) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            vals[16 * i + j] = -2.0f + j * 0.25f;
        }
    }
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    // first axis qType to 1/127;56 type => [-1.4409448818897637; 0.5590551181102362]
    // second axis qType 0.003;-21; type => [-0.318; 0.444]
    std::vector<double> scales{1.0 / 127.0, 0.003};
    std::vector<int64_t> zeroPoints{56, -21};
    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales,
            zeroPoints, 0, -127, 127);

    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> targetAxis0 = {-127, -127, -127, -103, -71, -39, -7, 24, 56, 88, 120, 127, 127, 127, 127, 127};
    std::vector<char> targetAxis1 = {-127, -127, -127, -127, -127, -127, -127, -104,
                                     -21,  62,   127,  127,  127,  127,  127,  127};
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(contentVals[i], targetAxis0[i]);
        EXPECT_EQ(contentVals[16 + i], targetAxis1[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PerAxisQuantize_s4) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals(baseType.getNumElements());

    // [0] =  -2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 1.75
    // [15] = -2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 1.75
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            vals[16 * i + j] = -2.0f + j * 0.25f;
        }
    }
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);

    // first axis qType to 1/8.0;3 type => [-1.375; 0.625]
    // second axis qType 0.3;-2; type => [-1.5; 3]
    std::vector<double> scales{1.0 / 8.0, 0.3};
    std::vector<int64_t> zeroPoints{3, -2};
    const auto quantType =
            mlir::quant::UniformQuantizedPerAxisType::get(mlir::quant::QuantizationFlags::Signed, getSInt4Type(&ctx),
                                                          mlir::Float32Type::get(&ctx), scales, zeroPoints, 0, -8, 7);

    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> targetAxis0 = {-8, -8, -8, -7, -5, -3, -1, 1, 3, 5, 7, 7, 7, 7, 7, 7};
    std::vector<char> targetAxis1 = {-8, -8, -7, -6, -5, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4};
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(contentVals[i], targetAxis0[i]);
        EXPECT_EQ(contentVals[16 + i], targetAxis1[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, DequantizeQuantize) {
    ctx.loadDialect<mlir::quant::QuantDialect>();

    const auto si8Type = getSInt8Type(&ctx);
    // Quantize to 0.003;-21;<-120;120> type => [-0.297;0.423]
    const auto quantType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, si8Type,
                                                                  mlir::Float16Type::get(&ctx), 0.003, 56, -120, 120);

    const auto baseType = mlir::RankedTensorType::get({1, 16}, si8Type);
    std::vector<char> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = -120 + 15 * i;
    }
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(quantType).dequantize().quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, OperationPrinting) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() -> tensor<1xf32> {
                %cst0 = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
                return %cst0 : tensor<1xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);
    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);
    mlir::OpBuilder builder(func);

    Const::DeclareOp constOp = nullptr;
    func.walk([&](Const::DeclareOp op) {
        constOp = op;
    });
    ASSERT_NE(constOp, nullptr);

    ASSERT_NO_THROW(constOp.dump());
    ASSERT_NO_THROW(llvm::outs() << constOp << "\n");
    // Note: use error-level-log so that it is actually called...
    ASSERT_NO_THROW(Logger::global().error("{0}", constOp));

    // Note: special case with "generic op form", selected internally by MLIR's
    // AsmPrinter when op verification fails.
    mlir::OpPrintingFlags printUsingGenericFallbackLogic;
    printUsingGenericFallbackLogic.printGenericOpForm();
    ASSERT_NO_THROW(constOp.print(llvm::outs(), printUsingGenericFallbackLogic));
    llvm::outs() << "\n";
}

using MLIR_ConstContentAttrTest_StableHash = MLIR_UnitBase;

TEST_F(MLIR_ConstContentAttrTest_StableHash, ShowTheDifference) {
    const auto getAddHash = [](const mlir::DialectRegistry& registry, bool secondCall) {
        mlir::MLIRContext ctx;
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();

        if (secondCall) {
            // "warmup" storage
            const auto dummyType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
            const auto dummyAttr = getContentAttr<float>(dummyType, [](Const::ContentSetup& setup) {
                return setup.reshape(ShapeRef{1, 2, 4, 3});
            });
            (void)dummyAttr;
        }
        const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
        const auto baseAttr = getContentAttr<float>(baseType, [](Const::ContentSetup& setup) {
            return setup.add(1);
        });

        auto stableHash = baseAttr.getTransformationHash();

        auto transformations = baseAttr.getTransformations();
        auto standardHash =
                static_cast<unsigned>(llvm::hash_combine_range(transformations.begin(), transformations.end()));

        return std::make_tuple(stableHash, standardHash);
    };

    auto [addStableHashFirst, addStandardHashFirst] = getAddHash(registry, /*secondCall=*/false);
    auto [addStableHashSecond, addStandardHashSecond] = getAddHash(registry, /*secondCall=*/true);

    // clang-format off
    // MLIR provides "standard" hash method for attributes/types/etc.
    // inline ::llvm::hash_code hash_value(Attribute arg) {
    //   return DenseMapInfo<const Attribute::ImplType *>::getHashValue(arg.impl);
    // }
    //
    // where:
    // static unsigned getHashValue(const T *PtrVal) {
    //   return (unsigned((uintptr_t)PtrVal) >> 4) ^
    //          (unsigned((uintptr_t)PtrVal) >> 9);
    // }
    //
    // It means that for given transformation attribute
    // hash is unique with current context
    // and changes between different launches.
    //
    // In contrast, "stable" hash has the same unique value between different launches.
    // clang-format on

    EXPECT_EQ(addStableHashFirst, addStableHashSecond);
    EXPECT_NE(addStandardHashFirst, addStandardHashSecond);
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, DifferentValues) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto baseAttrFirst = getContentAttr<float>(baseType, [](Const::ContentSetup& setup) {
        return setup.add(4);
    });

    auto stableHashAdd4 = baseAttrFirst.getTransformationHash();

    const auto baseAttrSecond = getContentAttr<float>(baseType, [](Const::ContentSetup& setup) {
        return setup.add(5);
    });

    auto stableHashAdd5 = baseAttrSecond.getTransformationHash();

    EXPECT_NE(stableHashAdd4, stableHashAdd5);
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, SameValues_CastElemTypeUniformPerAxis) {
    // Note: extend the lifetime of both contexts to ensure unique addresses
    // (otherwise, addresses for the second context may end up being the same as
    // in the first context, because the first is deallocated already).
    mlir::MLIRContext ctx1;
    ctx1.appendDialectRegistry(registry);
    ctx1.loadDialect<Const::ConstDialect>();
    ctx1.loadDialect<mlir::quant::QuantDialect>();
    mlir::MLIRContext ctx2;
    ctx2.appendDialectRegistry(registry);
    ctx2.loadDialect<Const::ConstDialect>();
    ctx2.loadDialect<mlir::quant::QuantDialect>();

    const auto getHash = [&](bool secondCall) {
        auto& currentCtx = secondCall ? ctx2 : ctx1;

        const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&currentCtx));

        const auto baseAttr = getContentAttr<float>(baseType, [&](Const::ContentSetup& setup) {
            std::vector<double> scales({2, 0.5});  // Note: different scales!
            std::vector<int64_t> zeroPoints{127, 127};
            const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
                    0, getUInt8Type(&currentCtx), mlir::Float32Type::get(&currentCtx), scales, zeroPoints,
                    /*quantization dim=*/1, 0, 255);
            return setup.castElemType(quantType);
        });

        return baseAttr.getTransformationHash();
    };

    auto stableHashFirst = getHash(/*secondCall=*/false);
    auto stableHashSecond = getHash(/*secondCall=*/true);

    EXPECT_EQ(stableHashFirst, stableHashSecond);

    // just to double check the value is the same across different runs
    const size_t expectedValue = 7072335194490813993ULL;
    EXPECT_EQ(expectedValue, static_cast<size_t>(stableHashFirst));
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, SameValues_CastElemTypeQuantilePerAxis) {
    // Note: extend the lifetime of both contexts to ensure unique addresses
    // (otherwise, addresses for the second context may end up being the same as
    // in the first context, because the first is deallocated already).
    mlir::MLIRContext ctx1;
    ctx1.appendDialectRegistry(registry);
    ctx1.loadDialect<Const::ConstDialect>();
    ctx1.loadDialect<mlir::quant::QuantDialect>();
    ctx1.loadDialect<vpux::type::QuantileDialect>();
    mlir::MLIRContext ctx2;
    ctx2.appendDialectRegistry(registry);
    ctx2.loadDialect<Const::ConstDialect>();
    ctx2.loadDialect<mlir::quant::QuantDialect>();
    ctx2.loadDialect<vpux::type::QuantileDialect>();

    const auto getHash = [&](bool secondCall) {
        auto& currentCtx = secondCall ? ctx2 : ctx1;

        const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&currentCtx));

        const auto baseAttr = getContentAttr<float>(baseType, [&](Const::ContentSetup& setup) {
            std::vector<double> quantiles(256, -0.5);  // Note: these are invalid but it doesn't matter for the test
            std::vector<double> scales({2, 0.5});      // Note: different scales!
            std::vector<int64_t> zeroPoints{127, 127};
            const auto newQuantileTypeStorage = vpux::type::QuantileType::get(
                    &currentCtx, getUInt8Type(&currentCtx), mlir::Float32Type::get(&currentCtx), quantiles);
            const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
                    0, newQuantileTypeStorage, mlir::Float32Type::get(&currentCtx), scales, zeroPoints,
                    /*quantization dim=*/1, 0, 255);
            return setup.castElemType(quantType);
        });

        return baseAttr.getTransformationHash();
    };

    auto stableHashFirst = getHash(/*secondCall=*/false);
    auto stableHashSecond = getHash(/*secondCall=*/true);

    EXPECT_EQ(stableHashFirst, stableHashSecond);

    // just to double check the value is the same across different runs
    const size_t expectedValue = 12863469656709118308ULL;
    EXPECT_EQ(expectedValue, static_cast<size_t>(stableHashFirst));
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, SameValuesButDifferentMnemonic) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto baseAttrFirst = getContentAttr<float>(baseType, [](Const::ContentSetup& setup) {
        return setup.add(5);
    });

    auto stableHashAdd5 = baseAttrFirst.getTransformationHash();

    const auto baseAttrSecond = getContentAttr<float>(baseType, [](Const::ContentSetup& setup) {
        return setup.rescale(5);
    });

    auto stableHashRescale5 = baseAttrSecond.getTransformationHash();

    EXPECT_NE(stableHashAdd5, stableHashRescale5);
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, NoTransformations) {
    // ---- different types ----

    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const auto floatType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto baseAttrFloat = getContentAttr<float>(floatType);

    auto stableHashFloat = baseAttrFloat.getTransformationHash();

    const auto intType = mlir::RankedTensorType::get({4, 4, 4, 4}, getInt64Type(&ctx));
    const auto baseAttrInt = getContentAttr<int64_t>(intType);

    auto stableHashInt = baseAttrInt.getTransformationHash();

    EXPECT_NE(stableHashFloat, stableHashInt) << "Different base content element types result in different hashes";

    const auto floatTypeNewShape = mlir::RankedTensorType::get({4, 2, 3, 1}, mlir::Float32Type::get(&ctx));
    const auto baseAttrFloatNewShape = getContentAttr<float>(floatTypeNewShape);

    auto stableHashFloatNewShape = baseAttrFloatNewShape.getTransformationHash();

    EXPECT_NE(stableHashFloat, stableHashFloatNewShape) << "Different base content shapes result in different hashes";

    // ---- different contexts ----

    const auto getHash = [](const mlir::DialectRegistry& registry) {
        mlir::MLIRContext ctx;
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();

        const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
        const auto baseAttr = getContentAttr<float>(baseType);

        return baseAttr.getTransformationHash();
    };

    auto stableHashFirst = getHash(registry);
    auto stableHashSecond = getHash(registry);

    EXPECT_EQ(stableHashFirst, stableHashSecond);

    // just to double check the value is the same across different runs
    const size_t expectedValue = 4579562972044447942ULL;
    EXPECT_EQ(expectedValue, static_cast<size_t>(stableHashFirst));
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, DimsOrder) {
    const auto getHash = [](const mlir::DialectRegistry& registry) {
        mlir::MLIRContext ctx;
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();

        const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
        const auto baseAttr = getContentAttr<float>(baseType, [](Const::ContentSetup& setup) {
            return setup.reorder(DimsOrder::NHWC);
        });

        return baseAttr.getTransformationHash();
    };

    auto stableHashFirst = getHash(registry);
    auto stableHashSecond = getHash(registry);

    EXPECT_EQ(stableHashFirst, stableHashSecond);

    // just to double check the value is the same across different runs
    const size_t expectedValue = 7992919923434012841ULL;
    EXPECT_EQ(expectedValue, static_cast<size_t>(stableHashFirst));
}

TEST_F(MLIR_ConstContentAttrTest_StableHash, MultipleTransformations) {
    const auto getHash = [](const mlir::DialectRegistry& registry) {
        mlir::MLIRContext ctx;
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();

        const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
        const auto baseAttr = getContentAttr<float>(baseType, [&](Const::ContentSetup& setup) {
            return setup.reshape(ShapeRef({1, 2, 3, 4}))
                    .padWithZero(ShapeRef({0, 2, 0, 0}), ShapeRef({0, 0, 3, 0}))
                    .reorder(DimsOrder::NHWC)
                    .castElemType(mlir::Float16Type::get(&ctx));
        });

        return baseAttr.getTransformationHash();
    };

    auto stableHashFirst = getHash(registry);
    auto stableHashSecond = getHash(registry);

    EXPECT_EQ(stableHashFirst, stableHashSecond);

    // just to double check the value is the same across different runs
    const size_t expectedValue = 15384721241491552282ULL;
    EXPECT_EQ(expectedValue, static_cast<size_t>(stableHashFirst));
}

TEST_F(MLIR_ConstContentAttrTest, AffineReshapeFuseAndSplitNoTranspose) {
    // 10x2x3 -> 5x2x6
    // 10  is reshaped into 5x2 (splitting)
    // 2x3 is reshaped into 6   (fusing)
    auto dimMapping = getArrayOfArrayAttr(&ctx, {{0, 1}, {2}, {2}});
    auto shapeValue = getArrayAttr(&ctx, {5, 2, 6});

    const auto baseType = mlir::RankedTensorType::get({10, 2, 3}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup contentAttrSetup(nullptr, baseType);
    contentAttrSetup = contentAttrSetup.affineReshape(dimMapping, shapeValue);

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(0.0f));
    const auto attr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto finalType = attr.getType();

    // templates in macros are tricky sometimes...
    const auto expectedShape = SmallVector<int64_t>{5, 2, 6};
    EXPECT_EQ(finalType.getShape().raw(), ArrayRef<int64_t>(expectedShape));
    EXPECT_EQ(finalType.getDimsOrder(), DimsOrder::fromNumDims(3));
}

TEST_F(MLIR_ConstContentAttrTest, AffineReshapeFuseAndSplitTranspose) {
    // 10x2x3 -> 6x5x2
    // 10  is reshaped into 5x2 (splitting)
    // 2x3 is reshaped into 6   (fusing)
    // Additionally, the dimensions are transposed.
    auto dimMapping = getArrayOfArrayAttr(&ctx, {{1, 2}, {0}, {0}});
    auto shapeValue = getArrayAttr(&ctx, {6, 5, 2});

    const auto baseType = mlir::RankedTensorType::get({10, 2, 3}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup contentAttrSetup(nullptr, baseType);
    contentAttrSetup = contentAttrSetup.affineReshape(dimMapping, shapeValue);

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(0.0f));
    const auto attr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto finalType = attr.getType();

    // templates in macros are tricky sometimes...
    const auto expectedShape = SmallVector<int64_t>{6, 5, 2};
    EXPECT_EQ(finalType.getShape().raw(), ArrayRef<int64_t>(expectedShape));
    EXPECT_EQ(finalType.getDimsOrder(), DimsOrder::HWC);
}

TEST_F(MLIR_ConstContentAttrTest, AffineReshapeFailedTypeInference) {
    // 2x3x4x5 -> 2x4x15
    auto dimMapping = getArrayOfArrayAttr(&ctx, {{0}, {2}, {1}, {2}});
    auto shapeValue = getArrayAttr(&ctx, {2, 4, 15});

    const auto baseType = mlir::RankedTensorType::get({2, 3, 4, 5}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup contentAttrSetup(nullptr, baseType);
    contentAttrSetup = contentAttrSetup.affineReshape(dimMapping, shapeValue);

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(0.0f));
    EXPECT_ANY_THROW(std::ignore = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup)));
}

TEST_F(MLIR_ConstContentAttrTest, Interpolate) {
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    auto expectedType = mlir::Float16Type::get(&ctx);
    const auto baseType = mlir::RankedTensorType::get({1, IC, IH, IW}, expectedType);

    const auto vals = generateValues<float>(baseType.getNumElements());

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseAttr, baseType);
    auto contentAttrSetup = baseContentAttrSetup.interpolate(
            getIntArrayAttr(&ctx, ArrayRef<int64_t>{2, 3}), getIntArrayAttr(&ctx, ArrayRef<int64_t>{2, 2}),
            mlir::StringAttr::get(&ctx, "CUBIC"), mlir::StringAttr::get(&ctx, "HALF_PIXEL"),
            mlir::StringAttr::get(&ctx, "FLOOR"), mlir::BoolAttr::get(&ctx, false),
            getIntArrayAttr(&ctx, ArrayRef<int64_t>{0, 0, 0, 0}), getIntArrayAttr(&ctx, ArrayRef<int64_t>{0, 0, 0, 0}),
            getFPAttr(&ctx, -0.75f));

    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);

    const auto content = contentAttr.fold();

    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto newShape = to_small_vector(content.getType().getShape());
    SmallVector<int64_t> expectedShape = {1, IC, 2, 2};
    EXPECT_EQ(newShape, expectedShape);

    auto contentValues = content.getValues<float>();
    std::vector<float> expectedValues = {2.031, 4.219, 10.781, 12.969};
    for (size_t i = 0; i < expectedValues.size(); ++i) {
        EXPECT_NEAR(contentValues[i], expectedValues[i], 1e-3);
    }
}

TEST_F(MLIR_ConstContentAttrTest, GatherElements) {
    const auto baseType = mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&ctx));
    const auto baseVals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    SmallVector<int64_t> indicesVals = {2, 1, 0, -1};
    const auto indicesType = mlir::RankedTensorType::get({2, 2}, getSInt64Type(&ctx));
    auto indicesAttr = mlir::DenseElementsAttr::get(indicesType, ArrayRef<int64_t>(indicesVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.gatherElements(axisAttr, indicesAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    std::vector<float> expectedValues = {2.0f, 1.0f, 3.0f, 5.0f};
    EXPECT_EQ(contentVals.size(), expectedValues.size());
    for (size_t i = 0; i < expectedValues.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedValues[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, GatherElementsWithSplatInput) {
    const auto baseType = mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&ctx));
    const float splatVal = 1.5f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));
    const auto axisAttr = getIntAttr(&ctx, 1);
    SmallVector<int64_t> indicesVals = {2, 1, 0, -1};
    const auto indicesType = mlir::RankedTensorType::get({2, 2}, getSInt64Type(&ctx));
    auto indicesAttr = mlir::DenseElementsAttr::get(indicesType, ArrayRef<int64_t>(indicesVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.gatherElements(axisAttr, indicesAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), 4u);
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

TEST_F(MLIR_ConstContentAttrTest, GatherElementsWithSplatIndices) {
    const auto baseType = mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&ctx));
    const auto baseVals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto indicesType = mlir::RankedTensorType::get({2, 2}, getSInt64Type(&ctx));
    auto indicesAttr = mlir::DenseElementsAttr::get(indicesType, ArrayRef<int64_t>({1}));
    EXPECT_TRUE(indicesAttr.isSplat());

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.gatherElements(axisAttr, indicesAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<float>();
    std::vector<float> expectedValues = {1.0f, 1.0f, 4.0f, 4.0f};
    EXPECT_EQ(contentVals.size(), expectedValues.size());
    for (size_t i = 0; i < expectedValues.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedValues[i]);
    }
}

// [1x3] + [1x3] -> [1x6] along axis=1, with add() transform on inputs
TEST_F(MLIR_ConstContentAttrTest, ConcatF32_Axis1) {
    const auto input0Type = mlir::RankedTensorType::get({1, 3}, mlir::Float32Type::get(&ctx));
    const auto input1Type = mlir::RankedTensorType::get({1, 3}, mlir::Float32Type::get(&ctx));
    const std::vector<float> input0Vals{1.0f, 2.0f, 3.0f};
    const std::vector<float> input1Vals{4.0f, 5.0f, 6.0f};

    const auto base0 = Const::createConstContent(input0Type, ArrayRef(input0Vals));
    const auto base1 = Const::createConstContent(input1Type, ArrayRef(input1Vals));

    // Apply add(10.0) transformation to each input to verify transforms are folded during concat
    Const::ContentSetup setup0(base0, input0Type);
    setup0 = setup0.add(10.0);
    auto content0 = Const::ContentAttr::get(base0, std::move(setup0));

    Const::ContentSetup setup1(base1, input1Type);
    setup1 = setup1.add(10.0);
    auto content1 = Const::ContentAttr::get(base1, std::move(setup1));

    std::vector<Const::ContentAttr> inputContents{content0, content1};

    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {0, 3}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    auto contentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/1, inputContents);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());

    EXPECT_EQ(content.getType().getShape().raw()[0], 1);
    EXPECT_EQ(content.getType().getShape().raw()[1], 6);

    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 6);

    // Original values + 10.0 from the add() transform
    std::vector<float> expected{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    for (size_t i = 0; i < resultVals.size(); ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], expected[i]);
    }
}

// Three ui8 inputs: [1x2] + [1x3] + [1x1] -> [1x6] along axis=1
TEST_F(MLIR_ConstContentAttrTest, ConcatUI8_ThreeInputs) {
    const auto input0Type = mlir::RankedTensorType::get({1, 2}, getUInt8Type(&ctx));
    const auto input1Type = mlir::RankedTensorType::get({1, 3}, getUInt8Type(&ctx));
    const auto input2Type = mlir::RankedTensorType::get({1, 1}, getUInt8Type(&ctx));

    const std::vector<uint8_t> input0Vals{10, 20};
    const std::vector<uint8_t> input1Vals{30, 40, 50};
    const std::vector<uint8_t> input2Vals{60};

    const auto base0 = Const::createConstContent(input0Type, ArrayRef(input0Vals));
    const auto base1 = Const::createConstContent(input1Type, ArrayRef(input1Vals));
    const auto base2 = Const::createConstContent(input2Type, ArrayRef(input2Vals));

    auto content0 = Const::ContentAttr::get(base0);
    auto content1 = Const::ContentAttr::get(base1);
    auto content2 = Const::ContentAttr::get(base2);

    std::vector<Const::ContentAttr> inputContents{content0, content1, content2};

    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {0, 2}, {0, 5}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    auto contentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/1, inputContents);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.getType().getShape().raw()[1], 6);

    const auto resultVals = content.getValues<uint8_t>();
    ASSERT_EQ(resultVals.size(), 6);

    std::vector<uint8_t> expected{10, 20, 30, 40, 50, 60};
    for (size_t i = 0; i < resultVals.size(); ++i) {
        EXPECT_EQ(resultVals[i], expected[i]);
    }
}

// 4D inner axis: [1x2x2x3] + [1x2x2x3] -> [1x2x4x3] along axis=2, with rescale on inputs
TEST_F(MLIR_ConstContentAttrTest, ConcatF32_4D_InnerAxis) {
    const auto input0Type = mlir::RankedTensorType::get({1, 2, 2, 3}, mlir::Float32Type::get(&ctx));
    const auto input1Type = mlir::RankedTensorType::get({1, 2, 2, 3}, mlir::Float32Type::get(&ctx));

    std::vector<float> input0Vals;
    for (int i = 1; i <= 12; ++i) {
        input0Vals.push_back(static_cast<float>(i));
    }
    std::vector<float> input1Vals;
    for (int i = 13; i <= 24; ++i) {
        input1Vals.push_back(static_cast<float>(i));
    }

    const auto base0 = Const::createConstContent(input0Type, ArrayRef(input0Vals));
    const auto base1 = Const::createConstContent(input1Type, ArrayRef(input1Vals));

    // Apply rescale(2.0) to verify transforms are folded during concat
    Const::ContentSetup setup0(base0, input0Type);
    setup0 = setup0.rescale(2.0);
    auto content0 = Const::ContentAttr::get(base0, std::move(setup0));

    Const::ContentSetup setup1(base1, input1Type);
    setup1 = setup1.rescale(2.0);
    auto content1 = Const::ContentAttr::get(base1, std::move(setup1));

    std::vector<Const::ContentAttr> inputContents{content0, content1};

    SmallVector<SmallVector<int64_t>> offsets = {{0, 0, 0, 0}, {0, 0, 2, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    auto contentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/2, inputContents);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.getType().getShape().raw()[2], 4);

    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 24);

    // Values rescaled by 2.0, then concatenated along axis=2
    std::vector<float> expected{2,  4,  6,  8,  10, 12, 26, 28, 30, 32, 34, 36,
                                14, 16, 18, 20, 22, 24, 38, 40, 42, 44, 46, 48};
    for (size_t i = 0; i < resultVals.size(); ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumInclusiveAxis1) {
    const auto baseType = mlir::RankedTensorType::get({1, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumExclusive) {
    const auto baseType = mlir::RankedTensorType::get({1, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto trueAttr = mlir::BoolAttr::get(&ctx, true);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, trueAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {0.0f, 1.0f, 3.0f, 6.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumReverse) {
    const auto baseType = mlir::RankedTensorType::get({1, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto trueAttr = mlir::BoolAttr::get(&ctx, true);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, trueAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {10.0f, 9.0f, 7.0f, 4.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumExclusiveReverse) {
    const auto baseType = mlir::RankedTensorType::get({1, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto trueAttr = mlir::BoolAttr::get(&ctx, true);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, trueAttr, trueAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {9.0f, 7.0f, 4.0f, 0.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumAxis0_2x4) {
    const auto baseType = mlir::RankedTensorType::get({2, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 0);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    // axis 0: [[1,2,3,4],[5,6,7,8]] -> [[1,2,3,4],[6,8,10,12]]
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumAxis0_2x4_Exclusive) {
    const auto baseType = mlir::RankedTensorType::get({2, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 0);
    const auto trueAttr = mlir::BoolAttr::get(&ctx, true);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, trueAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    // axis 0, exclusive: [[0,0,0,0],[1,2,3,4]]
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumAxis0_2x4_Reverse) {
    const auto baseType = mlir::RankedTensorType::get({2, 4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 0);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);
    const auto trueAttr = mlir::BoolAttr::get(&ctx, true);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, trueAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    // axis 0, reverse: [[1+5,2+6,3+7,4+8],[5,6,7,8]] = [[6,8,10,12],[5,6,7,8]]
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumMiddleAxis_2x3x2) {
    const auto baseType = mlir::RankedTensorType::get({2, 3, 2}, mlir::Float32Type::get(&ctx));
    // [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
    std::vector<float> baseVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<float>();
    // axis 1: cumsum along dim 1
    // batch 0: [1,2] -> [1,2]; [1+3,2+4]=[4,6]; [4+5,6+6]=[9,12]
    // batch 1: [7,8] -> [7,8]; [7+9,8+10]=[16,18]; [16+11,18+12]=[27,30]
    std::vector<float> expected = {1, 2, 4, 6, 9, 12, 7, 8, 16, 18, 27, 30};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumSplatInput) {
    const auto baseType = mlir::RankedTensorType::get({1, 4}, mlir::Float32Type::get(&ctx));
    const float splatVal = 1.0f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CumSumI32) {
    const auto baseType = mlir::RankedTensorType::get({1, 4}, getSInt32Type(&ctx));
    std::vector<int32_t> baseVals = {1, 2, 3, 4};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));
    const auto axisAttr = getIntAttr(&ctx, 1);
    const auto falseAttr = mlir::BoolAttr::get(&ctx, false);

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.cumSum(axisAttr, falseAttr, falseAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    const auto contentVals = content.getValues<int32_t>();
    std::vector<int32_t> expected = {1, 3, 6, 10};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SI8_AllZero) {
    const auto baseType = mlir::RankedTensorType::get({1, 2}, getSInt8Type(&ctx));
    std::vector<int8_t> baseVals = {0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    const auto contentVals = content.getValues<int8_t>();
    std::vector<int8_t> expected = {1, 1};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SI8_AllNonZero) {
    const auto baseType = mlir::RankedTensorType::get({1, 2}, getSInt8Type(&ctx));
    std::vector<int8_t> baseVals = {5, 5};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    const auto contentVals = content.getValues<int8_t>();
    std::vector<int8_t> expected = {0, 0};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SplatZero_SI8) {
    const auto baseType = mlir::RankedTensorType::get({2, 3, 4}, getSInt8Type(&ctx));
    const int8_t splatVal = 0;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getSplatValue<int8_t>(), static_cast<int8_t>(1));
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SplatNonZero_SI8) {
    const auto baseType = mlir::RankedTensorType::get({2, 3, 4}, getSInt8Type(&ctx));
    const int8_t splatVal = 5;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getSplatValue<int8_t>(), static_cast<int8_t>(0));
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SI32) {
    const auto baseType = mlir::RankedTensorType::get({3}, getSInt32Type(&ctx));
    std::vector<int32_t> baseVals = {0, 2, -1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    const auto contentVals = content.getValues<int32_t>();
    std::vector<int32_t> expected = {1, 0, 0};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SI64) {
    const auto baseType = mlir::RankedTensorType::get({3}, getSInt64Type(&ctx));
    std::vector<int64_t> baseVals = {0, 9, -2};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    const auto contentVals = content.getValues<int64_t>();
    std::vector<int64_t> expected = {1, 0, 0};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SplatNonZero_SI64) {
    const auto baseType = mlir::RankedTensorType::get({2, 3, 4}, getSInt64Type(&ctx));
    const int64_t splatVal = 7;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getSplatValue<int64_t>(), static_cast<int64_t>(0));
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SplatNonZero_F16) {
    const auto baseType = mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx));
    const float splatVal = 2.5f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getSplatValue<float>(), static_cast<float>(0.0f));
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_F16_Mixed) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::Float16Type::get(&ctx));
    std::vector<float> baseVals = {0.0f, 1.0f, 0.0f, -3.5f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {1.0f, 0.0f, 1.0f, 0.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_F32_Mixed) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::Float32Type::get(&ctx));
    std::vector<float> baseVals = {0.0f, 1.0f, 0.0f, -3.5f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    const auto contentVals = content.getValues<float>();
    std::vector<float> expected = {1.0f, 0.0f, 1.0f, 0.0f};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_F64_Mixed) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::Float64Type::get(&ctx));
    std::vector<double> baseVals = {0.0, 7.25, 0.0, -1.0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseVals));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    const auto contentVals = content.getValues<double>();
    std::vector<double> expected = {1.0, 0.0, 1.0, 0.0};
    EXPECT_EQ(contentVals.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(contentVals[i], expected[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, LogicalNot_SplatZero_F64) {
    const auto baseType = mlir::RankedTensorType::get({2, 2}, mlir::Float64Type::get(&ctx));
    const double splatVal = 0.0;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup contentAttrSetup(baseAttr, baseType);
    contentAttrSetup = contentAttrSetup.logicalNot();
    auto contentAttr = Const::ContentAttr::get(baseAttr, contentAttrSetup);
    const auto content = contentAttr.fold();

    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getSplatValue<double>(), static_cast<double>(1.0));
}
