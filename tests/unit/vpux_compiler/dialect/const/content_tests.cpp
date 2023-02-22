//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>
#include <vpux/compiler/utils/quantization.hpp>

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
}  // namespace

class MLIR_ConstContentAttrTest : public testing::Test {
public:
    mlir::MLIRContext ctx;

public:
    void SetUp() override {
        mlir::DialectRegistry registry;
        registerDialects(registry);

        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }
};

TEST_F(MLIR_ConstContentAttrTest, FromDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromSplatDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, splatVal);

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_TRUE(content.isSplat());

    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromOpaqueElementsAttr) {
    auto* dialect = ctx.getOrLoadDialect<IE::IEDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto bytes = StringRef(reinterpret_cast<const char*>(vals.data()), vals.size() * sizeof(float));
    const auto baseAttr = mlir::OpaqueElementsAttr::get(dialect, baseType, bytes);

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ConvertStorageElemType) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ConvertElemType) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.convertElemType(getSInt32Type(&ctx));
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ConvertElemTypeSplat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, splatVal);

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.convertElemType(getSInt32Type(&ctx));
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_TRUE(content.isSplat());

    EXPECT_EQ(content.getSplatValue<int32_t>(), static_cast<int32_t>(splatVal));

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<int32_t>(splatVal));
    }
}

TEST_F(MLIR_ConstContentAttrTest, Add) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto bias = 10;
    const auto vals = generateValues<float>(baseType.getNumElements());
    std::vector<float> expectedVals(vals.size());
    std::transform(vals.begin(), vals.end(), expectedVals.begin(), [bias](float item) {
        return item + bias;
    });

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.add(bias);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, QuantCast) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getUInt8Type(&ctx));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, 128, 0, 255);

    const auto contentAttr = baseContentAttr.quantCast(quantType);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<uint8_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<uint8_t>(i));
    }
}

TEST_F(MLIR_ConstContentAttrTest, Dequantize) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getSInt8Type(&ctx));

    std::vector<int8_t> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        // -127, 0, 127
        const auto choice = (static_cast<int>(i) % 3) - 1;
        vals[i] = static_cast<int8_t>(choice * 127);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const double scale = 2.0 / 254.0;
    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), scale, 0, -127, 127);

    const auto contentAttr = baseContentAttr.quantCast(quantType).dequantize();
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.reshape({1, 3, 3, 2});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.reverse(Dim(1));
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.reverse(Dim(0));
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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

TEST_F(MLIR_ConstContentAttrTest, Reorder) {
    const int64_t N = 1;
    const int64_t C = 2;
    const int64_t H = 2;
    const int64_t W = 2;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.reorder(DimsOrder::NHWC);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.reshape({N, C, H, W}).reorder(DimsOrder::NHWC);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const int64_t PC = 1;
    const int64_t PH = 1;
    const int64_t PW = 1;

    const auto contentAttr = baseContentAttr.padWithZero({PC, PH, PW}, {PC, PH, PW});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, splatVal);

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const int64_t PC = 1;
    const int64_t PH = 1;
    const int64_t PW = 1;

    const auto contentAttr = baseContentAttr.padWithZero({PC, PH, PW}, {PC, PH, PW});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    std::vector<int32_t> vals(baseType.getNumElements(), splatVal);
    checkPaddedBuffer<int32_t>(content, vals, {IC, IH, IW}, {PC, PH, PW}, 0);
}

TEST_F(MLIR_ConstContentAttrTest, PadUniformQuant) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t OC = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto zp = 128;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, zp, 0, 255);

    const auto quantContentAttr =
            baseContentAttr.convertElemType(normalizeQuantStorageType(quantType)).quantCast(quantType);
    ASSERT_NE(quantContentAttr, nullptr);
    EXPECT_NE(quantContentAttr.getType(), baseType);

    const int64_t PC = 2;
    const int64_t PH = 2;
    const int64_t PW = 2;

    const auto contentAttr = quantContentAttr.padWithZero({0, PC, PH, PW}, {0, PC, PH, PW});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    for (int64_t oc = 0; oc < OC; ++oc) {
        checkPaddedBuffer<float>(content, vals, {IC, IH, IW}, {PC, PH, PW}, zp,
                                 oc * (IC + 2 * PC) * (IW + 2 * PW) * (IH + 2 * PH), oc * IC * IW * IH);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PadPerAxisQuant) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t OC = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto zp = 127;
    std::vector<double> scales(2, 0.5);
    std::vector<int64_t> zeroPoints{zp, zp};
    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales, zeroPoints, 0, 0, 255);

    const auto quantContentAttr =
            baseContentAttr.convertElemType(normalizeQuantStorageType(quantType)).quantCast(quantType);
    ASSERT_NE(quantContentAttr, nullptr);
    EXPECT_NE(quantContentAttr.getType(), baseType);

    const int64_t POC = 2;
    const int64_t PIC = 2;
    const int64_t PH = 2;
    const int64_t PW = 2;

    const auto contentAttr = quantContentAttr.padWithZero({POC, PIC, PH, PW}, {POC, PIC, PH, PW});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    const auto contentAttr = baseContentAttr.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, splatVal);

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    const auto contentAttr = baseContentAttr.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_TRUE(content.isSplat());

    EXPECT_EQ(content.getSplatValue<int32_t>(), splatVal);
}

TEST_F(MLIR_ConstContentAttrTest, BitPack) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt8Type(&ctx));

    const auto vals = std::vector<int8_t>{1, 2, 3, 4, 5, 6};
    const auto expectedResult = std::vector<int8_t>{0x21, 0x43, 0x65};
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto bitWidth = 4;
    const auto contentAttr = baseContentAttr.bitPack(bitWidth);
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();
    const auto ndBaseType = baseType.cast<vpux::NDTypeInterface>();
    const auto expectedType = ndBaseType.changeElemType(
            mlir::IntegerType::get(ndBaseType.getContext(), bitWidth, mlir::IntegerType::SignednessSemantics::Signed));
    EXPECT_EQ(content.getType(), expectedType);

    std::vector<int8_t> actVals(vals.size() / 2, 0);
    auto buf = makeMutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, BitPackQuant) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt8Type(&ctx));

    const auto vals = std::vector<int8_t>{1, 2, 3, 1, 2, 3};
    const auto expectedResult = std::vector<int8_t>{0x21, 0x13, 0x32};
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const double scale = 1.0;
    const int64_t zeroPoint = 0;
    const int64_t storageTypeMin = -4;
    const int64_t storageTypeMax = 3;
    const auto quantType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed,
                                                                  getSInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  scale, zeroPoint, storageTypeMin, storageTypeMax);
    const auto quantContentAttr = baseContentAttr.quantCast(quantType);

    const auto bitWidth = 4;
    const auto contentAttr = quantContentAttr.bitPack(bitWidth);
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();
    const auto expectedQuantType = mlir::quant::UniformQuantizedType::get(
            mlir::quant::QuantizationFlags::Signed, mlir::IntegerType::get(&ctx, bitWidth, mlir::IntegerType::Signed),
            mlir::Float32Type::get(&ctx), scale, zeroPoint, storageTypeMin, storageTypeMax);
    const auto expectedType = baseType.cast<vpux::NDTypeInterface>().changeElemType(expectedQuantType);
    EXPECT_EQ(content.getType(), expectedType);

    std::vector<int8_t> actVals(vals.size() / 2, 0);
    auto buf = makeMutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, Transpose) {
    const int64_t N = 512;
    const int64_t C = 40;
    const auto baseType = mlir::RankedTensorType::get({N, C}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto permutationMap = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0}, &ctx);
    const auto orderAttr = DimsOrder::fromAffineMap(permutationMap);
    const auto contentAttr = baseContentAttr.transpose(orderAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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

TEST_F(MLIR_ConstContentAttrTest, BitPackIsLast) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getSInt8Type(&ctx));

    const auto vals = std::vector<int8_t>{1, 2, 3, 4, 5, 6};
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto bitWidth = 4;
    const auto contentAttr = baseContentAttr.bitPack(bitWidth);
    ASSERT_NE(contentAttr, nullptr);

    const auto addBitPackAttr = contentAttr.add(17.f);
    const auto addBitPackTransformations = addBitPackAttr.getTransformations();
    ASSERT_EQ(addBitPackTransformations.size(), 2);
    EXPECT_NE(addBitPackTransformations[0].dyn_cast<Const::AddAttr>(), nullptr);
    EXPECT_NE(addBitPackTransformations[1].dyn_cast<Const::BitPackAttr>(), nullptr);

    const auto addBroadcastBitPackAttr = addBitPackAttr.broadcast(Dim(1), 42);
    const auto addBroadcastBitPackTransformations = addBroadcastBitPackAttr.getTransformations();
    ASSERT_EQ(addBroadcastBitPackTransformations.size(), 3);
    EXPECT_NE(addBroadcastBitPackTransformations[0].dyn_cast<Const::AddAttr>(), nullptr);
    EXPECT_NE(addBroadcastBitPackTransformations[1].dyn_cast<Const::BroadcastAttr>(), nullptr);
    EXPECT_NE(addBroadcastBitPackTransformations[2].dyn_cast<Const::BitPackAttr>(), nullptr);

    // Expects input type to be quantized, while the input type will be SI8
    EXPECT_ANY_THROW(contentAttr.dequantize());

    EXPECT_NO_THROW(contentAttr.convertElemType(getSInt32Type(&ctx)));
    EXPECT_NO_THROW(contentAttr.padWithZero({0, 1, 2, 3}, {0, 3, 2, 1}));
    EXPECT_NO_THROW(contentAttr.reorder(DimsOrder::NHWC));
    EXPECT_NO_THROW(contentAttr.rescale(19.f));
    EXPECT_NO_THROW(contentAttr.reshape(Shape({IN * IC, IH, IW})));
    EXPECT_NO_THROW(contentAttr.subview({0, 0, 0, 0}, {IN, IC, IH, IW}));
    EXPECT_NO_THROW(contentAttr.transpose(DimsOrder::NHWC));

    // Inserting another transformation that has the LAST position requirement
    EXPECT_ANY_THROW(contentAttr.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::VPUX37XX)));

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), 0.078431372549019607, 0, -4, 3);
    const auto quantContentAttr = contentAttr.quantCast(quantType);
    EXPECT_NE(quantContentAttr, nullptr);
}

TEST_F(MLIR_ConstContentAttrTest, ExpandDilated) {
    const int64_t OC = 2;
    const int64_t IC = 2;
    const int64_t KY = 5;
    const int64_t KX = 5;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, KY, KX}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const int64_t dilY = 3;
    const int64_t dilX = 3;

    const auto contentAttr = baseContentAttr.expandDilated({dilY, dilX});
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), baseType);
    EXPECT_FALSE(content.isSplat());

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.getSparsityMap();
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();
    const auto ndBaseType = baseType.cast<vpux::NDTypeInterface>();
    const auto expectedType = ndBaseType.changeShapeElemType(
            Shape({OC, 1, 1, 128}), mlir::IntegerType::get(ndBaseType.getContext(), 1, mlir::IntegerType::Signless));
    EXPECT_EQ(content.getType(), expectedType);

    const auto valsSize = static_cast<size_t>(vals.size() / 8);
    const auto alignment = static_cast<size_t>(16);
    const auto alignedValsSize = vpux::alignVal(valsSize, alignment);
    std::vector<uint8_t> actVals(alignedValsSize, 0);
    auto buf = makeMutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.begin() + valsSize, expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, GetSparsityMapQuantized) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(0, baseType.getElementType(), mlir::Float32Type::get(&ctx), scale,
                                                   zeroPoint, storageTypeMin, storageTypeMax);
    const auto quantContentAttr = baseContentAttr.quantCast(quantType);

    const auto contentAttr = quantContentAttr.getSparsityMap();
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();
    const auto ndBaseType = baseType.cast<vpux::NDTypeInterface>();
    const auto expectedType = ndBaseType.changeShapeElemType(
            Shape({OC, 1, 1, 128}), mlir::IntegerType::get(ndBaseType.getContext(), 1, mlir::IntegerType::Signless));
    EXPECT_EQ(content.getType(), expectedType);

    const auto valsSize = static_cast<size_t>(vals.size() / 8);
    const auto alignment = static_cast<size_t>(16);
    const auto alignedValsSize = vpux::alignVal(valsSize, alignment);
    std::vector<uint8_t> actVals(alignedValsSize, 0);
    auto buf = makeMutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto contentAttr = baseContentAttr.sparsify(true);
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = makeMutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, SparsifyQuantized) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

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
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(0, baseType.getElementType(), mlir::Float32Type::get(&ctx), scale,
                                                   zeroPoint, storageTypeMin, storageTypeMax);
    const auto quantContentAttr = baseContentAttr.quantCast(quantType);

    const auto contentAttr = quantContentAttr.sparsify(true);
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = makeMutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
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

    const auto vals = std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto baseContentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(baseContentAttr, nullptr);
    EXPECT_EQ(baseContentAttr.getType(), baseType);

    // Inserting a transformation that has no position requirement
    const auto contentAttr1 = baseContentAttr.rescale(10.0);

    // Inserting a transformation that has the LAST position requirement
    const auto contentAttr2 = contentAttr1.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::VPUX37XX));

    // Inserting a transformation that has the PREFERRED_LAST position requirement
    const auto contentAttr3 = contentAttr2.sparsify(false);

    // Inserting another transformation that has no position requirement
    const auto contentAttr4 = contentAttr3.convertElemType(mlir::Float16Type::get(&ctx));

    const auto finalTransformations = contentAttr4.getTransformations();
    EXPECT_EQ(finalTransformations.size(), 4);
    EXPECT_EQ(finalTransformations[0].getTransformationName(), "Rescale");
    EXPECT_EQ(finalTransformations[1].getTransformationName(), "ConvertElemType");
    EXPECT_EQ(finalTransformations[2].getTransformationName(), "Sparsify");
    EXPECT_EQ(finalTransformations[3].getTransformationName(), "SwizzleConstant");
}
