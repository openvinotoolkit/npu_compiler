//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/stable_hash.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <llvm/ADT/Hashing.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <mlir/IR/Attributes.h>
#include <mlir/Parser/Parser.h>

#include <benchmark/benchmark.h>

namespace {
mlir::Type getSomeQuantPerAxisType(mlir::MLIRContext* ctx) {
    std::vector<double> scales(100, 0.5);
    scales[13] = 0.42;
    std::vector<int64_t> zeroPoints(100, 127);
    zeroPoints[13] = 130;
    return mlir::quant::UniformQuantizedPerAxisType::get(0, vpux::getUInt8Type(ctx), mlir::Float32Type::get(ctx),
                                                         scales, zeroPoints, 0, 0, 255);
}

mlir::Type getSomeQuantilePerAxisType(mlir::MLIRContext* ctx) {
    std::vector<double> quantiles(256, -0.69);  // Note: invalid values, but good enough for the test
    quantiles[13] = -0.5;
    std::vector<double> scales(100, 0.5);
    scales[13] = 0.42;
    std::vector<int64_t> zeroPoints(100, 127);
    zeroPoints[13] = 130;
    return mlir::quant::QuantileQuantizedPerAxisType::get(0, vpux::getUInt8Type(ctx), mlir::Float32Type::get(ctx),
                                                          mlir::Float32Type::get(ctx), quantiles, scales, zeroPoints, 0,
                                                          0, 255);
}
}  // namespace

// baseline - very fast
static void BM_StableHash_EnumValue(benchmark::State& state) {
    mlir::MLIRContext ctx;
    auto type = mlir::Float32Type::get(&ctx);

    enum HandRolledTypeIds : uint32_t { F32 = 0 };
    for (auto _ : state) {
        llvm::hash_code value = llvm::hash_value(HandRolledTypeIds::F32);
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

static void BM_StableHash_F32_StringSerialization(benchmark::State& state) {
    mlir::MLIRContext ctx;
    auto type = mlir::Float32Type::get(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = llvm::hash_value(vpux::formatv("{0}", type).str());
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

static void BM_StableHash_F32(benchmark::State& state) {
    mlir::MLIRContext ctx;
    auto type = mlir::Float32Type::get(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = vpux::getStableHash(type);
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

static void BM_StableHash_QuantPerAxis_StringSerialization(benchmark::State& state) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::quant::QuantDialect>();
    auto type = getSomeQuantPerAxisType(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = llvm::hash_value(vpux::formatv("{0}", type).str());
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

static void BM_StableHash_QuantPerAxis(benchmark::State& state) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::quant::QuantDialect>();
    auto type = getSomeQuantPerAxisType(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = vpux::getStableHash(type);
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

static void BM_StableHash_QuantilePerAxis_StringSerialization(benchmark::State& state) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::quant::QuantDialect>();
    auto type = getSomeQuantilePerAxisType(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = llvm::hash_value(vpux::formatv("{0}", type).str());
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

static void BM_StableHash_QuantilePerAxis(benchmark::State& state) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::quant::QuantDialect>();
    auto type = getSomeQuantilePerAxisType(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = vpux::getStableHash(type);
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_StableHash_EnumValue);
BENCHMARK(BM_StableHash_F32_StringSerialization);
BENCHMARK(BM_StableHash_F32);
BENCHMARK(BM_StableHash_QuantPerAxis_StringSerialization);
BENCHMARK(BM_StableHash_QuantPerAxis);
BENCHMARK(BM_StableHash_QuantilePerAxis_StringSerialization);
BENCHMARK(BM_StableHash_QuantilePerAxis);

// below are the tests that benchmark individual stable hashes for different
// constant transformations.

namespace {
mlir::Type getSomeQuantPerTensorType(mlir::MLIRContext* ctx) {
    std::vector<double> scales({13});
    std::vector<int64_t> zeroPoints({100});
    return mlir::quant::UniformQuantizedType::get(0, vpux::getUInt8Type(ctx), mlir::Float32Type::get(ctx), 0.42, 100, 0,
                                                  255);
}
}  // namespace

using CreateAttrFunc = vpux::FuncRef<vpux::Const::TransformAttrInterface(mlir::MLIRContext*)>;
static void BM_StableHash_IndividualAttr(benchmark::State& state, CreateAttrFunc create) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::Const::ConstDialect>();
    ctx.loadDialect<mlir::quant::QuantDialect>();
    const auto attr = create(&ctx);

    for (auto _ : state) {
        llvm::hash_code value = attr.getStableHashValue();
        benchmark::DoNotOptimize(value);
        benchmark::ClobberMemory();
    }
}

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Add, [](mlir::MLIRContext* ctx) {
    return vpux::Const::AddAttr::get(mlir::FloatAttr::get(mlir::Float32Type::get(ctx), 42));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Broadcast, [](mlir::MLIRContext* ctx) {
    return vpux::Const::BroadcastAttr::get(vpux::getIntAttr(ctx, 0), vpux::getIntAttr(ctx, 42));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, CastElemType, [](mlir::MLIRContext* ctx) {
    return vpux::Const::CastElemTypeAttr::get(mlir::Float16Type::get(ctx));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, CastElemType_QuantPerAxis, [](mlir::MLIRContext* ctx) {
    return vpux::Const::CastElemTypeAttr::get(getSomeQuantPerAxisType(ctx));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, ConvertElemType, [](mlir::MLIRContext* ctx) {
    return vpux::Const::ConvertElemTypeAttr::get(mlir::Float16Type::get(ctx));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, ConvertElemType_QuantPerAxis, [](mlir::MLIRContext* ctx) {
    return vpux::Const::ConvertElemTypeAttr::get(getSomeQuantPerAxisType(ctx));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Quantize, [](mlir::MLIRContext* ctx) {
    return vpux::Const::QuantizeAttr::get(ctx, mlir::cast<mlir::quant::QuantizedType>(getSomeQuantPerTensorType(ctx)));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Quantize_QuantPerAxis, [](mlir::MLIRContext* ctx) {
    return vpux::Const::QuantizeAttr::get(ctx, mlir::cast<mlir::quant::QuantizedType>(getSomeQuantPerAxisType(ctx)));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Dequantize, [](mlir::MLIRContext* ctx) {
    return vpux::Const::DequantizeAttr::get(ctx);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, LayoutCast, [](mlir::MLIRContext* ctx) {
    const auto identity = mlir::AffineMapAttr::get(mlir::AffineMap::getMinorIdentityMap(4, 4, ctx));
    return vpux::Const::LayoutCastAttr::get(identity);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, MemPermute, [](mlir::MLIRContext* ctx) {
    const auto identity = mlir::AffineMapAttr::get(mlir::AffineMap::getMinorIdentityMap(4, 4, ctx));
    return vpux::Const::MemPermuteAttr::get(identity, identity);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, PadWithZero, [](mlir::MLIRContext* ctx) {
    const int64_t padBefore[] = {0, 0, 0, 0};
    const int64_t padAfter[] = {10, 0, 0, 0};
    return vpux::Const::PadWithZeroAttr::get(vpux::getIntArrayAttr(ctx, padBefore),
                                             vpux::getIntArrayAttr(ctx, padAfter));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Reorder, [](mlir::MLIRContext* ctx) {
    const auto identity = mlir::AffineMapAttr::get(mlir::AffineMap::getMinorIdentityMap(4, 4, ctx));
    return vpux::Const::ReorderAttr::get(identity);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Rescale, [](mlir::MLIRContext* ctx) {
    return vpux::Const::RescaleAttr::get(mlir::FloatAttr::get(mlir::Float32Type::get(ctx), 42));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Reshape, [](mlir::MLIRContext* ctx) {
    const int64_t shape[] = {1, 2, 3, 4};
    return vpux::Const::ReshapeAttr::get(vpux::getIntArrayAttr(ctx, shape));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, ScalarMultInverse, [](mlir::MLIRContext* ctx) {
    return vpux::Const::ScalarMultInverseAttr::get(ctx);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, SubView, [](mlir::MLIRContext* ctx) {
    const int64_t offset[] = {0, 0, 0, 0};
    const int64_t shape[] = {1, 2, 3, 4};
    return vpux::Const::SubViewAttr::get(vpux::getIntArrayAttr(ctx, offset), vpux::getIntArrayAttr(ctx, shape));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Transpose, [](mlir::MLIRContext* ctx) {
    const auto identity = mlir::AffineMapAttr::get(mlir::AffineMap::getMinorIdentityMap(4, 4, ctx));
    return vpux::Const::TransposeAttr::get(identity);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Sparsify_False, [](mlir::MLIRContext* ctx) {
    return vpux::Const::SparsifyAttr::get(mlir::BoolAttr::get(ctx, false));
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, Sparsify_True, [](mlir::MLIRContext* ctx) {
    const int64_t values[] = {0, 1, 2, 3};
    const auto numElemsAttr = vpux::Const::createConstContent(mlir::RankedTensorType::get({4}, vpux::getInt64Type(ctx)),
                                                              mlir::ArrayRef(values));
    return vpux::Const::SparsifyAttr::get(mlir::BoolAttr::get(ctx, true), numElemsAttr);
});

BENCHMARK_CAPTURE(BM_StableHash_IndividualAttr, AffineReshape, [](mlir::MLIRContext* ctx) {
    const std::vector<int64_t> dimMapping[] = {{0, 1}, {2}, {2}};
    const int64_t shape[] = {5, 2, 6};
    return vpux::Const::AffineReshapeAttr::get(vpux::getIntArrayOfArray(ctx, dimMapping),
                                               vpux::getIntArrayAttr(ctx, shape));
});
