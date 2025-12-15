//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>

namespace {
void getValues(const double scale, const int64_t zeroPoint, vpux::Const::Content& input, vpux::Const::Content& output) {
    const auto qVals = input.getValues<int64_t>();
    auto realVals = output.getTempBuf<float>();

    for (size_t i = 0; i < realVals.size(); ++i) {
        realVals[i] = (qVals[i] - zeroPoint) * scale;
    }
}

void getTempBuf(const double scale, const int64_t zeroPoint, vpux::Const::Content& input,
                vpux::Const::Content& output) {
    const auto qVals = input.getTempBuf<uint8_t>();
    const uint8_t* qValsPtr = qVals.data();
    auto realVals = output.getTempBuf<float>();
    float* realValsPtr = realVals.data();

    for (size_t i = 0; i < realVals.size(); ++i) {
        realValsPtr[i] = (qValsPtr[i] - zeroPoint) * scale;
    }
}

void fuseMulAdd(const double scale, const int64_t zeroPoint, vpux::Const::Content& input,
                vpux::Const::Content& output) {
    const auto qVals = input.getTempBuf<uint8_t>();
    const uint8_t* qValsPtr = qVals.data();
    auto realVals = output.getTempBuf<float>();
    float* realValsPtr = realVals.data();

    const auto scaledZP = scale * zeroPoint;
    for (size_t i = 0; i < realVals.size(); ++i) {
        realValsPtr[i] = qValsPtr[i] * scale - scaledZP;
    }
}

void fuseMulAddDtype(const float scale, const int zeroPoint, vpux::Const::Content& input,
                     vpux::Const::Content& output) {
    const auto qVals = input.getTempBuf<uint8_t>();
    const uint8_t* qValsPtr = qVals.data();
    auto realVals = output.getTempBuf<float>();
    float* realValsPtr = realVals.data();

    const auto scaledZP = scale * zeroPoint;
    for (size_t i = 0; i < realVals.size(); ++i) {
        realValsPtr[i] = qValsPtr[i] * scale - scaledZP;
    }
}
}  // namespace

static void BM_GetValues(benchmark::State& state) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const std::vector<int64_t> shape{1024, 2048, 3, 3};
    const auto inElemType = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto inTensorType = mlir::RankedTensorType::get(shape, inElemType);
    const auto outElemType = mlir::Float32Type::get(&ctx);
    const auto outTensorType = mlir::RankedTensorType::get(shape, outElemType);

    auto input = vpux::Const::Content::allocTempBuffer(inTensorType, mlir::Float32Type::get(&ctx), false);
    auto output = vpux::Const::Content::allocTempBuffer(outTensorType, mlir::Float32Type::get(&ctx), false);
    for (auto _ : state) {
        getValues(2.0, 128, input, output);
        benchmark::ClobberMemory();
    }
}

static void BM_GetTmpBuff(benchmark::State& state) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const std::vector<int64_t> shape{1024, 2048, 3, 3};
    const auto inElemType = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto inTensorType = mlir::RankedTensorType::get(shape, inElemType);
    const auto outElemType = mlir::Float32Type::get(&ctx);
    const auto outTensorType = mlir::RankedTensorType::get(shape, outElemType);

    auto input = vpux::Const::Content::allocTempBuffer(inTensorType, mlir::Float32Type::get(&ctx), false);
    auto output = vpux::Const::Content::allocTempBuffer(outTensorType, mlir::Float32Type::get(&ctx), false);
    for (auto _ : state) {
        getTempBuf(2.0, 128, input, output);
        benchmark::ClobberMemory();
    }
}

static void BM_FuseMulAdd(benchmark::State& state) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const std::vector<int64_t> shape{1024, 2048, 3, 3};
    const auto inElemType = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto inTensorType = mlir::RankedTensorType::get(shape, inElemType);
    const auto outElemType = mlir::Float32Type::get(&ctx);
    const auto outTensorType = mlir::RankedTensorType::get(shape, outElemType);

    auto input = vpux::Const::Content::allocTempBuffer(inTensorType, mlir::Float32Type::get(&ctx), false);
    auto output = vpux::Const::Content::allocTempBuffer(outTensorType, mlir::Float32Type::get(&ctx), false);
    for (auto _ : state) {
        fuseMulAdd(2.0, 128, input, output);
        benchmark::ClobberMemory();
    }
}

static void BM_FuseMulAddDtype(benchmark::State& state) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const std::vector<int64_t> shape{1024, 2048, 3, 3};
    const auto inElemType = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto inTensorType = mlir::RankedTensorType::get(shape, inElemType);
    const auto outElemType = mlir::Float32Type::get(&ctx);
    const auto outTensorType = mlir::RankedTensorType::get(shape, outElemType);

    auto input = vpux::Const::Content::allocTempBuffer(inTensorType, mlir::Float32Type::get(&ctx), false);
    auto output = vpux::Const::Content::allocTempBuffer(outTensorType, mlir::Float32Type::get(&ctx), false);
    for (auto _ : state) {
        fuseMulAddDtype(2.0, 128, input, output);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_GetValues);
BENCHMARK(BM_GetTmpBuff);
BENCHMARK(BM_FuseMulAdd);
BENCHMARK(BM_FuseMulAddDtype);
