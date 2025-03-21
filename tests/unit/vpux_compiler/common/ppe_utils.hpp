//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <gtest/gtest.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

using namespace vpux;

template <class MainOpType>
class LayerWithPostOpModel final :
        public IE::LayerWithPostOpInterface::ExternalModel<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation*, mlir::Operation*, const LogCb&) const {
        return true;
    }

    bool isSupportedClampOp(mlir::Operation*, mlir::Operation*, const LogCb&) const {
        return true;
    }

    void setLayerClampOp(mlir::Operation*, mlir::Operation*) const {
    }
};

class MLIR_PpeRegistry : public testing::Test {
public:
    MLIR_PpeRegistry() {
        registry = createDialectRegistry();
        registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
            IE::ConvolutionOp::attachInterface<LayerWithPostOpModel<IE::ConvolutionOp>>(*ctx);
            IE::TransposedConvolutionOp::attachInterface<LayerWithPostOpModel<IE::TransposedConvolutionOp>>(*ctx);
            IE::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<IE::GroupConvolutionOp>>(*ctx);
            IE::MaxPoolOp::attachInterface<LayerWithPostOpModel<IE::MaxPoolOp>>(*ctx);
            IE::AvgPoolOp::attachInterface<LayerWithPostOpModel<IE::AvgPoolOp>>(*ctx);
            IE::AddOp::attachInterface<LayerWithPostOpModel<IE::AddOp>>(*ctx);
            IE::SubtractOp::attachInterface<LayerWithPostOpModel<IE::SubtractOp>>(*ctx);
            IE::MultiplyOp::attachInterface<LayerWithPostOpModel<IE::MultiplyOp>>(*ctx);
            IE::MatMulOp::attachInterface<LayerWithPostOpModel<IE::MatMulOp>>(*ctx);
        });
    }

protected:
    mlir::DialectRegistry registry;
};

class VPU_PpeUnitBase : public MLIR_PpeRegistry {
protected:
    VPU_PpeUnitBase(std::unique_ptr<VPU::IPpeFactory>&& ppeIfc)
            : MLIR_PpeRegistry(), _ctx(registry), _loc(mlir::UnknownLoc::get(&_ctx)), _ppeIfc(std::move(ppeIfc)) {
        _ctx.loadDialect<Const::ConstDialect>();
        _ctx.loadDialect<IE::IEDialect>();
        _ctx.loadDialect<VPU::VPUDialect>();
    }

protected:
    mlir::MLIRContext _ctx;
    mlir::Location _loc;
    VPU::PpeIfcPtr _ppeIfc;

protected:
    mlir::Type getF32Type() {
        return mlir::Float32Type::get(&_ctx);
    }

    mlir::Type getF16Type() {
        return mlir::Float16Type::get(&_ctx);
    }

    mlir::Type getU8Type() {
        return mlir::quant::UniformQuantizedType::getChecked(_loc, 0, getInt8Type(&_ctx), mlir::Float16Type::get(&_ctx),
                                                             0.002, 128, 0, 255);
    }

    mlir::Type getU8PerAxisType() {
        return mlir::quant::UniformQuantizedPerAxisType::getChecked(
                _loc, 0, getInt8Type(&_ctx), mlir::Float16Type::get(&_ctx), SmallVector<double>({0.002, 0.004}),
                SmallVector<int64_t>({128, 128}), 1, 0, 255);
    }

    mlir::Type getF8Type() {
        return mlir::quant::UniformQuantizedType::getChecked(_loc, 0, mlir::Float8E4M3FNType::get(&_ctx),
                                                             mlir::Float16Type::get(&_ctx), 0.002, 0, -448, 448);
    }

    VPU::PPEStubAttr getStubPPEAttr() {
        return VPU::PPEStubAttr::get(&_ctx);
    }

    IE::PostOpAttr createRelu() {
        const auto operationName = IE::ReLUOp::getOperationName();
        return IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName),
                                   mlir::DictionaryAttr::get(&_ctx));
    }

    IE::PostOpAttr createLeakyRelu(double negativeSlope) {
        const auto operationName = IE::LeakyReluOp::getOperationName();

        SmallVector<mlir::NamedAttribute> dicAttrFields;
        const auto negativeSlopeName =
                IE::LeakyReluOp::getNegativeSlopeAttrName(mlir::OperationName(operationName, &_ctx));
        const auto negativeSlopeAttr = getFPAttr(&_ctx, negativeSlope);
        dicAttrFields.emplace_back(negativeSlopeName, negativeSlopeAttr);

        const auto dicAttr = mlir::DictionaryAttr::get(&_ctx, dicAttrFields);
        return IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName), dicAttr);
    }

    IE::PostOpAttr createClamp(double min, double max) {
        const auto operationName = IE::ClampOp::getOperationName();

        SmallVector<mlir::NamedAttribute> dicAttrFields;
        const auto minClampName = IE::ClampOp::getMinAttrName(mlir::OperationName(operationName, &_ctx));
        const auto minClampValue = getFPAttr(&_ctx, min);
        dicAttrFields.emplace_back(minClampName, minClampValue);

        const auto maxClampName = IE::ClampOp::getMaxAttrName(mlir::OperationName(operationName, &_ctx));
        const auto maxClampValue = getFPAttr(&_ctx, max);
        dicAttrFields.emplace_back(maxClampName, maxClampValue);

        const auto dicAttr = mlir::DictionaryAttr::get(&_ctx, dicAttrFields);
        return IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName), dicAttr);
    }

    IE::PostOpAttr createTanh() {
        const auto operationName = IE::TanhOp::getOperationName();
        return IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName),
                                   mlir::DictionaryAttr::get(&_ctx));
    }

    IE::PostOpAttr createSigmoid() {
        const auto operationName = IE::SigmoidOp::getOperationName();
        return IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName),
                                   mlir::DictionaryAttr::get(&_ctx));
    }

    Const::DeclareOp createBias(ArrayRef<type::float16> bias) {
        mlir::OpBuilder builder(&_ctx);
        const auto shape = mlir::RankedTensorType::get({1, static_cast<int64_t>(bias.size()), 1, 1}, getF16Type());
        auto content = Const::ContentAttr::get(Const::createConstContent(shape, bias));

        return builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(&_ctx), content.getType(), std::move(content));
    }

    IE::AddOp createAdd(mlir::Type in1ElemType, mlir::Type in2ElemType, mlir::Type outElemType,
                        IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(&_ctx);
        auto input1 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in1ElemType);
        auto input2 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in2ElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        const auto broadcast = IE::AutoBroadcastTypeAttr::get(&_ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);

        return builder.create<IE::AddOp>(_loc, outType, input1.getResult(), input2.getResult(), broadcast, postOpAttr,
                                         nullptr, nullptr, nullptr);
    }

    IE::SubtractOp createSubtract(mlir::Type in1ElemType, mlir::Type in2ElemType, mlir::Type outElemType,
                                  IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(&_ctx);
        auto input1 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in1ElemType);
        auto input2 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in2ElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        const auto broadcast = IE::AutoBroadcastTypeAttr::get(&_ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);

        return builder.create<IE::SubtractOp>(_loc, outType, input1.getResult(), input2.getResult(), broadcast,
                                              postOpAttr, nullptr, nullptr, nullptr);
    }

    IE::MultiplyOp createMultiply(mlir::Type in1ElemType, mlir::Type in2ElemType, mlir::Type outElemType,
                                  IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(&_ctx);
        auto input1 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in1ElemType);
        auto input2 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in2ElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        const auto broadcast = IE::AutoBroadcastTypeAttr::get(&_ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);

        return builder.create<IE::MultiplyOp>(_loc, outType, input1.getResult(), input2.getResult(), broadcast,
                                              postOpAttr, nullptr, nullptr, nullptr);
    }

    IE::ConvolutionOp createConvolution(mlir::Type inElemType, mlir::Type weightsElemType, mlir::Type outElemType,
                                        double scale, IE::PostOpAttr postOpAttr, Const::DeclareOp bias) {
        mlir::OpBuilder builder(&_ctx);
        auto input = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                           ArrayRef<int64_t>{1, 16, 32, 32}, inElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);
        auto weights = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                             ArrayRef<int64_t>{1, 16, 32, 32}, weightsElemType);

        const auto strides = getIntArrayAttr(&_ctx, SmallVector<int64_t>{1, 1});
        const auto padsBegin = getIntArrayAttr(&_ctx, SmallVector<int64_t>{0, 0});
        const auto padsEnd = getIntArrayAttr(&_ctx, SmallVector<int64_t>{0, 0});
        const auto dilations = getIntArrayAttr(&_ctx, SmallVector<int64_t>{1, 1});
        const auto staticScale = getFPAttr(&_ctx, scale);

        return builder.create<IE::ConvolutionOp>(
                _loc, outType, input.getResult(), weights.getResult(), bias != nullptr ? bias.getResult() : nullptr,
                strides, padsBegin, padsEnd, dilations, postOpAttr, nullptr, staticScale, nullptr, nullptr);
    }

    IE::AvgPoolOp createAvgPool(mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int64_t> kernelShape,
                                double scale, IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(&_ctx);
        auto input = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                           ArrayRef<int64_t>{1, 16, 32, 32}, inElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        const auto strides = getIntArrayAttr(&_ctx, SmallVector<int64_t>{1, 1});
        const auto kernel = getIntArrayAttr(&_ctx, kernelShape);
        const auto padsBegin = getIntArrayAttr(&_ctx, SmallVector<int64_t>{0, 0});
        const auto padsEnd = getIntArrayAttr(&_ctx, SmallVector<int64_t>{0, 0});
        const auto rounding = IE::RoundingTypeAttr::get(&_ctx, IE::RoundingType::FLOOR);
        const auto staticScale = getFPAttr(&_ctx, scale);

        return builder.create<IE::AvgPoolOp>(_loc, outType, input.getResult(), kernel, strides, padsBegin, padsEnd,
                                             rounding, nullptr, postOpAttr, nullptr, staticScale, nullptr, nullptr);
    }

    IE::MaxPoolOp createMaxPool(mlir::Type inElemType, mlir::Type outElemType, IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(&_ctx);
        auto input = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                           ArrayRef<int64_t>{1, 16, 32, 32}, inElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        const auto strides = getIntArrayAttr(&_ctx, SmallVector<int64_t>{1, 1});
        const auto kernel = getIntArrayAttr(&_ctx, SmallVector<int64_t>{2, 2});
        const auto padsBegin = getIntArrayAttr(&_ctx, SmallVector<int64_t>{0, 0});
        const auto padsEnd = getIntArrayAttr(&_ctx, SmallVector<int64_t>{0, 0});
        const auto rounding = IE::RoundingTypeAttr::get(&_ctx, IE::RoundingType::FLOOR);

        return builder.create<IE::MaxPoolOp>(_loc, outType, input.getResult(), kernel, strides, padsBegin, padsEnd,
                                             rounding, postOpAttr, nullptr, nullptr, nullptr);
    }

    IE::MatMulOp createMatMul(mlir::Type in1ElemType, mlir::Type in2ElemType, mlir::Type outElemType) {
        mlir::OpBuilder builder(&_ctx);
        auto input1 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in1ElemType);
        auto input2 = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                            ArrayRef<int64_t>{1, 16, 32, 32}, in2ElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        return builder.create<IE::MatMulOp>(_loc, outType, input1.getResult(), input2.getResult(), nullptr, nullptr,
                                            nullptr);
    }

    IE::ReduceMeanOp createReduceMean(mlir::Type inElemType, mlir::Type outElemType) {
        mlir::OpBuilder builder(&_ctx);
        auto input = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                           ArrayRef<int64_t>{1, 16, 32, 32}, inElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        return builder.create<IE::ReduceMeanOp>(_loc, outType, input.getResult(), nullptr, nullptr, nullptr);
    }

    VPU::NCEInterpolateOp createNCEInterpolate(mlir::Type inElemType, mlir::Type weightsElemType,
                                               mlir::Type outElemType, VPU::PPEAttr oldPpeAttr) {
        mlir::OpBuilder builder(&_ctx);
        auto input = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                           ArrayRef<int64_t>{1, 16, 32, 32}, inElemType);
        auto weights = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                             ArrayRef<int64_t>{1, 16, 32, 32}, weightsElemType);
        auto weightsTable = builder.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&_ctx),
                                                                  ArrayRef<int64_t>{16, 1, 1, 4}, weightsElemType);
        const auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outElemType);

        const auto strides = getIntArrayAttr(&_ctx, SmallVector<int64_t>{1, 1});
        const auto rawFilterShape = getIntArrayAttr(&_ctx, mlir::cast<NDTypeInterface>(weights.getType()).getShape());

        return builder.create<VPU::NCEInterpolateOp>(_loc, outType, input, weights, weightsTable, strides, oldPpeAttr,
                                                     rawFilterShape, /*multiClusterStrategy=*/nullptr,
                                                     /*mode=*/nullptr);
    }
};
