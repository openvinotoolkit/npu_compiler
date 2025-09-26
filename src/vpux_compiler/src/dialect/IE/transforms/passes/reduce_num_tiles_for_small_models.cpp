//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <numeric>

namespace vpux::IE {
#define GEN_PASS_DECL_REDUCENUMTILESFORSMALLMODELSPASS
#define GEN_PASS_DEF_REDUCENUMTILESFORSMALLMODELSPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

constexpr double EXPERIMENTAL_SMALL_MODEL_MEAN_THRESHOLD = 70000.0;
constexpr double EXPERIMENTAL_SMALL_MODEL_STD_THRESHOLD = 340000.0;

constexpr double EXPERIMENTAL_SMALL_MODEL_MULTIPLY_RATIO = 0.2;
constexpr double EXPERIMENTAL_SMALL_MODEL_MATMUL_RATIO = 0.01;
constexpr double EXPERIMENTAL_SMALL_MODEL_SOFTMAX_RATIO = 0.01;

namespace {

//
// ReduceNumTilesForSmallModelsPass
//

class ReduceNumTilesForSmallModelsPass final :
        public IE::impl::ReduceNumTilesForSmallModelsPassBase<ReduceNumTilesForSmallModelsPass> {
public:
    explicit ReduceNumTilesForSmallModelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ReduceNumTilesForSmallModelsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    VPUX_THROW_UNLESS(config::hasTileExecutor(moduleOp), "Expected module to have 'IE::TileResourceOp'.");

    // Calculate mean of shapes
    llvm::DenseMap<mlir::Operation*, double> averageShapesSizes;
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        funcOp->walk([&](IE::LayerOpInterface op) {
            if (mlir::dyn_cast<IE::ViewLikeOpInterface>(op.getOperation())) {
                return;
            }
            double totalShapeSize = 0;
            for (auto input : op.getInputs()) {
                totalShapeSize +=
                        static_cast<double>(mlir::cast<vpux::NDTypeInterface>(input.getType()).getNumElements());
            }
            for (auto output : op.getOutputs()) {
                totalShapeSize +=
                        static_cast<double>(mlir::cast<vpux::NDTypeInterface>(output.getType()).getNumElements());
            }
            averageShapesSizes[op.getOperation()] = (totalShapeSize / (op.getInputs().size() + op.getOutputs().size()));
        });
    });
    auto sumOfAverageShapes =
            std::accumulate(averageShapesSizes.begin(), averageShapesSizes.end(), 0., [](double sum, const auto& avg) {
                return sum + avg.second;
            });
    auto mean = sumOfAverageShapes / averageShapesSizes.size();

    // Calculate standard deviation
    double sumOfSquares = 0;
    for (const auto& [_, averageShapeSize] : averageShapesSizes) {
        sumOfSquares += std::pow(averageShapeSize - mean, 2);
    }
    auto standardDeviation = std::sqrt(sumOfSquares / averageShapesSizes.size());

    // Calculate multiply/matmul/softmax ratios
    auto multiplyCount = std::count_if(averageShapesSizes.begin(), averageShapesSizes.end(), [](const auto& op) {
        return mlir::isa<IE::MultiplyOp>(op.first);
    });
    auto multiplyRatio = static_cast<double>(multiplyCount) / averageShapesSizes.size();
    auto matMulCount = std::count_if(averageShapesSizes.begin(), averageShapesSizes.end(), [](const auto& op) {
        return mlir::isa<IE::MatMulOp>(op.first);
    });
    auto matMulRatio = static_cast<double>(matMulCount) / averageShapesSizes.size();
    auto softmaxCount = std::count_if(averageShapesSizes.begin(), averageShapesSizes.end(), [](const auto& op) {
        return mlir::isa<IE::SoftMaxOp>(op.first);
    });
    auto softmaxRatio = static_cast<double>(softmaxCount) / averageShapesSizes.size();

    if (mean < EXPERIMENTAL_SMALL_MODEL_MEAN_THRESHOLD && standardDeviation < EXPERIMENTAL_SMALL_MODEL_STD_THRESHOLD &&
        multiplyRatio > EXPERIMENTAL_SMALL_MODEL_MULTIPLY_RATIO &&
        matMulRatio > EXPERIMENTAL_SMALL_MODEL_MATMUL_RATIO && softmaxRatio > EXPERIMENTAL_SMALL_MODEL_SOFTMAX_RATIO) {
        auto tileOp = config::getTileExecutor(moduleOp);
        tileOp.setCount(1);
        auto numDMAPorts = config::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN);
        numDMAPorts.setCount(1);
    }
}

}  // namespace

//
// createReduceNumTilesForSmallModelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReduceNumTilesForSmallModelsPass(Logger log) {
    return std::make_unique<ReduceNumTilesForSmallModelsPass>(log);
}
