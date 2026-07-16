//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
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
    explicit ReduceNumTilesForSmallModelsPass(bool enableSDPAContributions, Logger log)
            : _enableSDPAContributions(enableSDPAContributions) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    bool _enableSDPAContributions;
};

mlir::LogicalResult ReduceNumTilesForSmallModelsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (enableSDPAContributions.hasValue()) {
        _enableSDPAContributions = enableSDPAContributions.getValue();
    }

    return mlir::success();
}

void ReduceNumTilesForSmallModelsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    VPUX_THROW_UNLESS(config::hasTileExecutor(moduleOp), "Expected module to have 'config::ResourcesOp'.");

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

    auto matMulCount = std::count_if(averageShapesSizes.begin(), averageShapesSizes.end(), [](const auto& op) {
        return mlir::isa<IE::MatMulOp>(op.first);
    });

    auto softmaxCount = std::count_if(averageShapesSizes.begin(), averageShapesSizes.end(), [](const auto& op) {
        return mlir::isa<IE::SoftMaxOp>(op.first);
    });

    if (_enableSDPAContributions) {
        const auto sdpaCount = std::count_if(averageShapesSizes.begin(), averageShapesSizes.end(), [](const auto& op) {
            return mlir::isa<IE::SDPAOp>(op.first);
        });
        multiplyCount += sdpaCount;
        matMulCount += (sdpaCount * 2);
        softmaxCount += sdpaCount;
    }

    const auto archKind = config::getArch(moduleOp);
    auto multiplyRatio = static_cast<double>(multiplyCount) / averageShapesSizes.size();
    auto matMulRatio = static_cast<double>(matMulCount) / averageShapesSizes.size();
    auto softmaxRatio = static_cast<double>(softmaxCount) / averageShapesSizes.size();

    const auto tileCount = VPUIP::getNumTilesUsed(moduleOp);
    _log.info("Number of tiles used: {0}", tileCount);

    if (mean < EXPERIMENTAL_SMALL_MODEL_MEAN_THRESHOLD && standardDeviation < EXPERIMENTAL_SMALL_MODEL_STD_THRESHOLD &&
        multiplyRatio > EXPERIMENTAL_SMALL_MODEL_MULTIPLY_RATIO &&
        matMulRatio > EXPERIMENTAL_SMALL_MODEL_MATMUL_RATIO && softmaxRatio > EXPERIMENTAL_SMALL_MODEL_SOFTMAX_RATIO) {
        auto tileOp = config::getTileExecutor(moduleOp);
        tileOp.setCount(1);
        _log.info("Tiles number overwritten to {0} for small model", tileOp.getCount());

        if (archKind == config::ArchKind::NPU37XX || archKind == config::ArchKind::NPU40XX) {
            // 2 DMAs are not supported for 1T compilation before NPU50XX
            auto numDMAPorts = config::getAvailableExecutor(moduleOp, config::ExecutorKind::DMA_NN);
            numDMAPorts.setCount(1);
            _log.info("DMA ports overwritten to {0} for small model", numDMAPorts.getCount());
        }
    }
}

}  // namespace

//
// createReduceNumTilesForSmallModelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReduceNumTilesForSmallModelsPass(Logger log) {
    return std::make_unique<ReduceNumTilesForSmallModelsPass>(/*enableSDPAContributions=*/false, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createReduceNumTilesForSmallModelsPass(bool enableSDPAContributions, Logger log) {
    return std::make_unique<ReduceNumTilesForSmallModelsPass>(enableSDPAContributions, log);
}
