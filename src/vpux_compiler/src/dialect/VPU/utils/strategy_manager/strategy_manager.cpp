//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

StrategyManager::StrategyManager(mlir::func::FuncOp func, int64_t numTiles, bool enablePrefetchTiling,
                                 VPU::MCOptimizationScope mcOptimizationScope, Logger log,
                                 SiblingOpsAnalysis& siblingsOpsAnalysis)
        : _func(func),
          _numTiles(numTiles),
          _log(log),
          _costModel(func, enablePrefetchTiling, log, siblingsOpsAnalysis),
          _optimizer(func, enablePrefetchTiling, log, siblingsOpsAnalysis),
          _siblingsOpsAnalysis(siblingsOpsAnalysis),
          _mcOptimizationScope(mcOptimizationScope) {
}

void StrategyManager::assignMultiClusterStrategy(bool enableMultiClusterForSWLayer) {
    auto setLayerStrategy = [this](VPU::MultiClusterStrategy strategy, mlir::Operation* op) {
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
            strategy == VPU::MultiClusterStrategy::Clustering ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
            strategy == VPU::MultiClusterStrategy::HKSwitch || strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
            strategy == VPU::MultiClusterStrategy::SplitOverBatch ||
            strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
            llvm::TypeSwitch<mlir::Operation*, void>(op).Case<ClusteredOpInterface>(
                    [strategy](ClusteredOpInterface clusterOp) {
                        clusterOp.setMultiClusterStrategy(strategy);
                    });

            _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}' - '{2}'", strategy, op->getName(),
                       op->getLoc());
        } else {
            VPUX_THROW("Attempting to assign an invalid strategy {0} to operation {1}", strategy, op->getName());
        }
    };

    const auto callback = [&](mlir::Operation* op) {
        _log.trace("Getting strategy for op {0}", op->getName());
        // Currently the distributed tensor only supports the tiling scheme with numTile shape=4
        // TODO: #E81820

        // E#152917 [ShaveCodeGen] Analyze & settle on GenericSwLayerOp integration & vpux interface usage
        if (mlir::isa<VPU::GenericSwLayerOp>(op)) {
            _log.info("Compiler-generated sw layers are not supported by standard multicluster strategy assignment "
                      "infra");
            return;
        }

        for (const auto& input : op->getOperands()) {
            const auto inputShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
            if (inputShape.size() != RANK_REQUIRED_FOR_TILING && inputShape.size() != DimsGroups5D::Act::numDims) {
                return;
            }
        }
        for (const auto& output : op->getResults()) {
            const auto outputShape = output.getType().cast<vpux::NDTypeInterface>().getShape();
            if (outputShape.size() != RANK_REQUIRED_FOR_TILING && outputShape.size() != DimsGroups5D::Act::numDims) {
                return;
            }
        }

        if (IE::hasDynamicTensors(op)) {
            return;
        }

        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<NCEMaxPoolOp>([&](NCEMaxPoolOp origOp) {
                    const auto inputBatch = getShape(origOp.getInput())[Dims4D::Act::N];
                    if (inputBatch > VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
                        const auto isSOBSupported =
                                origOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverBatch, _numTiles);
                        VPUX_THROW_WHEN(!isSOBSupported,
                                        "NCEMaxPoolOp has unsupported batch size and cannot be assigned SOB strategy");
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverBatch, origOp.getOperation());
                    } else {
                        auto bestStrategy = _costModel.getOptimalLayerStrategy(
                                mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    }
                })
                .Case<NCEAveragePoolOp>([&](NCEAveragePoolOp origOp) {
                    const auto inputBatch = getShape(origOp.getInput())[Dims4D::Act::N];
                    if (inputBatch > VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
                        const auto isSOBSupported =
                                origOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverBatch, _numTiles);
                        VPUX_THROW_WHEN(
                                !isSOBSupported,
                                "NCEAveragePoolOp has unsupported batch size and cannot be assigned SOB strategy");
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverBatch, origOp.getOperation());
                    } else {
                        auto bestStrategy = _costModel.getOptimalLayerStrategy(
                                mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    }
                })
                .Case<NCEEltwiseOp>([&](NCEEltwiseOp origOp) {
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(
                            mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEConvolutionOp>([&](NCEConvolutionOp origOp) {
                    if (DimsOrder::fromValue(origOp.getInput()) == DimsOrder::NHWC) {
                        auto bestStrategy = _costModel.getOptimalLayerStrategy(
                                mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else if (DimsOrder::fromValue(origOp.getInput()) == DimsOrder::NCHW) {
                        setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
                    } else {
                        VPUX_THROW("Unsupported input layout {0} to convolution ",
                                   DimsOrder::fromValue(origOp.getInput()));
                    }
                    const auto inputBatch = getShape(origOp.getInput())[Dims4D::Act::N];
                    if (inputBatch > VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
                        const auto isSOBSupported =
                                origOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverBatch, _numTiles);
                        VPUX_THROW_WHEN(
                                !isSOBSupported,
                                "NCEConvolutionOp has unsupported batch size and cannot be assigned SOB strategy");
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverBatch, origOp.getOperation());
                    }
                })
                .Case<NCECompressConvolutionOp>([&](NCECompressConvolutionOp origOp) {
                    if (DimsOrder::fromValue(origOp.getInput()) == DimsOrder::NHWC) {
                        if (origOp.isOperationSplitOverHeightCompatible(
                                    /*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
                            setLayerStrategy(VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
                                             origOp.getOperation());
                        } else {
                            auto bestStrategy = _costModel.getOptimalLayerStrategy(
                                    mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                            setLayerStrategy(bestStrategy, origOp.getOperation());
                        }
                    } else {
                        VPUX_THROW("Unsupported input layout {0} to CompressConvolution ",
                                   DimsOrder::fromValue(origOp.getInput()));
                    }
                    const auto inputBatch = getShape(origOp.getInput())[Dims4D::Act::N];
                    if (inputBatch > VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
                        const auto isSOBSupported =
                                origOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverBatch, _numTiles);
                        VPUX_THROW_WHEN(!isSOBSupported, "NCECompressConvolutionOp has unsupported batch size and "
                                                         "cannot be assigned SOB strategy");
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverBatch, origOp.getOperation());
                    }
                })
                .Case<NCEDepthConvolutionOp>([&](NCEDepthConvolutionOp origOp) {
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(
                            mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEInterpolateOp>([&](NCEInterpolateOp origOp) {
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(
                            mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<SWOpInterface>([&](SWOpInterface origOp) {
                    if (!enableMultiClusterForSWLayer) {
                        return;
                    }

                    auto inputShape =
                            origOp.getOperation()->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
                    auto outputShape =
                            origOp.getOperation()->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();

                    // Non 4D Tensor or Tensor with larger batch size cannot be tiled properly.
                    // [E90039]MC support for Non 4D Tensor.
                    std::optional<VPU::MultiClusterStrategy> bestStrategy;
                    if ((inputShape.front() > SINGLE_BATCH && isSingleBatchRequired(op)) ||
                        inputShape.size() != RANK_REQUIRED_FOR_TILING ||
                        outputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace("Operation '{0}' at '{1}' has input shape {2} forcing clustering", origOp->getName(),
                                   origOp->getLoc(), inputShape);
                        bestStrategy = VPU::MultiClusterStrategy::Clustering;
                    } else {
                        if (origOp.supportCycleCostCalculation() &&
                            _costModel.doesLayerHaveVPUNNSupportedTypes(
                                    mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()))) {
                            bestStrategy = _costModel.getOptimalLayerStrategy(
                                    mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                        } else {
                            bestStrategy = VPU::getDefaultLayerStrategy(
                                    mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                        }
                    }
                    if (bestStrategy.has_value()) {
                        setLayerStrategy(bestStrategy.value(), origOp.getOperation());
                        _log.debug("SW Operation '{0}' {1} set to {2}", origOp->getName(), origOp->getLoc(),
                                   bestStrategy);
                    }
                    return;
                })
                .Case<ConcatOp>([&](ConcatOp origOp) {
                    const auto inputType = origOp.getInputs().front().getType().cast<vpux::NDTypeInterface>();
                    const auto inputShape = inputType.getShape();
                    // Currently the distributed tensor only supports the tiling scheme with numTile shape=4
                    // TODO: #E81820
                    if (inputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has input rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), inputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }

                    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
                    const auto outputShape = outputType.getShape();
                    if (outputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), outputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }

                    auto bestStrategy =
                            VPU::getDefaultLayerStrategy(mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation()));
                    if (bestStrategy.has_value()) {
                        setLayerStrategy(bestStrategy.value(), origOp.getOperation());
                    }
                    return;
                })
                .Case<NCEPermuteOp>([&](NCEPermuteOp origOp) {
                    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
                    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
                    const auto inputShape = inputType.getShape();
                    const auto outputShape = outputType.getShape();
                    // Such configurations cannot be tiled properly.
                    if (inputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has input rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), inputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }
                    if (outputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), outputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }

                    const auto isSOCSupportedWithoutTiling = [&](mlir::Operation* op) {
                        auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
                        if (clusteredOp == nullptr) {
                            return false;
                        }
                        if (!mlir::isa<VPU::SWOpInterface, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp,
                                       VPU::NCEAveragePoolOp>(op)) {
                            return false;
                        }
                        return clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverKernel,
                                                               _siblingsOpsAnalysis,
                                                               /*reservedMem=*/Byte(0)) &&
                               clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel,
                                                                      _numTiles) &&
                               clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(),
                                                                                /*offset=*/ShapeRef(),
                                                                                /*axis=*/ShapeRef());
                    };

                    const auto checkTileDim = [&](Dim tileDim) -> bool {
                        const int64_t MIN_DIM_SIZE_FOR_TILING =
                                IE::getTileExecutor(origOp.getOperation()->getParentOfType<mlir::ModuleOp>())
                                        .getCount();
                        if (inputShape[tileDim] < MIN_DIM_SIZE_FOR_TILING) {
                            _log.trace(
                                    "Operation '{0}' at '{1}' has size {2} over tiled dimension in input. Expected to "
                                    "have greater than or equal to: {3}.",
                                    origOp->getName(), origOp->getLoc(), inputShape[tileDim], MIN_DIM_SIZE_FOR_TILING);
                            return false;
                        }
                        if (outputShape[tileDim] < MIN_DIM_SIZE_FOR_TILING) {
                            _log.trace(
                                    "Operation '{0}' at '{1}' has size {2} over tiled dimension in output. Expected to "
                                    "have greater than or equal to: {3}.",
                                    origOp->getName(), origOp->getLoc(), outputShape[tileDim], MIN_DIM_SIZE_FOR_TILING);
                            return false;
                        }
                        return true;
                    };

                    const auto isSplitOverChannelPreferred = [&](VPU::NCEPermuteOp permuteOp) {
                        // Check if permute can use multi clusters
                        const auto outputShape = getShape(permuteOp.getOutput());
                        const auto numClusters = VPU::getOptimalNumClusters(permuteOp, outputShape,
                                                                            VPU::MultiClusterStrategy::SplitOverKernel);
                        if (numClusters <= 1) {
                            return false;
                        }
                        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(permuteOp.getOperation());
                        // Temporarily assign SOK to NCEPermute for ops around to get correct distribution mode
                        clusteredOp.setMultiClusterStrategy(VPU::MultiClusterStrategy::SplitOverKernel);
                        const auto inputOpSupportSOC =
                                isSOCSupportedWithoutTiling(permuteOp.getInput().getDefiningOp());
                        const auto outputOpsSupportSOC =
                                llvm::all_of(permuteOp->getUsers(), isSOCSupportedWithoutTiling);
                        // Remove temporary strategy
                        clusteredOp->removeAttr(multiClusterStrategy);
                        return inputOpSupportSOC || outputOpsSupportSOC;
                    };

                    // NCEPermute was disabled in subgraph optimizer, so need to assign correct strategy manually here.
                    // Tracked by: #E116504
                    if (origOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, _numTiles) &&
                        checkTileDim(Dims4D::Act::C) && isSplitOverChannelPreferred(origOp)) {
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverKernel, origOp.getOperation());
                    } else if (checkTileDim(Dims4D::Act::H)) {
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverHeightOverlapped, origOp.getOperation());
                    }
                })
                .Case<NCEMatMulOp>([&](NCEMatMulOp origOp) {
                    // NCEMatMulOp is supposed to be grouped layer with 5D shape like GNCHW and it can be multiclustered
                    // only over G
                    if (origOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverGroup, _numTiles)) {
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverGroup, origOp.getOperation());
                    }
                })
                .Default([&](mlir::Operation* unknownOp) -> void {
                    _log.trace("Operation '{0}' at '{1}' does not support multi cluster", unknownOp->getName(),
                               unknownOp->getLoc());
                });
    };

    _func.walk(callback);
}

void StrategyManager::optimizeMulticlusterStrategy() {
    if (_mcOptimizationScope == VPU::MCOptimizationScope::SUBGRAPH) {
        _optimizer.optimizeStrategyAvoidSpillingOnModel();
    }
}

// Temporary strategy is assigned to Concat to help strategy optimization. We need to remove it after strategy manager
// pass.
void StrategyManager::removeTemporaryMulticlusterStrategy() {
    const auto callbackConcat = [](VPU::ConcatOp concatOp) {
        concatOp.removeMultiClusterStrategyAttr();
    };
    _func.walk(callbackConcat);
}
