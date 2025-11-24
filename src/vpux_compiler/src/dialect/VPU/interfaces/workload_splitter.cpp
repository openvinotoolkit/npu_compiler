//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_workload_channels.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/workload_size_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/core/layers.hpp"

#include <array>

using namespace vpux;
using namespace VPU;

vpux::VPU::WorkloadSplitter::WorkloadSplitter(mlir::func::FuncOp funcOp, ArrayRef<int64_t> supportedChannelsForDW,
                                              vpux::Logger log)
        : _funcOp(funcOp),
          _supportedChannelsForDW(supportedChannelsForDW.begin(), supportedChannelsForDW.end()),
          _log(log) {
}

void vpux::VPU::WorkloadSplitter::correctInvalidWorkload(const VPU::SparsityConstraint& sparsityConstraint) {
    mlir::DenseSet<mlir::Operation*> handledNCEOps;

    _funcOp.walk([&](VPU::NCEOpInterface nceOp) {
        if (handledNCEOps.contains(nceOp)) {
            return;
        }

        // More than one operation might need to be handled at the same time for some sparse activations,
        // to satisfy the requirements of the consumer ops
        mlir::DenseSet<mlir::Operation*> producerNCEOps{nceOp};
        const auto invalidSparseOps = findInvalidSparseOps(nceOp, sparsityConstraint);
        if (!invalidSparseOps.empty()) {
            producerNCEOps.clear();
            producerNCEOps.insert(invalidSparseOps.begin(), invalidSparseOps.end());
        }

        auto supportedChannels = getSupportedChannels(producerNCEOps, sparsityConstraint);
        const auto invalidDepthwiseOps = findInvalidDepthwiseOps(producerNCEOps, supportedChannels);
        const auto invalidNCEPermuteOps = findInvalidNCEPermuteOps(producerNCEOps);
        if (invalidSparseOps.empty() && invalidDepthwiseOps.empty() && invalidNCEPermuteOps.empty()) {
            return;
        }

        _log.trace("supportedChannels {0} for nceOp {1} to correct workloads", supportedChannels, nceOp->getLoc());
        auto channelPadding = int64_t(0);  // used for NCEPermute

        int64_t opIdx = 1;
        for (auto op : producerNCEOps) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);

            auto isInvalidDepthwise = invalidDepthwiseOps.contains(op);
            auto isInvalidSparsity = invalidSparseOps.contains(op);
            auto isInvalidNCEPermuteOp = invalidNCEPermuteOps.contains(op);
            _log.trace("Correcting workloads for operation '{0}' at '{1}'. Necessary corrections: depthwise "
                       "'{2}', sparsity '{3}' ({4}/{5}), remove padding '{6}'",
                       op->getName(), op->getLoc(), isInvalidDepthwise, isInvalidSparsity, opIdx++,
                       producerNCEOps.size(), isInvalidNCEPermuteOp);

            const auto offsetsCorrectionNeeded = VPU::isNCEPermuteOffsetsCorrectionNeeded(nceOp);
            if (isInvalidNCEPermuteOp) {
                channelPadding =
                        mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape()[Dims4D::Act::C] -
                        mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getShape()[Dims4D::Act::C];
            }
            auto workloads = to_small_vector(nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>());
            // The workloads must have the same channel size when there's output sparsity
            // It's because the workloads share the unique storage element size
            // However the last workload can have any channel size which is smaller than the previous workloads
            //
            // We can't skip the last workload for DepthwiseConv and NCEPermute because they have extra constraints on
            // workload size
            // We can't skip the last workload if there are multiple NCE operations as producer because we don't know
            // which NCE operation provides the last workload
            if ((producerNCEOps.size() == 1) && isInvalidSparsity && !isInvalidDepthwise && !isInvalidNCEPermuteOp) {
                const auto lastWorkloadSizes = parseIntArrayAttr<int64_t>(workloads.back().getOutSizes());
                const auto lastChannel = lastWorkloadSizes[Dims4D::Act::C.ind()];
                auto canSkipTheLastWorkload = llvm::any_of(supportedChannels, [&](int64_t channel) -> bool {
                    return lastChannel < channel;
                });

                if (canSkipTheLastWorkload) {
                    workloads.pop_back();
                }
            }

            // In case ODU autopad is used and the existing channel padding needs to be removed, update the supported
            // channels to reflect the unpadded channels
            if (channelPadding != 0 && VPU::canAutopadOutput(op)) {
                const auto outputChannels =
                        mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape()[Dims4D::Act::C];
                for (auto& channel : supportedChannels) {
                    if (channel == outputChannels) {
                        channel -= channelPadding;
                    }
                }
            }

            for (auto workloadOp : llvm::make_early_inc_range(workloads)) {
                const auto wlSizes = parseIntArrayAttr<int64_t>(workloadOp.getOutSizes());
                auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
                if (llvm::find(supportedChannels, wlChannels) != supportedChannels.end()) {
                    continue;
                }

                splitWorkload(workloadOp, supportedChannels, isInvalidNCEPermuteOp, channelPadding,
                              offsetsCorrectionNeeded, _log);
            }

            handledNCEOps.insert(op);
        }
    });
}

SmallVector<Shape> vpux::VPU::WorkloadSplitter::getPerClusterShapesWhenSOK(VPU::NCEOpInterface nceOp) {
    SmallVector<Shape> perClusterShapes = {};
    auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value() &&
        clusterOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusterOp->getResult(0).getType());
        auto numClusters = VPU::getOptimalNumClusters(clusterOp, outputType.getShape(),
                                                      VPU::MultiClusterStrategy::SplitOverKernel);
        auto distributedType = getDistributedOutputTypeFromOp(clusterOp, outputType, numClusters,
                                                              VPU::MultiClusterStrategy::SplitOverKernel);
        perClusterShapes = mlir::cast<vpux::VPU::DistributedTensorType>(distributedType.getDistributedTypes().front())
                                   .getPerClusterComputeShapes();
    }
    return perClusterShapes;
}

// Get a set containing all the channels from the workloads of the given NCE operations
mlir::DenseSet<int64_t> vpux::VPU::WorkloadSplitter::getWorkloadsChannels(
        const mlir::DenseSet<mlir::Operation*>& nceOps, bool skipLastWorkload) {
    SmallVector<VPU::DPUWorkloadOp> allWorkloads;
    mlir::DenseSet<int64_t> workloadsChannels;
    for (auto op : nceOps) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);
        const auto workloads = to_small_vector(nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>());
        allWorkloads.insert(allWorkloads.end(), workloads.begin(), workloads.end());
    }

    if (!allWorkloads.empty()) {
        if (skipLastWorkload) {
            allWorkloads.pop_back();
        }

        auto channels = to_container<mlir::DenseSet<int64_t>>(
                allWorkloads | transformed([](VPU::DPUWorkloadOp workload) -> int64_t {
                    const auto wlSizes = parseIntArrayAttr<int64_t>(workload.getOutSizes());
                    return wlSizes[Dims4D::Act::C.ind()];
                }));
        workloadsChannels.insert(channels.begin(), channels.end());
    } else {
        for (auto op : nceOps) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
            auto perClusterShapes = getPerClusterShapesWhenSOK(nceOp);

            if (!perClusterShapes.empty()) {
                auto channels = to_container<mlir::DenseSet<int64_t>>(perClusterShapes |
                                                                      transformed([](Shape clusterShape) -> int64_t {
                                                                          return clusterShape[Dims4D::Act::C];
                                                                      }));
                workloadsChannels.insert(channels.begin(), channels.end());
            } else {
                const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp.getOperation()->getResult(0).getType());
                const auto OC = outputType.getShape()[vpux::Dims4D::Act::C];
                workloadsChannels.insert(OC);
            }
        }
    }

    return workloadsChannels;
}

// Find the operations which can consume the given value. The value should be of sparse type, therefore the
// consumers can be NCE, Desparsify or Return ops
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitter::findConsumerOps(mlir::Value value) {
    mlir::DenseSet<mlir::Operation*> consumerOps;
    for (auto userOp : value.getUsers()) {
        auto taskOp = userOp;

        if (mlir::isa<VPU::NCEOpInterface, VPU::DesparsifyOp, mlir::func::ReturnOp>(taskOp)) {
            consumerOps.insert(userOp);
        } else if (mlir::isa<VPU::CopyOp>(taskOp) || VPU::isPureViewOp(taskOp)) {
            auto ops = findConsumerOps(userOp->getResult(0));
            consumerOps.insert(ops.begin(), ops.end());
        }
    }
    return consumerOps;
}

// Find all the NCE operations that produce the value. Multiple operations can produce a value in case it is
// concatenated or grouped
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitter::findProducerNCEOps(mlir::Value value) {
    mlir::DenseSet<mlir::Operation*> producerNCEOps;

    auto producerOp = value.getDefiningOp();
    auto taskOp = producerOp;

    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(taskOp)) {
        producerNCEOps.insert(nceOp);
    } else if (mlir::isa<VPU::CopyOp>(taskOp)) {
        const auto ops = findProducerNCEOps(producerOp->getOperand(0));
        producerNCEOps.insert(ops.begin(), ops.end());
    } else if (VPU::isPureViewOp(producerOp)) {
        if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(producerOp)) {
            for (const auto& input : concatOp.getInputs()) {
                const auto ops = findProducerNCEOps(input);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        } else if (auto viewOp = mlir::dyn_cast<VPU::ViewLikeOpInterface>(producerOp)) {
            const auto ops = findProducerNCEOps(viewOp->getOperand(0));
            producerNCEOps.insert(ops.begin(), ops.end());
        } else if (auto viewOp = mlir::dyn_cast<MultiViewOpInterface>(producerOp)) {
            if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) {
                const auto source = viewOp.getViewSource(opResult.getResultNumber());
                const auto ops = findProducerNCEOps(source);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        } else if (auto viewOp = mlir::dyn_cast<GroupedViewOpInterface>(producerOp)) {
            for (const auto& source : viewOp.getViewSources()) {
                const auto ops = findProducerNCEOps(source);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        }
    }

    return producerNCEOps;
}

// Find all the consumer operations of the value, then find all the producer NCE operations for the input values of
// the consumers. This is then repeated until no new consumers are identified. Chains of operations such as the
// following ensure that all three input Convolutions are returned when the function is called on the value marked
// with '*':
//   Conv   Conv   Conv
//     \*  /    \  /
//     Concat  Concat
//       |       |
//      Conv    Conv
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitter::findProducersForConsumers(
        mlir::Value value, mlir::DenseSet<mlir::Operation*> processedConsumerOps) {
    mlir::DenseSet<mlir::Operation*> producerNCEOps;

    auto consumerOps = findConsumerOps(value);
    for (auto consumerOp : consumerOps) {
        if (processedConsumerOps.contains(consumerOp)) {
            continue;
        }

        auto producerOps = findProducerNCEOps(consumerOp->getOperand(0));
        producerNCEOps.insert(producerOps.begin(), producerOps.end());

        if (mlir::isa<VPU::NCEEltwiseOp>(consumerOp)) {
            auto producerOpsInput2 = findProducerNCEOps(consumerOp->getOperand(1));
            producerNCEOps.insert(producerOpsInput2.begin(), producerOpsInput2.end());
        }
    }
    processedConsumerOps.insert(consumerOps.begin(), consumerOps.end());

    mlir::DenseSet<mlir::Operation*> newProducerOps;
    for (auto producerOp : producerNCEOps) {
        if (producerOp->getResult(0) == value) {
            continue;
        }
        auto producerOps = findProducersForConsumers(producerOp->getResult(0), processedConsumerOps);
        newProducerOps.insert(producerOps.begin(), producerOps.end());
    }

    producerNCEOps.insert(newProducerOps.begin(), newProducerOps.end());

    return producerNCEOps;
}

// Invariants that produce sparse activations must satisfy two conditions:
// - all variants must produce the same number of channels
// - the number of channels is a power of two (for NPU37XX)
// Additionally, in case a consumer operation has its input produced by multiple NCE operations,
// all of the producer ops need to have the same number of channels for their variants.
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitter::findInvalidSparseOps(
        VPU::NCEOpInterface nceOp, const VPU::SparsityConstraint& sparsityConstraint) {
    mlir::DenseSet<mlir::Operation*> invalidSparseOps;

    if (!mlir::isa<vpux::VPU::SparseTensorType>(nceOp->getResult(0).getType())) {
        return invalidSparseOps;
    }

    auto result = nceOp->getResult(0);
    auto producerOps = findProducersForConsumers(result);
    auto workloadsChannels = getWorkloadsChannels(producerOps);

    std::optional<int64_t> numChannels = std::nullopt;
    auto invalidWorkloads = llvm::any_of(workloadsChannels, [&](int64_t channels) -> bool {
        if (!numChannels.has_value()) {
            numChannels = channels;
        }
        if (channels != numChannels) {
            return true;
        }
        if (!sparsityConstraint.areChannelsFitForSESize(channels)) {
            return true;
        }
        return false;
    });
    if (invalidWorkloads) {
        invalidSparseOps.insert(producerOps.begin(), producerOps.end());
    }

    return invalidSparseOps;
}

// Depthwise operations must have variants that produce 16, 32 or 64 channels
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitter::findInvalidDepthwiseOps(
        const mlir::DenseSet<mlir::Operation*>& nceOps, ArrayRef<int64_t> supportedChannels) {
    mlir::DenseSet<mlir::Operation*> invalidDepthwiseOps;
    for (auto op : nceOps) {
        if (!isDepthwiseOp(op)) {
            continue;
        }
        const auto workloadsChannels = getWorkloadsChannels({op});
        const auto invalidChannels = llvm::any_of(workloadsChannels, [&](const int64_t channels) -> bool {
            return llvm::find(supportedChannels, channels) == supportedChannels.end();
        });
        if (invalidChannels) {
            invalidDepthwiseOps.insert(op);
        }
    }
    return invalidDepthwiseOps;
}

mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitter::findInvalidNCEPermuteOps(
        const mlir::DenseSet<mlir::Operation*>& nceOps) {
    mlir::DenseSet<mlir::Operation*> invalidNCEPermuteOps;
    for (auto op : nceOps) {
        if (!mlir::isa<VPU::NCEPermuteOp>(op)) {
            continue;
        }
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);
        const auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
        const auto nonZeroPadding = llvm::any_of(workloads, [&](VPU::DPUWorkloadOp workload) -> bool {
            const auto expandChannels = mlir::cast<VPU::NCEPermuteOp>(op).getExpandedChannels();
            const auto origInChannels =
                    mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getShape()[Dims4D::Act::C];
            const auto zeroPadding = expandChannels == origInChannels;
            const auto wlOffsets = parseIntArrayAttr<int64_t>(workload.getOutOffsetsAttr());
            const auto isZeroPredicate = [](const int64_t value) -> bool {
                return value == 0;
            };
            const bool zeroOffsets = std::all_of(wlOffsets.begin(), wlOffsets.end(), isZeroPredicate);
            return !zeroPadding || !zeroOffsets;
        });
        if (nonZeroPadding) {
            invalidNCEPermuteOps.insert(op);
        }
    }

    return invalidNCEPermuteOps;
}

/// @brief Get all of the supported channels that can be used to split all of the given workloads, so that the
/// depthwise and sparsity requirements are met.
/// @details For the normal case, workload channel can support [16, 32, ... 8192]. If it's a depthwise op, workload
/// channels only support [16, 32, 64]. That's the first part of this function to generate a pool of
/// supportedChannels that can be used to split all of the given workloads, based on it's depthwise or not. Then if
/// there is output sparsity, we will collect all the workload channels excluding the last one. And we will check each
/// element in supportedChannels. If all workload channels (excluding the last one) are divisible by this element, it's
/// a supported channel. Otherwise, we remove it from supportedChannels. For example,
/// 1. There's a convolution with 256 OC and output sparsity, split on 6 tiles generates [48, 48, 48, 48, 48, 16].
/// The original supportedChannels are [16, 32, ... 8192]. After filtering out unsupported channels due to sparsity,
/// supportedChannels will be just [16, 48].
/// 2. There's a DW conv with 256 OC and output sparsity, split on 6 tiles generates [48, 48, 48, 48, 48, 16]. The
/// original supportedChannels are [16, 32, 64]. After filtering out unsupported channels, supportedChannels will be
/// just [16].
SmallVector<int64_t> vpux::VPU::WorkloadSplitter::getSupportedChannels(
        const mlir::DenseSet<mlir::Operation*>& nceOps, const VPU::SparsityConstraint& sparsityConstraint) {
    SmallVector<int64_t> supportedChannels;

    const auto hasDepthwiseOp = llvm::any_of(nceOps, isDepthwiseOp);
    if (hasDepthwiseOp) {
        supportedChannels.insert(supportedChannels.end(), _supportedChannelsForDW.begin(),
                                 _supportedChannelsForDW.end());
    }

    const auto hasSparseOutput = llvm::any_of(nceOps, [](mlir::Operation* op) {
        return mlir::isa<vpux::VPU::SparseTensorType>(op->getResult(0).getType());
    });
    if (hasSparseOutput) {
        if (supportedChannels.empty()) {
            for (int64_t channels = VPU::NCEInvariant::VPU_DIMENSION_LIMIT; channels >= 16; channels -= 16) {
                if (!sparsityConstraint.areChannelsFitForSESize(channels)) {
                    continue;
                }
                supportedChannels.push_back(channels);
            }
        }

        auto eraseInvalidChannels = [&](const mlir::DenseSet<int64_t>& workloadsChannels) -> void {
            supportedChannels.erase(std::remove_if(supportedChannels.begin(), supportedChannels.end(),
                                                   [&](const int64_t channels) {
                                                       for (auto wlChannels : workloadsChannels) {
                                                           if (wlChannels % channels != 0) {
                                                               return true;
                                                           }
                                                       }
                                                       return false;
                                                   }),
                                    supportedChannels.end());
        };

        // When we need to check multiple NCE ops with output sparsity, the logic of filtering out unsupported
        // channels become complicated. We need to check for all the workload channels from all NCE ops excluding
        // the last workload from the last NCE op. There's not a simple way to tell which NCE op provides the last
        // workload. It's not safe to rely on IR order because for the senario where multiple NCE ops are connected
        // to the concat's inputs, their execution order are not determined. So if nceOps.size() > 1, just keep the
        // logic as before where we collect all the workload channels from all NCE ops.
        if (nceOps.size() > 1) {
            eraseInvalidChannels(getWorkloadsChannels(nceOps));
        } else if (!nceOps.empty()) {
            eraseInvalidChannels(getWorkloadsChannels(nceOps, true));
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(*(nceOps.begin()));
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp.getOperation()->getResult(0).getType());
            auto lastChannel = outputType.getShape()[vpux::Dims4D::Act::C];
            auto workloads = to_small_vector(nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>());

            if (!workloads.empty()) {
                const auto lastWorkloadSizes = parseIntArrayAttr<int64_t>(workloads.back().getOutSizes());
                lastChannel = lastWorkloadSizes[Dims4D::Act::C.ind()];

            } else {
                auto perClusterShapes = getPerClusterShapesWhenSOK(nceOp);
                if (!perClusterShapes.empty()) {
                    lastChannel = perClusterShapes.back()[Dims4D::Act::C];
                }
            }

            // If the last workload's channel can't be supported, we need to make sure it can be represented as
            // a combination of supported channels. For example, A DW Conv which has single workload with
            // channel 96 should be split to {64, 32} A DW Conv which has single workload with channel 112
            // should be split to {32, 32 ,32}
            // However if there are multiple workloads and the last workload's channel is smaller than the current
            // supported channel, we can skip the check because the last workload can have any channel size which is
            // smaller than the previous workloads. For example, for a Conv with output sparsity which has 256 output
            // channel, we first split across clusters as {96, 96, 64} with three tiles config, then we can skip the
            // check of 64 and keep 96 as a supported channel
            if (llvm::find(supportedChannels, lastChannel) == supportedChannels.end()) {
                supportedChannels.erase(
                        std::remove_if(supportedChannels.begin(), supportedChannels.end(),
                                       [&](const int64_t channels) {
                                           auto remainder = lastChannel % channels;
                                           auto remainderIsNotSupported =
                                                   remainder && (llvm::find(supportedChannels, remainder) ==
                                                                 supportedChannels.end());
                                           auto needToCheckRemainderFromLastChannel =
                                                   ((lastChannel > channels) || (workloads.size() == 1));
                                           return needToCheckRemainderFromLastChannel && remainderIsNotSupported;
                                       }),
                        supportedChannels.end());
            }
        }
    }

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(*nceOps.begin());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", *nceOps.begin());

    // Filter supported channels through errata conditions for small spatial compute DW ops
    const auto arch = config::getArch(nceOp);
    auto workloadSizeConstraint = VPU::getWorkloadSizeConstraint(arch);
    if (workloadSizeConstraint.doesDWOperationNeedWorkloadSplit(nceOp)) {
        supportedChannels = workloadSizeConstraint.getChannelsSupportedBySmallSpatialComputeDwOp(supportedChannels);
    }

    // In case the autopadding feature is used, the output channels might not be aligned to be a multiple of 16
    // If this happens, the current output channel configuration can be considered a supported workload configuration
    for (auto op : nceOps) {
        if (VPU::canAutopadOutput(op)) {
            const auto outputChannels =
                    mlir::cast<NDTypeInterface>(op->getResult(0).getType()).getShape()[Dims4D::Act::C];
            supportedChannels.push_back(outputChannels);
            break;
        }
    }

    const auto kernelSize = nceOp.getKernelSizeVal();
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];
    const auto kernelStride = nceOp.getStridesVal();
    const auto SX = kernelStride[Dims4D::Strides::X.ind()];
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp.getOperation()->getResult(0).getType());
    const auto OC = outputType.getShape()[vpux::Dims4D::Act::C];
    const auto hasSparseInput = mlir::isa<VPU::SparseTensorType>(nceOp->getOperand(0).getType());

    if (VPU::hasAnyChannelSupportedByKernelOptimization(*nceOps.begin(), supportedChannels, KX, SX) &&
        nceOps.size() == 1 && isDepthwiseOp(*nceOps.begin()) && !hasSparseInput && !hasSparseOutput) {
        SmallVector<int64_t> workloadsChannels = {OC};
        // Get a set containing all the channels from the workloads of the given NCE operation if workloads has created
        // in current phase
        auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
        if (!workloads.empty()) {  // Already owns workloads
            workloadsChannels = to_container<SmallVector<int64_t>>(
                    workloads | transformed([](VPU::DPUWorkloadOp workload) -> int64_t {
                        const auto wlSizes = parseIntArrayAttr<int64_t>(workload.getOutSizes());
                        return wlSizes[Dims4D::Act::C.ind()];
                    }));
        } else {  // No workloads split
            const auto getPerClusterShapes = [&]() {
                auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(*nceOps.begin());
                if (clusterOp == nullptr || !clusterOp.getMultiClusterStrategy().has_value()) {
                    return SmallVector<Shape>{outputType.getShape().raw()};
                }
                // multi cluster case
                auto strategy = clusterOp.getMultiClusterStrategy().value();
                auto numClusters = VPU::getOptimalNumClusters(clusterOp, outputType.getShape(), strategy);
                auto distributedType = mlir::cast<VPU::DistributedTensorType>(
                        getDistributedOutputTypeFromOp(clusterOp, outputType, numClusters, strategy)
                                .getDistributedTypes()
                                .front());
                return distributedType.getPerClusterComputeShapes();
            };

            const auto perClusterShapes = getPerClusterShapes();
            if (!perClusterShapes.empty()) {
                workloadsChannels = to_container<SmallVector<int64_t>>(
                        perClusterShapes | transformed([](ShapeRef clusterShape) -> int64_t {
                            return clusterShape[Dims4D::Act::C];
                        }));
            }
        }

        const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(nceOp);
        const auto channelsSupportedByKernelOptimization = VPU::getChannelsSupportedByKernelOptimization(
                *nceOps.begin(), workloadsChannels, static_cast<int64_t>(maxSlotsSum));

        if (!channelsSupportedByKernelOptimization.empty()) {
            supportedChannels = std::move(channelsSupportedByKernelOptimization);
        }
    }

    _log.trace("getSupportedChannels: supportedChannels {0} on {1} for nceOp {2}", supportedChannels,
               stringifyArchKind(config::ArchKind(config::getArch(*nceOps.begin()))), (*nceOps.begin())->getLoc());
    return supportedChannels;
}

// Splits the workload channels so that they are composed out of the values in the `supportedChannels` array, if it
// is provided. Additionally, removes the padding and spatial offsets from the workload based on the `removePadding`
// flag
void vpux::VPU::WorkloadSplitter::splitWorkload(VPU::DPUWorkloadOp dpuWorkloadOp, ArrayRef<int64_t> supportedChannels,
                                                const bool isInvalidNCEPermuteOp, int64_t channelPadding,
                                                bool isNCEPermuteOffsetsCorrectionNeeded, Logger log) {
    auto wlSizes = parseIntArrayAttr<int64_t>(dpuWorkloadOp.getOutSizesAttr());
    auto wlOffsets = parseIntArrayAttr<int64_t>(dpuWorkloadOp.getOutOffsetsAttr());
    auto padsAttr = dpuWorkloadOp.getPad();

    mlir::OpBuilder builder(dpuWorkloadOp);
    if (isInvalidNCEPermuteOp) {
        const auto pads = dpuWorkloadOp.getPad();
        const auto top = pads.getTop().getInt();
        const auto bottom = pads.getBottom().getInt();
        const auto left = pads.getLeft().getInt();
        const auto right = pads.getRight().getInt();
        wlSizes[Dims4D::Act::H.ind()] -= (top + bottom);
        wlSizes[Dims4D::Act::W.ind()] -= (left + right);
        wlSizes[Dims4D::Act::C.ind()] -= channelPadding;

        padsAttr = VPU::getPaddingAttr(pads.getContext(), PadInfo(0, 0, 0, 0));

        if (isNCEPermuteOffsetsCorrectionNeeded) {
            wlOffsets = SmallVector<int64_t>{0, 0, 0, 0};
        }
    }

    SmallVector<int64_t> newWorkloadChannels;
    auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
    if (supportedChannels.empty()) {
        newWorkloadChannels.push_back(wlChannels);
    } else {
        newWorkloadChannels = splitWorkloadChannel(wlChannels, supportedChannels);
        VPUX_THROW_WHEN(newWorkloadChannels.size() == 0,
                        "splitWorkloadChannel failed please check wlChannel - {0}, supportedChannelsDW - {1}",
                        wlChannels, supportedChannels);
    }

    auto channelOffset = wlOffsets[Dims4D::Act::C.ind()];

    for (auto channelSize : newWorkloadChannels) {
        auto sizes = wlSizes;
        sizes[Dims4D::Act::C.ind()] = channelSize;

        auto offsets = wlOffsets;
        offsets[Dims4D::Act::C.ind()] = channelOffset;
        channelOffset += channelSize;

        const auto offsetsAttr = getIntArrayAttr(builder.getContext(), offsets);
        const auto sizesAttr = getIntArrayAttr(builder.getContext(), sizes);

        builder.create<VPU::DPUWorkloadOp>(dpuWorkloadOp.getLoc(), offsetsAttr, sizesAttr, padsAttr,
                                           dpuWorkloadOp.getMpeModeAttr(), dpuWorkloadOp.getClusterIdAttr());
    }

    log.nest().trace("Split workload of size '{0}' into '{1}'", wlSizes[Dims4D::Act::C.ind()], newWorkloadChannels);
    dpuWorkloadOp.erase();
}
