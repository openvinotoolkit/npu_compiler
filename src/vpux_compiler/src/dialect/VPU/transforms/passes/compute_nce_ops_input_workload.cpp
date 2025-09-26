//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_COMPUTENCEINPUTWORKLOADS
#define GEN_PASS_DEF_COMPUTENCEINPUTWORKLOADS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

int64_t getInputWorkloadStartCh(NCEOpInterface nceOp, const int64_t outputStartCh) {
    return llvm::TypeSwitch<mlir::Operation*, int64_t>(nceOp.getOperation())
            .Case<NCEConvolutionOp, NCECompressConvolutionOp, NCEInterpolateOp, NCEReduceOp>(
                    [&](mlir::Operation* /*op*/) {
                        return 0;
                    })
            .Case<NCEEltwiseOp>([&](mlir::Operation* /*op*/) {
                return outputStartCh;
            })
            .Case<NCEDepthConvolutionOp, NCEMaxPoolOp, NCEAveragePoolOp>([&](mlir::Operation* /*op*/) {
                return outputStartCh;
            })
            .Case<NCEPermuteOp>([&](mlir::Operation* /*op*/) {
                return outputStartCh;
            })
            .Case<NCEMatMulOp>([&](mlir::Operation* /*op*/) {
                return 0;
            })
            .Default([&](mlir::Operation* /*op*/) -> int64_t {
                VPUX_THROW("Unsupported operation type: {0}", nceOp);
            });
}

int64_t getInputWorkloadSizeCh(NCEOpInterface nceOp, const int64_t outputSizeCh, const int64_t outputStartCh,
                               const int64_t fullInputChannels) {
    return llvm::TypeSwitch<mlir::Operation*, int64_t>(nceOp.getOperation())
            .Case<NCEConvolutionOp, NCEInterpolateOp, NCEReduceOp>([&](mlir::Operation* /*op*/) {
                return fullInputChannels;
            })
            .Case<NCEEltwiseOp>([&](mlir::Operation* op) {
                return !VPU::canAutopadOutput(op) && fullInputChannels != outputSizeCh ? outputSizeCh
                                                                                       : fullInputChannels;
            })
            .Case<NCEDepthConvolutionOp, NCEMaxPoolOp, NCEAveragePoolOp>([&](mlir::Operation* op) {
                if (auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
                    const auto inChannelAlignment = alignedOp.getInputChannelAlignment();
                    if (outputSizeCh % inChannelAlignment == 0) {
                        return outputSizeCh;
                    }
                    if (fullInputChannels % inChannelAlignment == 0) {
                        return fullInputChannels;
                    }
                    VPUX_THROW("Cannot compute input workload channel size for op '{0}' at '{1}': workload output "
                               "channel size is {2}, full input channel size is {3}",
                               op->getName(), op->getLoc(), outputSizeCh, fullInputChannels);
                }
                return outputSizeCh;
            })
            .Case<NCECompressConvolutionOp>([&](mlir::Operation* /*op*/) {
                return VPU::NCEInvariant::getAlignment(
                        mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getElementType());
            })
            .Case<NCEPermuteOp>([&](mlir::Operation* /*op*/) {
                if ((outputStartCh + outputSizeCh) > fullInputChannels) {
                    return fullInputChannels - outputStartCh;
                }
                return outputSizeCh;
            })
            .Case<NCEMatMulOp>([&](mlir::Operation* /*op*/) {
                return fullInputChannels;
            })
            .Default([&](mlir::Operation* /*op*/) -> int64_t {
                VPUX_THROW("Unsupported operation type: {0}", nceOp);
            });
}

SmallVector<Shape> getInputOffsetsPerCluster(NCEOpInterface nceOp, Logger log) {
    auto input = nceOp->getOperand(0);

    auto inputTypes = mlir::dyn_cast<VPU::DistributedTypeInterface>(input.getType());
    if (inputTypes == nullptr) {
        return {};
    }
    if (!inputTypes.containsDistributedTypes()) {
        return {};
    }

    auto distributedInType = mlir::cast<vpux::VPU::DistributedTensorType>(*inputTypes.getDistributedTypes().begin());
    if (auto inputSparseType = mlir::dyn_cast<VPU::SparseTensorType>(inputTypes)) {
        auto effectiveSparseOutputType = getEffectiveSparseOutputType(inputSparseType);
        distributedInType = mlir::cast<vpux::VPU::DistributedTensorType>(effectiveSparseOutputType);
    }
    auto inputDistrAttr = distributedInType.getDistribution();
    auto modeInput = inputDistrAttr.getMode().getValue();

    const auto output = nceOp->getResult(0);
    const auto outputTypes = mlir::dyn_cast<VPU::DistributedTypeInterface>(output.getType());
    VPUX_THROW_WHEN(outputTypes == nullptr || !outputTypes.containsDistributedTypes(),
                    "nceOp has distributed type input and non distributed output type.");

    auto distributedOutputType =
            mlir::cast<vpux::VPU::DistributedTensorType>(*outputTypes.getDistributedTypes().begin());
    const auto outputDistrAttr = distributedOutputType.getDistribution();
    const auto outputPerClusterOffsetInFullTensor = distributedOutputType.getPerClusterMemoryShapeOffsets();
    const auto modeOutput = outputDistrAttr.getMode().getValue();

    // Offsets in full input tensor for each cluster
    const auto inputPerClusterOffsetInFullTensor = distributedInType.getPerClusterMemoryShapeOffsets();

    // Get the correction necessary to get offsets relative to the slice of data found in each cluster
    SmallVector<Shape> perClusterOffsetsAdjustment(inputDistrAttr.getNumClusters().getInt(),
                                                   Shape(distributedInType.getShape().size(), 0));

    const auto isDWOp =
            mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(nceOp.getOperation());
    const auto isELTOp = mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation());

    auto clusterHasFullTensor = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
               VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
    };

    const auto hasFullInput = clusterHasFullTensor(modeInput);
    const auto producesFullOutput = clusterHasFullTensor(modeOutput);

    const auto adjustmentNeededOnH = modeInput == VPU::DistributionMode::OVERLAPPED;
    const auto adjustmentNeededOnW = modeInput == VPU::DistributionMode::OVERLAPPED;
    const auto adjustmentNeededOnC = ((isDWOp && hasFullInput && VPU::isSegmentedOverC(outputDistrAttr)) ||
                                      (isDWOp && producesFullOutput && VPU::isSegmentedOverC(inputDistrAttr)) ||
                                      (!isDWOp && !isELTOp && VPU::isSegmentedOverC(outputDistrAttr)));
    if (!adjustmentNeededOnH && !adjustmentNeededOnW && !adjustmentNeededOnC) {
        // no adjustment needed
        return perClusterOffsetsAdjustment;
    }

    for (size_t clusterIdx = 0; clusterIdx < perClusterOffsetsAdjustment.size(); clusterIdx++) {
        const auto offsetsInFullTensor = inputPerClusterOffsetInFullTensor[clusterIdx];
        if (adjustmentNeededOnH) {
            perClusterOffsetsAdjustment[clusterIdx][Dims4D::Act::H] = -offsetsInFullTensor[Dims4D::Act::H];
        }
        if (adjustmentNeededOnW) {
            perClusterOffsetsAdjustment[clusterIdx][Dims4D::Act::W] = -offsetsInFullTensor[Dims4D::Act::W];
        }
        if (adjustmentNeededOnC) {
            if (isDWOp && hasFullInput && VPU::isSegmentedOverC(outputDistrAttr)) {
                // DW operations: DUP -> SEG
                // In this case output workloads start from 0 in each cluster and this offset backinferred to input
                // offsets. Since input is duplicated then it does not contain any memory offsets, take them from output
                // distributed tensor.
                perClusterOffsetsAdjustment[clusterIdx][Dims4D::Act::C] =
                        outputPerClusterOffsetInFullTensor[clusterIdx][Dims4D::Act::C];
                continue;
            }

            perClusterOffsetsAdjustment[clusterIdx][Dims4D::Act::C] = -offsetsInFullTensor[Dims4D::Act::C];
        }
    }

    log.trace("Per cluster offsets correction = {0}, distributed type  = {1}", perClusterOffsetsAdjustment,
              distributedInType);

    return perClusterOffsetsAdjustment;
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>> compute5DInputWorkload(NCEOpInterface nceOp,
                                                                             DPUWorkloadOp dpuTaskOp, Logger log) {
    const auto outWlStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutOffsets());
    const auto outWlSize = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutSizes());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    auto fullInputShape = inputType.getShape();
    if (auto inputSparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(inputType)) {
        auto effectiveSparseOutputType = getEffectiveSparseOutputType(inputSparseType);
        fullInputShape = mlir::cast<NDTypeInterface>(effectiveSparseOutputType).getShape();
    }

    const auto fullInputChannels = fullInputShape[DimsGroups5D::Act::C];
    const auto fullInputHeight = fullInputShape[DimsGroups5D::Act::H];
    const auto fullInputWidth = fullInputShape[DimsGroups5D::Act::W];

    const auto kernelSz = nceOp.getKernelSizeVal();
    const auto strides = nceOp.getStridesVal();

    const auto padding = nceOp.getPad();
    const auto kernelPadTop = padding.getTop().getInt();
    const auto kernelPadLeft = padding.getLeft().getInt();
    const auto kernelPadBottom = padding.getBottom().getInt();
    const auto kernelPadRight = padding.getRight().getInt();

    const DimRange outHeightTile(outWlStart[DimsGroups5D::Act::H.ind()],
                                 outWlStart[DimsGroups5D::Act::H.ind()] + outWlSize[DimsGroups5D::Act::H.ind()]);

    auto [inHeightTile, heightBefore, heightAfter] = vpux::inputForOutputDim(
            outHeightTile, kernelSz[DimsGroups5D::Kernel::Y.ind()], strides[DimsGroups5D::Kernel::Y.ind()],
            {0, fullInputHeight}, kernelPadTop, kernelPadBottom);

    const DimRange outWidthTile(outWlStart[DimsGroups5D::Act::W.ind()],
                                outWlStart[DimsGroups5D::Act::W.ind()] + outWlSize[DimsGroups5D::Act::W.ind()]);
    auto [inWidthTile, heightWidthBefore, heightWidthAfter] = vpux::inputForOutputDim(
            outWidthTile, kernelSz[DimsGroups5D::Kernel::X.ind()], strides[DimsGroups5D::Kernel::X.ind()],
            {0, fullInputWidth}, kernelPadLeft, kernelPadRight);

    const auto inStartWidth = inWidthTile.begin;
    const auto inStartHeight = inHeightTile.begin;
    const auto inStartChannels = getInputWorkloadStartCh(nceOp, outWlStart[DimsGroups5D::Act::C.ind()]);

    const auto start =
            SmallVector<int64_t>{outWlStart[DimsGroups5D::Act::G.ind()], outWlStart[DimsGroups5D::Act::N.ind()],
                                 inStartChannels, inStartHeight, inStartWidth};

    const auto inSizeWidth = inWidthTile.end - inWidthTile.begin;
    const auto inSizeHeight = inHeightTile.end - inHeightTile.begin;
    const auto inSizeChannels = getInputWorkloadSizeCh(nceOp, outWlSize[DimsGroups5D::Act::C.ind()],
                                                       outWlStart[DimsGroups5D::Act::C.ind()], fullInputChannels);
    VPUX_THROW_WHEN((mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation())) && (inStartWidth != 0) &&
                            (inSizeWidth != fullInputWidth),
                    "HW Permute does not support workload segmentation over W. Input workload start = {0}, Input "
                    "workload size = {1}",
                    inStartWidth, inSizeWidth);

    const auto size = SmallVector<int64_t>{outWlSize[DimsGroups5D::Act::G.ind()], outWlSize[DimsGroups5D::Act::N.ind()],
                                           inSizeChannels, inSizeHeight, inSizeWidth};

    log.trace("Computed input workload start/end for operation '{0}': start = {1}, size = {2}", dpuTaskOp.getLoc(),
              start, size);

    return std::make_pair(start, size);
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>> computeInputWorkload(NCEOpInterface nceOp,
                                                                           DPUWorkloadOp dpuTaskOp, Logger log) {
    const auto outWlStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutOffsets());
    if (outWlStart.size() == DimsGroups5D::Act::numDims) {
        return compute5DInputWorkload(nceOp, dpuTaskOp, log);
    }

    const auto outWlSize = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutSizes());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    auto fullInputShape = inputType.getShape();
    if (auto inputSparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(inputType)) {
        auto effectiveSparseOutputType = getEffectiveSparseOutputType(inputSparseType);
        fullInputShape = mlir::cast<NDTypeInterface>(effectiveSparseOutputType).getShape();
    }

    const auto fullInputChannels = fullInputShape[Dims4D::Act::C];
    const auto fullInputHeight = fullInputShape[Dims4D::Act::H];
    const auto fullInputWidth = fullInputShape[Dims4D::Act::W];

    const auto kernelSz = nceOp.getKernelSizeVal();
    const auto strides = nceOp.getStridesVal();

    const auto padding = nceOp.getPad();
    const auto kernelPadTop = padding.getTop().getInt();
    const auto kernelPadLeft = padding.getLeft().getInt();
    const auto kernelPadBottom = padding.getBottom().getInt();
    const auto kernelPadRight = padding.getRight().getInt();

    DimRange inHeightTile(0, 0);
    const DimRange outHeightTile(outWlStart[Dims4D::Act::H.ind()],
                                 outWlStart[Dims4D::Act::H.ind()] + outWlSize[Dims4D::Act::H.ind()]);
    std::tie(inHeightTile, std::ignore, std::ignore) =
            vpux::inputForOutputDim(outHeightTile, kernelSz[Dims4D::Kernel::Y.ind()], strides[Dims4D::Kernel::Y.ind()],
                                    {0, fullInputHeight}, kernelPadTop, kernelPadBottom);

    DimRange inWidthTile(0, 0);
    const DimRange outWidthTile(outWlStart[Dims4D::Act::W.ind()],
                                outWlStart[Dims4D::Act::W.ind()] + outWlSize[Dims4D::Act::W.ind()]);
    std::tie(inWidthTile, std::ignore, std::ignore) =
            vpux::inputForOutputDim(outWidthTile, kernelSz[Dims4D::Kernel::X.ind()], strides[Dims4D::Kernel::X.ind()],
                                    {0, fullInputWidth}, kernelPadLeft, kernelPadRight);

    const auto inStartWidth = inWidthTile.begin;
    const auto inStartHeight = inHeightTile.begin;
    const auto inStartChannels = getInputWorkloadStartCh(nceOp, outWlStart[Dims4D::Act::C.ind()]);

    const auto start =
            SmallVector<int64_t>{outWlStart[Dims4D::Act::N.ind()], inStartChannels, inStartHeight, inStartWidth};

    const auto inSizeWidth = inWidthTile.end - inWidthTile.begin;
    const auto inSizeHeight = inHeightTile.end - inHeightTile.begin;
    const auto inSizeChannels = getInputWorkloadSizeCh(nceOp, outWlSize[Dims4D::Act::C.ind()],
                                                       outWlStart[Dims4D::Act::C.ind()], fullInputChannels);
    VPUX_THROW_WHEN((mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation())) && (inStartWidth != 0) &&
                            (inSizeWidth != fullInputWidth),
                    "HW Permute does not support workload segmentation over W. Input workload start = {0}, Input "
                    "workload size = {1}",
                    inStartWidth, inSizeWidth);

    const auto size = SmallVector<int64_t>{outWlSize[Dims4D::Act::N.ind()], inSizeChannels, inSizeHeight, inSizeWidth};

    log.trace("Computed input workload start/end for operation '{0}': start = {1}, size = {2}", dpuTaskOp.getLoc(),
              start, size);

    return std::make_pair(start, size);
}

//
// CorrectNCEWorkloads
//

class ComputeNCEInputWorkloadsPass final :
        public VPU::impl::ComputeNCEInputWorkloadsBase<ComputeNCEInputWorkloadsPass> {
public:
    explicit ComputeNCEInputWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ComputeNCEInputWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();

    // TODO: Add support for VPUX37XX input workloads computation
    // Ticket: E#63055

    func.walk([&](NCEOpInterface nceOp) {
        const auto perClusterStartOffsets = getInputOffsetsPerCluster(nceOp, _log);
        auto workloads = nceOp.getWorkloads().getOps<DPUWorkloadOp>();
        for (auto workload : llvm::make_early_inc_range(workloads)) {
            if (!workload.getInOffsets().has_value() || !workload.getInSizes().has_value()) {
                mlir::OpBuilder builder(workload);
                SmallVector<int64_t> inStart = {};
                SmallVector<int64_t> inSize = {};
                std::tie(inStart, inSize) = computeInputWorkload(nceOp, workload, _log);

                if (!perClusterStartOffsets.empty()) {
                    VPUX_THROW_UNLESS(workload.getClusterId().has_value(),
                                      "DPUWorkload should have cluster_id set: {0}", workload);

                    const auto globalInputOffsets = perClusterStartOffsets[workload.getClusterId().value()];
                    inStart[Dims4D::Act::C.ind()] += globalInputOffsets[Dims4D::Act::C];
                    inStart[Dims4D::Act::H.ind()] += globalInputOffsets[Dims4D::Act::H];
                    inStart[Dims4D::Act::W.ind()] += globalInputOffsets[Dims4D::Act::W];

                    VPUX_THROW_UNLESS(
                            inStart[Dims4D::Act::H.ind()] >= 0 && inStart[Dims4D::Act::W.ind()] >= 0,
                            "An NCEClusterTaskOp must not need more overlap lines/columns than provided by the "
                            "DistributedType of its input, workload = {0}, op = {1}",
                            workload, nceOp);
                    VPUX_THROW_UNLESS(inStart[Dims4D::Act::C.ind()] >= 0,
                                      "An NCEClusterTaskOp has negative input workload channel offset, workload = {0}, "
                                      "op = {1}, inStart C = {2}, offset was {3}",
                                      workload, nceOp, inStart[Dims4D::Act::C.ind()],
                                      globalInputOffsets[Dims4D::Act::C]);
                }

                const auto inStartAttr = getIntArrayAttr(builder.getContext(), inStart);
                const auto inSizeAttr = getIntArrayAttr(builder.getContext(), inSize);

                auto workloadWithInputOffsets = builder.create<DPUWorkloadOp>(
                        workload.getLoc(), workload.getOutOffsets(), workload.getOutSizes(), inStartAttr, inSizeAttr,
                        workload.getPad(), workload.getMpeMode(), workload.getClusterIdAttr());

                _log.trace("DpuTaskOp '{0}' replaced with {1}", workload, workloadWithInputOffsets);

                workload.erase();
            }
        }
    });
}

}  // namespace

//
// createComputeNCEInputWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createComputeNCEInputWorkloadsPass(Logger log) {
    return std::make_unique<ComputeNCEInputWorkloadsPass>(log);
}
