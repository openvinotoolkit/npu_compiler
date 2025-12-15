//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <cstdint>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTNCEINTERPOLATETODW
#define GEN_PASS_DEF_CONVERTNCEINTERPOLATETODW
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

/*
    SEP + DW.Conv cannot have start_z of a workload != 0. As such, we can only have 1 workload per cluster and that
    workload must have channels in DEPTHWISE_WORKLOAD_SIZES values. If that requirement is not met, tiling needs to
    be added to legalize the workloads. Tiling pipeline takes care of this, but the increased number of ops
    may lead to performance regressions. Therefore, this transformation is not always beneficial to apply.

    Based on experimental runs and attempting to minimize the chance of perf regressions as a consequence of
    this decision, the following conditions must be met to convert NCE.Interpolate from Conv to DW.Conv:
    * if single cluster op -> execute as DW.Conv if num channels of op is in DEPTHWISE_WORKLOAD_SIZES and L1aOpt
                              conditions are met;
    * if multi cluster op -> execute as DW.Conv if:
       - num channels of op is in DEPTHWISE_WORKLOAD_SIZES and L1aOpt conditions are met;
       - more than 3 (experimental number) tiles on C are needed for SEP DW.Conv to have per cluster workloads with
         channels in DEPTHWISE_WORKLOAD_SIZES that also meet L1aOpt workload reqs.
*/
bool isDepthwiseConvMorePerformant(VPU::NCEInterpolateOp origOp, config::ArchKind arch, Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto numChannels = outputType.getShape()[Dims4D::Act::C];
    const auto inputType = mlir::cast<VPU::SparseTensorType>(origOp.getInput().getType());
    const auto elemType = inputType.getElementType();

    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(origOp.getOperation());
    const auto kernelSize = nceOpInterface.getKernelSizeVal();
    const auto kernelStride = nceOpInterface.getStridesVal();
    const auto pads = nceOpInterface.getPad();

    if (mlir::isa<VPU::SparseTensorType>(outputType)) {
        log.trace("Output is sparse.");
        return false;
    }

    auto isSupportedSEPDwWorkloadSize = [](const int64_t channelSz) -> bool {
        return llvm::find(VPU::NCEInvariant::DEPTHWISE_WORKLOAD_SIZES, channelSz) !=
               VPU::NCEInvariant::DEPTHWISE_WORKLOAD_SIZES.end();
    };

    auto multiclusterStrategy = origOp.getMultiClusterStrategyAttr();
    if (multiclusterStrategy == nullptr) {
        if (!isSupportedSEPDwWorkloadSize(numChannels)) {
            log.trace("Channel size is not supported for SEP DW.Conv.");
            return false;
        }

        log.trace("Checking if single cluster Interpolate can have DW L1aOpt applied.");
        return VPU::NCEInvariant::doesWorkloadSupportSmallKernelOpt(
                arch, kernelSize[Dims4D::Kernel::X.ind()], kernelStride[Dims4D::Kernel::X.ind()],
                outputType.getShape().raw(), elemType.isF16(), kernelSize[Dims4D::Kernel::Y.ind()],
                pads.getLeft().getInt());
    }

    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    const auto numClusters = VPU::getOptimalNumClusters(origOp, outputType.getShape(), multiclusterStrategy.getValue());
    auto outDistributedTypeIf = getDistributedOutputTypeFromOp(
            clusteredOp, outputType, numClusters, multiclusterStrategy.getValue(), {}, TileInfo(ShapeRef()), true);

    const auto distribution = mlir::cast<VPU::DistributedTensorType>(outDistributedTypeIf.getDistributedTypes().front())
                                      .getDistribution();
    auto perClusterPadding = [&]() -> SmallVector<PadInfo> {
        const auto padsInfo = toPadInfo(pads);
        if (distribution.getMode().getValue() == VPU::DistributionMode::OVERLAPPED) {
            return VPU::getPerClusterPadding(distribution, padsInfo);
        }

        return SmallVector<PadInfo>(numClusters, padsInfo);
    }();

    Shape outputShape(outputType.getShape());
    auto isOptimalWithTiling = [&](SmallVector<SmallVector<int64_t>> perClusterOutWorkloads, const bool isSOK) -> bool {
        constexpr int64_t MAX_ALLOWED_TILING = 3;
        log.trace("Attempting to find tiling such that DW L1aOpt can be applied to the resulting workloads.");
        for (int64_t divisor = 2; divisor <= MAX_ALLOWED_TILING; divisor++) {
            const auto supportedChannelTiling =
                    vpux::divideChannelForSEPDWConv(origOp.getOperation(), numChannels, divisor);
            if (supportedChannelTiling.empty()) {
                log.nest().trace("Reject tiling with with channel divisor {0}; cannot divide into supporte chanels for "
                                 "SEP DW.Conv",
                                 divisor);
                continue;
            }

            bool doAllWorkloadsSupportSmallKernelOpt = true;
            for (const auto tileChannel : supportedChannelTiling) {
                if (isSOK) {
                    // divide tiling channel size into clusters and ensure SEP DW workload restrictions are still met
                    outputShape[Dims4D::Act::C] = tileChannel;
                    auto overlapParams = VPU::getSupportedPerClusterShapesAndOffsetsForSEPDWConv(
                            clusteredOp, outputShape, numClusters, Dims4D::Act::C, false);
                    if (mlir::failed(overlapParams)) {
                        log.nest().trace("Reject tiling with with channel divisor {0}; SOK no longer supported.",
                                         divisor);
                        doAllWorkloadsSupportSmallKernelOpt = false;
                        break;
                    }

                    perClusterOutWorkloads = overlapParams.value().getComputeShapes();
                } else {
                    // For non-SOK strategy, adapt per cluster workload with the new channel size, post-tiling
                    std::transform(perClusterOutWorkloads.begin(), perClusterOutWorkloads.end(),
                                   perClusterOutWorkloads.begin(), [&](ArrayRef<int64_t> workload) {
                                       SmallVector<int64_t> tiledWorkload(workload);
                                       tiledWorkload[Dims4D::Act::C.ind()] = tileChannel;
                                       return tiledWorkload;
                                   });
                }

                const bool isSmallOpt =
                        llvm::all_of(zip(perClusterOutWorkloads, perClusterPadding), [&](auto clusterItem) {
                            const auto workload = std::get<0>(clusterItem);
                            const auto padding = std::get<1>(clusterItem);
                            return VPU::NCEInvariant::doesWorkloadSupportSmallKernelOpt(
                                    arch, kernelSize[Dims4D::Kernel::X.ind()], kernelStride[Dims4D::Kernel::X.ind()],
                                    workload, elemType.isF16(), kernelSize[Dims4D::Kernel::Y.ind()], padding.left);
                        });

                if (!isSmallOpt) {
                    doAllWorkloadsSupportSmallKernelOpt = false;
                    log.nest().trace(
                            "Reject tiling with with channel divisor {0}; not every workload can support L1aOpt "
                            "for DW.Conv.",
                            divisor);
                    break;
                }
            }

            if (doAllWorkloadsSupportSmallKernelOpt) {
                log.nest().trace("Found possible supported tiling with channel divisor {0}", divisor);
                return true;
            }
        }

        log.trace("Need more than {0} tiles to run as DW.Conv.", MAX_ALLOWED_TILING);
        return false;
    };

    if (multiclusterStrategy.getValue() != vpux::VPU::MultiClusterStrategy::SplitOverKernel) {
        auto perClusterOutWorkloads = parseIntArrayOfArrayAttr<int64_t>(distribution.getComputeShapes());

        if (isSupportedSEPDwWorkloadSize(numChannels)) {
            log.trace("Non-SOK Interpolate has supported channels, checking if DW L1aopt can be applied.");
            return llvm::all_of(zip(perClusterOutWorkloads, perClusterPadding), [&](auto clusterItem) {
                const auto workload = std::get<0>(clusterItem);
                const auto padding = std::get<1>(clusterItem);
                return VPU::NCEInvariant::doesWorkloadSupportSmallKernelOpt(
                        arch, kernelSize[Dims4D::Kernel::X.ind()], kernelStride[Dims4D::Kernel::X.ind()], workload,
                        elemType.isF16(), kernelSize[Dims4D::Kernel::Y.ind()], padding.left);
            });
        }

        return isOptimalWithTiling(std::move(perClusterOutWorkloads), false);
    }

    auto overlapParams = VPU::getSupportedPerClusterShapesAndOffsetsForSEPDWConv(clusteredOp, outputShape, numClusters,
                                                                                 Dims4D::Act::C, false);
    if (mlir::succeeded(overlapParams)) {
        log.trace("Got per cluster distribution for SOK Interpolate such that each has supported channels for DW; "
                  "checking if DW L1aopt can be applied.");
        return llvm::all_of(overlapParams.value().getComputeShapes(), [&](auto workload) {
            return VPU::NCEInvariant::doesWorkloadSupportSmallKernelOpt(
                    arch, kernelSize[Dims4D::Kernel::X.ind()], kernelStride[Dims4D::Kernel::X.ind()], workload,
                    elemType.isF16(), kernelSize[Dims4D::Kernel::Y.ind()], pads.getLeft().getInt());
        });
    }

    return isOptimalWithTiling(overlapParams.value().getComputeShapes(), true);
}

mlir::Value createSparseInput(mlir::OpBuilder builder, VPU::NCEInterpolateOp origOp, mlir::Value data,
                              mlir::Value sparsityMap, VPU::StorageElementTableOp origSeTableOp) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(data.getType());
    auto inputShape = inputType.getShape();

    if (origOp.getMultiClusterStrategy() != VPU::MultiClusterStrategy::SplitOverKernel) {
        return origOp.getInput();
    }

    // For SOK ops, SETable op will be SEGMENTED along the depth dimension.
    // As such we must ensure seDepth >= numClusters, so we can divide the depth dimension into
    // clusters. It is not enough to have seDepth == numClusters, due to the Tiling passes.
    // If tiling decides to slice along the channel dimension, the resulting SETable tensor
    // must be able to be SEGMENTED on channel into the n clusters.
    // That is why, here we assign maximum seDepth possible for this channel size. After tiling,
    // the depth will be corrected for each individual op to result in 1 depth per cluster.
    // The downside is that Tiling pipeline will see more required memory than strictly necessary.
    const int64_t seSize = 16;
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    const auto seSizeArr = SmallVector<int64_t>(seDepth, seSize);
    auto seTableOp = builder.create<VPU::StorageElementTableOp>(
            origSeTableOp->getLoc(), origSeTableOp.getDataShapeAttr(), origSeTableOp.getDataElemTypeAttr(),
            getIntArrayAttr(builder, seSizeArr), getIntAttr(builder, seDepth), origSeTableOp.getSeAttr().value(),
            nullptr, nullptr);

    auto groupOp = builder.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), data, sparsityMap, seTableOp.getOutput(),
                                                            origSeTableOp.getSeAttr().value());
    return groupOp.getOutput();
}

mlir::Value createWeightsConstant(mlir::OpBuilder builder, Const::DeclareOp weights,
                                  VPU::StorageElementTableOp origSeTableOp) {
    auto ctx = builder.getContext();
    auto convWeightsType = mlir::cast<vpux::NDTypeInterface>(weights.getOutput().getType());
    auto convWeightsShape = convWeightsType.getShape();
    auto dwConvWeightShape = Shape({convWeightsShape[Dims4D::Filter::OC], 1, convWeightsShape[Dims4D::Filter::KY],
                                    convWeightsShape[Dims4D::Filter::KX]});

    const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(dwConvWeightShape.raw(), convWeightsType.getElementType(), tensorAttr));
    const auto order = weightsType.getDimsOrder();

    const auto weightsNumElems = weightsType.getNumElements();

    auto interpAttr = mlir::cast<VPU::SEInterpolateAttr>(origSeTableOp.getSeAttr().value());
    const auto mode = interpAttr.getMode().getValue();
    const auto coordMode = interpAttr.getCoordinateTransformationMode();
    const auto scales = parseFPArrayAttr<double>(interpAttr.getScale());

    auto quantElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(convWeightsType.getElementType());
    const double scale = quantElemType == nullptr ? 1 : quantElemType.getScale();
    auto kernel = VPU::getNCEInterpolateKernelContent(
            {convWeightsShape[Dims4D::Filter::KY], convWeightsShape[Dims4D::Filter::KX]}, mode, coordMode.getValue(),
            scales);

    if (!isDoubleEqual(scale, 1)) {
        std::transform(kernel.begin(), kernel.end(), kernel.begin(), [&](const float value) -> float {
            return static_cast<float>(value / scale);
        });
    }

    SmallVector<float> content(weightsNumElems, 0.0f);
    const auto kernelSizeCount = dwConvWeightShape[Dims4D::Filter::KY] * dwConvWeightShape[Dims4D::Filter::KX];
    const auto eachWeightSizeCount = dwConvWeightShape[Dims4D::Filter::IC] * kernelSizeCount;
    loop_2d(LoopExecPolicy::Parallel, ctx, convWeightsShape[Dims4D::Filter::OC], kernelSizeCount,
            [&](int64_t channelIdx, int64_t kernelSizeIdx) {
                const auto contentIdx = channelIdx * eachWeightSizeCount + kernelSizeIdx;
                content[contentIdx] = kernel[kernelSizeIdx];
            });

    const auto dataStorageType = mlir::RankedTensorType::get(dwConvWeightShape.raw(), mlir::Float32Type::get(ctx));

    const auto dataAttr = Const::createConstContent(dataStorageType, ArrayRef(content));

    Const::ContentSetup contentAttrSetup(dataStorageType);

    if (const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(convWeightsType.getElementType())) {
        contentAttrSetup = contentAttrSetup.castElemType(qElemType);
    } else if (mlir::isa<mlir::Float16Type>(convWeightsType.getElementType())) {
        contentAttrSetup = contentAttrSetup.castElemType(mlir::Float16Type::get(ctx));
    }
    if (order != DimsOrder::fromNumDims(dwConvWeightShape.size())) {
        contentAttrSetup = contentAttrSetup.reorder(order);
    }

    auto weightsConstOp = builder.create<Const::DeclareOp>(weights->getLoc(), weightsType,
                                                           Const::ContentAttr::get(dataAttr, contentAttrSetup));

    return VPU::alignDepthWiseWeightsTensor(builder, weights.getLoc(), weightsConstOp.getOutput());
}

//
// ConvertNCEInterpolateToDWPass
//

class ConvertNCEInterpolateToDWPass final :
        public VPU::impl::ConvertNCEInterpolateToDWBase<ConvertNCEInterpolateToDWPass> {
public:
    explicit ConvertNCEInterpolateToDWPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void convertToDWConv(VPU::NCEInterpolateOp origOp, VPU::GroupSparseTensorOp groupSparseOp,
                         VPU::StorageElementTableOp storageElementTable, Const::DeclareOp origWeights,
                         config::ArchKind arch) const;
};

void ConvertNCEInterpolateToDWPass::convertToDWConv(VPU::NCEInterpolateOp origOp,
                                                    VPU::GroupSparseTensorOp groupSparseOp,
                                                    VPU::StorageElementTableOp storageElementTable,
                                                    Const::DeclareOp origWeights, config::ArchKind arch) const {
    auto nestedLog = _log.nest();
    mlir::OpBuilder builder(origOp);
    auto data = groupSparseOp.getData();
    auto sparsityMap = groupSparseOp.getSparsityMap();

    const auto sparseInput = createSparseInput(builder, origOp, data, sparsityMap, storageElementTable);
    nestedLog.trace("New sparse input = {0}", sparseInput);

    const auto weights = createWeightsConstant(builder, origWeights, storageElementTable);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto origPpeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    const auto adaptedOutElemType =
            VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    origPpeAttr, outputType.getElementType());

    auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
    auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
    const auto weightsTableVec =
            VPU::createWeightsTableData(sparseInput, adaptedOutElemType, weights, {}, OC, ppeConverter, biasConverter,
                                        nullptr, VPU::canAutopadOutput(origOp));
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC);
    const auto weightsTable = VPU::createWeightsTableTensor(builder, origOp->getLoc(), weightsTableVec, wtShape);

    const auto origWeightsShape = getShape(origWeights);
    const auto rawFilterShape = getIntArrayAttr(
            builder,
            SmallVector<int64_t>{OC, 1, origWeightsShape[Dims4D::Filter::KY], origWeightsShape[Dims4D::Filter::KX]});
    const auto padding = VPU::getPaddingAttr(origOp.getContext(), 0, 0, 0, 0);
    auto dwConv = builder.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), outputType, sparseInput, weights, weightsTable, /*dataPointerTensor=*/nullptr,
            /*sparsityPointerTensor=*/nullptr, /*scaleTensor=*/nullptr, /*biasTensor=*/nullptr,
            /*zeroPointTensor=*/nullptr, origOp.getStridesAttr(), padding, origOp.getPpeAttr(), rawFilterShape,
            origOp.getMultiClusterStrategyAttr(), nullptr, nullptr);

    nestedLog.trace("Created DWConv with SEP for Interpolate.");

    origOp.getOutput().replaceAllUsesWith(dwConv->getResult(0));
}

void ConvertNCEInterpolateToDWPass::safeRunOnFunc() {
    auto func = getOperation();
    const auto arch = config::getArch(func);
    SmallVector<VPU::NCEInterpolateOp> interpsToErase{};

    func.walk([&](VPU::NCEInterpolateOp interpOp) {
        _log.trace("Got '{0}' at '{1}'", interpOp->getName(), interpOp->getLoc());

        auto groupSparseOp = interpOp.getInput().getDefiningOp<VPU::GroupSparseTensorOp>();
        if (groupSparseOp == nullptr || !groupSparseOp->hasOneUse()) {
            _log.trace("No GroupSparseTensorOp as direct producer or it has multiple consumers.");
            return;
        }

        auto storageElementTable = groupSparseOp.getStorageElementTable().getDefiningOp<VPU::StorageElementTableOp>();
        if (storageElementTable == nullptr) {
            _log.trace("No StorageElementTableOp as direct producer of GroupSparseTensorOp.");
            return;
        }

        auto weights = interpOp.getWeights();
        auto weightsParentOp = weights.getDefiningOp();
        if (mlir::isa_and_nonnull<VPU::GroupSparseTensorOp>(weightsParentOp)) {
            weightsParentOp = weightsParentOp->getOperand(0).getDefiningOp();
        }

        auto weightsConst = mlir::dyn_cast_or_null<Const::DeclareOp>(weightsParentOp);
        if (weightsConst == nullptr) {
            _log.trace("Cannot find weights constant.");
            return;
        }

        if (!isDepthwiseConvMorePerformant(interpOp, arch, _log.nest())) {
            _log.trace("Interpolate is more performant as DPU Conv than DPU DWConv.");
            return;
        }

        convertToDWConv(interpOp, groupSparseOp, storageElementTable, weightsConst, arch);
        interpsToErase.push_back(interpOp);
    });

    for (auto interp : llvm::make_early_inc_range(interpsToErase)) {
        interp->erase();
    }
}

}  // namespace

//
// createConvertNCEInterpolateToDWPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertNCEInterpolateToDWPass(Logger log) {
    return std::make_unique<ConvertNCEInterpolateToDWPass>(log);
}
