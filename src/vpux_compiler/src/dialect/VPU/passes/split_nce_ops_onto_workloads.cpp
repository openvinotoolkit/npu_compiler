//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// Upper bound for workload numbers
//

constexpr int64_t MAX_SPLIT_NUMBER = 50;

//
// MPE mode utilities
//

bool isMixedPrecisionSupportedForVPUX30XX(mlir::Type inElemType, mlir::Type outElemType, mlir::Operation* operation) {
    return inElemType.isa<mlir::quant::QuantizedType>() && !outElemType.isa<mlir::quant::QuantizedType>() &&
           mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(
                   operation);
}

VPU::MPEMode getMpeModeForVPUX30XX(mlir::Type inElemType, mlir::Type outElemType, mlir::Operation* operation,
                                   ShapeRef shape) {
    if (isMixedPrecisionSupportedForVPUX30XX(inElemType, outElemType, operation)) {
        return VPU::MPEMode::VECTOR_FP16;
    }
    if (inElemType.isa<mlir::quant::QuantizedType>() || outElemType.isa<mlir::quant::QuantizedType>()) {
        const double W = static_cast<double>(shape[Dims4D::Act::W]);
        const double H = static_cast<double>(shape[Dims4D::Act::H]);
        // VPU::MPEMode::MATRIX process tensor using W=4 H=4 parts, calculate grid cells count for it
        const double matrixPartsCount = std::ceil(W / 4.0) * std::ceil(H / 4.0);
        // VPU::MPEMode::VECTOR process tensor using W=16 H=1 parts, calculate grid cells count for it
        const double vectorPartsCount = std::ceil(W / 16.0) * H;
        // Cells count is in direct ratio with work size, so choose smaller one
        return (vectorPartsCount <= matrixPartsCount) ? VPU::MPEMode::VECTOR : VPU::MPEMode::MATRIX;
    }

    if (inElemType.isF16() || inElemType.isBF16() || outElemType.isF16() || outElemType.isBF16()) {
        return VPU::MPEMode::VECTOR_FP16;
    }

    // Let's fall back to vector (might be a bad idea though).
    return VPU::MPEMode::VECTOR;
}

inline std::vector<unsigned int> getNTHWNTKGrid(VPU::MPEMode mode) {
    switch (mode) {
    case VPU::MPEMode::CUBOID_4x16:
        return {8, 8, 256};
    case VPU::MPEMode::CUBOID_8x16:
        return {16, 8, 128};
    default:
        return {16, 16, 64};
    }
}

VPU::MPEMode getMpeModeForVPUX37XXConv(ShapeRef shape) {
    std::vector<std::pair<double, VPU::MPEMode>> MPECost = {{0.0, VPU::MPEMode::CUBOID_4x16},
                                                            {0.0, VPU::MPEMode::CUBOID_8x16},
                                                            {0.0, VPU::MPEMode::CUBOID_16x16}};

    for (unsigned int idx = 0; idx < MPECost.size(); idx++) {
        auto grid = getNTHWNTKGrid(MPECost[idx].second);

        // Get the number of weights and activation grid reads
        double numWtGrids = std::ceil((double)shape[Dims4D::Act::C] / (double)grid[2]);
        double numActGrids = std::ceil((double)shape[Dims4D::Act::H] / (double)grid[1]) *
                             std::ceil((double)shape[Dims4D::Act::W] / (double)grid[0]);

        // Compute the simplfied number of reads
        double actReads = numWtGrids * shape[Dims4D::Act::H] * shape[Dims4D::Act::W];
        double wtReads = numActGrids * shape[Dims4D::Act::C];

        MPECost[idx].first = actReads + wtReads;
    }

    // Select the one that has min number of reads
    return (*std::min_element(MPECost.begin(), MPECost.end())).second;
}

VPU::MPEMode getMpeModeForVPUX37XX(mlir::Type, mlir::Type, mlir::Operation* operation, ShapeRef shape) {
    if (mlir::isa<VPU::NCEConvolutionOp>(operation)) {
        return getMpeModeForVPUX37XXConv(shape);
    } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(operation) || mlir::isa<VPU::NCEMaxPoolOp>(operation) ||
               mlir::isa<VPU::NCEAveragePoolOp>(operation)) {
        return VPU::MPEMode::CUBOID_16x16;
    } else if (mlir::isa<VPU::NCEEltwiseOp>(operation)) {
        return VPU::MPEMode::CUBOID_8x16;
    }

    return VPU::MPEMode::CUBOID_16x16;
}

using GetMpeModeCb = VPU::MPEMode (*)(mlir::Type, mlir::Type, mlir::Operation*, ShapeRef);

const EnumMap<VPU::ArchKind, GetMpeModeCb> mpeMap = {
        {VPU::ArchKind::VPUX30XX, getMpeModeForVPUX30XX},
        {VPU::ArchKind::VPUX311X, getMpeModeForVPUX30XX},
        {VPU::ArchKind::VPUX37XX, getMpeModeForVPUX37XX},
};

//
// generateWorkloads
//

// for workloads in sub tensors, offsets need to be from original full output tensor
void addSubTensorOffset(TileInfo& tileInfo, ShapeRef tensorOffset) {
    VPUX_THROW_WHEN(tileInfo.offsets.size() != tensorOffset.size(),
                    "Invalid size for TileInfo.offset {0} and sub tensor offset {1}", tileInfo.offsets.size(),
                    tensorOffset.size());

    for (auto d : irange(tileInfo.offsets.size())) {
        const auto dim = Dim(d);
        tileInfo.offsets[dim] += tensorOffset[dim];
    }
}

void generateWorkloads(mlir::OpBuilder& builder, VPU::NCEOpInterface origOp,
                       const VPUIP::WorkloadCostParams& costParams, VPU::MPEMode mpeMode, bool isTileOverZSupported,
                       const std::shared_ptr<VPUNN::VPUCostModel>& costModel, mlir::IntegerAttr clusterId = nullptr,
                       ShapeRef subTensorOffset = {}) {
    VPUIP::DpuTiler dpuTiler(costParams.outputShape, mpeMode);

    VPUIP::WorkloadSplitPool splitPoolSet;

    auto inElemType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (costParams.arch == VPU::ArchKind::VPUX30XX &&
        isMixedPrecisionSupportedForVPUX30XX(inElemType, outElemType, origOp.getOperation())) {
        dpuTiler.tileOverHWMixedPrecision(splitPoolSet);
    } else {
        dpuTiler.tileOverH(costParams.numDPU, splitPoolSet);

        // Invariants that produce sparse activations must have the same number of channels across the variants
        const auto requiresEqualZ = (origOp->getResult(0).getType().dyn_cast<VPU::SparseTensorType>() != nullptr);

        const auto splitNumPool = costParams.arch == VPU::ArchKind::VPUX37XX
                                          ? dpuTiler.generateSplitNumberPool(costParams.numDPU, 1)
                                          : dpuTiler.generateSplitNumberPool(costParams.numDPU, MAX_SPLIT_NUMBER);

        for (const auto& splitNum : splitNumPool) {
            dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_HW, splitPoolSet);
            if (isTileOverZSupported) {
                dpuTiler.tileOverZ(splitNum, splitPoolSet, requiresEqualZ);
            }
        }
    }

    // select workload with minimum cost
    auto splitPool = to_std_vector(splitPoolSet);
    VPUX_THROW_WHEN(splitPool.empty(), "Workload split pool is empty");

    std::vector<int64_t> splitPoolCosts(splitPool.size(), 0);
    for (const auto ind : irange(splitPool.size())) {
        auto& curSplit = splitPool[ind];

        if (clusterId != nullptr) {
            for (auto& wl : curSplit) {
                auto& outTile = std::get<0>(wl);
                addSubTensorOffset(outTile, subTensorOffset);
            }
        }

        splitPoolCosts[ind] = VPUIP::computeSplitCost(curSplit, costParams, costModel);
    }

    const auto bestSplitInd = std::min_element(splitPoolCosts.begin(), splitPoolCosts.end()) - splitPoolCosts.begin();
    const auto& bestSplit = splitPool[bestSplitInd];

    origOp->setAttr(DPUCost, getIntAttr(origOp->getContext(), splitPoolCosts[bestSplitInd]));

    const auto kernel = origOp.getKernelSize();
    const auto strides = origOp.getStrides();

    for (const auto& wl : bestSplit) {
        const auto& outTile = std::get<0>(wl);
        const auto mpeMode = std::get<1>(wl);

        const auto padsTileConf =
                backInferPadsTile(outTile, costParams.fullInputShape, costParams.padInfo, kernel, strides);
        auto tilePad = VPU::getPaddingAttr(builder.getContext(), padsTileConf);

        origOp.addWorkload(builder, origOp.getLoc(), outTile.offsets, outTile.shape, tilePad, mpeMode, clusterId);
    }
}

//
// splitOntoWorkloads
//

void splitOntoWorkloads(mlir::OpBuilder& builder, VPU::NCEOpInterface origOp, VPUIP::WorkloadCostParams& costParams,
                        VPU::MPEMode mpeMode, bool isTileOverZSupported,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    if (auto clusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(origOp->getParentOp())) {
        const auto outputs = clusterOp->getResults();
        VPUX_THROW_UNLESS(outputs.size() == 1, "Wrong outputs size: {0}", outputs.size());

        const auto output = *outputs.begin();

        auto getDistributedTensor = [](const mlir::Value value) -> VPU::DistributedTensorType {
            if (auto sparseTensor = value.getType().dyn_cast<VPU::SparseTensorType>()) {
                return sparseTensor.getData().dyn_cast<VPU::DistributedTensorType>();
            }
            return value.getType().dyn_cast<VPU::DistributedTensorType>();
        };

        auto distributedOutputType = getDistributedTensor(output);
        VPUX_THROW_WHEN(distributedOutputType == nullptr, "Wrong output type {0} for NCEClusterTilingOp",
                        output.getType());

        const auto outputSubTensorShapes = distributedOutputType.getPerClusterComputeShapes();
        auto outputSubTensorOffsets = distributedOutputType.getPerClusterComputeShapeOffsets();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != outputSubTensorOffsets.size(),
                        "sub tensor size:{0} not equal to offset size:{1}", outputSubTensorShapes.size(),
                        outputSubTensorOffsets.size());

        const auto inputs = clusterOp->getOperands();
        VPUX_THROW_UNLESS(inputs.size() >= 1, "Wrong inputs size: {0}", inputs.size());

        const auto input = *inputs.begin();
        auto distributedInputType = getDistributedTensor(input);
        VPUX_THROW_WHEN(distributedInputType == nullptr, "Wrong input type {0} for NCEClusterTilingOp",
                        input.getType());

        const auto inputSubTensorShapes = distributedInputType.getPerClusterComputeShapes();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != inputSubTensorShapes.size(),
                        "output tensor size:{0} not equal to input tensor size:{1}", outputSubTensorShapes.size(),
                        inputSubTensorShapes.size());

        // ----
        // In the case of an non broadcasted SOK, outputSubTensorOffsets don't need to be applied
        const auto distributionAttr = distributedOutputType.getDistribution();

        if (distributionAttr.mode().getValue() == VPU::DistributionMode::SEGMENTED) {
            const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
            const auto totalTiles = std::accumulate(numTiles.begin(), numTiles.end(), static_cast<int64_t>(1),
                                                    std::multiplies<int64_t>());

            if (numTiles[Dims4D::Act::C.ind()] > 1 && totalTiles == numTiles[Dims4D::Act::C.ind()]) {
                for (auto& shapeOffset : outputSubTensorOffsets) {
                    std::fill(shapeOffset.begin(), shapeOffset.end(), 0);
                }
            }
        }
        // ----

        for (size_t clusterId = 0; clusterId < outputSubTensorShapes.size(); clusterId++) {
            auto clusterIdAttr = getIntAttr(origOp->getContext(), clusterId);
            costParams.inputShape = inputSubTensorShapes[clusterId];
            costParams.outputShape = outputSubTensorShapes[clusterId];

            if (costParams.arch == VPU::ArchKind::VPUX37XX && mlir::isa<VPU::NCEConvolutionOp>(origOp)) {
                mpeMode = getMpeModeForVPUX37XXConv(outputSubTensorShapes[clusterId]);
            }
            generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverZSupported, costModel, clusterIdAttr,
                              outputSubTensorOffsets[clusterId]);
        }
    } else {
        generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverZSupported, costModel);
    }
}

//
// GenericNCERewrite
//

class GenericNCERewrite final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    GenericNCERewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch,
                      std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx),
              _numDPU(numDPU),
              _arch(arch),
              _costModel(std::move(costModel)),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numDPU;
    VPU::ArchKind _arch;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    Logger _log;
};

mlir::LogicalResult GenericNCERewrite::matchAndRewrite(VPU::NCEOpInterface nceOp,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto inputType = nceOp->getOperand(0).getType().cast<NDTypeInterface>();
    const auto outputType = nceOp->getResult(0).getType().cast<NDTypeInterface>();

    const auto inElemType = inputType.getElementType();
    const auto outElemType = outputType.getElementType();

    const auto inputShape = inputType.getShape();
    const auto outputShape = outputType.getShape();

    const auto pads = nceOp.getPad();

    const auto mpeByType = mpeMap.at(_arch);
    const auto mpeMode = mpeByType(inElemType, outElemType, nceOp, outputShape);

    VPUIP::WorkloadCostParams params;
    params.dataType = inElemType;
    params.numDPU = _numDPU;
    params.arch = _arch;
    params.fullInputShape = inputShape.raw();
    params.inputShape = inputShape.raw();
    params.outputShape = outputShape.raw();
    params.padInfo = VPU::toPadInfo(pads);
    params.kernelSize = nceOp.getKernelSize();
    params.kernelStride = nceOp.getStrides();

    bool isTileOverZSupported = mpeMode == VPU::MPEMode::VECTOR;

    llvm::TypeSwitch<mlir::Operation*, void>(nceOp.getOperation())
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp) {
                const auto inOrder = inputType.getDimsOrder();
                const auto isCMajor = inOrder == DimsOrder::NCHW;

                params.nceTaskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
                isTileOverZSupported |= !isCMajor;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::DWCONV;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::MAXPOOL;
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::AVEPOOL;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
                isTileOverZSupported = false;
            })
            .Case<VPU::NCEPermuteQuantizeOp>([&](VPU::NCEPermuteQuantizeOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
                isTileOverZSupported = false;
            })
            .Default([](mlir::Operation* op) {
                VPUX_THROW("Unsupported NCE operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });

    rewriter.updateRootInPlace(nceOp, [&]() {
        splitOntoWorkloads(rewriter, nceOp, params, mpeMode, isTileOverZSupported, _costModel);
    });

    return mlir::success();
}

//
// SplitNCEOpsOntoWorkloads
//

class SplitNCEOpsOntoWorkloadsPass final : public SplitNCEOpsOntoWorkloadsBase<SplitNCEOpsOntoWorkloadsPass> {
public:
    explicit SplitNCEOpsOntoWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SplitNCEOpsOntoWorkloadsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    VPUX_THROW_UNLESS(mpeMap.find(arch) != mpeMap.end(), "Failed to map MPE mode to target arch");

    auto nceCluster = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(VPU::ExecutorKindAttr::get(&ctx, VPU::ExecutorKind::DPU));
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    const auto numDPUs = dpuExec.count();

    const auto costModel = VPU::createCostModel(arch);

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op)) {
            return true;
        }
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            return !nceOp.workloads().empty();
        }
        return true;
    });
    target.addLegalOp<VPU::DPUWorkloadOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericNCERewrite>(&ctx, numDPUs, arch, costModel, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitNCEOpsOntoWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitNCEOpsOntoWorkloadsPass(Logger log) {
    return std::make_unique<SplitNCEOpsOntoWorkloadsPass>(log);
}
