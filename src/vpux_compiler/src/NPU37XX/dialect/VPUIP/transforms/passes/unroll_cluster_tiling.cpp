//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/profiling.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch37xx {
#define GEN_PASS_DECL_UNROLLCLUSTERTILING
#define GEN_PASS_DEF_UNROLLCLUSTERTILING
#include "vpux/compiler/NPU37XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch37xx

using namespace vpux;

namespace {
void updateSwProfilingMetadata(VPUIP::SwKernelOp newTask, VPUIP::SwProfilingMetadataAttr attr, size_t clusterId) {
    if (attr == nullptr) {
        return;
    }
    const size_t bufferId = attr.getBufferId().getInt();
    const size_t bufferOffset = attr.getBufferOffset().getInt();
    const size_t clusterSize = attr.getClusterSize().getInt();
    const size_t dataIndex = attr.getDataIndex().getInt();
    const size_t tileId = attr.getTileId().getInt();
    auto profMeta = vpux::getSwProfilingMetaAttr(attr.getContext(), bufferId, bufferOffset, clusterSize, dataIndex,
                                                 tileId, clusterId);
    newTask.setProfilingMetadataAttr(profMeta);
}
};  // namespace

//
// ClusterSWRewriter
//

void VPUIP::arch37xx::ClusterSWRewriter::matchAndRewrite(VPUIP::SwKernelOp swTask, mlir::OpBuilder& builder) const {
    _log.trace("Process SW op: '{0}'", swTask);

    auto vpurtTask = swTask->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    builder.setInsertionPointAfter(vpurtTask);

    if (swTask.getInputs().empty() || swTask.getOutputs().empty()) {
        // append "cluster_0" suffix to cache handling operation's location
        auto oldLoc = swTask->getLoc();
        if (stringifyPrimaryLocation(oldLoc).find("/cluster_") == std::string::npos) {
            swTask->setLoc(appendLoc(oldLoc, "cluster_0"));
        }
        return;
    }

    auto input = *swTask.getInputs().begin();
    auto output = *swTask.getOutputs().begin();

    auto inputType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto outputType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();

    if (inputType == nullptr && outputType == nullptr) {
        _log.trace("Input and output types are not distributed, nothing to unroll");
        auto oldLoc = swTask->getLoc();
        VPUX_THROW_WHEN(stringifyPrimaryLocation(oldLoc).find("/cluster_") != std::string::npos,
                        "/cluster_ suffix should not be present yet but was found in {0}", oldLoc);
        swTask->setLoc(appendLoc(oldLoc, "cluster_0"));
        return;
    }

    auto inDistributionMode =
            inputType != nullptr ? inputType.getDistribution().getMode().getValue() : VPU::DistributionMode::NONE;
    auto outDistributionMode =
            outputType != nullptr ? outputType.getDistribution().getMode().getValue() : VPU::DistributionMode::NONE;
    VPUX_THROW_WHEN(outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for SW op {0}; output in OVERLAPPED mode.", swTask->getLoc());
    VPUX_THROW_WHEN(inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When SW op has input in OVERLAPPED mode then output must be segmented. op = {0}, out mode = '{1}'",
                    swTask->getLoc(), VPU::stringifyDistributionMode(outDistributionMode));

    const auto distributionAttr = inputType != nullptr ? inputType.getDistribution() : outputType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();

    auto loc = swTask->getLoc();

    auto parentInputBuffs = swTask.getInputs();
    auto parentOutputBuffs = swTask.getOutputBuffs();

    // store inputs/outputs per cluster
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> inputBuffs;
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> outputBuffs;
    SmallVector<TileInfo> outputTiles;
    SmallVector<TilingInfo> inputTiles;

    auto allowDiscontinuousBuffers = VPUIP::isStridedDataAccessSupported(swTask);
    for (const auto& input : parentInputBuffs) {
        auto currBuffs = VPUIP::getPerClusterSWMemoryBuffers(_ctx, loc, "input", swTask, input, numClusters, builder,
                                                             _log, allowDiscontinuousBuffers);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            inputBuffs[clusterId].push_back(currBuffs[clusterId]);
        }
    }

    for (const auto& output : parentOutputBuffs) {
        auto currBuffs = VPUIP::getPerClusterSWComputeBuffers(_ctx, loc, "outputBuff", swTask, output, numClusters,
                                                              builder, _log, true);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            outputBuffs[clusterId].push_back(currBuffs[clusterId]);
        }
    }

    auto getPerClusterTileInfo = [&numClusters](ShapeRef shape, ShapeRef offset, std::optional<int64_t> tileDim) {
        Shape axis(shape.size(), 1);
        if (tileDim.has_value()) {
            axis[Dim(tileDim.value())] = numClusters;
        }
        return TileInfo(shape, offset, axis);
    };

    // For overlapped input, the Swkernel's attr need to be updated according to its input/output tiles
    const auto kernelEntryName = getSwKernelEntryName(swTask);
    auto needUpdateAttrs =
            inDistributionMode == VPU::DistributionMode::OVERLAPPED || kernelEntryName == "lstm_sequence";

    if (needUpdateAttrs) {
        auto outTileIndex = VPUIP::getTilingDimIndex(outputType);
        VPUX_THROW_UNLESS(outTileIndex.has_value(), "Can not get tiling dim for {0}", outputType);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            SmallVector<TileInfo> tiles;
            for (const auto& operand : parentInputBuffs) {
                auto distributedType = operand.getType().dyn_cast<VPUIP::DistributedBufferType>();
                auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
                auto tileInfo =
                        getPerClusterTileInfo(distributedType.getPerClusterMemoryShapes()[clusterId],
                                              distributedType.getPerClusterMemoryShapeOffsets()[clusterId], tileIndex);
                tiles.push_back(tileInfo);
            }
            auto inTiles = TilingInfo(tiles);
            auto outTile = getPerClusterTileInfo(outputType.getPerClusterComputeShapes()[clusterId],
                                                 outputType.getPerClusterComputeShapeOffsets()[clusterId],
                                                 outTileIndex.value());
            inputTiles.push_back(inTiles);
            outputTiles.push_back(outTile);
        }
    }

    auto numClustersOfProfilingData = numClusters;
    if (swTask.getProfilingData()) {
        // Get numClusters of profiling data from its own distributed type.
        // This is to prevent incompatibility between the distributed types of profiling data and input.
        // For example: for the MVN layer with below configuration on NPU4:
        //  - input Shape [1, 32, 262144, 1]
        //  - acrossChannel is true
        // MC strategy is SOK and tiling dimension is on channel.
        // The 32 channels are split into [6, 6, 5, 5, 5, 5].
        // For the sub-tile with 5 channels, num_clusters is 5 in input distributed type, while profiling data's
        // distributed type is created with num_clusters = 6.
        // Unrolling profiling data to 5 clusters would cause error with getPerClusterMemoryShapes.
        if (auto profilingDataType = swTask.getProfilingData().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            numClustersOfProfilingData = profilingDataType.getDistribution().getNumClusters().getInt();
        }
    }
    auto profilingBuffs = VPUIP::getPerClusterSWMemoryBuffers(
            _ctx, loc, "profilingBuff", swTask, swTask.getProfilingData(), numClustersOfProfilingData, builder, _log);

    auto taskArgs = kernelArgsRange(swTask);

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, "cluster_{0}", clusterId);
        mlir::Value profilingData = nullptr;
        mlir::Type profilingOutputType = nullptr;

        if (swTask.getProfilingData()) {
            profilingOutputType = profilingBuffs[clusterId].getType();
            profilingData = profilingBuffs[clusterId];
            VPUX_THROW_WHEN(swTask.getProfilingMetadataAttr() == nullptr, "Missing profiling metadata for '{0}'",
                            swTask);
        }

        SmallVector<mlir::Type> inputTypes;
        for (auto& temp : inputBuffs[clusterId]) {
            inputTypes.push_back(temp.getType());
        }
        for (auto& temp : outputBuffs[clusterId]) {
            inputTypes.push_back(temp.getType());
        }

        auto newArgs = needUpdateAttrs ? VPUIP::getSwkernelNewAttrsAfterTiling(swTask, taskArgs, inputTiles[clusterId],
                                                                               outputTiles[clusterId], _log.nest())
                                       : taskArgs;
        for (auto& arg : newArgs) {
            const auto typedAttr = arg.dyn_cast_or_null<mlir::TypedAttr>();
            const auto type = typedAttr != nullptr ? typedAttr.getType() : mlir::NoneType::get(_ctx);
            inputTypes.push_back(type);
        }

        VPUIP::createRuntimeKernelDefinition(_module, _log.nest(), VPU::getArch(swTask.getOperation()));

        auto module = swTask->getParentOfType<mlir::ModuleOp>();
        auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swTask.getKernelFunctionAttr());
        VPUX_THROW_UNLESS(kernelFunc, "Invalid function call : '{0}', undefined kernel name",
                          swTask.getKernelFunctionAttr());

        const auto kernelCode = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
        const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
        auto kernelName = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_name");
        if (kernelName == nullptr) {
            kernelName = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
        }

        auto newOperands = kernelFunc.getName();

        auto builtInFunction = VPUIP::createBuiltInFunction(_module, newOperands, inputTypes, kernelEntryPoint,
                                                            kernelCode, kernelName, _log);

        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                builder, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffs[clusterId],
                outputBuffs[clusterId], profilingData, builtInFunction, getIntAttr(builder, clusterId));
        updateSwProfilingMetadata(newTask, swTask.getProfilingMetadataAttr(), clusterId);

        initSwKernel(newTask, inputBuffs[clusterId], outputBuffs[clusterId], newArgs, _log.nest());

        _log.trace("Task created: {0}", newTask);
    }

    vpurtTask->erase();
}

//
// ClusterNCERewriter
//

void VPUIP::arch37xx::ClusterNCERewriter::getInputBuffers(
        SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
        SmallVector<mlir::Value>& parentInputSparsityMap, SmallVector<mlir::Value>& inputSparsityMapBuffs,
        SmallVector<mlir::Value>& parentInputSETable, SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
        VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters, mlir::OpBuilder& builder) const {
    inputBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "input", nceTask.getInput(), numClusters, builder);
    auto parentInput = *nceTask.getInputs().begin();
    auto parentInputType = parentInput.getType().cast<VPUIP::DistributedBufferType>();

    mlir::UnitAttr isSegmented = isSegmentedNCETask(parentInputType);

    parentInputBuffs = VPU::isSegmentedOverC(parentInputType.getDistribution())
                               ? inputBuffs
                               : SmallVector<mlir::Value>(numClusters, parentInput);

    inputSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSparsityMap",
                                                              nceTask.getInputSparsityMap(), numClusters, builder);
    inputSETableBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSETable",
                                                          nceTask.getInputStorageElementTable(), numClusters, builder);

    auto arch = VPU::getArch(nceTask);
    bool isDWOpAndNeedsAlign = VPU::isDWOpAndNeedsAlign(arch, nceTask.getTaskType());
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        // For 37XX arch, ensure we have H_per_cluster x W as a multiple of 4 (or 8 for sparse inputs).
        // If the storage element table is present, its segment size has to fit this restriction
        if (isSegmented && clusterId != (numClusters - 1) &&
            (nceTask.getTaskType() == VPUIP::NCETaskType::CONV || isDWOpAndNeedsAlign)) {
            auto inShape = inputBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            if (nceTask.getInputStorageElementTable() != nullptr) {
                inShape = inputSETableBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            }
            const auto isInputSparse =
                    nceTask.getInputSparsityMap() != nullptr || nceTask.getInputStorageElementTable() != nullptr;
            const auto hAlignment = VPU::getSOHPerClusterHeightAlignment(inShape[Dims4D::Act::W], isInputSparse);
            VPUX_THROW_UNLESS((inShape[Dims4D::Act::H] % hAlignment) == 0,
                              "For segmented cluster we must have alignment to {0}, type: {1}", hAlignment,
                              inputBuffs[clusterId].getType());
        }
    }

    parentInputSparsityMap = SmallVector<mlir::Value>(numClusters, nceTask.getInputSparsityMap());
    parentInputSETable = SmallVector<mlir::Value>(numClusters, nceTask.getInputStorageElementTable());
}

void VPUIP::arch37xx::ClusterNCERewriter::getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs,
                                                           SmallVector<mlir::Value>& outputBuffs,
                                                           SmallVector<mlir::Value>& parentOutputSparsityMap,
                                                           SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                                           SmallVector<SmallVector<mlir::Value>>& /*outputItiBuffs*/,
                                                           mlir::Location loc, VPUIP::NCEClusterTaskOp nceTask,
                                                           const int64_t numClusters, mlir::OpBuilder& builder) const {
    auto parentInputType = (*nceTask.getInputs().begin()).getType().cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = (*nceTask.getOutputs().begin()).getType().cast<VPUIP::DistributedBufferType>();

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();
    // Elementwise operations may support overlapping for trailing convolution.
    // In that case both input and output modes are OVERLAPPED.
    const auto isEltwise = (nceTask.getTaskType() == VPUIP::NCETaskType::ELTWISE);
    VPUX_THROW_WHEN(!isEltwise && outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for NCE output in OVERLAPPED mode.");
    VPUX_THROW_WHEN(!isEltwise && inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When NCE has input in OVERLAPPED mode then output must be segmented. out mode = '{0}'",
                    VPU::stringifyDistributionMode(outDistributionMode));

    parentOutputSparsityMap = SmallVector<mlir::Value>(numClusters, nceTask.getOutputSparsityMapBuff());

    outputBuffs = VPUIP::getPerClusterComputeBuffers(_ctx, loc, "outputBuff", nceTask.getOutputBuff(), parentOutputType,
                                                     numClusters, builder, true);
    outputSparsityMapBuffs = VPUIP::getPerClusterComputeBuffers(
            _ctx, loc, "outputSparsityMapBuff", nceTask.getOutputSparsityMapBuff(), numClusters, builder, true);

    parentOutputBuffs = SmallVector<mlir::Value>(numClusters, *nceTask.getOutputs().begin());
    if (VPU::isSegmentedOverC(outDistribution)) {
        // for SEG SOK parent output buffers = output buffers
        parentOutputBuffs = outputBuffs;
    }
}

mlir::UnitAttr VPUIP::arch37xx::ClusterNCERewriter::isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const {
    // Only for explicit SEGMENTED mode, not in combination with
    // DUPLICATED or MULTICASTED
    if (inputType.getDistribution().getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return nullptr;
    }

    // Segmentation not present on H axis
    const auto numTiles = parseIntArrayAttr<int64_t>(inputType.getDistribution().getNumTiles());
    if (numTiles[Dims4D::Act::H.ind()] <= 1) {
        return nullptr;
    }

    // Segmentation not supported with non NHWC input such as CM Conv
    if (inputType.getDimsOrder() != DimsOrder::NHWC) {
        return nullptr;
    }

    return mlir::UnitAttr::get(_ctx);
}

namespace {

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::arch37xx::impl::UnrollClusterTilingBase<UnrollClusterTilingPass> {
public:
    explicit UnrollClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollClusterTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    const VPUIP::ClusterDMARewriter dmaRewriter(&ctx, dmaPortCount, /*dmaFusionHandler=*/{}, _log);
    const VPUIP::arch37xx::ClusterSWRewriter swRewriter(&ctx, module, _log);
    const VPUIP::arch37xx::ClusterNCERewriter nceRewriter(&ctx, _log);

    mlir::SmallVector<mlir::Operation*> toRemove;

    func.walk<mlir::WalkOrder::PostOrder>([&](VPURT::TaskOp vpurtTask) {
        auto op = vpurtTask.getInnerTaskOp();
        if (op == nullptr) {
            return;
        }

        mlir::OpBuilder builder(op);
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            dmaRewriter.matchAndRewrite(nndmaOp, builder);
        } else if (auto taskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            nceRewriter.matchAndRewrite(taskOp, builder);
        } else if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            swRewriter.matchAndRewrite(swOp, builder);
        }
    });
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch37xx::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
