//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/IRMapping.h>
#include <vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp>

namespace vpux {
namespace VPU {

TilingMode getTilingSupportedMode(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling, Logger log) {
    if (!enablePrefetchTiling) {
        return TilingMode::ISOLATED;
    }

    auto op = origOp.getOperation();

    // Pipelining for MVN1Normalize
    if (mlir::isa<VPU::MVN1NormalizeOp>(op)) {
        return TilingMode::PIPELINING;
    }

    // Prefetching for HW layers
    if (mlir::isa<VPU::NCEOpInterface>(op)) {
        auto tilingInfo = mlir::cast<VPU::TilingInfoOpInterface>(op);
        const auto resShape = getShape(op->getResult(0));
        const Shape neutralTile(resShape.size(), 1);
        auto fillTiles = fillDividedTiles(op, neutralTile, resShape);
        const auto isSupportIsolated =
                tilingInfo.isSupportedTiling(fillTiles.value(), TilingMode::ISOLATED, log.nest());
        const auto isPrefetchable = VPU::prefetchTilingConditionSatisfied(op, log.nest());
        auto tilingMode = isSupportIsolated && isPrefetchable ? TilingMode::PREFETCHING : TilingMode::PIPELINING;
        return tilingMode;
    }

    return TilingMode::ISOLATED;
}

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     Logger log) {
    auto tilingMode = TilingMode::ISOLATED;
    return getLayerTilingStrategy(origOp, enablePrefetchTiling, tilingMode, log);
}

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     TilingMode& mode, Logger log) {
    log.trace("getLayerTilingStrategy for op '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    log.nest().trace("Enable Prefetch Tiling: {0}", enablePrefetchTiling);

    mode = getTilingSupportedMode(origOp, enablePrefetchTiling, log);

    log.nest().trace("Assigning {0} tiling strategy", getTilingModeStr(mode));
    return origOp.getTilingStrategy(mode, log.nest());
}

mlir::LogicalResult checkAndAlignActInputTiling(vpux::VPU::NCEOpInterface nceOp, InputTiling& inputTiling,
                                                vpux::Logger log) {
    auto origInputType = nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    // use effective sparse output type to reduce the validation for the input sparse type.
    const auto inType = mlir::isa<VPU::SparseTensorType>(origInputType)
                                ? mlir::cast<NDTypeInterface>(VPU::getEffectiveSparseOutputType(origInputType))
                                : origInputType;

    auto tiledInputType = inType.extractDenseTile(inputTiling.tiles[0].offsets, inputTiling.tiles[0].shape);
    if (mlir::succeeded(nceOp.verifyInputType(tiledInputType))) {
        return mlir::success();
    }
    log.trace("Inferred activation input tiling {0} is invalid for {1}", inputTiling.tiles[0], nceOp->getName());
    auto stride = nceOp.getStridesVal()[Dims4D::Strides::X.ind()];  // get W side strides
    int64_t bias = 0;
    auto newInputActTiling = inputTiling.tiles[0];
    while (++bias < stride) {
        auto alignedShape =
                Shape({inputTiling.tiles[0].shape[Dims4D::Act::N], inputTiling.tiles[0].shape[Dims4D::Act::C],
                       inputTiling.tiles[0].shape[Dims4D::Act::H], inputTiling.tiles[0].shape[Dims4D::Act::W] + bias});
        newInputActTiling = TileInfo(alignedShape, inputTiling.tiles[0].offsets, inputTiling.tiles[0].axis);
        auto newInputActType = inType.extractDenseTile(newInputActTiling.offsets, newInputActTiling.shape);
        if (mlir::succeeded(nceOp.verifyInputType(newInputActType))) {
            inputTiling.tiles[0] = std::move(newInputActTiling);
            log.trace("Input tiling is corrected to {0}", inputTiling.tiles[0]);
            return mlir::success();
        }
    }
    VPUX_THROW("Cannot find aligned act input tiling for op {0} at {1}", nceOp->getName(), nceOp->getLoc());
}

SmallVector<mlir::Value> reifyTiles(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile,
                                    mlir::OpBuilder& builder, Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);

    auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    auto& inTiles = inputTiling.tiles;

    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::IRMapping mapper;
    for (auto p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();

        const auto valName = printToString("input {0}", inputIdx);
        const auto tiledInput = vpux::VPU::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);

        mapper.map(origInput, tiledInput);
    }

    const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);

    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());

    tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

    vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::ALL);

    return tiledOp->getResults();
}

mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                      mlir::PatternRewriter& rewriter, Logger log) {
    const auto results = origOp->getResults();

    auto resultTileValues = SmallVector<SmallVector<mlir::Value>>(results.size());
    auto resultTileOffsets = SmallVector<SmallVector<Shape>>(results.size());

    for (const auto& outputTile : tiles) {
        auto tiledResults = reifyTiles(origOp, outputTile, rewriter, log);
        const auto outputTiling = origOp.getOutputTiling(outputTile, log);
        VPUX_THROW_UNLESS(results.size() == outputTiling.size(),
                          "Number of results '{0}' doesn't match with number of output tiles '{1}' at '{2}'",
                          results.size(), outputTiling.size(), origOp->getLoc());

        for (const auto i : irange(results.size())) {
            const auto& outputTile = outputTiling[i];
            auto tiledResult = tiledResults[i];

            const auto tiledShape = getShape(tiledResult);
            VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                              "Inferred output shape '{0}' doesn't match tiled shape '{1}' at '{2}'", tiledShape,
                              outputTile.shape, tiledResult.getDefiningOp()->getLoc());

            const auto resultType = mlir::cast<vpux::NDTypeInterface>(results[i].getType());
            const auto resultDenseTile = resultType.extractDenseTile(outputTile.offsets, outputTile.shape);

            tiledResult.setType(resultDenseTile);

            resultTileValues[i].push_back(tiledResult);
            resultTileOffsets[i].push_back(outputTiling[i].offsets);
        }
    }

    SmallVector<mlir::Value> concatOps;
    for (const auto i : irange(results.size())) {
        auto resultType = origOp->getResult(i).getType();
        auto tileValues = mlir::ValueRange(resultTileValues[i]);
        auto tileOffsets = ArrayRef(resultTileOffsets[i]);

        auto concatOp = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), resultType, tileValues, tileOffsets);

        concatOps.push_back(concatOp.getOutput());
    }

    rewriter.replaceOp(origOp, concatOps);

    return mlir::success();
}

bool hasMultiBranches(mlir::Operation* op) {
    // not the only result
    if (op->getResults().size() != 1) {
        return true;
    }
    // only one user
    if (op->getResult(0).hasOneUse()) {
        return false;
    }
    // only one result but multiple users
    auto user1 = op->getResult(0).user_begin();
    for (auto remainUser : llvm::drop_begin(op->getResult(0).getUsers())) {
        if (remainUser != *user1) {
            return true;
        }
    }
    return false;
}

mlir::Operation* getParentComputeOp(mlir::Operation* op) {
    // for const prefetch ignore cases where activation is handled by
    // intermediate operations and causes a stall
    // prefetch is wanted from current op to parent op
    const std::function<bool(mlir::Operation*)> isOpIgnorable = [&](mlir::Operation* op) -> bool {
        if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(op)) {
            return isOpIgnorable(vfOp.getBody()->getTerminator()->getOperands().back().getDefiningOp());
        }
        if (auto nceEltwiseAnd = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
            return nceEltwiseAnd.getOpType() == VPU::EltwiseType::AND;
        }
        if (mlir::isa<VPU::MemPermuteOp, VPU::DepthToSpaceOp, VPU::SpaceToDepthOp>(op) &&
            !VPUIP::isLegalAndBeneficialConvertToDMA(op)) {
            // don't ignore layers that will be converted to SW but not SWOpInterface now
            return true;
        }

        return !mlir::isa<VPU::NCEOpInterface>(op) && !mlir::isa<VPU::SWOpInterface>(op);
    };

    mlir::Operation* parentOp = op->getOperand(0).getDefiningOp();
    if (parentOp == nullptr) {
        if (auto vfOp = op->getParentOfType<VPU::VerticalFusionOp>()) {
            if (auto vfArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(op->getOperand(0))) {
                parentOp = vfOp.getOperand(vfArg.getArgNumber()).getDefiningOp();
            }
        }
    }
    while (parentOp && isOpIgnorable(parentOp)) {
        // skip the Permute, Reshape and And
        if (parentOp->getOperands().size() < 1) {
            break;
        }
        if (hasMultiBranches(parentOp)) {
            // for parallel sub-graphs, the order is undecided yet
            // abandon prefetching these cases
            return nullptr;
        }
        parentOp = parentOp->getOperand(0).getDefiningOp();
    }
    // check the last op
    return (parentOp == nullptr || hasMultiBranches(parentOp)) ? nullptr : parentOp;
}

bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log) {
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        return false;
    }
    auto parentOp = getParentComputeOp(op);
    if (parentOp == nullptr) {
        return false;
    }
    auto opTilingInter = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    auto parentTilingInter = mlir::isa<VPU::TilingInfoOpInterface, VPU::VerticalFusionOp>(parentOp);
    if (!opTilingInter || !parentTilingInter) {
        return false;
    }
    auto opTilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!opTilingBuilder) {
        return false;
    }

    // For parallel sub-graphs, the order is undecided yet
    // Abandon prefetching these cases
    if (!parentOp->getResult(0).hasOneUse()) {
        auto user1 = *parentOp->getResult(0).getUsers().begin();
        for (auto remainUser : parentOp->getResult(0).getUsers()) {
            if (remainUser != user1) {
                return false;
            }
        }
    }

    // Check if tile pattern is supported
    const auto resShape = getShape(op->getResult(0));
    const Shape neutralTile(resShape.size(), 1);
    auto fillTiles = fillDividedTiles(op, neutralTile, resShape);
    if (mlir::failed(fillTiles)) {
        return false;
    }
    if (opTilingInter.isSupportedTiling(fillTiles.value(), TilingMode::PREFETCHING, log)) {
        return false;
    }
    log.nest(1).trace("Attempting to satisfy PREFETCHING tiling.");
    auto tiles = opTilingBuilder.getTilingStrategy(TilingMode::PREFETCHING, log.nest());
    if (mlir::failed(tiles)) {
        return false;
    }

    return tiles.value().begin()->axis != neutralTile;
}

bool isLargeConstOp(mlir::Operation* op, Logger log) {
    // The operation should have constant filter
    if (!mlir::isa<VPU::NCEConvolutionOp>(op) && !mlir::isa<VPU::NCEDepthConvolutionOp>(op) &&
        !mlir::isa<VPU::NCECompressConvolutionOp>(op)) {
        return false;
    }
    auto filter = op->getOperand(1).getDefiningOp<Const::DeclareOp>();
    if (filter == nullptr) {
        return false;
    }

    Byte filterSize(0);
    auto filterType = filter.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (op->hasAttr(multiClusterStrategy)) {
        auto nceOp = mlir::cast<VPU::NCEOpInterface>(op);
        auto clusterOp = mlir::cast<VPU::ClusteredOpInterface>(op);
        auto outputType = clusterOp->getResult(0).getType().cast<NDTypeInterface>();
        auto numClusters = VPU::getOptimalNumClusters(
                clusterOp, outputType.getShape(),
                clusterOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        auto filterDistributedType = VPU::getDistributedFilterTypeFromOp(nceOp, filterType, numClusters);
        for (auto filterType : filterDistributedType.getDistributedTypes()) {
            filterSize += filterType.cast<VPU::DistributedTensorType>().getTotalAllocSize();
        }
    } else {
        filterSize = filterType.getTotalAllocSize();
    }

    auto cmxThreshold = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(VPU::getTotalCMXSize(op).count()) * LARGE_CONST_THRESHOLD_RATIO)));
    if (filterSize > cmxThreshold) {
        log.nest(1).trace("filter size {0} is larger than cmxThreshold {1}", filterSize, cmxThreshold);
        return true;
    }
    return false;
}

bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log) {
    // Check if the operation has large constant filter
    if (!isLargeConstOp(op, log)) {
        return false;
    }
    auto opTilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!opTilingBuilder) {
        return false;
    }

    // Find the available tiling size over C
    // The pipelining should be doable with this tiling size
    log.nest(1).trace("Checking large const pipeline tiling.");
    auto tiles = opTilingBuilder.getTilingStrategy(TilingMode::PIPELINING, log.nest());
    if (mlir::failed(tiles)) {
        return false;
    }

    if (tiles.value().begin()->axis != Shape(getShape(op->getResult(0)).size(), 1)) {
        log.nest(1).trace("Found pipelining tiling strategy {0}", tiles.value().begin()->axis);
        return true;
    }

    return false;
}

bool archSupportsSwLayerTiling(VPU::ArchKind arch) {
    VPUX_UNUSED(arch);
    // Support all NPU37XX+ platforms.
    return true;
}

bool opNeedsTiling(mlir::Operation* op, bool enablePrefetchTiling, Logger log) {
    if (mlir::isa<VPU::SliceOp, VPU::ConcatOp, VPU::NCEClusterTilingOp>(op) ||
        op->getParentOfType<VPU::NCEClusterTilingOp>()) {
        return false;
    }
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (func == nullptr) {
        return false;
    }
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    if (!mlir::isa<VPU::NCEOpInterface>(op) && !VPU::archSupportsSwLayerTiling(arch)) {
        return false;
    }
    if (IE::hasDynamicTensors(op)) {
        log.warning("Tiling is not applied to the operation '{0}' at '{1}' because it has at least one operand or "
                    "result with a dynamic shape",
                    op->getName(), op->getLoc());
        // Track number: E-147023
        return false;
    }

    if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
        log.trace("Check: '{0}' at '{1}'", op->getName(), op->getLoc());
        const auto resType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        Shape resShape = resType.getShape().toValues();
        // TODO(E#113258): getShape needs to return shape based on upper bounds to avoid parsing bounds
        if (auto boundedTensor = resType.dyn_cast_or_null<vpux::BoundedTypeInterface>();
            boundedTensor != nullptr && boundedTensor.getBounds() != nullptr) {
            const auto bounds = parseIntArrayAttr<int64_t>(boundedTensor.getBounds());
            resShape = Shape(bounds.begin(), bounds.end());
        }
        TileInfo outputTile(resShape);
        // Mark the output tile as completed so that the inferred input shape contains the whole input
        outputTile.isCompletedTile = true;
        if (!iface.isSupportedTiling({std::move(outputTile)}, TilingMode::ISOLATED, log.nest())) {
            log.nest().trace("ISOLATED tiling or PIPELINING tiling required");
            return true;
        }
        if (enablePrefetchTiling && mlir::isa<VPU::NCEOpInterface>(op)) {
            if (VPU::prefetchTilingConditionSatisfied(op, log.nest())) {
                log.nest().trace("PREFETCHING tiling required");
                return true;
            }
            if (VPU::largeConstPipelineConditionSatisfied(op, log.nest())) {
                log.nest().trace("PIPELINING tiling for large constant weights required");
                return true;
            }
        }
    }
    return false;
}

std::optional<std::pair<size_t, size_t>> getWorkLoadInformationForNCEWithSparseOutput(
        VPU::ArchKind arch, ArrayRef<Shape> perClusterShapes, ArrayRef<int64_t> supportedChannels) {
    auto getWorkloadNum = [&](int64_t channelSupported) {
        size_t wlMaxNumPerCluster = 0;
        size_t wlNumInTotal = 0;
        for (const auto& perClusterShape : perClusterShapes) {
            size_t wlNum;
            const auto perClusterChannel = perClusterShape[vpux::Dims4D::Act::C];
            if (perClusterChannel % channelSupported == 0) {
                wlNum = perClusterChannel / channelSupported;
            } else {
                wlNum = divUp(perClusterChannel, channelSupported);
            }

            if (wlMaxNumPerCluster < wlNum) {
                wlMaxNumPerCluster = wlNum;
            }
            wlNumInTotal += wlNum;
        }
        return std::make_pair(wlMaxNumPerCluster, wlNumInTotal);
    };

    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    for (const auto channelSupported : supportedChannels) {
        if (!sparsityConstraint.areChannelsFitForSESize(channelSupported)) {
            continue;
        }

        // Only the last cluster can have the not-even channels for workloads
        // For exapmle, we need to split OC = 736 on 6 clusters, the tiled size will be
        // { {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64 ,32} }.
        //
        size_t numOfClusterNotEven = 0;
        size_t indexOfClusterNotEven = 0;
        for (const auto index : irange(perClusterShapes.size())) {
            if (perClusterShapes[index][vpux::Dims4D::Act::C] % channelSupported != 0) {
                numOfClusterNotEven++;
                indexOfClusterNotEven = index;
            }
        }

        if (numOfClusterNotEven == 0) {
            return getWorkloadNum(channelSupported);
        } else if (numOfClusterNotEven == 1) {
            if (indexOfClusterNotEven != perClusterShapes.size() - 1) {
                continue;
            }

            const auto clusterChannel = perClusterShapes[indexOfClusterNotEven][vpux::Dims4D::Act::C];
            const auto lastChannelNum = clusterChannel % channelSupported;
            if (llvm::count(supportedChannels, lastChannelNum)) {
                return getWorkloadNum(channelSupported);
            }
        }
    }
    return std::nullopt;
}

// All variants of a invariant update a single barrier, therefore the barrier count would be the number of variants.
// And the available slots of a barrier is limited to a architecture specific count. So the variants count must be
// less than a specific number.
bool doesNCEOpChannelSatisfyWorkload(mlir::Operation* nceOp, const TileInfo& outputTile) {
    auto channelAlignedIface = mlir::dyn_cast<VPU::AlignedWorkloadChannelsOpInterface>(nceOp);
    if (channelAlignedIface == nullptr) {
        return true;
    }
    const auto supportedChannels = channelAlignedIface.getSupportedWorkLoadChannels();
    auto log = Logger::global().nest();
    log.trace("supportedChannels - {0}", supportedChannels);
    const auto minSupportedChannel = supportedChannels.back();
    const auto tileChannel = outputTile.shape[Dims4D::Act::C];
    if (tileChannel % minSupportedChannel != 0) {
        log.trace("tileChannel {0} can not be divisible by minSupportedChannel {1}", tileChannel, minSupportedChannel);
        return false;
    }

    auto getDataType = [](mlir::Type type) {
        if (auto sparseTensor = type.dyn_cast<VPU::SparseTensorType>()) {
            return sparseTensor.getData();
        }
        return type;
    };

    const auto getPerClusterShapes = [&]() {
        auto outputType = getDataType(nceOp->getResult(0).getType()).cast<NDTypeInterface>();
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp);
        if (clusterOp == nullptr || !clusterOp.getMultiClusterStrategy().has_value()) {
            return SmallVector<Shape>{outputTile.shape};
        }
        // multi cluster case
        auto strategy = clusterOp.getMultiClusterStrategy().value();
        auto numClusters = VPU::getOptimalNumClusters(clusterOp, outputTile.shape, strategy);
        auto distributedType = getDistributedOutputTypeFromOp(clusterOp, outputTileType, numClusters, strategy);
        return getDataType(distributedType).cast<VPU::DistributedTensorType>().getPerClusterComputeShapes();
    };

    const auto perClusterShapes = getPerClusterShapes();

    // for some patterns, e.g. NCE(SOK and sparse output)->Concat->NCE, the sparse output would be removed in the
    // following pass
    auto nceOpIf = mlir::dyn_cast<VPU::NCEOpInterface>(nceOp);
    const auto isSparseRemoved =
            nceOpIf != nullptr && VPU::shouldRemoveOutputSparsity(nceOp) == VPU::SparsityRemovalFlag::Success;

    size_t wlMaxNumPerCluster = 0;
    size_t wlNumInTotal = 0;
    if (nceOp->getResult(0).getType().isa<VPU::SparseTensorType>() && !isSparseRemoved) {
        // NCE operations with sparse outputs must have all variants with the same number of channels
        // except of the last one which can have fewer channels than the rest
        const auto workloadInformation =
                getWorkLoadInformationForNCEWithSparseOutput(getArch(nceOp), perClusterShapes, supportedChannels);
        if (!workloadInformation.has_value()) {
            return false;
        }
        auto [wlMaxNumPerClusterTmp, wlNumInTotalTmp] = workloadInformation.value();
        wlMaxNumPerCluster = wlMaxNumPerClusterTmp;
        wlNumInTotal = wlNumInTotalTmp;
    } else {
        for (const auto& perClusterShape : perClusterShapes) {
            const auto perClusterChannel = perClusterShape[vpux::Dims4D::Act::C];
            auto wlChannels = splitWorkloadChannel(perClusterChannel, supportedChannels);
            // There may be some invalid tileChannel passed into. For example, channel is 16 but supportedChannels is
            // [32]. We can't split it over C in that case.
            if (wlChannels.size() == 0) {
                log.debug("splitWorkloadChannel failed: perClusterChannel - {0}, supportedChannels - {1}",
                          perClusterChannel, supportedChannels);
                return false;
            }
            if (wlMaxNumPerCluster < wlChannels.size()) {
                wlMaxNumPerCluster = wlChannels.size();
            }
            wlNumInTotal += wlChannels.size();
        }
    }

    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    // TODO: E#107973: more bigger / relaxing availableSlot to decrease tiling
    const auto maxAvailableSlots = VPUIP::getBarrierMaxVariantCount(nceOp);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(nceOp);
    const auto availableSlot = std::min(maxAvailableSlots, maxSlotsSum) / 2;

    // the variants count should be less than availableSlot on each cluster, otherwise there could be an illegal
    // scenario for the barrier
    //
    // the sum of variants count from all clusters should be less than maxSlotsSum, otherwise there could a serialized
    // dpu execution between clusters
    //
    // but if there's no tiling for the layer when we don't consider the constraint for the sum of variants, it's not
    // worth to introduce the extra tiling to parallelize dpu execution
    // it's because this extra tiling will be on channel dimension and it will introduce stride dma which takes more
    // time than serialized dpu execution
    const auto isTiled = llvm::any_of(outputTile.axis, [](auto axis) {
        return axis > 1;
    });
    if (!isTiled) {
        // for non-tiled operations it may not be performant to introduce extra tiling
        return wlMaxNumPerCluster <= availableSlot;
    }

    // allow all clusters to execute in parallel - driven by a single barrier
    return wlNumInTotal < maxSlotsSum;
}

std::optional<DimArr> getSEPConvTilingOrder(mlir::Operation* op) {
    auto nceConv = mlir::dyn_cast<VPU::NCEConvolutionOp>(op);
    if (nceConv == nullptr) {
        return std::nullopt;
    }
    auto sparseInput = nceConv.getInput().getType().dyn_cast<VPU::SparseTensorType>();
    if (sparseInput == nullptr) {
        return std::nullopt;
    }

    auto seAttr = sparseInput.getSeAttr().dyn_cast_or_null<VPU::SERollAttr>();
    if (seAttr != nullptr) {
        return VPU::getRollSEPConvTilingOrder(seAttr);
    }
    return std::nullopt;
}

mlir::FailureOr<OutputTiling> getBestHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode,
                                                           const std::shared_ptr<LayerCostModel>& costModel,
                                                           Logger log) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    VPUX_THROW_WHEN(nceOp == nullptr, "Operation '{0}' doesn't implement NCEop Interface", op->getName());
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);

    // Temporarily not apply cost-based tiling strategy to NCE ops with INT4 weights based on VPUNN cost.
    // This can be removed when VPUNN is upgraded to support INT4 data type, tracked in E#113316.
    if (VPU::isNCEWithInt4Weights(op)) {
        return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
    }

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    const auto outputShape = getShape(op->getResult(0));

    VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == 5,
                      "Unsupported operation '{0}' at '{1}', it has non 4D/5D result", op->getName(), op->getLoc());

    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    auto clusteredNCEOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    if (clusteredNCEOp != nullptr) {
        auto strategy = clusteredNCEOp.getMultiClusterStrategy();
        if (strategy.has_value()) {
            mcStrategy = strategy.value();
        }
    }

    mlir::FailureOr<OutputTiling> bestTilingStrategy = SmallVector({TileInfo(1)});
    auto bestCost = static_cast<double>(INVALID_COST_BASE);
    auto outputTiles = getAllHWLayerTilingStrategies(op, tilingMode, tileDimOrder, log);
    for (const auto& outputTile : outputTiles) {
        auto currentCost = costModel->getDPUandDMATimeCostWithCustomTiling(nceOp, mcStrategy, outputTile);
        log.trace("For Op {0} Name {1}: Got pair of MC Strategy {2} tiling strategy {3} with cost {4} ", op->getLoc(),
                  op->getName(), mcStrategy, outputTile[0].axis, currentCost);
        if (currentCost >= INVALID_COST_BASE) {
            log.trace("Choosing default tiling due to invalid cost");
            return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
        }
        if (currentCost < bestCost ||
            (currentCost == bestCost && isNewTileWithSameCostHasPotentialDMABenefits(
                                                op, bestTilingStrategy.value()[0].axis, outputTile[0].axis))) {
            bestTilingStrategy = outputTile;
            bestCost = currentCost;
        }
    }
    // if all feasible tile dim orders fail, return the default tiling strategy from HW layer without checking cost
    if (bestCost >= INVALID_COST_BASE) {
        return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
    }

    log.trace("For Op {0} Name {1}: Best pair of MC Strategy {2} tiling strategy {3} with cost {4} ", op->getLoc(),
              op->getName(), mcStrategy, bestTilingStrategy.value()[0].axis, bestCost);
    return bestTilingStrategy.value();
}

mlir::FailureOr<OutputTiling> getHWLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                       const std::shared_ptr<LayerCostModel>& costModel, Logger log) {
    log.trace("getHWLayerTilingStrategy for op '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    log.nest().trace("Enable Prefetch Tiling: {0}", enablePrefetchTiling);
    auto mode = getTilingSupportedMode(origOp, enablePrefetchTiling, log);
    log.nest().trace("Assigning {0} tiling strategy", getTilingModeStr(mode));
    return getBestHWLayerTilingStrategy(origOp, mode, costModel, log);
}

static constexpr auto MODE_ON = "true";
static constexpr auto MODE_OFF = "false";
static constexpr auto MODE_AUTO = "auto";

VPU::EnableShaveDDRAccessOptimization getShaveDDRAccessOptimizationMode(StringRef enableShaveDDRAccessOptimization) {
    std::string strMode = enableShaveDDRAccessOptimization.str();
    std::transform(strMode.begin(), strMode.end(), strMode.begin(), ::tolower);

    if (strMode == MODE_ON) {
        return VPU::EnableShaveDDRAccessOptimization::TRUE;
    } else if (strMode == MODE_OFF) {
        return VPU::EnableShaveDDRAccessOptimization::FALSE;
    } else if (strMode == MODE_AUTO) {
        VPUX_THROW("auto EnableShaveDDRAccessOptimization is not supported for now");
    }

    VPUX_THROW("Unknown value for the shave DDR access optimization mode: {0}", strMode);
}

}  // namespace VPU
}  // namespace vpux
