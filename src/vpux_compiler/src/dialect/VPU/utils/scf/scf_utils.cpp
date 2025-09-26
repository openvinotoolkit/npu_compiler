//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "mlir/IR/Attributes.h"

using namespace vpux::VPU;

mlir::OpFoldResult vpux::VPU::getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim) {
    const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());
    if (!outputType.hasStaticShape() && !operation->getOperand(0).hasOneUse()) {
        auto dimUser = llvm::find_if(operation->getOperand(0).getUsers(), [](auto* user) {
            return mlir::isa<mlir::tensor::DimOp>(user);
        });

        if (dimUser != operation->getOperand(0).getUsers().end()) {
            return dimUser->getResult(0);
        }
    }

    mlir::ReifiedRankedShapedTypeDims resultShape;
    if (mlir::failed(reifyResultShapes(builder, operation, resultShape))) {
        return builder.getIndexAttr(outputType.getDimSize(dim));
    }

    return resultShape[0][dim];
}

mlir::Value vpux::VPU::generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                                    const SCFTileInfo& inputTileInfo) {
    auto origType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());

    auto staticNewShape = mlir::getConstantIntValues(inputTileInfo.shape);
    if (origType.getShape().isStatic() && staticNewShape.has_value() &&
        llvm::equal(origType.getShape().raw(), staticNewShape.value())) {
        return origInput;
    }

    SmallVector<mlir::OpFoldResult> defaultStrides(inputTileInfo.offsets.size(), builder.getIndexAttr(1));

    auto extractTile = builder.create<mlir::tensor::ExtractSliceOp>(
            appendLoc(loc, "extractSlice"), origInput, inputTileInfo.offsets, inputTileInfo.shape, defaultStrides);

    auto newShape = getShape(extractTile.getResult());
    auto newType = origType.changeShape(ShapeRef(newShape));
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
        newType = boundedType.changeBounds(inputTileInfo.bounds);
    }

    // by default output type loses NPU-specific attributes so we have to set it manually
    extractTile->getResult(0).setType(newType);

    return extractTile;
}

mlir::Type vpux::VPU::extractResultType(mlir::Type origType, SCFShapeRef newShape, BoundsRef bounds) {
    auto ndTensorType = mlir::cast<vpux::NDTypeInterface>(origType);
    auto origElemType = ndTensorType.getElementType();

    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(origElemType),
                    "Per axis quantized types are not supported in scf");

    const auto tensorDesc =
            vpux::getTensorAttr(origElemType.getContext(), ndTensorType.getDimsOrder(), ndTensorType.getMemSpace(),
                                mlir::isa<Core::BoundedTensorType>(origType) ? bounds : BoundsRef{});

    SmallVector<mlir::Value> dynamicDims;  // unused cause for shape static dims are enough
    SmallVector<int64_t> staticDims;
    mlir::dispatchIndexOpFoldResults(newShape, dynamicDims, staticDims);
    return mlir::RankedTensorType::get(staticDims, origElemType, tensorDesc);
}

SCFTileInfo vpux::VPU::getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                              const SCFTileInfo& outputTile) {
    auto origWeightsTableShape = mlir::cast<mlir::ShapedType>(origWeightsTableType).getShape();

    SCFTileInfo weightsTableTile(origWeightsTableShape, builder);
    weightsTableTile.offsets[0] = outputTile.offsets[Dims4D::Act::C.ind()];
    weightsTableTile.shape[0] = outputTile.shape[Dims4D::Act::C.ind()];
    return weightsTableTile;
}

std::pair<std::optional<mlir::Range>, std::optional<int64_t>> vpux::VPU::solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
        const int64_t stride, const int64_t origInputSize, const int64_t origOutputSize,
        const std::pair<int64_t, int64_t>& origPadding, mlir::OpFoldResult& padBefore, mlir::OpFoldResult& padAfter) {
    auto zero = builder.getIndexAttr(0);
    auto one = builder.getIndexAttr(1);
    mlir::Range inputRange = {zero, zero, one};
    mlir::Range outputRange = {outputTile.offsets[dim.ind()], outputTile.shape[dim.ind()], one};

    // define dimensions (d0, d1, ...) as variables which are represented by loop dim identifier
    // and symbols (s0, s1, ...) which are either known constants or known attributes of operation
    mlir::AffineExpr s0, s1, d0, d1, d2;
    bindDims(builder.getContext(), d0, d1, d2);
    bindSymbols(builder.getContext(), s0, s1);

    mlir::AffineExpr requiredInputSizeExpr = (d0 - 1) * stride + kernel;
    auto requiredInputSizeMap = mlir::AffineMap::get(1, 0, {requiredInputSizeExpr}, builder.getContext());

    std::optional<int64_t> dimBound;
    if (!outputTile.bounds.raw().empty()) {
        auto outputTileBound = builder.getIntegerAttr(builder.getIndexType(), outputTile.bounds[dim]);
        SmallVector<mlir::Attribute> resultsAttrs;
        if (requiredInputSizeMap.constantFold({outputTileBound}, resultsAttrs).succeeded()) {
            if (auto result = mlir::dyn_cast<mlir::IntegerAttr>(resultsAttrs.front())) {
                dimBound = result.getInt() - origPadding.first - origPadding.second;
            }
        }
    }

    if (mlir::isConstantIntValue(outputTile.axis[dim.ind()], 1)) {
        return {std::nullopt, dimBound};
    }

    const auto hasPadBefore = origPadding.first != 0;
    const auto hasPadAfter = origPadding.second != 0;

    // input offset is based on output tile offset and operation's parameters
    // current calculation is
    // offset: max((output offset) * stride - padding, 0).
    // size: (output size - 1) * stride + kernel - padding
    // if operation has padding, the median tile size will be corrected later if needed
    if (!hasPadBefore && stride == 1) {
        inputRange.offset = outputRange.offset;
    } else {
        mlir::AffineExpr offsetExpr = d0 * stride - origPadding.first;
        auto offsetMap = mlir::AffineMap::get(1, 1, {offsetExpr, s0}, builder.getContext());
        inputRange.offset = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "inputOffset"), offsetMap,
                                                                      {outputRange.offset, zero});

        auto maxDiffMap = mlir::AffineMap::get(1, 2, {s0 - offsetExpr, s1}, builder.getContext());
        auto maxDiffValue = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "maxDiff"), maxDiffMap,
                                                                      {outputRange.offset, zero, zero});
        auto padBeforeMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padBefore = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "paddingBefore"), padBeforeMap,
                                                              {maxDiffValue, builder.getIndexAttr(origPadding.first)});
    }

    if (hasPadAfter) {
        auto maxDiffMap = mlir::AffineMap::get(2, 2, {d1 + requiredInputSizeExpr - s0, s1}, builder.getContext());
        auto maxDiffValue = mlir::affine::makeComposedFoldedAffineMax(
                builder, appendLoc(loc, "maxDiff"), maxDiffMap,
                {outputRange.size, inputRange.offset, builder.getIndexAttr(origInputSize), zero});
        auto padAfterMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padAfter = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "paddingAfter"), padAfterMap,
                                                             {maxDiffValue, builder.getIndexAttr(origPadding.second)});
    }

    const auto numTiles = mlir::getConstantIntValue(outputTile.axis[dim.ind()]);
    const bool hasTwoTiles = numTiles.has_value() && numTiles.value() == 2;
    const bool outputSizeDivisible = numTiles.has_value() && origOutputSize % numTiles.value() == 0;
    const bool symmetricPadding = origPadding.first == origPadding.second;
    if (hasTwoTiles && outputSizeDivisible && symmetricPadding) {
        auto sizeMap = mlir::AffineMap::get(1, 1, {requiredInputSizeExpr - s0}, builder.getContext());
        inputRange.size = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputSize"), sizeMap,
                {outputRange.size, builder.getIndexAttr(origPadding.first)});
    } else {
        auto sizeMap = mlir::AffineMap::get(3, 0, {requiredInputSizeExpr - d1 - d2}, builder.getContext());
        inputRange.size = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputSize"), sizeMap,
                {outputRange.size, hasPadBefore ? padBefore : zero, hasPadAfter ? padAfter : zero});
    }

    return {inputRange, dimBound};
}

bool vpux::VPU::checkFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate) {
    // TODO E-172888 rewrite unified code for checking compatibility with current VF

    if (!mlir::isa<mlir::TilingInterface>(producerCandidate.getOwner())) {
        return false;
    }

    if (VPU::isPureViewOp(producerCandidate.getOwner()) || VPU::isPureViewOp(consumer.getOwner())) {
        return true;
    }

    const auto hasMCStategy = [](mlir::Operation* operation) {
        auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    auto consumerHasStrategy = hasMCStategy(consumer.getOwner());
    auto producerHasStrategy = hasMCStategy(producerCandidate.getOwner());

    if (!consumerHasStrategy && !producerHasStrategy) {
        return true;
    }

    if (consumerHasStrategy ^ producerHasStrategy) {
        return false;
    }

    auto producerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(producerCandidate.getOwner());
    auto consumerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(consumer.getOwner());

    VPU::SiblingOpsAnalysis siblingAnalisys(consumer.getOwner());

    auto consumerDistrType = mlir::cast<VPU::DistributedTensorType>(
            consumerClusterOp.getDistributedTypeForOpOperand(consumer, false, siblingAnalisys));
    auto producerDistrType = mlir::cast<VPU::DistributedTensorType>(producerClusterOp.getDistributedTypeForOpResult(
            producerCandidate, producerClusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false));

    return areDistributionAttrsCompatible(producerDistrType, consumerDistrType, true).succeeded();
}

bool vpux::VPU::isNceOpWithPadAttr(mlir::Operation* op) {
    return (mlir::isa<mlir::TilingInterface>(op) && mlir::isa<VPU::NCEOpInterface>(op) && op->hasAttr("pad"));
}

void iterativeDfs(llvm::ArrayRef<mlir::Operation*> startNodes,
                  llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, 16>(mlir::Operation*)> getNeighbors,
                  llvm::function_ref<void(mlir::Operation*)> visitPostOrder,
                  llvm::function_ref<bool(mlir::Operation*)> stopCheckFn) {
    llvm::SmallPtrSet<mlir::Operation*, 32> visited;
    struct Node {
        mlir::Operation* operation;
        bool visited;
    };

    llvm::SmallVector<Node> stack;
    for (auto node : startNodes) {
        if (node == nullptr || visited.count(node)) {
            continue;
        }

        stack.push_back({node, false});

        while (!stack.empty()) {
            auto& currentNode = stack.back();
            mlir::Operation* currentOp = currentNode.operation;

            if (!currentNode.visited) {
                if (!visited.insert(currentOp).second) {
                    stack.pop_back();
                    continue;
                }

                if (stopCheckFn && stopCheckFn(currentOp)) {
                    visitPostOrder(currentOp);
                    stack.pop_back();
                    continue;
                }

                currentNode.visited = true;
                auto neighbors = getNeighbors(currentOp);
                for (auto neighbor : neighbors) {
                    if (neighbor && !visited.count(neighbor)) {
                        stack.push_back({neighbor, false});
                    }
                }
            } else {
                visitPostOrder(currentOp);
                stack.pop_back();
            }
        }
    }
}

llvm::SmallVector<mlir::Operation*> vpux::VPU::collectOpsInTopologicalOrder(
        llvm::ArrayRef<mlir::Operation*> startNodes,
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, 16>(mlir::Operation*)> getNeighbors,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn) {
    auto defGetNeighbors = [&](mlir::Operation* op) -> llvm::SmallSetVector<mlir::Operation*, 16> {
        llvm::SmallSetVector<mlir::Operation*, 16> neighbors;
        for (auto operand : op->getOperands()) {
            if (auto definingOp = operand.getDefiningOp()) {
                neighbors.insert(definingOp);
            }
        }
        return neighbors;
    };

    llvm::SmallVector<mlir::Operation*> sortedOps;
    iterativeDfs(
            startNodes, getNeighbors ? getNeighbors : defGetNeighbors,
            [&](mlir::Operation* op) {
                sortedOps.push_back(op);
            },
            stopCheckFn);

    return sortedOps;
}

AffineChainUtils::AffineChainUtils(Logger log): _log(log) {
}

std::pair<mlir::AffineMap, mlir::ValueRange> AffineChainUtils::getAffineMapAndOperands(mlir::Operation* op) {
    if (auto affineOp = mlir::dyn_cast<mlir::affine::AffineMinOp>(op)) {
        return {affineOp.getAffineMap(), affineOp.getOperands()};
    }
    if (auto affineOp = mlir::dyn_cast<mlir::affine::AffineMaxOp>(op)) {
        return {affineOp.getAffineMap(), affineOp.getOperands()};
    }
    if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
        return {applyOp.getAffineMap(), applyOp.getOperands()};
    }

    VPUX_THROW("Unsupported affine operation type: {0}", op->getName());
}

int64_t AffineChainUtils::getAffineResult(mlir::Operation* op, llvm::ArrayRef<int64_t> results) {
    if (results.empty()) {
        VPUX_THROW("Empty results array for operation: {0}", op->getName());
    }

    if (mlir::isa<mlir::affine::AffineMinOp>(op)) {
        return *llvm::min_element(results);
    }
    if (mlir::isa<mlir::affine::AffineMaxOp>(op)) {
        return *llvm::max_element(results);
    }
    if (mlir::isa<mlir::affine::AffineApplyOp>(op)) {
        return results[0];
    }

    VPUX_THROW("Unsupported affine operation type: {0}", op->getName());
}

std::optional<int64_t> AffineChainUtils::getIntegerFromValue(mlir::Value value) {
    auto getConstantInt = [](mlir::Value val) -> std::optional<int64_t> {
        if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(val.getDefiningOp())) {
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                return intAttr.getInt();
            }
        }
        return std::nullopt;
    };

    if (auto dimOp = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(value.getDefiningOp())) {
        auto dimIdx = getConstantInt(dimOp.getIndex());
        if (!dimIdx.has_value()) {
            _log.warning("Dim index is not a constant!");
            return std::nullopt;
        }

        if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(dimOp.getSource().getType())) {
            const auto dimIndex = dimIdx.value();
            if (rankedType.hasStaticShape() && dimIndex < rankedType.getRank()) {
                return rankedType.getShape()[dimIndex];
            }

            if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(rankedType)) {
                _log.trace("Got BoundedTensorType for dim value extraction. Use bounds attribute for shape extraction");
                auto bounds = boundedType.getBounds().raw();
                if (dimIndex < static_cast<int64_t>(bounds.size())) {
                    return bounds[dimIndex];
                }
            }
        }

        return std::nullopt;
    }

    return getConstantInt(value);
}

void AffineChainUtils::updateChainCache(mlir::Value val, const llvm::SmallSetVector<mlir::Operation*, 4>& chain) const {
    _chainCache[val] = chain;
}

llvm::SmallSetVector<mlir::Operation*, 4> AffineChainUtils::collectAffineOpsChain(mlir::Value val) {
    if (auto it = _chainCache.find(val); it != _chainCache.end()) {
        return it->second;
    }

    if (!val || !val.getDefiningOp()) {
        return {};
    }

    llvm::SmallVector<mlir::Operation*, 4> startNodes = {val.getDefiningOp()};

    auto stopSearch = [](mlir::Operation* op) {
        return mlir::isa<mlir::tensor::DimOp>(op);
    };

    auto getNeighbors = [](mlir::Operation* op) -> llvm::SmallSetVector<mlir::Operation*, 16> {
        llvm::SmallSetVector<mlir::Operation*, 16> neighbors;
        for (auto operand : op->getOperands()) {
            if (auto definingOp = operand.getDefiningOp()) {
                if (!mlir::isa<mlir::tensor::DimOp>(definingOp)) {
                    neighbors.insert(definingOp);
                }
            }
        }
        return neighbors;
    };

    auto results = vpux::VPU::collectOpsInTopologicalOrder(startNodes, getNeighbors, stopSearch);
    llvm::SmallSetVector<mlir::Operation*, 4> affineChain(results.begin(), results.end());

    updateChainCache(val, affineChain);
    return affineChain;
}

std::optional<int64_t> AffineChainUtils::processAffineCallChain(
        mlir::Value val, llvm::DenseMap<mlir::Value, SmallVector<int64_t>>& valueMap) {
    auto callChain = collectAffineOpsChain(val);
    if (callChain.empty()) {
        _log.trace("Empty affine chain for value");
        return std::nullopt;
    }

    // Collect block operands
    llvm::DenseSet<mlir::Value> blockOperands;
    for (auto op : callChain) {
        for (auto operand : op->getOperands()) {
            if (operand.getDefiningOp() == nullptr) {
                blockOperands.insert(operand);
            }
        }
    }

    if (blockOperands.empty()) {
        _log.trace("No block operands found in affine chain");
        return std::nullopt;
    }

    // valueMap contains the values for block operands to use during evaluation.
    // If no values are found, initialize with default values obtained from the scf::ForOp bounds.
    if (valueMap.empty()) {
        if (blockOperands.size() != 1) {
            _log.trace("Multiple block operands found, cannot initialize default values");
            return std::nullopt;
        }

        auto blockOperand = *blockOperands.begin();
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(blockOperand)) {
            if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
                auto low = getIntegerFromValue(forOp.getLowerBound());
                auto step = getIntegerFromValue(forOp.getStep());
                // For upper bound, if the defining op is tensor::DimOp, get the value from bounds attribute if
                // available
                auto high = getIntegerFromValue(forOp.getUpperBound());

                SmallVector<int64_t> vals;
                if (low.has_value() && step.has_value() && high.has_value() && step.value() > 0) {
                    for (int64_t i = low.value(); i < high.value(); i += step.value()) {
                        vals.push_back(i);
                    }
                }
                valueMap[blockOperand] = vals.empty() ? SmallVector<int64_t>{0} : vals;
            } else {
                valueMap[blockOperand] = SmallVector<int64_t>{0};
            }
        } else {
            valueMap[blockOperand] = SmallVector<int64_t>{0};
        }
    }

    // Find primary operand with valid values
    mlir::Value primaryOperand;
    SmallVector<int64_t> valueRange;
    for (auto blockOperand : blockOperands) {
        auto it = valueMap.find(blockOperand);
        if (it != valueMap.end() && !it->second.empty()) {
            primaryOperand = blockOperand;
            valueRange = it->second;
            break;
        }
    }

    if (!primaryOperand || valueRange.empty()) {
        _log.trace("No valid primary operand found");
        return std::nullopt;
    }

    // Evaluate chain for each value
    SmallVector<int64_t> resultRange;
    for (int64_t value : valueRange) {
        llvm::DenseMap<mlir::Value, int64_t> localOperandMap;
        localOperandMap[primaryOperand] = value;

        bool success = true;
        for (auto currentOp : callChain) {
            auto [affineMap, mapOperands] = getAffineMapAndOperands(currentOp);
            SmallVector<mlir::Attribute> operandAttrs;

            for (auto operand : mapOperands) {
                int64_t operandValue;
                auto it = localOperandMap.find(operand);
                if (it != localOperandMap.end()) {
                    operandValue = it->second;
                } else {
                    auto intVal = getIntegerFromValue(operand);
                    if (!intVal.has_value()) {
                        success = false;
                        break;
                    }
                    operandValue = intVal.value();
                    localOperandMap[operand] = operandValue;
                }
                operandAttrs.push_back(mlir::IntegerAttr::get(operand.getType(), operandValue));
            }

            if (!success) {
                break;
            }

            SmallVector<mlir::Attribute> resultsAttrs;
            if (affineMap.constantFold(operandAttrs, resultsAttrs).failed()) {
                success = false;
                break;
            }

            SmallVector<int64_t> results;
            for (auto attr : resultsAttrs) {
                results.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
            }

            int64_t result = getAffineResult(currentOp, results);
            localOperandMap[currentOp->getResult(0)] = result;
        }

        if (!success) {
            return std::nullopt;
        }

        auto it = localOperandMap.find(val);
        if (it != localOperandMap.end()) {
            resultRange.push_back(it->second);
        } else {
            return std::nullopt;
        }
    }

    return resultRange.empty() ? std::nullopt : std::make_optional(*llvm::max_element(resultRange));
}

std::optional<int64_t> AffineChainUtils::getOpFoldResultValue(
        mlir::OpFoldResult val, llvm::DenseMap<mlir::Value, SmallVector<int64_t>>& valueMap) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(val)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
            return intAttr.getInt();
        }
        return std::nullopt;
    }

    if (auto value = mlir::dyn_cast<mlir::Value>(val)) {
        if (auto constantOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValueAttr())) {
                return intAttr.getInt();
            }
        }
        return processAffineCallChain(value, valueMap);
    }

    return std::nullopt;
}
