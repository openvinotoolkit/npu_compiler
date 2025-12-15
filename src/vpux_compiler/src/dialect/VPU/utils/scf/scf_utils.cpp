//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include <vpux/compiler/utils/infer_output_shape.hpp>
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include "mlir/IR/Attributes.h"

using namespace vpux::VPU;

// Constants for container sizes
static constexpr size_t DEFAULT_OPERATION_SET_SIZE = 32;
static constexpr size_t DEFAULT_NEIGHBOR_SET_SIZE = 16;

mlir::LogicalResult vpux::VPU::getResultTileBounds(mlir::Operation* operation, unsigned resultNumber,
                                                   DimArrRef tilingDims, ArrayRef<mlir::OpFoldResult> sizes,
                                                   Bounds& resultBounds) {
    if (tilingDims.empty() || sizes.empty()) {
        return mlir::failure();
    }

    auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(resultNumber).getType());
    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType);
    if (boundedType == nullptr) {
        return mlir::failure();
    }

    resultBounds = boundedType.getBounds().toValues();

    const auto strategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
    SmallVector<mlir::Operation*> operationList = {operation};
    const auto tiles = fillDividedTiles(operationList, strategy, ShapeRef(boundedType.getBounds().raw()));
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }
    for (auto dim : tilingDims) {
        resultBounds[dim] = tiles.value()[resultNumber].shape[dim];
    }

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim) {
    const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());

    mlir::ReifiedRankedShapedTypeDims resultShape;
    if (mlir::failed(reifyResultShapes(builder, operation, resultShape))) {
        return builder.getIndexAttr(outputType.getDimSize(dim));
    }

    return resultShape[0][dim];
}

mlir::Value vpux::VPU::generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                                    const SCFTileInfo& inputTileInfo, SmallVector<mlir::Operation*>& generatedSlices) {
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

    /*For cases where we cannot evenly divide tensor with static shapes into pieces of equal size, SCF tiling
     * generates dynamic shapes for extract and pad from static shapes. For such cases setting the bounds here
     * based on the original input shape. For other cases bounds are inferred from available input bounds*/

    auto newType = origType.changeShape(ShapeRef(newShape));
    if (newShape.isDynamic()) {
        if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
            newType = boundedType.changeBounds(inputTileInfo.bounds);
        } else {
            auto origInputShape = to_small_vector(getShape(origInput));
            SmallVector<int64_t, 4> boundsValue(newShape.begin(), newShape.end());
            for (size_t ind = 0, sz = boundsValue.size(); ind < sz; ++ind) {
                if (boundsValue[ind] == mlir::ShapedType::kDynamic) {
                    boundsValue[ind] = origInputShape[ind];
                }
            }
            auto newBounds = vpux::BoundsRef(ArrayRef<int64_t>(boundsValue.data(), boundsValue.size()));
            newType = Core::BoundedTensorType::get(newType, newBounds);
        }
    }

    // by default output type loses NPU-specific attributes so we have to set it manually
    extractTile->getResult(0).setType(newType);

    generatedSlices.emplace_back(extractTile);

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

void iterativeDfs(
        llvm::ArrayRef<mlir::Operation*> startNodes,
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE>(mlir::Operation*)>
                getNeighbors,
        llvm::function_ref<void(mlir::Operation*)> visitPostOrder,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn) {
    llvm::SmallPtrSet<mlir::Operation*, DEFAULT_OPERATION_SET_SIZE> visited;
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
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE>(mlir::Operation*)>
                getNeighbors,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn) {
    llvm::SmallVector<mlir::Operation*> sortedOps;
    iterativeDfs(
            startNodes, getNeighbors,
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

// Use bounds attribute on the dimOp for fetching integer value
std::optional<int64_t> AffineChainUtils::getIntValueFromDimOp(mlir::tensor::DimOp dimOp) {
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

std::optional<int64_t> AffineChainUtils::getIntegerFromValue(mlir::Value value, bool processOpChain) {
    if (auto dimOp = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(value.getDefiningOp())) {
        return getIntValueFromDimOp(dimOp);
    }

    auto intVal = getConstantInt(value);
    if (intVal.has_value()) {
        return intVal;
    }

    // Special case to handle the opChain
    if (processOpChain) {
        llvm::DenseMap<mlir::Value, int64_t> localOperandMap;
        auto opChain = collectAffineOpsChain(value);

        if (evaluateOpChain(opChain, localOperandMap)) {
            if (localOperandMap.contains(value)) {
                return localOperandMap[value];
            }
        }
    }

    return std::nullopt;
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

    auto getNeighbors = [&](mlir::Operation* op) -> llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE> {
        llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE> neighbors;
        for (auto operand : op->getOperands()) {
            if (auto definingOp = operand.getDefiningOp()) {
                if (!stopSearch(definingOp)) {
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

bool AffineChainUtils::processAffineOp(mlir::Operation* op, llvm::DenseMap<mlir::Value, int64_t>& valueMap) {
    auto [affineMap, mapOperands] = getAffineMapAndOperands(op);
    SmallVector<mlir::Attribute> operandAttrs;
    bool success = true;
    for (auto operand : mapOperands) {
        int64_t operandValue;
        auto it = valueMap.find(operand);
        if (it != valueMap.end()) {
            operandValue = it->second;
        } else {
            auto intVal = getIntegerFromValue(operand);
            if (!intVal.has_value()) {
                success = false;
                break;
            }
            operandValue = intVal.value();
            valueMap[operand] = operandValue;
        }
        operandAttrs.push_back(mlir::IntegerAttr::get(operand.getType(), operandValue));
    }

    if (!success) {
        return false;
    }

    SmallVector<mlir::Attribute> resultsAttrs;
    if (affineMap.constantFold(operandAttrs, resultsAttrs).failed()) {
        return false;
    }

    SmallVector<int64_t> results;
    for (auto attr : resultsAttrs) {
        results.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }

    int64_t result = getAffineResult(op, results);
    valueMap[op->getResult(0)] = result;
    return true;
}

bool AffineChainUtils::processArithOp(mlir::Operation* op, llvm::DenseMap<mlir::Value, int64_t>& valueMap) {
    if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr())) {
            valueMap[op->getResult(0)] = intAttr.getInt();
            return true;
        }

        return false;
    }

    for (auto operand : op->getOperands()) {
        if (!valueMap.contains(operand)) {
            auto intVal = getIntegerFromValue(operand);
            if (intVal.has_value()) {
                valueMap[operand] = intVal.value();
            } else {
                return false;
            }
        }
    }

    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<mlir::arith::AddIOp>([&](auto op) {
                auto addOp = mlir::cast<mlir::arith::AddIOp>(op);
                auto resultValue = valueMap[addOp.getLhs()] + valueMap[addOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::SubIOp>([&](auto op) {
                auto subOp = mlir::cast<mlir::arith::SubIOp>(op);
                auto resultValue = valueMap[subOp.getLhs()] - valueMap[subOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::DivSIOp>([&](auto op) {
                auto divOp = mlir::cast<mlir::arith::DivSIOp>(op);
                if (valueMap[divOp.getRhs()] == 0) {
                    _log.trace("Division by zero encountered in DivSIOp");
                    return false;
                }
                auto resultValue = valueMap[divOp.getLhs()] / valueMap[divOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::DivUIOp>([&](auto op) {
                auto divOp = mlir::cast<mlir::arith::DivUIOp>(op);
                if (valueMap[divOp.getRhs()] == 0) {
                    _log.trace("Division by zero encountered in DivUIOp");
                    return false;
                }
                auto resultValue = valueMap[divOp.getLhs()] / valueMap[divOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::MulIOp>([&](auto op) {
                auto mulOp = mlir::cast<mlir::arith::MulIOp>(op);
                auto resultValue = valueMap[mulOp.getLhs()] * valueMap[mulOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Default([&](mlir::Operation* op) {
                _log.trace("Unsupported arith operation encountered while evaluating op chain: {0}", op->getName());
                return false;
            });
}

bool AffineChainUtils::evaluateOpChain(llvm::SmallSetVector<mlir::Operation*, 4>& opChain,
                                       llvm::DenseMap<mlir::Value, int64_t>& localOperandMap) {
    for (auto op : opChain) {
        if (mlir::isa<mlir::affine::AffineDialect>(op->getDialect())) {
            auto result = processAffineOp(op, localOperandMap);
            if (!result) {
                _log.trace("Failed to process affine operation: {0}", op->getName());
                return false;
            }
        } else if (mlir::isa<mlir::arith::ArithDialect>(op->getDialect())) {
            auto result = processArithOp(op, localOperandMap);
            if (!result) {
                _log.trace("Failed to process arith operation: {0}", op->getName());
                return false;
            }
        } else {
            _log.trace("Unsupported dialect ({0}) encountered while evaluating op chain", op->getDialect());
            return false;
        }
    }

    return true;
}

std::optional<llvm::SmallVector<int64_t>> AffineChainUtils::processAffineCallChain(
        mlir::Value val, llvm::DenseMap<mlir::Value, SmallVector<int64_t>>& valueMap, AffineChainUtils::MODE mode) {
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

    if (blockOperands.size() != 1) {
        _log.trace("Multiple block operands found in the opChain. Analysis does not support this usecase");
        return std::nullopt;
    }

    // valueMap contains the values for block operands to use during evaluation.
    // If no values are computed, initialize with default values obtained from the scf::ForOp bounds.
    if (valueMap.empty()) {
        auto blockOperand = *blockOperands.begin();
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(blockOperand)) {
            if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
                auto low = getIntegerFromValue(forOp.getLowerBound());
                auto step = getIntegerFromValue(forOp.getStep());
                // For upper bound, there are three possible cases:
                // 1. If the defining op is tensor::DimOp, get the value from bounds attribute if available
                // 2. If the defining op is a constant, get the value directly
                // 3. If the defining op is chain of ops, evaluate op chain
                auto high = getIntegerFromValue(forOp.getUpperBound(), true);

                _log.trace("ForOp bounds: low = {0}, step = {1}, high = {2}", low, step, high);
                SmallVector<int64_t> vals;
                if (low.has_value() && step.has_value() && high.has_value() && step.value() > 0) {
                    for (int64_t i = low.value(); i < high.value(); i += step.value()) {
                        vals.push_back(i);
                    }
                } else {
                    _log.trace("Failed to get bounds for ForOp. Defaulting to 0");
                }
                valueMap[blockOperand] = vals.empty() ? SmallVector<int64_t>{0} : vals;
            } else {
                valueMap[blockOperand] = SmallVector<int64_t>{0};
            }
        } else {
            _log.warning("Owner of Block operand is not forOp. Use default value of 0");
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
        if (!evaluateOpChain(callChain, localOperandMap)) {
            return std::nullopt;
        }

        auto it = localOperandMap.find(val);
        if (it != localOperandMap.end()) {
            resultRange.push_back(it->second);
        } else {
            return std::nullopt;
        }
    }

    if (resultRange.empty()) {
        return std::nullopt;
    }

    if (mode == MODE::MAX_VALUE) {
        return llvm::SmallVector<int64_t>({*llvm::max_element(resultRange)});
    }

    return resultRange;
}

std::optional<llvm::SmallVector<int64_t>> AffineChainUtils::getOpFoldResultValue(
        mlir::OpFoldResult val, llvm::DenseMap<mlir::Value, SmallVector<int64_t>>& valueMap,
        AffineChainUtils::MODE mode) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(val)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
            return llvm::SmallVector<int64_t>{intAttr.getInt()};
        }
        return std::nullopt;
    }

    if (auto value = mlir::dyn_cast<mlir::Value>(val)) {
        if (auto constantOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValueAttr())) {
                return llvm::SmallVector<int64_t>{intAttr.getInt()};
            }
        }
        return processAffineCallChain(value, valueMap, mode);
    }

    return std::nullopt;
}

/**
 * @brief Clones an existing MLIR function operation with optional type modification
 *
 * This function creates a deep copy of an MLIR function operation, allowing for optional
 * function type changes. It performs a complete clone of the function body, including
 * all blocks, operations, and value mappings. When a new function type is provided,
 * it also performs type inference for VPU dialect operations and updates the function's
 * return types based on the terminator operations.
 *
 * @param originalFunc The source function operation to be cloned
 * @param newName The name for the newly created function
 * @param moduleOp The target module where the new function will be inserted
 * @param newFuncType Optional new function type for the cloned function. If nullptr,
 *                    the original function's type is preserved
 *
 * @return The newly created function operation that is a clone of the original
 *
 * @note The cloned function is positioned immediately after the original function
 *       in the module. When newFuncType is provided, VPU dialect operations undergo
 *       shape inference and the function's return types are updated based on the
 *       actual terminator operand types.
 */
mlir::func::FuncOp vpux::VPU::cloneFuncOp(mlir::func::FuncOp originalFunc, const std::string& newName,
                                          mlir::FunctionType newFuncType) {
    assert(newFuncType != nullptr && "newFuncType should be a valid FunctionType");
    auto moduleOp = vpux::getModuleOp(originalFunc);
    mlir::OpBuilder builder(moduleOp.getContext());
    auto newFunc = builder.create<mlir::func::FuncOp>(takeOpLoc(originalFunc, newName), newName, newFuncType);
    moduleOp.push_back(newFunc);
    newFunc->moveAfter(originalFunc);

    mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
    for (auto& oldBlock : originalFunc.getBody()) {
        auto* newBlock = newFunc.addEntryBlock();
        for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
            oldToNewMap[oldArg] = newArg;
        }

        builder.setInsertionPointToStart(newBlock);
        for (auto& oldOp : oldBlock.getOperations()) {
            mlir::IRMapping mapper;
            for (auto operand : oldOp.getOperands()) {
                mapper.map(operand, oldToNewMap[operand]);
            }

            auto newOp = builder.clone(oldOp, mapper);
            if (mlir::isa<VPU::VPUDialect>(newOp->getDialect())) {
                vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);
            }

            for (size_t i = 0; i < oldOp.getNumResults(); ++i) {
                oldToNewMap[oldOp.getResult(i)] = newOp->getResult(i);
            }
        }
    }

    SmallVector<mlir::Type> newReturnTypes;
    for (auto& block : newFunc.getBody()) {
        auto terminatorOp = block.getTerminator();
        for (auto operands : terminatorOp->getOperands()) {
            newReturnTypes.push_back(operands.getType());
        }
    }

    auto modifiedType =
            mlir::FunctionType::get(newFunc.getContext(), newFunc.getFunctionType().getInputs(), newReturnTypes);
    newFunc.setType(modifiedType);
    return newFunc;
}

mlir::RankedTensorType vpux::VPU::removeBoundsAttr(mlir::RankedTensorType type) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    const auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                                ndType.getMemSpace(), {}, {});
    return tensorType;
}

// Check if all operands of op are defined before 'beforeOp' and op has no side effects
bool isSafeToMoveBefore(mlir::Operation* op, mlir::Operation* beforeOp) {
    // Check SSA dependencies
    for (auto operand : op->getOperands()) {
        if (auto definingOp = operand.getDefiningOp()) {
            if (definingOp->getBlock() != op->getBlock()) {
                continue;
            }

            if (!definingOp->isBeforeInBlock(beforeOp)) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Moves affine and arithmetic operations early in the execution order within a block
 *
 * This function optimizes the placement of affine dialect and arithmetic dialect operations
 * by moving them as early as possible in the block's execution order, while maintaining
 * correctness. It also handles scf.if operations that return only index types by treating
 * them similarly to affine/arithmetic operations.
 *
 * The function performs a two-phase operation:
 * 1. Identifies all affine/arithmetic operations and scf.if operations that return only
 *    index types as candidates for early movement
 * 2. For each candidate operation, finds the earliest safe position in the block where
 *    it can be moved without violating dependencies
 *
 * @param block The MLIR block within which to perform the optimization
 */
void vpux::VPU::moveAffineArithOpsEarly(mlir::Block& block) {
    SmallVector<mlir::Operation*> affineArithOps;

    // Collect affine/arithmetic operations and qualifying scf.if operations
    for (auto& op : block) {
        if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(op.getDialect())) {
            affineArithOps.push_back(&op);
            continue;
        }

        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            // Only move scf.if operations that return only index types
            bool allIndexTypes = llvm::all_of(ifOp.getResultTypes(), [](mlir::Type type) {
                return mlir::isa<mlir::IndexType>(type);
            });
            if (allIndexTypes) {
                affineArithOps.push_back(&op);
            }
        }
    }

    // Move operations to earliest safe positions
    for (auto* op : affineArithOps) {
        mlir::Operation* targetPosition = nullptr;

        // Find the earliest position where this operation can be safely moved
        for (auto& candidate : block) {
            if (&candidate == op) {
                break;  // Don't move past current position
            }

            // Skip other affine/arithmetic operations - we want to move before them
            if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(candidate.getDialect())) {
                continue;
            }

            // Check if it's safe to move before this candidate
            if (isSafeToMoveBefore(op, &candidate)) {
                targetPosition = &candidate;
                break;
            }
        }

        // Only move if we found a valid earlier position
        if (targetPosition != nullptr) {
            op->moveBefore(targetPosition);
        }
    }
}

void vpux::VPU::addCheckForBlockSize(mlir::OpBuilder& builder, mlir::tensor::DimOp dimOp, mlir::Value blockSize,
                                     mlir::func::FuncOp funcOp, llvm::StringRef errorMsg) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto blockSizeDefiningOp = blockSize.getDefiningOp();
    mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
    if (dimOp->getBlock() == blockSizeDefiningOp->getBlock()) {
        if (dimOp->isBeforeInBlock(blockSizeDefiningOp)) {
            builder.setInsertionPointAfter(blockSizeDefiningOp);
            loc = appendLoc(blockSizeDefiningOp->getLoc(), "tile0_check");
        } else {
            builder.setInsertionPointAfter(dimOp);
            loc = appendLoc(dimOp->getLoc(), "tile0_check");
        }
    } else {
        mlir::DominanceInfo dom(funcOp);
        if (dom.dominates(blockSizeDefiningOp, dimOp)) {
            builder.setInsertionPointAfter(dimOp);
            loc = appendLoc(dimOp->getLoc(), "tile0_check");
        } else {
            builder.setInsertionPointAfter(blockSizeDefiningOp);
            loc = appendLoc(blockSizeDefiningOp->getLoc(), "tile0_check");
        }
    }

    auto stepGreaterThanBound =
            builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, dimOp, blockSize);
    builder.create<mlir::cf::AssertOp>(takeOpLoc(stepGreaterThanBound, "assert"), stepGreaterThanBound, errorMsg);
}

mlir::LogicalResult vpux::VPU::getTensorDimOpFromIndex(mlir::OpBuilder& builder, mlir::Value tensor, int64_t dimIdx,
                                                       mlir::tensor::DimOp& dimOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    bool tensorIsBlockArg = false;
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
        auto blockParentOp = blockArg.getOwner()->getParentOp();
        tensorIsBlockArg = true;
        while (blockParentOp != nullptr) {
            if (mlir::isa<mlir::func::FuncOp>(blockParentOp)) {
                break;
            }

            auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockParentOp);
            if (forOp == nullptr) {
                return vpux::errorAt(blockParentOp->getLoc(),
                                     "Expected parent scf.for operation for block argument but got {0}",
                                     blockParentOp->getName());
            }

            unsigned idx = blockArg.getArgNumber() - 1;
            mlir::Value initVal = forOp.getInitArgs()[idx];
            auto definingOp = initVal.getDefiningOp();
            if (definingOp != nullptr) {
                tensor = initVal;
                tensorIsBlockArg = false;
                break;
            }

            blockParentOp = blockParentOp->getParentOp();
        }
    }

    if (tensorIsBlockArg) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor);
        auto blockParentOp = blockArg.getOwner()->getParentOp();
        builder.setInsertionPointToStart(blockArg.getOwner());
        dimOp = builder.create<mlir::tensor::DimOp>(takeOpLoc(blockParentOp, "dim_" + std::to_string(dimIdx)), tensor,
                                                    dimIdx);
    } else {
        auto loc = tensor.getDefiningOp();
        builder.setInsertionPointAfter(loc);
        dimOp = builder.create<mlir::tensor::DimOp>(takeOpLoc(loc, "dim_" + std::to_string(dimIdx)), tensor, dimIdx);
    }
    return mlir::success();
}
