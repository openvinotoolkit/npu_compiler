//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_analyzer.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/dialect_processors.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Operation.h>

#include <array>
#include <climits>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace vpux::VPU {

constexpr llvm::StringLiteral EVALUATED_ATTR_NAME = "dynamicDimsEvaluatedSizes";

void OpDebugInfo::printDimensionData(std::ostream& stream, size_t indentLevel, const char* header,
                                     ArrayRef<SmallVector<int64_t>> data) {
    constexpr int VALUE_WIDTH = 3;       // Fixed width for all values
    constexpr int DIM_LABEL_WIDTH = 10;  // Fixed width for dimension labels

    // Save stream format state to restore later
    std::ios_base::fmtflags originalFlags = stream.flags();

    // Create indentation string
    std::string headerIndent(indentLevel, ' ');
    std::string msgIndent(indentLevel + 1, ' ');

    // Check if all dimensions have single values for compact format
    bool allSingleValues = !llvm::any_of(data, [](const auto& values) {
        return values.size() != 1;
    });
    if (allSingleValues) {
        // Compact format: print all values in single line
        stream << headerIndent << header << ": [";
        bool first = true;
        for (const auto& values : data) {
            if (!first) {
                stream << ", ";
            }
            stream << values[0];
            first = false;
        }
        stream << "]\n";

        stream.flags(originalFlags);
        return;
    }

    // Detailed format: print header and each dimension separately
    stream << headerIndent << "  " << std::left << std::setw(DIM_LABEL_WIDTH) << header << "\n";

    auto nchwStr = DimsOrder::NCHW.getCanonicalName();
    // For each dynamic dimension, print all values from all combinations
    for (auto [dynamicDim, values] : llvm::enumerate(data)) {
        if (values.empty()) {
            continue;
        }

        // Get dimension label
        auto label = (dynamicDim < nchwStr.size()) ? nchwStr[dynamicDim] : '?';
        std::string dimLabel = "  Dim " + std::to_string(dynamicDim) + " (" + label + "):";
        stream << msgIndent << std::right << std::setw(DIM_LABEL_WIDTH + 2) << dimLabel << " [";

        // Extract values for this dimension from all combinations
        bool first = true;
        for (const auto& value : values) {
            if (!first) {
                stream << ", ";
            }
            stream << std::setw(VALUE_WIDTH) << value;
            first = false;
        }

        stream << "]\n";
    }
    stream.flags(originalFlags);
}

// OffsetSizeStrideOpDebugInfo implementation
std::tuple<SmallVector<mlir::OpFoldResult>, SmallVector<mlir::OpFoldResult>>
OffsetSizeStrideOpDebugInfo::getOffsetsAndSizes() {
    if (auto sliceOp = mlir::dyn_cast_or_null<mlir::OffsetSizeAndStrideOpInterface>(_op)) {
        return std::make_tuple(sliceOp.getMixedOffsets(), sliceOp.getMixedSizes());
    }

    return std::make_tuple(SmallVector<mlir::OpFoldResult>{}, SmallVector<mlir::OpFoldResult>{});
}

void OffsetSizeStrideOpDebugInfo::evaluateDynamicDims(ValueRangeMap& valueMap) {
    if (valueMap.empty()) {
        _log.warning("Value map is empty during OffsetSizeStrideOpDebugInfo construction");
    }

    auto getAnalysisResults = [&](ArrayRef<mlir::OpFoldResult> data, SmallVector<SmallVector<int64_t>>& evaluatedData) {
        evaluatedData.reserve(data.size());
        for (auto offset : data) {
            auto cstValue = mlir::getConstantIntValue(offset);
            if (cstValue.has_value()) {
                evaluatedData.push_back(SmallVector<int64_t>{cstValue.value()});
                continue;
            }

            auto offsetVal = mlir::cast<mlir::Value>(offset);
            auto result = _opAnalyzer.getOpFoldResultValue(offsetVal, valueMap, OpChainAnalysis::MODE::ALL_VALUES);
            VPUX_THROW_WHEN(!result.has_value(), "Failed to evaluate dynamic offset value");
            evaluatedData.push_back(result.value());
        }
    };

    auto [offsets, sizes] = getOffsetsAndSizes();
    getAnalysisResults(offsets, _evaluatedOffsets);
    getAnalysisResults(sizes, _evaluatedSizes);
}

void OffsetSizeStrideOpDebugInfo::print(std::ostream& stream, size_t indentLevel, bool skipOffsets) {
    printDimensionData(stream, indentLevel, "Block sizes", _evaluatedSizes);
    if (!skipOffsets) {
        printDimensionData(stream, indentLevel, "Offsets", _evaluatedOffsets);
    }
}

bool OffsetSizeStrideOpDebugInfo::equals(const OpDebugInfo& other) const {
    auto* otherSlice = dynamic_cast<const OffsetSizeStrideOpDebugInfo*>(&other);
    if (otherSlice == nullptr) {
        return false;
    }

    return _evaluatedSizes == otherSlice->_evaluatedSizes;
}

int64_t OffsetSizeStrideOpDebugInfo::computeHash() const {
    int64_t hash = 0;

    // Hash sizes
    for (const auto& sizeVec : _evaluatedSizes) {
        for (auto val : sizeVec) {
            hash = llvm::hash_combine(hash, val);
        }
    }

    return hash;
}

llvm::SmallVector<mlir::NamedAttribute> getDynamicDimAttributeDict(ArrayRef<mlir::OpFoldResult> values,
                                                                   ArrayRef<SmallVector<int64_t>> evaluatedValues,
                                                                   mlir::MLIRContext* ctx, const std::string& prefix,
                                                                   const vpux::Logger& log) {
    llvm::SmallVector<mlir::NamedAttribute> dimAttrs;
    for (auto [dimIdx, value] : llvm::enumerate(values)) {
        if (mlir::getConstantIntValue(value).has_value()) {
            continue;
        }

        if (dimIdx >= evaluatedValues.size()) {
            log.warning("Dimension index {0} exceeds evaluated values size {1}", dimIdx, evaluatedValues.size());
            continue;
        }

        // Collect unique values in order of first appearance
        llvm::SmallSetVector<int64_t, DEFAULT_ARG_SET_SIZE> uniqueValues(evaluatedValues[dimIdx].begin(),
                                                                         evaluatedValues[dimIdx].end());

        log.trace("Dimension {0} unique evaluated sizes: {1} Total values {2}", dimIdx,
                  ArrayRef<int64_t>(uniqueValues.begin(), uniqueValues.end()), evaluatedValues[dimIdx].size());

        if (uniqueValues.empty()) {
            continue;
        }

        // Use NCHW labels for dimensions 0-3, otherwise use dim_X
        auto nchwStr = DimsOrder::NCHW.getCanonicalName();
        auto baseDimStr =
                (dimIdx < nchwStr.size()) ? std::string(1, nchwStr[dimIdx]) : ("dim_" + std::to_string(dimIdx));
        auto attrName = prefix.empty() ? std::move(baseDimStr) : (prefix + "_" + baseDimStr);
        auto nameAttr = mlir::StringAttr::get(ctx, attrName);
        llvm::SmallVector<int64_t, 4> valuesVec(uniqueValues.begin(), uniqueValues.end());
        auto valuesAttr = mlir::DenseI64ArrayAttr::get(ctx, valuesVec);
        dimAttrs.push_back(mlir::NamedAttribute(nameAttr, valuesAttr));
    }
    return dimAttrs;
}

void OffsetSizeStrideOpDebugInfo::setAttribute() {
    auto [offsets, sizes] = getOffsetsAndSizes();
    if (sizes.empty()) {
        return;
    }

    _log.trace("Setting dynamic dimension evaluated sizes attribute for operation {0}", _op->getName());

    // Build dictionary mapping dynamic dimension index to unique size values
    mlir::MLIRContext* ctx = _op->getContext();
    auto dimAttrs = getDynamicDimAttributeDict(sizes, _evaluatedSizes, ctx, "", _log);
    auto dictAttr = mlir::DictionaryAttr::get(ctx, dimAttrs);

    _log.trace("Dynamic dimension evaluated sizes attribute: {0}", dictAttr);
    _op->setAttr(EVALUATED_ATTR_NAME, dictAttr);
}

void PadOpDebugInfo::evaluate(ValueRangeMap& valueMap) {
    if (valueMap.empty()) {
        _log.warning("Iteration space is empty during PadOpDebugInfo construction");
        return;
    }

    auto padOp = mlir::dyn_cast<mlir::tensor::PadOp>(_op);
    if (padOp == nullptr) {
        _log.warning("Operation is not a tensor::PadOp");
        return;
    }

    // Helper function to evaluate pad values (low or high)
    auto evaluatePadValues = [&](mlir::OpFoldResult padOpFoldResult, const char* errorContext) -> SmallVector<int64_t> {
        if (!mlir::getConstantIntValue(padOpFoldResult).has_value()) {
            auto padVal = mlir::cast<mlir::Value>(padOpFoldResult);
            auto result = _opAnalyzer.getOpFoldResultValue(padVal, valueMap, OpChainAnalysis::MODE::ALL_VALUES);
            VPUX_THROW_UNLESS(result.has_value(), "Failed to evaluate dynamic {0} pad value", errorContext);
            return result.value();
        }
        return SmallVector<int64_t>{mlir::getConstantIntValue(padOpFoldResult).value()};
    };

    // Evaluate low pad values
    llvm::transform(padOp.getMixedLowPad(), std::back_inserter(_evaluatedLowPads),
                    [&](mlir::OpFoldResult padOpFoldResult) -> SmallVector<int64_t> {
                        return evaluatePadValues(padOpFoldResult, "low");
                    });

    // Evaluate high pad values
    llvm::transform(padOp.getMixedHighPad(), std::back_inserter(_evaluatedHighPads),
                    [&](mlir::OpFoldResult padOpFoldResult) -> SmallVector<int64_t> {
                        return evaluatePadValues(padOpFoldResult, "high");
                    });
}

void PadOpDebugInfo::print(std::ostream& stream, size_t indentLevel, bool) {
    printDimensionData(stream, indentLevel, "Low pads", _evaluatedLowPads);
    printDimensionData(stream, indentLevel, "High pads", _evaluatedHighPads);
}

bool PadOpDebugInfo::equals(const OpDebugInfo& other) const {
    auto* otherPad = dynamic_cast<const PadOpDebugInfo*>(&other);
    if (otherPad == nullptr) {
        return false;
    }

    return _evaluatedHighPads == otherPad->_evaluatedHighPads && _evaluatedLowPads == otherPad->_evaluatedLowPads;
}

int64_t PadOpDebugInfo::computeHash() const {
    int64_t hash = 0;

    // Hash low pads
    for (const auto& padVec : _evaluatedLowPads) {
        for (auto val : padVec) {
            hash = llvm::hash_combine(hash, val);
        }
    }

    // Hash high pads
    for (const auto& padVec : _evaluatedHighPads) {
        for (auto val : padVec) {
            hash = llvm::hash_combine(hash, val);
        }
    }

    return hash;
}

void PadOpDebugInfo::setAttribute() {
    mlir::MLIRContext* ctx = _op->getContext();
    llvm::SmallVector<mlir::NamedAttribute> dimAttrs;

    auto padOp = mlir::dyn_cast_or_null<mlir::tensor::PadOp>(_op);
    if (padOp == nullptr) {
        _log.warning("Operation is not a tensor::PadOp");
        return;
    }

    _log.trace("Setting dynamic dimension evaluated pad sizes attribute for operation {0}", _op->getName());

    auto lowPadAttrs = getDynamicDimAttributeDict(padOp.getMixedLowPad(), _evaluatedLowPads, ctx, "low_pad", _log);
    auto highPadAttrs = getDynamicDimAttributeDict(padOp.getMixedHighPad(), _evaluatedHighPads, ctx, "high_pad", _log);

    if (!lowPadAttrs.empty()) {
        _log.trace("Low pad attributes: ");
        dimAttrs.insert(dimAttrs.end(), lowPadAttrs.begin(), lowPadAttrs.end());
    }

    if (!highPadAttrs.empty()) {
        _log.trace("High pad attributes: ");
        dimAttrs.insert(dimAttrs.end(), highPadAttrs.begin(), highPadAttrs.end());
    }

    if (!dimAttrs.empty()) {
        _op->setAttr(EVALUATED_ATTR_NAME, mlir::DictionaryAttr::get(ctx, dimAttrs));
    }
}

void AffineDebugInfo::evaluate(ValueRangeMap& valueMap) {
    if (valueMap.empty()) {
        _log.warning("Iteration space is empty during AffineDebugInfo construction");
        return;
    }

    auto resultValues =
            _opAnalyzer.getOpFoldResultValue(_op->getResult(0), valueMap, OpChainAnalysis::MODE::ALL_VALUES);
    if (!resultValues.has_value()) {
        _log.warning("Failed to evaluate affine operation result values");
        return;
    }
    _evaluatedValues = resultValues.value();

    for (auto val : _evaluatedValues) {
        _log.trace("Evaluated affine op result value: {0}", val);
    }
}

void AffineDebugInfo::print(std::ostream& stream, size_t indentLevel, bool) {
    constexpr int VALUE_WIDTH = 3;       // Fixed width for all values
    constexpr int DIM_LABEL_WIDTH = 10;  // Fixed width for dimension labels

    // Save stream format state to restore later
    std::ios_base::fmtflags originalFlags = stream.flags();

    // Create indentation string
    std::string headerIndent(indentLevel, ' ');
    std::string msgIndent(indentLevel + 1, ' ');

    // Print header
    stream << headerIndent << "  " << std::left << std::setw(DIM_LABEL_WIDTH) << "Results" << "\n";
    stream << msgIndent << std::right << std::setw(DIM_LABEL_WIDTH + 2) << "Values: [";

    // Print all result values
    bool first = true;
    for (auto value : _evaluatedValues) {
        if (!first) {
            stream << ", ";
        }
        stream << std::setw(VALUE_WIDTH) << value;
        first = false;
    }
    stream << "]\n";

    // Restore original stream format state
    stream.flags(originalFlags);
}

bool AffineDebugInfo::equals(const OpDebugInfo& other) const {
    auto* otherAffineOp = dynamic_cast<const AffineDebugInfo*>(&other);
    if (otherAffineOp == nullptr) {
        return false;
    }

    return _evaluatedValues == otherAffineOp->_evaluatedValues;
}

int64_t AffineDebugInfo::computeHash() const {
    int64_t hash = 0;
    for (const auto& val : _evaluatedValues) {
        hash = llvm::hash_combine(hash, val);
    }

    return hash;
}

void AffineDebugInfo::setAttribute() {
    mlir::MLIRContext* ctx = _op->getContext();
    llvm::SmallVector<mlir::NamedAttribute> dimAttrs;

    auto valuesAttr = mlir::DenseI64ArrayAttr::get(ctx, _evaluatedValues);
    auto attr = mlir::NamedAttribute(mlir::StringAttr::get(ctx, "results"), valuesAttr);
    _op->setAttr(EVALUATED_ATTR_NAME, mlir::DictionaryAttr::get(ctx, {attr}));
}

SmallVector<mlir::scf::ForOp> ScfAnalysisInfo::getParentForOps(mlir::Operation* op) const {
    SmallVector<mlir::scf::ForOp> parentForOps;

    // Start from the operation and traverse up the parent hierarchy
    mlir::Operation* currentOp = op;
    while (currentOp) {
        // Get the parent operation
        currentOp = currentOp->getParentOp();

        // If the parent is a scf.for operation, add it to the list
        if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(currentOp)) {
            parentForOps.push_back(forOp);
        }
    }

    // Reverse to get outermost to innermost order
    std::reverse(parentForOps.begin(), parentForOps.end());

    return parentForOps;
}

SmallVector<InputRange> ScfAnalysisInfo::getIterationSpace(ArrayRef<mlir::scf::ForOp> forOpChain) const {
    OpChainAnalysis analysis;
    SmallVector<InputRange> inputRange;
    for (auto forOp : forOpChain) {
        auto blockArg = forOp.getInductionVar();
        auto [low, high, step] = analysis.getLoopBoundsAndStep(forOp);

        if (forOp->hasAttr(UPPERBOUND_ATTR)) {
            high = forOp->getAttrOfType<mlir::IntegerAttr>(UPPERBOUND_ATTR).getInt();
        }

        SmallVector<int64_t> values;
        for (auto i = low; i < high; i += step) {
            values.push_back(i);
        }

        inputRange.push_back({blockArg, std::move(values), step});
    }
    return inputRange;
}

//
// ScfBlockAnalyzer implementation
//

bool ScfBlockAnalyzer::StableState::equals(const StableState& other) const {
    if (hashValue != other.hashValue) {
        return false;
    }
    if (opStates.size() != other.opStates.size()) {
        return false;
    }

    return llvm::all_of(opStates, [&](const auto& it) {
        auto found = other.opStates.find(it.first);
        return found != other.opStates.end() && it.second->equals(*found->second);
    });
}

std::string ScfBlockAnalyzer::StateRegion::toString() const {
    std::stringstream ss;
    ss << "State " << stateId << ": ";

    // Display as point list with count
    size_t count = points.size();
    ss << "{" << count << " pt";
    if (count != 1) {
        ss << "s";
    }
    ss << "} ";

    // Show actual points
    if (count <= 6) {
        // Show all points if 6 or fewer
        ss << "[";
        for (size_t i = 0; i < points.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << "(";
            for (size_t d = 0; d < points[i].size(); ++d) {
                if (d > 0) {
                    ss << ",";
                }
                ss << points[i][d];
            }
            ss << ")";
        }
        ss << "]";
    } else {
        // Show first 3 and last 3 with ellipsis
        ss << "[";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << "(";
            for (size_t d = 0; d < points[i].size(); ++d) {
                if (d > 0) {
                    ss << ",";
                }
                ss << points[i][d];
            }
            ss << ")";
        }
        ss << ", ..., ";
        for (size_t i = points.size() - 3; i < points.size(); ++i) {
            if (i > points.size() - 3) {
                ss << ", ";
            }
            ss << "(";
            for (size_t d = 0; d < points[i].size(); ++d) {
                if (d > 0) {
                    ss << ",";
                }
                ss << points[i][d];
            }
            ss << ")";
        }
        ss << "]";
    }

    return ss.str();
}

std::string ScfBlockAnalyzer::StateRegion::toConstraintString() const {
    std::stringstream ss;
    ss << "State " << stateId << " constraints: {";

    if (bounds.empty()) {
        ss << "empty}";
        return ss.str();
    }

    size_t numDims = bounds.size();
    bool firstConstraint = true;

    // Bounding box constraints
    for (size_t d = 0; d < numDims; ++d) {
        if (!firstConstraint) {
            ss << ", ";
        }
        ss << bounds[d].first << " ≤ d" << d << " ≤ " << bounds[d].second;
        firstConstraint = false;
    }

    // Stride constraints (if applicable)
    if (!strides.empty()) {
        for (size_t d = 0; d < strides.size() && d < numDims; ++d) {
            if (strides[d] > 0 && bounds[d].first != bounds[d].second) {
                ss << ", d" << d << " ≡ " << bounds[d].first << " (mod " << strides[d] << ")";
            }
        }
    }

    ss << "}";
    return ss.str();
}

void ScfBlockAnalyzer::detectStableStateRegions(AnalysisContext& analysisContext) {
    _log.trace("Starting stable state detection");

    // Clear previous results
    _uniqueStates.clear();
    _hashToStateId.clear();
    _allRegions.clear();
    _stateToRegions.clear();

    auto iterationPoints = analysisContext.getIterationPoints();
    if (iterationPoints.empty()) {
        _log.warning("No iteration points to analyze");
        return;
    }

    _log.trace("Analyzing {0} iteration points in {1} dimensions", iterationPoints.size(), iterationPoints[0].size());

    // Phase 1: Sample iteration space and map each point to a stable state
    SmallVector<std::pair<SmallVector<int64_t>, size_t>> pointToState;
    sampleIterationSpace(analysisContext, pointToState);

    _log.trace("Detected {0} unique stable states", _uniqueStates.size());

    // Phase 2: Coalesce adjacent points with same state into regions
    // Compute dimension sizes for coalescing
    SmallVector<int64_t> dimSizes;
    if (!iterationPoints.empty()) {
        size_t numDims = iterationPoints[0].size();
        dimSizes.resize(numDims);
        for (size_t dim = 0; dim < numDims; ++dim) {
            int64_t maxVal = std::numeric_limits<int64_t>::min();
            for (const auto& point : iterationPoints) {
                maxVal = std::max(maxVal, point[dim]);
            }
            dimSizes[dim] = maxVal + 1;
        }
    }

    coalesceRegions(pointToState, dimSizes, analysisContext);

    _log.trace("Coalesced into {0} regions", _allRegions.size());
}

void ScfBlockAnalyzer::sampleIterationSpace(AnalysisContext& analysisContext,
                                            SmallVector<std::pair<SmallVector<int64_t>, size_t>>& pointToState) {
    auto iterationPoints = analysisContext.getIterationPoints();
    pointToState.reserve(iterationPoints.size());

    llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>> currentStates;
    for (size_t pointIdx = 0; pointIdx < iterationPoints.size(); ++pointIdx) {
        const auto& point = iterationPoints[pointIdx];

        // Evaluate all operations at this point
        auto valueMap = analysisContext.getValueMapAtIndex(pointIdx);
        evaluateAtPoint(valueMap, currentStates);

        // Register or find the stable state
        size_t stateId = registerState(currentStates);

        // Store mapping
        pointToState.push_back({SmallVector<int64_t>(point.begin(), point.end()), stateId});
        currentStates.clear();
    }
}

size_t ScfBlockAnalyzer::registerState(
        const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& opStates) {
    int64_t hash = computeStateHash(opStates);

    // Check if state already exists
    auto it = _hashToStateId.find(hash);
    if (it != _hashToStateId.end()) {
        // Hash collision check - verify actual equality
        const auto& existingState = _uniqueStates[it->second];
        if (compareStates(existingState.opStates, opStates)) {
            return it->second;  // Found existing state
        }
        // Hash collision - need to handle (for now, create new state)
        _log.warning("Hash collision detected for state hash {0}", hash);
    }

    // Create new state
    size_t newStateId = _uniqueStates.size();
    StableState newState;
    newState.stateId = newStateId;
    newState.hashValue = hash;

    // Deep copy the operation states
    for (const auto& [op, state] : opStates) {
        newState.opStates[op] = state;
    }

    _uniqueStates.push_back(std::move(newState));
    _hashToStateId[hash] = newStateId;

    _log.trace("Registered new stable state {0} with hash {1}", newStateId, hash);

    return newStateId;
}

llvm::DenseMap<size_t, SmallVector<size_t>> ScfBlockAnalyzer::buildConnectivityGraph(
        ArrayRef<std::pair<SmallVector<int64_t>, size_t>> pointToState, AnalysisContext& analysisContext,
        SmallVector<size_t>& parent) {
    size_t numPoints = pointToState.size();

    auto find = [&](size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](size_t x, size_t y) {
        size_t rootX = find(x);
        size_t rootY = find(y);
        if (rootX != rootY) {
            parent[rootX] = rootY;
        }
    };

    // Only check connectivity within groups that share the same state
    llvm::DenseMap<size_t, SmallVector<size_t>> stateGroups;
    for (size_t i = 0; i < numPoints; ++i) {
        stateGroups[pointToState[i].second].push_back(i);
    }

    // For each state group, check connectivity only within that group
    for (const auto& [stateId, indices] : stateGroups) {
        for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = i + 1; j < indices.size(); ++j) {
                size_t idx1 = indices[i];
                size_t idx2 = indices[j];
                const auto& point1 = pointToState[idx1].first;
                const auto& point2 = pointToState[idx2].first;

                if (analysisContext.arePointsContiguous(point1, point2)) {
                    unite(idx1, idx2);
                }
            }
        }
    }

    llvm::DenseMap<size_t, SmallVector<size_t>> components;
    for (size_t i = 0; i < numPoints; ++i) {
        components[find(i)].push_back(i);
    }
    return components;
}

void ScfBlockAnalyzer::createRegionsFromComponents(const llvm::DenseMap<size_t, SmallVector<size_t>>& components,
                                                   ArrayRef<std::pair<SmallVector<int64_t>, size_t>> pointToState,
                                                   size_t numDims, AnalysisContext& analysisContext) {
    for (const auto& [root, indices] : components) {
        if (indices.empty()) {
            continue;
        }

        StateRegion region;
        region.stateId = pointToState[indices[0]].second;
        region.bounds.resize(numDims);

        region.points.reserve(indices.size());
        for (size_t idx : indices) {
            region.points.push_back(pointToState[idx].first);
        }

        const auto& firstPoint = pointToState[indices[0]].first;
        for (size_t d = 0; d < numDims; ++d) {
            region.bounds[d] = {firstPoint[d], firstPoint[d]};
        }

        for (size_t idx : indices) {
            const auto& point = pointToState[idx].first;
            for (size_t d = 0; d < numDims; ++d) {
                region.bounds[d].first = std::min(region.bounds[d].first, point[d]);
                region.bounds[d].second = std::max(region.bounds[d].second, point[d]);
            }
        }

        auto contextStrides = analysisContext.getStrides();
        region.strides.resize(numDims, 0);
        for (size_t d = 0; d < numDims; ++d) {
            if (d < contextStrides.size()) {
                region.strides[d] = contextStrides[d];
            } else {
                region.strides[d] = 1;
            }
        }

        size_t regionIdx = _allRegions.size();
        _allRegions.push_back(region);
        _stateToRegions[region.stateId].push_back(regionIdx);
    }
}

void ScfBlockAnalyzer::coalesceRegions(ArrayRef<std::pair<SmallVector<int64_t>, size_t>> pointToState,
                                       ArrayRef<int64_t> dimSizes, AnalysisContext& analysisContext) {
    if (pointToState.empty() || dimSizes.empty()) {
        return;
    }

    size_t numDims = dimSizes.size();
    size_t numPoints = pointToState.size();

    _log.trace("Coalescing {0} points using Connected Component Labeling", numPoints);

    SmallVector<size_t> parent(numPoints);
    std::iota(parent.begin(), parent.end(), 0);

    auto connectedGraph = buildConnectivityGraph(pointToState, analysisContext, parent);

    _log.trace("Found {0} connected components", connectedGraph.size());

    createRegionsFromComponents(connectedGraph, pointToState, numDims, analysisContext);
}

int64_t ScfBlockAnalyzer::computeStateHash(
        const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& opStates) const {
    // Hash only the operation parameters, NOT the operation pointers
    // This ensures that operations with identical parameters hash to the same value
    int64_t hash = 17;  // Prime number seed

    // MapVector preserves insertion order, so iteration is deterministic
    // No need to sort - operations are already in their original order

    for (const auto& [op, opInfo] : opStates) {
        // Query operation-specific hash from the debug info object
        int64_t opHash = opInfo->computeHash();
        hash = llvm::hash_combine(hash, opHash);
    }

    return hash;
}

bool ScfBlockAnalyzer::compareStates(
        const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& state1,
        const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& state2) const {
    if (state1.size() != state2.size()) {
        return false;
    }

    for (const auto& [op, info1] : state1) {
        auto it = state2.find(op);
        if (it == state2.end()) {
            return false;
        }
        if (!info1->equals(*it->second)) {
            return false;
        }
    }

    return true;
}

void ScfBlockAnalyzer::evaluateAtPoint(ValueRangeMap& valueMap,
                                       llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& states) {
    for (auto op : _operations) {
        auto it = _opInfoMap.find(op);
        if (it != _opInfoMap.end()) {
            // Create a copy of the OpDebugInfo for this evaluation
            mlir::TypeSwitch<mlir::Operation*>(op)
                    .Case<mlir::OffsetSizeAndStrideOpInterface>([&](auto) {
                        auto opInfo = std::make_shared<OffsetSizeStrideOpDebugInfo>(op);
                        opInfo->evaluate(valueMap);
                        states[op] = opInfo;
                    })
                    .Case<mlir::tensor::PadOp>([&](mlir::tensor::PadOp) {
                        auto opInfo = std::make_shared<PadOpDebugInfo>(op);
                        opInfo->evaluate(valueMap);
                        states[op] = opInfo;
                    });
        }
    }
}

void ScfBlockAnalyzer::printResults(std::ostream& stream, size_t indentLevel) const {
    std::string indent(indentLevel, ' ');
    stream << indent << "\n" << indent << "=== Stable State Detection Results ===\n";
    stream << indent << "Unique stable states: " << _uniqueStates.size() << "\n";
    stream << indent << "Total regions: " << _allRegions.size() << "\n\n";

    for (size_t stateId = 0; stateId < _uniqueStates.size(); ++stateId) {
        printState(stateId, stream, indentLevel);
    }
}

void ScfBlockAnalyzer::printState(size_t stateId, std::ostream& stream, size_t indentLevel) const {
    if (stateId >= _uniqueStates.size()) {
        return;
    }

    std::string indent(indentLevel, ' ');
    const auto& state = _uniqueStates[stateId];
    stream << indent << "--- Stable State " << stateId << " (hash: " << state.hashValue << ") ---\n";

    // Print operation states
    for (const auto& [op, opInfo] : state.opStates) {
        stream << indent << "  Operation: " << op->getName().getStringRef().str() << "\n";
        opInfo->print(stream, indentLevel + 4);
    }

    // Print regions for this state
    auto it = _stateToRegions.find(stateId);
    if (it != _stateToRegions.end()) {
        stream << indent << "  Regions (" << it->second.size() << "):\n";

        for (size_t regionIdx : it->second) {
            stream << indent << "    ";
            printRegion(_allRegions[regionIdx], stream, indentLevel);
            stream << "\n";
        }
    }
    stream << "\n";
}

void ScfBlockAnalyzer::printRegion(const StateRegion& region, std::ostream& stream, size_t indentLevel) const {
    std::string indent(indentLevel * 2, ' ');
    stream << region.toString();
    // Also show constraint representation on next line with indentation
    stream << "\n" << indent << "└─ " << region.toConstraintString();
}

}  // namespace vpux::VPU
