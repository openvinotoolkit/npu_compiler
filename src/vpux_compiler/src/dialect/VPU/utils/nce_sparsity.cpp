//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <limits>
#include <numeric>

#include <llvm/ADT/bit.h>

using namespace vpux;

namespace {

using namespace VPU::NCESparsity;

template <typename ScaleElemType>
llvm::unique_function<ScaleElemType(size_t)> getBiasFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                         mlir::Type weightsElemType, const Const::ContentAttr& bias,
                                                         VPU::NCESparsity::BiasConverterCb biasConverter, size_t OC) {
    if (bias == nullptr) {
        return [](int64_t) -> double {
            return 0.0f;
        };
    }

    auto biasContent = bias.fold();

    const auto isInQuantized = mlir::isa<mlir::quant::QuantizedType>(inElemType);
    const auto isOutQuantized = mlir::isa<mlir::quant::QuantizedType>(outElemType);
    const auto isWeightsQuantized = mlir::isa<mlir::quant::QuantizedType>(weightsElemType);
    const auto isQuant = isInQuantized && isOutQuantized;
    const auto isFloat = !isInQuantized && !isOutQuantized;
    const auto isMixed = !isQuant && !isFloat;
    const auto isQuantInFloatOut = isInQuantized && isMixed;
    const auto isFloatInQuantOut = isOutQuantized && isMixed;

    auto filterQuantScales =
            isWeightsQuantized ? extractScalesAndZeroPoints(weightsElemType).first : SmallVector<double>{1.0};

    if (isQuant || isQuantInFloatOut) {
        // PPE engages float by-pass in this case. Apply re-scaling.
        auto rescaledBias = VPU::NCESparsity::getRescaledBias(bias, inElemType, weightsElemType, OC);
        VPUX_THROW_WHEN(mlir::failed(rescaledBias), "Rescaled bias value is out of range");

        return [rescaledBiasValue = std::move(rescaledBias.value()), inElemType,
                biasConverter](size_t oc) -> ScaleElemType {
            return std::get<ScaleElemType>(biasConverter(rescaledBiasValue[oc], inElemType));
        };
    } else if (isFloat || isFloatInQuantOut) {
        return [biasContent = std::move(biasContent), inElemType, isWeightsQuantized,
                filterQuantScales = std::move(filterQuantScales), biasConverter](int64_t oc) -> ScaleElemType {
            auto getBiasValue = [&]() {
                if (biasContent.isSplat()) {
                    return biasContent.getSplatValue<float>();
                } else {
                    return biasContent.getValues<float>()[oc];
                }
            };
            auto biasVal = getBiasValue();
            if (isWeightsQuantized) {
                // check if filter is quantized per axis
                if (filterQuantScales.size() != 1) {
                    biasVal /= filterQuantScales[oc];
                } else {
                    biasVal /= filterQuantScales.front();
                }
            }
            return std::get<ScaleElemType>(biasConverter(biasVal, inElemType));
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. quant-quant, quant-float, float-quant or float-float type "
               "pairs required. Got: in type {0}, out type {1}",
               inElemType, outElemType);
}

}  // namespace

int32_t vpux::VPU::NCESparsity::toFixedPoint(const double realVal) {
    const double mult = 1 << 16;
    return std::lround(realVal * mult);
}

int32_t vpux::VPU::NCESparsity::toHex(double realVal) {
    return llvm::bit_cast<int32_t>(static_cast<float>(realVal));
}

int32_t vpux::VPU::NCESparsity::getWeightPtrStep(mlir::Value weights) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
    const auto convAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
    const auto weightsElementCount = IC * KY * KX;
    VPUX_THROW_UNLESS(weightsElementCount % convAlignment == 0,
                      "Convolution and Depthwise convolution weights size must be a multiple of {0}, got {1}",
                      convAlignment, weightsElementCount);

    const Bit eltSize = getElemTypeSize(weights.getType());
    return checked_cast<int32_t>(Byte(eltSize * IC * KY * KX).count());
}

std::vector<int32_t> vpux::VPU::NCESparsity::getExpandedWeightsTable(ArrayRef<int32_t> weightsTableVector, int64_t OC) {
    using vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    auto expandedWTVec = weightsTableVector.vec();

    for (auto oc = OC; oc < VPU_CHANNEL_ALIGNMENT; oc++) {
        expandedWTVec.push_back(expandedWTVec[0]);
        expandedWTVec.push_back(expandedWTVec[1]);
        expandedWTVec.push_back(0);
        expandedWTVec.push_back(0);
    }
    return expandedWTVec;
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(
        mlir::Type inElemType, mlir::Type outElemType, std::optional<int32_t> weightsPtr, int32_t weightsPtrStep,
        std::optional<int32_t> sparsityPtr, int32_t sparsityPtrStep, VPU::NCESparsity::PPEConverterCb ppeConverter,
        VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC, mlir::Type weightsElemType,
        const Const::ContentAttr& bias, mlir::FloatAttr constScale) {
    auto weightsPtrOffset = weightsPtr.has_value() ? weightsPtr.value() : 0;

    // In case of dense operation use sparsityPtrOffset beyond CMX memory range to satisfy HW requirements
    auto sparsityPtrOffset = sparsityPtr.has_value() ? sparsityPtr.value() : SPARSITY_PTR_WHEN_NO_SPARSITY;

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffset;
        weightsPtrOffset += weightsPtrStep;

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += sparsityPtrStep;
    }

    return getWeightsTable(inElemType, outElemType, weightsPtrs, sparsityPtrs, ppeConverter, biasConverter, OC,
                           weightsElemType, bias, constScale);
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(
        mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> weightsPtrs, ArrayRef<int32_t> sparsityPtrs,
        VPU::NCESparsity::PPEConverterCb ppeConverter, VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
        mlir::Type weightsElemType, const Const::ContentAttr& bias, mlir::FloatAttr constScale) {
    using vpux::VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Weights pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(static_cast<int64_t>(sparsityPtrs.size()) != OC,
                    "Sparsity pointers size {0} different than output channels {1}", sparsityPtrs.size(), OC);

    auto getMultShift = getMultShiftFunc<int32_t>(inElemType, outElemType, weightsElemType, ppeConverter,
                                                  checked_cast<size_t>(OC), constScale);
    auto getBiasFP = getBiasFunc<int32_t>(inElemType, outElemType, weightsElemType, bias, biasConverter,
                                          checked_cast<size_t>(OC));

    std::vector<std::int32_t> weightsTableVals(OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        const auto wtInd = oc * static_cast<size_t>(WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);
        VPUX_THROW_UNLESS(sparsityPtrs[oc] == SPARSITY_PTR_WHEN_NO_SPARSITY ||
                                  sparsityPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "sparsityPtrs[{0}] must be aligned to {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          sparsityPtrs[oc]);

        weightsTableVals[wtInd + 0] = weightsPtrs[oc];
        weightsTableVals[wtInd + 1] = sparsityPtrs[oc];
        weightsTableVals[wtInd + 2] = getMultShift(oc);
        weightsTableVals[wtInd + 3] = getBiasFP(oc);
    });

    return weightsTableVals;
}

std::vector<float> vpux::VPU::NCESparsity::getBiasTable(mlir::Type inElemType, mlir::Type outElemType,
                                                        VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                                        mlir::Type weightsElemType, const Const::ContentAttr& bias) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    auto getBiasFP =
            getBiasFunc<float>(inElemType, outElemType, weightsElemType, bias, biasConverter, checked_cast<size_t>(OC));

    std::vector<float> biasTableVals(OC, 0.0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        biasTableVals[oc] = getBiasFP(oc);
    });

    return biasTableVals;
}

std::vector<int32_t> vpux::VPU::NCESparsity::patchWeightsTableSparsityPtrs(
        const std::vector<std::int32_t>& weightsTableVals, const int32_t sparsityPtrOffset,
        const int32_t sparsityPtrStep, std::optional<int64_t> origOC) {
    int64_t OC = weightsTableVals.size() / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;
    auto validOC = OC;

    std::vector<std::int32_t> newWeightsTableVals(weightsTableVals.begin(), weightsTableVals.end());

    VPUX_THROW_UNLESS(sparsityPtrOffset % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                      "sparsityPtrOffset must be aligned to {0}, got {1}", ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                      sparsityPtrOffset);

    VPUX_THROW_UNLESS(sparsityPtrStep % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                      "sparsityPtrStep must be aligned to {0}, got {1}", ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                      sparsityPtrStep);

    int32_t offset = sparsityPtrOffset;
    if (origOC.has_value()) {
        validOC = origOC.value();
    }

    for (auto oc : irange(checked_cast<size_t>(validOC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        newWeightsTableVals[wtInd + 1] = offset;

        offset += sparsityPtrStep;
    }

    for (auto oc : irange(checked_cast<size_t>(validOC), checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);
        newWeightsTableVals[wtInd + 1] = weightsTableVals[1];
    }

    return newWeightsTableVals;
}

Shape vpux::VPU::NCESparsity::inferWeightsTableShape(int64_t OC, bool newFormat) {
    if (newFormat) {
        return Shape{OC, 1, 1, VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};
    }
    return Shape{OC, 1, 1, VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};
}

Shape vpux::VPU::NCESparsity::inferWeightsSparsityMapShape(ShapeRef dataShape) {
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected data shape to be 4D, while shape is {0}", dataShape);
    const auto workloadSize = std::accumulate(dataShape.begin() + 1, dataShape.end(), static_cast<int64_t>(1),
                                              std::multiplies<int64_t>());
    const auto alignment = Byte(16).to<Bit>().count();
    const auto alignedWorkloadSize = vpux::alignValUp(workloadSize, alignment);
    return Shape({dataShape.raw()[0], 1, 1, alignedWorkloadSize});
}

mlir::FailureOr<SmallVector<double>> vpux::VPU::NCESparsity::getRescaledBias(const Const::ContentAttr& biasAttr,
                                                                             mlir::Type inElemType,
                                                                             mlir::Type filterElemType, int64_t OC) {
    auto inQuantScale = mlir::isa<mlir::quant::QuantizedType>(inElemType) ? extractScalesAndZeroPoints(inElemType).first
                                                                          : SmallVector<double>{1.0};
    auto filterQuantScales = mlir::isa<mlir::quant::QuantizedType>(filterElemType)
                                     ? extractScalesAndZeroPoints(filterElemType).first
                                     : SmallVector<double>{1.0};
    broadcast(inQuantScale, OC);
    broadcast(filterQuantScales, OC);

    SmallVector<double> rescaledBias(OC, 1.0);
    std::transform(filterQuantScales.begin(), filterQuantScales.end(), inQuantScale.begin(), rescaledBias.begin(),
                   std::multiplies<>());

    auto biasContent = biasAttr.fold();
    auto biasValueRange = biasContent.getValues<double>();
    VPUX_THROW_UNLESS(biasValueRange.size() >= static_cast<size_t>(OC), "bias size {} is less than OC {}",
                      biasValueRange.size(), OC);

    std::transform(biasValueRange.begin(), biasValueRange.begin() + OC, rescaledBias.begin(), rescaledBias.begin(),
                   std::divides<>());

    const auto isValueOutOfRange = llvm::any_of(rescaledBias, [](double newBiasData) {
        return newBiasData <= std::numeric_limits<int32_t>::min() || newBiasData >= std::numeric_limits<int32_t>::max();
    });
    if (isValueOutOfRange) {
        return mlir::failure();
    }
    return rescaledBias;
}

double vpux::VPU::NCESparsity::getSparsityRatio(vpux::NDTypeInterface weightsType, int64_t compressedSize) {
    const auto elemByteSize = getElemTypeSize(weightsType.getElementType()).to<Byte>().count();
    const auto uncompressedSize = weightsType.getShape().totalSize() * elemByteSize;
    VPUX_THROW_WHEN(uncompressedSize == 0, "Uncompressed size should be non-zero");
    const auto sparsityRatio = 1.0 - checked_cast<double>(compressedSize) / checked_cast<double>(uncompressedSize);
    VPUX_THROW_WHEN(sparsityRatio < 0.0 || sparsityRatio > 1.0,
                    "Sparsity ratio should be in range [0.0, 1.0], while it is {0}", sparsityRatio);
    return sparsityRatio;
}

/*
 Compute sparsification ratio of weights. It computes effective compression ratio of weights in case of weights
 sparsification. Ratio depends on number of non-zero elements and HW requirements to alignment. Acceleration depends
 mostly on memory footprint saving therefore alignment must be taken into account while computing ratio. Weights are
 grouped into sets and have the format OCx(HxWxIC) where:
 - OC is output channels that is the number of weights sets
 - HxWxIC is weights set size, its size must be aligned according to HW requirements
 Ratio is computed as follows:
 - Count number of non-zero elements in each output channel, compute their size and align up to alignment value
 - Sum the size of all output channels/sets of weights
 - Effective ratio is: 1 - (size of non-zero vals)/(size of tensor)
*/
double vpux::VPU::NCESparsity::getSparsityRatio(vpux::NDTypeInterface weightsType,
                                                ArrayRef<int64_t> numNonSparseElemsPerOC) {
    const auto elemByteSize = getElemTypeSize(weightsType.getElementType()).to<Byte>().count();
    const auto alignedChanSizeDenseVals = [&](auto sum, auto elemsInChan) {
        return sum + vpux::alignValUp(elemsInChan * elemByteSize, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
    };
    const auto compressedSize = std::accumulate(numNonSparseElemsPerOC.begin(), numNonSparseElemsPerOC.end(), 0LL,
                                                alignedChanSizeDenseVals);
    return getSparsityRatio(weightsType, compressedSize);
}

bool vpux::VPU::NCESparsity::isSparsifiableWeightsOperand(mlir::Value operand) {
    const auto operandType = operand.getType();
    // already sparse
    if (mlir::isa<vpux::VPU::SparseTensorType>(operandType)) {
        return false;
    }
    auto sourceOp = operand.getDefiningOp<Const::DeclareOp>();
    if (!sourceOp) {
        return false;
    }
    for (const auto transformation : sourceOp.getContentAttr().getTransformations()) {
        if (mlir::isa<vpux::Const::SparsifyAttr, vpux::Const::GetSparsityMapAttr>(transformation)) {
            VPUX_THROW("Trying to sparsify already sparsity related content at '{0}'", sourceOp->getLoc());
        }
    }
    return true;
}

bool vpux::VPU::NCESparsity::isSuperdenseRequired(const DimsOrder outOrder, const ShapeRef outShape,
                                                  const mlir::Type outElemType) {
    // If the inner-most dimension of output shape is aligned, super-dense mode is not required.
    const auto outputMemShape = outOrder.toMemoryOrder(outShape);
    const auto outputInnerDim = outputMemShape.back();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto outputInnerDimRemainder = outputInnerDim % alignment;
    return outputInnerDimRemainder != 0;
}

vpux::VPU::NCESparsity::RuntimeSparsityStatsProvider::RuntimeSparsityStatsProvider(mlir::func::FuncOp func,
                                                                                   vpux::Logger log)
        : _logger(log), _lookup({}) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto statOps = to_small_vector(module.getOps<net::SparsityStatisticsOp>());
    VPUX_THROW_UNLESS(statOps.size() <= 1, "Module must contains 0 or 1 sparsity statistics, but got {0}",
                      statOps.size());
    if (statOps.empty()) {
        return;
    }

    auto stats = statOps.front();
    auto& infos = stats.getSparsityInfo().front().getOperations();
    for (auto& info : infos) {
        auto asOp = mlir::cast<net::SparsityInfoOp>(info);
        const auto key = asOp.getName().str();
        _lookup.emplace(key, asOp);
    }
}

bool vpux::VPU::NCESparsity::RuntimeSparsityStatsProvider::containsStatistics() const {
    return _lookup.size() > 0;
}

bool vpux::VPU::NCESparsity::RuntimeSparsityStatsProvider::likelySparsityConsumer(mlir::Operation* op,
                                                                                  int64_t requestedInputId) const {
    auto loc = mlir::dyn_cast<mlir::FusedLoc>(op->getLoc());
    if (loc == nullptr) {
        return false;
    }
    auto locParts = loc.getLocations();
    if (locParts.empty()) {
        return false;
    }
    auto keyNameLoc = mlir::dyn_cast<mlir::NameLoc>(locParts.front());
    if (keyNameLoc == nullptr) {
        return false;
    }
    const auto key = keyNameLoc.getName().strref().data();
    for (auto it = _lookup.find(key); it != _lookup.end() && it->first == key; ++it) {
        auto opStats = it->second;
        auto inputId = opStats.getInputId();
        if (inputId != requestedInputId) {
            continue;
        }
        const auto ratio = opStats.getRatioAttr().getValueAsDouble();
        _logger.trace("Found RT stats for input {0} of '{1}'.  Sparsity ratio is {2}", requestedInputId, op->getLoc(),
                      ratio);
        return ratio >= MINIMAL_SPARSITY_THRESHOLD;
    }
    return false;
}

//
// NewWeightsTableFormatMapper
//

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::normalizeKAndReturnCurrentGroupOf128ZeroPoints(
        int32_t index, int32_t& k) {
    // the pattern is repeating after 128 elements
    int32_t currentGroupOf128ZeroPoints = index / 128;
    int32_t countGroupsOf128ZeroPoints = k / 128;

    // check if the current zero-point is in the last group of 128 zero-points;
    // you may see the following corner case: if k is a multiple of 128, then
    // countGroupsOf128ZeroPoints == currentGroupOf128ZeroPoints will be always false
    // that's intended because otherwise the first branch will be executed and k will become 0 (it has to be 128)
    if (currentGroupOf128ZeroPoints == countGroupsOf128ZeroPoints) {
        k %= 128;
    } else {
        k = 128;
    }

    return currentGroupOf128ZeroPoints;
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::mathematicallyEncodePositionInNewZeroPointOnlyTableLayout(
        int32_t zeroPointIndex, int32_t k) {
    // the pattern is repeating after 128 elements
    int32_t currentGroupOf128ZeroPoints = normalizeKAndReturnCurrentGroupOf128ZeroPoints(zeroPointIndex, k);
    zeroPointIndex %= 128;

    int32_t elementsInASequence = k / 8;
    int32_t sequenceNumber = ((zeroPointIndex % 16) / 4) * 2 + zeroPointIndex % 2;
    int32_t positionInSequence = (zeroPointIndex / 16) * 2 + (zeroPointIndex % 4) / 2;

    int32_t elementPositionInTable = (elementsInASequence * sequenceNumber) + positionInSequence;
    return elementPositionInTable + currentGroupOf128ZeroPoints * 128;
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::mathematicallyDecodePositionInNewZeroPointOnlyTableLayout(
        int32_t position, int32_t k) {
    // the pattern is repeating after 128 elements
    int32_t currentGroupOf128ZeroPoints = normalizeKAndReturnCurrentGroupOf128ZeroPoints(position, k);
    position %= 128;

    int32_t elementsInASequence = k / 8;
    int32_t sequenceNumber = position / elementsInASequence;
    int32_t positionInSequence = position % elementsInASequence;

    int32_t elementPositionInTable =
            sequenceNumber / 2 * 4 + positionInSequence / 2 * 16 + (positionInSequence % 2) * 2 + sequenceNumber % 2;
    return elementPositionInTable + currentGroupOf128ZeroPoints * 128;
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::computeInversePermutation(
        std::vector<int32_t> v) {
    VPUX_THROW_WHEN(v.size() == 0, "Input vector is empty");

    std::unordered_set checkForDuplicates(v.begin(), v.end());
    int32_t minValue = v.size();
    int32_t maxValue = 0;
    for (auto elem : v) {
        if (elem > maxValue) {
            maxValue = elem;
        }
        if (elem < minValue) {
            minValue = elem;
        }
    }

    VPUX_THROW_WHEN(
            checkForDuplicates.size() != v.size() || minValue != 0 || maxValue != static_cast<int32_t>(v.size() - 1),
            "Input vector should contain the values from 0 to v.size() - 1");

    std::vector<int32_t> result(v.size());
    for (auto index = 0; index < static_cast<int32_t>(v.size()); index++) {
        result[v[index]] = index;
    }

    return result;
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::computePointerTable(
        const std::vector<int32_t>& v) {
    std::vector<int32_t> result(v.size(), NewWeightsTableFormatMapper::paddingValue);

    int32_t numberOfLines = WEIGHTS_TABLE_READER_COUNT;
    int32_t numberOfColumns = v.size() / WEIGHTS_TABLE_READER_COUNT;

    int32_t currIndex = 0;
    for (int32_t line = 0; line < numberOfLines; line++) {
        for (int32_t column = 0; column < numberOfColumns; column++) {
            result[v[currIndex++]] = line * 8 + column;
        }
    }

    return result;
}

// zeroPointsKx[index] = value means that the zero point that is at the position "value"
// will be mapped to position "index" in the new format
// for example, zeroPointsK64[3] = 18 means that the zero point that is at position 18
// will be mapped to position 3 in the new format
// Also, these zero points tables will be used to construct statically the tables (named pointersKx) that will correctly
// format the data/sparsity-pointer vectors
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK16 =
        std::vector<int32_t>{0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK32 =
        std::vector<int32_t>{0, 2,  16, 18, 1, 3,  17, 19, 4,  6,  20, 22, 5,  7,  21, 23,
                             8, 10, 24, 26, 9, 11, 25, 27, 12, 14, 28, 30, 13, 15, 29, 31};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK48 = std::vector<int32_t>{
        0, 2,  16, 18, 32, 34, 1, 3,  17, 19, 33, 35, 4,  6,  20, 22, 36, 38, 5,  7,  21, 23, 37, 39,
        8, 10, 24, 26, 40, 42, 9, 11, 25, 27, 41, 43, 12, 14, 28, 30, 44, 46, 13, 15, 29, 31, 45, 47};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK64 =
        std::vector<int32_t>{0,  2,  16, 18, 32, 34, 48, 50, 1,  3,  17, 19, 33, 35, 49, 51, 4,  6,  20, 22, 36, 38,
                             52, 54, 5,  7,  21, 23, 37, 39, 53, 55, 8,  10, 24, 26, 40, 42, 56, 58, 9,  11, 25, 27,
                             41, 43, 57, 59, 12, 14, 28, 30, 44, 46, 60, 62, 13, 15, 29, 31, 45, 47, 61, 63};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK80 = std::vector<int32_t>{
        0,  2,  16, 18, 32, 34, 48, 50, 64, 66, 1,  3,  17, 19, 33, 35, 49, 51, 65, 67, 4,  6,  20, 22, 36, 38, 52,
        54, 68, 70, 5,  7,  21, 23, 37, 39, 53, 55, 69, 71, 8,  10, 24, 26, 40, 42, 56, 58, 72, 74, 9,  11, 25, 27,
        41, 43, 57, 59, 73, 75, 12, 14, 28, 30, 44, 46, 60, 62, 76, 78, 13, 15, 29, 31, 45, 47, 61, 63, 77, 79};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK96 = std::vector<int32_t>{
        0,  2,  16, 18, 32, 34, 48, 50, 64, 66, 80, 82, 1,  3,  17, 19, 33, 35, 49, 51, 65, 67, 81, 83,
        4,  6,  20, 22, 36, 38, 52, 54, 68, 70, 84, 86, 5,  7,  21, 23, 37, 39, 53, 55, 69, 71, 85, 87,
        8,  10, 24, 26, 40, 42, 56, 58, 72, 74, 88, 90, 9,  11, 25, 27, 41, 43, 57, 59, 73, 75, 89, 91,
        12, 14, 28, 30, 44, 46, 60, 62, 76, 78, 92, 94, 13, 15, 29, 31, 45, 47, 61, 63, 77, 79, 93, 95};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK112 = std::vector<int32_t>{
        0,   2,  16, 18, 32,  34,  48, 50, 64,  66,  80, 82, 96, 98,  1,   3,  17, 19,  33,  35, 49, 51, 65,
        67,  81, 83, 97, 99,  4,   6,  20, 22,  36,  38, 52, 54, 68,  70,  84, 86, 100, 102, 5,  7,  21, 23,
        37,  39, 53, 55, 69,  71,  85, 87, 101, 103, 8,  10, 24, 26,  40,  42, 56, 58,  72,  74, 88, 90, 104,
        106, 9,  11, 25, 27,  41,  43, 57, 59,  73,  75, 89, 91, 105, 107, 12, 14, 28,  30,  44, 46, 60, 62,
        76,  78, 92, 94, 108, 110, 13, 15, 29,  31,  45, 47, 61, 63,  77,  79, 93, 95,  109, 111};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK128 = std::vector<int32_t>{
        0,   2,   16,  18,  32,  34,  48,  50,  64,  66,  80,  82,  96,  98,  112, 114, 1,   3,   17,  19,  33,  35,
        49,  51,  65,  67,  81,  83,  97,  99,  113, 115, 4,   6,   20,  22,  36,  38,  52,  54,  68,  70,  84,  86,
        100, 102, 116, 118, 5,   7,   21,  23,  37,  39,  53,  55,  69,  71,  85,  87,  101, 103, 117, 119, 8,   10,
        24,  26,  40,  42,  56,  58,  72,  74,  88,  90,  104, 106, 120, 122, 9,   11,  25,  27,  41,  43,  57,  59,
        73,  75,  89,  91,  105, 107, 121, 123, 12,  14,  28,  30,  44,  46,  60,  62,  76,  78,  92,  94,  108, 110,
        124, 126, 13,  15,  29,  31,  45,  47,  61,  63,  77,  79,  93,  95,  109, 111, 125, 127};

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK16InversePermutation =
        computeInversePermutation(zeroPointsK16);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK32InversePermutation =
        computeInversePermutation(zeroPointsK32);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK48InversePermutation =
        computeInversePermutation(zeroPointsK48);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK64InversePermutation =
        computeInversePermutation(zeroPointsK64);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK80InversePermutation =
        computeInversePermutation(zeroPointsK80);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK96InversePermutation =
        computeInversePermutation(zeroPointsK96);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK112InversePermutation =
        computeInversePermutation(zeroPointsK112);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK128InversePermutation =
        computeInversePermutation(zeroPointsK128);

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::paddingValue = -1;
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::pointersK16 =
        computePointerTable(zeroPointsK16);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::pointersK32 =
        computePointerTable(zeroPointsK32);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::pointersK48 =
        computePointerTable(zeroPointsK48);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::pointersK64 =
        computePointerTable(zeroPointsK64);

std::vector<std::vector<int32_t>> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointTables =
        std::vector<std::vector<int32_t>>{zeroPointsK16, zeroPointsK32, zeroPointsK48,  zeroPointsK64,
                                          zeroPointsK80, zeroPointsK96, zeroPointsK112, zeroPointsK128};

std::vector<std::vector<int32_t>>
        vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointInversePermutationTables =
                std::vector<std::vector<int32_t>>{zeroPointsK16InversePermutation,  zeroPointsK32InversePermutation,
                                                  zeroPointsK48InversePermutation,  zeroPointsK64InversePermutation,
                                                  zeroPointsK80InversePermutation,  zeroPointsK96InversePermutation,
                                                  zeroPointsK112InversePermutation, zeroPointsK128InversePermutation};

std::vector<std::vector<int32_t>> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::pointerTables =
        std::vector<std::vector<int32_t>>{pointersK16, pointersK32, pointersK48, pointersK64};

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointTableByK(int32_t k) {
    return zeroPointTables[k / WEIGHTS_TABLE_SETS_MIN_ALIGNMENT - 1];
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointInversePermutationTableByK(
        int32_t k) {
    return zeroPointInversePermutationTables[k / WEIGHTS_TABLE_SETS_MIN_ALIGNMENT - 1];
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::getPointerTableByK(int32_t k) {
    return pointerTables[k / WEIGHTS_TABLE_SETS_MIN_ALIGNMENT - 1];
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::encodePositionInNewZeroPointOnlyTableLayout(
        int32_t zeroPointIndex, int32_t k) {
    // the pattern is repeating after 128 elements
    normalizeKAndReturnCurrentGroupOf128ZeroPoints(zeroPointIndex, k);

    auto map = getZeroPointInversePermutationTableByK(k);
    auto oldPosOffset = zeroPointIndex - zeroPointIndex % 128;
    auto newPos = oldPosOffset + map[zeroPointIndex % 128];
    return newPos;
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::decodePositionInNewZeroPointOnlyTableLayout(
        int32_t position, int32_t k) {
    // the pattern is repeating after 128 elements
    normalizeKAndReturnCurrentGroupOf128ZeroPoints(position, k);

    auto map = getZeroPointTableByK(k);
    auto oldPosOffset = position - position % 128;
    auto newPos = oldPosOffset + map[position % 128];
    return newPos;
}

int8_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::extractOneZPFromZPPalletizedByte(int8_t zeroPoint,
                                                                                             bool lowerZP) {
    const uint8_t shiftToBeginningOfByte = lowerZP ? 0 : 4;
    constexpr uint8_t mask = 0x0F;
    constexpr uint8_t signBitMask = 0x08;
    constexpr uint8_t negativeSignPreserver = 0xF0;
    int8_t result = (zeroPoint >> shiftToBeginningOfByte) & mask;
    if (result & signBitMask) {
        result = result | negativeSignPreserver;
    }
    return result;
}

uint8_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::extractOneZPFromZPPalletizedByte(uint8_t zeroPoint,
                                                                                              bool lowerZP) {
    const uint8_t shiftToBeginningOfByte = lowerZP ? 0 : 4;
    constexpr uint8_t mask = 0x0F;
    return (zeroPoint >> shiftToBeginningOfByte) & mask;
}

void vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::mapElementsToNewPointerTableFormat(
        const std::vector<int32_t>& ptrs, int32_t start, int32_t end, int32_t ptrStartingIndex,
        std::vector<int32_t>& result) {
    int32_t range = end - start;

    VPUX_THROW_WHEN(start % WEIGHTS_TABLE_READER_ALIGNMENT != 0,
                    "The starting index of the range ({0}) is not a multiple of {1}", start,
                    WEIGHTS_TABLE_READER_ALIGNMENT);
    VPUX_THROW_WHEN(range > WEIGHTS_TABLE_READER_ALIGNMENT, "Range ({0}) shouldn't be greater than {1}", range,
                    WEIGHTS_TABLE_READER_ALIGNMENT);
    VPUX_THROW_WHEN(start / WEIGHTS_TABLE_READER_ALIGNMENT != (end - 1) / WEIGHTS_TABLE_READER_ALIGNMENT,
                    "All pointers have to be from the same group of (at most) {0} pointers: {1} / {0} != {2} / {0}",
                    WEIGHTS_TABLE_READER_ALIGNMENT, start, end - 1);

    int32_t alignEndUpToMultipleOf16 = vpux::alignValUp(end, WEIGHTS_TABLE_SETS_MIN_ALIGNMENT);
    int32_t alignRangeUpToMultipleOf16 = alignEndUpToMultipleOf16 - start;

    auto map = getPointerTableByK(alignRangeUpToMultipleOf16);
    for (auto index = start; index < end; index++) {
        auto newPos = start + map[index % WEIGHTS_TABLE_READER_ALIGNMENT];
        result[newPos] = ptrs[ptrStartingIndex++];
    }

    // set the remaining elements up to the first multiple of 16 to the value of the first weight set pointer, that is
    // result[start]
    for (auto index = end; index < alignEndUpToMultipleOf16; index++) {
        auto newPos = start + map[index % WEIGHTS_TABLE_READER_ALIGNMENT];
        result[newPos] = result[start];
    }
}

void vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::constructNewPointerTableForWorkload(
        std::vector<int32_t>& mappedTable, int32_t workloadStartingIndex, int32_t workloadSize,
        int32_t ptrStartingIndex, const std::vector<int32_t>& ptrs) {
    int32_t countGroupsOf64Pointers = workloadSize / WEIGHTS_TABLE_READER_ALIGNMENT;
    int32_t remainingPointersInLastGroup = workloadSize % WEIGHTS_TABLE_READER_ALIGNMENT;

    for (int index = 0; index < countGroupsOf64Pointers; index++) {
        mapElementsToNewPointerTableFormat(
                ptrs, workloadStartingIndex + index * WEIGHTS_TABLE_READER_ALIGNMENT,
                workloadStartingIndex + index * WEIGHTS_TABLE_READER_ALIGNMENT + WEIGHTS_TABLE_READER_ALIGNMENT,
                ptrStartingIndex + index * WEIGHTS_TABLE_READER_ALIGNMENT, mappedTable);
    }

    if (remainingPointersInLastGroup) {
        mapElementsToNewPointerTableFormat(
                ptrs, workloadStartingIndex + countGroupsOf64Pointers * WEIGHTS_TABLE_READER_ALIGNMENT,
                workloadStartingIndex + countGroupsOf64Pointers * WEIGHTS_TABLE_READER_ALIGNMENT +
                        remainingPointersInLastGroup,
                ptrStartingIndex + countGroupsOf64Pointers * WEIGHTS_TABLE_READER_ALIGNMENT, mappedTable);
    }
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::constructNewPointerTable(
        ArrayRef<int32_t> workloadsSizes, ArrayRef<int32_t> ptrs) {
    int32_t mappedTableSize = 0;
    for (auto workloadSize : workloadsSizes) {
        mappedTableSize += vpux::alignValUp(workloadSize, WEIGHTS_TABLE_READER_ALIGNMENT);
    }

    std::vector<int32_t> mappedTable(mappedTableSize, -1);

    int32_t workloadStartingIndex = 0;
    int32_t ptrStartingIndex = 0;
    for (unsigned long index = 0; index < workloadsSizes.size(); index++) {
        constructNewPointerTableForWorkload(mappedTable, workloadStartingIndex, workloadsSizes[index], ptrStartingIndex,
                                            ptrs);
        workloadStartingIndex += vpux::alignValUp(workloadsSizes[index], WEIGHTS_TABLE_READER_ALIGNMENT);
        ptrStartingIndex += workloadsSizes[index];
    }

    return mappedTable;
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::constructNewPointerTable(
        ArrayRef<VPUIP::DPUTaskOp> tasks, ArrayRef<int32_t> ptrs) {
    std::vector<int32_t> workloadsSizes(tasks.size(), -1);
    std::vector<VPUIP::DPUTaskOp> tasksVector(tasks.begin(), tasks.end());

    for (unsigned long index = 0; index < tasksVector.size(); index++) {
        int32_t cStart = mlir::dyn_cast<mlir::IntegerAttr>(tasksVector[index].getOutStart()[2]).getInt();
        int32_t cEnd = mlir::dyn_cast<mlir::IntegerAttr>(tasksVector[index].getOutEnd()[2]).getInt();

        workloadsSizes[index] = cEnd - cStart + 1;
    }

    return constructNewPointerTable(workloadsSizes, ptrs);
}

//
// 5D weights
//

int32_t vpux::VPU::NCESparsity::get5DWeightPtrStep(mlir::Value weights) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[DimsGroups5D::Filter::IC];
    const auto KY = filterShape[DimsGroups5D::Filter::KY];
    const auto KX = filterShape[DimsGroups5D::Filter::KX];

    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
    const auto convAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
    const auto weightsElementCount = IC * KY * KX;

    VPUX_THROW_WHEN((weightsElementCount % convAlignment) != 0,
                    "NCEMatMul weights size must be a multiple of {0} but got {1}", convAlignment, weightsElementCount);

    const Bit eltSize = getElemTypeSize(weights.getType());

    return checked_cast<int32_t>(Byte(eltSize * IC * KY * KX).count());
}

std::vector<int32_t> vpux::VPU::NCESparsity::create5DWeightsTableData(
        mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights, const Const::ContentAttr& bias,
        int64_t outputChannels, VPU::NCESparsity::PPEConverterCb ppeConverter,
        VPU::NCESparsity::BiasConverterCb biasConverter, bool hasAutopad) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = VPU::NCESparsity::get5DWeightPtrStep(weights);
    const auto sparsityPtrStep = 0;

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(opInput.getType()).getElementType();
    const auto weightsElemType =
            weights ? mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType() : nullptr;

    const auto wtVec = VPU::NCESparsity::getWeightsTable(inElemType, opOutputElemType, weightPtrOffset, weightPtrStep,
                                                         sparsityPtrOffset, sparsityPtrStep, ppeConverter,
                                                         biasConverter, outputChannels, weightsElemType, bias);

    if (hasAutopad) {
        return VPU::NCESparsity::getExpandedWeightsTable(wtVec, outputChannels);
    }

    return wtVec;
}

std::vector<int32_t> vpux::VPU::NCESparsity::create5DWeightsTableData(
        mlir::Value opInput, mlir::Value opOutput, mlir::Value weights, const Const::ContentAttr& bias,
        int64_t outputChannels, VPU::NCESparsity::PPEConverterCb ppeConverter,
        VPU::NCESparsity::BiasConverterCb biasConverter, bool hasAutopad) {
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();
    return create5DWeightsTableData(opInput, outElemType, weights, bias, outputChannels, ppeConverter, biasConverter,
                                    hasAutopad);
}

mlir::Value vpux::VPU::NCESparsity::create5DWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                               ArrayRef<int32_t> weightsTable, int64_t outputChannels,
                                                               int64_t groups) {
    const auto elemType = getSInt32Type(builder.getContext());
    const Shape weightTableShape = {groups, outputChannels, 1, 1, VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, weightsTable);
}
