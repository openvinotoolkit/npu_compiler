//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/algo.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/Value.h>

namespace vpux::VPUIP {
class DPUTaskOp;
}  // namespace vpux::VPUIP

namespace vpux {
namespace VPU {

namespace NCESparsity {

// base_ptr is 9bits size
const int BASE_PTR_SIZE = 9;

constexpr int32_t SPARSITY_PTR_WHEN_NO_SPARSITY = 0xFFFFFF;

const unsigned int DEFAULT_SPARSIFIABLE_INPUT_OPERAND_ID = 0;
const unsigned int ELTWISE_SPARSIFIABLE_SECOND_INPUT_OPERAND_ID = 1;

constexpr std::int32_t ALIGNMENT_REQUIREMENT_IN_ELEMENTS = 16;
constexpr std::int32_t WEIGHTS_TABLE_READER_COUNT = 8;
// each weight reader receives up to 32 bytes of data. for data/sparsity pointer tables, each element has 4 bytes, which
// means that there are (at most) 8 elements to be read by each weight reader. so, by multiplying
// WEIGHTS_TABLE_READER_COUNT with this value (8), we obtain WEIGHTS_TABLE_READER_ALIGNMENT = 64. For each workload, its
// size has to be rounded up to the nearest multiple of WEIGHTS_TABLE_READER_ALIGNMENT = 64, the values to be added
// representing the padding.
constexpr std::int32_t WEIGHTS_TABLE_READER_ALIGNMENT = WEIGHTS_TABLE_READER_COUNT * 8;
// In case of zero point only table, each workload is logically divided in groups of 128 zero-points, excepting the last
// one which will contain the remaining ones, padded with -1 up to the nearest multiple of
// WEIGHTS_TABLE_READER_ALIGNMENT=64.
// Regarding the storage, 1. For the 8-bit case the constructed zero-point table will be aligned to 64 bytes, while all
// non-last groups will be aligned to 128 bytes.
// 2. For the 4-bit case the constructed zero-point table will be aligned to 32 bytes (as 2 zero-points are stored in
// each byte), while all non-last groups will be aligned to 64 (half of 128).
// More details regarding the usage of this constant can be found in the constructNewZeroPointOnlyTableForWorkload
// function.
constexpr std::int32_t ZERO_POINT_TABLE_READER_ALIGNMENT = 128;
// in each final (and padded) group of 64 weights table sets there has to be a multiple of 16 valid sets (16, 32, 48 or
// 64), that can be read from memory
constexpr std::int32_t WEIGHTS_TABLE_SETS_MIN_ALIGNMENT = 16;

constexpr std::int32_t PADDING_POSITION_INDICATOR = -1;
constexpr std::int32_t INVALID_POSITION_OUT_OF_RANGE = -2;

enum class Mode { DW_CONV, POOL };

template <typename ScaleElemType>
llvm::unique_function<ScaleElemType(size_t)> getMultShiftFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                              mlir::Type weightsType,
                                                              VPU::NCESparsity::PPEConverterCb ppeConverter, size_t OC,
                                                              mlir::FloatAttr constMultiplyFpScale,
                                                              mlir::Type scalesType = nullptr) {
    if (weightsType != nullptr) {
        auto inStorageType = mlir::quant::QuantizedType::castToStorageType(inElemType);
        if ((mlir::isa<mlir::quant::QuantizedType>(inElemType) && !mlir::isa<mlir::quant::QuantizedType>(weightsType) &&
             !mlir::isa<mlir::Float8E5M2Type>(inStorageType) && !mlir::isa<mlir::Float8E4M3FNType>(inStorageType))) {
            VPUX_THROW("Unsupported In/Wt mixed precision. Got: in type {0}, wt type {1}", inElemType, weightsType);
        }
    }

    auto inQuantScales = extractScalesOrDefault(inElemType, 1.0);
    auto weightsQuantScales = extractScalesOrDefault(weightsType, 1.0);
    auto outQuantScales = extractScalesOrDefault(outElemType, 1.0);
    const auto constMultiplyScale = constMultiplyFpScale ? constMultiplyFpScale.getValueAsDouble() : 1.0;

    broadcast(inQuantScales, OC);
    broadcast(weightsQuantScales, OC);
    broadcast(outQuantScales, OC);

    // Weights table scale is computed as:
    // input_quant_scale * weights_quant_scale  * static_scale / output_quant_scale
    // splitNCEConvolutionOverIC is based on this assumption.
    std::vector<double> rescale;
    rescale.reserve(OC);
    for (size_t i = 0; i < OC; ++i) {
        const auto scale = (weightsQuantScales[i] * inQuantScales[i]) / outQuantScales[i] * constMultiplyScale;
        rescale.push_back(scale);
    }

    return [rescale = std::move(rescale), inElemType, scalesType, ppeConverter](size_t oc) {
        const auto quantScale = rescale[oc];

        // scalesType is set only when using the new weights table format, as in this case
        // the scale data type is independent of the input data type
        if (scalesType != nullptr) {
            auto multShift = ppeConverter(0, 0, rescale[oc], scalesType);
            return std::get<ScaleElemType>(multShift);
        } else {
            const QuantizationApproximation scaleApproximation(quantScale);
            auto multShift = ppeConverter(checked_cast<uint8_t>(scaleApproximation.shift()),
                                          checked_cast<int16_t>(scaleApproximation.mult()), rescale[oc], inElemType);

            return std::get<ScaleElemType>(multShift);
        }
    };
}

// this function is used for formatting both data pointers and sparsity pointers
// for getting a formatted sparsity pointer, provide only the pointer (as there is no zeropoint in this case)
template <typename T>
int32_t getDataOrSparsityPointer(int32_t pointer, T zeroPoint = 0) {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "Invalid zero-point type, expected int8_t or uint8_t");

    constexpr int32_t ZP_OFFSET = 24;
    T ZP_VALUE = zeroPoint;

    int32_t PTR_VALUE = pointer;
    constexpr int32_t PTR_VALIDATION_MASK = 0xFF00000F;

    if (PTR_VALUE & PTR_VALIDATION_MASK) {
        VPUX_THROW("The value that's stored for the pointer ({0}) has to fit on 24 bits, having the last 4 bits equal "
                   "to 0",
                   PTR_VALUE);
    }

    return (ZP_VALUE << ZP_OFFSET) | PTR_VALUE;
}

int64_t getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int32_t getWeightPtrStep(mlir::Value weights);

std::vector<int32_t> getExpandedWeightsTable(ArrayRef<int32_t> weightsTableVector, int64_t OC);

std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType,
                                     std::optional<int32_t> weightsPtrOffset, int32_t weightsPtrStep,
                                     std::optional<int32_t> sparsityPtrOffset, int32_t sparsityPtrStep,
                                     VPU::NCESparsity::PPEConverterCb ppeConverter,
                                     VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {},
                                     mlir::FloatAttr constScale = nullptr);
std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> weightPtrs,
                                     ArrayRef<int32_t> sparsityPtrs, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                     VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {},
                                     mlir::FloatAttr constScale = nullptr);

//
// NewWeightsTableFormatMapper
//

class NewWeightsTableFormatMapper {
public:
    // utility functions
    static std::vector<int32_t> computeInversePermutation(std::vector<int32_t> v);
    static std::vector<int32_t> computePointerTable(const std::vector<int32_t>& v);
    static std::vector<int32_t> getZeroPointTableByK(int32_t k);
    static std::vector<int32_t> getZeroPointInversePermutationTableByK(int32_t k);
    static std::vector<int32_t> getPointerTableByK(int32_t k);

    // Some observations regarding the below encoding and decoding functions between the zero point initial index and
    // its new index in the new format; these functions use statically-constructed vectors in order to deduce the result
    // 1A) regarding the encoding functions, the zeroPointIndex argument represents an element's position in the
    // original unshuffled zp table; on the other hand, this functions return the the position/index of the same element
    // in the shuffled and constructed zp table, hence the name "encoding"
    // 1B) for 4-bit zero-points, for accessing the actual zp from the constructed table, you should divide by 2 the
    // position returned by this function and extract the corresponding 4-bits from the element residing there (bits
    // 0...3 if returned position is a multiple of 2 and bits 4...7 otherwise)
    static int32_t encodePositionInWorkloadInNewZeroPointOnlyTableLayout(int32_t position, int32_t k);
    // 1C) encodePositionInNewZeroPointOnlyTableLayout returns -2 if the given position is invalid, out of the range of
    // valid indices (no zero-point resides there)
    static int32_t encodePositionInNewZeroPointOnlyTableLayout(int32_t position, std::vector<int32_t> workloads);
    // 2A) in these decoding functions, the position represents the index at which a given element is in the shuffled
    // and constructed zp table; on the other hand, they return the position/index of the same element in the original
    // unshuffled zp table, hence the name "decoding"
    // 2B) for 4-bit zero-points, for computing the position of the actual zp from the unshuffled table, you should
    // multiply by 2 the position of the zero-point in the shuffled table and pass it to the below function, as there
    // are two zero-points in each byte (add 1 as well if you are interested in the upper byte zp)
    static int32_t decodePositionInWorkloadInNewZeroPointOnlyTableLayout(int32_t position, int32_t k);
    // 2C) decodePositionInNewZeroPointOnlyTableLayout returns -1 if a padding value resides at that position and -2 if
    // the given position is invalid, out of the range of valid indices (no zero-point/padding value resides there)
    static int32_t decodePositionInNewZeroPointOnlyTableLayout(int32_t position, std::vector<int32_t> workloads);

    template <typename T>
    static void setFourBitZPInPalletizedByte(ArrayRef<T> table, std::vector<T>& result, int32_t newPos,
                                             int32_t ptrStartingIndex) {
        const T currElem = table[ptrStartingIndex] & 0x0F;
        const int32_t startPos = newPos % 2 == 1 ? 4 : 0;
        const uint8_t mask = 0x0F << startPos;
        result[newPos / 2] &= ~mask;
        result[newPos / 2] |= currElem << startPos;
    }

    // in a zero-point palletized byte there are stored 2 zero-points, each one on 4 bits
    // if lowerZP = true, the zero-point stored in the least significant 4 bits will be returned
    // if lowerZP = false, the zero-point stored in the most significant 4 bits will be returned
    static int8_t extractOneZPFromZPPalletizedByte(int8_t zeroPoint, bool lowerZP);

    // in a zero-point palletized byte there are stored 2 zero-points, each one on 4 bits
    // if lowerZP = true, the zero-point stored in the least significant 4 bits will be returned
    // if lowerZP = false, the zero-point stored in the most significant 4 bits will be returned
    static uint8_t extractOneZPFromZPPalletizedByte(uint8_t zeroPoint, bool lowerZP);

    // utility function used by constructNewZeroPointOnlyTableForWorkload
    template <typename T>
    static void mapElementsToNewZeroPointOnlyTableFormat(bool isZeroPoint4Bit, ArrayRef<T> ptrs, int32_t start,
                                                         int32_t end, int32_t ptrStartingIndex,
                                                         std::vector<T>& result) {
        int32_t range = end - start;

        VPUX_THROW_WHEN(start % WEIGHTS_TABLE_READER_ALIGNMENT != 0,
                        "The starting index of the range ({0}) is not a multiple of {1}", start,
                        WEIGHTS_TABLE_READER_ALIGNMENT);
        VPUX_THROW_WHEN(range % WEIGHTS_TABLE_SETS_MIN_ALIGNMENT != 0, "Range length ({0}) is not a multiple of {1}",
                        range, WEIGHTS_TABLE_SETS_MIN_ALIGNMENT);
        VPUX_THROW_WHEN(range < WEIGHTS_TABLE_SETS_MIN_ALIGNMENT || range > ZERO_POINT_TABLE_READER_ALIGNMENT,
                        "Range ({0}) should not be less than {1}, nor greater than {2}", range,
                        WEIGHTS_TABLE_SETS_MIN_ALIGNMENT, ZERO_POINT_TABLE_READER_ALIGNMENT);

        auto map = getZeroPointInversePermutationTableByK(range);
        for (auto index = start; index < end; index++) {
            if (isZeroPoint4Bit) {
                auto newPos = start + map[index - start];
                setFourBitZPInPalletizedByte(ptrs, result, newPos, ptrStartingIndex++);
            } else {
                auto newPos = start + map[index - start];
                result[newPos] = ptrs[ptrStartingIndex++];
            }
        }
    }

    template <typename T>
    static void constructNewZeroPointOnlyTableForWorkload(bool isZeroPoint4Bit, std::vector<T>& mappedTable,
                                                          int32_t workloadStartingIndex, int32_t workloadSize,
                                                          int32_t ptrStartingIndex, ArrayRef<T> ptrs) {
        int32_t alignment = ZERO_POINT_TABLE_READER_ALIGNMENT;
        int32_t countGroupsOf128ZP = workloadSize / alignment;
        int32_t remainingZPInLastGroup = workloadSize % alignment;

        // In what follows, we will use alignment as a shortcut for logical alignment, in order to keep this description
        // abstract and independent of the storage/physical part, in which each byte can store either one 8-bit
        // zero-point or two 4-bit zero-points.
        // We firstly arrange the workload zero-points based on chunks (groups) of 128 elements and finally (see the
        // next if conditional) we will arrange the remainig chunk of less than 128 elements, if that's the case; the
        // last chunk will be aligned to 64 if there are at most 64 elements in it and to 128 otherwise; therefore, due
        // to this last chunk/group, a workload can end up being aligned to 64 and not to 128 (see the
        // constructNewZeroPointOnlyTable function as well, where we align each workload to 64 elements). It should be
        // pointed out that this logic gets enforced because 128 zero points end up at different positions whether they
        // are aligned as being part of 2 groups of 64 elements or as part of a single group; compared to
        // data/sparsity-pointer tables, for which we have mapping tables only up to K=64 (e.g. pointersK48), for zero
        // point only tables we have such mappings for K up to 128 (e.g. zeroPointsK112).
        for (int index = 0; index < countGroupsOf128ZP; index++) {
            mapElementsToNewZeroPointOnlyTableFormat(isZeroPoint4Bit, ptrs, workloadStartingIndex + index * alignment,
                                                     workloadStartingIndex + index * alignment + alignment,
                                                     ptrStartingIndex + index * alignment, mappedTable);
        }

        if (remainingZPInLastGroup) {
            mapElementsToNewZeroPointOnlyTableFormat(
                    isZeroPoint4Bit, ptrs, workloadStartingIndex + countGroupsOf128ZP * alignment,
                    workloadStartingIndex + countGroupsOf128ZP * alignment + remainingZPInLastGroup,
                    ptrStartingIndex + countGroupsOf128ZP * alignment, mappedTable);
        }
    }

    // function that constructs a zero point only table (in the new format) starting from a vector that
    // contains the zero points in ascending order based on their indices
    // workloadSizes is needed for the shuffling logic
    // typename T should be one of uint8_t (for U4 and U8) and int8_t (for I4 and I8)
    // set isZeroPoint4Bit = true only if the zero points should be stored on 4 bits each
    template <typename T>
    static std::vector<T> constructNewZeroPointOnlyTable(bool isZeroPoint4Bit, ArrayRef<int32_t> workloadSizes,
                                                         ArrayRef<T> ptrs) {
        static_assert(
                std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Typename should be one of uint8_t (for U4 and U8 zero-points) and int8_t (for I4 and I8 zero-points)");

        int32_t mappedTableSize = 0;
        for (auto workloadSize : workloadSizes) {
            if (isZeroPoint4Bit) {
                mappedTableSize += vpux::alignValUp(workloadSize / 2, WEIGHTS_TABLE_READER_ALIGNMENT / 2);
            } else {
                mappedTableSize += vpux::alignValUp(workloadSize, WEIGHTS_TABLE_READER_ALIGNMENT);
            }
        }

        std::vector<T> mappedTable(mappedTableSize, -1);

        int32_t workloadStartingIndex = 0;
        int32_t ptrStartingIndex = 0;
        for (unsigned long index = 0; index < workloadSizes.size(); index++) {
            constructNewZeroPointOnlyTableForWorkload(isZeroPoint4Bit, mappedTable, workloadStartingIndex,
                                                      workloadSizes[index], ptrStartingIndex, ptrs);
            workloadStartingIndex += vpux::alignValUp(workloadSizes[index], WEIGHTS_TABLE_READER_ALIGNMENT);
            ptrStartingIndex += workloadSizes[index];
        }

        return mappedTable;
    }

    static void mapElementsToNewPointerTableFormat(const std::vector<int32_t>& ptrs, int32_t start, int32_t end,
                                                   int32_t ptrStartingIndex, std::vector<int32_t>& result);
    static void constructNewPointerTableForWorkload(std::vector<int32_t>& mappedTable, int32_t workloadStartingIndex,
                                                    int32_t workloadSize, int32_t ptrStartingIndex,
                                                    const std::vector<int32_t>& ptrs);
    static std::vector<int32_t> constructNewPointerTable(ArrayRef<int32_t> workloadSizes, ArrayRef<int32_t> ptrs);
    static std::vector<int32_t> constructNewPointerTable(ArrayRef<VPUIP::DPUTaskOp> tasks, ArrayRef<int32_t> ptrs);

private:
    static std::vector<int32_t> zeroPointsK16;
    static std::vector<int32_t> zeroPointsK32;
    static std::vector<int32_t> zeroPointsK48;
    static std::vector<int32_t> zeroPointsK64;
    static std::vector<int32_t> zeroPointsK80;
    static std::vector<int32_t> zeroPointsK96;
    static std::vector<int32_t> zeroPointsK112;
    static std::vector<int32_t> zeroPointsK128;

    static std::vector<int32_t> zeroPointsK16InversePermutation;
    static std::vector<int32_t> zeroPointsK32InversePermutation;
    static std::vector<int32_t> zeroPointsK48InversePermutation;
    static std::vector<int32_t> zeroPointsK64InversePermutation;
    static std::vector<int32_t> zeroPointsK80InversePermutation;
    static std::vector<int32_t> zeroPointsK96InversePermutation;
    static std::vector<int32_t> zeroPointsK112InversePermutation;
    static std::vector<int32_t> zeroPointsK128InversePermutation;

    static std::vector<int32_t> pointersK16;
    static std::vector<int32_t> pointersK32;
    static std::vector<int32_t> pointersK48;
    static std::vector<int32_t> pointersK64;

    static std::vector<std::vector<int32_t>> zeroPointTables;
    static std::vector<std::vector<int32_t>> zeroPointInversePermutationTables;
    static std::vector<std::vector<int32_t>> pointerTables;
};
/**
 * @brief generate a dense data-pointer table
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrs - the addresses at which the data-pointers will be stored
 * @param OC - number of output channels
 * @param zeroPoints - array containing per-channel zero-points (one zero-point for each channel is needed; if the array
 * is empty, each zero-point will be 0) - for each channel, the zero-point and the actual weights starting address will
 * be embedded in one entry of 32 bits, formally called data-pointer
 *
 * @return constructed and formatted data-pointer table.
 */
template <typename T>
std::vector<int32_t> getDataPointerTable(mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> workloadSizes,
                                         ArrayRef<int32_t> weightsPtrs, int64_t OC,
                                         ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "Invalid zero-point type, expected int8_t or uint8_t");

    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create data pointer table without operation input/output types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Data pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(
            static_cast<int64_t>(zeroPoints.size()) != OC && static_cast<int64_t>(zeroPoints.size()) != 0,
            "Zero-points size {0} different than output channels {1} (and different than 0 - the default value)",
            zeroPoints.size(), OC);

    std::vector<T> zeroPointsVector(zeroPoints.begin(), zeroPoints.end());
    if (zeroPointsVector.empty()) {
        zeroPointsVector.assign(OC, 0);
    }

    std::vector<std::int32_t> dataPointerTableVals(OC, 0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);

        dataPointerTableVals[oc] = getDataOrSparsityPointer(weightsPtrs[oc], zeroPointsVector[oc]);
    });

    return NewWeightsTableFormatMapper::constructNewPointerTable(workloadSizes, dataPointerTableVals);
}

/**
 * @brief generate a dense data-pointer table
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrOffset - the address at which the first data-pointer is stored (defaults to 0)
 * @param weightsPtrStep - distance between two consecutive data-pointer addresses
 * @param OC - number of output channels
 * @param zeroPoints - array containing per-channel zero-points (can be empty if not needed)
 *
 * @return constructed and formatted data-pointer table.
 */
template <typename T>
std::vector<int32_t> getDataPointerTable(mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> workloadSizes,
                                         std::optional<int32_t> weightsPtrOffset, int32_t weightsPtrStep, int64_t OC,
                                         ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    auto weightsPtrOffsetValue = weightsPtrOffset.value_or(0);

    SmallVector<int32_t> weightsPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffsetValue;
        weightsPtrOffsetValue += weightsPtrStep;
    }

    return getDataPointerTable<T>(inElemType, outElemType, workloadSizes, weightsPtrs, OC, zeroPoints);
}

/**
 * @brief generate sparse data-pointer tables
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrs - the addresses at which the data-pointers will be stored
 * @param sparsityPtrs - the addresses at which the sparsity-pointers will be stored
 * @param OC - number of output channels
 * @param zeroPoints - array containing per-channel zero-points (one zero-point for each channel is needed; if the array
 * is empty, each zero-point will be 0) - for each channel, the zero-point and the actual weights starting address will
 * be embedded in one entry of 32 bits, formally called data-pointer
 *
 * @return constructed and formatted data-pointer and sparsity-pointer tables.
 */
template <typename T>
std::pair<std::vector<int32_t>, std::vector<int32_t>> getSparseDataPointerTablePair(
        mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> workloadSizes, ArrayRef<int32_t> weightsPtrs,
        ArrayRef<int32_t> sparsityPtrs, int64_t OC, ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "Invalid zero-point type, expected int8_t or uint8_t");

    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create sparse data pointer tables without operation input/output/weights types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Data pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(static_cast<int64_t>(sparsityPtrs.size()) != OC,
                    "Sparsity pointers size {0} different than output channels {1}", sparsityPtrs.size(), OC);
    VPUX_THROW_WHEN(
            static_cast<int64_t>(zeroPoints.size()) != OC && static_cast<int64_t>(zeroPoints.size()) != 0,
            "Zero-points size {0} different than output channels {1} (and different than 0 - the default value)",
            zeroPoints.size(), OC);

    std::vector<T> zeroPointsVector(zeroPoints.begin(), zeroPoints.end());
    if (zeroPointsVector.empty()) {
        zeroPointsVector.assign(OC, 0);
    }

    std::vector<std::int32_t> dataPointerTableVals(OC, 0);
    std::vector<std::int32_t> sparsityPointerTableVals(OC, 0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);
        VPUX_THROW_UNLESS(sparsityPtrs[oc] == SPARSITY_PTR_WHEN_NO_SPARSITY ||
                                  sparsityPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "sparsityPtrs[{0}] must be aligned to {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          sparsityPtrs[oc]);

        dataPointerTableVals[oc] = getDataOrSparsityPointer(weightsPtrs[oc], zeroPointsVector[oc]);
        sparsityPointerTableVals[oc] = getDataOrSparsityPointer<uint8_t>(sparsityPtrs[oc]);
    });

    const auto dataPointerTableFormatted =
            NewWeightsTableFormatMapper::constructNewPointerTable(workloadSizes, dataPointerTableVals);
    const auto sparsityPointerTableFormatted =
            NewWeightsTableFormatMapper::constructNewPointerTable(workloadSizes, sparsityPointerTableVals);

    return {dataPointerTableFormatted, sparsityPointerTableFormatted};
}

/**
 * @brief generate sparse data-pointer tables
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrOffset - the address at which the first data-pointer is stored (defaults to 0)
 * @param weightsShape - weights tensor's shape
 * @param sparsityPtrOffset - the address at which the first sparsity-pointer is stored
 * @param sparsityMap - one sparsity value for each weight (sparsity == 0 means weight == 0, sparsity != 0 means
 * weight != 0)
 * @param OC - number of output channels
 * @param weightsElemType - weights tensor type
 * @param zeroPoints - array containing per-channel zero-points (can be empty if not needed)
 *
 * @return constructed and formatted data-pointer and sparsity-pointer tables.
 *
 * NOTE: even if sparsityMap can contain uint8 values, in the generated structure we will have only 0s (when weight ==
 * 0), and 1s (when weight != 0), each stored on 1 bit.
 */
template <typename T>
std::pair<std::vector<int32_t>, std::vector<int32_t>> getSparseDataPointerTablePair(
        mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> workloadSizes,
        std::optional<int32_t> weightsPtrOffset, ShapeRef weightsShape, int32_t sparsityPtrOffset,
        ArrayRef<uint8_t> sparsityMap, int64_t OC, mlir::Type weightsElemType, ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    VPUX_THROW_WHEN(weightsElemType == nullptr,
                    "Can't create sparse data pointer tables without operation weights type");

    const int32_t weightSetsCounter =
            weightsShape[Dims4D::Filter::IC] * weightsShape[Dims4D::Filter::KY] * weightsShape[Dims4D::Filter::KX];

    VPUX_THROW_WHEN(static_cast<size_t>(weightSetsCounter * weightsShape[Dims4D::Filter::OC]) != sparsityMap.size(),
                    "There has to be one sparse value for each weight");

    auto weightsPtrOffsetValue = weightsPtrOffset.value_or(0);
    auto sparsityPtrStep = vpux::alignValUp(weightSetsCounter / 8, ALIGNMENT_REQUIREMENT_IN_ELEMENTS);

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    int32_t sparsityTableOffset = 0;

    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffsetValue;

        int32_t nonZeroWeightsCounter =
                std::count_if(sparsityMap.begin() + sparsityTableOffset,
                              sparsityMap.begin() + sparsityTableOffset + weightSetsCounter, [](int elem) {
                                  return elem != 0;
                              });
        const Bit eltSize = getElemTypeSize(weightsElemType);

        auto weightsPtrStepSizeInBits =
                vpux::alignValUp(checked_cast<int32_t>(eltSize.count()) * nonZeroWeightsCounter, CHAR_BIT);
        auto weightsPtrStep = weightsPtrStepSizeInBits / CHAR_BIT;
        weightsPtrOffsetValue += vpux::alignValUp(weightsPtrStep, ALIGNMENT_REQUIREMENT_IN_ELEMENTS);

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += sparsityPtrStep;
        sparsityTableOffset += weightSetsCounter;
    }

    return getSparseDataPointerTablePair<T>(inElemType, outElemType, workloadSizes, weightsPtrs, sparsityPtrs, OC,
                                            zeroPoints);
}

template <typename T>
std::vector<T> getScaleTable(mlir::Type inElemType, mlir::Type outElemType,
                             VPU::NCESparsity::PPEConverterCb ppeConverter, int64_t OC,
                             mlir::Type weightsElemType = nullptr, mlir::FloatAttr constScale = nullptr) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    mlir::Type scalesType = nullptr;
    if (std::is_same<T, float>::value) {
        scalesType = mlir::Float32Type::get(inElemType.getContext());
    } else if (std::is_same<T, vpux::type::float8_e5m2>::value) {
        scalesType = mlir::Float8E5M2Type::get(inElemType.getContext());
    } else if (std::is_same<T, vpux::type::float8_e4m3>::value) {
        scalesType = mlir::Float8E4M3FNType::get(inElemType.getContext());
    } else {
        VPUX_THROW("Only F32/F8E5M2/F8E4M3 scales are supported in the new weights table format");
    }

    auto getMultShift = getMultShiftFunc<T>(inElemType, outElemType, weightsElemType, ppeConverter,
                                            checked_cast<size_t>(OC), constScale, scalesType);

    std::vector<T> scaleTableVals(OC, 0.0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        scaleTableVals[oc] = getMultShift(oc);
    });

    return scaleTableVals;
}

std::vector<float> getBiasTable(mlir::Type inElemType, mlir::Type outElemType,
                                VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {});

// typename T should be one of uint8_t (for U4 and U8) and int8_t (for I4 and I8)
// set isZeroPoint4Bit = true only if the zero points should be stored on 4 bits each
// so, for U4 zero points set: T=uint8_t and isZeroPoint4Bit=true
// for U8 zero points set: T=uint8_t and isZeroPoint4Bit=false
// for I4 zero points set: T=int8_t and isZeroPoint4Bit=true
// for I8 zero points set: T=int8_t and isZeroPoint4Bit=false
template <typename T>
std::vector<T> getZeroPointOnlyTable(ArrayRef<int32_t> workloadSizes, int64_t OC, mlir::Type weightsElemType,
                                     bool isZeroPoint4Bit, ArrayRef<T> zeroPoints) {
    VPUX_THROW_WHEN(weightsElemType == nullptr || !mlir::isa<mlir::quant::QuantizedType>(weightsElemType),
                    "weightsElemType has to be a quantized type, not {0}", weightsElemType);

    mlir::Type storageType = mlir::cast<mlir::quant::QuantizedType>(weightsElemType).getStorageType();

    if (std::is_same_v<T, uint8_t>) {
        VPUX_THROW_UNLESS(storageType.isUnsignedInteger(8) || (storageType.isUnsignedInteger(4) && isZeroPoint4Bit),
                          "The storage type of the quantized weightsElemType {0} has to be the same as the type of the "
                          "zero points",
                          storageType);
    } else if (std::is_same_v<T, int8_t>) {
        VPUX_THROW_UNLESS(
                ((storageType.isSignedInteger(8) || storageType.isSignlessInteger(8))) ||
                        ((storageType.isSignedInteger(4) || storageType.isSignlessInteger(4)) && isZeroPoint4Bit),
                "The storage type of the quantized weightsElemType {0} has to be the same as the type of the "
                "zero points",
                storageType);
    } else {
        VPUX_THROW("Template argument type has to be uint8_t or int8_t");
    }

    VPUX_THROW_WHEN(static_cast<int64_t>(zeroPoints.size()) != OC,
                    "Zero-points size {0} different than output channels {1}", zeroPoints.size(), OC);

    return NewWeightsTableFormatMapper::constructNewZeroPointOnlyTable(isZeroPoint4Bit, workloadSizes, zeroPoints);
}

std::vector<int32_t> patchWeightsTableSparsityPtrs(const std::vector<std::int32_t>& weightsTableVals,
                                                   const int32_t sparsityPtrOffset, const int32_t sparsityPtrStep,
                                                   std::optional<int64_t> origOC = std::nullopt);

// newFormat should be set to true only if the split weight table format is used, which consists of 5 tables that
// replace the legacy weight table: data pointer table, sparsity pointer table, scale table, bias table and per-channel
// zero-point table. In this case, this function returns the same shape for each of these tables, having only one value
// for each output channel.
Shape inferWeightsTableShape(int64_t OC, bool newFormat = false);
Shape infer5DWeightsTableShape(int64_t OC, int64_t groups, bool newFormat = false);
Shape inferWeightsSparsityMapShape(ShapeRef dataShape);

mlir::FailureOr<SmallVector<double>> getRescaledBias(const Const::ContentAttr& biasAttr, mlir::Type inElemType,
                                                     mlir::Type filterElemType, int64_t OC);

double getSparsityRatio(vpux::NDTypeInterface weightsType, int64_t compressedSize);
double getSparsityRatio(vpux::NDTypeInterface weightsType, ArrayRef<int64_t> numNonSparseElemsPerOC);

bool isSparsifiableWeightsOperand(mlir::Value operand);
bool isSuperdenseRequired(const DimsOrder outOrder, const ShapeRef outShape, const mlir::Type outElemType);

// 5D weights.
int32_t get5DWeightPtrStep(mlir::Value weights);

//
// Convert real numbers to fixed point S16.16 format.
//

int32_t toFixedPoint(const double realVal);

//
// Convert real numbers to hex format.
//

int32_t toHex(double realVal);

//
// RuntimeSparsityStatsProvider
//

class RuntimeSparsityStatsProvider {
    const double MINIMAL_SPARSITY_THRESHOLD = 0.2;

public:
    RuntimeSparsityStatsProvider(mlir::func::FuncOp func, vpux::Logger log);

    bool containsStatistics() const;
    bool likelySparsityConsumer(mlir::Operation* op, int64_t requestedInputId) const;

private:
    vpux::Logger _logger;
    std::multimap<std::string, net::SparsityInfoOp> _lookup;
};

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux
