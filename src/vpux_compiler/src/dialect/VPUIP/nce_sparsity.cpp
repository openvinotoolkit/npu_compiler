//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/enums.hpp"

using namespace vpux;
using namespace VPUIP;

namespace {

bool isMixedPrecisionSupported(VPUIP::ArchKind arch) {
    return arch == ArchKind::MTL;
}

template <class T>
void broadcast(SmallVectorImpl<T>& values, size_t size) {
    VPUX_THROW_UNLESS(values.size() <= size, "Cannot broadcast to size {0}", size);
    if (values.size() == size) {
        return;
    }
    VPUX_THROW_UNLESS(values.size() == 1, "Broadcast from scalar is only supported");
    values.resize(size, values.front());
}

Scales exractWeightsScales(mlir::Type weightsElemType) {
    if (weightsElemType == nullptr || !weightsElemType.isa<mlir::quant::QuantizedType>()) {
        return SmallVector<double>{1.0};
    }

    return extractScalesAndZeroPoints(weightsElemType).first;
}

mlir::Type tryGetQuantizedStorageType(mlir::Type elemType) {
    if (auto quant = elemType.dyn_cast_or_null<mlir::quant::QuantizedType>()) {
        return quant.getStorageType();
    }

    return elemType;
}

int64_t getWindowSize(int64_t kernelW, int64_t strideW, mlir::Type elemType) {
    VPUX_THROW_UNLESS(kernelW <= 11, "Unsupported kernel size {0}. Supported size up to 11", kernelW);

    // Select the maximum window size not exceeding 32 bytes
    // by iterating through the MPE_NUM values (2, 4, 8, 16)

    auto actualType = tryGetQuantizedStorageType(elemType);
    VPUX_THROW_UNLESS(actualType.isInteger(CHAR_BIT) || actualType.isF16(), "Supported only U8/I8 and FP16 types {0}",
                      actualType);

    // Only MPE0, MPE4, MPE8 and MPE12 support FP16 data format
    const int mpeNumLimit = actualType.isF16() ? 4 : 16;

    const Bit typeSizeInBits = getElemTypeSize(actualType);

    // Window size is limited to 32 bytes by HW. Size of the data type
    // needs to be accounted to find the max (32 for U8, 16 for FP16)
    const int64_t maxWindowSize = 32 / (typeSizeInBits.count() / 8);
    int64_t maxMpeWindowSize = 64;

    int64_t windowSize = 0;
    int mpeNum = 1;

    while (mpeNum <= mpeNumLimit) {
        if (strideW <= kernelW) {
            windowSize = kernelW + strideW * (mpeNum - 1);
        } else {
            windowSize = kernelW * mpeNum;
        }

        if (windowSize <= maxWindowSize)
            maxMpeWindowSize = windowSize;

        mpeNum *= 2;
    }

    return maxMpeWindowSize;
}

std::vector<uint8_t> getBitPattern(mlir::ArrayRef<int64_t> kernelSize, int64_t windowSize) {
    const auto kernelW = kernelSize[0];
    const auto kernelH = kernelSize[1];

    VPUX_THROW_UNLESS(windowSize >= kernelW,
                      "windowsSize must be greater than or equal to kernelW. windowsSize={0}, kernelW={1}", windowSize,
                      kernelW);

    const auto numBitsSet = kernelW;
    const auto numBitsClear = windowSize - kernelW;

    SmallVector<uint8_t> window;
    window.reserve(windowSize);
    window.insert(window.end(), numBitsSet, 1);
    window.insert(window.end(), numBitsClear, 0);

    const auto numOfRepeat = kernelH;

    std::vector<uint8_t> bitPattern;
    bitPattern.reserve(numOfRepeat * windowSize);
    for (auto i = 0; i < numOfRepeat; i++) {
        bitPattern.insert(bitPattern.end(), window.begin(), window.end());
    }

    return bitPattern;
}

constexpr std::int32_t SPARSITY_PTR_WHEN_NO_SPARISTY = 0xFFFFFF;

std::int32_t toHex(double realVal) {
    union f32toint32 {
        std::int32_t m_i32;
        float m_f32;
    };

    f32toint32 biasVal;
    biasVal.m_f32 = static_cast<float>(realVal);
    return biasVal.m_i32;
}

void computeQuantMultShift(double scale, uint32_t& shift, uint32_t& mult) {
    const static int32_t BITS = 15;
    int32_t exponent = 0;

    const double mantissa = std::frexp(scale, &exponent);
    shift = BITS - exponent;
    mult = static_cast<std::uint32_t>((mantissa * pow(2, BITS)));
}

constexpr std::int32_t getKMBScale(unsigned shift, unsigned mult, double, mlir::Type) {
    // FIXME: set value when PPE is LPRELU in quant mode
    int32_t PRELU_SCALE_OFFSET = 0;
    int32_t PRELU_SCALE_VALUE = 1;

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t ROUND_MODE_OFFSET = 14;
    int32_t ROUND_MODE_VALUE = 1;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
           (ROUND_MODE_VALUE << ROUND_MODE_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
}

std::int32_t getMTLScale(unsigned shift, unsigned mult, double rescale, mlir::Type inputType) {
    // MTL expects scale in IEEE754 format in NCE_DPU_PPE_FP_SCALE register in case input has FP16/BF16 type
    if (inputType.isF16() || inputType.isF32()) {
        return toHex(rescale);
    }
    int32_t PRELU_SCALE_OFFSET = 0;
    int32_t PRELU_SCALE_VALUE = 0;

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t ROUND_MODE_OFFSET = 14;
    int32_t ROUND_MODE_VALUE = 1;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
           (ROUND_MODE_VALUE << ROUND_MODE_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
}

llvm::unique_function<int32_t(size_t)> getBiasFunc(mlir::Type op_inElemType, mlir::Type op_outElemType,
                                                   mlir::Type weightsElemType, mlir::Value bias,
                                                   vpux::VPUIP::ArchKind arch, size_t OC) {
    if (bias == nullptr) {
        return [](int64_t) -> double {
            return 0.0f;
        };
    }

    auto biasConst = bias.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(biasConst != nullptr, "Only constant biases are supported, got '{0}'", bias);

    auto biasContent = biasConst.content();

    if (op_inElemType.isa<mlir::quant::QuantizedType>() && op_outElemType.isa<mlir::quant::QuantizedType>()) {
        auto inQuantScale = extractScalesAndZeroPoints(op_inElemType).first;
        auto weightsQuantScales = exractWeightsScales(weightsElemType);

        broadcast(inQuantScale, OC);
        broadcast(weightsQuantScales, OC);

        std::vector<double> rescale(OC, 1.0);
        std::transform(weightsQuantScales.begin(), weightsQuantScales.end(), inQuantScale.begin(), rescale.begin(),
                       std::multiplies<>());

        return [biasContent = std::move(biasContent), rescale = std::move(rescale)](size_t oc) -> int32_t {
            const auto newBiasData =
                    checked_cast<int64_t>(std::round(biasContent.getValues<double>()[oc] / rescale[oc]));
            VPUX_THROW_UNLESS(newBiasData > std::numeric_limits<int32_t>::min() &&
                                      newBiasData < std::numeric_limits<int32_t>::max(),
                              "Bias value is out of range {0}", newBiasData);

            return static_cast<int32_t>(newBiasData);
        };
    } else if (!op_inElemType.isa<mlir::quant::QuantizedType>() && !op_outElemType.isa<mlir::quant::QuantizedType>()) {
        return [biasContent = std::move(biasContent), arch](int64_t oc) -> int32_t {
            const auto biasVal = biasContent.getValues<float>()[oc];
            const auto biasConverter = vpux::VPUIP::NCESparsity::biasConvertersMap.at(arch);
            return biasConverter(biasVal);
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. Both types must be quantized or not quantized. Got: in "
               "type {0}, "
               "out type {1}",
               op_inElemType, op_outElemType);
}

llvm::unique_function<int32_t(size_t)> getMultShiftFunc(mlir::Type op_inElemType, mlir::Type op_outElemType,
                                                        mlir::Type weightsType, vpux::VPUIP::ArchKind arch, size_t OC) {
    if (!op_inElemType.isa<mlir::quant::QuantizedType>() && !op_outElemType.isa<mlir::quant::QuantizedType>()) {
        const auto ppeConverter = vpux::VPUIP::NCESparsity::ppeConvertersMap.at(arch);
        const int32_t multShift = ppeConverter(0, 1, 1, op_inElemType);
        return [multShift](size_t) {
            return multShift;
        };
    } else if ((op_inElemType.isa<mlir::quant::QuantizedType>() && op_outElemType.isa<mlir::quant::QuantizedType>()) ||
               isMixedPrecisionSupported(arch)) {
        auto inQuantScale = op_inElemType.isa<mlir::quant::QuantizedType>()
                                    ? extractScalesAndZeroPoints(op_inElemType).first
                                    : SmallVector<double>{1.0};
        auto outQuantScale = op_outElemType.isa<mlir::quant::QuantizedType>()
                                     ? extractScalesAndZeroPoints(op_outElemType).first
                                     : SmallVector<double>{1.0};
        auto weightsQuantScales = exractWeightsScales(weightsType);

        broadcast(inQuantScale, OC);
        broadcast(outQuantScale, OC);
        broadcast(weightsQuantScales, OC);

        std::vector<double> rescale(OC, 1.0);
        for (size_t i = 0; i < rescale.size(); i++) {
            rescale[i] = (weightsQuantScales[i] * inQuantScale[i]) / outQuantScale[i];
        }

        const auto ppeConverter = vpux::VPUIP::NCESparsity::ppeConvertersMap.at(arch);
        return [rescale = std::move(rescale), ppeConverter, op_inElemType](size_t oc) {
            unsigned shift = 0;
            unsigned mult = 0;
            computeQuantMultShift(rescale[oc], shift, mult);
            return ppeConverter(shift, mult, rescale[oc], op_inElemType);
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. Both types must be quantized or not quantized. Got: in "
               "type {0}, "
               "out type {1}",
               op_inElemType, op_outElemType);
}

}  // namespace

const vpux::EnumMap<vpux::VPUIP::ArchKind, vpux::VPUIP::NCESparsity::PPEConverterCb>
        vpux::VPUIP::NCESparsity::ppeConvertersMap = {
                {vpux::VPUIP::ArchKind::KMB, getKMBScale},
                {vpux::VPUIP::ArchKind::TBH, getKMBScale},
                {vpux::VPUIP::ArchKind::MTL, getMTLScale},
};

const vpux::EnumMap<vpux::VPUIP::ArchKind, vpux::VPUIP::NCESparsity::BiasConverterCb>
        vpux::VPUIP::NCESparsity::biasConvertersMap = {
                {vpux::VPUIP::ArchKind::KMB, vpux::toFixedPoint},
                {vpux::VPUIP::ArchKind::TBH, vpux::toFixedPoint},
                {vpux::VPUIP::ArchKind::MTL, toHex},
};

int64_t vpux::VPUIP::NCESparsity::getBitPatternSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                    mlir::Type elemType) {
    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());

    auto actualType = tryGetQuantizedStorageType(elemType);
    const auto windowSize = getWindowSize(kernelSize[0], strideW, actualType);
    return kernelSize[1] * windowSize;
}

int64_t vpux::VPUIP::NCESparsity::getActivationWindowSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                          mlir::Type elemType, int64_t inputChannels) {
    auto actualType = tryGetQuantizedStorageType(elemType);
    const auto bitPatternSize = getBitPatternSize(kernelSize, strideW, actualType);
    const auto perChannelSparsitySize = static_cast<std::size_t>(std::ceil(bitPatternSize / 128.0) * 16);
    const auto activationWindowSize = inputChannels * perChannelSparsitySize;

    return activationWindowSize;
}

std::vector<uint8_t> vpux::VPUIP::NCESparsity::getFakeSparsity(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                               mlir::Type elemType, int64_t inputChannels) {
    auto actualType = tryGetQuantizedStorageType(elemType);
    const auto windowSize = getWindowSize(kernelSize[0], strideW, actualType);
    const auto bitPattern = getBitPattern(kernelSize, windowSize);

    // To align each activation map entry to 16 bytes to abide the hw restriction
    const auto perChannelSparsitySize = static_cast<std::size_t>(std::ceil(bitPattern.size() / 128.0) * 16);

    // MaxPool is supported only in depth wise mode.
    // Depth wise does not support weights sparsity in the real sense,
    // but it will have to have an activation window pointer,
    // which is regarded as "fake sparsity"
    SmallVector<uint8_t> perChannelSparsity;
    perChannelSparsity.resize(perChannelSparsitySize);

    // Repackaging each byte from bitPattern to a bit from fakeSparsity
    // The rest of the bits remain zero
    for (size_t i = 0; i < bitPattern.size(); i++) {
        perChannelSparsity[(i / 128) * 16 + (i % 128) / 8] |= bitPattern[i] << (i % 8);
    }

    std::vector<uint8_t> fakeSparsity;
    fakeSparsity.reserve(inputChannels * perChannelSparsitySize);
    for (auto i = 0; i < inputChannels; i++) {
        fakeSparsity.insert(fakeSparsity.end(), perChannelSparsity.begin(), perChannelSparsity.end());
    }

    return fakeSparsity;
}

std::vector<int32_t> vpux::VPUIP::NCESparsity::getWeightsTable(mlir::Type op_inElemType, mlir::Type op_outElemType,
                                                               Optional<int32_t> weightPtrOffset, int32_t weightPtrStep,
                                                               Optional<int32_t> sparsityPtrOffset,
                                                               vpux::VPUIP::ArchKind arch, int64_t OC,
                                                               mlir::Type weightsElemType, mlir::Value bias) {
    VPUX_THROW_WHEN(op_inElemType == nullptr || op_outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    auto getMultShift =
            getMultShiftFunc(op_inElemType, op_outElemType, weightsElemType, arch, checked_cast<size_t>(OC));
    auto getBiasFP = getBiasFunc(op_inElemType, op_outElemType, weightsElemType, bias, arch, checked_cast<size_t>(OC));

    // In case of dense operation use sparsityPtr beyond CMX memory range to satisfy HW requirements
    auto sparsityPtr = sparsityPtrOffset.hasValue() ? sparsityPtrOffset.getValue() : SPARSITY_PTR_WHEN_NO_SPARISTY;
    if (weightsElemType == nullptr && arch == vpux::VPUIP::ArchKind::MTL) {
        sparsityPtr = SPARSITY_PTR_WHEN_NO_SPARISTY;
    }
    auto weightPtr = weightPtrOffset.hasValue() ? weightPtrOffset.getValue() : 0;

    std::vector<std::int32_t> weightsTableVals(OC * vpux::VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    for (auto oc : irange(checked_cast<std::size_t>(OC))) {
        const auto wtInd = oc * static_cast<std::size_t>(vpux::VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd + 0] = weightPtr;
        weightsTableVals[wtInd + 1] = sparsityPtr;
        weightsTableVals[wtInd + 2] = getMultShift(oc);
        weightsTableVals[wtInd + 3] = getBiasFP(oc);

        weightPtr += weightPtrStep;
        if (arch == vpux::VPUIP::ArchKind::MTL && weightsElemType) {
            auto elementSize = static_cast<Byte>(getElemTypeSize(weightsElemType));
            sparsityPtr += weightPtrStep / static_cast<int>(elementSize.count());
        }
    }

    return weightsTableVals;
}
