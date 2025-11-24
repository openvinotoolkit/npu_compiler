//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/blob_writer.hpp"

#include <npu_37xx_nnrt.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "Fp16Convert.h"

using namespace vpux;
using namespace npu37xx;

using VPUMI37XX::createDims;
using VPUMI37XX::createStrides;
using VPUMI37XX::createVector;

namespace MVCNN = vpux::VPUMI37XX::MVCNN;

// hard copied stuff from VPUIP_2

namespace {

std::optional<MVCNN::TensorReference> createTensorRef(mlir::Value val, ArrayRef<int64_t> sectionIndex,
                                                      int64_t byteOffset, std::optional<int64_t> sparsityMapOffset,
                                                      std::optional<int64_t> storageElementOffset,
                                                      std::optional<int64_t> storageElementSize,
                                                      std::optional<int64_t> swizzlingKey) {
    return VPUMI37XX::createTensorRef(mlir::cast<vpux::NDTypeInterface>(val.getType()), sectionIndex, byteOffset,
                                      sparsityMapOffset, storageElementOffset, storageElementSize, swizzlingKey);
}

std::optional<MVCNN::TensorReference> createTensorRef(VPURT::DeclareBufferOp bufOp,
                                                      std::optional<int64_t> sparsityMapOffset,
                                                      std::optional<int64_t> storageElementOffset,
                                                      std::optional<int64_t> storageElementSize) {
    auto sectionIndex = bufOp.getNonEmptySectionIndex();
    return createTensorRef(bufOp.getBuffer(), sectionIndex, bufOp.getByteOffset(), sparsityMapOffset,
                           storageElementOffset, storageElementSize, bufOp.getSwizzlingKey());
}

std::tuple<std::optional<MVCNN::TensorReference>, std::optional<MVCNN::TensorReference>,
           std::optional<MVCNN::TensorReference>>
createTensorRef(mlir::Value input, mlir::Value sparsityMap = nullptr, mlir::Value storageElementTable = nullptr,
                std::optional<int64_t> storageElementSize = std::nullopt) {
    if (input == nullptr) {
        return {};
    }

    auto parent = input.getDefiningOp();
    VPUX_THROW_WHEN(parent == nullptr, "Can't create TensorRef from I/O arg");

    std::optional<MVCNN::TensorReference> sparsityMapTensorRef;
    std::optional<MVCNN::TensorReference> storageElementTableRef;

    std::optional<int64_t> sparsityMapOffset = std::nullopt;
    std::optional<int64_t> storageElementOffset = std::nullopt;

    if (sparsityMap) {
        auto sparseBufferOp = sparsityMap.getDefiningOp<VPURT::DeclareBufferOp>();
        sparsityMapOffset = sparseBufferOp.getByteOffset();

        sparsityMapTensorRef = createTensorRef(sparseBufferOp, std::nullopt, std::nullopt, std::nullopt);

        if (storageElementTable) {
            auto seTableBufferOp = storageElementTable.getDefiningOp<VPURT::DeclareBufferOp>();
            storageElementOffset = seTableBufferOp.getByteOffset();

            storageElementTableRef = createTensorRef(seTableBufferOp, std::nullopt, std::nullopt, std::nullopt);
        }
    }

    if (auto bufOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(parent)) {
        auto bufTensorRef = createTensorRef(bufOp, sparsityMapOffset, storageElementOffset, storageElementSize);
        return std::make_tuple(bufTensorRef, sparsityMapTensorRef, storageElementTableRef);
    }

    VPUX_THROW("Unknown buffer declaration");
}

std::optional<MVCNN::TensorReference> createTensorRef(vpux::NDTypeInterface type, VPURT::DeclareBufferOp bufferOp,
                                                      ArrayRef<int64_t> ppeQuantMult, ArrayRef<int64_t> ppeQuantShift,
                                                      ArrayRef<uint8_t> quantZeroPoints,
                                                      std::optional<int64_t> sparsityMapOffset,
                                                      std::optional<int64_t> storageElementOffset,
                                                      std::optional<int64_t> storageElementSize) {
    return VPUMI37XX::createTensorRef(type, bufferOp.getNonEmptySectionIndex(), bufferOp.getByteOffset(), ppeQuantMult,
                                      ppeQuantShift, quantZeroPoints, sparsityMapOffset, storageElementOffset,
                                      storageElementSize, bufferOp.getSwizzlingKey());
}

// This is a helper routine to build new TensorReference out of NCE task for input, weights and output with provided
// quantization scale parameters
std::optional<MVCNN::TensorReference> getTensorReferenceWithUpdatedQuantParams(
        ArrayRef<int64_t> ppeQuantMult, ArrayRef<int64_t> ppeQuantShift, mlir::Value tensor,
        std::optional<int64_t> sparsityMapOffset, std::optional<int64_t> storageElementOffset,
        std::optional<int64_t> storageElementSize) {
    // Get also ZP
    SmallVector<uint8_t> quantZeroPoints;

    auto type = mlir::cast<vpux::NDTypeInterface>(tensor.getType());

    auto elementType = type.getElementType();
    if (const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType =
                       mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    VPUX_THROW_UNLESS(ppeQuantShift.size() == quantZeroPoints.size(),
                      "Mismatch of size between quant shift/mult vector and quant ZP:  {0} != {1}",
                      ppeQuantShift.size(), quantZeroPoints.size());

    // Find corresponding DeclareBufferOp to get all the data needed to build new TensorReference
    auto bufferOp = tensor.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Unable to find parent DeclareBufferOp to build new TensorReference");

    auto sectionIndex = bufferOp.getNonEmptySectionIndex();
    return createTensorRef(type, bufferOp, ppeQuantMult, ppeQuantShift, quantZeroPoints, sparsityMapOffset,
                           storageElementOffset, storageElementSize);
}

}  // namespace

namespace math {
template <typename T>
unsigned int count(T value) {
    unsigned int bits = 0;

    for (; value > 0; value >>= 1) {
        if (value & 1) {
            ++bits;
        }
    }

    return bits;
}
}  // namespace math

constexpr unsigned long long DEFAULT_INDEX = 999999999999999999;

constexpr uint32_t STRIDES(int dim) {
    return dim + 1;
}

enum {
    B,
    Z,
    Y,
    X,
};

enum {
    KERNEL_SIZE_MIN = 1,
    KERNEL_SIZE_MAX = 11,
    KERNEL_STRIDE_MIN = 1,
    KERNEL_STRIDE_MAX = 8,
};

enum MPEGrid { MPE_GRID_4x4, MPE_GRID_16x1 };

// PPE activation function choices in 2p7
enum activationFunction_t { no_activation_function, relu, relu_x, leaky_relu, unsupported };

typedef float fp32;
typedef union {
    uint32_t u32;
    fp32 f32;
} u32f32;

struct activationFunctionDesc {
    float alpha;
    u32f32 alphaFP32;
    uint32_t alphaMult;   // Mult Register value
    uint32_t alphaShift;  // Shift register value (number of bits to shift left by)
    activationFunction_t funcType;
    int32_t clampLow;
    int32_t clampHigh;

    activationFunctionDesc()
            : alpha(1.0), alphaMult(0), alphaShift(1), funcType(no_activation_function), clampLow(0), clampHigh(0) {
        alphaFP32.u32 = 0;
    }
};

uint8_t ConfigDtype(const MVCNN::DType dtype) {
    auto inputType = nn_public::VpuInputTensorDType::INPUT_DTYPE_UNKNOWN;

    switch (dtype) {
    case MVCNN::DType::DType_FP16:
        inputType = nn_public::VpuInputTensorDType::FP16;
        break;
    case MVCNN::DType::DType_U8:
        inputType = nn_public::VpuInputTensorDType::U8;
        break;
    case MVCNN::DType::DType_I8:
        inputType = nn_public::VpuInputTensorDType::I8;
        break;
    case MVCNN::DType::DType_U4:
        inputType = nn_public::VpuInputTensorDType::U4;
        break;
    case MVCNN::DType::DType_I4:
        inputType = nn_public::VpuInputTensorDType::I4;
        break;
    case MVCNN::DType::DType_I2:
        inputType = nn_public::VpuInputTensorDType::I2;
        break;
    case MVCNN::DType::DType_BFP16:
        inputType = nn_public::VpuInputTensorDType::BF16;
        break;
    case MVCNN::DType::DType_FP8:
        inputType = nn_public::VpuInputTensorDType::FP8;
        break;
    case MVCNN::DType::DType_BIN:
        inputType = nn_public::VpuInputTensorDType::BIN;
        break;
    default:
        VPUX_THROW("Invalid input data type {0}", dtype);
        break;
    }

    return static_cast<uint8_t>(inputType);
}

uint8_t ConfigOutputDtype(const MVCNN::DType dtype) {
    auto otype = nn_public::VpuOutputTensorDType::OUTPUT_DTYPE_UNKNOWN;

    switch (dtype) {
    case MVCNN::DType::DType_FP16:
        otype = nn_public::VpuOutputTensorDType::FP16;
        break;
    case MVCNN::DType::DType_FP32:
        otype = nn_public::VpuOutputTensorDType::FP32;
        break;
    case MVCNN::DType::DType_BFP16:
        otype = nn_public::VpuOutputTensorDType::FP16;
        break;  // Difference is in PPE settings
    case MVCNN::DType::DType_U8:
        otype = nn_public::VpuOutputTensorDType::G8;
        break;
    case MVCNN::DType::DType_I8:
        otype = nn_public::VpuOutputTensorDType::I8;
        break;
    case MVCNN::DType::DType_I32:
        otype = nn_public::VpuOutputTensorDType::I32;
        break;
    case MVCNN::DType::DType_U4:
        otype = nn_public::VpuOutputTensorDType::U4;
        break;
    case MVCNN::DType::DType_I4:
        otype = nn_public::VpuOutputTensorDType::I4;
        break;
    case MVCNN::DType::DType_I2:
        otype = nn_public::VpuOutputTensorDType::I2;
        break;
    case MVCNN::DType::DType_BIN:
        otype = nn_public::VpuOutputTensorDType::BIN;
        break;
    case MVCNN::DType::DType_LOG:
        otype = nn_public::VpuOutputTensorDType::LOG;
        break;
    default:
        VPUX_THROW("Invalid output data type");
        break;
    }

    return static_cast<uint8_t>(otype);
}

MVCNN::PPELayerType getPPELayerType(VPU::PPEMode ppeType) {
    switch (ppeType) {
    case VPU::PPEMode::STORE:
        return MVCNN::PPELayerType_STORE;
    case VPU::PPEMode::LOAD:
        return MVCNN::PPELayerType_LOAD;
    case VPU::PPEMode::CLEAR:
        return MVCNN::PPELayerType_CLEAR;
    case VPU::PPEMode::NOOP:
        return MVCNN::PPELayerType_NOOP;
    case VPU::PPEMode::HALT:
        return MVCNN::PPELayerType_HALT;
    case VPU::PPEMode::ADD:
        return MVCNN::PPELayerType_ADD;
    case VPU::PPEMode::SUB:
        return MVCNN::PPELayerType_SUB;
    case VPU::PPEMode::MULT:
        return MVCNN::PPELayerType_MULT;
    case VPU::PPEMode::MAXIMUM:
        return MVCNN::PPELayerType_MAXIMUM;
    case VPU::PPEMode::MINIMUM:
        return MVCNN::PPELayerType_MINIMUM;
    case VPU::PPEMode::AND:
        return MVCNN::PPELayerType_AND;
    case VPU::PPEMode::OR:
        return MVCNN::PPELayerType_OR;
    case VPU::PPEMode::XOR:
        return MVCNN::PPELayerType_XOR;
    case VPU::PPEMode::LRELU:
        return MVCNN::PPELayerType_LRELU;
    case VPU::PPEMode::LRELUX:
        return MVCNN::PPELayerType_LRELUX;
    case VPU::PPEMode::LPRELU:
        return MVCNN::PPELayerType_LPRELU;
    case VPU::PPEMode::CEIL:
        return MVCNN::PPELayerType_CEIL;
    case VPU::PPEMode::FLOOR:
        return MVCNN::PPELayerType_FLOOR;
    case VPU::PPEMode::EXP:
        return MVCNN::PPELayerType_EXP;
    case VPU::PPEMode::SIGMOID:
        return MVCNN::PPELayerType_SIGMOID;
    case VPU::PPEMode::TANH:
        return MVCNN::PPELayerType_TANH;
    case VPU::PPEMode::SQRT:
        return MVCNN::PPELayerType_SQRT;
    case VPU::PPEMode::RSQRT:
        return MVCNN::PPELayerType_RSQRT;
    case VPU::PPEMode::FLEXARB:
        return MVCNN::PPELayerType_FLEXARB;
    case VPU::PPEMode::NOT:
        return MVCNN::PPELayerType_NOT;
    case VPU::PPEMode::ABS:
        return MVCNN::PPELayerType_ABS;
    case VPU::PPEMode::NEG:
        return MVCNN::PPELayerType_NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

MVCNN::Permutation getODUPermutationType(DimsOrder outputDimsOrder) {
    if (outputDimsOrder == vpux::DimsOrder::NHWC) {
        return MVCNN::Permutation::Permutation_ZXY;
    } else if (outputDimsOrder == DimsOrder::fromCode(0x1432)) {  // NWHC
        return MVCNN::Permutation::Permutation_ZYX;
    } else if (outputDimsOrder == DimsOrder::fromCode(0x1423)) {  // NWCH
        return MVCNN::Permutation::Permutation_YZX;
    } else if (outputDimsOrder == DimsOrder::fromCode(0x1243)) {  // NCWH
        return MVCNN::Permutation::Permutation_YXZ;
    } else if (outputDimsOrder == vpux::DimsOrder::NHCW) {
        return MVCNN::Permutation::Permutation_XZY;
    } else if (outputDimsOrder == vpux::DimsOrder::NCHW) {
        return MVCNN::Permutation::Permutation_XYZ;
    } else {
        VPUX_THROW("Can't get ODU permutation by output dimsOrder: '{0}'", outputDimsOrder);
    }
}

static bool areSupportedInputOutputTypes(MVCNN::DType in_type, MVCNN::DType out_type) {
    bool in_supported = (in_type == MVCNN::DType::DType_BFP16) || (in_type == MVCNN::DType::DType_FP8) ||
                        (in_type == MVCNN::DType::DType_U8) || (in_type == MVCNN::DType::DType_I8) ||
                        (in_type == MVCNN::DType::DType_U4) || (in_type == MVCNN::DType::DType_I4) ||
                        (in_type == MVCNN::DType::DType_FP16);

    bool out_supported = (out_type == MVCNN::DType::DType_BFP16) || (out_type == MVCNN::DType::DType_FP8) ||
                         (out_type == MVCNN::DType::DType_U8) || (out_type == MVCNN::DType::DType_I8) ||
                         (out_type == MVCNN::DType::DType_I32) || (out_type == MVCNN::DType::DType_I4) ||
                         (out_type == MVCNN::DType::DType_FP16) || (out_type == MVCNN::DType::DType_FP32) ||
                         (out_type == MVCNN::DType::DType_U4);

    // Currently only support the following input & output types:
    if (!in_supported || !out_supported) {
        VPUX_THROW("Unsupported data type for PPE. In {0} Out {1}", in_type, out_type);
        return false;
    }

    return true;
}

// from register definition of NCE_DPU_PPE_PRELU
// 26:16 ppe_prelu_mult    prelu multiplier (u11)
// 12:8  ppe_prelu_shift   prelu shift (u5)
#define LRELU_MULT_MASK 0x7FF
#define LRELU_MULT_ERR 0x800
#define LRELU_SHIFT_MASK 0x1F
#define LRELU_SHIFT_ERR 0xFF

#define PACK_B16(x, y, z) ((x << 15) + (y << 7) + (z))

// Constants for converting to BF16
constexpr uint32_t fp32FracBits = 23;
constexpr uint32_t fp16ExpBias = 15;
constexpr uint32_t fp16FracBits = 10;
constexpr uint32_t bf16FracBits = 7;
constexpr uint32_t bf16NanOutput = 0x7FC0;
constexpr uint32_t fp32NanOutput = 0x7FC00000;  // Aligns with Synopsys DWC_FP_MULT fixed NAN output

static float readReluMult(const std::optional<MVCNN::PPEFixedFunction>& ff) {
    int32_t mult = ff->Lrelu_Mult();
    bool is_negative = false;
    // Currently gcc is having trouble converting a negative int32_t into a float, the result is equal with the
    // unsigned integer into float. To workaround this, until we understand if this is an expected behavior or not, we
    // convert to float, and then we add the sign, if the value was negative
    // Bug tracked in E#30236
    if (mult < 0) {
        mult = -mult;
        is_negative = true;
    }
    if (mult & ~LRELU_MULT_MASK) {
        return LRELU_MULT_ERR;
    }
    if (is_negative == false) {
        return (float)mult;
    } else {
        return -(float)mult;
    }
}

static int32_t readReluShift(const std::optional<MVCNN::PPEFixedFunction>& ff) {
    uint32_t shift = (uint32_t)ff->Lrelu_Shift();
    if (shift & ~LRELU_SHIFT_MASK) {
        return LRELU_SHIFT_ERR;
    }
    return (int32_t)ff->Lrelu_Shift();
}

static bool setupActivationFunction(const std::optional<MVCNN::PPETask>& ppe_task,
                                    const std::optional<MVCNN::TensorReference>& inputRef,
                                    activationFunctionDesc& actFuncDesc) {
    if (ppe_task && ppe_task->fixed_function()) {
        auto ff = ppe_task->fixed_function();
        if (ff->Ops()->size() == 0) {
            // If Ops.size is 0 or PPELayerType == NOOP, no activation function will be used.
            actFuncDesc.funcType = no_activation_function;
        } else if (ff->Ops()->size() == 1) {
            switch (ff->Ops()->Get(0)) {
            case MVCNN::PPELayerType_LRELU:
                actFuncDesc.funcType = relu;
                actFuncDesc.alphaFP32.f32 = -0.0;  // note: -0.0, to ensure zero-gained data uses positive zero in
                                                   // FP32 (0x00000000), not negative zero (0x80000000)
                break;
            case MVCNN::PPELayerType_LRELUX:
                actFuncDesc.funcType = relu_x;
                actFuncDesc.alphaFP32.f32 = -0.0;  // note: -0.0, to ensure zero-gained data uses positive zero in
                                                   // FP32 (0x00000000), not negative zero (0x80000000)
                break;
            case MVCNN::PPELayerType_LPRELU: {
                // LeakyReLU: alpha slope derived according to Alessandro's Fathom test script as follows (ca. line
                // 87:)
                // https://github.com/movidius/Fathom/blob/master/scripts/validation_test_script/mix_precision_blobs.py
                // scale_shift_to_fp(scale,shift): scale * 2 ** (-float(shift))
                // scale_shift_to_fp(ppe_ops["Lrelu_Mult"], ppe_ops["Lrelu_Shift"])
                actFuncDesc.funcType = leaky_relu;
                actFuncDesc.alphaFP32.f32 = ppe_task->fp_prelu_alpha();
                if (inputRef->data_dtype() == MVCNN::DType::DType_U8 ||
                    inputRef->data_dtype() == MVCNN::DType::DType_I8 ||
                    inputRef->data_dtype() == MVCNN::DType::DType_I32) {
                    float lReluMult = readReluMult(ff);
                    int32_t lReluShift = readReluShift(ff);
                    // The compiler is supposed to provide values in range, if either of these report an error,
                    // reject the blob.
                    if (lReluMult == LRELU_MULT_ERR || lReluShift == LRELU_SHIFT_ERR) {
                        return false;
                    }
                    actFuncDesc.alphaMult = static_cast<uint32_t>(lReluMult);
                    actFuncDesc.alphaShift = lReluShift;
                }
                break;
            }
            case MVCNN::PPELayerType_NOOP:
            default:
                actFuncDesc.funcType = no_activation_function;
                break;
            }
        } else {
            return false;
        }

        actFuncDesc.clampHigh = ff->Clamp_High();
        actFuncDesc.clampLow = ff->Clamp_Low();
    }

    return true;
}

static void setupQuantInQuantOut(nn_public::VpuDPUInvariantRegisters& regs, const activationFunctionDesc& actFuncDesc,
                                 const uint8_t out_zero_point, const int32_t lrelu_mult, const int32_t lrelu_shift) {
    // I8/U8/I4/U4 in, INT32 convolution, I8/U8/I4/U4 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000;  // INT32 convolution -> bypass FP clamp/gain
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    regs.ppe_fp_prelu = 0;

    if (actFuncDesc.funcType == leaky_relu) {
        // in this case, we have to convert a high-precision floating-point
        // LeakyReLU alpha value to integer multiply and shift register values
        if (lrelu_mult == 1 && lrelu_shift == 0) {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = actFuncDesc.alphaMult;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = actFuncDesc.alphaShift;
        } else {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = lrelu_mult;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = lrelu_shift;
        }
    } else if (actFuncDesc.funcType == relu_x) {
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0;  // ReLU zero negative slope
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
    } else if (actFuncDesc.funcType == relu) {
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0;  // ReLU zero negative slope
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
    } else {
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;   // no activation function
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;  // no activation function
    }

    // U8 Quantization logic requires a final addition of the zero point
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = out_zero_point;
}

static void setupQuantInFloatOut(nn_public::VpuDPUInvariantRegisters& regs, const activationFunctionDesc& actFuncDesc,
                                 const MVCNN::DType& out_type, const int32_t lrelu_mult, const int32_t lrelu_shift) {
    // U8 in, INT32 convolution, FP16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0;  // INT32 convolution -> bypass FP clamp/gain
    regs.ppe_misc.ppe_misc_bf.ppe_i32_convert =
            (out_type == MVCNN::DType::DType_FP8) ? 0x2 : 0x1;  // INT32 s17.15 fixed-point convert to FP8/FP16

    if (actFuncDesc.funcType == leaky_relu) {
        // in this case, we have to convert a high-precision floating-point
        // LeakyReLU alpha value to integer multiply and shift register values
        if (lrelu_mult == 1 && lrelu_shift == 0) {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = actFuncDesc.alphaMult;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = actFuncDesc.alphaShift;
        } else {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = lrelu_mult;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = lrelu_shift;
        }
    } else if (actFuncDesc.funcType == relu_x) {
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0;  // ReLU zero negative slope
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
    } else if (actFuncDesc.funcType == relu) {
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0;  // ReLU zero negative slope
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
    } else {
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;   // no activation function
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;  // no activation function
    }
}

static bool setupBFloat(MVCNN::DType in_dtype, nn_public::VpuDPUInvariantRegisters& regs,
                        const activationFunctionDesc& actFunc) {
    if (in_dtype == MVCNN::DType::DType_I8 || in_dtype == MVCNN::DType::DType_U8) {
        VPUX_THROW("X8 in, I32 convolution, BF16 out is not supported by the hardware");
        return false;
    }

    // FP8/FP16/BF16 in, FP32 convolution, BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x002;  // FP32 convolution -> BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 1;      // Round to Nearest, Ties to Even (RNE)

    // FP32 Prelu
    if ((actFunc.funcType == leaky_relu) || (actFunc.funcType == relu) || (actFunc.funcType == relu_x)) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
        regs.ppe_fp_prelu = actFunc.alphaFP32.u32;  // deliberately apply gain of zero to values less than zero
    } else {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    }

    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

    // Do not apply the scaling table to the integer PPE
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;

    return true;
}

static bool setupInt(MVCNN::DType in_type, MVCNN::DType out_type, nn_public::VpuDPUInvariantRegisters& regs,
                     activationFunctionDesc& actFuncDesc, uint8_t out_zero_point, int32_t lrelu_mult,
                     int32_t lrelu_shift) {
    switch (out_type) {
    case MVCNN::DType::DType_I8:
    case MVCNN::DType::DType_I4:
    case MVCNN::DType::DType_U4:
    case MVCNN::DType::DType_U8:
    case MVCNN::DType::DType_I32:
        setupQuantInQuantOut(regs, actFuncDesc, out_zero_point, lrelu_mult, lrelu_shift);
        break;
    case MVCNN::DType::DType_FP16:
    case MVCNN::DType::DType_FP8:
        setupQuantInFloatOut(regs, actFuncDesc, out_type, lrelu_mult, lrelu_shift);
        break;
    case MVCNN::DType::DType_BFP16:
        if (!setupBFloat(in_type, regs, actFuncDesc)) {
            return false;
        }
        break;
    default:
        VPUX_THROW("Unexpected dtype");
        return false;
    }

    return true;
}

static void setupFloatInQuantOut(nn_public::VpuDPUInvariantRegisters& regs, const activationFunctionDesc& actFuncDesc,
                                 const uint8_t out_zero_point) {
    // FP16/BF16/FP8 in, FP32 convolution, U8 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x004;  // FP32 convolution -> INT32 (and eventually U8) out

    // Derive fp _prelu
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

    // Scale override must not be required for the float input with quantized output.
    // Only operations without weights tables should require that override.
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    regs.ppe_bias = 0;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;  // can be overridden by LeakyReLU case

    // FP32 prelu
    if ((actFuncDesc.funcType == leaky_relu) || (actFuncDesc.funcType == relu) || (actFuncDesc.funcType == relu_x)) {
        // for LeakyReLU, apply alpha; for ReLU and ReLUX, apply a negative-X slope of 0
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
        regs.ppe_fp_prelu = actFuncDesc.alphaFP32.u32;
    }

    // U8 Quantization logic requires a final addition of the zero point
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = out_zero_point;
}

static void setupFloatInFloatOut(nn_public::VpuDPUInvariantRegisters& regs, const activationFunctionDesc& actFuncDesc,
                                 const MVCNN::DType& out_type) {
    // FP16 in, FP32 convolution, FP16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    if (out_type != MVCNN::DType::DType_FP32) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                (out_type == MVCNN::DType::DType_FP8) ? 0x003 : 0x001;  // FP32 convolution -> FP8/FP16 out
    }

    // FP32 Prelu
    if ((actFuncDesc.funcType == leaky_relu) || (actFuncDesc.funcType == relu) || (actFuncDesc.funcType == relu_x)) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
        regs.ppe_fp_prelu = actFuncDesc.alphaFP32.u32;  // deliberately apply gain of zero to values less than zero
    } else {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    }

    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

    // Do not apply the scaling table to the integer PPE
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
}

// FP8/FP16/FP32 out
static bool setupFloat(MVCNN::DType in_type, MVCNN::DType out_type, nn_public::VpuDPUInvariantRegisters& regs,
                       activationFunctionDesc& actFuncDesc, const uint8_t out_zero_point) {
    switch (out_type) {
    case MVCNN::DType::DType_I8:
    case MVCNN::DType::DType_I4:
    case MVCNN::DType::DType_U4:
    case MVCNN::DType::DType_U8:
    case MVCNN::DType::DType_I32:
        setupFloatInQuantOut(regs, actFuncDesc, out_zero_point);
        break;
    case MVCNN::DType::DType_FP32:
    case MVCNN::DType::DType_FP16:
    case MVCNN::DType::DType_FP8:
        setupFloatInFloatOut(regs, actFuncDesc, out_type);
        break;
    case MVCNN::DType::DType_BFP16:
        if (!setupBFloat(in_type, regs, actFuncDesc)) {
            return false;
        }
        break;
    default:
        VPUX_THROW("Unexpected dtype");
        return false;
    }

    return true;
}

inline uint8_t ConfigMpeActivationWeightDtype(uint8_t atype, uint8_t wtype) {
    // When the activations and weights are of different types,
    // MPE_MODE must be configured to the larger of the 2 data types.
    return std::min(atype, wtype);
}

void SetupInvariant_Grid(vpux::VPUMI37XX::DPUInvariantOp fb_invariant, nn_public::VpuDPUInvariantRegisters& invariant) {
    auto mpe_frequent_mode = fb_invariant.getMpeFrequentMode();
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (fb_invariant.getNceTaskType() == VPUIP::NCETaskType::ELTWISE) {
        mpe_frequent_mode = VPU::MPEMode::CUBOID_8x16;
    }
    // Sets up on NTHW on IDU
    switch (mpe_frequent_mode) {
    case VPU::MPEMode::CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        invariant.odu_cfg.odu_cfg_bf.grid = static_cast<unsigned int>(nn_public::VpuODUGrid::ODU_GRID_4x4);
        invariant.odu_cfg.odu_cfg_bf.nthw = static_cast<unsigned int>(nn_public::VpuODUNthw::ODU_NTHW_4);
        invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = nn_public::VpuMPEGrid::MPE_GRID_4x4;
        break;
    case VPU::MPEMode::CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    case VPU::MPEMode::VECTOR:       // nn_public::VpuMPE_GRID_16x1 not valid for VPUX37XX
        invariant.odu_cfg.odu_cfg_bf.grid = static_cast<unsigned int>(nn_public::VpuODUGrid::ODU_GRID_4x4);
        invariant.odu_cfg.odu_cfg_bf.nthw = static_cast<unsigned int>(nn_public::VpuODUNthw::ODU_NTHW_8);
        invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = nn_public::VpuMPEGrid::MPE_GRID_4x4;
        break;
    case VPU::MPEMode::CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        invariant.odu_cfg.odu_cfg_bf.grid = static_cast<unsigned int>(nn_public::VpuODUGrid::ODU_GRID_4x4);
        invariant.odu_cfg.odu_cfg_bf.nthw = static_cast<unsigned int>(nn_public::VpuODUNthw::ODU_NTHW_16);
        invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = nn_public::VpuMPEGrid::MPE_GRID_4x4;
        break;
    default:
        VPUX_THROW("invalid mpe_frequent_mode {0}", mpe_frequent_mode);
        break;
    }
}

unsigned int SOH_LinesPerCluster(unsigned int parentHeight, unsigned int height, unsigned int clusters) {
    unsigned int lines_per_cluster = (parentHeight + clusters - 1) / clusters;

    if (height < lines_per_cluster) {
        lines_per_cluster = (parentHeight - height) / (clusters - 1);
    } else {
        lines_per_cluster = height;
    }

    return lines_per_cluster;
}

unsigned int SetupVariant_SOH(vpux::VPUMI37XX::DPUInvariantOp fb_invariant, vpux::VPUMI37XX::DPUVariantOp fb_variant,
                              const std::optional<MVCNN::TensorReference>& out_tensor,
                              const std::optional<MVCNN::TensorReference>& parent_out_tensor,
                              nn_public::VpuDPUVariant& variant, unsigned int clusters) {
    auto workload_start = parseIntArrayAttr<int64_t>(fb_variant.getStartAttr());
    auto workload_end = parseIntArrayAttr<int64_t>(fb_variant.getEndAttr());

    auto output_start_y = static_cast<int16_t>(workload_start[1]);
    auto output_end_y = static_cast<int16_t>(workload_end[1]);

    if (out_tensor->dimensions()->Get(Y) != parent_out_tensor->dimensions()->Get(Y)) {
        unsigned int lines_per_cluster = SOH_LinesPerCluster(parent_out_tensor->dimensions()->Get(Y),
                                                             out_tensor->dimensions()->Get(Y), clusters);

        output_start_y %= lines_per_cluster;
        output_end_y %= lines_per_cluster;
        variant.registers_.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
        variant.registers_.te_end0.te_end0_bf.te_end_y = output_end_y;

        if (((unsigned)output_start_y > out_tensor->dimensions()->Get(Y)) ||
            ((unsigned)output_end_y > out_tensor->dimensions()->Get(Y))) {
            VPUX_THROW("SOH workload still too big: {0}-{1}, tensor dim_y {2}", output_start_y, output_end_y,
                       out_tensor->dimensions()->Get(Y));
        }

        // Workload start needs adjustment if SOH was not set in invariant
        // This should be a check for (invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment == 0)
        // We can't access that field from DPUVariantOp serialization
        // Logic breakdown concludes that we can check for segmentation to be off OR the task type to be ELTWISE
        // If the task type is ELTWISE sp_se_tbl_segment does not get set even if segmentation is ON
        if (!fb_invariant.getIsSegmented() || fb_invariant.getNceTaskType() == VPUIP::NCETaskType::ELTWISE) {
            int64_t kernelStridesH = 1;
            if (fb_invariant.getKernelStridesAttr() != nullptr) {
                const auto kernelStrides = parseIntArrayAttr<int64_t>(fb_invariant.getKernelStridesAttr());
                kernelStridesH = kernelStrides[0];
            }
            int64_t kernelPadTop = 0;
            if (fb_invariant.getKernelPaddingAttr() != nullptr) {
                kernelPadTop = checked_cast<int16_t>(fb_invariant.getKernelPaddingAttr().getTop().getInt());
            }

            int64_t variantPadTop = fb_variant.getPadAttr().getTop().getInt();

            auto stride_h = kernelStridesH;
            auto global_PT = kernelPadTop;
            auto local_PT = variantPadTop;
            variant.registers_.workload_start0.workload_start0_bf.workload_start_y =
                    (output_start_y * stride_h) - global_PT + local_PT;
        }

        // bool is_out_dense = fb_invariant->output_data()->data()->sparsity_index() == DEFAULT_INDEX;
        // if (!is_out_dense)
        //     variant.output_sparsity_offset_ |= fb_invariant->output_data()->locale_index()->Get(0) << 1;
    }

    return output_end_y - output_start_y + 1;
}

void Setup_Output_SOH(const std::optional<MVCNN::TensorReference>& outputRef,
                      const std::optional<MVCNN::TensorReference>& parentOutputRef,
                      nn_public::VpuDPUInvariantRegisters& invariant, bool is_out_dense) {
    // TODO: E#54009 remove these - copied directly from POC
    invariant.base_ptr_a = 0x1;
    invariant.base_ptr_b = 0x403;
    if (!is_out_dense && outputRef->dimensions()->Get(Y) != parentOutputRef->dimensions()->Get(Y)) {
        invariant.base_ptr_a = (0 << 9) | (1 << 0);
        invariant.base_ptr_b = (2 << 9) | (3 << 0);
    }
}

void SetupInput(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    // Parent input override is required for PermuteQuantize.
    // Needed because input dimensions are read from parent input
    const auto isPermuteQuantize = op.getIsPermuteQuantizeAttr() != nullptr;
    const auto overrideParentIn = isPermuteQuantize && (op.getParentInput().getType() != op.getInput().getType());
    auto parentInput = overrideParentIn ? op.getInput() : op.getParentInput();

    auto parentInputSparsityMap = op.getParentInputSparsityMap();
    auto parentInputSETables = op.getParentInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();

    auto [parentInputRef, parentInputSparsityMapRef, parentInputSETableRef] =
            createTensorRef(parentInput, parentInputSparsityMap, parentInputSETables, inputSESize);

    registers.tensor_size0.tensor_size0_bf.tensor_size_x = parentInputRef->dimensions()->Get(X);
    registers.tensor_size0.tensor_size0_bf.tensor_size_y = parentInputRef->dimensions()->Get(Y);
    registers.tensor_size1.tensor_size1_bf.tensor_size_z = parentInputRef->dimensions()->Get(Z);

    registers.z_config.z_config_bf.addr_format_sel = 1;

    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();

    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto amode = inputRef->data_dtype();
    auto dtype = ConfigDtype(amode);

    if (dtype == static_cast<uint8_t>(nn_public::VpuInputTensorDType::INPUT_DTYPE_UNKNOWN)) {
        VPUX_THROW("INVALID INPUT DTYPE");
    }

    registers.tensor_mode.tensor_mode_bf.amode = dtype;

    if (amode != MVCNN::DType::DType_FP16) {
        registers.mpe_cfg.mpe_cfg_bf.mpe_actbias =
                (inputRef->quant_zero() && inputRef->quant_zero()->size()) ? inputRef->quant_zero()->Get(0) : 0;
    }

    // pad_value is not set for VPUX37XX
    // registers.tensor_mode.tensor_mode_bf.pad_value =
    //         (inputRef->quant_zero() && inputRef->quant_zero()->size()) ? inputRef->quant_zero()->Get(0) : 0;

    bool is_act_dense = inputRef->data()->sparsity_index() == DEFAULT_INDEX;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = is_act_dense;
}

void SetupWeights(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    if (op.getNceTaskType() == vpux::VPUIP::NCETaskType::MAXPOOL) {
        registers.tensor_mode.tensor_mode_bf.wmode = static_cast<unsigned int>(nn_public::VpuInputTensorDType::I8);
        return;
    }

    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();

    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto weights = op.getWeights();
    auto weightsSparsityMap = op.getWeightsSparsityMap();

    auto [weightsRef, weightsSparsityMapRef, weightsSETableRef] =
            createTensorRef(weights, weightsSparsityMap, nullptr, std::nullopt);

    MVCNN::DType wmode;

    if (op.getNceTaskType() == vpux::VPUIP::NCETaskType::AVEPOOL) {
        wmode = inputRef->data_dtype();
        switch (wmode) {
        case MVCNN::DType::DType_U8:
        case MVCNN::DType::DType_I8:
            registers.elops_wload.elops_wload_bf.pool_wt_data = 0x0101;  // Two 8bit vals of 1
            break;
        case MVCNN::DType::DType_FP16:
            registers.elops_wload.elops_wload_bf.pool_wt_data = 0x3c00;  // FP16 1
            break;
        case MVCNN::DType::DType_BFP16:
            registers.elops_wload.elops_wload_bf.pool_wt_data = 0x3f80;  // BF16 1
            break;
        default:
            VPUX_THROW("INPUT DTYPE not supported");
            return;
            break;
        }
    } else {
        VPUX_THROW_WHEN(weights == nullptr, "NO WEIGHTS FOR non-POOL op");

        wmode = weightsRef->data_dtype();
    }

    auto dtype = ConfigDtype(wmode);

    if (dtype == static_cast<uint8_t>(nn_public::VpuInputTensorDType::INPUT_DTYPE_UNKNOWN)) {
        VPUX_THROW("UNKNOWN DATA TYPE");
    }

    registers.tensor_mode.tensor_mode_bf.wmode = dtype;

    if (wmode == MVCNN::DType::DType_U8) {
        registers.mpe_cfg.mpe_cfg_bf.mpe_wtbias =
                (weightsRef && weightsRef->quant_zero() && weightsRef->quant_zero()->size())
                        ? weightsRef->quant_zero()->Get(0)
                        : 0;
    }
}

void SetupKernel(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt = 1;
    // MPE_GRID_16x1 is deprecated for DPU 2.7 (VPUX37XX). Will always be MPE_GRID_4x4
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = MPE_GRID_4x4;

    int64_t kernelSizeH = 1, kernelSizeW = 1;
    int64_t kernelStridesH = 1, kernelStridesW = 1;
    if (op.getKernelSizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
        kernelSizeH = kernelSize[0];
        kernelSizeW = kernelSize[1];
    }
    if (op.getKernelStridesAttr() != nullptr) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
        kernelStridesH = kernelStrides[0];
        kernelStridesW = kernelStrides[1];
    }

    VPUX_THROW_WHEN(kernelSizeH < KERNEL_SIZE_MIN || kernelSizeH > KERNEL_SIZE_MAX, "KernelH out of accepted range");
    VPUX_THROW_WHEN(kernelSizeW < KERNEL_SIZE_MIN || kernelSizeW > KERNEL_SIZE_MAX, "KernelW out of accepted range");
    VPUX_THROW_WHEN(kernelStridesH < KERNEL_STRIDE_MIN || kernelStridesH > KERNEL_STRIDE_MAX,
                    "KernelStrideH out of accepted range");
    VPUX_THROW_WHEN(kernelStridesW < KERNEL_STRIDE_MIN || kernelStridesW > KERNEL_STRIDE_MAX,
                    "KernelStrideW out of accepted range");

    registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x = kernelSizeW;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y = kernelSizeH;
    registers.tensor_mode.tensor_mode_bf.stride = kernelStridesW - 1;
    if (kernelStridesW != kernelStridesH) {
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en = 1;
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y = kernelStridesH - 1;
    }
}

void SetupOutput(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    auto parentOutput = op.getParentOutput();
    auto parentOutputSparsityMap = op.getParentOutputSparsityMap();
    auto [parentOutputRef, parentOutputSparsityMapRef, parentOutputSETableRef] =
            createTensorRef(parentOutput, parentOutputSparsityMap, nullptr, outputSESize);

    bool is_out_dense = outputRef->data()->sparsity_index() == DEFAULT_INDEX;

    registers.odu_be_size = registers.odu_be_cnt = 0;
    registers.se_size = 0;

    auto dtype = ConfigOutputDtype(outputRef->data_dtype());
    if (dtype == static_cast<uint8_t>(nn_public::VpuOutputTensorDType::OUTPUT_DTYPE_UNKNOWN)) {
        VPUX_THROW("INVALID OUTPUT DATA TYPE");
    }

    registers.odu_cfg.odu_cfg_bf.dtype = dtype;

    registers.odu_cfg.odu_cfg_bf.mode = static_cast<uint32_t>(op.getIsSuperdense());

    registers.odu_cfg.odu_cfg_bf.grid = static_cast<unsigned int>(nn_public::VpuODUGrid::ODU_GRID_4x4);

    SetupInvariant_Grid(op, registers);

    registers.odu_cfg.odu_cfg_bf.write_ac = 1;               // Always write data out!
    registers.odu_cfg.odu_cfg_bf.write_pt = 0;               // Output SE table generation always disabled for NPU37XX
    registers.odu_cfg.odu_cfg_bf.write_sp = !is_out_dense;   // Enable/Disable output sparsity map generation
    registers.odu_cfg.odu_cfg_bf.sp_out_en = !is_out_dense;  // Enable/Disable compression of output activations

    registers.odu_cfg.odu_cfg_bf.swizzle_key = outputRef->swizzling_key();

    // Extract output permutation from output layout
    MVCNN::Permutation oduPermutation = getODUPermutationType(DimsOrder::fromValue(output));
    registers.odu_cfg.odu_cfg_bf.permutation = oduPermutation;

    registers.odu_cfg.odu_cfg_bf.sp_value = is_out_dense ? 0 : outputRef->quant_zero()->Get(0);

    registers.te_dim1.te_dim1_bf.te_dim_x = outputRef->dimensions()->Get(X) - 1;
    registers.te_dim0.te_dim0_bf.te_dim_y = outputRef->dimensions()->Get(Y) - 1;

    {  // ODU permutation dim_z calculation
        auto stride_b = outputRef->bit_strides()->Get(STRIDES(B));
        auto stride_x = outputRef->bit_strides()->Get(STRIDES(X));
        auto stride_y = outputRef->bit_strides()->Get(STRIDES(Y));
        auto stride_z = outputRef->bit_strides()->Get(STRIDES(Z));

        VPUX_THROW_WHEN(stride_x == 0 || stride_y == 0 || stride_z == 0 || stride_b == 0,
                        "stride_{xyzb} is zero, invalid configuration");

        uint64_t dim_x = 0, dim_y = 0, dim_z = 0;

        switch (oduPermutation) {
        case MVCNN::Permutation::Permutation_ZXY:
            // NHWC
            dim_y = stride_b / stride_y;
            dim_x = stride_y / stride_x;
            dim_z = stride_x / stride_z;
            break;
        case MVCNN::Permutation::Permutation_YZX:
            // NWCH
            dim_x = stride_b / stride_x;
            dim_z = stride_x / stride_z;
            dim_y = stride_z / stride_y;
            break;
        case MVCNN::Permutation::Permutation_ZYX:
            // NWHC
            dim_x = stride_b / stride_x;
            dim_y = stride_x / stride_y;
            dim_z = stride_y / stride_z;
            break;
        case MVCNN::Permutation::Permutation_XZY:
            // NHCW
            dim_y = stride_b / stride_y;
            dim_z = stride_y / stride_z;
            dim_x = stride_z / stride_x;
            break;
        case MVCNN::Permutation::Permutation_YXZ:
            // NCWH
            dim_z = stride_b / stride_z;
            dim_x = stride_z / stride_x;
            dim_y = stride_x / stride_y;
            break;
        case MVCNN::Permutation::Permutation_XYZ:
            // NCHW
            dim_z = stride_b / stride_z;
            dim_y = stride_z / stride_y;
            dim_x = stride_y / stride_x;
            break;
        default:
            VPUX_THROW("Wrong permutation, invalid configuration");
            break;
        }

        VPUX_THROW_WHEN(dim_x < 1 || dim_y < 1 || dim_z < 1, "All dimensions must be >= 1, invalid configuration");

        // HW registers require value to be decremented by 1.
        registers.te_dim1.te_dim1_bf.te_dim_x = checked_cast<uint32_t>(dim_x - 1);
        registers.te_dim0.te_dim0_bf.te_dim_y = checked_cast<uint32_t>(dim_y - 1);
        registers.te_dim0.te_dim0_bf.te_dim_z = checked_cast<uint32_t>(dim_z - 1);
    }

    // Sparse output split over H, since this is a special special case
    Setup_Output_SOH(outputRef, parentOutputRef, registers, is_out_dense);

    registers.base_adr[0] = 0;
}

uint32_t calc_se_size(uint32_t x) {
    uint32_t sz = 0;
    uint32_t x_orig = x;
    while (x) {
        x >>= 1;
        ++sz;
    }
    if ((sz == 0) || (x_orig != (1u << (sz - 1)))) {
        VPUX_THROW("storage_element_size is %d which is not a power of 2", x_orig);
    }
    // HW register NCE_DPU_Z_CONFIG.se_z_split has values: 1=16, 2=32....9=4096, 0=8192
    if (sz > 4) {
        sz -= 4;
    } else {
        sz = 1;
    }
    // if Z size is 8192 or bigger, adjust to HW value for 8192
    if (sz >= 10) {
        sz = 0;
        VPUX_THROW("storage_element_size bigger then 8192, HW value adjusted for 8192");
    }
    return sz;
}

unsigned int ConfigWorkloadSize(const std::optional<MVCNN::TensorReference>& inputRef, VPUIP::NCETaskType task_type,
                                unsigned int size) {
    switch (task_type) {
    case VPUIP::NCETaskType::CONV:
    case VPUIP::NCETaskType::ELTWISE:
        if (size != inputRef->dimensions()->Get(Z)) {
            size = inputRef->dimensions()->Get(Z);
        }
        break;

    default:
        break;
    }

    return size;
}

unsigned int ConfigWorkloadStart(VPUIP::NCETaskType task_type, unsigned int start) {
    switch (task_type) {
    case VPUIP::NCETaskType::CONV:
    case VPUIP::NCETaskType::ELTWISE:
        if (start != 0) {
            start = 0;
        }
        break;

    default:
        break;
    }

    return start;
}

MVCNN::MPE_Mode getMPEMode(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR:
        return MVCNN::MPE_Mode::MPE_Mode_VECTOR;
    case VPU::MPEMode::MATRIX:
        return MVCNN::MPE_Mode::MPE_Mode_MATRIX;
    case VPU::MPEMode::VECTOR_FP16:
        return MVCNN::MPE_Mode::MPE_Mode_VECTOR_FP16;
    case VPU::MPEMode::CUBOID_16x16:
        return MVCNN::MPE_Mode::MPE_Mode_CUBOID_16x16;
    case VPU::MPEMode::CUBOID_8x16:
        return MVCNN::MPE_Mode::MPE_Mode_CUBOID_8x16;
    case VPU::MPEMode::CUBOID_4x16:
        return MVCNN::MPE_Mode::MPE_Mode_CUBOID_4x16;
    case VPU::MPEMode::NOP:
        return MVCNN::MPE_Mode::MPE_Mode_NOP;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
}

void SetupVariant_NTHW_NTK(vpux::VPUMI37XX::DPUInvariantOp invariant, nn_public::VpuDPUVariantRegisters& registers) {
    auto mpe_frequent_mode = getMPEMode(invariant.getMpeFrequentMode());
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (invariant.getNceTaskType() == VPUIP::NCETaskType::ELTWISE) {
        mpe_frequent_mode = MVCNN::MPE_Mode::MPE_Mode_CUBOID_8x16;
    }

    // Sets up on NTHW on IDU
    switch (mpe_frequent_mode) {
    case MVCNN::MPE_Mode::MPE_Mode_VECTOR:
        registers.offset_addr.offset_addr_bf.nthw_ntk =
                static_cast<unsigned int>(nn_public::VpuIDUNthw_Ntk::IDU_NTHW_NTK_8_8);
        break;
    case MVCNN::MPE_Mode::MPE_Mode_CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        registers.offset_addr.offset_addr_bf.nthw_ntk =
                static_cast<unsigned int>(nn_public::VpuIDUNthw_Ntk::IDU_NTHW_NTK_4_16);
        break;
    case MVCNN::MPE_Mode::MPE_Mode_CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
        registers.offset_addr.offset_addr_bf.nthw_ntk =
                static_cast<unsigned int>(nn_public::VpuIDUNthw_Ntk::IDU_NTHW_NTK_8_8);
        break;
    case MVCNN::MPE_Mode::MPE_Mode_CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        registers.offset_addr.offset_addr_bf.nthw_ntk =
                static_cast<unsigned int>(nn_public::VpuIDUNthw_Ntk::IDU_NTHW_NTK_16_4);
        break;
    default:
        VPUX_THROW("unsipported mpe_frequent_mode {0}", mpe_frequent_mode);
        break;
    }
}

void SetupInvariant_SOH(const std::optional<MVCNN::TensorReference>& in_tensor,
                        const std::optional<MVCNN::TensorReference>& in_parent,
                        nn_public::VpuDPUInvariantRegisters& invariantRegisters, uint32_t clusters, bool isSegmented) {
    if (isSegmented && clusters > 1) {
        uint32_t seg_size = 0, sp_size = 0;
        bool is_act_dense = in_tensor->data()->sparsity_index() == DEFAULT_INDEX;
        unsigned int lines_per_cluster =
                SOH_LinesPerCluster(in_parent->dimensions()->Get(Y), in_tensor->dimensions()->Get(Y), clusters);

        auto in_dim_z = in_tensor->dimensions()->Get(Z);
        auto se_size = in_tensor->data()->storage_element_size();
        seg_size = in_parent->dimensions()->Get(X) * lines_per_cluster;
        sp_size = in_parent->dimensions()->Get(X) * lines_per_cluster * in_parent->dimensions()->Get(Z) >> 3;

        for (uint32_t i = 0; (i < clusters - 1) && (i < 3); i++) {
            if (is_act_dense) {
                invariantRegisters.se_sp_size[i].se_sp_size_bf.se_seg_size = seg_size >> 2;
            } else {
                auto num_ses = in_dim_z / se_size;
                if (in_dim_z % se_size) {
                    num_ses++;
                }

                // for dense_se=1 (no SE pointer table) we apply the same scaling as the act_dense case
                invariantRegisters.se_sp_size[i].se_sp_size_bf.se_seg_size = (seg_size >> 2) * num_ses;
                invariantRegisters.se_sp_size[i].se_sp_size_bf.sp_seg_size = sp_size >> 4;

                if (in_tensor->data()->storage_element_index() != DEFAULT_INDEX) {
                    invariantRegisters.se_sp_size[i].se_sp_size_bf.se_seg_size = seg_size * num_ses;
                }
            }

            switch (i + 1) {
            case 1:
                invariantRegisters.base_offset_a |= 0x1 << 9;
                break;
            case 2:
                invariantRegisters.base_offset_b |= 0x2 << 0;
                break;
            case 3:
                invariantRegisters.base_offset_b |= 0x3 << 9;
                break;
            }
        }

        // Assuming symmetric SOH layer split across both VPUX37XX tiles
        invariantRegisters.se_sp_size[1] = invariantRegisters.se_sp_size[0];
        invariantRegisters.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment = 1;
    }
}

void SetupInvariant_SOH_Input(const std::optional<MVCNN::TensorReference>& in_tensor,
                              const std::optional<MVCNN::TensorReference>& in_parent,
                              nn_public::VpuDPUInvariantRegisters& invariantRegisters) {
    bool is_act_dense = in_tensor->data()->sparsity_index() == DEFAULT_INDEX;

    if (!is_act_dense) {
        if ((invariantRegisters.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment == 0) &&
            (in_parent->dimensions()->Get(Y) != in_tensor->dimensions()->Get(Y))) {
            invariantRegisters.base_offset_a |= in_tensor->locale_index()->Get(0);
        }
    }
}

void SetupInvariant_Input_SE_Size(const std::optional<MVCNN::TensorReference>& in_tensor,
                                  nn_public::VpuDPUInvariantRegisters& invariantRegisterst) {
    bool is_act_dense = in_tensor->data()->sparsity_index() == DEFAULT_INDEX;
    auto in_dim_z = in_tensor->dimensions()->Get(Z);
    auto se_size = in_tensor->data()->storage_element_size();

    if (!is_act_dense && in_tensor->data()->storage_element_size()) {
        // storage_element_size for eltwise != Z dim ---- not tested

        // Z should be a power of 2
        invariantRegisterst.z_config.z_config_bf.se_z_split = calc_se_size(se_size);
        // num storage elements offset by 1 in HW. 0 = 1 SE Per Z direction
        invariantRegisterst.z_config.z_config_bf.num_ses_in_z_dir = (in_dim_z / se_size) - 1;
        if (in_dim_z % se_size) {
            invariantRegisterst.z_config.z_config_bf.num_ses_in_z_dir++;
            // Z not divisible with SE size
        }
    }
}

size_t getInputClusterCount(mlir::Value parentInput) {
    if (auto distributedBufferType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(parentInput.getType())) {
        return distributedBufferType.getDistribution().getNumClusters().getInt();
    }
    return 1;
}

void SetupInvariant_Convolution(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto weights = op.getWeights();
    auto weightsSparsityMap = op.getWeightsSparsityMap();

    createTensorRef(weights, weightsSparsityMap, nullptr, std::nullopt);

    auto parentInput = op.getParentInput();
    auto parentInputSparsityMap = op.getParentInputSparsityMap();
    auto parentInputSETables = op.getParentInputStorageElementTable();
    auto [parentInputRef, parentInputSparsityMapRef, parentInputSETableRef] =
            createTensorRef(parentInput, parentInputSparsityMap, parentInputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    const auto inputClusterCount = checked_cast<uint32_t>(getInputClusterCount(parentInput));

    registers.tensor_mode.tensor_mode_bf.zm_input = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

    bool is_wt_dense = op.getWeightsSparsityMap() == nullptr;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

    const auto cm_sp_pattern = op.getCmSpPattern().value_or(0);
    bool is_cm_mode = cm_sp_pattern != 0;

    SetupInvariant_SOH(inputRef, parentInputRef, registers, inputClusterCount, op.getIsSegmented());

    SetupInvariant_SOH_Input(inputRef, parentInputRef, registers);

    // Input Size
    if (registers.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment) {
        registers.tensor_size0.tensor_size0_bf.tensor_size_y = parentInputRef->dimensions()->Get(Y);
        registers.tensor_size1.tensor_size1_bf.tensor_size_z = parentInputRef->dimensions()->Get(Z);
        registers.tensor_size0.tensor_size0_bf.tensor_size_x = parentInputRef->dimensions()->Get(X);
    } else if (parentInputRef->dimensions()->Get(Y) != inputRef->dimensions()->Get(Y)) {
        registers.tensor_size0.tensor_size0_bf.tensor_size_y = inputRef->dimensions()->Get(Y);
    }

    SetupInvariant_Input_SE_Size(inputRef, registers);

    if (is_cm_mode) {
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = 1;
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = 1;

        registers.tensor_size1.tensor_size1_bf.tensor_size_z = 16;
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins = 1;

        registers.z_config.z_config_bf.cm_sp_pattern = checked_cast<uint32_t>(cm_sp_pattern);

        registers.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en = op.getInputChannelsCompression();

        if (outputRef->data_dtype() == MVCNN::DType::DType_FP16) {
            registers.odu_cfg.odu_cfg_bf.grid = static_cast<unsigned int>(nn_public::VpuODUGrid::ODU_GRID_4x4);
        }
    }
}

void SetupInvariant_MaxPool(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto parentInput = op.getParentInput();
    auto parentInputSparsityMap = op.getParentInputSparsityMap();
    auto parentInputSETables = op.getParentInputStorageElementTable();
    auto [parentInputRef, parentInputSparsityMapRef, parentInputSETableRef] =
            createTensorRef(parentInput, parentInputSparsityMap, parentInputSETables, inputSESize);

    const auto inputClusterCount = checked_cast<uint32_t>(getInputClusterCount(parentInput));

    registers.tensor_size0.tensor_size0_bf.tensor_size_x = parentInputRef->dimensions()->Get(X);

    registers.tensor_mode.tensor_mode_bf.workload_operation = 2;  // maxpool
    registers.tensor_mode.tensor_mode_bf.dw_input = 1;
    registers.tensor_mode.tensor_mode_bf.zm_input = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins = 1;
    registers.elops_wload.elops_wload_bf.pool_wt_rd_dis = 1;

    SetupInvariant_SOH(inputRef, parentInputRef, registers, inputClusterCount, op.getIsSegmented());
}

void SetupInvariant_DwConvolution(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto parentInput = op.getParentInput();
    auto parentInputSparsityMap = op.getParentInputSparsityMap();
    auto parentInputSETables = op.getParentInputStorageElementTable();
    auto [parentInputRef, parentInputSparsityMapRef, parentInputSETableRef] =
            createTensorRef(parentInput, parentInputSparsityMap, parentInputSETables, inputSESize);

    auto weights = op.getWeights();
    auto weightsSparsityMap = op.getWeightsSparsityMap();
    auto [weightsRef, weightsSparsityMapRef, weightsSETableRef] =
            createTensorRef(weights, weightsSparsityMap, nullptr, std::nullopt);

    const auto inputClusterCount = checked_cast<uint32_t>(getInputClusterCount(parentInput));

    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins = 1;  // enable the IDU to generate the weight sparsity
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

    // Dedicated HW for channel-major removed for 2.7 - all convolutions
    // processed as Z-major. Depthwise here is processed as a subset of
    // the Z-major convolution.
    registers.tensor_mode.tensor_mode_bf.dw_input = 1;
    registers.tensor_mode.tensor_mode_bf.zm_input = 1;

    bool is_wt_dense = weightsRef ? weightsRef->data()->sparsity_index() == DEFAULT_INDEX : true;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

    SetupInvariant_SOH(inputRef, parentInputRef, registers, inputClusterCount, op.getIsSegmented());
}

bool SetupInvariant_Eltwise(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& registers) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    // Parent input override is required for PermuteQuantize.
    // PermuteQuantize is always an ELTWISE op.
    const auto isPermuteQuantize = op.getIsPermuteQuantizeAttr() != nullptr;
    const auto overrideParentIn = isPermuteQuantize && (op.getParentInput().getType() != op.getInput().getType());
    auto parentInput = overrideParentIn ? op.getInput() : op.getParentInput();

    auto parentInputSparsityMap = op.getParentInputSparsityMap();
    auto parentInputSETables = op.getParentInputStorageElementTable();
    auto [parentInputRef, parentInputSparsityMapRef, parentInputSETableRef] =
            createTensorRef(parentInput, parentInputSparsityMap, parentInputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    auto parentOutput = op.getParentOutput();
    auto parentOutputSparsityMap = op.getParentOutputSparsityMap();
    auto [parentOutputRef, parentOutputSparsityMapRef, parentOutputSETableRef] =
            createTensorRef(parentOutput, parentOutputSparsityMap, nullptr, outputSESize);

    auto weights = op.getWeights();
    auto weightsSparsityMap = op.getWeightsSparsityMap();
    auto [weightsRef, weightsSparsityMapRef, weightsSETableRef] =
            createTensorRef(weights, weightsSparsityMap, nullptr, std::nullopt);

    int64_t kernelSizeH = 1, kernelSizeW = 1;
    if (op.getKernelSizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
        kernelSizeH = kernelSize[0];
        kernelSizeW = kernelSize[1];
    }

    if (kernelSizeH != 1 || kernelSizeW != 1) {
        VPUX_THROW("Eltwise only supports 1x1 kernel. Got {0}x{1}", kernelSizeW, kernelSizeH);
        return false;
    }

    if (parentOutputRef && parentOutputRef->dimensions()->Get(Z) != outputRef->dimensions()->Get(Z)) {
        VPUX_THROW("Eltwise does not support split over K\n");
        return false;
    }

    auto amode = inputRef->data_dtype();
    auto wmode = weightsRef->data_dtype();
    auto omode = outputRef->data_dtype();

    registers.tensor_mode.tensor_mode_bf.dw_input = 0;
    registers.tensor_mode.tensor_mode_bf.zm_input = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

    ::std::optional<SmallVector<int64_t>> in1QuantMult;
    ::std::optional<SmallVector<int64_t>> in2QuantMult;

    for (auto ppeOp : op.getPpe().getOps<VPUMI37XX::PPETaskOp>()) {
        auto ppeAttr = ppeOp.getPpeAttr();
        auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
        VPUX_THROW_WHEN(intPpeAttr == nullptr,
                        "Expected PPEIntAttr type but got {0}, make sure to use the right factory version", ppeAttr);
        if (const auto in1QuantMultAttr = intPpeAttr.getIn1QuantMult()) {
            in1QuantMult = parseIntArrayAttr<int64_t>(in1QuantMultAttr);
        }
        if (const auto in2QuantMultAttr = intPpeAttr.getIn2QuantMult()) {
            in2QuantMult = parseIntArrayAttr<int64_t>(in2QuantMultAttr);
        }
    }

    VPUX_THROW_WHEN(in1QuantMult.has_value() != in2QuantMult.has_value(),
                    "Both inputs must be either quantized or not.");
    const auto isInputQuantizationProvided = in1QuantMult.has_value() && in2QuantMult.has_value();

    if (isInputQuantizationProvided) {
        // Shifts must be set 0 for VPUX37XX and VPUX40XX runtime to be considered, otherwise runtime will ignore inputs
        // MULT.
        const auto inputSparsityMapOffset =
                inputSparsityMap ? std::optional<int64_t>{inputSparsityMap.getDefiningOp<VPURT::DeclareBufferOp>()
                                                                  .getByteOffset()}
                                 : std::nullopt;
        const auto inputStorageElementOffset =
                inputSETables
                        ? std::optional<int64_t>{inputSETables.getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset()}
                        : std::nullopt;
        inputRef = getTensorReferenceWithUpdatedQuantParams(in1QuantMult.value(), {0}, input, inputSparsityMapOffset,
                                                            inputStorageElementOffset, inputSESize);
        const auto weightsSparsityMapOffset =
                weightsSparsityMap ? std::optional<int64_t>{weightsSparsityMap.getDefiningOp<VPURT::DeclareBufferOp>()
                                                                    .getByteOffset()}
                                   : std::nullopt;
        weightsRef = getTensorReferenceWithUpdatedQuantParams(in2QuantMult.value(), {0}, weights,
                                                              weightsSparsityMapOffset, std::nullopt, std::nullopt);
    }

    uint32_t elop_scale_a = 1;
    uint32_t elop_scale_b = 1;
    // For VPUX37XX, original code computing eltwise scales on the VPU disabled (E#5671)
    // If blob contains scales, it is the final values (E#33155)
    if (inputRef->quant_mult()->size() && weightsRef->quant_mult()->size() && inputRef->quant_shift()->size() &&
        weightsRef->quant_shift()->size()) {
        // When the compiler supplies valid scales the corresponding shifts will be 0 (E#33155)
        if (!inputRef->quant_shift()->Get(0) && !weightsRef->quant_shift()->Get(0)) {
            elop_scale_a = inputRef->quant_mult()->Get(0);
            elop_scale_b = weightsRef->quant_mult()->Get(0);
        }
    }

    // elop_scale_* registers are U16 unlike ppe_scale_mult which are I16
    registers.elop_scale.elop_scale_bf.elop_scale_a = elop_scale_a;
    registers.elop_scale.elop_scale_bf.elop_scale_b = elop_scale_b;

    // For FP16 eltwise grid needs to be 4x4
    if (((amode == MVCNN::DType::DType_FP16) && (wmode == MVCNN::DType::DType_FP16)) ||
        (omode == MVCNN::DType::DType_FP16)) {
        registers.odu_cfg.odu_cfg_bf.grid = static_cast<unsigned int>(nn_public::VpuODUGrid::ODU_GRID_4x4);
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign =
                static_cast<unsigned int>(nn_public::VpuMPEGrid::MPE_GRID_4x4);
    }

    bool is_wt_dense = weightsRef->data()->sparsity_index() == DEFAULT_INDEX;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

    registers.elops_wload.elops_wload_bf.elop_wload =
            1;  // read in 2 tensors instead of a tensor and weight sets for a standard convolution.

    SetupInvariant_SOH_Input(inputRef, parentInputRef, registers);
    SetupInvariant_Input_SE_Size(inputRef, registers);

    return true;
}

std::optional<MVCNN::PPETask> getPPETask(vpux::VPUMI37XX::DPUInvariantOp op) {
    SmallVector<uint8_t> ppeList;
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t LreluMult = 1;
    uint32_t LreluShift = 0;
    ::std::optional<SmallVector<int64_t>> ppeQuantMult;
    ::std::optional<SmallVector<int64_t>> ppeQuantShift;
    ::std::optional<int64_t> ppeQuantPostShift;
    ::std::optional<float> ppeQuantScale;
    float fpPReluAlpha = 1.f;

    for (auto ppeOp : op.getPpe().getOps<VPUMI37XX::PPETaskOp>()) {
        auto ppeAttr = ppeOp.getPpeAttr();
        auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
        VPUX_THROW_WHEN(intPpeAttr == nullptr,
                        "Expected PPEintAttr type but got {0}, make sure to use the right factory version", ppeAttr);

        if (intPpeAttr.getMode()) {
            const auto ppeType = getPPELayerType(intPpeAttr.getMode().getValue());
            if (ppeType != MVCNN::PPELayerType_NOOP) {
                ppeList.push_back(ppeType);
            }
        } else {
            VPUX_THROW("PPEMode was not found.");
        }

        if (const auto clampLowAttr = intPpeAttr.getClampLow()) {
            clampLow = checked_cast<int32_t>(clampLowAttr.getValue().getSExtValue());
        }
        if (const auto clampHighAttr = intPpeAttr.getClampHigh()) {
            clampHigh = checked_cast<int32_t>(clampHighAttr.getValue().getSExtValue());
        }
        if (const auto LreluMultAttr = intPpeAttr.getLreluMult()) {
            LreluMult = checked_cast<int32_t>(LreluMultAttr.getValue().getSExtValue());
        }
        if (const auto LreluShiftAttr = intPpeAttr.getLreluShift()) {
            LreluShift = checked_cast<uint32_t>(LreluShiftAttr.getValue().getSExtValue());
        }
        if (const auto quantMultAttr = intPpeAttr.getQuantMult()) {
            ppeQuantMult = parseIntArrayAttr<int64_t>(quantMultAttr);
        }
        if (const auto quantShiftAttr = intPpeAttr.getQuantShift()) {
            ppeQuantShift = parseIntArrayAttr<int64_t>(quantShiftAttr);
        }
        if (const auto quantPostShiftAttr = intPpeAttr.getQuantPostShift()) {
            ppeQuantPostShift = checked_cast<int64_t>(quantPostShiftAttr.getValue().getSExtValue());
        }
        // Note: For values like 0.1, checked_cast fails, due to loss in precision when converting
        // from double to float and back, due to the static_cast<double>(static_cast<float>(value)) == value
        // check; use static_cast instead
        if (const auto quantScaleAttr = intPpeAttr.getQuantScale()) {
            auto floatScaleAttr = quantScaleAttr.getValue()[0];
            ppeQuantScale = static_cast<float>(mlir::dyn_cast<mlir::FloatAttr>(floatScaleAttr).getValueAsDouble());
        }
        if (const auto fpPReluAlphaAttr = intPpeAttr.getFpPreluAlpha()) {
            fpPReluAlpha = static_cast<float>(intPpeAttr.getFpPreluAlpha().getValueAsDouble());
        }
    }
    VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

    auto ppeLayerTypes = createVector(ppeList);
    auto ppeFixedFunction =
            MVCNN::PPEFixedFunction(std::move(ppeLayerTypes), clampLow, clampHigh, LreluMult, LreluShift);

    return MVCNN::PPETask(std::move(ppeFixedFunction), MVCNN::PPERoundingMode_RNE, ppeQuantScale.value_or(1.0),
                          fpPReluAlpha);
}

static bool applyClamping(nn_public::VpuDPUInvariantRegisters& regs, const activationFunctionDesc& actFuncDesc) {
    regs.ppe_scale_lclamp = (uint32_t)actFuncDesc.clampLow;
    regs.ppe_scale_hclamp = (uint32_t)actFuncDesc.clampHigh;
    return true;
}

static void setupTaskTypeDPUEltwise(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& regs) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    auto weights = op.getWeights();
    auto weightsSparsityMap = op.getWeightsSparsityMap();

    createTensorRef(weights, weightsSparsityMap, nullptr, std::nullopt);

    auto ppe_task = getPPETask(op);

    SmallVector<uint8_t> ppeList;
    ::std::optional<SmallVector<int64_t>> ppeQuantMult;
    ::std::optional<SmallVector<int64_t>> ppeQuantShift;
    ::std::optional<SmallVector<int64_t>> in1QuantMult;
    ::std::optional<SmallVector<int64_t>> in2QuantMult;

    for (auto ppeOp : op.getPpe().getOps<VPUMI37XX::PPETaskOp>()) {
        auto ppeAttr = ppeOp.getPpeAttr();
        auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
        VPUX_THROW_WHEN(intPpeAttr == nullptr,
                        "Expected PPEIntAttr type but got {0}, make sure to use the right factory version", ppeAttr);

        if (const auto quantMultAttr = intPpeAttr.getQuantMult()) {
            ppeQuantMult = parseIntArrayAttr<int64_t>(quantMultAttr);
        }
        if (const auto quantShiftAttr = intPpeAttr.getQuantShift()) {
            ppeQuantShift = parseIntArrayAttr<int64_t>(quantShiftAttr);
        }
        if (const auto in1QuantMultAttr = intPpeAttr.getIn1QuantMult()) {
            in1QuantMult = parseIntArrayAttr<int64_t>(in1QuantMultAttr);
        }
        if (const auto in2QuantMultAttr = intPpeAttr.getIn2QuantMult()) {
            in2QuantMult = parseIntArrayAttr<int64_t>(in2QuantMultAttr);
        }
    }

    const auto isQuantizationProvided = ppeQuantMult.has_value() && ppeQuantShift.has_value();

    if (isQuantizationProvided) {
        const auto outputSparsityMapOffset =
                outputSparsityMap ? std::optional<int64_t>{outputSparsityMap.getDefiningOp<VPURT::DeclareBufferOp>()
                                                                   .getByteOffset()}
                                  : std::nullopt;
        outputRef = getTensorReferenceWithUpdatedQuantParams(ppeQuantMult.value(), ppeQuantShift.value(), output,
                                                             outputSparsityMapOffset, std::nullopt, outputSESize);
    }

    // PPE scale override is required for both scenarios: when the input is float or quant.
    // Element-wise operations do not provide weights table to read the scale from.
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

    // Set PPE to read quant values from registers for eltwise since there
    // are no weights tables.
    // Compiler is responsible for setting scale.
    switch (inputRef->data_dtype()) {
    case MVCNN::DType::DType_I32:
    case MVCNN::DType::DType_I8:
    case MVCNN::DType::DType_U8:
    case MVCNN::DType::DType_I4:
    case MVCNN::DType::DType_U4: {
        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = ppe_task->rounding();
        // Scale is provided as a pair of integer mult and shift values.
        // ppe_scale_mult is a I16 register; implicit casting is done
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
                outputRef->quant_mult()->size() ? outputRef->quant_mult()->Get(0) : 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
                outputRef->quant_shift()->size() ? outputRef->quant_shift()->Get(0) : 0;

        // ppe_g8_bias_a & ppe_g8_bias_b are never set in RT
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0;
        // Bypass the FP pipeline, in case the value is arriving at the PPE as FP (there's no need to
        // scale it twice).
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
    } break;
    case MVCNN::DType::DType_FP32:
    case MVCNN::DType::DType_FP16:
    case MVCNN::DType::DType_BFP16:
    case MVCNN::DType::DType_FP8: {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 1;
        // Scale is provided as a float multiplier.
        u32f32 fp32_scale;
        fp32_scale.f32 = ppe_task->fp_scale_data();
        regs.ppe_fp_scale = fp32_scale.u32;

        // ppe_g8_bias_a & ppe_g8_bias_b are never set in RT
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0;
        // TODO: It's not correct to assume the op has no bias.
        // Should just program the global bias value given by compiler.
        regs.ppe_fp_bias = 0;
    } break;
    default: {
        VPUX_THROW("Unsupported input datatype for Eltwise operation!");
        break;
    }
    }
}

static void setupTaskTypeDPUMaxPool(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& regs) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    const bool hasFP16Input =
            inputRef->data_dtype() == MVCNN::DType::DType_FP16 || inputRef->data_dtype() == MVCNN::DType::DType_BFP16;
    const bool hasFloatInput = hasFP16Input || inputRef->data_dtype() == MVCNN::DType::DType_FP32 ||
                               inputRef->data_dtype() == MVCNN::DType::DType_FP8;
    const bool hasQuantInput = !hasFloatInput;

    const bool hasFP16Output = outputRef->data_dtype() == MVCNN::DType::DType_FP16;

    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

    if (hasQuantInput && hasFP16Output) {
        // Re-scale the result in mixed precision mode
        // ppe_scale_mult is a I16 register; implicit casting is done
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
                inputRef->quant_mult()->size() ? inputRef->quant_mult()->Get(0) : 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
                inputRef->quant_shift()->size() ? inputRef->quant_shift()->Get(0) : 0;
    } else {
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 0x1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0x0;
    }

    regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0x3;  // 0x3 - no round

    // ppe_g8_bias_a & ppe_g8_bias_b are never set in RT
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0x0;
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0x0;

    if (hasFP16Input) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0x1;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                0x0;  // FP16 MaxPool result is already FP16 with CRL FP MAC => no conversion
    }
    if (hasFloatInput) {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0x1;
        regs.ppe_fp_scale = 0x3f800000;  // fp32 equiv of 1
        regs.ppe_fp_bias = 0x0;
    }
}

static void setupTaskTypeDPUAvgPool(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& regs) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    auto ppe_task = getPPETask(op);

    SmallVector<uint8_t> ppeList;
    ::std::optional<SmallVector<int64_t>> ppeQuantMult;
    ::std::optional<SmallVector<int64_t>> ppeQuantShift;

    for (auto ppeOp : op.getPpe().getOps<VPUMI37XX::PPETaskOp>()) {
        auto ppeAttr = ppeOp.getPpeAttr();
        auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
        VPUX_THROW_WHEN(intPpeAttr == nullptr,
                        "Expected PPEIntAttr type but got {0}, make sure to use the right factory version", ppeAttr);
        if (intPpeAttr.getMode()) {
            const auto ppeType = getPPELayerType(intPpeAttr.getMode().getValue());
            if (ppeType != MVCNN::PPELayerType_NOOP) {
                ppeList.push_back(ppeType);
            }
        } else {
            VPUX_THROW("PPEMode was not found.");
        }
        if (const auto quantMultAttr = intPpeAttr.getQuantMult()) {
            ppeQuantMult = parseIntArrayAttr<int64_t>(quantMultAttr);
        }
        if (const auto quantShiftAttr = intPpeAttr.getQuantShift()) {
            ppeQuantShift = parseIntArrayAttr<int64_t>(quantShiftAttr);
        }
    }

    VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

    // PPE scale override is required for both scenarios: when the input is float or quant.
    // Average pooling does not provide weights table to read the scale from.
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

    // Compiler is responsible for setting scale to include both the
    // quantization parameters and the average pool division factor.
    switch (inputRef->data_dtype()) {
    case MVCNN::DType::DType_I32:
    case MVCNN::DType::DType_I8:
    case MVCNN::DType::DType_U8:
    case MVCNN::DType::DType_I4:
    case MVCNN::DType::DType_U4: {
        // Scale is provided as a pair of integer mult and shift values.
        if (ppeQuantMult.has_value()) {
            regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = ppeQuantMult.value()[0];
        } else {
            regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
        }
        if (ppeQuantShift.has_value()) {
            regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = ppeQuantShift.value()[0];
        } else {
            regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
        }

        // Bypass the FP pipeline, in case the value is arriving at the PPE as FP (there's no need to
        // scale it twice).
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
    } break;
    case MVCNN::DType::DType_FP32:
    case MVCNN::DType::DType_FP16:
    case MVCNN::DType::DType_BFP16:
    case MVCNN::DType::DType_FP8: {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 1;
        // Scale is provided as a float multiplier.
        u32f32 fp32_scale;
        fp32_scale.f32 = ppe_task->fp_scale_data();
        regs.ppe_fp_scale = fp32_scale.u32;
        // TODO: It's not correct to assume the op has no bias.
        // Should just program the global bias value given by compiler.
        regs.ppe_fp_bias = 0;
    } break;
    default: {
        VPUX_THROW("Unsupported input datatype for AVG Pool operation!");
        break;
    }
    }
}

static void setupTaskTypeDPUDwConv(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& regs) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    const auto in_data_type = inputRef->data_dtype();
    const auto out_data_type = outputRef->data_dtype();

    const auto hasFloatInput = in_data_type == MVCNN::DType::DType_FP16 || in_data_type == MVCNN::DType::DType_FP32 ||
                               in_data_type == MVCNN::DType::DType_BFP16 || in_data_type == MVCNN::DType::DType_FP8;
    const auto hasQuantOutput = out_data_type == MVCNN::DType::DType_I32 || out_data_type == MVCNN::DType::DType_I8 ||
                                out_data_type == MVCNN::DType::DType_U8 || out_data_type == MVCNN::DType::DType_I4 ||
                                out_data_type == MVCNN::DType::DType_U4;
    if (hasFloatInput && hasQuantOutput) {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    }
}

static void setupTaskType(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& regs) {
    if (op.getNceTaskType() == VPUIP::NCETaskType::ELTWISE) {
        setupTaskTypeDPUEltwise(op, regs);
    } else if (op.getNceTaskType() == VPUIP::NCETaskType::MAXPOOL) {
        setupTaskTypeDPUMaxPool(op, regs);
    } else if (op.getNceTaskType() == VPUIP::NCETaskType::AVEPOOL && regs.elops_wload.elops_wload_bf.pool_wt_rd_dis) {
        setupTaskTypeDPUAvgPool(op, regs);
    } else if (op.getNceTaskType() == VPUIP::NCETaskType::DWCONV) {
        setupTaskTypeDPUDwConv(op, regs);
    } else {
        // do nothing
    }
}

bool Setup_PPE(vpux::VPUMI37XX::DPUInvariantOp op, nn_public::VpuDPUInvariantRegisters& regs) {
    auto input = op.getInput();
    auto inputSparsityMap = op.getInputSparsityMap();
    auto inputSETables = op.getInputStorageElementTable();
    auto inputSESize = op.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto output = op.getOutputBuffs()[0];
    auto outputSparsityMap = op.getOutputSparsityMapBuff();
    auto outputSESize = op.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    auto ppe_task = getPPETask(op);

    auto lrelu_mult = ppe_task->fixed_function()->Lrelu_Mult();
    auto lrelu_shift = ppe_task->fixed_function()->Lrelu_Shift();

    // ppe_g8_bias_a & ppe_g8_bias_b are never set in RT
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0;
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0;

    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = 0;  // Used to set the zero point for u8

    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 0;
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0;
    regs.ppe_bias = 0;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
    regs.ppe_fp_bias = 0;
    regs.ppe_fp_scale = 0;

    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = lrelu_mult;    // Serialised in fixed function
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = lrelu_shift;  // Serialised in fixed function
    regs.ppe_scale_hclamp = 0;  // Default values applied per output data type, may be serialised in ff
    regs.ppe_scale_lclamp = 0;  // Default values applied per output data type, may be serialised in ff
    regs.ppe_misc.ppe_misc_bf.ppe_i32_convert = 0;  // Use in mixed precision when going fixed -> float point, infer
    regs.ppe_misc.ppe_misc_bf.ppe_fp16_clamp = 1;
    regs.ppe_misc.ppe_misc_bf.ppe_fp16_ftz = 0;  // Not serialised
    regs.ppe_fp_prelu = 0;                       // Derive from ppe_prelu_mult & ppe_prelu_shift
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;   // Set based on data types, if we see float point - don't bypass!
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0;  // Default to no conversion but set based on output data type

    uint8_t out_zero_point = (outputRef->quant_zero()->size() && op.getNceTaskType() != VPUIP::NCETaskType::MAXPOOL)
                                     ? outputRef->quant_zero()->Get(0)
                                     : 0;

    activationFunctionDesc actFuncDesc;

    if (!areSupportedInputOutputTypes(inputRef->data_dtype(), outputRef->data_dtype())) {
        return false;
    }

    // Check if there's a PPE fixed-function task, and if so, whether it
    // could be an activation function (requires GFS 3.22.x)
    if (!setupActivationFunction(ppe_task, inputRef, actFuncDesc)) {
        return false;
    }

    bool successful = false;

    switch (inputRef->data_dtype()) {
    case MVCNN::DType::DType_I4:
    case MVCNN::DType::DType_U4:
    case MVCNN::DType::DType_I8:
    case MVCNN::DType::DType_U8:
    case MVCNN::DType::DType_I32:
        successful = setupInt(inputRef->data_dtype(), outputRef->data_dtype(), regs, actFuncDesc, out_zero_point,
                              lrelu_mult, lrelu_shift);
        break;
    case MVCNN::DType::DType_FP8:
    case MVCNN::DType::DType_FP16:
    case MVCNN::DType::DType_BFP16:
    case MVCNN::DType::DType_FP32:
        successful = setupFloat(inputRef->data_dtype(), outputRef->data_dtype(), regs, actFuncDesc, out_zero_point);
        break;
    default:
        VPUX_THROW("Unsupported dtype");
        successful = false;
    }

    VPUX_THROW_WHEN(!successful, "Cannot set up PPE");
    successful = applyClamping(regs, actFuncDesc);
    VPUX_THROW_WHEN(!successful, "Cannot set up PPE");

    setupTaskType(op, regs);
    for (auto ppeOp : op.getPpe().getOps<VPUMI37XX::PPETaskOp>()) {
        if (ppeOp.getPpeFpScale().has_value() || ppeOp.getPpeFpBias().has_value()) {
            regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 1;
        }

        if (ppeOp.getPpeFpScale().has_value()) {
            const auto fpScale = static_cast<float>(ppeOp.getPpeFpScale().value().convertToDouble());
            regs.ppe_fp_scale = llvm::bit_cast<int32_t>(fpScale);
        }

        if (ppeOp.getPpeFpBias().has_value()) {
            const auto fpBias = static_cast<float>(ppeOp.getPpeFpBias().value().convertToDouble());
            regs.ppe_fp_bias = llvm::bit_cast<int32_t>(fpBias);
        }
    }

    return true;
}

void parseVariant(vpux::VPUMI37XX::DPUVariantOp op, vpux::VPUMI37XX::DPUInvariantOp invariant,
                  nn_public::VpuDPUVariant& taskWrapper) {
    nn_public::VpuDPUVariantRegisters& registers = taskWrapper.registers_;

    auto input = invariant.getInput();
    auto inputSparsityMap = invariant.getInputSparsityMap();
    auto inputSETables = invariant.getInputStorageElementTable();
    auto inputSESize = invariant.getInputSeSize();
    auto [inputRef, inputSparsityMapRef, inputSETableRef] =
            createTensorRef(input, inputSparsityMap, inputSETables, inputSESize);

    auto output = invariant.getOutputBuffs()[0];
    auto outputSparsityMap = invariant.getOutputSparsityMapBuff();
    auto outputSESize = invariant.getOutputSeSize();
    auto [outputRef, outputSparsityMapRef, outputSETableRef] =
            createTensorRef(output, outputSparsityMap, nullptr, outputSESize);

    auto weights = invariant.getWeights();
    auto weightsSparsityMap = invariant.getWeightsSparsityMap();
    auto [weightsRef, weightsSparsityMapRef, weightsSETableRef] =
            createTensorRef(weights, weightsSparsityMap, nullptr, std::nullopt);

    auto parentInput = invariant.getParentInput();
    auto parentInputSparsityMap = invariant.getParentInputSparsityMap();
    auto parentInputSETables = invariant.getParentInputStorageElementTable();
    auto [parentInputRef, parentInputSparsityMapRef, parentInputSETableRef] =
            createTensorRef(parentInput, parentInputSparsityMap, parentInputSETables, inputSESize);

    auto parentOutput = invariant.getParentOutput();
    auto parentOutputSparsityMap = invariant.getParentOutputSparsityMap();
    auto [parentOutputRef, parentOutputSparsityMapRef, parentOutputSETableRef] =
            createTensorRef(parentOutput, parentOutputSparsityMap, nullptr, outputSESize);

    const auto inputClusterCount = checked_cast<unsigned int>(getInputClusterCount(parentInput));

    int64_t kernelSizeH = 1, kernelSizeW = 1;
    int64_t kernelStridesH = 1, kernelStridesW = 1;
    int64_t kernelPadL = 0, kernelPadT = 0;
    if (invariant.getKernelSizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(invariant.getKernelSizeAttr());
        kernelSizeH = kernelSize[0];
        kernelSizeW = kernelSize[1];
    }
    if (invariant.getKernelStridesAttr() != nullptr) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(invariant.getKernelStridesAttr());
        kernelStridesH = kernelStrides[0];
        kernelStridesW = kernelStrides[1];
    }
    if (invariant.getKernelPaddingAttr() != nullptr) {
        const auto kernelPadding = invariant.getKernelPaddingAttr();
        kernelPadL = kernelPadding.getLeft().getInt();
        kernelPadT = kernelPadding.getTop().getInt();
    }

    const auto localPadding = op.getPadAttr();
    auto local_PL = localPadding.getLeft().getInt();
    auto local_PR = localPadding.getRight().getInt();
    auto local_PT = localPadding.getTop().getInt();
    auto local_PB = localPadding.getBottom().getInt();

    auto workload_start = parseIntArrayAttr<int64_t>(op.getStartAttr());
    auto workload_end = parseIntArrayAttr<int64_t>(op.getEndAttr());

    auto output_start_x = static_cast<int16_t>(workload_start[0]);
    auto output_start_y = static_cast<int16_t>(workload_start[1]);
    auto output_start_z = static_cast<int16_t>(workload_start[2]);
    auto output_end_x = static_cast<int16_t>(workload_end[0]);
    auto output_end_y = static_cast<int16_t>(workload_end[1]);
    auto output_end_z = static_cast<int16_t>(workload_end[2]);

    auto op_size_x = output_end_x - output_start_x + 1;
    auto op_size_y = output_end_y - output_start_y + 1;
    auto op_size_z = output_end_z - output_start_z + 1;

    auto workload_id = op.getWorkloadId().value_or(-1);
    if (workload_id < 0) {
        registers.offset_addr.offset_addr_bf.idu_stat_en = 0;
        registers.offset_addr.offset_addr_bf.odu_stat_en = 0;
    } else {
        registers.offset_addr.offset_addr_bf.idu_stat_en = 1;
        registers.offset_addr.offset_addr_bf.odu_stat_en = 1;
    }

    registers.weight_num = op_size_z;

    auto task_type = invariant.getNceTaskType();
    switch (task_type) {
    case VPUIP::NCETaskType::CONV:
        if (inputRef->dimensions()->Get(Z) < 16) {
            registers.weight_size = checked_cast<uint32_t>(16 * kernelSizeW * kernelSizeH);
            registers.weight_num = weightsRef->dimensions()->Get(B);
        } else {
            registers.weight_size = inputRef->dimensions()->Get(Z) * checked_cast<uint32_t>(kernelSizeW * kernelSizeH);
        }
        break;
    case VPUIP::NCETaskType::DWCONV:
    case VPUIP::NCETaskType::AVEPOOL:
    case VPUIP::NCETaskType::MAXPOOL:
        registers.weight_size = checked_cast<uint32_t>(op_size_z * kernelSizeW * kernelSizeH);
        break;
    case VPUIP::NCETaskType::ELTWISE:
        registers.weight_size =
                inputRef->dimensions()->Get(X) * inputRef->dimensions()->Get(Y) * inputRef->dimensions()->Get(Z);
        break;
    default:
        VPUX_THROW("Can't setup weight size. Layer type unknown : {0}", task_type);
        return;
    }

    registers.offset_addr.offset_addr_bf.dense_se = inputRef->data()->storage_element_index() != DEFAULT_INDEX ? 0 : 1;

    registers.offset_addr.offset_addr_bf.conv_cond = invariant.getIsContinued();

    registers.workload_size0.workload_size0_bf.workload_size_x =
            kernelStridesW * (op_size_x - 1) + kernelSizeW - local_PL - local_PR;
    registers.workload_size0.workload_size0_bf.workload_size_y =
            kernelStridesH * (op_size_y - 1) + kernelSizeH - local_PT - local_PB;
    registers.workload_size1.workload_size1_bf.workload_size_z = ConfigWorkloadSize(inputRef, task_type, op_size_z);
    registers.workload_size1.workload_size1_bf.pad_count_up = local_PT;
    registers.workload_size1.workload_size1_bf.pad_count_down = local_PB;
    registers.workload_size1.workload_size1_bf.pad_count_left = local_PL;
    registers.workload_size1.workload_size1_bf.pad_count_right = local_PR;
    registers.workload_start0.workload_start0_bf.workload_start_x =
            (output_start_x * kernelStridesW) - kernelPadL + local_PL;
    ;
    registers.workload_start0.workload_start0_bf.workload_start_y =
            (output_start_y * kernelStridesH) - kernelPadT + local_PT;
    registers.workload_start1.workload_start1_bf.workload_start_z = ConfigWorkloadStart(task_type, output_start_z);

    registers.te_beg1.te_beg1_bf.te_beg_x = output_start_x;
    registers.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
    registers.te_beg0.te_beg0_bf.te_beg_z = output_start_z;

    registers.te_end1.te_end1_bf.te_end_x = output_end_x;
    registers.te_end0.te_end0_bf.te_end_y = output_end_y;
    registers.te_end0.te_end0_bf.te_end_z = output_end_z;

    taskWrapper.weight_table_offset_ = output_start_z;

    if (task_type == VPUIP::NCETaskType::DWCONV || task_type == VPUIP::NCETaskType::MAXPOOL ||
        task_type == VPUIP::NCETaskType::AVEPOOL) {
        registers.workload_start1.workload_start1_bf.workload_start_z = output_start_z;
        registers.workload_size1.workload_size1_bf.workload_size_z = op_size_z;

    } else if (parentInputRef->dimensions()->Get(Z) < 16) {
        registers.workload_start1.workload_start1_bf.workload_start_z = 0;
        registers.workload_size1.workload_size1_bf.workload_size_z = 16;
    } else {
        // All input channels required for one output channel
        registers.workload_start1.workload_start1_bf.workload_start_z = 0;
        registers.workload_size1.workload_size1_bf.workload_size_z = inputRef->dimensions()->Get(Z);
    }

    // Split over K, and also streaming over K for now ....
    // ODU has a view of the full output tensor, yet as an optimization
    // in each cluster we bring weights and weight_table portions for each
    // output channel subset we compute in that particular cluster
    if (outputRef->dimensions()->Get(Z) != parentOutputRef->dimensions()->Get(Z)) {
        // Fathom style split logic. out_channel_offset may be set when it is not needed
        // Split calculation logic may not be correct, in flux from Fathom

        if (invariant.getOutputBuffs().size() > 1) {
            auto wto = (output_start_z - invariant.getOutChannelOffset().value_or(0)) % outputRef->dimensions()->Get(Z);
            taskWrapper.weight_table_offset_ = checked_cast<uint32_t>(wto);
        }
    }

    // Point into the 16 byte weight table entry, corresponding to the output channels subset
    taskWrapper.weight_table_offset_ = taskWrapper.weight_table_offset_ << 4;

    op_size_y = SetupVariant_SOH(invariant, op, outputRef, parentOutputRef, taskWrapper, inputClusterCount);

    SetupVariant_NTHW_NTK(invariant, registers);

    registers.offset_addr.offset_addr_bf.swizzle_key = inputRef->swizzling_key();

    registers.offset_addr.offset_addr_bf.wt_swizzle_key = weightsRef ? weightsRef->swizzling_key() : 0;
    registers.offset_addr.offset_addr_bf.wt_swizzle_sel = 1; /** use separate swizzle key for weights */

    registers.offset_addr.offset_addr_bf.shave_l2_cache_en = 1;
}
