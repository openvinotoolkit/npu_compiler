//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Operation.h>

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/composers/dma_composer.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>
#include <npu_40xx_nnrt.hpp>

namespace vpux {
namespace NPUReg40XX {

using namespace Descriptors;
using namespace npu40xx;

namespace {

bool isWorkLoadManagementDMA(mlir::Operation* op) {
    return mlir::isa<VPUASM::DPUInvariantOp, VPUASM::DPUVariantOp, VPUIPDPU::DPUInvariantOp, VPUIPDPU::DPUVariantOp,
                     VPUASM::ActKernelInvocationOp, VPUASM::ActKernelRangeOp, VPUASM::DeclareTaskBufferOp>(op);
}

uint64_t getTensorMode(mlir::Type type) {
    if (auto quantized = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        return getTensorMode(quantized.getStorageType());
    }

    if (type.isSignedInteger() || type.isUnsignedInteger() || type.isSignlessInteger()) {
        return DMA_ACC_DTYPE_INT8_UINT8;
    } else {
        return DMA_ACC_DTYPE_FP16_BF16;
    }

    VPUX_THROW("Invalid tensor type for DMA Acceleration configuration {0}", type);
}

void setDMAConversionMode(DMARegister& initValues, mlir::Type inputType, uint64_t srcSize, mlir::Type outputType,
                          uint64_t dstSize) {
    uint64_t conversionCfg = 0;
    if (inputType != outputType) {
        if (inputType.isF32() && outputType.isF16()) {
            conversionCfg = DMA_DATA_CONV_FP32_FP16;
        } else if (inputType.isF32() && outputType.isBF16()) {
            conversionCfg = DMA_DATA_CONV_FP32_BF16;
        } else {
            VPUX_THROW("Unsupported DMA data conversion");
        }

        VPUX_THROW_WHEN(dstSize != (srcSize / 2), "Source and destination length do not match");
    }

    initValues.write<Fields::dma_cfg_fields_conversion_cfg>(conversionCfg);
}

uint32_t getActCompressionEntryTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap) {
    auto actCompressionSizeEntry = dmaOp.getActCompressionSizeEntry();
    if (actCompressionSizeEntry.has_value()) {
        auto actCompBufferRef = symRefMap.lookupSymbol(actCompressionSizeEntry.value());
        VPUX_THROW_UNLESS(actCompBufferRef, "Could not find symbol name entry for {0} of {1}",
                          actCompressionSizeEntry.value(), dmaOp);

        if (mlir::isa<VPUASM::DeclareBufferOp>(actCompBufferRef)) {
            auto actCompBuffer = mlir::cast<VPUASM::DeclareBufferOp>(actCompBufferRef);
            return NPUReg40XX::getTileSelectMaskForBuffer(actCompBuffer);
        }
    }
    return 0;
}

void setDMAAccelerationCompress(DMARegister& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType inputType,
                                mlir::MemRefType outputType, ELF::SymbolReferenceMap& symRefMap) {
    const auto dmaDescriptor = origOp.getDmaDescriptorAttr();
    VPUX_THROW_UNLESS(dmaDescriptor, "NNDMAOp missing DMADescriptorAttr");
    const auto srcWidth = dmaDescriptor.getSrcWidth().getInt();
    const auto dstWidth = outputType.cast<vpux::NDTypeInterface>().getTotalAllocSize().count();

    const auto uncompressedBufSize = inputType.cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
    VPUX_THROW_UNLESS(uncompressedBufSize > ACT_COMPRESSION_MIN_BUF_SIZE,
                      "Uncompressed buffer size '{0}' needs to be larger then '{1}'", uncompressedBufSize,
                      ACT_COMPRESSION_MIN_BUF_SIZE);

    initValues.write<Fields::dma_width_src>(srcWidth);
    initValues.write<Fields::dma_width_dst>(dstWidth);

    if (origOp.getActCompressionSizeEntry().has_value()) {
        initValues.write<Fields::dma_cfg_fields_rws_en>(true);
        initValues.write<Fields::dma_remote_width_store>(getActCompressionEntryTileMask(origOp, symRefMap));
    }

    initValues.write<Fields::dma_cfg_fields_acceleration_cfg>(DMA_ACCEL_COMPRESS);
    initValues.write<Fields::dma_acc_info_compress_dtype>(getTensorMode(inputType.getElementType()));
    initValues.write<Fields::dma_acc_info_compress_bitc_en>(1);
}

void setDMAAccelerationDecompress(DMARegister& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType outputType,
                                  ELF::SymbolReferenceMap& symRefMap) {
    auto actCompressionSizeEntry = origOp.getActCompressionSizeEntry();
    if (actCompressionSizeEntry.has_value()) {
        initValues.write<Fields::dma_cfg_fields_rwf_en>(true);
        initValues.write<Fields::dma_remote_width_fetch>(getActCompressionEntryTileMask(origOp, symRefMap));
    }

    initValues.write<Fields::dma_cfg_fields_acceleration_cfg>(DMA_ACCEL_DECOMPRESS);
    initValues.write<Fields::dma_acc_info_decompress_dtype>(getTensorMode(outputType.getElementType()));
    initValues.write<Fields::dma_acc_info_decompress_bitc_en>(1);
}

void setDMAAccelerationMode(DMARegister& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType inputType,
                            mlir::MemRefType outputType, ELF::SymbolReferenceMap& symRefMap) {
    auto accMode = origOp.getAccelerationMode();
    switch (accMode) {
    case VPUIP::DMAAccMode::DISABLE:
        // nothing to do
        break;
    case VPUIP::DMAAccMode::COMPRESSION:
        setDMAAccelerationCompress(initValues, origOp, inputType, outputType, symRefMap);
        break;
    case VPUIP::DMAAccMode::DECOMPRESSION:
        setDMAAccelerationDecompress(initValues, origOp, outputType, symRefMap);
        break;
    default:
        VPUX_THROW("{0} acceleration mode is not supported", accMode);
        break;
    }
}

void setEnableMemorySideCaching(DMARegister& initValues) {
    initValues.write<Fields::dma_src_aub>(DMA_AUB_SRC_DST);
    initValues.write<Fields::dma_dst_aub>(DMA_AUB_SRC_DST);
    initValues.write<Fields::dma_cfg_fields_axi_user_bits_cfg>(DMA_AUB_SRC_DST);
}

void setGatherMode(ELF::SymbolReferenceMap& symRefMap, const ::mlir::SymbolRefAttr& indices, DMARegister& initValues,
                   const mlir::MemRefType& outputType, Bit elemOutSize) {
    mlir::MemRefType indicesType;
    auto indicesBufferRep = symRefMap.lookupSymbol(indices);
    if (mlir::isa<VPUASM::DeclareBufferOp>(indicesBufferRep)) {
        auto indicesBuffer = mlir::cast<VPUASM::DeclareBufferOp>(indicesBufferRep);
        indicesType = indicesBuffer.getBufferType().getMemref();
    }
    // DMA copies data block by block example here with
    // input 50257x768xf32 indices 1024xi64
    // output 1024x768xf32 so dma will copy 1024 blocks of 768xsizeof(f32) blocks so elementsize here would be 3072
    // We dont need axis info here as gather cant be done here if indices are pointing so leftmost dimension who has
    // size bigger than 1. There condition will be enforced on upper level dialects.
    const auto dma_element_size =
            (outputType.getNumElements() / indicesType.getNumElements()) * elemOutSize.to<Byte>().count();

    initValues.write<Fields::dma_cfg_fields_src_list_cfg>(DMA_LIST_REL_INDEX);
    initValues.write<Fields::dma_cfg_fields_dst_list_cfg>(0);
    initValues.write<Fields::dma_list_size_src>(indicesType.getNumElements());
    initValues.write<Fields::dma_stride_dst_1>(dma_element_size);
    initValues.write<Fields::dma_width_src>(dma_element_size);
    initValues.write<Fields::dma_dim_size_dst_1>(0);
}

}  // namespace

namespace DMADescriptorComposer {

DMATransactionConfig configurePatternFromDescriptorAttr(VPUIP::DMADescriptorAttr& descriptor) {
    DMATransactionConfig transactionConfig{};

    auto numPlanes = descriptor.getNumPlanes().getInt();
    auto length = descriptor.getLen().getInt();
    auto srcWidth = descriptor.getSrcWidth().getInt();
    auto srcStride = descriptor.getSrcStride().getInt();
    auto srcPlaneStride = descriptor.getSrcPlaneStride().getInt();
    auto dstWidth = descriptor.getDstWidth().getInt();
    auto dstStride = descriptor.getDstStride().getInt();
    auto dstPlaneStride = descriptor.getDstPlaneStride().getInt();

    auto srcDimSize1 = srcWidth ? static_cast<int64_t>((length / srcWidth) - 1) : 0;
    auto dstDimSize1 = (dstWidth && (length > dstWidth)) ? static_cast<int64_t>((length / dstWidth) - 1) : 0;

    int64_t numDims = 0;
    if (numPlanes > 1) {
        numDims = DMA_3D;
    } else if (srcWidth == srcStride && dstWidth == dstStride) {
        numDims = DMA_1D;
    } else {
        numDims = DMA_2D;
    }
    transactionConfig.numDims = numDims;

    switch (numDims) {
    case DMA_3D:
        VPUX_THROW_WHEN(numPlanes == 0, "numPlanes cannot be 0 for a 3D transaction");
        transactionConfig.srcDimSizes[2] = numPlanes - 1;
        transactionConfig.dstDimSizes[2] = numPlanes - 1;

        transactionConfig.srcStrides[2] = srcPlaneStride;
        transactionConfig.dstStrides[2] = dstPlaneStride;

        [[fallthrough]];
    case DMA_2D:
        transactionConfig.srcDimSizes[1] = srcDimSize1;
        transactionConfig.dstDimSizes[1] = dstDimSize1;

        transactionConfig.srcStrides[1] = srcStride;
        transactionConfig.dstStrides[1] = dstStride;

        [[fallthrough]];
    case DMA_1D:
        transactionConfig.srcDimSizes[0] = srcWidth;
        transactionConfig.dstDimSizes[0] = dstWidth;
        break;
    default:
        VPUX_THROW("Error at configureTransaction. Unsupported numDims={0}", numDims);
        break;
    }

    return transactionConfig;
}

DMATransactionConfig configurePatternFromTransactionAttr(DMATransaction& transaction) {
    DMATransactionConfig transactionConfig{};

    VPUX_THROW_WHEN(transaction.inputs.size() != 1, "DMA transaction with unsupported number of input patterns");
    VPUX_THROW_WHEN(transaction.outputs.size() != 1, "DMA transaction with unsupported number of output patterns");

    auto& inputPattern = transaction.inputs.front();
    auto& outputPattern = transaction.outputs.front();

    auto checkPatternComponent = [&](auto& input, auto& result) {
        VPUX_THROW_WHEN(input.size() == 0, "DMA pattern conversion check failure");
        VPUX_THROW_WHEN(input.size() > result.size(), "DMA pattern conversion check failure");
    };

    checkPatternComponent(inputPattern.dims, transactionConfig.srcDimSizes);
    checkPatternComponent(inputPattern.strides, transactionConfig.srcStrides);
    checkPatternComponent(outputPattern.dims, transactionConfig.dstDimSizes);
    checkPatternComponent(outputPattern.strides, transactionConfig.dstStrides);

    VPUX_THROW_WHEN(inputPattern.dims.size() != inputPattern.strides.size(),
                    "Mismatch between pattern dim count and stride count");
    VPUX_THROW_WHEN(outputPattern.dims.size() != outputPattern.strides.size(),
                    "Mismatch between pattern dim count and stride count");

    // Pattern layout
    // ________________________________
    // Index      || 0  | 1  | 2  | 3  |
    // Dim        || d3 | d2 | d1 | d0 |
    // Stride     || s3 | s2 | s1 | s0 |
    //                ^                |
    //          highest rank           |
    // ________________________________|
    //

    // DMA layout
    // ________________________________
    // Index      || 0  | 1  | 2  | 3  |
    // Dim        || d0 | d1 | d2 | d3 |
    // Stride     || 0  | s0 | s1 | s2 |
    // ________________________________|

    // Reverse dims and strides from memref order to DMA order
    std::copy(inputPattern.dims.rbegin(), inputPattern.dims.rend(), transactionConfig.srcDimSizes.begin());
    std::copy(inputPattern.strides.rbegin(), inputPattern.strides.rend() - 1, transactionConfig.srcStrides.begin() + 1);
    std::copy(outputPattern.dims.rbegin(), outputPattern.dims.rend(), transactionConfig.dstDimSizes.begin());
    std::copy(outputPattern.strides.rbegin(), outputPattern.strides.rend() - 1,
              transactionConfig.dstStrides.begin() + 1);

    const auto minOne = [](auto& val) {
        val > 1 ? val -= 1 : val = 0;
    };

    // Subtract 1 from all dims except the first one, as required by the corresponding registers
    std::for_each(transactionConfig.srcDimSizes.begin() + 1, transactionConfig.srcDimSizes.end(), minOne);
    std::for_each(transactionConfig.dstDimSizes.begin() + 1, transactionConfig.dstDimSizes.end(), minOne);

    transactionConfig.numDims = std::max(inputPattern.dims.size(), outputPattern.dims.size()) - 1;

    return transactionConfig;
}

DMARegister compose(VPUASM::NNDMAOp origOp, ELF::SymbolReferenceMap& symRefMap) {
    Descriptors::DMARegister descriptor;
    // VPUASM ops already contain information about input/output buffers in `dma_descriptor` field
    // we should use it instead of looking related memref's by sym names
    // TODO: E#73178
    auto inputBufferRef = symRefMap.lookupSymbol(origOp.getInput());
    VPUX_THROW_UNLESS(inputBufferRef, "Could not find symbol name entry for {0} of {1}", origOp.getInput(), origOp);
    mlir::MemRefType inputType;

    uint32_t inputTileMask = 0;
    bool isDMAInputForWLMDMA = false;
    if (mlir::isa<VPUASM::DeclareBufferOp>(inputBufferRef)) {
        auto inputBuffer = mlir::cast<VPUASM::DeclareBufferOp>(inputBufferRef);
        inputType = inputBuffer.getBufferType().getMemref();
        inputTileMask = vpux::NPUReg40XX::getTileSelectMaskForBuffer(inputBuffer);
    } else if (mlir::isa<VPUASM::ConstBufferOp>(inputBufferRef)) {
        auto inputBuffer = mlir::cast<VPUASM::ConstBufferOp>(inputBufferRef);
        inputType = inputBuffer.getBufferType().getMemref();
    } else if (isWorkLoadManagementDMA(inputBufferRef)) {
        isDMAInputForWLMDMA = true;
    } else {
        VPUX_THROW("Could not find symbol name entry for {0}", origOp.getInput());
    }

    auto broadcastTileMask = uint32_t{0};
    if (origOp.getTileIndexes().has_value()) {
        broadcastTileMask = VPUMI40XX::generateTileMask(parseIntArrayAttr<uint32_t>(origOp.getTileIndexes().value()));
    }

    const int barrierEn = 1;
    const int ord = !origOp.getIsOutOfOrder();

    uint32_t linkAddressTileMask = 0;
    if (origOp.getNextLink().has_value()) {
        auto nextDMARef = symRefMap.lookupSymbol(origOp.getNextLink().value());

        auto nextDMATaskBuffer = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(nextDMARef);
        if (nextDMATaskBuffer) {
            linkAddressTileMask = NPUReg40XX::getTileSelectMaskForBuffer(nextDMATaskBuffer);
        }
    }

    DMATransactionConfig npu4config{};
    if (auto transactionAttr = origOp.getDmaTransactionAttr()) {
        auto transaction = transactionAttr.getDMATransaction();
        npu4config = configurePatternFromTransactionAttr(transaction);
    } else {
        if (auto descriptorAttr = origOp.getDmaDescriptorAttr()) {
            npu4config = configurePatternFromDescriptorAttr(descriptorAttr);
        } else {
            VPUX_THROW("Transaction cannot be composed without both transaction and descriptor attributes");
        }
    }

    descriptor.write<Fields::dma_cfg_fields_num_dim>(npu4config.numDims);
    descriptor.write<Fields::dma_cfg_fields_barrier_en>(barrierEn);
    descriptor.write<Fields::dma_cfg_fields_atp_en>(1);
    descriptor.write<Fields::dma_cfg_fields_src_burst_length>(15);
    descriptor.write<Fields::dma_cfg_fields_dst_burst_length>(15);
    descriptor.write<Fields::dma_cfg_fields_arb_qos>(255);
    descriptor.write<Fields::dma_cfg_fields_ord>(ord);
    descriptor.write<Fields::dma_cfg_fields_hwp_id_en>(1);
    descriptor.write<Fields::dma_cfg_fields_hwp_id>(origOp.getDmaHwpId().value_or(0));

    descriptor.write<Fields::dma_dim_size_src_1>(npu4config.srcDimSizes[1]);
    descriptor.write<Fields::dma_dim_size_src_2>(npu4config.srcDimSizes[2]);
    descriptor.write<Fields::dma_dim_size_src_3>(npu4config.srcDimSizes[3]);
    descriptor.write<Fields::dma_dim_size_src_4>(npu4config.srcDimSizes[4]);
    descriptor.write<Fields::dma_dim_size_src_5>(npu4config.srcDimSizes[5]);

    descriptor.write<Fields::dma_dim_size_dst_1>(npu4config.dstDimSizes[1]);
    descriptor.write<Fields::dma_dim_size_dst_2>(npu4config.dstDimSizes[2]);
    descriptor.write<Fields::dma_dim_size_dst_3>(npu4config.dstDimSizes[3]);
    descriptor.write<Fields::dma_dim_size_dst_4>(npu4config.dstDimSizes[4]);
    descriptor.write<Fields::dma_dim_size_dst_5>(npu4config.dstDimSizes[5]);

    descriptor.write<Fields::dma_stride_src_1>(npu4config.srcStrides[1]);
    descriptor.write<Fields::dma_stride_src_2>(npu4config.srcStrides[2]);
    descriptor.write<Fields::dma_stride_src_3>(npu4config.srcStrides[3]);
    descriptor.write<Fields::dma_stride_src_4>(npu4config.srcStrides[4]);
    descriptor.write<Fields::dma_stride_src_5>(npu4config.srcStrides[5]);

    descriptor.write<Fields::dma_stride_dst_1>(npu4config.dstStrides[1]);
    descriptor.write<Fields::dma_stride_dst_2>(npu4config.dstStrides[2]);
    descriptor.write<Fields::dma_stride_dst_3>(npu4config.dstStrides[3]);
    descriptor.write<Fields::dma_stride_dst_4>(npu4config.dstStrides[4]);
    descriptor.write<Fields::dma_stride_dst_5>(npu4config.dstStrides[5]);

    descriptor.write<Fields::dma_src>(inputTileMask);
    descriptor.write<Fields::dma_dst>(broadcastTileMask);
    descriptor.write<Fields::dma_barrier_prod_mask_lower>(vpux::VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers()));
    descriptor.write<Fields::dma_barrier_cons_mask_lower>(vpux::VPUMI40XX::computeMaskLo(origOp.getWaitBarriers()));
    descriptor.write<Fields::dma_barrier_prod_mask_upper>(vpux::VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers()));
    descriptor.write<Fields::dma_barrier_cons_mask_upper>(vpux::VPUMI40XX::computeMaskHi(origOp.getWaitBarriers()));
    descriptor.write<Fields::dma_link_address>(linkAddressTileMask);
    descriptor.write<Registers::dma_barriers_sched, Fields::start_after_>(origOp.getStartAfter());
    descriptor.write<Registers::dma_barriers_sched, Fields::clean_after_>(origOp.getCleanAfter());

    // In the case of SyncDMA (a DMA that does nothing),
    // the compiler must set an extra bit in dma_cfg to indicate that this DMA is not real,
    // and the SRC_ADDR should be ignored. More details in E152711
    if (npu4config.srcDimSizes[0] == 0 && npu4config.srcDimSizes[0] == npu4config.dstDimSizes[0]) {
        descriptor.write<Fields::dma_cfg_fields_memset_en>(1);
    }

    if (auto enableMemorySideCaching = origOp.getEnableMscAttr()) {
        setEnableMemorySideCaching(descriptor);
    }

    auto actCompFlag = origOp.getActCompressionSizeEntry().has_value();
    auto accMode = origOp.getAccelerationMode();
    if (!actCompFlag) {
        // dma_width register conflicts with remote_width_fetch and should not be programmed in case of decompression
        // In case of compression it should not be programmed because dstWidth requires adjustment for worst case size
        descriptor.write<Fields::dma_width_src>(npu4config.srcDimSizes[0]);
        descriptor.write<Fields::dma_width_dst>(npu4config.dstDimSizes[0]);
    }

    if (!actCompFlag || accMode != vpux::VPUIP::DMAAccMode::COMPRESSION) {
        // dma_stride_dst_2 register conflicts with remote_width_store and should not be programmed in case of
        // compression
        descriptor.write<Fields::dma_stride_dst_2>(npu4config.dstStrides[2]);
    }

    if (!isDMAInputForWLMDMA) {
        auto outputBufferSym = origOp.getOutputBuffs()[0].dyn_cast_or_null<mlir::SymbolRefAttr>();
        VPUX_THROW_UNLESS(outputBufferSym, "`output_buffs` attribute should contain SymbolRefAttr but it doesn't");

        auto outputBufferRef = symRefMap.lookupSymbol(outputBufferSym);
        auto outputBuffer = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(outputBufferRef);
        VPUX_THROW_UNLESS(outputBuffer, "Could not find symbol name entry for {0}", outputBufferRef);
        auto outputType = outputBuffer.getBufferType().getMemref();

        const auto elemInSize = vpux::getElemTypeSize(inputType);
        const auto elemOutSize = vpux::getElemTypeSize(outputType);

        auto totalInSizeBits = alignMemSize(inputType.getNumElements() * elemInSize, Byte(1));
        auto totalOutSizeBits = alignMemSize(outputType.getNumElements() * elemOutSize, Byte(1));

        // DMA only does FP32 -> FP16/BF16 conversions,
        // Because of this, dstDimSize1 will always be half of the original value
        if (inputType.getElementType() != outputType.getElementType() && npu4config.dstDimSizes[1]) {
            VPUX_THROW_UNLESS(elemInSize == elemOutSize * 2, "Element sizes in conversion are not supported");
            long newDstDimSize1 = ((npu4config.dstDimSizes[1] + 1) / 2) - 1;
            descriptor.write<Fields::dma_dim_size_dst_1>(newDstDimSize1);
        }

        if (accMode != vpux::VPUIP::DMAAccMode::DISABLE) {
            VPUX_THROW_WHEN(npu4config.srcDimSizes[1] != 0 || npu4config.dstDimSizes[1] != 0 ||
                                    npu4config.srcDimSizes[2] != 0 || npu4config.dstDimSizes[2] != 0 ||
                                    npu4config.srcStrides[2] != 0 || npu4config.dstStrides[2] != 0,
                            "Activation compression is supported only for 1D DMAs");
            setDMAAccelerationMode(descriptor, origOp, inputType, outputType, symRefMap);
        } else {
            // Conversion
            setDMAConversionMode(descriptor, inputType.getElementType(), totalInSizeBits.to<Byte>().count(),
                                 outputType.getElementType(), totalOutSizeBits.to<Byte>().count());
        }

        auto indices = origOp.getIndices();
        if (indices.has_value()) {
            setGatherMode(symRefMap, indices.value(), descriptor, outputType, elemOutSize);
        }
    }

    return descriptor;
}

}  // namespace DMADescriptorComposer
}  // namespace NPUReg40XX
}  // namespace vpux
