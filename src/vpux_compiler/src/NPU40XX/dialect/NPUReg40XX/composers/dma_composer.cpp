//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Operation.h>

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/composers/dma_composer.hpp"
#include "vpux/compiler/dialect/VPUASM/dma_transaction.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux {
namespace NPUReg40XX {

using namespace Descriptors;
using namespace npu40xx;

namespace {

uint64_t getTensorMode(mlir::Type type) {
    if (auto quantized = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        return getTensorMode(quantized.getStorageType());
    }

    if (type.isSignedInteger() || type.isUnsignedInteger() || type.isSignlessInteger()) {
        return DMA_ACC_DTYPE_INT8_UINT8;
    } else {
        return DMA_ACC_DTYPE_FP16_BF16;
    }
}

void setDMAConversionMode(DMARegister& initValues, mlir::Type inputType, uint64_t srcSize, mlir::Type outputType,
                          uint64_t dstSize) {
    uint64_t conversionCfg = 0;
    auto isQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputType) &&
                           mlir::dyn_cast<mlir::quant::QuantizedType>(outputType);
    if (inputType != outputType && !isQuantizedType) {
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

void setDMAAccelerationCompress(DMARegister& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType inputType,
                                mlir::MemRefType outputType, ELF::SymbolReferenceMap& symRefMap) {
    const auto dmaDescriptor = origOp.getDmaDescriptorAttr();
    VPUX_THROW_UNLESS(dmaDescriptor, "NNDMAOp missing DMADescriptorAttr");
    const auto srcWidth = dmaDescriptor.getSrcWidth().getInt();
    const auto dstWidth = mlir::cast<vpux::NDTypeInterface>(outputType).getTotalAllocSize().count();

    const auto uncompressedBufSize = mlir::cast<vpux::NDTypeInterface>(inputType).getTotalAllocSize().count();
    VPUX_THROW_UNLESS(uncompressedBufSize > ACT_COMPRESSION_MIN_BUF_SIZE,
                      "Uncompressed buffer size '{0}' needs to be larger than '{1}'", uncompressedBufSize,
                      ACT_COMPRESSION_MIN_BUF_SIZE);

    initValues.write<Fields::dma_width_src>(srcWidth);
    initValues.write<Fields::dma_width_dst>(dstWidth);

    if (origOp.getActCompressionSizeEntry().has_value()) {
        initValues.write<Fields::dma_cfg_fields_rws_en>(true);
        initValues.write<Fields::dma_remote_width_store>(VPUASM::getActCompressionEntryTileMask(origOp, symRefMap));
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
        initValues.write<Fields::dma_remote_width_fetch>(VPUASM::getActCompressionEntryTileMask(origOp, symRefMap));
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

void setGatherMode(VPUASM::NNDMAOp origOp, ELF::SymbolReferenceMap& symRefMap, const ::mlir::SymbolRefAttr& indices,
                   DMARegister& initValues, const mlir::MemRefType& outputType, Bit elemOutSize) {
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

    auto addressingMode = origOp.getAddressingMode().has_value() ? origOp.getAddressingMode().value()
                                                                 : VPUIP::GatherAddressingMode::INDEXED;
    switch (addressingMode) {
    case VPUIP::GatherAddressingMode::ABSOLUTE:
        initValues.write<Fields::dma_cfg_fields_src_list_cfg>(DMA_LIST_ABS_INDEX);
        break;
    case VPUIP::GatherAddressingMode::INDEXED:
    default:
        initValues.write<Fields::dma_cfg_fields_src_list_cfg>(DMA_LIST_REL_INDEX);
        break;
    }
    initValues.write<Fields::dma_cfg_fields_dst_list_cfg>(0);
    initValues.write<Fields::dma_list_size_src>(indicesType.getNumElements());
    initValues.write<Fields::dma_stride_dst_1>(dma_element_size);
    initValues.write<Fields::dma_width_src>(dma_element_size);
    initValues.write<Fields::dma_dim_size_dst_1>(0);
}

}  // namespace

namespace DMADescriptorComposer {

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
        inputTileMask = VPUASM::getTileSelectMaskForBuffer(inputBuffer);
    } else if (mlir::isa<VPUASM::ConstBufferOp>(inputBufferRef)) {
        auto inputBuffer = mlir::cast<VPUASM::ConstBufferOp>(inputBufferRef);
        inputType = inputBuffer.getBufferType().getMemref();
    } else if (VPUASM::isWorkLoadManagementDMA(inputBufferRef)) {
        isDMAInputForWLMDMA = true;
    } else {
        VPUX_THROW("Could not find symbol name entry for {0}", origOp.getInput());
    }

    auto broadcastTileMask = uint32_t{0};
    auto outputRef = mlir::cast<mlir::SymbolRefAttr>(origOp.getOutputBuffsAttr()[0]);
    auto outputBuffRef = symRefMap.lookupSymbol(outputRef);
    if (mlir::isa<VPUASM::DeclareBufferOp>(outputBuffRef)) {
        auto outputBuffer = mlir::cast<VPUASM::DeclareBufferOp>(outputBuffRef);
        const auto section = outputBuffer.getBufferType().getLocation().getSection();
        if (section == VPURT::BufferSection::Register) {
            broadcastTileMask = outputBuffer.getMemoryOffset();
        }
    }

    // E#141779 Once this is resolved we can remove the selective tile mask application
    if (origOp.getTileIndexes().has_value() && broadcastTileMask == 0) {
        broadcastTileMask = VPUMI40XX::generateTileMask(parseIntArrayAttr<uint32_t>(origOp.getTileIndexes().value()));
    }

    const int barrierEn = 1;
    const int ord = !origOp.getIsOutOfOrder();

    uint32_t linkAddressTileMask = 0;
    if (origOp.getNextLink().has_value()) {
        auto nextDMARef = symRefMap.lookupSymbol(origOp.getNextLink().value());

        auto nextDMATaskBuffer = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(nextDMARef);
        if (nextDMATaskBuffer) {
            linkAddressTileMask = VPUASM::getTileSelectMaskForBuffer(nextDMATaskBuffer);
        }
    }

    auto transactionConfig = VPUASM::getDMATransactionConfig(origOp);

    descriptor.write<Fields::dma_cfg_fields_num_dim>(transactionConfig.numDims);
    descriptor.write<Fields::dma_cfg_fields_barrier_en>(barrierEn);
    descriptor.write<Fields::dma_cfg_fields_atp_en>(1);
    descriptor.write<Fields::dma_cfg_fields_src_burst_length>(15);
    descriptor.write<Fields::dma_cfg_fields_dst_burst_length>(15);
    descriptor.write<Fields::dma_cfg_fields_arb_qos>(255);
    descriptor.write<Fields::dma_cfg_fields_ord>(ord);
    descriptor.write<Fields::dma_cfg_fields_hwp_id_en>(1);
    descriptor.write<Fields::dma_cfg_fields_hwp_id>(origOp.getDmaHwpId().value_or(0));

    descriptor.write<Fields::dma_dim_size_src_1>(transactionConfig.srcDimSizes[1]);
    descriptor.write<Fields::dma_dim_size_src_2>(transactionConfig.srcDimSizes[2]);
    descriptor.write<Fields::dma_dim_size_src_3>(transactionConfig.srcDimSizes[3]);
    descriptor.write<Fields::dma_dim_size_src_4>(transactionConfig.srcDimSizes[4]);
    descriptor.write<Fields::dma_dim_size_src_5>(transactionConfig.srcDimSizes[5]);

    descriptor.write<Fields::dma_dim_size_dst_1>(transactionConfig.dstDimSizes[1]);
    descriptor.write<Fields::dma_dim_size_dst_2>(transactionConfig.dstDimSizes[2]);
    descriptor.write<Fields::dma_dim_size_dst_3>(transactionConfig.dstDimSizes[3]);
    descriptor.write<Fields::dma_dim_size_dst_4>(transactionConfig.dstDimSizes[4]);
    descriptor.write<Fields::dma_dim_size_dst_5>(transactionConfig.dstDimSizes[5]);

    descriptor.write<Fields::dma_stride_src_1>(transactionConfig.srcStrides[1]);
    descriptor.write<Fields::dma_stride_src_2>(transactionConfig.srcStrides[2]);
    descriptor.write<Fields::dma_stride_src_3>(transactionConfig.srcStrides[3]);
    descriptor.write<Fields::dma_stride_src_4>(transactionConfig.srcStrides[4]);
    descriptor.write<Fields::dma_stride_src_5>(transactionConfig.srcStrides[5]);

    descriptor.write<Fields::dma_stride_dst_1>(transactionConfig.dstStrides[1]);
    descriptor.write<Fields::dma_stride_dst_2>(transactionConfig.dstStrides[2]);
    descriptor.write<Fields::dma_stride_dst_3>(transactionConfig.dstStrides[3]);
    descriptor.write<Fields::dma_stride_dst_4>(transactionConfig.dstStrides[4]);
    descriptor.write<Fields::dma_stride_dst_5>(transactionConfig.dstStrides[5]);

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
    if (transactionConfig.srcDimSizes[0] == 0 && transactionConfig.srcDimSizes[0] == transactionConfig.dstDimSizes[0]) {
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
        descriptor.write<Fields::dma_width_src>(transactionConfig.srcDimSizes[0]);
        descriptor.write<Fields::dma_width_dst>(transactionConfig.dstDimSizes[0]);
    }

    if (!actCompFlag || accMode != vpux::VPUIP::DMAAccMode::COMPRESSION) {
        // dma_stride_dst_2 register conflicts with remote_width_store and should not be programmed in case of
        // compression
        descriptor.write<Fields::dma_stride_dst_2>(transactionConfig.dstStrides[2]);
    }

    if (!isDMAInputForWLMDMA) {
        auto outputBuffs = origOp.getOutputBuffs();
        VPUX_THROW_WHEN(outputBuffs.empty(), "Output buffer is missing.");

        auto outputBufferSym = mlir::dyn_cast_or_null<mlir::SymbolRefAttr>(outputBuffs[0]);

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
        auto isQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputType) &&
                               mlir::dyn_cast<mlir::quant::QuantizedType>(outputType);
        if (inputType.getElementType() != outputType.getElementType() && !isQuantizedType &&
            transactionConfig.dstDimSizes[1]) {
            VPUX_THROW_UNLESS(elemInSize == elemOutSize * 2, "Element sizes in conversion are not supported");
            long newDstDimSize1 = ((transactionConfig.dstDimSizes[1] + 1) / 2) - 1;
            descriptor.write<Fields::dma_dim_size_dst_1>(newDstDimSize1);
        }

        if (accMode != vpux::VPUIP::DMAAccMode::DISABLE) {
            VPUX_THROW_WHEN(transactionConfig.srcDimSizes[1] != 0 || transactionConfig.dstDimSizes[1] != 0 ||
                                    transactionConfig.srcDimSizes[2] != 0 || transactionConfig.dstDimSizes[2] != 0 ||
                                    transactionConfig.srcStrides[2] != 0 || transactionConfig.dstStrides[2] != 0,
                            "Activation compression is supported only for 1D DMAs");
            setDMAAccelerationMode(descriptor, origOp, inputType, outputType, symRefMap);
        } else {
            // Conversion
            setDMAConversionMode(descriptor, inputType.getElementType(), totalInSizeBits.to<Byte>().count(),
                                 outputType.getElementType(), totalOutSizeBits.to<Byte>().count());
        }

        auto indices = origOp.getIndices();
        if (indices.has_value()) {
            setGatherMode(origOp, symRefMap, indices.value(), descriptor, outputType, elemOutSize);
        }
    }

    return descriptor;
}

}  // namespace DMADescriptorComposer
}  // namespace NPUReg40XX
}  // namespace vpux
