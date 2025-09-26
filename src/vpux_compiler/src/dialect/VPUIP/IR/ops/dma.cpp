//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Support/LLVM.h>

using namespace vpux;

namespace {

mlir::LogicalResult verifyTensorSize(config::ArchKind arch, mlir::Location loc, mlir::Value tensor) {
    const auto size = static_cast<Byte>(getCompactSize(tensor));

    auto maxTransferSize = VPUIP::DMA::getEngineLimits(arch).getTransferLimits().max();

    if (size <= maxTransferSize) {
        return mlir::success();
    }

    return errorAt(loc, "The size of the DMA transaction {0} for a {1} tensor is greater than the limit {2}", size,
                   getShape(tensor), maxTransferSize);
}

mlir::LogicalResult verifyInOutElementType(mlir::Location loc, mlir::Value inTensor, mlir::Value outTensor) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(inTensor.getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(outTensor.getType());

    if (inType.getElementType() != outType.getElementType()) {
        return errorAt(loc, "Input element type '{0}' doesn't match output element type '{1}'", inType.getElementType(),
                       outType.getElementType());
    }

    return mlir::success();
}

mlir::LogicalResult verifyCompressedBufferAllocSize(mlir::Location loc, mlir::Value origTensor,
                                                    mlir::Value compressedTensor, mlir::Value sparsityMapBuffer) {
    const auto origType = mlir::cast<vpux::NDTypeInterface>(origTensor.getType());
    const auto compressedType = mlir::cast<vpux::NDTypeInterface>(compressedTensor.getType());
    const auto origShape = origType.getShape();
    int64_t bitmapsize = 0;

    if (sparsityMapBuffer != nullptr) {
        const auto bitmapType = mlir::cast<vpux::NDTypeInterface>(sparsityMapBuffer.getType());
        bitmapsize = bitmapType.getTotalAllocSize().count();
    }

    const auto origSizeWithUpdateForCompression =
            updateSizeForCompression(origType.getTotalAllocSize().count(), origShape, bitmapsize);
    const auto compressedSize = compressedType.getTotalAllocSize().count();

    if (compressedSize < origSizeWithUpdateForCompression) {
        return errorAt(loc, "Compressed buffer size ('{0}' bytes) should be at least '{1}' bytes large", compressedSize,
                       origSizeWithUpdateForCompression);
    }

    return mlir::success();
}

}  // namespace

//
// NNDMAOp
//

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*compress_candidate=*/nullptr, /*dma_hwp_id=*/nullptr, /* profilingMetadata= */ nullptr,
          /*split_candidate=*/false, /*profiling_buffer_mgmt=*/false, /*fusionId=*/nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*compress_candidate=*/nullptr, /*dma_hwp_id=*/nullptr, /* profilingMetadata= */ nullptr,
          /*split_candidate=*/false, /*profiling_buffer_mgmt=*/false, /*fusionId=*/nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, int32_t dma_hwp_id) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*compress_candidate=*/nullptr, vpux::getIntAttr(builder, dma_hwp_id),
          /* profilingMetadata= */ nullptr, /*split_candidate=*/false, /*profiling_buffer_mgmt=*/false,
          /*fusionId=*/nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, mlir::IntegerAttr port, mlir::UnitAttr is_out_of_order,
                                 mlir::UnitAttr is_critical, mlir::IntegerAttr spillId) {
    build(builder, state, input, output_buff, /*port=*/port,
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate=*/nullptr, /*dma_hwp_id=*/nullptr,
          /* profilingMetadata= */ nullptr, /*split_candidate=*/nullptr, /*profiling_buffer_mgmt=*/nullptr,
          /*fusionId=*/nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, bool is_out_of_order, bool is_critical,
                                 mlir::IntegerAttr spillId, mlir::UnitAttr compress_candidate) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate=*/compress_candidate,
          /*dma_hwp_id=*/nullptr,
          /* profilingMetadata= */ nullptr, /*split_candidate=*/false, /*profiling_buffer_mgmt=*/false,
          /*fusionId=*/nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, bool is_out_of_order, bool is_critical,
                                 mlir::IntegerAttr spillId) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate=*/nullptr, /*dma_hwp_id=*/nullptr,
          /* profilingMetadata=*/nullptr, /*split_candidate=*/false, /*profiling_buffer_mgmt=*/false,
          /*fusionId=*/nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, bool is_out_of_order, bool is_critical,
                                 mlir::IntegerAttr spillId, mlir::UnitAttr compress_candidate, bool split_candidate,
                                 mlir::IntegerAttr fusionId) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate=*/compress_candidate,
          /*dma_hwp_id=*/nullptr,
          /* profilingMetadata=*/nullptr, split_candidate, /*profiling_buffer_mgmt=*/false, fusionId);
}

mlir::LogicalResult vpux::VPUIP::NNDMAOp::verify() {
    auto loc = getLoc();

    if (getCompressCandidateAttr() != nullptr) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
        auto outType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
        if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN && outType.getMemoryKind() == VPU::MemoryKind::DDR) {
            auto compressionState = getCompressionState(getOutput().getType());
            if (compressionState != VPUIP::CompressionState::CompressionCandidate) {
                return errorAt(loc, "NNDMA spill write must have compression candidate "
                                    "buffer as output");
            }
        } else if (inType.getMemoryKind() == VPU::MemoryKind::DDR &&
                   outType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            auto compressionState = getCompressionState(getInput().getType());
            if (compressionState != VPUIP::CompressionState::CompressionCandidate) {
                return errorAt(loc, "NNDMA spill read must have compression candidate "
                                    "buffer as input");
            }
        }
    }

    return verifyTensorSize(config::getArch(getOperation()), loc, getInput());
}

size_t vpux::VPUIP::NNDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: Expose API to get arch from cost model
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// PermuteDMAOp
//

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DMADescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, /*port=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr, /*internalDataFlow=*/nullptr);
}

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DMADescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, /*port=*/port,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor, nullptr, /* profilingMetadata= */ nullptr,
          /*internalDataFlow=*/nullptr);
}

mlir::LogicalResult vpux::VPUIP::PermuteDMAOp::verify() {
    return verifyTensorSize(config::getArch(getOperation()), getLoc(), getInput());
}

size_t vpux::VPUIP::PermuteDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs PermuteDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// GatherDMAOp
//

void vpux::VPUIP::GatherDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value indices, mlir::Value outputBuff, mlir::IntegerAttr elementSize,
                                     mlir::IntegerAttr padding, int64_t port = 0) {
    GatherDMAOp::build(builder, state, input, indices, outputBuff, elementSize, padding,
                       /* port = */ vpux::getIntAttr(builder, port), /* channelType = */ nullptr,
                       /* isOutOfOrder = */ nullptr,
                       /* isCritical = */ nullptr,
                       /* DMADescriptor = */ nullptr, /* dmaHwpId = */ nullptr, /* profilingMetadata = */ nullptr,
                       /* gatherAddressingMode = */ nullptr);
}

void vpux::VPUIP::GatherDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value indices, mlir::Value outputBuff, int64_t elementSize, int64_t padding,
                                     int64_t port = 0) {
    GatherDMAOp::build(builder, state, input, indices, outputBuff, vpux::getIntAttr(builder, elementSize),
                       vpux::getIntAttr(builder, padding),
                       /* port = */ vpux::getIntAttr(builder, port), /* channelType = */ nullptr,
                       /* isOutOfOrder = */ nullptr,
                       /* isCritical = */ nullptr,
                       /* DMADescriptor = */ nullptr, /* dmaHwpId = */ nullptr, /* profilingMetadata = */ nullptr,
                       /* gatherAddressingMode = */ nullptr);
}

void vpux::VPUIP::GatherDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value indices, mlir::Value outputBuff, int64_t elementSize, int64_t padding,
                                     int64_t port = 0,
                                     GatherAddressingMode addressingMode = GatherAddressingMode::INDEXED) {
    GatherDMAOp::build(builder, state, input, indices, outputBuff, vpux::getIntAttr(builder, elementSize),
                       vpux::getIntAttr(builder, padding),
                       /* port = */ vpux::getIntAttr(builder, port), /* channelType = */ nullptr,
                       /* isOutOfOrder = */ nullptr,
                       /* isCritical = */ nullptr,
                       /* DMADescriptor = */ nullptr, /* dmaHwpId = */ nullptr, /* profilingMetadata = */ nullptr,
                       VPUIP::GatherAddressingModeAttr::get(builder.getContext(), addressingMode));
}

mlir::LogicalResult vpux::VPUIP::GatherDMAOp::verify() {
    auto loc = getLoc();
    auto arch = config::getArch(getOperation());

    // Skip checks if architecture is unknown, enables LIT tests.
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    // TODO: E#-86281 move to 40xx
    if (arch < config::ArchKind::NPU40XX) {
        return errorAt(loc, "Operation {0} is only supported for NPU40XX+ but got {1}.", getOperationName(), arch);
    }

    auto indicesType = mlir::cast<vpux::NDTypeInterface>(getIndices().getType());

    auto indicesMemKind = indicesType.getMemoryKind();
    size_t indicesLength = indicesType.getNumElements();

    // Check indices are in CMX.
    if (indicesMemKind != vpux::VPU::MemoryKind::CMX_NN) {
        return errorAt(loc, "Indices list must reside in CMX.");
    }

    const size_t DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED = VPU::getGatherDMAMaxIndicesListLength(arch);

    if (indicesLength > DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED) {
        return errorAt(loc, "Number of indices({0}) is greater than max supported({1}) by gather DMA.", indicesLength,
                       DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED);
    }

    return verifyTensorSize(config::getArch(getOperation()), loc, getOutput());
}

size_t vpux::VPUIP::GatherDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs GatherDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// ConvertDMAOp
//

void vpux::VPUIP::ConvertDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::ConvertDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::ConvertDMAOp::verify() {
    auto loc = getLoc();
    auto arch = config::getArch(getOperation());

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutputBuff().getType());
    const auto outputElementType = outputType.getElementType();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inputElementType = inputType.getElementType();

    if ((arch < config::ArchKind::NPU40XX) || !inputElementType.isF32() ||
        (!outputElementType.isF16() && !outputElementType.isBF16())) {
        return errorAt(loc,
                       "Operation {0} is only supported for NPU40XX+ arch for F32 to F16/BF16 "
                       "conversion. "
                       "Got arch {1} "
                       "and conversion from {2} to {3}",
                       getOperationName(), arch, inputElementType, outputElementType);
    }

    return verifyTensorSize(config::getArch(getOperation()), loc, getInput());
}

size_t vpux::VPUIP::ConvertDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs ConvertDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// DecompressDMAOp
//

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port, bool is_out_of_order,
                                         bool is_critical) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, /*act_compression_sparsity_map*/ nullptr,
          output_buff,
          /*port=*/vpux::getIntAttr(builder, port), /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, mlir::IntegerAttr port,
                                         mlir::UnitAttr is_out_of_order, mlir::UnitAttr is_critical) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, /*act_compression_sparsity_map*/ nullptr,
          output_buff,
          /*port=*/port, /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /*dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, /*act_compression_sparsity_map*/ nullptr,
          output_buff,
          /*port=*/nullptr, /*is_out_of_order=*/false, /*is_critical=*/false,
          /*dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*is_out_of_order=*/false,
          /*is_critical=*/false);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                         int64_t port) {
    build(builder, state, input, actCompressionSizeEntryBuff, /*act_compression_sparsity_map*/ nullptr, output_buff,
          /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false, /*is_critical=*/false,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff,
                                         mlir::Value act_compression_sparsity_map, mlir::Value output_buff,
                                         int64_t port) {
    build(builder, state, input, actCompressionSizeEntryBuff, act_compression_sparsity_map, output_buff,
          /*port=*/vpux::getIntAttr(builder, port), /*is_out_of_order=*/false, /*is_critical=*/false,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port,
                                         bool is_out_of_order, bool is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, /*act_compression_sparsity_map*/ nullptr, output_buff,
          /*port=*/vpux::getIntAttr(builder, port), /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                         mlir::IntegerAttr port, mlir::UnitAttr is_out_of_order,
                                         mlir::UnitAttr is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, /*act_compression_sparsity_map*/ nullptr, output_buff,
          /*port=*/port,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::DecompressDMAOp::verify() {
    auto loc = getLoc();

    auto compressionState = getCompressionState(getInput().getType());
    if ((compressionState != VPUIP::CompressionState::RuntimeCompressed) &&
        (compressionState != VPUIP::CompressionState::CompiletimeCompressed)) {
        return errorAt(loc, "Input to DecompressOp must be compressed");
    }

    if (mlir::failed(verifyInOutElementType(loc, getInput(), getOutput())) ||
        mlir::failed(verifyTensorSize(config::getArch(getOperation()), loc, getInput()))) {
        return mlir::failure();
    }

    if (getActCompressionSizeEntry() != nullptr &&
        mlir::failed(verifyCompressedBufferAllocSize(loc, getOutput(), getInput(), getActCompressionSparsityMap()))) {
        return mlir::failure();
    }

    return mlir::success();
}

size_t vpux::VPUIP::DecompressDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs DecompressDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// CompressDMAOp
//

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff,
                                       mlir::Value act_compression_sparsity_map, mlir::Value output_buff,
                                       int64_t port) {
    build(builder, state, input, actCompressionSizeEntryBuff, act_compression_sparsity_map, output_buff,
          /*port=*/vpux::getIntAttr(builder, port), /*is_out_of_order=*/false, /*is_critical=*/false,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port) {
    build(builder, state, input, actCompressionSizeEntryBuff, /*act_compression_sparsity_map*/ nullptr, output_buff,
          /*port=*/vpux::getIntAttr(builder, port), /*is_out_of_order=*/false, /*is_critical=*/false,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port,
                                       bool is_out_of_order, bool is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, /*act_compression_sparsity_map*/ nullptr, output_buff,
          /*port=*/vpux::getIntAttr(builder, port), /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                       mlir::IntegerAttr port, /*optional*/ mlir::UnitAttr is_out_of_order,
                                       /*optional*/ mlir::UnitAttr is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, /*act_compression_sparsity_map*/ nullptr, output_buff,
          /*port=*/port,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::CompressDMAOp::verify() {
    auto loc = getLoc();

    auto compressionState = getCompressionState(getOutput().getType());
    if (compressionState != VPUIP::CompressionState::RuntimeCompressed) {
        return errorAt(loc, "CompressDMAOp must have a compressed buffer as output");
    }

    if (mlir::failed(verifyInOutElementType(loc, getInput(), getOutput())) ||
        mlir::failed(verifyTensorSize(config::getArch(getOperation()), loc, getInput()))) {
        return mlir::failure();
    }

    if (getActCompressionSizeEntry() != nullptr &&
        mlir::failed(verifyCompressedBufferAllocSize(loc, getInput(), getOutput(), getActCompressionSparsityMap()))) {
        return mlir::failure();
    }

    return mlir::success();
}

size_t vpux::VPUIP::CompressDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs CompressDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// DepthToSpaceDMAOp
//

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, output_buff, /* port= */ nullptr, block_size, mode, dma_descriptor,
          padded_channels,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr, /*internalDataFlow= */ nullptr);
}

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port, vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, dma_descriptor, padded_channels,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr, /*internalDataFlow= */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::DepthToSpaceDMAOp::verify() {
    const auto loc = getLoc();
    const auto blockSize = getBlockSize();
    if (blockSize <= 0) {
        return errorAt(loc, "block size for D2S should be > 0; actual {0}", blockSize);
    }

    return verifyTensorSize(config::getArch(getOperation()), loc, getInput());
}

size_t vpux::VPUIP::DepthToSpaceDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs DepthToSpaceDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// SpaceToDepthDMAOp
//
void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode,
                                           VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, nullptr, block_size, mode, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr, nullptr);
}

void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr, nullptr);
}

mlir::LogicalResult vpux::VPUIP::SpaceToDepthDMAOp::verify() {
    const auto loc = getLoc();
    const auto blockSize = getBlockSize();
    if (blockSize <= 0) {
        return errorAt(loc, "block size for S2D should be > 0; actual {0}", blockSize);
    }

    return verifyTensorSize(config::getArch(getOperation()), loc, getInput());
}

size_t vpux::VPUIP::SpaceToDepthDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs SpaceToDepthDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// ExpandDMAOp
//

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DMADescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor,
          /*port=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DMADescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor, /*port=*/port,

          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::ExpandDMAOp::verify() {
    // In case ExpandDMA with input size large than max plane size.
    // It should be tiled with several sub ExpandDMA that will be done at Unroll Pass.
    // Descriptor is generated at Unroll pass so using Descriptor as a flag to check the tensor size.
    if (getDmaDescriptor().has_value()) {
        return verifyTensorSize(config::getArch(getOperation()), getLoc(), getInput());
    }

    return mlir::success();
}

size_t vpux::VPUIP::ExpandDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs ExpandDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// PerAxisTileDMAOp
//

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, nullptr, axis, tiles, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DMADescriptorAttr dma_descriptor,
                                          mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, axis, tiles, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::PerAxisTileDMAOp::verify() {
    return verifyTensorSize(config::getArch(getOperation()), getLoc(), getInput());
}

size_t vpux::VPUIP::PerAxisTileDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs PerAxisTileDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

//
// UpsamplingDMAOp
//

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ nullptr, /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, /* dma_hwp_id */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand,
                                         int64_t port) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ vpux::getIntAttr(odsBuilder, port), /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand, int64_t port,
                                         bool is_out_of_order, bool is_critical, mlir::IntegerAttr dma_hwp_id,
                                         VPUIP::DmaProfilingMetadataAttr profilingMetadata) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ vpux::getIntAttr(odsBuilder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical,
          /* dma_hwp_id */ dma_hwp_id, /* profilingMetadata */ profilingMetadata);
}

mlir::LogicalResult vpux::VPUIP::UpsamplingDMAOp::verify() {
    // In case UpsamplingDMA with input size large than max plane size.
    // It should be tiled with several sub UpsamplingDMA that will be done at Unroll Pass.
    // Descriptor is generated at Unroll pass so using Descriptor as a flag to check the tensor size.
    if (getDmaDescriptor().has_value()) {
        return verifyTensorSize(config::getArch(getOperation()), getLoc(), getInput());
    }

    return mlir::success();
}

size_t vpux::VPUIP::UpsamplingDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs UpsamplingDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

mlir::LogicalResult vpux::VPUIP::SyncDMAOp::verify() {
    auto loc = getLoc();
    const auto inSize = static_cast<Byte>(getCompactSize(getInput()));
    if (inSize.count() != 0) {
        return errorAt(loc, "input size should be zero {0}", inSize);
    }
    const auto outSize = static_cast<Byte>(getCompactSize(getResult()));
    if (outSize.count() != 0) {
        return errorAt(loc, "output size should be zero {0}", outSize);
    }
    return mlir::success();
}

size_t vpux::VPUIP::SyncDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>&) {
    return 0;
}

mlir::LogicalResult vpux::VPUIP::ReadOnlyDMAOp::verify() {
    auto loc = getLoc();
    if (!getResult().use_empty()) {
        return errorAt(loc, "ReadOnlyDMAOp result should have no users, but it does.");
    }
    return verifyTensorSize(config::getArch(getOperation()), getLoc(), getInput());
}

size_t vpux::VPUIP::ReadOnlyDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>&) {
    return 0;
}

mlir::Value vpux::VPUIP::ReadOnlyDMAOp::getOutputBuff() {
    return nullptr;
}

mlir::Value vpux::VPUIP::ReadOnlyDMAOp::getOutput() {
    return nullptr;
}

//
// BarProgDMAOp
//

mlir::LogicalResult vpux::VPUIP::BarProgDMAOp::verify() {
    auto range = getPhysicalBarrierRange();
    if (range.getStart().getValue().getSExtValue() >= range.getEnd().getValue().getSExtValue()) {
        return emitOpError("start_pid must be less than end_pid in pid_range");
    }
    return mlir::success();
}

size_t vpux::VPUIP::BarProgDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// EnqueueDMAOp
//

mlir::LogicalResult vpux::VPUIP::EnqueueDMAOp::verify() {
    auto loc = getLoc();
    auto enqueueDmaAttr = getEnqueueDmaAttr();

    auto startTaskIdx = enqueueDmaAttr.getStartTaskIdx().getValue().getSExtValue();
    auto endTaskIdx = enqueueDmaAttr.getEndTaskIdx().getValue().getSExtValue();

    if (endTaskIdx < startTaskIdx) {
        return errorAt(loc, "endTaskIdx: {0} - must be greater than or equal to startTaskIdx: {1}", endTaskIdx,
                       startTaskIdx);
    }

    if (enqueueDmaAttr.getTargetExecutorKindAttr().getValue() == VPU::ExecutorKind::DMA_NN) {
        return errorAt(loc, "DMA tasks cannot be enqueued by enqueue DMA");
    }

    if (auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>()) {
        if (auto tileExecutorOp = config::getTileExecutor(parentModule)) {
            const auto tilesCount = tileExecutorOp.getCount();
            int64_t tileIdx = enqueueDmaAttr.getTileIdx().getValue().getSExtValue();
            if (tileIdx < 0 || tileIdx >= tilesCount) {
                return errorAt(loc, "tileIdx: {0} - must be in range [0, {1}]", tileIdx, tilesCount);
            }
        }
    }

    return mlir::success();
}

size_t vpux::VPUIP::EnqueueDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// FetchDMAOp
//

mlir::LogicalResult vpux::VPUIP::FetchDMAOp::verify() {
    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (auto tileExecutorResourceOp = config::getTileExecutor(parentModule)) {
        const auto tilesCount = tileExecutorResourceOp.getCount();
        auto loc = getLoc();
        int64_t tileIdx = getFetchDma().getTileIdx().getValue().getSExtValue();
        if (tileIdx < 0 || tileIdx >= tilesCount) {
            return errorAt(loc, "tileIdx: {0} must be in range [0, {1}]", tileIdx, tilesCount);
        }
    }

    return mlir::success();
}

size_t vpux::VPUIP::FetchDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: Expose API to get arch from cost model
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}
