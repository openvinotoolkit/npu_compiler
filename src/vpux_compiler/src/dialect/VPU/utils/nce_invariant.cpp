//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/small_kernel_optimization.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Fuse PadOp check
//

bool vpux::VPU::NCEInvariant::verifyPads(int64_t KY, int64_t KX, int64_t padTop, int64_t padBottom, int64_t padLeft,
                                         int64_t padRight, LogCb logCb) {
    if (padTop < 0 || padTop > KY / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padTop, KY / 2));
        return false;
    }
    if (padBottom < 0 || padBottom > KY / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padBottom, KY / 2));
        return false;
    }
    if (padLeft < 0 || padLeft > KX / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padLeft, KX / 2));
        return false;
    }
    if (padRight < 0 || padRight > KX / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padRight, KX / 2));
        return false;
    }

    return true;
}

bool vpux::VPU::NCEInvariant::verifyPads(mlir::ArrayAttr kernelSizeAttr, mlir::ArrayAttr padBeginAttr,
                                         mlir::ArrayAttr padEndAttr, LogCb logCb) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);
    const auto KY = kernelSize[kernelSize.size() == 4 ? (Dims4D::Filter::KY.ind()) : (Dims4D::Kernel::Y.ind())];
    const auto KX = kernelSize[kernelSize.size() == 4 ? (Dims4D::Filter::KX.ind()) : (Dims4D::Kernel::X.ind())];

    const auto padsBegin = parseIntArrayAttr<int64_t>(padBeginAttr);
    const auto padsEnd = parseIntArrayAttr<int64_t>(padEndAttr);
    const auto padTop = padsBegin[Dims4D::PadsBegin::Top.ind()];
    const auto padLeft = padsBegin[Dims4D::PadsBegin::Left.ind()];
    const auto padBottom = padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    const auto padRight = padsEnd[Dims4D::PadsEnd::Right.ind()];

    return verifyPads(KY, KX, padTop, padBottom, padLeft, padRight, logCb);
}

//
// Attributes checks
//

bool vpux::VPU::NCEInvariant::isAttrsSupported(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                               int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                               LogCb logCb) {
    if (VPU::hasMaxKernelSize(op)) {
        const auto maxKernelSize = VPU::getMaxKernelSize(op);

        if (KY > maxKernelSize || KY <= 0) {
            logCb(formatv("Unsupported kernel height dimension '{0}', must be in range [1, {1}]", KY, maxKernelSize));
            return false;
        }
        if (KX > maxKernelSize || KX <= 0) {
            logCb(formatv("Unsupported kernel width dimension '{0}', must be in range [1, {1}]", KX, maxKernelSize));
            return false;
        }
    }

    static const int64_t NCE_MAX_STRIDE_SIZE = 8;

    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        logCb(formatv("Unsupported stride height dimension '{0}', must be in range [1, {1}]", SY, NCE_MAX_STRIDE_SIZE));
        return false;
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        logCb(formatv("Unsupported stride width dimension '{0}', must be in range [1, {1}]", SX, NCE_MAX_STRIDE_SIZE));
        return false;
    }

    return verifyPads(KY, KX, padTop, padBottom, padLeft, padRight, logCb);
}

//
// Activation type checks
//

bool vpux::VPU::NCEInvariant::isAligned(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb) {
    const auto shape = type.getShape();
    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);

    // In super-dense mode only channels must be aligned.
    const auto channels = type.getRank() == 4 ? shape[Dims4D::Act::C] : shape[DimsGroups5D::Act::C];
    if (channels % alignment == 0) {
        return true;
    }

    const auto innerDim = memShape.back();
    if (innerDim % alignment != 0) {
        logCb(formatv("Activation inner dimension '{0}' is not aligned to '{1}'", innerDim, alignment));
        return false;
    }

    return true;
}

bool is5DAligned(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb) {
    const auto shape = type.getShape();
    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);

    // In super-dense mode only channels must be aligned.
    const auto channels = shape[DimsGroups5D::Act::C];
    if (channels % alignment == 0) {
        return true;
    }

    const auto innerDim = memShape.back();
    if (innerDim % alignment != 0) {
        logCb(formatv("Activation inner dimension '{0}' is not aligned to '{1}'", innerDim, alignment));
        return false;
    }

    return true;
}

int64_t vpux::VPU::NCEInvariant::getAlignment(mlir::Type elemType) {
    const Bit typeSizeInBits = getElemTypeSize(elemType);
    return std::max<int64_t>(128 / typeSizeInBits.count(), 16);
}

bool vpux::VPU::NCEInvariant::isOutputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb) {
    if (type.getRank() == DimsGroups5D::Act::numDims) {
        const auto channels = type.getShape()[DimsGroups5D::Act::C];
        return channels % alignment == 0;
    }

    if (type.getRank() != 4) {
        logCb(formatv("Ouput activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    const auto OC = type.getShape()[Dims4D::Act::C];
    if (OC % alignment != 0) {
        logCb(formatv("Output input channels '{0}' are not aligned to '{1}'", OC, alignment));
        return false;
    }

    return true;
}

bool vpux::VPU::NCEInvariant::isInputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment,
                                                      bool supportsInputActCompression, LogCb logCb) {
    if (type.getRank() == DimsGroups5D::Act::numDims) {
        return isAligned(type, alignment, logCb);
    }

    if (type.getRank() != 4) {
        logCb(formatv("Input activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    if (supportsInputActCompression) {
        const auto IC = type.getShape()[Dims4D::Act::C];
        const bool inputChannelsMatch = (IC == VPU_COMPRESSED_INPUT_CHANNEL_NUM);
        if (!inputChannelsMatch) {
            logCb(formatv("Input channels do not match VPU_COMPRESSED_INPUT_CHANNEL_NUM: got {0} expected {1}", IC,
                          VPU_COMPRESSED_INPUT_CHANNEL_NUM));
        }
        return inputChannelsMatch;
    }

    return isAligned(type, alignment, logCb);
}

//
// WeightsTable information
//

Byte vpux::VPU::NCEInvariant::getWeightsTableSize(int64_t OC) {
    return OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;
}

// OC can be used to represent a number of output channels that is different from the number of output channels in op
// (e.g. when a new output type will be used)
mlir::FailureOr<SmallVector<Byte>> vpux::VPU::NCEInvariant::getWeightsTableSize(int64_t OC, mlir::Value weightsTable,
                                                                                mlir::Value weightTableScale,
                                                                                mlir::Value weightTableBias) {
    if (weightsTable != nullptr) {
        return SmallVector<Byte>{getWeightsTableSize(OC)};
    }

    SmallVector<Byte> newWeightTables{};
    if (weightTableScale != nullptr) {
        newWeightTables.push_back(OC * 4_Byte);
    }
    if (weightTableBias != nullptr) {
        newWeightTables.push_back(OC * 4_Byte);
    }

    return newWeightTables;
}

template <typename NCEOp>
mlir::LogicalResult getWeightTableBuffersBase(NCEOp nceOp, SmallVector<Byte>& buffers, int64_t OC) {
    auto weightTables = vpux::VPU::NCEInvariant::getWeightsTableSize(
            OC, nceOp.getWeightsTable(), nceOp.getWeightTableScale(), nceOp.getWeightTableBias());
    if (mlir::failed(weightTables)) {
        return mlir::failure();
    }

    auto weightTablesValue = weightTables.value();
    buffers.append(weightTablesValue.begin(), weightTablesValue.end());
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEInvariant::getWeightTableBuffers(mlir::Operation* op, SmallVector<Byte>& buffers,
                                                                   int64_t OC) {
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getWeightTableBuffersBase(convOp, buffers, OC);
    } else if (auto dwConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getWeightTableBuffersBase(dwConvOp, buffers, OC);
    }
    return mlir::failure();
}

//
// verifyWeightTables
//

template <typename NCEOp>
static mlir::LogicalResult verifyWeightTablesBase(NCEOp op, Logger) {
    if ((op.getWeightTableDataPtr() || op.getWeightTableSpPtr() || op.getWeightTableScale() ||
         op.getWeightTableBias() || op.getWeightZeroPoints())) {
        return errorAt(op, "Only weightsTable can be populated for NCEOp");
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyWeightTables(mlir::Operation* op, Logger log) {
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return verifyWeightTablesBase(convOp, log);
    } else if (auto dwConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return verifyWeightTablesBase(dwConvOp, log);
    }
    return mlir::failure();
}

//
// Common utility for AvgPool, MaxPool, Eltwise and DWConv
//

bool vpux::VPU::NCEInvariant::checkLayouts(mlir::TypeRange operandTypes, mlir::TypeRange resultTypes,
                                           const config::ArchKind& arch, const unsigned numInputOperands, LogCb logCb) {
    VPUX_UNUSED(resultTypes);
    VPUX_UNUSED(arch);

    for (unsigned opIdx = 0; opIdx < numInputOperands; opIdx++) {
        const auto actualInLayout = mlir::cast<vpux::NDTypeInterface>(operandTypes[opIdx]).getDimsOrder();
        const auto& expectedInLayout = DimsOrder::NHWC;
        if (actualInLayout != expectedInLayout) {
            logCb(formatv("Unsupported input layout. Expected: {0}, got: {1}", expectedInLayout, actualInLayout));
            return false;
        }
    }

    return true;
}

bool vpux::VPU::NCEInvariant::isEltwiseMultiplySubtractSupported(const config::ArchKind arch) {
    return arch > config::ArchKind::NPU40XX;
}

mlir::LogicalResult vpux::VPU::NCEInvariant::isSupported(mlir::Operation* op, Logger) {
    const bool checkLayout = false;
    const bool checkChannelAlignment = false;
    const bool allowDifferentScales = true;
    const bool allowDifferentZp = true;

    return mlir::success(
            llvm::TypeSwitch<mlir::Operation*, bool>(op)
                    .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                        return VPU::NCEConvolutionOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                  checkChannelAlignment);
                    })
                    .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp origOp) {
                        return VPU::NCEMaxPoolOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::AvgPoolOp>([&](IE::AvgPoolOp origOp) {
                        return VPU::NCEAveragePoolOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                  checkChannelAlignment);
                    })
                    .Case<IE::AddOp>([&](IE::AddOp origOp) {
                        return VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp,
                                                              emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    // #E157147: Do not set layout for NCE multiply. It will be enabled once it is optimal.
                    .Case<IE::SubtractOp>([&](auto origOp) {
                        const auto arch = config::getArch(origOp);
                        if (!isEltwiseMultiplySubtractSupported(arch)) {
                            return false;
                        }
                        // #E172315: Do not set layout for NCE subtract when using dynamic shapes.
                        auto outType = origOp.getOutput().getType();
                        if (auto boundedTensor = mlir::dyn_cast<Core::BoundedTensorType>(outType)) {
                            return false;
                        }
                        return VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp,
                                                              emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::ReduceMeanOp>([&](IE::ReduceMeanOp origOp) {
                        return VPU::NCEReduceOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::ReduceSumOp>([&](IE::ReduceSumOp origOp) {
                        return VPU::NCEReduceOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp origOp) {
                        return VPU::NCEDepthConvolutionOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                       checkChannelAlignment);
                    })
                    .Case<IE::InterpolateOp>([&](IE::InterpolateOp origOp) {
                        return VPU::NCEInterpolateOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                  checkChannelAlignment, /*checkBatch=*/false);
                    })
                    .Case<IE::TransposedConvolutionOp>([&](IE::TransposedConvolutionOp origOp) {
                        return isSupportedSEPTransposedConv(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::PadOp>([&](IE::PadOp origOp) {
                        return isSupportedSEPPadOp(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::RollOp>([&](IE::RollOp origOp) {
                        return VPU::isSupportedSEPRoll(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::MatMulOp>([&](IE::MatMulOp origOp) {
                        return VPU::NCEMatMulOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Default([](mlir::Operation*) -> bool {
                        return false;
                    }));
}

bool vpux::VPU::NCEInvariant::doesWorkloadSupportSmallKernelOpt([[maybe_unused]] config::ArchKind arch,
                                                                const int64_t KX, const int64_t SX,
                                                                ArrayRef<int64_t> workloadOutSz, bool isFp16Input,
                                                                [[maybe_unused]] const int64_t KY,
                                                                [[maybe_unused]] const int64_t padLeft) {
    return VPU::doesWorkloadSupportSmallKernelOpt(arch, KX, SX, workloadOutSz, isFp16Input, KY, padLeft);
}

bool vpux::VPU::NCEInvariant::isSmallKernelOptimizationSupported(const config::ArchKind arch, mlir::Operation* op) {
    if (!mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        return false;
    }
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    // Skip Sparse Ops
    if (mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceOp->getResult(0).getType()) != nullptr) {
        return false;
    }

    const auto kernelSize = nceOp.getKernelSizeVal();
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];
    [[maybe_unused]] const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto kernelStride = nceOp.getStridesVal();
    const auto SX = kernelStride[Dims4D::Strides::X.ind()];

    // Get a set containing all the channels from the workloads of the given NCE operations
    const auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();

    return VPU::isSmallKernelOptimizationSupported(op, arch, KX, KY, SX,
                                                   SmallVector<VPU::DPUWorkloadOp>(workloads.begin(), workloads.end()));
}

//
// verifyKernel
//

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyKernel(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY,
                                                          int64_t SX, int64_t padTop, int64_t padBottom,
                                                          int64_t padLeft, int64_t padRight, Logger log) {
    log.setName("NCEInvariant");
    auto loc = op->getLoc();
    if (VPU::hasMaxKernelSize(op)) {
        const auto maxKernelSize = VPU::getMaxKernelSize(op);

        if (KY > maxKernelSize || KY <= 0) {
            log.trace("[{0}] Unsupported kernel height dimension '{1}', must be in range [1, {2}]", loc, KY,
                      maxKernelSize);
            return mlir::failure();
        }
        if (KX > maxKernelSize || KX <= 0) {
            log.trace("[{0}] Unsupported kernel width dimension '{1}', must be in range [1, {2}]", loc, KX,
                      maxKernelSize);
            return mlir::failure();
        }
    }

    static const int32_t NCE_MAX_STRIDE_SIZE = 8;
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        log.trace("[{0}] Unsupported stride height dimension '{1}', must be in range [1, {2}]", loc, SY,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        log.trace("[{0}] Unsupported stride width dimension '{1}', must be in range [1, {2}]", loc, SX,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    if (padTop < 0 || (padTop > 1 && padTop > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padTop, KY / 2);
        return mlir::failure();
    }
    if (padBottom < 0 || (padBottom > 1 && padBottom > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padBottom, KY / 2);
        return mlir::failure();
    }
    if (padLeft < 0 || (padLeft > 1 && padLeft > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padLeft, KX / 2);
        return mlir::failure();
    }
    if (padRight < 0 || (padRight > 1 && padRight > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padRight, KX / 2);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyKernel(mlir::Operation* op, Logger) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                return VPU::NCEConvolutionOp::verifyKernel(origOp);
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp origOp) {
                return VPU::NCEMaxPoolOp::verifyKernel(origOp);
            })
            .Case<IE::AvgPoolOp>([&](IE::AvgPoolOp origOp) {
                return VPU::NCEAveragePoolOp::verifyKernel(origOp);
            })
            .Case<IE::AddOp>([&](IE::AddOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp origOp) {
                return VPU::NCEDepthConvolutionOp::verifyKernel(origOp);
            })
            .Case<IE::TransposedConvolutionOp>([&](IE::TransposedConvolutionOp origOp) {
                return VPU::NCEConvolutionOp::verifyKernel(origOp);
            })
            .Case<IE::MatMulOp>([&](IE::MatMulOp origOp) {
                return VPU::NCEMatMulOp::verifyKernel(origOp);
            })
            .Default([](mlir::Operation*) -> mlir::LogicalResult {
                return mlir::failure();
            });
}

//
// verifyPoolCMX
//

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module,
                                                           vpux::NDTypeInterface inputType,
                                                           vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelSize,
                                                           mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: {0}", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelSize.size());

    const auto outputShape = outputType.getShape();
    const auto OC = outputShape[Dims4D::Act::C];

    const auto requiredCMX = VPU::getRequiredCMXSizeForNCEOps({inputType, outputType}, OC);

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Pooling, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

bool vpux::VPU::NCEInvariant::isParentOptimalForAlignment(mlir::Operation* parentOp) {
    // To avoid regression caused by expand and slice,
    // we only allow NCE + SoftMax pattern now
    if (parentOp == nullptr) {
        return false;
    }
    // Skip if the parent is a slice
    if (auto sliceOp = mlir::dyn_cast<IE::SliceOp>(parentOp)) {
        auto parentParentOp = sliceOp.getSource().getDefiningOp();
        parentOp = parentParentOp;
    }
    if (isSupported(parentOp).failed()) {
        return false;
    }
    return true;
}

bool vpux::VPU::NCEInvariant::isAlignmentBeneficial(mlir::Operation* op) {
    // Experiments shows that alignment can be beneficial for SW ops
    // #E164667: For SoftMax with axis on C and IC > 256, alignment bring significant SHAVE performance gain
    if (auto softMaxOp = mlir::dyn_cast<IE::SoftMaxOp>(op)) {
        if (!isParentOptimalForAlignment(softMaxOp.getInput().getDefiningOp())) {
            return false;
        }
        auto inputType = mlir::cast<vpux::NDTypeInterface>(softMaxOp.getInput().getType());
        auto inputShape = inputType.getShape();
        const auto IC = inputShape[Dims4D::Act::C];

        if (softMaxOp.getAxisInd() == Dims4D::Act::C.ind() && IC > 256) {
            return true;
        }
    }

    return false;
}

bool vpux::VPU::NCEInvariant::hasDimensionExceedingVPULimit(ShapeRef shape) {
    return llvm::any_of(shape, [](auto dim) {
        return dim > VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    });
}
