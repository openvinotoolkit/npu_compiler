//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// DDRAccessGatherOpModel
//

class DDRAccessGatherOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGatherOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto gatherOp = mlir::dyn_cast<VPU::GatherOp>(op);
        VPUX_THROW_WHEN(gatherOp == nullptr, "Unexpected op {0} at '{1}'", op->getName(), op->getLoc());

        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(gatherOp).to<Byte>().count();
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getInput().getType());
        const auto inputShape = inputType.getShape().raw();
        const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
        int64_t axisValue =
                mlir::dyn_cast_or_null<mlir::IntegerAttr>(gatherOp.getAxisValueAttr()).getValue().getSExtValue();
        const auto axisDimSizeBytes = inputShape[axisValue] * inputByteSize;

        // Can't get feasible tiling strategy because axis dimension of gatherOp can't be tiled.
        if (axisDimSizeBytes > cmxAvailableBytes) {
            log.nest(1).trace("Can't still fit into CMX after tiling. The case should be solved with DDR solution.");
            return true;
        }

        // "DDR Access" is preferred for scenarios with large inputs and small outputs
        // If the Output buffer exceeds CMX memory size, memory allocation follows:
        // Input (DDR) + Indices (CMX) -> Output (DDR)
        // Experiments indicate significant performance degradation (E#123794 for details)
        const auto isStrideSrcData = std::any_of(inputShape.begin(), inputShape.begin() + axisValue, [](auto dimSize) {
            return dimSize != 1;
        });

        if (!isStrideSrcData) {
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getOutput().getType());
            const auto outputByteSize = outputType.getElemTypeSize().to<Byte>().count();
            const auto isBeneficialScenario = (inputType.getShape().totalSize() * inputByteSize) > cmxAvailableBytes &&
                                              (outputType.getShape().totalSize() * outputByteSize) < cmxAvailableBytes;
            if (isBeneficialScenario) {
                log.nest(1).trace("Gather layer {0} has large input and output buffer in CMX, DDR Access is preferred "
                                  "for better performance.",
                                  gatherOp);
                return true;
            }
        }

        return false;
    }
};

//
// DDRAccessGRUSequenceOpModel
//

class DDRAccessGRUSequenceOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGRUSequenceOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto gruSequenceOp = mlir::dyn_cast<VPU::GRUSequenceOp>(op);
        auto outputShape = mlir::cast<vpux::NDTypeInterface>(gruSequenceOp.getMiddleHiddenState().getType()).getShape();
        Shape minShapeAfterTiling(outputShape.size(), 1);
        minShapeAfterTiling[Dim(3)] = outputShape[Dim(3)];
        auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
        if (!iface.isSupportedTiling({TileInfo(minShapeAfterTiling)}, TilingMode::ISOLATED, log.nest())) {
            log.nest(1).trace("Can't still fit into CMX after tiling.DDR access will be used for GRUSequenceOp.");
            return true;
        }
        return false;
    }
};

//
// DDRAccessGRUSequenceLastPartOpModel
//

class DDRAccessGRUSequenceLastPartOpModel final :
        public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGRUSequenceLastPartOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto gruSequenceLastPartOp = mlir::dyn_cast<VPU::GRUSequenceLastPartOp>(op);
        auto outputShape =
                mlir::cast<vpux::NDTypeInterface>(gruSequenceLastPartOp.getMiddleHiddenState().getType()).getShape();
        Shape minShapeAfterTiling(outputShape.size(), 1);
        minShapeAfterTiling[Dim(3)] = outputShape[Dim(3)];
        auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
        if (!iface.isSupportedTiling({TileInfo(minShapeAfterTiling)}, TilingMode::ISOLATED, log.nest())) {
            log.nest(1).trace(
                    "Can't still fit into CMX after tiling.DDR access will be used for GRUSequenceLastPartOp.");
            return true;
        }
        return false;
    }
};

//
// DDRAccessGridSampleOpModel
//

class DDRAccessGridSampleOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGridSampleOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        const auto inputShape = inputType.getShape();

        // For GridSample op, input cannot be tiled over H&W since grid coordinates are based on input size. So if
        // GridSample's input spatial size is larger than CMX size, DDR access is necessary.
        const auto totalSize = inputType.getTotalAllocSize();
        const auto spatialSize = totalSize / (inputShape[Dims4D::Act::N] * inputShape[Dims4D::Act::C]);

        if (spatialSize > vpux::VPU::getTotalCMXSize(op)) {
            log.nest(1).trace("GridSample op cannot be tiled, need DDR access");
            return true;
        }
        return false;
    }
};

//
// DDRAccessDeformableConvolutionOpModel
//

class DDRAccessDeformableConvolutionOpModel final :
        public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessDeformableConvolutionOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto defConvOp = mlir::dyn_cast<VPU::DeformableConvolutionOp>(op);
        VPUX_THROW_WHEN(defConvOp == nullptr, "Unexpected op {0} at '{1}'", op->getName(), op->getLoc());

        const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        const auto inputShape = inputType.getShape();

        const auto offsetType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType());
        const auto offsetShape = offsetType.getShape();

        const auto kernelType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(2).getType());
        const auto kernelShape = kernelType.getShape();

        const auto maskType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(3).getType());
        const auto maskShape = maskType.getShape();

        /* For DeformableConvolution op, first input cannot be tiled over C & H & W.
            Offset input and mask input cannot be tiled over C.
            Kernel input cannot be tiled over C & H & W.
            So if DeformableConvolution's inputs required size is larger than
            CMX size, DDR access is necessary. */

        const auto totalSizeInput = inputType.getTotalAllocSize();
        const auto requiredSizeInput = totalSizeInput / (inputShape[Dims4D::Act::N]);

        const auto totalSizeOffset = offsetType.getTotalAllocSize();
        const auto requiredSizeOffset = totalSizeOffset / (offsetShape[Dims4D::Act::N] * offsetShape[Dims4D::Act::H] *
                                                           offsetShape[Dims4D::Act::W]);

        const auto totalSizeKernel = kernelType.getTotalAllocSize();
        const auto requiredSizeKernel = totalSizeKernel / (kernelShape[Dims4D::Act::N]);

        const auto totalSizeMask = maskType.getTotalAllocSize();
        const auto requiredSizeMask =
                totalSizeMask / (maskShape[Dims4D::Act::N] * maskShape[Dims4D::Act::H] * maskShape[Dims4D::Act::W]);

        const auto individualCheck = requiredSizeInput > vpux::VPU::getTotalCMXSize(op) ||
                                     requiredSizeOffset > vpux::VPU::getTotalCMXSize(op) ||
                                     requiredSizeKernel > vpux::VPU::getTotalCMXSize(op) ||
                                     requiredSizeMask > vpux::VPU::getTotalCMXSize(op);

        const auto sumCheck = (requiredSizeInput + requiredSizeOffset + requiredSizeKernel + requiredSizeMask) >
                              vpux::VPU::getTotalCMXSize(op);

        if (individualCheck || sumCheck) {
            log.nest(1).trace(
                    "Can't still fit into CMX after tiling. DDR access will be needed for DeformableConvolutionOp.");
            return true;
        }
        return false;
    }
};

//
// DDRAccessRandomUniformOpModel
//

class DDRAccessRandomUniformOpModel final :
        public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessRandomUniformOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto randomUniformOp = mlir::dyn_cast<VPU::RandomUniformOp>(op);
        VPUX_THROW_WHEN(randomUniformOp == nullptr, "Unexpected op {0} at '{1}'", op->getName(), op->getLoc());

        // For RandomUniform op, it cannot be tiled if globalSeed != 0 or opSeed != 0.
        // If both seed values equal to zero, RandomUniform generates non-deterministic sequence.
        const auto globalSeed = randomUniformOp.getGlobalSeed();
        const auto opSeed = randomUniformOp.getOpSeed();
        if (globalSeed != 0 || opSeed != 0) {
            log.nest(1).trace("RandomUniform op cannot be tiled with non-zero seeds.");
            return true;
        }
        return false;
    }
};

//
// DDRAccessTopKOpModel
//

class DDRAccessTopKOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessTopKOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto topkOp = mlir::dyn_cast<VPU::TopKOp>(op);
        VPUX_THROW_WHEN(topkOp == nullptr, "Unexpected op {0} at '{1}'", op->getName(), op->getLoc());

        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(topkOp).to<Byte>().count();
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(topkOp.getInput().getType());
        const auto inputShape = inputType.getShape().raw();
        const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
        int64_t axisValue = mlir::dyn_cast_or_null<mlir::IntegerAttr>(topkOp.getAxisAttr()).getValue().getSExtValue();
        const auto axisDimSizeBytes = inputShape[axisValue] * inputByteSize;

        // Can't get feasible tiling strategy because axis dimension of TopKOp can't be tiled.
        if (axisDimSizeBytes > cmxAvailableBytes) {
            log.nest(1).trace("Can't fit into CMX after tiling. The case should be solved with DDR solution.");
            return true;
        }

        return false;
    }
};

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::arch37xx::registerDDRAccessOpModelInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::GatherOp::attachInterface<DDRAccessGatherOpModel>(*ctx);
        VPU::GridSampleOp::attachInterface<DDRAccessGridSampleOpModel>(*ctx);
        VPU::DeformableConvolutionOp::attachInterface<DDRAccessDeformableConvolutionOpModel>(*ctx);
        VPU::TopKOp::attachInterface<DDRAccessTopKOpModel>(*ctx);
        VPU::RandomUniformOp::attachInterface<DDRAccessRandomUniformOpModel>(*ctx);
        VPU::GRUSequenceOp::attachInterface<DDRAccessGRUSequenceOpModel>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<DDRAccessGRUSequenceLastPartOpModel>(*ctx);
    });
}
