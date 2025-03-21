//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// LayerWithPostOpModel37XX
//

bool isSupportedHWPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(postOp)
            .Case<IE::ReLUOp>([&](auto) {
                if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
                    logCb(llvm::formatv("{0} does not support fusing with {1} for this HW platform at `{2}`",
                                        mainOp->getName(), postOp->getName(), postOp->getLoc()));
                    return false;
                }

                return true;
            })
            // TODO: remove option after E#-83187
            .Case<IE::ClampOp>([&](IE::ClampOp clampOp) {
                const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, postOp);
                const auto minVal = clampOp.getMinAttr().getValueAsDouble();
                if (!isDoubleEqual(minVal, 0.0) && !isQuantized) {
                    logCb(llvm::formatv("{0} is not quantized and does not have 0 as minVal at `{1}`",
                                        postOp->getName(), postOp->getLoc()));
                    return false;
                }
                // Disable MaxPool fused with Clamp since it is not fully supported by firmware.
                // Tracking Number: E#-145636
                if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
                    const auto maxVal = clampOp.getMaxAttr().getValueAsDouble();
                    const auto maxValueFP16 = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
                    // Given upper bound as fp16 max value, keep fusing Clamp into MaxPool to pass CI
                    // Tracking Number: E#-146652
                    if ((!isDoubleEqual(maxVal, maxValueFP16))) {
                        logCb(llvm::formatv("{0} at `{1}` cannot be fused into MaxPool due to lack of firmware support",
                                            postOp->getName(), postOp->getLoc()));
                        return false;
                    }
                }
                return true;
            })
            .Case<IE::LeakyReluOp>([&](auto) {
                if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
                    logCb(llvm::formatv("{0} does not support fusing with {1} for this HW platform at `{2}`",
                                        mainOp->getName(), postOp->getName(), postOp->getLoc()));
                    return false;
                }

                const auto inElemType = mainOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
                const auto outElemType = mainOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
                // Because of the convert to float, the prelu shift will be bypassed. Check PPE diagram
                if (inElemType.isa<mlir::quant::QuantizedType>() && !outElemType.isa<mlir::quant::QuantizedType>()) {
                    logCb(llvm::formatv("{0} does not support fusing with {1} for this HW platform at `{2}`",
                                        mainOp->getName(), postOp->getName(), postOp->getLoc()));
                    return false;
                }

                return true;
            })
            .Default([&](mlir::Operation*) {
                logCb(llvm::formatv("{0} at `{1}` is not supported on this HW platform", postOp->getName(),
                                    postOp->getLoc()));
                return false;
            });
}

template <class MainOpType>
class LayerWithPostOpModel final :
        public IE::LayerWithPostOpInterface::ExternalModel<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) const {
        if (VPU::getCompilationMode(postOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!isSupportedHWPostOp(mainOp, postOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(mainOp)).succeeded();
    }

    bool isSupportedClampOp(mlir::Operation* mainOp, mlir::Operation* clampOp, const LogCb& logCb) const {
        if (VPU::getCompilationMode(clampOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!VPU::isSupportedHWClampOp(mainOp, clampOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(mainOp)).succeeded();
    }

    void setLayerClampOp(mlir::Operation* mainOp, mlir::Operation* activationOp) const {
        VPU::setHWClampOp(mainOp, activationOp);
    }
};

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::arch37xx::registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPostOpModel<IE::ConvolutionOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<LayerWithPostOpModel<IE::TransposedConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<IE::GroupConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<LayerWithPostOpModel<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<LayerWithPostOpModel<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<LayerWithPostOpModel<IE::AddOp>>(*ctx);
        IE::SubtractOp::attachInterface<LayerWithPostOpModel<IE::SubtractOp>>(*ctx);
        IE::MatMulOp::attachInterface<LayerWithPostOpModel<IE::MatMulOp>>(*ctx);
    });
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::TransposedConvolutionOp::attachInterface<LayerWithPostOpModel<VPU::TransposedConvolutionOp>>(*ctx);
    });
}
