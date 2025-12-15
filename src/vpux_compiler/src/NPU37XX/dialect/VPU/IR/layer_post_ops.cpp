//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
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

                const auto inElemType =
                        mlir::cast<vpux::NDTypeInterface>(mainOp->getOperand(0).getType()).getElementType();
                const auto outElemType =
                        mlir::cast<vpux::NDTypeInterface>(mainOp->getResult(0).getType()).getElementType();
                // Because of the convert to float, the prelu shift will be bypassed. Check PPE diagram
                if (mlir::isa<mlir::quant::QuantizedType>(inElemType) &&
                    !mlir::isa<mlir::quant::QuantizedType>(outElemType)) {
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

template <typename ConcreteModel, typename MainOpType>
class LayerWithPostOpModelBase : public VPU::LayerWithClampOpModel<ConcreteModel, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) const {
        if (config::getCompilationMode(postOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!isSupportedHWPostOp(mainOp, postOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::isSupported(mlir::cast<MainOpType>(mainOp)).succeeded();
    }
};

template <class MainOpType>
class LayerWithPostOpUsingBiasAndStaticScaleModel final :
        public LayerWithPostOpModelBase<LayerWithPostOpUsingBiasAndStaticScaleModel<MainOpType>, MainOpType> {
public:
    bool supportsFuseBiasScale(mlir::Operation*) const {
        return true;
    }
};

template <class MainOpType>
class LayerWithPostOpModel final : public LayerWithPostOpModelBase<LayerWithPostOpModel<MainOpType>, MainOpType> {};

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::arch37xx::registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPostOpUsingBiasAndStaticScaleModel<IE::ConvolutionOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<
                LayerWithPostOpUsingBiasAndStaticScaleModel<IE::TransposedConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPostOpUsingBiasAndStaticScaleModel<IE::GroupConvolutionOp>>(
                *ctx);
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
