//
// Copyright (C) 2023-2026 Intel Corporation
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

template <class MainOpType>
class LayerWithPostOpModel : public VPU::LayerWithPostOpModelBase<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    static bool isSupportedHWClampOp(mlir::Operation* mainOp, double minValue, double maxValue, mlir::Type type,
                                     const LogCb& logCb) {
        const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, type);
        return isSupportedHWClampOp(mainOp, minValue, maxValue, mainOp->getLoc(), isQuantized, logCb);
    }

    static bool isSupportedHWClampOp(mlir::Operation* mainOp, IE::ClampOp clampOp, const LogCb& logCb) {
        const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, clampOp);
        const auto minValue = clampOp.getMinAttr().getValueAsDouble();
        const auto maxValue = clampOp.getMaxAttr().getValueAsDouble();
        return isSupportedHWClampOp(mainOp, minValue, maxValue, clampOp->getLoc(), isQuantized, logCb);
    }

    static bool isSupportedHWPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) {
        return llvm::TypeSwitch<mlir::Operation*, bool>(postOp)
                .Case<IE::ReLUOp>([&](auto) {
                    if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
                        logCb(llvm::formatv("{0} does not support fusing with {1} for this HW platform at `{2}`",
                                            mainOp->getName(), postOp->getName(), postOp->getLoc()));
                        return false;
                    }

                    return true;
                })
                .template Case<IE::LeakyReluOp>([&](auto) {
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

    bool supportsFuseBiasScale(mlir::Operation*) const {
        return _supportsFuseBiasScale;
    }

protected:
    bool _supportsFuseBiasScale = false;

private:
    static bool isSupportedHWClampOp(mlir::Operation* mainOp, double minValue, double maxValue, mlir::Location loc,
                                     bool isQuantized, const LogCb& logCb) {
        if (!isDoubleEqual(minValue, 0.0) && !isQuantized) {
            logCb(llvm::formatv("Clamp at '{0}' is not quantized and does not have 0 as minValue", loc));
            return false;
        }

        // Disable MaxPool fused with Clamp since it is not fully supported by firmware.
        // For more detailed information: E#-145636
        if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
            const auto maxValueFP16 = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
            // Given upper bound as fp16 maxValue value, keep fusing Clamp into MaxPool to pass CI
            if ((!isDoubleEqual(maxValue, maxValueFP16))) {
                logCb(llvm::formatv("Clamp at loc '{0}' cannot be fused into MaxPool due to lack of firmware support",
                                    loc));
                return false;
            }
        }
        return true;
    }
};

template <class MainOpType>
class LayerWithPostOpUsingBiasAndStaticScaleModel final : public LayerWithPostOpModel<MainOpType> {
public:
    LayerWithPostOpUsingBiasAndStaticScaleModel() {
        this->_supportsFuseBiasScale = true;
    }
};

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
        VPU::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<VPU::GroupConvolutionOp>>(*ctx);
    });
}
