//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// LayerWithPostOpModel50XX
//

template <class MainOpType>
class LayerWithPostOpModel : public VPU::LayerWithPostOpModelBase<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    static bool isSupportedHWClampOp(mlir::Operation*, double, double, mlir::Type, const LogCb&) {
        return true;
    }
    static bool isSupportedHWClampOp(mlir::Operation*, IE::ClampOp, const LogCb&) {
        return true;
    }

    static bool isSupportedHWPostOp(mlir::Operation* mainOp, IE::PostOpAttr postOpAttr, mlir::Value fusedOutput,
                                    const LogCb& logCb) {
        const auto isSprLUTSupported = [&]() {
            return config::isSprLUTEnabled(mainOp) && !VPU::hasPerChannelQuantizedOutput(fusedOutput);
        };

        return llvm::TypeSwitch<mlir::Attribute, bool>(postOpAttr)
                .template Case<IE::ReluAttr, IE::LeakyReluAttr>([](auto) {
                    return true;
                })
                .template Case<IE::TanhAttr, IE::SigmoidAttr, IE::GeluAttr, IE::HSwishAttr>([&](auto) {
                    return isSprLUTSupported();
                })
                .template Case<IE::SwishAttr>([&](IE::SwishAttr swish) {
                    // Only beta >= 1.0 is supported in sprLUT
                    if (swish.getBeta().getValueAsDouble() < 1.0) {
                        return false;
                    }

                    return isSprLUTSupported();
                })
                .template Case<IE::ExpAttr>([&](auto) {
                    // if the output is in fp32, we cannot use sprlut to estimate exp function
                    for (auto result : mainOp->getResults()) {
                        auto type = result.getType();
                        if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
                            if (shapedType.getElementType().isF32()) {
                                return false;
                            }
                        }
                    }

                    return isSprLUTSupported();
                })
                .Default([&](mlir::Attribute) {
                    logCb(llvm::formatv("{0} at '{1}' is not supported on this HW platform", postOpAttr,
                                        mainOp->getLoc()));
                    return false;
                });
    }

    bool supportsFuseBiasScale(mlir::Operation*) const {
        return _supportsFuseBiasScale;
    }

protected:
    bool _supportsFuseBiasScale = false;
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

void vpux::VPU::arch50xx::registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry) {
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
        // #E157147: Disable fuse postOp for multiply. It will be enabled once it is optimal.
        // IE::MultiplyOp::attachInterface<LayerWithPostOpModel<IE::MultiplyOp>>(*ctx);
        IE::MatMulOp::attachInterface<LayerWithPostOpModel<IE::MatMulOp>>(*ctx);
    });
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::TransposedConvolutionOp::attachInterface<LayerWithPostOpModel<VPU::TransposedConvolutionOp>>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<VPU::GroupConvolutionOp>>(*ctx);
    });
}
