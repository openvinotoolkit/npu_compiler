//
// Copyright (C) 2024-2025 Intel Corporation.
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

bool isSupportedHWPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(postOp)
            .Case<IE::ReLUOp, IE::LeakyReluOp, IE::ClampOp>([](auto) {
                return true;
            })
            .Case<IE::TanhOp, IE::SigmoidOp, IE::SwishOp, IE::GeluOp, IE::HSwishOp>([](auto postOp) {
                if constexpr (std::is_same_v<std::decay_t<decltype(postOp)>, IE::SwishOp>) {
                    auto betaAttr = postOp.getBetaValue();
                    // Only beta >= 1.0 is supported in sprLUT
                    if (betaAttr.has_value() && betaAttr.value().convertToDouble() < 1.0) {
                        return false;
                    }
                }
                // Cannot apply per-channel output scale when using sprLUT.
                return config::isSprLUTEnabled(postOp) && !VPU::hasPerChannelQuantizedOutput(postOp);
            })
            .Case<IE::ExpOp>([&](auto) {
                // if postOp output is in fp32, we cannot use sprlut to estimate exp function
                for (auto result : mainOp->getResults()) {
                    auto type = result.getType();
                    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
                        if (shapedType.getElementType().isF32()) {
                            return false;
                        }
                    }
                }
                return config::isSprLUTEnabled(postOp) && !VPU::hasPerChannelQuantizedOutput(postOp);
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
    });
}
