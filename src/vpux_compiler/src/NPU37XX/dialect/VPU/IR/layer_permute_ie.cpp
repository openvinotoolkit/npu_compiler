//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_utils/layer_permute_ie.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

namespace {

//
// LayerWithPermuteInterface
//

template <class MainOpType>
class LayerWithPermuteInterface final :
        public IE::LayerWithPermuteInterface::ExternalModel<LayerWithPermuteInterface<MainOpType>, MainOpType> {
public:
    bool isSupportedPermutation(mlir::Operation* nceOp, mlir::Operation* permuteOp) const {
        auto concreteOp = mlir::cast<MainOpType>(nceOp);

        if (!VPU::isSupportedPermutation(nceOp, permuteOp)) {
            return false;
        }

        return VPU::NCEInvariant::verifyKernel(concreteOp).succeeded();
    }
};

}  // namespace

//
// registerLayerWithPermuteInterfaceForIE
//

void vpux::VPU::arch37xx::registerLayerWithPermuteInterfaceForIE(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::GroupConvolutionOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::TransposedConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<LayerWithPermuteInterface<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<LayerWithPermuteInterface<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<LayerWithPermuteInterface<IE::AddOp>>(*ctx);
        IE::InterpolateOp::attachInterface<LayerWithPermuteInterface<IE::InterpolateOp>>(*ctx);
    });
}
