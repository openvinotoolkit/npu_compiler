//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::Value VPUIP::PermuteCastOp::getViewSource() {
    return getSource();
}

//
// fold
//

mlir::OpFoldResult vpux::VPUIP::PermuteCastOp::fold(FoldAdaptor adaptor) {
    if (getSource().getType() == getResult().getType() && getMemPerm().isIdentity()) {
        return getSource();
    }

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getSource())) {
        // This is a fallback solution. In some cases we get VPUIP::PermuteCastOps that should not
        // be allowed. However, the verifier doesn't check this.
        // TODO: #-141102 Remove this fallback solution as soon as the correct verifier is implemented.
        mlir::SmallVector<mlir::Type> inferredReturnTypes;
        VPU::inferPermuteReturnTypes(getSource(), getMemPerm(), getDstOrder(), inferredReturnTypes);
        if (inferredReturnTypes.front() != getResult().getType()) {
            auto restored = static_cast<Const::ContentAttr>(attr);
            if (restored.getType().getShape() != getShape(getResult())) {
                restored = restored.transform().reshape(getShape(getResult())).get();
            }
            return restored.transform().reorder(DimsOrder::fromAffineMap(getDstOrder())).get();
        }

        // PermuteCastOp ensures that it is always a trivial permutation. That's why we can just add MemPermuteAttr
        // which will not perform any data movements.
        auto result =
                attr.transform()
                        .memPermute(DimsOrder::fromAffineMap(getDstOrder()), DimsOrder::fromAffineMap(getMemPerm()))
                        .get();
        return result;
    }

    return nullptr;
}

mlir::LogicalResult vpux::VPUIP::PermuteCastOp::verify() {
    const auto op = getOperation();
    auto distributedInType = getSource().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = getResult().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        auto outputDistribution = distributedOutType.getDistribution();

        auto expectedOutputDistribution = applyPermutationOnDistributionInfoAttr(
                distributedInType, getMemPerm(), distributedInType.getDimsOrder(), distributedOutType.getDimsOrder(),
                distributedInType.getShape(), distributedOutType.getShape());
        if (mlir::failed(expectedOutputDistribution)) {
            return errorAt(op, "PermuteCast unsupported input distribution: in = {0}",
                           distributedInType.getDistribution());
        }

        if (outputDistribution != expectedOutputDistribution.value()) {
            return errorAt(op,
                           "PermuteCast input and output distributions are incompatible: in = {0}, out = {1},"
                           "expected = {2}",
                           distributedInType.getDistribution(), outputDistribution, expectedOutputDistribution.value());
        }
    }

    const auto inType = getSource().getType().cast<vpux::NDTypeInterface>();
    const auto outType = getResult().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "PermuteCast input and output must have the same number of elements");
    }

    const auto inRank = inType.getRank();
    if (inRank != getDstOrder().getNumDims()) {
        return errorAt(op, "PermuteCast input rank {0} does not match 'dst_order' {1}", inRank,
                       getDstOrder().getNumDims());
    }
    if (inRank != getMemPerm().getNumDims()) {
        return errorAt(op, "PermuteCast input rank {0} does not match 'dst_order' {1}", inRank,
                       getMemPerm().getNumDims());
    }

    return mlir::success();
}
