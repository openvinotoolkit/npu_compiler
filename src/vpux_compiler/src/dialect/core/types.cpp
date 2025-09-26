//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include <vpux/compiler/dialect/core/types.hpp>

#include <vpux/compiler/dialect/core/IR/tensor_attr.hpp>
#include <vpux/compiler/dialect/core/interfaces/type_interfaces.hpp>
#include <vpux/compiler/utils/attributes.hpp>
#include <vpux/compiler/utils/types.hpp>
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::Core {

//
// BoundedTensorType
//

bool BoundedTensorType::classof(mlir::Type type) {
    auto rankedType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(type);
    if (rankedType == nullptr) {
        return false;
    }

    if (const auto desc = vpux::getTensorAttr(rankedType)) {
        return !desc.getBounds().empty();
    }

    return false;
}

BoundedTensorType BoundedTensorType::get(mlir::Type type, BoundsRef bounds) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

    VPUX_THROW_UNLESS(ndType.getShape().isDynamic(), "Failed to create BoundedTensorType with static tensor shape {0}",
                      ndType);

    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    VPUX_THROW_UNLESS(boundedType == nullptr, "BoundedTensorType must be created with tensor without bounds, got {0}",
                      type);

    auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                          ndType.getMemSpace(), bounds, {});

    return mlir::cast<BoundedTensorType>(tensorType);
}

BoundsRef BoundedTensorType::getBounds() const {
    auto rankedType = mlir::cast<mlir::RankedTensorType>(*this);

    if (const auto desc = vpux::getTensorAttr(rankedType)) {
        return desc.getBounds();
    }

    return {};
}

BoundedShape BoundedTensorType::getDynamicShape() const {
    return makeShape<BoundedShape>(mlir::cast<NDTypeInterface>(*this).getShape(), getBounds());
}

BoundedTensorType BoundedTensorType::changeBounds(BoundsRef bounds) const {
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(*this);
    const auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                                ndType.getMemSpace(), bounds, {});

    return mlir::cast<BoundedTensorType>(tensorType);
}

//
// DynamicDimsMaskTensorType
//

bool DynamicDimsMaskTensorType::classof(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, bool>(type)
            .Case<mlir::RankedTensorType>([](auto rankedType) {
                if (const auto desc = vpux::getTensorAttr(rankedType)) {
                    return !desc.getDynamicDimsMask().empty();
                }
                return false;
            })
            .Case<VPU::DistributedTensorType>([](auto distributedType) {
                if (auto mask = distributedType.getDynamicDimsMask()) {
                    return !mask.empty();
                }
                return false;
            })
            .Default([](mlir::Type /*type*/) -> bool {
                return false;
            });
}

DynamicDimsMaskTensorType DynamicDimsMaskTensorType::get(mlir::Type type, DynamicDimsMaskRef dynamicDimsMask) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

    VPUX_THROW_UNLESS(ndType.getShape().isStatic(),
                      "Failed to create DynamicDimsMaskTensorType with dynamic tensor shape {0}", ndType);

    auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type);
    VPUX_THROW_UNLESS(dynamicDimsMaskType == nullptr,
                      "DynamicDimsMaskTensorType must be created with tensor without dynamic dims mask, got {0}", type);

    auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                          ndType.getMemSpace(), {}, dynamicDimsMask);

    return mlir::cast<DynamicDimsMaskTensorType>(tensorType);
}

DynamicDimsMaskRef DynamicDimsMaskTensorType::getDynamicDimsMask() const {
    auto rankedType = mlir::cast<mlir::RankedTensorType>(*this);

    if (const auto desc = vpux::getTensorAttr(rankedType)) {
        return desc.getDynamicDimsMask();
    }

    return {};
}

DimsMaskedShape DynamicDimsMaskTensorType::getDynamicShape() const {
    return makeShape<DimsMaskedShape>(mlir::cast<NDTypeInterface>(*this).getShape(), getDynamicDimsMask());
}

DynamicDimsMaskTensorType DynamicDimsMaskTensorType::changeDynamicDimsMask(DynamicDimsMaskRef dynamicDimsMask) const {
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(*this);
    const auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                                ndType.getMemSpace(), {}, dynamicDimsMask);

    return mlir::cast<DynamicDimsMaskTensorType>(tensorType);
}

}  // namespace vpux::Core
