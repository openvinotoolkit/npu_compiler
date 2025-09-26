//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/utils/dynamic_shape_propagation.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {

void assignDynamicTypeComponents(TypeComponents& typeComponents, VPU::BoundsRepresentation boundsRepresentation,
                                 ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds) {
    if (boundsRepresentation == VPU::BoundsRepresentation::BOUNDS) {
        typeComponents.setShape(ShapeRef(shape)).setBounds(Bounds(bounds));
    } else {
        const auto boundedShape = makeShape<BoundedShape>(ShapeRef(shape), BoundsRef(bounds));
        typeComponents.setShapeWithRepresentation(shapeCast<DimsMaskedShape>(boundedShape));
    }
}

}  // namespace vpux
