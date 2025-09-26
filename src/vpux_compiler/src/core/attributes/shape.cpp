//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>
#include <numeric>

#include <cassert>
#include <cstdint>

using namespace vpux;

//
// Shape utils
//

bool vpux::details::isDynamicDimValues(ArrayRef<int64_t> shape) {
    return std::any_of(shape.begin(), shape.end(), [](int64_t val) {
        return val < 0;
    });
}

int64_t vpux::details::calcTotalShapeSize(ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t acc, int64_t d) {
        VPUX_THROW_UNLESS(d >= 0, "Can't compute total shape size on dynamic shape");
        const auto mult = acc * d;
        VPUX_THROW_UNLESS(mult >= 0, "integer overflow in element count computation");
        return mult;
    });
}

//
// Shape
//

ShapeRef vpux::getShape(mlir::Value val) {
    auto type = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getShape();
}

//
// getBoundedShape
//

ShapeRef vpux::getBoundedShape(mlir::Value val) {
    return getBoundedShape(val.getType());
}

ShapeRef vpux::getBoundedShape(mlir::Type type) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    if (auto boundedShape = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        return ShapeRef{boundedShape.getBounds().raw()};
    }

    if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
        if (!mlir::isa<Core::BoundedTensorType>(sparseType.getData())) {
            return ndType.getShape();
        }

        auto dynEffectiveShape = mlir::cast<Core::BoundedTensorType>(VPU::getEffectiveSparseOutputType(sparseType));
        return ShapeRef{dynEffectiveShape.getBounds().raw()};
    }

    return ndType.getShape();
}

//
// MemShape
//

MemShape vpux::getMemShape(mlir::Value val) {
    auto type = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getMemShape();
}

//
// getBoundedMemShape
//

MemShape vpux::getBoundedMemShape(mlir::Value val) {
    auto type = mlir::cast<vpux::NDTypeInterface>(val.getType());
    return getBoundedMemShape(type);
}

MemShape vpux::getBoundedMemShape(mlir::Type type) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

    const auto dimsOrder = ndType.getDimsOrder();
    const auto shape = getBoundedShape(ndType);
    return dimsOrder.toMemoryOrder(shape);
}

MemShape vpux::getMemIndexND(int64_t memIndex1D, MemShapeRef memShape) {
    MemShape memIndexND(memShape.size());

    int64_t tempIndex1D = memIndex1D;

    // memShape stores shape in major to minor
    for (const auto ind : irange(memShape.size()) | reversed) {
        const auto md = MemDim(ind);
        const auto mdSize = memShape[md];

        memIndexND[md] = tempIndex1D % mdSize;
        tempIndex1D = (tempIndex1D - memIndexND[md]) / mdSize;

        VPUX_THROW_UNLESS(tempIndex1D >= 0, "Memory index 1D '{0}' is not compatible with memory shape '{1}'",
                          memIndex1D, memShape);
    }

    VPUX_THROW_UNLESS(tempIndex1D == 0, "Memory index 1D '{0}' is not compatible with memory shape '{1}'", memIndex1D,
                      memShape);

    return memIndexND;
}

int64_t vpux::getMemIndex1D(MemShapeRef memIndexND, MemShapeRef memShape) {
    VPUX_THROW_UNLESS(memIndexND.size() == memShape.size(),
                      "Memory index ND '{0}' is not compatible with memory shape '{1}'", memIndexND, memShape);

    int64_t memIndex1D = 0;

    int64_t tempSize = 1;

    // memShape stores shape in major to minor
    for (const auto ind : irange(memShape.size()) | reversed) {
        const auto md = MemDim(ind);
        const auto mdSize = memShape[md];
        const auto mdInd = memIndexND[md];

        VPUX_THROW_UNLESS(mdInd < mdSize, "Memory index ND '{0}' is not compatible with memory shape '{1}'", memIndexND,
                          memShape);

        memIndex1D += mdInd * tempSize;
        tempSize *= mdSize;
    }

    return memIndex1D;
}
