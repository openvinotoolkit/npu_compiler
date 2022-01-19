//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/utils/core/error.hpp"

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
        return val <= 0;
    });
}

int64_t vpux::details::calcTotalShapeSize(ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t acc, int64_t d) {
        VPUX_THROW_UNLESS(d > 0, "Can't compute total shape size on dynamic shape");
        return acc * d;
    });
}

//
// Shape
//

ShapeRef vpux::getShape(mlir::ShapedType type) {
    return ShapeRef(type.getShape());
}

ShapeRef vpux::getShape(mlir::Value val) {
    auto type = VPURT::SparseBufferType::getDataType(val).dyn_cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getShape(type);
}

//
// MemShape
//

MemShape vpux::getMemShape(mlir::ShapedType type) {
    const auto dimsOrder = DimsOrder::fromType(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

MemShape vpux::getMemShape(mlir::Value val) {
    auto type = VPURT::SparseBufferType::getDataType(val).dyn_cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getMemShape(type);
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
