//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <cassert>
#include <map>
#include <optional>
#include <vector>

namespace vpux {
struct DebatchedCallOpData final {
    using ValueType = uint32_t;
    DebatchedCallOpData(ValueType callIndex, ValueType batchSize): callOpIndex(callIndex), totalBatchSize(batchSize) {
    }

    ValueType getCallIndex() const {
        return callOpIndex;
    }
    ValueType getBatchSize() const {
        return totalBatchSize;
    }

    static bool canBeDeserialized(const SmallVector<ValueType>& array);
    static DebatchedCallOpData deserialize(const SmallVector<ValueType>& array);
    SmallVector<ValueType> serialize() const;

    std::string to_string() const;

private:
    ValueType callOpIndex;
    ValueType totalBatchSize;
};

class DebatchedCallOpAttributeView {
    DebatchedCallOpData data;

public:
    static constexpr std::string_view name() {
        return "debatched";
    }

    const DebatchedCallOpData& getCallData() const;
    static std::optional<DebatchedCallOpAttributeView> extract(mlir::func::CallOp callOp);

    template <class... Args>
    static DebatchedCallOpAttributeView inject(mlir::func::CallOp callOp, Args&&... args) {
        DebatchedCallOpAttributeView view{std::forward<Args>(args)...};
        view.injectImpl(callOp);
        return view;
    }

    static constexpr std::string_view availableTilesAttrName() {
        return "available_tiles";
    }

    static bool hasAvailableTilesAttr(mlir::func::CallOp callOp);
    static void setAvailableTilesAttr(mlir::func::CallOp callOp, DebatchedCallOpData::ValueType val);
    static void removeAvailableTilesAttr(mlir::func::CallOp callOp);
    static DebatchedCallOpData::ValueType getAvailableTilesVal(mlir::func::CallOp callOp);

    static constexpr std::string_view reorderingAttrName() {
        return "reordering";
    }

    static bool hasReorderingAttr(mlir::func::CallOp callOp);
    static void setReorderingAttr(mlir::func::CallOp callOp);

private:
    template <class... Args>
    DebatchedCallOpAttributeView(Args&&... args): data(std::forward<Args>(args)...) {
    }

    void injectImpl(mlir::func::CallOp callOp) const;
};

struct DebatchCoeffDescription {
    Dim batchPositionIndex = Dims4D::Act::N;
    int64_t desiredBatchValue = 1;

    static DebatchCoeffDescription createFromString(std::string_view descr);
    static DebatchCoeffDescription createFromShapes(ShapeRef inShape, ShapeRef outShape);
    Shape apply(ShapeRef shape) const;
    Shape applyProportionFromShape(ShapeRef fromShape, ShapeRef toShape) const;
    std::string to_string() const;
};

struct DebatchCoefficients {
    static std::optional<DebatchCoefficients> create(std::string_view tensorsCoeffFormatted);
    static Shape applyDefault(ShapeRef shape);
    Shape apply(ShapeRef shape, size_t index) const;
    Shape apply(ShapeRef shape, const std::string& nodeName) const;
    size_t size() const;
    std::optional<DebatchCoeffDescription> getCoefficient(size_t index) const;
    std::optional<DebatchCoeffDescription> getCoefficient(const std::string& nodeName) const;
    std::string to_string(bool includeNodeNames = false) const;

private:
    std::multimap<std::string, DebatchCoeffDescription> orderedInputCoefficients;
};
}  // namespace vpux
