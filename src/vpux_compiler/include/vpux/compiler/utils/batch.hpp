//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <cassert>

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
}  // namespace vpux
