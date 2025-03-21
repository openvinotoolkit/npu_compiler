//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/STLExtras.h>

#include <cassert>
#include <string>

namespace {
constexpr mlir::StringLiteral orderName = "order";
constexpr mlir::StringLiteral stridesName = "strides";
constexpr mlir::StringLiteral allocSizeName = "allocSize";
constexpr mlir::StringLiteral coreAttributesNames[] = {
        allocSizeName,
        orderName,
        stridesName,
};

/// Returns whether attr is a core attribute of MemRefAttr, that is, an
/// attribute with direct accessor function
bool isCoreAttribute(const mlir::NamedAttribute& attr) {
    VPUX_THROW_UNLESS(llvm::is_sorted(coreAttributesNames), "The names must be sorted for the binary search");
    return std::binary_search(std::begin(coreAttributesNames), std::end(coreAttributesNames), attr.getName());
}

// E#88494: leftover: cannot use the attribute key directly - the default
// implementation of the HwSpecificMemRefField interface uses the `mnemonic` of
// the attribute; to preserve compatibility with the existing code (mostly
// MLIR), reformat the string
std::string standardizeMemRefKey(mlir::StringRef key) {
    assert(!key.empty());
    constexpr mlir::StringLiteral attrSuffix = "Attr";
    if (key.ends_with(attrSuffix)) {
        key = key.drop_back(attrSuffix.size());
    }

    auto result = key.str();
    result[0] = std::tolower(result[0]);
    return result;
}

using NamedHwFields = mlir::SmallVector<mlir::NamedAttribute, vpux::MemRefAttr::maxCountOfHwSpecificFields>;

/// Creates a NamedAttribute array of HW-specific fields of MemRefAttr
NamedHwFields makeNamedHwSpecificFields(mlir::MLIRContext* ctx, mlir::ArrayRef<vpux::HwSpecificMemRefField> fields) {
    const auto isValid = [](const vpux::HwSpecificMemRefField& field) {
        // NamedAttribute does not allow nullptr attributes
        return field != nullptr;
    };

    const auto toNamedAttr = [&](const vpux::HwSpecificMemRefField& field) {
        assert(field != nullptr);
        auto rawKey = field.memRefKey();
        auto name = mlir::StringAttr::get(ctx, standardizeMemRefKey(rawKey));
        return mlir::NamedAttribute(name, field);
    };

    return NamedHwFields(fields | vpux::filtered(isValid) | vpux::transformed(toNamedAttr));
}
}  // namespace

namespace vpux {
bool MemRefAttr::classof(mlir::Attribute attr) {
    if (attr == nullptr) {
        return false;
    }

    auto derived = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
    if (derived == nullptr) {
        return false;
    }
    int64_t numAbsentCoreAttrs = 0;

    auto order = derived.get(orderName);
    if (order == nullptr || !mlir::isa<mlir::AffineMapAttr>(order)) {
        return false;
    }

    const auto isSignlessInt64 = [](const mlir::Attribute& attr) {
        return attr && mlir::isa<mlir::IntegerAttr>(attr) &&
               mlir::cast<mlir::IntegerAttr>(attr).getType().isSignlessInteger(64);
    };
    const auto isArrayOfSignlessInt64 = [&](const mlir::Attribute& attr) {
        return mlir::isa<mlir::ArrayAttr>(attr) && llvm::all_of(mlir::cast<mlir::ArrayAttr>(attr), isSignlessInt64);
    };
    auto strides = derived.get(stridesName);
    if (strides == nullptr) {
        ++numAbsentCoreAttrs;
    } else if (!isArrayOfSignlessInt64(strides)) {
        return false;
    }

    auto allocSize = derived.get(allocSizeName);
    if (allocSize == nullptr) {
        ++numAbsentCoreAttrs;
    } else if (!mlir::isa<mlir::IntegerAttr>(allocSize)) {
        return false;
    }

    const auto isValidHwSpecificField = [](const mlir::NamedAttribute& attr) {
        if (isCoreAttribute(attr)) {
            return false;
        }
        auto hwSpecificAttr = mlir::dyn_cast_or_null<vpux::HwSpecificMemRefField>(attr.getValue());
        return hwSpecificAttr != nullptr && attr.getName() == standardizeMemRefKey(hwSpecificAttr.memRefKey());
    };
    const auto numHwSpecificFields = vpux::checked_cast<int64_t>(llvm::count_if(derived, isValidHwSpecificField));

    const auto numPresentCoreOrUnknownAttrs = vpux::checked_cast<int64_t>(derived.size()) - numHwSpecificFields;
    // if there are unknown attributes, the lhs is larger than the rhs
    return numPresentCoreOrUnknownAttrs + numAbsentCoreAttrs ==
           vpux::checked_cast<int64_t>(std::size(coreAttributesNames));
}

MemRefAttr MemRefAttr::get(mlir::AffineMapAttr order, mlir::ArrayAttr strides, mlir::IntegerAttr allocSize,
                           mlir::ArrayRef<vpux::HwSpecificMemRefField> hwFields, mlir::MLIRContext* context) {
    constexpr auto expectedFieldCount = std::size(coreAttributesNames) + MemRefAttr::maxCountOfHwSpecificFields;
    SmallVector<mlir::NamedAttribute, expectedFieldCount> fields;

    VPUX_THROW_WHEN(order == nullptr, "Order is a non-optional attribute");
    auto orderId = mlir::StringAttr::get(context, orderName);
    fields.emplace_back(orderId, order);

    if (strides) {
        auto stridesId = mlir::StringAttr::get(context, stridesName);
        fields.emplace_back(stridesId, strides);
    }

    if (allocSize) {
        auto allocSizeId = mlir::StringAttr::get(context, allocSizeName);
        fields.emplace_back(allocSizeId, allocSize);
    }

    fields.append(makeNamedHwSpecificFields(context, hwFields));

    return mlir::dyn_cast<MemRefAttr>(mlir::DictionaryAttr::get(context, fields));
}

mlir::AffineMapAttr MemRefAttr::order() const {
    auto order = DictionaryAttr::get(orderName);
    // Order is a non-optional attribute, so must be present
    return mlir::cast<mlir::AffineMapAttr>(order);
}

mlir::ArrayAttr MemRefAttr::strides() const {
    auto strides = DictionaryAttr::get(stridesName);
    if (strides == nullptr) {
        return nullptr;
    }
    return mlir::cast<mlir::ArrayAttr>(strides);
}

mlir::IntegerAttr MemRefAttr::allocSize() const {
    auto allocSize = DictionaryAttr::get(allocSizeName);
    if (allocSize == nullptr) {
        return nullptr;
    }
    return mlir::cast<mlir::IntegerAttr>(allocSize);
}

MemRefAttr::HwFields MemRefAttr::hwSpecificFields() const {
    return MemRefAttr::HwFields(*this | vpux::filtered(std::not_fn(isCoreAttribute)) |
                                vpux::transformed([](const mlir::NamedAttribute& attr) {
                                    return mlir::cast<vpux::HwSpecificMemRefField>(attr.getValue());
                                }));
}

mlir::Attribute MemRefAttr::hwSpecificField(mlir::StringRef rawKey) const {
    return DictionaryAttr::get(standardizeMemRefKey(rawKey));
}

//
// MemRefAttrLayout
//

mlir::AffineMap MemRefAttrLayout::getAffineMap(mlir::Attribute attr) const {
    const auto desc = mlir::dyn_cast<MemRefAttr>(attr);
    VPUX_THROW_WHEN(desc == nullptr, "Unsupported MemRef layout '{0}'", attr);

    const auto orderMap = desc.order().getValue();
    if (!desc.strides()) {
        return orderMap;
    }

    const auto elemStrides = parseIntArrayAttr<int64_t>(desc.strides());
    const auto stridesMap = mlir::makeStridedLinearLayoutMap(elemStrides, 0, attr.getContext());

    return stridesMap.compose(orderMap);
}

mlir::LogicalResult MemRefAttrLayout::verifyLayout(mlir::Attribute attr, ArrayRef<int64_t> shape,
                                                   FuncRef<mlir::InFlightDiagnostic()> emitError) const {
    const auto desc = attr.dyn_cast<MemRefAttr>();
    if (desc == nullptr) {
        return printTo(emitError(), "Unsupported MemRef layout '{0}'", attr);
    }

    if (!desc.order().getValue().isPermutation()) {
        return printTo(emitError(), "Dims order '{0}' is not a permutation affine map", desc.order());
    }

    if (auto stridesAttr = desc.strides()) {
        const auto order = DimsOrder::fromAffineMap(desc.order().getValue());

        const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);

        const auto memShape = order.toMemoryOrder(ShapeRef(shape));

        const auto elemSize = 1_Bit;
        const auto strides = Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                                         return stride * elemSize;
                                                     })));
        const auto memStrides = order.toMemoryOrder(strides);

        StrideReqs reqs;

        if (!reqs.checkStrides(memStrides, elemSize, memShape)) {
            return printTo(emitError(), "Strides '{0}' do not match with shape '{1}' and order '{2}'", desc.strides(),
                           shape, order);
        }
    }

    return mlir::success();
}

}  // namespace vpux
