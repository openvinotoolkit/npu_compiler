//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

#include <cassert>
#include <string>
#include <vpux/utils/core/format.hpp>

namespace {
constexpr mlir::StringLiteral orderName = "order";
constexpr mlir::StringLiteral stridesName = "strides";
constexpr mlir::StringLiteral boundsName = "bounds";
constexpr mlir::StringLiteral allocSizeName = "allocSize";
constexpr mlir::StringLiteral offsetName = "offset";
constexpr mlir::StringLiteral coreAttributesNames[] = {
        allocSizeName, boundsName, offsetName, orderName, stridesName,
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

// The offset field is only ever present when explicitly set to a non-zero (or dynamic) value via
// MemRefAttr::setOffset; its absence always means static zero.
int64_t getOffsetOrZero(const vpux::MemRefAttr& desc) {
    auto offsetAttr = desc.offset();
    if (offsetAttr == nullptr) {
        return 0;
    }
    return offsetAttr.getValue().getSExtValue();
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

    auto bounds = derived.get(boundsName);
    if (bounds == nullptr) {
        ++numAbsentCoreAttrs;
    } else if (!mlir::isa<vpux::Const::OpaqueI64ElementsAttr>(bounds)) {
        return false;
    }

    auto offset = derived.get(offsetName);
    if (offset == nullptr) {
        ++numAbsentCoreAttrs;
    } else if (!isSignlessInt64(offset)) {
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
                           BoundsRef bounds, mlir::ArrayRef<vpux::HwSpecificMemRefField> hwFields,
                           mlir::MLIRContext* context) {
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

    if (!bounds.empty()) {
        auto boundsId = mlir::StringAttr::get(context, boundsName);
        const auto opaqueAttr = vpux::Const::OpaqueI64ElementsAttr::get(
                mlir::RankedTensorType::get({checked_cast<int64_t>(bounds.size())}, vpux::getSInt64Type(context)),
                bounds.raw());
        fields.emplace_back(boundsId, opaqueAttr);
    }

    fields.append(makeNamedHwSpecificFields(context, hwFields));

    return mlir::cast<MemRefAttr>(mlir::DictionaryAttr::get(context, fields));
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

BoundsRef MemRefAttr::bounds() const {
    auto bounds = DictionaryAttr::get(boundsName);
    if (bounds == nullptr) {
        return {};
    }
    return BoundsRef(mlir::cast<Const::OpaqueI64ElementsAttr>(bounds).getValue());
}

mlir::IntegerAttr MemRefAttr::offset() const {
    auto offset = DictionaryAttr::get(offsetName);
    if (offset == nullptr) {
        return nullptr;
    }
    return mlir::cast<mlir::IntegerAttr>(offset);
}

MemRefAttr MemRefAttr::addOffset(int64_t offset) const {
    auto* context = getContext();

    SmallVector<mlir::NamedAttribute, MemRefAttr::maxCountOfHwSpecificFields + std::size(coreAttributesNames)> fields(
            *this | vpux::filtered([](const mlir::NamedAttribute& attr) {
                return attr.getName() != offsetName;
            }));

    if (offset != 0) {
        auto offsetId = mlir::StringAttr::get(context, offsetName);
        fields.emplace_back(offsetId, mlir::IntegerAttr::get(vpux::getInt64Type(context), offset));
    }

    return mlir::cast<MemRefAttr>(mlir::DictionaryAttr::get(context, fields));
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

mlir::AffineMap MemRefAttr::getAffineMap() const {
    // Delegate to the external model (MemRefAttrLayout)
    auto iface = mlir::cast<mlir::MemRefLayoutAttrInterface>(*this);
    return iface.getAffineMap();
}

//
// MemRefAttrLayout
//

mlir::AffineMap MemRefAttrLayout::getAffineMap(mlir::Attribute attr) const {
    const auto desc = mlir::dyn_cast<MemRefAttr>(attr);
    VPUX_THROW_WHEN(desc == nullptr, "Unsupported MemRef layout '{0}'", attr);

    const auto orderMap = desc.order().getValue();
    if (!desc.strides()) {
        // Offset is set, but in this case it would be dropped; the check below prevents silently losing it.
        VPUX_THROW_WHEN(desc.offset() != nullptr,
                        "MemRefAttr '{0}' has an offset but no strides: cannot build an "
                        "affine map that reflects the offset without a shape",
                        attr);
        return orderMap;
    }

    const auto elemStrides = parseIntArrayAttr<int64_t>(desc.strides());
    const auto stridesMap = mlir::makeStridedLinearLayoutMap(elemStrides, getOffsetOrZero(desc), attr.getContext());

    return stridesMap.compose(orderMap);
}

mlir::LogicalResult MemRefAttrLayout::verifyLayout(mlir::Attribute attr, ArrayRef<int64_t> shape,
                                                   FuncRef<mlir::InFlightDiagnostic()> emitError) const {
    const auto desc = mlir::dyn_cast<vpux::MemRefAttr>(attr);
    if (desc == nullptr) {
        return printTo(emitError(), "Unsupported MemRef layout '{0}'", attr);
    }

    if (!desc.order().getValue().isPermutation()) {
        return printTo(emitError(), "Dims order '{0}' is not a permutation affine map", desc.order());
    }

    if (desc.order().getValue().getNumDims() != shape.size()) {
        return printTo(emitError(), "Dims order '{0}' rank does not match shape '{1}' rank", desc.order(), shape);
    }

    if (auto stridesAttr = desc.strides()) {
        const auto order = DimsOrder::fromAffineMap(desc.order().getValue());

        const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);

        ShapeRef effectiveShape{shape};
        if (!desc.bounds().empty() && ShapeRef(shape).isDynamic()) {
            effectiveShape = ShapeRef(desc.bounds().raw());
        }
        const auto memShape = order.toMemoryOrder(effectiveShape);

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

    if (llvm::any_of(shape, mlir::ShapedType::isDynamic) && desc.bounds().empty()) {
        return printTo(emitError(), "Dynamic shape '{0}' requires bounds in MemRefAttr", shape);
    }

    if (!desc.bounds().empty()) {
        const auto bounds = desc.bounds();
        if (bounds.size() != shape.size()) {
            return printTo(emitError(), "Bounds '{0}' do not match with shape '{1}'", bounds, shape);
        }

        for (const auto& [idx, dimAndBound] : llvm::enumerate(llvm::zip(shape, bounds.raw()))) {
            const auto [dim, bound] = dimAndBound;

            if (bound < 0) {
                return printTo(emitError(), "Bounds '{0}' has negative value '{1}' at dim '{2}'", bounds, bound, idx);
            }

            if (!mlir::ShapedType::isDynamic(dim) && bound != dim) {
                return printTo(emitError(),
                               "Bounds '{0}' does not match shape '{1}' at dim '{2}': expected '{3}', got '{4}'",
                               bounds, shape, idx, dim, bound);
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult MemRefAttrLayout::getStridesAndOffset(mlir::Attribute attr, mlir::ArrayRef<int64_t> shape,
                                                          llvm::SmallVectorImpl<int64_t>& strides,
                                                          int64_t& offset) const {
    const auto desc = mlir::dyn_cast<vpux::MemRefAttr>(attr);
    if (desc == nullptr) {
        return mlir::failure();
    }

    offset = getOffsetOrZero(desc);

    // Note: this is the exact same as MemRefNDTypeInterface::getStrides() but
    // without being able to produce strides in bytes as element size is
    // unknown (perhaps it has to be recorded in the MemRefAttr?).

    if (auto stridesAttr = desc.strides()) {
        strides = parseIntArrayAttr<int64_t>(stridesAttr);
        return mlir::success();
    }

    // For dynamic shapes, either explicit strides or per-dimension upper bounds must be provided on the MemRefAttr,
    // because compact strides cannot be derived from the dynamic shape alone.
    if (ShapeRef(shape).isDynamic() && desc.bounds().empty()) {
        return mlir::failure();
    }
    auto effectiveShape = ShapeRef(shape);
    if (!desc.bounds().empty()) {
        effectiveShape = ShapeRef(desc.bounds().raw());
    }

    const auto order = DimsOrder::fromAffineMap(desc.order().getValue());
    const auto memShape = order.toMemoryOrder(effectiveShape);
    MemStrides memStrides;
    StrideReqs::compact(order.numDims()).calcStrides(memStrides, vpux::Bit(1), memShape);
    const auto logicalStrides = order.toLogicalOrder(memStrides);

    strides.resize(logicalStrides.size());
    std::transform(logicalStrides.begin(), logicalStrides.end(), strides.begin(), [](vpux::Bit value) {
        // Note: as we don't know the element size, "bits" here are effectively
        // the same as "units".
        return value.count();
    });

    return mlir::success();
}

}  // namespace vpux
