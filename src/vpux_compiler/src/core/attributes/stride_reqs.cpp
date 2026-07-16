//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include "vpux/compiler/dialect/VPURT/IR/types.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>

using namespace vpux;

//
// StrideReqKind
//

StringLiteral vpux::stringifyEnum(StrideReqKind val) {
    switch (val) {
    case StrideReqKind::Compact:
        return "Compact";
    case StrideReqKind::Aligned:
        return "Aligned";
    case StrideReqKind::Fixed:
        return "Fixed";
    default:
        return "<UNKNOWN>";
    }
}

//
// DimStrideReq
//

void vpux::DimStrideReq::verifyAttrs(StrideReqKind kind, Bit extraValue) {
    if (kind == StrideReqKind::Compact) {
        VPUX_THROW_UNLESS(extraValue.count() == 0, "Got non zero extraValue {0} for {1} DimStrideReq", extraValue,
                          kind);

        return;
    }

    VPUX_THROW_UNLESS(extraValue.count() > 0, "Got negative or zero extraValue {0} in {1} DimStrideReq", extraValue,
                      kind);

    if (kind == StrideReqKind::Aligned) {
        VPUX_THROW_UNLESS(isPowerOfTwo(extraValue.count()),
                          "Alignment value {0} is not a power-of-two in {1} DimStrideReq", extraValue, kind);
    }
}

void vpux::DimStrideReq::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}:{1}", memDim(), kind());

    switch (kind()) {
    case StrideReqKind::Aligned:
    case StrideReqKind::Fixed:
        printTo(stream, ":{0}", _extraValue);
        break;

    default:
        break;
    }
}

bool vpux::operator==(const DimStrideReq& req1, const DimStrideReq& req2) {
    return req1.memDim() == req2.memDim() && req1.kind() == req2.kind() && req1.extraValue() == req2.extraValue();
}

bool vpux::operator!=(const DimStrideReq& req1, const DimStrideReq& req2) {
    return !(req1 == req2);
}

//
// StrideReqs
//

StrideReqs vpux::StrideReqs::simple(size_t numDims) {
    numDims = numDims ? numDims - 1 : 0;

    StrideReqs r;
    r.add(DimStrideReq::compact(MemDim(numDims)));
    return r;
}

StrideReqs vpux::StrideReqs::compact(size_t numDims) {
    DimsOrder::validateNumDims(numDims);

    StrideReqs r;
    for (size_t d = 0; d < numDims; ++d) {
        r.add(DimStrideReq::compact(MemDim(d)));
    }
    return r;
}

StrideReqs vpux::StrideReqs::fixed(StridesRef strides) {
    StrideReqs r;
    for (const auto& p : strides | indexed) {
        r.add(DimStrideReq::fixed(MemDim(p.index()), p.value()));
    }
    return r;
}

StrideReqs& vpux::StrideReqs::add(const DimStrideReq& req) {
    assert(!hasReqFor(req.memDim()));

    _cont.push_back(req);
    return *this;
}

StrideReqs& vpux::StrideReqs::remove(MemDim memDim) {
    const auto it = std::remove_if(_cont.begin(), _cont.end(), [memDim](const DimStrideReq& req) {
        return req.memDim() == memDim;
    });
    _cont.erase(it, _cont.end());
    return *this;
}

bool vpux::StrideReqs::hasReqFor(MemDim memDim) const {
    return (*this)[memDim].has_value();
}

std::optional<DimStrideReq> vpux::StrideReqs::operator[](MemDim memDim) const {
    return StrideReqsRef(*this)[memDim];
}

void vpux::StrideReqs::calcStrides(MemStrides& memStrides, Bit elemSize, MemShapeRef memShape) const {
    StrideReqsRef(*this).calcStrides(memStrides, elemSize, memShape);
}

MemStrides vpux::StrideReqs::calcStrides(Bit elemSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).calcStrides(elemSize, memShape);
}

MemStrides vpux::StrideReqs::calcStrides(const DimsOrder& order, vpux::NDTypeInterface type) const {
    return StrideReqsRef(*this).calcStrides(order, type);
}

bool vpux::StrideReqs::checkStrides(vpux::NDTypeInterface type) const {
    return StrideReqsRef(*this).checkStrides(type);
}

bool vpux::StrideReqs::checkStrides(mlir::Value val) const {
    return StrideReqsRef(*this).checkStrides(val);
}

bool vpux::StrideReqs::checkStrides(MemStridesRef memStrides, Bit elemSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).checkStrides(memStrides, elemSize, memShape);
}

StrideReqs vpux::StrideReqs::join(StrideReqsRef other, Bit elemSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).join(other, elemSize, memShape);
}

void vpux::StrideReqs::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}", raw());
}

//
// StrideReqsRef
//

bool vpux::StrideReqsRef::hasReqFor(MemDim memDim) const {
    return (*this)[memDim].has_value();
}

std::optional<DimStrideReq> vpux::StrideReqsRef::operator[](MemDim memDim) const {
    const auto it = std::find_if(_ref.begin(), _ref.end(), [memDim](const DimStrideReq& req) {
        return req.memDim() == memDim;
    });

    if (it != _ref.end()) {
        return *it;
    } else {
        return std::nullopt;
    }
}

namespace {

Bit applyStrideReq(Bit baseStride, const DimStrideReq& req) {
    switch (req.kind()) {
    case StrideReqKind::Compact:
        return baseStride;
    case StrideReqKind::Aligned:
        return Bit(alignValUp(baseStride.count(), req.alignment().count()));
    case StrideReqKind::Fixed:
        return req.fixedValue();
    default:
        VPUX_THROW("Uncovered stride requirement {0}", req);
    }
}

}  // namespace

void vpux::StrideReqsRef::calcStrides(MemStrides& memStrides, Bit elemSize, MemShapeRef memShape) const {
    assert(memShape.isStatic());

    memStrides.resize(memShape.size());

    if (memShape.empty()) {
        return;
    }

    for (const auto ind : irange(memShape.size()) | reversed) {
        const auto memDim = MemDim(ind);
        const auto req = (*this)[memDim];

        if (ind == memShape.size() - 1) {
            memStrides[memDim] = elemSize;
        } else {
            const auto prevMemDim = MemDim(ind + 1);
            memStrides[memDim] = memStrides[prevMemDim] * memShape[prevMemDim];
        }

        if (req.has_value()) {
            memStrides[memDim] = applyStrideReq(memStrides[memDim], req.value());
        }
    }
}

MemStrides vpux::StrideReqsRef::calcStrides(Bit elemSize, MemShapeRef memShape) const {
    MemStrides memStrides;
    calcStrides(memStrides, elemSize, memShape);
    return memStrides;
}

MemStrides vpux::StrideReqsRef::calcStrides(const DimsOrder& order, vpux::NDTypeInterface type) const {
    const Bit elemSize = type.getElemTypeSize();
    const auto shape = type.getShape();
    const auto memShape = order.toMemoryOrder(shape);
    return calcStrides(elemSize, memShape);
}

bool vpux::StrideReqsRef::checkStrides(vpux::NDTypeInterface type) const {
    const Bit elemSize = type.getElemTypeSize();
    const auto shape = type.getShape();
    const auto strides = type.getStrides();
    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);
    const auto memStrides = order.toMemoryOrder(strides);
    return checkStrides(memStrides, elemSize, memShape);
}

bool vpux::StrideReqsRef::checkStrides(mlir::Value val) const {
    const auto type = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(val.getType());
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return checkStrides(type);
}

bool vpux::StrideReqsRef::checkStrides(MemStridesRef memStrides, Bit elemSize, MemShapeRef memShape) const {
    assert(memShape.isStatic());
    assert(memStrides.isStatic());
    assert(memStrides.size() == memShape.size());

    for (const auto ind : irange(memShape.size())) {
        const auto memDim = MemDim(ind);
        const auto req = (*this)[memDim];
        const auto strideVal = memStrides[memDim];

        if (ind == memShape.size() - 1) {
            if (strideVal < elemSize) {
                return false;
            }
        } else {
            const auto prevMemDim = MemDim(ind + 1);

            // Check the stride if below the compact stride value
            // For StridedSlice operation, there are extreme cases where last value of Output is the last value of Input
            // eg: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] shape==10, stride==3, stridedSlice output is [0, 3, 6, 9]
            // Compact stride value should be memStrides[prevMemDim] * (memShape[prevMemDim] - 1) + elemSize
            if (strideVal != Bit(0) && strideVal < memStrides[prevMemDim] * (memShape[prevMemDim] - 1) + elemSize) {
                return false;
            }

            // To handle abnormal case where shape==0, track ticket E#144962
            if (strideVal == Bit(0) && strideVal < memStrides[prevMemDim] * memShape[prevMemDim]) {
                return false;
            }
        }

        if (!req.has_value()) {
            continue;
        } else if (req.value().kind() == StrideReqKind::Fixed) {
            if (strideVal != req.value().fixedValue()) {
                return false;
            }
        } else if (req.value().kind() == StrideReqKind::Aligned) {
            if (strideVal % req.value().alignment() != 0) {
                return false;
            }
        } else if (req.value().kind() == StrideReqKind::Compact) {
            if (ind == memShape.size() - 1) {
                if (strideVal != elemSize) {
                    return false;
                }
            } else {
                const auto prevMemDim = MemDim(ind + 1);

                if (strideVal != memStrides[prevMemDim] * memShape[prevMemDim]) {
                    return false;
                }
            }
        } else {
            VPUX_THROW("Uncovered stride requirement {0}", req);
        }
    }

    return true;
}

StrideReqs vpux::StrideReqsRef::join(StrideReqsRef other, Bit elemSize, MemShapeRef memShape) const {
    StrideReqs merged;

    const auto mergeReq = [&merged](const DimStrideReq& curReq, const DimStrideReq& otherReq) {
        assert(curReq.memDim() == otherReq.memDim());

        if (curReq.kind() == StrideReqKind::Fixed && otherReq.kind() == StrideReqKind::Fixed) {
            VPUX_THROW_UNLESS(curReq.fixedValue() == otherReq.fixedValue(),
                              "StrideReqs::join : got fixed strides with "
                              "different values at MemDim {0} : "
                              "{1} and {2}",
                              curReq.memDim(), curReq.fixedValue(), otherReq.fixedValue());
        }

        if (otherReq.kind() == StrideReqKind::Fixed) {
            merged.add(otherReq);
        } else {
            merged.add(curReq);
        }
    };

    for (const auto& curReq : *this) {
        const auto otherReq = other[curReq.memDim()];

        if (otherReq.has_value()) {
            mergeReq(curReq, otherReq.value());
        } else {
            merged.add(curReq);
        }
    }

    for (const auto& otherReq : other) {
        const auto curReq = (*this)[otherReq.memDim()];

        if (curReq.has_value()) {
            mergeReq(curReq.value(), otherReq);
        } else {
            merged.add(otherReq);
        }
    }

    const auto mergedStrides = merged.calcStrides(elemSize, memShape);

    VPUX_THROW_UNLESS(checkStrides(mergedStrides, elemSize, memShape),
                      "StrideReqs::join : mergedStrides {0} doesn't satisfy original "
                      "StrideReqs {1} (elemSize {2}, memShape {3})",
                      mergedStrides, *this, elemSize, memShape);
    VPUX_THROW_UNLESS(other.checkStrides(mergedStrides, elemSize, memShape),
                      "StrideReqs::join : mergedStrides {0} doesn't satisfy original "
                      "StrideReqs {1} (elemSize {2}, memShape {3})",
                      mergedStrides, other, elemSize, memShape);

    return merged;
}

void vpux::StrideReqsRef::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}", raw());
}

StrideReqs vpux::StrideReqsRef::toValues() const {
    return StrideReqs(StrideReqs::ContainerType(begin(), end()));
}
