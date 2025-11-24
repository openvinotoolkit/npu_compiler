//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <array>

using namespace vpux;

//
// DimsOrder
//

constexpr size_t vpux::DimsOrder::BITS_PER_DIM;
constexpr size_t vpux::DimsOrder::MAX_DIM_IND;

const DimsOrder vpux::DimsOrder::C = DimsOrder(0x1);
const DimsOrder vpux::DimsOrder::NC = DimsOrder(0x12);
const DimsOrder vpux::DimsOrder::CN = DimsOrder(0x21);
const DimsOrder vpux::DimsOrder::CHW = DimsOrder(0x123);
const DimsOrder vpux::DimsOrder::HWC = DimsOrder(0x231);
const DimsOrder vpux::DimsOrder::HCW = DimsOrder(0x213);
const DimsOrder vpux::DimsOrder::WHC = DimsOrder(0x321);
const DimsOrder vpux::DimsOrder::CWH = DimsOrder(0x132);
const DimsOrder vpux::DimsOrder::NCHW = DimsOrder(0x1234);
const DimsOrder vpux::DimsOrder::NHWC = DimsOrder(0x1342);
const DimsOrder vpux::DimsOrder::NHCW = DimsOrder(0x1324);
const DimsOrder vpux::DimsOrder::NWHC = DimsOrder(0x1432);
const DimsOrder vpux::DimsOrder::NWCH = DimsOrder(0x1423);
const DimsOrder vpux::DimsOrder::NCWH = DimsOrder(0x1243);
const DimsOrder vpux::DimsOrder::WCHN = DimsOrder(0x4231);
const DimsOrder vpux::DimsOrder::WHNC = DimsOrder(0x4312);
const DimsOrder vpux::DimsOrder::HWCN = DimsOrder(0x3421);
const DimsOrder vpux::DimsOrder::HCNW = DimsOrder(0x3214);
const DimsOrder vpux::DimsOrder::HNWC = DimsOrder(0x3142);
const DimsOrder vpux::DimsOrder::CWNH = DimsOrder(0x2413);
const DimsOrder vpux::DimsOrder::CNHW = DimsOrder(0x2134);
const DimsOrder vpux::DimsOrder::CHWN = DimsOrder(0x2341);
const DimsOrder vpux::DimsOrder::HCWN = DimsOrder(0x3241);

const DimsOrder vpux::DimsOrder::NCDHW = DimsOrder(0x12345);
const DimsOrder vpux::DimsOrder::NDHWC = DimsOrder(0x13452);

const DimsOrder vpux::DimsOrder::GNCHW = DimsOrder(0x12345);
const DimsOrder vpux::DimsOrder::GNHWC = DimsOrder(0x12453);
const DimsOrder vpux::DimsOrder::GHNWC = DimsOrder(0x14253);

const DimsOrder vpux::DimsOrder::OIYX = DimsOrder(0x1234);
const DimsOrder vpux::DimsOrder::OYXI = DimsOrder(0x1342);
const DimsOrder vpux::DimsOrder::YXOI = DimsOrder(0x3412);

const DimsOrder vpux::DimsOrder::GOIYX = DimsOrder(0x12345);
const DimsOrder vpux::DimsOrder::GOYXI = DimsOrder(0x12453);

const DimsOrder vpux::DimsOrder::IOYX = DimsOrder(0x2134);
const DimsOrder vpux::DimsOrder::OYIX = DimsOrder(0x1324);

const DimsOrder vpux::DimsOrder::OGIYX = DimsOrder(0x21345);
const DimsOrder vpux::DimsOrder::GIOYX = DimsOrder(0x13245);

const DimsOrder vpux::DimsOrder::GOIZYX = DimsOrder(0x123456);
const DimsOrder vpux::DimsOrder::GIOZYX = DimsOrder(0x132456);

namespace {

const DimsOrder::StorageType INDEX_MASK = 0xF;

}  // namespace

void vpux::DimsOrder::validateCode(StorageType code) {
    std::array<bool, MAX_NUM_DIMS> dimUsed;
    dimUsed.fill(false);

    size_t numDims = 0;
    auto codeCopy = code;

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, ++numDims, codeCopy >>= DimsOrder::BITS_PER_DIM) {
        auto dimInd = codeCopy & INDEX_MASK;
        if (dimInd == 0) {
            break;
        }

        --dimInd;

        // Check if dimension was used more than once.
        VPUX_THROW_UNLESS(!dimUsed[dimInd], "Dimension {0} was used twice in DimsOrder code {1}", dimInd, code);

        dimUsed[dimInd] = true;
    }

    // The usedDims should contain dimensions in range [0, numDims).

    for (const auto dimInd : irange(MAX_NUM_DIMS)) {
        if (!dimUsed[dimInd]) {
            break;
        }

        VPUX_THROW_UNLESS(dimInd < numDims, "Dimension {0} in DimsOrder code {1} is out of range [0, {2})", dimInd,
                          code, numDims);
    }

    // All digits on positions upper or equal to the order length should be
    // UNDEF.

    codeCopy = code >> (DimsOrder::BITS_PER_DIM * numDims);

    for (size_t i = numDims; i < MAX_NUM_DIMS; ++i, codeCopy >>= DimsOrder::BITS_PER_DIM) {
        auto dimInd = codeCopy & INDEX_MASK;
        VPUX_THROW_UNLESS(dimInd == 0, "DimsOrder code {0} is not contiguous", code);
    }
}

DimsOrder vpux::DimsOrder::fromCode(StorageType code) {
    validateCode(code);

    return DimsOrder(code);
}

void vpux::DimsOrder::validateNumDims(size_t numDims) {
    VPUX_THROW_UNLESS(numDims <= MAX_NUM_DIMS,
                      "Number of Dims {0} in DimsOrder exceeds maximal "
                      "supported value {1}",
                      numDims, MAX_NUM_DIMS);
}

DimsOrder::StorageType vpux::DimsOrder::getCodeFromNumDims(size_t numDims) {
    validateNumDims(numDims);

    StorageType code = 0;

    for (StorageType i = 0; i < numDims; ++i) {
        const StorageType shift = checked_cast<StorageType>(DimsOrder::BITS_PER_DIM * i);
        const StorageType dimDigit = checked_cast<StorageType>(numDims - i);

        code |= (dimDigit << shift);
    }

    return code;
}

DimsOrder vpux::DimsOrder::fromNumDims(size_t numDims) {
    return DimsOrder(getCodeFromNumDims(numDims));
}

void vpux::DimsOrder::validatePermutation(DimArrRef perm) {
    std::array<bool, MAX_NUM_DIMS> dimUsed;
    dimUsed.fill(false);

    for (const auto& p : perm | indexed) {
        const auto& d = p.value();

        VPUX_THROW_UNLESS(static_cast<StorageType>(d.ind()) <= MAX_DIM_IND,
                          "Dim {0} is too large to be used in DimsOrder permutation {1}, "
                          "supported only up to {2}",
                          d, perm, MAX_DIM_IND);

        // The perm should contain dimensions in range [0, perm.size()).
        VPUX_THROW_UNLESS(static_cast<StorageType>(d.ind()) < perm.size(),
                          "Dim {0} is out of DimsOrder permutation range {1}", d, perm);

        // Check if dimension was used more than once.
        VPUX_THROW_UNLESS(!dimUsed[static_cast<size_t>(d.ind())], "Dim {0} was used twice in DimsOrder permutation {1}",
                          d, perm);

        dimUsed[static_cast<size_t>(d.ind())] = true;
    }
}

DimsOrder::StorageType vpux::DimsOrder::getCodeFromPermutation(DimArrRef perm) {
    validatePermutation(perm);

    StorageType code = 0;

    for (const auto& p : perm | indexed) {
        const auto& d = p.value();

        const StorageType dimDigit = checked_cast<StorageType>(d.ind() + 1);

        code <<= BITS_PER_DIM;
        code |= dimDigit;
    }

    return code;
}

DimsOrder vpux::DimsOrder::fromPermutation(DimArrRef perm) {
    return DimsOrder(getCodeFromPermutation(perm));
}

size_t vpux::DimsOrder::numDims() const {
    size_t out = 0;

    auto code = this->code();

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        const auto digit = code & INDEX_MASK;
        if (digit == 0) {
            break;
        }

        ++out;
    }

    return out;
}

bool vpux::DimsOrder::hasDim(Dim d) const {
    const auto dimDigit = static_cast<StorageType>(d.ind()) + 1;

    auto code = this->code();

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        const auto curDigit = code & INDEX_MASK;
        if (curDigit == 0) {
            break;
        }

        if (curDigit == dimDigit) {
            return true;
        }
    }

    return false;
}

size_t vpux::DimsOrder::dimPos(Dim d) const {
    const auto dimDigit = static_cast<StorageType>(d.ind()) + 1;
    auto code = _invertedCode;

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        const auto curDigit = code & INDEX_MASK;
        if (curDigit == 0) {
            break;
        }

        if (curDigit == dimDigit) {
            return i;
        }
    }

    VPUX_THROW("Dim {0} is not available in layout {1}", d, *this);
}

Dim vpux::DimsOrder::dimAt(size_t pos) const {
    auto code = _invertedCode;
    code >>= checked_cast<StorageType>(pos * BITS_PER_DIM);

    const auto curDigit = code & INDEX_MASK;
    VPUX_THROW_UNLESS(curDigit > 0, "DimsOrder {0} doesn't have Dim at pos {1}", *this, pos);

    return Dim(curDigit - 1);
}

MemDim vpux::DimsOrder::toMemDim(Dim d) const {
    return MemDim(dimPos(d));
}

Dim vpux::DimsOrder::toDim(MemDim d) const {
    return dimAt(d.ind());
}

DimArr vpux::DimsOrder::toPermutation() const {
    DimArr out;

    auto code = _invertedCode;

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        auto curDigit = code & INDEX_MASK;
        if (curDigit == 0) {
            break;
        }

        out.emplace_back(curDigit - 1);
    }

    return out;
}

bool vpux::DimsOrder::isIdentity() const {
    return *this == DimsOrder::fromNumDims(numDims());
}

DimsOrder vpux::DimsOrder::fromAffineMap(mlir::AffineMap map) {
    VPUX_THROW_UNLESS(map.isPermutation(), "Can't get DimsOrder from non-permutation AffineMap '{0}'", map);

    const auto perm = to_container<DimArr>(map.getResults() | transformed([](mlir::AffineExpr expr) {
                                               const auto dim = mlir::cast<mlir::AffineDimExpr>(expr);
                                               const auto dimPos = dim.getPosition();
                                               return Dim(dimPos);
                                           }));

    return fromPermutation(perm);
}

mlir::AffineMap vpux::DimsOrder::toAffineMap(mlir::MLIRContext* ctx) const {
    const auto permutation = to_small_vector(toPermutation() | transformed([](Dim d) {
                                                 return static_cast<unsigned>(d.ind());
                                             }));

    return permutation.empty() ? mlir::AffineMap::get(ctx) : mlir::AffineMap::getPermutationMap(permutation, ctx);
}

DimsOrder vpux::DimsOrder::fromValue(mlir::Value val) {
    const auto type = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getDimsOrder();
}

bool vpux::DimsOrder::isCompatibleLayout(mlir::MemRefType type) const {
    if (checked_cast<size_t>(type.getRank()) != numDims()) {
        return false;
    }

    const auto logicalStrides = mlir::cast<vpux::NDTypeInterface>(type).getStrides();
    const auto currPerm = toPermutation();

    if (currPerm.size() <= 1) {
        return true;
    }

    for (size_t i = 0; i < currPerm.size() - 1; i++) {
        if (logicalStrides[currPerm[i]] < logicalStrides[currPerm[i + 1]]) {
            return false;
        }
    }

    return true;
}

bool vpux::DimsOrder::isCompatibleLayout(mlir::Value val) const {
    const auto type = mlir::dyn_cast<mlir::MemRefType>(val.getType());
    VPUX_THROW_UNLESS(type != nullptr, "Can't get DimsOrder from Type '{0}'", val.getType());
    return isCompatibleLayout(type);
}

StringLiteral vpux::DimsOrder::getCanonicalName() const {
    if (*this == DimsOrder()) {
        return StringLiteral("SCALAR");
    } else if (*this == DimsOrder::C) {
        return StringLiteral("C");
    } else if (*this == DimsOrder::NC) {
        return StringLiteral("NC");
    } else if (*this == DimsOrder::CN) {
        return StringLiteral("CN");
    } else if (*this == DimsOrder::CHW) {
        return StringLiteral("CHW");
    } else if (*this == DimsOrder::HWC) {
        return StringLiteral("HWC");
    } else if (*this == DimsOrder::HCW) {
        return StringLiteral("HCW");
    } else if (*this == DimsOrder::NCHW) {
        return StringLiteral("NCHW");
    } else if (*this == DimsOrder::NHWC) {
        return StringLiteral("NHWC");
    } else if (*this == DimsOrder::NHCW) {
        return StringLiteral("NHCW");
    } else if (*this == DimsOrder::NWHC) {
        return StringLiteral("NWHC");
    } else if (*this == DimsOrder::NWCH) {
        return StringLiteral("NWCH");
    } else if (*this == DimsOrder::NCWH) {
        return StringLiteral("NCWH");
    } else if (*this == DimsOrder::NCDHW) {
        return StringLiteral("NCDHW");
    } else if (*this == DimsOrder::NDHWC) {
        return StringLiteral("NDHWC");
    } else if (*this == DimsOrder::GNHWC) {
        return StringLiteral("GNHWC");
    } else if (*this == DimsOrder::YXOI) {
        return StringLiteral("YXOI");
    } else if (*this == DimsOrder::GOIYX) {
        return StringLiteral("GOIYX");
    } else if (*this == DimsOrder::GOYXI) {
        return StringLiteral("GOYXI");
    } else {
        return StringLiteral("");
    }
}

void vpux::DimsOrder::printFormat(llvm::raw_ostream& stream) const {
    if (const auto name = getCanonicalName(); !name.empty()) {
        stream << name;
    } else {
        printTo(stream, "{0}", toPermutation());
    }
}

DimsOrder::DimsOrder(StorageType code): _code(code) {
    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        auto dimDigit = code & INDEX_MASK;
        if (dimDigit == 0) {
            break;
        }

        _invertedCode <<= BITS_PER_DIM;
        _invertedCode |= dimDigit;
    }
}

std::optional<Dim> vpux::getHighestNonTrivialDim(ShapeRef shape, const DimsOrder& dimOrder) {
    for (auto idx : irange(dimOrder.numDims())) {
        auto curDim = dimOrder.dimAt(idx);
        if (shape[curDim] != 1) {
            return curDim;
        }
    }

    return std::nullopt;
}

std::optional<Dim> vpux::getInnermostNonTrivialDim(ShapeRef shape, const DimsOrder& dimOrder) {
    for (auto idx : irange(dimOrder.numDims()) | reversed) {
        auto curDim = dimOrder.dimAt(idx);
        if (shape[curDim] != 1) {
            return curDim;
        }
    }

    return std::nullopt;
}
