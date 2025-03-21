//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <cassert>

namespace vpux {

//
// StrideReqKind
//

// Kind of stride requirement

enum class StrideReqKind : int32_t {
    Compact = 0,  // Stride must be compact for specified Dimension
    Aligned = 1,  // Stride must be aligned by some value for specified Dimension
    Fixed = 2,    // Stride will have fixed value
};

StringLiteral stringifyEnum(StrideReqKind val);

//
// DimStrideReq
//

class DimStrideReq final {
public:
    static void verifyAttrs(StrideReqKind kind, Bit extraValue);

public:
    DimStrideReq() = default;

    DimStrideReq(MemDim memDim, StrideReqKind kind, Bit extraValue)
            : _memDim(memDim), _kind(kind), _extraValue(extraValue) {
        verifyAttrs(_kind, _extraValue);
    }

public:
    static DimStrideReq compact(MemDim memDim) {
        return DimStrideReq(memDim, StrideReqKind::Compact, 0_Bit);
    }

    static DimStrideReq aligned(MemDim memDim, Bit alignment) {
        return DimStrideReq(memDim, StrideReqKind::Aligned, alignment);
    }

    static DimStrideReq fixed(MemDim memDim, Bit fixedValue) {
        return DimStrideReq(memDim, StrideReqKind::Fixed, fixedValue);
    }

public:
    MemDim memDim() const {
        return _memDim;
    }

    StrideReqKind kind() const {
        return _kind;
    }

    Bit alignment() const {
        assert(kind() == StrideReqKind::Aligned);
        return Bit(_extraValue);
    }

    Bit fixedValue() const {
        assert(kind() == StrideReqKind::Fixed);
        return Bit(_extraValue);
    }

    Bit extraValue() const {
        return _extraValue;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const;

private:
    MemDim _memDim;
    StrideReqKind _kind = StrideReqKind::Compact;
    Bit _extraValue;
};

bool operator==(const DimStrideReq& req1, const DimStrideReq& req2);
bool operator!=(const DimStrideReq& req1, const DimStrideReq& req2);

//
// StrideReqs
//

// Container for stride requirement per each dimensions.

class StrideReqsRef;

class StrideReqs final {
public:
    using ContainerType = SmallVector<DimStrideReq>;

    using value_type = typename ContainerType::value_type;

    using iterator = typename ContainerType::iterator;
    using reverse_iterator = typename ContainerType::reverse_iterator;

    using const_iterator = typename ContainerType::const_iterator;
    using const_reverse_iterator = typename ContainerType::const_reverse_iterator;

    using size_type = typename ContainerType::size_type;

public:
    StrideReqs() = default;

    explicit StrideReqs(ContainerType&& cont): _cont(std::move(cont)) {
    }

public:
    static StrideReqs simple(size_t numDims);
    static StrideReqs compact(size_t numDims);
    static StrideReqs fixed(StridesRef strides);

public:
    StrideReqs& add(const DimStrideReq& req);
    StrideReqs& remove(MemDim memDim);

public:
    bool hasReqFor(MemDim memDim) const;
    std::optional<DimStrideReq> operator[](MemDim memDim) const;

public:
    void calcStrides(MemStrides& memStrides, Bit elemSize, MemShapeRef memShape) const;

    MemStrides calcStrides(Bit elemSize, MemShapeRef memShape) const;
    MemStrides calcStrides(DimsOrder order, vpux::NDTypeInterface type) const;

public:
    bool checkStrides(vpux::NDTypeInterface type) const;
    bool checkStrides(mlir::Value val) const;
    bool checkStrides(MemStridesRef memStrides, Bit elemSize, MemShapeRef memShape) const;

public:
    StrideReqs join(StrideReqsRef other, Bit elemSize, MemShapeRef memShape) const;

public:
    size_t size() const {
        return _cont.size();
    }

    bool empty() const {
        return _cont.empty();
    }

public:
    iterator begin() {
        return _cont.begin();
    }
    iterator end() {
        return _cont.end();
    }

    const_iterator begin() const {
        return _cont.begin();
    }
    const_iterator end() const {
        return _cont.end();
    }

    reverse_iterator rbegin() {
        return _cont.rbegin();
    }
    reverse_iterator rend() {
        return _cont.rend();
    }

    const_reverse_iterator rbegin() const {
        return _cont.rbegin();
    }
    const_reverse_iterator rend() const {
        return _cont.rend();
    }

public:
    bool operator==(const StrideReqs& other) const {
        return _cont == other._cont;
    }
    bool operator!=(const StrideReqs& other) const {
        return _cont != other._cont;
    }

public:
    const ContainerType& raw() const {
        return _cont;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const;

private:
    ContainerType _cont;
};

//
// StrideReqsRef
//

class StrideReqsRef final {
    using BaseRef = ArrayRef<DimStrideReq>;

public:
    using value_type = typename BaseRef::value_type;

    using iterator = typename BaseRef::iterator;
    using reverse_iterator = typename BaseRef::reverse_iterator;

    using const_iterator = typename BaseRef::const_iterator;
    using const_reverse_iterator = typename BaseRef::const_reverse_iterator;

    using size_type = typename BaseRef::size_type;

public:
    StrideReqsRef() = default;

    StrideReqsRef(const StrideReqs& reqs): _ref(reqs.raw()) {
    }

    explicit StrideReqsRef(BaseRef ref): _ref(ref) {
    }

public:
    bool hasReqFor(MemDim memDim) const;
    std::optional<DimStrideReq> operator[](MemDim memDim) const;

public:
    void calcStrides(MemStrides& memStrides, Bit elemSize, MemShapeRef memShape) const;

    MemStrides calcStrides(Bit elemSize, MemShapeRef memShape) const;
    MemStrides calcStrides(DimsOrder order, vpux::NDTypeInterface type) const;

public:
    bool checkStrides(vpux::NDTypeInterface type) const;
    bool checkStrides(mlir::Value val) const;
    bool checkStrides(MemStridesRef memStrides, Bit elemSize, MemShapeRef memShape) const;

public:
    StrideReqs join(StrideReqsRef other, Bit elemSize, MemShapeRef memShape) const;

public:
    size_t size() const {
        return _ref.size();
    }

    bool empty() const {
        return _ref.empty();
    }

public:
    iterator begin() const {
        return _ref.begin();
    }
    iterator end() const {
        return _ref.end();
    }

    reverse_iterator rbegin() const {
        return _ref.rbegin();
    }
    reverse_iterator rend() const {
        return _ref.rend();
    }

public:
    bool operator==(StrideReqsRef other) const {
        return _ref == other._ref;
    }
    bool operator!=(StrideReqsRef other) const {
        return _ref != other._ref;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const;

public:
    StrideReqs toValues() const;

public:
    const BaseRef& raw() const {
        return _ref;
    }

private:
    BaseRef _ref;
};

}  // namespace vpux
