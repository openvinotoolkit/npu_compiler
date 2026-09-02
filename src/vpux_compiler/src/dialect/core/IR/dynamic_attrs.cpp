//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

using namespace vpux;

namespace {

inline void assertBound(int64_t dimValue, int64_t bound) {
    VPUX_THROW_WHEN(bound < 0, "Got negative shape dim bound: '{0}'", bound);
    VPUX_THROW_WHEN(dimValue != bound && dimValue > 0, "Got mismatching shape dim size: '{0}' and bound: '{1}'",
                    dimValue, bound);
}

inline void assertMask(int64_t dimValue) {
    VPUX_THROW_WHEN(dimValue < 0, "Got negative shape dim size or bound: '{0}'", dimValue);
}

}  // namespace

BoundsRef vpux::getBounds(mlir::Type type) {
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        return boundedType.getBounds();
    }
    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
        if (auto memRefAttr = mlir::dyn_cast_or_null<vpux::MemRefAttr>(memref.getLayout())) {
            return memRefAttr.bounds();
        }
    }

    return {};
}

DynamicDimsMaskRef vpux::getDynamicDimsMask(mlir::Type type) {
    if (auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type)) {
        return dynamicDimsMaskType.getDynamicDimsMask();
    }

    return {};
}

//
// BoundedDim
//

BoundedDim::BoundedDim(int64_t dimValue, int64_t bound): _dimValue(dimValue), _bound(bound) {
    assertBound(_dimValue, _bound);
}

BoundedDim::BoundedDim(int64_t dimValue): BoundedDim(dimValue, /*bound=*/dimValue) {
}

BoundedDim::BoundedDim(const MaskedDim& maskedDim): BoundedDim(maskedDim.dimValue(), maskedDim.reifiedSize()) {
}

int64_t BoundedDim::dimValue() const {
    return _dimValue;
}

int64_t BoundedDim::representation() const {
    return _bound;
}

bool BoundedDim::isDynamic() const {
    return _dimValue == mlir::ShapedType::kDynamic;
}

int64_t BoundedDim::reifiedSize() const {
    return _bound;
}

BoundedDim BoundedDim::operator+(const BoundedDim& other) const {
    return apply(*this, other, std::plus<>());
}

BoundedDim BoundedDim::operator-(const BoundedDim& other) const {
    return apply(*this, other, std::minus<>());
}

BoundedDim BoundedDim::operator*(const BoundedDim& other) const {
    return apply(*this, other, std::multiplies<>());
}

BoundedDim BoundedDim::operator/(const BoundedDim& other) const {
    return apply(*this, other, std::divides<>());
}

BoundedDim& BoundedDim::operator+=(const BoundedDim& other) {
    return *this = *this + other;
}

BoundedDim& BoundedDim::operator-=(const BoundedDim& other) {
    return *this = *this - other;
}

BoundedDim& BoundedDim::operator*=(const BoundedDim& other) {
    return *this = *this * other;
}

BoundedDim& BoundedDim::operator/=(const BoundedDim& other) {
    return *this = *this / other;
}

bool BoundedDim::operator==(const BoundedDim& other) const {
    return this->reifiedSize() == other.reifiedSize();
}

bool BoundedDim::operator!=(const BoundedDim& other) const {
    return !(*this == other);
}

bool BoundedDim::operator<(const BoundedDim& other) const {
    return this->reifiedSize() < other.reifiedSize();
}

bool BoundedDim::operator>(const BoundedDim& other) const {
    return this->reifiedSize() > other.reifiedSize();
}

bool BoundedDim::operator<=(const BoundedDim& other) const {
    return this->reifiedSize() <= other.reifiedSize();
}

bool BoundedDim::operator>=(const BoundedDim& other) const {
    return this->reifiedSize() >= other.reifiedSize();
}

BoundedDim vpux::operator+(int64_t x, const BoundedDim& y) {
    return BoundedDim::apply(BoundedDim(x), y, std::plus<>());
}

BoundedDim vpux::operator-(int64_t x, const BoundedDim& y) {
    return BoundedDim::apply(BoundedDim(x), y, std::minus<>());
}

BoundedDim vpux::operator*(int64_t x, const BoundedDim& y) {
    return BoundedDim::apply(BoundedDim(x), y, std::multiplies<>());
}

BoundedDim vpux::operator/(int64_t x, const BoundedDim& y) {
    return BoundedDim::apply(BoundedDim(x), y, std::divides<>());
}

bool vpux::operator==(int64_t x, const BoundedDim& y) {
    return x == y.reifiedSize();
}

bool vpux::operator!=(int64_t x, const BoundedDim& y) {
    return !(x == y);
}

bool vpux::operator<(int64_t x, const BoundedDim& y) {
    return x < y.reifiedSize();
}

bool vpux::operator>(int64_t x, const BoundedDim& y) {
    return x > y.reifiedSize();
}

bool vpux::operator<=(int64_t x, const BoundedDim& y) {
    return x <= y.reifiedSize();
}

bool vpux::operator>=(int64_t x, const BoundedDim& y) {
    return x >= y.reifiedSize();
}

//
// MaskedDim
//

MaskedDim::MaskedDim(int64_t dimValue, int64_t isDynamic): _dimValue(dimValue), _isDynamic(isDynamic) {
    assertMask(_dimValue);
}

MaskedDim::MaskedDim(int64_t dimValue): MaskedDim(dimValue, /*isDynamic=*/false) {
}

MaskedDim::MaskedDim(const BoundedDim& boundedDim): MaskedDim(boundedDim.reifiedSize(), boundedDim.isDynamic()) {
}

int64_t MaskedDim::dimValue() const {
    return _isDynamic ? mlir::ShapedType::kDynamic : _dimValue;
}

int64_t MaskedDim::representation() const {
    return _isDynamic;
}

bool MaskedDim::isDynamic() const {
    return _isDynamic;
}

int64_t MaskedDim::reifiedSize() const {
    return _dimValue;
}

MaskedDim MaskedDim::operator+(const MaskedDim& other) const {
    return apply(*this, other, std::plus<>());
}

MaskedDim MaskedDim::operator-(const MaskedDim& other) const {
    return apply(*this, other, std::minus<>());
}

MaskedDim MaskedDim::operator*(const MaskedDim& other) const {
    return apply(*this, other, std::multiplies<>());
}

MaskedDim MaskedDim::operator/(const MaskedDim& other) const {
    return apply(*this, other, std::divides<>());
}

MaskedDim& MaskedDim::operator+=(const MaskedDim& other) {
    return *this = *this + other;
}

MaskedDim& MaskedDim::operator-=(const MaskedDim& other) {
    return *this = *this - other;
}

MaskedDim& MaskedDim::operator*=(const MaskedDim& other) {
    return *this = *this * other;
}

MaskedDim& MaskedDim::operator/=(const MaskedDim& other) {
    return *this = *this / other;
}

bool MaskedDim::operator==(const MaskedDim& other) const {
    return this->reifiedSize() == other.reifiedSize();
}

bool MaskedDim::operator!=(const MaskedDim& other) const {
    return !(*this == other);
}

bool MaskedDim::operator<(const MaskedDim& other) const {
    return this->reifiedSize() < other.reifiedSize();
}

bool MaskedDim::operator>(const MaskedDim& other) const {
    return this->reifiedSize() > other.reifiedSize();
}

bool MaskedDim::operator<=(const MaskedDim& other) const {
    return this->reifiedSize() <= other.reifiedSize();
}

bool MaskedDim::operator>=(const MaskedDim& other) const {
    return this->reifiedSize() >= other.reifiedSize();
}

MaskedDim vpux::operator+(int64_t x, const MaskedDim& y) {
    return MaskedDim::apply(MaskedDim(x), y, std::plus<>());
}

MaskedDim vpux::operator-(int64_t x, const MaskedDim& y) {
    return MaskedDim::apply(MaskedDim(x), y, std::minus<>());
}

MaskedDim vpux::operator*(int64_t x, const MaskedDim& y) {
    return MaskedDim::apply(MaskedDim(x), y, std::multiplies<>());
}

MaskedDim vpux::operator/(int64_t x, const MaskedDim& y) {
    return MaskedDim::apply(MaskedDim(x), y, std::divides<>());
}

bool vpux::operator==(int64_t x, const MaskedDim& y) {
    return x == y.reifiedSize();
}

bool vpux::operator!=(int64_t x, const MaskedDim& y) {
    return !(x == y);
}

bool vpux::operator<(int64_t x, const MaskedDim& y) {
    return x < y.reifiedSize();
}

bool vpux::operator>(int64_t x, const MaskedDim& y) {
    return x > y.reifiedSize();
}

bool vpux::operator<=(int64_t x, const MaskedDim& y) {
    return x <= y.reifiedSize();
}

bool vpux::operator>=(int64_t x, const MaskedDim& y) {
    return x >= y.reifiedSize();
}
