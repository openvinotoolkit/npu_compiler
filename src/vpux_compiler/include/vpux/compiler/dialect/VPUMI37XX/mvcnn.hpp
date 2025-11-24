//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "vpux/compiler/dialect/VPUIP/device.hpp"
#include "vpux/utils/core/error.hpp"

#include <cstdint>
#include <limits>

namespace vpux::VPUMI37XX::MVCNN {

using vpux::VPUIP::DType;
using vpux::VPUIP::MemoryLocation;
using vpux::VPUIP::MPE_Mode;
using vpux::VPUIP::Permutation;

enum PPELayerType : uint8_t {
    PPELayerType_STORE = 0,
    PPELayerType_LOAD = 1,
    PPELayerType_CLEAR = 2,
    PPELayerType_NOOP = 3,
    PPELayerType_HALT = 4,
    PPELayerType_ADD = 5,
    PPELayerType_SUB = 6,
    PPELayerType_MULT = 7,
    PPELayerType_LRELU = 8,
    PPELayerType_LRELUX = 9,
    PPELayerType_LPRELU = 10,
    PPELayerType_MAXIMUM = 11,
    PPELayerType_MINIMUM = 12,
    PPELayerType_CEIL = 13,
    PPELayerType_FLOOR = 14,
    PPELayerType_AND = 15,
    PPELayerType_OR = 16,
    PPELayerType_XOR = 17,
    PPELayerType_NOT = 18,
    PPELayerType_ABS = 19,
    PPELayerType_NEG = 20,
    PPELayerType_POW = 21,
    PPELayerType_EXP = 22,
    PPELayerType_SIGMOID = 23,
    PPELayerType_TANH = 24,
    PPELayerType_SQRT = 25,
    PPELayerType_RSQRT = 26,
    PPELayerType_FLEXARB = 27,
    PPELayerType_MIN = PPELayerType_STORE,
    PPELayerType_MAX = PPELayerType_FLEXARB
};

enum PPERoundingMode : int8_t {
    PPERoundingMode_RNE = 0,
    PPERoundingMode_RNTZ = 1,
    PPERoundingMode_RNAZ = 2,
    PPERoundingMode_RUP = 3,
    PPERoundingMode_MIN = PPERoundingMode_RNE,
    PPERoundingMode_MAX = PPERoundingMode_RUP
};

class IndirectDataReference {
    uint64_t _data_index = 999999999999999999ULL;
    uint64_t _sparsity_index = 999999999999999999ULL;
    uint64_t _storage_element_index = 999999999999999999ULL;
    uint32_t _storage_element_size = 0;

public:
    void add_data_index(uint64_t data_index) {
        _data_index = data_index;
    }
    uint64_t data_index() const {
        return _data_index;
    }

    void add_sparsity_index(uint64_t sparsity_index) {
        _sparsity_index = sparsity_index;
    }
    uint64_t sparsity_index() const {
        return _sparsity_index;
    }

    void add_storage_element_index(uint64_t storage_element_index) {
        _storage_element_index = storage_element_index;
    }
    uint64_t storage_element_index() const {
        return _storage_element_index;
    }

    void add_storage_element_size(uint32_t storage_element_size) {
        _storage_element_size = storage_element_size;
    }
    uint32_t storage_element_size() const {
        return _storage_element_size;
    }
};

template <typename T>
struct Vector : public llvm::SmallVector<T> {
    using llvm::SmallVector<T>::SmallVector;
    T Get(typename llvm::SmallVector<T>::size_type index) const {
        VPUX_THROW_UNLESS(index < this->size(), "Out of range");
        return (*this)[index];
    }
};

class PPEFixedFunction {
    Vector<uint8_t> _Ops;
    int32_t _Clamp_Low = std::numeric_limits<int32_t>::min();
    int32_t _Clamp_High = std::numeric_limits<int32_t>::max();
    int32_t _Lrelu_Mult = 1;
    uint32_t _Lrelu_Shift = 0;

public:
    PPEFixedFunction(Vector<uint8_t> Ops, int32_t Clamp_Low, int32_t Clamp_High, int32_t Lrelu_Mult,
                     uint32_t Lrelu_Shift)
            : _Ops(std::move(Ops)),
              _Clamp_Low(Clamp_Low),
              _Clamp_High(Clamp_High),
              _Lrelu_Mult(Lrelu_Mult),
              _Lrelu_Shift(Lrelu_Shift) {
    }

    const Vector<uint8_t>* Ops() const {
        return &_Ops;
    }
    int32_t Clamp_Low() const {
        return _Clamp_Low;
    }
    int32_t Clamp_High() const {
        return _Clamp_High;
    }
    int32_t Lrelu_Mult() const {
        return _Lrelu_Mult;
    }
    uint32_t Lrelu_Shift() const {
        return _Lrelu_Shift;
    }
};

class TensorReference {
    Vector<uint32_t> _dimensions;
    Vector<uint64_t> _bit_strides;
    IndirectDataReference _data = {};
    Vector<uint32_t> _locale_index;
    MVCNN::DType _data_dtype = MVCNN::DType::DType_NOT_SET;
    Vector<uint8_t> _quant_zero;
    Vector<uint16_t> _quant_mult;
    Vector<uint8_t> _quant_shift;
    uint8_t _swizzling_key = 0;

public:
    void add_dimensions(const Vector<uint32_t>& dimensions) {
        _dimensions = dimensions;
    }
    const Vector<uint32_t>* dimensions() const {
        return &_dimensions;
    }

    void add_bit_strides(const Vector<uint64_t>& bit_strides) {
        _bit_strides = bit_strides;
    }
    const Vector<uint64_t>* bit_strides() const {
        return &_bit_strides;
    }

    void add_data(const IndirectDataReference& data) {
        _data = data;
    }
    const IndirectDataReference* data() const {
        return &_data;
    }

    void add_locale_index(const Vector<uint32_t>& locale_index) {
        _locale_index = locale_index;
    }
    const Vector<uint32_t>* locale_index() const {
        return &_locale_index;
    }

    void add_data_dtype(MVCNN::DType data_dtype) {
        _data_dtype = data_dtype;
    }
    MVCNN::DType data_dtype() const {
        return _data_dtype;
    }

    void add_quant_zero(const Vector<uint8_t>& quant_zero) {
        _quant_zero = quant_zero;
    }
    const Vector<uint8_t>* quant_zero() const {
        return &_quant_zero;
    }

    void add_quant_mult(const Vector<uint16_t>& quant_mult) {
        _quant_mult = quant_mult;
    }
    const Vector<uint16_t>* quant_mult() const {
        return &_quant_mult;
    }

    void add_quant_shift(const Vector<uint8_t>& quant_shift) {
        _quant_shift = quant_shift;
    }
    const Vector<uint8_t>* quant_shift() const {
        return &_quant_shift;
    }

    void add_swizzling_key(uint8_t swizzling_key) {
        _swizzling_key = swizzling_key;
    }
    uint8_t swizzling_key() const {
        return _swizzling_key;
    }
};

class PPETask {
    std::optional<PPEFixedFunction> _fixed_function;
    MVCNN::PPERoundingMode _rounding;
    float _fp_scale_data;
    float _fp_prelu_alpha;

public:
    PPETask(std::optional<PPEFixedFunction> fixed_function, MVCNN::PPERoundingMode rounding, float fp_scale_data,
            float fp_prelu_alpha)
            : _fixed_function(std::move(fixed_function)),
              _rounding(rounding),
              _fp_scale_data(fp_scale_data),
              _fp_prelu_alpha(fp_prelu_alpha) {
    }

    const std::optional<PPEFixedFunction>& fixed_function() const {
        return _fixed_function;
    }
    MVCNN::PPERoundingMode rounding() const {
        return _rounding;
    }
    float fp_scale_data() const {
        return _fp_scale_data;
    }
    float fp_prelu_alpha() const {
        return _fp_prelu_alpha;
    }
};

}  // namespace vpux::VPUMI37XX::MVCNN
