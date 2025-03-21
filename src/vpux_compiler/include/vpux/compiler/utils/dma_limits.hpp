//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <limits>

#include <vpux/compiler/utils/types.hpp>
#include <vpux/utils/core/mem_size.hpp>

namespace vpux::VPUIP::DMA {

template <typename T>
class Limits {
public:
    template <typename Min, typename Max>
    explicit Limits(Min min, Max max): _min(min), _max(max) {
        VPUX_THROW_UNLESS(fitsInStorageType<T>(min) && fitsInStorageType<T>(max),
                          "Incoming values are out of the range of the storage type");

        VPUX_THROW_WHEN(_min > _max, "Min value is larger than Max value");
    }

    T min() const {
        return _min;
    }
    T max() const {
        return _max;
    }

private:
    // min and max are intended to be used as closed interval limits, i.e. [min, max]
    T _min;
    T _max;

    template <typename StorageType, typename IncomingType>
    static typename std::enable_if<std::is_integral<StorageType>::value && std::is_integral<IncomingType>::value,
                                   bool>::type
    fitsInStorageType(const IncomingType value) {
        const auto storageMin = static_cast<intmax_t>(std::numeric_limits<StorageType>::min());
        const auto incomingMin = static_cast<intmax_t>(std::numeric_limits<IncomingType>::min());
        const auto storageMax = static_cast<uintmax_t>(std::numeric_limits<StorageType>::max());
        const auto incomingMax = static_cast<uintmax_t>(std::numeric_limits<IncomingType>::max());

        return !((incomingMin < storageMin && value < static_cast<IncomingType>(storageMin)) ||
                 (incomingMax > storageMax && value > static_cast<IncomingType>(storageMax)));
    }

    template <typename StorageType, typename IncomingType>
    static typename std::enable_if<!std::is_integral<StorageType>::value || !std::is_integral<IncomingType>::value,
                                   bool>::type
    fitsInStorageType(const IncomingType) {
        return true;
    }
};

class SizeLimits : public Limits<int64_t> {
    using Limits<int64_t>::Limits;
};

class StrideLimits : public Limits<int64_t> {
    using Limits<int64_t>::Limits;
};

class TransferLimits : public Limits<Byte> {
    using Limits<Byte>::Limits;
};

class DimCountLimits : public Limits<int64_t> {
    using Limits<int64_t>::Limits;
};

class StrideCountLimits : public Limits<int64_t> {
    using Limits<int64_t>::Limits;
};

class DimLimits {
public:
    class SubLimits {
    public:
        explicit SubLimits(SizeLimits sizeLimits, std::optional<StrideLimits> strideLimits)
                : _sizeLimits(sizeLimits), _strideLimits(strideLimits) {
        }

        SizeLimits getSizeLimits() const {
            return _sizeLimits;
        }
        std::optional<StrideLimits> getStrideLimits() const {
            return _strideLimits;
        }

    private:
        SizeLimits _sizeLimits;
        std::optional<StrideLimits> _strideLimits;
    };

    explicit DimLimits(SizeLimits sizeLimits, std::optional<StrideLimits> strideLimits,
                       std::optional<SubLimits> subLimits)
            : _sizeLimits(sizeLimits), _strideLimits(strideLimits), _subLimits(subLimits) {
    }

    SizeLimits getSizeLimits() const {
        return _sizeLimits;
    }

    std::optional<StrideLimits> getStrideLimits() const {
        return _strideLimits;
    }

    std::optional<SubLimits> getSubLimits() const {
        return _subLimits;
    }

private:
    SizeLimits _sizeLimits;
    std::optional<StrideLimits> _strideLimits;

    std::optional<SubLimits> _subLimits;  // Some NPU generations have "residual" sub-dims that cannot exceed
                                          // the maximum size of the parent dim
};

class EngineLimits {
public:
    EngineLimits(TransferLimits transferLimits, DimCountLimits dimCountLimits, StrideCountLimits strideCountLimits,
                 mlir::SmallVector<DimLimits> dimLimits)
            : _transferLimits(transferLimits),
              _dimCountLimits(dimCountLimits),
              _strideCountLimits(strideCountLimits),
              _dimLimits(std::move(dimLimits)) {
    }

    const auto& getTransferLimits() const {
        return _transferLimits;
    }

    const auto& getDimCountLimits() const {
        return _dimCountLimits;
    }

    const auto& getStrideCountLimits() const {
        return _strideCountLimits;
    }

    ArrayRef<DimLimits> getDimsLimits() const {
        return _dimLimits;
    }

    const auto& getDimLimits(size_t dimIndex) const {
        VPUX_THROW_UNLESS(indexIsInRange(_dimLimits, dimIndex), "Dim index out of range");

        return _dimLimits[dimIndex];
    }

    auto getDimMinSize(size_t dimIndex) const {
        VPUX_THROW_UNLESS(indexIsInRange(_dimLimits, dimIndex), "Dim index out of range");

        return _dimLimits[dimIndex].getSizeLimits().min();
    }

    auto getDimMaxSize(size_t dimIndex) const {
        VPUX_THROW_UNLESS(indexIsInRange(_dimLimits, dimIndex), "Dim index out of range");

        return _dimLimits[dimIndex].getSizeLimits().max();
    }

    auto getMinLength() const {
        return getDimMinSize(0);
    }

    auto getMaxLength() const {
        return getDimMaxSize(0);
    }

    auto getMinNumPlanes() const {
        return getDimMinSize(1);
    }

    auto getMaxNumPlanes() const {
        return getDimMaxSize(1);
    }

    auto getMinDimCount() const {
        return _dimCountLimits.min();
    }

    auto getMaxDimCount() const {
        return _dimCountLimits.max();
    }

    auto getMinStrideCount() const {
        return _strideCountLimits.min();
    }

    auto getMaxStrideCount() const {
        return _strideCountLimits.max();
    }

private:
    // Transfer limits - expressed in Bytes
    TransferLimits _transferLimits;
    // Dim count limits - expressed as number of dims
    DimCountLimits _dimCountLimits;
    // Stride count limits - expressed as number of stride levels
    StrideCountLimits _strideCountLimits;
    // Full dim limits description - expressed as number of elements
    // By convention, lowest order dim (index 0) is expressed in Bytes
    mlir::SmallVector<DimLimits> _dimLimits;

    template <typename T>
    inline bool indexIsInRange(T&& container, size_t index) const {
        return index < container.size();
    }
};

const EngineLimits& getEngineLimits(VPU::ArchKind arch);

}  // namespace vpux::VPUIP::DMA
