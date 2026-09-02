//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <memory>

namespace vpux::VPUIP {

// In case of more than one DPU task belonging to the same NCEClusterTaskOp, address offsets for the weight table
// buffers must be computed for each variant. The reason is that the buffers are attached to the NCEClusterTaskOp and
// contain weight table data for all variants, while the buffer addresses are part of the DPU variant descriptors.
// The offset computation depends on the weight table layout (data-pointer table or zero-point table).
// These offsets are passed down to the weight table address fields in the DPU variant descriptors and added to the
// start of buffer addresses during the address relocation phase.
//
// WtOffsetBuilder is the base of a small polymorphic hierarchy. It owns the per-variant offset accumulation in
// maybeSetWeightTableOffsetAttr() and delegates the specific alignment to getAlignment(), which each derived builder
// implements. The static create() factory inspects the NCEClusterTaskOp operands and always returns a builder: a
// layout-specific builder when per-variant offsets are required, otherwise a disabled builder whose
// maybeSetWeightTableOffsetAttr() is a no-op. Callers can therefore invoke maybeSetWeightTableOffsetAttr()
// unconditionally without checking for a null builder.
class WtOffsetBuilder {
public:
    virtual ~WtOffsetBuilder() = default;

    WtOffsetBuilder(const WtOffsetBuilder&) = default;
    WtOffsetBuilder& operator=(const WtOffsetBuilder&) = default;
    WtOffsetBuilder(WtOffsetBuilder&&) = default;
    WtOffsetBuilder& operator=(WtOffsetBuilder&&) = default;

    // Selects the offset builder for the weight table layout of nceOp. Always returns a non-null builder; the returned
    // builder's maybeSetWeightTableOffsetAttr() is a no-op when per-variant offsets are not required.
    static std::unique_ptr<WtOffsetBuilder> create(NCEClusterTaskOp nceOp, bool hasMultipleVariants);

    // Same as above, but deriving whether nceOp has multiple variants from the DPU workloads region.
    static std::unique_ptr<WtOffsetBuilder> create(NCEClusterTaskOp nceOp, mlir::Region& workloads);

    // Computes the offset for the DPU variant covering output channels [zStart, zEnd] and sets it on dpuTask.
    // Named as "maybe", because when weight table offset attribute is not needed, this method is a no-op.
    virtual void maybeSetWeightTableOffsetAttr(DPUTaskOp dpuTask, int64_t zStart, int64_t zEnd);

protected:
    WtOffsetBuilder() = default;

    virtual int64_t getAlignment(int64_t zSize) const = 0;

private:
    int64_t _prevZStart = -1;
    int64_t _prevZEnd = -1;
    int64_t _cumulativeWtOffset = 0;
    int64_t _prevClusterId = -1;
};

// Builder for the data-pointer table layout.
class DataPointerTableWtOffsetBuilder final : public WtOffsetBuilder {
protected:
    int64_t getAlignment(int64_t zSize) const override;
};

// Builder for the zero-point table layout.
class ZeroPointTableWtOffsetBuilder final : public WtOffsetBuilder {
public:
    explicit ZeroPointTableWtOffsetBuilder(bool is4bit): _is4bit(is4bit) {
    }

protected:
    int64_t getAlignment(int64_t zSize) const override;

private:
    bool _is4bit;
};

// Builder for cases when per-variant offsets are not required. Its maybeSetWeightTableOffsetAttr() is a no-op,
// letting callers invoke it unconditionally.
class DisabledWtOffsetBuilder final : public WtOffsetBuilder {
public:
    void maybeSetWeightTableOffsetAttr(DPUTaskOp dpuTask, int64_t zStart, int64_t zEnd) override;

protected:
    int64_t getAlignment(int64_t zSize) const override;
};

}  // namespace vpux::VPUIP
