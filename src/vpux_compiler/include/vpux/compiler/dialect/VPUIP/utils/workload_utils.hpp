//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

namespace vpux {
namespace VPUIP {

struct WorkloadComponents {
    SmallVector<int64_t> inStart;
    SmallVector<int64_t> inEnd;
    SmallVector<int64_t> outStart;
    SmallVector<int64_t> outEnd;
    VPU::Padding pad;

    bool operator==(const WorkloadComponents& other) const {
        return inStart == other.inStart && inEnd == other.inEnd && outStart == other.outStart &&
               outEnd == other.outEnd && pad == other.pad;
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "WorkloadComponents { inStart={0}, inEnd={1}, outStart={2}, outEnd={3} pad={4} }", inStart,
                inEnd, outStart, outEnd, pad);
    }
};

inline std::ostream& operator<<(std::ostream& os, const WorkloadComponents& workload) {
    return os << formatv("{0}", workload).str();
}

WorkloadComponents reduceToOneOutputPixel(const WorkloadComponents& workload, ArrayRef<int64_t> kernelSize);

}  // namespace VPUIP
}  // namespace vpux
