//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/compiler/utils/types.hpp>

namespace vpux {
class NDTypeInterface;

class DMAPattern {
public:
    DMAPattern(mlir::SmallVector<int64_t> dims, mlir::SmallVector<int64_t> strides)
            : dims(std::move(dims)), strides(std::move(strides)) {
    }
    DMAPattern() = default;

    mlir::SmallVector<int64_t> dims;
    mlir::SmallVector<int64_t> strides;
};

class DMATransaction {
public:
    DMATransaction(llvm::SmallVector<DMAPattern> inputs, mlir::SmallVector<DMAPattern> outputs)
            : inputs(std::move(inputs)), outputs(std::move(outputs)) {
    }
    DMATransaction() = default;

    mlir::SmallVector<DMAPattern> inputs;
    mlir::SmallVector<DMAPattern> outputs;
};

DMAPattern reduceDimsForDma(SmallVector<int64_t> memShape, SmallVector<vpux::MemSize<vpux::MemType::Bit>> memStrides,
                            int64_t elemSize);

void patchDimsForNPU37XX(DMAPattern& dmaPattern);

DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::AffineMap loopOrder);

DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::SmallVector<int64_t> loopOrder);

DMATransaction getDMATransactionFromExpand(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                           mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd);

std::tuple<SmallVector<int64_t>, SmallVector<Bit>, int64_t> getTypeInfo(NDTypeInterface type);

}  // namespace vpux
