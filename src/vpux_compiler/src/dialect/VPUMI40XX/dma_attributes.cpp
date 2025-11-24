//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <vpux/compiler/core/attributes/strides.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/attributes.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/dialect.hpp>
#include <vpux/compiler/utils/types.hpp>

#include "vpux/compiler/utils/dma_transaction_utils.hpp"

using namespace vpux;

DMATransaction VPUMI40XX::NNDMATransactionAttr::getDMATransaction() const {
    DMATransaction dmaTransaction;

    auto inType = mlir::cast<vpux::NDTypeInterface>(getInputType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(getOutputType());
    auto [inputMemShape, inputMemStrides, inputElemSize] = getTypeInfo(inType);
    auto [outputMemShape, outputMemStrides, outputElemSize] = getTypeInfo(outType);

    dmaTransaction.inputs.push_back(
            reduceDimsForDma(std::move(inputMemShape), std::move(inputMemStrides), inputElemSize));
    dmaTransaction.outputs.push_back(
            reduceDimsForDma(std::move(outputMemShape), std::move(outputMemStrides), outputElemSize));

    return dmaTransaction;
}

// PermuteDMATransactionAttr has the same semantic as VPUIP::InternalDataFlowAttr. Some details about the
// VPUIP::InternalDataFlowAttr members can be found together with its the table-gen definition.
DMATransaction VPUMI40XX::PermuteDMATransactionAttr::getDMATransaction() const {
    DMATransaction dmaTransaction;

    auto inType = mlir::cast<vpux::NDTypeInterface>(getInputType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(getOutputType());

    return getDMATransactionFromPermutation(inType, outType, getMappingOrder().getAffineMap(),
                                            getLoopOrder().getAffineMap());
}

DMATransaction VPUMI40XX::ExpandDMATransactionAttr::getDMATransaction() const {
    DMATransaction dmaTransaction;

    auto inType = mlir::cast<vpux::NDTypeInterface>(getInputType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(getOutputType());

    return getDMATransactionFromExpand(inType, outType, getPadsBegin(), getPadsEnd());
}

DMATransaction VPUMI40XX::PerAxisTileDMATransactionAttr::getDMATransaction() const {
    DMATransaction dmaTransaction;

    auto inType = mlir::cast<vpux::NDTypeInterface>(getInputType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(getOutputType());

    auto [inputMemShape, inputMemStrides, inputElemSize] = getTypeInfo(inType);
    auto [outputMemShape, outputMemStrides, outputElemSize] = getTypeInfo(outType);

    auto axis = getAxis().getInt();
    auto tiles = getTiles().getInt();

    auto memAxis = inType.getDimsOrder().dimPos(Dim(axis));

    // Insert fake axis before the replicated axis
    inputMemShape.insert(inputMemShape.begin() + memAxis, tiles);
    inputMemStrides.insert(inputMemStrides.begin() + memAxis, Bit(0));

    dmaTransaction.inputs.push_back(
            reduceDimsForDma(std::move(inputMemShape), std::move(inputMemStrides), inputElemSize));
    dmaTransaction.outputs.push_back(
            reduceDimsForDma(std::move(outputMemShape), std::move(outputMemStrides), outputElemSize));

    return dmaTransaction;
}
