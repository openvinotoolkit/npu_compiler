// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <numeric>

using namespace vpux;

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::SparsifyAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                      mlir::BoolAttr compressOutputType,
                                                      mlir::ElementsAttr numActualElements) {
    if (compressOutputType == nullptr) {
        return printTo(emitError(), "Got NULL 'compressOutputType' in 'SparsifyAttr'");
    }
    if (numActualElements != nullptr) {
        if (!numActualElements.getShapedType().getElementType().isIntOrIndex()) {
            return printTo(emitError(), "Got unsupported 'numActualElements' in 'SparsifyAttr' : '{0}'",
                           numActualElements.getShapedType().getElementType());
        }
        if (!numActualElements.isa<mlir::DenseElementsAttr>()) {
            return printTo(emitError(), "Got unsupported 'numActualElements' in 'SparsifyAttr'");
        }
        auto values = numActualElements.getValues<int64_t>();
        for (const auto value : values) {
            if (value < 0) {
                return printTo(emitError(),
                               "Got negative elem number value '{0}' in 'numActualElements' for 'SparsifyAttr'", value);
            }
        }
    }

    return mlir::success();
}

//
// SparsifyAttr::print
//

void vpux::Const::SparsifyAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getCompressOutputType());
    if (getNumActualElements() != nullptr) {
        printer << ", ";
        printer.printAttribute(getNumActualElements());
    }
    printer << ">";
}

//
// SparsifyAttr::parse
//

mlir::Attribute vpux::Const::SparsifyAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::BoolAttr compressOutputType;
    if (mlir::failed(parser.parseAttribute(compressOutputType))) {
        return nullptr;
    }

    mlir::ElementsAttr numActualElements;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::failed(parser.parseAttribute(numActualElements))) {
            return nullptr;
        }
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::SparsifyAttr>(compressOutputType, numActualElements);
}

//
// SparsifyAttr::inferOutputType
//

vpux::NDTypeInterface compressType(vpux::NDTypeInterface inputType, ArrayRef<int64_t> numElemsPerOC) {
    const auto elemByteSize = getElemTypeSize(inputType).to<Byte>().count();
    int64_t totalByteSize = 0;
    for (auto num : numElemsPerOC) {
        totalByteSize += alignValUp<int64_t>(num * elemByteSize, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
    }
    const auto newShape = Shape({totalByteSize, 1, 1, 1});
    return inputType.changeShapeElemType(newShape, getUInt8Type(inputType.getContext()));
}

vpux::NDTypeInterface vpux::Const::SparsifyAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    // Note: type inference optionally returns compressed type - see
    // ::transfom() for details.
    if (const bool returnCompressed = getCompressOutputType().getValue(); returnCompressed) {
        VPUX_THROW_WHEN(getNumActualElements() == nullptr, "Missing number of actual elements");
        const auto numElemsPerOC = to_small_vector(getNumActualElements().getValues<int64_t>());
        return compressType(inputType, numElemsPerOC);
    }
    return inputType;
}

bool vpux::Const::SparsifyAttr::inferOutputSplat(bool, vpux::NDTypeInterface) {
    return false;
}

namespace {

template <typename StorageType>
Const::Content sparsify(const Const::Content& content, int64_t sparsifyValue, NDTypeInterface inputType,
                        NDTypeInterface outputType) {
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(),
                                                  Const::SparsifyAttr::inferOutputSplat(content.isSplat(), inputType));
    output.fillWithZero();
    auto outBuf = output.getRawTempBuf();
    auto outBlobPtr = reinterpret_cast<StorageType*>(outBuf.data());

    auto inputValues = content.getValues<StorageType>();

    auto inputShape = inputType.getShape();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected 4D input shape. Got {0}", inputShape);

    const auto OC = inputShape[Dims4D::Filter::OC];
    const auto IC = inputShape[Dims4D::Filter::IC];
    const auto KY = inputShape[Dims4D::Filter::KY];
    const auto KX = inputShape[Dims4D::Filter::KX];
    const auto workloadSize = IC * KY * KX;

    const auto castedSparsifyValue = checked_cast<StorageType>(sparsifyValue);
    constexpr auto byteSize = sizeof(castedSparsifyValue);

    int64_t outputIndex = 0;
    for (int64_t oc = 0; oc < OC; ++oc) {
        auto begin = oc * workloadSize;
        auto end = (oc + 1) * workloadSize;
        for (auto inputIndex = begin; inputIndex < end; ++inputIndex) {
            const auto inputValue = inputValues[inputIndex];
            if (inputValue == castedSparsifyValue) {
                continue;
            }
            outBlobPtr[outputIndex++] = inputValue;
        }
        const auto outputIndexByte = outputIndex * byteSize;
        if (outputIndexByte % VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT != 0) {
            const auto alignedOutputIndexByte =
                    alignValUp<int64_t>(outputIndexByte, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
            outputIndex = alignedOutputIndexByte / byteSize;
        }
    }
    return output;
}

}  // namespace

//
// SparsifyAttr::transform
//

Const::Content Const::SparsifyAttr::transform(Const::Content& input) const {
    auto inputType = input.getType();
    const auto numElemsPerOC = (getNumActualElements() != nullptr)
                                       ? to_small_vector(getNumActualElements().getValues<int64_t>())
                                       : vpux::countNonSparseElementsPerOC(input, inputType.getElementType());
    // Note: sparsify transformation is special, that is, the internal data
    // shape is NOT what SparsifyAttr::inferOutputType() returns. the real
    // content is always "compressed", but for the sake of the users of the
    // constant (e.g. tiling) we *sometimes* pretend that the output type is not
    // compressed. for instance,
    // * sparsify(false) -> ContentAttr's type is NOT compressed
    // * sparsify(true, ...) -> ContentAttr's type is compressed
    const auto outputType = compressType(inputType, numElemsPerOC);

    auto inputElementType = inputType.getElementType();
    int64_t sparsifyValue = getSparsifyValue(inputElementType);

    if (inputElementType.isSignedInteger(8)) {
        return sparsify<int8_t>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isUnsignedInteger(8)) {
        return sparsify<uint8_t>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isF16()) {
        return sparsify<vpux::type::float16>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isBF16()) {
        return sparsify<vpux::type::bfloat16>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isF32()) {
        return sparsify<float>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isFloat8E5M2()) {
        return sparsify<vpux::type::float8_e5m2>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isFloat8E4M3FN()) {
        return sparsify<vpux::type::float8_e4m3>(input, sparsifyValue, inputType, outputType);
    }
    VPUX_THROW("Unexpected weights data type: {0}", inputElementType);
}

//
// SparsifyAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::SparsifyAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::PREFERRED_LAST;
}

//
// SparsifyAttr::getStableHashValue
//

llvm::hash_code vpux::Const::SparsifyAttr::getStableHashValue() const {
    const bool returnCompressed = getCompressOutputType().getValue();
    const auto numElems = getNumActualElements().getValues<int64_t>();
    return llvm::hash_combine(getMnemonic(), returnCompressed,
                              llvm::hash_combine_range(numElems.begin(), numElems.end()));
}
