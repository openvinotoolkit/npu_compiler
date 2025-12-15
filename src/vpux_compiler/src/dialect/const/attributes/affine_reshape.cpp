//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

mlir::LogicalResult vpux::Const::AffineReshapeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                           mlir::ArrayAttr dimMapping, mlir::ArrayAttr shapeValue) {
    // Just check if the types of dimMapping and shapeValue make sense. Actual validity of the values is then
    // ensured in inferOutputType() because that depends on the input type.

    auto isIntegerAttr = [](auto attribute) {
        return mlir::isa_and_nonnull<mlir::IntegerAttr>(attribute);
    };

    auto isIntegerArrayAttr = [&](auto attribute) {
        auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attribute);
        return arrayAttr != nullptr && llvm::all_of(arrayAttr, isIntegerAttr);
    };

    auto isIntegerArrayArrayAttr = [&](auto attribute) {
        auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attribute);
        return arrayAttr != nullptr && llvm::all_of(arrayAttr, isIntegerArrayAttr);
    };

    if (!isIntegerArrayArrayAttr(dimMapping)) {
        return printTo(emitError(), "'dimMapping' is not a valid non-null array of arrays of integers");
    }

    if (!isIntegerArrayAttr(shapeValue)) {
        return printTo(emitError(), "'shapeValue' is not a valid non-null array of integers");
    }

    return mlir::success();
}

vpux::NDTypeInterface vpux::Const::AffineReshapeAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    const auto inputLayout = inputType.getDimsOrder();
    // Reshaping is always a view-like operation. However, for some dimMappings this actually might make the
    // logical view look like a permutation. To combat this, we want to find a output layout that results in a logical
    // view as if we just apply a reshape (in the numpy sense). If no such layout can be found, then this combination of
    // ordering and shape is not permissible for AffineReshapeAttr.
    const auto outputLayout = inferAffineReshapeOutputLayout(inputLayout.toPermutation(), getDimMapping());
    VPUX_THROW_WHEN(!outputLayout.has_value(),
                    "'AffineReshapeAttr' cannot infer output layout for input layout '{1}' and dim_mapping '{0}'",
                    getDimMapping(), inputLayout);

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(getDimMapping());
    const auto outShape = parseIntArrayAttr<int64_t>(getShapeValue());

    const auto elemTypeInferResult =
            inferElemTypeAffineReshape(inputType.getShape(), inputType.getElementType(), dimMapping, outShape);
    const auto outElemType = elemTypeInferResult.has_value() ? elemTypeInferResult.value() : inputType.getElementType();

    return inputType.changeTypeComponents(TypeComponents()
                                                  .setElementType(outElemType)
                                                  .setShape(ShapeRef(outShape))
                                                  .setDimsOrder(outputLayout.value()));
}

bool vpux::Const::AffineReshapeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

Const::Content vpux::Const::AffineReshapeAttr::transform(vpux::Const::Content& input) const {
    const auto outputType = inferOutputType(input.getType());
    return Const::Content::moveBuffer(outputType, std::move(input));
}

mlir::Attribute vpux::Const::AffineReshapeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    // We are trying to parse '#const.AffineReshape<[[...]], [...]>'.
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr dimMapping;
    if (mlir::failed(parser.parseAttribute(dimMapping))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::AffineReshapeAttr>(dimMapping, shape);
}

void vpux::Const::AffineReshapeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getDimMapping());
    printer << ", ";
    printer.printAttribute(getShapeValue());
    printer << ">";
}

llvm::hash_code vpux::Const::AffineReshapeAttr::getStableHashValue() const {
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(getDimMapping());
    const auto dimMappingRefs = llvm::map_range(dimMapping, [](const auto& array) {
        return ArrayRef<int64_t>(array);
    });
    const auto shapeValue = parseIntArrayAttr<int64_t>(getShapeValue());

    return llvm::hash_combine(getMnemonic(), llvm::hash_combine_range(dimMappingRefs.begin(), dimMappingRefs.end()),
                              llvm::hash_value(ArrayRef<int64_t>(shapeValue)));
}
