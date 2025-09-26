//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

//
// ConstDialect::materializeConstant
//

mlir::Operation* vpux::Const::ConstDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!mlir::isa<mlir::RankedTensorType, mlir::MemRefType>(type)) {
        (void)errorAt(loc, "Can't materialize Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

//
// ConstDialect: bufferize Const::DeclareOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, Const::DeclareOp origOp, Const::DeclareOp::Adaptor&,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-ConstDeclareOp", 0);
    log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    const auto newType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newType, origOp.getContentAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

void vpux::registerConstDeclareBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::Const::ConstDialect*) {
        Const::DeclareOp::attachInterface<VpuGenericOneShotBufferizeModel<Const::DeclareOp>>(*ctx);
    });
}

//
// DeclareOp::fold
//

mlir::OpFoldResult vpux::Const::DeclareOp::fold(FoldAdaptor adaptor) {
    VPUX_THROW_UNLESS(adaptor.getOperands().empty(), "constant has no operands");
    return getContentAttr();
}

//
// DeclareOp::serialize
//

void vpux::Const::DeclareOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto cnt = getContent();
    auto ptr = binDataSection.getCurrentWriteAddr();
    const auto size = getBinarySize();
    MutableArrayRef<char> tempBuf(reinterpret_cast<char*>(ptr), reinterpret_cast<char*>(ptr) + size);
    cnt.copyTo(tempBuf);
    binDataSection.shiftCurrentWriteAddr(size);
}

//
// DeclareOp::getBinarySize
//

size_t vpux::Const::DeclareOp::getBinarySize() {
    vpux::Const::Content cnt = getContent();

    return cnt.getType().getTotalAllocSize().count();
}

//
// DeclareOp::getAlignmentRequirements
//

size_t vpux::Const::DeclareOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}

//
// DeclareOp::getMemorySpace
//

vpux::VPURT::BufferSection vpux::Const::DeclareOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::Constant;
}

//
// DeclareOp::getAccessingProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getAccessingProcs() {
    auto tempFlagsVal = vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE;

    for (auto user : getResult().getUsers()) {
        if (auto binaryIface = mlir::dyn_cast<vpux::ELFNPU37XX::BinaryOpInterface>(user)) {
            tempFlagsVal = tempFlagsVal | binaryIface.getUserProcs();
        }
    }

    return tempFlagsVal;
}

//
// DeclareOp::getUserProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

//
// DeclareOp::verify
//

mlir::LogicalResult vpux::Const::DeclareOp::verify() {
    const auto op = getOperation();
    const auto attrType = getContentAttr().getType();
    const auto opType = mlir::cast<vpux::NDTypeInterface>(getType());

    auto emitError = [&]() {
        return op->emitError();
    };

    // For ContentAttr using dense resource additionaly
    // verify that dense resource. This can't be done as part
    // of ContentAttr::verify during IR parsing since dense resource
    // value won't be accessible at that time.
    auto contentAttr = getContentAttr();
    if (auto denseResource = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(contentAttr.getBaseContent())) {
        if (mlir::failed(ContentAttr::verifyDenseResource(emitError, denseResource, contentAttr.isSplat()))) {
            return mlir::failure();
        }
    }

    // For type with swizzling skip the shape check as the content
    // might have been flattened to accomodate swizzled buffer.
    if (!vpux::getSwizzlingSchemeAttr(opType)) {
        if (opType.getShape() != attrType.getShape()) {
            return errorAt(op, "'Const.Declare' has mismatch in value shape '{0}' and result shape '{1}'",
                           attrType.getShape(), opType.getShape());
        }
    }
    if (opType.getElementType() != attrType.getElementType()) {
        if (!mlir::isa<mlir::quant::QuantizedType>(opType.getElementType()) &&
            !mlir::isa<mlir::IntegerType>(attrType.getElementType())) {
            return errorAt(op, "'Const.Declare' has mismatch in value element type '{0}' and result element type '{1}'",
                           attrType.getElementType(), opType.getElementType());
        }
    }

    const auto attrOrder = attrType.getDimsOrder();
    const auto opOrder = opType.getDimsOrder();

    if (opOrder != attrOrder) {
        return errorAt(op, "'Const.Declare' has mismatch in value DimsOrder '{0}' and result DimsOrder '{1}'",
                       attrOrder, opOrder);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::Const::DeclareOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    // check if content is of type SymElementsAttr, otherwise this doesn't apply
    auto symElementsAttr = mlir::dyn_cast_or_null<vpux::Const::SymElementsAttr>(getContentAttr().getBaseContent());

    if (symElementsAttr == nullptr) {
        return mlir::success();
    }

    auto annotatedType = symElementsAttr.getType();
    auto symName = symElementsAttr.getSymName();

    // SymbolTableCollection::lookupNearestSymbolFrom relies on the nearest symbol table. In this case
    // of const.BundleData, which would fail to lookup the symbol we are actually interested in. That's
    // why we have to go from the parent module op.
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (moduleOp == nullptr) {
        return emitOpError("is expected to be encapsulated by 'Module' op directly or indirectly");
    }

    // lookup and check op type
    auto op = symbolTable.lookupNearestSymbolFrom(moduleOp, symName);

    if (auto rodataOp = mlir::dyn_cast_or_null<vpux::Const::RodataOp>(op)) {
        // lookup the const.Rodataop and check if the annotated type matches the underlying type
        auto realType = rodataOp.getContent().getType();
        if (annotatedType != realType) {
            return emitOpError(formatv("annotated type '{0}' and real type '{1}' of symbol '{2}' do not match",
                                       annotatedType, realType, symName));
        }

        return mlir::success();
    }

    return emitOpError(formatv("symbol '{0}' does not point to a valid 'const.Rodata' op", symName));
}

vpux::Const::ContentAttr vpux::Const::DeclareOp::getContentAttr() const {
    // Note: getProperties() is not 'const' in MLIR...
    return const_cast<Const::DeclareOp&>(*this).getProperties().content;
}

namespace {
template <typename ContentAttrT>
void genericDeclareOpBuild(mlir::OpBuilder&, mlir::OperationState& state, mlir::Type outputType,
                           ContentAttrT&& content) {
    auto& props = state.getOrAddProperties<Const::DeclareOp::Properties>();
    props.content = std::forward<ContentAttrT>(content);
    state.addTypes(ArrayRef{outputType});
}
}  // namespace

void vpux::Const::DeclareOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outputType,
                                   const Const::ContentAttr& content) {
    genericDeclareOpBuild(builder, state, outputType, content);
}

void vpux::Const::DeclareOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outputType,
                                   Const::ContentAttr&& content) {
    genericDeclareOpBuild(builder, state, outputType, std::move(content));
}

//
// DeclareOp::canonicalizer
//

/**
 * Removes the tiled information ('strides' attribute) from memref types of const.Declare operations. For example:
 *   Before:
 *     const.Declare memref<1x96x256x16xf16, {order = #NHWC, strides = [786432, 1, 1536, 96]}> = ...
 *   After:
 *     const.Declare memref<1x96x256x16xf16, #NHWC> = ...
 *
 * This is necessary because the previous behvaiour of this tiled info erasure relied on a "bug" in MLIR which
 * is fixed in LLVM18. This rewrite pattern is designed to mostly work in conjunction with the canonicalization
 * of SubView operations is expected to NOT work in certain situation (see
 * ./tests/lit/NPU/dialect/const/ops/invalid.mlir). At the moment only MemRefType subtypes are supported because
 * they implement the correct layout.
 *
 * See E-120399 for more information.
 *
 */
class EraseTiledInfo final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    EraseTiledInfo(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

private:
    static bool hasStridesAttr(mlir::MemRefType type) {
        auto layout = type.getLayout();
        auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout);
        return descAttr != nullptr && descAttr.strides() != nullptr;
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult EraseTiledInfo::matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const {
    auto type = mlir::cast<vpux::NDTypeInterface>(constOp.getOutput().getType());

    if (mlir::isa<mlir::BaseMemRefType>(type)) {
        VPUX_THROW_WHEN(!mlir::isa<mlir::MemRefType>(type),
                        "Only mlir::MemRefType is supported for constants bufferization");
    } else {
        return mlir::failure();
    }

    // Only memref types can have a 'strides' attribute.
    auto memRefType = mlir::dyn_cast<mlir::MemRefType>(type);

    // If type is already free of 'strides' we ignore it and return failure to
    // avoid infinite loops.
    if (!hasStridesAttr(memRefType)) {
        return mlir::failure();
    }

    // Otherwise, create a new constant op with the new erased type.
    rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, type.eraseTiledInfo(), constOp.getContentAttr());

    return mlir::success();
}

void vpux::Const::DeclareOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<EraseTiledInfo>(ctx);
}

//
// RodataBundleOp::verifySymbolUses
//

mlir::LogicalResult Const::RodataBundleOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    // SymbolTableCollection::lookupNearestSymbolFrom relies on the nearest symbol table. In this case
    // of const.BundleData, which would fail to lookup the symbol we are actually interested in. That's
    // why we have to go from the parent module op.
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (moduleOp == nullptr) {
        return emitOpError("is expected to indirectly be encapsulated by 'Module' op");
    }

    for (auto& rodataSym : getValues()) {
        // Check that the symbol points to a 'const.Rodata' op
        auto rodataOp = symbolTable.lookupNearestSymbolFrom<vpux::Const::RodataOp>(moduleOp, rodataSym);
        if (rodataOp == nullptr) {
            return emitOpError(formatv("symbol '{0}' does not point to a valid 'const.Rodata' op", rodataSym));
        }

        // Check that the following types match:
        // const.Rodata @weights_1 dense<1.0> : tensor<4x4xf32>
        //                                      ^^^^^^^^^^^^^^^
        // const.RodataBundle @bundle = [@Data::@weights_1, ...] : tensor<4x4xf32>
        //                                                         ^^^^^^^^^^^^^^^
        auto underlyingType = rodataOp.getContent().getType();
        if (auto thisType = getBundleType(); thisType != underlyingType) {
            return emitOpError(formatv("'const.Rodata' op type '{0}' pointed to by symbol '{1}' and "
                                       "'const.RodataBundle' op type '{2}' do not match",
                                       underlyingType, rodataSym, thisType));
        }
    }

    return mlir::success();
}

//
// MultiDeclareOp::verifySymbolUses
//

mlir::LogicalResult Const::MultiDeclareOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    // lookup the parent ModuleOp
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (moduleOp == nullptr) {
        return emitOpError("is expected to indirectly be encapsulated by 'Module' op");
    }

    auto multiContentSymbol = getMultiContentSymbol();

    // lookup the RodataBundle op and verify its usage
    auto bundleSymbol = multiContentSymbol.getBundleSymbol();
    auto rodataBundleOp = symbolTable.lookupNearestSymbolFrom<vpux::Const::RodataBundleOp>(moduleOp, bundleSymbol);
    if (rodataBundleOp == nullptr) {
        return emitOpError(formatv("Symbol '{0}' does not point to a valid 'const.RodataBundle' op", bundleSymbol));
    }

    if (rodataBundleOp.getValues().empty()) {
        return emitOpError(formatv("Symbol '{0}' must point to a non-empty 'const.RodataBundle' op", bundleSymbol));
    }

    // Check that the following types match:
    // const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
    //                                                                       ^^^^^^^^^^^^^^^
    // const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32> [#const.Add<2.0>]
    //                                                              ^^^^^^^^^^^^^^^
    auto bundleSymbolType = multiContentSymbol.getBundleSymbolType();
    auto bundleType = rodataBundleOp.getBundleType();
    if (bundleSymbolType != bundleType) {
        return emitOpError(formatv(
                "'MultiContentSymbolAttr' bundle type '{0}' and 'const.RodataBundle' op bundle type '{1}' do not match",
                bundleSymbolType, bundleType));
    }

    // Check that the following types after applying transformations match:
    // const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32> [#const.Add<2.0>]
    //                    ^^^^^^^^^^^^^^^                           ^^^^^^^^^^^^^^^
    auto transformations = multiContentSymbol.getTransformations();
    auto finalType = inferFinalType(bundleSymbolType, transformations);
    if (auto thisType = getType(); finalType != thisType) {
        return emitOpError(formatv("'const.RodataBundle' op final type '{0}' pointed to by symbol '{1}' and "
                                   "'const.MultiDeclare' op type '{2}' do not match",
                                   finalType, bundleSymbol, thisType));
    }

    return mlir::success();
}

//
// MultiDeclareOp::dereferenceMultiContentSymbol
//

Const::MultiContentAttr Const::MultiDeclareOp::dereferenceMultiContentSymbol() {
    // The validity of the following operations has been verified prior by verifySymbolUses()
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto multiContentSymbol = getMultiContentSymbol();
    auto bundleSymbol = multiContentSymbol.getBundleSymbol();
    auto rodataBundleOp =
            mlir::SymbolTable::lookupNearestSymbolFrom<vpux::Const::RodataBundleOp>(moduleOp, bundleSymbol);
    auto symbols = rodataBundleOp.getValues();

    mlir::SmallVector<mlir::ElementsAttr> elementsAttrVec(symbols.size());
    llvm::transform(symbols, elementsAttrVec.begin(), [&](mlir::SymbolRefAttr symbol) {
        auto rodataOp = mlir::SymbolTable::lookupNearestSymbolFrom<vpux::Const::RodataOp>(moduleOp, symbol);
        return rodataOp.getContentAttr();
    });

    auto transformations = multiContentSymbol.getTransformations();
    return MultiContentAttr::get(getContext(), elementsAttrVec, transformations);
}

//
// MultiDeclareOp::fold
//

mlir::OpFoldResult Const::MultiDeclareOp::fold(FoldAdaptor) {
    // lazy folding (or no folding), just as in DeclareOp::fold
    return dereferenceMultiContentSymbol();
}

//
// setupExtraInterfaces
//

void Const::ConstDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        mlir::RankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::UnrankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::MemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
        mlir::UnrankedMemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
    });
}

// Note: for some reason, this cpp-only printer method has to be declared in
// vpux::Const namespace.
namespace vpux::Const {
void printContentAttr(mlir::OpAsmPrinter& printer, const DeclareOp&, const ContentAttr& content) {
    printContentAttr(printer, content);
}
}  // namespace vpux::Const

//
// Generated
//

#include <vpux/compiler/dialect/const/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/ops.cpp.inc>
