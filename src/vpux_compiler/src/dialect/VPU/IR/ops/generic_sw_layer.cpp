//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/dialect_processors.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <iterator>

using namespace vpux;

// E#152917 Analyze & settle on GenericSwLayerOp integration & vpux interface usage

namespace {

// Keywords used by the custom assembly format of GenericSwLayerOp.
constexpr llvm::StringLiteral kTilingKeyword = "tiling";
constexpr llvm::StringLiteral kSizesKeyword = "sizes";
constexpr llvm::StringLiteral kOffsetsKeyword = "offsets";

mlir::ArrayAttr buildReturnedTypesAttr(mlir::Builder& builder, mlir::TypeRange resultTypes) {
    SmallVector<mlir::Attribute> returnTypeAttrs;
    returnTypeAttrs.reserve(resultTypes.size());
    for (auto type : resultTypes) {
        returnTypeAttrs.push_back(mlir::TypeAttr::get(type));
    }
    return mlir::ArrayAttr::get(builder.getContext(), returnTypeAttrs);
}

}  // namespace

//
// Helpers
//

/// Returns the callee function of a GenericSwLayerOp, or failure if not found.
static mlir::FailureOr<mlir::func::FuncOp> getCalleeFunc(VPU::GenericSwLayerOp op) {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
        return mlir::failure();
    }
    auto calleeFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
    if (!calleeFunc) {
        return mlir::failure();
    }
    return calleeFunc;
}

/// Returns the KernelInfoAttr attached to the callee function of a GenericSwLayerOp,
/// or std::nullopt when the callee or its KernelInfo attribute cannot be found.
static std::optional<VPU::KernelInfoAttr> getCalleeKernelInfo(VPU::GenericSwLayerOp op) {
    auto calleeFunc = getCalleeFunc(op);
    if (mlir::failed(calleeFunc)) {
        return std::nullopt;
    }
    auto kernelInfo = (*calleeFunc)->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
    if (!kernelInfo) {
        return std::nullopt;
    }
    return kernelInfo;
}

/// Verifies the signature of a tilingInfoFunc against the GenericSwLayerOp inputs:
///   arguments: scalar (non-tensor) inputs + 2*numTilingAxes index arguments
///   results:   sum of 2*rank(input_i) for tensor inputs i < numSlicedInputs, plus
///              sum of 2*rank(scratch_j) for each auxiliary buffer
static mlir::LogicalResult verifyTilingInfoFuncSignature(mlir::func::FuncOp infoFunc, mlir::ValueRange inputs,
                                                         mlir::ValueRange scratchInputs, int64_t numTilingAxes,
                                                         int64_t numSlicedInputs) {
    // Build expected argument types.
    auto indexType = mlir::IndexType::get(infoFunc.getContext());
    SmallVector<mlir::Type> expectedArgTypes;
    for (auto input : inputs) {
        if (!mlir::isa<vpux::NDTypeInterface>(input.getType())) {
            expectedArgTypes.push_back(input.getType());
        }
    }
    expectedArgTypes.append(2 * numTilingAxes, indexType);

    if (infoFunc.getNumArguments() != expectedArgTypes.size()) {
        return infoFunc.emitError() << "KernelInfo.tilingInfoFunc '" << infoFunc.getName() << "' has "
                                    << infoFunc.getNumArguments() << " arguments, expected " << expectedArgTypes.size()
                                    << " (" << (expectedArgTypes.size() - 2 * numTilingAxes) << " scalar inputs + 2 * "
                                    << numTilingAxes << " tiling axes)";
    }

    for (const auto& [argIdx, expected] : llvm::enumerate(expectedArgTypes)) {
        auto actual = infoFunc.getFunctionType().getInput(argIdx);
        if (actual != expected) {
            return infoFunc.emitError() << "KernelInfo.tilingInfoFunc '" << infoFunc.getName() << "' argument "
                                        << argIdx << " has type " << actual << ", expected " << expected;
        }
    }

    int64_t expectedNumResults = 0;
    for (auto input : inputs.take_front(numSlicedInputs)) {
        auto tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(input.getType());
        if (!tensorType) {
            return infoFunc.emitError() << "KernelInfo.numSlicedInputs references input of non-ranked-tensor type "
                                        << input.getType();
        }
        expectedNumResults += 2 * tensorType.getRank();
    }

    for (auto scratch : scratchInputs) {
        auto tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(scratch.getType());
        if (!tensorType) {
            return infoFunc.emitError() << "GenericSwLayerOp scratch input of non-ranked-tensor type "
                                        << scratch.getType();
        }
        expectedNumResults += 2 * tensorType.getRank();
    }

    if (infoFunc.getFunctionType().getNumResults() != static_cast<size_t>(expectedNumResults)) {
        return infoFunc.emitError() << "KernelInfo.tilingInfoFunc '" << infoFunc.getName() << "' has "
                                    << infoFunc.getFunctionType().getNumResults() << " results, expected "
                                    << expectedNumResults
                                    << " (sum of 2 * rank for each tensor input i < numSlicedInputs and each "
                                       "scratch buffer)";
    }

    for (const auto& [resIdx, resType] : llvm::enumerate(infoFunc.getFunctionType().getResults())) {
        if (resType != indexType) {
            return infoFunc.emitError() << "KernelInfo.tilingInfoFunc '" << infoFunc.getName() << "' result " << resIdx
                                        << " has type " << resType << ", expected index";
        }
    }

    return mlir::success();
}

//
// SymbolUserOpInterface
//

mlir::LogicalResult VPU::GenericSwLayerOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    if (!getTilingProperties()) {
        return mlir::success();
    }

    auto calleeFunc = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, getCalleeAttr());
    if (!calleeFunc) {
        // Symbol resolution failure is reported separately by the symbol verifier.
        return mlir::success();
    }

    const auto kernelInfo = calleeFunc->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
    if (!kernelInfo) {
        return emitOpError("requires '" + VPU::KernelInfoAttr::kFuncAttrName +
                           "' attribute on callee function when 'tilingProperties' is present");
    }

    auto infoFunc = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(calleeFunc, kernelInfo.getTilingInfoFunc());
    if (!infoFunc) {
        return emitOpError("KernelInfo.tilingInfoFunc must reference an existing func.func symbol");
    }

    const int64_t numAxes = kernelInfo.getTilingAxes().size();

    if (!kernelInfo.getNumSlicedInputs().getType().isSignlessInteger(64)) {
        return emitOpError("KernelInfo.numSlicedInputs must be a signless i64 integer attribute");
    }

    const auto tiling = getTilingProperties();
    const auto staticOffsets = tiling->getStaticOffsets();
    const auto staticSizes = tiling->getStaticSizes();

    if (staticOffsets.size() != numAxes) {
        return emitOpError("expected static_offsets size (")
               << staticOffsets.size() << ") to equal the number of tiling axes (" << numAxes << ")";
    }
    if (staticSizes.size() != numAxes) {
        return emitOpError("expected static_sizes size (")
               << staticSizes.size() << ") to equal the number of tiling axes (" << numAxes << ")";
    }

    if (getNumResults() == 0) {
        return emitOpError("expected at least one result when tilingProperties is present");
    }

    const auto firstResultType = mlir::dyn_cast<mlir::ShapedType>(getResult(0).getType());
    if (!firstResultType) {
        return emitOpError("expected first result to be a shaped type when tilingProperties is present");
    }
    const int64_t rank = firstResultType.getRank();
    const auto axes = kernelInfo.getTilingAxes();
    for (int64_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < 0 || axes[i] >= rank) {
            return emitOpError("tiling axis ")
                   << axes[i] << " at index " << i << " must be a non-negative integer less than the result rank ("
                   << rank << ")";
        }
        if (i > 0 && axes[i] <= axes[i - 1]) {
            return emitOpError("tiling axes must be sorted in strictly ascending order, but axis ")
                   << axes[i] << " at index " << i << " is not greater than " << axes[i - 1];
        }
    }

    const int64_t numSlicedInputs = kernelInfo.getNumSlicedInputs().getInt();
    if (numSlicedInputs < 0) {
        return emitOpError("KernelInfo.numSlicedInputs must be a non-negative integer, got ") << numSlicedInputs;
    }
    if (numSlicedInputs > static_cast<int64_t>(getInputs().size())) {
        return emitOpError("KernelInfo.numSlicedInputs (")
               << numSlicedInputs << ") exceeds the number of inputs (" << getInputs().size() << ")";
    }

    for (int64_t i = 0; i < numSlicedInputs; ++i) {
        if (!mlir::isa<vpux::NDTypeInterface>(getInputs()[i].getType())) {
            return emitOpError("expected input ")
                   << i << " to be a ranked tensor type, but got " << getInputs()[i].getType();
        }
    }

    return verifyTilingInfoFuncSignature(infoFunc, getInputs(), getScratchInputs(), numAxes, numSlicedInputs);
}

//
// CallOpInterface
//

mlir::CallInterfaceCallable VPU::GenericSwLayerOp::getCallableForCallee() {
    return getOperation()->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName());
}

void VPU::GenericSwLayerOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
    setCalleeAttr(mlir::cast<mlir::SymbolRefAttr>(callee));
}

mlir::Operation::operand_range VPU::GenericSwLayerOp::getArgOperands() {
    return {operand_begin(), operand_end()};
}

mlir::MutableOperandRange VPU::GenericSwLayerOp::getArgOperandsMutable() {
    return mlir::MutableOperandRange(this->getOperation());
}

// -----------------------------------------------------------------------------
// CallOpInterface attribute hooks
// -----------------------------------------------------------------------------
// CallOpInterface requires operations to provide accessors for
// per-argument and per-result call-site attributes. We do not
// model call-site metadata, so these operations do not store such attributes.
// We return nullptr for the getters and treat setters as unsupported.
//
// These methods are required only to satisfy the CallOpInterface contract for
// tooling and generic passes. Actual call-site attributes are not used or
// expected
// -----------------------------------------------------------------------------
mlir::ArrayAttr vpux::VPU::GenericSwLayerOp::getArgAttrsAttr() {
    // no call-site arg attrs supported
    return nullptr;
}

void vpux::VPU::GenericSwLayerOp::setArgAttrsAttr(mlir::ArrayAttr) {
    VPUX_THROW("Call-site argument attributes are not supported for this op");
}

mlir::Attribute vpux::VPU::GenericSwLayerOp::removeArgAttrsAttr() {
    // no call-site arg attrs supported
    return nullptr;
}

mlir::ArrayAttr vpux::VPU::GenericSwLayerOp::getResAttrsAttr() {
    // no call-site result attrs supported
    return nullptr;
}

void vpux::VPU::GenericSwLayerOp::setResAttrsAttr(mlir::ArrayAttr) {
    VPUX_THROW("Call-site result attributes are not supported for this op");
}

mlir::Attribute vpux::VPU::GenericSwLayerOp::removeResAttrsAttr() {
    // no call-site result attrs supported
    return nullptr;
}

//
// print
//

void vpux::VPU::GenericSwLayerOp::print(mlir::OpAsmPrinter& p) {
    p << "(";
    llvm::interleaveComma(getInputs(), p, [&](mlir::Value input) {
        p.printOperand(input);
    });
    if (!getInputs().empty()) {
        p << " : ";
        llvm::interleaveComma(getInputs().getTypes(), p);
    }
    p << ")";

    // Print optional scratch section
    if (!getScratchInputs().empty()) {
        p << " scratch(";
        llvm::interleaveComma(getScratchInputs(), p, [&](mlir::Value scratch) {
            p.printOperand(scratch);
        });
        p << " : ";
        llvm::interleaveComma(getScratchInputs().getTypes(), p);
        p << ")";
    }

    // Print callee
    p << ' ';
    p.printAttributeWithoutType(getCalleeAttr());

    // Print tiling section with mixed static/dynamic sizes and offsets
    if (auto tiling = getTilingProperties()) {
        p << ' ' << kTilingKeyword << "(" << kSizesKeyword << " = ";
        printDynamicIndexList(p, *this, getDynamicSizes(), tiling->getStaticSizes());
        p << ", " << kOffsetsKeyword << " = ";
        printDynamicIndexList(p, *this, getDynamicOffsets(), tiling->getStaticOffsets());
        p << ")";
    }

    // Print remaining attributes, eliding those handled explicitly
    p.printOptionalAttrDict((*this)->getAttrs(),
                            {getCalleeAttrName().strref(), getTilingPropertiesAttrName().strref(),
                             getOperandSegmentSizesAttrName().strref(), getReturnTypesAttrName().strref()});

    // Print result types
    p << " -> ";
    llvm::interleaveComma(getResultTypes(), p, [&](mlir::Type type) {
        p.printType(type);
    });
}

//
// parse
//

mlir::ParseResult vpux::VPU::GenericSwLayerOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    auto& builder = parser.getBuilder();

    // Parse inputs: (operands : types)
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputs;
    llvm::SmallVector<mlir::Type> inputTypes;
    if (parser.parseLParen()) {
        return mlir::failure();
    }
    if (mlir::failed(parser.parseOptionalRParen())) {
        if (parser.parseOperandList(inputs) || parser.parseColonTypeList(inputTypes) || parser.parseRParen()) {
            return mlir::failure();
        }
    }
    if (parser.resolveOperands(inputs, inputTypes, parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }

    // Parse optional scratch section: scratch(<operands> : <types>)
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> scratchInputs;
    llvm::SmallVector<mlir::Type> scratchTypes;
    if (mlir::succeeded(parser.parseOptionalKeyword("scratch"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }
        if (mlir::failed(parser.parseOptionalRParen())) {
            if (parser.parseOperandList(scratchInputs) || parser.parseColonTypeList(scratchTypes) ||
                parser.parseRParen()) {
                return mlir::failure();
            }
        }
        if (parser.resolveOperands(scratchInputs, scratchTypes, parser.getCurrentLocation(), result.operands)) {
            return mlir::failure();
        }
    }

    // Parse callee
    mlir::SymbolRefAttr calleeAttr;
    if (parser.parseAttribute(calleeAttr)) {
        return mlir::failure();
    }
    result.addAttribute(getCalleeAttrName(result.name), calleeAttr);

    // Parse optional tiling section
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> dynSizes, dynOffsets;
    mlir::DenseI64ArrayAttr staticSizesAttr, staticOffsetsAttr;

    if (mlir::succeeded(parser.parseOptionalKeyword(kTilingKeyword))) {
        // Parse sizes = [...]
        if (parser.parseLParen() || parser.parseKeyword(kSizesKeyword) || parser.parseEqual() ||
            parseDynamicIndexList(parser, dynSizes, staticSizesAttr) || parser.parseComma()) {
            return mlir::failure();
        }

        // Parse offsets = [...]
        if (parser.parseKeyword(kOffsetsKeyword) || parser.parseEqual() ||
            parseDynamicIndexList(parser, dynOffsets, staticOffsetsAttr) || parser.parseRParen()) {
            return mlir::failure();
        }

        auto tilingAttr = VPU::JITTilingAttr::get(builder.getContext(), staticSizesAttr, staticOffsetsAttr);
        result.addAttribute(getTilingPropertiesAttrName(result.name), tilingAttr);

        // Resolve dynamic size/offset operands
        auto indexType = builder.getIndexType();
        if (parser.resolveOperands(dynSizes, indexType, result.operands) ||
            parser.resolveOperands(dynOffsets, indexType, result.operands)) {
            return mlir::failure();
        }
    }

    // Parse optional attribute dictionary
    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    // 'returnTypes' is inferred from result types and must not appear in the attr-dict.
    if (result.attributes.get(getReturnTypesAttrName(result.name))) {
        return parser.emitError(parser.getCurrentLocation(),
                                "'returnTypes' is an inferred attribute and must not be specified explicitly");
    }

    // Parse result types: -> type (, type)*
    llvm::SmallVector<mlir::Type> resultTypes;
    if (parser.parseArrow()) {
        return mlir::failure();
    }
    do {
        mlir::Type type;
        if (parser.parseType(type)) {
            return mlir::failure();
        }
        resultTypes.push_back(type);
    } while (mlir::succeeded(parser.parseOptionalComma()));
    result.addTypes(resultTypes);
    result.addAttribute(getReturnTypesAttrName(result.name), buildReturnedTypesAttr(builder, resultTypes));

    // Set operand segment sizes: [inputs, scratch_inputs, dynamic_sizes, dynamic_offsets]
    result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                        builder.getDenseI32ArrayAttr(
                                {static_cast<int32_t>(inputs.size()), static_cast<int32_t>(scratchInputs.size()),
                                 static_cast<int32_t>(dynSizes.size()), static_cast<int32_t>(dynOffsets.size())}));

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::GenericSwLayerOp::verify() {
    auto returnTypesAttr = getReturnTypes();
    auto resultTypes = getResults().getTypes();

    if (returnTypesAttr.size() != resultTypes.size()) {
        return emitOpError("expected returnTypes attribute size (")
               << returnTypesAttr.size() << ") to match number of results (" << resultTypes.size() << ")";
    }

    // Note: full type compatibility checking is intentionally omitted here because
    // it is already performed as part of InferTypeOpInterface invariant checking after verify() completes.
    for (size_t i = 0; i < resultTypes.size(); ++i) {
        auto encodedType = mlir::dyn_cast<mlir::TypeAttr>(returnTypesAttr[i]);
        if (!encodedType) {
            return emitOpError("expected returnTypes[") << i << "] to be a TypeAttr, got " << returnTypesAttr[i];
        }
    }

    const bool hasDynamicOperands = !getDynamicSizes().empty() || !getDynamicOffsets().empty();
    if (hasDynamicOperands && !getTilingProperties()) {
        return emitOpError("requires 'tilingProperties' attribute when dynamic_sizes or dynamic_offsets are present");
    }

    if (auto tiling = getTilingProperties()) {
        const auto staticOffsets = tiling->getStaticOffsets();
        const auto staticSizes = tiling->getStaticSizes();

        if (staticOffsets.size() != staticSizes.size()) {
            return emitOpError("expected static_offsets and static_sizes to have the same length, but got ")
                   << staticOffsets.size() << " and " << staticSizes.size();
        }

        const auto numDynOffsets = llvm::count(staticOffsets.asArrayRef(), mlir::ShapedType::kDynamic);
        if (numDynOffsets != static_cast<int64_t>(getDynamicOffsets().size())) {
            return emitOpError("expected dynamic_offsets count (")
                   << getDynamicOffsets().size() << ") to equal the number of kDynamic entries in static_offsets ("
                   << numDynOffsets << ")";
        }
        const auto numDynSizes = llvm::count(staticSizes.asArrayRef(), mlir::ShapedType::kDynamic);
        if (numDynSizes != static_cast<int64_t>(getDynamicSizes().size())) {
            return emitOpError("expected dynamic_sizes count (")
                   << getDynamicSizes().size() << ") to equal the number of kDynamic entries in static_sizes ("
                   << numDynSizes << ")";
        }
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::GenericSwLayerOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::GenericSwLayerOpAdaptor swOpAdaptor(operands, attrs, prop);
    if (mlir::failed(swOpAdaptor.verify(loc))) {
        return mlir::failure();
    }
    for (size_t i = 0; i < swOpAdaptor.getReturnTypes().size(); ++i) {
        auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(swOpAdaptor.getReturnTypes()[i]);
        if (!typeAttr) {
            return mlir::emitError(loc) << "expected returnTypes[" << i << "] to be a TypeAttr, got "
                                        << swOpAdaptor.getReturnTypes()[i];
        }
        inferredReturnTypes.push_back(typeAttr.getValue());
    }
    return mlir::success();
}

//
// FoldSizesAndOffsets
//

namespace {

/// Absorb arith.constant index operands from dynamic_sizes/dynamic_offsets into
/// the corresponding static_sizes/static_offsets fields of tilingProperties.
/// Each kDynamic entry in the static array corresponds to the next dynamic operand.
/// When a dynamic operand is a constant, replace the kDynamic sentinel with
/// the constant value and remove the operand.
class FoldSizesAndOffsets final : public mlir::OpRewritePattern<VPU::GenericSwLayerOp> {
public:
    using mlir::OpRewritePattern<VPU::GenericSwLayerOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::GenericSwLayerOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FoldSizesAndOffsets::matchAndRewrite(VPU::GenericSwLayerOp op,
                                                         mlir::PatternRewriter& rewriter) const {
    auto tiling = op.getTilingProperties();
    if (!tiling) {
        return mlir::failure();
    }

    const auto isConstant = [](mlir::Value v) {
        return v.getDefiningOp<mlir::arith::ConstantIndexOp>() != nullptr;
    };
    if (llvm::none_of(op.getDynamicSizes(), isConstant) && llvm::none_of(op.getDynamicOffsets(), isConstant)) {
        return mlir::failure();
    }

    auto* ctx = rewriter.getContext();
    auto mixedSizes = mlir::getMixedValues(tiling->getStaticSizes(), op.getDynamicSizes(), ctx);
    auto mixedOffsets = mlir::getMixedValues(tiling->getStaticOffsets(), op.getDynamicOffsets(), ctx);

    const bool sizesChanged = mlir::succeeded(mlir::foldDynamicIndexList(mixedSizes));
    const bool offsetsChanged = mlir::succeeded(mlir::foldDynamicIndexList(mixedOffsets));

    if (!sizesChanged && !offsetsChanged) {
        return mlir::failure();
    }

    SmallVector<int64_t> staticSizes, staticOffsets;
    SmallVector<mlir::Value> dynamicSizes, dynamicOffsets;
    std::tie(staticSizes, dynamicSizes) = mlir::decomposeMixedValues(mixedSizes);
    std::tie(staticOffsets, dynamicOffsets) = mlir::decomposeMixedValues(mixedOffsets);

    auto newTiling = VPU::JITTilingAttr::get(ctx, mlir::DenseI64ArrayAttr::get(ctx, staticSizes),
                                             mlir::DenseI64ArrayAttr::get(ctx, staticOffsets));

    rewriter.modifyOpInPlace(op, [&]() {
        op.setTilingPropertiesAttr(newTiling);
        op.getDynamicSizesMutable().assign(dynamicSizes);
        op.getDynamicOffsetsMutable().assign(dynamicOffsets);
    });

    return mlir::success();
}

}  // namespace

//
// AuxiliaryBufferOpInterface
//

SmallVector<mlir::OpOperand*> VPU::GenericSwLayerOp::getAuxiliaryBuffers() {
    SmallVector<mlir::OpOperand*> result;
    for (auto& scratch : getScratchInputsMutable()) {
        result.push_back(&scratch);
    }
    return result;
}

//
// Builder implementation
//

// Builder automatically creates returnTypes from resultTypes
void VPU::GenericSwLayerOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::TypeRange resultTypes,
                                  mlir::SymbolRefAttr callee, mlir::ValueRange inputs, mlir::Attribute tilingProperties,
                                  mlir::ValueRange dynamic_sizes, mlir::ValueRange dynamic_offsets,
                                  mlir::ValueRange scratch_inputs) {
    state.addTypes(resultTypes);
    state.addAttribute(getCalleeAttrName(state.name), callee);
    state.addOperands(inputs);
    state.addOperands(scratch_inputs);
    if (tilingProperties) {
        state.addAttribute(getTilingPropertiesAttrName(state.name), tilingProperties);
    }
    state.addOperands(dynamic_sizes);
    state.addOperands(dynamic_offsets);
    state.addAttribute(getReturnTypesAttrName(state.name), buildReturnedTypesAttr(builder, resultTypes));
    state.addAttribute(
            getOperandSegmentSizesAttrName(state.name),
            builder.getDenseI32ArrayAttr(
                    {static_cast<int32_t>(inputs.size()), static_cast<int32_t>(scratch_inputs.size()),
                     static_cast<int32_t>(dynamic_sizes.size()), static_cast<int32_t>(dynamic_offsets.size())}));
}

//
// SWOpInterface
//

bool vpux::VPU::GenericSwLayerOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::GenericSwLayerOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GenericSwLayerOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

static TilingInfo getGlobalTileInfo(VPU::GenericSwLayerOp op, SmallVector<int64_t>& iterationSpaceSizes,
                                    SmallVector<int64_t>& iterationSpaceOffsets) {
    if (!op.getTilingProperties()) {
        VPUX_THROW("GenericSwLayerOp requires tilingProperties to adjust attributes");
    }

    // We may be performing a tiling transformation on an already tiled op, so we need to extract the original tile info
    // from the tiling info func. We can use that to adjust our offsets.
    // If tilingProperties are present we are guaranteed by the verifier to have the KernelInfo attribute
    // on the callee and a valid reference to the tiling info function.
    auto calleeFunc = getCalleeFunc(op);
    assert(mlir::succeeded(calleeFunc) && "GenericSwLayerOp: cannot find callee function");
    const auto kernelInfo = (*calleeFunc)->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
    assert(kernelInfo && "GenericSwLayerOp requires KernelInfo on callee to adjust attributes");
    auto func = (*calleeFunc)
                        ->getParentOfType<mlir::ModuleOp>()
                        .lookupSymbol<mlir::func::FuncOp>(kernelInfo.getTilingInfoFunc());
    assert(func && "GenericSwLayerOp: cannot find tilingInfoFunc referenced in KernelInfo");
    VPUX_THROW_UNLESS(!func.getBody().empty(),
                      "GenericSwLayerOp: tilingInfoFunc '{0}' has no body; cannot evaluate tiling",
                      kernelInfo.getTilingInfoFunc());
    auto terminator = mlir::dyn_cast<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(terminator, "GenericSwLayerOp: tilingInfoFunc '{0}' entry block does not end with func.return",
                      kernelInfo.getTilingInfoFunc());
    llvm::DenseMap<mlir::Value, int64_t> valueMap;

    auto tilingAxes = kernelInfo.getTilingAxes();
    uint64_t numSlicedInputs = kernelInfo.getNumSlicedInputs().getInt();
    auto numFuncArgs = func.getNumArguments();

    // Verification guarantees that tilingInfoFunc is well-formed, and we make use of this to guarantee
    // that the following indexing/argument mapping is safe.
    for (const auto& idx : llvm::enumerate(tilingAxes.asArrayRef())) {
        // The last 2*size(tilingAxes) arguments are sizes followed by offsets into the iteration space.
        valueMap[func.getArgument(numFuncArgs - 2 * tilingAxes.size() + idx.index())] =
                iterationSpaceSizes[idx.index()];
        valueMap[func.getArgument(numFuncArgs - tilingAxes.size() + idx.index())] = iterationSpaceOffsets[idx.index()];
    }

    auto registry = vpux::VPU::DialectProcessorRegistry::createDefault();
    func->walk([&](mlir::Operation* op) {
        if (mlir::isa<mlir::func::ReturnOp, mlir::func::FuncOp>(op)) {
            return;
        }
        auto processor = registry->getProcessor(op);
        VPUX_THROW_UNLESS(processor != nullptr, "No dialect processor registered for operation '{0}'", op->getName());
        if (!processor->processOperation(op, valueMap)) {
            VPUX_THROW("Dialect processor '{0}' failed to process operation '{1}'", processor->getDialectName(),
                       op->getName());
        }
    });

    // Collect auxiliary (scratch) operand pointers
    const auto auxBuffers = op.getAuxiliaryBuffers();

    SmallVector<TileInfo> inputTiles;
    size_t infoIdx = 0;
    for (auto operand : op->getOperands() | indexed) {
        if (!mlir::isa<mlir::RankedTensorType>(operand.value().getType())) {
            continue;
        }
        auto rank = mlir::cast<mlir::RankedTensorType>(operand.value().getType()).getRank();
        // Broadcast inputs (between numSlicedInputs and the end of regular inputs) use the
        // full tensor shape.
        const bool isScratchInput = llvm::is_contained(auxBuffers, &op->getOpOperand(operand.index()));
        if (!isScratchInput && operand.index() >= numSlicedInputs) {
            // Broadcasted operands are not tiled: keep full logical shape with zero offsets.
            auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.value().getType());
            inputTiles.emplace_back(TileInfo(ShapeRef(operandType.getShape().raw())));
            continue;
        }

        // If this is a tiled op extract the tiling information.
        SmallVector<int64_t, 4> shape;
        SmallVector<int64_t, 4> offsets;
        SmallVector<int64_t, 4> axis;
        const auto getMappedValueOrThrow = [&](mlir::Value key, llvm::StringRef kind, int64_t dimIdx) -> int64_t {
            const auto it = valueMap.find(key);
            VPUX_THROW_UNLESS(it != valueMap.end(),
                              "Missing {0} value for GenericSwLayerOp tiling at operand {1}, dim {2}. "
                              "The dialect processor did not produce a required mapping",
                              kind, operand.index(), dimIdx);
            return it->second;
        };
        for (int64_t i = 0; i < rank; ++i) {
            shape.push_back(getMappedValueOrThrow(terminator.getOperand(infoIdx + i), "shape", i));
            offsets.push_back(getMappedValueOrThrow(terminator.getOperand(infoIdx + rank + i), "offset", i));
            // Note that all axes are hardcoded to 1.
            //
            // The axis concept implies a regular, non-overlapping split count, but
            // GenericSwLayerOp's tiling can produce arbitrary, overlapping input regions
            // (e.g. 5 tiles of a 2-D input arranged as 4 rectangles in a grid plus a 5th that
            // overlaps all of them - unless we can prove otherwise). There is no well-defined
            // "number of splits" for such cases, so the axis concept does not apply cleanly
            // to GenericSwLayerOp input tiles.
            //
            // The hard-coded value is safe at the moment because VF is disabled for GenericSwLayerOp.
            // We can work around this using analysis/restricting VF dimensions (same as for multiclustering).
            // E#226133: reliably signal unknown input tiling axes.
            axis.push_back(1);
        }
        const auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.value().getType());
        const auto logicalShape = operandType.getDimsOrder().toLogicalOrder(MemShape(shape)).raw();
        const auto logicalOffsets = operandType.getDimsOrder().toLogicalOrder(MemShape(offsets)).raw();
        inputTiles.emplace_back(TileInfo(ShapeRef(logicalShape), ShapeRef(logicalOffsets), ShapeRef(axis)));
        infoIdx += 2 * rank;
    }

    return TilingInfo(inputTiles);
}

InputTiling vpux::VPU::GenericSwLayerOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto tiling = getTilingProperties();
    if (!tiling) {
        VPUX_THROW("GenericSwLayerOp requires tilingProperties to back infer tile info");
    }

    // We should have rejected ops with dynamic scalar inputs from tiling in the first place.
    // Note that this happens only in the case where the original untiled op was dynamic.
    for (auto operand : getOperands()) {
        VPUX_THROW_UNLESS(mlir::isa<vpux::NDTypeInterface>(operand.getType()),
                          "getGlobalTileInfo called on a dynamic GenericSwLayerOp: "
                          "operand has non-tensor type '{0}'",
                          operand.getType());
    }

    const auto resultNdType = mlir::cast<vpux::NDTypeInterface>(getResult(0).getType());
    const auto dimsOrder = resultNdType.getDimsOrder();

    const auto kernelInfo = getCalleeKernelInfo(*this);
    VPUX_THROW_UNLESS(kernelInfo, "GenericSwLayerOp requires KernelInfo on callee to back-infer tile info");
    const auto tilingAxes = kernelInfo->getTilingAxes();

    SmallVector<int64_t> opIterationSpaceSizes;
    SmallVector<int64_t> opIterationSpaceOffsets;
    for (const auto& axis : llvm::enumerate(tilingAxes.asArrayRef())) {
        opIterationSpaceSizes.push_back(tiling->getStaticSizes()[axis.index()]);
        opIterationSpaceOffsets.push_back(tiling->getStaticOffsets()[axis.index()]);
    }

    // Evaluating the tilingInfoFunc (via getGlobalTileInfo) produces global offsets (offsets into the untiled
    // inputs), and we need to convert these into offsets of the current (possibly already tiled) op inputs.
    // The block below does this per input, per dimension: find the (global) offset of the current input and subtract
    // it from the (global) offset of our tile.
    TilingInfo opTileInfo = getGlobalTileInfo(*this, opIterationSpaceSizes, opIterationSpaceOffsets);

    // getGlobalTileInfo works in memory layout order; convert logical output tile data to memory order first.
    SmallVector<int64_t> sliceIterationSpaceSizes;
    SmallVector<int64_t> sliceIterationSpaceOffsets;

    for (const auto& axis : llvm::enumerate(tilingAxes.asArrayRef())) {
        const auto logicalDimIdx = dimsOrder.toDim(MemDim(axis.value())).ind();
        sliceIterationSpaceSizes.push_back(outputTile.shape.raw()[logicalDimIdx]);
        // Convert offsets to the global iteration space.
        sliceIterationSpaceOffsets.push_back(outputTile.offsets.raw()[logicalDimIdx] +
                                             tiling->getStaticOffsets()[axis.index()]);
    }

    TilingInfo sliceTileInfo = getGlobalTileInfo(*this, sliceIterationSpaceSizes, sliceIterationSpaceOffsets);
    VPUX_THROW_UNLESS(sliceTileInfo.tiles.size() == opTileInfo.tiles.size(),
                      "Unexpected tile count: expected {0}, but got {1}", opTileInfo.tiles.size(),
                      sliceTileInfo.tiles.size());

    for (auto tileIdx : irange(sliceTileInfo.tiles.size())) {
        VPUX_THROW_UNLESS(sliceTileInfo.tiles[tileIdx].offsets.size() == opTileInfo.tiles[tileIdx].offsets.size(),
                          "Unexpected rank for tile {0}: expected {1}, but got {2}", tileIdx,
                          opTileInfo.tiles[tileIdx].offsets.size(), sliceTileInfo.tiles[tileIdx].offsets.size());
        for (auto dimIdx : irange(sliceTileInfo.tiles[tileIdx].offsets.size())) {
            sliceTileInfo.tiles[tileIdx].offsets[Dim(dimIdx)] -= opTileInfo.tiles[tileIdx].offsets[Dim(dimIdx)];
        }
    }

    return sliceTileInfo;
}

void vpux::VPU::GenericSwLayerOp::adjustAttrs(const TilingInfo&, const TileInfo& outputTile) {
    auto tiling = getTilingProperties();
    if (!tiling) {
        VPUX_THROW("GenericSwLayerOp requires tilingProperties to adjust attributes");
    }

    SmallVector<int64_t> newOffsets;
    SmallVector<int64_t> newSizes;
    const auto kernelInfo = getCalleeKernelInfo(*this);
    VPUX_THROW_UNLESS(kernelInfo, "GenericSwLayerOp requires KernelInfo on callee to adjust attributes");
    const auto tilingAxes = kernelInfo->getTilingAxes();
    const auto staticOffsets = tiling->getStaticOffsets();

    // tilingAxes are stored in memory layout order; convert each to the corresponding logical
    // dimension index before indexing into outputTile, which uses logical (NCHW) order.
    const auto resultNdType = mlir::cast<vpux::NDTypeInterface>(getResult(0).getType());
    const auto dimsOrder = resultNdType.getDimsOrder();

    for (const auto& axis : llvm::enumerate(tilingAxes.asArrayRef())) {
        const auto logicalDimIdx = dimsOrder.toDim(MemDim(axis.value())).ind();
        // Maintain the offset as an offset into the global iteration space by adding the output tile offset to the
        // original offset.
        newOffsets.push_back(outputTile.offsets.raw()[logicalDimIdx] + staticOffsets[axis.index()]);
        newSizes.push_back(outputTile.shape.raw()[logicalDimIdx]);
    }

    // Create updated tiling attribute with new offsets and sizes
    auto newTiling = VPU::JITTilingAttr::get(getContext(), mlir::DenseI64ArrayAttr::get(getContext(), newSizes),
                                             mlir::DenseI64ArrayAttr::get(getContext(), newOffsets));
    setTilingPropertiesAttr(newTiling);

    // Update returnTypeAttrs
    // E#219995: support multiple outputs, for now we only handle one result
    VPUX_THROW_UNLESS(getNumResults() == 1, "GenericSwLayerOp::adjustAttrs: expected exactly one result, got {0}",
                      getNumResults());
    auto retTy = mlir::cast<vpux::NDTypeInterface>(this->getResult(0).getType());
    retTy = retTy.changeShape(outputTile.shape);
    SmallVector<mlir::Attribute> retTypeAttrs(1, mlir::TypeAttr::get(retTy));
    this->setReturnTypesAttr(mlir::ArrayAttr::get(getContext(), retTypeAttrs));
}

mlir::FailureOr<OutputTiling> vpux::VPU::GenericSwLayerOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    if (!getTilingProperties()) {
        return mlir::failure();
    }

    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// getCanonicalizationPatterns
//

void vpux::VPU::GenericSwLayerOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* ctx) {
    patterns.add<FoldSizesAndOffsets>(ctx);
}
