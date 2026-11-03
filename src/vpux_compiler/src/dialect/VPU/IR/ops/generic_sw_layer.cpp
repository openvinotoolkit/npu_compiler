//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
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

/// Verifies KernelInfo.inputSegmentationAxes against the operands of a GenericSwLayerOp.
static mlir::LogicalResult verifySegmentationAxes(VPU::GenericSwLayerOp op, VPU::KernelInfoAttr kernelInfo) {
    const auto segmentationAxes = kernelInfo.getInputSegmentationAxes();
    if (segmentationAxes.empty()) {
        // The classification has not been performed yet.
        return mlir::success();
    }

    for (const auto& [entryIdx, entry] : llvm::enumerate(segmentationAxes)) {
        // The element type and the entry size are already guaranteed by KernelInfoAttr::verify.
        const auto segmentationDims = mlir::cast<mlir::DenseI64ArrayAttr>(entry).asArrayRef();

        size_t operandIdx = 0;
        for (auto operand : op->getOperands()) {
            const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType());
            if (ndType == nullptr) {
                continue;
            }
            const auto segmentationDim = segmentationDims[operandIdx];
            if (segmentationDim >= ndType.getRank()) {
                return op.emitOpError("KernelInfo.inputSegmentationAxes entry ")
                       << entryIdx << " has segmentation dim " << segmentationDim << " for operand " << operandIdx
                       << ", which must be less than that operand's rank (" << ndType.getRank() << ")";
            }
            ++operandIdx;
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
    if (!kernelInfo.getNumTensorInputs().getType().isSignlessInteger(64)) {
        return emitOpError("KernelInfo.numTensorInputs must be a signless i64 integer attribute");
    }
    if (!kernelInfo.getNumAuxBuffers().getType().isSignlessInteger(64)) {
        return emitOpError("KernelInfo.numAuxBuffers must be a signless i64 integer attribute");
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
                   << i << " to be a ranked or distributed tensor type, but got " << getInputs()[i].getType();
        }
    }

    const int64_t numTensorInputs = kernelInfo.getNumTensorInputs().getInt();
    const auto numTensorOperands = llvm::count_if(getInputs(), [](mlir::Value input) {
        return mlir::isa<vpux::NDTypeInterface>(input.getType());
    });
    if (numTensorOperands != numTensorInputs) {
        return emitOpError("KernelInfo.numTensorInputs (")
               << numTensorInputs << ") does not match the number of tensor inputs (" << numTensorOperands << ")";
    }

    const int64_t numAuxBuffers = kernelInfo.getNumAuxBuffers().getInt();
    if (numAuxBuffers != static_cast<int64_t>(getScratchInputs().size())) {
        return emitOpError("KernelInfo.numAuxBuffers (")
               << numAuxBuffers << ") does not match the number of scratch inputs (" << getScratchInputs().size()
               << ")";
    }

    if (mlir::failed(verifySegmentationAxes(*this, kernelInfo))) {
        return mlir::failure();
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
// ClusteredOpInterface
//

bool vpux::VPU::GenericSwLayerOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t /*numTiles*/) {
    const auto kernelInfo = getCalleeKernelInfo(*this);
    if (!kernelInfo) {
        return false;
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    const auto firstResultNdType = mlir::dyn_cast<vpux::NDTypeInterface>(getResult(0).getType());
    if (!firstResultNdType || firstResultNdType.getRank() != 4) {
        return false;
    }

    if (kernelInfo->getInputSegmentationAxes().empty()) {
        return false;
    }

    const auto dimsOrder = firstResultNdType.getDimsOrder();
    // Ensure we can produce valid distributed tensors for all tensor-like operands
    // given a segmentation of the output along an axis.
    // Each operand needs to be classified as either segmented or distributed.
    const auto isValidAxis = [&](Dim logicalDim) {
        const int64_t memDimIdx = static_cast<int64_t>(dimsOrder.toMemDim(logicalDim).ind());
        const auto axisIdx = kernelInfo->getTilingAxisIndex(memDimIdx);
        if (!axisIdx.has_value()) {
            return false;
        }
        const auto segmentationMemDims = kernelInfo->getInputSegmentationDims(*axisIdx);
        return !llvm::is_contained(segmentationMemDims, VPU::KernelInfoAttr::kUnknownDim);
    };

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isValidAxis(Dims4D::Act::H);
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isValidAxis(Dims4D::Act::W);
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isValidAxis(Dims4D::Act::C);
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return isValidAxis(Dims4D::Act::N);
    default:
        return false;
    }
}

vpux::VPU::DistributionInfo vpux::VPU::GenericSwLayerOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /*memoryNumTiles*/) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
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
    VPUX_THROW_WHEN(func.getBody().empty(),
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
        VPUX_THROW_WHEN(processor == nullptr, "No dialect processor registered for operation '{0}'", op->getName());
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
        if (!mlir::isa<vpux::NDTypeInterface>(operand.value().getType())) {
            continue;
        }
        auto rank = mlir::cast<vpux::NDTypeInterface>(operand.value().getType()).getRank();
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
        // Use a placeholder for the axis value, fillInputTileAxes will compute the actual values.
        const SmallVector<int64_t, 4> axis(static_cast<size_t>(rank), 1);
        const auto getMappedValueOrThrow = [&](mlir::Value key, llvm::StringRef kind, int64_t dimIdx) -> int64_t {
            const auto it = valueMap.find(key);
            VPUX_THROW_WHEN(it == valueMap.end(),
                            "Missing {0} value for GenericSwLayerOp tiling at operand {1}, dim {2}. "
                            "The dialect processor did not produce a required mapping",
                            kind, operand.index(), dimIdx);
            return it->second;
        };
        for (int64_t i = 0; i < rank; ++i) {
            shape.push_back(getMappedValueOrThrow(terminator.getOperand(infoIdx + i), "shape", i));
            offsets.push_back(getMappedValueOrThrow(terminator.getOperand(infoIdx + rank + i), "offset", i));
        }
        const auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.value().getType());
        const auto logicalShape = operandType.getDimsOrder().toLogicalOrder(MemShape(shape)).raw();
        const auto logicalOffsets = operandType.getDimsOrder().toLogicalOrder(MemShape(offsets)).raw();
        inputTiles.emplace_back(TileInfo(ShapeRef(logicalShape), ShapeRef(logicalOffsets), ShapeRef(axis)));
        infoIdx += 2 * rank;
    }

    return TilingInfo(inputTiles);
}

// Fills in the axes value of input tiles according to the output axes and the
// kernels recorded axes segmentation mapping. When this mapping is not available
// the axes value is left as 1, however users which need to consume the axes
// (multiclustering, vertical fusion) will avoid tiling along these unknown
// axes.
// E#226133: reliably signal unknown input tiling axes.
static void fillInputTileAxes(VPU::GenericSwLayerOp op, VPU::KernelInfoAttr kernelInfo, const TileInfo& outputTile,
                              TilingInfo& inputTiling) {
    if (kernelInfo.getInputSegmentationAxes().empty()) {
        return;
    }

    const auto tilingAxes = kernelInfo.getTilingAxes().asArrayRef();
    const auto outDimsOrder = mlir::cast<vpux::NDTypeInterface>(op.getResult(0).getType()).getDimsOrder();

    // tilingAxes are memory dimensions of the first result, in the same order as inputSegmentationAxes.
    for (const auto& tilingAxis : tilingAxes | indexed) {
        const auto numSplits = outputTile.axis[outDimsOrder.toDim(MemDim(tilingAxis.value()))];
        if (numSplits == 1) {
            continue;
        }

        const auto segmentationMemDims = kernelInfo.getInputSegmentationDims(tilingAxis.index());
        VPUX_THROW_UNLESS(segmentationMemDims.size() == inputTiling.tiles.size(),
                          "GenericSwLayerOp: inputSegmentationAxes entry {0} has {1} inputs, but {2} input tiles "
                          "were inferred",
                          tilingAxis.index(), segmentationMemDims.size(), inputTiling.tiles.size());

        // The tiles follow the order of the tensor operands, which matches the order of the kernel
        // arguments the segmentation axes were computed for: tensor inputs first, then aux buffers.
        size_t tileIdx = 0;
        for (auto operand : op->getOperands()) {
            if (!mlir::isa<vpux::NDTypeInterface>(operand.getType())) {
                continue;
            }
            const auto segmentationMemDim = segmentationMemDims[tileIdx];
            // Ignore broadcast or unclassified cases.
            if (segmentationMemDim != VPU::KernelInfoAttr::kBroadcastDim &&
                segmentationMemDim != VPU::KernelInfoAttr::kUnknownDim) {
                const auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
                inputTiling.tiles[tileIdx].axis[operandType.getDimsOrder().toDim(MemDim(segmentationMemDim))] =
                        numSplits;
            }
            ++tileIdx;
        }
    }
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

    fillInputTileAxes(*this, *kernelInfo, outputTile, sliceTileInfo);

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

//
// VerticalFusionOpInterface
//

// Returns the logical dimensions of the first result the op can be fused on, as a subset of the
// tiling axes.
//
// Two properties are needed to fuse on an axis:
//  - the segmentation of the tensor inputs along it must be known, with the exception
//    of scratch/aux buffers
//  - if an input is segmented then the segmentation must happen on the same logical
//    dimension the result is tiled on due to current limitation in vertical fusion.
static DimArr getFusionAxes(VPU::GenericSwLayerOp op, VPU::KernelInfoAttr kernelInfo) {
    const auto resultNdType = mlir::cast<vpux::NDTypeInterface>(op.getResult(0).getType());
    const auto dimsOrder = resultNdType.getDimsOrder();

    if (kernelInfo.getInputSegmentationAxes().empty()) {
        // The input classification has not been computed, so nothing is known about any axis.
        return {};
    }

    const auto tilingAxes = kernelInfo.getTilingAxes().asArrayRef();
    const auto numTensorInputs = static_cast<size_t>(kernelInfo.getNumTensorInputs().getInt());

    // The segmentation entries are indexed by tensor operand position, so gather those first.
    const auto tensorInputs = to_small_vector(op.getInputs() | filtered([](mlir::Value input) {
                                                  return mlir::isa<vpux::NDTypeInterface>(input.getType());
                                              }));
    VPUX_THROW_UNLESS(tensorInputs.size() == numTensorInputs,
                      "GenericSwLayerOp: has {0} tensor inputs, but KernelInfo.numTensorInputs is {1}",
                      tensorInputs.size(), numTensorInputs);

    DimArr fusionAxes;
    for (const auto& tilingAxis : tilingAxes | indexed) {
        const auto axisDim = dimsOrder.toDim(MemDim(tilingAxis.value()));
        const auto segmentationMemDims = kernelInfo.getInputSegmentationDims(tilingAxis.index());

        const auto isSegmentedOnAxisDim = [&](size_t inputIdx) {
            const int64_t segmentationMemDim = segmentationMemDims[inputIdx];
            if (segmentationMemDim == VPU::KernelInfoAttr::kBroadcastDim) {
                // Broadcast input: not sliced along this axis.
                return true;
            }
            if (segmentationMemDim == VPU::KernelInfoAttr::kUnknownDim) {
                // The split of this input along this axis is unknown, so the axis cannot be handed
                // to the operand's producer.
                return false;
            }
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(tensorInputs[inputIdx].getType());
            return inputType.getDimsOrder().toDim(MemDim(segmentationMemDim)) == axisDim;
        };

        if (llvm::all_of(irange(tensorInputs.size()), isSegmentedOnAxisDim)) {
            fusionAxes.push_back(axisDim);
        }
    }
    return fusionAxes;
}

bool vpux::VPU::GenericSwLayerOp::isVFSupported() {
    const auto kernelInfo = getCalleeKernelInfo(*this);
    return kernelInfo.has_value() && !getFusionAxes(*this, *kernelInfo).empty();
}

DimArr vpux::VPU::GenericSwLayerOp::restrictedFusionAxes() {
    const auto resultNdType = mlir::cast<vpux::NDTypeInterface>(getResult(0).getType());
    const auto rank = static_cast<size_t>(resultNdType.getShape().size());

    const auto kernelInfo = getCalleeKernelInfo(*this);
    VPUX_THROW_UNLESS(kernelInfo, "GenericSwLayerOp: restrictedFusionAxes called but isVFSupported() returned false");
    const auto fusionAxes = getFusionAxes(*this, *kernelInfo);
    DimArr restricted;
    for (size_t i = 0; i < rank; ++i) {
        if (!llvm::is_contained(fusionAxes, Dim(i))) {
            restricted.push_back(Dim(i));
        }
    }
    return restricted;
}

//
// Distribution methods/helpers
//

// Returns the position of `operand` within an inputSegmentationAxes entry, or std::nullopt when the
// operand is neither a tensor input nor an auxiliary buffer. Tensor inputs are identified by
// NDTypeInterface, so that the count also holds once the operands have been made distributed, and
// the op verifier keeps it equal to KernelInfo.numTensorInputs.
static std::optional<int64_t> getSegmentationEntryIndex(VPU::GenericSwLayerOp op, mlir::OpOperand& operand) {
    VPUX_THROW_UNLESS(operand.getOwner() == op.getOperation(), "Operand {0} does not belong to GenericSwLayerOp '{1}'",
                      operand.getOperandNumber(), op.getCallee());

    const auto isTensor = [](const mlir::OpOperand& use) {
        return mlir::isa<vpux::NDTypeInterface>(use.get().getType());
    };

    if (!isTensor(operand)) {
        return std::nullopt;
    }
    return llvm::count_if(op->getOpOperands().take_front(operand.getOperandNumber()), isTensor);
}

// Returns the logical dimension of `operand` that is segmented when tiling `op` under `strategy`,
// KernelInfoAttr::kBroadcastDim when that input is broadcasted, or std::nullopt for a Clustering
// strategy or when the operand is neither a tensor input nor an auxiliary buffer.
static std::optional<int64_t> getGenericSwLayerInputSegmentationDim(VPU::GenericSwLayerOp op,
                                                                    VPU::MultiClusterStrategy strategy,
                                                                    mlir::OpOperand& operand,
                                                                    vpux::NDTypeInterface inputType) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return std::nullopt;
    }

    const auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "GenericSwLayerOp '{0}' has no parent ModuleOp", op.getCallee());
    const auto calleeFunc =
            mlir::dyn_cast_or_null<mlir::func::FuncOp>(mlir::SymbolTable::lookupSymbolIn(moduleOp, op.getCallee()));
    VPUX_THROW_UNLESS(calleeFunc, "Could not find callee function '{0}' for GenericSwLayerOp", op.getCallee());
    const auto kernelInfo = calleeFunc->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
    VPUX_THROW_UNLESS(kernelInfo, "Callee '{0}' has no KernelInfoAttr but a non-Clustering strategy was assigned",
                      op.getCallee());

    const auto entryIdx = getSegmentationEntryIndex(op, operand);
    if (!entryIdx.has_value()) {
        return std::nullopt;
    }

    VPUX_THROW_WHEN(kernelInfo.getInputSegmentationAxes().empty(),
                    "Strategy '{0}' assigned to GenericSwLayerOp '{1}' but its inputs have not been classified",
                    strategy, op.getCallee());

    const auto firstResultNdType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getResult(0).getType());
    VPUX_THROW_UNLESS(firstResultNdType, "GenericSwLayerOp '{0}' result(0) has no NDTypeInterface", op.getCallee());
    const auto dimsOrder = firstResultNdType.getDimsOrder();

    Dim logicalDim;
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        logicalDim = Dims4D::Act::H;
        break;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        logicalDim = Dims4D::Act::W;
        break;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        logicalDim = Dims4D::Act::C;
        break;
    case VPU::MultiClusterStrategy::SplitOverBatch:
        logicalDim = Dims4D::Act::N;
        break;
    default:
        VPUX_THROW("Strategy '{0}' assigned to a GenericSwLayerOp in unsupported' ", strategy);
    }
    const int64_t activeMemDim = static_cast<int64_t>(dimsOrder.toMemDim(logicalDim).ind());

    const auto axisIdx = kernelInfo.getTilingAxisIndex(activeMemDim);
    VPUX_THROW_UNLESS(axisIdx.has_value(),
                      "Strategy '{0}' maps to mem dim '{1}' which is not a tiling axis of GenericSwLayerOp '{2}'",
                      strategy, activeMemDim, op.getCallee());

    const auto segmentationMemDims = kernelInfo.getInputSegmentationDims(*axisIdx);
    VPUX_THROW_UNLESS(*entryIdx < static_cast<int64_t>(segmentationMemDims.size()),
                      "Input index {0} out of range for inputSegmentationAxes (size {1}) on GenericSwLayerOp '{2}'",
                      *entryIdx, segmentationMemDims.size(), op.getCallee());
    const int64_t segmentationMemDim = segmentationMemDims[*entryIdx];

    // checkStrategyCompatibility only accepts an axis along which every input is classified, so an
    // unknown one here means a strategy was assigned that should have been rejected.
    VPUX_THROW_WHEN(segmentationMemDim == VPU::KernelInfoAttr::kUnknownDim,
                    "Strategy '{0}' assigned to GenericSwLayerOp '{1}' but the segmentation of input {2} along mem "
                    "dim {3} is unknown",
                    strategy, op.getCallee(), *entryIdx, activeMemDim);

    // Broadcast signals that the input is not sliced; return as-is so callers can detect it.
    if (segmentationMemDim == VPU::KernelInfoAttr::kBroadcastDim) {
        return segmentationMemDim;
    }

    // Convert memory dim to logical dim using the input tensor's layout.
    const auto segmentationLogicalDim = inputType.getDimsOrder().dimAt(static_cast<size_t>(segmentationMemDim));
    return static_cast<int64_t>(segmentationLogicalDim.ind());
}

vpux::VPU::DistributionMode vpux::VPU::GenericSwLayerOp::getSWInputTensorDistributionMode(
        VPU::MultiClusterStrategy strategy, mlir::OpOperand& operand, vpux::NDTypeInterface /*inputType*/) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    }

    const auto inputNdType = mlir::cast<vpux::NDTypeInterface>(operand.get().getType());
    const auto segmentationDimOpt = getGenericSwLayerInputSegmentationDim(*this, strategy, operand, inputNdType);
    if (!segmentationDimOpt.has_value() || *segmentationDimOpt < 0) {
        return DistributionMode::DUPLICATED;
    }

    return DistributionMode::SEGMENTED;
}

SmallVector<int64_t> vpux::VPU::GenericSwLayerOp::getSWInputTensorNumTiles(int64_t numClusters,
                                                                           VPU::MultiClusterStrategy strategy,
                                                                           mlir::OpOperand& operand,
                                                                           vpux::NDTypeInterface inputType) {
    SmallVector<int64_t> noTile(inputType.getRank(), 1);

    const auto segmentationDimOpt = getGenericSwLayerInputSegmentationDim(*this, strategy, operand, inputType);
    if (!segmentationDimOpt.has_value() || *segmentationDimOpt < 0) {
        return noTile;
    }

    noTile[*segmentationDimOpt] = numClusters;
    return noTile;
}
