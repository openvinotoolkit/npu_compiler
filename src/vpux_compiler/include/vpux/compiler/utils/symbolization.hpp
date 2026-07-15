//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <utility>

namespace vpux {

// Generates a symbolic name for an operation result based on stripped op name,
// VPURegMapped::IndexType indices, or an explicit counter.
// Used both by SymbolizationPattern::createSymbolicName and populateSymbolMappings.
mlir::FlatSymbolRefAttr buildSymbolicName(mlir::Operation* op, mlir::MLIRContext* ctx,
                                          const std::optional<std::string>& taskTypeString = std::nullopt,
                                          std::optional<size_t> counter = std::nullopt);

struct SymbolizationResult {
    mlir::Operation* newOp = nullptr;
    mlir::SmallVector<mlir::StringAttr> refsToUpdate;

    // default SymbolizationResult is used in rewriters for cases where no movement of op into section is needed
    // e.g. original op is simply removed
    SymbolizationResult() = default;

    SymbolizationResult(mlir::Operation* op): newOp(op) {
    }

    SymbolizationResult(mlir::Operation* op, mlir::SmallVector<mlir::StringAttr> attributes)
            : newOp(op), refsToUpdate(std::move(attributes)) {
    }
};

// A sub-specialization of the default OpConversionPattern, dedicated for symbolization conversions.
// An opConversion pattern has hooks for type system conversions. Symbolization pattern is a form of type conversion
// where OpOperand relationships can materialize as symbolic relationships. The typeConverter does not natively support
// such semantics, since symolization relationships are materialized by attributes. However, it is still required, as
// even tough opOperand relationships dissapear, we would need to register types that would need intermediate
// materialization. The SymbolizationPattern overrides the default matchAndRewrite conversion hook and provides
// one additional hook:
//  - symbolize: the actual conversion method, where instead of the Operand adaptor, we provide a symbolic map
//                  that can be used to lookup an ops symbolic name
template <typename SourceOp>
class SymbolizationPattern : public mlir::OpConversionPattern<SourceOp> {
public:
    using BaseOpT = SourceOp;
    using OneToNOpAdaptor = typename mlir::OpConversionPattern<SourceOp>::OneToNOpAdaptor;
    using SymbolMapper = typename llvm::DenseMap<mlir::Value, mlir::SymbolRefAttr>;
    using SectionMapper = typename std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface>;
    SymbolizationPattern(mlir::func::FuncOp parentFunc, mlir::TypeConverter& typeConverter, SymbolMapper& mapper,
                         SectionMapper& sectionMap, mlir::MLIRContext* ctx)
            : mlir::OpConversionPattern<SourceOp>(typeConverter, ctx),
              _sectionMap(&sectionMap),
              _parentFunc(parentFunc),
              _mapper(&mapper) {
    }

    virtual mlir::FailureOr<SymbolizationResult> symbolize(SourceOp op, SymbolMapper& mapper,
                                                           mlir::ConversionPatternRewriter& rewriter) const = 0;

    void initialize() {
    }

    // helper to unify approach to build unique symbolic names
    SmallVector<mlir::FlatSymbolRefAttr> createSymbolicName(
            SourceOp op, const std::optional<std::string>& taskTypeString = std::nullopt,
            std::optional<size_t> counter = std::nullopt) {
        VPUX_THROW_UNLESS(op->getNumResults() == 1,
                          "Default symbolic converter only supports ops with exactly one result. For {0} got {1}",
                          SourceOp::getOperationName(), op->getNumResults());
        return {buildSymbolicName(op.getOperation(), op.getContext(), taskTypeString, counter)};
    }

    std::pair<mlir::ArrayAttr, mlir::ArrayAttr> processDynamicShapes(mlir::MLIRContext* context,
                                                                     mlir::OperandRangeRange inputShapes,
                                                                     mlir::OperandRangeRange outputShapes) const;

protected:
    mlir::SymbolRefAttr findSym(mlir::Value val) const;

private:
    mlir::LogicalResult matchAndRewrite(SourceOp op, OneToNOpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

protected:
    SectionMapper* _sectionMap;

private:
    mlir::func::FuncOp _parentFunc;
    SymbolMapper* _mapper;
};

template <typename SourceOp>
std::pair<mlir::ArrayAttr, mlir::ArrayAttr> SymbolizationPattern<SourceOp>::processDynamicShapes(
        mlir::MLIRContext* context, mlir::OperandRangeRange inputShapes, mlir::OperandRangeRange outputShapes) const {
    auto placeholderSymbol = mlir::SymbolRefAttr::get(context, "placeholder_symbol");

    auto buildFlatShapeAttr = [&](const mlir::OperandRangeRange& shapes) -> mlir::ArrayAttr {
        size_t totalSymbols = 0;
        for (const auto& values : shapes) {
            totalSymbols += values.empty() ? 1 : values.size();
        }

        SmallVector<mlir::Attribute> flatSyms;
        flatSyms.reserve(totalSymbols);

        for (const auto& values : shapes) {
            if (values.empty()) {
                flatSyms.push_back(placeholderSymbol);
                continue;
            }

            for (auto val : values) {
                flatSyms.push_back(findSym(val));
            }
        }

        return mlir::ArrayAttr::get(context, flatSyms);
    };

    mlir::ArrayAttr inputsShapeAttr = buildFlatShapeAttr(inputShapes);
    mlir::ArrayAttr outputsShapeAttr = buildFlatShapeAttr(outputShapes);

    return {inputsShapeAttr, outputsShapeAttr};
}

template <typename SourceOp>
mlir::LogicalResult SymbolizationPattern<SourceOp>::matchAndRewrite(SourceOp op, OneToNOpAdaptor,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
    auto sym = symbolize(op, *_mapper, rewriter);
    if (mlir::failed(sym)) {
        return mlir::failure();
    }

    auto symRes = sym.value();
    if (!symRes.newOp) {
        return mlir::success();
    }

    auto symbol = moveOpToSection(symRes.newOp, *_sectionMap, rewriter);
    if (symbol != nullptr) {
        (*_mapper)[op.getResult()] = symbol;

        for (auto& attr : symRes.refsToUpdate) {
            symRes.newOp->setAttr(attr, ELF::cloneSectionSymbol(
                                                symbol, mlir::cast<mlir::SymbolRefAttr>(symRes.newOp->getAttr(attr))));
        }
    }

    return mlir::success();
}

template <typename SourceOp>
mlir::SymbolRefAttr SymbolizationPattern<SourceOp>::findSym(mlir::Value val) const {
    auto it = _mapper->find(val);

    VPUX_THROW_WHEN(it == _mapper->end(), "Could not find symbol name entry for {0}, val {1}",
                    SourceOp::getOperationName(), val);

    return it->getSecond();
}

// Wrapper around the native RewritePatternSet, which adds the extra verification that each pattern added is a
// symbolizationPattern. By current design, SymbolizationPatterns are designed to be exclusive. Mixing symbolization
// patterns with simple OpConversion patterns can prove to be too complex with a lot of potential corner cases.
// While symbolizationPatterns should work with the base RewritePatterSet, it is recommended to use
// SymbolizationPatternSet
class SymbolizationPatternSet : private mlir::RewritePatternSet {
    using NativePatternListT = std::vector<std::unique_ptr<mlir::RewritePattern>>;

public:
    // Intentionally do not automatically inherit all constructors,
    SymbolizationPatternSet(mlir::MLIRContext* context): mlir::RewritePatternSet(context) {
    }

    template <typename OpT>
    SymbolizationPatternSet(mlir::MLIRContext* context, std::unique_ptr<SymbolizationPattern<OpT>> pattern)
            : mlir::RewritePatternSet(context, pattern) {
    }

    mlir::MLIRContext* getContext() const {
        return mlir::RewritePatternSet::getContext();
    }

    static mlir::FrozenRewritePatternSet freeze(SymbolizationPatternSet&& symbolPatterns) {
        // mlir::RewritePatternSet &patterns = *this;
        return mlir::FrozenRewritePatternSet(std::move(symbolPatterns));
    }

    NativePatternListT& getNativePatterns() {
        return RewritePatternSet::getNativePatterns();
    }

    //===--------------------------------------------------------------------===//
    // 'add' methods for adding patterns to the set.
    //===--------------------------------------------------------------------===//

    /// Add an instance of each of the pattern types 'Ts' to the pattern list with
    /// the given arguments. Return a reference to `this` for chaining insertions.
    /// Note: ConstructorArg is necessary here to separate the two variadic lists.
    template <typename... Ts, typename ConstructorArg, typename... ConstructorArgs,
              typename = std::enable_if_t<sizeof...(Ts) != 0>>
    SymbolizationPatternSet& add(ConstructorArg&& arg, ConstructorArgs&&... args) {
        (addImpl<Ts, typename Ts::BaseOpT>(/*debugLabels=*/{}, arg, args...), ...);
        return *this;
    }
    /// An overload of the above `add` method that allows for attaching a set
    /// of debug labels to the attached patterns. This is useful for labeling
    /// groups of patterns that may be shared between multiple different
    /// passes/users.
    template <typename... Ts, typename ConstructorArg, typename... ConstructorArgs,
              typename = std::enable_if_t<sizeof...(Ts) != 0>>
    SymbolizationPatternSet& addWithLabel(ArrayRef<StringRef> debugLabels, ConstructorArg&& arg,
                                          ConstructorArgs&&... args) {
        (addImpl<Ts, typename Ts::BaseOpT>(debugLabels, arg, args...), ...);
        return *this;
    }

    /// Add an instance of each of the pattern types 'Ts'. Return a reference to
    /// `this` for chaining insertions.
    template <typename... Ts>
    SymbolizationPatternSet& add() {
        (addImpl<Ts, Ts::BaseOpT>({}), ...);
        return *this;
    }

    /// Add the given native SymbolizationPattern to the pattern list. Return a reference to
    /// `this` for chaining insertions.
    template <typename OpT>
    SymbolizationPatternSet& add(std::unique_ptr<SymbolizationPattern<OpT>> pattern) {
        RewritePatternSet::add(std::move(pattern));
        return *this;
    }

private:
    /// Add an instance of the pattern type 'T'. Return a reference to `this` for
    /// chaining insertions.
    template <typename T, typename OpT, typename... Args>
    std::enable_if_t<std::is_base_of<SymbolizationPattern<OpT>, T>::value> addImpl(ArrayRef<StringRef> debugLabels,
                                                                                   Args&&... args) {
        RewritePatternSet::addWithLabel<T>(debugLabels, std::forward<Args>(args)...);
    }
};

}  // namespace vpux
