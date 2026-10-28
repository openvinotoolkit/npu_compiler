//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/InliningUtils.h>
#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::Core {

/// @brief This interface is called by the inliner when it processes 'func' operations. It then dispatches
/// to other inliner interfaces if that operation has the "inliner_dispatch" attribute.
/// @details When the inliner processes an operation, it dispatches to the interface that is associated
/// with that operation's dialect. For the 'func' dialect this is precisely UnifiedFuncInlinerInterface.
/// UnifiedFuncInlinerInterface then dispatches further to the interface that is associated with the attribute
/// "inliner_dispatch" of that operation. Users can register custom interfaces using
/// registerDispatchedInlinerInterface(). If no attribute named "inliner_dispatch" is found or no interface is
/// registered to that attribute, a fallback implementation is used. This fallback implementation mirrors MLIR's 'func'
/// inliner extension. This means that if the user wants "default" behaviour for inlining, no additional interfaces have
/// to be registered here.
struct UnifiedFuncInlinerInterface final : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable, bool wouldBeCloned) const final;

    bool isLegalToInline(mlir::Region* dest, mlir::Region* src, bool wouldBeCloned,
                         mlir::IRMapping& valueMapping) const final;

    bool isLegalToInline(mlir::Operation* op, mlir::Region* dest, bool wouldBeCloned,
                         mlir::IRMapping& valueMapping) const final;

    void handleTerminator(mlir::Operation*, mlir::ValueRange) const final;

    void processInlinedCallBlocks(mlir::Operation* call,
                                  mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const final;

    std::tuple<mlir::Block*, mlir::Block::iterator> getInlineBlockAndPoint(mlir::Operation* call) const final;

    void eraseCall(mlir::Operation* call) const final;

    /// @brief Map a particular attribute to a dialect inliner interface. Both have to be provided as a type. Make sure
    /// that the calling dialect has loaded MLIR's func dialect before calling this function!
    template <class AttrT, class DispatchedDialectInlinerInterfaceT>
    void registerDispatchedInlinerInterface() {
        static_assert(std::is_base_of_v<mlir::Attribute, AttrT>);
        static_assert(std::is_base_of_v<mlir::DialectInlinerInterface, DispatchedDialectInlinerInterfaceT>);

        auto funcDialect = getContext()->getLoadedDialect<mlir::func::FuncDialect>();
        VPUX_THROW_WHEN(funcDialect == nullptr,
                        "MLIR's func dialect has not been loaded. Make sure to add the func dialect "
                        "as a dependency using 'let dependentDialects = ...'!");

        auto dialectInterface = std::make_unique<DispatchedDialectInlinerInterfaceT>(funcDialect);
        // Prefer getTypeID() over using an instance of AttrT because not all dialects have been completely loaded
        // yet, which makes printing the attribute impossible.
        const auto wasInserted = _dispatchTable.insert({AttrT::getTypeID(), std::move(dialectInterface)}).second;
        VPUX_THROW_UNLESS(wasInserted, "Cannot register interface for already existing attribute {0}",
                          AttrT::getMnemonic());
    }

private:
    mlir::DialectInlinerInterface* getDispatchInterface(mlir::Operation* op) const;

    mlir::DenseMap<mlir::TypeID, std::unique_ptr<mlir::DialectInlinerInterface>> _dispatchTable;
};

/// Convenience wrapper around UnifiedFuncInlinerInterface::registerDispatchedInlinerInterface()
template <class AttrT, class DispatchedDialectInlinerInterfaceT>
void registerDispatchedInlinerInterface(mlir::MLIRContext* context) {
    auto funcDialect = context->getLoadedDialect<mlir::func::FuncDialect>();
    VPUX_THROW_WHEN(funcDialect == nullptr,
                    "MLIR's func dialect has not been loaded. Make sure to add the func dialect "
                    "as a dependency using 'let dependentDialects = ...'!");

    auto interface = funcDialect->getRegisteredInterface<Core::UnifiedFuncInlinerInterface>();
    VPUX_THROW_WHEN(interface == nullptr,
                    "'Core::UnifiedFuncInlinerInterface' has not been loaded. Make sure that the 'Core' dialect is a "
                    "dependency of this dialect and adds 'Core::UnifiedFuncInlinerInterface' as an extension to the "
                    "'func' dialect!");

    interface->registerDispatchedInlinerInterface<AttrT, DispatchedDialectInlinerInterfaceT>();
}

}  // namespace vpux::Core
