//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/IR/unified_func_inliner_interface.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/TensorEncoding.h>

using namespace vpux;

namespace {
struct CoreInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

class TensorEncodingVerifier final :
        public mlir::VerifiableTensorEncoding::ExternalModel<TensorEncodingVerifier, vpux::TensorAttr> {
public:
    using ConcreteEntity = mlir::DictionaryAttr;

    mlir::LogicalResult verifyEncoding(mlir::Attribute attr, ArrayRef<int64_t> shape, mlir::Type,
                                       FuncRef<mlir::InFlightDiagnostic()> emitError) const {
        const auto desc = mlir::dyn_cast<vpux::TensorAttr>(attr);

        if (desc == nullptr) {
            return printTo(emitError(), "Unsupported TensorType encoding '{0}'", attr);
        }

        if (const auto orderAttr = desc.getOrder()) {
            const auto map = orderAttr.getValue();

            if (!map.isPermutation()) {
                return printTo(emitError(), "TensorType order '{0}' is not a permutation", map);
            }

            if (checked_cast<size_t>(map.getNumResults()) != shape.size()) {
                return printTo(emitError(), "TensorType order '{0}' doesn't match to shape '{1}'", map, shape);
            }
        }

        return verifyBounds(desc, shape, emitError);
    }

private:
    // Bounds must match the rank and be non-negative. Existing producers still leave bounds on
    // fully static shapes or bounds that differ from static dimensions.
    // TODO: E#233281 and E#233282 fix the producers before rejecting these cases.
    static mlir::LogicalResult verifyBounds(vpux::TensorAttr desc, ArrayRef<int64_t> shape,
                                            FuncRef<mlir::InFlightDiagnostic()> emitError) {
        const auto bounds = desc.getBounds();
        if (bounds.empty()) {
            // A dynamic shape without bounds stays legal. Unlike a MemRefAttr, a TensorAttr may
            // reach a pass that has no bounds to propagate yet.
            return mlir::success();
        }

        if (bounds.size() != shape.size()) {
            return printTo(emitError(), "TensorType bounds '{0}' do not match to shape '{1}'", bounds, shape);
        }

        for (const auto& [idx, bound] : llvm::enumerate(bounds.raw())) {
            if (bound < 0) {
                return printTo(emitError(), "TensorType bounds '{0}' has negative value '{1}' at dim '{2}'", bounds,
                               bound, idx);
            }
        }

        return mlir::success();
    }
};

}  // namespace

//
// initialize
//

void vpux::Core::CoreDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/core/ops.cpp.inc>
            >();

    addInterfaces<CoreInlinerInterface>();
    addInterfaces<ShaveBinaryResourcesCache>();
}

void vpux::Core::CoreDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext*, mlir::func::FuncDialect* dialect) {
        dialect->addInterfaces<Core::UnifiedFuncInlinerInterface>();
    });

    // TensorAttr is a DictionaryAttr subclass, so it belongs to the builtin dialect and not to this
    // one, and parsing a tensor type never loads the Core dialect. vpux-opt in turn loads a dialect
    // only when it first parses that dialect's prefix. Attaching from initialize() would therefore
    // leave the verifier unreachable for every type in a file that holds nothing but func and
    // builtin ops, because the type parser only calls verifyEncoding on an encoding that already
    // carries the interface. The MLIRContext constructor loads BuiltinDialect, and
    // appendDialectRegistry applies an extension to a dialect that is already loaded, so attaching
    // here lands before the first parse.
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        vpux::TensorAttr::attachInterface<TensorEncodingVerifier>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/core/dialect.cpp.inc>
