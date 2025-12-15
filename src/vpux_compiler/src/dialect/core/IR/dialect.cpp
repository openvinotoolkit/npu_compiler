//
// Copyright (C) 2025 Intel Corporation.
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

    vpux::TensorAttr::attachInterface<TensorEncodingVerifier>(*getContext());
}

void vpux::Core::CoreDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext*, mlir::func::FuncDialect* dialect) {
        dialect->addInterfaces<Core::UnifiedFuncInlinerInterface>();
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/core/dialect.cpp.inc>
