//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/IR/Quant.h>

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>

#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

using namespace vpux;

//
// registerDialects
//

namespace {

class MemRefElementTypeModel final : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefElementTypeModel> {};

void registerDialects(mlir::DialectRegistry& registry) {
    registry.insert<vpux::Const::ConstDialect,                //
                    vpux::Core::CoreDialect,                  //
                    vpux::config::ConfigDialect,              //
                    vpux::net::NetDialect,                    //
                    vpux::IE::IEDialect,                      //
                    vpux::VPU::VPUDialect,                    //
                    vpux::VPUIP::VPUIPDialect,                //
                    vpux::VPUIPDPU::VPUIPDPUDialect,          //
                    vpux::VPURT::VPURTDialect,                //
                    vpux::VPUMI37XX::VPUMI37XXDialect,        //
                    vpux::VPUMI40XX::VPUMI40XXDialect,        //
                    vpux::VPUASM::VPUASMDialect,              //
                    vpux::VPURegMapped::VPURegMappedDialect,  //
                    vpux::ELF::ELFDialect,                    //
                    vpux::HostExec::HostExecDialect,          //
                    vpux::NPUReg37XX::NPUReg37XXDialect,      //
                    vpux::NPUReg40XX::NPUReg40XXDialect,      //
                    vpux::NPUReg50XX::NPUReg50XXDialect,      //
                    vpux::ELFNPU37XX::ELFNPU37XXDialect,      //
                    vpux::type::QuantileFloatDialect>();

    registry.insert<mlir::func::FuncDialect,       //
                    mlir::async::AsyncDialect,     //
                    mlir::memref::MemRefDialect,   //
                    mlir::quant::QuantDialect,     //
                    mlir::tensor::TensorDialect,   //
                    mlir::arith::ArithDialect,     //
                    mlir::affine::AffineDialect,   //
                    mlir::scf::SCFDialect,         //
                    mlir::math::MathDialect,       //
                    mlir::cf::ControlFlowDialect,  //
                    mlir::LLVM::LLVMDialect,       //
                    mlir::linalg::LinalgDialect,   //
                    mlir::index::IndexDialect>();
}

}  // namespace

mlir::DialectRegistry vpux::createDialectRegistry(DummyOpMode dummyOpMode) {
    mlir::DialectRegistry registry;
    registerDialects(registry);

    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::quant::QuantDialect*) {
        mlir::quant::AnyQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::UniformQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::UniformQuantizedPerAxisType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::QuantileQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::QuantileQuantizedPerAxisType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::CalibratedQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::type::QuantileFloatDialect*) {
        vpux::type::NF4Type::attachInterface<MemRefElementTypeModel>(*ctx);
    });

    Const::ConstDialect::setupExtraInterfaces(registry);
    Core::CoreDialect::setupExtraInterfaces(registry);
    VPUIP::VPUIPDialect::setupExtraInterfaces(registry);
    VPU::registerAlignedChannelsOpInterfacesVPU(registry);

    // Register the translation to LLVM IR with MLIR
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::LLVM::registerInlinerInterface(registry);
    if (dummyOpMode == DummyOpMode::ENABLED) {
        VPUIP::VPUIPDialect::setupExtraInterfacesAdditional(registry);
    }

    return registry;
}
