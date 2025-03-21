//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <optional>

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux {
namespace VPUIP {

constexpr int64_t NPU40XX_SW_KERNEL_ADDRESS_ALIGNMENT = 32;

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<mlir::Attribute> attrStorage;

    for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        if (kernelRun.getAttrs().has_value()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                attrStorage.push_back(attr);
            }
        }
    }
    return attrStorage;
}

mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }
    return innerModule;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, StringRef kernelEntryName,
                                          StringRef kernelSourceFileName, StringRef layerName, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    auto vpuswModule = getVPUSWModule(module, log);
    // First try is to keep request name, buildin_+VPUOpName. This happens if VPU Op have just one entry_point kernel
    // used in the model. If more VpuOp are present, but with different implementation, then we will create unique name
    // by append to the name the code entry_point string.
    // Move locally to check first original name and after, by appending ".entry_point", that any name already exist and
    // not create new one.
    SmallString builtInFunctionNameFinal(builtInFunctionName);
    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(builtInFunctionName)) {
        // Check if already created prebuiltFunction refer to the same code entry_point. If this happens, then no new
        // function will be created and will return the old created name.
        const auto prebuiltKernelEntryPoint = prebuiltFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
        if (prebuiltKernelEntryPoint == kernelEntryName) {
            log.trace("Found builtin function: {0}", builtInFunctionName);
            auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
            auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});
            return builtInFunction;
        }
        // In this case builtInFunctionName waa found, but not have same entr_point, so not reffere to same kernel code.
        // It will be check if version with append ".entry_point" already exist.
        builtInFunctionNameFinal.append(".");
        builtInFunctionNameFinal.append(kernelEntryName);
        if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(builtInFunctionNameFinal)) {
            log.trace("Found builtin function: {0}", builtInFunctionNameFinal);
            auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionNameFinal);
            auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});
            return builtInFunction;
        }
    }

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionNameFinal);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto builtInOp = innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionNameFinal,
                                                                   funcType);

    // modifying attributes
    builtInOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    builtInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    builtInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));
    builtInOp->setAttr("VPU.kernel_name", mlir::StringAttr::get(ctx, layerName));
    builtInOp->setAttr("VPU.task_type",
                       mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE)));

    log.trace("Added new builtin function: {0}", builtInFunctionNameFinal);
    return builtInFunction;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, VPU::LayerOpInterface origOp,
                                          ArrayRef<mlir::Value> operands, ArrayRef<mlir::Value> results,
                                          const VPUIP::KernelInfo& kernelInfo, const Logger& log) {
    OpBuilderLogger builderLog(log);

    SmallString builtInFunctionName{VPUIP::SW_KERNEL_NAME_PREFIX};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    const auto convertToUnrankedTypes = [](mlir::Value operand) -> SmallVector<mlir::Type> {
        SmallVector<mlir::Type> result;
        if (auto type = mlir::dyn_cast_or_null<mlir::MemRefType>(operand.getType())) {
            result.emplace_back(mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace()));
        } else if (auto type = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(operand.getType())) {
            const auto dataNDType = mlir::cast<NDTypeInterface>(type.getData());
            result.emplace_back(mlir::UnrankedMemRefType::get(dataNDType.getElementType(), dataNDType.getMemSpace()));
            const auto shapeNDType = mlir::cast<NDTypeInterface>(type.getDynamicShape());
            result.emplace_back(mlir::UnrankedMemRefType::get(shapeNDType.getElementType(), shapeNDType.getMemSpace()));
        }
        VPUX_THROW_UNLESS(!result.empty(),
                          "Only MemRef or VPUIP::BoundedBufferType type are supported as createBuiltInFunction "
                          "operands, got: {0}",
                          operand.getType());
        return result;
    };

    auto& args = kernelInfo.args;
    SmallVector<mlir::Type> inputTypes;
    for (const auto& operand : operands) {
        inputTypes.append(convertToUnrankedTypes(operand));
    };
    for (const auto& result : results) {
        inputTypes.append(convertToUnrankedTypes(result));
    };
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [&module](mlir::Attribute arg) {
        const auto typedAttr = arg.dyn_cast<mlir::TypedAttr>();
        return typedAttr != nullptr ? typedAttr.getType() : mlir::NoneType::get(module.getContext());
    });

    return VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypes, kernelInfo.entryName,
                                        kernelInfo.sourceFileName, kernelInfo.layerName, log);
}

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::VPU::ArchKind arch) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName = static_cast<const SmallString>("nnActEntry");

    // check if runtimeKernel already created
    auto runtimeKernelFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(runtimeKernelName);
    if (runtimeKernelFunction) {
        log.trace("Found builtin function: {0}", runtimeKernelName);
        return;
    }

    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    // creating runtime kernel function
    const auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto runtimeFunctionOp =
            innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), runtimeKernelName, funcType);

    // modifying attributes
    runtimeFunctionOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    runtimeFunctionOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, runtimeKernelEntryName));

    log.trace("Added runtime kernel function: {0}", runtimeKernelEntryName);

    // creating name symbol
    auto runtimeFlatSym = mlir::SymbolRefAttr::get(ctx, runtimeKernelName);
    auto runtimeSym = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {runtimeFlatSym});

    static constexpr int64_t defaultStackSize = 4096;

    // TODO: always extract num shaves info from VPURT::SW.Runtime, which can be extracted from module
    auto maxShaves = 4;
    if (arch == vpux::VPU::ArchKind::NPU40XX) {
        maxShaves = 12;
    }
    SmallVector<int64_t> stacksArray(maxShaves, defaultStackSize);

    //  adding runtime kernel configuration - stacks, etc
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    moduleBuilder.create<VPURT::SWRunTimeOp>(mlir::UnknownLoc::get(ctx), runtimeSym, getIntArrayAttr(ctx, stacksArray));
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  ArrayRef<mlir::Attribute> args, const Logger& log, VPUIP::SwKernelRun swKernelRunOp) {
    OpBuilderLogger builderLog(log);
    auto* ctx = swKernelOp.getContext();
    auto& bodyRegion = swKernelOp.getBody();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);

    auto argsAttr = args.empty() ? nullptr : mlir::ArrayAttr::get(ctx, args);

    if (swKernelRunOp != nullptr) {
        auto numBlockArgs = swKernelBlock.getNumArguments();
        auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
        VPUX_THROW_UNLESS(numSwKernelRunArgs != 0, "SW Kernel Run has 0 Operands at '{0}'", swKernelOp->getLoc());
        VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'",
                          swKernelOp->getLoc());
        auto tileNum = numBlockArgs / numSwKernelRunArgs;

        VPUX_THROW_UNLESS(swKernelOp.getInputs().size() % tileNum == 0 && swKernelOp.getResults().size() % tileNum == 0,
                          "Invalid block arg num at '{0}'", swKernelOp->getLoc());
        auto numSwKernelRunInputs = swKernelOp.getInputs().size() / tileNum;
        auto numSwKernelRunOutputs = swKernelOp.getResults().size() / tileNum;

        if (argsAttr != nullptr) {
            swKernelRunOp.setAttrsAttr(argsAttr);
        }

        for (auto tileIdx : irange(tileNum)) {
            auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
            for (auto argInputIdx : irange(numSwKernelRunInputs)) {
                newRunOp->setOperand(checked_cast<unsigned int>(argInputIdx),
                                     swKernelBlock.getArgument(
                                             checked_cast<unsigned int>(tileIdx * numSwKernelRunInputs + argInputIdx)));
            }

            for (auto argOutputIdx : irange(numSwKernelRunOutputs)) {
                newRunOp->setOperand(
                        checked_cast<unsigned int>(numSwKernelRunInputs + argOutputIdx),
                        swKernelBlock.getArgument(checked_cast<unsigned int>(
                                tileNum * numSwKernelRunInputs + tileIdx * numSwKernelRunOutputs + argOutputIdx)));
            }

            log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
        }
    } else {
        swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands),
                                                        argsAttr);
    }
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log) {
    auto& bodyRegion = swKernelOp.getBody();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    OpBuilderLogger builderLog(log);
    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(swKernelOp.getInputs());
    addBlockArgs(swKernelOp.getOutputBuffs());

    auto numBlockArgs = swKernelBlock.getNumArguments();
    auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
    VPUX_THROW_UNLESS(numSwKernelRunArgs != 0, "SW Kernel Run has 0 Operands at '{0}'", swKernelOp->getLoc());
    VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto tileNum = numBlockArgs / numSwKernelRunArgs;

    VPUX_THROW_UNLESS(swKernelOp.getInputs().size() % tileNum == 0 && swKernelOp.getResults().size() % tileNum == 0,
                      "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto numSwKernelRunInputs = swKernelOp.getInputs().size() / tileNum;
    auto numSwKernelRunOutputs = swKernelOp.getResults().size() / tileNum;

    // pack input/outputs and constants into several sw_kernel_run calls
    // For example: For Operation that has 2 inputs, 1 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First input with 1st tile
    //         [INPUT1_TILE0] as %arg1: Second input with 1st tile
    //         [INPUT0_TILE1] as %arg2: First input with 2nd tile
    //         [INPUT1_TILE1] as %arg3: Second input with 2nd tile
    // outputs:[OUTPUT_TILE0] as %arg4: Output of 1st tile
    //         [OUTPUT_TILE1] as %arg5: Output of 2nd tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg1, %arg4)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg2, %arg3, %arg5)
    // For example: For Operation that has 1 input, 2 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First input with 1st tile
    //         [INPUT0_TILE1] as %arg1: First input with 2nd tile
    // outputs:[OUTPUT_TILE0] as %arg2: First Output of 1st tile
    //         [OUTPUT_TILE1] as %arg3: Second Output of 1st tile
    //         [OUTPUT_TILE0] as %arg4: First Output of 2nd tile
    //         [OUTPUT_TILE1] as %arg5: Second Output of 2nd tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg2, %arg3)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg1, %arg4, %arg5)
    for (auto tileIdx : irange(tileNum)) {
        auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
        for (auto argInputIdx : irange(numSwKernelRunInputs)) {
            newRunOp->setOperand(checked_cast<unsigned int>(argInputIdx),
                                 swKernelBlock.getArgument(
                                         checked_cast<unsigned int>(tileIdx * numSwKernelRunInputs + argInputIdx)));
        }

        for (auto argOutputIdx : irange(numSwKernelRunOutputs)) {
            newRunOp->setOperand(
                    checked_cast<unsigned int>(numSwKernelRunInputs + argOutputIdx),
                    swKernelBlock.getArgument(checked_cast<unsigned int>(
                            tileNum * numSwKernelRunInputs + tileIdx * numSwKernelRunOutputs + argOutputIdx)));
        }

        log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
    }
}

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp) {
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    VPUX_THROW_WHEN(kernelFunc == nullptr, "Cannot find kernel function symbol at '{0}'", swKernelOp->getLoc());
    auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_name");
    // Ensure backward compatibility; kernel_name can be the same as kernel_entry.
    if (kernelEntryPoint == nullptr) {
        kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    }
    VPUX_THROW_WHEN(kernelEntryPoint == nullptr, "Cannot find kernel entry point at '{0}'", swKernelOp->getLoc());
    return kernelEntryPoint.getValue();
}

// Check whether SwKernelOp is activation.
bool isActivationSwKernelOp(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_ACTIVATION_KERNELS, kernelEntryName) != SW_ACTIVATION_KERNELS.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp supports tiling.
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_SUPPORTING_TILING, kernelEntryName) != SW_KERNELS_SUPPORTING_TILING.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp support discontinuous input/output.
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    // SubView can be used for Softmax because it is always tilied on the highest dimension.
    if (kernelEntryName == "softmax" ||
        llvm::find(SW_KERNELS_SUPPORTING_STRIDE, kernelEntryName) != SW_KERNELS_SUPPORTING_STRIDE.end()) {
        return true;
    }
    return false;
}

namespace {
// reverse int attribute from the physical order
int64_t reverseMemDim(DimsOrder inOrder, int64_t dimIdx) {
    const auto origPerm = inOrder.toPermutation();
    return origPerm[origPerm.size() - 1 - dimIdx].ind();
}

// reverse int array attribute from the physical order
SmallVector<int64_t> reverseIntArrayAttr(DimsOrder inOrder, mlir::ArrayAttr arrayAttr) {
    const auto origPerm = inOrder.toPermutation();
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> permArray(arrayAttr.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revDstInd] = origArray[revSrcInd];
    }
    return permArray;
}

// permute int array attribute in the physical order
SmallVector<int64_t> permuteIntArrayAttr(DimsOrder inOrder, ArrayRef<int64_t> origArray) {
    const auto origPerm = inOrder.toPermutation();
    SmallVector<int64_t> permArray(origArray.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revSrcInd] = origArray[revDstInd];
    }
    return permArray;
}

InputTiling backInferInterpolateSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                  Logger& log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();
    auto inOrder = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getDimsOrder();

    std::optional<SmallVector<int64_t>> coordinatesShape;
    std::optional<SmallVector<int64_t>> lambdasShape;
    if (inputs.size() >= 2) {
        const auto coordinates = inputs[1];
        coordinatesShape = to_small_vector(getShape(coordinates));
    }
    if (inputs.size() >= 3) {
        const auto lambdas = inputs[2];
        lambdasShape = to_small_vector(getShape(lambdas));
    }

    const auto interpolateMode = static_cast<IE::InterpolateMode>(attrs[1].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto coordMode = static_cast<IE::InterpolateCoordMode>(attrs[2].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto nearestMode = static_cast<IE::InterpolateNearestMode>(attrs[3].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto initialInputDims = reverseIntArrayAttr(inOrder, attrs[6].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputDims = reverseIntArrayAttr(inOrder, attrs[7].dyn_cast<mlir::ArrayAttr>());
    const auto initialInputOffset = reverseIntArrayAttr(inOrder, attrs[10].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputOffset = reverseIntArrayAttr(inOrder, attrs[11].dyn_cast<mlir::ArrayAttr>());

    const auto currentInputDims = to_small_vector(getShape(inputs[0]));

    return vpux::backInferInterpolateTile(outputTile, initialInputDims, initialOutputDims, initialInputOffset,
                                          initialOutputOffset, currentInputDims, coordinatesShape, lambdasShape,
                                          interpolateMode, coordMode, nearestMode, log);
}

int64_t convertKernelAxisToOrigAxis(mlir::Value tensorArg, int64_t kernelAxis) {
    const auto shape = getShape(tensorArg);
    // Dims/Order sequence is not same on kernel-FW & compiler side. Convert the axis from kernel to compiler
    // representation.
    auto nDims = checked_cast<uint32_t>(shape.size());

    return nDims - 1 - kernelAxis;
}

InputTiling backInferGatherSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();

    const auto kernelAxis = attrs[0].dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto axisValue = convertKernelAxisToOrigAxis(inputs[0], kernelAxis);
    const auto batchDims = attrs[1].dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origIndicesShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    const auto indicesRank = attrs[2].dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();

    return vpux::backInferGatherTile(outputTile, origInputShape, origIndicesShape, axisValue, batchDims, false,
                                     indicesRank, log);
}

InputTiling backInferGatherElementsSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                     Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();

    const auto kernelAxis = attrs[0].dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto axisValue = convertKernelAxisToOrigAxis(inputs[0], kernelAxis);

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origIndicesShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    return vpux::backInferGatherElementsTile(outputTile, origInputShape, origIndicesShape, axisValue,
                                             origIndicesShape.size(), log);
}

InputTiling backInferRMSSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                          Logger /*log*/) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();
    TileInfo gammaTile(getShape(inputs[1]));
    const auto inShape = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    auto inTile = outputTile;
    gammaTile.shape[Dim(0)] = inShape[Dim(0)];

    return TilingInfo{{std::move(inTile), std::move(gammaTile)}};
}

InputTiling backInferRoPESwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                           Logger /*log*/) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();
    TileInfo cosTile(getShape(inputs[1]));
    TileInfo sinTile(getShape(inputs[2]));
    auto inTile = outputTile;

    cosTile.shape[Dim(2)] = inTile.shape[Dim(2)];
    cosTile.offsets[Dim(3)] = inTile.offsets[Dim(3)];
    cosTile.offsets[Dim(2)] = inTile.offsets[Dim(2)];
    cosTile.offsets[Dim(0)] = cosTile.offsets[Dim(1)] = inTile.offsets[Dim(1)];

    sinTile.shape[Dim(2)] = inTile.shape[Dim(2)];
    sinTile.offsets[Dim(3)] = inTile.offsets[Dim(3)];
    sinTile.offsets[Dim(2)] = inTile.offsets[Dim(2)];
    sinTile.offsets[Dim(0)] = sinTile.offsets[Dim(1)] = inTile.offsets[Dim(1)];

    return TilingInfo{{std::move(inTile), std::move(cosTile), std::move(sinTile)}};
}

InputTiling backInferDepthToSpaceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                   Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    auto inShape = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>().getShape();
    const auto blockSize = attrs[0].cast<mlir::IntegerAttr>().getInt();

    return vpux::backInferDepthToSpaceTile(outputTile, inShape, blockSize, log);
}

InputTiling backInferPadSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto origInputType = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto origInputShape = origInputType.getShape();
    const auto origOutputShape = swKernelOp.getResults()[0].getType().cast<vpux::NDTypeInterface>().getShape();
    const auto order = origInputType.getDimsOrder();

    // Padding attr at VPUIP dialect are stored in memory order so convert to default order
    // to be aligned with shape representation
    const auto origPadsBegin = reverseIntArrayAttr(order, attrs[0].dyn_cast<mlir::ArrayAttr>());
    const auto origPadsEnd = reverseIntArrayAttr(order, attrs[1].dyn_cast<mlir::ArrayAttr>());

    return backInferPadTile(outputTile, origInputShape, origOutputShape, Shape(origPadsBegin), Shape(origPadsEnd), log);
}

InputTiling backInferReduceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             StringRef kernelEntryName, Logger log) {
    log.trace("Try to back infer input tiling for {0}, output tile: {1}", kernelEntryName, outputTile);

    const auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    const auto numInputs = swKernelOp.getInputs().size();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    VPUX_THROW_UNLESS(numInputs, "SwKernelOp {0} should have 1 input, got '{1}'", swKernelOp, numInputs);

    const auto input = swKernelOp.getOperand(0);
    const auto inputOrder = input.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    const auto inputShape = getShape(input);
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(attrs.size() == 3, "SwKernelOp {0} should have 3 attributes, got '{1}'", swKernelOp,
                      attrs.size());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto inputTile = outputTile;
    const auto reversedAxes = parseIntArrayAttr<int64_t>(attrs[2].cast<mlir::ArrayAttr>());
    for (const auto reversedAxis : reversedAxes) {
        const auto axis = reverseMemDim(inputOrder, reversedAxis);
        const auto d = Dim(axis);
        inputTile.shape[d] = inputShape[d];
    }

    return TilingInfo{std::move(inputTile)};
}

InputTiling backInferMatMulSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             Logger log) {
    log.trace("Try to back infer input tiling for matmul, output tile: {0}", outputTile);

    const auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    const auto numInputs = swKernelOp.getInputs().size();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    VPUX_THROW_UNLESS(numInputs == 2, "SwKernelOp {0} should have 2 inputs, got '{1}'", swKernelOp, numInputs);

    const auto input1 = swKernelOp.getOperand(0);
    const auto input2 = swKernelOp.getOperand(1);
    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(attrs.size() == 2, "SwKernelOp {0} should have 2 attributes, got '{1}'", swKernelOp,
                      attrs.size());
    VPUX_THROW_UNLESS(input1Shape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());
    VPUX_THROW_UNLESS(input2Shape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto input1Tile = outputTile;
    input1Tile.shape[Dim(input1Tile.shape.size() - 2)] = input1Shape[Dim(input1Shape.size() - 2)];
    input1Tile.shape[Dim(input1Tile.shape.size() - 1)] = input1Shape[Dim(input1Shape.size() - 1)];

    auto input2Tile = outputTile;
    input2Tile.shape[Dim(input2Tile.shape.size() - 2)] = input2Shape[Dim(input2Shape.size() - 2)];
    input2Tile.shape[Dim(input2Tile.shape.size() - 1)] = input2Shape[Dim(input2Shape.size() - 1)];

    return InputTiling{{std::move(input1Tile), std::move(input2Tile)}};
}

SmallVector<mlir::Attribute> getInterpolateSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                       ArrayRef<mlir::Attribute> origAttr,
                                                                       const TilingInfo& inputTiling,
                                                                       const TileInfo& outTile, Logger log) {
    log.trace("update attrs for SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    // Get output tile against the original output
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    auto dim = swKernelOp.getInputs()[0].getType().dyn_cast<vpux::NDTypeInterface>().getDimsOrder();
    TileInfo inputTile = inputTiling.tiles[0];
    const auto initialInputDims = reverseIntArrayAttr(dim, attrs[6].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputDims = reverseIntArrayAttr(dim, attrs[7].dyn_cast<mlir::ArrayAttr>());
    const auto initialInputOffset = reverseIntArrayAttr(dim, attrs[10].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputOffset = reverseIntArrayAttr(dim, attrs[11].dyn_cast<mlir::ArrayAttr>());
    const auto localInputOffset = to_small_vector(inputTile.offsets);
    const auto localOutputOffset = to_small_vector(outTile.offsets);
    SmallVector<int64_t> inputTileOffset;
    SmallVector<int64_t> outputTileOffset;
    std::transform(localInputOffset.begin(), localInputOffset.end(), initialInputOffset.begin(),
                   std::back_inserter(inputTileOffset), std::plus<int64_t>());
    std::transform(localOutputOffset.begin(), localOutputOffset.end(), initialOutputOffset.begin(),
                   std::back_inserter(outputTileOffset), std::plus<int64_t>());
    auto newInputTiling = inputTiling;
    newInputTiling.tiles[0].offsets = Shape(inputTileOffset);
    auto newOutputTile = outTile;
    newOutputTile.offsets = Shape(outputTileOffset);
    newAttrs[10] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, inputTileOffset));
    newAttrs[11] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, outputTileOffset));
    return newAttrs;
}

SmallVector<mlir::Attribute> getPadSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                               ArrayRef<mlir::Attribute> origAttr,
                                                               const TileInfo& outTile, Logger log) {
    log.trace("update attrs for Pad SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    const auto outType = swKernelOp.getResults()[0].getType().cast<vpux::NDTypeInterface>();
    const auto outShape = outType.getShape();
    auto order = outType.getDimsOrder();

    // Padding attrs at VPUIP dialect are stored in memory-order so convert to default-order
    // to be aligned with shape representation
    auto padsBegin = reverseIntArrayAttr(order, attrs[0].dyn_cast<mlir::ArrayAttr>());
    auto padsEnd = reverseIntArrayAttr(order, attrs[1].dyn_cast<mlir::ArrayAttr>());

    vpux::updatePadOpAttrsAfterTiling(outShape, outTile, padsBegin, padsEnd);

    // Convert new pads back to memory-order
    newAttrs[0] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(order, padsBegin));
    newAttrs[1] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(order, padsEnd));
    return newAttrs;
}

SmallVector<mlir::Attribute> getLstmSequenceSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                        ArrayRef<mlir::Attribute> origAttr,
                                                                        const TileInfo& outTile, Logger log) {
    log.trace("Update attrs for LSTMSequence SwKernelOp at '{0}' for out tile {1}", swKernelOp, outTile);

    SmallVector<mlir::Attribute> newAttrs(origAttr.begin(), origAttr.end());
    const auto isTileOverNumDirections = outTile.axis[Dims4D::Act::C] > 1;
    if (!isTileOverNumDirections) {
        return newAttrs;
    }

    // If the operator is tiled along the numDirections dimension, it indicates that it is bidirectional
    // (directionAttr = 2) and is split into one forward (directionAttr = 0) and one reverse (directionAttr = 1)
    // operator. The new attribute value conveniently matches the offset of the numDirections dimension in the output
    // tile.
    const auto numDirectionsDimOffset = outTile.offsets[Dims4D::Act::C];
    const auto newDirectionAttr = getIntAttr(swKernelOp.getContext(), numDirectionsDimOffset);
    newAttrs[0] = newDirectionAttr;

    return newAttrs;
}

InputTiling backInferTopKSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto inOrder = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    const auto attrs = swKernelRun.getAttrs().value();
    const auto axis = reverseMemDim(inOrder, attrs[0].cast<mlir::IntegerAttr>().getInt());

    const auto isLargerThanZero = [](const int64_t dimSize) -> bool {
        return dimSize > 0;
    };

    const auto inShape = getShape(swKernelOp.getInputs()[0]);
    SmallVector<TileInfo> inputTiles;

    VPUX_THROW_UNLESS(inShape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        const auto d = Dim(ind);
        if (axis == d.ind()) {
            curTile.shape[d] = inShape[d];
        }
    }
    inputTiles.push_back(curTile);
    if (swKernelOp.getInputs().size() > 1) {
        const auto topKBufferShape = getShape(swKernelOp.getInputs()[1]);
        TileInfo topKBufferTile(topKBufferShape);
        topKBufferTile.shape[Dims4D::Act::W] = topKBufferShape[Dims4D::Act::W] / 2;
        if (llvm::any_of(outputTile.offsets, isLargerThanZero)) {
            topKBufferTile.offsets[Dims4D::Act::W] = topKBufferShape[Dims4D::Act::W] / 2;
        }
        inputTiles.push_back(topKBufferTile);
    }

    return TilingInfo{inputTiles};
}

bool isReduceKernelEntry(StringRef kernelEntryName) {
    static const std::unordered_set<std::string> reduceEntryNames = {
            "reduce_l1",   "reduce_l2",  "reduce_logical_and", "reduce_logical_or", "reduce_max",
            "reduce_mean", "reduce_min", "reduce_prod",        "reduce_sum"};

    return reduceEntryNames.find(kernelEntryName.str()) != reduceEntryNames.end();
}

InputTiling backInferGRUSequenceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTileY,
                                                  Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origInitialHiddenStateShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origWShape = inputs[2].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origRShape = inputs[3].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origBShape = inputs[4].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo wTile(origWShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(wTile), std::move(rTile),
                        std::move(bTile)}};
}

InputTiling backInferGRUSequenceLastPartSwKernelInputTile(VPUIP::SwKernelOp swKernelOp,
                                                          const vpux::TileInfo& outputTileY, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origInitialHiddenStateShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origRShape = inputs[2].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origBShape = inputs[3].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(rTile), std::move(bTile)}};
}

InputTiling backInferLSTMGatesSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        auto curTile = outputTile;
        curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];

        inputTiles.push_back(curTile);
    }

    return TilingInfo{inputTiles};
}

InputTiling backInferLSTMCellSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                               Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;

    // inputs
    const auto inputData = swKernelOp.getInputs()[0];
    const auto initialHiddenState = swKernelOp.getInputs()[1];
    //  weight
    const auto weights = swKernelOp.getInputs()[3];
    const auto weightsHidden = swKernelOp.getInputs()[4];
    const auto biases = swKernelOp.getInputs()[5];

    const auto inputDataShape = getShape(inputData);
    const auto initialHiddenStateShape = getShape(initialHiddenState);

    const auto weightsShape = getShape(weights);
    const auto weightsHiddenShape = getShape(weightsHidden);
    const auto biasesShape = getShape(biases);

    TileInfo inputDataTile(inputDataShape);
    TileInfo initialHiddenStateTile(initialHiddenStateShape);
    TileInfo initialCellStateTile = outputTile;

    TileInfo weightsTile(weightsShape);
    weightsTile.shape[Dim(weightsShape.size() - 2)] = outputTile.shape.back();
    weightsTile.offsets[Dim(weightsShape.size() - 2)] = outputTile.offsets.back();
    weightsTile.axis[Dim(weightsShape.size() - 2)] = outputTile.axis.back();

    TileInfo weightsHiddenTile(weightsHiddenShape);
    weightsHiddenTile.shape[Dim(weightsHiddenShape.size() - 2)] = outputTile.shape.back();
    weightsHiddenTile.offsets[Dim(weightsHiddenShape.size() - 2)] = outputTile.offsets.back();
    weightsHiddenTile.axis[Dim(weightsHiddenShape.size() - 2)] = outputTile.axis.back();

    TileInfo biasesTile(biasesShape);
    biasesTile.shape[Dim(biasesShape.size() - 1)] = outputTile.shape.back();
    biasesTile.offsets[Dim(biasesShape.size() - 1)] = outputTile.offsets.back();
    biasesTile.axis[Dim(biasesShape.size() - 1)] = outputTile.axis.back();

    inputTiles.push_back(inputDataTile);
    inputTiles.push_back(initialHiddenStateTile);
    inputTiles.push_back(initialCellStateTile);
    inputTiles.push_back(weightsTile);
    inputTiles.push_back(weightsHiddenTile);
    inputTiles.push_back(biasesTile);

    log.trace("backInferLSTMCellSwKernelInputTile  outputTile '{0}'", outputTile);
    log.trace("backInferLSTMCellSwKernelInputTile  inputTiles '{0}'", inputTiles);

    return TilingInfo{inputTiles};
}

InputTiling backInferLSTMSequenceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                   Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    // inputs
    const auto inputData = swKernelOp.getInputs()[0];
    const auto initialHiddenState = swKernelOp.getInputs()[1];
    const auto initialCellState = swKernelOp.getInputs()[2];
    const auto weightsHidden = swKernelOp.getInputs()[3];
    const auto syncBuffer = swKernelOp.getInputs()[4];

    const auto inputDataShape = getShape(inputData);
    const auto initialHiddenStateShape = getShape(initialHiddenState);
    const auto initialCellStateShape = getShape(initialCellState);
    const auto weightsHiddenShape = getShape(weightsHidden);
    const auto syncBufferShape = getShape(syncBuffer);

    TileInfo inputDataTile(inputDataShape);
    TileInfo initialHiddenStateTile(initialHiddenStateShape);
    TileInfo initialCellStateTile(initialCellStateShape);
    TileInfo weightsHiddenTile(weightsHiddenShape);
    TileInfo syncBufferTile(syncBufferShape);

    const auto batchSize = outputTile.shape[Dims4D::Act::N];
    const auto batchOffset = outputTile.offsets[Dims4D::Act::N];
    const auto numDirections = outputTile.shape[Dims4D::Act::C];
    const auto numDirectionsOffset = outputTile.offsets[Dims4D::Act::C];

    inputDataTile.shape[Dims4D::Act::N] = batchSize;
    inputDataTile.shape[Dims4D::Act::C] = numDirections;
    inputDataTile.offsets[Dims4D::Act::N] = batchOffset;
    inputDataTile.offsets[Dims4D::Act::C] = numDirectionsOffset;

    initialHiddenStateTile.shape[Dims4D::Act::N] = batchSize;
    initialHiddenStateTile.shape[Dims4D::Act::C] = numDirections;
    initialHiddenStateTile.offsets[Dims4D::Act::N] = batchOffset;
    initialHiddenStateTile.offsets[Dims4D::Act::C] = numDirectionsOffset;

    initialCellStateTile.shape[Dims4D::Act::N] = batchSize;
    initialCellStateTile.shape[Dims4D::Act::C] = numDirections;
    initialCellStateTile.offsets[Dims4D::Act::N] = batchOffset;
    initialCellStateTile.offsets[Dims4D::Act::C] = numDirectionsOffset;

    weightsHiddenTile.shape[Dims4D::Act::N] = numDirections;
    weightsHiddenTile.offsets[Dims4D::Act::N] = numDirectionsOffset;

    const SmallVector<TileInfo> inputTiles = {std::move(inputDataTile), std::move(initialHiddenStateTile),
                                              std::move(initialCellStateTile), std::move(weightsHiddenTile),
                                              std::move(syncBufferTile)};

    log.trace("backInferLSTMCellSwKernelInputTile  outputTile '{0}'", outputTile);
    log.trace("backInferLSTMCellSwKernelInputTile  inputTiles '{0}'", inputTiles);

    return TilingInfo{inputTiles};
}

InputTiling backInferMvn1SumSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto tileAxis = outputTile.axis;
    auto tilingDims = getNonOneDim(tileAxis);
    VPUX_THROW_UNLESS(tilingDims.size() == 1 && tileAxis.size() == 4,
                      "Only support 4D tensor shape with one dim tiling");
    auto tilingDim = tilingDims.front();

    const auto inShape = getShape(swKernelOp.getInputs()[0]);
    const auto outShape = getShape(swKernelOp.getResult(0));

    TileInfo inputTile(inShape);
    if (tilingDim == Dims4D::Act::N || tilingDim == Dims4D::Act::C) {
        inputTile.shape[tilingDim] = outputTile.shape[tilingDim];
        inputTile.offsets[tilingDim] = outputTile.offsets[tilingDim];
        return TilingInfo{std::move(inputTile)};
    }

    // When tiling at the height dimension, it is a very specific operation
    // where any input H size can only produce one line of output H at each shave excluster
    // The important thing is to establish a rule that can infer the input shape from the output tile shape (1)
    // Here, the same rule as with TileActShavePass and UnrollClusterTilingPass is maintained
    //
    // For example, using 3 clusters and 6 shaves, with an output height of 6 and an input height of 76:
    // If subview is used or MC & MS are tiling on the same dimension,
    // The splitInShapeH list would be:
    //   [Tile0(13), Tile1(13), Tile2(13), Tile3(12), Tile4(13), Tile5(12)]
    // The inTileOffsetH list would be:
    //   [Tile0(0), Tile1(13), Tile2(26), Tile3(39), Tile4(51), Tile5(64)]
    // The index distribution looks like:
    //         SHV0        SHV1
    //   CL0  [Tile0(13)   Tile1(13)]
    //   CL1  [Tile2(13)   Tile3(12)]
    //   CL2  [Tile4(13)   Tile5(12)]
    // At TileActShavePass (outTileShapeH > 1), the Input Tile Info for all clusters at each shave is:
    //   SHV0: data_size: Tile0 + Tile2 + Tile4; data_offset: 0
    //   SHV1: data_size: Tile1 + Tile3 + Tile5; data_offset: Tile0 + Tile2 + Tile4
    // At UnrollClusterTilingPass (outTileShapeH == 1), the Input Tile Info for each shave and each cluster is:
    //   Directly get values from splitInShape and inTileOffsetH through the index.

    const auto inH = inShape[Dims4D::Act::H];
    const auto outH = outShape[Dims4D::Act::H];
    auto largeNumb = static_cast<int64_t>(inH % outH);
    auto baseSize = static_cast<int64_t>(inH / outH);

    SmallVector<int64_t> splitInShape(outH, baseSize);

    // keep first shave indices equal for correct unroll cmx input buffer offsets
    // Distribute on first shave if largeNumb is greater than outH / 2
    int64_t startIdx = (largeNumb > outH / 2) ? 0 : 1;
    int64_t idx = startIdx;
    for (int64_t replaced = 0; replaced < largeNumb; replaced++) {
        splitInShape[idx] += 1;
        idx += 2;

        // Distribute remaining values to the second shave
        if (idx >= outH) {
            idx = 1;
        }
    }

    SmallVector<int64_t> splitInOffset(outH);
    splitInOffset[0] = 0;
    std::partial_sum(splitInShape.begin(), splitInShape.end() - 1, splitInOffset.begin() + 1);

    auto outTileShapeH = outputTile.shape[Dims4D::Act::H];
    auto outTileOffsetH = outputTile.offsets[Dims4D::Act::H];

    auto inTileShapeH = 0;
    auto inTileOffsetH = 0;
    if (outTileShapeH == 1) {
        inTileShapeH = splitInShape[outTileOffsetH];
        inTileOffsetH = splitInOffset[outTileOffsetH];
    } else {
        const auto isLastShaveData = (outTileOffsetH + outTileShapeH == outH);
        int64_t firstShaveSize = std::accumulate(splitInShape.begin(), splitInShape.end(), 0,
                                                 [index = 0](int64_t acc, int64_t val) mutable {
                                                     return (index++ % 2 == 0) ? acc + val : acc;
                                                 });
        inTileShapeH = isLastShaveData ? inH - firstShaveSize : firstShaveSize;
        inTileOffsetH = isLastShaveData ? firstShaveSize : 0;
    }

    inputTile.shape[Dims4D::Act::H] = inTileShapeH;
    inputTile.offsets[Dims4D::Act::H] = inTileOffsetH;

    return TilingInfo{std::move(inputTile)};
}

InputTiling backInferMvn1NormSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto origMeanVarType = swKernelOp.getInputs()[1].getType().cast<vpux::NDTypeInterface>();
    const auto origMeanVarShape = origMeanVarType.getShape();

    TileInfo inDataTile(outputTile);
    TileInfo inMeanVarTile(origMeanVarShape);

    const auto acrossChannels = attrs[0].cast<mlir::BoolAttr>().getValue();
    if (!acrossChannels) {
        inMeanVarTile.shape[Dims4D::Act::C] = inDataTile.shape[Dims4D::Act::C];
        inMeanVarTile.offsets[Dims4D::Act::C] = inDataTile.offsets[Dims4D::Act::C];
    }
    inMeanVarTile.shape[Dims4D::Act::N] = inDataTile.shape[Dims4D::Act::N];
    inMeanVarTile.offsets[Dims4D::Act::N] = inDataTile.offsets[Dims4D::Act::N];

    return TilingInfo{{std::move(inDataTile), std::move(inMeanVarTile)}};
}

}  // namespace

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const SmallVector<vpux::TileInfo>& outputTiles,
                                       int tileId, Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    const auto arch = VPU::getArch(swKernelOp);
    const auto& outputTile = outputTiles[tileId];
    if (kernelEntryName == "interpolate") {
        return backInferInterpolateSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "topk") {
        return backInferTopKSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gather") {
        return backInferGatherSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gather_elements") {
        return backInferGatherElementsSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "rms_norm") {
        return backInferRMSSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "rope") {
        return backInferRoPESwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "pad") {
        return backInferPadSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "mvn1_sum") {
        return backInferMvn1SumSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "mvn1_norm") {
        return backInferMvn1NormSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "depth_to_space") {
        return backInferDepthToSpaceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_sequence") {
        return backInferGRUSequenceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_sequence_last_part") {
        return backInferGRUSequenceLastPartSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (isReduceKernelEntry(kernelEntryName)) {
        return backInferReduceSwKernelInputTile(swKernelOp, outputTile, kernelEntryName, log);
    } else if (kernelEntryName == "matmul") {
        return backInferMatMulSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_gates") {
        return backInferLSTMGatesSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_cell") {
        return backInferLSTMCellSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_sequence") {
        return backInferLSTMSequenceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if ((kernelEntryName == "detection_output_sort") && (arch == VPU::ArchKind::NPU37XX)) {
        return vpux::VPU::DetectionOutputSortOpInputTilingOnShave(swKernelOp, outputTile, tileId, outputTiles.size(),
                                                                  log);
    }

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log) {
    log.trace("Update SwKernel attrs after tiling at '{0}'", swKernelOp->getLoc());
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "interpolate") {
        return getInterpolateSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, inputTiling, outTile, log);
    } else if (kernelEntryName == "pad") {
        return getPadSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else if (kernelEntryName == "lstm_sequence") {
        return getLstmSequenceSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else {
        return SmallVector<mlir::Attribute>(origAttr.begin(), origAttr.end());
    }
}

SmallVector<int64_t> getPopulateWeightTableSwKernelEntries(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<int64_t> weightsPerClusterPtrs;
    if (!swKernelOp->hasAttr(vpux::VPUIP::weightsPtrsPerClusterAttr)) {
        return weightsPerClusterPtrs;
    }

    return parseIntArrayAttr<int64_t>(
            swKernelOp->getAttrOfType<mlir::ArrayAttr>(vpux::VPUIP::weightsPtrsPerClusterAttr));
}

void updatePopulateWeightTableSwKernel(VPUIP::SwKernelOp swKernelOp, int64_t currOffset, Logger log) {
    log.trace("Update offsets for SwKernel Op at '{0}'", swKernelOp);

    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    for (auto entry : swKernelRun | indexed) {
        auto attrs = entry.value().getAttrs().value();
        SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
        const auto newOffset = newAttrs[0].cast<mlir::IntegerAttr>().getInt() + currOffset;
        newAttrs[0] = getIntAttr(swKernelOp->getContext(), newOffset);
        entry.value().setAttrsAttr(mlir::ArrayAttr::get(swKernelOp->getContext(), newAttrs));
        log.trace("Updated base offset to {0}", newOffset);
    }
    log.trace("update offsets for SwKernel Op at '{0}' {1}", swKernelOp, currOffset);
}

// Return all tensor types of SwKernelOp that will be tiled
SmallVector<vpux::NDTypeInterface> getSwKernelTiledTypes(VPUIP::SwKernelOp swKernelOp, Dim tileDim) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "topk") {
        // For SW TopK, input, output and target shape will be tiled
        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto auxType = swKernelOp->getOperand(1).getType();
        const auto outputType = swKernelOp->getResult(0).getType();
        const auto targetShapeType = swKernelOp->getResult(1).getType();
        return {inputType, auxType, outputType, targetShapeType};
    } else if (kernelEntryName == "gather") {
        auto args = kernelArgsRange(swKernelOp);
        const auto kernelAxisAttr = args.begin()[0].dyn_cast<mlir::IntegerAttr>();
        VPUX_THROW_UNLESS(kernelAxisAttr != nullptr, "Failed to extract axis at '{0}'", swKernelOp->getLoc());
        const auto kernelAxis = kernelAxisAttr.getValue().getSExtValue();
        const auto axisVal = convertKernelAxisToOrigAxis(swKernelOp.getInputs()[0], kernelAxis);

        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto indicesType = swKernelOp->getOperand(1).getType();
        const auto outputType = swKernelOp->getResult(0).getType();

        const auto tileDimVal = static_cast<int64_t>(tileDim.ind());

        if (tileDimVal == axisVal) {
            return {indicesType, outputType};
        }

        return {inputType, outputType};
    } else if (kernelEntryName == "gru_sequence") {
        // For SW GRUSequence, inputData, initialHiddenState and outputs will be tiled
        const auto inputDataType = swKernelOp->getOperand(0).getType();
        const auto initialHiddenStateType = swKernelOp->getOperand(1).getType();
        const auto outputYType = swKernelOp->getResult(0).getType();
        const auto outputHoType = swKernelOp->getResult(1).getType();
        return {inputDataType, initialHiddenStateType, outputYType, outputHoType};
    } else if (kernelEntryName == "gru_sequence_last_part") {
        // For SW GRUSequenceLastPart, inputData, initialHiddenState and outputs will be tiled
        const auto inputDataType = swKernelOp->getOperand(0).getType();
        const auto initialHiddenStateType = swKernelOp->getOperand(1).getType();
        const auto outputYType = swKernelOp->getResult(0).getType();
        const auto outputHoType = swKernelOp->getResult(1).getType();
        return {inputDataType, initialHiddenStateType, outputYType, outputHoType};
    } else if (kernelEntryName == "accumulate") {
        const auto lhsType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto rhsType = swKernelOp->getOperand(1).getType().cast<vpux::NDTypeInterface>();
        const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

        SmallVector<vpux::NDTypeInterface> tiledTypes = {lhsType, rhsType};

        const auto lhsScale = swKernelOp->getOperand(2);
        if (lhsScale != nullptr) {
            const auto lhsScaleType = lhsScale.getType().cast<vpux::NDTypeInterface>();

            // lhs Scale is broadcasted on tile axis
            if (lhsScaleType.getShape()[tileDim] != 1) {
                tiledTypes.push_back(lhsScaleType);
            }
        }

        const auto rhsScale = swKernelOp->getOperand(3);
        if (rhsScale != nullptr) {
            const auto rhsScaleType = rhsScale.getType().cast<vpux::NDTypeInterface>();

            // rhs Scale is broadcasted on tile axis
            if (rhsScaleType.getShape()[tileDim] != 1) {
                tiledTypes.push_back(rhsScaleType);
            }
        }

        tiledTypes.push_back(outputType);
        return tiledTypes;
    } else if (kernelEntryName == "eltwise_mul" || kernelEntryName == "eltwise_power" ||
               kernelEntryName == "eltwise_div" || kernelEntryName == "prelu_fp16" ||
               kernelEntryName == "eltwise_greater" || kernelEntryName == "eltwise_less" ||
               kernelEntryName == "eltwise_sub" || kernelEntryName == "eltwise_add" ||
               kernelEntryName == "eltwise_select") {
        // For SW Eltwise Op with multi inputs
        // Only the input which does not need broadcast and output will be tiled
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const auto outputShape = outputType.getShape();
        for (auto input : swKernelOp->getOperands()) {
            const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
            const auto inputShape = inputType.getShape();
            if (inputShape == outputShape) {
                tiledTypes.push_back(inputType);
            }
        }
        tiledTypes.push_back(outputType);

        return tiledTypes;
    } else if (kernelEntryName == "mvn6") {
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        // Optional scale/bias with broadcast on 'tileDim' are not tiled
        for (auto input : swKernelOp->getOperands()) {
            const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
            const auto inputShape = inputType.getShape();
            if (inputShape[tileDim] != 1) {
                tiledTypes.push_back(inputType);
            }
        }
        const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        tiledTypes.push_back(outputType);
        return tiledTypes;
    } else if (kernelEntryName == "mvn1_norm") {
        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto outputType = swKernelOp->getResult(0).getType();

        auto args = kernelArgsRange(swKernelOp);
        const auto isAcrossChannelsAttr = args.begin()[0].dyn_cast<mlir::BoolAttr>();
        VPUX_THROW_UNLESS(isAcrossChannelsAttr != nullptr, "Failed to extract AcrossChannelsAttr at '{0}'",
                          swKernelOp->getLoc());
        const bool isAcrossChannels = isAcrossChannelsAttr.getValue();

        if (tileDim == Dims4D::Act::C && !isAcrossChannels) {
            const auto meanVarType = swKernelOp->getOperand(1).getType();
            return {inputType, meanVarType, outputType};
        }

        return {inputType, outputType};
    } else {
        // By default, all inputs and outputs will be tiled
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        for (const auto& input : swKernelOp->getOperands()) {
            const auto inputType = input.getType();
            tiledTypes.push_back(inputType);
        }
        for (const auto& output : swKernelOp->getResults()) {
            const auto outputType = output.getType();
            tiledTypes.push_back(outputType);
        }
        return tiledTypes;
    }
}

bool isCacheOpTaskType(mlir::SymbolRefAttr kernelTaskType, bool includePrefetch) {
    if (!kernelTaskType) {
        return false;
    }
    auto taskTypeVal = VPU::symbolizeActShaveTaskType(kernelTaskType.getLeafReference().strref());
    VPUX_THROW_UNLESS(taskTypeVal.has_value(), "VPU::ActShaveTaskType has no value.");
    std::unordered_set<VPU::ActShaveTaskType> actShaveCacheOpTypes = {VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE,
                                                                      VPU::ActShaveTaskType::CACHE_FLUSH,
                                                                      VPU::ActShaveTaskType::CACHE_INVALIDATE};
    if (includePrefetch) {
        actShaveCacheOpTypes.insert(VPU::ActShaveTaskType::CACHE_PREFETCH);
    }

    return actShaveCacheOpTypes.count(taskTypeVal.value()) > 0;
}

bool isCacheOpTaskType(std::optional<::mlir::SymbolRefAttr> kernelTaskType, bool includePrefetch) {
    return kernelTaskType.has_value() ? isCacheOpTaskType(kernelTaskType.value(), includePrefetch) : false;
}

bool isCacheHandlingOp(VPUIP::SwKernelOp swKernelOp) {
    auto moduleOp = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    if (kernelFunc == nullptr) {
        return false;
    }

    auto kernelTaskType = kernelFunc->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");
    if (kernelTaskType == nullptr) {
        return false;
    }

    return isCacheOpTaskType(kernelTaskType);
}

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers) {
    mlir::SmallVector<mlir::Value> ddrBuffers;
    llvm::copy(buffers | vpux::filtered([](mlir::Value buffer) {
                   auto bufferType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
                   return bufferType.getMemoryKind() == VPU::MemoryKind::DDR;
               }),
               std::back_inserter(ddrBuffers));

    return ddrBuffers;
}

bool hasInputsInDDR(VPUIP::SwKernelOp swKernelTask) {
    return llvm::any_of(swKernelTask.getInputs(), [](mlir::Value buffer) {
        auto bufferType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
        if (bufferType.getMemoryKind() == VPU::MemoryKind::DDR) {
            return true;
        }
        return false;
    });
}

int64_t getSwKernelTilingAddressAlignment(VPUIP::SwKernelOp swkernelOp, VPU::ArchKind arch) {
    if (arch == VPU::ArchKind::NPU37XX) {
        return 1;
    }

    auto name = getSwKernelEntryName(swkernelOp);
    if (llvm::find(SW_KERNELS_NEED_TILING_ALIGNMENT, name) == SW_KERNELS_NEED_TILING_ALIGNMENT.end()) {
        return 1;
    }
    return NPU40XX_SW_KERNEL_ADDRESS_ALIGNMENT;
}
}  // namespace VPUIP
}  // namespace vpux
