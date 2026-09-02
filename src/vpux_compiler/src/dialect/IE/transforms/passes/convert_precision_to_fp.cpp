//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/dialect/net/utils/precision_info_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <optional>
#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPRECISIONTOFP
#define GEN_PASS_DEF_CONVERTPRECISIONTOFP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

namespace {

// E#160872: The compiler must have a much more general solution: either all
// fp16 values must come out as HALF_MAX / HALF_MIN or none. the preferred way
// seems to be "all". However, clamping non-splats produced inaccurate results
// according to tests. This needs to be debugged properly to understand why. The
// logic here is a *workaround* to have sustainable development in the meantime.
Const::ContentAttr clampF16Splat(const Const::ContentAttr& input) {
    if (!input.isSplat()) {
        // Note: clamping values in the pass here is OK, because this logic only
        // works for *splats* (aka single-value constants) and calculating
        // splats is *fast*. If there is ever a need for a proper solution, this
        // has to be migrated to a constant transformation (likely, to
        // #const.ConvertElemType<f16>).
        return nullptr;
    }

    // Note: fp16 internal ctor would convert any type to float thus get float
    // value from the input directly.
    const auto splatValue = input.fold().getSplatValue<float>();
    const auto splatValueFp16 = vpux::type::float16(splatValue);
    if (!std::isinf(splatValueFp16)) {
        // no need to clamp - can use the existing content attr
        return nullptr;
    }
    const auto clampedFp16 = std::numeric_limits<vpux::type::float16>::clamp(splatValueFp16);

    const auto dataType = input.getType().changeElemType(mlir::Float16Type::get(input.getContext()));
    const auto content = Const::createConstContent(mlir::cast<mlir::ShapedType>(dataType), ArrayRef{clampedFp16});

    // Note: Cast to current input type to ensure compatibility with existing
    // constant operation. This cast costs nothing and is likely to be fused
    // with other casts.
    return Const::ContentAttr::get(content, {Const::CastElemTypeAttr::get(input.getType().getElementType())});
}

void insertEpsilonClamp(mlir::OpBuilder& builder, mlir::Operation* divLikeOp, mlir::Operation* op) {
    // Return if there is a subsequent Select designed to screen out Inf/NaN,
    // possibly through a chain of ConvertOps between divide and Select.
    if (divLikeOp != nullptr && divLikeOp->hasOneUse()) {
        mlir::OpOperand* dataOperand = &*divLikeOp->getUses().begin();
        // Walk through any number of ConvertOps to find the Select
        auto* user = dataOperand->getOwner();
        while (mlir::isa_and_nonnull<IE::ConvertOp>(user) && user->hasOneUse()) {
            dataOperand = &*user->getUses().begin();
            user = dataOperand->getOwner();
        }
        if (auto selOp = mlir::dyn_cast_or_null<IE::SelectOp>(dataOperand->getOwner())) {
            auto dataIndex = dataOperand->getOperandNumber();
            auto maybeZeros = dataIndex == 1 ? selOp.getInput3().getDefiningOp<Const::DeclareOp>()
                                             : selOp.getInput2().getDefiningOp<Const::DeclareOp>();
            if (dataIndex > 0 && maybeZeros != nullptr && Const::hasAllZeroValues(maybeZeros.getContent())) {
                return;
            }
        }
    }

    if (auto sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(op)) {
        // Skip over Sqrt Op since epsilon Add/Max/Clamp Op is usually located before it.
        op = sqrtOp.getInput().getDefiningOp();
    }

    if (op == nullptr || !op->hasOneUse() || mlir::isa_and_nonnull<Const::DeclareOp>(op)) {
        return;
    }

    if (auto inputVal = op->getOperands().front()) {
        if (inputVal.getDefiningOp<Const::DeclareOp>() != nullptr) {
            return;
        }
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    if (!outputType.getElementType().isF16()) {
        return;
    }

    // Insert a new Clamp op to prevent float16 precision divide-by-zero in subsequent Op.
    builder.setInsertionPointAfter(op);
    auto& useResult = *op->getUses().begin();

    auto ctx = op->getContext();
    auto clampMin = getFPAttr(ctx, std::numeric_limits<type::float16>::nearest_non_zero_positive_value);
    auto clampMax = getFPAttr(ctx, static_cast<double>(std::numeric_limits<type::float16>::max()));
    auto newClampOp = builder.create<IE::ClampOp>(takeOpLoc(op, "epsilon"), op->getResult(0), clampMin, clampMax);

    useResult.assign(newClampOp.getResult());
}

// Check if the given AddOp output feeds into an IE.RMSOp (fused RMSNorm).
bool isAddBeforeRMSNorm(IE::AddOp addOp) {
    const auto addOutput = addOp.getOutput();

    for (auto* user : addOp->getUsers()) {
        auto rmsOp = mlir::dyn_cast_or_null<IE::RMSOp>(user);
        if (rmsOp != nullptr && rmsOp.getInput() == addOutput) {
            return true;
        }
    }

    return false;
}

// Keeps the scales operand of a scales-as-parameter Interpolate at f32 through
// the precision conversion instead of round-tripping it via f16.
//
// Why f32 matters: InterpolateDMA mirrors OpenVINO's epsilon-tolerant shape
// inference outShape = floor((scale + 1e-6) * inShape); an fp16 scale loses
// the epsilon (e.g. fp16(1.1) * 320 = 351.87 -> 351 instead of 352).
//
// Preservation only applies when the scales root at an originally-f32 network
// input whose forward uses are exclusively Interpolate scales operands, possibly
// relayed through func.call chains (e.g. after outlining). Every func arg and
// call operand on that chain stays f32; any non-forwarding, non-Interpolate use
// disqualifies the whole chain.
//
// Collection records, per func, the preserved arg indices and, per Interpolate/
// CallOp, the preserved operand indices. These feed:
//   - PreserveArgType / PreserveOperand callbacks driving the rewrite,
//   - getInterpolateLegality / getCallLegality / getFuncSignatureLegality driving
//     dynamic legality.
// Legality is keyed structurally (operand precision + callee symbol, not op
// pointers), so it stays correct for the clones the dialect-conversion driver
// produces, and the f32 scales is never materialized as f16.
// Preserver functionality can be optimized further - TODO: E#221311
class InterpolateScalesPreserver {
public:
    InterpolateScalesPreserver(mlir::ModuleOp module, Logger log): _log(log) {
        collectPreservedScales(module);
    }

    // Original (f32) type for a preserved scales argument; nullptr otherwise.
    // Consumed by the function-signature conversion (PreserveArgTypeCb).
    mlir::Type getPreservedArgType(mlir::func::FuncOp funcOp, unsigned argIdx) const {
        const auto it = _preservedArgIndices.find(funcOp.getOperation());
        if (it != _preservedArgIndices.end() && it->second.contains(argIdx)) {
            return funcOp.getFunctionType().getInput(argIdx);
        }
        return nullptr;
    }

    // Whether the operand is the preserved f32 scales of a SAP Interpolate.
    // Consumed by the operand conversion (PreserveOperandCb).
    bool isPreservedScalesOperand(mlir::Operation* op, unsigned operandIdx) const {
        const auto it = _preservedScalesOperandIdx.find(op);
        return it != _preservedScalesOperandIdx.end() && it->second.contains(operandIdx);
    }

    // Legality of an Interpolate during conversion: nullopt when its scales is not a
    // preserved f32 block argument (caller defers to the default type-legality check),
    // otherwise legal once data and output are f16 with scales kept f32.
    std::optional<bool> getInterpolateLegality(IE::InterpolateOp interpOp) const {
        const auto scalesVal = interpOp.getScales();
        // Interpolate may carry scales as an attribute instead of an operand, leaving getScales() null.
        // Such ops are never preserved, so defer to the default type-legality check.
        if (scalesVal == nullptr) {
            return std::nullopt;
        }
        const auto scalesElem = mlir::cast<NDTypeInterface>(scalesVal.getType()).getElementType();
        // A preserved scales is the only f32 block-argument scales that survives
        if (!scalesElem.isF32() || !mlir::isa<mlir::BlockArgument>(scalesVal)) {
            return std::nullopt;
        }
        const auto inElem = mlir::cast<NDTypeInterface>(interpOp.getInput().getType()).getElementType();
        const auto outElem = mlir::cast<NDTypeInterface>(interpOp.getOutput().getType()).getElementType();
        return inElem.isF16() && outElem.isF16();
    }

    // Legality of a CallOp forwarding preserved f32 scales: defers to the default
    // type-legality check when the callee has no preserved argument; otherwise legal
    // once every operand feeding a preserved callee argument stays f32 and every
    // other operand/result is converted.
    bool getCallLegality(mlir::func::CallOp callOp, const mlir::TypeConverter& typeConverter) const {
        auto module = callOp->getParentOfType<mlir::ModuleOp>();
        auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
        VPUX_THROW_WHEN(callee == nullptr, "CallOp callee '{0}' not found in module", callOp.getCallee());
        const auto entry = _preservedArgIndices.find(callee.getOperation());
        if (entry == _preservedArgIndices.end()) {
            return typeConverter.isLegal(callOp);
        }
        const auto& preservedIndices = entry->second;
        for (const auto& operand : llvm::enumerate(callOp.getOperands())) {
            const auto operandIdx = static_cast<unsigned>(operand.index());
            const auto operandType = operand.value().getType();
            if (preservedIndices.contains(operandIdx)) {
                const auto ndType = mlir::dyn_cast<NDTypeInterface>(operandType);
                if (ndType == nullptr || !ndType.getElementType().isF32()) {
                    return false;
                }
            } else if (!typeConverter.isLegal(operandType)) {
                return false;
            }
        }
        return llvm::all_of(callOp.getResultTypes(), [&](mlir::Type resultType) {
            return typeConverter.isLegal(resultType);
        });
    }

    // Legality of a func during conversion: defers to the default signature legality
    // when it has no preserved args; otherwise legal once preserved inputs stay f32
    // and every other input/result is converted.
    bool getFuncSignatureLegality(mlir::func::FuncOp funcOp, const mlir::TypeConverter& typeConverter) const {
        const auto entry = _preservedArgIndices.find(funcOp.getOperation());
        if (entry == _preservedArgIndices.end()) {
            return typeConverter.isSignatureLegal(funcOp.getFunctionType());
        }
        const auto& preservedIndices = entry->second;
        const auto funcType = funcOp.getFunctionType();
        for (const auto& [funcInputIndex, funcInputType] : llvm::enumerate(funcType.getInputs())) {
            if (preservedIndices.contains(static_cast<unsigned>(funcInputIndex))) {
                const auto ndType = mlir::dyn_cast<NDTypeInterface>(funcInputType);
                if (ndType == nullptr || !ndType.getElementType().isF32()) {
                    return false;
                }
            } else if (!typeConverter.isLegal(funcInputType)) {
                return false;
            }
        }
        return llvm::all_of(funcType.getResults(), [&](mlir::Type resultType) {
            return typeConverter.isLegal(resultType);
        });
    }

private:
    using ArgIndexMap = mlir::DenseMap<mlir::Operation*, llvm::SmallDenseSet<unsigned>>;
    using OperandIndexMap = mlir::DenseMap<mlir::Operation*, llvm::SmallDenseSet<unsigned>>;
    using VisitedSet = llvm::DenseSet<std::pair<mlir::Operation*, unsigned>>;

    // For each originally-f32 entry input, trace its forward uses and commit
    // the marks all-or-nothing.
    void collectPreservedScales(mlir::ModuleOp module) {
        // Cheap guard: inspect the network signature only when at least one SAP
        // Interpolate reads its scales from a block argument
        bool hasInterpolateSAP = false;
        module.walk([&](IE::InterpolateOp interpOp) {
            const auto scales = interpOp.getScales();
            if (IE::isScalesAsParameter(scales, interpOp.getScalesAttr()) &&
                mlir::isa_and_nonnull<mlir::BlockArgument>(scales)) {
                hasInterpolateSAP = true;
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
        });
        if (!hasInterpolateSAP) {
            return;
        }

        auto [netInfo, netFunc] = net::getFromModule(module);
        auto inputsInfo = netInfo.getInputsDataInfo();
        for (unsigned inputIdx = 0; inputIdx < netFunc.getNumArguments(); ++inputIdx) {
            VPUX_THROW_UNLESS(inputIdx < inputsInfo.size(), "Input index {0} out of range {1}", inputIdx,
                              inputsInfo.size());
            const auto userType = mlir::cast<vpux::NDTypeInterface>(inputsInfo[inputIdx].getUserType());
            if (!userType.getElementType().isF32()) {
                continue;
            }
            // Stage the marks; commit only when every forward use qualifies.
            ArgIndexMap argMarks;
            OperandIndexMap operandMarks;
            VisitedSet visited;
            if (tryTraceScalesCone(netFunc, inputIdx, module, visited, argMarks, operandMarks)) {
                for (const auto& [op, indices] : argMarks) {
                    _preservedArgIndices[op].insert(indices.begin(), indices.end());
                }
                for (const auto& [op, indices] : operandMarks) {
                    _preservedScalesOperandIdx[op].insert(indices.begin(), indices.end());
                }
            } else {
                _log.trace("Network input #{0} feeds a non-Interpolate consumer; leaving as f16", inputIdx);
            }
        }
    }

    // Forward DFS over uses: every use must be a SAP Interpolate scales operand
    // or a func.call forwarding into another preservable callee argument; any
    // other consumer fails the trace.
    bool tryTraceScalesCone(mlir::func::FuncOp funcOp, unsigned argIdx, mlir::ModuleOp module, VisitedSet& visited,
                            ArgIndexMap& argMarks, OperandIndexMap& operandMarks) const {
        if (!visited.insert({funcOp.getOperation(), argIdx}).second) {
            return true;
        }
        const auto arg = funcOp.getArgument(argIdx);
        // An argument with no uses should not be preserved,
        // it is not target for preserver.
        if (arg.use_empty()) {
            return false;
        }
        for (mlir::OpOperand& use : arg.getUses()) {
            auto* owner = use.getOwner();
            const auto operandIdx = use.getOperandNumber();
            if (auto interpOp = mlir::dyn_cast<IE::InterpolateOp>(owner)) {
                if (interpOp.getScales() == arg &&
                    IE::isScalesAsParameter(interpOp.getScales(), interpOp.getScalesAttr())) {
                    operandMarks[interpOp.getOperation()].insert(operandIdx);
                    continue;
                }
                return false;
            }
            if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(owner)) {
                auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                if (callee == nullptr || callee.isExternal() || operandIdx >= callee.getNumArguments()) {
                    return false;
                }
                if (!tryTraceScalesCone(callee, operandIdx, module, visited, argMarks, operandMarks)) {
                    return false;
                }
                // Staged in local maps; caller discards the whole batch if any use fails.
                operandMarks[callOp.getOperation()].insert(operandIdx);
                continue;
            }
            return false;
        }
        argMarks[funcOp.getOperation()].insert(argIdx);
        return true;
    }

    Logger _log;
    ArgIndexMap _preservedArgIndices;
    OperandIndexMap _preservedScalesOperandIdx;
};

bool isReduceSquareCoreOp(mlir::Operation* op) {
    return mlir::isa<IE::PowerOp, IE::ReduceSumOp, IE::SqrtOp>(op);
}

bool hasReduceSquareCoreProducer(mlir::Operation* producer) {
    if (producer != nullptr && isReduceSquareCoreOp(producer)) {
        return true;
    }

    auto prevConvert = mlir::dyn_cast_or_null<IE::ConvertOp>(producer);
    if (prevConvert == nullptr) {
        return false;
    }

    return isReduceSquareCoreOp(prevConvert.getInput().getDefiningOp());
}

bool hasReduceSquareCoreUser(mlir::Value output) {
    for (auto* user : output.getUsers()) {
        if (isReduceSquareCoreOp(user)) {
            return true;
        }

        auto nextConvert = mlir::dyn_cast<IE::ConvertOp>(user);
        if (nextConvert == nullptr) {
            continue;
        }

        if (llvm::any_of(nextConvert.getOutput().getUsers(), [](mlir::Operation* nextUser) {
                return isReduceSquareCoreOp(nextUser);
            })) {
            return true;
        }
    }

    return false;
}

bool isConvertConnectedToReduceSquare(IE::ConvertOp op) {
    return hasReduceSquareCoreProducer(op.getInput().getDefiningOp()) || hasReduceSquareCoreUser(op.getOutput());
}

//
// ConvertPrecisionToFPPass
//

class ConvertPrecisionToFPPass final : public IE::impl::ConvertPrecisionToFPBase<ConvertPrecisionToFPPass> {
public:
    explicit ConvertPrecisionToFPPass(Logger log, StringRef computeLayersWithHigherPrecision)
            : _computeLayersWithHigherPrecision(computeLayersWithHigherPrecision.str()) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    std::string _computeLayersWithHigherPrecision;
};

mlir::LogicalResult ConvertPrecisionToFPPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (computeLayersWithHigherPrecision.hasValue()) {
        _computeLayersWithHigherPrecision = computeLayersWithHigherPrecision.getValue();
    }

    return mlir::success();
}

void ConvertPrecisionToFPPass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto convertElemType = [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isF64() || elemType.isSignlessInteger(8)) {
            return mlir::Float16Type::get(elemType.getContext());
        } else if (const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType);
                   qType != nullptr && qType.getExpressedType().isF32()) {
            return changeExpressedType(qType, mlir::Float16Type::get(qType.getContext()));
        } else {
            return elemType;
        }
    };

    mlir::TypeConverter fp16TypeConverter;
    setupConvertPrecision(fp16TypeConverter, convertElemType);
    auto module = getOperation();
    auto rtPrecision = net::PrecisionSensitiveOps{module, _log};

    // Keeps scales-as-parameter Interpolate inputs at f32 across the conversion.
    // Owns the function arguments and operands to preserve, plus the legality
    // predicates that protect them.
    const InterpolateScalesPreserver scalesPreserver{module, _log};

    const auto isLegalOp = [&](mlir::Operation* op) {
        if (rtPrecision.isPrecisionSensitiveOp(op)) {
            return true;
        }
        if (auto interpOp = mlir::dyn_cast<IE::InterpolateOp>(op)) {
            if (const auto legal = scalesPreserver.getInterpolateLegality(interpOp)) {
                return *legal;
            }
        }
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            return scalesPreserver.getCallLegality(callOp, fp16TypeConverter);
        }
        return fp16TypeConverter.isLegal(op);
    };

    const auto hasDynamicDequantizeUser = [](mlir::Operation* op) {
        return llvm::any_of(op->getUsers(), [](const auto user) {
            return mlir::isa<IE::DynamicDequantizeOp>(user);
        });
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::OneHotOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<IE::DynamicQuantizeOp>();
    target.addLegalOp<IE::DynamicDequantizeOp>();
    target.addDynamicallyLegalOp<IE::QuantizeCastOp>([&](mlir::Operation* op) {
        return isLegalOp(op) || hasDynamicDequantizeUser(op);
    });
    target.addDynamicallyLegalOp<IE::QuantizeOp>([&](mlir::Operation* op) {
        return isLegalOp(op) || hasDynamicDequantizeUser(op);
    });
    target.addLegalOp<IE::IfOp>();
    target.addLegalOp<IE::YieldOp>();
    target.addLegalOp<IE::LoopSelectOp>();
    target.addLegalOp<IE::EqualOp>();
    target.addLegalOp<IE::LessOp>();
    target.addLegalOp<IE::LessEqualOp>();
    target.addLegalOp<IE::GreaterOp>();
    target.addLegalOp<IE::NotEqualOp>();
    target.addLegalOp<IE::IsInfOp>();
    target.addLegalOp<IE::IsFiniteOp>();
    // AssignOp & ReadValueOp represent inputs/outputs. Cannot convert their type internally.
    target.addLegalOp<IE::AssignOp>();
    target.addLegalOp<IE::ReadValueOp>();
    target.addLegalOp<IE::BitwiseAndOp>();
    target.addLegalOp<IE::BitwiseOrOp>();
    target.addLegalOp<IE::BitwiseXorOp>();
    target.addLegalOp<IE::BitwiseNotOp>();
    target.addLegalOp<IE::BitwiseRightShiftOp>();
    target.addLegalOp<IE::BitwiseLeftShiftOp>();
    target.addLegalOp<IE::RangeOp>();
    target.addLegalOp<IE::ReduceL2Op>();
    target.addLegalOp<IE::InverseOp>();
    target.addLegalOp<IE::SelectOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return scalesPreserver.getFuncSignatureLegality(funcOp, fp16TypeConverter);
    });

    SmallVector<mlir::OperationName> highPrecisionOps;
    const auto internalHintPrefix = "internal_";
    auto internalMvnHighNorm = false;
    auto hasReduceSquareHigherPrecision = false;

    // Supported values for compute-layers-with-higher-precision:
    //   "Add"               — keep ALL Add ops in f32
    //   "Add_RMSNorm"       — keep only the epsilon Add inside RMS decomposition
    //                         (ReduceMean -> Add -> Sqrt)
    //   "Add_BeforeRMSNorm" — keep only the residual Add whose output feeds into
    //                         IE.RMSOp. Should be used together with "RMS" to
    //                         ensure the full Add -> RMS path stays in f32.
    //   "ReduceSquare"      — keep Power, ReduceSum and Sqrt in the
    //                         Power(x,2) -> ReduceSum -> Sqrt pattern in f32

    if (!_computeLayersWithHigherPrecision.empty()) {
        std::istringstream optionsStream(_computeLayersWithHigherPrecision);
        std::string dialectNamespace = IE::IEDialect::getDialectNamespace().str() + ".";
        const std::string addRMSNormOption = "Add_RMSNorm";
        const std::string addBeforeRMSNormOption = "Add_BeforeRMSNorm";
        const std::string reduceSquareOption = "ReduceSquare";
        std::string option;
        while (std::getline(optionsStream, option, ',')) {
            bool isAddRMSNorm = option == addRMSNormOption;
            bool isAddBeforeRMSNormOption = option == addBeforeRMSNormOption;
            bool isReduceSquare = option == reduceSquareOption;
            // Both Add_RMSNorm and Add_BeforeRMSNorm are sub-options of "Add" op,
            // so map them to "Add" for opname validation
            if (isAddRMSNorm || isAddBeforeRMSNormOption) {
                option = std::string("Add");
            }
            // ReduceSquare is a virtual op name — the actual pattern is Power+ReduceSum+Sqrt
            if (isReduceSquare) {
                option = std::string("Power");
            }

            if (option.find(internalHintPrefix, 0) != std::string::npos) {
                const auto internalMvnOption = std::string(internalHintPrefix) + "MvnNormalize";
                if (option.find(internalMvnOption, 0) != std::string::npos) {
                    internalMvnHighNorm = true;
                }
                continue;
            }

            std::string fullOption = dialectNamespace + option;
            StringRef opnameRef(fullOption);
            auto opname = mlir::OperationName(opnameRef, &ctx);
            VPUX_THROW_UNLESS(opname.isRegistered(), "Invalid input layer '{0}'", opname);
            // Keep the original precision for all instances of specified layer name(s) during the conversion to FP16

            // If AddOp is listed into computeLayersWithHigherPrecision list,
            // keep precision only in RMSNorm pattern
            if (isAddRMSNorm) {
                target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) {
                    // Try to find RMSNorm pattern
                    // ReaduceMeanOp -> AddOp -> SqrtOp
                    if ((op.getInput1().getDefiningOp<IE::ReduceMeanOp>() != nullptr ||
                         op.getInput2().getDefiningOp<IE::ReduceMeanOp>() != nullptr) &&
                        mlir::isa_and_nonnull<IE::SqrtOp>(*op.getOutput().getUsers().begin())) {
                        return true;
                    }
                    return isLegalOp(op);
                });
            } else if (isAddBeforeRMSNormOption) {
                // Add_BeforeRMSNorm: keep the residual Add whose output feeds into RMS
                target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) {
                    if (isAddBeforeRMSNorm(op)) {
                        return true;
                    }
                    return isLegalOp(op);
                });
            } else if (isReduceSquare) {
                hasReduceSquareHigherPrecision = true;
                // ReduceSquare pattern: Power(x,2) -> ReduceSum -> Sqrt
                // Keep all three ops at higher precision
                target.addDynamicallyLegalOp<IE::PowerOp>([&](IE::PowerOp op) {
                    auto constOp = op.getInput2().getDefiningOp<Const::DeclareOp>();
                    if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
                        return isLegalOp(op);
                    }
                    if (constOp.getContent().getSplatValue<double>() != 2.0) {
                        return isLegalOp(op);
                    }
                    if (!op->hasOneUse() || !mlir::isa<IE::ReduceSumOp>(*op->getUsers().begin())) {
                        return isLegalOp(op);
                    }
                    return true;
                });
                target.addDynamicallyLegalOp<IE::ReduceSumOp>([&](IE::ReduceSumOp op) {
                    auto inputOp = op.getInput().getDefiningOp<IE::PowerOp>();
                    if (inputOp == nullptr || !inputOp->hasOneUse()) {
                        return isLegalOp(op);
                    }
                    auto constOp = inputOp.getInput2().getDefiningOp<Const::DeclareOp>();
                    if (constOp == nullptr || !constOp.getContentAttr().isSplat() ||
                        constOp.getContent().getSplatValue<double>() != 2.0) {
                        return isLegalOp(op);
                    }
                    if (!op->hasOneUse() || !mlir::isa<IE::SqrtOp>(*op->getUsers().begin())) {
                        return isLegalOp(op);
                    }
                    return true;
                });
                target.addDynamicallyLegalOp<IE::SqrtOp>([&](IE::SqrtOp op) {
                    auto reduceSumOp = op.getInput().getDefiningOp<IE::ReduceSumOp>();
                    if (reduceSumOp == nullptr || !reduceSumOp->hasOneUse()) {
                        return isLegalOp(op);
                    }
                    auto inputOp = reduceSumOp.getInput().getDefiningOp<IE::PowerOp>();
                    if (inputOp == nullptr || !inputOp->hasOneUse()) {
                        return isLegalOp(op);
                    }
                    auto constOp = inputOp.getInput2().getDefiningOp<Const::DeclareOp>();
                    if (constOp == nullptr || !constOp.getContentAttr().isSplat() ||
                        constOp.getContent().getSplatValue<double>() != 2.0) {
                        return isLegalOp(op);
                    }
                    return true;
                });

                // if any of these remain in FP64, convert them to FP32.
                highPrecisionOps.push_back(mlir::OperationName(IE::PowerOp::getOperationName(), &ctx));
                highPrecisionOps.push_back(mlir::OperationName(IE::ReduceSumOp::getOperationName(), &ctx));
                highPrecisionOps.push_back(mlir::OperationName(IE::SqrtOp::getOperationName(), &ctx));
            } else {
                target.addLegalOp(opname);
                highPrecisionOps.push_back(opname);
            }
        }
    }

    // E#160869: Splat floating-point constants must be clamped, not converted,
    // to ensure accurate results (e.g. when comparing to CPU inference).
    module.walk([&](Const::DeclareOp op) {
        const auto inElemType = op.getContentAttr().getType().getElementType();
        const auto precisionLoweringToF16 =
                mlir::isa<mlir::FloatType>(inElemType) && (inElemType.getIntOrFloatBitWidth() >= (sizeof(float) * 8));
        const auto newContentAttr = precisionLoweringToF16 ? clampF16Splat(op.getContentAttr()) : nullptr;
        if (newContentAttr != nullptr) {
            VPUX_THROW_WHEN(op.getType() != newContentAttr.getType(),
                            "Const op type {0} must match new content attribute type {1}", op.getType(),
                            newContentAttr.getType());
            op.setContentAttr(newContentAttr);
        }
    });

    // Some ops infer their output type based on a member type attribute, which should also be converted.
    module.walk([&](mlir::Operation* op) {
        if (!target.isIllegal(op)) {
            return;
        }

        mlir::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::DequantizeOp, IE::QuantizeCastOp, IE::QuantizeOp>([&](auto op) {
                    op.setDstElemType(convertElemType(op.getDstElemType()));
                })
                .Case<IE::OneHotOp, IE::RandomUniformOp, IE::EyeOp>([&](auto op) {
                    op.setOutputType(convertElemType(op.getOutputType()));
                })
                .Case<IE::MVNOp>([&](auto op) {
                    if (internalMvnHighNorm) {
                        // fp16 buffs with f32 internal processing for normalization
                        op.setHighPrecisionNormalize(true);
                    }
                })
                .Case<IE::DynamicDataMaskOp>([&](auto op) {
                    auto outTensorType = mlir::cast<NDTypeInterface>(op.getOutputTensorType());
                    op.setOutputTensorType(
                            outTensorType.changeElemType(convertElemType(outTensorType.getElementType())));
                });
    });

    const auto getPreservedArgType = [&](mlir::func::FuncOp funcOp, unsigned argIdx) {
        return scalesPreserver.getPreservedArgType(funcOp, argIdx);
    };
    const auto shouldPreserveOperand = [&](mlir::Operation* op, unsigned operandIdx) {
        return scalesPreserver.isPreservedScalesOperand(op, operandIdx);
    };

    if (mlir::failed(runConvertPrecision(module, fp16TypeConverter, target, _log, getPreservedArgType,
                                         shouldPreserveOperand))) {
        signalPassFailure();
    }

    // Since we could not support FP64, so for ops listed in computeLayersWithHigherPrecision,
    // we need to convert them from FP64 to FP32.
    if (!highPrecisionOps.empty()) {
        mlir::TypeConverter fp32TypeConverter;
        setupConvertPrecision(fp32TypeConverter, [](mlir::Type elemType) -> mlir::Type {
            if (elemType.isF64()) {
                return mlir::Float32Type::get(elemType.getContext());
            }
            return elemType;
        });
        mlir::ConversionTarget fp32Target(ctx);
        fp32Target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
            return true;
        });
        for (const auto& opname : highPrecisionOps) {
            fp32Target.addDynamicallyLegalOp(opname, [&fp32TypeConverter](mlir::Operation* op) {
                return fp32TypeConverter.isLegal(op);
            });
        }

        if (hasReduceSquareHigherPrecision) {
            // IE.Convert infers its output type from dstElemType. When FP64->FP32 conversion
            // rewrites result types in ReduceSquare path, keep dstElemType in sync.
            module.walk([&](IE::ConvertOp op) {
                if (!isConvertConnectedToReduceSquare(op)) {
                    return;
                }
                if (op.getDstElemType().isF64()) {
                    op.setDstElemType(mlir::Float32Type::get(op.getContext()));
                }
            });

            fp32Target.addDynamicallyLegalOp<IE::ConvertOp>([&](IE::ConvertOp op) {
                if (!isConvertConnectedToReduceSquare(op)) {
                    return true;
                }

                return fp32TypeConverter.isLegal(op.getOperation());
            });
        }

        // When ReduceSquare higher precision is active we have set a custom dynamic legality
        // predicate for IE::ConvertOp on fp32Target. runConvertPrecision() would unconditionally
        // call target.addLegalOp<IE::ConvertOp>(), overriding that predicate, so call
        // runConvertOpTypes() directly to preserve it.
        const auto fp32ConversionResult = hasReduceSquareHigherPrecision
                                                  ? runConvertOpTypes(module, fp32TypeConverter, fp32Target, _log)
                                                  : runConvertPrecision(module, fp32TypeConverter, fp32Target, _log);
        if (mlir::failed(fp32ConversionResult)) {
            signalPassFailure();
        }
    }

    mlir::TypeConverter uniquifyPrecisionTypeConverter;
    setupConvertPrecision(uniquifyPrecisionTypeConverter, [](mlir::Type elemType) -> mlir::Type {
        return mlir::Float16Type::get(elemType.getContext());
    });
    const auto isPrecisionUniquifiedOp = [](mlir::Operation* op) {
        auto operandTypes = op->getOperandTypes();
        return std::all_of(operandTypes.begin() + 1, operandTypes.end(), [&](auto operandType) {
            return mlir::cast<NDTypeInterface>(operandType).getElementType() ==
                   mlir::cast<NDTypeInterface>(operandTypes.front()).getElementType();
        });
    };
    mlir::ConversionTarget uniquifyPrecisionTarget(ctx);
    uniquifyPrecisionTarget.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });
    uniquifyPrecisionTarget.addDynamicallyLegalOp<IE::AndOp>(isPrecisionUniquifiedOp);
    if (mlir::failed(runConvertPrecision(module, uniquifyPrecisionTypeConverter, uniquifyPrecisionTarget, _log))) {
        signalPassFailure();
    }

    mlir::TypeConverter additionalTypeConverter;
    setupConvertPrecision(additionalTypeConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isF64() || elemType.isSignedInteger(64)) {
            return mlir::Float16Type::get(elemType.getContext());
        } else {
            return elemType;
        }
    });

    mlir::Operation* consumerOp = nullptr;
    auto walkThroughViewLikeOps = [&consumerOp](mlir::Operation* currentOp) -> mlir::FailureOr<mlir::Operation*> {
        while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface>(currentOp)) {
            if (!currentOp->hasOneUse()) {
                return mlir::failure();
            }
            consumerOp = currentOp;
            currentOp = *currentOp->getUsers().begin();
        }
        return currentOp;
    };

    const auto isLegalAdditionalOp = [&](mlir::Operation* op) {
        if (rtPrecision.isPrecisionSensitiveOp(op)) {
            return true;
        }

        if (mlir::isa_and_nonnull<IE::ClampOp>(op) && op->hasOneUse()) {
            // Not convert I64 ClampOp to FP16 if it is used as indices of GatherOp
            // TODO: E#208331 Remove check for pattern "ClampOp + GatherOp" once isPrecisionSensitiveOp supported
            consumerOp = op;
            auto implicitOpOrFailure = walkThroughViewLikeOps(*op->getUsers().begin());
            if (mlir::succeeded(implicitOpOrFailure)) {
                if (auto gatherOp = mlir::dyn_cast<IE::GatherOp>(*implicitOpOrFailure.value())) {
                    auto indicesOp = gatherOp.getIndices().getDefiningOp();
                    auto outputElemType = mlir::cast<NDTypeInterface>(op->getResult(0).getType()).getElementType();
                    if (indicesOp == consumerOp && outputElemType.isSignedInteger(64)) {
                        return true;
                    }
                }
            }
        }

        return additionalTypeConverter.isLegal(op);
    };

    mlir::ConversionTarget additionalTarget(ctx);
    additionalTarget.addDynamicallyLegalOp<IE::LessOp>(isLegalAdditionalOp);
    additionalTarget.addDynamicallyLegalOp<IE::LessEqualOp>(isLegalAdditionalOp);
    additionalTarget.addDynamicallyLegalOp<IE::GreaterOp>(isLegalAdditionalOp);
    additionalTarget.addDynamicallyLegalOp<IE::ClampOp>(isLegalAdditionalOp);
    additionalTarget.addLegalOp<mlir::ModuleOp>();
    if (mlir::failed(runConvertPrecision(module, additionalTypeConverter, additionalTarget, _log))) {
        signalPassFailure();
    }

    // SelectOp
    mlir::TypeConverter selectOpConverter;
    setupConvertPrecision(selectOpConverter, [](mlir::Type elemType) -> mlir::Type {
        return mlir::Float16Type::get(elemType.getContext());
    });

    const auto isLegalSelectOp = [&](IE::SelectOp op) {
        if (rtPrecision.isPrecisionSensitiveOp(op.getOperation())) {
            return true;
        }
        const auto dataElemType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getElementType();
        if (dataElemType.isFloat()) {
            return selectOpConverter.isLegal(op.getOperation());
        }
        return true;
    };

    mlir::ConversionTarget selectOpTarget(ctx);
    selectOpTarget.addDynamicallyLegalOp<IE::SelectOp>(isLegalSelectOp);
    selectOpTarget.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });
    if (mlir::failed(runConvertPrecision(module, selectOpConverter, selectOpTarget, _log))) {
        signalPassFailure();
    }

    mlir::OpBuilder builder(module);
    // Insert Clamp Ops to prevent divide-by-zero in low precision Divide and Divide-like Power Ops
    // to ensure accurate results (e.g. when comparing to CPU inference).
    module.walk([&](IE::DivideOp divideOp) {
        insertEpsilonClamp(builder, divideOp, divideOp.getInput2().getDefiningOp());
    });

    module.walk([&](IE::PowerOp powerOp) {
        auto exponent = IE::getExponentSplatVal(powerOp);
        const bool powerIsDivisionLike = (exponent.has_value() && exponent.value() < 0.f);
        if (!powerIsDivisionLike) {
            return;
        }

        insertEpsilonClamp(builder, powerOp, powerOp.getInput1().getDefiningOp());
    });
}

}  // namespace

//
// createConvertPrecisionToFPPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToFPPass(Logger log,
                                                                     StringRef computeLayersWithHigherPrecision) {
    return std::make_unique<ConvertPrecisionToFPPass>(log, computeLayersWithHigherPrecision);
}
