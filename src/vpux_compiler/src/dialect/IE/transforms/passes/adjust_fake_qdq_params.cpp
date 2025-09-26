//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <queue>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTFAKEQDQPARAMS
#define GEN_PASS_DEF_ADJUSTFAKEQDQPARAMS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

const float FP16_MAXIMUM = checked_cast<float>(std::numeric_limits<vpux::type::float16>::max());
const float FP16_MINIMUM = checked_cast<float>(std::numeric_limits<vpux::type::float16>::lowest());
// For the model PSD7 the script from architecture team looks at scale in QDQ layers.
// In order to match all the nodes that script modifies to achieve accuracy, we need
// to use an extra 0.6 factor.
constexpr float FP16_SCALE_FACTOR = 0.6f;
const float FP16_MIN_SCALED = FP16_SCALE_FACTOR * FP16_MINIMUM;
const float FP16_MAX_SCALED = FP16_SCALE_FACTOR * FP16_MAXIMUM;

enum class TraversalDir {
    INVALID,
    DOWN,
    UP,
};

bool isInstanceNormOp(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::MVN6Op>([&](IE::MVN6Op) {
                return true;
            })
            .Case<IE::MVNOp>([&](IE::MVNOp) {
                return true;
            })
            .Default([&](mlir::Operation*) {
                return false;
            });
}

bool isStopOperation(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::SoftMaxOp>([&](IE::SoftMaxOp) {
                return true;
            })
            .Default([&](mlir::Operation* op) {
                return isInstanceNormOp(op);
            });
}

// STEP 6: Handle Memory Ops
bool isMemoryOp(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::TransposeOp>([&](IE::TransposeOp) {
                return true;
            })
            .Case<IE::ReorderOp>([&](IE::ReorderOp) {
                return true;
            })
            .Case<IE::ReshapeOp>([&](IE::ReshapeOp) {
                return true;
            })
            .Default([&](mlir::Operation*) {
                return false;
            });
}

inline bool hasExceededFp16Range(float low, float high) {
    const auto retval = high >= FP16_MAX_SCALED || low <= FP16_MIN_SCALED;
    return retval;
}

// Trait to detect if Op is IE::QuantizeOp or IE::DequantizeOp
template <typename T>
struct is_qdq_op {
    static const bool value = false;
};

template <>
struct is_qdq_op<IE::QuantizeOp> {
    static const bool value = true;
};

template <>
struct is_qdq_op<IE::DequantizeOp> {
    static const bool value = true;
};

std::tuple<float, float, float, float> getFqValues(IE::FakeQuantizeOp fq) {
    return std::make_tuple(IE::getConst(fq.getInputLow().getDefiningOp<Const::DeclareOp>())[0],
                           IE::getConst(fq.getInputHigh().getDefiningOp<Const::DeclareOp>())[0],
                           IE::getConst(fq.getOutputLow().getDefiningOp<Const::DeclareOp>())[0],
                           IE::getConst(fq.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0]);
}

template <typename T, class = typename std::enable_if<is_qdq_op<T>::value>::type>
inline float get_qdq_scale(T& qdqop) {
    auto outputTypeQuantize = mlir::cast<mlir::ShapedType>(qdqop.getType());
    auto outElemType = outputTypeQuantize.getElementType();

    auto outUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outElemType);
    VPUX_THROW_WHEN(!outUniformType, "ERROR: Could not get uniform quant type to determine scale for QDQ OP");
    const auto quantizeScale = outUniformType.getScale();

    return quantizeScale;
}

// Computing Rescale coefficient.
inline float rescaleCoefficient(IE::QuantizeOp* op) {
    const auto fxv = get_qdq_scale(*op);
    float scale = fxv > 0.6 ? 0.5 * fxv : 1.0f;
    return scale;
}

inline float rescaleCoefficient(IE::DequantizeOp* op) {
    const auto fxv = get_qdq_scale(*op);
    float scale = fxv > 0.6 ? 0.5 * fxv : 1.0f;
    return scale;
}

inline float rescaleCoefficient(IE::FakeQuantizeOp* fakeQuantOp) {
    VPUX_THROW_WHEN(nullptr == fakeQuantOp, "ERROR: The operation parameter cannot be null");
    float scale = 1.0f;
    auto levels_opt = fakeQuantOp->getLevels();
    if (!levels_opt.has_value()) {
        return scale;
    }
    auto levels = levels_opt.value();

    // E#171515: max(inHigh-inLow, outHigh-outLow) / levels) / 0.5f)
    auto [inLow, inHigh, outLow, outHigh] = getFqValues(*fakeQuantOp);
    scale = (std::max(inHigh - inLow, outHigh - outLow) / levels) / 0.5f;
    return scale;
}

inline float rescaleCoefficient(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, float>(op)
            .Case<IE::FakeQuantizeOp>([&](IE::FakeQuantizeOp fqop) {
                return rescaleCoefficient(&fqop);
            })
            .Case<IE::QuantizeOp>([&](IE::QuantizeOp qop) {
                return rescaleCoefficient(&qop);
            })
            .Case<IE::DequantizeOp>([&](IE::DequantizeOp dqop) {
                return rescaleCoefficient(&dqop);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW("ERROR: Cannot calculate scale for an op");
                return -1.0f;
            });
}

bool isFqRangeOutOfBounds(IE::FakeQuantizeOp fqOp, float inScale = 1.0f, float outScale = 1.0f) {
    auto [inLow, inHigh, outLow, outHigh] = getFqValues(fqOp);
    return (hasExceededFp16Range(inLow * inScale, inHigh * inScale) ||
            hasExceededFp16Range(outLow * outScale, outHigh * outScale));
}

bool areFQValsEqual(IE::FakeQuantizeOp fqOp) {
    auto [inLow, inHigh, outLow, outHigh] = getFqValues(fqOp);
    return (inLow == outLow) && (inHigh == outHigh);
}

//
// FakeQdqParamsRewriter
//
class FakeQdqParamsRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQdqParamsRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/// <summary>
/// Create a multiply operation after the input "op".
/// </summary>
vpux::IE::MultiplyOp createMultiplyOp(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                      float scale) {
    VPUX_THROW_WHEN(!op, "Operation pointer cannot be null while creating multiply op");
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    auto tensorType = mlir::RankedTensorType::get({1}, mlir::Float32Type::get(ctx));
    const auto newScaleConst = Const::createFloatConst(rewriter, op->getLoc(), tensorType, {scale});

    auto multiplyOp = rewriter.create<IE::MultiplyOp>(takeOpLoc(op, "as_mul"), op->getResult(0).getType(),
                                                      op->getResult(0), newScaleConst, IE::AutoBroadcastType::NUMPY,
                                                      /*post_op=*/nullptr,
                                                      /*clamp=*/nullptr,
                                                      /*output_channels=*/nullptr,
                                                      /*input_channels=*/nullptr);
    return multiplyOp;
}

/// <summary>
/// Overloaded version of Creating a Multiply op to handle the case of creating a multiply op
/// between an input argument and a user of that argument. This is called when the defining op is null.
/// </summary>
vpux::IE::MultiplyOp createMultiplyOp(mlir::Value value, mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                      float scale) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(value);
    auto tensorType = mlir::RankedTensorType::get({1}, mlir::Float32Type::get(ctx));
    const auto newScaleConst = Const::createFloatConst(rewriter, value.getLoc(), tensorType, {scale});

    auto multiplyOp = rewriter.create<IE::MultiplyOp>(appendLoc(value.getLoc(), "as_mul"), value.getType(), value,
                                                      newScaleConst, IE::AutoBroadcastType::NUMPY,
                                                      /*post_op=*/nullptr,
                                                      /*clamp=*/nullptr,
                                                      /*output_channels=*/nullptr,
                                                      /*input_channels=*/nullptr);
    return multiplyOp;
}

struct OpParamdata {
public:
    enum class ScaleMode {
        SYMMETRIC,
        OUTPUT_ONLY,  // So far we have not found a need for INPUT_ONLY
    };

    float scale;
    mlir::Operation* op;
    TraversalDir tdir;
    ScaleMode scaleMode;

    OpParamdata(): scale(1.0f), op(nullptr), tdir(TraversalDir::INVALID), scaleMode(ScaleMode::SYMMETRIC) {
    }

    // TODO: Tech-debt, there is perhaps a better design
    OpParamdata(float os, mlir::Operation* currOp, TraversalDir travDir, ScaleMode smode = ScaleMode::SYMMETRIC)
            : scale(os), op(currOp), tdir(travDir), scaleMode(smode) {
    }
};
using SubgraphMetaData = llvm::DenseMap<mlir::Operation*, OpParamdata>;

class OpParamScalerBase {
public:
    // The logic to update an mlir::Operation goes here.
    // The expectation is the old op is replaced by a new op.
    // In any case the function updatedOp is expected to return the
    // latest version of the op for further processing.
    virtual inline std::pair<mlir::LogicalResult, bool> operator()(mlir::Operation*, mlir::PatternRewriter&,
                                                                   mlir::MLIRContext*, TraversalDir, float,
                                                                   OpParamdata::ScaleMode) {
        return {mlir::failure(), false};
    }

    // Once an is updated it needs to store the updated op
    // and return the latest version.
    virtual mlir::Operation* updatedOp() {
        return nullptr;
    }
    virtual ~OpParamScalerBase() {
    }
};

template <typename T>
class OpParamScaler : public OpParamScalerBase {
public:
    OpParamScaler(): OpParamScalerBase() {
    }
    virtual inline std::pair<mlir::LogicalResult, bool> operator()(mlir::Operation*, mlir::PatternRewriter&,
                                                                   mlir::MLIRContext*, TraversalDir, float,
                                                                   OpParamdata::ScaleMode) override {
        return {mlir::failure(), false};
    }
    virtual ~OpParamScaler() {
    }
};

template <>
class OpParamScaler<IE::FakeQuantizeOp> : public OpParamScalerBase {
    mlir::Operation* _updatedOp;

public:
    OpParamScaler<IE::FakeQuantizeOp>(): OpParamScalerBase(), _updatedOp(nullptr) {
    }
    virtual inline std::pair<mlir::LogicalResult, bool> operator()(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                                                                   mlir::MLIRContext* ctx, TraversalDir traversalDir,
                                                                   float scale, OpParamdata::ScaleMode) override;
    virtual mlir::Operation* updatedOp() override {
        return _updatedOp;
    }
    virtual ~OpParamScaler<IE::FakeQuantizeOp>() {
    }
};

inline bool canPropagateToFQDQ(mlir::Operation*, float, TraversalDir) {
    // Based on conversation with Alex, even if new scaling ruins a
    // a previously good FQ or Q, DQ op we propagate the scaling.
    return true;
#ifdef ENABLE_RESTRICTED_FQDQ_PROPAGATION
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::FakeQuantizeOp>([&](auto fqOp) {
                auto [inLow, inHigh, outLow, outHigh] = getFqValues(fqOp);
                // This is a previously good node, are we going to ruin it?
                const bool scalingBreaksFP16Range = hasExceededFp16Range(inLow * scale, inHigh * scale) ||
                                                    hasExceededFp16Range(outLow * scale, outHigh * scale);
                // This is a strict criterion, regardless of whether the node without scaling had
                // parameters within range.
                // Alternatively we could do something like
                // bool nodeGoodWithoutScaling = hasExceededFp16Range(inLow, inHigh) && !hasExceededFp16Range(outLow,
                // outHigh); return nodeGoodWithoutScaling ? !scalingBreaksFP16Range : true;
                return !scalingBreaksFP16Range;
            })
            .Case<IE::DequantizeOp>([&](auto dqOp) {
                float qscale1 = get_qdq_scale(dqOp);
                const bool scalingGoesOutOfRange = qscale1 * scale > QDQ_SCALE_MAXIMUM;
                return !scalingGoesOutOfRange;
            })
            .Case<IE::QuantizeOp>([&](auto qOp) {
                float qscale2 = get_qdq_scale(qOp);
                const bool scalingGoesOutOfRange = qscale2 * scale > QDQ_SCALE_MAXIMUM;
                return !scalingGoesOutOfRange;
            })
            .Default([&](mlir::Operation*) {
                // Not FQ or QDQ op so return true.
                return true;
            });
#endif
}

// Update FakeQuantize as outlined STEP 1 and STEP 3 of E#171489
IE::FakeQuantizeOp updateFqParams(IE::FakeQuantizeOp origFq, float scale, OpParamdata::ScaleMode scaleMode,
                                  mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPoint(origFq);
    auto [inLow, inHigh, outLow, outHigh] = getFqValues(origFq);

    float newInLow = inLow * scale;
    float newInHigh = inHigh * scale;

    // Restricting to output scaling only for quantized inputs esp. PSD7
    if (OpParamdata::ScaleMode::OUTPUT_ONLY == scaleMode) {
        newInLow = inLow;
        newInHigh = inHigh;
    }

    auto newInputLo = Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getInputLow().getType(), newInLow);
    auto newInputHi = Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getInputHigh().getType(), newInHigh);
    auto newOutputLo =
            Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getOutputLow().getType(), outLow * scale);
    auto newOutputHi =
            Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getOutputHigh().getType(), outHigh * scale);

    return rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origFq, origFq.getInput(), newInputLo, newInputHi,
                                                           newOutputLo, newOutputHi, origFq.getLevelsAttr(),
                                                           origFq.getLowFpTypeAttr(), origFq.getAutoBroadcastAttr());
}

inline std::pair<mlir::LogicalResult, bool> OpParamScaler<IE::FakeQuantizeOp>::operator()(
        mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::MLIRContext*, TraversalDir traversalDir,
        float scale, OpParamdata::ScaleMode scaleMode) {
    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        if (!canPropagateToFQDQ(op, scale, traversalDir)) {
            return {mlir::failure(), true};
        }
        _updatedOp = updateFqParams(fqOp, scale, scaleMode, rewriter);
        return {mlir::success(), true};
    }
    return {mlir::failure(), true};
}

// Check if an op is quantized const
inline bool isQuantizedConstOp(mlir::Operation* op) {
    if (!op) {
        return false;
    }
    auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(op);
    auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getContentAttr().getBaseContent().getType());
    auto elemType = cstType.getElementType();
    if (auto elemTypeInt = mlir::dyn_cast_or_null<mlir::IntegerType>(elemType)) {
        return elemTypeInt.getWidth() == 8 && elemTypeInt.isUnsigned();
    }
    return false;
}

// Metadata required to create multiply operation.
// This is created during graph traversal.
struct MulPropData {
    TraversalDir traversalDir;
    float scale;
    bool active;
    mlir::Operation* user;  // create mul op between user->getOperand(user_operand_index) and user
    size_t user_operand_index;

    MulPropData()
            : traversalDir(TraversalDir::INVALID), scale(-1.0f), active(false), user(nullptr), user_operand_index(0) {
    }
    MulPropData(TraversalDir travDir, float scaleValue, bool enableProp, mlir::Operation* usr, size_t usr_operand_idx)
            : traversalDir(travDir),
              scale(scaleValue),
              active(enableProp),
              user(usr),
              user_operand_index(usr_operand_idx) {
        VPUX_THROW_WHEN(!user, "User parameter cannot be null");
        VPUX_THROW_WHEN(usr_operand_idx >= user->getNumOperands(),
                        "Operand index too large, {0}, loc: {1}, idx: {2}, nopers: {3}", user->getName(),
                        user->getLoc(), user_operand_index, user->getNumOperands());
        auto operand = user->getOperand(usr_operand_idx);
        active = operand ? active : false;
    }

    mlir::Operation* createMulOp(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx) {
        if (!active) {
            // Prop data has been disabled don't create it.
            return nullptr;
        }

        // STEP 7: Fuse to constant op.
        if (auto cstOp =
                    mlir::dyn_cast_or_null<Const::DeclareOp>(user->getOperand(user_operand_index).getDefiningOp())) {
            auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
            auto newContentAttr = cstOp.getContentAttr().transform().rescale(scale).get();
            mlir::OpBuilder builder(cstOp);
            auto newCstOp = builder.create<vpux::Const::DeclareOp>(cstOp.getLoc(), cstType, std::move(newContentAttr));
            user->setOperand(user_operand_index, newCstOp);
            return newCstOp;
        }
        auto opervalue = user->getOperand(user_operand_index);
        auto defoper = opervalue.getDefiningOp();
        auto mulOp = defoper ? createMultiplyOp(defoper, rewriter, ctx, scale)
                             : createMultiplyOp(opervalue, rewriter, ctx, scale);
        user->setOperand(user_operand_index, mulOp);
        return mulOp;
    }
};

std::unique_ptr<OpParamScalerBase> nodeParamScalerFactory(mlir::Operation* op) {
    // Right now there is support only for FakeQuantize, in future we could add other things
    // e.g., Quantize, Dequantize etc.
    if (mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        return std::make_unique<OpParamScaler<IE::FakeQuantizeOp>>();
    }

    return std::unique_ptr<OpParamScalerBase>{};
}

// Create Multiply operations using the mulProps vector
// Update FakeQuantize nodes using opMetaData
mlir::LogicalResult createMulsAndUpdateOps(llvm::SmallVector<mlir::Operation*>& subgraph,
                                           llvm::SmallVector<MulPropData>& mulProps, mlir::PatternRewriter& rewriter,
                                           mlir::MLIRContext* ctx,
                                           llvm::DenseMap<mlir::Operation*, OpParamdata>& opMetaData) {
    for (size_t i = 0; i < mulProps.size(); ++i) {
        // inactive mul ops are ignored in the member function call below,
        // so no need to check again.
        mulProps[i].createMulOp(rewriter, ctx);
    }

    // Maintain a map of current operations
    // Why not update subgraph[i]?
    // Just updating subgraph[i] does not work because it can be repeated in the array.
    // A node can be repeated in the subgraph if
    // for e.g., a Mul propagates upward from an Add operation with distinct operands.
    // Besides that, once subgraph[i] is processed the index i is not revisited.
    llvm::DenseMap<mlir::Operation*, mlir::Operation*> currOpStore;
    for (auto& sop : subgraph) {
        currOpStore[sop] = sop;
    }
    for (size_t i = 0; i < subgraph.size(); ++i) {
        auto& metadata = opMetaData[subgraph[i]];
        auto nps = nodeParamScalerFactory(subgraph[i]);
        if (nullptr == nps) {
            continue;
        }
        auto res = (*nps)(currOpStore[subgraph[i]], rewriter, ctx, metadata.tdir, metadata.scale, metadata.scaleMode);
        if (!res.second) {
            return mlir::failure();
        }
        if (mlir::failed(res.first)) {
            return mlir::failure();
        }
        currOpStore[subgraph[i]] = nps->updatedOp();
    }
    return mlir::success();
}

void moveMulToOutputWithFilter(const MulPropData mop, llvm::SmallVector<MulPropData>& mulProps,
                               std::queue<size_t>& mulQ, float newScale,
                               llvm::DenseSet<mlir::Operation*>& ignoreUserSet) {
    for (size_t j = 0; j < mop.user->getNumResults(); ++j) {
        llvm::DenseSet<mlir::Operation*> visited;
        for (auto tuuser : mop.user->getResult(j).getUsers()) {
            if (ignoreUserSet.contains(tuuser) || visited.contains(tuuser)) {
                // Each user is visited only once.
                continue;
            }
            visited.insert(tuuser);
            for (size_t k = 0; k < tuuser->getNumOperands(); ++k) {
                auto oper = tuuser->getOperand(k).getDefiningOp();
                if (oper != mop.user) {
                    continue;
                }
                mulProps.emplace_back(MulPropData(TraversalDir::DOWN, newScale, true, tuuser, k));
                mulQ.push(mulProps.size() - 1);
                // A multiply is introduced on one operand of a given user only.
                break;
            }
        }
    }
}

void moveMulToOutputWithFilter(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ,
                               float newScale, llvm::DenseSet<mlir::Operation*>& ignoreUserSet) {
    mulProps[mopIndex].active = false;
    moveMulToOutputWithFilter(mulProps[mopIndex], mulProps, mulQ, newScale, ignoreUserSet);
}
// Wrapper function for the majority use case where there are no ignored users
void moveMulToOutput(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ,
                     float newScale) {
    llvm::DenseSet<mlir::Operation*> ignoreUserSet;
    moveMulToOutputWithFilter(mopIndex, mulProps, mulQ, newScale, ignoreUserSet);
}

void moveMulToInputIndex(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ,
                         float newScale, size_t idx) {
    mulProps[mopIndex].active = false;
    // look at operands of toperand since we are traversing up.
    auto toperand = mulProps[mopIndex].user->getOperand(mulProps[mopIndex].user_operand_index).getDefiningOp();
    mulProps.emplace_back(MulPropData(TraversalDir::UP, newScale, true, toperand, idx));
    mulQ.push(mulProps.size() - 1);
}

void moveMulToAllInputs(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ,
                        float newScale) {
    mulProps[mopIndex].active = false;
    // look at operands of toperand since we are traversing up.
    auto toperand = mulProps[mopIndex].user->getOperand(mulProps[mopIndex].user_operand_index).getDefiningOp();
    for (size_t i = 0; i < toperand->getNumOperands(); ++i) {
        mulProps.emplace_back(MulPropData(TraversalDir::UP, newScale, true, toperand, i));
        mulQ.push(mulProps.size() - 1);
    }
}

// Handles FQ update and considers the quantized weights case as well which is required for the const case of STEP 5.
bool setupFQUpdate(IE::FakeQuantizeOp& fqOp, float scale, TraversalDir tdir, SubgraphMetaData& opMetaData,
                   llvm::SmallVector<mlir::Operation*>& subgraph) {
    const size_t input_index = 0;
    auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(fqOp.getOperand(input_index).getDefiningOp());
    auto opd = OpParamdata(scale, fqOp, tdir);
    // If cstOp is not null and we have quantized weights we don't want to change those.
    bool quantConstOp = nullptr != cstOp && isQuantizedConstOp(cstOp);
    if (quantConstOp) {
        // In addition we only want to scale the output low and output high for the FQs
        // and leave the input range untouched.
        opd.scaleMode = OpParamdata::ScaleMode::OUTPUT_ONLY;
    }
    if (opMetaData.contains(fqOp)) {
        opd.scale *= opMetaData[fqOp].scale;
        opMetaData[fqOp] = opd;
    } else {
        opMetaData.try_emplace(fqOp, opd);
        subgraph.push_back(fqOp);
    }
    return quantConstOp;
}

// STEP3 : Handle FQ of E#171489
void handleFQOpDown(size_t mopIndex, SubgraphMetaData& opMetaData, llvm::SmallVector<mlir::Operation*>& subgraph,
                    llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(mulProps[mopIndex].user);
    if (!fqOp) {
        return;
    }

    mulProps[mopIndex].active = false;

    if (!setupFQUpdate(fqOp, 1.0f / mulProps[mopIndex].scale, mulProps[mopIndex].traversalDir, opMetaData, subgraph)) {
        // Propagate to output of FQ.
        moveMulToOutput(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale);
    }
}

// STEP 3: Handle FQ of E#171489
void handleFQOpUp(size_t mopIndex, SubgraphMetaData& opMetaData, llvm::SmallVector<mlir::Operation*>& subgraph,
                  llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(
            mulProps[mopIndex].user->getOperand(mulProps[mopIndex].user_operand_index).getDefiningOp());
    if (!fqOp) {
        return;
    }

    // During the up propagation the muls are propagated to all parameters of FQ
    // and they reach all the way to the top of the graph and scale the constants.
    // So updating them below
    // will cause duplicate updates if we go with Option A moveMulToAllInputs
    // OPTION A: Move mul
    // moveMulToAllInputs(mop, mulProps, mulQ, mop.scale);

    // OPTION B: Move mul only to index 0 input and rescale the FQ op.
    mulProps[mopIndex].active = false;

    // If cstOp is not null and we have quantized weights we don't want to change those.
    if (!setupFQUpdate(fqOp, mulProps[mopIndex].scale, mulProps[mopIndex].traversalDir, opMetaData, subgraph)) {
        const size_t input_index = 0;
        moveMulToInputIndex(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale, input_index);
    }

    // We cannot simply call moveMulToOutput because we want to prevent multiply from being applied to mop.user.
    // So mop.user is added to ignoredUsers so mul does not go back down.
    const float downScale = 1.0f / mulProps[mopIndex].scale;
    const size_t fakeOpIndex = 0;  // This is ignored in moveMulToOutput.
    MulPropData tempMop(TraversalDir::DOWN, downScale, true, fqOp, fakeOpIndex);
    llvm::DenseSet<mlir::Operation*> ignoredUsers;
    ignoredUsers.insert(mulProps[mopIndex].user);
    moveMulToOutputWithFilter(tempMop, mulProps, mulQ, downScale, ignoredUsers);
    tempMop.active = false;
}

// STEP 4: Handle Add of E#171489
void handleAddSubDown(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    mulProps[mopIndex].active = false;
    // Need to propagate mul down and up.
    for (size_t i = 0; i < mulProps[mopIndex].user->getNumOperands(); ++i) {
        if (i == mulProps[mopIndex].user_operand_index) {
            continue;
        }
        if (mulProps[mopIndex].user->getOperand(i).getDefiningOp() ==
            mulProps[mopIndex].user->getOperand(mulProps[mopIndex].user_operand_index).getDefiningOp()) {
            // This is an identical input so don't propagate here.
            continue;
        }
        mulProps.emplace_back(
                MulPropData(TraversalDir::UP, 1.0f / mulProps[mopIndex].scale, true, mulProps[mopIndex].user, i));
        mulQ.push(mulProps.size() - 1);
    }

    moveMulToOutput(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale);
}

// STEP 4: Handle Add of E#171489
void handleAddSubUp(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    moveMulToAllInputs(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale);
}

bool hasConstWeightsOpAtIndex(mlir::Operation* oper, size_t index) {
    return nullptr != mlir::dyn_cast_or_null<Const::DeclareOp>(oper->getOperand(index).getDefiningOp());
}

bool fuseConstScale(mlir::Operation* oper, size_t operand_index, float scale) {
    if (auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(oper->getOperand(operand_index).getDefiningOp())) {
        auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
        auto newContentAttr = cstOp.getContentAttr().transform().rescale(scale).get();
        mlir::OpBuilder builder(cstOp);
        auto newCstOp = builder.create<vpux::Const::DeclareOp>(cstOp.getLoc(), cstType, std::move(newContentAttr));
        oper->setOperand(operand_index, newCstOp);
        return true;
    }
    return false;
}

bool propagateToConstWeights(float scale, mlir::Operation* oper) {
    if (!mlir::isa<IE::MultiplyOp, IE::MatMulOp, IE::ConvolutionOp>(oper)) {
        return false;
    }

    VPUX_THROW_WHEN(oper->getNumOperands() < 2, "Multiply, MatMul, Convolution Ops must have at least 2 operands");

    size_t expected_count = 0;
    // For multiply, matmul and convolution multiply weights.
    llvm::SmallVector<size_t> operandIndices;
    if (hasConstWeightsOpAtIndex(oper, 1)) {
        operandIndices.push_back(1);
        ++expected_count;
    }

    // For convolution multiply biases as well.
    if (mlir::isa<IE::ConvolutionOp>(oper) && oper->getNumOperands() > 2) {
        if (hasConstWeightsOpAtIndex(oper, 2)) {
            operandIndices.push_back(2);
            ++expected_count;
        }
    }
    if (0 == expected_count) {
        // no const ops;
        return false;
    }

    size_t processed_count = 0;
    for (size_t i = 0; i < operandIndices.size(); ++i) {
        auto index = operandIndices[i];
        if (fuseConstScale(oper, index, scale)) {
            ++processed_count;
        }
    }
    VPUX_THROW_WHEN(expected_count != processed_count,
                    "ERROR: Could not fuse all const operands or found a mix of const and non-const operands");
    return expected_count == processed_count;
}

bool propagateToConstWeightsDown(const MulPropData& mop) {
    if (propagateToConstWeights(mop.scale, mop.user)) {
        return true;
    }
    return false;
}

bool propagateToConstWeightsUp(const MulPropData& mop) {
    auto oper = mop.user->getOperand(mop.user_operand_index).getDefiningOp();
    if (propagateToConstWeights(mop.scale, oper)) {
        return true;
    }
    return false;
}

// STEP 5: Handle Multiply, Matmul and Conv
void handleMultFamilyDown(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    llvm::DenseSet<size_t> processedIndices;
    if (propagateToConstWeightsDown(mulProps[mopIndex])) {
        mulProps[mopIndex].active = false;
        return;
    }
    moveMulToOutput(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale);
}

// STEP 5: Handle Multiply, Matmul and Conv
void handleMultFamilyUp(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    llvm::DenseSet<size_t> processedIndices;
    if (propagateToConstWeightsUp(mulProps[mopIndex])) {
        mulProps[mopIndex].active = false;
        return;
    }
    mulProps[mopIndex].active = false;
    auto toperand = mulProps[mopIndex].user->getOperand(mulProps[mopIndex].user_operand_index).getDefiningOp();
    if (!toperand) {
        return;
    }
    VPUX_THROW_WHEN(toperand->getNumOperands() < 2, "Found multiply op with less than two operands");
    const size_t secondOperandIndex = 1;
    mulProps.emplace_back(MulPropData(TraversalDir::UP, mulProps[mopIndex].scale, true, toperand, secondOperandIndex));
    mulQ.push(mulProps.size() - 1);

    if (mlir::isa<IE::ConvolutionOp>(toperand) && toperand->getNumOperands() > 2) {
        // propagate to biases as well.
        const size_t thirdOperandIndex = 2;
        mulProps.emplace_back(
                MulPropData(TraversalDir::UP, mulProps[mopIndex].scale, true, toperand, thirdOperandIndex));
        mulQ.push(mulProps.size() - 1);
    }
}

void handleMemoryOpDown(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    moveMulToOutput(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale);
}

void handleMemoryOpUp(size_t mopIndex, llvm::SmallVector<MulPropData>& mulProps, std::queue<size_t>& mulQ) {
    moveMulToAllInputs(mopIndex, mulProps, mulQ, mulProps[mopIndex].scale);
}

// Traverse the subgraph and determine where the multiply operations are to be introduced.
// The introduced multiply operations are propagated through the graph.
// During propagation some of the operations may become inactive and MulPropData for those
// operations is updated accordingly.
// Further for FakeQuantize operations, the scaling information is stored in OpParamData.
// Once the traversal is done, there is information to update the input graph by introducing
// the multiply operations and updating the FQ parameters.
mlir::LogicalResult traverseSubgraph(llvm::SmallVector<mlir::Operation*>& subgraph,
                                     llvm::SmallVector<MulPropData>& mulProps,
                                     llvm::DenseMap<mlir::Operation*, OpParamdata>& opMetaData) {
    auto front = subgraph.front();
    auto rc = rescaleCoefficient(front);
    std::queue<size_t> mulQ;

    for (size_t i = 0; i < front->getNumOperands(); ++i) {
        auto oper = front->getOperand(i).getDefiningOp();
        if (auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(oper)) {
            continue;
        }
        mulProps.emplace_back(MulPropData(TraversalDir::UP, 1.0f / rc, true, front, i));
        mulQ.push(mulProps.size() - 1);
    }

    // Code reuse:
    // Just use a fake MulProp and reuse moveMulToOutput.
    const size_t dont_care_op_index = 0;
    auto frontMp = MulPropData(TraversalDir::DOWN, rc, true, front, dont_care_op_index);
    llvm::DenseSet<mlir::Operation*> ignoredUsers;
    moveMulToOutputWithFilter(frontMp, mulProps, mulQ, rc, ignoredUsers);
    frontMp.active = false;

    opMetaData[front] = OpParamdata(1.0f / rc, front, TraversalDir::INVALID);

    // Mechanism to break a circular dependency.
    // Keep a count of number of times an operation is added to mulQ.
    // If it exceeds a threshold return failure.
    llvm::DenseMap<mlir::Operation*, size_t> visitCount;
    auto increment_visit_count = [&](mlir::Operation* op) {
        if (visitCount.contains(op)) {
            visitCount[op] += 1;
            return;
        }
        visitCount[op] = 1;
    };
    increment_visit_count(front);
    auto visits_limit_exceeded = [&](mlir::Operation* op) {
        if (!op) {
            return false;
        }
        const size_t count_threshold = 20;
        increment_visit_count(op);
        return visitCount[op] > count_threshold;
    };

    // The traversal does a breadth first search starting with the first
    // node which was breaking FP16 threshold. For each operation encountered
    // during traversal we do one or more of the following:
    // - create a MulProp and add it to queue for further propagation.
    // - create or update metadata for an operation in opMetaData, for e.g., to update FQ nodes.
    // - Nothing - either we are at a stop node or an unrecognized operation.
    while (!mulQ.empty()) {
        auto itop = mulQ.front();

        mulQ.pop();
        if (!mulProps[itop].active) {
            continue;
        }
        if (visits_limit_exceeded(mulProps[itop].user)) {
            return mlir::failure();
        }

        auto tuser = mulProps[itop].user;
        VPUX_THROW_WHEN(!tuser || mulProps[itop].user_operand_index >= mulProps[itop].user->getNumOperands(),
                        "User and user operand index must be valid for an active mul prop");
        auto toperandVal = tuser->getOperand(mulProps[itop].user_operand_index);
        VPUX_THROW_WHEN(!toperandVal, "The operand value cannot be null for an active op");
        auto toperand = toperandVal.getDefiningOp();
        if (visits_limit_exceeded(toperand)) {
            return mlir::failure();
        }

        if (mulProps[itop].traversalDir == TraversalDir::DOWN) {
            llvm::TypeSwitch<mlir::Operation*>(tuser)
                    .Case<IE::AddOp>([&](IE::AddOp) {
                        handleAddSubDown(itop, mulProps, mulQ);
                    })
                    .Case<IE::SubtractOp>([&](IE::SubtractOp) {
                        handleAddSubDown(itop, mulProps, mulQ);
                    })
                    .Case<IE::FakeQuantizeOp>([&](IE::FakeQuantizeOp) {
                        handleFQOpDown(itop, opMetaData, subgraph, mulProps, mulQ);
                    })
                    .Case<IE::MultiplyOp>([&](IE::MultiplyOp) {
                        handleMultFamilyDown(itop, mulProps, mulQ);
                    })
                    .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp) {
                        handleMultFamilyDown(itop, mulProps, mulQ);
                    })
                    .Case<IE::MatMulOp>([&](IE::MatMulOp) {
                        handleMultFamilyDown(itop, mulProps, mulQ);
                    })
                    // Const::DeclareOps are handled in CreateMultiplyOp
                    //.Case<Const::DeclareOp>([&](auto constOp) {
                    // top.active = false;
                    // opMetaData.try_emplace(constOp, Metadata(top.scale, top.scale, constOp, nullptr,
                    // top.traversalDir)); subgraph.push_back(constOp);
                    //})
                    .Default([&](mlir::Operation* op) {
                        if (isStopOperation(op)) {
                            // STEP 2
                            mulProps[itop].active = false;
                        }
                        if (isMemoryOp(op)) {
                            handleMemoryOpDown(itop, mulProps, mulQ);
                        } else {
                            // Unsupported yet.
                        }
                    });
        } else if (toperand && mulProps[itop].traversalDir == TraversalDir::UP) {
            // Note if toperand is nullptr then the defining op is null we just need to create a mul op using the value.
            // That is why the if statement above is different compared to down traversal.
            llvm::TypeSwitch<mlir::Operation*>(toperand)
                    .Case<IE::AddOp>([&](IE::AddOp) {
                        handleAddSubUp(itop, mulProps, mulQ);
                    })
                    .Case<IE::SubtractOp>([&](IE::SubtractOp) {
                        handleAddSubUp(itop, mulProps, mulQ);
                    })
                    .Case<IE::FakeQuantizeOp>([&](IE::FakeQuantizeOp) {
                        handleFQOpUp(itop, opMetaData, subgraph, mulProps, mulQ);
                    })
                    .Case<IE::MultiplyOp>([&](IE::MultiplyOp) {
                        handleMultFamilyUp(itop, mulProps, mulQ);
                    })
                    .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp) {
                        handleMultFamilyUp(itop, mulProps, mulQ);
                    })
                    .Case<IE::MatMulOp>([&](IE::MatMulOp) {
                        handleMultFamilyUp(itop, mulProps, mulQ);
                    })
                    // .Case<Const::DeclareOp>([&](Const::DeclareOp constOp) {
                    // Const::DeclareOps are handled in CreateMultiplyOp
                    // top.active = false;
                    // opMetaData.try_emplace(constOp, Metadata(top.scale, top.scale, constOp, nullptr,
                    // top.traversalDir)); subgraph.push_back(constOp);
                    // })
                    .Default([&](mlir::Operation* op) {
                        if (isStopOperation(op)) {
                            // STEP 2
                            // nothing to do here, just stop propagation
                        } else if (isMemoryOp(op)) {
                            handleMemoryOpUp(itop, mulProps, mulQ);
                        } else {
                            // Unsupported yet.
                        }
                    });
        }
    }

    return mlir::success();
}

// For FakeQuantize nodes with parameters outside FP16 range this pass rescales the parameters and introduces
// multiply operations through the graph, as outlined in the ticket E#171489.
mlir::LogicalResult FakeQdqParamsRewriter::matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto levels = fakeQuantizeOp.getLevels();

    // Maximum number of levels that don't exceeds I8/U8 storage type . TODO: E#169024 adjust logic for int16 quant
    // levels.
    if (!levels.has_value() || *levels <= QuantizationLevels::QUANT_LEVELS_8BIT) {
        return matchFailed(rewriter, fakeQuantizeOp,
                           "Skipping AdjustFQParams pass for quantization range < i8 {0} at {1}",
                           fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());
    }

    // FQ with in_low != out_low or in_high != out_high is replaced with a ScaleShift op in HandleU16FakeQuantize pass.
    if (!IE::isPerTensorFQ({fakeQuantizeOp}) || !isFqRangeOutOfBounds(fakeQuantizeOp) ||
        !areFQValsEqual(fakeQuantizeOp)) {
        return matchFailed(rewriter, fakeQuantizeOp, "Skipping AdjustFQParams pass as FQ {0} at {1} is in range",
                           fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());
    }
    // Followup: Generalize current implementation to:
    // per-channel and multi-channel FakeQuantize. Look at E#177612

    llvm::SmallVector<mlir::Operation*> subgraph;
    llvm::DenseMap<mlir::Operation*, OpParamdata> subgraphMetaData;
    subgraph.push_back(fakeQuantizeOp);

    llvm::SmallVector<MulPropData> mulProps;

    // PHASE 1: Traverse subgraph starting from the matched FakeQuantize Op and determine where graph should be changed.
    if (mlir::failed(traverseSubgraph(subgraph, mulProps, subgraphMetaData))) {
        return matchFailed(rewriter, fakeQuantizeOp, "Graph Traversal Failed, for FQ: {0}, loc: {1}",
                           fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());
    }

    // PHASE 2: Update the input graph by introducing multiply operations and/or updating FakeQuantize nodes.
    if (mlir::failed(createMulsAndUpdateOps(subgraph, mulProps, rewriter, getContext(), subgraphMetaData))) {
        return matchFailed(rewriter, fakeQuantizeOp, "Create Multiply Ops Failed, for FQ: {0}, loc: {1}",
                           fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());
    }

    return mlir::success();
}

//
// AdjustFakeQdqParams
//
class AdjustFakeQdqParamsPass final : public IE::impl::AdjustFakeQdqParamsBase<AdjustFakeQdqParamsPass> {
public:
    explicit AdjustFakeQdqParamsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustFakeQdqParamsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FakeQdqParamsRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustFakeQdqParamsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustFakeQdqParamsPass(Logger log) {
    return std::make_unique<AdjustFakeQdqParamsPass>(log);
}
