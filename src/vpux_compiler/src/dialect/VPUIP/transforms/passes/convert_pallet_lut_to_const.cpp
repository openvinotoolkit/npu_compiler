//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/convert_lut_to_const.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTPALLETLUTTOCONST
#define GEN_PASS_DEF_CONVERTPALLETLUTTOCONST
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// PalletLUTConverter
//

class PalletLUTConverter final : public VPUIP::LUTConverterBase {
public:
    PalletLUTConverter(mlir::MLIRContext* ctx, Logger log, mlir::func::FuncOp netFunc)
            : LUTConverterBase(ctx, log, netFunc) {
        setDebugName("ConvertPalletLUTToConstPass::PalletLUTConverter");
    }

private:
    SmallVector<uint16_t> fillPalletTable(mlir::Type quantileType, ArrayRef<double> quantilesLut) const;
    mlir::Value createLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                       mlir::PatternRewriter& rewriter) const override;
    void replaceWithConstInput(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value lutNceInput,
                               mlir::PatternRewriter& rewriter) const override;
};

SmallVector<uint16_t> PalletLUTConverter::fillPalletTable(mlir::Type quantileType,
                                                          ArrayRef<double> quantilesLut) const {
    // For 8 bit quantileType, the 16 bit pallet LUT entries get duplicated as 2 x 8 bit elems.
    // If the value is not correctly dimensioned for the 8 bit range, truncations will occur.
    auto getPalletModeBitValue = [quantileType](const double value) -> uint16_t {
        if (quantileType.isF16()) {
            vpux::type::float16 f16(static_cast<float>(value));
            return f16.to_bits();
        } else if (quantileType.isBF16()) {
            vpux::type::bfloat16 bf16(static_cast<float>(value));
            return bf16.to_bits();
        } else if (quantileType.isUnsignedInteger(8)) {
            uint16_t u8Masked = value < 0. ? 0u : static_cast<uint8_t>(value);
            return u8Masked << 8 | u8Masked;  // Duplicate the value for 16-bit representation
        } else if (quantileType.isSignedInteger(8)) {
            uint16_t i8Masked = static_cast<uint16_t>(static_cast<int16_t>(value) & 0x00FF);
            return i8Masked << 8 | i8Masked;
        } else if (mlir::isa<mlir::Float8E5M2Type>(quantileType)) {
            vpux::type::float8_e5m2 bf8(static_cast<float>(value));
            uint16_t bf8Ext = bf8.to_bits();
            return bf8Ext << 8 | bf8Ext;
        } else if (mlir::isa<mlir::Float8E4M3FNType>(quantileType)) {
            vpux::type::float8_e4m3 hf8(static_cast<float>(value));
            uint16_t hf8Ext = hf8.to_bits();
            return hf8Ext << 8 | hf8Ext;
        } else {
            VPUX_THROW("getPalletModeBitValue: Unsupported quantileType for palletization table {0}", quantileType);
        }
        return 0;
    };

    constexpr unsigned PALLETIZATION_TABLE_16BIT_ENTRIES = 64;
    SmallVector<uint16_t> lutValues(PALLETIZATION_TABLE_16BIT_ENTRIES, 0);
    for (unsigned i = 0; i < quantilesLut.size(); ++i) {
        lutValues[i] = getPalletModeBitValue(quantilesLut[i]);
    }

    return lutValues;
}

mlir::Value PalletLUTConverter::createLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto weightsType =
            mlir::dyn_cast<vpux::NDTypeInterface>(nceClusterTask.getWeights().getType()).getElementType();

    const auto [quantileType, quantileLUT] = [&]() {
        if (const auto quantileUniformType = mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedType>(weightsType)) {
            return std::tuple<mlir::Type, ArrayRef<double>>(quantileUniformType.getQuantileType(),
                                                            quantileUniformType.getQuantiles());
        }
        if (const auto quantilePerAxisType =
                    mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(weightsType)) {
            return std::tuple<mlir::Type, ArrayRef<double>>(quantilePerAxisType.getQuantileType(),
                                                            quantilePerAxisType.getQuantiles());
        }
        VPUX_THROW("{0}: expected palletized weight type but {1} type was found instead", getDebugName(), weightsType);
    }();

    auto uint16PalletLUT = fillPalletTable(quantileType, quantileLUT);
    auto uint16Type =
            mlir::IntegerType::get(rewriter.getContext(), 16, mlir::IntegerType::SignednessSemantics::Unsigned);
    auto palletLUTType = mlir::RankedTensorType::get({checked_cast<int64_t>(uint16PalletLUT.size())}, uint16Type);
    auto palletLUTAttr = mlir::DenseElementsAttr::get(palletLUTType, ArrayRef(uint16PalletLUT));

    const auto bufferType = vpux::getBufferType(palletLUTType);
    Const::ContentSetup setup(mlir::cast<mlir::Type>(bufferType));
    const auto contentAttr = Const::ContentAttr::get(palletLUTAttr, setup);
    return rewriter.create<Const::DeclareOp>(nceClusterTask->getLoc(), bufferType, contentAttr).getOutput();
}

void PalletLUTConverter::replaceWithConstInput(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value lutNceInput,
                                               mlir::PatternRewriter& rewriter) const {
    auto newInput = [&]() -> mlir::Value {
        if (vpux::VPUIP::hasDistributedOperand(nceClusterTask)) {
            const auto palletLUTOutType = mlir::dyn_cast<VPUIP::DistributedBufferType>(lutNceInput.getType());
            VPUX_THROW_WHEN(palletLUTOutType == nullptr,
                            "{0}: pallet LUT output type is expected to be DistributedBufferType, but got {1}",
                            getDebugName(), lutNceInput.getType());
            nceClusterTask.getPalletLookupTableMutable().append(lutNceInput);
        }
        return lutNceInput;
    }();
    rewriter.modifyOpInPlace(nceClusterTask, [&] {
        nceClusterTask.getPalletLookupTableMutable().assign(newInput);
    });
}

//
// ConvertPalletLUTToConstPass
//

class ConvertPalletLUTToConstPass final : public VPUIP::impl::ConvertPalletLUTToConstBase<ConvertPalletLUTToConstPass> {
public:
    explicit ConvertPalletLUTToConstPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertPalletLUTToConstPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget palletLutTarget(ctx);
    palletLutTarget.addLegalOp<Const::DeclareOp, VPUIP::CopyOp, VPURT::AllocDistributed, mlir::memref::AllocOp>();
    palletLutTarget.addDynamicallyLegalOp<VPUIP::NCEClusterTaskOp>([](VPUIP::NCEClusterTaskOp op) {
        if (op.getPalletLookupTable() != nullptr) {
            return true;
        }
        if (auto weights = op.getWeights()) {
            const auto weightsType = mlir::dyn_cast<vpux::NDTypeInterface>(weights.getType()).getElementType();
            if (mlir::isa_and_nonnull<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(
                        weightsType)) {
                return false;
            }
        }
        return true;
    });

    mlir::RewritePatternSet palletLutPatterns(&ctx);
    palletLutPatterns.add<PalletLUTConverter>(&ctx, _log, func);
    if (mlir::failed(applyPartialConversion(func, palletLutTarget, std::move(palletLutPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPalletLUTToConstPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertPalletLUTToConstPass(Logger log) {
    return std::make_unique<ConvertPalletLUTToConstPass>(log);
}
