//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <llvm/ADT/STLExtras.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

namespace vpux::VPU {

// Out-of-line definition satisfies Coverity's Rule of Three check (C.21).
PrecomputedStrategyTable::~PrecomputedStrategyTable() = default;

void PrecomputedStrategyTable::setBuiltinTable(const uint8_t* data, std::size_t size) {
    _builtinTable = {data, size};
    Logger::global().info("[PTC] Builtin precomputed strategy table registered ({0} bytes)", size);
}

ArrayRef<uint8_t> PrecomputedStrategyTable::getBuiltinTable() const {
    return _builtinTable;
}

namespace {

constexpr StringLiteral tilingStrategy = "tilingStrategy";

// ── FNV-1a 64-bit hash ────────────────────────────────────────────────────────
// Custom stable combiner used instead of llvm::hash_combine to guarantee
// identical results across LLVM versions — required for .ptc binary portability.

constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;

uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

uint64_t fnv1a(StringRef s, uint64_t h) {
    return fnv1a(s.data(), s.size(), h);
}

uint64_t fnv1a(int64_t v, uint64_t h) {
    return fnv1a(&v, sizeof(v), h);
}

/// Hashes the descriptor for a single tensor value: rank, each shape dim,
/// layout canonical name, and element type string.  Missing value hashes a sentinel.
uint64_t hashTensorDescriptor(mlir::Value val, uint64_t h) {
    if (!val) {
        return fnv1a(int64_t(-1), h);
    }
    auto nd = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
    if (!nd) {
        return fnv1a(int64_t(-2), h);
    }
    const auto shape = nd.getShape();
    h = fnv1a(int64_t(shape.size()), h);
    for (auto dim : shape) {
        h = fnv1a(int64_t(dim), h);
    }
    h = fnv1a(nd.getDimsOrder().getCanonicalName(), h);
    mlir::Type elemType = nd.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        elemType = qType.getStorageType();
        if (auto qfType = mlir::dyn_cast<vpux::type::QuantileType>(elemType)) {
            elemType = qfType.getStorageType();
        }
    }
    std::string ets;
    llvm::raw_string_ostream os(ets);
    elemType.print(os);
    return fnv1a(ets, h);
}

/// Compact representation of execution engine for direct numeric hashing.
enum class EngineKind : uint8_t { SHAVE = 0, SCL = 1, DCIM = 2 };

EngineKind engineKindForOp(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (!nceOp) {
        return EngineKind::SHAVE;
    }
    auto mpeEngine = nceOp.getMpeEngine();
    if (!mpeEngine.has_value() || mpeEngine.value() == nullptr) {
        return EngineKind::SCL;
    }
    return llvm::TypeSwitch<VPU::MPEEngineAttr, EngineKind>(mpeEngine.value())
            .Case<VPU::MPEEngine37XXAttr>([](VPU::MPEEngine37XXAttr) -> EngineKind {
                return EngineKind::SCL;
            })
            .Default([](VPU::MPEEngineAttr) -> EngineKind {
                return EngineKind::SCL;
            });
}

/// Formats hash as a zero-padded 16-character lowercase hex string.
std::string hashToHex(uint64_t hash) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, hash);
    return buf;
}

/// Returns the DPU tile cluster count for the module containing op.
int64_t getNumDpuTiles(mlir::Operation* op) {
    if (auto moduleOp = op->getParentOfType<mlir::ModuleOp>()) {
        if (auto tileExecutor = config::getTileExecutor(moduleOp)) {
            return tileExecutor.getCount();
        }
    }
    return 0;
}

}  // namespace

uint64_t computeOpStrategyHash(mlir::Operation* op, StringRef device, int64_t numTiles) {
    VPUX_THROW_UNLESS(op != nullptr, "Cannot compute strategy hash for null operation");

    uint64_t h = FNV_OFFSET;

    // Op name (e.g. "VPU.NCEConvolution").
    h = fnv1a(op->getName().getStringRef(), h);

    // Always hash both input slots for stability when operand count varies.
    const size_t nOperands = op->getNumOperands();
    h = hashTensorDescriptor(nOperands > 0 ? op->getOperand(0) : mlir::Value{}, h);
    h = hashTensorDescriptor(nOperands > 1 ? op->getOperand(1) : mlir::Value{}, h);

    // First output tensor descriptor.
    h = hashTensorDescriptor(op->getNumResults() > 0 ? op->getResult(0) : mlir::Value{}, h);

    // Strides and padding (NCE only; the op name already differentiates NCE from SW).
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
        const auto strides = nceOp.getStridesVal();
        h = fnv1a(int64_t(strides.size()), h);
        for (auto s : strides) {
            h = fnv1a(int64_t(s), h);
        }
        if (auto padAttr = nceOp.getPad()) {
            h = fnv1a(int64_t(padAttr.getLeft().getInt()), h);
            h = fnv1a(int64_t(padAttr.getRight().getInt()), h);
            h = fnv1a(int64_t(padAttr.getTop().getInt()), h);
            h = fnv1a(int64_t(padAttr.getBottom().getInt()), h);
        }
    }

    // Device string (e.g. "NPU40XX").
    h = fnv1a(device, h);

    // Engine as a stable 1-byte enum value.
    const auto ek = static_cast<uint8_t>(engineKindForOp(op));
    h = fnv1a(&ek, 1, h);

    // DPU tile count.
    h = fnv1a(numTiles, h);

    return h;
}

namespace {

// ── PTC binary format (version 2) ──────────────────────────────────────────────
// Must match the format written by tools/precomputed-strategy-table/convert_precomputed_strategy_table.py.
//
// Layout: [PtcHeader 16B] [HT: ht_slots × uint32 record indices] [PtcRecord × n_entries 64B]
//
// PtcRecord v2:
//   key[8]           64-bit FNV-1a hash (little-endian uint64_t)
//   spatial_idx[2]   MultiClusterStrategy enum index; PTC_NONE_U16 = none
//   pipeline_idx[2]  pipeline-mode index; PTC_NONE_U16 = none (placeholder)
//   _pad[4]          reserved, zero
//   temporal[5×4]    int32 tiling values; trailing zeros = unused dims (valid >= 1)
//   _reserved[28]    reserved for future use
//
// Probe: uint32(key) & (ht_slots-1), linear probing, empty slot = PTC_EMPTY_SLOT.
// All multi-byte integers are little-endian.

constexpr char PTC_MAGIC[4] = {'V', 'P', 'X', 'T'};
constexpr uint16_t PTC_VERSION = 2;
constexpr uint16_t PTC_NONE_U16 = 0xFFFFu;
constexpr uint32_t PTC_EMPTY_SLOT = 0xFFFFFFFFu;

struct PtcHeader {
    char magic[4];       // 'V','P','X','T'
    uint16_t version;    // must be PTC_VERSION (2)
    uint16_t _pad;       // reserved, zero
    uint32_t n_entries;  // number of PtcRecord entries
    uint32_t ht_slots;   // hash-table slot count (power of 2, >= 2 × n_entries)
};
static_assert(sizeof(PtcHeader) == 16, "PtcHeader size mismatch");

struct PtcRecord {
    uint64_t key;           // 64-bit FNV-1a hash
    uint16_t spatial_idx;   // MultiClusterStrategy enum index; PTC_NONE_U16 = none
    uint16_t pipeline_idx;  // pipeline-mode index; PTC_NONE_U16 = none (placeholder)
    uint8_t _pad[4];        // reserved, zero
    int32_t temporal[5];    // temporal-tiling values; trailing zeros = unused dims
    uint8_t _reserved[28];  // reserved for future use
};
static_assert(sizeof(PtcRecord) == 64, "PtcRecord size mismatch");

// Applies decoded spatial and/or temporal strategy fields to an op and marks
// it as pinned.  Either argument may be absent (std::nullopt / null attr).
void pinOpStrategies(mlir::Operation* op, std::optional<VPU::MultiClusterStrategy> spatial, mlir::Attribute temporal) {
    auto* ctx = op->getContext();
    if (spatial.has_value()) {
        if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
            clusteredOp.setMultiClusterStrategy(*spatial);
            op->setAttr(pinnedStrategy, mlir::UnitAttr::get(ctx));
        }
    }
    if (temporal != nullptr && op->getParentOfType<VPU::VerticalFusionOp>() == nullptr) {
        op->setAttr(tilingStrategy, temporal);
        op->setAttr(pinnedStrategy, mlir::UnitAttr::get(ctx));
    }
}

}  // namespace

void applyPrecomputedStrategyTableBinary(ArrayRef<uint8_t> buf, mlir::func::FuncOp func, Logger log) {
    const auto device = config::stringifyArchKind(config::getArch(func.getOperation())).str();
    const auto bufSize = buf.size();
    if (bufSize < sizeof(PtcHeader)) {
        log.warning("[PTC] builtin table too small for a valid header");
        return;
    }
    PtcHeader hdr;
    std::memcpy(&hdr, buf.data(), sizeof(PtcHeader));
    if (std::memcmp(hdr.magic, PTC_MAGIC, 4) != 0) {
        log.warning("[PTC] builtin table has invalid magic bytes");
        return;
    }
    if (hdr.version != PTC_VERSION) {
        log.warning("[PTC] builtin table has unsupported version {0} (expected {1}); "
                    "regenerate the .ptc file with the current compiler",
                    hdr.version, PTC_VERSION);
        return;
    }
    if (hdr.ht_slots == 0 || (hdr.ht_slots & (hdr.ht_slots - 1u)) != 0) {
        log.warning("[PTC] builtin table has invalid ht_slots={0} (must be a non-zero power of two)", hdr.ht_slots);
        return;
    }
    const uint64_t expectedSize = sizeof(PtcHeader) + static_cast<uint64_t>(hdr.ht_slots) * sizeof(uint32_t) +
                                  static_cast<uint64_t>(hdr.n_entries) * sizeof(PtcRecord);
    if (static_cast<uint64_t>(bufSize) < expectedSize) {
        log.warning("[PTC] builtin table is truncated ({0} < {1} bytes)", bufSize, expectedSize);
        return;
    }
    log.debug("[PTC] Header OK: version={0}, n_entries={1}, ht_slots={2}", hdr.version, hdr.n_entries, hdr.ht_slots);

    const size_t htOffset = sizeof(PtcHeader);
    const size_t recOffset = htOffset + hdr.ht_slots * sizeof(uint32_t);

    const auto readU32 = [&](size_t byteOffset) -> uint32_t {
        uint32_t v;
        std::memcpy(&v, buf.data() + byteOffset, sizeof(uint32_t));
        return v;
    };

    const auto readRecord = [&](uint32_t idx) -> PtcRecord {
        PtcRecord rec;
        std::memcpy(&rec, buf.data() + recOffset + idx * sizeof(PtcRecord), sizeof(PtcRecord));
        return rec;
    };

    // Probe: lower 32 bits of key & (ht_slots-1), linear probing.
    const auto findRecord = [&](uint64_t hashKey) -> std::optional<PtcRecord> {
        uint32_t probe = static_cast<uint32_t>(hashKey) & (hdr.ht_slots - 1u);
        for (uint32_t step = 0; step < hdr.ht_slots; ++step) {
            const uint32_t idx = readU32(htOffset + probe * sizeof(uint32_t));
            if (idx == PTC_EMPTY_SLOT) {
                return std::nullopt;
            }
            if (idx >= hdr.n_entries) {
                return std::nullopt;  // corrupted slot
            }
            const PtcRecord rec = readRecord(idx);
            if (rec.key == hashKey) {
                return rec;
            }
            probe = (probe + 1u) & (hdr.ht_slots - 1u);
        }
        return std::nullopt;
    };

    const int64_t numTiles = getNumDpuTiles(func.getOperation());
    uint32_t nHits = 0, nMisses = 0, nOps = 0;

    func->walk([&](VPU::LayerOpInterface layerOp) {
        mlir::Operation* op = layerOp.getOperation();
        if (!mlir::isa<VPU::NCEOpInterface>(op) && !mlir::isa<VPU::SWOpInterface>(op)) {
            return;
        }
        ++nOps;

        const uint64_t hashKey = computeOpStrategyHash(op, device, numTiles);
        log.debug("[PTC] op '{0}' hash={1}", vpux::stringifyPrimaryLocation(op->getLoc()), hashToHex(hashKey));

        const auto rec = findRecord(hashKey);
        if (!rec.has_value()) {
            ++nMisses;
            log.debug("[PTC]   MISS");
            return;
        }
        ++nHits;
        log.debug("[PTC]   HIT: spatial={0}", rec->spatial_idx);

        std::optional<VPU::MultiClusterStrategy> spatialVal;
        if (rec->spatial_idx != PTC_NONE_U16) {
            const auto s = symbolizeMultiClusterStrategy(static_cast<uint64_t>(rec->spatial_idx));
            if (s.has_value()) {
                log.debug("[PTC]   applying spatial_tiling={0}", stringifyMultiClusterStrategy(*s));
                spatialVal = *s;
            } else {
                log.warning("[PTC]   unknown spatial_tiling index {0} for op '{1}'", rec->spatial_idx,
                            vpux::stringifyPrimaryLocation(op->getLoc()));
            }
        }

        // Infer ndims from last non-zero element (trailing zeros are unused sentinels).
        const auto countDims = [](const int32_t(&arr)[5]) -> uint8_t {
            for (int d = 4; d >= 0; --d) {
                if (arr[d] != 0) {
                    return static_cast<uint8_t>(d + 1);
                }
            }
            return 0;
        };
        const uint8_t nTemporal = countDims(rec->temporal);
        log.debug("[PTC]   inferred n_temporal={0}", nTemporal);

        mlir::Attribute temporalVal;
        if (nTemporal > 0) {
            bool validTiling = true;
            for (uint8_t d = 0; d < nTemporal; ++d) {
                if (rec->temporal[d] < 1) {
                    log.warning("[PTC]   temporal_tiling dim[{0}]={1} < 1 for op '{2}' — skipping", d, rec->temporal[d],
                                vpux::stringifyPrimaryLocation(op->getLoc()));
                    validTiling = false;
                    break;
                }
            }
            if (validTiling) {
                Shape tilingShape(nTemporal, 1);
                for (uint8_t d = 0; d < nTemporal; ++d) {
                    tilingShape[Dim(d)] = static_cast<int64_t>(rec->temporal[d]);
                }
                log.debug("[PTC]   applying temporal_tiling={0}", tilingShape);
                temporalVal = getIntArrayAttr(op->getContext(), tilingShape);
            }
        }

        pinOpStrategies(op, spatialVal, temporalVal);
    });

    log.info("[PTC] Done: {0} hit(s), {1} miss(es) out of {2} operation(s)", nHits, nMisses, nOps);
}

// ── JSON descriptor string helpers ───────────────────────────────────────────
// These exist solely to populate human-readable fields in the JSON output;
// they are not used for hash computation.

namespace {

// Formats an integer sequence as "(a, b, c)"; used for shapes, strides and padding.
std::string formatParenList(ArrayRef<int64_t> values) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "(";
    llvm::interleaveComma(values, os);
    os << ")";
    return s;
}

// Returns the value's NDTypeInterface, or nullptr if val is null or untyped.
vpux::NDTypeInterface ndTypeOrNull(mlir::Value val) {
    return val ? mlir::dyn_cast<vpux::NDTypeInterface>(val.getType()) : nullptr;
}

std::string shapeToStr(mlir::Value val) {
    const auto nd = ndTypeOrNull(val);
    return nd ? formatParenList(nd.getShape().raw()) : "N/A";
}

std::string layoutToStr(mlir::Value val) {
    const auto nd = ndTypeOrNull(val);
    return nd ? nd.getDimsOrder().getCanonicalName().str() : "N/A";
}

std::string elemTypeToStr(mlir::Value val) {
    const auto nd = ndTypeOrNull(val);
    if (!nd) {
        return "N/A";
    }
    mlir::Type elemType = nd.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        elemType = qType.getStorageType();
        if (auto qfType = mlir::dyn_cast<vpux::type::QuantileType>(elemType)) {
            elemType = qfType.getStorageType();
        }
    }
    std::string s;
    llvm::raw_string_ostream(s) << elemType;
    return s;
}

std::string stridesForOp(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    return nceOp ? formatParenList(nceOp.getStridesVal()) : "N/A";
}

std::string paddingForOp(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (!nceOp) {
        return "N/A";
    }
    auto padAttr = nceOp.getPad();
    if (!padAttr) {
        return "(0, 0, 0, 0)";
    }
    const SmallVector<int64_t> sides = {padAttr.getLeft().getInt(), padAttr.getRight().getInt(),
                                        padAttr.getTop().getInt(), padAttr.getBottom().getInt()};
    return formatParenList(sides);
}

std::string engineStrForOp(mlir::Operation* op) {
    switch (engineKindForOp(op)) {
    case EngineKind::SHAVE:
        return "SHAVE";
    case EngineKind::DCIM:
        return "DCIM";
    default:
        return "SCL";
    }
}

// Returns operand `idx`, or a null Value if the op has fewer operands.
mlir::Value operandOrNull(mlir::Operation* op, unsigned idx) {
    return idx < op->getNumOperands() ? op->getOperand(idx) : mlir::Value{};
}

// Returns result `idx`, or a null Value if the op has fewer results.
mlir::Value resultOrNull(mlir::Operation* op, unsigned idx) {
    return idx < op->getNumResults() ? op->getResult(idx) : mlir::Value{};
}

}  // namespace

void createPrecomputedStrategyTableJSON(llvm::json::Value& json, mlir::func::FuncOp func) {
    const auto device = config::stringifyArchKind(config::getArch(func.getOperation())).str();
    const int64_t numTiles = getNumDpuTiles(func.getOperation());

    // Seed from existing entries so multiple funcs accumulate into one table.
    // json is overwritten below, so steal its contents instead of copying them.
    auto* existingObj = json.getAsObject();
    llvm::json::Object entries = (existingObj != nullptr) ? std::move(*existingObj) : llvm::json::Object{};

    func->walk([&](VPU::LayerOpInterface layerOp) {
        mlir::Operation* op = layerOp.getOperation();
        if (!mlir::isa<VPU::NCEOpInterface>(op) && !mlir::isa<VPU::SWOpInterface>(op)) {
            return;
        }
        // Only include ops decided to need temporal tiling.
        if (!op->hasAttr(tilingStrategy)) {
            return;
        }

        const auto key = hashToHex(computeOpStrategyHash(op, device, numTiles));
        llvm::json::Object entry;

        entry["layer_name"] = vpux::stringifyPrimaryLocation(op->getLoc());
        entry["op_type"] = op->getName().getStringRef().str();

        const auto input1 = operandOrNull(op, 0);
        const auto input2 = operandOrNull(op, 1);
        const auto output = resultOrNull(op, 0);

        entry["input1_shape"] = shapeToStr(input1);
        entry["input1_layout"] = layoutToStr(input1);
        entry["input1_element_type"] = elemTypeToStr(input1);
        entry["input2_shape"] = shapeToStr(input2);
        entry["input2_layout"] = layoutToStr(input2);
        entry["input2_element_type"] = elemTypeToStr(input2);
        entry["output_shape"] = shapeToStr(output);
        entry["output_layout"] = layoutToStr(output);
        entry["output_element_type"] = elemTypeToStr(output);

        entry["strides"] = stridesForOp(op);
        entry["padding"] = paddingForOp(op);
        entry["device"] = device;
        entry["engine"] = engineStrForOp(op);
        entry["num_tiles"] = numTiles;

        entry[ptcSpatialTiling.str()] = op->hasAttr(vpux::multiClusterStrategy)
                                                ? convertAttrToJSON(op->getAttr(vpux::multiClusterStrategy))
                                                : llvm::json::Value(vpux::defaultNoValue.str());
        entry[ptcTemporalTiling.str()] = convertAttrToJSON(op->getAttr(tilingStrategy));
        entry[ptcPipelineMode.str()] = llvm::json::Value(vpux::defaultNoValue.str());
        entry["strategy_cost_cycles"] = llvm::json::Value(vpux::defaultNoValue.str());

        if (entries.find(key) != entries.end()) {
            Logger::global().warning("[PTC] Hash collision or duplicate op: key '{0}' already exists in the table; "
                                     "overwriting entry for op '{1}'",
                                     key, vpux::stringifyPrimaryLocation(op->getLoc()));
        }
        entries[key] = llvm::json::Value(std::move(entry));
    });

    json = llvm::json::Value(std::move(entries));
}

void writePrecomputedStrategyTableJSON(mlir::func::FuncOp func) {
#if defined(VPUX_DEVELOPER_BUILD)
    bool writePtc = false;
    std::string writePtcLoc = "precomputed_strategy_table.json";
    parseEnv("VPUX_PTC_WRITE_ENABLED", writePtc);
    parseEnv("VPUX_PTC_WRITE_LOCATION", writePtcLoc);
    if (!writePtc) {
        return;
    }

    static std::mutex ptcFileMutex;
    std::lock_guard<std::mutex> lock(ptcFileMutex);

    llvm::json::Value tableJson(nullptr);
    if (std::ifstream existing(writePtcLoc); existing.good()) {
        auto read = readManualStrategyJSON(writePtcLoc);
        if (read) {
            tableJson = std::move(*read);
        } else {
            Logger::global().warning("[PTC] Failed to parse existing JSON at '{0}': {1}; starting from an empty table",
                                     writePtcLoc, llvm::toString(read.takeError()));
        }
    }
    createPrecomputedStrategyTableJSON(tableJson, func);
    writeManualStrategyJSON(writePtcLoc, tableJson);
#else
    (void)func;
#endif  // defined(VPUX_DEVELOPER_BUILD)
}

}  // namespace vpux::VPU
