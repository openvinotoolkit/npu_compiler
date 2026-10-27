//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include <gtest/gtest.h>
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>

using namespace vpux;

static mlir::MLIRContext ctx;

mlir::Location getLoc(std::string name) {
    auto attr = mlir::StringAttr::get(&ctx, llvm::Twine(name));
    return mlir::NameLoc::get(attr);
}

mlir::Location getFusedLoc(mlir::ArrayRef<mlir::Location> locations, mlir::Attribute metadata) {
    return mlir::FusedLoc::get(&ctx, locations, metadata);
}

mlir::Location getFusedLoc(mlir::ArrayRef<std::string> parts, mlir::Attribute metadata) {
    mlir::SmallVector<mlir::Location> locs;
    for (const auto& part : parts) {
        locs.push_back(getLoc(part));
    }
    return getFusedLoc(locs, metadata);
}

mlir::Attribute getMetadata(const std::map<std::string, std::string>& metadata) {
    SmallVector<mlir::NamedAttribute> fields;
    for (auto keyValue : metadata) {
        fields.emplace_back(mlir::StringAttr::get(&ctx, keyValue.first), mlir::StringAttr::get(&ctx, keyValue.second));
    }
    auto metaDict = mlir::DictionaryAttr::get(&ctx, fields);
    return metaDict;
}

mlir::Attribute getMetadataWithType(std::string type) {
    return getMetadata({{"type", type}});
}

mlir::Location getFusedLoc(mlir::ArrayRef<std::string> parts) {
    return getFusedLoc(parts, nullptr);
}

mlir::Location getCallSiteLoc(mlir::Location callee, mlir::Location caller) {
    return mlir::CallSiteLoc::get(callee, caller);
}

TEST(MLIR_ProfilingUtils, stringifyPrimaryLocation) {
    mlir::SmallVector<std::pair<mlir::Location, std::string>> cases = {
            // without fusedLoc metadata
            {getFusedLoc({"Layer"}), "Layer"},
            {getFusedLoc({"Layer", "suffix"}), "Layer?suffix"},
            {getFusedLoc({"Layer", "suffix1", "suffix2"}), "Layer?suffix1/suffix2"},

            // with fusedLoc metadata
            {getFusedLoc({"Layer"}, getMetadataWithType("Type")), "Layer?t_Type"},
            {getFusedLoc({"Layer", "suffix"}, getMetadataWithType("Type")), "Layer?t_Type/suffix"},
            {getFusedLoc({"Layer", "suffix1", "suffix2"}, getMetadataWithType("Type")), "Layer?t_Type/suffix1/suffix2"},

            // with nested fusedLocs
            {getFusedLoc(
                     {
                             getFusedLoc({"Layer1"}, getMetadataWithType("Type1")),
                             getFusedLoc({"Layer2"}, getMetadataWithType("Type2")),
                     },
                     nullptr),
             "Layer1?t_Type1"},
            {getFusedLoc({getFusedLoc({"Layer1"}, getMetadataWithType("Type1")),
                          getFusedLoc({"Layer2"}, getMetadataWithType("Type2")), getLoc("suffix")},
                         nullptr),
             "Layer1?t_Type1/suffix"},
            {getFusedLoc({getFusedLoc({"Layer1"}, getMetadataWithType("Type1")),
                          getFusedLoc({"Layer2"}, getMetadataWithType("Type2")), getLoc("suffix1"), getLoc("suffix2")},
                         nullptr),
             "Layer1?t_Type1/suffix1/suffix2"},
            {getFusedLoc(
                     {
                             getFusedLoc({"Layer1"}, getMetadataWithType("Type1")),
                             getFusedLoc({"Layer2"}, getMetadataWithType("Type2")),
                     },
                     getMetadataWithType("Type3")),
             "Layer1?t_Type3"},
            {getFusedLoc(
                     {
                             getFusedLoc({"Layer1", "suffix1"}, getMetadataWithType("Type1")),
                             getFusedLoc({"Layer2", "suffix2"}, getMetadataWithType("Type2")),
                     },
                     getMetadataWithType("Type3")),
             "Layer1?t_Type3/suffix1"},

            // with CallSiteLoc, produced e.g. by the inliner pass when merging an outlined function back into its
            // caller: the Callee (the original op, carrying the real layer name/type) must stay primary, with the
            // Caller (the call site, e.g. main func loc with a 'partN_call' suffix) appended after
            {getCallSiteLoc(getFusedLoc({"Layer"}, getMetadataWithType("Type")), getFusedLoc({"main", "part1_call"})),
             "Layer?t_Type/main/part1_call"},
            {getFusedLoc(
                     {
                             getCallSiteLoc(getFusedLoc({"Layer"}, getMetadataWithType("Type")),
                                            getFusedLoc({"main", "part1_call"})),
                             getLoc("expanded"),
                     },
                     nullptr),
             "Layer?t_Type/main/part1_call/expanded"},

    };

    for (auto it : cases) {
        auto location = std::get<0>(it);
        auto expectedString = std::get<1>(it);
        auto actualString = vpux::stringifyPrimaryLocation(location);
        EXPECT_EQ(actualString, expectedString);
    }
}
