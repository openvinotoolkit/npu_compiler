#!/usr/bin/env python3
#
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import json
from dataclasses import dataclass
import typing as tp


def parse_args():
    parser = argparse.ArgumentParser(description="Tool generate model_strategies.cpp from multiple strategies")
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='Required. List of strategy files')

    return parser.parse_args()


def wrapBrackets(x):
    return '{' + x + '}'


@dataclass
class GeneratedStrategyEntry:
    modelHashVar: str
    modelHashValue: str
    modelStrategyVar: str
    generatedLines: tp.List[str]

    @staticmethod
    def loadFromFile(filename: str, entryId: int):
        with open(filename) as f:
            modelStrategy = json.load(f)

        strategies = modelStrategy['Op']
        modelHashValue = modelStrategy['ModelHash']
        modelHashVar = f'MODEL_{entryId}_HASH'
        modelStrategyVar = f'MODEL_{entryId}_STRATEGIES'

        declareHash = f'static const StringRef {modelHashVar} = "{modelHashValue}";'
        layerStrategies = []
        for layerHash, layerStrategy in strategies.items():
            hexHash = format(int(layerHash), '#018x')
            strat = f'S::{layerStrategy["multiClusterStrategy"]}' if layerStrategy["multiClusterStrategy"] != 'NONE' else 'std::nullopt'
            layerStrategies.append(
                wrapBrackets(f'{hexHash}, {strat}')
            )

        content = wrapBrackets(', '.join(layerStrategies))
        declareStategies = f'static const mlir::DenseMap<size_t, std::optional<VPU::MultiClusterStrategy>> {modelStrategyVar} = {content};'
        generatedLines = [declareHash, declareStategies, '']

        return GeneratedStrategyEntry(modelHashVar, modelHashValue, modelStrategyVar, generatedLines)


HEADER = """
//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <optional>

using namespace vpux;

namespace {

using S = vpux::VPU::MultiClusterStrategy;

"""


def generateBranch(key, value):
    return "if (modelHash == " + key + ") { return " + value + "; }"


def generateFooter(generatedStrategies: tp.List[GeneratedStrategyEntry]) -> str:
    maybeGetStrategyForBody = ' else '.join(
        [generateBranch(s.modelHashVar, s.modelStrategyVar) for s in generatedStrategies])
    footer = """
};
std::optional<mlir::DenseMap<size_t, std::optional<VPU::MultiClusterStrategy>>> vpux::maybeGetStrategyFor(StringRef modelHash) {
"""
    footer += maybeGetStrategyForBody + "\n return std::nullopt;\n}\n"
    isStrategyPreConfiguredFunc = """
bool vpux::isStrategyPreConfigured(StringRef modelHash) {
    const std::vector<StringRef> CONFIGURED_HASHES = DUMMY_REPLACE_KEY ;
    for(const StringRef hash: CONFIGURED_HASHES) {
        if (hash == modelHash) {
            return true;
        }
    }
    return false;
}
""".replace('DUMMY_REPLACE_KEY', wrapBrackets(', '.join(s.modelHashVar for s in generatedStrategies)))
    footer += isStrategyPreConfiguredFunc
    return footer


if __name__ == '__main__':
    args = parse_args()
    modelHashes = set()
    entries = []
    for strategyFile in args.inputs:
        entry = GeneratedStrategyEntry.loadFromFile(strategyFile, len(modelHashes))
        if entry.modelHashValue in modelHashes:
            raise RuntimeError(f'{strategyFile} describes model, which is already in generation list')
        modelHashes.add(entry.modelHashValue)
        entries.append(entry)
    body = '\n'.join(sum([e.generatedLines for e in entries], []))
    content = HEADER + body + generateFooter(entries)
    with open('model_strategies.cpp', 'w') as f:
        f.write(content)
