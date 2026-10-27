//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Debug reporting is implemented in the parser

#include "vpux/utils/profiling/parser/api.hpp"

#include "vpux/utils/profiling/parser/parser.hpp"
#include "vpux/utils/profiling/parser/records.hpp"

#include "vpux/utils/core/range.hpp"

#include <iomanip>
#include <memory>
#include <ostream>

using namespace vpux::profiling;
using vpux::indexed;

namespace {

void printDebugProfilingInfoSection(const RawProfilingRecords& records, std::ostream& outStream, size_t commonOffset) {
    using DebugFormattableRecordPtr = std::shared_ptr<DebugFormattableRecordMixin>;
    std::map<size_t, DebugFormattableRecordPtr> orderedRecords;
    for (const auto& record : records) {
        const auto debugRecordPtr = std::dynamic_pointer_cast<DebugFormattableRecordMixin>(record);
        VPUX_THROW_WHEN(debugRecordPtr == nullptr, "Expected formatable record");
        orderedRecords[debugRecordPtr->getInMemoryOffset()] = debugRecordPtr;
    }

    bool firstTime = true;
    const auto ostreamFlags = outStream.flags();
    for (const auto& offsetAndRecordIdx : orderedRecords | indexed) {
        const auto index = offsetAndRecordIdx.index();
        const auto offsetAndRecord = offsetAndRecordIdx.value();
        const auto record = offsetAndRecord.second;
        const auto taskOffset = offsetAndRecord.first * record->getDebugDataSize();
        if (firstTime) {
            outStream << std::setw(8) << "Index" << std::setw(8) << "Offset" << std::setw(14) << "Engine";
            record->printDebugHeader(outStream);
            outStream << std::left << std::setw(2) << ""
                      << "Task" << std::right << '\n';
            firstTime = false;
        }
        const auto taskGlobalOffset = commonOffset + taskOffset;
        const auto asRawRecord = std::dynamic_pointer_cast<RawProfilingRecord>(record);
        VPUX_THROW_WHEN(asRawRecord == nullptr, "Invalid record");
        outStream << std::setw(8) << std::dec << index << std::setw(8) << std::hex << taskGlobalOffset << std::setw(14)
                  << convertExecTypeToName(asRawRecord->getExecutorType());
        record->printDebugInfo(outStream);
        outStream << std::left << std::setw(2) << "" << asRawRecord->getTaskName() << std::right << '\n';
    }
    outStream << std::endl;
    outStream.flags(ostreamFlags);
}

void printDebugWorkpointsSetup(const RawProfilingData& rawProfData, std::ostream& outStream) {
    const auto workpointDbgInfos = rawProfData.workpoints;
    if (workpointDbgInfos.empty()) {
        return;
    }

    const auto ostreamFlags = outStream.flags();
    outStream << std::hex << std::setw(8) << "Index" << std::setw(8) << "Offset" << std::setw(14) << "Engine"
              << std::setw(17) << "PLL Value" << std::setw(15) << "CFGID" << std::endl;
    for (const auto& workpointDbgInfoIdx : workpointDbgInfos | indexed) {
        const auto index = workpointDbgInfoIdx.index();
        const auto workpointDbgInfo = workpointDbgInfoIdx.value();
        const auto workpointCfg = workpointDbgInfo.first;
        const auto offset = workpointDbgInfo.second;

        outStream << std::hex << std::setw(8) << index << std::setw(8) << offset << std::setw(14)
                  << convertExecTypeToName(ExecutorType::WORKPOINT) << std::setw(17) << workpointCfg.pllMultiplier
                  << std::setw(15) << workpointCfg.configId << std::endl;
    }
    outStream.flags(ostreamFlags);
}

RawProfilingRecords getTaskOfType(const RawProfilingData& rawRecords, ExecutorType type) {
    switch (type) {
    case ExecutorType::DMA_HW:
    case ExecutorType::DMA_SW:
        return rawRecords.dmaTasks;
    case ExecutorType::DPU:
        return rawRecords.dpuTasks;
    case ExecutorType::ACTSHAVE:
        return rawRecords.swTasks;
    default:
        VPUX_THROW("Unsupported executor type");
    }
}

}  // namespace

void vpux::profiling::writeDebugProfilingInfo(std::ostream& outStream, const uint8_t* blobData, size_t blobSize,
                                              const uint8_t* profData, size_t profSize) {
    const auto rawData = getRawProfilingTasks(blobData, blobSize, profData, profSize,
                                              /*ignoreSanitizationErrors =*/true);
    const auto rawRecords = rawData.rawRecords;
    for (const auto& typeAndOffset : rawRecords.parseOrder) {
        const auto tasks = getTaskOfType(rawRecords, typeAndOffset.first);
        printDebugProfilingInfoSection(tasks, outStream, typeAndOffset.second);
    }
    printDebugWorkpointsSetup(rawRecords, outStream);
}
