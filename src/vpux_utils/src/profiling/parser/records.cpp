//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/parser/records.hpp"
#include "vpux/utils/profiling/utils.hpp"

#include <sstream>

namespace {
std::string getHexString(uint64_t number) {
    std::stringstream ss;
    ss << "0x" << std::hex << number;

    return ss.str();
}
}  // namespace

namespace vpux::profiling {

CustomArgsVector RawProfilingDMA40Record::getCustomArgs(FrequenciesSetup frequenciesSetup) const {
    CustomArgsVector customArgs;
    customArgs.push_back({"Address:", getHexString(_record.desc_addr)});
    customArgs.push_back({"Time to ready:", formatDuration(convertTicksToNs(_record.ready_time - _record.fetch_time,
                                                                            frequenciesSetup.profClk))});
    customArgs.push_back({"Time to start:", formatDuration(convertTicksToNs(_record.start_time - _record.ready_time,
                                                                            frequenciesSetup.profClk))});
    customArgs.push_back({"Transfer time:", formatDuration(convertTicksToNs(_record.wdone_time - _record.start_time,
                                                                            frequenciesSetup.profClk))});
    customArgs.push_back({"Time to finish:", formatDuration(convertTicksToNs(_record.finish_time - _record.wdone_time,
                                                                             frequenciesSetup.profClk))});
    customArgs.push_back({"Link agent id:", std::to_string(_record.la_id)});
    customArgs.push_back({"Channel id:", std::to_string(_record.ch_id)});
    customArgs.push_back({"Read stall cycles:", std::to_string(_record.rstall_cnt)});
    customArgs.push_back({"Write stall cycles:", std::to_string(_record.wstall_cnt)});
    customArgs.push_back({"Total bytes:", std::to_string(_record.twbytes_cnt)});
    customArgs.push_back({"Total cycles:", std::to_string(_record.chcycle_cnt)});
    return customArgs;
}

CustomArgsVector RawProfilingACTRecord::getCustomArgs(FrequenciesSetup) const {
    CustomArgsVector customArgs;
    customArgs.push_back({"Total cycles:", std::to_string(_data.clockCycles)});
    customArgs.push_back({"Active cycles:", std::to_string(_data.executedInstructions)});
    customArgs.push_back({"Stall cycles:", std::to_string(_data.clockCycles - _data.executedInstructions)});
    return customArgs;
}

CustomArgsVector RawProfilingACTExRecord::getCustomArgs(FrequenciesSetup) const {
    CustomArgsVector customArgs;
    customArgs.push_back({"Total cycles:", std::to_string(_data.clockCycles)});
    customArgs.push_back({"Active cycles:", std::to_string(_data.executedInstructions)});
    customArgs.push_back({"Stall cycles:", std::to_string(_data.clockCycles - _data.executedInstructions)});
    customArgs.push_back({"LSU0 stalls:", std::to_string(_data.lsu0Stalls)});
    customArgs.push_back({"LSU1 stalls:", std::to_string(_data.lsu1Stalls)});
    customArgs.push_back({"Instruction stalls:", std::to_string(_data.instStalls)});
    return customArgs;
}

}  // namespace vpux::profiling
