//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_loader/vpux_loader.hpp>

#include <vpux_headers/buffer_specs.hpp>
#include <vpux_headers/device_buffer.hpp>

#include "vpux_hpi.hpp"

#include <cstdint>
#include <ctime>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#ifndef VPUX_ELF_UNIT_NAME
#define VPUX_ELF_UNIT_NAME Loader
#endif
#include <vpux_elf/utils/log.hpp>

using namespace elf;

namespace {

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Set verbosity"));

}  // namespace

class BuffManagerTest : public elf::BufferManager {
public:
    elf::DeviceBuffer allocate(const elf::BufferSpecs& buffSpecs) override;
    void deallocate(elf::DeviceBuffer& devBuffer) override;
    void lock(elf::DeviceBuffer&) override;
    void unlock(elf::DeviceBuffer&) override;
    size_t copy(elf::DeviceBuffer& to, const uint8_t* from, size_t count) override;
};

int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    if (verbose.getValue()) {
        Logger::setGlobalLevel(LogLevel::LOG_DEBUG);
    }

    // FIXME: lines below are treated as function declaration, these classes don't have default ctor
    // test were testing nothing in practice
    // VPUXLoader loader();
    // HostParsedInference hpi();
    VPUX_ELF_LOG(LogLevel::LOG_DEBUG, "loaderTest : DEBUG : Just checking if we can use VPUXLoader headers");
    VPUX_ELF_LOG(LogLevel::LOG_INFO, "loaderTest : INFO : Just checking if we can use VPUXLoader headers");

    return 0;
}
