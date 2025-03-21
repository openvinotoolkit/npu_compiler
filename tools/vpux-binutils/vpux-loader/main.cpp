//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <chrono>
#include <ctime>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <vpux_hpi.hpp>

#include "blob_scanner.hpp"
#include "hpi_runner.hpp"
#include "io_container.hpp"

using namespace elf;
using namespace std;
using namespace chrono;

namespace {

llvm::cl::OptionCategory appOptionCategory("Application options",
                                           "Options for controlling the execution of the npu-loader application");

llvm::cl::opt<std::string> appArgBlobPathAndName(llvm::cl::Positional, llvm::cl::Required,
                                                 llvm::cl::value_desc("Input blob full path + name"),
                                                 llvm::cl::init("model.blob"),
                                                 llvm::cl::desc("<Input blob full path + name>"),
                                                 llvm::cl::cat(appOptionCategory));

llvm::cl::opt<std::string> appArgArchName("arch",
                                          llvm::cl::desc("Arch name as defined by ELF library (default: VPUX37XX)"),
                                          llvm::cl::init("VPUX37XX"), llvm::cl::cat(appOptionCategory));

enum class RunMode {
    SimpleLoad = 0,
    SimpleClone,
};

llvm::cl::opt<RunMode> appArgRunMode(
        "mode", llvm::cl::desc("Run mode"), llvm::cl::init(RunMode::SimpleLoad),
        llvm::cl::values(clEnumValN(RunMode::SimpleLoad, "simple_load",
                                    "Run a simple load of a single HostParsedInference and get memory consumption "
                                    "projection and actual memory consumption"),
                         clEnumValN(RunMode::SimpleClone, "simple_clone",
                                    "Run a clone after deleting access to the blob")),
        llvm::cl::cat(appOptionCategory));

llvm::cl::opt<AccessManagerType> appArgAccessManagerType(
        "access", llvm::cl::desc("AccessManager type"), llvm::cl::init(AccessManagerType::FSAccessManager),
        llvm::cl::values(clEnumValN(AccessManagerType::DDRAccessManager, "DDR", "DDRAccessManager"),
                         clEnumValN(AccessManagerType::FSAccessManager, "FS", "FSAccessManager")),
        llvm::cl::cat(appOptionCategory));

llvm::cl::opt<bool> appArgVerbose("v", llvm::cl::desc("Enable/Disable verbosity"), llvm::cl::cat(appOptionCategory));

}  // namespace

template <typename HPIRunnerDerived>
void run(HPIRunner<HPIRunnerDerived>&& runner) {
    runner.run();
}

struct SimpleLoadRunner : public HPIRunner<SimpleLoadRunner> {
    SimpleLoadRunner(): HPIRunner(appArgArchName, appArgBlobPathAndName, appArgAccessManagerType) {
    }

    void runImpl() {
        // Get a memory consumption projection from the blob
        BlobScanner blobScanner(mAccessManager.get(), defaultProcMap);
        blobScanner.printResult();
        llvm::outs() << llvm::formatv("\n");

        llvm::outs() << llvm::formatv("Loading HPI...\n");

        auto start = high_resolution_clock::now();

        HostParsedInference hpi(mHpiBufferManager.get(), mAccessManager.get(), mHpiConfig);
        hpi.load();

        auto end = high_resolution_clock::now();
        llvm::outs() << llvm::formatv("HPI loaded in {0} ms\n\n", duration_cast<milliseconds>(end - start).count());

        IOBuffersContainer ioContainer(mIoBufferManager, hpi.getInputBuffers(), hpi.getOutputBuffers(),
                                       hpi.getProfBuffers());

        hpi.applyInputOutput(ioContainer.getInputBuffers(), ioContainer.getOutputBuffers(),
                             ioContainer.getProfilingBuffers());

        llvm::outs() << llvm::formatv(
                "Projected NPU memory for HPI: {0} bytes\n",
                blobScanner.getRequirementsByAllocationType().getTotalRequired() + hpi.getHPISize());

        llvm::outs() << llvm::formatv("Total NPU buffers tracked by HPI object: {0}\n",
                                      hpi.getAllocatedBuffers().size());
        llvm::outs() << llvm::formatv("\n");

        mHpiBufferManager->printAllocationStats();
    }
};

class SimpleCloneRunner : public HPIRunner<SimpleCloneRunner> {
public:
    SimpleCloneRunner(): HPIRunner(appArgArchName, appArgBlobPathAndName, appArgAccessManagerType) {
    }

    void runImpl() {
        llvm::outs() << llvm::formatv("Loading first HPI...\n");
        HostParsedInference hpi(mHpiBufferManager.get(), mAccessManager.get(), mHpiConfig);
        hpi.load();
        llvm::outs() << llvm::formatv("First HPI loaded\n\n");

        // Delete AccessManager
        mAccessManager = nullptr;

        // IO bindings must be possible after blob release
        IOBuffersContainer ioContainer(mIoBufferManager, hpi.getInputBuffers(), hpi.getOutputBuffers(),
                                       hpi.getProfBuffers());
        hpi.applyInputOutput(ioContainer.getInputBuffers(), ioContainer.getOutputBuffers(),
                             ioContainer.getProfilingBuffers());

        mHpiBufferManager->printAllocationStats();
        llvm::outs() << llvm::formatv("Loading second HPI...\n");

        // Cloning must still work after AccessManager was deleted
        HostParsedInference hpiClone(hpi);

        llvm::outs() << llvm::formatv("Second HPI loaded\n\n");

        mHpiBufferManager->printAllocationStats();
        hpiClone.getMetadata();
    }
};

int main(int argc, char* argv[]) {
    llvm::cl::HideUnrelatedOptions(appOptionCategory);
    llvm::cl::ParseCommandLineOptions(argc, argv);

    if (appArgVerbose) {
        Logger::setGlobalLevel(LogLevel::LOG_DEBUG);
    }

    switch (appArgRunMode) {
    case RunMode::SimpleLoad: {
        run(SimpleLoadRunner());
        break;
    }
    case RunMode::SimpleClone: {
        run(SimpleCloneRunner());
        break;
    }
    default: {
        // Default should be unreachable since LLVM CL should sanitize the arg
        throw(std::runtime_error("Unknown test type"));
    }
    }

    return 0;
}
