//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/thread_safe_accessors.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <gtest/gtest.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>

#include <functional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

using namespace vpux;

// Note: for the tests here, it is quite hard to have a "convincing" check that
// ensures the thread-safe primitives indeed work. the best effort attempt is
// made to make sure that tests are at least flaky, if not consistently failing
// in case there's an error in the implementation of the primitives.

TEST(MLIR_ThreadSafeAccessors, SimpleUsage) {
    SimpleThreadSafeAccessor<std::vector<int>> threadSafeWrapper;

    // Note: every access here to the vector locks the mutex implicitly
    ASSERT_TRUE(threadSafeWrapper->empty());
    threadSafeWrapper->push_back(42);
    ASSERT_FALSE(threadSafeWrapper->empty());
    ASSERT_EQ(threadSafeWrapper->back(), 42);
}

TEST(MLIR_ThreadSafeAccessors, SimpleConstUsage) {
    std::vector<int> init = {42};
    const SimpleThreadSafeAccessor<std::vector<int>> threadSafeWrapper(std::move(init));

    // only const methods can be used
    ASSERT_FALSE(threadSafeWrapper->empty());
    ASSERT_EQ(threadSafeWrapper->back(), 42);
}

TEST(MLIR_ThreadSafeAccessors, SimpleUsage_Handle) {
    SimpleThreadSafeAccessor<std::vector<int>> threadSafeWrapper;

    {
        // Note: a single implicit mutex lock for the whole scope
        auto handle = threadSafeWrapper.lock();
        auto& vector = *handle;
        ASSERT_TRUE(vector.empty());
        vector.push_back(42);
        ASSERT_FALSE(vector.empty());
        ASSERT_EQ(vector.back(), 42);
    }

    ASSERT_EQ(threadSafeWrapper->back(), 42);
}

TEST(MLIR_ThreadSafeAccessors, SimpleConstUsage_Handle) {
    std::vector<int> init = {42};
    const SimpleThreadSafeAccessor<std::vector<int>> threadSafeWrapper(init);

    // only const methods can be used
    auto handle = threadSafeWrapper.lock();
    auto& vector = *handle;
    ASSERT_FALSE(vector.empty());
    ASSERT_EQ(vector.back(), 42);
}

namespace {
void doThreadedAccessToTheSharedObject(size_t threadCount, std::function<void(size_t /*thread id*/)> callback) {
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (size_t i = 0; i < threadCount; ++i) {
        threads.emplace_back([i, &callback]() {
            callback(i);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
constexpr size_t THREAD_COUNT = 256;  // magic number for total threads
}  // namespace

// Note: if this test is flaky, there is a bug in SimpleThreadSafeAccessor.
TEST(MLIR_ThreadSafeAccessors, ThreadSafeUsage) {
    // I/O streams are "known" to be quite special when it comes to thread
    // safety. We should be guaranteed to at least see *interleaved* characters
    // if there's no thread-safety - according to
    // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2760.htm

    SimpleThreadSafeAccessor<std::stringstream> threadSafeStream;

    const std::string commonPrefix = "Thread ";
    // doing sufficiently many simultaneous accesses to the stream should result
    // in a "convincing" attempt to stress test the primitives.
    doThreadedAccessToTheSharedObject(THREAD_COUNT, [&](size_t threadId) {
        const std::string str = commonPrefix + std::to_string(threadId) + "\n";
        threadSafeStream->write(str.c_str(), str.size());
    });

    // all threads are done, we should be able to see well-formed data
    auto unnecessaryHandle = threadSafeStream.lock();
    const std::stringstream& filledStream = *unnecessaryHandle;
    const auto lines = splitAndTrimStringByDelimiter(filledStream.str(), '\n');

    llvm::DenseSet<size_t> paranoidCheckSet;
    ASSERT_EQ(lines.size(), THREAD_COUNT);
    for (const auto& line : lines) {
        llvm::StringRef lineRef(line);
        ASSERT_TRUE(lineRef.starts_with(commonPrefix));
        const auto currentThreadIdStr = lineRef.drop_front(commonPrefix.size()).trim();
        const size_t currentThreadId = std::stoull(currentThreadIdStr.str());
        ASSERT_TRUE(currentThreadId < THREAD_COUNT)
                << "Thread id must be within the range of [0, " << THREAD_COUNT - 1 << "]";
        ASSERT_FALSE(paranoidCheckSet.contains(currentThreadId)) << "Every thread id must appear exactly once";

        // at this point, "first line" is valid
        paranoidCheckSet.insert(currentThreadId);
    }
}

// Note: if this test is flaky, there is a bug in SimpleThreadSafeAccessor.
TEST(MLIR_ThreadSafeAccessors, ThreadSafeUsage_Handle) {
    SimpleThreadSafeAccessor<std::stringstream> threadSafeStream;

    const std::string commonPrefix = "Thread ";
    // doing sufficiently many simultaneous accesses to the stream should result
    // in a "convincing" attempt to stress test the primitives.
    doThreadedAccessToTheSharedObject(THREAD_COUNT, [&](size_t threadId) {
        auto handle = threadSafeStream.lock();

        std::stringstream& stream = *handle;
        stream << commonPrefix << threadId << "\n";
        stream << commonPrefix << threadId << " second line\n";
    });

    // all threads are done, we should be able to see well-formed data
    auto unnecessaryHandle = threadSafeStream.lock();
    const std::stringstream& filledStream = *unnecessaryHandle;
    const auto lines = splitAndTrimStringByDelimiter(filledStream.str(), '\n');

    llvm::DenseSet<size_t> paranoidCheckSet;
    ASSERT_EQ(lines.size(), THREAD_COUNT * 2);
    for (size_t i = 0; i < lines.size(); i += 2) {
        llvm::StringRef firstLine(lines[i]);
        ASSERT_TRUE(firstLine.starts_with(commonPrefix));
        const auto currentThreadIdStr = firstLine.drop_front(commonPrefix.size()).trim();
        const size_t currentThreadId = std::stoull(currentThreadIdStr.str());
        ASSERT_TRUE(currentThreadId < THREAD_COUNT)
                << "Thread id must be within the range of [0, " << THREAD_COUNT - 1 << "]";
        ASSERT_FALSE(paranoidCheckSet.contains(currentThreadId))
                << "Every thread id must appear exactly once in the first / second lines";

        // at this point, "first line" is valid
        paranoidCheckSet.insert(currentThreadId);

        llvm::StringRef secondLine(lines[i + 1]);
        ASSERT_TRUE(secondLine.starts_with(commonPrefix));
        secondLine = secondLine.drop_front(commonPrefix.size()).trim();
        ASSERT_TRUE(secondLine.starts_with(currentThreadIdStr)) << "Second line must share the same thread id";
        secondLine = secondLine.drop_front(currentThreadIdStr.size()).trim();
        ASSERT_EQ(secondLine, "second line") << "Second line must end with 'second line'";
    }
}
