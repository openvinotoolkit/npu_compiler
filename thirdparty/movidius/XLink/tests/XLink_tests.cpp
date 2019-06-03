//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <iostream>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include <XLinkPrivateDefines.h>
#include "XLink.h"

#define MAX_NAME_LENGTH 16
#define MAX_DEVICES     32
#define MAX_PATH        255

#define MYRIADX         0x2485
#define MYRIAD2         0x2150
#define MYRIAD_BOOTED   0xf63b
#define MYRIAD_UNBOOTED -1

static XLinkGlobalHandler_t globalHandler;

class XLinkTests : public ::testing::Test {
public:
    static void SetUpTestCase() {
        ASSERT_EQ(X_LINK_SUCCESS, XLinkInitialize(&globalHandler));
        // Waiting for initialization
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
protected:
    virtual ~XLinkTests() {}

    void getFirmwarePath(char* devAddr, char* firmwarePath) {
        char* p = strchr(devAddr, '-');
        if (p == nullptr) {
            EXPECT_TRUE(false) << "Invalid device address";
        }
#if (!defined(_WIN32) && !defined(_WIN64))
        snprintf(firmwarePath, 40, "./lib/MvNCAPI%s.mvcmd", p);
#else
        snprintf(firmwarePath, 40, "./MvNCAPI%s.mvcmd", p);
#endif  // #if (!defined(_WIN32) && !defined(_WIN64))
    }

    void bootAnyDevice() {
        deviceDesc_t deviceDesc = {};
        char firmwarePath[MAX_PATH];

        // Get device name
        ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &deviceDesc, MYRIAD_UNBOOTED, X_LINK_USB_VSC));
        getFirmwarePath(deviceDesc.name, firmwarePath);

        printf("Would boot (%s) device with firmware (%s) \n", deviceDesc.name, firmwarePath);

        // Boot device
        ASSERT_EQ(X_LINK_SUCCESS, XLinkBootRemote(&deviceDesc, firmwarePath));
        // FIXME: need to find a way to avoid this sleep
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Check, that device booted
        deviceDesc_t bootedDeviceDesc = {};
        ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &bootedDeviceDesc, MYRIAD_BOOTED, X_LINK_USB_VSC));
    }

    void closeDevice(char* bootedName) {
        XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
        handler->devicePath = bootedName;
        ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
        // FIXME: need to find a way to avoid this sleep
        std::this_thread::sleep_for(std::chrono::seconds(1));

        ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
        free(handler);

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    void closeDeviceWithHandler(XLinkHandler_t* handler) {
        ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
        free(handler);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

};

TEST_F(XLinkTests, CanBootConnectAndResetDevice) {
    deviceDesc_t deviceDesc = {};
    char firmwarePath[MAX_PATH];

    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &deviceDesc, MYRIAD_UNBOOTED, X_LINK_USB_VSC));
    getFirmwarePath(deviceDesc.name, firmwarePath);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkBootRemote(&deviceDesc, firmwarePath));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));

    deviceDesc_t bootedDesc = {};
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (X_LINK_SUCCESS == XLinkGetDeviceName(i, &bootedDesc, MYRIAD_BOOTED, X_LINK_USB_VSC)) {
            break;
        }
    }

    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    handler->protocol = bootedDesc.protocol;
    handler->devicePath = bootedDesc.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    free(handler);
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

class XLinkOpenStreamTests : public XLinkTests {
protected:
    virtual ~XLinkOpenStreamTests() {

    }
    void SetUp() override {
        deviceDesc_t deviceDesc = {};
        ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &deviceDesc, MYRIADX, X_LINK_USB_VSC));
        ASSERT_EQ(X_LINK_SUCCESS, XLinkBootRemote(&deviceDesc, "./lib/MvNCAPI-ma2480.mvcmd"));

        std::this_thread::sleep_for(std::chrono::seconds(1));

        deviceDesc_t bootedDesc = {};
        for (int i = 0; i < MAX_DEVICES; i++) {
            if (X_LINK_SUCCESS == XLinkGetDeviceName(i, &bootedDesc, MYRIAD_BOOTED, X_LINK_USB_VSC)) {
                break;
            }
        }

        handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
        handler->protocol = bootedDesc.protocol;
        handler->devicePath = bootedDesc.name;
        ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    void TearDown() override {
        ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
        free(handler);
        // FIXME: need to find a way to avoid this sleep
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    XLinkHandler_t *handler;
};

TEST_F(XLinkOpenStreamTests, CanOpenAndCloseStream) {
    streamId_t stream = XLinkOpenStream(handler->linkId, "mySuperStream", 1024);
    ASSERT_NE(INVALID_STREAM_ID, stream);
    ASSERT_NE(INVALID_STREAM_ID_OUT_OF_MEMORY, stream);
    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream));
}

TEST_F(XLinkOpenStreamTests, CannotOpenStreamMoreThanMemoryOnDevice) {
    const int _512MB = 512 * 1024 * 1024;
    streamId_t stream = XLinkOpenStream(handler->linkId, "mySuperStream", _512MB);
    ASSERT_EQ(INVALID_STREAM_ID_OUT_OF_MEMORY, stream);
}

// FIXME: the test doesn't work
// TODO: is it correct behavior, should we accept the same names
TEST_F(XLinkOpenStreamTests, DISABLED_CannotOpenTwoStreamsWithTheSameName) {
    const int _1KB = 1 * 1024;
    const char streamName[] = "mySuperStream";
    streamId_t stream0 = XLinkOpenStream(handler->linkId, streamName, _1KB);
    ASSERT_NE(INVALID_STREAM_ID, stream0);

    streamId_t stream1 = XLinkOpenStream(handler->linkId, streamName, _1KB);
    ASSERT_EQ(INVALID_STREAM_ID, stream1);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream0));
}

// FIXME: XLinkOpenStream doesn't allocate any memory on device
TEST_F(XLinkOpenStreamTests, DISABLED_CannotOpenStreamsMoreThanMemoryOnDevice) {
    const int _256MB = 256 * 1024 * 1024;
    streamId_t stream0 = XLinkOpenStream(handler->linkId, "mySuperStream0", _256MB);
    ASSERT_NE(INVALID_STREAM_ID, stream0);

    streamId_t stream1 = XLinkOpenStream(handler->linkId, "mySuperStream1", _256MB);
    ASSERT_EQ(INVALID_STREAM_ID, stream1);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream0));
    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream1));
}

/**
 * @brief XLinkGetDeviceName function tests
 */
class XLinkGetDeviceNameTests : public XLinkTests {
protected:
    ~XLinkGetDeviceNameTests() override = default;
};

// TODO Can compose list of all devices

/**
 * @brief XLinkGetDeviceName should return error if index argument is invalid
 */
TEST_F(XLinkGetDeviceNameTests, ReturnErrorOnIncorrectIndex) {
    deviceDesc_t deviceDesc = {};
    deviceDesc.name[0] = '\0';
    ASSERT_EQ(X_LINK_ERROR, XLinkGetDeviceName(-1, &deviceDesc, 0, X_LINK_USB_VSC));
    ASSERT_TRUE(strlen(deviceDesc.name) == 0);
}

/**
 * @brief XLinkGetDeviceName should return device name in AUTO_PID mode (pid = 0)
 */
TEST_F(XLinkGetDeviceNameTests, ReturnAnyDeviceName) {
    deviceDesc_t deviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &deviceDesc, 0, X_LINK_USB_VSC));
    ASSERT_TRUE(strlen(deviceDesc.name) > 2);
}

/**
 * @brief XLinkGetDeviceName should return M2 device name if pid = MYRIAD2 (0x2150)
 */
TEST_F(XLinkGetDeviceNameTests, ReturnCorrectM2DeviceName) {
    deviceDesc_t deviceDesc = {};

    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &deviceDesc, MYRIAD2, X_LINK_USB_VSC));
    ASSERT_TRUE(strstr(deviceDesc.name, "ma2450") != nullptr);
}

/**
 * @brief XLinkGetDeviceName should return MX device name if pid = MYRIADX (0x2485)
 */
TEST_F(XLinkGetDeviceNameTests, ReturnCorrectMXDeviceName) {
    deviceDesc_t deviceDesc = {};

    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &deviceDesc, MYRIADX, X_LINK_USB_VSC));
    ASSERT_TRUE(strstr(deviceDesc.name, "ma2480") != nullptr);
}

/**
 * @brief XLinkGetDeviceName should return booted MX device name if pid = MYRIAD_BOOTED (0xf63b)
 */
TEST_F(XLinkGetDeviceNameTests, ReturnCorrectBootedDeviceName) {
    bootAnyDevice();

    deviceDesc_t bootedDeviceDescr = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &bootedDeviceDescr, MYRIAD_BOOTED, X_LINK_USB_VSC));
    ASSERT_TRUE(strstr(bootedDeviceDescr.name, "ma2480") == nullptr);
    ASSERT_TRUE(strstr(bootedDeviceDescr.name, "ma2450") == nullptr);

    closeDevice(bootedDeviceDescr.name);
}

/**
 * @brief XLinkResetAll function tests
 */
class XLinkResetAllTests : public XLinkTests {
protected:
    ~XLinkResetAllTests() override = default;
};

/**
 * @brief XLinkResetAll function should reset all booted devices
 */
TEST_F(XLinkResetAllTests, ResetBootedDevice) {
    // TODO Boot all available devices
    bootAnyDevice();

    // Without connection to device XLinkResetAll doesn't work
    // Connect to device
    deviceDesc_t bootedDeviceDescr = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &bootedDeviceDescr, MYRIAD_BOOTED, X_LINK_USB_VSC));

    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    handler->protocol = bootedDeviceDescr.protocol;
    handler->devicePath = bootedDeviceDescr.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));

    // Try to reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetAll());
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // No one booted device should be found
    deviceDesc_t afterResetBootedDescr = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND, XLinkGetDeviceName(0, &afterResetBootedDescr, MYRIAD_BOOTED, X_LINK_USB_VSC));
}

/**
 * @brief XLinkConnect function tests
 */
class XLinkConnectTests : public XLinkTests {
protected:
    ~XLinkConnectTests() override = default;
};

TEST_F(XLinkConnectTests, InvalidHanler) {
    ASSERT_EQ(X_LINK_ERROR, XLinkConnect(nullptr));
}

TEST_F(XLinkConnectTests, ConnectToDevice) {
    bootAnyDevice();

    deviceDesc_t bootedDeviceDescr = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &bootedDeviceDescr, MYRIAD_BOOTED, X_LINK_USB_VSC));

    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    handler->protocol = bootedDeviceDescr.protocol;
    handler->devicePath = bootedDeviceDescr.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));

    closeDeviceWithHandler(handler);
}

class XLinkBootRemoteTests: public XLinkTests {
public:
    ~XLinkBootRemoteTests() override = default;
};

TEST_F(XLinkBootRemoteTests, USBDeviceNameChangedAfterBoot) {
    deviceDesc_t unbootedDeviceDescr = {};
    char firmwarePath[MAX_PATH];

    // Get device name
    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &unbootedDeviceDescr, MYRIAD_UNBOOTED, X_LINK_USB_VSC));
    getFirmwarePath(unbootedDeviceDescr.name, firmwarePath);

    // Boot device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBootRemote(&unbootedDeviceDescr, firmwarePath));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Booted device appear
    deviceDesc_t bootedDeviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkGetDeviceName(0, &bootedDeviceDesc, MYRIAD_BOOTED, X_LINK_USB_VSC));

    // Previous device don't disappear
    bool before_booted_found = false;
    deviceDesc_t deviceDesc = {};
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (X_LINK_SUCCESS == XLinkGetDeviceName(i, &deviceDesc, MYRIAD_UNBOOTED, X_LINK_USB_VSC)) {
            if (strcmp(deviceDesc.name, unbootedDeviceDescr.name) == 0) {
                before_booted_found = true;
            }
            break;
        }
    }

    ASSERT_FALSE(before_booted_found);

    closeDevice(bootedDeviceDesc.name);
}