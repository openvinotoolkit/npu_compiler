//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "float_utils.hpp"

TEST(float16, conversions) {
    vpux::type::float16 f16;
    const char* source_string;
    std::string f16_string;

    // 1.f
    source_string = "0  01111  00 0000 0000";
    f16 = vpux::type::utils::bits_to_float16(source_string);
    EXPECT_EQ(f16, vpux::type::float16(1.0));
    f16_string = vpux::type::utils::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), 1.0);

    // -1.f
    source_string = "1  01111  00 0000 0000";
    f16 = vpux::type::utils::bits_to_float16(source_string);
    EXPECT_EQ(f16, vpux::type::float16(-1.0));
    f16_string = vpux::type::utils::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), -1.0);

    // 0.f
    source_string = "0  00000  00 0000 0000";
    f16 = vpux::type::utils::bits_to_float16(source_string);
    EXPECT_EQ(f16, vpux::type::float16(0.0));
    f16_string = vpux::type::utils::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), 0.0);

    // 1.5f
    source_string = "0  01111  10 0000 0000";
    f16 = vpux::type::utils::bits_to_float16(source_string);
    EXPECT_EQ(f16, vpux::type::float16(1.5));
    f16_string = vpux::type::utils::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), 1.5);
}

TEST(float16, assigns) {
    vpux::type::float16 f16;
    f16 = 2.0;
    EXPECT_EQ(f16, vpux::type::float16(2.0));

    std::vector<float> f32vec{1.0, 2.0, 4.0};
    std::vector<vpux::type::float16> f16vec;
    std::copy(f32vec.begin(), f32vec.end(), std::back_inserter(f16vec));
    for (size_t i = 0; i < f32vec.size(); ++i) {
        EXPECT_EQ(f32vec.at(i), f16vec.at(i));
    }

    float f32arr[] = {1.0, 2.0, 4.0};
    vpux::type::float16 f16arr[sizeof(f32arr)];
    for (size_t i = 0; i < sizeof(f32arr) / sizeof(f32arr[0]); ++i) {
        f16arr[i] = f32arr[i];
        EXPECT_EQ(f32arr[i], f16arr[i]);
    }
}

TEST(float16, values) {
    EXPECT_EQ(static_cast<vpux::type::float16>(vpux::type::utils::FloatUnion(0, 112 - 8, (1 << 21) + 0).f).to_bits(),
              vpux::type::float16(0, 0, 2).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(vpux::type::utils::FloatUnion(0, 112 - 8, (1 << 21) + 1).f).to_bits(),
              vpux::type::float16(0, 0, 3).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(1.0 / (256.0 * 65536.0)).to_bits(),
              vpux::type::float16(0, 0, 1).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(1.5 / (256.0 * 65536.0)).to_bits(),
              vpux::type::float16(0, 0, 2).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(1.25 / (256.0 * 65536.0)).to_bits(),
              vpux::type::float16(0, 0, 1).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(1.0 / (128.0 * 65536.0)).to_bits(),
              vpux::type::float16(0, 0, 2).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(1.5 / (128.0 * 65536.0)).to_bits(),
              vpux::type::float16(0, 0, 3).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(1.25 / (128.0 * 65536.0)).to_bits(),
              vpux::type::float16(0, 0, 2).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(std::numeric_limits<float>::infinity()).to_bits(),
              vpux::type::float16(0, 0x1F, 0).to_bits());
    EXPECT_EQ(static_cast<vpux::type::float16>(-std::numeric_limits<float>::infinity()).to_bits(),
              vpux::type::float16(1, 0x1F, 0).to_bits());
    EXPECT_TRUE(std::isnan(static_cast<vpux::type::float16>(std::numeric_limits<float>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(static_cast<vpux::type::float16>(std::numeric_limits<float>::signaling_NaN())));
    EXPECT_EQ(static_cast<vpux::type::float16>(2.73786e-05).to_bits(), 459);
    EXPECT_EQ(static_cast<vpux::type::float16>(3.87722e-05).to_bits(), 650);
    EXPECT_EQ(static_cast<vpux::type::float16>(-0.0223043).to_bits(), 42422);
    EXPECT_EQ(static_cast<vpux::type::float16>(5.10779e-05).to_bits(), 857);
    EXPECT_EQ(static_cast<vpux::type::float16>(-5.10779e-05).to_bits(), 0x8359);
    EXPECT_EQ(static_cast<vpux::type::float16>(-2.553895e-05).to_bits(), 0x81ac);
    EXPECT_EQ(static_cast<vpux::type::float16>(-0.0001021558).to_bits(), 0x86b2);
    EXPECT_EQ(static_cast<vpux::type::float16>(5.960464477539063e-08).to_bits(), 0x01);
    EXPECT_EQ(static_cast<vpux::type::float16>(8.940696716308594e-08).to_bits(), 0x02);
    EXPECT_EQ(static_cast<vpux::type::float16>(65536.0).to_bits(), 0x7c00);
    EXPECT_EQ(static_cast<vpux::type::float16>(65519.0).to_bits(), 0x7bff);
    EXPECT_EQ(static_cast<vpux::type::float16>(65520.0).to_bits(), 0x7c00);

    // underflow cases
    EXPECT_EQ(static_cast<vpux::type::float16>(3.45023e-14).to_bits(), 0x0000);  // 0x0000 = 0
    EXPECT_EQ(static_cast<vpux::type::float16>(-4.2796e-14).to_bits(), 0x8000);  // 0x8000 = -0

    // iszero cases
    EXPECT_TRUE(vpux::type::iszero(vpux::type::float16(0, 0, 0)));
    EXPECT_TRUE(vpux::type::iszero(vpux::type::float16(1, 0, 0)));
    EXPECT_FALSE(vpux::type::iszero(vpux::type::float16(0, 1, 0)));
    EXPECT_FALSE(vpux::type::iszero(vpux::type::float16(0, 0, 1)));
    EXPECT_FALSE(vpux::type::iszero(vpux::type::float16(1, 1, 0)));
    EXPECT_FALSE(vpux::type::iszero(vpux::type::float16(1, 0, 1)));

    // float16 operator ==
    EXPECT_TRUE(vpux::type::float16(0, 0, 0) == vpux::type::float16(0, 0, 0));         // +0 == +0
    EXPECT_TRUE(vpux::type::float16(1, 0, 0) == vpux::type::float16(1, 0, 0));         // -0 == -0
    EXPECT_TRUE(vpux::type::float16(0, 0, 0) == vpux::type::float16(1, 0, 0));         // +0 == -0
    EXPECT_TRUE(vpux::type::float16(0, 0x1F, 0) == vpux::type::float16(0, 0x1F, 0));   // +Inf == +Inf
    EXPECT_TRUE(vpux::type::float16(1, 0x1F, 0) == vpux::type::float16(1, 0x1F, 0));   // -Inf == -Inf
    EXPECT_FALSE(vpux::type::float16(0, 0x1F, 0) == vpux::type::float16(1, 0x1F, 0));  // +Inf != -Inf
    EXPECT_FALSE(vpux::type::float16(0, 0x1F, 1) == vpux::type::float16(0, 0x1F, 1));  // NaN != NaN
    EXPECT_FALSE(vpux::type::float16(0, 0x1F, 1) == vpux::type::float16(1, 0x1F, 1));  // NaN != NaN
    EXPECT_TRUE(vpux::type::float16(0, 1, 1) == vpux::type::float16(0, 1, 1));
    EXPECT_FALSE(vpux::type::float16(0, 1, 1) == vpux::type::float16(1, 1, 1));
    EXPECT_FALSE(vpux::type::float16(0, 1, 1) == vpux::type::float16(0, 1, 2));
    EXPECT_FALSE(vpux::type::float16(0, 1, 1) == vpux::type::float16(0, 2, 1));

    // float16 operator !=
    EXPECT_TRUE(vpux::type::float16(0, 0x1F, 1) != vpux::type::float16(0, 0x1F, 1));  // NaN != NaN
    EXPECT_TRUE(vpux::type::float16(1, 0x1F, 1) != vpux::type::float16(1, 0x1F, 1));  // NaN != NaN
    EXPECT_TRUE(vpux::type::float16(0, 0x1F, 1) != vpux::type::float16(0, 1, 1));     // NaN != normal float16
    EXPECT_TRUE(vpux::type::float16(0, 0x1F, 1) != vpux::type::float16(0, 0x1F, 0));  // NaN != +Inf
    EXPECT_TRUE(vpux::type::float16(0, 0x1F, 1) != vpux::type::float16(1, 0x1F, 0));  // NaN != -Inf
    EXPECT_TRUE(vpux::type::float16(0, 0x1F, 1) != vpux::type::float16(0, 0, 0));     // NaN != +0
}
