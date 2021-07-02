// {% copyright %}
///
/// @file      swcFrameTypes.h
/// @copyright All code copyright Movidius Ltd 2013, all rights reserved
///            For License Warranty see: common/license.txt
///
/// @brief     Frametypes used in Myriad common code
///

#ifndef _SWC_FRAME_TYPES_H_
#define _SWC_FRAME_TYPES_H_

#include <stdint.h>
#include <iostream>

typedef enum frameTypes {
    YUV422i,  // interleaved 8 bit
    YUV444p,  // planar 4:4:4 format
    YUV420p,  // planar 4:2:0 format
    YUV422p,  // planar 8 bit
    YUV400p,  // 8-bit greyscale
    RGBA8888, // RGBA interleaved stored in 32 bit word
    RGB888,   // Planar 8 bit RGB data
    RGB888p,  // Planar 8 bit RGB data
    RGB888i,  // Interleaved 8 bit RGB data
    BGR888p,  // Planar 8 bit BGR data
    BGR888i,  // Interleaved 8 bit BGR data
    LUT2,     // 1 bit  per pixel, Lookup table (used for graphics layers)
    LUT4,     // 2 bits per pixel, Lookup table (used for graphics layers)
    LUT16,    // 4 bits per pixel, Lookup table (used for graphics layers)
    RAW16,    // save any raw type (8, 10, 12bit) on 16 bits
    RAW14,    // 14bit value in 16bit storage
    RAW12,    // 12bit value in 16bit storage
    RAW10,    // 10bit value in 16bit storage
    RAW8,
    PACK10, // SIPP 10bit packed format
    PACK12, // SIPP 12bit packed format
    YUV444i,
    NV12,
    NV21,
    BITSTREAM, // used for video encoder bitstream
    HDR,
    NONE
} frameType;

typedef struct frameSpecs {
    frameType type;
    uint32_t height;  // width in pixels
    uint32_t width;   // width in pixels
    uint32_t stride;  // defined as distance in bytes from pix(y,x) to pix(y+1,x)
    uint32_t bytesPP; // bytes per pixel (for LUT types set this to 1)
} frameSpec;

typedef struct frameElements {
    frameSpec spec;
    uint8_t *p1; // Pointer to first image plane
    uint8_t *p2; // Pointer to second image plane (if used)
    uint8_t *p3; // Pointer to third image plane  (if used)
} frameBuffer;

static inline uint32_t size_from_framespec(const frameSpec *spec) {
    if (!spec)
        throw std::invalid_argument("Argument must not be NULL");

    switch (spec->type) {
        case RAW8:
        case YUV400p:
            return spec->stride * spec->height;
        case YUV420p:
            return spec->stride * spec->height * 3 / 2;
        case BITSTREAM:
            return spec->width;
        case YUV444p:
        case RGB888:
            return spec->width * spec->height * 3;
        case RAW16:
            return spec->stride * spec->height * 2;
        default:
            throw std::invalid_argument("Error - Unsupported frametype: ");
            {}
    }
    return 0;
}

#endif // _SWC_FRAME_TYPES_H_
