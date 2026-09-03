#include "aethelarch/aethelarch.h"
#include <string.h>
#include <stdint.h>

int32_t aethelarch_reference_dot(
    const int8_t* w_ternary,
    const int8_t* a_binary,
    size_t len
) {
    if (!w_ternary || !a_binary) return 0;
    int32_t acc = 0;
    for (size_t i = 0; i < len; ++i) {
        acc += ((int32_t)w_ternary[i]) * ((int32_t)a_binary[i]);
    }
    return acc;
}

#if !defined(__aarch64__) && !(defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__))
int32_t aethelarch_dot_product(
    const uint8_t* w_pos,
    const uint8_t* w_neg,
    const uint8_t* act_bits,
    size_t num_bits
) {
    if (!w_pos || !w_neg || !act_bits || num_bits == 0) return 0;

    size_t full_bytes = num_bits / 8;
    size_t rem_bits = num_bits % 8;
    int64_t total_pos = 0;
    int64_t total_neg = 0;
    size_t i = 0;

    while (i + 8 <= full_bytes) {
        uint64_t a, wp, wn;
        memcpy(&a, act_bits + i, 8);
        memcpy(&wp, w_pos + i, 8);
        memcpy(&wn, w_neg + i, 8);
        total_pos += __builtin_popcountll(a & wp);
        total_neg += __builtin_popcountll(a & wn);
        i += 8;
    }

    while (i < full_bytes) {
        total_pos += __builtin_popcount(act_bits[i] & w_pos[i]);
        total_neg += __builtin_popcount(act_bits[i] & w_neg[i]);
        i++;
    }

    if (rem_bits > 0) {
        uint8_t mask = (uint8_t)((1u << rem_bits) - 1);
        total_pos += __builtin_popcount((act_bits[full_bytes] & mask) & (w_pos[full_bytes] & mask));
        total_neg += __builtin_popcount((act_bits[full_bytes] & mask) & (w_neg[full_bytes] & mask));
    }

    return (int32_t)(total_pos - total_neg);
}
#endif
