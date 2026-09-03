#include "aethelarch/aethelarch.h"
#include "aethelarch/internal.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>

int32_t aethelarch_dot_product(
    const uint8_t* w_pos,
    const uint8_t* w_neg,
    const uint8_t* act_bits,
    size_t num_bits
) {
    if (!w_pos || !w_neg || !act_bits || num_bits == 0) return 0;

    size_t full_bytes = num_bits / 8;
    size_t rem_bits = num_bits % 8;

    uint32x4_t acc_p = vdupq_n_u32(0);
    uint32x4_t acc_n = vdupq_n_u32(0);

    size_t i = 0;
    while (i + 16 <= full_bytes) {
        uint8x16_t va = vld1q_u8(act_bits + i);
        uint8x16_t vwp = vld1q_u8(w_pos + i);
        uint8x16_t vwn = vld1q_u8(w_neg + i);

        uint8x16_t match_p = vandq_u8(va, vwp);
        uint8x16_t match_n = vandq_u8(va, vwn);

        uint8x16_t cnt_p = vcntq_u8(match_p);
        uint8x16_t cnt_n = vcntq_u8(match_n);

        uint16x8_t p16 = vpaddlq_u8(cnt_p);
        uint16x8_t n16 = vpaddlq_u8(cnt_n);

        acc_p = vpadalq_u16(acc_p, p16);
        acc_n = vpadalq_u16(acc_n, n16);

        i += 16;
    }

    int64_t total_pos = vaddvq_u32(acc_p);
    int64_t total_neg = vaddvq_u32(acc_n);

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
