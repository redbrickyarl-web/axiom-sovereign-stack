#include "aethelarch/aethelarch.h"
#include "aethelarch/internal.h"
#include <string.h>

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
#include <immintrin.h>

int32_t aethelarch_dot_product(
    const uint8_t* w_pos,
    const uint8_t* w_neg,
    const uint8_t* act_bits,
    size_t num_bits
) {
    if (!w_pos || !w_neg || !act_bits || num_bits == 0) return 0;

    size_t full_bytes = num_bits / 8;
    size_t rem_bits = num_bits % 8;

    __m512i acc_p = _mm512_setzero_si512();
    __m512i acc_n = _mm512_setzero_si512();

    size_t i = 0;
    while (i + 64 <= full_bytes) {
        __m512i va = _mm512_loadu_si512((const void*)(act_bits + i));
        __m512i vwp = _mm512_loadu_si512((const void*)(w_pos + i));
        __m512i vwn = _mm512_loadu_si512((const void*)(w_neg + i));

        __m512i match_p = _mm512_and_si512(va, vwp);
        __m512i match_n = _mm512_and_si512(va, vwn);

        __m512i cnt_p = _mm512_popcnt_epi64(match_p);
        __m512i cnt_n = _mm512_popcnt_epi64(match_n);

        acc_p = _mm512_add_epi64(acc_p, cnt_p);
        acc_n = _mm512_add_epi64(acc_n, cnt_n);

        i += 64;
    }

    int64_t total_pos = _mm512_reduce_add_epi64(acc_p);
    int64_t total_neg = _mm512_reduce_add_epi64(acc_n);

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
