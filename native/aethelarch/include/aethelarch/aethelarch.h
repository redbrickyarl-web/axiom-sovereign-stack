#ifndef AETHELARCH_H
#define AETHELARCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t rows;
    size_t cols;
    size_t row_stride_bytes;
    uint8_t* w_pos;
    uint8_t* w_neg;
    float scale;
} aethelarch_matrix_t;

static inline size_t aethelarch_act_bytes(size_t dim) {
    size_t raw = (dim + 7) / 8;
    return (raw + 63) & ~(size_t)63;
}

aethelarch_matrix_t* aethelarch_matrix_create(size_t rows, size_t cols);
void aethelarch_matrix_free(aethelarch_matrix_t* mat);

bool aethelarch_encode_dense(
    aethelarch_matrix_t* dst,
    const int8_t* src_ternary,
    size_t rows,
    size_t cols
);

void aethelarch_quantize_activation(
    uint8_t* dst_bits,
    const float* src_activations,
    size_t dim
);

int32_t aethelarch_dot_product(
    const uint8_t* w_pos,
    const uint8_t* w_neg,
    const uint8_t* act_bits,
    size_t len_bits
);

bool aethelarch_gemv(
    float* dst_out,
    const aethelarch_matrix_t* mat,
    const uint8_t* act_bits
);

int32_t aethelarch_reference_dot(
    const int8_t* w_ternary,
    const int8_t* a_binary,
    size_t len
);

#ifdef __cplusplus
}
#endif

#endif /* AETHELARCH_H */
