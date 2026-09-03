#ifndef AETHELARCH_INTERNAL_H
#define AETHELARCH_INTERNAL_H

#include <stdlib.h>

#if defined(_MSC_VER)
    #define AETH_INLINE __forceinline
    #define AETH_ALIGNED(x) __declspec(align(x))
#else
    #define AETH_INLINE static inline __attribute__((always_inline))
    #define AETH_ALIGNED(x) __attribute__((aligned(x)))
#endif

static inline void* aeth_aligned_alloc(size_t alignment, size_t size) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return aligned_alloc(alignment, (size + alignment - 1) & ~(alignment - 1));
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
#else
    return malloc(size);
#endif
}

static inline void aeth_aligned_free(void* ptr) {
    free(ptr);
}

#endif /* AETHELARCH_INTERNAL_H */
