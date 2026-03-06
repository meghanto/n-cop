use std::arch::x86_64::{
    __m256i, _mm256_alignr_epi8, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_extract_epi16,
    _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_testz_si256, _mm256_xor_si256,
};

#[cfg(target_feature = "avx512vl")]
use std::arch::x86_64::_mm256_ternarylogic_epi64;
