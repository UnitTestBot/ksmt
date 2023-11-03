#pragma once

#if __cplusplus
extern "C" {
#endif

void bitwuzla_extension_sort_dec_ref(BitwuzlaSort sort_id);

void bitwuzla_extension_term_dec_ref(BitwuzlaTerm term_id);

uint64_t bitwuzla_extension_bv_value_uint64(BitwuzlaTerm term);

const char *bitwuzla_extension_bv_value_str(BitwuzlaTerm term, uint32_t base);

#if __cplusplus
}
#endif
