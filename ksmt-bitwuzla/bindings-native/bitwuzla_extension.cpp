#include "bitwuzla/c/bitwuzla.h"
#include <bitwuzla_extension.hpp>


#if __cplusplus
extern "C" {
#endif

void bitwuzla_extension_sort_dec_ref(BitwuzlaSort sort_id) {
    bitwuzla_sort_dec_ref(sort_id);
}

void bitwuzla_extension_term_dec_ref(BitwuzlaTerm term_id) {
    bitwuzla_term_dec_ref(term_id);
}

uint64_t bitwuzla_extension_bv_value_uint64(BitwuzlaTerm term) {
    return bitwuzla_bv_value_uint64(term);
}

const char *bitwuzla_extension_bv_value_str(BitwuzlaTerm term, uint32_t base) {
    return bitwuzla_bv_value_str(term, base);
}

#if __cplusplus
}
#endif
