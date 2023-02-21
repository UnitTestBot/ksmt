#include <bitwuzla.h>
#include "bitwuzla_extension.h"

Bzla* bitwuzla_extension_get_bzla(Bitwuzla* bitwuzla) {
    return bitwuzla_get_bzla(bitwuzla);
}

BzlaMemMgr* bitwuzla_extension_get_bzla_memory(Bzla* bitwuzla) {
    return bitwuzla->mm;
}

void bitwuzla_extension_bzla_node_inc_ext_ref_counter(Bzla* bzla, BzlaNode* e) {
    bzla_node_inc_ext_ref_counter(bzla, e);
}

void bitwuzla_extension_bzla_bv_free(BzlaMemMgr* mm, BzlaBitVector* bv) {
    bzla_bv_free(mm, bv);
}

BzlaNode* bitwuzla_extension_bzla_exp_bv_const(Bzla* bzla, const BzlaBitVector* bits) {
    return bzla_exp_bv_const(bzla, bits);
}

BzlaBitVector* bitwuzla_extension_bzla_bv_concat(BzlaMemMgr* mm, const BzlaBitVector* a, const BzlaBitVector* b) {
    return bzla_bv_concat(mm, a, b);
}

BzlaBitVector*
bitwuzla_extension_bzla_bv_slice(BzlaMemMgr* mm, const BzlaBitVector* bv, uint32_t upper, uint32_t lower) {
    return bzla_bv_slice(mm, bv, upper, lower);
}

BzlaBitVector* bitwuzla_extension_bzla_bv_uint64_to_bv(BzlaMemMgr* mm, uint64_t value, uint32_t bw) {
    return bzla_bv_uint64_to_bv(mm, value, bw);
}

uint32_t bitwuzla_extension_bv_get_width(const BzlaBitVector* bv) {
    return bzla_bv_get_width(bv);
}

uint64_t bitwuzla_extension_bv_to_uint64(const BzlaBitVector* bv) {
    return bzla_bv_to_uint64(bv);
}

BzlaBitVector* bitwuzla_extension_node_bv_const_get_bits(BitwuzlaTerm* exp) {
    return bzla_node_bv_const_get_bits((BzlaNode*) exp);
}

BzlaFloatingPoint* bitwuzla_extension_node_fp_const_get_fp(BitwuzlaTerm* exp) {
    return bzla_node_fp_const_get_fp((BzlaNode*) exp);
}

BzlaBitVector* bitwuzla_extension_fp_bits_as_bv_bits(Bzla* bzla, BzlaFloatingPoint* fp) {
    return bzla_fp_as_bv(bzla, fp);
}

