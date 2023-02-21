#pragma once

#include <bitwuzla.h>
#include <bzlacore.h>
#include <stdint.h>

/** Get Bitwuzla core. */
Bzla* bitwuzla_get_bzla(Bitwuzla* bitwuzla);

BzlaNode *bzla_exp_bv_const(Bzla *bzla, const BzlaBitVector *bits);

#if __cplusplus
extern "C" {
#endif

Bzla* bitwuzla_extension_get_bzla(Bitwuzla* bitwuzla);

BzlaMemMgr* bitwuzla_extension_get_bzla_memory(Bzla* bitwuzla);

void bitwuzla_extension_bzla_node_inc_ext_ref_counter(Bzla* bzla, BzlaNode* e);

void bitwuzla_extension_bzla_bv_free(BzlaMemMgr* mm, BzlaBitVector* bv);

BzlaNode* bitwuzla_extension_bzla_exp_bv_const(Bzla* bzla, const BzlaBitVector* bits);

BzlaBitVector*
bitwuzla_extension_bzla_bv_concat(BzlaMemMgr* mm, const BzlaBitVector* a, const BzlaBitVector* b);

BzlaBitVector*
bitwuzla_extension_bzla_bv_slice(BzlaMemMgr* mm,
              const BzlaBitVector* bv,
              uint32_t upper,
              uint32_t lower);

BzlaBitVector*
bitwuzla_extension_bzla_bv_uint64_to_bv(BzlaMemMgr* mm, uint64_t value, uint32_t bw);

uint32_t bitwuzla_extension_bv_get_width(const BzlaBitVector* bv);

uint64_t bitwuzla_extension_bv_to_uint64(const BzlaBitVector* bv);


BzlaBitVector* bitwuzla_extension_node_bv_const_get_bits(BitwuzlaTerm* exp);

BzlaFloatingPoint * bitwuzla_extension_node_fp_const_get_fp(BitwuzlaTerm* exp);

BzlaBitVector* bitwuzla_extension_fp_bits_as_bv_bits(Bzla* bzla, BzlaFloatingPoint* fp);

#if __cplusplus
}
#endif
