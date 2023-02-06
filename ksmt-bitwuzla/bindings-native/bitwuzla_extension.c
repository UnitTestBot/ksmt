#include <bitwuzla.h>
#include "bitwuzla_extension.h"

BzlaBitVector* bitwuzla_extension_node_bv_const_get_bits(BitwuzlaTerm* exp) {
	return bzla_node_bv_const_get_bits(exp);
}

uint32_t bitwuzla_extension_bv_get_width(const BzlaBitVector* bv) {
	return bzla_bv_get_width(bv);
}

uint64_t bitwuzla_extension_bv_to_uint64(const BzlaBitVector* bv) {
	return bzla_bv_to_uint64(bv);
}

uint32_t bitwuzla_extension_bv_get_bit(const BzlaBitVector* bv, uint32_t pos) {
	return bzla_bv_get_bit(bv, pos);
}

void* bitwuzla_extension_get_bzla(Bitwuzla* bitwuzla) {
	return bitwuzla_get_bzla(bitwuzla);
}

void* bitwuzla_extension_node_fp_const_get_fp(BitwuzlaTerm* exp) {
	return bzla_node_fp_const_get_fp(exp);
}

BzlaBitVector* bitwuzla_extension_fp_as_bv(void* bzla, void* fp) {
	return bzla_fp_as_bv(bzla, fp);
}
