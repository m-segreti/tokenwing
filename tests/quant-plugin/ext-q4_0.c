// A test quant plugin (ggml-quant-plugin.h). Plugin type slot 0 ("ext_q4_0") is byte for byte Q4_0 and forwards to
// ggml's own Q4_0 routines, so a model whose expert tensors were turned into this type (make_model_file.py
// --quant-plugin) must give the same text on the CPU as the same weights in Q4_0. With EXT_Q4_0_CUDA it also gives
// simple GPU kernels (ext-q4_0-cuda.cu). EXT_Q4_0_FAIL=N builds a plugin that is wrong in one way on purpose, to check
// that the loader refuses it:
//   1 no entry symbol           2 another interface version   3 init fails
//   4 no to_float               5 no vec_dot                  6 a vec_dot_type the CPU cannot convert to
//   7 another type name         8 slot 0 not provided         9 the model file's aux blob is refused

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quant-plugin.h"

#include <stdint.h>
#include <string.h>

#ifndef EXT_Q4_0_FAIL
#define EXT_Q4_0_FAIL 0
#endif

#ifdef EXT_Q4_0_CUDA
#define EXT_Q4_0_HAS_CUDA 1
int32_t ext_q4_0_mul_mat_cuda(
        int32_t type, const void * vx, const float * vy, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t nb01, size_t nb02, size_t nb03,
        int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1, size_t nb2, size_t nb3,
        int32_t device, void * stream);
int32_t ext_q4_0_mul_mat_id_cuda(
        int32_t type, const void * vx, const float * vy, const int32_t * ids, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02,
        size_t nb01, size_t nb02,
        int64_t ne11, int64_t ne12,
        size_t nb11, size_t nb12,
        int64_t n_expert_used,
        size_t ids_nb0, size_t ids_nb1,
        size_t nb1, size_t nb2,
        int32_t first_zero,
        int32_t device, void * stream);
#else
#define EXT_Q4_0_HAS_CUDA 0
#endif

#define QK 32           // values per Q4_0 block
#define QB 18           // bytes per Q4_0 block
#define AUX_BYTES 64    // the converter writes 64 bytes of 0xA5 as the model file's aux blob; loading checks them

static const struct ggml_type_traits     * tq;   // ggml's Q4_0 (to_float, from_float_ref)
static const struct ggml_type_traits_cpu * tc;   // ggml's CPU Q4_0 (vec_dot)
static const char * last_err = "";

static int32_t plug_init(void) {
    if (EXT_Q4_0_FAIL == 3) {
        last_err = "init fails on purpose";
        return 1;
    }
    tq = ggml_get_type_traits(GGML_TYPE_Q4_0);
    tc = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    if (!tq || !tq->to_float || !tc || !tc->vec_dot) {
        last_err = "ggml's Q4_0 routines are missing";
        return 1;
    }
    return 0;
}

static const char * plug_last_error(void) {
    return last_err;
}

static void ext_to_float(const void * x, float * y, int64_t k) {
    tq->to_float(x, y, k);
}

static void ext_from_float_ref(const float * x, void * y, int64_t k) {
    tq->from_float_ref(x, y, k);
}

static void ext_vec_dot(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    tc->vec_dot(n, s, bs, vx, bx, vy, by, nrc);
}

// four rows at vx, vx + bx, vx + 2 bx, vx + 3 bx against the same column: the same bits as the one-row kernel
static void ext_vec_dot_x4(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf) {
    (void) width;
    (void) pf;
    for (int r = 0; r < 4; ++r) {
        tc->vec_dot(n, &s[r], 0, (const char *) vx + (size_t) r*bx, 0, vy, 0, 1);
    }
}

static int32_t plug_type_info(int32_t slot, struct ggml_quant_plugin_type_info * o) {
    memset(o, 0, sizeof(*o));
    if (slot != 0 || EXT_Q4_0_FAIL == 8) {
        return 0;   // only slot 0 is provided
    }
    o->enabled        = 1;
    o->type_name      = EXT_Q4_0_FAIL == 7 ? "ext_other" : "ext_q4_0";
    o->blck_size      = QK;
    o->type_size      = QB;
    o->vec_dot_type   = EXT_Q4_0_FAIL == 6 ? GGML_TYPE_IQ2_XXS : GGML_TYPE_Q8_0;
    o->has_cuda       = EXT_Q4_0_HAS_CUDA;
    o->to_float       = EXT_Q4_0_FAIL == 4 ? NULL : ext_to_float;
    o->from_float_ref = ext_from_float_ref;
    o->vec_dot        = EXT_Q4_0_FAIL == 5 ? NULL : ext_vec_dot;
    o->vec_dot_x4     = ext_vec_dot_x4;
    return 0;
}

static int32_t plug_load_aux(const void * data, size_t size) {
    if (EXT_Q4_0_FAIL == 9) {
        last_err = "the aux blob is refused on purpose";
        return 1;
    }
    if (size != AUX_BYTES) {
        last_err = "the aux blob has the wrong size";
        return 1;
    }
    for (size_t i = 0; i < size; ++i) {
        if (((const uint8_t *) data)[i] != 0xA5) {
            last_err = "the aux blob has the wrong bytes";
            return 1;
        }
    }
    return 0;
}

// --- converter (make_model_file.py --quant-plugin): Q4_0 expert tensors get the plugin type, bytes unchanged ---

static int conv_state;

static void * conv_create(void) {
    return &conv_state;
}

static void conv_destroy(void * conv) {
    (void) conv;
}

static int32_t conv_probe(void * conv, const char * name, int32_t src_type, int32_t n_dims, const uint64_t * src_ne,
        const void * head, size_t head_len, uint64_t max_span, struct ggml_quant_plugin_tensor_plan * plan) {
    (void) conv; (void) head; (void) head_len; (void) max_span;
    memset(plan, 0, sizeof(*plan));
    plan->action     = GGML_QUANT_PLUGIN_TENSOR_KEEP;
    plan->out_type   = src_type;
    plan->out_n_dims = n_dims;
    for (int i = 0; i < n_dims && i < 4; ++i) {
        plan->out_ne[i] = src_ne[i];
    }
    if (n_dims == 3 && src_type == GGML_TYPE_Q4_0 && strstr(name, "_exps.weight") != NULL && src_ne[0] % QK == 0) {
        const uint64_t nbytes = src_ne[0]/QK*QB*src_ne[1]*src_ne[2];
        plan->action     = GGML_QUANT_PLUGIN_TENSOR_REHEADER;
        plan->out_type   = GGML_QUANT_PLUGIN_TYPE_FIRST;
        plan->src_nbytes = nbytes;
        plan->out_nbytes = nbytes;
    }
    return 0;
}

static uint64_t conv_aux_size(void * conv) {
    (void) conv;
    return AUX_BYTES;
}

static int32_t conv_transcode(void * conv, const char * name, int32_t src_type, int32_t n_dims, const uint64_t * src_ne,
        const void * src, size_t src_len, void * dst, size_t dst_len) {
    (void) conv; (void) name; (void) src_type; (void) n_dims; (void) src_ne; (void) src; (void) src_len; (void) dst; (void) dst_len;
    last_err = "this plugin only re-labels tensors";
    return 1;
}

static int32_t conv_finalize_aux(void * conv, void * out, size_t cap, size_t * written) {
    (void) conv;
    if (cap < AUX_BYTES) {
        last_err = "no room for the aux blob";
        return 1;
    }
    memset(out, 0xA5, AUX_BYTES);
    *written = AUX_BYTES;
    return 0;
}

// --- in-slot repack (--moe-stream-cpu-repack 1): the identity, so the repacked slots are Q4_0 as well ---

static int32_t plug_repack(int32_t type, void * dst, const void * src, int64_t ne0, int64_t ne1) {
    if (type != GGML_QUANT_PLUGIN_TYPE_FIRST || ne0 % QK != 0) {
        return 1;
    }
    if (dst == NULL && src == NULL) {
        return 0;   // the probe: every shape of this type is accepted
    }
    memcpy(dst, src, (size_t) (ne0/QK*QB*ne1));
    return 0;
}

static const struct ggml_quant_plugin_api api = {
    .abi_version = GGML_QUANT_PLUGIN_ABI_VERSION + (EXT_Q4_0_FAIL == 2 ? 1 : 0),
    .plugin_name = "ext-q4_0-test",
    .plugin_version = "1",
    .init = plug_init,
    .shutdown = NULL,
    .last_error = plug_last_error,
    .get_type_info = plug_type_info,
    .load_aux_blob = plug_load_aux,
#ifdef EXT_Q4_0_CUDA
    .mul_mat_cuda = ext_q4_0_mul_mat_cuda,
    .mul_mat_id_cuda = ext_q4_0_mul_mat_id_cuda,
#else
    .mul_mat_cuda = NULL,
    .mul_mat_id_cuda = NULL,
#endif
    .converter_create = conv_create,
    .converter_destroy = conv_destroy,
    .converter_probe_tensor = conv_probe,
    .converter_planned_aux_size = conv_aux_size,
    .converter_transcode_tensor = conv_transcode,
    .converter_finalize_aux = conv_finalize_aux,
    .repack_expert_slot = plug_repack,
};

#if defined(_WIN32)
#define EXT_Q4_0_EXPORT __declspec(dllexport)
#else
#define EXT_Q4_0_EXPORT __attribute__((visibility("default")))
#endif

#if EXT_Q4_0_FAIL == 1
EXT_Q4_0_EXPORT const struct ggml_quant_plugin_api * ext_q4_0_wrong_name(void) {
#else
EXT_Q4_0_EXPORT const struct ggml_quant_plugin_api * ggml_quant_plugin_get_api(void) {
#endif
    return &api;
}
