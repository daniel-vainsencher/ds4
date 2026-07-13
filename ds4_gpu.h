#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label);
#endif
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
#ifdef DS4_ROCM_BUILD
void ds4_gpu_release_q8_f16_cache(void);
#endif
int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_set_ssd_streaming(bool enabled);
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
uint64_t ds4_gpu_recommended_working_set_size(void);
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);
uint32_t ds4_gpu_stream_expert_cache_current_count(void);
typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void);
void ds4_gpu_stream_expert_cache_release_resident(void);
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_prepare_selected_batch_zero_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
#endif
#ifdef DS4_ROCM_BUILD
int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table);
int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected);
int ds4_gpu_stream_expert_cache_release_layer_cache(void);
#endif
int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
void ds4_gpu_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps);

int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected);

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* =========================================================================
 * Quantized Dot Product Test Kernels.
 * =========================================================================
 *
 * These functions expose the Q2_K and IQ2_XXS dot product kernels for testing
 * GPU implementations against CPU reference implementations.  Each computes
 * dot products between quantized weight blocks and Q8_K activation blocks.
 *
 * Parameters:
 *   out       - Output tensor for dot product results (n_rows floats)
 *   weights   - Device tensor containing quantized weight blocks (Q2_K or IQ2_XXS)
 *   x_q8      - Device tensor containing Q8_K quantized activation blocks
 *   n_rows    - Number of output rows (dot products to compute)
 *   n_blocks  - Number of quantized blocks per row (row_dim / 256)
 *
 * Returns nonzero on success, 0 on failure.
 */
int ds4_gpu_test_dot_q2_K_q8_K_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x_q8,
        uint32_t              n_rows,
        uint32_t              n_blocks);

int ds4_gpu_test_dot_iq2_xxs_q8_K_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x_q8,
        uint32_t              n_rows,
        uint32_t              n_blocks);

/**
 * Test MoE gate/up/mid computation for a single (token, expert) pair.
 *
 * This performs the forward pass of the first half of an MoE expert:
 *   gate[row] = sum_b dot_iq2_xxs_q8_K(gate_weights[row,b], xq[b])
 *   up[row]   = sum_b dot_iq2_xxs_q8_K(up_weights[row,b], xq[b])
 *   mid[row]  = swiglu(gate[row]) * up[row] * router_weight
 *
 * where swiglu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Parameters:
 *   gate_out     - Output: [expert_mid_dim] float gate values
 *   up_out       - Output: [expert_mid_dim] float up values
 *   mid_out      - Output: [expert_mid_dim] float mid values (SwiGLU output)
 *   gate_weights - [expert_mid_dim, n_blocks] IQ2_XXS gate projection weights
 *   up_weights   - [expert_mid_dim, n_blocks] IQ2_XXS up projection weights
 *   x_q8         - [n_blocks] Q8_K quantized input activations
 *   expert_mid_dim - Number of output rows (MoE intermediate dimension)
 *   n_blocks     - Number of Q8_K blocks per row (expert_in_dim / 256)
 *   router_weight - Routing weight for this expert (typically from softmax)
 *   clamp        - Activation clamping value (0 to disable)
 *
 * Returns nonzero on success, 0 on failure.
 */
int ds4_gpu_test_moe_gate_up_mid_tensor(
        ds4_gpu_tensor       *gate_out,
        ds4_gpu_tensor       *up_out,
        ds4_gpu_tensor       *mid_out,
        const ds4_gpu_tensor *gate_weights,
        const ds4_gpu_tensor *up_weights,
        const ds4_gpu_tensor *x_q8,
        uint32_t              expert_mid_dim,
        uint32_t              n_blocks,
        float                 router_weight,
        float                 clamp);

/**
 * Test RMS normalization with learned weight.
 *
 * out[row,i] = x[row,i] * rsqrt(mean(x[row]^2) + eps) * weight[i]
 *
 * Parameters:
 *   out    - Output: [rows, n] float normalized values
 *   x      - Input: [rows, n] float values to normalize
 *   weight - Weight: [n] float per-channel scale
 *   n      - Dimension per row
 *   rows   - Number of rows
 *   eps    - Epsilon for numerical stability
 */
int ds4_gpu_test_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *weight,
        uint32_t              n,
        uint32_t              rows,
        float                 eps);

/**
 * Test Q8_0 batched matrix multiplication (uses WMMA when available).
 *
 * out[t,r] = sum_k(dequant(weights[r,k]) * x[t,k])
 *
 * Parameters:
 *   out      - Output: [n_tokens, out_dim] float
 *   weights  - Q8_0 quantized weights: [out_dim, in_dim/32] blocks of 34 bytes
 *   x        - Input: [n_tokens, in_dim] float
 *   n_tokens - Number of input tokens
 *   in_dim   - Input dimension (must be multiple of 32)
 *   out_dim  - Output dimension
 */
int ds4_gpu_test_matmul_q8_0_f32_batch_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        uint32_t              in_dim,
        uint32_t              out_dim);

/**
 * Test F16 matrix multiplication (simplified test harness).
 *
 * out[t,r] = sum_k(f16_to_f32(weights[r,k]) * x[t,k])
 *
 * Parameters:
 *   out      - Output: [n_tok, out_dim] float
 *   weights  - FP16 weights: [out_dim, in_dim] half
 *   x        - Input: [n_tok, in_dim] float
 *   n_tok    - Number of input tokens
 *   in_dim   - Input dimension
 *   out_dim  - Output dimension
 */
int ds4_gpu_test_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              in_dim,
        uint32_t              out_dim);

/**
 * Test attention decode with mixed raw/compressed KV cache.
 *
 * Uses the heads8 online softmax kernel for head_dim=512.
 *
 * Parameters:
 *   heads     - Output: [n_tokens, n_head, head_dim] attention output
 *   sinks     - Attention sink scores: [n_head]
 *   q         - Query: [n_tokens, n_head, head_dim]
 *   raw_kv    - Raw KV cache: [raw_cap, head_dim]
 *   comp_kv   - Compressed KV cache: [n_comp, head_dim] (NULL if n_comp=0)
 *   n_tokens  - Number of query tokens
 *   pos0      - Position of first token
 *   n_raw     - Number of raw KV entries
 *   raw_cap   - Capacity of raw KV ring buffer
 *   raw_start - Start index in ring buffer
 *   n_comp    - Number of compressed KV entries
 *   window    - Attention window size (0 for unlimited)
 *   ratio     - Compression ratio (0 to disable visibility limit)
 *   n_head    - Number of attention heads
 *   head_dim  - Dimension per head
 */
int ds4_gpu_test_attention_decode_mixed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *sinks,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim);

/**
 * Test MoE gate/up projection with Q2_K weights using WMMA.
 *
 * Tests the first half of an MoE expert using WMMA acceleration:
 *   mid[t,r] = swiglu(gate_dot) * up_dot * router_weight
 *
 * Returns 0 on platforms without WMMA support.
 *
 * Parameters:
 *   mid_out        - Output: [n_tokens, expert_mid_dim] float
 *   gate_weights   - Q2_K gate weights: [expert_mid_dim, expert_in_dim/256] blocks
 *   up_weights     - Q2_K up weights: [expert_mid_dim, expert_in_dim/256] blocks
 *   x              - Input: [n_tokens, expert_in_dim] float
 *   router_weight  - Scalar routing weight
 *   n_tokens       - Number of input tokens
 *   expert_in_dim  - Input dimension (must be multiple of 256)
 *   expert_mid_dim - Output dimension
 *   clamp          - Activation clamping value (0 to disable)
 */
int ds4_gpu_test_moe_gate_up_mid_q2k_wmma_tensor(
        ds4_gpu_tensor       *mid_out,
        const ds4_gpu_tensor *gate_weights,
        const ds4_gpu_tensor *up_weights,
        const ds4_gpu_tensor *x,
        float                 router_weight,
        uint32_t              n_tokens,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        float                 clamp);

/**
 * Test MoE down projection with Q2_K weights using WMMA.
 *
 * Tests the second half of an MoE expert using WMMA acceleration:
 *   out[t,r] = dot(down_weights[r], mid[t])
 *
 * Returns 0 on platforms without WMMA support.
 *
 * Parameters:
 *   out            - Output: [n_tokens, out_dim] float
 *   down_weights   - Q2_K down weights: [out_dim, expert_mid_dim/256] blocks
 *   mid            - Input: [n_tokens, expert_mid_dim] float
 *   n_tokens       - Number of input tokens
 *   expert_mid_dim - Mid dimension (must be multiple of 256)
 *   out_dim        - Output dimension
 */
int ds4_gpu_test_moe_down_q2k_wmma_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *down_weights,
        const ds4_gpu_tensor *mid,
        uint32_t              n_tokens,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim);

/* HC Sinkhorn Split Test Harness
 * =========================================================================
 *
 * Tests the HC (hyper-connection) split with Sinkhorn normalization.
 * This is a core prefill operation that produces pre-sublayer mixing weights
 * from the HC state.
 *
 * The operation computes:
 *   1. pre[i]  = sigmoid(mix[i] * scale[0] + base[i]) + eps
 *   2. post[i] = 2 * sigmoid(mix[n_hc+i] * scale[1] + base[n_hc+i])
 *   3. comb[dst,src] = softmax_row(mix[2*n_hc + idx] * scale[2] + base[...])
 *   4. Sinkhorn iterations: alternating row/column normalization
 *
 * Output layout for n_hc=4 (24 floats per row):
 *   [0..3]    = pre weights (4 floats)
 *   [4..7]    = post weights (4 floats)
 *   [8..23]   = comb matrix (4x4 = 16 floats, row-major)
 *
 * Parameters:
 *   out            - Output: [n_rows, 24] float (pre + post + comb)
 *   mix            - Input:  [n_rows, 24] float (mixing coefficients)
 *   scale          - Scale:  [3] float (pre_scale, post_scale, comb_scale)
 *   base           - Base:   [24] float (bias terms)
 *   n_hc           - Number of hyper-connection streams (must be 4)
 *   n_rows         - Number of rows (tokens) to process
 *   sinkhorn_iters - Number of Sinkhorn iterations (typically 3-5)
 *   eps            - Epsilon for numerical stability (typically 1e-6)
 */
int ds4_gpu_test_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *scale,
        const ds4_gpu_tensor *base,
        uint32_t              n_hc,
        uint32_t              n_rows,
        uint32_t              sinkhorn_iters,
        float                 eps);

/* =========================================================================
 * Prefill Test Entry Points.
 * =========================================================================
 *
 * These functions provide clean entry points for testing the full prefill
 * computation path without session management overhead. They take a loaded
 * engine and an array of tokens, and produce logits for the final token
 * plus optionally the raw KV cache state.
 *
 * Both CPU and GPU versions are provided for oracle comparison testing.
 * The functions are designed for ds4rust consumption via FFI.
 */

/**
 * CPU prefill test - runs full prefill on CPU and returns final logits.
 *
 * This executes the complete prefill computation using the CPU reference
 * implementation (layer-major order with batched attention/FFN).
 *
 * Parameters:
 *   engine       - Opened ds4 engine with loaded model
 *   tokens       - Input token IDs [n_tokens]
 *   n_tokens     - Number of tokens to prefill (must be > 0)
 *   out_logits   - Output: logits for final token [n_vocab floats]
 *                  May be NULL if only KV cache output is needed
 *   out_kv_raw   - Output: raw KV cache data for all layers (optional, may be NULL)
 *                  Layout: [n_layer][min(n_tokens, swa_window)][head_dim] floats
 *   out_kv_raw_size - Output: bytes written to out_kv_raw (set even if out_kv_raw is NULL)
 *
 * Returns 0 on success, nonzero on error.
 */
int ds4_test_prefill_cpu(
        void *engine,
        const int32_t *tokens,
        uint32_t n_tokens,
        float *out_logits,
        float *out_kv_raw,
        uint64_t *out_kv_raw_size);

/**
 * GPU prefill test - runs full prefill on GPU and returns final logits.
 *
 * This executes the complete prefill computation using the GPU implementation
 * (layer-major order with batched kernels).
 *
 * Parameters:
 *   engine       - Opened ds4 engine with loaded model and initialized GPU
 *   tokens       - Input token IDs [n_tokens]
 *   n_tokens     - Number of tokens to prefill (must be > 0)
 *   out_logits   - Output: logits for final token [n_vocab floats]
 *                  May be NULL if only KV cache output is needed
 *   out_kv_raw   - Output: raw KV cache data for all layers (optional, may be NULL)
 *                  Layout: [n_layer][min(n_tokens, swa_window)][head_dim] floats
 *   out_kv_raw_size - Output: bytes written to out_kv_raw (set even if out_kv_raw is NULL)
 *
 * Returns 0 on success, nonzero on error.
 */
int ds4_test_prefill_gpu(
        void *engine,
        const int32_t *tokens,
        uint32_t n_tokens,
        float *out_logits,
        float *out_kv_raw,
        uint64_t *out_kv_raw_size);

/**
 * Get the vocabulary size for the loaded model.
 * Useful for allocating the out_logits buffer.
 */
int ds4_test_get_vocab_size(void *engine);

/**
 * Get the number of layers in the model.
 */
int ds4_test_get_n_layer(void *engine);

/**
 * Get the sliding window attention size.
 */
int ds4_test_get_swa_window(void *engine);

/**
 * Get the head dimension (KV cache row size).
 */
int ds4_test_get_head_dim(void *engine);

/* =========================================================================
 * Decode Test Entry Points.
 * =========================================================================
 *
 * These functions test single-token decode (generation) after a prefill.
 * They take a context (previously prefilled tokens) and a new token,
 * then produce logits for the next token prediction.
 *
 * For testing, the context is re-prefilled internally to establish the
 * KV cache state, then a single decode step is performed.
 */

/**
 * CPU decode test - prefills context, then decodes one token on CPU.
 *
 * This first runs CPU prefill on the context tokens to establish KV cache,
 * then performs a single decode step for the new token.
 *
 * Parameters:
 *   engine         - Opened ds4 engine with loaded model
 *   context_tokens - Previous context token IDs [n_context]
 *   n_context      - Number of context tokens (must be > 0)
 *   new_token      - Single token to decode
 *   out_logits     - Output: logits for next token prediction [n_vocab floats]
 *
 * Returns 0 on success, nonzero on error.
 */
int ds4_test_decode_cpu(
        void *engine,
        const int32_t *context_tokens,
        uint32_t n_context,
        int32_t new_token,
        float *out_logits);

/**
 * GPU decode test - prefills context, then decodes one token on GPU.
 *
 * This first runs GPU prefill on the context tokens to establish KV cache,
 * then performs a single decode step for the new token.
 *
 * Parameters:
 *   engine         - Opened ds4 engine with loaded model and initialized GPU
 *   context_tokens - Previous context token IDs [n_context]
 *   n_context      - Number of context tokens (must be > 0)
 *   new_token      - Single token to decode
 *   out_logits     - Output: logits for next token prediction [n_vocab floats]
 *
 * Returns 0 on success, nonzero on error.
 */
int ds4_test_decode_gpu(
        void *engine,
        const int32_t *context_tokens,
        uint32_t n_context,
        int32_t new_token,
        float *out_logits);

#ifdef __cplusplus
}
#endif

#endif
