# EdgeLLM Inference Engine - Implementation Checklist

> 각 Phase별 체크리스트. 순서대로 구현하되, Phase 0A/0B/0C는 병렬 가능.

---

## Phase 0: Compressor 수정 (전제 조건)

### 0A. BFloat16 지원

- [x] `compressor/config/pack.py`: `PackConfig`에 `dtype: str = "float16"` 필드 추가
- [x] `compressor/compress.py` (line 190): `torch_dtype=torch.float16` → `--dtype` CLI 인자로 설정 가능하게
- [x] `compressor/evaluate.py`: 동일하게 dtype 파라미터화
- [x] `compressor/packer/pack.py`:
  - `build_state_dict()`: `dtype` 파라미터 추가, 모든 `.half()` → `.to(dtype)` 변경 (lines 41, 49, 52, 77, 81)
  - `build_metadata()`: metadata에 `"dtype"` 필드 추가
- [x] `compressor/packer/convert.py`: `scale.half()`, `zero.half()` → `.to(dtype)` (lines 35-36, 모든 format 함수)
- [x] **검증**: 작은 모델을 FP16과 BF16으로 각각 압축, 출력 텐서 dtype 확인

### 0B. 엔진-포맷 아키텍처 리팩터링 + Marlin 호환 Weight Packing (W4A16)

#### 0B-1. format 모듈 구조 생성

- [x] `compressor/packer/format/__init__.py` 생성:
  - `FormatHandler` protocol 정의 (`convert_linear` 메서드)
  - `_REGISTRY` dict: `{"w4a16": W4A16Format, "w4a8": W4A8Format, "w8a8": W8A8Format}`
  - `resolve(format_type: str) -> FormatHandler` 함수
- [x] `compressor/packer/format/w4a8.py` 생성: `W4A8Format` stub (향후 엔진 확정 시 구현)
- [x] `compressor/packer/format/w8a8.py` 생성: `W8A8Format` stub (향후 엔진 확정 시 구현)

#### 0B-2. W4A16 Marlin 포맷 구현

- [x] `compressor/packer/format/w4a16.py` 생성: `W4A16Format` 클래스
  - `convert_linear(weight, qparams, qparams_key, args, dtype)` → `{"qweight": int32, "scales": permuted, "zeros": permuted}`
  - `pack_weight(q, size_k, size_n)` → int32 Marlin 16×16 타일 포맷
  - `permute_scales(scales, size_k, size_n, group_size)` → Marlin permuted scales
  - `permute_zeros(zeros, ...)` → Marlin permuted zeros (asymmetric용)
  - `_get_scale_perms()` → permutation 인덱스 (vLLM `marlin_utils.py` L293-313에서 포팅)

#### 0B-3. 기존 코드 제거 및 연동

- [x] `compressor/packer/convert.py` **삭제** (format 모듈로 대체)
- [x] `compressor/packer/base.py` 수정: `infer_format` import를 `format` 모듈로 변경, `mma` 인자 제거
- [x] `compressor/packer/pack.py` 수정: `build_state_dict()`에서 `format.resolve()` 사용, `mma` 파라미터 제거
- [x] `compressor/utils/pack.py` **삭제** (`infer_format`은 format 모듈로 이동, `apply_mma` 제거)
- [x] `compressor/utils/__init__.py`: `from .pack import *` 제거
- [x] `compressor/packer/__init__.py`: `from .convert import *` 제거, `from .format import *` 추가

#### 0B-4. 검증

- [x] W4A16 Marlin 패킹: qweight shape = `[size_k // 16, size_n * 2]` (int32)
- [x] Marlin permuted scales shape 검증
- [x] `format.resolve("w4a16")` → `W4A16Format` 인스턴스 반환 확인

### 0C. Asymmetric Quantization 지원

- [x] `compressor/packer/format/w4a16.py`: `W4A16Format.convert_linear()`에서 `args.symmetric == False`일 때 `permute_zeros()` 적용하여 저장
- [x] `compressor/packer/pack.py`: `build_metadata()`에 `"symmetric": bool` 추가, `"mma"` 필드 제거
- [x] **검증**: asymmetric W4A16 양자화 + Marlin packing이 유효한 출력 생성

### 0D. Phase 0 통합 검증

- [x] 압축 모델 출력 SafeTensors의 텐서 shape 확인 (qweight, scales, zeros)
- [x] config.json metadata 완전성 확인 (format, dtype, symmetric)

---

## Phase 1: Marlin GEMM CUDA 커널 (W4A16)

### 1A. 빌드 시스템

- [ ] `engine/__init__.py` 생성
- [ ] `engine/csrc/w4a16/` 디렉토리 생성
- [ ] `engine/setup.py` 생성: `torch.utils.cpp_extension.CUDAExtension`
  - SM >= 75 (Turing), SM >= 80 (Ampere) 지원
  - 컴파일러 플래그: `-O3`, `--use_fast_math`, SM별 `-gencode`
- [ ] `pyproject.toml`: `include = ["compressor*"]` → `include = ["compressor*", "engine*"]`
- [ ] **검증**: 빈 커널이라도 빌드 성공 확인

### 1B. Marlin 커널 코어 파일 (vLLM에서 최소화 추출)

**`engine/csrc/w4a16/marlin.cuh`** (from `vllm/.../marlin.cuh`):
- [ ] 유지: `default_threads`, `pipe_stages`, tile/thread 상수, `Vec`, `I4`, `div_ceil`, cp_async
- [ ] 변경: namespace를 `edgellm`으로

**`engine/csrc/w4a16/marlin_dtypes.cuh`** (from `vllm/.../marlin_dtypes.cuh`):
- [ ] 유지: `MarlinScalarType<half>`, `MarlinScalarType<nv_bfloat16>`
- [ ] 제거: `vllm::ScalarTypeId` 의존성, FP8/INT8 타입
- [ ] 추가: 단순 enum `WeightType { kU4B8, kU4 }`, `ScalarType { kFloat16, kBFloat16 }`

**`engine/csrc/w4a16/dequant.h`** (from `vllm/.../dequant.h`):
- [ ] 유지: INT4→FP16 (`half2`, kU4B8/kU4), INT4→BF16 (`nv_bfloat162`, kU4B8/kU4)
- [ ] 제거: INT8, FP8, FP4 dequantization 전체

**`engine/csrc/w4a16/marlin_mma.h`** (from `vllm/.../marlin_mma.h`):
- [ ] 유지: FP16 m16n8k16 MMA, BF16 m16n8k16 MMA, SM75 FP16 m16n8k8
- [ ] 제거: INT8 MMA, FP8 MMA, FP16 accumulation

**`engine/csrc/w4a16/marlin_template.h`** (from `vllm/.../marlin_template.h`, 81KB → ~30KB):
- [ ] 제거: `has_act_order` 전체 코드패스 (~30%)
- [ ] 제거: FP8/INT8 activation paths (`is_a_8bit` 분기)
- [ ] 제거: `global_scale` 처리 (FP8 전용)
- [ ] 제거: FP4/MXFP4 paths
- [ ] 유지: group_blocks 변형 (-1, 2, 4, 8), m_block_size_8, atomic_add, fp32_reduce

**`engine/csrc/w4a16/repack.cu`** (from `vllm/.../gptq_marlin_repack.cu`):
- [ ] 유지: 기본 repacking 로직 (permutation 없는 버전)
- [ ] 제거: `has_perm` (act_order) 변형

**`engine/csrc/w4a16/marlin.cu`** (from `vllm/.../marlin.cu`):
- [ ] 단순화된 `marlin_gemm()` 인터페이스 구현
- [ ] 제거: act_order 로직, FP8/INT8 dispatch
- [ ] 유지: thread config 선택, shared memory 검증

### 1C. Python Ops Binding

- [ ] `engine/ops.py` 생성:
  - `marlin_gemm(a, b_q_weight, b_scales, b_zeros, workspace, M, N, K, is_symmetric)` → Tensor
  - `marlin_repack(b_q_weight, size_k, size_n)` → Tensor (선택적)
  - Lazy import 패턴

### 1D. 검증

- [ ] `marlin_gemm` vs `torch.nn.functional.linear` (역양자화 weight) 비교
- [ ] Symmetric (kU4B8) + Asymmetric (kU4) 테스트
- [ ] FP16, BF16 activation 테스트
- [ ] 다양한 shape: (1, 4096, 4096), (32, 4096, 11008), (128, 4096, 4096)
- [ ] Group size 128 (group_blocks=8), channelwise (group_blocks=-1) 테스트

---

## Phase 2: 추론 Python Layer (모델 로딩 + 기본 추론)

### 2A. QuantizedLinear Module

- [ ] `engine/model/linear.py` 생성
  - `QuantizedLinear(nn.Module)`: qweight(int32), scales, zeros를 buffer로 저장
  - `forward(x)`: `engine.ops.marlin_gemm` 호출
  - `from_state_dict(state_dict, prefix, config)`: SafeTensors 키에서 구성

### 2B. Model Loader

- [ ] `engine/model/loader.py` 생성
  - `ModelConfig` dataclass: config.json 파싱 (architecture, quantization, dtype, symmetric)
  - `load_model(model_dir, device)`: SafeTensors 로딩 → 양자화 모델 구성
  - 비양자화 레이어: Embedding, RMSNorm, lm_head (FP16/BF16)

### 2C. Attention Layer

- [ ] `engine/model/attention.py` 생성
  - `Attention(nn.Module)`: QKV projection (QuantizedLinear), O projection
  - GQA 지원 (kv_heads repeat)
  - RoPE 적용 (PyTorch 구현)
  - Prefill: FlashAttention / SDPA fallback
  - Decode: 단순 concatenation KV cache (Phase 3에서 paged로 교체)

### 2D. Transformer

- [ ] `engine/model/transformer.py` 생성
  - `RMSNorm(nn.Module)`
  - `DecoderLayer(nn.Module)`: pre-norm → Attention → residual → pre-norm → MLP (SiLU-GLU) → residual
  - `Transformer(nn.Module)`: Embedding → [DecoderLayer × N] → Norm → lm_head

### 2E. Sampling

- [ ] `engine/model/sampling.py` 생성
  - `greedy(logits)` → token_id
  - `sample(logits, temperature, top_k, top_p)` → token_id

### 2F. Generation Loop

- [ ] `engine/model/generate.py` 생성
  - `Generator`: 모델 + 토크나이저 래핑
  - `generate(prompt, max_tokens, **sampling_kwargs)`:
    1. 토큰화
    2. Prefill (전체 프롬프트 forward)
    3. Decode loop (autoregressive)
    4. 정지 조건: max_tokens, EOS
  - Streaming yield 지원

### 2G. 검증

- [ ] 압축 모델 로딩 → 텍스트 생성 end-to-end 테스트
- [ ] WikiText PPL 측정, compressor 평가 결과와 비교

---

## Phase 3: Paged Attention + KV Cache CUDA 커널

### 3A. Paged KV Cache Manager (Python)

- [ ] `engine/core/kv_cache.py` 생성:
  - `KVCacheConfig`: block_size(16), num_blocks, kv_dtype("float16"/"int4"/"int8")
  - `KVCacheManager`: GPU 메모리 할당, block free list 관리
- [ ] `engine/core/block_table.py` 생성:
  - `BlockTable`: sequence별 block 매핑, slot_mapping 계산

### 3B. Paged Attention CUDA 커널 (OmniServe pure_dense에서 추출)

- [ ] `engine/csrc/attention/paged_attention.cu` 생성
  - OmniServe `decoderMaskedMultiheadAttention.cu` (pure_dense) 기반
  - 제거: sparse attention, retrieval/streaming split, head_rank_table, sub-chunk statistics
  - 유지: paged block table access, multi-head attention, warp-level softmax
- [ ] `engine/csrc/attention/kv_cache_utils.h` 생성
  - `KvCacheDataType` enum: BASE, INT8, INT4, ZINT8, ZINT4
  - `load_8bits_kv_cache_vec()` / `store_8bits_kv_cache_vec()`
  - `load_4bits_kv_cache_vec()` / `store_4bits_kv_cache_vec()`
  - Zero-point 지원
- [ ] FP16/BF16 computation + INT4/INT8 KV cache 지원
- [ ] **검증**: paged attention vs SDPA 결과 비교

### 3C. Fused RoPE + KV Cache Update 커널 (OmniServe에서 추출)

- [ ] `engine/csrc/cache/rope_update_kv.cu` 생성
  - 3개 연산 fuse: RoPE 적용 + KV quantization + paged cache write
  - 제거: streaming/retrieval split, head_rank_table, bias (Llama에 QKV bias 없음)
  - 유지: RoPE base/dim 설정, paged block write, optional INT4/INT8 KV quantization
- [ ] **검증**: fused 커널 vs 개별 (RoPE + cache write) 결과 비교

### 3D. Attention Layer 업데이트

- [ ] `engine/model/attention.py` 수정:
  - Prefill: FlashAttention/SDPA + paged cache write
  - Decode: fused RoPE+KV write + paged attention 커널
  - Phase 2의 concat KV cache를 paged로 교체

### 3E. 빌드 시스템 & Ops 업데이트

- [ ] `engine/setup.py`: attention, cache CUDA extensions 추가
- [ ] `engine/ops.py`: `paged_attention()`, `rope_update_kv_cache()` 추가

### 3F. 검증

- [ ] Paged KV cache로 생성한 텍스트가 Phase 2 결과와 동일
- [ ] KV cache INT4/INT8 양자화 시 품질 저하 정도 측정

---

## Phase 4: Continuous Batching

### 4A. Request 관리

- [ ] `engine/core/request.py` 생성:
  - `RequestStatus`: WAITING, PREFILLING, DECODING, FINISHED_STOP, FINISHED_LENGTH, FINISHED_ABORT
  - `Request`: request_id, prompt_token_ids, output_token_ids, status, block_table, max_tokens, sampling_params
  - `SamplingParams`: temperature, top_k, top_p

### 4B. FCFS Scheduler

- [ ] `engine/core/scheduler.py` 생성:
  - `Scheduler`: waiting queue (deque), running list
  - `add_request(request)`: waiting queue에 추가
  - `schedule()` → `ScheduleOutput`:
    1. 모든 running decode 요청 계속
    2. FCFS로 waiting prefill 요청 스케줄
    3. KV cache 메모리 예산 확인
  - Preemption: 없음 (향후 추가)

### 4C. Engine Loop

- [ ] `engine/core/engine.py` 생성:
  - `LLMEngine.__init__(model_dir, max_batch_size, max_seq_len)`
  - `add_request(request_id, prompt, sampling_params)`
  - `step()` → `List[RequestOutput]`:
    1. Schedule → 2. Batch 준비 → 3. Forward → 4. Sample → 5. 상태 업데이트

### 4D. 검증

- [ ] 3개 요청 동시 제출, 모두 올바르게 완료 확인
- [ ] 메모리 부족 시 요청 대기 → 해제 후 재개 확인

---

## Phase 5: OpenAI 호환 Serving API

### 5A. API

- [ ] `engine/serving/api.py` 생성:
  - Pydantic: `ChatCompletionRequest`, `ChatCompletionResponse`
  - `POST /v1/chat/completions`: non-streaming + SSE streaming
  - `GET /v1/models`: 모델 정보
  - 에러 처리: 400, 500

### 5B. Server

- [ ] `engine/serving/server.py` 생성:
  - CLI: `--model`, `--host`, `--port`, `--max-batch-size`, `--max-seq-len`
  - Engine 초기화
  - Background engine step loop (asyncio)
  - Uvicorn 시작

### 5C. 검증

- [ ] `curl` 또는 OpenAI SDK로 `/v1/chat/completions` 테스트
- [ ] Streaming 응답 확인
- [ ] 동시 다중 요청 처리 확인

---

## Phase 6: 통합 테스트 & 벤치마크

### 6A. Unit Tests

- [ ] `tests/test_marlin_gemm.py`: 커널 정확도
- [ ] `tests/test_quantized_linear.py`: QuantizedLinear 모듈
- [ ] `tests/test_model_loader.py`: 모델 로딩
- [ ] `tests/test_attention.py`: Attention (paged vs non-paged)
- [ ] `tests/test_generation.py`: end-to-end 생성
- [ ] `tests/test_scheduler.py`: 스케줄러
- [ ] `tests/test_kv_cache.py`: KV cache 관리

### 6B. Integration Test

- [ ] 작은 모델 (Llama-3.2-1B 등) W4A16 압축 → 엔진 로딩 → 텍스트 생성 → 품질 확인
- [ ] PPL 비교: compressor 평가 결과와 일치 확인

### 6C. Benchmark

- [ ] Token throughput (tokens/sec) by batch size
- [ ] Time to first token (TTFT)
- [ ] HuggingFace FP16 추론 대비 비교

---

## 참조 소스

### CUDA 커널 참조 (읽기 전용)

| 대상 | 참조 경로 | 핵심 파일 |
|------|-----------|-----------|
| Marlin GEMM 커널 | `/home/ho/ai/refs/vllm/csrc/quantization/marlin/` | `marlin_template.h`, `dequant.h`, `marlin_mma.h`, `marlin.cu` |
| Marlin Weight Repack | `/home/ho/ai/refs/vllm/csrc/quantization/marlin/` | `gptq_marlin_repack.cu` |
| Paged Attention | `/home/ho/ai/refs/omniserve/kernels/csrc/fused_attention/fused_attention_pure_dense/` | `decoderMaskedMultiheadAttention.cu`, `decoderMaskedMultiheadAttentionTemplate.hpp` |
| KV Cache 유틸 | `/home/ho/ai/refs/omniserve/kernels/csrc/fused_attention/common/` | `kvCacheUtils.h`, `decoderMaskedMultiheadAttentionUtils.h` |
| Fused RoPE+KV | `/home/ho/ai/refs/omniserve/kernels/csrc/fused_attention/fused_attention_pure_dense/` | `update_kv_cache.cu`, `applyBiasRopeUpdateKVCache.h` |

### Python 참조 (읽기 전용)

| 대상 | 참조 경로 | 핵심 함수 |
|------|-----------|-----------|
| Marlin Python 유틸 | `/home/ho/ai/refs/vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py` | `apply_gptq_marlin_linear()`, `marlin_permute_scales()`, `get_scale_perms()` |
| Paged Attention Ops | `/home/ho/ai/refs/vllm/vllm/v1/attention/ops/paged_attn.py` | `PagedAttention.split_kv_cache()`, `write_to_paged_cache()` |
| KV Cache Manager | `/home/ho/ai/refs/vllm/vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()` |
| Scheduler | `/home/ho/ai/refs/vllm/vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` |
| Block Table | `/home/ho/ai/refs/vllm/vllm/v1/worker/block_table.py` | `BlockTable` 구조 |
| OmniServe KV Config | `/home/ho/ai/refs/omniserve/omniserve/worker/cache_engine.py` | INT4/INT8 KV cache config |

### 재사용할 기존 EdgeLLM 코드

| 유틸 | 경로 | 용도 |
|------|------|------|
| `LLMConfig` | `compressor/nn/struct/llm.py` | 모델 아키텍처 정보 참조 |
| `QuantArgs` | `compressor/config/quant.py` | 양자화 파라미터 (bits, symmetric, group_shapes) |
| `infer_format` / `resolve` | `compressor/packer/format/__init__.py` | 양자화 스킴 추론 / 포맷 핸들러 해석 |
| `W4A16Format` | `compressor/packer/format/w4a16.py` | Marlin 호환 W4A16 패킹 |
