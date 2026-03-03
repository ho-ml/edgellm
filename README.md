# EdgeLLM

Edge 디바이스 배포를 위한 LLM 양자화 프레임워크입니다.

GPTQ 양자화와 Hadamard rotation, SmoothQuant, channel reordering 등의 사전 최적화 기법을 결합하여 모델 품질 저하를 최소화하면서 LLM을 압축합니다. 레이어 단위 순차 처리로 대형 모델도 단일 GPU에서 압축할 수 있으며, safetensors 포맷의 int4/int8 패킹된 배포용 모델을 생성합니다.

**지원 포맷:** W4A16 · W4A8 · W8A8

## 1. Installation

**Requirements:** Python >= 3.10, PyTorch >= 2.2.0, CUDA 지원 GPU

```bash
git clone https://github.com/ho-ml/edgellm.git
cd edgellm
pip install .
```

선택 의존성:

```bash
pip install ".[flash-attn]"   # Flash Attention
pip install ".[dev]"          # pytest, black, ruff
```

## 2. Quick start

### CLI

```bash
# 기본 설정 (W4A8, Llama-3.2-1B)
./examples/run.sh

# 모델 및 설정 지정
CONFIG=examples/configs/w4a16.yaml MODEL=meta-llama/Llama-2-7b ./examples/run.sh

# 디버그 모드
./examples/run.sh --debug
```

직접 실행:

```bash
python -m compressor.compress --config examples/configs/w4a8.yaml --model meta-llama/Llama-3.2-1B
```

### Python API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressor.config import CompressorConfig, EvalConfig
from compressor.compress import Compressor
from compressor.evaluate import Evaluator

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=torch.float16, device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 압축
config = CompressorConfig.from_yaml("examples/configs/w4a8.yaml")
compressor = Compressor(config, model, tokenizer)
compressed_model = compressor.run()

# 평가
eval_config = EvalConfig.from_yaml("examples/configs/w4a8.yaml")
Evaluator(eval_config, compressed_model, tokenizer).run()
```

### Evaluation

```bash
python -m compressor.evaluate --config examples/configs/w4a8.yaml --model meta-llama/Llama-3.2-1B
```

## 3. Quantization format

EdgeLLM은 세 가지 양자화 포맷을 지원합니다. 포맷마다 적용되는 기법의 조합이 다르며, 압축률과 정확도 사이의 트레이드오프가 달라집니다.

| 포맷 | Weights | Activations | 사전 최적화 |
|------|---------|-------------|-------------|
| W4A16 | 4-bit group-128 | FP16 (유지) | 없음 |
| W8A8 | 8-bit per-channel | 8-bit dynamic per-token | SmoothQuant |
| W4A8 | 4-bit progressive | 8-bit dynamic per-token | Rotation + Reorder + Smooth |

### 3-1. W4A16 — Weight-Only 4-bit Quantization

가장 단순한 구성입니다. 가중치만 4-bit로 양자화하고 활성화는 FP16을 그대로 유지합니다. 사전 최적화 없이 GPTQ만 적용하므로 압축 속도가 빠르고, 활성화 정밀도가 보존되어 안정적입니다.

#### 적용 기법

**GPTQ (Gradient-based Post-Training Quantization)**

RTN(Round-to-Nearest)은 각 가중치를 독립적으로 가장 가까운 양자화 값으로 반올림합니다. 단순하지만, 한 가중치의 반올림 오차가 다른 가중치와 어떻게 상호작용하는지를 고려하지 않아 오차가 누적됩니다.

GPTQ는 이 문제를 2차 정보(Hessian)를 이용해 해결합니다. 핵심 아이디어는 "이미 양자화한 가중치의 오차를 아직 양자화하지 않은 가중치에 보상하는 것"입니다.

캘리브레이션 데이터의 입력 활성화 `X`로부터 Hessian 행렬을 계산합니다:

```
H = (1/N) · Σ xᵢ · xᵢᵀ
```

이 행렬의 역행렬 `H⁻¹`은 각 가중치의 양자화 오차가 출력에 미치는 영향의 크기를 나타냅니다. GPTQ는 가중치를 열 단위로 순차 처리하면서, 각 열의 양자화 오차를 아직 처리하지 않은 열들에 분배합니다:

```
for each column i:
    qᵢ = quantize(wᵢ)                    # 현재 열 양자화
    errᵢ = (wᵢ - qᵢ) / H⁻¹[i,i]         # 오차를 Hessian 역행렬로 정규화
    w[j>i] -= errᵢ · H⁻¹[i, j]           # 미래 열에 오차 보상
```

이를 통해 전체 레이어의 출력 재구성 오차를 최소화합니다.

**Group Quantization (group-128)**

전체 채널에 하나의 scale/zero를 사용하면(per-channel), 채널 내 값의 범위 차이로 인해 정밀도가 떨어집니다. Group quantization은 채널을 128개씩 그룹으로 나누어 각 그룹에 독립적인 scale과 zero point를 할당합니다. 그룹이 작을수록 정밀도는 높아지지만 메타데이터 오버헤드가 증가하므로, 128은 정밀도와 효율의 균형점입니다.

#### Configuration

```yaml
# examples/configs/w4a16.yaml
calib:
  data: "pileval"
  num_samples: 128
  seq_length: 2048
  seed: 42

quant:
  weight:
    args:
      bits: 4
      strategy: "group"
      group_shapes: [[1, 128]]       # group-128
      symmetric: true
      observer: "memoryless-minmax"
    mod:
      method: "gptq"
      block_size: 128                 # 열을 128개씩 블록 처리
      perc_damp: 0.01                 # Hessian 대각선에 1% damping

eval:
  ppl:
    datasets: [wikitext]
    num_samples: 128
    seq_length: 2048
  compare_baseline: true

pack:
  enabled: true
  output_dir: "output/w4a16"
```

### 3-2. W8A8 — 8-bit Weight + 8-bit Activation

가중치와 활성화 모두 8-bit로 양자화합니다. INT8 행렬 곱셈을 활용할 수 있어 실제 추론 속도 향상이 가능합니다. 다만 활성화를 양자화하면 활성화 outlier가 심각한 정확도 저하를 유발하므로, SmoothQuant으로 이 문제를 먼저 해결합니다.

#### 적용 기법

**SmoothQuant — 활성화 outlier를 가중치로 이전**

Transformer의 활성화에는 특정 채널에 극단적으로 큰 값(outlier)이 나타납니다. 이 outlier가 있으면 양자화 범위가 넓어져 대부분의 값이 소수의 양자화 bin에 몰리게 되고, 정밀도가 크게 떨어집니다.

SmoothQuant의 핵심 아이디어는 활성화의 outlier를 가중치 쪽으로 옮기는 것입니다. 가중치는 정적(static)이므로 넓은 범위를 더 잘 수용할 수 있습니다. 수학적으로, 선형 레이어 `Y = X · W`에서 채널별 스케일링 벡터 `s`를 도입합니다:

```
Y = X · W = (X · diag(s)⁻¹) · (diag(s) · W) = X' · W'
```

`X' = X / s`로 활성화 범위를 줄이고, `W' = s · W`로 가중치 범위를 늘립니다. 스케일 `s`는 활성화와 가중치의 분포를 기반으로 결정합니다:

```
sⱼ = (amax_j ^ α) / (wmax_j ^ β)
```

- `α`가 클수록 활성화 outlier를 더 많이 흡수합니다
- `β`가 클수록 가중치의 원래 분포를 보존합니다
- W8A8에서는 `α=0.85, β=0.15`로 활성화 smoothing에 비중을 둡니다

SmoothQuant은 여러 위치에 적용됩니다:
- `smooth_attn`: attention key의 outlier를 query로 이전
- `smooth_qkv`: input layernorm → QKV projection 사이
- `smooth_ffn`: post-attention layernorm → gate/up projection 사이

**GPTQ (per-channel)**

W4A16과 동일한 GPTQ 알고리즘을 적용하되, 8-bit per-channel 양자화를 사용합니다. 8-bit는 4-bit보다 양자화 오차가 작으므로, 그룹 양자화 없이 per-channel만으로 충분한 정밀도를 확보합니다.

**Dynamic Activation Quantization (per-token)**

활성화는 입력마다 분포가 달라지므로 정적 scale로는 대응할 수 없습니다. 추론 시점에 각 토큰별로 활성화의 min/max를 계산하고 scale을 동적으로 결정합니다. 오버헤드가 있지만, 토큰별로 최적의 양자화 범위를 사용하므로 정밀도가 높습니다.

#### Configuration

```yaml
# examples/configs/w8a8.yaml
calib:
  data: "pileval"
  num_samples: 128
  seq_length: 512
  seed: 42

transform:
  smooth:
    enabled: true
    smooth_attn: true               # K outlier → Q로 이전
    smooth_qkv: true                # input norm → QKV smoothing
    smooth_ffn: true                # post-attn norm → gate/up smoothing
    smooth_vo: false
    smooth_down: false
    proj_alpha: 0.85                # 활성화 smoothing 비중 높음
    proj_beta: 0.15
    attn_alpha: 0.5
    attn_beta: 0

quant:
  weight:
    args:
      bits: 8
      strategy: "channel"           # per-channel (그룹 불필요)
      symmetric: true
      observer: "memoryless-minmax"
    mod:
      method: "gptq"
      block_size: 128
      perc_damp: 0.01
  input:
    args:
      bits: 8
      symmetric: true
      strategy: "token"             # 토큰 단위 dynamic quantization
      dynamic: true
      observer: "minmax"

eval:
  ppl:
    datasets: [wikitext]
    num_samples: 128
    seq_length: 2048
  compare_baseline: true

pack:
  enabled: true
  output_dir: "output/w8a8"
```

### 3-3. W4A8 — Progressive 4-bit Weight + 8-bit Activation

가장 공격적인 압축 포맷입니다. 가중치를 4-bit까지 줄이면서 활성화도 8-bit로 양자화합니다. 4-bit 양자화는 정밀도 손실이 크기 때문에, 세 가지 사전 최적화(rotation, reorder, smooth)를 모두 적용하고, progressive quantization으로 정밀도를 보완합니다.

#### 적용 기법

**1) Hadamard Rotation — 값 분포 균일화**

양자화의 적은 가중치 분포가 균일할수록 유리합니다. 특정 차원에 에너지가 집중되어 있으면 그 차원의 양자화 오차가 전체 정확도를 지배합니다.

Hadamard rotation은 직교 변환 행렬 `H`를 가중치에 곱하여 값의 분포를 차원 간에 골고루 분산시킵니다:

```
W' = H · W      (출력 채널 rotation)
W' = W · Hᵀ     (입력 채널 rotation)
```

Hadamard 행렬은 {+1, -1}로만 구성된 직교 행렬로, 연산이 효율적이면서도 에너지를 균일하게 분산합니다. 직교 변환이므로 정보 손실이 없습니다 (`||Y|| = ||Y'||`). 또한 rotation 과정에서 LayerNorm의 가중치를 인접 선형 레이어에 융합(fuse)하여 추론 시 불필요한 연산을 제거합니다.

**2) Channel Reordering — 그룹 내 분포 균질화**

Group quantization에서 128개 채널이 하나의 scale을 공유합니다. 만약 한 그룹 안에 크기가 극단적으로 다른 채널들이 섞이면, 큰 값에 맞춰진 scale 때문에 작은 값의 정밀도가 희생됩니다.

Channel reordering은 캘리브레이션 데이터로 각 채널의 활성화 크기(absmax)를 측정한 뒤, 크기가 비슷한 채널끼리 인접하도록 재배열합니다. GQA(Grouped Query Attention) 구조에서는 헤드 내부에서만 재배열하여 어텐션 구조를 유지합니다.

```
rank = absmax.argsort()     # 채널 중요도 기준 정렬
W' = W[:, rank]             # 입력 채널 재배열
```

결과적으로 각 그룹 내 값의 범위가 좁아져서 그룹 양자화의 정밀도가 향상됩니다.

**3) SmoothQuant**

W8A8과 동일한 원리로 활성화 outlier를 가중치로 이전합니다. W4A8에서는 `α=0.3, β=0.7`으로 가중치 보존에 더 비중을 둡니다. 4-bit 양자화에서는 가중치 범위가 매우 제한적이므로, 가중치에 너무 많은 outlier를 옮기면 오히려 역효과가 발생하기 때문입니다. 추가로 `smooth_down`을 활성화하여 FFN의 up/gate → down projection 사이에도 smoothing을 적용합니다.

**4) Progressive Quantization — 2단계 그룹 양자화**

4-bit 양자화에서 단일 group-128 scale만 사용하면, 채널 간 범위 차이가 큰 경우 오버플로우가 발생하거나 정밀도가 크게 떨어집니다. Progressive quantization은 2단계로 나누어 이 문제를 해결합니다:

```
Level 0 (per-channel):  scale_0 = max(|w|) / pmax     # 채널별 대략적 정규화
Level 1 (group-128):    scale_1, zero = qparams(w / scale_0)   # 정규화된 공간에서 세밀한 양자화
```

1단계에서 per-channel scale로 값을 `intermediate_bits`(8-bit) 범위 내로 정규화하고, 2단계에서 group-128 scale로 세밀하게 양자화합니다. 이를 통해 넓은 범위의 가중치도 오버플로우 없이 4-bit로 표현할 수 있습니다.

**5) GPTQ + Dynamic Activation Quantization**

위의 모든 사전 최적화가 적용된 상태에서 GPTQ로 가중치를 4-bit 양자화하고, 활성화는 토큰 단위 dynamic 8-bit 양자화를 적용합니다. 출력 활성화에는 group-128 기반의 4-bit dynamic 양자화도 추가로 적용됩니다.

#### Configuration

```yaml
# examples/configs/w4a8.yaml
calib:
  data: "pileval"
  num_samples: 128
  seq_length: 1024
  seed: 42

transform:
  rotate:
    enabled: true
    rotate_out: true                # 출력 채널 rotation (v_proj 등)
    rotate_down: false

  reorder:
    enabled: true
    reorder_out: false
    reorder_down: true              # down_proj 입력 채널 재배열

  smooth:
    enabled: true
    smooth_attn: true               # K outlier → Q 이전
    smooth_down: true               # up/gate → down smoothing
    proj_alpha: 0.3                 # 가중치 보존 비중 높음 (4-bit이므로)
    proj_beta: 0.7

quant:
  weight:
    args:
      bits: 4
      symmetric: false
      strategy: "group"
      group_shapes: [[1, -1], [1, 128]]  # Level 0: per-channel, Level 1: group-128
      intermediate_bits: 8                 # 중간 정규화 범위
      observer: "memoryless-minmax"
    mod:
      method: "gptq"
      block_size: 128
      perc_damp: 0.01

  input:                            # 활성화 8-bit dynamic
    args:
      bits: 8
      symmetric: true
      strategy: "token"
      dynamic: true
      observer: "minmax"

  output:                           # 출력 활성화 4-bit dynamic
    args:
      bits: 4
      symmetric: false
      strategy: "group"
      group_shapes: [[1, 128]]
      dynamic: true
      observer: "minmax"
    mod:
      skips: "q_proj"              # q_proj 출력은 양자화 제외

eval:
  ppl:
    datasets: [wikitext]
    num_samples: 128
    seq_length: 2048
  compare_baseline: true

pack:
  enabled: true
  output_dir: "output/w4a8"
```

#### 기법 적용 순서와 이유

```
Rotation ──→ Reorder ──→ Smooth ──→ Activation Quant ──→ Weight Quant (GPTQ)
```

Rotation이 가장 먼저 적용되는 이유는 값 분포를 전역적으로 균일화해야 이후 reorder와 smooth가 효과적으로 동작하기 때문입니다. Reorder는 rotation 이후의 분포에서 그룹 내 균질성을 높이고, smooth는 reorder 이후에도 남은 활성화 outlier를 처리합니다. 이 모든 최적화가 완료된 뒤에 GPTQ가 Hessian 기반으로 최적의 양자화를 진행합니다.

## 4. 프로젝트 구조

```
edgellm/
├── compressor/                  # 메인 패키지
│   ├── compress.py              # Compressor — 압축 파이프라인 오케스트레이터
│   ├── evaluate.py              # Evaluator — PPL 및 lm-eval 벤치마크
│   ├── calib/                   # 캘리브레이션 데이터 로더
│   │   └── loader.py            #   레이어별 hidden state 전파
│   ├── config/                  # 설정 (Pydantic dataclass)
│   │   ├── compressor.py        #   CompressorConfig — 전체 파이프라인 설정
│   │   ├── quant.py             #   QuantArgs, QuantConfig — 양자화 파라미터
│   │   ├── calib.py             #   CalibConfig — 캘리브레이션 설정
│   │   ├── evaluator.py         #   EvalConfig — 평가 설정
│   │   └── pack.py              #   PackConfig — 패킹 설정
│   ├── data/                    # 데이터셋
│   │   ├── wikitext.py          #   WikiText-2
│   │   ├── c4.py                #   C4
│   │   ├── pileval.py           #   PileEval
│   │   ├── ultrachat.py         #   UltraChat
│   │   └── custom.py            #   사용자 정의 데이터
│   ├── nn/                      # 모델 구조 분석
│   │   ├── struct/              #   LLMStruct, DecoderStruct — 모델 introspection
│   │   └── patch/               #   RoPE, LayerNorm 패치
│   ├── modifier/                # 압축 기법 (Modifier 인터페이스)
│   │   ├── rotate/              #   Hadamard rotation
│   │   ├── reorder/             #   Channel reordering
│   │   ├── smooth/              #   SmoothQuant
│   │   ├── gptq/                #   GPTQ (Hessian 계산 포함)
│   │   ├── rtn/                 #   Round-to-Nearest
│   │   ├── weight/              #   가중치 양자화 래퍼 + progressive
│   │   └── activation/          #   동적 활성화 양자화
│   ├── observers/               # 양자화 파라미터 결정
│   │   ├── minmax.py            #   MinMax observer (static/memoryless)
│   │   └── mse.py               #   MSE observer (grid search 기반)
│   ├── packer/                  # 배포용 모델 패킹
│   │   ├── convert.py           #   fake-quant → integer 변환
│   │   └── pack.py              #   int4 → int8 비트 패킹, safetensors 저장
│   └── utils/                   # 유틸리티
│       ├── quant.py             #   quantize, dequantize, fake_quantize
│       ├── pack.py              #   포맷 추론, MMA 레이아웃
│       ├── device.py            #   GPU/CPU 디바이스 관리
│       └── registry.py          #   데이터셋/observer 레지스트리
├── examples/
│   ├── run.sh                   # 실행 스크립트
│   └── configs/                 # YAML 설정 파일 (w4a8, w4a16, w8a8)
├── docs/                        # 설계 문서
└── pyproject.toml
```
