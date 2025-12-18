# 🔧 Korean Tool Calling LoRA Training

Kanana, Qwen 등 다양한 sLLM에 Tool Calling 능력을 학습시키기 위한 LoRA 파인튜닝 스크립트입니다.

## 📊 데이터셋

HuggingFace Hub에서 자동으로 로드됩니다:

| 데이터셋 | 설명 | 샘플 수 |
|---------|------|--------|
| [NotoriousH2/instructkr-toolflow](https://huggingface.co/datasets/NotoriousH2/instructkr-toolflow) | 한국어 헬프데스크 시나리오 | ~1,000 |
| [NotoriousH2/instructkr-when2call](https://huggingface.co/datasets/NotoriousH2/instructkr-when2call) | Tool 호출 여부 판단 | ~15,000 |
| [NotoriousH2/instructkr-apigen](https://huggingface.co/datasets/NotoriousH2/instructkr-apigen) | API 생성 멀티턴 대화 | ~5,000 |

### 데이터 형식

데이터셋은 `messages` + `tools` 형식이며, 학습 시 자동으로 Llama 3 chat template `text`로 변환됩니다:

```json
{
  "messages": [
    {"role": "user", "content": "오늘 날씨 알려줘"},
    {"role": "assistant", "content": "<function=get_weather>{\"location\": \"서울\"}</function>"}
  ],
  "tools": [
    "{\"name\": \"get_weather\", \"description\": \"날씨 조회\", ...}"
  ]
}
```

→ 자동 변환 →

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You have access to the following functions:
Use the function 'get_weather' to '날씨 조회'
{"name": "get_weather", ...}
...<|eot_id|><|start_header_id|>user<|end_header_id|>

오늘 날씨 알려줘<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<function=get_weather>{"location": "서울"}</function><|eot_id|>
```

## 🚀 Quick Start

### 1. 환경 설정

⚠️ **설치 순서가 중요합니다!** Unsloth는 반드시 마지막에 설치해야 합니다.

```bash
# Step 1: 기본 패키지 설치
pip install -r requirements_lora.txt

# Step 2: Unsloth 설치 (반드시 마지막에!)
pip install "unsloth[cu128-torch271] @ git+https://github.com/unslothai/unsloth.git"

# Step 3: Flash Attention 설치 (선택, 성능 향상)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

> **Note**: Unsloth는 transformers, torch 등을 monkey-patch하므로 다른 패키지 설치 후에 설치해야 합니다.

### 2. 학습 실행

```bash
# 기본 실행 (Kanana 2.1B, rank 16)
python train_lora.py

# LoRA Rank 변경
python train_lora.py --rank 8
python train_lora.py --rank 32
python train_lora.py --rank 64

# 다른 모델 사용
python train_lora.py --model Qwen/Qwen2.5-3B-Instruct --rank 16
```

### 3. 결과 시각화

```bash
# 로스 커브 비교
python visualize_loss.py experiments/*/training_log.csv

# 이미지 저장
python visualize_loss.py -o comparison.png experiments/*/training_log.csv

# 요약 테이블도 저장
python visualize_loss.py -o comparison.png -s summary.csv experiments/*/training_log.csv
```

## ⚙️ 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--model` | `kakaocorp/kanana-1.5-2.1b-instruct-2505` | HuggingFace 모델 ID |
| `--rank` | 16 | LoRA rank (alpha는 자동으로 rank×2) |
| `--lr` | 2e-4 | Learning rate |
| `--epochs` | 3 | 학습 에폭 수 |
| `--batch_size` | 4 | 배치 사이즈 |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--max_seq_length` | 9000 | 최대 시퀀스 길이 |

### 전체 옵션 보기

```bash
python train_lora.py --help
```

## 📁 출력 구조

```
experiments/
├── kanana_1.5_2.1b_instruct_2505_r16_a32_lr2em04_ep3_20251218_143022/
│   ├── experiment_config.json    # 실험 설정
│   ├── training_log.csv          # 로스 로그
│   ├── checkpoint-200/           # 중간 체크포인트
│   ├── checkpoint-400/
│   └── final_model/              # 최종 LoRA 어댑터
└── ...
```

## 🔬 실험 예시: LoRA Rank 비교

여러 GPU 클라우드에서 병렬로 실행:

```bash
# GPU 1
python train_lora.py --rank 4

# GPU 2
python train_lora.py --rank 8

# GPU 3
python train_lora.py --rank 16

# GPU 4
python train_lora.py --rank 32

# GPU 5
python train_lora.py --rank 64
```

결과 CSV 파일들을 모아서 시각화:

```bash
python visualize_loss.py -o rank_comparison.png -s summary.csv experiments/*/training_log.csv
```

## 📝 학습 방식: 샌드위치 마스킹

Tool Calling 학습에서 중요한 점:
- `system`, `user`, `tool/ipython` 블록 → **마스킹** (-100)
- `assistant` 블록 → **학습 대상**

이를 통해 Tool 결과를 환각(Hallucination)하는 것을 방지합니다.

## 🛠️ 요구사항

- Python 3.12+
- CUDA 12.x
- GPU Memory: 16GB+ 권장

## 📜 License

MIT License

