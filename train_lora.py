#!/usr/bin/env python3
"""
Kanana Tool Calling LoRA Training Script
다양한 LoRA Rank와 모델에 대한 실험을 위한 스크립트

사용법:
    python train_lora.py --model kakaocorp/kanana-1.5-2.1b-instruct-2505 --rank 16
    python train_lora.py --model kakaocorp/kanana-1.5-2.1b-instruct-2505 --rank 32 --lr 1e-4
    
예시 (다양한 rank 실험):
    python train_lora.py --rank 4
    python train_lora.py --rank 8
    python train_lora.py --rank 16
    python train_lora.py --rank 32
    python train_lora.py --rank 64

데이터셋:
    기본적으로 HuggingFace Hub에서 데이터를 로드합니다:
    - NotoriousH2/instructkr-toolflow
    - NotoriousH2/instructkr-when2call
    - NotoriousH2/instructkr-apigen
    
    로컬 파일을 사용하려면 --local_data 플래그를 추가하세요.
"""

# ============================================================
# ⚠️ Unsloth는 반드시 다른 패키지보다 먼저 import해야 합니다!
# (Unsloth가 transformers, torch 등을 monkey-patch하기 때문)
# ============================================================
from unsloth import FastLanguageModel

import argparse
import json
import os
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig


# ============================================================
# HuggingFace Hub 데이터셋 ID
# ============================================================
HF_DATASETS = [
    "NotoriousH2/instructkr-toolflow",
    "NotoriousH2/instructkr-when2call", 
    "NotoriousH2/instructkr-apigen",
]


# ============================================================
# Tool Calling 시스템 프롬프트 템플릿
# ============================================================
TOOL_SYSTEM_PROMPT_TEMPLATE = """You have access to the following functions:

{tools_description}

Think very carefully before calling functions.
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line"""


# ============================================================
# messages + tools → text 변환 함수
# ============================================================
def format_tools_for_system_prompt(tools: list) -> str:
    """tools 리스트를 시스템 프롬프트용 문자열로 변환"""
    if not tools:
        return ""
    
    tools_descriptions = []
    for tool in tools:
        # tool이 문자열인 경우 JSON 파싱
        if isinstance(tool, str):
            try:
                tool_dict = json.loads(tool)
            except json.JSONDecodeError:
                continue
        else:
            tool_dict = tool
        
        name = tool_dict.get("name", "")
        description = tool_dict.get("description", "")
        
        # parameters를 JSON Schema 형식으로 변환
        params = tool_dict.get("parameters", {})
        if "properties" in params:
            # type을 dict → object로 변환
            if params.get("type") == "dict":
                params["type"] = "object"
            # properties 내 type 변환
            for prop_name, prop_value in params.get("properties", {}).items():
                if prop_value.get("type") == "str":
                    prop_value["type"] = "string"
                elif prop_value.get("type") == "int":
                    prop_value["type"] = "integer"
        
        tool_json = json.dumps({
            "name": name,
            "description": description,
            "parameters": params,
            "required": tool_dict.get("required", [])
        }, ensure_ascii=False)
        
        tools_descriptions.append(f"Use the function '{name}' to '{description}'\n{tool_json}")
    
    return "\n\n".join(tools_descriptions)


def parse_tools(tools) -> list:
    """
    다양한 형식의 tools를 표준 리스트로 변환
    
    지원 형식:
    - ["{json1}", "{json2}"] - 문자열 리스트
    - "[{json1}, {json2}]" - 전체가 하나의 JSON 문자열
    - [{"name": ...}, ...] - 이미 파싱된 dict 리스트
    - None 또는 빈 값
    """
    if not tools:
        return []
    
    # 문자열인 경우 (전체 배열이 JSON 문자열로 인코딩된 경우)
    if isinstance(tools, str):
        try:
            parsed = json.loads(tools)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            return []
    
    # 리스트인 경우
    if isinstance(tools, list):
        return tools
    
    return []


def convert_messages_tools_to_text(messages: list, tools, tokenizer) -> str:
    """
    messages + tools 형식을 Llama 3 chat template text로 변환
    
    Args:
        messages: [{"role": "user/assistant/system", "content": "..."}]
        tools: 다양한 형식 지원 (리스트, JSON 문자열 등)
        tokenizer: 토크나이저 (chat_template 적용용)
    
    Returns:
        Llama 3 포맷의 text 문자열
    """
    # tools를 표준 형식으로 파싱
    parsed_tools = parse_tools(tools)
    
    # tools가 있으면 시스템 프롬프트 생성
    formatted_messages = []
    
    if parsed_tools:
        tools_description = format_tools_for_system_prompt(parsed_tools)
        if tools_description:
            system_content = TOOL_SYSTEM_PROMPT_TEMPLATE.format(tools_description=tools_description)
            formatted_messages.append({"role": "system", "content": system_content})
    
    # 기존 messages 추가 (이미 system이 있으면 병합 고려)
    for msg in messages:
        # msg가 dict가 아니면 스킵
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # role이 없거나 content가 없으면 스킵
        if not role:
            continue
        
        # tool role은 ipython으로 변환 (Llama 3 형식)
        if role == "tool":
            role = "ipython"
        
        # 기존 system message가 있으면 tools system과 병합
        if role == "system" and formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] = formatted_messages[0]["content"] + "\n\n" + content
        else:
            formatted_messages.append({"role": role, "content": content})
    
    # chat template 적용
    try:
        text = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        # fallback: 수동으로 Llama 3 형식 생성
        text = "<|begin_of_text|>"
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    return text


def convert_dataset_to_text(dataset, tokenizer):
    """
    Dataset의 messages + tools를 text로 변환
    
    지원하는 형식:
    - {"text": "..."} - 이미 변환된 경우 그대로 사용
    - {"messages": [...], "tools": [...]} - 변환 필요
    """
    def convert_example(example):
        # 이미 text 필드가 있으면 그대로 사용
        if "text" in example and example["text"]:
            return example
        
        messages = example.get("messages", [])
        tools = example.get("tools", [])
        
        text = convert_messages_tools_to_text(messages, tools, tokenizer)
        return {"text": text}
    
    # 첫 번째 샘플로 형식 확인
    first_example = dataset[0]
    if "text" in first_example and first_example["text"]:
        print("✅ 데이터셋에 이미 'text' 필드가 있습니다.")
        return dataset
    
    print("🔄 messages + tools → text 변환 중...")
    converted_dataset = dataset.map(
        convert_example,
        desc="데이터 변환"
    )
    print("✅ 변환 완료")
    
    return converted_dataset


# ============================================================
# CSV 로깅 콜백
# ============================================================
class CSVLoggingCallback(TrainerCallback):
    """학습 로그를 CSV 파일로 저장하는 콜백"""
    
    def __init__(self, csv_path: str, experiment_info: dict):
        self.csv_path = csv_path
        self.experiment_info = experiment_info
        self.logs = []
        
        # CSV 헤더 작성
        with open(self.csv_path, 'w', encoding='utf-8') as f:
            # 메타 정보를 주석으로 저장
            f.write(f"# experiment_info: {json.dumps(experiment_info, ensure_ascii=False)}\n")
            f.write("step,epoch,train_loss,eval_loss,learning_rate,timestamp\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        row = {
            'step': state.global_step,
            'epoch': round(state.epoch, 4) if state.epoch else 0,
            'train_loss': logs.get('loss', ''),
            'eval_loss': logs.get('eval_loss', ''),
            'learning_rate': logs.get('learning_rate', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{row['step']},{row['epoch']},{row['train_loss']},{row['eval_loss']},{row['learning_rate']},{row['timestamp']}\n")


# ============================================================
# Labels 생성 함수 (샌드위치 마스킹)
# ============================================================
def create_labels_for_tool_calling(text: str, tokenizer, max_length: int = 9000) -> dict:
    """
    Tool Calling 학습을 위한 labels 생성 함수 (샌드위치 마스킹)
    
    - system, user, tool/ipython 블록: 마스킹 (-100)
    - assistant 블록: 학습 (실제 token_id)
    """
    encoding = tokenizer(text, truncation=True, max_length=max_length, return_tensors=None)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    labels = [-100] * len(input_ids)
    
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    tool_start = "<|start_header_id|>ipython<|end_header_id|>"
    eot_token = "<|eot_id|>"
    
    assistant_token_ids = tokenizer.encode(assistant_start, add_special_tokens=False)
    tool_token_ids = tokenizer.encode(tool_start, add_special_tokens=False)
    eot_token_ids = tokenizer.encode(eot_token, add_special_tokens=False)
    
    def find_all_positions(sequence, pattern):
        positions = []
        if len(pattern) == 0:
            return positions
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                positions.append(i)
        return positions
    
    assistant_positions = find_all_positions(input_ids, assistant_token_ids)
    tool_positions = find_all_positions(input_ids, tool_token_ids)
    eot_positions = find_all_positions(input_ids, eot_token_ids)
    
    for asst_pos in assistant_positions:
        start_pos = asst_pos + len(assistant_token_ids)
        
        end_pos = None
        for eot_pos in eot_positions:
            if eot_pos > asst_pos:
                end_pos = eot_pos + len(eot_token_ids)
                break
        
        if end_pos is None:
            end_pos = len(input_ids)
        
        is_followed_by_tool = any(asst_pos < tp < end_pos for tp in tool_positions)
        
        if not is_followed_by_tool or asst_pos < min(tool_positions, default=float('inf')):
            for i in range(start_pos, end_pos):
                if i < len(labels):
                    labels[i] = input_ids[i]
    
    for tool_pos in tool_positions:
        start_pos = tool_pos
        end_pos = None
        for eot_pos in eot_positions:
            if eot_pos > tool_pos:
                end_pos = eot_pos + len(eot_token_ids)
                break
        
        if end_pos is None:
            end_pos = len(input_ids)
        
        for i in range(start_pos, end_pos):
            if i < len(labels):
                labels[i] = -100
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def preprocess_function(examples, tokenizer, max_seq_length=9000):
    """배치 전처리 함수"""
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    texts = examples['text']
    if isinstance(texts, str):
        texts = [texts]
    
    for text in texts:
        result = create_labels_for_tool_calling(text, tokenizer, max_seq_length)
        all_input_ids.append(result['input_ids'])
        all_attention_masks.append(result['attention_mask'])
        all_labels.append(result['labels'])
    
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    }


# ============================================================
# 데이터 로드
# ============================================================
def load_training_data_from_hub(dataset_ids: list[str], tokenizer, seed: int = 42):
    """HuggingFace Hub에서 데이터셋 로드 및 병합"""
    all_datasets = []
    
    print("\n📥 HuggingFace Hub에서 데이터셋 로드 중...")
    
    for dataset_id in dataset_ids:
        try:
            ds = load_dataset(dataset_id, split="train")
            print(f"✅ {dataset_id}: {len(ds)}개 샘플 로드")
            
            # 각 데이터셋을 먼저 text로 변환 (스키마 통일)
            ds = convert_dataset_to_text(ds, tokenizer)
            
            # text 필드만 유지 (스키마 차이 문제 해결)
            if "text" in ds.column_names:
                ds = ds.select_columns(["text"])
            
            all_datasets.append(ds)
        except Exception as e:
            print(f"❌ {dataset_id} 로드 실패: {e}")
    
    if not all_datasets:
        raise ValueError("로드된 데이터셋이 없습니다!")
    
    # 데이터셋 병합 (모두 {"text": ...} 형식으로 통일됨)
    if len(all_datasets) == 1:
        combined_dataset = all_datasets[0]
    else:
        combined_dataset = concatenate_datasets(all_datasets)
    
    print(f"\n📊 총 데이터 수: {len(combined_dataset)}개")
    
    # 셔플 및 분할
    combined_dataset = combined_dataset.shuffle(seed=seed)
    
    split = combined_dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = split['train']
    valid_dataset = split['test']
    
    print(f"Train 데이터: {len(train_dataset)}개")
    print(f"Valid 데이터: {len(valid_dataset)}개")
    
    return train_dataset, valid_dataset


def load_training_data_from_local(data_files: list[str], seed: int = 42):
    """로컬 파일에서 학습 데이터 로드 및 분할"""
    all_data = []
    
    print("\n📂 로컬 파일에서 데이터 로드 중...")
    
    for file_path in data_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ {file_path}: {len(data)}개 샘플 로드")
            all_data.extend(data)
        except FileNotFoundError:
            print(f"⚠️ {file_path} 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ {file_path} 로드 실패: {e}")
    
    if not all_data:
        raise ValueError("로드된 데이터가 없습니다!")
    
    print(f"\n📊 총 데이터 수: {len(all_data)}개")
    
    random.seed(seed)
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    valid_data = all_data[split_idx:]
    
    print(f"Train 데이터: {len(train_data)}개")
    print(f"Valid 데이터: {len(valid_data)}개")
    
    # Dataset 객체로 변환
    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(valid_data)
    
    return train_dataset, valid_dataset


# ============================================================
# 실험 이름 생성
# ============================================================
def generate_experiment_name(model_name: str, rank: int, lr: float, epochs: int) -> str:
    """실험 식별을 위한 고유 이름 생성"""
    # 모델 이름에서 핵심 부분 추출
    model_short = model_name.split('/')[-1].replace('-', '_')
    
    # learning rate 포맷팅
    lr_str = f"{lr:.0e}".replace('-', 'm').replace('+', 'p')
    
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{model_short}_r{rank}_a{rank*2}_lr{lr_str}_ep{epochs}_{timestamp}"


# ============================================================
# 메인 학습 함수
# ============================================================
def train(args):
    """메인 학습 함수"""
    
    # 실험 이름 생성
    experiment_name = generate_experiment_name(
        args.model, args.rank, args.lr, args.epochs
    )
    
    print("=" * 70)
    print(f"🚀 실험 시작: {experiment_name}")
    print("=" * 70)
    
    # 출력 디렉토리 설정
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 로그 경로
    csv_log_path = os.path.join(output_dir, f"training_log.csv")
    
    # 실험 정보
    experiment_info = {
        'model': args.model,
        'rank': args.rank,
        'alpha': args.rank * 2,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation': args.grad_accum,
        'max_seq_length': args.max_seq_length,
        'seed': args.seed,
        'experiment_name': experiment_name,
        'data_source': 'local' if args.local_data else 'huggingface_hub',
        'datasets': args.data_files if args.local_data else args.hf_datasets,
        'start_time': datetime.now().isoformat()
    }
    
    # 실험 설정 저장
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    print(f"📝 실험 설정 저장: {config_path}")
    
    # ============================================================
    # 모델 로드
    # ============================================================
    print(f"\n📦 모델 로드 중: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"✅ 모델 로드 완료")
    
    # ============================================================
    # LoRA 설정
    # ============================================================
    print(f"\n🔧 LoRA 설정: rank={args.rank}, alpha={args.rank * 2}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,  # Alpha = Rank * 2
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj"
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.print_trainable_parameters()
    
    # ============================================================
    # 데이터 로드 및 전처리
    # ============================================================
    if args.local_data:
        # 로컬 파일에서 로드 (이미 text 필드가 있는 경우)
        raw_train_dataset, raw_valid_dataset = load_training_data_from_local(
            args.data_files, args.seed
        )
    else:
        # HuggingFace Hub에서 로드 (messages + tools → text 변환 포함)
        raw_train_dataset, raw_valid_dataset = load_training_data_from_hub(
            args.hf_datasets, tokenizer, args.seed
        )
    
    preprocess_fn = partial(
        preprocess_function, 
        tokenizer=tokenizer, 
        max_seq_length=args.max_seq_length
    )
    
    train_dataset = raw_train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_train_dataset.column_names,
        desc="Train 데이터 전처리"
    )
    
    valid_dataset = raw_valid_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_valid_dataset.column_names,
        desc="Valid 데이터 전처리"
    )
    
    print(f"✅ 전처리 완료: Train {len(train_dataset)}, Valid {len(valid_dataset)}")
    
    # ============================================================
    # SFTTrainer 설정
    # ============================================================
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        
        # 학습 설정
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        # 옵티마이저
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_8bit",
        
        # 로깅 및 저장
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        
        # 평가
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 기타
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=args.seed,
        report_to="none",
    )
    
    # CSV 로깅 콜백
    csv_callback = CSVLoggingCallback(csv_log_path, experiment_info)
    
    # Trainer 생성
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=sft_config,
        callbacks=[csv_callback],
    )
    
    # ============================================================
    # 학습 실행
    # ============================================================
    print("\n" + "=" * 70)
    print("🏃 학습 시작")
    print("=" * 70)
    print(f"모델: {args.model}")
    print(f"LoRA Rank: {args.rank}, Alpha: {args.rank * 2}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Epochs: {args.epochs}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Logging Steps: {args.logging_steps}")
    print(f"Eval Steps: {args.eval_steps}")
    print(f"데이터 소스: {'로컬 파일' if args.local_data else 'HuggingFace Hub'}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"CSV 로그: {csv_log_path}")
    print("=" * 70 + "\n")
    
    trainer_stats = trainer.train()
    
    # ============================================================
    # 모델 저장
    # ============================================================
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ 최종 모델 저장: {final_model_path}")
    
    # 학습 완료 정보 저장
    experiment_info['end_time'] = datetime.now().isoformat()
    experiment_info['final_train_loss'] = trainer_stats.training_loss
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"🎉 학습 완료: {experiment_name}")
    print(f"📊 CSV 로그: {csv_log_path}")
    print("=" * 70)
    
    return csv_log_path


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Kanana Tool Calling LoRA Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 모델 설정
    parser.add_argument(
        "--model", type=str,
        default="kakaocorp/kanana-1.5-2.1b-instruct-2505",
        help="HuggingFace 모델 ID 또는 로컬 경로"
    )
    
    # LoRA 설정
    parser.add_argument(
        "--rank", "-r", type=int, default=16,
        help="LoRA rank (alpha는 자동으로 rank*2로 설정됨)"
    )
    
    # 학습 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 사이즈")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=9000, help="최대 시퀀스 길이")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    
    # 로깅/저장 설정
    parser.add_argument("--logging_steps", type=int, default=10, help="로깅 스텝 간격")
    parser.add_argument("--eval_steps", type=int, default=50, help="평가 스텝 간격")
    parser.add_argument("--save_steps", type=int, default=200, help="저장 스텝 간격")
    
    # 데이터 설정
    parser.add_argument(
        "--local_data", action="store_true",
        help="로컬 파일에서 데이터 로드 (기본: HuggingFace Hub에서 로드)"
    )
    parser.add_argument(
        "--hf_datasets", nargs="+",
        default=[
            "NotoriousH2/instructkr-toolflow",
            "NotoriousH2/instructkr-when2call",
            "NotoriousH2/instructkr-apigen"
        ],
        help="HuggingFace Hub 데이터셋 ID들"
    )
    parser.add_argument(
        "--data_files", nargs="+",
        default=[
            "sft_when2call_korean.json",
            "sft_synth_helpdesk.json",
            "sft_apigen_mt_5k_korean.json"
        ],
        help="로컬 학습 데이터 파일 경로들 (--local_data 사용 시)"
    )
    
    # 출력 경로
    parser.add_argument(
        "--output_dir", type=str, default="./experiments",
        help="실험 결과 저장 디렉토리"
    )
    
    # 기타
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # GPU 정보 출력
    print("=" * 70)
    print("🖥️ 시스템 정보")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 70 + "\n")
    
    # 학습 실행
    train(args)


if __name__ == "__main__":
    main()

