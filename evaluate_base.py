#!/usr/bin/env python3
"""
Tool Calling 베이스 모델 평가 스크립트

LoRA 학습 전의 베이스 모델 Tool Calling 성능을 평가합니다.

평가 지표:
- Tool Selection Accuracy: 올바른 Tool을 선택했는지
- Parameter Exact Match: 파라미터가 정확히 일치하는지
- When-to-Call Accuracy: Tool 호출 여부 판단이 맞는지
- JSON Parse Success: 생성된 응답이 유효한 JSON인지

사용법:
    python evaluate_base.py
    python evaluate_base.py --num_samples 1000
    python evaluate_base.py --base_model kakaocorp/kanana-nano-2.1b-instruct
"""

# ============================================================
# ⚠️ Unsloth는 반드시 다른 패키지보다 먼저 import해야 합니다!
# ============================================================
from unsloth import FastLanguageModel

import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# HuggingFace Hub 데이터셋 ID
# ============================================================
HF_DATASET = "NotoriousH2/instructkr-sft"


# ============================================================
# Tool Call 추출 함수 (모델 응답용)
# ============================================================
def extract_tool_call_from_response(response: str) -> Optional[dict]:
    """
    모델 응답에서 Tool Call 추출
    
    Returns:
        {"name": "tool_name", "arguments": {...}} 또는 None (Tool Call 없음)
    """
    if not response:
        return None
    
    # 형식 1: <function=name>{"param": "value"}</function>
    pattern1 = r'<function=([^>]+)>(.+?)</function>'
    match1 = re.search(pattern1, response, re.DOTALL)
    if match1:
        tool_name = match1.group(1).strip()
        try:
            arguments = json.loads(match1.group(2).strip())
            return {"name": tool_name, "arguments": arguments}
        except json.JSONDecodeError:
            return {"name": tool_name, "arguments": None, "parse_error": True}
    
    # 형식 2: JSON 형식
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict) and "name" in parsed:
            return {
                "name": parsed.get("name", ""),
                "arguments": parsed.get("arguments", {})
            }
    except json.JSONDecodeError:
        pass
    
    return None


def extract_tool_call_from_data(msg: dict) -> Optional[dict]:
    """
    데이터의 메시지에서 Tool Call 추출 (tool_calls 필드 사용)
    """
    tool_calls = msg.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return None
    
    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None
    
    func_info = first_call.get("function", {})
    if not func_info:
        return None
    
    name = func_info.get("name", "")
    args_str = func_info.get("arguments", "{}")
    
    try:
        if isinstance(args_str, str):
            arguments = json.loads(args_str)
        else:
            arguments = args_str
    except json.JSONDecodeError:
        arguments = {}
    
    return {"name": name, "arguments": arguments}


# ============================================================
# 비교 함수
# ============================================================
def compare_tool_calls(pred: Optional[dict], gold: Optional[dict], gold_has_tool_call: bool) -> dict:
    """예측과 정답 Tool Call 비교 (When-to-Call 포함)"""
    pred_has_tool_call = pred is not None
    
    result = {
        "gold_has_tool_call": gold_has_tool_call,
        "pred_has_tool_call": pred_has_tool_call,
        "when_to_call_correct": (pred_has_tool_call == gold_has_tool_call),
        "tool_selection_correct": False,
        "parameter_exact_match": False,
        "json_parse_success": False,
    }
    
    # Tool Call이 있는 경우에만 세부 비교
    if gold_has_tool_call and gold:
        if pred:
            result["json_parse_success"] = not pred.get("parse_error", False)
            
            pred_name = pred.get("name", "").lower().strip()
            gold_name = gold.get("name", "").lower().strip()
            result["tool_selection_correct"] = (pred_name == gold_name)
            
            if result["tool_selection_correct"]:
                pred_args = pred.get("arguments", {})
                gold_args = gold.get("arguments", {})
                if isinstance(pred_args, dict) and isinstance(gold_args, dict):
                    result["parameter_exact_match"] = (pred_args == gold_args)
    
    return result


# ============================================================
# 메트릭 계산
# ============================================================
def calculate_metrics(results: list[dict]) -> dict:
    """전체 평가 결과에서 메트릭 계산"""
    total = len(results)
    if total == 0:
        return {}
    
    # Tool Call이 있는 샘플만 필터링
    tool_call_samples = [r for r in results if r["gold_has_tool_call"]]
    no_call_samples = [r for r in results if not r["gold_has_tool_call"]]
    
    metrics = {
        "total_samples": total,
        "tool_call_samples": len(tool_call_samples),
        "no_call_samples": len(no_call_samples),
        
        # When-to-Call (전체)
        "when_to_call_accuracy": sum(r["when_to_call_correct"] for r in results) / total,
        
        # Tool Call 샘플에서의 정확도
        "tool_selection_accuracy": (
            sum(r["tool_selection_correct"] for r in tool_call_samples) / len(tool_call_samples)
            if tool_call_samples else 0
        ),
        "parameter_exact_match": (
            sum(r["parameter_exact_match"] for r in tool_call_samples) / len(tool_call_samples)
            if tool_call_samples else 0
        ),
        "json_parse_success_rate": (
            sum(r["json_parse_success"] for r in tool_call_samples) / len(tool_call_samples)
            if tool_call_samples else 0
        ),
        
        # No-Call 샘플에서의 정확도 (False Positive Rate)
        "no_call_accuracy": (
            sum(r["when_to_call_correct"] for r in no_call_samples) / len(no_call_samples)
            if no_call_samples else 0
        ),
    }
    
    return metrics


# ============================================================
# 평가 샘플 추출 (모든 assistant 턴)
# ============================================================
def extract_all_eval_samples(messages: list, tools, source: str = "") -> list[dict]:
    """
    대화에서 모든 assistant 턴을 평가 샘플로 추출 (멀티턴 Tool Calling 지원)
    """
    if not messages:
        return []
    
    eval_samples = []
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        
        # 이 assistant 턴 이전까지가 context
        context = messages[:i]
        
        # user 메시지가 하나도 없으면 스킵
        if not any(m.get("role") == "user" for m in context if isinstance(m, dict)):
            continue
        
        # Tool Call 추출
        gold_tool_call = extract_tool_call_from_data(msg)
        has_tool_call = gold_tool_call is not None
        
        eval_samples.append({
            "context": context,
            "gold_response": msg.get("content", ""),
            "gold_tool_call": gold_tool_call,
            "has_tool_call": has_tool_call,
            "tools": tools,
            "source": source,
            "turn_index": i
        })
    
    return eval_samples


# ============================================================
# HuggingFace Hub에서 데이터 로드 (균형 샘플링)
# ============================================================
def load_test_data(dataset_id: str, num_samples: int, seed: int = 42):
    """HuggingFace Hub에서 테스트 데이터 로드 (모든 턴 추출 + 균형 샘플링)"""
    tool_call_samples = []
    no_call_samples = []
    
    print(f"\n📥 HuggingFace Hub에서 테스트 데이터 로드 중: {dataset_id}")
    
    ds = load_dataset(dataset_id, split="train")
    print(f"✅ {len(ds)}개 대화 로드")
    
    tc_count = 0
    nc_count = 0
    
    for sample in ds:
        # JSON 문자열 파싱 (통합 데이터셋 형식)
        messages = parse_messages(sample.get("messages", []))
        tools = parse_tools(sample.get("tools", []))
        
        # 모든 assistant 턴을 평가 샘플로 추출
        eval_samples = extract_all_eval_samples(messages, tools, dataset_id)
        for eval_sample in eval_samples:
            if eval_sample["has_tool_call"]:
                tool_call_samples.append(eval_sample)
                tc_count += 1
            else:
                no_call_samples.append(eval_sample)
                nc_count += 1
    
    print(f"   → Tool Call 턴: {tc_count}개, No-Call 턴: {nc_count}개")
    print(f"\n📊 전체 평가 가능 턴: Tool Call {len(tool_call_samples)}개, No-Call {len(no_call_samples)}개")
    
    # 균형 샘플링
    random.seed(seed)
    
    half_samples = num_samples // 2
    
    random.shuffle(tool_call_samples)
    selected_tc = tool_call_samples[:min(half_samples, len(tool_call_samples))]
    
    random.shuffle(no_call_samples)
    selected_nc = no_call_samples[:min(half_samples, len(no_call_samples))]
    
    all_samples = selected_tc + selected_nc
    random.shuffle(all_samples)
    
    print(f"📊 샘플링 결과: Tool Call {len(selected_tc)}개 + No-Call {len(selected_nc)}개 = {len(all_samples)}개")
    
    return all_samples


# ============================================================
# 프롬프트 생성
# ============================================================
def parse_messages(messages) -> list:
    """JSON 문자열로 저장된 messages를 리스트로 파싱"""
    if not messages:
        return []
    if isinstance(messages, str):
        try:
            return json.loads(messages)
        except json.JSONDecodeError:
            return []
    if isinstance(messages, list):
        return messages
    return []


def parse_tools(tools) -> list:
    """다양한 형식의 tools를 표준 리스트로 변환"""
    if not tools:
        return []
    
    if isinstance(tools, str):
        try:
            parsed = json.loads(tools)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return []
    
    return tools if isinstance(tools, list) else []


def format_tools_for_prompt(tools: list) -> str:
    """tools를 시스템 프롬프트용 문자열로 변환"""
    tools_descriptions = []
    
    for tool in tools:
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except json.JSONDecodeError:
                continue
        
        if not isinstance(tool, dict):
            continue
        
        name = tool.get("name", "")
        description = tool.get("description", "")
        params = tool.get("parameters", {})
        
        tool_json = json.dumps({
            "name": name,
            "description": description,
            "parameters": params,
            "required": tool.get("required", [])
        }, ensure_ascii=False)
        
        tools_descriptions.append(f"Use the function '{name}' to '{description}'\n{tool_json}")
    
    tools_text = "\n\n".join(tools_descriptions)
    
    return f"""You have access to the following functions:

{tools_text}

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
"""


def create_eval_prompt(sample: dict, tokenizer) -> str:
    """평가용 프롬프트 생성"""
    tools = parse_tools(sample.get("tools", []))
    context = sample.get("context", [])
    
    system_content = format_tools_for_prompt(tools)
    messages = [{"role": "system", "content": system_content}]
    
    for msg in context:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "tool":
            role = "ipython"
        
        # content가 없는 경우 tool_calls에서 생성
        if not content and role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "{}")
                content = f"<function={name}>{args}</function>"
        
        if content:
            messages.append({"role": role, "content": content})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


# ============================================================
# 평가 실행 (배치 추론)
# ============================================================
def run_evaluation(
    model,
    tokenizer,
    test_samples: list,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    progress_bar: bool = True
) -> list[dict]:
    """배치 평가 실행"""
    results = []
    
    # 패딩 설정
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 배치 단위로 처리
    num_batches = (len(test_samples) + batch_size - 1) // batch_size
    iterator = tqdm(range(num_batches), desc="평가 중") if progress_bar else range(num_batches)
    
    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_samples))
        batch_samples = test_samples[start_idx:end_idx]
        
        # 프롬프트 생성
        prompts = [create_eval_prompt(s, tokenizer) for s in batch_samples]
        
        # 배치 토크나이징
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=8192
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 결과 처리
        # 배치 내 모든 샘플은 동일한 길이로 패딩됨
        input_tensor_len = inputs["input_ids"].shape[1]
        
        for i, sample in enumerate(batch_samples):
            # 입력 텐서 길이 이후가 생성된 토큰
            generated = outputs[i][input_tensor_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            
            # 전체 출력 시퀀스 (입력 + 생성, 특수 토큰 포함)
            full_output = tokenizer.decode(outputs[i], skip_special_tokens=False)
            
            pred_tool_call = extract_tool_call_from_response(response)
            gold_tool_call = sample.get("gold_tool_call")
            has_tool_call = sample.get("has_tool_call", False)
            
            comparison = compare_tool_calls(pred_tool_call, gold_tool_call, has_tool_call)
            
            result = {
                "source": sample.get("source", ""),
                "turn_index": sample.get("turn_index", 0),
                "has_tool_call": has_tool_call,
                "input_prompt": prompts[i],  # 전체 입력 프롬프트
                "gold_response": sample.get("gold_response", ""),  # 정답 응답
                "generated_output": response,  # 새로 생성된 토큰만
                "full_output_sequence": full_output,  # 입력 + 생성 전체 (디버깅용)
                "pred_tool_call": pred_tool_call,
                "gold_tool_call": gold_tool_call,
                **comparison
            }
            results.append(result)
    
    return results


# ============================================================
# 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Tool Calling 베이스 모델 평가")
    
    parser.add_argument("--base_model", type=str, default="kakaocorp/kanana-nano-2.1b-instruct",
                        help="베이스 모델 ID")
    parser.add_argument("--num_samples", type=int, default=100, help="평가할 샘플 수")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기 (unsloth 호환성 문제로 기본값 1)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="생성 최대 토큰 수")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="최대 시퀀스 길이")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="결과 저장 디렉토리")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 Tool Calling 베이스 모델 평가")
    print("=" * 60)
    print(f"베이스 모델: {args.base_model}")
    print(f"평가 샘플 수: {args.num_samples}")
    print(f"배치 크기: {args.batch_size}")
    print(f"최대 생성 토큰: {args.max_new_tokens}")
    print("=" * 60)
    
    # 베이스 모델 로드
    print("\n🚀 베이스 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Unsloth 최적화를 위해 dummy LoRA 어댑터 추가
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    FastLanguageModel.for_inference(model)
    print("모델 로드 완료")
    
    # 테스트 데이터 로드
    test_samples = load_test_data(HF_DATASET, args.num_samples, args.seed)
    
    # 평가 실행 (배치 추론)
    print("\n🔬 평가 시작...")
    results = run_evaluation(
        model, tokenizer, test_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    
    # 메트릭 계산
    metrics = calculate_metrics(results)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 베이스 모델 평가 결과")
    print("=" * 60)
    print(f"총 평가 샘플: {metrics.get('total_samples', 0)}")
    print(f"  - Tool Call 샘플: {metrics.get('tool_call_samples', 0)}")
    print(f"  - No-Call 샘플: {metrics.get('no_call_samples', 0)}")
    print("-" * 60)
    print(f"When-to-Call Accuracy: {metrics.get('when_to_call_accuracy', 0):.2%}")
    print(f"  - Tool Call 정확도: {metrics.get('tool_selection_accuracy', 0):.2%}")
    print(f"  - No-Call 정확도: {metrics.get('no_call_accuracy', 0):.2%}")
    print("-" * 60)
    print(f"Tool Selection Accuracy: {metrics.get('tool_selection_accuracy', 0):.2%}")
    print(f"Parameter Exact Match: {metrics.get('parameter_exact_match', 0):.2%}")
    print(f"JSON Parse Success Rate: {metrics.get('json_parse_success_rate', 0):.2%}")
    print("=" * 60)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(args.output_dir, f"base_evaluation_results_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 상세 결과 저장: {results_csv}")
    
    summary = {
        "model_type": "base",
        "base_model": args.base_model,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": timestamp,
        "metrics": metrics
    }
    summary_json = os.path.join(args.output_dir, f"base_evaluation_summary_{timestamp}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 요약 저장: {summary_json}")
    
    print("\n✅ 베이스 모델 평가 완료!")


if __name__ == "__main__":
    main()
