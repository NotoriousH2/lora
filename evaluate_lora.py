#!/usr/bin/env python3
"""
Tool Calling LoRA 모델 평가 스크립트

학습된 LoRA 모델의 Tool Calling 성능을 평가합니다.

평가 지표:
- Tool Selection Accuracy: 올바른 Tool을 선택했는지
- Parameter Exact Match: 파라미터가 정확히 일치하는지
- When-to-Call Accuracy: Tool 호출 여부 판단이 맞는지
- JSON Parse Success: 생성된 응답이 유효한 JSON인지

사용법:
    python evaluate_lora.py --model_path experiments/final_model
    python evaluate_lora.py --model_path experiments/final_model --num_samples 1000
    python evaluate_lora.py --model_path experiments/final_model --output_dir ./eval_results
"""

# ============================================================
# ⚠️ Unsloth는 반드시 다른 패키지보다 먼저 import해야 합니다!
# ============================================================
from unsloth import FastLanguageModel

import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


# ============================================================
# HuggingFace Hub 데이터셋 ID
# ============================================================
HF_DATASETS = [
    "NotoriousH2/instructkr-toolflow",
    "NotoriousH2/instructkr-when2call",
    "NotoriousH2/instructkr-apigen",
]


# ============================================================
# Tool Call 추출 함수
# ============================================================
def extract_tool_call(response: str) -> Optional[dict]:
    """
    모델 응답에서 Tool Call 추출
    
    지원하는 형식:
    1. <function=name>{"param": "value"}</function>
    2. {"name": "...", "arguments": {...}}
    
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
    
    # 형식 2: {"name": "...", "arguments": {...}}
    try:
        # JSON 객체 찾기
        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}'
        match2 = re.search(json_pattern, response, re.DOTALL)
        if match2:
            parsed = json.loads(match2.group())
            if "name" in parsed:
                return {
                    "name": parsed.get("name", ""),
                    "arguments": parsed.get("arguments", {})
                }
    except json.JSONDecodeError:
        pass
    
    # 간단한 JSON 형식: {"name": "...", "arguments": "..."}
    try:
        # 전체 응답이 JSON인지 확인
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict) and "name" in parsed:
            return {
                "name": parsed.get("name", ""),
                "arguments": parsed.get("arguments", {})
            }
    except json.JSONDecodeError:
        pass
    
    return None


def extract_tool_call_from_gold(message: dict) -> Optional[dict]:
    """
    정답 메시지에서 Tool Call 추출
    
    assistant 메시지의 content가 Tool Call 형식인 경우 추출
    """
    content = message.get("content", "")
    return extract_tool_call(content)


# ============================================================
# 비교 함수
# ============================================================
def compare_tool_calls(pred: Optional[dict], gold: Optional[dict]) -> dict:
    """
    예측과 정답 Tool Call 비교
    
    Returns:
        {
            "tool_selection_correct": bool,
            "parameter_exact_match": bool,
            "when_to_call_correct": bool,
            "json_parse_success": bool,
        }
    """
    result = {
        "tool_selection_correct": False,
        "parameter_exact_match": False,
        "when_to_call_correct": False,
        "json_parse_success": False,
    }
    
    # When-to-Call: 둘 다 None이거나 둘 다 Tool Call이 있으면 정답
    pred_has_call = pred is not None
    gold_has_call = gold is not None
    result["when_to_call_correct"] = (pred_has_call == gold_has_call)
    
    # Tool Call이 없는 경우
    if not gold_has_call:
        if not pred_has_call:
            # 둘 다 Tool Call 없음 - 정답
            result["tool_selection_correct"] = True
            result["parameter_exact_match"] = True
            result["json_parse_success"] = True
        return result
    
    if not pred_has_call:
        # 정답은 있는데 예측이 없음
        return result
    
    # JSON 파싱 성공 여부
    result["json_parse_success"] = not pred.get("parse_error", False)
    
    # Tool 이름 비교
    pred_name = pred.get("name", "").lower().strip()
    gold_name = gold.get("name", "").lower().strip()
    result["tool_selection_correct"] = (pred_name == gold_name)
    
    # 파라미터 비교 (Tool 이름이 맞을 때만)
    if result["tool_selection_correct"]:
        pred_args = pred.get("arguments", {})
        gold_args = gold.get("arguments", {})
        
        # 둘 다 dict인 경우에만 비교
        if isinstance(pred_args, dict) and isinstance(gold_args, dict):
            # 키와 값이 모두 일치하는지 확인
            result["parameter_exact_match"] = (pred_args == gold_args)
        elif pred_args == gold_args:
            result["parameter_exact_match"] = True
    
    return result


# ============================================================
# 메트릭 계산
# ============================================================
def calculate_metrics(results: list[dict]) -> dict:
    """전체 평가 결과에서 메트릭 계산"""
    total = len(results)
    if total == 0:
        return {}
    
    metrics = {
        "total_samples": total,
        "tool_selection_accuracy": sum(r["tool_selection_correct"] for r in results) / total,
        "parameter_exact_match": sum(r["parameter_exact_match"] for r in results) / total,
        "when_to_call_accuracy": sum(r["when_to_call_correct"] for r in results) / total,
        "json_parse_success_rate": sum(r["json_parse_success"] for r in results) / total,
    }
    
    # Tool Call이 있는 샘플만 따로 계산
    has_tool_call = [r for r in results if r.get("gold_has_tool_call", False)]
    if has_tool_call:
        metrics["tool_call_samples"] = len(has_tool_call)
        metrics["tool_selection_accuracy_on_calls"] = sum(
            r["tool_selection_correct"] for r in has_tool_call
        ) / len(has_tool_call)
        metrics["parameter_exact_match_on_calls"] = sum(
            r["parameter_exact_match"] for r in has_tool_call
        ) / len(has_tool_call)
    
    # Tool Call이 없는 샘플
    no_tool_call = [r for r in results if not r.get("gold_has_tool_call", False)]
    if no_tool_call:
        metrics["no_tool_call_samples"] = len(no_tool_call)
        metrics["no_call_accuracy"] = sum(
            r["when_to_call_correct"] for r in no_tool_call
        ) / len(no_tool_call)
    
    return metrics


# ============================================================
# 데이터 로드
# ============================================================
def load_test_data(dataset_ids: list[str], num_samples: int, seed: int = 42):
    """HuggingFace Hub에서 테스트 데이터 로드"""
    all_datasets = []
    
    print("\n📥 HuggingFace Hub에서 테스트 데이터 로드 중...")
    
    for dataset_id in dataset_ids:
        try:
            ds = load_dataset(dataset_id, split="train")
            print(f"✅ {dataset_id}: {len(ds)}개 샘플")
            all_datasets.append(ds)
        except Exception as e:
            print(f"❌ {dataset_id} 로드 실패: {e}")
    
    if not all_datasets:
        raise ValueError("로드된 데이터셋이 없습니다!")
    
    # 병합
    if len(all_datasets) == 1:
        combined = all_datasets[0]
    else:
        # 스키마 통일을 위해 필요한 컬럼만 선택
        unified_datasets = []
        for ds in all_datasets:
            if "messages" in ds.column_names and "tools" in ds.column_names:
                unified_datasets.append(ds.select_columns(["messages", "tools"]))
        
        if unified_datasets:
            combined = concatenate_datasets(unified_datasets)
        else:
            combined = all_datasets[0]
    
    # 셔플 및 샘플링
    combined = combined.shuffle(seed=seed)
    
    if num_samples < len(combined):
        combined = combined.select(range(num_samples))
    
    print(f"\n📊 테스트 샘플 수: {len(combined)}개")
    
    return combined


# ============================================================
# 평가용 프롬프트 생성
# ============================================================
def parse_tools(tools) -> list:
    """다양한 형식의 tools를 표준 리스트로 변환"""
    if not tools:
        return []
    
    if isinstance(tools, str):
        try:
            parsed = json.loads(tools)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            return []
    
    if isinstance(tools, list):
        return tools
    
    return []


def create_eval_prompt(messages: list, tools, tokenizer) -> tuple[str, Optional[dict]]:
    """
    평가용 프롬프트 생성
    
    Returns:
        (prompt, gold_tool_call)
        - prompt: 모델에 입력할 프롬프트 (마지막 assistant 응답 제외)
        - gold_tool_call: 정답 Tool Call (있는 경우)
    """
    # 마지막 assistant 응답 찾기
    gold_response = None
    gold_tool_call = None
    
    # 메시지에서 마지막 assistant 응답 분리
    prompt_messages = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        
        role = msg.get("role", "")
        
        # 마지막 assistant 응답 저장
        if role == "assistant":
            gold_response = msg.get("content", "")
            gold_tool_call = extract_tool_call(gold_response)
            # 이전까지의 메시지만 프롬프트에 포함
            break
        
        prompt_messages.append(msg)
    
    # tools를 시스템 프롬프트로 변환
    parsed_tools = parse_tools(tools)
    
    if parsed_tools:
        tools_text = format_tools_for_prompt(parsed_tools)
        system_msg = {"role": "system", "content": tools_text}
        prompt_messages = [system_msg] + prompt_messages
    
    # 프롬프트 생성
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback
        prompt = "<|begin_of_text|>"
        for msg in prompt_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt, gold_tool_call


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
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line"""


# ============================================================
# 메인 평가 함수
# ============================================================
def run_evaluation(model, tokenizer, test_data, max_new_tokens: int = 512) -> list[dict]:
    """메인 평가 루프"""
    results = []
    
    # 추론 모드로 전환
    FastLanguageModel.for_inference(model)
    
    for idx, sample in enumerate(tqdm(test_data, desc="평가 진행")):
        messages = sample.get("messages", [])
        tools = sample.get("tools", [])
        
        # 프롬프트 생성
        try:
            prompt, gold_tool_call = create_eval_prompt(messages, tools, tokenizer)
        except Exception as e:
            print(f"⚠️ 샘플 {idx} 프롬프트 생성 실패: {e}")
            continue
        
        # 모델 추론
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # 응답 디코딩
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            print(f"⚠️ 샘플 {idx} 추론 실패: {e}")
            response = ""
        
        # Tool Call 추출
        pred_tool_call = extract_tool_call(response)
        
        # 비교
        comparison = compare_tool_calls(pred_tool_call, gold_tool_call)
        
        # 결과 저장
        result = {
            "sample_idx": idx,
            "gold_has_tool_call": gold_tool_call is not None,
            "pred_has_tool_call": pred_tool_call is not None,
            "gold_tool_name": gold_tool_call.get("name", "") if gold_tool_call else "",
            "pred_tool_name": pred_tool_call.get("name", "") if pred_tool_call else "",
            "response_preview": response[:200] if response else "",
            **comparison
        }
        results.append(result)
    
    return results


# ============================================================
# 결과 저장
# ============================================================
def save_results(results: list[dict], metrics: dict, output_dir: str):
    """평가 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 저장
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ 상세 결과 저장: {csv_path}")
    
    # JSON 요약 저장
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 요약 저장: {summary_path}")


def print_metrics(metrics: dict):
    """메트릭 출력"""
    print("\n" + "=" * 60)
    print("📊 평가 결과")
    print("=" * 60)
    print(f"총 샘플 수: {metrics.get('total_samples', 0)}")
    print()
    print("전체 정확도:")
    print(f"  - Tool Selection Accuracy: {metrics.get('tool_selection_accuracy', 0):.2%}")
    print(f"  - Parameter Exact Match:   {metrics.get('parameter_exact_match', 0):.2%}")
    print(f"  - When-to-Call Accuracy:   {metrics.get('when_to_call_accuracy', 0):.2%}")
    print(f"  - JSON Parse Success:      {metrics.get('json_parse_success_rate', 0):.2%}")
    
    if "tool_call_samples" in metrics:
        print()
        print(f"Tool Call이 필요한 샘플 ({metrics['tool_call_samples']}개):")
        print(f"  - Tool Selection: {metrics.get('tool_selection_accuracy_on_calls', 0):.2%}")
        print(f"  - Parameter Match: {metrics.get('parameter_exact_match_on_calls', 0):.2%}")
    
    if "no_tool_call_samples" in metrics:
        print()
        print(f"Tool Call이 불필요한 샘플 ({metrics['no_tool_call_samples']}개):")
        print(f"  - No-Call Accuracy: {metrics.get('no_call_accuracy', 0):.2%}")
    
    print("=" * 60)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tool Calling LoRA 모델 평가",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="평가할 모델 경로 (LoRA 어댑터 또는 전체 모델)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="평가할 샘플 수"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="생성할 최대 토큰 수"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096,
        help="최대 시퀀스 길이"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 Tool Calling LoRA 모델 평가")
    print("=" * 60)
    print(f"모델 경로: {args.model_path}")
    print(f"샘플 수: {args.num_samples}")
    print(f"출력 디렉토리: {args.output_dir}")
    print("=" * 60)
    
    # 모델 로드
    print("\n📦 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ 모델 로드 완료")
    
    # 테스트 데이터 로드
    test_data = load_test_data(HF_DATASETS, args.num_samples, args.seed)
    
    # 평가 실행
    print("\n🏃 평가 시작...")
    results = run_evaluation(model, tokenizer, test_data, args.max_new_tokens)
    
    # 메트릭 계산
    metrics = calculate_metrics(results)
    
    # 결과 출력
    print_metrics(metrics)
    
    # 결과 저장
    save_results(results, metrics, args.output_dir)
    
    print("\n🎉 평가 완료!")


if __name__ == "__main__":
    main()

