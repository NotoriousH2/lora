#!/usr/bin/env python3
"""
여러 실험의 로스 커브를 비교 시각화하는 스크립트

사용법:
    # 모든 실험 CSV 비교
    python visualize_loss.py experiments/*/training_log.csv
    
    # 출력 파일 지정
    python visualize_loss.py --output comparison.png experiments/*/training_log.csv
    
    # 요약 테이블도 저장
    python visualize_loss.py -o comparison.png -s summary.csv experiments/*/training_log.csv
    
    # X축을 epoch으로 표시
    python visualize_loss.py --by_epoch experiments/*/training_log.csv
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 한글 폰트 설정 (선택적)
try:
    import platform
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def load_csv_with_metadata(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """CSV 파일과 메타데이터 로드"""
    metadata = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line.startswith('# experiment_info:'):
            json_str = first_line.replace('# experiment_info:', '').strip()
            try:
                metadata = json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    df = pd.read_csv(csv_path, comment='#')
    return df, metadata


def extract_rank_from_path(csv_path: str) -> int:
    """파일 경로에서 rank 추출"""
    match = re.search(r'_r(\d+)_', csv_path)
    if match:
        return int(match.group(1))
    return 0


def extract_model_from_path(csv_path: str) -> str:
    """파일 경로에서 모델명 추출"""
    path = Path(csv_path)
    parent_name = path.parent.name
    # 모델명 추출 시도 (첫 번째 언더스코어로 구분된 부분들)
    parts = parent_name.split('_')
    # r숫자 패턴이 나오기 전까지가 모델명
    model_parts = []
    for part in parts:
        if re.match(r'^r\d+$', part):
            break
        model_parts.append(part)
    return '_'.join(model_parts) if model_parts else parent_name


def plot_loss_curves(csv_files: list[str], output_path: str = None, 
                     by_epoch: bool = False, title: str = None):
    """여러 실험의 로스 커브 시각화"""
    
    if not csv_files:
        print("❌ CSV 파일이 없습니다.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # rank 순서로 정렬
    csv_files_sorted = sorted(csv_files, key=extract_rank_from_path)
    
    # 색상 팔레트
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(csv_files_sorted), 10)))
    
    model_name = ""
    
    for idx, csv_path in enumerate(csv_files_sorted):
        try:
            df, metadata = load_csv_with_metadata(csv_path)
        except Exception as e:
            print(f"⚠️ {csv_path} 로드 실패: {e}")
            continue
        
        # 레이블 생성
        if metadata:
            label = f"r{metadata.get('rank', '?')} (α={metadata.get('alpha', '?')})"
            if not model_name:
                model_name = metadata.get('model', '').split('/')[-1]
        else:
            rank = extract_rank_from_path(csv_path)
            label = f"r{rank}" if rank else Path(csv_path).stem
            if not model_name:
                model_name = extract_model_from_path(csv_path)
        
        x_col = 'epoch' if by_epoch else 'step'
        
        # Train Loss
        train_df = df[df['train_loss'].notna() & (df['train_loss'] != '')]
        if not train_df.empty:
            try:
                axes[0].plot(
                    train_df[x_col].astype(float), 
                    train_df['train_loss'].astype(float),
                    label=label, color=colors[idx % len(colors)], alpha=0.8, linewidth=1.5
                )
            except Exception as e:
                print(f"⚠️ Train loss 플롯 실패 ({csv_path}): {e}")
        
        # Eval Loss
        eval_df = df[df['eval_loss'].notna() & (df['eval_loss'] != '')]
        if not eval_df.empty:
            try:
                axes[1].plot(
                    eval_df[x_col].astype(float), 
                    eval_df['eval_loss'].astype(float),
                    label=label, color=colors[idx % len(colors)], 
                    marker='o', markersize=3, linewidth=1.5
                )
            except Exception as e:
                print(f"⚠️ Eval loss 플롯 실패 ({csv_path}): {e}")
    
    # 스타일링
    x_label = 'Epoch' if by_epoch else 'Step'
    
    axes[0].set_xlabel(x_label, fontsize=11)
    axes[0].set_ylabel('Train Loss', fontsize=11)
    axes[0].set_title('Training Loss by LoRA Rank', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel(x_label, fontsize=11)
    axes[1].set_ylabel('Eval Loss', fontsize=11)
    axes[1].set_title('Evaluation Loss by LoRA Rank', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 전체 타이틀
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    elif model_name:
        fig.suptitle(f'LoRA Rank Comparison - {model_name}', fontsize=14, fontweight='bold')
    else:
        fig.suptitle('LoRA Rank Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 그래프 저장: {output_path}")
    
    plt.show()


def plot_final_loss_comparison(csv_files: list[str], output_path: str = None):
    """최종 로스 값을 막대 그래프로 비교"""
    
    results = []
    
    for csv_path in csv_files:
        try:
            df, metadata = load_csv_with_metadata(csv_path)
            
            train_losses = df[df['train_loss'].notna() & (df['train_loss'] != '')]['train_loss'].astype(float)
            eval_losses = df[df['eval_loss'].notna() & (df['eval_loss'] != '')]['eval_loss'].astype(float)
            
            rank = metadata.get('rank', extract_rank_from_path(csv_path))
            
            results.append({
                'rank': rank,
                'final_train_loss': train_losses.iloc[-1] if len(train_losses) > 0 else None,
                'min_eval_loss': eval_losses.min() if len(eval_losses) > 0 else None,
            })
        except Exception as e:
            print(f"⚠️ {csv_path} 처리 실패: {e}")
    
    if not results:
        print("❌ 처리할 데이터가 없습니다.")
        return
    
    df_results = pd.DataFrame(results).sort_values('rank')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = range(len(df_results))
    width = 0.6
    
    # Final Train Loss
    axes[0].bar(x, df_results['final_train_loss'], width, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('LoRA Rank', fontsize=11)
    axes[0].set_ylabel('Final Train Loss', fontsize=11)
    axes[0].set_title('Final Training Loss by Rank', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"r{r}" for r in df_results['rank']])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Min Eval Loss
    axes[1].bar(x, df_results['min_eval_loss'], width, color='coral', alpha=0.8)
    axes[1].set_xlabel('LoRA Rank', fontsize=11)
    axes[1].set_ylabel('Min Eval Loss', fontsize=11)
    axes[1].set_title('Minimum Evaluation Loss by Rank', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"r{r}" for r in df_results['rank']])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        bar_output = output_path.replace('.png', '_bar.png').replace('.jpg', '_bar.jpg')
        plt.savefig(bar_output, dpi=150, bbox_inches='tight')
        print(f"✅ 막대 그래프 저장: {bar_output}")
    
    plt.show()


def create_summary_table(csv_files: list[str], output_path: str = None) -> pd.DataFrame:
    """실험 결과 요약 테이블 생성"""
    
    results = []
    
    for csv_path in csv_files:
        try:
            df, metadata = load_csv_with_metadata(csv_path)
            
            # 최종 로스 값 추출
            train_losses = df[df['train_loss'].notna() & (df['train_loss'] != '')]['train_loss'].astype(float)
            eval_losses = df[df['eval_loss'].notna() & (df['eval_loss'] != '')]['eval_loss'].astype(float)
            
            result = {
                'model': metadata.get('model', '').split('/')[-1] if metadata else extract_model_from_path(csv_path),
                'rank': metadata.get('rank', extract_rank_from_path(csv_path)),
                'alpha': metadata.get('alpha', ''),
                'lr': metadata.get('learning_rate', ''),
                'epochs': metadata.get('epochs', ''),
                'final_train_loss': round(train_losses.iloc[-1], 4) if len(train_losses) > 0 else None,
                'min_train_loss': round(train_losses.min(), 4) if len(train_losses) > 0 else None,
                'final_eval_loss': round(eval_losses.iloc[-1], 4) if len(eval_losses) > 0 else None,
                'min_eval_loss': round(eval_losses.min(), 4) if len(eval_losses) > 0 else None,
                'csv_path': csv_path,
            }
            results.append(result)
        except Exception as e:
            print(f"⚠️ {csv_path} 처리 실패: {e}")
    
    if not results:
        print("❌ 처리할 데이터가 없습니다.")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('rank')
    
    # 출력용 컬럼 선택 (csv_path 제외)
    display_cols = ['model', 'rank', 'alpha', 'lr', 'epochs', 
                    'final_train_loss', 'min_train_loss', 'final_eval_loss', 'min_eval_loss']
    
    print("\n" + "=" * 100)
    print("📊 실험 결과 요약")
    print("=" * 100)
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 100)
    
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\n✅ 요약 테이블 저장: {output_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 실험 로스 커브 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python visualize_loss.py experiments/*/training_log.csv
    python visualize_loss.py --output comparison.png experiments/*/training_log.csv
    python visualize_loss.py -o comparison.png -s summary.csv experiments/*/training_log.csv
    python visualize_loss.py --by_epoch experiments/*/training_log.csv
        """
    )
    parser.add_argument("csv_files", nargs="+", help="CSV 로그 파일들")
    parser.add_argument("--output", "-o", type=str, default=None, help="출력 이미지 경로")
    parser.add_argument("--by_epoch", action="store_true", help="X축을 epoch으로 표시")
    parser.add_argument("--summary", "-s", type=str, default=None, help="요약 CSV 저장 경로")
    parser.add_argument("--title", "-t", type=str, default=None, help="그래프 타이틀")
    parser.add_argument("--bar", action="store_true", help="최종 로스 막대 그래프도 표시")
    
    args = parser.parse_args()
    
    print(f"\n📁 {len(args.csv_files)}개 CSV 파일 로드 중...")
    for f in args.csv_files:
        print(f"   - {f}")
    
    # 요약 테이블 생성
    create_summary_table(args.csv_files, args.summary)
    
    # 로스 커브 시각화
    plot_loss_curves(args.csv_files, args.output, by_epoch=args.by_epoch, title=args.title)
    
    # 막대 그래프 (선택적)
    if args.bar:
        plot_final_loss_comparison(args.csv_files, args.output)


if __name__ == "__main__":
    main()

