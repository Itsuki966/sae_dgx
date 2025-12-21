"""
🧠 Feedback迎合性分析スクリプト

このスクリプトは、Feedbackデータセットを使用してLLMの迎合性（Sycophancy）分析を実行します。

特徴:
- Gemma-2-9B-it対応: 大規模モデルでの詳細分析
- SAE内部状態抽出: Layer 9/20/31のSAE活性化を詳細に記録
- 最適な分析位置: プロンプトの最後のトークン（応答生成直前）の内部状態を取得
- テンプレート比較: 5種類のプロンプトテンプレートでの応答を比較

実行方法:
    python run_feedback_analysis.py
"""

import os
import sys
import warnings
import torch
import json
import time
import datetime
from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np

# メモリ最適化設定
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# メインコードのインポート
try:
    from feedback_analyzer import FeedbackAnalyzer
    from config import (
        FEEDBACK_GEMMA2_9B_IT_CONFIG,
        FEEDBACK_GEMMA2_9B_IT_LAYER20_CONFIG,
        FEEDBACK_GEMMA2_9B_IT_LAYER9_CONFIG,
        FeedbackConfig
    )
    print("✅ メインコードのインポート完了")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("💡 feedback_analyzer.py と config.py が同じディレクトリにあることを確認してください。")
    sys.exit(1)


def setup_experiment_config(
    base_config,
    save_all_tokens=False,
    max_new_tokens=512,
    temperature=0.7,
    verbose=True,
    show_prompts=True,
    show_responses=True,
    response_tokens_to_capture=8
):
    """
    実験パラメータを設定
    
    Args:
        base_config: ベース設定（FEEDBACK_GEMMA2_9B_IT_CONFIG等）
        save_all_tokens: 全プロンプトトークンを保存するか（デフォルト: False）
        max_new_tokens: 生成する最大トークン数
        temperature: 生成温度（0.0-1.0）
        verbose: 詳細ログを表示
        show_prompts: プロンプトを表示
        show_responses: 応答を表示
        response_tokens_to_capture: 応答の最初の何トークンを取得するか
    
    Returns:
        設定されたExperimentConfig
    """
    config = deepcopy(base_config)
    
    # 生成パラメータ
    config.generation.max_new_tokens = max_new_tokens
    config.generation.temperature = temperature
    
    # デバッグ設定
    config.debug.verbose = verbose
    config.debug.show_prompts = show_prompts
    config.debug.show_responses = show_responses
    
    # Feedback専用設定
    if not hasattr(config, 'feedback'):
        config.feedback = FeedbackConfig()
    config.feedback.save_all_tokens = save_all_tokens
    config.feedback.response_tokens_to_capture = response_tokens_to_capture
    
    return config


def print_config_summary(config, start_index=None, end_index=None):
    """設定サマリーを表示"""
    print("🎯 実験設定:")
    print(f"   📱 モデル: {config.model.name}")
    if start_index is not None and end_index is not None:
        print(f"   🎱 Start Question ID: {start_index}, End Question ID: {end_index}")
    print(f"   📊 テンプレート数: 5種類/問題")
    print(f"   💾 プロンプト分析位置: {'全プロンプトトークン' if config.feedback.save_all_tokens else 'プロンプト最終トークン（応答生成直前）'}")
    print(f"   💬 応答分析: 最初の{config.feedback.response_tokens_to_capture}トークン")
    print(f"   🎯 対象レイヤー: {config.feedback.target_layer}")
    print(f"   🌡️  温度: {config.generation.temperature}")
    print(f"   🔢 最大トークン: {config.generation.max_new_tokens}")
    print(f"   🔍 詳細ログ: {config.debug.verbose}")
    print(f"   🔍 SAE: {config.model.sae_release}/{config.model.sae_id}")


def check_dataset_path(config):
    """データセットの存在チェック"""
    ds_path = config.data.dataset_path
    if not os.path.exists(ds_path):
        print(f"\n⚠️ データセットが見つかりません: {ds_path}")
        # デフォルトパスを試行
        default_file = os.path.join('eval_dataset', 'feedback.jsonl')
        if os.path.exists(default_file):
            config.data.dataset_path = default_file
            print(f"   ✅ 自動補正: dataset_path を {default_file} に変更しました")
        else:
            print("   ❌ eval_dataset/feedback.jsonl が見つかりません。")
            sys.exit(1)


def run_analysis(config, start_index=None, end_index=None):
    """
    Feedback迎合性分析を実行
    
    Args:
        config: ExperimentConfig
        start_index: 開始インデックス（0-based）
        end_index: 終了インデックス（0-based）
    
    Returns:
        分析結果のリスト
    """
    print("🚀 Feedback迎合性分析を開始...")
    print("=" * 60)
    
    # 実験メタデータの準備
    experiment_metadata = {
        'experiment_start': datetime.datetime.now().isoformat(),
        'config_params': {
            'model_name': config.model.name,
            'save_all_tokens': config.feedback.save_all_tokens,
            'response_tokens_to_capture': config.feedback.response_tokens_to_capture,
            'target_layer': config.feedback.target_layer,
            'temperature': config.generation.temperature,
            'max_new_tokens': config.generation.max_new_tokens,
            'start_index': start_index,
            'end_index': end_index,
        }
    }
    
    print(f"📋 実験メタデータ:")
    print(f"   ⏰ 開始時刻: {experiment_metadata['experiment_start']}")
    print(f"   📊 設定: {experiment_metadata['config_params']}")
    
    # 分析器の初期化
    print(f"\n🔧 分析器を初期化中...")
    analyzer = FeedbackAnalyzer(config)
    
    # メモリ使用状況の表示
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 初期GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
        experiment_metadata['initial_memory_gb'] = round(memory_used, 2)
    
    # 完全分析の実行
    print(f"\n🔄 完全分析を実行中...")
    print(f"   📋 この処理には数分〜数十分かかる場合があります...")
    
    # 実行時間測定開始
    start_time = time.time()
    
    # メイン分析の実行
    analyzer.run_complete_analysis(start_index=start_index, end_index=end_index)
    
    # 実行時間測定終了
    end_time = time.time()
    execution_time = end_time - start_time
    experiment_metadata['execution_time_seconds'] = round(execution_time, 2)
    experiment_metadata['experiment_end'] = datetime.datetime.now().isoformat()
    
    print(f"\n" + "=" * 60)
    print("✅ 分析完了！")
    print(f"⏱️  実行時間: {execution_time:.1f}秒 ({execution_time/60:.1f}分)")
    
    # 結果の簡易表示
    if hasattr(analyzer, 'results') and analyzer.results:
        num_questions = len(analyzer.results)
        total_variations = sum(len(r.variations) for r in analyzer.results)
        
        # 実験メタデータに結果を追加
        experiment_metadata['results'] = {
            'num_questions': num_questions,
            'total_variations': total_variations,
        }
        
        print(f"\n📊 分析結果サマリー:")
        print(f"   📝 分析した問題数: {num_questions}")
        print(f"   📈 総バリエーション数: {total_variations}")
        print(f"   💾 トークン保存モード: {'全トークン' if config.feedback.save_all_tokens else '最後のトークンのみ'}")
        
        # 最初の結果の概要を表示
        if num_questions > 0:
            first_result = analyzer.results[0]
            print(f"\n📄 最初の質問の概要:")
            print(f"   Dataset: {first_result.dataset}")
            print(f"   テンプレート数: {len(first_result.variations)}")
            for var in first_result.variations[:3]:  # 最初の3つだけ表示
                print(f"   - {var.prompt_info.prompt_template_type or '(base)'}: {len(var.response_text)} 文字")
    
    # メモリ使用状況の最終確認
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n💾 最終GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
        experiment_metadata['final_memory_gb'] = round(memory_used, 2)
    
    # 実験ログの保存
    experiment_log_file = f"experiment_log_{experiment_metadata['experiment_start'][:19].replace(':', '-')}.json"
    with open(experiment_log_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 実験ログ保存: {experiment_log_file}")
    
    return analyzer.results


def analyze_results(results, config):
    """
    分析結果を詳細に確認
    
    Args:
        results: 分析結果のリスト
        config: ExperimentConfig
    """
    print("\n📈 分析結果の詳細を表示...")
    print("=" * 60)
    
    # 基本統計
    num_questions = len(results)
    total_variations = sum(len(r.variations) for r in results)
    
    print(f"📊 基本統計:")
    print(f"   📝 分析した問題数: {num_questions}")
    print(f"   📈 総バリエーション数: {total_variations}")
    print(f"   💾 平均バリエーション数/問題: {total_variations/num_questions:.1f}")
    
    # データセット別の統計
    dataset_counts = {}
    for result in results:
        dataset = result.dataset
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print(f"\n📚 データセット別の内訳:")
    for dataset, count in dataset_counts.items():
        print(f"   - {dataset}: {count} 問題")
    
    # テンプレートタイプ別の統計
    template_stats = {}
    for result in results:
        for variation in result.variations:
            template_type = variation.prompt_info.prompt_template_type or "(base)"
            if template_type not in template_stats:
                template_stats[template_type] = {
                    'count': 0,
                    'avg_response_length': 0,
                    'total_length': 0,
                    'avg_active_features': 0,
                    'total_features': 0
                }
            
            stats = template_stats[template_type]
            stats['count'] += 1
            stats['total_length'] += len(variation.response_text)
            stats['total_features'] += len(variation.top_k_features)
    
    # 平均を計算
    for template_type, stats in template_stats.items():
        stats['avg_response_length'] = stats['total_length'] / stats['count']
        stats['avg_active_features'] = stats['total_features'] / stats['count']
    
    print(f"\n📝 テンプレートタイプ別の統計:")
    for template_type, stats in template_stats.items():
        print(f"   {template_type}:")
        print(f"      サンプル数: {stats['count']}")
        print(f"      平均応答長: {stats['avg_response_length']:.0f} 文字")
        print(f"      平均活性化特徴数: {stats['avg_active_features']:.1f}")
    
    # 具体例の表示（最初の2問）
    print(f"\n📝 具体例（最初の2問）:")
    print("-" * 60)
    
    for i, result in enumerate(results[:2]):
        print(f"\n問題 {i+1} (ID: {result.question_id}):")
        print(f"  データセット: {result.dataset}")
        print(f"  ベーステキスト: {result.base_text[:100]}...")
        print(f"  テンプレート数: {len(result.variations)}")
        
        for j, variation in enumerate(result.variations):
            template_type = variation.prompt_info.prompt_template_type or "(base)"
            print(f"\n  バリエーション {j+1}: {template_type}")
            print(f"    応答: {variation.response_text[:150]}...")
            print(f"    応答長: {len(variation.response_text)} 文字")
            print(f"    Top-3 SAE特徴:")
            for feat_id, feat_val in variation.top_k_features[:3]:
                print(f"      Feature {feat_id}: {feat_val:.4f}")
            print(f"    生成時間: {variation.metadata.get('generation_time_ms', 0):.0f} ms")
    
    # DataFrameに変換して保存
    print(f"\n💾 結果をDataFrameに変換中...")
    
    rows = []
    for result in results:
        for variation in result.variations:
            row = {
                'question_id': result.question_id,
                'dataset': result.dataset,
                'template_type': variation.prompt_info.prompt_template_type or "(base)",
                'response_length': len(variation.response_text),
                'num_active_features': len(variation.top_k_features),
                'top_feature_id': variation.top_k_features[0][0] if variation.top_k_features else None,
                'top_feature_value': variation.top_k_features[0][1] if variation.top_k_features else None,
                'generation_time_ms': variation.metadata.get('generation_time_ms', 0)
            }
            rows.append(row)
    
    df_results = pd.DataFrame(rows)
    
    print(f"✅ DataFrame作成完了 ({len(df_results)} 行)")
    print(f"\nDataFrame サンプル:")
    print(df_results.head(10))
    
    # CSVに保存
    csv_path = f"results/feedback/feedback_analysis_summary.csv"
    os.makedirs("results/feedback", exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 結果を保存しました: {csv_path}")
    
    print(f"\n" + "=" * 60)
    print("✅ 結果確認完了！")


def main():
    """メイン関数"""
    print("🧠 Feedback迎合性分析スクリプト")
    print("=" * 60)
    
    # GPU確認
    if torch.cuda.is_available():
        try:
            print(f"GPU検出: {torch.cuda.get_device_name(0)}")
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {memory_total:.1f}GB")
        except Exception:
            print("GPU検出: 利用可能 (デバイス名の取得に失敗)")
    else:
        print("❌ GPU利用不可 (CPUモード)")
    
    # === 実験パラメータの設定 ===
    # ここを変更して実験条件をカスタマイズしてください
    
    # ベース設定を選択（以下から1つを選択）
    # base_config = FEEDBACK_GEMMA2_9B_IT_CONFIG  # Layer 31
    base_config = FEEDBACK_GEMMA2_9B_IT_LAYER20_CONFIG  # Layer 20
    # base_config = FEEDBACK_GEMMA2_9B_IT_LAYER9_CONFIG  # Layer 9
    
    # 分析範囲の設定
    start = 0  # 開始インデックス（0-based）
    end = 10    # 終了インデックス（0-based）
    
    # 実験設定のカスタマイズ
    config = setup_experiment_config(
        base_config=base_config,
        save_all_tokens=False,  # プロンプトの最後のトークンのみ（推奨）
        max_new_tokens=512,     # 生成する最大トークン数
        temperature=0.7,        # 生成温度
        verbose=True,           # 詳細ログを表示
        show_prompts=True,      # プロンプトを表示
        show_responses=True,    # 応答を表示
        response_tokens_to_capture=8  # 応答の最初の8トークンを取得
    )
    
    # 設定サマリーを表示
    print_config_summary(config, start_index=start, end_index=end)
    
    # データセットの存在チェック
    check_dataset_path(config)
    
    print(f"\n✅ 実験パラメータ設定完了")
    
    # === 分析の実行 ===
    try:
        results = run_analysis(config, start_index=start, end_index=end)
        
        # === 結果の分析 ===
        if results:
            analyze_results(results, config)
        
        print(f"\n🎉 すべての処理が完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー時のメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
