"""
介入実験実行スクリプト

intervention_experiment.ipynb と同じ処理を行うコマンドラインスクリプト。
特定されたSAE特徴量に対してGeometric Subtractionによる介入を行い、
Baseline vs Intervention の比較実験を実行します。

Usage:
    # CSVから特徴量IDを読み込んで実験を実行
    python run_intervention_experiment.py --features-csv results/intervention/candidates/intervention_candidates_20251211_193847.csv
    
    # 特定の範囲で実験を実行
    python run_intervention_experiment.py --features-csv results/intervention/candidates/intervention_candidates_20251211_193847.csv --start-index 100 --end-index 200
    
    # サンプルサイズを指定
    python run_intervention_experiment.py --features-csv results/intervention/candidates/intervention_candidates_20251211_193847.csv --sample-size 10
    
    # 特徴量IDを直接指定
    python run_intervention_experiment.py --features 123,456,789 --sample-size 5
"""

import argparse
import json
import sys
import torch
import pandas as pd
from pathlib import Path
from typing import List, Optional

from intervention_runner import InterventionRunner
from config import INTERVENTION_GEMMA2_9B_IT_CONFIG


def load_feature_ids_from_csv(csv_path: str) -> List[int]:
    """
    CSVファイルから特徴量IDリストを読み込む
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        特徴量IDのリスト
    """
    df = pd.read_csv(csv_path)
    
    # 'feature_index' または 'feature_id' カラムから読み込み
    if 'feature_index' in df.columns:
        feature_ids = list(df['feature_index'])
    elif 'feature_id' in df.columns:
        feature_ids = list(df['feature_id'])
    else:
        raise ValueError(f"CSV file must contain 'feature_index' or 'feature_id' column. Found columns: {df.columns.tolist()}")
    
    return feature_ids


def load_feature_ids_from_json(json_path: str) -> List[int]:
    """
    JSONファイルから特徴量IDリストを読み込む
    
    Args:
        json_path: JSONファイルのパス
        
    Returns:
        特徴量IDのリスト
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 様々な形式に対応
    if isinstance(data, list):
        return data
    elif 'top_k_features' in data:
        return data['top_k_features']
    elif 'feature_ids' in data:
        return data['feature_ids']
    elif 'intervention_features' in data:
        return data['intervention_features']
    else:
        raise ValueError(f"JSON file must contain feature IDs. Found keys: {data.keys()}")


def print_gpu_info():
    """GPU情報を表示"""
    if torch.cuda.is_available():
        print(f"🎮 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ No GPU available. Using CPU (this will be slow)")


def print_configuration(config, feature_ids: List[int]):
    """実験設定を表示"""
    print("\n" + "="*60)
    print("⚙️  Experiment Configuration")
    print("="*60)
    print(f"Model: {config.model.name}")
    print(f"SAE: {config.model.sae_release}/{config.model.sae_id}")
    print(f"Hook: {config.model.hook_name}")
    print(f"Dataset: {config.data.dataset_path}")
    print(f"Sample size: {config.data.sample_size}")
    print(f"Max new tokens: {config.generation.max_new_tokens}")
    print(f"Temperature: {config.generation.temperature}")
    print(f"Do sample: {config.generation.do_sample}")
    print(f"Number of intervention features: {len(feature_ids)}")
    print(f"Feature IDs (first 5): {feature_ids[:5]}...")
    print("="*60 + "\n")


def analyze_results(results_path: str):
    """
    実験結果を分析して表示
    
    Args:
        results_path: 結果ファイルのパス
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("📊 Experiment Results Summary")
    print("="*60)
    print(f"Model: {results['metadata']['model_name']}")
    print(f"Intervention method: {results['metadata']['intervention_method']}")
    print(f"Number of intervention features: {results['metadata']['num_intervention_features']}")
    print(f"Total questions processed: {results['metadata']['num_questions']}")
    print(f"Question ID range: {results['metadata']['question_id_range']['start']} - {results['metadata']['question_id_range']['end']}")
    print(f"Timestamp: {results['metadata']['timestamp']}")
    print("="*60)
    
    # 活性化サマリの表示
    if 'activation_summary' in results:
        act_summary = results['activation_summary']
        print("\n📈 Activation Summary:")
        print(f"  Total prompts processed: {act_summary['num_prompts']}")
        print(f"  Number of intervention features: {act_summary['num_intervention_features']}")
    
    # サンプル結果の表示
    if results['results']:
        first_question = results['results'][0]
        
        print("\n📝 Sample Result (Question 1):")
        print("="*60)
        print(f"Dataset: {first_question['dataset']}")
        print(f"Base text: {first_question['base_text'][:100]}...")
        print(f"Number of variations: {len(first_question['variations'])}")
        
        # 最初のバリエーションの詳細
        first_variation = first_question['variations'][0]
        print(f"\n--- Variation 1: {first_variation['template']} ---")
        print(f"Prompt: {first_variation['prompt'][:150]}...")
        print(f"\nBaseline Response:")
        print(f"  {first_variation['baseline_response']}")
        print(f"\nIntervention Response:")
        print(f"  {first_variation['intervention_response']}")
        print("="*60)
    
    # 応答変化率の分析
    total_variations = 0
    changed_variations = 0
    
    for question in results['results']:
        for variation in question['variations']:
            total_variations += 1
            baseline = variation['baseline_response'].strip()
            intervention = variation['intervention_response'].strip()
            
            if baseline != intervention:
                changed_variations += 1
    
    change_rate = (changed_variations / total_variations * 100) if total_variations > 0 else 0
    
    print("\n📈 Response Change Analysis:")
    print("="*60)
    print(f"Total variations processed: {total_variations}")
    print(f"Variations with changed response: {changed_variations}")
    print(f"Change rate: {change_rate:.2f}%")
    print("="*60)
    
    print("\nℹ️  Next steps:")
    print("  1. Use GPT-4o to evaluate sycophancy flags for baseline vs intervention")
    print("  2. Use GPT-4o to rate naturalness scores (1-5 scale)")
    print("  3. Perform McNemar's test for statistical significance")
    print("  4. Analyze qualitative changes in response content")


def main():
    parser = argparse.ArgumentParser(
        description="介入実験を実行して、特定のSAE特徴量の効果を評価します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CSVから特徴量IDを読み込んで実験を実行
  python run_intervention_experiment.py --features-csv results/intervention/candidates/intervention_candidates_20251211_193847.csv
  
  # 特定の範囲で実験を実行
  python run_intervention_experiment.py --features-csv results/intervention/candidates/intervention_candidates_20251211_193847.csv --start-index 100 --end-index 200
  
  # サンプルサイズを指定
  python run_intervention_experiment.py --features-csv results/intervention/candidates/intervention_candidates_20251211_193847.csv --sample-size 10
  
  # 特徴量IDを直接指定
  python run_intervention_experiment.py --features 123,456,789 --sample-size 5
        """
    )
    
    # 特徴量ID指定オプション
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument(
        '--features-csv',
        type=str,
        help='特徴量IDを含むCSVファイルのパス'
    )
    feature_group.add_argument(
        '--features-json',
        type=str,
        help='特徴量IDを含むJSONファイルのパス'
    )
    feature_group.add_argument(
        '--features',
        type=str,
        help='カンマ区切りの特徴量IDリスト（例: 123,456,789）'
    )
    
    # 実験範囲指定オプション
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='処理するサンプル数（デフォルト: configの設定を使用）'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=None,
        help='開始インデックス（0-based）'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='終了インデックス（0-based）'
    )
    
    # その他のオプション
    parser.add_argument(
        '--config',
        type=str,
        default='INTERVENTION_GEMMA2_9B_IT_CONFIG',
        help='使用する設定名（デフォルト: INTERVENTION_GEMMA2_9B_IT_CONFIG）'
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='実験実行後の結果分析をスキップ'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='結果の出力ディレクトリ（デフォルト: results/intervention）'
    )
    
    args = parser.parse_args()
    
    # 特徴量IDの読み込み
    print("📂 Loading intervention feature IDs...")
    if args.features_csv:
        feature_ids = load_feature_ids_from_csv(args.features_csv)
        print(f"✅ Loaded {len(feature_ids)} features from CSV: {args.features_csv}")
    elif args.features_json:
        feature_ids = load_feature_ids_from_json(args.features_json)
        print(f"✅ Loaded {len(feature_ids)} features from JSON: {args.features_json}")
    else:
        feature_ids = [int(x.strip()) for x in args.features.split(',')]
        print(f"✅ Using {len(feature_ids)} features from command line")
    
    # GPU情報の表示
    print_gpu_info()
    
    # 設定の読み込み
    print(f"\n📋 Loading configuration: {args.config}")
    import config as config_module
    if not hasattr(config_module, args.config):
        print(f"❌ Error: Configuration '{args.config}' not found in config.py")
        sys.exit(1)
    
    experiment_config = getattr(config_module, args.config)
    
    # サンプルサイズの上書き
    if args.sample_size is not None:
        experiment_config.data.sample_size = args.sample_size
    
    # 設定の表示
    print_configuration(experiment_config, feature_ids)
    
    # InterventionRunnerの初期化
    print("🔧 Initializing InterventionRunner...")
    runner = InterventionRunner(
        config=experiment_config,
        intervention_feature_ids=feature_ids
    )
    print("✅ InterventionRunner initialized")
    
    # 出力ディレクトリの設定
    if args.output_dir:
        runner.results_dir = Path(args.output_dir)
        runner.results_dir.mkdir(parents=True, exist_ok=True)
    
    # 実験の実行
    print("\n🚀 Starting intervention experiment...")
    try:
        output_path = runner.run_complete_experiment(
            sample_size=args.sample_size,
            start_index=args.start_index,
            end_index=args.end_index
        )
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        
        # 結果の分析
        if not args.no_analysis:
            analyze_results(output_path)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
