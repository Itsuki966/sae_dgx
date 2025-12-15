"""
SAE Activation抽出の実行スクリプト
Teacher Forcingを使用して指定レイヤーのSAE activationを抽出します
"""

import os
import sys
import argparse
import torch
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sae_activation_extractor import (
    SAEActivationExtractor,
    ExtractionConfig
)


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="Extract SAE activations using Teacher Forcing"
    )
    
    # 必須引数
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path (e.g., results/labeled_data/combined_feedback_data.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated with timestamp)"
    )
    
    # モデル設定
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="Model name (default: google/gemma-2-9b-it)"
    )
    
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-9b-pt-res-canonical",
        help="SAE release name"
    )
    
    parser.add_argument(
        "--sae-id",
        type=str,
        default="layer_20/width_16k/canonical",
        help="SAE ID (default: layer_20/width_16k/canonical)"
    )
    
    parser.add_argument(
        "--target-layer",
        type=int,
        default=20,
        help="Target layer number (default: 20)"
    )
    
    parser.add_argument(
        "--hook-name",
        type=str,
        default=None,
        help="Hook name (default: auto-generated from target-layer)"
    )
    
    # 抽出設定
    parser.add_argument(
        "--save-all-tokens",
        action="store_true",
        help="Save activations for all tokens (default: only last token before response)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top features to save (default: 50)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 SAE Activation Extraction")
    print("=" * 60)
    
    # 入力ファイルのチェック
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Hook名の自動生成
    hook_name = args.hook_name
    if hook_name is None:
        hook_name = f"blocks.{args.target_layer}.hook_resid_post"
    
    # 設定の作成
    config = ExtractionConfig(
        model_name=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        target_layer=args.target_layer,
        hook_name=hook_name,
        top_k_features=args.top_k,
        dtype=torch.bfloat16
    )
    
    try:
        # Extractorの初期化
        print(f"\n🔧 Initializing SAE Activation Extractor...")
        extractor = SAEActivationExtractor(config)
        
        # 完全な抽出実行（読み込み + 分析 + 保存）
        extractor.run_complete_extraction(
            input_json_path=args.input,
            sample_size=args.max_samples,
            save_all_tokens=args.save_all_tokens,
            verbose=args.verbose
        )
        
        # クリーンアップ
        print(f"\n🧹 Cleaning up...")
        extractor.cleanup()
        
        print(f"\n" + "=" * 60)
        print(f"✅ Extraction completed successfully!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
