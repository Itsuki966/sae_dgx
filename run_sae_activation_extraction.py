"""
SAE Activation抽出の実行スクリプト
Teacher Forcingを使用して指定レイヤーのSAE activationを抽出します
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sae_activation_extractor import (
    SAEActivationExtractor,
    ExtractionConfig,
    load_samples_from_json,
    save_results_to_json
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
    
    # 出力ファイルパスの生成
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_stem = Path(args.input).stem
        output_path = f"results/feedback/{input_stem}_layer{args.target_layer}_{timestamp}.json"
    
    # 設定の作成
    config = ExtractionConfig(
        model_name=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        target_layer=args.target_layer,
        hook_name=hook_name,
        top_k_features=args.top_k
    )
    
    # 設定の表示
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   SAE Release: {config.sae_release}")
    print(f"   SAE ID: {config.sae_id}")
    print(f"   Target Layer: {config.target_layer}")
    print(f"   Hook Name: {config.hook_name}")
    print(f"   Top-K Features: {config.top_k_features}")
    print(f"   Save All Tokens: {args.save_all_tokens}")
    print(f"   Input: {args.input}")
    print(f"   Output: {output_path}")
    
    try:
        # サンプルの読み込み
        print(f"\n📂 Loading samples from {args.input}...")
        samples = load_samples_from_json(args.input)
        
        if args.max_samples is not None:
            samples = samples[:args.max_samples]
        
        print(f"✅ Loaded {len(samples)} samples")
        
        # Extractorの初期化
        print(f"\n🔧 Initializing SAE Activation Extractor...")
        extractor = SAEActivationExtractor(config)
        
        # モデルとSAEのロード
        extractor.load_model_and_sae()
        
        # Activation抽出の実行
        print(f"\n🔬 Extracting SAE activations...")
        print(f"   This may take several minutes...")
        
        extraction_results = extractor.extract_batch(
            samples=samples,
            save_all_tokens=args.save_all_tokens,
            verbose=args.verbose
        )
        
        # 統計情報の表示
        success_count = sum(1 for r in extraction_results if r.get("status") == "success")
        error_count = len(extraction_results) - success_count
        
        print(f"\n📊 Extraction Summary:")
        print(f"   Total Samples: {len(extraction_results)}")
        print(f"   Successful: {success_count}")
        print(f"   Errors: {error_count}")
        
        # エラーの詳細表示（最初の5件）
        if error_count > 0:
            print(f"\n⚠️ Error Details (first 5):")
            error_samples = [r for r in extraction_results if r.get("status") != "success"]
            for i, err in enumerate(error_samples[:5]):
                print(f"   {i+1}. Q{err.get('question_id')} - {err.get('error', 'Unknown error')}")
        
        # 結果の保存
        print(f"\n💾 Saving results...")
        save_results_to_json(
            original_json_path=args.input,
            extraction_results=extraction_results,
            output_json_path=output_path,
            config=config
        )
        
        # クリーンアップ
        print(f"\n🧹 Cleaning up...")
        extractor.cleanup()
        
        print(f"\n" + "=" * 60)
        print(f"✅ Extraction completed successfully!")
        print(f"📁 Results saved to: {output_path}")
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
