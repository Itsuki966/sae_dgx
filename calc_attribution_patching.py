#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attribution Patching Analysis for Sycophancy Detection
SAE特徴量の因果的寄与を分析するスクリプト
"""

import os
import sys
import json
import gc
import torch
from typing import Dict, Any, Generator
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformer_lens import HookedTransformer
from sae_lens import SAE

# メモリ最適化設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_grad_enabled(True)  # 勾配計算を有効化


def yield_sycophancy_samples(data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    JSONデータからAttribution Patching用のサンプルを生成するジェネレータ
    """
    results = data.get("results", [])

    for result in results:
        variations = result.get("variations", [])
        question_id = result.get("question_id")

        # 1. Base回答の特定
        base_variation = None
        base_idx = -1  # Baseのインデックスを保持
        for idx, var in enumerate(variations):
            t_type = var.get("template_type")
            if t_type == "base" or t_type == "(base)" or not t_type:
                base_variation = var
                base_idx = idx
                break

        if not base_variation:
            continue

        # 2. ターゲット（迎合）回答の特定とペアリング
        for idx, target_variation in enumerate(variations):
            if target_variation is base_variation:
                continue

            if target_variation.get("sycophancy_flag") == 1:
                yield {
                    "question_id": question_id,
                    "variation_index": idx,
                    "base_variation_index": base_idx,  # Baseのインデックスを返す
                    "template_type": target_variation.get("template_type"),
                    "prompt": target_variation.get("prompt"),
                    "target_response": target_variation.get("response"),
                    "base_response": base_variation.get("response")
                }


class AttributionPatchingAnalyzer:
    def __init__(self, model: HookedTransformer, sae: SAE, config: Any):
        self.model = model
        self.sae = sae
        self.config = config
        # hook_name を直接指定（config.py から取得するのが理想）
        # Gemma Scope SAE の sae_id から推定
        # 例: "layer_31/width_16k/canonical" → "blocks.31.hook_resid_post"
        sae_id = config.model.sae_id
        layer_num = sae_id.split('/')[0].replace('layer_', '')
        self.hook_name = f"blocks.{layer_num}.hook_resid_post"
        print(f"   🎯 Using hook: {self.hook_name}")

    def _find_answer_start_position(self, full_tokens: torch.Tensor, prompt_str: str) -> int:
        """
        プロンプトの終わり（回答の始まり）のトークン位置を特定する
        """
        # プロンプト単体でのトークン長を取得
        # Note: BOSトークン等の扱いに注意。Gemmaはadd_bos_token=Trueがデフォルト
        prompt_tokens = self.model.to_tokens(prompt_str, prepend_bos=True)
        return prompt_tokens.shape[1] - 1  # 0-indexed なので -1 (最後のトークン位置)

    def calculate_atp_for_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        1サンプルに対するAttribution Patchingを実行
        """
        prompt = sample["prompt"]
        response = sample["target_response"]
        base_response = sample["base_response"]

        # 1. トークン化 (Teacher Forcing Input)
        # プロンプト + 実際の回答
        full_text = prompt + response
        input_tokens = self.model.to_tokens(full_text, prepend_bos=True)

        # 回答開始位置（プロンプトの最後のトークン位置）を特定
        # ここが「次のトークン（回答の1文字目）」を予測する位置になる
        target_pos = self._find_answer_start_position(input_tokens, prompt)

        # 入力長チェック (モデルのコンテキスト長を超えないか)
        if input_tokens.shape[1] > self.model.cfg.n_ctx:
            return {"error": "Sequence too long"}

        # 2. Target Token と Base Token の ID を取得
        # Base回答をトークン化
        base_full_text = prompt + base_response
        base_input_tokens = self.model.to_tokens(base_full_text, prepend_bos=True)
        base_target_pos = self._find_answer_start_position(base_input_tokens, prompt)

        # 複数トークン位置を試して、異なるトークンペアを見つける
        max_tokens_to_check = 20  # 最大20トークンまでチェック
        target_token_id = None
        base_token_id = None
        token_offset = 1  # プロンプト直後からスタート

        for offset in range(1, max_tokens_to_check + 1):
            try:
                candidate_target = input_tokens[0, target_pos + offset].item()
                candidate_base = base_input_tokens[0, base_target_pos + offset].item()

                # 異なるトークンが見つかったら採用
                if candidate_target != candidate_base:
                    target_token_id = candidate_target
                    base_token_id = candidate_base
                    token_offset = offset
                    break
            except IndexError:
                # どちらかの回答が短すぎる場合
                break

        # すべてのトークンが同一、または取得失敗の場合
        if target_token_id is None or base_token_id is None:
            return {"skipped": "No differing tokens found in first 5 positions"}

        # 使用するトークン位置を更新（Logit取得用）
        # target_pos はプロンプト最後の位置なので、offset-1 の位置のLogitを見る
        logit_pos = target_pos + token_offset - 1

        # 3-0. Base回答での特徴量取得（Attribution Patchingの差分計算用）
        # 勾配計算は不要、値のみ保存
        self.model.eval()
        base_f_acts = None
        
        with torch.no_grad():
            base_storage = {}
            
            def base_hook(activation, hook):
                """Base回答の特徴量を取得（勾配不要）"""
                base_act = activation[:, base_target_pos:base_target_pos+1, :]
                base_storage['acts'] = self.sae.encode(base_act)
                return activation
            
            try:
                _ = self.model.run_with_hooks(
                    base_input_tokens,
                    fwd_hooks=[(self.hook_name, base_hook)]
                )
                base_f_acts = base_storage['acts'].detach().cpu()
            except Exception as e:
                # Base特徴量の取得に失敗した場合は警告のみ（処理は継続）
                print(f"⚠️ Failed to get base features: {e}")
            finally:
                del base_storage
                torch.cuda.empty_cache()

        # 3. Forward Pass & Metric Calculation
        self.model.zero_grad()

        # フック内でデータをキャプチャするためのコンテナ
        feature_acts_storage = {}

        def atp_hook(activation, hook):
            """
            Activationを取得し、SAEを通して勾配を流すフック
            """
            # activation: [batch, seq, d_model]
            # ターゲット位置（回答直前）のみを抽出
            # batch=1 前提
            target_act = activation[:, target_pos:target_pos+1, :]

            # SAE Encode (Feature Activation計算)
            # SAEの入力次元に合わせて調整
            f_acts = self.sae.encode(target_act)  # [1, 1, n_features]

            # 勾配計算のために保存 (retain_grad重要)
            f_acts.requires_grad_(True)
            f_acts.retain_grad()
            feature_acts_storage['acts'] = f_acts

            # SAE Decode (Reconstruction)
            x_hat = self.sae.decode(f_acts)

            # Gradient Trick:
            # Forward: 元のActivation (x) をそのまま流す (Teacher Forcingの精度維持)
            # Backward: Reconstruction (x_hat) を通して勾配を流す (SAE特徴量へのPathを作る)
            # x_out = x_hat + (x - x_hat).detach()
            # これにより、Metricの勾配は x_hat -> f_acts と伝播する

            x_out = x_hat + (target_act - x_hat).detach()

            # 元のシーケンスに戻す
            activation[:, target_pos:target_pos+1, :] = x_out
            return activation

        # モデル実行
        try:
            logits = self.model.run_with_hooks(
                input_tokens,
                fwd_hooks=[(self.hook_name, atp_hook)]
            )

            # 4. Metric Calculation (Logit Difference)
            # logit_pos の位置での予測を見る（offsetに応じた位置）
            target_logit = logits[0, logit_pos, target_token_id]
            base_logit = logits[0, logit_pos, base_token_id]
            metric = target_logit - base_logit

            # 5. Backward Pass
            metric.backward()

            # 6. AtP Score Calculation
            # Score = (Target特徴量 - Base特徴量) * Gradient
            # これにより「Base→Targetの変化が引き起こした効果」を測定
            f_acts = feature_acts_storage['acts']
            f_grad = f_acts.grad

            if f_acts is None or f_grad is None:
                return {"error": "Failed to capture gradients"}

            # Target特徴量をCPUに移動
            f_acts_cpu = f_acts.detach().cpu().squeeze()  # [n_features]
            f_grad_cpu = f_grad.detach().cpu().squeeze()  # [n_features]
            
            # Attribution Patching: 差分 × 勾配
            if base_f_acts is not None:
                base_f_acts_squeezed = base_f_acts.squeeze()  # [n_features]
                delta_f = f_acts_cpu - base_f_acts_squeezed  # Target - Base
                atp_scores = delta_f * f_grad_cpu
            else:
                # Baseの取得に失敗した場合はフォールバック（従来の方法）
                print("⚠️ Using fallback: f_acts * f_grad (no base comparison)")
                atp_scores = f_acts_cpu * f_grad_cpu

            # 結果の抽出（Top-K & Non-zero）
            # メモリ節約のため、スコアが高いものだけを保存
            top_k = 50
            top_indices = torch.topk(atp_scores.abs(), k=top_k).indices

            top_features = []
            for idx in top_indices:
                idx_val = idx.item()
                score = atp_scores[idx_val].item()
                target_act = f_acts_cpu[idx_val].item()
                
                feature_dict = {
                    "id": str(idx_val),
                    "score": score,
                    "target_activation": target_act,
                    "gradient": f_grad_cpu[idx_val].item()
                }
                
                # Base特徴量がある場合は差分情報も追加
                if base_f_acts is not None:
                    base_act = base_f_acts.squeeze()[idx_val].item()
                    feature_dict["base_activation"] = base_act
                    feature_dict["activation_delta"] = target_act - base_act
                
                top_features.append(feature_dict)

            return {
                "status": "success",
                "target_token": self.model.to_string(target_token_id),
                "base_token": self.model.to_string(base_token_id),
                "token_position": token_offset,  # どの位置のトークンを使ったか記録
                "logit_diff": metric.item(),
                "top_features": top_features
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return {"error": "OOM"}
            raise e
        finally:
            # メモリクリーンアップ
            self.model.zero_grad()
            del feature_acts_storage
            if 'logits' in locals():
                del logits
            if 'f_acts' in locals():
                del f_acts
            if 'f_grad' in locals():
                del f_grad
            if 'base_f_acts' in locals():
                del base_f_acts
            torch.cuda.empty_cache()


def run_attribution_patching_pipeline(
    input_json_path: str = None,
    output_json_path: str = None,
    config_name: str = None,
    layer: int = None
):
    """
    Attribution Patching分析のメインパイプライン
    
    Args:
        input_json_path: 入力JSONファイルのパス（指定しない場合は自動検索）
        output_json_path: 出力JSONファイルのパス（指定しない場合は自動生成）
        config_name: 使用するconfig名（config.pyから読み込み、layerと併用不可）
        layer: 解析対象のlayer番号（9, 20, 31をサポート、config_nameより優先）
    """
    # プロジェクトルートの取得
    project_root = Path(__file__).parent.absolute()
    
    # デフォルトのパス設定
    if input_json_path is None:
        input_json_path = project_root / "results/labeled_data/combined_feedback_data.json"
    else:
        input_json_path = Path(input_json_path)
    
    # ファイル存在確認
    if not input_json_path.exists():
        # テスト用に最新の結果ファイルを探す
        search_dir = project_root / "results/feedback"
        files = list(search_dir.glob("feedback_analysis_*.json"))
        if files:
            input_json_path = sorted(files)[-1]
            print(f"⚠️ 指定されたファイルが見つからないため、最新のファイルを使用します: {input_json_path}")
        else:
            raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")

    # 出力パスの設定
    if output_json_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json_path = project_root / f"results/feedback/atp_results_gemma-2-9b-it_{timestamp}.json"
    else:
        output_json_path = Path(output_json_path)
    
    # 出力ディレクトリの作成
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. データの読み込み
    print(f"📂 Loading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Configの読み込み
    if layer is not None:
        # layer番号からconfigを自動選択
        if layer == 9:
            from config import FEEDBACK_GEMMA2_9B_IT_LAYER9_CONFIG
            config = FEEDBACK_GEMMA2_9B_IT_LAYER9_CONFIG
            print(f"   📍 Using Layer 9 config")
        elif layer == 20:
            from config import FEEDBACK_GEMMA2_9B_IT_LAYER20_CONFIG
            config = FEEDBACK_GEMMA2_9B_IT_LAYER20_CONFIG
            print(f"   📍 Using Layer 20 config")
        elif layer == 31:
            from config import FEEDBACK_GEMMA2_9B_IT_CONFIG
            config = FEEDBACK_GEMMA2_9B_IT_CONFIG
            print(f"   📍 Using Layer 31 config")
        else:
            raise ValueError(f"Unsupported layer: {layer}. Supported layers: 9, 20, 31")
    elif config_name is not None:
        # config名から直接読み込み
        import config as config_module
        config = getattr(config_module, config_name)
        print(f"   📍 Using config: {config_name}")
    else:
        # デフォルトはLayer 31
        from config import FEEDBACK_GEMMA2_9B_IT_CONFIG
        config = FEEDBACK_GEMMA2_9B_IT_CONFIG
        print(f"   📍 Using default Layer 31 config")

    # 3. モデルとSAEの準備
    print("🔄 Loading Model & SAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    model = HookedTransformer.from_pretrained_no_processing(
        config.model.name,
        device=device,
        dtype=torch.bfloat16
    )

    # SAEロード
    sae = SAE.from_pretrained(
        release=config.model.sae_release,
        sae_id=config.model.sae_id,
        device=device
    )

    analyzer = AttributionPatchingAnalyzer(model, sae, config)

    # 4. メインループ
    samples = list(yield_sycophancy_samples(data))
    print(f"🚀 Starting ATP analysis for {len(samples)} samples...")
    print(f"💾 Results will be saved to: {output_json_path}")

    for i, sample in enumerate(tqdm(samples)):
        res = analyzer.calculate_atp_for_sample(sample)

        # 元のJSONに結果を統合
        question_id = sample["question_id"]
        variation_idx = sample["variation_index"]
        base_variation_idx = sample["base_variation_index"]

        # デバッグ: 結果の内容を表示（最初の数サンプルのみ）
        if i < 3:
            print(f"\n🔍 Sample {i} result: {res}")

        # 該当するvariationを探して atp_analysis フィールドを追加
        for result in data["results"]:
            if result["question_id"] == question_id:
                variations = result["variations"]
                if variation_idx < len(variations):
                    if res.get("status") == "success":
                        # atp_analysis フィールドに保存
                        variations[variation_idx]["atp_analysis"] = {
                            "top_features": res["top_features"],
                            "target_token": res["target_token"],
                            "base_token": res["base_token"],
                            "token_position": res["token_position"],
                            "logit_diff": res["logit_diff"]
                        }
                        
                        # ★ sae_activations フィールドにも保存（後続スクリプト用） ★
                        activation_key = "prompt_last_token"  # デフォルトのキー名
                        
                        # Target活性値の辞書を作成 {feature_id: activation}
                        target_acts_dict = {
                            f["id"]: f["target_activation"] 
                            for f in res["top_features"]
                        }
                        
                        # sae_activations がなければ作成
                        if "sae_activations" not in variations[variation_idx]:
                            variations[variation_idx]["sae_activations"] = {}
                        
                        variations[variation_idx]["sae_activations"][activation_key] = target_acts_dict
                        
                        # Base variation にも活性値を保存（Log Ratio計算用）
                        if base_variation_idx >= 0 and base_variation_idx < len(variations):
                            base_var = variations[base_variation_idx]
                            
                            # Base活性値の辞書を作成
                            base_acts_dict = {
                                f["id"]: f["base_activation"]
                                for f in res["top_features"]
                                if "base_activation" in f
                            }
                            
                            if "sae_activations" not in base_var:
                                base_var["sae_activations"] = {}
                            
                            # 既存の値があればマージ（複数のTargetから参照される可能性）
                            if activation_key not in base_var["sae_activations"]:
                                base_var["sae_activations"][activation_key] = {}
                            
                            base_var["sae_activations"][activation_key].update(base_acts_dict)
                        
                    else:
                        # エラーの場合、詳細情報も保存
                        variations[variation_idx]["atp_analysis"] = {
                            "error": res.get("error") or res.get("skipped") or "unknown",
                            "details": res
                        }
                        # 最初のエラーを表示
                        if i < 10:
                            print(f"⚠️ Sample {i} (Q{question_id}, Var{variation_idx}): {res}")
                break

        # 定期的に保存
        if (i + 1) % 10 == 0:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            gc.collect()
            torch.cuda.empty_cache()

    # 最終保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Analysis completed. Saved to {output_json_path}")


def main():
    """コマンドラインからの実行用エントリーポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Attribution Patching Analysis for Sycophancy Detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON file path (default: results/labeled_data/combined_feedback_data.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=[9, 20, 31],
        help="Layer number to analyze (9, 20, or 31). Overrides --config if specified."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config name to use from config.py (ignored if --layer is specified)"
    )
    
    args = parser.parse_args()
    
    try:
        run_attribution_patching_pipeline(
            input_json_path=args.input,
            output_json_path=args.output,
            config_name=args.config,
            layer=args.layer
        )
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
