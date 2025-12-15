"""
SAE Activation Extractor
Teacher Forcingを利用して指定レイヤーのSAE activationを取得するクラス
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from transformer_lens import HookedTransformer
from sae_lens import SAE
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ExtractionConfig:
    """SAE Activation抽出の設定"""
    model_name: str
    sae_release: str
    sae_id: str
    target_layer: int
    hook_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    top_k_features: int = 50  # 保存する上位特徴数


class SAEActivationExtractor:
    """
    Teacher Forcingを使用してSAE activationを抽出するクラス
    
    使用例:
        config = ExtractionConfig(
            model_name="google/gemma-2-9b-it",
            sae_release="gemma-scope-9b-pt-res-canonical",
            sae_id="layer_20/width_16k/canonical",
            target_layer=20,
            hook_name="blocks.20.hook_resid_post"
        )
        extractor = SAEActivationExtractor(config)
        result = extractor.extract_activations(prompt, response)
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model: Optional[HookedTransformer] = None
        self.sae: Optional[SAE] = None
        
    def load_model_and_sae(self):
        """モデルとSAEをロード"""
        print(f"🔄 Loading model: {self.config.model_name}")
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model_name,
            device=self.config.device,
            dtype=self.config.dtype
        )
        
        print(f"🔄 Loading SAE: {self.config.sae_release}/{self.config.sae_id}")
        self.sae = SAE.from_pretrained(
            release=self.config.sae_release,
            sae_id=self.config.sae_id,
            device=self.config.device
        )
        
        print(f"✅ Model and SAE loaded successfully")
        print(f"   Target Layer: {self.config.target_layer}")
        print(f"   Hook Name: {self.config.hook_name}")
        
    def _find_answer_start_position(self, full_tokens: torch.Tensor, prompt_str: str) -> int:
        """
        プロンプトの終わり（回答の始まり）のトークン位置を特定
        
        Args:
            full_tokens: フルシーケンスのトークン
            prompt_str: プロンプト文字列
            
        Returns:
            回答開始位置のインデックス（0-indexed）
        """
        prompt_tokens = self.model.to_tokens(prompt_str, prepend_bos=True)
        return prompt_tokens.shape[1] - 1
    
    def extract_activations(
        self,
        prompt: str,
        response: str,
        save_all_tokens: bool = False
    ) -> Dict[str, Any]:
        """
        Teacher Forcingを使用してSAE activationを抽出
        
        Args:
            prompt: プロンプト文字列
            response: 応答文字列
            save_all_tokens: Trueの場合、全トークンのactivationを保存
                           Falseの場合、プロンプト最後のトークン（応答直前）のみ保存
                           
        Returns:
            抽出結果を含む辞書
        """
        if self.model is None or self.sae is None:
            raise RuntimeError("Model and SAE must be loaded first. Call load_model_and_sae().")
        
        # Teacher Forcing Input: プロンプト + 実際の応答
        full_text = prompt + response
        input_tokens = self.model.to_tokens(full_text, prepend_bos=True)
        
        # コンテキスト長チェック
        if input_tokens.shape[1] > self.model.cfg.n_ctx:
            return {
                "status": "error",
                "error": f"Sequence too long: {input_tokens.shape[1]} > {self.model.cfg.n_ctx}"
            }
        
        # 回答開始位置を特定
        answer_start_pos = self._find_answer_start_position(input_tokens, prompt)
        
        # Activationを保存するための辞書
        activation_storage = {}
        
        def capture_hook(activation, hook):
            """
            Activationをキャプチャするフック
            activation: [batch, seq, d_model]
            """
            if save_all_tokens:
                # 全トークンのactivationを保存
                for pos in range(activation.shape[1]):
                    act = activation[:, pos:pos+1, :]
                    f_acts = self.sae.encode(act)  # [1, 1, n_features]
                    activation_storage[f"token_{pos}"] = f_acts.detach().cpu()
            else:
                # プロンプト最後のトークン（応答直前）のみ保存
                target_act = activation[:, answer_start_pos:answer_start_pos+1, :]
                f_acts = self.sae.encode(target_act)  # [1, 1, n_features]
                activation_storage["last_token"] = f_acts.detach().cpu()
            
            return activation
        
        # Forward Pass with Hook
        self.model.eval()
        with torch.no_grad():
            _ = self.model.run_with_hooks(
                input_tokens,
                fwd_hooks=[(self.config.hook_name, capture_hook)]
            )
        
        # 結果を整形
        result = {
            "status": "success",
            "answer_start_position": answer_start_pos,
            "total_tokens": input_tokens.shape[1],
            "prompt_length": len(prompt),
            "response_length": len(response),
            "activations": {}
        }
        
        # Top-k特徴を抽出して保存
        for token_key, f_acts in activation_storage.items():
            f_acts_flat = f_acts.squeeze()  # [n_features]
            
            # Top-k特徴を取得
            top_values, top_indices = torch.topk(f_acts_flat, k=self.config.top_k_features)
            
            # 非ゼロの特徴のみを保存
            top_features = []
            for idx, val in zip(top_indices, top_values):
                if val.item() > 0:
                    top_features.append({
                        "id": int(idx.item()),
                        "activation": float(val.item())
                    })
            
            result["activations"][token_key] = {
                "top_k_features": top_features,
                "num_active_features": len(top_features)
            }
        
        return result
    
    def extract_batch(
        self,
        samples: List[Dict[str, Any]],
        save_all_tokens: bool = False,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        複数サンプルに対してバッチ処理
        
        Args:
            samples: サンプルのリスト。各サンプルは {'prompt': str, 'response': str, ...} の形式
            save_all_tokens: 全トークンのactivationを保存するか
            verbose: 進捗を表示するか
            
        Returns:
            各サンプルの抽出結果のリスト
        """
        results = []
        
        for i, sample in enumerate(samples):
            if verbose and (i + 1) % 10 == 0:
                print(f"   Processing sample {i + 1}/{len(samples)}...")
            
            try:
                result = self.extract_activations(
                    prompt=sample['prompt'],
                    response=sample['response'],
                    save_all_tokens=save_all_tokens
                )
                
                # 元のサンプル情報を保持
                result.update({
                    "question_id": sample.get("question_id"),
                    "template_type": sample.get("template_type"),
                    "variation_index": sample.get("variation_index")
                })
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e),
                    "question_id": sample.get("question_id"),
                    "template_type": sample.get("template_type")
                })
                if verbose:
                    print(f"   ⚠️ Error processing sample {i}: {e}")
            
            # メモリクリーンアップ
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def cleanup(self):
        """メモリクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.sae is not None:
            del self.sae
            self.sae = None
        torch.cuda.empty_cache()
        print("✅ Memory cleaned up")


def load_samples_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    分析結果JSONからサンプルを読み込む
    
    Args:
        json_path: 分析結果JSONファイルのパス
        
    Returns:
        サンプルのリスト
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    results = data.get("results", [])
    
    for result in results:
        question_id = result.get("question_id")
        variations = result.get("variations", [])
        
        for idx, variation in enumerate(variations):
            sample = {
                "question_id": question_id,
                "variation_index": idx,
                "template_type": variation.get("template_type", "(base)"),
                "prompt": variation.get("prompt", ""),
                "response": variation.get("response", "")
            }
            samples.append(sample)
    
    return samples


def save_results_to_json(
    original_json_path: str,
    extraction_results: List[Dict[str, Any]],
    output_json_path: str,
    config: ExtractionConfig
):
    """
    抽出結果を元のJSONに統合して保存
    
    Args:
        original_json_path: 元の分析結果JSONパス
        extraction_results: 抽出結果のリスト
        output_json_path: 出力先JSONパス
        config: 抽出設定
    """
    # 元のJSONを読み込み
    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # メタデータを追加
    if "sae_extraction_metadata" not in data:
        data["sae_extraction_metadata"] = {}
    
    data["sae_extraction_metadata"][f"layer_{config.target_layer}"] = {
        "sae_id": config.sae_id,
        "sae_release": config.sae_release,
        "hook_name": config.hook_name,
        "top_k_features": config.top_k_features
    }
    
    # 結果を統合
    for extract_result in extraction_results:
        question_id = extract_result.get("question_id")
        variation_idx = extract_result.get("variation_index")
        
        if question_id is None or variation_idx is None:
            continue
        
        # 該当するvariationを探して結果を追加
        for result in data["results"]:
            if result["question_id"] == question_id:
                variations = result["variations"]
                if variation_idx < len(variations):
                    # 既存のSAE activationフィールドに追加
                    if "sae_activations" not in variations[variation_idx]:
                        variations[variation_idx]["sae_activations"] = {}
                    
                    variations[variation_idx]["sae_activations"][f"layer_{config.target_layer}"] = {
                        "activations": extract_result.get("activations", {}),
                        "answer_start_position": extract_result.get("answer_start_position"),
                        "total_tokens": extract_result.get("total_tokens"),
                        "status": extract_result.get("status")
                    }
                break
    
    # 保存
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")
