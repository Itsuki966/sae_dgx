"""
SAE Activation Extractor
Teacher Forcingを利用して指定レイヤーのSAE activationを取得するクラス
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from transformer_lens import HookedTransformer
from sae_lens import SAE
from dataclasses import dataclass, asdict
import json
import os
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


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


@dataclass
class PromptInfo:
    """プロンプト情報（feedback_analyzer.pyのFeedbackPromptInfo相当）"""
    dataset: str
    prompt_template_type: str
    prompt: str
    base_data: Dict[str, Any]


@dataclass
class ExtractionResponse:
    """1つのプロンプトに対する応答とSAE状態（feedback_analyzer.pyのFeedbackResponse相当）"""
    prompt_info: PromptInfo
    response_text: str
    sae_activations: Dict[str, Any]  # {token_key: {feature_id: activation_value}}
    top_k_features: List[Tuple[int, float]]  # [(feature_id, value), ...]
    metadata: Dict[str, Any]


@dataclass
class QuestionResult:
    """1つの質問の分析結果（feedback_analyzer.pyのFeedbackQuestionResult相当）"""
    question_id: int
    dataset: str
    base_text: str
    variations: List[ExtractionResponse]
    timestamp: str


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
        self.results: List[QuestionResult] = []
        self.save_all_tokens: bool = False  # デフォルトは最後のトークンのみ
        
        # 結果保存ディレクトリの作成
        self.results_dir = Path("results/feedback")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
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
        response: str
    ) -> Tuple[Dict[str, Any], List[Tuple[int, float]]]:
        """
        Teacher Forcingを使用してSAE activationを抽出
        feedback_analyzer.pyのgenerate_with_saeと同じ形式で返す
        
        Args:
            prompt: プロンプト文字列
            response: 応答文字列（既に生成済み）
                           
        Returns:
            (sae_activations, top_k_features)のタプル
            - sae_activations: {token_key: {feature_id: activation_value}}の疎ベクトル形式
            - top_k_features: [(feature_id, value), ...]のリスト
        """
        if self.model is None or self.sae is None:
            raise RuntimeError("Model and SAE must be loaded first. Call load_model_and_sae().")
        
        # Teacher Forcing Input: プロンプト + 実際の応答
        full_text = prompt + response
        input_tokens = self.model.to_tokens(full_text, prepend_bos=True)
        
        # コンテキスト長チェック
        if input_tokens.shape[1] > self.model.cfg.n_ctx:
            raise ValueError(f"Sequence too long: {input_tokens.shape[1]} > {self.model.cfg.n_ctx}")
        
        # 回答開始位置を特定
        answer_start_pos = self._find_answer_start_position(input_tokens, prompt)
        
        # Forward Passでactivationを取得
        self.model.eval()
        with torch.no_grad():
            _, cache = self.model.run_with_cache(input_tokens)
            
            # 対象レイヤーのactivationを取得
            activations = cache[self.config.hook_name]  # [batch, seq, d_model]
            
            # SAEエンコード
            sae_features = self.sae.encode(activations)  # [batch, seq, n_features]
            
            # NumPy配列に変換
            if self.save_all_tokens:
                # 全トークンのactivationを保存
                sae_activations_np = sae_features[0].cpu().numpy()  # [seq_len, n_features]
            else:
                # プロンプト最後のトークン（応答直前）のみ保存（デフォルト、推奨）
                sae_activations_np = sae_features[0, answer_start_pos:answer_start_pos+1].cpu().numpy()  # [1, n_features]
            
            # Top-k特徴を抽出（ログ・可視化用）
            if self.save_all_tokens:
                mean_activations = sae_activations_np.mean(axis=0)
            else:
                mean_activations = sae_activations_np[0]
            
            top_k_indices = np.argsort(mean_activations)[-self.config.top_k_features:][::-1]
            top_k_features = [(int(idx), float(mean_activations[idx])) for idx in top_k_indices]
            
            # 0より大きい全ての活性化を保存（疎ベクトル形式）
            active_features = {}
            
            if self.save_all_tokens:
                # 各トークン位置での活性化を保存
                for token_idx in range(sae_activations_np.shape[0]):
                    token_activations = sae_activations_np[token_idx]
                    active_indices = np.where(token_activations > 0)[0]
                    if len(active_indices) > 0:
                        active_features[f"token_{token_idx}"] = {
                            int(idx): float(token_activations[idx]) 
                            for idx in active_indices
                        }
            else:
                # プロンプト最後のトークンのみ（推奨）
                token_activations = sae_activations_np[0]
                active_indices = np.where(token_activations > 0)[0]
                active_features["prompt_last_token"] = {
                    int(idx): float(token_activations[idx]) 
                    for idx in active_indices
                }
        
        return active_features, top_k_features
    
    def analyze_sample(self, prompt_info: PromptInfo, response_text: str) -> ExtractionResponse:
        """
        1つのサンプルを分析（feedback_analyzer.pyのanalyze_prompt_variationと同じ）
        
        Args:
            prompt_info: プロンプト情報
            response_text: 応答テキスト（既に生成済み）
        
        Returns:
            ExtractionResponse オブジェクト
        """
        # SAE activationを抽出
        start_time = datetime.now()
        sae_activations, top_k_features = self.extract_activations(
            prompt_info.prompt, 
            response_text
        )
        end_time = datetime.now()
        
        # メタデータ
        metadata = {
            "extraction_time_ms": (end_time - start_time).total_seconds() * 1000,
            "response_length": len(response_text),
            "timestamp": datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            metadata["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1e6
        
        return ExtractionResponse(
            prompt_info=prompt_info,
            response_text=response_text,
            sae_activations=sae_activations,
            top_k_features=top_k_features,
            metadata=metadata
        )
    
    def analyze_question_group(
        self,
        question_id: int,
        dataset: str,
        base_text: str,
        variations: List[Dict[str, Any]],
        verbose: bool = True
    ) -> QuestionResult:
        """
        1つの質問（複数のバリエーション）を分析
        
        Args:
            question_id: 質問ID
            dataset: データセット名
            base_text: ベーステキスト
            variations: バリエーションのリスト
            verbose: 詳細ログを表示するか
        
        Returns:
            QuestionResult オブジェクト
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 Analyzing Question {question_id} ({len(variations)} variations)")
            print(f"{'='*60}")
        
        variation_results = []
        
        for variation in variations:
            # PromptInfoを作成
            prompt_info = PromptInfo(
                dataset=dataset,
                prompt_template_type=variation.get("template_type", "(base)"),
                prompt=variation.get("prompt", ""),
                base_data={"text": base_text}
            )
            
            # 分析実行
            response = self.analyze_sample(
                prompt_info=prompt_info,
                response_text=variation.get("response", "")
            )
            variation_results.append(response)
            
            if verbose:
                print(f"   ✅ {prompt_info.prompt_template_type}: {len(response.sae_activations)} token positions")
        
        return QuestionResult(
            question_id=question_id,
            dataset=dataset,
            base_text=base_text,
            variations=variation_results,
            timestamp=datetime.now().isoformat()
        )
    
    def run_extraction(
        self,
        input_json_path: str,
        sample_size: Optional[int] = None,
        save_all_tokens: bool = False,
        verbose: bool = True
    ):
        """
        分析を実行（feedback_analyzer.pyのrun_analysisと同じ）
        
        Args:
            input_json_path: 入力JSONファイルパス
            sample_size: 処理するサンプル数（Noneの場合は全て）
            save_all_tokens: 全トークンのactivationを保存するか
            verbose: 詳細ログを表示するか
        """
        self.save_all_tokens = save_all_tokens
        
        if verbose:
            print("\n" + "="*60)
            print("🚀 Starting SAE Activation Extraction")
            print("="*60)
            print(f"   Model: {self.config.model_name}")
            print(f"   Target Layer: {self.config.target_layer}")
            print(f"   SAE ID: {self.config.sae_id}")
            print(f"   Save All Tokens: {save_all_tokens}")
        
        # 入力JSONを読み込み
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", [])
        total_questions = len(results)
        
        # サンプル数の調整
        if sample_size is not None and sample_size < total_questions:
            results = results[:sample_size]
            if verbose:
                print(f"   📊 Processing {sample_size} questions (out of {total_questions})")
        else:
            if verbose:
                print(f"   📊 Processing all {total_questions} questions")
        
        # モデルとSAEのロード
        if self.model is None or self.sae is None:
            self.load_model_and_sae()
        
        # 各質問グループを分析
        for result in tqdm(results, desc="Processing questions"):
            question_id = result.get("question_id")
            dataset = result.get("dataset", "unknown")
            base_text = result.get("base_text", "")
            variations = result.get("variations", [])
            
            try:
                question_result = self.analyze_question_group(
                    question_id=question_id,
                    dataset=dataset,
                    base_text=base_text,
                    variations=variations,
                    verbose=False  # プログレスバー使用時は個別ログを抑制
                )
                self.results.append(question_result)
                
                # メモリクリーンアップ
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Error processing question {question_id}: {e}")
                continue
        
        if verbose:
            print("\n" + "="*60)
            print("✅ Extraction Complete")
            print("="*60)
            print(f"📊 Processed {len(self.results)} questions")
            print(f"💾 Total variations: {sum(len(r.variations) for r in self.results)}")
    
    def save_results(self, output_path: Optional[str] = None):
        """
        分析結果を保存（feedback_analyzer.pyのsave_resultsと同じ形式）
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
        """
        if not self.results:
            print("⚠️ No results to save")
            return
        
        if output_path is None:
            # ファイル名: model_layer{XX}_{position}_YYYYMMDD_HHMMSS.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model_name.replace("/", "_").replace("-", "_")
            position = "all_tokens" if self.save_all_tokens else "last_token"
            output_path = self.results_dir / f"{model_name}_layer{self.config.target_layer}_{position}_{timestamp}.json"
        
        # 結果を辞書に変換（feedback_analyzer.pyと同じ形式）
        output_data = {
            "metadata": {
                "model_name": self.config.model_name,
                "sae_release": self.config.sae_release,
                "sae_id": self.config.sae_id,
                "target_layer": self.config.target_layer,
                "hook_name": self.config.hook_name,
                "num_questions": len(self.results),
                "save_all_tokens": self.save_all_tokens,
                "analyzed_position": "all_prompt_tokens" if self.save_all_tokens else "prompt_last_token",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "top_k_features": self.config.top_k_features,
                    "device": self.config.device,
                    "dtype": str(self.config.dtype)
                }
            },
            "results": []
        }
        
        # 各質問の結果を追加
        for result in self.results:
            question_data = {
                "question_id": result.question_id,
                "dataset": result.dataset,
                "base_text": result.base_text[:200] + "..." if len(result.base_text) > 200 else result.base_text,
                "variations": []
            }
            
            for variation in result.variations:
                variation_data = {
                    "template_type": variation.prompt_info.prompt_template_type,
                    "prompt": variation.prompt_info.prompt,
                    "response": variation.response_text,
                    "sae_activations": variation.sae_activations,
                    "top_k_features": variation.top_k_features,
                    "metadata": variation.metadata
                }
                question_data["variations"].append(variation_data)
            
            output_data["results"].append(question_data)
        
        # JSONファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"   📦 File size: {file_size:.2f} MB")
        
        return output_path
    
    def run_complete_extraction(
        self,
        input_json_path: str,
        sample_size: Optional[int] = None,
        save_all_tokens: bool = False,
        verbose: bool = True
    ):
        """
        抽出の実行と結果保存を一括で行う
        
        Args:
            input_json_path: 入力JSONファイルパス
            sample_size: 処理するサンプル数
            save_all_tokens: 全トークンのactivationを保存するか
            verbose: 詳細ログを表示するか
        """
        self.run_extraction(
            input_json_path=input_json_path,
            sample_size=sample_size,
            save_all_tokens=save_all_tokens,
            verbose=verbose
        )
        self.save_results()
        
        if verbose:
            print("\n🎉 Complete extraction finished!")
    
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
