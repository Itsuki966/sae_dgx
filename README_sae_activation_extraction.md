# SAE Activation Extraction Tool

Teacher Forcingを利用して指定レイヤーのSAE activationを取得するツールです。

## 概要

このツールは、既存の分析結果（プロンプトと応答）に対してTeacher Forcingを適用し、指定したレイヤーのSAE activationを抽出します。**`feedback_analyzer.py`と完全に同じ出力形式**で結果を保存します。

## 主な機能

- **Teacher Forcing**: プロンプト+応答の完全なシーケンスを入力として、実験時と同じモデル状態を再現
- **任意レイヤー対応**: Layer番号とSAE IDを指定することで、任意のレイヤーのactivationを取得可能
- **feedback_analyzer.py互換**: 出力形式、データ構造、ファイル命名規則が完全に一致
- **柔軟な保存オプション**:
  - デフォルト: プロンプトの最後のトークン（応答直前）のactivationのみ保存
  - `--save-all-tokens`: 全トークンのactivationを保存
- **疎ベクトル形式**: 活性化値が0より大きい全特徴を保存（ML学習に最適）

## ファイル構成

- `sae_activation_extractor.py`: コアクラス（`SAEActivationExtractor`）の実装
- `run_sae_activation_extraction.py`: 実行スクリプト
- `README_sae_activation_extraction.md`: このドキュメント

## 使用方法

### 基本的な使い方

```bash
# Layer 20を使用（デフォルト）
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json

# Layer 9を使用
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --layer 9
```

### 主要なオプション

#### 必須パラメータ

- `--input`: 入力JSONファイルパス（分析結果を含むファイル）

#### Layer設定（推奨）

- `--layer`: 対象レイヤー番号（9または20、デフォルト: 20）
  - **自動設定**: `--layer`を指定するだけで、`sae-id`と`hook-name`が自動的に設定されます

#### モデル設定（高度な使用）

- `--model`: モデル名（デフォルト: `google/gemma-2-9b-it`）
- `--sae-release`: SAEリリース名（デフォルト: `gemma-scope-9b-pt-res-canonical`）
- `--sae-id`: SAE ID（オプショナル、`--layer`使用時は自動設定）
- `--hook-name`: Hook名（オプショナル、`--layer`使用時は自動設定）
- `--target-layer`: [非推奨] `--layer`を使用してください

#### 抽出設定

- `--save-all-tokens`: 全トークンのactivationを保存（デフォルトは最後のトークンのみ）
- `--top-k`: Top-K特徴数（デフォルト: 50）※ログ・可視化用、全活性化は別途保存
- `--max-samples`: 処理する最大サンプル数（デフォルト: 全て）
- `--verbose`: 詳細な進捗を表示

**注意**: `--output`オプションは削除されました。ファイル名は自動生成されます（下記「出力ファイル名」参照）

### Layer番号とSAE設定の対応

| Layer | SAE ID | Hook Name | SAE幅 |
|-------|--------|-----------|-------|
| 9 | `layer_9/width_131k/canonical` | `blocks.9.hook_resid_post` | 131k |
| 20 | `layer_20/width_131k/canonical` | `blocks.20.hook_resid_post` | 131k |

**`--layer`を使用すると、上記の設定が自動的に適用されます。**

### 使用例

#### 1. Layer 20のactivationを取得（デフォルト設定）

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json
```

#### 2. Layer 9のactivationを取得

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --layer 9
```

#### 3. 全トークンのactivationを保存（メモリ注意）

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --layer 20 \
  --save-all-tokens
```

#### 4. 少数サンプルでテスト実行

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --layer 9 \
  --max-samples 10 \
  --verbose
```

#### 5. 高度な使用：手動でSAE設定を指定

```bash
# カスタムSAE IDとhook nameを指定する場合
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --layer 20 \
  --sae-id layer_20/width_16k/canonical \
  --hook-name blocks.20.hook_resid_post
  --target-layer 20
```

## 出力ファイル名

ファイル名は以下の形式で自動生成されます：

```
{model}_layer{XX}_{position}_{timestamp}.json
```

- `{model}`: モデル名（スラッシュとハイフンをアンダースコアに置換）
- `{XX}`: レイヤー番号（例: 20, 31）
- `{position}`: トークン位置（`last_token`または`all_tokens`）
- `{timestamp}`: タイムスタンプ（YYYYMMDD_HHMMSS形式）

**例**:
- `google_gemma_2_9b_it_layer20_last_token_20251215_143022.json`
- `google_gemma_2_9b_it_layer31_all_tokens_20251215_150530.json`

## 出力フォーマット

`feedback_analyzer.py`と完全に同じ形式で保存されます：

```json
{
  "metadata": {
    "model_name": "google/gemma-2-9b-it",
    "sae_release": "gemma-scope-9b-pt-res-canonical",
    "sae_id": "layer_20/width_16k/canonical",
    "target_layer": 20,
    "hook_name": "blocks.20.hook_resid_post",
    "num_questions": 100,
    "save_all_tokens": false,
    "analyzed_position": "prompt_last_token",
    "timestamp": "2025-12-15T14:30:22.123456",
    "config": {
      "top_k_features": 50,
      "device": "cuda",
      "dtype": "torch.bfloat16"
    }
  },
  "results": [
    {
      "question_id": 0,
      "dataset": "arguments",
      "base_text": "...",
      "variations": [
        {
          "template_type": "base",
          "prompt": "...",
          "response": "...",
          "sae_activations": {
            "prompt_last_token": {
              "1234": 0.8765,
              "5678": 0.5432,
              "9012": 0.3210
            }
          },
          "top_k_features": [
            [1234, 0.8765],
            [5678, 0.5432],
            [9012, 0.3210]
          ],
          "metadata": {
            "extraction_time_ms": 123.45,
            "response_length": 256,
            "timestamp": "2025-12-15T14:30:25.123456",
            "gpu_memory_mb": 15234.56
          }
        }
      ]
    }
  ]
}
```

### フィールドの説明

- **`metadata`**: 実験全体のメタデータ
  - `model_name`: 使用したモデル名
  - `sae_release`, `sae_id`: SAEの情報
  - `target_layer`: 対象レイヤー番号
  - `num_questions`: 処理した質問数
  - `save_all_tokens`: 全トークン保存の有無
  - `analyzed_position`: 分析位置（`prompt_last_token`または`all_prompt_tokens`）

- **`results`**: 各質問の結果リスト
  - `question_id`: 質問ID
  - `dataset`: データセット名（arguments, math, poems等）
  - `base_text`: ベーステキスト
  - **`variations`**: プロンプトバリエーションのリスト
    - `template_type`: テンプレートタイプ（base, positive, negative等）
    - `prompt`: プロンプト文字列
    - `response`: 応答文字列
    - **`sae_activations`**: SAE活性化（**疎ベクトル形式**）
      - `prompt_last_token`: プロンプト最後のトークンの活性化
        - キー: 特徴ID（文字列）
        - 値: 活性化値（0より大きい値のみ保存）
      - `token_0`, `token_1`, ...: 全トークン保存時の各位置
    - `top_k_features`: 可視化用のTop-K特徴（`[[id, value], ...]`形式）
    - `metadata`: 個別の処理メタデータ

## プログラムから使用する場合

```python
from sae_activation_extractor import (
    SAEActivationExtractor,
    ExtractionConfig
)

# 設定の作成
config = ExtractionConfig(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-pt-res-canonical",
    sae_id="layer_20/width_16k/canonical",
    target_layer=20,
    hook_name="blocks.20.hook_resid_post",
    top_k_features=50
)

# Extractorの初期化
extractor = SAEActivationExtractor(config)

# 完全な抽出実行（読み込み + 分析 + 保存）
extractor.run_complete_extraction(
    input_json_path="results/labeled_data/combined_feedback_data.json",
    sample_size=None,  # 全て処理
    save_all_tokens=False,  # プロンプト最後のトークンのみ
    verbose=True
)

# クリーンアップ
extractor.cleanup()
```

### 手動での段階的実行

```python
# 初期化
extractor = SAEActivationExtractor(config)

# モデルとSAEをロード
extractor.load_model_and_sae()

# 抽出実行
extractor.run_extraction(
    input_json_path="results/labeled_data/combined_feedback_data.json",
    sample_size=10,  # テスト用に10問のみ
    save_all_tokens=False,
    verbose=True
)

# 結果を保存
extractor.save_results()  # ファイル名は自動生成

# クリーンアップ
extractor.cleanup()
```

## 注意事項

### メモリ管理

- **GPU メモリ**: Gemma-2-9Bで約20GB、Gemma-2-27Bで約60GB必要
- **全トークン保存**: `--save-all-tokens` を使用すると、メモリ使用量が大幅に増加します
- **バッチサイズ**: 内部では1サンプルずつ処理され、20サンプルごとにメモリクリーンアップが実行されます

### データ形式

- **疎ベクトル形式**: 活性化値が0より大きい特徴のみを保存
- **ML学習に最適**: XGBoostなどの機械学習モデルでの特徴選択に適した形式
- **メモリ効率**: 通常、全16384特徴のうち数百〜数千の特徴のみが活性化するため、効率的

### 処理時間

- サンプル数と応答長に依存
- 目安: 100問題（500バリエーション）で10-20分程度（GPU環境）

### SAE ID とレイヤーの対応

SAE IDは必ずターゲットレイヤーと一致させてください：

| Target Layer | SAE ID | Hook Name |
|--------------|--------|-----------|
| 20 | `layer_20/width_16k/canonical` | `blocks.20.hook_resid_post` |
| 31 | `layer_31/width_16k/canonical` | `blocks.31.hook_resid_post` |
| 10 | `layer_10/width_16k/canonical` | `blocks.10.hook_resid_post` |

## トラブルシューティング

### OOM (Out of Memory) エラー

```bash
# 処理サンプル数を制限
--max-samples 50

# 全トークン保存を無効化（デフォルトで無効）
# --save-all-tokens を指定しない
```

**注意**: `--top-k`はメモリ使用量に影響しません（全活性化は別途疎ベクトル形式で保存されます）

### モデルロードエラー

- HuggingFace トークンが設定されているか確認
- インターネット接続を確認
- モデル名が正しいか確認

### 入力ファイルが見つからない

```bash
# ファイルパスを確認
ls -la results/labeled_data/

# 絶対パスで指定
--input /full/path/to/file.json
```

### SAE.from_pretrained のアンパック警告

`sae, _, _ = SAE.from_pretrained(...)` のような記述は不要です。最新版では：

```python
self.sae = SAE.from_pretrained(
    release=self.config.sae_release,
    sae_id=self.config.sae_id,
    device=self.config.device
)
```

## feedback_analyzer.pyとの比較

| 項目 | feedback_analyzer.py | sae_activation_extractor.py |
|------|---------------------|----------------------------|
| **用途** | モデルで応答を生成しながらSAE抽出 | 既存の応答からSAE抽出 |
| **入力** | プロンプトのみ | プロンプト + 応答 |
| **処理方法** | 生成時にリアルタイムで取得 | Teacher Forcingで再現 |
| **出力形式** | ✅ 完全に同じ | ✅ 完全に同じ |
| **データ構造** | ✅ 完全に同じ | ✅ 完全に同じ |
| **ファイル命名** | ✅ 同じ規則 | ✅ 同じ規則 |
| **疎ベクトル** | ✅ 同じ形式 | ✅ 同じ形式 |

**使い分け**:
- **新規分析**: `feedback_analyzer.py`（応答生成 + SAE抽出）
- **追加レイヤー**: `sae_activation_extractor.py`（既存応答で別レイヤー抽出）

## 関連ファイル

- `feedback_analyzer.py`: Feedback迎合性分析（応答生成 + SAE抽出）
- `calc_attribution_patching.ipynb`: Attribution Patching分析
- `config.py`: 実験設定

## 技術詳細

### Teacher Forcingの仕組み

1. **入力**: `prompt + response` の完全なシーケンスをトークン化
2. **Forward Pass**: モデルに入力し、指定レイヤーでactivationを取得
3. **SAE Encoding**: そのレイヤーのactivationをSAEでエンコードして特徴を抽出
4. **疎ベクトル化**: 活性化値が0より大きい特徴のみを保存（ML学習用）
5. **Top-K抽出**: ログ・可視化用にTop-K特徴も別途保存

### 分析位置の選択

デフォルトでは**プロンプトの最後のトークン**（応答生成直前）のactivationを取得します。
これは以下の理由によります：

- モデルがプロンプト全体を処理し終えた状態
- 次に生成する応答の方針が最も明確に表現されている位置
- 迎合性などのタスク依存的な特徴が顕在化しやすい

### データ形式の詳細

**疎ベクトル形式**（`sae_activations`）:
```json
{
  "prompt_last_token": {
    "1234": 0.8765,  // 特徴ID: 活性化値
    "5678": 0.5432,
    "9012": 0.3210
  }
}
```

**Top-K形式**（`top_k_features`、可視化用）:
```json
[
  [1234, 0.8765],  // [特徴ID, 活性化値]
  [5678, 0.5432],
  [9012, 0.3210]
]
```

## ライセンス

プロジェクトのライセンスに従います。
