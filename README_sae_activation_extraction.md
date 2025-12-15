# SAE Activation Extraction Tool

Teacher Forcingを利用して指定レイヤーのSAE activationを取得するツールです。

## 概要

このツールは、既存の分析結果（プロンプトと応答）に対してTeacher Forcingを適用し、指定したレイヤーのSAE activationを抽出します。抽出されたactivationは元のJSONファイルに統合されて保存されます。

## 主な機能

- **Teacher Forcing**: プロンプト+応答の完全なシーケンスを入力として、実験時と同じモデル状態を再現
- **任意レイヤー対応**: Layer番号とSAE IDを指定することで、任意のレイヤーのactivationを取得可能
- **柔軟な保存オプション**:
  - デフォルト: プロンプトの最後のトークン（応答直前）のactivationのみ保存
  - `--save-all-tokens`: 全トークンのactivationを保存
- **Top-K特徴**: メモリ効率のため、活性化値が高い上位K個の特徴のみを保存

## ファイル構成

- `sae_activation_extractor.py`: コアクラス（`SAEActivationExtractor`）の実装
- `run_sae_activation_extraction.py`: 実行スクリプト
- `README_sae_activation_extraction.md`: このドキュメント

## 使用方法

### 基本的な使い方

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --target-layer 20 \
  --sae-id layer_20/width_16k/canonical
```

### 主要なオプション

#### 必須パラメータ

- `--input`: 入力JSONファイルパス（分析結果を含むファイル）

#### モデル設定

- `--model`: モデル名（デフォルト: `google/gemma-2-9b-it`）
- `--sae-release`: SAEリリース名（デフォルト: `gemma-scope-9b-pt-res-canonical`）
- `--sae-id`: SAE ID（デフォルト: `layer_20/width_16k/canonical`）
- `--target-layer`: 対象レイヤー番号（デフォルト: 20）
- `--hook-name`: Hook名（デフォルト: 自動生成 `blocks.{target_layer}.hook_resid_post`）

#### 抽出設定

- `--save-all-tokens`: 全トークンのactivationを保存（デフォルトは最後のトークンのみ）
- `--top-k`: 保存する上位特徴数（デフォルト: 50）
- `--max-samples`: 処理する最大サンプル数（デフォルト: 全て）
- `--output`: 出力ファイルパス（デフォルト: タイムスタンプ付き自動生成）
- `--verbose`: 詳細な進捗を表示

### 使用例

#### 1. Layer 20のactivationを取得（デフォルト設定）

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json
```

#### 2. Layer 31のactivationを取得

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --target-layer 31 \
  --sae-id layer_31/width_16k/canonical
```

#### 3. 全トークンのactivationを保存（メモリ注意）

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --target-layer 20 \
  --save-all-tokens
```

#### 4. 少数サンプルでテスト実行

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --target-layer 20 \
  --max-samples 10 \
  --verbose
```

#### 5. Gemma-2-27B モデルでの実行

```bash
python run_sae_activation_extraction.py \
  --input results/labeled_data/combined_feedback_data.json \
  --model google/gemma-2-27b-it \
  --sae-release gemma-scope-27b-pt-res \
  --sae-id layer_20/width_16k/canonical \
  --target-layer 20
```

## 出力フォーマット

抽出されたactivationは、元のJSONファイルに以下の形式で追加されます：

```json
{
  "sae_extraction_metadata": {
    "layer_20": {
      "sae_id": "layer_20/width_16k/canonical",
      "sae_release": "gemma-scope-9b-pt-res-canonical",
      "hook_name": "blocks.20.hook_resid_post",
      "top_k_features": 50
    }
  },
  "results": [
    {
      "question_id": 123,
      "variations": [
        {
          "template_type": "base",
          "prompt": "...",
          "response": "...",
          "sae_activations": {
            "layer_20": {
              "activations": {
                "last_token": {
                  "top_k_features": [
                    {"id": 1234, "activation": 0.8765},
                    {"id": 5678, "activation": 0.5432}
                  ],
                  "num_active_features": 45
                }
              },
              "answer_start_position": 42,
              "total_tokens": 128,
              "status": "success"
            }
          }
        }
      ]
    }
  ]
}
```

### フィールドの説明

- `sae_extraction_metadata`: 抽出設定のメタデータ
- `sae_activations.layer_XX`: 指定レイヤーのactivation情報
  - `activations`: トークンごとのactivation
    - `last_token`: プロンプト最後のトークン（応答直前）
    - `token_0`, `token_1`, ...: 全トークン保存時の各位置
  - `top_k_features`: 上位K個の特徴
    - `id`: 特徴ID
    - `activation`: 活性化値
  - `num_active_features`: 非ゼロの特徴数
  - `answer_start_position`: 回答開始位置のインデックス
  - `total_tokens`: 全トークン数

## プログラムから使用する場合

```python
from sae_activation_extractor import (
    SAEActivationExtractor,
    ExtractionConfig,
    load_samples_from_json,
    save_results_to_json
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
extractor.load_model_and_sae()

# 単一サンプルの処理
result = extractor.extract_activations(
    prompt="Your question here",
    response="Model's response here",
    save_all_tokens=False
)

# バッチ処理
samples = load_samples_from_json("path/to/input.json")
results = extractor.extract_batch(samples, save_all_tokens=False)

# 結果の保存
save_results_to_json(
    original_json_path="path/to/input.json",
    extraction_results=results,
    output_json_path="path/to/output.json",
    config=config
)

# クリーンアップ
extractor.cleanup()
```

## 注意事項

### メモリ管理

- **GPU メモリ**: Gemma-2-9Bで約20GB、Gemma-2-27Bで約60GB必要
- **全トークン保存**: `--save-all-tokens` を使用すると、メモリ使用量が大幅に増加します
- **バッチサイズ**: 内部では1サンプルずつ処理され、20サンプルごとにメモリクリーンアップが実行されます

### 処理時間

- サンプル数と応答長に依存
- 目安: 100サンプルで5-10分程度（GPU環境）

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
# Top-K数を減らす
--top-k 20

# 処理サンプル数を制限
--max-samples 50

# 全トークン保存を無効化（デフォルトで無効）
# --save-all-tokens を指定しない
```

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

## 関連ファイル

- `feedback_analyzer.py`: Feedback迎合性分析（既存実験）
- `calc_attribution_patching.ipynb`: Attribution Patching分析
- `config.py`: 実験設定

## 技術詳細

### Teacher Forcingの仕組み

1. **入力**: `prompt + response` の完全なシーケンスをトークン化
2. **Forward Pass**: モデルに入力し、指定レイヤーでhookを実行
3. **SAE Encoding**: そのレイヤーのactivationをSAEでエンコードして特徴を抽出
4. **保存**: Top-K特徴のみを保存（メモリ効率化）

### 分析位置の選択

デフォルトでは**プロンプトの最後のトークン**（応答生成直前）のactivationを取得します。
これは以下の理由によります：

- モデルがプロンプト全体を処理し終えた状態
- 次に生成する応答の方針が最も明確に表現されている位置
- 迎合性などのタスク依存的な特徴が顕在化しやすい

## ライセンス

プロジェクトのライセンスに従います。
