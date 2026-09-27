# Selected checkpoint: run_025 / 5716

![Selected policy, 10-second excerpt](walking.gif)

GIF: 保存済み20秒評価の先頭10秒、240×270、8 fps、約1.3 MB。Isaac Lab + Newton上の状態をMuJoCoで描画。目標0.4 m/sに対して元の20秒動画の平均速度は0.673 m/s、リセット0回です。低速追従の誤差は残っています。

## 選定基準

保存済みの外力ゼロ評価から、0 / 0.4 / 0.8 / 1.2 m/sの**全指令で20秒生存率97%以上**を満たす方策を抽出し、4指令の速度MAEの単純平均が最小のものを選びました。この事後的な基準ではrun_025の5716更新が選ばれます。未使用の検証データ・複数seedによる最良性の証明ではありません。

| 指令 m/s | 20秒生存率 | MAE m/s |
|---:|---:|---:|
| 0.0 | 97.66% | 0.0595 |
| 0.4 | 99.90% | 0.2770 |
| 0.8 | 100% | 0.1396 |
| 1.2 | 99.51% | 0.1622 |

平均MAE 0.15954 m/s。最終run_026/6716の平均MAE 0.18693 m/sより小さいため、最終重みではなくこちらを収録しています。

## 同梱物

- `checkpoint.pt`: 元の約17 MBのチェックポイントそのまま。actor・critic・観測正規化・optimizer等を含む。量子化・精度削減なし。
- `model.xml.template`, `initial.npz`: 学習と同じ354筋モデル設定・初期状態。mesh/texture本体は元の`musclemimic_models`から別途取得。
- `evaluate.py`: モデルのパスを解決し、学習せずにGPU評価する入口。
- `metrics.json`, `training_config.json`, `selection.json`: 原評価、原学習設定、選定方法と重みのハッシュ。
- `rollout.npz`, `foot_motion.json`: 保存済み代表20秒の状態記録と歩容集計。
- `walking.gif`: 同じ重みの動画から作った軽量プレビュー。

## 再評価

[ルートREADME](../../README.md)のインストール手順を実行し、リポジトリのルートで次を実行します。

```bash
"$ISAACLAB_PYTHON" scripts/run.py evaluate --num-envs 4096 --run-dir outputs/eval_best
```

モデルはインストール済み`musclemimic_models`から自動検出されます。生成物は`outputs/eval_best/evaluations/evaluation/`の評価JSON、軌道、MP4です。描画不要なら`--no-render`を付けます。

`evaluate.py`も共通入口へ転送する互換スクリプトとして残しています。AMASSの生データと学習用参照バンクは推論には不要です。学習用の小規模な派生配列は別途`artifacts/reference/`に収録しています。

`smoke_validation.json`は以前の4環境・1秒テストです。最新の配布用経路の学習・推論・描画テストは[ポータビリティ検証](../../validation/portable/README.md)を参照してください。短時間・少数環境での動作確認は4096環境の歩行性能再現を意味しません。
