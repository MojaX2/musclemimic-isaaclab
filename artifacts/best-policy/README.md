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

ルートREADMEに記載したIsaac Lab / Newton / RSL-RL環境と、元のMuscleMimicモデルパッケージが必要です。元データやmeshのライセンスはそれぞれに従ってください。モデル本体のディレクトリには`meshes/`と`scene/`が含まれます。

このリポジトリのルートから、Isaac Lab環境のPythonで実行します：

```bash
/path/to/IsaacLab/.venv/bin/python artifacts/best-policy/evaluate.py \
  --model-assets /path/to/site-packages/musclemimic_models/model \
  --num-envs 4096 \
  --run-dir outputs/best_policy_eval
```

`outputs/isaac_velocity/assets/`にテンプレートからモデルと初期値を準備します。異なる既存ファイルは上書きしません。評価は原条件と同じ直立開始・ランダム位相・速度摂動0.02・20秒・確率的方策・外力ゼロです。4指令は環境ごとに割り当てられます。4096環境では各指令1024環境です。

参照リセット・参照報酬はこの推論に不要なので、AMASSの生データや学習用参照バンクは同梱していません。これは評価の再実行用であり、元学習全体の再現用データ一式ではありません。動画描画は省略し、指標と状態記録を保存します。別途描画する場合は既存の`render_velocity.py`を使い、描画用Pythonとffmpegのパスを環境に合わせてください。

小規模な動作確認：

```bash
/path/to/IsaacLab/.venv/bin/python artifacts/best-policy/evaluate.py \
  --model-assets /path/to/site-packages/musclemimic_models/model \
  --num-envs 4 --eval-seconds 1 --run-dir outputs/best_policy_smoke
```

この短時間テストは読み込み・推論・GPU動作の確認で、歩行性能の再検証には4096環境・20秒の評価を使ってください。
