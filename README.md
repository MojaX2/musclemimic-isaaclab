# MuscleMimic Isaac Lab Environment

MuscleMimicの全身354筋モデルを **Isaac Lab＋Newton＋MuJoCo Warp** で動かす実験環境です。このリポジトリから、新規PPO学習・継続学習・同梱方策の推論を実行できます。別のMuscleMimic本体へのコードコピーやAMASSの生データ準備は、以下の手順には不要です。

![選定した方策の歩行・10秒抜粋](artifacts/best-policy/walking.gif)

外力補助なしの交互歩行は得られましたが、低速指令では速く歩きすぎます。汎用的な速度追従の完成や、原論文の性能再現を主張しません。標準のGym登録済み`ManagerBasedRLEnv`ではなく、Isaac LabのSimulationContextと独自環境クラスを使います。

## 1. 実行環境の準備

確認対象は **Linux x86_64、Python 3.12、NVIDIA GPU** です。4096環境はRTX 5090・32 GBで検証しました。VRAMが少ない場合は`--num-envs 4`から開始してください。CPU実行・Windows・別バージョンのIsaac Labは未検証です。Isaac Sim / AppLauncherは不要です。

Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、動作するNVIDIAドライバを準備し、空の作業ディレクトリで実行します。このリポジトリはprivateなのでGitHub認証が必要です。

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout be270ed88bbc0ea46201ce5cdd4cd0a502971cbc
uv sync --project IsaacLab --python 3.12

git clone https://github.com/MojaX2/musclemimic-isaaclab.git
cd musclemimic-isaaclab
export ISAACLAB_PYTHON="$(cd ../IsaacLab && pwd)/.venv/bin/python"
uv pip install --python "$ISAACLAB_PYTHON" -r requirements-runtime.txt

"$ISAACLAB_PYTHON" scripts/run.py doctor
```

`doctor`はGPU、主要依存バージョン、モデル、描画依存を確認します。モデルは`musclemimic_models==1.0.6`から自動検出します。自分で用意したモデルアセットを使う場合だけ`--model-assets /path/to/musclemimic_models/model`を指定します。

**バージョンを固定してください。** Isaac Labは3.0早期アクセスの上記コミット、Newton 1.6.0、MuJoCo / MuJoCo Warp 3.12.0、Warp 1.17.0、RSL-RL 5.5.1です。MuscleMimic本体のJAX用requirementsをこの環境にインストールするとMuJoCo等が別バージョンになるため、ここでは使いません。既存Isaac Lab環境を利用する場合は`ISAACLAB_PYTHON`をそのPythonに設定して、追加requirementsとdoctorから始めてください。

## 2. 保存済み方策で推論する

まず少数環境で確認します：

```bash
"$ISAACLAB_PYTHON" scripts/run.py evaluate \
  --num-envs 4 --eval-seconds 1 --run-dir outputs/eval_smoke
```

20秒・4096環境で、保存済みの評価と同じ条件で再評価します：

```bash
"$ISAACLAB_PYTHON" scripts/run.py evaluate \
  --num-envs 4096 --eval-seconds 20 --run-dir outputs/eval_best
```

`artifacts/best-policy/checkpoint.pt`を既定で読み込みます。指令0 / 0.4 / 0.8 / 1.2 m/sを各1024環境へ割り当て、直立開始・ランダム位相・初期速度摂動0.02・確率的方策・外力ゼロで評価します。別の重みは`--checkpoint /path/to/checkpoint.pt`で指定します。同じネットワーク構成の本実装チェックポイント用であり、原論文のJAXチェックポイントとは互換ではありません。

生成物は`outputs/eval_best/evaluations/evaluation/`の`metrics.json`、`rollout.npz`、`policy.mp4`、`foot_motion.json`です。動画は代表1環境（0.4 m/s指令）の記録済み状態をMuJoCoで描画します。

## 3. 新規学習する

最初に、4環境・2更新で学習経路を確認します：

```bash
"$ISAACLAB_PYTHON" scripts/run.py train \
  --num-envs 4 --iterations 2 --eval-seconds 1 --run-dir outputs/train_smoke
```

ゼロから歩行を学習するbootstrap設定：

```bash
"$ISAACLAB_PYTHON" scripts/run.py train \
  --preset bootstrap --num-envs 4096 --iterations 2000 \
  --run-dir outputs/train_bootstrap
```

`--resume`を省略すると**新規重み・新規optimizer**から開始します。bootstrapはrun_021で交互歩行を獲得した設定に基づき、1 m/s固定指令、参照リセット80%、正のCartesian報酬17.5、体幹高さ報酬5、位相4調波、10秒エピソード、外力ゼロです。必要な小規模の参照配列を同梱しています。[参照データの出典・制限](artifacts/reference/README.md)を確認してください。

次に速度範囲を広げる場合：

```bash
"$ISAACLAB_PYTHON" scripts/run.py train \
  --preset velocity --resume outputs/train_bootstrap/latest.pt \
  --num-envs 4096 --iterations 1000 --run-dir outputs/train_velocity
```

velocityは0.4–1.2 m/s、停止指令あり、参照リセット50%、Cartesian報酬8.75、20秒エピソードです。既存のよい歩容から継続するための設定であり、段階を省略して必ず成功する保証はありません。

## 4. 同梱した重みから継続学習する

```bash
"$ISAACLAB_PYTHON" scripts/run.py train \
  --preset velocity --resume artifacts/best-policy/checkpoint.pt \
  --num-envs 4096 --iterations 1000 --run-dir outputs/train_from_best
```

`--iterations`は**追加更新数**です。重み・optimizer・正規化を復元します。終了済み実験の自動再開は行わず、このコマンドを実行した場合だけ学習が始まります。

## 出力・評価・停止

- 各実行には新しい`--run-dir`を使ってください。既存の結果があるディレクトリへの上書きは拒否します。
- `latest.pt`、評価時の`model_*.pt`、`metrics.jsonl`、`status.json`、`tensorboard/`、`evaluations/`を保存します。
- 最初の評価は300秒後、以降は評価完了から約1時間ごと。`--first-eval-seconds`、`--eval-interval`、`--video-interval`で変更できます。学習終了時も評価します。
- `touch outputs/train_bootstrap/EVALUATE`で追加評価・動画を要求できます。
- `touch outputs/train_bootstrap/STOP`で更新の区切りに停止し、`stopped.pt`を保存します。評価中はその処理を終えてから停止します。
- 描画不要なら`--no-render`を付けます。評価JSONと軌道記録は保存します。
- 初回はCUDAカーネルのコンパイルで時間がかかります。EGL描画が使えないマシンでは`--no-render`を使い、GPUドライバのEGL環境を確認してください。

## 検証の範囲

配布ファイルのみを別ディレクトリにコピーし、PyPIから取得したモデルを使って、既存の固定バージョンIsaac Lab環境上で検証します。結果は[ポータビリティ検証](validation/portable/README.md)に記録しています。OS・ドライバを含む新規マシンへのインストールや、別GPUでの学習収束を検証したものではありません。

- [保存した方策・選定基準・評価結果](artifacts/best-policy/README.md)
- [終了時の実験結果](RESULTS.md)
- [Isaac Lab実装の詳細と実験履歴](docs/EXPERIMENT_OVERVIEW.md)
- [mjlabのモデル読み込み・ランダム駆動・速度測定](validation/mjlab/README.md)

mjlabは別の検証先です。上記学習・推論コマンドはIsaac Lab＋Newtonを使い、mjlabでのPPO学習を主張するものではありません。

## ライセンス

コードの[Apache-2.0ライセンス](LICENSE)と元プロジェクトの帰属を保持しています。モデル・参照モーション・依存ソフトウェアのライセンスはそれぞれに従います。原著者による公式Isaac Lab移植ではありません。
