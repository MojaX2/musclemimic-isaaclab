> Historical experiment snapshot. For supported setup and runnable commands, use the [current README](../README.md). Path and packaging limitations below describe the earlier archive.

# MuscleMimic Isaac Lab Environment

MuscleMimicの全身筋骨格モデルを **Isaac LabのNewtonバックエンド＋MuJoCo Warp** で動かし、GPU並列PPOによる歩行学習を試した実験コードです。

**到達点：外力補助なしで20秒間の交互歩行が得られました。ただし低速指令への追従には誤差が残り、汎用的な速度追従歩行の完成には至っていません。** 原論文の学習・性能の再現を主張するものではありません。

## リポジトリの位置づけ

Isaac Lab向け筋骨格歩行環境を中心に、[mjlab検証](../validation/mjlab/README.md)を比較・検証先として収録しています。現在の環境クラスは学習スクリプト内にあり、Gym登録や標準Isaac Lab環境APIへの分離は未実装です。実験コードの保存と、標準環境としての完成を区別します。

## 選定した重みとデモ

[選定したチェックポイント・再評価手順](../artifacts/best-policy/README.md)を同梱しています。

![外力なしの歩行・10秒抜粋](../artifacts/best-policy/walking.gif)

## 実装したもの

- 外部アダプターによるMJCF読み込み。空間腱、MuJoCoネイティブの筋アクチュエータ、51個の関節等式制約を保持。
- Newton変換時の筋長範囲、接触margin/gap、センサー、関節順序・観測タイミングの補正。
- Isaac Lab本体やインストール済みNewtonのソースを変更せず、独自manager / solverを差し込む構成。
- 354筋の興奮度を出力するPPO学習。4096環境、2 ms物理刻み、20 ms制御周期。
- G1 Flatを参考にした速度・姿勢・足滑り・片足支持報酬に、遊脚高さ、交互着地、体幹高さ、周期的な参照部位位置の報酬を追加。
- 参照姿勢への等式制約投影、参照状態からのリセット、周期位相の方策分岐、外力補助の比較実験。
- 定期評価、チェックポイント、動画、転倒前の歩数・交互着地・外力ゼロの検証。

構成は `Isaac Lab SimulationContext → custom NewtonMJWarpManager → custom SolverMuJoCo → MuJoCo Warp` です。PhysXへの移植ではなく、stockのManagerBasedRLEnv / Articulationとして完成した実装でもありません。

## 確認環境

| 項目 | 実験環境 |
|---|---|
| Isaac Lab | v3.0.0-EA-26-gbe270ed88（早期アクセス版） |
| Newton | 1.6.0 |
| MuJoCo / MuJoCo Warp | 3.12.0 |
| Warp | 1.17.0 |
| PyTorch / RSL-RL | 2.12 cu130 / 5.5.1 |
| GPU | NVIDIA RTX 5090、32 GB |
| MuscleMimicベース | `5d4c8af12784f44bd2bb276e71db9cda6c9712fc` |

原実装・CPU描画側には別のPython 3.11 / MuJoCo 3.4環境を使用しました。内部APIに依存するため、上記以外のバージョンでの動作は未検証です。

## コードの入口

`examples/isaaclab_newton_fullbody/` 以下に実装があります。

| ファイル | 役割 |
|---|---|
| `newton_fullbody_adapter.py` | モデル変換、筋・等式制約・接触の補正 |
| `check_isaaclab_fullbody.py` | GPU上の物理実行のスモークテスト |
| `train_velocity_g1.py` | G1系報酬の基本学習 |
| `train_velocity_curriculum.py` | 最終実験で使った参照報酬・初期化・位相分岐付き学習 |
| `g1_muscle_rewards.py` | 基本報酬と歩容報酬 |
| `cartesian_reference_reward.py` | 骨盤相対の部位位置に対する周期参照報酬 |
| `project_joint_equalities.py` | 参照状態を関節等式制約へ投影 |
| `phase_residual_actor.py` | 周期位相から筋出力平均への分岐 |
| `evaluate_pretrained.py` | 元の事前学習方策の転送試験 |
| `record_direct_viewer.py` | Newtonビューワーからの直接録画 |
| `render_velocity.py` | 実行済み状態をMuJoCoで描画 |

既存の[詳細実験ノート](../examples/isaaclab_newton_fullbody/README.md)と[時系列メモ](../examples/isaaclab_newton_fullbody/EXPERIMENT_STATUS.md)も保存しています。古い段階の制限・結果を含むため、現在の到達点はこのREADMEとresultsを参照してください。

## 再実行の前提

このリポジトリは追加実装のスナップショットです。**単独cloneだけで学習が始まるパッケージではありません。** 元の[MuscleMimic](https://github.com/amathislab/musclemimic)とモデル、Isaac LabのNewton環境が必要です。

1. MuscleMimicを上記コミットで別途用意し、本リポジトリの`examples/isaaclab_newton_fullbody/`を同じ相対位置へコピーします。
2. 元の環境でモデルと初期状態を準備します。学習コードは`outputs/isaac_velocity/assets/model.xml`と`initial.npz`を読みます。モデルの参照meshも必要です。
3. 参照付き実験にはAMASS由来の歩行データと、`outputs/isaac_velocity_g1/reference_candidates/`の投影済みリセット・周期ターゲットが必要です。生成スクリプトは`prepare_*`にあります。元データの利用権は別途必要で、このリポジトリには同梱していません。
4. スクリプト内のローカル絶対パス（Isaac Lab環境、キャッシュ、ffmpegなど）を自分の環境に合わせます。原実験のパスを保存しており、完全なポータブル化はしていません。
5. 小規模な物理・学習テストを実行してから4096環境へ増やします。

独立リポジトリに全依存をvendor化するのではなく、元プロジェクトに追加する形で保存しています。選定した重みと再評価用の小規模データは`artifacts/best-policy/`に収録しています。モデルのmesh・texture本体と学習用の元データは別途取得してください。

## 最後の実験条件

run_026はrun_025の`model_5716.pt`から再開し、訓練エピソード長だけを10秒から20秒へ変更しました。run_021が新規初期化で、その後run_022→026へ継続しています。原論文の事前学習方策からの再開ではありません。

- 外力補助0、全354筋を学習、4096環境。
- 速度指令0.4–1.2 m/s、standing probability 0.3、参照リセット確率0.5。
- 参照リセットが指令を上書きするため、全リセット中の停止指令は約15%です。
- 周期位相4調波、位相周波数0.892857 Hz、正のCartesian参照報酬重み8.75、体幹高さ報酬5、関節角参照報酬0。
- solver iterations 20、line-search 8、PPO desired KL 0.2、学習可能な探索標準偏差。
- 最初の評価は300秒後、その後は評価完了から3600秒ごと。20秒間、直立から開始し位相をランダム化、初期速度に0.02の摂動。
- `--iterations`は再開後の追加更新数です。run_026は1000更新、終了更新番号6716。

保存した設定と評価は[results](../results/)を参照してください。設定JSONに含まれるローカルパスは原実験の記録です。

## 結果の読み方と限界

評価の`survival`は**リセット前の最初のエピソードで20秒間転倒しなかった割合**です。速度MAE、歩数、外力ゼロと併せて判断します。立ち続けるだけでもsurvivalは高くなります。動画の一本だけで集団全体の性能を判断しません。

run_026の初回評価（5792更新）では、停止/0.4/0.8/1.2 m/sの20秒生存率は98.4/99.6/99.7/99.4%、速度MAEは0.055/0.300/0.158/0.155 m/sでした。0.4 m/s指令の代表動画は20秒リセットなしですが平均速度0.716 m/sで、低速追従は未達です。終了時の評価は[RESULTS.md](../RESULTS.md)を参照してください。

- 最終評価は確率的方策です。決定論的方策と同じ性能を保証しません。
- 指令は訓練エピソード内で固定。指令切替・長時間・複数seedでの頑健性は未検証です。
- 参照歩容と位相周波数は速度に応じて変化しません。速度追従との競合が残る可能性があります。
- 参照報酬や初期化を含むため、純粋な速度報酬だけで歩行を獲得した実験ではありません。
- 学習動画はIsaac Lab＋Newton上で生成した状態をMuJoCoで再描画しています。別の動力学で歩き直したものではありません。直接ビューワー録画とは区別しています。
- 初期スモーク試験の416筋モデルと、指を無効化した学習用354筋モデルは別設定です。

## ライセンス・帰属

ベースとなるMuscleMimicのApache-2.0ライセンスを[LICENSE](../LICENSE)に保持しています。これは元のモデル、AMASSなどのデータ、Isaac Lab・Newton・各依存物のライセンスを置き換えるものではありません。原プロジェクトの著者による公式Isaac Lab移植ではありません。
