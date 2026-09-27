# mjlab検証

Isaac Lab環境の移植に先立つ、別バックエンドでのモデル読み込み・ランダム駆動・GPU並列実行の検証です。mjlab上で今回の歩行PPOを学習した結果ではありません。

RTX 5090、mjlab 1.6.0、MuJoCo / MuJoCo Warp 3.11.0、416筋・424腱の全身モデル。2 ms物理刻み、decimation 5、implicitfast、solver iterations 50、line-search 20。各試行は各環境1秒分の物理実行です。

| 並列環境数 | physics env-steps/s | control env-steps/s |
|---|---:|---:|
| 4096 | 131,645 | 26,329 |
| 8192 | 139,667 | 27,933 |

計測範囲は物理演算、ランダム興奮度の補間・設定、GPU上のoverflow/contact/constraint監視です。初期化・reset・検証・方策・報酬・描画・PPO更新は含みません。Isaac Lab学習との設定が異なるため、これを直接の速度比較として扱わないでください。JSONのsolver名NewtonはMuJoCoの制約ソルバー名で、Newton物理フレームワーク経由という意味ではありません。

スクリプト内のモデル・一時ディレクトリの絶対パスは原実験の記録です。再実行時は環境に合わせて変更してください。結果JSONは保存済み測定値であり、配布用コピーを別環境で再実行したものではありません。
