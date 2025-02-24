# Autoencoder-for-Channel-Estimation
詳細は文献[^MyPaper]を参照．
[^MyPaper]: https://ken.ieice.org/ken/paper/20230125LCpY/

## シミュレーションパラメータの説明

### 学習・テスト 共通設定
- SIM_base.M (int) - 送信アンテナ本数
- SIM_base.N (int) - 受信アンテナ本数
- SIM_base.Q_str (int) - 1アンテナあたりの多値数．$2$または $4^n$ ($n$は任意の自然数) で指定．
- SIM_bse.Kp (int) - パイロット長
- SIM_base.channel_type (str) - 通信路の状態．現在"wireless"のみ指定可能．
  - "wireless" - 通信環境はレイリーフェージング環境となり，通信路行列の各要素は平均0の複素ガウス分布に従う．
- SIM_base.channel_change (int) - 通信路状態の変化の仕方を指定できる．現在は0のみ指定可能．
  - 0 - 通信路状態は同一データフレーム内では変化しない．
- SIM_base.channel_estimate (str) - 通信路推定方式 <br>
  - "pilot" - 通常のパイロット信号のみによる繰り返し無しの通信路推定推定
  - "successive" - 逐次通信路推定
  - "iterative" - 繰り返し通信路推定
SIM_base.repeat_max (int) - 繰り返し推定回数（繰り返し通信路推定時のみ有効）
SIM_base.nworker (int) - 並列ワーカー数（並列処理は計算機サーバーでの実行時のみ行う）

### 学習設定
#### 通信環境
- SIM_train.noise_pilot (bool) - Falseならパイロットシンボルに雑音が乗らず（カンニング），Trueなら雑音が乗る．
- SIM_train.Kd (int) - 学習時のデータフレーム長
- SIM_train.EsN0 (float) - 学習時の $E_\mathrm{s} / N_0 \ [\mathrm{dB}]$
- SIM_train.upd_th (int) - 通信路推定行列の更新閾値（逐次通信路推定時のみ有効）
#### 層構造
- SIM_train.hidden_depth (int) - 中間層（隠れ層）の深さ
- SIM_train.hidden_dim (int) - 中間層（隠れ層）の次元
- SIM_train.activation_func (function) - 中間層（隠れ層）の活性化関数
#### 学習条件
- SIM_train.InitialLearnRate (float) - 初期学習率
- SIM_train.epochs (int) - エポック数
- SIM_train.scheduler_factor (float) - 学習率スケジューラの減衰係数
- SIM_train.scheduler_start (int) - 学習率スケジューラの減衰開始エポック
- SIM_train.scheduler_step (int) - 学習率スケジューラの減衰間隔

### テスト設定
- SIM_test.noise_pilot (bool) - Falseならパイロットシンボルに雑音が乗らず（カンニング），Trueなら雑音が乗る．
- SIM_test.Kd (int) - テスト時のデータフレーム長
- SIM_test.EsN0Vec (list, ndarray) - テスト時の $E_\mathrm{s} / N_0 \ [\mathrm{dB}]$を並べた一次元配列
- SIM_test.upd_th (int) - 通信路推定行列の更新閾値（逐次通信路推定時のみ有効）
- SIM_test.nloop_max (int, float) - $E_\mathrm{s}/N_0$毎のシミュレーション回数 <br>
十分なエラー数が得られるまで計算をしたい場合はfloat('inf')に設定する[^NotInf]．
後述する，シンボルエラー数がSymError_maxに達するまで計算を続けたい場合はfloat('inf')に設定する[^NotInf]．
- SIM_test.SymError_max (int, float) - 早期終了条件 <br>
シンボルエラー数がこの設定値に達すると，その $E_\mathrm{s} / N_0$でのシミュレーションを早期終了する．早期終了しない場合はfloat('inf')に設定する[^NotInf]．
[^NotInf]: nloop_maxとSymError_maxを両方ともfloat('inf')に設定すると計算が終わらないので，片方は有限値に設定すること．