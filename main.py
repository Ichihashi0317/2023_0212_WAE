# %% [markdown]
# ## モジュールをインポート

# %%
import math
import copy
from multiprocessing import Pool

import numpy as np
import torch
import matplotlib.pyplot as plt

# %%
import os
if os.getcwd() == "/content":   # 作業ディレクトリがGoogle Driveなら実行
  from google.colab import drive
  drive.mount("/content/drive")   # Google Driveをマウント
  os.chdir("/content/drive/MyDrive/Colab Notebooks/8_autoencoder_wireless_parallel")   #作業ディレクトリを移動
import module_autoencoder_ver2_4_ as ae

# %% [markdown]
# ## 定数の設定

# %%
class Constant():
    pass


# if (__name__ == '__main__'):


### 正規化規則
# 0:QAM 1:Energy 2:Average power 3:Energy-stream 4:Average power-stream
normalize_name = ("QAM", "Energy", "Average power", "Energy-stream", "Average power-stream", "Energy-stream(same mapping)", "Average power-stream (same mapping)")
normalize_list = [0, 1, 2, 3, 4]


### 共通パラメータ
SIM_base = Constant()
# 送受信アンテナ本数
SIM_base.M = 2
SIM_base.N = 2
# 変調多値数
SIM_base.Q_str = 4 # 1アンテナあたりの多値数
SIM_base.Q = SIM_base.Q_str ** SIM_base.M # アンテナ全体での多値数
SIM_base.q = int(math.log2(SIM_base.Q)) # 1シンボルあたりのビット数
# パイロット長
SIM_base.Kp = 2
# 通信路状態
SIM_base.channel_type = "wireless" #
SIM_base.channel_change = 0 #
# 通信路推定方式
SIM_base.channel_estimate = "pilot"
# SIM_base.channel_estimate = "successive"
# SIM_base.channel_estimate = "iterative"
SIM_base.repeat_max = 1 # 繰り返し推定回数（繰り返し通信路推定）
# 実行環境
SIM_base.env_server = '\\nas001\share' in os.getcwd() # 実行環境が研究室サーバーかどうか
SIM_base.nworker = 50 # 並列ワーカー数


### 訓練
SIM_train = copy.deepcopy(SIM_base)
# 通信環境
SIM_train.noise_pilot = True # パイロットに雑音を乗せるか
SIM_train.Kd = 1024 # データ長
SIM_train.EsN0 = 10 # Es/N0（雑音レベル）
SIM_train.upd_th = 2 # 更新閾値（逐次通信路推定）
# 層構造
SIM_train.hidden_depth = 0
# SIM_train.hidden_dim   = SIM_train.Q * 5
# SIM_train.activation_func = torch.nn.ReLU() #
# 訓練条件
SIM_train.InitialLearnRate = 0.1
SIM_train.epochs = 1200
SIM_train.scheduler_factor = 0.995
SIM_train.scheduler_start  = 100
SIM_train.scheduler_step   = 1 #
# 表示
print(SIM_train.__dict__)


### テスト
SIM_test = copy.deepcopy(SIM_base)
# 通信環境
SIM_test.noise_pilot = True
SIM_test.Kd = 1024
SIM_test.EsN0Vec = np.arange(0, 30+1, 3)
SIM_test.upd_th = 1
# ループ数
SIM_test.nloop_max = 1e4
SIM_test.SymError_max = 1e6
if SIM_test.env_server:
    SIM_test.nloop_max /= SIM_test.nworker #
    SIM_test.SymError_min /= SIM_test.nworker #
# 表示
print(SIM_test.__dict__)

# %% [markdown]
# # ネットワークの生成

# %%
model_list = [ae.Net(SIM_train, normalize) if normalize != 0 else None for normalize in normalize_list]
model_list_AE = [model for model in model_list if model != None]

# %% [markdown]
# # ネットワークの学習

# %%
SIM_train.serial = (SIM_train.env_server == False) or (SIM_train.nworker == 1) or (len(model_list_AE) == 1)

### 訓練ループ
if SIM_train.serial:
    RES_train = []
    for i in range(len(model_list)):
        if model_list[i] == None:   # AEのときのみ訓練を行う
            continue
        normalize = normalize_name[normalize_list[i]]
        print(normalize)
        # 訓練
        RES = ae.train_model(SIM_train, model_list[i])
        # 結果保存
        RES.normalize = normalize
        RES_train.append(RES)
        # 結果表示
        print(RES.loss[-1], RES.distance[-1], RES.time,"[sec]")
    print()
else:
    with Pool(SIM_train.nworker) as p:
        RES_train = p.starmap(ae.train_model, [(SIM_train, model) for model in model_list_AE])

### 最小シンボル間距離表示
for RES in RES_train:
    print(RES.distance[-1])
print()

### 最小シンボル間距離グラフ
if SIM_train.env_server == False:
    for RES in RES_train:
        plt.plot(RES.distance, label=RES.normalize)
    plt.legend()
    plt.grid()
    plt.ylim([0, 1.5])
    plt.xlabel("epochs")
    plt.ylabel("min symbol distance")
    plt.show()

# %% [markdown]
# ## テスト

# %%
### テストループ
RES_test = []
for i in range(len(model_list)):
    normalize = normalize_name[normalize_list[i]]
    print(normalize)
    # テスト
    RES = ae.test_model(SIM_test, model_list[i])
    # 結果保存
    RES.normalize = normalize
    RES_test.append(RES)
    # 結果表示
    print(RES.time,"[sec]")
    print()

### SER特性 グラフ表示
if SIM_test.env_server == False:
    for RES in RES_test:
        plt.plot(SIM_test.EsN0Vec, RES.SER, label=RES.normalize)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlim([SIM_test.EsN0Vec[0], SIM_test.EsN0Vec[-1]])
    plt.xlabel("SNR[dB]")
    plt.ylabel("SER")
    plt.show()

# %% [markdown]
# # スプレッドシート転記用出力

# %%
# 正規化規則
for normalize in normalize_list:
    print(normalize, end=" ")
print()

# 最小シンボル間距離
for RES in RES_test:
    print(RES.distance, end=" ")
print()

# SER
for i in range(SIM_test.EsN0Vec.shape[0]):
    for RES in RES_test:
        print(RES.SER[i], end=" ")
    print()


