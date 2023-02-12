# %% [markdown]
# ## モジュールをインポート

# %%
import math
import time
import itertools
from multiprocessing import Pool

import numpy as np
import numpy.linalg as nlinalg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as tlinalg

# %% [markdown]
# ## 変数格納用クラス

# %%
class Storage():
    pass

# %% [markdown]
# ## アダマール行列を生成

# %%
def hadamard(n):
    H = 1
    H2 = np.array([[1, 1], [1, -1]])
    for i in range(int(math.log2(n))):
        H = np.kron(H2, H)
    return H

# %% [markdown]
# ## 雑音パラメータ計算

# %%
def noise_setting(SIM, EsN0_dB):
    EsN0 = 10**(EsN0_dB/10)
    SIM.N0 = SIM.M / EsN0
    SIM.sigma_noise = math.sqrt(SIM.N0/2)

# %% [markdown]
# ## 信号点間最小距離を計算

# %%
def min_symbol_distance(x):
    x = np.atleast_3d(x)
    distance = nlinalg.norm(x.transpose(2,1,0) - x.transpose(1,2,0), axis=2)
    min_distance = distance[distance!=0].min()
    return min_distance

# %% [markdown]
# ## 通信路（訓練用）

# %%
class train_channel():
  def __init__(self, SIM):
      # 変数取り込み
      self.SIM = SIM
      # 通信路選択
      if SIM.channel_type == "wireless":
          if SIM.channel_change == 0:
              self.channel = self.wireless_const
  

  ### 無線通信路・時間一定
  def wireless_const(self, X):
      # 変数取り込み
      M = self.SIM.M
      N = self.SIM.N
      Kp = self.SIM.Kp
      Kd = self.SIM.Kd
      sigma_noise = self.SIM.sigma_noise
      
      # 通信路行列
      H = torch.normal(torch.zeros(N,M), std=1/math.sqrt(2)) + 1j * torch.normal(torch.zeros(N,M), std=1/math.sqrt(2))
      # 通信路行列正規化
      H = H * (math.sqrt(M*N) / H.norm('fro'))
      
      # 雑音
      Z = torch.normal(torch.zeros(N, Kp+Kd), std=sigma_noise) + 1j * torch.normal(torch.zeros(N, Kp+Kd), std=sigma_noise)
      if self.SIM.noise_pilot == False:
          Z[:, :Kp] = 0

      # 通信路伝搬
      Y = H @ X + Z
      
      return Y

# %% [markdown]
# ## 復号化器（訓練用）

# %%
class train_decoder():
  def __init__(self, SIM):
      # 変数取り込み
      self.SIM = SIM
      # 通信路推定方式選択
      if   SIM.channel_estimate == "pilot":
          self.decode = self.pilot_channel_estimate
      elif SIM.channel_estimate == "successive":
          self.decode = self.successive_channel_estimate
      elif SIM.channel_estimate == "iterative":
          self.decode = self.iterative_channel_estimate
  

  ### パイロットのみによる通信路推定
  def pilot_channel_estimate(self, x_replica, Xp, Yp, Yd):

      ### 変数取り込み
      Q = self.SIM.Q
      K = self.SIM.Kd
      N0 = self.SIM.N0
      
      ### 通信路推定
      H_hat = Yp @ tlinalg.pinv(Xp)

      ### 信号検出のための変数用意
      # 受信レプリカ
      y_replica = H_hat @ x_replica
      # 対数尤度
      llv = torch.empty(K, Q)

      ### 信号検出
      Yd = torch.atleast_3d(Yd).permute(1,2,0)                # [K,1,N]=[N,K].permute(1,2,0)
      y_replica = torch.atleast_3d(y_replica).permute(2,1,0)  # [1,Q,N]=[N,Q].permute(2,1,0)
      # 対数尤度計算
      llv = - ((Yd - y_replica).abs()**2 ).sum(dim=2) / N0    # [K,Q]=([K,1,N]-[1,Q,N]).sum(dim=2)
      
      return llv
    
  
  ### 逐次通信路推定
  def successive_channel_estimate(self, x_replica, Xp, Yp, Yd):

      ### 変数取り込み
      Q = self.SIM.Q
      wid = self.SIM.Kp
      K = self.SIM.Kd
      N0 = self.SIM.N0
      upd_th = self.SIM.upd_th
      
      ### 通信路推定のための変数用意
      # XとYの時空間小行列
      Xsub = Xp
      Ysub = Yp
      # 通信路推定
      H_hat = Ysub @ tlinalg.pinv(Xsub)

      ### 信号検出のための変数用意
      # 受信レプリカ
      y_replica = H_hat @ x_replica
      # 対数尤度
      llv = torch.empty(K, Q)
      # 受信シンボル全系列
      Y = torch.hstack([Yp, Yd])

      for k in range(K):
          ### 信号検出
          # 対数尤度計算
          llv[k, :] = - ((Yd[:, k].reshape(-1,1) - y_replica).abs()**2 ).sum(dim=0) / N0
          # 硬判定
          a_hat = torch.argmax(llv[k, :])
          
          ### 通信路推定値更新
          # 仮判定送信小行列
          Xsub = torch.hstack([Xsub[:, 1:], x_replica[:, a_hat].reshape(-1,1)])
          # 受信小行列
          Ysub = Y[:, k+1:wid+k+1]
          # 通信路推定値・受信レプリカ更新
          if ( Xsub @ Xsub.conj().T ).det().abs() > upd_th:   #determinantが小さいものを入れ込むと精度が悪くなるため更新から除外
              H_hat = Ysub @ tlinalg.pinv(Xsub)
              y_replica = H_hat @ x_replica
      
      return llv
  
  
  ### 繰り返し通信路推定
  def iterative_channel_estimate(self, x_replica, Xp, Yp, Yd):

      ### 変数取り込み
      Q = self.SIM.Q
      K = self.SIM.Kd
      N0 = self.SIM.N0
      repeat_max = self.SIM.repeat_max

      ### 通信路推定
      H_hat = Yp @ tlinalg.pinv(Xp)

      ### 信号検出のための変数用意
      # 受信レプリカ
      y_replica = H_hat @ x_replica
      # 対数尤度
      llv = torch.empty(K, Q)
      # 受信シンボル全系列
      Y = torch.hstack([Yp, Yd])

      ### 仮判定
      Yd = torch.atleast_3d(Yd).permute(1,2,0)                # [K,1,N]=[N,K].permute(1,2,0)
      y_replica = torch.atleast_3d(y_replica).permute(2,1,0)  # [1,Q,N]=[N,Q].permute(2,1,0)
      # 対数尤度計算
      llv = - ((Yd - y_replica).abs()**2 ).sum(dim=2) / N0    # [K,Q]=([K,1,N]-[1,Q,N]).sum(dim=2)
      
      ### 繰り返し推定
      a_hat = torch.full([K], -1)
      idx_repeat = 0
      while idx_repeat < repeat_max:
          a_hat_= a_hat
          a_hat = torch.argmax(llv, dim=1)
          if (a_hat == a_hat_).all():
              break
          Xd_hat = x_replica[:, a_hat]
          X_hat = torch.hstack([Xp, Xd_hat])
          H_hat = Y @ tlinalg.pinv(X_hat)
          y_replica = H_hat @ x_replica
          y_replica = torch.atleast_3d(y_replica).permute(2,1,0)  # [1,Q,N]=[N,Q].permute(2,1,0)
          llv = - ((Yd - y_replica).abs()**2 ).sum(dim=2) / N0    # [K,Q]=([K,1,N]-[1,Q,N]).sum(dim=2)
          idx_repeat += 1
      
      return llv

# %% [markdown]
# ## 正規化層

# %%
class normalization_layer():
    def __init__(self, normalize_rule):
        # 変数
        self.averaged = False
        # 正規化規則選択
        if   normalize_rule == 1:  # エネルギー制約
            self.normalize = self.normalize_Energy
        elif normalize_rule == 2:  # 平均電力制約
            self.normalize = self.normalize_AveragePower
        elif normalize_rule == 3:  # エネルギーストリーム制約
            self.normalize = self.normalize_EnergyStream
        elif normalize_rule == 4:  # 平均電力ストリーム制約
            self.normalize = self.normalize_AveragePowerStream
        
    # エネルギー制約
    def normalize_Energy(self, x):
        return x / (x.abs()**2).mean(dim=0).sqrt()

    # 平均電力制約
    def normalize_AveragePower(self, x):
        if self.averaged == False:  # 初めにレプリカ系列を入力する
            self.averaged = True
            self.normalize_factor = 1 / (x.abs()**2).mean().sqrt()
        return x * self.normalize_factor

    # エネルギーストリーム制約
    def normalize_EnergyStream(self, x):
        return x / x.abs()

    # 平均電力ストリーム制約
    def normalize_AveragePowerStream(self, x):
        if self.averaged == False:  # 初めにレプリカ系列を入力する
            self.averaged = True
            self.normalize_factor = 1 / (x.abs()**2).mean(dim=1, keepdim=True).sqrt()
        return x * self.normalize_factor

# %% [markdown]
# ## ネットワークの定義

# %%
class Net(nn.Module, train_channel):
    def __init__(self, SIM, normalize):
        super().__init__()

        # 変数取り込み
        self.SIM = SIM
        Q = SIM.Q
        Q_str = SIM.Q_str
        M = SIM.M
        Kp = SIM.Kp
        hidden_depth = SIM.hidden_depth
        if hidden_depth > 0:
            hidden_dim = SIM.hidden_dim
            activation_func = SIM.activation_func
        
        # NN層構造定義
        if   normalize in [1, 2, 3, 4]:
            input_NN = Q
            output_NN = 2 * M
        elif normalize in [5, 6]:
            input_NN = Q_str
            output_NN = 2
        self.layers = nn.ModuleList()
        input_layer = input_NN
        for i in range(hidden_depth):
            output_layer = hidden_dim
            self.layers += [nn.Linear(input_layer, output_layer), activation_func]
            input_layer = hidden_dim
        output_layer = output_NN
        self.layers += [nn.Linear(input_layer, output_layer)]
        self.input_NN = input_NN
        self.output_NN = int(output_NN/2)

        # 符号化層
        if   normalize in [1, 2, 3, 4]:
            self.encode = self.encode_base
        elif normalize in [5, 6]:
            self.encode = self.encode_same_constellation
        
        # 正規化層
        if   normalize in [1, 2, 3, 4]:
            self.nml = normalization_layer(normalize)
        elif normalize == 5:
            self.nml = normalization_layer(1)
        elif normalize == 6:
            self.nml = normalization_layer(2)
        
        # 通信路
        self.ch = train_channel(SIM)

        # 復号化器
        self.dec = train_decoder(SIM)

        # 直交パイロット系列生成
        a_pilot = np.arange(Kp) % M
        self.X_pilot = torch.from_numpy( hadamard(M)[:, a_pilot] ).cfloat()
    
    ### 符号化
    def encode_base(self, a):
        # 変数取り込み
        input_NN = self.input_NN
        output_NN = self.output_NN
        # one-hot化
        # x = torch.nn.functional.one_hot(a, num_classes=self.SIM.Q).float()    # M=4,16QAMのときはこっちの方が速い
        x = torch.eye(input_NN)[a,:]
        # NN伝播
        for layer in self.layers:
            x = layer(x)
        x = x.T     # 実数等価
        x = x[:output_NN,:] + 1j * x[output_NN:,:]  # 複素化
        x = self.nml.normalize(x)   # 正規化
        return x
    
    ### 符号化（各ストリームで同じ信号点配置）
    def encode_same_constellation(self, a):
        # 変数取り込み
        M = self.SIM.M
        Q_str = self.SIM.Q_str
        K = a.shape[0]
        # ストリームごとのラベルに分割
        a_str = torch.empty([M, K], dtype=int)
        for m in range(M):
            a_str[m,:] = a % Q_str
            a = torch.div(a, Q_str, rounding_mode='trunc')  # 商(ゼロ方向への丸め)
        a_serial = a_str.flatten()              # ラベルを1次元化
        x_serial = self.encode_base(a_serial)   # 符号化
        x = x_serial.reshape(M,-1)              # ストリームごとの複素信号に分割
        return x
        
    ### 順伝播
    def forward(self, a_data):
        # 変数取り込み
        Q = self.SIM.Q
        Kp = self.SIM.Kp
        X_pilot = self.X_pilot

        # 符号化器
        self.nml.averaged = False
        x_replica = self.encode(torch.arange(Q))    # レプリカ
        X_data = self.encode(a_data)                # データ

        # 通信路
        Y = self.ch.channel(torch.hstack([X_pilot, X_data]))
        
        # 復号化器
        llv = self.dec.decode(x_replica, X_pilot, Y[:, :Kp], Y[:, Kp:])
        return llv, x_replica

# %% [markdown]
# ## 訓練ループ

# %%
def train_model(SIM, model):
    time_start = time.time()
    
    ### 変数取り込み
    epochs = SIM.epochs
    K = SIM.Kd
    Q = SIM.Q
    
    ###変数用意
    # 訓練用変数
    noise_setting(SIM, SIM.EsN0)  # 雑音パラメータ計算
    loss_func = nn.CrossEntropyLoss()   # 損失関数：交差エントロピー
    optimizer = optim.Adam(model.parameters(), lr=SIM.InitialLearnRate)     # 学習率最適化アルゴリズム
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=list(range(SIM.scheduler_start, epochs+1, SIM.scheduler_step)),
                                               gamma=SIM.scheduler_factor)  # 学習率減衰スケジュール
    # 結果格納用変数
    RES_loss     = np.zeros(epochs)
    RES_distance = np.zeros(epochs)

    ### 訓練ループ
    for i in range(epochs):
        a = torch.randint(0, Q, size=[K])
        optimizer.zero_grad()   #勾配値を0に初期化
        llv, x = model(a)
        loss = loss_func(llv, a)
        loss.backward()
        optimizer.step()    #学習率最適化
        scheduler.step()    #学習率減衰
        RES_loss[i] = loss.item()
        RES_distance[i] = min_symbol_distance( x.detach().numpy() )
        if SIM.serial and (i % (epochs//10) == 0):   #途中経過表示
            print(i, end=' ')
    print()

    ### 出力変数格納
    RES = Storage()
    RES.loss = RES_loss
    RES.distance = RES_distance
    RES.time = time.time() - time_start
    return RES

# %% [markdown]
# ## QAM信号点生成クラス

# %%
class MIMO_QAM_modulation():
  def __init__(self, Q_str, M=1, real_eq=False):
      Q = Q_str**M    # 全多値数
      if Q_str == 2:  # BPSK
          Q_dim = 2
      else:           # QAM
          Q_dim = int(math.sqrt(Q_str))   # 1次元あたりの多値数

      label_mat = np.empty([2*M, Q], dtype=int)
      label_dim = np.arange(Q)
      for i in range(2*M):
          label_mat[i,:] = label_dim % Q_dim
          label_dim //= Q_dim
      
      symtab1 = self.gray2binary(label_mat)                     # データラベル --> シンボル位置
      symtab2 = symtab1 - symtab1.mean(axis=1, keepdims=True)   # 平均 0
      symtab_R = symtab2 / np.sqrt((symtab2**2).sum()/(M*Q))    # 平均電力 1
      symtab_C = symtab_R[:M, :] + 1j * symtab_R[M:, :]         # 複素化
    
      if real_eq == True:
          self.symtab = symtab_R
      else:
          self.symtab = symtab_C
      
  def gray2binary(self, gray):
      mask = gray >> 1
      while mask.any():
          gray ^= mask
          mask >>= 1
      return gray
  
  def __call__(self, label):
      return self.symtab[:, label]

# %% [markdown]
# ## AE信号点生成クラス（テスト用）

# %%
class AE_modulation():
  def __init__(self, model, Q):
      self.model = model
      self.Q = Q
  
  def __call__(self, np_label):
      with torch.no_grad():
          torch_label = torch.from_numpy(np_label).long()
          x = self.model.encode(torch_label)
      return x.detach().numpy()

# %% [markdown]
# ## 通信路（テスト用）

# %%
class test_channel():
  def __init__(self, SIM):
      # 変数取り込み
      self.SIM = SIM
      # 通信路選択
      if SIM.channel_type == "wireless":
          if SIM.channel_change == 0:
              self.channel = self.wireless_const
  
  
  ### 無線通信路・時間一定
  def wireless_const(self, X):
      # 変数取り込み
      M = self.SIM.M
      N = self.SIM.N
      Kp = self.SIM.Kp
      Kd = self.SIM.Kd
      sigma_noise = self.SIM.sigma_noise

      # 通信路行列
      H = np.random.normal(0, 1/math.sqrt(2), size=[N,M]) + 1j * np.random.normal(0, 1/math.sqrt(2), size=[N,M])
      
      # 雑音
      Z = np.random.normal(0, sigma_noise, size=[N, Kp+Kd]) + 1j * np.random.normal(0, sigma_noise, size=[N, Kp+Kd])
      if self.SIM.noise_pilot == False:
          Z[:, :Kp] = 0

      # 通信路伝搬
      Y = H @ X + Z

      return Y

# %% [markdown]
# ## 復号化器（テスト用）

# %%
class test_decoder():
  def __init__(self, SIM):
      # 変数取り込み
      self.SIM = SIM
      # 通信路推定方式選択
      if   SIM.channel_estimate == "pilot":
          self.decode = self.pilot_channel_estimate
      elif SIM.channel_estimate == "successive":
          self.decode = self.successive_channel_estimate
      elif SIM.channel_estimate == "iterative":
          self.decode = self.iterative_channel_estimate
    
    
  ### パイロットのみによる通信路推定
  def pilot_channel_estimate(self, x_replica, Xp, Yp, Yd):

      ### 変数取り込み
      Q = self.SIM.Q
      K = self.SIM.Kd
      N0 = self.SIM.N0

      ### 通信路推定
      H_hat = Yp @ nlinalg.pinv(Xp)

      ### 信号検出のための変数用意
      # 受信レプリカ
      y_replica = H_hat @ x_replica
      # 対数尤度
      llv = np.empty([K, Q])

      ### 信号検出
      Yd = np.atleast_3d(Yd).transpose(1,2,0)                 # [K,1,N]=[N,K].transpose(1,2,0)
      y_replica = np.atleast_3d(y_replica).transpose(2,1,0)   # [1,Q,N]=[N,Q].transpose(2,1,0)
      # 対数尤度計算
      llv = - (np.abs(Yd - y_replica)**2).sum(axis=2) / N0    # [K,Q]=([K,1,N]-[1,Q,N]).sum(axis=2)
      
      return llv
  

  ##############################
  ### 逐次通信路推定
  def successive_channel_estimate(self, x_replica, Xp, Yp, Yd):

      ### 変数取り込み
      Q = self.SIM.Q
      wid = self.SIM.Kp
      K = self.SIM.Kd
      N0 = self.SIM.N0
      upd_th = self.SIM.upd_th

      ### 通信路推定のための変数用意
      # XとYの時空間小行列
      Xsub = Xp
      Ysub = Yp
      # 通信路推定
      H_hat = Ysub @ nlinalg.pinv(Xsub)

      ### 信号検出のための変数用意
      # 受信レプリカ
      y_replica = H_hat @ x_replica
      # 対数尤度
      llv = np.empty([K, Q])
      # 受信シンボル全系列
      Y = np.append(Yp, Yd, axis=1)

      for k in range(K):
          ### 信号検出
          # 対数尤度計算
          llv[k, :] = - (np.abs(Yd[:, k].reshape(-1,1) - y_replica)**2).sum(axis=0) / N0
          # 硬判定
          a_hat = np.argmax(llv[k, :])
          
          ### 通信路推定値更新
          # 仮判定送信小行列
          Xsub = np.append(Xsub[:, 1:], x_replica[:, a_hat].reshape(-1,1), axis=1)
          # 受信小行列
          Ysub = Y[:, k+1:wid+k+1]
          # 通信路推定値・受信レプリカ更新
          if np.abs(nlinalg.det( Xsub @ Xsub.conj().T )) > upd_th:   #determinantが小さいものを入れ込むと精度が悪くなるため更新から除外
              H_hat = Ysub @ nlinalg.pinv(Xsub)
              y_replica = H_hat @ x_replica
      
      return llv
  
  
  ### 繰り返し通信路推定
  def iterative_channel_estimate(self, x_replica, Xp, Yp, Yd):

      ### 変数取り込み
      Q = self.SIM.Q
      K = self.SIM.Kd
      N0 = self.SIM.N0
      repeat_max = self.SIM.repeat_max

      ### 通信路推定
      H_hat = Yp @ nlinalg.pinv(Xp)

      ### 信号検出のための変数用意
      # 受信レプリカ
      y_replica = H_hat @ x_replica
      # 対数尤度
      llv = np.empty([K, Q])
      # 受信シンボル全系列
      Y = np.append(Yp, Yd, axis=1)

      ### 仮判定
      Yd = np.atleast_3d(Yd).transpose(1,2,0)                 # [K,1,N]=[N,K].transpose(1,2,0)
      y_replica = np.atleast_3d(y_replica).transpose(2,1,0)   # [1,Q,N]=[N,Q].transpose(2,1,0)
      # 対数尤度計算
      llv = - (np.abs(Yd - y_replica)**2).sum(axis=2) / N0    # [K,Q]=([K,1,N]-[1,Q,N]).sum(axis=2)

      ### 繰り返し推定
      a_hat = np.full([K], -1)
      idx_repeat = 0
      while idx_repeat < repeat_max:
          a_hat_= a_hat
          a_hat = np.argmax(llv, axis=1)
          if (a_hat == a_hat_).all():
              break
          Xd_hat = x_replica[:, a_hat]
          X_hat = np.append(Xp, Xd_hat, axis=1)
          H_hat = Y @ nlinalg.pinv(X_hat)
          y_replica = H_hat @ x_replica
          y_replica = np.atleast_3d(y_replica).transpose(2,1,0)   # [1,Q,N]=[N,Q].transpose(2,1,0)
          llv = - (np.abs(Yd - y_replica)**2).sum(axis=2) / N0    # [K,Q]=([K,1,N]-[1,Q,N]).sum(axis=2)
          idx_repeat += 1
      
      return llv

# %% [markdown]
# ## テストループ

# %%
def test_loops(SIM, x_replica, X_pilot, mod, ch, dec, bittab):

    # 変数取り込み
    q = SIM.q
    Q = SIM.Q
    Kp = SIM.Kp
    Kd = SIM.Kd

    # 結果格納用変数
    idx_loop = 0
    noe = np.zeros(3)
    nod = np.zeros(3)

    # テストループ
    while (idx_loop < SIM.nloop_max) and (noe[1] < SIM.SymError_max):
        
        # 送信データ
        a = np.random.randint(0, Q, size=Kd)
        X_data = mod(a)

        # 通信路
        Y = ch.channel(np.append(X_pilot, X_data, axis=1))

        # 復号化器
        llv = dec.decode(x_replica, X_pilot, Y[:, :Kp], Y[:, Kp:])
        
        # 硬判定
        ahat = np.argmax(llv, axis=1)

        # エラー数カウント
        noe_ins = np.sum( bittab[a,:] != bittab[ahat,:] , axis=1)                 # シンボルごとのビットエラー数 size=(K,1)
        noe = noe + np.array([np.sum(noe_ins), np.sum(noe_ins!=0), np.sum(noe_ins)!=0])   # 各種エラー数カウント [ビット, シンボル, ブロック]  
        nod = nod + np.array([Kd*q, Kd, 1])                                                 # 全送信回数カウント [ビット, シンボル, ブロック]
        
        idx_loop += 1
    
    return idx_loop, noe, nod


def test_model(SIM, model=None):
    time_start = time.time()

    # 変数取り込み
    M = SIM.M
    q = SIM.q
    Q = SIM.Q
    Q_str = SIM.Q_str
    Kp = SIM.Kp
    EsN0Vec = SIM.EsN0Vec

    ### テスト用インスタンス生成
    # シンボル生成メソッド
    if model == None:
        mod = MIMO_QAM_modulation(Q_str, M)
    else:
        model.nml.averaged = False
        mod = AE_modulation(model, Q)
    # 通信路インスタンス
    ch  = test_channel(SIM)
    # 復号化器インスタンス
    dec = test_decoder(SIM)
    
    ### テスト用変数生成
    # 送信レプリカ
    x_replica = mod(np.arange(Q))
    # パイロット生成
    a_pilot = np.arange(Kp) % M
    X_pilot = hadamard(M)[:, a_pilot].astype(np.complex64)
    # ビット表現
    bittab = np.array( list( itertools.product([0, 1], repeat=q) ) )  # 0 ~ Q-1 を表す2進数の配列 size=(Q,q)

    ### 結果格納用変数
    RES_ER = np.zeros([EsN0Vec.shape[0],3])
    time_cusum = 0

    ### テスト
    for idx in range(EsN0Vec.shape[0]):
        time_loops = -time.time()

        # Es/N0設定
        EsN0 = EsN0Vec[idx]
        noise_setting(SIM, EsN0)

        # 前のEs/N0でSER=0ならば強制終了
        if idx > 0:
          if RES_ER[idx-1, 1] == 0:
            print(EsN0,"[dB], break")
            continue
        
        # エラー数を計算
        if (SIM.env_server == False) or (SIM.nworker == 1):
            nlooped, noe, nod = test_loops(SIM, x_replica, X_pilot, mod, ch, dec, bittab)
        else:
            with Pool(SIM.nworker) as p:
                res_list = p.starmap(test_loops, [(SIM, x_replica, X_pilot, mod, ch, dec, bittab)] * SIM.nworker)
            nlooped = sum([res[0] for res in res_list])
            noe     = sum([res[1] for res in res_list])
            nod     = sum([res[2] for res in res_list])

        # EsN0ごとにエラーレートを格納
        RES_ER[idx,:] = noe / nod

        # 時間計測
        time_loops += time.time()
        time_cusum += time_loops

        # テスト経過表示
        print(EsN0,"[dB], nlooped =",nlooped,", noe =",noe,", ER =",RES_ER[idx,:],",",time_loops,"[sec]",time_cusum,"[sec]")

    ### 出力変数格納
    RES = Storage()
    RES.ER = RES_ER
    RES.SER = RES_ER[:,1]
    RES.distance = min_symbol_distance(x_replica)
    RES.time = time.time() - time_start
    return RES


