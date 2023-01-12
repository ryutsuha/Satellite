from pyparsing import Or
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import pprint as pp
from itertools import combinations
from geopy.distance import geodesic


#最適化アルゴリズム
class adam:
    # インスタンス変数を定義
    # def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
    def __init__(self):
        self.lr    = [0.001, 1000]  # 学習率
        self.beta1 = 0.9    # mの減衰率
        self.beta2 = 0.999  # vの減衰率
        self.iter  = 0      # 試行回数を初期化
        self.m = None       # モーメンタム
        self.v = None       # 適合的な学習係数
    

    # パラメータの更新メソッドを定義
    def update(self, params, grads, total, beam_user, char):
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}

            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0
        
        # パラメータごとに値を更新
        self.iter += 1 # 更新回数をカウント

        if char == "power" :
            lr_t  = self.lr[0] * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter) 
        else :
            lr_t  = self.lr[1] * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter) 


        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            ss = params[key] - lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            # paramsが0以下にならないようにする

            if char == "power" :
                if ss < 0.1:
                    params[key] = 0.1
                else:
                    params[key] = ss
            
            else :
                if ss < 100000:
                    params[key] = 100000
                else:
                    params[key] = ss

        # paramsの合計がtotalになるように調整
        if sum(params.values()) < total:
            am = total - sum(params.values())

            for key in params.keys():                
                if char == "power" :
                    params[key] +=  am * (beam_user[key] / sum(beam_user))
                else :
                    params[key] +=  am * (char[key] / sum(beam_user))
            

        else :
            sa = sum(params.values()) - total

            for key in params.keys():
                if char == "power" :
                    params[key] -= sa * (beam_user[key] / sum(beam_user))
                else :
                    params[key] -= sa * (char[key] / sum(beam_user))