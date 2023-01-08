from pyparsing import Or
from tomlkit import integer
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
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr        # 学習率
        self.beta1 = beta1  # mの減衰率
        self.beta2 = beta2  # vの減衰率
        self.iter = 0       # 試行回数を初期化
        self.m = None       # モーメンタム
        self.v = None       # 適合的な学習係数
    
    # パラメータの更新メソッドを定義
    def updateW(self, params, grads, W):

        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}

            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0
        
        # パラメータごとに値を更新
        self.iter += 1 # 更新回数をカウント
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter) 

        for key in params.keys():
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            ss = params[key] - lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # if ss < 0:
            #     params[key] *=  0.1

            # else:
            #     params[key] = ss

            if sum(params.values()) < W:
                am = W - sum(params.values())

                for key in params.keys():                
                    params[key] +=  am / len(params.values())

            elif sum(params.values()) > W:
                sa = sum(params.values()) - W

                for key in params.keys():
                    params[key] -= (params[key] / sum(params.values())) * sa # ここでparamsの値を更新

    def sample() :
        print("sample")