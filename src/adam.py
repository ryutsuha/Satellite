from pyparsing import Or
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import pprint as pp
from itertools import combinations
from geopy.distance import geodesic
from matplotlib import path


#最適化アルゴリズム
class adam:
    # インスタンス変数を定義
    # def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
    def __init__(self):
        # self.lr    = [0.0001, 10000000, 0.01, 0.01]  # 学習率
        # self.lr    = [0.001, 100000, 0.01, 0.01]  # 学習率
        self.lr    = [0.1, 1, 1, 1]  # 学習率
        self.beta1 = 0.9    # mの減衰率
        self.beta2 = 0.999  # vの減衰率
        self.iter  = [0] * 4      # 試行回数を初期化
        self.m = None       # モーメンタム
        self.v = None       # 適合的な学習係数
        self.points = path.Path([[45, 142], [45, 144], [43, 144], [35, 140], [32, 131], [32, 129], [34, 129], [37, 138]])
    

    # パラメータの更新メソッドを定義
    def update_power(self, params, grads, total, beam_user, char):
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}

            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0
        
        # パラメータごとに値を更新
        self.iter[0] += 1 # 更新回数をカウント
        lr_t  = self.lr[0] * np.sqrt(1.0 - self.beta2 ** self.iter[0]) / (1.0 - self.beta1 ** self.iter[0]) 

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] += lr_t * np.sqrt(1 - self.beta2) / (1 - self.beta1) * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # paramsが0.0001以下にならないようにする
            if params[key] < 1e-7:
                params[key] = 1e-7
        
        total_params = sum(params.values())

        for key in params.keys() :
            params[key] = params[key] / (total_params / total)
        

    # パラメータの更新メソッドを定義
    def update_bandwidth(self, params, grads, total, beam_user, char):
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}

            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0
        
        # パラメータごとに値を更新
        self.iter[1] += 1 # 更新回数をカウント
        lr_t  = self.lr[1] * np.sqrt(1.0 - self.beta2 ** self.iter[1]) / (1.0 - self.beta1 ** self.iter[1]) 

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] -= lr_t * np.sqrt(1 - self.beta2) / (1 - self.beta1) * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            # paramsが100000以下にならないようにする
            if params[key] < 100000:
                params[key] = 100000

        total_params = sum(params.values())

        for key in params.keys() :
            params[key] = params[key] / (total_params / total)
    

    # パラメータの更新メソッドを定義
    def update_beam(self, params, grads):
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}

            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0

        # パラメータごとに値を更新
        self.iter[2] += 1 # 更新回数をカウント
        lr_t  = self.lr[2] * np.sqrt(1.0 - self.beta2 ** self.iter[2]) / (1.0 - self.beta1 ** self.iter[2]) 

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] -= lr_t * np.sqrt(1 - self.beta2) / (1 - self.beta1) * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


    # パラメータの更新メソッドを定義
    def update_sat_radius(self, params, grads):
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}

            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0

        # パラメータごとに値を更新
        self.iter[3] += 1 # 更新回数をカウント
        lr_t  = self.lr[3] * np.sqrt(1.0 - self.beta2 ** self.iter[3]) / (1.0 - self.beta1 ** self.iter[3]) 

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] -= lr_t * np.sqrt(1 - self.beta2) / (1 - self.beta1) * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)