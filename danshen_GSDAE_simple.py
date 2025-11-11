#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSDAE (Group Selective Deep AutoEncoder) for Danshen Analysis
基于方案文档的改进版本，包含：
1. 组稀疏正则化 (Group Lasso)
2. 半监督学习机制
3. 预测头 (Prediction Head)
4. 复合损失函数
5. 两层重要性分析
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import optimizers, initializers

class ZeroToOneClip(tf.keras.constraints.Constraint):
    """权重约束：限制在0-1之间"""
    def __call__(self, w):
        return tf.clip_by_value(w, 0, 1)

class GroupSelectiveLayer(keras.layers.Layer):
    """组选择层 - 支持组稀疏正则化的特征选择层"""
    def __init__(self, feature_groups, group_lasso_rate=0.01, l1_rate=0.001, **kwargs):
        super().__init__(**kwargs)
        self.feature_groups = feature_groups
        self.group_lasso_rate = group_lasso_rate
        self.l1_rate = l1_rate
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", 
            shape=(int(input_shape[-1]),),
            initializer=initializers.RandomUniform(minval=0.999, maxval=1.0),
            constraint=ZeroToOneClip(),
            trainable=True
        )
        
    def call(self, inputs):
        weighted_features = tf.multiply(inputs, self.kernel)
        
        # 添加组稀疏正则化损失
        group_loss = 0.0
        for group_indices in self.feature_groups.values():
            if len(group_indices) > 0:
                group_weights = tf.gather(self.kernel, group_indices)
                group_l2_norm = tf.norm(group_weights, ord=2)
                group_loss += group_l2_norm
        
        l1_loss = tf.reduce_sum(tf.abs(self.kernel))
        
        self.add_loss(self.group_lasso_rate * group_loss)
        self.add_loss(self.l1_rate * l1_loss)
        
        return weighted_features

def create_feature_groups(feature_names):
    """根据特征名称创建特征分组"""
    groups = {
        '土壤元素': [], '作物元素': [], '土壤养分': [], '地理信息': [], '气候环境': [],
        '省份': [], '城市': [], '地貌': [], '土壤类型': [], '栽培类型': [], '气候类型': []
    }
    
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        
        if name.endswith('_S'):
            groups['土壤元素'].append(i)
        elif name.endswith('_P'):
            groups['作物元素'].append(i)
        elif any(nutrient in name_lower for nutrient in ['ph', 'om', 'tn', 'tp', 'tk', 'an', 'ap', 'ak']):
            groups['土壤养分'].append(i)
        elif any(geo in name_lower for geo in ['lat', 'lon', 'alt', 'elevation']):
            groups['地理信息'].append(i)
        elif any(climate in name_lower for climate in ['temp', 'prec', 'humi', 'wind', 'sun']):
            groups['气候环境'].append(i)
        elif 'province' in name_lower:
            groups['省份'].append(i)
        elif 'city' in name_lower:
            groups['城市'].append(i)
        elif 'landscape' in name_lower:
            groups['地貌'].append(i)
        elif 'soiltype' in name_lower or 'soilclass' in name_lower:
            groups['土壤类型'].append(i)
        elif 'cultivation' in name_lower:
            groups['栽培类型'].append(i)
        elif 'climate' in name_lower and 'type' in name_lower:
            groups['气候类型'].append(i)
    
    # 移除空组
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups

def build_GSDAE(input_shape, target_dim, feature_groups, nbr_hidden_layers=3, 
                hidden_layer_shape=12, encodings_nbr=6, activation="relu", 
                group_lasso_rate=0.01, l1_rate=0.001):
    """构建GSDAE模型"""
    
    # 输入层
    feature_inputs = Input(shape=[input_shape], name='input')
    
    # 组选择层
    group_selective_layer = GroupSelectiveLayer(
        feature_groups=feature_groups,
        group_lasso_rate=group_lasso_rate,
        l1_rate=l1_rate,
        name='group_selective_layer'
    )
    selected_features = group_selective_layer(feature_inputs)
    
    # 编码器 - 原始特征路径
    encoder_full = feature_inputs
    for i in range(nbr_hidden_layers):
        encoder_full = Dense(hidden_layer_shape, activation=activation, 
                           name=f'encoder_hidden_layer_full_{i}')(encoder_full)
    
    # 编码器 - 选择特征路径  
    encoder_select = selected_features
    for i in range(nbr_hidden_layers):
        encoder_select = Dense(hidden_layer_shape, activation=activation, 
                             name=f'encoder_hidden_layer_select_{i}')(encoder_select)
    
    # 编码层
    encoding_full = Dense(encodings_nbr, activation=activation, name='encoding_layer_full')(encoder_full)
    encoding_select = Dense(encodings_nbr, activation=activation, name='encoding_layer_select')(encoder_select)
    
    # 预测头 - 用于半监督学习
    prediction_head = Dense(32, activation='relu', name='pred_hidden')(encoding_select)
    target_prediction = Dense(target_dim, activation='linear', name='target_prediction')(prediction_head)
    
    # 解码器 - 共享权重
    decoder_layers = []
    for i in range(nbr_hidden_layers):
        decoder_layer = Dense(hidden_layer_shape, activation=activation, name=f'decoder_hidden_layer_{i}')
        decoder_layers.append(decoder_layer)
    
    reconstruction_layer = Dense(input_shape, activation='linear', name='reconstruction_layer')
    
    # 应用解码器
    decoder_full = encoding_full
    decoder_select = encoding_select
    
    for decoder_layer in decoder_layers:
        decoder_full = decoder_layer(decoder_full)
        decoder_select = decoder_layer(decoder_select)
    
    # 重构输出
    reconstruction_full = reconstruction_layer(decoder_full)
    reconstruction_select = reconstruction_layer(decoder_select)
    
    # 构建模型
    gsdae_model = Model(inputs=feature_inputs, outputs=[reconstruction_select, target_prediction], name='GSDAE')
    full_autoencoder = Model(inputs=feature_inputs, outputs=reconstruction_full, name='FullAutoEncoder')
    feature_selector = Model(inputs=feature_inputs, outputs=selected_features, name='FeatureSelector')
    full_encoder = Model(inputs=feature_inputs, outputs=encoding_full, name='FullEncoder')
    select_encoder = Model(inputs=feature_inputs, outputs=encoding_select, name='SelectEncoder')
    
    return gsdae_model, full_autoencoder, feature_selector, full_encoder, select_encoder, group_selective_layer

def analyze_feature_importance(group_selective_layer, feature_groups, feature_names):
    """两层重要性分析"""
    weights = group_selective_layer.kernel.numpy()
    
    # 组重要性评估
    group_importance = {}
    for group_name, indices in feature_groups.items():
        if len(indices) > 0:
            group_weights = weights[indices]
            group_l2_norm = np.linalg.norm(group_weights, ord=2)
            group_importance[group_name] = group_l2_norm
    
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 组内关键特征识别
    feature_importance = {}
    for group_name, indices in feature_groups.items():
        if len(indices) > 0:
            group_weights = weights[indices]
            group_features = [feature_names[i] for i in indices]
            feature_weight_pairs = list(zip(group_features, group_weights))
            sorted_features = sorted(feature_weight_pairs, key=lambda x: x[1], reverse=True)
            feature_importance[group_name] = sorted_features
    
    return sorted_groups, feature_importance, weights

def load_and_preprocess_data():
    """数据加载和预处理"""
    # 读取数据
    data = pd.read_csv('../data/丹参数据salvia_all_20240425 - 副本.csv')
    print(f"原始数据形状: {data.shape}")
    
    # 删除前三列
    data = data.iloc[:, 3:]
    
    # 定义目标变量
    target_columns = [
        'CS', 'MT', 'TSIIA', 'TSI', 'DTSI', 'SumTS', 'PD', 'CFA', 'FA', 'SAD', 'SF', 'DSS',
        'SAC', 'SAE', 'MCF', 'RA', 'SAA', 'LA', 'SAY', 'TA', 'CTA', 'MA', 'FMA', 'SUA', 'SAB'
    ]
    
    # 分离目标变量和特征
    available_targets = [col for col in target_columns if col in data.columns]
    target_data = data[available_targets] if available_targets else None
    
    drop_cols = available_targets + ['Soil_sampleN', 'etestN', 'testNp', 'etestpatch']
    feature_data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    # 数据预处理
    thresh = len(feature_data) * 0.5
    feature_data = feature_data.dropna(axis=1, thresh=thresh)
    
    if target_data is not None:
        valid_indices = feature_data.dropna().index
        feature_data = feature_data.loc[valid_indices]
        target_data = target_data.loc[valid_indices]
        
        target_valid_indices = target_data.dropna().index
        feature_data = feature_data.loc[target_valid_indices]
        target_data = target_data.loc[target_valid_indices]
    else:
        feature_data = feature_data.dropna()
    
    # 独热编码
    categorical_columns = ["Province", "City", "Microb", "Landscape", "SoilType", "soilclass", 
                          "CultivationType", "ClimateType", "按气候聚类划分的类型"]
    categorical_columns = [col for col in categorical_columns if col in feature_data.columns]
    feature_data = pd.get_dummies(feature_data, columns=categorical_columns, drop_first=True)
    
    # 准备目标变量
    if target_data is not None and 'SumTS' in target_data.columns:
        main_target = target_data[['SumTS']]
        print("使用SumTS作为主要目标变量")
    elif target_data is not None:
        main_target = target_data.iloc[:, :1]
        print(f"使用{target_data.columns[0]}作为主要目标变量")
    else:
        main_target = pd.DataFrame(np.random.randn(len(feature_data), 1), columns=['dummy_target'])
        print("创建虚拟目标变量")
    
    print(f"最终特征数据形状: {feature_data.shape}")
    print(f"目标变量形状: {main_target.shape}")
    
    return feature_data, main_target

if __name__ == "__main__":
    print("🔄 GSDAE (Group Selective Deep AutoEncoder) 简化版本")
    print("=" * 60)
    print("主要改进:")
    print("1. ✅ 组稀疏正则化 (Group Lasso)")
    print("2. ✅ 半监督学习机制") 
    print("3. ✅ 预测头 (Prediction Head)")
    print("4. ✅ 复合损失函数")
    print("5. ✅ 两层重要性分析")
    print("6. ✅ 特征分组结构")
    print("=" * 60)
    print("请运行 danshen_GSDAE_simple.ipynb 进行完整训练和分析")
