#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSDAE (Group Selective Deep AutoEncoder) 实现
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
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, optimizers, initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

class ZeroToOneClip(tf.keras.constraints.Constraint):
    """权重约束：限制在0-1之间"""
    def __call__(self, w):
        return tf.clip_by_value(w, 0, 1)

class GroupSelectiveLayer(keras.layers.Layer):
    """
    组选择层 - 支持组稀疏正则化的特征选择层
    """
    def __init__(self, feature_groups, group_lasso_rate=0.01, l1_rate=0.001, **kwargs):
        super().__init__(**kwargs)
        self.feature_groups = feature_groups  # 特征分组信息
        self.group_lasso_rate = group_lasso_rate
        self.l1_rate = l1_rate
        
    def build(self, input_shape):
        # 为每个特征创建权重
        self.kernel = self.add_weight(
            "kernel", 
            shape=(int(input_shape[-1]),),
            initializer=initializers.RandomUniform(minval=0.999, maxval=1.0),
            constraint=ZeroToOneClip(),
            trainable=True
        )
        
    def call(self, inputs):
        # 应用特征权重
        weighted_features = tf.multiply(inputs, self.kernel)
        
        # 添加组稀疏正则化损失
        group_loss = 0.0
        for group_indices in self.feature_groups.values():
            if len(group_indices) > 0:
                group_weights = tf.gather(self.kernel, group_indices)
                group_l2_norm = tf.norm(group_weights, ord=2)
                group_loss += group_l2_norm
        
        # 添加L1正则化
        l1_loss = tf.reduce_sum(tf.abs(self.kernel))
        
        # 将正则化损失添加到模型
        self.add_loss(self.group_lasso_rate * group_loss)
        self.add_loss(self.l1_rate * l1_loss)
        
        return weighted_features
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'group_lasso_rate': self.group_lasso_rate,
            'l1_rate': self.l1_rate
        })
        return config

def create_feature_groups(feature_names):
    """
    根据特征名称创建特征分组
    按照方案中的分组：土壤元素、作物元素、土壤养分、地理信息、气候环境等
    """
    groups = {
        '土壤元素': [],
        '作物元素': [],  
        '土壤养分': [],
        '地理信息': [],
        '气候环境': [],
        '省份': [],
        '城市': [],
        '地貌': [],
        '土壤类型': [],
        '栽培类型': [],
        '气候类型': []
    }
    
    # 根据特征名称模式匹配分组
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        
        # 土壤元素 (以_S结尾的元素)
        if name.endswith('_S'):
            groups['土壤元素'].append(i)
        # 作物元素 (以_P结尾的元素)  
        elif name.endswith('_P'):
            groups['作物元素'].append(i)
        # 土壤养分
        elif any(nutrient in name_lower for nutrient in ['ph', 'om', 'tn', 'tp', 'tk', 'an', 'ap', 'ak']):
            groups['土壤养分'].append(i)
        # 地理信息
        elif any(geo in name_lower for geo in ['lat', 'lon', 'alt', 'elevation']):
            groups['地理信息'].append(i)
        # 气候环境
        elif any(climate in name_lower for climate in ['temp', 'prec', 'humi', 'wind', 'sun']):
            groups['气候环境'].append(i)
        # 省份
        elif 'province' in name_lower:
            groups['省份'].append(i)
        # 城市
        elif 'city' in name_lower:
            groups['城市'].append(i)
        # 地貌
        elif 'landscape' in name_lower:
            groups['地貌'].append(i)
        # 土壤类型
        elif 'soiltype' in name_lower or 'soilclass' in name_lower:
            groups['土壤类型'].append(i)
        # 栽培类型
        elif 'cultivation' in name_lower:
            groups['栽培类型'].append(i)
        # 气候类型
        elif 'climate' in name_lower and 'type' in name_lower:
            groups['气候类型'].append(i)
    
    # 移除空组
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    
    return groups

def build_GSDAE(input_shape, target_dim, feature_groups,
                nbr_hidden_layers=3, hidden_layer_shape=12, 
                encodings_nbr=6, activation="relu",
                group_lasso_rate=0.01, l1_rate=0.001,
                dropout_rate=0.2):
    """
    构建GSDAE模型
    
    参数:
    - input_shape: 输入特征维度
    - target_dim: 目标变量维度（丹参酮含量）
    - feature_groups: 特征分组字典
    - 其他参数: 网络结构参数
    """
    
    # 输入层
    feature_inputs = Input(shape=[input_shape], name='feature_input')
    target_inputs = Input(shape=[target_dim], name='target_input')
    
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
        encoder_full = Dense(
            hidden_layer_shape, 
            activation=activation,
            name=f'encoder_full_{i}'
        )(encoder_full)
        encoder_full = Dropout(dropout_rate)(encoder_full)
    
    # 编码器 - 选择特征路径  
    encoder_select = selected_features
    for i in range(nbr_hidden_layers):
        encoder_select = Dense(
            hidden_layer_shape,
            activation=activation, 
            name=f'encoder_select_{i}'
        )(encoder_select)
        encoder_select = Dropout(dropout_rate)(encoder_select)
    
    # 编码层
    encoding_full = Dense(encodings_nbr, activation=activation, name='encoding_full')(encoder_full)
    encoding_select = Dense(encodings_nbr, activation=activation, name='encoding_select')(encoder_select)
    
    # 预测头 - 用于半监督学习
    prediction_head = Dense(32, activation='relu', name='pred_hidden')(encoding_select)
    prediction_head = Dropout(dropout_rate)(prediction_head)
    target_prediction = Dense(target_dim, activation='linear', name='target_prediction')(prediction_head)
    
    # 解码器
    decoder_full = encoding_full
    decoder_select = encoding_select
    
    for i in range(nbr_hidden_layers):
        decoder_layer = Dense(hidden_layer_shape, activation=activation, name=f'decoder_{i}')
        decoder_full = decoder_layer(decoder_full)
        decoder_select = decoder_layer(decoder_select)
        decoder_full = Dropout(dropout_rate)(decoder_full)
        decoder_select = Dropout(dropout_rate)(decoder_select)
    
    # 重构层
    reconstruction_full = Dense(input_shape, activation='linear', name='reconstruction_full')(decoder_full)
    reconstruction_select = Dense(input_shape, activation='linear', name='reconstruction_select')(decoder_select)
    
    # 构建不同的模型
    # 完整的GSDAE模型（用于训练）
    gsdae_model = Model(
        inputs=[feature_inputs, target_inputs],
        outputs=[reconstruction_select, target_prediction],
        name='GSDAE'
    )
    
    # 特征选择模型
    feature_selector = Model(
        inputs=feature_inputs,
        outputs=selected_features,
        name='FeatureSelector'
    )
    
    # 编码器模型
    encoder_model = Model(
        inputs=feature_inputs,
        outputs=encoding_select,
        name='Encoder'
    )
    
    # 预测模型
    predictor_model = Model(
        inputs=feature_inputs,
        outputs=target_prediction,
        name='Predictor'
    )
    
    return gsdae_model, feature_selector, encoder_model, predictor_model, group_selective_layer

def custom_loss_function(reconstruction_weight=1.0, prediction_weight=0.5):
    """
    自定义复合损失函数
    包含重建误差和预测误差
    """
    def loss(y_true, y_pred):
        # y_true和y_pred都是列表，包含[reconstruction_target, prediction_target]
        reconstruction_target, prediction_target = y_true
        reconstruction_pred, prediction_pred = y_pred
        
        # 重建损失
        reconstruction_loss = tf.reduce_mean(tf.square(reconstruction_target - reconstruction_pred))
        
        # 预测损失  
        prediction_loss = tf.reduce_mean(tf.square(prediction_target - prediction_pred))
        
        # 复合损失
        total_loss = reconstruction_weight * reconstruction_loss + prediction_weight * prediction_loss
        
        return total_loss
    
    return loss

def analyze_feature_importance(group_selective_layer, feature_groups, feature_names):
    """
    两层重要性分析
    1. 组（类别）重要性评估
    2. 组内关键特征识别
    """
    # 获取选择层权重
    weights = group_selective_layer.kernel.numpy()

    # 第一层：组重要性评估
    group_importance = {}
    for group_name, indices in feature_groups.items():
        if len(indices) > 0:
            group_weights = weights[indices]
            # 计算L2范数作为组重要性
            group_l2_norm = np.linalg.norm(group_weights, ord=2)
            group_importance[group_name] = group_l2_norm

    # 按重要性排序
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

    # 第二层：组内关键特征识别
    feature_importance = {}
    for group_name, indices in feature_groups.items():
        if len(indices) > 0:
            group_weights = weights[indices]
            group_features = [feature_names[i] for i in indices]

            # 按权重排序组内特征
            feature_weight_pairs = list(zip(group_features, group_weights))
            sorted_features = sorted(feature_weight_pairs, key=lambda x: x[1], reverse=True)

            feature_importance[group_name] = sorted_features

    return sorted_groups, feature_importance, weights

def plot_importance_analysis(sorted_groups, feature_importance, top_n_groups=5, top_n_features=3):
    """
    可视化重要性分析结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制组重要性
    groups = [item[0] for item in sorted_groups[:top_n_groups]]
    importance = [item[1] for item in sorted_groups[:top_n_groups]]

    ax1.barh(groups, importance)
    ax1.set_xlabel('组重要性 (L2范数)')
    ax1.set_title('特征组重要性排名')
    ax1.grid(True, alpha=0.3)

    # 绘制关键特征（来自最重要的组）
    if sorted_groups:
        top_group = sorted_groups[0][0]
        top_features = feature_importance[top_group][:top_n_features]

        feature_names = [item[0] for item in top_features]
        feature_weights = [item[1] for item in top_features]

        ax2.barh(feature_names, feature_weights)
        ax2.set_xlabel('特征权重')
        ax2.set_title(f'"{top_group}"组内关键特征')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig

def prepare_danshen_data(data_path):
    """
    准备丹参数据，包括特征和目标变量的分离
    """
    # 读取数据
    data = pd.read_csv(data_path)

    # 删除前三列
    data = data.iloc[:, 3:]

    # 定义目标变量（丹参酮相关成分）
    target_columns = [
        'CS', 'MT', 'TSIIA', 'TSI', 'DTSI', 'SumTS', 'PD', 'CFA', 'FA', 'SAD', 'SF', 'DSS',
        'SAC', 'SAE', 'MCF', 'RA', 'SAA', 'LA', 'SAY', 'TA', 'CTA', 'MA', 'FMA', 'SUA', 'SAB'
    ]

    # 分离目标变量和特征
    available_targets = [col for col in target_columns if col in data.columns]
    target_data = data[available_targets] if available_targets else None

    # 删除目标变量和其他不需要的列
    drop_cols = available_targets + ['Soil_sampleN', 'etestN', 'testNp', 'etestpatch']
    feature_data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # 删除空白值较多的列
    thresh = len(feature_data) * 0.5
    feature_data = feature_data.dropna(axis=1, thresh=thresh)

    # 如果有目标变量，也要对应删除缺失样本
    if target_data is not None:
        # 找到特征数据的有效索引
        valid_indices = feature_data.dropna().index
        feature_data = feature_data.loc[valid_indices]
        target_data = target_data.loc[valid_indices]

        # 删除目标变量中的缺失值
        target_valid_indices = target_data.dropna().index
        feature_data = feature_data.loc[target_valid_indices]
        target_data = target_data.loc[target_valid_indices]
    else:
        feature_data = feature_data.dropna()

    # 独热编码分类特征
    categorical_columns = [
        "Province", "City", "Microb", "Landscape", "SoilType", "soilclass",
        "CultivationType", "ClimateType", "按气候聚类划分的类型"
    ]
    categorical_columns = [col for col in categorical_columns if col in feature_data.columns]
    feature_data = pd.get_dummies(feature_data, columns=categorical_columns, drop_first=True)

    return feature_data, target_data

def main_training_example():
    """
    主训练示例 - 展示如何使用GSDAE
    """
    print("🔄 GSDAE (Group Selective Deep AutoEncoder) 训练示例")
    print("=" * 60)

    # 数据路径
    data_path = 'D:/课题会/丹参/danshen_code/SDAE/SDAE-main/data/丹参数据salvia_all_20240425 - 副本.csv'

    # 准备数据
    print("📊 准备数据...")
    feature_data, target_data = prepare_danshen_data(data_path)

    print(f"特征维度: {feature_data.shape}")
    if target_data is not None:
        print(f"目标变量维度: {target_data.shape}")
        # 使用总丹参酮含量作为主要目标（如果有SumTS列）
        if 'SumTS' in target_data.columns:
            main_target = target_data[['SumTS']]
        else:
            main_target = target_data.iloc[:, :1]  # 使用第一个目标变量
    else:
        print("⚠️ 未找到目标变量，将使用无监督模式")
        main_target = np.zeros((len(feature_data), 1))

    # 创建特征分组
    feature_groups = create_feature_groups(feature_data.columns.tolist())
    print(f"📋 特征分组: {list(feature_groups.keys())}")

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(feature_data)
    y_scaled = scaler_y.fit_transform(main_target)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 构建模型
    print("🏗️ 构建GSDAE模型...")
    gsdae_model, feature_selector, encoder_model, predictor_model, group_selective_layer = build_GSDAE(
        input_shape=X_train.shape[1],
        target_dim=y_train.shape[1],
        feature_groups=feature_groups,
        nbr_hidden_layers=3,
        hidden_layer_shape=12,
        encodings_nbr=6,
        group_lasso_rate=0.01,
        l1_rate=0.001
    )

    print("✅ 模型构建完成！")
    print(f"📈 可进行训练和重要性分析")

    return {
        'model': gsdae_model,
        'feature_selector': feature_selector,
        'encoder_model': encoder_model,
        'predictor_model': predictor_model,
        'group_selective_layer': group_selective_layer,
        'feature_groups': feature_groups,
        'feature_names': feature_data.columns.tolist(),
        'data': {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        },
        'scalers': {'X': scaler_X, 'y': scaler_y}
    }

if __name__ == "__main__":
    # 运行示例（注释掉以避免实际执行）
    # results = main_training_example()
    print("GSDAE模型代码已准备完成！")
    print("主要改进:")
    print("1. ✅ 组稀疏正则化 (Group Lasso)")
    print("2. ✅ 半监督学习机制")
    print("3. ✅ 预测头 (Prediction Head)")
    print("4. ✅ 复合损失函数")
    print("5. ✅ 两层重要性分析")
    print("6. ✅ 特征分组结构")
