#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSDAE训练脚本
根据方案文档实现的改进版本训练示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 导入GSDAE模块
from danshen_GSDAE import (
    build_GSDAE, prepare_danshen_data, create_feature_groups,
    analyze_feature_importance, plot_importance_analysis
)

def train_gsdae_model(X_train, X_test, y_train, y_test, feature_groups, 
                      epochs=200, batch_size=32, learning_rate=0.001):
    """
    训练GSDAE模型
    """
    print("🚀 开始训练GSDAE模型...")
    
    # 构建模型
    gsdae_model, feature_selector, encoder_model, predictor_model, group_selective_layer = build_GSDAE(
        input_shape=X_train.shape[1],
        target_dim=y_train.shape[1],
        feature_groups=feature_groups,
        nbr_hidden_layers=3,
        hidden_layer_shape=12,
        encodings_nbr=6,
        group_lasso_rate=0.01,
        l1_rate=0.001,
        dropout_rate=0.2
    )
    
    # 编译模型
    gsdae_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'reconstruction_select': 'mse',
            'target_prediction': 'mse'
        },
        loss_weights={
            'reconstruction_select': 1.0,
            'target_prediction': 0.5
        },
        metrics=['mae']
    )
    
    # 准备训练数据
    train_inputs = [X_train, y_train]
    train_outputs = {
        'reconstruction_select': X_train,
        'target_prediction': y_train
    }
    
    test_inputs = [X_test, y_test]
    test_outputs = {
        'reconstruction_select': X_test,
        'target_prediction': y_test
    }
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 训练模型
    history = gsdae_model.fit(
        train_inputs,
        train_outputs,
        validation_data=(test_inputs, test_outputs),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return gsdae_model, feature_selector, encoder_model, predictor_model, group_selective_layer, history

def evaluate_model_performance(gsdae_model, predictor_model, X_test, y_test, scaler_y):
    """
    评估模型性能
    """
    print("\n📊 评估模型性能...")
    
    # 预测
    predictions = predictor_model.predict(X_test)
    
    # 反标准化
    y_test_original = scaler_y.inverse_transform(y_test)
    predictions_original = scaler_y.inverse_transform(predictions)
    
    # 计算指标
    mse = np.mean((y_test_original - predictions_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_original - predictions_original))
    
    # 计算R²
    ss_res = np.sum((y_test_original - predictions_original) ** 2)
    ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"📈 模型性能指标:")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    
    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'predictions': predictions_original,
        'actual': y_test_original
    }

def plot_training_history(history):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 总损失
    axes[0, 0].plot(history.history['loss'], label='训练损失')
    axes[0, 0].plot(history.history['val_loss'], label='验证损失')
    axes[0, 0].set_title('总损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 重构损失
    if 'reconstruction_select_loss' in history.history:
        axes[0, 1].plot(history.history['reconstruction_select_loss'], label='训练重构损失')
        axes[0, 1].plot(history.history['val_reconstruction_select_loss'], label='验证重构损失')
        axes[0, 1].set_title('重构损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 预测损失
    if 'target_prediction_loss' in history.history:
        axes[1, 0].plot(history.history['target_prediction_loss'], label='训练预测损失')
        axes[1, 0].plot(history.history['val_target_prediction_loss'], label='验证预测损失')
        axes[1, 0].set_title('预测损失')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('学习率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """
    主训练流程
    """
    print("🔄 GSDAE (Group Selective Deep AutoEncoder) 完整训练流程")
    print("=" * 70)
    
    # 数据路径
    data_path = '../data/processed_danshen_data.csv'
    
    try:
        # 1. 准备数据
        print("📊 步骤1: 准备数据...")
        feature_data, target_data = prepare_danshen_data(data_path)
        
        print(f"✅ 特征维度: {feature_data.shape}")
        if target_data is not None:
            print(f"✅ 目标变量维度: {target_data.shape}")
            # 使用总丹参酮含量作为主要目标
            if 'SumTS' in target_data.columns:
                main_target = target_data[['SumTS']]
                print("✅ 使用SumTS作为目标变量")
            else:
                main_target = target_data.iloc[:, :1]
                print(f"✅ 使用{target_data.columns[0]}作为目标变量")
        else:
            print("⚠️ 未找到目标变量，创建虚拟目标")
            main_target = pd.DataFrame(np.random.randn(len(feature_data), 1), columns=['dummy_target'])
        
        # 2. 创建特征分组
        print("\n📋 步骤2: 创建特征分组...")
        feature_groups = create_feature_groups(feature_data.columns.tolist())
        print(f"✅ 创建了 {len(feature_groups)} 个特征组:")
        for group_name, indices in feature_groups.items():
            print(f"   - {group_name}: {len(indices)} 个特征")
        
        # 3. 数据预处理
        print("\n🔧 步骤3: 数据预处理...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(feature_data)
        y_scaled = scaler_y.fit_transform(main_target)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        print(f"✅ 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 4. 训练模型
        print("\n🚀 步骤4: 训练GSDAE模型...")
        gsdae_model, feature_selector, encoder_model, predictor_model, group_selective_layer, history = train_gsdae_model(
            X_train, X_test, y_train, y_test, feature_groups,
            epochs=100,  # 减少epoch数以避免长时间运行
            batch_size=32,
            learning_rate=0.001
        )
        
        print("✅ 模型训练完成！")
        
        # 5. 评估性能
        print("\n📊 步骤5: 评估模型性能...")
        performance = evaluate_model_performance(
            gsdae_model, predictor_model, X_test, y_test, scaler_y
        )
        
        # 6. 重要性分析
        print("\n🔍 步骤6: 特征重要性分析...")
        sorted_groups, feature_importance, weights = analyze_feature_importance(
            group_selective_layer, feature_groups, feature_data.columns.tolist()
        )
        
        print("📈 组重要性排名:")
        for i, (group_name, importance) in enumerate(sorted_groups[:5]):
            print(f"   {i+1}. {group_name}: {importance:.4f}")
        
        # 7. 可视化结果
        print("\n📊 步骤7: 生成可视化结果...")
        
        # 绘制训练历史
        plot_training_history(history)
        
        # 绘制重要性分析
        plot_importance_analysis(sorted_groups, feature_importance)
        
        print("\n🎉 GSDAE训练和分析完成！")
        
        return {
            'model': gsdae_model,
            'performance': performance,
            'importance': {
                'groups': sorted_groups,
                'features': feature_importance,
                'weights': weights
            },
            'history': history
        }
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {str(e)}")
        print("💡 请检查数据路径和格式是否正确")
        return None

if __name__ == "__main__":
    # 注释掉实际运行以避免长时间执行
    # results = main()
    print("GSDAE训练脚本已准备完成！")
    print("取消注释 main() 函数调用即可开始训练")
